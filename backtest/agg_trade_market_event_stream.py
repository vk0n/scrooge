from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import heapq
from io import BytesIO, TextIOWrapper
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import zipfile

import pandas as pd
from tqdm import tqdm

from backtest.progress import BacktestProgressReporter
from backtest.time_windows import resolve_backtest_time_range
from bot.event_log import get_technical_logger
from core.feature_engine import FeatureFrameBuilder, build_feature_frame, merge_feature_frames
from core.market_events import CandleClosedEvent, IndicatorSnapshotEvent, MarketEvent, PriceTickEvent, market_event_to_dict


DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL = os.getenv(
    "SCROOGE_BINANCE_PUBLIC_DATA_BASE_URL",
    "https://data.binance.vision/data/futures/um",
).rstrip("/")
DEFAULT_AGG_TRADE_REST_BASE_URL = os.getenv(
    "SCROOGE_FUTURES_REST_BASE_URL",
    "https://fapi.binance.com",
).rstrip("/")
DEFAULT_AGG_TRADE_CACHE_DIR = os.getenv(
    "SCROOGE_AGG_TRADE_CACHE_DIR",
    "data/agg_trades",
)
MARKET_EVENT_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
technical_logger = get_technical_logger()


@dataclass(slots=True)
class AggTradeMarketEventStreamSummary:
    source: str
    symbol: str
    cache_hit: bool
    cache_path: str | None
    raw_agg_trades: int
    price_ticks: int
    candle_events_small: int
    candle_events_medium: int
    candle_events_big: int
    indicator_snapshots: int
    total_events: int
    first_trade_ts: str | None
    last_trade_ts: str | None


def _format_ts(value: datetime | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        resolved = value.tz_convert(None) if value.tzinfo is not None else value
        return resolved.strftime(MARKET_EVENT_TS_FORMAT)
    return value.strftime(MARKET_EVENT_TS_FORMAT)


def _tqdm_position(extra_offset: int = 0) -> int | None:
    raw = os.getenv("SCROOGE_TQDM_POSITION_BASE", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw) + int(extra_offset))
    except ValueError:
        return None


def _tqdm_desc(desc: str) -> str:
    prefix = os.getenv("SCROOGE_TQDM_DESC_PREFIX", "").strip()
    if not prefix:
        return desc
    return f"{prefix} {desc}"


def _tqdm_kwargs(*, desc: str, extra_offset: int = 0, **kwargs: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "desc": _tqdm_desc(desc),
        "dynamic_ncols": True,
        **kwargs,
    }
    position = _tqdm_position(extra_offset=extra_offset)
    if position is not None:
        merged["position"] = position
    return merged


def _resolve_agg_trade_cache_path(
    *,
    cache_dir: str | Path | None,
    symbol: str,
    source: str,
    cache_key: str,
) -> Path | None:
    if cache_dir is None:
        return None
    normalized = str(cache_dir).strip()
    if not normalized:
        return None
    target_dir = Path(normalized).expanduser()
    normalized_cache_key = (
        str(cache_key or "rolling")
        .strip()
        .replace(" ", "_")
        .replace(":", "-")
        .replace("/", "-")
    )
    return target_dir / f"{symbol}_aggTrades_{source}_{normalized_cache_key}.pkl"


def _resolve_archive_day_cache_dir(
    *,
    cache_dir: str | Path | None,
    symbol: str,
) -> Path | None:
    if cache_dir is None:
        return None
    normalized = str(cache_dir).strip()
    if not normalized:
        return None
    return Path(normalized).expanduser() / symbol / "archive_daily"


def _resolve_archive_day_cache_path(
    *,
    cache_dir: str | Path | None,
    symbol: str,
    day: date,
) -> Path | None:
    target_dir = _resolve_archive_day_cache_dir(cache_dir=cache_dir, symbol=symbol)
    if target_dir is None:
        return None
    return target_dir / f"{day.isoformat()}.pkl"


def _resolve_agg_trade_cache_key(
    *,
    backtest_period_days: int,
    backtest_period_end_time: str,
    backtest_period_start_time: str,
) -> str:
    return resolve_backtest_time_range(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
        backtest_period_start_time=backtest_period_start_time,
    ).cache_key


def _read_agg_trade_cache(path: Path) -> pd.DataFrame:
    cached = pd.read_pickle(path)
    if not isinstance(cached, pd.DataFrame):
        raise ValueError(f"Agg trade cache is not a DataFrame: {path}")
    expected_columns = {"ts", "price", "qty", "source"}
    if not expected_columns.issubset(set(cached.columns)):
        raise ValueError(f"Agg trade cache is missing columns: {path}")
    out = cached.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["source"] = out["source"].astype(str)
    out = out.dropna(subset=["ts", "price", "qty"]).sort_values("ts").reset_index(drop=True)
    return out[["ts", "price", "qty", "source"]]


def _write_agg_trade_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def _archive_day_specs(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
) -> list[tuple[date, str]]:
    start_date = start_time.date()
    end_date = end_time.date()
    specs: list[tuple[date, str]] = []
    current = start_date
    while current <= end_date:
        datestr = current.isoformat()
        specs.append((current, f"{base_url}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{datestr}.zip"))
        current += timedelta(days=1)
    return specs


def _download_response_payload(url: str, *, desc: str, leave_progress: bool = False) -> bytes:
    with urlopen(url) as response:  # noqa: S310
        total_bytes_header = response.headers.get("Content-Length")
        try:
            total_bytes = int(total_bytes_header) if total_bytes_header else 0
        except ValueError:
            total_bytes = 0

        if total_bytes <= 0:
            return response.read()

        chunks: list[bytes] = []
        with tqdm(
            total=total_bytes,
            **_tqdm_kwargs(
                desc=desc,
                extra_offset=1,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=leave_progress,
            ),
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                progress.update(len(chunk))
        return b"".join(chunks)


def _parse_archive_csv_rows(file_obj) -> list[tuple[int, float, float]]:
    text_stream = TextIOWrapper(file_obj, encoding="utf-8")
    reader = csv.reader(text_stream)
    rows: list[tuple[int, float, float]] = []
    column_index: dict[str, int] | None = None

    for raw_row in reader:
        if not raw_row:
            continue
        row = [item.strip() for item in raw_row]
        if not row:
            continue
        lower_row = [item.lower() for item in row]
        if column_index is None:
            header_like = {"agg_tradeid", "agg_trade_id", "price", "quantity", "qty", "transact_time", "timestamp", "time", "p", "q", "t"}
            if any(item in header_like for item in lower_row):
                column_index = {
                    "price": next((idx for idx, item in enumerate(lower_row) if item in {"price", "p"}), -1),
                    "qty": next((idx for idx, item in enumerate(lower_row) if item in {"quantity", "qty", "q"}), -1),
                    "ts": next((idx for idx, item in enumerate(lower_row) if item in {"transact_time", "timestamp", "time", "t"}), -1),
                }
                continue
            column_index = {"price": 1, "qty": 2, "ts": 5}

        try:
            price = float(row[column_index["price"]])
            qty = float(row[column_index["qty"]])
            ts = int(float(row[column_index["ts"]]))
        except (IndexError, ValueError, TypeError):
            continue
        rows.append((ts, price, qty))

    return rows


def _download_archive_day_rows(
    *,
    symbol: str,
    day: date,
    base_url: str,
) -> list[tuple[int, float, float]]:
    url = f"{base_url}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    try:
        payload = _download_response_payload(
            url,
            desc=f"  {day.isoformat()}",
            leave_progress=False,
        )
    except HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    except URLError:
        raise

    rows: list[tuple[int, float, float]] = []
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        member_name = next((name for name in zf.namelist() if name.lower().endswith(".csv")), None)
        if member_name is None:
            return rows
        with zf.open(member_name) as file_obj:
            rows.extend(_parse_archive_csv_rows(file_obj))
    return rows


def _filter_trade_frame_to_range(
    df: pd.DataFrame,
    *,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    filtered = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()
    return filtered.sort_values("ts").reset_index(drop=True)


def _split_trade_frame_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    if df.empty:
        return {}
    working = df.copy()
    working["day"] = working["ts"].dt.date
    partitions: dict[date, pd.DataFrame] = {}
    for day_value, partition in working.groupby("day", sort=True):
        if not isinstance(day_value, date):
            continue
        partitions[day_value] = partition.drop(columns=["day"]).reset_index(drop=True)
    return partitions


def _seed_archive_day_caches_from_frame(
    *,
    df: pd.DataFrame,
    cache_dir: str | Path | None,
    symbol: str,
) -> int:
    written = 0
    for day_value, partition in _split_trade_frame_by_day(df).items():
        shard_path = _resolve_archive_day_cache_path(
            cache_dir=cache_dir,
            symbol=symbol,
            day=day_value,
        )
        if shard_path is None or shard_path.exists():
            continue
        _write_agg_trade_cache(shard_path, partition)
        written += 1
    return written


def _download_archive_rows(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = DEFAULT_AGG_TRADE_CACHE_DIR,
) -> tuple[list[tuple[int, float, float]], int, int]:
    day_specs = _archive_day_specs(symbol=symbol, start_time=start_time, end_time=end_time, base_url=base_url)
    day_frames: list[pd.DataFrame] = []
    cache_hits = 0
    downloaded_days = 0
    today_utc = datetime.now(UTC).date()

    with tqdm(day_specs, **_tqdm_kwargs(desc="aggTrades Archive", unit="day")) as archive_progress:
        for day_value, _url in archive_progress:
            day_label = day_value.isoformat()
            archive_progress.set_postfix_str(day_label)

            shard_path = _resolve_archive_day_cache_path(
                cache_dir=cache_dir if cache_enabled else None,
                symbol=symbol,
                day=day_value,
            )
            if cache_enabled and shard_path is not None and shard_path.exists():
                day_frames.append(_read_agg_trade_cache(shard_path))
                cache_hits += 1
                archive_progress.set_postfix_str(f"{day_label} cache")
                continue

            try:
                day_rows = _download_archive_day_rows(
                    symbol=symbol,
                    day=day_value,
                    base_url=base_url,
                )
            except HTTPError as exc:
                if exc.code == 404:
                    archive_progress.set_postfix_str(f"{day_label} missing")
                    continue
                raise
            except URLError:
                raise

            if day_rows:
                day_df = pd.DataFrame(day_rows, columns=["ts", "price", "qty"])
                day_df = day_df.sort_values("ts").drop_duplicates(subset=["ts", "price", "qty"], keep="last")
                day_df["ts"] = pd.to_datetime(day_df["ts"], unit="ms")
                day_df["source"] = "archive"
                day_df = day_df.reset_index(drop=True)
            else:
                day_df = pd.DataFrame(columns=["ts", "price", "qty", "source"])

            if (
                cache_enabled
                and shard_path is not None
                and (not day_df.empty or day_value < today_utc)
            ):
                _write_agg_trade_cache(shard_path, day_df)
            downloaded_days += 1
            day_frames.append(day_df)
            archive_progress.set_postfix_str(f"{day_label} cache={cache_hits} dl={downloaded_days}")

    if not day_frames:
        return [], cache_hits, downloaded_days

    combined = pd.concat(day_frames, ignore_index=True) if len(day_frames) > 1 else day_frames[0].copy()
    filtered = _filter_trade_frame_to_range(
        combined,
        start_time=start_time,
        end_time=end_time,
    )
    if filtered.empty:
        return [], cache_hits, downloaded_days

    rows: list[tuple[int, float, float]] = []
    for row in filtered.itertuples(index=False):
        rows.append((int(pd.Timestamp(row.ts).timestamp() * 1000), float(row.price), float(row.qty)))
    return rows, cache_hits, downloaded_days


def _download_rest_rows(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
    limit: int = 1000,
) -> list[tuple[int, float, float]]:
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    rows: list[tuple[int, float, float]] = []
    cursor = start_ms

    total_hours = max((end_ms - start_ms) / (60 * 60 * 1000), 1.0)
    with tqdm(total=total_hours, **_tqdm_kwargs(desc="aggTrades REST", unit="h")) as rest_progress:
        while cursor <= end_ms:
            query = urlencode(
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": limit,
                }
            )
            url = f"{base_url}/fapi/v1/aggTrades?{query}"
            with urlopen(url) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list) or not payload:
                break

            batch_max_ts = cursor
            batch_added = 0
            for item in payload:
                if not isinstance(item, dict):
                    continue
                try:
                    ts = int(item["T"])
                    price = float(item["p"])
                    qty = float(item["q"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ts < start_ms or ts > end_ms:
                    continue
                rows.append((ts, price, qty))
                batch_added += 1
                if ts > batch_max_ts:
                    batch_max_ts = ts

            covered_hours = max(0.0, (min(batch_max_ts, end_ms) - start_ms) / (60 * 60 * 1000))
            rest_progress.update(max(0.0, covered_hours - rest_progress.n))
            rest_progress.set_postfix(rows=len(rows))

            if batch_added == 0:
                break
            if len(payload) < limit:
                break
            cursor = batch_max_ts + 1

    return rows


def _normalize_trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts", "price", "qty", "source"])
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    if "source" not in out.columns:
        out["source"] = "archive"
    out["source"] = out["source"].astype(str)
    out = out.dropna(subset=["ts", "price", "qty"]).sort_values("ts")
    out = out.drop_duplicates(subset=["ts", "price", "qty"], keep="last").reset_index(drop=True)
    return out[["ts", "price", "qty", "source"]]


def _trade_frame_from_rows(
    rows: list[tuple[int, float, float]],
    *,
    source: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "price", "qty", "source"])
    df = pd.DataFrame(rows, columns=["ts", "price", "qty"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["source"] = source
    return _normalize_trade_frame(df)


def _iter_requested_days(start_time: datetime, end_time: datetime) -> list[date]:
    return [day_value for day_value, _ in _archive_day_specs(symbol="", start_time=start_time, end_time=end_time, base_url="") for day_value in [day_value]]


def _resolve_day_time_bounds(day_value: date, *, start_time: datetime, end_time: datetime) -> tuple[datetime, datetime]:
    day_start = datetime.combine(day_value, datetime.min.time())
    next_day_start = day_start + timedelta(days=1)
    effective_start = max(start_time, day_start)
    effective_end_exclusive = min(end_time + timedelta(microseconds=1), next_day_start)
    effective_end_inclusive = effective_end_exclusive - timedelta(milliseconds=1)
    return effective_start, effective_end_inclusive


def _load_archive_trade_day_frame(
    *,
    symbol: str,
    day_value: date,
    start_time: datetime,
    end_time: datetime,
    source: str,
    archive_base_url: str,
    rest_base_url: str,
    cache_enabled: bool,
    cache_dir: str | Path | None,
) -> tuple[pd.DataFrame, bool, str]:
    normalized_source = str(source or "archive").strip().lower() or "archive"
    if normalized_source not in {"archive", "rest", "auto"}:
        raise ValueError("agg_trade_source must be one of: archive, rest, auto")

    effective_start, effective_end = _resolve_day_time_bounds(
        day_value,
        start_time=start_time,
        end_time=end_time,
    )
    if effective_end < effective_start:
        return pd.DataFrame(columns=["ts", "price", "qty", "source"]), True, "archive"

    shard_path = _resolve_archive_day_cache_path(
        cache_dir=cache_dir if cache_enabled else None,
        symbol=symbol,
        day=day_value,
    )
    if cache_enabled and shard_path is not None and shard_path.exists():
        cached = _read_agg_trade_cache(shard_path)
        return _filter_trade_frame_to_range(cached, start_time=effective_start, end_time=effective_end), True, str(
            cached["source"].iloc[0]
        ) if not cached.empty else "archive"

    rows: list[tuple[int, float, float]] = []
    resolved_source = normalized_source
    if normalized_source in {"archive", "auto"}:
        try:
            rows = _download_archive_day_rows(
                symbol=symbol,
                day=day_value,
                base_url=archive_base_url,
            )
            if rows:
                resolved_source = "archive"
        except (HTTPError, URLError, OSError):
            rows = []

    if not rows and normalized_source in {"archive", "auto", "rest"}:
        rows = _download_rest_rows(
            symbol=symbol,
            start_time=effective_start,
            end_time=effective_end,
            base_url=rest_base_url,
        )
        resolved_source = "rest"

    day_df = _trade_frame_from_rows(rows, source=resolved_source)
    filtered = _filter_trade_frame_to_range(day_df, start_time=effective_start, end_time=effective_end)

    today_utc = datetime.now(UTC).date()
    if cache_enabled and shard_path is not None and (not day_df.empty or day_value < today_utc):
        _write_agg_trade_cache(shard_path, day_df)

    return filtered, False, resolved_source


def _resample_candle_frame(candles: pd.DataFrame, *, interval: str) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    working = candles.copy()
    working["open_time"] = pd.to_datetime(working["open_time"], errors="coerce")
    working = working.dropna(subset=["open_time"]).sort_values("open_time")
    working = working.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    freq = {"m": "min", "h": "h", "d": "d"}[interval[-1]]
    rule = f"{int(interval[:-1])}{freq}"
    resampled = (
        working.resample(rule, label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled.rename(columns={"open_time": "open_time"})


def write_historical_agg_trade_market_event_stream(
    *,
    symbol: str,
    backtest_period_days: int,
    backtest_period_end_time: str,
    backtest_period_start_time: str,
    intervals: dict[str, str],
    output_path: str | Path,
    source: str = "archive",
    tick_interval: str = "1s",
    archive_base_url: str = DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL,
    rest_base_url: str = DEFAULT_AGG_TRADE_REST_BASE_URL,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = DEFAULT_AGG_TRADE_CACHE_DIR,
    progress_reporter: BacktestProgressReporter | None = None,
) -> tuple[pd.DataFrame, AggTradeMarketEventStreamSummary]:
    archive_base_url = str(archive_base_url or DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL).strip().rstrip("/")
    rest_base_url = str(rest_base_url or DEFAULT_AGG_TRADE_REST_BASE_URL).strip().rstrip("/")
    time_range = resolve_backtest_time_range(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
        backtest_period_start_time=backtest_period_start_time,
    )
    start_time, end_time = time_range.start_time, time_range.end_time
    requested_days = _iter_requested_days(start_time, end_time)
    if not requested_days:
        empty_df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "EMA", "RSI", "BBL", "BBM", "BBU", "ATR"])
        return empty_df, AggTradeMarketEventStreamSummary(
            source="historical_agg_trade",
            symbol=symbol,
            cache_hit=False,
            cache_path=str(_resolve_archive_day_cache_dir(cache_dir=cache_dir, symbol=symbol)) if cache_dir else None,
            raw_agg_trades=0,
            price_ticks=0,
            candle_events_small=0,
            candle_events_medium=0,
            candle_events_big=0,
            indicator_snapshots=0,
            total_events=0,
            first_trade_ts=None,
            last_trade_ts=None,
        )

    small_interval = str(intervals["small"])
    medium_interval = str(intervals["medium"])
    big_interval = str(intervals["big"])

    indicator_frames: list[pd.DataFrame] = []
    raw_agg_trades = 0
    first_trade_ts: str | None = None
    last_trade_ts: str | None = None
    sources_seen: set[str] = set()
    all_days_from_cache = True
    archive_cache_path = _resolve_archive_day_cache_dir(cache_dir=cache_dir, symbol=symbol)
    feature_builder = FeatureFrameBuilder()
    empty_df = pd.DataFrame(
        columns=["open_time", "open", "high", "low", "close", "volume", "EMA", "RSI", "BBL", "BBM", "BBU", "ATR"]
    )
    target_path = Path(output_path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    price_tick_count = 0
    small_candle_count = 0
    medium_candle_count = 0
    big_candle_count = 0
    indicator_snapshot_count = 0

    with target_path.open("w", encoding="utf-8") as file_obj:
        if progress_reporter is not None:
            progress_reporter.start_stage("aggTrades Stream", len(requested_days))
        with tqdm(requested_days, **_tqdm_kwargs(desc="aggTrades Stream", unit="day", disable=progress_reporter is not None)) as pass_days:
            for day_value in pass_days:
                day_df, cache_hit, day_source = _load_archive_trade_day_frame(
                    symbol=symbol,
                    day_value=day_value,
                    start_time=start_time,
                    end_time=end_time,
                    source=source,
                    archive_base_url=archive_base_url,
                    rest_base_url=rest_base_url,
                    cache_enabled=cache_enabled,
                    cache_dir=cache_dir,
                )
                all_days_from_cache = all_days_from_cache and cache_hit
                sources_seen.add(day_source)
                if not day_df.empty:
                    raw_agg_trades += len(day_df)
                    first_trade_ts = first_trade_ts or _format_ts(day_df["ts"].iloc[0])
                    last_trade_ts = _format_ts(day_df["ts"].iloc[-1])
                    price_tick_events = _build_price_tick_events(
                        day_df,
                        symbol=symbol,
                        tick_interval=tick_interval,
                        source_label=f"historical_agg_trade_{day_source}",
                    )
                else:
                    price_tick_events = []
                small_day = _resample_trade_frame_to_candles(day_df, interval=small_interval)
                medium_day = _resample_candle_frame(small_day, interval=medium_interval)
                big_day = _resample_candle_frame(small_day, interval=big_interval)
                medium_feature_day = feature_builder.update_timeframe_frame(timeframe="medium", df=medium_day)
                big_feature_day = feature_builder.update_timeframe_frame(timeframe="big", df=big_day)
                indicator_day = merge_feature_frames(
                    df_small=small_day,
                    timeframe_feature_frames={
                        "medium": medium_feature_day,
                        "big": big_feature_day,
                    },
                    output_columns=feature_builder.output_columns,
                )
                if not indicator_day.empty:
                    indicator_frames.append(indicator_day)

                small_candle_events = _build_candle_events(small_day, symbol=symbol, interval=small_interval)
                medium_candle_events = _build_candle_events(medium_day, symbol=symbol, interval=medium_interval)
                big_candle_events = _build_candle_events(big_day, symbol=symbol, interval=big_interval)
                indicator_snapshot_events = _build_indicator_snapshot_events(
                    indicator_day,
                    symbol=symbol,
                    small_interval=small_interval,
                )

                merged_day_events = heapq.merge(
                    price_tick_events,
                    small_candle_events,
                    medium_candle_events,
                    big_candle_events,
                    indicator_snapshot_events,
                    key=lambda event: (
                        str(getattr(event, "ts", "")),
                        _event_priority(
                            event,
                            small_interval=small_interval,
                            medium_interval=medium_interval,
                            big_interval=big_interval,
                        ),
                    ),
                )
                for event in merged_day_events:
                    file_obj.write(json.dumps(market_event_to_dict(event), ensure_ascii=True, sort_keys=True))
                    file_obj.write("\n")
                price_tick_count += len(price_tick_events)
                small_candle_count += len(small_candle_events)
                medium_candle_count += len(medium_candle_events)
                big_candle_count += len(big_candle_events)
                indicator_snapshot_count += len(indicator_snapshot_events)
                pass_days.set_postfix_str(f"{day_value.isoformat()} rows={raw_agg_trades}")
                if progress_reporter is not None:
                    progress_reporter.advance()
        if progress_reporter is not None:
            progress_reporter.complete_stage()

    indicator_df = (
        pd.concat(indicator_frames, ignore_index=True)
        if indicator_frames
        else empty_df.copy()
    )

    source_label = "historical_agg_trade"
    if sources_seen:
        if sources_seen == {"archive"}:
            source_label = "historical_agg_trade_archive"
        elif sources_seen == {"rest"}:
            source_label = "historical_agg_trade_rest"
        else:
            source_label = "historical_agg_trade_archive+rest"

    return indicator_df, AggTradeMarketEventStreamSummary(
        source=source_label,
        symbol=symbol,
        cache_hit=all_days_from_cache,
        cache_path=str(archive_cache_path) if archive_cache_path is not None else None,
        raw_agg_trades=raw_agg_trades,
        price_ticks=price_tick_count,
        candle_events_small=small_candle_count,
        candle_events_medium=medium_candle_count,
        candle_events_big=big_candle_count,
        indicator_snapshots=indicator_snapshot_count,
        total_events=price_tick_count + small_candle_count + medium_candle_count + big_candle_count + indicator_snapshot_count,
        first_trade_ts=first_trade_ts,
        last_trade_ts=last_trade_ts,
    )


def fetch_historical_agg_trades(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    source: str = "archive",
    archive_base_url: str = DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL,
    rest_base_url: str = DEFAULT_AGG_TRADE_REST_BASE_URL,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = DEFAULT_AGG_TRADE_CACHE_DIR,
    cache_key: str | None = None,
) -> tuple[pd.DataFrame, bool, Path | None]:
    normalized_source = str(source or "archive").strip().lower() or "archive"
    archive_base_url = str(archive_base_url or DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL).strip().rstrip("/")
    rest_base_url = str(rest_base_url or DEFAULT_AGG_TRADE_REST_BASE_URL).strip().rstrip("/")
    if normalized_source not in {"archive", "rest", "auto"}:
        raise ValueError("agg_trade_source must be one of: archive, rest, auto")

    cache_path = _resolve_agg_trade_cache_path(
        cache_dir=cache_dir if cache_enabled else None,
        symbol=symbol,
        source=normalized_source,
        cache_key=(str(cache_key).strip() if cache_key is not None and str(cache_key).strip() else "rolling"),
    )
    archive_daily_cache_dir = _resolve_archive_day_cache_dir(
        cache_dir=cache_dir if cache_enabled else None,
        symbol=symbol,
    )
    if cache_enabled and cache_path is not None and cache_path.exists():
        technical_logger.info("agg_trade_cache_found path=%s", cache_path)
        cached = _read_agg_trade_cache(cache_path)
        if normalized_source in {"archive", "auto"}:
            seeded = _seed_archive_day_caches_from_frame(
                df=cached,
                cache_dir=cache_dir,
                symbol=symbol,
            )
            if seeded > 0:
                technical_logger.info(
                    "agg_trade_daily_cache_seeded source=%s path=%s shards=%s",
                    normalized_source,
                    archive_daily_cache_dir,
                    seeded,
                )
        return cached, True, cache_path

    rows: list[tuple[int, float, float]] = []
    resolved_source = normalized_source
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    cache_hit = False
    resolved_cache_path = cache_path

    if normalized_source in {"archive", "auto"}:
        rows, archive_cache_hits, archive_downloaded_days = _download_archive_rows(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            base_url=archive_base_url,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )
        if cache_enabled and archive_daily_cache_dir is not None:
            resolved_cache_path = archive_daily_cache_dir
        cache_hit = archive_cache_hits > 0 and archive_downloaded_days == 0
        had_archive_rows = bool(rows)
        if rows:
            resolved_source = "archive"
            if cache_enabled and archive_daily_cache_dir is not None:
                technical_logger.info(
                    "agg_trade_daily_cache_used path=%s range=%s..%s",
                    archive_daily_cache_dir,
                    start_time.date().isoformat(),
                    end_time.date().isoformat(),
                )
        latest_ts = max((item[0] for item in rows), default=(start_ms - 1))
        if latest_ts < end_ms:
            rest_start_time = datetime.fromtimestamp((latest_ts + 1) / 1000)
            try:
                gap_rows = _download_rest_rows(
                    symbol=symbol,
                    start_time=rest_start_time,
                    end_time=end_time,
                    base_url=rest_base_url,
                )
            except (HTTPError, URLError, OSError) as exc:
                if had_archive_rows:
                    technical_logger.warning(
                        "agg_trade_rest_gap_fill_failed symbol=%s start=%s end=%s error=%s fallback=archive_only",
                        symbol,
                        rest_start_time.isoformat(),
                        end_time.isoformat(),
                        exc,
                    )
                    gap_rows = []
                else:
                    raise
            if gap_rows:
                rows.extend(gap_rows)
                resolved_source = "archive+rest" if had_archive_rows else "rest"
                cache_hit = False

    if not rows and normalized_source in {"rest", "auto"}:
        rows = _download_rest_rows(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            base_url=rest_base_url,
        )
        resolved_source = "rest"
        resolved_cache_path = cache_path
        cache_hit = False

    if not rows:
        return pd.DataFrame(columns=["ts", "price", "qty", "source"]), cache_hit, resolved_cache_path

    df = pd.DataFrame(rows, columns=["ts", "price", "qty"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts", "price", "qty"], keep="last")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["source"] = resolved_source
    output = df.reset_index(drop=True)
    if cache_enabled and cache_path is not None and normalized_source == "rest":
        _write_agg_trade_cache(cache_path, output)
        technical_logger.info("agg_trade_cache_written path=%s rows=%s", cache_path, len(output))
    if cache_enabled and normalized_source in {"archive", "auto"}:
        seeded = _seed_archive_day_caches_from_frame(
            df=output[output["source"].astype(str).str.contains("archive", na=False)].copy(),
            cache_dir=cache_dir,
            symbol=symbol,
        )
        if seeded > 0:
            technical_logger.info(
                "agg_trade_daily_cache_seeded source=%s path=%s shards=%s",
                normalized_source,
                archive_daily_cache_dir,
                seeded,
            )
    return output, cache_hit, resolved_cache_path


def _resample_trade_frame_to_candles(trades: pd.DataFrame, *, interval: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    working = trades.sort_values("ts").set_index("ts")[["price", "qty"]]
    freq = {"m": "min", "h": "h", "d": "d"}[interval[-1]]
    rule = f"{int(interval[:-1])}{freq}"
    resampled = (
        working.resample(rule, label="left", closed="left")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("qty", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
        .rename(columns={"ts": "open_time"})
    )
    return resampled


def _compute_indicator_snapshots(
    *,
    small_df: pd.DataFrame,
    medium_df: pd.DataFrame,
    big_df: pd.DataFrame,
) -> pd.DataFrame:
    return build_feature_frame(
        df_small=small_df,
        df_medium=medium_df,
        df_big=big_df,
    )


def _close_time_from_open_time(open_time: pd.Timestamp, interval: str) -> str:
    value = str(interval or "").strip().lower()
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "m":
        delta = timedelta(minutes=amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    elif unit == "d":
        delta = timedelta(days=amount)
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    return _format_ts(open_time + delta - timedelta(seconds=1))


def _normalize_tick_interval_seconds(value: str) -> int | None:
    normalized = str(value or "1s").strip().lower() or "1s"
    if normalized == "raw":
        return None
    if not normalized.endswith("s"):
        raise ValueError("agg_trade_tick_interval must be 'raw' or an integer number of seconds like 1s, 5s, 15s")
    try:
        seconds = int(normalized[:-1])
    except ValueError as exc:
        raise ValueError("agg_trade_tick_interval must be 'raw' or an integer number of seconds like 1s, 5s, 15s") from exc
    if seconds <= 0:
        raise ValueError("agg_trade_tick_interval seconds must be greater than zero")
    return seconds


def _build_price_tick_events(
    trades: pd.DataFrame,
    *,
    symbol: str,
    tick_interval: str,
    source_label: str,
) -> list[PriceTickEvent]:
    if trades.empty:
        return []

    interval_seconds = _normalize_tick_interval_seconds(tick_interval)
    if interval_seconds is None:
        tick_df = trades[["ts", "price"]].copy()
    else:
        normalized = f"{interval_seconds}s"
        tick_df = (
            trades.assign(bucket=trades["ts"].dt.floor(normalized))
            .groupby("bucket", as_index=False)
            .agg(price=("price", "last"))
            .rename(columns={"bucket": "ts"})
        )

    return [
        PriceTickEvent(
            symbol=symbol,
            ts=_format_ts(row.ts),
            price=float(row.price),
            source=source_label,
        )
        for row in tick_df.itertuples(index=False)
    ]


def _build_candle_events(candles: pd.DataFrame, *, symbol: str, interval: str) -> list[CandleClosedEvent]:
    return [
        CandleClosedEvent(
            symbol=symbol,
            ts=_close_time_from_open_time(pd.Timestamp(row.open_time), interval),
            interval=interval,
            open_time=_format_ts(pd.Timestamp(row.open_time)),
            close_time=_close_time_from_open_time(pd.Timestamp(row.open_time), interval),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in candles.itertuples(index=False)
    ]


def _build_indicator_snapshot_events(indicator_df: pd.DataFrame, *, symbol: str, small_interval: str) -> list[IndicatorSnapshotEvent]:
    events: list[IndicatorSnapshotEvent] = []
    for row in indicator_df.itertuples(index=False):
        open_time = pd.Timestamp(row.open_time)
        close_time = _close_time_from_open_time(open_time, small_interval)
        events.append(
            IndicatorSnapshotEvent(
                symbol=symbol,
                ts=close_time,
                interval="discrete_snapshot",
                values={
                    "EMA": (float(row.EMA) if pd.notna(row.EMA) else None),
                    "RSI": (float(row.RSI) if pd.notna(row.RSI) else None),
                    "BBL": (float(row.BBL) if pd.notna(row.BBL) else None),
                    "BBM": (float(row.BBM) if pd.notna(row.BBM) else None),
                    "BBU": (float(row.BBU) if pd.notna(row.BBU) else None),
                    "ATR": (float(row.ATR) if pd.notna(row.ATR) else None),
                },
            )
        )
    return events


def _event_priority(event: MarketEvent, *, small_interval: str, medium_interval: str, big_interval: str) -> int:
    if isinstance(event, PriceTickEvent):
        return 10
    if isinstance(event, CandleClosedEvent):
        if event.interval == small_interval:
            return 20
        if event.interval == medium_interval:
            return 30
        if event.interval == big_interval:
            return 40
        return 45
    if isinstance(event, IndicatorSnapshotEvent):
        return 50
    return 90


def build_market_events_from_agg_trade_frame(
    trades: pd.DataFrame,
    *,
    symbol: str,
    intervals: dict[str, str],
    tick_interval: str = "1s",
    source_label: str = "historical_agg_trade",
) -> tuple[list[MarketEvent], AggTradeMarketEventStreamSummary]:
    if trades.empty:
        return [], AggTradeMarketEventStreamSummary(
            source=source_label,
            symbol=symbol,
            cache_hit=False,
            cache_path=None,
            raw_agg_trades=0,
            price_ticks=0,
            candle_events_small=0,
            candle_events_medium=0,
            candle_events_big=0,
            indicator_snapshots=0,
            total_events=0,
            first_trade_ts=None,
            last_trade_ts=None,
        )

    small_interval = str(intervals["small"])
    medium_interval = str(intervals["medium"])
    big_interval = str(intervals["big"])

    normalized = trades.copy()
    normalized["ts"] = pd.to_datetime(normalized["ts"], errors="coerce")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["qty"] = pd.to_numeric(normalized["qty"], errors="coerce")
    normalized = normalized.dropna(subset=["ts", "price", "qty"]).sort_values("ts").reset_index(drop=True)
    if normalized.empty:
        return [], AggTradeMarketEventStreamSummary(
            source=source_label,
            symbol=symbol,
            cache_hit=False,
            cache_path=None,
            raw_agg_trades=0,
            price_ticks=0,
            candle_events_small=0,
            candle_events_medium=0,
            candle_events_big=0,
            indicator_snapshots=0,
            total_events=0,
            first_trade_ts=None,
            last_trade_ts=None,
        )

    small_candles = _resample_trade_frame_to_candles(normalized, interval=small_interval)
    medium_candles = _resample_trade_frame_to_candles(normalized, interval=medium_interval)
    big_candles = _resample_trade_frame_to_candles(normalized, interval=big_interval)
    indicator_df = _compute_indicator_snapshots(
        small_df=small_candles,
        medium_df=medium_candles,
        big_df=big_candles,
    )

    events: list[MarketEvent] = []
    price_tick_events = _build_price_tick_events(
        normalized,
        symbol=symbol,
        tick_interval=tick_interval,
        source_label=source_label,
    )
    small_candle_events = _build_candle_events(small_candles, symbol=symbol, interval=small_interval)
    medium_candle_events = _build_candle_events(medium_candles, symbol=symbol, interval=medium_interval)
    big_candle_events = _build_candle_events(big_candles, symbol=symbol, interval=big_interval)
    indicator_events = _build_indicator_snapshot_events(indicator_df, symbol=symbol, small_interval=small_interval)

    events.extend(price_tick_events)
    events.extend(small_candle_events)
    events.extend(medium_candle_events)
    events.extend(big_candle_events)
    events.extend(indicator_events)
    events.sort(
        key=lambda event: (
            str(getattr(event, "ts", "")),
            _event_priority(
                event,
                small_interval=small_interval,
                medium_interval=medium_interval,
                big_interval=big_interval,
            ),
        )
    )

    return events, AggTradeMarketEventStreamSummary(
        source=source_label,
        symbol=symbol,
        cache_hit=False,
        cache_path=None,
        raw_agg_trades=len(normalized),
        price_ticks=len(price_tick_events),
        candle_events_small=len(small_candle_events),
        candle_events_medium=len(medium_candle_events),
        candle_events_big=len(big_candle_events),
        indicator_snapshots=len(indicator_events),
        total_events=len(events),
        first_trade_ts=_format_ts(normalized["ts"].iloc[0]),
        last_trade_ts=_format_ts(normalized["ts"].iloc[-1]),
    )


def build_historical_agg_trade_market_event_stream(
    *,
    symbol: str,
    backtest_period_days: int,
    backtest_period_end_time: str,
    backtest_period_start_time: str,
    intervals: dict[str, str],
    source: str = "archive",
    tick_interval: str = "1s",
    archive_base_url: str = DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL,
    rest_base_url: str = DEFAULT_AGG_TRADE_REST_BASE_URL,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = DEFAULT_AGG_TRADE_CACHE_DIR,
) -> tuple[list[MarketEvent], AggTradeMarketEventStreamSummary]:
    archive_base_url = str(archive_base_url or DEFAULT_AGG_TRADE_ARCHIVE_BASE_URL).strip().rstrip("/")
    rest_base_url = str(rest_base_url or DEFAULT_AGG_TRADE_REST_BASE_URL).strip().rstrip("/")
    cache_key = _resolve_agg_trade_cache_key(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
        backtest_period_start_time=backtest_period_start_time,
    )
    time_range = resolve_backtest_time_range(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
        backtest_period_start_time=backtest_period_start_time,
    )
    start_time, end_time = time_range.start_time, time_range.end_time
    trades, cache_hit, cache_path = fetch_historical_agg_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        source=source,
        archive_base_url=archive_base_url,
        rest_base_url=rest_base_url,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    events, summary = build_market_events_from_agg_trade_frame(
        trades,
        symbol=symbol,
        intervals=intervals,
        tick_interval=tick_interval,
        source_label=("historical_agg_trade" if trades.empty else f"historical_agg_trade_{str(trades['source'].iloc[0])}"),
    )
    summary.cache_hit = cache_hit
    summary.cache_path = str(cache_path) if cache_path is not None else None
    return events, summary
