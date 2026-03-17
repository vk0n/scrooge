from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
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
import pandas_ta as ta

from bot.event_log import get_technical_logger
from core.market_events import CandleClosedEvent, IndicatorSnapshotEvent, MarketEvent, PriceTickEvent


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


def _resolve_time_range(*, backtest_period_days: int, backtest_period_end_time: str) -> tuple[datetime, datetime]:
    if str(backtest_period_end_time or "").strip():
        end_time = datetime.fromisoformat(str(backtest_period_end_time).strip())
    else:
        end_time = datetime.now()
    start_time = end_time - timedelta(days=int(backtest_period_days))
    return start_time, end_time


def _resolve_agg_trade_cache_path(
    *,
    cache_dir: str | Path | None,
    symbol: str,
    source: str,
    start_time: datetime,
    end_time: datetime,
) -> Path | None:
    if cache_dir is None:
        return None
    normalized = str(cache_dir).strip()
    if not normalized:
        return None
    target_dir = Path(normalized).expanduser()
    return target_dir / (
        f"{symbol}_aggTrades_{source}_"
        f"{int(start_time.timestamp() * 1000)}_{int(end_time.timestamp() * 1000)}.pkl"
    )


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


def _archive_day_urls(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
) -> list[str]:
    start_date = start_time.date()
    end_date = end_time.date()
    urls: list[str] = []
    current = start_date
    while current <= end_date:
        datestr = current.isoformat()
        urls.append(f"{base_url}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{datestr}.zip")
        current += timedelta(days=1)
    return urls


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


def _download_archive_rows(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    for url in _archive_day_urls(symbol=symbol, start_time=start_time, end_time=end_time, base_url=base_url):
        try:
            with urlopen(url) as response:  # noqa: S310
                payload = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        except URLError:
            raise

        with zipfile.ZipFile(BytesIO(payload)) as zf:
            member_name = next((name for name in zf.namelist() if name.lower().endswith(".csv")), None)
            if member_name is None:
                continue
            with zf.open(member_name) as file_obj:
                for ts, price, qty in _parse_archive_csv_rows(file_obj):
                    if ts < start_ms or ts > end_ms:
                        continue
                    rows.append((ts, price, qty))

    return rows


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

        if batch_added == 0:
            break
        if len(payload) < limit:
            break
        cursor = batch_max_ts + 1

    return rows


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
        start_time=start_time,
        end_time=end_time,
    )
    if cache_enabled and cache_path is not None and cache_path.exists():
        technical_logger.info("agg_trade_cache_found path=%s", cache_path)
        return _read_agg_trade_cache(cache_path), True, cache_path

    rows: list[tuple[int, float, float]] = []
    resolved_source = normalized_source
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    if normalized_source in {"archive", "auto"}:
        rows = _download_archive_rows(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            base_url=archive_base_url,
        )
        had_archive_rows = bool(rows)
        if rows:
            resolved_source = "archive"
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

    if not rows and normalized_source in {"rest", "auto"}:
        rows = _download_rest_rows(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            base_url=rest_base_url,
        )
        resolved_source = "rest"

    if not rows:
        return pd.DataFrame(columns=["ts", "price", "qty", "source"]), False, cache_path

    df = pd.DataFrame(rows, columns=["ts", "price", "qty"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts", "price", "qty"], keep="last")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["source"] = resolved_source
    output = df.reset_index(drop=True)
    if cache_enabled and cache_path is not None:
        _write_agg_trade_cache(cache_path, output)
        technical_logger.info("agg_trade_cache_written path=%s rows=%s", cache_path, len(output))
    return output, False, cache_path


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
    small = small_df.copy()
    medium = medium_df.copy()
    big = big_df.copy()

    small["open_time"] = pd.to_datetime(small["open_time"], errors="coerce")
    medium["open_time"] = pd.to_datetime(medium["open_time"], errors="coerce")
    big["open_time"] = pd.to_datetime(big["open_time"], errors="coerce")

    big = big.set_index("open_time")
    big["RSI"] = ta.rsi(big["close"], length=11)
    big["EMA"] = ta.ema(big["close"], length=50)
    big = big[["RSI", "EMA"]]

    medium = medium.set_index("open_time")
    bb = ta.bbands(medium["close"], length=20, std=2)
    if isinstance(bb, pd.DataFrame):
        medium["BBL"] = bb.get("BBL_20_2.0_2.0")
        medium["BBM"] = bb.get("BBM_20_2.0_2.0")
        medium["BBU"] = bb.get("BBU_20_2.0_2.0")
    else:
        medium["BBL"] = pd.Series(index=medium.index, dtype="float64")
        medium["BBM"] = pd.Series(index=medium.index, dtype="float64")
        medium["BBU"] = pd.Series(index=medium.index, dtype="float64")
    atr = ta.atr(medium["high"], medium["low"], medium["close"], length=14)
    medium["ATR"] = atr if atr is not None else pd.Series(index=medium.index, dtype="float64")
    medium = medium[["BBL", "BBM", "BBU", "ATR"]]

    merged = small.set_index("open_time")
    merged = merged.merge(medium, left_index=True, right_index=True, how="left").ffill()
    merged = merged.merge(big, left_index=True, right_index=True, how="left").ffill()
    merged.reset_index(inplace=True)
    return merged


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
    start_time, end_time = _resolve_time_range(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
    )
    trades, cache_hit, cache_path = fetch_historical_agg_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        source=source,
        archive_base_url=archive_base_url,
        rest_base_url=rest_base_url,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
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
