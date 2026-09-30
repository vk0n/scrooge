from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from threading import local
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from backtest.spot_scenario import SpotBacktestAsset, SpotBacktestScenario


BINANCE_SPOT_API = "https://api.binance.com"
REQUEST_ATTEMPTS = 4
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


class SpotMarketDataProgress(Protocol):
    def start_asset(self, symbol: str, *, index: int, total: int) -> None:
        ...

    def start_phase(self, symbol: str, label: str, *, total: int | None = None) -> None:
        ...

    def advance(self, symbol: str, amount: int = 1) -> None:
        ...

    def complete_asset(self, symbol: str, *, source: str, rows: int) -> None:
        ...


@dataclass(frozen=True, slots=True)
class SpotCandle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_binance_kline(self) -> list[Any]:
        return [
            self.open_time_ms,
            str(self.open),
            str(self.high),
            str(self.low),
            str(self.close),
            str(self.volume),
            self.close_time_ms,
        ]


@dataclass(frozen=True)
class SpotHistoricalDataset:
    candles: dict[str, tuple[SpotCandle, ...]]
    symbol_info: dict[str, dict[str, Any]]
    interval: str
    interval_ms: int
    source: str
    unavailable_open_times: dict[str, frozenset[int]] = field(default_factory=dict)


def interval_milliseconds(interval: str) -> int:
    normalized = str(interval or "").strip().lower()
    if normalized not in INTERVAL_SECONDS:
        raise ValueError(f"Unsupported Spot backtest interval: {interval}")
    return INTERVAL_SECONDS[normalized] * 1000


class BinanceSpotHistoricalAdapter:
    """Deterministic Binance Spot candle cache with strict gap validation."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        base_url: str = BINANCE_SPOT_API,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._progress: SpotMarketDataProgress | None = None
        self._progress_context = local()

    def load(
        self,
        scenario: SpotBacktestScenario,
        *,
        progress: SpotMarketDataProgress | None = None,
        max_workers: int = 1,
    ) -> SpotHistoricalDataset:
        self._progress = progress
        interval_ms = interval_milliseconds(scenario.interval)
        warmup_start = scenario.start - timedelta(
            milliseconds=interval_ms * scenario.warmup_candles
        )
        worker_count = max(1, min(int(max_workers), len(scenario.assets)))
        try:
            if worker_count == 1:
                loaded_assets = [
                    self._load_asset(
                        asset,
                        index=index,
                        total=len(scenario.assets),
                        interval=scenario.interval,
                        interval_ms=interval_ms,
                        start=warmup_start,
                        end=scenario.end,
                    )
                    for index, asset in enumerate(scenario.assets, start=1)
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="spot-market-data",
                ) as executor:
                    futures = [
                        executor.submit(
                            self._load_asset,
                            asset,
                            index=index,
                            total=len(scenario.assets),
                            interval=scenario.interval,
                            interval_ms=interval_ms,
                            start=warmup_start,
                            end=scenario.end,
                        )
                        for index, asset in enumerate(scenario.assets, start=1)
                    ]
                    loaded_assets = [future.result() for future in futures]
        finally:
            self._progress = None

        loaded_assets = self._repair_common_market_gaps(
            loaded_assets,
            interval_ms=interval_ms,
            start=warmup_start,
            end=scenario.end,
        )
        for item in loaded_assets:
            self._validate_candles(
                item["candles"],
                interval_ms=interval_ms,
                symbol=str(item["symbol_info"].get("symbol") or item["symbol"]),
            )

        return SpotHistoricalDataset(
            candles={item["symbol"]: item["candles"] for item in loaded_assets},
            symbol_info={item["symbol"]: item["symbol_info"] for item in loaded_assets},
            interval=scenario.interval,
            interval_ms=interval_ms,
            source=(
                "cache"
                if all(item["source"] == "cache" for item in loaded_assets)
                else "binance_spot_rest"
            ),
            unavailable_open_times={
                item["symbol"]: item["unavailable"]
                for item in loaded_assets
                if item["unavailable"]
            },
        )

    def _load_asset(
        self,
        asset: SpotBacktestAsset,
        *,
        index: int,
        total: int,
        interval: str,
        interval_ms: int,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        progress_symbol = asset.market_symbol
        self._progress_context.symbol = progress_symbol
        if self._progress is not None:
            self._progress.start_asset(progress_symbol, index=index, total=total)
        if asset.history_segments:
            rows, source, unavailable = self._load_stitched_candles(
                asset,
                interval=interval,
                start=start,
                end=end,
            )
        else:
            rows, source = self.load_candles(
                asset.market_symbol,
                interval=interval,
                start=start,
                end=end,
            )
            unavailable = set()
        self._start_progress_phase(f"Loading {asset.market_symbol} rules", total=1)
        symbol_info = asset.symbol_info or self.load_symbol_info(asset.market_symbol)
        self._advance_progress(1)
        if self._progress is not None:
            self._progress.complete_asset(progress_symbol, source=source, rows=len(rows))
        return {
            "symbol": asset.symbol,
            "candles": tuple(rows),
            "symbol_info": symbol_info,
            "source": source,
            "unavailable": frozenset(unavailable),
        }

    @staticmethod
    def _gap_ranges(
        rows: list[SpotCandle],
        *,
        interval_ms: int,
        start_ms: int,
        end_ms: int,
        symbol: str,
    ) -> tuple[tuple[int, int], ...]:
        if not rows:
            raise ValueError(f"No historical Spot candles are available for {symbol}.")
        gaps: list[tuple[int, int]] = []
        expected = start_ms
        for row in rows:
            if row.open_time_ms < expected:
                raise ValueError(f"{symbol} candles are duplicated or out of order.")
            if row.open_time_ms > expected:
                gaps.append((expected, row.open_time_ms))
            expected = row.open_time_ms + interval_ms
        if expected < end_ms:
            gaps.append((expected, end_ms))
        if expected > end_ms:
            raise ValueError(f"{symbol} candle cache extends beyond the requested range.")
        return tuple(gaps)

    @classmethod
    def _repair_common_market_gaps(
        cls,
        loaded_assets: list[dict[str, Any]],
        *,
        interval_ms: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        if not loaded_assets:
            return loaded_assets
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        gaps_by_symbol = {
            str(item["symbol"]): cls._gap_ranges(
                item["candles"],
                interval_ms=interval_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                symbol=str(item["symbol"]),
            )
            for item in loaded_assets
        }
        common_gaps = set.intersection(
            *(set(ranges) for ranges in gaps_by_symbol.values())
        )
        for symbol, ranges in gaps_by_symbol.items():
            unique = [gap for gap in ranges if gap not in common_gaps]
            if unique:
                gap_start, gap_end = unique[0]
                previous_time = datetime.fromtimestamp(
                    (gap_start - interval_ms) / 1000,
                    tz=UTC,
                ).isoformat()
                current_time = datetime.fromtimestamp(gap_end / 1000, tz=UTC).isoformat()
                raise ValueError(
                    f"Missing {symbol} candle between {previous_time} and "
                    f"{current_time}; gap is not market-wide, replay aborted."
                )
        if not common_gaps:
            return loaded_assets

        repaired: list[dict[str, Any]] = []
        for item in loaded_assets:
            rows_by_time = {row.open_time_ms: row for row in item["candles"]}
            unavailable = set(item.get("unavailable") or ())
            for gap_start, gap_end in sorted(common_gaps):
                previous = rows_by_time.get(gap_start - interval_ms)
                if previous is None:
                    raise ValueError(
                        f"Cannot value market-wide gap at replay boundary for {item['symbol']}."
                    )
                previous_close = previous.close
                for open_time_ms in range(gap_start, gap_end, interval_ms):
                    rows_by_time[open_time_ms] = SpotCandle(
                        open_time_ms=open_time_ms,
                        close_time_ms=open_time_ms + interval_ms - 1,
                        open=previous_close,
                        high=previous_close,
                        low=previous_close,
                        close=previous_close,
                        volume=0.0,
                    )
                    unavailable.add(open_time_ms)
            repaired.append(
                {
                    **item,
                    "candles": [rows_by_time[key] for key in sorted(rows_by_time)],
                    "unavailable": frozenset(unavailable),
                }
            )
        return repaired

    def _start_progress_phase(self, label: str, *, total: int | None = None) -> None:
        if self._progress is not None:
            self._progress.start_phase(self._progress_symbol(), label, total=total)

    def _advance_progress(self, amount: int = 1) -> None:
        if self._progress is not None and amount > 0:
            self._progress.advance(self._progress_symbol(), amount)

    def _progress_symbol(self) -> str:
        return str(getattr(self._progress_context, "symbol", "Spot"))

    def _load_stitched_candles(
        self,
        asset: SpotBacktestAsset,
        *,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> tuple[list[SpotCandle], str, set[int]]:
        interval_ms = interval_milliseconds(interval)
        combined: dict[int, SpotCandle] = {}
        sources: list[str] = []
        for segment in asset.history_segments:
            segment_start = max(start, segment.start) if segment.start is not None else start
            segment_end = min(end, segment.end) if segment.end is not None else end
            if segment_end <= segment_start:
                continue
            rows, source = self.load_candles(
                segment.market_symbol,
                interval=interval,
                start=segment_start,
                end=segment_end,
            )
            expected_start_ms = int(segment_start.astimezone(UTC).timestamp() * 1000)
            expected_end_ms = int(segment_end.astimezone(UTC).timestamp() * 1000)
            if (
                not rows
                or rows[0].open_time_ms != expected_start_ms
                or rows[-1].open_time_ms != expected_end_ms - interval_ms
            ):
                raise ValueError(
                    f"History segment {segment.market_symbol} does not completely cover its declared range."
                )
            sources.append(source)
            for row in rows:
                existing = combined.get(row.open_time_ms)
                if existing is not None and existing != row:
                    raise ValueError(
                        f"Overlapping history segments disagree for {asset.symbol} at {row.open_time_ms}."
                    )
                combined[row.open_time_ms] = row

        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        output: list[SpotCandle] = []
        unavailable: set[int] = set()
        previous_close: float | None = None
        expected_rows = max(0, (end_ms - start_ms) // interval_ms)
        self._start_progress_phase(f"Stitching {asset.symbol} history", total=expected_rows)
        progress_chunk = 0
        for open_time_ms in range(start_ms, end_ms, interval_ms):
            progress_chunk += 1
            if progress_chunk >= 10_000:
                self._advance_progress(progress_chunk)
                progress_chunk = 0
            row = combined.get(open_time_ms)
            if row is not None:
                output.append(row)
                previous_close = row.close
                continue
            if previous_close is None:
                raise ValueError(
                    f"Stitched history for {asset.symbol} does not cover the requested start."
                )
            unavailable.add(open_time_ms)
            output.append(
                SpotCandle(
                    open_time_ms=open_time_ms,
                    close_time_ms=open_time_ms + interval_ms - 1,
                    open=previous_close,
                    high=previous_close,
                    low=previous_close,
                    close=previous_close,
                    volume=0.0,
                )
            )
        self._advance_progress(progress_chunk)
        if not output or output[-1].open_time_ms != end_ms - interval_ms:
            raise ValueError(f"Stitched history for {asset.symbol} does not cover the requested end.")
        source = "cache" if sources and all(item == "cache" for item in sources) else "binance_spot_rest"
        return output, source, unavailable

    def load_candles(
        self,
        symbol: str,
        *,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> tuple[list[SpotCandle], str]:
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        cache_path = self.cache_dir / "klines" / (
            f"{symbol.upper()}-{interval}-{start_ms}-{end_ms}.csv"
        )
        covering_cache = self._covering_cache_path(
            symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not cache_path.exists() and covering_cache is not None:
            self._start_progress_phase(
                f"Reading {symbol.upper()} cache",
                total=self._cached_row_capacity(covering_cache, interval_ms=interval_milliseconds(interval)),
            )
            rows = [
                row
                for row in self._read_candles(covering_cache)
                if start_ms <= row.open_time_ms < end_ms
            ]
            interval_ms = interval_milliseconds(interval)
            if (
                rows
                and rows[0].open_time_ms == start_ms
                and rows[-1].open_time_ms == end_ms - interval_ms
            ):
                return rows, "cache"
        overlapping_cache = self._best_overlapping_cache_path(
            symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not cache_path.exists() and overlapping_cache is not None:
            interval_ms = interval_milliseconds(interval)
            self._start_progress_phase(
                f"Reading {symbol.upper()} cache",
                total=self._cached_row_capacity(
                    overlapping_cache,
                    interval_ms=interval_ms,
                ),
            )
            cached_rows = [
                row
                for row in self._read_candles(overlapping_cache)
                if start_ms <= row.open_time_ms < end_ms
            ]
            downloaded: list[SpotCandle] = []
            if cached_rows:
                if cached_rows[0].open_time_ms > start_ms:
                    downloaded.extend(
                        self._download_candles(
                            symbol.upper(),
                            interval=interval,
                            start_ms=start_ms,
                            end_ms=cached_rows[0].open_time_ms,
                        )
                    )
                tail_start_ms = cached_rows[-1].open_time_ms + interval_ms
                if tail_start_ms < end_ms:
                    downloaded.extend(
                        self._download_candles(
                            symbol.upper(),
                            interval=interval,
                            start_ms=tail_start_ms,
                            end_ms=end_ms,
                        )
                    )
                merged = {row.open_time_ms: row for row in (*cached_rows, *downloaded)}
                rows = [merged[key] for key in sorted(merged)]
                self._start_progress_phase(f"Writing {symbol.upper()} cache", total=len(rows))
                self._write_candles(cache_path, rows)
                return rows, "binance_spot_rest" if downloaded else "cache"
        if cache_path.exists():
            self._start_progress_phase(
                f"Reading {symbol.upper()} cache",
                total=self._cached_row_capacity(cache_path, interval_ms=interval_milliseconds(interval)),
            )
            rows = self._read_candles(cache_path)
            interval_ms = interval_milliseconds(interval)
            downloaded: list[SpotCandle] = []
            if not rows:
                downloaded = self._download_candles(
                    symbol.upper(),
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            else:
                if rows[0].open_time_ms > start_ms:
                    downloaded.extend(
                        self._download_candles(
                            symbol.upper(),
                            interval=interval,
                            start_ms=start_ms,
                            end_ms=rows[0].open_time_ms,
                        )
                    )
                tail_start_ms = rows[-1].open_time_ms + interval_ms
                if tail_start_ms < end_ms:
                    downloaded.extend(
                        self._download_candles(
                            symbol.upper(),
                            interval=interval,
                            start_ms=tail_start_ms,
                            end_ms=end_ms,
                        )
                    )
            if downloaded:
                merged = {row.open_time_ms: row for row in (*rows, *downloaded)}
                rows = [merged[key] for key in sorted(merged)]
                self._start_progress_phase(f"Writing {symbol.upper()} cache", total=len(rows))
                self._write_candles(cache_path, rows)
                return rows, "binance_spot_rest"
            return rows, "cache"
        rows = self._download_candles(
            symbol.upper(),
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_progress_phase(f"Writing {symbol.upper()} cache", total=len(rows))
        self._write_candles(cache_path, rows)
        return rows, "binance_spot_rest"

    @staticmethod
    def _cached_row_capacity(path: Path, *, interval_ms: int) -> int | None:
        bounds = path.stem.rsplit("-", 2)
        if len(bounds) != 3:
            return None
        try:
            start_ms = int(bounds[-2])
            end_ms = int(bounds[-1])
        except ValueError:
            return None
        return max(0, (end_ms - start_ms) // interval_ms)

    def _covering_cache_path(
        self,
        symbol: str,
        *,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> Path | None:
        directory = self.cache_dir / "klines"
        prefix = f"{symbol.upper()}-{interval}-"
        candidates: list[tuple[int, Path]] = []
        for path in directory.glob(f"{prefix}*.csv"):
            bounds = path.stem[len(prefix):].split("-", 1)
            if len(bounds) != 2:
                continue
            try:
                cached_start, cached_end = (int(value) for value in bounds)
            except ValueError:
                continue
            if cached_start <= start_ms and cached_end >= end_ms:
                candidates.append((cached_end - cached_start, path))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _best_overlapping_cache_path(
        self,
        symbol: str,
        *,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> Path | None:
        directory = self.cache_dir / "klines"
        prefix = f"{symbol.upper()}-{interval}-"
        candidates: list[tuple[int, Path]] = []
        for path in directory.glob(f"{prefix}*.csv"):
            bounds = path.stem[len(prefix):].split("-", 1)
            if len(bounds) != 2:
                continue
            try:
                cached_start, cached_end = (int(value) for value in bounds)
            except ValueError:
                continue
            overlap = min(cached_end, end_ms) - max(cached_start, start_ms)
            if overlap > 0:
                candidates.append((overlap, path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def load_symbol_info(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.upper()
        cache_path = self.cache_dir / "exchange_info" / f"{normalized}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                return payload
        payload = self._request_json("/api/v3/exchangeInfo", {"symbol": normalized})
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(symbols, list) or not symbols or not isinstance(symbols[0], dict):
            raise ValueError(f"Binance Spot exchange info is unavailable for {normalized}.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as file_obj:
            json.dump(symbols[0], file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
        return symbols[0]

    def _download_candles(
        self,
        symbol: str,
        *,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[SpotCandle]:
        output: list[SpotCandle] = []
        cursor = start_ms
        interval_ms = interval_milliseconds(interval)
        self._start_progress_phase(
            f"Downloading {symbol.upper()}",
            total=max(0, (end_ms - start_ms) // interval_ms),
        )
        while cursor < end_ms:
            payload = self._request_json(
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list) or not payload:
                break
            page = [self._parse_kline(item) for item in payload]
            accepted = [candle for candle in page if candle.open_time_ms < end_ms]
            output.extend(accepted)
            self._advance_progress(len(accepted))
            next_cursor = int(page[-1].open_time_ms) + interval_ms
            if next_cursor <= cursor:
                raise RuntimeError(f"Binance Spot candle pagination stalled for {symbol}.")
            cursor = next_cursor
            if len(page) < 1000:
                break
        unique = {row.open_time_ms: row for row in output}
        return [unique[key] for key in sorted(unique)]

    def _request_json(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "Scrooge-Spot-Research/1.0"})
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except (TimeoutError, urllib.error.URLError):
                if attempt == REQUEST_ATTEMPTS - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError("Binance Spot request retry loop ended unexpectedly.")

    @staticmethod
    def _parse_kline(raw: Any) -> SpotCandle:
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            raise ValueError("Binance Spot candle row is malformed.")
        candle = SpotCandle(
            open_time_ms=int(raw[0]),
            close_time_ms=int(raw[6]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
        )
        if candle.open <= 0 or candle.close <= 0 or candle.low <= 0:
            raise ValueError("Binance Spot candle prices must be positive.")
        if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
            raise ValueError("Binance Spot candle OHLC values are inconsistent.")
        return candle

    def _validate_candles(self, rows: list[SpotCandle], *, interval_ms: int, symbol: str) -> None:
        if not rows:
            raise ValueError(f"No historical Spot candles are available for {symbol}.")
        progress_chunk = 1
        for previous, current in zip(rows, rows[1:]):
            if current.open_time_ms - previous.open_time_ms != interval_ms:
                previous_time = datetime.fromtimestamp(previous.open_time_ms / 1000, tz=UTC).isoformat()
                current_time = datetime.fromtimestamp(current.open_time_ms / 1000, tz=UTC).isoformat()
                raise ValueError(
                    f"Missing {symbol} candle between {previous_time} and {current_time}; replay aborted."
                )
            progress_chunk += 1
            if progress_chunk >= 10_000:
                self._advance_progress(progress_chunk)
                progress_chunk = 0
        self._advance_progress(progress_chunk)

    def _read_candles(self, path: Path) -> list[SpotCandle]:
        rows: list[SpotCandle] = []
        progress_chunk = 0
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                rows.append(SpotCandle(
                    open_time_ms=int(row["open_time_ms"]),
                    close_time_ms=int(row["close_time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
                progress_chunk += 1
                if progress_chunk >= 10_000:
                    self._advance_progress(progress_chunk)
                    progress_chunk = 0
        self._advance_progress(progress_chunk)
        return rows

    def _write_candles(self, path: Path, rows: list[SpotCandle]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"],
            )
            writer.writeheader()
            progress_chunk = 0
            for candle in rows:
                writer.writerow(
                    {
                        "open_time_ms": candle.open_time_ms,
                        "close_time_ms": candle.close_time_ms,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
                progress_chunk += 1
                if progress_chunk >= 10_000:
                    self._advance_progress(progress_chunk)
                    progress_chunk = 0
            self._advance_progress(progress_chunk)
