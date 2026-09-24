from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

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

    def load(self, scenario: SpotBacktestScenario) -> SpotHistoricalDataset:
        interval_ms = interval_milliseconds(scenario.interval)
        warmup_start = scenario.start - timedelta(
            milliseconds=interval_ms * scenario.warmup_candles
        )
        candles: dict[str, tuple[SpotCandle, ...]] = {}
        symbol_info: dict[str, dict[str, Any]] = {}
        unavailable_open_times: dict[str, frozenset[int]] = {}
        source_parts: list[str] = []
        for asset in scenario.assets:
            if asset.history_segments:
                rows, source, unavailable = self._load_stitched_candles(
                    asset,
                    interval=scenario.interval,
                    start=warmup_start,
                    end=scenario.end,
                )
                unavailable_open_times[asset.symbol] = frozenset(unavailable)
            else:
                rows, source = self.load_candles(
                    asset.market_symbol,
                    interval=scenario.interval,
                    start=warmup_start,
                    end=scenario.end,
                )
            self._validate_candles(rows, interval_ms=interval_ms, symbol=asset.market_symbol)
            candles[asset.symbol] = tuple(rows)
            symbol_info[asset.symbol] = asset.symbol_info or self.load_symbol_info(asset.market_symbol)
            source_parts.append(source)
        return SpotHistoricalDataset(
            candles=candles,
            symbol_info=symbol_info,
            interval=scenario.interval,
            interval_ms=interval_ms,
            source=(
                "cache"
                if all(item == "cache" for item in source_parts)
                else "binance_spot_rest"
            ),
            unavailable_open_times=unavailable_open_times,
        )

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
        for open_time_ms in range(start_ms, end_ms, interval_ms):
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
        if cache_path.exists():
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
        self._write_candles(cache_path, rows)
        return rows, "binance_spot_rest"

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
            output.extend(candle for candle in page if candle.open_time_ms < end_ms)
            next_cursor = int(page[-1].open_time_ms) + interval_milliseconds(interval)
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

    @staticmethod
    def _validate_candles(rows: list[SpotCandle], *, interval_ms: int, symbol: str) -> None:
        if not rows:
            raise ValueError(f"No historical Spot candles are available for {symbol}.")
        for previous, current in zip(rows, rows[1:]):
            if current.open_time_ms - previous.open_time_ms != interval_ms:
                previous_time = datetime.fromtimestamp(previous.open_time_ms / 1000, tz=UTC).isoformat()
                current_time = datetime.fromtimestamp(current.open_time_ms / 1000, tz=UTC).isoformat()
                raise ValueError(
                    f"Missing {symbol} candle between {previous_time} and {current_time}; replay aborted."
                )

    @staticmethod
    def _read_candles(path: Path) -> list[SpotCandle]:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            return [
                SpotCandle(
                    open_time_ms=int(row["open_time_ms"]),
                    close_time_ms=int(row["close_time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for row in csv.DictReader(file_obj)
            ]

    @staticmethod
    def _write_candles(path: Path, rows: list[SpotCandle]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"],
            )
            writer.writeheader()
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
