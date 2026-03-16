from __future__ import annotations

import csv
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.config_service import load_config
from services.history_service import load_balance_history, load_trade_history
from services.state_service import load_state


BINANCE_FUTURES_KLINES_URL = os.getenv("SCROOGE_CHART_KLINES_URL", "https://fapi.binance.com/fapi/v1/klines")
CHART_MAX_CANDLES = int(os.getenv("SCROOGE_CHART_MAX_CANDLES", "1500"))
CHART_DATASET_MAX_CANDLES = int(os.getenv("SCROOGE_CHART_DATASET_MAX_CANDLES", "10000"))
CHART_HTTP_TIMEOUT_SECONDS = float(os.getenv("SCROOGE_CHART_TIMEOUT_SECONDS", "10"))
CHART_SOURCE_DEFAULT = os.getenv("SCROOGE_CHART_SOURCE", "auto").strip().lower()
CHART_DATASET_PATH = os.getenv("SCROOGE_CHART_DATASET_PATH", "").strip()

_INTERVAL_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}
_PERIOD_UNIT_TO_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}
_PERIOD_RE = re.compile(r"^(?P<count>\d+)(?P<unit>[mhdwMHDW])$")
_INTERVAL_SEQUENCE: list[tuple[str, int]] = sorted(_INTERVAL_TO_MS.items(), key=lambda item: item[1])
STRATEGY_RSI_PERIOD = 11
STRATEGY_EMA_PERIOD = 50
STRATEGY_BB_PERIOD = 20
STRATEGY_BB_STD_MULT = 2.0
DEFAULT_STRATEGY_MEDIUM_INTERVAL = "1h"
DEFAULT_STRATEGY_BIG_INTERVAL = "4h"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _iso_from_ts_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _parse_time_to_ms(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:  # milliseconds
            return int(raw)
        if raw > 100_000_000:  # seconds
            return int(raw * 1000)
        return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.isdigit():
            numeric = int(raw)
            if numeric > 10_000_000_000:
                return numeric
            if numeric > 100_000_000:
                return numeric * 1000

        normalized = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            pass

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt).replace(tzinfo=UTC)
                return int(parsed.timestamp() * 1000)
            except ValueError:
                continue

    return None


def _parse_interval_to_ms(interval: str) -> int:
    key = interval.strip()
    if key not in _INTERVAL_TO_MS:
        supported = ", ".join(sorted(_INTERVAL_TO_MS))
        raise ValueError(f"Unsupported interval: {interval}. Supported: {supported}")
    return _INTERVAL_TO_MS[key]


def _parse_period_to_ms(period: str) -> int:
    cleaned = period.strip()
    match = _PERIOD_RE.fullmatch(cleaned)
    if not match:
        raise ValueError("Invalid period format. Use values like 1d, 6h, 30m, 2w.")
    count = int(match.group("count"))
    if count < 1:
        raise ValueError("period count must be >= 1")
    unit = match.group("unit").lower()
    return count * _PERIOD_UNIT_TO_MS[unit]


def _resolve_chart_interval(period_ms: int, requested_interval: str, max_candles: int) -> tuple[str, list[str]]:
    warnings: list[str] = []
    requested_ms = _parse_interval_to_ms(requested_interval)
    max_candles = max(1, max_candles)
    min_interval_ms = math.ceil(period_ms / max_candles)

    if requested_ms >= min_interval_ms:
        return requested_interval, warnings

    resolved_interval = _INTERVAL_SEQUENCE[-1][0]
    for interval_name, interval_ms in _INTERVAL_SEQUENCE:
        if interval_ms >= min_interval_ms and interval_ms >= requested_ms:
            resolved_interval = interval_name
            break

    warnings.append(
        "Requested period/interval exceeds candle limit. "
        f"Interval adjusted from {requested_interval} to {resolved_interval} "
        f"to fit max {max_candles} candles."
    )
    return resolved_interval, warnings


def _parse_chart_source(source: str | None) -> str:
    raw = (source or CHART_SOURCE_DEFAULT or "auto").strip().lower()
    if raw not in {"auto", "binance", "dataset"}:
        raise ValueError("Unsupported chart source. Use auto, dataset, or binance.")
    return raw


def _resolve_dataset_file_path() -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    candidates: list[Path] = []

    if CHART_DATASET_PATH:
        configured = Path(CHART_DATASET_PATH).expanduser()
        if configured.is_dir():
            candidates.extend(sorted(configured.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True))
            if not candidates:
                warnings.append(f"No CSV files found in dataset directory: {configured}")
        else:
            candidates.append(configured)

    default_runtime = Path("/runtime/chart_dataset.csv")
    default_local = _project_root() / "runtime" / "chart_dataset.csv"
    default_data_dir = _project_root() / "data"
    candidates.extend([default_runtime, default_local])
    if default_data_dir.exists():
        candidates.extend(sorted(default_data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate, warnings

    warnings.append("Dataset source requested, but no dataset CSV file was found.")
    return None, warnings


def _read_dataset_rows(dataset_path: Path, symbol: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if not reader.fieldnames:
                warnings.append(f"Dataset CSV has no header: {dataset_path}")
                return [], warnings

            open_time_key = None
            for key in ("open_time", "time", "timestamp"):
                if key in reader.fieldnames:
                    open_time_key = key
                    break
            if open_time_key is None:
                warnings.append(f"Dataset CSV missing time column (open_time/time/timestamp): {dataset_path}")
                return [], warnings

            symbol_key = "symbol" if "symbol" in reader.fieldnames else None

            for raw in reader:
                if symbol_key is not None:
                    csv_symbol = str(raw.get(symbol_key, "")).strip().upper()
                    if csv_symbol and csv_symbol != symbol:
                        continue

                ts_ms = _parse_time_to_ms(raw.get(open_time_key))
                open_price = _to_float(raw.get("open"))
                high_price = _to_float(raw.get("high"))
                low_price = _to_float(raw.get("low"))
                close_price = _to_float(raw.get("close"))
                volume = _to_float(raw.get("volume"))
                balance = _to_float(raw.get("balance"))
                ema = _to_float(raw.get("EMA") if "EMA" in raw else raw.get("ema"))
                rsi = _to_float(raw.get("RSI") if "RSI" in raw else raw.get("rsi"))
                bbl = _to_float(raw.get("BBL") if "BBL" in raw else raw.get("bbl"))
                bbm = _to_float(raw.get("BBM") if "BBM" in raw else raw.get("bbm"))
                bbu = _to_float(raw.get("BBU") if "BBU" in raw else raw.get("bbu"))
                atr = _to_float(raw.get("ATR") if "ATR" in raw else raw.get("atr"))
                if ts_ms is None or None in (open_price, high_price, low_price, close_price, volume):
                    continue
                rows.append(
                    {
                        "ts_ms": ts_ms,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "balance": balance,
                        "ema": ema,
                        "rsi": rsi,
                        "bbl": bbl,
                        "bbm": bbm,
                        "bbu": bbu,
                        "atr": atr,
                    }
                )
    except OSError as exc:
        warnings.append(f"Failed to read dataset file {dataset_path}: {exc}")
        return [], warnings

    if not rows:
        warnings.append(f"Dataset file has no valid OHLC rows: {dataset_path}")
        return [], warnings

    rows.sort(key=lambda row: int(row["ts_ms"]))
    deduped: list[dict[str, Any]] = []
    for row in rows:
        if deduped and int(deduped[-1]["ts_ms"]) == int(row["ts_ms"]):
            deduped[-1] = row
            continue
        deduped.append(row)

    indicator_keys = ("ema", "rsi", "bbl", "bbm", "bbu", "atr")
    has_indicator_values = any(
        _to_float(row.get(key)) is not None
        for row in deduped
        for key in indicator_keys
    )
    if has_indicator_values:
        trim_index = 0
        while trim_index < len(deduped):
            row = deduped[trim_index]
            if all(_to_float(row.get(key)) is not None for key in indicator_keys):
                break
            trim_index += 1
        if 0 < trim_index < len(deduped):
            deduped = deduped[trim_index:]

    return deduped, warnings


def _aggregate_rows(rows: list[dict[str, Any]], interval_ms: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    aggregated: list[dict[str, Any]] = []
    bucket_start = (int(rows[0]["ts_ms"]) // interval_ms) * interval_ms
    current = {
        "ts_ms": bucket_start,
        "open": float(rows[0]["open"]),
        "high": float(rows[0]["high"]),
        "low": float(rows[0]["low"]),
        "close": float(rows[0]["close"]),
        "volume": float(rows[0]["volume"]),
        "balance": _to_float(rows[0].get("balance")),
        "ema": _to_float(rows[0].get("ema")),
        "rsi": _to_float(rows[0].get("rsi")),
        "bbl": _to_float(rows[0].get("bbl")),
        "bbm": _to_float(rows[0].get("bbm")),
        "bbu": _to_float(rows[0].get("bbu")),
        "atr": _to_float(rows[0].get("atr")),
    }

    for row in rows[1:]:
        ts_ms = int(row["ts_ms"])
        bucket = (ts_ms // interval_ms) * interval_ms
        if bucket != int(current["ts_ms"]):
            aggregated.append(current)
            current = {
                "ts_ms": bucket,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "balance": _to_float(row.get("balance")),
                "ema": _to_float(row.get("ema")),
                "rsi": _to_float(row.get("rsi")),
                "bbl": _to_float(row.get("bbl")),
                "bbm": _to_float(row.get("bbm")),
                "bbu": _to_float(row.get("bbu")),
                "atr": _to_float(row.get("atr")),
            }
            continue

        current["high"] = max(float(current["high"]), float(row["high"]))
        current["low"] = min(float(current["low"]), float(row["low"]))
        current["close"] = float(row["close"])
        current["volume"] = float(current["volume"]) + float(row["volume"])
        row_balance = _to_float(row.get("balance"))
        if row_balance is not None:
            current["balance"] = row_balance
        current["ema"] = _to_float(row.get("ema"))
        current["rsi"] = _to_float(row.get("rsi"))
        current["bbl"] = _to_float(row.get("bbl"))
        current["bbm"] = _to_float(row.get("bbm"))
        current["bbu"] = _to_float(row.get("bbu"))
        current["atr"] = _to_float(row.get("atr"))

    aggregated.append(current)
    return aggregated


def _rows_to_candles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    for row in rows:
        candle = {
            "time": _iso_from_ts_ms(int(row["ts_ms"])),
            "ts_ms": int(row["ts_ms"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        balance = _to_float(row.get("balance"))
        if balance is not None:
            candle["balance"] = balance
        candle["ema"] = _to_float(row.get("ema"))
        candle["rsi"] = _to_float(row.get("rsi"))
        candle["bbl"] = _to_float(row.get("bbl"))
        candle["bbm"] = _to_float(row.get("bbm"))
        candle["bbu"] = _to_float(row.get("bbu"))
        candle["atr"] = _to_float(row.get("atr"))
        candles.append(candle)
    return candles


def _fetch_candles_from_dataset(
    symbol: str,
    period_ms: int,
    interval: str,
    end_ms: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    dataset_path, path_warnings = _resolve_dataset_file_path()
    warnings.extend(path_warnings)
    if dataset_path is None:
        return [], warnings

    rows, row_warnings = _read_dataset_rows(dataset_path=dataset_path, symbol=symbol)
    warnings.extend(row_warnings)
    if not rows:
        return [], warnings

    interval_ms = _parse_interval_to_ms(interval)
    if interval_ms > _INTERVAL_TO_MS["1m"]:
        rows = _aggregate_rows(rows=rows, interval_ms=interval_ms)

    last_ts_ms = int(rows[-1]["ts_ms"])
    effective_end_ms = end_ms if end_ms is not None else last_ts_ms
    effective_start_ms = max(0, effective_end_ms - period_ms)
    ranged = [row for row in rows if effective_start_ms <= int(row["ts_ms"]) <= effective_end_ms]

    if not ranged:
        warnings.append(
            "Dataset source has no candles in requested range. "
            f"File: {dataset_path}"
        )
        return [], warnings

    limit = max(1, CHART_DATASET_MAX_CANDLES)
    if len(ranged) > limit:
        ranged_count = len(ranged)
        step = math.ceil(len(ranged) / limit)
        sampled = ranged[::step]
        if sampled[-1]["ts_ms"] != ranged[-1]["ts_ms"]:
            sampled.append(ranged[-1])
        ranged = sampled
        warnings.append(
            f"Dataset candles downsampled from {ranged_count} to {len(ranged)} for performance "
            f"(SCROOGE_CHART_DATASET_MAX_CANDLES={CHART_DATASET_MAX_CANDLES})."
        )

    return _rows_to_candles(ranged), warnings


def _fetch_candles_from_binance(symbol: str, period_ms: int, interval: str, end_ms: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    interval_ms = _parse_interval_to_ms(interval)
    requested = max(20, math.ceil(period_ms / interval_ms))
    limit = min(CHART_MAX_CANDLES, requested)

    if limit < requested:
        warnings.append(
            f"Requested {requested} candles, truncated to {limit} due to CHART_MAX_CANDLES={CHART_MAX_CANDLES}."
        )

    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_ms is not None:
        start_ms = max(0, end_ms - period_ms)
        params["startTime"] = int(start_ms)
        params["endTime"] = int(end_ms)

    query = urllib.parse.urlencode(params)
    url = f"{BINANCE_FUTURES_KLINES_URL}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=CHART_HTTP_TIMEOUT_SECONDS) as response:
            raw_payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        warnings.append(f"Failed to fetch candles from Binance: {exc}")
        return [], warnings
    except TimeoutError:
        warnings.append("Timed out while fetching candles from Binance.")
        return [], warnings

    try:
        rows = json.loads(raw_payload)
    except json.JSONDecodeError:
        warnings.append("Malformed Binance candles response.")
        return [], warnings

    if not isinstance(rows, list):
        warnings.append("Unexpected Binance candles payload shape.")
        return [], warnings

    candles: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        ts_ms = _parse_time_to_ms(row[0])
        open_price = _to_float(row[1])
        high_price = _to_float(row[2])
        low_price = _to_float(row[3])
        close_price = _to_float(row[4])
        volume = _to_float(row[5])
        if ts_ms is None or None in (open_price, high_price, low_price, close_price, volume):
            continue
        candles.append(
            {
                "time": _iso_from_ts_ms(ts_ms),
                "ts_ms": ts_ms,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

    if not candles:
        warnings.append("No valid candles returned from Binance.")
    return candles, warnings


def _fetch_candles(
    source: str,
    symbol: str,
    period_ms: int,
    interval: str,
    end_ms: int | None,
) -> tuple[list[dict[str, Any]], list[str], str, str]:
    if source == "dataset":
        candles, warnings = _fetch_candles_from_dataset(symbol=symbol, period_ms=period_ms, interval=interval, end_ms=end_ms)
        return candles, warnings, "dataset", interval

    if source == "binance":
        candles, warnings = _fetch_candles_from_binance(symbol=symbol, period_ms=period_ms, interval=interval, end_ms=end_ms)
        return candles, warnings, "binance", interval

    # auto
    dataset_candles, dataset_warnings = _fetch_candles_from_dataset(
        symbol=symbol,
        period_ms=period_ms,
        interval=interval,
        end_ms=end_ms,
    )
    if dataset_candles:
        return dataset_candles, dataset_warnings, "dataset", interval

    warnings = list(dataset_warnings)
    warnings.append("Falling back to Binance candles.")
    fallback_interval, fallback_warnings = _resolve_chart_interval(
        period_ms=period_ms,
        requested_interval=interval,
        max_candles=CHART_MAX_CANDLES,
    )
    warnings.extend(fallback_warnings)
    binance_candles, binance_warnings = _fetch_candles_from_binance(
        symbol=symbol,
        period_ms=period_ms,
        interval=fallback_interval,
        end_ms=end_ms,
    )
    warnings.extend(binance_warnings)
    return binance_candles, warnings, "binance", fallback_interval


def _ema(values: list[float], period: int) -> list[float | None]:
    if not values:
        return []
    if period < 1:
        return [None for _ in values]
    output: list[float | None] = [None for _ in values]
    if len(values) < period:
        return output

    seed = sum(values[:period]) / period
    output[period - 1] = seed
    alpha = 2 / (period + 1)
    ema_value = seed
    for idx in range(period, len(values)):
        ema_value = (values[idx] - ema_value) * alpha + ema_value
        output[idx] = ema_value
    return output


def _bollinger(values: list[float], period: int, std_mult: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
    upper: list[float | None] = [None for _ in values]
    middle: list[float | None] = [None for _ in values]
    lower: list[float | None] = [None for _ in values]
    if period < 1 or len(values) < period:
        return upper, middle, lower

    for idx in range(period - 1, len(values)):
        window = values[idx - period + 1 : idx + 1]
        mean = sum(window) / period
        variance = sum((point - mean) ** 2 for point in window) / period
        std = math.sqrt(variance)
        middle[idx] = mean
        upper[idx] = mean + std_mult * std
        lower[idx] = mean - std_mult * std
    return upper, middle, lower


def _rsi(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None for _ in values]
    if period < 1 or len(values) <= period:
        return output

    gains = [0.0]
    losses = [0.0]
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    def compute_rsi(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    output[period] = compute_rsi(avg_gain, avg_loss)
    for idx in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[idx]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[idx]) / period
        output[idx] = compute_rsi(avg_gain, avg_loss)
    return output


def _points_from_series(times: list[str], values: list[float | None]) -> list[dict[str, Any]]:
    return [{"time": times[idx], "value": values[idx]} for idx in range(min(len(times), len(values)))]


def _normalize_strategy_interval(value: Any, default: str) -> str:
    raw = str(value or default).strip()
    if raw in _INTERVAL_TO_MS:
        return raw
    return default


def _build_indicator_spec(config: dict[str, Any]) -> dict[str, Any]:
    intervals = config.get("intervals")
    interval_map = intervals if isinstance(intervals, dict) else {}
    medium_interval = _normalize_strategy_interval(interval_map.get("medium"), DEFAULT_STRATEGY_MEDIUM_INTERVAL)
    big_interval = _normalize_strategy_interval(interval_map.get("big"), DEFAULT_STRATEGY_BIG_INTERVAL)
    return {
        "ema": {
            "period": STRATEGY_EMA_PERIOD,
            "interval": big_interval,
        },
        "rsi": {
            "period": STRATEGY_RSI_PERIOD,
            "interval": big_interval,
        },
        "bollinger": {
            "period": STRATEGY_BB_PERIOD,
            "std_mult": STRATEGY_BB_STD_MULT,
            "interval": medium_interval,
        },
    }


def _align_series_to_times(
    source_times_ms: list[int],
    source_values: list[float | None],
    target_times_ms: list[int],
) -> list[float | None]:
    aligned: list[float | None] = []
    source_index = 0
    current_value: float | None = None
    source_count = min(len(source_times_ms), len(source_values))
    for target_ts_ms in target_times_ms:
        while source_index < source_count and source_times_ms[source_index] <= target_ts_ms:
            current_value = source_values[source_index]
            source_index += 1
        aligned.append(current_value)
    return aligned


def _build_indicators(candles: list[dict[str, Any]], indicator_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    closes = [float(candle["close"]) for candle in candles]
    times = [str(candle["time"]) for candle in candles]
    spec = indicator_spec or _build_indicator_spec({})
    ema_period = int(spec["ema"]["period"])
    rsi_period = int(spec["rsi"]["period"])
    bollinger_period = int(spec["bollinger"]["period"])
    bollinger_std_mult = float(spec["bollinger"]["std_mult"])

    ema_values = _ema(closes, period=ema_period)
    bb_upper, bb_middle, bb_lower = _bollinger(closes, period=bollinger_period, std_mult=bollinger_std_mult)
    rsi_values = _rsi(closes, period=rsi_period)

    return {
        "ema": _points_from_series(times, ema_values),
        "bollinger": {
            "upper": _points_from_series(times, bb_upper),
            "middle": _points_from_series(times, bb_middle),
            "lower": _points_from_series(times, bb_lower),
        },
        "rsi": _points_from_series(times, rsi_values),
    }


def _build_indicators_from_candle_fields(candles: list[dict[str, Any]]) -> dict[str, Any]:
    if not candles:
        return {}

    times = [str(candle["time"]) for candle in candles]
    ema_values = [_to_float(candle.get("ema")) for candle in candles]
    rsi_values = [_to_float(candle.get("rsi")) for candle in candles]
    bb_upper = [_to_float(candle.get("bbu")) for candle in candles]
    bb_middle = [_to_float(candle.get("bbm")) for candle in candles]
    bb_lower = [_to_float(candle.get("bbl")) for candle in candles]

    has_dataset_indicators = any(
        value is not None
        for series in (ema_values, rsi_values, bb_upper, bb_middle, bb_lower)
        for value in series
    )
    if not has_dataset_indicators:
        return {}

    return {
        "ema": _points_from_series(times, ema_values),
        "bollinger": {
            "upper": _points_from_series(times, bb_upper),
            "middle": _points_from_series(times, bb_middle),
            "lower": _points_from_series(times, bb_lower),
        },
        "rsi": _points_from_series(times, rsi_values),
    }


def _build_strategy_fallback_indicators(
    symbol: str,
    config: dict[str, Any],
    target_candles: list[dict[str, Any]],
    period_ms: int,
    end_ms: int | None,
) -> tuple[dict[str, Any], list[str]]:
    if not target_candles:
        return {}, []

    indicator_spec = _build_indicator_spec(config)
    target_times = [str(candle["time"]) for candle in target_candles]
    target_times_ms = [int(candle["ts_ms"]) for candle in target_candles if candle.get("ts_ms") is not None]
    if not target_times_ms:
        return {}, []

    warnings: list[str] = []
    indicators: dict[str, Any] = {}
    effective_end_ms = end_ms if end_ms is not None else target_times_ms[-1]

    ema_spec = indicator_spec["ema"]
    rsi_spec = indicator_spec["rsi"]
    bollinger_spec = indicator_spec["bollinger"]

    big_interval = str(ema_spec["interval"])
    big_interval_ms = _parse_interval_to_ms(big_interval)
    big_warmup_candles = max(int(ema_spec["period"]), int(rsi_spec["period"]) + 1)
    big_candles, big_warnings = _fetch_candles_from_binance(
        symbol=symbol,
        period_ms=period_ms + big_warmup_candles * big_interval_ms,
        interval=big_interval,
        end_ms=effective_end_ms,
    )
    warnings.extend(big_warnings)
    if big_candles:
        big_closes = [float(candle["close"]) for candle in big_candles]
        big_times_ms = [int(candle["ts_ms"]) for candle in big_candles]
        ema_values = _align_series_to_times(
            source_times_ms=big_times_ms,
            source_values=_ema(big_closes, period=int(ema_spec["period"])),
            target_times_ms=target_times_ms,
        )
        rsi_values = _align_series_to_times(
            source_times_ms=big_times_ms,
            source_values=_rsi(big_closes, period=int(rsi_spec["period"])),
            target_times_ms=target_times_ms,
        )
        if any(value is not None for value in ema_values):
            indicators["ema"] = _points_from_series(target_times, ema_values)
        if any(value is not None for value in rsi_values):
            indicators["rsi"] = _points_from_series(target_times, rsi_values)

    medium_interval = str(bollinger_spec["interval"])
    medium_interval_ms = _parse_interval_to_ms(medium_interval)
    medium_warmup_candles = int(bollinger_spec["period"])
    medium_candles, medium_warnings = _fetch_candles_from_binance(
        symbol=symbol,
        period_ms=period_ms + medium_warmup_candles * medium_interval_ms,
        interval=medium_interval,
        end_ms=effective_end_ms,
    )
    warnings.extend(medium_warnings)
    if medium_candles:
        medium_closes = [float(candle["close"]) for candle in medium_candles]
        medium_times_ms = [int(candle["ts_ms"]) for candle in medium_candles]
        bb_upper_values, bb_middle_values, bb_lower_values = _bollinger(
            medium_closes,
            period=int(bollinger_spec["period"]),
            std_mult=float(bollinger_spec["std_mult"]),
        )
        aligned_upper = _align_series_to_times(
            source_times_ms=medium_times_ms,
            source_values=bb_upper_values,
            target_times_ms=target_times_ms,
        )
        aligned_middle = _align_series_to_times(
            source_times_ms=medium_times_ms,
            source_values=bb_middle_values,
            target_times_ms=target_times_ms,
        )
        aligned_lower = _align_series_to_times(
            source_times_ms=medium_times_ms,
            source_values=bb_lower_values,
            target_times_ms=target_times_ms,
        )
        if any(
            value is not None
            for series in (aligned_upper, aligned_middle, aligned_lower)
            for value in series
        ):
            indicators["bollinger"] = {
                "upper": _points_from_series(target_times, aligned_upper),
                "middle": _points_from_series(target_times, aligned_middle),
                "lower": _points_from_series(target_times, aligned_lower),
            }

    return indicators, warnings


def _normalize_open_position(position: Any) -> dict[str, Any] | None:
    if not isinstance(position, dict):
        return None

    entry = _to_float(position.get("entry"))
    sl = _to_float(position.get("sl"))
    tp = _to_float(position.get("tp"))
    trail_price = _to_float(position.get("trail_price"))
    liq_price = _to_float(position.get("liq_price"))
    size = _to_float(position.get("size"))
    time_ms = _parse_time_to_ms(position.get("time"))

    return {
        "side": position.get("side"),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "trail_price": trail_price,
        "liq_price": liq_price,
        "size": size,
        "time": _iso_from_ts_ms(time_ms) if time_ms is not None else None,
        "trail_active": bool(position.get("trail_active", False)),
    }


def _build_current_levels(open_position: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not open_position:
        return []
    levels: list[dict[str, Any]] = []
    sl = _to_float(open_position.get("sl"))
    tp = _to_float(open_position.get("tp"))
    trail_price = _to_float(open_position.get("trail_price"))
    trail_active = bool(open_position.get("trail_active", False))
    if sl is not None:
        levels.append({"type": "sl", "price": sl})
    if trail_active:
        active_trail = trail_price if trail_price is not None else tp
        if active_trail is not None:
            levels.append({"type": "trail", "price": active_trail})
    elif tp is not None:
        levels.append({"type": "tp", "price": tp})
    return levels


def _build_rsi_levels(config: dict[str, Any]) -> dict[str, float | None]:
    params = config.get("params")
    params_map = params if isinstance(params, dict) else {}
    return {
        "long_tp": _to_float(params_map.get("rsi_long_tp_threshold")),
        "short_tp": _to_float(params_map.get("rsi_short_tp_threshold")),
    }


def _build_markers(state: dict[str, Any], range_start_ms: int | None, range_end_ms: int | None) -> dict[str, list[dict[str, Any]]]:
    markers = {
        "entries": [],
        "exits": [],
        "stop_loss": [],
        "take_profit": [],
        "liquidation": [],
    }

    history = state.get("trade_history")
    if not isinstance(history, list):
        return markers

    for item in history:
        if not isinstance(item, dict):
            continue

        entry_time_ms = _parse_time_to_ms(item.get("time"))
        exit_time_ms = _parse_time_to_ms(item.get("exit_time")) or entry_time_ms

        def in_range(ts_ms: int | None) -> bool:
            if ts_ms is None:
                return False
            if range_start_ms is not None and ts_ms < range_start_ms:
                return False
            if range_end_ms is not None and ts_ms > range_end_ms:
                return False
            return True

        side = str(item.get("side", "")).lower()
        entry_price = _to_float(item.get("entry"))
        exit_price = _to_float(item.get("exit"))
        reason = str(item.get("exit_reason", "")).lower()

        if entry_price is not None and in_range(entry_time_ms):
            markers["entries"].append(
                {
                    "time": _iso_from_ts_ms(entry_time_ms),
                    "price": entry_price,
                    "side": side,
                }
            )

        if exit_price is not None and in_range(exit_time_ms):
            exit_marker = {
                "time": _iso_from_ts_ms(exit_time_ms),
                "price": exit_price,
                "side": side,
                "reason": reason,
                "net_pnl": _to_float(item.get("net_pnl")),
            }
            markers["exits"].append(exit_marker)
            if reason == "stop_loss":
                markers["stop_loss"].append(exit_marker)
            elif reason == "take_profit":
                markers["take_profit"].append(exit_marker)
            elif reason == "liquidation":
                markers["liquidation"].append(exit_marker)

    return markers


def _build_equity_curve_from_candle_balance(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for candle in candles:
        balance = _to_float(candle.get("balance"))
        time_value = candle.get("time")
        if balance is None or time_value is None:
            continue
        points.append({"time": str(time_value), "balance": balance})
    return points


def _build_equity_curve_from_balance_history(state: dict[str, Any], candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = state.get("balance_history")
    if not isinstance(history, list):
        return []

    balances: list[float] = []
    for item in history:
        value = _to_float(item)
        if value is not None:
            balances.append(value)

    if not balances:
        return []

    candle_times = [str(candle.get("time")) for candle in candles if candle.get("time")]
    points: list[dict[str, Any]] = []

    if len(candle_times) >= len(balances):
        x_values = candle_times[-len(balances) :]
    else:
        x_values = [str(idx + 1) for idx in range(len(balances))]

    for idx, balance in enumerate(balances):
        points.append({"time": x_values[idx], "balance": balance})
    return points


def _resolve_initial_balance(state: dict[str, Any], config: dict[str, Any]) -> float | None:
    balance_history = state.get("balance_history")
    if isinstance(balance_history, list) and balance_history:
        first_balance = _to_float(balance_history[0])
        if first_balance is not None:
            return first_balance

    configured = _to_float(config.get("initial_balance"))
    if configured is not None:
        return configured

    current = _to_float(state.get("balance"))
    if current is not None:
        return current

    return None


def _build_equity_curve(
    state: dict[str, Any],
    candles: list[dict[str, Any]],
    config: dict[str, Any],
    range_start_ms: int | None,
    range_end_ms: int | None,
) -> list[dict[str, Any]]:
    history = state.get("trade_history")
    if not isinstance(history, list) or not history:
        return _build_equity_curve_from_balance_history(state=state, candles=candles)

    initial_balance = _resolve_initial_balance(state=state, config=config)
    if initial_balance is None:
        return _build_equity_curve_from_balance_history(state=state, candles=candles)

    trade_points: list[tuple[int, float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        ts_ms = _parse_time_to_ms(item.get("exit_time")) or _parse_time_to_ms(item.get("time"))
        pnl = _to_float(item.get("net_pnl"))
        if ts_ms is None or pnl is None:
            continue
        trade_points.append((ts_ms, pnl))

    if not trade_points:
        return _build_equity_curve_from_balance_history(state=state, candles=candles)

    trade_points.sort(key=lambda point: point[0])
    current_balance = float(initial_balance)
    points: list[dict[str, Any]] = []

    if range_start_ms is not None:
        for ts_ms, pnl in trade_points:
            if ts_ms < range_start_ms:
                current_balance += pnl
                continue
            break
        points.append({"time": _iso_from_ts_ms(range_start_ms), "balance": current_balance})

    for ts_ms, pnl in trade_points:
        if range_start_ms is not None and ts_ms < range_start_ms:
            continue
        if range_end_ms is not None and ts_ms > range_end_ms:
            break
        # Keep a pre-event point at the same timestamp so UI can render step-like jumps.
        if points:
            points.append({"time": _iso_from_ts_ms(ts_ms), "balance": current_balance})
        current_balance += pnl
        points.append({"time": _iso_from_ts_ms(ts_ms), "balance": current_balance})

    if range_end_ms is not None:
        has_range_end = bool(points and points[-1]["time"] == _iso_from_ts_ms(range_end_ms))
        if not has_range_end:
            points.append({"time": _iso_from_ts_ms(range_end_ms), "balance": current_balance})

    return points


def build_chart_payload(
    symbol: str | None,
    period: str,
    interval: str,
    include_indicators: bool = True,
    end: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    config: dict[str, Any] = {}
    state: dict[str, Any] = {}

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))

    try:
        loaded_state, state_warnings = load_state()
        state = loaded_state
        warnings.extend(state_warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))
        state = {
            "position": None,
            "balance": None,
        }

    trade_history, trade_history_warnings = load_trade_history()
    balance_history, balance_history_warnings = load_balance_history()
    warnings.extend(trade_history_warnings)
    warnings.extend(balance_history_warnings)

    if not trade_history:
        legacy_trades = state.get("trade_history")
        if isinstance(legacy_trades, list):
            trade_history = [item for item in legacy_trades if isinstance(item, dict)]

    if not balance_history:
        legacy_balances = state.get("balance_history")
        if isinstance(legacy_balances, list):
            balance_history = []
            for item in legacy_balances:
                parsed = _to_float(item)
                if parsed is not None:
                    balance_history.append(parsed)

    state_for_chart = dict(state)
    state_for_chart["trade_history"] = trade_history
    state_for_chart["balance_history"] = balance_history
    if state_for_chart.get("balance") is None and balance_history:
        state_for_chart["balance"] = balance_history[-1]

    resolved_symbol = (symbol or str(config.get("symbol", "BTCUSDT"))).strip().upper()
    if not resolved_symbol:
        resolved_symbol = "BTCUSDT"

    resolved_source = _parse_chart_source(source)
    period_ms = _parse_period_to_ms(period)
    if resolved_source == "dataset":
        max_candles_for_source = CHART_DATASET_MAX_CANDLES
    elif resolved_source == "binance":
        max_candles_for_source = CHART_MAX_CANDLES
    else:
        max_candles_for_source = max(CHART_DATASET_MAX_CANDLES, CHART_MAX_CANDLES)
    resolved_interval, interval_warnings = _resolve_chart_interval(
        period_ms=period_ms,
        requested_interval=interval,
        max_candles=max_candles_for_source,
    )
    warnings.extend(interval_warnings)

    end_ms = None
    if end:
        end_ms = _parse_time_to_ms(end)
        if end_ms is None:
            raise ValueError("Invalid end format. Use ISO datetime or unix timestamp.")

    candles, candle_warnings, source_used, interval_used = _fetch_candles(
        source=resolved_source,
        symbol=resolved_symbol,
        period_ms=period_ms,
        interval=resolved_interval,
        end_ms=end_ms,
    )
    warnings.extend(candle_warnings)

    range_start_ms = candles[0]["ts_ms"] if candles else None
    range_end_ms = candles[-1]["ts_ms"] if candles else None

    markers = _build_markers(state=state_for_chart, range_start_ms=range_start_ms, range_end_ms=range_end_ms)
    open_position = _normalize_open_position(state_for_chart.get("position"))
    current_levels = _build_current_levels(open_position)
    rsi_levels = _build_rsi_levels(config)
    indicator_spec = _build_indicator_spec(config)
    indicators: dict[str, Any] = {}
    if include_indicators and candles:
        indicator_candles = candles
        if source_used != "dataset":
            dataset_indicator_candles, dataset_indicator_warnings = _fetch_candles_from_dataset(
                symbol=resolved_symbol,
                period_ms=period_ms,
                interval=interval_used,
                end_ms=end_ms,
            )
            for warning in dataset_indicator_warnings:
                if warning not in warnings:
                    warnings.append(warning)
            if dataset_indicator_candles:
                indicator_candles = dataset_indicator_candles

        indicators = _build_indicators_from_candle_fields(indicator_candles)
        if not indicators:
            strategy_fallback_indicators, strategy_fallback_warnings = _build_strategy_fallback_indicators(
                symbol=resolved_symbol,
                config=config,
                target_candles=candles,
                period_ms=period_ms,
                end_ms=range_end_ms,
            )
            for warning in strategy_fallback_warnings:
                if warning not in warnings:
                    warnings.append(warning)
            if strategy_fallback_indicators:
                warnings.append("Dataset indicators unavailable, falling back to Binance strategy indicators.")
                indicators = strategy_fallback_indicators
            else:
                warnings.append(
                    "Dataset indicators unavailable, and strategy-aligned Binance indicators failed. "
                    "Falling back to computed indicators from chart candles."
                )
                indicators = _build_indicators(candles, indicator_spec=indicator_spec)
    equity_curve = _build_equity_curve_from_candle_balance(candles)
    if not equity_curve:
        equity_curve = _build_equity_curve(
            state=state_for_chart,
            candles=candles,
            config=config,
            range_start_ms=range_start_ms,
            range_end_ms=range_end_ms,
        )

    # Remove internal timestamp helper before returning response.
    public_candles = [{k: v for k, v in candle.items() if k != "ts_ms"} for candle in candles]

    return {
        "symbol": resolved_symbol,
        "interval": interval_used,
        "requested_interval": interval,
        "source": source_used,
        "requested_source": resolved_source,
        "period": period,
        "candles": public_candles,
        "markers": markers,
        "current_levels": current_levels,
        "rsi_levels": rsi_levels,
        "indicator_spec": indicator_spec,
        "open_position": open_position,
        "indicators": indicators,
        "equity_curve": equity_curve,
        "range_start": _iso_from_ts_ms(range_start_ms) if range_start_ms is not None else None,
        "range_end": _iso_from_ts_ms(range_end_ms) if range_end_ms is not None else None,
        "warnings": warnings,
    }
