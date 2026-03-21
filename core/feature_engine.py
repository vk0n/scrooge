from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable

import pandas as pd

from core.market_events import CandleClosedEvent


DEFAULT_FEATURE_ENGINE_LIMITS = {
    "small": 240,
    "medium": 240,
    "big": 240,
}


def _interval_to_freq(interval: str) -> str:
    normalized = str(interval or "").strip().lower()
    if normalized.endswith("m"):
        return f"{int(normalized[:-1])}min"
    if normalized.endswith("h"):
        return f"{int(normalized[:-1])}h"
    if normalized.endswith("d"):
        return f"{int(normalized[:-1])}d"
    raise ValueError(f"Unsupported interval for feature engine: {interval}")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class Candle:
    open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> Candle | None:
        open_time = pd.Timestamp(payload.get("open_time"))
        if pd.isna(open_time):
            return None
        if open_time.tzinfo is not None:
            open_time = open_time.tz_convert(None)
        open_price = _to_float(payload.get("open"))
        high_price = _to_float(payload.get("high"))
        low_price = _to_float(payload.get("low"))
        close_price = _to_float(payload.get("close"))
        volume = _to_float(payload.get("volume")) or 0.0
        if open_price is None or high_price is None or low_price is None or close_price is None:
            return None
        return cls(
            open_time=open_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

    @classmethod
    def from_event(cls, event: CandleClosedEvent) -> Candle:
        open_time = pd.Timestamp(event.open_time)
        if open_time.tzinfo is not None:
            open_time = open_time.tz_convert(None)
        return cls(
            open_time=open_time,
            open=float(event.open),
            high=float(event.high),
            low=float(event.low),
            close=float(event.close),
            volume=float(event.volume),
        )


class TimeframeState:
    def __init__(self, *, interval: str, limit: int) -> None:
        self.interval = str(interval)
        self.freq = _interval_to_freq(self.interval)
        self.limit = max(1, int(limit))
        self.closed: deque[Candle] = deque(maxlen=self.limit)
        self.forming: Candle | None = None

    def bootstrap(self, candles: Iterable[Candle]) -> None:
        self.closed.clear()
        self.forming = None
        for candle in candles:
            self.upsert_closed(candle)

    def upsert_closed(self, candle: Candle) -> None:
        if self.closed and self.closed[-1].open_time == candle.open_time:
            self.closed[-1] = candle
        else:
            self.closed.append(candle)
        if self.forming is not None and self.forming.open_time <= candle.open_time:
            self.forming = None

    def update_from_price_tick(self, ts_value: pd.Timestamp, price: float) -> None:
        open_time = ts_value.floor(self.freq)
        if self.forming is None or self.forming.open_time != open_time:
            self.forming = Candle(
                open_time=open_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
            )
            return
        self.forming.high = max(self.forming.high, price)
        self.forming.low = min(self.forming.low, price)
        self.forming.close = price

    def set_forming_from_small_frame(self, df_small: pd.DataFrame) -> None:
        if df_small is None or df_small.empty:
            self.forming = None
            return
        normalized = _normalized_frame(df_small)
        if normalized.empty:
            self.forming = None
            return
        current_open_time = normalized["open_time"].iloc[-1].floor(self.freq)
        latest_closed_open_time = self.closed[-1].open_time if self.closed else None
        if latest_closed_open_time is not None and latest_closed_open_time == current_open_time:
            self.forming = None
            return
        current_slice = normalized[normalized["open_time"].dt.floor(self.freq) == current_open_time]
        candle = _aggregate_frame_to_candle(current_slice, open_time=current_open_time)
        self.forming = candle


class EMAFeature:
    def __init__(self, period: int) -> None:
        self.period = int(period)
        self.alpha = 2.0 / (self.period + 1.0)
        self.reset()

    def reset(self) -> None:
        self._seed_sum = 0.0
        self._seed_count = 0
        self._closed_value: float | None = None

    def bootstrap(self, closes: Iterable[float]) -> None:
        self.reset()
        for close in closes:
            self.on_closed(float(close))

    def on_closed(self, close: float) -> None:
        if self._closed_value is None and self._seed_count < self.period:
            self._seed_sum += close
            self._seed_count += 1
            if self._seed_count == self.period:
                self._closed_value = self._seed_sum / self.period
            return
        if self._closed_value is None:
            return
        self._closed_value = ((close - self._closed_value) * self.alpha) + self._closed_value

    def closed_value(self) -> float | None:
        return self._closed_value

    def intrabar_value(self, close: float | None) -> float | None:
        if close is None:
            return self._closed_value
        if self._closed_value is None:
            if self._seed_count != self.period - 1:
                return None
            return (self._seed_sum + close) / self.period
        return ((close - self._closed_value) * self.alpha) + self._closed_value


class RSIFeature:
    def __init__(self, period: int) -> None:
        self.period = int(period)
        self.reset()

    def reset(self) -> None:
        self._previous_close: float | None = None
        self._seed_gain_sum = 0.0
        self._seed_loss_sum = 0.0
        self._seed_count = 0
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None

    def bootstrap(self, closes: Iterable[float]) -> None:
        self.reset()
        for close in closes:
            self.on_closed(float(close))

    def on_closed(self, close: float) -> None:
        if self._previous_close is None:
            self._previous_close = close
            return
        delta = close - self._previous_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._previous_close = close
        if self._avg_gain is None or self._avg_loss is None:
            self._seed_gain_sum += gain
            self._seed_loss_sum += loss
            self._seed_count += 1
            if self._seed_count == self.period:
                self._avg_gain = self._seed_gain_sum / self.period
                self._avg_loss = self._seed_loss_sum / self.period
            return
        self._avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
        self._avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period

    def closed_value(self) -> float | None:
        return _rsi_from_averages(self._avg_gain, self._avg_loss)

    def intrabar_value(self, close: float | None) -> float | None:
        if close is None:
            return self.closed_value()
        if self._previous_close is None or self._avg_gain is None or self._avg_loss is None:
            return None
        delta = close - self._previous_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
        avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period
        return _rsi_from_averages(avg_gain, avg_loss)


class ATRFeature:
    def __init__(self, period: int) -> None:
        self.period = int(period)
        self.reset()

    def reset(self) -> None:
        self._previous_close: float | None = None
        self._seed_tr_sum = 0.0
        self._seed_count = 0
        self._closed_value: float | None = None

    def bootstrap(self, candles: Iterable[Candle]) -> None:
        self.reset()
        for candle in candles:
            self.on_closed(candle)

    def on_closed(self, candle: Candle) -> None:
        tr = _true_range(
            high=candle.high,
            low=candle.low,
            previous_close=self._previous_close,
        )
        self._previous_close = candle.close
        if self._closed_value is None and self._seed_count < self.period:
            self._seed_tr_sum += tr
            self._seed_count += 1
            if self._seed_count == self.period:
                self._closed_value = self._seed_tr_sum / self.period
            return
        if self._closed_value is None:
            return
        self._closed_value = ((self._closed_value * (self.period - 1)) + tr) / self.period

    def closed_value(self) -> float | None:
        return self._closed_value

    def intrabar_value(self, candle: Candle | None) -> float | None:
        if candle is None:
            return self._closed_value
        tr = _true_range(
            high=candle.high,
            low=candle.low,
            previous_close=self._previous_close,
        )
        if self._closed_value is None:
            if self._seed_count != self.period - 1:
                return None
            return (self._seed_tr_sum + tr) / self.period
        return ((self._closed_value * (self.period - 1)) + tr) / self.period


class BollingerBandsFeature:
    def __init__(self, period: int, std_mult: float) -> None:
        self.period = int(period)
        self.std_mult = float(std_mult)
        self.reset()

    def reset(self) -> None:
        self._closes: deque[float] = deque(maxlen=self.period)
        self._sum = 0.0
        self._sum_sq = 0.0

    def bootstrap(self, closes: Iterable[float]) -> None:
        self.reset()
        for close in closes:
            self.on_closed(float(close))

    def on_closed(self, close: float) -> None:
        if len(self._closes) == self.period:
            removed = self._closes.popleft()
            self._sum -= removed
            self._sum_sq -= removed * removed
        self._closes.append(close)
        self._sum += close
        self._sum_sq += close * close

    def closed_values(self) -> dict[str, float] | None:
        if len(self._closes) < self.period:
            return None
        return _bollinger_values(self._sum, self._sum_sq, self.period, self.std_mult)

    def intrabar_values(self, close: float | None) -> dict[str, float] | None:
        if close is None:
            return self.closed_values()
        if len(self._closes) < self.period - 1:
            return None
        if len(self._closes) == self.period:
            removed = self._closes[0]
            next_sum = self._sum - removed + close
            next_sum_sq = self._sum_sq - (removed * removed) + (close * close)
            return _bollinger_values(next_sum, next_sum_sq, self.period, self.std_mult)
        next_sum = self._sum + close
        next_sum_sq = self._sum_sq + (close * close)
        return _bollinger_values(next_sum, next_sum_sq, self.period, self.std_mult)


class FeatureEngine:
    def __init__(
        self,
        *,
        intervals: dict[str, str],
        limits: dict[str, int] | None = None,
    ) -> None:
        resolved_limits = dict(DEFAULT_FEATURE_ENGINE_LIMITS)
        if limits is not None:
            for key, value in limits.items():
                resolved_limits[str(key)] = max(1, int(value))
        self.timeframes = {
            key: TimeframeState(interval=str(intervals[key]), limit=resolved_limits.get(key, 240))
            for key in ("small", "medium", "big")
        }
        self._bb = BollingerBandsFeature(period=20, std_mult=2.0)
        self._atr = ATRFeature(period=14)
        self._rsi = RSIFeature(period=11)
        self._ema = EMAFeature(period=50)

    def bootstrap_from_frames(
        self,
        *,
        df_small: pd.DataFrame,
        df_medium: pd.DataFrame,
        df_big: pd.DataFrame,
    ) -> None:
        self.timeframes["small"].bootstrap(_candles_from_frame(df_small))
        self.timeframes["medium"].bootstrap(_candles_from_frame(df_medium))
        self.timeframes["big"].bootstrap(_candles_from_frame(df_big))
        self.timeframes["medium"].set_forming_from_small_frame(df_small)
        self.timeframes["big"].set_forming_from_small_frame(df_small)
        self._bb.bootstrap(candle.close for candle in self.timeframes["medium"].closed)
        self._atr.bootstrap(self.timeframes["medium"].closed)
        self._rsi.bootstrap(candle.close for candle in self.timeframes["big"].closed)
        self._ema.bootstrap(candle.close for candle in self.timeframes["big"].closed)

    def on_price_tick(self, *, ts_value: pd.Timestamp, price: float) -> None:
        normalized_ts = pd.Timestamp(ts_value)
        if normalized_ts.tzinfo is not None:
            normalized_ts = normalized_ts.tz_convert(None)
        for state in self.timeframes.values():
            state.update_from_price_tick(normalized_ts, float(price))

    def on_candle_closed(self, *, timeframe: str, event: CandleClosedEvent) -> None:
        candle = Candle.from_event(event)
        state = self.timeframes[timeframe]
        state.upsert_closed(candle)
        if timeframe == "medium":
            self._bb.on_closed(candle.close)
            self._atr.on_closed(candle)
        elif timeframe == "big":
            self._rsi.on_closed(candle.close)
            self._ema.on_closed(candle.close)

    def current_price(self) -> float | None:
        small = self.timeframes["small"]
        if small.forming is not None:
            return small.forming.close
        if small.closed:
            return small.closed[-1].close
        return None

    def realtime_values(self) -> dict[str, float | None] | None:
        medium_forming = self.timeframes["medium"].forming
        big_forming = self.timeframes["big"].forming
        bb_values = self._bb.intrabar_values(medium_forming.close if medium_forming is not None else None)
        values = {
            "EMA": self._ema.intrabar_value(big_forming.close if big_forming is not None else None),
            "RSI": self._rsi.intrabar_value(big_forming.close if big_forming is not None else None),
            "ATR": self._atr.intrabar_value(medium_forming),
            "BBL": None if bb_values is None else bb_values["BBL"],
            "BBM": None if bb_values is None else bb_values["BBM"],
            "BBU": None if bb_values is None else bb_values["BBU"],
        }
        if all(value is None for value in values.values()):
            return None
        return values


def _candles_from_frame(df: pd.DataFrame | None) -> list[Candle]:
    normalized = _normalized_frame(df)
    candles: list[Candle] = []
    for row in normalized.to_dict(orient="records"):
        candle = Candle.from_mapping(row)
        if candle is not None:
            candles.append(candle)
    return candles


def _normalized_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open_time", "open", "high", "low", "close"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return out[["open_time", "open", "high", "low", "close", "volume"]]


def _aggregate_frame_to_candle(df: pd.DataFrame, *, open_time: pd.Timestamp) -> Candle | None:
    if df.empty:
        return None
    return Candle(
        open_time=open_time,
        open=float(df["open"].iloc[0]),
        high=float(df["high"].max()),
        low=float(df["low"].min()),
        close=float(df["close"].iloc[-1]),
        volume=float(df["volume"].sum()),
    )


def _true_range(*, high: float, low: float, previous_close: float | None) -> float:
    if previous_close is None:
        return high - low
    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


def _rsi_from_averages(avg_gain: float | None, avg_loss: float | None) -> float | None:
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_values(sum_value: float, sum_sq_value: float, period: int, std_mult: float) -> dict[str, float]:
    mean = sum_value / period
    if period <= 1:
        variance = 0.0
    else:
        variance = max((sum_sq_value - (period * mean * mean)) / (period - 1), 0.0)
    std_dev = math.sqrt(variance)
    return {
        "BBL": mean - (std_mult * std_dev),
        "BBM": mean,
        "BBU": mean + (std_mult * std_dev),
    }
