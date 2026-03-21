from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from core.market_events import CandleClosedEvent


DEFAULT_FEATURE_ENGINE_LIMITS = {
    "small": 240,
    "medium": 240,
    "big": 240,
}
FEATURE_INPUT_CLOSE = "close"
FEATURE_INPUT_CANDLE = "candle"


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


class OnlineFeatureRuntime(Protocol):
    def bootstrap(self, values: Iterable[Any]) -> None:
        ...

    def on_closed(self, value: Any) -> None:
        ...

    def closed_payload(self) -> dict[str, float | None] | None:
        ...

    def intrabar_payload(self, value: Any) -> dict[str, float | None] | None:
        ...


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    key: str
    timeframe: str
    input_kind: str
    outputs: tuple[str, ...]
    warmup_bars: int
    factory: Callable[[], OnlineFeatureRuntime]


@dataclass(slots=True)
class FeatureBinding:
    spec: FeatureSpec
    runtime: OnlineFeatureRuntime


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

    def closed_payload(self) -> dict[str, float | None] | None:
        return {"EMA": self.closed_value()}

    def intrabar_payload(self, close: float | None) -> dict[str, float | None] | None:
        return {"EMA": self.intrabar_value(close)}


class RSIFeature:
    def __init__(self, period: int) -> None:
        self.period = int(period)
        self.alpha = 1.0 / self.period if self.period > 0 else 0.5
        self.reset()

    def reset(self) -> None:
        self._previous_close: float | None = None
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
            self._avg_gain = gain
            self._avg_loss = loss
            return
        self._avg_gain = (self.alpha * gain) + ((1.0 - self.alpha) * self._avg_gain)
        self._avg_loss = (self.alpha * loss) + ((1.0 - self.alpha) * self._avg_loss)

    def closed_value(self) -> float | None:
        return _rsi_from_averages(self._avg_gain, self._avg_loss)

    def intrabar_value(self, close: float | None) -> float | None:
        if close is None:
            return self.closed_value()
        if self._previous_close is None:
            return None
        delta = close - self._previous_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if self._avg_gain is None or self._avg_loss is None:
            avg_gain = gain
            avg_loss = loss
        else:
            avg_gain = (self.alpha * gain) + ((1.0 - self.alpha) * self._avg_gain)
            avg_loss = (self.alpha * loss) + ((1.0 - self.alpha) * self._avg_loss)
        return _rsi_from_averages(avg_gain, avg_loss)

    def closed_payload(self) -> dict[str, float | None] | None:
        return {"RSI": self.closed_value()}

    def intrabar_payload(self, close: float | None) -> dict[str, float | None] | None:
        return {"RSI": self.intrabar_value(close)}


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

    def closed_payload(self) -> dict[str, float | None] | None:
        return {"ATR": self.closed_value()}

    def intrabar_payload(self, candle: Candle | None) -> dict[str, float | None] | None:
        return {"ATR": self.intrabar_value(candle)}


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

    def closed_payload(self) -> dict[str, float | None] | None:
        return self.closed_values()

    def intrabar_payload(self, close: float | None) -> dict[str, float | None] | None:
        return self.intrabar_values(close)


def _build_default_feature_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            key="bb",
            timeframe="medium",
            input_kind=FEATURE_INPUT_CLOSE,
            outputs=("BBL", "BBM", "BBU"),
            warmup_bars=20,
            factory=lambda: BollingerBandsFeature(period=20, std_mult=2.0),
        ),
        FeatureSpec(
            key="atr",
            timeframe="medium",
            input_kind=FEATURE_INPUT_CANDLE,
            outputs=("ATR",),
            warmup_bars=14,
            factory=lambda: ATRFeature(period=14),
        ),
        FeatureSpec(
            key="rsi",
            timeframe="big",
            input_kind=FEATURE_INPUT_CLOSE,
            outputs=("RSI",),
            warmup_bars=11,
            factory=lambda: RSIFeature(period=11),
        ),
        FeatureSpec(
            key="ema",
            timeframe="big",
            input_kind=FEATURE_INPUT_CLOSE,
            outputs=("EMA",),
            warmup_bars=50,
            factory=lambda: EMAFeature(period=50),
        ),
    )


DEFAULT_FEATURE_SPECS = _build_default_feature_specs()


class FeatureEngine:
    def __init__(
        self,
        *,
        intervals: dict[str, str],
        limits: dict[str, int] | None = None,
        feature_specs: Iterable[FeatureSpec] | None = None,
    ) -> None:
        resolved_limits = dict(DEFAULT_FEATURE_ENGINE_LIMITS)
        if limits is not None:
            for key, value in limits.items():
                resolved_limits[str(key)] = max(1, int(value))
        self.timeframes = {
            key: TimeframeState(interval=str(intervals[key]), limit=resolved_limits.get(key, 240))
            for key in ("small", "medium", "big")
        }
        specs = DEFAULT_FEATURE_SPECS if feature_specs is None else tuple(feature_specs)
        self.feature_specs = specs
        self.output_columns = tuple(column for spec in specs for column in spec.outputs)
        self._features: tuple[FeatureBinding, ...] = tuple(
            FeatureBinding(spec=spec, runtime=spec.factory())
            for spec in specs
        )
        self._features_by_timeframe: dict[str, tuple[FeatureBinding, ...]] = {
            timeframe: tuple(binding for binding in self._features if binding.spec.timeframe == timeframe)
            for timeframe in self.timeframes
        }
        for binding in self._features:
            if binding.spec.timeframe not in self.timeframes:
                raise ValueError(f"Unknown feature timeframe: {binding.spec.timeframe}")
            if binding.spec.input_kind not in {FEATURE_INPUT_CLOSE, FEATURE_INPUT_CANDLE}:
                raise ValueError(
                    f"Unsupported feature input kind for {binding.spec.key}: {binding.spec.input_kind}"
                )

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
        for binding in self._features:
            timeframe_state = self.timeframes[binding.spec.timeframe]
            binding.runtime.bootstrap(_iter_feature_inputs(timeframe_state.closed, input_kind=binding.spec.input_kind))

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
        for binding in self._features_by_timeframe.get(timeframe, ()):
            binding.runtime.on_closed(_feature_input_from_candle(candle, input_kind=binding.spec.input_kind))

    def current_price(self) -> float | None:
        small = self.timeframes["small"]
        if small.forming is not None:
            return small.forming.close
        if small.closed:
            return small.closed[-1].close
        return None

    def closed_values(self) -> dict[str, float | None] | None:
        return _merge_feature_payloads(
            binding.runtime.closed_payload()
            for binding in self._features
        )

    def realtime_values(self) -> dict[str, float | None] | None:
        return _merge_feature_payloads(
            binding.runtime.intrabar_payload(
                _feature_input_from_forming_candle(
                    self.timeframes[binding.spec.timeframe].forming,
                    input_kind=binding.spec.input_kind,
                )
            )
            for binding in self._features
        )


class FeatureFrameBuilder:
    def __init__(
        self,
        *,
        feature_specs: Iterable[FeatureSpec] | None = None,
    ) -> None:
        specs = DEFAULT_FEATURE_SPECS if feature_specs is None else tuple(feature_specs)
        self.feature_specs = specs
        self.output_columns = _resolve_output_columns(specs)
        self._bindings_by_timeframe: dict[str, tuple[FeatureBinding, ...]] = {
            timeframe: tuple(
                FeatureBinding(spec=spec, runtime=spec.factory())
                for spec in specs
                if spec.timeframe == timeframe
            )
            for timeframe in ("small", "medium", "big")
        }

    def update_timeframe_frame(self, *, timeframe: str, df: pd.DataFrame) -> pd.DataFrame | None:
        normalized_timeframe = str(timeframe)
        if normalized_timeframe not in self._bindings_by_timeframe:
            raise ValueError(f"Unknown feature timeframe: {timeframe}")
        return _build_bound_timeframe_feature_frame(
            df,
            bindings=self._bindings_by_timeframe[normalized_timeframe],
        )


def build_feature_frame(
    *,
    df_small: pd.DataFrame,
    df_medium: pd.DataFrame,
    df_big: pd.DataFrame,
    feature_specs: Iterable[FeatureSpec] | None = None,
) -> pd.DataFrame:
    builder = FeatureFrameBuilder(feature_specs=feature_specs)
    return merge_feature_frames(
        df_small=df_small,
        timeframe_feature_frames={
            "medium": builder.update_timeframe_frame(timeframe="medium", df=df_medium),
            "big": builder.update_timeframe_frame(timeframe="big", df=df_big),
        },
        output_columns=builder.output_columns,
    )


def _candles_from_frame(df: pd.DataFrame | None) -> list[Candle]:
    normalized = _normalized_frame(df)
    candles: list[Candle] = []
    for row in normalized.to_dict(orient="records"):
        candle = Candle.from_mapping(row)
        if candle is not None:
            candles.append(candle)
    return candles


def _iter_feature_inputs(candles: Iterable[Candle], *, input_kind: str) -> Iterable[Any]:
    if input_kind == FEATURE_INPUT_CANDLE:
        return candles
    return (candle.close for candle in candles)


def _feature_input_from_candle(candle: Candle, *, input_kind: str) -> Any:
    if input_kind == FEATURE_INPUT_CANDLE:
        return candle
    return candle.close


def _feature_input_from_forming_candle(candle: Candle | None, *, input_kind: str) -> Any:
    if input_kind == FEATURE_INPUT_CANDLE:
        return candle
    if candle is None:
        return None
    return candle.close


def _merge_feature_payloads(payloads: Iterable[dict[str, float | None] | None]) -> dict[str, float | None] | None:
    values: dict[str, float | None] = {}
    saw_value = False
    for payload in payloads:
        if payload is None:
            continue
        for key, value in payload.items():
            values[key] = value
            saw_value = saw_value or value is not None
    if not values or not saw_value:
        return None
    return values


def _resolve_output_columns(feature_specs: Iterable[FeatureSpec]) -> tuple[str, ...]:
    seen: set[str] = set()
    output_columns: list[str] = []
    for spec in feature_specs:
        for column in spec.outputs:
            if column in seen:
                continue
            seen.add(column)
            output_columns.append(column)
    return tuple(output_columns)


def merge_feature_frames(
    *,
    df_small: pd.DataFrame,
    timeframe_feature_frames: dict[str, pd.DataFrame | None],
    output_columns: Iterable[str],
) -> pd.DataFrame:
    base_columns = ("open_time", "open", "high", "low", "close", "volume")
    small = _normalized_frame(df_small)
    resolved_output_columns = tuple(output_columns)
    if small.empty:
        return pd.DataFrame(columns=[*base_columns, *resolved_output_columns])

    merged = small.set_index("open_time")
    for timeframe in ("small", "medium", "big"):
        feature_df = timeframe_feature_frames.get(timeframe)
        if feature_df is None or feature_df.empty:
            continue
        merged = merged.merge(feature_df.set_index("open_time"), left_index=True, right_index=True, how="left")
        if timeframe != "small":
            merged = merged.ffill()

    merged.reset_index(inplace=True)
    for column in resolved_output_columns:
        if column not in merged.columns:
            merged[column] = None
    return merged[[*base_columns, *resolved_output_columns]]


def _build_bound_timeframe_feature_frame(
    df: pd.DataFrame,
    *,
    bindings: Iterable[FeatureBinding],
) -> pd.DataFrame | None:
    resolved_bindings = tuple(bindings)
    if not resolved_bindings:
        return None

    output_columns = _resolve_output_columns(binding.spec for binding in resolved_bindings)
    if df.empty:
        return pd.DataFrame(columns=["open_time", *output_columns])

    rows: list[dict[str, Any]] = []
    for candle in _candles_from_frame(df):
        row: dict[str, Any] = {"open_time": candle.open_time}
        for binding in resolved_bindings:
            binding.runtime.on_closed(
                _feature_input_from_candle(candle, input_kind=binding.spec.input_kind)
            )
            payload = binding.runtime.closed_payload() or {}
            for column in binding.spec.outputs:
                row[column] = payload.get(column)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    for column in output_columns:
        if column not in feature_df.columns:
            feature_df[column] = None
    return feature_df[["open_time", *output_columns]]


def _build_timeframe_feature_frame(
    df: pd.DataFrame,
    *,
    timeframe: str,
    feature_specs: Iterable[FeatureSpec],
) -> pd.DataFrame | None:
    return _build_bound_timeframe_feature_frame(
        df,
        bindings=tuple(
            FeatureBinding(spec=spec, runtime=spec.factory())
            for spec in feature_specs
            if spec.timeframe == timeframe
        ),
    )


def _normalized_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out["open_time"]):
        out["open_time"] = pd.to_datetime(out["open_time"], unit="ms", errors="coerce")
    else:
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
