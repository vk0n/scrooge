from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Iterable

import pandas as pd
from tqdm import tqdm

from bot.event_log import emit_event, get_technical_logger
from bot.state import load_state, save_state, update_balance, update_position
from bot.trade import (
    can_open_trade,
    close_position,
    compute_qty,
    get_balance,
    get_order_execution_summary,
    open_position,
)
from core.feature_engine import FeatureEngine, floor_ns, interval_to_ns
from core.indicator_inputs import (
    INDICATOR_COLUMNS,
    IndicatorSelectionPlan,
    indicator_input_mode_for_column,
    indicator_selection_plan,
    normalize_indicator_inputs,
    uses_realtime_indicator_inputs,
)
from core.market_events import (
    AccountBalanceEvent,
    CandleClosedEvent,
    IndicatorSnapshotEvent,
    MarketEvent,
    OrderTradeUpdateEvent,
    PositionSnapshotEvent,
    PriceTickEvent,
    market_event_to_dict,
)


@dataclass(slots=True)
class DiscreteRowSnapshot:
    raw_row: Any
    price: Any
    lower: Any
    upper: Any
    mid: Any
    atr: Any
    rsi: Any
    ema: Any
    row_ts: str
    log_ts: str


@dataclass(slots=True)
class StrategyRuntime:
    state: dict[str, Any]
    balance: float
    position: dict[str, Any] | None
    trade_history: list[dict[str, Any]]
    balance_history: list[float]
    log_buffer: list[str]
    live: bool
    use_state: bool
    enable_logs: bool
    execution_mode: str
    execution_sync: ExecutionSyncContext
    execution_events: list[MarketEvent]


@dataclass(slots=True)
class ExecutionSyncContext:
    total_events: int = 0
    account_balance_events: int = 0
    position_snapshot_events: int = 0
    order_trade_update_events: int = 0
    trade_execution_events: int = 0
    filled_order_events: int = 0
    divergence_events: int = 0
    realized_pnl_total: float = 0.0
    commission_total: float = 0.0
    last_event_type: str | None = None
    last_event_ts: str | None = None
    balance_alignment: str | None = None
    balance_drift: float | None = None
    position_alignment: str | None = None


@dataclass(slots=True)
class StrategyConfig:
    qty: float | None
    sl_mult: float
    tp_mult: float
    symbol: str
    leverage: float
    use_full_balance: bool
    fee_rate: float
    rsi_extreme_long: float
    rsi_extreme_short: float
    rsi_long_open_threshold: float
    rsi_long_qty_threshold: float
    rsi_long_tp_threshold: float
    rsi_long_close_threshold: float
    rsi_short_open_threshold: float
    rsi_short_qty_threshold: float
    rsi_short_tp_threshold: float
    rsi_short_close_threshold: float
    trail_atr_mult: float
    allow_entries: bool
    execution_mode: str
    runtime_mode: str
    strategy_mode: str
    indicator_inputs: dict[str, str]


@dataclass(slots=True)
class RealtimeStrategyProcessor:
    runtime: StrategyRuntime
    feature_engine: FeatureEngine
    selection_plan: IndicatorSelectionPlan
    emit_on_price_tick: bool
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None]
    target_symbol: str | None
    intervals: dict[str, str]
    latest_discrete_indicator_values: dict[str, float | None] | None
    pending_small_candles: dict[tuple[str, str], tuple[CandleClosedEvent, int | None]]
    small_interval_ns: int
    processed_price_ticks: int = 0
    emitted_snapshots: int = 0
    tick_seen_small_open_times: set[int] = field(default_factory=set)

    def bootstrap_from_frames(
        self,
        *,
        df_small: pd.DataFrame,
        df_medium: pd.DataFrame,
        df_big: pd.DataFrame,
    ) -> None:
        self.feature_engine.bootstrap_from_frames(
            df_small=df_small,
            df_medium=df_medium,
            df_big=df_big,
        )
        self.latest_discrete_indicator_values = self.feature_engine.closed_values()

    def _resolve_timeframe(self, interval_value: str) -> str | None:
        for timeframe, configured in self.intervals.items():
            if configured == interval_value:
                return timeframe
        return None

    def _emit_snapshot(self, snapshot: DiscreteRowSnapshot | None) -> None:
        if snapshot is None:
            return
        self.on_row(snapshot, self.runtime)
        if not self.runtime.live:
            self.runtime.balance_history.append(self.runtime.balance)
        self.emitted_snapshots += 1

    def process_event(self, event: MarketEvent) -> None:
        if apply_market_event_runtime_sync(self.runtime, event, target_symbol=self.target_symbol):
            return

        event_symbol = getattr(event, "symbol", None)
        if self.target_symbol and str(event_symbol or "").strip().upper() != self.target_symbol:
            return
        event_type = getattr(event, "event_type", "")

        if event_type == "candle_closed":
            timeframe = self._resolve_timeframe(event.interval)
            if timeframe is None:
                return
            self.feature_engine.on_candle_closed(timeframe=timeframe, event=event)
            if timeframe == "small":
                self.pending_small_candles[(event.symbol, event.ts)] = (event, _event_ts_to_ns(event.open_time))
            return

        if event_type == "indicator_snapshot" and event.interval == "discrete_snapshot":
            self.latest_discrete_indicator_values = _indicator_values_from_snapshot_event(event)
            key = (event.symbol, event.ts)
            pending_candle_entry = self.pending_small_candles.pop(key, None)
            if pending_candle_entry is None:
                return
            pending_candle, open_key = pending_candle_entry
            if open_key is None:
                open_key = _event_ts_to_ns(pending_candle.open_time)
            if open_key is None:
                return
            if self.emit_on_price_tick and open_key in self.tick_seen_small_open_times:
                return
            self._emit_snapshot(
                _build_realtime_snapshot(
                    event_ts=(event.ts if isinstance(event.ts, str) else format_event_timestamp(event.ts)),
                    feature_engine=self.feature_engine,
                    selection_plan=self.selection_plan,
                    discrete_indicator_values=self.latest_discrete_indicator_values,
                    current_price=float(pending_candle.close),
                )
            )
            return

        if event_type != "price_tick":
            return

        event_ts_ns = _event_ts_to_ns(event.ts)
        price = to_float(event.price)
        if event_ts_ns is None or price is None:
            return
        self.processed_price_ticks += 1
        self.tick_seen_small_open_times.add(floor_ns(event_ts_ns, interval_ns=self.small_interval_ns))
        self.feature_engine.on_price_tick(ts_ns=event_ts_ns, price=price)
        if not self.emit_on_price_tick:
            return
        self._emit_snapshot(
            _build_realtime_snapshot(
                event_ts=(event.ts if isinstance(event.ts, str) else format_event_timestamp(event.ts)),
                feature_engine=self.feature_engine,
                selection_plan=self.selection_plan,
                discrete_indicator_values=self.latest_discrete_indicator_values,
                current_price=price,
            )
        )


@dataclass(slots=True)
class EntryDecision:
    side: str
    size: float
    entry: float
    sl: float
    tp: float
    liq_price: float
    stake_mode: str
    trigger: str


@dataclass(slots=True)
class EntryGuardRejection:
    side: str
    entry: float
    sl: float
    liq_price: float
    trigger: str


@dataclass(slots=True)
class PositionMetrics:
    side: str
    price: float
    entry_price: float
    position_value: float
    fee_close: float
    gross_pnl: float
    margin_used: float | None
    base_sl: float
    base_tp: float
    liquidation_price: float


@dataclass(slots=True)
class TrailDecision:
    event_code: str
    level: str
    side: str
    position_updates: dict[str, Any]
    event_context: dict[str, Any]


@dataclass(slots=True)
class ExitDecision:
    event_code: str
    category: str
    level: str
    reason: str
    side: str
    exit: float
    net_pnl: float
    margin_used: float | None
    gross_pnl: float | None = None
    fee_total: float | None = None
    via_tail_guard: bool = False
    liq_price: float | None = None
    rsi: float | None = None
    threshold: float | None = None


_TRANSIENT_POSITION_FIELDS = {
    "last_price",
    "last_price_updated_at",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "position_notional",
    "margin_used",
    "roi_pct",
    "distance_to_sl_pct",
    "distance_to_tp_pct",
    "updated_at",
    "exchange_position_amt",
    "exchange_entry_price",
    "exchange_unrealized_pnl",
    "exchange_position_side",
    "exchange_position_updated_at",
    "exchange_isolated_margin",
    "close_pending",
    "close_requested_at",
    "pending_close_trade",
    "pending_close_event",
    "pending_close_event_ts",
    "pending_close_runtime_mode",
    "pending_close_strategy_mode",
    "pending_close_order_id",
    "open_pending",
    "open_requested_at",
    "pending_open_event",
    "pending_open_event_ts",
    "pending_open_runtime_mode",
    "pending_open_strategy_mode",
    "pending_open_order_id",
}


LOG_FILE = os.getenv("SCROOGE_LOG_FILE", "runtime/trading_log.txt")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SEARCH_STATUS_LABELS = {
    "looking_for_buy_opportunity": "Looking for a buy opportunity...",
    "looking_for_sell_opportunity": "Looking for a sell opportunity...",
}
REALTIME_WARMUP_LIMITS = {
    "small": 240,
    "medium": 240,
    "big": 240,
}


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


def _tqdm_kwargs(*, desc: str, disable: bool, extra_offset: int = 0) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "desc": _tqdm_desc(desc),
        "disable": disable,
        "dynamic_ncols": True,
    }
    position = _tqdm_position(extra_offset=extra_offset)
    if position is not None:
        kwargs["position"] = position
    return kwargs


def _iter_with_progress(
    iterable: Iterable[Any],
    *,
    show_progress: bool,
    desc: str,
    total: int | None = None,
    progress_reporter: Any | None = None,
) -> Iterable[Any]:
    if progress_reporter is not None:
        def _reported_iter():
            for item in iterable:
                yield item
                progress_reporter.advance()

        return _reported_iter()

    if not show_progress:
        return iterable

    if total is None:
        return tqdm(iterable, **_tqdm_kwargs(desc=desc, disable=False))

    def _tqdm_iter():
        with tqdm(total=total, **_tqdm_kwargs(desc=desc, disable=False)) as progress:
            for item in iterable:
                yield item
                progress.update(1)

    return _tqdm_iter()


def save_log(log_buffer: list[str]) -> None:
    log_path = os.getenv("SCROOGE_LOG_FILE", LOG_FILE)
    with open(log_path, "a") as f:
        f.write("\n".join(log_buffer) + "\n")


def format_event_timestamp(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        dt_value = value.tz_convert(None) if value.tzinfo is not None else value
        return dt_value.strftime(TIMESTAMP_FORMAT)

    if isinstance(value, datetime):
        dt_value = value.replace(tzinfo=None) if value.tzinfo is not None else value
        return dt_value.strftime(TIMESTAMP_FORMAT)

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 10_000_000_000:
            numeric_value /= 1000.0
        try:
            return datetime.fromtimestamp(numeric_value).strftime(TIMESTAMP_FORMAT)
        except (OSError, OverflowError, ValueError):
            return datetime.now().strftime(TIMESTAMP_FORMAT)

    text_value = str(value).strip()
    return text_value or datetime.now().strftime(TIMESTAMP_FORMAT)


def _event_ts_to_timestamp(value: Any) -> pd.Timestamp | None:
    text_value = str(value or "").strip()
    if not text_value:
        return None
    try:
        ts_value = pd.Timestamp(text_value)
    except Exception:  # noqa: BLE001
        return None
    if ts_value.tzinfo is not None:
        ts_value = ts_value.tz_convert(None)
    return ts_value


_EPOCH_ORDINAL = datetime(1970, 1, 1).toordinal()
_NS_PER_SECOND = 1_000_000_000


def _naive_datetime_to_ns(dt_value: datetime) -> int:
    return (
        ((dt_value.toordinal() - _EPOCH_ORDINAL) * 86_400)
        + (dt_value.hour * 3_600)
        + (dt_value.minute * 60)
        + dt_value.second
    ) * _NS_PER_SECOND + (dt_value.microsecond * 1_000)


@lru_cache(maxsize=1024)
def _day_start_ns_from_text(day_text: str) -> int:
    year = int(day_text[0:4])
    month = int(day_text[5:7])
    day = int(day_text[8:10])
    return ((datetime(year, month, day).toordinal() - _EPOCH_ORDINAL) * 86_400) * _NS_PER_SECOND


def _event_ts_to_ns(value: Any) -> int | None:
    if isinstance(value, pd.Timestamp):
        normalized = value.tz_convert(None) if value.tzinfo is not None else value
        return int(normalized.value)

    if isinstance(value, datetime):
        normalized = (
            value.astimezone(timezone.utc).replace(tzinfo=None)
            if value.tzinfo is not None
            else value
        )
        return _naive_datetime_to_ns(normalized)

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 10_000_000_000:
            return int(numeric_value) * 1_000_000
        return int(numeric_value * _NS_PER_SECOND)

    text_value = str(value or "").strip()
    if not text_value:
        return None
    if (
        len(text_value) == 19
        and text_value[4] == "-"
        and text_value[7] == "-"
        and text_value[10] == " "
        and text_value[13] == ":"
        and text_value[16] == ":"
    ):
        return _day_start_ns_from_text(text_value[:10]) + (
            (
                (int(text_value[11:13]) * 3_600)
                + (int(text_value[14:16]) * 60)
                + int(text_value[17:19])
            )
            * _NS_PER_SECOND
        )
    try:
        dt_value = datetime.fromisoformat(text_value)
    except ValueError:
        ts_value = _event_ts_to_timestamp(text_value)
        if ts_value is None:
            return None
        return int(ts_value.value)
    if dt_value.tzinfo is not None:
        dt_value = dt_value.astimezone(timezone.utc).replace(tzinfo=None)
    return _naive_datetime_to_ns(dt_value)


def _interval_to_freq(interval: str) -> str:
    normalized = str(interval or "").strip().lower()
    if normalized.endswith("m"):
        return f"{int(normalized[:-1])}min"
    if normalized.endswith("h"):
        return f"{int(normalized[:-1])}h"
    if normalized.endswith("d"):
        return f"{int(normalized[:-1])}d"
    raise ValueError(f"Unsupported interval for realtime mode: {interval}")


def _empty_market_candle_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])


def _normalize_market_candle_frame(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df.empty:
        return _empty_market_candle_frame()
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open_time", "open", "high", "low", "close"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").tail(limit)
    out.reset_index(drop=True, inplace=True)
    return out[["open_time", "open", "high", "low", "close", "volume"]]


def _closed_event_to_row(event: CandleClosedEvent) -> dict[str, Any]:
    return {
        "open_time": pd.Timestamp(event.open_time),
        "open": float(event.open),
        "high": float(event.high),
        "low": float(event.low),
        "close": float(event.close),
        "volume": float(event.volume),
    }


def _upsert_closed_market_candle(df: pd.DataFrame, event: CandleClosedEvent, *, limit: int) -> pd.DataFrame:
    row_df = pd.DataFrame([_closed_event_to_row(event)])
    merged = pd.concat([df, row_df], ignore_index=True)
    return _normalize_market_candle_frame(merged, limit)


def _build_forming_candle(open_time: pd.Timestamp, price: float) -> dict[str, Any]:
    return {
        "open_time": open_time,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 0.0,
    }


def _update_forming_candle(
    candle: dict[str, Any] | None,
    *,
    open_time: pd.Timestamp,
    price: float,
) -> dict[str, Any]:
    if candle is None or pd.Timestamp(candle["open_time"]) != open_time:
        return _build_forming_candle(open_time, price)

    candle["high"] = max(float(candle["high"]), price)
    candle["low"] = min(float(candle["low"]), price)
    candle["close"] = price
    return candle


def _materialize_market_frame(
    closed_df: pd.DataFrame,
    forming_candle: dict[str, Any] | None,
    *,
    limit: int,
) -> pd.DataFrame:
    if forming_candle is None:
        return _normalize_market_candle_frame(closed_df, limit)
    row_df = pd.DataFrame([forming_candle])
    merged = pd.concat([closed_df, row_df], ignore_index=True)
    return _normalize_market_candle_frame(merged, limit)


def _build_realtime_snapshot(
    *,
    event_ts: str,
    feature_engine: FeatureEngine,
    selection_plan: IndicatorSelectionPlan,
    discrete_indicator_values: dict[str, float | None] | None = None,
    current_price: float | None = None,
) -> DiscreteRowSnapshot | None:
    realtime_indicator_values = None
    if selection_plan.requires_realtime:
        realtime_indicator_values = feature_engine.realtime_values(
            selected_keys=selection_plan.realtime_keys,
        )
    discrete_values = discrete_indicator_values
    if selection_plan.requires_discrete and discrete_values is None:
        return None
    if selection_plan.requires_realtime and realtime_indicator_values is None:
        return None
    price = current_price
    if price is None:
        price = feature_engine.current_price()
    if price is None:
        return None
    lower_source = realtime_indicator_values if selection_plan.bbl_mode == "intrabar" else discrete_values
    mid_source = realtime_indicator_values if selection_plan.bbm_mode == "intrabar" else discrete_values
    upper_source = realtime_indicator_values if selection_plan.bbu_mode == "intrabar" else discrete_values
    atr_source = realtime_indicator_values if selection_plan.atr_mode == "intrabar" else discrete_values
    rsi_source = realtime_indicator_values if selection_plan.rsi_mode == "intrabar" else discrete_values
    ema_source = realtime_indicator_values if selection_plan.ema_mode == "intrabar" else discrete_values
    if (
        lower_source is None
        or mid_source is None
        or upper_source is None
        or atr_source is None
        or rsi_source is None
        or ema_source is None
    ):
        return None
    lower = lower_source.get("BBL")
    mid = mid_source.get("BBM")
    upper = upper_source.get("BBU")
    atr = atr_source.get("ATR")
    rsi = rsi_source.get("RSI")
    ema = ema_source.get("EMA")
    if lower is None or mid is None or upper is None or atr is None or rsi is None or ema is None:
        return None
    return DiscreteRowSnapshot(
        raw_row=None,
        price=price,
        lower=lower,
        upper=upper,
        mid=mid,
        atr=atr,
        rsi=rsi,
        ema=ema,
        row_ts=event_ts,
        log_ts=event_ts,
    )


def initialize_realtime_strategy_processor(
    *,
    live: bool,
    initial_balance: float,
    qty: float | None,
    sl_mult: float,
    tp_mult: float,
    symbol: str,
    leverage: float,
    use_full_balance: bool,
    fee_rate: float,
    state: dict[str, Any] | None,
    use_state: bool,
    enable_logs: bool,
    rsi_extreme_long: float,
    rsi_extreme_short: float,
    rsi_long_open_threshold: float,
    rsi_long_qty_threshold: float,
    rsi_long_tp_threshold: float,
    rsi_long_close_threshold: float,
    rsi_short_open_threshold: float,
    rsi_short_qty_threshold: float,
    rsi_short_tp_threshold: float,
    rsi_short_close_threshold: float,
    trail_atr_mult: float,
    allow_entries: bool,
    execution_mode: str,
    runtime_mode: str | None,
    indicator_inputs: dict[str, str] | None,
    intervals: dict[str, str],
    target_symbol: str | None = None,
    emit_on_price_tick: bool = True,
) -> RealtimeStrategyProcessor:
    normalized_runtime_mode = str(
        runtime_mode or os.getenv("SCROOGE_RUNTIME_MODE", "live" if live else "backtest")
    ).strip().lower() or ("live" if live else "backtest")
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        load_state_fn=load_state,
        provided_state=state,
    )
    runtime.enable_logs = enable_logs
    normalized_indicator_inputs = normalize_indicator_inputs(
        indicator_inputs,
        strategy_mode="realtime",
    )
    config = StrategyConfig(
        qty=qty,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        symbol=symbol,
        leverage=leverage,
        use_full_balance=use_full_balance,
        fee_rate=fee_rate,
        rsi_extreme_long=rsi_extreme_long,
        rsi_extreme_short=rsi_extreme_short,
        rsi_long_open_threshold=rsi_long_open_threshold,
        rsi_long_qty_threshold=rsi_long_qty_threshold,
        rsi_long_tp_threshold=rsi_long_tp_threshold,
        rsi_long_close_threshold=rsi_long_close_threshold,
        rsi_short_open_threshold=rsi_short_open_threshold,
        rsi_short_qty_threshold=rsi_short_qty_threshold,
        rsi_short_tp_threshold=rsi_short_tp_threshold,
        rsi_short_close_threshold=rsi_short_close_threshold,
        trail_atr_mult=trail_atr_mult,
        allow_entries=allow_entries,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        runtime_mode=normalized_runtime_mode,
        strategy_mode="realtime",
        indicator_inputs=normalized_indicator_inputs,
    )
    technical_logger = get_technical_logger()

    def on_row(snapshot: DiscreteRowSnapshot, row_runtime: StrategyRuntime) -> None:
        config.allow_entries = bool(row_runtime.state.get("trading_enabled", True))
        process_discrete_row(snapshot, row_runtime, config, technical_logger)

    return RealtimeStrategyProcessor(
        runtime=runtime,
        feature_engine=FeatureEngine(
            intervals=intervals,
            limits=REALTIME_WARMUP_LIMITS,
        ),
        selection_plan=indicator_selection_plan(normalized_indicator_inputs),
        emit_on_price_tick=emit_on_price_tick,
        on_row=on_row,
        target_symbol=str(target_symbol or symbol or "").strip().upper() or None,
        intervals=dict(intervals),
        latest_discrete_indicator_values=None,
        pending_small_candles={},
        small_interval_ns=interval_to_ns(intervals["small"]),
    )


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key)
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, key):
        return getattr(row, key)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key)
    return None


def _snapshot_indicator_value(row: Any, column: str, *, indicator_inputs: dict[str, str] | None) -> Any:
    mode = indicator_input_mode_for_column(indicator_inputs, column)
    if mode == "realtime":
        realtime_key = f"{column}_RT"
        realtime_value = _row_value(row, realtime_key)
        if realtime_value is not None:
            return realtime_value
    return _row_value(row, column)


def _indicator_values_from_snapshot_event(event: IndicatorSnapshotEvent) -> dict[str, float | None]:
    return {
        column: to_float(event.values.get(column))
        for column in INDICATOR_COLUMNS
    }


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio_to_percent(numerator: Any, denominator: Any) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * 100.0


def status_from_code(code: str | None, labels: dict[str, str]) -> dict[str, str] | None:
    label = labels.get(code or "")
    if label is None:
        return None
    return {"code": code or "", "label": label}


_SEARCH_STATUS_BUY_CODE = "looking_for_buy_opportunity"
_SEARCH_STATUS_SELL_CODE = "looking_for_sell_opportunity"
_SEARCH_STATUS_BUY_LABEL = SEARCH_STATUS_LABELS[_SEARCH_STATUS_BUY_CODE]
_SEARCH_STATUS_SELL_LABEL = SEARCH_STATUS_LABELS[_SEARCH_STATUS_SELL_CODE]


def resolve_search_status(price: Any, ema: Any, previous_status: Any) -> dict[str, str] | None:
    price_value = to_float(price)
    ema_value = to_float(ema)
    previous_code = None
    if isinstance(previous_status, dict):
        raw_code = str(previous_status.get("code", "")).strip()
        if raw_code in SEARCH_STATUS_LABELS:
            previous_code = raw_code

    if price_value is None or ema_value is None:
        return previous_status if previous_code else None

    if price_value > ema_value:
        if previous_code == _SEARCH_STATUS_BUY_CODE:
            return previous_status
        return {"code": _SEARCH_STATUS_BUY_CODE, "label": _SEARCH_STATUS_BUY_LABEL}
    if price_value < ema_value:
        if previous_code == _SEARCH_STATUS_SELL_CODE:
            return previous_status
        return {"code": _SEARCH_STATUS_SELL_CODE, "label": _SEARCH_STATUS_SELL_LABEL}
    if previous_code:
        return previous_status
    return {"code": _SEARCH_STATUS_BUY_CODE, "label": _SEARCH_STATUS_BUY_LABEL}


def _clear_position_runtime_metrics(position: dict[str, Any]) -> None:
    position["unrealized_pnl"] = None
    position["unrealized_pnl_pct"] = None
    position["position_notional"] = None
    position["margin_used"] = None
    position["roi_pct"] = None
    position["distance_to_sl_pct"] = None
    position["distance_to_tp_pct"] = None


def _to_float_fast(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return to_float(value)


def normalize_manual_trade_suggestion(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None

    side = str(value.get("side", "")).strip().lower()
    if side not in {"buy", "sell"}:
        return None

    return {
        "side": side,
        "requested_at": str(value.get("requested_at", "")).strip() or None,
        "requested_by": str(value.get("requested_by", "")).strip() or None,
    }


def refresh_position_snapshot(position: dict[str, Any], price: Any, leverage: Any, ts_label: str) -> None:
    if "last_price" in position:
        del position["last_price"]
    if "last_price_updated_at" in position:
        del position["last_price_updated_at"]

    if position.get("entry_time") is None and position.get("time") is not None:
        position["entry_time"] = position.get("time")
    if position.get("time") is None and position.get("entry_time") is not None:
        position["time"] = position.get("entry_time")

    side_raw = position.get("side", "")
    side = side_raw if side_raw in {"long", "short"} else str(side_raw).strip().lower()
    size = _to_float_fast(position.get("size")) or 0.0
    entry_price = _to_float_fast(position.get("entry"))
    mark_price = _to_float_fast(price)
    sl_price = _to_float_fast(position.get("sl"))
    tp_price = _to_float_fast(position.get("tp"))

    if mark_price is None:
        return

    position["updated_at"] = ts_label

    if entry_price is None or entry_price <= 0 or size <= 0:
        _clear_position_runtime_metrics(position)
        return

    notional = abs(size) * entry_price
    if side == "short":
        unrealized_pnl = (entry_price - mark_price) * size
    else:
        unrealized_pnl = (mark_price - entry_price) * size

    leverage_value = _to_float_fast(leverage)
    margin_used = (notional / leverage_value) if leverage_value and leverage_value > 0 else None
    unrealized_pnl_pct = ratio_to_percent(unrealized_pnl, notional)
    roi_pct = ratio_to_percent(unrealized_pnl, margin_used) if margin_used else None

    if side == "short":
        distance_to_sl_pct = ratio_to_percent((sl_price - mark_price), mark_price) if sl_price is not None else None
        distance_to_tp_pct = ratio_to_percent((mark_price - tp_price), mark_price) if tp_price is not None else None
    else:
        distance_to_sl_pct = ratio_to_percent((mark_price - sl_price), mark_price) if sl_price is not None else None
        distance_to_tp_pct = ratio_to_percent((tp_price - mark_price), mark_price) if tp_price is not None else None

    position["unrealized_pnl"] = unrealized_pnl
    position["unrealized_pnl_pct"] = unrealized_pnl_pct
    position["position_notional"] = notional
    position["margin_used"] = margin_used
    position["roi_pct"] = roi_pct
    position["distance_to_sl_pct"] = distance_to_sl_pct
    position["distance_to_tp_pct"] = distance_to_tp_pct


def refresh_market_snapshot(state: dict[str, Any], price: Any, ts_label: str) -> None:
    market_price = to_float(price)
    if market_price is None:
        return
    state["last_price"] = market_price
    state["last_price_updated_at"] = ts_label
    state["updated_at"] = ts_label


def _normalized_symbol(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _normalized_position_side(value: Any, position_amt: float | None = None) -> str | None:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    amount = to_float(position_amt)
    if amount is None:
        return None
    if amount > 0:
        return "LONG"
    if amount < 0:
        return "SHORT"
    return None


def _ensure_execution_sync_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("execution_sync")
    if not isinstance(payload, dict):
        payload = {}
        state["execution_sync"] = payload
    return payload


def _sync_execution_context_state(runtime: StrategyRuntime) -> None:
    payload = _ensure_execution_sync_state(runtime.state)
    payload.update(asdict(runtime.execution_sync))


def _resolve_position_alignment(
    runtime_position: dict[str, Any] | None,
    *,
    symbol: str,
    position_amt: float,
    position_side: str | None,
    entry_price: float | None,
) -> str:
    has_exchange_position = abs(position_amt) > 1e-12
    if runtime_position is None:
        return "exchange_only_position" if has_exchange_position else "aligned"

    if not has_exchange_position:
        return "runtime_only_position"

    runtime_side = str(runtime_position.get("side", "")).strip().lower()
    exchange_side = str(position_side or "").strip().lower()
    if exchange_side == "long" and runtime_side != "long":
        return "side_mismatch"
    if exchange_side == "short" and runtime_side != "short":
        return "side_mismatch"

    runtime_entry = to_float(runtime_position.get("entry"))
    if runtime_entry is not None and entry_price is not None:
        allowed_drift = max(1.0, abs(runtime_entry) * 0.0005)
        if abs(runtime_entry - entry_price) > allowed_drift:
            return "entry_mismatch"

    runtime_symbol = _normalized_symbol(runtime_position.get("symbol"))
    if runtime_symbol and runtime_symbol != symbol:
        return "symbol_mismatch"
    return "aligned"


def _apply_account_balance_market_event(runtime: StrategyRuntime, event: AccountBalanceEvent) -> None:
    runtime.execution_sync.total_events += 1
    runtime.execution_sync.account_balance_events += 1
    runtime.execution_sync.last_event_type = event.event_type
    runtime.execution_sync.last_event_ts = event.ts

    state = runtime.state
    state["exchange_balance"] = {
        "asset": str(event.asset).strip().upper(),
        "wallet_balance": float(event.wallet_balance),
        "cross_wallet_balance": (float(event.cross_wallet_balance) if event.cross_wallet_balance is not None else None),
        "balance_delta": (float(event.balance_delta) if event.balance_delta is not None else None),
        "updated_at": event.ts,
        "source": event.source,
    }

    if str(event.asset).strip().upper() == "USDT":
        drift = float(event.wallet_balance) - float(runtime.balance)
        runtime.execution_sync.balance_drift = drift
        runtime.execution_sync.balance_alignment = "aligned" if abs(drift) <= 1e-6 else "drifted"
        if runtime.execution_mode == "observed":
            runtime.balance = float(event.wallet_balance)
            state["balance"] = runtime.balance

    _sync_execution_context_state(runtime)


def _apply_position_snapshot_market_event(
    runtime: StrategyRuntime,
    event: PositionSnapshotEvent,
    *,
    target_symbol: str | None,
) -> None:
    runtime.execution_sync.total_events += 1
    runtime.execution_sync.position_snapshot_events += 1
    runtime.execution_sync.last_event_type = event.event_type
    runtime.execution_sync.last_event_ts = event.ts

    symbol = _normalized_symbol(event.symbol)
    if target_symbol is not None and symbol != target_symbol:
        _sync_execution_context_state(runtime)
        return

    position_amt = float(event.position_amt)
    position_side = _normalized_position_side(event.position_side, position_amt)
    alignment = _resolve_position_alignment(
        runtime.position if isinstance(runtime.position, dict) else None,
        symbol=symbol or "",
        position_amt=position_amt,
        position_side=position_side,
        entry_price=(float(event.entry_price) if event.entry_price is not None else None),
    )
    runtime.execution_sync.position_alignment = alignment
    if alignment != "aligned":
        runtime.execution_sync.divergence_events += 1

    runtime.state["exchange_position"] = {
        "symbol": symbol,
        "position_amt": position_amt,
        "entry_price": (float(event.entry_price) if event.entry_price is not None else None),
        "unrealized_pnl": (float(event.unrealized_pnl) if event.unrealized_pnl is not None else None),
        "position_side": position_side,
        "updated_at": event.ts,
        "source": event.source,
        "alignment": alignment,
    }

    if runtime.execution_mode == "observed":
        if abs(position_amt) <= 1e-12:
            runtime.position = None
            runtime.state["position"] = None
        else:
            observed_position = {
                "symbol": symbol,
                "side": ("long" if position_side == "LONG" else "short") if position_side in {"LONG", "SHORT"} else None,
                "size": abs(position_amt),
                "entry": (float(event.entry_price) if event.entry_price is not None else None),
                "sl": None,
                "tp": None,
                "liq_price": None,
                "trail_active": False,
                "trail_price": None,
                "time": event.ts,
                "entry_time": event.ts,
                "observed_execution_only": True,
                "exchange_position_amt": position_amt,
                "exchange_entry_price": (float(event.entry_price) if event.entry_price is not None else None),
                "exchange_unrealized_pnl": (float(event.unrealized_pnl) if event.unrealized_pnl is not None else None),
                "exchange_position_side": position_side,
                "exchange_position_updated_at": event.ts,
                "updated_at": event.ts,
            }
            runtime.position = observed_position
            runtime.state["position"] = observed_position

    if isinstance(runtime.position, dict):
        runtime.position["exchange_position_amt"] = position_amt
        runtime.position["exchange_entry_price"] = (
            float(event.entry_price) if event.entry_price is not None else None
        )
        runtime.position["exchange_unrealized_pnl"] = (
            float(event.unrealized_pnl) if event.unrealized_pnl is not None else None
        )
        runtime.position["exchange_position_side"] = position_side
        runtime.position["exchange_position_updated_at"] = event.ts

    _sync_execution_context_state(runtime)


def _apply_order_trade_update_market_event(
    runtime: StrategyRuntime,
    event: OrderTradeUpdateEvent,
    *,
    target_symbol: str | None,
) -> None:
    runtime.execution_sync.total_events += 1
    runtime.execution_sync.order_trade_update_events += 1
    runtime.execution_sync.last_event_type = event.event_type
    runtime.execution_sync.last_event_ts = event.ts

    symbol = _normalized_symbol(event.symbol)
    if target_symbol is not None and symbol != target_symbol:
        _sync_execution_context_state(runtime)
        return

    execution_type = str(event.execution_type or "").strip().upper() or None
    order_status = str(event.order_status or "").strip().upper() or None
    if execution_type == "TRADE":
        runtime.execution_sync.trade_execution_events += 1
    if order_status == "FILLED":
        runtime.execution_sync.filled_order_events += 1
    if event.realized_pnl is not None:
        runtime.execution_sync.realized_pnl_total += float(event.realized_pnl)
    if event.commission is not None:
        runtime.execution_sync.commission_total += float(event.commission)

    runtime.state["last_order_trade_update"] = {
        "symbol": symbol,
        "order_side": (str(event.order_side).strip().upper() if event.order_side else None),
        "order_type": (str(event.order_type).strip().upper() if event.order_type else None),
        "execution_type": execution_type,
        "order_status": order_status,
        "order_id": event.order_id,
        "trade_id": event.trade_id,
        "last_filled_qty": (float(event.last_filled_qty) if event.last_filled_qty is not None else None),
        "accumulated_filled_qty": (
            float(event.accumulated_filled_qty) if event.accumulated_filled_qty is not None else None
        ),
        "last_filled_price": (float(event.last_filled_price) if event.last_filled_price is not None else None),
        "average_price": (float(event.average_price) if event.average_price is not None else None),
        "realized_pnl": (float(event.realized_pnl) if event.realized_pnl is not None else None),
        "commission_asset": (str(event.commission_asset).strip().upper() if event.commission_asset else None),
        "commission": (float(event.commission) if event.commission is not None else None),
        "reduce_only": bool(event.reduce_only),
        "updated_at": event.ts,
        "source": event.source,
    }

    _sync_execution_context_state(runtime)


def apply_market_event_runtime_sync(
    runtime: StrategyRuntime,
    event: MarketEvent,
    *,
    target_symbol: str | None,
) -> bool:
    event_type = getattr(event, "event_type", "")
    if event_type == "account_balance":
        _apply_account_balance_market_event(runtime, event)
        return True
    if event_type == "position_snapshot":
        _apply_position_snapshot_market_event(runtime, event, target_symbol=target_symbol)
        return True
    if event_type == "order_trade_update":
        _apply_order_trade_update_market_event(runtime, event, target_symbol=target_symbol)
        return True
    return False


def refresh_runtime_state_from_price_tick(state, last_price, position_price, leverage, ts_label):
    refresh_market_snapshot(state, last_price, ts_label)

    position = state.get("position")
    if isinstance(position, dict):
        snapshot_price = to_float(position_price)
        if snapshot_price is not None:
            refresh_position_snapshot(position, snapshot_price, leverage, ts_label)


def consume_manual_trade_suggestion(state: dict[str, Any], live: bool) -> dict[str, str | None] | None:
    suggestion = normalize_manual_trade_suggestion(state.get("manual_trade_suggestion"))
    if suggestion is None:
        return None
    state["manual_trade_suggestion"] = None
    if live:
        save_state(state)
    return suggestion


def initialize_strategy_runtime(
    *,
    live: bool,
    initial_balance: float,
    use_state: bool,
    execution_mode: str,
    load_state_fn: Callable[..., dict[str, Any]],
    provided_state: dict[str, Any] | None = None,
) -> StrategyRuntime:
    if provided_state is not None:
        state = provided_state
    elif use_state:
        state = load_state_fn(include_history=not live)
    else:
        state = {
            "position": None,
            "trade_history": [],
            "balance_history": [],
            "manual_trade_suggestion": None,
            "updated_at": None,
            "search_status": None,
            "bot_status": None,
            "trade_status": None,
            "session_start": None,
            "session_end": None,
        }

    return StrategyRuntime(
        state=state,
        balance=initial_balance,
        position=state["position"],
        trade_history=state.get("trade_history", []),
        balance_history=state.get("balance_history", []),
        log_buffer=[],
        live=live,
        use_state=use_state,
        enable_logs=True,
        execution_mode=execution_mode,
        execution_sync=ExecutionSyncContext(),
        execution_events=[],
    )


def _append_execution_event(runtime: StrategyRuntime, event: MarketEvent) -> None:
    runtime.execution_events.append(event)


def _simulated_position_amt(position: dict[str, Any]) -> float:
    size = float(position.get("size") or 0.0)
    side = str(position.get("side", "")).strip().lower()
    return size if side == "long" else -size


def _position_entry_fee(position: dict[str, Any], fee_rate: float) -> float:
    size = float(position.get("size") or 0.0)
    entry_price = float(position.get("entry") or 0.0)
    if fee_rate <= 0 or size <= 0 or entry_price <= 0:
        return 0.0
    return size * entry_price * fee_rate


def _resolve_backtest_exit_balance_delta(
    position: dict[str, Any],
    decision: ExitDecision,
    *,
    fee_rate: float,
) -> float:
    entry_fee = _position_entry_fee(position, fee_rate)
    if decision.gross_pnl is not None:
        return float(decision.gross_pnl) - entry_fee
    return float(decision.net_pnl) + entry_fee


def emit_simulated_entry_execution_events(
    runtime: StrategyRuntime,
    config: StrategyConfig,
    position: dict[str, Any],
    *,
    ts: str,
    balance_after: float,
) -> None:
    size = float(position.get("size") or 0.0)
    entry_price = float(position.get("entry") or 0.0)
    open_fee = _position_entry_fee(position, config.fee_rate)
    order_side = "BUY" if str(position.get("side", "")).strip().lower() == "long" else "SELL"

    _append_execution_event(
        runtime,
        OrderTradeUpdateEvent(
            symbol=config.symbol,
            ts=ts,
            order_side=order_side,
            order_type="MARKET",
            execution_type="TRADE",
            order_status="FILLED",
            last_filled_qty=size,
            accumulated_filled_qty=size,
            last_filled_price=entry_price,
            average_price=entry_price,
            realized_pnl=0.0,
            commission_asset="USDT",
            commission=open_fee,
            reduce_only=False,
            source="simulated_execution",
        ),
    )
    _append_execution_event(
        runtime,
        PositionSnapshotEvent(
            symbol=config.symbol,
            ts=ts,
            position_amt=_simulated_position_amt(position),
            entry_price=entry_price,
            unrealized_pnl=0.0,
            position_side=str(position.get("side", "")).strip().upper(),
            source="simulated_execution",
        ),
    )
    _append_execution_event(
        runtime,
        AccountBalanceEvent(
            asset="USDT",
            ts=ts,
            wallet_balance=balance_after,
            cross_wallet_balance=balance_after,
            balance_delta=-open_fee,
            source="simulated_execution",
        ),
    )


def emit_simulated_exit_execution_events(
    runtime: StrategyRuntime,
    config: StrategyConfig,
    position: dict[str, Any],
    decision: ExitDecision,
    *,
    ts: str,
    balance_after: float,
) -> None:
    size = float(position.get("size") or 0.0)
    entry_price = float(position.get("entry") or 0.0)
    exit_price = float(decision.exit)
    exit_fee = _position_entry_fee(position, config.fee_rate)
    balance_delta = _resolve_backtest_exit_balance_delta(
        position,
        decision,
        fee_rate=config.fee_rate,
    )
    order_side = "SELL" if str(position.get("side", "")).strip().lower() == "long" else "BUY"
    realized_pnl = decision.gross_pnl if decision.gross_pnl is not None else decision.net_pnl

    _append_execution_event(
        runtime,
        OrderTradeUpdateEvent(
            symbol=config.symbol,
            ts=ts,
            order_side=order_side,
            order_type="MARKET",
            execution_type="TRADE",
            order_status="FILLED",
            last_filled_qty=size,
            accumulated_filled_qty=size,
            last_filled_price=exit_price,
            average_price=exit_price,
            realized_pnl=realized_pnl,
            commission_asset="USDT",
            commission=exit_fee,
            reduce_only=True,
            source="simulated_execution",
        ),
    )
    _append_execution_event(
        runtime,
        PositionSnapshotEvent(
            symbol=config.symbol,
            ts=ts,
            position_amt=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            position_side=None,
            source="simulated_execution",
        ),
    )
    _append_execution_event(
        runtime,
        AccountBalanceEvent(
            asset="USDT",
            ts=ts,
            wallet_balance=balance_after,
            cross_wallet_balance=balance_after,
            balance_delta=balance_delta,
            source="simulated_execution",
        ),
    )


def iter_discrete_rows(rows: Any, *, live: bool, show_progress: bool, progress_reporter: Any | None = None) -> Any:
    if isinstance(rows, pd.DataFrame):
        if live:
            return [rows.iloc[-1]]
        df_iter = [rows.iloc[i] for i in range(1, len(rows))]
        return _iter_with_progress(
            df_iter,
            show_progress=show_progress,
            desc="Backtest Progress",
            total=len(df_iter),
            progress_reporter=progress_reporter,
        )

    row_list = list(rows)
    if live:
        return row_list[-1:] if row_list else []

    tape_iter = row_list[1:]
    return _iter_with_progress(
        tape_iter,
        show_progress=show_progress,
        desc="Backtest Progress",
        total=len(tape_iter),
        progress_reporter=progress_reporter,
    )


def build_row_snapshot(
    row: Any,
    *,
    live: bool,
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
    indicator_inputs: dict[str, str] | None = None,
) -> DiscreteRowSnapshot:
    row_ts = timestamp_formatter(_row_value(row, "open_time"))
    log_ts = datetime.now().strftime(timestamp_format) if live else row_ts
    return DiscreteRowSnapshot(
        raw_row=row,
        price=_row_value(row, "close"),
        lower=_snapshot_indicator_value(row, "BBL", indicator_inputs=indicator_inputs),
        upper=_snapshot_indicator_value(row, "BBU", indicator_inputs=indicator_inputs),
        mid=_snapshot_indicator_value(row, "BBM", indicator_inputs=indicator_inputs),
        atr=_snapshot_indicator_value(row, "ATR", indicator_inputs=indicator_inputs),
        rsi=_snapshot_indicator_value(row, "RSI", indicator_inputs=indicator_inputs),
        ema=_snapshot_indicator_value(row, "EMA", indicator_inputs=indicator_inputs),
        row_ts=row_ts,
        log_ts=log_ts,
    )


def calc_liquidation_price(entry: float, leverage_value: float, side: str) -> float:
    if side == "long":
        return entry * (1 - 1 / leverage_value)
    return entry * (1 + 1 / leverage_value)


def resolve_entry_decision(
    snapshot: DiscreteRowSnapshot,
    *,
    config: StrategyConfig,
    qty_local: float,
    manual_side: str | None,
) -> EntryDecision | EntryGuardRejection | None:
    price = snapshot.price
    lower = snapshot.lower
    upper = snapshot.upper
    atr = snapshot.atr
    rsi = snapshot.rsi
    ema = snapshot.ema

    manual_long = manual_side == "buy"
    manual_short = manual_side == "sell"

    if (price < lower and rsi < config.rsi_long_open_threshold and price > ema) or manual_long:
        size = qty_local * 0.5 if rsi > config.rsi_long_qty_threshold else qty_local
        sl = price - atr * config.sl_mult
        tp = price + atr * config.tp_mult
        liq_price = calc_liquidation_price(price, config.leverage, "long")
        trigger = "manual_suggestion" if manual_long else "strategy_rules"
        if sl < liq_price:
            return EntryGuardRejection(
                side="long",
                entry=price,
                sl=sl,
                liq_price=liq_price,
                trigger=trigger,
            )
        return EntryDecision(
            side="long",
            size=size,
            entry=price,
            sl=sl,
            tp=tp,
            liq_price=liq_price,
            stake_mode="half" if rsi > config.rsi_long_qty_threshold else "full",
            trigger=trigger,
        )

    if (price > upper and rsi > config.rsi_short_open_threshold and price < ema) or manual_short:
        size = qty_local * 0.5 if rsi < config.rsi_short_qty_threshold else qty_local
        sl = price + atr * config.sl_mult
        tp = price - atr * config.tp_mult
        liq_price = calc_liquidation_price(price, config.leverage, "short")
        trigger = "manual_suggestion" if manual_short else "strategy_rules"
        if sl > liq_price:
            return EntryGuardRejection(
                side="short",
                entry=price,
                sl=sl,
                liq_price=liq_price,
                trigger=trigger,
            )
        return EntryDecision(
            side="short",
            size=size,
            entry=price,
            sl=sl,
            tp=tp,
            liq_price=liq_price,
            stake_mode="half" if rsi < config.rsi_short_qty_threshold else "full",
            trigger=trigger,
        )

    return None


def build_position_from_entry(decision: EntryDecision, *, row_ts: str) -> dict[str, Any]:
    return {
        "side": decision.side,
        "size": decision.size,
        "entry": decision.entry,
        "sl": decision.sl,
        "tp": decision.tp,
        "liq_price": decision.liq_price,
        "trail_active": False,
        "trail_price": None,
        "time": row_ts,
        "entry_time": row_ts,
        "trigger": decision.trigger,
        "stake_mode": decision.stake_mode,
    }


def _execution_summary_ts(summary: dict[str, Any] | None, fallback: str) -> str:
    if not isinstance(summary, dict):
        return fallback
    return format_event_timestamp(summary.get("update_time_ms")) if summary.get("update_time_ms") is not None else fallback


def _apply_entry_execution_summary(
    position: dict[str, Any],
    summary: dict[str, Any] | None,
    *,
    side: str,
    fallback_ts: str,
) -> None:
    if not isinstance(summary, dict):
        return

    avg_price = to_float(summary.get("avg_price"))
    executed_qty = to_float(summary.get("executed_qty"))
    commission = to_float(summary.get("commission"))
    order_id = summary.get("order_id")
    ts_label = _execution_summary_ts(summary, fallback=fallback_ts)

    if avg_price is not None and avg_price > 0:
        position["entry"] = avg_price
        position["exchange_entry_price"] = avg_price
    if executed_qty is not None and executed_qty > 0:
        position["size"] = executed_qty
        position["exchange_position_amt"] = executed_qty if side == "long" else -executed_qty
    if commission is not None:
        position["entry_fee_paid"] = commission
    if order_id is not None:
        position["entry_order_id"] = order_id
    position["entry_time"] = ts_label
    position["time"] = ts_label
    position["entry_execution_time"] = ts_label
    position["exchange_position_side"] = "LONG" if side == "long" else "SHORT"
    position["exchange_position_updated_at"] = ts_label


def _apply_exit_execution_summary(
    trade: dict[str, Any],
    position: dict[str, Any],
    summary: dict[str, Any] | None,
    *,
    fee_rate: float,
) -> float:
    entry_fee_paid = to_float(position.get("entry_fee_paid"))
    if entry_fee_paid is None:
        entry_fee_paid = _position_entry_fee(position, fee_rate)

    if not isinstance(summary, dict):
        return float(trade["net_pnl"])

    avg_price = to_float(summary.get("avg_price"))
    close_commission = to_float(summary.get("commission"))
    realized_pnl = to_float(summary.get("realized_pnl"))
    executed_qty = to_float(summary.get("executed_qty"))
    order_id = summary.get("order_id")
    execution_ts = _execution_summary_ts(summary, fallback=str(trade.get("exit_time") or ""))

    if avg_price is not None and avg_price > 0:
        trade["exit"] = avg_price
    if executed_qty is not None and executed_qty > 0:
        trade["exit_execution_qty"] = executed_qty
    if order_id is not None:
        trade["exit_order_id"] = order_id
    trade["exit_time"] = execution_ts
    trade["exit_execution_time"] = execution_ts
    if entry_fee_paid is not None:
        trade["entry_fee"] = entry_fee_paid
    if close_commission is not None:
        trade["exit_fee"] = close_commission

    gross_pnl = realized_pnl if realized_pnl is not None else to_float(trade.get("gross_pnl"))
    if gross_pnl is not None:
        trade["gross_pnl"] = gross_pnl

    total_fee = None
    if entry_fee_paid is not None and close_commission is not None:
        total_fee = entry_fee_paid + close_commission
    elif to_float(trade.get("fee")) is not None:
        total_fee = to_float(trade.get("fee"))

    if total_fee is not None:
        trade["fee"] = total_fee

    if gross_pnl is not None and total_fee is not None:
        trade["net_pnl"] = gross_pnl - total_fee
    elif gross_pnl is not None and close_commission is None and entry_fee_paid is not None:
        trade["net_pnl"] = gross_pnl - entry_fee_paid

    return float(trade["net_pnl"])


def sanitize_trade_for_history(trade: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in trade.items():
        if key in _TRANSIENT_POSITION_FIELDS:
            continue
        sanitized[key] = value
    return sanitized


def _mark_position_close_pending(
    position: dict[str, Any],
    *,
    trade: dict[str, Any],
    event_payload: dict[str, Any],
    event_ts: str,
    runtime_mode: str,
    strategy_mode: str | None,
    order_id: Any = None,
) -> dict[str, Any]:
    pending_position = dict(position)
    pending_position["close_pending"] = True
    pending_position["close_requested_at"] = event_ts
    pending_position["pending_close_trade"] = sanitize_trade_for_history(trade)
    pending_position["pending_close_event"] = dict(event_payload)
    pending_position["pending_close_event_ts"] = str(trade.get("exit_time") or event_ts)
    pending_position["pending_close_runtime_mode"] = runtime_mode
    pending_position["pending_close_strategy_mode"] = strategy_mode
    if order_id is not None:
        pending_position["pending_close_order_id"] = order_id
    return pending_position


def _mark_position_open_pending(
    position: dict[str, Any],
    *,
    event_payload: dict[str, Any],
    event_ts: str,
    runtime_mode: str,
    strategy_mode: str | None,
    order_id: Any = None,
) -> dict[str, Any]:
    pending_position = dict(position)
    pending_position["open_pending"] = True
    pending_position["open_requested_at"] = event_ts
    pending_position["pending_open_event"] = dict(event_payload)
    pending_position["pending_open_event_ts"] = event_ts
    pending_position["pending_open_runtime_mode"] = runtime_mode
    pending_position["pending_open_strategy_mode"] = strategy_mode
    if order_id is not None:
        pending_position["pending_open_order_id"] = order_id
    return pending_position


def build_exit_trade(
    position: dict[str, Any],
    decision: ExitDecision,
    *,
    row_ts: str,
) -> dict[str, Any]:
    trade = {
        **position,
        "exit": decision.exit,
        "exit_time": row_ts,
        "exit_reason": decision.reason,
        "net_pnl": decision.net_pnl,
    }
    if decision.gross_pnl is not None:
        trade["gross_pnl"] = decision.gross_pnl
    if decision.fee_total is not None:
        trade["fee"] = decision.fee_total
    if decision.via_tail_guard:
        trade["via_tail_guard"] = True
    if decision.rsi is not None:
        trade["exit_rsi"] = decision.rsi
    if decision.threshold is not None:
        trade["exit_threshold"] = decision.threshold
    return trade


def apply_backtest_exit_transition(
    balance: float,
    trade_history: list[dict[str, Any]],
    position: dict[str, Any],
    decision: ExitDecision,
    *,
    row_ts: str,
    fee_rate: float,
) -> tuple[float, dict[str, Any]]:
    trade = build_exit_trade(position, decision, row_ts=row_ts)
    updated_balance = balance + _resolve_backtest_exit_balance_delta(
        position,
        decision,
        fee_rate=fee_rate,
    )
    sanitized_trade = sanitize_trade_for_history(trade)
    trade_history.append(sanitized_trade)
    return updated_balance, sanitized_trade


def build_entry_guard_event_payload(
    rejection: EntryGuardRejection,
    *,
    symbol: str,
) -> dict[str, Any]:
    return {
        "code": "entry_skipped_liquidation_guard",
        "category": "risk",
        "level": "warning",
        "symbol": symbol,
        "side": rejection.side,
        "entry": rejection.entry,
        "sl": rejection.sl,
        "liq_price": rejection.liq_price,
        "trigger": rejection.trigger,
    }


def build_trade_opened_event_payload(
    decision: EntryDecision,
    *,
    symbol: str,
    leverage: float,
    fee: float,
    rsi: float,
) -> dict[str, Any]:
    return {
        "code": "trade_opened",
        "category": "trade",
        "level": "info",
        "notify": True,
        "symbol": symbol,
        "side": decision.side,
        "entry": decision.entry,
        "sl": decision.sl,
        "tp": decision.tp,
        "size": decision.size,
        "leverage": leverage,
        "fee": fee,
        "rsi": rsi,
        "stake_mode": decision.stake_mode,
        "trigger": decision.trigger,
    }


def build_trail_event_payload(
    decision: TrailDecision,
    *,
    symbol: str,
) -> dict[str, Any]:
    return {
        "code": decision.event_code,
        "category": "trade",
        "level": decision.level,
        "symbol": symbol,
        "side": decision.side,
        **decision.event_context,
    }


def build_exit_event_payload(
    decision: ExitDecision,
    *,
    symbol: str,
    net_pnl: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": decision.event_code,
        "category": decision.category,
        "level": decision.level,
        "notify": True,
        "symbol": symbol,
        "side": decision.side,
        "exit": decision.exit,
        "net_pnl": net_pnl,
    }
    if decision.margin_used:
        payload["roi_pct"] = (net_pnl / decision.margin_used) * 100.0
    if decision.via_tail_guard:
        payload["via_tail_guard"] = True
    if decision.liq_price is not None:
        payload["liq_price"] = decision.liq_price
    if decision.rsi is not None:
        payload["rsi"] = decision.rsi
    if decision.threshold is not None:
        payload["threshold"] = decision.threshold
    return payload


def emit_exit_event(log_buffer, config: StrategyConfig, log_ts: str, decision: ExitDecision, net_pnl: float) -> None:
    event_kwargs = build_exit_event_payload(
        decision,
        symbol=config.symbol,
        net_pnl=net_pnl,
    )
    emit_event(
        ts=log_ts,
        log_buffer=log_buffer,
        runtime_mode=config.runtime_mode,
        strategy_mode=config.strategy_mode,
        **event_kwargs,
    )


def apply_trail_decision(
    position: dict[str, Any],
    state: dict[str, Any],
    live: bool,
    log_buffer: list[str],
    config: StrategyConfig,
    log_ts: str,
    decision: TrailDecision,
) -> None:
    position.update(decision.position_updates)
    if live:
        update_position(state, position)
    event_kwargs = build_trail_event_payload(decision, symbol=config.symbol)
    emit_event(
        ts=log_ts,
        log_buffer=log_buffer,
        runtime_mode=config.runtime_mode,
        strategy_mode=config.strategy_mode,
        **event_kwargs,
    )


def build_position_metrics(
    position: dict[str, Any],
    snapshot: DiscreteRowSnapshot,
    *,
    leverage: float,
    fee_rate: float,
) -> PositionMetrics:
    side = str(position["side"]).strip().lower()
    size = float(position["size"])
    entry_price = float(position["entry"])
    price = float(snapshot.price)
    position_value = size * entry_price
    fee_close = position_value * fee_rate
    gross_pnl = (
        (price - entry_price) / entry_price * position_value
        if side == "long"
        else (entry_price - price) / entry_price * position_value
    )
    margin_used = position_value / leverage if leverage > 0 else None
    return PositionMetrics(
        side=side,
        price=price,
        entry_price=entry_price,
        position_value=position_value,
        fee_close=fee_close,
        gross_pnl=gross_pnl,
        margin_used=margin_used,
        base_sl=float(position["sl"]),
        base_tp=float(position["tp"]),
        liquidation_price=float(position["liq_price"]),
    )


def resolve_management_decision(
    position: dict[str, Any],
    snapshot: DiscreteRowSnapshot,
    *,
    config: StrategyConfig,
    metrics: PositionMetrics,
) -> TrailDecision | ExitDecision | None:
    side = metrics.side
    price = metrics.price
    atr = float(snapshot.atr)
    rsi = float(snapshot.rsi)

    if side == "long" and price < metrics.liquidation_price:
        return ExitDecision(
            event_code="trade_liquidated",
            category="risk",
            level="error",
            reason="liquidation",
            side="long",
            exit=price,
            net_pnl=-(metrics.margin_used or 0.0),
            margin_used=metrics.margin_used,
            liq_price=metrics.liquidation_price,
        )

    if side == "short" and price > metrics.liquidation_price:
        return ExitDecision(
            event_code="trade_liquidated",
            category="risk",
            level="error",
            reason="liquidation",
            side="short",
            exit=price,
            net_pnl=-(metrics.margin_used or 0.0),
            margin_used=metrics.margin_used,
            liq_price=metrics.liquidation_price,
        )

    if side == "long":
        if rsi > config.rsi_extreme_long:
            fee_total = metrics.fee_close * 2
            net_pnl = metrics.gross_pnl - fee_total
            return ExitDecision(
                event_code="trade_closed_rsi_extreme",
                category="trade",
                level="info",
                reason="rsi_extreme",
                side="long",
                exit=price,
                net_pnl=net_pnl,
                margin_used=metrics.margin_used,
                gross_pnl=metrics.gross_pnl,
                fee_total=fee_total,
                rsi=rsi,
                threshold=config.rsi_extreme_long,
            )

        if not position["trail_active"]:
            if price > metrics.base_tp and rsi < config.rsi_long_tp_threshold:
                trail_price = price - atr * config.trail_atr_mult
                return TrailDecision(
                    event_code="trail_activated",
                    level="info",
                    side="long",
                    position_updates={
                        "trail_active": True,
                        "trail_max": price,
                        "trail_price": trail_price,
                    },
                    event_context={
                        "market_price": price,
                        "base_tp": metrics.base_tp,
                        "trail_price": trail_price,
                    },
                )
        else:
            trail_max = float(position["trail_max"])
            if price > trail_max:
                previous_trail = position.get("trail_price")
                current_trail_tp = price - atr * config.trail_atr_mult
                return TrailDecision(
                    event_code="trail_moved",
                    level="info",
                    side="long",
                    position_updates={
                        "trail_max": price,
                        "tp": current_trail_tp,
                        "trail_price": current_trail_tp,
                    },
                    event_context={
                        "previous_trail": previous_trail,
                        "trail_price": current_trail_tp,
                        "anchor_price": price,
                    },
                )
            if price < trail_max - atr * config.trail_atr_mult or rsi > config.rsi_long_close_threshold:
                fee_total = metrics.fee_close * 2
                net_pnl = metrics.gross_pnl - fee_total
                return ExitDecision(
                    event_code="trade_closed_take_profit",
                    category="trade",
                    level="info",
                    reason="take_profit",
                    side="long",
                    exit=price,
                    net_pnl=net_pnl,
                    margin_used=metrics.margin_used,
                    gross_pnl=metrics.gross_pnl,
                    fee_total=fee_total,
                    via_tail_guard=True,
                )

        if price < metrics.base_sl:
            fee_total = metrics.fee_close * 2
            net_pnl = metrics.gross_pnl - fee_total
            return ExitDecision(
                event_code="trade_closed_stop_loss",
                category="trade",
                level="warning",
                reason="stop_loss",
                side="long",
                exit=price,
                net_pnl=net_pnl,
                margin_used=metrics.margin_used,
                gross_pnl=metrics.gross_pnl,
                fee_total=fee_total,
            )

        return None

    if rsi < config.rsi_extreme_short:
        fee_total = metrics.fee_close * 2
        net_pnl = metrics.gross_pnl - fee_total
        return ExitDecision(
            event_code="trade_closed_rsi_extreme",
            category="trade",
            level="info",
            reason="rsi_extreme",
            side="short",
            exit=price,
            net_pnl=net_pnl,
            margin_used=metrics.margin_used,
            gross_pnl=metrics.gross_pnl,
            fee_total=fee_total,
            rsi=rsi,
            threshold=config.rsi_extreme_short,
        )

    if not position["trail_active"]:
        if price < metrics.base_tp and rsi > config.rsi_short_tp_threshold:
            trail_price = price + atr * config.trail_atr_mult
            return TrailDecision(
                event_code="trail_activated",
                level="info",
                side="short",
                position_updates={
                    "trail_active": True,
                    "trail_min": price,
                    "trail_price": trail_price,
                },
                event_context={
                    "market_price": price,
                    "base_tp": metrics.base_tp,
                    "trail_price": trail_price,
                },
            )
    else:
        trail_min = float(position["trail_min"])
        if price < trail_min:
            previous_trail = position.get("trail_price")
            current_trail_tp = price + atr * config.trail_atr_mult
            return TrailDecision(
                event_code="trail_moved",
                level="info",
                side="short",
                position_updates={
                    "trail_min": price,
                    "tp": current_trail_tp,
                    "trail_price": current_trail_tp,
                },
                event_context={
                    "previous_trail": previous_trail,
                    "trail_price": current_trail_tp,
                    "anchor_price": price,
                },
            )
        if price > trail_min + atr * config.trail_atr_mult or rsi < config.rsi_short_close_threshold:
            fee_total = metrics.fee_close * 2
            net_pnl = metrics.gross_pnl - fee_total
            return ExitDecision(
                event_code="trade_closed_take_profit",
                category="trade",
                level="info",
                reason="take_profit",
                side="short",
                exit=price,
                net_pnl=net_pnl,
                margin_used=metrics.margin_used,
                gross_pnl=metrics.gross_pnl,
                fee_total=fee_total,
                via_tail_guard=True,
            )

    if price > metrics.base_sl:
        fee_total = metrics.fee_close * 2
        net_pnl = metrics.gross_pnl - fee_total
        return ExitDecision(
            event_code="trade_closed_stop_loss",
            category="trade",
            level="warning",
            reason="stop_loss",
            side="short",
            exit=price,
            net_pnl=net_pnl,
            margin_used=metrics.margin_used,
            gross_pnl=metrics.gross_pnl,
            fee_total=fee_total,
        )

    return None


def apply_exit_decision(
    *,
    runtime: StrategyRuntime,
    decision: ExitDecision,
    position: dict[str, Any],
    state: dict[str, Any],
    live: bool,
    balance: float,
    trade_history: list[dict[str, Any]],
    log_buffer: list[str],
    config: StrategyConfig,
    row_ts: str,
    log_ts: str,
) -> tuple[float, dict[str, Any] | None]:
    if live:
        trade = build_exit_trade(position, decision, row_ts=row_ts)
        close_order = None
        if decision.reason != "liquidation":
            close_order = close_position(config.symbol)
            execution_summary = get_order_execution_summary(config.symbol, close_order)
            _apply_exit_execution_summary(
                trade,
                position,
                execution_summary,
                fee_rate=config.fee_rate,
            )
        current_balance = get_balance()
        balance = current_balance
        pending_event = build_exit_event_payload(
            decision,
            symbol=config.symbol,
            net_pnl=float(trade["net_pnl"]),
        )
        pending_position = _mark_position_close_pending(
            position,
            trade=trade,
            event_payload=pending_event,
            event_ts=log_ts,
            runtime_mode=config.runtime_mode,
            strategy_mode=config.strategy_mode,
            order_id=(close_order.get("orderId") if isinstance(close_order, dict) else None),
        )
        state["position"] = pending_position
        update_balance(state, balance)
        position = pending_position
    else:
        balance, _ = apply_backtest_exit_transition(
            balance,
            trade_history,
            position,
            decision,
            row_ts=row_ts,
            fee_rate=config.fee_rate,
        )
        emit_exit_event(log_buffer, config, log_ts, decision, decision.net_pnl)
        if runtime.execution_mode == "simulated":
            emit_simulated_exit_execution_events(
                runtime,
                config,
                position,
                decision,
                ts=row_ts,
                balance_after=balance,
            )

    return balance, position if live else None


def process_discrete_row(
    snapshot: DiscreteRowSnapshot,
    runtime: StrategyRuntime,
    config: StrategyConfig,
    technical_logger,
) -> None:
    state = runtime.state
    balance = runtime.balance
    position = runtime.position
    trade_history = runtime.trade_history
    log_buffer = runtime.log_buffer
    live = runtime.live
    observed_execution_mode = runtime.execution_mode == "observed"

    price = snapshot.price
    lower = snapshot.lower
    upper = snapshot.upper
    mid = snapshot.mid
    atr = snapshot.atr
    rsi = snapshot.rsi
    ema = snapshot.ema
    row_ts = snapshot.row_ts
    log_ts = snapshot.log_ts

    if live:
        refresh_market_snapshot(state, price, row_ts)
    previous_search_status = state.get("search_status")
    next_search_status = resolve_search_status(price, ema, previous_search_status)
    if next_search_status is not previous_search_status:
        state["search_status"] = next_search_status

    if position is None:
        if observed_execution_mode:
            runtime.balance = balance
            runtime.position = position
            return
        if not config.allow_entries:
            runtime.balance = balance
            runtime.position = position
            return

        manual_trade_suggestion = consume_manual_trade_suggestion(state, live) if live else None
        manual_side = manual_trade_suggestion["side"] if manual_trade_suggestion else None

        qty_local = compute_qty(
            config.symbol,
            balance,
            config.leverage,
            price,
            config.qty,
            config.use_full_balance,
            live,
        )
        entry_decision = resolve_entry_decision(
            snapshot,
            config=config,
            qty_local=qty_local,
            manual_side=manual_side,
        )

        if isinstance(entry_decision, EntryGuardRejection):
            event_kwargs = build_entry_guard_event_payload(
                entry_decision,
                symbol=config.symbol,
            )
            emit_event(
                ts=log_ts,
                log_buffer=log_buffer,
                runtime_mode=config.runtime_mode,
                strategy_mode=config.strategy_mode,
                **event_kwargs,
            )
        elif isinstance(entry_decision, EntryDecision):
            position = build_position_from_entry(entry_decision, row_ts=row_ts)
            position["entry_rsi"] = rsi
            refresh_position_snapshot(position, price, config.leverage, row_ts)
            entry_fee = entry_decision.size * price * config.fee_rate

            if live:
                if can_open_trade(config.symbol, entry_decision.size, config.leverage):
                    side_command = "BUY" if entry_decision.side == "long" else "SELL"
                    open_order = open_position(
                        config.symbol,
                        side_command,
                        entry_decision.size,
                        entry_decision.sl,
                        entry_decision.tp,
                        config.leverage,
                    )
                    execution_summary = get_order_execution_summary(config.symbol, open_order)
                    _apply_entry_execution_summary(
                        position,
                        execution_summary,
                        side=entry_decision.side,
                        fallback_ts=row_ts,
                    )
                    refresh_position_snapshot(position, price, config.leverage, row_ts)
                    factual_entry_fee = to_float(position.get("entry_fee_paid"))
                    if factual_entry_fee is not None:
                        entry_fee = factual_entry_fee
                    balance = get_balance()
                    event_kwargs = build_trade_opened_event_payload(
                        entry_decision,
                        symbol=config.symbol,
                        leverage=config.leverage,
                        fee=entry_fee,
                        rsi=rsi,
                    )
                    event_kwargs["entry"] = float(position["entry"])
                    event_kwargs["size"] = float(position["size"])
                    pending_position = _mark_position_open_pending(
                        position,
                        event_payload=event_kwargs,
                        event_ts=log_ts,
                        runtime_mode=config.runtime_mode,
                        strategy_mode=config.strategy_mode,
                        order_id=(open_order.get("orderId") if isinstance(open_order, dict) else None),
                    )
                    state["position"] = pending_position
                    update_balance(state, balance)
                    position = pending_position
            else:
                balance -= _position_entry_fee(position, config.fee_rate)
                event_kwargs = build_trade_opened_event_payload(
                    entry_decision,
                    symbol=config.symbol,
                    leverage=config.leverage,
                    fee=entry_fee,
                    rsi=rsi,
                )
                emit_event(
                    ts=log_ts,
                    log_buffer=log_buffer,
                    runtime_mode=config.runtime_mode,
                    strategy_mode=config.strategy_mode,
                    **event_kwargs,
                )
                if runtime.execution_mode == "simulated":
                    emit_simulated_entry_execution_events(
                        runtime,
                        config,
                        position,
                        ts=row_ts,
                        balance_after=balance,
                    )

    else:
        if observed_execution_mode and bool(position.get("observed_execution_only")):
            runtime.balance = balance
            runtime.position = position
            return
        if live and bool(position.get("open_pending")):
            runtime.balance = balance
            runtime.position = position
            return
        side = position["side"]
        refresh_position_snapshot(position, price, config.leverage, row_ts)
        if live:
            update_position(state, position)
        if live and bool(position.get("close_pending")):
            runtime.balance = balance
            runtime.position = position
            return
        metrics = build_position_metrics(
            position,
            snapshot,
            leverage=config.leverage,
            fee_rate=config.fee_rate,
        )
        management_decision = resolve_management_decision(
            position,
            snapshot,
            config=config,
            metrics=metrics,
        )

        if isinstance(management_decision, TrailDecision):
            apply_trail_decision(
                position,
                state,
                live,
                log_buffer,
                config,
                log_ts,
                management_decision,
            )
        elif isinstance(management_decision, ExitDecision):
            balance, position = apply_exit_decision(
                runtime=runtime,
                decision=management_decision,
                position=position,
                state=state,
                live=live,
                balance=balance,
                trade_history=trade_history,
                log_buffer=log_buffer,
                config=config,
                row_ts=row_ts,
                log_ts=log_ts,
            )
            runtime.balance = balance
            runtime.position = position
            return

        if live and position is not None:
            technical_logger.debug(
                "runtime_open_position_pnl side=%s symbol=%s gross_pnl=%.2f",
                side,
                config.symbol,
                metrics.gross_pnl,
            )

    if live:
        technical_logger.debug(
            "runtime_indicator_snapshot symbol=%s price=%.2f bbl=%.2f bbm=%.2f bbu=%.2f rsi=%.2f ema=%.2f",
            config.symbol,
            price,
            lower,
            mid,
            upper,
            rsi,
            ema,
        )

    runtime.balance = balance
    runtime.position = position


def run_strategy(
    df,
    live: bool = False,
    initial_balance: float = 1000,
    qty: float | None = None,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    symbol: str = "BTCUSDT",
    leverage: float = 1,
    use_full_balance: bool = True,
    fee_rate: float = 0.0005,
    state=None,
    use_state: bool = True,
    enable_logs: bool = True,
    show_progress: bool = True,
    rsi_extreme_long: float = 75,
    rsi_extreme_short: float = 25,
    rsi_long_open_threshold: float = 50,
    rsi_long_qty_threshold: float = 30,
    rsi_long_tp_threshold: float = 58,
    rsi_long_close_threshold: float = 70,
    rsi_short_open_threshold: float = 50,
    rsi_short_qty_threshold: float = 70,
    rsi_short_tp_threshold: float = 42,
    rsi_short_close_threshold: float = 30,
    trail_atr_mult: float = 0.5,
    allow_entries: bool = True,
    execution_mode: str = "simulated",
    runtime_mode: str | None = None,
    indicator_inputs: dict[str, str] | None = None,
    progress_reporter: Any | None = None,
):
    normalized_runtime_mode = str(
        runtime_mode or os.getenv("SCROOGE_RUNTIME_MODE", "live" if live else "backtest")
    ).strip().lower() or ("live" if live else "backtest")
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=execution_mode,
        load_state_fn=load_state,
        provided_state=state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
    normalized_indicator_inputs = normalize_indicator_inputs(
        indicator_inputs,
        strategy_mode="discrete",
    )
    config = StrategyConfig(
        qty=qty,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        symbol=symbol,
        leverage=leverage,
        use_full_balance=use_full_balance,
        fee_rate=fee_rate,
        rsi_extreme_long=rsi_extreme_long,
        rsi_extreme_short=rsi_extreme_short,
        rsi_long_open_threshold=rsi_long_open_threshold,
        rsi_long_qty_threshold=rsi_long_qty_threshold,
        rsi_long_tp_threshold=rsi_long_tp_threshold,
        rsi_long_close_threshold=rsi_long_close_threshold,
        rsi_short_open_threshold=rsi_short_open_threshold,
        rsi_short_qty_threshold=rsi_short_qty_threshold,
        rsi_short_tp_threshold=rsi_short_tp_threshold,
        rsi_short_close_threshold=rsi_short_close_threshold,
        trail_atr_mult=trail_atr_mult,
        allow_entries=allow_entries,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        runtime_mode=normalized_runtime_mode,
        strategy_mode="discrete",
        indicator_inputs=normalized_indicator_inputs,
    )

    def on_row(snapshot, row_runtime):
        process_discrete_row(snapshot, row_runtime, config, technical_logger)

    return run_discrete_engine(
        df,
        runtime=runtime,
        show_progress=show_progress,
        progress_reporter=progress_reporter,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_formatter=format_event_timestamp,
        indicator_inputs=config.indicator_inputs,
        on_row=on_row,
        save_log_fn=save_log,
        save_state_fn=save_state,
    )


def run_strategy_on_snapshot(
    row: Any,
    live: bool = False,
    initial_balance: float = 1000,
    qty: float | None = None,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    symbol: str = "BTCUSDT",
    leverage: float = 1,
    use_full_balance: bool = True,
    fee_rate: float = 0.0005,
    state=None,
    use_state: bool = True,
    enable_logs: bool = True,
    rsi_extreme_long: float = 75,
    rsi_extreme_short: float = 25,
    rsi_long_open_threshold: float = 50,
    rsi_long_qty_threshold: float = 30,
    rsi_long_tp_threshold: float = 58,
    rsi_long_close_threshold: float = 70,
    rsi_short_open_threshold: float = 50,
    rsi_short_qty_threshold: float = 70,
    rsi_short_tp_threshold: float = 42,
    rsi_short_close_threshold: float = 30,
    trail_atr_mult: float = 0.5,
    allow_entries: bool = True,
    execution_mode: str = "simulated",
    runtime_mode: str | None = None,
    indicator_inputs: dict[str, str] | None = None,
):
    normalized_runtime_mode = str(
        runtime_mode or os.getenv("SCROOGE_RUNTIME_MODE", "live" if live else "backtest")
    ).strip().lower() or ("live" if live else "backtest")
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=execution_mode,
        load_state_fn=load_state,
        provided_state=state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
    normalized_indicator_inputs = normalize_indicator_inputs(
        indicator_inputs,
        strategy_mode="discrete",
    )
    config = StrategyConfig(
        qty=qty,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        symbol=symbol,
        leverage=leverage,
        use_full_balance=use_full_balance,
        fee_rate=fee_rate,
        rsi_extreme_long=rsi_extreme_long,
        rsi_extreme_short=rsi_extreme_short,
        rsi_long_open_threshold=rsi_long_open_threshold,
        rsi_long_qty_threshold=rsi_long_qty_threshold,
        rsi_long_tp_threshold=rsi_long_tp_threshold,
        rsi_long_close_threshold=rsi_long_close_threshold,
        rsi_short_open_threshold=rsi_short_open_threshold,
        rsi_short_qty_threshold=rsi_short_qty_threshold,
        rsi_short_tp_threshold=rsi_short_tp_threshold,
        rsi_short_close_threshold=rsi_short_close_threshold,
        trail_atr_mult=trail_atr_mult,
        allow_entries=allow_entries,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        runtime_mode=normalized_runtime_mode,
        strategy_mode="discrete",
        indicator_inputs=normalized_indicator_inputs,
    )

    snapshot = build_row_snapshot(
        row,
        live=live,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_formatter=format_event_timestamp,
        indicator_inputs=normalized_indicator_inputs,
    )
    process_discrete_row(snapshot, runtime, config, technical_logger)
    runtime.balance_history.append(runtime.balance)

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log,
        save_state_fn=save_state,
    )


def run_strategy_on_tape(
    tape_rows,
    **kwargs,
):
    return run_strategy(tape_rows, **kwargs)


def run_strategy_on_market_events(
    market_events: Iterable[MarketEvent],
    *,
    candle_interval: str = "1m",
    intervals: dict[str, str] | None = None,
    market_event_total: int | None = None,
    strategy_mode: str = "discrete",
    strict_indicator_alignment: bool = False,
    live: bool = False,
    initial_balance: float = 1000,
    qty: float | None = None,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    symbol: str = "BTCUSDT",
    leverage: float = 1,
    use_full_balance: bool = True,
    fee_rate: float = 0.0005,
    state=None,
    use_state: bool = True,
    enable_logs: bool = True,
    show_progress: bool = True,
    rsi_extreme_long: float = 75,
    rsi_extreme_short: float = 25,
    rsi_long_open_threshold: float = 50,
    rsi_long_qty_threshold: float = 30,
    rsi_long_tp_threshold: float = 58,
    rsi_long_close_threshold: float = 70,
    rsi_short_open_threshold: float = 50,
    rsi_short_qty_threshold: float = 70,
    rsi_short_tp_threshold: float = 42,
    rsi_short_close_threshold: float = 30,
    trail_atr_mult: float = 0.5,
    allow_entries: bool = True,
    execution_mode: str = "simulated",
    runtime_mode: str | None = None,
    indicator_inputs: dict[str, str] | None = None,
    progress_reporter: Any | None = None,
):
    normalized_strategy_mode = str(strategy_mode or "discrete").strip().lower() or "discrete"
    normalized_runtime_mode = str(
        runtime_mode or os.getenv("SCROOGE_RUNTIME_MODE", "live" if live else "backtest")
    ).strip().lower() or ("live" if live else "backtest")
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        load_state_fn=load_state,
        provided_state=state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
    normalized_indicator_inputs = normalize_indicator_inputs(
        indicator_inputs,
        strategy_mode=normalized_strategy_mode,
    )
    config = StrategyConfig(
        qty=qty,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        symbol=symbol,
        leverage=leverage,
        use_full_balance=use_full_balance,
        fee_rate=fee_rate,
        rsi_extreme_long=rsi_extreme_long,
        rsi_extreme_short=rsi_extreme_short,
        rsi_long_open_threshold=rsi_long_open_threshold,
        rsi_long_qty_threshold=rsi_long_qty_threshold,
        rsi_long_tp_threshold=rsi_long_tp_threshold,
        rsi_long_close_threshold=rsi_long_close_threshold,
        rsi_short_open_threshold=rsi_short_open_threshold,
        rsi_short_qty_threshold=rsi_short_qty_threshold,
        rsi_short_tp_threshold=rsi_short_tp_threshold,
        rsi_short_close_threshold=rsi_short_close_threshold,
        trail_atr_mult=trail_atr_mult,
        allow_entries=allow_entries,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        runtime_mode=normalized_runtime_mode,
        strategy_mode=normalized_strategy_mode,
        indicator_inputs=normalized_indicator_inputs,
    )

    def on_row(snapshot, row_runtime):
        process_discrete_row(snapshot, row_runtime, config, technical_logger)

    if normalized_strategy_mode == "realtime" or uses_realtime_indicator_inputs(normalized_indicator_inputs):
        resolved_intervals = {
            "small": "1m",
            "medium": "1h",
            "big": "4h",
        }
        if intervals is not None:
            resolved_intervals.update({key: str(value) for key, value in intervals.items()})
        return run_realtime_market_event_engine(
            market_events,
            runtime=runtime,
            intervals=resolved_intervals,
            indicator_inputs=normalized_indicator_inputs,
            emit_on_price_tick=(normalized_strategy_mode == "realtime"),
            show_progress=show_progress,
            market_event_total=market_event_total,
            progress_reporter=progress_reporter,
            on_row=on_row,
            save_log_fn=save_log,
            save_state_fn=save_state,
            symbol=symbol,
        )

    return run_market_event_engine(
        market_events,
        runtime=runtime,
        candle_interval=candle_interval,
        show_progress=show_progress,
        market_event_total=market_event_total,
        progress_reporter=progress_reporter,
        indicator_inputs=normalized_indicator_inputs,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_formatter=format_event_timestamp,
        on_row=on_row,
        save_log_fn=save_log,
        save_state_fn=save_state,
        symbol=symbol,
        strict_indicator_alignment=strict_indicator_alignment,
    )


def finalize_strategy_runtime(
    runtime: StrategyRuntime,
    *,
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    if runtime.log_buffer and runtime.enable_logs:
        save_log_fn(runtime.log_buffer)

    if runtime.execution_sync.total_events > 0:
        _sync_execution_context_state(runtime)

    serialized_execution_events = [market_event_to_dict(event) for event in runtime.execution_events]
    runtime.state.pop("simulated_execution_events", None)

    if runtime.live:
        save_state_fn(runtime.state)
    elif runtime.use_state:
        runtime.state["position"] = runtime.position
        runtime.state["balance"] = runtime.balance
        runtime.state["trade_history"] = runtime.trade_history
        runtime.state["balance_history"] = runtime.balance_history
        save_state_fn(runtime.state)
        runtime.state["trade_history"] = runtime.trade_history
        runtime.state["balance_history"] = runtime.balance_history

    if serialized_execution_events:
        runtime.state["simulated_execution_events"] = serialized_execution_events

    return (
        runtime.balance,
        pd.DataFrame(runtime.state.get("trade_history", [])),
        runtime.state.get("balance_history", []),
        runtime.state,
    )


def run_discrete_engine(
    df: pd.DataFrame,
    *,
    runtime: StrategyRuntime,
    show_progress: bool,
    progress_reporter: Any | None,
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
    indicator_inputs: dict[str, str] | None,
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    for row in iter_discrete_rows(df, live=runtime.live, show_progress=show_progress, progress_reporter=progress_reporter):
        snapshot = build_row_snapshot(
            row,
            live=runtime.live,
            timestamp_format=timestamp_format,
            timestamp_formatter=timestamp_formatter,
            indicator_inputs=indicator_inputs,
        )
        on_row(snapshot, runtime)
        runtime.balance_history.append(runtime.balance)

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log_fn,
        save_state_fn=save_state_fn,
    )


def run_market_event_engine(
    market_events: Any,
    *,
    runtime: StrategyRuntime,
    candle_interval: str,
    show_progress: bool,
    market_event_total: int | None,
    progress_reporter: Any | None,
    indicator_inputs: dict[str, str],
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
    symbol: str | None = None,
    strict_indicator_alignment: bool = False,
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    target_symbol = str(symbol or "").strip().upper() or None
    pending_candles: dict[tuple[str, str], CandleClosedEvent] = {}
    pending_indicators: dict[tuple[str, str], IndicatorSnapshotEvent] = {}
    iterator = _iter_with_progress(
        market_events,
        show_progress=show_progress,
        desc="Backtest Event Replay",
        total=market_event_total,
        progress_reporter=progress_reporter,
    )

    def try_emit(symbol_key: str, ts: str) -> None:
        key = (symbol_key, ts)
        candle = pending_candles.get(key)
        indicator = pending_indicators.get(key)
        if candle is None or indicator is None:
            return

        pending_candles.pop(key, None)
        pending_indicators.pop(key, None)
        row_payload = {
            "open_time": candle.open_time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "EMA": indicator.values.get("EMA"),
            "RSI": indicator.values.get("RSI"),
            "BBL": indicator.values.get("BBL"),
            "BBM": indicator.values.get("BBM"),
            "BBU": indicator.values.get("BBU"),
            "ATR": indicator.values.get("ATR"),
        }
        snapshot = build_row_snapshot(
            row_payload,
            live=runtime.live,
            timestamp_format=timestamp_format,
            timestamp_formatter=timestamp_formatter,
            indicator_inputs=indicator_inputs,
        )
        on_row(snapshot, runtime)
        runtime.balance_history.append(runtime.balance)

    for event in iterator:
        if apply_market_event_runtime_sync(runtime, event, target_symbol=target_symbol):
            continue

        event_symbol = getattr(event, "symbol", None)
        if target_symbol and str(event_symbol or "").strip().upper() != target_symbol:
            continue
        event_type = getattr(event, "event_type", "")

        if event_type == "candle_closed":
            if event.interval != candle_interval:
                continue
            key = (event.symbol, event.ts)
            pending_candles[key] = event
            try_emit(event.symbol, event.ts)
            continue

        if event_type == "indicator_snapshot" and event.interval == "discrete_snapshot":
            key = (event.symbol, event.ts)
            pending_indicators[key] = event
            try_emit(event.symbol, event.ts)

    if strict_indicator_alignment and pending_candles:
        missing = next(iter(pending_candles.values()))
        raise ValueError(
            "Market event replay is missing indicator snapshot "
            f"for symbol={missing.symbol} ts={missing.ts} interval={missing.interval}"
        )

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log_fn,
        save_state_fn=save_state_fn,
    )


def run_realtime_market_event_engine(
    market_events: Any,
    *,
    runtime: StrategyRuntime,
    intervals: dict[str, str],
    indicator_inputs: dict[str, str],
    emit_on_price_tick: bool,
    show_progress: bool,
    market_event_total: int | None,
    progress_reporter: Any | None,
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
    symbol: str | None = None,
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    target_symbol = str(symbol or "").strip().upper() or None
    processor = RealtimeStrategyProcessor(
        runtime=runtime,
        feature_engine=FeatureEngine(
            intervals=intervals,
            limits=REALTIME_WARMUP_LIMITS,
        ),
        selection_plan=indicator_selection_plan(indicator_inputs),
        emit_on_price_tick=emit_on_price_tick,
        on_row=on_row,
        target_symbol=target_symbol,
        intervals=intervals,
        latest_discrete_indicator_values=None,
        pending_small_candles={},
        small_interval_ns=interval_to_ns(intervals["small"]),
    )
    iterator = _iter_with_progress(
        market_events,
        show_progress=show_progress,
        desc="Realtime Event Replay",
        total=market_event_total,
        progress_reporter=progress_reporter,
    )

    for event in iterator:
        processor.process_event(event)

    if processor.processed_price_ticks == 0 and processor.emitted_snapshots == 0:
        raise ValueError("Realtime market event replay requires at least one price_tick event.")
    if processor.emitted_snapshots == 0:
        raise ValueError("Realtime market event replay produced no valid snapshots; check warmup history and indicator coverage.")

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log_fn,
        save_state_fn=save_state_fn,
    )
