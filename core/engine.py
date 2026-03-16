from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

from bot.event_log import emit_event, get_technical_logger
from bot.state import add_closed_trade, load_state, save_state, update_balance, update_position
from bot.trade import can_open_trade, close_position, compute_qty, get_balance, open_position
from core.market_events import (
    AccountBalanceEvent,
    CandleClosedEvent,
    IndicatorSnapshotEvent,
    MarketEvent,
    OrderTradeUpdateEvent,
    PositionSnapshotEvent,
    PriceTickEvent,
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


def save_log(log_buffer: list[str]) -> None:
    with open(LOG_FILE, "a") as f:
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


def _compute_realtime_indicator_values(
    medium_df: pd.DataFrame,
    big_df: pd.DataFrame,
) -> dict[str, float | None] | None:
    if medium_df.empty or big_df.empty:
        return None

    medium = medium_df.copy().set_index("open_time")
    bb = ta.bbands(medium["close"], length=20, std=2)
    atr = ta.atr(medium["high"], medium["low"], medium["close"], length=14)
    if bb is None or atr is None:
        return None

    big = big_df.copy().set_index("open_time")
    rsi = ta.rsi(big["close"], length=11)
    ema = ta.ema(big["close"], length=50)
    if rsi is None or ema is None:
        return None

    latest_values = {
        "BBL": to_float(bb["BBL_20_2.0_2.0"].iloc[-1]) if "BBL_20_2.0_2.0" in bb else None,
        "BBM": to_float(bb["BBM_20_2.0_2.0"].iloc[-1]) if "BBM_20_2.0_2.0" in bb else None,
        "BBU": to_float(bb["BBU_20_2.0_2.0"].iloc[-1]) if "BBU_20_2.0_2.0" in bb else None,
        "ATR": to_float(atr.iloc[-1]),
        "RSI": to_float(rsi.iloc[-1]),
        "EMA": to_float(ema.iloc[-1]),
    }
    if any(value is None for value in latest_values.values()):
        return None
    return latest_values


def _build_realtime_snapshot(
    *,
    event_ts: str,
    forming_small: dict[str, Any] | None,
    medium_df: pd.DataFrame,
    big_df: pd.DataFrame,
) -> DiscreteRowSnapshot | None:
    if forming_small is None:
        return None
    indicator_values = _compute_realtime_indicator_values(medium_df, big_df)
    if indicator_values is None:
        return None
    price = to_float(forming_small.get("close"))
    if price is None:
        return None
    row_payload = {
        "open_time": event_ts,
        "close": price,
        "BBL": indicator_values["BBL"],
        "BBM": indicator_values["BBM"],
        "BBU": indicator_values["BBU"],
        "ATR": indicator_values["ATR"],
        "RSI": indicator_values["RSI"],
        "EMA": indicator_values["EMA"],
    }
    return DiscreteRowSnapshot(
        raw_row=row_payload,
        price=price,
        lower=indicator_values["BBL"],
        upper=indicator_values["BBU"],
        mid=indicator_values["BBM"],
        atr=indicator_values["ATR"],
        rsi=indicator_values["RSI"],
        ema=indicator_values["EMA"],
        row_ts=event_ts,
        log_ts=event_ts,
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


def resolve_search_status(price: Any, ema: Any, previous_status: Any) -> dict[str, str] | None:
    price_value = to_float(price)
    ema_value = to_float(ema)
    previous_code = None
    if isinstance(previous_status, dict):
        raw_code = str(previous_status.get("code", "")).strip()
        if raw_code in SEARCH_STATUS_LABELS:
            previous_code = raw_code

    if price_value is None or ema_value is None:
        return status_from_code(previous_code, SEARCH_STATUS_LABELS) if previous_code else None

    if price_value > ema_value:
        return status_from_code("looking_for_buy_opportunity", SEARCH_STATUS_LABELS)
    if price_value < ema_value:
        return status_from_code("looking_for_sell_opportunity", SEARCH_STATUS_LABELS)
    if previous_code:
        return status_from_code(previous_code, SEARCH_STATUS_LABELS)
    return status_from_code("looking_for_buy_opportunity", SEARCH_STATUS_LABELS)


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
    position.pop("last_price", None)
    position.pop("last_price_updated_at", None)

    if position.get("entry_time") is None and position.get("time") is not None:
        position["entry_time"] = position.get("time")
    if position.get("time") is None and position.get("entry_time") is not None:
        position["time"] = position.get("entry_time")

    side = str(position.get("side", "")).strip().lower()
    size = to_float(position.get("size")) or 0.0
    entry_price = to_float(position.get("entry"))
    mark_price = to_float(price)
    sl_price = to_float(position.get("sl"))
    tp_price = to_float(position.get("tp"))

    if mark_price is None:
        return

    position["updated_at"] = ts_label
    position["unrealized_pnl"] = None
    position["unrealized_pnl_pct"] = None
    position["position_notional"] = None
    position["margin_used"] = None
    position["roi_pct"] = None
    position["distance_to_sl_pct"] = None
    position["distance_to_tp_pct"] = None

    if entry_price is None or entry_price <= 0 or size <= 0:
        return

    notional = abs(size) * entry_price
    if side == "short":
        unrealized_pnl = (entry_price - mark_price) * size
    else:
        unrealized_pnl = (mark_price - entry_price) * size

    unrealized_pnl_pct = ratio_to_percent(unrealized_pnl, notional)
    leverage_value = to_float(leverage)
    margin_used = (notional / leverage_value) if leverage_value and leverage_value > 0 else None
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
    if isinstance(event, AccountBalanceEvent):
        _apply_account_balance_market_event(runtime, event)
        return True
    if isinstance(event, PositionSnapshotEvent):
        _apply_position_snapshot_market_event(runtime, event, target_symbol=target_symbol)
        return True
    if isinstance(event, OrderTradeUpdateEvent):
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
) -> StrategyRuntime:
    if use_state:
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
    )


def iter_discrete_rows(rows: Any, *, live: bool, show_progress: bool) -> Any:
    if isinstance(rows, pd.DataFrame):
        if live:
            return [rows.iloc[-1]]
        df_iter = [rows.iloc[i] for i in range(1, len(rows))]
        return tqdm(df_iter, desc="Backtest Progress", disable=not show_progress)

    row_list = list(rows)
    if live:
        return row_list[-1:] if row_list else []

    tape_iter = row_list[1:]
    return tqdm(tape_iter, desc="Backtest Progress", disable=not show_progress)


def build_row_snapshot(
    row: Any,
    *,
    live: bool,
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
) -> DiscreteRowSnapshot:
    row_ts = timestamp_formatter(_row_value(row, "open_time"))
    log_ts = datetime.now().strftime(timestamp_format) if live else row_ts
    return DiscreteRowSnapshot(
        raw_row=row,
        price=_row_value(row, "close"),
        lower=_row_value(row, "BBL"),
        upper=_row_value(row, "BBU"),
        mid=_row_value(row, "BBM"),
        atr=_row_value(row, "ATR"),
        rsi=_row_value(row, "RSI"),
        ema=_row_value(row, "EMA"),
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
    }


def sanitize_trade_for_history(trade: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in trade.items():
        if key in _TRANSIENT_POSITION_FIELDS:
            continue
        sanitized[key] = value
    return sanitized


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
    return trade


def apply_backtest_exit_transition(
    balance: float,
    trade_history: list[dict[str, Any]],
    position: dict[str, Any],
    decision: ExitDecision,
    *,
    row_ts: str,
) -> tuple[float, dict[str, Any]]:
    trade = build_exit_trade(position, decision, row_ts=row_ts)
    updated_balance = balance + decision.net_pnl
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
) -> tuple[float, None]:
    if live:
        trade = build_exit_trade(position, decision, row_ts=row_ts)
        if decision.reason != "liquidation":
            close_position(config.symbol)
        current_balance = get_balance()
        trade["net_pnl"] = current_balance - balance
        balance = current_balance
        update_position(state, None)
        update_balance(state, balance)
        add_closed_trade(state, sanitize_trade_for_history(trade))
        emit_exit_event(log_buffer, config, log_ts, decision, trade["net_pnl"])
    else:
        balance, _ = apply_backtest_exit_transition(
            balance,
            trade_history,
            position,
            decision,
            row_ts=row_ts,
        )
        emit_exit_event(log_buffer, config, log_ts, decision, decision.net_pnl)

    return balance, None


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
    state["search_status"] = resolve_search_status(price, ema, state.get("search_status"))

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
                **event_kwargs,
            )
        elif isinstance(entry_decision, EntryDecision):
            position = build_position_from_entry(entry_decision, row_ts=row_ts)
            refresh_position_snapshot(position, price, config.leverage, row_ts)
            event_kwargs = build_trade_opened_event_payload(
                entry_decision,
                symbol=config.symbol,
                leverage=config.leverage,
                fee=entry_decision.size * price * config.fee_rate,
                rsi=rsi,
            )

            if live:
                if can_open_trade(config.symbol, entry_decision.size, config.leverage):
                    side_command = "BUY" if entry_decision.side == "long" else "SELL"
                    open_position(
                        config.symbol,
                        side_command,
                        entry_decision.size,
                        entry_decision.sl,
                        entry_decision.tp,
                        config.leverage,
                    )
                    balance = get_balance()
                    update_position(state, position)
                    update_balance(state, balance)
                emit_event(
                    ts=log_ts,
                    log_buffer=log_buffer,
                    **event_kwargs,
                )
            else:
                balance -= entry_decision.size * price * config.fee_rate
                emit_event(
                    ts=log_ts,
                    log_buffer=log_buffer,
                    **event_kwargs,
                )

    else:
        if observed_execution_mode and bool(position.get("observed_execution_only")):
            runtime.balance = balance
            runtime.position = position
            return
        side = position["side"]
        refresh_position_snapshot(position, price, config.leverage, row_ts)
        if live:
            update_position(state, position)
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
):
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=execution_mode,
        load_state_fn=load_state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
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
    )

    def on_row(snapshot, row_runtime):
        process_discrete_row(snapshot, row_runtime, config, technical_logger)

    return run_discrete_engine(
        df,
        runtime=runtime,
        show_progress=show_progress,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_formatter=format_event_timestamp,
        on_row=on_row,
        save_log_fn=save_log,
        save_state_fn=save_state,
    )


def run_strategy_on_tape(
    tape_rows,
    **kwargs,
):
    return run_strategy(tape_rows, **kwargs)


def run_strategy_on_market_events(
    market_events: list[MarketEvent] | tuple[MarketEvent, ...],
    *,
    candle_interval: str = "1m",
    intervals: dict[str, str] | None = None,
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
):
    normalized_strategy_mode = str(strategy_mode or "discrete").strip().lower() or "discrete"
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        execution_mode=str(execution_mode or "simulated").strip().lower() or "simulated",
        load_state_fn=load_state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
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
    )

    def on_row(snapshot, row_runtime):
        process_discrete_row(snapshot, row_runtime, config, technical_logger)

    if normalized_strategy_mode == "realtime":
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
            show_progress=show_progress,
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
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    for row in iter_discrete_rows(df, live=runtime.live, show_progress=show_progress):
        snapshot = build_row_snapshot(
            row,
            live=runtime.live,
            timestamp_format=timestamp_format,
            timestamp_formatter=timestamp_formatter,
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
    iterator = tqdm(market_events, desc="Backtest Event Replay", disable=not show_progress)

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
        )
        on_row(snapshot, runtime)
        runtime.balance_history.append(runtime.balance)

    for event in iterator:
        if apply_market_event_runtime_sync(runtime, event, target_symbol=target_symbol):
            continue

        event_symbol = getattr(event, "symbol", None)
        if target_symbol and str(event_symbol or "").strip().upper() != target_symbol:
            continue

        if isinstance(event, CandleClosedEvent):
            if event.interval != candle_interval:
                continue
            key = (event.symbol, event.ts)
            pending_candles[key] = event
            try_emit(event.symbol, event.ts)
            continue

        if isinstance(event, IndicatorSnapshotEvent) and event.interval == "discrete_snapshot":
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
    show_progress: bool,
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
    symbol: str | None = None,
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    target_symbol = str(symbol or "").strip().upper() or None
    interval_freqs = {key: _interval_to_freq(value) for key, value in intervals.items()}
    closed_frames = {
        "small": _empty_market_candle_frame(),
        "medium": _empty_market_candle_frame(),
        "big": _empty_market_candle_frame(),
    }
    forming_candles: dict[str, dict[str, Any] | None] = {
        "small": None,
        "medium": None,
        "big": None,
    }
    iterator = tqdm(market_events, desc="Realtime Event Replay", disable=not show_progress)
    processed_price_ticks = 0
    emitted_snapshots = 0
    tick_seen_small_open_times: set[str] = set()

    def resolve_timeframe(interval_value: str) -> str | None:
        for timeframe, configured in intervals.items():
            if configured == interval_value:
                return timeframe
        return None

    for event in iterator:
        if apply_market_event_runtime_sync(runtime, event, target_symbol=target_symbol):
            continue

        event_symbol = getattr(event, "symbol", None)
        if target_symbol and str(event_symbol or "").strip().upper() != target_symbol:
            continue

        if isinstance(event, CandleClosedEvent):
            timeframe = resolve_timeframe(event.interval)
            if timeframe is None:
                continue
            closed_frames[timeframe] = _upsert_closed_market_candle(
                closed_frames[timeframe],
                event,
                limit=REALTIME_WARMUP_LIMITS[timeframe],
            )
            current_forming = forming_candles.get(timeframe)
            if current_forming is not None:
                current_open = pd.Timestamp(current_forming["open_time"])
                event_open = pd.Timestamp(event.open_time)
                if current_open <= event_open:
                    forming_candles[timeframe] = None
            if timeframe == "small":
                open_key = str(event.open_time)
                if open_key in tick_seen_small_open_times:
                    continue
                medium_df = _materialize_market_frame(
                    closed_frames["medium"],
                    forming_candles["medium"],
                    limit=REALTIME_WARMUP_LIMITS["medium"],
                )
                big_df = _materialize_market_frame(
                    closed_frames["big"],
                    forming_candles["big"],
                    limit=REALTIME_WARMUP_LIMITS["big"],
                )
                fallback_small = {
                    "open_time": pd.Timestamp(event.open_time),
                    "open": float(event.open),
                    "high": float(event.high),
                    "low": float(event.low),
                    "close": float(event.close),
                    "volume": float(event.volume),
                }
                snapshot = _build_realtime_snapshot(
                    event_ts=format_event_timestamp(event.ts),
                    forming_small=fallback_small,
                    medium_df=medium_df,
                    big_df=big_df,
                )
                if snapshot is not None:
                    on_row(snapshot, runtime)
                    runtime.balance_history.append(runtime.balance)
                    emitted_snapshots += 1
            continue

        if not isinstance(event, PriceTickEvent):
            continue

        event_ts_value = _event_ts_to_timestamp(event.ts)
        price = to_float(event.price)
        if event_ts_value is None or price is None:
            continue
        processed_price_ticks += 1
        tick_seen_small_open_times.add(str(event_ts_value.floor(interval_freqs["small"])))

        for timeframe, freq in interval_freqs.items():
            interval_open = event_ts_value.floor(freq)
            forming_candles[timeframe] = _update_forming_candle(
                forming_candles.get(timeframe),
                open_time=interval_open,
                price=price,
            )

        medium_df = _materialize_market_frame(
            closed_frames["medium"],
            forming_candles["medium"],
            limit=REALTIME_WARMUP_LIMITS["medium"],
        )
        big_df = _materialize_market_frame(
            closed_frames["big"],
            forming_candles["big"],
            limit=REALTIME_WARMUP_LIMITS["big"],
        )
        snapshot = _build_realtime_snapshot(
            event_ts=format_event_timestamp(event.ts),
            forming_small=forming_candles["small"],
            medium_df=medium_df,
            big_df=big_df,
        )
        if snapshot is None:
            continue

        on_row(snapshot, runtime)
        runtime.balance_history.append(runtime.balance)
        emitted_snapshots += 1

    if processed_price_ticks == 0 and emitted_snapshots == 0:
        raise ValueError("Realtime market event replay requires at least one price_tick event.")
    if emitted_snapshots == 0:
        raise ValueError("Realtime market event replay produced no valid snapshots; check warmup history and indicator coverage.")

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log_fn,
        save_state_fn=save_state_fn,
    )
