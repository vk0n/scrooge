from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


STATE_PATH = Path(os.getenv("SCROOGE_STATE_PATH", str(_project_root() / "runtime" / "state.json"))).expanduser()

SEARCH_STATUS_LABELS = {
    "looking_for_buy_opportunity": "Looking for a buy opportunity...",
    "looking_for_sell_opportunity": "Looking for a sell opportunity...",
}
BOT_STATUS_LABELS = {
    "resting": "Resting...",
    **SEARCH_STATUS_LABELS,
    "managing_open_trade": "Managing an open trade...",
}
TRADE_STATUS_LABELS = {
    "locking_in_profit": "Locking in profit...",
    "waiting_for_more_profit": "Waiting for more profit...",
    "waiting_for_profit": "Waiting for profit...",
}


class UiStatus(TypedDict):
    code: str
    label: str


class OpenTradeInfo(TypedDict):
    side: str
    size: float
    entry: float
    sl: float | None
    tp: float | None
    liq_price: float | None
    trail_active: bool
    trail_price: float | None
    trail_max: float | None
    trail_min: float | None
    entry_time: str
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    position_notional: float | None
    margin_used: float | None
    roi_pct: float | None
    distance_to_sl_pct: float | None
    distance_to_tp_pct: float | None
    updated_at: str | None


def _ratio_to_percent(numerator: Any, denominator: Any) -> float | None:
    left = _maybe_float(numerator)
    right = _maybe_float(denominator)
    if left is None or right is None or right == 0:
        return None
    return (left / right) * 100.0


def load_state() -> tuple[dict[str, Any], list[str]]:
    if not STATE_PATH.exists():
        return (
            {
                "position": None,
                "balance": None,
                "last_price": None,
                "last_price_updated_at": None,
                "updated_at": None,
                "search_status": None,
                "bot_status": None,
                "trade_status": None,
                "session_start": None,
                "session_end": None,
                "trading_enabled": True,
            },
            [f"State file not found: {STATE_PATH}"],
        )

    try:
        with STATE_PATH.open("r", encoding="utf-8") as file_obj:
            raw = json.load(file_obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed state JSON: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read state file: {STATE_PATH}") from exc

    if not isinstance(raw, dict):
        raise ValueError("State must be a JSON object")

    raw.setdefault("trading_enabled", True)
    return raw, []


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_position_side(value: Any) -> str | None:
    if value is None:
        return None
    side = str(value).strip().lower()
    if side not in {"long", "short"}:
        return None
    return side


def _normalize_exchange_position_side(value: Any, position_amt: Any) -> str | None:
    side = str(value or "").strip().upper()
    if side == "LONG":
        return "long"
    if side == "SHORT":
        return "short"

    amount = _maybe_float(position_amt)
    if amount is None:
        return None
    if amount > 0:
        return "long"
    if amount < 0:
        return "short"
    return None


def _extract_exchange_position_snapshot(
    position: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    state_exchange = state.get("exchange_position") if isinstance(state, dict) else None
    if isinstance(state_exchange, dict):
        position_amt = _maybe_float(state_exchange.get("position_amt"))
        entry_price = _maybe_float(state_exchange.get("entry_price"))
        unrealized_pnl = _maybe_float(state_exchange.get("unrealized_pnl"))
        side = _normalize_exchange_position_side(state_exchange.get("position_side"), position_amt)
        if position_amt is not None and abs(position_amt) > 1e-12 and side is not None:
            return {
                "side": side,
                "size": abs(position_amt),
                "entry": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "margin_used": _maybe_float(state_exchange.get("isolated_margin")),
                "updated_at": _maybe_text(state_exchange.get("updated_at")),
            }

    position_amt = _maybe_float(position.get("exchange_position_amt"))
    side = _normalize_exchange_position_side(position.get("exchange_position_side"), position_amt)
    if position_amt is not None and abs(position_amt) > 1e-12 and side is not None:
        return {
            "side": side,
            "size": abs(position_amt),
            "entry": _maybe_float(position.get("exchange_entry_price")),
            "unrealized_pnl": _maybe_float(position.get("exchange_unrealized_pnl")),
            "margin_used": _maybe_float(position.get("exchange_isolated_margin")),
            "updated_at": _maybe_text(position.get("exchange_position_updated_at")),
        }

    return None


def _status_from_code(code: str, labels: dict[str, str]) -> UiStatus | None:
    label = labels.get(code)
    if label is None:
        return None
    return {"code": code, "label": label}


def _resolve_ui_status(value: Any, labels: dict[str, str]) -> UiStatus | None:
    if isinstance(value, dict):
        raw_code = value.get("code")
    else:
        raw_code = value

    code = _maybe_text(raw_code)
    if code is None:
        return None
    return _status_from_code(code, labels)


def resolve_balance(state: dict[str, Any], default_balance: Any = None) -> float | None:
    balance = _maybe_float(state.get("balance"))
    if balance is not None:
        return balance

    history = state.get("balance_history")
    if isinstance(history, list) and history:
        balance = _maybe_float(history[-1])
        if balance is not None:
            return balance

    return _maybe_float(default_balance)


def resolve_last_price(state: dict[str, Any]) -> float | None:
    return _maybe_float(state.get("last_price"))


def resolve_last_price_updated_at(state: dict[str, Any]) -> str | None:
    return _maybe_text(state.get("last_price_updated_at"))


def resolve_bot_status(state: dict[str, Any]) -> UiStatus | None:
    position = state.get("position")
    open_trade_info = resolve_open_trade_info(
        position,
        state=state,
        last_price=resolve_last_price(state),
    )
    trading_enabled = resolve_trading_enabled(state)

    bot_status = _resolve_ui_status(state.get("bot_status"), BOT_STATUS_LABELS)
    if open_trade_info is not None:
        if bot_status is not None and bot_status["code"] == "managing_open_trade":
            return bot_status
        return _status_from_code("managing_open_trade", BOT_STATUS_LABELS)

    if not trading_enabled:
        return _status_from_code("resting", BOT_STATUS_LABELS)

    if bot_status is not None and bot_status["code"] in SEARCH_STATUS_LABELS:
        return bot_status

    search_status = _resolve_ui_status(state.get("search_status"), SEARCH_STATUS_LABELS)
    return search_status


def resolve_open_trade_info(
    position: Any,
    *,
    state: dict[str, Any] | None = None,
    leverage: Any = None,
    last_price: Any = None,
) -> OpenTradeInfo | None:
    if not isinstance(position, dict):
        return None

    exchange_snapshot = _extract_exchange_position_snapshot(position, state=state)
    if bool(position.get("open_pending")) and exchange_snapshot is None:
        return None

    side = (
        (exchange_snapshot.get("side") if isinstance(exchange_snapshot, dict) else None)
        or _normalize_position_side(position.get("side"))
    )
    size = (
        (exchange_snapshot.get("size") if isinstance(exchange_snapshot, dict) else None)
        or _maybe_float(position.get("size"))
    )
    entry = (
        (exchange_snapshot.get("entry") if isinstance(exchange_snapshot, dict) else None)
        or _maybe_float(position.get("entry"))
    )
    if side is None or size is None or entry is None or size <= 0 or entry <= 0:
        return None

    entry_time = _maybe_text(position.get("entry_time")) or _maybe_text(position.get("time"))
    if entry_time is None:
        return None

    current_price = _maybe_float(last_price) or _maybe_float(position.get("last_price"))
    sl = _maybe_float(position.get("sl"))
    tp = _maybe_float(position.get("tp"))
    unrealized_pnl = (
        (exchange_snapshot.get("unrealized_pnl") if isinstance(exchange_snapshot, dict) else None)
        if isinstance(exchange_snapshot, dict)
        else None
    )
    if unrealized_pnl is None:
        unrealized_pnl = _maybe_float(position.get("unrealized_pnl"))

    position_notional = abs(size) * entry
    leverage_value = _maybe_float(leverage)
    margin_used = (
        (exchange_snapshot.get("margin_used") if isinstance(exchange_snapshot, dict) else None)
        if isinstance(exchange_snapshot, dict)
        else None
    )
    if margin_used is None:
        margin_used = (position_notional / leverage_value) if leverage_value is not None and leverage_value > 0 else _maybe_float(position.get("margin_used"))
    unrealized_pnl_pct = _ratio_to_percent(unrealized_pnl, position_notional)
    roi_pct = _ratio_to_percent(unrealized_pnl, margin_used)
    if current_price is not None and current_price > 0:
        if side == "short":
            distance_to_sl_pct = _ratio_to_percent((sl - current_price), current_price) if sl is not None else None
            distance_to_tp_pct = _ratio_to_percent((current_price - tp), current_price) if tp is not None else None
        else:
            distance_to_sl_pct = _ratio_to_percent((current_price - sl), current_price) if sl is not None else None
            distance_to_tp_pct = _ratio_to_percent((tp - current_price), current_price) if tp is not None else None
    else:
        distance_to_sl_pct = _maybe_float(position.get("distance_to_sl_pct"))
        distance_to_tp_pct = _maybe_float(position.get("distance_to_tp_pct"))

    return {
        "side": side,
        "size": size,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "liq_price": _maybe_float(position.get("liq_price")),
        "trail_active": bool(position.get("trail_active", False)),
        "trail_price": _maybe_float(position.get("trail_price")),
        "trail_max": _maybe_float(position.get("trail_max")),
        "trail_min": _maybe_float(position.get("trail_min")),
        "entry_time": entry_time,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "position_notional": position_notional,
        "margin_used": margin_used,
        "roi_pct": roi_pct,
        "distance_to_sl_pct": distance_to_sl_pct,
        "distance_to_tp_pct": distance_to_tp_pct,
        "updated_at": (
            (exchange_snapshot.get("updated_at") if isinstance(exchange_snapshot, dict) else None)
            or _maybe_text(position.get("updated_at"))
        ),
    }


def resolve_trailing_state(position: Any, *, open_trade_info: OpenTradeInfo | None = None) -> dict[str, Any] | None:
    if open_trade_info is None:
        open_trade_info = resolve_open_trade_info(position)
    if open_trade_info is None:
        return None

    trail_active = open_trade_info["trail_active"]
    trail_price = None
    if trail_active:
        trail_price = open_trade_info["trail_price"]
        if trail_price is None:
            trail_price = open_trade_info["tp"]

    return {
        "trail_active": trail_active,
        "trail_max": open_trade_info["trail_max"],
        "trail_min": open_trade_info["trail_min"],
        "trail_price": trail_price,
        "tp": open_trade_info["tp"],
        "sl": open_trade_info["sl"],
    }


def resolve_trade_status(state: dict[str, Any]) -> UiStatus | None:
    position = state.get("position")
    open_trade_info = resolve_open_trade_info(
        position,
        state=state,
        last_price=resolve_last_price(state),
    )
    if open_trade_info is None:
        return None

    trade_status = _resolve_ui_status(state.get("trade_status"), TRADE_STATUS_LABELS)
    if trade_status is not None:
        return trade_status

    if open_trade_info["trail_active"]:
        return _status_from_code("locking_in_profit", TRADE_STATUS_LABELS)

    unrealized_pnl = open_trade_info["unrealized_pnl"]
    if unrealized_pnl is not None and unrealized_pnl < 0:
        return _status_from_code("waiting_for_profit", TRADE_STATUS_LABELS)
    return _status_from_code("waiting_for_more_profit", TRADE_STATUS_LABELS)


def resolve_last_update_timestamp(state: dict[str, Any]) -> str | None:
    updated_at = _maybe_text(state.get("updated_at"))
    if updated_at is not None:
        return updated_at

    last_price_updated_at = resolve_last_price_updated_at(state)
    if last_price_updated_at is not None:
        return last_price_updated_at

    position = state.get("position")
    if isinstance(position, dict):
        updated_at = _maybe_text(position.get("updated_at"))
        if updated_at is not None:
            return updated_at

    exchange_position = state.get("exchange_position")
    if isinstance(exchange_position, dict):
        updated_at = _maybe_text(exchange_position.get("updated_at"))
        if updated_at is not None:
            return updated_at

    for key in ("session_end", "session_start"):
        value = state.get(key)
        if value is None:
            continue
        try:
            ts_ms = int(value)
            return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()
        except (TypeError, ValueError, OSError):
            continue
    return None


def resolve_trading_enabled(state: dict[str, Any]) -> bool:
    return bool(state.get("trading_enabled", True))
