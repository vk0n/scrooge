from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


STATE_PATH = Path(os.getenv("SCROOGE_STATE_PATH", str(_project_root() / "state.json"))).expanduser()

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
    open_trade_info = resolve_open_trade_info(position)
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


def resolve_open_trade_info(position: Any) -> OpenTradeInfo | None:
    if not isinstance(position, dict):
        return None

    side = _normalize_position_side(position.get("side"))
    size = _maybe_float(position.get("size"))
    entry = _maybe_float(position.get("entry"))
    if side is None or size is None or entry is None or size <= 0 or entry <= 0:
        return None

    entry_time = _maybe_text(position.get("entry_time")) or _maybe_text(position.get("time"))
    if entry_time is None:
        return None

    return {
        "side": side,
        "size": size,
        "entry": entry,
        "sl": _maybe_float(position.get("sl")),
        "tp": _maybe_float(position.get("tp")),
        "liq_price": _maybe_float(position.get("liq_price")),
        "trail_active": bool(position.get("trail_active", False)),
        "trail_price": _maybe_float(position.get("trail_price")),
        "trail_max": _maybe_float(position.get("trail_max")),
        "trail_min": _maybe_float(position.get("trail_min")),
        "entry_time": entry_time,
        "unrealized_pnl": _maybe_float(position.get("unrealized_pnl")),
        "unrealized_pnl_pct": _maybe_float(position.get("unrealized_pnl_pct")),
        "position_notional": _maybe_float(position.get("position_notional")),
        "margin_used": _maybe_float(position.get("margin_used")),
        "roi_pct": _maybe_float(position.get("roi_pct")),
        "distance_to_sl_pct": _maybe_float(position.get("distance_to_sl_pct")),
        "distance_to_tp_pct": _maybe_float(position.get("distance_to_tp_pct")),
        "updated_at": _maybe_text(position.get("updated_at")),
    }


def resolve_trailing_state(position: Any) -> dict[str, Any] | None:
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
    open_trade_info = resolve_open_trade_info(position)
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
