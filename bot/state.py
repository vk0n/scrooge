from __future__ import annotations

import math
from typing import Any
from bot.event_log import get_technical_logger
from shared.runtime_db import (
    append_balance_history_row as append_balance_history_db_row,
    append_trade_history_row as append_trade_history_db_row,
    list_balance_history_values as list_balance_history_db_values,
    list_trade_history_rows as list_trade_history_db_rows,
    load_runtime_state_snapshot as load_runtime_state_db_snapshot,
    replace_balance_history_snapshot as replace_balance_history_db_snapshot,
    replace_trade_history_snapshot as replace_trade_history_db_snapshot,
    save_runtime_state_snapshot as save_runtime_state_db_snapshot,
)
from shared.time_utils import utc_now_ms, utc_now_text

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

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
technical_logger = get_technical_logger()


def _now_ms() -> int:
    return utc_now_ms()


def _now_text() -> str:
    return utc_now_text(TIMESTAMP_FORMAT)


def _default_state() -> dict[str, Any]:
    now_ms = _now_ms()
    return {
        "position": None,
        "balance": None,
        "manual_trade_suggestion": None,
        "last_price": None,
        "last_price_updated_at": None,
        "updated_at": _now_text(),
        "search_status": None,
        "bot_status": None,
        "trade_status": None,
        "session_start": now_ms,
        "session_end": None,
        "trading_enabled": True,
    }


def _as_int_or_default(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_manual_trade_suggestion(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None

    side = _as_text_or_none(value.get("side"))
    if side not in {"buy", "sell"}:
        return None

    return {
        "side": side,
        "requested_at": _as_text_or_none(value.get("requested_at")),
        "requested_by": _as_text_or_none(value.get("requested_by")),
    }


def _status_from_code(code: str, labels: dict[str, str]) -> dict[str, str] | None:
    label = labels.get(code)
    if label is None:
        return None
    return {"code": code, "label": label}


def _normalize_ui_status(value: Any, labels: dict[str, str]) -> dict[str, str] | None:
    if isinstance(value, dict):
        raw_code = value.get("code")
    else:
        raw_code = value

    code = _as_text_or_none(raw_code)
    if code is None:
        return None
    return _status_from_code(code, labels)


def _trade_status_from_position(position: Any) -> dict[str, str] | None:
    if not isinstance(position, dict):
        return None

    if bool(position.get("trail_active", False)):
        return _status_from_code("locking_in_profit", TRADE_STATUS_LABELS)

    unrealized_pnl = _as_float_or_none(position.get("unrealized_pnl"))
    if unrealized_pnl is not None and unrealized_pnl < 0:
        return _status_from_code("waiting_for_profit", TRADE_STATUS_LABELS)
    return _status_from_code("waiting_for_more_profit", TRADE_STATUS_LABELS)


def _sync_ui_statuses(state: dict[str, Any]) -> None:
    position = state.get("position")
    trading_enabled = bool(state.get("trading_enabled", True))

    search_status = _normalize_ui_status(state.get("search_status"), SEARCH_STATUS_LABELS)
    if search_status is None:
        search_status = _normalize_ui_status(state.get("bot_status"), SEARCH_STATUS_LABELS)

    trade_status = _trade_status_from_position(position)
    if isinstance(position, dict):
        bot_status = _status_from_code("managing_open_trade", BOT_STATUS_LABELS)
        state["manual_trade_suggestion"] = None
    elif not trading_enabled:
        bot_status = _status_from_code("resting", BOT_STATUS_LABELS)
    else:
        bot_status = search_status

    state["search_status"] = search_status
    state["bot_status"] = bot_status
    state["trade_status"] = trade_status


def _normalize_state(raw_state: Any) -> dict[str, Any]:
    default = _default_state()
    if not isinstance(raw_state, dict):
        return default

    normalized = dict(raw_state)

    position = normalized.get("position")
    if position is not None and not isinstance(position, dict):
        normalized["position"] = None
    else:
        normalized.setdefault("position", None)

    normalized["balance"] = _as_float_or_none(normalized.get("balance"))
    normalized["manual_trade_suggestion"] = _normalize_manual_trade_suggestion(normalized.get("manual_trade_suggestion"))
    normalized["last_price"] = _as_float_or_none(normalized.get("last_price"))
    normalized["last_price_updated_at"] = _as_text_or_none(normalized.get("last_price_updated_at"))
    normalized["updated_at"] = _as_text_or_none(normalized.get("updated_at"))
    normalized["search_status"] = _normalize_ui_status(normalized.get("search_status"), SEARCH_STATUS_LABELS)
    normalized["bot_status"] = _normalize_ui_status(normalized.get("bot_status"), BOT_STATUS_LABELS)
    normalized["trade_status"] = _normalize_ui_status(normalized.get("trade_status"), TRADE_STATUS_LABELS)
    normalized["session_start"] = _as_int_or_default(normalized.get("session_start"), default["session_start"])
    normalized["session_end"] = _as_int_or_default(normalized.get("session_end"), None)
    normalized["trading_enabled"] = bool(normalized.get("trading_enabled", True))

    for key, value in default.items():
        normalized.setdefault(key, value)

    return normalized


def _normalize_trade_history(raw_history: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_history, list):
        return []
    output: list[dict[str, Any]] = []
    for item in raw_history:
        if isinstance(item, dict):
            output.append(item)
    return output


def _normalize_balance_history(raw_history: Any) -> list[float]:
    if not isinstance(raw_history, list):
        return []
    output: list[float] = []
    for item in raw_history:
        if isinstance(item, dict):
            candidate = item.get("balance")
        else:
            candidate = item
        value = _as_float_or_none(candidate)
        if value is not None:
            output.append(value)
    return output


def _migrate_legacy_history_keys(state: dict[str, Any]) -> dict[str, Any]:
    has_trade_key = "trade_history" in state
    has_balance_key = "balance_history" in state
    if not has_trade_key and not has_balance_key:
        return state

    trade_history = _normalize_trade_history(state.get("trade_history"))
    balance_history = _normalize_balance_history(state.get("balance_history"))

    if has_trade_key and trade_history:
        try:
            replace_trade_history_db_snapshot(trade_history)
            technical_logger.info("trade_history_migrated_to_db count=%s", len(trade_history))
        except OSError as exc:
            technical_logger.warning("trade_history_migration_failed error=%s", exc)

    if has_balance_key and balance_history:
        try:
            replace_balance_history_db_snapshot(balance_history)
            technical_logger.info("balance_history_migrated_to_db count=%s", len(balance_history))
        except OSError as exc:
            technical_logger.warning("balance_history_migration_failed error=%s", exc)

    state.pop("trade_history", None)
    state.pop("balance_history", None)
    return state


def load_trade_history(limit: int | None = None) -> list[dict[str, Any]]:
    try:
        trades = list_trade_history_db_rows()
    except OSError as exc:
        technical_logger.warning("trade_history_db_read_failed error=%s", exc)
        return []
    if limit is not None and limit > 0:
        return trades[-limit:]
    return trades


def load_balance_history(limit: int | None = None) -> list[float]:
    try:
        values = list_balance_history_db_values()
    except OSError as exc:
        technical_logger.warning("balance_history_db_read_failed error=%s", exc)
        return []
    return values[-limit:] if limit is not None and limit > 0 else values


def _write_trade_history_snapshot(history: list[dict[str, Any]]) -> None:
    replace_trade_history_db_snapshot(history)


def _write_balance_history_snapshot(history: list[float]) -> None:
    replace_balance_history_db_snapshot(history)


def load_state(include_history: bool = False) -> dict[str, Any]:
    """
    Load runtime state snapshot.
    `trade_history` and `balance_history` are loaded only when include_history=True.
    """
    try:
        db_state = load_runtime_state_db_snapshot()
    except OSError as exc:
        technical_logger.warning("state_db_read_failed error=%s", exc)
        db_state = None

    if isinstance(db_state, dict):
        normalized = _normalize_state(db_state)
        normalized = _migrate_legacy_history_keys(normalized)
        if include_history:
            normalized["trade_history"] = load_trade_history()
            normalized["balance_history"] = load_balance_history()
        return normalized

    default = _default_state()
    if include_history:
        default["trade_history"] = load_trade_history()
        default["balance_history"] = load_balance_history()
    return default


def save_state(state: dict[str, Any]) -> None:
    """
    Persist the normalized runtime snapshot to the SQLite runtime store.
    If legacy history keys are present in-memory, migrate them into DB snapshots.
    """
    normalized = _normalize_state(state)
    trade_history = _normalize_trade_history(normalized.pop("trade_history", None))
    balance_history = _normalize_balance_history(normalized.pop("balance_history", None))

    if trade_history:
        try:
            _write_trade_history_snapshot(trade_history)
        except OSError as exc:
            technical_logger.warning("trade_history_snapshot_write_failed error=%s", exc)

    if balance_history:
        try:
            _write_balance_history_snapshot(balance_history)
        except OSError as exc:
            technical_logger.warning("balance_history_snapshot_write_failed error=%s", exc)

    normalized["session_end"] = _now_ms()
    if normalized.get("session_start") is None:
        normalized["session_start"] = normalized["session_end"]
    if normalized.get("updated_at") is None:
        normalized["updated_at"] = normalized.get("last_price_updated_at") or _now_text()
    _sync_ui_statuses(normalized)

    if normalized.get("balance") is None:
        latest = load_balance_history(limit=1)
        if latest:
            normalized["balance"] = latest[-1]

    try:
        save_runtime_state_db_snapshot(normalized)
    except OSError as exc:
        technical_logger.warning("state_db_write_failed error=%s", exc)
    state.clear()
    state.update(normalized)


def add_closed_trade(state: dict[str, Any], trade: dict[str, Any]) -> None:
    """
    Append a closed trade to trade history and persist.
    """
    try:
        append_trade_history_db_row(trade)
    except OSError as exc:
        technical_logger.warning("trade_history_db_append_failed error=%s", exc)
    save_state(state)


def update_position(state: dict[str, Any], position: dict[str, Any] | None) -> None:
    """
    Update current open position in state and persist.
    """
    state["position"] = position
    save_state(state)


def update_balance(state: dict[str, Any], balance: float) -> None:
    """
    Update current balance in state and persist.
    Append to balance history only when the value changed.
    """
    balance_value = _as_float_or_none(balance)
    previous_value = _as_float_or_none(state.get("balance"))
    should_append = False
    if balance_value is not None:
        if previous_value is None:
            latest = load_balance_history(limit=1)
            previous_value = latest[-1] if latest else None
        should_append = previous_value is None or not math.isclose(
            balance_value,
            previous_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    if should_append:
        balance_row = {"time": _now_ms(), "balance": balance_value}
        try:
            append_balance_history_db_row(balance_row)
        except OSError as exc:
            technical_logger.warning("balance_history_db_append_failed error=%s", exc)
    state["balance"] = balance_value
    save_state(state)


def set_trading_enabled(state: dict[str, Any], enabled: bool) -> None:
    """
    Update trading enabled flag and persist.
    """
    state["trading_enabled"] = bool(enabled)
    save_state(state)
