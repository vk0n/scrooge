from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Callable

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - bot image installs redis dependency
    redis = None
    RedisError = Exception


REDIS_HOST = os.getenv("SCROOGE_REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("SCROOGE_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("SCROOGE_REDIS_DB", "0"))
CONTROL_QUEUE_KEY = os.getenv("SCROOGE_CONTROL_QUEUE_KEY", "scrooge:control:queue")
COMMAND_STATUS_PREFIX = os.getenv("SCROOGE_COMMAND_STATUS_PREFIX", "scrooge:control:command:")
COMMAND_STATUS_TTL_SECONDS = int(os.getenv("SCROOGE_COMMAND_STATUS_TTL_SECONDS", "86400"))
SUPPORTED_ACTIONS = {"start", "stop", "restart", "close_position", "suggest_trade", "update_sl", "update_tp"}
TRADE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _status_key(command_id: str) -> str:
    return f"{COMMAND_STATUS_PREFIX}{command_id}"


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:  # NaN check without math import
        return None
    return numeric


def _extract_exit_price(order_response: Any) -> float | None:
    if not isinstance(order_response, dict):
        return None
    for key in ("avgPrice", "price", "stopPrice"):
        candidate = _as_float(order_response.get(key))
        if candidate is not None and candidate > 0:
            return candidate
    return None


def _validate_level_update(action: str, new_value: float, position: dict[str, Any]) -> None:
    side = str(position.get("side", "")).lower()
    entry = _as_float(position.get("entry"))
    liq_price = _as_float(position.get("liq_price"))

    if action == "update_sl":
        if side == "long":
            if entry is not None and new_value >= entry:
                raise ValueError("SL for long position must be below entry price")
            if liq_price is not None and new_value <= liq_price:
                raise ValueError("SL for long position must stay above liquidation price")
        elif side == "short":
            if entry is not None and new_value <= entry:
                raise ValueError("SL for short position must be above entry price")
            if liq_price is not None and new_value >= liq_price:
                raise ValueError("SL for short position must stay below liquidation price")
    elif action == "update_tp":
        if side == "long" and entry is not None and new_value <= entry:
            raise ValueError("TP for long position must be above entry price")
        if side == "short" and entry is not None and new_value >= entry:
            raise ValueError("TP for short position must be below entry price")


def _normalize_suggested_side(value: Any) -> str | None:
    if value is None:
        return None
    side = str(value).strip().lower()
    return side if side in {"buy", "sell"} else None


def _status_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "trading_enabled": bool(state.get("trading_enabled", True)),
    }
    position = state.get("position")
    if isinstance(position, dict):
        snapshot["position"] = {
            "side": position.get("side"),
            "entry": _as_float(position.get("entry")),
            "sl": _as_float(position.get("sl")),
            "tp": _as_float(position.get("tp")),
            "liq_price": _as_float(position.get("liq_price")),
            "trail_active": bool(position.get("trail_active", False)),
        }
    else:
        snapshot["position"] = None
    return snapshot


def get_control_client() -> Any | None:
    """
    Create Redis client for bot-side control command polling.
    Returns None when Redis support is unavailable or unreachable.
    """
    if redis is None:
        print("[CONTROL] redis package is not installed; command polling disabled")
        return None

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    try:
        client.ping()
    except RedisError as exc:
        print(f"[CONTROL] Redis unavailable ({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}): {exc}")
        return None
    return client


def _update_status(client: Any, command_id: str, fields: dict[str, Any], logger: Callable[[str], None]) -> None:
    payload = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in fields.items()}
    key = _status_key(command_id)
    try:
        pipe = client.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.expire(key, COMMAND_STATUS_TTL_SECONDS)
        pipe.execute()
    except RedisError as exc:
        logger(f"[CONTROL] Failed to update command status for {command_id}: {exc}")


def process_pending_commands(
    client: Any | None,
    state: dict[str, Any],
    save_state_fn: Callable[[dict[str, Any]], None],
    logger: Callable[[str], None] = print,
    symbol: str | None = None,
    close_position_fn: Callable[[str], Any] | None = None,
    get_open_position_fn: Callable[[str], Any] | None = None,
    get_balance_fn: Callable[[], float] | None = None,
    update_position_fn: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None,
    update_balance_fn: Callable[[dict[str, Any], float], None] | None = None,
    add_closed_trade_fn: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Consume queued control commands and apply them to bot runtime state.
    Returns updated state and a restart marker (config reload requested).
    """
    if client is None:
        return state, False

    restart_requested = False

    while True:
        try:
            raw_payload = client.lpop(CONTROL_QUEUE_KEY)
        except RedisError as exc:
            logger(f"[CONTROL] Failed to poll command queue: {exc}")
            break

        if raw_payload is None:
            break

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            logger(f"[CONTROL] Ignoring malformed command payload: {raw_payload}")
            continue

        command_id = str(payload.get("id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        command_payload = payload.get("payload")
        if not isinstance(command_payload, dict):
            command_payload = {}
        if not command_id:
            logger(f"[CONTROL] Ignoring command without id: {payload}")
            continue

        _update_status(
            client,
            command_id,
            {"status": "processing", "updated_at": _now_iso(), "message": ""},
            logger,
        )

        if action not in SUPPORTED_ACTIONS:
            _update_status(
                client,
                command_id,
                {
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "message": f"Unsupported action: {action}",
                },
                logger,
            )
            continue

        try:
            was_enabled = bool(state.get("trading_enabled", True))
            if action == "start":
                state["trading_enabled"] = True
                message = "Trading resumed." if not was_enabled else "Trading is already running."
            elif action == "stop":
                state["trading_enabled"] = False
                state["manual_trade_suggestion"] = None
                message = "Trading paused." if was_enabled else "Trading is already paused."
            elif action == "restart":
                state["trading_enabled"] = True
                restart_requested = True
                message = "Trading restart requested. Config reload scheduled."
                save_state_fn(state)
            elif action == "suggest_trade":
                if not bool(state.get("trading_enabled", True)):
                    raise ValueError("Cannot suggest a trade while trading is paused")
                if isinstance(state.get("position"), dict):
                    raise ValueError("Cannot suggest a trade while a position is already open")

                requested_side = _normalize_suggested_side(command_payload.get("side"))
                if requested_side is None:
                    raise ValueError("Invalid trade side. Expected 'buy' or 'sell'.")

                state["manual_trade_suggestion"] = {
                    "side": requested_side,
                    "requested_at": _now_iso(),
                    "requested_by": str(payload.get("requested_by", "unknown")).strip() or "unknown",
                }
                save_state_fn(state)
                message = (
                    f"{requested_side.capitalize()} suggestion recorded. "
                    f"Scrooge will try it on the next live tick and manage it by the usual rules."
                )
            elif action in {"update_sl", "update_tp"}:
                position = state.get("position")
                if not isinstance(position, dict):
                    message = "Skipped: no open position in state."
                else:
                    new_value = _as_float(command_payload.get("value"))
                    if new_value is None or new_value <= 0:
                        raise ValueError("Invalid price value. Expected positive number.")
                    _validate_level_update(action, new_value, position)

                    field_name = "sl" if action == "update_sl" else "tp"
                    old_value = _as_float(position.get(field_name))
                    position[field_name] = new_value
                    state["position"] = position
                    save_state_fn(state)
                    message = (
                        f"Updated {field_name.upper()} from {old_value if old_value is not None else 'n/a'} "
                        f"to {new_value}."
                    )
            elif action == "close_position":
                if close_position_fn is None:
                    raise ValueError("Close-position handler is not configured")
                if not symbol:
                    raise ValueError("Close-position requires configured symbol")

                exchange_position = get_open_position_fn(symbol) if get_open_position_fn else None
                local_position = state.get("position")
                has_local_position = isinstance(local_position, dict)
                has_exchange_position = isinstance(exchange_position, dict)
                if not has_local_position and not has_exchange_position:
                    message = "Skipped: no open position to close."
                else:
                    previous_balance = _as_float(state.get("balance"))
                    fallback_exit_price = _as_float(exchange_position.get("markPrice")) if has_exchange_position else None
                    order_result = close_position_fn(symbol)
                    if has_exchange_position and order_result is None:
                        raise RuntimeError("Exchange close request did not return order confirmation")

                    if has_local_position and update_position_fn is not None:
                        update_position_fn(state, None)
                    elif has_local_position:
                        state["position"] = None
                        save_state_fn(state)

                    latest_balance = None
                    if get_balance_fn is not None:
                        latest_balance = _as_float(get_balance_fn())
                        if latest_balance is not None and update_balance_fn is not None:
                            update_balance_fn(state, latest_balance)
                        elif latest_balance is not None:
                            state["balance"] = latest_balance
                            save_state_fn(state)

                    if has_local_position and add_closed_trade_fn is not None and isinstance(local_position, dict):
                        trade_record = dict(local_position)
                        trade_record["exit_reason"] = "manual_close"
                        trade_record["exit_time"] = datetime.now().strftime(TRADE_TIMESTAMP_FORMAT)
                        exit_price = _extract_exit_price(order_result)
                        if exit_price is None:
                            exit_price = fallback_exit_price
                        if exit_price is None:
                            exit_price = _as_float(local_position.get("entry"))
                        if exit_price is not None:
                            trade_record["exit"] = exit_price
                        if previous_balance is not None and latest_balance is not None:
                            trade_record["net_pnl"] = latest_balance - previous_balance
                        add_closed_trade_fn(state, trade_record)

                    message = "Manual close command executed."
                    if not has_local_position and has_exchange_position:
                        message = "Closed exchange position; local state had no open position."
            else:
                raise ValueError(f"Unsupported action: {action}")

            if action in {"start", "stop"}:
                save_state_fn(state)
        except Exception as exc:  # noqa: BLE001
            _update_status(
                client,
                command_id,
                {
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "message": f"Failed to apply command: {exc}",
                },
                logger,
            )
            continue

        _update_status(
            client,
            command_id,
            {
                "status": "completed",
                "updated_at": _now_iso(),
                "message": message,
                "trading_status": _status_snapshot(state),
            },
            logger,
        )
        logger(f"[CONTROL] Applied command {action} ({command_id})")

    return state, restart_requested
