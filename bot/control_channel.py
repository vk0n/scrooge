from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable
from bot.event_log import emit_event, get_technical_logger
from core.engine import sanitize_trade_for_history
from shared.time_utils import utc_now_iso, utc_now_text, utc_text_from_timestamp

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
technical_logger = get_technical_logger()


def _now_iso() -> str:
    return utc_now_iso()


def _status_key(command_id: str) -> str:
    return f"{COMMAND_STATUS_PREFIX}{command_id}"


def _apply_status_ttl(pipe: Any, status_key: str) -> None:
    if COMMAND_STATUS_TTL_SECONDS > 0:
        pipe.expire(status_key, COMMAND_STATUS_TTL_SECONDS)


def _log_message(logger: Callable[[str], None] | None, level: str, message: str) -> None:
    if logger is not None:
        logger(message)
        return
    getattr(technical_logger, level, technical_logger.info)(message)


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


def _ratio_to_percent(numerator: Any, denominator: Any) -> float | None:
    left = _as_float(numerator)
    right = _as_float(denominator)
    if left is None or right is None or right == 0:
        return None
    return (left / right) * 100.0


def _manual_close_trade_pnl(
    position: dict[str, Any],
    *,
    exit_price: float | None,
    fee_rate: float | None,
    leverage: float | None,
    entry_fee_paid: float | None = None,
    realized_pnl: float | None = None,
    exit_commission: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    side = str(position.get("side", "")).strip().lower()
    size = _as_float(position.get("size"))
    entry_price = _as_float(position.get("exchange_entry_price")) or _as_float(position.get("entry"))
    margin_used = _as_float(position.get("exchange_isolated_margin")) or _as_float(position.get("margin_used"))
    if margin_used is None and entry_price is not None and size is not None:
        leverage_value = _as_float(leverage)
        if leverage_value is not None and leverage_value > 0:
            margin_used = (entry_price * size) / leverage_value

    if side not in {"long", "short"} or size is None or size <= 0 or entry_price is None or entry_price <= 0 or exit_price is None:
        return None, None, None, margin_used

    position_value = entry_price * size
    if realized_pnl is not None:
        gross_pnl = realized_pnl
    elif side == "long":
        gross_pnl = (exit_price - entry_price) * size
    else:
        gross_pnl = (entry_price - exit_price) * size

    fee_total = None
    if entry_fee_paid is not None and exit_commission is not None:
        fee_total = entry_fee_paid + exit_commission
    else:
        fee_rate_value = _as_float(fee_rate)
        if fee_rate_value is not None and fee_rate_value >= 0:
            fee_total = position_value * fee_rate_value * 2.0

    net_pnl = gross_pnl - fee_total if fee_total is not None else gross_pnl
    return net_pnl, gross_pnl, fee_total, margin_used


def _validate_level_update(action: str, new_value: float, position: dict[str, Any]) -> None:
    side = str(position.get("side", "")).lower()
    entry = _as_float(position.get("exchange_entry_price")) or _as_float(position.get("entry"))
    sl = _as_float(position.get("sl"))
    tp = _as_float(position.get("tp"))
    liq_price = _as_float(position.get("exchange_liq_price")) or _as_float(position.get("liq_price"))

    if action == "update_sl":
        if side == "long":
            if liq_price is not None and new_value <= liq_price:
                raise ValueError("SL for long position must stay above liquidation price")
            if tp is not None and new_value >= tp:
                raise ValueError("Safety Net for long position must stay below Treasure Mark")
        elif side == "short":
            if liq_price is not None and new_value >= liq_price:
                raise ValueError("SL for short position must stay below liquidation price")
            if tp is not None and new_value <= tp:
                raise ValueError("Safety Net for short position must stay above Treasure Mark")
    elif action == "update_tp":
        if side == "long":
            if entry is not None and new_value <= entry:
                raise ValueError("TP for long position must be above entry price")
            if sl is not None and new_value <= sl:
                raise ValueError("Treasure Mark for long position must stay above Safety Net")
        if side == "short":
            if entry is not None and new_value >= entry:
                raise ValueError("TP for short position must be below entry price")
            if sl is not None and new_value >= sl:
                raise ValueError("Treasure Mark for short position must stay below Safety Net")


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
            "entry": _as_float(position.get("exchange_entry_price")) or _as_float(position.get("entry")),
            "sl": _as_float(position.get("sl")),
            "tp": _as_float(position.get("tp")),
            "liq_price": _as_float(position.get("exchange_liq_price")) or _as_float(position.get("liq_price")),
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
        technical_logger.warning("control_channel_disabled reason=redis_package_missing")
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
        technical_logger.warning(
            "control_redis_unavailable host=%s port=%s db=%s error=%s",
            REDIS_HOST,
            REDIS_PORT,
            REDIS_DB,
            exc,
        )
        return None
    return client


def _update_status(client: Any, command_id: str, fields: dict[str, Any], logger: Callable[[str], None] | None) -> None:
    payload = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in fields.items()}
    key = _status_key(command_id)
    try:
        pipe = client.pipeline()
        pipe.hset(key, mapping=payload)
        _apply_status_ttl(pipe, key)
        pipe.execute()
    except RedisError as exc:
        _log_message(logger, "warning", f"[CONTROL] Failed to update command status for {command_id}: {exc}")


def process_pending_commands(
    client: Any | None,
    state: dict[str, Any],
    save_state_fn: Callable[[dict[str, Any]], None],
    logger: Callable[[str], None] | None = None,
    symbol: str | None = None,
    close_position_fn: Callable[[str], Any] | None = None,
    get_open_position_fn: Callable[[str], Any] | None = None,
    get_balance_fn: Callable[[], float] | None = None,
    update_position_fn: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None,
    update_balance_fn: Callable[[dict[str, Any], float], None] | None = None,
    add_closed_trade_fn: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    get_order_execution_summary_fn: Callable[[str, Any], dict[str, Any] | None] | None = None,
    leverage: float | None = None,
    fee_rate: float | None = None,
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
            _log_message(logger, "warning", f"[CONTROL] Failed to poll command queue: {exc}")
            break

        if raw_payload is None:
            break

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            _log_message(logger, "warning", f"[CONTROL] Ignoring malformed command payload: {raw_payload}")
            continue

        command_id = str(payload.get("id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        command_payload = payload.get("payload")
        if not isinstance(command_payload, dict):
            command_payload = {}
        if not command_id:
            _log_message(logger, "warning", f"[CONTROL] Ignoring command without id: {payload}")
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
            event_ts = utc_now_text(TRADE_TIMESTAMP_FORMAT)
            if action == "start":
                state["trading_enabled"] = True
                message = "Trading resumed." if not was_enabled else "Trading is already running."
                if not was_enabled:
                    emit_event(
                        code="bot_started",
                        category="lifecycle",
                        ts=event_ts,
                        persist_ui=True,
                        symbol=symbol,
                    )
            elif action == "stop":
                state["trading_enabled"] = False
                state["manual_trade_suggestion"] = None
                message = "Trading paused." if was_enabled else "Trading is already paused."
                if was_enabled:
                    emit_event(
                        code="bot_stopped",
                        category="lifecycle",
                        ts=event_ts,
                        persist_ui=True,
                        symbol=symbol,
                    )
            elif action == "restart":
                state["trading_enabled"] = True
                restart_requested = True
                message = "Trading restart requested. Config reload scheduled."
                save_state_fn(state)
                emit_event(
                    code="bot_restarted",
                    category="lifecycle",
                    ts=event_ts,
                    persist_ui=True,
                    symbol=symbol,
                    notify=True,
                )
            elif action == "suggest_trade":
                if isinstance(state.get("position"), dict):
                    raise ValueError("Cannot suggest a trade while a position is already open")
                exchange_position = get_open_position_fn(symbol) if (get_open_position_fn and symbol) else None
                if isinstance(exchange_position, dict):
                    raise ValueError(
                        "Cannot suggest a trade while the exchange holds an unmanaged live position. "
                        "Please clear it first."
                    )

                requested_side = _normalize_suggested_side(command_payload.get("side"))
                if requested_side is None:
                    raise ValueError("Invalid trade side. Expected 'buy' or 'sell'.")

                resumed_for_suggestion = not bool(state.get("trading_enabled", True))
                if resumed_for_suggestion:
                    state["trading_enabled"] = True
                    emit_event(
                        code="bot_started",
                        category="lifecycle",
                        ts=event_ts,
                        persist_ui=True,
                        symbol=symbol,
                    )

                state["manual_trade_suggestion"] = {
                    "side": requested_side,
                    "requested_at": _now_iso(),
                    "requested_by": str(payload.get("requested_by", "unknown")).strip() or "unknown",
                }
                save_state_fn(state)
                if resumed_for_suggestion:
                    message = (
                        f"{requested_side.capitalize()} suggestion recorded. "
                        "The office has reopened, and Scrooge will try it on the next live tick."
                    )
                else:
                    message = (
                        f"{requested_side.capitalize()} suggestion recorded. "
                        f"Scrooge will try it on the next live tick and manage it by the usual rules."
                    )
                emit_event(
                    code="manual_trade_suggested",
                    category="command",
                    ts=event_ts,
                    persist_ui=True,
                    symbol=symbol,
                    side="long" if requested_side == "buy" else "short",
                    requested_by=payload.get("requested_by"),
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
                    emit_event(
                        code="level_updated",
                        category="risk",
                        ts=event_ts,
                        persist_ui=True,
                        symbol=symbol,
                        side=position.get("side"),
                        level_type=field_name,
                        old_value=old_value,
                        new_value=new_value,
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
                elif not has_local_position and has_exchange_position:
                    raise ValueError(
                        "Refusing to close the live exchange position because it is not tracked in Scrooge state."
                    )
                elif has_local_position and bool(local_position.get("close_pending")):
                    message = "Close request already in flight; awaiting exchange confirmation."
                else:
                    pending_position = dict(local_position)
                    pending_position["close_pending"] = True
                    pending_position["close_requested_at"] = event_ts
                    state["position"] = pending_position
                    if update_position_fn is not None:
                        update_position_fn(state, pending_position)
                    else:
                        save_state_fn(state)

                    previous_balance = _as_float(state.get("balance"))
                    fallback_exit_price = _as_float(exchange_position.get("markPrice")) if has_exchange_position else None
                    order_result = close_position_fn(symbol)
                    if has_exchange_position and order_result is None:
                        state["position"] = local_position
                        if update_position_fn is not None:
                            update_position_fn(state, local_position)
                        else:
                            save_state_fn(state)
                        raise RuntimeError("Exchange close request did not return order confirmation")

                    latest_balance = None
                    if get_balance_fn is not None:
                        latest_balance = _as_float(get_balance_fn())

                    if has_local_position and isinstance(local_position, dict):
                        trade_record = dict(local_position)
                        trade_record["exit_reason"] = "manual_close"
                        trade_record["exit_time"] = utc_now_text(TRADE_TIMESTAMP_FORMAT)
                        execution_summary = (
                            get_order_execution_summary_fn(symbol, order_result)
                            if get_order_execution_summary_fn is not None
                            else None
                        )
                        if isinstance(execution_summary, dict) and execution_summary.get("update_time_ms") is not None:
                            try:
                                trade_record["exit_time"] = utc_text_from_timestamp(
                                    execution_summary["update_time_ms"],
                                    TRADE_TIMESTAMP_FORMAT,
                                )
                            except (TypeError, ValueError, OSError):
                                pass
                        exit_price = _extract_exit_price(order_result)
                        if isinstance(execution_summary, dict):
                            summary_exit_price = _as_float(execution_summary.get("avg_price"))
                            if summary_exit_price is not None and summary_exit_price > 0:
                                exit_price = summary_exit_price
                        if exit_price is None:
                            exit_price = fallback_exit_price
                        if exit_price is None:
                            exit_price = _as_float(local_position.get("entry"))
                        if exit_price is not None:
                            trade_record["exit"] = exit_price
                        entry_fee_paid = _as_float(local_position.get("entry_fee_paid"))
                        realized_pnl = (
                            _as_float(execution_summary.get("realized_pnl"))
                            if isinstance(execution_summary, dict)
                            else None
                        )
                        exit_commission = (
                            _as_float(execution_summary.get("commission"))
                            if isinstance(execution_summary, dict)
                            else None
                        )
                        computed_net_pnl, computed_gross_pnl, computed_fee_total, margin_used = _manual_close_trade_pnl(
                            local_position,
                            exit_price=exit_price,
                            fee_rate=fee_rate,
                            leverage=leverage,
                            entry_fee_paid=entry_fee_paid,
                            realized_pnl=realized_pnl,
                            exit_commission=exit_commission,
                        )
                        if computed_net_pnl is not None:
                            trade_record["net_pnl"] = computed_net_pnl
                        elif previous_balance is not None and latest_balance is not None:
                            trade_record["net_pnl"] = latest_balance - previous_balance
                        if computed_gross_pnl is not None:
                            trade_record["gross_pnl"] = computed_gross_pnl
                        if computed_fee_total is not None:
                            trade_record["fee"] = computed_fee_total
                        if margin_used is not None:
                            trade_record["margin_used"] = margin_used
                            if trade_record.get("net_pnl") is not None and margin_used > 0:
                                try:
                                    trade_record["roi_pct"] = (float(trade_record["net_pnl"]) / margin_used) * 100.0
                                except (TypeError, ValueError):
                                    pass
                        if entry_fee_paid is not None:
                            trade_record["entry_fee"] = entry_fee_paid
                        if exit_commission is not None:
                            trade_record["exit_fee"] = exit_commission
                        if isinstance(execution_summary, dict):
                            trade_record["exit_order_id"] = execution_summary.get("order_id")
                        pending_position = dict(state.get("position")) if isinstance(state.get("position"), dict) else dict(local_position)
                        pending_position["close_pending"] = True
                        pending_position["close_requested_at"] = event_ts
                        pending_position["pending_close_trade"] = sanitize_trade_for_history(trade_record)
                        pending_position["pending_close_event"] = {
                            "code": "trade_closed_manual",
                            "category": "trade",
                            "level": "info",
                            "notify": True,
                            "symbol": symbol,
                            "side": local_position.get("side"),
                            "exit": trade_record.get("exit"),
                            "net_pnl": trade_record.get("net_pnl"),
                            "roi_pct": _ratio_to_percent(
                                trade_record.get("net_pnl"),
                                trade_record.get("margin_used"),
                            ),
                        }
                        pending_position["pending_close_event_ts"] = str(trade_record.get("exit_time") or event_ts)
                        pending_position["pending_close_runtime_mode"] = "live"
                        pending_position["pending_close_strategy_mode"] = None
                        if isinstance(execution_summary, dict):
                            pending_position["pending_close_order_id"] = execution_summary.get("order_id")
                        state["position"] = pending_position
                        if latest_balance is not None and update_balance_fn is not None:
                            if update_position_fn is not None:
                                update_position_fn(state, pending_position)
                            else:
                                save_state_fn(state)
                            update_balance_fn(state, latest_balance)
                        else:
                            if latest_balance is not None:
                                state["balance"] = latest_balance
                            if update_position_fn is not None:
                                update_position_fn(state, pending_position)
                            else:
                                save_state_fn(state)
                    else:
                        if latest_balance is not None and update_balance_fn is not None:
                            update_balance_fn(state, latest_balance)
                        elif latest_balance is not None:
                            state["balance"] = latest_balance
                            save_state_fn(state)

                    message = "Manual close submitted; awaiting exchange confirmation."
            else:
                raise ValueError(f"Unsupported action: {action}")

            if action in {"start", "stop"}:
                save_state_fn(state)
        except Exception as exc:  # noqa: BLE001
            emit_event(
                code="command_failed",
                category="error",
                ts=utc_now_text(TRADE_TIMESTAMP_FORMAT),
                level="warning",
                persist_ui=True,
                action=action,
                reason=str(exc),
                symbol=symbol,
            )
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
        _log_message(logger, "info", f"[CONTROL] Applied command {action} ({command_id})")

    return state, restart_requested
