from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.auth_service import require_ws_auth
from services.config_service import load_config
from services.log_service import LOG_PATH, read_last_log_lines
from services.state_service import (
    load_state,
    resolve_balance,
    resolve_bot_status,
    resolve_last_price,
    resolve_last_price_updated_at,
    resolve_last_update_timestamp,
    resolve_open_trade_info,
    resolve_trade_status,
    resolve_trading_enabled,
    resolve_trailing_state,
)

router = APIRouter()

WS_PUSH_INTERVAL_SECONDS = float(os.getenv("SCROOGE_WS_PUSH_INTERVAL_SECONDS", "2"))
WS_DEFAULT_LOG_LINES = int(os.getenv("SCROOGE_WS_LOG_LINES", "200"))


def _is_live_mode(config: dict[str, Any]) -> bool:
    live_value = config.get("live")
    if live_value is None:
        return True
    if isinstance(live_value, str):
        return live_value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(live_value)


def _default_balance_for_mode(config: dict[str, Any]) -> Any | None:
    return None if _is_live_mode(config) else config.get("initial_balance")


def _build_status_snapshot() -> dict[str, Any]:
    warnings: list[str] = []
    config: dict[str, Any] = {}
    state: dict[str, Any] = {
        "position": None,
        "balance": None,
        "last_price": None,
        "last_price_updated_at": None,
        "bot_status": None,
        "trade_status": None,
        "session_start": None,
        "session_end": None,
        "trading_enabled": True,
    }

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

    trading_enabled = resolve_trading_enabled(state)
    position = state.get("position")
    open_trade_info = resolve_open_trade_info(position)
    balance_value = resolve_balance(state, default_balance=_default_balance_for_mode(config))
    if _is_live_mode(config) and balance_value is None:
        warnings.append("Live balance unavailable: exchange balance has not been written to state yet.")

    return {
        "bot_running_status": "running" if trading_enabled else "paused",
        "trading_enabled": trading_enabled,
        "balance": balance_value,
        "last_price": resolve_last_price(state),
        "last_price_updated_at": resolve_last_price_updated_at(state),
        "bot_status": resolve_bot_status(state),
        "current_position": open_trade_info["side"] if open_trade_info else None,
        "leverage": config.get("leverage"),
        "symbol": config.get("symbol"),
        "trailing_state": resolve_trailing_state(position),
        "trade_status": resolve_trade_status(state),
        "open_trade_info": open_trade_info,
        "last_update_timestamp": resolve_last_update_timestamp(state),
        "warnings": warnings,
    }


def _build_logs_snapshot(lines: int) -> dict[str, Any]:
    try:
        tail_lines = read_last_log_lines(lines)
        warnings: list[str] = []
    except Exception as exc:  # noqa: BLE001
        tail_lines = []
        warnings = [str(exc)]
    return {
        "path": str(LOG_PATH),
        "requested_lines": lines,
        "returned_lines": len(tail_lines),
        "lines": tail_lines,
        "warnings": warnings,
    }


def _resolve_lines(raw_value: str | None) -> int:
    if not raw_value:
        return WS_DEFAULT_LOG_LINES
    try:
        parsed = int(raw_value)
    except ValueError:
        return WS_DEFAULT_LOG_LINES
    return max(1, min(parsed, 5000))


async def _stream_status(websocket: WebSocket) -> None:
    lines = _resolve_lines(websocket.query_params.get("lines"))

    await websocket.send_json(
        {
            "type": "hello",
            "mode": "live",
            "message": "Scrooge control WS connected",
            "timestamp": datetime.now(UTC).isoformat(),
            "push_interval_seconds": WS_PUSH_INTERVAL_SECONDS,
            "log_lines": lines,
        }
    )

    while True:
        now_iso = datetime.now(UTC).isoformat()
        await websocket.send_json(
            {
                "type": "status",
                "timestamp": now_iso,
                "data": _build_status_snapshot(),
            }
        )
        await websocket.send_json(
            {
                "type": "logs",
                "timestamp": now_iso,
                "data": _build_logs_snapshot(lines),
            }
        )
        await asyncio.sleep(max(0.5, WS_PUSH_INTERVAL_SECONDS))


@router.websocket("")
async def websocket_status(websocket: WebSocket) -> None:
    if not require_ws_auth(websocket):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()
    try:
        await _stream_status(websocket)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        # Socket may already be closed by client.
        return


@router.websocket("/status")
async def websocket_status_path(websocket: WebSocket) -> None:
    if not require_ws_auth(websocket):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()
    try:
        await _stream_status(websocket)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return
