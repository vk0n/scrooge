from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.config_service import load_config
from services.state_service import (
    load_state,
    resolve_balance,
    resolve_current_position,
    resolve_last_update_timestamp,
    resolve_trading_enabled,
    resolve_trailing_state,
)

router = APIRouter()


def _default_balance_for_mode(config: dict[str, object]) -> object | None:
    live_value = config.get("live")
    if live_value is None:
        is_live_mode = True
    elif isinstance(live_value, str):
        is_live_mode = live_value.strip().lower() not in {"0", "false", "no", "off"}
    else:
        is_live_mode = bool(live_value)
    return None if is_live_mode else config.get("initial_balance")


@router.get("")
def get_status() -> dict[str, object]:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        state, warnings = load_state()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    trading_enabled = resolve_trading_enabled(state)
    bot_running_status = "running" if trading_enabled else "paused"

    position = state.get("position")
    return {
        "bot_running_status": bot_running_status,
        "trading_enabled": trading_enabled,
        "balance": resolve_balance(state, default_balance=_default_balance_for_mode(config)),
        "current_position": resolve_current_position(position),
        "leverage": config.get("leverage"),
        "symbol": config.get("symbol"),
        "trailing_state": resolve_trailing_state(position),
        "open_trade_info": position,
        "last_update_timestamp": resolve_last_update_timestamp(state),
        "warnings": warnings,
    }
