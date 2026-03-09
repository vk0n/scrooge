from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.config_service import load_config
from services.state_service import (
    load_state,
    resolve_balance,
    resolve_current_position,
    resolve_last_update_timestamp,
    resolve_trailing_state,
)

router = APIRouter()


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

    position = state.get("position")
    return {
        "bot_running_status": "unknown",
        "balance": resolve_balance(state, default_balance=config.get("initial_balance")),
        "current_position": resolve_current_position(position),
        "leverage": config.get("leverage"),
        "symbol": config.get("symbol"),
        "trailing_state": resolve_trailing_state(position),
        "open_trade_info": position,
        "last_update_timestamp": resolve_last_update_timestamp(state),
        "warnings": warnings,
    }
