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
from services.system_service import get_service_status

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

    service_status = None
    try:
        svc = get_service_status()
        service_status = {
            "name": svc.service_name,
            "running": svc.running,
            "active_state": svc.active_state,
            "sub_state": svc.sub_state,
            "unit_file_state": svc.unit_file_state,
        }
        bot_running_status = "running" if svc.running else "stopped"
    except RuntimeError as exc:
        warnings.append(str(exc))
        bot_running_status = "unknown"

    position = state.get("position")
    return {
        "bot_running_status": bot_running_status,
        "service_status": service_status,
        "balance": resolve_balance(state, default_balance=config.get("initial_balance")),
        "current_position": resolve_current_position(position),
        "leverage": config.get("leverage"),
        "symbol": config.get("symbol"),
        "trailing_state": resolve_trailing_state(position),
        "open_trade_info": position,
        "last_update_timestamp": resolve_last_update_timestamp(state),
        "warnings": warnings,
    }
