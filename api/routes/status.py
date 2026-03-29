from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.config_service import load_config
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


def _is_live_mode(config: dict[str, object]) -> bool:
    live_value = config.get("live")
    if live_value is None:
        return True
    if isinstance(live_value, str):
        return live_value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(live_value)


def _default_balance_for_mode(config: dict[str, object]) -> object | None:
    return None if _is_live_mode(config) else config.get("initial_balance")


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
    last_price = resolve_last_price(state)
    open_trade_info = resolve_open_trade_info(
        position,
        state=state,
        leverage=config.get("leverage"),
        last_price=last_price,
    )
    balance_value = resolve_balance(state, default_balance=_default_balance_for_mode(config))
    if _is_live_mode(config) and balance_value is None:
        warnings.append("Live balance unavailable: exchange balance has not been written to state yet.")

    return {
        "bot_running_status": bot_running_status,
        "trading_enabled": trading_enabled,
        "balance": balance_value,
        "last_price": last_price,
        "last_price_updated_at": resolve_last_price_updated_at(state),
        "bot_status": resolve_bot_status(state),
        "current_position": open_trade_info["side"] if open_trade_info else None,
        "leverage": config.get("leverage"),
        "symbol": config.get("symbol"),
        "trailing_state": resolve_trailing_state(position, open_trade_info=open_trade_info),
        "trade_status": resolve_trade_status(state),
        "open_trade_info": open_trade_info,
        "last_update_timestamp": resolve_last_update_timestamp(state),
        "warnings": warnings,
    }
