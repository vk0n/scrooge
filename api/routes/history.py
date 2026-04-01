from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.history_service import load_trade_history_page, load_trade_history_summary

router = APIRouter()


@router.get("/trades")
def get_trade_history(
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=30, ge=1, le=200),
    lookback_days: int | None = Query(default=None, ge=1, le=3650),
) -> dict[str, object]:
    try:
        payload, warnings = load_trade_history_page(
            page=page,
            page_size=page_size,
            lookback_days=lookback_days,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {**payload, "warnings": warnings}


@router.get("/summary")
def get_trade_history_summary(
    lookback_days: int | None = Query(default=None, ge=1, le=3650),
) -> dict[str, object]:
    try:
        payload, warnings = load_trade_history_summary(lookback_days=lookback_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {**payload, "warnings": warnings}
