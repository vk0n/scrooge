from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from services.ledger_service import load_ledger

router = APIRouter()


@router.get("")
def get_ledger(
    scope: Literal["all", "trades", "treasury"] = Query(default="all"),
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict[str, object]:
    try:
        return load_ledger(scope=scope, limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
