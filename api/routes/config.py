from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ConfigPatch(BaseModel):
    updates: dict[str, object]


@router.get("")
def get_config() -> dict[str, object]:
    return {
        "mode": "mock",
        "config": {
            "symbol": "BTCUSDT",
            "leverage": 3,
            "live": False,
        },
    }


@router.patch("")
def patch_config(payload: ConfigPatch) -> dict[str, object]:
    if not payload.updates:
        raise HTTPException(status_code=400, detail="No config updates provided")

    return {
        "mode": "mock",
        "applied": payload.updates,
        "note": "Stage 1 mock only: no file was mutated",
    }
