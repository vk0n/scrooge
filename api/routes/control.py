from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ControlCommand(BaseModel):
    action: str


_ALLOWED_ACTIONS: set[str] = {"start", "stop", "restart"}


@router.post("")
def control(payload: ControlCommand) -> dict[str, str]:
    action = payload.action.strip().lower()
    if action not in _ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {payload.action}")

    logger.info("Received control action (mock): %s", action)
    return {
        "mode": "mock",
        "result": f"Accepted '{action}' in stage 1 mock mode",
    }
