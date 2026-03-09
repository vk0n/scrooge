from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_status() -> dict[str, object]:
    return {
        "bot": "scrooge",
        "mode": "mock",
        "runtime": "stopped",
        "last_update": datetime.now(UTC).isoformat(),
        "open_positions": 0,
        "balance": 0.0,
    }
