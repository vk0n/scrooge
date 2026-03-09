from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
    }
