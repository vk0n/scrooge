from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_logs(limit: int = 100) -> dict[str, object]:
    bounded_limit = max(1, min(limit, 1000))
    return {
        "mode": "mock",
        "limit": bounded_limit,
        "lines": [
            "[INFO] control plane booted",
            "[INFO] no live trading engine integration in stage 1",
        ],
    }
