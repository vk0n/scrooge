from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.chart_service import build_chart_payload

router = APIRouter()


@router.get("")
def get_chart(
    symbol: str | None = Query(default=None, min_length=3, max_length=20),
    period: str = Query(default="1d"),
    interval: str = Query(default="1m"),
    indicators: bool = Query(default=True),
    end: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> dict[str, object]:
    try:
        payload = build_chart_payload(
            symbol=symbol,
            period=period,
            interval=interval,
            include_indicators=indicators,
            end=end,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return payload
