from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.history_service import TRADE_HISTORY_PATH, load_trade_history

router = APIRouter()


def _trade_sort_key(trade: dict[str, Any]) -> str:
    for field_name in ("exit_time", "entry_time", "time"):
        value = trade.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_trade_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _trade_datetime(trade: dict[str, Any]) -> datetime | None:
    for field_name in ("exit_time", "entry_time", "time"):
        parsed = _parse_trade_timestamp(trade.get(field_name))
        if parsed is not None:
            return parsed
    return None


@router.get("/trades")
def get_trade_history(
    limit: int = Query(default=0, ge=0, le=5000),
    lookback_days: int | None = Query(default=None, ge=1, le=3650),
) -> dict[str, object]:
    try:
        trades, warnings = load_trade_history(limit=None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ordered_trades = sorted(
        trades,
        key=lambda trade: _trade_datetime(trade) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    if lookback_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ordered_trades = [
            trade for trade in ordered_trades if (_trade_datetime(trade) is not None and _trade_datetime(trade) >= cutoff)
        ]

    total_trades = len(ordered_trades)
    limited_trades = ordered_trades if limit <= 0 else ordered_trades[:limit]

    return {
        "path": str(TRADE_HISTORY_PATH),
        "requested_limit": limit if limit > 0 else None,
        "lookback_days_applied": lookback_days,
        "total_trades": total_trades,
        "returned_trades": len(limited_trades),
        "trades": limited_trades,
        "warnings": warnings,
    }
