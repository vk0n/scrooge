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


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:
        return None
    return numeric


def _filter_trades_by_lookback(
    trades: list[dict[str, Any]],
    lookback_days: int | None,
) -> list[dict[str, Any]]:
    if lookback_days is None:
        return list(trades)

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return [trade for trade in trades if (_trade_datetime(trade) is not None and _trade_datetime(trade) >= cutoff)]


@router.get("/trades")
def get_trade_history(
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=30, ge=1, le=200),
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
    ordered_trades = _filter_trades_by_lookback(ordered_trades, lookback_days)

    total_trades = len(ordered_trades)
    total_pages = (total_trades + page_size - 1) // page_size if total_trades > 0 else 0
    effective_page = min(page, total_pages - 1) if total_pages > 0 else 0
    page_start = effective_page * page_size
    page_end = page_start + page_size
    paged_trades = ordered_trades[page_start:page_end]

    return {
        "path": str(TRADE_HISTORY_PATH),
        "lookback_days_applied": lookback_days,
        "page": effective_page,
        "page_size": page_size,
        "total_trades": total_trades,
        "total_pages": total_pages,
        "has_previous_page": effective_page > 0,
        "has_next_page": total_pages > 0 and effective_page < total_pages - 1,
        "returned_trades": len(paged_trades),
        "trades": paged_trades,
        "warnings": warnings,
    }


@router.get("/summary")
def get_trade_history_summary(
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

    filtered_trades = _filter_trades_by_lookback(list(trades), lookback_days)

    winning_trades = 0
    losing_trades = 0
    breakeven_trades = 0
    net_pnl_total = 0.0

    for trade in filtered_trades:
        net_pnl = _as_float(trade.get("net_pnl"))
        if net_pnl is None:
            breakeven_trades += 1
            continue
        net_pnl_total += net_pnl
        if net_pnl > 0:
            winning_trades += 1
        elif net_pnl < 0:
            losing_trades += 1
        else:
            breakeven_trades += 1

    total_trades = len(filtered_trades)
    win_rate_pct = (winning_trades / total_trades) * 100.0 if total_trades > 0 else None

    return {
        "path": str(TRADE_HISTORY_PATH),
        "lookback_days_applied": lookback_days,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "win_rate_pct": win_rate_pct,
        "net_pnl_total": net_pnl_total,
        "warnings": warnings,
    }
