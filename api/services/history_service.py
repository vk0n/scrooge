from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from shared.runtime_db import (  # noqa: E402
    count_trade_history_rows as get_trade_history_db_total_count,
    list_balance_history_values as list_balance_history_db_values,
    list_trade_history_rows as list_trade_history_db_rows,
    runtime_db_path,
    summarize_trade_history as summarize_trade_history_db,
)

RUNTIME_DB_PATH = runtime_db_path()


def load_trade_history(limit: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        trades = list_trade_history_db_rows()
    except OSError as exc:
        warnings.append(f"Failed to read trade history DB: {RUNTIME_DB_PATH} ({exc})")
        return [], warnings

    if limit is not None and limit > 0:
        trades = trades[-limit:]
    return trades, warnings


def load_balance_history(limit: int | None = None) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    try:
        balances = list_balance_history_db_values()
    except OSError as exc:
        warnings.append(f"Failed to read balance history DB: {RUNTIME_DB_PATH} ({exc})")
        return [], warnings

    if limit is not None and limit > 0:
        balances = balances[-limit:]
    return balances, warnings


def trade_history_source_path() -> Path:
    return RUNTIME_DB_PATH


def load_trade_history_page(
    *,
    page: int,
    page_size: int,
    lookback_days: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        total_trades = get_trade_history_db_total_count(lookback_days=lookback_days)
        total_pages = (total_trades + page_size - 1) // page_size if total_trades > 0 else 0
        effective_page = min(page, total_pages - 1) if total_pages > 0 else 0
        page_start = effective_page * page_size
        trades = list_trade_history_db_rows(
            limit=page_size,
            offset=page_start,
            lookback_days=lookback_days,
            newest_first=True,
        )
        return {
            "path": str(RUNTIME_DB_PATH),
            "lookback_days_applied": lookback_days,
            "page": effective_page,
            "page_size": page_size,
            "total_trades": total_trades,
            "total_pages": total_pages,
            "has_previous_page": effective_page > 0,
            "has_next_page": total_pages > 0 and effective_page < total_pages - 1,
            "returned_trades": len(trades),
            "trades": trades,
        }, warnings
    except OSError as exc:
        warnings.append(f"Failed to read paged trade history DB: {RUNTIME_DB_PATH} ({exc})")

    return {
        "path": str(RUNTIME_DB_PATH),
        "lookback_days_applied": lookback_days,
        "page": 0,
        "page_size": page_size,
        "total_trades": 0,
        "total_pages": 0,
        "has_previous_page": False,
        "has_next_page": False,
        "returned_trades": 0,
        "trades": [],
    }, warnings


def load_trade_history_summary(
    *,
    lookback_days: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        summary = summarize_trade_history_db(lookback_days=lookback_days)
        return {
            "path": str(RUNTIME_DB_PATH),
            "lookback_days_applied": lookback_days,
            **summary,
        }, warnings
    except OSError as exc:
        warnings.append(f"Failed to summarize trade history DB: {RUNTIME_DB_PATH} ({exc})")

    return {
        "path": str(RUNTIME_DB_PATH),
        "lookback_days_applied": lookback_days,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate_pct": None,
        "net_pnl_total": 0.0,
    }, warnings
