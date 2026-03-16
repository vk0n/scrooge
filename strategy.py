"""
Backward-compatible shim.

The shared discrete engine and runtime/state logic now live in `core_engine.py`.
This module keeps historical imports working until we do the larger repo reorg
and the final rename away from `strategy.py`.
"""

from core_engine import (
    LOG_FILE,
    SEARCH_STATUS_LABELS,
    TIMESTAMP_FORMAT,
    format_event_timestamp,
    refresh_runtime_state_from_price_tick,
    run_strategy,
    save_log,
)

__all__ = [
    "LOG_FILE",
    "SEARCH_STATUS_LABELS",
    "TIMESTAMP_FORMAT",
    "format_event_timestamp",
    "refresh_runtime_state_from_price_tick",
    "run_strategy",
    "save_log",
]
