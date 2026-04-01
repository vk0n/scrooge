from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_ms() -> int:
    return int(utc_now().timestamp() * 1000)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def utc_now_text(fmt: str = DEFAULT_TIMESTAMP_FORMAT) -> str:
    return utc_now().strftime(fmt)


def utc_datetime_from_timestamp(value: Any) -> datetime:
    numeric = float(value)
    if numeric > 10_000_000_000:
        numeric /= 1000.0
    return datetime.fromtimestamp(numeric, tz=UTC)


def utc_text_from_timestamp(value: Any, fmt: str = DEFAULT_TIMESTAMP_FORMAT, fallback: str | None = None) -> str:
    try:
        return utc_datetime_from_timestamp(value).strftime(fmt)
    except (TypeError, ValueError, OSError, OverflowError):
        return fallback if fallback is not None else utc_now_text(fmt)
