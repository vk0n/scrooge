from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(slots=True)
class BacktestTimeRange:
    start_time: datetime
    end_time: datetime
    cache_key: str
    duration_days: int


def normalize_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def parse_backtest_datetime(raw_value: str | datetime | None) -> datetime | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return normalize_utc_naive(raw_value)
    text = str(raw_value or "").strip()
    if not text:
        return None
    return normalize_utc_naive(datetime.fromisoformat(text))


def _duration_days(start_time: datetime, end_time: datetime) -> int:
    total_seconds = max(0.0, (end_time - start_time).total_seconds())
    whole_days = int(total_seconds // 86400)
    if total_seconds % 86400:
        whole_days += 1
    return max(1, whole_days)


def _resolve_cache_key(
    *,
    backtest_period_days: int,
    backtest_period_start_time: str,
    backtest_period_end_time: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    raw_start = str(backtest_period_start_time or "").strip()
    raw_end = str(backtest_period_end_time or "").strip()
    if raw_start:
        return f"{start_time.date().isoformat()}_to_{end_time.date().isoformat()}"
    if not raw_end:
        return f"{int(backtest_period_days)}d_rolling"
    return f"{int(backtest_period_days)}d_until_{end_time.date().isoformat()}"


def resolve_backtest_time_range(
    *,
    backtest_period_days: int,
    backtest_period_end_time: str = "",
    backtest_period_start_time: str = "",
) -> BacktestTimeRange:
    start_time = parse_backtest_datetime(backtest_period_start_time)
    end_time = parse_backtest_datetime(backtest_period_end_time)

    if start_time is None and end_time is None:
        end_time = datetime.now(UTC).replace(tzinfo=None)
        start_time = end_time - timedelta(days=int(backtest_period_days))
    elif start_time is None:
        end_time = end_time or datetime.now(UTC).replace(tzinfo=None)
        start_time = end_time - timedelta(days=int(backtest_period_days))
    elif end_time is None:
        end_time = start_time + timedelta(days=int(backtest_period_days))

    if start_time is None or end_time is None:
        raise ValueError("Failed to resolve backtest time range")
    if end_time <= start_time:
        raise ValueError("backtest_period_end_time must be later than backtest_period_start_time")

    return BacktestTimeRange(
        start_time=start_time,
        end_time=end_time,
        cache_key=_resolve_cache_key(
            backtest_period_days=backtest_period_days,
            backtest_period_start_time=backtest_period_start_time,
            backtest_period_end_time=backtest_period_end_time,
            start_time=start_time,
            end_time=end_time,
        ),
        duration_days=_duration_days(start_time, end_time),
    )
