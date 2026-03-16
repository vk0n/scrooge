from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from backtest.discrete_tape import DiscreteMarketTapeRow, TAPE_TIMESTAMP_FORMAT, discrete_market_tape_to_frame
from core.market_events import CandleClosedEvent, IndicatorSnapshotEvent, MarketEvent, PriceTickEvent


def _parse_interval_to_timedelta(interval: str) -> timedelta:
    normalized = str(interval or "").strip().lower()
    if not normalized:
        raise ValueError("interval must not be empty")

    unit = normalized[-1]
    try:
        value = int(normalized[:-1])
    except ValueError as exc:  # noqa: PERF203
        raise ValueError(f"Unsupported interval format: {interval}") from exc

    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"Unsupported interval unit: {interval}")


def _interval_to_pandas_freq(interval: str) -> str:
    normalized = str(interval or "").strip().lower()
    if not normalized:
        raise ValueError("interval must not be empty")

    unit = normalized[-1]
    value = normalized[:-1]
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}d"
    raise ValueError(f"Unsupported interval unit: {interval}")


def _format_ts(value: datetime | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        resolved = value.tz_convert(None) if value.tzinfo is not None else value
        return resolved.strftime(TAPE_TIMESTAMP_FORMAT)
    return value.strftime(TAPE_TIMESTAMP_FORMAT)


def _parse_open_time(value: str) -> datetime:
    return datetime.strptime(value, TAPE_TIMESTAMP_FORMAT)


def _close_time_from_open_time(open_time: str, interval: str) -> str:
    open_dt = _parse_open_time(open_time)
    close_dt = open_dt + _parse_interval_to_timedelta(interval) - timedelta(seconds=1)
    return close_dt.strftime(TAPE_TIMESTAMP_FORMAT)


def _expected_small_rows_per_interval(*, small_interval: str, target_interval: str) -> int:
    small_delta = _parse_interval_to_timedelta(small_interval)
    target_delta = _parse_interval_to_timedelta(target_interval)
    small_seconds = int(small_delta.total_seconds())
    target_seconds = int(target_delta.total_seconds())
    if small_seconds <= 0 or target_seconds <= 0 or target_seconds % small_seconds != 0:
        raise ValueError(f"Interval ratio must be integral: small={small_interval} target={target_interval}")
    return target_seconds // small_seconds


def _build_intrabar_tick_events(
    row: DiscreteMarketTapeRow,
    *,
    interval: str,
    source: str,
) -> list[PriceTickEvent]:
    open_dt = _parse_open_time(row.open_time)
    interval_delta = _parse_interval_to_timedelta(interval)
    close_dt = open_dt + interval_delta - timedelta(seconds=1)
    total_seconds = max(int(interval_delta.total_seconds()) - 1, 1)

    if row.close >= row.open:
        prices = [row.open, row.low, row.high, row.close]
    else:
        prices = [row.open, row.high, row.low, row.close]

    tick_offsets = [0, max(total_seconds // 4, 1), max((total_seconds * 3) // 4, 1), total_seconds]
    events: list[PriceTickEvent] = []
    last_signature: tuple[str, float] | None = None
    for offset_seconds, price in zip(tick_offsets, prices, strict=True):
        ts_value = open_dt + timedelta(seconds=int(offset_seconds))
        signature = (_format_ts(ts_value), float(price))
        if signature == last_signature:
            continue
        events.append(
            PriceTickEvent(
                symbol=row.symbol,
                ts=signature[0],
                price=signature[1],
                source=source,
            )
        )
        last_signature = signature
    return events


def _build_small_candle_events(
    tape: Iterable[DiscreteMarketTapeRow],
    *,
    small_interval: str,
    emit_discrete_snapshot: bool,
    source: str,
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for row in tape:
        events.extend(
            _build_intrabar_tick_events(
                row,
                interval=small_interval,
                source=source,
            )
        )
        close_time = _close_time_from_open_time(row.open_time, small_interval)
        events.append(
            CandleClosedEvent(
                symbol=row.symbol,
                ts=close_time,
                interval=small_interval,
                open_time=row.open_time,
                close_time=close_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
        )
        if emit_discrete_snapshot:
            events.append(
                IndicatorSnapshotEvent(
                    symbol=row.symbol,
                    ts=close_time,
                    interval="discrete_snapshot",
                    values={
                        "EMA": row.EMA,
                        "RSI": row.RSI,
                        "BBL": row.BBL,
                        "BBM": row.BBM,
                        "BBU": row.BBU,
                        "ATR": row.ATR,
                    },
                )
            )
    return events


def _build_resampled_candle_events(
    tape_frame: pd.DataFrame,
    *,
    symbol: str,
    small_interval: str,
    target_interval: str,
) -> list[CandleClosedEvent]:
    if tape_frame.empty:
        return []

    expected_rows = _expected_small_rows_per_interval(
        small_interval=small_interval,
        target_interval=target_interval,
    )
    resampled = (
        tape_frame.sort_values("open_time")
        .set_index("open_time")[["open", "high", "low", "close", "volume"]]
        .resample(_interval_to_pandas_freq(target_interval), label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            row_count=("close", "count"),
        )
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled[resampled["row_count"] >= expected_rows]

    events: list[CandleClosedEvent] = []
    for open_time, row in resampled.iterrows():
        open_time_text = _format_ts(open_time)
        close_time = _close_time_from_open_time(open_time_text, target_interval)
        events.append(
            CandleClosedEvent(
                symbol=symbol,
                ts=close_time,
                interval=target_interval,
                open_time=open_time_text,
                close_time=close_time,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return events


def _event_priority(event: MarketEvent, *, small_interval: str, medium_interval: str, big_interval: str) -> int:
    if isinstance(event, PriceTickEvent):
        return 10
    if isinstance(event, CandleClosedEvent):
        if event.interval == small_interval:
            return 20
        if event.interval == medium_interval:
            return 30
        if event.interval == big_interval:
            return 40
        return 45
    if isinstance(event, IndicatorSnapshotEvent):
        return 50
    return 90


def build_historical_market_event_stream(
    tape: Iterable[DiscreteMarketTapeRow],
    *,
    intervals: dict[str, str],
    emit_discrete_snapshot: bool = True,
    price_tick_source: str = "historical_intrabar",
) -> list[MarketEvent]:
    rows = list(tape)
    if not rows:
        return []

    small_interval = str(intervals["small"])
    medium_interval = str(intervals["medium"])
    big_interval = str(intervals["big"])

    events: list[MarketEvent] = _build_small_candle_events(
        rows,
        small_interval=small_interval,
        emit_discrete_snapshot=emit_discrete_snapshot,
        source=price_tick_source,
    )

    tape_frame = discrete_market_tape_to_frame(rows)
    symbol = rows[0].symbol
    events.extend(
        _build_resampled_candle_events(
            tape_frame,
            symbol=symbol,
            small_interval=small_interval,
            target_interval=medium_interval,
        )
    )
    events.extend(
        _build_resampled_candle_events(
            tape_frame,
            symbol=symbol,
            small_interval=small_interval,
            target_interval=big_interval,
        )
    )

    events.sort(
        key=lambda event: (
            getattr(event, "ts", ""),
            _event_priority(
                event,
                small_interval=small_interval,
                medium_interval=medium_interval,
                big_interval=big_interval,
            ),
        )
    )
    return events
