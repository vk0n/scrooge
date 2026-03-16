from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from backtest.discrete_tape import DiscreteMarketTapeRow
from core.market_events import (
    CandleClosedEvent,
    IndicatorSnapshotEvent,
    MarketEvent,
    read_market_event_stream,
    write_market_event_stream,
)


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


def _close_time_from_open_time(open_time: str, interval: str) -> str:
    open_dt = datetime.strptime(open_time, "%Y-%m-%d %H:%M:%S")
    close_dt = open_dt + _parse_interval_to_timedelta(interval) - timedelta(seconds=1)
    return close_dt.strftime("%Y-%m-%d %H:%M:%S")


def build_discrete_market_event_stream(
    tape: Iterable[DiscreteMarketTapeRow],
    *,
    candle_interval: str,
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for row in tape:
        close_time = _close_time_from_open_time(row.open_time, candle_interval)
        events.append(
            CandleClosedEvent(
                symbol=row.symbol,
                ts=close_time,
                interval=candle_interval,
                open_time=row.open_time,
                close_time=close_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
        )
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


def write_discrete_market_event_stream(path: str | Path, events: Iterable[MarketEvent]) -> Path:
    return write_market_event_stream(path, events)


def read_discrete_market_event_stream(path: str | Path) -> list[MarketEvent]:
    return read_market_event_stream(path)


def rebuild_discrete_tape_from_market_events(
    events: Iterable[MarketEvent],
    *,
    require_indicator_snapshot: bool = True,
) -> list[DiscreteMarketTapeRow]:
    indicator_by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    candle_by_key: dict[tuple[str, str], CandleClosedEvent] = {}
    rebuilt: list[DiscreteMarketTapeRow] = []

    def try_emit(symbol: str, ts: str) -> None:
        key = (symbol, ts)
        candle = candle_by_key.get(key)
        if candle is None:
            return
        indicator_values = indicator_by_key.get(key)
        if indicator_values is None and require_indicator_snapshot:
            return

        candle_by_key.pop(key, None)
        indicator_by_key.pop(key, None)
        indicator_values = indicator_values or {}
        rebuilt.append(
            DiscreteMarketTapeRow(
                symbol=candle.symbol,
                open_time=candle.open_time,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                EMA=indicator_values.get("EMA"),
                RSI=indicator_values.get("RSI"),
                BBL=indicator_values.get("BBL"),
                BBM=indicator_values.get("BBM"),
                BBU=indicator_values.get("BBU"),
                ATR=indicator_values.get("ATR"),
            )
        )

    for event in events:
        if isinstance(event, CandleClosedEvent):
            key = (event.symbol, event.ts)
            candle_by_key[key] = event
            try_emit(event.symbol, event.ts)
            continue

        if isinstance(event, IndicatorSnapshotEvent) and event.interval == "discrete_snapshot":
            key = (event.symbol, event.ts)
            indicator_by_key[key] = {
                "EMA": event.values.get("EMA"),
                "RSI": event.values.get("RSI"),
                "BBL": event.values.get("BBL"),
                "BBM": event.values.get("BBM"),
                "BBU": event.values.get("BBU"),
                "ATR": event.values.get("ATR"),
            }
            try_emit(event.symbol, event.ts)

    if require_indicator_snapshot and candle_by_key:
        missing = next(iter(candle_by_key.values()))
        raise ValueError(
            "Discrete market event stream is missing indicator snapshot "
            f"for symbol={missing.symbol} ts={missing.ts}"
        )

    rebuilt.sort(key=lambda row: row.open_time)
    return rebuilt
