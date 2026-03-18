from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backtest.discrete_tape import DiscreteMarketTapeRow
from backtest.progress import BacktestProgressReporter
from core.market_events import (
    CandleClosedEvent,
    IndicatorSnapshotEvent,
    MarketEvent,
    iter_market_event_stream,
)


@dataclass(slots=True)
class DiscreteTapeProjectionSummary:
    total_events: int
    matched_candles: int
    matched_indicator_snapshots: int
    projected_rows: int
    ignored_events: int


def read_projected_discrete_tape_from_market_event_stream(
    path: str | Path,
    *,
    candle_interval: str,
    symbol: str | None = None,
    require_indicator_snapshot: bool = True,
    progress_reporter: BacktestProgressReporter | None = None,
) -> tuple[list[DiscreteMarketTapeRow], DiscreteTapeProjectionSummary]:
    events = iter_market_event_stream(path)
    tape, summary = project_discrete_tape_from_market_events(
        events,
        candle_interval=candle_interval,
        symbol=symbol,
        require_indicator_snapshot=require_indicator_snapshot,
        progress_reporter=progress_reporter,
    )
    return tape, summary


def project_discrete_tape_from_market_events(
    events: Iterable[MarketEvent],
    *,
    candle_interval: str,
    symbol: str | None = None,
    require_indicator_snapshot: bool = True,
    progress_reporter: BacktestProgressReporter | None = None,
) -> tuple[list[DiscreteMarketTapeRow], DiscreteTapeProjectionSummary]:
    filtered_symbol = str(symbol or "").strip().upper() or None
    normalized_interval = str(candle_interval or "").strip()
    if not normalized_interval:
        raise ValueError("candle_interval must not be empty")

    indicator_by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    candle_by_key: dict[tuple[str, str], CandleClosedEvent] = {}
    rebuilt: list[DiscreteMarketTapeRow] = []
    total_events = 0
    matched_candles = 0
    matched_indicator_snapshots = 0
    ignored_events = 0

    def try_emit(symbol_key: str, ts: str) -> None:
        key = (symbol_key, ts)
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
        total_events += 1
        if progress_reporter is not None:
            progress_reporter.advance()

        event_symbol = getattr(event, "symbol", None)
        if filtered_symbol and str(event_symbol or "").upper() != filtered_symbol:
            ignored_events += 1
            continue

        if isinstance(event, CandleClosedEvent):
            if event.interval != normalized_interval:
                ignored_events += 1
                continue
            key = (event.symbol, event.ts)
            candle_by_key[key] = event
            matched_candles += 1
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
            matched_indicator_snapshots += 1
            try_emit(event.symbol, event.ts)
            continue

        ignored_events += 1

    if require_indicator_snapshot and candle_by_key:
        missing = next(iter(candle_by_key.values()))
        raise ValueError(
            "Market event stream is missing indicator snapshot "
            f"for symbol={missing.symbol} ts={missing.ts} interval={missing.interval}"
        )

    rebuilt.sort(key=lambda row: row.open_time)
    return rebuilt, DiscreteTapeProjectionSummary(
        total_events=total_events,
        matched_candles=matched_candles,
        matched_indicator_snapshots=matched_indicator_snapshots,
        projected_rows=len(rebuilt),
        ignored_events=ignored_events,
    )
