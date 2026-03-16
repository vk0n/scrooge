from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from core.market_events import (
    AccountBalanceEvent,
    MarketEvent,
    OrderTradeUpdateEvent,
    PositionSnapshotEvent,
    write_market_event_stream,
)


def _normalized_symbol(value: str | None) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


@dataclass(slots=True)
class MarketExecutionSummary:
    symbol: str | None
    total_market_events: int
    execution_events: int
    account_balance_events: int
    position_snapshot_events: int
    order_trade_update_events: int
    trade_execution_events: int
    filled_order_events: int
    realized_pnl_total: float
    commission_total: float
    last_wallet_balance: float | None
    last_cross_wallet_balance: float | None
    last_position_amt: float | None
    last_entry_price: float | None
    last_unrealized_pnl: float | None
    first_ts: str | None
    last_ts: str | None


def filter_execution_market_events(
    events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
) -> list[MarketEvent]:
    target_symbol = _normalized_symbol(symbol)
    filtered: list[MarketEvent] = []
    for event in events:
        if isinstance(event, AccountBalanceEvent):
            filtered.append(event)
            continue
        if isinstance(event, (PositionSnapshotEvent, OrderTradeUpdateEvent)):
            event_symbol = _normalized_symbol(getattr(event, "symbol", None))
            if target_symbol is not None and event_symbol != target_symbol:
                continue
            filtered.append(event)
    return filtered


def summarize_execution_market_events(
    events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
) -> MarketExecutionSummary:
    events_list = list(events)
    filtered = filter_execution_market_events(events_list, symbol=symbol)
    balance_events = [event for event in filtered if isinstance(event, AccountBalanceEvent)]
    position_events = [event for event in filtered if isinstance(event, PositionSnapshotEvent)]
    order_events = [event for event in filtered if isinstance(event, OrderTradeUpdateEvent)]
    trade_execution_events = [
        event
        for event in order_events
        if str(event.execution_type or "").strip().upper() == "TRADE"
    ]
    filled_order_events = [
        event
        for event in order_events
        if str(event.order_status or "").strip().upper() == "FILLED"
    ]

    timestamps = [str(getattr(event, "ts", "")).strip() for event in filtered if str(getattr(event, "ts", "")).strip()]
    timestamps.sort()
    return MarketExecutionSummary(
        symbol=_normalized_symbol(symbol),
        total_market_events=len(events_list),
        execution_events=len(filtered),
        account_balance_events=len(balance_events),
        position_snapshot_events=len(position_events),
        order_trade_update_events=len(order_events),
        trade_execution_events=len(trade_execution_events),
        filled_order_events=len(filled_order_events),
        realized_pnl_total=sum(float(event.realized_pnl or 0.0) for event in order_events),
        commission_total=sum(float(event.commission or 0.0) for event in order_events),
        last_wallet_balance=(balance_events[-1].wallet_balance if balance_events else None),
        last_cross_wallet_balance=(balance_events[-1].cross_wallet_balance if balance_events else None),
        last_position_amt=(position_events[-1].position_amt if position_events else None),
        last_entry_price=(position_events[-1].entry_price if position_events else None),
        last_unrealized_pnl=(position_events[-1].unrealized_pnl if position_events else None),
        first_ts=(timestamps[0] if timestamps else None),
        last_ts=(timestamps[-1] if timestamps else None),
    )


def write_market_event_execution_artifacts(
    events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
    summary_path: str | Path,
    events_path: str | Path,
) -> MarketExecutionSummary:
    events_list = list(events)
    filtered = filter_execution_market_events(events_list, symbol=symbol)
    summary = summarize_execution_market_events(events_list, symbol=symbol)

    resolved_summary_path = Path(summary_path).expanduser()
    resolved_events_path = Path(events_path).expanduser()
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_events_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(summary), file_obj, ensure_ascii=True, indent=2, sort_keys=True)
        file_obj.write("\n")

    write_market_event_stream(resolved_events_path, filtered)
    return summary
