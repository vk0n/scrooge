from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
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


def _normalized_side(value: str | None) -> str | None:
    text = str(value or "").strip().upper()
    if text == "BUY":
        return "long"
    if text == "SELL":
        return "short"
    return None


def _sort_ts_key(value: str | None) -> tuple[int, str]:
    text = str(value or "").strip()
    if not text:
        return (1, "")
    try:
        return (0, datetime.strptime(text, "%Y-%m-%d %H:%M:%S").isoformat())
    except ValueError:
        return (1, text)


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
    observed_fill_events: int
    observed_total_trades: int
    observed_closed_trades: int
    observed_open_trades: int
    first_ts: str | None
    last_ts: str | None


@dataclass(slots=True)
class ObservedExecutionFill:
    symbol: str
    ts: str
    side: str
    signed_qty: float
    fill_qty: float
    fill_price: float
    accumulated_filled_qty: float | None
    realized_pnl: float
    commission: float
    commission_asset: str | None
    order_status: str | None
    order_id: int | None
    trade_id: int | None
    reduce_only: bool
    source: str


@dataclass(slots=True)
class ObservedExecutionTrade:
    symbol: str
    side: str
    entry_ts: str
    exit_ts: str | None
    entry_price: float
    exit_price: float | None
    open_qty: float
    close_qty: float
    remaining_qty: float
    max_position_qty: float
    realized_pnl: float
    commission_total: float
    fill_count: int
    status: str


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
    fills = build_observed_execution_fills(filtered, symbol=symbol)
    observed_trades = reconstruct_observed_execution_trades(fills)

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
        observed_fill_events=len(fills),
        observed_total_trades=len(observed_trades),
        observed_closed_trades=sum(1 for trade in observed_trades if trade.status == "closed"),
        observed_open_trades=sum(1 for trade in observed_trades if trade.status == "open"),
        first_ts=(timestamps[0] if timestamps else None),
        last_ts=(timestamps[-1] if timestamps else None),
    )


def build_observed_execution_fills(
    events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
) -> list[ObservedExecutionFill]:
    target_symbol = _normalized_symbol(symbol)
    fills: list[ObservedExecutionFill] = []
    for event in events:
        if not isinstance(event, OrderTradeUpdateEvent):
            continue
        event_symbol = _normalized_symbol(event.symbol)
        if target_symbol is not None and event_symbol != target_symbol:
            continue
        if str(event.execution_type or "").strip().upper() != "TRADE":
            continue
        fill_qty = float(event.last_filled_qty or 0.0)
        if fill_qty <= 0:
            continue
        fill_price = event.last_filled_price if event.last_filled_price is not None else event.average_price
        if fill_price is None:
            continue
        side = _normalized_side(event.order_side)
        if side is None:
            continue
        signed_qty = fill_qty if side == "long" else -fill_qty
        fills.append(
            ObservedExecutionFill(
                symbol=event_symbol or "",
                ts=str(event.ts),
                side=side,
                signed_qty=signed_qty,
                fill_qty=fill_qty,
                fill_price=float(fill_price),
                accumulated_filled_qty=(float(event.accumulated_filled_qty) if event.accumulated_filled_qty is not None else None),
                realized_pnl=float(event.realized_pnl or 0.0),
                commission=float(event.commission or 0.0),
                commission_asset=(str(event.commission_asset).strip().upper() if event.commission_asset else None),
                order_status=(str(event.order_status).strip().upper() if event.order_status else None),
                order_id=event.order_id,
                trade_id=event.trade_id,
                reduce_only=bool(event.reduce_only),
                source=event.source,
            )
        )
    fills.sort(key=lambda item: (_sort_ts_key(item.ts), item.order_id or 0, item.trade_id or 0))
    return fills


def reconstruct_observed_execution_trades(
    fills: Iterable[ObservedExecutionFill],
) -> list[ObservedExecutionTrade]:
    output: list[ObservedExecutionTrade] = []
    current_trade: ObservedExecutionTrade | None = None
    current_qty = 0.0
    current_entry_price: float | None = None

    def close_current(*, exit_ts: str, exit_price: float) -> None:
        nonlocal current_trade, current_qty, current_entry_price
        if current_trade is None:
            return
        current_trade.exit_ts = exit_ts
        current_trade.exit_price = exit_price
        current_trade.remaining_qty = 0.0
        current_trade.status = "closed"
        output.append(current_trade)
        current_trade = None
        current_qty = 0.0
        current_entry_price = None

    def open_new(fill: ObservedExecutionFill, *, qty: float, commission: float) -> None:
        nonlocal current_trade, current_qty, current_entry_price
        current_entry_price = fill.fill_price
        current_qty = qty if fill.side == "long" else -qty
        current_trade = ObservedExecutionTrade(
            symbol=fill.symbol,
            side=fill.side,
            entry_ts=fill.ts,
            exit_ts=None,
            entry_price=fill.fill_price,
            exit_price=None,
            open_qty=qty,
            close_qty=0.0,
            remaining_qty=qty,
            max_position_qty=qty,
            realized_pnl=0.0,
            commission_total=commission,
            fill_count=1,
            status="open",
        )

    for fill in fills:
        fill_sign = 1.0 if fill.signed_qty > 0 else -1.0
        fill_qty = abs(fill.signed_qty)

        if current_trade is None or abs(current_qty) <= 1e-12:
            open_new(fill, qty=fill_qty, commission=fill.commission)
            current_trade.realized_pnl += fill.realized_pnl
            continue

        current_sign = 1.0 if current_qty > 0 else -1.0
        if fill_sign == current_sign:
            existing_qty = abs(current_qty)
            new_qty = existing_qty + fill_qty
            weighted_entry = (((current_entry_price or fill.fill_price) * existing_qty) + (fill.fill_price * fill_qty)) / new_qty
            current_entry_price = weighted_entry
            current_qty += fill.signed_qty
            current_trade.entry_price = weighted_entry
            current_trade.open_qty += fill_qty
            current_trade.remaining_qty = abs(current_qty)
            current_trade.max_position_qty = max(current_trade.max_position_qty, abs(current_qty))
            current_trade.realized_pnl += fill.realized_pnl
            current_trade.commission_total += fill.commission
            current_trade.fill_count += 1
            continue

        closing_qty = min(abs(current_qty), fill_qty)
        remaining_qty = fill_qty - closing_qty
        close_fraction = (closing_qty / fill_qty) if fill_qty > 0 else 1.0
        close_commission = fill.commission * close_fraction
        open_commission = fill.commission - close_commission

        current_trade.close_qty += closing_qty
        current_trade.realized_pnl += fill.realized_pnl
        current_trade.commission_total += close_commission
        current_trade.fill_count += 1

        updated_qty = current_qty + fill.signed_qty
        if abs(updated_qty) <= 1e-12:
            close_current(exit_ts=fill.ts, exit_price=fill.fill_price)
            continue

        updated_sign = 1.0 if updated_qty > 0 else -1.0
        if updated_sign != current_sign:
            close_current(exit_ts=fill.ts, exit_price=fill.fill_price)
            if remaining_qty > 1e-12:
                flipped_fill = ObservedExecutionFill(
                    symbol=fill.symbol,
                    ts=fill.ts,
                    side=fill.side,
                    signed_qty=(remaining_qty if fill.side == "long" else -remaining_qty),
                    fill_qty=remaining_qty,
                    fill_price=fill.fill_price,
                    accumulated_filled_qty=fill.accumulated_filled_qty,
                    realized_pnl=0.0,
                    commission=open_commission,
                    commission_asset=fill.commission_asset,
                    order_status=fill.order_status,
                    order_id=fill.order_id,
                    trade_id=fill.trade_id,
                    reduce_only=fill.reduce_only,
                    source=fill.source,
                )
                open_new(flipped_fill, qty=remaining_qty, commission=open_commission)
            continue

        current_qty = updated_qty
        current_trade.remaining_qty = abs(current_qty)
        current_trade.exit_ts = fill.ts
        current_trade.exit_price = fill.fill_price

    if current_trade is not None:
        output.append(current_trade)
    return output


def write_market_event_execution_artifacts(
    events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
    summary_path: str | Path,
    events_path: str | Path,
    fills_path: str | Path | None = None,
    trades_path: str | Path | None = None,
) -> MarketExecutionSummary:
    target_symbol = _normalized_symbol(symbol)
    filtered: list[MarketEvent] = []
    fills: list[ObservedExecutionFill] = []
    total_market_events = 0
    account_balance_events = 0
    position_snapshot_events = 0
    order_trade_update_events = 0
    trade_execution_events = 0
    filled_order_events = 0
    realized_pnl_total = 0.0
    commission_total = 0.0
    last_wallet_balance: float | None = None
    last_cross_wallet_balance: float | None = None
    last_position_amt: float | None = None
    last_entry_price: float | None = None
    last_unrealized_pnl: float | None = None
    first_ts: str | None = None
    last_ts: str | None = None

    for event in events:
        total_market_events += 1

        if isinstance(event, AccountBalanceEvent):
            filtered.append(event)
            account_balance_events += 1
            last_wallet_balance = float(event.wallet_balance)
            last_cross_wallet_balance = (
                float(event.cross_wallet_balance) if event.cross_wallet_balance is not None else None
            )
        elif isinstance(event, PositionSnapshotEvent):
            event_symbol = _normalized_symbol(getattr(event, "symbol", None))
            if target_symbol is not None and event_symbol != target_symbol:
                continue
            filtered.append(event)
            position_snapshot_events += 1
            last_position_amt = float(event.position_amt)
            last_entry_price = float(event.entry_price) if event.entry_price is not None else None
            last_unrealized_pnl = float(event.unrealized_pnl) if event.unrealized_pnl is not None else None
        elif isinstance(event, OrderTradeUpdateEvent):
            event_symbol = _normalized_symbol(getattr(event, "symbol", None))
            if target_symbol is not None and event_symbol != target_symbol:
                continue
            filtered.append(event)
            order_trade_update_events += 1

            execution_type = str(event.execution_type or "").strip().upper()
            order_status = str(event.order_status or "").strip().upper()
            if execution_type == "TRADE":
                trade_execution_events += 1
            if order_status == "FILLED":
                filled_order_events += 1
            realized_pnl_total += float(event.realized_pnl or 0.0)
            commission_total += float(event.commission or 0.0)

            if execution_type == "TRADE":
                fill_qty = float(event.last_filled_qty or 0.0)
                fill_price = event.last_filled_price if event.last_filled_price is not None else event.average_price
                side = _normalized_side(event.order_side)
                if fill_qty > 0 and fill_price is not None and side is not None:
                    fills.append(
                        ObservedExecutionFill(
                            symbol=event_symbol or "",
                            ts=str(event.ts),
                            side=side,
                            signed_qty=(fill_qty if side == "long" else -fill_qty),
                            fill_qty=fill_qty,
                            fill_price=float(fill_price),
                            accumulated_filled_qty=(
                                float(event.accumulated_filled_qty) if event.accumulated_filled_qty is not None else None
                            ),
                            realized_pnl=float(event.realized_pnl or 0.0),
                            commission=float(event.commission or 0.0),
                            commission_asset=(str(event.commission_asset).strip().upper() if event.commission_asset else None),
                            order_status=(str(event.order_status).strip().upper() if event.order_status else None),
                            order_id=event.order_id,
                            trade_id=event.trade_id,
                            reduce_only=bool(event.reduce_only),
                            source=event.source,
                        )
                    )
        else:
            continue

        ts_text = str(getattr(event, "ts", "") or "").strip()
        if ts_text:
            if first_ts is None or _sort_ts_key(ts_text) < _sort_ts_key(first_ts):
                first_ts = ts_text
            if last_ts is None or _sort_ts_key(ts_text) > _sort_ts_key(last_ts):
                last_ts = ts_text

    fills.sort(key=lambda item: (_sort_ts_key(item.ts), item.order_id or 0, item.trade_id or 0))
    trades = reconstruct_observed_execution_trades(fills)
    summary = MarketExecutionSummary(
        symbol=target_symbol,
        total_market_events=total_market_events,
        execution_events=len(filtered),
        account_balance_events=account_balance_events,
        position_snapshot_events=position_snapshot_events,
        order_trade_update_events=order_trade_update_events,
        trade_execution_events=trade_execution_events,
        filled_order_events=filled_order_events,
        realized_pnl_total=realized_pnl_total,
        commission_total=commission_total,
        last_wallet_balance=last_wallet_balance,
        last_cross_wallet_balance=last_cross_wallet_balance,
        last_position_amt=last_position_amt,
        last_entry_price=last_entry_price,
        last_unrealized_pnl=last_unrealized_pnl,
        observed_fill_events=len(fills),
        observed_total_trades=len(trades),
        observed_closed_trades=sum(1 for trade in trades if trade.status == "closed"),
        observed_open_trades=sum(1 for trade in trades if trade.status == "open"),
        first_ts=first_ts,
        last_ts=last_ts,
    )

    resolved_summary_path = Path(summary_path).expanduser()
    resolved_events_path = Path(events_path).expanduser()
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_events_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(summary), file_obj, ensure_ascii=True, indent=2, sort_keys=True)
        file_obj.write("\n")

    write_market_event_stream(resolved_events_path, filtered)
    if fills_path is not None:
        resolved_fills_path = Path(fills_path).expanduser()
        resolved_fills_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_fills_path.open("w", encoding="utf-8") as file_obj:
            for fill in fills:
                file_obj.write(json.dumps(asdict(fill), ensure_ascii=True, sort_keys=True))
                file_obj.write("\n")
    if trades_path is not None:
        resolved_trades_path = Path(trades_path).expanduser()
        resolved_trades_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_trades_path.open("w", encoding="utf-8") as file_obj:
            for trade in trades:
                file_obj.write(json.dumps(asdict(trade), ensure_ascii=True, sort_keys=True))
                file_obj.write("\n")
    return summary
