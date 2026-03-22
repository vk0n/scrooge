from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


MARKET_EVENT_SCHEMA_VERSION = 1
MARKET_EVENT_STREAM_FILE = os.getenv("SCROOGE_MARKET_EVENT_STREAM_FILE", "runtime/market_events.jsonl")
MARKET_EVENT_LOGGER_NAME = "scrooge.market_events"


@dataclass(slots=True)
class PriceTickEvent:
    symbol: str
    ts: str
    price: float
    source: str = "last_price"
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="price_tick")


@dataclass(slots=True)
class MarkPriceEvent:
    symbol: str
    ts: str
    mark_price: float
    funding_rate: float | None = None
    next_funding_time: str | None = None
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="mark_price")


@dataclass(slots=True)
class CandleClosedEvent:
    symbol: str
    ts: str
    interval: str
    open_time: str
    close_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="candle_closed")


@dataclass(slots=True)
class IndicatorSnapshotEvent:
    symbol: str
    ts: str
    interval: str
    values: dict[str, float | None]
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="indicator_snapshot")


@dataclass(slots=True)
class AccountBalanceEvent:
    asset: str
    ts: str
    wallet_balance: float
    cross_wallet_balance: float | None = None
    balance_delta: float | None = None
    source: str = "account_update"
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="account_balance")


@dataclass(slots=True)
class PositionSnapshotEvent:
    symbol: str
    ts: str
    position_amt: float
    entry_price: float | None = None
    unrealized_pnl: float | None = None
    position_side: str | None = None
    source: str = "account_update"
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="position_snapshot")


@dataclass(slots=True)
class OrderTradeUpdateEvent:
    symbol: str
    ts: str
    order_side: str | None = None
    order_type: str | None = None
    execution_type: str | None = None
    order_status: str | None = None
    order_id: int | None = None
    trade_id: int | None = None
    last_filled_qty: float | None = None
    accumulated_filled_qty: float | None = None
    last_filled_price: float | None = None
    average_price: float | None = None
    realized_pnl: float | None = None
    commission_asset: str | None = None
    commission: float | None = None
    reduce_only: bool = False
    source: str = "order_trade_update"
    schema_version: int = field(init=False, default=MARKET_EVENT_SCHEMA_VERSION)
    event_type: str = field(init=False, default="order_trade_update")


MarketEvent = (
    PriceTickEvent
    | MarkPriceEvent
    | CandleClosedEvent
    | IndicatorSnapshotEvent
    | AccountBalanceEvent
    | PositionSnapshotEvent
    | OrderTradeUpdateEvent
)


class MarketEventStore(Protocol):
    def append(self, event: MarketEvent) -> None:
        ...

    def iter_events(self) -> Iterator[MarketEvent]:
        ...


def _logger() -> logging.Logger:
    return logging.getLogger(MARKET_EVENT_LOGGER_NAME)


def resolve_market_event_stream_path(path: str | Path | None = None) -> Path:
    return Path(path or os.getenv("SCROOGE_MARKET_EVENT_STREAM_FILE", MARKET_EVENT_STREAM_FILE)).expanduser()


def market_event_to_dict(event: MarketEvent) -> dict[str, Any]:
    return asdict(event)


def market_event_from_dict(payload: dict[str, Any]) -> MarketEvent:
    return _build_market_event(payload)


class JsonlMarketEventStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_market_event_stream_path(path)

    def append(self, event: MarketEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(market_event_to_dict(event), ensure_ascii=True, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(serialized + "\n")

    def iter_events(self) -> Iterator[MarketEvent]:
        if not self.path.exists():
            return

        loads = json.loads
        build_market_event = _build_market_event
        warning = _logger().warning
        with self.path.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line_number, raw_line in enumerate(file_obj, start=1):
                if not raw_line or raw_line == "\n":
                    continue
                try:
                    payload = loads(raw_line)
                except json.JSONDecodeError as exc:
                    warning("market_event_line_malformed path=%s line=%s error=%s", self.path, line_number, exc)
                    continue
                if not isinstance(payload, dict):
                    warning("market_event_line_invalid path=%s line=%s expected=json_object", self.path, line_number)
                    continue
                try:
                    yield build_market_event(payload)
                except Exception as exc:  # noqa: BLE001
                    warning("market_event_line_unreadable path=%s line=%s error=%s", self.path, line_number, exc)


def write_market_event_stream(path: str | Path, events: Iterable[MarketEvent]) -> Path:
    target_path = Path(path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file_obj:
        for event in events:
            file_obj.write(json.dumps(market_event_to_dict(event), ensure_ascii=True, sort_keys=True))
            file_obj.write("\n")
    return target_path


def _build_market_event(payload: dict[str, Any]) -> MarketEvent:
    event_type = payload.get("event_type", "")
    if event_type == "price_tick":
        return PriceTickEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            price=payload["price"],
            source=payload.get("source", "last_price"),
        )
    if event_type == "mark_price":
        return MarkPriceEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            mark_price=payload["mark_price"],
            funding_rate=payload.get("funding_rate"),
            next_funding_time=payload.get("next_funding_time"),
        )
    if event_type == "candle_closed":
        return CandleClosedEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            interval=payload["interval"],
            open_time=payload["open_time"],
            close_time=payload["close_time"],
            open=payload["open"],
            high=payload["high"],
            low=payload["low"],
            close=payload["close"],
            volume=payload["volume"],
        )
    if event_type == "indicator_snapshot":
        raw_values = payload.get("values")
        values = raw_values if isinstance(raw_values, dict) else {}
        return IndicatorSnapshotEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            interval=payload["interval"],
            values=values,
        )
    if event_type == "account_balance":
        return AccountBalanceEvent(
            asset=payload["asset"],
            ts=payload["ts"],
            wallet_balance=payload["wallet_balance"],
            cross_wallet_balance=payload.get("cross_wallet_balance"),
            balance_delta=payload.get("balance_delta"),
            source=payload.get("source", "account_update"),
        )
    if event_type == "position_snapshot":
        return PositionSnapshotEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            position_amt=payload["position_amt"],
            entry_price=payload.get("entry_price"),
            unrealized_pnl=payload.get("unrealized_pnl"),
            position_side=payload.get("position_side"),
            source=payload.get("source", "account_update"),
        )
    if event_type == "order_trade_update":
        return OrderTradeUpdateEvent(
            symbol=payload["symbol"],
            ts=payload["ts"],
            order_side=payload.get("order_side"),
            order_type=payload.get("order_type"),
            execution_type=payload.get("execution_type"),
            order_status=payload.get("order_status"),
            order_id=payload.get("order_id"),
            trade_id=payload.get("trade_id"),
            last_filled_qty=payload.get("last_filled_qty"),
            accumulated_filled_qty=payload.get("accumulated_filled_qty"),
            last_filled_price=payload.get("last_filled_price"),
            average_price=payload.get("average_price"),
            realized_pnl=payload.get("realized_pnl"),
            commission_asset=payload.get("commission_asset"),
            commission=payload.get("commission"),
            reduce_only=payload.get("reduce_only", False),
            source=payload.get("source", "order_trade_update"),
        )
    raise ValueError(f"Unknown market event type: {event_type or '<empty>'}")


def read_market_event_stream(path: str | Path) -> list[MarketEvent]:
    return list(JsonlMarketEventStore(path).iter_events())


def iter_market_event_stream(path: str | Path) -> Iterator[MarketEvent]:
    return JsonlMarketEventStore(path).iter_events()


def count_market_event_stream(path: str | Path) -> int:
    target_path = Path(path).expanduser()
    if not target_path.exists():
        return 0
    count = 0
    with target_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for raw_line in file_obj:
            if raw_line.strip():
                count += 1
    return count
