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
    return Path(path or MARKET_EVENT_STREAM_FILE).expanduser()


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

        with self.path.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line_number, raw_line in enumerate(file_obj, start=1):
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    _logger().warning("market_event_line_malformed path=%s line=%s error=%s", self.path, line_number, exc)
                    continue
                if not isinstance(payload, dict):
                    _logger().warning("market_event_line_invalid path=%s line=%s expected=json_object", self.path, line_number)
                    continue
                try:
                    yield _build_market_event(payload)
                except Exception as exc:  # noqa: BLE001
                    _logger().warning("market_event_line_unreadable path=%s line=%s error=%s", self.path, line_number, exc)


def write_market_event_stream(path: str | Path, events: Iterable[MarketEvent]) -> Path:
    target_path = Path(path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file_obj:
        for event in events:
            file_obj.write(json.dumps(market_event_to_dict(event), ensure_ascii=True, sort_keys=True))
            file_obj.write("\n")
    return target_path


def _build_market_event(payload: dict[str, Any]) -> MarketEvent:
    event_type = str(payload.get("event_type", "")).strip()
    if event_type == "price_tick":
        return PriceTickEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            price=float(payload["price"]),
            source=str(payload.get("source", "last_price")),
        )
    if event_type == "mark_price":
        return MarkPriceEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            mark_price=float(payload["mark_price"]),
            funding_rate=(float(payload["funding_rate"]) if payload.get("funding_rate") is not None else None),
            next_funding_time=(str(payload["next_funding_time"]) if payload.get("next_funding_time") else None),
        )
    if event_type == "candle_closed":
        return CandleClosedEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            interval=str(payload["interval"]),
            open_time=str(payload["open_time"]),
            close_time=str(payload["close_time"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
        )
    if event_type == "indicator_snapshot":
        raw_values = payload.get("values") or {}
        values = {
            str(key): (float(value) if value is not None else None)
            for key, value in raw_values.items()
        }
        return IndicatorSnapshotEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            interval=str(payload["interval"]),
            values=values,
        )
    if event_type == "account_balance":
        return AccountBalanceEvent(
            asset=str(payload["asset"]),
            ts=str(payload["ts"]),
            wallet_balance=float(payload["wallet_balance"]),
            cross_wallet_balance=(float(payload["cross_wallet_balance"]) if payload.get("cross_wallet_balance") is not None else None),
            balance_delta=(float(payload["balance_delta"]) if payload.get("balance_delta") is not None else None),
            source=str(payload.get("source", "account_update")),
        )
    if event_type == "position_snapshot":
        return PositionSnapshotEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            position_amt=float(payload["position_amt"]),
            entry_price=(float(payload["entry_price"]) if payload.get("entry_price") is not None else None),
            unrealized_pnl=(float(payload["unrealized_pnl"]) if payload.get("unrealized_pnl") is not None else None),
            position_side=(str(payload["position_side"]) if payload.get("position_side") else None),
            source=str(payload.get("source", "account_update")),
        )
    if event_type == "order_trade_update":
        return OrderTradeUpdateEvent(
            symbol=str(payload["symbol"]),
            ts=str(payload["ts"]),
            order_side=(str(payload["order_side"]) if payload.get("order_side") else None),
            order_type=(str(payload["order_type"]) if payload.get("order_type") else None),
            execution_type=(str(payload["execution_type"]) if payload.get("execution_type") else None),
            order_status=(str(payload["order_status"]) if payload.get("order_status") else None),
            order_id=(int(payload["order_id"]) if payload.get("order_id") is not None else None),
            trade_id=(int(payload["trade_id"]) if payload.get("trade_id") is not None else None),
            last_filled_qty=(float(payload["last_filled_qty"]) if payload.get("last_filled_qty") is not None else None),
            accumulated_filled_qty=(float(payload["accumulated_filled_qty"]) if payload.get("accumulated_filled_qty") is not None else None),
            last_filled_price=(float(payload["last_filled_price"]) if payload.get("last_filled_price") is not None else None),
            average_price=(float(payload["average_price"]) if payload.get("average_price") is not None else None),
            realized_pnl=(float(payload["realized_pnl"]) if payload.get("realized_pnl") is not None else None),
            commission_asset=(str(payload["commission_asset"]) if payload.get("commission_asset") else None),
            commission=(float(payload["commission"]) if payload.get("commission") is not None else None),
            reduce_only=bool(payload.get("reduce_only", False)),
            source=str(payload.get("source", "order_trade_update")),
        )
    raise ValueError(f"Unknown market event type: {event_type or '<empty>'}")


def read_market_event_stream(path: str | Path) -> list[MarketEvent]:
    return list(JsonlMarketEventStore(path).iter_events())
