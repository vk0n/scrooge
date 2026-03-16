from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


MARKET_EVENT_SCHEMA_VERSION = 1


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


MarketEvent = PriceTickEvent | MarkPriceEvent | CandleClosedEvent | IndicatorSnapshotEvent


def market_event_to_dict(event: MarketEvent) -> dict[str, Any]:
    return asdict(event)


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
    raise ValueError(f"Unknown market event type: {event_type or '<empty>'}")


def read_market_event_stream(path: str | Path) -> list[MarketEvent]:
    target_path = Path(path).expanduser()
    if not target_path.exists():
        return []

    events: list[MarketEvent] = []
    with target_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            events.append(_build_market_event(payload))
    return events
