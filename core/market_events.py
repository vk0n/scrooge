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


MarketEvent = PriceTickEvent | MarkPriceEvent | CandleClosedEvent | IndicatorSnapshotEvent


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
    raise ValueError(f"Unknown market event type: {event_type or '<empty>'}")


def read_market_event_stream(path: str | Path) -> list[MarketEvent]:
    return list(JsonlMarketEventStore(path).iter_events())
