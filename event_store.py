from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Protocol, TypedDict
from uuid import uuid4


EVENT_SCHEMA_VERSION = 1
EVENT_LOG_FILE = os.getenv("SCROOGE_EVENT_LOG_FILE", "event_history.jsonl")
DEFAULT_RUNTIME_MODE = os.getenv("SCROOGE_RUNTIME_MODE", "").strip() or None
DEFAULT_STRATEGY_MODE = os.getenv("SCROOGE_STRATEGY_MODE", "").strip() or None
EVENT_STORE_LOGGER_NAME = "scrooge.event_store"


class EventRecord(TypedDict):
    schema_version: int
    event_id: str
    ts: str
    level: str
    code: str
    category: str
    ui_message: str
    notify: bool
    runtime_mode: str | None
    strategy_mode: str | None
    context: dict[str, Any]


class EventStore(Protocol):
    def append(self, event: EventRecord) -> None:
        ...

    def iter_events(self) -> Iterator[EventRecord]:
        ...


def _logger() -> logging.Logger:
    return logging.getLogger(EVENT_STORE_LOGGER_NAME)


def resolve_event_log_path(path: str | Path | None = None) -> Path:
    raw_path = Path(path or EVENT_LOG_FILE).expanduser()
    return raw_path


def build_event_record(
    *,
    ts: str,
    level: str,
    code: str,
    category: str,
    ui_message: str,
    notify: bool,
    context: dict[str, Any] | None = None,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
) -> EventRecord:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": uuid4().hex,
        "ts": ts,
        "level": str(level or "info"),
        "code": str(code or "").strip(),
        "category": str(category or "").strip(),
        "ui_message": str(ui_message or "").strip(),
        "notify": bool(notify),
        "runtime_mode": (runtime_mode if runtime_mode is not None else DEFAULT_RUNTIME_MODE),
        "strategy_mode": (strategy_mode if strategy_mode is not None else DEFAULT_STRATEGY_MODE),
        "context": dict(context or {}),
    }


class JsonlEventStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_event_log_path(path)

    def append(self, event: EventRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event, ensure_ascii=True, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(serialized + "\n")

    def iter_events(self) -> Iterator[EventRecord]:
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
                    _logger().warning("event_store_line_malformed path=%s line=%s error=%s", self.path, line_number, exc)
                    continue
                if not isinstance(payload, dict):
                    _logger().warning("event_store_line_invalid path=%s line=%s expected=json_object", self.path, line_number)
                    continue
                yield payload  # type: ignore[misc]


_DEFAULT_EVENT_STORE: JsonlEventStore | None = None


def get_event_store() -> JsonlEventStore:
    global _DEFAULT_EVENT_STORE
    if _DEFAULT_EVENT_STORE is None:
        _DEFAULT_EVENT_STORE = JsonlEventStore()
    return _DEFAULT_EVENT_STORE


def read_event_records(path: str | Path | None = None) -> list[EventRecord]:
    return list(JsonlEventStore(path).iter_events())
