from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Protocol, TypedDict
from uuid import uuid4

from shared.runtime_db import (
    append_event_record as append_event_db_record,
    event_history_row_count as get_event_history_db_row_count,
    list_event_records as list_event_db_records,
    runtime_artifact_dir,
)


EVENT_SCHEMA_VERSION = 1
EVENT_LOG_FILE = (str(os.getenv("SCROOGE_EVENT_LOG_FILE", "") or "").strip() or "runtime/event_history.jsonl")
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
    if path is not None:
        explicit_path = str(path).strip()
        if explicit_path:
            return Path(explicit_path).expanduser()

    env_path = str(os.getenv("SCROOGE_EVENT_LOG_FILE", "") or "").strip()
    if env_path:
        return Path(env_path).expanduser()

    return runtime_artifact_dir() / "event_history.jsonl"


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

    def _bootstrap_db_from_jsonl(self) -> None:
        if not self.path.exists():
            return
        payloads: list[EventRecord] = []
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
                payloads.append(payload)  # type: ignore[arg-type]

        for payload in payloads:
            append_event_db_record(payload)

    def append(self, event: EventRecord) -> None:
        try:
            if get_event_history_db_row_count() == 0:
                self._bootstrap_db_from_jsonl()
            append_event_db_record(event)
        except OSError:
            _logger().exception("event_store_db_append_failed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event, ensure_ascii=True, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(serialized + "\n")

    def iter_events(self) -> Iterator[EventRecord]:
        try:
            if get_event_history_db_row_count() > 0:
                for payload in list_event_db_records():
                    yield payload  # type: ignore[misc]
                return
        except OSError:
            _logger().exception("event_store_db_read_failed")

        if not self.path.exists():
            return

        try:
            self._bootstrap_db_from_jsonl()
            for payload in list_event_db_records():
                yield payload  # type: ignore[misc]
            return
        except OSError:
            _logger().exception("event_store_db_bootstrap_failed path=%s", self.path)

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


def reset_event_store(path: str | Path | None = None) -> JsonlEventStore:
    global _DEFAULT_EVENT_STORE
    _DEFAULT_EVENT_STORE = JsonlEventStore(path)
    return _DEFAULT_EVENT_STORE


def read_event_records(path: str | Path | None = None) -> list[EventRecord]:
    return list(JsonlEventStore(path).iter_events())
