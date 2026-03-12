from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_FILE = os.getenv("SCROOGE_STATE_FILE", "state.json")
TRADE_HISTORY_FILE = os.getenv("SCROOGE_TRADE_HISTORY_FILE", "trade_history.jsonl")
BALANCE_HISTORY_FILE = os.getenv("SCROOGE_BALANCE_HISTORY_FILE", "balance_history.jsonl")
STATE_BACKUP_SUFFIX = ".bak"


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _state_path() -> Path:
    return Path(STATE_FILE).expanduser()


def _trade_history_path() -> Path:
    path = Path(TRADE_HISTORY_FILE).expanduser()
    if path.is_absolute():
        return path
    return _state_path().parent / path


def _balance_history_path() -> Path:
    path = Path(BALANCE_HISTORY_FILE).expanduser()
    if path.is_absolute():
        return path
    return _state_path().parent / path


def _default_state() -> dict[str, Any]:
    now_ms = _now_ms()
    return {
        "position": None,
        "balance": None,
        "last_price": None,
        "last_price_updated_at": None,
        "session_start": now_ms,
        "session_end": None,
        "trading_enabled": True,
    }


def _as_int_or_default(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_state(raw_state: Any) -> dict[str, Any]:
    default = _default_state()
    if not isinstance(raw_state, dict):
        return default

    normalized = dict(raw_state)

    position = normalized.get("position")
    if position is not None and not isinstance(position, dict):
        normalized["position"] = None
    else:
        normalized.setdefault("position", None)

    normalized["balance"] = _as_float_or_none(normalized.get("balance"))
    normalized["last_price"] = _as_float_or_none(normalized.get("last_price"))
    normalized["last_price_updated_at"] = _as_text_or_none(normalized.get("last_price_updated_at"))
    normalized["session_start"] = _as_int_or_default(normalized.get("session_start"), default["session_start"])
    normalized["session_end"] = _as_int_or_default(normalized.get("session_end"), None)
    normalized["trading_enabled"] = bool(normalized.get("trading_enabled", True))

    for key, value in default.items():
        normalized.setdefault(key, value)

    return normalized


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except json.JSONDecodeError as exc:
        print(f"[state] Malformed JSON in {path}: {exc}")
        return None
    except OSError as exc:
        print(f"[state] Failed to read {path}: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[state] Invalid state format in {path}: expected JSON object")
        return None
    return data


def _atomic_write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + STATE_BACKUP_SUFFIX)

    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(state, file_obj, indent=4, default=str)

    if path.exists():
        try:
            shutil.copy2(path, backup_path)
        except OSError as exc:
            print(f"[state] Failed to update backup {backup_path}: {exc}")

    tmp_path.replace(path)


def _jsonl_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _atomic_write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, default=str, separators=(",", ":")))
            file_obj.write("\n")
    tmp_path.replace(path)


def _append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, default=str, separators=(",", ":")))
        file_obj.write("\n")


def _load_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []

    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for idx, line in enumerate(file_obj, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    print(f"[state] Skipping malformed JSONL line {idx} in {path}: {exc}")
    except OSError as exc:
        print(f"[state] Failed to read history file {path}: {exc}")
        return []
    return rows


def _normalize_trade_history(raw_history: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_history, list):
        return []
    output: list[dict[str, Any]] = []
    for item in raw_history:
        if isinstance(item, dict):
            output.append(item)
    return output


def _normalize_balance_history(raw_history: Any) -> list[float]:
    if not isinstance(raw_history, list):
        return []
    output: list[float] = []
    for item in raw_history:
        if isinstance(item, dict):
            candidate = item.get("balance")
        else:
            candidate = item
        value = _as_float_or_none(candidate)
        if value is not None:
            output.append(value)
    return output


def _migrate_legacy_history_keys(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    has_trade_key = "trade_history" in state
    has_balance_key = "balance_history" in state
    if not has_trade_key and not has_balance_key:
        return state

    trade_history = _normalize_trade_history(state.get("trade_history"))
    balance_history = _normalize_balance_history(state.get("balance_history"))
    trade_path = _trade_history_path()
    balance_path = _balance_history_path()

    if has_trade_key and trade_history:
        try:
            if _jsonl_has_content(trade_path):
                print(f"[state] Preserving existing trade history file: {trade_path}")
            else:
                _atomic_write_jsonl(trade_path, trade_history)
                print(f"[state] Migrated {len(trade_history)} trades to {trade_path}")
        except OSError as exc:
            print(f"[state] Failed to migrate trade history: {exc}")

    if has_balance_key and balance_history:
        try:
            if _jsonl_has_content(balance_path):
                print(f"[state] Preserving existing balance history file: {balance_path}")
            else:
                balance_rows = [{"time": None, "balance": value} for value in balance_history]
                _atomic_write_jsonl(balance_path, balance_rows)
                print(f"[state] Migrated {len(balance_history)} balance points to {balance_path}")
        except OSError as exc:
            print(f"[state] Failed to migrate balance history: {exc}")

    state.pop("trade_history", None)
    state.pop("balance_history", None)
    try:
        _atomic_write_state(state_path, state)
        print(f"[state] Removed legacy history keys from {state_path}")
    except OSError as exc:
        print(f"[state] Failed to rewrite migrated state file {state_path}: {exc}")
    return state


def load_trade_history(limit: int | None = None) -> list[dict[str, Any]]:
    rows = _load_jsonl(_trade_history_path())
    output = [row for row in rows if isinstance(row, dict)]
    if limit is not None and limit > 0:
        return output[-limit:]
    return output


def load_balance_history(limit: int | None = None) -> list[float]:
    rows = _load_jsonl(_balance_history_path())
    output: list[float] = []
    for item in rows:
        if isinstance(item, dict):
            candidate = item.get("balance")
        else:
            candidate = item
        value = _as_float_or_none(candidate)
        if value is not None:
            output.append(value)
    if limit is not None and limit > 0:
        return output[-limit:]
    return output


def _write_trade_history_snapshot(history: list[dict[str, Any]]) -> None:
    _atomic_write_jsonl(_trade_history_path(), history)


def _write_balance_history_snapshot(history: list[float]) -> None:
    rows = [{"time": None, "balance": value} for value in history]
    _atomic_write_jsonl(_balance_history_path(), rows)


def load_state(include_history: bool = False) -> dict[str, Any]:
    """
    Load runtime state snapshot.
    `trade_history` and `balance_history` are loaded only when include_history=True.
    """
    path = _state_path()
    backup_path = path.with_suffix(path.suffix + STATE_BACKUP_SUFFIX)

    primary = _read_json(path)
    if primary is not None:
        normalized = _normalize_state(primary)
        normalized = _migrate_legacy_history_keys(normalized, path)
        if include_history:
            normalized["trade_history"] = load_trade_history()
            normalized["balance_history"] = load_balance_history()
        return normalized

    backup = _read_json(backup_path)
    if backup is not None:
        restored = _normalize_state(backup)
        try:
            _atomic_write_state(path, restored)
            print(f"[state] Restored state from backup: {backup_path}")
        except OSError as exc:
            print(f"[state] Failed to restore primary state file from backup: {exc}")
        restored = _migrate_legacy_history_keys(restored, path)
        if include_history:
            restored["trade_history"] = load_trade_history()
            restored["balance_history"] = load_balance_history()
        return restored

    default = _default_state()
    if include_history:
        default["trade_history"] = load_trade_history()
        default["balance_history"] = load_balance_history()
    return default


def save_state(state: dict[str, Any]) -> None:
    """
    Persist runtime snapshot atomically and keep backup copy for crash-safe recovery.
    If legacy history keys are present in-memory, write them to separate history files.
    """
    normalized = _normalize_state(state)
    trade_history = _normalize_trade_history(normalized.pop("trade_history", None))
    balance_history = _normalize_balance_history(normalized.pop("balance_history", None))

    if trade_history:
        try:
            _write_trade_history_snapshot(trade_history)
        except OSError as exc:
            print(f"[state] Failed to write trade history snapshot: {exc}")

    if balance_history:
        try:
            _write_balance_history_snapshot(balance_history)
        except OSError as exc:
            print(f"[state] Failed to write balance history snapshot: {exc}")

    normalized["session_end"] = _now_ms()
    if normalized.get("session_start") is None:
        normalized["session_start"] = normalized["session_end"]

    if normalized.get("balance") is None:
        latest = load_balance_history(limit=1)
        if latest:
            normalized["balance"] = latest[-1]

    _atomic_write_state(_state_path(), normalized)
    state.clear()
    state.update(normalized)


def add_closed_trade(state: dict[str, Any], trade: dict[str, Any]) -> None:
    """
    Append a closed trade to trade history and persist.
    """
    try:
        _append_jsonl(_trade_history_path(), trade)
    except OSError as exc:
        print(f"[state] Failed to append trade history: {exc}")
    save_state(state)


def update_position(state: dict[str, Any], position: dict[str, Any] | None) -> None:
    """
    Update current open position in state and persist.
    """
    state["position"] = position
    save_state(state)


def update_balance(state: dict[str, Any], balance: float) -> None:
    """
    Update current balance in state and persist.
    Append to balance history only when the value changed.
    """
    balance_value = _as_float_or_none(balance)
    previous_value = _as_float_or_none(state.get("balance"))
    should_append = False
    if balance_value is not None:
        if previous_value is None:
            latest = load_balance_history(limit=1)
            previous_value = latest[-1] if latest else None
        should_append = previous_value is None or not math.isclose(
            balance_value,
            previous_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    if should_append:
        try:
            _append_jsonl(
                _balance_history_path(),
                {"time": _now_ms(), "balance": balance_value},
            )
        except OSError as exc:
            print(f"[state] Failed to append balance history: {exc}")
    state["balance"] = balance_value
    save_state(state)


def set_trading_enabled(state: dict[str, Any], enabled: bool) -> None:
    """
    Update trading enabled flag and persist.
    """
    state["trading_enabled"] = bool(enabled)
    save_state(state)
