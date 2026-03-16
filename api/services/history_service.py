from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_state_path() -> Path:
    raw_state_path = os.getenv("SCROOGE_STATE_PATH", str(_project_root() / "runtime" / "state.json"))
    return Path(raw_state_path).expanduser()


def _resolve_history_path(env_name: str, default_filename: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _default_state_path().parent / default_filename


TRADE_HISTORY_PATH = _resolve_history_path("SCROOGE_TRADE_HISTORY_PATH", "trade_history.jsonl")
BALANCE_HISTORY_PATH = _resolve_history_path("SCROOGE_BALANCE_HISTORY_PATH", "balance_history.jsonl")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> tuple[list[Any], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return [], warnings

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
                    warnings.append(f"Malformed JSONL line {idx} in {path}: {exc}")
    except OSError as exc:
        warnings.append(f"Failed to read history file: {path} ({exc})")
        return [], warnings

    return rows, warnings


def load_trade_history(limit: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    rows, warnings = _load_jsonl(TRADE_HISTORY_PATH)
    trades = [row for row in rows if isinstance(row, dict)]
    if limit is not None and limit > 0:
        trades = trades[-limit:]
    return trades, warnings


def load_balance_history(limit: int | None = None) -> tuple[list[float], list[str]]:
    rows, warnings = _load_jsonl(BALANCE_HISTORY_PATH)
    balances: list[float] = []
    for row in rows:
        if isinstance(row, dict):
            candidate = row.get("balance")
        else:
            candidate = row
        value = _to_float(candidate)
        if value is not None:
            balances.append(value)
    if limit is not None and limit > 0:
        balances = balances[-limit:]
    return balances, warnings
