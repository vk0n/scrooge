from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


STATE_PATH = Path(os.getenv("SCROOGE_STATE_PATH", str(_project_root() / "state.json"))).expanduser()


def load_state() -> tuple[dict[str, Any], list[str]]:
    if not STATE_PATH.exists():
        return (
            {
                "position": None,
                "balance": None,
                "session_start": None,
                "session_end": None,
                "trading_enabled": True,
            },
            [f"State file not found: {STATE_PATH}"],
        )

    try:
        with STATE_PATH.open("r", encoding="utf-8") as file_obj:
            raw = json.load(file_obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed state JSON: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read state file: {STATE_PATH}") from exc

    if not isinstance(raw, dict):
        raise ValueError("State must be a JSON object")

    raw.setdefault("trading_enabled", True)
    return raw, []


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_balance(state: dict[str, Any], default_balance: Any = None) -> float | None:
    balance = _maybe_float(state.get("balance"))
    if balance is not None:
        return balance

    history = state.get("balance_history")
    if isinstance(history, list) and history:
        balance = _maybe_float(history[-1])
        if balance is not None:
            return balance

    return _maybe_float(default_balance)


def resolve_current_position(position: Any) -> str | None:
    if not isinstance(position, dict):
        return None

    side = position.get("side")
    return str(side) if side is not None else "open"


def resolve_trailing_state(position: Any) -> dict[str, Any] | None:
    if not isinstance(position, dict):
        return None

    return {
        "trail_active": bool(position.get("trail_active", False)),
        "trail_max": position.get("trail_max"),
        "trail_min": position.get("trail_min"),
        "tp": position.get("tp"),
        "sl": position.get("sl"),
    }


def resolve_last_update_timestamp(state: dict[str, Any]) -> str | None:
    for key in ("session_end", "session_start"):
        value = state.get(key)
        if value is None:
            continue
        try:
            ts_ms = int(value)
            return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()
        except (TypeError, ValueError, OSError):
            continue
    return None


def resolve_trading_enabled(state: dict[str, Any]) -> bool:
    return bool(state.get("trading_enabled", True))
