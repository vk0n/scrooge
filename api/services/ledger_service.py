from __future__ import annotations

import base64
import binascii
import json
import sys
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from shared.runtime_db import count_ledger_entries, list_ledger_entries, runtime_db_path  # noqa: E402
from shared.treasury_ledger import project_portfolio_transactions  # noqa: E402


def _encode_cursor(cursor: tuple[int, int] | None) -> str | None:
    if cursor is None:
        return None
    raw = json.dumps({"ts": cursor[0], "id": cursor[1]}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[int, int] | None:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return int(payload["ts"]), int(payload["id"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("Invalid Ledger cursor.") from exc


def load_ledger(*, scope: str = "all", limit: int = 30, cursor: str | None = None) -> dict[str, Any]:
    project_portfolio_transactions()
    entries, next_key = list_ledger_entries(
        scope=scope,
        limit=limit,
        before=_decode_cursor(cursor),
    )
    return {
        "path": str(runtime_db_path()),
        "scope": scope,
        "entries": entries,
        "returned_entries": len(entries),
        "total_entries": count_ledger_entries(scope=scope),
        "has_earlier": next_key is not None,
        "next_cursor": _encode_cursor(next_key),
        "warnings": [],
    }
