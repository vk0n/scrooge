from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from shared.runtime_db import list_ui_log_lines, runtime_db_path  # noqa: E402


RUNTIME_DB_PATH = runtime_db_path()


def read_last_log_lines(lines: int) -> list[str]:
    if lines < 1:
        raise ValueError("lines must be >= 1")
    return list_ui_log_lines(limit=lines)


def log_source_path() -> Path:
    return RUNTIME_DB_PATH
