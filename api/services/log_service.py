from __future__ import annotations

from collections import deque
import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


LOG_PATH = Path(os.getenv("SCROOGE_LOG_PATH", str(_project_root() / "trading_log.txt"))).expanduser()


def _ensure_log_file_exists() -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.touch(exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to initialize log file: {LOG_PATH}") from exc


def read_last_log_lines(lines: int) -> list[str]:
    if lines < 1:
        raise ValueError("lines must be >= 1")

    _ensure_log_file_exists()

    buffer: deque[str] = deque(maxlen=lines)
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line in file_obj:
                buffer.append(line.rstrip("\n"))
    except OSError as exc:
        raise OSError(f"Failed to read log file: {LOG_PATH}") from exc

    return list(buffer)
