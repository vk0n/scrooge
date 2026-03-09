from __future__ import annotations

from collections import deque
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


LOG_PATH = _project_root() / "trading_log.txt"


def read_last_log_lines(lines: int) -> list[str]:
    if lines < 1:
        raise ValueError("lines must be >= 1")
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    buffer: deque[str] = deque(maxlen=lines)
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line in file_obj:
                buffer.append(line.rstrip("\n"))
    except OSError as exc:
        raise OSError(f"Failed to read log file: {LOG_PATH}") from exc

    return list(buffer)

