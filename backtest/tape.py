from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TAPE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(slots=True)
class DiscreteMarketTapeRow:
    symbol: str
    open_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    EMA: float | None
    RSI: float | None
    BBL: float | None
    BBM: float | None
    BBU: float | None
    ATR: float | None


def _format_open_time(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        dt_value = value.tz_convert(None) if value.tzinfo is not None else value
        return dt_value.strftime(TAPE_TIMESTAMP_FORMAT)

    if isinstance(value, datetime):
        dt_value = value.replace(tzinfo=None) if value.tzinfo is not None else value
        return dt_value.strftime(TAPE_TIMESTAMP_FORMAT)

    text = str(value).strip()
    return text or datetime.now().strftime(TAPE_TIMESTAMP_FORMAT)


def _to_optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def build_discrete_market_tape(df: pd.DataFrame, *, symbol: str) -> list[DiscreteMarketTapeRow]:
    if df is None or df.empty:
        return []

    tape: list[DiscreteMarketTapeRow] = []
    for row in df.itertuples(index=False):
        tape.append(
            DiscreteMarketTapeRow(
                symbol=symbol,
                open_time=_format_open_time(getattr(row, "open_time", None)),
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume")),
                EMA=_to_optional_float(getattr(row, "EMA", None)),
                RSI=_to_optional_float(getattr(row, "RSI", None)),
                BBL=_to_optional_float(getattr(row, "BBL", None)),
                BBM=_to_optional_float(getattr(row, "BBM", None)),
                BBU=_to_optional_float(getattr(row, "BBU", None)),
                ATR=_to_optional_float(getattr(row, "ATR", None)),
            )
        )
    return tape


def write_discrete_market_tape(path: str | Path, tape: Iterable[DiscreteMarketTapeRow]) -> Path:
    target_path = Path(path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file_obj:
        for row in tape:
            file_obj.write(json.dumps(asdict(row), ensure_ascii=True, sort_keys=True))
            file_obj.write("\n")
    return target_path


def discrete_market_tape_to_frame(tape: Iterable[DiscreteMarketTapeRow]) -> pd.DataFrame:
    rows = [asdict(row) for row in tape]
    if not rows:
        return pd.DataFrame(
            columns=[
                "open_time",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "EMA",
                "RSI",
                "BBL",
                "BBM",
                "BBU",
                "ATR",
            ]
        )

    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], format=TAPE_TIMESTAMP_FORMAT, errors="coerce")
    return df


def read_discrete_market_tape(path: str | Path) -> list[DiscreteMarketTapeRow]:
    target_path = Path(path).expanduser()
    if not target_path.exists():
        return []

    rows: list[DiscreteMarketTapeRow] = []
    with target_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            rows.append(DiscreteMarketTapeRow(**payload))
    return rows


def read_discrete_market_tape_frame(path: str | Path) -> pd.DataFrame:
    return discrete_market_tape_to_frame(read_discrete_market_tape(path))
