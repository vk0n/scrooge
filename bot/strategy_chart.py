from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.engine import DiscreteRowSnapshot, StrategyRuntime
from shared.runtime_db import save_strategy_chart_snapshots


class StrategyChartRecorder:
    """Keep minute samples and exact entry decisions from the running strategy."""

    def __init__(self, symbol: str, *, path: Path | None = None) -> None:
        self.symbol = symbol
        self.path = path
        self.pending: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.last_flush = 0.0
        self.last_entry: tuple[str, str] | None = None

    def observe(self, snapshot: DiscreteRowSnapshot, runtime: StrategyRuntime) -> None:
        decision_time = snapshot.log_ts if runtime.live and snapshot.raw_row is not None else snapshot.row_ts
        timestamp = datetime.fromisoformat(decision_time.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        ts_ms = int(timestamp.timestamp() * 1000)
        row = {
            "symbol": self.symbol,
            "ts_ms": ts_ms,
            "time": timestamp.astimezone(UTC).isoformat(),
            "price": float(snapshot.price),
            "ema": float(snapshot.ema),
            "rsi": float(snapshot.rsi),
            "bbl": float(snapshot.lower),
            "bbm": float(snapshot.mid),
            "bbu": float(snapshot.upper),
            "atr": float(snapshot.atr),
            "kind": "sample",
            "bucket_ms": ts_ms // 60_000 * 60_000,
        }
        sample_key = (self.symbol, "sample", row["bucket_ms"])
        if self.pending.get(sample_key, {}).get("ts_ms", -1) <= ts_ms:
            self.pending[sample_key] = row
        position = runtime.position
        entry_key = (self.symbol, snapshot.row_ts)
        if position and position.get("decision_time") == snapshot.row_ts and entry_key != self.last_entry:
            # A later sample in this minute must not erase the entry decision.
            entry_row = {**row, "kind": "entry", "bucket_ms": ts_ms}
            self.pending[(self.symbol, "entry", ts_ms)] = entry_row
            self.last_entry = entry_key

    def flush(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not self.pending or (not force and now - self.last_flush < 1.0):
            return
        save_strategy_chart_snapshots(list(self.pending.values()), path=self.path)
        self.pending.clear()
        self.last_flush = now
