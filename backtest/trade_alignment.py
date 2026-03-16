from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest.market_event_replay import (
    ObservedExecutionTrade,
    build_observed_execution_fills,
    reconstruct_observed_execution_trades,
)
from core.market_events import MarketEvent


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _delta_seconds(left: Any, right: Any) -> float | None:
    left_ts = _parse_ts(left)
    right_ts = _parse_ts(right)
    if left_ts is None or right_ts is None:
        return None
    return (left_ts - right_ts).total_seconds()


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


@dataclass(slots=True)
class TradeAlignmentPair:
    pair_index: int
    strategy_side: str | None
    observed_side: str | None
    side_match: bool
    strategy_entry_ts: str | None
    observed_entry_ts: str | None
    strategy_exit_ts: str | None
    observed_exit_ts: str | None
    entry_ts_delta_seconds: float | None
    exit_ts_delta_seconds: float | None
    strategy_net_pnl: float | None
    observed_realized_pnl: float | None
    pnl_delta: float | None


@dataclass(slots=True)
class TradeAlignmentSummary:
    strategy_total_trades: int
    observed_total_trades: int
    strategy_closed_trades: int
    observed_closed_trades: int
    paired_trades: int
    side_matches: int
    side_mismatches: int
    strategy_net_pnl: float
    observed_realized_pnl: float
    pnl_delta: float
    average_entry_ts_delta_seconds: float | None
    average_exit_ts_delta_seconds: float | None


def _normalize_strategy_trades(trades: pd.DataFrame | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(trades, pd.DataFrame):
        return trades.to_dict(orient="records")
    output: list[dict[str, Any]] = []
    for item in trades:
        if isinstance(item, dict):
            output.append(item)
    return output


def _closed_strategy_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trade in trades:
        exit_ts = str(trade.get("exit_time") or "").strip()
        if exit_ts:
            output.append(trade)
    return output


def build_trade_alignment_pairs(
    strategy_trades: pd.DataFrame | Iterable[dict[str, Any]],
    observed_trades: Iterable[ObservedExecutionTrade],
) -> list[TradeAlignmentPair]:
    strategy_rows = _closed_strategy_trades(_normalize_strategy_trades(strategy_trades))
    observed_rows = [trade for trade in observed_trades if trade.status == "closed"]
    pair_count = min(len(strategy_rows), len(observed_rows))
    output: list[TradeAlignmentPair] = []

    for idx in range(pair_count):
        strategy_trade = strategy_rows[idx]
        observed_trade = observed_rows[idx]
        strategy_net_pnl = _as_float(strategy_trade.get("net_pnl"))
        observed_realized_pnl = _as_float(observed_trade.realized_pnl)
        pnl_delta = None
        if strategy_net_pnl is not None and observed_realized_pnl is not None:
            pnl_delta = strategy_net_pnl - observed_realized_pnl

        strategy_side = str(strategy_trade.get("side") or "").strip().lower() or None
        observed_side = str(observed_trade.side or "").strip().lower() or None
        output.append(
            TradeAlignmentPair(
                pair_index=idx + 1,
                strategy_side=strategy_side,
                observed_side=observed_side,
                side_match=(strategy_side == observed_side),
                strategy_entry_ts=str(strategy_trade.get("entry_time") or strategy_trade.get("time") or "").strip() or None,
                observed_entry_ts=observed_trade.entry_ts,
                strategy_exit_ts=str(strategy_trade.get("exit_time") or "").strip() or None,
                observed_exit_ts=observed_trade.exit_ts,
                entry_ts_delta_seconds=_delta_seconds(
                    strategy_trade.get("entry_time") or strategy_trade.get("time"),
                    observed_trade.entry_ts,
                ),
                exit_ts_delta_seconds=_delta_seconds(
                    strategy_trade.get("exit_time"),
                    observed_trade.exit_ts,
                ),
                strategy_net_pnl=strategy_net_pnl,
                observed_realized_pnl=observed_realized_pnl,
                pnl_delta=pnl_delta,
            )
        )

    return output


def summarize_trade_alignment(
    strategy_trades: pd.DataFrame | Iterable[dict[str, Any]],
    observed_trades: Iterable[ObservedExecutionTrade],
) -> TradeAlignmentSummary:
    strategy_rows = _normalize_strategy_trades(strategy_trades)
    strategy_closed = _closed_strategy_trades(strategy_rows)
    observed_rows = list(observed_trades)
    observed_closed = [trade for trade in observed_rows if trade.status == "closed"]
    pairs = build_trade_alignment_pairs(strategy_rows, observed_rows)

    entry_deltas = [pair.entry_ts_delta_seconds for pair in pairs if pair.entry_ts_delta_seconds is not None]
    exit_deltas = [pair.exit_ts_delta_seconds for pair in pairs if pair.exit_ts_delta_seconds is not None]
    strategy_net_pnl = sum(_as_float(trade.get("net_pnl")) or 0.0 for trade in strategy_closed)
    observed_realized_pnl = sum(float(trade.realized_pnl or 0.0) for trade in observed_closed)

    return TradeAlignmentSummary(
        strategy_total_trades=len(strategy_rows),
        observed_total_trades=len(observed_rows),
        strategy_closed_trades=len(strategy_closed),
        observed_closed_trades=len(observed_closed),
        paired_trades=len(pairs),
        side_matches=sum(1 for pair in pairs if pair.side_match),
        side_mismatches=sum(1 for pair in pairs if not pair.side_match),
        strategy_net_pnl=strategy_net_pnl,
        observed_realized_pnl=observed_realized_pnl,
        pnl_delta=strategy_net_pnl - observed_realized_pnl,
        average_entry_ts_delta_seconds=(sum(entry_deltas) / len(entry_deltas) if entry_deltas else None),
        average_exit_ts_delta_seconds=(sum(exit_deltas) / len(exit_deltas) if exit_deltas else None),
    )


def write_trade_alignment_artifacts(
    strategy_trades: pd.DataFrame | Iterable[dict[str, Any]],
    market_events: Iterable[MarketEvent],
    *,
    symbol: str | None = None,
    summary_path: str | Path,
    pairs_path: str | Path,
) -> TradeAlignmentSummary:
    fills = build_observed_execution_fills(market_events, symbol=symbol)
    observed_trades = reconstruct_observed_execution_trades(fills)
    summary = summarize_trade_alignment(strategy_trades, observed_trades)
    pairs = build_trade_alignment_pairs(strategy_trades, observed_trades)

    resolved_summary_path = Path(summary_path).expanduser()
    resolved_pairs_path = Path(pairs_path).expanduser()
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(summary), file_obj, ensure_ascii=True, indent=2, sort_keys=True)
        file_obj.write("\n")

    with resolved_pairs_path.open("w", encoding="utf-8") as file_obj:
        for pair in pairs:
            file_obj.write(json.dumps(asdict(pair), ensure_ascii=True, sort_keys=True))
            file_obj.write("\n")

    return summary
