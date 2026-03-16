from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from core.event_store import EventRecord, read_event_records, resolve_event_log_path


OPEN_TRADE_CODE = "trade_opened"
TRAIL_EVENT_CODES = {"trail_activated", "trail_moved"}
CLOSE_EVENT_CODES = {
    "trade_closed_take_profit",
    "trade_closed_stop_loss",
    "trade_closed_rsi_extreme",
    "trade_closed_manual",
    "trade_liquidated",
}
REPLAY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, REPLAY_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _normalized_side(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"buy", "long"}:
        return "long"
    if normalized in {"sell", "short"}:
        return "short"
    return normalized or "unknown"


def _event_symbol(event: EventRecord) -> str | None:
    symbol = str(event.get("context", {}).get("symbol") or "").strip().upper()
    return symbol or None


def _filter_value_match(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    return (actual or "").strip().lower() == expected.strip().lower()


@dataclass(slots=True)
class ReplayTrade:
    symbol: str
    side: str
    entry_ts: str
    exit_ts: str | None
    entry: float | None
    exit: float | None
    sl: float | None
    tp: float | None
    size: float | None
    leverage: float | None
    trigger: str | None
    stake_mode: str | None
    opening_fee: float | None
    net_pnl: float | None
    roi_pct: float | None
    exit_code: str | None
    exit_reason: str | None
    via_tail_guard: bool
    trail_activated_at: str | None
    trail_moves: int
    duration_seconds: float | None
    open_event_id: str
    close_event_id: str | None


@dataclass(slots=True)
class ReplaySummary:
    event_path: str
    runtime_mode: str | None
    strategy_mode: str | None
    symbol: str | None
    total_events: int
    total_trades: int
    closed_trades: int
    open_trades: int
    wins: int
    losses: int
    breakeven: int
    net_pnl: float
    average_roi_pct: float | None
    symbols: list[str]
    exit_code_counts: dict[str, int]
    trail_activations: int
    trail_moves: int
    manual_suggestions: int
    command_failures: int


def load_replay_events(
    path: str | Path | None = None,
    *,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
    symbol: str | None = None,
) -> list[EventRecord]:
    target_symbol = symbol.strip().upper() if isinstance(symbol, str) and symbol.strip() else None
    events = read_event_records(path)
    output: list[EventRecord] = []
    for event in events:
        if not _filter_value_match(event.get("runtime_mode"), runtime_mode):
            continue
        if not _filter_value_match(event.get("strategy_mode"), strategy_mode):
            continue
        if target_symbol is not None and _event_symbol(event) != target_symbol:
            continue
        output.append(event)
    return output


def replay_discrete_trades(events: Iterable[EventRecord]) -> list[ReplayTrade]:
    trades: list[ReplayTrade] = []
    open_trade: dict[str, Any] | None = None

    for event in events:
        code = str(event.get("code") or "").strip()
        context = dict(event.get("context") or {})

        if code == OPEN_TRADE_CODE:
            if open_trade is not None:
                trades.append(_build_replay_trade(open_trade))
            open_trade = {
                "symbol": _event_symbol(event) or "UNKNOWN",
                "side": _normalized_side(context.get("side")),
                "entry_ts": str(event.get("ts") or "").strip(),
                "entry": _as_float(context.get("entry")),
                "sl": _as_float(context.get("sl")),
                "tp": _as_float(context.get("tp")),
                "size": _as_float(context.get("size")),
                "leverage": _as_float(context.get("leverage")),
                "trigger": str(context.get("trigger") or "").strip() or None,
                "stake_mode": str(context.get("stake_mode") or "").strip() or None,
                "opening_fee": _as_float(context.get("fee")),
                "open_event_id": str(event.get("event_id") or "").strip(),
                "close_event_id": None,
                "trail_activated_at": None,
                "trail_moves": 0,
                "exit_ts": None,
                "exit": None,
                "net_pnl": None,
                "roi_pct": None,
                "exit_code": None,
                "exit_reason": None,
                "via_tail_guard": False,
            }
            continue

        if open_trade is None:
            continue

        if code in TRAIL_EVENT_CODES:
            if open_trade["trail_activated_at"] is None and code == "trail_activated":
                open_trade["trail_activated_at"] = str(event.get("ts") or "").strip() or None
            if code == "trail_moved":
                open_trade["trail_moves"] += 1
            continue

        if code in CLOSE_EVENT_CODES:
            open_trade["exit_ts"] = str(event.get("ts") or "").strip() or None
            open_trade["exit"] = _as_float(context.get("exit") or context.get("liq_price"))
            open_trade["net_pnl"] = _as_float(context.get("net_pnl"))
            open_trade["roi_pct"] = _as_float(context.get("roi_pct"))
            open_trade["exit_code"] = code
            open_trade["close_event_id"] = str(event.get("event_id") or "").strip() or None
            open_trade["via_tail_guard"] = bool(context.get("via_tail_guard", False))
            if code == "trade_closed_take_profit":
                open_trade["exit_reason"] = "take_profit_tail" if open_trade["via_tail_guard"] else "take_profit"
            elif code == "trade_closed_stop_loss":
                open_trade["exit_reason"] = "stop_loss"
            elif code == "trade_closed_rsi_extreme":
                open_trade["exit_reason"] = "rsi_extreme"
            elif code == "trade_closed_manual":
                open_trade["exit_reason"] = "manual_close"
            elif code == "trade_liquidated":
                open_trade["exit_reason"] = "liquidated"

            trades.append(_build_replay_trade(open_trade))
            open_trade = None

    if open_trade is not None:
        trades.append(_build_replay_trade(open_trade))

    return trades


def summarize_replay(
    path: str | Path | None = None,
    *,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
    symbol: str | None = None,
) -> ReplaySummary:
    resolved_path = resolve_event_log_path(path)
    events = load_replay_events(
        resolved_path,
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        symbol=symbol,
    )
    trades = replay_discrete_trades(events)

    symbols = sorted({trade.symbol for trade in trades if trade.symbol})
    closed_trades = [trade for trade in trades if trade.exit_code is not None]
    pnl_values = [trade.net_pnl for trade in closed_trades if trade.net_pnl is not None]
    roi_values = [trade.roi_pct for trade in closed_trades if trade.roi_pct is not None]
    exit_counts = Counter(trade.exit_code for trade in closed_trades if trade.exit_code)

    wins = sum(1 for value in pnl_values if value > 0)
    losses = sum(1 for value in pnl_values if value < 0)
    breakeven = sum(1 for value in pnl_values if value == 0)

    return ReplaySummary(
        event_path=str(resolved_path),
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        symbol=symbol.strip().upper() if isinstance(symbol, str) and symbol.strip() else None,
        total_events=len(events),
        total_trades=len(trades),
        closed_trades=len(closed_trades),
        open_trades=len(trades) - len(closed_trades),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        net_pnl=sum(pnl_values),
        average_roi_pct=(sum(roi_values) / len(roi_values)) if roi_values else None,
        symbols=symbols,
        exit_code_counts=dict(exit_counts),
        trail_activations=sum(1 for event in events if event.get("code") == "trail_activated"),
        trail_moves=sum(1 for event in events if event.get("code") == "trail_moved"),
        manual_suggestions=sum(1 for event in events if event.get("code") == "manual_trade_suggested"),
        command_failures=sum(1 for event in events if event.get("code") == "command_failed"),
    )


def _build_replay_trade(raw_trade: dict[str, Any]) -> ReplayTrade:
    entry_dt = _parse_ts(raw_trade.get("entry_ts"))
    exit_dt = _parse_ts(raw_trade.get("exit_ts"))
    duration_seconds = None
    if entry_dt is not None and exit_dt is not None:
        duration_seconds = max(0.0, (exit_dt - entry_dt).total_seconds())

    return ReplayTrade(
        symbol=str(raw_trade.get("symbol") or "UNKNOWN"),
        side=_normalized_side(raw_trade.get("side")),
        entry_ts=str(raw_trade.get("entry_ts") or ""),
        exit_ts=raw_trade.get("exit_ts"),
        entry=_as_float(raw_trade.get("entry")),
        exit=_as_float(raw_trade.get("exit")),
        sl=_as_float(raw_trade.get("sl")),
        tp=_as_float(raw_trade.get("tp")),
        size=_as_float(raw_trade.get("size")),
        leverage=_as_float(raw_trade.get("leverage")),
        trigger=raw_trade.get("trigger"),
        stake_mode=raw_trade.get("stake_mode"),
        opening_fee=_as_float(raw_trade.get("opening_fee")),
        net_pnl=_as_float(raw_trade.get("net_pnl")),
        roi_pct=_as_float(raw_trade.get("roi_pct")),
        exit_code=raw_trade.get("exit_code"),
        exit_reason=raw_trade.get("exit_reason"),
        via_tail_guard=bool(raw_trade.get("via_tail_guard", False)),
        trail_activated_at=raw_trade.get("trail_activated_at"),
        trail_moves=int(raw_trade.get("trail_moves") or 0),
        duration_seconds=duration_seconds,
        open_event_id=str(raw_trade.get("open_event_id") or ""),
        close_event_id=raw_trade.get("close_event_id"),
    )


def replay_trades_as_dicts(
    path: str | Path | None = None,
    *,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    events = load_replay_events(
        path,
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        symbol=symbol,
    )
    return [asdict(trade) for trade in replay_discrete_trades(events)]


def write_replay_artifacts(
    path: str | Path | None = None,
    *,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
    symbol: str | None = None,
    summary_path: str | Path | None = None,
    trades_path: str | Path | None = None,
) -> ReplaySummary:
    resolved_event_path = resolve_event_log_path(path)
    summary = summarize_replay(
        resolved_event_path,
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        symbol=symbol,
    )
    trades = replay_trades_as_dicts(
        resolved_event_path,
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        symbol=symbol,
    )

    resolved_summary_path = Path(summary_path) if summary_path is not None else resolved_event_path.with_name("replay_summary.json")
    resolved_trades_path = Path(trades_path) if trades_path is not None else resolved_event_path.with_name("replay_trades.jsonl")

    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_trades_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(summary), file_obj, ensure_ascii=True, indent=2, sort_keys=True)
        file_obj.write("\n")

    with resolved_trades_path.open("w", encoding="utf-8") as file_obj:
        for trade in trades:
            file_obj.write(json.dumps(trade, ensure_ascii=True, sort_keys=True, default=str))
            file_obj.write("\n")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Scrooge canonical event history.")
    parser.add_argument("path", nargs="?", default=None, help="Path to event_history.jsonl")
    parser.add_argument("--runtime-mode", default=None, help="Filter by runtime_mode")
    parser.add_argument("--strategy-mode", default=None, help="Filter by strategy_mode")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="Print reconstructed discrete trades instead of only summary",
    )
    args = parser.parse_args()

    if args.show_trades:
        print(
            json.dumps(
                replay_trades_as_dicts(
                    args.path,
                    runtime_mode=args.runtime_mode,
                    strategy_mode=args.strategy_mode,
                    symbol=args.symbol,
                ),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                asdict(
                    summarize_replay(
                        args.path,
                        runtime_mode=args.runtime_mode,
                        strategy_mode=args.strategy_mode,
                        symbol=args.symbol,
                    )
                ),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
