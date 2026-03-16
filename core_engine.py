from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm


@dataclass(slots=True)
class DiscreteRowSnapshot:
    raw_row: Any
    price: Any
    lower: Any
    upper: Any
    mid: Any
    atr: Any
    rsi: Any
    ema: Any
    row_ts: str
    log_ts: str


@dataclass(slots=True)
class StrategyRuntime:
    state: dict[str, Any]
    balance: float
    position: dict[str, Any] | None
    trade_history: list[dict[str, Any]]
    balance_history: list[float]
    log_buffer: list[str]
    live: bool
    use_state: bool
    enable_logs: bool


@dataclass(slots=True)
class StrategyConfig:
    qty: float | None
    sl_mult: float
    tp_mult: float
    symbol: str
    leverage: float
    use_full_balance: bool
    fee_rate: float
    rsi_extreme_long: float
    rsi_extreme_short: float
    rsi_long_open_threshold: float
    rsi_long_qty_threshold: float
    rsi_long_tp_threshold: float
    rsi_long_close_threshold: float
    rsi_short_open_threshold: float
    rsi_short_qty_threshold: float
    rsi_short_tp_threshold: float
    rsi_short_close_threshold: float
    trail_atr_mult: float
    allow_entries: bool


@dataclass(slots=True)
class EntryDecision:
    side: str
    size: float
    entry: float
    sl: float
    tp: float
    liq_price: float
    stake_mode: str
    trigger: str


@dataclass(slots=True)
class EntryGuardRejection:
    side: str
    entry: float
    sl: float
    liq_price: float
    trigger: str


def initialize_strategy_runtime(
    *,
    live: bool,
    initial_balance: float,
    use_state: bool,
    load_state_fn: Callable[..., dict[str, Any]],
) -> StrategyRuntime:
    if use_state:
        state = load_state_fn(include_history=not live)
    else:
        state = {
            "position": None,
            "trade_history": [],
            "balance_history": [],
            "manual_trade_suggestion": None,
            "updated_at": None,
            "search_status": None,
            "bot_status": None,
            "trade_status": None,
            "session_start": None,
            "session_end": None,
        }

    return StrategyRuntime(
        state=state,
        balance=initial_balance,
        position=state["position"],
        trade_history=state.get("trade_history", []),
        balance_history=state.get("balance_history", []),
        log_buffer=[],
        live=live,
        use_state=use_state,
        enable_logs=True,
    )


def iter_discrete_rows(df: pd.DataFrame, *, live: bool, show_progress: bool) -> Any:
    if live:
        return [df.iloc[-1]]

    df_iter = [df.iloc[i] for i in range(1, len(df))]
    return tqdm(df_iter, desc="Backtest Progress", disable=not show_progress)


def build_row_snapshot(
    row: Any,
    *,
    live: bool,
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
) -> DiscreteRowSnapshot:
    row_ts = timestamp_formatter(row.get("open_time"))
    log_ts = datetime.now().strftime(timestamp_format) if live else row_ts
    return DiscreteRowSnapshot(
        raw_row=row,
        price=row["close"],
        lower=row["BBL"],
        upper=row["BBU"],
        mid=row["BBM"],
        atr=row["ATR"],
        rsi=row["RSI"],
        ema=row["EMA"],
        row_ts=row_ts,
        log_ts=log_ts,
    )


def calc_liquidation_price(entry: float, leverage_value: float, side: str) -> float:
    if side == "long":
        return entry * (1 - 1 / leverage_value)
    return entry * (1 + 1 / leverage_value)


def resolve_entry_decision(
    snapshot: DiscreteRowSnapshot,
    *,
    config: StrategyConfig,
    qty_local: float,
    manual_side: str | None,
) -> EntryDecision | EntryGuardRejection | None:
    price = snapshot.price
    lower = snapshot.lower
    upper = snapshot.upper
    atr = snapshot.atr
    rsi = snapshot.rsi
    ema = snapshot.ema

    manual_long = manual_side == "buy"
    manual_short = manual_side == "sell"

    if (price < lower and rsi < config.rsi_long_open_threshold and price > ema) or manual_long:
        size = qty_local * 0.5 if rsi > config.rsi_long_qty_threshold else qty_local
        sl = price - atr * config.sl_mult
        tp = price + atr * config.tp_mult
        liq_price = calc_liquidation_price(price, config.leverage, "long")
        trigger = "manual_suggestion" if manual_long else "strategy_rules"
        if sl < liq_price:
            return EntryGuardRejection(
                side="long",
                entry=price,
                sl=sl,
                liq_price=liq_price,
                trigger=trigger,
            )
        return EntryDecision(
            side="long",
            size=size,
            entry=price,
            sl=sl,
            tp=tp,
            liq_price=liq_price,
            stake_mode="half" if rsi > config.rsi_long_qty_threshold else "full",
            trigger=trigger,
        )

    if (price > upper and rsi > config.rsi_short_open_threshold and price < ema) or manual_short:
        size = qty_local * 0.5 if rsi < config.rsi_short_qty_threshold else qty_local
        sl = price + atr * config.sl_mult
        tp = price - atr * config.tp_mult
        liq_price = calc_liquidation_price(price, config.leverage, "short")
        trigger = "manual_suggestion" if manual_short else "strategy_rules"
        if sl > liq_price:
            return EntryGuardRejection(
                side="short",
                entry=price,
                sl=sl,
                liq_price=liq_price,
                trigger=trigger,
            )
        return EntryDecision(
            side="short",
            size=size,
            entry=price,
            sl=sl,
            tp=tp,
            liq_price=liq_price,
            stake_mode="half" if rsi < config.rsi_short_qty_threshold else "full",
            trigger=trigger,
        )

    return None


def build_position_from_entry(decision: EntryDecision, *, row_ts: str) -> dict[str, Any]:
    return {
        "side": decision.side,
        "size": decision.size,
        "entry": decision.entry,
        "sl": decision.sl,
        "tp": decision.tp,
        "liq_price": decision.liq_price,
        "trail_active": False,
        "trail_price": None,
        "time": row_ts,
        "entry_time": row_ts,
    }


def finalize_strategy_runtime(
    runtime: StrategyRuntime,
    *,
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    if runtime.log_buffer and runtime.enable_logs:
        save_log_fn(runtime.log_buffer)

    if runtime.live:
        save_state_fn(runtime.state)
    elif runtime.use_state:
        runtime.state["position"] = runtime.position
        runtime.state["balance"] = runtime.balance
        runtime.state["trade_history"] = runtime.trade_history
        runtime.state["balance_history"] = runtime.balance_history
        save_state_fn(runtime.state)
        runtime.state["trade_history"] = runtime.trade_history
        runtime.state["balance_history"] = runtime.balance_history

    return (
        runtime.balance,
        pd.DataFrame(runtime.state.get("trade_history", [])),
        runtime.state.get("balance_history", []),
        runtime.state,
    )


def run_discrete_engine(
    df: pd.DataFrame,
    *,
    runtime: StrategyRuntime,
    show_progress: bool,
    timestamp_format: str,
    timestamp_formatter: Callable[[Any], str],
    on_row: Callable[[DiscreteRowSnapshot, StrategyRuntime], None],
    save_log_fn: Callable[[list[str]], None],
    save_state_fn: Callable[[dict[str, Any]], None],
) -> tuple[float, pd.DataFrame, list[float], dict[str, Any]]:
    for row in iter_discrete_rows(df, live=runtime.live, show_progress=show_progress):
        snapshot = build_row_snapshot(
            row,
            live=runtime.live,
            timestamp_format=timestamp_format,
            timestamp_formatter=timestamp_formatter,
        )
        on_row(snapshot, runtime)
        runtime.balance_history.append(runtime.balance)

    return finalize_strategy_runtime(
        runtime,
        save_log_fn=save_log_fn,
        save_state_fn=save_state_fn,
    )
