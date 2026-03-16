# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import os
import pandas as pd
from datetime import datetime
from core_engine import (
    DiscreteRowSnapshot,
    EntryDecision,
    EntryGuardRejection,
    ExitDecision,
    StrategyRuntime,
    StrategyConfig,
    TrailDecision,
    build_position_metrics,
    build_position_from_entry,
    initialize_strategy_runtime,
    resolve_management_decision,
    resolve_entry_decision,
    run_discrete_engine,
)
from event_log import emit_event, get_technical_logger
from trade import compute_qty, can_open_trade, open_position, close_position, get_balance
from state import (
    load_state,
    save_state,
    update_position,
    update_balance,
    add_closed_trade,
)


LOG_FILE = os.getenv("SCROOGE_LOG_FILE", "trading_log.txt")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SEARCH_STATUS_LABELS = {
    "looking_for_buy_opportunity": "Looking for a buy opportunity...",
    "looking_for_sell_opportunity": "Looking for a sell opportunity...",
}

def save_log(log_buffer):
    """Log message to file with timestamp (no print to avoid clutter)."""
    with open(LOG_FILE, "a") as f:
        f.write("\n".join(log_buffer) + "\n")


def _format_event_timestamp(value) -> str:
    """Normalize candle/open_time values to a stable timestamp string."""
    if isinstance(value, pd.Timestamp):
        dt_value = value.tz_convert(None) if value.tzinfo is not None else value
        return dt_value.strftime(TIMESTAMP_FORMAT)

    if isinstance(value, datetime):
        dt_value = value.replace(tzinfo=None) if value.tzinfo is not None else value
        return dt_value.strftime(TIMESTAMP_FORMAT)

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 10_000_000_000:
            numeric_value /= 1000.0
        try:
            return datetime.fromtimestamp(numeric_value).strftime(TIMESTAMP_FORMAT)
        except (OSError, OverflowError, ValueError):
            return datetime.now().strftime(TIMESTAMP_FORMAT)

    text_value = str(value).strip()
    return text_value or datetime.now().strftime(TIMESTAMP_FORMAT)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_to_percent(numerator, denominator):
    if denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * 100.0


def _status_from_code(code, labels):
    label = labels.get(code)
    if label is None:
        return None
    return {"code": code, "label": label}


def _resolve_search_status(price, ema, previous_status):
    price_value = _to_float(price)
    ema_value = _to_float(ema)
    previous_code = None
    if isinstance(previous_status, dict):
        raw_code = str(previous_status.get("code", "")).strip()
        if raw_code in SEARCH_STATUS_LABELS:
            previous_code = raw_code

    if price_value is None or ema_value is None:
        return _status_from_code(previous_code, SEARCH_STATUS_LABELS) if previous_code else None

    if price_value > ema_value:
        return _status_from_code("looking_for_buy_opportunity", SEARCH_STATUS_LABELS)
    if price_value < ema_value:
        return _status_from_code("looking_for_sell_opportunity", SEARCH_STATUS_LABELS)
    if previous_code:
        return _status_from_code(previous_code, SEARCH_STATUS_LABELS)
    return _status_from_code("looking_for_buy_opportunity", SEARCH_STATUS_LABELS)


def _normalize_manual_trade_suggestion(value):
    if not isinstance(value, dict):
        return None

    side = str(value.get("side", "")).strip().lower()
    if side not in {"buy", "sell"}:
        return None

    return {
        "side": side,
        "requested_at": str(value.get("requested_at", "")).strip() or None,
        "requested_by": str(value.get("requested_by", "")).strip() or None,
    }


def _refresh_position_snapshot(position, price, leverage, ts_label):
    # Keep ticker price at state level (not per-position).
    position.pop("last_price", None)
    position.pop("last_price_updated_at", None)

    # Keep stable naming in runtime state for open-position timestamps.
    if position.get("entry_time") is None and position.get("time") is not None:
        position["entry_time"] = position.get("time")
    if position.get("time") is None and position.get("entry_time") is not None:
        position["time"] = position.get("entry_time")

    side = str(position.get("side", "")).strip().lower()
    size = _to_float(position.get("size")) or 0.0
    entry_price = _to_float(position.get("entry"))
    mark_price = _to_float(price)
    sl_price = _to_float(position.get("sl"))
    tp_price = _to_float(position.get("tp"))

    if mark_price is None:
        return

    position["updated_at"] = ts_label
    position["unrealized_pnl"] = None
    position["unrealized_pnl_pct"] = None
    position["position_notional"] = None
    position["margin_used"] = None
    position["roi_pct"] = None
    position["distance_to_sl_pct"] = None
    position["distance_to_tp_pct"] = None

    if entry_price is None or entry_price <= 0 or size <= 0:
        return

    notional = abs(size) * entry_price
    if side == "short":
        unrealized_pnl = (entry_price - mark_price) * size
    else:
        unrealized_pnl = (mark_price - entry_price) * size

    unrealized_pnl_pct = _ratio_to_percent(unrealized_pnl, notional)
    leverage_value = _to_float(leverage)
    margin_used = (notional / leverage_value) if leverage_value and leverage_value > 0 else None
    roi_pct = _ratio_to_percent(unrealized_pnl, margin_used) if margin_used else None

    if side == "short":
        distance_to_sl_pct = _ratio_to_percent((sl_price - mark_price), mark_price) if sl_price is not None else None
        distance_to_tp_pct = _ratio_to_percent((mark_price - tp_price), mark_price) if tp_price is not None else None
    else:
        distance_to_sl_pct = _ratio_to_percent((mark_price - sl_price), mark_price) if sl_price is not None else None
        distance_to_tp_pct = _ratio_to_percent((tp_price - mark_price), mark_price) if tp_price is not None else None

    position["unrealized_pnl"] = unrealized_pnl
    position["unrealized_pnl_pct"] = unrealized_pnl_pct
    position["position_notional"] = notional
    position["margin_used"] = margin_used
    position["roi_pct"] = roi_pct
    position["distance_to_sl_pct"] = distance_to_sl_pct
    position["distance_to_tp_pct"] = distance_to_tp_pct


_TRANSIENT_POSITION_FIELDS = {
    "last_price",
    "last_price_updated_at",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "position_notional",
    "margin_used",
    "roi_pct",
    "distance_to_sl_pct",
    "distance_to_tp_pct",
    "updated_at",
}


def _refresh_market_snapshot(state, price, ts_label):
    market_price = _to_float(price)
    if market_price is None:
        return
    state["last_price"] = market_price
    state["last_price_updated_at"] = ts_label
    state["updated_at"] = ts_label


def refresh_runtime_state_from_price_tick(state, last_price, position_price, leverage, ts_label):
    """
    Apply live websocket price updates to runtime state without changing strategy cadence.
    - `last_price` drives the top-level ticker price shown in UI.
    - `position_price` can use mark price for open-position metrics when available.
    """
    _refresh_market_snapshot(state, last_price, ts_label)

    position = state.get("position")
    if isinstance(position, dict):
        snapshot_price = _to_float(position_price)
        if snapshot_price is not None:
            _refresh_position_snapshot(position, snapshot_price, leverage, ts_label)


def _sanitize_trade_for_history(trade):
    sanitized = {}
    for key, value in trade.items():
        if key in _TRANSIENT_POSITION_FIELDS:
            continue
        sanitized[key] = value
    return sanitized

def _consume_manual_trade_suggestion(state, live):
    suggestion = _normalize_manual_trade_suggestion(state.get("manual_trade_suggestion"))
    if suggestion is None:
        return None
    state["manual_trade_suggestion"] = None
    if live:
        save_state(state)
    return suggestion


def _emit_exit_event(log_buffer, config, log_ts, decision, net_pnl):
    event_kwargs = {
        "symbol": config.symbol,
        "side": decision.side,
        "exit": decision.exit,
        "net_pnl": net_pnl,
    }
    if decision.margin_used:
        event_kwargs["roi_pct"] = _ratio_to_percent(net_pnl, decision.margin_used)
    if decision.via_tail_guard:
        event_kwargs["via_tail_guard"] = True
    if decision.liq_price is not None:
        event_kwargs["liq_price"] = decision.liq_price
    if decision.rsi is not None:
        event_kwargs["rsi"] = decision.rsi
    if decision.threshold is not None:
        event_kwargs["threshold"] = decision.threshold

    emit_event(
        code=decision.event_code,
        category=decision.category,
        ts=log_ts,
        level=decision.level,
        notify=True,
        log_buffer=log_buffer,
        **event_kwargs,
    )


def _apply_trail_decision(position, state, live, log_buffer, config, log_ts, decision):
    position.update(decision.position_updates)
    if live:
        update_position(state, position)
    emit_event(
        code=decision.event_code,
        category="trade",
        ts=log_ts,
        level=decision.level,
        log_buffer=log_buffer,
        symbol=config.symbol,
        side=decision.side,
        **decision.event_context,
    )


def _apply_exit_decision(
    *,
    decision,
    position,
    state,
    live,
    balance,
    trade_history,
    log_buffer,
    config,
    row_ts,
    log_ts,
):
    trade = {
        **position,
        "exit": decision.exit,
        "exit_time": row_ts,
        "exit_reason": decision.reason,
    }
    if decision.gross_pnl is not None:
        trade["gross_pnl"] = decision.gross_pnl
    if decision.fee_total is not None:
        trade["fee"] = decision.fee_total

    if live:
        if decision.reason != "liquidation":
            close_position(config.symbol)
        current_balance = get_balance()
        trade["net_pnl"] = current_balance - balance
        balance = current_balance
        update_position(state, None)
        update_balance(state, balance)
        add_closed_trade(state, _sanitize_trade_for_history(trade))
        _emit_exit_event(log_buffer, config, log_ts, decision, trade["net_pnl"])
    else:
        trade["net_pnl"] = decision.net_pnl
        balance += decision.net_pnl
        trade_history.append(_sanitize_trade_for_history(trade))
        _emit_exit_event(log_buffer, config, log_ts, decision, decision.net_pnl)

    return balance, None


def _process_discrete_row(
    snapshot: DiscreteRowSnapshot,
    runtime: StrategyRuntime,
    config: StrategyConfig,
    technical_logger,
):
    state = runtime.state
    balance = runtime.balance
    position = runtime.position
    trade_history = runtime.trade_history
    log_buffer = runtime.log_buffer
    live = runtime.live

    price = snapshot.price
    lower = snapshot.lower
    upper = snapshot.upper
    mid = snapshot.mid
    atr = snapshot.atr
    rsi = snapshot.rsi
    ema = snapshot.ema
    row_ts = snapshot.row_ts
    log_ts = snapshot.log_ts

    if live:
        _refresh_market_snapshot(state, price, row_ts)
    state["search_status"] = _resolve_search_status(price, ema, state.get("search_status"))

    if position is None:
        if not config.allow_entries:
            runtime.balance = balance
            runtime.position = position
            return

        manual_trade_suggestion = _consume_manual_trade_suggestion(state, live) if live else None
        manual_side = manual_trade_suggestion["side"] if manual_trade_suggestion else None

        qty_local = compute_qty(
            config.symbol,
            balance,
            config.leverage,
            price,
            config.qty,
            config.use_full_balance,
            live,
        )
        entry_decision = resolve_entry_decision(
            snapshot,
            config=config,
            qty_local=qty_local,
            manual_side=manual_side,
        )

        if isinstance(entry_decision, EntryGuardRejection):
            emit_event(
                code="entry_skipped_liquidation_guard",
                category="risk",
                ts=log_ts,
                level="warning",
                log_buffer=log_buffer,
                symbol=config.symbol,
                side=entry_decision.side,
                entry=entry_decision.entry,
                sl=entry_decision.sl,
                liq_price=entry_decision.liq_price,
                trigger=entry_decision.trigger,
            )
        elif isinstance(entry_decision, EntryDecision):
            position = build_position_from_entry(entry_decision, row_ts=row_ts)
            _refresh_position_snapshot(position, price, config.leverage, row_ts)

            if live:
                if can_open_trade(config.symbol, entry_decision.size, config.leverage):
                    side_command = "BUY" if entry_decision.side == "long" else "SELL"
                    open_position(
                        config.symbol,
                        side_command,
                        entry_decision.size,
                        entry_decision.sl,
                        entry_decision.tp,
                        config.leverage,
                    )
                    balance = get_balance()
                    update_position(state, position)
                    update_balance(state, balance)
                emit_event(
                    code="trade_opened",
                    category="trade",
                    ts=log_ts,
                    level="info",
                    notify=True,
                    log_buffer=log_buffer,
                    symbol=config.symbol,
                    side=entry_decision.side,
                    entry=entry_decision.entry,
                    sl=entry_decision.sl,
                    tp=entry_decision.tp,
                    size=entry_decision.size,
                    leverage=config.leverage,
                    fee=entry_decision.size * price * config.fee_rate,
                    rsi=rsi,
                    stake_mode=entry_decision.stake_mode,
                    trigger=entry_decision.trigger,
                )
            else:
                balance -= entry_decision.size * price * config.fee_rate
                emit_event(
                    code="trade_opened",
                    category="trade",
                    ts=log_ts,
                    level="info",
                    notify=True,
                    log_buffer=log_buffer,
                    symbol=config.symbol,
                    side=entry_decision.side,
                    entry=entry_decision.entry,
                    sl=entry_decision.sl,
                    tp=entry_decision.tp,
                    size=entry_decision.size,
                    leverage=config.leverage,
                    fee=entry_decision.size * price * config.fee_rate,
                    rsi=rsi,
                    stake_mode=entry_decision.stake_mode,
                    trigger=entry_decision.trigger,
                )

    else:
        side = position["side"]
        _refresh_position_snapshot(position, price, config.leverage, row_ts)
        if live:
            update_position(state, position)
        metrics = build_position_metrics(
            position,
            snapshot,
            leverage=config.leverage,
            fee_rate=config.fee_rate,
        )
        management_decision = resolve_management_decision(
            position,
            snapshot,
            config=config,
            metrics=metrics,
        )

        if isinstance(management_decision, TrailDecision):
            _apply_trail_decision(
                position,
                state,
                live,
                log_buffer,
                config,
                log_ts,
                management_decision,
            )
        elif isinstance(management_decision, ExitDecision):
            balance, position = _apply_exit_decision(
                decision=management_decision,
                position=position,
                state=state,
                live=live,
                balance=balance,
                trade_history=trade_history,
                log_buffer=log_buffer,
                config=config,
                row_ts=row_ts,
                log_ts=log_ts,
            )
            runtime.balance = balance
            runtime.position = position
            return

        if live and position is not None:
            technical_logger.debug(
                "runtime_open_position_pnl side=%s symbol=%s gross_pnl=%.2f",
                side,
                config.symbol,
                metrics.gross_pnl,
            )

    if live:
        technical_logger.debug(
            "runtime_indicator_snapshot symbol=%s price=%.2f bbl=%.2f bbm=%.2f bbu=%.2f rsi=%.2f ema=%.2f",
            config.symbol,
            price,
            lower,
            mid,
            upper,
            rsi,
            ema,
        )

    runtime.balance = balance
    runtime.position = position


def run_strategy(df, live=False, initial_balance=1000,
                 qty=None, sl_mult = 1.5, tp_mult = 3.0,
                 symbol="BTCUSDT", leverage=1, use_full_balance=True, fee_rate=0.0005,
                 state=None, use_state=True, enable_logs=True, show_progress=True,
                 rsi_extreme_long=75, rsi_extreme_short=25,
                 rsi_long_open_threshold=50, rsi_long_qty_threshold=30, rsi_long_tp_threshold=58, rsi_long_close_threshold=70,
                 rsi_short_open_threshold=50, rsi_short_qty_threshold=70, rsi_short_tp_threshold=42, rsi_short_close_threshold=30,
                 trail_atr_mult=0.5, allow_entries=True):
    """
    Bollinger Bands strategy with SL/TP, dynamic stop, state persistence, and logging.
    """
    runtime = initialize_strategy_runtime(
        live=live,
        initial_balance=initial_balance,
        use_state=use_state,
        load_state_fn=load_state,
    )
    runtime.enable_logs = enable_logs
    technical_logger = get_technical_logger()
    config = StrategyConfig(
        qty=qty,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        symbol=symbol,
        leverage=leverage,
        use_full_balance=use_full_balance,
        fee_rate=fee_rate,
        rsi_extreme_long=rsi_extreme_long,
        rsi_extreme_short=rsi_extreme_short,
        rsi_long_open_threshold=rsi_long_open_threshold,
        rsi_long_qty_threshold=rsi_long_qty_threshold,
        rsi_long_tp_threshold=rsi_long_tp_threshold,
        rsi_long_close_threshold=rsi_long_close_threshold,
        rsi_short_open_threshold=rsi_short_open_threshold,
        rsi_short_qty_threshold=rsi_short_qty_threshold,
        rsi_short_tp_threshold=rsi_short_tp_threshold,
        rsi_short_close_threshold=rsi_short_close_threshold,
        trail_atr_mult=trail_atr_mult,
        allow_entries=allow_entries,
    )

    def on_row(snapshot, row_runtime):
        _process_discrete_row(snapshot, row_runtime, config, technical_logger)

    return run_discrete_engine(
        df,
        runtime=runtime,
        show_progress=show_progress,
        timestamp_format=TIMESTAMP_FORMAT,
        timestamp_formatter=_format_event_timestamp,
        on_row=on_row,
        save_log_fn=save_log,
        save_state_fn=save_state,
    )
