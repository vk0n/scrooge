# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from trade import compute_qty, can_open_trade, open_position, close_position, get_balance
from state import (
    load_state,
    save_state,
    update_position,
    update_balance,
    add_closed_trade,
)


LOG_FILE = "trading_log.txt"

def save_log(log_buffer):
    """Log message to file with timestamp (no print to avoid clutter)."""
    with open(LOG_FILE, "a") as f:
        f.write("\n".join(log_buffer) + "\n")


def run_strategy(df, live=False, initial_balance=1000,
                 qty=None, sl_mult = 1.5, tp_mult = 3.0,
                 symbol="BTCUSDT", leverage=1, use_full_balance=True, fee_rate=0.0005,
                 state=None, use_state=True, enable_logs=True, show_progress=True,
                 rsi_extreme_long=75, rsi_extreme_short=25,
                 rsi_long_open_threshold=50, rsi_long_qty_threshold=30, rsi_long_tp_threshold=58, rsi_long_close_threshold=70,
                 rsi_short_open_threshold=50, rsi_short_qty_threshold=70, rsi_short_tp_threshold=42, rsi_short_close_threshold=30,
                 trail_atr_mult=0.5):
    """
    Bollinger Bands strategy with SL/TP, dynamic stop, state persistence, and logging.
    """
    if use_state:
        state = load_state()
    else:
        state = {
            "position": None,
            "trade_history": [],
            "balance_history": [],
            "session_start": None,
            "session_end": None
        }
    balance = initial_balance
    position = state["position"]
    trade_history = state.get("trade_history", [])
    balance_history = state.get("balance_history", [])
    log_buffer = []

    if live:
        df_iter = [df.iloc[-1]]
        iterator = df_iter
    else:
        df_iter = [df.iloc[i] for i in range(1, len(df))]
        iterator = tqdm(df_iter, desc="Backtest Progress", disable=not show_progress)

    for row in iterator:
        price = row["close"]
        lower = row["BBL"]
        upper = row["BBU"]
        mid   = row["BBM"]
        atr   = row["ATR"]
        rsi   = row["RSI"]
        ema   = row["EMA"]
        
        if position is None:
            # determine position size
            qty_local = compute_qty(symbol, balance, leverage, price, qty, use_full_balance, live)
            
            # --- Helper: liquidation price estimation (isolated margin logic) ---
            def calc_liquidation_price(entry, leverage, side):
                # Binance Futures isolated margin formula approximation
                if side == "long":
                    return entry * (1 - 1 / leverage)
                else:
                    return entry * (1 + 1 / leverage)

            # OPEN LOGIC
            if price < lower and rsi < rsi_long_open_threshold and price > ema:
                qty_open = qty_local * 0.5 if rsi > rsi_long_qty_threshold else qty_local
                fb = False if rsi > rsi_long_qty_threshold else True 
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult
                liquidation_price = calc_liquidation_price(price, leverage, "long")

                # skip if SL is beyond liquidation
                if sl < liquidation_price:
                    if live:
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ⚠️ Skipped LONG entry at {price}: SL ({sl:.1f}) below liquidation ({liquidation_price:.1f})")
                    else:
                        log_buffer.append(f"[timestamp] ⚠️ Skipped LONG entry at {price}: SL ({sl:.1f}) below liquidation ({liquidation_price:.1f})")
                else:
                    position = {
                        "side": "long",
                        "size": qty_open,
                        "entry": price,
                        "sl": sl,
                        "tp": tp,
                        "liq_price": liquidation_price,
                        "trail_active": False,
                        "time": row["open_time"]
                    }

                    if live:
                        if can_open_trade(symbol, qty_open, leverage):
                            open_position(symbol, "BUY", qty_open, sl, tp, leverage)
                            balance = get_balance()
                            update_position(state, position)
                            update_balance(state, balance)
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Opened LONG {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate:.2f}")
                    else:
                        balance -= qty_open * price * fee_rate
                        log_buffer.append(f"[timestamp] Opened LONG {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate:.2f}, full_ballance: {fb}")

            elif price > upper and rsi > rsi_short_open_threshold and price < ema:
                qty_open = qty_local * 0.5 if rsi < rsi_short_qty_threshold else qty_local
                fb = False if rsi < rsi_short_qty_threshold else True 
                sl = price + atr * sl_mult
                tp = price - atr * tp_mult
                liquidation_price = calc_liquidation_price(price, leverage, "short")

                # skip if SL is beyond liquidation
                if sl > liquidation_price:
                    if live:
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ⚠️ Skipped SHORT entry at {price}: SL ({sl:.1f}) above liquidation ({liquidation_price:.1f})")
                    else:
                        log_buffer.append(f"[timestamp] ⚠️ Skipped SHORT entry at {price}: SL ({sl:.1f}) above liquidation ({liquidation_price:.1f})")
                else:
                    position = {
                        "side": "short",
                        "size": qty_open,
                        "entry": price,
                        "sl": sl,
                        "tp": tp,
                        "liq_price": liquidation_price,
                        "trail_active": False,
                        "time": row["open_time"]
                    }
                    
                    if live:
                        if can_open_trade(symbol, qty_open, leverage):
                            open_position(symbol, "SELL", qty_open, sl, tp, leverage)
                            balance = get_balance()
                            update_position(state, position)
                            update_balance(state, balance)
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Opened SHORT {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate:.2f}")
                    else:
                        balance -= qty_open * price * fee_rate
                        log_buffer.append(f"[timestamp] Opened SHORT {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate:.2f}, full_ballance: {fb}")

        else:
            # MANAGEMENT for open position
            size = position["size"]
            entry_price = position["entry"]
            position_value = size * entry_price
            fee_close = position_value * fee_rate
            side = position["side"]
            base_sl = position["sl"]
            base_tp = position["tp"]
            liquidation_price = position["liq_price"]
            trade = {
                **position,
                "exit": price,
            }

            # --- Liquidation Check ---
            if side == "long" and price < liquidation_price:
                if live:
                    current_balance = get_balance()
                    loss = current_balance - balance
                    trade["net_pnl"] = loss
                    trade["exit_reason"] = "liquidation"
                    balance = current_balance
                    update_position(state, None)
                    update_balance(state, balance)
                    add_closed_trade(state, trade)
                    log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] LONG LIQUIDATED at {price}, liq={liquidation_price:.1f}, loss={loss:.2f}")
                else:
                    # Margin used for this trade (portion of balance at risk)
                    margin_used = position_value / leverage
                    # Deduct the loss from account balance
                    balance -= margin_used
                    trade["net_pnl"] = -margin_used
                    trade["exit_reason"] = "liquidation"
                    trade_history.append(trade) # Long LIQUIDATED
                    log_buffer.append(f"[timestamp] LONG LIQUIDATED at {price}, liq={liquidation_price:.1f}, loss=-{margin_used}")
                
                # Close position and continue
                position = None
                continue

            if side == "short" and price > liquidation_price:
                if live:
                    current_balance = get_balance()
                    loss = current_balance - balance
                    trade["net_pnl"] = loss
                    trade["exit_reason"] = "liquidation"
                    balance = current_balance
                    update_position(state, None)
                    update_balance(state, balance)
                    add_closed_trade(state, trade)
                    log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] SHORT LIQUIDATED at {price}, liq={liquidation_price:.1f}, loss={loss:.2f}")
                else:
                    # Margin used for this trade
                    margin_used = position_value / leverage
                    # Deduct the loss from account balance
                    balance -= margin_used
                    trade["net_pnl"] = -margin_used
                    trade["exit_reason"] = "liquidation"
                    trade_history.append(trade) # Short LIQUIDATED
                    log_buffer.append(f"[timestamp] SHORT LIQUIDATED at {price}, liq={liquidation_price:.1f}, loss=-{margin_used}")

                # Close position and continue
                position = None
                continue

            # --- LONG SIDE LOGIC ---
            if side == "long":
                # Emergency RSI exit (extreme overbought)
                gross_pnl = (price - entry_price) / entry_price * position_value
                if rsi > rsi_extreme_long:
                    fee_total = fee_close * 2
                    net_pnl = gross_pnl - fee_total
                    trade = {
                        **position,
                        "exit": price,
                        "gross_pnl": gross_pnl,
                        "fee": fee_total,
                        "net_pnl": net_pnl,
                        "exit_reason": "rsi_extreme"
                    }

                    if live:
                        close_position(symbol)
                        current_balance = get_balance()
                        trade["net_pnl"] = current_balance - balance
                        balance = current_balance
                        update_position(state, None)
                        update_balance(state, balance)
                        add_closed_trade(state, trade)
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} due to RSI>{rsi_extreme_long} (extreme overbought), net={net_pnl:.2f}")
                    else:
                        balance += net_pnl
                        trade_history.append(trade) # Long RSI
                        log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} due to RSI>{rsi_extreme_long} (extreme overbought), net={net_pnl:.2f}")
                    
                    position = None

                elif position is not None:
                    # Activate trailing TP once base TP reached (but RSI not overbought yet)
                    if not position["trail_active"]:
                        if price > base_tp and rsi < rsi_long_tp_threshold:
                            position["trail_active"] = True
                            position["trail_max"] = price
                            if live:
                                update_position(state, position)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Activated trailing TP for LONG {symbol} at {price}")
                            else:
                                log_buffer.append(f"[timestamp] Activated trailing TP for LONG {symbol} at {price}")
                    else:
                        # Update or close trailing TP
                        if price > position["trail_max"]:
                            position["trail_max"] = price
                            current_trail_tp = position["trail_max"] - atr * trail_atr_mult
                            position["tp"] = current_trail_tp
                            if live:
                                update_position(state, position)
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Trailing stop moved up to {current_trail_tp:.1f}")
                            else:
                                log_buffer.append(f"[timestamp] Trailing stop moved up to {current_trail_tp:.1f}")
                        elif price < position["trail_max"] - atr * trail_atr_mult or rsi > rsi_long_close_threshold:
                            fee_total = fee_close * 2
                            net_pnl = gross_pnl - fee_total
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "take_profit"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} by trailing TP, net={net_pnl:.2f}")
                            else:
                                balance += net_pnl
                                trade_history.append(trade) # Long TP
                                log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} by trailing TP, net={net_pnl:.2f}")
                            
                            position = None

                    # Fallback to normal SL/TP logic if no trail active
                    if position is not None:
                        if price < base_sl:
                            fee_total = fee_close * 2
                            net_pnl = gross_pnl - fee_total
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "stop_loss"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} by SL, net={net_pnl:.2f}")
                            else:
                                balance += net_pnl
                                trade_history.append(trade) # Long SL
                                log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} by SL, net={net_pnl:.2f}")
                            
                            position = None

            # --- SHORT SIDE LOGIC ---
            elif side == "short":
                # Emergency RSI exit (extreme oversold)
                gross_pnl = (entry_price - price) / entry_price * position_value
                if rsi < rsi_extreme_short:
                    fee_total = fee_close * 2
                    net_pnl = gross_pnl - fee_total
                    trade = {
                        **position,
                        "exit": price,
                        "gross_pnl": gross_pnl,
                        "fee": fee_total,
                        "net_pnl": net_pnl,
                        "exit_reason": "rsi_extreme"
                    }

                    if live:
                        close_position(symbol)
                        current_balance = get_balance()
                        trade["net_pnl"] = current_balance - balance
                        balance = current_balance
                        update_position(state, None)
                        update_balance(state, balance)
                        add_closed_trade(state, trade)
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} due to RSI<{rsi_extreme_short} (extreme oversold), net={net_pnl:.2f}")
                    else:
                        balance += net_pnl
                        trade_history.append(trade) # Short RSI
                        log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} due to RSI<{rsi_extreme_short} (extreme oversold), net={net_pnl:.2f}")
                    
                    position = None

                elif position is not None:
                    # Activate trailing TP once base TP reached (but RSI not oversold yet)
                    if not position["trail_active"]:
                        if price < base_tp and rsi > rsi_short_tp_threshold:
                            position["trail_active"] = True
                            position["trail_min"] = price
                            if live:
                                update_position(state, position)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Activated trailing TP for SHORT {symbol} at {price}")
                            else:
                                log_buffer.append(f"[timestamp] Activated trailing TP for SHORT {symbol} at {price}")
                    else:
                        # Update or close trailing TP
                        if price < position["trail_min"]:
                            position["trail_min"] = price
                            current_trail_tp = position["trail_min"] + atr * trail_atr_mult
                            position["tp"] = current_trail_tp
                            if live:
                                update_position(state, position)
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Trailing stop moved down to {current_trail_tp:.1f}")
                            else:
                                log_buffer.append(f"[timestamp] Trailing stop moved down to {current_trail_tp:.1f}")
                        elif price > position["trail_min"] + atr * trail_atr_mult or rsi < rsi_short_close_threshold:
                            fee_total = fee_close * 2
                            net_pnl = gross_pnl - fee_total
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "take_profit"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} by trailing TP, net={net_pnl:.2f}")
                            else:
                                balance += net_pnl
                                trade_history.append(trade) # Short TP
                                log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} by trailing TP, net={net_pnl:.2f}")
                            
                            position = None

                    # Fallback to normal SL/TP logic if no trail active
                    if position is not None:
                        if price > base_sl:
                            fee_total = fee_close * 2
                            net_pnl = gross_pnl - fee_total
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "stop_loss"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} by SL, net={net_pnl:.2f}")
                            else:
                                balance += net_pnl
                                trade_history.append(trade) # Short SL
                                log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} by SL, net={net_pnl:.2f}")
                            
                            position = None

            if live and position is not None:
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Unrealized PnL: {gross_pnl:.2f}")

        if live:
            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Price: {price} | BBL: {lower:.1f} | BBM: {mid:.1f} | BBU: {upper:.1f} | RSI: {rsi:.1f} | EMA: {ema:.1f}")

        balance_history.append(balance)
    
    if log_buffer and enable_logs:
        save_log(log_buffer)
    
    if live:
        save_state(state)
    elif use_state:
        state["position"] = position
        state["balance"] = balance
        state["balance_history"] = balance_history
        save_state(state)

    return balance, pd.DataFrame(state.get("trade_history", [])), state.get("balance_history", []), state
