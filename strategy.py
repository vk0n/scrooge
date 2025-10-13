# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import os
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from tqdm import tqdm
from binance.client import Client
import tempfile
import webbrowser
from datetime import datetime, timedelta
from trade import *
from state import *
from data import *


load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)

LOG_FILE = "trading_log.txt"

def save_log(log_buffer):
    """Log message to file with timestamp (no print to avoid clutter)."""
    with open(LOG_FILE, "a") as f:
        f.write("\n".join(log_buffer) + "\n")


def run_strategy(df, live=False, initial_balance=1000,
                 qty=None, sl_mult = 1.5, tp_mult = 3.0,
                 symbol="BTCUSDT", leverage=1, use_full_balance=True, fee_rate=0.0004,
                 state=None, use_state=True, enable_logs=True,
                 rsi_extreme_long=75, rsi_extreme_short=25,
                 rsi_long_open_threshold=50, rsi_long_qty_threshold=30, rsi_long_close_threshold=70,
                 rsi_short_open_threshold=50, rsi_short_qty_threshold=70, rsi_short_close_threshold=30,
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
        iterator = tqdm(df_iter, desc="Backtest Progress")

    for row in iterator:
        price = row["close"]
        lower = row["BBL"]
        upper = row["BBU"]
        mid   = row["BBM"]
        atr   = row["ATR"]
        rsi   = row["RSI"]
        ema   = row["EMA"]

        if live:
            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Price: {price} | BBL: {lower} | BBM: {mid} | BBU: {upper} | RSI: {rsi} | EMA: {ema}")
        
        if position is None:
            # determine position size
            qty_local = compute_qty(symbol, balance, leverage, price, qty, use_full_balance, live)
            # OPEN LOGIC
            if price <= lower and rsi < rsi_long_open_threshold and price > ema:
                qty_open = qty_local * 0.5 if rsi > rsi_long_qty_threshold else qty_local
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult
                position = {
                    "side": "long",
                    "size": qty_open,
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "dyn_sl": None,
                    "trail_active": False,
                    "time": row["open_time"]
                }

                if live:
                    if can_open_trade(symbol, qty_open, leverage):
                        open_position(symbol, "BUY", qty_open, sl, tp, leverage)
                        balance = get_balance()
                        update_position(state, position)
                        update_balance(state, balance)
                    log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Opened LONG {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate}")
                else:
                    balance -= qty_open * price * fee_rate
                    log_buffer.append(f"[timestamp] Opened LONG {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate}")

            elif price >= upper and rsi > rsi_short_open_threshold and price < ema:
                qty_open = qty_local * 0.5 if rsi < rsi_short_qty_threshold else qty_local
                sl = price + atr * sl_mult
                tp = price - atr * tp_mult
                position = {
                    "side": "short",
                    "size": qty_open,
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "dyn_sl": None,
                    "trail_active": False,
                    "time": row["open_time"]
                }
                
                if live:
                    if can_open_trade(symbol, qty_open, leverage):
                        open_position(symbol, "SELL", qty_open, sl, tp, leverage)
                        balance = get_balance()
                        update_position(state, position)
                        update_balance(state, balance)
                    log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Opened SHORT {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate}")
                else:
                    balance -= qty_open * price * fee_rate
                    log_buffer.append(f"[timestamp] Opened SHORT {qty_open} {symbol} at {price}, fee={qty_open * price * fee_rate}")

        else:
            # MANAGEMENT for open position
            size = position["size"]
            entry_price = position["entry"]
            position_value = size * entry_price
            fee_close = position_value * fee_rate
            side = position["side"]
            base_sl = position["sl"]
            base_tp = position["tp"]

            # --- LONG SIDE LOGIC ---
            if side == "long":
                # Emergency RSI exit (extreme overbought)
                gross_pnl = (price - entry_price) / entry_price * position_value
                if rsi > rsi_extreme_long:
                    net_pnl = gross_pnl - fee_close
                    fee_total = fee_close * 2
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
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} due to RSI>{rsi_extreme_long} (extreme overbought), net={net_pnl}")
                    else:
                        balance += net_pnl
                        log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} due to RSI>{rsi_extreme_long} (extreme overbought), net={net_pnl}")
                    
                    trade_history.append(trade)
                    position = None

                elif position is not None:
                    # Activate trailing TP once base TP reached (but RSI not overbought yet)
                    if not position["trail_active"]:
                        if price >= base_tp and rsi < rsi_long_close_threshold:
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
                        elif price <= position["trail_max"] - atr * trail_atr_mult or rsi >= rsi_long_close_threshold:
                            net_pnl = gross_pnl - fee_close
                            fee_total = fee_close * 2
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "trailing_tp"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} by trailing TP, net={net_pnl}")
                            else:
                                balance += net_pnl
                                log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} by trailing TP, net={net_pnl}")
                            
                            trade_history.append(trade)
                            position = None

                    # Fallback to normal SL/TP logic if no trail active
                    if position is not None:
                        effective_sl = position["dyn_sl"] if position["dyn_sl"] is not None else base_sl
                        if price <= effective_sl:
                            net_pnl = gross_pnl - fee_close
                            fee_total = fee_close * 2
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
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed LONG {size} {symbol} by SL, net={net_pnl}")
                            else:
                                balance += net_pnl
                                log_buffer.append(f"[timestamp] Closed LONG {size} {symbol} by SL, net={net_pnl}")
                            
                            trade_history.append(trade)
                            position = None

            # --- SHORT SIDE LOGIC ---
            elif side == "short":
                # Emergency RSI exit (extreme oversold)
                gross_pnl = (entry_price - price) / entry_price * position_value
                if rsi < rsi_extreme_short:
                    net_pnl = gross_pnl - fee_close
                    fee_total = fee_close * 2
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
                        log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} due to RSI<{rsi_extreme_short} (extreme oversold), net={net_pnl}")
                    else:
                        balance += net_pnl
                        log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} due to RSI<{rsi_extreme_short} (extreme oversold), net={net_pnl}")
                    
                    trade_history.append(trade)
                    position = None

                elif position is not None:
                    # Activate trailing TP once base TP reached (but RSI not oversold yet)
                    if not position["trail_active"]:
                        if price <= base_tp and rsi > rsi_short_close_threshold:
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
                        elif price >= position["trail_min"] + atr * trail_atr_mult or rsi <= rsi_short_close_threshold:
                            net_pnl = gross_pnl - fee_close
                            fee_total = fee_close * 2
                            trade = {
                                **position,
                                "exit": price,
                                "gross_pnl": gross_pnl,
                                "fee": fee_total,
                                "net_pnl": net_pnl,
                                "exit_reason": "trailing_tp"
                            }

                            if live:
                                close_position(symbol)
                                current_balance = get_balance()
                                trade["net_pnl"] = current_balance - balance
                                balance = current_balance
                                update_position(state, None)
                                update_balance(state, balance)
                                add_closed_trade(state, trade)
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} by trailing TP, net={net_pnl}")
                            else:
                                balance += net_pnl
                                log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} by trailing TP, net={net_pnl}")
                            
                            trade_history.append(trade)
                            position = None

                    # Fallback to normal SL/TP logic if no trail active
                    if position is not None:
                        effective_sl = position["dyn_sl"] if position["dyn_sl"] is not None else base_sl
                        if price >= effective_sl:
                            net_pnl = gross_pnl - fee_close
                            fee_total = fee_close * 2
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
                                log_buffer.append(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Closed SHORT {size} {symbol} by SL, net={net_pnl}")
                            else:
                                balance += net_pnl
                                log_buffer.append(f"[timestamp] Closed SHORT {size} {symbol} by SL, net={net_pnl}")
                            
                            trade_history.append(trade)
                            position = None

            if live and position is not None:
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Unrealized PnL: {gross_pnl}")

        balance_history.append(balance)
    
    if log_buffer and enable_logs:
        save_log(log_buffer)
    
    if live:
        save_state(state)
    elif use_state:
        state["position"] = position
        state["balance"] = balance
        state["trade_history"] = trade_history
        state["balance_history"] = balance_history
        save_state(state)

    return balance, pd.DataFrame(state.get("trade_history", [])), state.get("balance_history", []), state
