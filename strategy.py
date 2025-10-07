# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import os
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


load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)

LOG_FILE = "trading_log.txt"

def log_event(message):
    """Log message to file with timestamp (no print to avoid clutter)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def fetch_historical(symbol="BTCUSDT", interval="15m", limit=500):
    """Fetch historical klines from Binance Futures."""
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]


def fetch_historical_paginated(symbol="BTCUSDT", interval="1m", start_time=None, end_time=None):
    """
    Fetch historical klines from Binance Futures with pagination.
    Returns full DataFrame from start_time to end_time (or last available candle if end_time=None).
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime

    dfs = []
    limit = 1500  # max allowed by Binance
    start_ts = int(start_time.timestamp() * 1000) if start_time else None
    end_ts = int(end_time.timestamp() * 1000) if end_time else None

    while True:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=limit
        )
        if not klines:
            break

        df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume","close_time","qav",
            "num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close"] = pd.to_numeric(df["close"])
        dfs.append(df[["open_time", "close"]])

        last_ts = int(klines[-1][0]) + 1  # next millisecond after last candle
        if end_ts and last_ts >= end_ts:
            break
        if len(klines) < limit:
            break
        start_ts = last_ts

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    else:
        return pd.DataFrame(columns=["open_time", "close"])


def prepare_multi_tf(df_small, df_medium, df_big):
    """Prepare multi timeframe DataFrame:
    - df_small : 1m (price)
    - df_medium: 5m (Bollinger Bands)
    - df_big   : 1h (RSI)
    """
    # --- Convert time ---
    df_small["open_time"] = pd.to_datetime(df_small["open_time"], unit="ms")
    df_medium["open_time"] = pd.to_datetime(df_medium["open_time"], unit="ms")
    df_big["open_time"] = pd.to_datetime(df_big["open_time"], unit="ms")

    # --- RSI on 1h (big) ---
    df_big = df_big.set_index("open_time")
    df_big["RSI"] = ta.rsi(df_big["close"], length=6)
    df_big = df_big[["RSI"]]

    # --- Bollinger Bands on 5m (medium) ---
    df_medium = df_medium.set_index("open_time")
    bb = ta.bbands(df_medium["close"], length=20, std=2)
    df_medium["BBL"] = bb[f"BBL_20_2.0_2.0"]
    df_medium["BBM"] = bb[f"BBM_20_2.0_2.0"]
    df_medium["BBU"] = bb[f"BBU_20_2.0_2.0"]
    df_medium = df_medium[["BBL", "BBM", "BBU"]]

    # --- Merge into 1m (small) ---
    df_merged = df_small.set_index("open_time")

    # Merge medium -> small and forward-fill bands
    df_merged = df_merged.merge(df_medium, left_index=True, right_index=True, how="left")
    df_merged[["BBL", "BBM", "BBU"]] = df_merged[["BBL", "BBM", "BBU"]].ffill()

    # Merge big -> small and forward-fill RSI
    df_merged = df_merged.merge(df_big, left_index=True, right_index=True, how="left")
    df_merged["RSI"] = df_merged["RSI"].ffill()

    df_merged = df_merged.reset_index()

    return df_merged


def compute_stats(initial_balance, final_balance, trades, balance_history):
    """Compute basic performance metrics."""
    stats = {}

    # Basic
    stats["Initial Balance"] = initial_balance
    stats["Final Balance"] = final_balance
    stats["Total Return %"] = (final_balance - initial_balance) / initial_balance * 100

    # Trades
    n_trades = len(trades)
    stats["Number of Trades"] = n_trades

    if n_trades > 0:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]

        stats["Win Rate %"] = len(wins) / n_trades * 100
        stats["Average PnL"] = trades["net_pnl"].mean()
        stats["Average Profit"] = wins["net_pnl"].mean() if len(wins) > 0 else 0
        stats["Average Loss"] = losses["net_pnl"].mean() if len(losses) > 0 else 0
        stats["Best Trade"] = trades["net_pnl"].max()
        stats["Worst Trade"] = trades["net_pnl"].min()

        total_profit = wins["net_pnl"].sum()
        total_loss = abs(losses["net_pnl"].sum())
        stats["Profit Factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else np.inf
    else:
        stats["Win Rate %"] = 0
        stats["Average PnL"] = 0
        stats["Best Trade"] = 0
        stats["Worst Trade"] = 0
        stats["Profit Factor"] = 0

    # Max drawdown
    equity = np.array(balance_history)
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    stats["Max Drawdown %"] = drawdowns.min() * 100

    return stats


def run_strategy(df, initial_balance=1000, qty=None, sl_pct=0.005, tp_pct=0.01,
                 live=False, symbol="BTCUSDT", leverage=10, use_full_balance=True,
                 fee_rate=0.0004, dyn_sl_buffer=0.002, state=None):
    """
    Bollinger Bands strategy with SL/TP, dynamic stop, state persistence, and logging.
    """
    if state is None:
        state = load_state()
    balance = initial_balance
    position = state["position"]
    trade_history = state.get("trade_history", [])
    balance_history = state.get("balance_history", [])

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
        rsi   = row["RSI"]

        if live:
            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Price: {price} | BBL: {lower} | BBM: {mid} | BBU: {upper} | RSI: {rsi}")
        
        if position is None:
            # determine position size
            qty_local = compute_qty(symbol, balance, leverage, price, qty, use_full_balance, live)
            # OPEN LOGIC
            if price <= lower:
                qty_open = qty_local * 0.5 if rsi >= 60 else qty_local
                entry_price = price
                sl = entry_price * (1 - sl_pct + fee_rate)
                tp = entry_price * (1 + tp_pct + fee_rate)
                position = {
                    "side": "long",
                    "size": qty_open,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "dyn_sl": None,
                    "time": row["open_time"]
                }

                if live:
                    if can_open_trade(symbol, qty_open, leverage):
                        open_position(symbol, "BUY", qty_open, sl, tp, leverage)
                        balance = get_balance()
                        update_position(state, position)
                        update_balance(state, balance)
                else:
                    balance -= qty_open * entry_price * fee_rate
                log_event(f"Opened LONG {qty_open} {symbol} at {entry_price}, fee={qty_open * entry_price * fee_rate}")

            elif price >= upper:
                qty_open = qty_local * 0.5 if rsi <= 40 else qty_local
                entry_price = price
                sl = entry_price * (1 + sl_pct - fee_rate)
                tp = entry_price * (1 - tp_pct - fee_rate)
                position = {
                    "side": "short",
                    "size": qty_open,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "dyn_sl": None,
                    "time": row["open_time"]
                }
                
                if live:
                    if can_open_trade(symbol, qty_open, leverage):
                        open_position(symbol, "SELL", qty_open, sl, tp, leverage)
                        balance = get_balance()
                        update_position(state, position)
                        update_balance(state, balance)
                else:
                    balance -= qty_open * entry_price * fee_rate
                log_event(f"Opened SHORT {qty_open} {symbol} at {entry_price}, fee={qty_open * entry_price * fee_rate}")

        else:
            # MANAGEMENT for open position
            size = position["size"]
            entry_price = position["entry"]
            position_value = size * entry_price
            fee_close = position_value * fee_rate
            side = position["side"]
            base_sl = position["sl"]
            base_tp = position["tp"]

            if side == "long":
                if mid is not None and price >= mid:
                    candidate_dyn_sl = mid * (1 - dyn_sl_buffer)
                    break_even_min = entry_price * (1 + 0.001)
                    new_dyn = max(candidate_dyn_sl, break_even_min)
                    current_dyn = position.get("dyn_sl")
                    if current_dyn is None or new_dyn > current_dyn:
                        position["dyn_sl"] = new_dyn
                        update_position(state, position)
                        log_event(f"Updated LONG dynamic SL to {new_dyn} for trade at {entry_price}")

                # effective stop check (dynamic if set else base)
                effective_sl = position.get("dyn_sl")
                if effective_sl is None:
                    effective_sl = base_sl

                if price <= effective_sl or (price >= base_tp and rsi >= 60):
                    gross_pnl = (price - entry_price) / entry_price * position_value
                    if live:
                        close_position(symbol)
                        current_balance = get_balance()
                        net_pnl = current_balance - balance
                        balance = current_balance
                        update_position(state, None)
                        update_balance(state, balance)
                        add_closed_trade(state, trade)
                    else:
                        net_pnl = gross_pnl - fee_close
                        balance += net_pnl
                    
                    trade = {
                        **position,
                        "exit": price,
                        "gross_pnl": gross_pnl,
                        "fee": fee_close,
                        "net_pnl": net_pnl,
                        "exit_reason": "take_profit" if price >= base_tp else "stop_loss"
                    }
                    trade_history.append(trade)
                    position = None
                    log_event(f"Closed LONG {size} {symbol} at {price}, reason={trade['exit_reason']}, net={net_pnl}")

            elif side == "short":
                if mid is not None and price <= mid:
                    candidate_dyn_sl = mid * (1 + dyn_sl_buffer)
                    break_even_max = entry_price * (1 - 0.001)
                    new_dyn = min(candidate_dyn_sl, break_even_max)
                    current_dyn = position.get("dyn_sl")
                    if current_dyn is None or new_dyn < current_dyn:
                        position["dyn_sl"] = new_dyn
                        update_position(state, position)
                        log_event(f"Updated SHORT dynamic SL to {new_dyn} for trade at {entry_price}")

                # effective stop check (dynamic if set else base)
                effective_sl = position.get("dyn_sl")
                if effective_sl is None:
                    effective_sl = base_sl

                if price >= effective_sl or (price <= base_tp and rsi <= 40):
                    gross_pnl = (entry_price - price) / entry_price * position_value
                    if live:
                        close_position(symbol)
                        current_balance = get_balance()
                        net_pnl = current_balance - balance
                        balance = current_balance
                        update_position(state, None)
                        update_balance(state, balance)
                        add_closed_trade(state, trade)
                    else:
                        net_pnl = gross_pnl - fee_close
                        balance += net_pnl
                    
                    trade = {
                        **position,
                        "exit": price,
                        "gross_pnl": gross_pnl,
                        "fee": fee_close,
                        "net_pnl": net_pnl,
                        "exit_reason": "take_profit" if price <= base_tp else "stop_loss"
                    }
                    trade_history.append(trade)
                    position = None
                    log_event(f"Closed SHORT {size} {symbol} at {price}, reason={trade['exit_reason']}, net={net_pnl}")

        balance_history.append(balance)
    
    if live:
        save_state(state)
    else:
        state["position"] = position
        state["balance"] = balance
        state["trade_history"] = trade_history
        state["balance_history"] = balance_history
        save_state(state)

    return balance, pd.DataFrame(state.get("trade_history", [])), state.get("balance_history", []), state


def plot_results(df, trades, balance_history):
    """Plot price with Bollinger Bands, RSI and Equity Curve."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'height_ratios':[3,1,1]})

    # Price + Bollinger Bands
    ax1.plot(df["open_time"], df["close"], label="Price", color="blue")
    ax1.plot(df["open_time"], df["BBL"], label="Lower BB", color="red", linestyle="--")
    ax1.plot(df["open_time"], df["BBM"], label="Middle BB", color="black", linestyle="--")
    ax1.plot(df["open_time"], df["BBU"], label="Upper BB", color="green", linestyle="--")

    # Plot trades
    for _, trade in trades.iterrows():
        if pd.notna(trade["exit"]):
            if trade["exit_reason"] == "stop_loss":
                color = "red"
            elif trade["exit_reason"] == "take_profit":
                color = "green"
            else:
                color = "orange"

            ax1.scatter(trade["time"], trade["entry"], marker="^" if trade["side"]=="long" else "v",
                        color="blue", s=80)
            ax1.scatter(trade["time"], trade["exit"], marker="x", color=color, s=80)

    ax1.set_title("Bollinger Bands Strategy Backtest (with Dynamic SL and Fees)")
    ax1.legend()

    # RSI subplot
    ax2.plot(df["open_time"], df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax2.set_title("RSI")
    ax2.legend()

    # Equity Curve
    ax3.plot(df["open_time"][:len(balance_history)], balance_history, color="purple", label="Equity Curve")
    ax3.set_title("Equity Curve")
    ax3.legend()

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, dpi=150)
        webbrowser.get("firefox").open(tmpfile.name)
