import json
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import webbrowser
from binance.client import Client
from datetime import datetime
import tempfile
import os
from dotenv import load_dotenv
import numpy as np

STATE_FILE = "state.json"
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(API_KEY, API_SECRET)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

def fetch_session_klines(symbol, interval, start_ts, end_ts):
    """Fetch historical klines from Binance for session period."""
    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        endTime=end_ts,
        limit=1500
    )
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]

def compute_session_stats(trades, balance_history):
    """Compute basic session performance metrics."""
    stats = {}
    stats["Number of Trades"] = len(trades)
    if len(trades) > 0:
        wins = [t for t in trades if t.get("net_pnl", 0) > 0]
        losses = [t for t in trades if t.get("net_pnl", 0) < 0]
        stats["Win Rate %"] = len(wins) / len(trades) * 100
        stats["Average PnL"] = np.mean([t.get("net_pnl", 0) for t in trades])
        stats["Average Profit"] = np.mean([t.get("net_pnl", 0) for t in wins]) if wins else 0
        stats["Average Loss"] = np.mean([t.get("net_pnl", 0) for t in losses]) if losses else 0
        stats["Best Trade"] = max([t.get("net_pnl", 0) for t in trades])
        stats["Worst Trade"] = min([t.get("net_pnl", 0) for t in trades])
        total_profit = sum([t.get("net_pnl", 0) for t in wins])
        total_loss = abs(sum([t.get("net_pnl", 0) for t in losses]))
        stats["Profit Factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else np.inf
    else:
        stats["Win Rate %"] = 0
        stats["Average PnL"] = 0
        stats["Best Trade"] = 0
        stats["Worst Trade"] = 0
        stats["Profit Factor"] = 0

    # Max drawdown
    equity = np.array(balance_history)
    if len(equity) > 0:
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        stats["Max Drawdown %"] = drawdowns.min() * 100
    else:
        stats["Max Drawdown %"] = 0

    return stats

def plot_session(state, symbol="BTCUSDT", interval="1m", show_bbands=False):
    """Visualize session trades, price movement, and equity curve."""
    session_start = state.get("session_start")
    session_end = state.get("session_end")
    trades = state.get("trade_history", [])
    balance_history = state.get("balance_history", [])

    if session_start is None or session_end is None:
        print("Session timestamps missing in state.json")
        return

    # --- Fetch session price data ---
    df = fetch_session_klines(symbol, interval, session_start, session_end)
    if df.empty:
        print("No klines fetched for the session.")
        return

    # --- Optional: Calculate Bollinger Bands for market context ---
    if show_bbands:
        bb = ta.bbands(df["close"], length=20, std=2)
        df["BBL"] = bb["BBL_20_2.0"]
        df["BBM"] = bb["BBM_20_2.0"]
        df["BBU"] = bb["BBU_20_2.0"]

    # --- Downsample if dataset is too large (for faster, clearer plotting) ---
    if len(df) > 5000:
        df = df.iloc[::len(df)//3000]

    # --- Create subplots: price & equity ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # ========== PRICE CHART ==========
    ax1.plot(df["open_time"], df["close"], label="Price", color="blue", linewidth=1)

    # --- Bollinger Bands (if enabled) ---
    if show_bbands:
        ax1.plot(df["open_time"], df["BBL"], linestyle="--", color="red", alpha=0.5, label="BB Lower")
        ax1.plot(df["open_time"], df["BBM"], linestyle="--", color="black", alpha=0.5, label="BB Mid")
        ax1.plot(df["open_time"], df["BBU"], linestyle="--", color="green", alpha=0.5, label="BB Upper")

    # --- Plot trade entries and exits ---
    for trade in trades:
        # Parse trade time safely (handle both string and timestamp formats)
        entry_time = None
        if isinstance(trade.get("time"), (int, float)):
            entry_time = pd.to_datetime(trade["time"], unit="ms", errors="coerce")
        else:
            entry_time = pd.to_datetime(trade["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        entry_price = trade.get("entry")
        exit_price = trade.get("exit")
        side = trade.get("side", "unknown")

        # Entry marker
        ax1.scatter(
            entry_time, entry_price,
            marker="^" if side == "long" else "v",
            color="lime" if side == "long" else "red",
            s=80, zorder=5, label="Entry" if side == "long" else None
        )

        # Exit marker
        if exit_price:
            exit_reason = trade.get("exit_reason", "")
            color = "green" if exit_reason == "take_profit" else "orange"
            ax1.scatter(
                entry_time, exit_price,
                marker="x", color=color, s=60, zorder=5, label="Exit"
            )

    ax1.set_title(f"{symbol} Price and Trades")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Equity curve ---
    if balance_history:
        # Try to align balances with candle times
        if len(balance_history) == len(df):
            time_index = df["open_time"]
        else:
            # fallback: linearly spread across session
            time_index = pd.date_range(
                start=df["open_time"].iloc[0],
                end=df["open_time"].iloc[-1],
                periods=len(balance_history)
            )

        ax2.plot(time_index, balance_history,
                 color="purple", label="Equity Curve")
        ax2.legend()
    ax2.set_title("Equity Curve")

    # --- Format x-axis timestamps ---
    fig.autofmt_xdate()

    plt.tight_layout()

    # --- Save plot to temp file and open in browser ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, dpi=150)
        webbrowser.get("firefox").open(tmpfile.name)
    print("Session report plotted.")

    # --- Compute and print session stats ---
    stats = compute_session_stats(trades, balance_history)
    print("\n=== SESSION STATISTICS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    state = load_state()
    if state:
        plot_session(state, show_bbands=False)
    else:
        print("No state found. Run a trading session first.")
