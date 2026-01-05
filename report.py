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
from data import *

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
        stats["Total Fee"] = trades["fee"].sum()

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


def plot_results(df, trades, balance_history):
    """Plot price with Bollinger Bands, RSI and Equity Curve."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'height_ratios':[3,1,1]})

    # Price + Bollinger Bands
    ax1.plot(df["open_time"], df["close"], label="Price", color="blue")
    ax1.plot(df["open_time"], df["BBL"], label="Lower BB", color="red", linestyle="--")
    ax1.plot(df["open_time"], df["BBM"], label="Middle BB", color="black", linestyle="--")
    ax1.plot(df["open_time"], df["BBU"], label="Upper BB", color="green", linestyle="--")
    ax1.plot(df["open_time"], df["EMA"], label="EMA", color="purple", alpha=0.5)

    # Plot trades
    for _, trade in trades.iterrows():
        if pd.notna(trade["exit"]):
            if trade["exit_reason"] in ["stop_loss", "liquidation"]:
                color = "red"
            elif trade["exit_reason"] in ["take_profit", "rsi"]:
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

def plot_session(state, symbol="BTCUSDT", interval="1m", show_bbands=True):
    """Visualize session trades, price movement, and equity curve."""
    session_start = pd.to_datetime(state.get("session_start"), unit="ms")
    session_end = pd.to_datetime(state.get("session_end"), unit="ms")
    trades = state.get("trade_history", [])
    balance_history = state.get("balance_history", [])

    if session_start is None or session_end is None:
        print("Session timestamps missing in state.json")
        return

    # --- Fetch session price data ---
    df_small = fetch_historical_paginated(symbol, "1m", session_start, session_end)
    df_medium = fetch_historical_paginated(symbol, "1h", session_start, session_end)
    df_big = fetch_historical_paginated(symbol, "4h", session_start, session_end)
    df = prepare_multi_tf(df_small, df_medium, df_big)
    if df.empty:
        print("No klines fetched for the session.")
        return

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
    ax1.plot(df["open_time"], df["EMA"], color="purple", alpha=0.5, label="EMA")

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
            if exit_reason in ["stop_loss", "liquidation"]:
                color = "red"
            elif exit_reason in ["take_profit", "rsi"]:
                color = "green"
            else:
                color = "orange"
            ax1.scatter(
                entry_time, exit_price,
                marker="x", color=color, s=60, zorder=5, label="Exit"
            )

    ax1.set_title(f"{symbol} Price and Trades")
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


def monte_carlo_from_equity(df, balance_history, start_balance=10000, sims=10000, horizon_months=None, block_len=3, show_plot=True):
    """
    Advanced Monte Carlo stress test based on monthly equity returns.
    Includes CAGR, Volatility and Sharpe Ratio metrics.
    """

    # --- 1. Generate synthetic datetime index for the equity curve ---
    if len(balance_history) == 0:
        print("No equity data found for Monte Carlo test.")
        return {}

    start_time = pd.to_datetime(df["open_time"].iloc[0])
    end_time = pd.to_datetime(df["open_time"].iloc[-1])
    synthetic_index = pd.date_range(start=start_time, end=end_time, periods=len(balance_history))

    equity = pd.Series(balance_history, index=synthetic_index)

    # --- 2. Resample monthly and compute returns ---
    eq_monthly = equity.resample("ME").last().dropna()
    monthly_returns = eq_monthly.pct_change().dropna()

    if len(monthly_returns) < 3:
        print("Not enough data for monthly Monte Carlo test.")
        return {}

    # --- 3. Convert to log returns for numerical stability ---
    log_returns = np.log1p(monthly_returns.values)

    # --- 4. Simulation setup ---
    if horizon_months is None:
        horizon_months = len(log_returns)

    results = np.empty(sims)
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(horizon_months / block_len))

    # --- 5. Monte Carlo block bootstrap ---
    for i in range(sims):
        seq = []
        for _ in range(n_blocks):
            start_idx = rng.integers(0, len(log_returns) - block_len + 1)
            seq.append(log_returns[start_idx:start_idx + block_len])
        sampled = np.concatenate(seq)[:horizon_months]
        results[i] = start_balance * np.exp(sampled.sum())

    # --- 6. Compute percentiles ---
    p5, p50, p95 = np.percentile(results, [5, 50, 95])

    # --- 7. Risk metrics (CAGR, Volatility, Sharpe) ---
    years = horizon_months / 12
    cagr = ((p50 / start_balance) ** (1 / years) - 1) * 100
    volatility = np.std(monthly_returns) * np.sqrt(12) * 100
    risk_free_rate = 0.03  # 3% baseline annual risk-free rate
    sharpe = (cagr / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else np.nan

    # --- 8. Prepare summary dictionary ---
    summary = {
        "Simulations": sims,
        "Months per Run": horizon_months,
        "5th Percentile (Pessimistic)": round(p5, 2),
        "50th Percentile (Median)": round(p50, 2),
        "95th Percentile (Optimistic)": round(p95, 2),
        "Expected Range": f"{round(p5,2)} → {round(p95,2)}",
        "CAGR %": round(cagr, 2),
        "Volatility %": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

    # --- 9. Optional visualization ---
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.hist(results, bins=80, color="purple", alpha=0.7)
        plt.axvline(p5, color="red", linestyle="--", label="5th Percentile")
        plt.axvline(p50, color="black", linestyle="-", label="Median")
        plt.axvline(p95, color="green", linestyle="--", label="95th Percentile")
        plt.title("Monte Carlo Distribution Based on Monthly Equity Returns")
        plt.xlabel("Final Balance ($)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        # --- Save plot to temp file and open in browser ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            webbrowser.get("firefox").open(tmpfile.name)

    # --- 10. Print formatted summary ---
    print("\nMonte Carlo Stress-Test Summary (Monthly):")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


if __name__ == "__main__":
    state = load_state()
    if state:
        plot_session(state)
    else:
        print("No state found. Run a trading session first.")
