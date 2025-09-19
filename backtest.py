# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from binance.client import Client
import tempfile
import webbrowser

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)

def fetch_historical(symbol="BTCUSDT", interval="15m", limit=500):
    """Fetch historical klines from Binance Futures"""
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]

def prepare_multi_tf(df_small, df_big):
    # Ensure datetime
    df_small["open_time"] = pd.to_datetime(df_small["open_time"], unit="ms")
    df_big["open_time"] = pd.to_datetime(df_big["open_time"], unit="ms")

    # Set index to datetime for 4h
    df_big = df_big.set_index("open_time")

    # --- RSI on 4h ---
    df_big["RSI"] = ta.momentum.rsi(close=df_big["close"], window=14)

    # Merge 4h into 1h by forward filling (align)
    df_big = df_big["RSI"]
    df_merged = df_small.merge(df_big, left_on="open_time", right_index=True, how="left")
    df_merged["RSI"] = df_merged["RSI"].ffill()

    # Compute Bollinger Bands
    length = 20
    std = 2
    bb = ta.bbands(df_merged["close"], length=length, std=std)
    df_merged["BBL"] = bb[f"BBL_{length}_{std}.0_2.0"]
    df_merged["BBM"] = bb[f"BBM_{length}_{std}.0_2.0"]
    df_merged["BBU"] = bb[f"BBU_{length}_{std}.0_2.0"]

    return df_merged

def compute_stats(initial_balance, final_balance, trades, balance_history):
    stats = {}

    # Basic
    stats["Initial Balance"] = initial_balance
    stats["Final Balance"] = final_balance
    stats["Total Return %"] = (final_balance - initial_balance) / initial_balance * 100

    # Trades
    n_trades = len(trades)
    stats["Number of Trades"] = n_trades

    if n_trades > 0:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        
        stats["Win Rate %"] = len(wins) / n_trades * 100
        stats["Average PnL"] = trades["pnl"].mean()
        stats["Average Profit"] = wins["pnl"].mean()
        stats["Average Loss"] = losses["pnl"].mean()
        stats["Best Trade"] = trades["pnl"].max()
        stats["Worst Trade"] = trades["pnl"].min()

        total_profit = wins["pnl"].sum()
        total_loss = abs(losses["pnl"].sum())
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

def backtest_bollinger(df, initial_balance=10000, position_size=1, sl_pct=0.005, tp_pct=0.01):
    """
    Backtest Bollinger Bands strategy with SL/TP:
    - Entry long if price touches lower band
    - Entry short if price touches upper band
    - Exit long when price reaches middle band, SL or TP
    - Exit short when price reaches middle band, SL or TP
    """
    balance = initial_balance
    position = None
    entry_price = 0
    trades = []
    balance_history = []

    for i in range(1, len(df)):
        price = df.iloc[i]["close"]
        lower = df.iloc[i]["BBL"]
        upper = df.iloc[i]["BBU"]
        mid   = df.iloc[i]["BBM"]
        rsi   = df.iloc[i]["RSI"]
        
        size = position_size

        if position is None:
            # Open position
            if price <= lower:
                if rsi >= 60:
                    size = 0.5 * size
                position = "long"
                entry_price = price
                trades.append({
                    "side": "long",
                    "size": size,
                    "entry": price,
                    "exit": None,
                    "pnl": None,
                    "time": df.iloc[i]["open_time"]
                })
            elif price >= upper:
                if rsi <= 40:
                    size = 0.5 * size
                position = "short"
                entry_price = price
                trades.append({
                    "side": "short",
                    "size": size,
                    "entry": price,
                    "exit": None,
                    "pnl": None,
                    "time": df.iloc[i]["open_time"]
                })

        else:
            size = trades[-1]["size"]
            # Long position management
            if position == "long":
                sl_level = entry_price * (1 - sl_pct)
                tp_level = entry_price * (1 + tp_pct)

                if price <= sl_level:  # Stop-loss
                    loss = (price - entry_price) / entry_price * balance * size
                    balance += loss
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = loss
                    trades[-1]["exit_reason"] = "stop_loss"
                    position = None

                elif price >= tp_level and rsi >= 60:  # Take-profit
                    profit = (price - entry_price) / entry_price * balance * size
                    balance += profit
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = profit
                    trades[-1]["exit_reason"] = "take_profit"
                    position = None

                elif price >= mid:  # Exit by mid band
                    profit = (price - entry_price) / entry_price * balance * size
                    balance += profit
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = profit
                    trades[-1]["exit_reason"] = "mid_band"
                    position = None

            # Short position management
            elif position == "short":
                sl_level = entry_price * (1 + sl_pct)
                tp_level = entry_price * (1 - tp_pct)

                if price >= sl_level:  # Stop-loss
                    loss = (entry_price - price) / entry_price * balance * size
                    balance += loss
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = loss
                    trades[-1]["exit_reason"] = "stop_loss"
                    position = None

                elif price <= tp_level and rsi <= 40:  # Take-profit
                    profit = (entry_price - price) / entry_price * balance * size
                    balance += profit
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = profit
                    trades[-1]["exit_reason"] = "take_profit"
                    position = None

                elif price <= mid:  # Exit by mid band
                    profit = (entry_price - price) / entry_price * balance * size
                    balance += profit
                    trades[-1]["exit"] = price
                    trades[-1]["pnl"] = profit
                    trades[-1]["exit_reason"] = "mid_band"
                    position = None

        balance_history.append(balance)

    return balance, pd.DataFrame(trades), balance_history


def plot_results(df, trades, balance_history):
    """Plot price with Bollinger Bands, RSI and Equity Curve"""
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
                        color="blue", s=100)
            ax1.scatter(trade["time"], trade["exit"], marker="x", color=color, s=100)

    ax1.set_title("Bollinger Bands Strategy Backtest")
    ax1.legend()

    # --- RSI subplot ---
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

if __name__ == "__main__":
    df_small = fetch_historical("BTCUSDT", interval="5m", limit=1500)
    df_big = fetch_historical("BTCUSDT", interval="1h", limit=375)
    df = prepare_multi_tf(df_small, df_big)
    final_balance, trades, balance_history = backtest_bollinger(df)
    stats = compute_stats(10000, final_balance, trades, balance_history)

    for k, v in stats.items():
        print(f"{k}: {v}")
    
    plot_results(df, trades, balance_history)
