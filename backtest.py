# requirements:
# pip install python-binance pandas pandas_ta matplotlib

import os
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

def compute_bbands(df, length=20, std=2):
    """Compute Bollinger Bands"""
    bb = ta.bbands(df["close"], length=length, std=std)
    df["BBL"] = bb[f"BBL_{length}_{std}.0_2.0"]
    df["BBM"] = bb[f"BBM_{length}_{std}.0_2.0"]
    df["BBU"] = bb[f"BBU_{length}_{std}.0_2.0"]
    return df

def backtest_bollinger(df, initial_balance=1000, position_size=0.1):
    """
    Simple backtest:
    - Entry long if price touches lower band
    - Entry short if price touches upper band
    - Exit when price returns to middle band
    """
    balance = initial_balance
    position = None
    entry_price = 0
    trades = []

    for i in range(1, len(df)):
        price = df.iloc[i]["close"]
        lower = df.iloc[i]["BBL"]
        upper = df.iloc[i]["BBU"]
        mid   = df.iloc[i]["BBM"]

        if position is None:
            # Open position
            if price <= lower:
                position = "long"
                entry_price = price
                trades.append({"side":"long","entry":price,"exit":None,"pnl":None,"time":df.iloc[i]["open_time"]})
            elif price >= upper:
                position = "short"
                entry_price = price
                trades.append({"side":"short","entry":price,"exit":None,"pnl":None,"time":df.iloc[i]["open_time"]})

        else:
            # Close position
            if position == "long" and price >= mid:
                profit = (price - entry_price) / entry_price * balance * position_size
                balance += profit
                trades[-1]["exit"] = price
                trades[-1]["pnl"] = profit
                position = None

            elif position == "short" and price <= mid:
                profit = (entry_price - price) / entry_price * balance * position_size
                balance += profit
                trades[-1]["exit"] = price
                trades[-1]["pnl"] = profit
                position = None

    return balance, pd.DataFrame(trades)

def plot_results(df, trades):
    """Plot price, Bollinger Bands and trades"""
    plt.figure(figsize=(14,7))
    plt.plot(df["open_time"], df["close"], label="Price", color="blue")
    plt.plot(df["open_time"], df["BBU"], label="Upper Band", color="red", linestyle="--")
    plt.plot(df["open_time"], df["BBM"], label="Middle Band", color="black", linestyle="--")
    plt.plot(df["open_time"], df["BBL"], label="Lower Band", color="green", linestyle="--")

    # Mark trades
    for _, trade in trades.iterrows():
        if trade["exit"] is None:
            continue
        if trade["side"] == "long":
            plt.scatter(trade["time"], trade["entry"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(trade["time"], trade["exit"], marker="v", color="red", s=100, label="Long Exit")
        elif trade["side"] == "short":
            plt.scatter(trade["time"], trade["entry"], marker="v", color="red", s=100, label="Short Entry")
            plt.scatter(trade["time"], trade["exit"], marker="^", color="green", s=100, label="Short Exit")

    plt.title("Bollinger Bands Strategy Backtest")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, dpi=150)
        webbrowser.get("firefox").open(tmpfile.name)

if __name__ == "__main__":
    df = fetch_historical("BTCUSDT", interval="4h", limit=500)
    df = compute_bbands(df)
    final_balance, trades = backtest_bollinger(df)

    print("Initial balance: 1000 USDT")
    print("Final balance:", round(final_balance,2), "USDT")
    print("Total trades:", len(trades))

    plot_results(df, trades)
