print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta

import os
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.enums import *

# --- 1. Connect ---
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = "https://testnet.binancefuture.com"   # Testnet endpoint

# --- 2. Fetch data ---
def fetch_klines(symbol="BTCUSDT", interval="1m", limit=100):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]

# --- 3. Bollinger Bands ---
def compute_bbands(df, length=20, std=2):
    df = df.copy()
    bb = ta.bbands(df["close"], length=length, std=std)
    df["BBL"] = bb[f"BBL_{length}_{std}.0_2.0"]
    df["BBM"] = bb[f"BBM_{length}_{std}.0_2.0"]
    df["BBU"] = bb[f"BBU_{length}_{std}.0_2.0"]
    return df

# --- 4. Signal ---
def bollinger_signal(df):
    last = df.iloc[-1]
    if last["close"] <= last["BBL"]:
        return "long"
    elif last["close"] >= last["BBU"]:
        return "short"
    else:
        return None

# --- 5. Orders ---
def place_order(symbol, side, quantity=0.001):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "long" else SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"Order executed: {order['orderId']} | Side: {side} | Qty: {quantity}")
    except Exception as e:
        print("Order failed:", e)

# --- 6. Main loop ---
if __name__ == "__main__":
    symbol = "BTCUSDT"
    df = fetch_klines(symbol, interval="1m", limit=100)
    df = compute_bbands(df)
    signal = bollinger_signal(df)
    print(df.tail(3)[["open_time","close","BBL","BBM","BBU"]])
    print("Signal:", signal)

    if signal:
        place_order(symbol, signal, quantity=0.001)   # minimum volume
