import os
import time
import yaml
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from binance.client import Client
from datetime import datetime, timedelta

# --- Load configuration ---
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

symbol = config.get("symbol", "BTCUSDT")
interval_small = config["intervals"]["small"]
interval_medium = config["intervals"]["medium"]
interval_big = config["intervals"]["big"]
backtest_period_days = config["backtest_period_days"]
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

output_filename = f"{symbol}_{interval_small}_{interval_medium}_{interval_big}_{backtest_period_days}.pkl"
output_path = os.path.join(output_dir, output_filename)

# --- Binance Client ---
client = Client(API_KEY, API_SECRET)


def fetch_historical_paginated(symbol="BTCUSDT", interval="1m", start_time=None, end_time=None, sleep=0):
    """Fetch long historical klines from Binance Futures with pagination."""
    dfs = []
    limit = 1500
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
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open","high","low","close"]] = df[["open","high","low","close"]].apply(pd.to_numeric)
        dfs.append(df[["open_time","open","high","low","close","volume"]])

        last_ts = int(klines[-1][0]) + 1
        if end_ts and last_ts >= end_ts:
            break
        if len(klines) < limit:
            break
        start_ts = last_ts
        if sleep > 0:
            time.sleep(sleep)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["open_time", "close"])


def prepare_multi_tf(df_small, df_medium, df_big):
    """Merge multi-timeframe data (small=price, medium=BB/ATR, big=RSI/EMA)."""
    df_small["open_time"] = pd.to_datetime(df_small["open_time"], unit="ms")
    df_medium["open_time"] = pd.to_datetime(df_medium["open_time"], unit="ms")
    df_big["open_time"] = pd.to_datetime(df_big["open_time"], unit="ms")

    # --- RSI and EMA on big timeframe ---
    df_big = df_big.set_index("open_time")
    df_big["RSI"] = ta.rsi(df_big["close"], length=6)
    df_big["EMA"] = ta.ema(df_big["close"], length=50)
    df_big = df_big[["RSI", "EMA"]]

    # --- Bollinger Bands and ATR on medium timeframe ---
    df_medium = df_medium.set_index("open_time")
    bb = ta.bbands(df_medium["close"], length=20, std=2)
    atr = ta.atr(df_medium["high"], df_medium["low"], df_medium["close"], length=14)
    df_medium["BBL"] = bb[f"BBL_20_2.0_2.0"]
    df_medium["BBM"] = bb[f"BBM_20_2.0_2.0"]
    df_medium["BBU"] = bb[f"BBU_20_2.0_2.0"]
    df_medium["ATR"] = atr
    df_medium = df_medium[["BBL", "BBM", "BBU", "ATR"]]

    # --- Merge into small timeframe ---
    df_merged = df_small.set_index("open_time")
    df_merged = df_merged.merge(df_medium, left_index=True, right_index=True, how="left").ffill()
    df_merged = df_merged.merge(df_big, left_index=True, right_index=True, how="left").ffill()
    df_merged.reset_index(inplace=True)
    return df_merged


def build_dataset():
    """Fetch and build dataset from Binance, or load if already cached."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=backtest_period_days)

    if os.path.exists(output_path):
        print(f"[CACHE] Found existing dataset: {output_path}")
        return pd.read_pickle(output_path)

    print(f"[FETCH] Fetching data for lst {backtest_period_days} days on {symbol}: {interval_small}, {interval_medium}, {interval_big}")
    df_small = fetch_historical_paginated(symbol, interval_small, start_time, end_time)
    df_medium = fetch_historical_paginated(symbol, interval_medium, start_time, end_time)
    df_big = fetch_historical_paginated(symbol, interval_big, start_time, end_time)

    print("[MERGE] Preparing multi-timeframe dataframe...")
    df_merged = prepare_multi_tf(df_small, df_medium, df_big)

    df_merged.to_pickle(output_path)
    df_merged.to_csv(output_path.replace(".pkl", ".csv"), index=False)
    print(f"[DONE] Saved merged dataset → {output_path} ({len(df_merged)} rows)")

    return df_merged


if __name__ == "__main__":
    df = build_dataset()
