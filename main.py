print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta

import os
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
from datetime import datetime
from strategy import *
from trade import check_balance, get_open_position

if __name__ == "__main__":
    symbol = "BTCUSDT"
    qty = 0.001  # position size
    interval_small = "5m"
    interval_big = "1h"

    MODE = "backtest"  # "backtest" or "live"

    if MODE == "backtest":
        print("Running BACKTEST...")
        df_small = fetch_historical(symbol, interval=interval_small, limit=1500)
        df_big = fetch_historical(symbol, interval=interval_big, limit=375)
        df = prepare_multi_tf(df_small, df_big)

        initial_balance = 1000
        final_balance, trades, balance_history = run_strategy(df, initial_balance, qty, live=False)
        stats = compute_stats(initial_balance, final_balance, trades, balance_history)

        for k, v in stats.items():
            print(f"{k}: {v}")
        
        plot_results(df, trades, balance_history)

    elif MODE == "live":
        print("Running LIVE on Binance Testnet...")

        while True:
            try:
                # Log account balance and current position
                check_balance()
                pos = get_open_position(symbol)
                if pos:
                    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Open position: {pos['positionAmt']} {pos['symbol']}")
                else:
                    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] No open positions")

                # Fetch recent historical data
                df_small = fetch_historical(symbol, interval=interval_small, limit=1500)
                df_big = fetch_historical(symbol, interval=interval_big, limit=375)
                df = prepare_multi_tf(df_small, df_big)

                # Run strategy on the latest data
                run_strategy(df, live=True, symbol=symbol, qty=qty)

                # Wait until next candle
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Waiting for next check...")
                time.sleep(300)  # wait 5 minutes (interval_small)

            except Exception as e:
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error in live loop:", e)
                time.sleep(10)
