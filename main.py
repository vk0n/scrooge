print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta
import signal
import sys
import os
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
from datetime import datetime
from strategy import *
from trade import *

state = None


def handle_exit(sig, frame):
    """Handler for Ctrl+C (SIGINT) to gracefully save state and exit."""
    global state
    if state:
        print("\n[EXIT] Saving state before quitting...")
        save_state(state)
        print("[EXIT] State saved.")
    sys.exit(0)


# Register SIGINT handler
signal.signal(signal.SIGINT, handle_exit)


if __name__ == "__main__":
    symbol = "BTCUSDT"
    initial_balance = 1000
    use_full_balance = True
    qty = None  # position size
    interval_small = "1m"
    interval_medium = "15m"
    interval_big = "1h"
    limit_small = 1500
    limit_medium = 500
    limit_big = 100
    lvrg = 1
    sl_pct = 0.005
    tp_pct = 0.01

    live = True  # "backtest" or "live"
    
    # Load or create state
    state = load_state()

    if live:
        print("Running LIVE on Binance Futures...")
        set_leverage(symbol, lvrg)
        

        while True:
            try:
                current_balance = get_balance()
                # Log account balance and current position
                check_balance()
                pos = get_open_position(symbol)
                if pos:
                    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Open position: {pos['positionAmt']} {pos['symbol']}")
                else:
                    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] No open positions")

                # Fetch recent historical data
                df_small = fetch_historical(symbol, interval_small, limit_small)
                df_medium = fetch_historical(symbol, interval_medium, limit_medium)
                df_big = fetch_historical(symbol, interval_big, limit_big)
                df = prepare_multi_tf(df_small, df_medium, df_big)

                # Run strategy on the latest data
                run_strategy(df, current_balance, qty, sl_pct, tp_pct, live, symbol, lvrg, use_full_balance)

                # Wait until next candle
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Waiting for next check...")
                time.sleep(60)  # wait 1 minute (interval_small)

            except Exception as e:
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error in live loop:", e)
                time.sleep(10)

    else:
        print("Running BACKTEST...")
        df_small = fetch_historical(symbol, interval_small, limit_small)
        df_medium = fetch_historical(symbol, interval_medium, limit_medium)
        df_big = fetch_historical(symbol, interval_big, limit_big)
        df = prepare_multi_tf(df_small, df_medium, df_big)

        final_balance, trades, balance_history = run_strategy(df, initial_balance, qty, sl_pct, tp_pct, live, symbol, lvrg, use_full_balance)
        stats = compute_stats(initial_balance, final_balance, trades, balance_history)

        for k, v in stats.items():
            print(f"{k}: {v}")
        
        plot_results(df, trades, balance_history)
