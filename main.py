print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta
import signal
import yaml
import sys
import os
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
from datetime import datetime, timedelta
from strategy import *
from trade import *
from data import *
from report import *

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

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()

    live = cfg["live"] # "backtest" or "live"

    symbol = cfg["symbol"]
    lvrg = cfg["leverage"]
    initial_balance = cfg["initial_balance"]
    qty = cfg["qty"]
    use_full_balance = cfg["use_full_balance"]

    intervals = cfg["intervals"]
    limits = cfg["limits"]
    backtest_period_days = cfg["backtest_period_days"]
    enable_plot = cfg["enable_plot"]
    run_mc = cfg["run_monte_carlo"]

    params = cfg["params"]

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
                df_small = fetch_historical(symbol, intervals["small"], limits["small"])
                df_medium = fetch_historical(symbol, intervals["medium"], limits["medium"])
                df_big = fetch_historical(symbol, intervals["big"], limits["big"])
                df = prepare_multi_tf(df_small, df_medium, df_big)

                # Run strategy on the latest data
                balance, trades, balance_history, state = run_strategy(
                    df, live, current_balance, qty,
                    symbol=symbol, leverage=lvrg,
                    use_full_balance=use_full_balance, state=state, **params
                )

                # Wait until next candle
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Waiting for next check...")
                time.sleep(60)  # wait 1 minute (interval_small)

            except Exception as e:
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error in live loop:", e)
                time.sleep(10)

    else:
        print("Running BACKTEST...")
        df = build_dataset()

        final_balance, trades, balance_history, state = run_strategy(
            df, live, initial_balance, qty,
            symbol=symbol, leverage=lvrg,
            use_full_balance=use_full_balance, **params
        )
        stats = compute_stats(initial_balance, final_balance, trades, balance_history)

        for k, v in stats.items():
            print(f"{k}: {v}")
        
        if enable_plot:
            plot_results(df, trades, balance_history)

        if run_mc:
            monte_carlo_from_equity(df, balance_history, start_balance=initial_balance)

