print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta
import signal
import yaml
import sys
import os
import time
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
from strategy import run_strategy
from trade import set_leverage, get_balance, check_balance, get_open_position
from data import build_dataset, fetch_historical, prepare_multi_tf
from report import (
    compute_stats,
    plot_results,
    plot_results_interactive,
    monte_carlo_from_equity,
    rolling_window_backtest_distribution,
)
from state import load_state, save_state
import data as data_module
import trade as trade_module
import report as report_module

state = None


def handle_exit(sig, frame):
    """Handler for Ctrl+C (SIGINT) to gracefully save state and exit."""
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
    backtest_period_end_time=cfg["backtest_period_end_time"]
    enable_plot = cfg["enable_plot"]
    plot_split_by_year = cfg.get("plot_split_by_year", True)
    run_mc = cfg["run_monte_carlo"]
    run_rw = cfg["run_rolling_window_backtest_distribution"]
    rolling_window_days = cfg["rolling_window_days"]
    rolling_window_workers = cfg.get("rolling_window_workers")

    params = cfg["params"]

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    data_module.set_client(client)
    trade_module.set_client(client)
    report_module.set_client(client)

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
                    position = state["position"]
                    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Open position: {pos['positionAmt']} {pos['symbol']} | TP: {position['tp']:.1f} | SL: {position['sl']:.1f}")
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
        df = build_dataset(
            symbol=symbol,
            intervals=intervals,
            backtest_period_days=backtest_period_days,
            backtest_period_end_time=backtest_period_end_time
        )

        final_balance, trades, balance_history, state = run_strategy(
            df, live, initial_balance, qty,
            symbol=symbol, leverage=lvrg,
            use_full_balance=use_full_balance, **params
        )
        stats = compute_stats(initial_balance, final_balance, trades, balance_history)

        for k, v in stats.items():
            print(f"{k}: {v}")
        
        if enable_plot:
            plot_results_interactive(df, trades, balance_history, split_by_year=plot_split_by_year)

        if run_mc:
            monte_carlo_from_equity(df, balance_history, start_balance=initial_balance)

        if run_rw:
            rolling_window_backtest_distribution(
                df,
                k_days=rolling_window_days,
                n_days=backtest_period_days,
                start_balance=initial_balance,
                max_workers=rolling_window_workers,
                leverage=lvrg,
                **params
            )
