print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from binance.client import Client
from dotenv import load_dotenv

import data as data_module
import report as report_module
import trade as trade_module
from control_channel import get_control_client, process_pending_commands
from data import build_dataset, fetch_historical, prepare_multi_tf
from report import (
    compute_stats,
    monte_carlo_from_equity,
    plot_results_interactive,
    rolling_window_backtest_distribution,
)
from state import load_state, save_state
from strategy import run_strategy
from trade import check_balance, get_balance, get_open_position, set_leverage

state: dict[str, Any] | None = None


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        print(f"[{_ts()}] Invalid integer for {name}: {raw}. Using {default}")
        return default


def handle_exit(sig: int, frame: Any) -> None:  # noqa: ARG001
    """Handler for Ctrl+C (SIGINT) to gracefully save state and exit."""
    if state:
        print("\n[EXIT] Saving state before quitting...")
        save_state(state)
        print("[EXIT] State saved.")
    sys.exit(0)


# Register SIGINT handler
signal.signal(signal.SIGINT, handle_exit)


def load_config() -> dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML object")
    return config


def _sleep_with_command_poll(
    current_state: dict[str, Any],
    control_client: Any,
    wait_seconds: int,
    poll_slice_seconds: int,
) -> tuple[dict[str, Any], bool]:
    if wait_seconds <= 0:
        return current_state, False

    if control_client is None:
        time.sleep(wait_seconds)
        return current_state, False

    restart_requested = False
    deadline = time.monotonic() + wait_seconds

    while True:
        now = time.monotonic()
        if now >= deadline:
            break

        time.sleep(min(poll_slice_seconds, max(0.1, deadline - now)))
        current_state, restart_now = process_pending_commands(control_client, current_state, save_state)
        restart_requested = restart_requested or restart_now

    return current_state, restart_requested


def _ensure_runtime_log_file() -> None:
    """
    Ensure runtime log file exists so API logs endpoint can read it immediately.
    """
    try:
        log_path = Path("trading_log.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
    except OSError as exc:
        print(f"[{_ts()}] Failed to ensure log file exists: {exc}")


if __name__ == "__main__":
    _ensure_runtime_log_file()
    cfg = load_config()

    live = cfg["live"]  # "backtest" or "live"

    symbol = cfg["symbol"]
    lvrg = cfg["leverage"]
    initial_balance = cfg["initial_balance"]
    qty = cfg["qty"]
    use_full_balance = cfg["use_full_balance"]

    intervals = cfg["intervals"]
    limits = cfg["limits"]
    backtest_period_days = cfg["backtest_period_days"]
    backtest_period_end_time = cfg["backtest_period_end_time"]
    enable_plot = cfg["enable_plot"]
    plot_split_by_year = cfg.get("plot_split_by_year", True)
    run_mc = cfg["run_monte_carlo"]
    run_rw = cfg["run_rolling_window_backtest_distribution"]
    rolling_window_days = cfg["rolling_window_days"]
    rolling_window_workers = cfg.get("rolling_window_workers")

    params = cfg["params"]

    live_poll_seconds = _env_int("SCROOGE_LIVE_POLL_SECONDS", 60)
    control_poll_slice_seconds = _env_int("SCROOGE_CONTROL_POLL_SLICE_SECONDS", 1)

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
        control_client = get_control_client()
        restart_requested = False
        last_trading_enabled: bool | None = None

        print("Running LIVE on Binance Futures...")
        set_leverage(symbol, lvrg)

        while True:
            try:
                if control_client is None:
                    control_client = get_control_client()

                state, restart_now = process_pending_commands(control_client, state, save_state)
                restart_requested = restart_requested or restart_now

                if restart_requested:
                    cfg = load_config()
                    symbol = cfg["symbol"]
                    lvrg = cfg["leverage"]
                    qty = cfg["qty"]
                    use_full_balance = cfg["use_full_balance"]
                    intervals = cfg["intervals"]
                    limits = cfg["limits"]
                    params = cfg["params"]
                    set_leverage(symbol, lvrg)
                    restart_requested = False
                    print(f"[{_ts()}] Restart command applied: config reloaded")

                trading_enabled = bool(state.get("trading_enabled", True))
                if trading_enabled != last_trading_enabled:
                    print(f"[{_ts()}] Trading status: {'running' if trading_enabled else 'paused'}")
                    last_trading_enabled = trading_enabled

                if not trading_enabled and state.get("position") is None:
                    print(f"[{_ts()}] Trading paused (idle). Waiting for next check...")
                    state, restart_now = _sleep_with_command_poll(
                        state,
                        control_client,
                        live_poll_seconds,
                        control_poll_slice_seconds,
                    )
                    restart_requested = restart_requested or restart_now
                    continue

                current_balance = get_balance()
                # Log account balance and current position
                check_balance()
                pos = get_open_position(symbol)
                if pos:
                    position = state.get("position") if isinstance(state.get("position"), dict) else {}
                    print(
                        f"[{_ts()}] Open position: {pos.get('positionAmt')} {pos.get('symbol')} "
                        f"| TP: {position.get('tp', 'n/a')} | SL: {position.get('sl', 'n/a')}"
                    )
                else:
                    print(f"[{_ts()}] No open positions")

                # Fetch recent historical data
                df_small = fetch_historical(symbol, intervals["small"], limits["small"])
                df_medium = fetch_historical(symbol, intervals["medium"], limits["medium"])
                df_big = fetch_historical(symbol, intervals["big"], limits["big"])
                df = prepare_multi_tf(df_small, df_medium, df_big)

                # Run strategy on the latest data
                balance, trades, balance_history, state = run_strategy(
                    df,
                    live,
                    current_balance,
                    qty,
                    symbol=symbol,
                    leverage=lvrg,
                    use_full_balance=use_full_balance,
                    state=state,
                    allow_entries=trading_enabled,
                    **params,
                )

                # Wait until next candle
                print(f"[{_ts()}] Waiting for next check...")
                state, restart_now = _sleep_with_command_poll(
                    state,
                    control_client,
                    live_poll_seconds,
                    control_poll_slice_seconds,
                )
                restart_requested = restart_requested or restart_now

            except Exception as e:  # noqa: BLE001
                print(f"[{_ts()}] Error in live loop: {e}")
                state, restart_now = _sleep_with_command_poll(
                    state,
                    control_client,
                    10,
                    control_poll_slice_seconds,
                )
                restart_requested = restart_requested or restart_now

    else:
        print("Running BACKTEST...")
        df = build_dataset(
            symbol=symbol,
            intervals=intervals,
            backtest_period_days=backtest_period_days,
            backtest_period_end_time=backtest_period_end_time,
        )

        final_balance, trades, balance_history, state = run_strategy(
            df,
            live,
            initial_balance,
            qty,
            symbol=symbol,
            leverage=lvrg,
            use_full_balance=use_full_balance,
            **params,
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
                **params,
            )
