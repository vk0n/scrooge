print("Hello!\nI am Scrooge...")

# requirements:
# pip install python-binance pandas pandas_ta
import csv
import math
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
from state import add_closed_trade, load_state, save_state, update_balance, update_position
from strategy import run_strategy
from trade import close_position, get_balance, get_open_position, set_leverage

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
    config_path = Path(os.getenv("SCROOGE_CONFIG_PATH", "config.yaml")).expanduser()
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML object")
    return config


def _sleep_with_command_poll(
    current_state: dict[str, Any],
    control_client: Any,
    wait_seconds: int,
    poll_slice_seconds: int,
    command_kwargs: dict[str, Any] | None = None,
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
        current_state, restart_now = process_pending_commands(
            control_client,
            current_state,
            save_state,
            **(command_kwargs or {}),
        )
        restart_requested = restart_requested or restart_now

    return current_state, restart_requested


def _ensure_runtime_log_file() -> None:
    """
    Ensure runtime log file exists so API logs endpoint can read it immediately.
    """
    try:
        log_path = Path(os.getenv("SCROOGE_LOG_FILE", "trading_log.txt")).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
    except OSError as exc:
        print(f"[{_ts()}] Failed to ensure log file exists: {exc}")


def _parse_open_time_to_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000_000:
            return int(numeric)
        if numeric > 100_000_000:
            return int(numeric * 1000)
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        numeric = int(text)
        if numeric > 10_000_000_000:
            return numeric
        if numeric > 100_000_000:
            return numeric * 1000
    normalized = text.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def _format_open_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if text:
        return text
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _coerce_optional_float(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return numeric


def _read_last_chart_dataset_ts_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        last_line = ""
        with path.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line in file_obj:
                if line.strip():
                    last_line = line.strip()
        if not last_line or last_line.startswith("open_time,"):
            return None
        first_cell = last_line.split(",", 1)[0]
        return _parse_open_time_to_ms(first_cell)
    except OSError as exc:
        print(f"[{_ts()}] Failed reading chart dataset tail: {exc}")
        return None


def _read_dataset_header_columns(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file_obj:
            header = [column.strip() for column in file_obj.readline().strip().split(",") if column.strip()]
        return header or None
    except OSError:
        return None


def _append_latest_chart_candle(
    df: Any,
    symbol: str,
    path: Path,
    last_ts_ms: int | None,
    balance: float | None = None,
) -> int | None:
    if df is None or len(df) == 0:  # noqa: PLR2004
        return last_ts_ms

    latest = df.iloc[-1]
    ts_ms = _parse_open_time_to_ms(latest.get("open_time"))
    if ts_ms is None:
        return last_ts_ms
    if last_ts_ms is not None and ts_ms <= last_ts_ms:
        return last_ts_ms

    open_price = latest.get("open")
    high_price = latest.get("high")
    low_price = latest.get("low")
    close_price = latest.get("close")
    volume = latest.get("volume")
    if None in (open_price, high_price, low_price, close_price, volume):
        return last_ts_ms

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        existing_header = _read_dataset_header_columns(path)
        default_header = [
            "open_time",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "balance",
            "EMA",
            "RSI",
            "BBL",
            "BBM",
            "BBU",
            "ATR",
        ]
        header = default_header if write_header else (existing_header or default_header)

        row_map: dict[str, float | str] = {
            "open_time": _format_open_time(latest.get("open_time")),
            "symbol": symbol,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
            "balance": _coerce_optional_float(balance),
            "EMA": _coerce_optional_float(latest.get("EMA")),
            "RSI": _coerce_optional_float(latest.get("RSI")),
            "BBL": _coerce_optional_float(latest.get("BBL")),
            "BBM": _coerce_optional_float(latest.get("BBM")),
            "BBU": _coerce_optional_float(latest.get("BBU")),
            "ATR": _coerce_optional_float(latest.get("ATR")),
        }

        with path.open("a", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            if write_header:
                writer.writerow(header)
            row_values = [row_map.get(column, "") for column in header]
            writer.writerow(row_values)
        return ts_ms
    except OSError as exc:
        print(f"[{_ts()}] Failed to append chart dataset candle: {exc}")
        return last_ts_ms


def _write_chart_dataset_snapshot(df: Any, symbol: str, path: Path, balance_history: list[float] | None = None) -> None:
    if df is None or len(df) == 0:  # noqa: PLR2004
        return
    required = {"open_time", "open", "high", "low", "close", "volume"}
    columns = set(df.columns)
    if not required.issubset(columns):
        print(f"[{_ts()}] Skip chart dataset snapshot: missing required columns")
        return

    aligned_balance_history: list[float | str] = []
    if balance_history:
        clean_history: list[float] = []
        for value in balance_history:
            try:
                clean_history.append(float(value))
            except (TypeError, ValueError):
                continue
        if clean_history:
            rows_count = len(df)
            history_count = len(clean_history)
            if history_count >= rows_count:
                aligned_balance_history = clean_history[-rows_count:]
            else:
                # Right-align shorter history with candles so newest balances match newest candles.
                aligned_balance_history = ([""] * (rows_count - history_count)) + clean_history

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                ["open_time", "symbol", "open", "high", "low", "close", "volume", "balance", "EMA", "RSI", "BBL", "BBM", "BBU", "ATR"]
            )
            for idx, row in enumerate(df.itertuples(index=False)):
                balance_value: float | str = ""
                if aligned_balance_history and idx < len(aligned_balance_history):
                    balance_value = aligned_balance_history[idx]
                writer.writerow(
                    [
                        _format_open_time(getattr(row, "open_time", None)),
                        symbol,
                        float(getattr(row, "open")),
                        float(getattr(row, "high")),
                        float(getattr(row, "low")),
                        float(getattr(row, "close")),
                        float(getattr(row, "volume")),
                        balance_value,
                        _coerce_optional_float(getattr(row, "EMA", None)),
                        _coerce_optional_float(getattr(row, "RSI", None)),
                        _coerce_optional_float(getattr(row, "BBL", None)),
                        _coerce_optional_float(getattr(row, "BBM", None)),
                        _coerce_optional_float(getattr(row, "BBU", None)),
                        _coerce_optional_float(getattr(row, "ATR", None)),
                    ]
                )
    except OSError as exc:
        print(f"[{_ts()}] Failed to write chart dataset snapshot: {exc}")


def _compress_balance_history_for_state(balance_history: list[float] | None) -> list[float]:
    """
    Keep only balance change points for persisted state history.
    This keeps backtest-derived history compact while preserving step changes.
    """
    if not balance_history:
        return []

    compressed: list[float] = []
    last_value: float | None = None
    for item in balance_history:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if last_value is None or not math.isclose(value, last_value, rel_tol=0.0, abs_tol=1e-9):
            compressed.append(value)
            last_value = value
    return compressed


def _build_command_kwargs(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "close_position_fn": close_position,
        "get_open_position_fn": get_open_position,
        "get_balance_fn": get_balance,
        "update_position_fn": update_position,
        "update_balance_fn": update_balance,
        "add_closed_trade_fn": add_closed_trade,
    }


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

    params = cfg["params"]

    live_poll_seconds = _env_int("SCROOGE_LIVE_POLL_SECONDS", 60)
    control_poll_slice_seconds = _env_int("SCROOGE_CONTROL_POLL_SLICE_SECONDS", 1)
    chart_dataset_path = Path(os.getenv("SCROOGE_RUNTIME_CHART_DATASET_PATH", "chart_dataset.csv")).expanduser()

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    data_module.set_client(client)
    trade_module.set_client(client)
    report_module.set_client(client)

    # Load or create state
    state = load_state()
    state_path = Path(os.getenv("SCROOGE_STATE_FILE", "state.json")).expanduser()
    if not state_path.exists():
        save_state(state)
        print(f"[{_ts()}] Initialized state file: {state_path}")

    if live:
        control_client = get_control_client()
        restart_requested = False
        last_trading_enabled: bool | None = None
        last_chart_dataset_ts_ms = _read_last_chart_dataset_ts_ms(chart_dataset_path)
        command_kwargs = _build_command_kwargs(symbol)

        print("Running LIVE on Binance Futures...")
        set_leverage(symbol, lvrg)

        while True:
            try:
                if control_client is None:
                    control_client = get_control_client()

                state, restart_now = process_pending_commands(
                    control_client,
                    state,
                    save_state,
                    **command_kwargs,
                )
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
                    command_kwargs = _build_command_kwargs(symbol)
                    restart_requested = False
                    print(f"[{_ts()}] Restart command applied: config reloaded")

                trading_enabled = bool(state.get("trading_enabled", True))
                if trading_enabled != last_trading_enabled:
                    print(f"[{_ts()}] Trading status: {'running' if trading_enabled else 'paused'}")
                    last_trading_enabled = trading_enabled

                current_balance = get_balance()
                update_balance(state, current_balance)
                print(f"[{_ts()}] Balance: {current_balance:.2f} USDT")

                if not trading_enabled and state.get("position") is None:
                    print(f"[{_ts()}] Trading paused (idle). Waiting for next check...")
                    state, restart_now = _sleep_with_command_poll(
                        state,
                        control_client,
                        live_poll_seconds,
                        control_poll_slice_seconds,
                        command_kwargs=command_kwargs,
                    )
                    restart_requested = restart_requested or restart_now
                    continue

                # Log current position
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
                last_chart_dataset_ts_ms = _append_latest_chart_candle(
                    df=df,
                    symbol=symbol,
                    path=chart_dataset_path,
                    last_ts_ms=last_chart_dataset_ts_ms,
                    balance=float(balance),
                )

                # Wait until next candle
                print(f"[{_ts()}] Waiting for next check...")
                state, restart_now = _sleep_with_command_poll(
                    state,
                    control_client,
                    live_poll_seconds,
                    control_poll_slice_seconds,
                    command_kwargs=command_kwargs,
                )
                restart_requested = restart_requested or restart_now

            except Exception as e:  # noqa: BLE001
                print(f"[{_ts()}] Error in live loop: {e}")
                state, restart_now = _sleep_with_command_poll(
                    state,
                    control_client,
                    10,
                    control_poll_slice_seconds,
                    command_kwargs=command_kwargs,
                )
                restart_requested = restart_requested or restart_now

    else:
        backtest_period_days = cfg["backtest_period_days"]
        backtest_period_end_time = cfg["backtest_period_end_time"]
        enable_plot = cfg["enable_plot"]
        plot_split_by_year = cfg.get("plot_split_by_year", True)
        run_mc = cfg["run_monte_carlo"]
        run_rw = cfg["run_rolling_window_backtest_distribution"]
        rolling_window_days = cfg["rolling_window_days"]
        rolling_window_workers = cfg.get("rolling_window_workers")

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
            use_state=False,
            **params,
        )
        state["balance_history"] = _compress_balance_history_for_state(balance_history)
        state["balance"] = float(final_balance)
        if len(state["balance_history"]) != len(balance_history):
            print(
                f"[{_ts()}] Compressed backtest balance history "
                f"from {len(balance_history)} to {len(state['balance_history'])} points for state persistence"
            )
        save_state(state)
        _write_chart_dataset_snapshot(
            df=df,
            symbol=symbol,
            path=chart_dataset_path,
            balance_history=balance_history,
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
