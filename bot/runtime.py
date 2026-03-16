import csv
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from binance.client import Client
from dotenv import load_dotenv

import bot.trade as trade_module
import backtest.dataset as data_module
from bot.control_channel import get_control_client, process_pending_commands
from backtest.dataset import fetch_historical, prepare_multi_tf
from bot.event_log import get_technical_logger
from bot.market_stream import LiveMarketStream
from bot.state import add_closed_trade, load_state, save_state, update_balance, update_position
from bot.trade import (
    close_position,
    get_balance,
    get_cached_balance_age_seconds,
    get_cached_balance,
    get_cached_open_position,
    get_cached_position_age_seconds,
    get_open_position,
    set_leverage,
)
from core.engine import run_strategy

RLockType = type(threading.RLock())

state: dict[str, Any] | None = None
state_lock: RLockType | None = None
live_market_stream: LiveMarketStream | None = None
technical_logger = get_technical_logger()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        technical_logger.warning("invalid_env_integer name=%s raw=%s fallback=%s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        technical_logger.warning("invalid_env_float name=%s raw=%s fallback=%s", name, raw, default)
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def handle_exit(sig: int, frame: Any) -> None:  # noqa: ARG001
    """Handler for Ctrl+C (SIGINT) to gracefully save state and exit."""
    global live_market_stream
    if state:
        if state_lock is not None:
            with state_lock:
                technical_logger.info("graceful_exit_saving_state")
                save_state(state)
                technical_logger.info("graceful_exit_state_saved")
        else:
            technical_logger.info("graceful_exit_saving_state")
            save_state(state)
            technical_logger.info("graceful_exit_state_saved")
    if live_market_stream is not None:
        live_market_stream.stop()
        live_market_stream = None
    sys.exit(0)


# Register SIGINT handler
signal.signal(signal.SIGINT, handle_exit)


def load_config() -> dict[str, Any]:
    config_path = Path(os.getenv("SCROOGE_CONFIG_PATH", "config/live.yaml")).expanduser()
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
    state_lock_obj: RLockType | None = None,
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
        if state_lock_obj is not None:
            with state_lock_obj:
                current_state, restart_now = process_pending_commands(
                    control_client,
                    current_state,
                    save_state,
                    **(command_kwargs or {}),
                )
        else:
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
        log_path = Path(os.getenv("SCROOGE_LOG_FILE", "runtime/trading_log.txt")).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
    except OSError as exc:
        technical_logger.warning("runtime_log_init_failed path=%s error=%s", log_path, exc)


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
        technical_logger.warning("chart_dataset_tail_read_failed path=%s error=%s", path, exc)
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
        technical_logger.warning("chart_dataset_append_failed path=%s symbol=%s error=%s", path, symbol, exc)
        return last_ts_ms


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
    initial_balance = cfg.get("initial_balance")
    qty = cfg["qty"]
    use_full_balance = cfg["use_full_balance"]

    intervals = cfg["intervals"]
    limits = cfg["limits"]

    params = cfg["params"]

    live_poll_seconds = _env_int("SCROOGE_LIVE_POLL_SECONDS", 60)
    control_poll_slice_seconds = _env_int("SCROOGE_CONTROL_POLL_SLICE_SECONDS", 1)
    user_stream_stale_after_seconds = _env_float("SCROOGE_USER_STREAM_STALE_AFTER_SECONDS", 120.0)
    debug_strategy_ticks = _env_flag("SCROOGE_DEBUG_STRATEGY_TICKS", False)
    chart_dataset_path = Path(os.getenv("SCROOGE_RUNTIME_CHART_DATASET_PATH", "runtime/chart_dataset.csv")).expanduser()

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    data_module.set_client(client)
    trade_module.set_client(client)

    # Load or create state
    state = load_state()
    state_path = Path(os.getenv("SCROOGE_STATE_FILE", "runtime/state.json")).expanduser()
    if not state_path.exists():
        save_state(state)
        technical_logger.info("state_initialized path=%s", state_path)

    if live:
        state_lock = threading.RLock()
        control_client = get_control_client()
        restart_requested = False
        last_trading_enabled: bool | None = None
        cache_health_flags = {
            "balance_cache_stale_logged": False,
            "position_cache_stale_logged": False,
        }
        last_balance_refresh_monotonic = 0.0
        last_strategy_candle_open_time: str | None = None
        last_chart_dataset_ts_ms = _read_last_chart_dataset_ts_ms(chart_dataset_path)
        command_kwargs = _build_command_kwargs(symbol)
        runtime_context: dict[str, Any] = {"state": state}

        technical_logger.info("bot_mode_live_started symbol=%s leverage=%s", symbol, lvrg)
        set_leverage(symbol, lvrg)
        live_market_stream = LiveMarketStream(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            leverage=lvrg,
            intervals=intervals,
            limits=limits,
            state_getter=lambda: runtime_context["state"],
            state_lock=state_lock,
            save_state_fn=save_state,
            fetch_historical_fn=fetch_historical,
            prepare_multi_tf_fn=prepare_multi_tf,
            get_balance_fn=get_balance,
            get_open_position_fn=get_open_position,
        )
        if not live_market_stream.start():
            raise RuntimeError("Live market stream failed to start.")

        def resolve_live_balance() -> float:
            cached_balance = get_cached_balance()
            cache_age = get_cached_balance_age_seconds()
            if (
                cached_balance is not None
                and cache_age is not None
                and cache_age <= user_stream_stale_after_seconds
            ):
                if cache_health_flags["balance_cache_stale_logged"]:
                    technical_logger.info(
                        "user_stream_balance_cache_restored age=%.2f threshold=%.2f",
                        cache_age,
                        user_stream_stale_after_seconds,
                    )
                    cache_health_flags["balance_cache_stale_logged"] = False
                return cached_balance

            if (
                cached_balance is not None
                and cache_age is not None
                and not cache_health_flags["balance_cache_stale_logged"]
            ):
                technical_logger.warning(
                    "user_stream_balance_cache_stale age=%.2f threshold=%.2f fallback=rest",
                    cache_age,
                    user_stream_stale_after_seconds,
                )
                cache_health_flags["balance_cache_stale_logged"] = True

            return get_balance()

        def resolve_live_position(current_symbol: str, expect_position: bool) -> dict[str, Any] | None:
            cached_position = get_cached_open_position(current_symbol)
            cache_age = get_cached_position_age_seconds(current_symbol)
            if (
                cached_position is not None
                and cache_age is not None
                and cache_age <= user_stream_stale_after_seconds
            ):
                if cache_health_flags["position_cache_stale_logged"]:
                    technical_logger.info(
                        "user_stream_position_cache_restored symbol=%s age=%.2f threshold=%.2f",
                        current_symbol,
                        cache_age,
                        user_stream_stale_after_seconds,
                    )
                    cache_health_flags["position_cache_stale_logged"] = False
                return cached_position

            if (
                cached_position is not None
                and cache_age is not None
                and expect_position
                and not cache_health_flags["position_cache_stale_logged"]
            ):
                technical_logger.warning(
                    "user_stream_position_cache_stale symbol=%s age=%.2f threshold=%.2f fallback=rest",
                    current_symbol,
                    cache_age,
                    user_stream_stale_after_seconds,
                )
                cache_health_flags["position_cache_stale_logged"] = True

            pos = get_open_position(current_symbol) if expect_position else get_cached_open_position(current_symbol)
            if pos is not None:
                cache_health_flags["position_cache_stale_logged"] = False
            return pos

        try:
            while True:
                try:
                    if control_client is None:
                        control_client = get_control_client()

                    with state_lock:
                        state, restart_now = process_pending_commands(
                            control_client,
                            state,
                            save_state,
                            **command_kwargs,
                        )
                        runtime_context["state"] = state
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
                        if live_market_stream is not None:
                            live_market_stream.update_config(
                                symbol=symbol,
                                leverage=lvrg,
                                intervals=intervals,
                                limits=limits,
                            )
                            if not live_market_stream.is_running():
                                raise RuntimeError("Live market stream failed to restart after config reload.")
                        restart_requested = False
                        cache_health_flags["balance_cache_stale_logged"] = False
                        cache_health_flags["position_cache_stale_logged"] = False
                        last_balance_refresh_monotonic = 0.0
                        last_strategy_candle_open_time = None
                        technical_logger.info("config_restart_applied symbol=%s leverage=%s", symbol, lvrg)

                    with state_lock:
                        trading_enabled = bool(state.get("trading_enabled", True))
                        has_position = isinstance(state.get("position"), dict)

                    if trading_enabled != last_trading_enabled:
                        technical_logger.info(
                            "trading_status_changed status=%s",
                            "running" if trading_enabled else "paused",
                        )
                        last_trading_enabled = trading_enabled

                    now_monotonic = time.monotonic()
                    if (now_monotonic - last_balance_refresh_monotonic) >= live_poll_seconds:
                        current_balance = resolve_live_balance()
                        with state_lock:
                            update_balance(state, current_balance)
                            runtime_context["state"] = state
                        last_balance_refresh_monotonic = now_monotonic
                        technical_logger.debug("live_balance balance=%.2f", current_balance)

                    df = live_market_stream.take_ready_strategy_frame() if live_market_stream is not None else None
                    if df is None:
                        time.sleep(control_poll_slice_seconds)
                        continue

                    latest_candle_open_time = "unknown"
                    try:
                        latest_candle_value = df.iloc[-1]["open_time"]
                        latest_candle_open_time = str(latest_candle_value)
                    except Exception:  # noqa: BLE001
                        latest_candle_open_time = "unknown"

                    if latest_candle_open_time == last_strategy_candle_open_time:
                        if debug_strategy_ticks:
                            technical_logger.info(
                                "live_strategy_tick_duplicate_skipped candle_open_time=%s",
                                latest_candle_open_time,
                            )
                        continue

                    current_balance = resolve_live_balance()
                    with state_lock:
                        update_balance(state, current_balance)
                        runtime_context["state"] = state
                        trading_enabled = bool(state.get("trading_enabled", True))
                        has_position = isinstance(state.get("position"), dict)
                    last_balance_refresh_monotonic = time.monotonic()
                    technical_logger.debug("live_balance balance=%.2f", current_balance)

                    if not trading_enabled and not has_position:
                        technical_logger.debug("live_skip_strategy_on_closed_candle trading_enabled=false has_position=false")
                        continue

                    pos = resolve_live_position(symbol, has_position)
                    if pos:
                        with state_lock:
                            position = state.get("position") if isinstance(state.get("position"), dict) else {}
                        technical_logger.debug(
                            "live_open_position amount=%s symbol=%s tp=%s sl=%s",
                            pos.get("positionAmt"),
                            pos.get("symbol"),
                            position.get("tp", "n/a"),
                            position.get("sl", "n/a"),
                        )
                    else:
                        technical_logger.debug("live_no_open_positions symbol=%s", symbol)

                    if debug_strategy_ticks:
                        technical_logger.info(
                            "live_strategy_tick candle_open_time=%s trading_enabled=%s has_position=%s rows=%s",
                            latest_candle_open_time,
                            trading_enabled,
                            has_position,
                            len(df),
                        )

                    with state_lock:
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
                        runtime_context["state"] = state
                    last_strategy_candle_open_time = latest_candle_open_time
                    last_chart_dataset_ts_ms = _append_latest_chart_candle(
                        df=df,
                        symbol=symbol,
                        path=chart_dataset_path,
                        last_ts_ms=last_chart_dataset_ts_ms,
                        balance=float(balance),
                    )

                except Exception as e:  # noqa: BLE001
                    technical_logger.exception("live_loop_error error=%s", e)
                    time.sleep(max(1, control_poll_slice_seconds))
        finally:
            if live_market_stream is not None:
                live_market_stream.stop()
                live_market_stream = None

    else:
        from backtest.runner import build_discrete_backtest_config, run_discrete_backtest

        backtest_config = build_discrete_backtest_config(
            cfg,
            chart_dataset_path=chart_dataset_path,
            event_log_path=os.getenv("SCROOGE_EVENT_LOG_FILE", "runtime/event_history.jsonl"),
            runtime_mode=os.getenv("SCROOGE_RUNTIME_MODE", "backtest"),
            strategy_mode=os.getenv("SCROOGE_STRATEGY_MODE", "discrete"),
            client=client,
        )
        run_discrete_backtest(
            backtest_config,
            technical_logger=technical_logger,
        )
