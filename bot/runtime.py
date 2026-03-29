import csv
import math
import os
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from dotenv import load_dotenv

import bot.trade as trade_module
import backtest.dataset as data_module
from bot.control_channel import get_control_client, process_pending_commands
from backtest.dataset import fetch_historical
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
    get_order_execution_summary,
    set_leverage,
)
from core.event_store import reset_event_store
from core.engine import initialize_realtime_strategy_processor, run_strategy_on_snapshot
from core.indicator_inputs import normalize_indicator_inputs
from core.binance_retry import create_binance_client

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


def _backtest_run_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_symlink_latest(root_dir: Path, run_dir: Path) -> None:
    latest_link = root_dir / "latest"
    try:
        latest_link.parent.mkdir(parents=True, exist_ok=True)
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.resolve())
    except OSError as exc:
        technical_logger.warning("backtest_run_latest_link_failed root=%s run=%s error=%s", root_dir, run_dir, exc)


def _resolve_backtest_run_dir(cfg: dict[str, Any]) -> Path | None:
    raw_run_dir = str(os.getenv("SCROOGE_BACKTEST_RUN_DIR", cfg.get("backtest_run_dir", "")) or "").strip()
    raw_root_dir = str(os.getenv("SCROOGE_BACKTEST_ROOT", cfg.get("backtest_run_root", "runtime/backtests")) or "").strip()

    if not raw_run_dir:
        return None

    if raw_run_dir.lower() != "auto":
        return Path(raw_run_dir).expanduser().resolve()

    root_dir = Path(raw_root_dir or "runtime/backtests").expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root_dir / _backtest_run_timestamp()
    return run_dir


def _apply_backtest_run_dir(run_dir: Path, cfg: dict[str, Any]) -> tuple[Path, bool]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SCROOGE_BACKTEST_RUN_DIR"] = str(resolved_run_dir)

    os.environ["SCROOGE_STATE_FILE"] = str(resolved_run_dir / "state.json")
    os.environ["SCROOGE_TRADE_HISTORY_FILE"] = str(resolved_run_dir / "trade_history.jsonl")
    os.environ["SCROOGE_BALANCE_HISTORY_FILE"] = str(resolved_run_dir / "balance_history.jsonl")
    os.environ["SCROOGE_LOG_FILE"] = str(resolved_run_dir / "trading_log.txt")
    os.environ["SCROOGE_EVENT_LOG_FILE"] = str(resolved_run_dir / "event_history.jsonl")
    os.environ["SCROOGE_MARKET_EVENT_STREAM_FILE"] = str(resolved_run_dir / "market_events.jsonl")
    os.environ["SCROOGE_RUNTIME_CHART_DATASET_PATH"] = str(resolved_run_dir / "chart_dataset.csv")

    reset_event_store(os.environ["SCROOGE_EVENT_LOG_FILE"])

    raw_root_dir = str(os.getenv("SCROOGE_BACKTEST_ROOT", cfg.get("backtest_run_root", "runtime/backtests")) or "").strip()
    root_dir = Path(raw_root_dir or "runtime/backtests").expanduser().resolve()
    latest_updated = False
    try:
        if resolved_run_dir.parent == root_dir or resolved_run_dir.is_relative_to(root_dir):
            _safe_symlink_latest(root_dir, resolved_run_dir)
            latest_updated = True
    except AttributeError:
        try:
            resolved_run_dir.relative_to(root_dir)
            _safe_symlink_latest(root_dir, resolved_run_dir)
            latest_updated = True
        except ValueError:
            latest_updated = False

    return resolved_run_dir, latest_updated


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


def _resolve_runtime_event_log_file() -> Path:
    raw_event_log_path = str(os.getenv("SCROOGE_EVENT_LOG_FILE", "") or "").strip()
    if raw_event_log_path:
        return Path(raw_event_log_path).expanduser()

    log_path = Path(os.getenv("SCROOGE_LOG_FILE", "runtime/trading_log.txt")).expanduser()
    return log_path.parent / "event_history.jsonl"


def _ensure_runtime_event_log_file() -> None:
    """
    Ensure canonical event log path is always a real file under the runtime directory.
    """
    event_log_path = _resolve_runtime_event_log_file()
    os.environ["SCROOGE_EVENT_LOG_FILE"] = str(event_log_path)
    try:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.touch(exist_ok=True)
        reset_event_store(event_log_path)
    except OSError as exc:
        technical_logger.warning("runtime_event_log_init_failed path=%s error=%s", event_log_path, exc)


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


def _row_value(row: Any, key: str) -> Any:
    if row is None:
        return None
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(row, key, None)


def _resolve_latest_row(payload: Any) -> Any:
    if payload is None:
        return None
    iloc = getattr(payload, "iloc", None)
    if iloc is not None:
        try:
            return iloc[-1]
        except Exception:  # noqa: BLE001
            return None
    return payload


def _append_latest_chart_candle(
    row: Any,
    symbol: str,
    path: Path,
    last_ts_ms: int | None,
    balance: float | None = None,
) -> int | None:
    latest = _resolve_latest_row(row)
    if latest is None:
        return last_ts_ms

    ts_ms = _parse_open_time_to_ms(_row_value(latest, "open_time"))
    if ts_ms is None:
        return last_ts_ms
    if last_ts_ms is not None and ts_ms <= last_ts_ms:
        return last_ts_ms

    open_price = _row_value(latest, "open")
    high_price = _row_value(latest, "high")
    low_price = _row_value(latest, "low")
    close_price = _row_value(latest, "close")
    volume = _row_value(latest, "volume")
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
            "open_time": _format_open_time(_row_value(latest, "open_time")),
            "symbol": symbol,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
            "balance": _coerce_optional_float(balance),
            "EMA": _coerce_optional_float(_row_value(latest, "EMA")),
            "RSI": _coerce_optional_float(_row_value(latest, "RSI")),
            "BBL": _coerce_optional_float(_row_value(latest, "BBL")),
            "BBM": _coerce_optional_float(_row_value(latest, "BBM")),
            "BBU": _coerce_optional_float(_row_value(latest, "BBU")),
            "ATR": _coerce_optional_float(_row_value(latest, "ATR")),
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


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _sync_exchange_position_snapshot_into_state(
    current_state: dict[str, Any],
    *,
    symbol: str,
    position_snapshot: dict[str, Any],
    ts_label: str,
) -> None:
    position_amt = _optional_float(position_snapshot.get("positionAmt"))
    entry_price = _optional_float(position_snapshot.get("entryPrice"))
    unrealized_pnl = _optional_float(position_snapshot.get("unRealizedProfit"))
    isolated_margin = (
        _optional_float(position_snapshot.get("isolatedWallet"))
        or _optional_float(position_snapshot.get("isolatedMargin"))
    )
    mark_price = _optional_float(position_snapshot.get("markPrice"))
    liq_price = _optional_float(position_snapshot.get("liquidationPrice"))
    break_even_price = _optional_float(position_snapshot.get("breakEvenPrice"))
    position_side = position_snapshot.get("positionSide")

    current_state["exchange_position"] = {
        "symbol": symbol,
        "position_amt": position_amt,
        "entry_price": entry_price,
        "unrealized_pnl": unrealized_pnl,
        "position_side": position_side,
        "isolated_margin": isolated_margin,
        "mark_price": mark_price,
        "liq_price": liq_price,
        "break_even_price": break_even_price,
        "updated_at": ts_label,
        "source": "rest_position_snapshot",
    }

    current_position = current_state.get("position")
    if not isinstance(current_position, dict):
        return

    current_position["exchange_position_amt"] = position_amt
    current_position["exchange_entry_price"] = entry_price
    current_position["exchange_unrealized_pnl"] = unrealized_pnl
    current_position["exchange_position_side"] = position_side
    current_position["exchange_isolated_margin"] = isolated_margin
    current_position["exchange_mark_price"] = mark_price
    current_position["exchange_liq_price"] = liq_price
    current_position["exchange_break_even_price"] = break_even_price
    current_position["exchange_position_updated_at"] = ts_label


def _build_command_kwargs(symbol: str, *, leverage: float, fee_rate: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "close_position_fn": close_position,
        "get_open_position_fn": get_open_position,
        "get_balance_fn": get_balance,
        "update_position_fn": update_position,
        "update_balance_fn": update_balance,
        "add_closed_trade_fn": add_closed_trade,
        "get_order_execution_summary_fn": get_order_execution_summary,
        "leverage": leverage,
        "fee_rate": fee_rate,
    }


def _resolve_strategy_mode(cfg: dict[str, Any]) -> str:
    normalized = str(
        cfg.get("strategy_mode", os.getenv("SCROOGE_STRATEGY_MODE", "discrete"))
    ).strip().lower() or "discrete"
    if normalized not in {"discrete", "realtime"}:
        raise ValueError("strategy_mode must be one of: discrete, realtime")
    return normalized


def _resolve_live_indicator_inputs(cfg: dict[str, Any], *, strategy_mode: str) -> dict[str, str]:
    return normalize_indicator_inputs(
        cfg.get("indicator_inputs"),
        strategy_mode=strategy_mode,
    )


if __name__ == "__main__":
    cfg = load_config()

    live = cfg["live"]  # "backtest" or "live"
    if not live:
        backtest_run_dir = _resolve_backtest_run_dir(cfg)
        if backtest_run_dir is not None:
            resolved_run_dir, latest_updated = _apply_backtest_run_dir(backtest_run_dir, cfg)
            technical_logger.info(
                "backtest_run_directory_prepared path=%s latest_updated=%s",
                resolved_run_dir,
                latest_updated,
            )

    _ensure_runtime_log_file()
    _ensure_runtime_event_log_file()

    symbol = cfg["symbol"]
    lvrg = cfg["leverage"]
    initial_balance = cfg.get("initial_balance")
    qty = cfg["qty"]
    use_full_balance = cfg["use_full_balance"]
    strategy_mode = _resolve_strategy_mode(cfg)

    intervals = cfg["intervals"]
    indicator_inputs = _resolve_live_indicator_inputs(cfg, strategy_mode=strategy_mode)

    params = cfg["params"]

    live_poll_seconds = _env_int("SCROOGE_LIVE_POLL_SECONDS", 60)
    control_poll_slice_seconds = _env_int("SCROOGE_CONTROL_POLL_SLICE_SECONDS", 1)
    user_stream_stale_after_seconds = _env_float("SCROOGE_USER_STREAM_STALE_AFTER_SECONDS", 120.0)
    debug_strategy_ticks = _env_flag("SCROOGE_DEBUG_STRATEGY_TICKS", False)
    chart_dataset_path = Path(os.getenv("SCROOGE_RUNTIME_CHART_DATASET_PATH", "runtime/chart_dataset.csv")).expanduser()

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = create_binance_client(api_key, api_secret, logger=technical_logger)
    data_module.set_client(client)
    trade_module.set_client(client)

    if live:
        # Load or create state
        state = load_state()
        state_path = Path(os.getenv("SCROOGE_STATE_FILE", "runtime/state.json")).expanduser()
        if not state_path.exists():
            save_state(state)
            technical_logger.info("state_initialized path=%s", state_path)

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
        command_kwargs = _build_command_kwargs(symbol, leverage=lvrg, fee_rate=0.0005)
        runtime_context: dict[str, Any] = {"state": state}

        technical_logger.info(
            "bot_mode_live_started symbol=%s leverage=%s strategy_mode=%s",
            symbol,
            lvrg,
            strategy_mode,
        )
        set_leverage(symbol, lvrg)
        live_market_stream = LiveMarketStream(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            leverage=lvrg,
            intervals=intervals,
            state_getter=lambda: runtime_context["state"],
            state_lock=state_lock,
            save_state_fn=save_state,
            fetch_historical_fn=fetch_historical,
            indicator_inputs=indicator_inputs,
            get_balance_fn=get_balance,
            get_open_position_fn=get_open_position,
        )
        if not live_market_stream.start():
            raise RuntimeError("Live market stream failed to start.")

        def resolve_live_balance() -> float:
            cached_balance = get_cached_balance()
            cache_age = get_cached_balance_age_seconds()
            user_stream_connected = live_market_stream.is_user_stream_connected() if live_market_stream is not None else False
            if (
                cached_balance is not None
                and (
                    user_stream_connected
                    or (cache_age is not None and cache_age <= user_stream_stale_after_seconds)
                )
            ):
                if cache_health_flags["balance_cache_stale_logged"]:
                    technical_logger.info(
                        "user_stream_balance_cache_restored age=%.2f threshold=%.2f connected=%s",
                        -1.0 if cache_age is None else cache_age,
                        user_stream_stale_after_seconds,
                        user_stream_connected,
                    )
                    cache_health_flags["balance_cache_stale_logged"] = False
                return cached_balance

            if (
                cached_balance is not None
                and cache_age is not None
                and not user_stream_connected
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
            user_stream_connected = live_market_stream.is_user_stream_connected() if live_market_stream is not None else False
            if (
                cached_position is not None
                and (
                    user_stream_connected
                    or (cache_age is not None and cache_age <= user_stream_stale_after_seconds)
                )
            ):
                if cache_health_flags["position_cache_stale_logged"]:
                    technical_logger.info(
                        "user_stream_position_cache_restored symbol=%s age=%.2f threshold=%.2f connected=%s",
                        current_symbol,
                        -1.0 if cache_age is None else cache_age,
                        user_stream_stale_after_seconds,
                        user_stream_connected,
                    )
                    cache_health_flags["position_cache_stale_logged"] = False
                return cached_position

            if (
                cached_position is not None
                and cache_age is not None
                and expect_position
                and not user_stream_connected
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
                with state_lock:
                    current_state = runtime_context.get("state")
                    if isinstance(current_state, dict):
                        _sync_exchange_position_snapshot_into_state(
                            current_state,
                            symbol=current_symbol,
                            position_snapshot=pos,
                            ts_label=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        runtime_context["state"] = current_state
                cache_health_flags["position_cache_stale_logged"] = False
            return pos

        def initialize_live_realtime_processor(current_state: dict[str, Any]) -> Any:
            if live_market_stream is None:
                raise RuntimeError("Live market stream must be running before realtime strategy initialization.")
            current_balance = resolve_live_balance()
            processor = initialize_realtime_strategy_processor(
                live=True,
                initial_balance=current_balance,
                qty=qty,
                sl_mult=params["sl_mult"],
                tp_mult=params["tp_mult"],
                symbol=symbol,
                leverage=lvrg,
                use_full_balance=use_full_balance,
                fee_rate=0.0005,
                state=current_state,
                use_state=True,
                enable_logs=True,
                rsi_extreme_long=params["rsi_extreme_long"],
                rsi_extreme_short=params["rsi_extreme_short"],
                rsi_long_open_threshold=params["rsi_long_open_threshold"],
                rsi_long_qty_threshold=params["rsi_long_qty_threshold"],
                rsi_long_tp_threshold=params["rsi_long_tp_threshold"],
                rsi_long_close_threshold=params["rsi_long_close_threshold"],
                rsi_short_open_threshold=params["rsi_short_open_threshold"],
                rsi_short_qty_threshold=params["rsi_short_qty_threshold"],
                rsi_short_tp_threshold=params["rsi_short_tp_threshold"],
                rsi_short_close_threshold=params["rsi_short_close_threshold"],
                trail_atr_mult=params["trail_atr_mult"],
                allow_entries=bool(current_state.get("trading_enabled", True)),
                execution_mode="simulated",
                runtime_mode="live",
                indicator_inputs=indicator_inputs,
                intervals={key: str(value) for key, value in intervals.items()},
                target_symbol=symbol,
                emit_on_price_tick=True,
            )
            df_small, df_medium, df_big = live_market_stream.snapshot_candle_frames()
            processor.bootstrap_from_frames(
                df_small=df_small,
                df_medium=df_medium,
                df_big=df_big,
            )
            return processor

        realtime_processor = (
            initialize_live_realtime_processor(state)
            if strategy_mode == "realtime"
            else None
        )

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
                        strategy_mode = _resolve_strategy_mode(cfg)
                        intervals = cfg["intervals"]
                        indicator_inputs = _resolve_live_indicator_inputs(cfg, strategy_mode=strategy_mode)
                        params = cfg["params"]
                        set_leverage(symbol, lvrg)
                        command_kwargs = _build_command_kwargs(symbol, leverage=lvrg, fee_rate=0.0005)
                        if live_market_stream is not None:
                            live_market_stream.update_config(
                                symbol=symbol,
                                leverage=lvrg,
                                intervals=intervals,
                                indicator_inputs=indicator_inputs,
                            )
                            if not live_market_stream.is_running():
                                raise RuntimeError("Live market stream failed to restart after config reload.")
                            live_market_stream.take_pending_market_events()
                        realtime_processor = (
                            initialize_live_realtime_processor(state)
                            if strategy_mode == "realtime"
                            else None
                        )
                        restart_requested = False
                        cache_health_flags["balance_cache_stale_logged"] = False
                        cache_health_flags["position_cache_stale_logged"] = False
                        last_balance_refresh_monotonic = 0.0
                        last_strategy_candle_open_time = None
                        technical_logger.info(
                            "config_restart_applied symbol=%s leverage=%s strategy_mode=%s",
                            symbol,
                            lvrg,
                            strategy_mode,
                        )

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

                    if strategy_mode == "realtime":
                        if realtime_processor is None:
                            raise RuntimeError("Realtime live strategy processor is not initialized.")
                        events = live_market_stream.take_pending_market_events() if live_market_stream is not None else []
                        if not events:
                            time.sleep(control_poll_slice_seconds)
                            continue

                        with state_lock:
                            trading_enabled = bool(state.get("trading_enabled", True))
                            has_position = isinstance(state.get("position"), dict)

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
                            technical_logger.debug(
                                "live_strategy_events count=%s trading_enabled=%s has_position=%s",
                                len(events),
                                trading_enabled,
                                has_position,
                            )

                        with state_lock:
                            for event in events:
                                realtime_processor.process_event(event)
                            state = realtime_processor.runtime.state
                            runtime_context["state"] = state
                            chart_balance = state.get("balance")
                        chart_row = live_market_stream.take_ready_strategy_row() if live_market_stream is not None else None
                        if chart_row is not None:
                            try:
                                chart_balance_value = float(chart_balance) if chart_balance is not None else None
                            except (TypeError, ValueError):
                                chart_balance_value = None
                            last_chart_dataset_ts_ms = _append_latest_chart_candle(
                                row=chart_row,
                                symbol=symbol,
                                path=chart_dataset_path,
                                last_ts_ms=last_chart_dataset_ts_ms,
                                balance=chart_balance_value,
                            )
                        continue

                    row = live_market_stream.take_ready_strategy_row() if live_market_stream is not None else None
                    if row is None:
                        time.sleep(control_poll_slice_seconds)
                        continue

                    latest_candle_open_time = str(_row_value(row, "open_time") or "unknown")

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
                            "live_strategy_tick candle_open_time=%s trading_enabled=%s has_position=%s",
                            latest_candle_open_time,
                            trading_enabled,
                            has_position,
                        )

                    with state_lock:
                        balance, trades, balance_history, state = run_strategy_on_snapshot(
                            row,
                            live,
                            current_balance,
                            qty,
                            symbol=symbol,
                            leverage=lvrg,
                            use_full_balance=use_full_balance,
                            state=state,
                            allow_entries=trading_enabled,
                            indicator_inputs=indicator_inputs,
                            **params,
                        )
                        runtime_context["state"] = state
                    last_strategy_candle_open_time = latest_candle_open_time
                    last_chart_dataset_ts_ms = _append_latest_chart_candle(
                        row=row,
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
        from backtest.runner import build_backtest_config, run_backtest

        backtest_config = build_backtest_config(
            cfg,
            chart_dataset_path=chart_dataset_path,
            event_log_path=os.getenv("SCROOGE_EVENT_LOG_FILE", "runtime/event_history.jsonl"),
            runtime_mode="backtest",
            client=client,
        )
        run_backtest(
            backtest_config,
            technical_logger=technical_logger,
        )
