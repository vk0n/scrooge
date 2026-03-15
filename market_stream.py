from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from event_log import get_technical_logger
from state import update_balance, update_position
from strategy import refresh_runtime_state_from_price_tick
from trade import (
    clear_runtime_account_cache,
    set_cached_balance,
    set_cached_position,
)

try:
    from binance import ThreadedWebsocketManager
    from binance.enums import FuturesType
except ImportError:  # pragma: no cover - dependency installed in bot image
    ThreadedWebsocketManager = None
    FuturesType = None

try:
    import websocket
except ImportError:  # pragma: no cover - dependency installed in bot image
    websocket = None


technical_logger = get_technical_logger()
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MARKET_STREAM_ENABLED = os.getenv("SCROOGE_MARKET_STREAM_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
MARKET_STREAM_PERSIST_INTERVAL_SECONDS = max(
    0.25,
    float(os.getenv("SCROOGE_MARKET_STREAM_PERSIST_INTERVAL_SECONDS", "1")),
)
MARKET_STREAM_SETTLE_SECONDS = max(
    0.25,
    float(os.getenv("SCROOGE_MARKET_STREAM_SETTLE_SECONDS", "1.5")),
)
USER_STREAM_ENABLED = os.getenv("SCROOGE_USER_STREAM_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
USER_STREAM_KEEPALIVE_SECONDS = max(
    60,
    int(float(os.getenv("SCROOGE_USER_STREAM_KEEPALIVE_SECONDS", "1800"))),
)
USER_STREAM_RECONNECT_SECONDS = max(
    1.0,
    float(os.getenv("SCROOGE_USER_STREAM_RECONNECT_SECONDS", "5")),
)
FUTURES_REST_BASE_URL = os.getenv("SCROOGE_FUTURES_REST_BASE_URL", "https://fapi.binance.com").rstrip("/")
FUTURES_WS_BASE_URL = os.getenv("SCROOGE_FUTURES_WS_BASE_URL", "wss://fstream.binance.com/ws").rstrip("/")


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:  # NaN check without math import
        return None
    return numeric


def _to_int(value: Any, fallback: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if numeric > 0 else fallback


def _format_event_timestamp_ms(value: Any) -> str:
    try:
        ts_ms = int(value)
    except (TypeError, ValueError):
        return datetime.now().strftime(TIMESTAMP_FORMAT)
    try:
        return datetime.fromtimestamp(ts_ms / 1000).strftime(TIMESTAMP_FORMAT)
    except (OSError, OverflowError, ValueError):
        return datetime.now().strftime(TIMESTAMP_FORMAT)


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    text = str(interval).strip().lower()
    if not text:
        raise ValueError("Interval must not be empty.")

    unit = text[-1]
    value = int(text[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    raise ValueError(f"Unsupported Binance interval: {interval}")


def _empty_candle_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])


def _normalize_candle_frame(df: pd.DataFrame | None, limit: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_candle_frame()

    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"])
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open_time", "open", "high", "low", "close"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").tail(limit)
    out.reset_index(drop=True, inplace=True)
    return out[["open_time", "open", "high", "low", "close", "volume"]]


def _candle_row_from_kline(kline: dict[str, Any]) -> dict[str, Any] | None:
    try:
        open_time_ms = int(kline["t"])
    except (KeyError, TypeError, ValueError):
        return None

    open_price = _to_float(kline.get("o"))
    high_price = _to_float(kline.get("h"))
    low_price = _to_float(kline.get("l"))
    close_price = _to_float(kline.get("c"))
    volume = _to_float(kline.get("v")) or 0.0
    if open_price is None or high_price is None or low_price is None or close_price is None:
        return None

    return {
        "open_time": pd.to_datetime(open_time_ms, unit="ms"),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
    }


class LiveMarketStream:
    def __init__(
        self,
        *,
        api_key: str | None,
        api_secret: str | None,
        symbol: str,
        leverage: float | int,
        intervals: dict[str, Any],
        limits: dict[str, Any],
        state_getter: Callable[[], dict[str, Any]],
        state_lock: threading.RLock,
        save_state_fn: Callable[[dict[str, Any]], None],
        fetch_historical_fn: Callable[[str, str, int], pd.DataFrame],
        prepare_multi_tf_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame],
        get_balance_fn: Callable[[], float],
        get_open_position_fn: Callable[[str], dict[str, Any] | None],
    ) -> None:
        self._api_key = api_key or None
        self._api_secret = api_secret or None
        self._symbol = str(symbol).upper()
        self._leverage = leverage
        self._state_getter = state_getter
        self._state_lock = state_lock
        self._save_state_fn = save_state_fn
        self._fetch_historical_fn = fetch_historical_fn
        self._prepare_multi_tf_fn = prepare_multi_tf_fn
        self._get_balance_fn = get_balance_fn
        self._get_open_position_fn = get_open_position_fn

        self._intervals = {
            "small": str(intervals["small"]),
            "medium": str(intervals["medium"]),
            "big": str(intervals["big"]),
        }
        self._limits = {
            "small": _to_int(limits.get("small"), 1500),
            "medium": _to_int(limits.get("medium"), 500),
            "big": _to_int(limits.get("big"), 100),
        }
        self._interval_deltas = {
            key: _interval_to_timedelta(value)
            for key, value in self._intervals.items()
        }

        self._cache_lock = threading.RLock()
        self._small_df = _empty_candle_frame()
        self._medium_df = _empty_candle_frame()
        self._big_df = _empty_candle_frame()
        self._needs_resync = False
        self._pending_small_open_time_ms: int | None = None
        self._pending_emit_after_monotonic: float | None = None
        self._last_emitted_small_open_time_ms: int | None = None

        self._twm: ThreadedWebsocketManager | None = None
        self._market_stream_name: str | None = None
        self._user_stream_name: str | None = None
        self._user_listen_key: str | None = None
        self._user_socket_app: Any = None
        self._user_socket_thread: threading.Thread | None = None
        self._user_keepalive_thread: threading.Thread | None = None
        self._user_stop_event = threading.Event()
        self._last_ticker_price: float | None = None
        self._last_mark_price: float | None = None
        self._last_persist_monotonic: float | None = None
        self._running = False

    def start(self) -> bool:
        if self._running:
            return True

        if not MARKET_STREAM_ENABLED:
            technical_logger.info("market_stream_disabled_by_env symbol=%s", self._symbol)
            return False

        if ThreadedWebsocketManager is None or FuturesType is None:
            technical_logger.warning("market_stream_unavailable reason=python_binance_missing")
            return False

        clear_runtime_account_cache()
        self._bootstrap_account_cache()
        self._bootstrap_candle_caches()

        streams = list(
            dict.fromkeys(
                [
                    f"{self._symbol.lower()}@ticker",
                    f"{self._symbol.lower()}@markPrice@1s",
                    f"{self._symbol.lower()}@kline_{self._intervals['small']}",
                    f"{self._symbol.lower()}@kline_{self._intervals['medium']}",
                    f"{self._symbol.lower()}@kline_{self._intervals['big']}",
                ]
            )
        )

        twm = None
        try:
            twm = ThreadedWebsocketManager(api_key=self._api_key, api_secret=self._api_secret)
            twm.start()
            market_stream_name = twm.start_futures_multiplex_socket(
                callback=self._handle_message,
                streams=streams,
                futures_type=FuturesType.USD_M,
            )
            user_stream_name = "raw_listen_key_socket" if self._start_user_stream() else None
        except Exception as exc:  # noqa: BLE001
            technical_logger.exception("market_stream_start_failed symbol=%s error=%s", self._symbol, exc)
            if twm is not None:
                try:
                    twm.stop()
                except Exception:  # noqa: BLE001
                    pass
            return False

        self._twm = twm
        self._market_stream_name = market_stream_name
        self._user_stream_name = user_stream_name
        self._last_persist_monotonic = None
        self._running = True
        technical_logger.info(
            "market_stream_started symbol=%s streams=%s user_stream=%s persist_interval=%s settle=%s",
            self._symbol,
            ",".join(streams),
            bool(user_stream_name),
            MARKET_STREAM_PERSIST_INTERVAL_SECONDS,
            MARKET_STREAM_SETTLE_SECONDS,
        )
        return True

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        twm = self._twm
        market_stream_name = self._market_stream_name
        user_stream_name = self._user_stream_name
        self._running = False
        self._twm = None
        self._market_stream_name = None
        self._user_stream_name = None
        self._last_persist_monotonic = None
        self._stop_user_stream()

        if twm is None:
            return

        try:
            if market_stream_name:
                twm.stop_socket(market_stream_name)
            if user_stream_name:
                twm.stop_socket(user_stream_name)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("market_stream_socket_stop_failed symbol=%s error=%s", self._symbol, exc)

        try:
            twm.stop()
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("market_stream_stop_failed symbol=%s error=%s", self._symbol, exc)
        else:
            technical_logger.info("market_stream_stopped symbol=%s", self._symbol)

    def update_config(
        self,
        *,
        symbol: str,
        leverage: float | int,
        intervals: dict[str, Any],
        limits: dict[str, Any],
    ) -> None:
        next_symbol = str(symbol).upper()
        next_leverage = leverage
        next_intervals = {
            "small": str(intervals["small"]),
            "medium": str(intervals["medium"]),
            "big": str(intervals["big"]),
        }
        next_limits = {
            "small": _to_int(limits.get("small"), 1500),
            "medium": _to_int(limits.get("medium"), 500),
            "big": _to_int(limits.get("big"), 100),
        }

        symbol_changed = next_symbol != self._symbol
        intervals_changed = next_intervals != self._intervals
        limits_changed = next_limits != self._limits

        self._leverage = next_leverage
        if not symbol_changed and not intervals_changed and not limits_changed:
            return

        was_running = self._running
        self.stop()
        self._symbol = next_symbol
        self._intervals = next_intervals
        self._limits = next_limits
        self._interval_deltas = {
            key: _interval_to_timedelta(value)
            for key, value in self._intervals.items()
        }
        self._last_ticker_price = None
        self._last_mark_price = None

        with self._cache_lock:
            self._small_df = _empty_candle_frame()
            self._medium_df = _empty_candle_frame()
            self._big_df = _empty_candle_frame()
            self._needs_resync = False
            self._pending_small_open_time_ms = None
            self._pending_emit_after_monotonic = None
            self._last_emitted_small_open_time_ms = None

        if was_running:
            self.start()

    def take_ready_strategy_frame(self) -> pd.DataFrame | None:
        pending_open_time_ms: int | None = None
        should_resync = False

        with self._cache_lock:
            if self._pending_small_open_time_ms is None or self._pending_emit_after_monotonic is None:
                return None
            if time.monotonic() < self._pending_emit_after_monotonic:
                return None
            pending_open_time_ms = self._pending_small_open_time_ms
            should_resync = self._needs_resync

        if should_resync:
            technical_logger.warning("market_stream_resync_started symbol=%s", self._symbol)
            self._bootstrap_candle_caches()

        with self._cache_lock:
            if (
                pending_open_time_ms is None
                or self._pending_small_open_time_ms != pending_open_time_ms
                or self._last_emitted_small_open_time_ms == pending_open_time_ms
            ):
                return None
            if self._small_df.empty or self._medium_df.empty or self._big_df.empty:
                return None

            df_small = self._small_df.copy()
            df_medium = self._medium_df.copy()
            df_big = self._big_df.copy()
            self._last_emitted_small_open_time_ms = pending_open_time_ms
            self._pending_small_open_time_ms = None
            self._pending_emit_after_monotonic = None

        return self._prepare_multi_tf_fn(df_small, df_medium, df_big)

    def _bootstrap_account_cache(self) -> None:
        try:
            balance = self._get_balance_fn()
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("user_stream_balance_bootstrap_failed symbol=%s error=%s", self._symbol, exc)
        else:
            set_cached_balance(balance)

        try:
            position = self._get_open_position_fn(self._symbol)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("user_stream_position_bootstrap_failed symbol=%s error=%s", self._symbol, exc)
        else:
            set_cached_position(self._symbol, position)

    def _start_user_stream(self) -> bool:
        if not USER_STREAM_ENABLED:
            technical_logger.info("user_stream_disabled_by_env symbol=%s", self._symbol)
            return False

        if websocket is None:
            technical_logger.warning("user_stream_unavailable reason=websocket_client_missing")
            return False

        if not self._api_key:
            technical_logger.warning("user_stream_unavailable reason=missing_api_key")
            return False

        self._user_stop_event.clear()
        self._user_socket_thread = threading.Thread(
            target=self._run_user_socket_loop,
            name=f"scrooge-user-stream-{self._symbol.lower()}",
            daemon=True,
        )
        self._user_keepalive_thread = threading.Thread(
            target=self._run_user_keepalive_loop,
            name=f"scrooge-user-keepalive-{self._symbol.lower()}",
            daemon=True,
        )
        self._user_socket_thread.start()
        self._user_keepalive_thread.start()
        return True

    def _stop_user_stream(self) -> None:
        self._user_stop_event.set()

        socket_app = self._user_socket_app
        self._user_socket_app = None
        if socket_app is not None:
            try:
                socket_app.close()
            except Exception as exc:  # noqa: BLE001
                technical_logger.warning("user_stream_socket_close_failed symbol=%s error=%s", self._symbol, exc)

        for thread_obj in (self._user_socket_thread, self._user_keepalive_thread):
            if thread_obj is not None and thread_obj.is_alive():
                thread_obj.join(timeout=2.0)

        self._user_socket_thread = None
        self._user_keepalive_thread = None

        listen_key = self._user_listen_key
        self._user_listen_key = None
        if listen_key:
            self._close_listen_key(listen_key)

    def _request_listen_key(self, method: str, *, listen_key: str | None = None) -> dict[str, Any] | None:
        url = f"{FUTURES_REST_BASE_URL}/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self._api_key or ""}
        request = urllib.request.Request(url, method=method.upper(), headers=headers)
        if listen_key:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
            payload = f"listenKey={listen_key}".encode("utf-8")
        else:
            payload = b""

        try:
            with urllib.request.urlopen(request, data=payload, timeout=10) as response:
                body = response.read().decode("utf-8").strip()
        except urllib.error.HTTPError as exc:
            technical_logger.warning(
                "user_stream_listen_key_request_failed symbol=%s method=%s status=%s error=%s",
                self._symbol,
                method.upper(),
                exc.code,
                exc.reason,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning(
                "user_stream_listen_key_request_failed symbol=%s method=%s error=%s",
                self._symbol,
                method.upper(),
                exc,
            )
            return None

        if not body:
            return {}
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            technical_logger.warning(
                "user_stream_listen_key_invalid_response symbol=%s method=%s body=%s",
                self._symbol,
                method.upper(),
                body,
            )
            return None
        if isinstance(data, dict):
            return data
        technical_logger.warning(
            "user_stream_listen_key_invalid_payload symbol=%s method=%s payload=%s",
            self._symbol,
            method.upper(),
            data,
        )
        return None

    def _create_listen_key(self) -> str | None:
        payload = self._request_listen_key("POST")
        if not isinstance(payload, dict):
            return None
        listen_key = str(payload.get("listenKey", "")).strip()
        if not listen_key:
            technical_logger.warning("user_stream_listen_key_missing symbol=%s", self._symbol)
            return None
        return listen_key

    def _keepalive_listen_key(self, listen_key: str) -> bool:
        payload = self._request_listen_key("PUT", listen_key=listen_key)
        return payload is not None

    def _close_listen_key(self, listen_key: str) -> None:
        self._request_listen_key("DELETE", listen_key=listen_key)

    def _run_user_socket_loop(self) -> None:
        while not self._user_stop_event.is_set():
            listen_key = self._user_listen_key
            if not listen_key:
                listen_key = self._create_listen_key()
                if not listen_key:
                    if self._user_stop_event.wait(USER_STREAM_RECONNECT_SECONDS):
                        return
                    continue
                self._user_listen_key = listen_key

            ws_url = f"{FUTURES_WS_BASE_URL}/{listen_key}"
            socket_app = websocket.WebSocketApp(
                ws_url,
                on_open=self._handle_user_socket_open,
                on_message=self._handle_user_socket_message,
                on_error=self._handle_user_socket_error,
                on_close=self._handle_user_socket_close,
            )
            self._user_socket_app = socket_app

            try:
                socket_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:  # noqa: BLE001
                technical_logger.warning("user_stream_run_failed symbol=%s error=%s", self._symbol, exc)
            finally:
                self._user_socket_app = None

            if self._user_stop_event.is_set():
                break

            technical_logger.warning("user_stream_reconnecting symbol=%s delay=%s", self._symbol, USER_STREAM_RECONNECT_SECONDS)
            if self._user_stop_event.wait(USER_STREAM_RECONNECT_SECONDS):
                break

    def _run_user_keepalive_loop(self) -> None:
        while not self._user_stop_event.wait(USER_STREAM_KEEPALIVE_SECONDS):
            listen_key = self._user_listen_key
            if not listen_key:
                continue
            if not self._keepalive_listen_key(listen_key):
                technical_logger.warning("user_stream_keepalive_failed symbol=%s", self._symbol)
                self._user_listen_key = None
                if self._user_socket_app is not None:
                    try:
                        self._user_socket_app.close()
                    except Exception:  # noqa: BLE001
                        pass

    def _handle_user_socket_open(self, ws_app: Any) -> None:  # noqa: ARG002
        technical_logger.info("user_stream_connected symbol=%s", self._symbol)

    def _handle_user_socket_message(self, ws_app: Any, raw_message: Any) -> None:  # noqa: ARG002
        try:
            message = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
        except json.JSONDecodeError:
            technical_logger.warning("user_stream_message_invalid_json symbol=%s raw=%s", self._symbol, raw_message)
            return
        self._handle_user_message(message)

    def _handle_user_socket_error(self, ws_app: Any, error: Any) -> None:  # noqa: ARG002
        if self._user_stop_event.is_set():
            return
        technical_logger.warning("user_stream_socket_error symbol=%s error=%s", self._symbol, error)

    def _handle_user_socket_close(self, ws_app: Any, status_code: Any, message: Any) -> None:  # noqa: ARG002
        if self._user_stop_event.is_set():
            return
        technical_logger.warning(
            "user_stream_disconnected symbol=%s status=%s message=%s",
            self._symbol,
            status_code,
            message,
        )

    def _bootstrap_candle_caches(self) -> None:
        technical_logger.info(
            "market_stream_bootstrap_started symbol=%s small=%s medium=%s big=%s",
            self._symbol,
            self._intervals["small"],
            self._intervals["medium"],
            self._intervals["big"],
        )
        df_small = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["small"], self._limits["small"]),
            self._limits["small"],
        )
        df_medium = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["medium"], self._limits["medium"]),
            self._limits["medium"],
        )
        df_big = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["big"], self._limits["big"]),
            self._limits["big"],
        )

        with self._cache_lock:
            self._small_df = df_small
            self._medium_df = df_medium
            self._big_df = df_big
            self._needs_resync = False

        technical_logger.info(
            "market_stream_bootstrap_completed symbol=%s rows_small=%s rows_medium=%s rows_big=%s",
            self._symbol,
            len(df_small),
            len(df_medium),
            len(df_big),
        )

    def _handle_user_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return

        if message.get("e") == "error":
            technical_logger.warning("user_stream_error symbol=%s payload=%s", self._symbol, message)
            return

        event_type = str(message.get("e", "")).strip()
        event_ts = message.get("E") or message.get("T") or int(time.time() * 1000)

        if event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(message.get("a"), event_ts)
            return

        if event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_trade_update(message.get("o"), event_ts)
            return

        if event_type == "listenKeyExpired":
            technical_logger.warning("user_stream_listen_key_expired symbol=%s", self._symbol)
            expired_key = self._user_listen_key
            self._user_listen_key = None
            if expired_key:
                self._close_listen_key(expired_key)
            if self._user_socket_app is not None:
                try:
                    self._user_socket_app.close()
                except Exception:  # noqa: BLE001
                    pass

    def _handle_account_update(self, account_data: Any, event_ts: Any) -> None:
        if not isinstance(account_data, dict):
            return

        balance_entry = None
        for candidate in account_data.get("B", []) or []:
            if str(candidate.get("a", "")).upper() == "USDT":
                balance_entry = candidate
                break

        balance_value = None
        if isinstance(balance_entry, dict):
            balance_value = _to_float(balance_entry.get("wb"))
            if balance_value is None:
                balance_value = _to_float(balance_entry.get("cw"))
        if balance_value is not None:
            set_cached_balance(balance_value)

        position_update = None
        for candidate in account_data.get("P", []) or []:
            if str(candidate.get("s", "")).upper() == self._symbol:
                position_update = candidate
                break

        if position_update is not None:
            self._apply_account_position_update(position_update, event_ts)

        if balance_value is not None:
            with self._state_lock:
                state = self._state_getter()
                if isinstance(state, dict):
                    previous_balance = _to_float(state.get("balance"))
                    if previous_balance is None or abs(previous_balance - balance_value) > 1e-9:
                        update_balance(state, balance_value)

    def _handle_order_trade_update(self, order_data: Any, event_ts: Any) -> None:
        if not isinstance(order_data, dict):
            return

        symbol = str(order_data.get("s", "")).upper()
        if symbol and symbol != self._symbol:
            return

        execution_type = str(order_data.get("x", "")).strip().upper()
        order_status = str(order_data.get("X", "")).strip().upper()
        order_type = str(order_data.get("o", "")).strip().upper()
        technical_logger.debug(
            "user_stream_order_update symbol=%s execution_type=%s status=%s order_type=%s event_time=%s",
            self._symbol,
            execution_type,
            order_status,
            order_type,
            _format_event_timestamp_ms(event_ts),
        )

    def _apply_account_position_update(self, position_update: dict[str, Any], event_ts: Any) -> None:
        position_amt = _to_float(position_update.get("pa"))
        entry_price = _to_float(position_update.get("ep"))
        if position_amt is None:
            return

        if abs(position_amt) <= 1e-12:
            set_cached_position(self._symbol, None)
        else:
            cached_position = {
                "symbol": self._symbol,
                "positionAmt": position_amt,
                "entryPrice": entry_price,
                "unRealizedProfit": _to_float(position_update.get("up")),
                "positionSide": position_update.get("ps"),
            }
            set_cached_position(self._symbol, cached_position)

        ts_label = _format_event_timestamp_ms(event_ts)

        with self._state_lock:
            state = self._state_getter()
            if not isinstance(state, dict):
                return

            current_position = state.get("position")
            if not isinstance(current_position, dict):
                if abs(position_amt) > 1e-12:
                    technical_logger.warning(
                        "user_stream_unmanaged_position_detected symbol=%s position_amt=%s entry=%s",
                        self._symbol,
                        position_amt,
                        entry_price,
                    )
                return

            if abs(position_amt) <= 1e-12:
                update_position(state, None)
                return

            side = "short" if position_amt < 0 else "long"
            next_position = dict(current_position)
            next_position["side"] = side
            next_position["size"] = abs(position_amt)
            if entry_price is not None and entry_price > 0:
                next_position["entry"] = entry_price
            next_position.setdefault("time", next_position.get("entry_time") or ts_label)
            next_position.setdefault("entry_time", next_position.get("time") or ts_label)

            snapshot_price = (
                self._last_mark_price
                if self._last_mark_price is not None
                else self._last_ticker_price
            )
            if snapshot_price is None and entry_price is not None:
                snapshot_price = entry_price

            if snapshot_price is not None:
                state["position"] = next_position
                refresh_runtime_state_from_price_tick(
                    state,
                    last_price=self._last_ticker_price if self._last_ticker_price is not None else snapshot_price,
                    position_price=snapshot_price,
                    leverage=self._leverage,
                    ts_label=ts_label,
                )
                next_position = state.get("position")

            update_position(state, next_position if isinstance(next_position, dict) else current_position)

    def _handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return

        payload = message.get("data") if isinstance(message.get("data"), dict) else message
        if not isinstance(payload, dict):
            return

        if payload.get("e") == "error":
            technical_logger.warning("market_stream_error symbol=%s payload=%s", self._symbol, payload)
            return

        payload_symbol = str(payload.get("s", "")).upper().strip()
        if payload_symbol and payload_symbol != self._symbol:
            return

        event_type = str(payload.get("e", "")).strip()
        event_ts = payload.get("E") or payload.get("T") or int(time.time() * 1000)

        updated = False
        if event_type == "24hrTicker":
            ticker_price = _to_float(payload.get("c"))
            if ticker_price is None:
                ticker_price = _to_float(payload.get("p"))
            if ticker_price is not None:
                self._last_ticker_price = ticker_price
                updated = True
        elif event_type == "markPriceUpdate":
            mark_price = _to_float(payload.get("p"))
            if mark_price is not None:
                self._last_mark_price = mark_price
                updated = True
        elif event_type == "kline":
            self._handle_kline_event(payload.get("k"))

        if updated:
            self._apply_runtime_update(event_ts)

    def _handle_kline_event(self, kline: Any) -> None:
        if not isinstance(kline, dict) or not bool(kline.get("x")):
            return

        interval = str(kline.get("i", "")).strip()
        candle = _candle_row_from_kline(kline)
        if candle is None:
            return

        if interval == self._intervals["small"]:
            self._upsert_closed_candle("small", candle)
            open_time_ms = int(kline["t"])
            with self._cache_lock:
                if self._last_emitted_small_open_time_ms == open_time_ms:
                    return
                self._pending_small_open_time_ms = open_time_ms
                self._pending_emit_after_monotonic = time.monotonic() + MARKET_STREAM_SETTLE_SECONDS
            return

        if interval == self._intervals["medium"]:
            self._upsert_closed_candle("medium", candle)
            return

        if interval == self._intervals["big"]:
            self._upsert_closed_candle("big", candle)

    def _upsert_closed_candle(self, timeframe: str, candle: dict[str, Any]) -> None:
        limit = self._limits[timeframe]
        expected_delta = self._interval_deltas[timeframe]
        row_df = pd.DataFrame([candle])

        with self._cache_lock:
            current_df = self._get_frame(timeframe)
            previous_open_time = None
            if not current_df.empty:
                previous_open_time = current_df["open_time"].iloc[-1]

            merged_df = _normalize_candle_frame(pd.concat([current_df, row_df], ignore_index=True), limit)
            self._set_frame(timeframe, merged_df)

            latest_open_time = merged_df["open_time"].iloc[-1] if not merged_df.empty else None
            if (
                previous_open_time is not None
                and latest_open_time is not None
                and latest_open_time > previous_open_time + expected_delta
            ):
                self._needs_resync = True
                technical_logger.warning(
                    "market_stream_gap_detected symbol=%s timeframe=%s previous=%s latest=%s",
                    self._symbol,
                    timeframe,
                    previous_open_time,
                    latest_open_time,
                )

    def _get_frame(self, timeframe: str) -> pd.DataFrame:
        if timeframe == "small":
            return self._small_df
        if timeframe == "medium":
            return self._medium_df
        return self._big_df

    def _set_frame(self, timeframe: str, df: pd.DataFrame) -> None:
        if timeframe == "small":
            self._small_df = df
        elif timeframe == "medium":
            self._medium_df = df
        else:
            self._big_df = df

    def _apply_runtime_update(self, event_ts: Any) -> None:
        display_price = self._last_ticker_price if self._last_ticker_price is not None else self._last_mark_price
        position_price = self._last_mark_price if self._last_mark_price is not None else display_price
        if display_price is None or position_price is None:
            return

        ts_label = _format_event_timestamp_ms(event_ts)

        with self._state_lock:
            state = self._state_getter()
            if not isinstance(state, dict):
                return

            refresh_runtime_state_from_price_tick(
                state,
                last_price=display_price,
                position_price=position_price,
                leverage=self._leverage,
                ts_label=ts_label,
            )

            now_monotonic = time.monotonic()
            should_persist = (
                self._last_persist_monotonic is None
                or (now_monotonic - self._last_persist_monotonic) >= MARKET_STREAM_PERSIST_INTERVAL_SECONDS
            )
            if not should_persist:
                return

            try:
                self._save_state_fn(state)
            except Exception as exc:  # noqa: BLE001
                technical_logger.warning("market_stream_state_persist_failed symbol=%s error=%s", self._symbol, exc)
                return

            self._last_persist_monotonic = now_monotonic
