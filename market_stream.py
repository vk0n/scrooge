from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Callable

from event_log import get_technical_logger
from strategy import refresh_runtime_state_from_price_tick

try:
    from binance import ThreadedWebsocketManager
    from binance.enums import FuturesType
except ImportError:  # pragma: no cover - dependency installed in bot image
    ThreadedWebsocketManager = None
    FuturesType = None


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


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:  # NaN check without math import
        return None
    return numeric


def _format_event_timestamp_ms(value: Any) -> str:
    try:
        ts_ms = int(value)
    except (TypeError, ValueError):
        return datetime.now().strftime(TIMESTAMP_FORMAT)
    try:
        return datetime.fromtimestamp(ts_ms / 1000).strftime(TIMESTAMP_FORMAT)
    except (OSError, OverflowError, ValueError):
        return datetime.now().strftime(TIMESTAMP_FORMAT)


class LiveMarketStream:
    def __init__(
        self,
        *,
        api_key: str | None,
        api_secret: str | None,
        symbol: str,
        leverage: float | int,
        state_getter: Callable[[], dict[str, Any]],
        state_lock: threading.RLock,
        save_state_fn: Callable[[dict[str, Any]], None],
    ) -> None:
        self._api_key = api_key or None
        self._api_secret = api_secret or None
        self._symbol = str(symbol).upper()
        self._leverage = leverage
        self._state_getter = state_getter
        self._state_lock = state_lock
        self._save_state_fn = save_state_fn

        self._twm: ThreadedWebsocketManager | None = None
        self._stream_name: str | None = None
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

        streams = [
            f"{self._symbol.lower()}@ticker",
            f"{self._symbol.lower()}@markPrice@1s",
        ]

        twm = None
        try:
            twm = ThreadedWebsocketManager(api_key=self._api_key, api_secret=self._api_secret)
            twm.start()
            stream_name = twm.start_futures_multiplex_socket(
                callback=self._handle_message,
                streams=streams,
                futures_type=FuturesType.USD_M,
            )
        except Exception as exc:  # noqa: BLE001
            technical_logger.exception("market_stream_start_failed symbol=%s error=%s", self._symbol, exc)
            if twm is not None:
                try:
                    twm.stop()
                except Exception:  # noqa: BLE001
                    pass
            return False

        self._twm = twm
        self._stream_name = stream_name
        self._last_persist_monotonic = None
        self._running = True
        technical_logger.info(
            "market_stream_started symbol=%s streams=%s persist_interval=%s",
            self._symbol,
            ",".join(streams),
            MARKET_STREAM_PERSIST_INTERVAL_SECONDS,
        )
        return True

    def stop(self) -> None:
        twm = self._twm
        stream_name = self._stream_name
        self._running = False
        self._twm = None
        self._stream_name = None
        self._last_persist_monotonic = None

        if twm is None:
            return

        try:
            if stream_name:
                twm.stop_socket(stream_name)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("market_stream_socket_stop_failed symbol=%s error=%s", self._symbol, exc)

        try:
            twm.stop()
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("market_stream_stop_failed symbol=%s error=%s", self._symbol, exc)
        else:
            technical_logger.info("market_stream_stopped symbol=%s", self._symbol)

    def update_config(self, *, symbol: str, leverage: float | int) -> None:
        next_symbol = str(symbol).upper()
        next_leverage = leverage

        if next_symbol == self._symbol:
            self._leverage = next_leverage
            return

        was_running = self._running
        self.stop()
        self._symbol = next_symbol
        self._leverage = next_leverage
        self._last_ticker_price = None
        self._last_mark_price = None

        if was_running:
            self.start()

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

        if updated:
            self._apply_runtime_update(event_ts)

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
