from __future__ import annotations

from collections import deque
import json
import os
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from datetime import UTC, datetime
from typing import Any, Callable

import pandas as pd

from bot.event_log import emit_event, get_technical_logger
from bot.state import add_closed_trade, update_balance, update_position
from bot.trade import (
    clear_runtime_account_cache,
    set_cached_balance,
    set_cached_position,
)
from core.feature_engine import FeatureEngine
from core.engine import refresh_runtime_state_from_price_tick
from core.indicator_inputs import indicator_selection_plan, merge_indicator_decision_values
from core.market_events import (
    AccountBalanceEvent,
    CandleClosedEvent,
    IndicatorSnapshotEvent,
    JsonlMarketEventStore,
    MarkPriceEvent,
    MarketEvent,
    OrderTradeUpdateEvent,
    PriceTickEvent,
    PositionSnapshotEvent,
)
from shared.time_utils import utc_now_text, utc_text_from_timestamp

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
MARKET_EVENT_STREAM_ENABLED = os.getenv("SCROOGE_MARKET_EVENT_STREAM_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
MARKET_EVENT_STREAM_FILE = os.getenv("SCROOGE_MARKET_EVENT_STREAM_FILE", "runtime/market_events.jsonl").strip()
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
MARKET_STREAM_RECONNECT_SECONDS = max(
    1.0,
    float(os.getenv("SCROOGE_MARKET_STREAM_RECONNECT_SECONDS", "5")),
)
FUTURES_REST_BASE_URL = os.getenv("SCROOGE_FUTURES_REST_BASE_URL", "https://fapi.binance.com").rstrip("/")
FUTURES_PRIVATE_WS_BASE_URL = os.getenv(
    "SCROOGE_FUTURES_PRIVATE_WS_BASE_URL",
    "wss://fstream.binance.com/private",
).rstrip("/")
FUTURES_MARKET_WS_STREAM_URL = os.getenv(
    "SCROOGE_FUTURES_MARKET_WS_STREAM_URL",
    "wss://fstream.binance.com/market/stream",
).rstrip("/")
USER_STREAM_EVENTS = tuple(
    event.strip()
    for event in os.getenv(
        "SCROOGE_USER_STREAM_EVENTS",
        "ACCOUNT_UPDATE,ORDER_TRADE_UPDATE",
    ).split(",")
    if event.strip()
)
BINANCE_MAX_KLINE_LIMIT = 1500
INDICATOR_WARMUP_ROWS = {
    "small": 2,
    "medium": 24,
    "big": 54,
}


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:  # NaN check without math import
        return None
    return numeric


def _to_optional_int(value: Any) -> int | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _normalize_rest_exchange_position_snapshot(symbol: str, snapshot: dict[str, Any], *, source: str, updated_at: str) -> dict[str, Any]:
    position_amt = _to_float(snapshot.get("positionAmt"))
    entry_price = _to_float(snapshot.get("entryPrice"))
    unrealized_pnl = _to_float(snapshot.get("unRealizedProfit"))
    isolated_margin = _to_float(snapshot.get("isolatedWallet")) or _to_float(snapshot.get("isolatedMargin"))
    return {
        "symbol": str(symbol).upper(),
        "position_amt": position_amt,
        "entry_price": entry_price,
        "unrealized_pnl": unrealized_pnl,
        "position_side": snapshot.get("positionSide"),
        "isolated_margin": isolated_margin,
        "mark_price": _to_float(snapshot.get("markPrice")),
        "liq_price": _to_float(snapshot.get("liquidationPrice")),
        "break_even_price": _to_float(snapshot.get("breakEvenPrice")),
        "updated_at": updated_at,
        "source": source,
    }


def _normalize_exchange_position_side(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _select_symbol_position_update(
    candidates: list[dict[str, Any]],
    *,
    symbol: str,
    local_position: dict[str, Any] | None,
) -> dict[str, Any] | None:
    matching = [candidate for candidate in candidates if str(candidate.get("s", "")).upper() == symbol]
    if not matching:
        return None

    local_side = str(local_position.get("side")).strip().lower() if isinstance(local_position, dict) else ""
    preferred_position_side = "LONG" if local_side == "long" else "SHORT" if local_side == "short" else None

    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for candidate in matching:
        position_amt = abs(_to_float(candidate.get("pa")) or 0.0)
        position_side = _normalize_exchange_position_side(candidate.get("ps"))
        score = 0
        if preferred_position_side and position_side == preferred_position_side:
            score += 4
        if position_amt > 1e-12:
            score += 2
        if position_side == "BOTH":
            score += 1
        ranked.append((score, position_amt, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _format_event_timestamp_ms(value: Any) -> str:
    try:
        ts_ms = int(value)
    except (TypeError, ValueError):
        return utc_now_text(TIMESTAMP_FORMAT)
    return utc_text_from_timestamp(ts_ms, TIMESTAMP_FORMAT)


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


def _derive_candle_limits(intervals: dict[str, str]) -> dict[str, int]:
    small_interval = str(intervals["small"])
    medium_interval = str(intervals["medium"])
    big_interval = str(intervals["big"])

    small_delta = _interval_to_timedelta(small_interval)
    coverage_span = small_delta * BINANCE_MAX_KLINE_LIMIT

    def required_rows(interval: str, *, timeframe: str) -> int:
        interval_delta = _interval_to_timedelta(interval)
        coverage_rows = int((coverage_span / interval_delta)) + 2
        derived = max(coverage_rows, INDICATOR_WARMUP_ROWS[timeframe])
        return max(1, min(BINANCE_MAX_KLINE_LIMIT, derived))

    return {
        "small": BINANCE_MAX_KLINE_LIMIT,
        "medium": required_rows(medium_interval, timeframe="medium"),
        "big": required_rows(big_interval, timeframe="big"),
    }


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
        state_getter: Callable[[], dict[str, Any]],
        state_lock: threading.RLock,
        save_state_fn: Callable[[dict[str, Any]], None],
        fetch_historical_fn: Callable[[str, str, int], pd.DataFrame],
        indicator_inputs: dict[str, str] | None,
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
        self._indicator_inputs = dict(indicator_inputs or {})
        self._indicator_selection_plan = indicator_selection_plan(self._indicator_inputs)
        self._get_balance_fn = get_balance_fn
        self._get_open_position_fn = get_open_position_fn

        self._intervals = {
            "small": str(intervals["small"]),
            "medium": str(intervals["medium"]),
            "big": str(intervals["big"]),
        }
        self._candle_limits = _derive_candle_limits(self._intervals)
        self._interval_deltas = {
            key: _interval_to_timedelta(value)
            for key, value in self._intervals.items()
        }
        self._market_event_lock = threading.RLock()
        self._market_event_store = (
            JsonlMarketEventStore(MARKET_EVENT_STREAM_FILE)
            if MARKET_EVENT_STREAM_ENABLED and MARKET_EVENT_STREAM_FILE
            else None
        )
        self._runtime_event_lock = threading.RLock()
        self._runtime_event_queue: deque[MarketEvent] = deque()

        self._cache_lock = threading.RLock()
        self._small_df = _empty_candle_frame()
        self._medium_df = _empty_candle_frame()
        self._big_df = _empty_candle_frame()
        self._needs_resync = False
        self._pending_small_open_time_ms: int | None = None
        self._pending_emit_after_monotonic: float | None = None
        self._last_emitted_small_open_time_ms: int | None = None

        self._market_socket_app: Any = None
        self._market_socket_thread: threading.Thread | None = None
        self._market_stop_event = threading.Event()
        self._market_stream_connected = False
        self._last_market_stream_message_monotonic: float | None = None
        self._last_market_stream_connected_monotonic: float | None = None
        self._last_market_stream_disconnected_monotonic: float | None = None
        self._user_listen_key: str | None = None
        self._user_socket_app: Any = None
        self._user_socket_thread: threading.Thread | None = None
        self._user_keepalive_thread: threading.Thread | None = None
        self._user_stop_event = threading.Event()
        self._user_stream_connected = False
        self._last_user_stream_message_monotonic: float | None = None
        self._last_user_stream_connected_monotonic: float | None = None
        self._last_user_stream_disconnected_monotonic: float | None = None
        self._last_ticker_price: float | None = None
        self._last_mark_price: float | None = None
        self._last_persist_monotonic: float | None = None
        self._running = False

    def _merge_exchange_position_snapshot_into_state(self, state: dict[str, Any], snapshot: dict[str, Any]) -> None:
        state["exchange_position"] = dict(snapshot)

        current_position = state.get("position")
        if not isinstance(current_position, dict):
            return

        current_position["exchange_position_amt"] = snapshot.get("position_amt")
        current_position["exchange_entry_price"] = snapshot.get("entry_price")
        current_position["exchange_unrealized_pnl"] = snapshot.get("unrealized_pnl")
        current_position["exchange_position_side"] = snapshot.get("position_side")
        current_position["exchange_isolated_margin"] = snapshot.get("isolated_margin")
        current_position["exchange_mark_price"] = snapshot.get("mark_price")
        current_position["exchange_liq_price"] = snapshot.get("liq_price")
        current_position["exchange_break_even_price"] = snapshot.get("break_even_price")
        current_position["exchange_position_updated_at"] = snapshot.get("updated_at")

    def _state_exchange_position_needs_enrichment(self, state: dict[str, Any]) -> bool:
        exchange_position = state.get("exchange_position")
        if not isinstance(exchange_position, dict):
            return True
        return exchange_position.get("liq_price") is None or exchange_position.get("break_even_price") is None

    def _fetch_rest_exchange_position_snapshot(self, *, source: str, updated_at: str) -> dict[str, Any] | None:
        try:
            position = self._get_open_position_fn(self._symbol)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("exchange_position_enrichment_failed symbol=%s source=%s error=%s", self._symbol, source, exc)
            return None
        if not isinstance(position, dict):
            return None
        set_cached_position(self._symbol, position)
        return _normalize_rest_exchange_position_snapshot(
            self._symbol,
            position,
            source=source,
            updated_at=updated_at,
        )

    def _reconcile_position_from_exchange_snapshot(self, *, source: str, event_ts: Any) -> None:
        try:
            position = self._get_open_position_fn(self._symbol)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("exchange_position_reconcile_failed symbol=%s source=%s error=%s", self._symbol, source, exc)
            return

        set_cached_position(self._symbol, position)
        if isinstance(position, dict):
            self._apply_account_position_update(
                {
                    "pa": position.get("positionAmt"),
                    "ep": position.get("entryPrice"),
                    "up": position.get("unRealizedProfit"),
                    "ps": position.get("positionSide"),
                    "iw": position.get("isolatedWallet") or position.get("isolatedMargin"),
                },
                event_ts,
            )
            return

        self._apply_account_position_update(
            {
                "pa": 0.0,
                "ep": 0.0,
                "up": 0.0,
                "ps": None,
                "iw": 0.0,
            },
            event_ts,
        )

    def start(self) -> bool:
        if self._running:
            return True

        if not MARKET_STREAM_ENABLED:
            technical_logger.info("market_stream_disabled_by_env symbol=%s", self._symbol)
            return False

        if websocket is None:
            technical_logger.warning("market_stream_unavailable reason=websocket_client_missing")
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

        try:
            market_stream_started = self._start_market_stream(streams)
            user_stream_started = self._start_user_stream()
        except Exception as exc:  # noqa: BLE001
            technical_logger.exception("market_stream_start_failed symbol=%s error=%s", self._symbol, exc)
            self._stop_market_stream()
            self._stop_user_stream()
            return False

        if not market_stream_started:
            self._stop_market_stream()
            self._stop_user_stream()
            return False

        self._last_persist_monotonic = None
        self._running = True
        technical_logger.info(
            "market_stream_started symbol=%s streams=%s user_stream=%s persist_interval=%s settle=%s",
            self._symbol,
            ",".join(streams),
            bool(user_stream_started),
            MARKET_STREAM_PERSIST_INTERVAL_SECONDS,
            MARKET_STREAM_SETTLE_SECONDS,
        )
        return True

    def is_running(self) -> bool:
        return self._running

    def is_user_stream_connected(self) -> bool:
        return self._user_stream_connected

    def get_user_stream_message_age_seconds(self) -> float | None:
        last_message = self._last_user_stream_message_monotonic
        if last_message is None:
            return None
        return max(0.0, time.monotonic() - last_message)

    def stop(self) -> None:
        self._running = False
        self._last_persist_monotonic = None
        self._stop_market_stream()
        self._stop_user_stream()
        technical_logger.info("market_stream_stopped symbol=%s", self._symbol)

    def update_config(
        self,
        *,
        symbol: str,
        leverage: float | int,
        intervals: dict[str, Any],
        indicator_inputs: dict[str, str] | None = None,
    ) -> None:
        next_symbol = str(symbol).upper()
        next_leverage = leverage
        next_intervals = {
            "small": str(intervals["small"]),
            "medium": str(intervals["medium"]),
            "big": str(intervals["big"]),
        }
        next_candle_limits = _derive_candle_limits(next_intervals)

        symbol_changed = next_symbol != self._symbol
        intervals_changed = next_intervals != self._intervals
        limits_changed = next_candle_limits != self._candle_limits

        self._leverage = next_leverage
        if indicator_inputs is not None:
            self._indicator_inputs = dict(indicator_inputs)
            self._indicator_selection_plan = indicator_selection_plan(self._indicator_inputs)
        if not symbol_changed and not intervals_changed and not limits_changed:
            return

        was_running = self._running
        self.stop()
        self._symbol = next_symbol
        self._intervals = next_intervals
        self._candle_limits = next_candle_limits
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

    def take_ready_strategy_row(self) -> dict[str, Any] | None:
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

        feature_engine = FeatureEngine(intervals=self._intervals)
        feature_engine.bootstrap_from_frames(
            df_small=df_small,
            df_medium=df_medium,
            df_big=df_big,
        )
        latest = df_small.iloc[-1]
        discrete_indicator_values = feature_engine.closed_values()
        realtime_indicator_values = None
        if self._indicator_selection_plan.requires_realtime:
            realtime_indicator_values = feature_engine.realtime_values(
                selected_keys=self._indicator_selection_plan.realtime_keys,
            )
        indicator_values = merge_indicator_decision_values(
            indicator_inputs=self._indicator_inputs,
            discrete_values=discrete_indicator_values,
            realtime_values=realtime_indicator_values,
            selection_plan=self._indicator_selection_plan,
        )
        if indicator_values is None:
            return None
        self._emit_indicator_snapshot(
            open_time=latest.get("open_time"),
            values=discrete_indicator_values,
        )
        return {
            "open_time": latest.get("open_time"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "volume": latest.get("volume"),
            "EMA": indicator_values["EMA"],
            "RSI": indicator_values["RSI"],
            "BBL": indicator_values["BBL"],
            "BBM": indicator_values["BBM"],
            "BBU": indicator_values["BBU"],
            "ATR": indicator_values["ATR"],
        }

    def take_pending_market_events(self, *, limit: int | None = None) -> list[MarketEvent]:
        with self._runtime_event_lock:
            if not self._runtime_event_queue:
                return []
            if limit is None or limit <= 0 or limit >= len(self._runtime_event_queue):
                events = list(self._runtime_event_queue)
                self._runtime_event_queue.clear()
                return events
            events = [self._runtime_event_queue.popleft() for _ in range(limit)]
            return events

    def snapshot_candle_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with self._cache_lock:
            return (
                self._small_df.copy(),
                self._medium_df.copy(),
                self._big_df.copy(),
            )

    def _append_market_event(
        self,
        event: (
            PriceTickEvent
            | MarkPriceEvent
            | CandleClosedEvent
            | IndicatorSnapshotEvent
            | AccountBalanceEvent
            | PositionSnapshotEvent
            | OrderTradeUpdateEvent
        ),
    ) -> None:
        with self._runtime_event_lock:
            self._runtime_event_queue.append(event)
        if self._market_event_store is None:
            return
        try:
            with self._market_event_lock:
                self._market_event_store.append(event)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning("market_event_append_failed symbol=%s type=%s error=%s", self._symbol, event.event_type, exc)

    def _emit_indicator_snapshot(
        self,
        *,
        open_time: Any,
        values: dict[str, float | None] | None,
    ) -> None:
        if open_time is None:
            return

        if pd.isna(open_time):
            return

        open_dt = pd.Timestamp(open_time)
        if open_dt.tzinfo is not None:
            open_dt = open_dt.tz_convert(None)

        close_dt = open_dt + self._interval_deltas["small"] - pd.Timedelta(seconds=1)
        payload_values = {
            "EMA": _to_float(None if values is None else values.get("EMA")),
            "RSI": _to_float(None if values is None else values.get("RSI")),
            "BBL": _to_float(None if values is None else values.get("BBL")),
            "BBM": _to_float(None if values is None else values.get("BBM")),
            "BBU": _to_float(None if values is None else values.get("BBU")),
            "ATR": _to_float(None if values is None else values.get("ATR")),
        }
        self._append_market_event(
            IndicatorSnapshotEvent(
                symbol=self._symbol,
                ts=(
                    close_dt.tz_localize(UTC).strftime(TIMESTAMP_FORMAT)
                    if close_dt.tzinfo is None
                    else close_dt.tz_convert(UTC).strftime(TIMESTAMP_FORMAT)
                ),
                interval="discrete_snapshot",
                values=payload_values,
            )
        )

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
            bootstrap_event_ts = int(time.time() * 1000)
            if isinstance(position, dict):
                ts_label = utc_now_text(TIMESTAMP_FORMAT)
                snapshot = _normalize_rest_exchange_position_snapshot(
                    self._symbol,
                    position,
                    source="rest_bootstrap",
                    updated_at=ts_label,
                )
                with self._state_lock:
                    state = self._state_getter()
                    if isinstance(state, dict):
                        self._merge_exchange_position_snapshot_into_state(state, snapshot)
                        self._save_state_fn(state)
            else:
                self._apply_account_position_update(
                    {
                        "pa": 0.0,
                        "ep": 0.0,
                        "up": 0.0,
                        "ps": None,
                        "iw": 0.0,
                    },
                    bootstrap_event_ts,
                )

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

    def _start_market_stream(self, streams: list[str]) -> bool:
        self._market_stop_event.clear()
        self._market_socket_thread = threading.Thread(
            target=self._run_market_socket_loop,
            args=(tuple(streams),),
            name=f"scrooge-market-stream-{self._symbol.lower()}",
            daemon=True,
        )
        self._market_socket_thread.start()
        return True

    def _stop_market_stream(self) -> None:
        self._market_stop_event.set()

        socket_app = self._market_socket_app
        self._market_socket_app = None
        if socket_app is not None:
            try:
                socket_app.close()
            except Exception as exc:  # noqa: BLE001
                technical_logger.warning("market_stream_socket_close_failed symbol=%s error=%s", self._symbol, exc)

        thread_obj = self._market_socket_thread
        if thread_obj is not None and thread_obj.is_alive():
            thread_obj.join(timeout=2.0)
        self._market_socket_thread = None
        self._market_stream_connected = False
        self._last_market_stream_disconnected_monotonic = time.monotonic()

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
        self._user_stream_connected = False
        self._last_user_stream_disconnected_monotonic = time.monotonic()

        listen_key = self._user_listen_key
        self._user_listen_key = None
        if listen_key:
            self._close_listen_key(listen_key)

    def _run_market_socket_loop(self, streams: tuple[str, ...]) -> None:
        stream_path = "/".join(streams)
        ws_url = f"{FUTURES_MARKET_WS_STREAM_URL}?streams={stream_path}"

        while not self._market_stop_event.is_set():
            socket_app = websocket.WebSocketApp(
                ws_url,
                on_open=self._handle_market_socket_open,
                on_message=self._handle_market_socket_message,
                on_error=self._handle_market_socket_error,
                on_close=self._handle_market_socket_close,
            )
            self._market_socket_app = socket_app

            try:
                socket_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:  # noqa: BLE001
                technical_logger.warning("market_stream_run_failed symbol=%s error=%s", self._symbol, exc)
            finally:
                self._market_socket_app = None

            if self._market_stop_event.is_set():
                break

            technical_logger.warning(
                "market_stream_reconnecting symbol=%s delay=%s",
                self._symbol,
                MARKET_STREAM_RECONNECT_SECONDS,
            )
            if self._market_stop_event.wait(MARKET_STREAM_RECONNECT_SECONDS):
                break

    def _handle_market_socket_open(self, ws_app: Any) -> None:  # noqa: ARG002
        self._market_stream_connected = True
        self._last_market_stream_connected_monotonic = time.monotonic()
        technical_logger.info("market_stream_connected symbol=%s", self._symbol)

    def _handle_market_socket_message(self, ws_app: Any, raw_message: Any) -> None:  # noqa: ARG002
        self._market_stream_connected = True
        self._last_market_stream_message_monotonic = time.monotonic()
        try:
            message = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
        except json.JSONDecodeError:
            technical_logger.warning("market_stream_message_invalid_json symbol=%s raw=%s", self._symbol, raw_message)
            return
        self._handle_message(message)

    def _handle_market_socket_error(self, ws_app: Any, error: Any) -> None:  # noqa: ARG002
        if self._market_stop_event.is_set():
            return
        self._market_stream_connected = False
        self._last_market_stream_disconnected_monotonic = time.monotonic()
        technical_logger.warning("market_stream_socket_error symbol=%s error=%s", self._symbol, error)

    def _handle_market_socket_close(self, ws_app: Any, status_code: Any, message: Any) -> None:  # noqa: ARG002
        if self._market_stop_event.is_set():
            return
        self._market_stream_connected = False
        self._last_market_stream_disconnected_monotonic = time.monotonic()
        technical_logger.warning(
            "market_stream_disconnected symbol=%s status=%s message=%s",
            self._symbol,
            status_code,
            message,
        )

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

            query_params: dict[str, str] = {"listenKey": listen_key}
            if USER_STREAM_EVENTS:
                query_params["events"] = "/".join(USER_STREAM_EVENTS)
            ws_url = f"{FUTURES_PRIVATE_WS_BASE_URL}/ws?{urllib.parse.urlencode(query_params)}"
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
        self._user_stream_connected = True
        self._last_user_stream_connected_monotonic = time.monotonic()
        technical_logger.info("user_stream_connected symbol=%s", self._symbol)

    def _handle_user_socket_message(self, ws_app: Any, raw_message: Any) -> None:  # noqa: ARG002
        self._user_stream_connected = True
        self._last_user_stream_message_monotonic = time.monotonic()
        try:
            message = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
        except json.JSONDecodeError:
            technical_logger.warning("user_stream_message_invalid_json symbol=%s raw=%s", self._symbol, raw_message)
            return
        self._handle_user_message(message)

    def _handle_user_socket_error(self, ws_app: Any, error: Any) -> None:  # noqa: ARG002
        if self._user_stop_event.is_set():
            return
        self._user_stream_connected = False
        self._last_user_stream_disconnected_monotonic = time.monotonic()
        technical_logger.warning("user_stream_socket_error symbol=%s error=%s", self._symbol, error)

    def _handle_user_socket_close(self, ws_app: Any, status_code: Any, message: Any) -> None:  # noqa: ARG002
        if self._user_stop_event.is_set():
            return
        self._user_stream_connected = False
        self._last_user_stream_disconnected_monotonic = time.monotonic()
        technical_logger.warning(
            "user_stream_disconnected symbol=%s status=%s message=%s",
            self._symbol,
            status_code,
            message,
        )

    def _bootstrap_candle_caches(self) -> None:
        technical_logger.info(
            "market_stream_bootstrap_started symbol=%s small=%s medium=%s big=%s lookback=%s/%s/%s",
            self._symbol,
            self._intervals["small"],
            self._intervals["medium"],
            self._intervals["big"],
            self._candle_limits["small"],
            self._candle_limits["medium"],
            self._candle_limits["big"],
        )
        df_small = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["small"], self._candle_limits["small"]),
            self._candle_limits["small"],
        )
        df_medium = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["medium"], self._candle_limits["medium"]),
            self._candle_limits["medium"],
        )
        df_big = _normalize_candle_frame(
            self._fetch_historical_fn(self._symbol, self._intervals["big"], self._candle_limits["big"]),
            self._candle_limits["big"],
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
        ts_label = _format_event_timestamp_ms(event_ts)

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
                self._append_market_event(
                    AccountBalanceEvent(
                        asset=str(balance_entry.get("a", "USDT")).strip().upper() or "USDT",
                        ts=ts_label,
                        wallet_balance=balance_value,
                        cross_wallet_balance=_to_float(balance_entry.get("cw")),
                        balance_delta=_to_float(balance_entry.get("bc")),
                    )
                )
        if balance_value is not None:
            set_cached_balance(balance_value)

        current_position = None
        with self._state_lock:
            state = self._state_getter()
            if isinstance(state, dict) and isinstance(state.get("position"), dict):
                current_position = state.get("position")

        raw_position_candidates = [
            candidate for candidate in (account_data.get("P", []) or []) if isinstance(candidate, dict)
        ]
        position_update = _select_symbol_position_update(
            raw_position_candidates,
            symbol=self._symbol,
            local_position=current_position,
        )

        if position_update is not None:
            position_amt = _to_float(position_update.get("pa"))
            if position_amt is not None:
                self._append_market_event(
                    PositionSnapshotEvent(
                        symbol=self._symbol,
                        ts=ts_label,
                        position_amt=position_amt,
                        entry_price=_to_float(position_update.get("ep")),
                        unrealized_pnl=_to_float(position_update.get("up")),
                        position_side=(str(position_update.get("ps")).strip() if position_update.get("ps") else None),
                    )
                )
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
        self._append_market_event(
            OrderTradeUpdateEvent(
                symbol=self._symbol,
                ts=_format_event_timestamp_ms(event_ts),
                order_side=(str(order_data.get("S")).strip().upper() if order_data.get("S") else None),
                order_type=(order_type or None),
                execution_type=(execution_type or None),
                order_status=(order_status or None),
                order_id=_to_optional_int(order_data.get("i")),
                trade_id=_to_optional_int(order_data.get("t")),
                last_filled_qty=_to_float(order_data.get("l")),
                accumulated_filled_qty=_to_float(order_data.get("z")),
                last_filled_price=_to_float(order_data.get("L")),
                average_price=_to_float(order_data.get("ap")),
                realized_pnl=_to_float(order_data.get("rp")),
                commission_asset=(str(order_data.get("N")).strip().upper() if order_data.get("N") else None),
                commission=_to_float(order_data.get("n")),
                reduce_only=bool(order_data.get("R", False)),
            )
        )
        technical_logger.debug(
            "user_stream_order_update symbol=%s execution_type=%s status=%s order_type=%s event_time=%s",
            self._symbol,
            execution_type,
            order_status,
            order_type,
            _format_event_timestamp_ms(event_ts),
        )

        if execution_type != "TRADE" or order_status not in {"FILLED", "PARTIALLY_FILLED"}:
            return

        should_reconcile = False
        order_id = _to_optional_int(order_data.get("i"))
        should_flag_unexpected_fill = False
        unexpected_fill_context: dict[str, Any] = {}
        order_side = str(order_data.get("S", "")).strip().upper()
        with self._state_lock:
            state = self._state_getter()
            if isinstance(state, dict):
                current_position = state.get("position")
                if isinstance(current_position, dict):
                    pending_close_order_id = _to_optional_int(current_position.get("pending_close_order_id"))
                    pending_open_order_id = _to_optional_int(current_position.get("pending_open_order_id"))
                    if bool(current_position.get("close_pending")):
                        should_reconcile = pending_close_order_id is None or pending_close_order_id == order_id
                    elif bool(current_position.get("open_pending")):
                        should_reconcile = pending_open_order_id is None or pending_open_order_id == order_id
                    else:
                        local_side = str(current_position.get("side") or "").strip().lower()
                        closes_local_position = (
                            (local_side == "long" and order_side == "SELL")
                            or (local_side == "short" and order_side == "BUY")
                        )
                        should_flag_unexpected_fill = closes_local_position
                        unexpected_fill_context = {
                            "side": current_position.get("side"),
                            "entry": current_position.get("entry"),
                        }

        if should_reconcile:
            self._reconcile_position_from_exchange_snapshot(
                source="rest_after_order_trade_update",
                event_ts=event_ts,
            )
            return

        if should_flag_unexpected_fill:
            technical_logger.warning(
                "user_stream_unexpected_fill symbol=%s order_id=%s order_side=%s order_type=%s status=%s avg_price=%s realized_pnl=%s reduce_only=%s",
                self._symbol,
                order_id,
                order_side,
                order_type,
                order_status,
                _to_float(order_data.get("ap")),
                _to_float(order_data.get("rp")),
                bool(order_data.get("R", False)),
            )
            emit_event(
                code="position_sync_unexpected_fill",
                category="error",
                ts=_format_event_timestamp_ms(event_ts),
                level="warning",
                notify=True,
                persist_ui=True,
                runtime_mode="live",
                ui_message=(
                    f"The exchange reported an unexpected fill on {self._symbol} while Scrooge was managing "
                    "a live position without a pending open or close request. Please inspect orders and the ledger."
                ),
                symbol=self._symbol,
                order_id=order_id,
                order_side=(order_side or None),
                order_type=(order_type or None),
                order_status=(order_status or None),
                average_price=_to_float(order_data.get("ap")),
                realized_pnl=_to_float(order_data.get("rp")),
                reduce_only=bool(order_data.get("R", False)),
                **unexpected_fill_context,
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
                "isolatedWallet": _to_float(position_update.get("iw")),
            }
            set_cached_position(self._symbol, cached_position)

        ts_label = _format_event_timestamp_ms(event_ts)
        enriched_snapshot = None

        with self._state_lock:
            state = self._state_getter()
            if not isinstance(state, dict):
                return
            current_position = state.get("position")
            should_enrich = abs(position_amt) > 1e-12 and (
                (isinstance(current_position, dict) and bool(current_position.get("open_pending")))
                or self._state_exchange_position_needs_enrichment(state)
            )

        if should_enrich:
            enriched_snapshot = self._fetch_rest_exchange_position_snapshot(
                source="rest_after_account_update",
                updated_at=ts_label,
            )

        with self._state_lock:
            state = self._state_getter()
            if not isinstance(state, dict):
                return

            previous_exchange_position = state.get("exchange_position") if isinstance(state.get("exchange_position"), dict) else {}
            state["exchange_position"] = {
                "symbol": self._symbol,
                "position_amt": position_amt,
                "entry_price": entry_price,
                "unrealized_pnl": _to_float(position_update.get("up")),
                "position_side": position_update.get("ps"),
                "isolated_margin": _to_float(position_update.get("iw")),
                "mark_price": self._last_mark_price,
                "liq_price": _to_float(previous_exchange_position.get("liq_price")),
                "break_even_price": _to_float(previous_exchange_position.get("break_even_price")),
                "updated_at": ts_label,
                "source": "user_stream_account_update",
            }
            if isinstance(enriched_snapshot, dict):
                self._merge_exchange_position_snapshot_into_state(state, enriched_snapshot)

            current_position = state.get("position")
            if not isinstance(current_position, dict):
                if abs(position_amt) > 1e-12:
                    technical_logger.warning(
                        "user_stream_unmanaged_position_detected symbol=%s position_amt=%s entry=%s",
                        self._symbol,
                        position_amt,
                        entry_price,
                    )
                    emit_event(
                        code="position_sync_unmanaged",
                        category="error",
                        ts=ts_label,
                        level="warning",
                        notify=True,
                        persist_ui=True,
                        runtime_mode="live",
                        ui_message=(
                            f"The exchange reported a live {self._symbol} position of {position_amt:.4f}, "
                            "but I had no local position in state. Please inspect the office ledger."
                        ),
                        symbol=self._symbol,
                        position_amt=position_amt,
                        entry=entry_price,
                    )
                    self._save_state_fn(state)
                return

            if abs(position_amt) <= 1e-12:
                if bool(current_position.get("open_pending")):
                    emit_event(
                        code="position_sync_cleared",
                        category="error",
                        ts=ts_label,
                        level="warning",
                        notify=True,
                        persist_ui=True,
                        runtime_mode="live",
                        ui_message=(
                            f"The exchange still reports {self._symbol} as flat after an open request, "
                            "so I cleared the pending local position from state. Please inspect fills and the ledger."
                        ),
                        symbol=self._symbol,
                        side=current_position.get("side"),
                        entry=current_position.get("entry"),
                    )
                    update_position(state, None)
                    return
                if bool(current_position.get("close_pending")):
                    pending_trade = current_position.get("pending_close_trade")
                    pending_event = current_position.get("pending_close_event")
                    pending_event_ts = str(
                        current_position.get("pending_close_event_ts")
                        or (pending_trade.get("exit_time") if isinstance(pending_trade, dict) else "")
                        or ts_label
                    )
                    state["position"] = None
                    if isinstance(pending_trade, dict):
                        add_closed_trade(state, pending_trade)
                    else:
                        update_position(state, None)
                    if isinstance(pending_event, dict):
                        emit_event(
                            ts=pending_event_ts,
                            runtime_mode=str(current_position.get("pending_close_runtime_mode") or "live"),
                            strategy_mode=(
                                str(current_position.get("pending_close_strategy_mode")).strip()
                                if current_position.get("pending_close_strategy_mode") is not None
                                else None
                            ),
                            **pending_event,
                        )
                    return
                emit_event(
                    code="position_sync_cleared",
                    category="error",
                    ts=ts_label,
                    level="warning",
                    notify=True,
                    persist_ui=True,
                    runtime_mode="live",
                    ui_message=(
                        f"The exchange reported {self._symbol} as flat, so I cleared the local "
                        f"{str(current_position.get('side') or 'position')} from state without a matched close event. "
                        "Please inspect fills and the ledger."
                    ),
                    symbol=self._symbol,
                    side=current_position.get("side"),
                    entry=current_position.get("entry"),
                )
                update_position(state, None)
                return

            side = "short" if position_amt < 0 else "long"
            next_position = dict(current_position)
            was_open_pending = bool(next_position.get("open_pending"))
            pending_open_event = next_position.get("pending_open_event") if was_open_pending else None
            pending_open_event_ts = str(next_position.get("pending_open_event_ts") or ts_label) if was_open_pending else ts_label
            next_position["side"] = side
            next_position["size"] = abs(position_amt)
            if entry_price is not None and entry_price > 0:
                next_position["entry"] = entry_price
            next_position["exchange_position_amt"] = position_amt
            next_position["exchange_entry_price"] = entry_price
            next_position["exchange_unrealized_pnl"] = _to_float(position_update.get("up"))
            next_position["exchange_position_side"] = position_update.get("ps")
            next_position["exchange_isolated_margin"] = _to_float(position_update.get("iw"))
            next_position["exchange_mark_price"] = self._last_mark_price
            next_position["exchange_liq_price"] = _to_float(next_position.get("exchange_liq_price"))
            next_position["exchange_break_even_price"] = _to_float(next_position.get("exchange_break_even_price"))
            next_position["exchange_position_updated_at"] = ts_label
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

            if isinstance(next_position, dict) and was_open_pending:
                next_position.pop("open_pending", None)
                next_position.pop("open_requested_at", None)
                next_position.pop("pending_open_runtime_mode", None)
                next_position.pop("pending_open_strategy_mode", None)
                next_position.pop("pending_open_order_id", None)
                next_position.pop("pending_open_event_ts", None)
                next_position.pop("pending_open_event", None)
            update_position(state, next_position if isinstance(next_position, dict) else current_position)
            if was_open_pending and isinstance(pending_open_event, dict):
                pending_open_event = dict(pending_open_event)
                pending_open_event["entry"] = (
                    float(entry_price)
                    if entry_price is not None and entry_price > 0
                    else pending_open_event.get("entry")
                )
                pending_open_event["size"] = abs(position_amt)
                emit_event(
                    ts=pending_open_event_ts,
                    runtime_mode=str(current_position.get("pending_open_runtime_mode") or "live"),
                    strategy_mode=(
                        str(current_position.get("pending_open_strategy_mode")).strip()
                        if current_position.get("pending_open_strategy_mode") is not None
                        else None
                    ),
                    **pending_open_event,
                )

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
                self._append_market_event(
                    PriceTickEvent(
                        symbol=self._symbol,
                        ts=_format_event_timestamp_ms(event_ts),
                        price=ticker_price,
                        source="ticker",
                    )
                )
                updated = True
        elif event_type == "markPriceUpdate":
            mark_price = _to_float(payload.get("p"))
            if mark_price is not None:
                self._last_mark_price = mark_price
                self._append_market_event(
                    MarkPriceEvent(
                        symbol=self._symbol,
                        ts=_format_event_timestamp_ms(event_ts),
                        mark_price=mark_price,
                        funding_rate=_to_float(payload.get("r")),
                        next_funding_time=_format_event_timestamp_ms(payload.get("T")) if payload.get("T") is not None else None,
                    )
                )
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
        close_time_ms = kline.get("T")
        close_time = _format_event_timestamp_ms(close_time_ms)
        self._append_market_event(
            CandleClosedEvent(
                symbol=self._symbol,
                ts=close_time,
                interval=interval,
                open_time=(
                    pd.Timestamp(candle["open_time"]).tz_localize(UTC).strftime(TIMESTAMP_FORMAT)
                    if pd.Timestamp(candle["open_time"]).tzinfo is None
                    else pd.Timestamp(candle["open_time"]).tz_convert(UTC).strftime(TIMESTAMP_FORMAT)
                ),
                close_time=close_time,
                open=float(candle["open"]),
                high=float(candle["high"]),
                low=float(candle["low"]),
                close=float(candle["close"]),
                volume=float(candle["volume"]),
            )
        )

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
        limit = self._candle_limits[timeframe]
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

            state["mark_price"] = position_price
            state["mark_price_updated_at"] = ts_label
            exchange_position = state.get("exchange_position")
            if isinstance(exchange_position, dict):
                exchange_position["mark_price"] = position_price
                exchange_position["updated_at"] = ts_label
            current_position = state.get("position")
            if isinstance(current_position, dict):
                current_position["exchange_mark_price"] = position_price
                current_position["exchange_position_updated_at"] = ts_label

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
