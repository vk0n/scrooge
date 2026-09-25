from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from core.binance_retry import run_binance_with_retries
from core.feature_engine import (
    ATRFeature,
    BollingerBandsFeature,
    Candle,
    EMAFeature,
    RSIFeature,
)
from shared.runtime_db import (
    list_portfolio_asset_policies,
    mark_spot_signal_snapshot_error,
    save_spot_signal_snapshot,
)
from shared.spot_signal import (
    DEFAULT_BASE_TRANCHES_PCT,
    DEFAULT_SIGNAL_LEVELS_PCT,
    SpotSignalConfig,
    evaluate_rolling_24h_opportunity,
    parse_percentage_series,
)
from shared.spot_sizing import IndicatorSizingConfig
from shared.spot_strategy import finalize_spot_strategy_signal, spot_policy_eligibility

DEFAULT_ACCOUNT_KEY = "manual_spot"
SPOT_INDICATOR_RSI_PERIOD = 11
SPOT_INDICATOR_EMA_PERIOD = 50
SPOT_INDICATOR_BB_PERIOD = 20
SPOT_INDICATOR_BB_STD_MULT = 2.0
SPOT_INDICATOR_ATR_PERIOD = 14
SPOT_INDICATOR_KLINE_LIMIT = 60
BINANCE_KLINE_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w",
}


def spot_signal_config_from_env() -> SpotSignalConfig:
    levels_raw = os.getenv(
        "SCROOGE_SPOT_SIGNAL_LEVELS_PCT",
        ",".join(str(value) for value in DEFAULT_SIGNAL_LEVELS_PCT),
    )
    tranches_raw = os.getenv(
        "SCROOGE_SPOT_SIGNAL_BASE_TRANCHES_PCT",
        ",".join(str(value) for value in DEFAULT_BASE_TRANCHES_PCT),
    )
    return SpotSignalConfig(
        levels_pct=parse_percentage_series(levels_raw, field_name="SCROOGE_SPOT_SIGNAL_LEVELS_PCT"),
        base_tranches_pct=parse_percentage_series(
            tranches_raw,
            field_name="SCROOGE_SPOT_SIGNAL_BASE_TRANCHES_PCT",
        ),
    )


def spot_indicator_sizing_config_from_env() -> IndicatorSizingConfig:
    values = parse_percentage_series(
        os.getenv("SCROOGE_SPOT_INDICATOR_SIZING_MODIFIERS", "0.5,1,1.25,1.5"),
        field_name="SCROOGE_SPOT_INDICATOR_SIZING_MODIFIERS",
    )
    if len(values) != 4:
        raise ValueError("SCROOGE_SPOT_INDICATOR_SIZING_MODIFIERS must contain exactly four values.")
    return IndicatorSizingConfig(
        weak_modifier=values[0],
        neutral_modifier=values[1],
        strong_modifier=values[2],
        very_strong_modifier=values[3],
    )


def spot_indicator_interval_from_env() -> str:
    interval = str(os.getenv("SCROOGE_SPOT_INDICATOR_INTERVAL", "1h") or "1h").strip().lower()
    if interval not in BINANCE_KLINE_INTERVALS:
        raise ValueError(f"Unsupported SCROOGE_SPOT_INDICATOR_INTERVAL: {interval}")
    return interval


def normalize_rolling_ticker(payload: Any) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError("Binance rolling ticker response must be an object.")
    try:
        current_price = float(payload["lastPrice"])
        reference_price = float(payload["openPrice"])
        current_at_ms = int(payload["closeTime"])
        reference_at_ms = int(payload["openTime"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Binance rolling ticker response is missing 24-hour price data.") from exc
    return {
        "current_price": current_price,
        "reference_price": reference_price,
        "current_at_ms": current_at_ms,
        "reference_at_ms": reference_at_ms,
    }


def calculate_spot_indicator_context(
    payload: Any,
    *,
    current_price: float,
    evaluated_at_ms: int,
    interval: str,
) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise ValueError("Binance Spot klines response must be a list.")
    candles: list[tuple[int, int, Candle]] = []
    for raw in payload:
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            continue
        try:
            open_time_ms = int(raw[0])
            close_time_ms = int(raw[6])
            prices = tuple(float(raw[index]) for index in range(1, 5))
            volume = float(raw[5])
            if any(not math.isfinite(value) or value <= 0 for value in prices):
                continue
            if not math.isfinite(volume) or volume < 0:
                continue
            if prices[1] < max(prices[0], prices[3]) or prices[2] > min(prices[0], prices[3]):
                continue
            if prices[1] < prices[2]:
                continue
            candle = Candle(
                open_time=pd.Timestamp(open_time_ms, unit="ms"),
                open=prices[0],
                high=prices[1],
                low=prices[2],
                close=prices[3],
                volume=volume,
            )
        except (TypeError, ValueError):
            continue
        if close_time_ms <= int(evaluated_at_ms):
            candles.append((open_time_ms, close_time_ms, candle))
    candles.sort(key=lambda item: item[0])
    deduplicated = {open_time_ms: (close_time_ms, candle) for open_time_ms, close_time_ms, candle in candles}
    ordered = [(open_time_ms, *deduplicated[open_time_ms]) for open_time_ms in sorted(deduplicated)]
    if len(ordered) < SPOT_INDICATOR_EMA_PERIOD:
        raise ValueError(
            f"Spot indicator sizing requires at least {SPOT_INDICATOR_EMA_PERIOD} closed candles."
        )

    closed_candles = [item[2] for item in ordered]
    closes = [candle.close for candle in closed_candles]
    ema_feature = EMAFeature(SPOT_INDICATOR_EMA_PERIOD)
    rsi_feature = RSIFeature(SPOT_INDICATOR_RSI_PERIOD)
    bb_feature = BollingerBandsFeature(SPOT_INDICATOR_BB_PERIOD, SPOT_INDICATOR_BB_STD_MULT)
    atr_feature = ATRFeature(SPOT_INDICATOR_ATR_PERIOD)
    ema_feature.bootstrap(closes)
    rsi_feature.bootstrap(closes)
    bb_feature.bootstrap(closes)
    atr_feature.bootstrap(closed_candles)
    bollinger = bb_feature.closed_values()
    if bollinger is None:
        raise ValueError("Spot Bollinger context is not warmed up.")
    current = float(current_price)
    atr = atr_feature.closed_value()
    return {
        "interval": interval,
        "candle_count": len(closed_candles),
        "latest_closed_at_ms": ordered[-1][1],
        "latest_closed_price": closes[-1],
        "ema": ema_feature.closed_value(),
        "rsi": rsi_feature.closed_value(),
        "bb_lower": bollinger["BBL"],
        "bb_middle": bollinger["BBM"],
        "bb_upper": bollinger["BBU"],
        "atr": atr,
        "atr_pct": (atr / current) * 100.0 if atr is not None and current > 0 else None,
    }


def _policy_eligibility(policy: dict[str, Any], *, execution_enabled: bool) -> tuple[bool, str]:
    return spot_policy_eligibility(policy, execution_enabled=execution_enabled)


class RollingSpotSignalMonitor:
    def __init__(
        self,
        client: Any,
        *,
        interval_seconds: float,
        execution_enabled: bool,
        logger: Any,
        db_path: Path | None = None,
        config: SpotSignalConfig | None = None,
        sizing_config: IndicatorSizingConfig | None = None,
        indicator_interval: str | None = None,
        account_key: str = DEFAULT_ACCOUNT_KEY,
        snapshot_handler: Callable[[dict[str, Any]], Any] | None = None,
        snapshot_orderer: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.client = client
        self.interval_seconds = max(30.0, float(interval_seconds))
        self.execution_enabled = bool(execution_enabled)
        self.logger = logger
        self.db_path = db_path
        self.config = config or spot_signal_config_from_env()
        self.sizing_config = sizing_config or spot_indicator_sizing_config_from_env()
        self.indicator_interval = str(indicator_interval or spot_indicator_interval_from_env()).strip().lower()
        if self.indicator_interval not in BINANCE_KLINE_INTERVALS:
            raise ValueError(f"Unsupported Spot indicator interval: {self.indicator_interval}")
        self.account_key = str(account_key or DEFAULT_ACCOUNT_KEY).strip() or DEFAULT_ACCOUNT_KEY
        self.snapshot_handler = snapshot_handler
        self.snapshot_orderer = snapshot_orderer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_states: dict[tuple[str, str], tuple[object, ...]] = {}
        self._last_errors: dict[tuple[str, str], str] = {}

    def start(self) -> None:
        if not self.execution_enabled:
            self.logger.info("spot_signal_monitor_disabled reason=execution_disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="scrooge-spot-signals", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def refresh_once(self) -> list[dict[str, Any]]:
        if not self.execution_enabled:
            return []
        policies = list_portfolio_asset_policies(account_key=self.account_key, path=self.db_path)
        results: list[dict[str, Any]] = []
        for policy in policies:
            try:
                target_quantity = float(policy.get("target_quantity") or 0.0)
            except (TypeError, ValueError):
                target_quantity = 0.0
            if target_quantity <= 0:
                continue
            result = self._evaluate_policy(policy)
            if result is not None:
                results.append(result)
        if self.snapshot_handler is not None:
            ordered = results
            if self.snapshot_orderer is not None:
                try:
                    ordered = self.snapshot_orderer(results)
                except Exception as exc:  # noqa: BLE001
                    self.logger.exception("spot_strategy_snapshot_ordering_failed error=%s", exc)
            for saved in ordered:
                try:
                    self.snapshot_handler(saved)
                except Exception as exc:  # noqa: BLE001
                    self.logger.exception(
                        "spot_strategy_snapshot_handler_failed symbol=%s error=%s",
                        saved.get("market_symbol"),
                        exc,
                    )
        return results

    def _evaluate_policy(self, policy: dict[str, Any]) -> dict[str, Any] | None:
        asset_symbol = str(policy.get("asset_symbol") or "").strip().upper()
        quote_symbol = str(policy.get("quote_symbol") or "USDT").strip().upper() or "USDT"
        market_symbol = f"{asset_symbol}{quote_symbol}"
        key = (asset_symbol, quote_symbol)
        try:
            payload = run_binance_with_retries(
                lambda: self.client.get_ticker(symbol=market_symbol),
                operation_name=f"binance_spot_rolling_ticker_{market_symbol.lower()}",
                logger=self.logger,
                attempts=2,
                initial_delay_seconds=1.0,
                max_delay_seconds=2.0,
            )
            ticker = normalize_rolling_ticker(payload)
            signal = evaluate_rolling_24h_opportunity(**ticker, config=self.config)
            indicator_context = None
            indicator_error = None
            if signal["opportunity"] != "hold":
                try:
                    klines = run_binance_with_retries(
                        lambda: self.client.get_klines(
                            symbol=market_symbol,
                            interval=self.indicator_interval,
                            limit=SPOT_INDICATOR_KLINE_LIMIT,
                        ),
                        operation_name=f"binance_spot_indicator_klines_{market_symbol.lower()}",
                        logger=self.logger,
                        attempts=2,
                        initial_delay_seconds=1.0,
                        max_delay_seconds=2.0,
                    )
                    indicator_context = calculate_spot_indicator_context(
                        klines,
                        current_price=signal["current_price"],
                        evaluated_at_ms=signal["current_at_ms"],
                        interval=self.indicator_interval,
                    )
                except Exception as exc:  # noqa: BLE001
                    indicator_error = str(exc)[:500]
            sized_signal = finalize_spot_strategy_signal(
                signal,
                indicator_context,
                policy,
                execution_enabled=self.execution_enabled,
                sizing_config=self.sizing_config,
                indicator_error=indicator_error,
            )
            snapshot = {
                **sized_signal,
                "account_key": self.account_key,
                "asset_symbol": asset_symbol,
                "quote_symbol": quote_symbol,
                "market_symbol": market_symbol,
            }
            saved = save_spot_signal_snapshot(snapshot, account_key=self.account_key, path=self.db_path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:500]
            try:
                mark_spot_signal_snapshot_error(
                    asset_symbol,
                    error,
                    quote_symbol=quote_symbol,
                    account_key=self.account_key,
                    path=self.db_path,
                )
            except OSError as persist_error:
                self.logger.warning(
                    "spot_signal_error_persist_failed symbol=%s error=%s",
                    market_symbol,
                    persist_error,
                )
            if self._last_errors.get(key) != error:
                self.logger.warning("spot_signal_evaluation_failed symbol=%s error=%s", market_symbol, error)
            self._last_errors[key] = error
            return None

        state = (
            saved["opportunity"],
            saved["level"],
            saved["strategy_eligible"],
            saved["eligibility_reason"],
            saved.get("indicator_assessment", {}).get("tier"),
        )
        if self._last_errors.pop(key, None) is not None:
            self.logger.info("spot_signal_evaluation_restored symbol=%s", market_symbol)
        if self._last_states.get(key) != state:
            self.logger.info(
                "spot_signal_changed symbol=%s opportunity=%s level=%s change_pct=%.4f "
                "base_tranche_pct=%.2f sizing_modifier=%.2f final_tranche_pct=%.2f "
                "strategy_eligible=%s eligibility_reason=%s",
                market_symbol,
                saved["opportunity"],
                saved["level"],
                saved["rolling_change_pct"],
                saved["base_tranche_pct"],
                saved["sizing_modifier"],
                saved["final_tranche_pct"],
                saved["strategy_eligible"],
                saved["eligibility_reason"],
            )
        self._last_states[key] = state
        return saved

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh_once()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("spot_signal_monitor_cycle_failed error=%s", exc)
            self._stop_event.wait(self.interval_seconds)
