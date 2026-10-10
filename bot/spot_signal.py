from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from core.binance_retry import run_binance_with_retries
from core.runtime_db import (
    list_portfolio_asset_policies,
    mark_spot_signal_snapshot_error,
    save_spot_signal_snapshot,
)
from core.spot_signal import SpotSignalConfig, evaluate_rolling_24h_opportunity
from core.spot_strategy import finalize_spot_strategy_signal, spot_policy_eligibility

DEFAULT_ACCOUNT_KEY = "manual_spot"


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
        account_key: str = DEFAULT_ACCOUNT_KEY,
        snapshot_handler: Callable[[dict[str, Any]], Any] | None = None,
        snapshot_orderer: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        pending_recovery_handler: Callable[[], Any] | None = None,
    ) -> None:
        self.client = client
        self.interval_seconds = max(30.0, float(interval_seconds))
        self.execution_enabled = bool(execution_enabled)
        self.logger = logger
        self.db_path = db_path
        self.config = config or SpotSignalConfig()
        self.account_key = str(account_key or DEFAULT_ACCOUNT_KEY).strip() or DEFAULT_ACCOUNT_KEY
        self.snapshot_handler = snapshot_handler
        self.snapshot_orderer = snapshot_orderer
        self.pending_recovery_handler = pending_recovery_handler
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_states: dict[tuple[str, str], tuple[object, ...]] = {}
        self._last_errors: dict[tuple[str, str], str] = {}

    def start(self, *, refresh_immediately: bool = True) -> None:
        if not self.execution_enabled:
            self.logger.info("spot_signal_monitor_disabled reason=execution_disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"refresh_immediately": refresh_immediately},
            name="scrooge-spot-signals",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> bool:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        if self._thread is not None and self._thread.is_alive():
            self.logger.error("spot_signal_monitor_stop_timed_out")
            return False
        self._thread = None
        return True

    def refresh_once(self) -> list[dict[str, Any]]:
        if not self.execution_enabled:
            return []
        cycle_started = time.monotonic()
        if self.pending_recovery_handler is not None:
            try:
                self.pending_recovery_handler()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("spot_order_cycle_recovery_failed error=%s", exc)
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
        elapsed_seconds = time.monotonic() - cycle_started
        if elapsed_seconds >= self.interval_seconds:
            self.logger.warning(
                "spot_signal_cycle_slow signals=%s elapsed_seconds=%.2f interval_seconds=%.2f",
                len(results),
                elapsed_seconds,
                self.interval_seconds,
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
            finalized_signal = finalize_spot_strategy_signal(
                signal,
                policy,
                execution_enabled=self.execution_enabled,
            )
            snapshot = {
                **finalized_signal,
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
            saved.get("accumulation_tranche_pct"),
        )
        if self._last_errors.pop(key, None) is not None:
            self.logger.info("spot_signal_evaluation_restored symbol=%s", market_symbol)
        if self._last_states.get(key) != state:
            self.logger.info(
                "spot_signal_changed symbol=%s opportunity=%s level=%s change_pct=%.4f "
                "base_tranche_pct=%.2f accumulation_tranche_pct=%.2f "
                "final_tranche_pct=%.2f "
                "strategy_eligible=%s eligibility_reason=%s",
                market_symbol,
                saved["opportunity"],
                saved["level"],
                saved["rolling_change_pct"],
                saved["base_tranche_pct"],
                saved.get("accumulation_tranche_pct") or 0.0,
                saved["final_tranche_pct"],
                saved["strategy_eligible"],
                saved["eligibility_reason"],
            )
        self._last_states[key] = state
        return saved

    def _run(self, *, refresh_immediately: bool = True) -> None:
        if not refresh_immediately and self._stop_event.wait(self.interval_seconds):
            return
        while not self._stop_event.is_set():
            try:
                self.refresh_once()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("spot_signal_monitor_cycle_failed error=%s", exc)
            self._stop_event.wait(self.interval_seconds)
