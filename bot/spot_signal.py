from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from core.binance_retry import run_binance_with_retries
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

DEFAULT_ACCOUNT_KEY = "manual_spot"


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
    if not execution_enabled:
        return False, "execution_disabled"
    objective = str(policy.get("trading_objective") or "").strip().lower()
    if objective not in {"accumulate_cash", "accumulate_asset"}:
        return False, "trading_objective_unset"
    try:
        minimum_holding_pct = float(policy.get("minimum_holding_pct", 100.0))
    except (TypeError, ValueError):
        return False, "invalid_policy"
    if minimum_holding_pct >= 100.0:
        return False, "fully_protected"
    return True, "eligible"


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
    ) -> None:
        self.client = client
        self.interval_seconds = max(30.0, float(interval_seconds))
        self.execution_enabled = bool(execution_enabled)
        self.logger = logger
        self.db_path = db_path
        self.config = config or spot_signal_config_from_env()
        self.account_key = str(account_key or DEFAULT_ACCOUNT_KEY).strip() or DEFAULT_ACCOUNT_KEY
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
            strategy_eligible, eligibility_reason = _policy_eligibility(
                policy,
                execution_enabled=self.execution_enabled,
            )
            snapshot = {
                **signal,
                "account_key": self.account_key,
                "asset_symbol": asset_symbol,
                "quote_symbol": quote_symbol,
                "market_symbol": market_symbol,
                "trading_objective": policy.get("trading_objective"),
                "strategy_eligible": strategy_eligible,
                "eligibility_reason": eligibility_reason,
                "evaluated_at_ms": signal["current_at_ms"],
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
        )
        if self._last_errors.pop(key, None) is not None:
            self.logger.info("spot_signal_evaluation_restored symbol=%s", market_symbol)
        if self._last_states.get(key) != state:
            self.logger.info(
                "spot_signal_changed symbol=%s opportunity=%s level=%s change_pct=%.4f "
                "base_tranche_pct=%.2f strategy_eligible=%s eligibility_reason=%s",
                market_symbol,
                saved["opportunity"],
                saved["level"],
                saved["rolling_change_pct"],
                saved["base_tranche_pct"],
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
