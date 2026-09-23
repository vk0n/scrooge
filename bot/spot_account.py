from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.binance_retry import run_binance_with_retries
from shared.runtime_db import mark_exchange_account_snapshot_error, save_exchange_account_snapshot


def normalize_spot_account_snapshot(payload: Any, *, captured_at_ms: int | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Binance Spot account response must be an object.")
    raw_balances = payload.get("balances")
    if not isinstance(raw_balances, list):
        raise ValueError("Binance Spot account response did not include balances.")

    balances: list[dict[str, Any]] = []
    for raw_balance in raw_balances:
        if not isinstance(raw_balance, dict):
            continue
        asset_symbol = str(raw_balance.get("asset") or "").strip().upper()
        try:
            free = float(raw_balance.get("free", 0.0))
            locked = float(raw_balance.get("locked", 0.0))
        except (TypeError, ValueError):
            continue
        if not asset_symbol or abs(free + locked) < 0.000000000001:
            continue
        balances.append(
            {
                "asset_symbol": asset_symbol,
                "free": free,
                "locked": locked,
            }
        )

    return {
        "venue": "binance",
        "account_type": "spot",
        "captured_at_ms": int(captured_at_ms or datetime.now(UTC).timestamp() * 1000),
        "can_trade": bool(payload.get("canTrade")),
        "balances": balances,
    }


class SpotBalanceMonitor:
    def __init__(
        self,
        client: Any,
        *,
        interval_seconds: float,
        logger: Any,
        db_path: Path | None = None,
    ) -> None:
        self.client = client
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.logger = logger
        self.db_path = db_path
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._has_logged_success = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="scrooge-spot-balances", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def refresh_once(self) -> dict[str, Any] | None:
        attempted_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        try:
            payload = run_binance_with_retries(
                lambda: self.client.get_account(recvWindow=5000),
                operation_name="binance_spot_account_snapshot",
                logger=self.logger,
                attempts=2,
                initial_delay_seconds=1.0,
                max_delay_seconds=2.0,
            )
            snapshot = normalize_spot_account_snapshot(payload, captured_at_ms=attempted_at_ms)
            saved = save_exchange_account_snapshot(snapshot, path=self.db_path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:500]
            try:
                mark_exchange_account_snapshot_error(error, attempted_at_ms=attempted_at_ms, path=self.db_path)
            except OSError as persist_error:
                self.logger.warning("spot_balance_snapshot_error_persist_failed error=%s", persist_error)
            if error != self._last_error:
                self.logger.warning("spot_balance_snapshot_failed error=%s", error)
            self._last_error = error
            return None

        if self._last_error is not None:
            self.logger.info("spot_balance_snapshot_restored assets=%s", len(saved["balances"]))
        elif not self._has_logged_success:
            self.logger.info("spot_balance_snapshot_started assets=%s", len(saved["balances"]))
        else:
            self.logger.debug("spot_balance_snapshot_updated assets=%s", len(saved["balances"]))
        self._last_error = None
        self._has_logged_success = True
        return saved

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.refresh_once()
            self._stop_event.wait(self.interval_seconds)
