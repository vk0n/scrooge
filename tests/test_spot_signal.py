import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.spot_signal import RollingSpotSignalMonitor, spot_signal_config_from_env
from shared.runtime_db import (
    bootstrap_runtime_db,
    load_spot_signal_snapshot,
    upsert_portfolio_asset_policy,
)
from shared.spot_signal import ROLLING_WINDOW_MS, SpotSignalConfig, evaluate_rolling_24h_opportunity


class FakeTickerClient:
    def __init__(self, tickers):
        self.tickers = tickers
        self.calls = []

    def get_ticker(self, **kwargs):
        self.calls.append(kwargs)
        return self.tickers[kwargs["symbol"]]


def rolling_ticker(open_price: float, last_price: float) -> dict:
    return {
        "openPrice": str(open_price),
        "lastPrice": str(last_price),
        "openTime": 1_000,
        "closeTime": 1_000 + ROLLING_WINDOW_MS,
    }


class SpotSignalDomainTests(unittest.TestCase):
    def evaluate(self, current_price: float) -> dict:
        return evaluate_rolling_24h_opportunity(
            current_price=current_price,
            reference_price=100,
            current_at_ms=1_000 + ROLLING_WINDOW_MS,
            reference_at_ms=1_000,
        )

    def test_default_levels_create_progressive_sell_opportunities(self):
        expectations = [(104.9, 0, 0), (105, 1, 10), (108, 2, 20), (112, 3, 30), (118, 4, 40)]

        for price, level, tranche in expectations:
            with self.subTest(price=price):
                result = self.evaluate(price)
                self.assertEqual(result["opportunity"], "hold" if level == 0 else "sell")
                self.assertEqual(result["level"], level)
                self.assertEqual(result["base_tranche_pct"], tranche)

    def test_negative_move_creates_buy_opportunity(self):
        result = self.evaluate(88)

        self.assertEqual(result["opportunity"], "buy")
        self.assertEqual(result["level"], 3)
        self.assertEqual(result["base_tranche_pct"], 30)
        self.assertAlmostEqual(result["rolling_change_pct"], -12)

    def test_reference_must_be_approximately_24_hours_old(self):
        with self.assertRaisesRegex(ValueError, "approximately 24 hours"):
            evaluate_rolling_24h_opportunity(
                current_price=110,
                reference_price=100,
                current_at_ms=1_000 + 60 * 60 * 1000,
                reference_at_ms=1_000,
            )

    def test_config_rejects_misaligned_or_unsorted_levels(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            SpotSignalConfig(levels_pct=(5, 8), base_tranches_pct=(10,))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SpotSignalConfig(levels_pct=(8, 5), base_tranches_pct=(10, 20))

    def test_config_is_loaded_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "SCROOGE_SPOT_SIGNAL_LEVELS_PCT": "4,7",
                "SCROOGE_SPOT_SIGNAL_BASE_TRANCHES_PCT": "15,35",
            },
        ):
            config = spot_signal_config_from_env()

        self.assertEqual(config.levels_pct, (4.0, 7.0))
        self.assertEqual(config.base_tranches_pct, (15.0, 35.0))


class RollingSpotSignalMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        bootstrap_runtime_db(self.db_path)

    def policy(self, asset: str, *, objective: str | None, minimum: float = 80) -> None:
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": asset,
                "quote_symbol": "USDT",
                "target_quantity": 100,
                "minimum_holding_pct": minimum,
                "trading_objective": objective,
            },
            path=self.db_path,
        )

    def monitor(self, client, *, execution_enabled: bool = True) -> RollingSpotSignalMonitor:
        return RollingSpotSignalMonitor(
            client,
            interval_seconds=300,
            execution_enabled=execution_enabled,
            logger=logging.getLogger("test.spot-signal"),
            db_path=self.db_path,
        )

    def test_monitor_persists_signal_and_policy_eligibility_separately(self):
        self.policy("NEAR", objective="accumulate_cash")
        self.policy("XRP", objective=None)
        client = FakeTickerClient(
            {
                "NEARUSDT": rolling_ticker(4, 4.4),
                "XRPUSDT": rolling_ticker(1, 0.9),
            }
        )

        results = self.monitor(client).refresh_once()
        near = load_spot_signal_snapshot("NEAR", path=self.db_path)
        xrp = load_spot_signal_snapshot("XRP", path=self.db_path)

        self.assertEqual(len(results), 2)
        self.assertEqual(near["opportunity"], "sell")
        self.assertEqual(near["level"], 2)
        self.assertTrue(near["strategy_eligible"])
        self.assertEqual(near["eligibility_reason"], "eligible")
        self.assertEqual(xrp["opportunity"], "buy")
        self.assertFalse(xrp["strategy_eligible"])
        self.assertEqual(xrp["eligibility_reason"], "trading_objective_unset")

    def test_fully_protected_policy_is_not_strategy_eligible(self):
        self.policy("NEAR", objective="accumulate_asset", minimum=100)
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.8)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["opportunity"], "sell")
        self.assertFalse(snapshot["strategy_eligible"])
        self.assertEqual(snapshot["eligibility_reason"], "fully_protected")

    def test_disabled_monitor_does_not_call_binance(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})

        results = self.monitor(client, execution_enabled=False).refresh_once()

        self.assertEqual(results, [])
        self.assertEqual(client.calls, [])
        self.assertIsNone(load_spot_signal_snapshot("NEAR", path=self.db_path))

    def test_market_error_preserves_last_signal_but_disables_eligibility(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})
        monitor = self.monitor(client)
        monitor.refresh_once()

        with patch("bot.spot_signal.run_binance_with_retries", side_effect=RuntimeError("offline")):
            self.assertEqual(monitor.refresh_once(), [])
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["opportunity"], "sell")
        self.assertFalse(snapshot["strategy_eligible"])
        self.assertEqual(snapshot["eligibility_reason"], "signal_unavailable")
        self.assertEqual(snapshot["error"], "offline")


if __name__ == "__main__":
    unittest.main()
