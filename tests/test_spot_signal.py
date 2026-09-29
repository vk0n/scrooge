import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.spot_signal import RollingSpotSignalMonitor
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
        expectations = [
            (101.9, 0, 0, 0),
            (102, 1, 10, 1),
            (103, 2, 20, 3),
            (104, 3, 30, 5),
            (106, 4, 40, 10),
        ]

        for price, level, tranche, accumulation_tranche in expectations:
            with self.subTest(price=price):
                result = self.evaluate(price)
                self.assertEqual(result["opportunity"], "hold" if level == 0 else "sell")
                self.assertEqual(result["level"], level)
                self.assertEqual(result["base_tranche_pct"], tranche)
                self.assertEqual(result["accumulation_tranche_pct"], accumulation_tranche)

    def test_negative_move_creates_buy_opportunity(self):
        result = self.evaluate(96)

        self.assertEqual(result["opportunity"], "buy")
        self.assertEqual(result["level"], 3)
        self.assertEqual(result["accumulation_tranche_pct"], 5)
        self.assertEqual(result["base_tranche_pct"], 30)
        self.assertAlmostEqual(result["rolling_change_pct"], -4)

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

    def test_monitor_uses_cash_objective_when_policy_does_not_specify_one(self):
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
        self.assertEqual(near["level"], 4)
        self.assertTrue(near["strategy_eligible"])
        self.assertEqual(near["eligibility_reason"], "eligible")
        self.assertEqual(xrp["opportunity"], "buy")
        self.assertEqual(xrp["trading_objective"], "accumulate_cash")
        self.assertTrue(xrp["strategy_eligible"])
        self.assertEqual(xrp["eligibility_reason"], "eligible")

    def test_monitor_orders_complete_signal_batch_before_execution(self):
        self.policy("NEAR", objective="accumulate_cash")
        self.policy("XRP", objective="accumulate_cash")
        client = FakeTickerClient(
            {
                "NEARUSDT": rolling_ticker(4, 4.4),
                "XRPUSDT": rolling_ticker(1, 0.9),
            }
        )
        handled: list[str] = []
        monitor = RollingSpotSignalMonitor(
            client,
            interval_seconds=300,
            execution_enabled=True,
            logger=logging.getLogger("test.spot-signal"),
            db_path=self.db_path,
            snapshot_handler=lambda signal: handled.append(signal["asset_symbol"]),
            snapshot_orderer=lambda signals: list(reversed(signals)),
        )

        results = monitor.refresh_once()

        self.assertEqual([item["asset_symbol"] for item in results], ["NEAR", "XRP"])
        self.assertEqual(handled, ["XRP", "NEAR"])

    def test_monitor_recovers_pending_orders_before_loading_new_signals(self):
        events = []
        monitor = RollingSpotSignalMonitor(
            FakeTickerClient({}),
            interval_seconds=300,
            execution_enabled=True,
            logger=logging.getLogger("test.spot-signal"),
            db_path=self.db_path,
            pending_recovery_handler=lambda: events.append("recovered"),
        )

        self.assertEqual(monitor.refresh_once(), [])
        self.assertEqual(events, ["recovered"])

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

    def test_actionable_signal_uses_fixed_tranche_without_indicator_request(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["final_tranche_pct"], 40)
        self.assertNotIn("indicator_context", snapshot)
        self.assertNotIn("sizing_modifier", snapshot)
        self.assertEqual(client.calls, [{"symbol": "NEARUSDT"}])

    def test_hold_keeps_zero_final_tranche(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.05)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["opportunity"], "hold")
        self.assertEqual(snapshot["final_tranche_pct"], 0)


if __name__ == "__main__":
    unittest.main()
