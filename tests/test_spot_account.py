import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.spot_account import SpotBalanceMonitor, normalize_spot_account_snapshot
from shared.runtime_db import load_exchange_account_snapshot


class FakeSpotClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get_account(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.payload


class SpotAccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        self.env = patch.dict(os.environ, {"SCROOGE_DB_PATH": str(self.db_path)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_normalize_snapshot_keeps_only_nonzero_balances(self):
        snapshot = normalize_spot_account_snapshot(
            {
                "canTrade": True,
                "balances": [
                    {"asset": "BTC", "free": "0.4", "locked": "0.1"},
                    {"asset": "ETH", "free": "0", "locked": "0"},
                ],
            },
            captured_at_ms=123,
        )

        self.assertTrue(snapshot["can_trade"])
        self.assertEqual(snapshot["captured_at_ms"], 123)
        self.assertEqual(snapshot["balances"], [{"asset_symbol": "BTC", "free": 0.4, "locked": 0.1}])

    def test_monitor_persists_sanitized_spot_balances(self):
        client = FakeSpotClient(
            {
                "canTrade": True,
                "balances": [{"asset": "BTC", "free": "0.4", "locked": "0.1"}],
            }
        )
        monitor = SpotBalanceMonitor(
            client,
            interval_seconds=60,
            logger=logging.getLogger("test.spot-account"),
            db_path=self.db_path,
        )

        result = monitor.refresh_once()
        stored = load_exchange_account_snapshot(path=self.db_path)

        self.assertIsNotNone(result)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.kwargs, {"recvWindow": 5000})
        self.assertEqual(stored["status"], "ok")
        self.assertEqual(stored["balances"][0]["total"], 0.5)

    def test_monitor_marks_failure_without_discarding_last_balance(self):
        client = FakeSpotClient(
            {
                "canTrade": True,
                "balances": [{"asset": "BTC", "free": "0.4", "locked": "0.1"}],
            }
        )
        monitor = SpotBalanceMonitor(
            client,
            interval_seconds=60,
            logger=logging.getLogger("test.spot-account"),
            db_path=self.db_path,
        )
        monitor.refresh_once()

        with patch("bot.spot_account.run_binance_with_retries", side_effect=RuntimeError("offline")):
            self.assertIsNone(monitor.refresh_once())

        stored = load_exchange_account_snapshot(path=self.db_path)
        self.assertEqual(stored["status"], "error")
        self.assertEqual(stored["error"], "offline")
        self.assertEqual(stored["balances"][0]["total"], 0.5)


if __name__ == "__main__":
    unittest.main()
