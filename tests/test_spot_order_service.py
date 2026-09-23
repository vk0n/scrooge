import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

if "redis" not in sys.modules:
    redis_stub = types.ModuleType("redis")
    redis_stub.Redis = object
    redis_stub.RedisError = RuntimeError
    redis_stub.client = types.SimpleNamespace(Pipeline=object)
    sys.modules["redis"] = redis_stub

from services import portfolio_service, spot_order_service
from shared.runtime_db import save_exchange_account_snapshot


class SpotOrderServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        self.env = patch.dict(
            os.environ,
            {
                "SCROOGE_DB_PATH": str(self.db_path),
                "SCROOGE_SPOT_EXECUTION_ENABLED": "1",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.prices = patch.object(
            portfolio_service,
            "_fetch_market_price",
            return_value=(100.0, None, "2026-09-23 10:00:00"),
        )
        self.prices.start()
        self.addCleanup(self.prices.stop)
        portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "buy",
                "asset_symbol": "BTC",
                "quantity": 1,
                "price": 90,
                "quote_symbol": "USDT",
                "custody_location": "binance",
            }
        )
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [
                    {"asset_symbol": "BTC", "free": 1, "locked": 0},
                    {"asset_symbol": "USDT", "free": 100, "locked": 0},
                ],
            }
        )

    def test_repeated_confirmation_queues_one_command(self):
        preview = portfolio_service.create_spot_order_preview(
            {"asset_symbol": "BTC", "side": "buy", "quantity": 0.25}
        )
        command = {
            "command_id": preview["intent_id"],
            "status": "pending",
            "queued_at": "2026-09-23T10:00:00+00:00",
            "action": "spot_order",
        }
        with patch.object(spot_order_service, "enqueue_control_command", return_value=command) as enqueue:
            first = spot_order_service.queue_spot_order_intent(preview["intent_id"], requested_by="tester")
            second = spot_order_service.queue_spot_order_intent(preview["intent_id"], requested_by="tester")

        self.assertEqual(enqueue.call_count, 1)
        self.assertEqual(first["command_id"], preview["intent_id"])
        self.assertEqual(first["intent"]["status"], "queued")
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(second["command_id"], preview["intent_id"])

    def test_execution_safety_switch_blocks_queueing(self):
        preview = portfolio_service.create_spot_order_preview(
            {"asset_symbol": "BTC", "side": "buy", "quantity": 0.25}
        )
        with patch.dict(os.environ, {"SCROOGE_SPOT_EXECUTION_ENABLED": "0"}):
            with self.assertRaisesRegex(ValueError, "safety switch"):
                spot_order_service.queue_spot_order_intent(preview["intent_id"], requested_by="tester")


if __name__ == "__main__":
    unittest.main()
