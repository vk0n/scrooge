import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from api.services import portfolio_service
from bot import spot_execution
from bot.spot_execution import SpotOrderAccountingError, SpotOrderExecutor
from shared.runtime_db import (
    list_portfolio_transactions,
    load_spot_order_intent,
    reserve_spot_order_intent,
    save_exchange_account_snapshot,
    update_spot_order_intent,
)


class FakeSpotExecutionClient:
    def __init__(self, *, btc_free: float = 1.0, usdt_free: float = 100.0):
        self.btc_free = btc_free
        self.usdt_free = usdt_free
        self.create_calls = 0

    def get_account(self, **kwargs):
        return {
            "canTrade": True,
            "balances": [
                {"asset": "BTC", "free": str(self.btc_free), "locked": "0"},
                {"asset": "USDT", "free": str(self.usdt_free), "locked": "0"},
            ],
        }

    def get_symbol_ticker(self, **kwargs):
        return {"symbol": kwargs["symbol"], "price": "100"}

    def get_symbol_info(self, symbol):
        return {
            "symbol": symbol,
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            ],
        }

    def create_order(self, **kwargs):
        self.create_calls += 1
        self.order_params = kwargs
        return {
            "symbol": kwargs["symbol"],
            "orderId": 42,
            "clientOrderId": kwargs["newClientOrderId"],
            "transactTime": 1_790_120_000_000,
            "status": "FILLED",
            "executedQty": kwargs["quantity"],
            "cummulativeQuoteQty": str(float(kwargs["quantity"]) * 100),
        }

    def get_order(self, **kwargs):
        return {
            "symbol": kwargs["symbol"],
            "orderId": 42,
            "clientOrderId": self.order_params["newClientOrderId"],
            "updateTime": 1_790_120_000_000,
            "status": "FILLED",
            "executedQty": self.order_params["quantity"],
            "cummulativeQuoteQty": str(float(self.order_params["quantity"]) * 100),
        }

    def get_my_trades(self, **kwargs):
        quantity = float(self.order_params["quantity"])
        return [
            {
                "orderId": kwargs["orderId"],
                "price": "100",
                "qty": str(quantity),
                "quoteQty": str(quantity * 100),
                "commission": "0.05",
                "commissionAsset": "USDT",
                "time": 1_790_120_000_000,
            }
        ]


class SpotExecutionTests(unittest.TestCase):
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

    def _preview_and_queue(self, side: str, quantity: float):
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
        if side == "sell":
            portfolio_service.update_portfolio_asset_policy(
                "BTC",
                {"target_quantity": 1, "minimum_holding_pct": 50},
            )
        preview = portfolio_service.create_spot_order_preview(
            {"asset_symbol": "BTC", "side": side, "quantity": quantity}
        )
        reserved, acquired = reserve_spot_order_intent(
            preview["intent_id"],
            command_id=preview["intent_id"],
            path=self.db_path,
        )
        self.assertTrue(acquired)
        self.assertIsNotNone(reserved)
        update_spot_order_intent(
            preview["intent_id"],
            {"status": "queued"},
            expected_statuses={"queueing"},
            path=self.db_path,
        )
        return preview

    def test_filled_buy_is_recorded_once_in_treasury_ledger(self):
        preview = self._preview_and_queue("buy", 0.25)
        client = FakeSpotExecutionClient()
        executor = SpotOrderExecutor(client, logger=logging.getLogger("test.spot-execution"), db_path=self.db_path)

        result = executor.execute(preview["intent_id"])

        self.assertEqual(result["status"], "FILLED")
        self.assertEqual(result["executed_quantity"], 0.25)
        self.assertEqual(client.create_calls, 1)
        self.assertEqual(client.order_params["newClientOrderId"], preview["client_order_id"])
        exchange_entries = [
            item
            for item in list_portfolio_transactions(path=self.db_path)
            if item.get("source") == "binance_manual"
        ]
        self.assertEqual(len(exchange_entries), 1)
        self.assertEqual(exchange_entries[0]["tx_type"], "buy")
        self.assertEqual(exchange_entries[0]["external_order_id"], "42")

        replay = executor.execute(preview["intent_id"])
        self.assertEqual(replay["order_id"], "42")
        self.assertEqual(client.create_calls, 1)
        self.assertEqual(
            len([item for item in list_portfolio_transactions(path=self.db_path) if item.get("source") == "binance_manual"]),
            1,
        )

    def test_sell_is_revalidated_against_changed_exchange_balance(self):
        preview = self._preview_and_queue("sell", 0.25)
        client = FakeSpotExecutionClient(btc_free=0.1)
        executor = SpotOrderExecutor(client, logger=logging.getLogger("test.spot-execution"), db_path=self.db_path)

        with self.assertRaisesRegex(ValueError, "immediately sellable"):
            executor.execute(preview["intent_id"])

        self.assertEqual(client.create_calls, 0)
        self.assertFalse(any(item.get("source") == "binance_manual" for item in list_portfolio_transactions(path=self.db_path)))

    def test_confirmed_fill_with_ledger_failure_is_not_marked_as_failed(self):
        preview = self._preview_and_queue("buy", 0.25)
        client = FakeSpotExecutionClient()
        executor = SpotOrderExecutor(client, logger=logging.getLogger("test.spot-execution"), db_path=self.db_path)

        with patch.object(spot_execution, "append_portfolio_transaction", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(SpotOrderAccountingError, "Do not retry"):
                executor.execute(preview["intent_id"])

        intent = load_spot_order_intent(preview["intent_id"], path=self.db_path)
        self.assertEqual(intent["status"], "accounting_error")
        self.assertEqual(intent["exchange_order_id"], "42")
        self.assertEqual(intent["executed_quantity"], 0.25)
        self.assertIn("accounting failed", intent["error"])
        self.assertEqual(client.create_calls, 1)


if __name__ == "__main__":
    unittest.main()
