import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from services import portfolio_service


class PortfolioPhaseOneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"SCROOGE_DB_PATH": str(Path(self.tmp.name) / "runtime.sqlite3")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.prices = patch.object(
            portfolio_service,
            "_fetch_market_price",
            side_effect=lambda asset, quote: (
                {"BTC": 100.0, "ETH": 20.0}.get(asset),
                None,
                "2026-09-22 13:45:00",
            ),
        )
        self.prices.start()
        self.addCleanup(self.prices.stop)

    def add(self, asset: str, quantity: float, price: float) -> dict:
        result, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "buy",
                "asset_symbol": asset,
                "quantity": quantity,
                "price": price,
                "quote_symbol": "USDT",
            }
        )
        return result["transaction"]

    def test_holdings_are_sorted_by_allocation(self):
        self.add("ETH", 1, 10)
        self.add("BTC", 1, 90)
        snapshot, _ = portfolio_service.load_portfolio_snapshot()
        self.assertEqual([holding["asset_symbol"] for holding in snapshot["holdings"]], ["BTC", "ETH"])
        self.assertEqual([round(holding["allocation_pct"], 2) for holding in snapshot["holdings"]], [83.33, 16.67])
        self.assertEqual(snapshot["summary"]["prices_updated_at"], "2026-09-22 13:45:00")
        self.assertTrue(all(holding["market_price_updated_at"] for holding in snapshot["holdings"]))

    def test_void_and_restore_recalculate_holdings_without_deleting_ledger_entry(self):
        btc = self.add("BTC", 1, 90)
        self.add("ETH", 1, 10)

        result, _ = portfolio_service.set_portfolio_transaction_status(btc["transaction_id"], "voided")
        self.assertEqual(result["transaction"]["status"], "voided")
        self.assertEqual([holding["asset_symbol"] for holding in result["portfolio"]["holdings"]], ["ETH"])
        self.assertEqual(result["portfolio"]["transaction_count"], 2)

        restored, _ = portfolio_service.set_portfolio_transaction_status(btc["transaction_id"], "settled")
        self.assertEqual(restored["transaction"]["status"], "settled")
        self.assertEqual([holding["asset_symbol"] for holding in restored["portfolio"]["holdings"]], ["BTC", "ETH"])

    def test_unknown_transaction_cannot_be_voided(self):
        with self.assertRaisesRegex(LookupError, "not found"):
            portfolio_service.set_portfolio_transaction_status("missing", "voided")

    def test_reduce_cannot_exceed_current_stack(self):
        self.add("BTC", 1, 90)
        with self.assertRaisesRegex(ValueError, "below zero"):
            portfolio_service.create_portfolio_transaction(
                {
                    "tx_type": "sell",
                    "asset_symbol": "BTC",
                    "quantity": 1.1,
                    "price": 100,
                    "quote_symbol": "USDT",
                }
            )
        snapshot, _ = portfolio_service.load_portfolio_snapshot()
        self.assertEqual(snapshot["transaction_count"], 1)
        self.assertEqual(snapshot["holdings"][0]["quantity"], 1)

    def test_void_cannot_make_stack_negative(self):
        buy = self.add("BTC", 1, 90)
        portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "withdraw",
                "asset_symbol": "BTC",
                "quantity": 1,
                "quote_symbol": "USDT",
            }
        )
        with self.assertRaisesRegex(ValueError, "below zero"):
            portfolio_service.set_portfolio_transaction_status(buy["transaction_id"], "voided")
        snapshot, _ = portfolio_service.load_portfolio_snapshot()
        self.assertEqual(snapshot["transactions"][-1]["status"], "settled")

    def test_ledger_returns_five_latest_transactions_with_pagination(self):
        transaction_ids = [self.add("BTC", 1, 90)["transaction_id"] for _ in range(7)]

        first_page, _ = portfolio_service.load_portfolio_snapshot()
        second_page, _ = portfolio_service.load_portfolio_snapshot(transaction_offset=5)

        self.assertEqual(first_page["transaction_count"], 7)
        self.assertEqual(first_page["transaction_limit"], 5)
        self.assertEqual(first_page["transaction_offset"], 0)
        self.assertEqual(
            [transaction["transaction_id"] for transaction in first_page["transactions"]],
            list(reversed(transaction_ids))[:5],
        )
        self.assertEqual(second_page["transaction_offset"], 5)
        self.assertEqual(
            [transaction["transaction_id"] for transaction in second_page["transactions"]],
            list(reversed(transaction_ids))[5:],
        )

    def test_timeline_keeps_one_latest_valuation_per_day(self):
        self.add("BTC", 1, 90)
        self.add("ETH", 1, 10)

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(len(snapshot["timeline"]), 1)
        point = snapshot["timeline"][0]
        self.assertEqual(point["total_value"], 120.0)
        self.assertEqual(point["invested_capital"], 100.0)
        self.assertEqual(point["unrealized_pnl"], 20.0)
        self.assertEqual(
            [holding["asset_symbol"] for holding in point["holdings"]],
            ["BTC", "ETH"],
        )

    def test_timeline_skips_incomplete_market_valuation(self):
        self.prices.stop()
        with patch.object(
            portfolio_service,
            "_fetch_market_price",
            return_value=(None, "price unavailable", None),
        ):
            self.add("SOL", 1, 10)
            snapshot, warnings = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(snapshot["timeline"], [])
        self.assertTrue(any("Timeline was not updated" in warning for warning in warnings))

    def test_existing_entries_default_to_unassigned_custody(self):
        self.add("BTC", 1, 90)

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        holding = snapshot["holdings"][0]
        self.assertEqual(holding["quantity"], 1)
        self.assertEqual(holding["unassigned_quantity"], 1)
        self.assertEqual(holding["binance_quantity"], 0)
        self.assertEqual(snapshot["transactions"][0]["custody_location"], "unassigned")

    def test_custody_transfer_preserves_total_quantity_and_cost_basis(self):
        self.add("BTC", 2, 90)
        before, _ = portfolio_service.load_portfolio_snapshot()

        result, _ = portfolio_service.create_custody_transfer(
            {
                "asset_symbol": "BTC",
                "quote_symbol": "USDT",
                "quantity": 0.75,
                "source_custody": "unassigned",
                "destination_custody": "binance",
            }
        )

        before_holding = before["holdings"][0]
        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(holding["quantity"], before_holding["quantity"])
        self.assertEqual(holding["invested_capital"], before_holding["invested_capital"])
        self.assertEqual(holding["binance_quantity"], 0.75)
        self.assertEqual(holding["unassigned_quantity"], 1.25)
        self.assertEqual(result["transaction"]["tx_type"], "custody_transfer")

    def test_custody_transfer_cannot_exceed_source_location(self):
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

        with self.assertRaisesRegex(ValueError, "binance below zero"):
            portfolio_service.create_custody_transfer(
                {
                    "asset_symbol": "BTC",
                    "quantity": 1.1,
                    "source_custody": "binance",
                    "destination_custody": "cold_storage",
                }
            )

    def test_custody_transfer_can_be_voided_and_restored(self):
        self.add("BTC", 1, 90)
        created, _ = portfolio_service.create_custody_transfer(
            {
                "asset_symbol": "BTC",
                "quantity": 1,
                "source_custody": "unassigned",
                "destination_custody": "cold_storage",
            }
        )
        transaction_id = created["transaction"]["transaction_id"]

        voided, _ = portfolio_service.set_portfolio_transaction_status(transaction_id, "voided")
        self.assertEqual(voided["portfolio"]["holdings"][0]["unassigned_quantity"], 1)
        restored, _ = portfolio_service.set_portfolio_transaction_status(transaction_id, "settled")
        self.assertEqual(restored["portfolio"]["holdings"][0]["cold_storage_quantity"], 1)

    def test_schema_migrates_old_portfolio_rows_to_unassigned(self):
        database_path = Path(self.tmp.name) / "runtime.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE portfolio_accounts (
                    account_key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    account_type TEXT NOT NULL DEFAULT 'spot',
                    base_currency TEXT NOT NULL DEFAULT 'USDT',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE TABLE portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL UNIQUE,
                    account_key TEXT NOT NULL,
                    executed_at_ms INTEGER NOT NULL,
                    executed_at_text TEXT NOT NULL,
                    tx_type TEXT NOT NULL,
                    asset_symbol TEXT NOT NULL,
                    quote_symbol TEXT NOT NULL DEFAULT 'USDT',
                    quantity REAL NOT NULL,
                    price REAL,
                    fee_amount REAL,
                    fee_asset TEXT,
                    source TEXT NOT NULL DEFAULT 'manual',
                    status TEXT NOT NULL DEFAULT 'settled',
                    note TEXT,
                    external_order_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                INSERT INTO portfolio_accounts VALUES ('manual_spot', 'Manual Spot', 'spot', 'USDT', 0, 1, 1);
                INSERT INTO portfolio_transactions (
                    transaction_id, account_key, executed_at_ms, executed_at_text, tx_type,
                    asset_symbol, quote_symbol, quantity, price, source, status, payload_json,
                    created_at_ms, updated_at_ms
                ) VALUES ('legacy', 'manual_spot', 1, '2026-01-01 00:00:00', 'buy',
                    'BTC', 'USDT', 1, 90, 'manual', 'settled', '{}', 1, 1);
                """
            )

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(snapshot["holdings"][0]["unassigned_quantity"], 1)
        self.assertEqual(snapshot["transactions"][0]["custody_location"], "unassigned")


if __name__ == "__main__":
    unittest.main()
