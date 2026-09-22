import os
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
