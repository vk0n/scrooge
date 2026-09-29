import unittest
from unittest.mock import Mock, patch

from bot.treasury_transfer import TreasuryTransferExecutor


class TreasuryTransferExecutorTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.logger = Mock()
        self.executor = TreasuryTransferExecutor(self.client, logger=self.logger)
        self.enabled = patch("bot.treasury_transfer.treasury_transfer_enabled", return_value=True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    @patch("bot.treasury_transfer.record_confirmed_office_transfer")
    @patch("bot.treasury_transfer.load_portfolio_snapshot")
    def test_transfer_to_office_revalidates_free_treasury_cash(self, load_snapshot, record_transfer):
        load_snapshot.return_value = (
            {
                "summary": {
                    "vault_reserve_available": 40,
                    "vault_reserve_spendable": 40,
                    "vault_reserve_retained": 0,
                },
                "exchange": {"usdt_free": 35},
            },
            [],
        )
        self.client.universal_transfer.return_value = {"tranId": 123}
        record_transfer.return_value = (
            {"transaction": {"transaction_id": "office-transfer:ref-1"}},
            [],
        )

        result = self.executor.execute(
            {"direction": "to_office", "quantity": 30, "transfer_ref": "ref-1"}
        )

        self.client.universal_transfer.assert_called_once_with(
            type="MAIN_UMFUTURE",
            asset="USDT",
            amount="30",
            clientTranId="ref-1",
        )
        self.assertEqual(result["ledger_transaction_id"], "office-transfer:ref-1")

    @patch("bot.treasury_transfer.consume_portfolio_retained_cash")
    @patch("bot.treasury_transfer.record_confirmed_office_transfer")
    @patch("bot.treasury_transfer.load_portfolio_snapshot")
    def test_transfer_to_office_requires_opt_in_and_consumes_only_protected_remainder(
        self,
        load_snapshot,
        record_transfer,
        consume_retained,
    ):
        load_snapshot.return_value = (
            {
                "summary": {
                    "vault_reserve_available": 40,
                    "vault_reserve_spendable": 10,
                    "vault_reserve_retained": 30,
                },
                "exchange": {"usdt_free": 40},
            },
            [],
        )
        self.client.universal_transfer.return_value = {"tranId": 456}
        record_transfer.return_value = (
            {"transaction": {"transaction_id": "office-transfer:protected-ref"}},
            [],
        )
        consume_retained.return_value = {"consumed_quote": 10, "reference_id": "protected-use"}

        with self.assertRaisesRegex(ValueError, "Enable Protected Cash"):
            self.executor.execute(
                {"direction": "to_office", "quantity": 20, "transfer_ref": "protected-ref"}
            )

        result = self.executor.execute(
            {
                "direction": "to_office",
                "quantity": 20,
                "transfer_ref": "protected-ref",
                "use_protected_cash": True,
            }
        )

        consume_retained.assert_called_once()
        self.assertEqual(consume_retained.call_args.args[0], 10)
        self.assertEqual(result["protected_cash_used"], 10)

    @patch("bot.treasury_transfer.consume_portfolio_retained_cash")
    @patch("bot.treasury_transfer.record_confirmed_office_transfer")
    @patch("bot.treasury_transfer.load_portfolio_snapshot")
    def test_transfer_to_office_can_draw_only_from_retained_cash(
        self,
        load_snapshot,
        record_transfer,
        consume_retained,
    ):
        load_snapshot.return_value = (
            {
                "summary": {
                    "vault_reserve_available": 100,
                    "vault_reserve_spendable": 80,
                    "vault_reserve_retained": 20,
                },
                "exchange": {"usdt_free": 100},
            },
            [],
        )
        self.client.universal_transfer.return_value = {"tranId": 789}
        record_transfer.return_value = (
            {
                "transaction": {
                    "transaction_id": "office-transfer:retained-ref",
                    "protected_cash_required": 15,
                }
            },
            [],
        )
        consume_retained.return_value = {"consumed_quote": 15}

        result = self.executor.execute(
            {
                "direction": "to_office",
                "quantity": 15,
                "transfer_ref": "retained-ref",
                "cash_bucket": "retained",
            }
        )

        consume_retained.assert_called_once()
        self.assertEqual(consume_retained.call_args.args[0], 15)
        self.assertEqual(result["cash_bucket"], "retained")
        self.assertEqual(result["protected_cash_used"], 15)

    @patch("bot.treasury_transfer.credit_portfolio_retained_cash")
    @patch("bot.treasury_transfer.record_confirmed_office_transfer")
    @patch("bot.treasury_transfer.load_portfolio_snapshot")
    def test_transfer_from_office_can_credit_retained_cash(
        self,
        load_snapshot,
        record_transfer,
        credit_retained,
    ):
        load_snapshot.return_value = ({"summary": {}, "exchange": {}}, [])
        self.client.futures_account_balance.return_value = [
            {"asset": "USDT", "balance": "100", "availableBalance": "25"}
        ]
        self.client.universal_transfer.return_value = {"tranId": 790}
        record_transfer.return_value = (
            {"transaction": {"transaction_id": "office-transfer:credit-ref"}},
            [],
        )
        credit_retained.return_value = {"credited_quote": 12}

        result = self.executor.execute(
            {
                "direction": "from_office",
                "quantity": 12,
                "transfer_ref": "credit-ref",
                "cash_bucket": "retained",
            }
        )

        credit_retained.assert_called_once()
        self.assertEqual(result["protected_cash_credited"], 12)

    @patch("bot.treasury_transfer.record_confirmed_office_transfer")
    @patch("bot.treasury_transfer.load_portfolio_snapshot")
    def test_transfer_from_office_checks_futures_available_balance(self, load_snapshot, record_transfer):
        load_snapshot.return_value = ({"summary": {}, "exchange": {}}, [])
        self.client.futures_account_balance.return_value = [
            {"asset": "USDT", "balance": "100", "availableBalance": "12.5"}
        ]

        with self.assertRaisesRegex(ValueError, "exceeds available"):
            self.executor.execute(
                {"direction": "from_office", "quantity": 13, "transfer_ref": "ref-2"}
            )

        self.client.universal_transfer.assert_not_called()
        record_transfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
