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
                "summary": {"vault_reserve_available": 40},
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
