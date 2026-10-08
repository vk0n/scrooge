from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from shared.runtime_db import (
    append_portfolio_transaction,
    append_ui_log_entry,
    count_ledger_entries,
    list_ledger_entries,
)
from shared.treasury_ledger import (
    project_portfolio_transaction,
    project_portfolio_transactions,
    spot_order_settlement_message,
    treasury_transaction_presentation,
)


class LedgerProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        self.env = patch.dict(os.environ, {"SCROOGE_DB_PATH": str(self.db_path)})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_existing_ui_logs_migrate_to_trades_scope(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TABLE ui_log_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sort_ts_ms INTEGER,
                    ts_text TEXT,
                    line_text TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO ui_log_entries(sort_ts_ms, ts_text, line_text, created_at_ms) VALUES (?, ?, ?, ?)",
                (1000, "2026-01-01 00:00:01", "[2026-01-01 00:00:01] Old trade event.", 1000),
            )
            connection.commit()

        entries, _ = list_ledger_entries(scope="trades", path=self.db_path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["scope"], "trades")
        self.assertEqual(entries[0]["message"], "Old trade event.")

    def test_scope_cursor_and_source_reference_are_stable(self) -> None:
        append_ui_log_entry(
            "2026-01-01 00:00:01",
            "[2026-01-01 00:00:01] Trade.",
            self.db_path,
            entry_id="trade-1",
            scope="trades",
            message="Trade.",
        )
        first_insert = append_ui_log_entry(
            "2026-01-01 00:00:02",
            "[2026-01-01 00:00:02] Treasury.",
            self.db_path,
            entry_id="treasury-1",
            scope="treasury",
            message="Treasury.",
            source_ref="tx:1",
        )
        duplicate_insert = append_ui_log_entry(
            "2026-01-01 00:00:03",
            "[2026-01-01 00:00:03] Duplicate.",
            self.db_path,
            entry_id="treasury-duplicate",
            scope="treasury",
            message="Duplicate.",
            source_ref="tx:1",
        )

        first_page, cursor = list_ledger_entries(scope="all", limit=1, path=self.db_path)
        second_page, _ = list_ledger_entries(scope="all", limit=1, before=cursor, path=self.db_path)
        treasury_entries, _ = list_ledger_entries(scope="treasury", path=self.db_path)

        self.assertTrue(first_insert)
        self.assertFalse(duplicate_insert)
        self.assertEqual(first_page[0]["entry_id"], "treasury-1")
        self.assertEqual(second_page[0]["entry_id"], "trade-1")
        self.assertEqual([entry["entry_id"] for entry in treasury_entries], ["treasury-1"])
        self.assertEqual(count_ledger_entries(scope="all", path=self.db_path), 2)
        self.assertEqual(count_ledger_entries(scope="treasury", path=self.db_path), 1)

    def test_portfolio_backfill_is_idempotent_and_formats_spot_fill(self) -> None:
        transaction = append_portfolio_transaction(
            {
                "transaction_id": "spot-order:intent-1",
                "account_key": "manual_spot",
                "executed_at": "2026-01-01 12:00:00",
                "tx_type": "buy",
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "quantity": 12.5,
                "price": 4.2,
                "source": "binance_manual",
                "status": "settled",
                "custody_location": "binance",
            },
            path=self.db_path,
        )

        self.assertTrue(project_portfolio_transaction(transaction, path=self.db_path))
        self.assertEqual(project_portfolio_transactions(path=self.db_path), 0)
        entries, _ = list_ledger_entries(scope="treasury", path=self.db_path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source_ref"], "portfolio_transaction:spot-order:intent-1")
        self.assertEqual(
            entries[0]["message"],
            "I bought 12.5 NEAR at $4.2 on Binance Spot at your request.",
        )

    def test_strategy_spot_messages_explain_open_profit_and_cleanup(self) -> None:
        cases = (
            (
                {
                    "tx_type": "sell",
                    "asset_symbol": "TIA",
                    "quantity": 45.14,
                    "price": 0.4645,
                    "source": "binance_strategy",
                    "reason": {
                        "action_type": "open",
                        "signal_level": 3,
                        "rolling_change_pct": 4.2,
                        "level_allocation_pct": 30,
                    },
                },
                "I sold 45.14 TIA at $0.4645 on Binance Spot. L3 rise +4.2%; I put the 30% campaign stake to work.",
            ),
            (
                {
                    "tx_type": "buy",
                    "asset_symbol": "FIL",
                    "quantity": 9,
                    "price": 1.1007,
                    "source": "binance_strategy",
                    "reason": {
                        "action_type": "close",
                        "close_reason": "profit_target",
                        "favorable_move_pct": 4.34,
                        "close_profit_pct": 4,
                    },
                },
                "I bought back 9 FIL at $1.1007 on Binance Spot. Bargain Goal cleared at +4.34% against 4%.",
            ),
            (
                {
                    "tx_type": "buy",
                    "asset_symbol": "DYDX",
                    "quantity": 650,
                    "price": 0.14006148,
                    "source": "binance_strategy",
                    "reason": {
                        "action_type": "close",
                        "close_reason": "deep_loss_cleanup",
                        "age_days": 18.25,
                        "unrealized_pnl_pct_before_cleanup": -27.8,
                        "actual_reverse_signal_level": 1,
                    },
                },
                "I bought back 650 DYDX at $0.14006148 on Binance Spot. Deep-loss cleanup: 18.2d old, -27.8%, reverse L1.",
            ),
        )

        for transaction, expected in cases:
            with self.subTest(expected=expected):
                _, _, message = treasury_transaction_presentation(transaction)
                self.assertEqual(message, expected)

    def test_internal_spot_cash_legs_are_not_projected_to_the_ledger(self) -> None:
        quote_leg = append_portfolio_transaction(
            {
                "transaction_id": "spot-order:intent-2:quote",
                "account_key": "manual_spot",
                "executed_at": "2026-01-01 12:00:00",
                "tx_type": "buy",
                "asset_symbol": "USDT",
                "quote_symbol": "USDT",
                "quantity": 50,
                "price": 1,
                "source": "binance_strategy",
                "status": "settled",
                "custody_location": "binance",
                "spot_quote_leg": True,
            },
            path=self.db_path,
        )

        self.assertFalse(project_portfolio_transaction(quote_leg, path=self.db_path))
        self.assertEqual(project_portfolio_transactions(path=self.db_path), 0)
        entries, _ = list_ledger_entries(scope="treasury", path=self.db_path)
        self.assertEqual(entries, [])

    def test_deferred_spot_transaction_is_projected_once_after_settlement(self) -> None:
        transaction = append_portfolio_transaction(
            {
                "transaction_id": "spot-order:atomic-intent",
                "account_key": "manual_spot",
                "executed_at": "2026-01-01 12:00:00",
                "tx_type": "buy",
                "asset_symbol": "XRP",
                "quote_symbol": "USDT",
                "quantity": 5.2,
                "price": 1.343,
                "source": "binance_strategy",
                "status": "settled",
                "custody_location": "binance",
                "ledger_projection_deferred": True,
                "reason": {
                    "action_type": "accumulate_asset",
                    "signal_level": 4,
                    "rolling_change_pct": -5.23,
                    "accumulation_tranche_pct": 10,
                },
            },
            path=self.db_path,
        )

        self.assertEqual(project_portfolio_transactions(path=self.db_path), 0)
        entries, _ = list_ledger_entries(scope="treasury", path=self.db_path)
        self.assertEqual(entries, [])

        settlement = {
            "accumulation_ratchet": {
                "deployed_quote_quantity": 6.98,
                "applied_gain_quantity": 5.2,
                "next_target_quantity": 2_006.8,
            }
        }
        self.assertTrue(
            project_portfolio_transaction(
                transaction,
                settlement=settlement,
                path=self.db_path,
            )
        )
        self.assertFalse(
            project_portfolio_transaction(
                transaction,
                settlement=settlement,
                path=self.db_path,
            )
        )
        entries, _ = list_ledger_entries(scope="treasury", path=self.db_path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source_ref"], "portfolio_transaction:spot-order:atomic-intent")
        self.assertEqual(
            entries[0]["message"],
            "I bought 5.2 XRP at $1.343 on Binance Spot. L4 dip -5.23%; I deployed 10% of "
            "Spendable Reserve. The fill used $6.98 from Free Vault Reserve and raised Target "
            "by 5.2 XRP to 2,006.8 XRP. Result: Target +5.2 XRP.",
        )

    def test_close_settlement_combines_retention_and_target_effects(self) -> None:
        transaction = {
            "tx_type": "buy",
            "asset_symbol": "FIL",
            "quantity": 9,
            "price": 1.1007,
            "source": "binance_strategy",
            "reason": {
                "action_type": "close",
                "close_reason": "profit_target",
                "favorable_move_pct": 6.27,
                "close_profit_pct": 6,
            },
        }

        cash_message = spot_order_settlement_message(
            transaction,
            {
                "cash_retention": {
                    "retained_quote": 9.45,
                    "eligible_cash_gain_quote": 15.75,
                },
                "swing_economics": {
                    "status": "closed",
                    "trading_objective": "accumulate_cash",
                    "realized_pnl_quote": 15.75,
                    "realized_pnl_pct": 6.27,
                },
            },
        )
        asset_message = spot_order_settlement_message(
            transaction,
            {
                "target_ratchet": {
                    "applied_gain_quantity": 0.52,
                    "next_target_quantity": 1_928.52,
                },
                "swing_economics": {
                    "status": "closed",
                    "trading_objective": "accumulate_asset",
                    "realized_net_asset_change": 0.52,
                    "realized_net_asset_change_pct": 6.27,
                },
            },
        )

        self.assertEqual(
            cash_message,
            "I bought back 9 FIL at $1.1007 on Binance Spot. Bargain Goal cleared at +6.27% "
            "against 6%. I retained $9.45 from $15.75 of realized cash profit. "
            "Result: +$15.75 (+6.27%).",
        )
        self.assertEqual(
            asset_message,
            "I bought back 9 FIL at $1.1007 on Binance Spot. Bargain Goal cleared at +6.27% "
            "against 6%. I secured 0.52 FIL of Swing profit in Target, now 1,928.52 FIL. "
            "Result: +0.52 FIL (+6.27%).",
        )


if __name__ == "__main__":
    unittest.main()
