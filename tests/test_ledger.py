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
from shared.treasury_ledger import project_portfolio_transaction, project_portfolio_transactions


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
        self.assertEqual(entries[0]["message"], "Binance Spot bought 12.5 NEAR at $4.2.")


if __name__ == "__main__":
    unittest.main()
