import os
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from services import portfolio_service
from shared.runtime_db import (
    append_spot_swing_execution,
    create_spot_swing,
    mark_exchange_account_snapshot_error,
    save_exchange_account_snapshot,
)


class PortfolioPhaseOneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "SCROOGE_DB_PATH": str(Path(self.tmp.name) / "runtime.sqlite3"),
                "SCROOGE_SPOT_EXECUTION_ENABLED": "1",
            },
        )
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

    def add(self, asset: str, quantity: float, price: float, custody_location: str = "unassigned") -> dict:
        result, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "buy",
                "asset_symbol": asset,
                "quantity": quantity,
                "price": price,
                "quote_symbol": "USDT",
                "custody_location": custody_location,
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

    def test_summary_counts_only_open_spot_swings(self):
        create_spot_swing(
            {
                "swing_id": "open-btc-swing",
                "asset_symbol": "BTC",
                "quote_symbol": "USDT",
                "origin_side": "buy",
                "trading_objective": "accumulate_cash",
                "source": "strategy",
            }
        )
        create_spot_swing(
            {
                "swing_id": "closed-eth-swing",
                "asset_symbol": "ETH",
                "quote_symbol": "USDT",
                "origin_side": "buy",
                "trading_objective": "accumulate_cash",
                "source": "strategy",
            }
        )
        for execution_id, side, price in (
            ("closed-eth-buy", "buy", 10),
            ("closed-eth-sell", "sell", 12),
        ):
            append_spot_swing_execution(
                {
                    "execution_id": execution_id,
                    "swing_id": "closed-eth-swing",
                    "symbol": "ETHUSDT",
                    "side": side,
                    "quantity": 1,
                    "price": price,
                    "source": "strategy",
                }
            )

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(snapshot["summary"]["open_swing_count"], 1)
        self.assertEqual(snapshot["summary"]["open_swing_asset_count"], 1)

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

    def test_asset_ledger_combines_transactions_and_independent_swings(self):
        portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "buy",
                "asset_symbol": "BTC",
                "quantity": 1,
                "price": 90,
                "quote_symbol": "USDT",
                "executed_at": "1970-01-01 00:00:01",
            }
        )
        create_spot_swing(
            {
                "swing_id": "closed-swing",
                "asset_symbol": "BTC",
                "quote_symbol": "USDT",
                "origin_side": "buy",
                "trading_objective": "accumulate_cash",
                "source": "strategy",
                "opened_at_ms": 2_000,
            }
        )
        append_spot_swing_execution(
            {
                "execution_id": "closed-buy",
                "swing_id": "closed-swing",
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 1,
                "price": 80,
                "source": "strategy",
                "executed_at_ms": 2_100,
            }
        )
        append_spot_swing_execution(
            {
                "execution_id": "closed-sell",
                "swing_id": "closed-swing",
                "symbol": "BTCUSDT",
                "side": "sell",
                "quantity": 1,
                "price": 100,
                "source": "strategy",
                "executed_at_ms": 2_200,
            }
        )
        create_spot_swing(
            {
                "swing_id": "open-swing",
                "asset_symbol": "BTC",
                "quote_symbol": "USDT",
                "origin_side": "sell",
                "trading_objective": "accumulate_asset",
                "source": "strategy",
                "opened_at_ms": 3_000,
            }
        )
        append_spot_swing_execution(
            {
                "execution_id": "open-sell",
                "swing_id": "open-swing",
                "symbol": "BTCUSDT",
                "side": "sell",
                "quantity": 0.25,
                "price": 110,
                "fee_amount": 0.01,
                "fee_asset": "BNB",
                "source": "strategy",
                "executed_at_ms": 3_100,
            }
        )

        ledger = portfolio_service.load_portfolio_asset_ledger("BTC")

        self.assertEqual(ledger["entry_count"], 3)
        self.assertEqual(
            [entry["entry_type"] for entry in ledger["entries"]],
            ["swing", "swing", "transaction"],
        )
        open_swing = ledger["entries"][0]["swing"]
        self.assertEqual(open_swing["swing_id"], "open-swing")
        self.assertEqual(open_swing["economics"]["status"], "open")
        self.assertEqual(open_swing["economics"]["remaining_quantity"], 0.25)
        self.assertEqual(open_swing["economics"]["unrealized_pnl_quote"], 2.5)
        self.assertEqual(open_swing["economics"]["unpriced_fees_by_asset"], {"BNB": 0.01})
        closed_swing = ledger["entries"][1]["swing"]
        self.assertEqual(closed_swing["economics"]["realized_pnl_quote"], 20)
        self.assertEqual(len(closed_swing["executions"]), 2)

        open_ledger = portfolio_service.load_portfolio_asset_ledger("BTC", entry_filter="open")
        closed_ledger = portfolio_service.load_portfolio_asset_ledger("BTC", entry_filter="closed")
        self.assertEqual([entry["swing"]["swing_id"] for entry in open_ledger["entries"]], ["open-swing"])
        self.assertEqual([entry["swing"]["swing_id"] for entry in closed_ledger["entries"]], ["closed-swing"])

    def test_asset_ledger_is_filtered_and_paginated(self):
        btc_ids = [self.add("BTC", 1, 90)["transaction_id"] for _ in range(7)]
        self.add("ETH", 1, 10)

        first_page = portfolio_service.load_portfolio_asset_transactions("btc")
        second_page = portfolio_service.load_portfolio_asset_transactions("BTC", transaction_offset=5)

        self.assertEqual(first_page["asset_symbol"], "BTC")
        self.assertEqual(first_page["transaction_count"], 7)
        self.assertEqual(first_page["transaction_limit"], 5)
        self.assertEqual(
            [transaction["transaction_id"] for transaction in first_page["transactions"]],
            list(reversed(btc_ids))[:5],
        )
        self.assertEqual(
            [transaction["transaction_id"] for transaction in second_page["transactions"]],
            list(reversed(btc_ids))[5:],
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

    def test_non_stable_deposit_requires_entry_cost(self):
        with self.assertRaisesRegex(ValueError, "Entry cost is required"):
            portfolio_service.create_portfolio_transaction(
                {
                    "tx_type": "deposit",
                    "asset_symbol": "BTC",
                    "quantity": 0.5,
                    "quote_symbol": "USDT",
                    "custody_location": "binance",
                }
            )

    def test_deposit_and_withdraw_cross_treasury_boundary_without_mutating_target(self):
        self.add("BTC", 1, 90, "binance")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1, "minimum_holding_pct": 75},
        )

        deposited, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "deposit",
                "asset_symbol": "BTC",
                "quantity": 0.5,
                "price": 80,
                "quote_symbol": "USDT",
                "custody_location": "cold_storage",
            }
        )
        deposited_holding = deposited["portfolio"]["holdings"][0]
        self.assertEqual(deposited_holding["quantity"], 1.5)
        self.assertEqual(deposited_holding["invested_capital"], 130)
        self.assertEqual(deposited_holding["cold_storage_quantity"], 0.5)
        self.assertEqual(deposited_holding["target_quantity"], 1)
        self.assertEqual(deposited_holding["minimum_holding_pct"], 75)

        withdrawn, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "withdraw",
                "asset_symbol": "BTC",
                "quantity": 0.25,
                "quote_symbol": "USDT",
                "custody_location": "cold_storage",
            }
        )
        withdrawn_holding = withdrawn["portfolio"]["holdings"][0]
        self.assertEqual(withdrawn_holding["quantity"], 1.25)
        self.assertEqual(withdrawn_holding["invested_capital"], 110)
        self.assertEqual(withdrawn_holding["cold_storage_quantity"], 0.25)
        self.assertEqual(withdrawn_holding["target_quantity"], 1)
        self.assertEqual(withdrawn_holding["minimum_holding_pct"], 75)

    def test_stable_deposit_without_price_uses_par_cost_basis(self):
        result, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "deposit",
                "asset_symbol": "USDT",
                "quantity": 50,
                "quote_symbol": "USDT",
                "custody_location": "binance",
            }
        )

        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(holding["quantity"], 50)
        self.assertEqual(holding["invested_capital"], 50)
        self.assertEqual(holding["binance_quantity"], 50)

    def test_policy_stays_dormant_after_full_release_and_returns_with_asset(self):
        self.add("BTC", 1, 90, "binance")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1, "minimum_holding_pct": 60},
        )
        released, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "withdraw",
                "asset_symbol": "BTC",
                "quantity": 1,
                "quote_symbol": "USDT",
                "custody_location": "binance",
            }
        )
        self.assertEqual(released["portfolio"]["holdings"], [])

        returned, _ = portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "deposit",
                "asset_symbol": "BTC",
                "quantity": 0.5,
                "price": 80,
                "quote_symbol": "USDT",
                "custody_location": "cold_storage",
            }
        )
        holding = returned["portfolio"]["holdings"][0]
        self.assertEqual(holding["target_quantity"], 1)
        self.assertEqual(holding["minimum_holding_pct"], 60)
        self.assertEqual(holding["cold_storage_quantity"], 0.5)

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
        self.assertEqual(snapshot["holdings"][0]["target_quantity"], 1)
        self.assertEqual(snapshot["holdings"][0]["minimum_holding_pct"], 100)

    def test_existing_holding_initializes_safe_policy_once(self):
        self.add("BTC", 1, 90)
        initial, _ = portfolio_service.load_portfolio_snapshot()

        self.add("BTC", 0.5, 80)
        portfolio_service.create_portfolio_transaction(
            {
                "tx_type": "sell",
                "asset_symbol": "BTC",
                "quantity": 0.25,
                "price": 100,
                "quote_symbol": "USDT",
            }
        )
        updated, _ = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(initial["holdings"][0]["target_quantity"], 1)
        self.assertEqual(initial["holdings"][0]["minimum_holding_pct"], 100)
        self.assertEqual(initial["holdings"][0]["trading_objective"], "accumulate_cash")
        self.assertEqual(updated["holdings"][0]["quantity"], 1.25)
        self.assertEqual(updated["holdings"][0]["target_quantity"], 1)

    def test_spot_mode_locks_assets_without_mutating_stored_policy(self):
        self.add("BTC", 1, 90, "binance")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1, "minimum_holding_pct": 75},
        )
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [{"asset_symbol": "BTC", "free": 1, "locked": 0}],
            }
        )

        with patch.dict(os.environ, {"SCROOGE_SPOT_EXECUTION_ENABLED": "0"}):
            read_only, _ = portfolio_service.load_portfolio_snapshot()

        locked = read_only["holdings"][0]
        self.assertEqual(locked["spot_trading_state"], "locked")
        self.assertEqual(locked["spot_trading_state_reason"], "execution_disabled")
        self.assertEqual(locked["immediately_sellable_quantity"], 0)
        self.assertEqual(locked["minimum_holding_pct"], 75)

        enabled, _ = portfolio_service.load_portfolio_snapshot()
        unlocked = enabled["holdings"][0]
        self.assertEqual(unlocked["spot_trading_state"], "unlocked")
        self.assertEqual(unlocked["spot_trading_state_reason"], "policy_allows_trading")
        self.assertEqual(unlocked["minimum_holding_pct"], 75)

    def test_spot_order_preview_is_unavailable_in_read_only_mode(self):
        with patch.dict(os.environ, {"SCROOGE_SPOT_EXECUTION_ENABLED": "0"}):
            with self.assertRaisesRegex(ValueError, "safety switch"):
                portfolio_service.create_spot_order_preview(
                    {"asset_symbol": "BTC", "side": "buy", "quantity": 0.25}
                )

    def test_enabled_spot_mode_keeps_fully_protected_asset_locked(self):
        self.add("BTC", 1, 90)

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        holding = snapshot["holdings"][0]
        self.assertEqual(holding["minimum_holding_pct"], 100)
        self.assertEqual(holding["spot_trading_state"], "locked")
        self.assertEqual(holding["spot_trading_state_reason"], "fully_protected")

    def test_asset_policy_can_be_updated_without_changing_holding(self):
        self.add("BTC", 1, 90)

        result, _ = portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {
                "quote_symbol": "USDT",
                "target_quantity": 0.8,
                "minimum_holding_pct": 75,
            },
        )

        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(result["policy"]["target_quantity"], 0.8)
        self.assertEqual(result["policy"]["minimum_holding_pct"], 75)
        self.assertEqual(holding["quantity"], 1)
        self.assertEqual(holding["target_quantity"], 0.8)
        self.assertEqual(holding["minimum_holding_pct"], 75)

    def test_asset_policy_validates_target_and_minimum_holding(self):
        self.add("BTC", 1, 90)

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            portfolio_service.update_portfolio_asset_policy(
                "BTC",
                {"target_quantity": 0, "minimum_holding_pct": 100},
            )
        with self.assertRaisesRegex(ValueError, "between 0% and 100%"):
            portfolio_service.update_portfolio_asset_policy(
                "BTC",
                {"target_quantity": 1, "minimum_holding_pct": 101},
            )

    def test_unknown_asset_policy_cannot_be_created(self):
        with self.assertRaisesRegex(LookupError, "not found"):
            portfolio_service.update_portfolio_asset_policy(
                "BTC",
                {"target_quantity": 1, "minimum_holding_pct": 100},
            )

    def test_sellable_inventory_respects_policy_and_binance_balance(self):
        self.add("BTC", 300, 90, "binance")
        self.add("BTC", 800, 90, "cold_storage")

        result, _ = portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(holding["quantity"], 1100)
        self.assertEqual(holding["protected_floor_quantity"], 800)
        self.assertEqual(holding["amount_above_protected_floor"], 300)
        self.assertEqual(holding["policy_sellable_quantity"], 300)
        self.assertEqual(holding["immediately_sellable_quantity"], 0)
        self.assertFalse(holding["sellable_inventory_is_exchange_verified"])
        self.assertEqual(holding["target_delta_quantity"], 100)

    def test_sellable_inventory_is_capped_by_binance_custody(self):
        self.add("BTC", 100, 90, "binance")
        self.add("BTC", 1000, 90, "cold_storage")

        result, _ = portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(holding["amount_above_protected_floor"], 300)
        self.assertEqual(holding["policy_sellable_quantity"], 100)
        self.assertEqual(holding["immediately_sellable_quantity"], 0)

    def test_holding_below_protected_floor_has_no_sellable_inventory(self):
        self.add("BTC", 700, 90, "binance")

        result, _ = portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        holding = result["portfolio"]["holdings"][0]
        self.assertEqual(holding["protected_floor_quantity"], 800)
        self.assertEqual(holding["amount_below_protected_floor"], 100)
        self.assertEqual(holding["amount_above_protected_floor"], 0)
        self.assertEqual(holding["immediately_sellable_quantity"], 0)

    def test_custody_move_changes_liquidity_but_not_policy_surplus(self):
        self.add("BTC", 1000, 90)
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )
        before, _ = portfolio_service.load_portfolio_snapshot()

        moved, _ = portfolio_service.create_custody_transfer(
            {
                "asset_symbol": "BTC",
                "quantity": 150,
                "source_custody": "unassigned",
                "destination_custody": "binance",
            }
        )

        before_holding = before["holdings"][0]
        holding = moved["portfolio"]["holdings"][0]
        self.assertEqual(before_holding["amount_above_protected_floor"], 200)
        self.assertEqual(before_holding["immediately_sellable_quantity"], 0)
        self.assertEqual(holding["amount_above_protected_floor"], 200)
        self.assertEqual(holding["policy_sellable_quantity"], 150)
        self.assertEqual(holding["immediately_sellable_quantity"], 0)

    def test_fresh_exchange_snapshot_caps_immediate_inventory_by_free_balance(self):
        self.add("BTC", 300, 90, "binance")
        self.add("BTC", 800, 90, "cold_storage")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [
                    {"asset_symbol": "BTC", "free": 220, "locked": 80},
                    {"asset_symbol": "USDT", "free": 125, "locked": 5},
                ],
            }
        )

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        holding = snapshot["holdings"][0]
        self.assertEqual(holding["policy_sellable_quantity"], 300)
        self.assertEqual(holding["exchange_binance_free_quantity"], 220)
        self.assertEqual(holding["exchange_binance_locked_quantity"], 80)
        self.assertEqual(holding["immediately_sellable_quantity"], 220)
        self.assertEqual(holding["binance_custody_variance"], 0)
        self.assertTrue(holding["sellable_inventory_is_exchange_verified"])
        self.assertEqual(snapshot["exchange"]["usdt_free"], 125)
        self.assertEqual(snapshot["summary"]["binance_spot_usdt_free"], 125)

    def test_exchange_balance_is_visibility_only_and_not_double_counted(self):
        self.add("BTC", 1, 90, "binance")
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [
                    {"asset_symbol": "BTC", "free": 1, "locked": 0},
                    {"asset_symbol": "USDT", "free": 500, "locked": 0},
                ],
            }
        )

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        self.assertEqual(snapshot["summary"]["total_value"], 100)
        self.assertEqual(snapshot["summary"]["dry_powder"], 0)
        self.assertEqual(snapshot["exchange"]["usdt_free"], 500)

    def test_stale_or_failed_exchange_snapshot_is_not_immediately_sellable(self):
        self.add("BTC", 2, 90, "binance")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1, "minimum_holding_pct": 0},
        )
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int((time.time() - portfolio_service.SPOT_BALANCE_STALE_AFTER_SECONDS - 1) * 1000),
                "can_trade": True,
                "balances": [{"asset_symbol": "BTC", "free": 2, "locked": 0}],
            }
        )

        stale, _ = portfolio_service.load_portfolio_snapshot()
        stale_holding = stale["holdings"][0]
        self.assertTrue(stale["exchange"]["is_stale"])
        self.assertEqual(stale_holding["immediately_sellable_quantity"], 0)
        self.assertFalse(stale_holding["sellable_inventory_is_exchange_verified"])

        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [{"asset_symbol": "BTC", "free": 2, "locked": 0}],
            }
        )
        mark_exchange_account_snapshot_error("temporary failure")
        failed, _ = portfolio_service.load_portfolio_snapshot()
        failed_holding = failed["holdings"][0]
        self.assertEqual(failed["exchange"]["status"], "error")
        self.assertEqual(failed_holding["exchange_binance_free_quantity"], 2)
        self.assertEqual(failed_holding["immediately_sellable_quantity"], 0)
        self.assertFalse(failed_holding["sellable_inventory_is_exchange_verified"])

    def test_spot_sell_preview_enforces_policy_and_persists_intent(self):
        self.add("BTC", 1, 90, "binance")
        portfolio_service.update_portfolio_asset_policy(
            "BTC",
            {"target_quantity": 1, "minimum_holding_pct": 50},
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

        preview = portfolio_service.create_spot_order_preview(
            {"asset_symbol": "BTC", "side": "sell", "quantity": 0.25}
        )

        self.assertEqual(preview["status"], "previewed")
        self.assertEqual(preview["side"], "sell")
        self.assertEqual(preview["estimated_quote_value"], 25)
        self.assertEqual(preview["protected_floor_quantity"], 0.5)
        self.assertEqual(preview["projected_holding_quantity"], 0.75)
        self.assertEqual(len(preview["client_order_id"]), 35)

        with self.assertRaisesRegex(ValueError, "immediately sellable"):
            portfolio_service.create_spot_order_preview(
                {"asset_symbol": "BTC", "side": "sell", "quantity": 0.6}
            )

    def test_spot_buy_preview_requires_available_usdt(self):
        self.add("BTC", 1, 90, "binance")
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [
                    {"asset_symbol": "BTC", "free": 1, "locked": 0},
                    {"asset_symbol": "USDT", "free": 20, "locked": 0},
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "only \\$20.00 USDT"):
            portfolio_service.create_spot_order_preview(
                {"asset_symbol": "BTC", "side": "buy", "quantity": 0.25}
            )

    def test_treasury_intake_previews_buy_without_creating_empty_holding(self):
        save_exchange_account_snapshot(
            {
                "captured_at_ms": int(time.time() * 1000),
                "can_trade": True,
                "balances": [{"asset_symbol": "USDT", "free": 100, "locked": 0}],
            }
        )

        with self.assertRaisesRegex(LookupError, "Treasury asset was not found"):
            portfolio_service.create_spot_order_preview(
                {"asset_symbol": "ETH", "side": "buy", "quantity": 2}
            )

        preview = portfolio_service.create_spot_order_preview(
            {
                "asset_symbol": "ETH",
                "side": "buy",
                "quantity": 2,
                "treasury_intake": True,
            }
        )

        self.assertEqual(preview["estimated_quote_value"], 40)
        self.assertEqual(preview["projected_holding_quantity"], 2)
        self.assertTrue(preview["request"]["treasury_intake"])
        self.assertEqual(preview["request"]["initial_target_quantity"], 2)
        snapshot, _ = portfolio_service.load_portfolio_snapshot()
        self.assertEqual(snapshot["holdings"], [])

    def test_dry_powder_does_not_receive_asset_policy(self):
        self.add("USDT", 500, 1, "binance")

        snapshot, _ = portfolio_service.load_portfolio_snapshot()

        holding = snapshot["holdings"][0]
        self.assertIsNone(holding["target_quantity"])
        self.assertEqual(holding["immediately_sellable_quantity"], 0)
        with self.assertRaisesRegex(ValueError, "Dry Powder"):
            portfolio_service.update_portfolio_asset_policy(
                "USDT",
                {"target_quantity": 500, "minimum_holding_pct": 100},
            )


if __name__ == "__main__":
    unittest.main()
