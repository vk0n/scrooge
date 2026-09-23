import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from shared.runtime_db import (
    append_spot_swing_execution,
    bootstrap_runtime_db,
    create_spot_order_intent,
    create_spot_swing,
    list_portfolio_asset_policies,
    list_portfolio_transactions,
    list_spot_swing_executions,
    list_spot_swings,
    load_spot_order_intent,
    load_spot_swing,
    upsert_portfolio_asset_policy,
)
from shared.spot_swing import calculate_swing_economics, calculate_target_ratchet


class SpotSwingDomainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        self.env = patch.dict(os.environ, {"SCROOGE_DB_PATH": str(self.db_path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        bootstrap_runtime_db(self.db_path)

    def create_swing(
        self,
        swing_id: str,
        *,
        side: str = "sell",
        objective: str | None = "accumulate_cash",
    ) -> dict:
        return create_spot_swing(
            {
                "swing_id": swing_id,
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "origin_side": side,
                "trading_objective": objective,
                "planned_quantity": 50,
                "reference_state": {"rolling_24h_change_pct": 9.2, "level": 2},
                "strategy_reason": {"signal": side, "base_tranche_pct": 20},
                "source": "strategy",
                "opened_at_ms": 1_000,
            },
            path=self.db_path,
        )

    def add_execution(
        self,
        swing_id: str,
        execution_id: str,
        side: str,
        quantity: float,
        price: float,
        *,
        executed_at_ms: int,
        fee_amount: float | None = None,
        fee_asset: str | None = None,
        exchange_execution_key: str | None = None,
    ) -> dict:
        return append_spot_swing_execution(
            {
                "execution_id": execution_id,
                "swing_id": swing_id,
                "venue": "binance",
                "symbol": "NEARUSDT",
                "side": side,
                "quantity": quantity,
                "price": price,
                "quote_quantity": quantity * price,
                "fee_amount": fee_amount,
                "fee_asset": fee_asset,
                "exchange_order_id": execution_id,
                "exchange_execution_key": exchange_execution_key,
                "source": "strategy",
                "executed_at_ms": executed_at_ms,
            },
            path=self.db_path,
        )

    def economics(self, swing_id: str, *, current_price: float | None = None) -> dict:
        swing = load_spot_swing(swing_id, path=self.db_path)
        self.assertIsNotNone(swing)
        return calculate_swing_economics(
            swing,
            list_spot_swing_executions(swing_id, path=self.db_path),
            current_price=current_price,
        )

    def test_swing_with_one_execution_remains_open(self):
        self.create_swing("swing-one")
        self.add_execution("swing-one", "sell-one", "sell", 50, 5, executed_at_ms=2_000)

        result = self.economics("swing-one", current_price=4.5)

        self.assertEqual(result["status"], "open")
        self.assertEqual(result["remaining_quantity"], 50)
        self.assertEqual(result["realized_pnl_quote"], 0)
        self.assertEqual(result["unrealized_pnl_quote"], 25)

    def test_multiple_executions_support_partial_and_full_close(self):
        self.create_swing("swing-partial")
        self.add_execution("swing-partial", "sell-50", "sell", 50, 5, executed_at_ms=2_000)
        self.add_execution("swing-partial", "buy-20", "buy", 20, 4.5, executed_at_ms=3_000)

        partial = self.economics("swing-partial")
        self.assertEqual(partial["status"], "partially_closed")
        self.assertEqual(partial["remaining_quantity"], 30)
        self.assertEqual(load_spot_swing("swing-partial", path=self.db_path)["status"], "partially_closed")

        self.add_execution("swing-partial", "buy-15a", "buy", 15, 4, executed_at_ms=4_000)
        self.add_execution("swing-partial", "buy-15b", "buy", 15, 4.8, executed_at_ms=5_000)
        closed = self.economics("swing-partial")

        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["remaining_quantity"], 0)
        self.assertAlmostEqual(closed["realized_pnl_quote"], 28)
        self.assertEqual(load_spot_swing("swing-partial", path=self.db_path)["closed_at_ms"], 5_000)

    def test_sell_origin_profit_and_quote_fees(self):
        self.create_swing("swing-sell")
        self.add_execution(
            "swing-sell", "sell", "sell", 50, 5, executed_at_ms=2_000, fee_amount=1, fee_asset="USDT"
        )
        self.add_execution(
            "swing-sell", "buy", "buy", 20, 4.5, executed_at_ms=3_000, fee_amount=0.2, fee_asset="USDT"
        )

        result = self.economics("swing-sell")

        self.assertAlmostEqual(result["realized_gross_pnl_quote"], 10)
        self.assertAlmostEqual(result["realized_fee_quote"], 0.6)
        self.assertAlmostEqual(result["realized_pnl_quote"], 9.4)

    def test_buy_origin_profitability(self):
        self.create_swing("swing-buy", side="buy")
        self.add_execution("swing-buy", "buy", "buy", 10, 4, executed_at_ms=2_000)
        self.add_execution("swing-buy", "sell", "sell", 5, 5, executed_at_ms=3_000)

        result = self.economics("swing-buy")

        self.assertEqual(result["remaining_quantity"], 5)
        self.assertEqual(result["realized_pnl_quote"], 5)

    def test_fee_asset_is_preserved_without_fake_quote_conversion(self):
        self.create_swing("swing-fee")
        persisted = self.add_execution(
            "swing-fee", "sell", "sell", 10, 5, executed_at_ms=2_000, fee_amount=0.01, fee_asset="BNB"
        )

        result = self.economics("swing-fee")

        self.assertEqual(persisted["fee_asset"], "BNB")
        self.assertEqual(result["fees_by_asset"], {"BNB": 0.01})
        self.assertEqual(result["unpriced_fees_by_asset"], {"BNB": 0.01})
        self.assertEqual(result["realized_fee_quote"], 0)

    def test_independent_swings_for_same_asset_do_not_blend_economics(self):
        self.create_swing("swing-profit")
        self.create_swing("swing-loss")
        self.add_execution("swing-profit", "sell-profit", "sell", 10, 5, executed_at_ms=2_000)
        self.add_execution("swing-profit", "buy-profit", "buy", 10, 4, executed_at_ms=3_000)
        self.add_execution("swing-loss", "sell-loss", "sell", 10, 3, executed_at_ms=4_000)

        swings = list_spot_swings(asset_symbol="NEAR", path=self.db_path)

        self.assertEqual(len(swings), 2)
        self.assertEqual(self.economics("swing-profit")["realized_pnl_quote"], 10)
        self.assertEqual(self.economics("swing-loss", current_price=4)["unrealized_pnl_quote"], -10)

    def test_portfolio_average_cost_is_not_used_for_swing_pnl(self):
        swing = self.create_swing("swing-own-economics")
        swing["portfolio_average_cost"] = 1.997
        self.add_execution("swing-own-economics", "sell", "sell", 50, 5, executed_at_ms=2_000)
        self.add_execution("swing-own-economics", "buy", "buy", 50, 4.6, executed_at_ms=3_000)

        result = calculate_swing_economics(
            swing,
            list_spot_swing_executions("swing-own-economics", path=self.db_path),
        )

        self.assertAlmostEqual(result["realized_pnl_quote"], 20)

    def test_accumulate_cash_realizes_quote_gain_without_target_ratchet(self):
        swing = self.create_swing("swing-cash", objective="accumulate_cash")
        self.add_execution("swing-cash", "cash-sell", "sell", 100, 5, executed_at_ms=2_000)
        self.add_execution("swing-cash", "cash-buy", "buy", 100, 4.5, executed_at_ms=3_000)

        result = self.economics("swing-cash")
        ratchet = calculate_target_ratchet(
            swing,
            result,
            target_quantity=1_000,
            minimum_holding_pct=80,
        )

        self.assertEqual(result["status"], "closed")
        self.assertAlmostEqual(result["realized_cash_gain_quote"], 50)
        self.assertAlmostEqual(result["realized_net_asset_change"], 0)
        self.assertAlmostEqual(result["realized_asset_gain"], 0)
        self.assertAlmostEqual(result["target_ratchet_quantity"], 0)
        self.assertAlmostEqual(ratchet["next_target_quantity"], 1_000)
        self.assertAlmostEqual(ratchet["next_protected_floor_quantity"], 800)

    def test_accumulate_asset_realizes_net_asset_gain_and_proposes_upward_ratchet(self):
        swing = self.create_swing("swing-asset", objective="accumulate_asset")
        self.add_execution("swing-asset", "asset-sell", "sell", 100, 5, executed_at_ms=2_000)
        self.add_execution(
            "swing-asset",
            "asset-buy",
            "buy",
            105,
            4.5,
            executed_at_ms=3_000,
            fee_amount=0.2,
            fee_asset="NEAR",
        )

        result = self.economics("swing-asset")
        ratchet = calculate_target_ratchet(
            swing,
            result,
            target_quantity=1_000,
            minimum_holding_pct=80,
        )

        self.assertEqual(result["status"], "closed")
        self.assertAlmostEqual(result["realized_cash_gain_quote"], 27.5)
        self.assertAlmostEqual(result["realized_net_asset_change"], 4.8)
        self.assertAlmostEqual(result["realized_asset_gain"], 4.8)
        self.assertAlmostEqual(result["target_ratchet_quantity"], 4.8)
        self.assertAlmostEqual(ratchet["previous_target_quantity"], 1_000)
        self.assertAlmostEqual(ratchet["next_target_quantity"], 1_004.8)
        self.assertAlmostEqual(ratchet["next_protected_floor_quantity"], 803.84)

        self.assertEqual(list_portfolio_asset_policies(path=self.db_path), [])

    def test_accumulate_asset_ratchet_never_reduces_target(self):
        swing = self.create_swing(
            "swing-asset-loss",
            side="buy",
            objective="accumulate_asset",
        )
        self.add_execution("swing-asset-loss", "loss-buy", "buy", 100, 5, executed_at_ms=2_000)
        self.add_execution("swing-asset-loss", "loss-sell", "sell", 101, 4, executed_at_ms=3_000)

        result = self.economics("swing-asset-loss")
        ratchet = calculate_target_ratchet(
            swing,
            result,
            target_quantity=1_000,
            minimum_holding_pct=80,
        )

        self.assertEqual(result["status"], "closed")
        self.assertAlmostEqual(result["realized_cash_gain_quote"], -96)
        self.assertAlmostEqual(result["realized_net_asset_change"], -1)
        self.assertAlmostEqual(result["realized_asset_gain"], 0)
        self.assertAlmostEqual(ratchet["next_target_quantity"], 1_000)
        self.assertAlmostEqual(ratchet["next_protected_floor_quantity"], 800)

    def test_manual_spot_intent_remains_valid_without_swing(self):
        intent = create_spot_order_intent(
            {
                "intent_id": "manual-intent",
                "client_order_id": "scrooge-manual-intent",
                "symbol": "NEARUSDT",
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "side": "buy",
                "requested_quantity": 1,
                "estimated_price": 4,
                "estimated_quote_value": 4,
            },
            path=self.db_path,
        )

        self.assertEqual(intent["source"], "manual")
        self.assertIsNone(intent["swing_id"])
        self.assertEqual(intent["reason"], {})

    def test_legacy_policy_update_preserves_existing_trading_objective(self):
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "target_quantity": 1_000,
                "minimum_holding_pct": 80,
                "trading_objective": "accumulate_asset",
            },
            path=self.db_path,
        )

        updated = upsert_portfolio_asset_policy(
            {
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "target_quantity": 1_100,
                "minimum_holding_pct": 75,
            },
            path=self.db_path,
        )

        self.assertEqual(updated["trading_objective"], "accumulate_asset")

    def test_duplicate_exchange_execution_is_idempotent_and_conflicts_are_rejected(self):
        self.create_swing("swing-dedup")
        first = self.add_execution(
            "swing-dedup",
            "execution-one",
            "sell",
            10,
            5,
            executed_at_ms=2_000,
            exchange_execution_key="binance:NEARUSDT:trade:77",
        )
        replay = self.add_execution(
            "swing-dedup",
            "execution-retry",
            "sell",
            10,
            5,
            executed_at_ms=2_001,
            exchange_execution_key="binance:NEARUSDT:trade:77",
        )

        self.assertFalse(first["idempotent_replay"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["execution_id"], "execution-one")
        self.assertEqual(len(list_spot_swing_executions("swing-dedup", path=self.db_path)), 1)

        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.add_execution(
                "swing-dedup",
                "execution-conflict",
                "sell",
                10,
                6,
                executed_at_ms=2_002,
                exchange_execution_key="binance:NEARUSDT:trade:77",
            )


class SpotSwingMigrationTests(unittest.TestCase):
    def test_existing_treasury_and_manual_intent_survive_additive_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE portfolio_accounts (
                        account_key TEXT PRIMARY KEY, name TEXT NOT NULL, account_type TEXT NOT NULL,
                        base_currency TEXT NOT NULL, archived INTEGER NOT NULL,
                        created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
                    );
                    CREATE TABLE portfolio_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL UNIQUE,
                        account_key TEXT NOT NULL, executed_at_ms INTEGER NOT NULL,
                        executed_at_text TEXT NOT NULL, tx_type TEXT NOT NULL, asset_symbol TEXT NOT NULL,
                        quote_symbol TEXT NOT NULL, quantity REAL NOT NULL, price REAL, fee_amount REAL,
                        fee_asset TEXT, source TEXT NOT NULL, status TEXT NOT NULL, note TEXT,
                        external_order_id TEXT, custody_location TEXT NOT NULL DEFAULT 'unassigned',
                        source_custody TEXT, destination_custody TEXT, payload_json TEXT NOT NULL,
                        created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
                    );
                    CREATE TABLE portfolio_asset_policies (
                        account_key TEXT NOT NULL, asset_symbol TEXT NOT NULL, quote_symbol TEXT NOT NULL,
                        target_quantity REAL NOT NULL, minimum_holding_pct REAL NOT NULL,
                        created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (account_key, asset_symbol, quote_symbol)
                    );
                    CREATE TABLE spot_order_intents (
                        intent_id TEXT PRIMARY KEY, account_key TEXT NOT NULL, venue TEXT NOT NULL,
                        symbol TEXT NOT NULL, asset_symbol TEXT NOT NULL, quote_symbol TEXT NOT NULL,
                        side TEXT NOT NULL, requested_quantity REAL NOT NULL, estimated_price REAL NOT NULL,
                        estimated_quote_value REAL NOT NULL, available_quote_quantity REAL,
                        available_asset_quantity REAL, protected_floor_quantity REAL,
                        policy_sellable_quantity REAL, projected_holding_quantity REAL, status TEXT NOT NULL,
                        command_id TEXT UNIQUE, client_order_id TEXT NOT NULL UNIQUE, exchange_order_id TEXT,
                        executed_quantity REAL, executed_quote_quantity REAL, average_price REAL,
                        fee_amount REAL, fee_asset TEXT, error TEXT, request_json TEXT NOT NULL,
                        result_json TEXT, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
                    );
                    INSERT INTO portfolio_accounts VALUES
                        ('manual_spot', 'Manual Spot', 'spot', 'USDT', 0, 1, 1);
                    INSERT INTO portfolio_transactions VALUES
                        (1, 'opening-near', 'manual_spot', 1000, '2026-01-01 00:00:01', 'buy',
                         'NEAR', 'USDT', 1000, 1.997, 0, 'USDT', 'manual', 'settled', NULL,
                         NULL, 'binance', NULL, NULL, '{}', 1, 1);
                    INSERT INTO portfolio_asset_policies VALUES
                        ('manual_spot', 'NEAR', 'USDT', 1000, 80, 1, 1);
                    INSERT INTO spot_order_intents (
                        intent_id, account_key, venue, symbol, asset_symbol, quote_symbol, side,
                        requested_quantity, estimated_price, estimated_quote_value,
                        available_quote_quantity, available_asset_quantity, protected_floor_quantity,
                        policy_sellable_quantity, projected_holding_quantity, status, command_id,
                        client_order_id, exchange_order_id, executed_quantity, executed_quote_quantity,
                        average_price, fee_amount, fee_asset, error, request_json, result_json,
                        created_at_ms, updated_at_ms
                    ) VALUES (
                        'legacy-intent', 'manual_spot', 'binance', 'NEARUSDT', 'NEAR', 'USDT',
                        'sell', 10, 5, 50, 0, 10, 800, 200, 990, 'previewed', NULL,
                        'legacy-client-id', NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                        '{"intent_id":"legacy-intent"}', NULL, 1, 1
                    );
                    """
                )

            bootstrap_runtime_db(db_path)

            transactions = list_portfolio_transactions(path=db_path)
            policies = list_portfolio_asset_policies(path=db_path)
            intent = load_spot_order_intent("legacy-intent", path=db_path)

            self.assertEqual(len(transactions), 1)
            self.assertEqual(transactions[0]["quantity"], 1000)
            self.assertEqual(transactions[0]["price"], 1.997)
            self.assertEqual(policies[0]["target_quantity"], 1000)
            self.assertEqual(policies[0]["minimum_holding_pct"], 80)
            self.assertIsNone(policies[0]["trading_objective"])
            self.assertEqual(intent["source"], "manual")
            self.assertIsNone(intent["swing_id"])
            self.assertEqual(list_spot_swings(path=db_path), [])


if __name__ == "__main__":
    unittest.main()
