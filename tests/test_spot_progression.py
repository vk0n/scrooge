import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.spot_strategy import ProgressiveSpotSwingExecutor
from shared.runtime_db import (
    append_spot_swing_execution,
    apply_spot_swing_target_ratchet,
    bootstrap_runtime_db,
    complete_spot_strategy_campaign_level,
    create_spot_swing,
    list_portfolio_asset_policies,
    list_spot_strategy_actions,
    list_spot_swings,
    upsert_portfolio_asset_policy,
    update_spot_strategy_action,
)
from shared.spot_progression import ProgressiveSwingConfig, plan_opening_quantity, plan_profitable_close
from shared.spot_strategy import plan_spot_strategy_action
from shared.spot_swing import calculate_swing_economics


class ProgressiveSwingDomainTests(unittest.TestCase):
    def test_sell_tranche_uses_policy_trading_capacity(self):
        plan = plan_opening_quantity(
            {
                "opportunity": "sell",
                "trading_objective": "accumulate_cash",
                "final_tranche_pct": 25,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["strategic_capacity"], 200)
        self.assertEqual(plan["quantity"], 50)

    def test_zero_minimum_holding_exposes_full_target_capacity(self):
        plan = plan_opening_quantity(
            {
                "opportunity": "sell",
                "trading_objective": "accumulate_cash",
                "final_tranche_pct": 25,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 0},
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["strategic_capacity"], 1000)
        self.assertEqual(plan["quantity"], 250)

    def test_accumulate_asset_does_not_open_buy_origin_in_v1(self):
        plan = plan_opening_quantity(
            {
                "opportunity": "buy",
                "trading_objective": "accumulate_asset",
                "final_tranche_pct": 20,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["reason"], "accumulate_asset_sell_origin_only_v1")

    def test_accumulate_cash_buy_origin_is_capped_by_available_vault_reserve(self):
        decision = plan_spot_strategy_action(
            {
                "opportunity": "buy",
                "level": 1,
                "strategy_eligible": True,
                "trading_objective": "accumulate_cash",
                "final_tranche_pct": 25,
                "current_price": 5,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80, "market_price": 5},
            {"active_side": "buy", "highest_completed_level": 0, "campaign_id": "buy-campaign"},
            [],
            available_quote=1000,
            available_opening_quote=40,
        )

        self.assertEqual(decision["side"], "buy")
        self.assertEqual(decision["requested_quantity"], 2)

    def test_close_uses_swing_basis_and_can_reacquire_more_asset(self):
        swing = {
            "origin_side": "sell",
            "trading_objective": "accumulate_asset",
            "asset_symbol": "NEAR",
            "quote_symbol": "USDT",
            "status": "open",
        }
        executions = [{"side": "sell", "quantity": 100, "price": 5, "fee_amount": 0, "fee_asset": "USDT"}]
        economics = calculate_swing_economics(swing, executions, current_price=4.5)

        plan = plan_profitable_close(
            swing,
            economics,
            current_price=4.5,
            available_quote_quantity=1000,
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["side"], "buy")
        self.assertAlmostEqual(plan["quantity"], 500 / 4.5)

    def test_partial_accumulate_asset_close_reuses_only_unspent_quote(self):
        swing = {
            "origin_side": "sell",
            "trading_objective": "accumulate_asset",
            "asset_symbol": "NEAR",
            "quote_symbol": "USDT",
            "status": "partially_closed",
        }
        executions = [
            {"side": "sell", "quantity": 100, "price": 5},
            {"side": "buy", "quantity": 50, "price": 4.5},
        ]
        economics = calculate_swing_economics(swing, executions, current_price=4.5)

        plan = plan_profitable_close(
            swing,
            economics,
            current_price=4.5,
            available_quote_quantity=1000,
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
        )

        self.assertAlmostEqual(plan["quantity"], (500 - 225) / 4.5)


class RecordingProgressiveExecutor(ProgressiveSpotSwingExecutor):
    def _execute_action(self, action):
        updated = update_spot_strategy_action(
            action["action_key"],
            {"status": "completed", "completed_at_ms": action["updated_at_ms"]},
            path=self.db_path,
        ) or action
        if action["action_type"] == "open":
            complete_spot_strategy_campaign_level(
                account_key=action["account_key"],
                asset_symbol=action["asset_symbol"],
                quote_symbol=action["quote_symbol"],
                campaign_id=action["campaign_id"],
                signal_level=action["signal_level"],
                path=self.db_path,
            )
        return updated


class ProgressiveSwingPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        bootstrap_runtime_db(self.db_path)
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "target_quantity": 1000,
                "minimum_holding_pct": 80,
                "trading_objective": "accumulate_cash",
            },
            path=self.db_path,
        )

    def signal(self, level: int, tranche: float) -> dict:
        return {
            "account_key": "manual_spot",
            "asset_symbol": "NEAR",
            "quote_symbol": "USDT",
            "opportunity": "sell",
            "level": level,
            "strategy_eligible": True,
            "trading_objective": "accumulate_cash",
            "current_price": 5,
            "reference_price": 4.5,
            "rolling_change_pct": 11.1,
            "base_tranche_pct": tranche,
            "sizing_modifier": 1,
            "final_tranche_pct": tranche,
            "evaluated_at_ms": 1_800_000_000_000,
        }

    def portfolio(self):
        return (
            {
                "holdings": [
                    {
                        "asset_symbol": "NEAR",
                        "quote_symbol": "USDT",
                        "target_quantity": 1000,
                        "minimum_holding_pct": 80,
                        "immediately_sellable_quantity": 200,
                        "market_price": 5,
                    }
                ],
                "exchange": {"usdt_free": 1000},
            },
            [],
        )

    def test_each_new_level_creates_one_independent_swing(self):
        executor = RecordingProgressiveExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        with patch("bot.spot_strategy.load_portfolio_snapshot", side_effect=self.portfolio):
            executor.handle_signal(self.signal(1, 10))
            executor.handle_signal(self.signal(1, 10))
            executor.handle_signal(self.signal(2, 20))

        swings = list_spot_swings(path=self.db_path)
        actions = list_spot_strategy_actions(path=self.db_path)
        self.assertEqual(len(swings), 2)
        self.assertEqual({action["signal_level"] for action in actions}, {1, 2})
        self.assertEqual({swing["swing_id"] for swing in swings}, {action["swing_id"] for action in actions})

    def test_target_ratchet_is_applied_exactly_once(self):
        swing_id = "swing-ratchet"
        create_spot_swing(
            {
                "swing_id": swing_id,
                "account_key": "manual_spot",
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "origin_side": "sell",
                "trading_objective": "accumulate_asset",
                "source": "strategy",
            },
            path=self.db_path,
        )
        for execution_id, side, quantity, price in (
            ("open", "sell", 100, 5),
            ("close", "buy", 105, 4.7),
        ):
            append_spot_swing_execution(
                {
                    "execution_id": execution_id,
                    "swing_id": swing_id,
                    "symbol": "NEARUSDT",
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "source": "strategy",
                    "executed_at_ms": 1 if side == "sell" else 2,
                },
                path=self.db_path,
            )

        first = apply_spot_swing_target_ratchet(swing_id, path=self.db_path)
        second = apply_spot_swing_target_ratchet(swing_id, path=self.db_path)
        policy = list_portfolio_asset_policies(path=self.db_path)[0]

        self.assertEqual(first["applied_gain_quantity"], 5)
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(policy["target_quantity"], 1005)

    def test_profitable_swing_can_close_while_another_remains_under_target(self):
        executor = ProgressiveSpotSwingExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        for swing_id, price in (("swing-near-5", 5), ("swing-near-6", 6)):
            create_spot_swing(
                {
                    "swing_id": swing_id,
                    "account_key": "manual_spot",
                    "asset_symbol": "NEAR",
                    "quote_symbol": "USDT",
                    "origin_side": "sell",
                    "trading_objective": "accumulate_cash",
                    "source": "strategy",
                },
                path=self.db_path,
            )
            append_spot_swing_execution(
                {
                    "execution_id": f"open-{swing_id}",
                    "swing_id": swing_id,
                    "symbol": "NEARUSDT",
                    "side": "sell",
                    "quantity": 10,
                    "price": price,
                    "source": "strategy",
                },
                path=self.db_path,
            )

        action = executor._plan_close(
            asset="NEAR",
            quote="USDT",
            current_price=4.8,
            available_quote=1000,
        )

        self.assertEqual(action["swing_id"], "swing-near-6")
        self.assertEqual(action["side"], "buy")


if __name__ == "__main__":
    unittest.main()
