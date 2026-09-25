import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from bot.spot_strategy import ProgressiveSpotSwingExecutor
from shared.runtime_db import (
    append_spot_swing_execution,
    apply_spot_accumulation_target_ratchet,
    apply_spot_swing_target_ratchet,
    bootstrap_runtime_db,
    complete_spot_strategy_campaign_level,
    create_spot_swing,
    ensure_spot_strategy_action,
    initialize_spot_strategy_campaign_capacity,
    list_portfolio_asset_policies,
    list_spot_accumulation_target_ratchets,
    list_spot_strategy_actions,
    list_spot_swings,
    load_spot_strategy_campaign,
    sync_spot_strategy_campaign,
    upsert_portfolio_asset_policy,
    update_spot_strategy_action,
)
from shared.spot_progression import (
    ProgressiveSwingConfig,
    initialize_sell_campaign_capacity,
    plan_opening_quantity,
    plan_profitable_close,
    plan_treasury_accumulation,
    policy_sellable_reference,
)
from shared.spot_strategy import plan_spot_strategy_action, transition_spot_strategy_campaign
from shared.spot_swing import calculate_swing_economics
from shared.spot_waiter_cleanup import WaiterCleanupConfig


class ProgressiveSwingDomainTests(unittest.TestCase):
    def test_policy_sellable_reference_uses_only_current_target_and_minimum(self):
        self.assertEqual(policy_sellable_reference(1000, 60), 400)
        self.assertEqual(policy_sellable_reference(2000, 75), 500)
        self.assertEqual(policy_sellable_reference(1100, 60), 440)

    def test_campaign_capacity_boundary_uses_full_deploy_at_exactly_25_pct(self):
        expected = ((400, 200, "normal"), (300, 150, "normal"), (101, 50.5, "normal"),
                    (100, 100, "full_deploy"), (80, 80, "full_deploy"))
        for remaining, capacity, mode in expected:
            with self.subTest(remaining=remaining):
                campaign = initialize_sell_campaign_capacity(
                    {"active_side": "sell", "campaign_id": f"sell-{remaining}"},
                    {
                        "target_quantity": 1000,
                        "minimum_holding_pct": 60,
                        "policy_sellable_quantity": remaining,
                    },
                )
                self.assertEqual(campaign["campaign_capacity_quantity"], capacity)
                self.assertEqual(campaign["campaign_capacity_mode"], mode)

    def test_fixed_level_allocations_ignore_conviction(self):
        campaign = initialize_sell_campaign_capacity(
            {"active_side": "sell", "campaign_id": "sell-fixed"},
            {
                "target_quantity": 1000,
                "minimum_holding_pct": 60,
                "policy_sellable_quantity": 400,
            },
        )
        expected = {1: 20, 2: 40, 3: 60, 4: 80}
        for level, quantity in expected.items():
            for tier, modifier in (("weak", 0.5), ("very_strong", 1.5)):
                with self.subTest(level=level, tier=tier):
                    plan = plan_opening_quantity(
                        {
                            "opportunity": "sell",
                            "base_tranche_pct": level * 10,
                            "sizing_modifier": modifier,
                            "indicator_assessment": {"tier": tier},
                        },
                        {"immediately_sellable_quantity": 400},
                        campaign,
                    )
                    self.assertEqual(plan["quantity"], quantity)

    def test_active_campaign_snapshot_does_not_resize_after_target_change(self):
        started = initialize_sell_campaign_capacity(
            {"active_side": "sell", "campaign_id": "sell-frozen"},
            {
                "target_quantity": 1000,
                "minimum_holding_pct": 60,
                "policy_sellable_quantity": 300,
            },
        )
        resumed = initialize_sell_campaign_capacity(
            started,
            {
                "target_quantity": 1100,
                "minimum_holding_pct": 60,
                "policy_sellable_quantity": 260,
            },
        )
        self.assertEqual(resumed["campaign_start_policy_sellable_reference"], 400)
        self.assertEqual(resumed["campaign_capacity_quantity"], 150)

    def test_current_policy_and_exchange_inventory_cap_frozen_budget(self):
        campaign = {
            "active_side": "sell",
            "campaign_id": "sell-capped",
            "campaign_capacity_quantity": 200,
            "campaign_consumed_quantity": 140,
        }
        plan = plan_opening_quantity(
            {"opportunity": "sell", "base_tranche_pct": 40},
            {"immediately_sellable_quantity": 25},
            campaign,
        )
        self.assertEqual(plan["quantity"], 25)

    def test_sell_tranche_uses_frozen_campaign_capacity(self):
        plan = plan_opening_quantity(
            {
                "opportunity": "sell",
                "trading_objective": "accumulate_cash",
                "final_tranche_pct": 25,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["campaign_capacity_quantity"], 100)
        self.assertEqual(plan["quantity"], 25)

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
        self.assertEqual(plan["campaign_capacity_quantity"], 500)
        self.assertEqual(plan["quantity"], 125)

    def test_buy_signal_never_opens_a_bargain(self):
        plan = plan_opening_quantity(
            {
                "opportunity": "buy",
                "trading_objective": "accumulate_asset",
                "final_tranche_pct": 20,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80},
        )

        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["reason"], "no_opening_opportunity")

    def test_accumulate_cash_buy_signal_does_not_open_bargain(self):
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
            available_accumulation_quote=40,
        )

        self.assertEqual(decision["action_type"], "campaign_only")
        self.assertEqual(decision["requested_quantity"], 0)
        self.assertNotIn("swing_id", decision)

    def test_accumulate_asset_buy_uses_a_fraction_of_free_reserve(self):
        plan = plan_treasury_accumulation(
            {
                "opportunity": "buy",
                "final_tranche_pct": 25,
            },
            free_reserve_quote=40,
            current_price=5,
            config=ProgressiveSwingConfig(
                treasury_accumulation_enabled=True,
                estimated_fee_rate=0,
            ),
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["quote_to_spend"], 10)
        self.assertEqual(plan["quantity"], 2)

    def test_accumulate_asset_buy_is_a_standalone_treasury_action(self):
        decision = plan_spot_strategy_action(
            {
                "opportunity": "buy",
                "level": 2,
                "strategy_eligible": True,
                "trading_objective": "accumulate_asset",
                "final_tranche_pct": 25,
                "current_price": 5,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80, "market_price": 5},
            {"active_side": "buy", "highest_completed_level": 1, "campaign_id": "buy-campaign"},
            [],
            available_quote=1000,
            available_accumulation_quote=40,
            config=ProgressiveSwingConfig(
                treasury_accumulation_enabled=True,
                estimated_fee_rate=0,
            ),
        )

        self.assertEqual(decision["action_type"], "accumulate_asset")
        self.assertEqual(decision["requested_quantity"], 2)
        self.assertNotIn("swing_id", decision)

    def test_accumulation_fails_closed_without_an_explicit_free_reserve_projection(self):
        decision = plan_spot_strategy_action(
            {
                "opportunity": "buy",
                "level": 1,
                "strategy_eligible": True,
                "trading_objective": "accumulate_asset",
                "final_tranche_pct": 25,
                "current_price": 5,
            },
            {"target_quantity": 1000, "minimum_holding_pct": 80, "market_price": 5},
            {"active_side": "buy", "highest_completed_level": 0, "campaign_id": "buy-campaign"},
            [],
            available_quote=1000,
            config=ProgressiveSwingConfig(treasury_accumulation_enabled=True),
        )

        self.assertIsNone(decision)

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
        self.assertEqual(plan["quantity_basis"], "reusable_quote")

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


class SpotStrategyCampaignTests(unittest.TestCase):
    def transition(
        self,
        previous: dict | None,
        side: str,
        level: int,
        tick: int,
    ) -> dict:
        return transition_spot_strategy_campaign(
            previous,
            opportunity=side,
            signal_level=level,
            signal_at_ms=tick,
            new_campaign_id=f"campaign-{tick}",
        )

    @staticmethod
    def complete(campaign: dict, level: int) -> dict:
        return {
            **campaign,
            "highest_completed_level": max(
                int(campaign.get("highest_completed_level") or 0),
                level,
            ),
        }

    @staticmethod
    def opening(campaign: dict, side: str, level: int) -> dict | None:
        return plan_spot_strategy_action(
            {
                "opportunity": side,
                "level": level,
                "strategy_eligible": True,
                "trading_objective": "accumulate_cash",
                "final_tranche_pct": level * 10,
                "current_price": 10,
            },
            {
                "target_quantity": 100,
                "minimum_holding_pct": 0,
                "immediately_sellable_quantity": 100,
                "market_price": 10,
            },
            campaign,
            [],
            available_quote=10_000,
            cleanup_config=WaiterCleanupConfig(enabled=False),
        )

    def test_sell_l1_hold_sell_l1_does_not_reopen_l1(self):
        campaign = self.complete(self.transition(None, "sell", 1, 1), 1)
        campaign_id = campaign["campaign_id"]
        campaign = self.transition(campaign, "hold", 0, 2)
        campaign = self.transition(campaign, "sell", 1, 3)

        self.assertEqual(campaign["campaign_id"], campaign_id)
        self.assertEqual(campaign["highest_completed_level"], 1)
        self.assertIsNone(self.opening(campaign, "sell", 1))

    def test_sell_l1_hold_sell_l2_allows_l2(self):
        campaign = self.complete(self.transition(None, "sell", 1, 1), 1)
        campaign = self.transition(campaign, "hold", 0, 2)
        campaign = self.transition(campaign, "sell", 2, 3)

        decision = self.opening(campaign, "sell", 2)

        self.assertEqual(decision["action_type"], "open")
        self.assertEqual(decision["signal_level"], 2)

    def test_sell_l2_hold_sell_l1_does_not_open(self):
        campaign = self.complete(self.transition(None, "sell", 2, 1), 2)
        campaign = self.transition(campaign, "hold", 0, 2)
        campaign = self.transition(campaign, "sell", 1, 3)

        self.assertIsNone(self.opening(campaign, "sell", 1))

    def test_sell_l2_hold_sell_l3_allows_l3(self):
        campaign = self.complete(self.transition(None, "sell", 2, 1), 2)
        campaign = self.transition(campaign, "hold", 0, 2)
        campaign = self.transition(campaign, "sell", 3, 3)

        decision = self.opening(campaign, "sell", 3)

        self.assertEqual(decision["action_type"], "open")
        self.assertEqual(decision["signal_level"], 3)

    def test_direct_l3_jump_executes_only_l3_allocation(self):
        campaign = self.transition(None, "sell", 3, 1)

        decision = self.opening(campaign, "sell", 3)

        self.assertEqual(decision["requested_quantity"], 15)

    def test_opposite_actionable_signal_starts_new_campaign(self):
        sell = self.complete(self.transition(None, "sell", 1, 1), 1)
        held = self.transition(sell, "hold", 0, 2)
        buy = self.transition(held, "buy", 1, 3)

        self.assertNotEqual(buy["campaign_id"], sell["campaign_id"])
        self.assertEqual(buy["active_side"], "buy")
        self.assertEqual(buy["highest_completed_level"], 0)
        decision = self.opening(buy, "buy", 1)
        self.assertEqual(decision["action_type"], "campaign_only")

    def test_sell_l1_is_available_after_buy_campaign_resets_sell(self):
        sell = self.complete(self.transition(None, "sell", 3, 1), 3)
        buy = self.complete(self.transition(sell, "buy", 1, 2), 1)
        next_sell = self.transition(buy, "sell", 1, 3)

        self.assertNotEqual(next_sell["campaign_id"], sell["campaign_id"])
        self.assertEqual(next_sell["active_side"], "sell")
        self.assertEqual(next_sell["highest_completed_level"], 0)
        self.assertEqual(self.opening(next_sell, "sell", 1)["action_type"], "open")

    def test_hold_before_any_actionable_signal_has_no_campaign(self):
        campaign = self.transition(None, "hold", 0, 1)

        self.assertIsNone(campaign["campaign_id"])
        self.assertIsNone(campaign["active_side"])
        self.assertEqual(campaign["highest_completed_level"], 0)
        self.assertEqual(campaign["last_signal_side"], "hold")

    def test_multiple_holds_preserve_campaign_progress(self):
        campaign = self.complete(self.transition(None, "sell", 2, 1), 2)
        identity = campaign["campaign_id"]
        campaign = self.transition(campaign, "hold", 0, 2)
        campaign = self.transition(campaign, "hold", 0, 3)

        self.assertEqual(campaign["campaign_id"], identity)
        self.assertEqual(campaign["active_side"], "sell")
        self.assertEqual(campaign["highest_completed_level"], 2)
        self.assertEqual(campaign["last_signal_side"], "hold")
        self.assertEqual(campaign["last_signal_at_ms"], 3)

    def test_campaigns_are_independent_per_asset(self):
        near = self.complete(self.transition(None, "sell", 2, 1), 2)
        xrp = self.complete(self.transition(None, "buy", 1, 2), 1)
        near = self.transition(near, "hold", 0, 3)

        self.assertEqual(near["active_side"], "sell")
        self.assertEqual(near["highest_completed_level"], 2)
        self.assertEqual(xrp["active_side"], "buy")
        self.assertEqual(xrp["highest_completed_level"], 1)

    def test_persisted_campaign_survives_restart_and_blocks_duplicate_l1(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            bootstrap_runtime_db(db_path)
            opened = sync_spot_strategy_campaign(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                opportunity="sell",
                signal_level=1,
                signal_at_ms=1,
                path=db_path,
            )
            complete_spot_strategy_campaign_level(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                campaign_id=opened["campaign_id"],
                signal_level=1,
                path=db_path,
            )

            bootstrap_runtime_db(db_path)
            held = sync_spot_strategy_campaign(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                opportunity="hold",
                signal_level=0,
                signal_at_ms=2,
                path=db_path,
            )
            resumed = sync_spot_strategy_campaign(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                opportunity="sell",
                signal_level=1,
                signal_at_ms=3,
                path=db_path,
            )
            persisted = load_spot_strategy_campaign("NEAR", path=db_path)

            self.assertEqual(held["campaign_id"], opened["campaign_id"])
            self.assertEqual(resumed["campaign_id"], opened["campaign_id"])
            self.assertEqual(resumed["highest_completed_level"], 1)
            self.assertEqual(persisted, resumed)
            self.assertIsNone(self.opening(resumed, "sell", 1))

    def test_campaign_capacity_and_consumption_survive_restart_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            bootstrap_runtime_db(db_path)
            opened = sync_spot_strategy_campaign(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                opportunity="sell",
                signal_level=1,
                signal_at_ms=1,
                path=db_path,
            )
            snapshot = initialize_sell_campaign_capacity(
                opened,
                {
                    "target_quantity": 1000,
                    "minimum_holding_pct": 60,
                    "policy_sellable_quantity": 400,
                },
            )
            persisted = initialize_spot_strategy_campaign_capacity(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                campaign_id=opened["campaign_id"],
                snapshot=snapshot,
                path=db_path,
            )
            ensure_spot_strategy_action(
                {
                    "action_key": "open:restart-safe:level:1",
                    "account_key": "manual_spot",
                    "asset_symbol": "NEAR",
                    "quote_symbol": "USDT",
                    "campaign_id": opened["campaign_id"],
                    "action_type": "open",
                    "side": "sell",
                    "signal_level": 1,
                    "swing_id": None,
                    "requested_quantity": 20,
                    "reason": {},
                },
                path=db_path,
            )
            for _ in range(2):
                complete_spot_strategy_campaign_level(
                    account_key="manual_spot",
                    asset_symbol="NEAR",
                    quote_symbol="USDT",
                    campaign_id=opened["campaign_id"],
                    signal_level=1,
                    consumed_quantity=20,
                    action_key="open:restart-safe:level:1",
                    path=db_path,
                )

            bootstrap_runtime_db(db_path)
            resumed = load_spot_strategy_campaign("NEAR", path=db_path)

            self.assertEqual(persisted["campaign_capacity_quantity"], 200)
            self.assertEqual(resumed["campaign_start_policy_sellable_reference"], 400)
            self.assertEqual(resumed["campaign_capacity_quantity"], 200)
            self.assertEqual(resumed["campaign_consumed_quantity"], 20)
            self.assertEqual(resumed["highest_completed_level"], 1)

    def test_campaign_migration_preserves_existing_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE spot_strategy_campaigns (
                        account_key TEXT NOT NULL,
                        asset_symbol TEXT NOT NULL,
                        quote_symbol TEXT NOT NULL,
                        campaign_id TEXT,
                        active_side TEXT,
                        highest_completed_level INTEGER NOT NULL DEFAULT 0,
                        last_signal_level INTEGER NOT NULL DEFAULT 0,
                        last_signal_at_ms INTEGER,
                        created_at_ms INTEGER NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (account_key, asset_symbol, quote_symbol)
                    );
                    INSERT INTO spot_strategy_campaigns VALUES (
                        'manual_spot', 'NEAR', 'USDT', 'legacy-sell', 'sell', 2, 2, 1, 1, 1
                    );
                    """
                )

            bootstrap_runtime_db(db_path)
            migrated = load_spot_strategy_campaign("NEAR", path=db_path)
            held = sync_spot_strategy_campaign(
                account_key="manual_spot",
                asset_symbol="NEAR",
                quote_symbol="USDT",
                opportunity="hold",
                signal_level=0,
                signal_at_ms=2,
                path=db_path,
            )

            self.assertIsNone(migrated["last_signal_side"])
            self.assertEqual(held["campaign_id"], "legacy-sell")
            self.assertEqual(held["active_side"], "sell")
            self.assertEqual(held["highest_completed_level"], 2)
            self.assertEqual(held["last_signal_side"], "hold")

    def test_action_migration_preserves_historical_open_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            bootstrap_runtime_db(db_path)
            action = ensure_spot_strategy_action(
                {
                    "action_key": "open:legacy-campaign:level:1",
                    "account_key": "manual_spot",
                    "asset_symbol": "NEAR",
                    "quote_symbol": "USDT",
                    "campaign_id": "legacy-campaign",
                    "action_type": "open",
                    "side": "sell",
                    "signal_level": 1,
                    "swing_id": "legacy-swing",
                    "requested_quantity": 10,
                    "reason": {"action_type": "open"},
                },
                path=db_path,
            )
            with closing(sqlite3.connect(db_path)) as connection:
                connection.executescript(
                    """
                    DROP TABLE spot_accumulation_target_ratchets;
                    ALTER TABLE spot_strategy_actions RENAME TO spot_strategy_actions_v15;
                    CREATE TABLE spot_strategy_actions (
                        action_key TEXT PRIMARY KEY,
                        account_key TEXT NOT NULL,
                        asset_symbol TEXT NOT NULL,
                        quote_symbol TEXT NOT NULL DEFAULT 'USDT',
                        campaign_id TEXT,
                        action_type TEXT NOT NULL CHECK (action_type IN ('open', 'close')),
                        side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                        signal_level INTEGER,
                        swing_id TEXT NOT NULL,
                        intent_id TEXT,
                        status TEXT NOT NULL CHECK (
                            status IN ('planned', 'intent_created', 'executing', 'completed', 'retryable', 'blocked')
                        ),
                        requested_quantity REAL NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        reason_json TEXT NOT NULL DEFAULT '{}',
                        error TEXT,
                        created_at_ms INTEGER NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        completed_at_ms INTEGER
                    );
                    INSERT INTO spot_strategy_actions SELECT * FROM spot_strategy_actions_v15;
                    DROP TABLE spot_strategy_actions_v15;
                    """
                )

            bootstrap_runtime_db(db_path)
            migrated = list_spot_strategy_actions(path=db_path)
            nullable_swing = ensure_spot_strategy_action(
                {
                    "action_key": "campaign_only:new-campaign:level:1",
                    "account_key": "manual_spot",
                    "asset_symbol": "NEAR",
                    "quote_symbol": "USDT",
                    "campaign_id": "new-campaign",
                    "action_type": "campaign_only",
                    "side": "buy",
                    "signal_level": 1,
                    "swing_id": None,
                    "requested_quantity": 0,
                    "reason": {"action_type": "campaign_only"},
                },
                path=db_path,
            )

            self.assertEqual(migrated[0]["action_key"], action["action_key"])
            self.assertEqual(migrated[0]["swing_id"], "legacy-swing")
            self.assertIsNone(nullable_swing["swing_id"])


class RecordingProgressiveExecutor(ProgressiveSpotSwingExecutor):
    def _execute_action(self, action):
        updated = update_spot_strategy_action(
            action["action_key"],
            {"status": "completed", "completed_at_ms": action["updated_at_ms"]},
            path=self.db_path,
        ) or action
        if action["action_type"] in {"open", "accumulate_asset"}:
            complete_spot_strategy_campaign_level(
                account_key=action["account_key"],
                asset_symbol=action["asset_symbol"],
                quote_symbol=action["quote_symbol"],
                campaign_id=action["campaign_id"],
                signal_level=action["signal_level"],
                consumed_quantity=(
                    action["requested_quantity"] if action["action_type"] == "open" else 0
                ),
                action_key=(action["action_key"] if action["action_type"] == "open" else None),
                path=self.db_path,
            )
        return updated


class BlockingFirstProgressiveExecutor(RecordingProgressiveExecutor):
    def _execute_action(self, action):
        if action["swing_id"] == "swing-dust":
            return update_spot_strategy_action(
                action["action_key"],
                {
                    "status": "blocked",
                    "error": "Spot order quantity rounds to zero under the Binance step size.",
                },
                path=self.db_path,
            ) or action
        return super()._execute_action(action)


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

    def test_accumulation_target_ratchet_uses_net_fill_and_is_idempotent(self):
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "target_quantity": 1000,
                "minimum_holding_pct": 80,
                "trading_objective": "accumulate_asset",
            },
            path=self.db_path,
        )
        action = ensure_spot_strategy_action(
            {
                "action_key": "accumulate_asset:NEAR:campaign-buy:level:1",
                "account_key": "manual_spot",
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "campaign_id": "campaign-buy",
                "action_type": "accumulate_asset",
                "side": "buy",
                "signal_level": 1,
                "swing_id": None,
                "requested_quantity": 50,
                "reason": {"action_type": "accumulate_asset"},
            },
            path=self.db_path,
        )
        update_spot_strategy_action(
            action["action_key"],
            {"intent_id": "intent-accumulation"},
            path=self.db_path,
        )

        first = apply_spot_accumulation_target_ratchet(
            action["action_key"],
            intent_id="intent-accumulation",
            net_acquired_quantity=49.95,
            deployed_quote_quantity=100,
            average_price=2,
            fee_amount=0.05,
            fee_asset="NEAR",
            path=self.db_path,
        )
        second = apply_spot_accumulation_target_ratchet(
            action["action_key"],
            intent_id="intent-accumulation",
            net_acquired_quantity=49.95,
            deployed_quote_quantity=100,
            average_price=2,
            fee_amount=0.05,
            fee_asset="NEAR",
            path=self.db_path,
        )

        policy = list_portfolio_asset_policies(path=self.db_path)[0]
        self.assertEqual(first["applied_gain_quantity"], 49.95)
        self.assertTrue(second["idempotent_replay"])
        self.assertAlmostEqual(policy["target_quantity"], 1049.95)
        self.assertEqual(len(list_spot_accumulation_target_ratchets(path=self.db_path)), 1)

    def test_cash_buy_campaign_levels_are_consumed_without_order_or_swing(self):
        executor = RecordingProgressiveExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        buy = {
            **self.signal(1, 10),
            "opportunity": "buy",
            "current_price": 4,
            "rolling_change_pct": -10,
        }
        with patch("bot.spot_strategy.load_portfolio_snapshot", side_effect=self.portfolio):
            first = executor.handle_signal(buy)
            repeated = executor.handle_signal(buy)
            second = executor.handle_signal({**buy, "level": 2, "final_tranche_pct": 20})

        campaign = load_spot_strategy_campaign("NEAR", path=self.db_path)
        actions = list_spot_strategy_actions(path=self.db_path)
        self.assertEqual(first["action_type"], "campaign_only")
        self.assertIsNone(repeated)
        self.assertEqual(second["action_type"], "campaign_only")
        self.assertEqual(campaign["highest_completed_level"], 2)
        self.assertEqual(list_spot_swings(path=self.db_path), [])
        self.assertEqual(len(actions), 2)

    def test_asset_buy_levels_create_accumulations_but_no_swings(self):
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": "NEAR",
                "quote_symbol": "USDT",
                "target_quantity": 1000,
                "minimum_holding_pct": 80,
                "trading_objective": "accumulate_asset",
            },
            path=self.db_path,
        )
        executor = RecordingProgressiveExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
            config=ProgressiveSwingConfig(
                treasury_accumulation_enabled=True,
                estimated_fee_rate=0,
            ),
        )
        buy = {
            **self.signal(1, 10),
            "opportunity": "buy",
            "trading_objective": "accumulate_asset",
            "current_price": 4,
            "rolling_change_pct": -10,
        }
        portfolio = self.portfolio()[0]
        portfolio["summary"] = {"vault_reserve": 1000, "vault_reserve_available": 100}
        with patch("bot.spot_strategy.load_portfolio_snapshot", return_value=(portfolio, [])):
            first = executor.handle_signal(buy)
            repeated = executor.handle_signal(buy)
            second = executor.handle_signal({**buy, "level": 2, "final_tranche_pct": 20})

        self.assertEqual(first["action_type"], "accumulate_asset")
        self.assertIsNone(repeated)
        self.assertEqual(second["action_type"], "accumulate_asset")
        self.assertEqual(list_spot_swings(path=self.db_path), [])

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

    def test_temporary_blocked_close_is_replanned_with_fresh_quantity(self):
        executor = ProgressiveSpotSwingExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        first = executor._persist_close_action(
            "NEAR",
            "USDT",
            {
                "action_type": "close",
                "side": "buy",
                "swing_id": "swing-replanned-close",
                "requested_quantity": 4.8,
                "reason": {"quantity_basis": "available_quote"},
            },
        )
        update_spot_strategy_action(
            first["action_key"],
            {
                "status": "blocked",
                "error": "Partial Spot close would leave an untradeable remainder under Binance filters.",
            },
            path=self.db_path,
        )

        replanned = executor._persist_close_action(
            "NEAR",
            "USDT",
            {
                "action_type": "close",
                "side": "buy",
                "swing_id": "swing-replanned-close",
                "requested_quantity": 5.0,
                "reason": {"quantity_basis": "remaining_asset"},
            },
        )

        self.assertEqual(replanned["status"], "blocked")
        self.assertEqual(replanned["requested_quantity"], 5.0)
        self.assertEqual(replanned["reason"]["quantity_basis"], "remaining_asset")

    def test_live_signal_batch_prioritizes_close_before_other_asset_opening(self):
        executor = ProgressiveSpotSwingExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        create_spot_swing(
            {
                "swing_id": "swing-near-close",
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
                "execution_id": "open-near-close",
                "swing_id": "swing-near-close",
                "symbol": "NEARUSDT",
                "side": "sell",
                "quantity": 10,
                "price": 10,
                "source": "strategy",
            },
            path=self.db_path,
        )
        portfolio = {
            "holdings": [
                {"asset_symbol": "XRP", "quote_symbol": "USDT", "market_price": 1},
                {"asset_symbol": "NEAR", "quote_symbol": "USDT", "market_price": 5},
            ],
            "exchange": {"usdt_free": 1000},
            "summary": {"vault_reserve": 1000},
        }
        signals = [
            {"asset_symbol": "XRP", "quote_symbol": "USDT", "opportunity": "buy", "current_price": 1},
            {"asset_symbol": "NEAR", "quote_symbol": "USDT", "opportunity": "hold", "current_price": 5},
        ]

        with patch("bot.spot_strategy.load_portfolio_snapshot", return_value=(portfolio, [])):
            ordered = executor.order_signals_for_execution(signals)

        self.assertEqual([item["asset_symbol"] for item in ordered], ["NEAR", "XRP"])

    def test_blocked_dust_close_falls_through_to_next_profitable_swing(self):
        executor = BlockingFirstProgressiveExecutor(
            object(),
            logger=logging.getLogger("test.spot-progression"),
            db_path=self.db_path,
        )
        for swing_id, price, quantity in (
            ("swing-dust", 10, 0.001),
            ("swing-tradable", 8, 10),
        ):
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
                    "quantity": quantity,
                    "price": price,
                    "source": "strategy",
                },
                path=self.db_path,
            )

        with patch("bot.spot_strategy.load_portfolio_snapshot", side_effect=self.portfolio):
            result = executor.handle_signal(self.signal(1, 10))

        actions = list_spot_strategy_actions(path=self.db_path)
        self.assertEqual(result["swing_id"], "swing-tradable")
        self.assertEqual(
            {action["swing_id"]: action["status"] for action in actions},
            {"swing-dust": "blocked", "swing-tradable": "completed"},
        )


if __name__ == "__main__":
    unittest.main()
