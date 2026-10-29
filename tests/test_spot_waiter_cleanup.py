from __future__ import annotations

import unittest

from core.spot_progression import ProgressiveSwingConfig
from core.spot_strategy import plan_spot_strategy_action
from core.spot_swing import calculate_swing_economics
from core.spot_waiter_cleanup import (
    AgingCleanupRule,
    DAY_MS,
    WaiterCleanupConfig,
    remaining_unrealized_pnl_pct,
    required_cleanup_level,
    reverse_signal_satisfies,
)


NOW_MS = 200 * DAY_MS


def priority_test_cleanup_config(**overrides) -> WaiterCleanupConfig:
    values = {
        "max_open_bargains_per_asset": 10,
        "deep_loss_min_age_days": 15,
        "deep_loss_unrealized_pnl_pct": -25,
        "deep_loss_required_reverse_level": 1,
        "aging_rules": (
            AgingCleanupRule(min_age_days=30, required_reverse_level=3),
            AgingCleanupRule(min_age_days=60, required_reverse_level=2),
            AgingCleanupRule(min_age_days=90, required_reverse_level=1),
        ),
    }
    values.update(overrides)
    return WaiterCleanupConfig(**values)


def swing_state(
    swing_id: str,
    *,
    origin_side: str = "buy",
    age_days: float = 30,
    quantity: float = 10,
    price: float = 100,
    closing_quantity: float = 0,
    closing_price: float = 80,
) -> dict:
    executions = [
        {
            "execution_id": f"{swing_id}-open",
            "side": origin_side,
            "quantity": quantity,
            "price": price,
            "quote_quantity": quantity * price,
            "fee_amount": 0,
            "fee_asset": "USDT",
        }
    ]
    if closing_quantity:
        executions.append(
            {
                "execution_id": f"{swing_id}-partial",
                "side": "sell" if origin_side == "buy" else "buy",
                "quantity": closing_quantity,
                "price": closing_price,
                "quote_quantity": closing_quantity * closing_price,
                "fee_amount": 0,
                "fee_asset": "USDT",
            }
        )
    return {
        "swing": {
            "swing_id": swing_id,
            "origin_side": origin_side,
            "trading_objective": "accumulate_cash",
            "asset_symbol": "AAA",
            "quote_symbol": "USDT",
            "source": "strategy",
            "status": "partially_closed" if closing_quantity else "open",
            "opened_at_ms": int(NOW_MS - age_days * DAY_MS),
        },
        "executions": executions,
    }


def signal(side: str, level: int, price: float) -> dict:
    return {
        "opportunity": side,
        "level": level,
        "strategy_eligible": True,
        "trading_objective": "accumulate_cash",
        "final_tranche_pct": 10,
        "current_price": price,
        "current_at_ms": NOW_MS,
        "evaluated_at_ms": NOW_MS,
    }


def holding(price: float) -> dict:
    return {
        "target_quantity": 100,
        "minimum_holding_pct": 0,
        "immediately_sellable_quantity": 100,
        "market_price": price,
    }


class WaiterCleanupDomainTests(unittest.TestCase):
    def setUp(self):
        self.config = WaiterCleanupConfig()

    def requirement(self, age_days: float, pnl_pct: float = -5) -> dict | None:
        bargain = swing_state("age", age_days=age_days)["swing"]
        return required_cleanup_level(
            bargain,
            now_ms=NOW_MS,
            unrealized_pnl_pct=pnl_pct,
            config=self.config,
        )

    def test_age_schedule_boundaries(self):
        self.assertIsNone(self.requirement(44 + 23 / 24))
        self.assertEqual(self.requirement(45)["required_reverse_level"], 3)
        self.assertEqual(self.requirement(60)["required_reverse_level"], 2)
        self.assertEqual(self.requirement(120)["required_reverse_level"], 1)

    def test_deep_loss_boundary_and_minimum_age(self):
        eligible = self.requirement(9, -45)
        self.assertEqual(eligible["cleanup_reason"], "deep_loss_cleanup")
        self.assertEqual(eligible["required_reverse_level"], 4)
        self.assertIsNone(self.requirement(9 - 1 / 24, -45))

    def test_deep_loss_and_aging_switches_are_independent(self):
        bargain = swing_state("switches", age_days=130)["swing"]
        deep_only = required_cleanup_level(
            bargain,
            now_ms=NOW_MS,
            unrealized_pnl_pct=-50,
            config=WaiterCleanupConfig(aging_cleanup_enabled=False),
        )
        aging_only = required_cleanup_level(
            bargain,
            now_ms=NOW_MS,
            unrealized_pnl_pct=-50,
            config=WaiterCleanupConfig(deep_loss_enabled=False),
        )
        disabled = required_cleanup_level(
            bargain,
            now_ms=NOW_MS,
            unrealized_pnl_pct=-50,
            config=WaiterCleanupConfig(
                deep_loss_enabled=False,
                aging_cleanup_enabled=False,
            ),
        )

        self.assertEqual(deep_only["cleanup_reason"], "deep_loss_cleanup")
        self.assertEqual(aging_only["cleanup_reason"], "age_l1_cleanup")
        self.assertIsNone(disabled)

    def test_reverse_level_is_minimum_and_direction_must_match(self):
        self.assertTrue(
            reverse_signal_satisfies(
                origin_side="buy", signal_side="sell", signal_level=4, required_level=3
            )
        )
        self.assertFalse(
            reverse_signal_satisfies(
                origin_side="buy", signal_side="sell", signal_level=2, required_level=3
            )
        )
        self.assertFalse(
            reverse_signal_satisfies(
                origin_side="buy", signal_side="buy", signal_level=4, required_level=1
            )
        )
        self.assertTrue(
            reverse_signal_satisfies(
                origin_side="sell", signal_side="buy", signal_level=1, required_level=1
            )
        )

    def test_partial_bargain_loss_uses_remaining_open_economics(self):
        state = swing_state(
            "partial",
            age_days=16,
            quantity=10,
            price=100,
            closing_quantity=5,
            closing_price=80,
        )
        economics = calculate_swing_economics(
            state["swing"], state["executions"], current_price=75
        )

        self.assertEqual(economics["remaining_opening_quote_quantity"], 500)
        self.assertEqual(economics["unrealized_pnl_quote"], -125)
        self.assertEqual(economics["unrealized_pnl_pct"], -25)
        self.assertEqual(remaining_unrealized_pnl_pct(economics), -25)


class WaiterCleanupPriorityTests(unittest.TestCase):
    def decide(
        self,
        current_signal: dict,
        swings: list[dict],
        *,
        cleanup: WaiterCleanupConfig | None = None,
        config: ProgressiveSwingConfig | None = None,
        excluded_close_swing_ids: set[str] | None = None,
    ) -> dict | None:
        return plan_spot_strategy_action(
            current_signal,
            holding(float(current_signal["current_price"])),
            {
                "active_side": current_signal["opportunity"],
                "highest_completed_level": 0,
                "campaign_id": "campaign",
            },
            swings,
            available_quote=100_000,
            config=config or ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
            cleanup_config=cleanup or priority_test_cleanup_config(),
            excluded_close_swing_ids=excluded_close_swing_ids,
        )

    def test_profitable_close_beats_cleanup(self):
        decision = self.decide(signal("sell", 1, 106), [swing_state("profitable", age_days=100)])

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["reason"]["close_reason"], "profit_target")

    def test_cleanup_beats_new_open(self):
        decision = self.decide(signal("sell", 3, 85), [swing_state("waiter", age_days=30)])

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["swing_id"], "waiter")
        self.assertEqual(decision["reason"]["close_reason"], "age_l3_cleanup")

    def test_unexecutable_profit_candidate_does_not_block_next_bargain(self):
        swings = [
            swing_state("dust", origin_side="sell", price=120),
            swing_state("tradable", origin_side="sell", price=110),
        ]

        decision = self.decide(
            signal("buy", 1, 100),
            swings,
            excluded_close_swing_ids={"dust"},
        )

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["swing_id"], "tradable")
        self.assertEqual(decision["reason"]["close_reason"], "profit_target")

    def test_excluded_cleanup_candidate_falls_through_to_next_oldest(self):
        swings = [
            swing_state("oldest-dust", age_days=100),
            swing_state("next-tradable", age_days=90),
        ]

        decision = self.decide(
            signal("sell", 1, 75),
            swings,
            excluded_close_swing_ids={"oldest-dust"},
        )

        self.assertEqual(decision["swing_id"], "next-tradable")
        self.assertEqual(decision["reason"]["close_reason"], "deep_loss_cleanup")

    def test_cash_limited_cleanup_quantity_records_available_quote_basis(self):
        decision = plan_spot_strategy_action(
            signal("buy", 1, 125),
            {
                "target_quantity": 1000,
                "minimum_holding_pct": 80,
                "immediately_sellable_quantity": 200,
                "market_price": 125,
            },
            {},
            [swing_state("cash-limited", origin_side="sell", age_days=100)],
            available_quote=0.1,
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
            cleanup_config=priority_test_cleanup_config(),
        )

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["reason"]["quantity_basis"], "available_quote")

    def test_sell_cleanup_cannot_spend_another_bargains_committed_cash(self):
        decision = plan_spot_strategy_action(
            signal("buy", 1, 125),
            holding(125),
            {},
            [swing_state("lossy-sell", origin_side="sell", age_days=100, price=100)],
            # The whole reserve includes this Swing's $1,000 proceeds plus $5 genuinely free cash.
            available_quote=1005,
            free_quote_reserve=5,
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
            cleanup_config=priority_test_cleanup_config(),
        )

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["reason"]["close_reason"], "deep_loss_cleanup")
        self.assertEqual(decision["reason"]["quantity_basis"], "available_quote")
        self.assertAlmostEqual(decision["requested_quantity"], 1005 / 125)

    def test_maximum_open_bargains_prevents_eleventh_open(self):
        swings = [
            swing_state(f"sell-{index}", origin_side="sell", age_days=1)
            for index in range(10)
        ]
        decision = self.decide(signal("sell", 1, 120), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(decision["reason"]["hold_reason"], "max_open_bargains_per_asset")

    def test_cash_buy_campaign_does_not_create_capacity_hold(self):
        swings = [swing_state(f"buy-{index}", age_days=31) for index in range(10)]
        decision = self.decide(signal("buy", 1, 80), swings)

        self.assertEqual(decision["action_type"], "campaign_only")

    def test_sell_waiter_cleanup_precedes_asset_accumulation_buy(self):
        waiter = swing_state("asset-waiter", origin_side="sell", age_days=31)
        waiter["swing"]["trading_objective"] = "accumulate_asset"
        buy_signal = {
            **signal("buy", 3, 125),
            "trading_objective": "accumulate_asset",
            "final_tranche_pct": 30,
        }

        decision = plan_spot_strategy_action(
            buy_signal,
            holding(125),
            {"campaign_id": "buy-campaign", "active_side": "buy", "highest_completed_level": 0},
            [waiter],
            available_quote=1000,
            available_accumulation_quote=100,
            config=ProgressiveSwingConfig(
                close_profit_pct=5,
                estimated_fee_rate=0,
            ),
            cleanup_config=priority_test_cleanup_config(),
        )

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["swing_id"], "asset-waiter")
        self.assertEqual(decision["reason"]["close_reason"], "deep_loss_cleanup")

    def test_capacity_cleanup_marks_highest_pnl_eligible_bargain_on_blocked_open(self):
        swings = [
            swing_state("oldest", origin_side="sell", age_days=60, price=121),
            swing_state("highest-pnl", origin_side="sell", age_days=31, price=124),
            *[
                swing_state(
                    f"sell-{index}",
                    origin_side="sell",
                    age_days=32 + index,
                    price=100,
                )
                for index in range(8)
            ],
        ]
        decision = self.decide(signal("sell", 1, 120), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(decision["swing_id"], "highest-pnl")
        self.assertEqual(decision["reason"]["hold_reason"], "capacity_cleanup_marked")
        self.assertGreater(
            decision["reason"]["capacity_cleanup_mark"][
                "pnl_quote_after_estimated_fee"
            ],
            0,
        )

    def test_capacity_cleanup_mark_can_close_at_a_loss_on_next_buy_signal(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=31 + index,
            )
            for index in range(10)
        ]
        swings[4]["swing"]["reference_state"] = {
            "capacity_cleanup_mark": {
                "marked_at_ms": NOW_MS - DAY_MS,
                "pnl_quote_after_estimated_fee": -50,
                "required_reverse_level": 1,
            }
        }

        decision = self.decide(signal("buy", 1, 120), swings)

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["swing_id"], "sell-4")
        self.assertEqual(decision["reason"]["close_reason"], "capacity_cleanup")
        self.assertLess(
            decision["reason"]["capacity_cleanup_pnl_after_estimated_fee"],
            0,
        )

    def test_capacity_cleanup_does_not_mark_without_a_blocked_opening(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=31 + index,
            )
            for index in range(10)
        ]

        decision = self.decide(signal("buy", 1, 99), swings)

        self.assertEqual(decision["action_type"], "campaign_only")

    def test_capacity_cleanup_waits_for_configured_reverse_level(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=31 + index,
            )
            for index in range(10)
        ]
        swings[0]["swing"]["reference_state"] = {
            "capacity_cleanup_mark": {"marked_at_ms": NOW_MS - DAY_MS}
        }
        cleanup = priority_test_cleanup_config(
            capacity_cleanup_required_reverse_level=2
        )

        level_one = self.decide(signal("buy", 1, 120), swings, cleanup=cleanup)
        level_two = self.decide(signal("buy", 2, 120), swings, cleanup=cleanup)

        self.assertEqual(level_one["action_type"], "campaign_only")
        self.assertEqual(level_two["action_type"], "close")
        self.assertEqual(level_two["swing_id"], "sell-0")
        self.assertEqual(level_two["reason"]["required_cleanup_level"], 2)

    def test_capacity_cleanup_follows_current_config_for_existing_mark(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=31 + index,
            )
            for index in range(10)
        ]
        swings[0]["swing"]["reference_state"] = {
            "capacity_cleanup_mark": {
                "marked_at_ms": NOW_MS - DAY_MS,
                "required_reverse_level": 2,
            }
        }
        current_config = priority_test_cleanup_config(
            capacity_cleanup_required_reverse_level=1
        )

        level_one = self.decide(signal("buy", 1, 120), swings, cleanup=current_config)

        self.assertEqual(level_one["action_type"], "close")
        self.assertEqual(level_one["reason"]["required_cleanup_level"], 1)

    def test_existing_capacity_mark_is_not_replaced(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=31 + index,
            )
            for index in range(10)
        ]
        swings[0]["swing"]["reference_state"] = {
            "capacity_cleanup_mark": {"marked_at_ms": NOW_MS - DAY_MS}
        }

        decision = self.decide(signal("sell", 1, 120), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(
            decision["reason"]["hold_reason"],
            "max_open_bargains_per_asset",
        )

    def test_capacity_cleanup_switch_does_not_disable_capacity_cap(self):
        swings = [
            swing_state(f"sell-{index}", origin_side="sell", age_days=31 + index)
            for index in range(10)
        ]

        buy_decision = self.decide(
            signal("buy", 1, 99),
            swings,
            cleanup=priority_test_cleanup_config(capacity_cleanup_enabled=False),
        )
        sell_decision = self.decide(
            signal("sell", 1, 120),
            swings,
            cleanup=priority_test_cleanup_config(capacity_cleanup_enabled=False),
        )

        self.assertEqual(buy_decision["action_type"], "campaign_only")
        self.assertEqual(sell_decision["action_type"], "hold")
        self.assertEqual(
            sell_decision["reason"]["hold_reason"],
            "max_open_bargains_per_asset",
        )

    def test_capacity_cleanup_mark_requires_minimum_age(self):
        swings = [
            swing_state(
                f"sell-{index}",
                origin_side="sell",
                age_days=29,
            )
            for index in range(10)
        ]

        decision = self.decide(signal("sell", 1, 120), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(
            decision["reason"]["hold_reason"],
            "max_open_bargains_per_asset",
        )

    def test_capacity_cleanup_does_not_mark_buy_origin_bargains(self):
        swings = [
            swing_state(f"buy-{index}", origin_side="buy", age_days=60)
            for index in range(10)
        ]

        decision = self.decide(signal("sell", 1, 90), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(
            decision["reason"]["hold_reason"],
            "max_open_bargains_per_asset",
        )

    def test_disabled_cleanup_preserves_capacity_cap(self):
        swings = [
            swing_state(f"sell-{index}", origin_side="sell", age_days=100)
            for index in range(10)
        ]
        decision = self.decide(
            signal("sell", 1, 120),
            swings,
            cleanup=priority_test_cleanup_config(enabled=False),
        )

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(decision["reason"]["hold_reason"], "max_open_bargains_per_asset")

    def test_explicit_null_capacity_preserves_historical_uncapped_control(self):
        swings = [
            swing_state(f"sell-{index}", origin_side="sell", age_days=100)
            for index in range(10)
        ]
        decision = self.decide(
            signal("sell", 1, 120),
            swings,
            cleanup=priority_test_cleanup_config(
                enabled=False,
                max_open_bargains_per_asset=None,
            ),
        )

        self.assertEqual(decision["action_type"], "open")


if __name__ == "__main__":
    unittest.main()
