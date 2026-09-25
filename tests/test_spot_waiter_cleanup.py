from __future__ import annotations

import unittest

from shared.spot_progression import ProgressiveSwingConfig
from shared.spot_strategy import plan_spot_strategy_action
from shared.spot_swing import calculate_swing_economics
from shared.spot_waiter_cleanup import (
    DAY_MS,
    WaiterCleanupConfig,
    remaining_unrealized_pnl_pct,
    required_cleanup_level,
    reverse_signal_satisfies,
)


NOW_MS = 200 * DAY_MS


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
        self.assertIsNone(self.requirement(29 + 23 / 24))
        self.assertEqual(self.requirement(30)["required_reverse_level"], 3)
        self.assertEqual(self.requirement(60)["required_reverse_level"], 2)
        self.assertEqual(self.requirement(90)["required_reverse_level"], 1)

    def test_deep_loss_boundary_and_minimum_age(self):
        eligible = self.requirement(15, -20)
        self.assertEqual(eligible["cleanup_reason"], "deep_loss_cleanup")
        self.assertEqual(eligible["required_reverse_level"], 1)
        self.assertIsNone(self.requirement(15 - 1 / 24, -25))

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
        self.assertEqual(remaining_unrealized_pnl_pct(economics), -25)


class WaiterCleanupPriorityTests(unittest.TestCase):
    def decide(
        self,
        current_signal: dict,
        swings: list[dict],
        *,
        cleanup: WaiterCleanupConfig | None = None,
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
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
            cleanup_config=cleanup or WaiterCleanupConfig(),
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
            signal("sell", 1, 80),
            swings,
            excluded_close_swing_ids={"oldest-dust"},
        )

        self.assertEqual(decision["swing_id"], "next-tradable")
        self.assertEqual(decision["reason"]["close_reason"], "deep_loss_cleanup")

    def test_cash_limited_cleanup_quantity_is_marked_as_temporary(self):
        decision = plan_spot_strategy_action(
            signal("buy", 1, 120),
            {
                "target_quantity": 1000,
                "minimum_holding_pct": 80,
                "immediately_sellable_quantity": 200,
                "market_price": 120,
            },
            {},
            [swing_state("cash-limited", origin_side="sell", age_days=100)],
            available_quote=0.1,
            config=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=0),
            cleanup_config=WaiterCleanupConfig(),
        )

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["reason"]["quantity_basis"], "available_quote")

    def test_maximum_open_bargains_prevents_eleventh_open(self):
        swings = [swing_state(f"buy-{index}", age_days=1) for index in range(10)]
        decision = self.decide(signal("buy", 1, 80), swings)

        self.assertEqual(decision["action_type"], "hold")
        self.assertEqual(decision["reason"]["hold_reason"], "max_open_bargains_per_asset")

    def test_full_capacity_wrong_direction_holds(self):
        swings = [swing_state(f"buy-{index}", age_days=31) for index in range(10)]
        decision = self.decide(signal("buy", 1, 80), swings)

        self.assertEqual(decision["action_type"], "hold")

    def test_capacity_cleanup_chooses_oldest_eligible_bargain(self):
        swings = [
            swing_state(f"buy-{index}", age_days=31 + index)
            for index in range(10)
        ]
        decision = self.decide(signal("sell", 2, 85), swings)

        self.assertEqual(decision["action_type"], "close")
        self.assertEqual(decision["swing_id"], "buy-9")
        self.assertEqual(decision["reason"]["close_reason"], "capacity_cleanup")
        self.assertTrue(decision["reason"]["capacity_pressure"])

    def test_disabled_cleanup_preserves_uncapped_opening_behavior(self):
        swings = [swing_state(f"buy-{index}", age_days=100) for index in range(10)]
        decision = self.decide(
            signal("buy", 1, 80),
            swings,
            cleanup=WaiterCleanupConfig(enabled=False),
        )

        self.assertEqual(decision["action_type"], "open")


if __name__ == "__main__":
    unittest.main()
