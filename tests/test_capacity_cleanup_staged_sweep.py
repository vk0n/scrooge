import unittest

from backtest.capacity_cleanup_staged_sweep import (
    _balanced_choice,
    stage_1_variants,
    stage_2_variants,
    stage_3_variants,
)


def row(variant, *, edge, burn, blocked=10, oldest=30):
    return {
        **variant,
        "status": "ok",
        "edge_vs_hodl_pct_points": edge,
        "cleanup_cash_burn_ratio_pct": burn,
        "unique_capacity_blocked_opportunities": blocked,
        "oldest_open_days": oldest,
    }


class CapacityCleanupStagedSweepTests(unittest.TestCase):
    def test_stage_counts_and_fixed_policy(self):
        stage_1 = stage_1_variants()
        stage_2 = stage_2_variants(stage_1[0])

        self.assertEqual(len(stage_1), 16)
        self.assertEqual(len(stage_2), 72)
        self.assertEqual({item["max_open_bargains_per_asset"] for item in stage_2}, {7})
        self.assertEqual(
            {
                (
                    item["age_l3_min_age_days"],
                    item["age_l2_min_age_days"],
                    item["age_l1_min_age_days"],
                )
                for item in stage_2
            },
            {(30, 60, 90)},
        )

    def test_balanced_choice_uses_edge_burn_pareto_frontier(self):
        variants = stage_1_variants()[:3]
        selected = _balanced_choice(
            [
                row(variants[0], edge=10, burn=80),
                row(variants[1], edge=9, burn=60),
                row(variants[2], edge=8, burn=90),
            ]
        )

        self.assertIn(selected["name"], {variants[0]["name"], variants[1]["name"]})
        self.assertNotEqual(selected["name"], variants[2]["name"])

    def test_stage_3_is_unique_and_within_planned_size(self):
        stage_1 = [
            row(item, edge=20 - index / 10, burn=50 + index)
            for index, item in enumerate(stage_1_variants())
        ]
        capacity_winner = _balanced_choice(stage_1)
        stage_2 = [
            row(item, edge=18 - index / 100, burn=55 + index / 10)
            for index, item in enumerate(stage_2_variants(capacity_winner))
        ]

        variants, capacities, deep_losses = stage_3_variants(stage_1, stage_2)

        self.assertEqual(len(capacities), 5)
        self.assertEqual(len(deep_losses), 10)
        self.assertGreaterEqual(len(variants), 30)
        self.assertLessEqual(len(variants), 50)
        self.assertEqual(len({item["name"] for item in variants}), len(variants))


if __name__ == "__main__":
    unittest.main()
