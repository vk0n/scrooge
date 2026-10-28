import unittest

from backtest.full_cleanup_optimization import (
    AGING_SCHEDULES,
    _annotate_acceptance,
    _deduplicate,
    _exclude_evaluated,
    _pareto,
    _select_capacity_regions,
    complete_variant,
    initial_stage_variants,
)


class FullCleanupOptimizationTests(unittest.TestCase):
    def test_initial_stages_have_expected_unique_complete_candidates(self):
        stages = initial_stage_variants()

        self.assertEqual(len(stages["stage-0-baseline"]), 1)
        self.assertEqual(len(stages["stage-1-capacity"]), 29)
        self.assertEqual(len(stages["stage-2-deep-loss"]), 59)
        self.assertEqual(len(stages["stage-3-aging"]), len(AGING_SCHEDULES) - 1)
        for variants in stages.values():
            self.assertEqual(len(_deduplicate(variants)), len(variants))
            for variant in variants:
                self.assertTrue(variant["deep_loss_enabled"])
                self.assertTrue(variant["aging_cleanup_enabled"])
                self.assertTrue(variant["capacity_cleanup_enabled"])
                self.assertIsInstance(variant["max_open_bargains_per_asset"], int)
                self.assertLess(variant["age_l3_min_age_days"], variant["age_l2_min_age_days"])
                self.assertLess(variant["age_l2_min_age_days"], variant["age_l1_min_age_days"])

    def test_restore_cost_does_not_affect_operational_acceptance(self):
        baseline = {
            "maximum_drawdown_pct": -50.0,
            "weighted_effective_assets_pct": 110.0,
        }
        row = {
            "open_bargains_90_plus": 0,
            "oldest_open_days": 30.0,
            "maximum_drawdown_pct": -51.0,
            "weighted_effective_assets_pct": 105.0,
            "final_capacity_utilization_pct": 90.0,
            "restore_cost_quote": 10_000_000.0,
        }

        _annotate_acceptance([row], baseline)

        self.assertTrue(row["operationally_accepted"])

    def test_pareto_keeps_only_nondominated_edge_burn_rows(self):
        rows = [
            {"name": "a", "status": "ok", "edge_vs_hodl_pct_points": 20, "cleanup_cash_burn_ratio_pct": 80},
            {"name": "b", "status": "ok", "edge_vs_hodl_pct_points": 18, "cleanup_cash_burn_ratio_pct": 70},
            {"name": "c", "status": "ok", "edge_vs_hodl_pct_points": 17, "cleanup_cash_burn_ratio_pct": 90},
        ]

        self.assertEqual([row["name"] for row in _pareto(rows)], ["a", "b"])

    def test_invalid_aging_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            complete_variant(
                age_l3_min_age_days=60,
                age_l2_min_age_days=45,
                age_l1_min_age_days=90,
            )

    def test_later_stages_exclude_already_evaluated_parameters(self):
        baseline = complete_variant()
        fresh = complete_variant(max_open_bargains_per_asset=11)

        self.assertEqual(_exclude_evaluated([baseline, fresh], [baseline]), [fresh])

    def test_capacity_selection_keeps_winner_neighbor_and_distinct_caps(self):
        baseline = {
            "maximum_drawdown_pct": -50.0,
            "weighted_effective_assets_pct": 110.0,
        }

        def row(cap, age, edge):
            return {
                "max_open_bargains_per_asset": cap,
                "capacity_cleanup_min_age_days": age,
                "edge_vs_hodl_pct_points": edge,
                "open_bargains": cap * 6,
                "open_bargains_90_plus": 0,
                "oldest_open_days": 20.0,
                "maximum_drawdown_pct": -50.0,
                "weighted_effective_assets_pct": 110.0,
            }

        selected = _select_capacity_regions(
            [
                row(8, 30, 25),
                row(8, 45, 22),
                row(6, 30, 23),
                row(10, 21, 21),
                row(12, 30, 18),
            ],
            baseline,
        )

        self.assertEqual(
            [
                (item["max_open_bargains_per_asset"], item["capacity_cleanup_min_age_days"])
                for item in selected
            ],
            [(8, 30), (8, 45), (6, 30), (10, 21)],
        )


if __name__ == "__main__":
    unittest.main()
