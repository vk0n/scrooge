from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backtest.spot_regime_sweep import (
    _combined_summary,
    _scenario_for_regime,
    load_spot_regime_sweep_config,
    rank_combined_rows,
    run_spot_regime_sweep,
)
from backtest.spot_regime_selection import select_non_overlapping_regimes
from backtest.spot_scenario import scenario_as_dict


BASE_SCENARIO = """
spot_backtest:
  name: regime-base
  start: '2026-01-01T00:00:00+00:00'
  end: '2026-01-03T00:00:00+00:00'
  interval: 1h
  starting_usdt: 0
  assets:
    AAA:
      quantity: 100
      entry_cost: 1
      custody: {binance: 100, cold_storage: 0, unassigned: 0}
      target_holding: 100
      minimum_holding_pct: 80
      trading_objective: accumulate_cash
  strategy:
    signal:
      levels_pct: [2, 3, 4, 6]
      base_tranches_pct: [10, 20, 30, 40]
      accumulation_tranches_pct: [1, 3, 5, 10]
  data_cache_dir: data
  output_dir: unused
"""


def regime_report(edge: float, *, recovery: float = 100.0) -> dict:
    return {
        "portfolio": {
            "starting_treasury_value": 1000,
            "final_treasury_value": 1000 + edge * 10,
            "hodl_final_treasury_value": 1000,
            "difference_vs_hodl": edge * 10,
            "edge_vs_hodl_pct_points": edge,
            "total_return_pct": edge,
            "hodl_return_pct": 0,
            "maximum_treasury_drawdown_pct": -10,
            "maximum_hodl_drawdown_pct": -12,
            "relative_wealth": {
                "minimum": 0.9,
                "maximum": 1.1,
                "terminal": 1 + edge / 100,
                "maximum_drawdown_pct": -8,
            },
        },
        "success_metrics": {
            "free_reserve": {
                "quote": 25,
                "pct_of_initial_invested_capital": 2.5,
                "retained_quote": 10,
                "spendable_quote": 15,
                "committed_quote": 5,
            },
            "asset_recovery": {
                "average_effective_quantity_pct": recovery,
                "weighted_effective_quantity_pct": recovery,
            },
        },
        "swings": {
            "total_opened": 10,
            "total_closed": 8,
            "still_open": 2,
            "total_fills": 18,
            "oldest_open_days": 3,
            "realized_pnl_quote": 12,
            "unrealized_open_pnl_quote": -2,
            "median_duration_hours": 12,
            "maximum_concurrent": 4,
        },
        "bargain_analysis": {
            "overview": {
                "closure_rate_pct": 80,
                "closed_win_rate_pct": 75,
                "profit_factor": 2,
                "duration_p90_hours": 30,
                "fee_drag_pct": 0.2,
                "quote_fees": 2,
            }
        },
        "treasury_accumulation": {
            "earned_cash_generated_quote": 8,
            "earned_cash_allocated_quote": 3,
            "overview": {
                "count": 2,
                "usdt_deployed": 3,
                "net_asset_acquired": 1,
                "target_growth_quantity": 1,
            },
            "per_asset": {},
        },
        "waiter_cleanup": {
            "cleanup_closes_total": 1,
            "accounting": {
                "economic_pnl_quote": -5,
                "restored_inventory_pnl_quote": -4,
                "inventory_residual_pnl_quote": -1,
                "inventory_deficit_market_value_quote": 3,
                "reserve_deployed_quote": 4,
                "repurchase_spend_quote": 10,
            },
            "by_reason": {},
        },
        "per_asset": {},
    }


class SpotRegimeSweepTests(unittest.TestCase):
    def write_config(self, root: Path, *, variants: int = 1) -> Path:
        (root / "base.yaml").write_text(BASE_SCENARIO, encoding="utf-8")
        goals = "[4]" if variants == 1 else "[4, 5]"
        path = root / "sweep.yaml"
        path.write_text(
            f"""
spot_sweep:
  name: regime-sweep
  base_config: base.yaml
  output_dir: output
  replay_parallel: false
  regimes:
    bull: {{start: '2023-01-01T00:00:00+00:00', end: '2024-01-01T00:00:00+00:00'}}
    neutral: {{start: '2024-01-01T00:00:00+00:00', end: '2024-12-31T00:00:00+00:00'}}
    bear: {{start: '2021-01-01T00:00:00+00:00', end: '2022-01-01T00:00:00+00:00'}}
  parameter_grid:
    close_profit_pct: {goals}
    free_cash_retention_pct: [20]
    unrealized_pnl_pct: [-25]
""",
            encoding="utf-8",
        )
        return path

    def test_loads_three_frozen_non_overlapping_regimes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_spot_regime_sweep_config(self.write_config(Path(tmp)))

        self.assertEqual([item.name for item in config.regimes], ["bull", "neutral", "bear"])
        self.assertEqual(len(config.sweep.variants), 1)

    def test_rejects_overlapping_regimes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp))
            text = path.read_text(encoding="utf-8").replace(
                "2024-01-01T00:00:00+00:00', end: '2024-12-31",
                "2023-06-01T00:00:00+00:00', end: '2024-06-01",
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_spot_regime_sweep_config(path)

    def test_scenarios_are_independent_with_same_assets_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_spot_regime_sweep_config(self.write_config(Path(tmp)))
            variant = config.sweep.variants[0]
            scenarios = [
                _scenario_for_regime(config, variant, regime)
                for regime in config.regimes
            ]

        self.assertEqual(len({item.output_dir for item in scenarios}), 3)
        self.assertEqual(len({item.asset_order for item in scenarios}), 1)
        self.assertEqual(len({item.progression for item in scenarios}), 1)
        self.assertEqual(len({item.signal for item in scenarios}), 1)
        self.assertEqual(len({item.waiter_cleanup for item in scenarios}), 1)

    def test_combined_metrics_are_equal_weighted_and_not_raw_dollar_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_spot_regime_sweep_config(self.write_config(Path(tmp)))
            variant = config.sweep.variants[0]
        rows = []
        for regime, edge, dollars in zip(config.regimes, (30.0, 3.0, -9.0), (1, 1_000, 1_000_000)):
            rows.append(
                {
                    "regime": regime.name,
                    "edge_vs_hodl_pct_points": edge,
                    "weighted_effective_assets_pct": 100,
                    "fee_drag_pct": 0.1,
                    "total_fills": 1,
                    "duration_seconds": 1,
                    "difference_vs_hodl": dollars,
                    "maximum_relative_drawdown_pct": -1,
                    "cleanup_net_pnl": 0,
                    "retained_reserve_quote": 0,
                }
            )

        combined = _combined_summary(variant, rows)

        self.assertEqual(combined["mean_edge_vs_hodl_pp"], 8)
        self.assertEqual(combined["worst_regime_edge_vs_hodl_pp"], -9)
        self.assertEqual(combined["regimes_beating_hodl"], 2)
        self.assertNotIn("summed_difference_vs_hodl", combined)

    def test_ranking_uses_cross_regime_mean_then_robustness(self):
        base = {
            "status": "ok",
            "regimes_beating_hodl": 2,
            "worst_weighted_effective_assets_pct": 100,
            "mean_fee_drag_pct": 0.2,
        }
        ranked = rank_combined_rows(
            [
                {**base, "name": "single-regime-star", "mean_edge_vs_hodl_pp": 4, "worst_regime_edge_vs_hodl_pp": -20},
                {**base, "name": "robust", "mean_edge_vs_hodl_pp": 5, "worst_regime_edge_vs_hodl_pp": -2},
                {**base, "name": "tie-better-worst", "mean_edge_vs_hodl_pp": 5, "worst_regime_edge_vs_hodl_pp": 1},
            ]
        )

        self.assertEqual([item["name"] for item in ranked], ["tie-better-worst", "robust", "single-regime-star"])

    def test_selector_uses_hodl_only_and_enforces_non_overlap(self):
        candidates = [
            {"start": "2021-01-01", "end": "2022-01-01", "return_pct": -80},
            {"start": "2022-01-01", "end": "2023-01-01", "return_pct": 40},
            {"start": "2023-01-01", "end": "2024-01-01", "return_pct": 0.5},
            {"start": "2021-06-01", "end": "2022-06-01", "return_pct": 90},
        ]

        selected = select_non_overlapping_regimes(candidates)

        self.assertEqual(selected["bear"]["return_pct"], -80)
        self.assertEqual(selected["bull"]["return_pct"], 40)
        self.assertEqual(selected["neutral"]["return_pct"], 0.5)

    def test_one_candidate_runs_exactly_three_clean_backtests(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_spot_regime_sweep_config(self.write_config(Path(tmp)))
            edges = iter((10.0, 2.0, -4.0))
            with (
                patch("backtest.spot_regime_sweep.BinanceSpotHistoricalAdapter.load", return_value=object()),
                patch("backtest.spot_regime_sweep._write_experiment_manifest", return_value={}),
                patch("backtest.spot_regime_sweep.SpotPortfolioBacktester") as backtester,
                patch("backtest.spot_regime_sweep.write_spot_backtest_artifacts") as write,
            ):
                backtester.return_value.run.return_value = object()

                def artifacts(_result, output_dir):
                    payload = regime_report(next(edges))
                    output = Path(output_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                    (output / "scenario.resolved.json").write_text(
                        json.dumps({"scenario": scenario_as_dict(backtester.call_args.args[0])}),
                        encoding="utf-8",
                    )
                    return {"report": payload}

                write.side_effect = artifacts
                payload = run_spot_regime_sweep(config)

            self.assertEqual(backtester.call_count, 3)
            scenarios = [call.args[0] for call in backtester.call_args_list]
            self.assertEqual(len({id(item) for item in scenarios}), 3)
            self.assertEqual(payload["completed"], 1)
            self.assertTrue((config.output_dir / "candidates" / config.sweep.variants[0].name / "combined_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
