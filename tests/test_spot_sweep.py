from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backtest.spot_sweep import (
    _rank_rows,
    _scenario_for_variant,
    load_spot_sweep_config,
    run_spot_sweep,
)
from backtest.spot_scenario import scenario_as_dict


BASE_SCENARIO = """
spot_backtest:
  name: sweep-base
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
      levels_pct: [2, 3, 5, 7]
      base_tranches_pct: [10, 20, 30, 40]
      accumulation_tranches_pct: [1, 3, 5, 10]
  data_cache_dir: data
  output_dir: unused
"""


def report(edge: float) -> dict:
    return {
        "portfolio": {
            "final_treasury_value": 1000 + edge,
            "hodl_final_treasury_value": 1000,
            "difference_vs_hodl": edge,
            "edge_vs_hodl_pct_points": edge / 10,
            "maximum_treasury_drawdown_pct": -10,
        },
        "success_metrics": {
            "free_reserve": {
                "quote": 25,
                "pct_of_initial_invested_capital": 2.5,
            },
            "asset_recovery": {
                "weighted_effective_quantity_pct": 99,
                "average_effective_quantity_pct": 101,
            },
        },
        "swings": {
            "total_opened": 10,
            "still_open": 2,
            "oldest_open_days": 3,
            "realized_pnl_quote": 12,
            "unrealized_open_pnl_quote": -2,
        },
        "treasury_accumulation": {
            "earned_cash_generated_quote": 8,
            "earned_cash_allocated_quote": 3,
        },
        "waiter_cleanup": {
            "cleanup_closes_total": 1,
            "accounting": {
                "economic_pnl_quote": -5,
                "restored_inventory_pnl_quote": -4,
                "inventory_residual_pnl_quote": -1,
                "inventory_deficit_market_value_quote": 3,
            },
        },
    }


class SpotSweepTests(unittest.TestCase):
    def write_configs(self, root: Path, levels: str = "[[2, 3, 5, 7], [3, 5, 8, 11]]") -> Path:
        base = root / "base.yaml"
        base.write_text(BASE_SCENARIO, encoding="utf-8")
        sweep = root / "sweep.yaml"
        sweep.write_text(
            f"""
spot_sweep:
  name: test-sweep
  base_config: base.yaml
  start: '2025-12-01T00:00:00+00:00'
  end: '2026-01-01T00:00:00+00:00'
  output_dir: output
  replay_parallel: false
  levels_pct: {levels}
""",
            encoding="utf-8",
        )
        return sweep

    def test_loads_and_validates_declarative_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_spot_sweep_config(self.write_configs(root))

        self.assertEqual(config.name, "test-sweep")
        self.assertEqual(config.scenario.start.isoformat(), "2025-12-01T00:00:00+00:00")
        self.assertEqual([item.name for item in config.variants], ["2-3-5-7", "3-5-8-11"])
        self.assertEqual(config.output_dir, root / "output")
        self.assertFalse(config.replay_parallel)
        self.assertEqual(config.replay_max_workers, 2)

    def test_rejects_duplicate_or_invalid_level_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_spot_sweep_config(
                    self.write_configs(root, "[[2, 3, 5, 7], [2, 3, 5, 7]]")
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_spot_sweep_config(self.write_configs(root, "[[2, 5, 4, 7]]"))

    def test_parameter_grid_builds_cartesian_strategy_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.yaml").write_text(BASE_SCENARIO, encoding="utf-8")
            sweep = root / "parameters.yaml"
            sweep.write_text(
                """
spot_sweep:
  name: parameter-sweep
  base_config: base.yaml
  output_dir: output
  parameter_grid:
    close_profit_pct: [2, 3]
    free_cash_retention_pct: [10, 20]
    unrealized_pnl_pct: [-15, -20]
""",
                encoding="utf-8",
            )

            config = load_spot_sweep_config(sweep)
            baseline = next(
                item
                for item in config.variants
                if item.close_profit_pct == 3
                and item.free_cash_retention_pct == 20
                and item.unrealized_pnl_pct == -20
            )
            resolved = _scenario_for_variant(config, baseline)

        self.assertEqual(len(config.variants), 8)
        self.assertEqual(config.variants[0].name, "tp-2-ret-10-loss-neg15")
        self.assertEqual(resolved.signal.levels_pct, (2.0, 3.0, 5.0, 7.0))
        self.assertEqual(resolved.progression.close_profit_pct, 3)
        self.assertEqual(resolved.free_cash_retention_pct, 20)
        self.assertEqual(resolved.waiter_cleanup.deep_loss_unrealized_pnl_pct, -20)

    def test_ranking_prioritizes_edge_then_effective_assets(self):
        rows = [
            {
                "name": "lower",
                "status": "ok",
                "edge_vs_hodl_pct_points": 1,
                "weighted_effective_assets_pct": 110,
                "free_reserve_pct": 10,
            },
            {
                "name": "winner",
                "status": "ok",
                "edge_vs_hodl_pct_points": 2,
                "weighted_effective_assets_pct": 90,
                "free_reserve_pct": 1,
            },
        ]

        ranked = _rank_rows(rows)

        self.assertEqual([row["name"] for row in ranked], ["winner", "lower"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2])

    def test_runner_loads_market_data_once_and_writes_ranked_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_spot_sweep_config(self.write_configs(root))
            fake_dataset = object()
            edges = iter((10.0, 25.0))

            with (
                patch("backtest.spot_sweep.BinanceSpotHistoricalAdapter.load", return_value=fake_dataset) as load,
                patch("backtest.spot_sweep.SpotPortfolioBacktester") as backtester,
                patch("backtest.spot_sweep.write_spot_backtest_artifacts") as write_artifacts,
            ):
                backtester.return_value.run.return_value = object()

                def write_result(_result, output_dir):
                    current = report(next(edges))
                    output = Path(output_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "summary.json").write_text(json.dumps(current), encoding="utf-8")
                    return {"report": current}

                write_artifacts.side_effect = write_result
                payload = run_spot_sweep(config)

            load.assert_called_once()
            self.assertEqual(backtester.call_count, 2)
            self.assertEqual(payload["completed"], 2)
            self.assertEqual(payload["rows"][0]["name"], "3-5-8-11")
            self.assertTrue((config.output_dir / "comparison.csv").exists())
            self.assertTrue((config.output_dir / "comparison.html").exists())
            self.assertTrue((config.output_dir / "manifest.json").exists())

    def test_runner_resumes_json_serialized_scenario_with_tuple_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_spot_sweep_config(self.write_configs(root, "[[2, 3, 5, 7]]"))
            variant = config.variants[0]
            scenario = _scenario_for_variant(config, variant)
            scenario.output_dir.mkdir(parents=True)
            (scenario.output_dir / "scenario.resolved.json").write_text(
                json.dumps({"scenario": scenario_as_dict(scenario)}),
                encoding="utf-8",
            )
            (scenario.output_dir / "summary.json").write_text(
                json.dumps(report(10.0)),
                encoding="utf-8",
            )

            with (
                patch("backtest.spot_sweep.BinanceSpotHistoricalAdapter.load", return_value=object()),
                patch("backtest.spot_sweep.SpotPortfolioBacktester") as backtester,
            ):
                payload = run_spot_sweep(config)

            backtester.assert_not_called()
            self.assertEqual(payload["completed"], 1)
            self.assertTrue(payload["rows"][0]["resumed"])


if __name__ == "__main__":
    unittest.main()
