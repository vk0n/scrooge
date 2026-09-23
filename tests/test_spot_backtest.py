from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import SpotCandle, SpotHistoricalDataset
from backtest.spot_reporting import build_spot_backtest_report, write_spot_backtest_artifacts
from backtest.spot_scenario import (
    SpotBacktestAsset,
    SpotBacktestExecutionConfig,
    SpotBacktestScenario,
    export_current_treasury_scenario,
    load_spot_backtest_scenario,
)
from bot.spot_strategy import ProgressiveSpotSwingExecutor
from shared.spot_progression import ProgressiveSwingConfig
from shared.spot_signal import SpotSignalConfig, evaluate_rolling_24h_opportunity
from shared.spot_sizing import IndicatorSizingConfig
from shared.spot_strategy import plan_spot_strategy_action


HOUR_MS = 60 * 60 * 1000
SYMBOL_INFO = {
    "filters": [
        {
            "filterType": "MARKET_LOT_SIZE",
            "minQty": "0.000001",
            "maxQty": "100000000",
            "stepSize": "0.000001",
        },
        {"filterType": "MIN_NOTIONAL", "minNotional": "0.01"},
    ]
}


def asset(
    symbol: str,
    *,
    quantity: float = 100,
    binance: float | None = None,
    cold: float = 0,
    target: float = 100,
    minimum: float = 80,
    objective: str | None = "accumulate_cash",
) -> SpotBacktestAsset:
    binance_quantity = quantity - cold if binance is None else binance
    return SpotBacktestAsset(
        symbol=symbol,
        quantity=quantity,
        entry_cost=100,
        binance_quantity=binance_quantity,
        cold_storage_quantity=cold,
        unassigned_quantity=quantity - binance_quantity - cold,
        target_holding=target,
        minimum_holding_pct=minimum,
        trading_objective=objective,
        symbol_info=SYMBOL_INFO,
    )


def scenario(
    assets: tuple[SpotBacktestAsset, ...],
    *,
    hours: int = 36,
    starting_usdt: float = 1000,
    fee_rate: float = 0,
    slippage_bps: float = 0,
) -> SpotBacktestScenario:
    start = datetime(2026, 1, 10, tzinfo=UTC)
    return SpotBacktestScenario(
        name="synthetic",
        start=start,
        end=start + timedelta(hours=hours),
        interval="1h",
        warmup_candles=60,
        starting_usdt=starting_usdt,
        assets=assets,
        execution=SpotBacktestExecutionConfig(
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            force_close_at_end=False,
        ),
        signal=SpotSignalConfig(),
        sizing=IndicatorSizingConfig(),
        progression=ProgressiveSwingConfig(close_profit_pct=5, estimated_fee_rate=fee_rate),
        data_cache_dir=Path("/tmp/spot-data"),
        output_dir=Path("/tmp/spot-output"),
    )


def dataset(
    config: SpotBacktestScenario,
    price_by_symbol,
    *,
    future_extremes: bool = False,
) -> SpotHistoricalDataset:
    candle_map = {}
    for item in config.assets:
        rows = []
        first_open = config.start - timedelta(hours=config.warmup_candles)
        count = config.warmup_candles + int((config.end - config.start).total_seconds() // 3600)
        for index in range(count):
            replay_index = index - config.warmup_candles
            price = float(price_by_symbol(item.symbol, replay_index))
            open_at = first_open + timedelta(hours=index)
            open_ms = int(open_at.timestamp() * 1000)
            high = price * (10 if future_extremes and replay_index > 4 else 1.002)
            low = price * (0.1 if future_extremes and replay_index > 4 else 0.998)
            rows.append(
                SpotCandle(
                    open_time_ms=open_ms,
                    close_time_ms=open_ms + HOUR_MS - 1,
                    open=price,
                    high=high,
                    low=low,
                    close=price,
                    volume=100,
                )
            )
        candle_map[item.symbol] = tuple(rows)
    return SpotHistoricalDataset(
        candles=candle_map,
        symbol_info={item.symbol: SYMBOL_INFO for item in config.assets},
        interval="1h",
        interval_ms=HOUR_MS,
        source="synthetic",
    )


class SpotBacktestFrameworkTests(unittest.TestCase):
    def run_scenario(self, config, prices, **dataset_kwargs):
        return SpotPortfolioBacktester(config, dataset(config, prices, **dataset_kwargs)).run()

    def test_live_and_backtest_reference_the_same_decision_function(self):
        self.assertIs(
            ProgressiveSpotSwingExecutor.handle_signal.__globals__["plan_spot_strategy_action"],
            plan_spot_strategy_action,
        )

    def test_repeated_run_is_deterministic(self):
        config = scenario((asset("AAA"),))
        prices = lambda _symbol, index: 111 if 0 <= index < 3 else 100

        first = self.run_scenario(config, prices)
        second = self.run_scenario(config, prices)

        self.assertEqual(first.equity, second.equity)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.swings, second.swings)

    def test_warmup_signals_cannot_trade(self):
        config = scenario((asset("AAA"),), hours=8)
        prices = lambda _symbol, index: 130 if index < -24 else 100

        result = self.run_scenario(config, prices)

        self.assertEqual(result.actions, [])
        self.assertEqual(result.swings, [])

    def test_future_candle_extremes_do_not_change_decisions(self):
        config = scenario((asset("AAA"),))
        prices = lambda _symbol, index: 111 if 0 <= index < 3 else 100

        normal = self.run_scenario(config, prices)
        extreme = self.run_scenario(config, prices, future_extremes=True)

        self.assertEqual(normal.actions, extreme.actions)
        self.assertEqual(normal.equity, extreme.equity)

    def test_incomplete_historical_range_fails_instead_of_shortening_replay(self):
        config = scenario((asset("AAA"),), hours=6)
        historical = dataset(config, lambda _symbol, _index: 100)
        historical.candles["AAA"] = historical.candles["AAA"][:-1]

        with self.assertRaisesRegex(ValueError, "complete requested replay range"):
            SpotPortfolioBacktester(config, historical)

    def test_rolling_signal_is_the_live_domain_result(self):
        config = scenario((asset("AAA"),), hours=4)
        replay = SpotPortfolioBacktester(config, dataset(config, lambda _symbol, index: 108 if index >= 0 else 100))
        candle = replay._replay_rows()["AAA"][0]

        actual = replay._signal_for("AAA", candle)
        expected = evaluate_rolling_24h_opportunity(
            current_price=108,
            reference_price=100,
            current_at_ms=candle.close_time_ms,
            reference_at_ms=candle.close_time_ms - 24 * HOUR_MS,
        )

        self.assertEqual(actual["opportunity"], expected["opportunity"])
        self.assertEqual(actual["level"], expected["level"])
        self.assertEqual(actual["base_tranche_pct"], expected["base_tranche_pct"])

    def test_shared_usdt_contention_is_deterministic_by_asset_order(self):
        config = scenario(
            (asset("BBB", quantity=10, target=10), asset("AAA", quantity=10, target=10)),
            hours=6,
            starting_usdt=100,
        )
        result = self.run_scenario(config, lambda _symbol, index: 90 if index >= 0 else 100)
        buys = [item for item in result.actions if item["side"] == "buy"]

        self.assertEqual(config.asset_order, ("AAA", "BBB"))
        self.assertEqual([item["asset_symbol"] for item in buys[:2]], ["AAA", "BBB"])
        self.assertLess(buys[1]["executed_quantity"], buys[0]["executed_quantity"])
        self.assertGreaterEqual(result.final_usdt, -1e-8)

    def test_cold_storage_and_full_floor_cannot_be_sold(self):
        config = scenario(
            (
                asset("COLD", binance=0, cold=100, minimum=0),
                asset("LOCK", minimum=100),
            ),
            hours=6,
        )
        result = self.run_scenario(config, lambda _symbol, index: 110 if index >= 0 else 100)

        self.assertEqual(result.actions, [])
        self.assertEqual(result.assets["COLD"].cold_storage_quantity, 100)

    def test_unset_objective_is_valued_but_does_not_trade(self):
        config = scenario((asset("AAA", objective=None),), hours=6)
        result = self.run_scenario(config, lambda _symbol, index: 120 if index >= 0 else 100)

        self.assertEqual(result.actions, [])
        self.assertEqual(result.equity[-1]["treasury_value"], 12000 + config.starting_usdt)

    def test_progressive_levels_open_independent_swings_without_duplicates(self):
        config = scenario((asset("AAA", minimum=0),), hours=8)

        def prices(_symbol, index):
            if index < 0:
                return 100
            return 106 if index < 2 else 109

        result = self.run_scenario(config, prices)
        opens = [item for item in result.actions if item["action_type"] == "open"]

        self.assertEqual([item["signal_level"] for item in opens], [1, 2])
        self.assertEqual(len({item["swing_id"] for item in opens}), 2)

    def test_profitable_close_has_priority_and_open_swing_is_not_forced_closed(self):
        config = scenario((asset("AAA", minimum=0),), hours=8)
        prices = lambda _symbol, index: 111 if 0 <= index < 3 else 100

        result = self.run_scenario(config, prices)

        self.assertEqual(result.actions[0]["action_type"], "open")
        self.assertEqual(result.actions[1]["action_type"], "close")
        self.assertLessEqual(
            max(
                sum(1 for item in result.actions if item["asset_symbol"] == "AAA" and item["timestamp_ms"] == ts)
                for ts in {item["timestamp_ms"] for item in result.actions}
            ),
            1,
        )

        falling = self.run_scenario(
            scenario((asset("BBB"),), hours=6),
            lambda _symbol, index: 90 if index >= 0 else 100,
        )
        self.assertEqual(falling.swings[0]["status"], "open")

    def test_accumulate_cash_and_accumulate_asset_settlement(self):
        cash_config = scenario((asset("CASH", minimum=0),), hours=8)
        prices = lambda _symbol, index: 111 if 0 <= index < 3 else 100
        cash_result = self.run_scenario(cash_config, prices)
        cash_report = build_spot_backtest_report(cash_result)

        self.assertGreater(
            cash_report["per_asset"]["CASH"]["objective_metrics"]["realized_quote_cash_generated"],
            0,
        )
        self.assertEqual(cash_result.assets["CASH"].target_quantity, 100)

        asset_config = scenario(
            (asset("COIN", minimum=0, objective="accumulate_asset"),),
            hours=8,
        )
        asset_result = self.run_scenario(asset_config, prices)

        self.assertGreater(asset_result.assets["COIN"].target_quantity, 100)
        self.assertEqual(len(asset_result.target_history), 1)
        self.assertEqual(len({item["swing_id"] for item in asset_result.target_history}), 1)

    def test_losing_asset_swing_does_not_ratchet(self):
        config = scenario(
            (asset("AAA", minimum=0, objective="accumulate_asset"),),
            hours=6,
        )
        result = self.run_scenario(config, lambda _symbol, index: 111 if index >= 0 else 100)

        self.assertEqual(result.target_history, [])
        self.assertEqual(result.assets["AAA"].target_quantity, 100)

    def test_hodl_benchmark_matches_no_trade_starting_state(self):
        config = scenario((asset("AAA", objective=None),), hours=6, starting_usdt=500)
        result = self.run_scenario(config, lambda _symbol, index: 120 if index >= 0 else 100)
        report = build_spot_backtest_report(result)

        self.assertEqual(
            report["portfolio"]["final_treasury_value"],
            report["portfolio"]["hodl_final_treasury_value"],
        )
        self.assertEqual(report["portfolio"]["difference_vs_hodl"], 0)

    def test_fee_slippage_and_artifacts_are_explicit(self):
        config = scenario(
            (asset("AAA", quantity=10, target=10),),
            hours=6,
            starting_usdt=1000,
            fee_rate=0.001,
            slippage_bps=100,
        )
        result = self.run_scenario(config, lambda _symbol, index: 89 if index >= 0 else 100)

        execution = result.executions[0]
        self.assertAlmostEqual(execution["price"], 89.89)
        self.assertAlmostEqual(execution["fee_amount"], execution["quote_quantity"] * 0.001)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_spot_backtest_artifacts(result, tmp)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "equity.csv").exists())
            self.assertTrue((Path(tmp) / "swings.json").exists())
            self.assertTrue((Path(tmp) / "scenario.resolved.yaml").exists())
            self.assertEqual(artifacts["report"]["scenario"]["asset_order"], ["AAA"])
            self.assertIn("shared_usdt_reserved", result.equity[-1])
            self.assertIn(
                "maximum_simultaneously_underwater_sell_origin_swings",
                artifacts["report"]["bad_cases"],
            )

    def test_template_and_exported_current_treasury_are_reviewable_scenarios(self):
        template = load_spot_backtest_scenario("config/spot_backtest.template.yaml")
        self.assertEqual(len(template.assets), 10)
        self.assertTrue(all(item.quantity == 0 for item in template.assets))

        snapshot = {
            "holdings": [
                {
                    "asset_symbol": "USDT",
                    "quantity": 500,
                    "is_dry_powder": True,
                },
                {
                    "asset_symbol": "USDC",
                    "quantity": 25,
                    "is_dry_powder": True,
                },
                {
                    "asset_symbol": "NEAR",
                    "quantity": 1000,
                    "average_cost": 1.997,
                    "target_quantity": 1000,
                    "minimum_holding_pct": 80,
                    "trading_objective": "accumulate_asset",
                    "custody": {
                        "binance": {"quantity": 750},
                        "cold_storage": {"quantity": 250},
                        "unassigned": {"quantity": 0},
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "current.yaml"
            with patch(
                "api.services.portfolio_service.load_portfolio_snapshot",
                return_value=(snapshot, ["review me"]),
            ):
                export_current_treasury_scenario(
                    path,
                    start="2025-09-23",
                    end="2026-09-23",
                )
            exported = load_spot_backtest_scenario(path)

        self.assertEqual(exported.starting_usdt, 500)
        self.assertEqual(exported.assets[0].symbol, "NEAR")
        self.assertEqual(exported.assets[0].binance_quantity, 750)
        self.assertEqual(exported.assets[0].cold_storage_quantity, 250)
        self.assertEqual(exported.metadata["export_warnings"][0], "review me")
        self.assertIn("Excluded USDC", exported.metadata["export_warnings"][1])

    def test_scenario_rejects_inconsistent_custody(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.yaml"
            path.write_text(
                """
spot_backtest:
  start: 2026-01-01
  end: 2026-02-01
  assets:
    BTC:
      quantity: 2
      custody: {binance: 1, cold_storage: 0}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "custody totals"):
                load_spot_backtest_scenario(path)


if __name__ == "__main__":
    unittest.main()
