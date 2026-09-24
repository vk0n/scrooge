from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backtest.spot_bargain_analysis import build_bargain_analysis
from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import (
    BinanceSpotHistoricalAdapter,
    SpotCandle,
    SpotHistoricalDataset,
)
from backtest.spot_reporting import build_spot_backtest_report, write_spot_backtest_artifacts
from backtest.spot_report_html import display_spot_report_title
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
    unavailable_symbols: frozenset[str] = frozenset(),
) -> SpotHistoricalDataset:
    candle_map = {}
    unavailable_open_times: dict[str, frozenset[int]] = {}
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
        if item.symbol in unavailable_symbols:
            unavailable_open_times[item.symbol] = frozenset(
                row.open_time_ms for row in rows if row.open_time_ms >= int(config.start.timestamp() * 1000)
            )
    return SpotHistoricalDataset(
        candles=candle_map,
        symbol_info={item.symbol: SYMBOL_INFO for item in config.assets},
        interval="1h",
        interval_ms=HOUR_MS,
        source="synthetic",
        unavailable_open_times=unavailable_open_times,
    )


class SpotBacktestFrameworkTests(unittest.TestCase):
    def run_scenario(self, config, prices, **dataset_kwargs):
        return SpotPortfolioBacktester(config, dataset(config, prices, **dataset_kwargs)).run()

    def test_live_and_backtest_reference_the_same_decision_function(self):
        self.assertIs(
            ProgressiveSpotSwingExecutor.handle_signal.__globals__["plan_spot_strategy_action"],
            plan_spot_strategy_action,
        )

    def test_report_title_uses_replay_period_instead_of_market_migration_details(self):
        legacy_name = "treasury-10-assets-real-quantities-6m-ton-to-gram"

        self.assertEqual(
            display_spot_report_title(
                legacy_name,
                "2026-03-25T00:00:00+00:00",
                "2026-09-24T00:00:00+00:00",
            ),
            "Scrooge Treasury: Six-Month Portfolio Replay",
        )
        self.assertEqual(
            display_spot_report_title(
                legacy_name,
                "2025-09-24T00:00:00+00:00",
                "2026-09-24T00:00:00+00:00",
            ),
            "Scrooge Treasury: One-Year Portfolio Replay",
        )

    def test_repeated_run_is_deterministic(self):
        config = scenario((asset("AAA"),))
        prices = lambda _symbol, index: 111 if 0 <= index < 3 else 100

        first = self.run_scenario(config, prices)
        second = self.run_scenario(config, prices)

        self.assertEqual(first.equity, second.equity)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.swings, second.swings)

    def test_bargain_analysis_preserves_partial_pnl_and_external_fees(self):
        swings = [
            {
                "swing_id": "closed",
                "asset_symbol": "AAA",
                "status": "closed",
                "origin_side": "sell",
                "trading_objective": "accumulate_cash",
                "quote_symbol": "USDT",
                "age_seconds": 48 * 3600,
                "strategy_reason": {
                    "signal_level": 2,
                    "indicator_assessment": {"tier": "strong"},
                },
                "executions": [{}, {}],
                "economics": {
                    "opening_quote_quantity": 100,
                    "realized_pnl_quote": 10,
                    "unrealized_pnl_quote": 0,
                    "fees_by_asset": {"USDT": 1},
                },
            },
            {
                "swing_id": "partial",
                "asset_symbol": "AAA",
                "status": "partially_closed",
                "origin_side": "buy",
                "trading_objective": "accumulate_asset",
                "quote_symbol": "USDT",
                "age_seconds": 10 * 86400,
                "strategy_reason": {
                    "signal_level": 1,
                    "indicator_assessment": {"tier": "neutral"},
                },
                "executions": [{}],
                "economics": {
                    "opening_quote_quantity": 50,
                    "realized_pnl_quote": 2,
                    "unrealized_pnl_quote": -3,
                    "fees_by_asset": {"BNB": 0.01},
                },
            },
        ]

        analysis = build_bargain_analysis(swings)

        self.assertEqual(analysis["overview"]["closed"], 1)
        self.assertEqual(analysis["overview"]["partially_closed"], 1)
        self.assertEqual(analysis["overview"]["realized_pnl_quote"], 12)
        self.assertEqual(analysis["overview"]["unrealized_pnl_quote"], -3)
        self.assertEqual(analysis["overview"]["net_pnl_quote"], 9)
        self.assertEqual(analysis["overview"]["quote_fees"], 1)
        self.assertEqual(analysis["overview"]["fees_by_asset"]["BNB"], 0.01)
        self.assertEqual(analysis["risk"]["underwater_open_count"], 1)
        self.assertEqual(len(analysis["breakdowns"]["signal_level"]), 2)

    def test_warmup_signals_cannot_trade(self):
        config = scenario((asset("AAA"),), hours=8)
        prices = lambda _symbol, index: 130 if index < -24 else 100

        result = self.run_scenario(config, prices)

        self.assertEqual(result.actions, [])
        self.assertEqual(result.swings, [])

    def test_declared_market_migration_gap_cannot_signal_or_trade(self):
        config = scenario((asset("AAA"),), hours=8)
        historical = dataset(
            config,
            lambda _symbol, index: 110 if index >= 0 else 100,
            unavailable_symbols=frozenset({"AAA"}),
        )

        result = SpotPortfolioBacktester(config, historical).run()

        self.assertEqual(result.signals, [])
        self.assertEqual(result.actions, [])
        self.assertEqual(result.swings, [])

    def test_partial_candle_cache_downloads_and_persists_missing_tail(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=3)
        start_ms = int(start.timestamp() * 1000)

        def candle(index: int) -> SpotCandle:
            open_ms = start_ms + index * HOUR_MS
            return SpotCandle(
                open_time_ms=open_ms,
                close_time_ms=open_ms + HOUR_MS - 1,
                open=100 + index,
                high=101 + index,
                low=99 + index,
                close=100 + index,
                volume=10,
            )

        with tempfile.TemporaryDirectory() as tmp:
            adapter = BinanceSpotHistoricalAdapter(tmp)
            cache_path = Path(tmp) / "klines" / f"AAAUSDT-1h-{start_ms}-{int(end.timestamp() * 1000)}.csv"
            cache_path.parent.mkdir(parents=True)
            adapter._write_candles(cache_path, [candle(0), candle(1)])

            with patch.object(adapter, "_download_candles", return_value=[candle(2)]) as download:
                rows, source = adapter.load_candles(
                    "AAAUSDT",
                    interval="1h",
                    start=start,
                    end=end,
                )

            self.assertEqual([row.open_time_ms for row in rows], [candle(i).open_time_ms for i in range(3)])
            self.assertEqual(source, "binance_spot_rest")
            self.assertEqual(len(adapter._read_candles(cache_path)), 3)
            download.assert_called_once_with(
                "AAAUSDT",
                interval="1h",
                start_ms=candle(2).open_time_ms,
                end_ms=int(end.timestamp() * 1000),
            )

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
        cash_performance = cash_report["per_asset"]["CASH"]["capital_performance"]
        self.assertGreater(cash_performance["accumulated_cash_gain"], 0)
        self.assertLess(cash_performance["effective_entry_cost"], 100)
        self.assertAlmostEqual(
            cash_performance["total_gain"],
            cash_performance["market_gain"]
            + cash_performance["accumulated_cash_gain"]
            + cash_performance["open_bargain_pnl"],
        )
        self.assertEqual(cash_result.assets["CASH"].target_quantity, 100)

        asset_config = scenario(
            (asset("COIN", minimum=0, objective="accumulate_asset"),),
            hours=8,
        )
        asset_result = self.run_scenario(asset_config, prices)
        asset_report = build_spot_backtest_report(asset_result)
        asset_performance = asset_report["per_asset"]["COIN"]["capital_performance"]

        self.assertGreater(asset_result.assets["COIN"].target_quantity, 100)
        self.assertGreater(asset_performance["accumulated_asset_quantity"], 0)
        self.assertGreater(asset_performance["settled_quantity"], 100)
        self.assertLess(asset_performance["effective_entry_cost"], 100)
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
        performance = report["per_asset"]["AAA"]["capital_performance"]
        self.assertEqual(performance["initial_capital"], 10000)
        self.assertEqual(performance["effective_entry_cost"], 100)
        self.assertEqual(performance["market_gain"], 2000)
        self.assertEqual(performance["accumulated_cash_gain"], 0)
        self.assertEqual(performance["open_bargain_pnl"], 0)
        self.assertEqual(performance["total_gain"], 2000)
        self.assertEqual(report["portfolio"]["initial_invested_capital"], 10500)
        self.assertEqual(report["portfolio"]["total_gain_on_initial_capital"], 2000)

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
            self.assertTrue((Path(tmp) / "waiter_cleanup.json").exists())
            self.assertTrue((Path(tmp) / "waiter_cleanup_reasons.csv").exists())
            self.assertTrue((Path(tmp) / "scenario.resolved.yaml").exists())
            report_html = (Path(tmp) / "report.html").read_text(encoding="utf-8")
            self.assertIn("Scrooge Research", report_html)
            self.assertIn("Vault Value", report_html)
            self.assertIn("Final Allocation", report_html)
            self.assertIn("Bargain History", report_html)
            self.assertIn("Bargain Analytics", report_html)
            self.assertIn("Waiter Cleanup", report_html)
            self.assertIn("Entry Cost", report_html)
            self.assertIn("Total Gain", report_html)
            self.assertIn("<span>Closed</span><span>Open</span>", report_html)
            self.assertIn('"swingHistory"', report_html)
            self.assertIn('"bargainAnalysis"', report_html)
            self.assertIn(result.swings[0]["swing_id"], report_html)
            self.assertEqual(artifacts["report"]["scenario"]["asset_order"], ["AAA"])
            self.assertIn("shared_usdt_reserved", result.equity[-1])
            self.assertIn(
                "maximum_simultaneously_underwater_sell_origin_swings",
                artifacts["report"]["bad_cases"],
            )
            self.assertIn("breakdowns", artifacts["report"]["bargain_analysis"])

    def test_cleanup_close_uses_simulated_executor_and_persists_reason_and_fee(self):
        config = scenario(
            (asset("AAA", minimum=0),),
            hours=6,
            starting_usdt=5000,
            fee_rate=0.001,
        )
        engine = SpotPortfolioBacktester(config, dataset(config, lambda _symbol, _index: 80))
        engine.swings["waiter"] = {
            "swing_id": "waiter",
            "account_key": "spot_backtest",
            "asset_symbol": "AAA",
            "quote_symbol": "USDT",
            "origin_side": "buy",
            "trading_objective": "accumulate_cash",
            "status": "open",
            "source": "strategy",
            "opened_at_ms": engine.start_ms - 30 * 24 * HOUR_MS,
            "closed_at_ms": None,
            "executions": [
                {
                    "execution_id": "waiter-open",
                    "side": "buy",
                    "quantity": 10,
                    "price": 100,
                    "quote_quantity": 1000,
                    "fee_amount": 0,
                    "fee_asset": "USDT",
                }
            ],
        }
        reason = {
            "action_type": "close",
            "close_reason": "age_l3_cleanup",
            "required_cleanup_level": 3,
            "actual_reverse_signal_level": 4,
            "unrealized_pnl_before_cleanup": -200,
            "unrealized_pnl_pct_before_cleanup": -20,
            "capacity_pressure": False,
        }

        engine._execute_action(
            "AAA",
            {
                "action_type": "close",
                "side": "sell",
                "swing_id": "waiter",
                "requested_quantity": 10,
                "reason": reason,
            },
            observed_price=80,
            timestamp_ms=engine.start_ms,
        )

        closed = engine.swings["waiter"]
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["close_reason"], "age_l3_cleanup")
        self.assertEqual(closed["close_context"], reason)
        self.assertAlmostEqual(closed["executions"][-1]["fee_amount"], 0.8)
        self.assertEqual(closed["executions"][-1]["reason"]["close_reason"], "age_l3_cleanup")

    def test_template_and_exported_current_treasury_are_reviewable_scenarios(self):
        template = load_spot_backtest_scenario("config/spot_backtest.template.yaml")
        self.assertEqual(len(template.assets), 10)
        self.assertTrue(all(item.quantity == 0 for item in template.assets))
        self.assertTrue(template.waiter_cleanup.enabled)

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

    def test_scenario_preserves_explicit_history_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.yaml"
            path.write_text(
                """
spot_backtest:
  start: 2026-03-25
  end: 2026-09-24
  assets:
    GRAM:
      quantity: 1000
      custody: {binance: 1000}
      history_segments:
        - market_symbol: TONUSDT
          end: 2026-06-30T03:00:00Z
        - market_symbol: GRAMUSDT
          start: 2026-07-02T08:00:00Z
""",
                encoding="utf-8",
            )

            loaded = load_spot_backtest_scenario(path)

        self.assertEqual([item.market_symbol for item in loaded.assets[0].history_segments], ["TONUSDT", "GRAMUSDT"])
        self.assertEqual(loaded.assets[0].history_segments[0].end, datetime(2026, 6, 30, 3, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
