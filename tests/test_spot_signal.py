import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.spot_signal import (
    RollingSpotSignalMonitor,
    calculate_spot_indicator_context,
    spot_indicator_sizing_config_from_env,
    spot_signal_config_from_env,
)
from shared.runtime_db import (
    bootstrap_runtime_db,
    load_spot_signal_snapshot,
    upsert_portfolio_asset_policy,
)
from shared.spot_signal import ROLLING_WINDOW_MS, SpotSignalConfig, evaluate_rolling_24h_opportunity
from shared.spot_sizing import IndicatorSizingConfig, apply_indicator_sizing


class FakeTickerClient:
    def __init__(self, tickers, klines=None):
        self.tickers = tickers
        self.klines = klines or {}
        self.calls = []
        self.kline_calls = []

    def get_ticker(self, **kwargs):
        self.calls.append(kwargs)
        return self.tickers[kwargs["symbol"]]

    def get_klines(self, **kwargs):
        self.kline_calls.append(kwargs)
        symbol = kwargs["symbol"]
        if symbol in self.klines:
            return self.klines[symbol]
        return spot_klines(float(self.tickers[symbol]["openPrice"]))


def rolling_ticker(open_price: float, last_price: float) -> dict:
    return {
        "openPrice": str(open_price),
        "lastPrice": str(last_price),
        "openTime": 1_000,
        "closeTime": 1_000 + ROLLING_WINDOW_MS,
    }


def spot_klines(price: float, *, count: int = 60) -> list[list]:
    close_at_ms = 1_000 + ROLLING_WINDOW_MS
    interval_ms = 60 * 60 * 1000
    rows = []
    for index in range(count):
        open_time = close_at_ms - ((count - index) * interval_ms)
        close_time = open_time + interval_ms - 1
        rows.append(
            [
                open_time,
                str(price),
                str(price * 1.002),
                str(price * 0.998),
                str(price),
                "100",
                close_time,
            ]
        )
    return rows


class SpotSignalDomainTests(unittest.TestCase):
    def evaluate(self, current_price: float) -> dict:
        return evaluate_rolling_24h_opportunity(
            current_price=current_price,
            reference_price=100,
            current_at_ms=1_000 + ROLLING_WINDOW_MS,
            reference_at_ms=1_000,
        )

    def test_default_levels_create_progressive_sell_opportunities(self):
        expectations = [(104.9, 0, 0), (105, 1, 10), (108, 2, 20), (112, 3, 30), (118, 4, 40)]

        for price, level, tranche in expectations:
            with self.subTest(price=price):
                result = self.evaluate(price)
                self.assertEqual(result["opportunity"], "hold" if level == 0 else "sell")
                self.assertEqual(result["level"], level)
                self.assertEqual(result["base_tranche_pct"], tranche)

    def test_negative_move_creates_buy_opportunity(self):
        result = self.evaluate(88)

        self.assertEqual(result["opportunity"], "buy")
        self.assertEqual(result["level"], 3)
        self.assertEqual(result["base_tranche_pct"], 30)
        self.assertAlmostEqual(result["rolling_change_pct"], -12)

    def test_reference_must_be_approximately_24_hours_old(self):
        with self.assertRaisesRegex(ValueError, "approximately 24 hours"):
            evaluate_rolling_24h_opportunity(
                current_price=110,
                reference_price=100,
                current_at_ms=1_000 + 60 * 60 * 1000,
                reference_at_ms=1_000,
            )

    def test_config_rejects_misaligned_or_unsorted_levels(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            SpotSignalConfig(levels_pct=(5, 8), base_tranches_pct=(10,))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SpotSignalConfig(levels_pct=(8, 5), base_tranches_pct=(10, 20))

    def test_config_is_loaded_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "SCROOGE_SPOT_SIGNAL_LEVELS_PCT": "4,7",
                "SCROOGE_SPOT_SIGNAL_BASE_TRANCHES_PCT": "15,35",
            },
        ):
            config = spot_signal_config_from_env()

        self.assertEqual(config.levels_pct, (4.0, 7.0))
        self.assertEqual(config.base_tranches_pct, (15.0, 35.0))


class SpotIndicatorSizingTests(unittest.TestCase):
    def signal(self, opportunity: str, *, base_tranche_pct: float = 20, current_price: float = 110) -> dict:
        return {
            "opportunity": opportunity,
            "level": 2 if opportunity != "hold" else 0,
            "base_tranche_pct": base_tranche_pct if opportunity != "hold" else 0,
            "current_price": current_price,
        }

    def test_indicators_cannot_turn_hold_into_buy_or_sell(self):
        result = apply_indicator_sizing(
            self.signal("hold"),
            {"rsi": 99, "bb_lower": 80, "bb_upper": 90, "ema": 90, "atr": 4},
        )

        self.assertEqual(result["opportunity"], "hold")
        self.assertEqual(result["indicator_status"], "not_applicable")
        self.assertEqual(result["final_tranche_pct"], 0)

    def test_three_directional_confirmations_apply_very_strong_modifier(self):
        result = apply_indicator_sizing(
            self.signal("sell"),
            {"rsi": 78, "bb_lower": 90, "bb_upper": 105, "ema": 100, "atr": 4},
        )

        self.assertEqual(result["opportunity"], "sell")
        self.assertEqual(result["indicator_assessment"]["tier"], "very_strong")
        self.assertEqual(result["sizing_modifier"], 1.5)
        self.assertEqual(result["final_tranche_pct"], 30)

    def test_conflicting_context_reduces_but_does_not_reverse_signal(self):
        result = apply_indicator_sizing(
            self.signal("sell"),
            {"rsi": 20, "bb_lower": 115, "bb_upper": 125, "ema": 120, "atr": 4},
        )

        self.assertEqual(result["opportunity"], "sell")
        self.assertEqual(result["indicator_assessment"]["tier"], "weak")
        self.assertEqual(result["sizing_modifier"], 0.5)
        self.assertEqual(result["final_tranche_pct"], 10)

    def test_two_confirmations_without_conflicts_apply_strong_modifier(self):
        result = apply_indicator_sizing(
            self.signal("sell"),
            {"rsi": 75, "bb_lower": 90, "bb_upper": 120, "ema": 100, "atr": 4},
        )

        self.assertEqual(result["indicator_assessment"]["tier"], "strong")
        self.assertEqual(result["sizing_modifier"], 1.25)
        self.assertEqual(result["final_tranche_pct"], 25)

    def test_one_confirmation_uses_neutral_modifier(self):
        result = apply_indicator_sizing(
            self.signal("sell"),
            {"rsi": 50, "bb_lower": 90, "bb_upper": 120, "ema": 100, "atr": 4},
        )

        self.assertEqual(result["indicator_assessment"]["tier"], "neutral")
        self.assertEqual(result["sizing_modifier"], 1)
        self.assertEqual(result["final_tranche_pct"], 20)

    def test_missing_context_uses_conservative_modifier(self):
        result = apply_indicator_sizing(
            self.signal("buy"),
            None,
            indicator_error="klines unavailable",
        )

        self.assertEqual(result["opportunity"], "buy")
        self.assertEqual(result["indicator_status"], "unavailable")
        self.assertEqual(result["sizing_modifier"], 0.5)
        self.assertEqual(result["indicator_error"], "klines unavailable")

    def test_custom_modifiers_are_loaded_from_environment(self):
        with patch.dict(os.environ, {"SCROOGE_SPOT_INDICATOR_SIZING_MODIFIERS": "0.4,0.9,1.2,1.4"}):
            config = spot_indicator_sizing_config_from_env()

        self.assertEqual(
            (
                config.weak_modifier,
                config.neutral_modifier,
                config.strong_modifier,
                config.very_strong_modifier,
            ),
            (0.4, 0.9, 1.2, 1.4),
        )

    def test_indicator_context_uses_only_closed_candles(self):
        rows = spot_klines(100)
        current_at_ms = 1_000 + ROLLING_WINDOW_MS
        rows.append(
            [current_at_ms, "500", "500", "500", "500", "1", current_at_ms + 3_599_999]
        )

        context = calculate_spot_indicator_context(
            rows,
            current_price=110,
            evaluated_at_ms=current_at_ms,
            interval="1h",
        )

        self.assertEqual(context["candle_count"], 60)
        self.assertEqual(context["latest_closed_price"], 100)
        self.assertIn("atr_pct", context)

    def test_modifier_order_is_validated(self):
        with self.assertRaisesRegex(ValueError, "ordered"):
            IndicatorSizingConfig(
                weak_modifier=1,
                neutral_modifier=0.5,
                strong_modifier=1.25,
                very_strong_modifier=1.5,
            )

    def test_buy_or_sell_requires_positive_base_tranche(self):
        with self.assertRaisesRegex(ValueError, "positive base tranche"):
            apply_indicator_sizing(
                self.signal("buy", base_tranche_pct=0),
                {"rsi": 20, "bb_lower": 100, "bb_upper": 120, "ema": 115},
            )


class RollingSpotSignalMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.sqlite3"
        bootstrap_runtime_db(self.db_path)

    def policy(self, asset: str, *, objective: str | None, minimum: float = 80) -> None:
        upsert_portfolio_asset_policy(
            {
                "asset_symbol": asset,
                "quote_symbol": "USDT",
                "target_quantity": 100,
                "minimum_holding_pct": minimum,
                "trading_objective": objective,
            },
            path=self.db_path,
        )

    def monitor(self, client, *, execution_enabled: bool = True) -> RollingSpotSignalMonitor:
        return RollingSpotSignalMonitor(
            client,
            interval_seconds=300,
            execution_enabled=execution_enabled,
            logger=logging.getLogger("test.spot-signal"),
            db_path=self.db_path,
        )

    def test_monitor_uses_cash_objective_when_policy_does_not_specify_one(self):
        self.policy("NEAR", objective="accumulate_cash")
        self.policy("XRP", objective=None)
        client = FakeTickerClient(
            {
                "NEARUSDT": rolling_ticker(4, 4.4),
                "XRPUSDT": rolling_ticker(1, 0.9),
            }
        )

        results = self.monitor(client).refresh_once()
        near = load_spot_signal_snapshot("NEAR", path=self.db_path)
        xrp = load_spot_signal_snapshot("XRP", path=self.db_path)

        self.assertEqual(len(results), 2)
        self.assertEqual(near["opportunity"], "sell")
        self.assertEqual(near["level"], 2)
        self.assertTrue(near["strategy_eligible"])
        self.assertEqual(near["eligibility_reason"], "eligible")
        self.assertEqual(xrp["opportunity"], "buy")
        self.assertEqual(xrp["trading_objective"], "accumulate_cash")
        self.assertTrue(xrp["strategy_eligible"])
        self.assertEqual(xrp["eligibility_reason"], "eligible")

    def test_monitor_orders_complete_signal_batch_before_execution(self):
        self.policy("NEAR", objective="accumulate_cash")
        self.policy("XRP", objective="accumulate_cash")
        client = FakeTickerClient(
            {
                "NEARUSDT": rolling_ticker(4, 4.4),
                "XRPUSDT": rolling_ticker(1, 0.9),
            }
        )
        handled: list[str] = []
        monitor = RollingSpotSignalMonitor(
            client,
            interval_seconds=300,
            execution_enabled=True,
            logger=logging.getLogger("test.spot-signal"),
            db_path=self.db_path,
            snapshot_handler=lambda signal: handled.append(signal["asset_symbol"]),
            snapshot_orderer=lambda signals: list(reversed(signals)),
        )

        results = monitor.refresh_once()

        self.assertEqual([item["asset_symbol"] for item in results], ["NEAR", "XRP"])
        self.assertEqual(handled, ["XRP", "NEAR"])

    def test_fully_protected_policy_is_not_strategy_eligible(self):
        self.policy("NEAR", objective="accumulate_asset", minimum=100)
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.8)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["opportunity"], "sell")
        self.assertFalse(snapshot["strategy_eligible"])
        self.assertEqual(snapshot["eligibility_reason"], "fully_protected")

    def test_disabled_monitor_does_not_call_binance(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})

        results = self.monitor(client, execution_enabled=False).refresh_once()

        self.assertEqual(results, [])
        self.assertEqual(client.calls, [])
        self.assertIsNone(load_spot_signal_snapshot("NEAR", path=self.db_path))

    def test_market_error_preserves_last_signal_but_disables_eligibility(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})
        monitor = self.monitor(client)
        monitor.refresh_once()

        with patch("bot.spot_signal.run_binance_with_retries", side_effect=RuntimeError("offline")):
            self.assertEqual(monitor.refresh_once(), [])
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["opportunity"], "sell")
        self.assertFalse(snapshot["strategy_eligible"])
        self.assertEqual(snapshot["eligibility_reason"], "signal_unavailable")
        self.assertEqual(snapshot["error"], "offline")

    def test_sizing_context_is_persisted_for_valid_opportunity(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.4)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["indicator_status"], "ok")
        self.assertEqual(snapshot["indicator_context"]["interval"], "1h")
        self.assertEqual(snapshot["indicator_assessment"]["tier"], "strong")
        self.assertEqual(snapshot["sizing_modifier"], 1.25)
        self.assertEqual(snapshot["final_tranche_pct"], 25)
        self.assertEqual(client.kline_calls[0]["limit"], 60)

    def test_hold_does_not_request_indicator_klines(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient({"NEARUSDT": rolling_ticker(4, 4.1)})

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["opportunity"], "hold")
        self.assertEqual(snapshot["indicator_status"], "not_applicable")
        self.assertEqual(snapshot["final_tranche_pct"], 0)
        self.assertEqual(client.kline_calls, [])

    def test_incomplete_indicator_history_reduces_size_without_invalidating_signal(self):
        self.policy("NEAR", objective="accumulate_cash")
        client = FakeTickerClient(
            {"NEARUSDT": rolling_ticker(4, 4.4)},
            {"NEARUSDT": spot_klines(4, count=10)},
        )

        self.monitor(client).refresh_once()
        snapshot = load_spot_signal_snapshot("NEAR", path=self.db_path)

        self.assertEqual(snapshot["status"], "ok")
        self.assertEqual(snapshot["opportunity"], "sell")
        self.assertTrue(snapshot["strategy_eligible"])
        self.assertEqual(snapshot["indicator_status"], "unavailable")
        self.assertEqual(snapshot["sizing_modifier"], 0.5)
        self.assertEqual(snapshot["final_tranche_pct"], 10)


if __name__ == "__main__":
    unittest.main()
