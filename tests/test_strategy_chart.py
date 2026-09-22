import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from bot.strategy_chart import StrategyChartRecorder
from core.engine import DiscreteRowSnapshot, RealtimeStrategyProcessor, _build_realtime_snapshot
from core.feature_engine import EMAFeature
from core.indicator_inputs import indicator_selection_plan
from services import chart_service
from shared.runtime_db import list_strategy_chart_snapshots


def snapshot(ts="2026-09-01 11:00:10", *, ema=77_410.0):
    return DiscreteRowSnapshot(
        raw_row=None, price=78_000.0, lower=78_800.0, upper=79_200.0,
        mid=79_000.0, atr=100.0, rsi=35.0, ema=ema, row_ts=ts, log_ts=ts,
    )


class StrategyChartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "runtime.sqlite3"
        self.env = patch.dict(os.environ, {"SCROOGE_DB_PATH": str(self.db)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.recorder = StrategyChartRecorder("BTCUSDT")
        self.runtime = SimpleNamespace(live=False, position=None, balance=100, balance_history=[])

    def rows(self, symbol="BTCUSDT"):
        return list_strategy_chart_snapshots(symbol, 0, 9_999_999_999_999)

    def test_chart_uses_continuous_ema_and_keeps_entry_before_later_sample(self):
        prices = [65_000.0] * 100 + [79_000.0] * 54
        strategy = EMAFeature(50)
        strategy.bootstrap(prices)
        old_chart = EMAFeature(50)
        old_chart.bootstrap(prices[-54:])
        actual = snapshot(ema=strategy.intrabar_value(78_000.0))
        self.assertLess(actual.ema, actual.price)
        self.assertGreater(old_chart.intrabar_value(actual.price), actual.price)

        def open_trade(row, runtime):
            runtime.position = {"decision_time": row.row_ts, "time": "2026-09-01 11:00:12"}

        processor = RealtimeStrategyProcessor(
            runtime=self.runtime, feature_engine=None, selection_plan=None,
            emit_on_price_tick=True, on_row=open_trade, target_symbol="BTCUSDT",
            intervals={}, latest_discrete_indicator_values=None,
            pending_small_candles={}, small_interval_ns=60_000_000_000,
            snapshot_observer=self.recorder.observe,
        )
        processor._emit_snapshot(actual)
        self.recorder.flush(force=True)
        # A second tick with the same timestamp must not overwrite the entry.
        self.recorder.observe(snapshot(ema=78_500), self.runtime)
        self.recorder.observe(snapshot("2026-09-01 11:00:59", ema=78_600), self.runtime)
        self.recorder.flush(force=True)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["kind"], "entry")
        self.assertEqual(rows[0]["ema"], actual.ema)
        self.assertEqual(rows[1]["ema"], 78_600)
        self.assertNotEqual(rows[0]["time"], rows[1]["time"])

    def test_failed_write_retries_without_losing_entry(self):
        row = snapshot()
        self.runtime.position = {"decision_time": row.row_ts}
        self.recorder.observe(row, self.runtime)
        with patch("bot.strategy_chart.save_strategy_chart_snapshots", side_effect=OSError("busy")):
            with self.assertRaises(OSError):
                self.recorder.flush(force=True)
        self.assertEqual(len(self.recorder.pending), 2)
        self.recorder.flush(force=True)
        self.assertEqual(len(self.rows()), 2)
        self.assertFalse(self.recorder.pending)

    def test_mixed_closed_and_intrabar_values_are_preserved(self):
        closed = dict(EMA=77_000, RSI=40, BBL=78_100, BBM=79_000, BBU=79_900, ATR=100)
        intrabar = dict(EMA=77_500, RSI=60, BBL=77_000, BBM=78_000, BBU=79_000, ATR=120)
        engine = SimpleNamespace(realtime_values=lambda **kwargs: intrabar)
        plan = indicator_selection_plan(dict(ema="intrabar", rsi="closed", bb="closed", atr="intrabar"))
        row = _build_realtime_snapshot(
            event_ts="2026-09-01 11:00:10", feature_engine=engine,
            selection_plan=plan, discrete_indicator_values=closed, current_price=78_000,
        )
        self.recorder.observe(row, self.runtime)
        self.recorder.flush(force=True)
        stored = self.rows()[0]
        self.assertEqual((stored["ema"], stored["rsi"], stored["bbl"], stored["atr"]),
                         (77_500, 40, 78_100, 120))

    def test_symbol_and_time_window_isolation(self):
        self.recorder.observe(snapshot(), self.runtime)
        self.recorder.symbol = "ETHUSDT"
        self.recorder.observe(snapshot(ema=2_000), self.runtime)
        self.recorder.flush(force=True)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows("ETHUSDT")[0]["ema"], 2_000)
        ts_ms = self.rows()[0]["ts_ms"]
        self.assertEqual(list_strategy_chart_snapshots("BTCUSDT", 0, ts_ms - 1), [])

    def test_discrete_snapshot_uses_decision_time_not_candle_open(self):
        row = snapshot()
        row.raw_row = {"open_time": row.row_ts}
        row.log_ts = "2026-09-01 11:01:02"
        self.runtime.live = True
        self.recorder.observe(row, self.runtime)
        self.recorder.flush(force=True)
        self.assertEqual(self.rows()[0]["time"], "2026-09-01T11:01:02+00:00")

    def build_payload(self, *, recorded=True, source="dataset"):
        ts = int(datetime(2026, 9, 1, 11, tzinfo=UTC).timestamp() * 1000)
        candle = dict(time=chart_service._iso_from_ts_ms(ts), ts_ms=ts,
                      open=78_000, high=78_010, low=77_990, close=78_000, volume=1)
        if recorded:
            candle.update(ema=99_000, rsi=90, bbl=98_000, bbm=99_000, bbu=100_000)
        price_candle = candle if source == "dataset" else {
            key: value for key, value in candle.items() if key not in {"ema", "rsi", "bbl", "bbm", "bbu"}
        }
        with (
            patch.object(chart_service, "load_config", return_value={"strategy_mode": "realtime"}),
            patch.object(chart_service, "load_state", return_value=({}, [])),
            patch.object(chart_service, "load_trade_history", return_value=([], [])),
            patch.object(chart_service, "load_balance_history", return_value=([], [])),
            patch.object(chart_service, "_fetch_candles", return_value=([price_candle], [], source, "1m")),
            patch.object(chart_service, "_fetch_candles_from_dataset", return_value=([candle], [])),
            patch.object(chart_service, "_fetch_candles_from_binance", side_effect=AssertionError("recalculation")),
        ):
            return chart_service.build_chart_payload("BTCUSDT", "1d", "1m", end="2026-09-01T11:01:00Z")

    def test_api_prefers_actual_decisions_and_extends_range(self):
        self.recorder.observe(snapshot(), self.runtime)
        self.recorder.flush(force=True)
        payload = self.build_payload()
        self.assertEqual(payload["indicator_source"], "dataset+strategy_decisions")
        self.assertEqual([point["value"] for point in payload["indicators"]["ema"]], [99_000, 77_410])
        self.assertEqual(payload["range_end"], "2026-09-01T11:00:10+00:00")

    def test_partial_recording_keeps_history_for_all_indicators(self):
        history = chart_service._build_indicators_from_candle_fields([
            dict(time=f"2026-09-01T11:0{minute}:00+00:00", ema=78_000 + minute,
                 rsi=40 + minute, bbl=77_000 + minute, bbm=78_000 + minute, bbu=79_000 + minute)
            for minute in range(5)
        ])
        self.recorder.observe(snapshot("2026-09-01 11:02:00"), self.runtime)
        self.recorder.flush(force=True)
        recorded, _ = chart_service._build_recorded_strategy_indicators("BTCUSDT", 0, 9_999_999_999_999)
        merged = chart_service._merge_indicator_history(history, recorded)
        series = [merged["ema"], merged["rsi"], *merged["bollinger"].values()]
        for points in series:
            self.assertEqual(len(points), 3)
            self.assertEqual(points[-1]["time"], "2026-09-01T11:02:00+00:00")
            self.assertEqual([point["time"] for point in points[:2]], [
                "2026-09-01T11:00:00+00:00", "2026-09-01T11:01:00+00:00",
            ])
        self.assertEqual(merged["ema"][-1]["value"], 77_410)
        self.assertEqual(merged["rsi"][-1]["value"], 35)
        self.assertEqual(merged["bollinger"]["lower"][-1]["value"], 78_800)

    def test_recording_without_legacy_history_is_still_returned(self):
        self.recorder.observe(snapshot(), self.runtime)
        self.recorder.flush(force=True)
        payload = self.build_payload(recorded=False)
        self.assertEqual(payload["indicator_source"], "strategy_decisions")
        self.assertEqual(payload["indicators"]["ema"][0]["value"], 77_410)

    def test_binance_candles_keep_dataset_history_with_new_decisions(self):
        self.recorder.observe(snapshot(), self.runtime)
        self.recorder.flush(force=True)
        payload = self.build_payload(source="binance")
        self.assertEqual(payload["indicator_source"], "dataset+strategy_decisions")
        self.assertEqual([point["value"] for point in payload["indicators"]["rsi"]], [90, 35])

    def test_legacy_history_without_recordings_is_still_returned(self):
        payload = self.build_payload()
        self.assertEqual(payload["indicator_source"], "dataset")
        self.assertEqual(payload["indicators"]["ema"][0]["value"], 99_000)

    def test_missing_indicators_are_not_recomputed_on_chart_timeframe(self):
        payload = self.build_payload(recorded=False)
        self.assertEqual(payload["indicators"], {})
        self.assertTrue(any("No recorded strategy indicators" in value for value in payload["warnings"]))

    def test_downsampling_keeps_entry_decisions(self):
        for minute in range(4):
            row = snapshot(f"2026-09-01 11:0{minute}:10", ema=77_000 + minute)
            self.runtime.position = {"decision_time": row.row_ts} if minute == 1 else None
            self.recorder.observe(row, self.runtime)
        self.recorder.flush(force=True)
        with patch.object(chart_service, "CHART_DATASET_MAX_CANDLES", 1):
            indicators, _ = chart_service._build_recorded_strategy_indicators("BTCUSDT", 0, 9_999_999_999_999)
        self.assertEqual([point["value"] for point in indicators["ema"]], [77_000, 77_001, 77_003])


if __name__ == "__main__":
    unittest.main()
