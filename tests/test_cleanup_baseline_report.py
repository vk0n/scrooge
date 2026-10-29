import csv
import json
from pathlib import Path
import tempfile
import unittest

from backtest.cleanup_baseline_report import (
    _capacity_series,
    _compress_candles,
    _event_rows,
    _hourly_candles,
    render_html,
)


class CleanupBaselineReportTests(unittest.TestCase):
    def test_event_rows_keep_only_cleanup_closes(self):
        rows = _event_rows(
            [
                {
                    "swing_id": "cleanup",
                    "asset_symbol": "XRP",
                    "close_reason": "deep_loss_cleanup",
                    "closed_at_ms": 1_700_000_000_000,
                    "age_seconds": 10 * 86_400,
                    "trading_objective": "accumulate_asset",
                    "close_context": {
                        "unrealized_pnl_pct_before_cleanup": -50,
                        "required_cleanup_level": 3,
                        "actual_reverse_signal_level": 4,
                    },
                    "economics": {"realized_pnl_quote": -25},
                    "executions": [
                        {
                            "executed_at": "2023-11-14T00:00:00+00:00",
                            "side": "sell",
                            "quantity": 10,
                            "price": 1,
                            "quote_quantity": 10,
                            "fee_amount": 0.01,
                            "fee_asset": "USDT",
                            "reason": {"action_type": "open", "signal": "sell", "signal_level": 2},
                        }
                    ],
                },
                {"swing_id": "profit", "close_reason": "profit_target"},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason_label"], "Deep loss")
        self.assertEqual(rows[0]["age_days"], 10)
        self.assertEqual(rows[0]["pnl_quote"], -25)
        self.assertEqual(rows[0]["executions"][0]["phase"], "open")
        self.assertEqual(rows[0]["executions"][0]["signal_level"], 2)

    def test_capacity_series_uses_last_snapshot_of_each_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.csv"
            fields = [
                "timestamp_ms",
                "timestamp",
                "asset_symbol",
                "open_bargain_count",
                "underwater_sell_origin_swings",
                "tradable_inventory_utilization_pct",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for timestamp, date, opens in (
                    ("1", "2026-01-01T01:00:00+00:00", 2),
                    ("2", "2026-01-01T23:00:00+00:00", 4),
                    ("3", "2026-01-02T23:00:00+00:00", 5),
                ):
                    writer.writerow(
                        {
                            "timestamp_ms": timestamp,
                            "timestamp": date,
                            "asset_symbol": "XRP",
                            "open_bargain_count": opens,
                            "underwater_sell_origin_swings": opens,
                            "tradable_inventory_utilization_pct": 50,
                        }
                    )

            points, summary = _capacity_series(path, capacity=5)

        self.assertEqual([point["open"] for point in points], [4, 5])
        self.assertAlmostEqual(summary["average_open"], 11 / 3)
        self.assertAlmostEqual(summary["at_cap_pct"], 100 / 3)

    def test_minute_candles_are_aggregated_and_compressed_as_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XRPUSDT-1m-0-7200000.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"open_time_ms": 0, "close_time_ms": 59_999, "open": 10, "high": 12, "low": 9, "close": 11, "volume": 1},
                        {"open_time_ms": 60_000, "close_time_ms": 119_999, "open": 11, "high": 13, "low": 8, "close": 12, "volume": 1},
                        {"open_time_ms": 3_600_000, "close_time_ms": 3_659_999, "open": 12, "high": 14, "low": 10, "close": 13, "volume": 1},
                    ]
                )

            hourly = _hourly_candles(path, start_ms=0, end_ms=7_200_000)
            compressed, step = _compress_candles(hourly, max_points=1)

        self.assertEqual(len(hourly), 2)
        self.assertEqual(hourly[0], {"t": 0, "o": 10.0, "h": 13.0, "l": 8.0, "c": 12.0})
        self.assertEqual(step, 2)
        self.assertEqual(compressed[0], {"t": 0, "o": 10.0, "h": 14.0, "l": 8.0, "c": 13.0})

    def test_html_is_standalone_and_escapes_script_terminator(self):
        payload = {
            "meta": {"name": "baseline </script>"},
        }

        output = render_html(payload)

        self.assertIn("<!doctype html>", output)
        self.assertIn("baseline <\\/script>", output)
        self.assertNotIn("https://", output)


if __name__ == "__main__":
    unittest.main()
