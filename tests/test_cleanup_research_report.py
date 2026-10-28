import unittest

from backtest.cleanup_research_report import (
    _annotate_operational_rows,
    _html_report,
    _table,
)


class CleanupResearchReportTests(unittest.TestCase):
    def test_restore_cost_is_informational_not_an_acceptance_limit(self):
        baseline = {
            "status": "ok",
            "maximum_drawdown_pct": -50.0,
            "profit_target_closes": 10,
        }
        candidate = {
            "name": "high-restore-cost",
            "status": "ok",
            "open_bargains_90_plus": 2,
            "oldest_open_days": 90.0,
            "restore_cost_quote": 1_000_000.0,
            "maximum_drawdown_pct": -51.0,
            "profit_target_closes": 12,
        }

        _annotate_operational_rows([candidate], baseline=baseline)

        self.assertTrue(candidate["operationally_accepted"])

    def test_failed_rows_are_preserved_without_metric_access(self):
        baseline = {
            "status": "ok",
            "maximum_drawdown_pct": -50.0,
            "profit_target_closes": 10,
        }
        failed = {"name": "failed-candidate", "status": "failed", "error": "boom"}

        _annotate_operational_rows([failed], baseline=baseline)

        self.assertFalse(failed["operationally_accepted"])
        self.assertIsNone(failed["additional_profit_target_closes_vs_baseline"])
        rendered = "\n".join(_table([failed]))
        self.assertIn("failed-candidate", rendered)
        self.assertIn("boom", rendered)

    def test_html_report_has_sortable_numeric_and_text_columns(self):
        candidate = {
            "study": "phase-a",
            "name": "candidate-a",
            "status": "ok",
            "operationally_accepted": True,
            "edge_vs_hodl_pct_points": 12.5,
            "cleanup_cash_burn_ratio_pct": 33.0,
            "final_treasury_value": 1234.0,
            "open_bargains_90_plus": 1,
            "oldest_open_days": 91.0,
            "restore_cost_quote": 500.0,
        }

        rendered = _html_report(
            rows=[candidate],
            frontier=[candidate],
            recommended=candidate,
            recommendation_accepted=True,
            interval_start="2022-12-01",
            interval_end="2024-12-01",
        )

        self.assertIn('data-type="text">Candidate', rendered)
        self.assertIn('data-type="number">Edge pp', rendered)
        self.assertIn('data-sort="12.5"', rendered)
        self.assertIn("aria-sort", rendered)
        self.assertIn("addEventListener('click'", rendered)
        self.assertIn("2022-12-01..2024-12-01", rendered)


if __name__ == "__main__":
    unittest.main()
