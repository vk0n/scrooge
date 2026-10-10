import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from api.services import config_service
from core.treasury_strategy_config import treasury_strategy_config_from_mapping


def treasury_rules() -> dict:
    return {
        "signal_refresh_seconds": 60,
        "signal": {
            "levels_pct": [2, 3, 4, 6],
            "base_tranches_pct": [10, 20, 30, 40],
            "accumulation_tranches_pct": [1, 3, 5, 10],
        },
        "progression": {
            "close_profit_pct": 4,
            "campaign_capacity_pct": 50,
            "full_deploy_threshold_pct": 25,
        },
        "waiter_cleanup": {
            "enabled": True,
            "max_open_bargains_per_asset": 10,
            "deep_loss": {
                "min_age_days": 15,
                "unrealized_pnl_pct": -25,
                "required_reverse_level": 1,
            },
            "aging": [
                {"min_age_days": 30, "required_reverse_level": 3},
                {"min_age_days": 60, "required_reverse_level": 2},
                {"min_age_days": 90, "required_reverse_level": 1},
            ],
            "capacity_cleanup": {"enabled": True, "min_age_days": 30},
        },
    }


class TreasuryStrategyConfigTests(unittest.TestCase):
    def test_parses_all_live_rules(self):
        config = treasury_strategy_config_from_mapping(treasury_rules())

        self.assertEqual(config.signal.levels_pct, (2.0, 3.0, 4.0, 6.0))
        self.assertEqual(config.signal.accumulation_tranches_pct, (1.0, 3.0, 5.0, 10.0))
        self.assertEqual(config.progression.close_profit_pct, 4.0)
        self.assertEqual(config.progression.estimated_fee_rate, 0.001)
        self.assertEqual(config.waiter_cleanup.deep_loss_unrealized_pnl_pct, -25.0)

    def test_rejects_unknown_and_misaligned_rules(self):
        unknown = treasury_rules()
        unknown["mystery"] = True
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            treasury_strategy_config_from_mapping(unknown)

        technical_fee = treasury_rules()
        technical_fee["progression"]["estimated_fee_rate"] = 0.002
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            treasury_strategy_config_from_mapping(technical_fee)

        misaligned = treasury_rules()
        misaligned["signal"]["base_tranches_pct"] = [10, 20]
        with self.assertRaisesRegex(ValueError, "same length"):
            treasury_strategy_config_from_mapping(misaligned)

    def test_rejects_unsafe_refresh_and_positive_deep_loss(self):
        too_fast = treasury_rules()
        too_fast["signal_refresh_seconds"] = 10
        with self.assertRaisesRegex(ValueError, "at least 30"):
            treasury_strategy_config_from_mapping(too_fast)

        invalid_loss = treasury_rules()
        invalid_loss["waiter_cleanup"]["deep_loss"]["unrealized_pnl_pct"] = 5
        with self.assertRaisesRegex(ValueError, "zero or negative"):
            treasury_strategy_config_from_mapping(invalid_loss)

    def test_rejects_coerced_boolean_and_fractional_level(self):
        invalid_boolean = treasury_rules()
        invalid_boolean["waiter_cleanup"]["enabled"] = "false"
        with self.assertRaisesRegex(ValueError, "true or false"):
            treasury_strategy_config_from_mapping(invalid_boolean)

        fractional_level = treasury_rules()
        fractional_level["waiter_cleanup"]["deep_loss"]["required_reverse_level"] = 1.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            treasury_strategy_config_from_mapping(fractional_level)


class TreasuryRulesPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "live.yaml"
        self.original = {
            "live": True,
            "symbol": "BTCUSDT",
            "params": {"tp_mult": 3},
            "treasury": treasury_rules(),
        }
        self.config_path.write_text(yaml.safe_dump(self.original, sort_keys=False), encoding="utf-8")
        self.path_patch = patch.object(config_service, "CONFIG_PATH", self.config_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_update_changes_only_treasury_subtree_and_creates_backup(self):
        changed = treasury_rules()
        changed["progression"]["close_profit_pct"] = 5

        result = config_service.update_treasury_rules_text(
            yaml.safe_dump(changed, sort_keys=False)
        )
        saved = config_service.load_config()

        self.assertTrue(result["updated"])
        self.assertNotIn("progression", saved)
        self.assertEqual(saved["treasury"]["progression"]["close_profit_pct"], 5.0)
        self.assertNotIn("estimated_fee_rate", saved["treasury"]["progression"])
        self.assertEqual(saved["params"], self.original["params"])
        self.assertTrue(Path(result["backup_path"]).exists())

    def test_invalid_update_does_not_touch_file(self):
        before = self.config_path.read_text(encoding="utf-8")
        invalid = treasury_rules()
        invalid["signal"]["levels_pct"] = [5, 2, 7, 9]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            config_service.update_treasury_rules_text(yaml.safe_dump(invalid, sort_keys=False))

        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
