"""Validated Treasury strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from core.spot_progression import ProgressiveSwingConfig
from core.spot_signal import SpotSignalConfig
from core.spot_waiter_cleanup import AgingCleanupRule, WaiterCleanupConfig


_TREASURY_KEYS = {"signal_refresh_seconds", "signal", "progression", "waiter_cleanup"}
_SIGNAL_KEYS = {"levels_pct", "base_tranches_pct", "accumulation_tranches_pct"}
_PROGRESSION_KEYS = {
    "close_profit_pct",
    "campaign_capacity_pct",
    "full_deploy_threshold_pct",
}
_CLEANUP_KEYS = {"enabled", "max_open_bargains_per_asset", "deep_loss", "aging", "capacity_cleanup"}
_DEEP_LOSS_KEYS = {"min_age_days", "unrealized_pnl_pct", "required_reverse_level"}
_AGING_KEYS = {"min_age_days", "required_reverse_level"}
_CAPACITY_CLEANUP_KEYS = {"enabled", "min_age_days"}


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return value


def _reject_unknown(payload: dict[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported field(s): {', '.join(unknown)}")


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite.")
    return numeric


def _integer(value: Any, *, field: str) -> int:
    numeric = _number(value, field=field)
    if not numeric.is_integer():
        raise ValueError(f"{field} must be an integer.")
    return int(numeric)


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false.")
    return value


def _require(payload: dict[str, Any], required: set[str], *, field: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{field} is missing required field(s): {', '.join(missing)}")


def _series(value: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list of percentages.")
    return tuple(_number(item, field=f"{field}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class TreasuryStrategyConfig:
    signal_refresh_seconds: float
    signal: SpotSignalConfig
    progression: ProgressiveSwingConfig
    waiter_cleanup: WaiterCleanupConfig

    def __post_init__(self) -> None:
        if not math.isfinite(self.signal_refresh_seconds) or self.signal_refresh_seconds < 30:
            raise ValueError("treasury.signal_refresh_seconds must be at least 30 seconds.")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "signal_refresh_seconds": self.signal_refresh_seconds,
            "signal": {
                "levels_pct": list(self.signal.levels_pct),
                "base_tranches_pct": list(self.signal.base_tranches_pct),
                "accumulation_tranches_pct": list(self.signal.accumulation_tranches_pct),
            },
            "progression": {
                "close_profit_pct": self.progression.close_profit_pct,
                "campaign_capacity_pct": self.progression.campaign_capacity_pct,
                "full_deploy_threshold_pct": self.progression.full_deploy_threshold_pct,
            },
            "waiter_cleanup": {
                "enabled": self.waiter_cleanup.enabled,
                "max_open_bargains_per_asset": self.waiter_cleanup.max_open_bargains_per_asset,
                "deep_loss": {
                    "min_age_days": self.waiter_cleanup.deep_loss_min_age_days,
                    "unrealized_pnl_pct": self.waiter_cleanup.deep_loss_unrealized_pnl_pct,
                    "required_reverse_level": self.waiter_cleanup.deep_loss_required_reverse_level,
                },
                "aging": [
                    {
                        "min_age_days": rule.min_age_days,
                        "required_reverse_level": rule.required_reverse_level,
                    }
                    for rule in self.waiter_cleanup.aging_rules
                ],
                "capacity_cleanup": {
                    "enabled": self.waiter_cleanup.capacity_cleanup_enabled,
                    "min_age_days": self.waiter_cleanup.capacity_cleanup_min_age_days,
                },
            },
        }


def treasury_strategy_config_from_mapping(payload: Any) -> TreasuryStrategyConfig:
    treasury = _mapping(payload, field="treasury")
    _reject_unknown(treasury, _TREASURY_KEYS, field="treasury")
    _require(treasury, _TREASURY_KEYS, field="treasury")

    signal = _mapping(treasury["signal"], field="treasury.signal")
    progression = _mapping(treasury["progression"], field="treasury.progression")
    cleanup = _mapping(treasury["waiter_cleanup"], field="treasury.waiter_cleanup")
    _reject_unknown(signal, _SIGNAL_KEYS, field="treasury.signal")
    _reject_unknown(progression, _PROGRESSION_KEYS, field="treasury.progression")
    _reject_unknown(cleanup, _CLEANUP_KEYS, field="treasury.waiter_cleanup")
    _require(signal, _SIGNAL_KEYS, field="treasury.signal")
    _require(progression, _PROGRESSION_KEYS, field="treasury.progression")
    _require(cleanup, _CLEANUP_KEYS, field="treasury.waiter_cleanup")

    deep_loss = _mapping(cleanup.get("deep_loss"), field="treasury.waiter_cleanup.deep_loss")
    capacity = _mapping(
        cleanup.get("capacity_cleanup"),
        field="treasury.waiter_cleanup.capacity_cleanup",
    )
    _reject_unknown(deep_loss, _DEEP_LOSS_KEYS, field="treasury.waiter_cleanup.deep_loss")
    _reject_unknown(capacity, _CAPACITY_CLEANUP_KEYS, field="treasury.waiter_cleanup.capacity_cleanup")
    _require(deep_loss, _DEEP_LOSS_KEYS, field="treasury.waiter_cleanup.deep_loss")
    _require(capacity, _CAPACITY_CLEANUP_KEYS, field="treasury.waiter_cleanup.capacity_cleanup")
    raw_aging = cleanup.get("aging")
    if not isinstance(raw_aging, list):
        raise ValueError("treasury.waiter_cleanup.aging must be a list.")
    for index, item in enumerate(raw_aging):
        aging_rule = _mapping(item, field=f"treasury.waiter_cleanup.aging[{index}]")
        _reject_unknown(
            aging_rule,
            _AGING_KEYS,
            field=f"treasury.waiter_cleanup.aging[{index}]",
        )
        _require(
            aging_rule,
            _AGING_KEYS,
            field=f"treasury.waiter_cleanup.aging[{index}]",
        )

    return TreasuryStrategyConfig(
        signal_refresh_seconds=_number(
            treasury["signal_refresh_seconds"],
            field="treasury.signal_refresh_seconds",
        ),
        signal=SpotSignalConfig(
            levels_pct=_series(signal["levels_pct"], field="treasury.signal.levels_pct"),
            base_tranches_pct=_series(
                signal["base_tranches_pct"],
                field="treasury.signal.base_tranches_pct",
            ),
            accumulation_tranches_pct=_series(
                signal["accumulation_tranches_pct"],
                field="treasury.signal.accumulation_tranches_pct",
            ),
        ),
        progression=ProgressiveSwingConfig(
            close_profit_pct=_number(
                progression["close_profit_pct"],
                field="treasury.progression.close_profit_pct",
            ),
            campaign_capacity_pct=_number(
                progression["campaign_capacity_pct"],
                field="treasury.progression.campaign_capacity_pct",
            ),
            full_deploy_threshold_pct=_number(
                progression["full_deploy_threshold_pct"],
                field="treasury.progression.full_deploy_threshold_pct",
            ),
        ),
        waiter_cleanup=WaiterCleanupConfig(
            enabled=_boolean(cleanup["enabled"], field="treasury.waiter_cleanup.enabled"),
            max_open_bargains_per_asset=_integer(
                cleanup["max_open_bargains_per_asset"],
                field="treasury.waiter_cleanup.max_open_bargains_per_asset",
            ),
            deep_loss_min_age_days=_number(
                deep_loss["min_age_days"],
                field="treasury.waiter_cleanup.deep_loss.min_age_days",
            ),
            deep_loss_unrealized_pnl_pct=_number(
                deep_loss["unrealized_pnl_pct"],
                field="treasury.waiter_cleanup.deep_loss.unrealized_pnl_pct",
            ),
            deep_loss_required_reverse_level=_integer(
                deep_loss["required_reverse_level"],
                field="treasury.waiter_cleanup.deep_loss.required_reverse_level",
            ),
            aging_rules=tuple(
                AgingCleanupRule(
                    min_age_days=_number(
                        item["min_age_days"],
                        field=f"treasury.waiter_cleanup.aging[{index}].min_age_days",
                    ),
                    required_reverse_level=_integer(
                        item["required_reverse_level"],
                        field=f"treasury.waiter_cleanup.aging[{index}].required_reverse_level",
                    ),
                )
                for index, item in enumerate(raw_aging)
            ),
            capacity_cleanup_enabled=_boolean(
                capacity["enabled"],
                field="treasury.waiter_cleanup.capacity_cleanup.enabled",
            ),
            capacity_cleanup_min_age_days=_number(
                capacity["min_age_days"],
                field="treasury.waiter_cleanup.capacity_cleanup.min_age_days",
            ),
        ),
    )
