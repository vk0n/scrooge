from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


DAY_MS = 24 * 60 * 60 * 1000
CLEANUP_REASONS = {
    "age_l3_cleanup",
    "age_l2_cleanup",
    "age_l1_cleanup",
    "deep_loss_cleanup",
    "capacity_cleanup",
}


@dataclass(frozen=True)
class AgingCleanupRule:
    min_age_days: float
    required_reverse_level: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_age_days) or self.min_age_days < 0:
            raise ValueError("Cleanup age must be a non-negative number of days.")
        if self.required_reverse_level not in {1, 2, 3, 4}:
            raise ValueError("Cleanup reverse level must be between 1 and 4.")


def _default_aging_rules() -> tuple[AgingCleanupRule, ...]:
    return (
        AgingCleanupRule(min_age_days=30, required_reverse_level=3),
        AgingCleanupRule(min_age_days=60, required_reverse_level=2),
        AgingCleanupRule(min_age_days=90, required_reverse_level=1),
    )


@dataclass(frozen=True)
class WaiterCleanupConfig:
    enabled: bool = True
    max_open_bargains_per_asset: int = 10
    deep_loss_min_age_days: float = 15.0
    deep_loss_unrealized_pnl_pct: float = -20.0
    deep_loss_required_reverse_level: int = 1
    aging_rules: tuple[AgingCleanupRule, ...] = field(default_factory=_default_aging_rules)
    capacity_cleanup_enabled: bool = True
    capacity_cleanup_min_age_days: float = 30.0

    def __post_init__(self) -> None:
        if self.max_open_bargains_per_asset <= 0:
            raise ValueError("Maximum open Bargains per asset must be positive.")
        if not math.isfinite(self.deep_loss_min_age_days) or self.deep_loss_min_age_days < 0:
            raise ValueError("Deep-loss minimum age must be non-negative.")
        if (
            not math.isfinite(self.deep_loss_unrealized_pnl_pct)
            or self.deep_loss_unrealized_pnl_pct > 0
        ):
            raise ValueError("Deep-loss PnL threshold must be zero or negative.")
        if self.deep_loss_required_reverse_level not in {1, 2, 3, 4}:
            raise ValueError("Deep-loss reverse level must be between 1 and 4.")
        if not self.aging_rules:
            raise ValueError("At least one aging cleanup rule is required.")
        ages = [rule.min_age_days for rule in self.aging_rules]
        if ages != sorted(ages):
            raise ValueError("Aging cleanup rules must be ordered by minimum age.")
        if (
            not math.isfinite(self.capacity_cleanup_min_age_days)
            or self.capacity_cleanup_min_age_days < 0
        ):
            raise ValueError("Capacity-cleanup minimum age must be non-negative.")


def waiter_cleanup_config_from_mapping(payload: dict[str, Any] | None) -> WaiterCleanupConfig:
    data = payload or {}
    deep_loss = data.get("deep_loss") if isinstance(data.get("deep_loss"), dict) else {}
    capacity = (
        data.get("capacity_cleanup")
        if isinstance(data.get("capacity_cleanup"), dict)
        else {}
    )
    raw_aging = data.get("aging")
    aging_rules = _default_aging_rules()
    if raw_aging is not None:
        if not isinstance(raw_aging, list):
            raise ValueError("waiter_cleanup.aging must be a list.")
        aging_rules = tuple(
            AgingCleanupRule(
                min_age_days=float(item["min_age_days"]),
                required_reverse_level=int(item["required_reverse_level"]),
            )
            for item in raw_aging
            if isinstance(item, dict)
        )
        if len(aging_rules) != len(raw_aging):
            raise ValueError("Every waiter-cleanup aging rule must be an object.")
    return WaiterCleanupConfig(
        enabled=bool(data.get("enabled", True)),
        max_open_bargains_per_asset=int(data.get("max_open_bargains_per_asset", 10)),
        deep_loss_min_age_days=float(deep_loss.get("min_age_days", 15)),
        deep_loss_unrealized_pnl_pct=float(deep_loss.get("unrealized_pnl_pct", -20)),
        deep_loss_required_reverse_level=int(deep_loss.get("required_reverse_level", 1)),
        aging_rules=aging_rules,
        capacity_cleanup_enabled=bool(capacity.get("enabled", True)),
        capacity_cleanup_min_age_days=float(capacity.get("min_age_days", 30)),
    )


def bargain_age_days(bargain: dict[str, Any], *, now_ms: int) -> float:
    opened_at_ms = int(bargain.get("opened_at_ms") or now_ms)
    return max(0.0, (int(now_ms) - opened_at_ms) / DAY_MS)


def required_cleanup_level(
    bargain: dict[str, Any],
    *,
    now_ms: int,
    unrealized_pnl_pct: float | None,
    config: WaiterCleanupConfig,
) -> dict[str, Any] | None:
    if not config.enabled:
        return None
    age_days = bargain_age_days(bargain, now_ms=now_ms)
    if (
        age_days >= config.deep_loss_min_age_days
        and unrealized_pnl_pct is not None
        and unrealized_pnl_pct <= config.deep_loss_unrealized_pnl_pct
    ):
        return {
            "required_reverse_level": config.deep_loss_required_reverse_level,
            "cleanup_reason": "deep_loss_cleanup",
            "age_days": age_days,
        }
    eligible_rule = None
    for rule in config.aging_rules:
        if age_days >= rule.min_age_days:
            eligible_rule = rule
    if eligible_rule is None:
        return None
    return {
        "required_reverse_level": eligible_rule.required_reverse_level,
        "cleanup_reason": f"age_l{eligible_rule.required_reverse_level}_cleanup",
        "age_days": age_days,
    }


def reverse_signal_satisfies(
    *,
    origin_side: str,
    signal_side: str,
    signal_level: int,
    required_level: int,
) -> bool:
    normalized_origin = str(origin_side or "").strip().lower()
    normalized_signal = str(signal_side or "").strip().lower()
    expected = "sell" if normalized_origin == "buy" else "buy" if normalized_origin == "sell" else None
    return (
        expected is not None
        and normalized_signal == expected
        and int(signal_level) >= int(required_level)
    )


def remaining_unrealized_pnl_pct(economics: dict[str, Any]) -> float | None:
    unrealized = economics.get("unrealized_pnl_quote")
    remaining_notional = economics.get("remaining_opening_quote_quantity")
    if unrealized is None or remaining_notional is None:
        return None
    denominator = float(remaining_notional)
    if denominator <= 1e-12:
        return None
    return float(unrealized) / denominator * 100.0


def is_cleanup_reason(reason: str | None) -> bool:
    return str(reason or "").strip().lower() in CLEANUP_REASONS
