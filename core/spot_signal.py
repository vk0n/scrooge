"""Rolling Spot signals and progressive level allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

DEFAULT_SIGNAL_LEVELS_PCT = (2.0, 3.0, 4.0, 5.0)
DEFAULT_BASE_TRANCHES_PCT = (10.0, 20.0, 30.0, 40.0)
DEFAULT_ACCUMULATION_TRANCHES_PCT = (1.0, 3.0, 5.0, 10.0)
DEFAULT_ACCUMULATION_COOLDOWN_HOURS = 24.0
DEFAULT_ACCUMULATION_MIN_PRICE_IMPROVEMENT_PCT = 3.0
DEFAULT_ACCUMULATION_REARM_RECOVERY_PCT = 10.0
DEFAULT_ACCUMULATION_REARM_SIGNAL_ABOVE_PCT = -5.0
DEFAULT_ACCUMULATION_REARM_CONFIRMATION_HOURS = 12.0
ROLLING_WINDOW_MS = 24 * 60 * 60 * 1000
MIN_ROLLING_WINDOW_MS = 23 * 60 * 60 * 1000
MAX_ROLLING_WINDOW_MS = 25 * 60 * 60 * 1000


@dataclass(frozen=True)
class SpotSignalConfig:
    levels_pct: tuple[float, ...] = DEFAULT_SIGNAL_LEVELS_PCT
    base_tranches_pct: tuple[float, ...] = DEFAULT_BASE_TRANCHES_PCT
    accumulation_levels_pct: tuple[float, ...] | None = None
    accumulation_tranches_pct: tuple[float, ...] | None = None
    accumulation_cooldown_hours: float = DEFAULT_ACCUMULATION_COOLDOWN_HOURS
    accumulation_min_price_improvement_pct: float = (
        DEFAULT_ACCUMULATION_MIN_PRICE_IMPROVEMENT_PCT
    )
    accumulation_rearm_recovery_pct: float = DEFAULT_ACCUMULATION_REARM_RECOVERY_PCT
    accumulation_rearm_signal_above_pct: float = (
        DEFAULT_ACCUMULATION_REARM_SIGNAL_ABOVE_PCT
    )
    accumulation_rearm_confirmation_hours: float = (
        DEFAULT_ACCUMULATION_REARM_CONFIRMATION_HOURS
    )

    def __post_init__(self) -> None:
        levels = tuple(float(value) for value in self.levels_pct)
        tranches = tuple(float(value) for value in self.base_tranches_pct)
        accumulation_levels = (
            tuple(float(value) for value in self.accumulation_levels_pct)
            if self.accumulation_levels_pct is not None
            else levels
        )
        accumulation_tranches = (
            tuple(float(value) for value in self.accumulation_tranches_pct)
            if self.accumulation_tranches_pct is not None
            else (
                DEFAULT_ACCUMULATION_TRANCHES_PCT
                if len(accumulation_levels) == len(DEFAULT_ACCUMULATION_TRANCHES_PCT)
                else tranches
            )
        )
        if not levels:
            raise ValueError("At least one Spot signal level is required.")
        if len(levels) != len(tranches):
            raise ValueError("Spot signal levels and base tranches must have the same length.")
        if len(accumulation_levels) != len(accumulation_tranches):
            raise ValueError(
                "Spot accumulation levels and accumulation tranches must have the same length."
            )
        if any(not math.isfinite(value) or value <= 0 for value in levels):
            raise ValueError("Spot signal levels must be finite positive percentages.")
        if any(current <= previous for previous, current in zip(levels, levels[1:])):
            raise ValueError("Spot signal levels must be strictly increasing.")
        if any(not math.isfinite(value) or value <= 0 for value in accumulation_levels):
            raise ValueError("Spot accumulation levels must be finite positive percentages.")
        if any(
            current <= previous
            for previous, current in zip(accumulation_levels, accumulation_levels[1:])
        ):
            raise ValueError("Spot accumulation levels must be strictly increasing.")
        if any(not math.isfinite(value) or value <= 0 or value > 100 for value in tranches):
            raise ValueError("Spot base tranches must be finite percentages in the range (0, 100].")
        if any(
            not math.isfinite(value) or value <= 0 or value > 100
            for value in accumulation_tranches
        ):
            raise ValueError(
                "Spot accumulation tranches must be finite percentages in the range (0, 100]."
            )
        cooldown_hours = float(self.accumulation_cooldown_hours)
        minimum_improvement_pct = float(self.accumulation_min_price_improvement_pct)
        rearm_recovery_pct = float(self.accumulation_rearm_recovery_pct)
        rearm_signal_above_pct = float(self.accumulation_rearm_signal_above_pct)
        rearm_confirmation_hours = float(self.accumulation_rearm_confirmation_hours)
        if not math.isfinite(cooldown_hours) or cooldown_hours < 0:
            raise ValueError(
                "Spot accumulation cooldown must be a finite non-negative number of hours."
            )
        if (
            not math.isfinite(minimum_improvement_pct)
            or minimum_improvement_pct < 0
            or minimum_improvement_pct >= 100
        ):
            raise ValueError(
                "Spot accumulation minimum price improvement must be in the range [0, 100)."
            )
        if (
            not math.isfinite(rearm_recovery_pct)
            or rearm_recovery_pct <= 0
            or rearm_recovery_pct >= 100
        ):
            raise ValueError("Spot accumulation rearm recovery must be in the range (0, 100).")
        if not math.isfinite(rearm_signal_above_pct):
            raise ValueError("Spot accumulation rearm signal threshold must be finite.")
        if not math.isfinite(rearm_confirmation_hours) or rearm_confirmation_hours < 0:
            raise ValueError(
                "Spot accumulation rearm confirmation must be a finite non-negative number of hours."
            )
        object.__setattr__(self, "levels_pct", levels)
        object.__setattr__(self, "base_tranches_pct", tranches)
        object.__setattr__(self, "accumulation_levels_pct", accumulation_levels)
        object.__setattr__(self, "accumulation_tranches_pct", accumulation_tranches)
        object.__setattr__(self, "accumulation_cooldown_hours", cooldown_hours)
        object.__setattr__(
            self,
            "accumulation_min_price_improvement_pct",
            minimum_improvement_pct,
        )
        object.__setattr__(self, "accumulation_rearm_recovery_pct", rearm_recovery_pct)
        object.__setattr__(
            self,
            "accumulation_rearm_signal_above_pct",
            rearm_signal_above_pct,
        )
        object.__setattr__(
            self,
            "accumulation_rearm_confirmation_hours",
            rearm_confirmation_hours,
        )


def _positive_number(value: float, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")
    return numeric


def evaluate_rolling_24h_opportunity(
    *,
    current_price: float,
    reference_price: float,
    current_at_ms: int,
    reference_at_ms: int,
    config: SpotSignalConfig | None = None,
) -> dict[str, object]:
    resolved_config = config or SpotSignalConfig()
    current = _positive_number(current_price, "current_price")
    reference = _positive_number(reference_price, "reference_price")
    current_ts = int(current_at_ms)
    reference_ts = int(reference_at_ms)
    window_ms = current_ts - reference_ts
    if not MIN_ROLLING_WINDOW_MS <= window_ms <= MAX_ROLLING_WINDOW_MS:
        raise ValueError("Rolling Spot signal requires a reference price approximately 24 hours old.")

    change_pct = ((current / reference) - 1.0) * 100.0
    absolute_change_pct = abs(change_pct)
    level = 0
    tranche_pct = 0.0
    for index, threshold_pct in enumerate(resolved_config.levels_pct, start=1):
        if absolute_change_pct < threshold_pct and not math.isclose(
            absolute_change_pct,
            threshold_pct,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            break
        level = index
        tranche_pct = resolved_config.base_tranches_pct[index - 1]

    accumulation_level = 0
    accumulation_tranche_pct = 0.0
    if change_pct < 0:
        for index, threshold_pct in enumerate(resolved_config.accumulation_levels_pct, start=1):
            if absolute_change_pct < threshold_pct and not math.isclose(
                absolute_change_pct,
                threshold_pct,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                break
            accumulation_level = index
            accumulation_tranche_pct = resolved_config.accumulation_tranches_pct[index - 1]

    if level == 0:
        opportunity = "hold"
        reason_code = "below_level_1"
    else:
        opportunity = "sell" if change_pct > 0 else "buy"
        reason_code = "rolling_24h_level_reached"

    return {
        "opportunity": opportunity,
        "level": level,
        "bargain_level": level,
        "accumulation_level": accumulation_level,
        "base_tranche_pct": tranche_pct,
        "accumulation_tranche_pct": accumulation_tranche_pct,
        "rolling_change_pct": change_pct,
        "absolute_change_pct": absolute_change_pct,
        "current_price": current,
        "reference_price": reference,
        "current_at_ms": current_ts,
        "reference_at_ms": reference_ts,
        "window_ms": window_ms,
        "reason_code": reason_code,
        "levels_pct": list(resolved_config.levels_pct),
        "base_tranches_pct": list(resolved_config.base_tranches_pct),
        "accumulation_levels_pct": list(resolved_config.accumulation_levels_pct),
        "accumulation_tranches_pct": list(resolved_config.accumulation_tranches_pct),
        "accumulation_cooldown_hours": resolved_config.accumulation_cooldown_hours,
        "accumulation_min_price_improvement_pct": (
            resolved_config.accumulation_min_price_improvement_pct
        ),
        "accumulation_rearm_recovery_pct": (
            resolved_config.accumulation_rearm_recovery_pct
        ),
        "accumulation_rearm_signal_above_pct": (
            resolved_config.accumulation_rearm_signal_above_pct
        ),
        "accumulation_rearm_confirmation_hours": (
            resolved_config.accumulation_rearm_confirmation_hours
        ),
    }


def parse_percentage_series(raw: str, *, field_name: str) -> tuple[float, ...]:
    values: Sequence[str] = [item.strip() for item in str(raw or "").split(",")]
    if not values or any(not item for item in values):
        raise ValueError(f"{field_name} must be a comma-separated list of percentages.")
    try:
        return tuple(float(item) for item in values)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain only numeric percentages.") from exc
