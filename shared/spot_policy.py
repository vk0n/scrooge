from __future__ import annotations

import math
from typing import Any


def _non_negative(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return max(0.0, numeric)


def calculate_spot_inventory_policy(
    *,
    current_quantity: float,
    target_quantity: float,
    minimum_holding_pct: float,
    binance_quantity: float,
) -> dict[str, float | None]:
    """Project the shared live/backtest Protected Floor and custody capacity."""
    current = _non_negative(current_quantity, field_name="Current quantity")
    target = _non_negative(target_quantity, field_name="Target quantity")
    minimum_pct = min(
        100.0,
        _non_negative(minimum_holding_pct, field_name="Minimum Holding percentage"),
    )
    binance = _non_negative(binance_quantity, field_name="Binance quantity")
    protected_floor = target * minimum_pct / 100.0
    amount_above_floor = max(0.0, current - protected_floor)
    amount_below_floor = max(0.0, protected_floor - current)
    return {
        "protected_floor_quantity": protected_floor,
        "protected_holding_quantity": min(current, protected_floor),
        "amount_above_protected_floor": amount_above_floor,
        "amount_below_protected_floor": amount_below_floor,
        "policy_sellable_quantity": amount_above_floor,
        "custody_sellable_quantity": min(amount_above_floor, binance),
        "target_delta_quantity": current - target,
        "target_delta_pct": ((current - target) / target) * 100 if target > 0 else None,
    }
