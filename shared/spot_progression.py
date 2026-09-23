from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProgressiveSwingConfig:
    close_profit_pct: float = 5.0
    estimated_fee_rate: float = 0.001

    def __post_init__(self) -> None:
        if not math.isfinite(self.close_profit_pct) or self.close_profit_pct <= 0:
            raise ValueError("Spot Swing close profit must be a finite positive percentage.")
        if not math.isfinite(self.estimated_fee_rate) or not 0 <= self.estimated_fee_rate < 1:
            raise ValueError("Spot estimated fee rate must be in the range [0, 1).")


def plan_opening_quantity(signal: dict[str, Any], holding: dict[str, Any]) -> dict[str, Any]:
    """Convert a sized opportunity into an economic quantity before exchange filters."""
    side = str(signal.get("opportunity") or "").strip().lower()
    objective = str(signal.get("trading_objective") or "").strip().lower()
    tranche_pct = float(signal.get("final_tranche_pct") or 0.0)
    target_quantity = float(holding.get("target_quantity") or 0.0)
    minimum_holding_raw = holding.get("minimum_holding_pct")
    minimum_holding_pct = 100.0 if minimum_holding_raw is None else float(minimum_holding_raw)
    if side not in {"buy", "sell"} or tranche_pct <= 0 or target_quantity <= 0:
        return {"eligible": False, "reason": "no_opening_opportunity", "quantity": 0.0}
    if objective == "accumulate_asset" and side == "buy":
        return {
            "eligible": False,
            "reason": "accumulate_asset_sell_origin_only_v1",
            "quantity": 0.0,
        }

    fraction = min(100.0, tranche_pct) / 100.0
    if side == "sell":
        strategic_capacity = target_quantity * max(0.0, 100.0 - minimum_holding_pct) / 100.0
        quantity = strategic_capacity * fraction
    else:
        strategic_capacity = target_quantity
        quantity = target_quantity * fraction
    if not math.isfinite(quantity) or quantity <= 0:
        return {"eligible": False, "reason": "zero_strategy_capacity", "quantity": 0.0}
    return {
        "eligible": True,
        "reason": "progressive_level_open",
        "quantity": quantity,
        "strategic_capacity": strategic_capacity,
        "tranche_pct": tranche_pct,
    }


def plan_profitable_close(
    swing: dict[str, Any],
    economics: dict[str, Any],
    *,
    current_price: float,
    available_quote_quantity: float = 0.0,
    config: ProgressiveSwingConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one Swing against its own opening basis, never portfolio average cost."""
    resolved = config or ProgressiveSwingConfig()
    origin_side = str(swing.get("origin_side") or "").strip().lower()
    objective = str(swing.get("trading_objective") or "").strip().lower()
    opening_price = float(economics.get("weighted_opening_price") or 0.0)
    remaining_quantity = float(economics.get("remaining_quantity") or 0.0)
    market_price = float(current_price)
    if origin_side not in {"buy", "sell"} or opening_price <= 0 or remaining_quantity <= 0:
        return {"eligible": False, "reason": "swing_has_no_open_inventory", "quantity": 0.0}
    if not math.isfinite(market_price) or market_price <= 0:
        return {"eligible": False, "reason": "invalid_market_price", "quantity": 0.0}

    favorable_move_pct = (
        ((opening_price - market_price) / opening_price) * 100.0
        if origin_side == "sell"
        else ((market_price / opening_price) - 1.0) * 100.0
    )
    if favorable_move_pct + 1e-12 < resolved.close_profit_pct:
        return {
            "eligible": False,
            "reason": "close_profit_not_reached",
            "quantity": 0.0,
            "favorable_move_pct": favorable_move_pct,
        }

    closing_side = "buy" if origin_side == "sell" else "sell"
    quantity = remaining_quantity
    if origin_side == "sell" and objective == "accumulate_asset":
        opening_quote = float(economics.get("opening_quote_quantity") or 0.0)
        closing_quote = float(economics.get("closing_quote_quantity") or 0.0)
        opening_quote_fee = float((economics.get("fees_by_asset") or {}).get(swing.get("quote_symbol"), 0.0))
        reusable_quote = max(0.0, opening_quote - opening_quote_fee - closing_quote)
        desired_quantity = reusable_quote / (market_price * (1.0 + resolved.estimated_fee_rate))
        cash_limited_quantity = max(0.0, available_quote_quantity) / (
            market_price * (1.0 + resolved.estimated_fee_rate)
        )
        quantity = min(desired_quantity, cash_limited_quantity)

    if not math.isfinite(quantity) or quantity <= 0:
        return {
            "eligible": False,
            "reason": "insufficient_closing_inventory_or_cash",
            "quantity": 0.0,
            "favorable_move_pct": favorable_move_pct,
        }
    return {
        "eligible": True,
        "reason": "swing_profit_target_reached",
        "side": closing_side,
        "quantity": quantity,
        "favorable_move_pct": favorable_move_pct,
        "close_profit_pct": resolved.close_profit_pct,
        "opening_price": opening_price,
        "current_price": market_price,
    }
