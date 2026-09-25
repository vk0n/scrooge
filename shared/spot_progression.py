from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProgressiveSwingConfig:
    close_profit_pct: float = 5.0
    estimated_fee_rate: float = 0.001
    treasury_accumulation_enabled: bool = False
    campaign_capacity_pct: float = 50.0
    full_deploy_threshold_pct: float = 25.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.close_profit_pct) or self.close_profit_pct <= 0:
            raise ValueError("Spot Swing close profit must be a finite positive percentage.")
        if not math.isfinite(self.estimated_fee_rate) or not 0 <= self.estimated_fee_rate < 1:
            raise ValueError("Spot estimated fee rate must be in the range [0, 1).")
        if not math.isfinite(self.campaign_capacity_pct) or not 0 < self.campaign_capacity_pct <= 100:
            raise ValueError("Spot campaign capacity must be in the range (0, 100].")
        if not math.isfinite(self.full_deploy_threshold_pct) or not 0 <= self.full_deploy_threshold_pct <= 100:
            raise ValueError("Spot full-deploy threshold must be in the range [0, 100].")


def policy_sellable_reference(target_quantity: float, minimum_holding_pct: float) -> float:
    target = max(0.0, float(target_quantity))
    minimum = min(100.0, max(0.0, float(minimum_holding_pct)))
    return target * (1.0 - minimum / 100.0)


def initialize_sell_campaign_capacity(
    campaign: dict[str, Any],
    holding: dict[str, Any],
    *,
    config: ProgressiveSwingConfig | None = None,
) -> dict[str, Any]:
    """Freeze the current policy sellable budget when a SELL campaign starts."""
    if str(campaign.get("active_side") or "").lower() != "sell":
        return dict(campaign)
    if campaign.get("campaign_capacity_quantity") is not None:
        return dict(campaign)

    resolved = config or ProgressiveSwingConfig()
    target = max(0.0, float(holding.get("target_quantity") or 0.0))
    minimum_raw = holding.get("minimum_holding_pct")
    minimum = 100.0 if minimum_raw is None else float(minimum_raw)
    reference = policy_sellable_reference(target, minimum)
    remaining_raw = holding.get("policy_sellable_quantity")
    if remaining_raw is None:
        current = max(0.0, float(holding.get("quantity", target) or 0.0))
        remaining_raw = max(0.0, current - target * min(100.0, max(0.0, minimum)) / 100.0)
    remaining = max(0.0, float(remaining_raw or 0.0))
    ratio = remaining / reference if reference > 0 else 0.0
    full_deploy = reference > 0 and ratio <= resolved.full_deploy_threshold_pct / 100.0
    capacity = remaining if full_deploy else remaining * resolved.campaign_capacity_pct / 100.0
    return {
        **campaign,
        "campaign_start_target_quantity": target,
        "campaign_start_minimum_holding_pct": minimum,
        "campaign_start_policy_sellable_reference": reference,
        "campaign_start_remaining_sellable_quantity": remaining,
        "remaining_sellable_ratio_at_start": ratio,
        "campaign_capacity_pct": resolved.campaign_capacity_pct,
        "full_deploy_threshold_pct": resolved.full_deploy_threshold_pct,
        "campaign_capacity_quantity": capacity,
        "campaign_consumed_quantity": 0.0,
        "campaign_capacity_mode": "full_deploy" if full_deploy else "normal",
    }


def plan_opening_quantity(
    signal: dict[str, Any],
    holding: dict[str, Any],
    campaign: dict[str, Any] | None = None,
    *,
    config: ProgressiveSwingConfig | None = None,
) -> dict[str, Any]:
    """Convert a sized opportunity into an economic quantity before exchange filters."""
    side = str(signal.get("opportunity") or "").strip().lower()
    tranche_pct = float(signal.get("base_tranche_pct") or signal.get("final_tranche_pct") or 0.0)
    if side != "sell" or tranche_pct <= 0:
        return {"eligible": False, "reason": "no_opening_opportunity", "quantity": 0.0}

    initialized = initialize_sell_campaign_capacity(
        campaign or {"active_side": "sell"}, holding, config=config
    )
    capacity = max(0.0, float(initialized.get("campaign_capacity_quantity") or 0.0))
    consumed = max(0.0, float(initialized.get("campaign_consumed_quantity") or 0.0))
    remaining_capacity = max(0.0, capacity - consumed)
    fraction = min(100.0, tranche_pct) / 100.0
    requested_level_quantity = capacity * fraction
    immediate_raw = holding.get("immediately_sellable_quantity")
    immediately_sellable = max(
        0.0,
        float(
            initialized.get("campaign_start_remaining_sellable_quantity")
            if immediate_raw is None
            else immediate_raw or 0.0
        ),
    )
    quantity = min(requested_level_quantity, remaining_capacity, immediately_sellable)
    if not math.isfinite(quantity) or quantity <= 0:
        return {"eligible": False, "reason": "zero_campaign_capacity", "quantity": 0.0}
    return {
        "eligible": True,
        "reason": "progressive_level_open",
        "quantity": quantity,
        "campaign": initialized,
        "campaign_capacity_quantity": capacity,
        "campaign_consumed_quantity": consumed,
        "campaign_remaining_quantity": remaining_capacity,
        "requested_level_quantity": requested_level_quantity,
        "tranche_pct": tranche_pct,
    }


def plan_treasury_accumulation(
    signal: dict[str, Any],
    *,
    free_reserve_quote: float,
    current_price: float,
    config: ProgressiveSwingConfig | None = None,
) -> dict[str, Any]:
    """Size a standalone reserve deployment without creating Swing lifecycle state."""
    resolved = config or ProgressiveSwingConfig()
    if not resolved.treasury_accumulation_enabled:
        return {"eligible": False, "reason": "treasury_accumulation_disabled", "quantity": 0.0}
    tranche_pct = float(signal.get("base_tranche_pct") or signal.get("final_tranche_pct") or 0.0)
    reserve = max(0.0, float(free_reserve_quote))
    price = float(current_price)
    if tranche_pct <= 0 or reserve <= 0:
        return {"eligible": False, "reason": "no_free_vault_reserve", "quantity": 0.0}
    if not math.isfinite(price) or price <= 0:
        return {"eligible": False, "reason": "invalid_market_price", "quantity": 0.0}
    fraction = min(100.0, tranche_pct) / 100.0
    quote_to_spend = reserve * fraction
    quantity = quote_to_spend / (price * (1.0 + resolved.estimated_fee_rate))
    if not math.isfinite(quantity) or quantity <= 0:
        return {"eligible": False, "reason": "zero_accumulation_quantity", "quantity": 0.0}
    return {
        "eligible": True,
        "reason": "free_reserve_deployment",
        "quantity": quantity,
        "quote_to_spend": quote_to_spend,
        "free_reserve_quote": reserve,
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
    quantity_basis = "remaining_asset"
    if origin_side == "sell" and objective == "accumulate_asset":
        quantity_basis = "reusable_quote"
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
        "quantity_basis": quantity_basis,
    }
