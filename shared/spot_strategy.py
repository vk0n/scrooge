from __future__ import annotations

import math
from typing import Any

from shared.spot_progression import ProgressiveSwingConfig, plan_opening_quantity, plan_profitable_close
from shared.spot_sizing import IndicatorSizingConfig, apply_indicator_sizing
from shared.spot_swing import calculate_swing_economics


def spot_policy_eligibility(policy: dict[str, Any], *, execution_enabled: bool) -> tuple[bool, str]:
    if not execution_enabled:
        return False, "execution_disabled"
    objective = str(policy.get("trading_objective") or "").strip().lower()
    if objective not in {"accumulate_cash", "accumulate_asset"}:
        return False, "trading_objective_unset"
    try:
        minimum_holding_pct = float(policy.get("minimum_holding_pct", 100.0))
    except (TypeError, ValueError):
        return False, "invalid_policy"
    if minimum_holding_pct >= 100.0:
        return False, "fully_protected"
    return True, "eligible"


def transition_spot_strategy_campaign(
    previous: dict[str, Any] | None,
    *,
    opportunity: str,
    signal_level: int,
    signal_at_ms: int,
    new_campaign_id: str,
) -> dict[str, Any]:
    """Apply the live progressive-level campaign transition without persistence."""
    normalized_side = str(opportunity or "hold").strip().lower()
    if normalized_side not in {"hold", "buy", "sell"}:
        raise ValueError("Spot strategy opportunity must be hold, buy, or sell.")
    prior = previous or {}
    prior_side = str(prior.get("active_side") or "").strip().lower() or None
    if normalized_side == "hold":
        campaign_id = None
        active_side = None
        highest_level = 0
    elif prior_side != normalized_side:
        campaign_id = str(new_campaign_id or "").strip()
        if not campaign_id:
            raise ValueError("A new Spot strategy campaign requires a stable ID.")
        active_side = normalized_side
        highest_level = 0
    else:
        campaign_id = prior.get("campaign_id")
        active_side = normalized_side
        highest_level = int(prior.get("highest_completed_level") or 0)
    return {
        "campaign_id": campaign_id,
        "active_side": active_side,
        "highest_completed_level": highest_level,
        "last_signal_level": max(0, int(signal_level)),
        "last_signal_at_ms": int(signal_at_ms),
    }


def finalize_spot_strategy_signal(
    signal: dict[str, Any],
    indicator_context: dict[str, Any] | None,
    policy: dict[str, Any],
    *,
    execution_enabled: bool,
    sizing_config: IndicatorSizingConfig | None = None,
    indicator_error: str | None = None,
) -> dict[str, Any]:
    """Apply the production sizing and policy gate to an economic Spot signal."""
    sized = apply_indicator_sizing(
        signal,
        indicator_context,
        config=sizing_config,
        indicator_error=indicator_error,
    )
    eligible, eligibility_reason = spot_policy_eligibility(
        policy,
        execution_enabled=execution_enabled,
    )
    return {
        **sized,
        "trading_objective": policy.get("trading_objective"),
        "strategy_eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "evaluated_at_ms": int(sized["current_at_ms"]),
    }


def plan_spot_strategy_action(
    signal: dict[str, Any],
    holding: dict[str, Any],
    campaign: dict[str, Any],
    swings: list[dict[str, Any]],
    *,
    available_quote: float,
    config: ProgressiveSwingConfig | None = None,
) -> dict[str, Any] | None:
    """Choose at most one action, prioritizing the best profitable close."""
    current_price = float(signal.get("current_price") or holding.get("market_price") or 0.0)
    if not math.isfinite(current_price) or current_price <= 0:
        return None
    resolved_config = config or ProgressiveSwingConfig()
    close_candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for item in swings:
        swing = item.get("swing") if isinstance(item.get("swing"), dict) else item
        executions = item.get("executions") if isinstance(item.get("executions"), list) else []
        if swing.get("source") != "strategy" or swing.get("status") not in {"open", "partially_closed"}:
            continue
        economics = calculate_swing_economics(swing, executions, current_price=current_price)
        close = plan_profitable_close(
            swing,
            economics,
            current_price=current_price,
            available_quote_quantity=available_quote,
            config=resolved_config,
        )
        if not close.get("eligible"):
            continue
        close_candidates.append(
            (
                -float(close["favorable_move_pct"]),
                int(swing.get("opened_at_ms") or 0),
                str(swing.get("swing_id") or ""),
                {
                    "action_type": "close",
                    "side": close["side"],
                    "swing_id": swing["swing_id"],
                    "requested_quantity": close["quantity"],
                    "reason": {
                        "action_type": "close",
                        "close_reason": "strategy_profit",
                        "favorable_move_pct": close["favorable_move_pct"],
                        "close_profit_pct": close["close_profit_pct"],
                        "weighted_opening_price": close["opening_price"],
                        "market_price": current_price,
                        "remaining_quantity": economics["remaining_quantity"],
                    },
                },
            )
        )
    if close_candidates:
        close_candidates.sort(key=lambda item: item[:3])
        return close_candidates[0][3]

    opportunity = str(signal.get("opportunity") or "hold").strip().lower()
    level = int(signal.get("level") or 0)
    if not bool(signal.get("strategy_eligible")) or opportunity == "hold":
        return None
    if campaign.get("active_side") != opportunity or level <= int(campaign.get("highest_completed_level") or 0):
        return None

    opening = plan_opening_quantity(signal, holding)
    if not opening.get("eligible"):
        return None
    quantity = float(opening["quantity"])
    if opportunity == "sell":
        quantity = min(quantity, float(holding.get("immediately_sellable_quantity") or 0.0))
    else:
        cash_fraction = min(100.0, float(signal.get("final_tranche_pct") or 0.0)) / 100.0
        quantity = min(quantity, max(0.0, float(available_quote)) * cash_fraction / current_price)
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    return {
        "action_type": "open",
        "side": opportunity,
        "signal_level": level,
        "requested_quantity": quantity,
        "reason": {
            "action_type": "open",
            "signal": opportunity,
            "signal_level": level,
            "rolling_change_pct": signal.get("rolling_change_pct"),
            "base_tranche_pct": signal.get("base_tranche_pct"),
            "sizing_modifier": signal.get("sizing_modifier"),
            "final_tranche_pct": signal.get("final_tranche_pct"),
            "indicator_assessment": signal.get("indicator_assessment"),
            "campaign_id": campaign.get("campaign_id"),
            "strategy_capacity": opening.get("strategic_capacity"),
        },
    }
