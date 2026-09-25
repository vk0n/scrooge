from __future__ import annotations

import math
from collections.abc import Set
from typing import Any

from shared.spot_progression import (
    ProgressiveSwingConfig,
    plan_opening_quantity,
    plan_profitable_close,
    plan_treasury_accumulation,
)
from shared.spot_sizing import IndicatorSizingConfig, apply_indicator_sizing
from shared.spot_swing import calculate_swing_economics
from shared.spot_waiter_cleanup import (
    WaiterCleanupConfig,
    bargain_age_days,
    remaining_unrealized_pnl_pct,
    required_cleanup_level,
    reverse_signal_satisfies,
)


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
        campaign_id = prior.get("campaign_id")
        active_side = prior_side
        highest_level = int(prior.get("highest_completed_level") or 0)
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
    sizing_fields = (
        "campaign_start_target_quantity",
        "campaign_start_minimum_holding_pct",
        "campaign_start_policy_sellable_reference",
        "campaign_start_remaining_sellable_quantity",
        "remaining_sellable_ratio_at_start",
        "campaign_capacity_pct",
        "full_deploy_threshold_pct",
        "campaign_capacity_quantity",
        "campaign_consumed_quantity",
        "campaign_capacity_mode",
    )
    preserve_sizing = normalized_side == "hold" or prior_side == normalized_side
    return {
        "campaign_id": campaign_id,
        "active_side": active_side,
        "highest_completed_level": highest_level,
        "last_signal_side": normalized_side,
        "last_signal_level": max(0, int(signal_level)),
        "last_signal_at_ms": int(signal_at_ms),
        **{field: prior.get(field) if preserve_sizing else None for field in sizing_fields},
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
    available_accumulation_quote: float | None = None,
    config: ProgressiveSwingConfig | None = None,
    cleanup_config: WaiterCleanupConfig | None = None,
    excluded_close_swing_ids: Set[str] | None = None,
) -> dict[str, Any] | None:
    """Choose one profit close, waiter cleanup, opening, or capacity hold."""
    current_price = float(signal.get("current_price") or holding.get("market_price") or 0.0)
    if not math.isfinite(current_price) or current_price <= 0:
        return None
    resolved_config = config or ProgressiveSwingConfig()
    resolved_cleanup = cleanup_config or WaiterCleanupConfig()
    excluded_closes = excluded_close_swing_ids or set()
    close_candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    active_swings: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in swings:
        swing = item.get("swing") if isinstance(item.get("swing"), dict) else item
        executions = item.get("executions") if isinstance(item.get("executions"), list) else []
        if swing.get("source") != "strategy" or swing.get("status") not in {"open", "partially_closed"}:
            continue
        economics = calculate_swing_economics(swing, executions, current_price=current_price)
        active_swings.append((swing, economics))
        if str(swing.get("swing_id") or "") in excluded_closes:
            continue
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
                        "close_reason": "profit_target",
                        "favorable_move_pct": close["favorable_move_pct"],
                        "close_profit_pct": close["close_profit_pct"],
                        "weighted_opening_price": close["opening_price"],
                        "market_price": current_price,
                        "remaining_quantity": economics["remaining_quantity"],
                        "quantity_basis": close.get("quantity_basis", "remaining_asset"),
                    },
                },
            )
        )
    if close_candidates:
        close_candidates.sort(key=lambda item: item[:3])
        return close_candidates[0][3]

    opportunity = str(signal.get("opportunity") or "hold").strip().lower()
    level = int(signal.get("level") or 0)
    new_strategy_decision: dict[str, Any] | None = None
    if (
        bool(signal.get("strategy_eligible"))
        and opportunity != "hold"
        and campaign.get("active_side") == opportunity
        and level > int(campaign.get("highest_completed_level") or 0)
    ):
        objective = str(signal.get("trading_objective") or "").strip().lower()
        if opportunity == "buy" and objective == "accumulate_cash":
            new_strategy_decision = {
                "action_type": "campaign_only",
                "side": "buy",
                "signal_level": level,
                "requested_quantity": 0.0,
                "reason": {
                    "action_type": "campaign_only",
                    "campaign_reason": "accumulate_cash_buy_signal",
                    "signal": opportunity,
                    "signal_level": level,
                    "campaign_id": campaign.get("campaign_id"),
                },
            }
        elif opportunity == "buy" and objective == "accumulate_asset":
            # Reserve deployment fails closed when a caller has not projected free reserve.
            accumulation_quote = max(0.0, float(available_accumulation_quote or 0.0))
            accumulation = plan_treasury_accumulation(
                signal,
                free_reserve_quote=accumulation_quote,
                current_price=current_price,
                config=resolved_config,
            )
            if accumulation.get("eligible"):
                new_strategy_decision = {
                    "action_type": "accumulate_asset",
                    "side": "buy",
                    "signal_level": level,
                    "requested_quantity": float(accumulation["quantity"]),
                    "reason": {
                        "action_type": "accumulate_asset",
                        "signal": opportunity,
                        "signal_level": level,
                        "rolling_change_pct": signal.get("rolling_change_pct"),
                        "base_tranche_pct": signal.get("base_tranche_pct"),
                        "sizing_modifier": signal.get("sizing_modifier"),
                        "final_tranche_pct": signal.get("final_tranche_pct"),
                        "indicator_assessment": signal.get("indicator_assessment"),
                        "indicator_context": signal.get("indicator_context"),
                        "campaign_id": campaign.get("campaign_id"),
                        "free_reserve_quote": accumulation.get("free_reserve_quote"),
                        "planned_quote_to_spend": accumulation.get("quote_to_spend"),
                    },
                }
        elif opportunity == "sell":
            opening = plan_opening_quantity(signal, holding, campaign, config=resolved_config)
            if opening.get("eligible"):
                quantity = min(
                    float(opening["quantity"]),
                    float(holding.get("immediately_sellable_quantity") or 0.0),
                )
            else:
                quantity = 0.0
            if math.isfinite(quantity) and quantity > 0:
                campaign_snapshot = opening.get("campaign") or campaign
                new_strategy_decision = {
                    "action_type": "open",
                    "side": "sell",
                    "signal_level": level,
                    "requested_quantity": quantity,
                    "reason": {
                        "action_type": "open",
                        "signal": opportunity,
                        "signal_level": level,
                        "rolling_change_pct": signal.get("rolling_change_pct"),
                        "base_tranche_pct": signal.get("base_tranche_pct"),
                        "sizing_modifier": signal.get("sizing_modifier"),
                        "final_tranche_pct": signal.get("base_tranche_pct"),
                        "indicator_assessment": signal.get("indicator_assessment"),
                        "indicator_context": signal.get("indicator_context"),
                        "campaign_id": campaign.get("campaign_id"),
                        "campaign_start_target_quantity": campaign_snapshot.get("campaign_start_target_quantity"),
                        "campaign_start_minimum_holding_pct": campaign_snapshot.get("campaign_start_minimum_holding_pct"),
                        "campaign_start_policy_sellable_reference": campaign_snapshot.get("campaign_start_policy_sellable_reference"),
                        "campaign_start_remaining_sellable_quantity": campaign_snapshot.get("campaign_start_remaining_sellable_quantity"),
                        "remaining_sellable_ratio_at_start": campaign_snapshot.get("remaining_sellable_ratio_at_start"),
                        "campaign_capacity_pct": campaign_snapshot.get("campaign_capacity_pct"),
                        "full_deploy_threshold_pct": campaign_snapshot.get("full_deploy_threshold_pct"),
                        "campaign_capacity_mode": campaign_snapshot.get("campaign_capacity_mode"),
                        "campaign_capacity_quantity": opening.get("campaign_capacity_quantity"),
                        "campaign_consumed_quantity": opening.get("campaign_consumed_quantity"),
                        "campaign_remaining_quantity": opening.get("campaign_remaining_quantity"),
                        "level_allocation_pct": opening.get("tranche_pct"),
                        "requested_level_quantity": opening.get("requested_level_quantity"),
                    },
                }

    at_capacity = (
        resolved_cleanup.enabled
        and len(active_swings) >= resolved_cleanup.max_open_bargains_per_asset
    )
    capacity_pressure = (
        at_capacity
        and new_strategy_decision is not None
        and new_strategy_decision.get("action_type") == "open"
    )
    now_ms = int(signal.get("evaluated_at_ms") or signal.get("current_at_ms") or 0)
    cleanup_candidates: list[tuple[int, str, dict[str, Any]]] = []
    if resolved_cleanup.enabled and opportunity in {"buy", "sell"} and level > 0 and now_ms > 0:
        for swing, economics in active_swings:
            if str(swing.get("swing_id") or "") in excluded_closes:
                continue
            age_days = bargain_age_days(swing, now_ms=now_ms)
            pnl_pct = remaining_unrealized_pnl_pct(economics)
            scheduled = required_cleanup_level(
                swing,
                now_ms=now_ms,
                unrealized_pnl_pct=pnl_pct,
                config=resolved_cleanup,
            )
            scheduled_matches = scheduled is not None and reverse_signal_satisfies(
                origin_side=str(swing.get("origin_side") or ""),
                signal_side=opportunity,
                signal_level=level,
                required_level=int(scheduled["required_reverse_level"]),
            )
            capacity_matches = (
                capacity_pressure
                and resolved_cleanup.capacity_cleanup_enabled
                and age_days >= resolved_cleanup.capacity_cleanup_min_age_days
                and reverse_signal_satisfies(
                    origin_side=str(swing.get("origin_side") or ""),
                    signal_side=opportunity,
                    signal_level=level,
                    required_level=1,
                )
            )
            if not scheduled_matches and not capacity_matches:
                continue
            if scheduled_matches and scheduled["cleanup_reason"] == "deep_loss_cleanup":
                cleanup_reason = "deep_loss_cleanup"
                required_level = int(scheduled["required_reverse_level"])
            elif capacity_matches:
                cleanup_reason = "capacity_cleanup"
                required_level = 1
            else:
                cleanup_reason = str(scheduled["cleanup_reason"])
                required_level = int(scheduled["required_reverse_level"])
            close_side = "sell" if swing.get("origin_side") == "buy" else "buy"
            quantity = float(economics.get("remaining_quantity") or 0.0)
            quantity_basis = "remaining_asset"
            if close_side == "buy":
                cash_limited_quantity = (
                    max(0.0, float(available_quote))
                    / (current_price * (1.0 + resolved_config.estimated_fee_rate))
                )
                if cash_limited_quantity + 1e-12 < quantity:
                    quantity_basis = "available_quote"
                quantity = min(quantity, cash_limited_quantity)
            elif holding.get("immediately_sellable_quantity") is not None:
                quantity = min(
                    quantity,
                    max(0.0, float(holding.get("immediately_sellable_quantity") or 0.0)),
                )
            if not math.isfinite(quantity) or quantity <= 0:
                continue
            reason = {
                "action_type": "close",
                "close_reason": cleanup_reason,
                "required_cleanup_level": required_level,
                "actual_reverse_signal_level": level,
                "reverse_signal": opportunity,
                "age_days": age_days,
                "unrealized_pnl_before_cleanup": economics.get("unrealized_pnl_quote"),
                "unrealized_pnl_pct_before_cleanup": pnl_pct,
                "capacity_pressure": capacity_pressure,
                "market_price": current_price,
                "weighted_opening_price": economics.get("weighted_opening_price"),
                "remaining_quantity": economics.get("remaining_quantity"),
                "remaining_opening_quote_quantity": economics.get(
                    "remaining_opening_quote_quantity"
                ),
                "quantity_basis": quantity_basis,
            }
            cleanup_candidates.append(
                (
                    int(swing.get("opened_at_ms") or 0),
                    str(swing.get("swing_id") or ""),
                    {
                        "action_type": "close",
                        "side": close_side,
                        "swing_id": swing["swing_id"],
                        "requested_quantity": quantity,
                        "reason": reason,
                    },
                )
            )
    if cleanup_candidates:
        cleanup_candidates.sort(key=lambda item: item[:2])
        return cleanup_candidates[0][2]
    if capacity_pressure:
        return {
            "action_type": "hold",
            "reason": {
                "action_type": "hold",
                "hold_reason": "max_open_bargains_per_asset",
                "open_bargains": len(active_swings),
                "max_open_bargains_per_asset": resolved_cleanup.max_open_bargains_per_asset,
                "prevented_side": new_strategy_decision["side"],
                "prevented_signal_level": new_strategy_decision["signal_level"],
            },
        }
    return new_strategy_decision
