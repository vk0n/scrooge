from __future__ import annotations

import math
from collections.abc import Set
from typing import Any

from core.spot_progression import (
    ProgressiveSwingConfig,
    plan_opening_quantity,
    plan_profitable_close,
    plan_treasury_accumulation,
)
from core.spot_swing import calculate_sell_origin_committed_quote, calculate_swing_economics
from core.spot_waiter_cleanup import (
    WaiterCleanupConfig,
    bargain_age_days,
    remaining_unrealized_pnl_pct,
    required_cleanup_level,
    reverse_signal_satisfies,
)

SPOT_ACTION_PHASES = ("profit_close", "cleanup_close", "allocation")


def spot_action_phase(decision: dict[str, Any]) -> str:
    if decision.get("action_type") != "close":
        return "allocation"
    reason = decision.get("reason") if isinstance(decision.get("reason"), dict) else {}
    return "profit_close" if reason.get("close_reason") == "profit_target" else "cleanup_close"


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
    persistent_campaign_fields = (
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
        "last_opening_price",
    )
    accumulation_fields = (
        "accumulation_cycle_id",
        "accumulation_campaign_id",
        "accumulation_active",
        "accumulation_highest_completed_level",
        "accumulation_last_signal_level",
        "accumulation_last_signal_at_ms",
        "last_accumulation_opening_price",
        "accumulation_last_executed_at_ms",
        "accumulation_local_low_price",
        "accumulation_rearm_started_at_ms",
    )
    preserve_campaign = normalized_side == "hold" or prior_side == normalized_side
    return {
        "campaign_id": campaign_id,
        "active_side": active_side,
        "highest_completed_level": highest_level,
        "last_signal_side": normalized_side,
        "last_signal_level": max(0, int(signal_level)),
        "last_signal_at_ms": int(signal_at_ms),
        **{
            field: prior.get(field) if preserve_campaign else None
            for field in persistent_campaign_fields
        },
        **{field: prior.get(field) for field in accumulation_fields},
    }


def transition_spot_accumulation_ladder(
    previous: dict[str, Any] | None,
    *,
    rolling_change_pct: float,
    accumulation_level: int,
    signal_at_ms: int,
    current_price: float,
    new_cycle_id: str,
    rearm_recovery_pct: float = 10.0,
    rearm_signal_above_pct: float = -5.0,
    rearm_confirmation_hours: float = 12.0,
) -> dict[str, Any]:
    """Advance the independent accumulation ladder and recovery-based rearm state."""
    prior = previous or {}
    now_ms = int(signal_at_ms)
    price = float(current_price)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Spot accumulation ladder requires a positive current price.")

    cycle_id = str(
        prior.get("accumulation_cycle_id")
        or prior.get("accumulation_campaign_id")
        or ""
    ).strip() or None
    active = bool(prior.get("accumulation_active")) and cycle_id is not None
    highest_level = max(0, int(prior.get("accumulation_highest_completed_level") or 0))
    last_opening_price = prior.get("last_accumulation_opening_price")
    last_executed_at_ms = prior.get("accumulation_last_executed_at_ms")
    local_low = prior.get("accumulation_local_low_price")
    rearm_started_at_ms = prior.get("accumulation_rearm_started_at_ms")

    try:
        local_low = float(local_low) if local_low is not None else None
    except (TypeError, ValueError):
        local_low = None
    if active:
        if local_low is None:
            try:
                local_low = float(last_opening_price)
            except (TypeError, ValueError):
                local_low = price
        local_low = min(local_low, price)
        recovery_pct = ((price / local_low) - 1.0) * 100.0
        recovery_confirmed = (
            recovery_pct + 1e-12 >= float(rearm_recovery_pct)
            and float(rolling_change_pct) + 1e-12 >= float(rearm_signal_above_pct)
        )
        if recovery_confirmed:
            if rearm_started_at_ms is None:
                rearm_started_at_ms = now_ms
            confirmation_ms = (
                max(0.0, float(rearm_confirmation_hours)) * 60 * 60 * 1000
            )
            if now_ms - int(rearm_started_at_ms) >= confirmation_ms:
                cycle_id = None
                active = False
                highest_level = 0
                last_opening_price = None
                local_low = None
                rearm_started_at_ms = None
        else:
            rearm_started_at_ms = None

    reached_level = max(0, int(accumulation_level))
    if not active and reached_level > 0:
        cycle_id = str(new_cycle_id or "").strip()
        if not cycle_id:
            raise ValueError("A new Spot accumulation cycle requires a stable ID.")
        active = True
        highest_level = 0
        last_opening_price = None
        local_low = price
        rearm_started_at_ms = None

    return {
        **prior,
        "accumulation_cycle_id": cycle_id,
        # Keep the legacy key while the persisted schema is migrated in place.
        "accumulation_campaign_id": cycle_id,
        "accumulation_active": active,
        "accumulation_highest_completed_level": highest_level,
        "accumulation_last_signal_level": reached_level,
        "accumulation_last_signal_at_ms": now_ms,
        "last_accumulation_opening_price": last_opening_price,
        "accumulation_last_executed_at_ms": last_executed_at_ms,
        "accumulation_local_low_price": local_low,
        "accumulation_rearm_started_at_ms": rearm_started_at_ms,
    }


def finalize_spot_strategy_signal(
    signal: dict[str, Any],
    policy: dict[str, Any],
    *,
    execution_enabled: bool,
) -> dict[str, Any]:
    """Apply the production policy gate to a fixed-allocation Spot signal."""
    eligible, eligibility_reason = spot_policy_eligibility(
        policy,
        execution_enabled=execution_enabled,
    )
    return {
        **signal,
        "final_tranche_pct": (
            float(signal.get("base_tranche_pct") or 0.0)
            if signal.get("opportunity") != "hold"
            else 0.0
        ),
        "trading_objective": policy.get("trading_objective"),
        "strategy_eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "evaluated_at_ms": int(signal["current_at_ms"]),
    }


def _with_missing_level_allocation(
    signal: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    reached_level = int(signal.get("level") or 0)
    completed_level = max(0, int(campaign.get("highest_completed_level") or 0))
    if reached_level <= completed_level:
        return signal

    start = completed_level
    stop = reached_level
    allocated = dict(signal)
    allocated["level"] = reached_level
    for series_key, tranche_key in (("base_tranches_pct", "base_tranche_pct"),):
        series = signal.get(series_key)
        if not isinstance(series, (list, tuple)) or len(series) < stop:
            continue
        try:
            tranche = sum(float(value) for value in series[start:stop])
        except (TypeError, ValueError):
            continue
        allocated[tranche_key] = tranche
        allocated["final_tranche_pct"] = tranche
    return allocated


def _with_missing_accumulation_allocation(
    signal: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    reached_level = int(
        signal.get("accumulation_level", signal.get("level")) or 0
    )
    completed_level = max(
        0,
        int(campaign.get("accumulation_highest_completed_level") or 0),
    )
    allocated = dict(signal)
    allocated["level"] = reached_level
    series = signal.get("accumulation_tranches_pct")
    if reached_level <= completed_level or not isinstance(series, (list, tuple)):
        return allocated
    if len(series) < reached_level:
        return allocated
    try:
        tranche = sum(float(value) for value in series[completed_level:reached_level])
    except (TypeError, ValueError):
        return allocated
    allocated["accumulation_tranche_pct"] = tranche
    allocated["final_tranche_pct"] = tranche
    return allocated


def plan_spot_strategy_action(
    signal: dict[str, Any],
    holding: dict[str, Any],
    campaign: dict[str, Any],
    swings: list[dict[str, Any]],
    *,
    available_quote: float,
    free_quote_reserve: float | None = None,
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
        supplied_economics = item.get("economics") if isinstance(item, dict) else None
        economics = (
            supplied_economics
            if isinstance(supplied_economics, dict)
            else calculate_swing_economics(swing, executions, current_price=current_price)
        )
        active_swings.append((swing, economics))
        if str(swing.get("swing_id") or "") in excluded_closes:
            continue
        close_available_quote = available_quote
        if free_quote_reserve is not None and str(swing.get("origin_side") or "").lower() == "sell":
            close_available_quote = (
                max(0.0, float(free_quote_reserve))
                + calculate_sell_origin_committed_quote(swing, economics)
            )
        close = plan_profitable_close(
            swing,
            economics,
            current_price=current_price,
            available_quote_quantity=close_available_quote,
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
                        "profit_threshold_basis": close["profit_threshold_basis"],
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
    objective = str(signal.get("trading_objective") or "").strip().lower()
    new_strategy_decision: dict[str, Any] | None = None
    opening_price_hold: dict[str, Any] | None = None
    if (
        bool(signal.get("strategy_eligible"))
        and opportunity != "hold"
        and campaign.get("active_side") == opportunity
        and level > int(campaign.get("highest_completed_level") or 0)
    ):
        allocation_signal = _with_missing_level_allocation(signal, campaign)
        if opportunity == "buy":
            new_strategy_decision = {
                "action_type": "campaign_only",
                "side": "buy",
                "signal_level": level,
                "requested_quantity": 0.0,
                "reason": {
                    "action_type": "campaign_only",
                    "campaign_reason": "bargain_buy_signal",
                    "signal": opportunity,
                    "signal_level": level,
                    "campaign_id": campaign.get("campaign_id"),
                },
            }
        elif opportunity == "sell":
            previous_opening_price = campaign.get("last_opening_price")
            try:
                previous_opening_price = (
                    float(previous_opening_price)
                    if previous_opening_price is not None
                    else None
                )
            except (TypeError, ValueError):
                previous_opening_price = None
            if (
                previous_opening_price is not None
                and math.isfinite(previous_opening_price)
                and current_price <= previous_opening_price
            ):
                opening_price_hold = {
                    "action_type": "hold",
                    "reason": {
                        "action_type": "hold",
                        "hold_reason": "sell_level_price_not_improved",
                        "signal_level": level,
                        "market_price": current_price,
                        "previous_opening_price": previous_opening_price,
                    },
                }
            else:
                opening = plan_opening_quantity(
                    allocation_signal,
                    holding,
                    campaign,
                    config=resolved_config,
                )
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
                            "base_tranche_pct": allocation_signal.get("base_tranche_pct"),
                            "final_tranche_pct": allocation_signal.get("base_tranche_pct"),
                            "campaign_id": campaign.get("campaign_id"),
                            "minimum_opening_price_exclusive": previous_opening_price,
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

    accumulation_level = int(
        signal.get("accumulation_level", signal.get("level")) or 0
    )
    rolling_change_pct = float(signal.get("rolling_change_pct") or 0.0)
    accumulation_active = bool(
        campaign.get("accumulation_active", rolling_change_pct < 0.0)
    )
    accumulation_decision: dict[str, Any] | None = None
    if (
        new_strategy_decision is None
        and bool(signal.get("strategy_eligible"))
        and objective == "accumulate_asset"
        and rolling_change_pct < 0.0
        and accumulation_active
        and accumulation_level
        > int(campaign.get("accumulation_highest_completed_level") or 0)
    ):
        allocation_signal = _with_missing_accumulation_allocation(signal, campaign)
        signal_at_ms = int(
            signal.get("current_at_ms")
            or signal.get("evaluated_at_ms")
            or campaign.get("accumulation_last_signal_at_ms")
            or 0
        )
        last_executed_at_ms = campaign.get("accumulation_last_executed_at_ms")
        cooldown_ms = max(
            0.0,
            float(signal.get("accumulation_cooldown_hours", 24.0)),
        ) * 60 * 60 * 1000
        completed_accumulation_level = int(
            campaign.get("accumulation_highest_completed_level") or 0
        )
        cooldown_active = (
            completed_accumulation_level == 0
            and last_executed_at_ms is not None
            and signal_at_ms - int(last_executed_at_ms) < cooldown_ms
        )
        if cooldown_active:
            opening_price_hold = {
                "action_type": "hold",
                "reason": {
                    "action_type": "hold",
                    "hold_reason": "accumulation_cooldown",
                    "signal_level": accumulation_level,
                    "cooldown_until_ms": int(last_executed_at_ms + cooldown_ms),
                },
            }
        previous_opening_price = campaign.get("last_accumulation_opening_price")
        try:
            previous_opening_price = (
                float(previous_opening_price)
                if previous_opening_price is not None
                else None
            )
        except (TypeError, ValueError):
            previous_opening_price = None
        minimum_improvement_pct = max(
            0.0,
            float(signal.get("accumulation_min_price_improvement_pct", 3.0)),
        )
        maximum_opening_price = (
            previous_opening_price * (1.0 - minimum_improvement_pct / 100.0)
            if previous_opening_price is not None
            else None
        )
        if (
            not cooldown_active
            and previous_opening_price is not None
            and math.isfinite(previous_opening_price)
            and maximum_opening_price is not None
            and current_price >= maximum_opening_price
        ):
            opening_price_hold = {
                "action_type": "hold",
                "reason": {
                    "action_type": "hold",
                    "hold_reason": "accumulation_price_improvement_not_reached",
                    "signal_level": accumulation_level,
                    "market_price": current_price,
                    "previous_opening_price": previous_opening_price,
                    "minimum_price_improvement_pct": minimum_improvement_pct,
                    "maximum_opening_price_exclusive": maximum_opening_price,
                },
            }
        elif not cooldown_active:
            accumulation_quote = max(0.0, float(available_accumulation_quote or 0.0))
            accumulation = plan_treasury_accumulation(
                allocation_signal,
                free_reserve_quote=accumulation_quote,
                current_price=current_price,
                config=resolved_config,
            )
            if accumulation.get("eligible"):
                accumulation_decision = {
                    "action_type": "accumulate_asset",
                    "side": "buy",
                    "signal_level": accumulation_level,
                    "requested_quantity": float(accumulation["quantity"]),
                    "reason": {
                        "action_type": "accumulate_asset",
                        "signal": "accumulation",
                        "signal_level": accumulation_level,
                        "rolling_change_pct": signal.get("rolling_change_pct"),
                        "accumulation_tranche_pct": allocation_signal.get("accumulation_tranche_pct"),
                        "final_tranche_pct": allocation_signal.get("final_tranche_pct"),
                        "cycle_id": (
                            campaign.get("accumulation_cycle_id")
                            or campaign.get("accumulation_campaign_id")
                            or campaign.get("campaign_id")
                        ),
                        "campaign_id": (
                            campaign.get("accumulation_cycle_id")
                            or campaign.get("accumulation_campaign_id")
                            or campaign.get("campaign_id")
                        ),
                        "maximum_opening_price_exclusive": maximum_opening_price,
                        "minimum_price_improvement_pct": minimum_improvement_pct,
                        "free_reserve_quote": accumulation.get("free_reserve_quote"),
                        "planned_quote_to_spend": accumulation.get("quote_to_spend"),
                    },
                }

    if new_strategy_decision is None:
        new_strategy_decision = accumulation_decision

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
                close_available_quote = available_quote
                if free_quote_reserve is not None:
                    close_available_quote = (
                        max(0.0, float(free_quote_reserve))
                        + calculate_sell_origin_committed_quote(swing, economics)
                    )
                cash_limited_quantity = (
                    max(0.0, float(close_available_quote))
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
    if opening_price_hold is not None:
        return opening_price_hold
    return new_strategy_decision
