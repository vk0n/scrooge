from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean, median
import subprocess
from typing import Any

from backtest.spot_bargain_analysis import build_bargain_analysis
from backtest.spot_engine import SpotBacktestResult
from backtest.spot_report_html import display_spot_report_title, write_spot_backtest_html
from backtest.spot_scenario import scenario_as_dict, write_scenario_snapshot
from shared.spot_progression import policy_sellable_reference
from shared.spot_swing import calculate_swing_economics
from shared.spot_waiter_cleanup import CLEANUP_REASONS, is_cleanup_reason


AGE_BUCKETS = (
    ("under_7_days", 0, 7),
    ("7_to_30_days", 7, 30),
    ("30_to_90_days", 30, 90),
    ("90_to_180_days", 90, 180),
    ("180_plus_days", 180, None),
)


def _maximum_drawdown(values: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, ((value / peak) - 1.0) * 100.0)
    return worst


def _relative_wealth_metrics(equity: list[dict[str, Any]]) -> dict[str, float]:
    ratios = [
        float(item["treasury_value"]) / float(item["hodl_value"])
        for item in equity
        if float(item.get("hodl_value") or 0.0) > 0.0
    ]
    if not ratios:
        return {
            "minimum": 1.0,
            "maximum": 1.0,
            "terminal": 1.0,
            "maximum_drawdown_pct": 0.0,
        }
    start = ratios[0]
    normalized = [value / start for value in ratios]
    return {
        "minimum": min(normalized),
        "maximum": max(normalized),
        "terminal": normalized[-1],
        "maximum_drawdown_pct": _maximum_drawdown(normalized),
    }


def _maximum_concurrent(swings: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for swing in swings:
        events.append((int(swing["opened_at_ms"]), 1))
        if swing.get("closed_at_ms") is not None:
            events.append((int(swing["closed_at_ms"]), -1))
    current = 0
    maximum = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        current += change
        maximum = max(maximum, current)
    return maximum


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _swing_metrics(swings: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [item for item in swings if item["status"] == "closed"]
    opened = [item for item in swings if item["status"] != "closed"]
    profitable = [item for item in closed if float(item["economics"].get("realized_pnl_quote") or 0) > 0]
    losing = [item for item in closed if float(item["economics"].get("realized_pnl_quote") or 0) < 0]
    durations = [float(item["age_seconds"]) for item in closed]
    open_ages = [float(item["age_seconds"]) / 86400.0 for item in opened]
    age_buckets = {
        name: sum(1 for age in open_ages if age >= lower and (upper is None or age < upper))
        for name, lower, upper in AGE_BUCKETS
    }
    fees: dict[str, float] = {}
    for swing in swings:
        for asset, amount in swing["economics"].get("fees_by_asset", {}).items():
            fees[asset] = fees.get(asset, 0.0) + float(amount)
    return {
        "total_opened": len(swings),
        "total_closed": len(closed),
        "still_open": len(opened),
        "total_fills": sum(len(item.get("executions") or []) for item in swings),
        "profitable_closed": len(profitable),
        "losing_closed": len(losing),
        "win_rate_pct": (len(profitable) / len(closed)) * 100 if closed else None,
        "realized_pnl_quote": sum(float(item["economics"].get("realized_pnl_quote") or 0) for item in closed),
        "unrealized_open_pnl_quote": sum(
            float(item["economics"].get("unrealized_pnl_quote") or 0) for item in opened
        ),
        "fees_by_asset": fees,
        "median_duration_hours": median(durations) / 3600.0 if durations else None,
        "average_duration_hours": mean(durations) / 3600.0 if durations else None,
        "oldest_open_days": max(open_ages) if open_ages else None,
        "maximum_concurrent_open": _maximum_concurrent(swings),
        "open_age_buckets": age_buckets,
    }


def _accumulation_metrics(
    accumulations: list[dict[str, Any]],
    swings: list[dict[str, Any]],
) -> dict[str, Any]:
    accumulation_buys = [
        item
        for item in accumulations
        if str(item.get("action_type") or "accumulate_asset").strip().lower() == "accumulate_asset"
        and str(item.get("side") or "buy").strip().lower() == "buy"
    ]
    earned_cash_events = sorted(
        (
            int(swing.get("closed_at_ms") or 0),
            max(0.0, float((swing.get("economics") or {}).get("realized_cash_gain_quote") or 0.0)),
        )
        for swing in swings
        if swing.get("status") == "closed"
        and str(swing.get("origin_side") or "").strip().lower() == "sell"
        and float((swing.get("economics") or {}).get("realized_cash_gain_quote") or 0.0) > 0.0
    )
    earned_cash_generated = sum(amount for _, amount in earned_cash_events)
    earned_cash_available = 0.0
    earned_cash_allocated_by_execution: dict[str, float] = {}
    cash_events = [
        (timestamp_ms, 0, "earned", amount, None)
        for timestamp_ms, amount in earned_cash_events
    ] + [
        (
            int(item.get("timestamp_ms") or 0),
            1,
            "buy",
            max(0.0, float(item.get("deployed_quote_quantity") or 0.0)),
            str(item.get("execution_id") or ""),
        )
        for item in accumulation_buys
    ]
    for _, _, event_type, amount, execution_id in sorted(cash_events):
        if event_type == "earned":
            earned_cash_available += amount
            continue
        allocated = min(amount, earned_cash_available)
        earned_cash_available -= allocated
        if execution_id:
            earned_cash_allocated_by_execution[execution_id] = allocated

    annotated_accumulations = [
        {
            **item,
            "_earned_cash_allocated_quote": earned_cash_allocated_by_execution.get(
                str(item.get("execution_id") or ""),
                0.0,
            ),
        }
        for item in accumulation_buys
    ]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        deployed = sum(float(item.get("deployed_quote_quantity") or 0.0) for item in rows)
        acquired = sum(float(item.get("net_asset_acquired") or 0.0) for item in rows)
        fees: dict[str, float] = {}
        for item in rows:
            fee_asset = str(item.get("fee_asset") or "").strip().upper()
            if fee_asset:
                fees[fee_asset] = fees.get(fee_asset, 0.0) + float(item.get("fee_amount") or 0.0)
        return {
            "count": len(rows),
            "usdt_deployed": deployed,
            "earned_cash_allocated_quote": sum(
                float(item.get("_earned_cash_allocated_quote") or 0.0) for item in rows
            ),
            "net_asset_acquired": acquired,
            "target_growth_quantity": sum(
                float(item.get("target_growth_quantity") or 0.0) for item in rows
            ),
            "average_purchase_price": (
                sum(
                    float(item.get("quote_quantity") or 0.0)
                    for item in rows
                ) / acquired
                if acquired > 1e-12
                else None
            ),
            "fees_by_asset": fees,
        }

    def grouped(field: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in annotated_accumulations:
            key = str(item.get(field) or "unknown")
            buckets.setdefault(key, []).append(item)
        return {key: summarize(rows) for key, rows in sorted(buckets.items())}

    return {
        "overview": summarize(annotated_accumulations),
        "earned_cash_generated_quote": earned_cash_generated,
        "earned_cash_allocated_quote": sum(earned_cash_allocated_by_execution.values()),
        "earned_cash_allocated_pct": (
            sum(earned_cash_allocated_by_execution.values()) / earned_cash_generated * 100.0
            if earned_cash_generated > 1e-12
            else None
        ),
        "per_asset": grouped("asset_symbol"),
        "per_level": grouped("signal_level"),
    }


def _level_metrics(result: SpotBacktestResult, symbol: str) -> dict[str, Any]:
    actions = [
        item
        for item in result.actions
        if item["asset_symbol"] == symbol and item["action_type"] == "open"
    ]
    swing_map = {item["swing_id"]: item for item in result.swings}
    output = {}
    for level in range(1, len(result.scenario.signal.levels_pct) + 1):
        level_actions = [item for item in actions if int(item.get("signal_level") or 0) == level]
        pnl = 0.0
        for action in level_actions:
            swing = swing_map.get(action.get("swing_id"))
            if swing is not None:
                economics = swing["economics"]
                pnl += float(economics.get("realized_pnl_quote") or 0.0)
                pnl += float(economics.get("unrealized_pnl_quote") or 0.0)
        output[f"level_{level}"] = {
            "signals": int(result.signal_level_counts.get(symbol, {}).get(level, 0)),
            "swing_opens": len(level_actions),
            "executed_quantity": sum(float(item.get("executed_quantity") or 0.0) for item in level_actions),
            "executed_notional": sum(float(item.get("executed_value") or 0.0) for item in level_actions),
            "lifecycle_pnl_quote": pnl,
        }
    return output


def _campaign_metrics(result: SpotBacktestResult) -> dict[str, Any]:
    campaigns = result.sell_campaigns
    capacities = [float(item.get("campaign_capacity_quantity") or 0.0) for item in campaigns]
    utilizations = [
        float(item.get("campaign_consumed_quantity") or 0.0) / capacity * 100.0
        if capacity > 0 else 0.0
        for item, capacity in zip(campaigns, capacities)
    ]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row_capacities = [float(item.get("campaign_capacity_quantity") or 0.0) for item in rows]
        row_utilizations = [
            float(item.get("campaign_consumed_quantity") or 0.0) / capacity * 100.0
            if capacity > 0 else 0.0
            for item, capacity in zip(rows, row_capacities)
        ]
        return {
            "count": len(rows),
            "average_capacity_quantity": mean(row_capacities) if row_capacities else 0.0,
            "median_capacity_quantity": median(row_capacities) if row_capacities else 0.0,
            "average_utilization_pct": mean(row_utilizations) if row_utilizations else 0.0,
            "median_utilization_pct": median(row_utilizations) if row_utilizations else 0.0,
            "normal_capacity_campaigns": sum(
                1 for item in rows if item.get("campaign_capacity_mode") == "normal"
            ),
            "full_deploy_campaigns": sum(
                1 for item in rows if item.get("campaign_capacity_mode") == "full_deploy"
            ),
            "interrupted_by_opposite_signal": sum(
                1 for item in rows if item.get("ended_by_opposite_signal")
            ),
            **{
                f"campaigns_reaching_l{level}": sum(
                    1 for item in rows if int(item.get("highest_completed_level") or 0) >= level
                )
                for level in range(1, 5)
            },
        }

    per_asset = {
        symbol: summarize([item for item in campaigns if item.get("asset_symbol") == symbol])
        for symbol in result.scenario.asset_order
    }
    return {
        **summarize(campaigns),
        "average_capacity_quantity": mean(capacities) if capacities else 0.0,
        "median_capacity_quantity": median(capacities) if capacities else 0.0,
        "average_utilization_pct": mean(utilizations) if utilizations else 0.0,
        "median_utilization_pct": median(utilizations) if utilizations else 0.0,
        "per_asset": per_asset,
        "campaigns": [
            {
                **item,
                "campaign_remaining_quantity": max(
                    0.0,
                    float(item.get("campaign_capacity_quantity") or 0.0)
                    - float(item.get("campaign_consumed_quantity") or 0.0),
                ),
                "utilization_pct": (
                    float(item.get("campaign_consumed_quantity") or 0.0)
                    / float(item.get("campaign_capacity_quantity") or 0.0)
                    * 100.0
                    if float(item.get("campaign_capacity_quantity") or 0.0) > 0 else 0.0
                ),
            }
            for item in campaigns
        ],
    }


def _inventory_metrics(result: SpotBacktestResult, symbol: str) -> dict[str, Any]:
    rows = [item for item in result.inventory_history if item["asset_symbol"] == symbol]
    utilizations = [float(item["tradable_inventory_utilization_pct"]) for item in rows]
    distances = [float(item["distance_above_floor_pct"]) for item in rows]
    state = result.assets[symbol]
    reference = policy_sellable_reference(
        state.target_quantity,
        state.scenario.minimum_holding_pct,
    )
    campaigns = [item for item in result.sell_campaigns if item.get("asset_symbol") == symbol]
    sold = sum(
        float(item.get("executed_quantity") or 0.0)
        for item in result.actions
        if item.get("asset_symbol") == symbol
        and item.get("action_type") == "open"
        and item.get("side") == "sell"
    )
    return {
        "average_tradable_inventory_utilization_pct": mean(utilizations) if utilizations else 0.0,
        "maximum_tradable_inventory_utilization_pct": max(utilizations, default=0.0),
        "minimum_distance_above_floor_pct": min(distances, default=0.0),
        "near_floor_threshold_pct": result.scenario.near_floor_pct,
        "time_near_floor_pct": (
            sum(1 for value in distances if value <= result.scenario.near_floor_pct) / len(distances) * 100
            if distances
            else 0.0
        ),
        "maximum_simultaneously_underwater_sell_origin_swings": max(
            (int(item["underwater_sell_origin_swings"]) for item in rows),
            default=0,
        ),
        "maximum_open_buy_capital_tied_quote": max(
            (float(item["open_buy_capital_tied_quote"]) for item in rows),
            default=0.0,
        ),
        "current_target_quantity": state.target_quantity,
        "minimum_holding_pct": state.scenario.minimum_holding_pct,
        "current_policy_sellable_reference": reference,
        "current_remaining_sellable_quantity": state.policy_sellable,
        "remaining_sellable_pct_of_reference": (
            state.policy_sellable / reference * 100.0 if reference > 0 else 0.0
        ),
        "sell_campaign_count": len(campaigns),
        "total_quantity_sold": sold,
    }


def _maximum_portfolio_inventory_metric(result: SpotBacktestResult, field: str) -> float:
    totals: dict[int, float] = {}
    for row in result.inventory_history:
        timestamp_ms = int(row["timestamp_ms"])
        totals[timestamp_ms] = totals.get(timestamp_ms, 0.0) + float(row[field])
    return max(totals.values(), default=0.0)


def _bad_case_metrics(swings: list[dict[str, Any]]) -> dict[str, Any]:
    open_swings = [item for item in swings if item["status"] != "closed"]
    underwater_sells = [
        item
        for item in open_swings
        if item["origin_side"] == "sell"
        and float(item["economics"].get("unrealized_pnl_quote") or 0) < 0
    ]
    open_buys = [item for item in open_swings if item["origin_side"] == "buy"]
    return {
        "underwater_sell_origin_count": len(underwater_sells),
        "quantity_sold_not_restored": sum(
            float(item["economics"].get("remaining_quantity") or 0) for item in underwater_sells
        ),
        "value_required_to_restore": sum(
            float(item["economics"].get("remaining_quantity") or 0)
            * float(item.get("current_market_price") or 0)
            for item in underwater_sells
        ),
        "oldest_underwater_sell_days": max(
            (float(item["age_seconds"]) / 86400.0 for item in underwater_sells),
            default=None,
        ),
        "open_buy_capital_tied_quote": sum(
            float(item["economics"].get("remaining_quantity") or 0)
            * float(item["economics"].get("weighted_opening_price") or 0)
            for item in open_buys
        ),
        "open_buy_mark_to_market_pnl": sum(
            float(item["economics"].get("unrealized_pnl_quote") or 0) for item in open_buys
        ),
        "oldest_open_buy_days": max(
            (float(item["age_seconds"]) / 86400.0 for item in open_buys),
            default=None,
        ),
    }


def _asset_capital_performance(
    asset_config: Any,
    swings: list[dict[str, Any]],
    accumulations: list[dict[str, Any]],
    *,
    initial_price: float,
    final_price: float,
) -> dict[str, float | None]:
    accumulated_asset = 0.0
    accumulated_cash = 0.0
    open_bargain_pnl = 0.0
    for swing in swings:
        economics = swing["economics"]
        if economics.get("status") == "closed":
            accumulated_asset += float(economics.get("realized_net_asset_change") or 0.0)
            accumulated_cash += float(economics.get("realized_cash_gain_quote") or 0.0)
            continue
        open_bargain_pnl += (
            float(economics.get("realized_pnl_quote") or 0.0)
            + float(economics.get("unrealized_pnl_quote") or 0.0)
        )

    initial_quantity = float(asset_config.quantity)
    reserve_deployment_asset = sum(
        float(item.get("net_asset_acquired") or 0.0) for item in accumulations
    )
    settled_quantity = max(
        0.0,
        initial_quantity + accumulated_asset + reserve_deployment_asset,
    )
    initial_capital = (
        initial_quantity * float(initial_price)
        if initial_quantity > 0
        else 0.0
    )
    effective_cost_basis = initial_capital - accumulated_cash
    effective_entry_cost = (
        effective_cost_basis / settled_quantity
        if settled_quantity > 1e-12
        else None
    )
    market_gain = settled_quantity * final_price - initial_capital
    floating_gain = market_gain + accumulated_cash
    total_gain = floating_gain + open_bargain_pnl
    return {
        "initial_quantity": initial_quantity,
        "initial_price": float(initial_price),
        "initial_capital": initial_capital,
        "accumulated_asset_quantity": accumulated_asset,
        "reserve_deployment_asset_quantity": reserve_deployment_asset,
        "accumulated_cash_gain": accumulated_cash,
        "open_bargain_pnl": open_bargain_pnl,
        "settled_quantity": settled_quantity,
        "effective_cost_basis": effective_cost_basis,
        "effective_entry_cost": effective_entry_cost,
        "market_gain": market_gain,
        "floating_gain": floating_gain,
        "total_gain": total_gain,
        "total_gain_pct": (
            total_gain / initial_capital * 100.0
            if initial_capital > 0
            else None
        ),
    }


def _asset_recovery_metrics(
    asset_config: Any,
    swings: list[dict[str, Any]],
    *,
    final_quantity: float,
    final_price: float,
    fee_rate: float,
) -> dict[str, Any]:
    """Estimate end-of-run inventory after buyback funded only by open-sell cash."""
    initial_quantity = float(asset_config.quantity)
    open_sell_quantity = 0.0
    committed_cash = 0.0
    hypothetical_buyback_quantity = 0.0
    for swing in swings:
        if swing.get("status") == "closed" or swing.get("origin_side") != "sell":
            continue
        economics = swing.get("economics") or {}
        remaining = max(0.0, float(economics.get("remaining_quantity") or 0.0))
        if remaining <= 1e-12:
            continue
        quote_fees = float((economics.get("fees_by_asset") or {}).get("USDT", 0.0))
        swing_committed_cash = max(
            0.0,
            float(economics.get("opening_quote_quantity") or 0.0)
            - float(economics.get("closing_quote_quantity") or 0.0)
            - quote_fees,
        )
        open_sell_quantity += remaining
        committed_cash += swing_committed_cash
        hypothetical_buyback_quantity += min(
            remaining,
            swing_committed_cash / (final_price * (1.0 + fee_rate))
            if final_price > 0
            else 0.0,
        )

    effective_quantity = final_quantity + hypothetical_buyback_quantity
    actual_pct = (
        final_quantity / initial_quantity * 100.0
        if initial_quantity > 1e-12
        else None
    )
    effective_pct = (
        effective_quantity / initial_quantity * 100.0
        if initial_quantity > 1e-12
        else None
    )
    return {
        "initial_quantity": initial_quantity,
        "actual_final_quantity": final_quantity,
        "actual_final_quantity_pct": actual_pct,
        "open_sell_quantity": open_sell_quantity,
        "open_sell_committed_cash": committed_cash,
        "hypothetical_buyback_quantity": hypothetical_buyback_quantity,
        "effective_final_quantity": effective_quantity,
        "effective_final_quantity_pct": effective_pct,
        "open_sell_buyback_coverage_pct": (
            hypothetical_buyback_quantity / open_sell_quantity * 100.0
            if open_sell_quantity > 1e-12
            else 100.0
        ),
    }


def _cleanup_accounting_rows(cleanup_swings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleanup_accounting_rows: list[dict[str, Any]] = []
    for swing in cleanup_swings:
        cleanup_execution = next(
            (
                execution
                for execution in reversed(swing.get("executions") or [])
                if is_cleanup_reason((execution.get("reason") or {}).get("close_reason"))
            ),
            None,
        )
        if cleanup_execution is None:
            continue
        economics = calculate_swing_economics(
            swing,
            swing.get("executions") or [],
            current_price=swing.get("current_market_price"),
        )
        origin_side = str(swing.get("origin_side") or "").lower()
        quote_symbol = str(swing.get("quote_symbol") or "USDT").upper()
        opening_cash_flow = 0.0
        closing_cash_flow = 0.0
        for execution in swing.get("executions") or []:
            side = str(execution.get("side") or "").lower()
            quote_quantity = float(
                execution.get("quote_quantity")
                or float(execution.get("quantity") or 0.0)
                * float(execution.get("price") or 0.0)
            )
            quote_flow = quote_quantity if side == "sell" else -quote_quantity
            if str(execution.get("fee_asset") or "").upper() == quote_symbol:
                quote_flow -= float(execution.get("fee_amount") or 0.0)
            if side == origin_side:
                opening_cash_flow += quote_flow
            else:
                closing_cash_flow += quote_flow
        cash_change = float(economics.get("realized_cash_gain_quote") or 0.0)
        asset_quantity_change = float(economics.get("realized_net_asset_change") or 0.0)
        close_price = float(cleanup_execution.get("price") or 0.0)
        asset_value_change = asset_quantity_change * close_price
        realized_pnl = float(economics.get("realized_pnl_quote") or 0.0)
        inventory_residual_pnl = float(
            economics.get("terminal_residual_pnl_quote") or 0.0
        )
        unrecovered_quantity = float(economics.get("unrecovered_quantity") or 0.0)
        cleanup_accounting_rows.append(
            {
                "asset_symbol": str(swing["asset_symbol"]),
                "reason": str(swing.get("close_reason") or "unknown"),
                "cash_change_quote": cash_change,
                "committed_proceeds_quote": (
                    max(0.0, opening_cash_flow) if origin_side == "sell" else 0.0
                ),
                "repurchase_spend_quote": (
                    max(0.0, -closing_cash_flow) if origin_side == "sell" else 0.0
                ),
                "reserve_deployed_quote": max(0.0, -cash_change),
                "cash_released_quote": max(0.0, cash_change),
                "asset_quantity_change": asset_quantity_change,
                "asset_value_change_quote": asset_value_change,
                "unrecovered_quantity": unrecovered_quantity,
                "unrecovered_market_value_quote": unrecovered_quantity * close_price,
                "restored_inventory_pnl_quote": realized_pnl - inventory_residual_pnl,
                "inventory_residual_pnl_quote": inventory_residual_pnl,
                "economic_pnl_quote": realized_pnl,
                "reported_realized_pnl_quote": realized_pnl,
            }
        )
    return cleanup_accounting_rows


def _cleanup_accounting_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cash_changes = [float(row["cash_change_quote"]) for row in rows]
    asset_changes = [float(row["asset_value_change_quote"]) for row in rows]
    economic_results = [float(row["economic_pnl_quote"]) for row in rows]
    restored_results = [float(row["restored_inventory_pnl_quote"]) for row in rows]
    residual_results = [float(row["inventory_residual_pnl_quote"]) for row in rows]
    repurchase_spends = [
        float(row["repurchase_spend_quote"])
        for row in rows
        if float(row["repurchase_spend_quote"]) > 0
    ]
    all_quantities = {
        symbol: sum(
            float(row["asset_quantity_change"])
            for row in rows
            if row["asset_symbol"] == symbol
        )
        for symbol in sorted({str(row["asset_symbol"]) for row in rows})
    }
    quantities = {
        symbol: quantity
        for symbol, quantity in all_quantities.items()
        if abs(quantity) > 1e-12
    }
    inventory_deficits = {
        symbol: sum(
            float(row["unrecovered_quantity"])
            for row in rows
            if row["asset_symbol"] == symbol
        )
        for symbol in sorted({str(row["asset_symbol"]) for row in rows})
    }
    inventory_deficits = {
        symbol: quantity
        for symbol, quantity in inventory_deficits.items()
        if quantity > 1e-12
    }
    return {
        "economic_pnl_quote": sum(economic_results),
        "economic_loss_quote": sum(value for value in economic_results if value < 0),
        "economic_profit_quote": sum(value for value in economic_results if value > 0),
        "cash_change_quote": sum(cash_changes),
        "cash_loss_quote": sum(value for value in cash_changes if value < 0),
        "cash_gain_quote": sum(value for value in cash_changes if value > 0),
        "committed_proceeds_quote": sum(
            float(row["committed_proceeds_quote"]) for row in rows
        ),
        "repurchase_spend_quote": sum(float(row["repurchase_spend_quote"]) for row in rows),
        "repurchase_count": len(repurchase_spends),
        "average_repurchase_spend_quote": (
            mean(repurchase_spends) if repurchase_spends else None
        ),
        "median_repurchase_spend_quote": (
            median(repurchase_spends) if repurchase_spends else None
        ),
        "maximum_repurchase_spend_quote": max(repurchase_spends, default=None),
        "reserve_deployed_quote": sum(float(row["reserve_deployed_quote"]) for row in rows),
        "cash_released_quote": sum(float(row["cash_released_quote"]) for row in rows),
        "net_reserve_deployed_quote": max(0.0, -sum(cash_changes)),
        "net_cash_released_quote": max(0.0, sum(cash_changes)),
        "asset_value_change_quote": sum(asset_changes),
        "asset_value_loss_quote": sum(value for value in asset_changes if value < 0),
        "asset_value_gain_quote": sum(value for value in asset_changes if value > 0),
        "asset_quantity_change_by_asset": quantities,
        "restored_inventory_pnl_quote": sum(restored_results),
        "restored_inventory_loss_quote": sum(
            value for value in restored_results if value < 0
        ),
        "restored_inventory_profit_quote": sum(
            value for value in restored_results if value > 0
        ),
        "inventory_residual_pnl_quote": sum(residual_results),
        "inventory_residual_loss_quote": sum(
            value for value in residual_results if value < 0
        ),
        "inventory_residual_profit_quote": sum(
            value for value in residual_results if value > 0
        ),
        "inventory_deficit_quantity_by_asset": inventory_deficits,
        "inventory_deficit_market_value_quote": sum(
            float(row["unrecovered_market_value_quote"]) for row in rows
        ),
        "reconciliation_delta_quote": sum(
            float(row["restored_inventory_pnl_quote"])
            + float(row["inventory_residual_pnl_quote"])
            - float(row["economic_pnl_quote"])
            for row in rows
        ),
    }


def _waiter_cleanup_metrics(result: SpotBacktestResult) -> dict[str, Any]:
    cleanup_swings = [
        swing for swing in result.swings if is_cleanup_reason(swing.get("close_reason"))
    ]
    cleanup_accounting_rows = _cleanup_accounting_rows(cleanup_swings)
    cleanup_accounting = _cleanup_accounting_summary(cleanup_accounting_rows)
    cleanup_outcomes: list[dict[str, Any]] = []
    for swing in result.swings:
        executions: list[dict[str, Any]] = []
        prior_realized = 0.0
        for execution in swing.get("executions") or []:
            executions.append(execution)
            economics = calculate_swing_economics(
                swing,
                executions,
                current_price=swing.get("current_market_price"),
            )
            current_realized = float(economics.get("realized_pnl_quote") or 0.0)
            reason = (execution.get("reason") or {}).get("close_reason")
            if is_cleanup_reason(reason):
                cleanup_outcomes.append(
                    {
                        "asset_symbol": swing["asset_symbol"],
                        "reason": reason,
                        "realized_pnl_quote": current_realized - prior_realized,
                        "fee_amount": float(execution.get("fee_amount") or 0.0),
                        "fee_asset": str(execution.get("fee_asset") or "UNKNOWN"),
                    }
                )
            prior_realized = current_realized
    cleanup_results = [float(item["realized_pnl_quote"]) for item in cleanup_outcomes]
    cleanup_executions = [
        execution
        for execution in result.executions
        if is_cleanup_reason((execution.get("reason") or {}).get("close_reason"))
    ]
    cleanup_fees: dict[str, float] = {}
    for execution in cleanup_executions:
        fee_asset = str(execution.get("fee_asset") or "UNKNOWN")
        cleanup_fees[fee_asset] = cleanup_fees.get(fee_asset, 0.0) + float(
            execution.get("fee_amount") or 0.0
        )

    reason_breakdown: dict[str, dict[str, Any]] = {}
    for reason in sorted(CLEANUP_REASONS):
        reason_swings = [swing for swing in cleanup_swings if swing.get("close_reason") == reason]
        reason_outcomes = [item for item in cleanup_outcomes if item["reason"] == reason]
        reason_accounting = _cleanup_accounting_summary(
            [item for item in cleanup_accounting_rows if item["reason"] == reason]
        )
        reason_breakdown[reason] = {
            "closes": len(reason_swings),
            "cleanup_executions": len(reason_outcomes),
            "realized_pnl_quote": sum(
                float(item["realized_pnl_quote"]) for item in reason_outcomes
            ),
            **reason_accounting,
            "closing_fees_by_asset": {
                asset: sum(
                    float(item["fee_amount"])
                    for item in reason_outcomes
                    if item["fee_asset"] == asset
                )
                for asset in sorted({str(item["fee_asset"]) for item in reason_outcomes})
            },
        }

    capacity_holds = [
        action
        for action in result.actions
        if action.get("action_type") == "hold"
        and (action.get("reason") or {}).get("hold_reason") == "max_open_bargains_per_asset"
    ]
    pressure_cleanup_actions = [
        action
        for action in result.actions
        if action.get("action_type") == "close"
        and bool((action.get("reason") or {}).get("capacity_pressure"))
        and is_cleanup_reason((action.get("reason") or {}).get("close_reason"))
    ]
    capacity_cleanup_actions = [
        action
        for action in pressure_cleanup_actions
        if (action.get("reason") or {}).get("close_reason") == "capacity_cleanup"
    ]
    open_swings = [swing for swing in result.swings if swing.get("status") != "closed"]
    open_ages = [float(swing.get("age_seconds") or 0.0) / 86400.0 for swing in open_swings]
    underwater_open = [
        swing
        for swing in open_swings
        if (
            float(swing["economics"].get("realized_pnl_quote") or 0.0)
            + float(swing["economics"].get("unrealized_pnl_quote") or 0.0)
        ) < 0
    ]
    open_buy_swings = [swing for swing in open_swings if swing.get("origin_side") == "buy"]
    open_sell_swings = [swing for swing in open_swings if swing.get("origin_side") == "sell"]
    open_sell_quantity_by_asset = {
        symbol: sum(
            float(swing["economics"].get("remaining_quantity") or 0.0)
            for swing in open_sell_swings
            if swing["asset_symbol"] == symbol
        )
        for symbol in result.scenario.asset_order
        if any(swing["asset_symbol"] == symbol for swing in open_sell_swings)
    }
    inventory_by_asset: dict[str, list[int]] = {}
    for row in result.inventory_history:
        inventory_by_asset.setdefault(str(row["asset_symbol"]), []).append(
            int(row.get("open_bargain_count") or 0)
        )

    per_asset: dict[str, Any] = {}
    for symbol in result.scenario.asset_order:
        asset_cleanup = [swing for swing in cleanup_swings if swing["asset_symbol"] == symbol]
        asset_open = [swing for swing in open_swings if swing["asset_symbol"] == symbol]
        counts = inventory_by_asset.get(symbol, [])
        per_asset[symbol] = {
            "cleanup_closes": len(asset_cleanup),
            "realized_cleanup_pnl_quote": sum(
                float(item["realized_pnl_quote"])
                for item in cleanup_outcomes
                if item["asset_symbol"] == symbol
            ),
            "accounting": _cleanup_accounting_summary(
                [item for item in cleanup_accounting_rows if item["asset_symbol"] == symbol]
            ),
            "open_bargains_prevented_by_cap": sum(
                1
                for action in capacity_holds + pressure_cleanup_actions
                if action.get("asset_symbol") == symbol
            ),
            "capacity_forced_hold_cycles": sum(
                1 for action in capacity_holds if action.get("asset_symbol") == symbol
            ),
            "average_open_bargain_count": mean(counts) if counts else 0.0,
            "maximum_open_bargain_count": max(counts, default=0),
            "open_bargains_at_end": len(asset_open),
            "underwater_open_bargains_at_end": sum(
                1
                for swing in asset_open
                if (
                    float(swing["economics"].get("realized_pnl_quote") or 0.0)
                    + float(swing["economics"].get("unrealized_pnl_quote") or 0.0)
                ) < 0
            ),
        }

    return {
        "enabled": result.scenario.waiter_cleanup.enabled,
        "policy": {
            "max_open_bargains_per_asset": (
                result.scenario.waiter_cleanup.max_open_bargains_per_asset
            ),
            "deep_loss": {
                "min_age_days": result.scenario.waiter_cleanup.deep_loss_min_age_days,
                "unrealized_pnl_pct": (
                    result.scenario.waiter_cleanup.deep_loss_unrealized_pnl_pct
                ),
                "required_reverse_level": (
                    result.scenario.waiter_cleanup.deep_loss_required_reverse_level
                ),
            },
            "aging": [
                {
                    "min_age_days": rule.min_age_days,
                    "required_reverse_level": rule.required_reverse_level,
                }
                for rule in result.scenario.waiter_cleanup.aging_rules
            ],
            "capacity_cleanup": {
                "enabled": result.scenario.waiter_cleanup.capacity_cleanup_enabled,
                "min_age_days": result.scenario.waiter_cleanup.capacity_cleanup_min_age_days,
            },
        },
        "cleanup_closes_total": len(cleanup_swings),
        "cleanup_attempts_total": len(cleanup_executions),
        "realized_cleanup_loss_quote": sum(value for value in cleanup_results if value < 0),
        "realized_cleanup_profit_quote": sum(value for value in cleanup_results if value > 0),
        "accounting": cleanup_accounting,
        "average_cleanup_pnl_quote": mean(cleanup_results) if cleanup_results else None,
        "median_cleanup_pnl_quote": median(cleanup_results) if cleanup_results else None,
        "max_cleanup_loss_quote": min(cleanup_results, default=0.0),
        "cleanup_fees_by_asset": cleanup_fees,
        "by_reason": reason_breakdown,
        "capacity": {
            "open_bargains_prevented_by_cap": len(capacity_holds) + len(pressure_cleanup_actions),
            "capacity_forced_hold_cycles": len(capacity_holds),
            "capacity_cleanup_actions": len(capacity_cleanup_actions),
            "cleanup_actions_under_capacity_pressure": len(pressure_cleanup_actions),
        },
        "open_bargains": {
            "at_end": len(open_swings),
            "underwater_at_end": len(underwater_open),
            "age_30_plus": sum(age >= 30 for age in open_ages),
            "age_60_plus": sum(age >= 60 for age in open_ages),
            "age_90_plus": sum(age >= 90 for age in open_ages),
            "age_180_plus": sum(age >= 180 for age in open_ages),
            "average_age_days": mean(open_ages) if open_ages else None,
            "median_age_days": median(open_ages) if open_ages else None,
            "p90_age_days": _percentile(open_ages, 0.9),
            "oldest_age_days": max(open_ages, default=None),
        },
        "capital_lock": {
            "open_buy_origin_quote": sum(
                float(swing["economics"].get("remaining_opening_quote_quantity") or 0.0)
                for swing in open_buy_swings
            ),
            "open_sell_origin_asset_quantity_by_asset": open_sell_quantity_by_asset,
            "value_required_to_restore_sell_inventory": sum(
                float(swing["economics"].get("remaining_quantity") or 0.0)
                * float(swing.get("current_market_price") or 0.0)
                for swing in open_sell_swings
            ),
            "final_shared_usdt": result.final_usdt,
            "minimum_shared_usdt": result.minimum_usdt,
        },
        "per_asset": per_asset,
    }


def build_spot_backtest_report(result: SpotBacktestResult) -> dict[str, Any]:
    final_point = result.equity[-1]
    final_value = float(final_point["treasury_value"])
    hodl_final = float(final_point["hodl_value"])
    initial_value = result.starting_value
    cash_values = [float(item["shared_usdt"]) for item in result.equity]
    cash_reserved = [float(item["shared_usdt_reserved"]) for item in result.equity]
    cash_available = [float(item["shared_usdt_available"]) for item in result.equity]
    cash_retained = [float(item.get("shared_usdt_retained") or 0.0) for item in result.equity]
    cash_spendable = [
        float(item.get("shared_usdt_spendable", item["shared_usdt_available"]))
        for item in result.equity
    ]
    strong_limit = result.scenario.starting_usdt * (
        1.0 - result.scenario.strong_cash_utilization_pct / 100.0
    )
    per_asset: dict[str, Any] = {}
    for asset_config in result.scenario.assets:
        symbol = asset_config.symbol
        state = result.assets[symbol]
        swings = [item for item in result.swings if item["asset_symbol"] == symbol]
        accumulations = [item for item in result.accumulations if item["asset_symbol"] == symbol]
        capital_performance = _asset_capital_performance(
            asset_config,
            swings,
            accumulations,
            initial_price=result.initial_prices[symbol],
            final_price=result.final_prices[symbol],
        )
        recovery_metrics = _asset_recovery_metrics(
            asset_config,
            swings,
            final_quantity=state.quantity,
            final_price=result.final_prices[symbol],
            fee_rate=result.scenario.execution.fee_rate,
        )
        ratchets = [item for item in result.target_history if item["asset_symbol"] == symbol]
        swing_ratchets = [
            item for item in ratchets if item.get("ratchet_source") != "treasury_accumulation"
        ]
        initial_floor = asset_config.target_holding * asset_config.minimum_holding_pct / 100.0
        final_market_value = state.quantity * result.final_prices[symbol]
        objective_metrics: dict[str, Any] = {}
        if asset_config.trading_objective == "accumulate_asset":
            objective_metrics = {
                "initial_quantity": asset_config.quantity,
                "final_quantity": state.quantity,
                "net_asset_accumulated": state.quantity - asset_config.quantity,
                "initial_target": asset_config.target_holding,
                "final_target": state.target_quantity,
                "total_target_ratchet": state.target_quantity - asset_config.target_holding,
                "successful_ratchets": len(ratchets),
                "bargain_target_ratchets": len(swing_ratchets),
                "accumulation_buys": len(accumulations),
                "initial_protected_floor": initial_floor,
                "final_protected_floor": state.protected_floor,
                "bargain_asset_gain": sum(
                    float(item["applied_gain_quantity"]) for item in swing_ratchets
                ),
                "reserve_deployment_asset_gain": sum(
                    float(item["net_asset_acquired"]) for item in accumulations
                ),
                "usdt_deployed_into_accumulation": sum(
                    float(item["deployed_quote_quantity"]) for item in accumulations
                ),
            }
        elif asset_config.trading_objective == "accumulate_cash":
            objective_metrics = {
                "realized_quote_cash_generated": sum(
                    float(item["economics"].get("realized_cash_gain_quote") or 0)
                    for item in swings
                    if item["status"] == "closed"
                ),
                "completed_cash_cycles": sum(1 for item in swings if item["status"] == "closed"),
                "open_swing_exposure": sum(
                    float(item["economics"].get("unrealized_pnl_quote") or 0)
                    for item in swings
                    if item["status"] != "closed"
                ),
            }
        per_asset[symbol] = {
            "starting": {
                "quantity": asset_config.quantity,
                "entry_cost": result.initial_prices[symbol],
                "entry_cost_source": "start_candle_open",
                "binance_quantity": asset_config.binance_quantity,
                "cold_storage_quantity": asset_config.cold_storage_quantity,
                "target_holding": asset_config.target_holding,
                "minimum_holding_pct": asset_config.minimum_holding_pct,
                "protected_floor": initial_floor,
                "trading_objective": asset_config.trading_objective,
            },
            "market": {
                "starting_price": result.initial_prices[symbol],
                "ending_price": result.final_prices[symbol],
                "market_return_pct": (
                    (result.final_prices[symbol] / result.initial_prices[symbol]) - 1.0
                ) * 100.0,
            },
            "trading": {
                **_swing_metrics(swings),
                "levels": _level_metrics(result, symbol),
            },
            "inventory": _inventory_metrics(result, symbol),
            "bad_cases": _bad_case_metrics(swings),
            "asset_recovery": recovery_metrics,
            "objective_metrics": objective_metrics,
            "capital_performance": capital_performance,
            "final": {
                "quantity": state.quantity,
                "average_cost": state.average_cost,
                "effective_entry_cost": capital_performance["effective_entry_cost"],
                "target_holding": state.target_quantity,
                "protected_floor": state.protected_floor,
                "binance_quantity": state.binance_quantity,
                "cold_storage_quantity": state.cold_storage_quantity,
                "unassigned_quantity": state.unassigned_quantity,
                "market_value": final_market_value,
                "realized_portfolio_pnl_on_known_basis": state.realized_portfolio_pnl,
                "unrealized_portfolio_pnl_on_known_basis": state.unrealized_portfolio_pnl(
                    result.final_prices[symbol]
                ),
            },
            "benchmark": {
                "hodl_value": asset_config.quantity * result.final_prices[symbol],
                "scrooge_asset_value": final_market_value,
            },
        }
    known_initial_asset_capital = sum(
        float(item["capital_performance"]["initial_capital"] or 0.0)
        for item in per_asset.values()
    )
    initial_invested_capital = result.scenario.starting_usdt + known_initial_asset_capital
    total_gain_on_initial_capital = final_value - initial_invested_capital
    final_allocations = {
        symbol: (
            result.assets[symbol].quantity * result.final_prices[symbol] / final_value * 100
            if final_value > 0
            else 0.0
        )
        for symbol in result.scenario.asset_order
    }
    final_allocations["USDT"] = result.final_usdt / final_value * 100 if final_value > 0 else 0.0
    portfolio_levels = {}
    for level in range(1, len(result.scenario.signal.levels_pct) + 1):
        key = f"level_{level}"
        rows = [item["trading"]["levels"][key] for item in per_asset.values()]
        portfolio_levels[key] = {
            field: sum(float(row.get(field) or 0.0) for row in rows)
            for field in (
                "signals", "swing_opens", "executed_quantity",
                "executed_notional", "lifecycle_pnl_quote",
            )
        }
    recovery_pct_values = [
        float(item["asset_recovery"]["effective_final_quantity_pct"])
        for item in per_asset.values()
        if item["asset_recovery"]["effective_final_quantity_pct"] is not None
    ]
    recovery_weights = {
        symbol: (
            float(per_asset[symbol]["starting"]["quantity"])
            * float(per_asset[symbol]["market"]["starting_price"])
        )
        for symbol in result.scenario.asset_order
    }
    recovery_weight_total = sum(recovery_weights.values())
    weighted_recovery_pct = (
        sum(
            recovery_weights[symbol]
            * float(per_asset[symbol]["asset_recovery"]["effective_final_quantity_pct"])
            for symbol in result.scenario.asset_order
        )
        / recovery_weight_total
        if recovery_weight_total > 0
        else None
    )
    relative_wealth = _relative_wealth_metrics(result.equity)
    return {
        "scenario": {
            "name": result.scenario.name,
            "start": result.scenario.start.isoformat(),
            "end": result.scenario.end.isoformat(),
            "interval": result.scenario.interval,
            "asset_order": list(result.scenario.asset_order),
            "data_source": result.data_source,
        },
        "portfolio": {
            "starting_treasury_value": initial_value,
            "final_treasury_value": final_value,
            "hodl_final_treasury_value": hodl_final,
            "difference_vs_hodl": final_value - hodl_final,
            "total_return_pct": ((final_value / initial_value) - 1.0) * 100 if initial_value > 0 else 0.0,
            "hodl_return_pct": ((hodl_final / initial_value) - 1.0) * 100 if initial_value > 0 else 0.0,
            "initial_invested_capital": initial_invested_capital,
            "total_gain_on_initial_capital": total_gain_on_initial_capital,
            "total_gain_on_initial_capital_pct": (
                total_gain_on_initial_capital / initial_invested_capital * 100.0
                if initial_invested_capital > 0
                else None
            ),
            "edge_vs_hodl_pct_points": (
                ((final_value - hodl_final) / initial_value) * 100.0
                if initial_value > 0
                else None
            ),
            "edge_vs_hodl_relative_pct": (
                (final_value / hodl_final - 1.0) * 100.0
                if hodl_final > 0
                else None
            ),
            "maximum_treasury_drawdown_pct": _maximum_drawdown(
                [float(item["treasury_value"]) for item in result.equity]
            ),
            "maximum_hodl_drawdown_pct": _maximum_drawdown(
                [float(item["hodl_value"]) for item in result.equity]
            ),
            "relative_wealth": relative_wealth,
            "final_allocation_pct": final_allocations,
        },
        "shared_usdt": {
            "starting": result.scenario.starting_usdt,
            "ending": result.final_usdt,
            "minimum": result.minimum_usdt,
            "maximum": result.maximum_usdt,
            "average": mean(cash_values),
            "ending_reserved": cash_reserved[-1],
            "maximum_reserved": max(cash_reserved, default=0.0),
            "average_reserved": mean(cash_reserved),
            "ending_available": cash_available[-1],
            "minimum_available": min(cash_available, default=0.0),
            "average_available": mean(cash_available),
            "ending_retained": cash_retained[-1],
            "maximum_retained": max(cash_retained, default=0.0),
            "ending_spendable": cash_spendable[-1],
            "minimum_spendable": min(cash_spendable, default=0.0),
            "strong_utilization_definition": (
                f"Shared USDT at or below {100 - result.scenario.strong_cash_utilization_pct:g}% "
                "of its starting balance."
            ),
            "time_strongly_utilized_pct": (
                sum(1 for value in cash_values if value <= strong_limit) / len(cash_values) * 100
                if cash_values and result.scenario.starting_usdt > 0
                else 0.0
            ),
        },
        "success_metrics": {
            "edge_vs_hodl": {
                "quote": final_value - hodl_final,
                "strategy_return_pct": (
                    (final_value / initial_value - 1.0) * 100.0
                    if initial_value > 0
                    else None
                ),
                "hodl_return_pct": (
                    (hodl_final / initial_value - 1.0) * 100.0
                    if initial_value > 0
                    else None
                ),
                "difference_pct_points": (
                    (final_value - hodl_final) / initial_value * 100.0
                    if initial_value > 0
                    else None
                ),
                "relative_outperformance_pct": (
                    (final_value / hodl_final - 1.0) * 100.0
                    if hodl_final > 0
                    else None
                ),
            },
            "free_reserve": {
                "quote": cash_available[-1],
                "pct_of_initial_invested_capital": (
                    cash_available[-1] / initial_invested_capital * 100.0
                    if initial_invested_capital > 0
                    else None
                ),
                "committed_quote": cash_reserved[-1],
                "retained_quote": cash_retained[-1],
                "spendable_quote": cash_spendable[-1],
                "retention_pct": result.scenario.free_cash_retention_pct,
                "retention_accruals": len(result.cash_retentions),
                "total_usdt": result.final_usdt,
            },
            "asset_recovery": {
                "average_effective_quantity_pct": (
                    mean(recovery_pct_values) if recovery_pct_values else None
                ),
                "weighted_effective_quantity_pct": weighted_recovery_pct,
                "per_asset": {
                    symbol: item["asset_recovery"]
                    for symbol, item in per_asset.items()
                },
            },
        },
        "swings": _swing_metrics(result.swings),
        "treasury_accumulation": _accumulation_metrics(result.accumulations, result.swings),
        "sell_campaigns": _campaign_metrics(result),
        "sell_openings_by_level": portfolio_levels,
        "waiter_cleanup": _waiter_cleanup_metrics(result),
        "bargain_analysis": build_bargain_analysis(result.swings),
        "bad_cases": {
            **_bad_case_metrics(result.swings),
            "maximum_simultaneously_underwater_sell_origin_swings": int(
                _maximum_portfolio_inventory_metric(
                    result,
                    "underwater_sell_origin_swings",
                )
            ),
            "maximum_open_buy_capital_tied_quote": _maximum_portfolio_inventory_metric(
                result,
                "open_buy_capital_tied_quote",
            ),
        },
        "per_asset": per_asset,
        "rejected_orders": len(result.rejections),
    }


def build_monthly_results(result: SpotBacktestResult) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.equity:
        month = datetime.fromtimestamp(int(row["timestamp_ms"]) / 1000, tz=UTC).strftime("%Y-%m")
        grouped.setdefault(month, []).append(row)
    output: list[dict[str, Any]] = []
    for month, rows in sorted(grouped.items()):
        start = rows[0]
        end = rows[-1]
        opened = sum(
            1
            for item in result.swings
            if datetime.fromtimestamp(int(item["opened_at_ms"]) / 1000, tz=UTC).strftime("%Y-%m") == month
        )
        closed = sum(
            1
            for item in result.swings
            if item.get("closed_at_ms") is not None
            and datetime.fromtimestamp(int(item["closed_at_ms"]) / 1000, tz=UTC).strftime("%Y-%m") == month
        )
        output.append(
            {
                "month": month,
                "treasury_start_value": start["treasury_value"],
                "treasury_end_value": end["treasury_value"],
                "monthly_return_pct": (
                    (float(end["treasury_value"]) / float(start["treasury_value"])) - 1.0
                ) * 100 if float(start["treasury_value"]) > 0 else 0.0,
                "hodl_monthly_return_pct": (
                    (float(end["hodl_value"]) / float(start["hodl_value"])) - 1.0
                ) * 100 if float(start["hodl_value"]) > 0 else 0.0,
                "difference_vs_hodl": float(end["treasury_value"]) - float(end["hodl_value"]),
                "closed_swings": closed,
                "opened_swings": opened,
                "ending_open_swings": end["open_swings"],
                "ending_usdt": end["shared_usdt"],
            }
        )
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(_json_ready(payload), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_spot_backtest_artifacts(result: SpotBacktestResult, output_dir: str | Path | None = None) -> dict[str, Any]:
    target = Path(output_dir or result.scenario.output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    report = build_spot_backtest_report(result)
    monthly = build_monthly_results(result)
    reproducibility = {
        "scenario": scenario_as_dict(result.scenario),
        "strategy_code_revision": _git_revision(),
        "timing": (
            "Evaluate after candle N closes using data through N; execute every eligible action in "
            "portfolio-wide priority phases at candle N+1 open with configured slippage, refreshing "
            "simulated state after every fill. "
            "Warm-up candles never trade and the final close cannot create an unfillable action."
        ),
        "rolling_24h_reference": "Candle close exactly 24 hours before the evaluated candle close.",
        "missing_candles": "Fail the run; no interpolation or forward fill.",
        "cross_asset_order": {
            "rule": (
                "all profit closes run before all cleanup closes; only then may openings, reserve "
                "accumulations, and campaign-only actions run. Ties preserve asset-symbol order, "
                "matching the live signal executor"
            ),
            "symbols": list(result.scenario.asset_order),
        },
        "quantization": "Current cached Binance Spot filters, not historical filter versions.",
        "execution_model": (
            "Deterministic next-candle-open market-action batch plus configured fee and slippage."
        ),
    }
    _write_json(target / "summary.json", report)
    _write_json(target / "scenario.resolved.json", reproducibility)
    write_scenario_snapshot(result.scenario, target / "scenario.resolved.yaml")
    _write_json(target / "per_asset_summary.json", report["per_asset"])
    _write_json(target / "waiter_cleanup.json", report["waiter_cleanup"])
    _write_json(target / "swings.json", result.swings)
    _write_json(target / "treasury_accumulation.json", report["treasury_accumulation"])
    _write_json(target / "sell_campaigns.json", report["sell_campaigns"])
    _write_json(target / "final_state.json", {
        "shared_usdt": result.final_usdt,
        "assets": {
            symbol: {
                "quantity": state.quantity,
                "binance_quantity": state.binance_quantity,
                "cold_storage_quantity": state.cold_storage_quantity,
                "unassigned_quantity": state.unassigned_quantity,
                "average_cost": state.average_cost,
                "effective_entry_cost": report["per_asset"][symbol]["capital_performance"][
                    "effective_entry_cost"
                ],
                "initial_capital": report["per_asset"][symbol]["capital_performance"][
                    "initial_capital"
                ],
                "total_gain": report["per_asset"][symbol]["capital_performance"]["total_gain"],
                "target_holding": state.target_quantity,
                "protected_floor": state.protected_floor,
            }
            for symbol, state in result.assets.items()
        },
    })
    _write_csv(target / "equity.csv", result.equity)
    _write_csv(target / "monthly.csv", monthly)
    _write_csv(target / "executions.csv", result.executions)
    _write_csv(target / "treasury_accumulation.csv", result.accumulations)
    _write_csv(target / "sell_campaigns.csv", report["sell_campaigns"]["campaigns"])
    _write_csv(target / "signals.csv", result.signals)
    _write_csv(target / "actions.csv", result.actions)
    _write_csv(target / "target_history.csv", result.target_history)
    _write_csv(target / "inventory.csv", result.inventory_history)
    _write_csv(target / "rejections.csv", result.rejections)
    _write_csv(
        target / "waiter_cleanup_reasons.csv",
        [
            {"reason": reason, **metrics}
            for reason, metrics in report["waiter_cleanup"]["by_reason"].items()
        ],
    )

    portfolio = report["portfolio"]
    swing_metrics = report["swings"]
    bargain_analysis = report["bargain_analysis"]
    bargain_overview = bargain_analysis["overview"]
    cleanup_metrics = report["waiter_cleanup"]
    accumulation_metrics = report["treasury_accumulation"]["overview"]
    campaign_metrics = report["sell_campaigns"]
    success_metrics = report["success_metrics"]
    edge_metrics = success_metrics["edge_vs_hodl"]
    reserve_metrics = success_metrics["free_reserve"]
    recovery_metrics = success_metrics["asset_recovery"]
    closure_rate = bargain_overview["closure_rate_pct"]
    median_duration = bargain_overview["median_duration_hours"]
    p90_duration = bargain_overview["duration_p90_hours"]
    report_title = display_spot_report_title(
        result.scenario.name,
        result.scenario.start.isoformat(),
        result.scenario.end.isoformat(),
    )
    markdown = [
        f"# {report_title}",
        "",
        "## Portfolio",
        "",
        f"- Period: {result.scenario.start.isoformat()} to {result.scenario.end.isoformat()}",
        f"- Starting Treasury Value: ${portfolio['starting_treasury_value']:,.2f}",
        f"- Final Treasury Value: ${portfolio['final_treasury_value']:,.2f}",
        f"- HODL Final Treasury Value: ${portfolio['hodl_final_treasury_value']:,.2f}",
        f"- Difference vs HODL: ${portfolio['difference_vs_hodl']:,.2f}",
        f"- Total Return: {portfolio['total_return_pct']:.2f}%",
        f"- HODL Return: {portfolio['hodl_return_pct']:.2f}%",
        f"- Edge vs HODL: {edge_metrics['difference_pct_points']:.2f} pp ({edge_metrics['relative_outperformance_pct']:.2f}% relative)",
        f"- Free Reserve: ${reserve_metrics['quote']:,.2f} ({reserve_metrics['pct_of_initial_invested_capital']:.2f}% of initial invested capital)",
        f"- Average Nominal Asset Recovery: {recovery_metrics['average_effective_quantity_pct']:.2f}%",
        f"- Maximum Treasury Drawdown: {portfolio['maximum_treasury_drawdown_pct']:.2f}%",
        "",
        "## Bargains",
        "",
        f"- Opened: {swing_metrics['total_opened']}",
        f"- Closed: {swing_metrics['total_closed']}",
        f"- Still open: {swing_metrics['still_open']}",
        f"- Closed Bargain PnL: ${swing_metrics['realized_pnl_quote']:,.2f}",
        f"- Unrealized open PnL: ${swing_metrics['unrealized_open_pnl_quote']:,.2f}",
        f"- Lifecycle PnL: ${bargain_overview['net_pnl_quote']:,.2f}",
        f"- Closure rate: {closure_rate:.2f}%" if closure_rate is not None else "- Closure rate: N/A",
        (
            f"- Median closed duration: {median_duration:.2f} hours"
            if median_duration is not None
            else "- Median closed duration: N/A"
        ),
        (
            f"- P90 closed duration: {p90_duration:.2f} hours"
            if p90_duration is not None
            else "- P90 closed duration: N/A"
        ),
        f"- Underwater open Bargains: {bargain_analysis['risk']['underwater_open_count']}",
        f"- Underwater open lifecycle PnL: ${bargain_analysis['risk']['underwater_open_pnl_quote']:,.2f}",
        "",
        "## Treasury Accumulation",
        "",
        f"- Accumulation buys: {accumulation_metrics['count']}",
        f"- USDT deployed: ${accumulation_metrics['usdt_deployed']:,.2f}",
        f"- Net asset acquired: {accumulation_metrics['net_asset_acquired']:,.8f}",
        f"- Target growth: {accumulation_metrics['target_growth_quantity']:,.8f}",
        "",
        "## SELL Campaign Capacity",
        "",
        f"- Campaigns: {campaign_metrics['count']}",
        f"- Average utilization: {campaign_metrics['average_utilization_pct']:.2f}%",
        f"- Median utilization: {campaign_metrics['median_utilization_pct']:.2f}%",
        f"- Normal-capacity campaigns: {campaign_metrics['normal_capacity_campaigns']}",
        f"- Full-deploy campaigns: {campaign_metrics['full_deploy_campaigns']}",
        f"- Interrupted by opposite signal: {campaign_metrics['interrupted_by_opposite_signal']}",
        "",
        "## Waiter Cleanup",
        "",
        f"- Enabled: {'yes' if cleanup_metrics['enabled'] else 'no'}",
        f"- Cleanup closes: {cleanup_metrics['cleanup_closes_total']}",
        f"- Cleanup net PnL: ${cleanup_metrics['accounting']['economic_pnl_quote']:,.2f}",
        (
            "- Restored inventory PnL: "
            f"${cleanup_metrics['accounting']['restored_inventory_pnl_quote']:,.2f}"
        ),
        (
            "- Unrestored inventory PnL: "
            f"${cleanup_metrics['accounting']['inventory_residual_pnl_quote']:,.2f}"
        ),
        (
            "- Cumulative cleanup BUY volume (turnover, not capital at once): "
            f"${cleanup_metrics['accounting']['repurchase_spend_quote']:,.2f}"
        ),
        (
            "- Cleanup BUY count / average / maximum: "
            f"{cleanup_metrics['accounting']['repurchase_count']} / "
            f"${cleanup_metrics['accounting']['average_repurchase_spend_quote'] or 0:,.2f} / "
            f"${cleanup_metrics['accounting']['maximum_repurchase_spend_quote'] or 0:,.2f}"
        ),
        (
            "- Net reserve deployed: "
            f"${cleanup_metrics['accounting']['net_reserve_deployed_quote']:,.2f}"
        ),
        (
            "- Inventory deficit market value at close: "
            f"${cleanup_metrics['accounting']['inventory_deficit_market_value_quote']:,.2f}"
        ),
        (
            "- Inventory deficit quantity: "
            + ", ".join(
                f"{symbol} {quantity:,.8f}"
                for symbol, quantity in cleanup_metrics["accounting"][
                    "inventory_deficit_quantity_by_asset"
                ].items()
            )
        ),
        f"- Gross cleanup losses: ${cleanup_metrics['realized_cleanup_loss_quote']:,.2f}",
        f"- Gross cleanup gains: ${cleanup_metrics['realized_cleanup_profit_quote']:,.2f}",
        (
            "- Open Bargains prevented by cap: "
            f"{cleanup_metrics['capacity']['open_bargains_prevented_by_cap']}"
        ),
        f"- Open Bargains at end: {cleanup_metrics['open_bargains']['at_end']}",
        f"- Underwater open Bargains: {cleanup_metrics['open_bargains']['underwater_at_end']}",
        f"- 90+ day open Bargains: {cleanup_metrics['open_bargains']['age_90_plus']}",
        "",
        "This is a strategy backtest, not an order-book or microstructure simulation.",
    ]
    (target / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_spot_backtest_html(
        target / "report.html",
        report,
        result.equity,
        monthly,
        result.swings,
    )
    return {"output_dir": str(target), "report": report, "monthly": monthly}
