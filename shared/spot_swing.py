from __future__ import annotations

import math
from typing import Any

SWING_OBJECTIVES = frozenset({"accumulate_cash", "accumulate_asset"})
SWING_SOURCES = frozenset({"manual", "strategy"})
SWING_STATUSES = frozenset({"open", "partially_closed", "accepting_loss", "closed"})
SWING_SIDES = frozenset({"buy", "sell"})


class SpotSwingDomainError(ValueError):
    pass


def _number(value: Any, *, field: str, positive: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SpotSwingDomainError(f"{field} must be numeric.") from exc
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        qualifier = "greater than zero" if positive else "finite"
        raise SpotSwingDomainError(f"{field} must be {qualifier}.")
    return numeric


def _execution_unit_price(execution: dict[str, Any], quantity: float) -> float:
    quote_quantity = execution.get("quote_quantity")
    if quote_quantity is not None:
        normalized_quote = _number(quote_quantity, field="Execution quote quantity")
        if normalized_quote > 0:
            return normalized_quote / quantity
    return _number(execution.get("price"), field="Execution price", positive=True)


def calculate_swing_economics(
    swing: dict[str, Any],
    executions: list[dict[str, Any]],
    *,
    current_price: float | None = None,
) -> dict[str, Any]:
    """Calculate one Swing from its own executions, independent of portfolio cost basis."""
    origin_side = str(swing.get("origin_side") or "").strip().lower()
    if origin_side not in SWING_SIDES:
        raise SpotSwingDomainError("Swing origin side must be buy or sell.")
    closing_side = "sell" if origin_side == "buy" else "buy"
    objective = str(swing.get("trading_objective") or "").strip().lower() or None
    if objective is not None and objective not in SWING_OBJECTIVES:
        raise SpotSwingDomainError("Swing objective must be accumulate_cash or accumulate_asset.")
    asset_symbol = str(swing.get("asset_symbol") or "").strip().upper()
    quote_symbol = str(swing.get("quote_symbol") or "USDT").strip().upper() or "USDT"
    ordered = sorted(
        executions,
        key=lambda item: (
            int(item.get("executed_at_ms") or 0),
            str(item.get("execution_id") or ""),
        ),
    )

    opening_lots: list[dict[str, float]] = []
    opening_quantity = 0.0
    opening_quote = 0.0
    closing_quantity = 0.0
    closing_quote = 0.0
    realized_gross = 0.0
    realized_quote_fees = 0.0
    net_quote_cash_flow = 0.0
    net_asset_flow = 0.0
    opening_inventory_quantity = 0.0
    closing_inventory_quantity = 0.0
    fees_by_asset: dict[str, float] = {}
    epsilon = 0.0000000001

    for execution in ordered:
        side = str(execution.get("side") or "").strip().lower()
        if side not in SWING_SIDES:
            raise SpotSwingDomainError("Execution side must be buy or sell.")
        quantity = _number(execution.get("quantity"), field="Execution quantity", positive=True)
        unit_price = _execution_unit_price(execution, quantity)
        fee_amount = 0.0
        fee_asset = str(execution.get("fee_asset") or "").strip().upper()
        if execution.get("fee_amount") is not None:
            fee_amount = _number(execution.get("fee_amount"), field="Execution fee")
            if fee_amount < 0:
                raise SpotSwingDomainError("Execution fee cannot be negative.")
        if fee_amount and fee_asset:
            fees_by_asset[fee_asset] = fees_by_asset.get(fee_asset, 0.0) + fee_amount
        quote_flow = quantity * unit_price
        net_quote_cash_flow += quote_flow if side == "sell" else -quote_flow
        if fee_asset == quote_symbol:
            net_quote_cash_flow -= fee_amount
        asset_flow = quantity if side == "buy" else -quantity
        if fee_asset == asset_symbol:
            asset_flow -= fee_amount
        net_asset_flow += asset_flow

        if side == origin_side:
            opening_quantity += quantity
            opening_quote += quantity * unit_price
            opening_inventory_quantity += -asset_flow if origin_side == "sell" else asset_flow
            opening_lots.append(
                {
                    "remaining": quantity,
                    "unit_price": unit_price,
                    "quote_fee_per_unit": fee_amount / quantity if fee_asset == quote_symbol else 0.0,
                }
            )
            continue

        unmatched = sum(lot["remaining"] for lot in opening_lots)
        if unmatched <= epsilon and opening_quantity <= epsilon:
            raise SpotSwingDomainError("A Swing cannot close before its first opening execution.")
        closing_quantity += quantity
        closing_quote += quantity * unit_price
        closing_inventory_quantity += asset_flow if origin_side == "sell" else -asset_flow

        quantity_to_match = quantity
        matched_total = 0.0
        for lot in opening_lots:
            if quantity_to_match <= epsilon:
                break
            matched = min(lot["remaining"], quantity_to_match)
            if origin_side == "sell":
                realized_gross += matched * (lot["unit_price"] - unit_price)
            else:
                realized_gross += matched * (unit_price - lot["unit_price"])
            realized_quote_fees += matched * lot["quote_fee_per_unit"]
            lot["remaining"] -= matched
            quantity_to_match -= matched
            matched_total += matched
        if fee_asset == quote_symbol and quantity > epsilon:
            realized_quote_fees += fee_amount * (matched_total / quantity)

    remaining_quantity = max(0.0, opening_inventory_quantity - closing_inventory_quantity)
    remaining_open_quote_fees = sum(lot["remaining"] * lot["quote_fee_per_unit"] for lot in opening_lots)
    unrealized_gross: float | None = None
    unrealized_pnl: float | None = None
    if current_price is not None:
        market_price = _number(current_price, field="Current price", positive=True)
        if origin_side == "sell":
            unrealized_gross = sum(lot["remaining"] * (lot["unit_price"] - market_price) for lot in opening_lots)
        else:
            unrealized_gross = sum(lot["remaining"] * (market_price - lot["unit_price"]) for lot in opening_lots)
        unrealized_pnl = unrealized_gross - remaining_open_quote_fees

    stored_status = str(swing.get("status") or "open").strip().lower()
    if opening_inventory_quantity > epsilon and remaining_quantity <= epsilon:
        derived_status = "closed"
    elif closing_quantity > epsilon:
        derived_status = "partially_closed"
    else:
        derived_status = "open"
    if stored_status == "accepting_loss" and derived_status != "closed":
        derived_status = "accepting_loss"

    realized_cash_gain = net_quote_cash_flow if derived_status == "closed" else None
    realized_asset_gain = max(0.0, net_asset_flow) if derived_status == "closed" else None
    target_ratchet_quantity = (
        realized_asset_gain
        if objective == "accumulate_asset" and realized_asset_gain is not None and realized_asset_gain > epsilon
        else 0.0
    )
    return {
        "status": derived_status,
        "trading_objective": objective,
        "origin_side": origin_side,
        "closing_side": closing_side,
        "opening_quantity": opening_quantity,
        "opening_quote_quantity": opening_quote,
        "closing_quantity": closing_quantity,
        "closing_quote_quantity": closing_quote,
        "remaining_quantity": remaining_quantity,
        "weighted_opening_price": opening_quote / opening_quantity if opening_quantity > epsilon else None,
        "weighted_closing_price": closing_quote / closing_quantity if closing_quantity > epsilon else None,
        "realized_gross_pnl_quote": realized_gross,
        "realized_fee_quote": realized_quote_fees,
        "realized_pnl_quote": realized_gross - realized_quote_fees,
        "realized_cash_gain_quote": realized_cash_gain,
        "realized_net_asset_change": net_asset_flow if derived_status == "closed" else None,
        "realized_asset_gain": realized_asset_gain,
        "target_ratchet_quantity": target_ratchet_quantity,
        "unrealized_gross_pnl_quote": unrealized_gross,
        "unrealized_pnl_quote": unrealized_pnl,
        "fees_by_asset": fees_by_asset,
        "unpriced_fees_by_asset": {
            asset: amount for asset, amount in fees_by_asset.items() if asset != quote_symbol
        },
    }


def calculate_target_ratchet(
    swing: dict[str, Any],
    economics: dict[str, Any],
    *,
    target_quantity: float,
    minimum_holding_pct: float,
) -> dict[str, float]:
    """Return a settlement proposal without mutating portfolio policy."""
    current_target = _number(target_quantity, field="Target quantity")
    minimum_pct = _number(minimum_holding_pct, field="Minimum Holding percentage")
    if current_target < 0:
        raise SpotSwingDomainError("Target quantity cannot be negative.")
    if not 0 <= minimum_pct <= 100:
        raise SpotSwingDomainError("Minimum Holding percentage must be between 0 and 100.")

    objective = str(swing.get("trading_objective") or "").strip().lower() or None
    status = str(economics.get("status") or "").strip().lower()
    gain = _number(economics.get("realized_asset_gain") or 0.0, field="Realized asset gain")
    applied_gain = gain if objective == "accumulate_asset" and status == "closed" and gain > 0 else 0.0
    next_target = current_target + applied_gain
    return {
        "previous_target_quantity": current_target,
        "realized_asset_gain": gain,
        "applied_gain_quantity": applied_gain,
        "next_target_quantity": next_target,
        "minimum_holding_pct": minimum_pct,
        "next_protected_floor_quantity": next_target * minimum_pct / 100.0,
    }
