from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from typing import Any


def as_decimal(value: Any) -> Decimal | None:
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def normalize_market_quantity(
    symbol_info: dict[str, Any],
    requested_quantity: float,
) -> tuple[Decimal, Decimal]:
    """Apply the same Binance market lot rules for live and simulated orders."""
    filters = {
        str(item.get("filterType") or ""): item
        for item in symbol_info.get("filters", [])
        if isinstance(item, dict)
    }
    market_lot_filter = filters.get("MARKET_LOT_SIZE")
    market_step = as_decimal(market_lot_filter.get("stepSize")) if isinstance(market_lot_filter, dict) else None
    lot_filter = market_lot_filter if market_step is not None and market_step > 0 else filters.get("LOT_SIZE")
    if not isinstance(lot_filter, dict):
        raise ValueError("Binance did not return a market lot-size rule for this symbol.")
    step_size = as_decimal(lot_filter.get("stepSize"))
    min_quantity = as_decimal(lot_filter.get("minQty")) or Decimal("0")
    max_quantity = as_decimal(lot_filter.get("maxQty")) or Decimal("0")
    requested = as_decimal(requested_quantity)
    if requested is None or requested <= 0:
        raise ValueError("Spot order quantity must be greater than zero.")
    if step_size is None or step_size <= 0:
        normalized = requested
        step_size = Decimal("0")
    else:
        step_count = requested / step_size
        nearest_step_count = step_count.to_integral_value(rounding=ROUND_HALF_UP)
        # Quantities derived from prior float executions can land infinitesimally
        # below an exact exchange step (for example 261.17999999999995).
        if abs(step_count - nearest_step_count) <= Decimal("1e-9"):
            step_count = nearest_step_count
        normalized = step_count.to_integral_value(rounding=ROUND_DOWN) * step_size
    if normalized <= 0:
        raise ValueError("Spot order quantity rounds to zero under the Binance step size.")
    if normalized < min_quantity:
        raise ValueError(f"Quantity is below Binance minimum {format_decimal(min_quantity)}.")
    if max_quantity > 0 and normalized > max_quantity:
        raise ValueError(f"Quantity exceeds Binance maximum {format_decimal(max_quantity)}.")
    return normalized, step_size


def validate_market_notional(
    symbol_info: dict[str, Any],
    *,
    quantity: Decimal,
    price: float,
) -> None:
    filters = {
        str(item.get("filterType") or ""): item
        for item in symbol_info.get("filters", [])
        if isinstance(item, dict)
    }
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
    if not isinstance(notional_filter, dict):
        return
    min_notional = as_decimal(notional_filter.get("minNotional")) or Decimal("0")
    max_notional = as_decimal(notional_filter.get("maxNotional")) or Decimal("0")
    notional = quantity * Decimal(str(price))
    if min_notional > 0 and notional < min_notional:
        raise ValueError(f"Estimated order value is below Binance minimum ${format_decimal(min_notional)}.")
    if max_notional > 0 and notional > max_notional:
        raise ValueError(f"Estimated order value exceeds Binance maximum ${format_decimal(max_notional)}.")


def validate_sell_opening_round_trip(
    symbol_info: dict[str, Any],
    *,
    quantity: Decimal,
    price: float,
    trading_objective: str,
    close_profit_pct: float,
    estimated_fee_rate: float,
) -> None:
    """Reject a SELL opening that cannot produce a valid minimum-size profit close."""
    filters = {
        str(item.get("filterType") or ""): item
        for item in symbol_info.get("filters", [])
        if isinstance(item, dict)
    }
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
    if not isinstance(notional_filter, dict):
        return
    min_notional = as_decimal(notional_filter.get("minNotional")) or Decimal("0")
    if min_notional <= 0:
        return

    opening_price = as_decimal(price)
    profit_pct = as_decimal(close_profit_pct)
    fee_rate = as_decimal(estimated_fee_rate)
    if opening_price is None or opening_price <= 0:
        raise ValueError("Spot opening price must be greater than zero.")
    if profit_pct is None or not Decimal("0") < profit_pct < Decimal("100"):
        raise ValueError("Spot close profit must be between zero and 100 percent.")
    if fee_rate is None or not Decimal("0") <= fee_rate < Decimal("1"):
        raise ValueError("Spot estimated fee rate must be in the range [0, 1).")

    opening_notional = quantity * opening_price
    objective = str(trading_objective or "").strip().lower()
    if objective == "accumulate_cash":
        target_close_notional = opening_notional * (
            Decimal("1") - profit_pct / Decimal("100")
        )
    elif objective == "accumulate_asset":
        # The closing BUY reuses net opening proceeds and must leave room for its own fee.
        target_close_notional = (
            opening_notional * (Decimal("1") - fee_rate) / (Decimal("1") + fee_rate)
        )
    else:
        return

    if target_close_notional < min_notional:
        raise ValueError(
            "Strategy SELL opening is too small to remain closable at its profit target "
            f"under Binance minimum ${format_decimal(min_notional)}."
        )


def validate_market_close_remainder(
    symbol_info: dict[str, Any],
    *,
    remaining_quantity: float,
    closing_quantity: Decimal,
    price: float,
) -> None:
    """Reject a partial close that would strand an untradeable remainder."""
    remaining = as_decimal(remaining_quantity)
    if remaining is None or remaining <= 0:
        return
    remainder = remaining - closing_quantity
    if remainder <= Decimal("1e-12"):
        return
    try:
        normalized_remainder, step_size = normalize_market_quantity(
            symbol_info,
            float(remainder),
        )
        tolerance = max(Decimal("1e-12"), step_size * Decimal("1e-9"))
        if remainder - normalized_remainder > tolerance:
            raise ValueError("The remainder does not align with the Binance step size.")
        validate_market_notional(
            symbol_info,
            quantity=normalized_remainder,
            price=price,
        )
    except ValueError as exc:
        raise ValueError(
            "Partial Spot close would leave an untradeable remainder under Binance filters."
        ) from exc
