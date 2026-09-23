from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_DOWN
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
        normalized = (requested / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
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
