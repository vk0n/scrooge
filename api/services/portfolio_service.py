from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from shared.runtime_db import (  # noqa: E402
    append_portfolio_transaction,
    count_portfolio_transactions,
    list_portfolio_transactions,
    load_runtime_state_snapshot,
    runtime_db_path,
)

DEFAULT_ACCOUNT_KEY = "manual_spot"
DEFAULT_QUOTE = "USDT"
STABLE_ASSETS = {"USDT", "USDC", "FDUSD", "BUSD", "DAI", "TUSD", "USD"}
TRANSACTION_TYPES = {"buy", "sell", "deposit", "withdraw", "adjustment"}
PRICE_CACHE_SECONDS = float(os.getenv("SCROOGE_PORTFOLIO_PRICE_CACHE_SECONDS", "30") or "30")
PRICE_TIMEOUT_SECONDS = float(os.getenv("SCROOGE_PORTFOLIO_PRICE_TIMEOUT_SECONDS", "4") or "4")
PRICE_ENDPOINTS = [
    endpoint.strip()
    for endpoint in (
        os.getenv(
            "SCROOGE_PORTFOLIO_PRICE_ENDPOINTS",
            "https://api.binance.com/api/v3/ticker/price,https://fapi.binance.com/fapi/v1/ticker/price",
        )
        or ""
    ).split(",")
    if endpoint.strip()
]

_PRICE_CACHE: dict[str, tuple[float, float]] = {}


def _clean_symbol(value: Any, *, default: str | None = None) -> str:
    text = str(value or default or "").strip().upper()
    return "".join(char for char in text if char.isalnum())


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _state_price(asset_symbol: str, quote_symbol: str) -> float | None:
    try:
        state = load_runtime_state_snapshot()
    except OSError:
        return None
    if not isinstance(state, dict):
        return None
    state_symbol = _clean_symbol(state.get("symbol"))
    expected_symbol = f"{asset_symbol}{quote_symbol}"
    if state_symbol != expected_symbol:
        return None
    return _as_float(state.get("last_price"))


def _fetch_market_price(asset_symbol: str, quote_symbol: str) -> tuple[float | None, str | None]:
    if asset_symbol in STABLE_ASSETS and quote_symbol in STABLE_ASSETS:
        return 1.0, None

    state_price = _state_price(asset_symbol, quote_symbol)
    if state_price is not None:
        return state_price, None

    pair = f"{asset_symbol}{quote_symbol}"
    cached = _PRICE_CACHE.get(pair)
    now = time.monotonic()
    if cached is not None and now - cached[0] <= PRICE_CACHE_SECONDS:
        return cached[1], None

    last_error: str | None = None
    for endpoint in PRICE_ENDPOINTS:
        url = f"{endpoint}?{urllib.parse.urlencode({'symbol': pair})}"
        try:
            with urllib.request.urlopen(url, timeout=PRICE_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            continue
        price = _as_float(payload.get("price") if isinstance(payload, dict) else None)
        if price is not None:
            _PRICE_CACHE[pair] = (now, price)
            return price, None
    return None, f"Market price unavailable for {pair}: {last_error or 'no price source returned a value'}"


def _empty_bucket(asset_symbol: str, quote_symbol: str) -> dict[str, Any]:
    return {
        "asset_symbol": asset_symbol,
        "quote_symbol": quote_symbol,
        "quantity": 0.0,
        "cost_basis": 0.0,
    }


def _apply_transaction(bucket: dict[str, Any], transaction: dict[str, Any]) -> None:
    tx_type = str(transaction.get("tx_type") or "").lower()
    quantity = _as_float(transaction.get("quantity")) or 0.0
    price = _as_float(transaction.get("price"))
    fee_amount = _as_float(transaction.get("fee_amount")) or 0.0
    fee_asset = _clean_symbol(transaction.get("fee_asset"))
    asset_symbol = _clean_symbol(transaction.get("asset_symbol"))
    quote_symbol = _clean_symbol(transaction.get("quote_symbol"), default=DEFAULT_QUOTE)
    current_quantity = float(bucket["quantity"])
    current_cost = float(bucket["cost_basis"])

    if tx_type in {"buy", "deposit", "adjustment"}:
        bucket["quantity"] = current_quantity + quantity
        effective_price = price
        if effective_price is None and asset_symbol in STABLE_ASSETS and quote_symbol in STABLE_ASSETS:
            effective_price = 1.0
        if effective_price is not None:
            cost_addition = quantity * effective_price
            if fee_asset == quote_symbol:
                cost_addition += fee_amount
            bucket["cost_basis"] = current_cost + cost_addition
        return

    if tx_type in {"sell", "withdraw"}:
        average_cost = current_cost / current_quantity if current_quantity > 0 else 0.0
        next_quantity = current_quantity - quantity
        bucket["quantity"] = next_quantity
        bucket["cost_basis"] = max(0.0, current_cost - average_cost * quantity) if next_quantity > 0 else 0.0


def _derive_holdings(transactions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    warnings: list[str] = []

    for transaction in transactions:
        if str(transaction.get("status") or "settled").lower() != "settled":
            continue
        asset_symbol = _clean_symbol(transaction.get("asset_symbol"))
        quote_symbol = _clean_symbol(transaction.get("quote_symbol"), default=DEFAULT_QUOTE)
        if not asset_symbol:
            continue
        key = (asset_symbol, quote_symbol)
        bucket = buckets.setdefault(key, _empty_bucket(asset_symbol, quote_symbol))
        _apply_transaction(bucket, transaction)

    holdings: list[dict[str, Any]] = []
    priced_value_total = 0.0
    for bucket in buckets.values():
        quantity = float(bucket["quantity"])
        if abs(quantity) < 0.00000001:
            continue
        asset_symbol = str(bucket["asset_symbol"])
        quote_symbol = str(bucket["quote_symbol"])
        cost_basis = float(bucket["cost_basis"])
        market_price, warning = _fetch_market_price(asset_symbol, quote_symbol)
        if warning:
            warnings.append(warning)
        market_value = quantity * market_price if market_price is not None else None
        if market_value is not None:
            priced_value_total += market_value
        unrealized_pnl = market_value - cost_basis if market_value is not None else None
        holdings.append(
            {
                "asset_symbol": asset_symbol,
                "quote_symbol": quote_symbol,
                "quantity": quantity,
                "average_cost": cost_basis / quantity if quantity > 0 and cost_basis > 0 else None,
                "invested_capital": cost_basis,
                "market_price": market_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (unrealized_pnl / cost_basis) * 100 if unrealized_pnl is not None and cost_basis > 0 else None,
                "allocation_pct": None,
                "is_dry_powder": asset_symbol in STABLE_ASSETS,
            }
        )

    for holding in holdings:
        market_value = _as_float(holding.get("market_value"))
        holding["allocation_pct"] = (market_value / priced_value_total) * 100 if market_value is not None and priced_value_total > 0 else None

    holdings.sort(key=lambda item: _as_float(item.get("market_value")) or 0.0, reverse=True)
    return holdings, warnings


def _summary_from_holdings(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    total_value = sum(_as_float(holding.get("market_value")) or 0.0 for holding in holdings)
    invested_capital = sum(_as_float(holding.get("invested_capital")) or 0.0 for holding in holdings)
    floating_pnl = total_value - invested_capital if holdings else 0.0
    dry_powder = sum(
        _as_float(holding.get("market_value")) or 0.0
        for holding in holdings
        if bool(holding.get("is_dry_powder"))
    )
    largest = max(holdings, key=lambda item: _as_float(item.get("market_value")) or 0.0, default=None)
    return {
        "total_value": total_value,
        "invested_capital": invested_capital,
        "unrealized_pnl": floating_pnl,
        "unrealized_pnl_pct": (floating_pnl / invested_capital) * 100 if invested_capital > 0 else None,
        "dry_powder": dry_powder,
        "dry_powder_pct": (dry_powder / total_value) * 100 if total_value > 0 else None,
        "largest_position": largest,
        "holding_count": len(holdings),
    }


def load_portfolio_snapshot() -> tuple[dict[str, Any], list[str]]:
    transactions = list_portfolio_transactions(newest_first=False)
    holdings, warnings = _derive_holdings(transactions)
    newest_transactions = list_portfolio_transactions(limit=25, newest_first=True)
    return (
        {
            "path": str(runtime_db_path()),
            "summary": _summary_from_holdings(holdings),
            "holdings": holdings,
            "transactions": newest_transactions,
            "transaction_count": count_portfolio_transactions(),
        },
        warnings,
    )


def create_portfolio_transaction(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    tx_type = str(payload.get("tx_type") or "").strip().lower()
    if tx_type not in TRANSACTION_TYPES:
        raise ValueError("Unsupported portfolio transaction type.")

    asset_symbol = _clean_symbol(payload.get("asset_symbol"))
    if not asset_symbol:
        raise ValueError("Asset symbol is required.")

    quantity = _as_float(payload.get("quantity"))
    if quantity is None or quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    price = _as_float(payload.get("price"))
    if tx_type in {"buy", "sell"} and (price is None or price <= 0):
        raise ValueError("Entry cost is required for buy and sell transactions.")

    quote_symbol = _clean_symbol(payload.get("quote_symbol"), default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    fee_amount = _as_float(payload.get("fee_amount"))
    if fee_amount is not None and fee_amount < 0:
        raise ValueError("Fee cannot be negative.")

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "account_key": str(payload.get("account_key") or DEFAULT_ACCOUNT_KEY).strip() or DEFAULT_ACCOUNT_KEY,
        "executed_at": str(payload.get("executed_at") or "").strip() or _now_text(),
        "tx_type": tx_type,
        "asset_symbol": asset_symbol,
        "quote_symbol": quote_symbol,
        "quantity": quantity,
        "price": price,
        "fee_amount": fee_amount,
        "fee_asset": _clean_symbol(payload.get("fee_asset")) or quote_symbol,
        "source": str(payload.get("source") or "manual").strip().lower() or "manual",
        "status": str(payload.get("status") or "settled").strip().lower() or "settled",
        "note": str(payload.get("note") or "").strip() or None,
        "external_order_id": str(payload.get("external_order_id") or "").strip() or None,
    }
    appended = append_portfolio_transaction(transaction)
    snapshot, warnings = load_portfolio_snapshot()
    return {"transaction": appended, "portfolio": snapshot}, warnings
