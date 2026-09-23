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
    create_spot_order_intent,
    ensure_portfolio_asset_policies,
    load_exchange_account_snapshot,
    load_spot_order_intent,
    load_spot_swing,
    list_spot_order_status_events,
    list_spot_swing_executions,
    list_spot_swings,
    list_portfolio_daily_snapshots,
    list_portfolio_transactions,
    load_runtime_state_snapshot,
    runtime_db_path,
    upsert_portfolio_asset_policy,
    upsert_portfolio_daily_snapshot,
    update_portfolio_transaction_status,
)
from shared.spot_swing import calculate_swing_economics  # noqa: E402
from shared.treasury_ledger import append_treasury_event, project_portfolio_transaction  # noqa: E402

DEFAULT_ACCOUNT_KEY = "manual_spot"
DEFAULT_QUOTE = "USDT"
STABLE_ASSETS = {"USDT", "USDC", "FDUSD", "BUSD", "DAI", "TUSD", "USD"}
TRANSACTION_TYPES = {"buy", "sell", "deposit", "withdraw", "adjustment"}
CUSTODY_LOCATIONS = {"unassigned", "binance", "cold_storage"}
PRICE_CACHE_SECONDS = float(os.getenv("SCROOGE_PORTFOLIO_PRICE_CACHE_SECONDS", "30") or "30")
PRICE_TIMEOUT_SECONDS = float(os.getenv("SCROOGE_PORTFOLIO_PRICE_TIMEOUT_SECONDS", "4") or "4")
SPOT_BALANCE_STALE_AFTER_SECONDS = float(os.getenv("SCROOGE_SPOT_BALANCE_STALE_AFTER_SECONDS", "180") or "180")
SPOT_ORDER_PREVIEW_TTL_SECONDS = float(os.getenv("SCROOGE_SPOT_ORDER_PREVIEW_TTL_SECONDS", "60") or "60")
ASSET_LEDGER_PAGE_SIZE = 5
ASSET_LEDGER_FILTERS = {"all", "open", "closed"}
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

_PRICE_CACHE: dict[str, tuple[float, float, str]] = {}


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


def _format_quantity(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "0"
    return f"{numeric:,.8f}".rstrip("0").rstrip(".")


def _now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _spot_execution_enabled() -> bool:
    return str(os.getenv("SCROOGE_SPOT_EXECUTION_ENABLED", "0") or "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _state_price(asset_symbol: str, quote_symbol: str) -> tuple[float | None, str | None]:
    try:
        state = load_runtime_state_snapshot()
    except OSError:
        return None, None
    if not isinstance(state, dict):
        return None, None
    state_symbol = _clean_symbol(state.get("symbol"))
    expected_symbol = f"{asset_symbol}{quote_symbol}"
    if state_symbol != expected_symbol:
        return None, None
    timestamp = str(state.get("last_price_updated_at") or "").strip() or None
    return _as_float(state.get("last_price")), timestamp


def _fetch_market_price(asset_symbol: str, quote_symbol: str) -> tuple[float | None, str | None, str | None]:
    if asset_symbol in STABLE_ASSETS and quote_symbol in STABLE_ASSETS:
        return 1.0, None, _now_text()

    state_price, state_price_updated_at = _state_price(asset_symbol, quote_symbol)
    if state_price is not None:
        return state_price, None, state_price_updated_at

    pair = f"{asset_symbol}{quote_symbol}"
    cached = _PRICE_CACHE.get(pair)
    now = time.monotonic()
    if cached is not None and now - cached[0] <= PRICE_CACHE_SECONDS:
        return cached[1], None, cached[2]

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
            fetched_at = _now_text()
            _PRICE_CACHE[pair] = (now, price, fetched_at)
            return price, None, fetched_at
    return None, f"Market price unavailable for {pair}: {last_error or 'no price source returned a value'}", None


def _clean_custody(value: Any, *, default: str = "unassigned") -> str:
    normalized = str(value or default).strip().lower() or default
    if normalized not in CUSTODY_LOCATIONS:
        raise ValueError("Custody location must be Binance, Cold Storage, or Unassigned.")
    return normalized


def _empty_bucket(asset_symbol: str, quote_symbol: str, custody_location: str) -> dict[str, Any]:
    return {
        "asset_symbol": asset_symbol,
        "quote_symbol": quote_symbol,
        "custody_location": custody_location,
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


def _derive_buckets(transactions: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for transaction in transactions:
        if str(transaction.get("status") or "settled").lower() != "settled":
            continue
        asset_symbol = _clean_symbol(transaction.get("asset_symbol"))
        quote_symbol = _clean_symbol(transaction.get("quote_symbol"), default=DEFAULT_QUOTE)
        if not asset_symbol:
            continue
        tx_type = str(transaction.get("tx_type") or "").lower()
        if tx_type == "custody_transfer":
            source_custody = _clean_custody(transaction.get("source_custody"))
            destination_custody = _clean_custody(transaction.get("destination_custody"))
            quantity = _as_float(transaction.get("quantity")) or 0.0
            source_key = (asset_symbol, quote_symbol, source_custody)
            destination_key = (asset_symbol, quote_symbol, destination_custody)
            source_bucket = buckets.setdefault(
                source_key,
                _empty_bucket(asset_symbol, quote_symbol, source_custody),
            )
            destination_bucket = buckets.setdefault(
                destination_key,
                _empty_bucket(asset_symbol, quote_symbol, destination_custody),
            )
            source_quantity = float(source_bucket["quantity"])
            source_cost = float(source_bucket["cost_basis"])
            average_cost = source_cost / source_quantity if source_quantity > 0 else 0.0
            moved_cost = average_cost * quantity
            source_bucket["quantity"] = source_quantity - quantity
            source_bucket["cost_basis"] = max(0.0, source_cost - moved_cost)
            destination_bucket["quantity"] = float(destination_bucket["quantity"]) + quantity
            destination_bucket["cost_basis"] = float(destination_bucket["cost_basis"]) + moved_cost
            continue
        custody_location = _clean_custody(transaction.get("custody_location"))
        key = (asset_symbol, quote_symbol, custody_location)
        bucket = buckets.setdefault(key, _empty_bucket(asset_symbol, quote_symbol, custody_location))
        _apply_transaction(bucket, transaction)
    return buckets


def _validate_nonnegative_stacks(transactions: list[dict[str, Any]]) -> None:
    for bucket in _derive_buckets(transactions).values():
        quantity = float(bucket["quantity"])
        if quantity < -0.00000001:
            asset_symbol = str(bucket["asset_symbol"])
            quote_symbol = str(bucket["quote_symbol"])
            custody_location = str(bucket["custody_location"])
            raise ValueError(
                f"This change would reduce {asset_symbol}/{quote_symbol} in {custody_location} below zero "
                f"by {abs(quantity):.8f} {asset_symbol}."
            )


def _derive_holdings(transactions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    buckets = _derive_buckets(transactions)
    warnings: list[str] = []

    aggregate_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket in buckets.values():
        quantity = float(bucket["quantity"])
        if abs(quantity) < 0.00000001:
            continue
        key = (str(bucket["asset_symbol"]), str(bucket["quote_symbol"]))
        aggregate = aggregate_buckets.setdefault(
            key,
            {
                "asset_symbol": key[0],
                "quote_symbol": key[1],
                "quantity": 0.0,
                "cost_basis": 0.0,
                "custody": {location: {"quantity": 0.0, "cost_basis": 0.0} for location in CUSTODY_LOCATIONS},
            },
        )
        custody_location = str(bucket["custody_location"])
        aggregate["quantity"] += quantity
        aggregate["cost_basis"] += float(bucket["cost_basis"])
        aggregate["custody"][custody_location] = {
            "quantity": quantity,
            "cost_basis": float(bucket["cost_basis"]),
        }

    holdings: list[dict[str, Any]] = []
    priced_value_total = 0.0
    for bucket in aggregate_buckets.values():
        quantity = float(bucket["quantity"])
        if abs(quantity) < 0.00000001:
            continue
        asset_symbol = str(bucket["asset_symbol"])
        quote_symbol = str(bucket["quote_symbol"])
        cost_basis = float(bucket["cost_basis"])
        market_price, warning, market_price_updated_at = _fetch_market_price(asset_symbol, quote_symbol)
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
                "market_price_updated_at": market_price_updated_at,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (unrealized_pnl / cost_basis) * 100 if unrealized_pnl is not None and cost_basis > 0 else None,
                "allocation_pct": None,
                "is_dry_powder": asset_symbol in STABLE_ASSETS,
                "custody": bucket["custody"],
                "binance_quantity": float(bucket["custody"]["binance"]["quantity"]),
                "cold_storage_quantity": float(bucket["custody"]["cold_storage"]["quantity"]),
                "unassigned_quantity": float(bucket["custody"]["unassigned"]["quantity"]),
            }
        )

    for holding in holdings:
        market_value = _as_float(holding.get("market_value"))
        holding["allocation_pct"] = (market_value / priced_value_total) * 100 if market_value is not None and priced_value_total > 0 else None

    holdings.sort(
        key=lambda item: (
            -(_as_float(item.get("allocation_pct")) or 0.0),
            str(item.get("asset_symbol") or ""),
        )
    )
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
    price_timestamps = [
        str(holding["market_price_updated_at"])
        for holding in holdings
        if holding.get("market_price") is not None and holding.get("market_price_updated_at")
    ]
    return {
        "total_value": total_value,
        "invested_capital": invested_capital,
        "unrealized_pnl": floating_pnl,
        "unrealized_pnl_pct": (floating_pnl / invested_capital) * 100 if invested_capital > 0 else None,
        "dry_powder": dry_powder,
        "dry_powder_pct": (dry_powder / total_value) * 100 if total_value > 0 else None,
        "largest_position": largest,
        "holding_count": len(holdings),
        "prices_updated_at": min(price_timestamps) if price_timestamps else None,
    }


def _attach_asset_policies(holdings: list[dict[str, Any]]) -> None:
    managed_holdings = [holding for holding in holdings if not bool(holding.get("is_dry_powder"))]
    policies = ensure_portfolio_asset_policies(
        [
            {
                "asset_symbol": holding["asset_symbol"],
                "quote_symbol": holding["quote_symbol"],
                "target_quantity": holding["quantity"],
                "minimum_holding_pct": 100.0,
            }
            for holding in managed_holdings
        ],
        account_key=DEFAULT_ACCOUNT_KEY,
    )
    policy_by_asset = {
        (str(policy["asset_symbol"]), str(policy["quote_symbol"])): policy
        for policy in policies
    }
    for holding in holdings:
        if bool(holding.get("is_dry_powder")):
            holding.update(
                {
                    "target_quantity": None,
                    "minimum_holding_pct": None,
                    "trading_objective": None,
                    "protected_floor_quantity": 0.0,
                    "protected_holding_quantity": 0.0,
                    "amount_above_protected_floor": 0.0,
                    "amount_below_protected_floor": 0.0,
                    "policy_sellable_quantity": 0.0,
                    "immediately_sellable_quantity": 0.0,
                    "sellable_inventory_is_exchange_verified": False,
                    "target_delta_quantity": None,
                    "target_delta_pct": None,
                }
            )
            continue
        policy = policy_by_asset.get((holding["asset_symbol"], holding["quote_symbol"]))
        holding["target_quantity"] = _as_float(policy.get("target_quantity")) if policy else holding["quantity"]
        holding["minimum_holding_pct"] = _as_float(policy.get("minimum_holding_pct")) if policy else 100.0
        holding["trading_objective"] = policy.get("trading_objective") if policy else None
        _attach_inventory_state(holding)


def _attach_inventory_state(holding: dict[str, Any]) -> None:
    current_quantity = max(0.0, _as_float(holding.get("quantity")) or 0.0)
    target_quantity = max(0.0, _as_float(holding.get("target_quantity")) or 0.0)
    minimum_holding_pct = min(100.0, max(0.0, _as_float(holding.get("minimum_holding_pct")) or 0.0))
    binance_quantity = max(0.0, _as_float(holding.get("binance_quantity")) or 0.0)
    protected_floor = target_quantity * minimum_holding_pct / 100.0
    amount_above_floor = max(0.0, current_quantity - protected_floor)
    amount_below_floor = max(0.0, protected_floor - current_quantity)
    policy_sellable = min(amount_above_floor, binance_quantity)

    holding.update(
        {
            "protected_floor_quantity": protected_floor,
            "protected_holding_quantity": min(current_quantity, protected_floor),
            "amount_above_protected_floor": amount_above_floor,
            "amount_below_protected_floor": amount_below_floor,
            "policy_sellable_quantity": policy_sellable,
            "immediately_sellable_quantity": 0.0,
            "sellable_inventory_is_exchange_verified": False,
            "target_delta_quantity": current_quantity - target_quantity,
            "target_delta_pct": (
                ((current_quantity - target_quantity) / target_quantity) * 100
                if target_quantity > 0
                else None
            ),
        }
    )


def _load_spot_exchange_state() -> dict[str, Any]:
    snapshot = load_exchange_account_snapshot(venue="binance", account_type="spot")
    if snapshot is None:
        return {
            "venue": "binance",
            "account_type": "spot",
            "status": "unavailable",
            "captured_at": None,
            "last_attempt_at": None,
            "age_seconds": None,
            "is_stale": True,
            "is_balance_verified": False,
            "can_trade": None,
            "error": None,
            "balances": [],
            "usdt_free": None,
            "usdt_locked": None,
            "spot_execution_enabled": _spot_execution_enabled(),
        }

    captured_at_ms = _as_float(snapshot.get("captured_at_ms"))
    age_seconds = (
        max(0.0, time.time() - captured_at_ms / 1000.0)
        if captured_at_ms is not None
        else None
    )
    is_stale = age_seconds is None or age_seconds > SPOT_BALANCE_STALE_AFTER_SECONDS
    status = str(snapshot.get("status") or "unavailable")
    last_attempt_at_ms = _as_float(snapshot.get("last_attempt_at_ms"))
    balances = snapshot.get("balances") if isinstance(snapshot.get("balances"), list) else []
    balance_by_asset = {
        str(balance.get("asset_symbol") or "").upper(): balance
        for balance in balances
        if isinstance(balance, dict)
    }
    usdt = balance_by_asset.get("USDT")
    return {
        "venue": "binance",
        "account_type": "spot",
        "status": status,
        "captured_at": (
            datetime.fromtimestamp(captured_at_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if captured_at_ms is not None
            else None
        ),
        "last_attempt_at": (
            datetime.fromtimestamp(last_attempt_at_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if last_attempt_at_ms is not None
            else None
        ),
        "age_seconds": age_seconds,
        "is_stale": is_stale,
        "is_balance_verified": status == "ok" and not is_stale,
        "can_trade": snapshot.get("can_trade"),
        "error": snapshot.get("error"),
        "balances": balances,
        "usdt_free": _as_float(usdt.get("free")) if isinstance(usdt, dict) else 0.0,
        "usdt_locked": _as_float(usdt.get("locked")) if isinstance(usdt, dict) else 0.0,
        "spot_execution_enabled": _spot_execution_enabled(),
    }


def _attach_exchange_state(holdings: list[dict[str, Any]], exchange: dict[str, Any]) -> None:
    balance_by_asset = {
        str(balance.get("asset_symbol") or "").upper(): balance
        for balance in exchange.get("balances", [])
        if isinstance(balance, dict)
    }
    balance_verified = bool(exchange.get("is_balance_verified"))
    can_trade = exchange.get("can_trade") is True
    execution_enabled = bool(exchange.get("spot_execution_enabled"))
    for holding in holdings:
        asset_symbol = str(holding.get("asset_symbol") or "").upper()
        exchange_balance = balance_by_asset.get(asset_symbol, {})
        exchange_free = _as_float(exchange_balance.get("free")) or 0.0
        exchange_locked = _as_float(exchange_balance.get("locked")) or 0.0
        exchange_total = exchange_free + exchange_locked
        recorded_binance = _as_float(holding.get("binance_quantity")) or 0.0
        policy_sellable = _as_float(holding.get("policy_sellable_quantity")) or 0.0
        minimum_holding_pct = _as_float(holding.get("minimum_holding_pct"))
        if not execution_enabled:
            spot_trading_state = "locked"
            spot_trading_state_reason = "execution_disabled"
        elif bool(holding.get("is_dry_powder")) or minimum_holding_pct is None:
            spot_trading_state = "locked"
            spot_trading_state_reason = "not_policy_managed"
        elif minimum_holding_pct >= 100.0:
            spot_trading_state = "locked"
            spot_trading_state_reason = "fully_protected"
        else:
            spot_trading_state = "unlocked"
            spot_trading_state_reason = "policy_allows_trading"
        holding.update(
            {
                "exchange_binance_free_quantity": exchange_free,
                "exchange_binance_locked_quantity": exchange_locked,
                "exchange_binance_total_quantity": exchange_total,
                "binance_custody_variance": exchange_total - recorded_binance,
                "sellable_inventory_is_exchange_verified": balance_verified,
                "immediately_sellable_quantity": (
                    min(policy_sellable, exchange_free)
                    if execution_enabled and balance_verified and can_trade
                    else 0.0
                ),
                "spot_trading_state": spot_trading_state,
                "spot_trading_state_reason": spot_trading_state_reason,
            }
        )


PORTFOLIO_TRANSACTION_PAGE_SIZE = 5
PORTFOLIO_TIMELINE_DAYS = 180


def _portfolio_timeline(summary: dict[str, Any], holdings: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    prices_complete = all(holding.get("market_value") is not None for holding in holdings)
    if prices_complete:
        captured_at = datetime.now(timezone.utc)
        upsert_portfolio_daily_snapshot(
            {
                "snapshot_date": captured_at.date().isoformat(),
                "captured_at_ms": int(captured_at.timestamp() * 1000),
                "total_value": summary["total_value"],
                "invested_capital": summary["invested_capital"],
                "unrealized_pnl": summary["unrealized_pnl"],
                "dry_powder": summary["dry_powder"],
                "holdings": [
                    {
                        "asset_symbol": holding["asset_symbol"],
                        "market_value": holding["market_value"],
                        "allocation_pct": holding["allocation_pct"],
                    }
                    for holding in holdings
                ],
            },
            account_key=DEFAULT_ACCOUNT_KEY,
        )
    elif holdings:
        warnings.append("Treasury Timeline was not updated because one or more assets are awaiting a market price.")
    return list_portfolio_daily_snapshots(account_key=DEFAULT_ACCOUNT_KEY, limit=PORTFOLIO_TIMELINE_DAYS)


def load_portfolio_snapshot(*, transaction_offset: int = 0) -> tuple[dict[str, Any], list[str]]:
    normalized_offset = max(0, int(transaction_offset))
    transactions = list_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY, newest_first=False)
    holdings, warnings = _derive_holdings(transactions)
    _attach_asset_policies(holdings)
    exchange = _load_spot_exchange_state()
    _attach_exchange_state(holdings, exchange)
    summary = _summary_from_holdings(holdings)
    open_swings = [
        swing
        for swing in list_spot_swings(account_key=DEFAULT_ACCOUNT_KEY)
        if swing["status"] != "closed"
    ]
    summary["open_swing_count"] = len(open_swings)
    summary["open_swing_asset_count"] = len(
        {(swing["asset_symbol"], swing["quote_symbol"]) for swing in open_swings}
    )
    summary["binance_spot_usdt_free"] = exchange["usdt_free"]
    summary["binance_spot_usdt_locked"] = exchange["usdt_locked"]
    timeline = _portfolio_timeline(summary, holdings, warnings)
    newest_transactions = list_portfolio_transactions(
        limit=PORTFOLIO_TRANSACTION_PAGE_SIZE,
        offset=normalized_offset,
        newest_first=True,
        account_key=DEFAULT_ACCOUNT_KEY,
    )
    return (
        {
            "path": str(runtime_db_path()),
            "summary": summary,
            "exchange": exchange,
            "holdings": holdings,
            "timeline": timeline,
            "transactions": newest_transactions,
            "transaction_count": count_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY),
            "transaction_limit": PORTFOLIO_TRANSACTION_PAGE_SIZE,
            "transaction_offset": normalized_offset,
        },
        warnings,
    )


def load_portfolio_asset_transactions(
    asset_symbol: str,
    *,
    quote_symbol: str = DEFAULT_QUOTE,
    transaction_offset: int = 0,
) -> dict[str, Any]:
    normalized_asset = _clean_symbol(asset_symbol)
    normalized_quote = _clean_symbol(quote_symbol, default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    normalized_offset = max(0, int(transaction_offset))
    if not normalized_asset:
        raise ValueError("Asset symbol is required.")
    transactions = list_portfolio_transactions(
        limit=PORTFOLIO_TRANSACTION_PAGE_SIZE,
        offset=normalized_offset,
        newest_first=True,
        account_key=DEFAULT_ACCOUNT_KEY,
        asset_symbol=normalized_asset,
        quote_symbol=normalized_quote,
    )
    return {
        "asset_symbol": normalized_asset,
        "quote_symbol": normalized_quote,
        "transactions": transactions,
        "transaction_count": count_portfolio_transactions(
            account_key=DEFAULT_ACCOUNT_KEY,
            asset_symbol=normalized_asset,
            quote_symbol=normalized_quote,
        ),
        "transaction_limit": PORTFOLIO_TRANSACTION_PAGE_SIZE,
        "transaction_offset": normalized_offset,
    }


def load_portfolio_asset_ledger(
    asset_symbol: str,
    *,
    quote_symbol: str = DEFAULT_QUOTE,
    entry_filter: str = "all",
    entry_offset: int = 0,
) -> dict[str, Any]:
    normalized_asset = _clean_symbol(asset_symbol)
    normalized_quote = _clean_symbol(quote_symbol, default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    normalized_filter = str(entry_filter or "all").strip().lower()
    normalized_offset = max(0, int(entry_offset))
    if not normalized_asset:
        raise ValueError("Asset symbol is required.")
    if normalized_filter not in ASSET_LEDGER_FILTERS:
        raise ValueError("Asset Ledger filter must be all, open, or closed.")

    entries: list[dict[str, Any]] = []
    if normalized_filter == "all":
        transactions = list_portfolio_transactions(
            newest_first=False,
            account_key=DEFAULT_ACCOUNT_KEY,
            asset_symbol=normalized_asset,
            quote_symbol=normalized_quote,
        )
        entries.extend(
            {
                "entry_type": "transaction",
                "entry_id": f"transaction:{transaction['transaction_id']}",
                "occurred_at_ms": int(transaction["executed_at_ms"]),
                "occurred_at": transaction["executed_at"],
                "transaction": transaction,
            }
            for transaction in transactions
        )

    market_price, _, market_price_updated_at = _fetch_market_price(normalized_asset, normalized_quote)
    swings = list_spot_swings(
        account_key=DEFAULT_ACCOUNT_KEY,
        asset_symbol=normalized_asset,
        quote_symbol=normalized_quote,
    )
    for swing in swings:
        executions = list_spot_swing_executions(swing["swing_id"])
        economics = calculate_swing_economics(swing, executions, current_price=market_price)
        derived_status = str(economics["status"])
        if normalized_filter == "open" and derived_status == "closed":
            continue
        if normalized_filter == "closed" and derived_status != "closed":
            continue
        opened_at_ms = int(swing["opened_at_ms"])
        entries.append(
            {
                "entry_type": "swing",
                "entry_id": f"swing:{swing['swing_id']}",
                "occurred_at_ms": opened_at_ms,
                "occurred_at": datetime.fromtimestamp(
                    opened_at_ms / 1000,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "swing": {
                    **swing,
                    "status": derived_status,
                    "current_market_price": market_price,
                    "market_price_updated_at": market_price_updated_at,
                    "age_seconds": max(
                        0,
                        int(((swing.get("closed_at_ms") or time.time() * 1000) - opened_at_ms) / 1000),
                    ),
                    "economics": economics,
                    "executions": executions,
                },
            }
        )

    entries.sort(
        key=lambda item: (int(item["occurred_at_ms"]), str(item["entry_id"])),
        reverse=True,
    )
    paged_entries = entries[normalized_offset:normalized_offset + ASSET_LEDGER_PAGE_SIZE]
    return {
        "asset_symbol": normalized_asset,
        "quote_symbol": normalized_quote,
        "filter": normalized_filter,
        "entries": paged_entries,
        "entry_count": len(entries),
        "entry_limit": ASSET_LEDGER_PAGE_SIZE,
        "entry_offset": normalized_offset,
    }


def _create_spot_order_intent_preview(
    payload: dict[str, Any],
    *,
    source: str,
    swing_id: str | None = None,
    reason_text: str | None = None,
    reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _spot_execution_enabled():
        raise ValueError("Real Spot execution is disabled by the safety switch.")
    normalized_source = str(source or "").strip().lower()
    if normalized_source not in {"manual", "strategy"}:
        raise ValueError("Spot order source must be manual or strategy.")
    asset_symbol = _clean_symbol(payload.get("asset_symbol"))
    quote_symbol = _clean_symbol(payload.get("quote_symbol"), default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    side = str(payload.get("side") or "").strip().lower()
    requested_quantity = _as_float(payload.get("quantity"))
    if not asset_symbol or asset_symbol in STABLE_ASSETS:
        raise ValueError("Choose a managed Treasury asset for Spot execution.")
    if quote_symbol != DEFAULT_QUOTE:
        raise ValueError("Manual Spot execution currently supports the shared USDT pool only.")
    if side not in {"buy", "sell"}:
        raise ValueError("Spot order side must be buy or sell.")
    if requested_quantity is None or requested_quantity <= 0:
        raise ValueError("Spot order quantity must be greater than zero.")

    normalized_swing_id = str(swing_id or "").strip() or None
    normalized_reason_text = str(reason_text or "").strip() or None
    normalized_reason = reason if isinstance(reason, dict) else {}
    if normalized_source == "strategy":
        if normalized_swing_id is None:
            raise ValueError("Strategy Spot orders must belong to a Swing.")
        if not normalized_reason_text and not normalized_reason:
            raise ValueError("Strategy Spot orders require an explainable reason.")
        swing = load_spot_swing(normalized_swing_id)
        if swing is None:
            raise LookupError("The linked Spot Swing was not found.")
        if swing["status"] == "closed":
            raise ValueError("A closed Spot Swing cannot accept another strategy order.")
        if swing["account_key"] != DEFAULT_ACCOUNT_KEY:
            raise ValueError("The linked Spot Swing belongs to another Treasury account.")
        if swing["asset_symbol"] != asset_symbol or swing["quote_symbol"] != quote_symbol:
            raise ValueError("The linked Spot Swing market does not match the order intent.")

    snapshot, _ = load_portfolio_snapshot()
    exchange = snapshot["exchange"]
    if not exchange.get("is_balance_verified"):
        raise ValueError("A fresh Binance Spot balance snapshot is required before previewing a real order.")
    if exchange.get("can_trade") is not True:
        raise ValueError("Binance reports that Spot trading is unavailable for this account.")
    holding = next(
        (
            item
            for item in snapshot["holdings"]
            if item["asset_symbol"] == asset_symbol and item["quote_symbol"] == quote_symbol
        ),
        None,
    )
    if holding is None:
        raise LookupError("Treasury asset was not found.")
    estimated_price = _as_float(holding.get("market_price"))
    if estimated_price is None or estimated_price <= 0:
        raise ValueError("A current market price is required before previewing a real order.")

    estimated_quote_value = requested_quantity * estimated_price
    available_quote = _as_float(exchange.get("usdt_free")) or 0.0
    available_asset = _as_float(holding.get("exchange_binance_free_quantity")) or 0.0
    protected_floor = _as_float(holding.get("protected_floor_quantity")) or 0.0
    policy_sellable = _as_float(holding.get("policy_sellable_quantity")) or 0.0
    immediate_sellable = _as_float(holding.get("immediately_sellable_quantity")) or 0.0
    current_quantity = _as_float(holding.get("quantity")) or 0.0
    projected_holding = current_quantity + requested_quantity if side == "buy" else current_quantity - requested_quantity

    if side == "buy" and estimated_quote_value > available_quote + 0.00000001:
        raise ValueError(
            f"Estimated order value is ${estimated_quote_value:,.2f}, but Binance has only ${available_quote:,.2f} USDT free."
        )
    if side == "sell":
        if requested_quantity > immediate_sellable + 0.00000001:
            raise ValueError(
                f"Requested sell exceeds the immediately sellable inventory of {immediate_sellable:.8f} {asset_symbol}."
            )
        if projected_holding < protected_floor - 0.00000001:
            raise ValueError(
                f"Requested sell would reduce the holding below the Protected Floor of {protected_floor:.8f} {asset_symbol}."
            )

    intent_id = uuid.uuid4().hex
    return create_spot_order_intent(
        {
            "intent_id": intent_id,
            "client_order_id": f"scr{intent_id}",
            "account_key": DEFAULT_ACCOUNT_KEY,
            "venue": "binance",
            "symbol": f"{asset_symbol}{quote_symbol}",
            "asset_symbol": asset_symbol,
            "quote_symbol": quote_symbol,
            "side": side,
            "requested_quantity": requested_quantity,
            "estimated_price": estimated_price,
            "estimated_quote_value": estimated_quote_value,
            "available_quote_quantity": available_quote,
            "available_asset_quantity": available_asset,
            "protected_floor_quantity": protected_floor,
            "policy_sellable_quantity": policy_sellable,
            "projected_holding_quantity": projected_holding,
            "source": normalized_source,
            "swing_id": normalized_swing_id,
            "reason_text": normalized_reason_text,
            "reason": normalized_reason,
            "preview_expires_at_ms": int((time.time() + SPOT_ORDER_PREVIEW_TTL_SECONDS) * 1000),
        }
    )


def create_spot_order_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a user-confirmed manual intent; public payloads cannot impersonate strategy orders."""
    return _create_spot_order_intent_preview(payload, source="manual")


def create_strategy_spot_order_intent(payload: dict[str, Any]) -> dict[str, Any]:
    """Prepare a strategy intent without submitting it or bypassing the common executor."""
    return _create_spot_order_intent_preview(
        payload,
        source="strategy",
        swing_id=str(payload.get("swing_id") or "").strip() or None,
        reason_text=str(payload.get("reason_text") or "").strip() or None,
        reason=payload.get("reason") if isinstance(payload.get("reason"), dict) else None,
    )


def get_spot_order_intent(intent_id: str) -> dict[str, Any]:
    intent = load_spot_order_intent(intent_id)
    if intent is None:
        raise LookupError("Spot order preview was not found.")
    intent["preview_expires_at_ms"] = int(intent["created_at_ms"] + SPOT_ORDER_PREVIEW_TTL_SECONDS * 1000)
    intent["is_preview_expired"] = (
        intent["status"] == "previewed"
        and time.time() * 1000 > intent["preview_expires_at_ms"]
    )
    intent["status_history"] = list_spot_order_status_events(intent_id)
    return intent


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
    if (
        tx_type == "deposit"
        and asset_symbol not in STABLE_ASSETS
        and (price is None or price <= 0)
    ):
        raise ValueError("Entry cost is required when bringing a non-stable asset into Treasury.")

    custody_location = _clean_custody(payload.get("custody_location"))
    fee_amount = _as_float(payload.get("fee_amount"))
    if fee_amount is not None and fee_amount < 0:
        raise ValueError("Fee cannot be negative.")

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "account_key": DEFAULT_ACCOUNT_KEY,
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
        "custody_location": custody_location,
        "source_custody": None,
        "destination_custody": None,
    }
    transactions = list_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY, newest_first=False)
    _validate_nonnegative_stacks([*transactions, transaction])
    appended = append_portfolio_transaction(transaction)
    project_portfolio_transaction(appended)
    snapshot, warnings = load_portfolio_snapshot()
    return {"transaction": appended, "portfolio": snapshot}, warnings


def create_custody_transfer(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    asset_symbol = _clean_symbol(payload.get("asset_symbol"))
    if not asset_symbol:
        raise ValueError("Asset symbol is required.")
    quantity = _as_float(payload.get("quantity"))
    if quantity is None or quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    source_custody = _clean_custody(payload.get("source_custody"))
    destination_custody = _clean_custody(payload.get("destination_custody"))
    if source_custody == destination_custody:
        raise ValueError("Source and destination custody must be different.")
    quote_symbol = _clean_symbol(payload.get("quote_symbol"), default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "account_key": DEFAULT_ACCOUNT_KEY,
        "executed_at": str(payload.get("executed_at") or "").strip() or _now_text(),
        "tx_type": "custody_transfer",
        "asset_symbol": asset_symbol,
        "quote_symbol": quote_symbol,
        "quantity": quantity,
        "price": None,
        "fee_amount": None,
        "fee_asset": None,
        "source": "manual",
        "status": "settled",
        "note": str(payload.get("note") or "").strip() or None,
        "external_order_id": None,
        "custody_location": "unassigned",
        "source_custody": source_custody,
        "destination_custody": destination_custody,
    }
    transactions = list_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY, newest_first=False)
    _validate_nonnegative_stacks([*transactions, transaction])
    appended = append_portfolio_transaction(transaction)
    project_portfolio_transaction(appended)
    snapshot, warnings = load_portfolio_snapshot()
    return {"transaction": appended, "portfolio": snapshot}, warnings


def update_portfolio_asset_policy(
    asset_symbol: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    normalized_asset = _clean_symbol(asset_symbol)
    quote_symbol = _clean_symbol(payload.get("quote_symbol"), default=DEFAULT_QUOTE) or DEFAULT_QUOTE
    target_quantity = _as_float(payload.get("target_quantity"))
    minimum_holding_pct = _as_float(payload.get("minimum_holding_pct"))
    objective_was_provided = "trading_objective" in payload
    trading_objective = str(payload.get("trading_objective") or "").strip().lower() or None
    if not normalized_asset:
        raise ValueError("Asset symbol is required.")
    if normalized_asset in STABLE_ASSETS:
        raise ValueError("Dry Powder does not use an asset holding policy.")
    if target_quantity is None or target_quantity <= 0:
        raise ValueError("Target Holding must be greater than zero.")
    if minimum_holding_pct is None or not 0 <= minimum_holding_pct <= 100:
        raise ValueError("Minimum Holding must be between 0% and 100%.")
    if objective_was_provided and trading_objective not in {None, "accumulate_cash", "accumulate_asset"}:
        raise ValueError("Trading Objective must be Accumulate Cash, Accumulate Asset, or unset.")

    transactions = list_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY, newest_first=False)
    holdings, _ = _derive_holdings(transactions)
    if not any(
        holding["asset_symbol"] == normalized_asset and holding["quote_symbol"] == quote_symbol
        for holding in holdings
    ):
        raise LookupError("Treasury asset was not found.")

    policy_payload = {
        "asset_symbol": normalized_asset,
        "quote_symbol": quote_symbol,
        "target_quantity": target_quantity,
        "minimum_holding_pct": minimum_holding_pct,
    }
    if objective_was_provided:
        policy_payload["trading_objective"] = trading_objective
    policy = upsert_portfolio_asset_policy(
        policy_payload,
        account_key=DEFAULT_ACCOUNT_KEY,
    )
    append_treasury_event(
        code="treasury_policy_updated",
        message=(
            f"Updated {normalized_asset} policy: target {_format_quantity(target_quantity)} "
            f"{normalized_asset}, minimum holding {_format_quantity(minimum_holding_pct)}%, "
            f"objective {(policy.get('trading_objective') or 'unset').replace('_', ' ')}."
        ),
        source_ref=f"portfolio_policy_update:{uuid.uuid4()}",
        context=dict(policy),
    )
    snapshot, warnings = load_portfolio_snapshot()
    return {"policy": policy, "portfolio": snapshot}, warnings


def set_portfolio_transaction_status(
    transaction_id: str,
    status: str,
) -> tuple[dict[str, Any], list[str]]:
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"settled", "voided"}:
        raise ValueError("Transaction status must be settled or voided.")
    transactions = list_portfolio_transactions(account_key=DEFAULT_ACCOUNT_KEY, newest_first=False)
    transaction_found = False
    proposed_transactions: list[dict[str, Any]] = []
    for transaction in transactions:
        proposed = dict(transaction)
        if proposed.get("transaction_id") == transaction_id:
            proposed["status"] = normalized_status
            transaction_found = True
        proposed_transactions.append(proposed)
    if not transaction_found:
        raise LookupError("Treasury entry was not found.")
    _validate_nonnegative_stacks(proposed_transactions)
    transaction = update_portfolio_transaction_status(transaction_id, normalized_status)
    if transaction is None:
        raise LookupError("Treasury entry was not found.")
    action = "Restored" if normalized_status == "settled" else "Voided"
    append_treasury_event(
        code="treasury_transaction_restored" if normalized_status == "settled" else "treasury_transaction_voided",
        message=(
            f"{action} the {str(transaction.get('tx_type') or 'entry').replace('_', ' ')} entry for "
            f"{_format_quantity(transaction.get('quantity'))} {transaction.get('asset_symbol')}."
        ),
        tone="neutral" if normalized_status == "settled" else "negative",
        source_ref=f"portfolio_transaction_status:{transaction_id}:{normalized_status}:{uuid.uuid4()}",
        context={"transaction_id": transaction_id, "status": normalized_status},
    )
    snapshot, warnings = load_portfolio_snapshot()
    return {"transaction": transaction, "portfolio": snapshot}, warnings
