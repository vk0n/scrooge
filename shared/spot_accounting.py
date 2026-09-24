from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.runtime_db import append_portfolio_transaction, list_portfolio_transactions


def _number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric == numeric else 0.0


def build_spot_quote_leg(transaction: dict[str, Any]) -> dict[str, Any] | None:
    """Build the managed quote-currency leg for one confirmed strategy fill."""
    source = str(transaction.get("source") or "").strip().lower()
    side = str(transaction.get("tx_type") or "").strip().lower()
    quote_symbol = str(transaction.get("quote_symbol") or "USDT").strip().upper() or "USDT"
    transaction_id = str(transaction.get("transaction_id") or "").strip()
    if source != "binance_strategy" or side not in {"buy", "sell"} or not transaction_id:
        return None
    if bool(transaction.get("spot_quote_leg")):
        return None

    executed_quote = _number(transaction.get("executed_quote_quantity"))
    if executed_quote <= 0:
        executed_quote = _number(transaction.get("quantity")) * _number(transaction.get("price"))
    if executed_quote <= 0:
        return None

    commissions = transaction.get("commissions") if isinstance(transaction.get("commissions"), dict) else {}
    quote_fee = _number(commissions.get(quote_symbol))
    quantity = executed_quote + quote_fee if side == "buy" else max(0.0, executed_quote - quote_fee)
    if quantity <= 0:
        return None

    return {
        "transaction_id": f"{transaction_id}:quote",
        "account_key": transaction.get("account_key") or "manual_spot",
        "executed_at": transaction.get("executed_at"),
        "tx_type": "sell" if side == "buy" else "buy",
        "asset_symbol": quote_symbol,
        "quote_symbol": quote_symbol,
        "quantity": quantity,
        "price": 1.0,
        "source": source,
        "status": transaction.get("status") or "settled",
        "note": f"Managed {quote_symbol} leg for {transaction_id}.",
        "external_order_id": transaction.get("external_order_id"),
        "custody_location": "binance",
        "spot_quote_leg": True,
        "capital_effect": "none",
        "base_transaction_id": transaction_id,
        "spot_order_intent_id": transaction.get("spot_order_intent_id"),
        "swing_id": transaction.get("swing_id"),
    }


def ensure_spot_quote_leg(
    transaction: dict[str, Any],
    *,
    path: Path | None = None,
    existing_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    quote_leg = build_spot_quote_leg(transaction)
    if quote_leg is None:
        return None
    known_ids = existing_ids
    if known_ids is None:
        known_ids = {
            str(item.get("transaction_id") or "")
            for item in list_portfolio_transactions(account_key=quote_leg["account_key"], path=path)
        }
    if quote_leg["transaction_id"] in known_ids:
        return None
    appended = append_portfolio_transaction(quote_leg, path=path)
    known_ids.add(str(quote_leg["transaction_id"]))
    return appended


def backfill_spot_quote_legs(
    transactions: list[dict[str, Any]],
    *,
    path: Path | None = None,
) -> int:
    """Project historical strategy fills into managed quote cash exactly once."""
    existing_ids = {str(item.get("transaction_id") or "") for item in transactions}
    reserve_by_account: dict[tuple[str, str], float] = {}
    for transaction in transactions:
        if str(transaction.get("status") or "settled").lower() != "settled":
            continue
        asset = str(transaction.get("asset_symbol") or "").strip().upper()
        quote = str(transaction.get("quote_symbol") or "USDT").strip().upper() or "USDT"
        if asset != quote:
            continue
        key = (str(transaction.get("account_key") or "manual_spot"), asset)
        quantity = _number(transaction.get("quantity"))
        tx_type = str(transaction.get("tx_type") or "").strip().lower()
        if tx_type in {"buy", "deposit", "adjustment"}:
            reserve_by_account[key] = reserve_by_account.get(key, 0.0) + quantity
        elif tx_type in {"sell", "withdraw"}:
            reserve_by_account[key] = reserve_by_account.get(key, 0.0) - quantity

    appended_count = 0
    for transaction in transactions:
        quote_leg = build_spot_quote_leg(transaction)
        if quote_leg is None or quote_leg["transaction_id"] in existing_ids:
            continue
        key = (str(quote_leg["account_key"]), str(quote_leg["asset_symbol"]))
        reserve = reserve_by_account.get(key, 0.0)
        if quote_leg["tx_type"] == "sell" and quote_leg["quantity"] > reserve + 1e-8:
            shortfall = quote_leg["quantity"] - max(0.0, reserve)
            contribution_id = f"{transaction['transaction_id']}:reserve-funding"
            if contribution_id not in existing_ids:
                append_portfolio_transaction(
                    {
                        "transaction_id": contribution_id,
                        "account_key": quote_leg["account_key"],
                        "executed_at": quote_leg["executed_at"],
                        "tx_type": "deposit",
                        "asset_symbol": quote_leg["asset_symbol"],
                        "quote_symbol": quote_leg["quote_symbol"],
                        "quantity": shortfall,
                        "price": 1.0,
                        "source": "binance_strategy_funding",
                        "status": "settled",
                        "note": "Recovered external USDT funding used by a historical strategy buy.",
                        "custody_location": "binance",
                        "capital_effect": "contribution",
                        "spot_reserve_funding": True,
                        "base_transaction_id": transaction["transaction_id"],
                    },
                    path=path,
                )
                existing_ids.add(contribution_id)
                appended_count += 1
            reserve += shortfall
        append_portfolio_transaction(quote_leg, path=path)
        existing_ids.add(str(quote_leg["transaction_id"]))
        appended_count += 1
        reserve += quote_leg["quantity"] if quote_leg["tx_type"] == "buy" else -quote_leg["quantity"]
        reserve_by_account[key] = reserve
    return appended_count
