from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.runtime_db import append_ui_log_entry, list_ledger_source_refs, list_portfolio_transactions


def _number(value: Any, *, decimals: int = 8) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "0"
    rendered = f"{numeric:,.{decimals}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _money(value: Any) -> str:
    return f"${_number(value)}"


def _custody_name(value: Any) -> str:
    return {
        "binance": "Binance",
        "cold_storage": "Cold Storage",
        "unassigned": "Unassigned",
    }.get(str(value or "").strip().lower(), str(value or "Unassigned").replace("_", " ").title())


def treasury_transaction_presentation(transaction: dict[str, Any]) -> tuple[str, str, str]:
    tx_type = str(transaction.get("tx_type") or "adjustment").strip().lower()
    asset = str(transaction.get("asset_symbol") or "asset").strip().upper()
    quantity = _number(transaction.get("quantity"))
    price = transaction.get("price")
    source = str(transaction.get("source") or "manual").strip().lower()
    venue_prefix = "Binance Spot " if source == "binance_manual" else ""

    if tx_type == "custody_transfer":
        source_name = _custody_name(transaction.get("source_custody"))
        destination_name = _custody_name(transaction.get("destination_custody"))
        return (
            "treasury_custody_moved",
            "neutral",
            f"Moved {quantity} {asset} from {source_name} to {destination_name}.",
        )

    verb_by_type = {
        "buy": "bought",
        "sell": "sold",
        "deposit": "deposited",
        "withdraw": "withdrew",
        "adjustment": "adjusted",
    }
    verb = verb_by_type.get(tx_type, tx_type.replace("_", " "))
    price_suffix = f" at {_money(price)}" if price is not None and tx_type in {"buy", "sell"} else ""
    rendered_verb = verb if venue_prefix else verb.capitalize()
    message = f"{venue_prefix}{rendered_verb} {quantity} {asset}{price_suffix}."
    if str(transaction.get("status") or "settled").lower() == "voided":
        message = f"{message[:-1]} (currently voided)."
    tone = "positive" if tx_type in {"buy", "deposit"} else "negative" if tx_type in {"sell", "withdraw"} else "neutral"
    return f"treasury_{tx_type}", tone, message


def project_portfolio_transaction(transaction: dict[str, Any], *, path: Path | None = None) -> bool:
    transaction_id = str(transaction.get("transaction_id") or "").strip()
    if not transaction_id:
        return False
    timestamp = str(transaction.get("executed_at") or "").strip() or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    code, tone, message = treasury_transaction_presentation(transaction)
    source_ref = f"portfolio_transaction:{transaction_id}"
    return append_ui_log_entry(
        timestamp,
        f"[{timestamp}] {message}",
        path,
        entry_id=source_ref,
        scope="treasury",
        code=code,
        tone=tone,
        message=message,
        source_ref=source_ref,
        context=dict(transaction),
    )


def project_portfolio_transactions(*, path: Path | None = None) -> int:
    projected = 0
    projected_refs = list_ledger_source_refs(prefix="portfolio_transaction:", path=path)
    for transaction in list_portfolio_transactions(newest_first=False, path=path):
        source_ref = f"portfolio_transaction:{transaction.get('transaction_id')}"
        if source_ref in projected_refs:
            continue
        if project_portfolio_transaction(transaction, path=path):
            projected += 1
    return projected


def append_treasury_event(
    *,
    code: str,
    message: str,
    tone: str = "neutral",
    timestamp: str | None = None,
    source_ref: str | None = None,
    context: dict[str, Any] | None = None,
    path: Path | None = None,
) -> bool:
    resolved_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    resolved_source_ref = source_ref or f"treasury_event:{uuid.uuid4()}"
    return append_ui_log_entry(
        resolved_timestamp,
        f"[{resolved_timestamp}] {message}",
        path,
        entry_id=resolved_source_ref,
        scope="treasury",
        code=code,
        tone=tone,
        message=message,
        source_ref=resolved_source_ref,
        context=context or {},
    )
