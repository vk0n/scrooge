"""Role-styled Treasury Ledger messages and durable projections."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime_db import append_ui_log_entry, list_ledger_source_refs, list_portfolio_transactions


def _number(value: Any, *, decimals: int = 8) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "0"
    rendered = f"{numeric:,.{decimals}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _money(value: Any) -> str:
    return f"${_number(value)}"


def _signed_pct(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    sign = "+" if numeric > 0 else ""
    return f"{sign}{_number(numeric, decimals=2)}%"


def _signed_money(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    sign = "+" if numeric > 0 else "-" if numeric < 0 else ""
    return f"{sign}${_number(abs(numeric), decimals=2)}"


def _signed_quantity(value: Any, asset: str) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    sign = "+" if numeric > 0 else "-" if numeric < 0 else ""
    return f"{sign}{_number(abs(numeric))} {asset}"


def _level(value: Any) -> str | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return f"L{numeric}" if numeric > 0 else None


def _strategy_reason(reason: dict[str, Any]) -> str:
    action_type = str(reason.get("action_type") or "").strip().lower()
    level = _level(reason.get("signal_level"))
    move = _signed_pct(reason.get("rolling_change_pct"))

    if action_type == "open":
        tranche = reason.get("level_allocation_pct", reason.get("final_tranche_pct"))
        trigger = " ".join(part for part in (level, f"rise {move}" if move else None) if part)
        stake = f"{_number(tranche, decimals=2)}% campaign stake" if tranche is not None else "campaign stake"
        return f"{trigger or 'A rising signal'}; I put the {stake} to work."

    if action_type == "accumulate_asset":
        tranche = reason.get("accumulation_tranche_pct", reason.get("final_tranche_pct"))
        trigger = " ".join(part for part in (level, f"dip {move}" if move else None) if part)
        stake = f"{_number(tranche, decimals=2)}%" if tranche is not None else "a measured share"
        return f"{trigger or 'A buying signal'}; I deployed {stake} of Spendable Reserve."

    if action_type == "close":
        close_reason = str(reason.get("close_reason") or "").strip().lower()
        if close_reason == "profit_target":
            favorable = _signed_pct(reason.get("favorable_move_pct"))
            target = (
                f"{_number(reason['close_profit_pct'], decimals=2)}%"
                if reason.get("close_profit_pct") is not None
                else None
            )
            if favorable and target:
                return f"Bargain Goal cleared at {favorable} against {target}."
            return "Bargain Goal cleared."
        cleanup_labels = {
            "age_l1_cleanup": "L1 waiter cleanup",
            "age_l2_cleanup": "L2 waiter cleanup",
            "age_l3_cleanup": "L3 waiter cleanup",
            "deep_loss_cleanup": "Deep-loss cleanup",
            "capacity_cleanup": "Capacity cleanup",
        }
        if close_reason in cleanup_labels:
            details: list[str] = []
            if reason.get("age_days") is not None:
                details.append(f"{_number(reason['age_days'], decimals=1)}d old")
            pnl = _signed_pct(reason.get("unrealized_pnl_pct_before_cleanup"))
            if pnl:
                details.append(pnl)
            reverse_level = _level(reason.get("actual_reverse_signal_level"))
            if reverse_level:
                details.append(f"reverse {reverse_level}")
            suffix = f": {', '.join(details)}" if details else ""
            return f"{cleanup_labels[close_reason]}{suffix}."
        if close_reason == "manual":
            return "Closed at your request."

    return "My standing Treasury rules called for this fill."


def spot_order_presentation_message(transaction: dict[str, Any]) -> str:
    tx_type = str(transaction.get("tx_type") or transaction.get("side") or "trade").strip().lower()
    asset = str(transaction.get("asset_symbol") or transaction.get("symbol") or "asset").strip().upper()
    quantity = _number(transaction.get("quantity"))
    price = transaction.get("price")
    price_suffix = f" at {_money(price)}" if price is not None else ""
    reason = transaction.get("reason") if isinstance(transaction.get("reason"), dict) else {}
    action_type = str(reason.get("action_type") or transaction.get("strategy_action_type") or "").strip().lower()
    source = str(transaction.get("source") or "manual").strip().lower()
    is_close = action_type == "close"
    verb = (
        "bought back"
        if is_close and tx_type == "buy"
        else "sold out"
        if is_close
        else "bought"
        if tx_type == "buy"
        else "sold"
    )
    lead = f"I {verb} {quantity} {asset}{price_suffix} on Binance Spot."

    if source in {"strategy", "binance_strategy"}:
        return f"{lead} {_strategy_reason(reason)}"
    if transaction.get("treasury_intake"):
        return f"{lead} I brought it into the vault at your request."
    return f"{lead[:-1]} at your request."


def spot_order_settlement_message(
    transaction: dict[str, Any],
    settlement: dict[str, Any],
) -> str:
    message = spot_order_presentation_message(transaction)
    asset = str(transaction.get("asset_symbol") or transaction.get("symbol") or "asset").strip().upper()
    details: list[str] = []

    protected_cash = settlement.get("protected_cash_use")
    if isinstance(protected_cash, dict) and float(protected_cash.get("consumed_quote") or 0) > 0:
        details.append(
            f"I drew ${_number(protected_cash['consumed_quote'], decimals=2)} from Protected Cash."
        )

    cash_retention = settlement.get("cash_retention")
    if isinstance(cash_retention, dict) and float(cash_retention.get("retained_quote") or 0) > 0:
        details.append(
            f"I retained ${_number(cash_retention['retained_quote'], decimals=2)} from "
            f"${_number(cash_retention.get('eligible_cash_gain_quote'), decimals=2)} "
            "of realized cash profit."
        )

    target_ratchet = settlement.get("target_ratchet")
    if isinstance(target_ratchet, dict) and float(target_ratchet.get("applied_gain_quantity") or 0) > 0:
        details.append(
            f"I secured {_number(target_ratchet['applied_gain_quantity'])} {asset} of Swing profit "
            f"in Target, now {_number(target_ratchet.get('next_target_quantity'))} {asset}."
        )

    accumulation_ratchet = settlement.get("accumulation_ratchet")
    if isinstance(accumulation_ratchet, dict):
        details.append(
            f"The fill used ${_number(accumulation_ratchet.get('deployed_quote_quantity'), decimals=2)} "
            f"from Free Vault Reserve and raised Target by "
            f"{_number(accumulation_ratchet.get('applied_gain_quantity'))} {asset} to "
            f"{_number(accumulation_ratchet.get('next_target_quantity'))} {asset}."
        )

    manual_target_ratchet = settlement.get("manual_target_ratchet")
    if isinstance(manual_target_ratchet, dict):
        details.append(
            f"I moved Target from {_number(manual_target_ratchet.get('previous_target_quantity'))} {asset} "
            f"to {_number(manual_target_ratchet.get('next_target_quantity'))} {asset}."
        )

    result, _ = _spot_order_settlement_result(settlement, asset)
    if result:
        details.append(result)

    return " ".join((message, *details))


def _spot_order_settlement_result(
    settlement: dict[str, Any],
    asset: str,
) -> tuple[str | None, float | None]:
    economics = settlement.get("swing_economics")
    if isinstance(economics, dict) and economics.get("status") == "closed":
        objective = str(economics.get("trading_objective") or "").strip().lower()
        if objective == "accumulate_asset":
            value = economics.get("realized_net_asset_change")
            result = _signed_quantity(value, asset)
            numeric = float(value) if result is not None else None
            percentage_value = economics.get("realized_net_asset_change_pct")
        else:
            value = economics.get("realized_pnl_quote")
            result = _signed_money(value)
            numeric = float(value) if result is not None else None
            percentage_value = economics.get("realized_pnl_pct")
        percentage = _signed_pct(percentage_value)
        if result:
            percentage_suffix = f" ({percentage})" if percentage else ""
            return f"Result: {result}{percentage_suffix}.", numeric

    return None, None


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
    venue_prefix = "Binance Spot " if source.startswith("binance_") else ""

    office_direction = str(transaction.get("office_transfer_direction") or "").strip().lower()
    if office_direction in {"to_office", "from_office"}:
        direction_text = "Treasury to Futures Office" if office_direction == "to_office" else "Futures Office to Treasury"
        return (
            "treasury_office_transfer",
            "negative" if office_direction == "to_office" else "positive",
            f"Transferred {quantity} {asset} from {direction_text}.",
        )

    if tx_type == "custody_transfer":
        source_name = _custody_name(transaction.get("source_custody"))
        destination_name = _custody_name(transaction.get("destination_custody"))
        return (
            "treasury_custody_moved",
            "neutral",
            f"Moved {quantity} {asset} from {source_name} to {destination_name}.",
        )

    if source.startswith("binance_") and tx_type in {"buy", "sell"}:
        message = spot_order_presentation_message(transaction)
        if str(transaction.get("status") or "settled").lower() == "voided":
            message = f"{message[:-1]} (currently voided)."
        tone = "positive" if tx_type == "buy" else "negative"
        return f"treasury_{tx_type}", tone, message

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


def project_portfolio_transaction(
    transaction: dict[str, Any],
    *,
    settlement: dict[str, Any] | None = None,
    path: Path | None = None,
) -> bool:
    if transaction.get("spot_quote_leg") or transaction.get("spot_reserve_funding"):
        return False
    if transaction.get("ledger_projection_deferred") and settlement is None:
        return False
    transaction_id = str(transaction.get("transaction_id") or "").strip()
    if not transaction_id:
        return False
    timestamp = str(transaction.get("executed_at") or "").strip() or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    code, tone, message = treasury_transaction_presentation(transaction)
    context = dict(transaction)
    if settlement is not None:
        message = spot_order_settlement_message(transaction, settlement)
        context["settlement"] = settlement
        _, result_value = _spot_order_settlement_result(
            settlement,
            str(transaction.get("asset_symbol") or "asset").strip().upper(),
        )
        if result_value is not None:
            tone = "positive" if result_value > 0 else "negative" if result_value < 0 else "neutral"
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
        context=context,
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
