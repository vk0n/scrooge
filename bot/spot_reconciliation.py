"""Operator tooling for reconciling confirmed duplicate Binance Spot fills."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bot.spot_execution import _execution_summary
from core.binance_retry import create_binance_client
from core.runtime_db import (
    append_portfolio_transaction,
    bootstrap_runtime_db,
    list_portfolio_transactions,
    load_spot_order_intent,
    runtime_db_path,
    update_spot_order_intent,
)
from core.spot_accounting import ensure_spot_quote_leg
from core.treasury_ledger import project_portfolio_transaction


def _number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric == numeric else 0.0


def _fill_transaction(
    intent: dict[str, Any],
    summary: dict[str, Any],
    *,
    role: str,
    side: str,
) -> dict[str, Any]:
    order_id = str(summary.get("order_id") or "").strip()
    quantity = _number(summary.get("executed_quantity"))
    quote_quantity = _number(summary.get("executed_quote_quantity"))
    price = _number(summary.get("average_price"))
    executed_at_ms = int(_number(summary.get("executed_at_ms")))
    if not order_id or quantity <= 0 or quote_quantity <= 0 or price <= 0 or executed_at_ms <= 0:
        raise ValueError("The confirmed Binance fill is missing required execution details.")
    return {
        "transaction_id": f"spot-reconciliation:{intent['intent_id']}:{role}:{order_id}",
        "account_key": intent["account_key"],
        "executed_at": datetime.fromtimestamp(executed_at_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "tx_type": side,
        "asset_symbol": intent["asset_symbol"],
        "quote_symbol": intent["quote_symbol"],
        "quantity": quantity,
        "price": price,
        "fee_amount": summary.get("fee_amount"),
        "fee_asset": summary.get("fee_asset"),
        "source": "binance_manual",
        "status": "settled",
        "note": (
            "Reconciled an unintended duplicate Binance Spot BUY."
            if role == "duplicate-buy"
            else "Reconciled the Binance Spot SELL that corrected an unintended duplicate BUY."
        ),
        "external_order_id": order_id,
        "custody_location": "binance",
        "spot_order_intent_id": intent["intent_id"],
        "executed_quote_quantity": quote_quantity,
        "commissions": summary.get("commissions") or {},
        "reconciliation_role": role,
        "changes_target": False,
        "uses_protected_cash": False,
    }


def _persist_fill_transaction(
    intent: dict[str, Any],
    summary: dict[str, Any],
    *,
    role: str,
    side: str,
    path: Path,
) -> dict[str, Any]:
    transaction = _fill_transaction(intent, summary, role=role, side=side)
    existing = {
        str(item.get("transaction_id") or ""): item
        for item in list_portfolio_transactions(account_key=intent["account_key"], path=path)
    }
    persisted = existing.get(transaction["transaction_id"])
    if persisted is None:
        persisted = append_portfolio_transaction(transaction, path=path)
    ensure_spot_quote_leg(persisted, path=path, existing_ids=set(existing))
    project_portfolio_transaction(persisted, path=path)
    return persisted


def _duplicate_summary(intent: dict[str, Any]) -> dict[str, Any]:
    result = intent.get("result") if isinstance(intent.get("result"), dict) else {}
    reconciliation = result.get("reconciliation") if isinstance(result.get("reconciliation"), dict) else {}
    summary = reconciliation.get("duplicate_fill")
    if not isinstance(summary, dict):
        summary = result
    if not isinstance(summary, dict):
        raise ValueError("The intent does not contain a confirmed duplicate fill.")
    return summary


def _primary_result(intent: dict[str, Any], duplicate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "FILLED",
        "order_id": intent.get("exchange_order_id"),
        "client_order_id": intent.get("client_order_id"),
        "executed_quantity": intent.get("executed_quantity"),
        "executed_quote_quantity": intent.get("executed_quote_quantity"),
        "average_price": intent.get("average_price"),
        "fee_amount": intent.get("fee_amount"),
        "fee_asset": intent.get("fee_asset"),
        "intent_id": intent["intent_id"],
        "requested_quantity": intent.get("requested_quantity"),
        "side": intent.get("side"),
        "asset_symbol": intent.get("asset_symbol"),
        "quote_symbol": intent.get("quote_symbol"),
        "source": intent.get("source"),
        "swing_id": intent.get("swing_id"),
        "treasury_intake": bool((intent.get("request") or {}).get("treasury_intake")),
        "ledger_transaction_id": f"spot-order:{intent['intent_id']}",
        "reconciliation": {
            "reason": "duplicate_exchange_submission",
            "duplicate_fill": duplicate,
            "duplicate_position_status": "open",
        },
    }


def register_duplicate_buy(
    intent_id: str,
    *,
    expected_duplicate_order_id: str,
    path: Path,
    apply: bool,
) -> dict[str, Any]:
    intent = load_spot_order_intent(intent_id, path=path)
    if intent is None:
        raise LookupError("Spot order intent was not found.")
    if intent["side"] != "buy" or intent.get("swing_id"):
        raise ValueError("Only a standalone duplicate BUY can use this reconciliation.")

    duplicate = _duplicate_summary(intent)
    duplicate_order_id = str(duplicate.get("order_id") or "").strip()
    primary_order_id = str(intent.get("exchange_order_id") or "").strip()
    if duplicate_order_id != str(expected_duplicate_order_id).strip():
        raise ValueError("The duplicate order ID does not match the confirmed intent result.")
    if not primary_order_id or duplicate_order_id == primary_order_id:
        raise ValueError("The intent does not prove two distinct Binance orders.")
    if str(duplicate.get("status") or "").upper() != "FILLED":
        raise ValueError("The duplicate Binance order is not confirmed FILLED.")

    transaction = _fill_transaction(intent, duplicate, role="duplicate-buy", side="buy")
    report = {
        "action": "register_duplicate_buy",
        "apply": apply,
        "intent_id": intent_id,
        "primary_order_id": primary_order_id,
        "duplicate_order_id": duplicate_order_id,
        "quantity": transaction["quantity"],
        "quote_debit": transaction["executed_quote_quantity"],
        "transaction_id": transaction["transaction_id"],
        "target_change": 0.0,
        "protected_cash_change": 0.0,
    }
    if not apply:
        return report

    persisted = _persist_fill_transaction(
        intent,
        duplicate,
        role="duplicate-buy",
        side="buy",
        path=path,
    )
    result = _primary_result(intent, duplicate)
    updated = update_spot_order_intent(
        intent_id,
        {"status": "filled", "error": None, "result_json": result},
        expected_statuses={"accounting_error", "fill_confirmed", "accounting_updated", "filled"},
        path=path,
    )
    if updated is None:
        raise RuntimeError("The intent changed while the duplicate BUY was being reconciled.")
    report["persisted_transaction_id"] = persisted["transaction_id"]
    report["intent_status"] = updated["status"]
    return report


def register_corrective_sell(
    intent_id: str,
    sell_summary: dict[str, Any],
    *,
    path: Path,
    apply: bool,
) -> dict[str, Any]:
    intent = load_spot_order_intent(intent_id, path=path)
    if intent is None:
        raise LookupError("Spot order intent was not found.")
    duplicate = _duplicate_summary(intent)
    reconciliation = (intent.get("result") or {}).get("reconciliation") or {}
    if reconciliation.get("duplicate_position_status") not in {"open", "closed"}:
        raise ValueError("Register the duplicate BUY before its corrective SELL.")
    if str(sell_summary.get("status") or "").upper() != "FILLED":
        raise ValueError("The corrective Binance SELL is not confirmed FILLED.")
    duplicate_quantity = _number(duplicate.get("executed_quantity"))
    sold_quantity = _number(sell_summary.get("executed_quantity"))
    if abs(duplicate_quantity - sold_quantity) > max(1e-8, duplicate_quantity * 1e-8):
        raise ValueError("The corrective SELL quantity must exactly match the duplicate BUY quantity.")
    if int(_number(sell_summary.get("executed_at_ms"))) <= int(_number(duplicate.get("executed_at_ms"))):
        raise ValueError("The corrective SELL must execute after the duplicate BUY.")

    transaction = _fill_transaction(intent, sell_summary, role="corrective-sell", side="sell")
    commissions = sell_summary.get("commissions") if isinstance(sell_summary.get("commissions"), dict) else {}
    quote_fee = _number(commissions.get(intent["quote_symbol"]))
    duplicate_commissions = duplicate.get("commissions") if isinstance(duplicate.get("commissions"), dict) else {}
    duplicate_quote_fee = _number(duplicate_commissions.get(intent["quote_symbol"]))
    quote_cost = _number(duplicate.get("executed_quote_quantity")) + duplicate_quote_fee
    quote_return = _number(sell_summary.get("executed_quote_quantity")) - quote_fee
    duplicate_average_price = _number(duplicate.get("average_price"))
    sell_average_price = _number(sell_summary.get("average_price"))
    report = {
        "action": "register_corrective_sell",
        "apply": apply,
        "intent_id": intent_id,
        "duplicate_order_id": str(duplicate.get("order_id") or ""),
        "corrective_order_id": str(sell_summary.get("order_id") or ""),
        "quantity": sold_quantity,
        "quote_cost": quote_cost,
        "quote_return": quote_return,
        "realized_quote_result": quote_return - quote_cost,
        "duplicate_average_price": duplicate_average_price,
        "sell_average_price": sell_average_price,
        "sold_below_entry": sell_average_price < duplicate_average_price,
        "transaction_id": transaction["transaction_id"],
        "target_change": 0.0,
        "protected_cash_change": 0.0,
    }
    if not apply:
        return report

    persisted = _persist_fill_transaction(
        intent,
        sell_summary,
        role="corrective-sell",
        side="sell",
        path=path,
    )
    result = dict(intent.get("result") or _primary_result(intent, duplicate))
    next_reconciliation = dict(result.get("reconciliation") or {})
    next_reconciliation.update(
        {
            "reason": "duplicate_exchange_submission",
            "duplicate_fill": duplicate,
            "duplicate_position_status": "closed",
            "corrective_fill": sell_summary,
            "realized_quote_result": quote_return - quote_cost,
        }
    )
    result["reconciliation"] = next_reconciliation
    updated = update_spot_order_intent(
        intent_id,
        {"status": "filled", "error": None, "result_json": result},
        expected_statuses={"filled"},
        path=path,
    )
    if updated is None:
        raise RuntimeError("The intent changed while the corrective SELL was being reconciled.")
    report["persisted_transaction_id"] = persisted["transaction_id"]
    report["intent_status"] = updated["status"]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Persist the reconciliation; default is dry-run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    duplicate = subparsers.add_parser("register-duplicate-buy")
    duplicate.add_argument("--intent-id", required=True)
    duplicate.add_argument("--duplicate-order-id", required=True)

    correction = subparsers.add_parser("register-corrective-sell")
    correction.add_argument("--intent-id", required=True)
    correction.add_argument("--order-id", required=True)
    return parser


def main() -> None:
    load_dotenv()
    args = _parser().parse_args()
    path = args.db_path or runtime_db_path()
    bootstrap_runtime_db(path)
    if args.command == "register-duplicate-buy":
        report = register_duplicate_buy(
            args.intent_id,
            expected_duplicate_order_id=args.duplicate_order_id,
            path=path,
            apply=args.apply,
        )
    else:
        intent = load_spot_order_intent(args.intent_id, path=path)
        if intent is None:
            raise LookupError("Spot order intent was not found.")
        client = create_binance_client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET"),
            ping=False,
        )
        order = client.get_order(symbol=intent["symbol"], orderId=args.order_id, recvWindow=5000)
        if str(order.get("side") or "").upper() != "SELL":
            raise ValueError("The corrective Binance order must be a SELL.")
        summary = _execution_summary(client, intent["symbol"], order)
        report = register_corrective_sell(args.intent_id, summary, path=path, apply=args.apply)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
