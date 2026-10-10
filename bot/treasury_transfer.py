from __future__ import annotations

from decimal import Decimal
from typing import Any

from api.services.portfolio_service import (
    load_portfolio_snapshot,
    record_confirmed_office_transfer,
    treasury_transfer_enabled,
)
from core.runtime_db import consume_portfolio_retained_cash, credit_portfolio_retained_cash


def _number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Office transfer quantity must be numeric.") from exc
    if numeric != numeric or numeric <= 0:
        raise ValueError("Office transfer quantity must be greater than zero.")
    return numeric


def _amount_text(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f")


class TreasuryTransferExecutor:
    def __init__(self, client: Any, *, logger: Any) -> None:
        self.client = client
        self.logger = logger

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not treasury_transfer_enabled():
            raise ValueError("Treasury Office transfers are disabled by the safety switch.")
        direction = str(payload.get("direction") or "").strip().lower()
        if direction not in {"to_office", "from_office"}:
            raise ValueError("Office transfer direction must be to_office or from_office.")
        quantity = _number(payload.get("quantity"))
        transfer_ref = str(payload.get("transfer_ref") or "").strip()
        use_protected_cash = bool(payload.get("use_protected_cash"))
        cash_bucket = str(payload.get("cash_bucket") or "spendable").strip().lower()
        if not transfer_ref:
            raise ValueError("Office transfer reference is required.")
        if cash_bucket not in {"spendable", "retained", "mixed"}:
            raise ValueError("Office transfer cash bucket must be spendable, retained, or mixed.")
        if direction != "to_office" and use_protected_cash:
            raise ValueError("Protected Cash applies only to Treasury-to-Office transfers.")

        portfolio, _ = load_portfolio_snapshot()
        protected_required = 0.0
        if direction == "to_office":
            available = float(portfolio["summary"].get("vault_reserve_available") or 0.0)
            retained = float(portfolio["summary"].get("vault_reserve_retained") or 0.0)
            spendable_value = portfolio["summary"].get("vault_reserve_spendable")
            spendable = (
                float(spendable_value)
                if spendable_value is not None
                else max(0.0, available - retained)
            )
            exchange_free = float(portfolio["exchange"].get("usdt_free") or 0.0)
            if cash_bucket == "retained":
                permitted_reserve = retained
            elif cash_bucket == "mixed" or use_protected_cash:
                permitted_reserve = spendable + retained
            else:
                permitted_reserve = spendable
            if quantity > min(permitted_reserve, exchange_free) + 1e-8:
                hint = (
                    " Enable Protected Cash for this transfer."
                    if cash_bucket == "spendable" and not use_protected_cash and retained > 0
                    else ""
                )
                raise ValueError(f"Office transfer exceeds currently spendable Treasury USDT.{hint}")
            if quantity > available + 1e-8:
                raise ValueError("Office transfer exceeds currently available Treasury USDT.")
            protected_required = (
                quantity
                if cash_bucket == "retained"
                else max(0.0, quantity - spendable)
            )
            transfer_type = "MAIN_UMFUTURE"
        else:
            balances = self.client.futures_account_balance()
            usdt = next(
                (
                    item
                    for item in balances
                    if str(item.get("asset") or "").strip().upper() == "USDT"
                ),
                None,
            )
            futures_available = float(
                (usdt or {}).get("availableBalance")
                or (usdt or {}).get("balance")
                or 0.0
            )
            if quantity > futures_available + 1e-8:
                raise ValueError("Office transfer exceeds available USD-M Futures USDT.")
            transfer_type = "UMFUTURE_MAIN"

        response = self.client.universal_transfer(
            type=transfer_type,
            asset="USDT",
            amount=_amount_text(quantity),
            clientTranId=transfer_ref,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Binance did not confirm the Treasury Office transfer.")
        external_id = str(response.get("tranId") or response.get("clientTranId") or "").strip() or None
        recorded, _ = record_confirmed_office_transfer(
            transfer_ref=transfer_ref,
            direction=direction,
            quantity=quantity,
            external_transfer_id=external_id,
            protected_cash_required=protected_required,
        )
        transaction = recorded["transaction"]
        protected_required = float(
            transaction.get("protected_cash_required")
            if transaction.get("protected_cash_required") is not None
            else protected_required
        )
        protected_use = None
        if protected_required > 0:
            protected_use = consume_portfolio_retained_cash(
                protected_required,
                reference_id=f"office-transfer:{transfer_ref}:protected-cash",
                use_type="office_transfer",
                context={
                    "transfer_ref": transfer_ref,
                    "quantity": quantity,
                    "direction": direction,
                },
            )
        protected_credit = None
        if direction == "from_office" and cash_bucket == "retained":
            protected_credit = credit_portfolio_retained_cash(
                quantity,
                reference_id=f"office-transfer:{transfer_ref}:protected-cash-credit",
                credit_type="office_transfer",
                context={
                    "transfer_ref": transfer_ref,
                    "quantity": quantity,
                    "direction": direction,
                },
            )
        self.logger.info(
            "treasury_office_transfer_completed direction=%s quantity=%s transfer_ref=%s exchange_id=%s",
            direction,
            quantity,
            transfer_ref,
            external_id,
        )
        return {
            "direction": direction,
            "quantity": quantity,
            "asset_symbol": "USDT",
            "transfer_ref": transfer_ref,
            "exchange_transfer_id": external_id,
            "ledger_transaction_id": transaction["transaction_id"],
            "cash_bucket": cash_bucket,
            "protected_cash_used": (
                float(protected_use["consumed_quote"])
                if protected_use is not None
                else 0.0
            ),
            "protected_cash_credited": (
                float(protected_credit["credited_quote"])
                if protected_credit is not None
                else 0.0
            ),
        }
