from __future__ import annotations

from decimal import Decimal
from typing import Any

from api.services.portfolio_service import (
    load_portfolio_snapshot,
    record_confirmed_office_transfer,
    treasury_transfer_enabled,
)


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
        if not transfer_ref:
            raise ValueError("Office transfer reference is required.")

        portfolio, _ = load_portfolio_snapshot()
        if direction == "to_office":
            available = float(portfolio["summary"].get("vault_reserve_available") or 0.0)
            exchange_free = float(portfolio["exchange"].get("usdt_free") or 0.0)
            if quantity > min(available, exchange_free) + 1e-8:
                raise ValueError("Office transfer exceeds currently available Treasury USDT.")
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
        )
        transaction = recorded["transaction"]
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
        }
