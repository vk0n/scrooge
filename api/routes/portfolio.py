from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.portfolio_service import (
    create_custody_transfer,
    create_portfolio_transaction,
    load_portfolio_snapshot,
    set_portfolio_transaction_status,
)

router = APIRouter()


class PortfolioTransactionRequest(BaseModel):
    tx_type: Literal["buy", "sell", "deposit", "withdraw", "adjustment"]
    asset_symbol: str = Field(..., min_length=1, max_length=24)
    quantity: float = Field(..., gt=0)
    price: float | None = Field(default=None, gt=0)
    quote_symbol: str = Field(default="USDT", min_length=1, max_length=24)
    fee_amount: float | None = Field(default=None, ge=0)
    fee_asset: str | None = Field(default=None, max_length=24)
    executed_at: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=500)
    custody_location: Literal["unassigned", "binance", "cold_storage"] = "unassigned"


class CustodyTransferRequest(BaseModel):
    asset_symbol: str = Field(..., min_length=1, max_length=24)
    quote_symbol: str = Field(default="USDT", min_length=1, max_length=24)
    quantity: float = Field(..., gt=0)
    source_custody: Literal["unassigned", "binance", "cold_storage"]
    destination_custody: Literal["unassigned", "binance", "cold_storage"]
    executed_at: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=500)


class PortfolioTransactionStatusRequest(BaseModel):
    status: Literal["settled", "voided"]


@router.get("")
def get_portfolio(transaction_offset: int = Query(default=0, ge=0)) -> dict[str, object]:
    try:
        payload, warnings = load_portfolio_snapshot(transaction_offset=transaction_offset)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {**payload, "warnings": warnings}


@router.post("/transactions")
def add_portfolio_transaction(data: PortfolioTransactionRequest) -> dict[str, object]:
    try:
        payload, warnings = create_portfolio_transaction(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {**payload, "warnings": warnings}


@router.post("/custody-transfers")
def add_custody_transfer(data: CustodyTransferRequest) -> dict[str, object]:
    try:
        payload, warnings = create_custody_transfer(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {**payload, "warnings": warnings}


@router.post("/transactions/{transaction_id}/status")
def update_transaction_status(
    transaction_id: str,
    data: PortfolioTransactionStatusRequest,
) -> dict[str, object]:
    try:
        payload, warnings = set_portfolio_transaction_status(transaction_id, data.status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {**payload, "warnings": warnings}
