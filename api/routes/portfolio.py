from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from services.portfolio_service import (
    create_custody_transfer,
    create_portfolio_transaction,
    create_spot_order_preview,
    get_spot_order_intent,
    load_portfolio_asset_transactions,
    load_portfolio_asset_ledger,
    load_portfolio_snapshot,
    set_portfolio_transaction_status,
    update_portfolio_asset_policy,
)
from services.spot_order_service import queue_spot_order_intent
from services.system_service import get_service_status

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


class PortfolioAssetPolicyRequest(BaseModel):
    quote_symbol: str = Field(default="USDT", min_length=1, max_length=24)
    target_quantity: float = Field(..., gt=0)
    minimum_holding_pct: float = Field(..., ge=0, le=100)


class SpotOrderPreviewRequest(BaseModel):
    asset_symbol: str = Field(..., min_length=1, max_length=24)
    quote_symbol: str = Field(default="USDT", min_length=1, max_length=24)
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0)


class SpotOrderExecuteRequest(BaseModel):
    confirmation: Literal["CONFIRM_SPOT_ORDER"]


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


@router.post("/assets/{asset_symbol}/policy")
def update_asset_policy(asset_symbol: str, data: PortfolioAssetPolicyRequest) -> dict[str, object]:
    try:
        payload, warnings = update_portfolio_asset_policy(asset_symbol, data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {**payload, "warnings": warnings}


@router.get("/assets/{asset_symbol}/transactions")
def get_asset_transactions(
    asset_symbol: str,
    quote_symbol: str = Query(default="USDT", min_length=1, max_length=24),
    transaction_offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    try:
        return load_portfolio_asset_transactions(
            asset_symbol,
            quote_symbol=quote_symbol,
            transaction_offset=transaction_offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/assets/{asset_symbol}/ledger")
def get_asset_ledger(
    asset_symbol: str,
    quote_symbol: str = Query(default="USDT", min_length=1, max_length=24),
    entry_filter: Literal["all", "open", "closed"] = Query(default="all", alias="filter"),
    entry_offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    try:
        return load_portfolio_asset_ledger(
            asset_symbol,
            quote_symbol=quote_symbol,
            entry_filter=entry_filter,
            entry_offset=entry_offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


@router.post("/spot-orders/preview")
def preview_spot_order(data: SpotOrderPreviewRequest) -> dict[str, object]:
    try:
        return create_spot_order_preview(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/spot-orders/{intent_id}")
def spot_order_status(intent_id: str) -> dict[str, object]:
    try:
        return get_spot_order_intent(intent_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/spot-orders/{intent_id}/execute")
def execute_spot_order(
    intent_id: str,
    data: SpotOrderExecuteRequest,
    request: Request,
) -> dict[str, object]:
    del data
    try:
        service_status = get_service_status()
    except RuntimeError:
        service_status = None
    if service_status is not None and not service_status.running:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scrooge runtime is offline. A real Spot order cannot be delivered safely.",
        )
    requested_by = "basic-user" if request.headers.get("Authorization", "").startswith("Basic ") else "unknown"
    try:
        return queue_spot_order_intent(intent_id, requested_by=requested_by)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
