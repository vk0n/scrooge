from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.portfolio_service import create_portfolio_transaction, load_portfolio_snapshot

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


@router.get("")
def get_portfolio() -> dict[str, object]:
    try:
        payload, warnings = load_portfolio_snapshot()
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
