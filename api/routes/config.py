from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.config_service import CONFIG_PATH, extract_editable_config, load_config, update_editable_config

router = APIRouter()


class EditableParamsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sl_mult: float | None = Field(default=None, gt=0)
    tp_mult: float | None = Field(default=None, gt=0)
    trail_atr_mult: float | None = Field(default=None, gt=0)
    rsi_extreme_long: float | None = Field(default=None, ge=0, le=100)
    rsi_extreme_short: float | None = Field(default=None, ge=0, le=100)
    rsi_long_open_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_long_qty_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_long_tp_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_long_close_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_short_open_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_short_qty_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_short_tp_threshold: float | None = Field(default=None, ge=0, le=100)
    rsi_short_close_threshold: float | None = Field(default=None, ge=0, le=100)


class EditableConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = Field(default=None, min_length=3, max_length=20)
    leverage: int | None = Field(default=None, ge=1, le=125)
    use_full_balance: bool | None = None
    qty: float | None = Field(default=None, gt=0)
    params: EditableParamsPayload | None = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol cannot be empty")
        if not re.fullmatch(r"[A-Z0-9]{3,20}", normalized):
            raise ValueError("symbol must match [A-Z0-9]{3,20}")
        return normalized


@router.get("")
def get_config() -> dict[str, object]:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"config": config, "path": str(CONFIG_PATH)}


@router.get("/editable")
def get_editable() -> dict[str, object]:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"editable": extract_editable_config(config)}


@router.post("/editable")
def update_editable(payload: EditableConfigPayload) -> dict[str, object]:
    patch = payload.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=422, detail="No editable fields provided")

    try:
        result = update_editable_config(patch)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
