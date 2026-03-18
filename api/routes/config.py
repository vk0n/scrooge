from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.config_service import (
    CONFIG_PATH,
    extract_editable_config,
    load_config,
    load_raw_config_text,
    update_editable_config,
    update_raw_config_text,
)

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


class EditableIntervalsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    small: str | None = Field(default=None, min_length=2, max_length=6)
    medium: str | None = Field(default=None, min_length=2, max_length=6)
    big: str | None = Field(default=None, min_length=2, max_length=6)

    @field_validator("small", "medium", "big")
    @classmethod
    def validate_interval(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"\d+[mhdw]", normalized):
            raise ValueError("interval must look like 1m, 1h, 4h, 1d, or 1w")
        return normalized


class EditableIndicatorInputsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ema: str | None = Field(default=None)
    rsi: str | None = Field(default=None)
    bb: str | None = Field(default=None)
    atr: str | None = Field(default=None)

    @field_validator("ema", "rsi", "bb", "atr")
    @classmethod
    def validate_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized == "discrete":
            return "closed"
        if normalized == "realtime":
            return "intrabar"
        if normalized not in {"closed", "intrabar"}:
            raise ValueError("indicator input mode must be either closed or intrabar")
        return normalized


class EditableConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    live: bool | None = None
    symbol: str | None = Field(default=None, min_length=3, max_length=20)
    leverage: int | None = Field(default=None, ge=1, le=125)
    initial_balance: float | None = Field(default=None, gt=0)
    use_full_balance: bool | None = None
    qty: float | None = Field(default=None, gt=0)
    intervals: EditableIntervalsPayload | None = None
    indicator_inputs: EditableIndicatorInputsPayload | None = None
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


class RawConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)


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


@router.get("/raw")
def get_raw_config() -> dict[str, object]:
    try:
        raw_text = load_raw_config_text()
        config = load_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "raw_text": raw_text,
        "editable": extract_editable_config(config),
        "path": str(CONFIG_PATH),
    }


@router.post("/raw")
def update_raw_config(payload: RawConfigPayload) -> dict[str, object]:
    try:
        result = update_raw_config_text(payload.raw_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
