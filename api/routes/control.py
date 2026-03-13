from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.auth_service import require_control_or_basic_auth
from services.command_service import enqueue_control_command, get_command_status

router = APIRouter()


def _requested_by(request: Request) -> str:
    basic = request.headers.get("Authorization", "")
    if basic.startswith("Basic "):
        return "basic-user"
    return "token-user"


class UpdateLevelRequest(BaseModel):
    value: float = Field(..., gt=0)


class SuggestTradeRequest(BaseModel):
    side: Literal["buy", "sell"]


def _enqueue_action(
    request: Request,
    action: str,
    control_token: str | None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=control_token)
    try:
        return enqueue_control_command(action=action, requested_by=_requested_by(request), payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/start")
def start(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    return _enqueue_action(request=request, action="start", control_token=x_scrooge_control_token)


@router.post("/stop")
def stop(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    return _enqueue_action(request=request, action="stop", control_token=x_scrooge_control_token)


@router.post("/restart")
def restart(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    return _enqueue_action(request=request, action="restart", control_token=x_scrooge_control_token)


@router.post("/close-position")
def close_position(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    return _enqueue_action(request=request, action="close_position", control_token=x_scrooge_control_token)


@router.post("/suggest-trade")
def suggest_trade(
    data: SuggestTradeRequest,
    request: Request,
    x_scrooge_control_token: str | None = Header(default=None),
) -> dict[str, object]:
    return _enqueue_action(
        request=request,
        action="suggest_trade",
        control_token=x_scrooge_control_token,
        payload={"side": data.side},
    )


@router.post("/update-sl")
def update_sl(
    data: UpdateLevelRequest,
    request: Request,
    x_scrooge_control_token: str | None = Header(default=None),
) -> dict[str, object]:
    return _enqueue_action(
        request=request,
        action="update_sl",
        control_token=x_scrooge_control_token,
        payload={"value": data.value},
    )


@router.post("/update-tp")
def update_tp(
    data: UpdateLevelRequest,
    request: Request,
    x_scrooge_control_token: str | None = Header(default=None),
) -> dict[str, object]:
    return _enqueue_action(
        request=request,
        action="update_tp",
        control_token=x_scrooge_control_token,
        payload={"value": data.value},
    )


@router.get("/commands/{command_id}")
def command_status(
    command_id: str,
    request: Request,
    x_scrooge_control_token: str | None = Header(default=None),
) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        status_payload = get_command_status(command_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    if status_payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found")

    return status_payload
