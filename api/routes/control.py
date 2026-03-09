from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status

from services.auth_service import require_control_or_basic_auth
from services.command_service import enqueue_control_command, get_command_status

router = APIRouter()


def _requested_by(request: Request) -> str:
    basic = request.headers.get("Authorization", "")
    if basic.startswith("Basic "):
        return "basic-user"
    return "token-user"


@router.post("/start")
def start(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        command = enqueue_control_command(action="start", requested_by=_requested_by(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return command


@router.post("/stop")
def stop(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        command = enqueue_control_command(action="stop", requested_by=_requested_by(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return command


@router.post("/restart")
def restart(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        command = enqueue_control_command(action="restart", requested_by=_requested_by(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return command


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
