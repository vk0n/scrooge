from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status

from services.auth_service import require_control_or_basic_auth
from services.system_service import restart_service, start_service, stop_service

router = APIRouter()


def _status_response(action: str, svc_status: object) -> dict[str, object]:
    payload = {
        "action": action,
    }
    if hasattr(svc_status, "service_name"):
        payload["service_status"] = {
            "name": svc_status.service_name,
            "running": svc_status.running,
            "active_state": svc_status.active_state,
            "sub_state": svc_status.sub_state,
            "unit_file_state": svc_status.unit_file_state,
        }
    return payload


@router.post("/start")
def start(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        svc_status = start_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("start", svc_status)


@router.post("/stop")
def stop(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        svc_status = stop_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("stop", svc_status)


@router.post("/restart")
def restart(request: Request, x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    require_control_or_basic_auth(request=request, control_token=x_scrooge_control_token)
    try:
        svc_status = restart_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("restart", svc_status)
