from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException, status

from services.system_service import restart_service, start_service, stop_service

router = APIRouter()


def _authorize(control_token: str | None) -> None:
    expected = os.getenv("SCROOGE_CONTROL_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Control token is not configured on server (SCROOGE_CONTROL_TOKEN)",
        )
    if not control_token or control_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid control token")


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
def start(x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    _authorize(x_scrooge_control_token)
    try:
        svc_status = start_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("start", svc_status)


@router.post("/stop")
def stop(x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    _authorize(x_scrooge_control_token)
    try:
        svc_status = stop_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("stop", svc_status)


@router.post("/restart")
def restart(x_scrooge_control_token: str | None = Header(default=None)) -> dict[str, object]:
    _authorize(x_scrooge_control_token)
    try:
        svc_status = restart_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _status_response("restart", svc_status)

