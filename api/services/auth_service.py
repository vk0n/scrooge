from __future__ import annotations

import base64
import binascii
import os

from fastapi import HTTPException, Request, WebSocket, status


def _load_basic_auth_config() -> tuple[str, str] | None:
    username = os.getenv("SCROOGE_GUI_USERNAME", "").strip()
    password = os.getenv("SCROOGE_GUI_PASSWORD", "").strip()
    if username and password:
        return username, password
    return None


def _parse_basic_credentials(authorization_header: str | None) -> tuple[str, str] | None:
    if not authorization_header:
        return None
    if not authorization_header.startswith("Basic "):
        return None

    encoded = authorization_header[6:].strip()
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None

    if ":" not in decoded:
        return None
    username, password = decoded.split(":", 1)
    return username, password


def _is_authorized(
    authorization_header: str | None,
    ws_username: str | None,
    ws_password: str | None,
    cfg_username: str,
    cfg_password: str,
) -> bool:
    basic = _parse_basic_credentials(authorization_header)
    if basic:
        return basic[0] == cfg_username and basic[1] == cfg_password
    return (
        ws_username == cfg_username
        and ws_password == cfg_password
    )


def require_http_auth(request: Request) -> None:
    cfg = _load_basic_auth_config()
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GUI auth is not configured. Set SCROOGE_GUI_USERNAME and SCROOGE_GUI_PASSWORD.",
        )
    cfg_username, cfg_password = cfg

    if _is_authorized(
        authorization_header=request.headers.get("Authorization"),
        ws_username=None,
        ws_password=None,
        cfg_username=cfg_username,
        cfg_password=cfg_password,
    ):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def require_ws_auth(websocket: WebSocket) -> bool:
    cfg = _load_basic_auth_config()
    if not cfg:
        return False
    cfg_username, cfg_password = cfg

    authorized = _is_authorized(
        authorization_header=websocket.headers.get("Authorization"),
        ws_username=websocket.query_params.get("username"),
        ws_password=websocket.query_params.get("password"),
        cfg_username=cfg_username,
        cfg_password=cfg_password,
    )
    if not authorized:
        return False
    return True


def require_control_or_basic_auth(request: Request, control_token: str | None) -> None:
    cfg = _load_basic_auth_config()
    if cfg:
        cfg_username, cfg_password = cfg
        if _is_authorized(
            authorization_header=request.headers.get("Authorization"),
            ws_username=None,
            ws_password=None,
            cfg_username=cfg_username,
            cfg_password=cfg_password,
        ):
            return

    expected_control_token = os.getenv("SCROOGE_CONTROL_TOKEN", "").strip()
    if expected_control_token and control_token and control_token.strip() == expected_control_token:
        return

    if not cfg and not expected_control_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Control auth is not configured. Set SCROOGE_GUI_USERNAME/SCROOGE_GUI_PASSWORD "
                "or SCROOGE_CONTROL_TOKEN."
            ),
        )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
