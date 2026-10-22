from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from services.auth_service import (
    SESSION_TTL_SECONDS,
    create_browser_session,
    current_browser_session,
    revoke_browser_session,
    session_cookie_name,
    session_cookie_secure,
)


router = APIRouter()


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=4096)


def _disable_auth_response_caching(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.post("/login")
def login(data: LoginRequest, request: Request, response: Response) -> dict[str, object]:
    token = create_browser_session(request, data.username.strip(), data.password)
    response.set_cookie(
        key=session_cookie_name(),
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=session_cookie_secure(),
        samesite="strict",
        path="/",
    )
    _disable_auth_response_caching(response)
    return {"authenticated": True, "expires_in_seconds": SESSION_TTL_SECONDS}


@router.get("/session")
def session(request: Request, response: Response) -> dict[str, object]:
    record = current_browser_session(request)
    _disable_auth_response_caching(response)
    if record is None:
        response.status_code = 401
        return {"authenticated": False}
    return {"authenticated": True, "username": record.username}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    revoke_browser_session(request)
    for cookie_name in ("__Host-scrooge_session", "scrooge_session"):
        response.delete_cookie(
            key=cookie_name,
            path="/",
            secure=cookie_name.startswith("__Host-"),
            httponly=True,
            samesite="strict",
        )
    _disable_auth_response_caching(response)
    return {"authenticated": False}
