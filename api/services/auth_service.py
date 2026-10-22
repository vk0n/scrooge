from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping

from fastapi import HTTPException, Request, WebSocket, status


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


SESSION_TTL_SECONDS = _positive_int_env("SCROOGE_SESSION_TTL_SECONDS", 43_200)
AUTH_FAILURE_WINDOW_SECONDS = _positive_int_env("SCROOGE_AUTH_FAILURE_WINDOW_SECONDS", 300)
AUTH_MAX_FAILURES = _positive_int_env("SCROOGE_AUTH_MAX_FAILURES", 5)
AUTH_LOCKOUT_SECONDS = _positive_int_env("SCROOGE_AUTH_LOCKOUT_SECONDS", 900)
MAX_ACTIVE_SESSIONS = _positive_int_env("SCROOGE_AUTH_MAX_ACTIVE_SESSIONS", 32)


@dataclass(frozen=True)
class SessionRecord:
    username: str
    expires_at: float


_auth_lock = threading.Lock()
_sessions: dict[str, SessionRecord] = {}
_failed_attempts: dict[str, deque[float]] = {}
_blocked_until: dict[str, float] = {}


def _load_basic_auth_config() -> tuple[str, str] | None:
    username = os.getenv("SCROOGE_GUI_USERNAME", "").strip()
    password = os.getenv("SCROOGE_GUI_PASSWORD", "").strip()
    if username and password:
        return username, password
    return None


def _secure_equals(left: str, right: str) -> bool:
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _parse_basic_credentials(authorization_header: str | None) -> tuple[str, str] | None:
    if not authorization_header or not authorization_header.startswith("Basic "):
        return None

    encoded = authorization_header[6:].strip()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None

    if ":" not in decoded:
        return None
    username, password = decoded.split(":", 1)
    return username, password


def _credentials_match(username: str, password: str, cfg_username: str, cfg_password: str) -> bool:
    username_matches = _secure_equals(username, cfg_username)
    password_matches = _secure_equals(password, cfg_password)
    return username_matches and password_matches


def _is_basic_authorized(authorization_header: str | None, cfg_username: str, cfg_password: str) -> bool:
    basic = _parse_basic_credentials(authorization_header)
    if not basic:
        return False
    return _credentials_match(basic[0], basic[1], cfg_username, cfg_password)


def session_cookie_secure() -> bool:
    return os.getenv("SCROOGE_SESSION_COOKIE_SECURE", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def session_cookie_name() -> str:
    return "__Host-scrooge_session" if session_cookie_secure() else "scrooge_session"


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _purge_expired_sessions(now: float) -> None:
    expired = [digest for digest, record in _sessions.items() if record.expires_at <= now]
    for digest in expired:
        _sessions.pop(digest, None)


def _session_record(token: str | None) -> SessionRecord | None:
    if not token:
        return None
    now = time.monotonic()
    digest = _session_digest(token)
    with _auth_lock:
        _purge_expired_sessions(now)
        return _sessions.get(digest)


def _session_token_from_cookies(cookies: Mapping[str, str]) -> str | None:
    return cookies.get("__Host-scrooge_session") or cookies.get("scrooge_session")


def _normalized_ip(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    try:
        return str(ipaddress.ip_address(raw_value.strip()))
    except ValueError:
        return None


def _client_key(request: Request) -> str:
    cloudflare_ip = _normalized_ip(request.headers.get("CF-Connecting-IP"))
    if cloudflare_ip:
        return cloudflare_ip
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        forwarded_ip = _normalized_ip(forwarded.split(",")[-1])
        if forwarded_ip:
            return forwarded_ip
    if request.client:
        return request.client.host
    return "unknown"


def _remaining_lockout_seconds(client_key: str, now: float) -> int:
    with _auth_lock:
        blocked_until = _blocked_until.get(client_key, 0.0)
        if blocked_until <= now:
            _blocked_until.pop(client_key, None)
            return 0
        return max(1, int(blocked_until - now + 0.999))


def _require_not_rate_limited(client_key: str) -> None:
    retry_after = _remaining_lockout_seconds(client_key, time.monotonic())
    if retry_after:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _record_auth_failure(client_key: str) -> None:
    now = time.monotonic()
    cutoff = now - AUTH_FAILURE_WINDOW_SECONDS
    with _auth_lock:
        attempts = _failed_attempts.setdefault(client_key, deque())
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        attempts.append(now)
        if len(attempts) >= AUTH_MAX_FAILURES:
            _blocked_until[client_key] = now + AUTH_LOCKOUT_SECONDS
            attempts.clear()


def _clear_auth_failures(client_key: str) -> None:
    with _auth_lock:
        _failed_attempts.pop(client_key, None)
        _blocked_until.pop(client_key, None)


def create_browser_session(request: Request, username: str, password: str) -> str:
    cfg = _load_basic_auth_config()
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GUI auth is not configured. Set SCROOGE_GUI_USERNAME and SCROOGE_GUI_PASSWORD.",
        )

    client_key = _client_key(request)
    _require_not_rate_limited(client_key)
    cfg_username, cfg_password = cfg
    if not _credentials_match(username, password, cfg_username, cfg_password):
        _record_auth_failure(client_key)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    _clear_auth_failures(client_key)
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    record = SessionRecord(username=cfg_username, expires_at=now + SESSION_TTL_SECONDS)
    with _auth_lock:
        _purge_expired_sessions(now)
        if len(_sessions) >= MAX_ACTIVE_SESSIONS:
            oldest_digest = min(_sessions, key=lambda digest: _sessions[digest].expires_at)
            _sessions.pop(oldest_digest, None)
        _sessions[_session_digest(token)] = record
    return token


def revoke_browser_session(request: Request) -> None:
    token = _session_token_from_cookies(request.cookies)
    if not token:
        return
    with _auth_lock:
        _sessions.pop(_session_digest(token), None)


def current_browser_session(request: Request) -> SessionRecord | None:
    return _session_record(_session_token_from_cookies(request.cookies))


def request_auth_actor(request: Request, *, fallback: str = "unknown") -> str:
    if current_browser_session(request):
        return "session-user"
    if request.headers.get("Authorization", "").startswith("Basic "):
        return "basic-user"
    return fallback


def require_http_auth(request: Request) -> None:
    if current_browser_session(request):
        return

    cfg = _load_basic_auth_config()
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GUI auth is not configured. Set SCROOGE_GUI_USERNAME and SCROOGE_GUI_PASSWORD.",
        )

    authorization = request.headers.get("Authorization")
    if authorization:
        client_key = _client_key(request)
        _require_not_rate_limited(client_key)
        if _is_basic_authorized(authorization, cfg[0], cfg[1]):
            _clear_auth_failures(client_key)
            return
        _record_auth_failure(client_key)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def require_ws_auth(websocket: WebSocket) -> bool:
    if _session_record(_session_token_from_cookies(websocket.cookies)):
        return True

    cfg = _load_basic_auth_config()
    if not cfg:
        return False
    return _is_basic_authorized(websocket.headers.get("Authorization"), cfg[0], cfg[1])


def require_control_or_basic_auth(request: Request, control_token: str | None) -> None:
    if current_browser_session(request):
        return

    cfg = _load_basic_auth_config()
    authorization = request.headers.get("Authorization")
    if cfg and authorization:
        client_key = _client_key(request)
        _require_not_rate_limited(client_key)
        if _is_basic_authorized(authorization, cfg[0], cfg[1]):
            _clear_auth_failures(client_key)
            return
        _record_auth_failure(client_key)

    expected_control_token = os.getenv("SCROOGE_CONTROL_TOKEN", "").strip()
    if expected_control_token and control_token and _secure_equals(control_token.strip(), expected_control_token):
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
