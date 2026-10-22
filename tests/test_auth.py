from __future__ import annotations

import base64
import os
import sys
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from routes.auth import LoginRequest, login, logout, session  # noqa: E402
from services import auth_service  # noqa: E402


def _request(
    *,
    client_ip: str = "203.0.113.10",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers or [],
            "client": (client_ip, 12345),
            "server": ("testserver", 443),
        }
    )


def _cookie_request(cookie_name: str, token: str) -> Request:
    return _request(headers=[(b"cookie", f"{cookie_name}={token}".encode("ascii"))])


class _FakeWebSocket:
    def __init__(
        self,
        *,
        cookies: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
    ) -> None:
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.query_params = query_params or {}


class AuthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "SCROOGE_GUI_USERNAME": "operator",
                "SCROOGE_GUI_PASSWORD": "correct horse battery staple",
                "SCROOGE_SESSION_COOKIE_SECURE": "1",
            },
            clear=False,
        )
        self.env.start()
        self._reset_auth_state()

    def tearDown(self) -> None:
        self._reset_auth_state()
        self.env.stop()

    @staticmethod
    def _reset_auth_state() -> None:
        with auth_service._auth_lock:
            auth_service._sessions.clear()
            auth_service._failed_attempts.clear()
            auth_service._blocked_until.clear()

    def _login(self) -> tuple[str, str, Response]:
        response = Response()
        payload = login(
            LoginRequest(username="operator", password="correct horse battery staple"),
            _request(),
            response,
        )
        self.assertTrue(payload["authenticated"])
        cookie_header = response.headers["set-cookie"]
        parsed = SimpleCookie()
        parsed.load(cookie_header)
        cookie_name = auth_service.session_cookie_name()
        return cookie_name, parsed[cookie_name].value, response

    def test_login_issues_hardened_cookie_and_session_authenticates_http(self) -> None:
        cookie_name, token, response = self._login()
        cookie_header = response.headers["set-cookie"]

        self.assertEqual(cookie_name, "__Host-scrooge_session")
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("Secure", cookie_header)
        self.assertIn("SameSite=strict", cookie_header)
        self.assertIn("Path=/", cookie_header)
        self.assertEqual(response.headers["cache-control"], "no-store")
        auth_service.require_http_auth(_cookie_request(cookie_name, token))

    def test_session_cookie_authenticates_websocket_without_query_credentials(self) -> None:
        cookie_name, token, _ = self._login()
        websocket = _FakeWebSocket(cookies={cookie_name: token})

        self.assertTrue(auth_service.require_ws_auth(websocket))  # type: ignore[arg-type]

    def test_session_actor_is_preserved_for_audit_logs(self) -> None:
        cookie_name, token, _ = self._login()

        actor = auth_service.request_auth_actor(_cookie_request(cookie_name, token))

        self.assertEqual(actor, "session-user")

    def test_websocket_query_credentials_are_rejected(self) -> None:
        websocket = _FakeWebSocket(
            query_params={"username": "operator", "password": "correct horse battery staple"}
        )

        self.assertFalse(auth_service.require_ws_auth(websocket))  # type: ignore[arg-type]

    def test_logout_revokes_session_and_expires_both_cookie_names(self) -> None:
        cookie_name, token, _ = self._login()
        request = _cookie_request(cookie_name, token)
        response = Response()

        payload = logout(request, response)

        self.assertFalse(payload["authenticated"])
        self.assertIsNone(auth_service.current_browser_session(request))
        set_cookie_headers = [
            value.decode("latin-1")
            for key, value in response.raw_headers
            if key.lower() == b"set-cookie"
        ]
        self.assertEqual(len(set_cookie_headers), 2)
        self.assertTrue(all("Max-Age=0" in value for value in set_cookie_headers))

    def test_session_endpoint_rejects_missing_cookie(self) -> None:
        response = Response()

        payload = session(_request(), response)

        self.assertFalse(payload["authenticated"])
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_basic_auth_remains_available_for_machine_clients(self) -> None:
        raw = base64.b64encode(b"operator:correct horse battery staple").decode("ascii")
        request = _request(headers=[(b"authorization", f"Basic {raw}".encode("ascii"))])

        auth_service.require_http_auth(request)

    def test_repeated_bad_logins_are_rate_limited(self) -> None:
        request = _request(client_ip="203.0.113.99")
        payload = LoginRequest(username="operator", password="wrong password")

        for _ in range(auth_service.AUTH_MAX_FAILURES):
            with self.assertRaises(HTTPException) as raised:
                login(payload, request, Response())
            self.assertEqual(raised.exception.status_code, 401)

        with self.assertRaises(HTTPException) as raised:
            login(payload, request, Response())
        self.assertEqual(raised.exception.status_code, 429)
        self.assertIn("Retry-After", raised.exception.headers or {})


if __name__ == "__main__":
    unittest.main()
