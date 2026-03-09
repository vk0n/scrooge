from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Callable

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - bot image installs redis dependency
    redis = None
    RedisError = Exception


REDIS_HOST = os.getenv("SCROOGE_REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("SCROOGE_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("SCROOGE_REDIS_DB", "0"))
CONTROL_QUEUE_KEY = os.getenv("SCROOGE_CONTROL_QUEUE_KEY", "scrooge:control:queue")
COMMAND_STATUS_PREFIX = os.getenv("SCROOGE_COMMAND_STATUS_PREFIX", "scrooge:control:command:")
COMMAND_STATUS_TTL_SECONDS = int(os.getenv("SCROOGE_COMMAND_STATUS_TTL_SECONDS", "86400"))
SUPPORTED_ACTIONS = {"start", "stop", "restart"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _status_key(command_id: str) -> str:
    return f"{COMMAND_STATUS_PREFIX}{command_id}"


def get_control_client() -> Any | None:
    """
    Create Redis client for bot-side control command polling.
    Returns None when Redis support is unavailable or unreachable.
    """
    if redis is None:
        print("[CONTROL] redis package is not installed; command polling disabled")
        return None

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    try:
        client.ping()
    except RedisError as exc:
        print(f"[CONTROL] Redis unavailable ({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}): {exc}")
        return None
    return client


def _update_status(client: Any, command_id: str, fields: dict[str, Any], logger: Callable[[str], None]) -> None:
    payload = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in fields.items()}
    key = _status_key(command_id)
    try:
        pipe = client.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.expire(key, COMMAND_STATUS_TTL_SECONDS)
        pipe.execute()
    except RedisError as exc:
        logger(f"[CONTROL] Failed to update command status for {command_id}: {exc}")


def process_pending_commands(
    client: Any | None,
    state: dict[str, Any],
    save_state_fn: Callable[[dict[str, Any]], None],
    logger: Callable[[str], None] = print,
) -> tuple[dict[str, Any], bool]:
    """
    Consume queued control commands and apply them to bot runtime state.
    Returns updated state and a restart marker (config reload requested).
    """
    if client is None:
        return state, False

    restart_requested = False

    while True:
        try:
            raw_payload = client.lpop(CONTROL_QUEUE_KEY)
        except RedisError as exc:
            logger(f"[CONTROL] Failed to poll command queue: {exc}")
            break

        if raw_payload is None:
            break

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            logger(f"[CONTROL] Ignoring malformed command payload: {raw_payload}")
            continue

        command_id = str(payload.get("id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        if not command_id:
            logger(f"[CONTROL] Ignoring command without id: {payload}")
            continue

        _update_status(
            client,
            command_id,
            {"status": "processing", "updated_at": _now_iso(), "message": ""},
            logger,
        )

        if action not in SUPPORTED_ACTIONS:
            _update_status(
                client,
                command_id,
                {
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "message": f"Unsupported action: {action}",
                },
                logger,
            )
            continue

        try:
            was_enabled = bool(state.get("trading_enabled", True))
            if action == "start":
                state["trading_enabled"] = True
                message = "Trading resumed." if not was_enabled else "Trading is already running."
            elif action == "stop":
                state["trading_enabled"] = False
                message = "Trading paused." if was_enabled else "Trading is already paused."
            else:
                state["trading_enabled"] = True
                restart_requested = True
                message = "Trading restart requested. Config reload scheduled."

            save_state_fn(state)
        except Exception as exc:  # noqa: BLE001
            _update_status(
                client,
                command_id,
                {
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "message": f"Failed to apply command: {exc}",
                },
                logger,
            )
            continue

        _update_status(
            client,
            command_id,
            {
                "status": "completed",
                "updated_at": _now_iso(),
                "message": message,
                "trading_status": {
                    "trading_enabled": bool(state.get("trading_enabled", True)),
                },
            },
            logger,
        )
        logger(f"[CONTROL] Applied command {action} ({command_id})")

    return state, restart_requested
