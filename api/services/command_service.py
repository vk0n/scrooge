from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import redis


REDIS_HOST = os.getenv("SCROOGE_REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("SCROOGE_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("SCROOGE_REDIS_DB", "0"))
CONTROL_QUEUE_KEY = os.getenv("SCROOGE_CONTROL_QUEUE_KEY", "scrooge:control:queue")
COMMAND_STATUS_PREFIX = os.getenv("SCROOGE_COMMAND_STATUS_PREFIX", "scrooge:control:command:")
COMMAND_STATUS_TTL_SECONDS = int(os.getenv("SCROOGE_COMMAND_STATUS_TTL_SECONDS", "86400"))
SUPPORTED_ACTIONS = {
    "start",
    "stop",
    "restart",
    "close_position",
    "update_sl",
    "update_tp",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def _status_key(command_id: str) -> str:
    return f"{COMMAND_STATUS_PREFIX}{command_id}"


def enqueue_control_command(action: str, requested_by: str | None, payload: dict[str, Any] | None = None) -> dict[str, str]:
    normalized_action = action.strip().lower()
    if normalized_action not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")

    command_id = uuid.uuid4().hex
    created_at = _now_iso()
    status_key = _status_key(command_id)

    command_payload = {
        "id": command_id,
        "action": normalized_action,
        "payload": payload or {},
        "requested_by": requested_by or "unknown",
        "created_at": created_at,
    }
    status_payload = {
        "command_id": command_id,
        "action": normalized_action,
        "status": "pending",
        "message": "",
        "payload": json.dumps(payload or {}),
        "requested_by": requested_by or "unknown",
        "created_at": created_at,
        "updated_at": created_at,
    }

    try:
        client = _client()
        pipe = client.pipeline()
        pipe.hset(status_key, mapping=status_payload)
        pipe.expire(status_key, COMMAND_STATUS_TTL_SECONDS)
        pipe.rpush(CONTROL_QUEUE_KEY, json.dumps(command_payload))
        pipe.execute()
    except redis.RedisError as exc:
        raise RuntimeError(f"Failed to enqueue control command: {exc}") from exc

    return {
        "command_id": command_id,
        "status": "pending",
        "queued_at": created_at,
        "action": normalized_action,
    }


def get_command_status(command_id: str) -> dict[str, Any] | None:
    try:
        payload = _client().hgetall(_status_key(command_id))
    except redis.RedisError as exc:
        raise RuntimeError(f"Failed to read control command status: {exc}") from exc

    if not payload:
        return None

    service_status_raw = payload.get("service_status")
    if service_status_raw:
        try:
            payload["service_status"] = json.loads(service_status_raw)
        except json.JSONDecodeError:
            payload["service_status"] = service_status_raw
    raw_command_payload = payload.get("payload")
    if raw_command_payload:
        try:
            payload["payload"] = json.loads(raw_command_payload)
        except json.JSONDecodeError:
            payload["payload"] = raw_command_payload
    return payload
