from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import redis

from services.system_service import get_service_status


REDIS_HOST = os.getenv("SCROOGE_REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("SCROOGE_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("SCROOGE_REDIS_DB", "0"))
CONTROL_QUEUE_KEY = os.getenv("SCROOGE_CONTROL_QUEUE_KEY", "scrooge:control:queue")
COMMAND_STATUS_PREFIX = os.getenv("SCROOGE_COMMAND_STATUS_PREFIX", "scrooge:control:command:")
COMMAND_STATUS_TTL_SECONDS = int(os.getenv("SCROOGE_COMMAND_STATUS_TTL_SECONDS", "86400"))
COMMAND_STALE_AFTER_SECONDS = int(os.getenv("SCROOGE_COMMAND_STALE_AFTER_SECONDS", "20"))
SUPPORTED_ACTIONS = {
    "start",
    "stop",
    "restart",
    "close_position",
    "suggest_trade",
    "update_sl",
    "update_tp",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


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

    status_value = str(payload.get("status", "")).strip().lower()
    if status_value in {"pending", "processing"}:
        try:
            service_status = get_service_status()
        except RuntimeError:
            service_status = None

        if service_status is not None:
            payload["service_status"] = {
                "service_name": service_status.service_name,
                "running": service_status.running,
                "active_state": service_status.active_state,
                "sub_state": service_status.sub_state,
                "unit_file_state": service_status.unit_file_state,
            }
            if not service_status.running:
                payload["status"] = "failed"
                payload["message"] = (
                    f"Instruction could not be delivered because the Scrooge runtime is offline "
                    f"({service_status.active_state})."
                )
                return payload

        updated_at = _parse_iso(payload.get("updated_at"))
        created_at = _parse_iso(payload.get("created_at"))
        reference_ts = updated_at or created_at
        if reference_ts is not None:
            age_seconds = (datetime.now(UTC) - reference_ts.astimezone(UTC)).total_seconds()
            if age_seconds >= COMMAND_STALE_AFTER_SECONDS:
                payload["status"] = "failed"
                payload["message"] = (
                    "The instruction sat unanswered on Scrooge's desk for too long. "
                    "He did not acknowledge it in time; the office wire may be down or Scrooge may not be tending the command queue."
                )
                return payload
