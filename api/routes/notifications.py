from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from services.push_service import (
    get_push_status,
    remove_push_subscription,
    send_test_push,
    upsert_push_subscription,
)

router = APIRouter()


def _requested_by(request: Request) -> str:
    basic = request.headers.get("Authorization", "")
    if basic.startswith("Basic "):
        return "basic-user"
    return "token-user"


class PushSubscriptionKeysPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    p256dh: str = Field(min_length=1)
    auth: str = Field(min_length=1)


class PushSubscriptionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    endpoint: str = Field(min_length=1)
    expirationTime: Optional[int] = None
    keys: PushSubscriptionKeysPayload


class PushSubscriptionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subscription: PushSubscriptionPayload


class PushUnsubscribePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1)


@router.get("")
def get_notifications_config() -> dict[str, object]:
    return get_push_status()


@router.post("/subscribe")
def subscribe_notifications(payload: PushSubscriptionEnvelope, request: Request) -> dict[str, object]:
    try:
        stored = upsert_push_subscription(
            payload.subscription.model_dump(),
            requested_by=_requested_by(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "subscribed": True,
        "endpoint": stored["endpoint"],
        "updated_at": stored["updated_at"],
    }


@router.post("/unsubscribe")
def unsubscribe_notifications(payload: PushUnsubscribePayload) -> dict[str, object]:
    try:
        removed = remove_push_subscription(payload.endpoint)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"removed": removed, "endpoint": payload.endpoint}


@router.post("/test")
def test_notification(payload: PushSubscriptionEnvelope) -> dict[str, object]:
    try:
        delivered = send_test_push(payload.subscription.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"queued": delivered}
