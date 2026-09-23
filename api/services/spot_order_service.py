from __future__ import annotations

import os
from typing import Any

from services.command_service import enqueue_control_command
from services.portfolio_service import get_spot_order_intent
from shared.runtime_db import reserve_spot_order_intent, update_spot_order_intent


def queue_spot_order_intent(intent_id: str, *, requested_by: str) -> dict[str, Any]:
    execution_enabled = str(os.getenv("SCROOGE_SPOT_EXECUTION_ENABLED", "0") or "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    if not execution_enabled:
        raise ValueError("Real Binance Spot execution is disabled by the deployment safety switch.")
    intent = get_spot_order_intent(intent_id)
    if intent["is_preview_expired"]:
        update_spot_order_intent(
            intent_id,
            {"status": "expired", "error": "Order preview expired before confirmation."},
            expected_statuses={"previewed"},
        )
        raise ValueError("This order preview has expired. Create a fresh preview before executing.")

    if intent["status"] != "previewed":
        if intent["status"] in {
            "queueing",
            "queued",
            "processing",
            "validated",
            "submitted",
            "accepted",
            "fill_confirmed",
            "accounting_updated",
            "accounting_error",
            "uncertain",
            "filled",
            "partially_filled",
        }:
            return {
                "intent": intent,
                "command_id": intent["command_id"],
                "status": intent["status"],
                "idempotent_replay": True,
            }
        raise ValueError(f"Spot order intent cannot be executed from status {intent['status']}.")

    command_id = intent_id
    reserved, acquired = reserve_spot_order_intent(intent_id, command_id=command_id)
    if reserved is None:
        raise LookupError("Spot order preview was not found.")
    if not acquired:
        return {
            "intent": reserved,
            "command_id": reserved["command_id"],
            "status": reserved["status"],
            "idempotent_replay": True,
        }

    try:
        command = enqueue_control_command(
            action="spot_order",
            requested_by=requested_by,
            payload={"intent_id": intent_id},
            command_id=command_id,
        )
    except Exception as exc:
        update_spot_order_intent(
            intent_id,
            {"status": "failed", "error": str(exc)[:500]},
            expected_statuses={"queueing"},
        )
        raise

    queued = update_spot_order_intent(
        intent_id,
        {"status": "queued", "error": None},
        expected_statuses={"queueing"},
    )
    return {
        **command,
        "intent": queued or reserved,
        "idempotent_replay": False,
    }
