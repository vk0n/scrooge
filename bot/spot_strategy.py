from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from api.services.portfolio_service import create_strategy_spot_order_intent, load_portfolio_snapshot
from bot.spot_execution import SpotOrderExecutor
from shared.runtime_db import (
    complete_spot_strategy_campaign_level,
    create_spot_swing,
    ensure_spot_strategy_action,
    list_spot_strategy_actions,
    list_spot_swing_executions,
    list_spot_swings,
    load_spot_order_intent,
    load_spot_swing,
    reserve_spot_order_intent,
    sync_spot_strategy_campaign,
    update_spot_order_intent,
    update_spot_strategy_action,
)
from shared.spot_progression import (
    ProgressiveSwingConfig,
)
from shared.spot_strategy import plan_spot_strategy_action
from shared.spot_waiter_cleanup import WaiterCleanupConfig, is_cleanup_reason

ACTIVE_INTENT_STATUSES = {
    "queueing",
    "queued",
    "processing",
    "validated",
    "submitted",
    "accepted",
    "fill_confirmed",
    "accounting_error",
    "accounting_updated",
    "uncertain",
}
COMPLETED_INTENT_STATUSES = {"filled", "partially_filled"}
RETRYABLE_INTENT_STATUSES = {"failed", "rejected"}


def progressive_swing_config_from_env() -> ProgressiveSwingConfig:
    return ProgressiveSwingConfig(
        close_profit_pct=float(os.getenv("SCROOGE_SPOT_SWING_CLOSE_PROFIT_PCT", "5") or 5),
        estimated_fee_rate=float(os.getenv("SCROOGE_SPOT_ESTIMATED_FEE_RATE", "0.001") or 0.001),
    )


def _stable_key(*parts: object) -> str:
    payload = json.dumps(parts, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProgressiveSpotSwingExecutor:
    """Turns persisted signals into idempotent Swing actions through the unified executor."""

    def __init__(
        self,
        order_executor: SpotOrderExecutor,
        *,
        logger: Any,
        db_path: Path | None = None,
        config: ProgressiveSwingConfig | None = None,
        cleanup_config: WaiterCleanupConfig | None = None,
        account_key: str = "manual_spot",
    ) -> None:
        self.order_executor = order_executor
        self.logger = logger
        self.db_path = db_path
        self.config = config or progressive_swing_config_from_env()
        self.cleanup_config = cleanup_config or WaiterCleanupConfig()
        self.account_key = str(account_key or "manual_spot").strip() or "manual_spot"

    def handle_signal(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        asset = str(signal.get("asset_symbol") or "").strip().upper()
        quote = str(signal.get("quote_symbol") or "USDT").strip().upper() or "USDT"
        opportunity = str(signal.get("opportunity") or "hold").strip().lower()
        level = int(signal.get("level") or 0)
        signal_at_ms = int(signal.get("evaluated_at_ms") or signal.get("current_at_ms") or 0)
        if not asset or opportunity not in {"hold", "buy", "sell"} or signal_at_ms <= 0:
            return None

        campaign = sync_spot_strategy_campaign(
            account_key=self.account_key,
            asset_symbol=asset,
            quote_symbol=quote,
            opportunity=opportunity,
            signal_level=level,
            signal_at_ms=signal_at_ms,
            path=self.db_path,
        )
        pending = self._reconcile_actions(asset, quote)
        if pending is not None:
            return pending

        portfolio, _ = load_portfolio_snapshot()
        holding = next(
            (
                item
                for item in portfolio.get("holdings", [])
                if item.get("asset_symbol") == asset and item.get("quote_symbol") == quote
            ),
            None,
        )
        if holding is None:
            return None
        current_price = float(signal.get("current_price") or holding.get("market_price") or 0.0)
        if not math.isfinite(current_price) or current_price <= 0:
            return None
        exchange = portfolio.get("exchange") if isinstance(portfolio.get("exchange"), dict) else {}
        summary = portfolio.get("summary") if isinstance(portfolio.get("summary"), dict) else {}
        exchange_quote = max(0.0, float(exchange.get("usdt_free") or 0.0))
        managed_quote = max(
            0.0,
            float(summary.get("vault_reserve") or summary.get("dry_powder") or 0.0),
        )
        opening_quote = max(0.0, float(summary.get("vault_reserve_available") or 0.0))
        available_quote = min(exchange_quote, managed_quote)
        available_opening_quote = min(exchange_quote, opening_quote)

        decision = plan_spot_strategy_action(
            signal,
            holding,
            campaign,
            self._swing_states(asset, quote),
            available_quote=available_quote,
            available_opening_quote=available_opening_quote,
            config=self.config,
            cleanup_config=self.cleanup_config,
        )
        if decision is None:
            return None
        if decision["action_type"] == "hold":
            return None
        if decision["action_type"] == "close":
            return self._execute_action(self._persist_close_action(asset, quote, decision))

        campaign_id = str(campaign.get("campaign_id") or "")
        quantity = float(decision["requested_quantity"])
        action_key = f"open:{campaign_id}:level:{level}"
        swing_id = f"swing-{_stable_key(action_key)[:24]}"
        reason = decision["reason"]
        action = ensure_spot_strategy_action(
            {
                "action_key": action_key,
                "account_key": self.account_key,
                "asset_symbol": asset,
                "quote_symbol": quote,
                "campaign_id": campaign_id,
                "action_type": "open",
                "side": opportunity,
                "signal_level": level,
                "swing_id": swing_id,
                "requested_quantity": quantity,
                "reason": reason,
            },
            path=self.db_path,
        )
        if action["status"] in {"planned", "retryable"}:
            action = update_spot_strategy_action(
                action_key,
                {"requested_quantity": quantity, "reason_json": reason},
                path=self.db_path,
            ) or action
        if load_spot_swing(swing_id, path=self.db_path) is None:
            create_spot_swing(
                {
                    "swing_id": swing_id,
                    "account_key": self.account_key,
                    "asset_symbol": asset,
                    "quote_symbol": quote,
                    "origin_side": opportunity,
                    "trading_objective": signal.get("trading_objective"),
                    "planned_quantity": quantity,
                    "reference_state": {
                        "rolling_reference_price": signal.get("reference_price"),
                        "signal_price": current_price,
                        "signal_at_ms": signal_at_ms,
                        "campaign_id": campaign_id,
                        "signal_level": level,
                    },
                    "strategy_reason": reason,
                    "source": "strategy",
                    "opened_at_ms": signal_at_ms,
                },
                path=self.db_path,
            )
        return self._execute_action(action)

    def _plan_close(
        self,
        *,
        asset: str,
        quote: str,
        current_price: float,
        available_quote: float,
    ) -> dict[str, Any] | None:
        decision = plan_spot_strategy_action(
            {"opportunity": "hold", "current_price": current_price},
            {},
            {},
            self._swing_states(asset, quote),
            available_quote=available_quote,
            config=self.config,
            cleanup_config=self.cleanup_config,
        )
        if decision is None or decision["action_type"] != "close":
            return None
        return self._persist_close_action(asset, quote, decision)

    def _swing_states(self, asset: str, quote: str) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for swing in list_spot_swings(
            account_key=self.account_key,
            asset_symbol=asset,
            quote_symbol=quote,
            path=self.db_path,
        ):
            if swing["source"] != "strategy" or swing["status"] not in {"open", "partially_closed"}:
                continue
            states.append(
                {
                    "swing": swing,
                    "executions": list_spot_swing_executions(swing["swing_id"], path=self.db_path),
                }
            )
        return states

    def _persist_close_action(
        self,
        asset: str,
        quote: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        executions = list_spot_swing_executions(decision["swing_id"], path=self.db_path)
        candidate = {
            "action_key": (
                f"close:{decision['swing_id']}:"
                f"{_stable_key([item['execution_id'] for item in executions])[:16]}"
            ),
            "account_key": self.account_key,
            "asset_symbol": asset,
            "quote_symbol": quote,
            **decision,
        }
        action = ensure_spot_strategy_action(candidate, path=self.db_path)
        if action["status"] in {"planned", "retryable"}:
            action = update_spot_strategy_action(
                action["action_key"],
                {
                    "requested_quantity": candidate["requested_quantity"],
                    "reason_json": candidate["reason"],
                },
                path=self.db_path,
            ) or action
        return action

    def _reconcile_actions(self, asset: str, quote: str) -> dict[str, Any] | None:
        actions = list_spot_strategy_actions(
            account_key=self.account_key,
            asset_symbol=asset,
            quote_symbol=quote,
            statuses={"planned", "intent_created", "executing", "retryable"},
            path=self.db_path,
        )
        for action in actions:
            intent = load_spot_order_intent(action["intent_id"], path=self.db_path) if action.get("intent_id") else None
            if intent is not None and intent["status"] in COMPLETED_INTENT_STATUSES:
                self._complete_action(action)
                continue
            if intent is not None and intent["status"] in {"queueing", "queued"}:
                return self._execute_action(action)
            if intent is not None and intent["status"] in ACTIVE_INTENT_STATUSES:
                update_spot_strategy_action(action["action_key"], {"status": "executing"}, path=self.db_path)
                return action
            if intent is not None and intent["status"] in RETRYABLE_INTENT_STATUSES:
                action = update_spot_strategy_action(
                    action["action_key"],
                    {"status": "retryable", "error": intent.get("error")},
                    path=self.db_path,
                ) or action
        return None

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        intent = load_spot_order_intent(action["intent_id"], path=self.db_path) if action.get("intent_id") else None
        try:
            if intent is None or intent["status"] in RETRYABLE_INTENT_STATUSES:
                attempt = int(action.get("attempt_count") or 0) + 1
                intent = create_strategy_spot_order_intent(
                    {
                        "asset_symbol": action["asset_symbol"],
                        "quote_symbol": action["quote_symbol"],
                        "side": action["side"],
                        "quantity": action["requested_quantity"],
                        "swing_id": action["swing_id"],
                        "reason_text": (
                            "Progressive Swing opening tranche."
                            if action["action_type"] == "open"
                            else (
                                "Automated stale Bargain cleanup."
                                if is_cleanup_reason((action.get("reason") or {}).get("close_reason"))
                                else "Independent Bargain profit close."
                            )
                        ),
                        "reason": action["reason"],
                    }
                )
                action = update_spot_strategy_action(
                    action["action_key"],
                    {
                        "intent_id": intent["intent_id"],
                        "status": "intent_created",
                        "attempt_count": attempt,
                        "error": None,
                    },
                    path=self.db_path,
                ) or action
            if intent["status"] == "previewed":
                _, acquired = reserve_spot_order_intent(
                    intent["intent_id"],
                    command_id=f"strategy:{action['action_key']}:{action['attempt_count']}",
                    path=self.db_path,
                )
                if acquired:
                    update_spot_order_intent(
                        intent["intent_id"],
                        {"status": "queued"},
                        expected_statuses={"queueing"},
                        path=self.db_path,
                    )
            update_spot_strategy_action(action["action_key"], {"status": "executing"}, path=self.db_path)
            result = self.order_executor.execute(intent["intent_id"])
            completed = self._complete_action(action)
            self.logger.info(
                "spot_strategy_action_completed action_key=%s action_type=%s swing_id=%s intent_id=%s",
                action["action_key"],
                action["action_type"],
                action["swing_id"],
                intent["intent_id"],
            )
            return {**completed, "result": result}
        except Exception as exc:  # noqa: BLE001
            persisted_intent = (
                load_spot_order_intent(intent["intent_id"], path=self.db_path)
                if intent is not None
                else None
            )
            next_status = (
                "executing"
                if persisted_intent is not None and persisted_intent["status"] in ACTIVE_INTENT_STATUSES
                else "retryable"
            )
            updated = update_spot_strategy_action(
                action["action_key"],
                {"status": next_status, "error": str(exc)[:500]},
                path=self.db_path,
            )
            self.logger.warning(
                "spot_strategy_action_deferred action_key=%s status=%s error=%s",
                action["action_key"],
                next_status,
                exc,
            )
            return updated or action

    def _complete_action(self, action: dict[str, Any]) -> dict[str, Any]:
        intent = load_spot_order_intent(action.get("intent_id"), path=self.db_path) if action.get("intent_id") else None
        completed_at_ms = int(intent["updated_at_ms"]) if intent is not None else int(action["updated_at_ms"])
        updated = update_spot_strategy_action(
            action["action_key"],
            {"status": "completed", "error": None, "completed_at_ms": completed_at_ms},
            path=self.db_path,
        )
        if action["action_type"] == "open" and action.get("campaign_id") and action.get("signal_level"):
            complete_spot_strategy_campaign_level(
                account_key=action["account_key"],
                asset_symbol=action["asset_symbol"],
                quote_symbol=action["quote_symbol"],
                campaign_id=action["campaign_id"],
                signal_level=int(action["signal_level"]),
                path=self.db_path,
            )
        return updated or action
