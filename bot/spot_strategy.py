from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from api.services.portfolio_service import create_strategy_spot_order_intent, load_portfolio_snapshot
from bot.spot_execution import SpotOrderExecutor, SpotOrderValidationError
from core.runtime_db import (
    complete_spot_strategy_campaign_level,
    create_spot_swing,
    ensure_spot_strategy_action,
    initialize_spot_strategy_campaign_capacity,
    list_spot_strategy_actions,
    list_spot_swing_executions,
    list_spot_swings,
    load_spot_order_intent,
    load_spot_strategy_campaign,
    load_spot_swing,
    reconcile_spot_strategy_campaign_progress,
    reserve_spot_order_intent,
    sync_spot_strategy_campaign,
    update_spot_order_intent,
    update_spot_strategy_action,
)
from core.spot_progression import (
    ProgressiveSwingConfig,
    initialize_sell_campaign_capacity,
)
from core.spot_strategy import (
    SPOT_ACTION_PHASES,
    plan_spot_strategy_action,
    spot_action_phase,
)
from core.spot_waiter_cleanup import WaiterCleanupConfig, is_cleanup_reason

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


def _is_permanent_close_block(error: object, action: dict[str, Any] | None = None) -> bool:
    message = str(error or "")
    reason = action.get("reason") if isinstance(action, dict) and isinstance(action.get("reason"), dict) else {}
    fixed_asset_quantity = reason.get("quantity_basis", "remaining_asset") == "remaining_asset"
    return fixed_asset_quantity and (
        "rounds to zero" in message or "Quantity is below Binance minimum" in message
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
        self.config = config or ProgressiveSwingConfig()
        self.cleanup_config = cleanup_config or WaiterCleanupConfig()
        self.account_key = str(account_key or "manual_spot").strip() or "manual_spot"
        self._batch_portfolio: dict[str, Any] | None = None
        self._batch_signal_keys: set[tuple[str, str, int, str]] = set()
        self._batch_excluded_closes: dict[tuple[str, str, int], set[str]] = {}
        self._batch_reconciliation_results: dict[
            tuple[str, str, int], dict[str, Any] | None
        ] = {}

    def order_signals_for_execution(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Schedule one live snapshot in portfolio-wide action phases."""
        self._batch_portfolio = None
        self._batch_signal_keys = set()
        self._batch_excluded_closes = {}
        self._batch_reconciliation_results = {}
        portfolio, _ = load_portfolio_snapshot()
        self._batch_portfolio = portfolio
        staged = [
            {**signal, "_execution_phase": phase}
            for phase in SPOT_ACTION_PHASES
            for signal in signals
        ]
        self._batch_signal_keys = {
            (
                str(signal.get("asset_symbol") or "").strip().upper(),
                str(signal.get("quote_symbol") or "USDT").strip().upper() or "USDT",
                int(signal.get("evaluated_at_ms") or signal.get("current_at_ms") or 0),
                str(signal.get("_execution_phase") or ""),
            )
            for signal in staged
        }
        return staged

    def handle_signal(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        asset = str(signal.get("asset_symbol") or "").strip().upper()
        quote = str(signal.get("quote_symbol") or "USDT").strip().upper() or "USDT"
        opportunity = str(signal.get("opportunity") or "hold").strip().lower()
        level = int(signal.get("level") or 0)
        signal_at_ms = int(signal.get("evaluated_at_ms") or signal.get("current_at_ms") or 0)
        execution_phase = str(signal.get("_execution_phase") or "").strip() or None
        if not asset or opportunity not in {"hold", "buy", "sell"} or signal_at_ms <= 0:
            return None
        batch_key = (asset, quote, signal_at_ms)
        staged_signal_key = (*batch_key, execution_phase or "")
        use_batch_context = staged_signal_key in self._batch_signal_keys
        self._batch_signal_keys.discard(staged_signal_key)
        excluded_close_swing_ids = (
            self._batch_excluded_closes.setdefault(batch_key, set())
            if execution_phase is not None
            else set()
        )

        campaign = sync_spot_strategy_campaign(
            account_key=self.account_key,
            asset_symbol=asset,
            quote_symbol=quote,
            opportunity=opportunity,
            signal_level=level,
            signal_at_ms=signal_at_ms,
            path=self.db_path,
        )
        if execution_phase is not None:
            if batch_key not in self._batch_reconciliation_results:
                self._batch_reconciliation_results[batch_key] = self._reconcile_actions(
                    asset,
                    quote,
                )
            pending = self._batch_reconciliation_results[batch_key]
        else:
            pending = self._reconcile_actions(asset, quote)
        if pending is not None:
            return pending
        reconciled_campaign = reconcile_spot_strategy_campaign_progress(
            account_key=self.account_key,
            asset_symbol=asset,
            quote_symbol=quote,
            campaign_id=str(campaign.get("campaign_id") or ""),
            path=self.db_path,
        )
        if (
            reconciled_campaign is not None
            and int(reconciled_campaign.get("highest_completed_level") or 0)
            > int(campaign.get("highest_completed_level") or 0)
        ):
            self.logger.warning(
                "spot_strategy_campaign_progress_repaired symbol=%s%s campaign_id=%s level=%s",
                asset,
                quote,
                reconciled_campaign.get("campaign_id"),
                reconciled_campaign.get("highest_completed_level"),
            )
        campaign = reconciled_campaign or campaign

        holding, available_quote, free_quote_reserve, available_accumulation_quote = (
            self._load_execution_context(asset, quote, refresh=not use_batch_context)
        )
        if holding is None:
            return None
        if (
            bool(signal.get("strategy_eligible"))
            and campaign.get("active_side") == "sell"
            and campaign.get("campaign_capacity_quantity") is None
        ):
            snapshot = initialize_sell_campaign_capacity(campaign, holding, config=self.config)
            campaign = initialize_spot_strategy_campaign_capacity(
                account_key=self.account_key,
                asset_symbol=asset,
                quote_symbol=quote,
                campaign_id=str(campaign.get("campaign_id") or ""),
                snapshot=snapshot,
                path=self.db_path,
            )
        current_price = float(signal.get("current_price") or holding.get("market_price") or 0.0)
        if not math.isfinite(current_price) or current_price <= 0:
            return None

        completed_actions: list[dict[str, Any]] = []
        while True:
            swing_states = self._swing_states(asset, quote)
            decision = plan_spot_strategy_action(
                signal,
                holding,
                campaign,
                swing_states,
                available_quote=available_quote,
                free_quote_reserve=free_quote_reserve,
                available_accumulation_quote=available_accumulation_quote,
                config=self.config,
                cleanup_config=self.cleanup_config,
                excluded_close_swing_ids=excluded_close_swing_ids,
            )
            if decision is None or decision["action_type"] == "hold":
                return self._signal_batch_result(asset, completed_actions)
            if execution_phase is not None and spot_action_phase(decision) != execution_phase:
                return self._signal_batch_result(asset, completed_actions)

            action_type = str(decision["action_type"])
            if action_type == "close":
                action = self._persist_close_action(asset, quote, decision)
            else:
                action = self._persist_campaign_action(
                    asset=asset,
                    quote=quote,
                    opportunity=opportunity,
                    level=level,
                    signal=signal,
                    signal_at_ms=signal_at_ms,
                    current_price=current_price,
                    campaign=campaign,
                    decision=decision,
                )

            if (
                action_type == "close"
                and action["status"] == "blocked"
                and _is_permanent_close_block(action.get("error"), action)
            ):
                excluded_close_swing_ids.add(str(decision["swing_id"]))
                continue
            if action["status"] == "completed":
                result = self._complete_action(action)
            elif action_type == "campaign_only":
                result = self._complete_action(action)
            else:
                result = self._execute_action(action)

            if result.get("status") == "blocked" and action_type == "close":
                excluded_close_swing_ids.add(str(decision["swing_id"]))
                continue
            if result.get("status") != "completed":
                return result

            completed_actions.append(result)
            if action_type == "close":
                excluded_close_swing_ids.add(str(decision["swing_id"]))

            holding, available_quote, free_quote_reserve, available_accumulation_quote = (
                self._load_execution_context(asset, quote, refresh=True)
            )
            if holding is None:
                return self._signal_batch_result(asset, completed_actions)
            campaign = load_spot_strategy_campaign(
                asset,
                account_key=self.account_key,
                quote_symbol=quote,
                path=self.db_path,
            ) or campaign

    def _persist_campaign_action(
        self,
        *,
        asset: str,
        quote: str,
        opportunity: str,
        level: int,
        signal: dict[str, Any],
        signal_at_ms: int,
        current_price: float,
        campaign: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        campaign_id = str(campaign.get("campaign_id") or "")
        quantity = float(decision["requested_quantity"])
        action_type = str(decision["action_type"])
        action_key = f"{action_type}:{campaign_id}:level:{level}"
        swing_id = f"swing-{_stable_key(action_key)[:24]}" if action_type == "open" else None
        reason = decision["reason"]
        action = ensure_spot_strategy_action(
            {
                "action_key": action_key,
                "account_key": self.account_key,
                "asset_symbol": asset,
                "quote_symbol": quote,
                "campaign_id": campaign_id,
                "action_type": action_type,
                "side": opportunity,
                "signal_level": level,
                "swing_id": swing_id,
                "requested_quantity": quantity,
                "reason": reason,
            },
            path=self.db_path,
        )
        if action["status"] in {"planned", "retryable", "blocked"}:
            action = update_spot_strategy_action(
                action_key,
                {"requested_quantity": quantity, "reason_json": reason},
                path=self.db_path,
            ) or action
        if action_type == "open" and load_spot_swing(str(swing_id), path=self.db_path) is None:
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
        return action

    def _load_execution_context(
        self,
        asset: str,
        quote: str,
        *,
        refresh: bool = True,
    ) -> tuple[dict[str, Any] | None, float, float, float]:
        portfolio = self._batch_portfolio if not refresh else None
        if portfolio is None:
            portfolio, _ = load_portfolio_snapshot()
            self._batch_portfolio = portfolio
        holding = next(
            (
                item
                for item in portfolio.get("holdings", [])
                if item.get("asset_symbol") == asset and item.get("quote_symbol") == quote
            ),
            None,
        )
        exchange = portfolio.get("exchange") if isinstance(portfolio.get("exchange"), dict) else {}
        summary = portfolio.get("summary") if isinstance(portfolio.get("summary"), dict) else {}
        exchange_quote = max(0.0, float(exchange.get("usdt_free") or 0.0))
        managed_quote = max(
            0.0,
            float(summary.get("vault_reserve") or summary.get("dry_powder") or 0.0),
        )
        free_quote = max(
            0.0,
            float(
                summary.get(
                    "vault_reserve_spendable",
                    summary.get("vault_reserve_available"),
                )
                or 0.0
            ),
        )
        return (
            holding,
            min(exchange_quote, managed_quote),
            min(exchange_quote, free_quote),
            min(exchange_quote, free_quote),
        )

    def _signal_batch_result(
        self,
        asset: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not results:
            return None
        profit_closes = [
            result
            for result in results
            if result.get("action_type") == "close"
            and (result.get("reason") or {}).get("close_reason") == "profit_target"
        ]
        profit_swing_ids = [str(result.get("swing_id") or "") for result in profit_closes]
        self.logger.info(
            "spot_signal_batch_completed asset=%s count=%s action_types=%s",
            asset,
            len(results),
            [str(result.get("action_type") or "") for result in results],
        )
        return {
            **results[-1],
            "signal_action_count": len(results),
            "signal_action_keys": [str(result.get("action_key") or "") for result in results],
            "profit_close_batch_count": len(profit_closes),
            "profit_close_batch_swing_ids": profit_swing_ids,
        }

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
            materialized_only=True,
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
        if action["status"] in {"planned", "retryable", "blocked"}:
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
            if intent is not None and intent["status"] in {
                "previewed",
                "queueing",
                "queued",
                *RETRYABLE_INTENT_STATUSES,
            }:
                # No exchange order can still be live in these states. Require the
                # current signal planner to authorize a replacement instead of
                # blindly replaying an action from an older campaign or signal.
                if intent["status"] in {"previewed", "queueing", "queued"}:
                    update_spot_order_intent(
                        intent["intent_id"],
                        {
                            "status": "failed",
                            "error": "Strategy action is waiting for a fresh signal replan.",
                        },
                        expected_statuses={"previewed", "queueing", "queued"},
                        path=self.db_path,
                    )
                update_spot_strategy_action(
                    action["action_key"],
                    {
                        "status": "blocked",
                        "error": "Strategy action is waiting for a fresh signal replan.",
                    },
                    path=self.db_path,
                )
                continue
            if intent is not None and intent["status"] in ACTIVE_INTENT_STATUSES:
                result = self._execute_action(action)
                if result.get("status") == "blocked" and _is_permanent_close_block(result.get("error"), result):
                    continue
                if result.get("status") == "completed":
                    continue
                return result
        return None

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        intent = load_spot_order_intent(action["intent_id"], path=self.db_path) if action.get("intent_id") else None
        try:
            if intent is None or intent["status"] in RETRYABLE_INTENT_STATUSES:
                attempt = int(action.get("attempt_count") or 0) + 1
                try:
                    intent = create_strategy_spot_order_intent(
                        {
                            "asset_symbol": action["asset_symbol"],
                            "quote_symbol": action["quote_symbol"],
                            "side": action["side"],
                            "quantity": action["requested_quantity"],
                            "swing_id": action["swing_id"],
                            "strategy_action_type": action["action_type"],
                            "strategy_action_key": action["action_key"],
                            "reason_text": (
                                "Progressive Swing opening tranche."
                                if action["action_type"] == "open"
                                else "Treasury accumulation from Free Vault Reserve."
                                if action["action_type"] == "accumulate_asset"
                                else (
                                    "Automated stale Bargain cleanup."
                                    if is_cleanup_reason((action.get("reason") or {}).get("close_reason"))
                                    else "Independent Bargain profit close."
                                )
                            ),
                            "reason": action["reason"],
                        }
                    )
                except ValueError as exc:
                    blocked = update_spot_strategy_action(
                        action["action_key"],
                        {
                            "status": "blocked",
                            "attempt_count": attempt,
                            "error": str(exc)[:500],
                        },
                        path=self.db_path,
                    ) or action
                    self.logger.info(
                        "spot_strategy_action_blocked action_key=%s swing_id=%s error=%s",
                        action["action_key"],
                        action["swing_id"],
                        exc,
                    )
                    return blocked
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
        except SpotOrderValidationError as exc:
            blocked = update_spot_strategy_action(
                action["action_key"],
                {"status": "blocked", "error": str(exc)[:500]},
                path=self.db_path,
            ) or action
            self.logger.info(
                "spot_strategy_action_blocked action_key=%s swing_id=%s error=%s",
                action["action_key"],
                action["swing_id"],
                exc,
            )
            return blocked
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
        if (
            action["action_type"] in {"open", "accumulate_asset", "campaign_only"}
            and action.get("campaign_id")
            and action.get("signal_level")
        ):
            consumed_quantity = 0.0
            if action["action_type"] == "open" and intent is not None:
                consumed_quantity = float(intent.get("executed_quantity") or 0.0)
            complete_spot_strategy_campaign_level(
                account_key=action["account_key"],
                asset_symbol=action["asset_symbol"],
                quote_symbol=action["quote_symbol"],
                campaign_id=action["campaign_id"],
                signal_level=int(action["signal_level"]),
                consumed_quantity=consumed_quantity,
                action_key=action["action_key"] if consumed_quantity > 0 else None,
                path=self.db_path,
            )
        updated = update_spot_strategy_action(
            action["action_key"],
            {"status": "completed", "error": None, "completed_at_ms": completed_at_ms},
            path=self.db_path,
        )
        return updated or action
