from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from bot.spot_account import normalize_spot_account_snapshot
from shared.runtime_db import (
    apply_spot_accumulation_target_ratchet,
    apply_spot_swing_target_ratchet,
    append_portfolio_transaction,
    append_spot_swing_execution,
    ensure_portfolio_asset_policies,
    list_portfolio_transactions,
    list_spot_order_intents,
    list_spot_swing_executions,
    load_spot_order_intent,
    load_spot_swing,
    save_exchange_account_snapshot,
    update_spot_order_intent,
)
from shared.spot_accounting import ensure_spot_quote_leg
from shared.spot_execution_rules import (
    format_decimal as _format_decimal,
    normalize_market_quantity as _market_quantity,
    validate_market_close_remainder as _validate_close_remainder,
    validate_market_notional as _validate_notional,
)
from shared.spot_swing import calculate_swing_economics
from shared.treasury_ledger import append_treasury_event, project_portfolio_transaction


class SpotOrderUncertainError(RuntimeError):
    pass


class SpotOrderAccountingError(RuntimeError):
    pass


class SpotOrderRejectedError(RuntimeError):
    pass


class SpotOrderValidationError(ValueError):
    """A deterministic pre-submission rejection; Binance cannot have accepted the order."""


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _balance_map(account: dict[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for item in account.get("balances", []):
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset") or "").strip().upper()
        free = _as_float(item.get("free"))
        locked = _as_float(item.get("locked"))
        if asset and free is not None and locked is not None:
            output[asset] = {"free": free, "locked": locked, "total": free + locked}
    return output


def _execution_summary(client: Any, symbol: str, order: dict[str, Any]) -> dict[str, Any]:
    order_id = order.get("orderId")
    snapshot = dict(order)
    deadline = time.monotonic() + 5.0
    while str(snapshot.get("status") or "").upper() not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
        if time.monotonic() >= deadline or order_id is None:
            break
        time.sleep(0.2)
        try:
            latest = client.get_order(symbol=symbol, orderId=order_id)
        except Exception:  # Binance's FULL response remains authoritative fallback data.
            break
        if isinstance(latest, dict):
            snapshot = latest

    trades: list[dict[str, Any]] = []
    if order_id is not None:
        try:
            raw_trades = client.get_my_trades(symbol=symbol, orderId=order_id)
        except Exception:
            raw_trades = None
        if isinstance(raw_trades, list):
            trades = [item for item in raw_trades if isinstance(item, dict)]
    if not trades and isinstance(order.get("fills"), list):
        trades = [item for item in order["fills"] if isinstance(item, dict)]

    executed_quantity = _as_float(snapshot.get("executedQty")) or 0.0
    executed_quote = _as_float(snapshot.get("cummulativeQuoteQty")) or 0.0
    weighted_quote = 0.0
    weighted_quantity = 0.0
    commissions: dict[str, float] = {}
    fills: list[dict[str, Any]] = []
    latest_time_ms = snapshot.get("updateTime") or order.get("transactTime")
    for fill_index, trade in enumerate(trades):
        quantity = _as_float(trade.get("qty")) or 0.0
        price = _as_float(trade.get("price")) or 0.0
        quote_quantity = _as_float(trade.get("quoteQty"))
        weighted_quantity += quantity
        weighted_quote += quote_quantity if quote_quantity is not None else quantity * price
        commission = _as_float(trade.get("commission")) or 0.0
        commission_asset = str(trade.get("commissionAsset") or "").strip().upper()
        if commission_asset and commission > 0:
            commissions[commission_asset] = commissions.get(commission_asset, 0.0) + commission
        trade_time = trade.get("time")
        if trade_time is not None:
            latest_time_ms = max(int(latest_time_ms or 0), int(trade_time))
        if quantity > 0 and price > 0:
            trade_id = trade.get("id") if trade.get("id") is not None else trade.get("tradeId")
            fills.append(
                {
                    "fill_index": fill_index,
                    "trade_id": str(trade_id) if trade_id is not None else None,
                    "quantity": quantity,
                    "price": price,
                    "quote_quantity": quote_quantity if quote_quantity is not None else quantity * price,
                    "fee_amount": commission if commission > 0 else None,
                    "fee_asset": commission_asset or None,
                    "executed_at_ms": int(trade_time) if trade_time is not None else None,
                }
            )
    if weighted_quantity > 0:
        executed_quantity = weighted_quantity
        executed_quote = weighted_quote
    average_price = executed_quote / executed_quantity if executed_quantity > 0 else None
    if not fills and executed_quantity > 0 and average_price is not None:
        fills.append(
            {
                "fill_index": 0,
                "trade_id": None,
                "quantity": executed_quantity,
                "price": average_price,
                "quote_quantity": executed_quote,
                "fee_amount": None,
                "fee_asset": None,
                "executed_at_ms": int(latest_time_ms) if latest_time_ms is not None else None,
            }
        )
    fee_asset = next(iter(commissions)) if len(commissions) == 1 else None
    fee_amount = commissions.get(fee_asset) if fee_asset is not None else None
    return {
        "status": str(snapshot.get("status") or order.get("status") or "UNKNOWN").upper(),
        "order_id": str(order_id) if order_id is not None else None,
        "client_order_id": str(snapshot.get("clientOrderId") or order.get("clientOrderId") or "") or None,
        "executed_quantity": executed_quantity,
        "submitted_quantity": _as_float(snapshot.get("origQty") or order.get("origQty")) or executed_quantity,
        "executed_quote_quantity": executed_quote,
        "average_price": average_price,
        "fee_amount": fee_amount,
        "fee_asset": fee_asset,
        "commissions": commissions,
        "fills": fills,
        "executed_at_ms": int(latest_time_ms) if latest_time_ms is not None else int(datetime.now(UTC).timestamp() * 1000),
    }


class SpotOrderExecutor:
    def __init__(self, client: Any, *, logger: Any, db_path: Path | None = None) -> None:
        self.client = client
        self.logger = logger
        self.db_path = db_path

    def _refresh_account(self) -> dict[str, Any]:
        account = self.client.get_account(recvWindow=5000)
        snapshot = normalize_spot_account_snapshot(account)
        save_exchange_account_snapshot(snapshot, path=self.db_path)
        return account

    def recover_pending(self) -> list[dict[str, Any]]:
        """Resume only previously confirmed work; never invent a new intent."""
        if str(os.getenv("SCROOGE_SPOT_EXECUTION_ENABLED", "0") or "0").strip().lower() in {
            "", "0", "false", "no", "off",
        }:
            return []
        recoverable = list_spot_order_intents(
            statuses={
                "processing",
                "validated",
                "submitted",
                "accepted",
                "fill_confirmed",
                "accounting_error",
                "accounting_updated",
                "uncertain",
            },
            path=self.db_path,
        )
        outcomes: list[dict[str, Any]] = []
        for intent in recoverable:
            try:
                result = self.execute(intent["intent_id"])
                outcomes.append({"intent_id": intent["intent_id"], "status": "recovered", "result": result})
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    "spot_order_recovery_failed intent_id=%s prior_status=%s error=%s",
                    intent["intent_id"],
                    intent["status"],
                    exc,
                )
                outcomes.append({"intent_id": intent["intent_id"], "status": "unresolved", "error": str(exc)})
        return outcomes

    def _safe_error_state(
        self,
        intent_id: str,
        updates: dict[str, Any],
        *,
        expected_statuses: set[str],
    ) -> None:
        try:
            update_spot_order_intent(
                intent_id,
                updates,
                expected_statuses=expected_statuses,
                path=self.db_path,
            )
        except OSError as persist_error:
            self.logger.error(
                "spot_order_error_state_persist_failed intent_id=%s status=%s error=%s",
                intent_id,
                updates.get("status"),
                persist_error,
            )

    def _submit(self, intent: dict[str, Any], quantity: Decimal) -> dict[str, Any]:
        params = {
            "symbol": intent["symbol"],
            "side": intent["side"].upper(),
            "type": "MARKET",
            "quantity": _format_decimal(quantity),
            "newClientOrderId": intent["client_order_id"],
            "newOrderRespType": "FULL",
            "recvWindow": 5000,
        }
        try:
            response = self.client.create_order(**params)
        except Exception as submit_error:
            try:
                recovered = self.client.get_order(
                    symbol=intent["symbol"],
                    origClientOrderId=intent["client_order_id"],
                )
            except Exception as lookup_error:
                raise SpotOrderUncertainError(
                    "Binance order submission outcome is uncertain; reconcile by client order ID "
                    f"{intent['client_order_id']} before retrying."
                ) from lookup_error
            if not isinstance(recovered, dict):
                raise submit_error
            response = recovered
        if not isinstance(response, dict):
            raise RuntimeError("Binance did not return an order response.")
        return response

    def _validate_for_submission(self, intent: dict[str, Any]) -> Decimal:
        request = intent.get("request") if isinstance(intent.get("request"), dict) else {}
        treasury_intake = bool(request.get("treasury_intake"))
        strategy_action_type = str(request.get("strategy_action_type") or "").strip().lower()
        treasury_accumulation = (
            intent["source"] == "strategy"
            and not intent.get("swing_id")
            and intent["side"] == "buy"
            and strategy_action_type == "accumulate_asset"
        )
        accumulation_enabled = str(
            os.getenv("SCROOGE_SPOT_TREASURY_ACCUMULATION_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if treasury_accumulation and not accumulation_enabled:
            raise ValueError("Treasury accumulation is disabled by the production safety gate.")
        swing = None
        if intent["source"] == "strategy" and not intent.get("swing_id") and not treasury_accumulation:
            raise ValueError("A standalone Strategy Spot order must be Treasury accumulation.")
        if intent.get("swing_id"):
            swing = load_spot_swing(intent["swing_id"], path=self.db_path)
            if swing is None:
                raise ValueError("The linked Spot order references a missing Swing.")
            if swing["status"] == "closed":
                raise ValueError("A closed Spot Swing cannot accept another linked order.")
            if swing["account_key"] != intent["account_key"]:
                raise ValueError("Spot order and Swing belong to different Treasury accounts.")
            if swing["asset_symbol"] != intent["asset_symbol"] or swing["quote_symbol"] != intent["quote_symbol"]:
                raise ValueError("Spot order and Swing markets do not match.")
        account = self._refresh_account()
        if account.get("canTrade") is not True:
            raise ValueError("Binance reports that Spot trading is unavailable for this account.")
        balances = _balance_map(account)
        ticker = self.client.get_symbol_ticker(symbol=intent["symbol"])
        market_price = _as_float(ticker.get("price") if isinstance(ticker, dict) else None)
        if market_price is None or market_price <= 0:
            raise ValueError("Binance did not return a valid Spot market price.")
        symbol_info = self.client.get_symbol_info(intent["symbol"])
        if not isinstance(symbol_info, dict) or str(symbol_info.get("status") or "").upper() != "TRADING":
            raise ValueError("This Binance Spot symbol is not currently trading.")
        if str(symbol_info.get("baseAsset") or "").upper() != intent["asset_symbol"]:
            raise ValueError("Binance symbol base asset does not match the Treasury intent.")
        if str(symbol_info.get("quoteAsset") or "").upper() != intent["quote_symbol"]:
            raise ValueError("Binance symbol quote asset does not match the Treasury intent.")
        quantity, _ = _market_quantity(symbol_info, intent["requested_quantity"])
        _validate_notional(symbol_info, quantity=quantity, price=market_price)
        if swing is not None and intent["side"] != swing["origin_side"]:
            economics = calculate_swing_economics(
                swing,
                list_spot_swing_executions(swing["swing_id"], path=self.db_path),
                current_price=market_price,
            )
            _validate_close_remainder(
                symbol_info,
                remaining_quantity=float(economics.get("remaining_quantity") or 0.0),
                closing_quantity=quantity,
                price=market_price,
            )

        from api.services.portfolio_service import load_portfolio_snapshot

        portfolio, _ = load_portfolio_snapshot()
        holding = next(
            (
                item
                for item in portfolio["holdings"]
                if item["asset_symbol"] == intent["asset_symbol"]
                and item["quote_symbol"] == intent["quote_symbol"]
            ),
            None,
        )
        if treasury_intake and holding is not None:
            raise ValueError(
                f"{intent['asset_symbol']} entered the Treasury after preview. Use its Binance custody Trade action."
            )
        if holding is None and not (treasury_intake and intent["source"] == "manual" and intent["side"] == "buy"):
            raise ValueError("Treasury asset disappeared before execution.")
        if treasury_accumulation and holding.get("trading_objective") != "accumulate_asset":
            raise ValueError("Treasury accumulation objective changed before execution.")
        quantity_float = float(quantity)
        if intent["side"] == "sell":
            if holding is None:
                raise ValueError("A real Spot sell requires an existing Treasury holding.")
            immediate = _as_float(holding.get("immediately_sellable_quantity")) or 0.0
            projected = (_as_float(holding.get("quantity")) or 0.0) - quantity_float
            protected_floor = _as_float(holding.get("protected_floor_quantity")) or 0.0
            if quantity_float > immediate + 0.00000001:
                raise ValueError(
                    f"Sell rejected: only {immediate:.8f} {intent['asset_symbol']} is immediately sellable."
                )
            if projected < protected_floor - 0.00000001:
                raise ValueError("Sell rejected because it would breach the Protected Floor.")
            exchange_free = balances.get(intent["asset_symbol"], {}).get("free", 0.0)
            if quantity_float > exchange_free + 0.00000001:
                raise ValueError("Sell rejected because Binance free balance changed after preview.")
        else:
            quote_free = balances.get(intent["quote_symbol"], {}).get("free", 0.0)
            estimated_fee_rate = max(
                0.0,
                float(os.getenv("SCROOGE_SPOT_ESTIMATED_FEE_RATE", "0.001") or 0.001),
            )
            required_quote = quantity_float * market_price * (1.0 + estimated_fee_rate)
            if required_quote > quote_free + 0.00000001:
                raise ValueError("Buy rejected because Binance USDT balance changed after preview.")
            if treasury_accumulation:
                free_reserve = _as_float(portfolio["summary"].get("vault_reserve_available")) or 0.0
                if required_quote > free_reserve + 0.00000001:
                    raise ValueError(
                        "Treasury accumulation rejected because Free Vault Reserve changed after preview."
                    )
        return quantity

    def _recover_order(self, intent: dict[str, Any]) -> dict[str, Any]:
        try:
            order = self.client.get_order(
                symbol=intent["symbol"],
                origClientOrderId=intent["client_order_id"],
            )
        except Exception as lookup_error:
            raise SpotOrderUncertainError(
                "Binance order outcome is still uncertain; reconcile client order ID "
                f"{intent['client_order_id']} before any new submission."
            ) from lookup_error
        if not isinstance(order, dict):
            raise SpotOrderUncertainError("Binance did not return a recoverable order response.")
        return order

    def _confirm_fill(self, intent: dict[str, Any], order: dict[str, Any]) -> dict[str, Any]:
        summary = _execution_summary(self.client, intent["symbol"], order)
        if summary["executed_quantity"] <= 0:
            if summary["status"] in {"CANCELED", "REJECTED", "EXPIRED"}:
                terminal_status = "rejected" if summary["status"] == "REJECTED" else "failed"
                update_spot_order_intent(
                    intent["intent_id"],
                    {"status": terminal_status, "error": f"Binance order ended with status {summary['status']}.", "result_json": summary},
                    expected_statuses={"submitted", "accepted", "uncertain"},
                    path=self.db_path,
                )
                raise SpotOrderRejectedError(
                    f"Binance order ended with status {summary['status']} and no confirmed fill."
                )
            raise SpotOrderUncertainError(
                f"Binance order {summary['order_id'] or intent['client_order_id']} has no confirmed fill "
                f"and remains in status {summary['status']}. Reconcile it before retrying."
            )
        if summary["status"] not in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
            update_spot_order_intent(
                intent["intent_id"],
                {
                    "status": "uncertain",
                    "exchange_order_id": summary["order_id"],
                    "executed_quantity": summary["executed_quantity"],
                    "executed_quote_quantity": summary["executed_quote_quantity"],
                    "average_price": summary["average_price"],
                    "fee_amount": summary["fee_amount"],
                    "fee_asset": summary["fee_asset"],
                    "error": "The Binance order has partial fills but is not terminal yet.",
                    "result_json": summary,
                },
                expected_statuses={"submitted", "accepted", "uncertain"},
                path=self.db_path,
            )
            raise SpotOrderUncertainError(
                f"Binance order {summary['order_id'] or intent['client_order_id']} has partial fills and remains "
                f"in status {summary['status']}. Accounting will wait for a terminal exchange status."
            )
        update_spot_order_intent(
            intent["intent_id"],
            {
                "status": "fill_confirmed",
                "exchange_order_id": summary["order_id"],
                "executed_quantity": summary["executed_quantity"],
                "executed_quote_quantity": summary["executed_quote_quantity"],
                "average_price": summary["average_price"],
                "fee_amount": summary["fee_amount"],
                "fee_asset": summary["fee_asset"],
                "result_json": summary,
            },
            expected_statuses={"submitted", "accepted", "uncertain"},
            path=self.db_path,
        )
        return summary

    def _settle_confirmed_fill(self, intent: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        if intent.get("swing_id"):
            swing = load_spot_swing(intent["swing_id"], path=self.db_path)
            if swing is None:
                raise ValueError("The Spot order references a missing Swing.")
            if swing["account_key"] != intent["account_key"]:
                raise ValueError("Spot order and Swing belong to different Treasury accounts.")
            if swing["asset_symbol"] != intent["asset_symbol"] or swing["quote_symbol"] != intent["quote_symbol"]:
                raise ValueError("Spot order and Swing markets do not match.")

        commissions = summary.get("commissions") if isinstance(summary.get("commissions"), dict) else {}
        base_fee = _as_float(commissions.get(intent["asset_symbol"])) or 0.0
        quote_fee = _as_float(commissions.get(intent["quote_symbol"])) or 0.0
        ledger_quantity = float(summary["executed_quantity"])
        ledger_quantity += base_fee * (-1 if intent["side"] == "buy" else 1)
        ledger_price = _as_float(summary.get("average_price"))
        executed_quote = _as_float(summary.get("executed_quote_quantity")) or 0.0
        if intent["side"] == "buy" and ledger_quantity > 0 and executed_quote > 0:
            embedded_quote_fee = quote_fee if summary.get("fee_asset") != intent["quote_symbol"] else 0.0
            ledger_price = (executed_quote + embedded_quote_fee) / ledger_quantity
        transaction_id = f"spot-order:{intent['intent_id']}"
        source_label = "Strategy" if intent["source"] == "strategy" else "Manual"
        transaction = {
            "transaction_id": transaction_id,
            "account_key": intent["account_key"],
            "executed_at": datetime.fromtimestamp(int(summary["executed_at_ms"]) / 1000, tz=UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "tx_type": intent["side"],
            "asset_symbol": intent["asset_symbol"],
            "quote_symbol": intent["quote_symbol"],
            "quantity": ledger_quantity,
            "price": ledger_price,
            "fee_amount": summary.get("fee_amount"),
            "fee_asset": summary.get("fee_asset"),
            "source": f"binance_{intent['source']}",
            "status": "settled",
            "note": f"{source_label} Binance Spot {intent['side'].upper()} settled by the unified executor.",
            "external_order_id": summary.get("order_id"),
            "custody_location": "binance",
            "source_custody": None,
            "destination_custody": None,
            "spot_order_intent_id": intent["intent_id"],
            "swing_id": intent.get("swing_id"),
            "executed_quote_quantity": executed_quote,
            "commissions": summary.get("commissions") or {},
        }
        try:
            existing = list_portfolio_transactions(account_key=intent["account_key"], path=self.db_path)
            persisted_transaction = next(
                (item for item in existing if item.get("transaction_id") == transaction_id),
                None,
            )
            if persisted_transaction is None:
                persisted_transaction = append_portfolio_transaction(transaction, path=self.db_path)
            ensure_spot_quote_leg(persisted_transaction, path=self.db_path)
            project_portfolio_transaction(persisted_transaction, path=self.db_path)

            request = intent.get("request") if isinstance(intent.get("request"), dict) else {}
            if bool(request.get("treasury_intake")):
                initial_target = _as_float(request.get("initial_target_quantity"))
                if initial_target is None or initial_target <= 0:
                    raise ValueError("Treasury intake is missing its requested Target Holding.")
                ensure_portfolio_asset_policies(
                    [
                        {
                            "asset_symbol": intent["asset_symbol"],
                            "quote_symbol": intent["quote_symbol"],
                            "target_quantity": initial_target,
                            "minimum_holding_pct": 100.0,
                            "trading_objective": "accumulate_cash",
                        }
                    ],
                    account_key=intent["account_key"],
                    path=self.db_path,
                )

            if intent.get("swing_id"):
                fills = summary.get("fills") if isinstance(summary.get("fills"), list) else []
                if not fills:
                    fills = [
                        {
                            "fill_index": 0,
                            "trade_id": None,
                            "quantity": summary["executed_quantity"],
                            "price": summary["average_price"],
                            "quote_quantity": summary.get("executed_quote_quantity"),
                            "fee_amount": summary.get("fee_amount"),
                            "fee_asset": summary.get("fee_asset"),
                            "executed_at_ms": summary.get("executed_at_ms"),
                        }
                    ]
                for fill_index, fill in enumerate(fills):
                    trade_id = str(fill.get("trade_id") or "").strip() or None
                    execution_suffix = f"trade:{trade_id}" if trade_id else f"fill:{fill_index}"
                    append_spot_swing_execution(
                        {
                            "execution_id": f"spot-order:{intent['intent_id']}:{execution_suffix}",
                            "swing_id": intent["swing_id"],
                            "spot_order_intent_id": intent["intent_id"],
                            "venue": intent["venue"],
                            "symbol": intent["symbol"],
                            "side": intent["side"],
                            "quantity": fill["quantity"],
                            "price": fill["price"],
                            "quote_quantity": fill.get("quote_quantity"),
                            "fee_amount": fill.get("fee_amount"),
                            "fee_asset": fill.get("fee_asset"),
                            "exchange_order_id": summary.get("order_id"),
                            "exchange_trade_id": trade_id,
                            "exchange_execution_key": (
                                f"{intent['venue']}:{intent['symbol']}:trade:{trade_id}"
                                if trade_id
                                else f"{intent['venue']}:{intent['symbol']}:order:{summary.get('order_id')}:fill:{fill_index}"
                            ),
                            "source": intent["source"],
                            "reason_text": intent.get("reason_text"),
                            "reason": intent.get("reason") or {},
                            "executed_at_ms": fill.get("executed_at_ms") or summary["executed_at_ms"],
                            "order_summary": summary,
                        },
                        path=self.db_path,
                    )
                ratchet = apply_spot_swing_target_ratchet(intent["swing_id"], path=self.db_path)
                if ratchet is not None:
                    append_treasury_event(
                        code="treasury_target_ratchet_applied",
                        tone="positive",
                        message=(
                            f"I secured {ratchet['applied_gain_quantity']:,.8f} {intent['asset_symbol']} "
                            "of Swing profit inside the protected hoard."
                        ),
                        source_ref=f"spot_swing_target_ratchet:{intent['swing_id']}",
                        context=ratchet,
                        path=self.db_path,
                    )
            elif (
                intent["source"] == "strategy"
                and intent["side"] == "buy"
                and str(request.get("strategy_action_type") or "").strip().lower()
                == "accumulate_asset"
            ):
                action_key = str(request.get("strategy_action_key") or "").strip()
                ratchet = apply_spot_accumulation_target_ratchet(
                    action_key,
                    intent_id=intent["intent_id"],
                    net_acquired_quantity=ledger_quantity,
                    deployed_quote_quantity=executed_quote + quote_fee,
                    average_price=float(ledger_price or summary["average_price"]),
                    fee_amount=_as_float(summary.get("fee_amount")),
                    fee_asset=str(summary.get("fee_asset") or "").strip().upper() or None,
                    path=self.db_path,
                )
                append_treasury_event(
                    code="treasury_accumulation_completed",
                    tone="positive",
                    message=(
                        f"I deployed ${ratchet['deployed_quote_quantity']:,.2f} from Free Vault Reserve "
                        f"and protected {ratchet['applied_gain_quantity']:,.8f} "
                        f"{intent['asset_symbol']} in Target."
                    ),
                    source_ref=f"spot_accumulation_target_ratchet:{action_key}",
                    context=ratchet,
                    path=self.db_path,
                )
        except Exception as accounting_error:  # noqa: BLE001
            self._safe_error_state(
                intent["intent_id"],
                {
                    "status": "accounting_error",
                    "error": f"Binance fill was confirmed, but local accounting failed: {str(accounting_error)[:350]}",
                    "result_json": summary,
                },
                expected_statuses={"fill_confirmed", "accounting_error", "accounting_updated"},
            )
            raise SpotOrderAccountingError(
                f"Binance order {summary.get('order_id')} filled, but Treasury/Swing reconciliation failed. "
                "Do not retry this order; retry settlement only."
            ) from accounting_error

        result = {
            **summary,
            "intent_id": intent["intent_id"],
            "ledger_transaction_id": transaction_id,
            "requested_quantity": intent["requested_quantity"],
            "side": intent["side"],
            "asset_symbol": intent["asset_symbol"],
            "quote_symbol": intent["quote_symbol"],
            "source": intent["source"],
            "swing_id": intent.get("swing_id"),
        }
        updated = update_spot_order_intent(
            intent["intent_id"],
            {"status": "accounting_updated", "error": None, "result_json": result},
            expected_statuses={"fill_confirmed", "accounting_error", "accounting_updated"},
            path=self.db_path,
        )
        final_status = "filled" if str(summary.get("status") or "").upper() == "FILLED" else "partially_filled"
        update_spot_order_intent(
            intent["intent_id"],
            {"status": final_status, "error": None, "result_json": result},
            expected_statuses={"accounting_updated"} if updated is not None else {final_status},
            path=self.db_path,
        )
        return result

    def execute(self, intent_id: str) -> dict[str, Any]:
        execution_enabled = str(os.getenv("SCROOGE_SPOT_EXECUTION_ENABLED", "0") or "0").strip().lower() not in {
            "", "0", "false", "no", "off",
        }
        if not execution_enabled:
            raise RuntimeError("Real Binance Spot execution is disabled by the deployment safety switch.")
        intent = load_spot_order_intent(intent_id, path=self.db_path)
        if intent is None:
            raise LookupError("Spot order intent was not found.")
        if intent["status"] in {"filled", "partially_filled"}:
            return intent["result"] or intent
        if intent["status"] in {"fill_confirmed", "accounting_error", "accounting_updated"}:
            if not isinstance(intent.get("result"), dict):
                raise SpotOrderAccountingError("Confirmed fill data is missing; manual reconciliation is required.")
            return self._settle_confirmed_fill(intent, intent["result"])

        recover_exchange_order = intent["status"] in {"submitted", "accepted", "uncertain"}
        if intent["status"] not in {"queueing", "queued", "processing", "validated", "submitted", "accepted", "uncertain"}:
            raise ValueError(f"Spot order intent cannot execute from status {intent['status']}.")
        order_may_exist = recover_exchange_order
        try:
            if recover_exchange_order:
                order = self._recover_order(intent)
            else:
                update_spot_order_intent(
                    intent_id,
                    {"status": "processing", "error": None},
                    expected_statuses={"queueing", "queued", "processing", "validated"},
                    path=self.db_path,
                )
                try:
                    quantity = self._validate_for_submission(intent)
                except ValueError as exc:
                    raise SpotOrderValidationError(str(exc)) from exc
                update_spot_order_intent(
                    intent_id,
                    {"status": "validated", "error": None},
                    expected_statuses={"processing", "validated"},
                    path=self.db_path,
                )
                order = self._submit(intent, quantity)
                order_may_exist = True
                exchange_order_id = str(order.get("orderId")) if order.get("orderId") is not None else None
                update_spot_order_intent(
                    intent_id,
                    {"status": "submitted", "exchange_order_id": exchange_order_id, "result_json": {"submitted_order": order}},
                    expected_statuses={"validated"},
                    path=self.db_path,
                )
                update_spot_order_intent(
                    intent_id,
                    {"status": "accepted", "exchange_order_id": exchange_order_id, "result_json": {"submitted_order": order}},
                    expected_statuses={"submitted"},
                    path=self.db_path,
                )

            summary = self._confirm_fill(intent, order)
            result = self._settle_confirmed_fill(intent, summary)
            try:
                self._refresh_account()
            except Exception as refresh_error:  # noqa: BLE001
                self.logger.warning("spot_order_post_fill_refresh_failed intent_id=%s error=%s", intent_id, refresh_error)
            self.logger.info(
                "spot_order_filled intent_id=%s source=%s swing_id=%s order_id=%s side=%s symbol=%s quantity=%s",
                intent_id,
                intent["source"],
                intent.get("swing_id"),
                summary["order_id"],
                intent["side"],
                intent["symbol"],
                summary["executed_quantity"],
            )
            return result
        except SpotOrderUncertainError as exc:
            self._safe_error_state(
                intent_id,
                {"status": "uncertain", "error": str(exc)[:500]},
                expected_statuses={"processing", "validated", "submitted", "accepted", "uncertain"},
            )
            self.logger.error("spot_order_outcome_uncertain intent_id=%s error=%s", intent_id, exc)
            raise
        except SpotOrderAccountingError as exc:
            self.logger.error("spot_order_accounting_error intent_id=%s error=%s", intent_id, exc)
            raise
        except SpotOrderRejectedError as exc:
            self.logger.warning("spot_order_rejected intent_id=%s error=%s", intent_id, exc)
            raise
        except SpotOrderValidationError as exc:
            self._safe_error_state(
                intent_id,
                {"status": "failed", "error": str(exc)[:500]},
                expected_statuses={"processing", "validated"},
            )
            self.logger.info("spot_order_validation_failed intent_id=%s error=%s", intent_id, exc)
            raise
        except Exception as exc:
            if order_may_exist:
                uncertain = SpotOrderUncertainError(
                    "Binance may have accepted the order, but Scrooge could not complete reconciliation. "
                    f"Reconcile client order ID {intent['client_order_id']} before any new submission."
                )
                self._safe_error_state(
                    intent_id,
                    {"status": "uncertain", "error": str(uncertain)[:500]},
                    expected_statuses={"processing", "validated", "submitted", "accepted", "fill_confirmed", "uncertain"},
                )
                self.logger.error("spot_order_outcome_uncertain intent_id=%s error=%s", intent_id, exc)
                raise uncertain from exc
            self._safe_error_state(
                intent_id,
                {"status": "failed", "error": str(exc)[:500]},
                expected_statuses={"processing", "validated"},
            )
            self.logger.warning("spot_order_failed intent_id=%s error=%s", intent_id, exc)
            raise
