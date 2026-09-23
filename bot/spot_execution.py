from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any

from bot.spot_account import normalize_spot_account_snapshot
from shared.runtime_db import (
    append_portfolio_transaction,
    list_portfolio_transactions,
    load_spot_order_intent,
    save_exchange_account_snapshot,
    update_spot_order_intent,
)


class SpotOrderUncertainError(RuntimeError):
    pass


class SpotOrderAccountingError(RuntimeError):
    pass


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _as_decimal(value: Any) -> Decimal | None:
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _market_quantity(symbol_info: dict[str, Any], requested_quantity: float) -> tuple[Decimal, Decimal]:
    filters = {
        str(item.get("filterType") or ""): item
        for item in symbol_info.get("filters", [])
        if isinstance(item, dict)
    }
    market_lot_filter = filters.get("MARKET_LOT_SIZE")
    market_step = _as_decimal(market_lot_filter.get("stepSize")) if isinstance(market_lot_filter, dict) else None
    lot_filter = market_lot_filter if market_step is not None and market_step > 0 else filters.get("LOT_SIZE")
    if not isinstance(lot_filter, dict):
        raise ValueError("Binance did not return a market lot-size rule for this symbol.")
    step_size = _as_decimal(lot_filter.get("stepSize"))
    min_quantity = _as_decimal(lot_filter.get("minQty")) or Decimal("0")
    max_quantity = _as_decimal(lot_filter.get("maxQty")) or Decimal("0")
    requested = _as_decimal(requested_quantity)
    if requested is None or requested <= 0:
        raise ValueError("Spot order quantity must be greater than zero.")
    if step_size is None or step_size <= 0:
        normalized = requested
        step_size = Decimal("0")
    else:
        normalized = (requested / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
    if normalized != requested:
        raise ValueError(
            f"Quantity must align with Binance step size {_format_decimal(step_size)}; "
            f"use {_format_decimal(normalized)} or less."
        )
    if normalized < min_quantity:
        raise ValueError(f"Quantity is below Binance minimum {_format_decimal(min_quantity)}.")
    if max_quantity > 0 and normalized > max_quantity:
        raise ValueError(f"Quantity exceeds Binance maximum {_format_decimal(max_quantity)}.")
    return normalized, step_size


def _validate_notional(symbol_info: dict[str, Any], *, quantity: Decimal, price: float) -> None:
    filters = {
        str(item.get("filterType") or ""): item
        for item in symbol_info.get("filters", [])
        if isinstance(item, dict)
    }
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
    if not isinstance(notional_filter, dict):
        return
    min_notional = _as_decimal(notional_filter.get("minNotional")) or Decimal("0")
    max_notional = _as_decimal(notional_filter.get("maxNotional")) or Decimal("0")
    notional = quantity * Decimal(str(price))
    if min_notional > 0 and notional < min_notional:
        raise ValueError(f"Estimated order value is below Binance minimum ${_format_decimal(min_notional)}.")
    if max_notional > 0 and notional > max_notional:
        raise ValueError(f"Estimated order value exceeds Binance maximum ${_format_decimal(max_notional)}.")


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
    latest_time_ms = snapshot.get("updateTime") or order.get("transactTime")
    for trade in trades:
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
    if weighted_quantity > 0:
        executed_quantity = weighted_quantity
        executed_quote = weighted_quote
    average_price = executed_quote / executed_quantity if executed_quantity > 0 else None
    fee_asset = next(iter(commissions)) if len(commissions) == 1 else None
    fee_amount = commissions.get(fee_asset) if fee_asset is not None else None
    return {
        "status": str(snapshot.get("status") or order.get("status") or "UNKNOWN").upper(),
        "order_id": str(order_id) if order_id is not None else None,
        "client_order_id": str(snapshot.get("clientOrderId") or order.get("clientOrderId") or "") or None,
        "executed_quantity": executed_quantity,
        "executed_quote_quantity": executed_quote,
        "average_price": average_price,
        "fee_amount": fee_amount,
        "fee_asset": fee_asset,
        "commissions": commissions,
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

    def execute(self, intent_id: str) -> dict[str, Any]:
        execution_enabled = str(os.getenv("SCROOGE_SPOT_EXECUTION_ENABLED", "0") or "0").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }
        if not execution_enabled:
            raise RuntimeError("Real Binance Spot execution is disabled by the deployment safety switch.")
        intent = load_spot_order_intent(intent_id, path=self.db_path)
        if intent is None:
            raise LookupError("Spot order intent was not found.")
        if intent["status"] in {"filled", "partially_filled"}:
            return intent["result"] or intent
        if intent["status"] not in {"queueing", "queued", "processing"}:
            raise ValueError(f"Spot order intent cannot execute from status {intent['status']}.")
        update_spot_order_intent(
            intent_id,
            {"status": "processing", "error": None},
            expected_statuses={"queueing", "queued", "processing"},
            path=self.db_path,
        )

        order_may_exist = False
        try:
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
            if holding is None:
                raise ValueError("Treasury asset disappeared before execution.")
            quantity_float = float(quantity)
            if intent["side"] == "sell":
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
                estimated_quote = quantity_float * market_price
                if estimated_quote > quote_free + 0.00000001:
                    raise ValueError("Buy rejected because Binance USDT balance changed after preview.")

            order = self._submit(intent, quantity)
            order_may_exist = True
            update_spot_order_intent(
                intent_id,
                {
                    "status": "submitted",
                    "exchange_order_id": str(order.get("orderId")) if order.get("orderId") is not None else None,
                    "result_json": {"submitted_order": order},
                },
                expected_statuses={"processing"},
                path=self.db_path,
            )
            summary = _execution_summary(self.client, intent["symbol"], order)
            if summary["executed_quantity"] <= 0:
                if summary["status"] not in {"CANCELED", "REJECTED", "EXPIRED"}:
                    raise SpotOrderUncertainError(
                        f"Binance order {summary['order_id'] or intent['client_order_id']} has no confirmed fill "
                        f"and remains in status {summary['status']}. Reconcile it before retrying."
                    )
                order_may_exist = False
                raise RuntimeError(f"Binance order ended with status {summary['status']} and no confirmed fill.")

            update_spot_order_intent(
                intent_id,
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
                expected_statuses={"submitted"},
                path=self.db_path,
            )

            ledger_quantity = summary["executed_quantity"]
            if summary["fee_asset"] == intent["asset_symbol"] and summary["fee_amount"]:
                if intent["side"] == "buy":
                    ledger_quantity -= summary["fee_amount"]
                else:
                    ledger_quantity += summary["fee_amount"]
            ledger_price = summary["average_price"]
            if intent["side"] == "buy" and ledger_quantity > 0 and summary["executed_quote_quantity"] > 0:
                ledger_price = summary["executed_quote_quantity"] / ledger_quantity
            transaction_id = f"spot-order:{intent_id}"
            transaction = {
                "transaction_id": transaction_id,
                "account_key": intent["account_key"],
                "executed_at": datetime.fromtimestamp(summary["executed_at_ms"] / 1000, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "tx_type": intent["side"],
                "asset_symbol": intent["asset_symbol"],
                "quote_symbol": intent["quote_symbol"],
                "quantity": ledger_quantity,
                "price": ledger_price,
                "fee_amount": summary["fee_amount"],
                "fee_asset": summary["fee_asset"],
                "source": "binance_manual",
                "status": "settled",
                "note": f"Manual Binance Spot {intent['side'].upper()} executed from Control Plane.",
                "external_order_id": summary["order_id"],
                "custody_location": "binance",
                "source_custody": None,
                "destination_custody": None,
                "spot_order_intent_id": intent_id,
                "executed_quote_quantity": summary["executed_quote_quantity"],
                "commissions": summary["commissions"],
            }
            try:
                existing = list_portfolio_transactions(account_key=intent["account_key"], path=self.db_path)
                if not any(item.get("transaction_id") == transaction_id for item in existing):
                    append_portfolio_transaction(transaction, path=self.db_path)
            except Exception as accounting_error:  # noqa: BLE001
                self._safe_error_state(
                    intent_id,
                    {
                        "status": "accounting_error",
                        "error": (
                            "Binance fill was confirmed, but Treasury accounting failed: "
                            f"{str(accounting_error)[:350]}"
                        ),
                        "result_json": summary,
                    },
                    expected_statuses={"fill_confirmed", "submitted"},
                )
                raise SpotOrderAccountingError(
                    f"Binance order {summary['order_id']} filled, but Treasury Ledger reconciliation failed. "
                    "Do not retry this order."
                ) from accounting_error

            final_status = "filled" if summary["status"] == "FILLED" else "partially_filled"
            result = {
                **summary,
                "intent_id": intent_id,
                "ledger_transaction_id": transaction_id,
                "requested_quantity": intent["requested_quantity"],
                "side": intent["side"],
                "asset_symbol": intent["asset_symbol"],
                "quote_symbol": intent["quote_symbol"],
            }
            update_spot_order_intent(
                intent_id,
                {
                    "status": final_status,
                    "exchange_order_id": summary["order_id"],
                    "executed_quantity": summary["executed_quantity"],
                    "executed_quote_quantity": summary["executed_quote_quantity"],
                    "average_price": summary["average_price"],
                    "fee_amount": summary["fee_amount"],
                    "fee_asset": summary["fee_asset"],
                    "error": None,
                    "result_json": result,
                },
                expected_statuses={"fill_confirmed", "submitted"},
                path=self.db_path,
            )
            try:
                self._refresh_account()
            except Exception as refresh_error:  # noqa: BLE001
                self.logger.warning("spot_order_post_fill_refresh_failed intent_id=%s error=%s", intent_id, refresh_error)
            self.logger.info(
                "spot_order_filled intent_id=%s order_id=%s side=%s symbol=%s quantity=%s quote=%s status=%s",
                intent_id,
                summary["order_id"],
                intent["side"],
                intent["symbol"],
                summary["executed_quantity"],
                summary["executed_quote_quantity"],
                final_status,
            )
            return result
        except SpotOrderUncertainError as exc:
            self._safe_error_state(
                intent_id,
                {"status": "uncertain", "error": str(exc)[:500]},
                expected_statuses={"processing", "submitted"},
            )
            self.logger.error("spot_order_outcome_uncertain intent_id=%s error=%s", intent_id, exc)
            raise
        except SpotOrderAccountingError as exc:
            self.logger.error("spot_order_accounting_error intent_id=%s error=%s", intent_id, exc)
            raise
        except Exception as exc:
            if order_may_exist:
                uncertain = SpotOrderUncertainError(
                    "Binance accepted the order request, but Scrooge could not complete local reconciliation. "
                    f"Reconcile client order ID {intent['client_order_id']} before retrying."
                )
                self._safe_error_state(
                    intent_id,
                    {"status": "uncertain", "error": str(uncertain)[:500]},
                    expected_statuses={"processing", "submitted", "fill_confirmed"},
                )
                self.logger.error("spot_order_outcome_uncertain intent_id=%s error=%s", intent_id, exc)
                raise uncertain from exc
            self._safe_error_state(
                intent_id,
                {"status": "failed", "error": str(exc)[:500]},
                expected_statuses={"processing", "submitted"},
            )
            self.logger.warning("spot_order_failed intent_id=%s error=%s", intent_id, exc)
            raise
