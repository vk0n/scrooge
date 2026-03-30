import threading
import time
from typing import Any

from binance.helpers import round_step_size
import numpy as np
from bot.event_log import get_technical_logger

_client = None
technical_logger = get_technical_logger()
_account_cache_lock = threading.RLock()
_cached_balance = None
_cached_balance_updated_at = None
_cached_positions = None
_cached_positions_updated_at = None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def set_client(client):
    global _client
    _client = client


def _get_client():
    if _client is None:
        raise ValueError("Binance client not initialized. Call set_client().")
    return _client


def clear_runtime_account_cache():
    global _cached_balance, _cached_balance_updated_at, _cached_positions, _cached_positions_updated_at
    with _account_cache_lock:
        _cached_balance = None
        _cached_balance_updated_at = None
        _cached_positions = None
        _cached_positions_updated_at = None


def set_cached_balance(balance):
    global _cached_balance, _cached_balance_updated_at
    try:
        normalized = float(balance)
    except (TypeError, ValueError):
        return

    with _account_cache_lock:
        _cached_balance = normalized
        _cached_balance_updated_at = time.time()


def get_cached_balance():
    with _account_cache_lock:
        return _cached_balance


def get_cached_balance_age_seconds():
    with _account_cache_lock:
        updated_at = _cached_balance_updated_at
    if updated_at is None:
        return None
    return max(0.0, time.time() - updated_at)


def set_cached_position(symbol, position):
    global _cached_positions, _cached_positions_updated_at

    symbol_key = str(symbol or "").upper().strip()
    if not symbol_key:
        return

    normalized = None
    if isinstance(position, dict):
        normalized = dict(position)
        normalized["symbol"] = symbol_key

    with _account_cache_lock:
        if _cached_positions is None:
            _cached_positions = {}
        if _cached_positions_updated_at is None:
            _cached_positions_updated_at = {}

        if normalized is None:
            _cached_positions.pop(symbol_key, None)
            _cached_positions_updated_at.pop(symbol_key, None)
        else:
            _cached_positions[symbol_key] = normalized
            _cached_positions_updated_at[symbol_key] = time.time()


def get_cached_open_position(symbol):
    symbol_key = str(symbol or "").upper().strip()
    if not symbol_key:
        return None

    with _account_cache_lock:
        if _cached_positions is None:
            return None
        pos = _cached_positions.get(symbol_key)

    if not isinstance(pos, dict):
        return None

    try:
        if float(pos.get("positionAmt", 0.0)) == 0:
            return None
    except (TypeError, ValueError):
        return None
    return dict(pos)


def get_cached_position_age_seconds(symbol):
    symbol_key = str(symbol or "").upper().strip()
    if not symbol_key:
        return None

    with _account_cache_lock:
        if _cached_positions_updated_at is None:
            return None
        updated_at = _cached_positions_updated_at.get(symbol_key)
    if updated_at is None:
        return None
    return max(0.0, time.time() - updated_at)


def get_order_execution_summary(
    symbol: str,
    order_response: Any,
    *,
    timeout_seconds: float = 2.5,
    poll_interval_seconds: float = 0.2,
) -> dict[str, Any] | None:
    order_id = _as_int(order_response.get("orderId")) if isinstance(order_response, dict) else None
    if order_id is None:
        return None

    client = _get_client()
    symbol_key = str(symbol or "").upper().strip()
    if not symbol_key:
        return None

    order_snapshot = dict(order_response) if isinstance(order_response, dict) else {}
    deadline = time.monotonic() + max(0.0, timeout_seconds)

    while time.monotonic() <= deadline:
        try:
            latest_order = client.futures_get_order(symbol=symbol_key, orderId=order_id)
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning(
                "order_execution_summary_order_fetch_failed symbol=%s order_id=%s error=%s",
                symbol_key,
                order_id,
                exc,
            )
            latest_order = None

        if isinstance(latest_order, dict):
            order_snapshot = latest_order

        status = str(order_snapshot.get("status", "")).strip().upper()
        executed_qty = _as_float(order_snapshot.get("executedQty"))
        if status == "FILLED" or (executed_qty is not None and executed_qty > 0):
            break
        time.sleep(max(0.05, poll_interval_seconds))

    trades: list[dict[str, Any]] = []
    while time.monotonic() <= deadline and not trades:
        try:
            raw_trades = client.futures_account_trades(symbol=symbol_key, orderId=order_id)
            if isinstance(raw_trades, list):
                trades = [item for item in raw_trades if isinstance(item, dict)]
        except Exception as exc:  # noqa: BLE001
            technical_logger.warning(
                "order_execution_summary_trades_fetch_failed symbol=%s order_id=%s error=%s",
                symbol_key,
                order_id,
                exc,
            )
            break
        if trades:
            break
        time.sleep(max(0.05, poll_interval_seconds))

    executed_qty = _as_float(order_snapshot.get("executedQty"))
    avg_price = _as_float(order_snapshot.get("avgPrice"))
    quote_qty = _as_float(order_snapshot.get("cumQuote"))
    commission = None
    realized_pnl = None
    update_time_ms = _as_int(order_snapshot.get("updateTime"))

    if trades:
        total_qty = 0.0
        weighted_notional = 0.0
        total_quote_qty = 0.0
        total_commission = 0.0
        total_realized_pnl = 0.0
        saw_commission = False
        saw_realized_pnl = False
        last_trade_time_ms = None

        for item in trades:
            qty = _as_float(item.get("qty"))
            price = _as_float(item.get("price"))
            quote = _as_float(item.get("quoteQty"))
            if qty is not None and qty > 0:
                total_qty += qty
                if price is not None:
                    weighted_notional += price * qty
                if quote is not None:
                    total_quote_qty += quote
                elif price is not None:
                    total_quote_qty += price * qty
            commission_value = _as_float(item.get("commission"))
            if commission_value is not None:
                total_commission += commission_value
                saw_commission = True
            realized_value = _as_float(item.get("realizedPnl"))
            if realized_value is not None:
                total_realized_pnl += realized_value
                saw_realized_pnl = True
            trade_time_ms = _as_int(item.get("time"))
            if trade_time_ms is not None:
                last_trade_time_ms = max(last_trade_time_ms or trade_time_ms, trade_time_ms)

        if total_qty > 0:
            executed_qty = total_qty
            quote_qty = total_quote_qty if total_quote_qty > 0 else quote_qty
            avg_price = (weighted_notional / total_qty) if weighted_notional > 0 else avg_price
        if saw_commission:
            commission = total_commission
        if saw_realized_pnl:
            realized_pnl = total_realized_pnl
        if last_trade_time_ms is not None:
            update_time_ms = last_trade_time_ms

    if avg_price is None and executed_qty is not None and executed_qty > 0 and quote_qty is not None:
        avg_price = quote_qty / executed_qty

    status = str(order_snapshot.get("status", "")).strip().upper() or None
    if executed_qty is None and avg_price is None and commission is None and realized_pnl is None and status is None:
        return None

    return {
        "order_id": order_id,
        "status": status,
        "executed_qty": executed_qty,
        "avg_price": avg_price,
        "quote_qty": quote_qty,
        "commission": commission,
        "realized_pnl": realized_pnl,
        "update_time_ms": update_time_ms,
        "raw_order": order_snapshot,
        "raw_trades": trades,
    }

# --- Utility functions --- #
def set_leverage(symbol, leverage):
    """Set leverage for a given symbol"""
    try:
        client = _get_client()
        resp = client.futures_change_leverage(
            symbol=symbol,
            leverage=leverage
        )
        technical_logger.info("leverage_set symbol=%s leverage=%s response=%s", symbol, leverage, resp)
    except Exception as e:
        technical_logger.exception("leverage_set_failed symbol=%s leverage=%s error=%s", symbol, leverage, e)


def get_symbol_info(symbol):
    """Get precision and stepSize for symbol"""
    client = _get_client()
    info = client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            step_size = float(s["filters"][2]["stepSize"])
            tick_size = float(s["filters"][0]["tickSize"])
            qty_precision = int(round(-np.log10(step_size),0))
            price_precision = int(round(-np.log10(tick_size),0))
            return qty_precision, price_precision
    return 3, 2  # default fallback


def round_qty_price(symbol, qty, price):
    qty_precision, price_precision = 3, 2 #get_symbol_info(symbol)
    qty = round(qty, qty_precision)
    price = round(price, price_precision)
    return qty, price


def adjust_qty_to_margin(symbol, qty, price, leverage, buffer=0.95, client=None):
    """
    Adjusts position size to fit available margin with a safety buffer.
    """
    if client is None:
        client = _get_client()
    # Get futures account info
    account_info = client.futures_account()
    available_balance = float(account_info["availableBalance"])  

    # Maximum notional value allowed with available balance
    max_notional = available_balance * leverage * buffer  

    # Max quantity we can afford
    max_qty = max_notional / price  

    # Final adjusted quantity
    final_qty = min(qty, max_qty)

    # Binance requires step size rounding
    exchange_info = client.futures_exchange_info()
    symbol_info = next(s for s in exchange_info["symbols"] if s["symbol"] == symbol)
    step_size = None
    for f in symbol_info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
            break
    if step_size:
        final_qty = round_step_size(final_qty, step_size)

    return final_qty


def compute_qty(symbol, balance, leverage, price, qty=None, use_full_balance=True, live=False):
    """
    Compute the actual position size (qty) for opening a trade.
    - If qty is given explicitly, use it.
    - Otherwise, if use_full_balance, compute maximum qty based on balance, leverage, and price.
    - Round qty and adjust to margin.
    """
    if qty is not None:
        qty_local = qty
    elif use_full_balance:
        qty_local = (balance * leverage) / price
    else:
        qty_local = 0

    # Round to allowed precision for symbol
    qty_local, _ = round_qty_price(symbol, qty_local, price)

    # Adjust to available margin in live mode
    if live:
        qty_local = adjust_qty_to_margin(symbol, qty_local, price, leverage)
    else:
        qty_local *= 0.95

    return qty_local


def check_balance():
    client = _get_client()
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            technical_logger.debug("balance_checked asset=USDT balance=%s", b["balance"])
    return balances


def get_balance(*, prefer_cached=False):
    """
    Fetch USDT balance from Binance Futures account.
    Returns float balance in USDT.
    """
    if prefer_cached:
        cached = get_cached_balance()
        if cached is not None:
            return cached

    client = _get_client()
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            return float(b["balance"])
    raise ValueError("USDT balance not found")


def can_open_trade(symbol, qty, leverage=10, live=False):
    """Check if there is enough balance to open a trade"""
    usdt_balance = get_balance()
    client = _get_client()
    price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
    required_margin = (price * qty) / leverage
    return usdt_balance >= required_margin and qty > 0


def get_open_position(symbol, *, prefer_cached=False):
    """Return open position info, or None if no position"""
    if prefer_cached:
        cached = get_cached_open_position(symbol)
        if cached is not None:
            return cached

    client = _get_client()
    positions = client.futures_position_information(symbol=symbol)
    if len(positions) > 0:
        pos = positions[0]
        if float(pos["positionAmt"]) != 0:
            return pos
    return None


def cancel_sl_tp(symbol):
    """Cancel all SL/TP orders for this symbol"""
    client = _get_client()
    open_orders = client.futures_get_open_orders(symbol=symbol)
    for o in open_orders:
        if o["type"] in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
            technical_logger.info("protective_order_cancelled symbol=%s type=%s order_id=%s", symbol, o["type"], o["orderId"])


def _submit_market_close_order(symbol):
    pos = get_open_position(symbol)
    if not pos:
        technical_logger.warning(
            "position_close_skipped_no_exchange_position symbol=%s",
            symbol,
        )
        return None

    pos_amt = float(pos["positionAmt"])
    close_side = "SELL" if pos_amt > 0 else "BUY"
    close_qty = abs(pos_amt)
    close_qty, _ = round_qty_price(symbol, close_qty, 0)

    try:
        client = _get_client()
        order = client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=close_qty
        )
        technical_logger.info(
            "position_closed symbol=%s side=%s quantity=%s",
            symbol,
            close_side,
            close_qty,
        )
        cancel_sl_tp(symbol)
        return order
    except Exception as e:
        technical_logger.exception("position_close_failed symbol=%s error=%s", symbol, e)
        return None


def _submit_market_open_order(symbol, side, qty, sl=None, tp=None, leverage=10):
    if qty <= 0:
        technical_logger.warning("position_open_skipped_not_enough_margin symbol=%s side=%s", symbol, side)
        return None

    existing_position = get_open_position(symbol)
    if existing_position is not None:
        technical_logger.warning(
            "position_open_blocked_existing_exchange_position symbol=%s side=%s quantity=%s existing_position_amt=%s",
            symbol,
            side,
            qty,
            existing_position.get("positionAmt"),
        )
        return None

    qty, sl = round_qty_price(symbol, qty, sl) if sl else (qty, None)
    qty, tp = round_qty_price(symbol, qty, tp) if tp else (qty, None)

    try:
        client = _get_client()
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty
        )
        technical_logger.info(
            "position_opened symbol=%s side=%s quantity=%s leverage=%s",
            symbol,
            side,
            qty,
            leverage,
        )

        if sl:
            technical_logger.info("stop_loss_prepared symbol=%s side=%s stop_price=%s", symbol, side, sl)

        if tp:
            technical_logger.info("take_profit_prepared symbol=%s side=%s take_profit=%s", symbol, side, tp)

        return order
    except Exception as e:
        technical_logger.exception("position_open_failed symbol=%s side=%s quantity=%s error=%s", symbol, side, qty, e)
        return None


def open_position(symbol, side, qty, sl=None, tp=None, leverage=10):
    """
    Open a new trade only when the exchange is flat.
    """
    return _submit_market_open_order(symbol, side, qty, sl, tp, leverage)


def close_position(symbol):
    """
    Close an existing exchange position.
    """
    return _submit_market_close_order(symbol)
