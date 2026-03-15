import threading
import time
from typing import Any

from binance.helpers import round_step_size
import numpy as np
from event_log import get_technical_logger

_client = None
technical_logger = get_technical_logger()
_account_cache_lock = threading.RLock()
_cached_balance = None
_cached_balance_updated_at = None
_cached_positions = None
_cached_positions_updated_at = None

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


def open_or_close_trade(symbol, side=None, qty=0, sl=None, tp=None, leverage=10):
    """
    Open trade if no position, close if position exists.
    Handles SL/TP and checks balance.
    """
    pos = get_open_position(symbol)

    if pos:
        # --- Close existing position ---
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

    elif side and qty > 0:
        # --- Open new position ---
        if qty <= 0:
            technical_logger.warning("position_open_skipped_not_enough_margin symbol=%s side=%s", symbol, side)
            return

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

            # place SL/TP
            if sl:
                #client.futures_create_order(
                #    symbol=symbol,
                #    side="SELL" if side=="BUY" else "BUY",
                #    type="STOP_MARKET",
                #    stopPrice=sl,
                #    quantity=qty
                #)
                technical_logger.info("stop_loss_prepared symbol=%s side=%s stop_price=%s", symbol, side, sl)

            if tp:
                # client.futures_create_order(
                #     symbol=symbol,
                #     side="SELL" if side=="BUY" else "BUY",
                #     type="TAKE_PROFIT_MARKET",
                #     stopPrice=tp,
                #     quantity=qty
                # )
                technical_logger.info("take_profit_prepared symbol=%s side=%s take_profit=%s", symbol, side, tp)

            return order

        except Exception as e:
            technical_logger.exception("position_open_failed symbol=%s side=%s quantity=%s error=%s", symbol, side, qty, e)


def open_position(symbol, side, qty, sl=None, tp=None, leverage=10):
    """
    Wrapper to open a new trade.
    Uses the universal open_or_close_trade function.
    """
    open_or_close_trade(symbol, side, qty, sl, tp, leverage)


def close_position(symbol):
    """
    Wrapper to close any open position.
    Uses the universal open_or_close_trade function.
    """
    # side and qty are ignored when closing, universal function detects open position
    return open_or_close_trade(symbol)
