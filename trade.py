import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from binance.helpers import round_step_size
import numpy as np

# === Load API keys from environment ===
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Connect to Binance Futures
client = Client(API_KEY, API_SECRET)

# --- Utility functions --- #
def set_leverage(symbol, leverage):
    """Set leverage for a given symbol"""
    try:
        resp = client.futures_change_leverage(
            symbol=symbol,
            leverage=leverage
        )
        print(f"Leverage set: {resp}")
    except Exception as e:
        print("Error setting leverage:", e)


def get_symbol_info(symbol):
    """Get precision and stepSize for symbol"""
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
    qty_precision, price_precision = get_symbol_info(symbol)
    qty = round(qty, qty_precision)
    price = round(price, price_precision)
    return qty, price


def adjust_qty_to_margin(client, symbol, qty, price, leverage, buffer=0.95):
    """
    Adjusts position size to fit available margin with a safety buffer.
    """
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
        qty_local = adjust_qty_to_margin(client, symbol, qty_local, price, leverage)

    return qty_local


def check_balance():
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            print(f"Balance: {b['balance']} USDT")
    return balances


def get_balance():
    """
    Fetch USDT balance from Binance Futures account.
    Returns float balance in USDT.
    """
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            return float(b["balance"])
    raise ValueError("USDT balance not found")


def can_open_trade(symbol, qty, leverage=10):
    """Check if there is enough balance to open a trade"""
    balances = client.futures_account_balance()
    usdt_balance = next(float(b["balance"]) for b in balances if b["asset"]=="USDT")
    price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
    required_margin = (price * qty) / leverage
    return usdt_balance >= required_margin and qty > 0


def get_open_position(symbol):
    """Return open position info, or None if no position"""
    positions = client.futures_position_information(symbol=symbol)
    if len(positions) > 0:
        pos = positions[0]
        if float(pos["positionAmt"]) != 0:
            return pos
    return None


def cancel_sl_tp(symbol):
    """Cancel all SL/TP orders for this symbol"""
    open_orders = client.futures_get_open_orders(symbol=symbol)
    for o in open_orders:
        if o["type"] in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
            print(f"[CANCEL] Removed {o['type']} order (id={o['orderId']})")


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
            order = client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type="MARKET",
                quantity=close_qty
            )
            print(f"✅ Closed position: {close_side} {close_qty} {symbol}")
            cancel_sl_tp(symbol)
            return order
        except Exception as e:
            print("❌ Error closing position:", e)

    elif side and qty > 0:
        # --- Open new position ---
        if qty <= 0:
            print("❌ Not enough margin to open any position")
            return

        qty, sl = round_qty_price(symbol, qty, sl) if sl else (qty, None)
        qty, tp = round_qty_price(symbol, qty, tp) if tp else (qty, None)

        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty
            )
            print(f"✅ Opened position: {side} {qty} {symbol}")

            # place SL/TP
            if sl:
                client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side=="BUY" else "BUY",
                    type="STOP_MARKET",
                    stopPrice=sl,
                    quantity=qty
                )
                print(f"[SL] Stop-loss set at {sl}")

            if tp:
                client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side=="BUY" else "BUY",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=tp,
                    quantity=qty
                )
                print(f"[TP] Take-profit set at {tp}")

            return order

        except Exception as e:
            print("❌ Error opening position:", e)


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
    open_or_close_trade(symbol)