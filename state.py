# state.py
import json
import os
from datetime import datetime

STATE_FILE = "state.json"

def load_state():
    """
    Load the bot's state from JSON file.
    Returns a dict with 'balance', 'position', and 'trade_history'.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # default state if file does not exist
    return {
        "position": None,
        "balance": None,
        "trade_history": [],
        "balance_history": [],
        "session_start": int(datetime.now().timestamp() * 1000),  # milliseconds
        "session_end": None
    }

def save_state(state):
    """
    Save the bot's state to JSON file.
    """
    # Update session_end every save
    state["session_end"] = int(datetime.now().timestamp() * 1000)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4, default=str)

def add_closed_trade(state, trade):
    """
    Append a closed trade to trade_history.
    trade: dict containing trade info (side, entry, exit, pnl, fee, exit_reason, time)
    """
    state.setdefault("trade_history", []).append(trade)
    save_state(state)

def update_position(state, position):
    """
    Update the current open position in state.
    position: dict or None if no open position
    """
    state["position"] = position
    save_state(state)

def update_balance(state, balance):
    """
    Update the current balance in state.
    """
    state["balance"] = balance
    save_state(state)
