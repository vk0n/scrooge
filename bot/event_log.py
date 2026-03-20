from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from core.event_store import build_event_record, get_event_store

try:
    from api.services.push_service import dispatch_event_push
except ImportError:  # pragma: no cover - api package may be unavailable in some contexts
    dispatch_event_push = None


UI_LOG_PATH = Path(os.getenv("SCROOGE_LOG_FILE", "runtime/trading_log.txt")).expanduser()
TECHNICAL_LOGGER_NAME = "scrooge.bot"
KNOWN_QUOTE_ASSETS = (
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "BTC",
    "ETH",
    "EUR",
    "TRY",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _resolve_ui_log_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser()
    return Path(os.getenv("SCROOGE_LOG_FILE", str(UI_LOG_PATH))).expanduser()


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric == numeric:
        return None
    return numeric


def _base_asset(symbol: Any) -> str | None:
    raw_symbol = str(symbol or "").strip().upper()
    if not raw_symbol:
        return None
    for quote_asset in KNOWN_QUOTE_ASSETS:
        if raw_symbol.endswith(quote_asset) and len(raw_symbol) > len(quote_asset):
            return raw_symbol[: -len(quote_asset)]
    return raw_symbol


def _format_price(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


def _format_rsi(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


def _format_signed_currency(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    if abs(numeric) < 1e-12:
        return "$0.00"
    if numeric > 0:
        return f"+${numeric:.2f}"
    return f"-${abs(numeric):.2f}"


def _format_signed_pct(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.2f}%"


def _format_size(value: Any, symbol: Any = None) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    asset = _base_asset(symbol)
    if asset:
        return f"{numeric:.4f} {asset}"
    return f"{numeric:.4f}"


def _format_leverage(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"x{numeric:.2f}"


def _pretty_side(side: Any) -> str:
    normalized = str(side or "").strip().lower()
    if normalized in {"buy", "long"}:
        return "long"
    if normalized in {"sell", "short"}:
        return "short"
    return normalized or "trade"


def _format_result(pnl: Any, roi_pct: Any) -> str:
    pnl_label = _format_signed_currency(pnl)
    roi_label = _format_signed_pct(roi_pct)
    if pnl_label == "n/a" and roi_label == "n/a":
        return "Result: n/a."
    if roi_label == "n/a":
        return f"Result: {pnl_label}."
    if pnl_label == "n/a":
        return f"Result: {roi_label} ROI."
    return f"Result: {pnl_label} ({roi_label} ROI)."


def _format_currency(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"${abs(numeric):.2f}"


def _pretty_action(action: Any) -> str:
    normalized = str(action or "").strip().lower()
    mapping = {
        "update_sl": "the Safety Net update",
        "update_tp": "the Treasure Mark update",
        "close_position": "the close-position order",
        "suggest_trade": "the trade suggestion",
    }
    if normalized in mapping:
        return mapping[normalized]
    if normalized:
        return normalized.replace("_", " ")
    return "that order"


def _pretty_level_name(level_type: Any) -> str:
    normalized = str(level_type or "").strip().lower()
    if normalized == "sl":
        return "Safety Net"
    if normalized == "tp":
        return "Treasure Mark"
    if normalized == "trail":
        return "Tail Guard"
    return normalized or "level"


def _trigger_phrase(trigger: Any) -> str:
    normalized = str(trigger or "").strip().lower()
    if normalized == "manual_suggestion":
        return "This came from your suggestion rather than my usual timing signal."
    if normalized == "strategy_rules":
        return "This follows my usual timing rules."
    return ""


def _half_stake_phrase(stake_mode: Any, rsi: Any) -> str:
    if str(stake_mode or "").strip().lower() != "half":
        return ""
    rsi_label = _format_rsi(rsi)
    if rsi_label == "n/a":
        return " I cut the stake in half to stay prudent."
    return f" I cut the stake in half because RSI stood at {rsi_label}."


def _render_ui_message(code: str, context: dict[str, Any]) -> str:
    side = _pretty_side(context.get("side"))
    symbol = str(context.get("symbol") or "").strip().upper()
    trigger_phrase = _trigger_phrase(context.get("trigger"))

    if code == "bot_started":
        if symbol:
            return f"I have opened the office and resumed watching {symbol}."
        return "I have opened the office and resumed trading."

    if code == "bot_stopped":
        return "I have closed the office and suspended trading for now."

    if code == "bot_restarted":
        if symbol:
            return f"I have reopened the office and refreshed my contract for {symbol}."
        return "I have reopened the office and refreshed my contract."

    if code == "manual_trade_suggested":
        action = "buy" if side == "long" else "sell"
        if symbol:
            return (
                f"You have asked me to {action} {symbol} at the next live tick, "
                "and I shall manage the trade by the usual rules."
            )
        return f"You have asked me to {action} at the next live tick, and I shall manage it by the usual rules."

    if code == "trade_opened":
        entry = _format_price(context.get("entry"))
        sl = _format_price(context.get("sl"))
        tp = _format_price(context.get("tp"))
        size = _format_size(context.get("size"), symbol)
        leverage = _format_leverage(context.get("leverage"))
        fee = _format_currency(context.get("fee"))
        message = (
            f"I have opened a {side} on {symbol} at {entry}, with the Safety Net at {sl} "
            f"and the Treasure Mark at {tp}. Position size: {size}, leverage: {leverage}, opening fee: {fee}."
        )
        message += _half_stake_phrase(context.get("stake_mode"), context.get("rsi"))
        if trigger_phrase:
            message += f" {trigger_phrase}"
        return message

    if code == "entry_skipped_liquidation_guard":
        entry = _format_price(context.get("entry"))
        sl = _format_price(context.get("sl"))
        liq_price = _format_price(context.get("liq_price"))
        message = (
            f"I refused to open a {side} on {symbol} at {entry}. "
            f"The Safety Net at {sl} would cross the liquidation line at {liq_price}."
        )
        if trigger_phrase:
            message += f" {trigger_phrase}"
        return message

    if code == "trail_activated":
        market_price = _format_price(context.get("market_price"))
        trail_price = _format_price(context.get("trail_price"))
        base_tp = _format_price(context.get("base_tp"))
        return (
            f"I have armed Tail Guard for the {side} on {symbol}. "
            f"Price reached {market_price} beyond the Treasure Mark at {base_tp}, "
            f"so I am now guarding profit at {trail_price}."
        )

    if code == "trail_moved":
        previous_trail = _format_price(context.get("previous_trail"))
        trail_price = _format_price(context.get("trail_price"))
        anchor_price = _format_price(context.get("anchor_price"))
        return (
            f"I have moved Tail Guard for the {side} on {symbol} from {previous_trail} to {trail_price} "
            f"as the market pushed further in our favor to {anchor_price}."
        )

    if code == "trade_closed_take_profit":
        exit_price = _format_price(context.get("exit"))
        result = _format_result(context.get("net_pnl"), context.get("roi_pct"))
        if context.get("via_tail_guard"):
            return f"I have closed the {side} on {symbol} at {exit_price} under Tail Guard. {result}"
        return f"I have closed the {side} on {symbol} at {exit_price} by the Treasure Mark. {result}"

    if code == "trade_closed_stop_loss":
        exit_price = _format_price(context.get("exit"))
        result = _format_result(context.get("net_pnl"), context.get("roi_pct"))
        return f"I have closed the {side} on {symbol} at {exit_price} by the Safety Net. {result}"

    if code == "trade_closed_rsi_extreme":
        exit_price = _format_price(context.get("exit"))
        rsi = _format_rsi(context.get("rsi"))
        threshold = _format_rsi(context.get("threshold"))
        result = _format_result(context.get("net_pnl"), context.get("roi_pct"))
        return (
            f"I have closed the {side} on {symbol} at {exit_price} because RSI reached {rsi} "
            f"beyond my extreme bar of {threshold}. {result}"
        )

    if code == "trade_closed_manual":
        exit_price = _format_price(context.get("exit"))
        result = _format_result(context.get("net_pnl"), context.get("roi_pct"))
        return f"I have closed the {side} on {symbol} at your request near {exit_price}. {result}"

    if code == "trade_liquidated":
        liq_price = _format_price(context.get("liq_price"))
        pnl = _format_signed_currency(context.get("net_pnl"))
        return f"The {side} on {symbol} has been liquidated near {liq_price}. Result: {pnl}."

    if code == "level_updated":
        level_name = _pretty_level_name(context.get("level_type"))
        old_value = _format_price(context.get("old_value"))
        new_value = _format_price(context.get("new_value"))
        if symbol:
            return f"I have moved the {level_name} on {symbol} from {old_value} to {new_value}."
        return f"I have moved the {level_name} from {old_value} to {new_value}."

    if code == "command_failed":
        action = _pretty_action(context.get("action"))
        reason = str(context.get("reason") or "unknown trouble").strip().rstrip(".")
        return f"I could not carry out {action}: {reason}."

    fallback_message = str(context.get("message") or "").strip()
    if fallback_message:
        return fallback_message

    normalized_code = code.replace("_", " ").strip()
    return normalized_code.capitalize() + "."


def _ensure_technical_logger() -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("websocket").setLevel(logging.WARNING)
    return logging.getLogger(TECHNICAL_LOGGER_NAME)


def get_technical_logger() -> logging.Logger:
    return _ensure_technical_logger()


def _should_emit_event_to_stdout(event: dict[str, Any]) -> bool:
    runtime_mode = str(event.get("runtime_mode") or "").strip().lower()
    if runtime_mode == "backtest":
        return _env_flag("SCROOGE_BACKTEST_EVENT_STDOUT_ENABLED", False)
    return _env_flag("SCROOGE_EVENT_STDOUT_ENABLED", True)


def append_ui_log_line(ts: str, message: str, *, log_buffer: list[str] | None = None, ui_log_path: Path | None = None) -> None:
    line = f"[{ts}] {message}"
    if log_buffer is not None:
        log_buffer.append(line)
        return

    path = _resolve_ui_log_path(ui_log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(line + "\n")
    except OSError:
        _ensure_technical_logger().exception("Failed to append UI log line")


def _should_persist_ui_immediately(runtime_mode: str | None, persist_ui: bool) -> bool:
    if persist_ui:
        return True
    normalized_runtime_mode = str(runtime_mode or "").strip().lower()
    return normalized_runtime_mode == "live"


def emit_event(
    *,
    code: str,
    category: str,
    ts: str,
    level: str = "info",
    notify: bool = False,
    log_buffer: list[str] | None = None,
    persist_ui: bool = False,
    ui_log_path: Path | None = None,
    ui_message: str | None = None,
    runtime_mode: str | None = None,
    strategy_mode: str | None = None,
    **context: Any,
) -> dict[str, Any]:
    event_context = dict(context)
    rendered_ui_message = (ui_message or "").strip() or _render_ui_message(code, event_context)
    persist_ui_immediately = _should_persist_ui_immediately(runtime_mode, persist_ui)
    ui_log_buffer = None if persist_ui_immediately else log_buffer

    event = build_event_record(
        ts=ts,
        level=level,
        code=code,
        category=category,
        ui_message=rendered_ui_message,
        notify=bool(notify),
        context=event_context,
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
    )

    if ui_log_buffer is not None or persist_ui_immediately:
        append_ui_log_line(
            ts,
            rendered_ui_message,
            log_buffer=ui_log_buffer,
            ui_log_path=ui_log_path,
        )

    logger = _ensure_technical_logger()
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    try:
        get_event_store().append(event)
    except OSError:
        logger.exception("Failed to append canonical event record")
    if _should_emit_event_to_stdout(event):
        logger.log(log_level, json.dumps(event, ensure_ascii=True, sort_keys=True, default=str))
    if dispatch_event_push is not None:
        dispatch_event_push(event)
    return event
