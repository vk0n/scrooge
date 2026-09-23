from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from shared.spot_swing import (
    SWING_OBJECTIVES,
    SWING_SIDES,
    SWING_SOURCES,
    SWING_STATUSES,
    calculate_swing_economics,
    calculate_target_ratchet,
)
from shared.spot_strategy import transition_spot_strategy_campaign

DEFAULT_DB_FILENAME = "scrooge.sqlite3"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
STATE_SNAPSHOT_KEY = "current"
RUNTIME_DB_SCHEMA_VERSION = 13
RUNTIME_DB_SCHEMA_DESCRIPTION = "Progressive Spot Swing execution state"


class RuntimeDbError(OSError):
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_artifact_dir() -> Path:
    raw_db_path = (os.getenv("SCROOGE_DB_PATH", "") or "").strip()
    if raw_db_path:
        return Path(raw_db_path).expanduser().parent

    raw_runtime_dir = (os.getenv("SCROOGE_RUNTIME_DIR", "") or "").strip()
    if raw_runtime_dir:
        return Path(raw_runtime_dir).expanduser()

    return _project_root() / "runtime"


def runtime_db_path() -> Path:
    configured = (os.getenv("SCROOGE_DB_PATH", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_artifact_dir() / DEFAULT_DB_FILENAME


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _row_key(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _parse_timestamp_to_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def _trade_sort_ts_ms(trade: dict[str, Any]) -> int | None:
    for field_name in ("exit_time", "entry_time", "time"):
        parsed = _parse_timestamp_to_ms(trade.get(field_name))
        if parsed is not None:
            return parsed
    return None


def _as_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _trade_record(trade: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _row_key(trade),
        _trade_sort_ts_ms(trade),
        trade.get("exit_time"),
        trade.get("entry_time"),
        trade.get("time"),
        str(trade.get("side")) if trade.get("side") is not None else None,
        _as_float_or_none(trade.get("entry")),
        _as_float_or_none(trade.get("exit")),
        _as_float_or_none(trade.get("size")),
        _as_float_or_none(trade.get("net_pnl")),
        _as_float_or_none(trade.get("fee")),
        str(trade.get("exit_reason")) if trade.get("exit_reason") is not None else None,
        str(trade.get("trigger")) if trade.get("trigger") is not None else None,
        1 if bool(trade.get("trail_active")) else 0,
        _json_text(trade),
        int(datetime.now(timezone.utc).timestamp() * 1000),
    )


def _balance_payload(balance: float | dict[str, Any], *, snapshot_index: int | None = None) -> dict[str, Any]:
    if isinstance(balance, dict):
        payload = dict(balance)
    else:
        payload = {"time": None, "balance": balance}
    if snapshot_index is not None:
        payload["_snapshot_index"] = snapshot_index
    return payload


def _balance_record(balance: float | dict[str, Any], *, snapshot_index: int | None = None) -> tuple[Any, ...]:
    payload = _balance_payload(balance, snapshot_index=snapshot_index)
    sort_ts_ms = None
    time_value = payload.get("time")
    if isinstance(time_value, (int, float)):
        sort_ts_ms = int(time_value)
    elif isinstance(time_value, str):
        sort_ts_ms = _parse_timestamp_to_ms(time_value)

    return (
        _row_key(payload),
        sort_ts_ms,
        _as_float_or_none(payload.get("balance")),
        _json_text(payload),
        int(datetime.now(timezone.utc).timestamp() * 1000),
    )


def _event_sort_ts_ms(event: dict[str, Any]) -> int | None:
    return _parse_timestamp_to_ms(event.get("ts"))


def _event_record(event: dict[str, Any]) -> tuple[Any, ...]:
    context = event.get("context")
    return (
        str(event.get("event_id") or _row_key(event)),
        _event_sort_ts_ms(event),
        str(event.get("ts") or "").strip() or None,
        str(event.get("level") or "").strip() or None,
        str(event.get("code") or "").strip() or None,
        str(event.get("category") or "").strip() or None,
        1 if bool(event.get("notify")) else 0,
        str(event.get("runtime_mode") or "").strip() or None,
        str(event.get("strategy_mode") or "").strip() or None,
        str(event.get("ui_message") or "").strip() or None,
        _json_text(context if isinstance(context, dict) else {}),
        _json_text(event),
        int(datetime.now(timezone.utc).timestamp() * 1000),
    )


def _ui_log_record(ts: str, line: str) -> tuple[Any, ...]:
    return (
        _parse_timestamp_to_ms(ts),
        str(ts or "").strip() or None,
        str(line or "").rstrip("\n"),
        int(datetime.now(timezone.utc).timestamp() * 1000),
    )


def _portfolio_transaction_record(transaction: dict[str, Any]) -> tuple[Any, ...]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    payload = dict(transaction)
    executed_at_ms = _parse_timestamp_to_ms(payload.get("executed_at"))
    if executed_at_ms is None:
        executed_at_ms = now_ms
    executed_at_text = str(payload.get("executed_at") or "").strip() or datetime.fromtimestamp(
        executed_at_ms / 1000,
        tz=timezone.utc,
    ).strftime(TIMESTAMP_FORMAT)
    transaction_id = str(payload.get("transaction_id") or _row_key(payload))
    account_key = str(payload.get("account_key") or "manual_spot").strip() or "manual_spot"

    return (
        transaction_id,
        account_key,
        executed_at_ms,
        executed_at_text,
        str(payload.get("tx_type") or "").strip().lower(),
        str(payload.get("asset_symbol") or "").strip().upper(),
        str(payload.get("quote_symbol") or "USDT").strip().upper(),
        _as_float_or_none(payload.get("quantity")),
        _as_float_or_none(payload.get("price")),
        _as_float_or_none(payload.get("fee_amount")),
        str(payload.get("fee_asset") or "").strip().upper() or None,
        str(payload.get("source") or "manual").strip().lower(),
        str(payload.get("status") or "settled").strip().lower(),
        str(payload.get("note") or "").strip() or None,
        str(payload.get("external_order_id") or "").strip() or None,
        str(payload.get("custody_location") or "unassigned").strip().lower() or "unassigned",
        str(payload.get("source_custody") or "").strip().lower() or None,
        str(payload.get("destination_custody") or "").strip().lower() or None,
        _json_text(payload),
        now_ms,
        now_ms,
    )


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    applied_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_key TEXT NOT NULL UNIQUE,
            sort_ts_ms INTEGER,
            exit_time TEXT,
            entry_time TEXT,
            trade_time TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            size REAL,
            net_pnl REAL,
            fee REAL,
            exit_reason TEXT,
            trigger TEXT,
            trail_active INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trade_history_sort
        ON trade_history(sort_ts_ms DESC, id DESC);

        CREATE TABLE IF NOT EXISTS balance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_key TEXT NOT NULL UNIQUE,
            sort_ts_ms INTEGER,
            balance REAL NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_balance_history_sort
        ON balance_history(sort_ts_ms ASC, id ASC);

        CREATE TABLE IF NOT EXISTS runtime_state_snapshot (
            state_key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            sort_ts_ms INTEGER,
            ts_text TEXT,
            level TEXT,
            code TEXT,
            category TEXT,
            notify INTEGER NOT NULL DEFAULT 0,
            runtime_mode TEXT,
            strategy_mode TEXT,
            ui_message TEXT,
            context_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_event_history_sort
        ON event_history(sort_ts_ms DESC, id DESC);

        CREATE TABLE IF NOT EXISTS ui_log_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT UNIQUE,
            sort_ts_ms INTEGER,
            ts_text TEXT,
            line_text TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'trades',
            code TEXT,
            tone TEXT NOT NULL DEFAULT 'neutral',
            message_text TEXT,
            source_ref TEXT,
            context_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ui_log_entries_sort
        ON ui_log_entries(sort_ts_ms DESC, id DESC);

        CREATE TABLE IF NOT EXISTS strategy_chart_snapshots (
            symbol TEXT NOT NULL,
            kind TEXT NOT NULL,
            bucket_ms INTEGER NOT NULL,
            ts_ms INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (symbol, kind, bucket_ms)
        );
        CREATE INDEX IF NOT EXISTS idx_strategy_chart_time
        ON strategy_chart_snapshots(symbol, ts_ms);

        CREATE TABLE IF NOT EXISTS portfolio_accounts (
            account_key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            account_type TEXT NOT NULL DEFAULT 'spot',
            base_currency TEXT NOT NULL DEFAULT 'USDT',
            archived INTEGER NOT NULL DEFAULT 0,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            account_key TEXT NOT NULL,
            executed_at_ms INTEGER NOT NULL,
            executed_at_text TEXT NOT NULL,
            tx_type TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            quantity REAL NOT NULL,
            price REAL,
            fee_amount REAL,
            fee_asset TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            status TEXT NOT NULL DEFAULT 'settled',
            note TEXT,
            external_order_id TEXT,
            custody_location TEXT NOT NULL DEFAULT 'unassigned',
            source_custody TEXT,
            destination_custody TEXT,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_sort
        ON portfolio_transactions(executed_at_ms DESC, id DESC);

        CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_asset
        ON portfolio_transactions(asset_symbol, quote_symbol, status);

        CREATE TABLE IF NOT EXISTS portfolio_asset_policies (
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            target_quantity REAL NOT NULL,
            minimum_holding_pct REAL NOT NULL DEFAULT 100,
            trading_objective TEXT NOT NULL DEFAULT 'accumulate_cash' CHECK (
                trading_objective IN ('accumulate_cash', 'accumulate_asset')
            ),
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (account_key, asset_symbol, quote_symbol),
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE TABLE IF NOT EXISTS spot_swings (
            swing_id TEXT PRIMARY KEY,
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            origin_side TEXT NOT NULL CHECK (origin_side IN ('buy', 'sell')),
            trading_objective TEXT CHECK (
                trading_objective IS NULL OR trading_objective IN ('accumulate_cash', 'accumulate_asset')
            ),
            status TEXT NOT NULL DEFAULT 'open' CHECK (
                status IN ('open', 'partially_closed', 'accepting_loss', 'closed')
            ),
            planned_quantity REAL,
            reference_state_json TEXT NOT NULL DEFAULT '{}',
            strategy_reason_json TEXT NOT NULL DEFAULT '{}',
            source TEXT NOT NULL DEFAULT 'strategy' CHECK (source IN ('manual', 'strategy')),
            close_reason TEXT,
            opened_at_ms INTEGER NOT NULL,
            closed_at_ms INTEGER,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_swings_asset_status
        ON spot_swings(account_key, asset_symbol, quote_symbol, status, opened_at_ms DESC);

        CREATE TABLE IF NOT EXISTS exchange_account_snapshots (
            venue TEXT NOT NULL,
            account_type TEXT NOT NULL,
            status TEXT NOT NULL,
            captured_at_ms INTEGER,
            last_attempt_at_ms INTEGER NOT NULL,
            can_trade INTEGER,
            error TEXT,
            payload_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (venue, account_type)
        );

        CREATE TABLE IF NOT EXISTS exchange_asset_balances (
            venue TEXT NOT NULL,
            account_type TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            free REAL NOT NULL,
            locked REAL NOT NULL,
            total REAL NOT NULL,
            captured_at_ms INTEGER NOT NULL,
            PRIMARY KEY (venue, account_type, asset_symbol),
            FOREIGN KEY(venue, account_type)
                REFERENCES exchange_account_snapshots(venue, account_type)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS spot_order_intents (
            intent_id TEXT PRIMARY KEY,
            account_key TEXT NOT NULL,
            venue TEXT NOT NULL DEFAULT 'binance',
            symbol TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            requested_quantity REAL NOT NULL,
            estimated_price REAL NOT NULL,
            estimated_quote_value REAL NOT NULL,
            available_quote_quantity REAL,
            available_asset_quantity REAL,
            protected_floor_quantity REAL,
            policy_sellable_quantity REAL,
            projected_holding_quantity REAL,
            status TEXT NOT NULL,
            command_id TEXT UNIQUE,
            client_order_id TEXT NOT NULL UNIQUE,
            exchange_order_id TEXT,
            executed_quantity REAL,
            executed_quote_quantity REAL,
            average_price REAL,
            fee_amount REAL,
            fee_asset TEXT,
            source TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'strategy')),
            swing_id TEXT,
            reason_text TEXT,
            reason_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            request_json TEXT NOT NULL,
            result_json TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_order_intents_created
        ON spot_order_intents(created_at_ms DESC);

        CREATE TABLE IF NOT EXISTS spot_order_status_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}',
            occurred_at_ms INTEGER NOT NULL,
            FOREIGN KEY(intent_id) REFERENCES spot_order_intents(intent_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_spot_order_status_events_intent
        ON spot_order_status_events(intent_id, occurred_at_ms ASC, id ASC);

        CREATE TABLE IF NOT EXISTS spot_swing_executions (
            execution_id TEXT PRIMARY KEY,
            swing_id TEXT NOT NULL,
            spot_order_intent_id TEXT,
            venue TEXT NOT NULL DEFAULT 'binance',
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
            quantity REAL NOT NULL CHECK (quantity > 0),
            price REAL NOT NULL CHECK (price > 0),
            quote_quantity REAL,
            fee_amount REAL,
            fee_asset TEXT,
            exchange_order_id TEXT,
            exchange_trade_id TEXT,
            exchange_execution_key TEXT UNIQUE,
            source TEXT NOT NULL DEFAULT 'strategy' CHECK (source IN ('manual', 'strategy')),
            reason_text TEXT,
            reason_json TEXT NOT NULL DEFAULT '{}',
            executed_at_ms INTEGER NOT NULL,
            executed_at_text TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            FOREIGN KEY(swing_id) REFERENCES spot_swings(swing_id),
            FOREIGN KEY(spot_order_intent_id) REFERENCES spot_order_intents(intent_id)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_swing_executions_swing_time
        ON spot_swing_executions(swing_id, executed_at_ms ASC, execution_id ASC);

        CREATE TABLE IF NOT EXISTS spot_signal_snapshots (
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            market_symbol TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
            opportunity TEXT CHECK (opportunity IS NULL OR opportunity IN ('hold', 'buy', 'sell')),
            level INTEGER CHECK (level IS NULL OR level >= 0),
            base_tranche_pct REAL,
            rolling_change_pct REAL,
            current_price REAL,
            reference_price REAL,
            current_at_ms INTEGER,
            reference_at_ms INTEGER,
            window_ms INTEGER,
            strategy_eligible INTEGER NOT NULL DEFAULT 0,
            eligibility_reason TEXT,
            trading_objective TEXT CHECK (
                trading_objective IS NULL OR trading_objective IN ('accumulate_cash', 'accumulate_asset')
            ),
            levels_json TEXT NOT NULL DEFAULT '[]',
            base_tranches_json TEXT NOT NULL DEFAULT '[]',
            reason_code TEXT,
            error TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            evaluated_at_ms INTEGER,
            last_attempt_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (account_key, asset_symbol, quote_symbol),
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_signal_snapshots_opportunity
        ON spot_signal_snapshots(account_key, opportunity, level, evaluated_at_ms DESC);

        CREATE TABLE IF NOT EXISTS spot_strategy_campaigns (
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            campaign_id TEXT,
            active_side TEXT CHECK (active_side IS NULL OR active_side IN ('buy', 'sell')),
            highest_completed_level INTEGER NOT NULL DEFAULT 0,
            last_signal_level INTEGER NOT NULL DEFAULT 0,
            last_signal_at_ms INTEGER,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (account_key, asset_symbol, quote_symbol),
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE TABLE IF NOT EXISTS spot_strategy_actions (
            action_key TEXT PRIMARY KEY,
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            campaign_id TEXT,
            action_type TEXT NOT NULL CHECK (action_type IN ('open', 'close')),
            side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
            signal_level INTEGER,
            swing_id TEXT NOT NULL,
            intent_id TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('planned', 'intent_created', 'executing', 'completed', 'retryable', 'blocked')
            ),
            requested_quantity REAL NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            reason_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            completed_at_ms INTEGER,
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_spot_strategy_actions_asset_status
        ON spot_strategy_actions(account_key, asset_symbol, quote_symbol, status, created_at_ms ASC);

        CREATE TABLE IF NOT EXISTS spot_swing_target_ratchets (
            swing_id TEXT PRIMARY KEY,
            account_key TEXT NOT NULL,
            asset_symbol TEXT NOT NULL,
            quote_symbol TEXT NOT NULL DEFAULT 'USDT',
            previous_target_quantity REAL NOT NULL,
            applied_gain_quantity REAL NOT NULL,
            next_target_quantity REAL NOT NULL,
            applied_at_ms INTEGER NOT NULL,
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key),
            FOREIGN KEY(swing_id) REFERENCES spot_swings(swing_id)
        );

        CREATE TABLE IF NOT EXISTS portfolio_daily_snapshots (
            account_key TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            captured_at_ms INTEGER NOT NULL,
            total_value REAL NOT NULL,
            invested_capital REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            dry_powder REAL NOT NULL,
            holdings_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (account_key, snapshot_date),
            FOREIGN KEY(account_key) REFERENCES portfolio_accounts(account_key)
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_daily_snapshots_time
        ON portfolio_daily_snapshots(account_key, captured_at_ms ASC);
        """
    )
    transaction_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(portfolio_transactions)").fetchall()
    }
    for column_name, definition in (
        ("custody_location", "TEXT NOT NULL DEFAULT 'unassigned'"),
        ("source_custody", "TEXT"),
        ("destination_custody", "TEXT"),
    ):
        if column_name not in transaction_columns:
            connection.execute(f"ALTER TABLE portfolio_transactions ADD COLUMN {column_name} {definition}")
    policy_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(portfolio_asset_policies)").fetchall()
    }
    if "trading_objective" not in policy_columns:
        connection.execute(
            "ALTER TABLE portfolio_asset_policies ADD COLUMN trading_objective "
            "TEXT NOT NULL DEFAULT 'accumulate_cash' "
            "CHECK (trading_objective IN ('accumulate_cash', 'accumulate_asset'))"
        )
    connection.execute(
        "UPDATE portfolio_asset_policies SET trading_objective = 'accumulate_cash' "
        "WHERE trading_objective IS NULL"
    )
    intent_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(spot_order_intents)").fetchall()
    }
    for column_name, definition in (
        ("source", "TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'strategy'))"),
        ("swing_id", "TEXT"),
        ("reason_text", "TEXT"),
        ("reason_json", "TEXT NOT NULL DEFAULT '{}'")
    ):
        if column_name not in intent_columns:
            connection.execute(f"ALTER TABLE spot_order_intents ADD COLUMN {column_name} {definition}")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_spot_order_intents_swing ON spot_order_intents(swing_id, created_at_ms DESC)"
    )
    ledger_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(ui_log_entries)").fetchall()
    }
    for column_name, definition in (
        ("entry_id", "TEXT"),
        ("scope", "TEXT NOT NULL DEFAULT 'trades'"),
        ("code", "TEXT"),
        ("tone", "TEXT NOT NULL DEFAULT 'neutral'"),
        ("message_text", "TEXT"),
        ("source_ref", "TEXT"),
        ("context_json", "TEXT NOT NULL DEFAULT '{}'"),
    ):
        if column_name not in ledger_columns:
            connection.execute(f"ALTER TABLE ui_log_entries ADD COLUMN {column_name} {definition}")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ui_log_entries_entry_id ON ui_log_entries(entry_id) WHERE entry_id IS NOT NULL"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ui_log_entries_source_ref ON ui_log_entries(source_ref) WHERE source_ref IS NOT NULL"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_ui_log_entries_scope_sort ON ui_log_entries(scope, sort_ts_ms DESC, id DESC)"
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO portfolio_accounts (
            account_key,
            name,
            account_type,
            base_currency,
            archived,
            created_at_ms,
            updated_at_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("manual_spot", "Manual Spot", "spot", "USDT", 0, applied_at_ms, applied_at_ms),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, description, applied_at_ms)
        VALUES (?, ?, ?)
        """,
        (RUNTIME_DB_SCHEMA_VERSION, RUNTIME_DB_SCHEMA_DESCRIPTION, applied_at_ms),
    )


@contextmanager
def _connection(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    resolved_path = (path or runtime_db_path()).expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(resolved_path, timeout=30)
        _configure_connection(connection)
        _ensure_schema(connection)
    except sqlite3.Error as exc:
        raise RuntimeDbError(f"Failed to open runtime database at {resolved_path}: {exc}") from exc

    try:
        yield connection
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise RuntimeDbError(f"Runtime database operation failed at {resolved_path}: {exc}") from exc
    finally:
        connection.close()


def save_strategy_chart_snapshots(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    if not rows:
        return
    with _connection(path) as connection:
        connection.executemany(
            """INSERT INTO strategy_chart_snapshots
               (symbol, kind, bucket_ms, ts_ms, payload_json) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(symbol, kind, bucket_ms) DO UPDATE SET
               ts_ms=excluded.ts_ms, payload_json=excluded.payload_json
               WHERE excluded.ts_ms >= strategy_chart_snapshots.ts_ms""",
            [(row["symbol"], row["kind"], row["bucket_ms"], row["ts_ms"], _json_text(row)) for row in rows],
        )


def list_strategy_chart_snapshots(
    symbol: str, start_ms: int, end_ms: int, path: Path | None = None,
) -> list[dict[str, Any]]:
    with _connection(path) as connection:
        rows = connection.execute(
            """SELECT payload_json FROM strategy_chart_snapshots
               WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms, kind""",
            (symbol, start_ms, end_ms),
        ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def bootstrap_runtime_db(
    path: Path | None = None,
    *,
    initial_state: dict[str, Any] | None = None,
) -> Path:
    """
    Create or open the runtime DB, ensure schema is present, and optionally seed
    the initial runtime state snapshot. No legacy file import is performed here.
    """
    resolved_path = (path or runtime_db_path()).expanduser()
    with _connection(resolved_path) as connection:
        if initial_state is not None:
            row = connection.execute(
                "SELECT 1 FROM runtime_state_snapshot WHERE state_key = ? LIMIT 1",
                (STATE_SNAPSHOT_KEY,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO runtime_state_snapshot (state_key, payload_json, updated_at_ms)
                    VALUES (?, ?, ?)
                    """,
                    (
                        STATE_SNAPSHOT_KEY,
                        _json_text(initial_state),
                        int(datetime.now(timezone.utc).timestamp() * 1000),
                    ),
                )
    return resolved_path


def current_runtime_db_schema_version(path: Path | None = None) -> int:
    with _connection(path) as connection:
        row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    return int(row["version"]) if row is not None and row["version"] is not None else 0


def trade_history_row_count(path: Path | None = None) -> int:
    with _connection(path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM trade_history").fetchone()
    return int(row["count"]) if row is not None else 0


def balance_history_row_count(path: Path | None = None) -> int:
    with _connection(path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM balance_history").fetchone()
    return int(row["count"]) if row is not None else 0


def append_trade_history_row(trade: dict[str, Any], path: Path | None = None) -> None:
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO trade_history (
                row_key,
                sort_ts_ms,
                exit_time,
                entry_time,
                trade_time,
                side,
                entry_price,
                exit_price,
                size,
                net_pnl,
                fee,
                exit_reason,
                trigger,
                trail_active,
                payload_json,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _trade_record(trade),
        )


def replace_trade_history_snapshot(trades: list[dict[str, Any]], path: Path | None = None) -> None:
    records = [_trade_record(trade) for trade in trades]
    with _connection(path) as connection:
        connection.execute("DELETE FROM trade_history")
        if records:
            connection.executemany(
                """
                INSERT INTO trade_history (
                    row_key,
                    sort_ts_ms,
                    exit_time,
                    entry_time,
                    trade_time,
                    side,
                    entry_price,
                    exit_price,
                    size,
                    net_pnl,
                    fee,
                    exit_reason,
                    trigger,
                    trail_active,
                    payload_json,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )


def list_trade_history_rows(
    *,
    limit: int | None = None,
    offset: int = 0,
    lookback_days: int | None = None,
    newest_first: bool = False,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_clause = ""
    if lookback_days is not None:
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
        where_clause = "WHERE sort_ts_ms IS NOT NULL AND sort_ts_ms >= ?"
        params.append(cutoff_ms)

    sql = f"""
        SELECT payload_json
        FROM trade_history
        {where_clause}
        ORDER BY sort_ts_ms {"DESC" if newest_first else "ASC"}, id {"DESC" if newest_first else "ASC"}
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
        if offset > 0:
            sql += " OFFSET ?"
            params.append(offset)
    elif offset > 0:
        sql += " LIMIT -1 OFFSET ?"
        params.append(offset)

    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    output: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        if isinstance(payload, dict):
            output.append(payload)
    return output


def summarize_trade_history(lookback_days: int | None = None, path: Path | None = None) -> dict[str, Any]:
    params: list[Any] = []
    where_clause = ""
    if lookback_days is not None:
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
        where_clause = "WHERE sort_ts_ms IS NOT NULL AND sort_ts_ms >= ?"
        params.append(cutoff_ms)

    sql = f"""
        SELECT
            COUNT(*) AS total_trades,
            COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS winning_trades,
            COALESCE(SUM(CASE WHEN net_pnl < 0 THEN 1 ELSE 0 END), 0) AS losing_trades,
            COALESCE(SUM(CASE WHEN net_pnl IS NULL OR net_pnl = 0 THEN 1 ELSE 0 END), 0) AS breakeven_trades,
            COALESCE(SUM(COALESCE(net_pnl, 0)), 0.0) AS net_pnl_total
        FROM trade_history
        {where_clause}
    """

    with _connection(path) as connection:
        row = connection.execute(sql, params).fetchone()

    total_trades = int(row["total_trades"]) if row is not None else 0
    winning_trades = int(row["winning_trades"]) if row is not None else 0
    losing_trades = int(row["losing_trades"]) if row is not None else 0
    breakeven_trades = int(row["breakeven_trades"]) if row is not None else 0
    net_pnl_total = float(row["net_pnl_total"]) if row is not None else 0.0
    win_rate_pct = (winning_trades / total_trades) * 100.0 if total_trades > 0 else None

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "net_pnl_total": net_pnl_total,
        "win_rate_pct": win_rate_pct,
    }


def count_trade_history_rows(*, lookback_days: int | None = None, path: Path | None = None) -> int:
    params: list[Any] = []
    where_clause = ""
    if lookback_days is not None:
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
        where_clause = "WHERE sort_ts_ms IS NOT NULL AND sort_ts_ms >= ?"
        params.append(cutoff_ms)

    sql = f"SELECT COUNT(*) AS count FROM trade_history {where_clause}"
    with _connection(path) as connection:
        row = connection.execute(sql, params).fetchone()
    return int(row["count"]) if row is not None else 0


def append_balance_history_row(balance: float | dict[str, Any], path: Path | None = None) -> None:
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO balance_history (
                row_key,
                sort_ts_ms,
                balance,
                payload_json,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?)
            """,
            _balance_record(balance),
        )


def replace_balance_history_snapshot(history: list[float], path: Path | None = None) -> None:
    records = [_balance_record(value, snapshot_index=index) for index, value in enumerate(history)]
    with _connection(path) as connection:
        connection.execute("DELETE FROM balance_history")
        if records:
            connection.executemany(
                """
                INSERT INTO balance_history (
                    row_key,
                    sort_ts_ms,
                    balance,
                    payload_json,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )


def list_balance_history_values(*, limit: int | None = None, path: Path | None = None) -> list[float]:
    sql = """
        SELECT balance
        FROM balance_history
        ORDER BY
            CASE WHEN sort_ts_ms IS NULL THEN 1 ELSE 0 END ASC,
            sort_ts_ms ASC,
            id ASC
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [float(row["balance"]) for row in rows]


def runtime_state_snapshot_exists(path: Path | None = None) -> bool:
    with _connection(path) as connection:
        row = connection.execute(
            "SELECT 1 FROM runtime_state_snapshot WHERE state_key = ? LIMIT 1",
            (STATE_SNAPSHOT_KEY,),
        ).fetchone()
    return row is not None


def load_runtime_state_snapshot(path: Path | None = None) -> dict[str, Any] | None:
    with _connection(path) as connection:
        row = connection.execute(
            "SELECT payload_json FROM runtime_state_snapshot WHERE state_key = ? LIMIT 1",
            (STATE_SNAPSHOT_KEY,),
        ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload_json"])
    return payload if isinstance(payload, dict) else None


def save_runtime_state_snapshot(state: dict[str, Any], path: Path | None = None) -> None:
    payload_json = _json_text(state)
    updated_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO runtime_state_snapshot (state_key, payload_json, updated_at_ms)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at_ms = excluded.updated_at_ms
            """,
            (STATE_SNAPSHOT_KEY, payload_json, updated_at_ms),
        )


def event_history_row_count(path: Path | None = None) -> int:
    with _connection(path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM event_history").fetchone()
    return int(row["count"]) if row is not None else 0


def append_event_record(event: dict[str, Any], path: Path | None = None) -> None:
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO event_history (
                event_id,
                sort_ts_ms,
                ts_text,
                level,
                code,
                category,
                notify,
                runtime_mode,
                strategy_mode,
                ui_message,
                context_json,
                payload_json,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _event_record(event),
        )


def list_event_records(
    *,
    limit: int | None = None,
    newest_first: bool = False,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT payload_json
        FROM event_history
        ORDER BY sort_ts_ms {"DESC" if newest_first else "ASC"}, id {"DESC" if newest_first else "ASC"}
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    output: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        if isinstance(payload, dict):
            output.append(payload)
    return output


def ui_log_row_count(path: Path | None = None) -> int:
    with _connection(path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM ui_log_entries").fetchone()
    return int(row["count"]) if row is not None else 0


def append_ui_log_entry(
    ts: str,
    line: str,
    path: Path | None = None,
    *,
    entry_id: str | None = None,
    scope: str = "trades",
    code: str | None = None,
    tone: str = "neutral",
    message: str | None = None,
    source_ref: str | None = None,
    context: dict[str, Any] | None = None,
) -> bool:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    normalized_scope = str(scope or "trades").strip().lower()
    if normalized_scope not in {"trades", "treasury"}:
        normalized_scope = "trades"
    normalized_tone = str(tone or "neutral").strip().lower()
    if normalized_tone not in {"neutral", "open", "positive", "negative"}:
        normalized_tone = "neutral"
    normalized_line = str(line or "").rstrip("\n")
    with _connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO ui_log_entries (
                entry_id,
                sort_ts_ms,
                ts_text,
                line_text,
                scope,
                code,
                tone,
                message_text,
                source_ref,
                context_json,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry_id or "").strip() or None,
                _parse_timestamp_to_ms(ts) or now_ms,
                str(ts or "").strip() or None,
                normalized_line,
                normalized_scope,
                str(code or "").strip() or None,
                normalized_tone,
                str(message or "").strip() or None,
                str(source_ref or "").strip() or None,
                _json_text(context if isinstance(context, dict) else {}),
                now_ms,
            ),
        )
    return cursor.rowcount > 0


def list_ui_log_lines(*, limit: int, path: Path | None = None) -> list[str]:
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT line_text
            FROM ui_log_entries
            ORDER BY sort_ts_ms DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [str(row["line_text"]) for row in reversed(rows)]


def list_ledger_entries(
    *,
    scope: str = "all",
    limit: int = 30,
    before: tuple[int, int] | None = None,
    path: Path | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    normalized_scope = str(scope or "all").strip().lower()
    if normalized_scope not in {"all", "trades", "treasury"}:
        raise ValueError("Ledger scope must be all, trades, or treasury.")
    resolved_limit = max(1, min(int(limit), 100))
    conditions = ["NOT (entry_id IS NULL AND lower(line_text) LIKE '%manual spot order executed.%')"]
    params: list[Any] = []
    if normalized_scope != "all":
        conditions.append("scope = ?")
        params.append(normalized_scope)
    if before is not None:
        conditions.append("(COALESCE(sort_ts_ms, created_at_ms) < ? OR (COALESCE(sort_ts_ms, created_at_ms) = ? AND id < ?))")
        params.extend((before[0], before[0], before[1]))
    where_clause = " AND ".join(conditions)
    params.append(resolved_limit + 1)
    with _connection(path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                id,
                entry_id,
                COALESCE(sort_ts_ms, created_at_ms) AS effective_sort_ts_ms,
                ts_text,
                line_text,
                scope,
                code,
                tone,
                message_text,
                source_ref,
                context_json
            FROM ui_log_entries
            WHERE {where_clause}
            ORDER BY effective_sort_ts_ms DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    has_more = len(rows) > resolved_limit
    page_rows = rows[:resolved_limit]
    entries: list[dict[str, Any]] = []
    for row in page_rows:
        line_text = str(row["line_text"] or "")
        message_text = str(row["message_text"] or "").strip()
        if not message_text:
            closing_bracket = line_text.find("]")
            message_text = line_text[closing_bracket + 1 :].strip() if line_text.startswith("[") and closing_bracket >= 0 else line_text
        try:
            context = json.loads(row["context_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            context = {}
        entries.append(
            {
                "entry_id": str(row["entry_id"] or f"legacy:{row['id']}"),
                "scope": str(row["scope"] or "trades"),
                "timestamp": str(row["ts_text"] or "") or None,
                "sort_ts_ms": int(row["effective_sort_ts_ms"]),
                "code": str(row["code"] or "legacy_log"),
                "tone": str(row["tone"] or "neutral"),
                "message": message_text,
                "source_ref": str(row["source_ref"] or "") or None,
                "context": context if isinstance(context, dict) else {},
            }
        )
    next_cursor = None
    if has_more and page_rows:
        last = page_rows[-1]
        next_cursor = (int(last["effective_sort_ts_ms"]), int(last["id"]))
    return entries, next_cursor


def list_ledger_source_refs(*, prefix: str, path: Path | None = None) -> set[str]:
    normalized_prefix = str(prefix or "").strip()
    with _connection(path) as connection:
        rows = connection.execute(
            "SELECT source_ref FROM ui_log_entries WHERE source_ref LIKE ?",
            (f"{normalized_prefix}%",),
        ).fetchall()
    return {str(row["source_ref"]) for row in rows if row["source_ref"] is not None}


def count_ledger_entries(*, scope: str = "all", path: Path | None = None) -> int:
    normalized_scope = str(scope or "all").strip().lower()
    if normalized_scope not in {"all", "trades", "treasury"}:
        raise ValueError("Ledger scope must be all, trades, or treasury.")
    conditions = ["NOT (entry_id IS NULL AND lower(line_text) LIKE '%manual spot order executed.%')"]
    params: list[Any] = []
    if normalized_scope != "all":
        conditions.append("scope = ?")
        params.append(normalized_scope)
    with _connection(path) as connection:
        row = connection.execute(
            f"SELECT COUNT(*) AS count FROM ui_log_entries WHERE {' AND '.join(conditions)}",
            params,
        ).fetchone()
    return int(row["count"]) if row is not None else 0


def ensure_portfolio_account(
    *,
    account_key: str = "manual_spot",
    name: str = "Manual Spot",
    account_type: str = "spot",
    base_currency: str = "USDT",
    path: Path | None = None,
) -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO portfolio_accounts (
                account_key,
                name,
                account_type,
                base_currency,
                archived,
                created_at_ms,
                updated_at_ms
            )
            VALUES (?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(account_key)
            DO UPDATE SET
                name = excluded.name,
                account_type = excluded.account_type,
                base_currency = excluded.base_currency,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                str(account_key or "manual_spot").strip() or "manual_spot",
                str(name or "Manual Spot").strip() or "Manual Spot",
                str(account_type or "spot").strip().lower() or "spot",
                str(base_currency or "USDT").strip().upper() or "USDT",
                now_ms,
                now_ms,
            ),
        )


def append_portfolio_transaction(transaction: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    record = _portfolio_transaction_record(transaction)
    account_key = str(record[1])
    with _connection(path) as connection:
        account_row = connection.execute(
            "SELECT 1 FROM portfolio_accounts WHERE account_key = ? LIMIT 1",
            (account_key,),
        ).fetchone()
        if account_row is None:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            connection.execute(
                """
                INSERT INTO portfolio_accounts (
                    account_key,
                    name,
                    account_type,
                    base_currency,
                    archived,
                    created_at_ms,
                    updated_at_ms
                )
                VALUES (?, ?, 'spot', ?, 0, ?, ?)
                """,
                (account_key, account_key.replace("_", " ").title(), str(record[6] or "USDT"), now_ms, now_ms),
            )
        connection.execute(
            """
            INSERT INTO portfolio_transactions (
                transaction_id,
                account_key,
                executed_at_ms,
                executed_at_text,
                tx_type,
                asset_symbol,
                quote_symbol,
                quantity,
                price,
                fee_amount,
                fee_asset,
                source,
                status,
                note,
                external_order_id,
                custody_location,
                source_custody,
                destination_custody,
                payload_json,
                created_at_ms,
                updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record,
        )

    payload = dict(transaction)
    payload.setdefault("transaction_id", record[0])
    payload.setdefault("account_key", record[1])
    payload.setdefault("executed_at", record[3])
    return payload


def update_portfolio_transaction_status(
    transaction_id: str,
    status: str,
    path: Path | None = None,
) -> dict[str, Any] | None:
    normalized_id = str(transaction_id or "").strip()
    normalized_status = str(status or "").strip().lower()
    if not normalized_id or normalized_status not in {"settled", "voided"}:
        return None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        row = connection.execute(
            "SELECT payload_json FROM portfolio_transactions WHERE transaction_id = ? LIMIT 1",
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            payload = {}
        payload["status"] = normalized_status
        connection.execute(
            """
            UPDATE portfolio_transactions
            SET status = ?, payload_json = ?, updated_at_ms = ?
            WHERE transaction_id = ?
            """,
            (normalized_status, _json_text(payload), now_ms, normalized_id),
        )

    updated = list_portfolio_transactions(path=path)
    return next((item for item in updated if item.get("transaction_id") == normalized_id), None)


def list_portfolio_transactions(
    *,
    limit: int | None = None,
    offset: int = 0,
    newest_first: bool = True,
    account_key: str | None = None,
    asset_symbol: str | None = None,
    quote_symbol: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT
            id,
            transaction_id,
            account_key,
            executed_at_ms,
            executed_at_text,
            tx_type,
            asset_symbol,
            quote_symbol,
            quantity,
            price,
            fee_amount,
            fee_asset,
            source,
            status,
            note,
            external_order_id,
            custody_location,
            source_custody,
            destination_custody,
            payload_json,
            created_at_ms,
            updated_at_ms
        FROM portfolio_transactions
    """
    params: list[Any] = []
    conditions: list[str] = []
    if account_key is not None:
        conditions.append("account_key = ?")
        params.append(account_key)
    if asset_symbol is not None:
        conditions.append("asset_symbol = ?")
        params.append(str(asset_symbol).strip().upper())
    if quote_symbol is not None:
        conditions.append("quote_symbol = ?")
        params.append(str(quote_symbol).strip().upper())
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += f' ORDER BY executed_at_ms {"DESC" if newest_first else "ASC"}, id {"DESC" if newest_first else "ASC"}'
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
        if offset > 0:
            sql += " OFFSET ?"
            params.append(offset)
    elif offset > 0:
        sql += " LIMIT -1 OFFSET ?"
        params.append(offset)

    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    output: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            payload = {}
        payload.update(
            {
                "id": int(row["id"]),
                "transaction_id": str(row["transaction_id"]),
                "account_key": str(row["account_key"]),
                "executed_at": str(row["executed_at_text"]),
                "executed_at_ms": int(row["executed_at_ms"]),
                "tx_type": str(row["tx_type"]),
                "asset_symbol": str(row["asset_symbol"]),
                "quote_symbol": str(row["quote_symbol"]),
                "quantity": float(row["quantity"]),
                "price": float(row["price"]) if row["price"] is not None else None,
                "fee_amount": float(row["fee_amount"]) if row["fee_amount"] is not None else None,
                "fee_asset": str(row["fee_asset"]) if row["fee_asset"] is not None else None,
                "source": str(row["source"]),
                "status": str(row["status"]),
                "note": str(row["note"]) if row["note"] is not None else None,
                "external_order_id": str(row["external_order_id"]) if row["external_order_id"] is not None else None,
                "custody_location": str(row["custody_location"] or "unassigned"),
                "source_custody": str(row["source_custody"]) if row["source_custody"] is not None else None,
                "destination_custody": str(row["destination_custody"]) if row["destination_custody"] is not None else None,
            }
        )
        output.append(payload)
    return output


def count_portfolio_transactions(
    *,
    account_key: str | None = None,
    asset_symbol: str | None = None,
    quote_symbol: str | None = None,
    path: Path | None = None,
) -> int:
    conditions: list[str] = []
    params: list[Any] = []
    if account_key is not None:
        conditions.append("account_key = ?")
        params.append(account_key)
    if asset_symbol is not None:
        conditions.append("asset_symbol = ?")
        params.append(str(asset_symbol).strip().upper())
    if quote_symbol is not None:
        conditions.append("quote_symbol = ?")
        params.append(str(quote_symbol).strip().upper())
    sql = "SELECT COUNT(*) AS count FROM portfolio_transactions"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    with _connection(path) as connection:
        row = connection.execute(sql, params).fetchone()
    return int(row["count"]) if row is not None else 0


def list_portfolio_asset_policies(
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> list[dict[str, Any]]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT
                account_key,
                asset_symbol,
                quote_symbol,
                target_quantity,
                minimum_holding_pct,
                trading_objective,
                created_at_ms,
                updated_at_ms
            FROM portfolio_asset_policies
            WHERE account_key = ?
            ORDER BY asset_symbol ASC, quote_symbol ASC
            """,
            (normalized_account,),
        ).fetchall()
    return [
        {
            "account_key": str(row["account_key"]),
            "asset_symbol": str(row["asset_symbol"]),
            "quote_symbol": str(row["quote_symbol"]),
            "target_quantity": float(row["target_quantity"]),
            "minimum_holding_pct": float(row["minimum_holding_pct"]),
            "trading_objective": str(row["trading_objective"] or "accumulate_cash"),
            "created_at_ms": int(row["created_at_ms"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }
        for row in rows
    ]


def ensure_portfolio_asset_policies(
    policies: list[dict[str, Any]],
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> list[dict[str, Any]]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    records: list[tuple[Any, ...]] = []
    for policy in policies:
        asset_symbol = str(policy.get("asset_symbol") or "").strip().upper()
        target_quantity = _as_float_or_none(policy.get("target_quantity"))
        minimum_holding_pct = _as_float_or_none(policy.get("minimum_holding_pct"))
        trading_objective = str(policy.get("trading_objective") or "accumulate_cash").strip().lower()
        if not asset_symbol or target_quantity is None:
            continue
        records.append(
            (
                normalized_account,
                asset_symbol,
                str(policy.get("quote_symbol") or "USDT").strip().upper() or "USDT",
                target_quantity,
                minimum_holding_pct if minimum_holding_pct is not None else 100.0,
                trading_objective,
                now_ms,
                now_ms,
            )
        )
    if records:
        with _connection(path) as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO portfolio_asset_policies (
                    account_key,
                    asset_symbol,
                    quote_symbol,
                    target_quantity,
                    minimum_holding_pct,
                    trading_objective,
                    created_at_ms,
                    updated_at_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
    return list_portfolio_asset_policies(account_key=normalized_account, path=path)


def upsert_portfolio_asset_policy(
    policy: dict[str, Any],
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> dict[str, Any]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    asset_symbol = str(policy.get("asset_symbol") or "").strip().upper()
    quote_symbol = str(policy.get("quote_symbol") or "USDT").strip().upper() or "USDT"
    target_quantity = _as_float_or_none(policy.get("target_quantity"))
    minimum_holding_pct = _as_float_or_none(policy.get("minimum_holding_pct"))
    if "trading_objective" in policy:
        trading_objective = str(policy.get("trading_objective") or "accumulate_cash").strip().lower()
    else:
        existing = next(
            (
                item
                for item in list_portfolio_asset_policies(account_key=normalized_account, path=path)
                if item["asset_symbol"] == asset_symbol and item["quote_symbol"] == quote_symbol
            ),
            None,
        )
        trading_objective = existing.get("trading_objective") if existing is not None else "accumulate_cash"
        trading_objective = trading_objective or "accumulate_cash"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO portfolio_asset_policies (
                account_key,
                asset_symbol,
                quote_symbol,
                target_quantity,
                minimum_holding_pct,
                trading_objective,
                created_at_ms,
                updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_key, asset_symbol, quote_symbol) DO UPDATE SET
                target_quantity = excluded.target_quantity,
                minimum_holding_pct = excluded.minimum_holding_pct,
                trading_objective = excluded.trading_objective,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                normalized_account,
                asset_symbol,
                quote_symbol,
                target_quantity,
                minimum_holding_pct,
                trading_objective,
                now_ms,
                now_ms,
            ),
        )
    policies = list_portfolio_asset_policies(account_key=normalized_account, path=path)
    return next(
        policy
        for policy in policies
        if policy["asset_symbol"] == asset_symbol and policy["quote_symbol"] == quote_symbol
    )


def _spot_signal_snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"] or "{}")
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "account_key": str(row["account_key"]),
            "asset_symbol": str(row["asset_symbol"]),
            "quote_symbol": str(row["quote_symbol"]),
            "market_symbol": str(row["market_symbol"]),
            "status": str(row["status"]),
            "opportunity": str(row["opportunity"]) if row["opportunity"] is not None else None,
            "level": int(row["level"]) if row["level"] is not None else None,
            "base_tranche_pct": _as_float_or_none(row["base_tranche_pct"]),
            "rolling_change_pct": _as_float_or_none(row["rolling_change_pct"]),
            "current_price": _as_float_or_none(row["current_price"]),
            "reference_price": _as_float_or_none(row["reference_price"]),
            "current_at_ms": int(row["current_at_ms"]) if row["current_at_ms"] is not None else None,
            "reference_at_ms": int(row["reference_at_ms"]) if row["reference_at_ms"] is not None else None,
            "window_ms": int(row["window_ms"]) if row["window_ms"] is not None else None,
            "strategy_eligible": bool(row["strategy_eligible"]),
            "eligibility_reason": str(row["eligibility_reason"]) if row["eligibility_reason"] else None,
            "trading_objective": str(row["trading_objective"]) if row["trading_objective"] else None,
            "levels_pct": json.loads(row["levels_json"] or "[]"),
            "base_tranches_pct": json.loads(row["base_tranches_json"] or "[]"),
            "reason_code": str(row["reason_code"]) if row["reason_code"] else None,
            "error": str(row["error"]) if row["error"] else None,
            "evaluated_at_ms": int(row["evaluated_at_ms"]) if row["evaluated_at_ms"] is not None else None,
            "last_attempt_at_ms": int(row["last_attempt_at_ms"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }
    )
    return payload


def save_spot_signal_snapshot(
    snapshot: dict[str, Any],
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> dict[str, Any]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    asset_symbol = str(snapshot.get("asset_symbol") or "").strip().upper()
    quote_symbol = str(snapshot.get("quote_symbol") or "USDT").strip().upper() or "USDT"
    market_symbol = str(snapshot.get("market_symbol") or f"{asset_symbol}{quote_symbol}").strip().upper()
    evaluated_at_ms = int(snapshot.get("evaluated_at_ms") or datetime.now(timezone.utc).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO spot_signal_snapshots (
                account_key, asset_symbol, quote_symbol, market_symbol, status,
                opportunity, level, base_tranche_pct, rolling_change_pct,
                current_price, reference_price, current_at_ms, reference_at_ms,
                window_ms, strategy_eligible, eligibility_reason, trading_objective,
                levels_json, base_tranches_json, reason_code, error, payload_json,
                evaluated_at_ms, last_attempt_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            ON CONFLICT(account_key, asset_symbol, quote_symbol) DO UPDATE SET
                market_symbol = excluded.market_symbol,
                status = 'ok',
                opportunity = excluded.opportunity,
                level = excluded.level,
                base_tranche_pct = excluded.base_tranche_pct,
                rolling_change_pct = excluded.rolling_change_pct,
                current_price = excluded.current_price,
                reference_price = excluded.reference_price,
                current_at_ms = excluded.current_at_ms,
                reference_at_ms = excluded.reference_at_ms,
                window_ms = excluded.window_ms,
                strategy_eligible = excluded.strategy_eligible,
                eligibility_reason = excluded.eligibility_reason,
                trading_objective = excluded.trading_objective,
                levels_json = excluded.levels_json,
                base_tranches_json = excluded.base_tranches_json,
                reason_code = excluded.reason_code,
                error = NULL,
                payload_json = excluded.payload_json,
                evaluated_at_ms = excluded.evaluated_at_ms,
                last_attempt_at_ms = excluded.last_attempt_at_ms,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                normalized_account,
                asset_symbol,
                quote_symbol,
                market_symbol,
                snapshot.get("opportunity"),
                int(snapshot.get("level") or 0),
                _as_float_or_none(snapshot.get("base_tranche_pct")),
                _as_float_or_none(snapshot.get("rolling_change_pct")),
                _as_float_or_none(snapshot.get("current_price")),
                _as_float_or_none(snapshot.get("reference_price")),
                int(snapshot["current_at_ms"]),
                int(snapshot["reference_at_ms"]),
                int(snapshot["window_ms"]),
                1 if bool(snapshot.get("strategy_eligible")) else 0,
                str(snapshot.get("eligibility_reason") or "").strip() or None,
                str(snapshot.get("trading_objective") or "").strip().lower() or None,
                _json_text(snapshot.get("levels_pct") or []),
                _json_text(snapshot.get("base_tranches_pct") or []),
                str(snapshot.get("reason_code") or "").strip() or None,
                _json_text(snapshot),
                evaluated_at_ms,
                evaluated_at_ms,
                now_ms,
            ),
        )
    saved = load_spot_signal_snapshot(
        asset_symbol,
        quote_symbol=quote_symbol,
        account_key=normalized_account,
        path=path,
    )
    if saved is None:
        raise RuntimeDbError(f"Failed to load saved Spot signal snapshot for {market_symbol}.")
    return saved


def mark_spot_signal_snapshot_error(
    asset_symbol: str,
    error: str,
    *,
    quote_symbol: str = "USDT",
    account_key: str = "manual_spot",
    attempted_at_ms: int | None = None,
    path: Path | None = None,
) -> None:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    normalized_asset = str(asset_symbol or "").strip().upper()
    normalized_quote = str(quote_symbol or "USDT").strip().upper() or "USDT"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    attempted_ms = int(attempted_at_ms or now_ms)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO spot_signal_snapshots (
                account_key, asset_symbol, quote_symbol, market_symbol, status,
                strategy_eligible, eligibility_reason, error, payload_json, last_attempt_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, 'error', 0, 'signal_unavailable', ?, '{}', ?, ?)
            ON CONFLICT(account_key, asset_symbol, quote_symbol) DO UPDATE SET
                status = 'error',
                strategy_eligible = 0,
                eligibility_reason = 'signal_unavailable',
                error = excluded.error,
                last_attempt_at_ms = excluded.last_attempt_at_ms,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                normalized_account,
                normalized_asset,
                normalized_quote,
                f"{normalized_asset}{normalized_quote}",
                str(error)[:500],
                attempted_ms,
                now_ms,
            ),
        )


def load_spot_signal_snapshot(
    asset_symbol: str,
    *,
    quote_symbol: str = "USDT",
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> dict[str, Any] | None:
    with _connection(path) as connection:
        row = connection.execute(
            """
            SELECT * FROM spot_signal_snapshots
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ?
            LIMIT 1
            """,
            (
                str(account_key or "manual_spot").strip() or "manual_spot",
                str(asset_symbol or "").strip().upper(),
                str(quote_symbol or "USDT").strip().upper() or "USDT",
            ),
        ).fetchone()
    return _spot_signal_snapshot_from_row(row) if row is not None else None


def list_spot_signal_snapshots(
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> list[dict[str, Any]]:
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM spot_signal_snapshots
            WHERE account_key = ?
            ORDER BY asset_symbol ASC, quote_symbol ASC
            """,
            (str(account_key or "manual_spot").strip() or "manual_spot",),
        ).fetchall()
    return [_spot_signal_snapshot_from_row(row) for row in rows]


def _spot_strategy_campaign_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "account_key": str(row["account_key"]),
        "asset_symbol": str(row["asset_symbol"]),
        "quote_symbol": str(row["quote_symbol"]),
        "campaign_id": str(row["campaign_id"]) if row["campaign_id"] is not None else None,
        "active_side": str(row["active_side"]) if row["active_side"] is not None else None,
        "highest_completed_level": int(row["highest_completed_level"]),
        "last_signal_level": int(row["last_signal_level"]),
        "last_signal_at_ms": int(row["last_signal_at_ms"]) if row["last_signal_at_ms"] is not None else None,
        "created_at_ms": int(row["created_at_ms"]),
        "updated_at_ms": int(row["updated_at_ms"]),
    }


def sync_spot_strategy_campaign(
    *,
    account_key: str,
    asset_symbol: str,
    quote_symbol: str,
    opportunity: str,
    signal_level: int,
    signal_at_ms: int,
    path: Path | None = None,
) -> dict[str, Any]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    normalized_asset = str(asset_symbol or "").strip().upper()
    normalized_quote = str(quote_symbol or "USDT").strip().upper() or "USDT"
    normalized_side = str(opportunity or "hold").strip().lower()
    if normalized_side not in {"hold", "buy", "sell"}:
        raise ValueError("Spot strategy opportunity must be hold, buy, or sell.")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        row = connection.execute(
            """
            SELECT * FROM spot_strategy_campaigns
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ?
            """,
            (normalized_account, normalized_asset, normalized_quote),
        ).fetchone()
        transition = transition_spot_strategy_campaign(
            _spot_strategy_campaign_from_row(row) if row is not None else None,
            opportunity=normalized_side,
            signal_level=signal_level,
            signal_at_ms=signal_at_ms,
            new_campaign_id=_row_key(
                [normalized_account, normalized_asset, normalized_quote, normalized_side, int(signal_at_ms)]
            )[:24],
        )
        connection.execute(
            """
            INSERT INTO spot_strategy_campaigns (
                account_key, asset_symbol, quote_symbol, campaign_id, active_side,
                highest_completed_level, last_signal_level, last_signal_at_ms,
                created_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_key, asset_symbol, quote_symbol) DO UPDATE SET
                campaign_id = excluded.campaign_id,
                active_side = excluded.active_side,
                highest_completed_level = excluded.highest_completed_level,
                last_signal_level = excluded.last_signal_level,
                last_signal_at_ms = excluded.last_signal_at_ms,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                normalized_account,
                normalized_asset,
                normalized_quote,
                transition["campaign_id"],
                transition["active_side"],
                transition["highest_completed_level"],
                transition["last_signal_level"],
                transition["last_signal_at_ms"],
                now_ms,
                now_ms,
            ),
        )
        saved = connection.execute(
            """
            SELECT * FROM spot_strategy_campaigns
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ?
            """,
            (normalized_account, normalized_asset, normalized_quote),
        ).fetchone()
    if saved is None:
        raise RuntimeDbError("Spot strategy campaign was not persisted.")
    return _spot_strategy_campaign_from_row(saved)


def complete_spot_strategy_campaign_level(
    *,
    account_key: str,
    asset_symbol: str,
    quote_symbol: str,
    campaign_id: str,
    signal_level: int,
    path: Path | None = None,
) -> None:
    with _connection(path) as connection:
        connection.execute(
            """
            UPDATE spot_strategy_campaigns
            SET highest_completed_level = MAX(highest_completed_level, ?), updated_at_ms = ?
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ? AND campaign_id = ?
            """,
            (
                int(signal_level),
                int(datetime.now(timezone.utc).timestamp() * 1000),
                str(account_key),
                str(asset_symbol).strip().upper(),
                str(quote_symbol).strip().upper(),
                str(campaign_id),
            ),
        )


def _spot_strategy_action_from_row(row: sqlite3.Row) -> dict[str, Any]:
    reason = json.loads(row["reason_json"] or "{}")
    return {
        "action_key": str(row["action_key"]),
        "account_key": str(row["account_key"]),
        "asset_symbol": str(row["asset_symbol"]),
        "quote_symbol": str(row["quote_symbol"]),
        "campaign_id": str(row["campaign_id"]) if row["campaign_id"] is not None else None,
        "action_type": str(row["action_type"]),
        "side": str(row["side"]),
        "signal_level": int(row["signal_level"]) if row["signal_level"] is not None else None,
        "swing_id": str(row["swing_id"]),
        "intent_id": str(row["intent_id"]) if row["intent_id"] is not None else None,
        "status": str(row["status"]),
        "requested_quantity": float(row["requested_quantity"]),
        "attempt_count": int(row["attempt_count"]),
        "reason": reason if isinstance(reason, dict) else {},
        "error": str(row["error"]) if row["error"] is not None else None,
        "created_at_ms": int(row["created_at_ms"]),
        "updated_at_ms": int(row["updated_at_ms"]),
        "completed_at_ms": int(row["completed_at_ms"]) if row["completed_at_ms"] is not None else None,
    }


def ensure_spot_strategy_action(action: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    action_key = str(action.get("action_key") or "").strip()
    if not action_key:
        raise ValueError("Spot strategy action requires a stable action key.")
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO spot_strategy_actions (
                action_key, account_key, asset_symbol, quote_symbol, campaign_id,
                action_type, side, signal_level, swing_id, intent_id, status,
                requested_quantity, attempt_count, reason_json, error,
                created_at_ms, updated_at_ms, completed_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'planned', ?, 0, ?, NULL, ?, ?, NULL)
            """,
            (
                action_key,
                str(action.get("account_key") or "manual_spot").strip() or "manual_spot",
                str(action.get("asset_symbol") or "").strip().upper(),
                str(action.get("quote_symbol") or "USDT").strip().upper() or "USDT",
                str(action.get("campaign_id") or "").strip() or None,
                str(action.get("action_type") or "").strip().lower(),
                str(action.get("side") or "").strip().lower(),
                int(action["signal_level"]) if action.get("signal_level") is not None else None,
                str(action.get("swing_id") or "").strip(),
                float(action["requested_quantity"]),
                _json_text(action.get("reason") if isinstance(action.get("reason"), dict) else {}),
                now_ms,
                now_ms,
            ),
        )
        row = connection.execute(
            "SELECT * FROM spot_strategy_actions WHERE action_key = ?",
            (action_key,),
        ).fetchone()
    if row is None:
        raise RuntimeDbError("Spot strategy action was not persisted.")
    return _spot_strategy_action_from_row(row)


def update_spot_strategy_action(
    action_key: str,
    updates: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    allowed = {"intent_id", "status", "requested_quantity", "attempt_count", "reason_json", "error", "completed_at_ms"}
    normalized = {key: value for key, value in updates.items() if key in allowed}
    if "reason_json" in normalized:
        normalized["reason_json"] = _json_text(normalized["reason_json"])
    normalized["updated_at_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)
    assignments = ", ".join(f"{key} = ?" for key in normalized)
    with _connection(path) as connection:
        connection.execute(
            f"UPDATE spot_strategy_actions SET {assignments} WHERE action_key = ?",
            [*normalized.values(), str(action_key)],
        )
        row = connection.execute(
            "SELECT * FROM spot_strategy_actions WHERE action_key = ?",
            (str(action_key),),
        ).fetchone()
    return _spot_strategy_action_from_row(row) if row is not None else None


def list_spot_strategy_actions(
    *,
    account_key: str | None = None,
    asset_symbol: str | None = None,
    quote_symbol: str | None = None,
    statuses: set[str] | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("account_key", account_key),
        ("asset_symbol", str(asset_symbol).strip().upper() if asset_symbol is not None else None),
        ("quote_symbol", str(quote_symbol).strip().upper() if quote_symbol is not None else None),
    ):
        if value is not None:
            conditions.append(f"{column} = ?")
            params.append(value)
    if statuses:
        normalized_statuses = sorted({str(value).strip().lower() for value in statuses})
        conditions.append(f"status IN ({', '.join('?' for _ in normalized_statuses)})")
        params.extend(normalized_statuses)
    sql = "SELECT * FROM spot_strategy_actions"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY created_at_ms ASC, action_key ASC"
    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_spot_strategy_action_from_row(row) for row in rows]


def save_exchange_account_snapshot(
    snapshot: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    venue = str(snapshot.get("venue") or "binance").strip().lower() or "binance"
    account_type = str(snapshot.get("account_type") or "spot").strip().lower() or "spot"
    captured_at_ms = int(snapshot.get("captured_at_ms") or datetime.now(timezone.utc).timestamp() * 1000)
    balances = snapshot.get("balances") if isinstance(snapshot.get("balances"), list) else []
    normalized_balances: list[dict[str, Any]] = []
    for balance in balances:
        if not isinstance(balance, dict):
            continue
        asset_symbol = str(balance.get("asset_symbol") or balance.get("asset") or "").strip().upper()
        free = _as_float_or_none(balance.get("free"))
        locked = _as_float_or_none(balance.get("locked"))
        if not asset_symbol or free is None or locked is None:
            continue
        total = free + locked
        if abs(total) < 0.000000000001:
            continue
        normalized_balances.append(
            {
                "asset_symbol": asset_symbol,
                "free": free,
                "locked": locked,
                "total": total,
            }
        )
    payload = {
        "venue": venue,
        "account_type": account_type,
        "status": "ok",
        "captured_at_ms": captured_at_ms,
        "can_trade": bool(snapshot.get("can_trade")),
        "balances": normalized_balances,
    }
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO exchange_account_snapshots (
                venue,
                account_type,
                status,
                captured_at_ms,
                last_attempt_at_ms,
                can_trade,
                error,
                payload_json,
                updated_at_ms
            )
            VALUES (?, ?, 'ok', ?, ?, ?, NULL, ?, ?)
            ON CONFLICT(venue, account_type) DO UPDATE SET
                status = 'ok',
                captured_at_ms = excluded.captured_at_ms,
                last_attempt_at_ms = excluded.last_attempt_at_ms,
                can_trade = excluded.can_trade,
                error = NULL,
                payload_json = excluded.payload_json,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                venue,
                account_type,
                captured_at_ms,
                captured_at_ms,
                1 if payload["can_trade"] else 0,
                _json_text(payload),
                now_ms,
            ),
        )
        connection.execute(
            "DELETE FROM exchange_asset_balances WHERE venue = ? AND account_type = ?",
            (venue, account_type),
        )
        connection.executemany(
            """
            INSERT INTO exchange_asset_balances (
                venue,
                account_type,
                asset_symbol,
                free,
                locked,
                total,
                captured_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    venue,
                    account_type,
                    balance["asset_symbol"],
                    balance["free"],
                    balance["locked"],
                    balance["total"],
                    captured_at_ms,
                )
                for balance in normalized_balances
            ],
        )
    return payload


def mark_exchange_account_snapshot_error(
    error: str,
    *,
    venue: str = "binance",
    account_type: str = "spot",
    attempted_at_ms: int | None = None,
    path: Path | None = None,
) -> None:
    normalized_venue = str(venue or "binance").strip().lower() or "binance"
    normalized_account_type = str(account_type or "spot").strip().lower() or "spot"
    attempt_ms = int(attempted_at_ms or datetime.now(timezone.utc).timestamp() * 1000)
    normalized_error = str(error or "Exchange snapshot refresh failed.").strip()
    with _connection(path) as connection:
        existing = connection.execute(
            """
            SELECT captured_at_ms, can_trade, payload_json
            FROM exchange_account_snapshots
            WHERE venue = ? AND account_type = ?
            LIMIT 1
            """,
            (normalized_venue, normalized_account_type),
        ).fetchone()
        captured_at_ms = int(existing["captured_at_ms"]) if existing and existing["captured_at_ms"] is not None else None
        can_trade = int(existing["can_trade"]) if existing and existing["can_trade"] is not None else None
        payload_json = str(existing["payload_json"]) if existing else "{}"
        connection.execute(
            """
            INSERT INTO exchange_account_snapshots (
                venue,
                account_type,
                status,
                captured_at_ms,
                last_attempt_at_ms,
                can_trade,
                error,
                payload_json,
                updated_at_ms
            )
            VALUES (?, ?, 'error', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(venue, account_type) DO UPDATE SET
                status = 'error',
                last_attempt_at_ms = excluded.last_attempt_at_ms,
                error = excluded.error,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                normalized_venue,
                normalized_account_type,
                captured_at_ms,
                attempt_ms,
                can_trade,
                normalized_error,
                payload_json,
                attempt_ms,
            ),
        )


def load_exchange_account_snapshot(
    *,
    venue: str = "binance",
    account_type: str = "spot",
    path: Path | None = None,
) -> dict[str, Any] | None:
    normalized_venue = str(venue or "binance").strip().lower() or "binance"
    normalized_account_type = str(account_type or "spot").strip().lower() or "spot"
    with _connection(path) as connection:
        account = connection.execute(
            """
            SELECT status, captured_at_ms, last_attempt_at_ms, can_trade, error
            FROM exchange_account_snapshots
            WHERE venue = ? AND account_type = ?
            LIMIT 1
            """,
            (normalized_venue, normalized_account_type),
        ).fetchone()
        if account is None:
            return None
        balance_rows = connection.execute(
            """
            SELECT asset_symbol, free, locked, total
            FROM exchange_asset_balances
            WHERE venue = ? AND account_type = ?
            ORDER BY asset_symbol ASC
            """,
            (normalized_venue, normalized_account_type),
        ).fetchall()
    return {
        "venue": normalized_venue,
        "account_type": normalized_account_type,
        "status": str(account["status"]),
        "captured_at_ms": int(account["captured_at_ms"]) if account["captured_at_ms"] is not None else None,
        "last_attempt_at_ms": int(account["last_attempt_at_ms"]),
        "can_trade": bool(account["can_trade"]) if account["can_trade"] is not None else None,
        "error": str(account["error"]) if account["error"] is not None else None,
        "balances": [
            {
                "asset_symbol": str(row["asset_symbol"]),
                "free": float(row["free"]),
                "locked": float(row["locked"]),
                "total": float(row["total"]),
            }
            for row in balance_rows
        ],
    }


def _spot_order_intent_from_row(row: sqlite3.Row) -> dict[str, Any]:
    request_payload = json.loads(row["request_json"])
    result_payload = json.loads(row["result_json"]) if row["result_json"] else None
    reason_payload = json.loads(row["reason_json"]) if row["reason_json"] else {}
    return {
        "intent_id": str(row["intent_id"]),
        "account_key": str(row["account_key"]),
        "venue": str(row["venue"]),
        "symbol": str(row["symbol"]),
        "asset_symbol": str(row["asset_symbol"]),
        "quote_symbol": str(row["quote_symbol"]),
        "side": str(row["side"]),
        "requested_quantity": float(row["requested_quantity"]),
        "estimated_price": float(row["estimated_price"]),
        "estimated_quote_value": float(row["estimated_quote_value"]),
        "available_quote_quantity": (
            float(row["available_quote_quantity"])
            if row["available_quote_quantity"] is not None
            else None
        ),
        "available_asset_quantity": (
            float(row["available_asset_quantity"])
            if row["available_asset_quantity"] is not None
            else None
        ),
        "protected_floor_quantity": (
            float(row["protected_floor_quantity"])
            if row["protected_floor_quantity"] is not None
            else None
        ),
        "policy_sellable_quantity": (
            float(row["policy_sellable_quantity"])
            if row["policy_sellable_quantity"] is not None
            else None
        ),
        "projected_holding_quantity": (
            float(row["projected_holding_quantity"])
            if row["projected_holding_quantity"] is not None
            else None
        ),
        "status": str(row["status"]),
        "command_id": str(row["command_id"]) if row["command_id"] is not None else None,
        "client_order_id": str(row["client_order_id"]),
        "exchange_order_id": (
            str(row["exchange_order_id"])
            if row["exchange_order_id"] is not None
            else None
        ),
        "executed_quantity": (
            float(row["executed_quantity"])
            if row["executed_quantity"] is not None
            else None
        ),
        "executed_quote_quantity": (
            float(row["executed_quote_quantity"])
            if row["executed_quote_quantity"] is not None
            else None
        ),
        "average_price": float(row["average_price"]) if row["average_price"] is not None else None,
        "fee_amount": float(row["fee_amount"]) if row["fee_amount"] is not None else None,
        "fee_asset": str(row["fee_asset"]) if row["fee_asset"] is not None else None,
        "source": str(row["source"]),
        "swing_id": str(row["swing_id"]) if row["swing_id"] is not None else None,
        "reason_text": str(row["reason_text"]) if row["reason_text"] is not None else None,
        "reason": reason_payload if isinstance(reason_payload, dict) else {},
        "error": str(row["error"]) if row["error"] is not None else None,
        "request": request_payload if isinstance(request_payload, dict) else {},
        "result": result_payload if isinstance(result_payload, dict) else None,
        "created_at_ms": int(row["created_at_ms"]),
        "updated_at_ms": int(row["updated_at_ms"]),
    }


def create_spot_order_intent(intent: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    intent_id = str(intent.get("intent_id") or "").strip()
    account_key = str(intent.get("account_key") or "manual_spot").strip() or "manual_spot"
    client_order_id = str(intent.get("client_order_id") or intent_id).strip()
    if not intent_id or not client_order_id:
        raise ValueError("Spot order intent requires stable intent and client order IDs.")
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO spot_order_intents (
                intent_id, account_key, venue, symbol, asset_symbol, quote_symbol,
                side, requested_quantity, estimated_price, estimated_quote_value,
                available_quote_quantity, available_asset_quantity,
                protected_floor_quantity, policy_sellable_quantity,
                projected_holding_quantity, status, command_id, client_order_id,
                source, swing_id, reason_text, reason_json,
                request_json, created_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'previewed', NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                account_key,
                str(intent.get("venue") or "binance").strip().lower() or "binance",
                str(intent.get("symbol") or "").strip().upper(),
                str(intent.get("asset_symbol") or "").strip().upper(),
                str(intent.get("quote_symbol") or "USDT").strip().upper() or "USDT",
                str(intent.get("side") or "").strip().lower(),
                float(intent["requested_quantity"]),
                float(intent["estimated_price"]),
                float(intent["estimated_quote_value"]),
                _as_float_or_none(intent.get("available_quote_quantity")),
                _as_float_or_none(intent.get("available_asset_quantity")),
                _as_float_or_none(intent.get("protected_floor_quantity")),
                _as_float_or_none(intent.get("policy_sellable_quantity")),
                _as_float_or_none(intent.get("projected_holding_quantity")),
                client_order_id,
                str(intent.get("source") or "manual").strip().lower() or "manual",
                str(intent.get("swing_id") or "").strip() or None,
                str(intent.get("reason_text") or "").strip() or None,
                _json_text(intent.get("reason") if isinstance(intent.get("reason"), dict) else {}),
                _json_text(intent),
                now_ms,
                now_ms,
            ),
        )
        _append_spot_order_status_event(
            connection,
            intent_id,
            "previewed",
            {
                "source": str(intent.get("source") or "manual").strip().lower() or "manual",
                "swing_id": str(intent.get("swing_id") or "").strip() or None,
            },
            occurred_at_ms=now_ms,
        )
    created = load_spot_order_intent(intent_id, path=path)
    if created is None:
        raise RuntimeDbError("Spot order intent was not persisted.")
    return created


def load_spot_order_intent(intent_id: str, path: Path | None = None) -> dict[str, Any] | None:
    normalized_id = str(intent_id or "").strip()
    if not normalized_id:
        return None
    with _connection(path) as connection:
        row = connection.execute(
            "SELECT * FROM spot_order_intents WHERE intent_id = ? LIMIT 1",
            (normalized_id,),
        ).fetchone()
    return _spot_order_intent_from_row(row) if row is not None else None


def list_spot_order_intents(
    *,
    statuses: set[str] | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    sql = "SELECT * FROM spot_order_intents"
    if statuses:
        normalized_statuses = sorted({str(status).strip().lower() for status in statuses if str(status).strip()})
        if normalized_statuses:
            placeholders = ", ".join("?" for _ in normalized_statuses)
            sql += f" WHERE status IN ({placeholders})"
            params.extend(normalized_statuses)
    sql += " ORDER BY created_at_ms ASC, intent_id ASC"
    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_spot_order_intent_from_row(row) for row in rows]


def _append_spot_order_status_event(
    connection: sqlite3.Connection,
    intent_id: str,
    status: str,
    detail: dict[str, Any] | None = None,
    *,
    occurred_at_ms: int | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO spot_order_status_events (intent_id, status, detail_json, occurred_at_ms)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(intent_id or "").strip(),
            str(status or "").strip().lower(),
            _json_text(detail or {}),
            int(occurred_at_ms or datetime.now(timezone.utc).timestamp() * 1000),
        ),
    )


def list_spot_order_status_events(intent_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    normalized_id = str(intent_id or "").strip()
    if not normalized_id:
        return []
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT id, status, detail_json, occurred_at_ms
            FROM spot_order_status_events
            WHERE intent_id = ?
            ORDER BY occurred_at_ms ASC, id ASC
            """,
            (normalized_id,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "intent_id": normalized_id,
            "status": str(row["status"]),
            "detail": json.loads(row["detail_json"] or "{}"),
            "occurred_at_ms": int(row["occurred_at_ms"]),
        }
        for row in rows
    ]


def reserve_spot_order_intent(
    intent_id: str,
    *,
    command_id: str,
    path: Path | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    normalized_id = str(intent_id or "").strip()
    normalized_command_id = str(command_id or "").strip()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with _connection(path) as connection:
        cursor = connection.execute(
            """
            UPDATE spot_order_intents
            SET status = 'queueing', command_id = ?, updated_at_ms = ?
            WHERE intent_id = ? AND status = 'previewed'
            """,
            (normalized_command_id, now_ms, normalized_id),
        )
        if cursor.rowcount == 1:
            _append_spot_order_status_event(
                connection,
                normalized_id,
                "queueing",
                {"command_id": normalized_command_id},
                occurred_at_ms=now_ms,
            )
        row = connection.execute(
            "SELECT * FROM spot_order_intents WHERE intent_id = ? LIMIT 1",
            (normalized_id,),
        ).fetchone()
    return (_spot_order_intent_from_row(row) if row is not None else None, cursor.rowcount == 1)


def update_spot_order_intent(
    intent_id: str,
    updates: dict[str, Any],
    *,
    expected_statuses: set[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    allowed_fields = {
        "status",
        "command_id",
        "exchange_order_id",
        "executed_quantity",
        "executed_quote_quantity",
        "average_price",
        "fee_amount",
        "fee_asset",
        "error",
        "result_json",
    }
    normalized_updates = {key: value for key, value in updates.items() if key in allowed_fields}
    if "result_json" in normalized_updates:
        normalized_updates["result_json"] = _json_text(normalized_updates["result_json"])
    if not normalized_updates:
        return load_spot_order_intent(intent_id, path=path)
    normalized_updates["updated_at_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)
    assignments = ", ".join(f"{field} = ?" for field in normalized_updates)
    params = list(normalized_updates.values())
    sql = f"UPDATE spot_order_intents SET {assignments} WHERE intent_id = ?"
    normalized_id = str(intent_id or "").strip()
    params.append(normalized_id)
    if expected_statuses:
        placeholders = ", ".join("?" for _ in expected_statuses)
        sql += f" AND status IN ({placeholders})"
        params.extend(sorted(expected_statuses))
    previous_status: str | None = None
    with _connection(path) as connection:
        previous_row = connection.execute(
            "SELECT status FROM spot_order_intents WHERE intent_id = ? LIMIT 1",
            (normalized_id,),
        ).fetchone()
        if previous_row is not None:
            previous_status = str(previous_row["status"])
        cursor = connection.execute(sql, params)
        next_status = str(normalized_updates.get("status") or "").strip().lower()
        if cursor.rowcount == 1 and next_status and next_status != previous_status:
            _append_spot_order_status_event(
                connection,
                normalized_id,
                next_status,
                {
                    key: value
                    for key, value in updates.items()
                    if key in {"command_id", "exchange_order_id", "executed_quantity", "error"}
                },
                occurred_at_ms=int(normalized_updates["updated_at_ms"]),
            )
    if cursor.rowcount != 1:
        return None
    return load_spot_order_intent(intent_id, path=path)


def _spot_swing_from_row(row: sqlite3.Row) -> dict[str, Any]:
    reference_state = json.loads(row["reference_state_json"] or "{}")
    strategy_reason = json.loads(row["strategy_reason_json"] or "{}")
    return {
        "swing_id": str(row["swing_id"]),
        "account_key": str(row["account_key"]),
        "asset_symbol": str(row["asset_symbol"]),
        "quote_symbol": str(row["quote_symbol"]),
        "origin_side": str(row["origin_side"]),
        "trading_objective": (
            str(row["trading_objective"]) if row["trading_objective"] is not None else None
        ),
        "status": str(row["status"]),
        "planned_quantity": float(row["planned_quantity"]) if row["planned_quantity"] is not None else None,
        "reference_state": reference_state if isinstance(reference_state, dict) else {},
        "strategy_reason": strategy_reason if isinstance(strategy_reason, dict) else {},
        "source": str(row["source"]),
        "close_reason": str(row["close_reason"]) if row["close_reason"] is not None else None,
        "opened_at_ms": int(row["opened_at_ms"]),
        "closed_at_ms": int(row["closed_at_ms"]) if row["closed_at_ms"] is not None else None,
        "created_at_ms": int(row["created_at_ms"]),
        "updated_at_ms": int(row["updated_at_ms"]),
    }


def create_spot_swing(swing: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    swing_id = str(swing.get("swing_id") or "").strip()
    account_key = str(swing.get("account_key") or "manual_spot").strip() or "manual_spot"
    asset_symbol = str(swing.get("asset_symbol") or "").strip().upper()
    quote_symbol = str(swing.get("quote_symbol") or "USDT").strip().upper() or "USDT"
    origin_side = str(swing.get("origin_side") or "").strip().lower()
    objective = str(swing.get("trading_objective") or "").strip().lower() or None
    status = str(swing.get("status") or "open").strip().lower() or "open"
    source = str(swing.get("source") or "strategy").strip().lower() or "strategy"
    planned_quantity = _as_float_or_none(swing.get("planned_quantity"))
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    opened_at_ms = int(swing.get("opened_at_ms") or _parse_timestamp_to_ms(swing.get("opened_at")) or now_ms)
    closed_at_ms = swing.get("closed_at_ms") or _parse_timestamp_to_ms(swing.get("closed_at"))

    if not swing_id:
        raise ValueError("Spot Swing requires a stable swing ID.")
    if not asset_symbol:
        raise ValueError("Spot Swing requires an asset symbol.")
    if origin_side not in SWING_SIDES:
        raise ValueError("Spot Swing origin side must be buy or sell.")
    if objective is not None and objective not in SWING_OBJECTIVES:
        raise ValueError("Spot Swing objective must be accumulate_cash or accumulate_asset.")
    if status not in SWING_STATUSES:
        raise ValueError("Spot Swing status is invalid.")
    if source not in SWING_SOURCES:
        raise ValueError("Spot Swing source must be manual or strategy.")
    if planned_quantity is not None and planned_quantity <= 0:
        raise ValueError("Spot Swing planned quantity must be greater than zero.")

    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO spot_swings (
                swing_id, account_key, asset_symbol, quote_symbol, origin_side,
                trading_objective, status, planned_quantity, reference_state_json,
                strategy_reason_json, source, close_reason, opened_at_ms,
                closed_at_ms, created_at_ms, updated_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                swing_id,
                account_key,
                asset_symbol,
                quote_symbol,
                origin_side,
                objective,
                status,
                planned_quantity,
                _json_text(swing.get("reference_state") if isinstance(swing.get("reference_state"), dict) else {}),
                _json_text(swing.get("strategy_reason") if isinstance(swing.get("strategy_reason"), dict) else {}),
                source,
                str(swing.get("close_reason") or "").strip().lower() or None,
                opened_at_ms,
                int(closed_at_ms) if closed_at_ms is not None else None,
                now_ms,
                now_ms,
            ),
        )
        row = connection.execute("SELECT * FROM spot_swings WHERE swing_id = ?", (swing_id,)).fetchone()
    if row is None:
        raise RuntimeDbError("Spot Swing was not persisted.")
    return _spot_swing_from_row(row)


def load_spot_swing(swing_id: str, path: Path | None = None) -> dict[str, Any] | None:
    normalized_id = str(swing_id or "").strip()
    if not normalized_id:
        return None
    with _connection(path) as connection:
        row = connection.execute("SELECT * FROM spot_swings WHERE swing_id = ?", (normalized_id,)).fetchone()
    return _spot_swing_from_row(row) if row is not None else None


def list_spot_swings(
    *,
    account_key: str | None = None,
    asset_symbol: str | None = None,
    quote_symbol: str | None = None,
    status: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("account_key", account_key),
        ("asset_symbol", str(asset_symbol).strip().upper() if asset_symbol is not None else None),
        ("quote_symbol", str(quote_symbol).strip().upper() if quote_symbol is not None else None),
        ("status", str(status).strip().lower() if status is not None else None),
    ):
        if value is not None:
            conditions.append(f"{column} = ?")
            params.append(value)
    sql = "SELECT * FROM spot_swings"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY opened_at_ms DESC, swing_id DESC"
    with _connection(path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_spot_swing_from_row(row) for row in rows]


def _spot_swing_execution_from_row(row: sqlite3.Row) -> dict[str, Any]:
    reason = json.loads(row["reason_json"] or "{}")
    payload = json.loads(row["payload_json"] or "{}")
    return {
        "execution_id": str(row["execution_id"]),
        "swing_id": str(row["swing_id"]),
        "spot_order_intent_id": (
            str(row["spot_order_intent_id"]) if row["spot_order_intent_id"] is not None else None
        ),
        "venue": str(row["venue"]),
        "symbol": str(row["symbol"]),
        "side": str(row["side"]),
        "quantity": float(row["quantity"]),
        "price": float(row["price"]),
        "quote_quantity": float(row["quote_quantity"]) if row["quote_quantity"] is not None else None,
        "fee_amount": float(row["fee_amount"]) if row["fee_amount"] is not None else None,
        "fee_asset": str(row["fee_asset"]) if row["fee_asset"] is not None else None,
        "exchange_order_id": (
            str(row["exchange_order_id"]) if row["exchange_order_id"] is not None else None
        ),
        "exchange_trade_id": (
            str(row["exchange_trade_id"]) if row["exchange_trade_id"] is not None else None
        ),
        "exchange_execution_key": (
            str(row["exchange_execution_key"]) if row["exchange_execution_key"] is not None else None
        ),
        "source": str(row["source"]),
        "reason_text": str(row["reason_text"]) if row["reason_text"] is not None else None,
        "reason": reason if isinstance(reason, dict) else {},
        "executed_at_ms": int(row["executed_at_ms"]),
        "executed_at": str(row["executed_at_text"]),
        "payload": payload if isinstance(payload, dict) else {},
        "created_at_ms": int(row["created_at_ms"]),
    }


def list_spot_swing_executions(
    swing_id: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    normalized_id = str(swing_id or "").strip()
    if not normalized_id:
        return []
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM spot_swing_executions
            WHERE swing_id = ?
            ORDER BY executed_at_ms ASC, execution_id ASC
            """,
            (normalized_id,),
        ).fetchall()
    return [_spot_swing_execution_from_row(row) for row in rows]


def _refresh_spot_swing_lifecycle(connection: sqlite3.Connection, swing_id: str) -> None:
    swing_row = connection.execute("SELECT * FROM spot_swings WHERE swing_id = ?", (swing_id,)).fetchone()
    if swing_row is None:
        raise ValueError("Spot Swing was not found.")
    execution_rows = connection.execute(
        """
        SELECT * FROM spot_swing_executions
        WHERE swing_id = ?
        ORDER BY executed_at_ms ASC, execution_id ASC
        """,
        (swing_id,),
    ).fetchall()
    swing = _spot_swing_from_row(swing_row)
    executions = [_spot_swing_execution_from_row(row) for row in execution_rows]
    economics = calculate_swing_economics(swing, executions)
    status = str(economics["status"])
    closed_at_ms = max((item["executed_at_ms"] for item in executions), default=None) if status == "closed" else None
    close_reason = swing.get("close_reason")
    if status == "closed" and not close_reason:
        for execution in reversed(executions):
            reason = execution.get("reason") if isinstance(execution.get("reason"), dict) else {}
            if reason.get("action_type") == "close":
                close_reason = str(reason.get("close_reason") or "strategy_profit").strip().lower()
                break
    connection.execute(
        """
        UPDATE spot_swings
        SET status = ?, close_reason = ?, closed_at_ms = ?, updated_at_ms = ?
        WHERE swing_id = ?
        """,
        (status, close_reason, closed_at_ms, int(datetime.now(timezone.utc).timestamp() * 1000), swing_id),
    )


def append_spot_swing_execution(
    execution: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    execution_id = str(execution.get("execution_id") or "").strip()
    swing_id = str(execution.get("swing_id") or "").strip()
    venue = str(execution.get("venue") or "binance").strip().lower() or "binance"
    symbol = str(execution.get("symbol") or "").strip().upper()
    side = str(execution.get("side") or "").strip().lower()
    source = str(execution.get("source") or "strategy").strip().lower() or "strategy"
    quantity = _as_float_or_none(execution.get("quantity"))
    price = _as_float_or_none(execution.get("price"))
    quote_quantity = _as_float_or_none(execution.get("quote_quantity"))
    fee_amount = _as_float_or_none(execution.get("fee_amount"))
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    executed_at_ms = int(
        execution.get("executed_at_ms")
        or _parse_timestamp_to_ms(execution.get("executed_at"))
        or now_ms
    )
    executed_at_text = str(execution.get("executed_at") or "").strip() or datetime.fromtimestamp(
        executed_at_ms / 1000,
        tz=timezone.utc,
    ).strftime(TIMESTAMP_FORMAT)
    exchange_execution_key = str(execution.get("exchange_execution_key") or "").strip() or None
    fee_asset = str(execution.get("fee_asset") or "").strip().upper() or None

    if not execution_id or not swing_id:
        raise ValueError("Spot Swing execution requires stable execution and Swing IDs.")
    if not symbol:
        raise ValueError("Spot Swing execution requires a market symbol.")
    if side not in SWING_SIDES:
        raise ValueError("Spot Swing execution side must be buy or sell.")
    if source not in SWING_SOURCES:
        raise ValueError("Spot Swing execution source must be manual or strategy.")
    if quantity is None or quantity <= 0 or price is None or price <= 0:
        raise ValueError("Spot Swing execution quantity and price must be greater than zero.")
    if quote_quantity is not None and quote_quantity <= 0:
        raise ValueError("Spot Swing execution quote quantity must be greater than zero.")
    if fee_amount is not None and fee_amount < 0:
        raise ValueError("Spot Swing execution fee cannot be negative.")
    if fee_amount and not fee_asset:
        raise ValueError("Spot Swing execution fee asset is required when a fee is recorded.")

    with _connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO spot_swing_executions (
                execution_id, swing_id, spot_order_intent_id, venue, symbol, side,
                quantity, price, quote_quantity, fee_amount, fee_asset,
                exchange_order_id, exchange_trade_id, exchange_execution_key,
                source, reason_text, reason_json, executed_at_ms, executed_at_text,
                payload_json, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                swing_id,
                str(execution.get("spot_order_intent_id") or "").strip() or None,
                venue,
                symbol,
                side,
                quantity,
                price,
                quote_quantity,
                fee_amount,
                fee_asset,
                str(execution.get("exchange_order_id") or "").strip() or None,
                str(execution.get("exchange_trade_id") or "").strip() or None,
                exchange_execution_key,
                source,
                str(execution.get("reason_text") or "").strip() or None,
                _json_text(execution.get("reason") if isinstance(execution.get("reason"), dict) else {}),
                executed_at_ms,
                executed_at_text,
                _json_text(execution),
                now_ms,
            ),
        )
        if cursor.rowcount == 1:
            _refresh_spot_swing_lifecycle(connection, swing_id)
            row = connection.execute(
                "SELECT * FROM spot_swing_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            replay = False
        else:
            row = connection.execute(
                """
                SELECT * FROM spot_swing_executions
                WHERE execution_id = ? OR (? IS NOT NULL AND exchange_execution_key = ?)
                LIMIT 1
                """,
                (execution_id, exchange_execution_key, exchange_execution_key),
            ).fetchone()
            replay = True
        if row is None:
            raise RuntimeDbError("Spot Swing execution could not be persisted or recovered.")
        persisted = _spot_swing_execution_from_row(row)
        comparable = {
            "swing_id": swing_id,
            "venue": venue,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "quote_quantity": quote_quantity,
            "fee_amount": fee_amount,
            "fee_asset": fee_asset,
        }
        if any(persisted[key] != value for key, value in comparable.items()):
            raise ValueError("Duplicate Spot execution identity conflicts with the persisted execution.")
    return {**persisted, "idempotent_replay": replay}


def apply_spot_swing_target_ratchet(
    swing_id: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Apply one finalized ACCUMULATE_ASSET gain to Target exactly once."""
    normalized_id = str(swing_id or "").strip()
    if not normalized_id:
        raise ValueError("Spot Swing ID is required for Target settlement.")
    with _connection(path) as connection:
        existing = connection.execute(
            "SELECT * FROM spot_swing_target_ratchets WHERE swing_id = ?",
            (normalized_id,),
        ).fetchone()
        if existing is not None:
            return {
                "swing_id": str(existing["swing_id"]),
                "account_key": str(existing["account_key"]),
                "asset_symbol": str(existing["asset_symbol"]),
                "quote_symbol": str(existing["quote_symbol"]),
                "previous_target_quantity": float(existing["previous_target_quantity"]),
                "applied_gain_quantity": float(existing["applied_gain_quantity"]),
                "next_target_quantity": float(existing["next_target_quantity"]),
                "applied_at_ms": int(existing["applied_at_ms"]),
                "idempotent_replay": True,
            }
        swing_row = connection.execute(
            "SELECT * FROM spot_swings WHERE swing_id = ?",
            (normalized_id,),
        ).fetchone()
        if swing_row is None:
            raise ValueError("Spot Swing was not found for Target settlement.")
        execution_rows = connection.execute(
            """
            SELECT * FROM spot_swing_executions
            WHERE swing_id = ? ORDER BY executed_at_ms ASC, execution_id ASC
            """,
            (normalized_id,),
        ).fetchall()
        swing = _spot_swing_from_row(swing_row)
        economics = calculate_swing_economics(
            swing,
            [_spot_swing_execution_from_row(row) for row in execution_rows],
        )
        if swing.get("trading_objective") != "accumulate_asset" or economics.get("status") != "closed":
            return None
        policy_row = connection.execute(
            """
            SELECT * FROM portfolio_asset_policies
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ?
            """,
            (swing["account_key"], swing["asset_symbol"], swing["quote_symbol"]),
        ).fetchone()
        if policy_row is None:
            raise ValueError("Spot Swing Target policy was not found for settlement.")
        proposal = calculate_target_ratchet(
            swing,
            economics,
            target_quantity=float(policy_row["target_quantity"]),
            minimum_holding_pct=float(policy_row["minimum_holding_pct"]),
        )
        applied_gain = float(proposal["applied_gain_quantity"])
        if applied_gain <= 0:
            return None
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        connection.execute(
            """
            UPDATE portfolio_asset_policies
            SET target_quantity = ?, updated_at_ms = ?
            WHERE account_key = ? AND asset_symbol = ? AND quote_symbol = ?
            """,
            (
                proposal["next_target_quantity"],
                now_ms,
                swing["account_key"],
                swing["asset_symbol"],
                swing["quote_symbol"],
            ),
        )
        connection.execute(
            """
            INSERT INTO spot_swing_target_ratchets (
                swing_id, account_key, asset_symbol, quote_symbol,
                previous_target_quantity, applied_gain_quantity,
                next_target_quantity, applied_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_id,
                swing["account_key"],
                swing["asset_symbol"],
                swing["quote_symbol"],
                proposal["previous_target_quantity"],
                applied_gain,
                proposal["next_target_quantity"],
                now_ms,
            ),
        )
    return {
        "swing_id": normalized_id,
        "account_key": swing["account_key"],
        "asset_symbol": swing["asset_symbol"],
        "quote_symbol": swing["quote_symbol"],
        "previous_target_quantity": proposal["previous_target_quantity"],
        "applied_gain_quantity": applied_gain,
        "next_target_quantity": proposal["next_target_quantity"],
        "applied_at_ms": now_ms,
        "idempotent_replay": False,
    }


def upsert_portfolio_daily_snapshot(
    snapshot: dict[str, Any],
    *,
    account_key: str = "manual_spot",
    path: Path | None = None,
) -> dict[str, Any]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    now = datetime.now(timezone.utc)
    snapshot_date = str(snapshot.get("snapshot_date") or now.date().isoformat()).strip()
    captured_at_ms = int(snapshot.get("captured_at_ms") or now.timestamp() * 1000)
    holdings = snapshot.get("holdings") if isinstance(snapshot.get("holdings"), list) else []
    payload = {
        **snapshot,
        "account_key": normalized_account,
        "snapshot_date": snapshot_date,
        "captured_at_ms": captured_at_ms,
        "holdings": holdings,
    }
    total_value = _as_float_or_none(payload.get("total_value")) or 0.0
    invested_capital = _as_float_or_none(payload.get("invested_capital")) or 0.0
    unrealized_pnl = _as_float_or_none(payload.get("unrealized_pnl")) or 0.0
    dry_powder = _as_float_or_none(payload.get("dry_powder")) or 0.0

    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO portfolio_daily_snapshots (
                account_key,
                snapshot_date,
                captured_at_ms,
                total_value,
                invested_capital,
                unrealized_pnl,
                dry_powder,
                holdings_json,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_key, snapshot_date) DO UPDATE SET
                captured_at_ms = excluded.captured_at_ms,
                total_value = excluded.total_value,
                invested_capital = excluded.invested_capital,
                unrealized_pnl = excluded.unrealized_pnl,
                dry_powder = excluded.dry_powder,
                holdings_json = excluded.holdings_json,
                payload_json = excluded.payload_json
            """,
            (
                normalized_account,
                snapshot_date,
                captured_at_ms,
                total_value,
                invested_capital,
                unrealized_pnl,
                dry_powder,
                _json_text(holdings),
                _json_text(payload),
            ),
        )

    return {
        **payload,
        "total_value": total_value,
        "invested_capital": invested_capital,
        "unrealized_pnl": unrealized_pnl,
        "dry_powder": dry_powder,
    }


def list_portfolio_daily_snapshots(
    *,
    account_key: str = "manual_spot",
    limit: int = 180,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    normalized_account = str(account_key or "manual_spot").strip() or "manual_spot"
    normalized_limit = max(1, min(int(limit), 730))
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT
                snapshot_date,
                captured_at_ms,
                total_value,
                invested_capital,
                unrealized_pnl,
                dry_powder,
                holdings_json,
                payload_json
            FROM portfolio_daily_snapshots
            WHERE account_key = ?
            ORDER BY captured_at_ms DESC
            LIMIT ?
            """,
            (normalized_account, normalized_limit),
        ).fetchall()

    snapshots: list[dict[str, Any]] = []
    for row in reversed(rows):
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            payload = {}
        holdings = json.loads(row["holdings_json"])
        payload.update(
            {
                "account_key": normalized_account,
                "snapshot_date": str(row["snapshot_date"]),
                "captured_at_ms": int(row["captured_at_ms"]),
                "total_value": float(row["total_value"]),
                "invested_capital": float(row["invested_capital"]),
                "unrealized_pnl": float(row["unrealized_pnl"]),
                "dry_powder": float(row["dry_powder"]),
                "holdings": holdings if isinstance(holdings, list) else [],
            }
        )
        snapshots.append(payload)
    return snapshots
