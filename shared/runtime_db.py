from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_FILENAME = "scrooge.sqlite3"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
STATE_SNAPSHOT_KEY = "current"


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


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
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
            sort_ts_ms INTEGER,
            ts_text TEXT,
            line_text TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ui_log_entries_sort
        ON ui_log_entries(sort_ts_ms DESC, id DESC);
        """
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


def append_ui_log_entry(ts: str, line: str, path: Path | None = None) -> None:
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO ui_log_entries (
                sort_ts_ms,
                ts_text,
                line_text,
                created_at_ms
            ) VALUES (?, ?, ?, ?)
            """,
            _ui_log_record(ts, line),
        )


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
