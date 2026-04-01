# Runtime Storage Contract

Scrooge is now DB-first.

## Source Of Truth

Canonical runtime state lives in:
- `SCROOGE_DB_PATH`
- default: `runtime/scrooge.sqlite3`

SQLite is the source of truth for:
- current runtime state snapshot
- trade history
- balance history
- Ledger/UI log lines
- event history records

File artifacts remain only where they are still useful as raw or replay-oriented outputs:
- `event_history.jsonl`
- `market_events.jsonl`
- `chart_dataset.csv`

These file artifacts are not the canonical runtime state contract.

## Bootstrap Contract

On a clean instance:
1. `/runtime` may be empty.
2. Bot/API resolve `SCROOGE_DB_PATH`.
3. Runtime DB is created if missing.
4. Schema is initialized.
5. Live bot seeds the initial runtime state snapshot if none exists.

## Schema Contract

`schema_migrations` is the authoritative schema-version table.

Current schema version:
- `1`

Current runtime tables:
- `schema_migrations`
- `runtime_state_snapshot`
- `trade_history`
- `balance_history`
- `event_history`
- `ui_log_entries`

Schema changes must be introduced through explicit migration steps in `shared/runtime_db.py`.

## Fresh Start Rule

For a new production instance, only these classes of data should be carried over:
- secrets/credentials
- canonical config
- DB schema code

Runtime artifacts are intentionally disposable.
