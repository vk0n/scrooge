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
- `10`

Current runtime tables:
- `schema_migrations`
- `runtime_state_snapshot`
- `trade_history`
- `balance_history`
- `event_history`
- `ui_log_entries`
- `portfolio_accounts`
- `portfolio_transactions`
- `portfolio_asset_policies`
- `portfolio_daily_snapshots`
- `exchange_account_snapshots`
- `exchange_asset_balances`
- `spot_order_intents`
- `spot_swings`
- `spot_swing_executions`

`portfolio_transactions` remains the accounting source of truth for Treasury. `ui_log_entries` is a structured,
filterable Ledger projection spanning both Futures trade events and Treasury events; Treasury transaction entries are
reconciled idempotently by their transaction IDs.

`spot_swings` and `spot_swing_executions` are a separate trading-lifecycle projection. They preserve independent
Swing economics and exchange execution identity, but they do not replace `portfolio_transactions` or participate
directly in the holdings/cost-basis projection. Existing Treasury transactions are not backfilled into Swings.

A Swing objective is explicitly `accumulate_cash`, `accumulate_asset`, or unset. Closed Swing economics expose both
net quote cash flow and net asset change. A positive net asset gain from a closed `accumulate_asset` Swing can produce
an upward-only Target Holding ratchet proposal; Phase 1 does not apply that proposal or mutate portfolio policy. Normal
portfolio transactions, custody movements, and current balance changes never derive or rewrite Target Holding.

Schema changes must be introduced through explicit migration steps in `shared/runtime_db.py`.

## Fresh Start Rule

For a new production instance, only these classes of data should be carried over:
- secrets/credentials
- canonical config
- DB schema code

Runtime artifacts are intentionally disposable.
