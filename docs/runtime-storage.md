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

## Spot Execution Boundaries

- Swing logic works with economic quantities and does not apply Binance filters.
- The authoritative Spot executor owns `stepSize`, `tickSize`, `minQty`, `minNotional`, and other venue constraints.
- Swing accounting consumes actual Binance fill quantity and price, never requested or pre-quantized values.
- Fees retain their original `fee_amount` and `fee_asset`. Fees paid in BNB or another third asset remain unpriced until
  a future analytics layer can value them from historical market data.
- A closed `accumulate_asset` Swing may produce a Target ratchet proposal only after its net asset gain is final. A
  future authoritative settlement layer must apply that proposal atomically and idempotently exactly once. Partial
  closes never ratchet Target, and a losing Swing never lowers it.

## Treasury Policy Mode

`SCROOGE_SPOT_EXECUTION_ENABLED` is the only Spot execution mode switch. Treasury does not introduce a second global
switch or a per-asset auto-trading toggle.

- With execution disabled, every holding is `locked`, immediate sellable inventory is zero, and trading policy/order
  controls are omitted from the UI. Stored Target and Minimum Holding values remain unchanged.
- With execution enabled, a managed asset is `locked` at 100% Minimum Holding and `unlocked` below 100%.
- Dry Powder is not policy-managed and remains `locked`.

## Treasury Custody Boundaries

- `deposit` brings an asset under Treasury jurisdiction in a selected custody location. A non-stable asset must include
  its entry cost so the accounting projection cannot create a zero-cost holding.
- `withdraw` releases an asset from Treasury jurisdiction and removes the proportional cost basis from the selected
  custody location.
- `custody_transfer` moves an already managed asset between `unassigned`, `binance`, and `cold_storage` without changing
  total quantity, cost basis, or Target Holding.
- `buy` and `sell` remain economic executions. They are not used as manual aliases for bringing assets into or releasing
  them from Treasury.
- Deposits, withdrawals, and custody movements never mutate Target Holding. A fully withdrawn asset can disappear from
  current holdings while its stored policy remains dormant for a later return.

Schema changes must be introduced through explicit migration steps in `shared/runtime_db.py`.

## Fresh Start Rule

For a new production instance, only these classes of data should be carried over:
- secrets/credentials
- canonical config
- DB schema code

Runtime artifacts are intentionally disposable.
