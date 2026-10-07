# Runtime Storage Contract

Scrooge is DB-first. Canonical mutable runtime state lives in SQLite at `SCROOGE_DB_PATH`, normally `runtime/scrooge.sqlite3` locally or `/runtime/scrooge.sqlite3` in Compose.

## Ownership

SQLite is authoritative for:

- Futures runtime snapshots, trade history, balance history, chart samples, and event history;
- structured Ledger/UI entries;
- Treasury accounts, settled transactions, custody, asset policies, cash policy, and daily snapshots;
- Binance account snapshots and per-asset balances;
- Spot order intents, status transitions, Swing executions, signals, campaigns, actions, and target ratchets.

These files remain useful artifacts, but are not canonical state:

- `event_history.jsonl` - replay/debug mirror of domain events;
- `market_events.jsonl` - raw Futures market and account events;
- `chart_dataset.csv` - chart/replay data;
- generated backtest reports and CSV/JSON outputs.

Redis is a command queue and command-status channel. It is not a trading ledger and must never be used to reconstruct portfolio accounting.

## Bootstrap And Migration

On a clean instance the bot or API resolves `SCROOGE_DB_PATH`, creates the database, applies every migration, and initializes missing runtime state. `schema_migrations` is the authoritative version record.

Current schema version: **20**.

Schema changes belong in explicit, forward-safe migration steps in `shared/runtime_db.py`. Services may start against an older database and migrate it; they must not require an operator to edit tables manually.

## Tables

### Futures And Shared Runtime

- `schema_migrations`
- `runtime_state_snapshot`
- `trade_history`
- `balance_history`
- `event_history`
- `ui_log_entries`
- `strategy_chart_snapshots`

### Treasury Accounting

- `portfolio_accounts`
- `portfolio_transactions`
- `portfolio_asset_policies`
- `portfolio_cash_policies`
- `portfolio_cash_retention_uses`
- `portfolio_cash_retention_credits`
- `portfolio_daily_snapshots`

### Exchange State And Orders

- `exchange_account_snapshots`
- `exchange_asset_balances`
- `spot_order_intents`
- `spot_order_status_events`

### Spot Strategy And Bargains

- `spot_swings`
- `spot_swing_executions`
- `spot_swing_cash_retentions`
- `spot_signal_snapshots`
- `spot_strategy_campaigns`
- `spot_strategy_actions`
- `spot_strategy_campaign_consumptions`
- `spot_accumulation_target_ratchets`
- `spot_swing_target_ratchets`
- `spot_manual_target_ratchets`

## Treasury Accounting Source Of Truth

`portfolio_transactions` is the source of truth for managed quantity, cost basis, external capital flows, and custody. Holdings and overview totals are projections over settled transactions plus current prices; they are not copied from the Binance balance endpoint.

Confirmed Binance Spot trades record both economic legs:

- BUY records an asset increase and a USDT decrease;
- SELL records an asset decrease and a USDT increase.

The quote leg uses the same execution identity and is backfilled exactly once for older confirmed manual/strategy intents that predate quote accounting. Internal Binance manual trades carry `capital_effect: none`, so they alter assets and reserve without inflating Invested Capital. Deposits and withdrawals remain external capital flows.

`ui_log_entries` is a structured, filterable projection over Futures and Treasury events. Treasury entries are reconciled by stable transaction or source references, preventing duplicate Ledger lines after a restart.

## Bargain Source Of Truth

`spot_swings` and `spot_swing_executions` preserve each Bargain's lifecycle and exchange economics. They do not replace `portfolio_transactions`: confirmed fills settle the Swing and also produce the portfolio legs required to update holdings and cash.

The global Bargains Ledger and every per-asset Asset Ledger project the same Swing records. The UI does not maintain duplicate Bargain models. Filters, sorting, pagination, and expansion are presentation concerns over the shared API representation.

Open Bargain objectives follow the current asset Policy. Closed Bargains preserve their historical objective. Realized fields expose quote cash gain and net asset change so `accumulate_cash` and `accumulate_asset` can be displayed in their economically meaningful units.

## Target Ratchets

Target Holding changes are stored as separate idempotent projections:

- `spot_accumulation_target_ratchets` for standalone strategy accumulation;
- `spot_swing_target_ratchets` for finalized `accumulate_asset` Bargains;
- `spot_manual_target_ratchets` for confirmed standalone manual Binance BUY/SELL orders.

Manual BUY increases Target by net asset received. Manual SELL decreases Target by total asset debit. A Swing close and a Treasury-intake operation are excluded from the generic manual ratchet path to prevent double application. Deposits, withdrawals, adjustments, and custody transfers do not infer Target changes.

## Cash Protection

`portfolio_cash_policies.retained_quote_balance` is the current Protected Cash balance. It is an accrued amount, not a percentage applied to today's full reserve.

When an eligible profitable `accumulate_cash` Bargain closes, `spot_swing_cash_retentions` records the calculation and `portfolio_cash_retention_credits` applies the credit exactly once. Explicit owner uses and bucket transfers are recorded in `portfolio_cash_retention_uses` and credit records with stable references.

Automatic strategy actions cannot spend Protected Cash. Manual BUYs, manual loss closes, and Office transfers can use it only when the operator explicitly authorizes that path.

## Spot Execution Boundaries

- Swing and policy logic work in economic quantities and never pretend to apply Binance filters.
- The bot-side executor owns `stepSize`, `tickSize`, quantity/notional bounds, current balances, and final policy protection.
- Manual and strategy requests converge on the same executor and settlement path.
- Intent quantity is a request; accounting quantity and price always come from confirmed fills.
- Native fee amount and fee asset are preserved. Third-asset fees such as BNB remain unpriced unless reliable historical conversion data exists.
- Every intent transition is appended to `spot_order_status_events`.
- Stable intent, action, execution, and client-order identities make retries idempotent.

If submission outcome is ambiguous, the intent remains `uncertain`. If the exchange fill is known but local settlement fails, it remains `accounting_error`. Both states require reconciliation by client order ID before any replacement submission.

## Exchange Snapshots

The live bot periodically stores Binance account snapshots and free/locked balances. These snapshots constrain execution and expose custody consistency warnings. They do not overwrite Treasury ownership or cost basis.

A stale Spot balance snapshot blocks API preview/execution paths that cannot be validated safely. The freshness window is configured with `SCROOGE_SPOT_BALANCE_STALE_AFTER_SECONDS`.

## UTC Contract

Persisted timestamps are Unix milliseconds in UTC. Portfolio daily snapshots and change-since-midnight metrics use UTC day boundaries. Display timezone configuration is presentation-only.

## Backup And Reset

Back up at least the SQLite database and mounted YAML config before a production migration. The named `scrooge_runtime` volume survives normal container recreation.

`docker compose down -v` deletes the runtime volume and therefore the canonical database. Use it only for an intentional fresh start. A fresh production instance should carry secrets and reviewed config, not stale generated runtime artifacts.
