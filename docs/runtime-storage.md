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
- `13`

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
- `spot_order_status_events`
- `spot_swings`
- `spot_swing_executions`
- `spot_signal_snapshots`
- `spot_strategy_campaigns`
- `spot_strategy_actions`
- `spot_swing_target_ratchets`

`portfolio_transactions` remains the accounting source of truth for Treasury. `ui_log_entries` is a structured,
filterable Ledger projection spanning both Futures trade events and Treasury events; Treasury transaction entries are
reconciled idempotently by their transaction IDs.

`spot_swings` and `spot_swing_executions` are a separate trading-lifecycle projection. They preserve independent
Swing economics and exchange execution identity, but they do not replace `portfolio_transactions` or participate
directly in the holdings/cost-basis projection. Existing Treasury transactions are not backfilled into Swings.

The per-asset Asset Ledger is a read-only chronological projection over both domains. Its `All` view interleaves
Treasury accounting entries with compact Swing lifecycle entries; `Open` and `Closed` filter Swing lifecycle state.
Expanding a Swing exposes its own executions, fees, remaining quantity, and realized/unrealized economics. Building
this projection does not create Swings, submit orders, settle fills, or mutate portfolio accounting.

A Swing objective is explicitly `accumulate_cash`, `accumulate_asset`, or unset. Closed Swing economics expose both
net quote cash flow and net asset change. A positive net asset gain from a closed `accumulate_asset` Swing can produce
an upward-only Target Holding ratchet proposal. The authoritative fill settlement applies that ratchet atomically and
exactly once after full closure. Normal portfolio transactions, custody movements, and current balance changes never
derive or rewrite Target Holding.

## Spot Execution Boundaries

- Swing logic works with economic quantities and does not apply Binance filters.
- The authoritative Spot executor owns `stepSize`, `tickSize`, `minQty`, `minNotional`, and other venue constraints.
- Swing accounting consumes actual Binance fill quantity and price, never requested or pre-quantized values.
- Manual and strategy intents converge on the same authoritative executor. Public Control Plane previews are always
  manual; strategy intents must carry a valid `swing_id` and an explainable reason before they can be submitted.
- Intent quantities remain economic requests. Immediately before submission the executor rounds quantity down to the
  Binance `stepSize`, then validates `minQty`, `maxQty`, notional rules, current balances, and Protected Floor.
- Fees retain their original `fee_amount` and `fee_asset`. Fees paid in BNB or another third asset remain unpriced until
  a future analytics layer can value them from historical market data.
- Every execution transition is persisted in `spot_order_status_events`: previewed, queued/processing, validated,
  submitted, accepted, fill confirmed, accounting updated, and final/failed states. Stable Binance client order IDs
  prevent duplicate submission, while confirmed fills can retry local Treasury/Swing settlement without resubmission.
- `portfolio_transactions` receives one idempotent transaction for every confirmed order and remains the holdings/cost
  basis source of truth. A linked Swing separately receives the actual Binance fills, preserving exchange trade identity
  and fees; replay cannot double-count either projection.
- Runtime startup only resumes previously confirmed/in-flight intents. Recovery does not create Swings or new
  automatic orders.
- A closed `accumulate_asset` Swing may produce a Target ratchet only after its net asset gain is final. Authoritative
  settlement applies it atomically and idempotently exactly once. Partial closes never ratchet Target, and a losing
  Swing never lowers it.

## Treasury Policy Mode

`SCROOGE_SPOT_EXECUTION_ENABLED` is the only Spot execution mode switch. Treasury does not introduce a second global
switch or a per-asset auto-trading toggle.

- With execution disabled, every holding is `locked`, immediate sellable inventory is zero, and trading policy/order
  controls are omitted from the UI. Stored Target and Minimum Holding values remain unchanged.
- With execution enabled, a managed asset is `locked` at 100% Minimum Holding and `unlocked` below 100%.
- Dry Powder is not policy-managed and remains `locked`.

## Rolling Spot Signal Boundaries

- When Spot execution is enabled, the bot samples Binance rolling 24-hour tickers on a configurable interval and
  persists the latest explainable signal per managed asset in `spot_signal_snapshots`.
- The primary signal is `current price / approximately-24h reference price - 1`. It is independent of UTC midnight.
- Default absolute movement levels are `5,8,12,18%`, with base tranches `10,20,30,40%`. They can be overridden through
  `SCROOGE_SPOT_SIGNAL_LEVELS_PCT` and `SCROOGE_SPOT_SIGNAL_BASE_TRANCHES_PCT`; both lists must remain aligned.
- A move below Level 1 is `HOLD`; positive qualifying moves are `SELL` opportunities and negative qualifying moves are
  `BUY` opportunities.
- Market opportunity and strategy eligibility are stored separately. Missing Trading Objective, a fully protected
  policy, disabled execution, or unavailable market data cannot become an eligible strategy action.
- The signal monitor does not create a Swing, create an order intent, submit an order, or mutate Treasury accounting.
  Those remain later strategy/execution phases.

## Spot Indicator Sizing Boundaries

- Indicators are evaluated only after rolling 24-hour movement has produced `BUY` or `SELL`. `HOLD` does not fetch
  indicator candles and always keeps a zero final tranche.
- Spot uses its own strictly closed-candle context instead of coupling to Futures execution: RSI 11, EMA 50,
  Bollinger 20/2, and ATR 14 on Binance Spot candles configured by `SCROOGE_SPOT_INDICATOR_INTERVAL` (`1h` by default).
- RSI, Bollinger, and EMA provide directional sizing evidence. ATR is preserved as volatility context only and cannot
  create or reverse an opportunity.
- Initial sizing tiers are weak `0.5x`, neutral `1.0x`, strong `1.25x`, and very strong `1.5x`; the ordered modifiers
  are configurable with `SCROOGE_SPOT_INDICATOR_SIZING_MODIFIERS`.
- Three confirmations without conflicts are very strong; at least two are strong; conflicting evidence that dominates
  confirmations is weak; remaining mixed or partial context is neutral.
- Missing indicator data applies the conservative weak modifier while preserving the rolling signal. The context,
  confirmations, conflicts, tier, modifier, and final tranche percentage remain in the persisted signal snapshot.
- Indicator sizing still does not create Swings, order intents, or orders. Portfolio and exchange limits remain a later
  progressive execution concern.

## Progressive Spot Swing Execution

- An eligible rolling opportunity opens at most one independent Swing for each newly reached level in the current
  directional campaign. HOLD or a direction reversal starts a new campaign; completed levels survive restarts.
- The final indicator-sized tranche is converted to an economic quantity first. The unified Spot executor remains
  authoritative for Binance lot size, notional, current balances, and Protected Floor validation.
- Existing Swings are evaluated independently from their own weighted opening execution price. The default profitable
  close threshold is `5%` (`SCROOGE_SPOT_SWING_CLOSE_PROFIT_PCT`), and profitable closes take priority over new exposure.
- At most one strategy action per asset is submitted in a signal cycle. Durable action keys and existing client order
  recovery prevent restarts or retries from creating a second real order for the same decision.
- New strategy exposure opens only from SELL opportunities. BUY opportunities close existing SELL-origin Swings but do
  not create BUY-origin Swings. `accumulate_cash` restores the Swing quantity and leaves profit in shared quote cash.
  `accumulate_asset` reuses profitable sale proceeds to buy back more asset. Its finalized positive asset gain
  ratchets Target exactly once after the Swing is fully CLOSED.
- `SCROOGE_SPOT_ESTIMATED_FEE_RATE` is used only for conservative strategy sizing. Actual Swing and portfolio accounting
  always use confirmed Binance fills and their native fee amount/asset.

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
