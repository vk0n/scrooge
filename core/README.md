# Scrooge Core

`core/` is the common application kernel shared by Scrooge's two live systems and their research adapters.

- Office / Futures uses the event engine, market events, feature engine, indicator inputs, event store, and retry contracts.
- Treasury / Spot uses signals, policy, strategy planning, campaign progression, Bargain lifecycle, cleanup, execution rules, accounting, and validated strategy configuration.
- Both systems use the canonical SQLite runtime contract, UTC helpers, and role-styled Ledger projections.

Core modules may be called by `bot/`, `api/`, and `backtest/`. Live exchange polling and order submission belong in `bot/`; HTTP orchestration belongs in `api/`; historical data adapters and reporting belong in `backtest/`.
