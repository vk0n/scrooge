# Shared Domain

`shared/` contains deterministic Spot Treasury and persistence contracts used by live runtime, API projections, and backtests.

- `spot_signal.py` - rolling 24-hour levels and tranche selection.
- `spot_policy.py` - Target, Minimum Holding, objectives, and sellable inventory.
- `spot_strategy.py` - eligibility and ordered action planning.
- `spot_progression.py` - campaign capacity, opening size, and Bargain Goal.
- `spot_waiter_cleanup.py` - deep-loss, aging, and capacity relief.
- `spot_swing.py` - Bargain lifecycle, fills, PnL, retention, and Target proposals.
- `spot_execution_rules.py` - exchange quantity/notional validation.
- `spot_accounting.py` - confirmed-fill asset and quote transaction legs.
- `treasury_strategy_config.py` - strict `treasury` YAML schema.
- `treasury_ledger.py` - structured role-styled Ledger messages.
- `runtime_db.py` - SQLite schema, migrations, and atomic repositories.

Shared modules must not submit network requests or depend on frontend state. Exchange I/O belongs in `bot/`; HTTP orchestration belongs in `api/`; historical adapters belong in `backtest/`.

See [Spot Treasury](../docs/spot-treasury.md) and [Runtime Storage](../docs/runtime-storage.md).
