# Spot Treasury Backtesting

The Spot research engine replays the production Treasury domain over an isolated historical portfolio. It never submits orders or reads mutable production state during a run.

## What Is Shared With Live

Live and research call the same implementations for:

- rolling 24-hour levels and tranches in `core/spot_signal.py`;
- Policy eligibility and action selection in `core/spot_strategy.py`;
- campaign capacity and profitable-close planning in `core/spot_progression.py`;
- Bargain economics and Target proposals in `core/spot_swing.py`;
- waiter/capacity cleanup in `core/spot_waiter_cleanup.py`;
- Binance quantity and notional rules in `core/spot_execution_rules.py`.

Replay replaces the clock, market source, executor, persistence adapter, and reporting. The documented live baseline is `2/3/4/5%` signal levels and a `10%` Bargain Goal. A scenario or sweep may deliberately override those values; research files are not live configuration.

## Market And Fill Model

- Source: Binance Spot klines from `/api/v3/klines`, never Futures candles.
- Default interval: `1m`.
- Time: all scenario boundaries and candles are UTC.
- Warm-up: at least 24 hours before the requested start; warm-up cannot trade or mutate balances.
- Decision: after candle N closes, using only observations known by that close.
- Fill: candle N+1 open, adjusted by configured slippage and fee.
- Rolling reference: the close approximately 24 hours before the decision.
- Entry cost: first candle open at scenario start; legacy scenario `entry_cost` does not control replay cost basis.
- End: open Bargains remain marked to market unless the analysis-only `force_close_at_end` option is enabled.

All assets share one USDT pool. Asset batches beginning with a close execute before batches with only openings; ties preserve deterministic policy order. Within one asset, profitable closes run first, cleanup closes run oldest-first, and the current campaign action runs last. State and available cash are refreshed after every fill.

### Missing Candles

Asset-specific gaps abort replay because filling them would invent relative price behavior. A gap that is identical across every configured market is treated as an exchange-wide outage: candles are flat-filled for valuation continuity and marked unavailable, so the repaired interval cannot signal or trade. A common gap at a boundary that cannot be valued still aborts.

This distinction keeps a known market-wide Binance outage from destroying a multi-year run without hiding missing data for one coin.

## Execution And Accounting

The simulator applies current cached Binance symbol filters because historical filter versions are generally unavailable. It quantizes orders, checks minimum quantity/notional, charges the configured fee, and applies configured slippage. Opening also checks that the projected target close should remain exchange-valid.

SELL-origin closes may use the Bargain's committed quote plus the reserve allowed by the scenario. A terminal close finalizes restored inventory and any remaining deficit; it does not leave a synthetic dust position open forever.

`free_cash_retention_pct` is an accrual policy. It protects that share of eligible positive `accumulate_cash` settlement gain, not that share of the entire current reserve. `accumulate_asset` gain is finalized in asset units and ratchets Target exactly once.

The V1 simulator charges its configured fee in USDT. Core accounting preserves native third-asset fee data when supplied, but replay does not invent historical BNB conversion rates.

## Export And Run

Export current Treasury into a static scenario:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --export-current runtime/current-treasury-spot.yaml \
  --start 2025-10-07 --end 2026-10-07
```

Review quantities, cost bases, custody, policies, reserve, and objectives in the generated YAML. The replay will not consult the production database again.

Run an explicit period:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml \
  --start 2025-10-07 --end 2026-10-07
```

Run relative presets:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml --preset 6m --end 2026-10-07

./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml --preset 1y --end 2026-10-07
```

## Parameter Sweeps

`backtest.spot_sweep` expands a Cartesian grid over signal levels, Bargain Goal, cash retention, and deep-loss parameters. Market data is loaded once and shared; each variant receives isolated artifacts. Completed matching variants resume automatically.

```bash
./scrooge-env/bin/python -m backtest.spot_sweep \
  --config runtime/spot_backtests/sweeps/treasury-profit-retention-refinement-3y.yaml
```

Useful controls:

- `--dry-run` validates and lists the matrix without loading candles.
- `--force` deliberately reruns completed variants.
- `market_data_workers` parallelizes independent asset loading.
- `replay_parallel` and `replay_max_workers` parallelize variants, not actions inside one portfolio.

Ordinary sweep ranking first maximizes Edge vs HODL and then the unweighted average effective asset quantity. The second metric is intentionally nominal and equal-weighted across assets; volatile dollar prices do not give one coin more influence.

## Cross-Regime Sweeps

Use `backtest.spot_regime_selection` to discover auditable, non-overlapping Bull, Neutral, and Bear windows from HODL behavior. Freeze those periods before inspecting strategy results. Then run:

```bash
./scrooge-env/bin/python -m backtest.spot_regime_sweep \
  --config runtime/spot_backtests/sweeps/treasury-profit-retention-regimes-1y.yaml
```

Every candidate is replayed over all three frozen regimes. Ranking uses:

1. mean Edge vs HODL across regimes;
2. worst-regime Edge vs HODL;
3. number of regimes beating HODL;
4. worst-regime effective asset quantity;
5. mean fee drag.

The report labels a candidate robust only when its worst regime has non-negative Edge. This is a model-selection aid, not proof of future performance. Do not repeatedly redefine regimes around a preferred result.

## Artifacts

A standalone replay writes:

- `scenario.resolved.json` and `scenario.resolved.yaml`;
- `summary.json`, `report.md`, and self-contained `report.html`;
- `equity.csv`, `monthly.csv`, and `per_asset_summary.json`;
- `swings.json`, `executions.csv`, `signals.csv`, and `actions.csv`;
- `sell_campaigns.json`, `sell_campaigns.csv`, `inventory.csv`, and `target_history.csv`;
- `waiter_cleanup.json`, `waiter_cleanup_reasons.csv`, `rejections.csv`, and `final_state.json`.

A sweep additionally writes resumable `manifest.json` plus `comparison.json`, `comparison.csv`, Markdown, and HTML reports. Regime sweeps preserve per-candidate, per-regime directories and a combined summary.

Rebuild HTML without replaying:

```bash
./scrooge-env/bin/python -m backtest.spot_report_html \
  runtime/spot_backtests/runs/<run-id>
```

## Realism And Interpretation

The backtest is designed to be close enough for Spot swing parameter selection, not to reconstruct an exchange order book.

- A one-minute decision and next-open fill can differ from a live rolling-ticker order by several percent on a fast swing.
- Slippage is fixed by scenario, while real spread, impact, latency, and partial liquidity vary.
- Current exchange filters stand in for historical filters.
- Balances and fills are deterministic; live reconciliation and network uncertainty are absent.
- Market-wide gaps are made non-tradable, but surrounding candle OHLC still cannot describe intraminute path.
- Fees in a third asset cannot be valued exactly without another historical market.
- HODL is calculated from the same starting portfolio and period; it is not a cash-only benchmark.
- Portfolio drawdown is dominated by mark-to-market asset prices and should not be treated as a direct measure of platform correctness.

For this project's Spot swing use case, a few percent of execution error on an individual swing is an accepted approximation. Promote a parameter only after it remains sensible across untouched periods, frozen regimes, fees/slippage perturbations, and neighboring parameter values.
