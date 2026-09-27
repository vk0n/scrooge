# Spot Research Backtests

Phase H replays the production Spot strategy over an isolated historical Treasury. It does not submit orders or add live behavior.

## Shared Production Logic

Live and research both call the same implementations for:

- rolling 24-hour opportunity levels in `shared/spot_signal.py`
- indicator telemetry and HOLD invariants in `shared/spot_sizing.py`
- policy eligibility and action selection in `shared/spot_strategy.py`
- progressive opening and profitable close economics in `shared/spot_progression.py`
- Bargain accounting and Target ratchet proposals in `shared/spot_swing.py`
- Binance market quantity and notional rules in `shared/spot_execution_rules.py`

The replay replaces only the market source, clock, executor, state store, and reporting adapters.

## Timing And Data

- Data source: Binance Spot klines from `/api/v3/klines`, never Futures candles.
- Cache: deterministic CSV files under `data/spot_backtest/klines`; current exchange filters are cached as JSON.
- Timezone: all scenario and candle timestamps are UTC.
- Default interval: one minute, matching the live signal poll cadence.
- Warm-up: at least 60 hours (`3600` candles at `1m`) before the requested start. Warm-up cannot generate signals,
  orders, Bargains, or balance changes.
- Decision: candle N closes, then Scrooge evaluates only data whose close timestamp is at or before candle N close.
- Fill: every action that remains eligible is executed sequentially at candle N close, adjusted by configured slippage
  and fee. Portfolio, reserve, Swing, and campaign state are recalculated after every fill before planning the next
  action from that signal.
- Rolling reference: the candle close exactly 24 hours before candle N close.
- Missing candles: the run fails. There is no interpolation or forward fill.
- End of run: open Bargains remain open and are marked to market unless the analysis-only `force_close_at_end` flag is enabled.

This is a strategy backtest, not a Binance order-book or market-impact simulation. Current cached Binance symbol filters are used because historical filter versions are not generally available.

## Shared Cash And Ordering

All assets draw from one USDT pool. As in live execution, asset batches whose first action is a close run before batches
without a close; ties retain alphabetical policy order. Within one asset, profitable closes run first, then every
eligible waiter/capacity cleanup, then the current campaign-level opening or reserve accumulation. Every BUY is
constrained by the pool remaining after prior fills. The resolved asset order is recorded in every result. Cold Storage
contributes to value and Protected Floor economics but is never sellable.

Waiter cleanup is enabled by default. Profitable closes retain priority, then all eligible Bargains close oldest-first on
the same reverse signal. Deep loss at 15 days and -20% requires L1+; the 30/60/90-day thresholds require
L3+/L2+/L1+. A hard cap of 10 open Bargains per asset may use an eligible 30-day reverse signal for capacity cleanup,
otherwise the cycle holds instead of opening an eleventh Bargain. Cleanup uses each Bargain's remaining open economics
and the same executor, fees, quantization, and accounting path as every other close.

Opening and closing market actions are lifecycle-atomic. A SELL-origin close may spend only that Bargain's committed quote plus Free Vault Reserve and buys the maximum exchange-valid quantity available from that budget. Once the terminal market order fills, the Bargain closes without a partial or dust tail; any quantity that could not be restored is recorded as realized inventory deficit rather than left waiting for future reserve.

Strategy SELL openings also pass a round-trip exchange check. An opening is rejected when its projected profit-target BUY would fall below Binance minimum notional after the objective's quantity semantics and estimated fees; this prevents valid entries from becoming permanently uncloseable dust.

Set `strategy.waiter_cleanup.enabled: false` only for a research baseline. There is no separate production or UI toggle.

## Scenario Workflow

Export current Treasury into a static, reviewable scenario:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --export-current runtime/current-treasury-spot.yaml \
  --start 2025-09-23 --end 2026-09-23
```

The generated YAML no longer depends on the production database. Review quantities, cost bases, custody, policies, and objectives before running it.

Explicit period:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml \
  --start 2026-03-23 --end 2026-09-23
```

The CLI reports market-data loading, replay progress with elapsed time and ETA, and artifact generation. Interactive terminals receive a single updating progress bar; redirected logs receive a milestone every 5%.

Six-month preset relative to an explicit end:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml \
  --preset 6m --end 2026-09-23
```

One-year preset:

```bash
./scrooge-env/bin/python -m backtest.spot_runner \
  --config runtime/current-treasury-spot.yaml \
  --preset 1y --end 2026-09-23
```

## Artifacts

Each run writes:

- `scenario.resolved.json`
- `scenario.resolved.yaml`
- `summary.json`, `report.md`, and the self-contained visual `report.html`
- `equity.csv` and `monthly.csv`
- `per_asset_summary.json`
- `waiter_cleanup.json` and `waiter_cleanup_reasons.csv`
- `swings.json` and `executions.csv`
- `sell_campaigns.json` and `sell_campaigns.csv`
- `signals.csv` and `actions.csv`
- `inventory.csv` and `target_history.csv`
- `final_state.json` and `rejections.csv`

The report separates realized and unrealized Bargain economics, compares against the same-start HODL benchmark, and preserves third-asset fee structures if such executions are supplied. Bargain Analytics adds lifecycle PnL, closure and expectancy metrics, duration percentiles, fee drag, outcome and risk categories, an interactive cohort breakdown, and a duration-versus-return view. Each Portfolio Ledger asset row expands into filterable Bargain history, and each Bargain expands into its execution fills. The V1 simulator itself charges its configured fee in USDT.

To produce controlled six-month and one-year baseline comparisons from one scenario:

```bash
./scrooge-env/bin/python -m backtest.spot_waiter_comparison \
  --config runtime/current-treasury-spot.yaml \
  --output runtime/spot_backtests/comparisons/waiter-cleanup-defaults
```

The comparison directory contains both individual modes and `comparison.json`, `comparison.csv`, and a self-contained `comparison.html`.

To add or rebuild the visual report for an existing artifact directory without rerunning the replay:

```bash
./scrooge-env/bin/python -m backtest.spot_report_html \
  runtime/spot_backtests/runs/20260924T064219Z
```

The generated page embeds its sampled chart data and has no CDN, API, or frontend runtime dependency. Rebuilding also backfills `bargain_analysis` into an older run's `summary.json`.

## Known V1 Limits

- Live signals poll Binance's rolling ticker every minute by default; exported replay scenarios now evaluate `1m`
  candle closes and use the close exactly 24 hours earlier. Replay indicator telemetry is independently aggregated into
  strictly closed `1h` candles, matching live RSI/EMA/Bollinger inputs.
- Live orders face the real order book, latency, and changing exchange balances. Replay approximates immediate market
  execution at the observed candle close plus configured slippage, with the same sequential action ordering and
  state refresh after each fill.
- Current Binance Spot filters are used; historical filter changes are not reconstructed.
- The simulator's configured fee is charged in USDT. Shared Swing accounting still preserves third-asset fee amounts when such executions are supplied, but does not invent historical conversion rates.
- Every configured symbol must have a complete Binance Spot candle range. Missing, newly listed, or delisted markets fail the run rather than being filled or substituted.
- The exporter snapshots current state once. It never reads production state during replay, and non-USDT stable balances are excluded with a review warning rather than converted automatically.
