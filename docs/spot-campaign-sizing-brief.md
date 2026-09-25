# Spot SELL Campaign Capacity V1

Snapshot date: 25 September 2026.

## Architecture

Previously every SELL level used the asset's full strategic sellable reference multiplied by the indicator-adjusted final tranche. A strong early signal could therefore consume inventory intended for later levels or future excursions.

The new shared live/backtest sizing is:

```text
policy_sellable_reference = current Target * (1 - current Minimum Holding %)
remaining_sellable_ratio = current policy sellable / policy_sellable_reference

if remaining_sellable_ratio <= 25%:
    campaign_capacity = current policy sellable
else:
    campaign_capacity = current policy sellable * 50%

requested_level_quantity = frozen campaign_capacity * level_allocation
planned_quantity = min(
    requested_level_quantity,
    frozen campaign_capacity - consumed campaign quantity,
    current immediately sellable quantity,
)
```

The exactly-25% boundary uses full deploy. L1/L2/L3/L4 remain fixed at 10/20/30/40%. A direct first signal at L3 executes only L3's 30%, not L1+L2+L3. HOLD preserves the same campaign and frozen budget. An opposite actionable signal ends it; unused capacity is discarded. A later SELL campaign calculates a new reference from the then-current Target and Minimum Holding.

RSI, EMA, Bollinger Bands, ATR, conviction tier, and the legacy modifier remain recorded for research. They no longer affect SELL or Treasury accumulation quantity. A fixed asset/campaign/level therefore requests the same quantity for Weak and Very Strong telemetry.

Campaign-start Target, Minimum Holding, policy reference, remaining sellable, ratio, capacity settings, mode, frozen capacity, consumed quantity, and highest completed level are persisted in `spot_strategy_campaigns`. Actual confirmed SELL fills increment consumption through one idempotent `spot_strategy_campaign_consumptions` row per action. Schema version is 16. Retries, duplicate reconciliation, rounding, partial fills, and restart cannot enlarge or double-consume the budget. Current Protected Floor, policy sellable, Binance free balance, lot size, and notional rules remain authoritative.

Configuration defaults:

- `SCROOGE_SPOT_CAMPAIGN_CAPACITY_PCT=50`
- `SCROOGE_SPOT_FULL_DEPLOY_THRESHOLD_PCT=25`
- Signal levels `5/8/12/18%`
- Level allocations `10/20/30/40%`

## Controlled Comparison

The baseline is the previous corrected Treasury Accumulation V1. Data, portfolio, signal thresholds, profit target, cleanup, fees, slippage, and execution timing are unchanged.

| Metric | 6M previous | 6M capacity V1 | 1Y previous | 1Y capacity V1 |
|---|---:|---:|---:|---:|
| Final Treasury Value | $16,439.31 | **$16,617.22** | **$17,133.93** | $16,966.94 |
| Difference vs HODL | -$363.09 | **-$185.18** | **+$331.53** | +$164.54 |
| Maximum drawdown | **-33.97%** | -35.55% | **-69.87%** | -70.57% |
| Vault Reserve / free | $1,341.87 / $0 | $643.99 / $0 | $1,359.06 / $0 | $651.39 / $0 |
| Execution fees | $39.03 | **$16.94** | $94.85 | **$40.42** |
| Lifecycle PnL | -$389.33 | **-$194.11** | **+$675.00** | +$320.41 |
| Bargains opened / closed / open | 313 / 287 / 26 | 298 / 272 / 26 | 674 / 648 / 26 | 667 / 639 / 28 |
| Accumulation buys | 3 | 0 | 89 | 47 |
| Accumulation USDT | $22.15 | $0 | $743.87 | $317.94 |
| Target growth, native units | 155.79 | 0 | 2,346.26 | 1,057.46 |

The 6M result improved by **$177.91**, while the 1Y result declined by **$166.98** versus the previous corrected strategy. Both runs paid substantially less in fees. The new strategy still beat HODL over 1Y by **$164.54**, but trailed it over 6M by **$185.18**.

## Campaign Analytics

| Metric | 6M | 1Y |
|---|---:|---:|
| SELL campaigns | 177 | 401 |
| Average / median capacity | 517.39 / 95.20 units | 490.04 / 100.00 units |
| Average / median utilization | 33.20% / 29.84% | 30.88% / 20.00% |
| Reached L1 / L2 / L3 / L4 | 154 / 104 / 56 / 28 | 373 / 230 / 111 / 48 |
| Interrupted by opposite signal | 174 | 398 |
| Normal 50% mode | 177 | 401 |
| Full-deploy mode | **0** | **0** |

No campaign started at or below 25% of the current policy reference. Thus full-deploy did not affect either portfolio result. Every campaign snapshot, including start reference, remaining quantity, and ratio, is available in `sell_campaigns.json` and `sell_campaigns.csv` for later threshold analysis.

### SELL openings by level

Quantities are mixed native asset units and should not be compared or summed economically; notional and PnL are in USDT.

| Horizon | Level | Opens | Quantity | Notional | Lifecycle PnL |
|---|---:|---:|---:|---:|---:|
| 6M | L1 | 118 | 7,561.47 | $1,931.74 | -$53.00 |
| 6M | L2 | 98 | 10,378.55 | $2,704.60 | -$62.40 |
| 6M | L3 | 54 | 8,668.61 | $2,377.99 | -$52.42 |
| 6M | L4 | 28 | 5,926.84 | $1,787.09 | -$26.29 |
| 1Y | L1 | 298 | 16,245.33 | $5,581.93 | +$116.17 |
| 1Y | L2 | 214 | 22,255.85 | $7,264.24 | +$128.92 |
| 1Y | L3 | 107 | 13,322.90 | $4,669.09 | +$25.86 |
| 1Y | L4 | 48 | 8,325.12 | $3,039.12 | +$49.46 |

### Legacy conviction telemetry

| Horizon | Tier | Bargains | Lifecycle PnL | Return on notional |
|---|---|---:|---:|---:|
| 6M | Weak | 9 | -$1.66 | -0.98% |
| 6M | Neutral | 52 | +$36.02 | +2.92% |
| 6M | Strong | 77 | -$55.41 | -2.30% |
| 6M | Very Strong | 160 | -$173.06 | -3.47% |
| 1Y | Weak | 45 | +$58.63 | +4.81% |
| 1Y | Neutral | 140 | +$174.40 | +4.69% |
| 1Y | Strong | 153 | +$83.23 | +1.63% |
| 1Y | Very Strong | 329 | +$4.15 | +0.04% |

These cohorts are observational. Their differences cannot change sizing in this implementation.

## Inventory And Accumulation

Final 1Y SELL inventory state:

| Asset | Lifecycle PnL | Campaigns | Quantity sold | Remaining sellable | Current policy reference | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| ATOM | +$22.54 | 25 | 332.56 | 81.59 | 100.00 | 81.59% |
| DOT | $0.00 | 0 | 0.00 | 0.00 | 0.00 | N/A |
| DYDX | +$278.83 | 65 | 46,033.36 | 2,625.00 | 5,000.00 | 52.50% |
| FIL | +$38.91 | 53 | 625.89 | 70.74 | 100.00 | 70.74% |
| GRAM | +$30.04 | 34 | 834.30 | 190.00 | 200.00 | 95.00% |
| ICP | -$5.33 | 49 | 761.14 | 81.60 | 112.24 | 72.70% |
| NEAR | -$139.01 | 54 | 1,712.50 | 112.00 | 200.00 | 56.00% |
| TIA | +$31.46 | 47 | 2,380.45 | 230.76 | 329.43 | 70.05% |
| WCT | +$36.33 | 48 | 6,700.90 | 1,068.40 | 1,245.70 | 85.77% |
| XRP | +$26.64 | 26 | 768.10 | 187.94 | 204.74 | 91.79% |

The 1Y run made 47 standalone accumulation buys: ICP 13 / $83.85 / 24.26 units, TIA 16 / $113.79 / 168.20, WCT 14 / $97.69 / 852.10, and XRP 4 / $22.61 / 12.90. Net acquired quantity equals Target growth per asset. The 6M run made no accumulation purchase because no free reserve deployment passed the existing execution constraints; all final USDT was committed to open SELL Bargains.

## Validation And Interpretation

Tests cover policy-reference formulas, the 25% inclusive boundary, all capacity examples, fixed level allocation, conviction invariance, HOLD continuity, opposite reset, direct L3 behavior, Target-change freezing, current floor/exchange caps, restart persistence, duplicate consumption, cleanup priority, accumulation sizing, and live/backtest parity paths. The complete suite passes with 199 tests.

Unexpected observations:

- The 25% full-deploy mode was never reached, so its practical effect remains unmeasured.
- Almost every completed directional campaign ended through an opposite signal: 174/177 over 6M and 398/401 over 1Y.
- Smaller SELL budgets improved 6M lifecycle risk and result, but reduced annual lifecycle PnL and Treasury accumulation enough to lower the 1Y final value.
- Drawdown became modestly worse on both horizons despite lower fees and smaller campaign exposure.
- Final free reserve remained zero in both runs because all remaining USDT was committed to open SELL-origin obligations.

No parameter optimization was performed.

Interactive reports:

- [6M Campaign Capacity V1](../runtime/spot_backtests/comparisons/20260925T-campaign-capacity-v1/6m/new-strategy/report.html)
- [1Y Campaign Capacity V1](../runtime/spot_backtests/comparisons/20260925T-campaign-capacity-v1/1y/new-strategy/report.html)
- [6M previous corrected V1](../runtime/spot_backtests/comparisons/20260925T-treasury-accumulation-v1/6m/new-strategy/report.html)
- [1Y previous corrected V1](../runtime/spot_backtests/comparisons/20260925T-treasury-accumulation-v1/1y/new-strategy/report.html)
