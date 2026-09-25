# Scrooge Spot: strategy brief for Argo

Snapshot date: 25 September 2026.

## V1 architecture

Scrooge now separates two concepts that were previously mixed:

- **Bargain Swing trading** is SELL-origin only. Every Bargain has independent opening economics, fills, age, cleanup, settlement, and PnL.
- **Treasury accumulation** is a standalone BUY of an `ACCUMULATE_ASSET` holding from Free Vault Reserve. It has no profit target, close, waiter, or Bargain lifecycle.

| Objective | BUY signal | SELL signal |
|---|---|---|
| `ACCUMULATE_CASH` | Advance/reset BUY campaign only; no order | Open an independent SELL-origin Bargain |
| `ACCUMULATE_ASSET` | Buy asset from Free Vault Reserve; no Bargain | Open an independent SELL-origin Bargain |

No new strategy path can create a BUY-origin Bargain. Historical BUY-origin records remain readable for old reports.

## Signal, campaign, and priority

The primary signal remains the rolling 24-hour price change. Levels are **5% / 8% / 12% / 18%**, with fixed allocations **10% / 20% / 30% / 40%**. RSI, Bollinger Bands, EMA, ATR, and legacy conviction tiers are research telemetry only and do not affect execution quantity.

HOLD preserves the current campaign. A same-side signal continues it, while an opposite actionable L1+ signal starts a new campaign. For `ACCUMULATE_CASH`, non-trading BUY levels are consumed immediately. For `ACCUMULATE_ASSET`, a BUY level is consumed only after a meaningful confirmed execution; a zero-reserve, dust, filter, or min-notional rejection leaves it available for a later retry.

Per asset and cycle, priority remains:

1. Normal profitable Bargain close.
2. Waiter cleanup or forced settlement.
3. New SELL Bargain or Treasury accumulation action.
4. HOLD.

Thus a BUY signal that cleans up an old SELL-origin waiter does not also accumulate asset in the same cycle.

## Reserve and accounting

Treasury accumulation sizing uses the fixed signal-level allocation against the current reserve:

```text
Free Vault Reserve = managed USDT - quote committed to open SELL Bargains
quote_to_spend = current Free Vault Reserve * level_allocation_pct
```

The next level uses the newly reduced reserve, so deployment naturally diminishes. Submission is revalidated against current managed free reserve, actual Binance free USDT, policy objective, and Binance filters. Live accumulation has an additional `SCROOGE_SPOT_TREASURY_ACCUMULATION_ENABLED` gate that defaults OFF; research scenarios enable it explicitly.

A confirmed accumulation fill creates one normal `portfolio_transaction`, reduces USDT, and increases Binance asset inventory. It does not create `spot_swings`. Target increases exactly once by **net acquired base asset after fees**. Quote-asset fees reduce cash; third-asset fees stay in their native asset. The existing successful SELL-origin Bargain ratchet and the new reserve-deployment ratchet are stored and reported separately. Neither mechanism ratchets Target downward.

Stable strategy action identity, Binance client order ID, idempotent portfolio transaction ID, and the unique accumulation-ratchet row prevent retry/restart from submitting, spending, or ratcheting twice.

## Experiment

Both horizons use identical Treasury quantities, policies, candles, 1h interval, 60-candle warmup, 0.1% fee, zero configured slippage, signal thresholds, tranche sizes, indicator modifiers, and cleanup rules. The comparison changes only BUY architecture:

- **Baseline:** corrected campaign logic with BUY- and SELL-origin Bargains.
- **V1:** no BUY-origin Bargains; cash BUY is campaign-only; asset BUY is Treasury accumulation.

### Portfolio comparison

| Metric | 6m baseline | 6m V1 | 1y baseline | 1y V1 |
|---|---:|---:|---:|---:|
| Final Treasury Value | $16,405.13 | **$16,439.31** | $16,835.87 | **$17,133.93** |
| Difference vs HODL | -$397.27 | **-$363.09** | +$33.47 | **+$331.53** |
| Maximum drawdown | -35.12% | **-33.97%** | -70.77% | **-69.87%** |
| Final Vault Reserve | $488.59 | $1,341.87 | $546.83 | $1,359.06 |
| Final Free Reserve | $488.59 | $0.00 | $546.83 | $0.00 |
| Execution fees | $50.16 | **$39.03** | $122.48 | **$94.85** |
| Bargain lifecycle PnL | -$424.31 | **-$389.33** | +$106.90 | **+$675.00** |
| Bargains opened / closed / open | 422 / 386 / 36 | **313 / 287 / 26** | 898 / 861 / 37 | **674 / 648 / 26** |
| BUY-origin Bargains | 122 | **0** | 250 | **0** |
| SELL-origin lifecycle PnL | -$470.63 | **-$389.33** | **+$1,287.73** | +$675.00 |
| Accumulation buys | 0 | **3** | 0 | **89** |
| USDT deployed | $0.00 | **$22.15** | $0.00 | **$743.87** |

The V1 change improved Final Treasury by **$34.17** over the 6m baseline and by **$298.05** over the 1y baseline.

### Accumulation results

Quantities below are native asset units and must not be summed as economic value across unlike assets without prices.

| Horizon | Asset | Buys | USDT deployed | Net asset acquired | Target growth |
|---|---|---:|---:|---:|---:|
| 6m | ICP | 1 | $5.89 | 2.32 ICP | 2.32 ICP |
| 6m | TIA | 1 | $9.45 | 24.07 TIA | 24.07 TIA |
| 6m | WCT | 1 | $6.81 | 129.40 WCT | 129.40 WCT |
| 1y | ICP | 25 | $194.55 | 60.27 ICP | 60.27 ICP |
| 1y | TIA | 29 | $258.22 | 396.69 TIA | 396.69 TIA |
| 1y | WCT | 26 | $220.99 | 1,854.50 WCT | 1,854.50 WCT |
| 1y | XRP | 9 | $70.12 | 34.80 XRP | 34.80 XRP |

The 1y level attribution is L1: 40 buys / $259.43, L2: 34 / $307.78, L3: 9 / $97.45, and L4: 6 / $79.21. Conviction attribution is Weak: 1, Neutral: 14, Strong: 31, Very Strong: 43.

### Objective and asset observations

In the 6m V1 Bargain lifecycle, `ACCUMULATE_CASH` produced -$357.68 and `ACCUMULATE_ASSET` -$31.66. In the 1y V1 lifecycle, cash produced +$574.21 and asset +$100.79, before separately considering reserve-deployment inventory.

The strongest 1y marked asset contribution versus untouched starting quantity came from TIA (+$169.65), ICP (+$146.57), XRP (+$142.44), and WCT (+$84.54). Cash-objective assets remained the weak area, led by NEAR (-$801.42) and DYDX (-$602.73). Shared reserve is intentionally not attributed to one asset, so per-asset marked contributions do not sum directly to portfolio edge.

## Interpretation

The architecture change is positive on both tested paths, but not uniformly so. Removing 250 annual BUY-origin Bargains eliminated their **-$1,180.84** lifecycle result. However, annual SELL-origin lifecycle PnL also fell from **+$1,287.73** to **+$675.00**, because consuming BUY campaign levels without opening BUY Bargains changes later campaign resets and SELL opportunities. That interaction is the main unexpected result and should be investigated before tuning.

V1 still trails HODL by $363.09 over 6m. At both endpoints, Free Reserve is $0 while all remaining USDT is committed to open SELL-origin Bargains; this is safe for settlement, but leaves no immediate capital for another accumulation. The backtest is a deterministic candle replay, not an order-book, latency, market-impact, or historical-filter simulation.

Interactive reports:

- [6m V1 report](../runtime/spot_backtests/comparisons/20260925T-treasury-accumulation-v1/6m/new-strategy/report.html)
- [1y V1 report](../runtime/spot_backtests/comparisons/20260925T-treasury-accumulation-v1/1y/new-strategy/report.html)
- [6m baseline report](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/6m/buy-and-sell-current/report.html)
- [1y baseline report](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/1y/buy-and-sell-current/report.html)
