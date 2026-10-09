# Spot Treasury Contract

This document defines the intended live behavior of Scrooge Treasury. Code and persisted state remain authoritative for an individual execution; this is the human-readable operating contract.

## Domain Model

- A **Holding** is the settled Treasury quantity and cost basis for one asset.
- A **Policy** gives an asset its Target Holding, Minimum Holding, and Trading Objective.
- A **Campaign** tracks the currently active rolling-24-hour direction and which signal levels have been consumed.
- A **Bargain** is one independent SELL-origin Spot swing with its own opening fills, committed proceeds, close fills, fees, objective, and PnL.
- **Vault Reserve** is managed USDT. It is split into Spendable, Protected, and Committed cash.
- **Custody** records whether managed inventory is Unassigned, on Binance, or in Cold Storage.

Portfolio entries and exchange balances answer different questions. Settled Treasury entries define ownership, cost basis, and capital flows. Binance snapshots constrain what can actually be traded at that moment.

## Baseline Rules

| Clause | Baseline |
| --- | --- |
| Market bell | inspect each managed Spot market every `60s` |
| Signal ladder | L1-L4 at absolute rolling moves of `2% / 3% / 4% / 5%` |
| Cash campaign stakes | `10% / 20% / 30% / 40%` of frozen campaign capacity |
| Asset campaign stakes | `1% / 3% / 5% / 10%` of Spendable Vault Reserve |
| Bargain Goal | close at a `10%` gross favorable move from the Bargain basis |
| Campaign capacity | freeze `50%` of current custody-sellable Binance inventory |
| Full deployment | freeze all remaining custody-sellable inventory when at most `25%` of policy allowance remains |
| Waiter limit | at most `10` open Bargains per asset |
| Deep-loss relief | age `15d`, unrealized PnL at or below `-25%`, L1 reverse signal |
| Aging relief | age `30d` needs L3, `60d` needs L2, `90d` needs L1 |
| Capacity relief | enabled for Bargains aged at least `30d` |

The baseline belongs in the required `treasury` YAML subtree. `GET/POST /api/config/treasury-rules` reads and replaces only that subtree after strict validation. A config write reports that restart is required; it does not silently hot-swap a running strategy.

## Signal And Campaigns

The signal is the percentage move from an approximately 24-hour-old reference price to the current price. It is rolling and independent of UTC midnight.

- A move smaller than L1 is `HOLD`.
- A positive qualifying move is a `SELL` opportunity.
- A negative qualifying move is a `BUY` opportunity.
- Treasury does not use the Futures Bollinger Bands, RSI, EMA, or ATR indicators.

Each asset has a durable directional campaign. Repeated polls at an already-consumed level do not spend again. Reaching a new level applies the cumulative allocation for every newly crossed level without creating synthetic intermediate actions. `HOLD` preserves campaign state, while an actionable opposite direction starts a new campaign.

A SELL campaign freezes its capacity from inventory that is both policy-sellable and recorded in Binance custody. Cold Storage contributes to total ownership and Protected Floor economics, but never enlarges automated campaign capacity. Later Target, Minimum Holding, or custody changes affect future campaigns, not the frozen budget of the active one. Actual execution remains capped by the current Protected Floor, verified Binance free balance, and venue filters.

## Objectives

Every non-USDT policy has one of two objectives:

### Accumulate Cash

A SELL opportunity opens a Bargain by selling inventory. Its close buys back as much exchange-valid inventory as the Bargain's committed proceeds and any explicitly permitted reserve can fund. Net positive quote gain remains in Treasury cash; the configured share of eligible gain is credited to Protected Cash.

An otherwise unused BUY signal advances the campaign but does not open a BUY-origin Bargain.

### Accumulate Asset

A SELL opportunity also opens a SELL-origin Bargain. Its close reinvests proceeds into the asset. Positive finalized asset gain is measured in asset units and increases Target Holding exactly once.

An otherwise unused BUY signal may perform standalone accumulation using only Spendable Vault Reserve. The confirmed net asset acquired increases Target Holding exactly once.

Changing a policy objective updates every open Bargain for that asset. Closed Bargains retain their historical objective and economics.

## Bargain Economics

Every Bargain is valued from its own weighted opening execution price. The Bargain Goal is a gross favorable price-move threshold; fees are not added to that threshold. Internal fee estimation is used conservatively for sizing and round-trip validity checks, but is not an editable live strategy clause. Accounting uses only confirmed fills and preserves the actual fee amount and fee asset.

For `accumulate_cash`, ignoring fees, the favorable percentage move and quote PnL percentage are equivalent relative to opening notional. For `accumulate_asset`, dollar mark-to-market may be shown while open, but finalized economic gain is denominated in the asset and its percentage is relative to the Bargain's opening quantity/basis.

Profitable closes run before cleanup and new exposure. A terminal close does not leave an untradeable dust tail: restored inventory and any remaining deficit are finalized in the same lifecycle.

## Waiter Cleanup

Cleanup gives old, deeply losing, or capacity-blocking Bargains a controlled reverse-signal exit. It does not bypass the normal executor or accounting path.

The action order for one asset is deterministic:

1. Close Bargains that reached Bargain Goal.
2. Close every eligible waiter or capacity-relief Bargain, oldest first.
3. Execute the current campaign level opening or standalone accumulation.

The runtime refreshes balances, reserve, Swing state, and campaign state after every terminal fill. A retryable, non-terminal, or uncertain order stops the batch rather than risking duplicate spend.

## Policy And Target Holding

Minimum Holding protects a percentage of Target from automatic SELL campaigns. The inventory limits remain distinct:

- **Policy Sellable** is the economic surplus above the Protected Floor across all custody.
- **Custody Sellable** is the lesser of Policy Sellable and the quantity recorded in Binance custody; it initializes SELL campaign capacity.
- **Immediately Sellable** additionally applies the verified Binance free balance and is the execution-time order cap.

Target changes are intentional and durable:

- a standalone strategy accumulation increases Target by net acquired asset;
- a profitable `accumulate_asset` Bargain increases Target by finalized net asset gain;
- a manual Binance BUY increases Target by net received asset;
- a manual Binance SELL decreases Target by the total asset debit;
- deposits, withdrawals, and custody transfers do not rewrite Target;
- a Swing-linked close does not also apply the generic manual target ratchet.

Each automatic or manual ratchet has a durable identity, so retries and restarts cannot apply it twice.

## Cash And Capital

USDT has one shared accounting pool across all Spot assets:

- **Spendable** may fund automatic accumulation and explicitly requested manual actions.
- **Protected** is excluded from automatic strategy spending and loss coverage.
- **Committed** belongs to open Bargains and is not free reserve.

Protected Cash is owner-controlled. It can be moved between Protected and Spendable, transferred between Treasury and Futures Office when that feature is enabled, or explicitly authorized for a manual BUY or manual loss close. Every use or credit is persisted idempotently.

Confirmed Spot orders create both asset and quote accounting legs. A BUY debits USDT; a SELL credits USDT. Manual Binance trades change cash and Target but do not count as new external Invested Capital. Treasury deposits and withdrawals are external capital flows. This distinction keeps Total Gain from treating an internal exchange trade as a fresh contribution.

## Custody

- `deposit` brings an asset under Treasury management; a non-stable asset requires entry cost.
- `withdraw` releases managed quantity and its proportional cost basis.
- `custody_transfer` moves quantity between `unassigned`, `binance`, and `cold_storage` without changing total quantity, cost basis, or Target.
- Cold Storage contributes to Treasury value and policy floor economics but is never automatically sellable.
- `buy` and `sell` are economic executions, not aliases for custody intake or release.

## Execution Safety

Real Spot execution is disabled unless `SCROOGE_SPOT_EXECUTION_ENABLED=1` is set for both API and bot. Manual execution follows preview, explicit `CONFIRM_SPOT_ORDER`, runtime delivery, fresh-state revalidation, Binance quantization, idempotent submission, confirmed fill settlement, and Ledger recording.

The executor owns `stepSize`, `tickSize`, `minQty`, `maxQty`, and notional validation. It rounds quantities down immediately before submission and rechecks available balances and policy protection. Strategy openings also require the projected target close to remain exchange-valid.

An intent in `uncertain` or `accounting_error` state must be reconciled by Binance client order ID. Never issue a replacement merely because the API response was lost.

## Time And Presentation

Strategy timestamps, accounting boundaries, daily snapshots, and the overview's change-since-midnight values use UTC. `NEXT_PUBLIC_DISPLAY_TIMEZONE` changes only rendered timestamps. It does not change trading or accounting boundaries.

## Live And Replay

Live and backtest share the Spot domain rules, but not their adapters. Live uses rolling ticker observations, real balances, real order-book execution, and confirmed Binance fills. Replay uses historical one-minute Spot candles, next-candle-open fills, configured slippage/fees, and current cached exchange filters.

See [Spot Backtesting](spot-backtesting.md) before interpreting a sweep result as a live expectation.
