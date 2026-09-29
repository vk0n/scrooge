from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any, Callable

from backtest.spot_market_data import SpotCandle, SpotHistoricalDataset
from backtest.spot_scenario import SpotBacktestAsset, SpotBacktestScenario
from shared.spot_execution_rules import (
    normalize_market_quantity,
    validate_market_notional,
    validate_sell_opening_round_trip,
)
from shared.spot_policy import calculate_spot_inventory_policy
from shared.spot_progression import initialize_sell_campaign_capacity
from shared.spot_signal import ROLLING_WINDOW_MS, evaluate_rolling_24h_opportunity
from shared.spot_strategy import (
    finalize_spot_strategy_signal,
    plan_spot_strategy_action,
    transition_spot_strategy_campaign,
)
from shared.spot_swing import (
    calculate_cash_retention,
    calculate_sell_origin_committed_quote,
    calculate_swing_economics,
    calculate_target_ratchet,
)

HOUR_MS = 60 * 60 * 1000


@dataclass
class SimulatedAssetState:
    scenario: SpotBacktestAsset
    binance_quantity: float
    cold_storage_quantity: float
    unassigned_quantity: float
    target_quantity: float
    known_cost_quantity: float
    known_cost_basis: float
    realized_portfolio_pnl: float = 0.0

    @classmethod
    def from_scenario(cls, asset: SpotBacktestAsset) -> SimulatedAssetState:
        known_quantity = asset.quantity
        return cls(
            scenario=asset,
            binance_quantity=asset.binance_quantity,
            cold_storage_quantity=asset.cold_storage_quantity,
            unassigned_quantity=asset.unassigned_quantity,
            target_quantity=asset.target_holding,
            known_cost_quantity=known_quantity,
            # Backtests establish their cost basis from the replay's first candle.
            known_cost_basis=0.0,
        )

    @property
    def symbol(self) -> str:
        return self.scenario.symbol

    @property
    def quantity(self) -> float:
        return self.binance_quantity + self.cold_storage_quantity + self.unassigned_quantity

    @property
    def average_cost(self) -> float | None:
        if self.known_cost_quantity <= 1e-12:
            return None
        return self.known_cost_basis / self.known_cost_quantity

    @property
    def protected_floor(self) -> float:
        return float(self.policy_inventory["protected_floor_quantity"])

    @property
    def policy_sellable(self) -> float:
        return float(self.policy_inventory["policy_sellable_quantity"])

    @property
    def immediately_sellable(self) -> float:
        return float(self.policy_inventory["custody_sellable_quantity"])

    @property
    def policy_inventory(self) -> dict[str, float | None]:
        return calculate_spot_inventory_policy(
            current_quantity=self.quantity,
            target_quantity=self.target_quantity,
            minimum_holding_pct=self.scenario.minimum_holding_pct,
            binance_quantity=self.binance_quantity,
        )

    def unrealized_portfolio_pnl(self, market_price: float) -> float:
        return self.known_cost_quantity * market_price - self.known_cost_basis

    def rebase_cost_basis(self, initial_price: float) -> None:
        """Use the first replay candle open as the portfolio's entry basis."""
        normalized_price = float(initial_price)
        self.known_cost_basis = self.known_cost_quantity * normalized_price

    def holding(self, market_price: float) -> dict[str, Any]:
        return {
            "asset_symbol": self.symbol,
            "quote_symbol": "USDT",
            "quantity": self.quantity,
            "target_quantity": self.target_quantity,
            "minimum_holding_pct": self.scenario.minimum_holding_pct,
            "trading_objective": self.scenario.trading_objective,
            "protected_floor_quantity": self.protected_floor,
            "policy_sellable_quantity": self.policy_sellable,
            "immediately_sellable_quantity": self.immediately_sellable,
            "market_price": market_price,
        }

    def apply_buy(self, quantity: float, quote_quantity: float, fee: float) -> None:
        self.binance_quantity += quantity
        self.known_cost_quantity += quantity
        self.known_cost_basis += quote_quantity + fee

    def apply_sell(self, quantity: float, quote_quantity: float, fee: float) -> None:
        average_cost = self.average_cost
        costed_quantity = min(quantity, self.known_cost_quantity)
        removed_basis = (average_cost or 0.0) * costed_quantity
        if average_cost is not None:
            attributed_proceeds = (quote_quantity - fee) * (costed_quantity / quantity)
            self.realized_portfolio_pnl += attributed_proceeds - removed_basis
        self.known_cost_quantity = max(0.0, self.known_cost_quantity - costed_quantity)
        self.known_cost_basis = max(0.0, self.known_cost_basis - removed_basis)
        self.binance_quantity = max(0.0, self.binance_quantity - quantity)


@dataclass
class SpotBacktestResult:
    scenario: SpotBacktestScenario
    data_source: str
    starting_value: float
    initial_prices: dict[str, float]
    final_prices: dict[str, float]
    final_usdt: float
    final_retained_usdt: float
    minimum_usdt: float
    maximum_usdt: float
    assets: dict[str, SimulatedAssetState]
    equity: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    signal_level_counts: dict[str, dict[int, int]]
    actions: list[dict[str, Any]]
    swings: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    accumulations: list[dict[str, Any]]
    cash_retentions: list[dict[str, Any]]
    target_history: list[dict[str, Any]]
    inventory_history: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    sell_campaigns: list[dict[str, Any]]


class SpotPortfolioBacktester:
    """Replay the live Spot strategy over isolated historical Treasury state."""

    def __init__(self, scenario: SpotBacktestScenario, dataset: SpotHistoricalDataset) -> None:
        self.scenario = scenario
        self.dataset = dataset
        self.assets = {
            asset.symbol: SimulatedAssetState.from_scenario(asset)
            for asset in scenario.assets
        }
        self.usdt = float(scenario.starting_usdt)
        self.minimum_usdt = self.usdt
        self.maximum_usdt = self.usdt
        self.campaigns: dict[str, dict[str, Any]] = {}
        self.swings: dict[str, dict[str, Any]] = {}
        self._swing_ids_by_asset: dict[str, list[str]] = {
            symbol: [] for symbol in scenario.asset_order
        }
        self._active_swing_ids_by_asset: dict[str, list[str]] = {
            symbol: [] for symbol in scenario.asset_order
        }
        self._indexed_swing_ids: set[str] = set()
        self._economics_cache: dict[tuple[str, float | None], dict[str, Any]] = {}
        self._committed_quote_by_swing: dict[str, float] = {}
        self._committed_quote_reserve_cache: float | None = None
        self.executions: list[dict[str, Any]] = []
        self.accumulations: list[dict[str, Any]] = []
        self.cash_retentions: list[dict[str, Any]] = []
        self.retained_free_quote = 0.0
        self.actions: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.signal_level_counts: dict[str, dict[int, int]] = {
            symbol: {} for symbol in scenario.asset_order
        }
        self._last_signal_state: dict[str, tuple[Any, ...]] = {}
        self.target_history: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.sell_campaigns: list[dict[str, Any]] = []
        self.ratcheted_swings: set[str] = set()
        self.permanently_blocked_close_swings: set[str] = set()
        self.close_preflight_reasons: dict[str, str] = {}
        self.execution_counter = 0
        self.realized_swing_pnl = 0.0
        self.start_ms = int(scenario.start.timestamp() * 1000)
        self.end_ms = int(scenario.end.timestamp() * 1000)
        self._replay_start_indices: dict[str, int] = {}
        self._validate_dataset()

    def run(
        self,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> SpotBacktestResult:
        replay_count = (self.end_ms - self.start_ms) // self.dataset.interval_ms
        rolling_offset = ROLLING_WINDOW_MS // self.dataset.interval_ms
        rows_by_symbol = self.dataset.candles
        initial_prices = {
            symbol: rows_by_symbol[symbol][self._replay_start_indices[symbol]].open
            for symbol in self.scenario.asset_order
        }
        for symbol, state in self.assets.items():
            state.rebase_cost_basis(initial_prices[symbol])
        final_prices = {
            symbol: rows_by_symbol[symbol][
                self._replay_start_indices[symbol] + replay_count - 1
            ].close
            for symbol in self.scenario.asset_order
        }
        starting_value = self.usdt + sum(
            self.assets[symbol].quantity * initial_prices[symbol]
            for symbol in self.scenario.asset_order
        )
        equity = [self._equity_point(self.start_ms, initial_prices, starting_value)]

        step_count = replay_count
        if progress is not None:
            progress(0, step_count)
        for step in range(step_count):
            self._economics_cache.clear()
            candles = {
                symbol: rows_by_symbol[symbol][self._replay_start_indices[symbol] + step]
                for symbol in self.scenario.asset_order
            }
            prices = {symbol: candle.close for symbol, candle in candles.items()}
            if step > 0:
                signal_candles = {
                    symbol: rows_by_symbol[symbol][
                        self._replay_start_indices[symbol] + step - 1
                    ]
                    for symbol in self.scenario.asset_order
                }
                references = {
                    symbol: rows_by_symbol[symbol][
                        self._replay_start_indices[symbol] + step - 1 - rolling_offset
                    ]
                    for symbol in self.scenario.asset_order
                }
                self._evaluate_cycle(
                    signal_candles,
                    references=references,
                    execution_candles=candles,
                )
            timestamp_ms = candles[self.scenario.asset_order[0]].close_time_ms
            if (timestamp_ms + 1) % HOUR_MS == 0 or step + 1 == step_count:
                equity.append(self._equity_point(timestamp_ms, prices, starting_value))
                self._record_inventory(timestamp_ms, prices)
            if progress is not None:
                progress(step + 1, step_count)

        if self.scenario.execution.force_close_at_end:
            self._economics_cache.clear()
            self._force_close(final_prices, self.end_ms)
            equity.append(self._equity_point(self.end_ms, final_prices, starting_value))

        self._economics_cache.clear()
        swings = self._final_swings(final_prices)
        return SpotBacktestResult(
            scenario=self.scenario,
            data_source=self.dataset.source,
            starting_value=starting_value,
            initial_prices=initial_prices,
            final_prices=final_prices,
            final_usdt=self.usdt,
            final_retained_usdt=self._retained_free_reserve_quote(),
            minimum_usdt=self.minimum_usdt,
            maximum_usdt=self.maximum_usdt,
            assets=self.assets,
            equity=equity,
            signals=self.signals,
            signal_level_counts={
                symbol: dict(counts)
                for symbol, counts in self.signal_level_counts.items()
            },
            actions=self.actions,
            swings=swings,
            executions=list(self.executions),
            accumulations=list(self.accumulations),
            cash_retentions=list(self.cash_retentions),
            target_history=list(self.target_history),
            inventory_history=list(self.inventory_history),
            rejections=list(self.rejections),
            sell_campaigns=[dict(item) for item in self.sell_campaigns],
        )

    def _validate_dataset(self) -> None:
        expected = set(self.scenario.asset_order)
        if set(self.dataset.candles) != expected:
            raise ValueError("Historical Spot dataset assets do not match the scenario.")
        if set(self.dataset.symbol_info) != expected:
            raise ValueError("Historical Spot symbol filters do not match the scenario.")
        duration_ms = self.end_ms - self.start_ms
        if duration_ms % self.dataset.interval_ms != 0:
            raise ValueError("Spot backtest start and end must align to complete candle intervals.")
        expected_replay_count = duration_ms // self.dataset.interval_ms
        if ROLLING_WINDOW_MS % self.dataset.interval_ms != 0:
            raise ValueError("Spot backtest interval must divide the rolling 24-hour signal window.")
        rolling_offset = ROLLING_WINDOW_MS // self.dataset.interval_ms
        for symbol in expected:
            rows = self.dataset.candles[symbol]
            if not rows:
                raise ValueError(f"{symbol} has no historical candles.")
            offset_ms = self.start_ms - rows[0].open_time_ms
            if offset_ms < 0 or offset_ms % self.dataset.interval_ms != 0:
                raise ValueError(f"{symbol} candles are not aligned with the replay start.")
            replay_start = offset_ms // self.dataset.interval_ms
            replay_end = replay_start + expected_replay_count
            required_warmup = max(self.scenario.warmup_candles, rolling_offset)
            if replay_start < required_warmup:
                raise ValueError(
                    f"{symbol} has {replay_start} warm-up candles; {required_warmup} are required."
                )
            if expected_replay_count < 2:
                raise ValueError(f"{symbol} requires at least two replay candles.")
            if (
                replay_end > len(rows)
                or rows[replay_start].open_time_ms != self.start_ms
                or rows[replay_end - 1].open_time_ms != self.end_ms - self.dataset.interval_ms
            ):
                raise ValueError(
                    f"{symbol} does not cover the complete requested replay range."
                )
            self._replay_start_indices[symbol] = replay_start

    def _replay_rows(self) -> dict[str, list[SpotCandle]]:
        replay_count = (self.end_ms - self.start_ms) // self.dataset.interval_ms
        return {
            symbol: list(
                self.dataset.candles[symbol][
                    self._replay_start_indices[symbol] :
                    self._replay_start_indices[symbol] + replay_count
                ]
            )
            for symbol in self.scenario.asset_order
        }

    def _evaluate_cycle(
        self,
        candles: dict[str, SpotCandle],
        *,
        references: dict[str, SpotCandle],
        execution_candles: dict[str, SpotCandle] | None = None,
    ) -> None:
        fills = execution_candles or candles
        contexts: list[tuple[str, SpotCandle, dict[str, Any], dict[str, Any]]] = []
        for symbol in self.scenario.asset_order:
            candle = candles[symbol]
            if not self._is_market_available(symbol, candle.open_time_ms):
                continue
            signal = self._signal_for(symbol, candle, reference=references[symbol])
            if signal is None:
                continue
            level = int(signal["level"])
            counts = self.signal_level_counts[symbol]
            counts[level] = counts.get(level, 0) + 1
            signal_state = (
                signal["opportunity"],
                level,
                bool(signal["strategy_eligible"]),
                signal["eligibility_reason"],
            )
            if self._last_signal_state.get(symbol) != signal_state:
                self.signals.append(
                    {
                        "timestamp_ms": candle.close_time_ms,
                        "timestamp": self._timestamp(candle.close_time_ms),
                        "asset_symbol": symbol,
                        "opportunity": signal["opportunity"],
                        "level": level,
                        "rolling_change_pct": signal["rolling_change_pct"],
                        "base_tranche_pct": signal["base_tranche_pct"],
                        "accumulation_tranche_pct": signal.get("accumulation_tranche_pct"),
                        "final_tranche_pct": signal["final_tranche_pct"],
                        "strategy_eligible": signal["strategy_eligible"],
                        "eligibility_reason": signal["eligibility_reason"],
                    }
                )
                self._last_signal_state[symbol] = signal_state
            prior = self.campaigns.get(symbol)
            if (
                prior is not None
                and prior.get("active_side") == "sell"
                and signal["opportunity"] == "buy"
            ):
                prior["ended_by_opposite_signal"] = True
                for historical in reversed(self.sell_campaigns):
                    if historical.get("campaign_id") == prior.get("campaign_id"):
                        historical["ended_by_opposite_signal"] = True
                        break
            campaign_id = hashlib.sha256(
                f"{symbol}|{signal['opportunity']}|{candle.close_time_ms}".encode("utf-8")
            ).hexdigest()[:24]
            campaign = transition_spot_strategy_campaign(
                prior,
                opportunity=str(signal["opportunity"]),
                signal_level=int(signal["level"]),
                signal_at_ms=candle.close_time_ms,
                new_campaign_id=campaign_id,
            )
            if (
                bool(signal.get("strategy_eligible"))
                and campaign.get("active_side") == "sell"
                and campaign.get("campaign_capacity_quantity") is None
            ):
                campaign = initialize_sell_campaign_capacity(
                    campaign,
                    self.assets[symbol].holding(candle.close),
                    config=self.scenario.progression,
                )
                campaign["asset_symbol"] = symbol
                campaign["started_at_ms"] = candle.close_time_ms
                campaign["ended_by_opposite_signal"] = False
                self.sell_campaigns.append(campaign)
            self.campaigns[symbol] = campaign
            contexts.append((symbol, candle, signal, campaign))

        ordered = sorted(
            enumerate(contexts),
            key=lambda item: self._signal_context_priority(item[1], item[0]),
        )
        for _index, (symbol, candle, signal, campaign) in ordered:
            execution_candle = fills[symbol]
            if not self._is_market_available(symbol, execution_candle.open_time_ms):
                continue
            self._execute_signal_batch(
                symbol,
                candle,
                signal,
                campaign,
                execution_price=execution_candle.open,
                execution_timestamp_ms=execution_candle.open_time_ms,
            )

    def _signal_context_priority(
        self,
        context: tuple[str, SpotCandle, dict[str, Any], dict[str, Any]],
        index: int,
    ) -> tuple[int, int, float, int]:
        symbol, candle, signal, campaign = context
        committed_quote = self._committed_quote_reserve()
        free_quote = self._spendable_free_reserve_quote(committed_quote=committed_quote)
        decision = plan_spot_strategy_action(
            signal,
            self.assets[symbol].holding(candle.close),
            campaign,
            self._active_swing_states(symbol, current_price=candle.close),
            available_quote=self.usdt,
            free_quote_reserve=free_quote,
            available_accumulation_quote=free_quote,
            config=self.scenario.progression,
            cleanup_config=self.scenario.waiter_cleanup,
            excluded_close_swing_ids=self.permanently_blocked_close_swings,
        )
        reason = decision.get("reason") if isinstance(decision, dict) else {}
        is_close = isinstance(decision, dict) and decision.get("action_type") == "close"
        is_profit = is_close and reason.get("close_reason") == "profit_target"
        favorable = float(reason.get("favorable_move_pct") or 0.0)
        return (0 if is_close else 1, 0 if is_profit else 1, -favorable, index)

    def _execute_signal_batch(
        self,
        symbol: str,
        candle: SpotCandle,
        signal: dict[str, Any],
        campaign: dict[str, Any],
        *,
        execution_price: float | None = None,
        execution_timestamp_ms: int | None = None,
    ) -> None:
        fill_price = candle.close if execution_price is None else float(execution_price)
        fill_timestamp_ms = (
            candle.close_time_ms
            if execution_timestamp_ms is None
            else int(execution_timestamp_ms)
        )
        excluded_close_swing_ids = set(self.permanently_blocked_close_swings)
        while True:
            committed_quote = self._committed_quote_reserve()
            free_quote = self._spendable_free_reserve_quote(committed_quote=committed_quote)
            decision = self._plan_executable_action(
                symbol,
                signal=signal,
                campaign=campaign,
                observed_price=candle.close,
                available_quote=self.usdt,
                free_quote_reserve=free_quote,
                available_accumulation_quote=free_quote,
                timestamp_ms=fill_timestamp_ms,
                execution_price=fill_price,
                excluded_close_swing_ids=excluded_close_swing_ids,
            )
            if decision is None:
                return
            action_type = str(decision["action_type"])
            if action_type == "hold":
                self._record_non_order_action(symbol, candle, signal, decision)
                return
            if action_type == "campaign_only":
                campaign["highest_completed_level"] = max(
                    int(campaign.get("highest_completed_level") or 0),
                    int(signal.get("level") or 0),
                )
                self._record_non_order_action(symbol, candle, signal, decision)
                continue

            action = self._prepare_action(symbol, candle, signal, campaign, decision)
            completed = self._execute_action(
                symbol,
                action,
                fill_price,
                fill_timestamp_ms,
            )
            if action_type == "close":
                excluded_close_swing_ids.add(str(decision["swing_id"]))
                continue
            if not completed:
                return

    def _record_non_order_action(
        self,
        symbol: str,
        candle: SpotCandle,
        signal: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        self.actions.append(
            {
                "timestamp_ms": candle.close_time_ms,
                "timestamp": self._timestamp(candle.close_time_ms),
                "asset_symbol": symbol,
                "action_type": decision["action_type"],
                "side": decision.get("side"),
                "swing_id": None,
                "signal_level": signal.get("level"),
                "base_tranche_pct": signal.get("base_tranche_pct"),
                "final_tranche_pct": signal.get("final_tranche_pct"),
                "requested_quantity": 0.0,
                "executed_quantity": 0.0,
                "requested_value": 0.0,
                "executed_value": 0.0,
                "fee": 0.0,
                "reason": decision["reason"],
            }
        )

    def _prepare_action(
        self,
        symbol: str,
        candle: SpotCandle,
        signal: dict[str, Any],
        campaign: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        action = {
            **decision,
            "signal": signal,
            "campaign_id": campaign.get("campaign_id"),
            "action_key": (
                f"{decision['action_type']}:{campaign.get('campaign_id')}:"
                f"level:{int(signal['level'])}"
            ),
            "decided_at_ms": candle.close_time_ms,
            "reference_price": signal.get("reference_price"),
            "swing_id": decision.get("swing_id"),
        }
        if decision["action_type"] == "open":
            action["swing_id"] = self._opening_swing_id(symbol, campaign, int(signal["level"]))
        return action

    def _plan_executable_action(
        self,
        symbol: str,
        *,
        signal: dict[str, Any],
        campaign: dict[str, Any],
        observed_price: float,
        available_quote: float,
        available_accumulation_quote: float,
        timestamp_ms: int,
        execution_price: float | None = None,
        free_quote_reserve: float | None = None,
        excluded_close_swing_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = set(self.permanently_blocked_close_swings)
        excluded.update(excluded_close_swing_ids or ())
        while True:
            decision = plan_spot_strategy_action(
                signal,
                self.assets[symbol].holding(observed_price),
                campaign,
                self._active_swing_states(symbol, current_price=observed_price),
                available_quote=available_quote,
                free_quote_reserve=free_quote_reserve,
                available_accumulation_quote=available_accumulation_quote,
                config=self.scenario.progression,
                cleanup_config=self.scenario.waiter_cleanup,
                excluded_close_swing_ids=excluded,
            )
            if decision is None or decision["action_type"] != "close":
                return decision
            error, permanent = self._close_preflight_error(
                symbol,
                decision,
                observed_price=(
                    observed_price if execution_price is None else execution_price
                ),
                available_quote=(
                    available_quote
                    if free_quote_reserve is None
                    else self._close_quote_budget(
                        decision,
                        free_quote_reserve=free_quote_reserve,
                    )
                ),
            )
            swing_id = str(decision["swing_id"])
            if error is None:
                self.close_preflight_reasons.pop(swing_id, None)
                return decision
            excluded.add(swing_id)
            if excluded_close_swing_ids is not None:
                excluded_close_swing_ids.add(swing_id)
            self._remember_close_preflight_rejection(
                symbol,
                decision,
                error=error,
                permanent=permanent,
                timestamp_ms=timestamp_ms,
            )

    def _remember_close_preflight_rejection(
        self,
        symbol: str,
        decision: dict[str, Any],
        *,
        error: str,
        permanent: bool,
        timestamp_ms: int,
    ) -> None:
        swing_id = str(decision["swing_id"])
        if permanent:
            self.permanently_blocked_close_swings.add(swing_id)
        if self.close_preflight_reasons.get(swing_id) == error:
            return
        self.close_preflight_reasons[swing_id] = error
        self.rejections.append(
            {
                "timestamp_ms": timestamp_ms,
                "timestamp": self._timestamp(timestamp_ms),
                "asset_symbol": symbol,
                "side": decision["side"],
                "action_type": decision["action_type"],
                "swing_id": decision.get("swing_id"),
                "requested_quantity": decision["requested_quantity"],
                "reason": error,
            }
        )

    def _close_preflight_error(
        self,
        symbol: str,
        decision: dict[str, Any],
        *,
        observed_price: float,
        available_quote: float,
    ) -> tuple[str | None, bool]:
        side = str(decision["side"])
        slippage = self.scenario.execution.slippage_bps / 10_000.0
        price = observed_price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        requested = float(decision["requested_quantity"])
        try:
            requested_quantity, _ = normalize_market_quantity(
                self.dataset.symbol_info[symbol],
                requested,
            )
            validate_market_notional(
                self.dataset.symbol_info[symbol],
                quantity=requested_quantity,
                price=price,
            )
        except ValueError as exc:
            message = str(exc)
            permanent = (
                decision.get("reason", {}).get("quantity_basis", "remaining_asset") == "remaining_asset"
                and (
                    "rounds to zero" in message
                    or "Quantity is below Binance minimum" in message
                )
            )
            return message, permanent

        quantity = float(requested_quantity)
        if side == "buy":
            required_quote = quantity * price * (1.0 + self.scenario.execution.fee_rate)
            if required_quote > available_quote + 1e-9:
                return "Buy rejected because simulated USDT balance changed after planning.", False
        elif quantity > self.assets[symbol].immediately_sellable + 1e-9:
            return (
                "Sell rejected because simulated immediately sellable inventory changed after planning.",
                False,
            )
        return None, False

    def _is_market_available(self, symbol: str, open_time_ms: int) -> bool:
        return open_time_ms not in self.dataset.unavailable_open_times.get(symbol, frozenset())

    def _signal_for(
        self,
        symbol: str,
        candle: SpotCandle,
        *,
        reference: SpotCandle | None = None,
    ) -> dict[str, Any] | None:
        if reference is None:
            rows = self.dataset.candles[symbol]
            offset_ms = candle.open_time_ms - rows[0].open_time_ms
            if offset_ms < ROLLING_WINDOW_MS or offset_ms % self.dataset.interval_ms != 0:
                return None
            reference_index = (
                offset_ms - ROLLING_WINDOW_MS
            ) // self.dataset.interval_ms
            reference = rows[reference_index]
        signal = evaluate_rolling_24h_opportunity(
            current_price=candle.close,
            reference_price=reference.close,
            current_at_ms=candle.close_time_ms,
            reference_at_ms=reference.close_time_ms,
            config=self.scenario.signal,
        )
        asset = self.assets[symbol].scenario
        policy = {
            "trading_objective": asset.trading_objective,
            "minimum_holding_pct": asset.minimum_holding_pct,
        }
        return finalize_spot_strategy_signal(
            signal,
            policy,
            execution_enabled=True,
        )

    def _execute_action(
        self,
        symbol: str,
        action: dict[str, Any],
        observed_price: float,
        timestamp_ms: int,
    ) -> bool:
        state = self.assets[symbol]
        side = str(action["side"])
        slippage = self.scenario.execution.slippage_bps / 10_000.0
        price = observed_price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        requested = float(action["requested_quantity"])
        try:
            quantity_decimal, _ = normalize_market_quantity(
                self.dataset.symbol_info[symbol],
                requested,
            )
            validate_market_notional(
                self.dataset.symbol_info[symbol],
                quantity=quantity_decimal,
                price=price,
            )
            if action["action_type"] == "open" and side == "sell":
                validate_sell_opening_round_trip(
                    self.dataset.symbol_info[symbol],
                    quantity=quantity_decimal,
                    price=price,
                    trading_objective=state.scenario.trading_objective or "",
                    close_profit_pct=self.scenario.progression.close_profit_pct,
                    estimated_fee_rate=self.scenario.execution.fee_rate,
                )
            quantity = float(quantity_decimal)
            if side == "buy":
                required_quote = quantity * price * (1.0 + self.scenario.execution.fee_rate)
                if required_quote > self.usdt + 1e-9:
                    raise ValueError("Buy rejected because simulated USDT balance changed after planning.")
                if (
                    action["action_type"] == "accumulate_asset"
                    and required_quote > self._spendable_free_reserve_quote() + 1e-9
                ):
                    raise ValueError(
                        "Treasury accumulation rejected because Free Vault Reserve changed after planning."
                    )
            elif quantity > state.immediately_sellable + 1e-9:
                raise ValueError(
                    "Sell rejected because simulated immediately sellable inventory changed after planning."
                )
            if action["action_type"] == "close":
                swing = self.swings.get(str(action.get("swing_id") or ""))
                if swing is not None:
                    if side == "buy":
                        allowed_quote = self._close_quote_budget(action)
                        required_quote = quantity * price * (1.0 + self.scenario.execution.fee_rate)
                        if required_quote > allowed_quote + 1e-9:
                            raise ValueError(
                                "Buy close rejected because it would spend another Bargain's committed cash."
                            )
        except ValueError as exc:
            if action["action_type"] == "close":
                message = str(exc)
                swing_id = str(action["swing_id"])
                self.close_preflight_reasons[swing_id] = message
            self.rejections.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": self._timestamp(timestamp_ms),
                    "asset_symbol": symbol,
                    "side": side,
                    "action_type": action["action_type"],
                    "swing_id": action.get("swing_id"),
                    "requested_quantity": requested,
                    "reason": str(exc),
                }
            )
            return False
        quote_quantity = quantity * price
        fee = quote_quantity * self.scenario.execution.fee_rate
        if side == "buy" and quote_quantity + fee > self.usdt + 1e-9:
            return False

        if action["action_type"] == "accumulate_asset":
            self.usdt -= quote_quantity + fee
            state.apply_buy(quantity, quote_quantity, fee)
            self.minimum_usdt = min(self.minimum_usdt, self.usdt)
            self.maximum_usdt = max(self.maximum_usdt, self.usdt)
            previous_target = state.target_quantity
            state.target_quantity += quantity
            self.execution_counter += 1
            execution_id = f"sim-{self.execution_counter:08d}"
            accumulation = {
                "action_key": action["action_key"],
                "execution_id": execution_id,
                "action_type": "accumulate_asset",
                "side": "buy",
                "timestamp_ms": timestamp_ms,
                "timestamp": self._timestamp(timestamp_ms),
                "asset_symbol": symbol,
                "quote_symbol": "USDT",
                "campaign_id": action.get("campaign_id"),
                "signal_level": action.get("signal", {}).get("level"),
                "requested_quantity": requested,
                "executed_quantity": quantity,
                "net_asset_acquired": quantity,
                "price": price,
                "quote_quantity": quote_quantity,
                "deployed_quote_quantity": quote_quantity + fee,
                "fee_amount": fee,
                "fee_asset": "USDT",
                "previous_target_quantity": previous_target,
                "next_target_quantity": state.target_quantity,
                "target_growth_quantity": quantity,
            }
            self.accumulations.append(accumulation)
            self.executions.append(
                {
                    "execution_id": execution_id,
                    "swing_id": None,
                    "strategy_action_key": action["action_key"],
                    "venue": "binance-simulated",
                    "symbol": f"{symbol}USDT",
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "quote_quantity": quote_quantity,
                    "fee_amount": fee,
                    "fee_asset": "USDT",
                    "source": "strategy",
                    "reason": action.get("reason") or {},
                    "executed_at_ms": timestamp_ms,
                    "executed_at": self._timestamp(timestamp_ms),
                }
            )
            self.target_history.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": self._timestamp(timestamp_ms),
                    "asset_symbol": symbol,
                    "swing_id": None,
                    "action_key": action["action_key"],
                    "ratchet_source": "treasury_accumulation",
                    "previous_target_quantity": previous_target,
                    "applied_gain_quantity": quantity,
                    "next_target_quantity": state.target_quantity,
                }
            )
            campaign = self.campaigns[symbol]
            campaign["highest_completed_level"] = max(
                int(campaign.get("highest_completed_level") or 0),
                int(action.get("signal", {}).get("level") or 0),
            )
            self.actions.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": self._timestamp(timestamp_ms),
                    "asset_symbol": symbol,
                    "action_type": "accumulate_asset",
                    "side": "buy",
                    "swing_id": None,
                    "action_key": action["action_key"],
                    "signal_level": action.get("signal", {}).get("level"),
                    "base_tranche_pct": action.get("signal", {}).get("base_tranche_pct"),
                    "accumulation_tranche_pct": action.get("signal", {}).get("accumulation_tranche_pct"),
                    "final_tranche_pct": action.get("signal", {}).get("final_tranche_pct"),
                    "requested_quantity": requested,
                    "executed_quantity": quantity,
                    "requested_value": requested * observed_price,
                    "executed_value": quote_quantity,
                    "fee": fee,
                    "reason": dict(action.get("reason") or {}),
                }
            )
            return True

        swing_id = str(action["swing_id"])
        if action["action_type"] == "open" and swing_id not in self.swings:
            self.swings[swing_id] = {
                "swing_id": swing_id,
                "account_key": "spot_backtest",
                "asset_symbol": symbol,
                "quote_symbol": "USDT",
                "origin_side": side,
                "trading_objective": state.scenario.trading_objective,
                "status": "open",
                "planned_quantity": requested,
                "reference_state": {
                    "rolling_reference_price": action.get("reference_price"),
                    "signal_price": action.get("signal", {}).get("current_price"),
                    "signal_at_ms": action.get("decided_at_ms"),
                    "campaign_id": action.get("campaign_id"),
                    "signal_level": action.get("signal", {}).get("level"),
                },
                "strategy_reason": action.get("reason") or {},
                "source": "strategy",
                "opened_at_ms": timestamp_ms,
                "closed_at_ms": None,
                "executions": [],
            }
            self._register_swing(swing_id, self.swings[swing_id])
        swing = self.swings.get(swing_id)
        if swing is None:
            return False

        if side == "buy":
            self.usdt -= quote_quantity + fee
            state.apply_buy(quantity, quote_quantity, fee)
        else:
            self.usdt += quote_quantity - fee
            state.apply_sell(quantity, quote_quantity, fee)
        self.minimum_usdt = min(self.minimum_usdt, self.usdt)
        self.maximum_usdt = max(self.maximum_usdt, self.usdt)

        self.execution_counter += 1
        execution = {
            "execution_id": f"sim-{self.execution_counter:08d}",
            "swing_id": swing_id,
            "venue": "binance-simulated",
            "symbol": f"{symbol}USDT",
            "side": side,
            "quantity": quantity,
            "price": price,
            "quote_quantity": quote_quantity,
            "fee_amount": fee,
            "fee_asset": "USDT",
            "source": "strategy",
            "reason": action.get("reason") or {},
            "executed_at_ms": timestamp_ms,
            "executed_at": self._timestamp(timestamp_ms),
        }
        swing["executions"].append(execution)
        self.executions.append(execution)
        self._invalidate_swing_economics(swing_id)
        economics = self._swing_economics(swing, swing["executions"], current_price=price)
        swing["status"] = economics["status"]
        if economics["status"] == "closed":
            swing["closed_at_ms"] = timestamp_ms
            swing["close_reason"] = (action.get("reason") or {}).get("close_reason")
            swing["close_context"] = dict(action.get("reason") or {})
            self.realized_swing_pnl += float(economics.get("realized_pnl_quote") or 0.0)
            self._apply_cash_retention(swing, economics, timestamp_ms)
            self._deactivate_swing(swing_id, symbol)
            self._apply_target_ratchet(state, swing, economics, timestamp_ms)
        if action["action_type"] == "open":
            current_campaign = self.campaigns[symbol]
            campaign = next(
                (
                    item for item in reversed(self.sell_campaigns)
                    if item.get("campaign_id") == action.get("campaign_id")
                ),
                current_campaign,
            )
            for target_campaign in {id(campaign): campaign, id(current_campaign): current_campaign}.values():
                target_campaign["highest_completed_level"] = max(
                    int(target_campaign.get("highest_completed_level") or 0),
                    int(action.get("signal", {}).get("level") or 0),
                )
                target_campaign["campaign_consumed_quantity"] = min(
                    float(target_campaign.get("campaign_capacity_quantity") or 0.0),
                    float(target_campaign.get("campaign_consumed_quantity") or 0.0) + quantity,
                )
        self.actions.append(
            {
                "timestamp_ms": timestamp_ms,
                "timestamp": self._timestamp(timestamp_ms),
                "asset_symbol": symbol,
                "action_type": action["action_type"],
                "side": side,
                "swing_id": swing_id,
                "signal_level": action.get("signal", {}).get("level"),
                "base_tranche_pct": action.get("signal", {}).get("base_tranche_pct"),
                "final_tranche_pct": action.get("signal", {}).get("final_tranche_pct"),
                "requested_quantity": requested,
                "executed_quantity": quantity,
                "requested_value": requested * observed_price,
                "executed_value": quote_quantity,
                "fee": fee,
                "reason": dict(action.get("reason") or {}),
            }
        )
        return True

    def _apply_target_ratchet(
        self,
        state: SimulatedAssetState,
        swing: dict[str, Any],
        economics: dict[str, Any],
        timestamp_ms: int,
    ) -> None:
        swing_id = str(swing["swing_id"])
        if swing_id in self.ratcheted_swings:
            return
        proposal = calculate_target_ratchet(
            swing,
            economics,
            target_quantity=state.target_quantity,
            minimum_holding_pct=state.scenario.minimum_holding_pct,
        )
        applied = float(proposal["applied_gain_quantity"])
        if applied <= 0:
            return
        state.target_quantity = float(proposal["next_target_quantity"])
        self.ratcheted_swings.add(swing_id)
        self.target_history.append(
            {
                "timestamp_ms": timestamp_ms,
                "timestamp": self._timestamp(timestamp_ms),
                "asset_symbol": state.symbol,
                "swing_id": swing_id,
                **proposal,
            }
        )

    def _register_swing(self, swing_id: str, swing: dict[str, Any]) -> None:
        if swing_id in self._indexed_swing_ids:
            return
        self._indexed_swing_ids.add(swing_id)
        symbol = str(swing["asset_symbol"])
        self._swing_ids_by_asset.setdefault(symbol, []).append(swing_id)
        if swing["status"] in {"open", "partially_closed"}:
            self._active_swing_ids_by_asset.setdefault(symbol, []).append(swing_id)
        else:
            economics = self._swing_economics(swing, swing["executions"])
            self.realized_swing_pnl += float(economics.get("realized_pnl_quote") or 0.0)
        self._committed_quote_reserve_cache = None

    def _deactivate_swing(self, swing_id: str, symbol: str) -> None:
        active_ids = self._active_swing_ids_by_asset.get(symbol)
        if active_ids is not None:
            try:
                active_ids.remove(swing_id)
            except ValueError:
                pass

    def _sync_swing_index(self) -> None:
        if len(self._indexed_swing_ids) == len(self.swings):
            return
        for swing_id, swing in self.swings.items():
            self._register_swing(str(swing_id), swing)

    def _active_swing_states(
        self,
        symbol: str,
        *,
        current_price: float | None = None,
    ) -> list[dict[str, Any]]:
        self._sync_swing_index()
        return [
            {
                "swing": swing,
                "executions": swing["executions"],
                **(
                    {
                        "economics": self._swing_economics(
                            swing,
                            swing["executions"],
                            current_price=current_price,
                        )
                    }
                    if current_price is not None
                    else {}
                ),
            }
            for swing_id in self._active_swing_ids_by_asset.get(symbol, ())
            if (swing := self.swings.get(swing_id)) is not None
            and swing["status"] in {"open", "partially_closed"}
        ]

    def _swing_economics(
        self,
        swing: dict[str, Any],
        executions: list[dict[str, Any]],
        *,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        swing_id = str(swing.get("swing_id") or id(swing))
        key = (swing_id, None if current_price is None else float(current_price))
        cached = self._economics_cache.get(key)
        if cached is None:
            cached = calculate_swing_economics(
                swing,
                executions,
                current_price=current_price,
            )
            self._economics_cache[key] = cached
        return cached

    def _invalidate_swing_economics(self, swing_id: str) -> None:
        for key in tuple(self._economics_cache):
            if key[0] == swing_id:
                del self._economics_cache[key]
        self._committed_quote_by_swing.pop(swing_id, None)
        self._committed_quote_reserve_cache = None

    def _apply_cash_retention(
        self,
        swing: dict[str, Any],
        economics: dict[str, Any],
        timestamp_ms: int,
    ) -> None:
        if swing.get("cash_retention") is not None:
            return
        retention = calculate_cash_retention(
            swing,
            economics,
            retention_pct=self.scenario.free_cash_retention_pct,
        )
        if retention["eligible_cash_gain_quote"] <= 0:
            return
        retained_quote = float(retention["retained_quote"])
        if retained_quote <= 0:
            return
        record = {
            "swing_id": str(swing["swing_id"]),
            "asset_symbol": str(swing["asset_symbol"]),
            "quote_symbol": str(swing.get("quote_symbol") or "USDT"),
            "timestamp_ms": timestamp_ms,
            "timestamp": self._timestamp(timestamp_ms),
            **retention,
        }
        swing["cash_retention"] = record
        self.cash_retentions.append(record)
        self.retained_free_quote += retained_quote

    def _equity_point(
        self,
        timestamp_ms: int,
        prices: dict[str, float],
        starting_value: float,
    ) -> dict[str, Any]:
        treasury_value = self.usdt + sum(
            self.assets[symbol].quantity * prices[symbol]
            for symbol in self.scenario.asset_order
        )
        hodl_value = self.scenario.starting_usdt + sum(
            self.assets[symbol].scenario.quantity * prices[symbol]
            for symbol in self.scenario.asset_order
        )
        open_unrealized = 0.0
        realized = self.realized_swing_pnl
        open_count = 0
        for symbol in self.scenario.asset_order:
            for item in self._active_swing_states(symbol):
                economics = self._swing_economics(
                    item["swing"],
                    item["executions"],
                    current_price=prices[symbol],
                )
                open_count += 1
                open_unrealized += float(economics.get("unrealized_pnl_quote") or 0.0)
        realized_portfolio_pnl = sum(
            state.realized_portfolio_pnl for state in self.assets.values()
        )
        unrealized_portfolio_pnl = sum(
            self.assets[symbol].unrealized_portfolio_pnl(prices[symbol])
            for symbol in self.scenario.asset_order
        )
        reserved_quote = min(
            self.usdt,
            self._committed_quote_reserve(),
        )
        available_quote = max(0.0, self.usdt - reserved_quote)
        retained_quote = min(available_quote, self.retained_free_quote)
        return {
            "timestamp_ms": timestamp_ms,
            "timestamp": self._timestamp(timestamp_ms),
            "treasury_value": treasury_value,
            "hodl_value": hodl_value,
            "difference_vs_hodl": treasury_value - hodl_value,
            "shared_usdt": self.usdt,
            "shared_usdt_reserved": reserved_quote,
            "shared_usdt_available": available_quote,
            "shared_usdt_retained": retained_quote,
            "shared_usdt_spendable": max(0.0, available_quote - retained_quote),
            "realized_swing_pnl": realized,
            "open_swing_unrealized_pnl": open_unrealized,
            "realized_portfolio_pnl_on_known_basis": realized_portfolio_pnl,
            "unrealized_portfolio_pnl_on_known_basis": unrealized_portfolio_pnl,
            "open_swings": open_count,
            "return_pct": ((treasury_value / starting_value) - 1.0) * 100.0 if starting_value > 0 else 0.0,
        }

    def _swing_committed_quote(self, swing: dict[str, Any]) -> float:
        swing_id = str(swing.get("swing_id") or id(swing))
        cached = self._committed_quote_by_swing.get(swing_id)
        if cached is not None:
            return cached
        economics = self._swing_economics(
            swing,
            swing["executions"],
        )
        committed = calculate_sell_origin_committed_quote(
            swing,
            economics,
        )
        self._committed_quote_by_swing[swing_id] = committed
        return committed

    def _committed_quote_reserve(self) -> float:
        if self._committed_quote_reserve_cache is None:
            self._committed_quote_reserve_cache = sum(
                self._swing_committed_quote(swing) for swing in self.swings.values()
            )
        return self._committed_quote_reserve_cache

    def _free_reserve_quote(self) -> float:
        return max(0.0, self.usdt - self._committed_quote_reserve())

    def _retained_free_reserve_quote(self, *, free_quote: float | None = None) -> float:
        available = self._free_reserve_quote() if free_quote is None else max(0.0, free_quote)
        return min(available, max(0.0, self.retained_free_quote))

    def _spendable_free_reserve_quote(self, *, committed_quote: float | None = None) -> float:
        committed = self._committed_quote_reserve() if committed_quote is None else committed_quote
        free_quote = max(0.0, self.usdt - committed)
        return max(0.0, free_quote - self._retained_free_reserve_quote(free_quote=free_quote))

    def _close_quote_budget(
        self,
        decision: dict[str, Any],
        *,
        free_quote_reserve: float | None = None,
    ) -> float:
        if decision.get("action_type") != "close" or decision.get("side") != "buy":
            return self.usdt
        swing = self.swings.get(str(decision.get("swing_id") or ""))
        if swing is None or swing.get("origin_side") != "sell":
            return self.usdt
        free_quote = (
            self._spendable_free_reserve_quote()
            if free_quote_reserve is None
            else free_quote_reserve
        )
        return min(self.usdt, max(0.0, free_quote) + self._swing_committed_quote(swing))

    def _record_inventory(self, timestamp_ms: int, prices: dict[str, float]) -> None:
        for symbol in self.scenario.asset_order:
            state = self.assets[symbol]
            active_swing_states = self._active_swing_states(symbol)
            committed = 0.0
            underwater_sell_count = 0
            open_buy_count = 0
            open_buy_capital_tied = 0.0
            open_buy_mtm_pnl = 0.0
            for item in active_swing_states:
                swing = item["swing"]
                economics = self._swing_economics(
                    swing,
                    item["executions"],
                    current_price=prices[symbol],
                )
                if swing["origin_side"] == "sell":
                    committed += float(economics["remaining_quantity"])
                    if float(economics.get("unrealized_pnl_quote") or 0.0) < 0:
                        underwater_sell_count += 1
                else:
                    open_buy_count += 1
                    open_buy_capital_tied += (
                        float(economics.get("remaining_quantity") or 0.0)
                        * float(economics.get("weighted_opening_price") or 0.0)
                    )
                    open_buy_mtm_pnl += float(economics.get("unrealized_pnl_quote") or 0.0)
            capacity = state.policy_sellable + committed
            self.inventory_history.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": self._timestamp(timestamp_ms),
                    "asset_symbol": symbol,
                    "target_holding": state.target_quantity,
                    "protected_floor": state.protected_floor,
                    "current_total": state.quantity,
                    "binance_quantity": state.binance_quantity,
                    "cold_storage_quantity": state.cold_storage_quantity,
                    "policy_sellable_quantity": state.policy_sellable,
                    "quantity_committed_to_open_swings": committed,
                    "open_bargain_count": len(active_swing_states),
                    "underwater_sell_origin_swings": underwater_sell_count,
                    "open_buy_origin_swings": open_buy_count,
                    "open_buy_capital_tied_quote": open_buy_capital_tied,
                    "open_buy_mark_to_market_pnl": open_buy_mtm_pnl,
                    "amount_above_protected_floor": state.policy_sellable,
                    "tradable_inventory_utilization_pct": (committed / capacity) * 100 if capacity > 0 else 0.0,
                    "distance_above_floor_pct": (
                        ((state.quantity - state.protected_floor) / state.target_quantity) * 100
                        if state.target_quantity > 0
                        else 0.0
                    ),
                }
            )

    def _final_swings(self, final_prices: dict[str, float]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for swing in sorted(self.swings.values(), key=lambda item: (item["opened_at_ms"], item["swing_id"])):
            economics = self._swing_economics(
                swing,
                swing["executions"],
                current_price=final_prices[swing["asset_symbol"]],
            )
            end_ms = int(swing.get("closed_at_ms") or self.end_ms)
            output.append(
                {
                    **swing,
                    "status": economics["status"],
                    "age_seconds": max(0, (end_ms - int(swing["opened_at_ms"])) // 1000),
                    "current_market_price": final_prices[swing["asset_symbol"]],
                    "economics": economics,
                }
            )
        return output

    def _force_close(self, final_prices: dict[str, float], timestamp_ms: int) -> None:
        for symbol in self.scenario.asset_order:
            for item in list(self._active_swing_states(symbol)):
                swing = item["swing"]
                economics = self._swing_economics(
                    swing,
                    item["executions"],
                    current_price=final_prices[symbol],
                )
                quantity = float(economics["remaining_quantity"])
                if quantity <= 0:
                    continue
                self._execute_action(
                    symbol,
                    {
                        "action_type": "close",
                        "side": economics["closing_side"],
                        "swing_id": swing["swing_id"],
                        "requested_quantity": quantity,
                        "reason": {"action_type": "close", "close_reason": "force_close_at_end"},
                        "signal": {},
                    },
                    final_prices[symbol],
                    timestamp_ms,
                )

    @staticmethod
    def _opening_swing_id(symbol: str, campaign: dict[str, Any], level: int) -> str:
        identity = f"{symbol}|{campaign.get('campaign_id')}|{level}"
        return f"swing-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _timestamp(timestamp_ms: int) -> str:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
