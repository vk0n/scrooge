from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any, Callable

from backtest.spot_market_data import SpotCandle, SpotHistoricalDataset
from backtest.spot_scenario import SpotBacktestAsset, SpotBacktestScenario
from bot.spot_signal import calculate_spot_indicator_context
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
            known_cost_basis=known_quantity * float(asset.entry_cost or 0.0),
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
        """Use the first replay price as the portfolio's backtest entry basis."""
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
    minimum_usdt: float
    maximum_usdt: float
    assets: dict[str, SimulatedAssetState]
    equity: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    swings: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    accumulations: list[dict[str, Any]]
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
        self._indexed_swing_ids: set[str] = set()
        self._economics_cache: dict[tuple[str, float | None], dict[str, Any]] = {}
        self.executions: list[dict[str, Any]] = []
        self.accumulations: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.target_history: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.sell_campaigns: list[dict[str, Any]] = []
        self.ratcheted_swings: set[str] = set()
        self.permanently_blocked_close_swings: set[str] = set()
        self.close_preflight_reasons: dict[str, str] = {}
        self.execution_counter = 0
        self.start_ms = int(scenario.start.timestamp() * 1000)
        self.end_ms = int(scenario.end.timestamp() * 1000)
        self._validate_dataset()
        self._candle_by_close = {
            symbol: {row.close_time_ms: row for row in rows}
            for symbol, rows in self.dataset.candles.items()
        }
        self._indicator_candles = {
            symbol: self._hourly_indicator_candles(rows)
            for symbol, rows in self.dataset.candles.items()
        }
        self._indicator_close_times = {
            symbol: [row.close_time_ms for row in rows]
            for symbol, rows in self._indicator_candles.items()
        }

    def run(
        self,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> SpotBacktestResult:
        replay_rows = self._replay_rows()
        initial_prices = {symbol: rows[0].open for symbol, rows in replay_rows.items()}
        for symbol, state in self.assets.items():
            state.rebase_cost_basis(initial_prices[symbol])
        final_prices = {symbol: rows[-1].close for symbol, rows in replay_rows.items()}
        starting_value = self.usdt + sum(
            self.assets[symbol].quantity * initial_prices[symbol]
            for symbol in self.scenario.asset_order
        )
        equity = [self._equity_point(self.start_ms, initial_prices, starting_value)]

        step_count = len(next(iter(replay_rows.values())))
        if progress is not None:
            progress(0, step_count)
        for step in range(step_count):
            self._economics_cache.clear()
            candles = {symbol: replay_rows[symbol][step] for symbol in self.scenario.asset_order}
            prices = {symbol: candle.close for symbol, candle in candles.items()}
            self._evaluate_cycle(candles)
            equity.append(self._equity_point(candles[self.scenario.asset_order[0]].close_time_ms, prices, starting_value))
            self._record_inventory(candles[self.scenario.asset_order[0]].close_time_ms, prices)
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
            minimum_usdt=self.minimum_usdt,
            maximum_usdt=self.maximum_usdt,
            assets=self.assets,
            equity=equity,
            signals=self.signals,
            actions=self.actions,
            swings=swings,
            executions=list(self.executions),
            accumulations=list(self.accumulations),
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
        for symbol in expected:
            rows = self.dataset.candles[symbol]
            warmup = [row for row in rows if row.close_time_ms < self.start_ms]
            replay = [row for row in rows if row.open_time_ms >= self.start_ms and row.close_time_ms <= self.end_ms]
            if len(warmup) < self.scenario.warmup_candles:
                raise ValueError(
                    f"{symbol} has {len(warmup)} warm-up candles; {self.scenario.warmup_candles} are required."
                )
            if len(replay) < 2:
                raise ValueError(f"{symbol} requires at least two replay candles.")
            if (
                len(replay) != expected_replay_count
                or replay[0].open_time_ms != self.start_ms
                or replay[-1].open_time_ms != self.end_ms - self.dataset.interval_ms
            ):
                raise ValueError(
                    f"{symbol} does not cover the complete requested replay range."
                )

    def _replay_rows(self) -> dict[str, list[SpotCandle]]:
        output = {
            symbol: [
                row
                for row in self.dataset.candles[symbol]
                if row.open_time_ms >= self.start_ms and row.close_time_ms <= self.end_ms
            ]
            for symbol in self.scenario.asset_order
        }
        reference_times = [row.open_time_ms for row in output[self.scenario.asset_order[0]]]
        for symbol, rows in output.items():
            if [row.open_time_ms for row in rows] != reference_times:
                raise ValueError(f"{symbol} replay candles are not aligned with the portfolio timeline.")
        return output

    def _evaluate_cycle(self, candles: dict[str, SpotCandle]) -> None:
        contexts: list[tuple[str, SpotCandle, dict[str, Any], dict[str, Any]]] = []
        for symbol in self.scenario.asset_order:
            candle = candles[symbol]
            if not self._is_market_available(symbol, candle.open_time_ms):
                continue
            signal = self._signal_for(symbol, candle)
            if signal is None:
                continue
            self.signals.append(
                {
                    "timestamp_ms": candle.close_time_ms,
                    "timestamp": self._timestamp(candle.close_time_ms),
                    "asset_symbol": symbol,
                    "opportunity": signal["opportunity"],
                    "level": signal["level"],
                    "rolling_change_pct": signal["rolling_change_pct"],
                    "base_tranche_pct": signal["base_tranche_pct"],
                    "accumulation_tranche_pct": signal.get("accumulation_tranche_pct"),
                    "sizing_modifier": signal["sizing_modifier"],
                    "final_tranche_pct": signal["final_tranche_pct"],
                    "indicator_tier": (signal.get("indicator_assessment") or {}).get("tier"),
                    "rsi": (signal.get("indicator_context") or {}).get("rsi"),
                    "ema": (signal.get("indicator_context") or {}).get("ema"),
                    "bb_lower": (signal.get("indicator_context") or {}).get("bb_lower"),
                    "bb_upper": (signal.get("indicator_context") or {}).get("bb_upper"),
                    "atr": (signal.get("indicator_context") or {}).get("atr"),
                    "strategy_eligible": signal["strategy_eligible"],
                    "eligibility_reason": signal["eligibility_reason"],
                }
            )
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
            self._execute_signal_batch(symbol, candle, signal, campaign)

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
            self._active_swing_states(symbol),
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
    ) -> None:
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
                timestamp_ms=candle.close_time_ms,
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
                candle.close,
                candle.close_time_ms,
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
                "sizing_modifier": signal.get("sizing_modifier"),
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
                self._active_swing_states(symbol),
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
                observed_price=observed_price,
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

    def _signal_for(self, symbol: str, candle: SpotCandle) -> dict[str, Any] | None:
        desired_reference_ms = candle.close_time_ms - ROLLING_WINDOW_MS
        reference = self._candle_by_close[symbol].get(desired_reference_ms)
        if reference is None:
            return None
        signal = evaluate_rolling_24h_opportunity(
            current_price=candle.close,
            reference_price=reference.close,
            current_at_ms=candle.close_time_ms,
            reference_at_ms=reference.close_time_ms,
            config=self.scenario.signal,
        )
        indicator_context = None
        indicator_error = None
        if signal["opportunity"] != "hold":
            hourly_rows = self._indicator_candles[symbol]
            latest_hour = bisect_right(
                self._indicator_close_times[symbol],
                candle.close_time_ms,
            )
            visible = [row.as_binance_kline() for row in hourly_rows[max(0, latest_hour - 60):latest_hour]]
            try:
                indicator_context = calculate_spot_indicator_context(
                    visible,
                    current_price=candle.close,
                    evaluated_at_ms=candle.close_time_ms,
                    interval="1h",
                )
            except ValueError as exc:
                indicator_error = str(exc)
        asset = self.assets[symbol].scenario
        policy = {
            "trading_objective": asset.trading_objective,
            "minimum_holding_pct": asset.minimum_holding_pct,
        }
        return finalize_spot_strategy_signal(
            signal,
            indicator_context,
            policy,
            execution_enabled=True,
            sizing_config=self.scenario.sizing,
            indicator_error=indicator_error,
        )

    def _hourly_indicator_candles(
        self,
        rows: tuple[SpotCandle, ...],
    ) -> tuple[SpotCandle, ...]:
        if self.dataset.interval_ms == HOUR_MS:
            return rows
        if self.dataset.interval_ms > HOUR_MS or HOUR_MS % self.dataset.interval_ms != 0:
            return rows

        expected_rows = HOUR_MS // self.dataset.interval_ms
        buckets: dict[int, list[SpotCandle]] = {}
        for row in rows:
            hour_open_ms = (row.open_time_ms // HOUR_MS) * HOUR_MS
            buckets.setdefault(hour_open_ms, []).append(row)

        output: list[SpotCandle] = []
        for hour_open_ms in sorted(buckets):
            bucket = buckets[hour_open_ms]
            if (
                len(bucket) != expected_rows
                or bucket[0].open_time_ms != hour_open_ms
                or bucket[-1].close_time_ms != hour_open_ms + HOUR_MS - 1
            ):
                continue
            output.append(
                SpotCandle(
                    open_time_ms=hour_open_ms,
                    close_time_ms=hour_open_ms + HOUR_MS - 1,
                    open=bucket[0].open,
                    high=max(row.high for row in bucket),
                    low=min(row.low for row in bucket),
                    close=bucket[-1].close,
                    volume=sum(row.volume for row in bucket),
                )
            )
        return tuple(output)

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
                "conviction": (
                    action.get("signal", {}).get("indicator_assessment") or {}
                ).get("tier"),
                "sizing_modifier": action.get("signal", {}).get("sizing_modifier"),
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
                    "sizing_modifier": action.get("signal", {}).get("sizing_modifier"),
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
                "sizing_modifier": action.get("signal", {}).get("sizing_modifier"),
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
        self._swing_ids_by_asset.setdefault(str(swing["asset_symbol"]), []).append(swing_id)

    def _sync_swing_index(self) -> None:
        if len(self._indexed_swing_ids) == len(self.swings):
            return
        for swing_id, swing in self.swings.items():
            self._register_swing(str(swing_id), swing)

    def _active_swing_states(self, symbol: str) -> list[dict[str, Any]]:
        self._sync_swing_index()
        return [
            {"swing": swing, "executions": swing["executions"]}
            for swing_id in self._swing_ids_by_asset.get(symbol, ())
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
        realized = 0.0
        open_count = 0
        for swing in self.swings.values():
            economics = self._swing_economics(
                swing,
                swing["executions"],
                current_price=prices[swing["asset_symbol"]],
            )
            if economics["status"] == "closed":
                realized += float(economics.get("realized_pnl_quote") or 0.0)
            else:
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
        return {
            "timestamp_ms": timestamp_ms,
            "timestamp": self._timestamp(timestamp_ms),
            "treasury_value": treasury_value,
            "hodl_value": hodl_value,
            "difference_vs_hodl": treasury_value - hodl_value,
            "shared_usdt": self.usdt,
            "shared_usdt_reserved": reserved_quote,
            "shared_usdt_available": max(0.0, self.usdt - reserved_quote),
            "realized_swing_pnl": realized,
            "open_swing_unrealized_pnl": open_unrealized,
            "realized_portfolio_pnl_on_known_basis": realized_portfolio_pnl,
            "unrealized_portfolio_pnl_on_known_basis": unrealized_portfolio_pnl,
            "open_swings": open_count,
            "return_pct": ((treasury_value / starting_value) - 1.0) * 100.0 if starting_value > 0 else 0.0,
        }

    def _swing_committed_quote(self, swing: dict[str, Any]) -> float:
        economics = self._swing_economics(
            swing,
            swing["executions"],
        )
        return calculate_sell_origin_committed_quote(
            swing,
            economics,
        )

    def _committed_quote_reserve(self) -> float:
        return sum(self._swing_committed_quote(swing) for swing in self.swings.values())

    def _free_reserve_quote(self) -> float:
        return max(0.0, self.usdt - self._committed_quote_reserve())

    def _spendable_free_reserve_quote(self, *, committed_quote: float | None = None) -> float:
        committed = self._committed_quote_reserve() if committed_quote is None else committed_quote
        free_quote = max(0.0, self.usdt - committed)
        retention_pct = min(100.0, max(0.0, self.scenario.free_cash_retention_pct))
        return free_quote * (1.0 - retention_pct / 100.0)

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
