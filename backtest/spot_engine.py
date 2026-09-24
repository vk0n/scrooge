from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any

from backtest.spot_market_data import SpotCandle, SpotHistoricalDataset
from backtest.spot_scenario import SpotBacktestAsset, SpotBacktestScenario
from bot.spot_signal import calculate_spot_indicator_context
from shared.spot_execution_rules import normalize_market_quantity, validate_market_notional
from shared.spot_policy import calculate_spot_inventory_policy
from shared.spot_signal import ROLLING_WINDOW_MS, evaluate_rolling_24h_opportunity
from shared.spot_strategy import (
    finalize_spot_strategy_signal,
    plan_spot_strategy_action,
    transition_spot_strategy_campaign,
)
from shared.spot_swing import calculate_swing_economics, calculate_target_ratchet


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
        known_quantity = asset.quantity if asset.entry_cost is not None else 0.0
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
    target_history: list[dict[str, Any]]
    inventory_history: list[dict[str, Any]]
    rejections: list[dict[str, Any]]


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
        self.executions: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.target_history: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}
        self.ratcheted_swings: set[str] = set()
        self.execution_counter = 0
        self.start_ms = int(scenario.start.timestamp() * 1000)
        self.end_ms = int(scenario.end.timestamp() * 1000)
        self._validate_dataset()
        self._candle_by_close = {
            symbol: {row.close_time_ms: row for row in rows}
            for symbol, rows in self.dataset.candles.items()
        }
        self._candle_position_by_close = {
            symbol: {row.close_time_ms: index for index, row in enumerate(rows)}
            for symbol, rows in self.dataset.candles.items()
        }

    def run(self) -> SpotBacktestResult:
        replay_rows = self._replay_rows()
        initial_prices = {symbol: rows[0].open for symbol, rows in replay_rows.items()}
        final_prices = {symbol: rows[-1].close for symbol, rows in replay_rows.items()}
        starting_value = self.usdt + sum(
            self.assets[symbol].quantity * initial_prices[symbol]
            for symbol in self.scenario.asset_order
        )
        equity = [self._equity_point(self.start_ms, initial_prices, starting_value)]

        step_count = len(next(iter(replay_rows.values())))
        for step in range(step_count):
            candles = {symbol: replay_rows[symbol][step] for symbol in self.scenario.asset_order}
            self._execute_pending(candles)
            prices = {symbol: candle.close for symbol, candle in candles.items()}
            if step < step_count - 1:
                self._evaluate_cycle(candles, reserved_quote=0.0)
            equity.append(self._equity_point(candles[self.scenario.asset_order[0]].close_time_ms, prices, starting_value))
            self._record_inventory(candles[self.scenario.asset_order[0]].close_time_ms, prices)

        if self.scenario.execution.force_close_at_end:
            self._force_close(final_prices, self.end_ms)
            equity.append(self._equity_point(self.end_ms, final_prices, starting_value))

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
            target_history=list(self.target_history),
            inventory_history=list(self.inventory_history),
            rejections=list(self.rejections),
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

    def _execute_pending(self, candles: dict[str, SpotCandle]) -> None:
        pending = self.pending
        self.pending = {}
        for symbol in self.scenario.asset_order:
            action = pending.get(symbol)
            if action is not None:
                if not self._is_market_available(symbol, candles[symbol].open_time_ms):
                    self.rejections.append(
                        {
                            "timestamp_ms": candles[symbol].open_time_ms,
                            "timestamp": self._timestamp(candles[symbol].open_time_ms),
                            "asset_symbol": symbol,
                            "side": action["side"],
                            "action_type": action["action_type"],
                            "requested_quantity": action["requested_quantity"],
                            "reason": "Historical market unavailable during the declared symbol migration.",
                        }
                    )
                    continue
                self._execute_action(symbol, action, candles[symbol].open, candles[symbol].open_time_ms)

    def _evaluate_cycle(self, candles: dict[str, SpotCandle], *, reserved_quote: float) -> None:
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
                    "sizing_modifier": signal["sizing_modifier"],
                    "final_tranche_pct": signal["final_tranche_pct"],
                    "strategy_eligible": signal["strategy_eligible"],
                    "eligibility_reason": signal["eligibility_reason"],
                }
            )
            prior = self.campaigns.get(symbol)
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
            self.campaigns[symbol] = campaign
            state = self.assets[symbol]
            decision = plan_spot_strategy_action(
                signal,
                state.holding(candle.close),
                campaign,
                self._active_swing_states(symbol),
                available_quote=max(0.0, self.usdt - reserved_quote),
                config=self.scenario.progression,
                cleanup_config=self.scenario.waiter_cleanup,
            )
            if decision is None:
                continue
            if decision["action_type"] == "hold":
                self.actions.append(
                    {
                        "timestamp_ms": candle.close_time_ms,
                        "timestamp": self._timestamp(candle.close_time_ms),
                        "asset_symbol": symbol,
                        "action_type": "hold",
                        "side": None,
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
                continue
            pending = {
                **decision,
                "signal": signal,
                "campaign_id": campaign.get("campaign_id"),
                "decided_at_ms": candle.close_time_ms,
                "reference_price": signal.get("reference_price"),
            }
            if decision["action_type"] == "open":
                pending["swing_id"] = self._opening_swing_id(symbol, campaign, int(signal["level"]))
            self.pending[symbol] = pending
            if decision["side"] == "buy":
                reserved_quote += (
                    float(decision["requested_quantity"])
                    * candle.close
                    * (1.0 + self.scenario.execution.fee_rate)
                )

    def _is_market_available(self, symbol: str, open_time_ms: int) -> bool:
        return open_time_ms not in self.dataset.unavailable_open_times.get(symbol, frozenset())

    def _signal_for(self, symbol: str, candle: SpotCandle) -> dict[str, Any] | None:
        rows = self.dataset.candles[symbol]
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
            current_index = self._candle_position_by_close[symbol][candle.close_time_ms]
            history_size = max(60, self.scenario.warmup_candles)
            visible = [
                row.as_binance_kline()
                for row in rows[max(0, current_index - history_size + 1):current_index + 1]
            ]
            try:
                indicator_context = calculate_spot_indicator_context(
                    visible[-max(60, self.scenario.warmup_candles):],
                    current_price=candle.close,
                    evaluated_at_ms=candle.close_time_ms,
                    interval=self.scenario.interval,
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

    def _execute_action(self, symbol: str, action: dict[str, Any], observed_price: float, timestamp_ms: int) -> None:
        state = self.assets[symbol]
        side = str(action["side"])
        slippage = self.scenario.execution.slippage_bps / 10_000.0
        price = observed_price * (1.0 + slippage if side == "buy" else 1.0 - slippage)
        requested = float(action["requested_quantity"])
        constrained = requested
        if side == "buy":
            constrained = min(
                constrained,
                self.usdt / (price * (1.0 + self.scenario.execution.fee_rate)) if price > 0 else 0.0,
            )
        else:
            constrained = min(constrained, state.immediately_sellable)
        try:
            quantity_decimal, _ = normalize_market_quantity(
                self.dataset.symbol_info[symbol],
                constrained,
            )
            validate_market_notional(
                self.dataset.symbol_info[symbol],
                quantity=quantity_decimal,
                price=price,
            )
        except ValueError as exc:
            self.rejections.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": self._timestamp(timestamp_ms),
                    "asset_symbol": symbol,
                    "side": side,
                    "action_type": action["action_type"],
                    "requested_quantity": requested,
                    "reason": str(exc),
                }
            )
            return
        quantity = float(quantity_decimal)
        quote_quantity = quantity * price
        fee = quote_quantity * self.scenario.execution.fee_rate
        if side == "buy" and quote_quantity + fee > self.usdt + 1e-9:
            return

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
        swing = self.swings.get(swing_id)
        if swing is None:
            return

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
        economics = calculate_swing_economics(swing, swing["executions"], current_price=price)
        swing["status"] = economics["status"]
        if economics["status"] == "closed":
            swing["closed_at_ms"] = timestamp_ms
            swing["close_reason"] = (action.get("reason") or {}).get("close_reason")
            swing["close_context"] = dict(action.get("reason") or {})
            self._apply_target_ratchet(state, swing, economics, timestamp_ms)
        if action["action_type"] == "open":
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

    def _active_swing_states(self, symbol: str) -> list[dict[str, Any]]:
        return [
            {"swing": swing, "executions": swing["executions"]}
            for swing in self.swings.values()
            if swing["asset_symbol"] == symbol and swing["status"] in {"open", "partially_closed"}
        ]

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
            economics = calculate_swing_economics(
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
        reserved_quote = self._pending_quote_reserve(prices)
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

    def _pending_quote_reserve(self, prices: dict[str, float]) -> float:
        return sum(
            float(action["requested_quantity"])
            * prices[symbol]
            * (1.0 + self.scenario.execution.fee_rate)
            for symbol, action in self.pending.items()
            if action["side"] == "buy"
        )

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
                economics = calculate_swing_economics(
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
            economics = calculate_swing_economics(
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
                economics = calculate_swing_economics(
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
