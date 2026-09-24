from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from shared.spot_progression import ProgressiveSwingConfig
from shared.spot_signal import SpotSignalConfig
from shared.spot_sizing import IndicatorSizingConfig


VALID_OBJECTIVES = {None, "accumulate_cash", "accumulate_asset"}


def _utc_datetime(value: Any, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _number(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if numeric != numeric or (minimum is not None and numeric < minimum):
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return numeric


@dataclass(frozen=True)
class SpotHistorySegment:
    market_symbol: str
    start: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True)
class SpotBacktestAsset:
    symbol: str
    quantity: float
    entry_cost: float | None
    binance_quantity: float
    cold_storage_quantity: float
    unassigned_quantity: float
    target_holding: float
    minimum_holding_pct: float
    trading_objective: str | None
    symbol_info: dict[str, Any] | None = None
    history_segments: tuple[SpotHistorySegment, ...] = ()

    @property
    def market_symbol(self) -> str:
        return f"{self.symbol}USDT"


@dataclass(frozen=True)
class SpotBacktestExecutionConfig:
    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    force_close_at_end: bool = False


@dataclass(frozen=True)
class SpotBacktestScenario:
    name: str
    start: datetime
    end: datetime
    interval: str
    warmup_candles: int
    starting_usdt: float
    assets: tuple[SpotBacktestAsset, ...]
    execution: SpotBacktestExecutionConfig
    signal: SpotSignalConfig
    sizing: IndicatorSizingConfig
    progression: ProgressiveSwingConfig
    data_cache_dir: Path
    output_dir: Path
    near_floor_pct: float = 1.0
    strong_cash_utilization_pct: float = 80.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def asset_order(self) -> tuple[str, ...]:
        # Live policies are read in asset-symbol order by list_portfolio_asset_policies.
        return tuple(sorted(asset.symbol for asset in self.assets))


def _asset_from_mapping(symbol: str, payload: dict[str, Any]) -> SpotBacktestAsset:
    normalized_symbol = str(symbol or payload.get("symbol") or "").strip().upper()
    if not normalized_symbol or normalized_symbol == "USDT":
        raise ValueError("Backtest assets require a non-USDT symbol.")
    quantity = _number(payload.get("quantity"), field_name=f"{normalized_symbol}.quantity", minimum=0)
    custody = payload.get("custody") if isinstance(payload.get("custody"), dict) else {}
    binance_quantity = _number(
        custody.get("binance", payload.get("binance_quantity", 0)),
        field_name=f"{normalized_symbol}.custody.binance",
        minimum=0,
    )
    cold_quantity = _number(
        custody.get("cold_storage", payload.get("cold_storage_quantity", 0)),
        field_name=f"{normalized_symbol}.custody.cold_storage",
        minimum=0,
    )
    unassigned_quantity = _number(
        custody.get("unassigned", payload.get("unassigned_quantity", 0)),
        field_name=f"{normalized_symbol}.custody.unassigned",
        minimum=0,
    )
    custody_total = binance_quantity + cold_quantity + unassigned_quantity
    tolerance = max(1e-9, abs(quantity) * 1e-9)
    if abs(custody_total - quantity) > tolerance:
        raise ValueError(
            f"{normalized_symbol} custody totals {custody_total}, but starting quantity is {quantity}."
        )
    entry_cost_raw = payload.get("entry_cost")
    entry_cost = None if entry_cost_raw is None else _number(
        entry_cost_raw,
        field_name=f"{normalized_symbol}.entry_cost",
        minimum=0,
    )
    target = _number(
        payload.get("target_holding", quantity),
        field_name=f"{normalized_symbol}.target_holding",
        minimum=0,
    )
    minimum_pct = _number(
        payload.get("minimum_holding_pct", 100),
        field_name=f"{normalized_symbol}.minimum_holding_pct",
        minimum=0,
    )
    if minimum_pct > 100:
        raise ValueError(f"{normalized_symbol}.minimum_holding_pct cannot exceed 100.")
    objective_raw = payload.get("trading_objective")
    objective = str(objective_raw).strip().lower() if objective_raw is not None else None
    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"{normalized_symbol}.trading_objective must be ACCUMULATE_CASH, ACCUMULATE_ASSET, or null."
        )
    symbol_info = payload.get("symbol_info") if isinstance(payload.get("symbol_info"), dict) else None
    raw_segments = payload.get("history_segments")
    history_segments: tuple[SpotHistorySegment, ...] = ()
    if raw_segments is not None:
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"{normalized_symbol}.history_segments must be a non-empty list.")
        parsed_segments: list[SpotHistorySegment] = []
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                raise ValueError(f"{normalized_symbol}.history_segments[{index}] must be an object.")
            market_symbol = str(item.get("market_symbol") or "").strip().upper()
            if not market_symbol:
                raise ValueError(f"{normalized_symbol}.history_segments[{index}].market_symbol is required.")
            start = (
                _utc_datetime(item.get("start"), field_name=f"{normalized_symbol}.history_segments[{index}].start")
                if item.get("start") is not None
                else None
            )
            end = (
                _utc_datetime(item.get("end"), field_name=f"{normalized_symbol}.history_segments[{index}].end")
                if item.get("end") is not None
                else None
            )
            if start is not None and end is not None and end <= start:
                raise ValueError(f"{normalized_symbol}.history_segments[{index}] end must be after start.")
            parsed_segments.append(SpotHistorySegment(market_symbol=market_symbol, start=start, end=end))
        history_segments = tuple(parsed_segments)
    return SpotBacktestAsset(
        symbol=normalized_symbol,
        quantity=quantity,
        entry_cost=entry_cost,
        binance_quantity=binance_quantity,
        cold_storage_quantity=cold_quantity,
        unassigned_quantity=unassigned_quantity,
        target_holding=target,
        minimum_holding_pct=minimum_pct,
        trading_objective=objective,
        symbol_info=symbol_info,
        history_segments=history_segments,
    )


def load_spot_backtest_scenario(
    path: str | Path,
    *,
    start_override: str | None = None,
    end_override: str | None = None,
    preset: str | None = None,
    output_override: str | Path | None = None,
) -> SpotBacktestScenario:
    scenario_path = Path(path).expanduser().resolve()
    with scenario_path.open("r", encoding="utf-8") as file_obj:
        root = yaml.safe_load(file_obj)
    if not isinstance(root, dict):
        raise ValueError("Spot backtest scenario must contain a YAML object.")
    payload = root.get("spot_backtest") if isinstance(root.get("spot_backtest"), dict) else root
    end = _utc_datetime(end_override or payload.get("end"), field_name="spot_backtest.end")
    if start_override:
        start = _utc_datetime(start_override, field_name="spot_backtest.start")
    elif preset:
        normalized_preset = str(preset).strip().lower()
        if normalized_preset == "6m":
            start = end - timedelta(days=183)
        elif normalized_preset == "1y":
            start = end - timedelta(days=365)
        else:
            raise ValueError("Spot backtest preset must be 6m or 1y.")
    else:
        start = _utc_datetime(payload.get("start"), field_name="spot_backtest.start")
    if end <= start:
        raise ValueError("spot_backtest.end must be later than spot_backtest.start.")

    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, dict) or not raw_assets:
        raise ValueError("spot_backtest.assets must contain at least one asset.")
    assets = tuple(
        _asset_from_mapping(symbol, item)
        for symbol, item in raw_assets.items()
        if isinstance(item, dict)
    )
    if len(assets) != len(raw_assets):
        raise ValueError("Every Spot backtest asset must contain an object configuration.")
    if len({asset.symbol for asset in assets}) != len(assets):
        raise ValueError("Spot backtest asset symbols must be unique.")

    strategy = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
    signal_payload = strategy.get("signal") if isinstance(strategy.get("signal"), dict) else {}
    sizing_payload = strategy.get("indicator_sizing") if isinstance(strategy.get("indicator_sizing"), dict) else {}
    progression_payload = strategy.get("progression") if isinstance(strategy.get("progression"), dict) else {}
    execution_payload = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    fee_rate = _number(execution_payload.get("fee_rate", 0.001), field_name="execution.fee_rate", minimum=0)
    if fee_rate >= 1:
        raise ValueError("execution.fee_rate must be below 1.")

    base_dir = scenario_path.parent
    cache_path = Path(str(payload.get("data_cache_dir", "../data/spot_backtest"))).expanduser()
    output_path = Path(output_override or str(payload.get("output_dir", "../runtime/spot_backtests/latest"))).expanduser()
    if not cache_path.is_absolute():
        cache_path = (base_dir / cache_path).resolve()
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    return SpotBacktestScenario(
        name=str(payload.get("name") or scenario_path.stem),
        start=start,
        end=end,
        interval=str(payload.get("interval") or "1h").strip().lower(),
        warmup_candles=max(60, int(payload.get("warmup_candles", 60))),
        starting_usdt=_number(payload.get("starting_usdt", 0), field_name="starting_usdt", minimum=0),
        assets=assets,
        execution=SpotBacktestExecutionConfig(
            fee_rate=fee_rate,
            slippage_bps=_number(
                execution_payload.get("slippage_bps", 0),
                field_name="execution.slippage_bps",
                minimum=0,
            ),
            force_close_at_end=bool(execution_payload.get("force_close_at_end", False)),
        ),
        signal=SpotSignalConfig(
            levels_pct=tuple(signal_payload.get("levels_pct", (5, 8, 12, 18))),
            base_tranches_pct=tuple(signal_payload.get("base_tranches_pct", (10, 20, 30, 40))),
        ),
        sizing=IndicatorSizingConfig(
            weak_modifier=float(sizing_payload.get("weak_modifier", 0.5)),
            neutral_modifier=float(sizing_payload.get("neutral_modifier", 1.0)),
            strong_modifier=float(sizing_payload.get("strong_modifier", 1.25)),
            very_strong_modifier=float(sizing_payload.get("very_strong_modifier", 1.5)),
            rsi_oversold=float(sizing_payload.get("rsi_oversold", 30)),
            rsi_overbought=float(sizing_payload.get("rsi_overbought", 70)),
        ),
        progression=ProgressiveSwingConfig(
            close_profit_pct=float(progression_payload.get("close_profit_pct", 5)),
            estimated_fee_rate=float(progression_payload.get("estimated_fee_rate", fee_rate)),
        ),
        data_cache_dir=cache_path,
        output_dir=output_path,
        near_floor_pct=_number(payload.get("near_floor_pct", 1), field_name="near_floor_pct", minimum=0),
        strong_cash_utilization_pct=_number(
            payload.get("strong_cash_utilization_pct", 80),
            field_name="strong_cash_utilization_pct",
            minimum=0,
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def scenario_as_dict(scenario: SpotBacktestScenario) -> dict[str, Any]:
    payload = asdict(scenario)
    payload["start"] = scenario.start.isoformat()
    payload["end"] = scenario.end.isoformat()
    payload["data_cache_dir"] = str(scenario.data_cache_dir)
    payload["output_dir"] = str(scenario.output_dir)
    payload["assets"] = {
        asset.symbol: {
            "quantity": asset.quantity,
            "entry_cost": asset.entry_cost,
            "custody": {
                "binance": asset.binance_quantity,
                "cold_storage": asset.cold_storage_quantity,
                "unassigned": asset.unassigned_quantity,
            },
            "target_holding": asset.target_holding,
            "minimum_holding_pct": asset.minimum_holding_pct,
            "trading_objective": asset.trading_objective,
            "symbol_info": asset.symbol_info,
            "history_segments": [
                {
                    "market_symbol": segment.market_symbol,
                    "start": segment.start.isoformat() if segment.start is not None else None,
                    "end": segment.end.isoformat() if segment.end is not None else None,
                }
                for segment in asset.history_segments
            ] or None,
        }
        for asset in scenario.assets
    }
    return {"spot_backtest": payload}


def write_scenario_snapshot(scenario: SpotBacktestScenario, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(scenario_as_dict(scenario), file_obj, sort_keys=False)
    return target


def export_current_treasury_scenario(
    path: str | Path,
    *,
    start: str,
    end: str,
    name: str = "current-treasury-counterfactual",
) -> Path:
    from api.services.portfolio_service import load_portfolio_snapshot

    snapshot, snapshot_warnings = load_portfolio_snapshot()
    warnings = list(snapshot_warnings)
    assets: dict[str, Any] = {}
    starting_usdt = 0.0
    for holding in snapshot.get("holdings", []):
        if holding.get("is_dry_powder"):
            symbol = str(holding.get("asset_symbol") or "").strip().upper()
            if symbol == "USDT":
                starting_usdt += float(holding.get("quantity") or 0.0)
            else:
                warnings.append(
                    f"Excluded {symbol or 'unknown'} dry-powder balance; the V1 replay pool is USDT only."
                )
            continue
        symbol = str(holding["asset_symbol"]).upper()
        custody = holding.get("custody") if isinstance(holding.get("custody"), dict) else {}
        assets[symbol] = {
            "quantity": float(holding.get("quantity") or 0.0),
            "entry_cost": holding.get("average_cost"),
            "custody": {
                "binance": float((custody.get("binance") or {}).get("quantity") or 0.0),
                "cold_storage": float((custody.get("cold_storage") or {}).get("quantity") or 0.0),
                "unassigned": float((custody.get("unassigned") or {}).get("quantity") or 0.0),
            },
            "target_holding": (
                holding.get("target_quantity")
                if holding.get("target_quantity") is not None
                else float(holding.get("quantity") or 0.0)
            ),
            "minimum_holding_pct": (
                holding.get("minimum_holding_pct")
                if holding.get("minimum_holding_pct") is not None
                else 100
            ),
            "trading_objective": holding.get("trading_objective"),
        }
    payload = {
        "spot_backtest": {
            "name": name,
            "start": _utc_datetime(start, field_name="start").isoformat(),
            "end": _utc_datetime(end, field_name="end").isoformat(),
            "interval": "1h",
            "warmup_candles": 60,
            "starting_usdt": starting_usdt,
            "assets": assets,
            "execution": {"fee_rate": 0.001, "slippage_bps": 0, "force_close_at_end": False},
            "strategy": {
                "signal": {"levels_pct": [5, 8, 12, 18], "base_tranches_pct": [10, 20, 30, 40]},
                "indicator_sizing": {
                    "weak_modifier": 0.5,
                    "neutral_modifier": 1.0,
                    "strong_modifier": 1.25,
                    "very_strong_modifier": 1.5,
                },
                "progression": {"close_profit_pct": 5, "estimated_fee_rate": 0.001},
            },
            "data_cache_dir": "../data/spot_backtest",
            "output_dir": "../runtime/spot_backtests/latest",
            "metadata": {
                "assumption": "Counterfactual snapshot exported from the current Treasury.",
                "export_warnings": warnings,
            },
        }
    }
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)
    return target
