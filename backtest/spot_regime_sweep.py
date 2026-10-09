from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import multiprocessing
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time
from typing import Any

from tqdm import tqdm
import yaml

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter, SpotHistoricalDataset
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_runner import SpotMarketDataProgressBars, SpotReplayProgress
from backtest.spot_scenario import SpotBacktestScenario, scenario_as_dict
from backtest.spot_sweep import (
    SpotSweepConfig,
    SpotSweepVariant,
    _read_json,
    _resolved_scenario_matches,
    _scenario_for_variant,
    _summary_row,
    _write_csv,
    _write_json,
    load_spot_sweep_config,
)


REGIME_ORDER = ("bull", "neutral", "bear")


@dataclass(frozen=True)
class SpotRegime:
    name: str
    start: datetime
    end: datetime
    selection: dict[str, Any]


@dataclass(frozen=True)
class SpotRegimeSweepConfig:
    sweep: SpotSweepConfig
    regimes: tuple[SpotRegime, ...]
    selection_methodology: str
    common_history_start: datetime | None = None
    common_history_end: datetime | None = None

    @property
    def output_dir(self) -> Path:
        return self.sweep.output_dir


_WORKER_CONFIG: SpotRegimeSweepConfig | None = None
_WORKER_DATASET: SpotHistoricalDataset | None = None


def _utc_datetime(value: Any, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_spot_regime_sweep_config(
    path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> SpotRegimeSweepConfig:
    sweep = load_spot_sweep_config(path, output_override=output_override)
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file_obj:
        root = yaml.safe_load(file_obj)
    payload = root.get("spot_sweep") if isinstance(root, dict) else None
    if not isinstance(payload, dict):
        raise ValueError("Spot regime sweep config requires spot_sweep.")
    raw_regimes = payload.get("regimes")
    if not isinstance(raw_regimes, dict):
        raise ValueError("spot_sweep.regimes must define Bull, Neutral, and Bear.")
    normalized = {str(key).strip().lower(): value for key, value in raw_regimes.items()}
    if set(normalized) != set(REGIME_ORDER):
        raise ValueError("spot_sweep.regimes must contain exactly bull, neutral, and bear.")

    regimes: list[SpotRegime] = []
    for name in REGIME_ORDER:
        raw = normalized[name]
        if not isinstance(raw, dict):
            raise ValueError(f"spot_sweep.regimes.{name} must be an object.")
        start = _utc_datetime(raw.get("start"), field_name=f"regimes.{name}.start")
        end = _utc_datetime(raw.get("end"), field_name=f"regimes.{name}.end")
        if end <= start:
            raise ValueError(f"Regime {name} must end after it starts.")
        regimes.append(
            SpotRegime(
                name=name,
                start=start,
                end=end,
                selection=dict(raw.get("selection") or {}),
            )
        )

    chronological = sorted(regimes, key=lambda item: item.start)
    for previous, current in zip(chronological, chronological[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"Regime periods overlap: {previous.name} and {current.name}. "
                "Freeze non-overlapping periods for this experiment."
            )

    selection = payload.get("regime_selection") or {}
    if not isinstance(selection, dict):
        raise ValueError("spot_sweep.regime_selection must be an object.")
    common = selection.get("common_history") or {}
    if common and not isinstance(common, dict):
        raise ValueError("regime_selection.common_history must be an object.")
    return SpotRegimeSweepConfig(
        sweep=sweep,
        regimes=tuple(regimes),
        selection_methodology=str(
            selection.get("methodology")
            or "Frozen non-overlapping 365-day windows selected from monthly candidates using untouched HODL portfolio behavior only."
        ),
        common_history_start=(
            _utc_datetime(common.get("start"), field_name="common_history.start")
            if common.get("start") is not None
            else None
        ),
        common_history_end=(
            _utc_datetime(common.get("end"), field_name="common_history.end")
            if common.get("end") is not None
            else None
        ),
    )


def _scenario_for_regime(
    config: SpotRegimeSweepConfig,
    variant: SpotSweepVariant,
    regime: SpotRegime,
) -> SpotBacktestScenario:
    scenario = _scenario_for_variant(config.sweep, variant)
    metadata = {
        **scenario.metadata,
        "regime": regime.name,
        "regime_selection": regime.selection,
        "regime_selection_methodology": config.selection_methodology,
    }
    return replace(
        scenario,
        name=f"{config.sweep.name}-{variant.name}-{regime.name}",
        start=regime.start,
        end=regime.end,
        output_dir=config.output_dir / "candidates" / variant.name / regime.name,
        metadata=metadata,
    )


def _market_data_scenario(config: SpotRegimeSweepConfig) -> SpotBacktestScenario:
    return replace(
        config.sweep.scenario,
        start=min(item.start for item in config.regimes),
        end=max(item.end for item in config.regimes),
    )


def _maximum_drawdown(values: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value / peak - 1.0) * 100.0)
    return worst


def _hodl_metrics(
    dataset: SpotHistoricalDataset,
    scenario: SpotBacktestScenario,
    regime: SpotRegime,
) -> dict[str, Any]:
    start_ms = int(regime.start.timestamp() * 1000)
    end_ms = int(regime.end.timestamp() * 1000)
    rows: dict[str, tuple[Any, ...]] = {}
    starts: dict[str, int] = {}
    count = (end_ms - start_ms) // dataset.interval_ms
    for symbol in scenario.asset_order:
        candles = dataset.candles[symbol]
        index = bisect_left(candles, start_ms, key=lambda item: item.open_time_ms)
        if index >= len(candles) or candles[index].open_time_ms != start_ms:
            raise ValueError(f"{symbol} does not cover frozen {regime.name} start.")
        if index + count > len(candles):
            raise ValueError(f"{symbol} does not cover frozen {regime.name} end.")
        rows[symbol] = candles
        starts[symbol] = index

    quantity = {asset.symbol: asset.quantity for asset in scenario.assets}
    start_value = scenario.starting_usdt + sum(
        quantity[symbol] * rows[symbol][starts[symbol]].open
        for symbol in scenario.asset_order
    )
    final_value = scenario.starting_usdt + sum(
        quantity[symbol] * rows[symbol][starts[symbol] + count - 1].close
        for symbol in scenario.asset_order
    )
    values = []
    for offset in range(count):
        values.append(
            scenario.starting_usdt
            + sum(
                quantity[symbol] * rows[symbol][starts[symbol] + offset].close
                for symbol in scenario.asset_order
            )
        )
    return {
        "start": regime.start.isoformat(),
        "end": regime.end.isoformat(),
        "starting_value": start_value,
        "final_value": final_value,
        "return_pct": (final_value / start_value - 1.0) * 100.0,
        "maximum_drawdown_pct": _maximum_drawdown(values),
        "per_asset_market_return_pct": {
            symbol: (
                rows[symbol][starts[symbol] + count - 1].close
                / rows[symbol][starts[symbol]].open
                - 1.0
            )
            * 100.0
            for symbol in scenario.asset_order
        },
    }


def _value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    return default if value is None else float(value)


def _regime_summary(
    *,
    variant: SpotSweepVariant,
    regime: SpotRegime,
    scenario: SpotBacktestScenario,
    report: dict[str, Any],
    duration_seconds: float,
    resumed: bool,
) -> dict[str, Any]:
    row = _summary_row(
        variant=variant,
        scenario=scenario,
        report=report,
        duration_seconds=duration_seconds,
        resumed=resumed,
    )
    portfolio = report["portfolio"]
    reserve = report["success_metrics"]["free_reserve"]
    recovery = report["success_metrics"]["asset_recovery"]
    swings = report["swings"]
    analysis = (report.get("bargain_analysis") or {}).get("overview") or {}
    cleanup = report["waiter_cleanup"]
    accounting = cleanup.get("accounting") or {}
    accumulation = report["treasury_accumulation"]
    accumulation_overview = accumulation.get("overview") or {}
    by_reason = cleanup.get("by_reason") or {}
    relative = portfolio.get("relative_wealth") or {}
    per_asset = report.get("per_asset") or {}
    row.update(
        {
            "regime": regime.name,
            "start": regime.start.isoformat(),
            "end": regime.end.isoformat(),
            "starting_treasury_value": _value(portfolio, "starting_treasury_value"),
            "strategy_return_pct": _value(portfolio, "total_return_pct"),
            "hodl_return_pct": _value(portfolio, "hodl_return_pct"),
            "hodl_maximum_drawdown_pct": _value(
                portfolio, "maximum_hodl_drawdown_pct"
            ),
            "weighted_effective_assets_pct": _value(
                recovery,
                "weighted_effective_quantity_pct",
                _value(recovery, "average_effective_quantity_pct"),
            ),
            "minimum_relative_wealth": _value(relative, "minimum", 1.0),
            "maximum_relative_wealth": _value(relative, "maximum", 1.0),
            "terminal_relative_wealth": _value(relative, "terminal", 1.0),
            "maximum_relative_drawdown_pct": _value(
                relative, "maximum_drawdown_pct"
            ),
            "committed_reserve_quote": _value(reserve, "committed_quote"),
            "closure_rate_pct": _value(analysis, "closure_rate_pct"),
            "win_rate_pct": _value(analysis, "closed_win_rate_pct"),
            "profit_factor": (
                float(analysis["profit_factor"])
                if analysis.get("profit_factor") is not None
                else None
            ),
            "closed_bargains": int(swings.get("total_closed") or 0),
            "lifecycle_pnl": _value(swings, "realized_pnl_quote")
            + _value(swings, "unrealized_open_pnl_quote"),
            "median_duration_hours": _value(swings, "median_duration_hours"),
            "p90_duration_hours": _value(analysis, "duration_p90_hours"),
            "maximum_simultaneous_bargains": int(
                swings.get("maximum_concurrent") or 0
            ),
            "fee_drag_pct": _value(analysis, "fee_drag_pct"),
            "quote_fees": _value(analysis, "quote_fees"),
            "total_fills": int(swings.get("total_fills") or 0)
            + int(accumulation_overview.get("count") or 0),
            "accumulation_buy_count": int(accumulation_overview.get("count") or 0),
            "accumulation_cash_deployed": _value(
                accumulation_overview, "usdt_deployed"
            ),
            "accumulation_net_asset_quantity": _value(
                accumulation_overview, "net_asset_acquired"
            ),
            "accumulation_target_growth": _value(
                accumulation_overview, "target_growth_quantity"
            ),
            "cleanup_reserve_deployed": _value(
                accounting, "reserve_deployed_quote"
            ),
            "cleanup_buy_turnover": _value(
                accounting, "repurchase_spend_quote"
            ),
            "deep_loss_closes": int(
                (by_reason.get("deep_loss_cleanup") or {}).get("closes") or 0
            ),
            "age_l3_closes": int(
                (by_reason.get("age_l3_cleanup") or {}).get("closes") or 0
            ),
            "age_l2_closes": int(
                (by_reason.get("age_l2_cleanup") or {}).get("closes") or 0
            ),
            "age_l1_closes": int(
                (by_reason.get("age_l1_cleanup") or {}).get("closes") or 0
            ),
            "capacity_cleanup_closes": int(
                (by_reason.get("capacity_cleanup") or {}).get("closes") or 0
            ),
            "per_asset": {
                symbol: {
                    "market_return_pct": _value(item.get("market") or {}, "market_return_pct"),
                    "objective": (item.get("starting") or {}).get("trading_objective"),
                    "starting_quantity": _value(item.get("starting") or {}, "quantity"),
                    "final_quantity": _value(item.get("final") or {}, "quantity"),
                    "target": _value(item.get("final") or {}, "target_holding"),
                    "effective_quantity_pct": _value(
                        item.get("asset_recovery") or {},
                        "effective_final_quantity_pct",
                    ),
                    "bargain_result": _value(item.get("trading") or {}, "realized_pnl_quote")
                    + _value(item.get("trading") or {}, "unrealized_open_pnl_quote"),
                    "cash_contribution": _value(
                        item.get("objective_metrics") or {},
                        "realized_quote_cash_generated",
                    ),
                    "open_risk": _value(
                        item.get("trading") or {}, "unrealized_open_pnl_quote"
                    ),
                    "asset_value_vs_hodl_contribution": (
                        _value(item.get("benchmark") or {}, "scrooge_asset_value")
                        - _value(item.get("benchmark") or {}, "hodl_value")
                    ),
                    "opportunity_cost_quote": max(
                        0.0,
                        _value(item.get("benchmark") or {}, "hodl_value")
                        - _value(item.get("benchmark") or {}, "scrooge_asset_value"),
                    ),
                    "quantity_sold_not_restored": _value(
                        item.get("bad_cases") or {}, "quantity_sold_not_restored"
                    ),
                }
                for symbol, item in per_asset.items()
            },
            "accumulation_per_asset": accumulation.get("per_asset") or {},
        }
    )
    return row


def _parameters(variant: SpotSweepVariant) -> dict[str, Any]:
    return {
        "levels_pct": list(variant.levels_pct),
        "close_profit_pct": variant.close_profit_pct,
        "free_cash_retention_pct": variant.free_cash_retention_pct,
        "unrealized_pnl_pct": variant.unrealized_pnl_pct,
        "deep_loss_min_age_days": variant.deep_loss_min_age_days,
        "deep_loss_required_reverse_level": variant.deep_loss_required_reverse_level,
        "max_open_bargains_per_asset": variant.max_open_bargains_per_asset,
        "age_l3_min_age_days": variant.age_l3_min_age_days,
        "age_l2_min_age_days": variant.age_l2_min_age_days,
        "age_l1_min_age_days": variant.age_l1_min_age_days,
        "capacity_cleanup_min_age_days": variant.capacity_cleanup_min_age_days,
    }


def _combined_summary(
    variant: SpotSweepVariant,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_regime = {str(row["regime"]): row for row in rows}
    if set(by_regime) != set(REGIME_ORDER):
        raise ValueError(f"{variant.name} does not contain all three regime results.")
    edges = [float(by_regime[name]["edge_vs_hodl_pct_points"]) for name in REGIME_ORDER]
    recoveries = [
        float(by_regime[name]["weighted_effective_assets_pct"])
        for name in REGIME_ORDER
    ]
    mean_edge = mean(edges)
    best_name = max(REGIME_ORDER, key=lambda name: by_regime[name]["edge_vs_hodl_pct_points"])
    spread = max(edges) - min(edges)
    diagnostic = "robust" if min(edges) >= 0 else (
        f"{best_name}_specialist" if spread >= 10 and max(edges) - mean_edge >= 5 else "mixed"
    )
    output: dict[str, Any] = {
        "rank": None,
        "name": variant.name,
        "status": "ok",
        "parameters": _parameters(variant),
        **_parameters(variant),
        "regimes": by_regime,
        "mean_edge_vs_hodl_pp": mean_edge,
        "median_edge_vs_hodl_pp": median(edges),
        "worst_regime_edge_vs_hodl_pp": min(edges),
        "best_regime_edge_vs_hodl_pp": max(edges),
        "regimes_beating_hodl": sum(value > 0 for value in edges),
        "mean_weighted_effective_assets_pct": mean(recoveries),
        "worst_weighted_effective_assets_pct": min(recoveries),
        "mean_fee_drag_pct": mean(
            float(by_regime[name]["fee_drag_pct"]) for name in REGIME_ORDER
        ),
        "total_fills": sum(int(by_regime[name]["total_fills"]) for name in REGIME_ORDER),
        "duration_seconds": sum(
            float(by_regime[name]["duration_seconds"]) for name in REGIME_ORDER
        ),
        "diagnostic": diagnostic,
    }
    for name in REGIME_ORDER:
        row = by_regime[name]
        output[f"{name}_edge_pp"] = row["edge_vs_hodl_pct_points"]
        output[f"{name}_weighted_asset_recovery"] = row[
            "weighted_effective_assets_pct"
        ]
        output[f"{name}_max_relative_drawdown"] = row[
            "maximum_relative_drawdown_pct"
        ]
        output[f"{name}_cleanup_loss"] = row["cleanup_lifecycle_pnl"]
        output[f"{name}_retained_cash"] = row["retained_reserve_quote"]
    return output


def rank_combined_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    succeeded = [dict(row) for row in rows if row.get("status") == "ok"]
    succeeded.sort(
        key=lambda row: (
            -float(row["mean_edge_vs_hodl_pp"]),
            -float(row["worst_regime_edge_vs_hodl_pp"]),
            -int(row["regimes_beating_hodl"]),
            -float(row["worst_weighted_effective_assets_pct"]),
            float(row["mean_fee_drag_pct"]),
        )
    )
    for rank, row in enumerate(succeeded, start=1):
        row["rank"] = rank
    return succeeded + [dict(row) for row in rows if row.get("status") != "ok"]


def _regime_winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("status") == "ok"]
    return {
        name: max(succeeded, key=lambda row: float(row[f"{name}_edge_pp"]))
        if succeeded
        else None
        for name in REGIME_ORDER
    }


def _stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [row for row in rows if row.get("status") == "ok"]
    if not ranked:
        return {"winner": None, "neighbors": []}
    winner = ranked[0]
    neighbors = []
    for row in ranked[1:]:
        parameter_keys = set(winner["parameters"]) | set(row["parameters"])
        differences = sum(
            row["parameters"].get(key) != winner["parameters"].get(key)
            for key in parameter_keys
        )
        if differences == 1:
            neighbors.append(
                {
                    "name": row["name"],
                    "mean_edge_vs_hodl_pp": row["mean_edge_vs_hodl_pp"],
                    "worst_regime_edge_vs_hodl_pp": row[
                        "worst_regime_edge_vs_hodl_pp"
                    ],
                    "within_one_pp": abs(
                        float(row["mean_edge_vs_hodl_pp"])
                        - float(winner["mean_edge_vs_hodl_pp"])
                    )
                    <= 1.0,
                    "parameters": row["parameters"],
                }
            )
    return {"winner": winner["name"], "neighbors": neighbors}


def _comparison_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scrooge Spot Regime Sweep</title><style>
:root{{--bg:#080c11;--panel:#111821;--line:#293647;--ink:#edf2f7;--muted:#8d9bad;--gold:#e8b84a;--green:#48d6a2;--red:#ff647f}}
*{{box-sizing:border-box}}body{{margin:0;background:repeating-linear-gradient(135deg,#080c11,#080c11 12px,#0a0f15 12px,#0a0f15 24px);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Menlo,monospace}}main{{width:min(1540px,calc(100% - 28px));margin:28px auto 60px}}header,.panel{{border:1px solid var(--line);border-radius:18px;background:rgba(17,24,33,.97)}}header{{padding:28px;background:linear-gradient(125deg,#351020,#111923);margin-bottom:18px}}h1{{margin:8px 0;font-size:34px}}h2{{color:var(--gold)}}p{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{padding:15px;border:1px solid var(--line);border-radius:12px}}.panel{{padding:18px;overflow:auto;margin-top:18px}}table{{width:100%;border-collapse:collapse;min-width:1120px}}th,td{{padding:10px;border-top:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted)}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}canvas{{width:100%;height:300px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}h1{{font-size:26px}}}}
</style></head><body><main><header><span style="color:var(--gold)">SCROOGE RESEARCH / CROSS-REGIME SWEEP</span><h1 id="title"></h1><p id="meta"></p></header><section class="panel"><h2>Robust Winner</h2><div class="grid" id="winner"></div><table><thead><tr><th>Metric</th><th>Bull</th><th>Neutral</th><th>Bear</th><th>Mean / Worst</th></tr></thead><tbody id="winnerMetrics"></tbody></table></section><section class="panel"><h2>Top Configurations</h2><table><thead><tr><th>Config</th><th>Mean Edge</th><th>Worst Edge</th><th>Beat HODL</th><th>Bull</th><th>Neutral</th><th>Bear</th><th>Worst Assets</th><th>Fees</th><th>Diagnostic</th></tr></thead><tbody id="rows"></tbody></table></section><section class="panel"><h2>Mean Edge vs Worst-Regime Edge</h2><canvas id="scatter" width="1400" height="300"></canvas></section><section class="panel"><h2>Retention vs Edge by Regime</h2><canvas id="retention" width="1400" height="300"></canvas></section><section class="panel"><h2>Regime Winners</h2><div class="grid" id="regimeWinners"></div></section><section class="panel"><h2>Parameter Stability</h2><p id="stability"></p></section></main>
<script id="data" type="application/json">{data}</script><script>
const d=JSON.parse(document.getElementById('data').textContent),pct=v=>`${{v>=0?'+':''}}${{Number(v).toFixed(2)}}%`,tone=v=>v>0?'positive':v<0?'negative':'';
document.getElementById('title').textContent=d.name;document.getElementById('meta').textContent=`${{d.completed}}/${{d.total}} candidates · ${{d.completed*3}}/${{d.total*3}} regime replays`;
const w=d.rows.find(r=>r.status==='ok');document.getElementById('winner').innerHTML=['bull','neutral','bear'].map(k=>{{const r=w.regimes[k];return `<article class="card"><strong>${{k.toUpperCase()}}</strong><h2 class="${{tone(r.edge_vs_hodl_pct_points)}}">${{pct(r.edge_vs_hodl_pct_points)}} edge</h2><p>Strategy ${{pct(r.strategy_return_pct)}} · HODL ${{pct(r.hodl_return_pct)}}<br>Assets ${{pct(r.weighted_effective_assets_pct)}} · Relative DD ${{pct(r.maximum_relative_drawdown_pct)}}</p></article>`}}).join('');
const money=v=>`${{v<0?'-':''}}$${{Math.abs(Number(v)).toLocaleString('en-US',{{maximumFractionDigits:2}})}}`,reg=['bull','neutral','bear'],metricRows=[['Scrooge Return','strategy_return_pct',pct,'mean'],['HODL Return','hodl_return_pct',pct,'mean'],['Edge vs HODL','edge_vs_hodl_pct_points',pct,'worst'],['Max DD','maximum_drawdown_pct',pct,'worst'],['HODL Max DD','hodl_maximum_drawdown_pct',pct,'worst'],['Relative DD','maximum_relative_drawdown_pct',pct,'worst'],['Weighted Asset Qty','weighted_effective_assets_pct',pct,'worst'],['Final Reserve','free_reserve_quote',money,'mean'],['Lifecycle PnL','lifecycle_pnl',money,'mean'],['Cleanup Lifecycle PnL','cleanup_lifecycle_pnl',money,'mean'],['Quote Fees','quote_fees',money,'mean'],['Bargains','opened_bargains',v=>Number(v).toLocaleString(),'mean'],['Win Rate','win_rate_pct',pct,'mean'],['Profit Factor','profit_factor',v=>v==null?'N/A':Number(v).toFixed(2),'mean']];document.getElementById('winnerMetrics').innerHTML=metricRows.map(([label,key,fmt,mode])=>{{const vs=reg.map(k=>w.regimes[k][key]),aggregate=mode==='worst'?Math.min(...vs.map(Number)):vs.reduce((a,v)=>a+Number(v||0),0)/vs.length;return `<tr><td>${{label}}</td>${{vs.map(v=>`<td>${{fmt(v)}}</td>`).join('')}}<td>${{fmt(aggregate)}}</td></tr>`}}).join('');
document.getElementById('rows').innerHTML=d.rows.filter(r=>r.status==='ok').slice(0,50).map(r=>`<tr><td>#${{r.rank}} ${{r.name}}</td><td class="${{tone(r.mean_edge_vs_hodl_pp)}}">${{pct(r.mean_edge_vs_hodl_pp)}}</td><td class="${{tone(r.worst_regime_edge_vs_hodl_pp)}}">${{pct(r.worst_regime_edge_vs_hodl_pp)}}</td><td>${{r.regimes_beating_hodl}}/3</td><td>${{pct(r.bull_edge_pp)}}</td><td>${{pct(r.neutral_edge_pp)}}</td><td>${{pct(r.bear_edge_pp)}}</td><td>${{pct(r.worst_weighted_effective_assets_pct)}}</td><td>${{pct(r.mean_fee_drag_pct)}}</td><td>${{r.diagnostic}}</td></tr>`).join('');
document.getElementById('regimeWinners').innerHTML=['bull','neutral','bear'].map(k=>{{const r=d.regime_winners[k];return `<article class="card"><strong>${{k.toUpperCase()}}</strong><h2>${{r.name}}</h2><p>${{pct(r[k+'_edge_pp'])}} edge · overall rank #${{r.rank}}</p></article>`}}).join('');
const n=d.stability.neighbors,flat=n.filter(x=>x.within_one_pp);document.getElementById('stability').textContent=`${{n.length}} one-dimension neighbors inspected; ${{flat.length}} remain within 1 pp of the winner.`;
const c=document.getElementById('scatter'),x=c.getContext('2d'),rs=d.rows.filter(r=>r.status==='ok');x.fillStyle='#0b1118';x.fillRect(0,0,c.width,c.height);const xs=rs.map(r=>r.mean_edge_vs_hodl_pp),ys=rs.map(r=>r.worst_regime_edge_vs_hodl_pp),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),sx=v=>40+(v-loX)/(hiX-loX||1)*(c.width-80),sy=v=>c.height-30-(v-loY)/(hiY-loY||1)*(c.height-60);rs.forEach((r,i)=>{{x.fillStyle=i===0?'#e8b84a':'#48d6a2';x.beginPath();x.arc(sx(r.mean_edge_vs_hodl_pp),sy(r.worst_regime_edge_vs_hodl_pp),i===0?6:3,0,Math.PI*2);x.fill()}});
const rc=document.getElementById('retention'),rx=rc.getContext('2d'),colors={{bull:'#e8b84a',neutral:'#7eb6ed',bear:'#ff647f'}},rets=rs.map(r=>r.free_cash_retention_pct),allEdges=rs.flatMap(r=>reg.map(k=>r[k+'_edge_pp'])),rlo=Math.min(...rets),rhi=Math.max(...rets),elo=Math.min(...allEdges),ehi=Math.max(...allEdges),rsx=v=>40+(v-rlo)/(rhi-rlo||1)*(rc.width-80),rsy=v=>rc.height-30-(v-elo)/(ehi-elo||1)*(rc.height-60);rx.fillStyle='#0b1118';rx.fillRect(0,0,rc.width,rc.height);rs.forEach(r=>reg.forEach(k=>{{rx.fillStyle=colors[k];rx.globalAlpha=.55;rx.beginPath();rx.arc(rsx(r.free_cash_retention_pct),rsy(r[k+'_edge_pp']),3,0,Math.PI*2);rx.fill()}}));rx.globalAlpha=1;
</script></body></html>"""


def _write_comparison(
    config: SpotRegimeSweepConfig,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = rank_combined_rows(rows)
    payload = {
        "name": config.sweep.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(config.sweep.config_path),
        "base_config": str(config.sweep.base_config_path),
        "total": len(config.sweep.variants),
        "total_scenario_replays": len(config.sweep.variants) * len(config.regimes),
        "completed": sum(row.get("status") == "ok" for row in ranked),
        "failed": sum(row.get("status") == "failed" for row in ranked),
        "rows": ranked,
        "regime_winners": _regime_winners(ranked),
        "stability": _stability(ranked),
    }
    _write_json(config.output_dir / "comparison.json", payload)
    csv_rows = [{key: value for key, value in row.items() if key != "regimes"} for row in ranked]
    _write_csv(config.output_dir / "comparison.csv", csv_rows)
    (config.output_dir / "report.html").write_text(
        _comparison_html(payload), encoding="utf-8"
    )
    (config.output_dir / "report.md").write_text(
        _comparison_markdown(payload), encoding="utf-8"
    )
    return payload


def _comparison_markdown(payload: dict[str, Any]) -> str:
    winner = next(
        (row for row in payload["rows"] if row.get("status") == "ok"),
        None,
    )
    if winner is None:
        return f"# {payload['name']}\n\nNo successful candidates yet.\n"
    lines = [
        f"# {payload['name']}",
        "",
        "## Robust Winner",
        "",
        f"**{winner['name']}**",
        "",
        f"- Mean Edge vs HODL: {winner['mean_edge_vs_hodl_pp']:+.2f} pp",
        f"- Worst-regime Edge vs HODL: {winner['worst_regime_edge_vs_hodl_pp']:+.2f} pp",
        f"- Regimes beating HODL: {winner['regimes_beating_hodl']}/3",
        f"- Mean weighted effective asset quantity: {winner['mean_weighted_effective_assets_pct']:.2f}%",
        f"- Worst weighted effective asset quantity: {winner['worst_weighted_effective_assets_pct']:.2f}%",
        "",
        "| Metric | Bull | Neutral | Bear |",
        "|---|---:|---:|---:|",
    ]
    metrics = (
        ("Scrooge Return", "strategy_return_pct", "%"),
        ("HODL Return", "hodl_return_pct", "%"),
        ("Edge vs HODL", "edge_vs_hodl_pct_points", " pp"),
        ("Max DD", "maximum_drawdown_pct", "%"),
        ("HODL Max DD", "hodl_maximum_drawdown_pct", "%"),
        ("Relative DD", "maximum_relative_drawdown_pct", "%"),
        ("Weighted Asset Qty", "weighted_effective_assets_pct", "%"),
        ("Retained Cash", "retained_reserve_quote", " USD"),
        ("Cleanup Lifecycle PnL", "cleanup_lifecycle_pnl", " USD"),
        ("Fee Drag", "fee_drag_pct", "%"),
    )
    for label, key, suffix in metrics:
        values = [float(winner["regimes"][name].get(key) or 0.0) for name in REGIME_ORDER]
        lines.append(
            f"| {label} | {values[0]:+.2f}{suffix} | {values[1]:+.2f}{suffix} | {values[2]:+.2f}{suffix} |"
        )
    lines.extend(
        [
            "",
            "## Regime Winners",
            "",
            *[
                f"- {name.title()}: {payload['regime_winners'][name]['name']} "
                f"({payload['regime_winners'][name][name + '_edge_pp']:+.2f} pp)"
                for name in REGIME_ORDER
            ],
            "",
            "## Parameter Stability",
            "",
            f"Inspected {len(payload['stability']['neighbors'])} one-dimension neighbors; "
            f"{sum(bool(item['within_one_pp']) for item in payload['stability']['neighbors'])} "
            "remain within 1 pp of the robust winner.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_state(
    config: SpotRegimeSweepConfig,
    rows_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _write_json(
        config.output_dir / "manifest.json",
        {
            "name": config.sweep.name,
            "config": str(config.sweep.config_path),
            "updated_at": datetime.now(UTC).isoformat(),
            "rows": list(rows_by_name.values()),
        },
    )
    return _write_comparison(config, list(rows_by_name.values()))


def _git_revision(config: SpotRegimeSweepConfig) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.sweep.config_path.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_experiment_manifest(
    config: SpotRegimeSweepConfig,
    dataset: SpotHistoricalDataset,
) -> dict[str, Any]:
    scenario = config.sweep.scenario
    hodl = {
        regime.name: _hodl_metrics(dataset, scenario, regime)
        for regime in config.regimes
    }
    snapshot = scenario_as_dict(scenario)["spot_backtest"]
    payload = {
        "name": config.sweep.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(config),
        "asset_universe": list(scenario.asset_order),
        "resolution": scenario.interval,
        "common_history": {
            "start": config.common_history_start.isoformat()
            if config.common_history_start
            else None,
            "end": config.common_history_end.isoformat()
            if config.common_history_end
            else None,
        },
        "regime_selection_methodology": config.selection_methodology,
        "regimes": {
            regime.name: {
                "start": regime.start.isoformat(),
                "end": regime.end.isoformat(),
                "selection": regime.selection,
                "hodl": hodl[regime.name],
            }
            for regime in config.regimes
        },
        "starting_portfolio": {
            asset.symbol: asset.quantity for asset in scenario.assets
        },
        "minimum_holding_pct": {
            asset.symbol: asset.minimum_holding_pct for asset in scenario.assets
        },
        "objectives": {
            asset.symbol: asset.trading_objective for asset in scenario.assets
        },
        "fixed_strategy": snapshot["strategy"],
        "execution": snapshot["execution"],
        "starting_usdt": scenario.starting_usdt,
        "swept_parameters": [_parameters(item) for item in config.sweep.variants],
        "candidate_count": len(config.sweep.variants),
        "scenario_replay_count": len(config.sweep.variants) * len(config.regimes),
        "market_data_quality": {
            "unavailable_candles_by_asset": {
                symbol: len(open_times)
                for symbol, open_times in dataset.unavailable_open_times.items()
            },
            "unavailable_behavior": (
                "Market-wide missing candles are valued flat at the previous close; "
                "signals and executions are disabled for those timestamps."
            ),
        },
    }
    _write_json(config.output_dir / "experiment_manifest.json", payload)
    return payload


def _run_candidate(
    config: SpotRegimeSweepConfig,
    variant: SpotSweepVariant,
    dataset: SpotHistoricalDataset,
    *,
    progress_position: int,
    force: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for regime in config.regimes:
        scenario = _scenario_for_regime(config, variant, regime)
        run_dir = scenario.output_dir
        if config.sweep.resume and not force and _resolved_scenario_matches(run_dir, scenario):
            rows.append(
                _regime_summary(
                    variant=variant,
                    regime=regime,
                    scenario=scenario,
                    report=_read_json(run_dir / "summary.json"),
                    duration_seconds=0.0,
                    resumed=True,
                )
            )
            continue
        if run_dir.exists() and (run_dir / "summary.json").exists() and not force:
            raise RuntimeError(
                f"Existing run {run_dir} belongs to a different frozen scenario; use --force."
            )
        started = time.monotonic()
        progress = SpotReplayProgress(
            asset_count=len(scenario.asset_order),
            description=f"[{variant.name}/{regime.name}] Replay",
            position=progress_position,
            leave=False,
        )
        try:
            result = SpotPortfolioBacktester(scenario, dataset).run(progress=progress)
            artifacts = write_spot_backtest_artifacts(result, run_dir)
        finally:
            if progress.bar is not None:
                progress.bar.close()
        rows.append(
            _regime_summary(
                variant=variant,
                regime=regime,
                scenario=scenario,
                report=artifacts["report"],
                duration_seconds=time.monotonic() - started,
                resumed=False,
            )
        )
    combined = _combined_summary(variant, rows)
    _write_json(
        config.output_dir / "candidates" / variant.name / "combined_summary.json",
        combined,
    )
    return combined


def _initialize_worker(progress_lock: Any | None = None) -> None:
    if progress_lock is not None:
        tqdm.set_lock(progress_lock)


def _worker_task(
    variant: SpotSweepVariant,
    progress_position: int,
    force: bool,
) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_DATASET is None:
        raise RuntimeError("Regime sweep worker did not inherit shared context.")
    return _run_candidate(
        _WORKER_CONFIG,
        variant,
        _WORKER_DATASET,
        progress_position=progress_position,
        force=force,
    )


def _failed_row(variant: SpotSweepVariant, exc: BaseException) -> dict[str, Any]:
    return {
        "rank": None,
        "name": variant.name,
        "status": "failed",
        "parameters": _parameters(variant),
        **_parameters(variant),
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_spot_regime_sweep(
    config: SpotRegimeSweepConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name: dict[str, dict[str, Any]] = {}
    state_path = config.output_dir / "manifest.json"
    if state_path.exists():
        try:
            rows_by_name = {
                str(row["name"]): row
                for row in _read_json(state_path).get("rows", [])
                if isinstance(row, dict) and row.get("name")
            }
        except (OSError, ValueError, TypeError):
            rows_by_name = {}

    print("Loading shared Spot market data for frozen regimes...", file=sys.stderr, flush=True)
    market_progress = SpotMarketDataProgressBars()
    try:
        dataset = BinanceSpotHistoricalAdapter(config.sweep.scenario.data_cache_dir).load(
            _market_data_scenario(config),
            progress=market_progress,
            max_workers=config.sweep.market_data_workers,
        )
    finally:
        market_progress.close()
    unavailable_open_times = getattr(dataset, "unavailable_open_times", {})
    if unavailable_open_times:
        counts = ", ".join(
            f"{symbol}={len(open_times)}"
            for symbol, open_times in sorted(unavailable_open_times.items())
        )
        print(
            "Market-wide outage candles are unavailable for trading "
            f"({counts}); flat valuation only.",
            file=sys.stderr,
            flush=True,
        )
    _write_experiment_manifest(config, dataset)

    pending: list[SpotSweepVariant] = []
    for variant in config.sweep.variants:
        scenarios = [
            _scenario_for_regime(config, variant, regime) for regime in config.regimes
        ]
        if (
            config.sweep.resume
            and not force
            and all(_resolved_scenario_matches(item.output_dir, item) for item in scenarios)
        ):
            rows = [
                _regime_summary(
                    variant=variant,
                    regime=regime,
                    scenario=scenario,
                    report=_read_json(scenario.output_dir / "summary.json"),
                    duration_seconds=0.0,
                    resumed=True,
                )
                for regime, scenario in zip(config.regimes, scenarios)
            ]
            rows_by_name[variant.name] = _combined_summary(variant, rows)
        else:
            pending.append(variant)

    outer = tqdm(
        total=len(config.sweep.variants),
        initial=len(config.sweep.variants) - len(pending),
        desc="Spot Regime Sweep",
        unit="candidate",
        dynamic_ncols=True,
        position=0,
    )
    try:
        parallel = config.sweep.replay_parallel and len(pending) > 1
        max_workers = max(1, min(config.sweep.replay_max_workers, len(pending)))
        if parallel and "fork" not in multiprocessing.get_all_start_methods():
            parallel = False
        if parallel:
            print(
                f"Replaying {len(pending)} candidates x 3 regimes with {max_workers} workers...",
                file=sys.stderr,
                flush=True,
            )
            context = multiprocessing.get_context("fork")
            lock = context.RLock()
            tqdm.set_lock(lock)
            global _WORKER_CONFIG, _WORKER_DATASET
            _WORKER_CONFIG = config
            _WORKER_DATASET = dataset
            futures: dict[Future[dict[str, Any]], tuple[SpotSweepVariant, int]] = {}
            iterator = iter(pending)
            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(lock,),
                ) as executor:
                    for slot in range(max_workers):
                        variant = next(iterator, None)
                        if variant is None:
                            break
                        futures[executor.submit(_worker_task, variant, slot + 1, force)] = (
                            variant,
                            slot,
                        )
                    while futures:
                        future = next(as_completed(tuple(futures)))
                        variant, slot = futures.pop(future)
                        try:
                            rows_by_name[variant.name] = future.result()
                            outer.set_postfix_str(
                                f"{variant.name} mean={rows_by_name[variant.name]['mean_edge_vs_hodl_pp']:+.2f}pp",
                                refresh=False,
                            )
                        except Exception as exc:
                            rows_by_name[variant.name] = _failed_row(variant, exc)
                            if not config.sweep.continue_on_error:
                                raise
                        finally:
                            outer.update(1)
                            _write_state(config, rows_by_name)
                        next_variant = next(iterator, None)
                        if next_variant is not None:
                            futures[
                                executor.submit(_worker_task, next_variant, slot + 1, force)
                            ] = (next_variant, slot)
            finally:
                _WORKER_CONFIG = None
                _WORKER_DATASET = None
        else:
            for variant in pending:
                try:
                    rows_by_name[variant.name] = _run_candidate(
                        config,
                        variant,
                        dataset,
                        progress_position=1,
                        force=force,
                    )
                except Exception as exc:
                    rows_by_name[variant.name] = _failed_row(variant, exc)
                    if not config.sweep.continue_on_error:
                        raise
                finally:
                    outer.update(1)
                    _write_state(config, rows_by_name)
    finally:
        outer.close()
    return _write_state(config, rows_by_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable three-regime Spot parameter sweep."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_spot_regime_sweep_config(args.config, output_override=args.output)
    if args.dry_run:
        print(
            f"{config.sweep.name}: {len(config.sweep.variants)} candidates x "
            f"{len(config.regimes)} regimes = {len(config.sweep.variants) * len(config.regimes)} replays"
        )
        for regime in config.regimes:
            print(f"  {regime.name}: {regime.start.isoformat()} -> {regime.end.isoformat()}")
        return 0
    payload = run_spot_regime_sweep(config, force=args.force)
    print(f"Spot regime sweep artifacts: {config.output_dir}")
    winner = next((row for row in payload["rows"] if row.get("status") == "ok"), None)
    if winner:
        print(
            f"Robust winner: {winner['name']} | mean Edge vs HODL "
            f"{winner['mean_edge_vs_hodl_pp']:+.2f} pp | worst "
            f"{winner['worst_regime_edge_vs_hodl_pp']:+.2f} pp | "
            f"{winner['regimes_beating_hodl']}/3 regimes"
        )
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
