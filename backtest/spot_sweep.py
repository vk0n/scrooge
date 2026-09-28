from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from tqdm import tqdm
import yaml

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_runner import SpotMarketDataProgressBars, SpotReplayProgress
from backtest.spot_scenario import (
    SpotBacktestScenario,
    load_spot_backtest_scenario,
    scenario_as_dict,
)
from shared.spot_signal import SpotSignalConfig


@dataclass(frozen=True)
class SpotSweepVariant:
    name: str
    levels_pct: tuple[float, ...]


@dataclass(frozen=True)
class SpotSweepConfig:
    name: str
    config_path: Path
    base_config_path: Path
    output_dir: Path
    scenario: SpotBacktestScenario
    variants: tuple[SpotSweepVariant, ...]
    resume: bool = True
    continue_on_error: bool = True


def _format_level(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _variant_name(levels_pct: tuple[float, ...]) -> str:
    return "-".join(_format_level(value) for value in levels_pct)


def _required_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a YAML object.")
    return value


def load_spot_sweep_config(
    path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> SpotSweepConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file_obj:
        root = yaml.safe_load(file_obj)
    root = _required_mapping(root, field_name="Spot sweep config")
    payload = _required_mapping(root.get("spot_sweep"), field_name="spot_sweep")

    base_config_raw = str(payload.get("base_config") or "").strip()
    if not base_config_raw:
        raise ValueError("spot_sweep.base_config is required.")
    base_config_path = Path(base_config_raw).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = (config_path.parent / base_config_path).resolve()

    output_raw = output_override or payload.get("output_dir")
    if not output_raw:
        raise ValueError("spot_sweep.output_dir is required.")
    output_dir = Path(str(output_raw)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    start = str(payload["start"]) if payload.get("start") is not None else None
    end = str(payload["end"]) if payload.get("end") is not None else None
    scenario = load_spot_backtest_scenario(
        base_config_path,
        start_override=start,
        end_override=end,
    )

    raw_variants = payload.get("levels_pct")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("spot_sweep.levels_pct must contain at least one level combination.")
    variants: list[SpotSweepVariant] = []
    seen: set[tuple[float, ...]] = set()
    for index, raw_levels in enumerate(raw_variants):
        if not isinstance(raw_levels, list):
            raise ValueError(f"spot_sweep.levels_pct[{index}] must be a list.")
        levels = tuple(float(value) for value in raw_levels)
        # Constructing the strategy config applies the same validation as a real replay.
        SpotSignalConfig(
            levels_pct=levels,
            base_tranches_pct=scenario.signal.base_tranches_pct,
            accumulation_tranches_pct=scenario.signal.accumulation_tranches_pct,
        )
        if levels in seen:
            raise ValueError(f"Duplicate Spot sweep combination: {levels}.")
        seen.add(levels)
        variants.append(SpotSweepVariant(name=_variant_name(levels), levels_pct=levels))

    return SpotSweepConfig(
        name=str(payload.get("name") or config_path.stem),
        config_path=config_path,
        base_config_path=base_config_path,
        output_dir=output_dir,
        scenario=scenario,
        variants=tuple(variants),
        resume=bool(payload.get("resume", True)),
        continue_on_error=bool(payload.get("continue_on_error", True)),
    )


def _scenario_for_variant(config: SpotSweepConfig, variant: SpotSweepVariant) -> SpotBacktestScenario:
    signal = replace(config.scenario.signal, levels_pct=variant.levels_pct)
    metadata = {
        **config.scenario.metadata,
        "sweep_name": config.name,
        "sweep_config": str(config.config_path),
        "sweep_levels_pct": list(variant.levels_pct),
    }
    return replace(
        config.scenario,
        name=f"{config.name}-{variant.name}",
        signal=signal,
        output_dir=config.output_dir / "runs" / variant.name,
        metadata=metadata,
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolved_scenario_matches(run_dir: Path, scenario: SpotBacktestScenario) -> bool:
    resolved_path = run_dir / "scenario.resolved.json"
    summary_path = run_dir / "summary.json"
    if not resolved_path.exists() or not summary_path.exists():
        return False
    try:
        payload = _read_json(resolved_path)
    except (OSError, ValueError, TypeError):
        return False
    return payload.get("scenario") == scenario_as_dict(scenario)


def _summary_row(
    *,
    variant: SpotSweepVariant,
    scenario: SpotBacktestScenario,
    report: dict[str, Any],
    duration_seconds: float,
    resumed: bool,
) -> dict[str, Any]:
    portfolio = report["portfolio"]
    success = report["success_metrics"]
    reserve = success["free_reserve"]
    recovery = success["asset_recovery"]
    swings = report["swings"]
    accumulation = report["treasury_accumulation"]
    return {
        "rank": None,
        "name": variant.name,
        "levels_pct": list(variant.levels_pct),
        "status": "ok",
        "resumed": resumed,
        "duration_seconds": duration_seconds,
        "output_dir": str(scenario.output_dir),
        "final_treasury_value": float(portfolio["final_treasury_value"]),
        "hodl_final_treasury_value": float(portfolio["hodl_final_treasury_value"]),
        "difference_vs_hodl": float(portfolio["difference_vs_hodl"]),
        "edge_vs_hodl_pct_points": float(portfolio["edge_vs_hodl_pct_points"]),
        "free_reserve_quote": float(reserve["quote"]),
        "free_reserve_pct": float(reserve["pct_of_initial_invested_capital"]),
        "weighted_effective_assets_pct": float(recovery["weighted_effective_quantity_pct"]),
        "average_effective_assets_pct": float(recovery["average_effective_quantity_pct"]),
        "maximum_drawdown_pct": float(portfolio["maximum_treasury_drawdown_pct"]),
        "opened_bargains": int(swings["total_opened"]),
        "open_bargains": int(swings["still_open"]),
        "oldest_open_days": float(swings["oldest_open_days"] or 0.0),
        "realized_bargain_pnl": float(swings["realized_pnl_quote"]),
        "unrealized_open_pnl": float(swings["unrealized_open_pnl_quote"]),
        "earned_cash_generated": float(accumulation["earned_cash_generated_quote"]),
        "earned_cash_allocated": float(accumulation["earned_cash_allocated_quote"]),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    succeeded = [dict(row) for row in rows if row.get("status") == "ok"]
    succeeded.sort(
        key=lambda row: (
            -float(row["edge_vs_hodl_pct_points"]),
            -float(row["weighted_effective_assets_pct"]),
            -float(row["free_reserve_pct"]),
        )
    )
    for rank, row in enumerate(succeeded, start=1):
        row["rank"] = rank
    failures = [dict(row) for row in rows if row.get("status") != "ok"]
    return succeeded + failures


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _comparison_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scrooge Spot Sweep</title><style>
:root{{--bg:#080c11;--panel:#111821;--line:#293647;--ink:#edf2f7;--muted:#8d9bad;--gold:#e8b84a;--green:#48d6a2;--red:#ff647f}}
*{{box-sizing:border-box}}body{{margin:0;background:repeating-linear-gradient(135deg,#080c11,#080c11 12px,#0a0f15 12px,#0a0f15 24px);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{width:min(1500px,calc(100% - 28px));margin:28px auto 60px}}header,.panel{{border:1px solid var(--line);border-radius:18px;background:rgba(17,24,33,.97)}}header{{padding:28px;background:linear-gradient(125deg,#351020,#111923);margin-bottom:18px}}h1{{margin:8px 0;font-size:34px}}p{{color:var(--muted)}}.eyebrow{{color:var(--gold);letter-spacing:.14em;text-transform:uppercase}}.panel{{padding:18px;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1120px}}th,td{{padding:11px 10px;border-top:1px solid var(--line);text-align:right;white-space:nowrap}}th{{color:var(--muted);font-weight:400;position:sticky;top:0;background:var(--panel)}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:first-child td{{color:var(--green);font-weight:700}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}@media(max-width:700px){{h1{{font-size:26px}}}}
</style></head><body><main><header><span class="eyebrow">Scrooge Research / Spot Parameter Sweep</span><h1 id="title"></h1><p id="meta"></p></header><section class="panel"><table><thead><tr><th>Rank</th><th>Levels</th><th>Edge</th><th>Edge pp</th><th>Final</th><th>Free Reserve</th><th>Weighted Assets</th><th>Average Assets</th><th>Open</th><th>Oldest</th><th>Drawdown</th><th>Runtime</th></tr></thead><tbody id="rows"></tbody></table></section></main>
<script id="data" type="application/json">{data}</script><script>
const d=JSON.parse(document.getElementById('data').textContent),money=v=>`${{v<0?'-':''}}$${{Math.abs(v).toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}`,pct=v=>`${{v>=0?'+':''}}${{v.toFixed(2)}}%`,tone=v=>v>0?'positive':v<0?'negative':'';
document.getElementById('title').textContent=d.name;document.getElementById('meta').textContent=`${{d.start}} to ${{d.end}} | ${{d.completed}}/${{d.total}} completed`;
document.getElementById('rows').innerHTML=d.rows.filter(r=>r.status==='ok').map(r=>`<tr><td>${{r.rank}}</td><td>${{r.levels_pct.join(' / ')}}</td><td class="${{tone(r.difference_vs_hodl)}}">${{money(r.difference_vs_hodl)}}</td><td class="${{tone(r.edge_vs_hodl_pct_points)}}">${{pct(r.edge_vs_hodl_pct_points)}}</td><td>${{money(r.final_treasury_value)}}</td><td>${{money(r.free_reserve_quote)}} · ${{pct(r.free_reserve_pct)}}</td><td>${{pct(r.weighted_effective_assets_pct)}}</td><td>${{pct(r.average_effective_assets_pct)}}</td><td>${{r.open_bargains}}</td><td>${{r.oldest_open_days.toFixed(1)}}d</td><td>${{pct(r.maximum_drawdown_pct)}}</td><td>${{(r.duration_seconds/60).toFixed(1)}}m</td></tr>`).join('');
</script></body></html>"""


def _write_comparison(config: SpotSweepConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = _rank_rows(rows)
    payload = {
        "name": config.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(config.config_path),
        "base_config": str(config.base_config_path),
        "start": config.scenario.start.isoformat(),
        "end": config.scenario.end.isoformat(),
        "total": len(config.variants),
        "completed": sum(row.get("status") == "ok" for row in ranked),
        "failed": sum(row.get("status") == "failed" for row in ranked),
        "rows": ranked,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "comparison.json", payload)
    _write_csv(config.output_dir / "comparison.csv", ranked)
    (config.output_dir / "comparison.html").write_text(_comparison_html(payload), encoding="utf-8")
    return payload


def run_spot_sweep(
    config: SpotSweepConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.json"
    rows_by_name: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            previous = _read_json(manifest_path)
            rows_by_name = {
                str(row["name"]): row
                for row in previous.get("rows", [])
                if isinstance(row, dict) and row.get("name")
            }
        except (OSError, ValueError, TypeError):
            rows_by_name = {}
    variant_names = {variant.name for variant in config.variants}
    rows_by_name = {
        name: row for name, row in rows_by_name.items() if name in variant_names
    }

    print("Loading shared Spot market data...", file=sys.stderr, flush=True)
    market_progress = SpotMarketDataProgressBars()
    try:
        dataset = BinanceSpotHistoricalAdapter(config.scenario.data_cache_dir).load(
            config.scenario,
            progress=market_progress,
        )
    finally:
        market_progress.close()
    outer = tqdm(
        total=len(config.variants),
        desc="Spot Sweep",
        unit="run",
        dynamic_ncols=True,
        position=0,
    )
    try:
        for variant in config.variants:
            scenario = _scenario_for_variant(config, variant)
            run_dir = scenario.output_dir
            can_resume = config.resume and _resolved_scenario_matches(run_dir, scenario)
            if can_resume and not force:
                report = _read_json(run_dir / "summary.json")
                previous_row = rows_by_name.get(variant.name, {})
                rows_by_name[variant.name] = _summary_row(
                    variant=variant,
                    scenario=scenario,
                    report=report,
                    duration_seconds=float(previous_row.get("duration_seconds") or 0.0),
                    resumed=True,
                )
                outer.set_postfix_str(f"{variant.name} resumed", refresh=False)
                outer.update(1)
                continue
            if run_dir.exists() and (run_dir / "summary.json").exists() and not force:
                raise RuntimeError(
                    f"Existing run {run_dir} was produced by a different scenario; use --force to replace it."
                )

            started_at = time.monotonic()
            progress = SpotReplayProgress(
                asset_count=len(scenario.asset_order),
                description=f"[{variant.name}] Replay",
                position=1,
                leave=False,
            )
            try:
                result = SpotPortfolioBacktester(scenario, dataset).run(progress=progress)
                artifacts = write_spot_backtest_artifacts(result, run_dir)
                elapsed = time.monotonic() - started_at
                rows_by_name[variant.name] = _summary_row(
                    variant=variant,
                    scenario=scenario,
                    report=artifacts["report"],
                    duration_seconds=elapsed,
                    resumed=False,
                )
                outer.set_postfix_str(
                    f"{variant.name} edge={rows_by_name[variant.name]['edge_vs_hodl_pct_points']:+.2f}pp",
                    refresh=False,
                )
            except Exception as exc:
                rows_by_name[variant.name] = {
                    "rank": None,
                    "name": variant.name,
                    "levels_pct": list(variant.levels_pct),
                    "status": "failed",
                    "resumed": False,
                    "duration_seconds": time.monotonic() - started_at,
                    "output_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if not config.continue_on_error:
                    raise
            finally:
                if progress.bar is not None:
                    progress.bar.close()
                outer.update(1)
                manifest = {
                    "name": config.name,
                    "config": str(config.config_path),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "rows": list(rows_by_name.values()),
                }
                _write_json(manifest_path, manifest)
                _write_comparison(config, list(rows_by_name.values()))
    finally:
        outer.close()
    return _write_comparison(config, list(rows_by_name.values()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a resumable Spot parameter sweep.")
    parser.add_argument("--config", required=True, help="Spot sweep YAML config.")
    parser.add_argument("--output", help="Override sweep artifact directory.")
    parser.add_argument("--force", action="store_true", help="Rerun and replace completed variants.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list variants without loading data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_spot_sweep_config(args.config, output_override=args.output)
    if args.dry_run:
        print(
            f"{config.name}: {len(config.variants)} variants, "
            f"{config.scenario.start.isoformat()} to {config.scenario.end.isoformat()}"
        )
        for variant in config.variants:
            print(f"  {variant.name}: {', '.join(f'{value:g}' for value in variant.levels_pct)}")
        return 0

    payload = run_spot_sweep(config, force=args.force)
    print(f"Spot sweep artifacts: {config.output_dir}")
    if payload["rows"]:
        winner = next((row for row in payload["rows"] if row.get("status") == "ok"), None)
        if winner is not None:
            print(
                f"Winner: {'/'.join(f'{value:g}' for value in winner['levels_pct'])} | "
                f"Edge vs HODL {winner['edge_vs_hodl_pct_points']:+.2f} pp "
                f"(${winner['difference_vs_hodl']:+,.2f})"
            )
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
