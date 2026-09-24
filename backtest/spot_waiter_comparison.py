from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_scenario import load_spot_backtest_scenario


METRICS = (
    ("final_treasury_value", "Final Treasury Value", "money"),
    ("total_return_pct", "Return", "percent"),
    ("difference_vs_hodl", "Difference vs HODL", "money"),
    ("maximum_treasury_drawdown_pct", "Maximum Drawdown", "percent"),
    ("lifecycle_pnl_quote", "Bargain Lifecycle PnL", "money"),
    ("realized_pnl_quote", "Realized Bargain PnL", "money"),
    ("unrealized_open_pnl_quote", "Unrealized Open PnL", "money"),
    ("final_reserve", "Final Vault Reserve", "money"),
    ("fees", "Execution Fees", "money"),
    ("open_bargains", "Open Bargains", "number"),
    ("underwater_open_bargains", "Underwater Bargains", "number"),
    ("open_90_plus", "90+ Day Bargains", "number"),
    ("realized_cleanup_loss_quote", "Cleanup Losses Realized", "money"),
    ("open_buy_capital_tied_quote", "BUY Capital Tied", "money"),
    ("restore_cost", "SELL Inventory Restore Cost", "money"),
    ("oldest_open_days", "Oldest Open Bargain", "days"),
)


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    portfolio = report["portfolio"]
    swings = report["swings"]
    cleanup = report["waiter_cleanup"]
    overview = report["bargain_analysis"]["overview"]
    lock = cleanup["capital_lock"]
    open_metrics = cleanup["open_bargains"]
    return {
        "final_treasury_value": float(portfolio["final_treasury_value"]),
        "total_return_pct": float(portfolio["total_return_pct"]),
        "difference_vs_hodl": float(portfolio["difference_vs_hodl"]),
        "maximum_treasury_drawdown_pct": float(portfolio["maximum_treasury_drawdown_pct"]),
        "lifecycle_pnl_quote": float(overview["net_pnl_quote"]),
        "realized_pnl_quote": float(swings["realized_pnl_quote"]),
        "unrealized_open_pnl_quote": float(swings["unrealized_open_pnl_quote"]),
        "final_reserve": float(report["shared_usdt"]["ending"]),
        "fees": sum(float(value) for value in swings["fees_by_asset"].values()),
        "open_bargains": float(open_metrics["at_end"]),
        "underwater_open_bargains": float(open_metrics["underwater_at_end"]),
        "open_90_plus": float(open_metrics["age_90_plus"]),
        "realized_cleanup_loss_quote": float(cleanup["realized_cleanup_loss_quote"]),
        "open_buy_capital_tied_quote": float(lock["open_buy_origin_quote"]),
        "restore_cost": float(lock["value_required_to_restore_sell_inventory"]),
        "oldest_open_days": float(open_metrics["oldest_age_days"] or 0.0),
    }


def _comparison_rows(period: str, baseline: dict[str, Any], cleanup: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_metrics = _summary_metrics(baseline)
    cleanup_metrics = _summary_metrics(cleanup)
    return [
        {
            "period": period,
            "metric": key,
            "label": label,
            "format": value_format,
            "baseline": baseline_metrics[key],
            "cleanup": cleanup_metrics[key],
            "delta": cleanup_metrics[key] - baseline_metrics[key],
        }
        for key, label, value_format in METRICS
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _comparison_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scrooge Waiter Cleanup Research</title>
<style>
:root{{--bg:#070b10;--panel:#0e151f;--line:#273449;--ink:#eef2f7;--muted:#8f9caf;--gold:#e9b949;--green:#43d6a0;--red:#ff647f}}
*{{box-sizing:border-box}} body{{margin:0;background:repeating-linear-gradient(135deg,#070b10,#070b10 10px,#090e14 10px,#090e14 20px);color:var(--ink);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{width:min(1480px,calc(100% - 30px));margin:28px auto 60px}} header,.card{{border:1px solid var(--line);border-radius:18px;background:rgba(14,21,31,.96)}} header{{padding:30px;margin-bottom:18px;background:linear-gradient(125deg,#3b0b22,#121621)}} h1{{margin:8px 0;font-size:36px}} h2{{margin:0 0 14px}} .eyebrow{{color:var(--gold);letter-spacing:.16em;text-transform:uppercase}} .period{{margin-top:22px;padding:20px}} table{{width:100%;border-collapse:collapse}} th,td{{padding:12px 10px;border-top:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-weight:400}} .positive{{color:var(--green)}} .negative{{color:var(--red)}} .reasons{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}} .reason{{display:grid;grid-template-columns:1fr auto auto;gap:14px;padding:9px 0;border-top:1px solid var(--line)}} small{{color:var(--muted)}} @media(max-width:780px){{h1{{font-size:27px}}.reasons{{grid-template-columns:1fr}}.period{{overflow:auto}}table{{min-width:760px}}}}
</style></head><body><main><header><span class="eyebrow">Scrooge Research / Controlled A-B Replay</span><h1>Waiter Cleanup: Baseline vs Automatic</h1><p>Same portfolio, market data, entry strategy, fees, slippage, policies, and objectives. Only stale Bargain cleanup differs.</p></header><div id="content"></div></main>
<script id="data" type="application/json">{data}</script><script>
const data=JSON.parse(document.getElementById('data').textContent); const money=v=>`${{v<0?'-':''}}$${{Math.abs(v).toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}`; const fmt=(v,t)=>t==='money'?money(v):t==='percent'?`${{v>=0?'+':''}}${{v.toFixed(2)}}%`:t==='days'?`${{v.toFixed(1)}}d`:t==='quantity'?v.toLocaleString('en-US',{{maximumFractionDigits:4}}):v.toLocaleString('en-US',{{maximumFractionDigits:1}}); const tone=v=>v>0?'positive':v<0?'negative':'';
document.getElementById('content').innerHTML=data.periods.map(period=>`<section class="card period"><h2>${{period.label}}</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Cleanup</th><th>Delta</th></tr></thead><tbody>${{period.rows.map(row=>`<tr><td>${{row.label}}</td><td>${{fmt(row.baseline,row.format)}}</td><td>${{fmt(row.cleanup,row.format)}}</td><td class="${{tone(row.delta)}}">${{fmt(row.delta,row.format)}}</td></tr>`).join('')}}</tbody></table><div class="reasons"><div><h3>Cleanup Reasons</h3>${{Object.entries(period.cleanup_reasons).map(([reason,item])=>`<div class="reason"><span>${{reason.replaceAll('_',' ')}}</span><strong>${{item.closes}}</strong><strong class="${{tone(item.realized_pnl_quote)}}">${{money(item.realized_pnl_quote)}}</strong></div>`).join('')}}</div><div><h3>Capacity Effect</h3><p><strong>${{period.capacity.open_bargains_prevented_by_cap}}</strong> openings prevented</p><p><strong>${{period.capacity.capacity_forced_hold_cycles}}</strong> forced HOLD cycles</p><p><strong>${{period.capacity.capacity_cleanup_actions}}</strong> capacity cleanup actions</p><small>Cleanup parameters are intentionally untuned.</small></div></div></section>`).join('');
</script></body></html>"""


def run_comparison(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    periods: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for preset, label in (("6m", "Six Months"), ("1y", "One Year")):
        scenario = load_spot_backtest_scenario(config_path, preset=preset)
        dataset = BinanceSpotHistoricalAdapter(scenario.data_cache_dir).load(scenario)
        reports: dict[str, dict[str, Any]] = {}
        for mode, enabled in (("baseline", False), ("cleanup", True)):
            variant = replace(
                scenario,
                waiter_cleanup=replace(scenario.waiter_cleanup, enabled=enabled),
                output_dir=target / preset / mode,
            )
            result = SpotPortfolioBacktester(variant, dataset).run()
            reports[mode] = write_spot_backtest_artifacts(result, variant.output_dir)["report"]
        rows = _comparison_rows(preset, reports["baseline"], reports["cleanup"])
        csv_rows.extend(rows)
        periods.append(
            {
                "key": preset,
                "label": label,
                "rows": rows,
                "cleanup_reasons": reports["cleanup"]["waiter_cleanup"]["by_reason"],
                "capacity": reports["cleanup"]["waiter_cleanup"]["capacity"],
                "baseline_dir": str((target / preset / "baseline").resolve()),
                "cleanup_dir": str((target / preset / "cleanup").resolve()),
            }
        )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(Path(config_path).expanduser().resolve()),
        "periods": periods,
    }
    _write_json(target / "comparison.json", payload)
    _write_csv(target / "comparison.csv", csv_rows)
    (target / "comparison.html").write_text(_comparison_html(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline Spot replay with waiter cleanup.")
    parser.add_argument("--config", required=True, help="Spot backtest scenario YAML.")
    parser.add_argument("--output", required=True, help="Comparison artifact directory.")
    args = parser.parse_args()
    payload = run_comparison(args.config, args.output)
    print(f"Waiter cleanup comparison: {Path(args.output).expanduser().resolve()}")
    for period in payload["periods"]:
        final_row = next(row for row in period["rows"] if row["metric"] == "final_treasury_value")
        print(
            f"{period['label']}: baseline ${final_row['baseline']:,.2f}, "
            f"cleanup ${final_row['cleanup']:,.2f}, delta ${final_row['delta']:,.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
