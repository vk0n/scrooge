"""Aggregate staged cleanup-policy research into reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
from typing import Any


MAIN_STUDIES = (
    "phase-a",
    "phase-b-deep",
    "phase-b-deep-l2",
    "phase-b-aging",
    "phase-b-capacity",
    "finalist-sensitivity",
)
ROBUSTNESS_STUDIES = ("robustness-year-1", "robustness-year-2")
ACCEPTANCE = {
    "max_open_bargains_90_plus": 5,
    "max_oldest_open_days": 120.0,
    "max_drawdown_deterioration_pp_vs_baseline": 2.0,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _profit_target_metrics(summary: dict[str, Any]) -> tuple[int, float]:
    cash_flow = summary["bargain_cash_flow"]["by_close_type"]["profit_target"]
    cleanup_realized = sum(
        float(item.get("realized_pnl_quote") or 0.0)
        for item in summary["waiter_cleanup"]["by_reason"].values()
    )
    return (
        int(cash_flow["closed_sell_bargains"]),
        float(summary["swings"]["realized_pnl_quote"]) - cleanup_realized,
    )


def _load_study(root: Path, study: str) -> list[dict[str, Any]]:
    comparison = _read_json(root / study / "comparison.json")
    output = []
    for original in comparison["rows"]:
        if original.get("status") != "ok":
            output.append({**original, "study": study})
            continue
        row = {**original, "study": study}
        run_dir = Path(row["output_dir"])
        summary = _read_json(run_dir / "summary.json")
        profit_closes, profit_pnl = _profit_target_metrics(summary)
        row.update(
            {
                "profit_target_closes": profit_closes,
                "profit_target_realized_pnl": profit_pnl,
                "committed_cash_quote": float(
                    summary["success_metrics"]["free_reserve"]["committed_quote"]
                ),
                "relative_wealth_vs_hodl": float(
                    summary["portfolio"]["relative_wealth"]["terminal"]
                ),
                "accumulation_purchases": int(
                    summary["treasury_accumulation"]["overview"]["count"]
                ),
                "accumulation_usdt_deployed": float(
                    summary["treasury_accumulation"]["overview"]["usdt_deployed"]
                ),
                "accumulated_asset_quantities": {
                    symbol: float(metrics.get("net_asset_acquired") or 0.0)
                    for symbol, metrics in summary["treasury_accumulation"][
                        "per_asset"
                    ].items()
                },
                "closed_bargains": int(summary["swings"]["total_closed"]),
            }
        )
        output.append(row)
    return output


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cleanup_cash_burn_ratio_pct") is not None
        and not row["name"].startswith("G-")
    ]
    frontier = []
    for candidate in candidates:
        edge = float(candidate["edge_vs_hodl_pct_points"])
        burn = float(candidate["cleanup_cash_burn_ratio_pct"])
        if not any(
            float(other["edge_vs_hodl_pct_points"]) >= edge
            and float(other["cleanup_cash_burn_ratio_pct"]) <= burn
            and (
                float(other["edge_vs_hodl_pct_points"]) > edge
                or float(other["cleanup_cash_burn_ratio_pct"]) < burn
            )
            for other in candidates
            if other is not candidate
        ):
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            -float(row["edge_vs_hodl_pct_points"]),
            float(row["cleanup_cash_burn_ratio_pct"]),
        ),
    )


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _policy_label(row: dict[str, Any]) -> str:
    capacity = (
        f"capacity {row['capacity_cleanup_min_age_days']:g}d"
        if row["capacity_cleanup_enabled"]
        else "capacity OFF"
    )
    deep_loss = (
        f"deep loss {row['deep_loss_min_age_days']:g}d/"
        f"{row['unrealized_pnl_pct']:g}%/L{row['deep_loss_required_reverse_level']}"
        if row["deep_loss_enabled"]
        else "deep loss OFF"
    )
    aging = "aging ON" if row["aging_cleanup_enabled"] else "aging OFF"
    cap = row["max_open_bargains_per_asset"]
    return f"{capacity} + {deep_loss} + {aging} + cap {cap if cap is not None else 'OFF'}"


def _table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Candidate | Edge pp | Burn | Final | Cleanup top-up | Open | 90d+ | Oldest | Max DD | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('name', 'unknown')} | N/A | N/A | N/A | N/A | "
                f"N/A | N/A | N/A | N/A | no ({row.get('error', 'failed')}) |"
            )
            continue
        lines.append(
            f"| {row['name']} | {row['edge_vs_hodl_pct_points']:.2f} | "
            f"{_pct(row['cleanup_cash_burn_ratio_pct'])} | {_money(row['final_treasury_value'])} | "
            f"{_money(row['gross_cleanup_top_up'])} | {row['open_bargains']} | "
            f"{row['open_bargains_90_plus']} | {row['oldest_open_days']:.1f}d | "
            f"{row['maximum_drawdown_pct']:.2f}% | {'yes' if row['operationally_accepted'] else 'no'} |"
        )
    return lines


def _annotate_operational_rows(
    rows: list[dict[str, Any]],
    *,
    baseline: dict[str, Any],
) -> None:
    baseline_drawdown = float(baseline["maximum_drawdown_pct"])
    baseline_profit_closes = int(baseline["profit_target_closes"])
    for row in rows:
        if row.get("status") != "ok":
            row["operationally_accepted"] = False
            row["additional_profit_target_closes_vs_baseline"] = None
            continue
        row["operationally_accepted"] = (
            int(row["open_bargains_90_plus"])
            <= ACCEPTANCE["max_open_bargains_90_plus"]
            and float(row["oldest_open_days"])
            <= ACCEPTANCE["max_oldest_open_days"]
            and float(row["maximum_drawdown_pct"])
            >= baseline_drawdown
            - ACCEPTANCE["max_drawdown_deterioration_pp_vs_baseline"]
        )
        row["additional_profit_target_closes_vs_baseline"] = (
            int(row["profit_target_closes"]) - baseline_profit_closes
        )


def _html_report(
    *,
    rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    recommended: dict[str, Any],
    recommendation_accepted: bool,
    interval_start: str,
    interval_end: str,
) -> str:
    frontier_keys = {(row["study"], row["name"]) for row in frontier}
    body_rows = []
    for row_index, row in enumerate(rows):
        if row.get("status") != "ok":
            body_rows.append(
                f'<tr class="failed" data-order="{row_index}"><td data-sort="'
                + html.escape(str(row.get("study") or ""))
                + '">'
                + html.escape(str(row.get("study") or ""))
                + '</td><td data-sort="'
                + html.escape(str(row.get("name") or "unknown"))
                + '">'
                + html.escape(str(row.get("name") or "unknown"))
                + "</td><td data-sort=\"\">N/A</td><td data-sort=\"\">N/A</td>"
                + "<td data-sort=\"\">N/A</td><td data-sort=\"\">N/A</td>"
                + "<td data-sort=\"\">N/A</td><td data-sort=\"\">N/A</td>"
                + '<td data-sort="0">Failed: '
                + html.escape(str(row.get("error") or "unknown error"))
                + "</td></tr>"
            )
            continue
        classes = []
        if row.get("operationally_accepted"):
            classes.append("accepted")
        if (row["study"], row["name"]) in frontier_keys:
            classes.append("frontier")
        body_rows.append(
            f'<tr class="{" ".join(classes)}" data-order="{row_index}">'
            f'<td data-sort="{html.escape(row["study"])}">{html.escape(row["study"])}</td>'
            f'<td data-sort="{html.escape(row["name"])}">{html.escape(row["name"])}</td>'
            f'<td data-sort="{row["edge_vs_hodl_pct_points"]}">{row["edge_vs_hodl_pct_points"]:.2f}</td>'
            f'<td data-sort="{row["cleanup_cash_burn_ratio_pct"]}">{_pct(row["cleanup_cash_burn_ratio_pct"])}</td>'
            f'<td data-sort="{row["final_treasury_value"]}">{_money(row["final_treasury_value"])}</td>'
            f'<td data-sort="{row["open_bargains_90_plus"]}">{row["open_bargains_90_plus"]}</td>'
            f'<td data-sort="{row["oldest_open_days"]}">{row["oldest_open_days"]:.1f}d</td>'
            f'<td data-sort="{row["restore_cost_quote"]}">{_money(row["restore_cost_quote"])}</td>'
            f'<td data-sort="{int(row["operationally_accepted"])}">'
            f'{"yes" if row["operationally_accepted"] else "no"}</td></tr>'
        )
    return f"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scrooge Cleanup Research</title><style>
:root{{--bg:#080d12;--panel:#111923;--line:#2b3949;--ink:#e8edf3;--muted:#91a0b1;--gold:#e9b949;--green:#4ad7a5;--red:#ff667f}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% 0,#24131a 0,transparent 36%),var(--bg);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{width:min(1500px,calc(100% - 28px));margin:28px auto 60px}}header,.panel{{border:1px solid var(--line);border-radius:18px;background:rgba(17,25,35,.96)}}header{{padding:28px;margin-bottom:18px}}h1{{margin:8px 0 12px;font-size:32px}}p{{color:var(--muted)}}strong{{color:var(--gold)}}.hint{{margin:0 0 14px}}.panel{{overflow:auto;padding:18px}}table{{border-collapse:collapse;width:100%;min-width:1050px}}th,td{{border-top:1px solid var(--line);padding:10px;text-align:right;white-space:nowrap}}th{{color:var(--muted);padding:0;position:sticky;top:0;background:var(--panel);z-index:1}}th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}th button{{width:100%;padding:11px 10px;border:0;background:transparent;color:inherit;font:inherit;text-align:inherit;cursor:pointer}}th button:hover,th button:focus-visible{{color:var(--ink);background:#182331;outline:none}}.sort-mark{{display:inline-block;width:1.2em;color:var(--gold)}}tr.frontier td:nth-child(2){{color:var(--gold)}}tr.accepted td:last-child{{color:var(--green)}}tr.failed{{color:var(--red)}}
</style></head><body><main><header><span>SCROOGE / CLEANUP RESEARCH</span><h1>Neutral-bull, {html.escape(interval_start)}..{html.escape(interval_end)}</h1><p>{'Recommended' if recommendation_accepted else 'Research leader (no accepted candidates)'}: <strong>{html.escape(recommended['name'])}</strong> · Edge {recommended['edge_vs_hodl_pct_points']:.2f} pp · Burn {_pct(recommended['cleanup_cash_burn_ratio_pct'])}</p></header><section class="panel"><p class="hint">Натисніть на назву стовпця для сортування. Повторний клік змінює напрямок.</p><table id="results"><thead><tr><th aria-sort="none"><button type="button" data-type="text">Study <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="text">Candidate <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Edge pp <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Burn <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Final <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">90d+ <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Oldest <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Restore <span class="sort-mark"></span></button></th><th aria-sort="none"><button type="button" data-type="number">Accepted <span class="sort-mark"></span></button></th></tr></thead><tbody>{''.join(body_rows)}</tbody></table></section></main><script>
const table=document.getElementById('results'),body=table.tBodies[0],buttons=[...table.querySelectorAll('thead button')];let active=-1,direction=1;
buttons.forEach((button,index)=>button.addEventListener('click',()=>{{const type=button.dataset.type;if(active===index)direction*=-1;else{{active=index;direction=type==='number'?-1:1}}buttons.forEach((item,itemIndex)=>{{const selected=itemIndex===active;item.parentElement.setAttribute('aria-sort',selected?(direction===1?'ascending':'descending'):'none');item.querySelector('.sort-mark').textContent=selected?(direction===1?'▲':'▼'):''}});const rows=[...body.rows];rows.sort((left,right)=>{{const a=left.cells[index].dataset.sort,b=right.cells[index].dataset.sort;if(a===''&&b==='')return Number(left.dataset.order)-Number(right.dataset.order);if(a==='')return 1;if(b==='')return -1;const comparison=type==='number'?Number(a)-Number(b):a.localeCompare(b,undefined,{{numeric:true,sensitivity:'base'}});return comparison===0?Number(left.dataset.order)-Number(right.dataset.order):comparison*direction}});rows.forEach(row=>body.appendChild(row))}}));
</script></body></html>"""


def build(root: Path) -> dict[str, Any]:
    research_dir = root / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for study in MAIN_STUDIES for row in _load_study(root, study)]
    robustness = [
        row
        for study in ROBUSTNESS_STUDIES
        if (root / study / "comparison.json").exists()
        for row in _load_study(root, study)
    ]
    baseline = next(row for row in rows if row["name"] == "A-all-cleanup")
    if baseline.get("status") != "ok":
        raise ValueError("A-all-cleanup baseline did not complete successfully.")
    _annotate_operational_rows(rows, baseline=baseline)

    frontier = _pareto(rows)
    accepted = [row for row in rows if row["operationally_accepted"]]
    ranked_candidates = [
        row
        for row in rows
        if row.get("status") == "ok" and not row["name"].startswith("G-")
    ]
    selection_pool = accepted or ranked_candidates
    best_edge = max(selection_pool, key=lambda row: row["edge_vs_hodl_pct_points"])
    lowest_burn = min(
        (
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("cleanup_cash_burn_ratio_pct") is not None
            and not row["name"].startswith("G-")
        ),
        key=lambda row: row["cleanup_cash_burn_ratio_pct"],
    )
    inventory_health = min(
        selection_pool,
        key=lambda row: (
            row["open_bargains_90_plus"],
            row["oldest_open_days"],
            -row["edge_vs_hodl_pct_points"],
        ),
    )
    balanced = best_edge

    baseline_summary = _read_json(Path(baseline["output_dir"]) / "summary.json")
    balanced_summary = _read_json(Path(balanced["output_dir"]) / "summary.json")
    interval_start = str(baseline_summary["scenario"]["start"]).split("T", 1)[0]
    interval_end = str(baseline_summary["scenario"]["end"]).split("T", 1)[0]

    per_asset = []
    for candidate in (baseline, balanced):
        summary = _read_json(Path(candidate["output_dir"]) / "summary.json")
        for symbol, metrics in summary["waiter_cleanup"]["per_asset"].items():
            accounting = metrics["accounting"]
            per_asset.append(
                {
                    "candidate": candidate["name"],
                    "asset": symbol,
                    "cleanup_closes": metrics["cleanup_closes"],
                    "cleanup_lifecycle_pnl_quote": accounting["lifecycle_pnl_quote"],
                    "spendable_top_up_quote": accounting["reserve_deployed_quote"],
                    "cash_released_quote": accounting["cash_released_quote"],
                    "open_bargains_at_end": metrics["open_bargains_at_end"],
                    "maximum_open_bargain_count": metrics["maximum_open_bargain_count"],
                    "average_open_bargain_count": metrics["average_open_bargain_count"],
                }
            )

    reason_lines = [
        "| Candidate | Mechanism | Closes | Lifecycle PnL | Spendable top-up | Cash released |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate, summary in (
        (baseline, baseline_summary),
        (balanced, balanced_summary),
    ):
        for reason, metrics in summary["waiter_cleanup"]["by_reason"].items():
            if int(metrics["closes"]) == 0:
                continue
            reason_lines.append(
                f"| {candidate['name']} | {reason} | {metrics['closes']} | "
                f"{_money(metrics['lifecycle_pnl_quote'])} | "
                f"{_money(metrics['reserve_deployed_quote'])} | "
                f"{_money(metrics['cash_released_quote'])} |"
            )

    asset_lines = [
        "| Asset | Cleanup closes | Lifecycle PnL | Spendable top-up | Cash released | Open end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in (item for item in per_asset if item["candidate"] == balanced["name"]):
        asset_lines.append(
            f"| {row['asset']} | {row['cleanup_closes']} | "
            f"{_money(row['cleanup_lifecycle_pnl_quote'])} | "
            f"{_money(row['spendable_top_up_quote'])} | "
            f"{_money(row['cash_released_quote'])} | {row['open_bargains_at_end']} |"
        )

    robustness_lines = [
        "| Window | Candidate | Edge pp | Burn | 90d+ | Oldest |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in robustness:
        robustness_lines.append(
            f"| {row['study']} | {row['name']} | "
            f"{row['edge_vs_hodl_pct_points']:.2f} | "
            f"{_pct(row['cleanup_cash_burn_ratio_pct'])} | "
            f"{row['open_bargains_90_plus']} | {row['oldest_open_days']:.1f}d |"
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    phase_a = sorted(
        (row for row in rows if row["study"] == "phase-a"),
        key=lambda row: row["name"],
    )
    top = sorted(accepted, key=lambda row: -row["edge_vs_hodl_pct_points"])[:10]
    rows_by_name = {
        row["name"]: row for row in rows if row.get("status") == "ok"
    }
    accumulation_quantities = ", ".join(
        f"{symbol} {quantity:g}"
        for symbol, quantity in sorted(
            balanced["accumulated_asset_quantities"].items()
        )
    )
    sensitivity_names = ("cap25-deep45", "capacity-30d", "cap35-deep45")
    sensitivity = [rows_by_name[name] for name in sensitivity_names]
    sensitivity_text = ", ".join(
        f"{row['capacity_cleanup_min_age_days']:g}d = "
        f"{row['edge_vs_hodl_pct_points']:.2f} pp"
        for row in sensitivity
    )
    fallback = rows_by_name["aging-disabled"]
    robustness_by_key = {
        (row["study"], row["name"]): row
        for row in robustness
        if row.get("status") == "ok"
    }
    robustness_summary = []
    for label, study in (
        ("Year 1", "robustness-year-1"),
        ("Year 2", "robustness-year-2"),
    ):
        capacity = robustness_by_key.get((study, "best-edge-capacity-30d"))
        capacity_off = robustness_by_key.get((study, "balanced-capacity-off"))
        if capacity is None or capacity_off is None:
            continue
        robustness_summary.append(
            f"{label}: cap30 {capacity['edge_vs_hodl_pct_points']:.2f} pp проти "
            f"OFF {capacity_off['edge_vs_hodl_pct_points']:.2f} pp"
        )
    acceptance_text = (
        f"Operational acceptance: {len(accepted)}/{len(ranked_candidates)} candidates; "
        "<=5 open Bargains 90d+, oldest <=120d, drawdown не гірший за baseline "
        "більш ніж на 2 pp. Restore cost наведений лише як інформаційна метрика."
    )
    if not accepted:
        acceptance_text += (
            " Жоден кандидат не пройшов ці критерії. Research leader нижче не є "
            "production-рекомендацією."
        )
    lines = [
        "# Cleanup Optimization Research",
        "",
        "## A. Executive summary",
        "",
        f"Досліджено {len(rows)} дворічні policy variants на neutral-bull вікні "
        f"{interval_start}..{interval_end}. "
        f"{'Найкращий прийнятний кандидат' if accepted else 'Research leader без operational acceptance'} — "
        f"`{best_edge['name']}`: Edge `{best_edge['edge_vs_hodl_pct_points']:.2f} pp`, "
        f"Cleanup Cash Burn `{_pct(best_edge['cleanup_cash_burn_ratio_pct'])}`, "
        f"{best_edge['open_bargains_90_plus']} Bargain старше 90 днів.",
        "",
        f"Commit базового коду: `{commit}`. Робоче дерево містить описані в цьому "
        "звіті research-зміни. Execution: next 1m open, fee 0.1%, slippage 5 bps, "
        "TP 10%, без forced terminal close.",
        "",
        "## B. Baseline reproduction and ablation A-G",
        "",
        f"A дав {_money(baseline['final_treasury_value'])} і "
        f"{baseline['edge_vs_hodl_pct_points']:.3f} pp Edge; HODL return за це вікно "
        f"{baseline_summary['portfolio']['hodl_return_pct']:+.2f}%.",
        "",
        *_table(phase_a),
        "",
        "G наведений лише як історичний uncapped control і не входить до ranking. "
        "Operational acceptance застосована однаково до всіх capped candidates.",
        "",
        "## C. Parameter optimization",
        "",
        *_table(top),
        "",
        "Pareto frontier побудований без G за двома осями: Edge вище, burn нижче. "
        f"На frontier {len(frontier)} записів; повні дані лежать у `pareto_frontier.json/csv`.",
        "",
        f"Nearby capacity sensitivity: {sensitivity_text}. Ізольований максимум слід "
        "трактувати як підвищений ризик in-sample tuning.",
        "",
        "## D. Cash-flow analysis",
        "",
        f"Для рекомендованого кандидата gross cleanup top-up = "
        f"{_money(balanced['gross_cleanup_top_up'])}, cleanup cash released = "
        f"{_money(balanced['cleanup_cash_released'])}, net consumed = "
        f"{_money(balanced['net_cleanup_cash_consumed'])}, gross profit cash generated = "
        f"{_money(balanced['gross_profit_cash_generated'])}, burn = "
        f"{_pct(balanced['cleanup_cash_burn_ratio_pct'])}. Committed proceeds не "
        "рахуються generated cash, а accumulation purchases не входять у cleanup burn.",
        "",
        *reason_lines,
        "",
        f"Accumulation winner: {balanced['accumulation_purchases']} purchases, "
        f"{_money(balanced['accumulation_usdt_deployed'])} spendable USDT deployed. "
        f"Accumulated: {accumulation_quantities}. "
        f"Final reserve: {_money(balanced['spendable_reserve_quote'])} spendable + "
        f"{_money(balanced['committed_cash_quote'])} committed.",
        "",
        "## E. Capacity and inventory",
        "",
        f"`{balanced['name']}` закрив {balanced['capacity_cleanup_closes']} capacity "
        f"Bargains, завершив із {balanced['open_bargains']} open, "
        f"{balanced['open_bargains_90_plus']} старше 90d, oldest "
        f"{balanced['oldest_open_days']:.1f}d і restore cost "
        f"{_money(balanced['restore_cost_quote'])}. Було "
        f"{balanced['capacity_blocked_evaluations']:,} blocked evaluations та "
        f"{balanced['unique_capacity_blocked_opportunities']:,} unique blocked opportunities.",
        "",
        f"Profit-target closes: {balanced['profit_target_closes']} "
        f"({balanced['additional_profit_target_closes_vs_baseline']:+d} vs baseline A), "
        f"realized profit-target PnL {_money(balanced['profit_target_realized_pnl'])}. "
        f"Weighted effective inventory recovery "
        f"{balanced['weighted_effective_assets_pct']:.2f}%.",
        "",
        *asset_lines,
        "",
        acceptance_text + " Усі failed кандидати збережені, жоден не приховується.",
        "",
        "Blocked accumulation через нульовий spendable reserve наразі не "
        "інструментується як окрема подія: planner повертає `no_free_vault_reserve` "
        "без action. Тому цю метрику не можна чесно відновити з наявних artifacts.",
        "Так само точний wall-clock time at cap не зберігався; звіт використовує "
        "blocked evaluations і average/maximum inventory marks, не підміняючи ними час.",
        "",
        "## F. Recommendation",
        "",
        (
            f"Best Edge: `{best_edge['name']}`. Lowest raw burn: `{lowest_burn['name']}` "
            f"({'прийнятний' if lowest_burn['operationally_accepted'] else 'неприйнятний'} inventory). "
            + (
                "Best balanced і рекомендований production candidate: "
                if accepted
                else "Найкращий measured research candidate, без production acceptance: "
            )
            + f"`{_policy_label(balanced)}`. "
            f"Best inventory-health: `{inventory_health['name']}`."
        ),
        "",
        f"Перед live-зміною рекомендовано shadow-run. Capacity-OFF fallback "
        f"`{fallback['name']}` має {fallback['edge_vs_hodl_pct_points']:.2f} pp Edge і "
        f"{_pct(fallback['cleanup_cash_burn_ratio_pct'])} burn.",
        "",
        "## G. Robustness and limitations",
        "",
        (
            "; ".join(robustness_summary) + "."
            if robustness_summary
            else "Annual robustness ще не перезапущено для цього вікна."
        ),
        "",
        *robustness_lines,
        "",
        "Це in-sample дослідження одного neutral-bull вікна. Воно не є незалежним "
        "out-of-sample доказом; TP 5% sensitivity свідомо не змішувалась з primary TP 10% ranking.",
    ]

    payload = {
        "commit": commit,
        "interval": {"start": interval_start, "end": interval_end},
        "acceptance_thresholds": ACCEPTANCE,
        "baseline": baseline,
        "best_edge": best_edge,
        "lowest_burn": lowest_burn,
        "best_balanced": balanced,
        "best_inventory_health": inventory_health,
        "recommended_production_candidate": balanced if accepted else None,
        "research_leader": balanced,
        "rows": rows,
    }
    _write_json(research_dir / "full_results.json", payload)
    _write_csv(research_dir / "full_results.csv", rows)
    _write_json(research_dir / "pareto_frontier.json", frontier)
    _write_csv(research_dir / "pareto_frontier.csv", frontier)
    _write_json(research_dir / "per_asset_cleanup.json", per_asset)
    _write_csv(research_dir / "per_asset_cleanup.csv", per_asset)
    _write_json(research_dir / "robustness.json", robustness)
    _write_csv(research_dir / "robustness.csv", robustness)
    (research_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (research_dir / "report.html").write_text(
        _html_report(
            rows=rows,
            frontier=frontier,
            recommended=balanced,
            recommendation_accepted=bool(accepted),
            interval_start=interval_start,
            interval_end=interval_end,
        ),
        encoding="utf-8",
    )
    (research_dir / "commands.md").write_text(
        "# Reproduction commands\n\n"
        + "\n".join(
            f"`./scrooge-env/bin/python -m backtest.spot_sweep --config "
            f"runtime/spot_backtests/sweeps/cleanup-research-neutral-bull-2y-{name}.yaml`"
            for name in (
                "phase-a",
                "phase-b-deep",
                "phase-b-deep-l2",
                "phase-b-aging",
                "phase-b-capacity",
                "finalist-sensitivity",
            )
        )
        + "\n`./scrooge-env/bin/python -m backtest.spot_sweep --config "
        "runtime/spot_backtests/sweeps/cleanup-research-neutral-bull-year-1.yaml`"
        "\n`./scrooge-env/bin/python -m backtest.spot_sweep --config "
        "runtime/spot_backtests/sweeps/cleanup-research-neutral-bull-year-2.yaml`"
        + "\n\n`./scrooge-env/bin/python -m backtest.cleanup_research_report "
        f"runtime/spot_backtests/runs/{root.name}`\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    payload = build(args.root.expanduser().resolve())
    print(payload["research_leader"]["name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
