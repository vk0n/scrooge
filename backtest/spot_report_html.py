from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
from typing import Any

from backtest.spot_bargain_analysis import build_bargain_analysis


MAX_CHART_POINTS = 720


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sample_rows(rows: list[dict[str, Any]], limit: int = MAX_CHART_POINTS) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    indexes = {
        round(index * (len(rows) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [rows[index] for index in sorted(indexes)]


def _maximum_drawdown(values: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, ((value / peak) - 1.0) * 100.0)
    return worst


def display_spot_report_title(name: str, start: str | None = None, end: str | None = None) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "scrooge-treasury-portfolio-replay" or normalized.startswith(
        "treasury-10-assets-real-quantities"
    ):
        try:
            duration_days = (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).days
        except (TypeError, ValueError):
            duration_days = 0
        if duration_days >= 330:
            period = "One-Year"
        elif duration_days >= 150:
            period = "Six-Month"
        elif duration_days > 0:
            period = f"{duration_days}-Day"
        else:
            return "Scrooge Treasury: Portfolio Replay"
        return f"Scrooge Treasury: {period} Portfolio Replay"
    acronyms = {"6m": "6M", "1y": "1Y", "ton": "TON", "gram": "GRAM", "usdt": "USDT"}
    words = name.replace("_", " ").replace("-", " ").split()
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in words)


def _report_payload(
    report: dict[str, Any],
    equity: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    swings: list[dict[str, Any]],
) -> dict[str, Any]:
    sampled_equity = _sample_rows(equity)
    assets = []
    for symbol, item in sorted(report["per_asset"].items()):
        trading = item["trading"]
        benchmark = item["benchmark"]
        final = item["final"]
        starting = item["starting"]
        assets.append(
            {
                "symbol": symbol,
                "objective": starting.get("trading_objective"),
                "marketReturn": _number(item["market"].get("market_return_pct")),
                "deltaVsHodl": (
                    _number(benchmark.get("scrooge_asset_value"))
                    - _number(benchmark.get("hodl_value"))
                ),
                "hodlValue": _number(benchmark.get("hodl_value")),
                "scroogeValue": _number(benchmark.get("scrooge_asset_value")),
                "startingQuantity": _number(starting.get("quantity")),
                "finalQuantity": _number(final.get("quantity")),
                "targetQuantity": _number(final.get("target_holding")),
                "opened": int(trading.get("total_opened") or 0),
                "closed": int(trading.get("total_closed") or 0),
                "open": int(trading.get("still_open") or 0),
                "realizedPnl": _number(trading.get("realized_pnl_quote")),
                "unrealizedPnl": _number(trading.get("unrealized_open_pnl_quote")),
                "nearFloorPct": _number(item["inventory"].get("time_near_floor_pct")),
                "unrestoredQuantity": _number(
                    item["bad_cases"].get("quantity_sold_not_restored")
                ),
            }
        )

    fees = sum(_number(value) for value in report["swings"].get("fees_by_asset", {}).values())
    hodl_values = [_number(item.get("hodl_value")) for item in equity]
    portfolio = report["portfolio"]
    swing_history: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in report["scenario"]["asset_order"]
    }
    for swing in swings:
        economics = swing.get("economics") or {}
        strategy_reason = swing.get("strategy_reason") or {}
        compact_swing = {
            "id": swing.get("swing_id"),
            "status": swing.get("status"),
            "originSide": swing.get("origin_side"),
            "objective": swing.get("trading_objective"),
            "openedAtMs": int(swing.get("opened_at_ms") or 0),
            "closedAtMs": (
                int(swing["closed_at_ms"])
                if swing.get("closed_at_ms") is not None
                else None
            ),
            "ageSeconds": _number(swing.get("age_seconds")),
            "openingQuantity": _number(economics.get("opening_quantity")),
            "remainingQuantity": _number(economics.get("remaining_quantity")),
            "openingPrice": economics.get("weighted_opening_price"),
            "closingPrice": economics.get("weighted_closing_price"),
            "marketPrice": swing.get("current_market_price"),
            "realizedPnl": _number(economics.get("realized_pnl_quote")),
            "unrealizedPnl": _number(economics.get("unrealized_pnl_quote")),
            "realizedCashGain": economics.get("realized_cash_gain_quote"),
            "realizedAssetGain": economics.get("realized_asset_gain"),
            "fees": economics.get("fees_by_asset") or {},
            "signalLevel": strategy_reason.get("signal_level"),
            "rollingChangePct": strategy_reason.get("rolling_change_pct"),
            "sizingModifier": strategy_reason.get("sizing_modifier"),
            "signalTier": (strategy_reason.get("indicator_assessment") or {}).get("tier"),
            "returnPct": (
                (
                    _number(economics.get("realized_pnl_quote"))
                    + (
                        0.0
                        if swing.get("status") == "closed"
                        else _number(economics.get("unrealized_pnl_quote"))
                    )
                )
                / _number(economics.get("opening_quote_quantity"))
                * 100.0
                if _number(economics.get("opening_quote_quantity")) > 0
                else None
            ),
            "executions": [
                {
                    "id": execution.get("execution_id"),
                    "side": execution.get("side"),
                    "quantity": _number(execution.get("quantity")),
                    "price": _number(execution.get("price")),
                    "quoteQuantity": _number(execution.get("quote_quantity")),
                    "feeAmount": _number(execution.get("fee_amount")),
                    "feeAsset": execution.get("fee_asset"),
                    "executedAt": execution.get("executed_at"),
                }
                for execution in swing.get("executions") or []
            ],
        }
        symbol = str(swing.get("asset_symbol") or "")
        swing_history.setdefault(symbol, []).append(compact_swing)
    for items in swing_history.values():
        items.sort(key=lambda item: item["openedAtMs"], reverse=True)

    return {
        "scenario": report["scenario"],
        "portfolio": portfolio,
        "reserve": report["shared_usdt"],
        "swings": report["swings"],
        "bargainAnalysis": report.get("bargain_analysis") or build_bargain_analysis(swings),
        "badCases": report["bad_cases"],
        "rejectedOrders": report["rejected_orders"],
        "fees": fees,
        "hodlDrawdownPct": _maximum_drawdown(hodl_values),
        "equity": [
            {
                "timestamp": item.get("timestamp"),
                "treasury": _number(item.get("treasury_value")),
                "hodl": _number(item.get("hodl_value")),
                "reserve": _number(item.get("shared_usdt")),
                "openBargains": int(_number(item.get("open_swings"))),
            }
            for item in sampled_equity
        ],
        "monthly": [
            {
                "month": item.get("month"),
                "scroogeReturn": _number(item.get("monthly_return_pct")),
                "hodlReturn": _number(item.get("hodl_monthly_return_pct")),
                "difference": _number(item.get("difference_vs_hodl")),
                "openBargains": int(_number(item.get("ending_open_swings"))),
            }
            for item in monthly
        ],
        "assets": assets,
        "swingHistory": swing_history,
    }


def render_spot_backtest_html(
    report: dict[str, Any],
    equity: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    swings: list[dict[str, Any]] | None = None,
) -> str:
    payload = json.dumps(
        _report_payload(report, equity, monthly, swings or []),
        separators=(",", ":"),
        sort_keys=True,
    ).replace("</", "<\\/")
    scenario = report["scenario"]
    title = escape(
        display_spot_report_title(
            str(scenario.get("name") or "Spot Research"),
            scenario.get("start"),
            scenario.get("end"),
        )
    )
    return _HTML.replace("__REPORT_TITLE__", title).replace("__REPORT_DATA__", payload)


def write_spot_backtest_html(
    path: str | Path,
    report: dict[str, Any],
    equity: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    swings: list[dict[str, Any]] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.write_text(
        render_spot_backtest_html(report, equity, monthly, swings),
        encoding="utf-8",
    )
    return target


def write_report_from_artifacts(artifact_dir: str | Path) -> Path:
    root = Path(artifact_dir).expanduser().resolve()
    summary_path = root / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    with (root / "equity.csv").open("r", encoding="utf-8", newline="") as file_obj:
        equity = list(csv.DictReader(file_obj))
    with (root / "monthly.csv").open("r", encoding="utf-8", newline="") as file_obj:
        monthly = list(csv.DictReader(file_obj))
    swings = json.loads((root / "swings.json").read_text(encoding="utf-8"))
    scenario_name = str(report.get("scenario", {}).get("name") or "")
    if scenario_name.lower().replace("_", "-").startswith("treasury-10-assets-real-quantities"):
        report["scenario"]["name"] = "scrooge-treasury-portfolio-replay"
    report["bargain_analysis"] = build_bargain_analysis(swings)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return write_spot_backtest_html(root / "report.html", report, equity, monthly, swings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained Scrooge Spot research report.")
    parser.add_argument("artifact_dir", help="Directory containing summary.json and equity/monthly CSV files.")
    args = parser.parse_args(argv)
    print(write_report_from_artifacts(args.artifact_dir))
    return 0


_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Scrooge Research | __REPORT_TITLE__</title>
  <style>
    :root {
      --ink: #f3f0e8;
      --muted: #9199a9;
      --dim: #626b7d;
      --ground: #06080c;
      --panel: #0d121a;
      --panel-2: #111824;
      --line: #263142;
      --line-soft: rgba(138, 153, 178, .15);
      --gold: #e9b949;
      --gold-soft: #8d6922;
      --mint: #43d6a0;
      --red: #ff667d;
      --blue: #83a8e8;
      --burgundy: #3b1025;
      --shadow: 0 24px 70px rgba(0, 0, 0, .42);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Courier Prime", "IBM Plex Mono", "Courier New", monospace;
      background:
        radial-gradient(900px 430px at 78% -10%, rgba(128, 20, 48, .32), transparent 66%),
        radial-gradient(700px 420px at 0 30%, rgba(27, 65, 95, .18), transparent 70%),
        repeating-linear-gradient(135deg, rgba(255,255,255,.012) 0 2px, transparent 2px 11px),
        var(--ground);
    }

    button { font: inherit; }
    .shell { width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 70px; }
    .topbar {
      display: flex; align-items: center; justify-content: space-between; gap: 20px;
      border: 1px solid var(--line); border-radius: 18px; padding: 14px 18px;
      background: rgba(10, 14, 21, .9); box-shadow: var(--shadow); position: sticky; top: 12px;
      z-index: 10; backdrop-filter: blur(16px);
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: .06em; }
    .coin {
      width: 38px; height: 38px; display: grid; place-items: center; border-radius: 50%;
      color: #171006; background: linear-gradient(145deg, #ffe39a, var(--gold));
      border: 2px solid #765717; box-shadow: 0 0 22px rgba(233, 185, 73, .2); font-size: 21px;
    }
    .tag { border: 1px solid var(--gold-soft); border-radius: 999px; color: #f4cf79; padding: 6px 10px; font-size: 12px; }

    .hero {
      margin-top: 18px; padding: clamp(22px, 4vw, 48px); border: 1px solid #54273d;
      border-radius: 22px; overflow: hidden; position: relative;
      background: linear-gradient(125deg, rgba(61, 14, 38, .97), rgba(18, 14, 25, .97) 68%);
      box-shadow: var(--shadow);
    }
    .hero::after {
      content: "$"; position: absolute; right: 3%; top: -40%; color: rgba(233,185,73,.055);
      font-size: clamp(220px, 35vw, 520px); font-family: Georgia, serif; transform: rotate(8deg);
    }
    .eyebrow { color: var(--gold); font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }
    h1 { margin: 12px 0 8px; max-width: 900px; font-size: clamp(25px, 4vw, 48px); line-height: 1.08; }
    .period { color: #c4a9b8; font-size: 13px; }
    .verdict { margin-top: 24px; max-width: 760px; color: #f6dce8; font-size: clamp(16px, 2vw, 21px); line-height: 1.55; }

    .metrics { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; margin: 18px 0; }
    .metric {
      min-height: 118px; padding: 15px; border: 1px solid var(--line); border-radius: 15px;
      background: linear-gradient(155deg, rgba(18,25,36,.96), rgba(10,14,21,.98));
      box-shadow: inset 0 1px rgba(255,255,255,.025);
    }
    .metric.primary { grid-column: span 2; border-color: #536987; background: linear-gradient(145deg, #162333, #0b111a); }
    .metric-label { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .metric-value { margin-top: 13px; font-size: clamp(19px, 2vw, 28px); font-weight: 800; letter-spacing: -.03em; }
    .metric-note { margin-top: 6px; color: var(--dim); font-size: 11px; }
    .positive { color: var(--mint) !important; }
    .negative { color: var(--red) !important; }
    .gold { color: var(--gold) !important; }

    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }
    .card {
      grid-column: span 6; border: 1px solid var(--line); border-radius: 18px; padding: 17px;
      background: linear-gradient(180deg, rgba(15,21,31,.96), rgba(9,13,19,.98)); box-shadow: var(--shadow);
    }
    .card.wide { grid-column: span 8; }
    .card.narrow { grid-column: span 4; }
    .card.full { grid-column: 1 / -1; }
    .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
    h2 { margin: 0; font-size: 17px; }
    .subtitle { margin-top: 5px; color: var(--muted); font-size: 11px; line-height: 1.45; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 11px; }
    .legend-item::before { content: ""; width: 8px; height: 8px; display: inline-block; border-radius: 50%; margin-right: 6px; background: var(--series); }

    .chart-wrap { position: relative; height: 310px; }
    .chart-wrap.compact { height: 220px; }
    canvas { display: block; width: 100%; height: 100%; }
    .tooltip {
      position: absolute; pointer-events: none; opacity: 0; min-width: 160px; padding: 9px 11px;
      color: var(--ink); background: rgba(6,9,14,.96); border: 1px solid #3a465b; border-radius: 9px;
      font-size: 11px; line-height: 1.6; box-shadow: 0 12px 30px rgba(0,0,0,.45); transform: translate(-50%, -110%);
    }

    .allocation { display: grid; grid-template-columns: minmax(150px, 220px) 1fr; align-items: center; gap: 22px; }
    .donut { aspect-ratio: 1; border-radius: 50%; position: relative; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }
    .donut::after { content: ""; position: absolute; inset: 25%; border-radius: 50%; background: #0c1119; border: 1px solid var(--line); }
    .donut-center { position: absolute; inset: 0; z-index: 1; display: grid; place-content: center; text-align: center; }
    .donut-center strong { font-size: 24px; color: var(--gold); }
    .donut-center span { color: var(--muted); font-size: 10px; }
    .allocation-list { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 15px; }
    .allocation-row { display: flex; justify-content: space-between; gap: 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line-soft); font-size: 11px; }
    .allocation-name::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 7px; background: var(--swatch); }

    .bar-list { display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 52px 1fr 78px; align-items: center; gap: 10px; font-size: 11px; }
    .bar-track { height: 10px; background: #080c12; border: 1px solid var(--line-soft); border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; width: var(--width); background: var(--color); border-radius: inherit; }
    .bar-value { text-align: right; }

    .bargain-score { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin-bottom: 18px; }
    .score { padding: 12px; border: 1px solid var(--line-soft); border-radius: 12px; background: rgba(8,12,18,.65); }
    .score strong { display: block; margin-top: 7px; font-size: 21px; }
    .score span { color: var(--muted); font-size: 10px; }
    .callout { margin-top: 16px; border-left: 2px solid var(--red); padding: 10px 13px; color: #d7bac3; background: rgba(67,17,33,.27); font-size: 11px; line-height: 1.55; }

    #bargainAnalytics { padding: 22px; }
    #bargainAnalytics h2 { font-size: 22px; }
    #bargainAnalytics > .card-head .subtitle { font-size: 12px; }
    .analysis-kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 17px; }
    .analysis-kpi { min-width: 0; min-height: 104px; padding: 15px; border: 1px solid var(--line-soft); border-radius: 13px; background: linear-gradient(145deg, rgba(17,25,37,.92), rgba(8,12,18,.92)); }
    .analysis-kpi span { display: block; color: var(--muted); font-size: 11px; }
    .analysis-kpi strong { display: block; margin-top: 10px; overflow: hidden; font-size: 22px; text-overflow: ellipsis; }
    .analysis-kpi small { display: block; margin-top: 7px; color: var(--dim); font-size: 10px; line-height: 1.4; }
    .analysis-grid { display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(380px, .82fr); gap: 16px; }
    .analysis-panel { min-width: 0; padding: 17px; border: 1px solid var(--line-soft); border-radius: 15px; background: rgba(7,11,17,.62); }
    .analysis-panel-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; margin-bottom: 14px; }
    .analysis-panel-head strong { font-size: 15px; }
    .analysis-panel-head span { color: var(--muted); font-size: 11px; line-height: 1.45; }
    .analysis-tabs { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 13px; }
    .analysis-tab { border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; color: var(--muted); background: #0b1119; cursor: pointer; font-size: 11px; }
    .analysis-tab:hover, .analysis-tab.active { color: var(--ink); border-color: #5e7da8; background: #17253a; }
    .breakdown-table { display: grid; }
    .breakdown-row { display: grid; grid-template-columns: minmax(120px,1.25fr) 50px 56px 50px 104px 82px 72px; gap: 10px; align-items: center; padding: 11px 5px; border-top: 1px solid var(--line-soft); font-size: 12px; }
    .breakdown-row.header { border-top: 0; color: var(--dim); font-size: 10px; text-transform: uppercase; }
    .breakdown-row > :not(:first-child) { text-align: right; }
    .breakdown-label { display: flex; align-items: center; gap: 9px; min-width: 0; }
    .breakdown-label::before { content: ""; width: 7px; height: 26px; flex: 0 0 auto; border-radius: 4px; background: var(--row-color); }
    .breakdown-label strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .outcome-stack { display: flex; height: 22px; overflow: hidden; border: 1px solid var(--line-soft); border-radius: 999px; background: #080c12; }
    .outcome-slice { width: var(--width); min-width: 2px; background: var(--slice-color); }
    .outcome-list { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 15px; margin-top: 15px; }
    .outcome-item { display: grid; grid-template-columns: 10px 1fr auto; gap: 9px; align-items: center; font-size: 11px; }
    .outcome-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--slice-color); }
    .outcome-item span { color: var(--muted); }
    .scatter-wrap { height: 300px; }
    .risk-list { display: grid; gap: 8px; }
    .risk-row { display: grid; grid-template-columns: 58px minmax(0,1fr) 68px 90px; gap: 10px; align-items: center; padding: 11px 12px; border: 1px solid var(--line-soft); border-radius: 9px; color: inherit; background: #0a0f16; text-decoration: none; font-size: 11px; }
    .risk-row:hover { border-color: #5e718e; background: #101925; }
    .risk-row strong:nth-child(2) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .risk-row span { color: var(--muted); text-align: right; }
    .analysis-note { margin-top: 13px; color: var(--dim); font-size: 11px; line-height: 1.55; }

    .monthly { display: grid; grid-template-columns: repeat(7, minmax(72px, 1fr)); gap: 10px; align-items: end; min-height: 230px; overflow-x: auto; padding-top: 15px; }
    .month { display: grid; grid-template-rows: 180px auto; gap: 8px; min-width: 72px; }
    .month-bars { position: relative; border-bottom: 1px solid var(--line); }
    .month-bars::after { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: 1px dashed rgba(145,153,169,.2); }
    .month-bar { width: 22%; min-width: 12px; height: var(--height); background: var(--color); position: absolute; z-index: 1; }
    .month-bar.scrooge { left: 26%; }
    .month-bar.hodl { right: 26%; }
    .month-bar.up { bottom: 50%; border-radius: 4px 4px 1px 1px; }
    .month-bar.down { top: 50%; border-radius: 1px 1px 4px 4px; }
    .month-label { color: var(--muted); font-size: 10px; text-align: center; }
    .month-return { display: block; color: var(--ink); margin-bottom: 3px; }

    .table-wrap { overflow-x: auto; border: 1px solid var(--line-soft); border-radius: 12px; }
    table { border-collapse: collapse; width: 100%; min-width: 1040px; font-size: 11px; }
    th { color: var(--muted); font-weight: 400; text-align: right; padding: 12px 10px; background: #0a0f16; position: sticky; top: 0; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    td { padding: 12px 10px; border-top: 1px solid var(--line-soft); text-align: right; }
    .asset-summary { cursor: pointer; transition: background 130ms ease; }
    .asset-summary:hover, .asset-summary[aria-expanded="true"] { background: rgba(86,108,143,.1); }
    .asset-summary:focus-visible { outline: 2px solid var(--blue); outline-offset: -2px; }
    .asset { color: var(--ink); font-weight: 800; font-size: 13px; }
    .asset-toggle { display: inline-flex; align-items: center; gap: 9px; }
    .asset-chevron, .swing-chevron { width: 7px; height: 7px; border-right: 1px solid var(--muted); border-bottom: 1px solid var(--muted); transform: rotate(45deg); transition: transform 140ms ease; }
    .asset-summary[aria-expanded="true"] .asset-chevron { transform: rotate(225deg) translate(-2px,-2px); }
    .objective { color: var(--gold); text-transform: capitalize; }
    .asset-detail > td { padding: 0; border-top: 0; text-align: left; background: #080c12; }
    .asset-ledger { padding: 15px; border-top: 1px solid #344158; border-bottom: 1px solid var(--line); }
    .ledger-head { display: flex; justify-content: space-between; align-items: center; gap: 14px; margin-bottom: 12px; }
    .ledger-title { display: grid; gap: 4px; }
    .ledger-title strong { color: var(--ink); font-size: 13px; }
    .ledger-title span { color: var(--muted); font-size: 10px; }
    .ledger-filters { display: flex; gap: 6px; }
    .ledger-filter, .show-more {
      border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; color: var(--muted);
      background: #0d131c; cursor: pointer; font-size: 10px;
    }
    .ledger-filter:hover, .ledger-filter.active, .show-more:hover { color: var(--ink); border-color: #5674a0; background: #152236; }
    .swing-list { display: grid; gap: 8px; }
    .swing-row { overflow: hidden; border: 1px solid rgba(91,126,168,.42); border-radius: 12px; background: radial-gradient(circle at top right, rgba(57,111,169,.08), transparent 36%), #0b1119; }
    .swing-row.closed { border-color: rgba(77,139,108,.38); }
    .swing-toggle {
      display: grid; grid-template-columns: auto minmax(0,1fr) auto auto 10px; align-items: center;
      gap: 11px; width: 100%; border: 0; padding: 10px 11px; color: inherit; background: transparent;
      text-align: left; cursor: pointer; font: inherit;
    }
    .swing-toggle:hover { background: rgba(73,104,145,.08); }
    .swing-toggle[aria-expanded="true"] .swing-chevron { transform: rotate(225deg) translate(-2px,-2px); }
    .bargain-badge { border: 1px solid rgba(92,145,205,.72); border-radius: 999px; padding: 3px 8px; color: #cfe5ff; background: rgba(32,70,108,.3); font-size: 10px; }
    .swing-summary { display: grid; min-width: 0; gap: 3px; }
    .swing-summary strong { font-size: 11px; }
    .swing-summary span { overflow: hidden; color: #aeb9ca; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
    .swing-status { border: 1px solid rgba(83,141,198,.55); border-radius: 999px; padding: 3px 8px; color: #bcd9f7; background: rgba(27,66,103,.24); font-size: 9px; text-transform: uppercase; }
    .swing-status.closed { border-color: rgba(73,151,112,.48); color: #a9e1c2; background: rgba(28,83,56,.24); }
    .swing-pnl { min-width: 82px; text-align: right; font-size: 11px; }
    .swing-details { padding: 12px; border-top: 1px solid var(--line-soft); background: rgba(6,10,15,.72); }
    .swing-metrics { display: grid; grid-template-columns: repeat(8,minmax(100px,1fr)); gap: 7px; }
    .swing-metric { display: grid; gap: 5px; padding: 8px; border: 1px solid var(--line-soft); border-radius: 8px; background: #0c121b; }
    .swing-metric small { color: var(--muted); font-size: 9px; }
    .swing-metric strong { font-size: 10px; }
    .swing-note { margin: 10px 0; padding: 8px 10px; border-left: 2px solid var(--gold-soft); color: #c9b98f; background: rgba(67,48,17,.18); font-size: 10px; }
    .execution-head { display: flex; justify-content: space-between; margin: 10px 0 6px; color: var(--muted); font-size: 10px; }
    .executions { display: grid; gap: 5px; }
    .execution { display: grid; grid-template-columns: 45px minmax(190px,1fr) minmax(120px,.6fr) 145px; gap: 10px; align-items: center; padding: 7px 9px; border: 1px solid var(--line-soft); border-radius: 8px; font-size: 10px; }
    .execution-side { font-weight: 800; }
    .execution-side.buy { color: var(--mint); }
    .execution-side.sell { color: var(--red); }
    .execution time, .execution-fee { color: var(--muted); }
    .swing-footer { display: flex; justify-content: space-between; gap: 15px; margin-top: 10px; color: var(--dim); font-size: 9px; overflow-wrap: anywhere; }
    .ledger-empty { padding: 20px; color: var(--muted); text-align: center; }
    .show-more { display: block; width: 100%; margin-top: 9px; border-radius: 9px; }
    .footer { margin-top: 18px; color: var(--dim); font-size: 10px; line-height: 1.6; text-align: center; }

    @media (max-width: 1050px) {
      .metrics { grid-template-columns: repeat(3, 1fr); }
      .metric.primary { grid-column: span 1; }
      .card.wide, .card.narrow { grid-column: span 6; }
    }
    @media (max-width: 760px) {
      .shell { width: min(100% - 18px, 1440px); padding-top: 9px; }
      .topbar { top: 7px; border-radius: 14px; padding: 11px 12px; }
      .brand-copy small { display: none; }
      .hero { padding: 22px 18px; border-radius: 17px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .metric { min-height: 106px; }
      .metric.primary { grid-column: 1 / -1; }
      .card, .card.wide, .card.narrow { grid-column: 1 / -1; padding: 14px; }
      .chart-wrap { height: 255px; }
      .allocation { grid-template-columns: 1fr; }
      .donut { width: min(230px, 75vw); margin: 0 auto; }
      .bargain-score { grid-template-columns: 1fr 1fr; }
      .analysis-kpis { grid-template-columns: repeat(3, 1fr); }
      .analysis-grid { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 45px 1fr 68px; }
      .metric:last-child { grid-column: 1 / -1; }
      .ledger-head { align-items: flex-start; flex-direction: column; }
      .swing-toggle { grid-template-columns: auto minmax(0,1fr) auto 9px; }
      .swing-toggle .bargain-badge { display: none; }
      .swing-pnl { grid-column: 2; text-align: left; }
      .swing-status { grid-column: 3; grid-row: 1 / span 2; }
      .swing-chevron { grid-column: 4; grid-row: 1 / span 2; }
      .swing-metrics { grid-template-columns: repeat(2,minmax(100px,1fr)); }
      .execution { grid-template-columns: 42px 1fr; }
      .execution-fee, .execution time { grid-column: 2; }
      .swing-footer { flex-direction: column; }
    }
    @media (max-width: 520px) {
      #bargainAnalytics { padding: 16px; }
      .analysis-kpis { grid-template-columns: repeat(2, 1fr); }
      .breakdown-table { overflow-x: auto; }
      .breakdown-row { min-width: 740px; }
      .outcome-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="coin">$</span><span class="brand-copy">SCROOGE RESEARCH<br><small class="muted">CONTROL PLANE / SPOT LAB</small></span></div>
      <span class="tag">COUNTERFACTUAL REPLAY</span>
    </header>

    <section class="hero">
      <div class="eyebrow">The ledger has spoken</div>
      <h1>__REPORT_TITLE__</h1>
      <div class="period" id="period"></div>
      <p class="verdict" id="verdict"></p>
    </section>

    <section class="metrics" id="metrics"></section>

    <section class="grid">
      <article class="card wide">
        <div class="card-head"><div><h2>Vault Value</h2><div class="subtitle">Scrooge strategy against untouched inventory.</div></div><div class="legend"><span class="legend-item" style="--series:var(--gold)">Scrooge</span><span class="legend-item" style="--series:var(--blue)">HODL</span></div></div>
        <div class="chart-wrap"><canvas id="equityChart"></canvas><div class="tooltip"></div></div>
      </article>

      <article class="card narrow">
        <div class="card-head"><div><h2>Final Allocation</h2><div class="subtitle">Where the vault finished.</div></div></div>
        <div class="allocation"><div class="donut" id="allocationDonut"><div class="donut-center"><strong id="reserveShare"></strong><span>Vault Reserve</span></div></div><div class="allocation-list" id="allocationList"></div></div>
      </article>

      <article class="card">
        <div class="card-head"><div><h2>Drawdown</h2><div class="subtitle">Peak-to-trough pressure during the replay.</div></div><div class="legend"><span class="legend-item" style="--series:var(--gold)">Scrooge</span><span class="legend-item" style="--series:var(--blue)">HODL</span></div></div>
        <div class="chart-wrap compact"><canvas id="drawdownChart"></canvas><div class="tooltip"></div></div>
      </article>

      <article class="card">
        <div class="card-head"><div><h2>Vault Reserve</h2><div class="subtitle">Shared USDT generated and redeployed over time.</div></div><div class="legend"><span class="legend-item" style="--series:var(--mint)">USDT</span></div></div>
        <div class="chart-wrap compact"><canvas id="reserveChart"></canvas><div class="tooltip"></div></div>
      </article>

      <article class="card wide">
        <div class="card-head"><div><h2>Monthly Contest</h2><div class="subtitle">Return by month. Bars rise or fall from the center line.</div></div><div class="legend"><span class="legend-item" style="--series:var(--gold)">Scrooge</span><span class="legend-item" style="--series:var(--blue)">HODL</span></div></div>
        <div class="monthly" id="monthly"></div>
      </article>

      <article class="card narrow">
        <div class="card-head"><div><h2>Bargain Health</h2><div class="subtitle">Closed profit versus unfinished exposure.</div></div></div>
        <div class="bargain-score" id="bargainScore"></div>
        <div class="bar-list" id="ageBuckets"></div>
        <div class="callout" id="bargainCallout"></div>
      </article>

      <article class="card full" id="bargainAnalytics">
        <div class="card-head"><div><h2>Bargain Analytics</h2><div class="subtitle">Lifecycle returns, duration, signal quality, and capital still waiting for an exit.</div></div><span class="tag" id="analysisCount"></span></div>
        <div class="analysis-kpis" id="analysisKpis"></div>
        <div class="analysis-grid">
          <section class="analysis-panel">
            <div class="analysis-panel-head"><div><strong>Performance Breakdown</strong><br><span>Compare cohorts on the same lifecycle basis.</span></div></div>
            <div class="analysis-tabs" id="analysisTabs"></div>
            <div class="breakdown-table" id="breakdownTable"></div>
          </section>
          <section class="analysis-panel">
            <div class="analysis-panel-head"><div><strong>Duration vs Return</strong><br><span>Each point is one Bargain. Open exposure is marked in red.</span></div></div>
            <div class="chart-wrap scatter-wrap"><canvas id="bargainScatter"></canvas><div class="tooltip"></div></div>
          </section>
          <section class="analysis-panel">
            <div class="analysis-panel-head"><div><strong>Outcome Mix</strong><br><span>Lifecycle state at the end of the replay.</span></div></div>
            <div class="outcome-stack" id="outcomeStack"></div>
            <div class="outcome-list" id="outcomeList"></div>
            <div class="analysis-note" id="outcomeNote"></div>
          </section>
          <section class="analysis-panel">
            <div class="analysis-panel-head"><div><strong>Open Risk Watchlist</strong><br><span>Largest unfinished losses, linked to asset history.</span></div><span id="riskSummary"></span></div>
            <div class="risk-list" id="riskList"></div>
          </section>
        </div>
      </article>

      <article class="card full">
        <div class="card-head"><div><h2>Asset Contribution</h2><div class="subtitle">Final marked asset value versus holding the starting quantity. Shared USDT is shown separately above.</div></div></div>
        <div class="bar-list" id="assetBars"></div>
      </article>

      <article class="card full">
        <div class="card-head"><div><h2>Portfolio Ledger</h2><div class="subtitle">Market path, Bargains, inventory, and final policy state by asset.</div></div><span class="tag" id="assetCount"></span></div>
        <div class="table-wrap"><table><thead><tr><th>Asset</th><th>Objective</th><th>Market</th><th>vs HODL</th><th>Closed PnL</th><th>Open PnL</th><th>Bargains</th><th>Final / Start</th><th>Target</th><th>Near Floor</th></tr></thead><tbody id="assetTable"></tbody></table></div>
      </article>
    </section>

    <footer class="footer">Research simulation only. Next-candle-open fills use configured fees and slippage; this is not an order-book or microstructure simulation.</footer>
  </main>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("report-data").textContent);
    const COLORS = ["#e9b949", "#43d6a0", "#71a7df", "#e47c58", "#9b7bd4", "#65b8c8", "#d25f7c", "#8eaf68", "#d7923b", "#8994b2", "#d9dce5"];
    const money = (value, digits = 2) => `${value < 0 ? "-" : ""}$${Math.abs(value).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits})}`;
    const pct = value => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
    const qty = value => value.toLocaleString("en-US", {maximumFractionDigits: 4});
    const tone = value => value > 0 ? "positive" : value < 0 ? "negative" : "";
    const shortDate = value => new Date(value).toLocaleDateString("en-GB", {day:"2-digit", month:"short", year:"2-digit"});

    document.getElementById("period").textContent = `${shortDate(data.scenario.start)} to ${shortDate(data.scenario.end)} / ${data.scenario.interval} candles / ${data.scenario.data_source}`;
    const edge = data.portfolio.difference_vs_hodl;
    document.getElementById("verdict").innerHTML = edge >= 0
      ? `My bargains added <strong class="positive">${money(edge)}</strong> beyond simply guarding the coins.`
      : `The vault grew, but untouched coins kept <strong class="negative">${money(Math.abs(edge))}</strong> more. The unfinished bargains reveal where my gold was left waiting.`;

    const metrics = [
      ["Final Treasury", money(data.portfolio.final_treasury_value), `Started at ${money(data.portfolio.starting_treasury_value)}`, "primary gold"],
      ["Total Return", pct(data.portfolio.total_return_pct), `HODL ${pct(data.portfolio.hodl_return_pct)}`, tone(data.portfolio.total_return_pct)],
      ["Edge vs HODL", money(edge), `${(data.portfolio.total_return_pct-data.portfolio.hodl_return_pct).toFixed(2)} pp`, tone(edge)],
      ["Max Drawdown", pct(data.portfolio.maximum_treasury_drawdown_pct), `HODL ${pct(data.hodlDrawdownPct)}`, "negative"],
      ["Vault Reserve", money(data.reserve.ending), `${data.portfolio.final_allocation_pct.USDT.toFixed(2)}% of vault`, ""],
      ["Execution Fees", money(data.fees), `${data.rejectedOrders.toLocaleString()} filtered attempts`, ""],
    ];
    document.getElementById("metrics").innerHTML = metrics.map(([label,value,note,classes]) => `<div class="metric ${classes}"><div class="metric-label">${label}</div><div class="metric-value ${classes}">${value}</div><div class="metric-note">${note}</div></div>`).join("");

    function drawLineChart(canvas, points, series, options = {}) {
      const wrap = canvas.parentElement;
      const tooltip = wrap.querySelector(".tooltip");
      let positions = [];
      function draw() {
        const rect = wrap.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
        const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
        const width = rect.width, height = rect.height, pad = {l:58,r:14,t:10,b:28};
        const values = points.flatMap(point => series.map(item => item.value(point)));
        let min = options.zeroTop ? Math.min(...values, 0) : Math.min(...values);
        let max = options.zeroTop ? 0 : Math.max(...values);
        const gap = Math.max((max-min)*.08, options.money ? 1 : .1); min -= gap; max += options.zeroTop ? 0 : gap;
        const x = index => pad.l + (index / Math.max(points.length-1,1)) * (width-pad.l-pad.r);
        const y = value => pad.t + (max-value) / Math.max(max-min,.0001) * (height-pad.t-pad.b);
        ctx.font = "10px Courier New"; ctx.fillStyle = "#70798a"; ctx.strokeStyle = "rgba(135,149,171,.13)"; ctx.lineWidth = 1;
        for (let i=0;i<5;i++) { const value=min+(max-min)*i/4, yy=y(value); ctx.beginPath(); ctx.moveTo(pad.l,yy); ctx.lineTo(width-pad.r,yy); ctx.stroke(); ctx.fillText(options.money ? money(value,0) : `${value.toFixed(1)}%`, 3, yy+3); }
        series.forEach(item => { ctx.beginPath(); points.forEach((point,index) => { const xx=x(index), yy=y(item.value(point)); index ? ctx.lineTo(xx,yy) : ctx.moveTo(xx,yy); }); ctx.strokeStyle=item.color; ctx.lineWidth=item.width||2; ctx.stroke(); });
        ctx.fillStyle="#70798a"; ctx.fillText(shortDate(points[0].timestamp),pad.l,height-7); const end=shortDate(points.at(-1).timestamp); ctx.fillText(end,width-pad.r-ctx.measureText(end).width,height-7);
        positions = points.map((point,index) => ({x:x(index), point}));
      }
      canvas.addEventListener("mousemove", event => {
        if (!positions.length) return;
        const rect=canvas.getBoundingClientRect(); const mx=event.clientX-rect.left;
        const nearest=positions.reduce((best,item)=>Math.abs(item.x-mx)<Math.abs(best.x-mx)?item:best,positions[0]);
        tooltip.innerHTML=`<strong>${shortDate(nearest.point.timestamp)}</strong><br>${series.map(item=>`${item.name}: ${item.format(item.value(nearest.point))}`).join("<br>")}`;
        tooltip.style.left=`${Math.max(90,Math.min(rect.width-90,nearest.x))}px`; tooltip.style.top="52%"; tooltip.style.opacity=1;
      });
      canvas.addEventListener("mouseleave",()=>tooltip.style.opacity=0);
      new ResizeObserver(draw).observe(wrap); draw();
    }

    function drawBargainScatter(canvas, points) {
      const wrap=canvas.parentElement;
      const tooltip=wrap.querySelector(".tooltip");
      let positions=[];
      function draw() {
        const rect=wrap.getBoundingClientRect();
        const dpr=window.devicePixelRatio||1;
        canvas.width=Math.round(rect.width*dpr); canvas.height=Math.round(rect.height*dpr);
        const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
        const width=rect.width,height=rect.height,pad={l:58,r:15,t:13,b:32};
        const valid=points.filter(point=>Number.isFinite(point.returnPct));
        if (!valid.length) { ctx.fillStyle="#70798a";ctx.font="10px Courier New";ctx.fillText("No Bargains to chart",14,24);return; }
        const maxDays=Math.max(...valid.map(point=>point.ageDays),1);
        let minReturn=Math.min(...valid.map(point=>point.returnPct),0);
        let maxReturn=Math.max(...valid.map(point=>point.returnPct),0);
        const gap=Math.max((maxReturn-minReturn)*.08,1); minReturn-=gap;maxReturn+=gap;
        const x=value=>pad.l+(Math.log1p(value)/Math.log1p(maxDays))*(width-pad.l-pad.r);
        const y=value=>pad.t+(maxReturn-value)/Math.max(maxReturn-minReturn,.0001)*(height-pad.t-pad.b);
        ctx.font="11px Courier New";ctx.lineWidth=1;
        for(let index=0;index<5;index++){const value=minReturn+(maxReturn-minReturn)*index/4,yy=y(value);ctx.strokeStyle="rgba(135,149,171,.13)";ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(width-pad.r,yy);ctx.stroke();ctx.fillStyle="#70798a";ctx.fillText(`${value.toFixed(0)}%`,3,yy+3);}
        const zeroY=y(0);ctx.strokeStyle="rgba(233,185,73,.45)";ctx.beginPath();ctx.moveTo(pad.l,zeroY);ctx.lineTo(width-pad.r,zeroY);ctx.stroke();
        [0,7,30,90,maxDays].filter((value,index,items)=>items.indexOf(value)===index&&value<=maxDays).forEach(value=>{const xx=x(value);ctx.fillStyle="#70798a";ctx.fillText(`${Math.round(value)}d`,Math.max(2,xx-8),height-7);});
        positions=valid.map(point=>{const xx=x(point.ageDays),yy=y(point.returnPct);ctx.beginPath();ctx.arc(xx,yy,point.status==="closed"?3.2:4.1,0,Math.PI*2);ctx.fillStyle=point.status==="closed"?"rgba(67,214,160,.66)":"rgba(255,102,125,.78)";ctx.fill();return{x:xx,y:yy,point};});
      }
      canvas.addEventListener("mousemove",event=>{if(!positions.length)return;const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top;const nearest=positions.reduce((best,item)=>Math.hypot(item.x-mx,item.y-my)<Math.hypot(best.x-mx,best.y-my)?item:best,positions[0]);if(Math.hypot(nearest.x-mx,nearest.y-my)>24){tooltip.style.opacity=0;return;}tooltip.innerHTML=`<strong>${nearest.point.symbol} / ${bargainId(nearest.point.id)}</strong><br>${nearest.point.status==="closed"?"Closed":"Open"}: ${pct(nearest.point.returnPct)}<br>Age: ${nearest.point.ageDays.toFixed(1)}d`;tooltip.style.left=`${Math.max(90,Math.min(rect.width-90,nearest.x))}px`;tooltip.style.top=`${Math.max(55,nearest.y)}px`;tooltip.style.opacity=1;});
      canvas.addEventListener("mouseleave",()=>tooltip.style.opacity=0);
      new ResizeObserver(draw).observe(wrap);draw();
    }

    const drawdowns = (() => { let treasuryPeak=0, hodlPeak=0; return data.equity.map(point => { treasuryPeak=Math.max(treasuryPeak,point.treasury); hodlPeak=Math.max(hodlPeak,point.hodl); return {...point, treasuryDd:(point.treasury/treasuryPeak-1)*100, hodlDd:(point.hodl/hodlPeak-1)*100}; }); })();
    drawLineChart(document.getElementById("equityChart"), data.equity, [
      {name:"Scrooge",color:"#e9b949",value:p=>p.treasury,format:v=>money(v)},
      {name:"HODL",color:"#83a8e8",value:p=>p.hodl,format:v=>money(v)},
    ], {money:true});
    drawLineChart(document.getElementById("drawdownChart"), drawdowns, [
      {name:"Scrooge",color:"#e9b949",value:p=>p.treasuryDd,format:pct},
      {name:"HODL",color:"#83a8e8",value:p=>p.hodlDd,format:pct},
    ], {zeroTop:true});
    drawLineChart(document.getElementById("reserveChart"), data.equity, [
      {name:"Reserve",color:"#43d6a0",value:p=>p.reserve,format:v=>money(v)},
    ], {money:true});

    const allocations = Object.entries(data.portfolio.final_allocation_pct).sort((a,b)=>b[1]-a[1]);
    let cursor=0; const slices=[];
    allocations.forEach(([name,value],index)=>{ const next=cursor+value; slices.push(`${COLORS[index%COLORS.length]} ${cursor}% ${next}%`); cursor=next; });
    document.getElementById("allocationDonut").style.background=`conic-gradient(${slices.join(",")})`;
    document.getElementById("reserveShare").textContent=`${data.portfolio.final_allocation_pct.USDT.toFixed(1)}%`;
    document.getElementById("allocationList").innerHTML=allocations.map(([name,value],index)=>`<div class="allocation-row"><span class="allocation-name" style="--swatch:${COLORS[index%COLORS.length]}">${name}</span><strong>${value.toFixed(2)}%</strong></div>`).join("");

    const maxMonthly=Math.max(...data.monthly.flatMap(item=>[Math.abs(item.scroogeReturn),Math.abs(item.hodlReturn)]),1);
    document.getElementById("monthly").innerHTML=data.monthly.map(item=>{ const bar=(value,color,name)=>`<span class="month-bar ${name} ${value>=0?"up":"down"}" title="${pct(value)}" style="--height:${Math.max(Math.abs(value)/maxMonthly*45,1)}%;--color:${color}"></span>`; return `<div class="month"><div class="month-bars">${bar(item.scroogeReturn,"#e9b949","scrooge")}${bar(item.hodlReturn,"#83a8e8","hodl")}</div><div class="month-label"><strong class="month-return ${tone(item.scroogeReturn)}">${pct(item.scroogeReturn)}</strong>${item.month}</div></div>`; }).join("");

    const swing=data.swings;
    document.getElementById("bargainScore").innerHTML=[
      ["Opened",swing.total_opened,""], ["Closed",swing.total_closed,"positive"], ["Still Open",swing.still_open,"negative"],
      ["Closed PnL",money(swing.realized_pnl_quote),"positive"], ["Open PnL",money(swing.unrealized_open_pnl_quote),tone(swing.unrealized_open_pnl_quote)], ["Oldest",`${swing.oldest_open_days.toFixed(0)}d`,"negative"],
    ].map(([label,value,cls])=>`<div class="score"><span>${label}</span><strong class="${cls}">${value}</strong></div>`).join("");
    const ageLabels={"under_7_days":"< 7d","7_to_30_days":"7-30d","30_to_90_days":"30-90d","90_to_180_days":"90-180d","180_plus_days":"180d+"};
    const maxAge=Math.max(...Object.values(swing.open_age_buckets),1);
    document.getElementById("ageBuckets").innerHTML=Object.entries(swing.open_age_buckets).map(([key,value])=>`<div class="bar-row"><span>${ageLabels[key]}</span><div class="bar-track"><div class="bar-fill" style="--width:${value/maxAge*100}%;--color:${key.includes("180")?"var(--red)":"var(--gold)"}"></div></div><span class="bar-value">${value}</span></div>`).join("");
    document.getElementById("bargainCallout").textContent=`${data.badCases.underwater_sell_origin_count} underwater sell-origin Bargains need ${money(data.badCases.value_required_to_restore)} to restore ${qty(data.badCases.quantity_sold_not_restored)} units.`;

    const maxDelta=Math.max(...data.assets.map(item=>Math.abs(item.deltaVsHodl)),1);
    document.getElementById("assetBars").innerHTML=data.assets.slice().sort((a,b)=>b.deltaVsHodl-a.deltaVsHodl).map(item=>`<div class="bar-row"><strong>${item.symbol}</strong><div class="bar-track"><div class="bar-fill" style="--width:${Math.abs(item.deltaVsHodl)/maxDelta*100}%;--color:${item.deltaVsHodl>=0?"var(--mint)":"var(--red)"}"></div></div><strong class="bar-value ${tone(item.deltaVsHodl)}">${money(item.deltaVsHodl)}</strong></div>`).join("");

    const objective=value=>value ? value.replace("accumulate_","").replace("_"," ") : "Protected";
    const objectiveLabel=value=>value ? value.replace("accumulate_","Accumulate ").replace("_"," ").replace(/\b\w/g,letter=>letter.toUpperCase()) : "Protected";
    const dateTime=value=>value ? new Date(value).toLocaleString("en-GB",{day:"2-digit",month:"2-digit",year:"numeric",hour:"2-digit",minute:"2-digit",hour12:false}) : "Open";
    const bargainId=value=>`Bargain #${value.replaceAll("-","").slice(-6).toUpperCase()}`;
    const age=value=>{ const total=Math.max(0,Math.floor(value)); const days=Math.floor(total/86400); const hours=Math.floor((total%86400)/3600); return days?`${days}d ${hours}h`:hours?`${hours}h ${Math.floor((total%3600)/60)}m`:`${Math.floor(total/60)}m`; };
    const statusLabel=value=>value.replaceAll("_"," ").toUpperCase();

    const analysis=data.bargainAnalysis;
    const analysisOverview=analysis.overview;
    const optionalPct=value=>value===null||value===undefined?"N/A":pct(value);
    const hoursLabel=value=>value===null||value===undefined?"N/A":value>=24?`${(value/24).toFixed(1)}d`:`${value.toFixed(1)}h`;
    document.getElementById("analysisCount").textContent=`${analysisOverview.count} BARGAINS`;
    document.getElementById("analysisKpis").innerHTML=[
      ["Lifecycle PnL",money(analysisOverview.net_pnl_quote),`${money(analysisOverview.realized_pnl_quote)} realized`,tone(analysisOverview.net_pnl_quote)],
      ["Closure Rate",optionalPct(analysisOverview.closure_rate_pct),`${analysisOverview.open} still open`,""],
      ["Closed Expectancy",money(analysisOverview.closed_expectancy_quote),`${optionalPct(analysisOverview.average_closed_return_pct)} average return`,tone(analysisOverview.closed_expectancy_quote)],
      ["Median Duration",hoursLabel(analysisOverview.median_duration_hours),`P90 ${hoursLabel(analysisOverview.duration_p90_hours)}`,""],
      ["Fee Drag",optionalPct(analysisOverview.fee_drag_pct),`${money(analysisOverview.quote_fees)} quote fees`,""],
      ["Open Risk",money(analysis.risk.underwater_open_pnl_quote),`${analysis.risk.underwater_open_count} underwater`,"negative"],
    ].map(([label,value,note,cls])=>`<div class="analysis-kpi"><span>${label}</span><strong class="${cls}">${value}</strong><small>${note}</small></div>`).join("");

    const dimensions={asset:"Asset",objective:"Objective",origin:"Origin",signal_level:"Signal Level",conviction:"Conviction",duration:"Duration"};
    const analysisTabs=document.getElementById("analysisTabs");
    let activeDimension="asset";
    analysisTabs.innerHTML=Object.entries(dimensions).map(([key,label])=>`<button type="button" class="analysis-tab ${key===activeDimension?"active":""}" data-dimension="${key}">${label}</button>`).join("");
    function renderBreakdown() {
      const rows=analysis.breakdowns[activeDimension]||[];
      document.getElementById("breakdownTable").innerHTML=`<div class="breakdown-row header"><span>Category</span><span>Count</span><span>Closed</span><span>Open</span><span>Lifecycle PnL</span><span>Return</span><span>Median</span></div>${rows.map((row,index)=>`<div class="breakdown-row"><span class="breakdown-label" style="--row-color:${COLORS[index%COLORS.length]}"><strong>${row.label}</strong></span><span>${row.count}</span><span>${row.closed}</span><span>${row.open}</span><strong class="${tone(row.net_pnl_quote)}">${money(row.net_pnl_quote)}</strong><span class="${tone(row.return_on_notional_pct||0)}">${optionalPct(row.return_on_notional_pct)}</span><span>${hoursLabel(row.median_duration_hours)}</span></div>`).join("")}`;
    }
    analysisTabs.querySelectorAll(".analysis-tab").forEach(button=>button.addEventListener("click",()=>{activeDimension=button.dataset.dimension;analysisTabs.querySelectorAll(".analysis-tab").forEach(item=>item.classList.toggle("active",item===button));renderBreakdown();}));
    renderBreakdown();

    const outcomeColors={closed_profit:"#43d6a0",closed_flat:"#738096",closed_loss:"#ff667d",open_profit:"#83a8e8",open_flat:"#9b7bd4",open_underwater:"#b83352"};
    const outcomes=analysis.breakdowns.outcome||[];
    const outcomeTotal=Math.max(outcomes.reduce((total,row)=>total+row.count,0),1);
    document.getElementById("outcomeStack").innerHTML=outcomes.map(row=>`<span class="outcome-slice" title="${row.label}: ${row.count}" style="--width:${row.count/outcomeTotal*100}%;--slice-color:${outcomeColors[row.key]||"#738096"}"></span>`).join("");
    document.getElementById("outcomeList").innerHTML=outcomes.map(row=>`<div class="outcome-item"><i class="outcome-dot" style="--slice-color:${outcomeColors[row.key]||"#738096"}"></i><span>${row.label}</span><strong class="${tone(row.net_pnl_quote)}">${row.count} / ${money(row.net_pnl_quote)}</strong></div>`).join("");
    const feeAssets=Object.entries(analysisOverview.fees_by_asset).map(([asset,value])=>`${qty(value)} ${asset}`).join(" / ")||"none";
    document.getElementById("outcomeNote").textContent=`${analysisOverview.partially_closed} partially closed. Fees remain denominated in their actual assets: ${feeAssets}.`;

    const risks=analysis.notable.worst_open||[];
    document.getElementById("riskSummary").textContent=`${analysis.risk.open_90_plus_days} open 90d+`;
    document.getElementById("riskList").innerHTML=risks.length?risks.map(item=>`<a class="risk-row" href="#asset-${item.asset_symbol}" data-risk-asset="${item.asset_symbol}"><strong>${item.asset_symbol}</strong><strong>${bargainId(item.swing_id)}</strong><span>${item.age_days.toFixed(1)}d</span><strong class="${tone(item.pnl_quote)}">${money(item.pnl_quote)}</strong></a>`).join(""):`<div class="ledger-empty">No open Bargains are underwater.</div>`;

    const bargainPoints=Object.entries(data.swingHistory).flatMap(([symbol,items])=>items.map(item=>({symbol,id:item.id,status:item.status,ageDays:item.ageSeconds/86400,returnPct:item.returnPct})));
    drawBargainScatter(document.getElementById("bargainScatter"),bargainPoints);

    const assetTable=document.getElementById("assetTable");
    const ledgerState={};
    document.getElementById("assetCount").textContent=`${data.assets.length} ASSETS`;
    assetTable.innerHTML=data.assets.map(item=>`
      <tr class="asset-summary" id="asset-${item.symbol}" data-asset="${item.symbol}" tabindex="0" role="button" aria-expanded="false" aria-controls="ledger-${item.symbol}">
        <td class="asset"><span class="asset-toggle">${item.symbol}<i class="asset-chevron" aria-hidden="true"></i></span></td>
        <td class="objective">${objective(item.objective)}</td><td class="${tone(item.marketReturn)}">${pct(item.marketReturn)}</td><td class="${tone(item.deltaVsHodl)}">${money(item.deltaVsHodl)}</td><td class="positive">${money(item.realizedPnl)}</td><td class="${tone(item.unrealizedPnl)}">${money(item.unrealizedPnl)}</td><td>${item.closed} / ${item.opened}<br><span class="muted">${item.open} open</span></td><td>${qty(item.finalQuantity)} / ${qty(item.startingQuantity)}</td><td>${qty(item.targetQuantity)}</td><td>${item.nearFloorPct.toFixed(1)}%</td>
      </tr>
      <tr class="asset-detail" id="ledger-${item.symbol}" hidden><td colspan="10"><div class="asset-ledger" data-ledger="${item.symbol}"></div></td></tr>`).join("");

    function executionRows(swing, symbol) {
      if (!swing.executions.length) return `<div class="ledger-empty">No executions reached this Bargain.</div>`;
      return `<div class="executions">${swing.executions.map(fill=>`<div class="execution"><strong class="execution-side ${fill.side}">${fill.side.toUpperCase()}</strong><strong>${qty(fill.quantity)} ${symbol} at ${money(fill.price,6)}</strong><span class="execution-fee">Fee ${fill.feeAmount?`${qty(fill.feeAmount)} ${fill.feeAsset||"Unknown"}`:"none"}</span><time>${dateTime(fill.executedAt)}</time></div>`).join("")}</div>`;
    }

    function swingRow(swing, symbol) {
      const pnl=swing.status==="closed"?swing.realizedPnl:swing.unrealizedPnl;
      const fees=Object.entries(swing.fees).map(([asset,value])=>`${qty(value)} ${asset}`).join(" / ")||"None";
      const note=[swing.signalLevel?`Level ${swing.signalLevel}`:null,swing.rollingChangePct!==null&&swing.rollingChangePct!==undefined?`${pct(swing.rollingChangePct)} rolling move`:null,swing.sizingModifier?`${swing.sizingModifier}x sizing`:null].filter(Boolean).join(" / ");
      return `<article class="swing-row ${swing.status}">
        <button class="swing-toggle" type="button" aria-expanded="false">
          <span class="bargain-badge">Bargain</span><span class="swing-summary"><strong>${bargainId(swing.id)}</strong><span>${swing.originSide.toUpperCase()} ${qty(swing.openingQuantity)} ${symbol}${swing.openingPrice!==null?` at ${money(swing.openingPrice,6)}`:""}</span></span><span class="swing-status ${swing.status}">${statusLabel(swing.status)}</span><strong class="swing-pnl ${tone(pnl)}">${money(pnl)}</strong><i class="swing-chevron" aria-hidden="true"></i>
        </button>
        <div class="swing-details" hidden>
          <div class="swing-metrics">
            <span class="swing-metric"><small>Origin</small><strong>${swing.originSide.toUpperCase()}</strong></span>
            <span class="swing-metric"><small>Objective</small><strong>${objectiveLabel(swing.objective)}</strong></span>
            <span class="swing-metric"><small>Remaining</small><strong>${qty(swing.remainingQuantity)} ${symbol}</strong></span>
            <span class="swing-metric"><small>Opened</small><strong>${dateTime(swing.openedAtMs)}</strong></span>
            <span class="swing-metric"><small>Closed</small><strong>${dateTime(swing.closedAtMs)}</strong></span>
            <span class="swing-metric"><small>Age</small><strong>${age(swing.ageSeconds)}</strong></span>
            <span class="swing-metric"><small>Realized PnL</small><strong class="${tone(swing.realizedPnl)}">${money(swing.realizedPnl)}</strong></span>
            <span class="swing-metric"><small>Open PnL</small><strong class="${tone(swing.unrealizedPnl)}">${money(swing.unrealizedPnl)}</strong></span>
          </div>
          ${note?`<p class="swing-note">Scrooge's note: ${note}</p>`:""}
          <div class="execution-head"><strong>Executions</strong><span>${swing.executions.length} ${swing.executions.length===1?"fill":"fills"}</span></div>
          ${executionRows(swing,symbol)}
          <footer class="swing-footer"><span>Full ID <strong>${swing.id}</strong></span><span>Fees <strong>${fees}</strong></span></footer>
        </div>
      </article>`;
    }

    function renderAssetLedger(symbol) {
      const panel=document.querySelector(`[data-ledger="${symbol}"]`);
      const history=data.swingHistory[symbol]||[];
      const state=ledgerState[symbol]||(ledgerState[symbol]={filter:"all",limit:20});
      const filtered=state.filter==="all"?history:history.filter(item=>state.filter==="open"?item.status!=="closed":item.status==="closed");
      const visible=filtered.slice(0,state.limit);
      panel.innerHTML=`<div class="ledger-head"><div class="ledger-title"><strong>${symbol} Bargain History</strong><span>${filtered.length} ${filtered.length===1?"Bargain":"Bargains"}, newest first</span></div><div class="ledger-filters">${["all","open","closed"].map(filter=>`<button type="button" class="ledger-filter ${state.filter===filter?"active":""}" data-filter="${filter}">${filter[0].toUpperCase()+filter.slice(1)}</button>`).join("")}</div></div>${visible.length?`<div class="swing-list">${visible.map(item=>swingRow(item,symbol)).join("")}</div>`:`<div class="ledger-empty">No ${state.filter==="all"?"":state.filter} Bargains for ${symbol}.</div>`}${visible.length<filtered.length?`<button type="button" class="show-more">Show ${Math.min(20,filtered.length-visible.length)} more / ${filtered.length-visible.length} remaining</button>`:""}`;
      panel.querySelectorAll(".ledger-filter").forEach(button=>button.addEventListener("click",()=>{state.filter=button.dataset.filter;state.limit=20;renderAssetLedger(symbol);}));
      panel.querySelector(".show-more")?.addEventListener("click",()=>{state.limit+=20;renderAssetLedger(symbol);});
      panel.querySelectorAll(".swing-toggle").forEach(button=>button.addEventListener("click",()=>{const details=button.nextElementSibling;const expanded=button.getAttribute("aria-expanded")==="true";button.setAttribute("aria-expanded",String(!expanded));details.hidden=expanded;}));
    }

    function toggleAssetRow(row) {
      const expanded=row.getAttribute("aria-expanded")==="true";
      const detail=row.nextElementSibling;
      row.setAttribute("aria-expanded",String(!expanded)); detail.hidden=expanded;
      if (!expanded && !detail.dataset.rendered) { renderAssetLedger(row.dataset.asset); detail.dataset.rendered="true"; }
    }
    assetTable.querySelectorAll(".asset-summary").forEach(row=>{
      row.addEventListener("click",()=>toggleAssetRow(row));
      row.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();toggleAssetRow(row);}});
    });
    document.querySelectorAll("[data-risk-asset]").forEach(link=>link.addEventListener("click",event=>{
      event.preventDefault();
      const row=document.getElementById(`asset-${link.dataset.riskAsset}`);
      if(!row)return;
      if(row.getAttribute("aria-expanded")!=="true")toggleAssetRow(row);
      history.replaceState(null,"",link.getAttribute("href"));
      setTimeout(()=>window.scrollTo({top:row.getBoundingClientRect().top+window.scrollY-24,behavior:"smooth"}),30);
    }));
    const linkedAsset=location.hash.startsWith("#asset-")?document.getElementById(location.hash.slice(1)):null;
    if (linkedAsset) { toggleAssetRow(linkedAsset); setTimeout(()=>window.scrollTo({top:linkedAsset.getBoundingClientRect().top+window.scrollY-24}),150); }
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
