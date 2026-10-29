"""Build a standalone interactive report for one cleanup-policy backtest run."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from datetime import UTC, datetime
import html
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
HOUR_MS = 60 * 60 * 1000
REASON_ORDER = (
    "capacity_cleanup",
    "age_l3_cleanup",
    "age_l2_cleanup",
    "age_l1_cleanup",
    "deep_loss_cleanup",
)
REASON_LABELS = {
    "capacity_cleanup": "Capacity",
    "age_l3_cleanup": "Aging L3",
    "age_l2_cleanup": "Aging L2",
    "age_l1_cleanup": "Aging L1",
    "deep_loss_cleanup": "Deep loss",
}
REASON_COLORS = {
    "capacity_cleanup": "#53d6a7",
    "age_l3_cleanup": "#e7b94b",
    "age_l2_cleanup": "#77aee8",
    "age_l1_cleanup": "#a78bd4",
    "deep_loss_cleanup": "#ff667f",
}
AGE_BINS = (
    ("< 7d", 0, 7),
    ("7-14d", 7, 14),
    ("14-30d", 14, 30),
    ("30-45d", 30, 45),
    ("45-60d", 45, 60),
    ("60-90d", 60, 90),
    ("90-120d", 90, 120),
    ("120d+", 120, math.inf),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float:
    return float(value or 0)


def _iso_from_ms(value: int | float | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(float(value) / 1000, tz=UTC).isoformat()


def _event_rows(swings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for swing in swings:
        reason = str(swing.get("close_reason") or "")
        if reason not in REASON_LABELS:
            continue
        context = swing.get("close_context") or {}
        economics = swing.get("economics") or {}
        executions = []
        raw_executions = swing.get("executions") or []
        for index, execution in enumerate(raw_executions):
            execution_reason = execution.get("reason") or {}
            phase = str(execution_reason.get("action_type") or "")
            if not phase:
                phase = "open" if index == 0 else "close" if index == len(raw_executions) - 1 else "fill"
            executions.append(
                {
                    "at": str(execution.get("executed_at") or ""),
                    "at_ms": int(execution.get("executed_at_ms") or 0),
                    "phase": phase,
                    "side": str(execution.get("side") or ""),
                    "quantity": _number(execution.get("quantity")),
                    "price": _number(execution.get("price")),
                    "quote": _number(execution.get("quote_quantity")),
                    "fee": _number(execution.get("fee_amount")),
                    "fee_asset": str(execution.get("fee_asset") or ""),
                    "signal": execution_reason.get("signal"),
                    "signal_level": execution_reason.get("signal_level"),
                    "close_reason": execution_reason.get("close_reason"),
                }
            )
        age_days = _number(swing.get("age_seconds")) / 86_400
        closed_at_ms = int(swing.get("closed_at_ms") or 0)
        events.append(
            {
                "date": _iso_from_ms(closed_at_ms)[:10],
                "month": _iso_from_ms(closed_at_ms)[:7],
                "asset": str(swing.get("asset_symbol") or "?"),
                "reason": reason,
                "reason_label": REASON_LABELS[reason],
                "age_days": age_days,
                "pnl_quote": _number(economics.get("realized_pnl_quote")),
                "pnl_pct_before": context.get("unrealized_pnl_pct_before_cleanup"),
                "required_level": context.get("required_cleanup_level"),
                "actual_level": context.get("actual_reverse_signal_level"),
                "objective": str(swing.get("trading_objective") or "unknown"),
                "swing_id": str(swing.get("swing_id") or ""),
                "origin_side": str(swing.get("origin_side") or ""),
                "opened_at_ms": int(swing.get("opened_at_ms") or 0),
                "closed_at_ms": closed_at_ms,
                "executions": executions,
                "economics": {
                    "opening_quote": _number(economics.get("opening_quote_quantity")),
                    "closing_quote": _number(economics.get("closing_quote_quantity")),
                    "gross_pnl": _number(economics.get("realized_gross_pnl_quote")),
                    "fees": _number(economics.get("realized_fee_quote")),
                    "asset_change": _number(economics.get("realized_net_asset_change")),
                    "asset_gain": _number(economics.get("realized_asset_gain")),
                    "target_ratchet": _number(economics.get("target_ratchet_quantity")),
                    "unrecovered_quantity": _number(economics.get("unrecovered_quantity")),
                    "weighted_opening_price": economics.get("weighted_opening_price"),
                    "weighted_closing_price": economics.get("weighted_closing_price"),
                },
            }
        )
    return sorted(events, key=lambda event: (event["date"], event["swing_id"]))


def _covering_candle_cache(
    cache_dir: Path,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
) -> Path | None:
    prefix = f"{symbol.upper()}USDT-1m-"
    candidates: list[tuple[int, Path]] = []
    for path in (cache_dir / "klines").glob(f"{prefix}*.csv"):
        bounds = path.stem[len(prefix):].split("-", 1)
        if len(bounds) != 2:
            continue
        try:
            cached_start, cached_end = (int(value) for value in bounds)
        except ValueError:
            continue
        if cached_start <= start_ms and cached_end >= end_ms:
            candidates.append((cached_end - cached_start, path))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _hourly_candles(
    path: Path,
    *,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["open_time_ms"])
            if timestamp < start_ms:
                continue
            if timestamp >= end_ms:
                break
            bucket = timestamp // HOUR_MS * HOUR_MS
            open_value = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if current is None or current["t"] != bucket:
                if current is not None:
                    output.append(current)
                current = {"t": bucket, "o": open_value, "h": high, "l": low, "c": close}
                continue
            current["h"] = max(float(current["h"]), high)
            current["l"] = min(float(current["l"]), low)
            current["c"] = close
    if current is not None:
        output.append(current)
    return output


def _compress_candles(
    candles: list[dict[str, float | int]],
    *,
    max_points: int = 120,
) -> tuple[list[dict[str, float | int]], int]:
    if not candles:
        return [], 1
    step = max(1, math.ceil(len(candles) / max_points))
    output = []
    for offset in range(0, len(candles), step):
        group = candles[offset : offset + step]
        output.append(
            {
                "t": group[0]["t"],
                "o": group[0]["o"],
                "h": max(float(row["h"]) for row in group),
                "l": min(float(row["l"]) for row in group),
                "c": group[-1]["c"],
            }
        )
    return output, step


def _attach_price_paths(events: list[dict[str, Any]], cache_dir: Path) -> dict[str, Any]:
    available_assets = []
    missing_assets = []
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_asset.setdefault(event["asset"], []).append(event)
    for asset, asset_events in sorted(by_asset.items()):
        start_ms = min(event["opened_at_ms"] for event in asset_events)
        end_ms = max(event["closed_at_ms"] for event in asset_events) + HOUR_MS
        cache_path = _covering_candle_cache(
            cache_dir,
            asset,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if cache_path is None:
            missing_assets.append(asset)
            continue
        candles = _hourly_candles(cache_path, start_ms=start_ms, end_ms=end_ms)
        timestamps = [int(candle["t"]) for candle in candles]
        for event in asset_events:
            left = bisect_left(timestamps, event["opened_at_ms"] // HOUR_MS * HOUR_MS)
            right = bisect_right(timestamps, event["closed_at_ms"])
            compressed, step = _compress_candles(candles[left:right])
            event["price_path"] = compressed
            event["price_interval_hours"] = step
        available_assets.append(asset)
    return {
        "source": "cached Binance Spot 1m OHLC aggregated to 1h",
        "available_assets": available_assets,
        "missing_assets": missing_assets,
        "max_candles_per_swing": 120,
    }


def _flush_inventory_snapshot(
    state: dict[str, Any] | None,
    daily: dict[str, dict[str, Any]],
    stats: dict[str, float],
    capacity: int,
) -> None:
    if not state or not state["assets"]:
        return
    open_count = int(state["open_count"])
    point = {
        "date": state["date"],
        "open": open_count,
        "underwater": int(state["underwater"]),
        "utilization": mean(state["utilization"]),
    }
    daily[state["date"]] = point
    stats["samples"] += 1
    stats["open_sum"] += open_count
    stats["near_cap"] += open_count >= capacity * 0.9
    stats["at_cap"] += open_count >= capacity


def _capacity_series(path: Path, *, capacity: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    daily: dict[str, dict[str, Any]] = {}
    stats = {"samples": 0.0, "open_sum": 0.0, "near_cap": 0.0, "at_cap": 0.0}
    current: dict[str, Any] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = row["timestamp_ms"]
            if current is None or timestamp != current["timestamp"]:
                _flush_inventory_snapshot(current, daily, stats, capacity)
                current = {
                    "timestamp": timestamp,
                    "date": row["timestamp"][:10],
                    "assets": set(),
                    "open_count": 0,
                    "underwater": 0,
                    "utilization": [],
                }
            current["assets"].add(row["asset_symbol"])
            current["open_count"] += int(float(row["open_bargain_count"] or 0))
            current["underwater"] += int(float(row["underwater_sell_origin_swings"] or 0))
            current["utilization"].append(_number(row["tradable_inventory_utilization_pct"]))
    _flush_inventory_snapshot(current, daily, stats, capacity)
    samples = stats["samples"] or 1
    summary = {
        "average_open": stats["open_sum"] / samples,
        "near_cap_pct": stats["near_cap"] / samples * 100,
        "at_cap_pct": stats["at_cap"] / samples * 100,
    }
    return [daily[key] for key in sorted(daily)], summary


def _monthly_rows(
    path: Path,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_month: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_month[row["month"]] = {
                "month": row["month"],
                "treasury": _number(row["treasury_end_value"]),
                "difference_vs_hodl": _number(row["difference_vs_hodl"]),
                "ending_open": int(float(row["ending_open_swings"] or 0)),
                "cleanup_pnl": 0.0,
                "cleanup_count": 0,
                "reasons": {reason: 0 for reason in REASON_ORDER},
            }
    for event in events:
        month = event["month"]
        if month not in by_month:
            continue
        by_month[month]["cleanup_pnl"] += event["pnl_quote"]
        by_month[month]["cleanup_count"] += 1
        by_month[month]["reasons"][event["reason"]] += 1
    return [by_month[key] for key in sorted(by_month)]


def _age_histogram(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, lower, upper in AGE_BINS:
        counts = {reason: 0 for reason in REASON_ORDER}
        for event in events:
            if lower <= event["age_days"] < upper:
                counts[event["reason"]] += 1
        rows.append({"label": label, "counts": counts})
    return rows


def _reason_rows(
    cleanup: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for reason in REASON_ORDER:
        metrics = (cleanup.get("by_reason") or {}).get(reason) or {}
        reason_events = [event for event in events if event["reason"] == reason]
        ages = [event["age_days"] for event in reason_events]
        output.append(
            {
                "reason": reason,
                "label": REASON_LABELS[reason],
                "color": REASON_COLORS[reason],
                "closes": int(metrics.get("closes") or 0),
                "lifecycle_pnl": _number(metrics.get("lifecycle_pnl_quote")),
                "cash_change": _number(metrics.get("cash_change_quote")),
                "reserve_deployed": _number(metrics.get("reserve_deployed_quote")),
                "cash_released": _number(metrics.get("cash_released_quote")),
                "inventory_deficit": _number(metrics.get("inventory_deficit_market_value_quote")),
                "average_age_days": mean(ages) if ages else None,
                "median_age_days": median(ages) if ages else None,
                "average_repurchase": metrics.get("average_repurchase_spend_quote"),
            }
        )
    return output


def _asset_rows(cleanup: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for asset, metrics in sorted((cleanup.get("per_asset") or {}).items()):
        accounting = metrics.get("accounting") or {}
        output.append(
            {
                "asset": asset,
                "closes": int(metrics.get("cleanup_closes") or 0),
                "lifecycle_pnl": _number(accounting.get("lifecycle_pnl_quote")),
                "cash_change": _number(accounting.get("cash_change_quote")),
                "inventory_deficit": _number(
                    accounting.get("inventory_deficit_market_value_quote")
                ),
                "blocked": int(metrics.get("unique_capacity_blocked_opportunities") or 0),
                "average_open": _number(metrics.get("average_open_bargain_count")),
                "maximum_open": int(metrics.get("maximum_open_bargain_count") or 0),
                "ending_open": int(metrics.get("open_bargains_at_end") or 0),
            }
        )
    return output


def _policy(summary: dict[str, Any]) -> dict[str, Any]:
    policy = summary["waiter_cleanup"]["policy"]
    aging = sorted(policy["aging"], key=lambda row: row["required_reverse_level"], reverse=True)
    return {
        "max_open": policy["max_open_bargains_per_asset"],
        "capacity_age": policy["capacity_cleanup"]["min_age_days"],
        "capacity_reverse": policy["capacity_cleanup"].get("required_reverse_level", 1),
        "deep_age": policy["deep_loss"]["min_age_days"],
        "deep_pnl": policy["deep_loss"]["unrealized_pnl_pct"],
        "deep_reverse": policy["deep_loss"]["required_reverse_level"],
        "aging": [
            {
                "age": row["min_age_days"],
                "level": row["required_reverse_level"],
            }
            for row in aging
        ],
    }


def _insights(payload: dict[str, Any]) -> list[dict[str, str]]:
    reasons = {row["reason"]: row for row in payload["reasons"]}
    deep = reasons["deep_loss_cleanup"]
    capacity = reasons["capacity_cleanup"]
    negatives = sum(abs(min(0, row["lifecycle_pnl"])) for row in payload["reasons"])
    deep_share = abs(deep["lifecycle_pnl"]) / negatives * 100 if negatives else 0
    assets = sorted(payload["assets"], key=lambda row: row["lifecycle_pnl"])
    cash = payload["cash"]
    capacity_stats = payload["capacity"]["summary"]
    return [
        {
            "tone": "red",
            "eyebrow": "Primary drag",
            "title": f"Deep loss carries {deep_share:.1f}% of cleanup losses",
            "body": (
                f"{deep['closes']:,} closes produced ${abs(deep['lifecycle_pnl']):,.0f} "
                f"of lifecycle loss at an average age of {deep['average_age_days']:.1f} days."
            ),
        },
        {
            "tone": "green",
            "eyebrow": "Healthy mechanism",
            "title": f"Capacity cleanup earned ${capacity['lifecycle_pnl']:,.0f}",
            "body": (
                f"It cleared {capacity['closes']:,} Bargains without an inventory deficit. "
                "Capacity cleanup is not the source of cash burn in this baseline."
            ),
        },
        {
            "tone": "gold",
            "eyebrow": "Cash routing",
            "title": f"Cleanup consumed {cash['burn_ratio']:.1f}% of generated cash",
            "body": (
                f"${cash['generated']:,.0f} was generated; ${cash['cleanup_consumed']:,.0f} "
                f"went to cleanup and ${cash['accumulation']:,.0f} to accumulation."
            ),
        },
        {
            "tone": "blue",
            "eyebrow": "Capacity pressure",
            "title": f"The book was at 90%+ capacity {capacity_stats['near_cap_pct']:.1f}% of the time",
            "body": (
                f"Average open Bargains: {capacity_stats['average_open']:.1f} of "
                f"{payload['capacity']['limit']}. The cap is a persistent constraint, not an edge case."
            ),
        },
        {
            "tone": "red",
            "eyebrow": "Concentration",
            "title": f"{assets[0]['asset']} is the largest cleanup drag",
            "body": (
                f"Lifecycle PnL ${assets[0]['lifecycle_pnl']:,.0f} across "
                f"{assets[0]['closes']:,} cleanup closes; inventory deficit "
                f"${assets[0]['inventory_deficit']:,.0f}."
            ),
        },
    ]


def build_payload(
    run_dir: Path,
    *,
    cache_dir: Path | None = None,
    include_price_paths: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    required = ("summary.json", "waiter_cleanup.json", "swings.json", "monthly.csv", "inventory.csv")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report artifacts: {', '.join(missing)}")
    summary = _read_json(run_dir / "summary.json")
    cleanup = _read_json(run_dir / "waiter_cleanup.json")
    swings = _read_json(run_dir / "swings.json")
    events = _event_rows(swings)
    price_data = {
        "source": "not included",
        "available_assets": [],
        "missing_assets": sorted({event["asset"] for event in events}),
        "max_candles_per_swing": 0,
    }
    resolved_cache = (cache_dir or REPO_ROOT / "data" / "spot_backtest").expanduser().resolve()
    if include_price_paths and resolved_cache.exists():
        price_data = _attach_price_paths(events, resolved_cache)
    policy = _policy(summary)
    asset_count = len(summary["scenario"]["asset_order"])
    capacity_limit = int(policy["max_open"]) * asset_count
    capacity_points, capacity_summary = _capacity_series(
        run_dir / "inventory.csv",
        capacity=capacity_limit,
    )
    accounting = cleanup["accounting"]
    cash_burn = summary["bargain_cash_flow"]["cleanup_cash_burn"]
    accumulation = summary["treasury_accumulation"]["overview"]
    closed = int(summary["swings"]["total_closed"])
    payload = {
        "meta": {
            "name": summary["scenario"]["name"],
            "start": summary["scenario"]["start"][:10],
            "end": summary["scenario"]["end"][:10],
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "run_dir": str(run_dir),
        },
        "policy": policy,
        "portfolio": summary["portfolio"],
        "cash": {
            "generated": _number(cash_burn["gross_profit_cash_generated_quote"]),
            "cleanup_top_up": _number(cash_burn["gross_cleanup_top_up_quote"]),
            "cleanup_released": _number(cash_burn["cleanup_cash_released_quote"]),
            "cleanup_consumed": _number(cash_burn["net_cleanup_cash_consumed_quote"]),
            "burn_ratio": _number(cash_burn["ratio_pct"]),
            "accumulation": _number(accumulation["spendable_cash_deployed_quote"]),
            "ending_spendable": _number(summary["shared_usdt"]["ending_spendable"]),
        },
        "cleanup": {
            "closes": int(cleanup["cleanup_closes_total"]),
            "share_of_closed_pct": int(cleanup["cleanup_closes_total"]) / closed * 100,
            "lifecycle_pnl": _number(accounting["lifecycle_pnl_quote"]),
            "inventory_deficit": _number(accounting["inventory_deficit_market_value_quote"]),
            "restored_pnl": _number(accounting["restored_inventory_pnl_quote"]),
            "residual_pnl": _number(accounting["inventory_residual_pnl_quote"]),
            "turnover": _number(accounting["cleanup_turnover_quote"]),
            "average_pnl": _number(cleanup["average_cleanup_pnl_quote"]),
            "median_pnl": _number(cleanup["median_cleanup_pnl_quote"]),
        },
        "book": {
            "opened": int(summary["swings"]["total_opened"]),
            "closed": closed,
            "open": int(summary["swings"]["still_open"]),
            "oldest_open_days": _number(summary["swings"]["oldest_open_days"]),
            "unique_blocked": int(
                cleanup["capacity"]["unique_capacity_blocked_opportunities"]
            ),
        },
        "reasons": _reason_rows(cleanup, events),
        "assets": _asset_rows(cleanup),
        "monthly": _monthly_rows(run_dir / "monthly.csv", events),
        "ages": _age_histogram(events),
        "events": events,
        "price_data": price_data,
        "capacity": {
            "limit": capacity_limit,
            "points": capacity_points,
            "summary": capacity_summary,
        },
        "reason_order": list(REASON_ORDER),
        "reason_labels": REASON_LABELS,
        "reason_colors": REASON_COLORS,
    }
    payload["insights"] = _insights(payload)
    return payload


def _json_for_html(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html(payload: dict[str, Any]) -> str:
    title = html.escape(f"Cleanup Observatory · {payload['meta']['name']}")
    data = _json_for_html(payload)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ink:#eef3f8;--muted:#8f9bad;--bg:#070b10;--panel:#101721;--panel2:#0b1118;--line:#263444;--gold:#e7b94b;--green:#53d6a7;--red:#ff667f;--blue:#77aee8;--violet:#a78bd4}}
*{{box-sizing:border-box}}html{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Mono","Courier New",monospace}}body{{margin:0;background:radial-gradient(circle at 78% -12%,#401525 0,transparent 32%),linear-gradient(180deg,#090e14 0,#070b10 80%)}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:linear-gradient(#18222e 1px,transparent 1px),linear-gradient(90deg,#18222e 1px,transparent 1px);background-size:46px 46px;mask-image:linear-gradient(to bottom,#000,transparent 75%)}}
main{{width:min(1540px,calc(100% - 32px));margin:0 auto;padding:28px 0 64px;position:relative}}header{{padding:30px;border:1px solid #553044;border-radius:20px;background:linear-gradient(125deg,rgba(65,16,36,.92),rgba(16,23,33,.96) 58%);box-shadow:0 24px 80px #0008}}
.eyebrow{{color:var(--gold);letter-spacing:.16em;text-transform:uppercase;font-size:11px}}h1{{font-family:Georgia,"Times New Roman",serif;font-weight:500;font-size:clamp(32px,5vw,66px);line-height:.98;margin:14px 0 18px;max-width:950px}}header p{{color:#bbc5d2;max-width:860px;line-height:1.65;margin:0}}.policy{{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}}.chip{{border:1px solid #67552a;background:#1c1a12;color:#f1cc65;border-radius:999px;padding:7px 11px;font-size:11px}}
.kpis{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:14px 0}}.kpi,.panel,.insight{{background:rgba(16,23,33,.96);border:1px solid var(--line);border-radius:16px}}.kpi{{padding:16px;min-height:112px}}.kpi .label{{color:var(--muted);font-size:11px;line-height:1.35}}.kpi .value{{font-size:clamp(19px,2vw,29px);margin-top:12px;letter-spacing:-.04em}}.kpi .sub{{font-size:10px;color:var(--muted);margin-top:7px}}.negative{{color:var(--red)}}.positive{{color:var(--green)}}.gold{{color:var(--gold)}}
.section-title{{display:flex;justify-content:space-between;align-items:end;margin:30px 2px 12px}}.section-title h2{{font-family:Georgia,"Times New Roman",serif;font-weight:500;font-size:27px;margin:0}}.section-title span{{color:var(--muted);font-size:11px}}
.insights{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}}.insight{{padding:16px;min-height:170px;position:relative;overflow:hidden}}.insight:before{{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--tone)}}.insight h3{{font-size:15px;line-height:1.35;margin:10px 0}}.insight p{{color:var(--muted);font-size:11px;line-height:1.55;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:12px}}.panel{{padding:18px;overflow:hidden}}.span-7{{grid-column:span 7}}.span-5{{grid-column:span 5}}.span-6{{grid-column:span 6}}.span-12{{grid-column:span 12}}.panel h3{{font-size:13px;margin:0 0 5px}}.panel .lede{{font-size:10px;color:var(--muted);margin:0 0 16px;line-height:1.45}}
canvas{{display:block;width:100%;height:300px}}.chart-wrap{{position:relative}}.tooltip{{position:absolute;display:none;pointer-events:none;background:#06090dcc;border:1px solid #435468;border-radius:8px;padding:8px 10px;font-size:10px;line-height:1.5;box-shadow:0 8px 30px #000;z-index:2}}
.route{{height:38px;display:flex;border-radius:9px;overflow:hidden;background:#19222d;margin:18px 0 12px}}.route>div{{display:flex;align-items:center;justify-content:center;min-width:2px;font-size:10px;color:#07110d;font-weight:700}}.route .cleanup-route{{background:var(--red)}}.route .acc-route{{background:var(--green)}}.route .free-route{{background:var(--blue)}}.legend{{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:10px}}.dot{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}}
.reason-list,.asset-list{{display:grid;gap:9px}}.reason-row,.asset-row{{display:grid;grid-template-columns:110px 1fr 92px 92px;gap:10px;align-items:center;font-size:10px}}.reason-row .bar-track,.asset-row .bar-track{{height:8px;background:#1a2430;border-radius:99px;overflow:hidden}}.bar-fill{{height:100%;border-radius:99px}}.row-main strong{{display:block;font-size:11px}}.row-main span{{color:var(--muted)}}.right{{text-align:right}}
.table-tools{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}}select{{background:#0a1017;border:1px solid #314154;color:var(--ink);border-radius:9px;padding:8px 10px;font:11px inherit}}.table-wrap{{overflow:auto;max-height:720px;border:1px solid #202c39;border-radius:12px}}table{{width:100%;border-collapse:collapse;min-width:940px;font-size:10px}}th,td{{padding:9px 10px;border-bottom:1px solid #1f2a36;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#111923;color:var(--muted);z-index:3}}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){{text-align:left}}.event-summary{{cursor:pointer}}.event-summary:hover,.event-summary[aria-expanded="true"]{{background:#17212d}}.expand{{display:inline-grid;place-items:center;width:22px;height:22px;border:1px solid #34465a;border-radius:6px;color:var(--gold);transition:transform .15s}}.event-summary[aria-expanded="true"] .expand{{transform:rotate(90deg)}}.lifecycle-row[hidden]{{display:none}}.lifecycle-row td{{padding:0;white-space:normal;text-align:left;background:#090e14}}.life{{padding:16px 18px 20px;border-left:2px solid var(--gold)}}.life-summary{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-bottom:16px}}.life-summary div{{background:#101721;border:1px solid #202d3b;border-radius:10px;padding:10px}}.life-summary span{{display:block;color:var(--muted);font-size:9px}}.life-summary strong{{display:block;margin-top:6px;font-size:12px}}.life-price-wrap{{position:relative;border:1px solid #202d3b;border-radius:12px;background:#070c12;margin:0 0 18px;padding:10px}}.life-price{{height:230px}}.price-caption{{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;color:var(--muted);font-size:9px;margin:2px 3px 8px}}.execution-line{{display:grid;grid-template-columns:94px 66px 1fr;gap:12px;position:relative;padding:0 0 16px 0}}.execution-line:before{{content:"";position:absolute;left:110px;top:15px;bottom:-4px;width:1px;background:#314154}}.execution-line:last-child:before{{display:none}}.execution-line time{{color:var(--muted);font-size:9px}}.execution-badge{{position:relative;z-index:1;justify-self:start;border:1px solid #3a4d61;background:#111923;border-radius:999px;padding:5px 8px;text-transform:uppercase;font-size:9px}}.execution-copy strong{{font-size:11px}}.execution-copy span{{display:block;color:var(--muted);margin-top:4px;font-size:9px}}.count{{color:var(--muted);font-size:10px;margin-left:auto;align-self:center}}
.accounting{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}.accounting div{{border-left:1px solid var(--line);padding-left:12px}}.accounting strong{{display:block;font-size:16px;margin:5px 0}}.accounting span{{color:var(--muted);font-size:10px;line-height:1.5}}
details{{margin-top:14px;border-top:1px solid var(--line);padding-top:14px}}summary{{cursor:pointer;color:var(--gold);font-size:11px}}details p{{color:var(--muted);font-size:10px;line-height:1.65;max-width:1100px}}
footer{{color:var(--muted);font-size:10px;margin-top:22px;text-align:center}}
@media(max-width:1100px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.insights{{grid-template-columns:repeat(2,1fr)}}.span-7,.span-5,.span-6{{grid-column:span 12}}}}
@media(max-width:650px){{main{{width:min(100% - 18px,1540px);padding-top:10px}}header{{padding:21px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.insights{{grid-template-columns:1fr}}.accounting{{grid-template-columns:repeat(2,1fr)}}.reason-row,.asset-row{{grid-template-columns:84px 1fr 74px}}.reason-row>:last-child,.asset-row>:last-child{{display:none}}canvas{{height:260px}}}}
</style></head><body><main>
<header><div class="eyebrow">Scrooge · Cleanup Observatory · Four-Year Baseline</div><h1>What cleanup costs,<br>and what it buys.</h1><p id="header-copy"></p><div class="policy" id="policy"></div></header>
<section class="kpis" id="kpis"></section>
<div class="section-title"><h2>Executive read</h2><span>Mechanism-level conclusions, not just terminal PnL</span></div><section class="insights" id="insights"></section>
<div class="section-title"><h2>Cash and pressure</h2><span>Where profit cash went, and how often capacity constrained the book</span></div>
<section class="grid">
 <article class="panel span-5"><h3>Generated cash routing</h3><p class="lede">Committed cash recycling is excluded. This is spendable cash generated by profitable Bargains.</p><div class="route" id="route"></div><div class="legend" id="route-legend"></div><div class="accounting" id="accounting"></div></article>
 <article class="panel span-7"><h3>Capacity pressure</h3><p class="lede">Daily closing snapshot of open Bargains against the configured portfolio cap.</p><div class="chart-wrap"><canvas id="capacity-chart"></canvas><div class="tooltip"></div></div></article>
 <article class="panel span-7"><h3>Cleanup through time</h3><p class="lede">Stacked monthly closes by reason; line shows monthly lifecycle PnL.</p><div class="chart-wrap"><canvas id="monthly-chart"></canvas><div class="tooltip"></div></div></article>
 <article class="panel span-5"><h3>Age at cleanup</h3><p class="lede">Deep-loss acts early; aging rules only become eligible after their configured thresholds.</p><div class="chart-wrap"><canvas id="age-chart"></canvas><div class="tooltip"></div></div></article>
</section>
<div class="section-title"><h2>Mechanism anatomy</h2><span>Lifecycle PnL includes restored inventory and terminal inventory deficit</span></div>
<section class="grid">
 <article class="panel span-6"><h3>By cleanup reason</h3><p class="lede">Bar length is absolute lifecycle PnL. Green means the mechanism released value rather than consuming it.</p><div class="reason-list" id="reason-list"></div></article>
 <article class="panel span-6"><h3>By asset</h3><p class="lede">Compare cleanup loss, spendable cash change, inventory deficit and blocked opportunities.</p><div class="asset-list" id="asset-list"></div></article>
 <article class="panel span-12"><h3>Cleanup event explorer</h3><p class="lede">Inspect all closes. Select any row to unfold the full Swing lifecycle: opening, partial fills, cleanup decision and final economics.</p><div class="table-tools"><select id="reason-filter"><option value="">All reasons</option></select><select id="asset-filter"><option value="">All assets</option></select><select id="event-sort"><option value="pnl">Worst PnL first</option><option value="age">Oldest first</option><option value="date">Latest first</option></select><span class="count" id="event-count"></span></div><div class="table-wrap"><table><thead><tr><th></th><th>Date</th><th>Asset</th><th>Reason</th><th>Age</th><th>PnL before</th><th>Reverse</th><th>Lifecycle PnL</th><th>Objective</th></tr></thead><tbody id="events"></tbody></table></div></article>
 <article class="panel span-12"><h3>Accounting lens</h3><div class="accounting" id="definitions"></div><details><summary>How to read this report</summary><p>Cleanup lifecycle PnL measures the economic result of Bargains closed by cleanup. Net cleanup cash consumed is spendable cash added on top of each Bargain's own committed proceeds, net of cash released. Inventory deficit values asset quantity not restored by cleanup at the terminal market price. Capacity cleanup marks the highest-PnL eligible Bargain when a new opening is blocked, then closes it on the configured reverse BUY level. The close may realize a loss or an inventory deficit.</p></details></article>
</section><footer id="footer"></footer>
<script id="payload" type="application/json">{data}</script>
<script>
const d=JSON.parse(document.getElementById('payload').textContent),$=s=>document.querySelector(s);
const money=v=>`${{v<0?'-':''}}$${{Math.abs(v).toLocaleString(undefined,{{maximumFractionDigits:0}})}}`,num=v=>Number(v).toLocaleString(undefined,{{maximumFractionDigits:1}}),pct=v=>`${{Number(v).toFixed(1)}}%`;
const tone=v=>v>0?'positive':v<0?'negative':'';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
$('#header-copy').textContent=`${{d.meta.start}} → ${{d.meta.end}} · ${{d.events.length.toLocaleString()}} cleanup closes across ${{d.assets.length}} assets. This page separates useful capacity release from cash-consuming inventory repair.`;
const p=d.policy;$('#policy').innerHTML=[`Cap ${{p.max_open}} / asset`,`Capacity ≥ ${{p.capacity_age}}d · BUY L${{p.capacity_reverse}}`,`Deep loss ≥ ${{p.deep_age}}d · ${{p.deep_pnl}}% · L${{p.deep_reverse}}`,...p.aging.map(x=>`Aging L${{x.level}} ≥ ${{x.age}}d`)].map(x=>`<span class="chip">${{x}}</span>`).join('');
const kpis=[['Cleanup closes',num(d.cleanup.closes),`${{pct(d.cleanup.share_of_closed_pct)}} of all closes`],['Lifecycle PnL',money(d.cleanup.lifecycle_pnl),`Average ${{money(d.cleanup.average_pnl)}} / close`],['Cash burn',pct(d.cash.burn_ratio),`${{money(d.cash.cleanup_consumed)}} net consumed`],['Inventory deficit',money(d.cleanup.inventory_deficit),`${{money(d.cleanup.residual_pnl)}} residual PnL`],['Blocked opportunities',num(d.book.unique_blocked),`${{num(d.capacity.summary.average_open)}} / ${{d.capacity.limit}} average open`],['Edge vs HODL',`${{d.portfolio.edge_vs_hodl_pct_points.toFixed(2)}} pp`,`${{money(d.portfolio.difference_vs_hodl)}} terminal edge`]];
$('#kpis').innerHTML=kpis.map((x,i)=>`<div class="kpi"><div class="label">${{x[0]}}</div><div class="value ${{i===1||i===3?'negative':i===5?'positive':i===2?'gold':''}}">${{x[1]}}</div><div class="sub">${{x[2]}}</div></div>`).join('');
$('#insights').innerHTML=d.insights.map(x=>`<article class="insight" style="--tone:var(--${{x.tone}})"><div class="eyebrow">${{x.eyebrow}}</div><h3>${{x.title}}</h3><p>${{x.body}}</p></article>`).join('');
const total=d.cash.generated||1,accPct=d.cash.accumulation/total*100,cleanPct=d.cash.cleanup_consumed/total*100,freePct=Math.max(0,100-accPct-cleanPct);
$('#route').innerHTML=`<div class="cleanup-route" style="width:${{cleanPct}}%">${{cleanPct.toFixed(1)}}%</div><div class="acc-route" style="width:${{accPct}}%">${{accPct.toFixed(1)}}%</div><div class="free-route" style="width:${{freePct}}%"></div>`;
$('#route-legend').innerHTML=`<span><i class="dot" style="background:var(--red)"></i>Cleanup ${{money(d.cash.cleanup_consumed)}}</span><span><i class="dot" style="background:var(--green)"></i>Accumulation ${{money(d.cash.accumulation)}}</span><span><i class="dot" style="background:var(--blue)"></i>Ending spendable ${{money(d.cash.ending_spendable)}}</span>`;
$('#accounting').innerHTML=[['Generated',money(d.cash.generated),'Profit-target cash'],['Top-ups',money(d.cash.cleanup_top_up),'Gross spendable used'],['Released',money(d.cash.cleanup_released),'Cash returned by cleanup'],['Turnover',money(d.cleanup.turnover),'Cleanup buyback turnover']].map(x=>`<div><span>${{x[0]}}</span><strong>${{x[1]}}</strong><span>${{x[2]}}</span></div>`).join('');
const maxReason=Math.max(...d.reasons.map(x=>Math.abs(x.lifecycle_pnl)),1);
$('#reason-list').innerHTML=d.reasons.map(x=>`<div class="reason-row"><div class="row-main"><strong>${{x.label}}</strong><span>${{x.closes}} closes · ${{x.average_age_days===null?'—':x.average_age_days.toFixed(1)+'d'}}</span></div><div class="bar-track"><div class="bar-fill" style="width:${{Math.abs(x.lifecycle_pnl)/maxReason*100}}%;background:${{x.color}}"></div></div><div class="right ${{tone(x.lifecycle_pnl)}}">${{money(x.lifecycle_pnl)}}</div><div class="right">${{money(x.cash_change)}} cash</div></div>`).join('');
const maxAsset=Math.max(...d.assets.map(x=>Math.abs(x.lifecycle_pnl)),1);
$('#asset-list').innerHTML=d.assets.slice().sort((a,b)=>a.lifecycle_pnl-b.lifecycle_pnl).map(x=>`<div class="asset-row"><div class="row-main"><strong>${{x.asset}}</strong><span>${{x.closes}} closes · ${{x.blocked}} blocked</span></div><div class="bar-track"><div class="bar-fill" style="width:${{Math.abs(x.lifecycle_pnl)/maxAsset*100}}%;background:var(--red)"></div></div><div class="right negative">${{money(x.lifecycle_pnl)}}</div><div class="right">${{money(x.inventory_deficit)}} deficit</div></div>`).join('');
$('#definitions').innerHTML=[['Lifecycle PnL',money(d.cleanup.lifecycle_pnl),'Full economic outcome of cleanup-closed Bargains.'],['Restored inventory',money(d.cleanup.restored_pnl),'PnL on quantity successfully repurchased.'],['Residual inventory',money(d.cleanup.residual_pnl),'PnL of quantity that remained unrestored.'],['Final book',`${{d.book.open}} / ${{d.capacity.limit}} open`,`${{d.book.unique_blocked.toLocaleString()}} unique blocked opportunities.`]].map(x=>`<div><span>${{x[0]}}</span><strong class="${{x[1].startsWith('-')?'negative':''}}">${{x[1]}}</strong><span>${{x[2]}}</span></div>`).join('');

function setupCanvas(id){{const c=document.getElementById(id),r=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const x=c.getContext('2d');x.scale(r,r);return{{c,x,w,h,m:{{l:48,r:42,t:16,b:32}}}}}}
function axes(o,yMin,yMax,leftLabel,rightLabel=''){{const{{x,w,h,m}}=o;x.strokeStyle='#263444';x.fillStyle='#7f8da0';x.font='10px monospace';x.lineWidth=1;for(let i=0;i<5;i++){{const y=m.t+(h-m.t-m.b)*i/4,v=yMax-(yMax-yMin)*i/4;x.beginPath();x.moveTo(m.l,y);x.lineTo(w-m.r,y);x.stroke();x.fillText(num(v),4,y+3)}}x.fillText(leftLabel,m.l,10);if(rightLabel)x.fillText(rightLabel,w-m.r-70,10)}}
function monthlyChart(){{const o=setupCanvas('monthly-chart'),a=d.monthly,{{x,w,h,m}}=o,maxC=Math.max(...a.map(v=>v.cleanup_count),1),pnls=a.map(v=>v.cleanup_pnl),minP=Math.min(...pnls,0),maxP=Math.max(...pnls,0),cw=(w-m.l-m.r)/a.length;axes(o,0,maxC,'closes / month','monthly PnL');a.forEach((v,i)=>{{let y=h-m.b;d.reason_order.forEach(reason=>{{const n=v.reasons[reason],bh=n/maxC*(h-m.t-m.b);x.fillStyle=d.reason_colors[reason];x.fillRect(m.l+i*cw+1,y-bh,Math.max(1,cw-2),bh);y-=bh}});if(i%12===0){{x.fillStyle='#7f8da0';x.fillText(v.month.slice(0,4),m.l+i*cw,h-10)}}}});const sy=v=>m.t+(maxP-v)/(maxP-minP||1)*(h-m.t-m.b);x.strokeStyle='#eef3f8';x.lineWidth=1.5;x.beginPath();a.forEach((v,i)=>{{const px=m.l+(i+.5)*cw,py=sy(v.cleanup_pnl);i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke()}}
function capacityChart(){{const o=setupCanvas('capacity-chart'),a=d.capacity.points,{{x,w,h,m}}=o,limit=d.capacity.limit,sx=i=>m.l+i/(a.length-1||1)*(w-m.l-m.r),sy=v=>h-m.b-v/limit*(h-m.t-m.b);axes(o,0,limit,'open Bargains');x.setLineDash([4,4]);x.strokeStyle='#e7b94b';x.beginPath();x.moveTo(m.l,sy(limit*.9));x.lineTo(w-m.r,sy(limit*.9));x.stroke();x.setLineDash([]);x.strokeStyle='#77aee8';x.lineWidth=1.5;x.beginPath();a.forEach((v,i)=>i?x.lineTo(sx(i),sy(v.open)):x.moveTo(sx(i),sy(v.open)));x.stroke();for(let i=0;i<a.length;i+=365){{x.fillStyle='#7f8da0';x.fillText(a[i].date.slice(0,4),sx(i),h-10)}}}}
function ageChart(){{const o=setupCanvas('age-chart'),a=d.ages,{{x,w,h,m}}=o,max=Math.max(...a.map(v=>Object.values(v.counts).reduce((s,n)=>s+n,0)),1),cw=(w-m.l-m.r)/a.length;axes(o,0,max,'cleanup closes');a.forEach((v,i)=>{{let y=h-m.b;d.reason_order.forEach(reason=>{{const bh=v.counts[reason]/max*(h-m.t-m.b);x.fillStyle=d.reason_colors[reason];x.fillRect(m.l+i*cw+5,y-bh,Math.max(2,cw-10),bh);y-=bh}});x.save();x.translate(m.l+(i+.5)*cw,h-6);x.rotate(-.35);x.fillStyle='#7f8da0';x.textAlign='right';x.fillText(v.label,0,0);x.restore()}})}}
function populateFilters(){{$('#reason-filter').innerHTML+=[...new Set(d.events.map(x=>x.reason))].map(v=>`<option value="${{v}}">${{d.reason_labels[v]}}</option>`).join('');$('#asset-filter').innerHTML+=[...new Set(d.events.map(x=>x.asset))].sort().map(v=>`<option>${{v}}</option>`).join('')}}
function lifeHtml(x){{const e=x.economics,summary=[['Swing',x.swing_id],['Opening',money(e.opening_quote)],['Closing',money(e.closing_quote)],['Gross PnL',money(e.gross_pnl)],['Fees',money(e.fees)],['Net PnL',money(x.pnl_quote)]];const executions=x.executions.map((z,i)=>{{const descriptor=z.close_reason?d.reason_labels[z.close_reason]||z.close_reason:z.signal?`${{z.signal}} L${{z.signal_level??'—'}}`:z.phase;return `<div class="execution-line"><time>${{esc(z.at.slice(0,16).replace('T',' '))}}</time><span class="execution-badge">${{esc(z.phase)}} · ${{esc(z.side)}}</span><div class="execution-copy"><strong>${{num(z.quantity)}} ${{esc(x.asset)}} @ $${{num(z.price)}}</strong><span>${{money(z.quote)}} notional · fee ${{num(z.fee)}} ${{esc(z.fee_asset)}} · ${{esc(descriptor)}}</span></div></div>`}}).join('');const chart=x.price_path?.length?`<div class="life-price-wrap"><div class="price-caption"><span>Actual Binance Spot OHLC · ${{x.price_interval_hours}}h candles</span><span>${{x.price_path.length}} points · markers are executions</span></div><canvas class="life-price"></canvas></div>`:`<div class="life-price-wrap"><div class="price-caption"><span>Price history unavailable; execution prices remain exact.</span></div></div>`;return `<div class="life"><div class="life-summary">${{summary.map(v=>`<div><span>${{esc(v[0])}}</span><strong class="${{v[0].includes('PnL')?tone(x.pnl_quote):''}}">${{esc(v[1])}}</strong></div>`).join('')}}</div>${{chart}}<div class="execution-timeline">${{executions}}</div><div class="legend"><span>Origin: <b>${{esc(x.origin_side)}}</b></span><span>Weighted open: <b>$${{num(e.weighted_opening_price)}}</b></span><span>Weighted close: <b>$${{num(e.weighted_closing_price)}}</b></span><span>Asset change: <b>${{num(e.asset_change)}}</b></span><span>Unrecovered: <b>${{num(e.unrecovered_quantity)}}</b></span></div></div>`}}
function drawLifeChart(canvas,event){{const a=event.price_path;if(!canvas||!a?.length)return;const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const x=canvas.getContext('2d');x.scale(ratio,ratio);const m={{l:56,r:16,t:15,b:28}},low=Math.min(...a.map(v=>v.l)),high=Math.max(...a.map(v=>v.h)),pad=(high-low||high*.01||1)*.08,yMin=low-pad,yMax=high+pad,sy=v=>m.t+(yMax-v)/(yMax-yMin)*(h-m.t-m.b),cw=(w-m.l-m.r)/a.length;x.strokeStyle='#1d2936';x.fillStyle='#7f8da0';x.font='9px monospace';for(let i=0;i<4;i++){{const py=m.t+(h-m.t-m.b)*i/3,v=yMax-(yMax-yMin)*i/3;x.beginPath();x.moveTo(m.l,py);x.lineTo(w-m.r,py);x.stroke();x.fillText(v.toPrecision(5),2,py+3)}}a.forEach((v,i)=>{{const px=m.l+(i+.5)*cw,up=v.c>=v.o,color=up?'#53d6a7':'#ff667f';x.strokeStyle=color;x.fillStyle=color;x.beginPath();x.moveTo(px,sy(v.h));x.lineTo(px,sy(v.l));x.stroke();const top=sy(Math.max(v.o,v.c)),bottom=sy(Math.min(v.o,v.c));x.fillRect(px-Math.max(1,cw*.32),top,Math.max(1.5,cw*.64),Math.max(1,bottom-top))}});const t0=a[0].t,t1=a[a.length-1].t,tx=t=>m.l+(t-t0)/(t1-t0||1)*(w-m.l-m.r);event.executions.forEach((z,i)=>{{if(!z.at_ms||!z.price)return;const px=Math.max(m.l,Math.min(w-m.r,tx(z.at_ms))),py=sy(z.price);x.fillStyle=i===0?'#e7b94b':i===event.executions.length-1?'#eef3f8':'#77aee8';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();x.strokeStyle='#070b10';x.stroke()}});x.fillStyle='#7f8da0';x.fillText(new Date(t0).toISOString().slice(0,10),m.l,h-8);const end=new Date(t1).toISOString().slice(0,10);x.fillText(end,w-m.r-x.measureText(end).width,h-8)}}
function renderEvents(){{const rf=$('#reason-filter').value,af=$('#asset-filter').value,sort=$('#event-sort').value;let rows=d.events.filter(x=>(!rf||x.reason===rf)&&(!af||x.asset===af));rows.sort(sort==='age'?(a,b)=>b.age_days-a.age_days:sort==='date'?(a,b)=>b.date.localeCompare(a.date):(a,b)=>a.pnl_quote-b.pnl_quote);const visible=rows.slice(0,200);$('#event-count').textContent=`${{rows.length.toLocaleString()}} events · showing first ${{visible.length}}`;$('#events').innerHTML=visible.map((x,i)=>`<tr class="event-summary" tabindex="0" aria-expanded="false" data-life="life-${{i}}" data-index="${{i}}"><td><span class="expand">›</span></td><td>${{esc(x.date)}}</td><td>${{esc(x.asset)}}</td><td><i class="dot" style="background:${{d.reason_colors[x.reason]}}"></i>${{esc(x.reason_label)}}</td><td>${{x.age_days.toFixed(1)}}d</td><td class="${{tone(x.pnl_pct_before||0)}}">${{x.pnl_pct_before===null?'—':Number(x.pnl_pct_before).toFixed(1)+'%'}}</td><td>L${{x.actual_level??'—'}} / L${{x.required_level??'—'}}</td><td class="${{tone(x.pnl_quote)}}">${{money(x.pnl_quote)}}</td><td>${{esc(x.objective.replace('accumulate_',''))}}</td></tr><tr class="lifecycle-row" id="life-${{i}}" hidden><td colspan="9">${{lifeHtml(x)}}</td></tr>`).join('');document.querySelectorAll('.event-summary').forEach(row=>{{const toggle=()=>{{const detail=document.getElementById(row.dataset.life),open=row.getAttribute('aria-expanded')==='true';row.setAttribute('aria-expanded',String(!open));detail.hidden=open;if(!open)requestAnimationFrame(()=>drawLifeChart(detail.querySelector('.life-price'),visible[Number(row.dataset.index)]))}};row.addEventListener('click',toggle);row.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();toggle()}}}})}})}}
populateFilters();renderEvents();['reason-filter','asset-filter','event-sort'].forEach(id=>document.getElementById(id).addEventListener('change',renderEvents));
function draw(){{monthlyChart();capacityChart();ageChart()}}draw();new ResizeObserver(()=>requestAnimationFrame(draw)).observe(document.querySelector('.grid'));
$('#footer').textContent=`Generated ${{d.meta.generated_at.slice(0,19).replace('T',' ')}} UTC · Source: ${{d.meta.run_dir}}`;
</script></main></body></html>"""


def build(
    run_dir: Path,
    output: Path | None = None,
    *,
    cache_dir: Path | None = None,
    include_price_paths: bool = True,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    target = (output or run_dir / "cleanup_report.html").expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_html(
            build_payload(
                run_dir,
                cache_dir=cache_dir,
                include_price_paths=include_price_paths,
            )
        ),
        encoding="utf-8",
    )
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed Spot backtest run directory.")
    parser.add_argument("--output", type=Path, help="Output HTML path.")
    parser.add_argument("--cache-dir", type=Path, help="Spot market-data cache directory.")
    parser.add_argument(
        "--no-price-paths",
        action="store_true",
        help="Skip embedding actual OHLC paths for each cleanup Swing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        build(
            args.run_dir,
            args.output,
            cache_dir=args.cache_dir,
            include_price_paths=not args.no_price_paths,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
