from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean, median
import subprocess
from typing import Any

from backtest.spot_engine import SpotBacktestResult
from backtest.spot_report_html import write_spot_backtest_html
from backtest.spot_scenario import scenario_as_dict, write_scenario_snapshot


AGE_BUCKETS = (
    ("under_7_days", 0, 7),
    ("7_to_30_days", 7, 30),
    ("30_to_90_days", 30, 90),
    ("90_to_180_days", 90, 180),
    ("180_plus_days", 180, None),
)


def _maximum_drawdown(values: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, ((value / peak) - 1.0) * 100.0)
    return worst


def _maximum_concurrent(swings: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for swing in swings:
        events.append((int(swing["opened_at_ms"]), 1))
        if swing.get("closed_at_ms") is not None:
            events.append((int(swing["closed_at_ms"]), -1))
    current = 0
    maximum = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        current += change
        maximum = max(maximum, current)
    return maximum


def _swing_metrics(swings: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [item for item in swings if item["status"] == "closed"]
    opened = [item for item in swings if item["status"] != "closed"]
    profitable = [item for item in closed if float(item["economics"].get("realized_pnl_quote") or 0) > 0]
    losing = [item for item in closed if float(item["economics"].get("realized_pnl_quote") or 0) < 0]
    durations = [float(item["age_seconds"]) for item in closed]
    open_ages = [float(item["age_seconds"]) / 86400.0 for item in opened]
    age_buckets = {
        name: sum(1 for age in open_ages if age >= lower and (upper is None or age < upper))
        for name, lower, upper in AGE_BUCKETS
    }
    fees: dict[str, float] = {}
    for swing in swings:
        for asset, amount in swing["economics"].get("fees_by_asset", {}).items():
            fees[asset] = fees.get(asset, 0.0) + float(amount)
    return {
        "total_opened": len(swings),
        "total_closed": len(closed),
        "still_open": len(opened),
        "profitable_closed": len(profitable),
        "losing_closed": len(losing),
        "win_rate_pct": (len(profitable) / len(closed)) * 100 if closed else None,
        "realized_pnl_quote": sum(float(item["economics"].get("realized_pnl_quote") or 0) for item in closed),
        "unrealized_open_pnl_quote": sum(
            float(item["economics"].get("unrealized_pnl_quote") or 0) for item in opened
        ),
        "fees_by_asset": fees,
        "median_duration_hours": median(durations) / 3600.0 if durations else None,
        "average_duration_hours": mean(durations) / 3600.0 if durations else None,
        "oldest_open_days": max(open_ages) if open_ages else None,
        "maximum_concurrent_open": _maximum_concurrent(swings),
        "open_age_buckets": age_buckets,
    }


def _level_metrics(result: SpotBacktestResult, symbol: str) -> dict[str, Any]:
    signals = [item for item in result.signals if item["asset_symbol"] == symbol]
    actions = [
        item
        for item in result.actions
        if item["asset_symbol"] == symbol and item["action_type"] == "open"
    ]
    return {
        f"level_{level}": {
            "signals": sum(1 for item in signals if int(item["level"]) == level),
            "swing_opens": sum(1 for item in actions if int(item.get("signal_level") or 0) == level),
        }
        for level in range(1, len(result.scenario.signal.levels_pct) + 1)
    }


def _modifier_metrics(result: SpotBacktestResult) -> dict[str, Any]:
    output: dict[str, dict[str, float | int]] = {}
    swing_map = {item["swing_id"]: item for item in result.swings}
    for signal in result.signals:
        if signal["opportunity"] == "hold":
            continue
        key = f"{float(signal['sizing_modifier']):g}x"
        output.setdefault(
            key,
            {
                "applications": 0,
                "requested_trade_value": 0.0,
                "executed_trade_value": 0.0,
                "closed_swing_result_quote": 0.0,
            },
        )["applications"] += 1
    for action in result.actions:
        modifier = action.get("sizing_modifier")
        if modifier is None:
            continue
        key = f"{float(modifier):g}x"
        bucket = output.setdefault(
            key,
            {
                "applications": 0,
                "requested_trade_value": 0.0,
                "executed_trade_value": 0.0,
                "closed_swing_result_quote": 0.0,
            },
        )
        bucket["requested_trade_value"] += float(action.get("requested_value") or 0)
        bucket["executed_trade_value"] += float(action.get("executed_value") or 0)
        if action["action_type"] == "open":
            swing = swing_map.get(action["swing_id"])
            if swing is not None and swing["status"] == "closed":
                bucket["closed_swing_result_quote"] += float(
                    swing["economics"].get("realized_pnl_quote") or 0
                )
    return output


def _inventory_metrics(result: SpotBacktestResult, symbol: str) -> dict[str, Any]:
    rows = [item for item in result.inventory_history if item["asset_symbol"] == symbol]
    utilizations = [float(item["tradable_inventory_utilization_pct"]) for item in rows]
    distances = [float(item["distance_above_floor_pct"]) for item in rows]
    return {
        "average_tradable_inventory_utilization_pct": mean(utilizations) if utilizations else 0.0,
        "maximum_tradable_inventory_utilization_pct": max(utilizations, default=0.0),
        "minimum_distance_above_floor_pct": min(distances, default=0.0),
        "near_floor_threshold_pct": result.scenario.near_floor_pct,
        "time_near_floor_pct": (
            sum(1 for value in distances if value <= result.scenario.near_floor_pct) / len(distances) * 100
            if distances
            else 0.0
        ),
        "maximum_simultaneously_underwater_sell_origin_swings": max(
            (int(item["underwater_sell_origin_swings"]) for item in rows),
            default=0,
        ),
        "maximum_open_buy_capital_tied_quote": max(
            (float(item["open_buy_capital_tied_quote"]) for item in rows),
            default=0.0,
        ),
    }


def _maximum_portfolio_inventory_metric(result: SpotBacktestResult, field: str) -> float:
    totals: dict[int, float] = {}
    for row in result.inventory_history:
        timestamp_ms = int(row["timestamp_ms"])
        totals[timestamp_ms] = totals.get(timestamp_ms, 0.0) + float(row[field])
    return max(totals.values(), default=0.0)


def _bad_case_metrics(swings: list[dict[str, Any]]) -> dict[str, Any]:
    open_swings = [item for item in swings if item["status"] != "closed"]
    underwater_sells = [
        item
        for item in open_swings
        if item["origin_side"] == "sell"
        and float(item["economics"].get("unrealized_pnl_quote") or 0) < 0
    ]
    open_buys = [item for item in open_swings if item["origin_side"] == "buy"]
    return {
        "underwater_sell_origin_count": len(underwater_sells),
        "quantity_sold_not_restored": sum(
            float(item["economics"].get("remaining_quantity") or 0) for item in underwater_sells
        ),
        "value_required_to_restore": sum(
            float(item["economics"].get("remaining_quantity") or 0)
            * float(item.get("current_market_price") or 0)
            for item in underwater_sells
        ),
        "oldest_underwater_sell_days": max(
            (float(item["age_seconds"]) / 86400.0 for item in underwater_sells),
            default=None,
        ),
        "open_buy_capital_tied_quote": sum(
            float(item["economics"].get("remaining_quantity") or 0)
            * float(item["economics"].get("weighted_opening_price") or 0)
            for item in open_buys
        ),
        "open_buy_mark_to_market_pnl": sum(
            float(item["economics"].get("unrealized_pnl_quote") or 0) for item in open_buys
        ),
        "oldest_open_buy_days": max(
            (float(item["age_seconds"]) / 86400.0 for item in open_buys),
            default=None,
        ),
    }


def build_spot_backtest_report(result: SpotBacktestResult) -> dict[str, Any]:
    final_point = result.equity[-1]
    final_value = float(final_point["treasury_value"])
    hodl_final = float(final_point["hodl_value"])
    initial_value = result.starting_value
    cash_values = [float(item["shared_usdt"]) for item in result.equity]
    cash_reserved = [float(item["shared_usdt_reserved"]) for item in result.equity]
    cash_available = [float(item["shared_usdt_available"]) for item in result.equity]
    strong_limit = result.scenario.starting_usdt * (
        1.0 - result.scenario.strong_cash_utilization_pct / 100.0
    )
    per_asset: dict[str, Any] = {}
    for asset_config in result.scenario.assets:
        symbol = asset_config.symbol
        state = result.assets[symbol]
        swings = [item for item in result.swings if item["asset_symbol"] == symbol]
        ratchets = [item for item in result.target_history if item["asset_symbol"] == symbol]
        initial_floor = asset_config.target_holding * asset_config.minimum_holding_pct / 100.0
        final_market_value = state.quantity * result.final_prices[symbol]
        objective_metrics: dict[str, Any] = {}
        if asset_config.trading_objective == "accumulate_asset":
            objective_metrics = {
                "initial_quantity": asset_config.quantity,
                "final_quantity": state.quantity,
                "net_asset_accumulated": state.quantity - asset_config.quantity,
                "initial_target": asset_config.target_holding,
                "final_target": state.target_quantity,
                "total_target_ratchet": state.target_quantity - asset_config.target_holding,
                "successful_ratchets": len(ratchets),
                "initial_protected_floor": initial_floor,
                "final_protected_floor": state.protected_floor,
                "realized_asset_gain": sum(float(item["applied_gain_quantity"]) for item in ratchets),
            }
        elif asset_config.trading_objective == "accumulate_cash":
            objective_metrics = {
                "realized_quote_cash_generated": sum(
                    float(item["economics"].get("realized_cash_gain_quote") or 0)
                    for item in swings
                    if item["status"] == "closed"
                ),
                "completed_cash_cycles": sum(1 for item in swings if item["status"] == "closed"),
                "open_swing_exposure": sum(
                    float(item["economics"].get("unrealized_pnl_quote") or 0)
                    for item in swings
                    if item["status"] != "closed"
                ),
            }
        per_asset[symbol] = {
            "starting": {
                "quantity": asset_config.quantity,
                "entry_cost": asset_config.entry_cost,
                "binance_quantity": asset_config.binance_quantity,
                "cold_storage_quantity": asset_config.cold_storage_quantity,
                "target_holding": asset_config.target_holding,
                "minimum_holding_pct": asset_config.minimum_holding_pct,
                "protected_floor": initial_floor,
                "trading_objective": asset_config.trading_objective,
            },
            "market": {
                "starting_price": result.initial_prices[symbol],
                "ending_price": result.final_prices[symbol],
                "market_return_pct": (
                    (result.final_prices[symbol] / result.initial_prices[symbol]) - 1.0
                ) * 100.0,
            },
            "trading": {
                **_swing_metrics(swings),
                "levels": _level_metrics(result, symbol),
            },
            "inventory": _inventory_metrics(result, symbol),
            "bad_cases": _bad_case_metrics(swings),
            "objective_metrics": objective_metrics,
            "final": {
                "quantity": state.quantity,
                "average_cost": state.average_cost,
                "target_holding": state.target_quantity,
                "protected_floor": state.protected_floor,
                "binance_quantity": state.binance_quantity,
                "cold_storage_quantity": state.cold_storage_quantity,
                "unassigned_quantity": state.unassigned_quantity,
                "market_value": final_market_value,
                "realized_portfolio_pnl_on_known_basis": state.realized_portfolio_pnl,
                "unrealized_portfolio_pnl_on_known_basis": state.unrealized_portfolio_pnl(
                    result.final_prices[symbol]
                ),
            },
            "benchmark": {
                "hodl_value": asset_config.quantity * result.final_prices[symbol],
                "scrooge_asset_value": final_market_value,
            },
        }
    final_allocations = {
        symbol: (
            result.assets[symbol].quantity * result.final_prices[symbol] / final_value * 100
            if final_value > 0
            else 0.0
        )
        for symbol in result.scenario.asset_order
    }
    final_allocations["USDT"] = result.final_usdt / final_value * 100 if final_value > 0 else 0.0
    return {
        "scenario": {
            "name": result.scenario.name,
            "start": result.scenario.start.isoformat(),
            "end": result.scenario.end.isoformat(),
            "interval": result.scenario.interval,
            "asset_order": list(result.scenario.asset_order),
            "data_source": result.data_source,
        },
        "portfolio": {
            "starting_treasury_value": initial_value,
            "final_treasury_value": final_value,
            "hodl_final_treasury_value": hodl_final,
            "difference_vs_hodl": final_value - hodl_final,
            "total_return_pct": ((final_value / initial_value) - 1.0) * 100 if initial_value > 0 else 0.0,
            "hodl_return_pct": ((hodl_final / initial_value) - 1.0) * 100 if initial_value > 0 else 0.0,
            "maximum_treasury_drawdown_pct": _maximum_drawdown(
                [float(item["treasury_value"]) for item in result.equity]
            ),
            "final_allocation_pct": final_allocations,
        },
        "shared_usdt": {
            "starting": result.scenario.starting_usdt,
            "ending": result.final_usdt,
            "minimum": result.minimum_usdt,
            "maximum": result.maximum_usdt,
            "average": mean(cash_values),
            "ending_reserved": cash_reserved[-1],
            "maximum_reserved": max(cash_reserved, default=0.0),
            "average_reserved": mean(cash_reserved),
            "ending_available": cash_available[-1],
            "minimum_available": min(cash_available, default=0.0),
            "average_available": mean(cash_available),
            "strong_utilization_definition": (
                f"Shared USDT at or below {100 - result.scenario.strong_cash_utilization_pct:g}% "
                "of its starting balance."
            ),
            "time_strongly_utilized_pct": (
                sum(1 for value in cash_values if value <= strong_limit) / len(cash_values) * 100
                if cash_values and result.scenario.starting_usdt > 0
                else 0.0
            ),
        },
        "swings": _swing_metrics(result.swings),
        "bad_cases": {
            **_bad_case_metrics(result.swings),
            "maximum_simultaneously_underwater_sell_origin_swings": int(
                _maximum_portfolio_inventory_metric(
                    result,
                    "underwater_sell_origin_swings",
                )
            ),
            "maximum_open_buy_capital_tied_quote": _maximum_portfolio_inventory_metric(
                result,
                "open_buy_capital_tied_quote",
            ),
        },
        "indicator_modifiers": _modifier_metrics(result),
        "per_asset": per_asset,
        "rejected_orders": len(result.rejections),
    }


def build_monthly_results(result: SpotBacktestResult) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.equity:
        month = datetime.fromtimestamp(int(row["timestamp_ms"]) / 1000, tz=UTC).strftime("%Y-%m")
        grouped.setdefault(month, []).append(row)
    output: list[dict[str, Any]] = []
    for month, rows in sorted(grouped.items()):
        start = rows[0]
        end = rows[-1]
        opened = sum(
            1
            for item in result.swings
            if datetime.fromtimestamp(int(item["opened_at_ms"]) / 1000, tz=UTC).strftime("%Y-%m") == month
        )
        closed = sum(
            1
            for item in result.swings
            if item.get("closed_at_ms") is not None
            and datetime.fromtimestamp(int(item["closed_at_ms"]) / 1000, tz=UTC).strftime("%Y-%m") == month
        )
        output.append(
            {
                "month": month,
                "treasury_start_value": start["treasury_value"],
                "treasury_end_value": end["treasury_value"],
                "monthly_return_pct": (
                    (float(end["treasury_value"]) / float(start["treasury_value"])) - 1.0
                ) * 100 if float(start["treasury_value"]) > 0 else 0.0,
                "hodl_monthly_return_pct": (
                    (float(end["hodl_value"]) / float(start["hodl_value"])) - 1.0
                ) * 100 if float(start["hodl_value"]) > 0 else 0.0,
                "difference_vs_hodl": float(end["treasury_value"]) - float(end["hodl_value"]),
                "closed_swings": closed,
                "opened_swings": opened,
                "ending_open_swings": end["open_swings"],
                "ending_usdt": end["shared_usdt"],
            }
        )
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(_json_ready(payload), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_spot_backtest_artifacts(result: SpotBacktestResult, output_dir: str | Path | None = None) -> dict[str, Any]:
    target = Path(output_dir or result.scenario.output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    report = build_spot_backtest_report(result)
    monthly = build_monthly_results(result)
    reproducibility = {
        "scenario": scenario_as_dict(result.scenario),
        "strategy_code_revision": _git_revision(),
        "timing": (
            "Evaluate after candle N closes using data through N; execute one queued market action per asset "
            "at candle N+1 open. Warm-up candles never trade."
        ),
        "rolling_24h_reference": "Candle close exactly 24 hours before the evaluated candle close.",
        "missing_candles": "Fail the run; no interpolation or forward fill.",
        "cross_asset_order": {
            "rule": "asset symbol ascending, matching the live policy query order",
            "symbols": list(result.scenario.asset_order),
        },
        "quantization": "Current cached Binance Spot filters, not historical filter versions.",
        "execution_model": "Deterministic next-candle-open market fill plus configured fee and slippage.",
    }
    _write_json(target / "summary.json", report)
    _write_json(target / "scenario.resolved.json", reproducibility)
    write_scenario_snapshot(result.scenario, target / "scenario.resolved.yaml")
    _write_json(target / "per_asset_summary.json", report["per_asset"])
    _write_json(target / "swings.json", result.swings)
    _write_json(target / "final_state.json", {
        "shared_usdt": result.final_usdt,
        "assets": {
            symbol: {
                "quantity": state.quantity,
                "binance_quantity": state.binance_quantity,
                "cold_storage_quantity": state.cold_storage_quantity,
                "unassigned_quantity": state.unassigned_quantity,
                "average_cost": state.average_cost,
                "target_holding": state.target_quantity,
                "protected_floor": state.protected_floor,
            }
            for symbol, state in result.assets.items()
        },
    })
    _write_csv(target / "equity.csv", result.equity)
    _write_csv(target / "monthly.csv", monthly)
    _write_csv(target / "executions.csv", result.executions)
    _write_csv(target / "signals.csv", result.signals)
    _write_csv(target / "actions.csv", result.actions)
    _write_csv(target / "target_history.csv", result.target_history)
    _write_csv(target / "inventory.csv", result.inventory_history)
    _write_csv(target / "rejections.csv", result.rejections)

    portfolio = report["portfolio"]
    swing_metrics = report["swings"]
    markdown = [
        f"# Spot Research: {result.scenario.name}",
        "",
        "## Portfolio",
        "",
        f"- Period: {result.scenario.start.isoformat()} to {result.scenario.end.isoformat()}",
        f"- Starting Treasury Value: ${portfolio['starting_treasury_value']:,.2f}",
        f"- Final Treasury Value: ${portfolio['final_treasury_value']:,.2f}",
        f"- HODL Final Treasury Value: ${portfolio['hodl_final_treasury_value']:,.2f}",
        f"- Difference vs HODL: ${portfolio['difference_vs_hodl']:,.2f}",
        f"- Total Return: {portfolio['total_return_pct']:.2f}%",
        f"- HODL Return: {portfolio['hodl_return_pct']:.2f}%",
        f"- Maximum Treasury Drawdown: {portfolio['maximum_treasury_drawdown_pct']:.2f}%",
        "",
        "## Bargains",
        "",
        f"- Opened: {swing_metrics['total_opened']}",
        f"- Closed: {swing_metrics['total_closed']}",
        f"- Still open: {swing_metrics['still_open']}",
        f"- Realized quote PnL: ${swing_metrics['realized_pnl_quote']:,.2f}",
        f"- Unrealized open PnL: ${swing_metrics['unrealized_open_pnl_quote']:,.2f}",
        "",
        "This is a strategy backtest, not an order-book or microstructure simulation.",
    ]
    (target / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_spot_backtest_html(
        target / "report.html",
        report,
        result.equity,
        monthly,
        result.swings,
    )
    return {"output_dir": str(target), "report": report, "monthly": monthly}
