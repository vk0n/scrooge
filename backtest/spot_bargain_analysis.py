from __future__ import annotations

from collections.abc import Callable
from statistics import mean, median
from typing import Any


DURATION_BUCKETS = (
    ("under_1_day", "< 1 day", 0.0, 1.0),
    ("1_to_7_days", "1-7 days", 1.0, 7.0),
    ("7_to_30_days", "7-30 days", 7.0, 30.0),
    ("30_to_90_days", "30-90 days", 30.0, 90.0),
    ("90_plus_days", "90+ days", 90.0, None),
)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _is_closed(swing: dict[str, Any]) -> bool:
    return swing.get("status") == "closed"


def _lifecycle_pnl(swing: dict[str, Any]) -> float:
    economics = swing.get("economics") or {}
    realized = _number(economics.get("realized_pnl_quote"))
    if _is_closed(swing):
        return realized
    return realized + _number(economics.get("unrealized_pnl_quote"))


def _return_pct(swing: dict[str, Any]) -> float | None:
    basis = _number((swing.get("economics") or {}).get("opening_quote_quantity"))
    if basis <= 0:
        return None
    return _lifecycle_pnl(swing) / basis * 100.0


def _duration_bucket(swing: dict[str, Any]) -> tuple[str, str]:
    age_days = _number(swing.get("age_seconds")) / 86400.0
    for key, label, lower, upper in DURATION_BUCKETS:
        if age_days >= lower and (upper is None or age_days < upper):
            return key, label
    return DURATION_BUCKETS[-1][0], DURATION_BUCKETS[-1][1]


def _outcome_bucket(swing: dict[str, Any]) -> tuple[str, str]:
    pnl = _lifecycle_pnl(swing)
    if _is_closed(swing):
        if pnl > 0:
            return "closed_profit", "Closed Profit"
        if pnl < 0:
            return "closed_loss", "Closed Loss"
        return "closed_flat", "Closed Flat"
    if pnl > 0:
        return "open_profit", "Open Profit"
    if pnl < 0:
        return "open_underwater", "Open Underwater"
    return "open_flat", "Open Flat"


def _slice_metrics(swings: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [swing for swing in swings if _is_closed(swing)]
    open_swings = [swing for swing in swings if not _is_closed(swing)]
    realized = sum(
        _number((swing.get("economics") or {}).get("realized_pnl_quote"))
        for swing in swings
    )
    unrealized = sum(
        _number((swing.get("economics") or {}).get("unrealized_pnl_quote"))
        for swing in open_swings
    )
    closed_results = [_lifecycle_pnl(swing) for swing in closed]
    returns = [value for swing in swings if (value := _return_pct(swing)) is not None]
    closed_returns = [value for swing in closed if (value := _return_pct(swing)) is not None]
    durations = [_number(swing.get("age_seconds")) / 3600.0 for swing in closed]
    opening_notional = sum(
        _number((swing.get("economics") or {}).get("opening_quote_quantity"))
        for swing in swings
    )
    fees_by_asset: dict[str, float] = {}
    quote_fees = 0.0
    for swing in swings:
        economics = swing.get("economics") or {}
        quote_symbol = str(swing.get("quote_symbol") or "")
        for asset, raw_amount in (economics.get("fees_by_asset") or {}).items():
            amount = _number(raw_amount)
            fees_by_asset[asset] = fees_by_asset.get(asset, 0.0) + amount
            if asset == quote_symbol:
                quote_fees += amount
    gross_closed_wins = sum(value for value in closed_results if value > 0)
    gross_closed_losses = abs(sum(value for value in closed_results if value < 0))
    net_pnl = realized + unrealized
    return {
        "count": len(swings),
        "closed": len(closed),
        "open": len(open_swings),
        "partially_closed": sum(1 for swing in open_swings if swing.get("status") == "partially_closed"),
        "closure_rate_pct": len(closed) / len(swings) * 100.0 if swings else None,
        "closed_win_rate_pct": (
            sum(1 for value in closed_results if value > 0) / len(closed) * 100.0
            if closed
            else None
        ),
        "realized_pnl_quote": realized,
        "unrealized_pnl_quote": unrealized,
        "net_pnl_quote": net_pnl,
        "opening_notional_quote": opening_notional,
        "return_on_notional_pct": net_pnl / opening_notional * 100.0 if opening_notional else None,
        "average_return_pct": mean(returns) if returns else None,
        "median_return_pct": median(returns) if returns else None,
        "average_closed_return_pct": mean(closed_returns) if closed_returns else None,
        "median_closed_return_pct": median(closed_returns) if closed_returns else None,
        "average_closed_pnl_quote": mean(closed_results) if closed_results else None,
        "closed_expectancy_quote": mean(closed_results) if closed_results else None,
        "profit_factor": gross_closed_wins / gross_closed_losses if gross_closed_losses > 0 else None,
        "average_duration_hours": mean(durations) if durations else None,
        "median_duration_hours": median(durations) if durations else None,
        "duration_p25_hours": _percentile(durations, 0.25),
        "duration_p75_hours": _percentile(durations, 0.75),
        "duration_p90_hours": _percentile(durations, 0.90),
        "average_fills": (
            mean(len(swing.get("executions") or []) for swing in swings) if swings else None
        ),
        "quote_fees": quote_fees,
        "fee_drag_pct": quote_fees / opening_notional * 100.0 if opening_notional else None,
        "fees_by_asset": fees_by_asset,
    }


def _grouped_metrics(
    swings: list[dict[str, Any]],
    classifier: Callable[[dict[str, Any]], tuple[str, str]],
    *,
    ordered_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for swing in swings:
        key, label = classifier(swing)
        group = groups.setdefault(key, {"key": key, "label": label, "swings": []})
        group["swings"].append(swing)
    order = {key: index for index, key in enumerate(ordered_keys)}
    rows = [
        {"key": group["key"], "label": group["label"], **_slice_metrics(group["swings"])}
        for group in groups.values()
    ]
    return sorted(rows, key=lambda row: (order.get(row["key"], len(order)), row["label"]))


def _notable_swing(swing: dict[str, Any]) -> dict[str, Any]:
    reason = swing.get("strategy_reason") or {}
    return {
        "swing_id": swing.get("swing_id"),
        "asset_symbol": swing.get("asset_symbol"),
        "status": swing.get("status"),
        "objective": swing.get("trading_objective"),
        "origin_side": swing.get("origin_side"),
        "signal_level": reason.get("signal_level"),
        "signal_tier": (reason.get("indicator_assessment") or {}).get("tier"),
        "age_days": _number(swing.get("age_seconds")) / 86400.0,
        "pnl_quote": _lifecycle_pnl(swing),
        "return_pct": _return_pct(swing),
    }


def build_bargain_analysis(swings: list[dict[str, Any]]) -> dict[str, Any]:
    open_swings = [swing for swing in swings if not _is_closed(swing)]
    closed = [swing for swing in swings if _is_closed(swing)]
    underwater_open = [swing for swing in open_swings if _lifecycle_pnl(swing) < 0]
    profitable_open = [swing for swing in open_swings if _lifecycle_pnl(swing) > 0]
    return {
        "overview": _slice_metrics(swings),
        "breakdowns": {
            "asset": _grouped_metrics(
                swings,
                lambda swing: (
                    str(swing.get("asset_symbol") or "unknown").lower(),
                    str(swing.get("asset_symbol") or "Unknown"),
                ),
            ),
            "objective": _grouped_metrics(
                swings,
                lambda swing: (
                    str(swing.get("trading_objective") or "protected"),
                    str(swing.get("trading_objective") or "protected")
                    .replace("_", " ")
                    .title(),
                ),
                ordered_keys=("accumulate_cash", "accumulate_asset", "protected"),
            ),
            "origin": _grouped_metrics(
                swings,
                lambda swing: (
                    str(swing.get("origin_side") or "unknown"),
                    f"{str(swing.get('origin_side') or 'unknown').title()} Origin",
                ),
                ordered_keys=("sell", "buy", "unknown"),
            ),
            "signal_level": _grouped_metrics(
                swings,
                lambda swing: (
                    f"level_{(swing.get('strategy_reason') or {}).get('signal_level') or 'unknown'}",
                    f"Level {(swing.get('strategy_reason') or {}).get('signal_level') or 'Unknown'}",
                ),
                ordered_keys=("level_1", "level_2", "level_3", "level_4", "level_unknown"),
            ),
            "conviction": _grouped_metrics(
                swings,
                lambda swing: (
                    str(
                        ((swing.get("strategy_reason") or {}).get("indicator_assessment") or {}).get(
                            "tier"
                        )
                        or "unknown"
                    ),
                    str(
                        ((swing.get("strategy_reason") or {}).get("indicator_assessment") or {}).get(
                            "tier"
                        )
                        or "unknown"
                    )
                    .replace("_", " ")
                    .title(),
                ),
                ordered_keys=("very_strong", "strong", "neutral", "weak", "unknown"),
            ),
            "duration": _grouped_metrics(
                swings,
                _duration_bucket,
                ordered_keys=tuple(item[0] for item in DURATION_BUCKETS),
            ),
            "outcome": _grouped_metrics(
                swings,
                _outcome_bucket,
                ordered_keys=(
                    "closed_profit",
                    "closed_flat",
                    "closed_loss",
                    "open_profit",
                    "open_flat",
                    "open_underwater",
                ),
            ),
        },
        "risk": {
            "underwater_open_count": len(underwater_open),
            "underwater_open_pnl_quote": sum(_lifecycle_pnl(swing) for swing in underwater_open),
            "profitable_open_count": len(profitable_open),
            "profitable_open_pnl_quote": sum(_lifecycle_pnl(swing) for swing in profitable_open),
            "open_90_plus_days": sum(
                1 for swing in open_swings if _number(swing.get("age_seconds")) >= 90 * 86400
            ),
        },
        "notable": {
            "top_closed": [
                _notable_swing(swing)
                for swing in sorted(closed, key=_lifecycle_pnl, reverse=True)[:5]
            ],
            "worst_open": [
                _notable_swing(swing)
                for swing in sorted(open_swings, key=_lifecycle_pnl)[:8]
            ],
            "oldest_open": [
                _notable_swing(swing)
                for swing in sorted(
                    open_swings,
                    key=lambda item: _number(item.get("age_seconds")),
                    reverse=True,
                )[:5]
            ],
        },
    }
