from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import permutations
import json
from pathlib import Path
from typing import Any

from backtest.spot_market_data import BinanceSpotHistoricalAdapter, SpotHistoricalDataset
from backtest.spot_regime_sweep import (
    REGIME_ORDER,
    SpotRegime,
    SpotRegimeSweepConfig,
    _hodl_metrics,
    load_spot_regime_sweep_config,
)
from backtest.spot_runner import SpotMarketDataProgressBars


WINDOW_DAYS = 365


def _next_month_start(value: datetime) -> datetime:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    return datetime(year, month, 1, tzinfo=UTC)


def monthly_window_starts(start: datetime, end: datetime) -> tuple[datetime, ...]:
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    if cursor < start:
        cursor = _next_month_start(cursor)
    latest_start = end - timedelta(days=WINDOW_DAYS)
    output = []
    while cursor <= latest_start:
        output.append(cursor)
        cursor = _next_month_start(cursor)
    return tuple(output)


def select_non_overlapping_regimes(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(candidates) < 3:
        raise ValueError("At least three candidate windows are required.")
    bull_rank = {
        item["start"]: index
        for index, item in enumerate(
            sorted(candidates, key=lambda row: -float(row["return_pct"]))
        )
    }
    bear_rank = {
        item["start"]: index
        for index, item in enumerate(
            sorted(candidates, key=lambda row: float(row["return_pct"]))
        )
    }
    neutral_rank = {
        item["start"]: index
        for index, item in enumerate(
            sorted(candidates, key=lambda row: abs(float(row["return_pct"])))
        )
    }
    rank_maps = {"bull": bull_rank, "neutral": neutral_rank, "bear": bear_rank}
    best: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None
    for assigned in permutations(candidates, 3):
        selection = dict(zip(REGIME_ORDER, assigned))
        chronological = sorted(assigned, key=lambda row: row["start"])
        if any(
            current["start"] < previous["end"]
            for previous, current in zip(chronological, chronological[1:])
        ):
            continue
        ranks = {
            name: rank_maps[name][selection[name]["start"]] + 1
            for name in REGIME_ORDER
        }
        key = (
            sum(ranks.values()),
            max(ranks.values()),
            -float(selection["bull"]["return_pct"])
            + float(selection["bear"]["return_pct"]),
            abs(float(selection["neutral"]["return_pct"])),
            tuple(selection[name]["start"] for name in REGIME_ORDER),
        )
        if best is None or key < best[0]:
            best = (
                key,
                {
                    name: {**selection[name], "candidate_rank_for_label": ranks[name]}
                    for name in REGIME_ORDER
                },
            )
    if best is None:
        raise ValueError("No three non-overlapping 365-day candidate windows exist.")
    return best[1]


def discover_regimes(
    config: SpotRegimeSweepConfig,
    dataset: SpotHistoricalDataset,
) -> dict[str, Any]:
    if config.common_history_start is None or config.common_history_end is None:
        raise ValueError("Frozen common_history start/end are required for selection.")
    scenario = replace(
        config.sweep.scenario,
        interval="1d",
        warmup_candles=1,
        start=config.common_history_start,
        end=config.common_history_end,
    )
    candidates = []
    for start in monthly_window_starts(scenario.start, scenario.end):
        regime = SpotRegime(
            name="candidate",
            start=start,
            end=start + timedelta(days=WINDOW_DAYS),
            selection={},
        )
        candidates.append(_hodl_metrics(dataset, scenario, regime))
    selected = select_non_overlapping_regimes(candidates)
    frozen = {
        item.name: {"start": item.start.isoformat(), "end": item.end.isoformat()}
        for item in config.regimes
    }
    selected_periods = {
        name: {"start": row["start"], "end": row["end"]}
        for name, row in selected.items()
    }
    return {
        "common_history": {
            "start": scenario.start.isoformat(),
            "end": scenario.end.isoformat(),
        },
        "methodology": config.selection_methodology,
        "window_days": WINDOW_DAYS,
        "step": "calendar_month",
        "candidate_count": len(candidates),
        "selected": selected,
        "frozen_periods_match_selection": frozen == selected_periods,
        "candidates": candidates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select auditable Spot regimes from untouched HODL behavior."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_spot_regime_sweep_config(args.config)
    if config.common_history_start is None or config.common_history_end is None:
        raise ValueError("regime_selection.common_history is required.")
    scenario = replace(
        config.sweep.scenario,
        interval="1d",
        warmup_candles=1,
        start=config.common_history_start,
        end=config.common_history_end,
    )
    progress = SpotMarketDataProgressBars()
    try:
        dataset = BinanceSpotHistoricalAdapter(scenario.data_cache_dir).load(
            scenario,
            progress=progress,
            max_workers=config.sweep.market_data_workers,
        )
    finally:
        progress.close()
    report = discover_regimes(config, dataset)
    output = Path(args.output).expanduser().resolve() if args.output else (
        config.output_dir / "regime_selection.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Regime selection: {output}")
    for name in REGIME_ORDER:
        row = report["selected"][name]
        print(
            f"{name.title()}: {row['start']} -> {row['end']} | "
            f"HODL {row['return_pct']:+.2f}% | DD {row['maximum_drawdown_pct']:.2f}% | "
            f"rank {row['candidate_rank_for_label']}"
        )
    return 0 if report["frozen_periods_match_selection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
