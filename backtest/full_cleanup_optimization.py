"""Run and report staged optimization of the complete waiter-cleanup policy."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import html
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "runtime/spot_backtests/scenarios/treasury-accumulation-baseline-xrp-2y.yaml"
SWEEP_DIR = REPO_ROOT / "runtime/spot_backtests/sweeps/full-cleanup-optimization"
DEFAULT_ROOT = REPO_ROOT / "runtime/spot_backtests/runs/full-cleanup-optimization-neutral-bull-2y"
PRIMARY_START = "2022-12-01T00:00:00+00:00"
PRIMARY_END = "2024-12-01T00:00:00+00:00"
ROBUSTNESS_START = "2024-09-29T00:00:00+00:00"
ROBUSTNESS_END = "2026-09-29T00:00:00+00:00"
ASSET_COUNT = 7
PRIMARY_DAYS = 731

BASELINE = {
    "max_open_bargains_per_asset": 10,
    "capacity_cleanup_min_age_days": 30,
    "deep_loss_min_age_days": 15,
    "unrealized_pnl_pct": -25,
    "deep_loss_required_reverse_level": 1,
    "age_l3_min_age_days": 30,
    "age_l2_min_age_days": 60,
    "age_l1_min_age_days": 90,
}

# These thresholds are fixed before the optimization results are evaluated.
# Restore cost remains informational because an absolute dollar cap is not scale-safe.
ACCEPTANCE = {
    "max_open_bargains_90_plus": 5,
    "max_oldest_open_days": 120.0,
    "max_drawdown_deterioration_pp_vs_baseline": 2.0,
    "max_final_capacity_utilization_pct": 100.0,
    "max_weighted_recovery_deterioration_pp_vs_baseline": 10.0,
}
MEANINGFUL_EDGE_PP = 10.0
MEANINGFUL_GENERATED_CASH_BASELINE_FRACTION = 0.5

CAPACITY_VALUES = (6, 8, 10, 12, 15, 20)
CAPACITY_AGES = (15, 21, 30, 45, 60)
DEEP_AGES = (7, 15, 21, 30)
DEEP_THRESHOLDS = (-20, -25, -30, -35, -45)
DEEP_LEVELS = (1, 2, 3)
AGING_SCHEDULES = (
    (21, 45, 75),
    (21, 45, 90),
    (21, 45, 120),
    (21, 60, 90),
    (21, 60, 120),
    (21, 90, 120),
    (30, 45, 75),
    (30, 45, 90),
    (30, 60, 90),
    (30, 60, 120),
    (30, 90, 120),
    (30, 120, 180),
    (45, 60, 75),
    (45, 60, 90),
    (45, 60, 120),
    (45, 90, 120),
    (45, 90, 180),
    (60, 90, 120),
    (60, 120, 180),
)
INITIAL_STAGES = (
    "stage-0-baseline",
    "stage-1-capacity",
    "stage-2-deep-loss",
    "stage-3-aging",
)
PRIMARY_STAGES = INITIAL_STAGES + ("stage-4-joint", "stage-5-refinement")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _slug_number(value: float | int) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _variant_name(values: dict[str, Any]) -> str:
    return (
        f"cap{values['max_open_bargains_per_asset']}"
        f"-ca{_slug_number(values['capacity_cleanup_min_age_days'])}"
        f"-d{_slug_number(values['deep_loss_min_age_days'])}"
        f"-p{_slug_number(values['unrealized_pnl_pct'])}"
        f"-r{values['deep_loss_required_reverse_level']}"
        f"-a{_slug_number(values['age_l3_min_age_days'])}"
        f"-{_slug_number(values['age_l2_min_age_days'])}"
        f"-{_slug_number(values['age_l1_min_age_days'])}"
    )


def complete_variant(**overrides: Any) -> dict[str, Any]:
    values = {**BASELINE, **overrides}
    variant = {
        "name": _variant_name(values),
        "deep_loss_enabled": True,
        "aging_cleanup_enabled": True,
        "capacity_cleanup_enabled": True,
        **values,
    }
    validate_variant(variant)
    return variant


def validate_variant(variant: dict[str, Any]) -> None:
    for key in (
        "deep_loss_enabled",
        "aging_cleanup_enabled",
        "capacity_cleanup_enabled",
    ):
        if variant.get(key) is not True:
            raise ValueError(f"{variant.get('name', 'candidate')}: {key} must remain true.")
    cap = variant.get("max_open_bargains_per_asset")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError("max_open_bargains_per_asset must remain a positive integer.")
    l3 = float(variant["age_l3_min_age_days"])
    l2 = float(variant["age_l2_min_age_days"])
    l1 = float(variant["age_l1_min_age_days"])
    if not l3 < l2 < l1:
        raise ValueError(f"Invalid aging schedule: {l3:g} < {l2:g} < {l1:g} is required.")


def _deduplicate(variants: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for variant in variants:
        validate_variant(variant)
        key = _parameter_key(variant)
        if key in seen:
            continue
        seen.add(key)
        output.append(variant)
    return output


def _parameter_key(values: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(values[field] for field in BASELINE)


def _exclude_evaluated(
    variants: Iterable[dict[str, Any]],
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluated = {_parameter_key(row) for row in rows}
    return [
        variant
        for variant in _deduplicate(variants)
        if _parameter_key(variant) not in evaluated
    ]


def initial_stage_variants() -> dict[str, list[dict[str, Any]]]:
    return {
        "stage-0-baseline": [complete_variant()],
        "stage-1-capacity": [
            complete_variant(
                max_open_bargains_per_asset=cap,
                capacity_cleanup_min_age_days=age,
            )
            for cap in CAPACITY_VALUES
            for age in CAPACITY_AGES
            if (cap, age)
            != (
                BASELINE["max_open_bargains_per_asset"],
                BASELINE["capacity_cleanup_min_age_days"],
            )
        ],
        "stage-2-deep-loss": [
            complete_variant(
                deep_loss_min_age_days=age,
                unrealized_pnl_pct=threshold,
                deep_loss_required_reverse_level=level,
            )
            for age in DEEP_AGES
            for threshold in DEEP_THRESHOLDS
            for level in DEEP_LEVELS
            if (age, threshold, level)
            != (
                BASELINE["deep_loss_min_age_days"],
                BASELINE["unrealized_pnl_pct"],
                BASELINE["deep_loss_required_reverse_level"],
            )
        ],
        "stage-3-aging": [
            complete_variant(
                age_l3_min_age_days=l3,
                age_l2_min_age_days=l2,
                age_l1_min_age_days=l1,
            )
            for l3, l2, l1 in AGING_SCHEDULES
            if (l3, l2, l1)
            != (
                BASELINE["age_l3_min_age_days"],
                BASELINE["age_l2_min_age_days"],
                BASELINE["age_l1_min_age_days"],
            )
        ],
    }


def _sweep_payload(
    *,
    stage: str,
    root: Path,
    variants: list[dict[str, Any]],
    start: str = PRIMARY_START,
    end: str = PRIMARY_END,
    workers: int = 6,
) -> dict[str, Any]:
    variants = _deduplicate(variants)
    if not variants:
        raise ValueError(f"{stage} has no unique candidates.")
    return {
        "spot_sweep": {
            "name": f"full-cleanup-optimization-{stage}",
            "base_config": str(BASE_CONFIG),
            "output_dir": str(root / stage),
            "start": start,
            "end": end,
            "market_data_workers": 7,
            "replay_parallel": True,
            "replay_max_workers": workers,
            "resume": True,
            "continue_on_error": True,
            "variants": variants,
        }
    }


def write_stage_config(
    *,
    stage: str,
    root: Path,
    variants: list[dict[str, Any]],
    start: str = PRIMARY_START,
    end: str = PRIMARY_END,
    workers: int = 6,
) -> Path:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    path = SWEEP_DIR / f"{stage}.yaml"
    payload = _sweep_payload(
        stage=stage,
        root=root,
        variants=variants,
        start=start,
        end=end,
        workers=workers,
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def prepare_initial(root: Path, workers: int) -> list[Path]:
    return [
        write_stage_config(stage=stage, root=root, variants=variants, workers=workers)
        for stage, variants in initial_stage_variants().items()
    ]


def _load_stage_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    path = root / stage / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed stage: {path}")
    return [
        {**row, "stage": stage}
        for row in _read_json(path).get("rows", [])
        if row.get("status") == "ok"
    ]


def _pareto(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cleanup_cash_burn_ratio_pct") is not None
    ]
    output = []
    for row in candidates:
        edge = float(row["edge_vs_hodl_pct_points"])
        burn = float(row["cleanup_cash_burn_ratio_pct"])
        if any(
            float(other["edge_vs_hodl_pct_points"]) >= edge
            and float(other["cleanup_cash_burn_ratio_pct"]) <= burn
            and (
                float(other["edge_vs_hodl_pct_points"]) > edge
                or float(other["cleanup_cash_burn_ratio_pct"]) < burn
            )
            for other in candidates
            if other is not row
        ):
            continue
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            -float(row["edge_vs_hodl_pct_points"]),
            float(row["cleanup_cash_burn_ratio_pct"]),
        ),
    )


def _select_regions(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    count: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot select regions from an empty stage.")
    baseline_cash = max(float(row["gross_profit_cash_generated"]) for row in rows)
    meaningful = [
        row
        for row in rows
        if float(row["edge_vs_hodl_pct_points"]) >= MEANINGFUL_EDGE_PP
        and float(row["gross_profit_cash_generated"])
        >= baseline_cash * MEANINGFUL_GENERATED_CASH_BASELINE_FRACTION
    ] or rows
    ordered: list[dict[str, Any]] = []
    ordered.extend(_pareto(meaningful))
    ordered.extend(sorted(meaningful, key=lambda row: -float(row["edge_vs_hodl_pct_points"])))
    ordered.extend(
        sorted(
            meaningful,
            key=lambda row: (
                float(row["cleanup_cash_burn_ratio_pct"]),
                -float(row["edge_vs_hodl_pct_points"]),
            ),
        )
    )
    ordered.extend(
        sorted(
            meaningful,
            key=lambda row: (
                float(row["unique_capacity_blocked_opportunities"]),
                -float(row["edge_vs_hodl_pct_points"]),
            ),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in ordered:
        key = tuple(row[field] for field in fields)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} distinct regions available; expected {count}.")
    return selected


def _raw_operationally_accepted(
    row: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    capacity_slots = ASSET_COUNT * int(row["max_open_bargains_per_asset"])
    final_utilization = float(row["open_bargains"]) / capacity_slots * 100
    return (
        int(row["open_bargains_90_plus"])
        <= ACCEPTANCE["max_open_bargains_90_plus"]
        and float(row["oldest_open_days"])
        <= ACCEPTANCE["max_oldest_open_days"]
        and float(row["maximum_drawdown_pct"])
        >= float(baseline["maximum_drawdown_pct"])
        - ACCEPTANCE["max_drawdown_deterioration_pp_vs_baseline"]
        and final_utilization
        <= ACCEPTANCE["max_final_capacity_utilization_pct"]
        and float(row["weighted_effective_assets_pct"])
        >= float(baseline["weighted_effective_assets_pct"])
        - ACCEPTANCE["max_weighted_recovery_deterioration_pp_vs_baseline"]
    )


def _select_capacity_regions(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if _raw_operationally_accepted(row, baseline)]
    if not eligible:
        raise ValueError("No operationally accepted capacity regions are available.")
    winner = max(eligible, key=lambda row: float(row["edge_vs_hodl_pct_points"]))
    same_cap = sorted(
        (
            row
            for row in eligible
            if row is not winner
            and row["max_open_bargains_per_asset"]
            == winner["max_open_bargains_per_asset"]
        ),
        key=lambda row: -float(row["edge_vs_hodl_pct_points"]),
    )
    best_by_other_cap = []
    for cap in sorted({int(row["max_open_bargains_per_asset"]) for row in eligible}):
        if cap == int(winner["max_open_bargains_per_asset"]):
            continue
        best_by_other_cap.append(
            max(
                (row for row in eligible if int(row["max_open_bargains_per_asset"]) == cap),
                key=lambda row: float(row["edge_vs_hodl_pct_points"]),
            )
        )
    selected = [winner]
    if same_cap:
        selected.append(same_cap[0])
    selected.extend(
        sorted(
            best_by_other_cap,
            key=lambda row: -float(row["edge_vs_hodl_pct_points"]),
        )[: 4 - len(selected)]
    )
    if len(selected) != 4:
        raise ValueError(f"Only {len(selected)} capacity regions are available; expected 4.")
    return selected


def _row_overrides(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in BASELINE}


def prepare_joint(root: Path, workers: int) -> Path:
    baseline = _load_stage_rows(root, "stage-0-baseline")[0]
    capacities = _select_capacity_regions(
        _load_stage_rows(root, "stage-1-capacity"), baseline
    )
    deep_loss_rows = [
        row
        for row in _load_stage_rows(root, "stage-2-deep-loss")
        if _raw_operationally_accepted(row, baseline)
    ]
    deep_losses = _select_regions(
        deep_loss_rows,
        fields=(
            "deep_loss_min_age_days",
            "unrealized_pnl_pct",
            "deep_loss_required_reverse_level",
        ),
        count=4,
    )
    aging_rows = [
        row
        for row in _load_stage_rows(root, "stage-3-aging")
        if _raw_operationally_accepted(row, baseline)
    ]
    aging = _select_regions(
        aging_rows,
        fields=("age_l3_min_age_days", "age_l2_min_age_days", "age_l1_min_age_days"),
        count=3,
    )
    variants = []
    for capacity in capacities:
        for deep_loss in deep_losses:
            for ages in aging:
                variants.append(
                    complete_variant(
                        max_open_bargains_per_asset=capacity["max_open_bargains_per_asset"],
                        capacity_cleanup_min_age_days=capacity["capacity_cleanup_min_age_days"],
                        deep_loss_min_age_days=deep_loss["deep_loss_min_age_days"],
                        unrealized_pnl_pct=deep_loss["unrealized_pnl_pct"],
                        deep_loss_required_reverse_level=deep_loss[
                            "deep_loss_required_reverse_level"
                        ],
                        age_l3_min_age_days=ages["age_l3_min_age_days"],
                        age_l2_min_age_days=ages["age_l2_min_age_days"],
                        age_l1_min_age_days=ages["age_l1_min_age_days"],
                    )
                )
    previous = [
        row
        for stage in INITIAL_STAGES
        for row in _load_stage_rows(root, stage)
    ]
    variants = _exclude_evaluated(variants, previous)
    if len(variants) < 30:
        raise ValueError(f"Joint optimization produced only {len(variants)} new candidates.")
    return write_stage_config(
        stage="stage-4-joint",
        root=root,
        variants=variants,
        workers=workers,
    )


def _nearby(value: float, candidates: Iterable[float], count: int = 3) -> list[float]:
    return sorted(candidates, key=lambda candidate: (abs(candidate - value), candidate))[:count]


def prepare_refinement(root: Path, workers: int) -> Path:
    seeds = _select_regions(
        _load_stage_rows(root, "stage-4-joint"),
        fields=tuple(BASELINE),
        count=4,
    )
    variants: list[dict[str, Any]] = []
    for seed in seeds:
        base = _row_overrides(seed)
        variants.append(complete_variant(**base))
        for cap in _nearby(float(base["max_open_bargains_per_asset"]), range(6, 21), 5):
            variants.append(complete_variant(**{**base, "max_open_bargains_per_asset": int(cap)}))
        for age in _nearby(
            float(base["capacity_cleanup_min_age_days"]),
            (15, 21, 25, 28, 30, 32, 35, 40, 45, 60),
            5,
        ):
            variants.append(complete_variant(**{**base, "capacity_cleanup_min_age_days": age}))
        for age in _nearby(
            float(base["deep_loss_min_age_days"]),
            (7, 10, 12, 15, 18, 21, 25, 30),
            4,
        ):
            variants.append(complete_variant(**{**base, "deep_loss_min_age_days": age}))
        for threshold in _nearby(
            float(base["unrealized_pnl_pct"]),
            (-20, -25, -30, -35, -40, -45),
            4,
        ):
            variants.append(complete_variant(**{**base, "unrealized_pnl_pct": threshold}))
        for level in (1, 2, 3):
            variants.append(
                complete_variant(**{**base, "deep_loss_required_reverse_level": level})
            )
        l3 = float(base["age_l3_min_age_days"])
        l2 = float(base["age_l2_min_age_days"])
        l1 = float(base["age_l1_min_age_days"])
        for schedule in AGING_SCHEDULES:
            distance = abs(schedule[0] - l3) + abs(schedule[1] - l2) + abs(schedule[2] - l1)
            if distance <= 45:
                variants.append(
                    complete_variant(
                        **{
                            **base,
                            "age_l3_min_age_days": schedule[0],
                            "age_l2_min_age_days": schedule[1],
                            "age_l1_min_age_days": schedule[2],
                        }
                    )
                )
    previous = [
        row
        for stage in INITIAL_STAGES + ("stage-4-joint",)
        for row in _load_stage_rows(root, stage)
    ]
    unique = _exclude_evaluated(variants, previous)[:60]
    if len(unique) < 30:
        raise ValueError(f"Local refinement produced only {len(unique)} candidates.")
    return write_stage_config(
        stage="stage-5-refinement",
        root=root,
        variants=unique,
        workers=workers,
    )


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(Path(row["output_dir"]) / "summary.json")
    cleanup = summary["waiter_cleanup"]
    cash_flow = summary["bargain_cash_flow"]["by_close_type"]["profit_target"]
    profit_closes = int(cash_flow["closed_sell_bargains"])
    capacity_slots = ASSET_COUNT * int(row["max_open_bargains_per_asset"])
    asset_days = ASSET_COUNT * PRIMARY_DAYS
    averages = [
        float(item["average_open_bargain_count"])
        for item in cleanup["per_asset"].values()
    ]
    accumulated = summary["treasury_accumulation"]["per_asset"]
    return {
        **row,
        "committed_cash_quote": float(
            summary["success_metrics"]["free_reserve"]["committed_quote"]
        ),
        "relative_wealth_vs_hodl": float(summary["portfolio"]["relative_wealth"]["terminal"]),
        "profit_target_closes": profit_closes,
        "profit_target_closes_per_1000_asset_days": profit_closes / asset_days * 1000,
        "unique_blocked_per_1000_asset_days": float(
            row["unique_capacity_blocked_opportunities"]
        )
        / asset_days
        * 1000,
        "cleanup_top_up_per_profit_close": (
            float(row["gross_cleanup_top_up"]) / profit_closes if profit_closes else None
        ),
        "average_capacity_utilization_pct": sum(averages) / capacity_slots * 100,
        "final_capacity_utilization_pct": float(row["open_bargains"]) / capacity_slots * 100,
        "cleanup_closes_by_mechanism": {
            reason: int(metrics["closes"])
            for reason, metrics in cleanup["by_reason"].items()
        },
        "accumulation_usdt_deployed": float(
            summary["treasury_accumulation"]["overview"]["usdt_deployed"]
        ),
        "accumulated_asset_quantities": {
            symbol: float(metrics.get("net_asset_acquired") or 0.0)
            for symbol, metrics in accumulated.items()
        },
        "per_asset_cleanup": cleanup["per_asset"],
    }


def _annotate_acceptance(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    for row in rows:
        row["operationally_accepted"] = (
            int(row["open_bargains_90_plus"])
            <= ACCEPTANCE["max_open_bargains_90_plus"]
            and float(row["oldest_open_days"])
            <= ACCEPTANCE["max_oldest_open_days"]
            and float(row["maximum_drawdown_pct"])
            >= float(baseline["maximum_drawdown_pct"])
            - ACCEPTANCE["max_drawdown_deterioration_pp_vs_baseline"]
            and float(row["final_capacity_utilization_pct"])
            <= ACCEPTANCE["max_final_capacity_utilization_pct"]
            and float(row["weighted_effective_assets_pct"])
            >= float(baseline["weighted_effective_assets_pct"])
            - ACCEPTANCE["max_weighted_recovery_deterioration_pp_vs_baseline"]
        )


def _balanced_candidate(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if not frontier:
        raise ValueError("No Pareto-efficient candidates are available.")
    edges = [float(row["edge_vs_hodl_pct_points"]) for row in frontier]
    burns = [float(row["cleanup_cash_burn_ratio_pct"]) for row in frontier]
    edge_span = max(edges) - min(edges) or 1.0
    burn_span = max(burns) - min(burns) or 1.0
    return min(
        frontier,
        key=lambda row: math.hypot(
            (max(edges) - float(row["edge_vs_hodl_pct_points"])) / edge_span,
            (float(row["cleanup_cash_burn_ratio_pct"]) - min(burns)) / burn_span,
        ),
    )


def _all_primary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for stage in PRIMARY_STAGES:
        for raw in _load_stage_rows(root, stage):
            key = _parameter_key(raw)
            if key in seen:
                continue
            seen.add(key)
            rows.append(_enrich_row(raw))
    return rows


def _recommendations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accepted = [row for row in rows if row["operationally_accepted"]]
    pool = accepted or rows
    frontier = _pareto(pool)
    highest_edge = max(pool, key=lambda row: float(row["edge_vs_hodl_pct_points"]))
    meaningful = [
        row
        for row in pool
        if float(row["edge_vs_hodl_pct_points"]) >= MEANINGFUL_EDGE_PP
        and float(row["gross_profit_cash_generated"])
        >= float(highest_edge["gross_profit_cash_generated"])
        * MEANINGFUL_GENERATED_CASH_BASELINE_FRACTION
    ] or pool
    lowest_burn = min(
        meaningful,
        key=lambda row: (
            float(row["cleanup_cash_burn_ratio_pct"]),
            -float(row["edge_vs_hodl_pct_points"]),
        ),
    )
    capacity = min(
        meaningful,
        key=lambda row: (
            float(row["unique_blocked_per_1000_asset_days"]),
            -float(row["profit_target_closes_per_1000_asset_days"]),
            -float(row["edge_vs_hodl_pct_points"]),
        ),
    )
    return {
        "highest_edge": highest_edge,
        "lowest_burn_meaningful_edge": lowest_burn,
        "best_balance": _balanced_candidate(frontier),
        "best_capacity_efficiency": capacity,
    }


def prepare_robustness(root: Path, workers: int) -> Path:
    rows = _all_primary_rows(root)
    baseline = next(row for row in rows if row["stage"] == "stage-0-baseline")
    _annotate_acceptance(rows, baseline)
    recommendations = _recommendations(rows)
    finalists = _deduplicate(
        complete_variant(**_row_overrides(row)) for row in recommendations.values()
    )
    return write_stage_config(
        stage="robustness-2024-2026",
        root=root,
        variants=finalists,
        start=ROBUSTNESS_START,
        end=ROBUSTNESS_END,
        workers=workers,
    )


def _sample_equity(path: Path, points: int = 180) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return []
    step = max(1, len(rows) // points)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return [
        {
            "timestamp_ms": float(row["timestamp_ms"]),
            "treasury": float(row["treasury_value"]),
            "hodl": float(row["hodl_value"]),
        }
        for row in sampled
    ]


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _report_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html><html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scrooge Full Cleanup Optimization</title><style>
:root{{--bg:#080d12;--panel:#111923;--line:#2a3949;--ink:#e8edf3;--muted:#91a0b1;--gold:#e9b949;--green:#4ad7a5;--red:#ff667f;--blue:#65a7e8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 85% 0,#30121e 0,transparent 34%),var(--bg);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Menlo,monospace}}main{{width:min(1600px,calc(100% - 28px));margin:28px auto 60px}}header,.panel{{border:1px solid var(--line);border-radius:18px;background:rgba(17,25,35,.96)}}header{{padding:28px;margin-bottom:18px;background:linear-gradient(125deg,#351020,#111923)}}h1{{margin:8px 0 12px;font-size:32px}}h2{{font-size:17px}}p,.muted{{color:var(--muted)}}strong,.gold{{color:var(--gold)}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-bottom:18px}}.panel{{padding:18px;overflow:auto}}canvas{{width:100%;height:300px;background:#0b1118;border:1px solid #202e3d;border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:1450px}}th,td{{border-top:1px solid var(--line);padding:9px;text-align:right;white-space:nowrap}}th{{padding:0;color:var(--muted);position:sticky;top:0;background:var(--panel)}}th:first-child,td:first-child{{text-align:left}}th button{{width:100%;border:0;background:transparent;color:inherit;padding:10px;font:inherit;text-align:inherit;cursor:pointer}}.yes{{color:var(--green)}}.no{{color:var(--red)}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}h1{{font-size:24px}}}}</style></head><body><main><header><span class="gold">SCROOGE / FULL CLEANUP OPTIMIZATION</span><h1>All mechanisms enabled</h1><p id="summary"></p></header><section class="grid"><div class="panel"><h2>Edge vs Cleanup Cash Burn</h2><canvas id="edgeBurn"></canvas></div><div class="panel"><h2>Edge vs Max Open Bargains</h2><canvas id="edgeCap"></canvas></div><div class="panel"><h2>Cleanup Burn vs Max Open Bargains</h2><canvas id="burnCap"></canvas></div><div class="panel"><h2>Blocked Opportunities vs Max Open Bargains</h2><canvas id="blockedCap"></canvas></div><div class="panel"><h2>Deep-loss Age vs Threshold</h2><canvas id="deep"></canvas></div><div class="panel"><h2>Equity vs HODL</h2><canvas id="equity"></canvas></div><div class="panel"><h2>Winner Neighborhood</h2><canvas id="refinement"></canvas></div></section><section class="panel"><table id="results"><thead><tr><th><button data-type="text">Candidate</button></th><th><button data-type="text">Stage</button></th><th><button data-type="number">Edge pp</button></th><th><button data-type="number">Burn %</button></th><th><button data-type="number">Final</button></th><th><button data-type="number">Cap</button></th><th><button data-type="number">Cap age</button></th><th><button data-type="number">Deep age</button></th><th><button data-type="number">Deep PnL</button></th><th><button data-type="number">Reverse</button></th><th><button data-type="number">Aging L3/L2/L1</button></th><th><button data-type="number">Blocked/1k</button></th><th><button data-type="number">Final util</button></th><th><button data-type="number">90d+</button></th><th><button data-type="number">Oldest</button></th><th><button data-type="number">Accepted</button></th></tr></thead><tbody></tbody></table></section></main><script id="data" type="application/json">{data}</script><script>
const d=JSON.parse(document.getElementById('data').textContent),rows=d.rows;document.getElementById('summary').innerHTML=`${{rows.length}} unique candidates · <strong>${{d.accepted_count}} accepted</strong> · Best balance <strong>${{d.recommendations.best_balance.name}}</strong>`;
const tbody=document.querySelector('tbody');tbody.innerHTML=rows.map((r,i)=>`<tr data-i="${{i}}"><td data-sort="${{r.name}}">${{r.name}}</td><td data-sort="${{r.stage}}">${{r.stage}}</td><td data-sort="${{r.edge_vs_hodl_pct_points}}">${{r.edge_vs_hodl_pct_points.toFixed(2)}}</td><td data-sort="${{r.cleanup_cash_burn_ratio_pct}}">${{r.cleanup_cash_burn_ratio_pct.toFixed(2)}}</td><td data-sort="${{r.final_treasury_value}}">$${{r.final_treasury_value.toLocaleString()}}</td><td data-sort="${{r.max_open_bargains_per_asset}}">${{r.max_open_bargains_per_asset}}</td><td data-sort="${{r.capacity_cleanup_min_age_days}}">${{r.capacity_cleanup_min_age_days}}d</td><td data-sort="${{r.deep_loss_min_age_days}}">${{r.deep_loss_min_age_days}}d</td><td data-sort="${{r.unrealized_pnl_pct}}">${{r.unrealized_pnl_pct}}%</td><td data-sort="${{r.deep_loss_required_reverse_level}}">L${{r.deep_loss_required_reverse_level}}</td><td data-sort="${{r.age_l3_min_age_days}}">${{r.age_l3_min_age_days}}/${{r.age_l2_min_age_days}}/${{r.age_l1_min_age_days}}</td><td data-sort="${{r.unique_blocked_per_1000_asset_days}}">${{r.unique_blocked_per_1000_asset_days.toFixed(1)}}</td><td data-sort="${{r.final_capacity_utilization_pct}}">${{r.final_capacity_utilization_pct.toFixed(1)}}%</td><td data-sort="${{r.open_bargains_90_plus}}">${{r.open_bargains_90_plus}}</td><td data-sort="${{r.oldest_open_days}}">${{r.oldest_open_days.toFixed(1)}}d</td><td class="${{r.operationally_accepted?'yes':'no'}}" data-sort="${{+r.operationally_accepted}}">${{r.operationally_accepted?'yes':'no'}}</td></tr>`).join('');
document.querySelectorAll('th button').forEach((b,i)=>b.onclick=()=>{{const type=b.dataset.type,dir=b.dataset.dir==='up'?-1:1;b.dataset.dir=dir===1?'up':'down';[...tbody.rows].sort((a,z)=>{{let x=a.cells[i].dataset.sort,y=z.cells[i].dataset.sort;return(type==='number'?Number(x)-Number(y):x.localeCompare(y))*dir}}).forEach(r=>tbody.appendChild(r))}});
function chart(id,points,xKey,yKey,xLabel,yLabel,color='#e9b949'){{const c=document.getElementById(id),ctx=c.getContext('2d'),ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*ratio;c.height=h*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);const m={{l:52,r:18,t:18,b:35}},xs=points.map(p=>+p[xKey]),ys=points.map(p=>+p[yKey]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>m.l+(x-xmin)/(xmax-xmin||1)*(w-m.l-m.r),sy=y=>h-m.b-(y-ymin)/(ymax-ymin||1)*(h-m.t-m.b);ctx.strokeStyle='#293747';ctx.fillStyle='#91a0b1';ctx.font='11px monospace';ctx.beginPath();ctx.moveTo(m.l,m.t);ctx.lineTo(m.l,h-m.b);ctx.lineTo(w-m.r,h-m.b);ctx.stroke();ctx.fillText(xLabel,w/2,h-8);ctx.save();ctx.translate(12,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();points.forEach(p=>{{ctx.beginPath();ctx.fillStyle=p.pareto_efficient?'#4ad7a5':color;ctx.arc(sx(+p[xKey]),sy(+p[yKey]),p.pareto_efficient?4:2.5,0,Math.PI*2);ctx.fill()}})}}
chart('edgeBurn',rows,'cleanup_cash_burn_ratio_pct','edge_vs_hodl_pct_points','Burn %','Edge pp');chart('edgeCap',rows,'max_open_bargains_per_asset','edge_vs_hodl_pct_points','Cap','Edge pp','#65a7e8');chart('burnCap',rows,'max_open_bargains_per_asset','cleanup_cash_burn_ratio_pct','Cap','Burn %','#ff9f66');chart('blockedCap',rows,'max_open_bargains_per_asset','unique_blocked_per_1000_asset_days','Cap','Blocked / 1k','#ff667f');chart('deep',rows,'deep_loss_min_age_days','unrealized_pnl_pct','Deep age','Loss threshold','#65a7e8');chart('refinement',rows.filter(r=>r.stage==='stage-5-refinement'),'cleanup_cash_burn_ratio_pct','edge_vs_hodl_pct_points','Burn %','Edge pp');
function lines(id,series){{const points=series.flatMap(s=>s.points),c=document.getElementById(id),ctx=c.getContext('2d'),ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*ratio;c.height=h*ratio;ctx.scale(ratio,ratio);const m={{l:52,r:18,t:18,b:35}},xs=points.map(p=>p.timestamp_ms),ys=points.flatMap(p=>[p.treasury,p.hodl]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>m.l+(x-xmin)/(xmax-xmin)*(w-m.l-m.r),sy=y=>h-m.b-(y-ymin)/(ymax-ymin)*(h-m.t-m.b),colors=['#e9b949','#4ad7a5','#65a7e8'];ctx.strokeStyle='#293747';ctx.strokeRect(m.l,m.t,w-m.l-m.r,h-m.t-m.b);series.forEach((s,i)=>{{ctx.strokeStyle=colors[i%colors.length];ctx.beginPath();s.points.forEach((p,j)=>{{const y=s.name==='HODL'?p.hodl:p.treasury;j?ctx.lineTo(sx(p.timestamp_ms),sy(y)):ctx.moveTo(sx(p.timestamp_ms),sy(y))}});ctx.stroke();ctx.fillStyle=colors[i%colors.length];ctx.fillText(s.name,m.l+8,m.t+14+i*14)}})}}lines('equity',d.equity);
</script></body></html>"""


def build_report(root: Path) -> dict[str, Any]:
    rows = _all_primary_rows(root)
    baseline = next(row for row in rows if row["stage"] == "stage-0-baseline")
    _annotate_acceptance(rows, baseline)
    accepted = [row for row in rows if row["operationally_accepted"]]
    frontier = _pareto(accepted or rows)
    frontier_keys = {_parameter_key(row) for row in frontier}
    for row in rows:
        row["pareto_efficient"] = _parameter_key(row) in frontier_keys
    recommendations = _recommendations(rows)
    robustness_rows = []
    robustness_path = root / "robustness-2024-2026" / "comparison.json"
    if robustness_path.exists():
        robustness_rows = _read_json(robustness_path).get("rows", [])
    equity = [{"name": "HODL", "points": _sample_equity(Path(baseline["output_dir"]) / "equity.csv")}]
    for label, row in (
        ("Baseline", baseline),
        ("Highest Edge", recommendations["highest_edge"]),
        ("Best Balance", recommendations["best_balance"]),
    ):
        if any(item["name"] == label for item in equity):
            continue
        equity.append({"name": label, "points": _sample_equity(Path(row["output_dir"]) / "equity.csv")})
    per_asset = []
    mechanisms = []
    for category, row in recommendations.items():
        summary = _read_json(Path(row["output_dir"]) / "summary.json")
        for symbol, metrics in summary["waiter_cleanup"]["per_asset"].items():
            per_asset.append({"category": category, "candidate": row["name"], "asset": symbol, **metrics})
        for reason, metrics in summary["waiter_cleanup"]["by_reason"].items():
            mechanisms.append({"category": category, "candidate": row["name"], "mechanism": reason, **metrics})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": commit,
        "scenario": {"start": PRIMARY_START, "end": PRIMARY_END, "assets": ASSET_COUNT},
        "acceptance_thresholds": ACCEPTANCE,
        "accepted_count": len(accepted),
        "rows": sorted(rows, key=lambda row: -float(row["edge_vs_hodl_pct_points"])),
        "pareto_frontier": frontier,
        "recommendations": recommendations,
        "robustness": robustness_rows,
        "equity": equity,
    }
    report_dir = root / "research"
    _write_json(report_dir / "full_results.json", payload)
    _write_csv(report_dir / "full_results.csv", payload["rows"])
    _write_json(report_dir / "pareto_frontier.json", frontier)
    _write_csv(report_dir / "pareto_frontier.csv", frontier)
    _write_json(report_dir / "per_asset_cleanup.json", per_asset)
    _write_csv(report_dir / "per_asset_cleanup.csv", per_asset)
    _write_json(report_dir / "per_mechanism_cleanup.json", mechanisms)
    _write_csv(report_dir / "per_mechanism_cleanup.csv", mechanisms)
    _write_json(report_dir / "robustness.json", robustness_rows)
    _write_csv(report_dir / "robustness.csv", robustness_rows)
    highest = recommendations["highest_edge"]
    balance = recommendations["best_balance"]
    lines = [
        "# Повна оптимізація Cleanup",
        "",
        "## Резюме для Argo",
        "",
        f"Перевірено {len(rows)} унікальних конфігурацій на {PRIMARY_START[:10]}–{PRIMARY_END[:10]}; "
        f"усі три cleanup-механізми та Max Open Bargains були активні. "
        f"Operational acceptance пройшли {len(accepted)} кандидатів.",
        "",
        f"Найвищий Edge: `{highest['name']}` — {highest['edge_vs_hodl_pct_points']:.2f} pp, "
        f"burn {highest['cleanup_cash_burn_ratio_pct']:.2f}%, final {_money(highest['final_treasury_value'])}.",
        f"Найкращий баланс на Pareto frontier: `{balance['name']}` — "
        f"{balance['edge_vs_hodl_pct_points']:.2f} pp, burn {balance['cleanup_cash_burn_ratio_pct']:.2f}%.",
        "",
        "Restore Cost не є acceptance-обмеженням і наведений лише як інформаційна оцінка. "
        "Repeated per-minute blocked evaluations не трактуються як окремі можливості.",
        "",
        "## Рекомендації",
        "",
    ]
    for category, row in recommendations.items():
        lines.append(
            f"- **{category}**: `{row['name']}`; Edge {row['edge_vs_hodl_pct_points']:.2f} pp; "
            f"burn {row['cleanup_cash_burn_ratio_pct']:.2f}%; blocked "
            f"{row['unique_blocked_per_1000_asset_days']:.2f}/1k asset-days."
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"Acceptance thresholds: `{json.dumps(ACCEPTANCE, sort_keys=True)}`.",
            f"Git revision: `{commit}`. Primary ranking is based only on the 2022–2024 window; "
            "the 2024–2026 run is secondary robustness evidence.",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "report.html").write_text(_report_html(payload), encoding="utf-8")
    return payload


def _run_sweep(path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "backtest.spot_sweep", "--config", str(path)],
        cwd=REPO_ROOT,
        check=True,
    )


def run_all(root: Path, workers: int) -> dict[str, Any]:
    paths = prepare_initial(root, workers)
    for path in paths:
        _run_sweep(path)
    baseline = _load_stage_rows(root, "stage-0-baseline")[0]
    if (
        abs(float(baseline["final_treasury_value"]) - 67_251.69) > 1.0
        or abs(float(baseline["edge_vs_hodl_pct_points"]) - 19.54) > 0.05
        or abs(float(baseline["cleanup_cash_burn_ratio_pct"]) - 89.32) > 0.05
    ):
        raise RuntimeError("Baseline materially differs from the reproduced reference.")
    joint = prepare_joint(root, workers)
    _run_sweep(joint)
    refinement = prepare_refinement(root, workers)
    _run_sweep(refinement)
    robustness = prepare_robustness(root, workers)
    _run_sweep(robustness)
    return build_report(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "prepare-joint",
            "prepare-refinement",
            "prepare-robustness",
            "report",
            "run",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.action == "prepare":
        paths = prepare_initial(root, args.workers)
    elif args.action == "prepare-joint":
        paths = [prepare_joint(root, args.workers)]
    elif args.action == "prepare-refinement":
        paths = [prepare_refinement(root, args.workers)]
    elif args.action == "prepare-robustness":
        paths = [prepare_robustness(root, args.workers)]
    elif args.action == "report":
        payload = build_report(root)
        print(payload["recommendations"]["best_balance"]["name"])
        return 0
    else:
        payload = run_all(root, args.workers)
        print(payload["recommendations"]["best_balance"]["name"])
        return 0
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
