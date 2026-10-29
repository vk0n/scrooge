"""Run the staged neutral-bull optimization for the marked capacity-cleanup policy."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPO_ROOT
    / "runtime/spot_backtests/scenarios/treasury-accumulation-baseline-xrp-2y.yaml"
)
SWEEP_DIR = REPO_ROOT / "runtime/spot_backtests/sweeps/capacity-cleanup-policy"
DEFAULT_ROOT = (
    REPO_ROOT
    / "runtime/spot_backtests/runs/capacity-cleanup-policy-neutral-bull-2y"
)
START = "2022-12-01T00:00:00+00:00"
END = "2024-12-01T00:00:00+00:00"

CAPACITY_AGES = (15, 20, 30, 45)
CAPACITY_LEVELS = (1, 2, 3, 4)
DEEP_AGES = (5, 6, 7, 10, 12, 14)
DEEP_THRESHOLDS = (-40, -45, -50)
DEEP_LEVELS = (1, 2, 3, 4)


def _name(*, capacity_age: float, capacity_level: int, deep_age: float, deep_pnl: float, deep_level: int) -> str:
    return (
        f"cap-age-{capacity_age:g}-l{capacity_level}"
        f"-deep-{deep_age:g}-p{abs(deep_pnl):g}-l{deep_level}"
    )


def _variant(
    *,
    capacity_age: float,
    capacity_level: int,
    deep_age: float = 7,
    deep_pnl: float = -45,
    deep_level: int = 3,
) -> dict[str, Any]:
    return {
        "name": _name(
            capacity_age=capacity_age,
            capacity_level=capacity_level,
            deep_age=deep_age,
            deep_pnl=deep_pnl,
            deep_level=deep_level,
        ),
        "max_open_bargains_per_asset": 7,
        "aging_cleanup_enabled": True,
        "age_l3_min_age_days": 30,
        "age_l2_min_age_days": 60,
        "age_l1_min_age_days": 90,
        "capacity_cleanup_enabled": True,
        "capacity_cleanup_min_age_days": capacity_age,
        "capacity_cleanup_required_reverse_level": capacity_level,
        "deep_loss_enabled": True,
        "deep_loss_min_age_days": deep_age,
        "unrealized_pnl_pct": deep_pnl,
        "deep_loss_required_reverse_level": deep_level,
    }


def stage_1_variants() -> list[dict[str, Any]]:
    return [
        _variant(capacity_age=age, capacity_level=level)
        for age in CAPACITY_AGES
        for level in CAPACITY_LEVELS
    ]


def stage_2_variants(capacity: dict[str, Any]) -> list[dict[str, Any]]:
    age = float(capacity["capacity_cleanup_min_age_days"])
    level = int(capacity["capacity_cleanup_required_reverse_level"])
    return [
        _variant(
            capacity_age=age,
            capacity_level=level,
            deep_age=deep_age,
            deep_pnl=threshold,
            deep_level=deep_level,
        )
        for deep_age in DEEP_AGES
        for threshold in DEEP_THRESHOLDS
        for deep_level in DEEP_LEVELS
    ]


def _successful(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cleanup_cash_burn_ratio_pct") is not None
    ]
    if not output:
        raise ValueError("The completed stage contains no successful candidates.")
    return output


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        edge = float(row["edge_vs_hodl_pct_points"])
        burn = float(row["cleanup_cash_burn_ratio_pct"])
        if any(
            float(other["edge_vs_hodl_pct_points"]) >= edge
            and float(other["cleanup_cash_burn_ratio_pct"]) <= burn
            and (
                float(other["edge_vs_hodl_pct_points"]) > edge
                or float(other["cleanup_cash_burn_ratio_pct"]) < burn
            )
            for other in rows
            if other is not row
        ):
            continue
        output.append(row)
    return output


def _balanced_choice(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = _successful(rows)
    frontier = _pareto(candidates) or candidates
    edges = [float(row["edge_vs_hodl_pct_points"]) for row in frontier]
    burns = [float(row["cleanup_cash_burn_ratio_pct"]) for row in frontier]
    edge_span = max(edges) - min(edges) or 1.0
    burn_span = max(burns) - min(burns) or 1.0
    return min(
        frontier,
        key=lambda row: (
            math.hypot(
                (max(edges) - float(row["edge_vs_hodl_pct_points"])) / edge_span,
                (float(row["cleanup_cash_burn_ratio_pct"]) - min(burns)) / burn_span,
            ),
            int(row.get("unique_capacity_blocked_opportunities") or 0),
            float(row.get("oldest_open_days") or 0.0),
            str(row.get("name") or ""),
        ),
    )


def _select_diverse(
    rows: Iterable[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    count: int,
) -> list[dict[str, Any]]:
    candidates = _successful(rows)
    ordered: list[dict[str, Any]] = [_balanced_choice(candidates)]
    for field in fields:
        for value in sorted({row[field] for row in candidates}):
            ordered.append(
                _balanced_choice(row for row in candidates if row[field] == value)
            )
    ordered.extend(
        sorted(
            _pareto(candidates),
            key=lambda row: (
                -float(row["edge_vs_hodl_pct_points"]),
                float(row["cleanup_cash_burn_ratio_pct"]),
            ),
        )
    )
    ordered.extend(
        sorted(
            candidates,
            key=lambda row: (
                -float(row["edge_vs_hodl_pct_points"]),
                float(row["cleanup_cash_burn_ratio_pct"]),
            ),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        name = str(row["name"])
        if name in seen:
            continue
        seen.add(name)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct candidates available; expected {count}.")


def _parameter_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["capacity_cleanup_min_age_days"]),
        int(row["capacity_cleanup_required_reverse_level"]),
        float(row["deep_loss_min_age_days"]),
        float(row["unrealized_pnl_pct"]),
        int(row["deep_loss_required_reverse_level"]),
    )


def stage_3_variants(
    stage_1_rows: Iterable[dict[str, Any]],
    stage_2_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_1 = _successful(stage_1_rows)
    stage_2 = _successful(stage_2_rows)
    capacities = _select_diverse(
        stage_1,
        fields=("capacity_cleanup_required_reverse_level", "capacity_cleanup_min_age_days"),
        count=5,
    )
    deep_losses = _select_diverse(
        stage_2,
        fields=(
            "deep_loss_required_reverse_level",
            "unrealized_pnl_pct",
            "deep_loss_min_age_days",
        ),
        count=10,
    )
    already_evaluated = {
        _parameter_key(row)
        for row in stage_1 + stage_2
    }
    variants: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for capacity in capacities:
        for deep_loss in deep_losses:
            variant = _variant(
                capacity_age=float(capacity["capacity_cleanup_min_age_days"]),
                capacity_level=int(capacity["capacity_cleanup_required_reverse_level"]),
                deep_age=float(deep_loss["deep_loss_min_age_days"]),
                deep_pnl=float(deep_loss["unrealized_pnl_pct"]),
                deep_level=int(deep_loss["deep_loss_required_reverse_level"]),
            )
            key = _parameter_key(variant)
            if key in already_evaluated or key in seen:
                continue
            seen.add(key)
            variants.append(variant)
    if not 30 <= len(variants) <= 50:
        raise ValueError(
            f"Stage 3 produced {len(variants)} new candidates; expected between 30 and 50."
        )
    return variants, capacities, deep_losses


def _write_config(
    *,
    stage: str,
    root: Path,
    variants: list[dict[str, Any]],
    workers: int,
) -> Path:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    path = SWEEP_DIR / f"{stage}.yaml"
    payload = {
        "spot_sweep": {
            "name": f"capacity-cleanup-policy-{stage}",
            "base_config": str(BASE_CONFIG),
            "output_dir": str(root / stage),
            "start": START,
            "end": END,
            "market_data_workers": 7,
            "replay_parallel": True,
            "replay_max_workers": workers,
            "resume": True,
            "continue_on_error": True,
            "artifact_mode": "compact",
            "variants": variants,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _load_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    path = root / stage / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed stage: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _run(path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "backtest.spot_sweep", "--config", str(path)],
        cwd=REPO_ROOT,
        check=True,
    )


def _write_selection(root: Path, payload: dict[str, Any]) -> None:
    path = root / "staged_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_all(root: Path, workers: int) -> None:
    started_at = datetime.now(UTC).isoformat()
    stage_1_path = _write_config(
        stage="stage-1-capacity",
        root=root,
        variants=stage_1_variants(),
        workers=workers,
    )
    print(f"Starting Stage 1: {stage_1_path}", flush=True)
    _run(stage_1_path)
    stage_1_rows = _load_rows(root, "stage-1-capacity")
    capacity_winner = _balanced_choice(stage_1_rows)

    stage_2_path = _write_config(
        stage="stage-2-deep-loss",
        root=root,
        variants=stage_2_variants(capacity_winner),
        workers=workers,
    )
    print(
        "Starting Stage 2 with capacity "
        f"{capacity_winner['capacity_cleanup_min_age_days']:g}d/"
        f"L{capacity_winner['capacity_cleanup_required_reverse_level']}",
        flush=True,
    )
    _run(stage_2_path)
    stage_2_rows = _load_rows(root, "stage-2-deep-loss")

    variants, capacities, deep_losses = stage_3_variants(stage_1_rows, stage_2_rows)
    stage_3_path = _write_config(
        stage="stage-3-interaction",
        root=root,
        variants=variants,
        workers=workers,
    )
    _write_selection(
        root,
        {
            "started_at": started_at,
            "stage_1_capacity_winner": capacity_winner,
            "stage_3_capacity_regions": capacities,
            "stage_3_deep_loss_regions": deep_losses,
            "stage_3_candidates": len(variants),
        },
    )
    print(f"Starting Stage 3 with {len(variants)} interaction candidates", flush=True)
    _run(stage_3_path)
    print("All staged sweeps completed.", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-stage-1", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.action == "prepare-stage-1":
        path = _write_config(
            stage="stage-1-capacity",
            root=root,
            variants=stage_1_variants(),
            workers=max(1, args.workers),
        )
        print(path)
        return 0
    run_all(root, max(1, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
