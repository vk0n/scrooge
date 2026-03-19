from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARE_TEMPLATE_PATH = PROJECT_ROOT / "config" / "compare.sieves.yaml"
DEFAULT_PARAM_GRID_PATH = PROJECT_ROOT / "config" / "param_grid.yaml"
DEFAULT_COMPARE_OUTPUT_PATH = PROJECT_ROOT / "config" / "compare.generated.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_template_scenario(
    compare_payload: dict[str, Any],
    *,
    scenario_name: str | None,
) -> dict[str, Any]:
    explicit = compare_payload.get("candidate_template")
    if isinstance(explicit, dict):
        return copy.deepcopy(explicit)

    scenarios = compare_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("compare template must include scenarios or candidate_template")

    if scenario_name:
        for item in scenarios:
            if isinstance(item, dict) and str(item.get("name", "") or "").strip() == scenario_name:
                return copy.deepcopy(item)
        raise ValueError(f"scenario_template_name not found: {scenario_name}")

    if len(scenarios) != 1:
        raise ValueError(
            "compare template has multiple scenarios; specify --scenario-template-name "
            "or add candidate_template to the YAML"
        )
    only = scenarios[0]
    if not isinstance(only, dict):
        raise ValueError("template scenario must be a mapping")
    return copy.deepcopy(only)


def _load_param_grid(path: Path) -> dict[str, list[Any]]:
    payload = _load_yaml_mapping(path)
    grid: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            if not value:
                raise ValueError(f"param_grid entry {key} must not be empty")
            grid[str(key)] = list(value)
        else:
            grid[str(key)] = [value]
    if not grid:
        raise ValueError("param_grid must not be empty")
    return grid


def _scenario_name(prefix: str, index: int) -> str:
    return f"{prefix}-{index:04d}"


def generate_compare_config(
    *,
    compare_template_path: Path,
    param_grid_path: Path,
    output_path: Path,
    scenario_template_name: str | None,
    candidate_prefix: str | None,
) -> tuple[Path, int]:
    compare_payload = _load_yaml_mapping(compare_template_path)
    param_grid = _load_param_grid(param_grid_path)
    template_scenario = _resolve_template_scenario(
        compare_payload,
        scenario_name=scenario_template_name,
    )

    template_name = str(template_scenario.get("name", "") or "").strip() or "candidate"
    overrides = template_scenario.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("template scenario overrides must be a mapping")
    prefix = str(candidate_prefix or template_name).strip() or "candidate"

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    scenarios: list[dict[str, Any]] = []
    for index, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))
        scenario_overrides = _deep_merge_dicts(overrides, {"params": params})
        scenarios.append(
            {
                "name": _scenario_name(prefix, index),
                "overrides": scenario_overrides,
            }
        )

    generated = copy.deepcopy(compare_payload)
    generated["generated_from_compare_template"] = str(compare_template_path)
    generated["generated_from_param_grid"] = str(param_grid_path)
    generated["generated_candidate_count"] = len(scenarios)
    generated["generated_template_scenario_name"] = template_name
    generated["scenarios"] = scenarios
    generated.pop("candidate_template", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(generated, file_obj, sort_keys=False, allow_unicode=False)
    return output_path, len(scenarios)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compare YAML with many candidate scenarios from param_grid.")
    parser.add_argument(
        "--compare-template",
        default=str(DEFAULT_COMPARE_TEMPLATE_PATH),
        help="Base compare YAML to clone top-level settings from.",
    )
    parser.add_argument(
        "--param-grid",
        default=str(DEFAULT_PARAM_GRID_PATH),
        help="YAML file with params grid.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_COMPARE_OUTPUT_PATH),
        help="Where to write the generated compare YAML.",
    )
    parser.add_argument(
        "--scenario-template-name",
        default=None,
        help="Scenario name inside the compare template to use as the candidate template when multiple scenarios exist.",
    )
    parser.add_argument(
        "--candidate-prefix",
        default=None,
        help="Prefix for generated scenario names.",
    )
    args = parser.parse_args()

    output_path, count = generate_compare_config(
        compare_template_path=Path(args.compare_template).expanduser(),
        param_grid_path=Path(args.param_grid).expanduser(),
        output_path=Path(args.output).expanduser(),
        scenario_template_name=args.scenario_template_name,
        candidate_prefix=args.candidate_prefix,
    )
    print(f"Generated {count} candidate scenarios at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
