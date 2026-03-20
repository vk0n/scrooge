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


def _resolve_template_scenarios(
    compare_payload: dict[str, Any],
    *,
    scenario_names: list[str] | None,
) -> list[dict[str, Any]]:
    explicit = compare_payload.get("candidate_template")
    if isinstance(explicit, dict):
        return [copy.deepcopy(explicit)]

    scenarios = compare_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("compare template must include scenarios or candidate_template")

    if scenario_names:
        scenario_lookup: dict[str, dict[str, Any]] = {}
        for item in scenarios:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if name:
                scenario_lookup[name] = item
        resolved: list[dict[str, Any]] = []
        for scenario_name in scenario_names:
            if scenario_name not in scenario_lookup:
                raise ValueError(f"scenario_template_name not found: {scenario_name}")
            resolved.append(copy.deepcopy(scenario_lookup[scenario_name]))
        return resolved

    if len(scenarios) != 1:
        raise ValueError(
            "compare template has multiple scenarios; specify --scenario-template-name "
            "or add candidate_template to the YAML"
        )
    only = scenarios[0]
    if not isinstance(only, dict):
        raise ValueError("template scenario must be a mapping")
    return [copy.deepcopy(only)]


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


def _slugify_name(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "candidate"


def generate_compare_config(
    *,
    compare_template_path: Path,
    param_grid_path: Path,
    output_path: Path,
    scenario_template_names: list[str] | None,
    candidate_prefix: str | None,
) -> tuple[Path, int]:
    compare_payload = _load_yaml_mapping(compare_template_path)
    param_grid = _load_param_grid(param_grid_path)
    template_scenarios = _resolve_template_scenarios(
        compare_payload,
        scenario_names=scenario_template_names,
    )

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    scenarios: list[dict[str, Any]] = []
    template_names: list[str] = []
    for template_scenario in template_scenarios:
        template_name = str(template_scenario.get("name", "") or "").strip() or "candidate"
        template_names.append(template_name)
        overrides = template_scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("template scenario overrides must be a mapping")
        template_prefix_root = str(candidate_prefix or "").strip()
        if template_prefix_root:
            template_prefix = f"{template_prefix_root}-{_slugify_name(template_name)}"
        else:
            template_prefix = template_name

        for index, combo in enumerate(combinations, start=1):
            params = dict(zip(keys, combo))
            scenario_overrides = _deep_merge_dicts(overrides, {"params": params})
            scenarios.append(
                {
                    "name": _scenario_name(template_prefix, index),
                    "overrides": scenario_overrides,
                }
            )

    generated = copy.deepcopy(compare_payload)
    generated["generated_from_compare_template"] = str(compare_template_path)
    generated["generated_from_param_grid"] = str(param_grid_path)
    generated["generated_candidate_count"] = len(scenarios)
    generated["generated_template_scenario_names"] = template_names
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
        action="append",
        default=None,
        help=(
            "Scenario name inside the compare template to use as the candidate template when multiple scenarios exist. "
            "Can be passed multiple times or as a comma-separated list."
        ),
    )
    parser.add_argument(
        "--candidate-prefix",
        default=None,
        help="Prefix for generated scenario names.",
    )
    args = parser.parse_args()

    scenario_template_names: list[str] | None = None
    if args.scenario_template_name:
        scenario_template_names = []
        for raw_value in args.scenario_template_name:
            for part in str(raw_value).split(","):
                name = part.strip()
                if name:
                    scenario_template_names.append(name)

    output_path, count = generate_compare_config(
        compare_template_path=Path(args.compare_template).expanduser(),
        param_grid_path=Path(args.param_grid).expanduser(),
        output_path=Path(args.output).expanduser(),
        scenario_template_names=scenario_template_names,
        candidate_prefix=args.candidate_prefix,
    )
    print(f"Generated {count} candidate scenarios at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
