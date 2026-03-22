from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


INDICATOR_INPUT_MODES = {"closed", "intrabar"}
INDICATOR_INPUT_KEYS = ("ema", "rsi", "bb", "atr")
INDICATOR_COLUMNS = ("EMA", "RSI", "BBL", "BBM", "BBU", "ATR")
INDICATOR_GROUP_COLUMNS = {
    "ema": ("EMA",),
    "rsi": ("RSI",),
    "bb": ("BBL", "BBM", "BBU"),
    "atr": ("ATR",),
}
INDICATOR_KEY_BY_COLUMN = {
    column: key
    for key, columns in INDICATOR_GROUP_COLUMNS.items()
    for column in columns
}
LEGACY_INDICATOR_GROUP_ALIASES = {
    "bbl": "bb",
    "bbm": "bb",
    "bbu": "bb",
}
LEGACY_INDICATOR_MODE_ALIASES = {
    "discrete": "closed",
    "realtime": "intrabar",
}


@dataclass(frozen=True, slots=True)
class IndicatorSelectionPlan:
    column_sources: tuple[tuple[str, str], ...]
    realtime_keys: tuple[str, ...]
    requires_discrete: bool
    requires_realtime: bool


def _default_indicator_input_mode(strategy_mode: str | None) -> str:
    normalized = str(strategy_mode or "discrete").strip().lower() or "discrete"
    if normalized == "realtime":
        return "intrabar"
    return "closed"


def normalize_indicator_inputs(
    raw_value: Any,
    *,
    strategy_mode: str | None = None,
) -> dict[str, str]:
    default_mode = _default_indicator_input_mode(strategy_mode)
    normalized = {key: default_mode for key in INDICATOR_INPUT_KEYS}

    if raw_value is None:
        return normalized
    if not isinstance(raw_value, dict):
        raise ValueError("indicator_inputs must be an object mapping indicator names to closed/intrabar.")

    supported_keys = set(INDICATOR_INPUT_KEYS) | set(LEGACY_INDICATOR_GROUP_ALIASES)
    unknown_keys = sorted(set(raw_value) - supported_keys)
    if unknown_keys:
        unknown_joined = ", ".join(unknown_keys)
        raise ValueError(
            "indicator_inputs contains unsupported key(s): "
            f"{unknown_joined}. Allowed keys: {', '.join(INDICATOR_INPUT_KEYS)}"
        )

    seen_modes_by_key: dict[str, str] = {}
    for key, value in raw_value.items():
        canonical_key = LEGACY_INDICATOR_GROUP_ALIASES.get(key, key)
        mode = LEGACY_INDICATOR_MODE_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
        if mode not in INDICATOR_INPUT_MODES:
            raise ValueError(
                f"indicator_inputs.{canonical_key} must be one of: {', '.join(sorted(INDICATOR_INPUT_MODES))}"
            )
        previous_mode = seen_modes_by_key.get(canonical_key)
        if previous_mode is not None and previous_mode != mode:
            raise ValueError(
                f"indicator_inputs.{canonical_key} has conflicting values; use a single mode for the full indicator group."
            )
        normalized[canonical_key] = mode
        seen_modes_by_key[canonical_key] = mode

    return normalized


def indicator_input_mode_for_column(indicator_inputs: dict[str, str] | None, column: str) -> str:
    key = INDICATOR_KEY_BY_COLUMN[column]
    if indicator_inputs is None:
        return "closed"
    mode = str(indicator_inputs.get(key, "closed")).strip().lower() or "closed"
    return LEGACY_INDICATOR_MODE_ALIASES.get(mode, mode)


def indicator_selection_plan(indicator_inputs: dict[str, str] | None) -> IndicatorSelectionPlan:
    if indicator_inputs is None:
        mode_key = tuple("closed" for _ in INDICATOR_INPUT_KEYS)
    else:
        mode_key = tuple(
            LEGACY_INDICATOR_MODE_ALIASES.get(
                str(indicator_inputs.get(key, "closed")).strip().lower() or "closed",
                str(indicator_inputs.get(key, "closed")).strip().lower() or "closed",
            )
            for key in INDICATOR_INPUT_KEYS
        )
    return _cached_indicator_selection_plan(mode_key)


@lru_cache(maxsize=32)
def _cached_indicator_selection_plan(mode_key: tuple[str, ...]) -> IndicatorSelectionPlan:
    mode_by_key = dict(zip(INDICATOR_INPUT_KEYS, mode_key, strict=False))
    realtime_keys = tuple(
        key
        for key in INDICATOR_INPUT_KEYS
        if mode_by_key.get(key, "closed") == "intrabar"
    )
    column_sources = tuple(
        (column, mode_by_key.get(INDICATOR_KEY_BY_COLUMN[column], "closed"))
        for column in INDICATOR_COLUMNS
    )
    return IndicatorSelectionPlan(
        column_sources=column_sources,
        realtime_keys=realtime_keys,
        requires_discrete=any(mode == "closed" for _, mode in column_sources),
        requires_realtime=any(mode == "intrabar" for _, mode in column_sources),
    )


def uses_realtime_indicator_inputs(indicator_inputs: dict[str, str] | None) -> bool:
    return indicator_selection_plan(indicator_inputs).requires_realtime


def merge_indicator_decision_values(
    *,
    indicator_inputs: dict[str, str],
    discrete_values: dict[str, float | None] | None,
    realtime_values: dict[str, float | None] | None,
    selection_plan: IndicatorSelectionPlan | None = None,
) -> dict[str, float | None] | None:
    plan = selection_plan or indicator_selection_plan(indicator_inputs)
    if plan.requires_discrete and discrete_values is None:
        return None
    if plan.requires_realtime and realtime_values is None:
        return None
    selected: dict[str, float | None] = {}
    for column, mode in plan.column_sources:
        source = realtime_values if mode == "intrabar" else discrete_values
        if source is None:
            return None
        selected[column] = source.get(column)

    if any(value is None for value in selected.values()):
        return None
    return selected
