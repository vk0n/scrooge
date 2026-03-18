from __future__ import annotations

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


def uses_realtime_indicator_inputs(indicator_inputs: dict[str, str] | None) -> bool:
    if indicator_inputs is None:
        return False
    return any(LEGACY_INDICATOR_MODE_ALIASES.get(mode, mode) == "intrabar" for mode in indicator_inputs.values())


def merge_indicator_decision_values(
    *,
    indicator_inputs: dict[str, str],
    discrete_values: dict[str, float | None] | None,
    realtime_values: dict[str, float | None] | None,
) -> dict[str, float | None] | None:
    selected: dict[str, float | None] = {}
    for column in INDICATOR_COLUMNS:
        mode = indicator_input_mode_for_column(indicator_inputs, column)
        if mode == "intrabar":
            if realtime_values is None:
                return None
            selected[column] = realtime_values.get(column)
        else:
            if discrete_values is None:
                return None
            selected[column] = discrete_values.get(column)

    if any(value is None for value in selected.values()):
        return None
    return selected
