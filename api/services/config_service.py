from __future__ import annotations

import copy
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONFIG_PATH = Path(os.getenv("SCROOGE_CONFIG_PATH", str(_project_root() / "config" / "live.yaml"))).expanduser()

EDITABLE_TOP_LEVEL_KEYS = ("live", "symbol", "leverage", "initial_balance", "use_full_balance", "qty")
EDITABLE_INTERVAL_KEYS = ("small", "medium", "big")
EDITABLE_PARAM_KEYS = (
    "sl_mult",
    "tp_mult",
    "trail_atr_mult",
    "rsi_extreme_long",
    "rsi_extreme_short",
    "rsi_long_open_threshold",
    "rsi_long_qty_threshold",
    "rsi_long_tp_threshold",
    "rsi_long_close_threshold",
    "rsi_short_open_threshold",
    "rsi_short_qty_threshold",
    "rsi_short_tp_threshold",
    "rsi_short_close_threshold",
)


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
            raw_text = file_obj.read()
    except OSError as exc:
        raise OSError(f"Failed to read config file: {CONFIG_PATH}") from exc

    return parse_config_text(raw_text)


def load_raw_config_text() -> str:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
            return file_obj.read()
    except OSError as exc:
        raise OSError(f"Failed to read config file: {CONFIG_PATH}") from exc


def parse_config_text(raw_text: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config YAML: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    return raw


def extract_editable_config(config: dict[str, Any]) -> dict[str, Any]:
    params = config.get("params")
    params_map = params if isinstance(params, dict) else {}
    intervals = config.get("intervals")
    interval_map = intervals if isinstance(intervals, dict) else {}

    return {
        "live": config.get("live"),
        "symbol": config.get("symbol"),
        "leverage": config.get("leverage"),
        "initial_balance": config.get("initial_balance"),
        "use_full_balance": config.get("use_full_balance"),
        "qty": config.get("qty"),
        "intervals": {key: interval_map.get(key) for key in EDITABLE_INTERVAL_KEYS},
        "params": {key: params_map.get(key) for key in EDITABLE_PARAM_KEYS},
    }


def _create_backup() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.bak.{timestamp}")
    try:
        shutil.copy2(CONFIG_PATH, backup_path)
    except OSError as exc:
        raise OSError(f"Failed to create config backup at: {backup_path}") from exc
    return backup_path


def _write_config(config: dict[str, Any]) -> None:
    _write_raw_config_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False))


def _write_raw_config_text(raw_text: str) -> None:
    normalized = raw_text.rstrip() + "\n"
    temp_path = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            file_obj.write(normalized)
    except OSError as exc:
        raise OSError(f"Failed to write temporary config file: {temp_path}") from exc
    try:
        temp_path.replace(CONFIG_PATH)
        return
    except OSError:
        # Fallback for bind-mounted single files where atomic replace may fail.
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as file_obj:
                file_obj.write(normalized)
            return
        except OSError as exc:
            raise OSError(f"Failed to write config file: {CONFIG_PATH}") from exc
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def update_editable_config(patch: dict[str, Any]) -> dict[str, Any]:
    allowed_top = set(EDITABLE_TOP_LEVEL_KEYS) | {"intervals", "params"}
    unknown_top = set(patch) - allowed_top
    if unknown_top:
        unknown_joined = ", ".join(sorted(unknown_top))
        raise ValueError(f"Unsupported editable field(s): {unknown_joined}")

    intervals_patch = patch.get("intervals")
    if intervals_patch is not None:
        if not isinstance(intervals_patch, dict):
            raise ValueError("intervals must be an object")
        unknown_intervals = set(intervals_patch) - set(EDITABLE_INTERVAL_KEYS)
        if unknown_intervals:
            unknown_joined = ", ".join(sorted(unknown_intervals))
            raise ValueError(f"Unsupported editable intervals field(s): {unknown_joined}")

    params_patch = patch.get("params")
    if params_patch is not None:
        if not isinstance(params_patch, dict):
            raise ValueError("params must be an object")
        unknown_params = set(params_patch) - set(EDITABLE_PARAM_KEYS)
        if unknown_params:
            unknown_joined = ", ".join(sorted(unknown_params))
            raise ValueError(f"Unsupported editable params field(s): {unknown_joined}")

    current = load_config()
    updated = copy.deepcopy(current)
    changed_fields: list[str] = []

    for key in EDITABLE_TOP_LEVEL_KEYS:
        if key not in patch:
            continue
        new_value = patch[key]
        if key != "qty" and new_value is None:
            raise ValueError(f"{key} cannot be null")
        if updated.get(key) != new_value:
            updated[key] = new_value
            changed_fields.append(key)

    if intervals_patch is not None:
        current_intervals = updated.get("intervals")
        if not isinstance(current_intervals, dict):
            current_intervals = {}
            updated["intervals"] = current_intervals
        for key, value in intervals_patch.items():
            if value is None:
                raise ValueError(f"intervals.{key} cannot be null")
            if current_intervals.get(key) != value:
                current_intervals[key] = value
                changed_fields.append(f"intervals.{key}")

    if params_patch is not None:
        current_params = updated.get("params")
        if not isinstance(current_params, dict):
            current_params = {}
            updated["params"] = current_params
        for key, value in params_patch.items():
            if value is None:
                raise ValueError(f"params.{key} cannot be null")
            if current_params.get(key) != value:
                current_params[key] = value
                changed_fields.append(f"params.{key}")

    if not changed_fields:
        return {
            "updated": False,
            "restart_required": False,
            "changed_fields": [],
            "backup_path": None,
            "editable": extract_editable_config(current),
        }

    backup_path = _create_backup()
    _write_config(updated)

    return {
        "updated": True,
        "restart_required": True,
        "changed_fields": changed_fields,
        "backup_path": str(backup_path),
        "editable": extract_editable_config(updated),
    }


def update_raw_config_text(raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text cannot be empty")

    current_text = load_raw_config_text()
    normalized_current = current_text.rstrip() + "\n"
    normalized_next = raw_text.rstrip() + "\n"
    parsed_next = parse_config_text(normalized_next)

    if normalized_next == normalized_current:
        return {
            "updated": False,
            "restart_required": False,
            "changed_fields": [],
            "backup_path": None,
            "editable": extract_editable_config(parse_config_text(normalized_current)),
            "raw_text": normalized_current,
        }

    backup_path = _create_backup()
    _write_raw_config_text(normalized_next)
    return {
        "updated": True,
        "restart_required": True,
        "changed_fields": [],
        "backup_path": str(backup_path),
        "editable": extract_editable_config(parsed_next),
        "raw_text": normalized_next,
    }
