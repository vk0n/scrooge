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


CONFIG_PATH = Path(os.getenv("SCROOGE_CONFIG_PATH", str(_project_root() / "config.yaml"))).expanduser()

EDITABLE_TOP_LEVEL_KEYS = ("symbol", "leverage", "use_full_balance", "qty")
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
            raw = yaml.safe_load(file_obj)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config YAML: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file: {CONFIG_PATH}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    return raw


def extract_editable_config(config: dict[str, Any]) -> dict[str, Any]:
    params = config.get("params")
    params_map = params if isinstance(params, dict) else {}

    return {
        "symbol": config.get("symbol"),
        "leverage": config.get("leverage"),
        "use_full_balance": config.get("use_full_balance"),
        "qty": config.get("qty"),
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
    temp_path = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=False)
    except OSError as exc:
        raise OSError(f"Failed to write temporary config file: {temp_path}") from exc
    try:
        temp_path.replace(CONFIG_PATH)
        return
    except OSError:
        # Fallback for bind-mounted single files where atomic replace may fail.
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as file_obj:
                yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=False)
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
    allowed_top = set(EDITABLE_TOP_LEVEL_KEYS) | {"params"}
    unknown_top = set(patch) - allowed_top
    if unknown_top:
        unknown_joined = ", ".join(sorted(unknown_top))
        raise ValueError(f"Unsupported editable field(s): {unknown_joined}")

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
