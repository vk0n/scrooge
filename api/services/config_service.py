from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONFIG_PATH = Path(os.getenv("SCROOGE_CONFIG_PATH", str(_project_root() / "config.yaml"))).expanduser()


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
