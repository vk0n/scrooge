#!/bin/sh
set -eu

run_ts="$(date -u +%Y%m%dT%H%M%SZ)"
root_dir="${SCROOGE_BACKTEST_ROOT:-/runtime/backtests}"
run_dir="${root_dir}/${run_ts}"
latest_link="${root_dir}/latest"

mkdir -p "${run_dir}"
mkdir -p "${root_dir}"
ln -sfn "${run_dir}" "${latest_link}"

export SCROOGE_STATE_FILE="${run_dir}/state.json"
export SCROOGE_TRADE_HISTORY_FILE="${run_dir}/trade_history.jsonl"
export SCROOGE_BALANCE_HISTORY_FILE="${run_dir}/balance_history.jsonl"
export SCROOGE_LOG_FILE="${run_dir}/trading_log.txt"
export SCROOGE_RUNTIME_CHART_DATASET_PATH="${run_dir}/chart_dataset.csv"
export SCROOGE_CONFIG_PATH="${SCROOGE_CONFIG_PATH:-/runtime/config.backtest.yaml}"

export_settings="$(
python - <<'PY'
import os
from pathlib import Path

import yaml

config_path = Path(os.getenv("SCROOGE_CONFIG_PATH", "/runtime/config.backtest.yaml")).expanduser()
enabled = False
host_dir = "/host_artifacts"

try:
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
except Exception:
    config = {}

raw_enabled = config.get("export_artifacts_to_host", False)
if isinstance(raw_enabled, str):
    enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
else:
    enabled = bool(raw_enabled)

raw_host_dir = config.get("host_artifacts_dir")
if isinstance(raw_host_dir, str) and raw_host_dir.strip():
    host_dir = raw_host_dir.strip()

print(("1" if enabled else "0") + "|" + host_dir)
PY
)"
export_enabled="${export_settings%%|*}"
export_dir="${export_settings#*|}"

echo "[BACKTEST] Run directory: ${run_dir}"
echo "[BACKTEST] Config path: ${SCROOGE_CONFIG_PATH}"
echo "[BACKTEST] Export to host: ${export_enabled} (target: ${export_dir})"

cd /runtime
if python /app/main.py; then
  exit_code="0"
else
  exit_code="$?"
fi

if [ "${export_enabled}" = "1" ]; then
  host_root="${export_dir%/}"
  host_run_dir="${host_root}/${run_ts}"
  host_latest_link="${host_root}/latest"
  mkdir -p "${host_root}"
  mkdir -p "${host_run_dir}"
  cp -a "${run_dir}/." "${host_run_dir}/"
  ln -sfn "${host_run_dir}" "${host_latest_link}"
  echo "[BACKTEST] Exported artifacts to host: ${host_run_dir}"
fi

exit "${exit_code}"
