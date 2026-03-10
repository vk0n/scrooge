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

echo "[BACKTEST] Run directory: ${run_dir}"
echo "[BACKTEST] Config path: ${SCROOGE_CONFIG_PATH}"

cd /runtime
exec python /app/main.py
