import itertools
import pandas as pd
import yaml
from tqdm import tqdm
from strategy import *
from datetime import datetime, timedelta


symbol = "BTCUSDT"
lvrg = 1
initial_balance = 5000
interval_small = "1m"
interval_medium = "15m"
interval_big = "1h"
backtest_period_days = 365
end_time = datetime.now()
start_time = end_time - timedelta(days=backtest_period_days)

# -----------------------------
# Fetch your dataset
# -----------------------------
df_small = fetch_historical_paginated(symbol, interval_small, start_time=start_time, end_time=end_time)
df_medium = fetch_historical_paginated(symbol, interval_medium, start_time=start_time, end_time=end_time)
df_big = fetch_historical_paginated(symbol, interval_big, start_time=start_time, end_time=end_time)
df = prepare_multi_tf(df_small, df_medium, df_big)

# -----------------------------
# Define parameter grid
# -----------------------------
param_grid = {
    "sl_mult": [2.4],
    "tp_mult": [3.6],
    "rsi_extreme_long": [70, 75],
    "rsi_extreme_short": [25, 30],
    "rsi_long_open_threshold": [30, 40, 50],
    "rsi_long_qty_threshold": [30],
    "rsi_long_close_threshold": [60, 65, 70, 75],
    "rsi_short_open_threshold": [50, 60, 70],
    "rsi_short_qty_threshold": [70],
    "rsi_short_close_threshold": [25, 30, 35, 40],
    "trail_atr_mult": [0.3],
}

keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))

results = []

# -----------------------------
# Run optimization with progress bar
# -----------------------------
print(f"Starting optimization with {len(combinations)} combinations...\n")

for combo in tqdm(combinations, desc="Optimizing", ncols=100):
    params = dict(zip(keys, combo))
    final_balance, trades, balance_history, state = run_strategy(
        df,
        False,
        initial_balance,
        leverage=lvrg,
        enable_logs=False,
        use_state=False,
        **params
    )

    metrics = compute_stats(initial_balance, final_balance, trades, balance_history)
    profit_factor = metrics.get("Profit Factor", 0)
    score = profit_factor * (final_balance / initial_balance)

    results.append({
        **params,
        **metrics,
        "Score": score
    })

# -----------------------------
# Save all results
# -----------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Score", ascending=False)
df_results.to_csv("optimization_results.csv", index=False)

# -----------------------------
# Save best parameters to YAML
# -----------------------------
best_params = df_results.iloc[0][keys].to_dict()
best_metrics = df_results.iloc[0].to_dict()

with open("best_params.yaml", "w") as f:
    yaml.dump({
        "best_params": best_params,
        "best_metrics": best_metrics
    }, f, sort_keys=False)

print("\nOptimization completed!")
print(f"Best Score: {best_metrics['Score']:.4f}")
print(f"Best parameters and metrics saved to best_params.yaml")
