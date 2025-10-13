import itertools
import pandas as pd
import yaml
from tqdm import tqdm
from strategy import *
from report import *
from data import *
from datetime import datetime, timedelta

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

symbol = cfg["symbol"]
lvrg = cfg["leverage"]
initial_balance = cfg["initial_balance"]

def load_params():
    with open("param_grid.yaml", "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# Fetch your dataset
# -----------------------------
df = build_dataset()

# -----------------------------
# Define parameter grid
# -----------------------------
param_grid = load_params()

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
