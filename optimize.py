import itertools
import os
import pandas as pd
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
from binance.client import Client
from strategy import run_strategy
from report import compute_stats
from data import build_dataset
import data as data_module

CONFIG_PATH = "config.yaml"

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
data_module.set_client(Client(api_key, api_secret))

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
df = build_dataset(
    symbol=symbol,
    intervals=cfg["intervals"],
    backtest_period_days=cfg["backtest_period_days"],
    backtest_period_end_time=cfg["backtest_period_end_time"]
)

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
print("Best parameters and metrics saved to best_params.yaml")
