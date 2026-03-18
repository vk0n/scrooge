import backtest.dataset as data_module
import itertools
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from backtest.dataset import build_dataset
from backtest.stats import compute_stats
from binance.client import Client
from core.engine import run_strategy
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAM_GRID_PATH = PROJECT_ROOT / "config" / "param_grid.yaml"
RESULTS_PATH = PROJECT_ROOT / "optimization_results.csv"
BEST_PARAMS_PATH = PROJECT_ROOT / "best_params.yaml"

CONFIG_PATH = Path(
    os.getenv(
        "SCROOGE_BACKTEST_CONFIG_PATH",
        os.getenv("SCROOGE_CONFIG_PATH", str(PROJECT_ROOT / "config" / "backtest.yaml")),
    )
).expanduser()


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {CONFIG_PATH}")
    if cfg.get("live") is True:
        raise ValueError(f"Optimize requires backtest config (live must be false): {CONFIG_PATH}")
    return cfg

def load_params():
    with PARAM_GRID_PATH.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    data_module.set_client(Client(api_key, api_secret))

    cfg = load_config()
    symbol = cfg["symbol"]
    leverage = cfg["leverage"]
    initial_balance = cfg["initial_balance"]

    df = build_dataset(
        symbol=symbol,
        intervals=cfg["intervals"],
        backtest_period_days=cfg["backtest_period_days"],
        backtest_period_start_time=cfg.get("backtest_period_start_time", ""),
        backtest_period_end_time=cfg["backtest_period_end_time"],
    )

    param_grid = load_params()
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    results = []

    print(f"Starting optimization with {len(combinations)} combinations...\n")

    for combo in tqdm(combinations, desc="Optimizing", ncols=100):
        params = dict(zip(keys, combo))
        final_balance, trades, balance_history, state = run_strategy(  # noqa: F841
            df,
            False,
            initial_balance,
            leverage=leverage,
            enable_logs=False,
            use_state=False,
            **params,
        )

        metrics = compute_stats(initial_balance, final_balance, trades, balance_history)
        profit_factor = metrics.get("Profit Factor", 0)
        score = profit_factor * (final_balance / initial_balance)

        results.append(
            {
                **params,
                **metrics,
                "Score": score,
            }
        )

    df_results = pd.DataFrame(results).sort_values("Score", ascending=False)
    df_results.to_csv(RESULTS_PATH, index=False)

    best_params = df_results.iloc[0][keys].to_dict()
    best_metrics = df_results.iloc[0].to_dict()
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file_obj:
        yaml.dump(
            {
                "best_params": best_params,
                "best_metrics": best_metrics,
            },
            file_obj,
            sort_keys=False,
        )

    print("\nOptimization completed!")
    print(f"Best Score: {best_metrics['Score']:.4f}")
    print(f"Saved full results to {RESULTS_PATH}")
    print(f"Saved best parameters to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
