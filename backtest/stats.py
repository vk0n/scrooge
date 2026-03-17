from __future__ import annotations

import numpy as np


def compute_stats(initial_balance, final_balance, trades, balance_history):
    """Compute basic performance metrics without pulling plotting dependencies."""
    stats = {}

    stats["Initial Balance"] = initial_balance
    stats["Final Balance"] = final_balance
    stats["Total Return %"] = (final_balance - initial_balance) / initial_balance * 100

    n_trades = len(trades)
    stats["Number of Trades"] = n_trades

    if n_trades > 0:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]

        stats["Win Rate %"] = len(wins) / n_trades * 100
        stats["Average PnL"] = trades["net_pnl"].mean()
        stats["Average Profit"] = wins["net_pnl"].mean() if len(wins) > 0 else 0
        stats["Average Loss"] = losses["net_pnl"].mean() if len(losses) > 0 else 0
        stats["Best Trade"] = trades["net_pnl"].max()
        stats["Worst Trade"] = trades["net_pnl"].min()
        stats["Total Fee"] = trades["fee"].sum()

        total_profit = wins["net_pnl"].sum()
        total_loss = abs(losses["net_pnl"].sum())
        stats["Profit Factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else np.inf
    else:
        stats["Win Rate %"] = 0
        stats["Average PnL"] = 0
        stats["Best Trade"] = 0
        stats["Worst Trade"] = 0
        stats["Profit Factor"] = 0

    equity = np.array(balance_history)
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    stats["Max Drawdown %"] = drawdowns.min() * 100

    return stats
