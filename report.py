import json
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from data import fetch_historical_paginated, prepare_multi_tf
from strategy import run_strategy

STATE_FILE = "state.json"
_client = None
_rw_df = None
_rw_start_time = None
_rw_k_days = None
_rw_start_balance = None
_rw_strategy_kwargs = None


def set_client(client):
    global _client
    _client = client


def _get_client():
    if _client is None:
        raise ValueError("Binance client not initialized. Call set_client().")
    return _client


def _rw_init(df, start_time, k_days, start_balance, strategy_kwargs):
    global _rw_df, _rw_start_time, _rw_k_days, _rw_start_balance, _rw_strategy_kwargs
    _rw_df = df
    _rw_start_time = start_time
    _rw_k_days = k_days
    _rw_start_balance = start_balance
    _rw_strategy_kwargs = strategy_kwargs


def _rw_worker(offset):
    window_start = _rw_start_time + pd.Timedelta(days=offset)
    window_end = window_start + pd.Timedelta(days=_rw_k_days)
    df_window = _rw_df[(_rw_df["open_time"] >= window_start) & (_rw_df["open_time"] < window_end)]
    if df_window.empty:
        return None

    final_balance, trades, balance_history, _ = run_strategy(
        df_window,
        live=False,
        initial_balance=_rw_start_balance,
        use_state=False,
        enable_logs=False,
        show_progress=False,
        **_rw_strategy_kwargs
    )
    pnl_pct = (final_balance - _rw_start_balance) / _rw_start_balance * 100

    if trades is not None and len(trades) > 0:
        wins = trades[trades["net_pnl"] > 0]
        win_rate = len(wins) / len(trades) * 100
    else:
        win_rate = 0

    equity = np.array(balance_history)
    if len(equity) > 0:
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
    else:
        max_drawdown = 0

    return final_balance, pnl_pct, win_rate, max_drawdown

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

def fetch_session_klines(symbol, interval, start_ts, end_ts):
    """Fetch historical klines from Binance for session period."""
    client = _get_client()
    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        endTime=end_ts,
        limit=1500
    )
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]

def compute_stats(initial_balance, final_balance, trades, balance_history):
    """Compute basic performance metrics."""
    stats = {}

    # Basic
    stats["Initial Balance"] = initial_balance
    stats["Final Balance"] = final_balance
    stats["Total Return %"] = (final_balance - initial_balance) / initial_balance * 100

    # Trades
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

    # Max drawdown
    equity = np.array(balance_history)
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    stats["Max Drawdown %"] = drawdowns.min() * 100

    return stats


def plot_results(df, trades, balance_history):
    """Plot price with Bollinger Bands, RSI and Equity Curve."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'height_ratios':[3,1,1]})

    # Price + Bollinger Bands
    ax1.plot(df["open_time"], df["close"], label="Price", color="blue")
    ax1.plot(df["open_time"], df["BBL"], label="Lower BB", color="red", linestyle="--")
    ax1.plot(df["open_time"], df["BBM"], label="Middle BB", color="black", linestyle="--")
    ax1.plot(df["open_time"], df["BBU"], label="Upper BB", color="green", linestyle="--")
    ax1.plot(df["open_time"], df["EMA"], label="EMA", color="purple", alpha=0.5)

    # Plot trades
    for _, trade in trades.iterrows():
        if pd.notna(trade["exit"]):
            if trade["exit_reason"] in ["stop_loss", "liquidation"]:
                color = "red"
            elif trade["exit_reason"] in ["take_profit", "rsi"]:
                color = "green"
            else:
                color = "orange"

            ax1.scatter(trade["time"], trade["entry"], marker="^" if trade["side"]=="long" else "v",
                        color="blue", s=80)
            ax1.scatter(trade["time"], trade["exit"], marker="x", color=color, s=80)

    ax1.set_title("Bollinger Bands Strategy Backtest (with Dynamic SL and Fees)")
    ax1.legend()

    # RSI subplot
    ax2.plot(df["open_time"], df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax2.set_title("RSI")
    ax2.legend()

    # Equity Curve
    ax3.plot(df["open_time"][:len(balance_history)], balance_history, color="purple", label="Equity Curve")
    ax3.set_title("Equity Curve")
    ax3.legend()

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, dpi=150)
        webbrowser.get("firefox").open(tmpfile.name)


def compute_session_stats(trades, balance_history):
    """Compute basic session performance metrics."""
    stats = {}
    stats["Number of Trades"] = len(trades)
    if len(trades) > 0:
        wins = [t for t in trades if t.get("net_pnl", 0) > 0]
        losses = [t for t in trades if t.get("net_pnl", 0) < 0]
        stats["Win Rate %"] = len(wins) / len(trades) * 100
        stats["Average PnL"] = np.mean([t.get("net_pnl", 0) for t in trades])
        stats["Average Profit"] = np.mean([t.get("net_pnl", 0) for t in wins]) if wins else 0
        stats["Average Loss"] = np.mean([t.get("net_pnl", 0) for t in losses]) if losses else 0
        stats["Best Trade"] = max([t.get("net_pnl", 0) for t in trades])
        stats["Worst Trade"] = min([t.get("net_pnl", 0) for t in trades])
        total_profit = sum([t.get("net_pnl", 0) for t in wins])
        total_loss = abs(sum([t.get("net_pnl", 0) for t in losses]))
        stats["Profit Factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else np.inf
    else:
        stats["Win Rate %"] = 0
        stats["Average PnL"] = 0
        stats["Best Trade"] = 0
        stats["Worst Trade"] = 0
        stats["Profit Factor"] = 0

    # Max drawdown
    equity = np.array(balance_history)
    if len(equity) > 0:
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        stats["Max Drawdown %"] = drawdowns.min() * 100
    else:
        stats["Max Drawdown %"] = 0

    return stats

def plot_session(state, symbol="BTCUSDT", interval="1m", show_bbands=True):
    """Visualize session trades, price movement, and equity curve."""
    session_start = pd.to_datetime(state.get("session_start"), unit="ms")
    session_end = pd.to_datetime(state.get("session_end"), unit="ms")
    trades = state.get("trade_history", [])
    balance_history = state.get("balance_history", [])

    if session_start is None or session_end is None:
        print("Session timestamps missing in state.json")
        return

    # --- Fetch session price data ---
    df_small = fetch_historical_paginated(symbol, "1m", session_start, session_end)
    df_medium = fetch_historical_paginated(symbol, "1h", session_start, session_end)
    df_big = fetch_historical_paginated(symbol, "4h", session_start, session_end)
    df = prepare_multi_tf(df_small, df_medium, df_big)
    if df.empty:
        print("No klines fetched for the session.")
        return

    # --- Downsample if dataset is too large (for faster, clearer plotting) ---
    if len(df) > 5000:
        df = df.iloc[::len(df)//3000]

    # --- Create subplots: price & equity ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # ========== PRICE CHART ==========
    ax1.plot(df["open_time"], df["close"], label="Price", color="blue", linewidth=1)
    ax1.plot(df["open_time"], df["EMA"], color="purple", alpha=0.5, label="EMA")

    # --- Bollinger Bands (if enabled) ---
    if show_bbands:
        ax1.plot(df["open_time"], df["BBL"], linestyle="--", color="red", alpha=0.5, label="BB Lower")
        ax1.plot(df["open_time"], df["BBM"], linestyle="--", color="black", alpha=0.5, label="BB Mid")
        ax1.plot(df["open_time"], df["BBU"], linestyle="--", color="green", alpha=0.5, label="BB Upper")

    # --- Plot trade entries and exits ---
    for trade in trades:
        # Parse trade time safely (handle both string and timestamp formats)
        entry_time = None
        if isinstance(trade.get("time"), (int, float)):
            entry_time = pd.to_datetime(trade["time"], unit="ms", errors="coerce")
        else:
            entry_time = pd.to_datetime(trade["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        entry_price = trade.get("entry")
        exit_price = trade.get("exit")
        side = trade.get("side", "unknown")

        # Entry marker
        ax1.scatter(
            entry_time, entry_price,
            marker="^" if side == "long" else "v",
            color="lime" if side == "long" else "red",
            s=80, zorder=5, label="Entry" if side == "long" else None
        )

        # Exit marker
        if exit_price:
            exit_reason = trade.get("exit_reason", "")
            if exit_reason in ["stop_loss", "liquidation"]:
                color = "red"
            elif exit_reason in ["take_profit", "rsi"]:
                color = "green"
            else:
                color = "orange"
            ax1.scatter(
                entry_time, exit_price,
                marker="x", color=color, s=60, zorder=5, label="Exit"
            )

    ax1.set_title(f"{symbol} Price and Trades")
    ax1.grid(True, alpha=0.3)

    # --- Equity curve ---
    if balance_history:
        # Try to align balances with candle times
        if len(balance_history) == len(df):
            time_index = df["open_time"]
        else:
            # fallback: linearly spread across session
            time_index = pd.date_range(
                start=df["open_time"].iloc[0],
                end=df["open_time"].iloc[-1],
                periods=len(balance_history)
            )

        ax2.plot(time_index, balance_history,
                 color="purple", label="Equity Curve")
    ax2.set_title("Equity Curve")

    # --- Format x-axis timestamps ---
    fig.autofmt_xdate()

    plt.tight_layout()

    # --- Save plot to temp file and open in browser ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, dpi=150)
        webbrowser.get("firefox").open(tmpfile.name)
    print("Session report plotted.")

    # --- Compute and print session stats ---
    stats = compute_session_stats(trades, balance_history)
    print("\n=== SESSION STATISTICS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


def _write_plotly_fullscreen_html(fig, title):
    fig.update_layout(
        title=title,
        xaxis=dict(rangeslider=dict(visible=False)),
        hovermode="x unified",
        autosize=True,
        template="plotly_dark"
    )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        plot_html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={"responsive": True}
        )
        html = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<style>"
            "html,body{margin:0;padding:0;width:100%;height:100%;background:#111;}"
            "#plot{width:100vw;height:100vh;}"
            "#plot .plotly-graph-div{width:100vw !important;height:100vh !important;}"
            "</style></head><body><div id=\"plot\">"
            f"{plot_html}</div></body></html>"
        )
        with open(tmpfile.name, "w") as f:
            f.write(html)
        webbrowser.get("firefox").open(tmpfile.name)


def _build_interactive_figure(df, trades, equity_series):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )

    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["close"],
            mode="lines",
            name="Price",
            line=dict(color="blue")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["BBL"],
            mode="lines",
            name="Lower BB",
            line=dict(color="red", dash="dash")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["BBM"],
            mode="lines",
            name="Middle BB",
            line=dict(color="black", dash="dash")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["BBU"],
            mode="lines",
            name="Upper BB",
            line=dict(color="green", dash="dash")
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["EMA"],
            mode="lines",
            name="EMA",
            line=dict(color="purple")
        ),
        row=1,
        col=1
    )

    if trades is not None and not trades.empty:
        entries = trades[pd.notna(trades["entry"])]
        exits = trades[pd.notna(trades["exit"])]

        fig.add_trace(
            go.Scatter(
                x=entries["time"],
                y=entries["entry"],
                mode="markers",
                name="Entries",
                marker=dict(
                    symbol=entries["side"].map({"long": "triangle-up", "short": "triangle-down"}),
                    color="blue",
                    size=10
                ),
                hovertemplate="Entry: %{y}<br>Time: %{x}<extra></extra>"
            ),
            row=1,
            col=1
        )

        exit_colors = exits["exit_reason"].map({
            "stop_loss": "red",
            "liquidation": "red",
            "take_profit": "green",
            "rsi": "green"
        }).fillna("orange")

        fig.add_trace(
            go.Scatter(
                x=exits["time"],
                y=exits["exit"],
                mode="markers",
                name="Exits",
                marker=dict(symbol="x", color=exit_colors, size=9),
                hovertemplate="Exit: %{y}<br>Time: %{x}<extra></extra>"
            ),
            row=1,
            col=1
        )

    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple")
            ),
            row=2,
            col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    if equity_series is not None and not equity_series.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode="lines",
                name="Equity Curve",
                line=dict(color="purple")
            ),
            row=3,
            col=1
        )

    return fig


def plot_results_interactive(df, trades, balance_history, split_by_year=True):
    """Interactive plot with price, trades, RSI, and equity curve (zoom/pan)."""
    if df is None or df.empty:
        print("No data to plot.")
        return

    if balance_history:
        if len(balance_history) == len(df):
            equity_index = df["open_time"]
        else:
            equity_index = pd.date_range(
                start=df["open_time"].iloc[0],
                end=df["open_time"].iloc[-1],
                periods=len(balance_history)
            )
        equity_series = pd.Series(balance_history, index=equity_index)
    else:
        equity_series = None

    trades_local = trades
    if trades_local is not None and not trades_local.empty and "time" in trades_local.columns:
        trades_local = trades_local.copy()
        trades_local["time"] = pd.to_datetime(trades_local["time"], errors="coerce")

    years = sorted(df["open_time"].dt.year.unique())
    if not split_by_year or len(years) <= 1:
        fig = _build_interactive_figure(df, trades_local, equity_series)
        _write_plotly_fullscreen_html(fig, "Interactive Backtest (Zoom/Pan)")
        return

    for year in years:
        df_year = df[df["open_time"].dt.year == year]
        if df_year.empty:
            continue

        start_year = df_year["open_time"].iloc[0]
        end_year = df_year["open_time"].iloc[-1]

        trades_year = trades_local
        if trades_local is not None and not trades_local.empty and "time" in trades_local.columns:
            trades_year = trades_local[
                (trades_local["time"] >= start_year) & (trades_local["time"] <= end_year)
            ]

        equity_year = None
        if equity_series is not None and not equity_series.empty:
            equity_year = equity_series[(equity_series.index >= start_year) & (equity_series.index <= end_year)]

        fig = _build_interactive_figure(df_year, trades_year, equity_year)
        _write_plotly_fullscreen_html(fig, f"Interactive Backtest (Zoom/Pan) - {year}")


def monte_carlo_from_equity(df, balance_history, start_balance=10000, sims=10000, horizon_months=None, block_len=3, show_plot=True):
    """
    Advanced Monte Carlo stress test based on monthly equity returns.
    Includes CAGR, Volatility and Sharpe Ratio metrics.
    """

    # --- 1. Generate synthetic datetime index for the equity curve ---
    if len(balance_history) == 0:
        print("No equity data found for Monte Carlo test.")
        return {}

    start_time = pd.to_datetime(df["open_time"].iloc[0])
    end_time = pd.to_datetime(df["open_time"].iloc[-1])
    synthetic_index = pd.date_range(start=start_time, end=end_time, periods=len(balance_history))

    equity = pd.Series(balance_history, index=synthetic_index)

    # --- 2. Resample monthly and compute returns ---
    eq_monthly = equity.resample("ME").last().dropna()
    monthly_returns = eq_monthly.pct_change().dropna()

    if len(monthly_returns) < 3:
        print("Not enough data for monthly Monte Carlo test.")
        return {}

    # --- 3. Convert to log returns for numerical stability ---
    log_returns = np.log1p(monthly_returns.values)

    # --- 4. Simulation setup ---
    if horizon_months is None:
        horizon_months = len(log_returns)

    results = np.empty(sims)
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(horizon_months / block_len))

    # --- 5. Monte Carlo block bootstrap ---
    for i in range(sims):
        seq = []
        for _ in range(n_blocks):
            start_idx = rng.integers(0, len(log_returns) - block_len + 1)
            seq.append(log_returns[start_idx:start_idx + block_len])
        sampled = np.concatenate(seq)[:horizon_months]
        results[i] = start_balance * np.exp(sampled.sum())

    # --- 6. Compute percentiles ---
    p5, p50, p95 = np.percentile(results, [5, 50, 95])

    # --- 7. Risk metrics (CAGR, Volatility, Sharpe) ---
    years = horizon_months / 12
    cagr = ((p50 / start_balance) ** (1 / years) - 1) * 100
    volatility = np.std(monthly_returns) * np.sqrt(12) * 100
    risk_free_rate = 0.03  # 3% baseline annual risk-free rate
    sharpe = (cagr / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else np.nan

    # --- 8. Prepare summary dictionary ---
    summary = {
        "Simulations": sims,
        "Months per Run": horizon_months,
        "5th Percentile (Pessimistic)": round(p5, 2),
        "50th Percentile (Median)": round(p50, 2),
        "95th Percentile (Optimistic)": round(p95, 2),
        "Expected Range": f"{round(p5,2)} → {round(p95,2)}",
        "CAGR %": round(cagr, 2),
        "Volatility %": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

    # --- 9. Optional visualization ---
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.hist(results, bins=80, color="purple", alpha=0.7)
        plt.axvline(p5, color="red", linestyle="--", label="5th Percentile")
        plt.axvline(p50, color="black", linestyle="-", label="Median")
        plt.axvline(p95, color="green", linestyle="--", label="95th Percentile")
        plt.title("Monte Carlo Distribution Based on Monthly Equity Returns")
        plt.xlabel("Final Balance ($)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        # --- Save plot to temp file and open in browser ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            webbrowser.get("firefox").open(tmpfile.name)

    # --- 10. Print formatted summary ---
    print("\nMonte Carlo Stress-Test Summary (Monthly):")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


def rolling_window_backtest_distribution(
    df,
    k_days,
    n_days=None,
    start_balance=10000,
    show_plot=True,
    max_workers=None,
    **strategy_kwargs
):
    """
    Run rolling k-day backtests over an n-day period and return a distribution
    of final balances, similar to the Monte Carlo view.
    """
    if df is None or df.empty:
        print("No data for rolling window backtest.")
        return {}

    if k_days <= 0:
        print("k_days must be > 0.")
        return {}

    start_time = pd.to_datetime(df["open_time"].iloc[0])
    end_time = pd.to_datetime(df["open_time"].iloc[-1])
    total_days = int((end_time - start_time).total_seconds() // 86400)

    if n_days is None:
        n_days = total_days
    n_days = min(n_days, total_days)

    if n_days < k_days:
        print("n_days must be >= k_days.")
        return {}

    final_balances = []
    pnl_pcts = []
    win_rates = []
    max_drawdowns = []

    total_windows = n_days - k_days + 1
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1 or total_windows <= 1:
        for offset in tqdm(range(total_windows), desc="Rolling Backtest", ncols=100):
            window_start = start_time + pd.Timedelta(days=offset)
            window_end = window_start + pd.Timedelta(days=k_days)
            df_window = df[(df["open_time"] >= window_start) & (df["open_time"] < window_end)]

            if df_window.empty:
                continue

            final_balance, trades, balance_history, _ = run_strategy(
                df_window,
                live=False,
                initial_balance=start_balance,
                use_state=False,
                enable_logs=False,
                show_progress=False,
                **strategy_kwargs
            )
            final_balances.append(final_balance)
            pnl_pcts.append((final_balance - start_balance) / start_balance * 100)

            if trades is not None and len(trades) > 0:
                wins = trades[trades["net_pnl"] > 0]
                win_rates.append(len(wins) / len(trades) * 100)
            else:
                win_rates.append(0)

            equity = np.array(balance_history)
            if len(equity) > 0:
                rolling_max = np.maximum.accumulate(equity)
                drawdowns = (equity - rolling_max) / rolling_max
                max_drawdowns.append(drawdowns.min() * 100)
            else:
                max_drawdowns.append(0)
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_rw_init,
            initargs=(df, start_time, k_days, start_balance, strategy_kwargs)
        ) as executor:
            futures = [executor.submit(_rw_worker, offset) for offset in range(total_windows)]
            for fut in tqdm(as_completed(futures), total=total_windows, desc="Rolling Backtest", ncols=100):
                result = fut.result()
                if result is None:
                    continue
                final_balance, pnl_pct, win_rate, max_drawdown = result
                final_balances.append(final_balance)
                pnl_pcts.append(pnl_pct)
                win_rates.append(win_rate)
                max_drawdowns.append(max_drawdown)

    if not final_balances:
        print("No rolling windows produced results.")
        return {}

    results = np.array(final_balances)
    p5, p50, p95 = np.percentile(results, [5, 50, 95])
    pnl_p5, pnl_p50, pnl_p95 = np.percentile(pnl_pcts, [5, 50, 95])
    wr_p5, wr_p50, wr_p95 = np.percentile(win_rates, [5, 50, 95])
    dd_p5, dd_p50, dd_p95 = np.percentile(max_drawdowns, [5, 50, 95])

    summary = {
        "Windows": len(results),
        "Days per Window": k_days,
        "Total Days": n_days,
        "5th Percentile (Pessimistic)": round(p5, 2),
        "50th Percentile (Median)": round(p50, 2),
        "95th Percentile (Optimistic)": round(p95, 2),
        "Expected Range": f"{round(p5,2)} → {round(p95,2)}",
        "PnL % (5/50/95)": f"{round(pnl_p5,2)} / {round(pnl_p50,2)} / {round(pnl_p95,2)}",
        "Win Rate % (5/50/95)": f"{round(wr_p5,2)} / {round(wr_p50,2)} / {round(wr_p95,2)}",
        "Max Drawdown % (5/50/95)": f"{round(dd_p5,2)} / {round(dd_p50,2)} / {round(dd_p95,2)}"
    }

    if show_plot:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
        ax_bal, ax_pnl, ax_wr, ax_dd = axes

        ax_bal.hist(results, bins=60, color="teal", alpha=0.7)
        ax_bal.axvline(p5, color="red", linestyle="--", label="5th Percentile")
        ax_bal.axvline(p50, color="black", linestyle="-", label="Median")
        ax_bal.axvline(p95, color="green", linestyle="--", label="95th Percentile")
        ax_bal.set_title("Final Balance")
        ax_bal.set_xlabel("Final Balance ($)")
        ax_bal.set_ylabel("Frequency")
        ax_bal.legend()
        ax_bal.text(
            0.02,
            0.95,
            f"P5: {p5:.2f}\nP50: {p50:.2f}\nP95: {p95:.2f}",
            transform=ax_bal.transAxes,
            va="top"
        )

        ax_pnl.hist(pnl_pcts, bins=60, color="steelblue", alpha=0.7)
        ax_pnl.axvline(pnl_p5, color="red", linestyle="--", label="5th Percentile")
        ax_pnl.axvline(pnl_p50, color="black", linestyle="-", label="Median")
        ax_pnl.axvline(pnl_p95, color="green", linestyle="--", label="95th Percentile")
        ax_pnl.set_title("PnL %")
        ax_pnl.set_xlabel("PnL %")
        ax_pnl.set_ylabel("Frequency")
        ax_pnl.legend()
        ax_pnl.text(
            0.02,
            0.95,
            f"P5: {pnl_p5:.2f}\nP50: {pnl_p50:.2f}\nP95: {pnl_p95:.2f}",
            transform=ax_pnl.transAxes,
            va="top"
        )

        ax_wr.hist(win_rates, bins=60, color="purple", alpha=0.7)
        ax_wr.axvline(wr_p5, color="red", linestyle="--", label="5th Percentile")
        ax_wr.axvline(wr_p50, color="black", linestyle="-", label="Median")
        ax_wr.axvline(wr_p95, color="green", linestyle="--", label="95th Percentile")
        ax_wr.set_title("Win Rate %")
        ax_wr.set_xlabel("Win Rate %")
        ax_wr.set_ylabel("Frequency")
        ax_wr.legend()
        ax_wr.text(
            0.02,
            0.95,
            f"P5: {wr_p5:.2f}\nP50: {wr_p50:.2f}\nP95: {wr_p95:.2f}",
            transform=ax_wr.transAxes,
            va="top"
        )

        ax_dd.hist(max_drawdowns, bins=60, color="darkorange", alpha=0.7)
        ax_dd.axvline(dd_p5, color="red", linestyle="--", label="5th Percentile")
        ax_dd.axvline(dd_p50, color="black", linestyle="-", label="Median")
        ax_dd.axvline(dd_p95, color="green", linestyle="--", label="95th Percentile")
        ax_dd.set_title("Max Drawdown %")
        ax_dd.set_xlabel("Max Drawdown %")
        ax_dd.set_ylabel("Frequency")
        ax_dd.legend()
        ax_dd.text(
            0.02,
            0.95,
            f"P5: {dd_p5:.2f}\nP50: {dd_p50:.2f}\nP95: {dd_p95:.2f}",
            transform=ax_dd.transAxes,
            va="top"
        )

        fig.suptitle("Rolling Window Backtest Distributions")
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            webbrowser.get("firefox").open(tmpfile.name)

    print("\nRolling Window Backtest Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


if __name__ == "__main__":
    if _client is None:
        from dotenv import load_dotenv
        from binance.client import Client
        load_dotenv()
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        set_client(Client(api_key, api_secret))
    state = load_state()
    if state:
        plot_session(state)
    else:
        print("No state found. Run a trading session first.")
