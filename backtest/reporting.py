import os
import tempfile
import webbrowser
import json
import html
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest.discrete_tape import build_discrete_market_tape
from backtest.dataset import fetch_historical_paginated, prepare_multi_tf
from backtest.stats import compute_stats
from bot.event_log import get_technical_logger
from bot.state import (
    load_balance_history,
    load_state as load_runtime_state,
    load_trade_history,
)
from core.binance_retry import create_binance_client, run_binance_with_retries
from core.engine import run_strategy_on_market_events, run_strategy_on_tape
from core.indicator_inputs import uses_realtime_indicator_inputs
from core.market_events import iter_market_event_stream, market_event_from_dict

if TYPE_CHECKING:
    from backtest.runner import BacktestConfig


def _ensure_matplotlib_cache_dir() -> None:
    cache_root = Path(
        os.getenv("SCROOGE_MPLCONFIGDIR", os.getenv("MPLCONFIGDIR", "/tmp/scrooge-matplotlib"))
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root)


_ensure_matplotlib_cache_dir()

import matplotlib.pyplot as plt

_client = None
_rw_df = None
_rw_market_event_path = None
_rw_market_event_day_ranges = None
_rw_mode = None
_rw_start_time = None
_rw_k_days = None
_rw_start_balance = None
_rw_strategy_kwargs = None
_rw_symbol = None
_rw_candle_interval = None
_rw_intervals = None
_rw_strategy_mode = None
_rw_execution_mode = None
_rw_indicator_inputs = None
_rw_runtime_mode = None
technical_logger = get_technical_logger()

MAX_REPORT_POINTS = 6000
MAX_YEARLY_REPORT_POINTS = 2200
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if np.isnan(numeric) else numeric
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_equity_series(df: pd.DataFrame, balance_history: list[float] | None) -> pd.Series | None:
    if not balance_history or df is None or df.empty:
        return None
    if len(balance_history) == len(df):
        equity_index = pd.to_datetime(df["open_time"], errors="coerce")
    else:
        equity_index = pd.date_range(
            start=pd.to_datetime(df["open_time"].iloc[0]),
            end=pd.to_datetime(df["open_time"].iloc[-1]),
            periods=len(balance_history),
        )
    return pd.Series(balance_history, index=equity_index)


def _normalize_trades_frame(trades: pd.DataFrame | None) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    trades_local = trades.copy()
    if "time" in trades_local.columns:
        trades_local["time"] = pd.to_datetime(trades_local["time"], errors="coerce")
    return trades_local


def _sample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= max_points:
        return df.copy() if df is not None else pd.DataFrame()
    indices = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[np.unique(indices)].copy()


def _compute_monthly_returns(equity_series: pd.Series | None) -> pd.Series:
    if equity_series is None or equity_series.empty:
        return pd.Series(dtype=float)
    monthly = equity_series.resample("ME").last().dropna()
    return monthly.pct_change().dropna() * 100


def _summarize_monthly_returns(equity_series: pd.Series | None) -> dict[str, Any] | None:
    monthly_returns = _compute_monthly_returns(equity_series)
    if monthly_returns.empty:
        return None
    return {
        "months": int(len(monthly_returns)),
        "best_month_pct": round(float(monthly_returns.max()), 2),
        "worst_month_pct": round(float(monthly_returns.min()), 2),
        "positive_month_ratio_pct": round(float((monthly_returns > 0).mean() * 100), 2),
    }


def _extract_drawdown_episodes(equity_series: pd.Series | None) -> tuple[pd.Series, list[dict[str, Any]]]:
    if equity_series is None or equity_series.empty:
        return pd.Series(dtype=float), []

    rolling_max = equity_series.cummax()
    drawdown = ((equity_series / rolling_max) - 1.0) * 100.0
    timestamps = list(drawdown.index)
    values = list(drawdown.values)

    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    for index, value in enumerate(values):
        if value < -1e-9:
            if active is None:
                peak_index = max(index - 1, 0)
                active = {
                    "start": timestamps[peak_index],
                    "start_index": peak_index,
                    "trough": timestamps[index],
                    "trough_index": index,
                    "depth_pct": float(value),
                }
            elif value < active["depth_pct"]:
                active["depth_pct"] = float(value)
                active["trough"] = timestamps[index]
                active["trough_index"] = index
        elif active is not None:
            recovery_ts = timestamps[index]
            start_ts = pd.Timestamp(active["start"])
            trough_ts = pd.Timestamp(active["trough"])
            recovery_dt = pd.Timestamp(recovery_ts)
            episodes.append(
                {
                    "start": start_ts,
                    "trough": trough_ts,
                    "recovery": recovery_dt,
                    "depth_pct": round(float(active["depth_pct"]), 2),
                    "duration_days": round(float((recovery_dt - start_ts).total_seconds() / 86400.0), 2),
                    "time_to_trough_days": round(float((trough_ts - start_ts).total_seconds() / 86400.0), 2),
                }
            )
            active = None

    if active is not None:
        start_ts = pd.Timestamp(active["start"])
        trough_ts = pd.Timestamp(active["trough"])
        episodes.append(
            {
                "start": start_ts,
                "trough": trough_ts,
                "recovery": None,
                "depth_pct": round(float(active["depth_pct"]), 2),
                "duration_days": round(float((pd.Timestamp(timestamps[-1]) - start_ts).total_seconds() / 86400.0), 2),
                "time_to_trough_days": round(float((trough_ts - start_ts).total_seconds() / 86400.0), 2),
            }
        )

    return drawdown, episodes


def _summarize_drawdowns(equity_series: pd.Series | None) -> dict[str, Any] | None:
    drawdown_series, episodes = _extract_drawdown_episodes(equity_series)
    if drawdown_series.empty:
        return None
    completed = [item for item in episodes if item.get("recovery") is not None]
    durations = [float(item["duration_days"]) for item in completed]
    return {
        "episodes": int(len(episodes)),
        "max_drawdown_pct": round(float(drawdown_series.min()), 2),
        "longest_recovery_days": round(max(durations), 2) if durations else None,
        "median_recovery_days": round(float(np.median(durations)), 2) if durations else None,
    }


def _build_monthly_heatmap_figure(equity_series: pd.Series | None) -> go.Figure | None:
    monthly_returns = _compute_monthly_returns(equity_series)
    if monthly_returns.empty:
        return None
    returns_df = monthly_returns.to_frame("return_pct")
    returns_df["year"] = returns_df.index.year
    returns_df["month"] = returns_df.index.month
    pivot = returns_df.pivot(index="year", columns="month", values="return_pct").reindex(columns=range(1, 13))
    text = pivot.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{float(value):.1f}%"))

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.values,
                x=MONTH_LABELS,
                y=[str(year) for year in pivot.index],
                text=text.values,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#5c1824"],
                    [0.45, "#1a202c"],
                    [0.5, "#283241"],
                    [1.0, "#0f6b4c"],
                ],
                colorbar=dict(title="Return %"),
                zmid=0,
                hovertemplate="Year %{y}<br>Month %{x}<br>Return %{z:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(320, 120 + 48 * len(pivot.index)),
        margin=dict(l=48, r=28, t=52, b=48),
        title="Monthly Returns Heatmap",
    )
    return fig


def _build_drawdown_diagnostics_figure(equity_series: pd.Series | None) -> go.Figure | None:
    drawdown_series, episodes = _extract_drawdown_episodes(equity_series)
    if drawdown_series.empty:
        return None
    sampled_drawdown = drawdown_series
    if len(sampled_drawdown) > MAX_REPORT_POINTS:
        indices = np.linspace(0, len(sampled_drawdown) - 1, num=MAX_REPORT_POINTS, dtype=int)
        sampled_drawdown = sampled_drawdown.iloc[np.unique(indices)]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.11,
        row_heights=[0.56, 0.44],
        subplot_titles=("Equity Drawdown %", "Recovery Duration by Episode"),
    )
    fig.add_trace(
        go.Scatter(
            x=sampled_drawdown.index,
            y=sampled_drawdown.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ff7488", width=1.8),
            name="Drawdown %",
        ),
        row=1,
        col=1,
    )

    completed = [item for item in episodes if item.get("recovery") is not None]
    if completed:
        fig.add_trace(
            go.Bar(
                x=[f"#{index + 1}" for index in range(len(completed))],
                y=[float(item["duration_days"]) for item in completed],
                marker=dict(
                    color=[float(item["depth_pct"]) for item in completed],
                    colorscale="RdYlGn",
                    reversescale=True,
                    colorbar=dict(title="Depth %"),
                ),
                name="Recovery Days",
                hovertemplate=(
                    "Episode %{x}<br>"
                    "Recovery %{y:.2f} days<br>"
                    "Depth %{marker.color:.2f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=820,
        margin=dict(l=48, r=28, t=70, b=48),
        showlegend=False,
        title="Drawdown Diagnostics",
    )
    fig.update_yaxes(title_text="Drawdown %", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=2, col=1)
    return fig


def _build_trade_diagnostics_figure(trades: pd.DataFrame, equity_series: pd.Series | None) -> go.Figure | None:
    has_trade_data = not trades.empty and "net_pnl" in trades.columns
    monthly_returns = _compute_monthly_returns(equity_series)
    if not has_trade_data and monthly_returns.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.48, 0.52],
        subplot_titles=("Trade PnL Distribution", "Monthly Returns %"),
    )

    if has_trade_data:
        pnl_values = pd.to_numeric(trades["net_pnl"], errors="coerce").dropna()
        if not pnl_values.empty:
            fig.add_trace(
                go.Histogram(
                    x=pnl_values,
                    name="Trade PnL",
                    marker=dict(color="#d7a844"),
                    opacity=0.82,
                    nbinsx=min(80, max(20, int(np.sqrt(len(pnl_values))) * 2)),
                ),
                row=1,
                col=1,
            )

    if not monthly_returns.empty:
        bar_colors = np.where(monthly_returns >= 0, "#26d39a", "#ff7488")
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values,
                name="Monthly Return %",
                marker=dict(color=bar_colors),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=780,
        margin=dict(l=48, r=28, t=70, b=48),
        showlegend=False,
    )
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    return fig


def _build_monte_carlo_figure(report: dict[str, Any] | None) -> go.Figure | None:
    if not report:
        return None
    plot_data = report.get("plot_data")
    if not isinstance(plot_data, dict):
        return None
    results = plot_data.get("final_balances") or []
    if not results:
        return None
    percentiles = report.get("percentiles", {})
    p5 = percentiles.get("p5")
    p50 = percentiles.get("p50")
    p95 = percentiles.get("p95")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=results,
            name="Simulated Final Balance",
            marker=dict(color="#8c6cff"),
            opacity=0.82,
            nbinsx=min(120, max(30, int(np.sqrt(len(results))) * 2)),
        )
    )
    for value, color, label in (
        (p5, "#ff7488", "P5"),
        (p50, "#f5f7fb", "Median"),
        (p95, "#32e39b", "P95"),
    ):
        if value is None:
            continue
        fig.add_vline(x=value, line_color=color, line_dash="dash", annotation_text=label, annotation_position="top")
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=420,
        margin=dict(l=48, r=28, t=52, b=48),
        title="Monte Carlo Distribution",
        xaxis_title="Final Balance ($)",
        yaxis_title="Frequency",
    )
    return fig


def _add_percentile_markers(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    p5: float | None,
    p50: float | None,
    p95: float | None,
) -> None:
    for value, color, label in (
        (p5, "#ff7488", "P5"),
        (p50, "#f5f7fb", "Median"),
        (p95, "#32e39b", "P95"),
    ):
        if value is None:
            continue
        fig.add_vline(
            x=value,
            line_color=color,
            line_dash="dash" if label != "Median" else "solid",
            annotation_text=label,
            annotation_position="top",
            row=row,
            col=col,
        )


def _build_rolling_window_distribution_figure(report: dict[str, Any] | None) -> go.Figure | None:
    if not report:
        return None
    plot_data = report.get("plot_data")
    percentiles = report.get("percentiles")
    if not isinstance(plot_data, dict) or not isinstance(percentiles, dict):
        return None

    final_balances = plot_data.get("final_balances") or []
    pnl_pcts = plot_data.get("pnl_pcts") or []
    win_rates = plot_data.get("win_rates") or []
    max_drawdowns = plot_data.get("max_drawdowns") or []
    if not final_balances:
        return None

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Final Balance",
            "PnL %",
            "Win Rate %",
            "Max Drawdown %",
        ),
    )

    histogram_specs = (
        (final_balances, "#58b5b7", "Final Balance", "Final Balance ($)", percentiles.get("final_balance", {})),
        (pnl_pcts, "#8cb6df", "PnL %", "PnL %", percentiles.get("pnl_pct", {})),
        (win_rates, "#9c59b6", "Win Rate %", "Win Rate %", percentiles.get("win_rate_pct", {})),
        (max_drawdowns, "#ffad45", "Max Drawdown %", "Max Drawdown %", percentiles.get("max_drawdown_pct", {})),
    )

    for index, (values, color, name, xaxis_title, pct) in enumerate(histogram_specs, start=1):
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                marker=dict(color=color),
                opacity=0.86,
                nbinsx=min(90, max(24, int(np.sqrt(len(values))) * 2)),
                hovertemplate=f"{name}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=index,
            col=1,
        )
        _add_percentile_markers(
            fig,
            row=index,
            col=1,
            p5=pct.get("p5"),
            p50=pct.get("p50"),
            p95=pct.get("p95"),
        )
        fig.update_xaxes(title_text=xaxis_title, row=index, col=1)
        fig.update_yaxes(title_text="Frequency", row=index, col=1)

    fig.update_layout(
        template="plotly_dark",
        bargap=0.03,
        height=1320,
        margin=dict(l=48, r=28, t=78, b=48),
        showlegend=False,
        title="Rolling Window Distribution",
    )
    return fig


def _build_rolling_window_timeline_figure(report: dict[str, Any] | None) -> go.Figure | None:
    if not report:
        return None
    plot_data = report.get("plot_data")
    if not isinstance(plot_data, dict):
        return None
    window_starts = pd.to_datetime(plot_data.get("window_starts") or [], errors="coerce")
    final_balances = plot_data.get("final_balances") or []
    pnl_pcts = plot_data.get("pnl_pcts") or []
    max_drawdowns = plot_data.get("max_drawdowns") or []
    if len(window_starts) == 0 or len(final_balances) == 0:
        return None

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Rolling Final Balance", "Rolling Return %", "Rolling Max Drawdown %"),
    )

    fig.add_trace(
        go.Scatter(
            x=window_starts,
            y=final_balances,
            mode="lines",
            name="Final Balance",
            line=dict(color="#26d39a", width=1.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=window_starts,
            y=pnl_pcts,
            mode="lines",
            name="Return %",
            line=dict(color="#54a8ff", width=1.6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=window_starts,
            y=max_drawdowns,
            mode="lines",
            name="Max DD %",
            line=dict(color="#ff9f43", width=1.6),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=860,
        margin=dict(l=48, r=28, t=70, b=48),
        showlegend=False,
        title="Rolling Window Timeline",
    )
    fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    fig.update_yaxes(title_text="Max DD %", row=3, col=1)
    return fig


def _build_yearly_summary_rows(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    equity_series: pd.Series | None,
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: list[dict[str, Any]] = []
    years = sorted(pd.to_datetime(df["open_time"], errors="coerce").dropna().dt.year.unique())
    for year in years:
        df_year = df[df["open_time"].dt.year == year]
        if df_year.empty:
            continue
        trades_year = trades
        if not trades.empty and "time" in trades.columns:
            trades_year = trades[(trades["time"] >= df_year["open_time"].iloc[0]) & (trades["time"] <= df_year["open_time"].iloc[-1])]
        if equity_series is not None and not equity_series.empty:
            equity_year = equity_series[(equity_series.index >= df_year["open_time"].iloc[0]) & (equity_series.index <= df_year["open_time"].iloc[-1])]
            if len(equity_year) >= 2:
                year_return_pct = (equity_year.iloc[-1] - equity_year.iloc[0]) / equity_year.iloc[0] * 100
            else:
                year_return_pct = None
        else:
            year_return_pct = None
        rows.append(
            {
                "year": int(year),
                "candles": int(len(df_year)),
                "trades": int(len(trades_year)) if trades_year is not None else 0,
                "return_pct": None if year_return_pct is None else round(float(year_return_pct), 2),
                "start": pd.to_datetime(df_year["open_time"].iloc[0]).strftime("%Y-%m-%d"),
                "end": pd.to_datetime(df_year["open_time"].iloc[-1]).strftime("%Y-%m-%d"),
            }
        )
    return rows


def _build_overview_figure(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    equity_series: pd.Series | None,
    *,
    max_points: int = MAX_REPORT_POINTS,
) -> go.Figure | None:
    if df is None or df.empty:
        return None
    sampled = _sample_frame(df, max_points=max_points)
    sampled_trades = trades
    if not trades.empty and "time" in trades.columns:
        sampled_trades = trades[(trades["time"] >= sampled["open_time"].iloc[0]) & (trades["time"] <= sampled["open_time"].iloc[-1])]
    sampled_equity = equity_series
    if equity_series is not None and not equity_series.empty:
        sampled_equity = equity_series[(equity_series.index >= sampled["open_time"].iloc[0]) & (equity_series.index <= sampled["open_time"].iloc[-1])]
        if sampled_equity is not None and len(sampled_equity) > max_points:
            equity_indices = np.linspace(0, len(sampled_equity) - 1, num=max_points, dtype=int)
            sampled_equity = sampled_equity.iloc[np.unique(equity_indices)]
    fig = _build_interactive_figure(sampled, sampled_trades, sampled_equity)
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=1080,
        margin=dict(l=48, r=28, t=60, b=48),
        title="Backtest Overview",
    )
    return fig


def _figure_to_section(fig: go.Figure, *, section_id: str, include_plotlyjs: bool, title: str, subtitle: str | None = None) -> str:
    plot_html = fig.to_html(
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )
    subtitle_html = f"<p class=\"report-subtitle\">{subtitle}</p>" if subtitle else ""
    return (
        f"<section class=\"report-section\" id=\"{section_id}\">"
        f"<h2>{title}</h2>"
        f"{subtitle_html}"
        f"{plot_html}"
        "</section>"
    )


def _format_trade_value(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float | int | np.floating | np.integer):
        return f"{float(value):.{digits}f}{suffix}"
    return str(value)


def _render_trade_history_section(trades: pd.DataFrame) -> str:
    if trades.empty:
        return ""

    trades_local = trades.copy()
    for column in ("entry_ts", "exit_ts", "trail_activated_at"):
        if column in trades_local.columns:
            trades_local[column] = pd.to_datetime(trades_local[column], errors="coerce")

    if "exit_ts" in trades_local.columns:
        trades_local = trades_local.sort_values("exit_ts", ascending=False, na_position="last")

    blocks: list[str] = []
    for _, row in trades_local.iterrows():
        side = str(row.get("side", "trade")).strip().lower() or "trade"
        net_pnl = pd.to_numeric(row.get("net_pnl"), errors="coerce")
        roi_pct = pd.to_numeric(row.get("roi_pct"), errors="coerce")
        duration_seconds = pd.to_numeric(row.get("duration_seconds"), errors="coerce")
        duration_hours = None if pd.isna(duration_seconds) else float(duration_seconds) / 3600.0
        pnl_positive = bool(not pd.isna(net_pnl) and float(net_pnl) >= 0.0)
        tone_class = "trade-ledger-item--gain" if pnl_positive else "trade-ledger-item--loss"
        side_label = html.escape(side.upper())
        pnl_label = "N/A" if pd.isna(net_pnl) else f"{float(net_pnl):+.2f}"
        summary_title = (
            f"<span class=\"trade-ledger-summary-side\">{side_label}</span>"
            f"<span class=\"trade-ledger-summary-pnl\">{html.escape(pnl_label)}</span>"
        )

        detail_pairs = [
            ("Entry", _format_trade_value(pd.to_numeric(row.get("entry"), errors="coerce"))),
            ("Exit", _format_trade_value(pd.to_numeric(row.get("exit"), errors="coerce"))),
            ("Entry Time", row.get("entry_ts").strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("entry_ts")) else "N/A"),
            ("Exit Time", row.get("exit_ts").strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("exit_ts")) else "N/A"),
            ("Size", _format_trade_value(pd.to_numeric(row.get("size"), errors="coerce"), digits=6)),
            ("ROI %", _format_trade_value(roi_pct, suffix="%")),
            ("Fee", _format_trade_value(pd.to_numeric(row.get("opening_fee"), errors="coerce"))),
            ("Leverage", _format_trade_value(pd.to_numeric(row.get("leverage"), errors="coerce"), digits=0)),
            ("Duration", "N/A" if duration_hours is None else f"{duration_hours:.2f}h"),
            ("Exit Reason", str(row.get("exit_reason", "N/A"))),
            ("Trigger", str(row.get("trigger", "N/A"))),
            ("Stake Mode", str(row.get("stake_mode", "N/A"))),
            ("Stop Loss", _format_trade_value(pd.to_numeric(row.get("sl"), errors="coerce"))),
            ("Take Profit", _format_trade_value(pd.to_numeric(row.get("tp"), errors="coerce"))),
            ("Trail Moves", _format_trade_value(pd.to_numeric(row.get("trail_moves"), errors="coerce"), digits=0)),
            (
                "Tail Guard",
                "Yes" if bool(row.get("via_tail_guard")) else "No",
            ),
        ]

        detail_html = "".join(
            "<div class=\"trade-ledger-detail\">"
            f"<span class=\"trade-ledger-detail-label\">{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            "</div>"
            for label, value in detail_pairs
        )

        blocks.append(
            "<details class=\"trade-ledger-item {tone_class}\">"
            f"<summary>{summary_title}</summary>"
            f"<div class=\"trade-ledger-detail-grid\">{detail_html}</div>"
            "</details>".format(tone_class=tone_class)
        )

    return (
        "<details class=\"report-section report-section-collapsible\" id=\"trade-history\">"
        "<summary>"
        "<span>Trade History</span>"
        "<span class=\"report-collapse-hint\">Open closed trades ledger</span>"
        "</summary>"
        "<p class=\"report-subtitle\">Newest closed trades first. Expand any row for position details.</p>"
        "<div class=\"trade-ledger\">"
        f"{''.join(blocks)}"
        "</div>"
        "</details>"
    )


def write_backtest_report(
    config: "BacktestConfig",
    df: pd.DataFrame,
    trades: pd.DataFrame,
    balance_history: list[float],
    stats: dict[str, Any],
    *,
    monte_carlo_report: dict[str, Any] | None = None,
    rolling_window_report: dict[str, Any] | None = None,
    include_html: bool = True,
) -> dict[str, Path]:
    output_dir = config.event_log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "report.json"
    report_html_path = output_dir / "report.html"

    trades_local = _normalize_trades_frame(trades)
    equity_series = _build_equity_series(df, balance_history)
    yearly_summary = _build_yearly_summary_rows(df, trades_local, equity_series)
    monthly_return_summary = _summarize_monthly_returns(equity_series)
    drawdown_summary = _summarize_drawdowns(equity_series)

    payload = {
        "symbol": config.symbol,
        "strategy_mode": config.strategy_mode,
        "execution_mode": config.execution_mode,
        "backtest_input_mode": config.backtest_input_mode,
        "agg_trade_tick_interval": config.agg_trade_tick_interval,
        "period": {
            "start": str(pd.to_datetime(df["open_time"].iloc[0])),
            "end": str(pd.to_datetime(df["open_time"].iloc[-1])),
            "rows": int(len(df)),
        },
        "stats": stats,
        "yearly_summary": yearly_summary,
        "monthly_returns": monthly_return_summary,
        "drawdowns": drawdown_summary,
        "monte_carlo": None if monte_carlo_report is None else monte_carlo_report.get("summary", monte_carlo_report),
        "rolling_window": None if rolling_window_report is None else rolling_window_report.get("summary", rolling_window_report),
    }
    _write_json(report_json_path, payload)

    if not include_html:
        technical_logger.info("backtest_report_written json=%s", report_json_path)
        return {"json": report_json_path}

    sections: list[str] = []
    include_plotlyjs = True

    overview_fig = _build_overview_figure(df, trades_local, equity_series)
    if overview_fig is not None:
        sections.append(
            _figure_to_section(
                overview_fig,
                section_id="overview",
                include_plotlyjs=include_plotlyjs,
                title="Overview",
                subtitle=f"Decimated to at most {MAX_REPORT_POINTS} points per primary series for browser-safe rendering.",
            )
        )
        include_plotlyjs = False

    trade_history_section = _render_trade_history_section(trades_local)
    if trade_history_section:
        sections.insert(1 if sections else 0, trade_history_section)

    trade_fig = _build_trade_diagnostics_figure(trades_local, equity_series)
    if trade_fig is not None:
        sections.append(
            _figure_to_section(
                trade_fig,
                section_id="trade-diagnostics",
                include_plotlyjs=include_plotlyjs,
                title="Trade Diagnostics",
            )
        )
        include_plotlyjs = False

    monthly_heatmap_fig = _build_monthly_heatmap_figure(equity_series)
    if monthly_heatmap_fig is not None:
        sections.append(
            _figure_to_section(
                monthly_heatmap_fig,
                section_id="monthly-returns-heatmap",
                include_plotlyjs=include_plotlyjs,
                title="Monthly Returns Heatmap",
            )
        )
        include_plotlyjs = False

    drawdown_fig = _build_drawdown_diagnostics_figure(equity_series)
    if drawdown_fig is not None:
        sections.append(
            _figure_to_section(
                drawdown_fig,
                section_id="drawdown-diagnostics",
                include_plotlyjs=include_plotlyjs,
                title="Drawdown Diagnostics",
            )
        )
        include_plotlyjs = False

    monte_carlo_fig = _build_monte_carlo_figure(monte_carlo_report)
    if monte_carlo_fig is not None:
        sections.append(
            _figure_to_section(
                monte_carlo_fig,
                section_id="monte-carlo",
                include_plotlyjs=include_plotlyjs,
                title="Monte Carlo",
            )
        )
        include_plotlyjs = False

    rolling_distribution_fig = _build_rolling_window_distribution_figure(rolling_window_report)
    if rolling_distribution_fig is not None:
        sections.append(
            _figure_to_section(
                rolling_distribution_fig,
                section_id="rolling-window-distribution",
                include_plotlyjs=include_plotlyjs,
                title="Rolling Window Distribution",
                subtitle="Histogram view across all rolling windows, matching the old RWBD-style interpretation.",
            )
        )
        include_plotlyjs = False

    rolling_timeline_fig = _build_rolling_window_timeline_figure(rolling_window_report)
    if rolling_timeline_fig is not None:
        sections.append(
            _figure_to_section(
                rolling_timeline_fig,
                section_id="rolling-window-timeline",
                include_plotlyjs=include_plotlyjs,
                title="Rolling Window Timeline",
                subtitle="The same rolling windows, ordered by their start date to show when strong or weak regimes happened.",
            )
        )
        include_plotlyjs = False

    yearly_blocks = ""
    if yearly_summary:
        yearly_row_chunks: list[str] = []
        for row in yearly_summary:
            return_pct = row.get("return_pct")
            return_text = "N/A" if return_pct is None else f"{float(return_pct):.2f}%"
            yearly_row_chunks.append(
                "<tr>"
                f"<td>{row['year']}</td>"
                f"<td>{row['start']}</td>"
                f"<td>{row['end']}</td>"
                f"<td>{row['candles']}</td>"
                f"<td>{row['trades']}</td>"
                f"<td>{return_text}</td>"
                "</tr>"
            )
        yearly_rows_html = "".join(yearly_row_chunks)
        yearly_blocks = (
            "<section class=\"report-section\" id=\"yearly-summary\">"
            "<h2>Yearly Breakdown</h2>"
            "<div class=\"report-table-wrap\">"
            "<table class=\"report-table\">"
            "<thead><tr><th>Year</th><th>Start</th><th>End</th><th>Candles</th><th>Trades</th><th>Return %</th></tr></thead>"
            f"<tbody>{yearly_rows_html}</tbody>"
            "</table></div></section>"
        )

    card_items: list[tuple[str, Any]] = [
        ("Score Context", f"{config.symbol} · {config.strategy_mode} · {config.backtest_input_mode}"),
        ("Period", f"{pd.to_datetime(df['open_time'].iloc[0]).strftime('%Y-%m-%d')} → {pd.to_datetime(df['open_time'].iloc[-1]).strftime('%Y-%m-%d')}"),
        ("Return %", f"{float(stats.get('Total Return %', 0)):.2f}%"),
        ("Final Balance", f"{float(stats.get('Final Balance', 0)):.2f}"),
        ("Trades", int(stats.get("Number of Trades", 0))),
        ("Win Rate %", f"{float(stats.get('Win Rate %', 0)):.2f}%"),
        ("Profit Factor", stats.get("Profit Factor", "N/A")),
        ("Max DD %", f"{float(stats.get('Max Drawdown %', 0)):.2f}%"),
    ]
    if monthly_return_summary is not None:
        card_items.append(("Best Month %", f"{float(monthly_return_summary['best_month_pct']):.2f}%"))
        card_items.append(("Positive Months %", f"{float(monthly_return_summary['positive_month_ratio_pct']):.2f}%"))
    if drawdown_summary is not None and drawdown_summary.get("longest_recovery_days") is not None:
        card_items.append(("Longest Recovery", f"{float(drawdown_summary['longest_recovery_days']):.2f}d"))

    stat_cards = "".join(
        "<div class=\"report-card\">"
        f"<span class=\"report-card-label\">{key}</span>"
        f"<strong>{value}</strong>"
        "</div>"
        for key, value in card_items
    )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Scrooge Backtest Report</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #0b1016;
        --panel: #101823;
        --panel-strong: #161f2b;
        --border: rgba(122, 145, 178, 0.22);
        --border-strong: rgba(189, 152, 70, 0.38);
        --text: #e9eff8;
        --muted: #a6b4c7;
        --gold: #d7a844;
        --green: #32e39b;
        --red: #ff7488;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        padding: 1.4rem;
        background: linear-gradient(180deg, #0a0f15 0%, #0c1219 100%);
        color: var(--text);
        font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      }}
      .report-shell {{
        max-width: 1480px;
        margin: 0 auto;
        display: grid;
        gap: 1rem;
      }}
      .report-hero, .report-section {{
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(15, 22, 31, 0.96), rgba(11, 17, 24, 0.98));
        padding: 1rem 1.1rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
      }}
      .report-section-collapsible > summary {{
        list-style: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: -0.1rem 0 0;
        font-size: 1.05rem;
        font-weight: 600;
      }}
      .report-section-collapsible > summary::-webkit-details-marker {{
        display: none;
      }}
      .report-collapse-hint {{
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 400;
      }}
      .report-section-collapsible:not([open]) {{
        padding-bottom: 0.9rem;
      }}
      .report-section-collapsible[open] > summary {{
        margin-bottom: 0.65rem;
      }}
      .report-hero h1, .report-section h2 {{
        margin: 0 0 0.5rem;
        font-size: 1.05rem;
      }}
      .report-subtitle {{
        margin: 0 0 0.8rem;
        color: var(--muted);
        font-size: 0.9rem;
      }}
      .report-card-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.72rem;
      }}
      .report-card {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.7rem 0.8rem;
        background: linear-gradient(180deg, rgba(24, 31, 42, 0.9), rgba(16, 22, 31, 0.96));
      }}
      .report-card-label {{
        display: block;
        margin-bottom: 0.36rem;
        color: var(--muted);
        font-size: 0.78rem;
      }}
      .report-card strong {{
        font-size: 1rem;
      }}
      .report-table-wrap {{
        overflow-x: auto;
      }}
      .report-table {{
        width: 100%;
        border-collapse: collapse;
      }}
      .report-table th,
      .report-table td {{
        padding: 0.52rem 0.6rem;
        border-bottom: 1px solid rgba(122, 145, 178, 0.14);
        text-align: left;
        white-space: nowrap;
      }}
      .report-table th {{
        color: var(--gold);
        font-weight: 600;
      }}
      .trade-ledger {{
        display: grid;
        gap: 0.7rem;
      }}
      .trade-ledger-item {{
        border: 1px solid var(--border);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(21, 27, 38, 0.92), rgba(14, 19, 27, 0.96));
        overflow: hidden;
      }}
      .trade-ledger-item--gain {{
        border-color: rgba(50, 227, 155, 0.35);
        box-shadow: inset 0 0 0 1px rgba(50, 227, 155, 0.06);
      }}
      .trade-ledger-item--loss {{
        border-color: rgba(255, 116, 136, 0.32);
        box-shadow: inset 0 0 0 1px rgba(255, 116, 136, 0.05);
      }}
      .trade-ledger-item > summary {{
        list-style: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.8rem 0.95rem;
        font-weight: 600;
      }}
      .trade-ledger-item > summary::-webkit-details-marker {{
        display: none;
      }}
      .trade-ledger-summary-side {{
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }}
      .trade-ledger-summary-pnl {{
        margin-left: auto;
        font-variant-numeric: tabular-nums;
      }}
      .trade-ledger-detail-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.65rem;
        padding: 0 0.95rem 0.95rem;
      }}
      .trade-ledger-detail {{
        border: 1px solid rgba(122, 145, 178, 0.12);
        border-radius: 10px;
        padding: 0.58rem 0.7rem;
        background: rgba(13, 18, 26, 0.72);
      }}
      .trade-ledger-detail-label {{
        display: block;
        margin-bottom: 0.32rem;
        color: var(--muted);
        font-size: 0.78rem;
      }}
      .js-plotly-plot .plotly .modebar {{
        top: 10px !important;
        right: 10px !important;
      }}
      @media (max-width: 800px) {{
        body {{ padding: 0.7rem; }}
        .report-hero, .report-section {{ padding: 0.8rem; }}
        .trade-ledger-item > summary {{
          padding: 0.72rem 0.8rem;
        }}
        .trade-ledger-detail-grid {{
          grid-template-columns: 1fr 1fr;
          padding: 0 0.8rem 0.8rem;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="report-shell">
      <section class="report-hero">
        <h1>Scrooge Backtest Report</h1>
        <p class="report-subtitle">Unified artifact for the whole run. JSON summary lives alongside this page at <code>{report_json_path.name}</code>.</p>
        <div class="report-card-grid">{stat_cards}</div>
      </section>
      {''.join(sections)}
      {yearly_blocks}
    </main>
  </body>
</html>
"""
    report_html_path.write_text(html, encoding="utf-8")
    technical_logger.info("backtest_report_written json=%s html=%s", report_json_path, report_html_path)
    return {"json": report_json_path, "html": report_html_path}


def set_client(client):
    global _client
    _client = client


def _get_client():
    if _client is None:
        raise ValueError("Binance client not initialized. Call set_client().")
    return _client


def _rw_init(
    mode,
    df,
    market_event_path,
    market_event_day_ranges,
    start_time,
    k_days,
    start_balance,
    strategy_kwargs,
    symbol,
    candle_interval,
    intervals,
    strategy_mode,
    execution_mode,
    indicator_inputs,
    runtime_mode,
):
    global _rw_df, _rw_market_event_path, _rw_market_event_day_ranges, _rw_mode, _rw_start_time, _rw_k_days, _rw_start_balance, _rw_strategy_kwargs
    global _rw_symbol, _rw_candle_interval, _rw_intervals, _rw_strategy_mode, _rw_execution_mode, _rw_indicator_inputs, _rw_runtime_mode
    _rw_df = df
    _rw_market_event_path = market_event_path
    _rw_market_event_day_ranges = market_event_day_ranges
    _rw_mode = mode
    _rw_start_time = start_time
    _rw_k_days = k_days
    _rw_start_balance = start_balance
    _rw_strategy_kwargs = strategy_kwargs
    _rw_symbol = symbol
    _rw_candle_interval = candle_interval
    _rw_intervals = intervals
    _rw_strategy_mode = strategy_mode
    _rw_execution_mode = execution_mode
    _rw_indicator_inputs = indicator_inputs
    _rw_runtime_mode = runtime_mode


def _extract_event_day(raw_line: bytes) -> str | None:
    marker = b'"ts":"'
    position = raw_line.find(marker)
    if position < 0:
        return None
    start = position + len(marker)
    end = start + 10
    if end > len(raw_line):
        return None
    try:
        return raw_line[start:end].decode("ascii")
    except UnicodeDecodeError:
        return None


def _build_market_event_day_ranges(path) -> list[tuple[str, int, int]]:
    target_path = Path(path).expanduser()
    if not target_path.exists():
        return []

    day_ranges: list[tuple[str, int, int]] = []
    current_day: str | None = None
    current_start = 0
    offset = 0
    with target_path.open("rb") as file_obj:
        while True:
            line_start = offset
            raw_line = file_obj.readline()
            if not raw_line:
                break
            offset += len(raw_line)
            if not raw_line.strip():
                continue
            event_day = _extract_event_day(raw_line)
            if event_day is None:
                continue
            if current_day is None:
                current_day = event_day
                current_start = line_start
                continue
            if event_day != current_day:
                day_ranges.append((current_day, current_start, line_start))
                current_day = event_day
                current_start = line_start

    if current_day is not None:
        day_ranges.append((current_day, current_start, offset))

    return day_ranges


def _resolve_market_event_byte_range(day_ranges, window_start, window_end):
    if not day_ranges:
        return None
    start_day = window_start.strftime("%Y-%m-%d")
    end_day = (window_end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d")
    selected_start = None
    selected_end = None
    for event_day, day_start, day_end in day_ranges:
        if event_day < start_day:
            continue
        if event_day > end_day:
            break
        if selected_start is None:
            selected_start = day_start
        selected_end = day_end
    if selected_start is None or selected_end is None:
        return None
    return selected_start, selected_end


def _iter_market_events_in_window(path, window_start, window_end, day_ranges=None):
    byte_range = _resolve_market_event_byte_range(day_ranges, window_start, window_end) if day_ranges else None
    if byte_range is not None:
        start_offset, end_offset = byte_range
        with Path(path).expanduser().open("rb") as file_obj:
            file_obj.seek(start_offset)
            while file_obj.tell() < end_offset:
                raw_line = file_obj.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                event = market_event_from_dict(payload)
                event_ts = pd.to_datetime(getattr(event, "ts", None), errors="coerce")
                if pd.isna(event_ts):
                    continue
                if event_ts < window_start:
                    continue
                if event_ts >= window_end:
                    break
                yield event
        return

    for event in iter_market_event_stream(path):
        event_ts = pd.to_datetime(getattr(event, "ts", None), errors="coerce")
        if pd.isna(event_ts):
            continue
        if event_ts < window_start:
            continue
        if event_ts >= window_end:
            break
        yield event


def _run_with_event_persistence_suppressed(callback):
    previous = os.getenv("SCROOGE_DISABLE_EVENT_PERSISTENCE")
    os.environ["SCROOGE_DISABLE_EVENT_PERSISTENCE"] = "1"
    try:
        return callback()
    finally:
        if previous is None:
            os.environ.pop("SCROOGE_DISABLE_EVENT_PERSISTENCE", None)
        else:
            os.environ["SCROOGE_DISABLE_EVENT_PERSISTENCE"] = previous


def _run_window_backtest(
    *,
    mode,
    df,
    market_event_path,
    market_event_day_ranges,
    window_start,
    window_end,
    start_balance,
    strategy_kwargs,
    symbol,
    candle_interval,
    intervals,
    strategy_mode,
    execution_mode,
    indicator_inputs,
    runtime_mode,
):
    if mode == "market_events":
        final_balance, trades, balance_history, _ = _run_with_event_persistence_suppressed(
            lambda: run_strategy_on_market_events(
                _iter_market_events_in_window(
                    market_event_path,
                    window_start,
                    window_end,
                    day_ranges=market_event_day_ranges,
                ),
                candle_interval=candle_interval,
                intervals=intervals,
                market_event_total=None,
                strategy_mode=strategy_mode,
                execution_mode=execution_mode,
                strict_indicator_alignment=False,
                show_progress=False,
                enable_logs=False,
                use_state=False,
                runtime_mode=runtime_mode,
                indicator_inputs=indicator_inputs,
                initial_balance=start_balance,
                **strategy_kwargs,
            )
        )
    else:
        df_window = df[(df["open_time"] >= window_start) & (df["open_time"] < window_end)]
        if df_window.empty:
            return None
        tape_window = build_discrete_market_tape(df_window, symbol=symbol)
        if not tape_window:
            return None
        final_balance, trades, balance_history, _ = _run_with_event_persistence_suppressed(
            lambda: run_strategy_on_tape(
                tape_window,
                show_progress=False,
                enable_logs=False,
                use_state=False,
                runtime_mode=runtime_mode,
                indicator_inputs=indicator_inputs,
                initial_balance=start_balance,
                **strategy_kwargs,
            )
        )

    stats = compute_stats(start_balance, final_balance, trades, balance_history)
    return (
        final_balance,
        float(stats["Total Return %"]),
        float(stats["Win Rate %"]),
        float(stats["Max Drawdown %"]),
    )


def _rw_worker(offset):
    window_start = _rw_start_time + pd.Timedelta(days=offset)
    window_end = window_start + pd.Timedelta(days=_rw_k_days)
    return offset, _run_window_backtest(
        mode=_rw_mode,
        df=_rw_df,
        market_event_path=_rw_market_event_path,
        market_event_day_ranges=_rw_market_event_day_ranges,
        window_start=window_start,
        window_end=window_end,
        start_balance=_rw_start_balance,
        strategy_kwargs=_rw_strategy_kwargs,
        symbol=_rw_symbol,
        candle_interval=_rw_candle_interval,
        intervals=_rw_intervals,
        strategy_mode=_rw_strategy_mode,
        execution_mode=_rw_execution_mode,
        indicator_inputs=_rw_indicator_inputs,
        runtime_mode=_rw_runtime_mode,
    )

def load_state():
    state = load_runtime_state(include_history=False)
    state["trade_history"] = load_trade_history()
    state["balance_history"] = load_balance_history()
    return state

def fetch_session_klines(symbol, interval, start_ts, end_ts):
    """Fetch historical klines from Binance for session period."""
    client = _get_client()
    klines = run_binance_with_retries(
        lambda: client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=1500,
        ),
        operation_name=(
            f"reporting_futures_klines symbol={symbol} interval={interval} "
            f"start_ts={start_ts} end_ts={end_ts}"
        ),
        logger=technical_logger,
    )
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])
    return df[["open_time","close"]]

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
        technical_logger.warning("session_plot_missing_timestamps")
        return

    # --- Fetch session price data ---
    df_small = fetch_historical_paginated(symbol, "1m", session_start, session_end)
    df_medium = fetch_historical_paginated(symbol, "1h", session_start, session_end)
    df_big = fetch_historical_paginated(symbol, "4h", session_start, session_end)
    df = prepare_multi_tf(df_small, df_medium, df_big)
    if df.empty:
        technical_logger.warning("session_plot_no_klines symbol=%s", symbol)
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
    technical_logger.info("session_report_plotted symbol=%s", symbol)

    # --- Compute and print session stats ---
    stats = compute_session_stats(trades, balance_history)
    for k, v in stats.items():
        technical_logger.info("session_stat %s=%s", k, v)


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
    if "BBL" in df.columns:
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
    if "BBM" in df.columns:
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
    if "BBU" in df.columns:
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
    if "EMA" in df.columns:
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
        technical_logger.warning("interactive_plot_no_data")
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


def run_monte_carlo(df, balance_history, start_balance=10000, sims=10000, horizon_months=None, block_len=3, show_plot=False):
    """
    Advanced Monte Carlo stress test based on monthly equity returns.
    Includes CAGR, Volatility and Sharpe Ratio metrics.
    """

    # --- 1. Generate synthetic datetime index for the equity curve ---
    if len(balance_history) == 0:
        technical_logger.warning("monte_carlo_no_equity_data")
        return {}

    start_time = pd.to_datetime(df["open_time"].iloc[0])
    end_time = pd.to_datetime(df["open_time"].iloc[-1])
    synthetic_index = pd.date_range(start=start_time, end=end_time, periods=len(balance_history))

    equity = pd.Series(balance_history, index=synthetic_index)

    # --- 2. Resample monthly and compute returns ---
    eq_monthly = equity.resample("ME").last().dropna()
    monthly_returns = eq_monthly.pct_change().dropna()

    if len(monthly_returns) < 3:
        technical_logger.warning("monte_carlo_not_enough_monthly_data points=%s", len(monthly_returns))
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

    report = {
        "summary": summary,
        "percentiles": {
            "p5": round(float(p5), 2),
            "p50": round(float(p50), 2),
            "p95": round(float(p95), 2),
        },
        "plot_data": {
            "final_balances": results.tolist(),
            "monthly_returns": monthly_returns.tolist(),
            "horizon_months": int(horizon_months),
        },
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

    for k, v in summary.items():
        technical_logger.info("monte_carlo_stat %s=%s", k, v)

    return report


def monte_carlo_from_equity(df, balance_history, start_balance=10000, sims=10000, horizon_months=None, block_len=3, show_plot=False):
    return run_monte_carlo(
        df,
        balance_history,
        start_balance=start_balance,
        sims=sims,
        horizon_months=horizon_months,
        block_len=block_len,
        show_plot=show_plot,
    )


def run_rolling_window_backtest_distribution(
    config: "BacktestConfig",
    df: pd.DataFrame,
    *,
    k_days: int,
    n_days: int | None = None,
    start_balance: float = 10000,
    show_plot: bool = False,
    max_workers: int | None = None,
):
    return rolling_window_backtest_distribution(
        df,
        k_days=k_days,
        n_days=n_days,
        start_balance=start_balance,
        show_plot=show_plot,
        max_workers=max_workers,
        config=config,
    )


def rolling_window_backtest_distribution(
    df,
    k_days,
    n_days=None,
    start_balance=10000,
    show_plot=False,
    max_workers=None,
    config: "BacktestConfig | None" = None,
    **strategy_kwargs
):
    """
    Run rolling k-day backtests over an n-day period and return a distribution
    of final balances, similar to the Monte Carlo view.
    """
    if df is None or df.empty:
        technical_logger.warning("rolling_window_no_data")
        return {}

    if k_days <= 0:
        technical_logger.warning("rolling_window_invalid_k_days value=%s", k_days)
        return {}

    start_time = pd.to_datetime(df["open_time"].iloc[0])
    end_time = pd.to_datetime(df["open_time"].iloc[-1])
    total_days = int((end_time - start_time).total_seconds() // 86400)

    if n_days is None:
        n_days = total_days
    n_days = min(n_days, total_days)

    if n_days < k_days:
        technical_logger.warning("rolling_window_invalid_n_days n_days=%s k_days=%s", n_days, k_days)
        return {}

    strategy_kwargs = dict(strategy_kwargs)
    rolling_mode = "discrete_tape"
    market_event_path = None
    market_event_day_ranges = None
    candle_interval = "1m"
    intervals = {
        "small": "1m",
        "medium": "1h",
        "big": "4h",
    }
    strategy_mode = "discrete"
    execution_mode = "simulated"
    indicator_inputs = None
    runtime_mode = "backtest"
    symbol = str(strategy_kwargs.get("symbol", "BTCUSDT"))
    indicator_inputs = strategy_kwargs.pop("indicator_inputs", None)
    runtime_mode = str(strategy_kwargs.pop("runtime_mode", runtime_mode) or runtime_mode)
    strategy_mode = str(strategy_kwargs.pop("strategy_mode", strategy_mode) or strategy_mode)
    execution_mode = str(strategy_kwargs.pop("execution_mode", execution_mode) or execution_mode)
    strategy_kwargs.pop("initial_balance", None)
    strategy_kwargs.pop("show_progress", None)
    strategy_kwargs.pop("enable_logs", None)
    strategy_kwargs.pop("use_state", None)
    strategy_kwargs.pop("live", None)

    if config is not None:
        symbol = config.symbol
        candle_interval = str(config.intervals["small"])
        intervals = {key: str(value) for key, value in config.intervals.items()}
        strategy_mode = config.strategy_mode
        execution_mode = config.execution_mode
        indicator_inputs = config.indicator_inputs
        runtime_mode = config.runtime_mode
        strategy_kwargs = {
            "qty": config.qty,
            "symbol": config.symbol,
            "leverage": config.leverage,
            "use_full_balance": config.use_full_balance,
            **config.params,
        }
        if (
            config.strategy_mode == "realtime"
            or uses_realtime_indicator_inputs(config.indicator_inputs)
            or config.backtest_input_mode in {"market_event_stream", "agg_trade_stream"}
        ):
            market_event_path = config.market_event_stream_path
            if not market_event_path.exists():
                technical_logger.warning(
                    "rolling_window_missing_market_events path=%s",
                    market_event_path,
                )
                return {}
            rolling_mode = "market_events"
            market_event_day_ranges = _build_market_event_day_ranges(market_event_path)
            technical_logger.info(
                "rolling_window_market_event_day_index_ready days=%s path=%s",
                len(market_event_day_ranges),
                market_event_path,
            )

    final_balances = []
    pnl_pcts = []
    win_rates = []
    max_drawdowns = []
    window_starts = []

    total_windows = n_days - k_days + 1
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1 or total_windows <= 1:
        for offset in tqdm(range(total_windows), desc="Rolling Backtest", ncols=100):
            window_start = start_time + pd.Timedelta(days=offset)
            window_end = window_start + pd.Timedelta(days=k_days)
            result = _run_window_backtest(
                mode=rolling_mode,
                df=df,
                market_event_path=market_event_path,
                market_event_day_ranges=market_event_day_ranges,
                window_start=window_start,
                window_end=window_end,
                start_balance=start_balance,
                strategy_kwargs=strategy_kwargs,
                symbol=symbol,
                candle_interval=candle_interval,
                intervals=intervals,
                strategy_mode=strategy_mode,
                execution_mode=execution_mode,
                indicator_inputs=indicator_inputs,
                runtime_mode=runtime_mode,
            )
            if result is None:
                continue
            final_balance, pnl_pct, win_rate, max_drawdown = result
            window_starts.append(window_start)
            final_balances.append(final_balance)
            pnl_pcts.append(pnl_pct)
            win_rates.append(win_rate)
            max_drawdowns.append(max_drawdown)
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_rw_init,
            initargs=(
                rolling_mode,
                None if rolling_mode == "market_events" else df,
                market_event_path,
                market_event_day_ranges,
                start_time,
                k_days,
                start_balance,
                strategy_kwargs,
                symbol,
                candle_interval,
                intervals,
                strategy_mode,
                execution_mode,
                indicator_inputs,
                runtime_mode,
            ),
        ) as executor:
            futures = [executor.submit(_rw_worker, offset) for offset in range(total_windows)]
            for fut in tqdm(as_completed(futures), total=total_windows, desc="Rolling Backtest", ncols=100):
                offset, result = fut.result()
                if result is None:
                    continue
                final_balance, pnl_pct, win_rate, max_drawdown = result
                window_starts.append(start_time + pd.Timedelta(days=offset))
                final_balances.append(final_balance)
                pnl_pcts.append(pnl_pct)
                win_rates.append(win_rate)
                max_drawdowns.append(max_drawdown)

    if not final_balances:
        technical_logger.warning("rolling_window_no_results")
        return {}

    order = np.argsort(pd.to_datetime(window_starts).astype("int64"))
    window_starts = [window_starts[index] for index in order]
    final_balances = [final_balances[index] for index in order]
    pnl_pcts = [pnl_pcts[index] for index in order]
    win_rates = [win_rates[index] for index in order]
    max_drawdowns = [max_drawdowns[index] for index in order]

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

    report = {
        "summary": summary,
        "percentiles": {
            "final_balance": {"p5": round(float(p5), 2), "p50": round(float(p50), 2), "p95": round(float(p95), 2)},
            "pnl_pct": {"p5": round(float(pnl_p5), 2), "p50": round(float(pnl_p50), 2), "p95": round(float(pnl_p95), 2)},
            "win_rate_pct": {"p5": round(float(wr_p5), 2), "p50": round(float(wr_p50), 2), "p95": round(float(wr_p95), 2)},
            "max_drawdown_pct": {"p5": round(float(dd_p5), 2), "p50": round(float(dd_p50), 2), "p95": round(float(dd_p95), 2)},
        },
        "plot_data": {
            "window_starts": [timestamp.isoformat() for timestamp in window_starts],
            "final_balances": [float(value) for value in final_balances],
            "pnl_pcts": [float(value) for value in pnl_pcts],
            "win_rates": [float(value) for value in win_rates],
            "max_drawdowns": [float(value) for value in max_drawdowns],
        },
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

    for k, v in summary.items():
        technical_logger.info("rolling_window_stat %s=%s", k, v)

    return report


if __name__ == "__main__":
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        set_client(create_binance_client(api_key, api_secret, logger=technical_logger))
    state = load_state()
    if state:
        plot_session(state)
    else:
        technical_logger.warning("session_plot_no_state")
