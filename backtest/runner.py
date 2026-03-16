from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import backtest.dataset as dataset_module
from backtest.dataset import build_dataset
from backtest.replay import ReplaySummary, write_replay_artifacts
from backtest.tape import DiscreteMarketTapeRow, build_discrete_market_tape, write_discrete_market_tape
from bot.event_log import get_technical_logger
from bot.state import save_state
from core.engine import run_strategy_on_tape


@dataclass(slots=True)
class DiscreteBacktestConfig:
    symbol: str
    leverage: float
    initial_balance: float
    qty: float | None
    use_full_balance: bool
    intervals: dict[str, Any]
    params: dict[str, Any]
    backtest_period_days: int
    backtest_period_end_time: str
    enable_plot: bool
    plot_split_by_year: bool
    run_monte_carlo: bool
    run_rolling_window_backtest_distribution: bool
    rolling_window_days: int
    rolling_window_workers: int | None
    chart_dataset_path: Path
    event_log_path: Path
    market_tape_path: Path
    runtime_mode: str
    strategy_mode: str
    client: Any | None = None


@dataclass(slots=True)
class DiscreteBacktestResult:
    dataset: pd.DataFrame
    tape: list[DiscreteMarketTapeRow]
    final_balance: float
    trades: pd.DataFrame
    balance_history: list[float]
    state: dict[str, Any]
    replay_summary: ReplaySummary
    stats: dict[str, Any]


def build_discrete_backtest_config(
    cfg: dict[str, Any],
    *,
    chart_dataset_path: str | Path,
    event_log_path: str | Path,
    runtime_mode: str = "backtest",
    strategy_mode: str = "discrete",
    client: Any | None = None,
) -> DiscreteBacktestConfig:
    if bool(cfg.get("live", False)):
        raise ValueError("DiscreteBacktestConfig requires a backtest config (live=false).")

    initial_balance = cfg.get("initial_balance")
    if initial_balance is None:
        raise ValueError("Backtest config must include initial_balance.")

    return DiscreteBacktestConfig(
        symbol=str(cfg["symbol"]),
        leverage=float(cfg["leverage"]),
        initial_balance=float(initial_balance),
        qty=cfg.get("qty"),
        use_full_balance=bool(cfg["use_full_balance"]),
        intervals=dict(cfg["intervals"]),
        params=dict(cfg.get("params", {})),
        backtest_period_days=int(cfg["backtest_period_days"]),
        backtest_period_end_time=str(cfg.get("backtest_period_end_time", "")),
        enable_plot=bool(cfg["enable_plot"]),
        plot_split_by_year=bool(cfg.get("plot_split_by_year", True)),
        run_monte_carlo=bool(cfg["run_monte_carlo"]),
        run_rolling_window_backtest_distribution=bool(cfg["run_rolling_window_backtest_distribution"]),
        rolling_window_days=int(cfg["rolling_window_days"]),
        rolling_window_workers=cfg.get("rolling_window_workers"),
        chart_dataset_path=Path(chart_dataset_path).expanduser(),
        event_log_path=Path(event_log_path).expanduser(),
        market_tape_path=Path(event_log_path).expanduser().with_name("market_tape.jsonl"),
        runtime_mode=runtime_mode,
        strategy_mode=strategy_mode,
        client=client,
    )


def _format_open_time(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        dt_value = value.tz_convert(None) if value.tzinfo is not None else value
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, datetime):
        dt_value = value.replace(tzinfo=None) if value.tzinfo is not None else value
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")

    text = str(value).strip()
    return text or datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _coerce_optional_float(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(numeric):
        return ""
    return numeric


def _compress_balance_history_for_state(balance_history: list[float] | None) -> list[float]:
    if not balance_history:
        return []

    compressed: list[float] = []
    last_value: float | None = None
    for item in balance_history:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if pd.isna(value):
            continue
        if last_value is None or abs(value - last_value) > 1e-9:
            compressed.append(value)
            last_value = value
    return compressed


def _write_chart_dataset_snapshot(df: pd.DataFrame, symbol: str, path: Path, balance_history: list[float] | None = None) -> None:
    if df is None or len(df) == 0:  # noqa: PLR2004
        return

    required = {"open_time", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        get_technical_logger().warning("chart_dataset_snapshot_skipped_missing_columns path=%s", path)
        return

    aligned_balance_history: list[float | str] = []
    if balance_history:
        clean_history: list[float] = []
        for value in balance_history:
            try:
                clean_history.append(float(value))
            except (TypeError, ValueError):
                continue
        if clean_history:
            rows_count = len(df)
            history_count = len(clean_history)
            if history_count >= rows_count:
                aligned_balance_history = clean_history[-rows_count:]
            else:
                aligned_balance_history = ([""] * (rows_count - history_count)) + clean_history

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                ["open_time", "symbol", "open", "high", "low", "close", "volume", "balance", "EMA", "RSI", "BBL", "BBM", "BBU", "ATR"]
            )
            for idx, row in enumerate(df.itertuples(index=False)):
                balance_value: float | str = ""
                if aligned_balance_history and idx < len(aligned_balance_history):
                    balance_value = aligned_balance_history[idx]
                writer.writerow(
                    [
                        _format_open_time(getattr(row, "open_time", None)),
                        symbol,
                        float(getattr(row, "open")),
                        float(getattr(row, "high")),
                        float(getattr(row, "low")),
                        float(getattr(row, "close")),
                        float(getattr(row, "volume")),
                        balance_value,
                        _coerce_optional_float(getattr(row, "EMA", None)),
                        _coerce_optional_float(getattr(row, "RSI", None)),
                        _coerce_optional_float(getattr(row, "BBL", None)),
                        _coerce_optional_float(getattr(row, "BBM", None)),
                        _coerce_optional_float(getattr(row, "BBU", None)),
                        _coerce_optional_float(getattr(row, "ATR", None)),
                    ]
                )
    except OSError as exc:
        get_technical_logger().warning("chart_dataset_snapshot_write_failed path=%s error=%s", path, exc)


def run_discrete_backtest(
    config: DiscreteBacktestConfig,
    *,
    technical_logger: Any | None = None,
    dataset_builder: Callable[..., pd.DataFrame] = build_dataset,
    strategy_runner: Callable[..., tuple[float, pd.DataFrame, list[float], dict[str, Any]]] = run_strategy_on_tape,
    replay_writer: Callable[..., ReplaySummary] = write_replay_artifacts,
) -> DiscreteBacktestResult:
    logger = technical_logger or get_technical_logger()

    import backtest.reporting as report_module
    from backtest.reporting import (
        compute_stats,
        monte_carlo_from_equity,
        plot_results_interactive,
        rolling_window_backtest_distribution,
    )

    if config.client is not None:
        dataset_module.set_client(config.client)
        report_module.set_client(config.client)

    logger.info("bot_mode_backtest_started symbol=%s leverage=%s", config.symbol, config.leverage)
    df = dataset_builder(
        symbol=config.symbol,
        intervals=config.intervals,
        backtest_period_days=config.backtest_period_days,
        backtest_period_end_time=config.backtest_period_end_time,
    )
    tape = build_discrete_market_tape(df, symbol=config.symbol)
    write_discrete_market_tape(config.market_tape_path, tape)
    logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)

    final_balance, trades, balance_history, state = strategy_runner(
        tape,
        live=False,
        initial_balance=config.initial_balance,
        qty=config.qty,
        symbol=config.symbol,
        leverage=config.leverage,
        use_full_balance=config.use_full_balance,
        use_state=False,
        **config.params,
    )

    state["balance_history"] = _compress_balance_history_for_state(balance_history)
    state["balance"] = float(final_balance)
    if len(state["balance_history"]) != len(balance_history):
        logger.info(
            "backtest_balance_history_compressed original=%s persisted=%s",
            len(balance_history),
            len(state["balance_history"]),
        )

    save_state(state)
    _write_chart_dataset_snapshot(
        df=df,
        symbol=config.symbol,
        path=config.chart_dataset_path,
        balance_history=balance_history,
    )
    replay_summary = replay_writer(
        config.event_log_path,
        runtime_mode=config.runtime_mode,
        strategy_mode=config.strategy_mode,
        symbol=config.symbol,
    )
    logger.info(
        "backtest_replay_artifacts_written trades=%s net_pnl=%.8f path=%s",
        replay_summary.closed_trades,
        replay_summary.net_pnl,
        config.event_log_path,
    )

    stats = compute_stats(config.initial_balance, final_balance, trades, balance_history)
    for key, value in stats.items():
        logger.info("backtest_stat %s=%s", key, value)

    if config.enable_plot:
        plot_results_interactive(df, trades, balance_history, split_by_year=config.plot_split_by_year)

    if config.run_monte_carlo:
        monte_carlo_from_equity(df, balance_history, start_balance=config.initial_balance)

    if config.run_rolling_window_backtest_distribution:
        rolling_window_backtest_distribution(
            df,
            k_days=config.rolling_window_days,
            n_days=config.backtest_period_days,
            start_balance=config.initial_balance,
            max_workers=config.rolling_window_workers,
            leverage=config.leverage,
            **config.params,
        )

    return DiscreteBacktestResult(
        dataset=df,
        tape=tape,
        final_balance=float(final_balance),
        trades=trades,
        balance_history=balance_history,
        state=state,
        replay_summary=replay_summary,
        stats=stats,
    )
