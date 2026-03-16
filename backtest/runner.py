from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import backtest.dataset as dataset_module
from backtest.discrete_event_stream import (
    read_discrete_market_event_stream,
    write_discrete_market_event_stream,
)
from backtest.market_event_projection import (
    DiscreteTapeProjectionSummary,
    read_projected_discrete_tape_from_market_event_stream,
)
from backtest.historical_market_event_stream import build_historical_market_event_stream
from backtest.discrete_tape import (
    DiscreteMarketTapeRow,
    build_discrete_market_tape,
    discrete_market_tape_to_frame,
    read_discrete_market_tape,
    write_discrete_market_tape,
)
from backtest.dataset import build_dataset
from backtest.market_event_replay import (
    MarketExecutionSummary,
    write_market_event_execution_artifacts,
)
from backtest.replay import ReplaySummary, write_replay_artifacts
from backtest.trade_alignment import TradeAlignmentSummary, write_trade_alignment_artifacts
from bot.event_log import get_technical_logger
from bot.state import save_state
from core.engine import run_strategy_on_market_events, run_strategy_on_tape
from core.market_events import MarketEvent, market_event_from_dict


@dataclass(slots=True)
class DiscreteBacktestConfig:
    backtest_input_mode: str
    execution_mode: str
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
    market_event_stream_path: Path
    market_tape_input_path: Path | None
    market_event_input_path: Path | None
    runtime_mode: str
    strategy_mode: str
    client: Any | None = None


@dataclass(slots=True)
class DiscreteBacktestResult:
    dataset: pd.DataFrame
    tape: list[DiscreteMarketTapeRow]
    market_events: list[MarketEvent]
    market_event_projection: DiscreteTapeProjectionSummary | None
    market_event_execution_summary: MarketExecutionSummary | None
    trade_alignment_summary: TradeAlignmentSummary | None
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

    input_mode = str(cfg.get("backtest_input_mode", "build")).strip().lower() or "build"
    if input_mode == "tape":
        input_mode = "discrete_tape"
    if input_mode == "event_stream":
        input_mode = "market_event_stream"
    if input_mode == "discrete_event_stream":
        input_mode = "market_event_stream"
    if input_mode not in {"build", "discrete_tape", "market_event_stream"}:
        raise ValueError("backtest_input_mode must be one of: build, discrete_tape, market_event_stream")
    normalized_strategy_mode = str(cfg.get("strategy_mode", strategy_mode)).strip().lower() or strategy_mode
    if normalized_strategy_mode not in {"discrete", "realtime"}:
        raise ValueError("strategy_mode must be one of: discrete, realtime")
    execution_mode = str(cfg.get("execution_mode", "simulated")).strip().lower() or "simulated"
    if execution_mode not in {"simulated", "observed"}:
        raise ValueError("execution_mode must be one of: simulated, observed")
    if execution_mode == "observed" and input_mode != "market_event_stream":
        raise ValueError("execution_mode=observed requires backtest_input_mode=market_event_stream")

    raw_input_path = str(cfg.get("market_tape_input_path", "") or "").strip()
    market_tape_input_path = Path(raw_input_path).expanduser() if raw_input_path else None
    raw_market_event_input_path = str(cfg.get("market_event_input_path", "") or "").strip()
    market_event_input_path = Path(raw_market_event_input_path).expanduser() if raw_market_event_input_path else None

    return DiscreteBacktestConfig(
        backtest_input_mode=input_mode,
        execution_mode=execution_mode,
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
        market_event_stream_path=Path(event_log_path).expanduser().with_name("market_events.jsonl"),
        market_tape_input_path=market_tape_input_path,
        market_event_input_path=market_event_input_path,
        runtime_mode=runtime_mode,
        strategy_mode=normalized_strategy_mode,
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
    market_event_strategy_runner: Callable[..., tuple[float, pd.DataFrame, list[float], dict[str, Any]]] = run_strategy_on_market_events,
    replay_writer: Callable[..., ReplaySummary] = write_replay_artifacts,
    market_event_execution_writer: Callable[..., MarketExecutionSummary] = write_market_event_execution_artifacts,
    trade_alignment_writer: Callable[..., TradeAlignmentSummary] = write_trade_alignment_artifacts,
) -> DiscreteBacktestResult:
    logger = technical_logger or get_technical_logger()
    market_event_projection: DiscreteTapeProjectionSummary | None = None

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
    if config.backtest_input_mode == "discrete_tape":
        source_path = config.market_tape_input_path or config.market_tape_path
        tape = read_discrete_market_tape(source_path)
        if not tape:
            raise ValueError(f"No market tape rows found at {source_path}")
        df = discrete_market_tape_to_frame(tape)
        if df.empty:
            raise ValueError(f"Market tape frame is empty at {source_path}")
        logger.info("backtest_market_tape_loaded rows=%s path=%s", len(tape), source_path)
        if source_path != config.market_tape_path:
            write_discrete_market_tape(config.market_tape_path, tape)
            logger.info("backtest_market_tape_copied rows=%s path=%s", len(tape), config.market_tape_path)
        market_events = build_historical_market_event_stream(
            tape,
            intervals={key: str(value) for key, value in config.intervals.items()},
        )
        write_discrete_market_event_stream(config.market_event_stream_path, market_events)
        logger.info(
            "backtest_historical_market_event_stream_written events=%s path=%s",
            len(market_events),
            config.market_event_stream_path,
        )
    elif config.backtest_input_mode == "market_event_stream":
        source_path = config.market_event_input_path or config.market_event_stream_path
        market_events = read_discrete_market_event_stream(source_path)
        if not market_events:
            raise ValueError(f"No market events found at {source_path}")
        tape, market_event_projection = read_projected_discrete_tape_from_market_event_stream(
            source_path,
            candle_interval=str(config.intervals["small"]),
            symbol=config.symbol,
        )
        if not tape:
            raise ValueError(f"Market event stream produced no discrete tape rows at {source_path}")
        df = discrete_market_tape_to_frame(tape)
        logger.info("backtest_market_event_stream_loaded events=%s path=%s", len(market_events), source_path)
        if market_event_projection is not None:
            logger.info(
                "backtest_market_event_projection rows=%s candles=%s indicator_snapshots=%s ignored=%s",
                market_event_projection.projected_rows,
                market_event_projection.matched_candles,
                market_event_projection.matched_indicator_snapshots,
                market_event_projection.ignored_events,
            )
        if source_path != config.market_event_stream_path:
            write_discrete_market_event_stream(config.market_event_stream_path, market_events)
            logger.info(
                "backtest_market_event_stream_copied events=%s path=%s",
                len(market_events),
                config.market_event_stream_path,
            )
        write_discrete_market_tape(config.market_tape_path, tape)
        logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)
    else:
        df = dataset_builder(
            symbol=config.symbol,
            intervals=config.intervals,
            backtest_period_days=config.backtest_period_days,
            backtest_period_end_time=config.backtest_period_end_time,
        )
        tape = build_discrete_market_tape(df, symbol=config.symbol)
        write_discrete_market_tape(config.market_tape_path, tape)
        logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)
        market_events = build_historical_market_event_stream(
            tape,
            intervals={key: str(value) for key, value in config.intervals.items()},
        )
        write_discrete_market_event_stream(config.market_event_stream_path, market_events)
        logger.info(
            "backtest_historical_market_event_stream_written events=%s path=%s",
            len(market_events),
            config.market_event_stream_path,
        )

    strategy_kwargs = {
        "live": False,
        "initial_balance": config.initial_balance,
        "qty": config.qty,
        "symbol": config.symbol,
        "leverage": config.leverage,
        "use_full_balance": config.use_full_balance,
        "use_state": False,
        **config.params,
    }
    if config.strategy_mode == "realtime" or config.backtest_input_mode == "market_event_stream":
        final_balance, trades, balance_history, state = market_event_strategy_runner(
            market_events,
            candle_interval=str(config.intervals["small"]),
            intervals={key: str(value) for key, value in config.intervals.items()},
            strategy_mode=config.strategy_mode,
            execution_mode=config.execution_mode,
            strict_indicator_alignment=False,
            **strategy_kwargs,
        )
    else:
        final_balance, trades, balance_history, state = strategy_runner(
            tape,
            **strategy_kwargs,
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
    execution_artifact_events = market_events
    if config.execution_mode == "simulated":
        raw_simulated_execution_events = state.get("simulated_execution_events")
        if isinstance(raw_simulated_execution_events, list):
            execution_artifact_events = [
                market_event_from_dict(payload)
                for payload in raw_simulated_execution_events
                if isinstance(payload, dict)
            ]
    market_event_execution_summary = market_event_execution_writer(
        execution_artifact_events,
        symbol=config.symbol,
        summary_path=config.event_log_path.with_name("market_event_execution_summary.json"),
        events_path=config.event_log_path.with_name("market_event_execution_events.jsonl"),
        fills_path=config.event_log_path.with_name("market_event_execution_fills.jsonl"),
        trades_path=config.event_log_path.with_name("market_event_execution_trades.jsonl"),
    )
    logger.info(
        "backtest_market_event_execution_artifacts_written execution_events=%s order_updates=%s filled_orders=%s observed_fills=%s observed_trades=%s realized_pnl=%.8f commission=%.8f path=%s",
        market_event_execution_summary.execution_events,
        market_event_execution_summary.order_trade_update_events,
        market_event_execution_summary.filled_order_events,
        market_event_execution_summary.observed_fill_events,
        market_event_execution_summary.observed_total_trades,
        market_event_execution_summary.realized_pnl_total,
        market_event_execution_summary.commission_total,
        config.event_log_path.with_name("market_event_execution_summary.json"),
    )
    trade_alignment_summary: TradeAlignmentSummary | None = None
    if market_event_execution_summary.observed_total_trades > 0:
        trade_alignment_summary = trade_alignment_writer(
            trades,
            execution_artifact_events,
            symbol=config.symbol,
            summary_path=config.event_log_path.with_name("market_event_trade_alignment_summary.json"),
            pairs_path=config.event_log_path.with_name("market_event_trade_alignment_pairs.jsonl"),
        )
        logger.info(
            "backtest_market_event_trade_alignment_written paired=%s strategy_closed=%s observed_closed=%s side_mismatches=%s pnl_delta=%.8f path=%s",
            trade_alignment_summary.paired_trades,
            trade_alignment_summary.strategy_closed_trades,
            trade_alignment_summary.observed_closed_trades,
            trade_alignment_summary.side_mismatches,
            trade_alignment_summary.pnl_delta,
            config.event_log_path.with_name("market_event_trade_alignment_summary.json"),
        )
    execution_sync = state.get("execution_sync")
    if isinstance(execution_sync, dict) and execution_sync:
        logger.info(
            "backtest_market_event_runtime_sync total=%s balances=%s positions=%s orders=%s trade_updates=%s filled=%s balance_alignment=%s balance_drift=%s position_alignment=%s divergence_events=%s",
            execution_sync.get("total_events"),
            execution_sync.get("account_balance_events"),
            execution_sync.get("position_snapshot_events"),
            execution_sync.get("order_trade_update_events"),
            execution_sync.get("trade_execution_events"),
            execution_sync.get("filled_order_events"),
            execution_sync.get("balance_alignment"),
            execution_sync.get("balance_drift"),
            execution_sync.get("position_alignment"),
            execution_sync.get("divergence_events"),
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
        market_events=market_events,
        market_event_projection=market_event_projection,
        market_event_execution_summary=market_event_execution_summary,
        trade_alignment_summary=trade_alignment_summary,
        final_balance=float(final_balance),
        trades=trades,
        balance_history=balance_history,
        state=state,
        replay_summary=replay_summary,
        stats=stats,
    )
