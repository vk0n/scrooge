from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import backtest.dataset as dataset_module
from backtest.agg_trade_market_event_stream import (
    AggTradeMarketEventStreamSummary,
    write_historical_agg_trade_market_event_stream,
)
from backtest.discrete_event_stream import (
    write_discrete_market_event_stream,
)
from backtest.market_event_projection import (
    DiscreteTapeProjectionSummary,
    project_discrete_tape_from_market_events,
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
from backtest.progress import BacktestProgressReporter
from backtest.replay import ReplaySummary, write_replay_artifacts
from backtest.stats import compute_stats
from backtest.trade_alignment import TradeAlignmentSummary, write_trade_alignment_artifacts
from bot.event_log import get_technical_logger
from bot.state import save_state
from core.engine import run_strategy_on_market_events, run_strategy_on_tape
from core.market_events import MarketEvent, count_market_event_stream, iter_market_event_stream, market_event_from_dict


@dataclass(slots=True)
class BacktestConfig:
    backtest_input_mode: str
    execution_mode: str
    agg_trade_source: str
    agg_trade_tick_interval: str
    agg_trade_archive_base_url: str
    agg_trade_rest_base_url: str
    agg_trade_cache_enabled: bool
    agg_trade_cache_dir: Path
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
class BacktestResult:
    dataset: pd.DataFrame
    tape: list[DiscreteMarketTapeRow]
    market_events: list[MarketEvent] | None
    market_event_projection: DiscreteTapeProjectionSummary | None
    agg_trade_stream_summary: AggTradeMarketEventStreamSummary | None
    market_event_execution_summary: MarketExecutionSummary | None
    trade_alignment_summary: TradeAlignmentSummary | None
    final_balance: float
    trades: pd.DataFrame
    balance_history: list[float]
    state: dict[str, Any]
    replay_summary: ReplaySummary
    stats: dict[str, Any]


def build_backtest_config(
    cfg: dict[str, Any],
    *,
    chart_dataset_path: str | Path,
    event_log_path: str | Path,
    runtime_mode: str = "backtest",
    strategy_mode: str | None = None,
    client: Any | None = None,
) -> BacktestConfig:
    if bool(cfg.get("live", False)):
        raise ValueError("BacktestConfig requires a backtest config (live=false).")

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
    if input_mode not in {"build", "discrete_tape", "market_event_stream", "agg_trade_stream"}:
        raise ValueError("backtest_input_mode must be one of: build, discrete_tape, market_event_stream, agg_trade_stream")
    fallback_strategy_mode = str(strategy_mode or "discrete").strip().lower() or "discrete"
    normalized_strategy_mode = str(cfg.get("strategy_mode", fallback_strategy_mode)).strip().lower() or fallback_strategy_mode
    if normalized_strategy_mode not in {"discrete", "realtime"}:
        raise ValueError("strategy_mode must be one of: discrete, realtime")
    execution_mode = str(cfg.get("execution_mode", "simulated")).strip().lower() or "simulated"
    if execution_mode not in {"simulated", "observed"}:
        raise ValueError("execution_mode must be one of: simulated, observed")
    if execution_mode == "observed" and input_mode != "market_event_stream":
        raise ValueError("execution_mode=observed requires backtest_input_mode=market_event_stream")
    agg_trade_source = str(cfg.get("agg_trade_source", "archive")).strip().lower() or "archive"
    if agg_trade_source not in {"archive", "rest", "auto"}:
        raise ValueError("agg_trade_source must be one of: archive, rest, auto")
    agg_trade_tick_interval = str(cfg.get("agg_trade_tick_interval", "1s")).strip().lower() or "1s"
    if agg_trade_tick_interval != "raw":
        if not agg_trade_tick_interval.endswith("s"):
            raise ValueError("agg_trade_tick_interval must be 'raw' or an integer number of seconds like 1s, 5s, 15s")
        try:
            if int(agg_trade_tick_interval[:-1]) <= 0:
                raise ValueError("agg_trade_tick_interval seconds must be greater than zero")
        except ValueError as exc:
            raise ValueError("agg_trade_tick_interval must be 'raw' or an integer number of seconds like 1s, 5s, 15s") from exc
    agg_trade_archive_base_url = str(cfg.get("agg_trade_archive_base_url", "") or "").strip()
    agg_trade_rest_base_url = str(cfg.get("agg_trade_rest_base_url", "") or "").strip()
    agg_trade_cache_enabled = bool(cfg.get("agg_trade_cache_enabled", True))
    agg_trade_cache_dir = Path(str(cfg.get("agg_trade_cache_dir", "data/agg_trades") or "data/agg_trades")).expanduser()

    raw_input_path = str(cfg.get("market_tape_input_path", "") or "").strip()
    market_tape_input_path = Path(raw_input_path).expanduser() if raw_input_path else None
    raw_market_event_input_path = str(cfg.get("market_event_input_path", "") or "").strip()
    market_event_input_path = Path(raw_market_event_input_path).expanduser() if raw_market_event_input_path else None

    return BacktestConfig(
        backtest_input_mode=input_mode,
        execution_mode=execution_mode,
        agg_trade_source=agg_trade_source,
        agg_trade_tick_interval=agg_trade_tick_interval,
        agg_trade_archive_base_url=agg_trade_archive_base_url,
        agg_trade_rest_base_url=agg_trade_rest_base_url,
        agg_trade_cache_enabled=agg_trade_cache_enabled,
        agg_trade_cache_dir=agg_trade_cache_dir,
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


def run_backtest(
    config: BacktestConfig,
    *,
    technical_logger: Any | None = None,
    progress_reporter: BacktestProgressReporter | None = None,
    dataset_builder: Callable[..., pd.DataFrame] = build_dataset,
    strategy_runner: Callable[..., tuple[float, pd.DataFrame, list[float], dict[str, Any]]] = run_strategy_on_tape,
    market_event_strategy_runner: Callable[..., tuple[float, pd.DataFrame, list[float], dict[str, Any]]] = run_strategy_on_market_events,
    replay_writer: Callable[..., ReplaySummary] = write_replay_artifacts,
    market_event_execution_writer: Callable[..., MarketExecutionSummary] = write_market_event_execution_artifacts,
    trade_alignment_writer: Callable[..., TradeAlignmentSummary] = write_trade_alignment_artifacts,
) -> BacktestResult:
    logger = technical_logger or get_technical_logger()
    market_event_projection: DiscreteTapeProjectionSummary | None = None
    agg_trade_stream_summary: AggTradeMarketEventStreamSummary | None = None
    market_events: list[MarketEvent] | None = None
    market_event_iter_factory: Callable[[], Any] | None = None
    needs_reporting = bool(
        config.enable_plot
        or config.run_monte_carlo
        or config.run_rolling_window_backtest_distribution
    )
    report_module: Any | None = None
    monte_carlo_from_equity: Callable[..., Any] | None = None
    plot_results_interactive: Callable[..., Any] | None = None
    rolling_window_backtest_distribution: Callable[..., Any] | None = None

    if config.client is not None:
        dataset_module.set_client(config.client)

    def start_stage(label: str, total: int) -> None:
        if progress_reporter is not None:
            progress_reporter.start_stage(label, total)

    def advance_stage(amount: int = 1) -> None:
        if progress_reporter is not None:
            progress_reporter.advance(amount)

    def complete_stage() -> None:
        if progress_reporter is not None:
            progress_reporter.complete_stage()

    if needs_reporting:
        import backtest.reporting as report_module_import
        from backtest.reporting import (
            monte_carlo_from_equity as monte_carlo_from_equity_import,
            plot_results_interactive as plot_results_interactive_import,
            rolling_window_backtest_distribution as rolling_window_backtest_distribution_import,
        )

        report_module = report_module_import
        monte_carlo_from_equity = monte_carlo_from_equity_import
        plot_results_interactive = plot_results_interactive_import
        rolling_window_backtest_distribution = rolling_window_backtest_distribution_import

        if config.client is not None:
            report_module.set_client(config.client)

    logger.info("bot_mode_backtest_started symbol=%s leverage=%s", config.symbol, config.leverage)
    if config.backtest_input_mode == "discrete_tape":
        source_path = config.market_tape_input_path or config.market_tape_path
        start_stage("Load Market Tape", 1)
        tape = read_discrete_market_tape(source_path)
        if not tape:
            raise ValueError(f"No market tape rows found at {source_path}")
        df = discrete_market_tape_to_frame(tape)
        if df.empty:
            raise ValueError(f"Market tape frame is empty at {source_path}")
        advance_stage()
        complete_stage()
        logger.info("backtest_market_tape_loaded rows=%s path=%s", len(tape), source_path)
        if source_path != config.market_tape_path:
            start_stage("Write Market Tape", 1)
            write_discrete_market_tape(config.market_tape_path, tape)
            advance_stage()
            complete_stage()
            logger.info("backtest_market_tape_copied rows=%s path=%s", len(tape), config.market_tape_path)
        start_stage("Historical Event Stream", max(1, len(tape)))
        market_events = build_historical_market_event_stream(
            tape,
            intervals={key: str(value) for key, value in config.intervals.items()},
            progress_reporter=progress_reporter,
        )
        complete_stage()
        market_event_iter_factory = lambda: market_events
        start_stage("Write Market Events", 1)
        write_discrete_market_event_stream(config.market_event_stream_path, market_events)
        advance_stage()
        complete_stage()
        logger.info(
            "backtest_historical_market_event_stream_written events=%s path=%s",
            len(market_events),
            config.market_event_stream_path,
        )
    elif config.backtest_input_mode == "market_event_stream":
        source_path = config.market_event_input_path or config.market_event_stream_path
        projection_total = count_market_event_stream(source_path)
        start_stage("Project Market Tape", max(1, projection_total))
        tape, market_event_projection = read_projected_discrete_tape_from_market_event_stream(
            source_path,
            candle_interval=str(config.intervals["small"]),
            symbol=config.symbol,
            progress_reporter=progress_reporter,
        )
        complete_stage()
        if not tape:
            raise ValueError(f"Market event stream produced no discrete tape rows at {source_path}")
        df = discrete_market_tape_to_frame(tape)
        if market_event_projection is not None:
            if market_event_projection.total_events <= 0:
                raise ValueError(f"No market events found at {source_path}")
            logger.info("backtest_market_event_stream_loaded events=%s path=%s", market_event_projection.total_events, source_path)
            logger.info(
                "backtest_market_event_projection rows=%s candles=%s indicator_snapshots=%s ignored=%s",
                market_event_projection.projected_rows,
                market_event_projection.matched_candles,
                market_event_projection.matched_indicator_snapshots,
                market_event_projection.ignored_events,
            )
        target_market_event_path = source_path
        if source_path != config.market_event_stream_path:
            start_stage("Copy Market Events", 1)
            write_discrete_market_event_stream(config.market_event_stream_path, iter_market_event_stream(source_path))
            advance_stage()
            complete_stage()
            logger.info(
                "backtest_market_event_stream_copied events=%s path=%s",
                market_event_projection.total_events if market_event_projection is not None else 0,
                config.market_event_stream_path,
            )
            target_market_event_path = config.market_event_stream_path
        market_event_iter_factory = lambda path=target_market_event_path: iter_market_event_stream(path)
        start_stage("Write Market Tape", 1)
        write_discrete_market_tape(config.market_tape_path, tape)
        advance_stage()
        complete_stage()
        logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)
    elif config.backtest_input_mode == "agg_trade_stream":
        df, agg_trade_stream_summary = write_historical_agg_trade_market_event_stream(
            symbol=config.symbol,
            backtest_period_days=config.backtest_period_days,
            backtest_period_end_time=config.backtest_period_end_time,
            intervals={key: str(value) for key, value in config.intervals.items()},
            output_path=config.market_event_stream_path,
            source=config.agg_trade_source,
            tick_interval=config.agg_trade_tick_interval,
            archive_base_url=config.agg_trade_archive_base_url,
            rest_base_url=config.agg_trade_rest_base_url,
            cache_enabled=config.agg_trade_cache_enabled,
            cache_dir=config.agg_trade_cache_dir,
            progress_reporter=progress_reporter,
        )
        if agg_trade_stream_summary.total_events <= 0:
            raise ValueError("No market events were built from historical aggTrades.")
        market_event_iter_factory = lambda path=config.market_event_stream_path: iter_market_event_stream(path)
        logger.info(
            "backtest_agg_trade_market_event_stream_written source=%s cache_hit=%s cache_path=%s raw_trades=%s price_ticks=%s small_candles=%s medium_candles=%s big_candles=%s indicator_snapshots=%s total_events=%s path=%s",
            agg_trade_stream_summary.source,
            agg_trade_stream_summary.cache_hit,
            agg_trade_stream_summary.cache_path,
            agg_trade_stream_summary.raw_agg_trades,
            agg_trade_stream_summary.price_ticks,
            agg_trade_stream_summary.candle_events_small,
            agg_trade_stream_summary.candle_events_medium,
            agg_trade_stream_summary.candle_events_big,
            agg_trade_stream_summary.indicator_snapshots,
            agg_trade_stream_summary.total_events,
            config.market_event_stream_path,
        )
        start_stage("Project Market Tape", agg_trade_stream_summary.total_events)
        tape, market_event_projection = project_discrete_tape_from_market_events(
            market_event_iter_factory(),
            candle_interval=str(config.intervals["small"]),
            symbol=config.symbol,
            require_indicator_snapshot=True,
            progress_reporter=progress_reporter,
        )
        complete_stage()
        if not tape:
            raise ValueError("Historical aggTrade stream produced no discrete tape rows.")
        start_stage("Write Market Tape", 1)
        write_discrete_market_tape(config.market_tape_path, tape)
        advance_stage()
        complete_stage()
        logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)
    else:
        start_stage("Dataset Build", 1)
        df = dataset_builder(
            symbol=config.symbol,
            intervals=config.intervals,
            backtest_period_days=config.backtest_period_days,
            backtest_period_end_time=config.backtest_period_end_time,
        )
        advance_stage()
        complete_stage()
        tape = build_discrete_market_tape(df, symbol=config.symbol)
        start_stage("Write Market Tape", 1)
        write_discrete_market_tape(config.market_tape_path, tape)
        advance_stage()
        complete_stage()
        logger.info("backtest_market_tape_written rows=%s path=%s", len(tape), config.market_tape_path)
        start_stage("Historical Event Stream", max(1, len(tape)))
        market_events = build_historical_market_event_stream(
            tape,
            intervals={key: str(value) for key, value in config.intervals.items()},
            progress_reporter=progress_reporter,
        )
        complete_stage()
        market_event_iter_factory = lambda: market_events
        start_stage("Write Market Events", 1)
        write_discrete_market_event_stream(config.market_event_stream_path, market_events)
        advance_stage()
        complete_stage()
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
        "runtime_mode": config.runtime_mode,
        **config.params,
    }
    market_event_total: int | None = None
    if agg_trade_stream_summary is not None:
        market_event_total = agg_trade_stream_summary.total_events
    elif market_event_projection is not None:
        market_event_total = market_event_projection.total_events
    elif market_events is not None:
        market_event_total = len(market_events)

    if config.strategy_mode == "realtime" or config.backtest_input_mode in {"market_event_stream", "agg_trade_stream"}:
        if market_events is None and market_event_iter_factory is None:
            raise ValueError("market event strategy runner requires a market event source")
        start_stage("Strategy Replay", max(1, market_event_total or 0))
        final_balance, trades, balance_history, state = market_event_strategy_runner(
            market_events if market_events is not None else market_event_iter_factory(),
            candle_interval=str(config.intervals["small"]),
            intervals={key: str(value) for key, value in config.intervals.items()},
            market_event_total=market_event_total,
            strategy_mode=config.strategy_mode,
            execution_mode=config.execution_mode,
            strict_indicator_alignment=False,
            progress_reporter=progress_reporter,
            **strategy_kwargs,
        )
        complete_stage()
    else:
        start_stage("Strategy Replay", max(1, len(tape) - 1))
        final_balance, trades, balance_history, state = strategy_runner(
            tape,
            progress_reporter=progress_reporter,
            **strategy_kwargs,
        )
        complete_stage()

    start_stage("Artifacts", 5)
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
    advance_stage()
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
    advance_stage()
    execution_artifact_events = market_events if market_events is not None else (
        market_event_iter_factory() if market_event_iter_factory is not None else []
    )
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
    advance_stage()
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
    advance_stage()
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
    advance_stage()
    complete_stage()

    if config.enable_plot:
        if plot_results_interactive is None:
            raise RuntimeError("plot_results_interactive is unavailable because reporting helpers were not loaded")
        plot_results_interactive(df, trades, balance_history, split_by_year=config.plot_split_by_year)

    if config.run_monte_carlo:
        if monte_carlo_from_equity is None:
            raise RuntimeError("monte_carlo_from_equity is unavailable because reporting helpers were not loaded")
        monte_carlo_from_equity(df, balance_history, start_balance=config.initial_balance)

    if config.run_rolling_window_backtest_distribution:
        if rolling_window_backtest_distribution is None:
            raise RuntimeError("rolling_window_backtest_distribution is unavailable because reporting helpers were not loaded")
        rolling_window_backtest_distribution(
            df,
            k_days=config.rolling_window_days,
            n_days=config.backtest_period_days,
            start_balance=config.initial_balance,
            max_workers=config.rolling_window_workers,
            leverage=config.leverage,
            **config.params,
        )

    return BacktestResult(
        dataset=df,
        tape=tape,
        market_events=market_events,
        market_event_projection=market_event_projection,
        agg_trade_stream_summary=agg_trade_stream_summary,
        market_event_execution_summary=market_event_execution_summary,
        trade_alignment_summary=trade_alignment_summary,
        final_balance=float(final_balance),
        trades=trades,
        balance_history=balance_history,
        state=state,
        replay_summary=replay_summary,
        stats=stats,
    )


# Backward-compatible aliases while the rest of the repo and any external scripts catch up.
DiscreteBacktestConfig = BacktestConfig
DiscreteBacktestResult = BacktestResult
build_discrete_backtest_config = build_backtest_config
run_discrete_backtest = run_backtest
