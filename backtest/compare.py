from __future__ import annotations

import copy
import json
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import backtest.dataset as dataset_module
import bot.trade as trade_module
import yaml
from backtest.runner import BacktestResult, build_backtest_config, run_backtest
from binance.client import Client
from bot.event_log import get_technical_logger
from core.event_store import reset_event_store
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARE_CONFIG_PATH = PROJECT_ROOT / "config" / "compare.yaml"
DEFAULT_COMPARE_RUN_ROOT = PROJECT_ROOT / "runtime" / "compare"

_COMPARE_ENV_KEYS = (
    "SCROOGE_BACKTEST_RUN_DIR",
    "SCROOGE_STATE_FILE",
    "SCROOGE_TRADE_HISTORY_FILE",
    "SCROOGE_BALANCE_HISTORY_FILE",
    "SCROOGE_LOG_FILE",
    "SCROOGE_EVENT_LOG_FILE",
    "SCROOGE_MARKET_EVENT_STREAM_FILE",
    "SCROOGE_RUNTIME_CHART_DATASET_PATH",
    "SCROOGE_RUNTIME_MODE",
    "SCROOGE_STRATEGY_MODE",
)


@dataclass(slots=True)
class CompareScenario:
    name: str
    overrides: dict[str, Any]


@dataclass(slots=True)
class CompareConfig:
    path: Path
    base_backtest_config_path: Path
    compare_run_dir: str
    compare_run_root: Path
    base_backtest_overrides: dict[str, Any]
    scenarios: list[CompareScenario]


@dataclass(slots=True)
class CompareScenarioSummary:
    name: str
    status: str
    error: str | None
    duration_seconds: float
    run_dir: str
    symbol: str | None
    strategy_mode: str | None
    execution_mode: str | None
    backtest_input_mode: str | None
    agg_trade_tick_interval: str | None
    backtest_period_days: int | None
    backtest_period_end_time: str | None
    initial_balance: float | None
    final_balance: float | None
    total_return_pct: float | None
    number_of_trades: int | None
    win_rate_pct: float | None
    total_fee: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    replay_closed_trades: int | None
    replay_net_pnl: float | None
    execution_events: int | None
    observed_trades: int | None
    realized_pnl: float | None
    alignment_paired_trades: int | None
    alignment_pnl_delta: float | None
    agg_trade_raw_trades: int | None
    agg_trade_total_events: int | None
    agg_trade_price_ticks: int | None


@dataclass(slots=True)
class CompareRunResult:
    generated_at: str
    compare_run_dir: str
    base_backtest_config_path: str
    scenario_count: int
    succeeded: int
    failed: int
    scenarios: list[CompareScenarioSummary]


def _now_utc_text() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _compare_anchor_end_time() -> str:
    return datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "scenario"


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts:
        first_part = path.parts[0]
        if first_part in {"api", "backtest", "bot", "config", "core", "data", "docker", "frontend", "requirements", "runtime"}:
            return (PROJECT_ROOT / path).resolve()
    return (base_dir / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_compare_run_dir(raw_run_dir: str, root_dir: Path) -> Path:
    if not raw_run_dir or raw_run_dir.lower() == "auto":
        root_dir.mkdir(parents=True, exist_ok=True)
        return root_dir / _timestamp_slug()
    run_dir = Path(raw_run_dir).expanduser()
    if run_dir.is_absolute():
        return run_dir
    return (root_dir / run_dir).resolve()


def _safe_latest_symlink(root_dir: Path, run_dir: Path) -> None:
    latest = root_dir / "latest"
    try:
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.resolve())
    except OSError:
        get_technical_logger().warning("compare_latest_link_failed root=%s run=%s", root_dir, run_dir)


def load_compare_config(path: str | Path | None = None) -> CompareConfig:
    raw_path = Path(
        path
        or os.getenv("SCROOGE_COMPARE_CONFIG_PATH", str(DEFAULT_COMPARE_CONFIG_PATH))
    ).expanduser()
    config = _load_yaml_mapping(raw_path)

    raw_base_path = str(config.get("base_backtest_config_path", "config/backtest.yaml") or "").strip()
    if not raw_base_path:
        raise ValueError("compare config requires base_backtest_config_path")
    base_backtest_config_path = _resolve_path(raw_base_path, base_dir=raw_path.parent)

    compare_run_dir = str(config.get("compare_run_dir", "auto") or "").strip() or "auto"
    raw_compare_run_root = str(config.get("compare_run_root", str(DEFAULT_COMPARE_RUN_ROOT)) or "").strip()
    compare_run_root = _resolve_path(raw_compare_run_root, base_dir=raw_path.parent)

    base_backtest_overrides = config.get("base_backtest_overrides", {})
    if not isinstance(base_backtest_overrides, dict):
        raise ValueError("base_backtest_overrides must be a mapping")

    raw_scenarios = config.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("compare config requires a non-empty scenarios list")

    scenarios: list[CompareScenario] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(raw_scenarios, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"scenario #{idx} must be a mapping")
        name = str(item.get("name", "") or "").strip()
        if not name:
            raise ValueError(f"scenario #{idx} must include name")
        if name in seen_names:
            raise ValueError(f"duplicate scenario name: {name}")
        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"scenario {name} overrides must be a mapping")
        scenarios.append(CompareScenario(name=name, overrides=overrides))
        seen_names.add(name)

    return CompareConfig(
        path=raw_path.resolve(),
        base_backtest_config_path=base_backtest_config_path,
        compare_run_dir=compare_run_dir,
        compare_run_root=compare_run_root,
        base_backtest_overrides=dict(base_backtest_overrides),
        scenarios=scenarios,
    )


@contextmanager
def _scenario_env(run_dir: Path, *, strategy_mode: str | None) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _COMPARE_ENV_KEYS}
    os.environ["SCROOGE_BACKTEST_RUN_DIR"] = str(run_dir)
    os.environ["SCROOGE_STATE_FILE"] = str(run_dir / "state.json")
    os.environ["SCROOGE_TRADE_HISTORY_FILE"] = str(run_dir / "trade_history.jsonl")
    os.environ["SCROOGE_BALANCE_HISTORY_FILE"] = str(run_dir / "balance_history.jsonl")
    os.environ["SCROOGE_LOG_FILE"] = str(run_dir / "trading_log.txt")
    os.environ["SCROOGE_EVENT_LOG_FILE"] = str(run_dir / "event_history.jsonl")
    os.environ["SCROOGE_MARKET_EVENT_STREAM_FILE"] = str(run_dir / "market_events.jsonl")
    os.environ["SCROOGE_RUNTIME_CHART_DATASET_PATH"] = str(run_dir / "chart_dataset.csv")
    os.environ["SCROOGE_RUNTIME_MODE"] = "backtest"
    if strategy_mode:
        os.environ["SCROOGE_STRATEGY_MODE"] = strategy_mode
    else:
        os.environ.pop("SCROOGE_STRATEGY_MODE", None)

    reset_event_store(run_dir / "event_history.jsonl")
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_event_store(previous.get("SCROOGE_EVENT_LOG_FILE"))


def _write_yaml_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=True, sort_keys=False, default=str)
        file_obj.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=True, sort_keys=False, default=str))
            file_obj.write("\n")


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summarize_result(
    *,
    name: str,
    run_dir: Path,
    duration_seconds: float,
    config_payload: dict[str, Any],
    result: BacktestResult,
) -> CompareScenarioSummary:
    stats = result.stats
    agg_summary = result.agg_trade_stream_summary
    execution_summary = result.market_event_execution_summary
    alignment_summary = result.trade_alignment_summary
    replay_summary = result.replay_summary

    return CompareScenarioSummary(
        name=name,
        status="ok",
        error=None,
        duration_seconds=duration_seconds,
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=str(config_payload.get("backtest_input_mode") or "").strip().lower() or None,
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=_float_or_none(stats.get("Initial Balance")),
        final_balance=_float_or_none(stats.get("Final Balance")),
        total_return_pct=_float_or_none(stats.get("Total Return %")),
        number_of_trades=_int_or_none(stats.get("Number of Trades")),
        win_rate_pct=_float_or_none(stats.get("Win Rate %")),
        total_fee=_float_or_none(stats.get("Total Fee")),
        profit_factor=_float_or_none(stats.get("Profit Factor")),
        max_drawdown_pct=_float_or_none(stats.get("Max Drawdown %")),
        replay_closed_trades=int(replay_summary.closed_trades),
        replay_net_pnl=float(replay_summary.net_pnl),
        execution_events=(int(execution_summary.execution_events) if execution_summary is not None else None),
        observed_trades=(int(execution_summary.observed_total_trades) if execution_summary is not None else None),
        realized_pnl=(float(execution_summary.realized_pnl_total) if execution_summary is not None else None),
        alignment_paired_trades=(int(alignment_summary.paired_trades) if alignment_summary is not None else None),
        alignment_pnl_delta=(float(alignment_summary.pnl_delta) if alignment_summary is not None else None),
        agg_trade_raw_trades=(int(agg_summary.raw_agg_trades) if agg_summary is not None else None),
        agg_trade_total_events=(int(agg_summary.total_events) if agg_summary is not None else None),
        agg_trade_price_ticks=(int(agg_summary.price_ticks) if agg_summary is not None else None),
    )


def _summarize_error(
    *,
    name: str,
    run_dir: Path,
    duration_seconds: float,
    config_payload: dict[str, Any],
    error: BaseException,
) -> CompareScenarioSummary:
    return CompareScenarioSummary(
        name=name,
        status="failed",
        error=f"{type(error).__name__}: {error}",
        duration_seconds=duration_seconds,
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=str(config_payload.get("backtest_input_mode") or "").strip().lower() or None,
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=None,
        final_balance=None,
        total_return_pct=None,
        number_of_trades=None,
        win_rate_pct=None,
        total_fee=None,
        profit_factor=None,
        max_drawdown_pct=None,
        replay_closed_trades=None,
        replay_net_pnl=None,
        execution_events=None,
        observed_trades=None,
        realized_pnl=None,
        alignment_paired_trades=None,
        alignment_pnl_delta=None,
        agg_trade_raw_trades=None,
        agg_trade_total_events=None,
        agg_trade_price_ticks=None,
    )


def _format_cell(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_compare_table(path: Path, scenarios: list[CompareScenarioSummary]) -> None:
    header = [
        "Scenario",
        "Status",
        "Mode",
        "Input",
        "Exec",
        "Days",
        "Tick",
        "Return %",
        "Final Balance",
        "Trades",
        "Win Rate %",
        "Fee",
        "Profit Factor",
        "Max DD %",
    ]
    lines = [
        "# Compare Results",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for item in scenarios:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    item.status,
                    item.strategy_mode or "-",
                    item.backtest_input_mode or "-",
                    item.execution_mode or "-",
                    _format_cell(item.backtest_period_days, digits=0),
                    item.agg_trade_tick_interval or "-",
                    _format_cell(item.total_return_pct),
                    _format_cell(item.final_balance),
                    _format_cell(item.number_of_trades, digits=0),
                    _format_cell(item.win_rate_pct),
                    _format_cell(item.total_fee),
                    _format_cell(item.profit_factor),
                    _format_cell(item.max_drawdown_pct),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_scenario_name(
    scenarios: list[CompareScenarioSummary],
    *,
    key: Callable[[CompareScenarioSummary], float | None],
) -> str | None:
    candidates = [item for item in scenarios if item.status == "ok" and key(item) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(key(item) or float("-inf"))).name


def _scenario_dir(compare_run_dir: Path, index: int, name: str) -> Path:
    return compare_run_dir / "scenarios" / f"{index:02d}-{_slugify(name)}"


def _prepare_compare_run_dir(config: CompareConfig) -> Path:
    run_dir = _resolve_compare_run_dir(config.compare_run_dir, config.compare_run_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.compare_run_root.mkdir(parents=True, exist_ok=True)
    _safe_latest_symlink(config.compare_run_root, run_dir)
    return run_dir


def _apply_compare_end_time_anchor(
    payload: dict[str, Any],
    *,
    anchor_end_time: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    raw_end_time = str(updated.get("backtest_period_end_time", "") or "").strip()
    if not raw_end_time:
        updated["backtest_period_end_time"] = anchor_end_time
    return updated


def run_compare(
    config: CompareConfig,
    *,
    client: Any | None = None,
    technical_logger: Any | None = None,
    backtest_runner: Callable[..., BacktestResult] = run_backtest,
    backtest_config_builder: Callable[..., Any] = build_backtest_config,
) -> CompareRunResult:
    logger = technical_logger or get_technical_logger()
    compare_run_dir = _prepare_compare_run_dir(config)
    anchor_end_time = _compare_anchor_end_time()

    base_backtest_config = _load_yaml_mapping(config.base_backtest_config_path)
    compare_config_snapshot = {
        "base_backtest_config_path": str(config.base_backtest_config_path),
        "compare_run_dir": str(compare_run_dir),
        "compare_run_root": str(config.compare_run_root),
        "compare_anchor_end_time": anchor_end_time,
        "base_backtest_overrides": config.base_backtest_overrides,
        "scenarios": [{"name": item.name, "overrides": item.overrides} for item in config.scenarios],
    }
    _write_yaml_snapshot(compare_run_dir / "compare_config.resolved.yaml", compare_config_snapshot)

    summaries: list[CompareScenarioSummary] = []

    for index, scenario in enumerate(config.scenarios, start=1):
        scenario_dir = _scenario_dir(compare_run_dir, index, scenario.name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        merged_config = _deep_merge_dicts(base_backtest_config, config.base_backtest_overrides)
        merged_config = _deep_merge_dicts(merged_config, scenario.overrides)
        merged_config["live"] = False
        merged_config = _apply_compare_end_time_anchor(
            merged_config,
            anchor_end_time=anchor_end_time,
        )
        _write_yaml_snapshot(scenario_dir / "backtest_config.resolved.yaml", merged_config)

        scenario_start = time.perf_counter()
        logger.info(
            "compare_scenario_started name=%s run_dir=%s strategy_mode=%s input_mode=%s execution_mode=%s",
            scenario.name,
            scenario_dir,
            merged_config.get("strategy_mode"),
            merged_config.get("backtest_input_mode"),
            merged_config.get("execution_mode"),
        )
        with _scenario_env(
            scenario_dir,
            strategy_mode=str(merged_config.get("strategy_mode") or "").strip().lower() or None,
        ):
            try:
                backtest_config = backtest_config_builder(
                    merged_config,
                    chart_dataset_path=scenario_dir / "chart_dataset.csv",
                    event_log_path=scenario_dir / "event_history.jsonl",
                    runtime_mode="backtest",
                    client=client,
                )
                result = backtest_runner(
                    backtest_config,
                    technical_logger=logger,
                )
                duration_seconds = time.perf_counter() - scenario_start
                summary = _summarize_result(
                    name=scenario.name,
                    run_dir=scenario_dir,
                    duration_seconds=duration_seconds,
                    config_payload=merged_config,
                    result=result,
                )
                logger.info(
                    "compare_scenario_completed name=%s duration_seconds=%.2f final_balance=%.8f return_pct=%.8f trades=%s",
                    scenario.name,
                    duration_seconds,
                    summary.final_balance or 0.0,
                    summary.total_return_pct or 0.0,
                    summary.number_of_trades or 0,
                )
            except Exception as exc:  # noqa: BLE001
                duration_seconds = time.perf_counter() - scenario_start
                trace_text = traceback.format_exc()
                (scenario_dir / "compare_error.txt").write_text(trace_text, encoding="utf-8")
                summary = _summarize_error(
                    name=scenario.name,
                    run_dir=scenario_dir,
                    duration_seconds=duration_seconds,
                    config_payload=merged_config,
                    error=exc,
                )
                logger.exception("compare_scenario_failed name=%s duration_seconds=%.2f", scenario.name, duration_seconds)

        summaries.append(summary)

    succeeded = sum(1 for item in summaries if item.status == "ok")
    failed = len(summaries) - succeeded
    compare_result = CompareRunResult(
        generated_at=_now_utc_text(),
        compare_run_dir=str(compare_run_dir),
        base_backtest_config_path=str(config.base_backtest_config_path),
        scenario_count=len(summaries),
        succeeded=succeeded,
        failed=failed,
        scenarios=summaries,
    )

    summary_payload = {
        **asdict(compare_result),
        "best_by_total_return_pct": _best_scenario_name(
            summaries,
            key=lambda item: item.total_return_pct,
        ),
        "best_by_final_balance": _best_scenario_name(
            summaries,
            key=lambda item: item.final_balance,
        ),
    }
    _write_json(compare_run_dir / "compare_summary.json", summary_payload)
    _write_jsonl(compare_run_dir / "compare_runs.jsonl", [asdict(item) for item in summaries])
    _write_compare_table(compare_run_dir / "compare_table.md", summaries)
    return compare_result


def main() -> int:
    load_dotenv()
    logger = get_technical_logger()
    config = load_compare_config()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    dataset_module.set_client(client)
    trade_module.set_client(client)

    result = run_compare(
        config,
        client=client,
        technical_logger=logger,
    )
    if result.failed > 0:
        logger.warning(
            "compare_completed_with_failures compare_run_dir=%s succeeded=%s failed=%s",
            result.compare_run_dir,
            result.succeeded,
            result.failed,
        )
        return 1
    logger.info(
        "compare_completed compare_run_dir=%s scenarios=%s",
        result.compare_run_dir,
        result.scenario_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
