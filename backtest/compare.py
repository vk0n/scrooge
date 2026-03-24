from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
import json
import hashlib
import logging
import math
from multiprocessing import RLock
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
from backtest.agg_trade_market_event_stream import build_historical_agg_trade_market_event_stream_session
from backtest.progress import ScenarioProgressBar
from backtest.runner import BacktestResult, build_backtest_config, run_backtest
from backtest.sieves import CompareSieve, resolve_compare_sieves
from backtest.time_windows import resolve_backtest_time_range
from bot.event_log import get_technical_logger
from core.binance_retry import create_binance_client
from core.event_store import reset_event_store
from dotenv import load_dotenv
from tqdm import tqdm


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
    "SCROOGE_TQDM_POSITION_BASE",
    "SCROOGE_TQDM_DESC_PREFIX",
)

_COMPARE_SUMMARY_FILENAME = "compare_scenario_summary.json"
_SHARED_MARKET_EVENT_CONFIG_FILENAME = "shared_market_event_source.json"
_SHARED_MARKET_EVENT_SUMMARY_FILENAME = "shared_market_event_summary.json"
_BACKTEST_STAT_LOG_PATTERN = re.compile(r"backtest_stat (?P<key>[^=]+)=(?P<value>.+)$")
_COMPARE_COMPLETED_LOG_PATTERN = re.compile(
    r"compare_scenario_completed name=(?P<name>\S+) "
    r"duration_seconds=(?P<duration_seconds>-?\d+(?:\.\d+)?) "
    r"final_balance=(?P<final_balance>-?\d+(?:\.\d+)?) "
    r"return_pct=(?P<return_pct>-?\d+(?:\.\d+)?) "
    r"trades=(?P<trades>\d+)"
)
_AGG_TRADE_SUMMARY_LOG_PATTERN = re.compile(
    r"backtest_agg_trade_market_event_stream_written .*"
    r"raw_trades=(?P<raw_trades>\d+) .*"
    r"price_ticks=(?P<price_ticks>\d+) .*"
    r"total_events=(?P<total_events>\d+)"
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
    compare_parallel: bool
    compare_max_workers: int
    sieve_stage_keep_ratio: float
    sieve_min_stage_avg_win_rate_pct: float | None
    sieve_auto_advance_single_candidate: bool
    sieve_preset: str | None
    sieves: list[CompareSieve]
    base_backtest_overrides: dict[str, Any]
    scenarios: list[CompareScenario]


@dataclass(slots=True)
class CompareScenarioSummary:
    name: str
    base_name: str | None
    candidate_label: str | None
    param_signature: str | None
    sieve_stage: str | None
    sieve_name: str | None
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
    backtest_period_start_time: str | None
    backtest_period_end_time: str | None
    initial_balance: float | None
    final_balance: float | None
    score: float | None
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
    skipped: int
    failed: int
    scenarios: list[CompareScenarioSummary]


@dataclass(slots=True)
class CompareSieveAggregate:
    base_name: str
    scenario_count: int
    succeeded: int
    skipped: int
    failed: int
    positive_return_sieves: int
    profit_factor_ge_one_sieves: int
    avg_return_pct: float | None
    min_return_pct: float | None
    worst_max_drawdown_pct: float | None
    avg_score: float | None
    min_score: float | None
    stage_returns: dict[str, dict[str, float | None]]
    stage_scores: dict[str, dict[str, float | None]]
    stage_avg_return_pct: dict[str, float | None]
    stage_min_return_pct: dict[str, float | None]
    stage_avg_score: dict[str, float | None]
    stage_min_score: dict[str, float | None]
    stage_passes: dict[str, bool]
    overall_pass: bool


@dataclass(slots=True)
class CompareStageDecision:
    stage_name: str
    input_candidates: int
    completed_candidates: int
    eligible_candidates: int
    kept_candidates: int
    keep_ratio: float
    min_avg_win_rate_pct: float | None
    auto_advanced_single_candidate: bool
    survivors: list[str]
    dropped: list[dict[str, Any]]


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
    if run_dir.parts:
        first_part = run_dir.parts[0]
        if first_part in {"api", "backtest", "bot", "config", "core", "data", "docker", "frontend", "requirements", "runtime"}:
            return (PROJECT_ROOT / run_dir).resolve()
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

    compare_run_dir = str(
        os.getenv("SCROOGE_COMPARE_RUN_DIR", config.get("compare_run_dir", "auto")) or ""
    ).strip() or "auto"
    raw_compare_run_root = str(config.get("compare_run_root", str(DEFAULT_COMPARE_RUN_ROOT)) or "").strip()
    compare_run_root = _resolve_path(raw_compare_run_root, base_dir=raw_path.parent)
    compare_parallel = bool(config.get("compare_parallel", True))
    raw_keep_ratio = config.get("sieve_stage_keep_ratio", 0.5)
    try:
        sieve_stage_keep_ratio = float(raw_keep_ratio)
    except (TypeError, ValueError):
        sieve_stage_keep_ratio = 0.5
    if sieve_stage_keep_ratio <= 0.0:
        sieve_stage_keep_ratio = 0.5
    sieve_stage_keep_ratio = min(1.0, sieve_stage_keep_ratio)

    raw_min_stage_win_rate = config.get("sieve_min_stage_avg_win_rate_pct", 50.0)
    if raw_min_stage_win_rate in (None, "", False):
        sieve_min_stage_avg_win_rate_pct = None
    else:
        try:
            sieve_min_stage_avg_win_rate_pct = float(raw_min_stage_win_rate)
        except (TypeError, ValueError):
            sieve_min_stage_avg_win_rate_pct = 50.0
    sieve_auto_advance_single_candidate = bool(config.get("sieve_auto_advance_single_candidate", True))

    raw_compare_max_workers = config.get("compare_max_workers")
    default_workers = min(
        max(1, os.cpu_count() or 1),
        max(1, len(config.get("scenarios") or [])),
        2,
    )
    try:
        compare_max_workers = int(raw_compare_max_workers) if raw_compare_max_workers is not None else default_workers
    except (TypeError, ValueError):
        compare_max_workers = default_workers
    if compare_max_workers <= 0:
        compare_max_workers = 1

    base_backtest_overrides = config.get("base_backtest_overrides", {})
    if not isinstance(base_backtest_overrides, dict):
        raise ValueError("base_backtest_overrides must be a mapping")
    sieve_preset = str(config.get("sieve_preset", "") or "").strip() or None
    sieves = resolve_compare_sieves(
        preset=sieve_preset,
        raw_sieves=config.get("sieves"),
    )

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
        compare_parallel=compare_parallel,
        compare_max_workers=compare_max_workers,
        sieve_stage_keep_ratio=sieve_stage_keep_ratio,
        sieve_min_stage_avg_win_rate_pct=sieve_min_stage_avg_win_rate_pct,
        sieve_auto_advance_single_candidate=sieve_auto_advance_single_candidate,
        sieve_preset=sieve_preset,
        sieves=sieves,
        base_backtest_overrides=dict(base_backtest_overrides),
        scenarios=scenarios,
    )


@contextmanager
def _scenario_env(
    run_dir: Path,
    *,
    strategy_mode: str | None,
    progress_position_base: int | None = None,
    progress_desc_prefix: str | None = None,
) -> Iterator[None]:
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
    if progress_position_base is not None:
        os.environ["SCROOGE_TQDM_POSITION_BASE"] = str(progress_position_base)
    else:
        os.environ.pop("SCROOGE_TQDM_POSITION_BASE", None)
    if progress_desc_prefix:
        os.environ["SCROOGE_TQDM_DESC_PREFIX"] = progress_desc_prefix
    else:
        os.environ.pop("SCROOGE_TQDM_DESC_PREFIX", None)

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


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_persisted_compare_summary(scenario_dir: Path) -> CompareScenarioSummary | None:
    payload = _read_json_mapping(scenario_dir / _COMPARE_SUMMARY_FILENAME)
    if not payload:
        return None
    try:
        return CompareScenarioSummary(**payload)
    except TypeError:
        return None


def _persist_compare_summary(scenario_dir: Path, summary: CompareScenarioSummary) -> None:
    _write_json(scenario_dir / _COMPARE_SUMMARY_FILENAME, asdict(summary))


def _scenario_resume_matches(
    *,
    scenario_dir: Path,
    config_payload: dict[str, Any],
) -> bool:
    existing_payload = _read_json_mapping(scenario_dir / "backtest_config.resolved.yaml")
    if existing_payload is None:
        existing_path = scenario_dir / "backtest_config.resolved.yaml"
        if not existing_path.exists():
            return False
        try:
            existing_loaded = _load_yaml_mapping(existing_path)
        except (OSError, ValueError, yaml.YAMLError):
            return False
        existing_payload = existing_loaded

    comparable_keys = (
        "symbol",
        "strategy_mode",
        "execution_mode",
        "backtest_input_mode",
        "agg_trade_tick_interval",
        "backtest_period_days",
        "backtest_period_start_time",
        "backtest_period_end_time",
    )
    for key in comparable_keys:
        if existing_payload.get(key) != config_payload.get(key):
            return False
    if existing_payload.get("indicator_inputs") != config_payload.get("indicator_inputs"):
        return False
    if existing_payload.get("params") != config_payload.get("params"):
        return False
    return True


def _load_completed_summary_from_artifacts(
    *,
    name: str,
    run_dir: Path,
    config_payload: dict[str, Any],
) -> CompareScenarioSummary | None:
    log_path = run_dir / "compare_worker.log"
    if not log_path.exists():
        return None

    try:
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    completed_match: re.Match[str] | None = None
    stats: dict[str, str] = {}
    agg_trade_summary: dict[str, str] = {}
    for line in log_lines:
        completed = _COMPARE_COMPLETED_LOG_PATTERN.search(line)
        if completed and completed.group("name") == name:
            completed_match = completed
        stat_match = _BACKTEST_STAT_LOG_PATTERN.search(line)
        if stat_match:
            stats[stat_match.group("key").strip()] = stat_match.group("value").strip()
        agg_match = _AGG_TRADE_SUMMARY_LOG_PATTERN.search(line)
        if agg_match:
            agg_trade_summary = agg_match.groupdict()

    if completed_match is None:
        return None

    initial_balance = _float_or_none(stats.get("Initial Balance"))
    final_balance = _float_or_none(stats.get("Final Balance")) or _float_or_none(completed_match.group("final_balance"))
    profit_factor = _float_or_none(stats.get("Profit Factor"))
    score = _compute_compare_score(
        initial_balance=initial_balance,
        final_balance=final_balance,
        profit_factor=profit_factor,
    )
    replay_payload = _read_json_mapping(run_dir / "replay_summary.json") or {}
    execution_payload = _read_json_mapping(run_dir / "market_event_execution_summary.json") or {}
    alignment_payload = _read_json_mapping(run_dir / "market_event_trade_alignment_summary.json") or {}

    return CompareScenarioSummary(
        name=name,
        base_name=str(config_payload.get("__compare_base_name") or "").strip() or name,
        candidate_label=str(config_payload.get("__compare_candidate_label") or "").strip() or None,
        param_signature=str(config_payload.get("__compare_candidate_param_signature") or "").strip() or None,
        sieve_stage=str(config_payload.get("__compare_sieve_stage") or "").strip() or None,
        sieve_name=str(config_payload.get("__compare_sieve_name") or "").strip() or None,
        status="ok",
        error=None,
        duration_seconds=float(completed_match.group("duration_seconds")),
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=_summary_backtest_input_mode(config_payload),
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_start_time=str(config_payload.get("backtest_period_start_time") or "").strip() or None,
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=initial_balance,
        final_balance=final_balance,
        score=score,
        total_return_pct=_float_or_none(stats.get("Total Return %")) or _float_or_none(completed_match.group("return_pct")),
        number_of_trades=_int_or_none(stats.get("Number of Trades")) or _int_or_none(completed_match.group("trades")),
        win_rate_pct=_float_or_none(stats.get("Win Rate %")),
        total_fee=_float_or_none(stats.get("Total Fee")),
        profit_factor=profit_factor,
        max_drawdown_pct=_float_or_none(stats.get("Max Drawdown %")),
        replay_closed_trades=_int_or_none(replay_payload.get("closed_trades")),
        replay_net_pnl=_float_or_none(replay_payload.get("net_pnl")),
        execution_events=_int_or_none(execution_payload.get("execution_events")),
        observed_trades=_int_or_none(execution_payload.get("observed_total_trades")),
        realized_pnl=_float_or_none(execution_payload.get("realized_pnl_total")),
        alignment_paired_trades=_int_or_none(alignment_payload.get("paired_trades")),
        alignment_pnl_delta=_float_or_none(alignment_payload.get("pnl_delta")),
        agg_trade_raw_trades=_int_or_none(agg_trade_summary.get("raw_trades")),
        agg_trade_total_events=_int_or_none(agg_trade_summary.get("total_events")),
        agg_trade_price_ticks=_int_or_none(agg_trade_summary.get("price_ticks")),
    )


def _load_resumable_summary(
    *,
    name: str,
    scenario_dir: Path,
    config_payload: dict[str, Any],
) -> CompareScenarioSummary | None:
    if not scenario_dir.exists():
        return None
    if not _scenario_resume_matches(scenario_dir=scenario_dir, config_payload=config_payload):
        return None
    persisted = _load_persisted_compare_summary(scenario_dir)
    if persisted is not None:
        return persisted
    return _load_completed_summary_from_artifacts(
        name=name,
        run_dir=scenario_dir,
        config_payload=config_payload,
    )


def _compute_compare_score(
    *,
    initial_balance: float | None,
    final_balance: float | None,
    profit_factor: float | None,
) -> float | None:
    if (
        initial_balance is None
        or final_balance is None
        or initial_balance <= 0
        or profit_factor is None
        or math.isnan(profit_factor)
    ):
        return None
    balance_ratio = final_balance / initial_balance
    # A no-loss sieve can legitimately report Profit Factor=inf. Treat that as a
    # special case so a single one-trade winner does not dominate the ranking.
    if math.isinf(profit_factor):
        return balance_ratio
    return profit_factor * balance_ratio


def _summary_backtest_input_mode(config_payload: dict[str, Any]) -> str | None:
    original = str(config_payload.get("__compare_original_backtest_input_mode") or "").strip().lower()
    if original:
        return original
    resolved = str(config_payload.get("backtest_input_mode") or "").strip().lower()
    return resolved or None


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
    initial_balance = _float_or_none(stats.get("Initial Balance"))
    final_balance = _float_or_none(stats.get("Final Balance"))
    profit_factor = _float_or_none(stats.get("Profit Factor"))
    score = _compute_compare_score(
        initial_balance=initial_balance,
        final_balance=final_balance,
        profit_factor=profit_factor,
    )

    return CompareScenarioSummary(
        name=name,
        base_name=str(config_payload.get("__compare_base_name") or "").strip() or name,
        candidate_label=str(config_payload.get("__compare_candidate_label") or "").strip() or None,
        param_signature=str(config_payload.get("__compare_candidate_param_signature") or "").strip() or None,
        sieve_stage=str(config_payload.get("__compare_sieve_stage") or "").strip() or None,
        sieve_name=str(config_payload.get("__compare_sieve_name") or "").strip() or None,
        status="ok",
        error=None,
        duration_seconds=duration_seconds,
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=_summary_backtest_input_mode(config_payload),
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_start_time=str(config_payload.get("backtest_period_start_time") or "").strip() or None,
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=initial_balance,
        final_balance=final_balance,
        score=score,
        total_return_pct=_float_or_none(stats.get("Total Return %")),
        number_of_trades=_int_or_none(stats.get("Number of Trades")),
        win_rate_pct=_float_or_none(stats.get("Win Rate %")),
        total_fee=_float_or_none(stats.get("Total Fee")),
        profit_factor=profit_factor,
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
        base_name=str(config_payload.get("__compare_base_name") or "").strip() or name,
        candidate_label=str(config_payload.get("__compare_candidate_label") or "").strip() or None,
        param_signature=str(config_payload.get("__compare_candidate_param_signature") or "").strip() or None,
        sieve_stage=str(config_payload.get("__compare_sieve_stage") or "").strip() or None,
        sieve_name=str(config_payload.get("__compare_sieve_name") or "").strip() or None,
        status="failed",
        error=f"{type(error).__name__}: {error}",
        duration_seconds=duration_seconds,
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=_summary_backtest_input_mode(config_payload),
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_start_time=str(config_payload.get("backtest_period_start_time") or "").strip() or None,
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=None,
        final_balance=None,
        score=None,
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


def _summarize_skipped(
    *,
    name: str,
    run_dir: Path,
    config_payload: dict[str, Any],
    reason: str,
) -> CompareScenarioSummary:
    return CompareScenarioSummary(
        name=name,
        base_name=str(config_payload.get("__compare_base_name") or "").strip() or name,
        candidate_label=str(config_payload.get("__compare_candidate_label") or "").strip() or None,
        param_signature=str(config_payload.get("__compare_candidate_param_signature") or "").strip() or None,
        sieve_stage=str(config_payload.get("__compare_sieve_stage") or "").strip() or None,
        sieve_name=str(config_payload.get("__compare_sieve_name") or "").strip() or None,
        status="skipped",
        error=reason,
        duration_seconds=0.0,
        run_dir=str(run_dir),
        symbol=str(config_payload.get("symbol") or "").strip().upper() or None,
        strategy_mode=str(config_payload.get("strategy_mode") or "").strip().lower() or None,
        execution_mode=str(config_payload.get("execution_mode") or "").strip().lower() or None,
        backtest_input_mode=_summary_backtest_input_mode(config_payload),
        agg_trade_tick_interval=str(config_payload.get("agg_trade_tick_interval") or "").strip().lower() or None,
        backtest_period_days=_int_or_none(config_payload.get("backtest_period_days")),
        backtest_period_start_time=str(config_payload.get("backtest_period_start_time") or "").strip() or None,
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or "").strip() or None,
        initial_balance=None,
        final_balance=None,
        score=None,
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


def _group_sieves_by_stage(sieves: list[CompareSieve]) -> list[tuple[str, list[CompareSieve]]]:
    ordered_stage_names: list[str] = []
    grouped: dict[str, list[CompareSieve]] = {}
    for sieve in sieves:
        if sieve.stage_name not in grouped:
            grouped[sieve.stage_name] = []
            ordered_stage_names.append(sieve.stage_name)
        grouped[sieve.stage_name].append(sieve)
    return [(stage_name, grouped[stage_name]) for stage_name in ordered_stage_names]


def _candidate_stage_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    avg_score = float(metrics.get("avg_score") or float("-inf"))
    min_score = float(metrics.get("min_score") or float("-inf"))
    avg_return_pct = float(metrics.get("avg_return_pct") or float("-inf"))
    worst_max_drawdown_pct = float(metrics.get("worst_max_drawdown_pct") or float("-inf"))
    return (avg_score, min_score, avg_return_pct, worst_max_drawdown_pct)


def _select_stage_survivors(
    *,
    stage_name: str,
    candidate_names: list[str],
    stage_summaries: list[CompareScenarioSummary],
    stage_sieves: list[CompareSieve],
    keep_ratio: float,
    min_avg_win_rate_pct: float | None,
    auto_advance_single_candidate: bool,
) -> CompareStageDecision:
    grouped: dict[str, list[CompareScenarioSummary]] = {}
    for item in stage_summaries:
        grouped.setdefault(item.base_name or item.name, []).append(item)

    expected_sieve_names = {sieve.name for sieve in stage_sieves}
    stage_metrics: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []

    for candidate_name in candidate_names:
        items = grouped.get(candidate_name, [])
        ok_items = [item for item in items if item.status == "ok"]
        ok_sieve_names = {item.sieve_name for item in ok_items if item.sieve_name}
        if len(ok_items) != len(stage_sieves) or ok_sieve_names != expected_sieve_names:
            dropped.append(
                {
                    "candidate": candidate_name,
                    "reason": "stage_incomplete",
                }
            )
            continue

        score_values = [float(item.score) for item in ok_items if item.score is not None]
        return_values = [float(item.total_return_pct) for item in ok_items if item.total_return_pct is not None]
        win_rate_values = [float(item.win_rate_pct) for item in ok_items if item.win_rate_pct is not None]
        drawdown_values = [float(item.max_drawdown_pct) for item in ok_items if item.max_drawdown_pct is not None]
        stage_pass = (
            len(ok_items) == len(stage_sieves)
            and all(item.score is not None and float(item.score) >= 1.0 for item in ok_items)
        )
        stage_metrics[candidate_name] = {
            "candidate": candidate_name,
            "stage_pass": stage_pass,
            "avg_score": (sum(score_values) / len(score_values)) if score_values else None,
            "min_score": min(score_values) if score_values else None,
            "avg_return_pct": (sum(return_values) / len(return_values)) if return_values else None,
            "min_return_pct": min(return_values) if return_values else None,
            "avg_win_rate_pct": (sum(win_rate_values) / len(win_rate_values)) if win_rate_values else None,
            "worst_max_drawdown_pct": min(drawdown_values) if drawdown_values else None,
        }

    completed_candidates = list(stage_metrics.values())
    auto_advanced = False
    passed_candidates = [item for item in completed_candidates if bool(item.get("stage_pass"))]
    failed_candidates = [item for item in completed_candidates if not bool(item.get("stage_pass"))]
    for metrics in failed_candidates:
        dropped.append(
            {
                "candidate": metrics["candidate"],
                "reason": "stage_failed",
                "avg_score": metrics.get("avg_score"),
                "min_score": metrics.get("min_score"),
                "avg_return_pct": metrics.get("avg_return_pct"),
                "min_return_pct": metrics.get("min_return_pct"),
                "avg_win_rate_pct": metrics.get("avg_win_rate_pct"),
                "worst_max_drawdown_pct": metrics.get("worst_max_drawdown_pct"),
            }
        )
    eligible_candidates = passed_candidates

    if len(eligible_candidates) == 1 and auto_advance_single_candidate:
        survivors = [eligible_candidates[0]["candidate"]]
        auto_advanced = True
    else:
        filtered_candidates = eligible_candidates
        if min_avg_win_rate_pct is not None:
            threshold_pass = [
                item
                for item in filtered_candidates
                if item.get("avg_win_rate_pct") is None or float(item["avg_win_rate_pct"]) >= min_avg_win_rate_pct
            ]
            if threshold_pass:
                filtered_out_names = {item["candidate"] for item in filtered_candidates} - {
                    item["candidate"] for item in threshold_pass
                }
                for candidate_name in sorted(filtered_out_names):
                    metrics = stage_metrics[candidate_name]
                    dropped.append(
                        {
                            "candidate": candidate_name,
                            "reason": f"avg_win_rate_below_{min_avg_win_rate_pct:.2f}",
                            "avg_score": metrics.get("avg_score"),
                            "avg_return_pct": metrics.get("avg_return_pct"),
                            "avg_win_rate_pct": metrics.get("avg_win_rate_pct"),
                            "worst_max_drawdown_pct": metrics.get("worst_max_drawdown_pct"),
                        }
                    )
                filtered_candidates = threshold_pass

        eligible_candidates = filtered_candidates
        if len(eligible_candidates) == 1 and auto_advance_single_candidate:
            survivors = [eligible_candidates[0]["candidate"]]
            auto_advanced = True
        elif eligible_candidates:
            keep_count = max(1, math.ceil(len(eligible_candidates) * keep_ratio))
            ranked = sorted(eligible_candidates, key=_candidate_stage_sort_key, reverse=True)
            survivors = [item["candidate"] for item in ranked[:keep_count]]
            survivor_names = set(survivors)
            for metrics in ranked[keep_count:]:
                dropped.append(
                    {
                        "candidate": metrics["candidate"],
                        "reason": "ranked_out",
                        "avg_score": metrics.get("avg_score"),
                        "avg_return_pct": metrics.get("avg_return_pct"),
                        "avg_win_rate_pct": metrics.get("avg_win_rate_pct"),
                        "worst_max_drawdown_pct": metrics.get("worst_max_drawdown_pct"),
                    }
                )
        else:
            survivors = []

    dropped_names = {item["candidate"] for item in dropped}
    for candidate_name, metrics in stage_metrics.items():
        if candidate_name in dropped_names or candidate_name in survivors:
            continue
        dropped.append(
            {
                "candidate": candidate_name,
                "reason": "ranked_out",
                "avg_score": metrics.get("avg_score"),
                "avg_return_pct": metrics.get("avg_return_pct"),
                "avg_win_rate_pct": metrics.get("avg_win_rate_pct"),
                "worst_max_drawdown_pct": metrics.get("worst_max_drawdown_pct"),
            }
        )

    dropped.sort(key=lambda item: item["candidate"])
    return CompareStageDecision(
        stage_name=stage_name,
        input_candidates=len(candidate_names),
        completed_candidates=len(completed_candidates),
        eligible_candidates=len(eligible_candidates),
        kept_candidates=len(survivors),
        keep_ratio=keep_ratio,
        min_avg_win_rate_pct=min_avg_win_rate_pct,
        auto_advanced_single_candidate=auto_advanced,
        survivors=sorted(survivors),
        dropped=dropped,
    )


def _format_cell(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _avg_trade_pct(item: CompareScenarioSummary) -> float | None:
    if item.total_return_pct is None or item.number_of_trades is None:
        return None
    if item.number_of_trades <= 0:
        return None
    return float(item.total_return_pct) / float(item.number_of_trades)


def _fee_load_pct(item: CompareScenarioSummary) -> float | None:
    if item.total_fee is None:
        return None
    net_pnl = item.replay_net_pnl
    if net_pnl is None and item.final_balance is not None and item.initial_balance is not None:
        net_pnl = float(item.final_balance) - float(item.initial_balance)
    if net_pnl is None:
        return None
    denominator = abs(float(net_pnl))
    if denominator <= 1e-12:
        return None
    return (float(item.total_fee) / denominator) * 100.0


def _compare_table_sort_key(item: CompareScenarioSummary) -> tuple[float | int | str, ...]:
    status_order = {
        "ok": 0,
        "failed": 1,
        "skipped": 2,
    }
    score = item.score if item.score is not None else float("-inf")
    total_return_pct = item.total_return_pct if item.total_return_pct is not None else float("-inf")
    final_balance = item.final_balance if item.final_balance is not None else float("-inf")
    win_rate_pct = item.win_rate_pct if item.win_rate_pct is not None else float("-inf")
    label = item.candidate_label or item.name
    return (
        status_order.get(item.status, 3),
        -score,
        -total_return_pct,
        -final_balance,
        -win_rate_pct,
        label,
    )


def _display_scenario_label(item: CompareScenarioSummary) -> str:
    label = item.candidate_label or item.name
    if item.status == "ok":
        return label
    return f"{label} ({item.status})"


def _write_compare_table(path: Path, scenarios: list[CompareScenarioSummary]) -> None:
    sorted_scenarios = sorted(scenarios, key=_compare_table_sort_key)
    header = [
        "Scenario",
        "Score",
        "Return %",
        "Final Balance",
        "Trades",
        "Avg Trade %",
        "Win Rate %",
        "Fee",
        "Fee Load %",
        "Profit Factor",
        "Max DD %",
    ]
    lines = [
        "# Compare Results",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for item in sorted_scenarios:
        row = [
            _display_scenario_label(item),
            _format_cell(item.score),
            _format_cell(item.total_return_pct),
            _format_cell(item.final_balance),
            _format_cell(item.number_of_trades, digits=0),
            _format_cell(_avg_trade_pct(item)),
            _format_cell(item.win_rate_pct),
            _format_cell(item.total_fee),
            _format_cell(_fee_load_pct(item)),
            _format_cell(item.profit_factor),
            _format_cell(item.max_drawdown_pct),
        ]
        lines.append("| " + " | ".join(row) + " |")

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


def _write_compare_sieve_table(path: Path, aggregates: list[CompareSieveAggregate]) -> None:
    header = [
        "Candidate",
        "Executed",
        "Skipped",
        "Failed",
        "1m Pass",
        "90d Pass",
        "180d Pass",
        "Overall Pass",
        "Positive Sieves",
        "PF>=1 Sieves",
        "1m Bull %",
        "1m Bear %",
        "1m Neutral %",
        "90d Bull %",
        "90d Bear %",
        "90d Neutral %",
        "180d Bull %",
        "180d Bear %",
        "180d Neutral %",
        "180d Score",
        "Avg Score",
        "Min Score",
        "Avg %",
        "Min %",
        "Worst DD %",
    ]
    lines = [
        "# Three-Sieves Results",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for item in aggregates:
        month_returns = item.stage_returns.get("month", {})
        quarter_returns = item.stage_returns.get("quarter", {})
        half_year_returns = item.stage_returns.get("half_year", {})
        half_year_avg_score = item.stage_avg_score.get("half_year")
        lines.append(
            "| "
            + " | ".join(
                [
                    item.base_name,
                    f"{item.succeeded}/{item.scenario_count}",
                    str(item.skipped),
                    str(item.failed),
                    "yes" if item.stage_passes.get("month", False) else "no",
                    "yes" if item.stage_passes.get("quarter", False) else "no",
                    "yes" if item.stage_passes.get("half_year", False) else "no",
                    "yes" if item.overall_pass else "no",
                    str(item.positive_return_sieves),
                    str(item.profit_factor_ge_one_sieves),
                    _format_cell(month_returns.get("bull")),
                    _format_cell(month_returns.get("bear")),
                    _format_cell(month_returns.get("neutral")),
                    _format_cell(quarter_returns.get("bull")),
                    _format_cell(quarter_returns.get("bear")),
                    _format_cell(quarter_returns.get("neutral")),
                    _format_cell(half_year_returns.get("bull")),
                    _format_cell(half_year_returns.get("bear")),
                    _format_cell(half_year_returns.get("neutral")),
                    _format_cell(half_year_avg_score),
                    _format_cell(item.avg_score),
                    _format_cell(item.min_score),
                    _format_cell(item.avg_return_pct),
                    _format_cell(item.min_return_pct),
                    _format_cell(item.worst_max_drawdown_pct),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sieve_aggregate_sort_key(item: CompareSieveAggregate) -> tuple[float, ...]:
    final_stage_name = "half_year"
    final_stage_avg_score = item.stage_avg_score.get(final_stage_name)
    final_stage_min_score = item.stage_min_score.get(final_stage_name)
    final_stage_avg_return_pct = item.stage_avg_return_pct.get(final_stage_name)
    final_stage_min_return_pct = item.stage_min_return_pct.get(final_stage_name)
    passed_stage_count = sum(1 for passed in item.stage_passes.values() if passed)
    return (
        0 if item.overall_pass else 1,
        0 if item.stage_passes.get(final_stage_name, False) else 1,
        -passed_stage_count,
        -(final_stage_avg_score if final_stage_avg_score is not None else float("-inf")),
        -(final_stage_avg_return_pct if final_stage_avg_return_pct is not None else float("-inf")),
        -(final_stage_min_score if final_stage_min_score is not None else float("-inf")),
        -(final_stage_min_return_pct if final_stage_min_return_pct is not None else float("-inf")),
        -(item.avg_score if item.avg_score is not None else float("-inf")),
        -(item.avg_return_pct if item.avg_return_pct is not None else float("-inf")),
    )


def _aggregate_sieves(scenarios: list[CompareScenarioSummary]) -> list[CompareSieveAggregate]:
    grouped: dict[str, list[CompareScenarioSummary]] = {}
    for item in scenarios:
        if not item.base_name or not item.sieve_name or not item.sieve_stage:
            continue
        grouped.setdefault(item.base_name, []).append(item)

    aggregates: list[CompareSieveAggregate] = []
    for base_name, items in sorted(grouped.items()):
        ok_items = [item for item in items if item.status == "ok"]
        returns = [item.total_return_pct for item in ok_items if item.total_return_pct is not None]
        scores = [item.score for item in ok_items if item.score is not None]
        max_drawdowns = [item.max_drawdown_pct for item in ok_items if item.max_drawdown_pct is not None]
        stage_returns: dict[str, dict[str, float | None]] = {
            "month": {"bull": None, "bear": None, "neutral": None},
            "quarter": {"bull": None, "bear": None, "neutral": None},
            "half_year": {"bull": None, "bear": None, "neutral": None},
        }
        stage_scores: dict[str, dict[str, float | None]] = {
            "month": {"bull": None, "bear": None, "neutral": None},
            "quarter": {"bull": None, "bear": None, "neutral": None},
            "half_year": {"bull": None, "bear": None, "neutral": None},
        }
        stage_groups: dict[str, list[CompareScenarioSummary]] = {}
        for item in ok_items:
            if item.sieve_stage not in stage_returns:
                stage_returns[item.sieve_stage] = {}
                stage_scores[item.sieve_stage] = {}
            stage_returns[item.sieve_stage][item.sieve_name] = item.total_return_pct
            stage_scores[item.sieve_stage][item.sieve_name] = item.score
            stage_groups.setdefault(item.sieve_stage, []).append(item)
        stage_avg_return_pct = {
            stage: (
                sum(sub.total_return_pct for sub in group if sub.total_return_pct is not None)
                / len([sub for sub in group if sub.total_return_pct is not None])
                if any(sub.total_return_pct is not None for sub in group)
                else None
            )
            for stage, group in stage_groups.items()
        }
        stage_min_return_pct = {
            stage: (
                min(sub.total_return_pct for sub in group if sub.total_return_pct is not None)
                if any(sub.total_return_pct is not None for sub in group)
                else None
            )
            for stage, group in stage_groups.items()
        }
        stage_avg_score = {
            stage: (
                sum(sub.score for sub in group if sub.score is not None)
                / len([sub for sub in group if sub.score is not None])
                if any(sub.score is not None for sub in group)
                else None
            )
            for stage, group in stage_groups.items()
        }
        stage_min_score = {
            stage: (
                min(sub.score for sub in group if sub.score is not None)
                if any(sub.score is not None for sub in group)
                else None
            )
            for stage, group in stage_groups.items()
        }
        stage_passes = {
            stage: (
                len(group) == 3
                and all((sub.score or 0.0) >= 1.0 for sub in group)
            )
            for stage, group in stage_groups.items()
        }
        aggregates.append(
            CompareSieveAggregate(
                base_name=base_name,
                scenario_count=len(items),
                succeeded=len(ok_items),
                skipped=sum(1 for item in items if item.status == "skipped"),
                failed=sum(1 for item in items if item.status == "failed"),
                positive_return_sieves=sum(1 for item in ok_items if (item.total_return_pct or 0.0) > 0.0),
                profit_factor_ge_one_sieves=sum(1 for item in ok_items if (item.profit_factor or 0.0) >= 1.0),
                avg_return_pct=(sum(returns) / len(returns) if returns else None),
                min_return_pct=(min(returns) if returns else None),
                worst_max_drawdown_pct=(min(max_drawdowns) if max_drawdowns else None),
                avg_score=(sum(scores) / len(scores) if scores else None),
                min_score=(min(scores) if scores else None),
                stage_returns=stage_returns,
                stage_scores=stage_scores,
                stage_avg_return_pct=stage_avg_return_pct,
                stage_min_return_pct=stage_min_return_pct,
                stage_avg_score=stage_avg_score,
                stage_min_score=stage_min_score,
                stage_passes=stage_passes,
                overall_pass=all(stage_passes.get(stage, False) for stage in ("month", "quarter", "half_year")),
            )
        )
    return sorted(aggregates, key=_sieve_aggregate_sort_key)


def _prepare_compare_run_dir(config: CompareConfig) -> Path:
    run_dir = _resolve_compare_run_dir(config.compare_run_dir, config.compare_run_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.compare_run_root.mkdir(parents=True, exist_ok=True)
    _safe_latest_symlink(config.compare_run_root, run_dir)
    return run_dir


def _shared_market_event_signature(config_payload: dict[str, Any]) -> dict[str, Any] | None:
    input_mode = str(config_payload.get("backtest_input_mode") or "").strip().lower()
    if input_mode != "agg_trade_stream":
        return None

    intervals = config_payload.get("intervals")
    if not isinstance(intervals, dict) or not intervals:
        return None

    normalized_intervals = {
        str(key): str(value)
        for key, value in sorted(intervals.items(), key=lambda item: str(item[0]))
    }
    resolved_time_range = resolve_backtest_time_range(
        backtest_period_days=int(config_payload.get("backtest_period_days") or 0),
        backtest_period_start_time=str(config_payload.get("backtest_period_start_time") or ""),
        backtest_period_end_time=str(config_payload.get("backtest_period_end_time") or ""),
    )
    return {
        "symbol": str(config_payload.get("symbol") or "").strip().upper(),
        "agg_trade_source": str(config_payload.get("agg_trade_source") or "archive").strip().lower() or "archive",
        "agg_trade_tick_interval": str(config_payload.get("agg_trade_tick_interval") or "1s").strip().lower() or "1s",
        "agg_trade_archive_base_url": str(config_payload.get("agg_trade_archive_base_url") or "").strip(),
        "agg_trade_rest_base_url": str(config_payload.get("agg_trade_rest_base_url") or "").strip(),
        "agg_trade_cache_enabled": bool(config_payload.get("agg_trade_cache_enabled", True)),
        "agg_trade_cache_dir": str(config_payload.get("agg_trade_cache_dir") or "").strip(),
        "intervals": normalized_intervals,
        "backtest_period_days": int(config_payload.get("backtest_period_days") or 0),
        "backtest_period_start_time": str(config_payload.get("backtest_period_start_time") or "").strip(),
        "backtest_period_end_time": str(config_payload.get("backtest_period_end_time") or "").strip(),
        "resolved_start_time": resolved_time_range.start_time.isoformat(),
        "resolved_end_time": resolved_time_range.end_time.isoformat(),
        "resolved_cache_key": resolved_time_range.cache_key,
    }


def _shared_market_event_signature_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _shared_market_event_paths(
    *,
    compare_run_dir: Path,
    signature: dict[str, Any],
) -> tuple[Path, Path, Path]:
    resource_dir = compare_run_dir / "shared_market_events" / _shared_market_event_signature_hash(signature)
    return (
        resource_dir / "market_events.jsonl",
        resource_dir / _SHARED_MARKET_EVENT_CONFIG_FILENAME,
        resource_dir / _SHARED_MARKET_EVENT_SUMMARY_FILENAME,
    )


def _shared_market_event_stream_ready(
    *,
    compare_run_dir: Path,
    signature: dict[str, Any],
) -> Path | None:
    event_path, config_path, summary_path = _shared_market_event_paths(
        compare_run_dir=compare_run_dir,
        signature=signature,
    )
    existing_config = _read_json_mapping(config_path)
    if existing_config != signature:
        return None
    if not event_path.exists() or not summary_path.exists():
        return None
    return event_path


def _build_shared_market_event_stream(
    *,
    compare_run_dir: Path,
    signature: dict[str, Any],
    representative_config: dict[str, Any],
    logger: Any,
    scenario_count: int,
) -> Path:
    event_path, config_path, summary_path = _shared_market_event_paths(
        compare_run_dir=compare_run_dir,
        signature=signature,
    )
    ready_path = _shared_market_event_stream_ready(
        compare_run_dir=compare_run_dir,
        signature=signature,
    )
    signature_key = _shared_market_event_signature_hash(signature)
    if ready_path is not None:
        existing_summary = _read_json_mapping(summary_path) or {}
        logger.info(
            "compare_shared_market_event_stream_reused key=%s scenarios=%s total_events=%s path=%s",
            signature_key,
            scenario_count,
            existing_summary.get("total_events"),
            ready_path,
        )
        return ready_path

    logger.info(
        "compare_shared_market_event_stream_build_started key=%s scenarios=%s symbol=%s tick=%s range=%s..%s path=%s",
        signature_key,
        scenario_count,
        signature.get("symbol"),
        signature.get("agg_trade_tick_interval"),
        signature.get("backtest_period_start_time"),
        signature.get("backtest_period_end_time"),
        event_path,
    )
    load_dotenv()
    session = build_historical_agg_trade_market_event_stream_session(
        symbol=str(representative_config.get("symbol") or ""),
        backtest_period_days=int(representative_config.get("backtest_period_days") or 0),
        backtest_period_start_time=str(representative_config.get("backtest_period_start_time") or ""),
        backtest_period_end_time=str(representative_config.get("backtest_period_end_time") or ""),
        intervals={str(key): str(value) for key, value in dict(representative_config.get("intervals") or {}).items()},
        output_path=event_path,
        source=str(representative_config.get("agg_trade_source") or "archive").strip().lower() or "archive",
        tick_interval=str(representative_config.get("agg_trade_tick_interval") or "1s").strip().lower() or "1s",
        archive_base_url=str(representative_config.get("agg_trade_archive_base_url") or "").strip(),
        rest_base_url=str(representative_config.get("agg_trade_rest_base_url") or "").strip(),
        cache_enabled=bool(representative_config.get("agg_trade_cache_enabled", True)),
        cache_dir=str(representative_config.get("agg_trade_cache_dir") or "").strip() or None,
        progress_reporter=None,
    )
    event_iterator = session.iter_events(report_progress=True)
    with closing(event_iterator):
        for _ in event_iterator:
            pass
    summary = session.summary
    _write_json(config_path, signature)
    _write_json(summary_path, asdict(summary))
    logger.info(
        "compare_shared_market_event_stream_build_completed key=%s scenarios=%s total_events=%s raw_trades=%s path=%s",
        signature_key,
        scenario_count,
        summary.total_events,
        summary.raw_agg_trades,
        event_path,
    )
    return event_path


def _prepare_shared_market_event_task_payloads(
    *,
    task_payloads: list[dict[str, Any]],
    compare_run_dir: Path,
    logger: Any,
) -> list[dict[str, Any]]:
    grouped_tasks: dict[str, list[dict[str, Any]]] = {}
    grouped_signatures: dict[str, dict[str, Any]] = {}
    for task_payload in task_payloads:
        signature = _shared_market_event_signature(dict(task_payload["config"]))
        if signature is None:
            continue
        signature_key = _shared_market_event_signature_hash(signature)
        grouped_tasks.setdefault(signature_key, []).append(task_payload)
        grouped_signatures[signature_key] = signature

    if not grouped_tasks:
        return task_payloads

    for signature_key, grouped in grouped_tasks.items():
        signature = grouped_signatures[signature_key]
        ready_path = _shared_market_event_stream_ready(
            compare_run_dir=compare_run_dir,
            signature=signature,
        )
        if len(grouped) <= 1 and ready_path is None:
            continue

        shared_event_path = _build_shared_market_event_stream(
            compare_run_dir=compare_run_dir,
            signature=signature,
            representative_config=dict(grouped[0]["config"]),
            logger=logger,
            scenario_count=len(grouped),
        )
        for task_payload in grouped:
            updated_config = copy.deepcopy(dict(task_payload["config"]))
            updated_config["__compare_original_backtest_input_mode"] = (
                str(updated_config.get("backtest_input_mode") or "").strip().lower() or "agg_trade_stream"
            )
            updated_config["backtest_input_mode"] = "market_event_stream"
            updated_config["market_event_input_path"] = str(shared_event_path)
            updated_config["__compare_shared_market_event_stream_path"] = str(shared_event_path)
            task_payload["config"] = updated_config

        logger.info(
            "compare_shared_market_event_stream_applied key=%s scenarios=%s path=%s",
            signature_key,
            len(grouped),
            shared_event_path,
        )

    return task_payloads


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


def _scenario_task_payload(
    *,
    index: int,
    progress_slot: int,
    scenario: CompareScenario,
    scenario_dir: Path,
    merged_config: dict[str, Any],
    quiet_console_info: bool,
) -> dict[str, Any]:
    return {
        "index": index,
        "progress_slot": progress_slot,
        "name": scenario.name,
        "scenario_dir": str(scenario_dir),
        "config": merged_config,
        "quiet_console_info": quiet_console_info,
    }


def _build_merged_compare_config(
    *,
    base_backtest_config: dict[str, Any],
    base_backtest_overrides: dict[str, Any],
    scenario: CompareScenario,
    sieve: CompareSieve | None,
    anchor_end_time: str,
) -> dict[str, Any]:
    merged_config = _deep_merge_dicts(base_backtest_config, base_backtest_overrides)
    merged_config = _deep_merge_dicts(merged_config, scenario.overrides)
    merged_config["live"] = False
    if sieve is not None:
        time_range = resolve_backtest_time_range(
            backtest_period_days=sieve.duration_days,
            backtest_period_start_time=sieve.start_time,
            backtest_period_end_time=sieve.end_time,
        )
        merged_config["backtest_period_days"] = time_range.duration_days
        merged_config["backtest_period_start_time"] = sieve.start_time
        merged_config["backtest_period_end_time"] = sieve.end_time
        merged_config["__compare_sieve_stage"] = sieve.stage_name
        merged_config["__compare_sieve_name"] = sieve.name
    else:
        merged_config = _apply_compare_end_time_anchor(
            merged_config,
            anchor_end_time=anchor_end_time,
        )
        merged_config["__compare_sieve_stage"] = ""
        merged_config["__compare_sieve_name"] = ""
    merged_config["__compare_base_name"] = scenario.name
    return merged_config


def _execute_compare_task_payloads(
    *,
    task_payloads: list[dict[str, Any]],
    compare_run_dir: Path,
    logger: Any,
    compare_parallel: bool,
    compare_max_workers: int,
) -> dict[int, CompareScenarioSummary]:
    task_payloads = _prepare_shared_market_event_task_payloads(
        task_payloads=task_payloads,
        compare_run_dir=compare_run_dir,
        logger=logger,
    )
    summaries_by_index: dict[int, CompareScenarioSummary] = {}
    parallel_enabled = compare_parallel and len(task_payloads) > 1
    max_workers = max(1, min(compare_max_workers, len(task_payloads)))

    if parallel_enabled:
        for task_payload in task_payloads:
            task_payload["quiet_console_info"] = True
        logger.info(
            "compare_parallel_execution_enabled scenarios=%s max_workers=%s",
            len(task_payloads),
            max_workers,
        )
        progress_lock = RLock()
        tqdm.set_lock(progress_lock)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_compare_worker,
            initargs=(progress_lock,),
        ) as executor:
            futures = {
                executor.submit(_run_compare_scenario_task, task_payload): task_payload
                for task_payload in task_payloads
            }
            for future in as_completed(futures):
                task_payload = futures[future]
                scenario_dir = Path(str(task_payload["scenario_dir"]))
                merged_config = dict(task_payload["config"])
                try:
                    completed = future.result()
                    summaries_by_index[int(completed["index"])] = CompareScenarioSummary(**completed["summary"])
                except Exception as exc:  # noqa: BLE001
                    trace_text = traceback.format_exc()
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    (scenario_dir / "compare_error.txt").write_text(trace_text, encoding="utf-8")
                    logger.exception("compare_scenario_process_failed name=%s", task_payload["name"])
                    summaries_by_index[int(task_payload["index"])] = _summarize_error(
                        name=str(task_payload["name"]),
                        run_dir=scenario_dir,
                        duration_seconds=0.0,
                        config_payload=merged_config,
                        error=exc,
                    )
    else:
        for task_payload in task_payloads:
            completed = _run_compare_scenario_task(task_payload)
            summaries_by_index[int(completed["index"])] = CompareScenarioSummary(**completed["summary"])
    return summaries_by_index


def _progress_slot_for_task(*, index: int, parallel_enabled: bool, max_workers: int) -> int:
    if not parallel_enabled:
        return 0
    normalized_workers = max(1, int(max_workers))
    return max(0, (int(index) - 1) % normalized_workers)


def _configure_compare_worker_logging(
    *,
    scenario_dir: Path,
    quiet_console_info: bool,
) -> None:
    if not quiet_console_info:
        return

    worker_log_path = scenario_dir / "compare_worker.log"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = logging.FileHandler(worker_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger = logging.getLogger("scrooge.bot")
    logger.handlers = [file_handler, stderr_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    event_store_logger = logging.getLogger("scrooge.event_store")
    event_store_logger.handlers = [file_handler, stderr_handler]
    event_store_logger.setLevel(logging.INFO)
    event_store_logger.propagate = False


def _initialize_compare_worker(progress_lock: Any | None = None) -> None:
    if progress_lock is not None:
        tqdm.set_lock(progress_lock)


def _run_compare_scenario_task(task: dict[str, Any]) -> dict[str, Any]:
    scenario_name = str(task["name"])
    scenario_dir = Path(str(task["scenario_dir"])).expanduser()
    merged_config = dict(task["config"])
    quiet_console_info = bool(task.get("quiet_console_info", False))
    progress_position_base = max(0, int(task.get("progress_slot", 0)))
    progress_desc_prefix = f"[{scenario_name}]"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml_snapshot(scenario_dir / "backtest_config.resolved.yaml", merged_config)
    _configure_compare_worker_logging(
        scenario_dir=scenario_dir,
        quiet_console_info=quiet_console_info,
    )
    logger = get_technical_logger()

    load_dotenv()
    client = None
    normalized_input_mode = str(merged_config.get("backtest_input_mode", "build") or "").strip().lower() or "build"
    if normalized_input_mode == "build":
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        client = create_binance_client(api_key, api_secret, logger=logger)
        dataset_module.set_client(client)
        trade_module.set_client(client)

    scenario_start = time.perf_counter()
    logger.info(
        "compare_scenario_started name=%s run_dir=%s strategy_mode=%s input_mode=%s execution_mode=%s",
        scenario_name,
        scenario_dir,
        merged_config.get("strategy_mode"),
        merged_config.get("backtest_input_mode"),
        merged_config.get("execution_mode"),
    )
    scenario_progress = ScenarioProgressBar(
        scenario_name=scenario_name,
        position=progress_position_base,
        leave=False,
    )
    with _scenario_env(
        scenario_dir,
        strategy_mode=str(merged_config.get("strategy_mode") or "").strip().lower() or None,
        progress_position_base=progress_position_base,
        progress_desc_prefix=progress_desc_prefix,
    ):
        try:
            backtest_config = build_backtest_config(
                merged_config,
                chart_dataset_path=scenario_dir / "chart_dataset.csv",
                event_log_path=scenario_dir / "event_history.jsonl",
                runtime_mode="backtest",
                client=client,
            )
            result = run_backtest(
                backtest_config,
                technical_logger=logger,
                progress_reporter=scenario_progress,
            )
            duration_seconds = time.perf_counter() - scenario_start
            summary = _summarize_result(
                name=scenario_name,
                run_dir=scenario_dir,
                duration_seconds=duration_seconds,
                config_payload=merged_config,
                result=result,
            )
            logger.info(
                "compare_scenario_completed name=%s duration_seconds=%.2f final_balance=%.8f return_pct=%.8f trades=%s",
                scenario_name,
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
                name=scenario_name,
                run_dir=scenario_dir,
                duration_seconds=duration_seconds,
                config_payload=merged_config,
                error=exc,
            )
            logger.exception("compare_scenario_failed name=%s duration_seconds=%.2f", scenario_name, duration_seconds)
        finally:
            scenario_progress.close()

    _persist_compare_summary(scenario_dir, summary)
    return {
        "index": int(task["index"]),
        "summary": asdict(summary),
    }


def run_compare(
    config: CompareConfig,
    *,
    technical_logger: Any | None = None,
) -> CompareRunResult:
    logger = technical_logger or get_technical_logger()
    compare_run_dir = _prepare_compare_run_dir(config)
    anchor_end_time = _compare_anchor_end_time()

    base_backtest_config = _load_yaml_mapping(config.base_backtest_config_path)
    compare_config_snapshot = {
        "base_backtest_config_path": str(config.base_backtest_config_path),
        "compare_run_dir": str(compare_run_dir),
        "compare_run_root": str(config.compare_run_root),
        "compare_parallel": config.compare_parallel,
        "compare_max_workers": config.compare_max_workers,
        "sieve_stage_keep_ratio": config.sieve_stage_keep_ratio,
        "sieve_min_stage_avg_win_rate_pct": config.sieve_min_stage_avg_win_rate_pct,
        "sieve_auto_advance_single_candidate": config.sieve_auto_advance_single_candidate,
        "sieve_preset": config.sieve_preset,
        "sieves": [asdict(item) for item in config.sieves],
        "compare_anchor_end_time": anchor_end_time,
        "base_backtest_overrides": config.base_backtest_overrides,
        "scenarios": [{"name": item.name, "overrides": item.overrides} for item in config.scenarios],
    }
    _write_yaml_snapshot(compare_run_dir / "compare_config.resolved.yaml", compare_config_snapshot)

    summaries_by_index: dict[int, CompareScenarioSummary] = {}
    stage_decisions: list[CompareStageDecision] = []
    next_index = 1
    resumed_scenarios = 0

    if config.sieves:
        scenario_lookup = {scenario.name: scenario for scenario in config.scenarios}
        stage_groups = _group_sieves_by_stage(config.sieves)
        active_scenarios = list(config.scenarios)

        for stage_position, (stage_name, stage_sieves) in enumerate(stage_groups):
            if not active_scenarios:
                break

            logger.info(
                "compare_sieve_stage_started stage=%s candidates=%s windows=%s",
                stage_name,
                len(active_scenarios),
                len(stage_sieves),
            )
            task_payloads: list[dict[str, Any]] = []
            stage_results: dict[int, CompareScenarioSummary] = {}
            for scenario in active_scenarios:
                for sieve in stage_sieves:
                    index = next_index
                    next_index += 1
                    progress_slot = _progress_slot_for_task(
                        index=index,
                        parallel_enabled=config.compare_parallel and len(active_scenarios) * len(stage_sieves) > 1,
                        max_workers=config.compare_max_workers,
                    )
                    scenario_name = f"{scenario.name}-{sieve.stage_name}-{sieve.name}"
                    scenario_dir = _scenario_dir(compare_run_dir, index, scenario_name)
                    merged_config = _build_merged_compare_config(
                        base_backtest_config=base_backtest_config,
                        base_backtest_overrides=config.base_backtest_overrides,
                        scenario=scenario,
                        sieve=sieve,
                        anchor_end_time=anchor_end_time,
                    )
                    existing_summary = _load_resumable_summary(
                        name=scenario_name,
                        scenario_dir=scenario_dir,
                        config_payload=merged_config,
                    )
                    if existing_summary is not None:
                        stage_results[index] = existing_summary
                        resumed_scenarios += 1
                        continue
                    task_payloads.append(
                        _scenario_task_payload(
                            index=index,
                            progress_slot=progress_slot,
                            scenario=CompareScenario(name=scenario_name, overrides=scenario.overrides),
                            scenario_dir=scenario_dir,
                            merged_config=merged_config,
                            quiet_console_info=False,
                        )
                    )

            executed_stage_results: dict[int, CompareScenarioSummary] = {}
            if task_payloads:
                executed_stage_results = _execute_compare_task_payloads(
                    task_payloads=task_payloads,
                    compare_run_dir=compare_run_dir,
                    logger=logger,
                    compare_parallel=config.compare_parallel,
                    compare_max_workers=config.compare_max_workers,
                )
            stage_results.update(executed_stage_results)
            for index, summary in stage_results.items():
                summaries_by_index[index] = summary

            stage_summaries = [stage_results[index] for index in sorted(stage_results)]
            decision = _select_stage_survivors(
                stage_name=stage_name,
                candidate_names=[scenario.name for scenario in active_scenarios],
                stage_summaries=stage_summaries,
                stage_sieves=stage_sieves,
                keep_ratio=config.sieve_stage_keep_ratio,
                min_avg_win_rate_pct=config.sieve_min_stage_avg_win_rate_pct,
                auto_advance_single_candidate=config.sieve_auto_advance_single_candidate,
            )
            stage_decisions.append(decision)
            logger.info(
                "compare_sieve_stage_completed stage=%s input_candidates=%s completed_candidates=%s survivors=%s dropped=%s",
                stage_name,
                decision.input_candidates,
                decision.completed_candidates,
                len(decision.survivors),
                len(decision.dropped),
            )
            _write_json(compare_run_dir / "compare_stage_decisions.json", [asdict(item) for item in stage_decisions])

            dropped_by_candidate = {item["candidate"]: item for item in decision.dropped}
            remaining_stage_groups = stage_groups[stage_position + 1 :]
            for dropped_candidate in sorted(dropped_by_candidate):
                scenario = scenario_lookup.get(dropped_candidate)
                if scenario is None:
                    continue
                dropped_entry = dropped_by_candidate[dropped_candidate]
                reason = str(dropped_entry.get("reason") or "filtered_out")
                for future_stage_name, future_stage_sieves in remaining_stage_groups:
                    for sieve in future_stage_sieves:
                        index = next_index
                        next_index += 1
                        scenario_name = f"{scenario.name}-{future_stage_name}-{sieve.name}"
                        scenario_dir = _scenario_dir(compare_run_dir, index, scenario_name)
                        merged_config = _build_merged_compare_config(
                            base_backtest_config=base_backtest_config,
                            base_backtest_overrides=config.base_backtest_overrides,
                            scenario=scenario,
                            sieve=sieve,
                            anchor_end_time=anchor_end_time,
                        )
                        skip_reason = f"filtered_out_after_{stage_name}: {reason}"
                        summaries_by_index[index] = _summarize_skipped(
                            name=scenario_name,
                            run_dir=scenario_dir,
                            config_payload=merged_config,
                            reason=skip_reason,
                        )

            active_scenarios = [
                scenario_lookup[name]
                for name in decision.survivors
                if name in scenario_lookup
            ]
    else:
        task_payloads: list[dict[str, Any]] = []
        preloaded_results: dict[int, CompareScenarioSummary] = {}
        for scenario in config.scenarios:
            index = next_index
            next_index += 1
            progress_slot = _progress_slot_for_task(
                index=index,
                parallel_enabled=config.compare_parallel and len(config.scenarios) > 1,
                max_workers=config.compare_max_workers,
            )
            scenario_dir = _scenario_dir(compare_run_dir, index, scenario.name)
            merged_config = _build_merged_compare_config(
                base_backtest_config=base_backtest_config,
                base_backtest_overrides=config.base_backtest_overrides,
                scenario=scenario,
                sieve=None,
                anchor_end_time=anchor_end_time,
            )
            existing_summary = _load_resumable_summary(
                name=scenario.name,
                scenario_dir=scenario_dir,
                config_payload=merged_config,
            )
            if existing_summary is not None:
                preloaded_results[index] = existing_summary
                resumed_scenarios += 1
                continue
            task_payloads.append(
                _scenario_task_payload(
                    index=index,
                    progress_slot=progress_slot,
                    scenario=scenario,
                    scenario_dir=scenario_dir,
                    merged_config=merged_config,
                    quiet_console_info=False,
                )
            )
        stage_results = dict(preloaded_results)
        executed_stage_results: dict[int, CompareScenarioSummary] = {}
        if task_payloads:
            executed_stage_results = _execute_compare_task_payloads(
                task_payloads=task_payloads,
                compare_run_dir=compare_run_dir,
                logger=logger,
                compare_parallel=config.compare_parallel,
                compare_max_workers=config.compare_max_workers,
            )
        stage_results.update(executed_stage_results)
        for index, summary in stage_results.items():
            summaries_by_index[index] = summary

    if resumed_scenarios:
        logger.info("compare_resume_reused_scenarios count=%s", resumed_scenarios)

    summaries = [summaries_by_index[index] for index in sorted(summaries_by_index)]

    succeeded = sum(1 for item in summaries if item.status == "ok")
    skipped = sum(1 for item in summaries if item.status == "skipped")
    failed = sum(1 for item in summaries if item.status == "failed")
    compare_result = CompareRunResult(
        generated_at=_now_utc_text(),
        compare_run_dir=str(compare_run_dir),
        base_backtest_config_path=str(config.base_backtest_config_path),
        scenario_count=len(summaries),
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        scenarios=summaries,
    )

    summary_payload = {
        **asdict(compare_result),
        "stage_decisions": [asdict(item) for item in stage_decisions],
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
    if stage_decisions:
        _write_json(compare_run_dir / "compare_stage_decisions.json", [asdict(item) for item in stage_decisions])
    _write_jsonl(compare_run_dir / "compare_runs.jsonl", [asdict(item) for item in summaries])
    _write_compare_table(compare_run_dir / "compare_table.md", summaries)
    sieve_aggregates = _aggregate_sieves(summaries)
    if sieve_aggregates:
        _write_json(compare_run_dir / "compare_sieves_summary.json", [asdict(item) for item in sieve_aggregates])
        _write_compare_sieve_table(compare_run_dir / "compare_sieves_table.md", sieve_aggregates)
    return compare_result


def main() -> int:
    load_dotenv()
    logger = get_technical_logger()
    config = load_compare_config()

    result = run_compare(
        config,
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
