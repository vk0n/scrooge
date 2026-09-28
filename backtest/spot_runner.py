from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys
from threading import RLock

from tqdm import tqdm

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_scenario import export_current_treasury_scenario, load_spot_backtest_scenario
from shared.spot_signal import parse_percentage_series


class SpotMarketDataProgressBars:
    def __init__(self, *, position: int = 0, stream: object = sys.stderr) -> None:
        self.position = position
        self.stream = stream
        self.asset_bar: tqdm | None = None
        self.phase_bars: dict[str, tqdm] = {}
        self.asset_positions: dict[str, int] = {}
        self.lock = RLock()

    def start_asset(self, symbol: str, *, index: int, total: int) -> None:
        with self.lock:
            if self.asset_bar is None:
                self.asset_bar = tqdm(
                    total=total,
                    desc="Market Data",
                    unit="asset",
                    dynamic_ncols=True,
                    position=self.position,
                    leave=True,
                    file=self.stream,
                )
            self.asset_positions[symbol] = index
            self.asset_bar.set_postfix_str(f"started {index}/{total} {symbol}", refresh=True)

    def start_phase(self, symbol: str, label: str, *, total: int | None = None) -> None:
        with self.lock:
            existing = self.phase_bars.pop(symbol, None)
            if existing is not None:
                existing.close()
            self.phase_bars[symbol] = tqdm(
                total=total,
                desc=f"[{symbol}] {label}",
                unit="row",
                unit_scale=True,
                dynamic_ncols=True,
                mininterval=0.25,
                position=self.position + self.asset_positions.get(symbol, 1),
                leave=False,
                file=self.stream,
            )

    def advance(self, symbol: str, amount: int = 1) -> None:
        with self.lock:
            phase_bar = self.phase_bars.get(symbol)
            if phase_bar is not None and amount > 0:
                phase_bar.update(amount)

    def complete_asset(self, symbol: str, *, source: str, rows: int) -> None:
        with self.lock:
            phase_bar = self.phase_bars.pop(symbol, None)
            if phase_bar is not None:
                phase_bar.close()
            if self.asset_bar is not None:
                self.asset_bar.update(1)
                self.asset_bar.set_postfix_str(
                    f"{symbol} {source} {self._compact_count(rows)} rows",
                    refresh=True,
                )

    @staticmethod
    def _compact_count(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

    def close(self) -> None:
        with self.lock:
            for phase_bar in self.phase_bars.values():
                phase_bar.close()
            self.phase_bars.clear()
            if self.asset_bar is not None:
                self.asset_bar.close()
                self.asset_bar = None


class SpotReplayProgress:
    def __init__(
        self,
        *,
        asset_count: int,
        description: str = "Spot Replay",
        position: int | None = None,
        leave: bool = True,
        stream: object = sys.stderr,
    ) -> None:
        self.asset_count = asset_count
        self.description = description
        self.position = position
        self.leave = leave
        self.stream = stream
        self.bar: tqdm | None = None
        self.completed = 0

    @staticmethod
    def _compact_count(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

    def __call__(self, completed: int, total: int) -> None:
        if self.bar is None:
            kwargs: dict[str, object] = {
                "total": total,
                "desc": self.description,
                "unit": "cycle",
                "dynamic_ncols": True,
                "mininterval": 0.25,
                "leave": self.leave,
                "file": self.stream,
            }
            if self.position is not None:
                kwargs["position"] = self.position
            self.bar = tqdm(
                **kwargs,
            )
        delta = max(0, completed - self.completed)
        if delta:
            self.bar.update(delta)
            self.completed = completed
        asset_candles = completed * self.asset_count
        total_asset_candles = total * self.asset_count
        self.bar.set_postfix_str(
            "asset-candles="
            f"{self._compact_count(asset_candles)}/{self._compact_count(total_asset_candles)}",
            refresh=False,
        )
        if completed >= total:
            self.bar.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Scrooge's production Spot strategy over historical data.")
    parser.add_argument("--config", help="Spot backtest YAML scenario.")
    parser.add_argument("--start", help="Override scenario start in ISO-8601 UTC.")
    parser.add_argument("--end", help="Override scenario end in ISO-8601 UTC.")
    parser.add_argument("--preset", choices=("6m", "1y"), help="Derive start relative to the chosen end.")
    parser.add_argument("--output", help="Override artifact output directory.")
    parser.add_argument(
        "--close-profit-pct",
        type=float,
        help="Override the progression close-profit target for this replay.",
    )
    parser.add_argument(
        "--levels-pct",
        help="Override rolling 24H signal thresholds as comma-separated percentages, e.g. 5,8,12,18.",
    )
    parser.add_argument(
        "--market-data-workers",
        type=int,
        default=1,
        help="Load independent asset histories concurrently (default: 1).",
    )
    parser.add_argument("--export-current", metavar="PATH", help="Export current Treasury into a reviewable scenario.")
    parser.add_argument("--name", default="current-treasury-counterfactual", help="Exported scenario name.")
    return parser


def _resolve_output(path: Path) -> Path:
    if path.name.lower() != "auto":
        return path
    root = path.parent
    run_dir = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.resolve())
    except OSError:
        pass
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.export_current:
        if not args.start or not args.end:
            raise SystemExit("--export-current requires explicit --start and --end dates.")
        target = export_current_treasury_scenario(
            args.export_current,
            start=args.start,
            end=args.end,
            name=args.name,
        )
        print(f"Reviewable Spot scenario exported to {target}")
        return 0
    if not args.config:
        raise SystemExit("--config is required unless --export-current is used.")

    scenario = load_spot_backtest_scenario(
        args.config,
        start_override=args.start,
        end_override=args.end,
        preset=args.preset,
        output_override=str(Path(args.output).resolve()) if args.output else None,
        close_profit_pct_override=args.close_profit_pct,
        levels_pct_override=(
            parse_percentage_series(args.levels_pct, field_name="levels-pct")
            if args.levels_pct
            else None
        ),
    )
    output_dir = _resolve_output(scenario.output_dir)
    adapter = BinanceSpotHistoricalAdapter(scenario.data_cache_dir)
    print("Loading Spot market data...", file=sys.stderr, flush=True)
    market_progress = SpotMarketDataProgressBars()
    try:
        dataset = adapter.load(
            scenario,
            progress=market_progress,
            max_workers=max(1, args.market_data_workers),
        )
    finally:
        market_progress.close()
    print("Replaying Spot strategy...", file=sys.stderr, flush=True)
    progress = SpotReplayProgress(asset_count=len(scenario.asset_order))
    result = SpotPortfolioBacktester(scenario, dataset).run(progress=progress)
    print("Writing Spot research artifacts...", file=sys.stderr, flush=True)
    artifacts = write_spot_backtest_artifacts(result, output_dir)
    report = artifacts["report"]["portfolio"]
    print(f"Spot research artifacts: {artifacts['output_dir']}")
    print(f"Final Treasury Value: ${report['final_treasury_value']:,.2f}")
    print(f"HODL Final Treasury Value: ${report['hodl_final_treasury_value']:,.2f}")
    print(f"Difference vs HODL: ${report['difference_vs_hodl']:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
