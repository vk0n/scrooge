from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

from tqdm import tqdm

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_scenario import export_current_treasury_scenario, load_spot_backtest_scenario
from shared.spot_signal import parse_percentage_series


class _ReplayProgress:
    def __init__(self, *, asset_count: int, stream: object = sys.stderr) -> None:
        self.asset_count = asset_count
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
            self.bar = tqdm(
                total=total,
                desc="Spot Replay",
                unit="cycle",
                dynamic_ncols=True,
                mininterval=0.25,
                leave=True,
                file=self.stream,
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
    dataset = adapter.load(scenario)
    print("Replaying Spot strategy...", file=sys.stderr, flush=True)
    progress = _ReplayProgress(asset_count=len(scenario.asset_order))
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
