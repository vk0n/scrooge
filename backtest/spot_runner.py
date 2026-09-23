from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

from backtest.spot_engine import SpotPortfolioBacktester
from backtest.spot_market_data import BinanceSpotHistoricalAdapter
from backtest.spot_reporting import write_spot_backtest_artifacts
from backtest.spot_scenario import export_current_treasury_scenario, load_spot_backtest_scenario


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Scrooge's production Spot strategy over historical data.")
    parser.add_argument("--config", help="Spot backtest YAML scenario.")
    parser.add_argument("--start", help="Override scenario start in ISO-8601 UTC.")
    parser.add_argument("--end", help="Override scenario end in ISO-8601 UTC.")
    parser.add_argument("--preset", choices=("6m", "1y"), help="Derive start relative to the chosen end.")
    parser.add_argument("--output", help="Override artifact output directory.")
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
        output_override=args.output,
    )
    output_dir = _resolve_output(scenario.output_dir)
    adapter = BinanceSpotHistoricalAdapter(scenario.data_cache_dir)
    dataset = adapter.load(scenario)
    result = SpotPortfolioBacktester(scenario, dataset).run()
    artifacts = write_spot_backtest_artifacts(result, output_dir)
    report = artifacts["report"]["portfolio"]
    print(f"Spot research artifacts: {artifacts['output_dir']}")
    print(f"Final Treasury Value: ${report['final_treasury_value']:,.2f}")
    print(f"HODL Final Treasury Value: ${report['hodl_final_treasury_value']:,.2f}")
    print(f"Difference vs HODL: ${report['difference_vs_hodl']:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
