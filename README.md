# Scrooge

Scrooge is an event-driven Binance Futures trading system with:
- a live runtime in `bot/`
- shared strategy and event logic in `core/`
- replayable backtests, compare runs, reporting, and optimization in `backtest/`
- a local control plane in `api/` + `frontend/`

Today the project is centered on realtime execution and realtime-grade historical replay:
- live trading can run in `strategy_mode: realtime`
- backtests can replay native historical `market_events.jsonl`
- historical Binance Futures `aggTrades` can be converted into realtime-style event streams
- control actions go through a Redis-backed command channel instead of mutating bot state directly

## Structure

```text
scrooge/
├── api/                     # FastAPI control plane backend
├── frontend/                # Next.js control plane frontend
├── bot/                     # Live runtime, control polling, state persistence, exchange adapters
├── core/                    # Shared engine, event model, indicator-input selection, event store
├── backtest/                # Dataset build, event-stream build, compare, optimize, reporting
├── config/                  # Live/backtest/compare/grid configs
├── docker/                  # Dockerfiles and entrypoints
├── requirements/            # Split dependency sets
├── runtime/                 # Local runtime/backtest artifacts (gitignored)
├── docker-compose.yml
└── main.py                  # Thin entry shim
```

## Key Concepts

### Strategy Modes

- `strategy_mode: discrete`
  - legacy minute-snapshot style evaluation
  - decisions are taken once per closed `small` candle
- `strategy_mode: realtime`
  - event-driven evaluation on `price_tick`, `candle_closed`, `indicator_snapshot`, and account/order events
  - used both in live mode and in realtime historical replay

### Indicator Inputs

Strategy timing and strategy decision values are separated:
- `strategy_mode` decides **when** the strategy evaluates
- `indicator_inputs` decide **which indicator values** are used for the decision

Supported `indicator_inputs` keys:
- `ema`
- `rsi`
- `bb`
- `atr`

Supported modes:
- `closed` — use the last closed-candle indicator value
- `intrabar` — use the current intrabar/realtime indicator value

### Backtest Input Modes

- `build`
  - fetch historical candles, build dataset, derive tape, synthesize a historical event stream
- `discrete_tape`
  - replay from an existing `market_tape.jsonl`
- `market_event_stream`
  - replay directly from an existing `market_events.jsonl`
- `agg_trade_stream`
  - build a native historical market-event stream from Binance Futures `aggTrades`

## Installation

### Python runtime and backtest stack

```bash
git clone https://github.com/vk0n/scrooge.git
cd scrooge

python -m venv scrooge-env
source scrooge-env/bin/activate
pip install -r requirements/full.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Environment

Create `.env` with at least:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
SCROOGE_GUI_USERNAME=admin
SCROOGE_GUI_PASSWORD=strong_password_here
```

Optional machine-to-machine control token:

```env
SCROOGE_CONTROL_TOKEN=...
```

Optional push notifications:

```env
SCROOGE_PUSH_ENABLED=1
SCROOGE_PUSH_VAPID_SUBJECT=mailto:you@example.com
SCROOGE_PUSH_VAPID_PRIVATE_KEY=
SCROOGE_PUSH_VAPID_PUBLIC_KEY=
```

## Config Files

Primary working configs:
- [config/live.yaml](config/live.yaml)
- [config/backtest.yaml](config/backtest.yaml)
- [config/compare.yaml](config/compare.yaml)
- [config/param_grid.yaml](config/param_grid.yaml)

Treat checked-in configs as working presets:
- they are meant to be copied, edited, and compared
- they are not a promise that every checked-in file is the final production strategy preset
- the currently checked-in `live.yaml` and `backtest.yaml` already point to the tuned realtime winner:
  - `strategy_mode: realtime`
  - `indicator_inputs`: `ema=intrabar`, `rsi=closed`, `bb=closed`, `atr=intrabar`

## Running Scrooge

### Live runtime

By default:
- `main.py` loads `config/live.yaml`
- `config/live.yaml` currently runs `strategy_mode: realtime`
- the checked-in live preset also uses the tuned realtime indicator-input mix:
  - `ema: intrabar`
  - `rsi: closed`
  - `bb: closed`
  - `atr: intrabar`

Run locally:

```bash
python main.py
```

Equivalent direct entry:

```bash
python -m bot.runtime
```

Default live artifacts:
- `runtime/state.json`
- `runtime/trading_log.txt`
- `runtime/trade_history.jsonl`
- `runtime/balance_history.jsonl`
- `runtime/event_history.jsonl`
- `runtime/market_events.jsonl`
- `runtime/chart_dataset.csv`

Live runtime behavior:
- websocket-driven market stream
- websocket-driven user/account stream
- canonical append-only event log
- Redis-backed control command queue
- push notifications via web push when configured

### Local control plane

API:

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open:
- `http://localhost:3000`

Primary pages:
- `Office` → `/dashboard`
- `Market Map` → `/chart`
- `Ledger` → `/logs`

The control plane provides:
- status and open-trade visibility
- config editing
- manual control commands
- live/polling updates
- charting from runtime/backtest artifacts
- push notification subscription management

### Local backtests

Run against `config/backtest.yaml`:

```bash
SCROOGE_CONFIG_PATH=config/backtest.yaml python main.py
```

Equivalent direct entry:

```bash
SCROOGE_CONFIG_PATH=config/backtest.yaml python -m bot.runtime
```

Backtest runs automatically isolate artifacts when `backtest_run_dir: auto`:
- `runtime/backtests/<run_id>/...`
- `runtime/backtests/latest`

The checked-in backtest preset mirrors the tuned realtime baseline:
- `strategy_mode: realtime`
- `backtest_input_mode: agg_trade_stream`
- `agg_trade_tick_interval: 5s`
- `indicator_inputs`: `ema=intrabar`, `rsi=closed`, `bb=closed`, `atr=intrabar`

Typical backtest artifacts:
- `state.json`
- `trade_history.jsonl`
- `balance_history.jsonl`
- `trading_log.txt`
- `event_history.jsonl`
- `market_tape.jsonl`
- `market_events.jsonl`
- `chart_dataset.csv`
- `replay_summary.json`
- `replay_trades.jsonl`
- `market_event_execution_summary.json`
- `market_event_execution_events.jsonl`
- `market_event_execution_fills.jsonl`
- `market_event_execution_trades.jsonl`
- `market_event_trade_alignment_summary.json`
- `market_event_trade_alignment_pairs.jsonl`

### Historical aggTrades replay

Example:

```yaml
backtest_input_mode: agg_trade_stream
strategy_mode: realtime
execution_mode: simulated
agg_trade_source: archive
agg_trade_tick_interval: 5s
```

Notes:
- `archive` is the main long-range source
- `rest` is mostly useful for short windows or the latest missing tail
- `agg_trade_tick_interval` supports `raw`, `1s`, `5s`, `15s`, `30s`, `60s`, ...
- raw archive data is cached under `data/agg_trades`
- archive cache is sharded per UTC day under `data/agg_trades/<symbol>/archive_daily`

## Compare Runs and Sieves

Run a compare matrix:

```bash
python -m backtest.compare
```

Compare runs:
- use `config/backtest.yaml` as a base
- apply scenario overrides
- store one artifact set per scenario under `runtime/compare/<run_id>/scenarios/`
- can run scenarios in parallel worker processes

Main compare artifacts:
- `compare_summary.json`
- `compare_runs.jsonl`
- `compare_table.md`
- `compare_stage_decisions.json`
- `compare_sieves_summary.json`
- `compare_sieves_table.md`

Current compare flow also supports multi-stage sieve screening:
- `month`
- `quarter`
- `half_year`

Candidates that fail an earlier stage are skipped for later stages.

### Generate large candidate matrices

```bash
python -m backtest.generate_compare_candidates \
  --compare-template config/compare.yaml \
  --scenario-template-name realtime-30d-5s-closed \
  --param-grid config/param_grid.yaml \
  --output /tmp/scrooge.compare.generated.yaml \
  --candidate-prefix candidate
```

This clones one scenario from `config/compare.yaml`, expands the parameter grid from `config/param_grid.yaml`, and writes a generated compare config you can run with:

```bash
SCROOGE_COMPARE_CONFIG_PATH=/tmp/scrooge.compare.generated.yaml python -m backtest.compare
```

## Optimization

Run optimizer:

```bash
python -m backtest.optimize
```

The optimizer reads:
- [config/param_grid.yaml](config/param_grid.yaml)

## Reporting

Backtests can emit unified run reports from [backtest/reporting.py](backtest/reporting.py).

When reporting is enabled, outputs are written alongside the run artifacts:
- `report.json`
- `report.html` when `enable_plot: true`

Current report surface includes:
- overview chart
- trade diagnostics
- monthly returns heatmap
- drawdown diagnostics
- Monte Carlo summary
- rolling-window distribution
- rolling-window timeline
- yearly breakdown
- collapsible trade history

## Docker / Compose

For the compose deploy model, live profile, backtest profile, runtime volumes, and update flow, see:
- [DEPLOY_DOCKER.md](DEPLOY_DOCKER.md)

API details:
- [api/README.md](api/README.md)

Frontend details:
- [frontend/README.md](frontend/README.md)

## Current Architecture Summary

Scrooge is no longer just a discrete minute-polling bot.

The project now has:
- a realtime live path driven by exchange events
- a historical realtime replay path driven by stored or reconstructed `market_events.jsonl`
- a shared engine for live and replay execution
- a control plane that talks to the bot through queued commands
- a reporting/compare workflow for tuning and validation

The main practical split is now:
- `5s` replay for research and tuning
- `1s` replay for higher-fidelity validation

## Disclaimer

Scrooge is provided for educational and research purposes only.
Cryptocurrency trading involves substantial risk and may result in total capital loss.
Use this software at your own discretion.

## License

MIT License

## Philosophy of Scrooge

Scrooge is not human - and that's his greatest strength.
He does not hope, fear, hesitate, or overthink.
Where humans trade with emotion, Scrooge trades with mathematics.

He sees the market as a field of probabilities, not possibilities.
He doesn't chase euphoria or revenge after loss.
Every position is just data, every outcome a statistical event.

While traders battle psychology, Scrooge operates in logic.
He never over-leverages from greed, never panics on a red candle.
He acts only when signals align, exits only when math demands it.

Scrooge represents the elimination of emotion and the automation of discipline.
His only loyalty is to the algorithm - consistent, adaptive, and entirely emotionless.
