# Scrooge — Binance Futures Runtime, Control Plane, and Replayable Backtests

Scrooge is an event-driven trading system for Binance Futures with three cooperating layers:
- `bot/` for live runtime, exchange integration, and stateful execution
- `core/` for shared strategy and event logic
- `backtest/` for dataset building, discrete backtest running, replay, reporting, and optimization

It currently supports:
- live trading with websocket-driven market and account updates
- a local control plane (`api/` + `frontend/`)
- discrete backtests
- canonical event logging and replay artifacts
- a canonical discrete market tape for backtest runs

## Project Structure

```text
scrooge/
├── api/                     # FastAPI control plane backend
├── frontend/                # Next.js control plane frontend
├── bot/                     # Live runtime adapters and side effects
│   ├── runtime.py
│   ├── market_stream.py
│   ├── control_channel.py
│   ├── event_log.py
│   ├── state.py
│   └── trade.py
├── core/                    # Shared engine and canonical event storage
│   ├── engine.py
│   └── event_store.py
├── backtest/                # Dataset building, discrete runner, replay, reporting, optimization
│   ├── dataset.py
│   ├── runner.py
│   ├── replay.py
│   ├── reporting.py
│   └── optimize.py
├── config/
│   ├── live.yaml
│   ├── backtest.yaml
│   └── param_grid.yaml
├── requirements/
│   ├── bot.txt
│   ├── backtest.txt
│   └── full.txt
├── runtime/                 # Local default runtime artifacts (gitignored)
├── docker/
├── main.py                  # Thin entry shim; keeps the Scrooge greeting
└── docker-compose.yml
```

## Installation

### 1. Clone

```bash
git clone https://github.com/vk0n/scrooge.git
cd scrooge
```

### 2. Create a virtual environment

```bash
python -m venv scrooge-env
source scrooge-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements/full.txt
```

Notes:
- `requirements/full.txt` is the full local toolchain
- `docker/bot.Dockerfile` installs `requirements/bot.txt`
- `docker/backtest.Dockerfile` installs `requirements/backtest.txt`

### 4. Create `.env`

At minimum:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

If you run the control plane locally, also set:

```env
SCROOGE_GUI_USERNAME=admin
SCROOGE_GUI_PASSWORD=strong_password_here
```

## Configuration

Live trading config:
- [live.yaml](config/live.yaml)

Backtest config:
- [backtest.yaml](config/backtest.yaml)

Optimization grid:
- [param_grid.yaml](config/param_grid.yaml)

Current canonical timeframe setup is:
- `small`: price/decision frame
- `medium`: Bollinger Bands + ATR frame
- `big`: RSI + EMA frame

## Usage

### Local live runtime

Uses `config/live.yaml` by default:

```bash
python main.py
```

Local runtime artifacts will appear under:
- `runtime/state.json`
- `runtime/trading_log.txt`
- `runtime/trade_history.jsonl`
- `runtime/balance_history.jsonl`
- `runtime/event_history.jsonl`
- `runtime/chart_dataset.csv`

### Local backtest

Run against `config/backtest.yaml`:

```bash
SCROOGE_CONFIG_PATH=config/backtest.yaml python main.py
```

Backtest outputs include:
- runtime artifacts for the run
- `market_tape.jsonl`
- canonical `event_history.jsonl`
- `replay_summary.json`
- `replay_trades.jsonl`

The backtest execution path is now owned by:
- `backtest/runner.py`
- `backtest/tape.py`

Backtest input modes:
- `backtest_input_mode: build` — fetch/build dataset, then derive `market_tape.jsonl`
- `backtest_input_mode: tape` — start directly from an existing `market_tape.jsonl`

To replay from an existing tape, set in `config/backtest.yaml`:

```yaml
backtest_input_mode: tape
market_tape_input_path: /path/to/market_tape.jsonl
```

### Replay a canonical event log

```bash
python -m backtest.replay /path/to/event_history.jsonl --runtime-mode backtest --strategy-mode discrete
```

### Parameter optimization

```bash
python -m backtest.optimize
```

The optimizer reads:
- `config/param_grid.yaml`

### Docker / Compose

For current compose-based deploy flow, profiles, and runtime model, see:
- [DEPLOY_DOCKER.md](DEPLOY_DOCKER.md)

## Strategy Outline

Scrooge currently runs a multi-timeframe discrete strategy:
- entries are evaluated from the `small` frame with `medium` and `big` filters
- stop loss and take profit are ATR-based
- trailing protection can activate after price moves beyond the base target
- state, trade history, balance history, UI log, and canonical event history are persisted

The longer-term direction of the project is:
- keep `discrete` mode as the stable baseline
- build toward shared-core replay/backtest infrastructure
- later add a separate `realtime` strategy mode without losing the ability to compare it against the discrete baseline

## Disclaimer

Scrooge is provided for educational and research purposes only.  
Cryptocurrency trading involves substantial risk and may result in total capital loss.  
Use this software at your own discretion.

## License

MIT License


## Philosophy of Scrooge

Scrooge is not human — and that’s his greatest strength.
He does not hope, fear, hesitate, or overthink.
Where humans trade with emotion, Scrooge trades with mathematics.

He sees the market as a field of probabilities, not possibilities.
He doesn’t chase euphoria or revenge after loss.
Every position is just data, every outcome a statistical event.

While traders battle psychology, Scrooge operates in logic.
He never over-leverages from greed, never panics on a red candle.
He acts only when signals align, exits only when math demands it.

Scrooge represents the elimination of emotion and the automation of discipline.
His only loyalty is to the algorithm — consistent, adaptive, and entirely emotionless.
