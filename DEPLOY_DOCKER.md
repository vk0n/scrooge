# Scrooge Docker Deploy

Scrooge uses Docker Compose profiles for three main modes:
- control plane only
- live trading runtime
- one-shot backtest jobs

The stack is built from:
- `api` — FastAPI control plane backend
- `frontend` — Next.js UI
- `proxy` — nginx public entrypoint
- `redis` — command queue and command status storage
- `bot` — live trading runtime (`profile: live`)
- `backtest` — one-shot backtest runner (`profile: backtest`)
- `watchtower` — optional auto-update service (`profile: watchtower`)

All services share the Docker volume `scrooge_runtime`.

## Config Files

Live trading config:
- `config/live.yaml`

Backtest config:
- `config/backtest.yaml`

Optimization grid:
- `config/param_grid.yaml`

Compose mounts:
- `config/live.yaml -> /runtime/config.yaml`
- `config/backtest.yaml -> /runtime/config.backtest.yaml`

## Prepare `.env`

```bash
cp .env.example .env
```

At minimum set:

```env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
SCROOGE_GUI_USERNAME=admin
SCROOGE_GUI_PASSWORD=strong_password_here
```

For registry-based deploys also set image refs:

```env
SCROOGE_BOT_IMAGE=vk0n/scrooge-bot:latest
SCROOGE_BACKTEST_IMAGE=vk0n/scrooge-backtest:latest
SCROOGE_API_IMAGE=vk0n/scrooge-api:latest
SCROOGE_FRONTEND_IMAGE=vk0n/scrooge-frontend:latest
SCROOGE_PROXY_IMAGE=nginx:1.27-alpine
```

Optional Watchtower settings:

```env
SCROOGE_WATCHTOWER_SCOPE=scrooge
SCROOGE_WATCHTOWER_POLL_INTERVAL=300
SCROOGE_WATCHTOWER_HTTP_API_TOKEN=
```

If your registry requires auth, log in on the deployment host first:

```bash
docker login
```

## Run Modes

Control plane only:

```bash
docker compose up -d --build
```

Control plane with Watchtower:

```bash
docker compose --profile watchtower up -d
```

Control plane with live bot:

```bash
docker compose --profile live up -d --build
```

Control plane with live bot and Watchtower:

```bash
docker compose --profile live --profile watchtower up -d
```

One-shot backtest:

```bash
docker compose --profile backtest run --rm backtest
```

Open UI:

```text
http://<host>:3000
```

## Updating

### Local build on host

```bash
docker compose --profile live up -d --build --force-recreate
```

### Registry-based deploy

```bash
docker compose --profile live pull
docker compose --profile live up -d --force-recreate
```

If you also run Watchtower:

```bash
docker compose --profile live --profile watchtower pull
docker compose --profile live --profile watchtower up -d --force-recreate
```

Notes:
- `--force-recreate` recreates containers from the images you already have locally
- it does not fetch newer remote images by itself
- for registry deploys, run `pull` first

## Live Runtime Notes

Current live runtime is:
- websocket-driven for market data
- websocket-driven for user/account updates
- still `strategy_mode=discrete` by default

Useful live knobs from `.env`:

```env
SCROOGE_MARKET_STREAM_ENABLED=1
SCROOGE_MARKET_STREAM_PERSIST_INTERVAL_SECONDS=1
SCROOGE_MARKET_STREAM_SETTLE_SECONDS=1.5
SCROOGE_USER_STREAM_ENABLED=1
SCROOGE_USER_STREAM_KEEPALIVE_SECONDS=1800
SCROOGE_USER_STREAM_RECONNECT_SECONDS=5
SCROOGE_USER_STREAM_STALE_AFTER_SECONDS=120
SCROOGE_FUTURES_REST_BASE_URL=https://fapi.binance.com
SCROOGE_FUTURES_WS_BASE_URL=wss://fstream.binance.com/ws
SCROOGE_RUNTIME_MODE=live
SCROOGE_STRATEGY_MODE=discrete
SCROOGE_DEBUG_STRATEGY_TICKS=0
```

Behavior:
- market sockets drive live price and closed-candle triggers
- user stream drives faster balance and position updates
- if user-stream cache goes stale, the bot falls back to REST reads until freshness is restored

## Runtime Artifacts

Live runtime writes into `/runtime`:
- `state.json`
- `trade_history.jsonl`
- `balance_history.jsonl`
- `trading_log.txt`
- `event_history.jsonl`
- `chart_dataset.csv`

Notes:
- `event_history.jsonl` is the canonical append-only event log
- existing runtime artifacts remain compatible across normal upgrades as long as the `scrooge_runtime` volume is preserved

## Backtest Artifacts

Backtest runs write into:
- `/runtime/backtests/<run_id>/...`
- symlink `/runtime/backtests/latest`

If host export is enabled:
- set `export_artifacts_to_host: true` in `config/backtest.yaml`
- host bind target is controlled by:
  - `SCROOGE_HOST_BACKTEST_ARTIFACTS_DIR=./backtest_artifacts`

Exported output appears in:
- `./backtest_artifacts/<run_id>/...`
- symlink `./backtest_artifacts/latest`

Each backtest run emits:
- `state.json`
- `trade_history.jsonl`
- `balance_history.jsonl`
- `trading_log.txt`
- `market_tape.jsonl`
- `event_history.jsonl`
- `chart_dataset.csv`
- `replay_summary.json`
- `replay_trades.jsonl`

## Control Behavior

`/api/control/*` commands are queued through Redis.

Important:
- commands are processed only when `bot` is running with `--profile live`
- `start` resumes trading
- `stop` pauses trading
- `restart` resumes trading and reloads config

## Stop / Cleanup

Stop the stack:

```bash
docker compose down
```

Stop and remove the runtime volume:

```bash
docker compose down -v
```

Use `down -v` only if you intentionally want to wipe `/runtime`.
