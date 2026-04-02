# Scrooge Docker Deploy

Scrooge uses Docker Compose profiles for:
- control plane only
- live trading runtime
- one-shot backtests

The stack consists of:
- `redis` — command queue and command status storage
- `api` — FastAPI control plane backend
- `frontend` — Next.js control plane UI
- `proxy` — nginx public entrypoint
- `bot` — live trading runtime (`profile: live`)
- `backtest` — one-shot backtest runner (`profile: backtest`)
- `watchtower` — optional auto-update service (`profile: watchtower`)

Persistent runtime state lives in the `scrooge_runtime` Docker volume.

For local stand/debug work, you can switch `/runtime` to a bind mount with
`docker-compose.stand.yml`.

## Config Files

Main mounted configs:
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

Optional control token:

```env
SCROOGE_CONTROL_TOKEN=...
```

Optional image refs for registry deploys:

```env
SCROOGE_BOT_IMAGE=vk0n/scrooge-bot:latest
SCROOGE_BACKTEST_IMAGE=vk0n/scrooge-backtest:latest
SCROOGE_API_IMAGE=vk0n/scrooge-api:latest
SCROOGE_FRONTEND_IMAGE=vk0n/scrooge-frontend:latest
SCROOGE_PROXY_IMAGE=nginx:1.27-alpine
```

Optional push notifications:

```env
SCROOGE_PUSH_ENABLED=1
SCROOGE_PUSH_VAPID_SUBJECT=mailto:you@example.com
SCROOGE_PUSH_VAPID_PRIVATE_KEY=
SCROOGE_PUSH_VAPID_PUBLIC_KEY=
```

These variables are supported by the API/runtime code, but if you use plain Compose you should pass them into the `api` and `bot` services through an override file or explicit `environment:` entries.

Optional Watchtower:

```env
SCROOGE_WATCHTOWER_SCOPE=scrooge
SCROOGE_WATCHTOWER_POLL_INTERVAL=300
SCROOGE_WATCHTOWER_HTTP_API_TOKEN=
```

If your registry requires auth:

```bash
docker login
```

## Run Modes

Control plane only:

```bash
docker compose up -d --build
```

Local stand/debug with bind-mounted `./runtime`:

```bash
docker compose -f docker-compose.yml -f docker-compose.stand.yml up -d --build
```

Control plane + Watchtower:

```bash
docker compose --profile watchtower up -d
```

Control plane + live bot:

```bash
docker compose --profile live up -d --build
```

Control plane + live bot with bind-mounted `./runtime`:

```bash
docker compose -f docker-compose.yml -f docker-compose.stand.yml --profile live up -d --build
```

Control plane + live bot + Watchtower:

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

### Build on host

```bash
docker compose --profile live up -d --build --force-recreate
```

### Pull from registry

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
- `--force-recreate` recreates containers from already-present images
- it does not fetch new remote images by itself
- for registry deploys, `pull` first

## Live Runtime Notes

Compose mounts [config/live.yaml](config/live.yaml) into the bot and API runtime volume.

Current live runtime is:
- websocket-driven for market data
- websocket-driven for user/account updates
- event-driven when `strategy_mode: realtime`
- command-driven via Redis queue for control actions

Important:
- `config/live.yaml` is currently the main source of truth for live `strategy_mode`
- the Compose env var `SCROOGE_STRATEGY_MODE` is still available as an override, but if `strategy_mode` is set in the YAML config, the YAML value wins
- the checked-in live preset currently runs the tuned realtime winner:
  - `strategy_mode: realtime`
  - `indicator_inputs`: `ema=intrabar`, `rsi=closed`, `bb=closed`, `atr=intrabar`
  - tuned params from `config/live.yaml`

Useful live env knobs:

```env
SCROOGE_MARKET_STREAM_ENABLED=1
SCROOGE_MARKET_STREAM_PERSIST_INTERVAL_SECONDS=1
SCROOGE_MARKET_STREAM_SETTLE_SECONDS=1.5
SCROOGE_MARKET_EVENT_STREAM_ENABLED=1
SCROOGE_MARKET_EVENT_STREAM_FILE=/runtime/market_events.jsonl
SCROOGE_USER_STREAM_ENABLED=1
SCROOGE_USER_STREAM_KEEPALIVE_SECONDS=1800
SCROOGE_USER_STREAM_RECONNECT_SECONDS=5
SCROOGE_USER_STREAM_STALE_AFTER_SECONDS=120
SCROOGE_FUTURES_REST_BASE_URL=https://fapi.binance.com
SCROOGE_FUTURES_WS_BASE_URL=wss://fstream.binance.com/ws
SCROOGE_RUNTIME_MODE=live
SCROOGE_STRATEGY_MODE=realtime
SCROOGE_DEBUG_STRATEGY_TICKS=0
```

Behavior:
- market sockets drive price ticks and candle-close updates
- user stream drives balance, position, and order updates
- if user-stream cache goes stale, the bot falls back to REST reads until freshness is restored

## Runtime Artifacts

Live services write into `/runtime`:
- `scrooge.sqlite3`
- `event_history.jsonl`
- `market_events.jsonl`
- `chart_dataset.csv`
- `push_subscriptions.json`
- generated VAPID key files when push is enabled and keys are not pre-supplied

Notes:
- `scrooge.sqlite3` is the source of truth for runtime state, trade history, balance history, and Ledger/UI log lines
- `event_history.jsonl` remains a replay/debug event artifact mirrored alongside DB records
- `market_events.jsonl` carries both market and account/execution events
- runtime artifacts survive normal upgrades as long as the `scrooge_runtime` volume is preserved

## Backtest Artifacts

Backtest runs write into:
- `/runtime/backtests/<run_id>/...`
- symlink `/runtime/backtests/latest`

If host export is enabled:
- set `export_artifacts_to_host: true` in [config/backtest.yaml](config/backtest.yaml)
- host bind target comes from:
  - `SCROOGE_HOST_BACKTEST_ARTIFACTS_DIR=./backtest_artifacts`

Exported output appears in:
- `./backtest_artifacts/<run_id>/...`
- symlink `./backtest_artifacts/latest`

Typical run artifacts:
- `scrooge.sqlite3`
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
- `report.json`
- `report.html` when `enable_plot: true`

Notes:
- `scrooge.sqlite3` is the canonical backtest store for replay state, trade history, balance history, and UI log lines
- `market_event_execution_summary.json` summarizes the active execution path
- `market_event_trade_alignment_summary.json` compares strategy trades to observed execution trades
- `report.json` is the machine-readable summary
- `report.html` is the unified human-readable report page

Backtest input modes:
- `build`
- `discrete_tape`
- `market_event_stream`
- `agg_trade_stream`

`agg_trade_stream` notes:
- historical Binance `aggTrades` are cached under `data/agg_trades`
- archive cache is sharded per UTC day under `data/agg_trades/<symbol>/archive_daily`
- overlapping windows reuse the same raw day files

## Control Behavior

`/api/control/*` commands are queued through Redis and executed asynchronously by the bot.

Important:
- commands are processed only when the `bot` service is running with `--profile live`
- `start` = resume trading
- `stop` = pause trading
- `restart` = resume trading + reload config
- `close-position` = close active position
- `suggest-trade` = queue a manual buy/sell suggestion
- `update-sl` / `update-tp` = update active levels

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
