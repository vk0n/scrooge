# Scrooge Control Plane: Docker Compose profiles (live + backtest)

Architecture:
- `api` container: FastAPI control plane
- `frontend` container: Next.js UI
- `proxy` container: Nginx entrypoint (single public port)
- `redis` container: command queue and command status storage
- `bot` service (`profile: live`): long-running live trading runtime
- `backtest` service (`profile: backtest`): one-shot backtest runner job

All services use shared Docker volume `scrooge_runtime`.

## 1. Prepare configs

Live config:
- `config.yaml` (`live: true`)

Backtest config:
- `config.backtest.yaml` (`live: false`)

## 2. Prepare `.env`

```bash
cp .env.example .env
```

Set at least:

```env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
SCROOGE_GUI_USERNAME=admin
SCROOGE_GUI_PASSWORD=strong_password_here
```

For registry-based deploys with Watchtower, also set image refs:

```env
SCROOGE_BOT_IMAGE=vk0n/scrooge-bot:latest
SCROOGE_BACKTEST_IMAGE=vk0n/scrooge-backtest:latest
SCROOGE_API_IMAGE=vk0n/scrooge-api:latest
SCROOGE_FRONTEND_IMAGE=vk0n/scrooge-frontend:latest
SCROOGE_PROXY_IMAGE=nginx:1.27-alpine
SCROOGE_WATCHTOWER_SCOPE=scrooge
SCROOGE_WATCHTOWER_POLL_INTERVAL=300
```

If your registry requires auth, make sure the deployment host is already logged in, for example:

```bash
docker login
```

Note:
- `bot` uses a slim live-runtime image
- `backtest` uses a separate image with plotting/report dependencies

## 3. Run modes

Control plane only (no bot process):

```bash
docker compose up -d --build
```

Control plane + Watchtower auto-update:

```bash
docker compose --profile watchtower up -d
```

Control plane + live bot:

```bash
docker compose --profile live up -d --build
```

Control plane + live bot + Watchtower auto-update:

```bash
docker compose --profile live --profile watchtower up -d
```

Optional live socket knobs:

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
SCROOGE_EVENT_LOG_FILE=/runtime/event_history.jsonl
```

Notes:
- market sockets drive live price and closed-candle triggers
- user stream drives faster balance/position updates
- if the user stream goes stale beyond `SCROOGE_USER_STREAM_STALE_AFTER_SECONDS`, the bot falls back to REST reads until freshness is restored

Important for Watchtower:
- it watches only services labeled with `com.centurylinklabs.watchtower.enable=true`
- it is scoped by `SCROOGE_WATCHTOWER_SCOPE`
- it can auto-update only pullable images, so `SCROOGE_*_IMAGE` should point to registry tags, not just local-only build names
- `backtest` is explicitly excluded from Watchtower updates

One-shot backtest run:

```bash
docker compose --profile backtest run --rm backtest
```

Optional host export for backtest artifacts:
- Set in `config.backtest.yaml`:
  - `export_artifacts_to_host: true`
  - `host_artifacts_dir: /host_artifacts`
- Host bind path is controlled by `.env`:
  - `SCROOGE_HOST_BACKTEST_ARTIFACTS_DIR=./backtest_artifacts`
- Output appears in:
  - `./backtest_artifacts/<run_ts>/...`
  - symlink `./backtest_artifacts/latest`

Backtest artifacts are written into:
- `/runtime/backtests/<run_id>/...`
- symlink `/runtime/backtests/latest` points to last run
- each run now also emits:
  - `event_history.jsonl` (canonical append-only event log)
  - `replay_summary.json` (summary reconstructed from canonical events)
  - `replay_trades.jsonl` (trade timeline reconstructed from canonical events)

Open UI:
- `http://<host>:3000`

## 4. Runtime data model (live)

- API edits/reads `config.yaml` via `/runtime/config.yaml`.
- Live runtime writes:
  - `/runtime/state.json` (runtime snapshot only)
  - `/runtime/trade_history.jsonl`
  - `/runtime/balance_history.jsonl`
  - `/runtime/trading_log.txt`
  - `/runtime/event_history.jsonl` (canonical append-only event log)
  - `/runtime/chart_dataset.csv`

## 5. Control behavior

`/api/control/*` commands are queued via Redis.

Important:
- Commands are processed only when `bot` service is running (`--profile live`).
- `start` = resume trading loop
- `stop` = pause trading loop
- `restart` = resume + config reload

## 6. Stop / cleanup

Stop stack:

```bash
docker compose down
```

Stop and remove runtime volume:

```bash
docker compose down -v
```
