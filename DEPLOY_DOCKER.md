# Scrooge Control Plane: Docker Compose 3+1 (+ bot command queue)

Architecture:
- `bot` container: trading runtime (`main.py`)
- `api` container: FastAPI control plane
- `frontend` container: Next.js UI
- `proxy` container: Nginx entrypoint (single public port)
- `redis` container: command queue and command status storage

`bot` and `api` use one shared Docker volume (`scrooge_runtime`) for runtime files.

## 1. Prepare `.env`

Create `.env` next to `docker-compose.yml`:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

SCROOGE_GUI_USERNAME=admin
SCROOGE_GUI_PASSWORD=strong_password_here

# Optional for machine integrations (without Basic auth) on /api/control/*
SCROOGE_CONTROL_TOKEN=long_random_token_here

# Queue settings
SCROOGE_REDIS_HOST=redis
SCROOGE_REDIS_PORT=6379
SCROOGE_REDIS_DB=0
SCROOGE_CONTROL_QUEUE_KEY=scrooge:control:queue
SCROOGE_COMMAND_STATUS_PREFIX=scrooge:control:command:
SCROOGE_COMMAND_STATUS_TTL_SECONDS=86400

# Bot loop tuning
SCROOGE_LIVE_POLL_SECONDS=60
SCROOGE_CONTROL_POLL_SLICE_SECONDS=1

# WebSocket tuning
SCROOGE_WS_PUSH_INTERVAL_SECONDS=2
SCROOGE_WS_LOG_LINES=200

# Charting
SCROOGE_CHART_MAX_CANDLES=1500
SCROOGE_CHART_DATASET_MAX_CANDLES=10000
SCROOGE_CHART_TIMEOUT_SECONDS=10
SCROOGE_CHART_SOURCE=auto
SCROOGE_CHART_DATASET_PATH=/runtime/chart_dataset.csv
SCROOGE_RUNTIME_CHART_DATASET_PATH=/runtime/chart_dataset.csv
```

## 2. Start stack

```bash
docker compose up -d --build
```

Open UI:
- `http://<host>:3000`

## 3. Runtime data model

- `config.yaml` from host is mounted read-only into `bot` and read-write into `api` (for limited editable config API).
- `state.json` (runtime snapshot only), `trade_history.jsonl`, `balance_history.jsonl`, `trading_log.txt`, and `chart_dataset.csv` are created/updated inside shared volume `scrooge_runtime`.
- Bot writes live/backtest candles to `chart_dataset.csv`; chart API can read from this dataset (`source=dataset`) and use Binance as fallback (`source=auto`).
- Restarting containers does not lose runtime data (volume persists).

## 4. Control behavior

`/api/control/*` is queued via Redis:
- API enqueues command (`start/stop/restart`) and returns `command_id`.
- `bot` consumes queue directly and updates command status.
- UI polls `GET /api/control/commands/{command_id}` for final status.

Control semantics:
- `start` = resume trading (container keeps running)
- `stop` = pause trading (container keeps running)
- `restart` = resume trading + reload `config.yaml` inside bot loop

If you need full process restart/stop, use Docker operations manually (`docker compose restart bot`, etc.).

## 5. Stop / cleanup

Stop services:
```bash
docker compose down
```

Stop and remove runtime volume:
```bash
docker compose down -v
```
