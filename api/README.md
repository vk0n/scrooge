# Scrooge Control API

FastAPI backend for the Scrooge control plane.

It provides:
- runtime status
- chart payloads
- log streaming support
- config read/write endpoints
- Redis-backed control commands
- web push notification setup

## Run Locally

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Default CORS origins:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

Override:

```bash
export SCROOGE_GUI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://YOUR_HOST:3000
```

## Auth Model

- all `/api/*` endpoints require HTTP Basic auth
- `/ws` endpoints also require auth
- `/api/control/*` additionally accept `X-Scrooge-Control-Token`

Relevant env vars:
- `SCROOGE_GUI_USERNAME`
- `SCROOGE_GUI_PASSWORD`
- `SCROOGE_CONTROL_TOKEN`

## Runtime Path Env

The API reads bot/runtime artifacts from these paths:
- `SCROOGE_CONFIG_PATH`
- `SCROOGE_STATE_PATH`
- `SCROOGE_TRADE_HISTORY_PATH`
- `SCROOGE_BALANCE_HISTORY_PATH`
- `SCROOGE_LOG_PATH`

Chart-specific env:
- `SCROOGE_CHART_SOURCE` (`auto`, `dataset`, `binance`)
- `SCROOGE_CHART_MAX_CANDLES`
- `SCROOGE_CHART_DATASET_MAX_CANDLES`
- `SCROOGE_CHART_TIMEOUT_SECONDS`
- `SCROOGE_CHART_DATASET_PATH`

Redis/control env:
- `SCROOGE_REDIS_HOST`
- `SCROOGE_REDIS_PORT`
- `SCROOGE_REDIS_DB`
- `SCROOGE_CONTROL_QUEUE_KEY`
- `SCROOGE_COMMAND_STATUS_PREFIX`
- `SCROOGE_COMMAND_STATUS_TTL_SECONDS`
- `SCROOGE_WS_PUSH_INTERVAL_SECONDS`
- `SCROOGE_WS_LOG_LINES`

Push env:
- `SCROOGE_PUSH_ENABLED`
- `SCROOGE_PUSH_VAPID_SUBJECT`
- `SCROOGE_PUSH_VAPID_PRIVATE_KEY`
- `SCROOGE_PUSH_VAPID_PUBLIC_KEY`
- `SCROOGE_PUSH_SUBSCRIPTIONS_FILE`
- `SCROOGE_PUSH_VAPID_PRIVATE_KEY_FILE`
- `SCROOGE_PUSH_VAPID_PUBLIC_KEY_FILE`

## Main Endpoints

### Health

- `GET /health`
- `GET /`

### Runtime status

- `GET /api/status`

Returns:
- bot running/paused state
- current symbol and leverage
- balance
- last price and timestamp
- open trade info
- trailing state
- warnings

### Logs

- `GET /api/logs?lines=200`

### Chart

- `GET /api/chart?symbol=BTCUSDT&period=1d&interval=1m&indicators=true&source=auto`

Supports:
- runtime dataset-backed charts
- runtime/backtest artifact-backed charts
- direct Binance fallback when configured

### Config

- `GET /api/config`
- `GET /api/config/editable`
- `POST /api/config/editable`
- `GET /api/config/raw`
- `POST /api/config/raw`

Editable config includes:
- top-level runtime fields such as `strategy_mode`, `symbol`, `leverage`, `qty`
- timeframe `intervals`
- `indicator_inputs`
- selected strategy params

Important:
- editable writes create a config backup first
- responses return `backup_path`
- responses also indicate whether `restart_required`

`indicator_inputs` supported values:
- `closed`
- `intrabar`

### Control

- `POST /api/control/start`
- `POST /api/control/stop`
- `POST /api/control/restart`
- `POST /api/control/close-position`
- `POST /api/control/suggest-trade`
- `POST /api/control/update-sl`
- `POST /api/control/update-tp`
- `GET /api/control/commands/{command_id}`

Semantics:
- `start` = resume trading
- `stop` = pause trading
- `restart` = resume trading + reload config
- `close-position` = manual close active position
- `suggest-trade` = queue manual buy/sell suggestion
- `update-sl` / `update-tp` = adjust current position levels

Commands are queued through Redis and executed asynchronously by the live bot loop.

### Notifications

- `GET /api/notifications`
- `POST /api/notifications/subscribe`
- `POST /api/notifications/unsubscribe`
- `POST /api/notifications/test`

This powers the bell control in the frontend and stores subscriptions in runtime storage.

### WebSocket

- `ws://localhost:8000/ws`
- `ws://localhost:8000/ws/status`

Pushes:
- status snapshots
- log snapshots

The frontend uses websocket updates first and falls back to polling when needed.
