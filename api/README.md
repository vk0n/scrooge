# Scrooge Control API (Stage 5)

This is a Stage 5 backend for the Scrooge control plane.

## Run

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

For browser access from frontend, default allowed origins are:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

To override:
```bash
export SCROOGE_GUI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://YOUR_HOST:3000
```

## Notes

- API auth is required for all `/api/*` endpoints and `/ws`.
- Configure basic auth:
  - `SCROOGE_GUI_USERNAME`
  - `SCROOGE_GUI_PASSWORD`
- Optional control token for non-UI integrations:
  - `SCROOGE_CONTROL_TOKEN`
- Command queue settings:
  - `SCROOGE_REDIS_HOST`
  - `SCROOGE_REDIS_PORT`
  - `SCROOGE_REDIS_DB`
  - `SCROOGE_CONTROL_QUEUE_KEY`
  - `SCROOGE_COMMAND_STATUS_PREFIX`
  - `SCROOGE_COMMAND_STATUS_TTL_SECONDS`
  - `SCROOGE_WS_PUSH_INTERVAL_SECONDS`
  - `SCROOGE_WS_LOG_LINES`
  - `SCROOGE_CHART_MAX_CANDLES`
  - `SCROOGE_CHART_DATASET_MAX_CANDLES`
  - `SCROOGE_CHART_TIMEOUT_SECONDS`
  - `SCROOGE_CHART_SOURCE` (`auto`/`dataset`/`binance`)
  - `SCROOGE_CHART_DATASET_PATH` (CSV path or directory)
- Optional runtime file path overrides:
  - `SCROOGE_CONFIG_PATH`
  - `SCROOGE_STATE_PATH`
  - `SCROOGE_TRADE_HISTORY_PATH`
  - `SCROOGE_BALANCE_HISTORY_PATH`
  - `SCROOGE_LOG_PATH`

- Read endpoints:
  - `GET /health`
  - `GET /api/status`
  - `GET /api/chart?symbol=BTCUSDT&period=1d&interval=1m&indicators=true&source=auto`
  - `GET /api/config`
  - `GET /api/config/editable`
  - `GET /api/logs?lines=200`
- Config write endpoint:
  - `POST /api/config/editable`
  - Accepts only validated editable subset (`symbol`, `leverage`, `use_full_balance`, `qty`, selected `params.*`)
  - Creates `config.yaml` backup before overwrite and returns `backup_path`
  - Returns `restart_required` for save-and-restart flow
- Control endpoints:
  - `POST /api/control/start`
  - `POST /api/control/stop`
  - `POST /api/control/restart`
  - `POST /api/control/close-position`
  - `POST /api/control/update-sl` with JSON body `{ "value": <positive number> }`
  - `POST /api/control/update-tp` with JSON body `{ "value": <positive number> }`
  - `GET /api/control/commands/{command_id}`
  - Auth: either `Authorization: Basic ...` or header `X-Scrooge-Control-Token: ...`
  - Mutating actions are queued via Redis and executed asynchronously by bot runtime loop.
  - Semantics: `start` = resume trading, `stop` = pause trading, `restart` = resume + config reload, `close-position` = manual close active position, `update-sl`/`update-tp` = update current position levels.
  - Note: commands are processed only when live bot service is running (Compose `live` profile).
- `config.yaml`, `state.json`, `trade_history.jsonl`, `balance_history.jsonl`, and `trading_log.txt` are read from runtime paths.
- WebSocket live endpoints:
  - `ws://localhost:8000/ws`
  - `ws://localhost:8000/ws/status`
  - Pushes `status` and `logs` snapshots every few seconds.
