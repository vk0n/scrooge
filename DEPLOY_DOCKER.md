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

## 3. Run modes

Control plane only (no bot process):

```bash
docker compose up -d --build
```

Control plane + live bot:

```bash
docker compose --profile live up -d --build
```

One-shot backtest run:

```bash
docker compose --profile backtest run --rm backtest
```

Backtest artifacts are written into:
- `/runtime/backtests/<run_id>/...`
- symlink `/runtime/backtests/latest` points to last run

Open UI:
- `http://<host>:3000`

## 4. Runtime data model (live)

- API edits/reads `config.yaml` via `/runtime/config.yaml`.
- Live runtime writes:
  - `/runtime/state.json` (runtime snapshot only)
  - `/runtime/trade_history.jsonl`
  - `/runtime/balance_history.jsonl`
  - `/runtime/trading_log.txt`
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
