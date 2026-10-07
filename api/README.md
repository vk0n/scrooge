# Scrooge Control API

FastAPI backend for Scrooge's two live systems: Office / Futures and Treasury / Spot. Market Map and Ledger are shared operator surfaces. The API reads canonical SQLite state, creates validated previews, and sends asynchronous execution/control commands through Redis; it does not submit exchange orders itself.

## Run Locally

```bash
cd api
../scrooge-env/bin/uvicorn main:app --reload --port 8000
```

Default CORS origins are `http://localhost:3000` and `http://127.0.0.1:3000`. Override them with comma-separated `SCROOGE_GUI_CORS_ORIGINS`.

## Authentication

- Every `/api/*` endpoint requires HTTP Basic auth.
- WebSocket endpoints require the same credentials.
- `/api/control/*` also accepts `X-Scrooge-Control-Token` for machine clients.

Configure `SCROOGE_GUI_USERNAME`, `SCROOGE_GUI_PASSWORD`, and optionally `SCROOGE_CONTROL_TOKEN`.

## Treasury / Spot Safety

Real Spot execution requires `SCROOGE_SPOT_EXECUTION_ENABLED=1` in both API and bot. The flow is preview, explicit confirmation, Redis delivery, bot-side state/policy/filter revalidation, idempotent Binance submission, confirmed fill, and atomic Treasury settlement.

Relevant settings:

- `SCROOGE_DB_PATH`
- `SCROOGE_CONFIG_PATH`
- `SCROOGE_SPOT_BALANCE_STALE_AFTER_SECONDS`
- `SCROOGE_SPOT_ORDER_PREVIEW_TTL_SECONDS`
- `SCROOGE_SPOT_ORDER_COMMAND_STALE_AFTER_SECONDS`
- `SCROOGE_TREASURY_TRANSFER_ENABLED`

An `uncertain` or `accounting_error` intent must be reconciled by Binance client order ID before a replacement is attempted.

## Endpoint Map

### Shared Runtime And History

- `GET /health`
- `GET /api/status`
- `GET /api/history/trades`
- `GET /api/history/summary`
- `GET /api/logs`
- `GET /api/ledger`
- `GET /api/chart`

### Treasury / Spot

- `GET /api/portfolio` - overview, holdings, reserve, exchange state, and recent entries.
- `POST /api/portfolio/transactions` - accounting deposit, withdrawal, adjustment, or manual entry.
- `POST /api/portfolio/custody-transfers` - move managed quantity between custody locations.
- `POST /api/portfolio/assets/{asset}/policy` - Target, Minimum Holding, and objective.
- `GET /api/portfolio/assets/{asset}/transactions` - accounting entries for one asset.
- `GET /api/portfolio/assets/{asset}/ledger` - unified transactions and Bargains for one asset.
- `GET /api/portfolio/bargains` - the same Bargain objects across the full Treasury.
- `POST /api/portfolio/bargains/{swing_id}/close-preview` - preview a manual Bargain close.
- `POST /api/portfolio/cash-policy` - set future profitable-cash retention percentage.
- `POST /api/portfolio/cash-policy/release` - release Protected Cash.
- `POST /api/portfolio/cash-policy/transfer` - move cash between Protected and Spendable.
- `POST /api/portfolio/spot-orders/preview` - validate a manual Spot request.
- `GET /api/portfolio/spot-orders/{intent_id}` - inspect durable intent state.
- `POST /api/portfolio/spot-orders/{intent_id}/execute` - requires `CONFIRM_SPOT_ORDER`.
- `POST /api/portfolio/office-transfers` - optional confirmed Treasury/Futures USDT transfer.

Manual confirmed Binance BUY/SELL orders update asset quantity, USDT cash, and Target Holding. They are internal trades and do not count as external Invested Capital.

### Configuration

- `GET /api/config`
- `GET/POST /api/config/editable`
- `GET/POST /api/config/raw`
- `GET/POST /api/config/treasury-rules`

Writes create a backup and report `restart_required`. Treasury Rules strictly validate and replace only the `treasury` subtree.

### Office / Futures Control

- `POST /api/control/start`
- `POST /api/control/stop`
- `POST /api/control/restart`
- `POST /api/control/close-position`
- `POST /api/control/suggest-trade`
- `POST /api/control/update-sl`
- `POST /api/control/update-tp`
- `GET /api/control/commands/{command_id}`

Commands are asynchronous and require the live bot to be running.

### Notifications And WebSocket

- `GET/POST /api/notifications/*`
- `WS /ws`
- `WS /ws/status`

WebSocket payloads provide status/log updates; the frontend falls back to polling.

## Runtime Dependencies

The API needs the same mounted SQLite database and YAML config as the bot, plus the same Redis command namespace. Chart behavior is controlled by `SCROOGE_CHART_SOURCE`, `SCROOGE_CHART_DATASET_PATH`, and related limits. Push behavior uses the `SCROOGE_PUSH_*` settings documented in `.env.example`.

See [Runtime Storage](../docs/runtime-storage.md) and [Spot Treasury](../docs/spot-treasury.md) for state ownership and business semantics.
