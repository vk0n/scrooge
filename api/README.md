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

- Read endpoints:
  - `GET /health`
  - `GET /api/status`
  - `GET /api/config`
  - `GET /api/logs?lines=200`
- Control endpoints:
  - `POST /api/control/start`
  - `POST /api/control/stop`
  - `POST /api/control/restart`
  - Auth: either `Authorization: Basic ...` or header `X-Scrooge-Control-Token: ...`
- `config.yaml`, `state.json`, and `trading_log.txt` are read from the project root.
- Systemd service is configurable via `SCROOGE_SYSTEMD_SERVICE` (default `scrooge.service`).
- WebSocket endpoint remains available at `ws://localhost:8000/ws` (mock).
