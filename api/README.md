# Scrooge Control API (Stage 4)

This is a Stage 4 backend for the Scrooge control plane.

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

- Read endpoints:
  - `GET /health`
  - `GET /api/status`
  - `GET /api/config`
  - `GET /api/logs?lines=200`
- Control endpoints (token required):
  - `POST /api/control/start`
  - `POST /api/control/stop`
  - `POST /api/control/restart`
- `config.yaml`, `state.json`, and `trading_log.txt` are read from the project root.
- Systemd service is configurable via `SCROOGE_SYSTEMD_SERVICE` (default `scrooge.service`).
- Control auth token must be set via `SCROOGE_CONTROL_TOKEN`.
- WebSocket endpoint remains available at `ws://localhost:8000/ws` (mock).
