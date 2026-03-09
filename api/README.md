# Scrooge Control API (Stage 2)

This is a Stage 2 read-only backend for the Scrooge control plane.

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

- Read-only endpoints:
  - `GET /health`
  - `GET /api/status`
  - `GET /api/config`
  - `GET /api/logs?lines=200`
- `config.yaml`, `state.json`, and `trading_log.txt` are read from the project root.
- No write operations are exposed in this stage.
- WebSocket endpoint remains available at `ws://localhost:8000/ws` (mock).
