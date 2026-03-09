# Scrooge Control API (Stage 1)

This is a Stage 1 scaffold for the Scrooge control plane.

## Run

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Notes

- Endpoints return mock responses only.
- No integration with `main.py`, `strategy.py`, or trade execution exists in this stage.
- WebSocket endpoint is available at `ws://localhost:8000/ws`.
