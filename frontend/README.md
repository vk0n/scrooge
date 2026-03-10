# Scrooge Control Frontend (Stage 2)

This is the Stage 2 frontend for the Scrooge control plane.

## Run

```bash
cd frontend
npm install
npm run dev
```

Optional for split local run (frontend on `3000`, api on `8000`):
```bash
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```
This is also required for direct WebSocket live updates in split local mode.

For Docker Compose 3+1 run, frontend uses internal rewrite target from:
```bash
INTERNAL_API_BASE_URL=http://api:8000
```

Open `http://localhost:3000`.

Authentication:
- Open `/login`
- Enter `Basic` credentials
- Save credentials (stored in browser localStorage)

Pages:
- `/dashboard` uses live WebSocket updates (`/ws/status`) with polling fallback to `GET /api/status`
- `/chart` reads `GET /api/chart` and renders Plotly candlesticks, trade markers, indicators, and equity curve
- `/logs` uses live WebSocket updates (`/ws/status?lines=N`) with polling fallback to `GET /api/logs?lines=N`
- `/config` uses form-based editor backed by `GET /api/config/editable` and `POST /api/config/editable`
  - supports `Save` and `Save & Restart`
  - no free-form YAML editing
- `/controls` sends `POST /api/control/{start|stop|restart}`, then polls `GET /api/control/commands/{id}` for execution status (`start` resume, `stop` pause, `restart` reload config)
