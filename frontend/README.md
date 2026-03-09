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
- `/dashboard` reads `GET /api/status`, auto-refreshes every 60s, and has manual refresh
- `/logs` reads `GET /api/logs?lines=N`, supports manual refresh and optional auto-refresh every 60s
- `/config` reads `GET /api/config`, supports JSON/YAML read-only view (default YAML) and auto-refresh every 60s
- `/controls` sends `POST /api/control/{start|stop|restart}`, then polls `GET /api/control/commands/{id}` for execution status (`start` resume, `stop` pause, `restart` reload config)
