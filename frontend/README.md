# Scrooge Control Frontend

Next.js frontend for the Scrooge control plane.

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
  - includes runtime controls (`start` / `stop` / `restart`)
  - includes config quick editor backed by `GET /api/config/editable` and `POST /api/config/editable`
    - editable fields: `symbol`, `leverage`, `qty`, `use_full_balance`
    - supports `Save` and `Save & Restart`
    - strategy params are intentionally hidden in UI
  - includes open-position controls (`close-position`, `update-sl`, `update-tp`)
  - command execution status is shown inline in dashboard
- `/chart` reads `GET /api/chart` and renders Plotly candlesticks, trade markers, indicators, and equity curve
- `/logs` uses live WebSocket updates (`/ws/status?lines=N`) with polling fallback to `GET /api/logs?lines=N`
- `/config` redirects to `/dashboard` (legacy route kept for compatibility)
- `/controls` redirects to `/dashboard` (legacy route kept for compatibility)

UX notes:
- mobile-friendly layout (compact cards, responsive toolbars/forms)
- sticky top navigation + bottom mobile navigation for primary pages
- touch-friendly controls and readable log/config/chart blocks on phone screens
