# Scrooge Control Frontend

Next.js control plane UI for Scrooge.

The frontend is the operator surface for:
- login and auth persistence
- dashboard status and control actions
- chart/replay inspection
- live log viewing
- push notification setup

## Run

```bash
cd frontend
npm install
npm run dev
```

Optional for split local dev:

```bash
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

In Docker Compose the frontend uses:

```bash
INTERNAL_API_BASE_URL=http://api:8000
```

Open:

```text
http://localhost:3000
```

## Auth

- `/login` stores HTTP Basic credentials in browser local storage
- authenticated pages are wrapped by `AuthGate`
- top navigation shows `Step Out` to clear saved credentials

## Main Pages

Primary navigation:
- `Office` → `/dashboard`
- `Market Map` → `/chart`
- `Ledger` → `/logs`

Legacy compatibility routes:
- `/config` redirects to `/dashboard`
- `/controls` redirects to `/dashboard`

### Office / Dashboard

`/dashboard` shows:
- runtime status
- balance / leverage / symbol / last price
- current trade summary
- trailing state
- config editing
- live control actions
- inline command status feedback
- contract text built from the editable config

It prefers websocket updates from `/ws/status` and falls back to polling `GET /api/status`.

### Market Map

`/chart` renders Plotly charts using `GET /api/chart`.

Features:
- candlesticks
- trade entry/exit markers
- equity curve
- RSI
- shared x-axis sync
- fullscreen chart mode
- mobile-friendly controls

### Ledger

`/logs` shows live runtime logs using:
- websocket updates when available
- polling fallback to `GET /api/logs`

Features:
- adjustable row count
- auto-tail toggle
- newest-first toggle

## Push Notifications

The bell control in navigation and panel mode:
- checks browser support
- requests permission
- subscribes through `/api/notifications/subscribe`
- sends a test push through `/api/notifications/test`
- unregisters through `/api/notifications/unsubscribe`

It depends on:
- browser service workers
- `frontend/public/sw.js`
- server-side VAPID configuration in the API/runtime environment

## UX Notes

The current UI is optimized for:
- desktop control-plane usage
- compact mobile inspection
- sticky top navigation
- bottom mobile navigation
- Scrooge-style copy and compact dashboard cards

Realtime status UX is intentionally conservative:
- websocket mode is preferred
- fallback banners only appear on actual fallback, not on every initial load
