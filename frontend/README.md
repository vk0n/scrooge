# Scrooge Control Frontend

Next.js 14 operator interface for Scrooge's two live systems: Office / Futures and Treasury / Spot. Market Map and Ledger are shared supporting surfaces.

## Run

```bash
cd frontend
npm install
npm run dev
```

For split local development, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`. Compose uses `INTERNAL_API_BASE_URL=http://api:8000`.

## Pages

- **Office** (`/dashboard`) - Futures status, trade controls, performance, history, and My Contract.
- **Treasury** (`/treasury`) - Treasury overview, allocation/timeline, global Bargains Ledger, holdings, custody, asset Policy, Asset Ledger, manual Spot actions, and My Treasury Rules.
- **Market Map** (`/chart`) - Futures candles, engine-recorded indicators, trades, and equity.
- **Ledger** (`/logs`) - role-styled Office and Treasury events with filters and pagination.

`/config` and `/controls` remain compatibility redirects to Office.

## Treasury UX Contract

The overview and per-asset screens are projections of the same API state. The global and per-asset Bargain lists render the same Bargain representation rather than maintaining duplicate frontend models.

The Vault Reserve card scrolls to and expands USDT. Holding rows expose custody, policy, and ledger sections. Open and closed Bargains show PnL in the objective's meaningful unit and percentage. Manual exchange actions always show preview/confirmation and wait for durable intent state.

Daily overview deltas use UTC midnight. `NEXT_PUBLIC_DISPLAY_TIMEZONE` changes only timestamp formatting.

## Auth And Updates

- `/login` stores HTTP Basic credentials in browser local storage.
- `AuthGate` protects operator pages.
- `Step Out` clears saved credentials.
- Office/Ledger prefer WebSocket updates and fall back to polling.
- Treasury refreshes its DB-backed projection periodically and after mutations.

## Notifications

The bell uses `frontend/public/sw.js` and the notifications API to subscribe, test, and unsubscribe Web Push. It requires browser service-worker support and server-side VAPID configuration.

## Build

```bash
npm run build
npm start
```

The interface supports desktop and compact mobile layouts with sticky desktop navigation and bottom mobile navigation.
