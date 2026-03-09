# Scrooge Control Frontend (Stage 2)

This is the Stage 2 frontend for the Scrooge control plane.

## Run

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open `http://localhost:3000`.

Pages:
- `/dashboard` reads `GET /api/status`
- `/config` reads `GET /api/config`
- `/logs` reads `GET /api/logs?lines=200`
