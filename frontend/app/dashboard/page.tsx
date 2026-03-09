"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type StatusPayload = {
  bot_running_status: string;
  balance: number | null;
  current_position: string | null;
  leverage: number | null;
  symbol: string | null;
  trailing_state: Record<string, unknown> | null;
  open_trade_info: Record<string, unknown> | null;
  last_update_timestamp: string | null;
  warnings: string[];
};

export default function DashboardPage(): JSX.Element {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const payload = await fetchApi<StatusPayload>("/api/status");
        setData(payload);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load status");
      }
    })();
  }, []);

  return (
    <section className="panel">
      <h1>Dashboard</h1>
      <p className="muted">Read-only runtime snapshot from control API.</p>
      {error ? <p>{error}</p> : null}
      {!error && !data ? <p>Loading...</p> : null}
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : null}
    </section>
  );
}
