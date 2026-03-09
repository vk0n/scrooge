"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type LogsPayload = {
  path: string;
  requested_lines: number;
  returned_lines: number;
  lines: string[];
};

export default function LogsPage(): JSX.Element {
  const [data, setData] = useState<LogsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const payload = await fetchApi<LogsPayload>("/api/logs?lines=200");
        setData(payload);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load logs");
      }
    })();
  }, []);

  return (
    <section className="panel">
      <h1>Logs</h1>
      <p className="muted">Read-only tail of trading log.</p>
      {error ? <p>{error}</p> : null}
      {!error && !data ? <p>Loading...</p> : null}
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : null}
    </section>
  );
}
