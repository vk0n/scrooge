"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type ConfigPayload = {
  config: Record<string, unknown>;
  path: string;
};

export default function ConfigPage(): JSX.Element {
  const [data, setData] = useState<ConfigPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const payload = await fetchApi<ConfigPayload>("/api/config");
        setData(payload);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load config");
      }
    })();
  }, []);

  return (
    <section className="panel">
      <h1>Config</h1>
      <p className="muted">Read-only parsed config.</p>
      {error ? <p>{error}</p> : null}
      {!error && !data ? <p>Loading...</p> : null}
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : null}
    </section>
  );
}
