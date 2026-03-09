"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type ConfigPayload = {
  config: Record<string, unknown>;
  path: string;
};

const POLL_MS = 60000;

function scalarToYaml(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function toYaml(value: unknown, indent = 0): string {
  const pad = "  ".repeat(indent);

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "[]";
    }
    return value
      .map((item) => {
        if (item !== null && typeof item === "object") {
          return `${pad}- ${toYaml(item, indent + 1).trimStart()}`;
        }
        return `${pad}- ${scalarToYaml(item)}`;
      })
      .join("\n");
  }

  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return "{}";
    }
    return entries
      .map(([key, entryValue]) => {
        if (entryValue !== null && typeof entryValue === "object") {
          return `${pad}${key}:\n${toYaml(entryValue, indent + 1)}`;
        }
        return `${pad}${key}: ${scalarToYaml(entryValue)}`;
      })
      .join("\n");
  }

  return `${pad}${scalarToYaml(value)}`;
}

export default function ConfigPage(): JSX.Element {
  const [data, setData] = useState<ConfigPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"json" | "yaml">("yaml");

  async function loadConfig(): Promise<void> {
    setLoading(true);
    try {
      const payload = await fetchApi<ConfigPayload>("/api/config");
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load config");
    } finally {
      setLoading(false);
    }
  }

  useEffect((): (() => void) => {
    void (async () => {
      await loadConfig();
    })();
    const intervalId = window.setInterval(() => {
      void loadConfig();
    }, POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  return (
    <section className="panel">
      <h1>Config</h1>
      <p className="muted">Read-only config (auto-refresh every {POLL_MS / 1000}s).</p>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        <button type="button" onClick={() => setViewMode("yaml")} disabled={viewMode === "yaml"}>
          YAML
        </button>
        <button type="button" onClick={() => setViewMode("json")} disabled={viewMode === "json"}>
          JSON
        </button>
        <button type="button" onClick={() => void loadConfig()}>
          Refresh now
        </button>
        {loading ? <span className="muted">Loading...</span> : null}
      </div>
      {error ? <p>{error}</p> : null}
      {data ? (
        <>
          <pre>{viewMode === "json" ? JSON.stringify(data.config, null, 2) : toYaml(data.config)}</pre>
        </>
      ) : null}
    </section>
  );
}
