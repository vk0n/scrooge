"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type LogsPayload = {
  path: string;
  requested_lines: number;
  returned_lines: number;
  lines: string[];
};

const DEFAULT_LINES = 200;
const POLL_MS = 60000;

export default function LogsPage(): JSX.Element {
  const [data, setData] = useState<LogsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lineCount, setLineCount] = useState<number>(DEFAULT_LINES);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);

  async function loadLogs(lines: number): Promise<void> {
    setLoading(true);
    try {
      const payload = await fetchApi<LogsPayload>(`/api/logs?lines=${encodeURIComponent(String(lines))}`);
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load logs");
    } finally {
      setLoading(false);
    }
  }

  useEffect((): (() => void) => {
    void loadLogs(lineCount);
    if (!autoRefresh) {
      return () => undefined;
    }
    const intervalId = window.setInterval(() => {
      void loadLogs(lineCount);
    }, POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoRefresh, lineCount]);

  return (
    <section className="panel">
      <h1>Logs</h1>
      <p className="muted">Last N lines from trading log.</p>
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <label htmlFor="lineCount">
          Lines:
          <input
            id="lineCount"
            type="number"
            min={1}
            max={5000}
            value={lineCount}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isFinite(value) && value >= 1 && value <= 5000) {
                setLineCount(value);
              }
            }}
            style={{ marginLeft: "0.5rem", width: "6rem" }}
          />
        </label>
        <label htmlFor="autoRefresh">
          <input
            id="autoRefresh"
            type="checkbox"
            checked={autoRefresh}
            onChange={(event) => setAutoRefresh(event.target.checked)}
            style={{ marginRight: "0.5rem" }}
          />
          Auto-refresh ({POLL_MS / 1000}s)
        </label>
        <button type="button" onClick={() => void loadLogs(lineCount)}>
          Refresh now
        </button>
        {loading ? <span className="muted">Loading...</span> : null}
      </div>
      {error ? <p>{error}</p> : null}
      {data ? (
        <>
          <p className="muted">Returned: {data.returned_lines} / {data.requested_lines}</p>
          <pre>{data.lines.join("\n")}</pre>
        </>
      ) : null}
    </section>
  );
}
