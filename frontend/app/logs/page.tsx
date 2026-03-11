"use client";

import { useEffect, useState } from "react";

import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";

type LogsPayload = {
  path?: string;
  requested_lines: number;
  returned_lines: number;
  lines: string[];
  warnings?: string[];
};

const DEFAULT_LINES = 200;
const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;

export default function LogsPage(): JSX.Element {
  const [data, setData] = useState<LogsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lineCount, setLineCount] = useState<number>(DEFAULT_LINES);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [wsConnected, setWsConnected] = useState<boolean>(false);

  async function loadLogs(lines: number, silent = false): Promise<void> {
    if (!silent) {
      setLoading(true);
    }
    try {
      const payload = await fetchApi<LogsPayload>(`/api/logs?lines=${encodeURIComponent(String(lines))}`);
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load logs");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }

  useEffect((): (() => void) => {
    if (!autoRefresh) {
      setWsConnected(false);
      return () => undefined;
    }

    let closedByUser = false;
    let reconnectTimer: number | null = null;
    let socket: WebSocket | null = null;

    const creds = getSavedBasicCredentials();
    if (!creds) {
      setWsConnected(false);
      return () => undefined;
    }

    const connect = (): void => {
      const wsUrl = buildWebSocketUrl("/ws/status", {
        username: creds.username,
        password: creds.password,
        lines: String(lineCount),
      });
      if (!wsUrl) {
        setWsConnected(false);
        return;
      }

      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setWsConnected(true);
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as { type?: string; data?: LogsPayload };
          if (payload.type === "logs" && payload.data) {
            setData(payload.data);
            setError(null);
            setLoading(false);
          }
        } catch {
          // Ignore malformed WS payloads and keep fallback polling active.
        }
      };
      socket.onclose = () => {
        setWsConnected(false);
        if (!closedByUser && autoRefresh) {
          reconnectTimer = window.setTimeout(connect, WS_RECONNECT_MS);
        }
      };
      socket.onerror = () => {
        setWsConnected(false);
      };
    };

    connect();

    return () => {
      closedByUser = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, [autoRefresh, lineCount]);

  useEffect((): (() => void) => {
    void loadLogs(lineCount, false);
    if (!autoRefresh) {
      return () => undefined;
    }
    if (wsConnected) {
      return () => undefined;
    }
    const intervalId = window.setInterval(() => {
      void loadLogs(lineCount, true);
    }, POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoRefresh, lineCount, wsConnected]);

  return (
    <section className="panel page-shell">
      <h1>Logs</h1>
      <p className="muted">
        {autoRefresh
          ? wsConnected
            ? "Live mode: WebSocket updates."
            : `Fallback mode: polling every ${POLL_MS / 1000}s.`
          : "Manual mode: auto updates disabled."}
      </p>
      <div className="toolbar">
        <label htmlFor="lineCount" className="field-stack">
          Lines
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
          />
        </label>
        <label htmlFor="autoRefresh" className="field-inline">
          <input
            id="autoRefresh"
            type="checkbox"
            checked={autoRefresh}
            onChange={(event) => setAutoRefresh(event.target.checked)}
          />
          Auto updates
        </label>
        <button type="button" onClick={() => void loadLogs(lineCount, false)}>
          Refresh now
        </button>
        {loading ? <span className="muted">Loading...</span> : null}
      </div>
      {error ? <p>{error}</p> : null}
      {data ? (
        <>
          <p className="muted">Returned: {data.returned_lines} / {data.requested_lines}</p>
          {data.warnings?.length ? (
            <ul className="warning-list">
              {data.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
          <pre className="log-box">{data.lines.join("\n")}</pre>
        </>
      ) : null}
    </section>
  );
}
