"use client";

import { useEffect, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";

type LogsPayload = {
  path?: string;
  requested_lines: number;
  returned_lines: number;
  lines: string[];
  warnings?: string[];
};

type ParsedLogLine = {
  raw: string;
  timestamp: string | null;
  message: string;
  tone: "neutral" | "open" | "positive" | "negative";
  resultText: string | null;
  messageLead: string;
};

const DEFAULT_LINES = 200;
const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;

function classifyLogTone(message: string): ParsedLogLine["tone"] {
  const normalized = message.trim().toLowerCase();
  if (!normalized) {
    return "neutral";
  }
  if (normalized.includes("i have opened ")) {
    return "open";
  }
  if (normalized.includes("has been liquidated")) {
    return "negative";
  }
  if (normalized.includes("i have closed ")) {
    if (normalized.includes("result: +")) {
      return "positive";
    }
    if (normalized.includes("result: -")) {
      return "negative";
    }
  }
  return "neutral";
}

function parseLogLine(line: string): ParsedLogLine {
  const match = /^\[([^\]]+)\]\s*(.*)$/.exec(line);
  const baseMessage = match ? (match[2] ?? "") : line;
  const resultMatch = /(.*?)(Result:\s*[^\n]+)$/i.exec(baseMessage);
  const messageLead = resultMatch ? (resultMatch[1] ?? "").trimEnd() : baseMessage;
  const resultText = resultMatch ? (resultMatch[2] ?? null) : null;
  if (!match) {
    return {
      raw: line,
      timestamp: null,
      message: baseMessage,
      tone: classifyLogTone(baseMessage),
      resultText,
      messageLead,
    };
  }

  const message = baseMessage;
  return {
    raw: line,
    timestamp: match[1] ?? null,
    message,
    tone: classifyLogTone(message),
    resultText,
    messageLead,
  };
}

function LogsContent(): JSX.Element {
  const [data, setData] = useState<LogsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lineCount, setLineCount] = useState<number>(DEFAULT_LINES);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [newestFirst, setNewestFirst] = useState<boolean>(true);
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
    document.body.classList.add("logs-page-active");
    return () => {
      document.body.classList.remove("logs-page-active");
    };
  }, []);

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

  const displayedLines = data ? (newestFirst ? [...data.lines].reverse() : data.lines) : [];
  const parsedLines = displayedLines.map(parseLogLine);

  return (
    <section className="panel page-shell logs-page-panel">
      <p className="dialog-scrooge">
        {autoRefresh
          ? wsConnected
            ? newestFirst
              ? "Live quill: freshest entries arrive at the top."
              : "Live quill: entries stream in real time."
            : `Courier mode: polling every ${POLL_MS / 1000}s.`
          : "Manual reading mode: auto updates disabled."}
      </p>
      <div className="toolbar logs-toolbar">
        <div className="logs-toolbar-group">
          <label htmlFor="lineCount" className="line-count-inline dialog-user-field">
            <span>Rows</span>
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
          <div className="logs-toolbar-toggles">
            <label htmlFor="autoRefresh" className="field-inline dialog-user-toggle logs-toolbar-toggle">
              <input
                id="autoRefresh"
                type="checkbox"
                className="dialog-user-check"
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
              />
              Auto Tail
            </label>
            <label htmlFor="newestFirst" className="field-inline dialog-user-toggle logs-toolbar-toggle">
              <input
                id="newestFirst"
                type="checkbox"
                className="dialog-user-check"
                checked={newestFirst}
                onChange={(event) => setNewestFirst(event.target.checked)}
              />
              Newest First
            </label>
          </div>
        </div>
        <div className="logs-toolbar-rail">
          {loading ? <span className="dialog-scrooge dialog-scrooge-compact">Loading...</span> : null}
          <button type="button" className="dialog-user-btn logs-toolbar-btn" onClick={() => void loadLogs(lineCount, false)}>
            Read Again
          </button>
        </div>
      </div>
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
      {data ? (
        <>
          <p className="dialog-scrooge">
            Showing {newestFirst ? "newest" : "oldest"} visible order: {data.returned_lines} / {data.requested_lines}
          </p>
          {data.warnings?.length ? (
            <ul className="warning-list">
              {data.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
          <div className={`log-box log-feed${newestFirst ? " log-box-newest-first" : ""}`}>
            {parsedLines.map((line, index) => (
              <article key={`${line.raw}-${index}`} className={`log-line log-line-${line.tone}`}>
                {line.timestamp ? <span className="log-line-timestamp">[{line.timestamp}]</span> : null}
                <span className="log-line-message">
                  {line.messageLead}
                  {line.resultText ? (
                    <>
                      {line.messageLead ? " " : ""}
                      <span className={`log-line-result log-line-result-${line.tone}`}>{line.resultText}</span>
                    </>
                  ) : null}
                </span>
              </article>
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}

export default function LogsPage(): JSX.Element {
  return (
    <AuthGate>
      <LogsContent />
    </AuthGate>
  );
}
