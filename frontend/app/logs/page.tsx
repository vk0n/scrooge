"use client";

import { useEffect, useRef, useState } from "react";

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

const LOG_WINDOW_LINES = 200;
const LOG_PAGE_SIZE = 30;
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
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [logPage, setLogPage] = useState<number>(0);
  const logBoxRef = useRef<HTMLDivElement | null>(null);
  const pendingScrollTargetRef = useRef<"top" | "bottom" | null>(null);

  async function loadLogs(silent = false): Promise<void> {
    if (!silent) {
      setLoading(true);
    }
    try {
      const payload = await fetchApi<LogsPayload>(`/api/logs?lines=${encodeURIComponent(String(LOG_WINDOW_LINES))}`);
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
        lines: String(LOG_WINDOW_LINES),
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
        if (!closedByUser) {
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
  }, []);

  useEffect((): (() => void) => {
    void loadLogs(false);
    if (wsConnected) {
      return () => undefined;
    }
    const intervalId = window.setInterval(() => {
      void loadLogs(true);
    }, POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [wsConnected]);

  const totalLines = data?.lines.length ?? 0;
  const logPageCount = Math.max(1, Math.ceil(totalLines / LOG_PAGE_SIZE));
  const logPageIndex = Math.min(logPage, logPageCount - 1);

  useEffect(() => {
    setLogPage((currentPage) => Math.min(currentPage, logPageCount - 1));
  }, [logPageCount]);

  const pageEnd = Math.max(0, totalLines - logPageIndex * LOG_PAGE_SIZE);
  const pageStart = Math.max(0, pageEnd - LOG_PAGE_SIZE);
  const displayedLines = data ? data.lines.slice(pageStart, pageEnd).reverse() : [];
  const parsedLines = displayedLines.map(parseLogLine);
  const hasOlderLogs = pageStart > 0;
  const hasNewerLogs = logPageIndex > 0;

  useEffect(() => {
    if (!pendingScrollTargetRef.current) {
      return;
    }
    const node = logBoxRef.current;
    if (!node) {
      return;
    }
    const target = pendingScrollTargetRef.current;
    pendingScrollTargetRef.current = null;
    requestAnimationFrame(() => {
      if (target === "bottom") {
        node.scrollTop = node.scrollHeight;
        return;
      }
      node.scrollTop = 0;
    });
  }, [logPageIndex, displayedLines.length]);

  return (
    <section className="panel page-shell logs-page-panel">
      <p className="dialog-scrooge">
        Live quill: freshest entries arrive at the top.
      </p>
      {loading ? <p className="dialog-scrooge dialog-scrooge-compact">Opening the ledger...</p> : null}
      {!wsConnected ? <p className="dialog-scrooge dialog-scrooge-warning">Courier fallback is polling quietly.</p> : null}
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
      {data ? (
        <>
          {data.warnings?.length ? (
            <ul className="warning-list">
              {data.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
          <div ref={logBoxRef} className="log-box log-feed log-box-newest-first">
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
          {totalLines > LOG_PAGE_SIZE ? (
            <div className="toolbar logs-history-toolbar">
              <button
                type="button"
                className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-later"
                onClick={() => {
                  pendingScrollTargetRef.current = "bottom";
                  setLogPage((currentPage) => Math.max(0, currentPage - 1));
                }}
                disabled={!hasNewerLogs}
              >
                Later
              </button>
              {hasNewerLogs ? (
                <button
                  type="button"
                  className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-latest"
                  onClick={() => {
                    pendingScrollTargetRef.current = "top";
                    setLogPage(0);
                  }}
                >
                  Latest
                </button>
              ) : (
                <span className="logs-history-center-spacer" aria-hidden="true" />
              )}
              <button
                type="button"
                className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-earlier"
                onClick={() => {
                  pendingScrollTargetRef.current = "top";
                  setLogPage((currentPage) => Math.min(logPageCount - 1, currentPage + 1));
                }}
                disabled={!hasOlderLogs}
              >
                Earlier
              </button>
            </div>
          ) : null}
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
