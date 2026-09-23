"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type LedgerScope = "all" | "trades" | "treasury";
type LedgerTone = "neutral" | "open" | "positive" | "negative";

type LedgerEntry = {
  entry_id: string;
  scope: "trades" | "treasury";
  timestamp: string | null;
  code: string;
  tone: LedgerTone;
  message: string;
};

type LedgerPayload = {
  scope: LedgerScope;
  entries: LedgerEntry[];
  returned_entries: number;
  total_entries: number;
  has_earlier: boolean;
  next_cursor: string | null;
  warnings?: string[];
};

const LEDGER_PAGE_SIZE = 30;
const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;

function classifyLegacyTone(message: string): LedgerTone {
  const normalized = message.trim().toLowerCase();
  if (normalized.includes("i have opened ")) return "open";
  if (normalized.includes("has been liquidated")) return "negative";
  if (normalized.includes("i have closed ") && normalized.includes("result: +")) return "positive";
  if (normalized.includes("i have closed ") && normalized.includes("result: -")) return "negative";
  return "neutral";
}

function entryTone(entry: LedgerEntry): LedgerTone {
  return entry.code === "legacy_log" ? classifyLegacyTone(entry.message) : entry.tone;
}

function splitResult(message: string): { lead: string; result: string | null } {
  const resultMatch = /(.*?)(Result:\s*[^\n]+)$/i.exec(message);
  return {
    lead: resultMatch ? (resultMatch[1] ?? "").trimEnd() : message,
    result: resultMatch ? (resultMatch[2] ?? null) : null,
  };
}

function LogsContent(): JSX.Element {
  const [scope, setScope] = useState<LedgerScope>("all");
  const [data, setData] = useState<LedgerPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [cursorHistory, setCursorHistory] = useState<Array<string | null>>([null]);
  const [pageIndex, setPageIndex] = useState<number>(0);
  const logBoxRef = useRef<HTMLDivElement | null>(null);
  const pendingScrollTargetRef = useRef<"top" | "bottom" | null>(null);
  const currentCursor = cursorHistory[pageIndex] ?? null;

  const loadLedger = useCallback(async (cursor: string | null, silent = false): Promise<void> => {
    if (!silent) setLoading(true);
    const params = new URLSearchParams({ scope, limit: String(LEDGER_PAGE_SIZE) });
    if (cursor) params.set("cursor", cursor);
    try {
      const payload = await fetchApi<LedgerPayload>(`/api/ledger?${params.toString()}`);
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load the Ledger");
    } finally {
      if (!silent) setLoading(false);
    }
  }, [scope]);

  useEffect((): (() => void) => {
    document.body.classList.add("logs-page-active");
    return () => document.body.classList.remove("logs-page-active");
  }, []);

  useEffect(() => {
    void loadLedger(currentCursor, false);
  }, [currentCursor, loadLedger]);

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
      const wsUrl = buildWebSocketUrl("/ws/ledger", {
        username: creds.username,
        password: creds.password,
        scope,
        limit: String(LEDGER_PAGE_SIZE),
      });
      if (!wsUrl) return;
      socket = new WebSocket(wsUrl);
      socket.onopen = () => setWsConnected(true);
      socket.onmessage = (event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as { type?: string; data?: LedgerPayload };
          if (payload.type === "ledger" && payload.data && pageIndex === 0) {
            setData(payload.data);
            setError(null);
            setLoading(false);
          }
        } catch {
          // Keep fallback polling active when a malformed frame arrives.
        }
      };
      socket.onclose = () => {
        setWsConnected(false);
        if (!closedByUser) reconnectTimer = window.setTimeout(connect, WS_RECONNECT_MS);
      };
      socket.onerror = () => setWsConnected(false);
    };
    connect();
    return () => {
      closedByUser = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    };
  }, [scope, pageIndex]);

  useEffect((): (() => void) => {
    if (wsConnected || pageIndex !== 0) return () => undefined;
    const intervalId = window.setInterval(() => void loadLedger(null, true), POLL_MS);
    return () => window.clearInterval(intervalId);
  }, [wsConnected, pageIndex, loadLedger]);

  useEffect(() => {
    const target = pendingScrollTargetRef.current;
    const node = logBoxRef.current;
    if (!target || !node) return;
    pendingScrollTargetRef.current = null;
    requestAnimationFrame(() => {
      node.scrollTop = target === "bottom" ? node.scrollHeight : 0;
    });
  }, [data]);

  const hasNewerEntries = pageIndex > 0;
  const hasOlderEntries = Boolean(data?.has_earlier && data.next_cursor);

  return (
    <section className="panel page-shell logs-page-panel">
      <p className="dialog-scrooge">Live quill: freshest entries arrive at the top.</p>
      <div className="ledger-scope-switch" role="group" aria-label="Ledger entries">
        {(["all", "trades", "treasury"] as LedgerScope[]).map((option) => (
          <button
            key={option}
            type="button"
            className={`ledger-scope-button${scope === option ? " ledger-scope-button-active" : ""}`}
            aria-pressed={scope === option}
            onClick={() => {
              setScope(option);
              setCursorHistory([null]);
              setPageIndex(0);
              setData(null);
            }}
          >
            {option[0].toUpperCase() + option.slice(1)}
          </button>
        ))}
      </div>
      {loading ? <p className="dialog-scrooge dialog-scrooge-compact">Opening the ledger...</p> : null}
      {!wsConnected ? <p className="dialog-scrooge dialog-scrooge-warning">Courier fallback is polling quietly.</p> : null}
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
      {data?.warnings?.length ? (
        <ul className="warning-list">
          {data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
      {data ? (
        <>
          <div ref={logBoxRef} className="log-box log-feed log-box-newest-first">
            {data.entries.length ? data.entries.map((entry) => {
              const tone = entryTone(entry);
              const message = splitResult(entry.message);
              return (
                <article key={entry.entry_id} className={`log-line log-line-${tone}`}>
                  <span className="log-line-meta">
                    {entry.timestamp ? (
                      <span className="log-line-timestamp">[{formatDateTimeEu(entry.timestamp, entry.timestamp)}]</span>
                    ) : null}
                    {scope === "all" ? (
                      <span className={`ledger-source-badge ledger-source-badge-${entry.scope}`}>
                        {entry.scope === "trades" ? "Trade" : "Treasury"}
                      </span>
                    ) : null}
                  </span>
                  <span className="log-line-message">
                    {message.lead}
                    {message.result ? (
                      <>
                        {message.lead ? " " : ""}
                        <span className={`log-line-result log-line-result-${tone}`}>{message.result}</span>
                      </>
                    ) : null}
                  </span>
                </article>
              );
            }) : <p className="ledger-empty-state">No entries in this ledger yet.</p>}
          </div>
          {hasNewerEntries || hasOlderEntries ? (
            <div className="toolbar logs-history-toolbar logs-history-toolbar-with-indicator">
              <button
                type="button"
                className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-later"
                onClick={() => {
                  pendingScrollTargetRef.current = "bottom";
                  setPageIndex((current) => Math.max(0, current - 1));
                }}
                disabled={!hasNewerEntries}
              >
                Later
              </button>
              {hasNewerEntries ? (
                <button
                  type="button"
                  className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-latest"
                  onClick={() => {
                    pendingScrollTargetRef.current = "top";
                    setPageIndex(0);
                  }}
                >
                  Latest
                </button>
              ) : <span className="logs-history-center-spacer" aria-hidden="true" />}
              <span className="logs-history-indicator">
                Showing {pageIndex * LEDGER_PAGE_SIZE + 1}-
                {pageIndex * LEDGER_PAGE_SIZE + data.returned_entries} of {data.total_entries}
              </span>
              <button
                type="button"
                className="dialog-user-btn trade-history-nav-button logs-history-nav-button logs-history-nav-button-earlier"
                onClick={() => {
                  if (!data.next_cursor) return;
                  pendingScrollTargetRef.current = "top";
                  setCursorHistory((current) => [...current.slice(0, pageIndex + 1), data.next_cursor]);
                  setPageIndex((current) => current + 1);
                }}
                disabled={!hasOlderEntries}
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
  return <AuthGate><LogsContent /></AuthGate>;
}
