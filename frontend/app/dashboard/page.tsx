"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type StatusPayload = {
  bot_running_status: string;
  trading_enabled?: boolean;
  balance: number | null;
  current_position: string | null;
  leverage: number | null;
  symbol: string | null;
  trailing_state: Record<string, unknown> | null;
  open_trade_info: Record<string, unknown> | null;
  last_update_timestamp: string | null;
  warnings: string[];
};

const POLL_MS = 60000;

function displayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(2) : "N/A";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

function readPositionNumber(openTradeInfo: Record<string, unknown> | null, key: string): number | null {
  if (!openTradeInfo) {
    return null;
  }
  const value = openTradeInfo[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function readUnrealizedPnl(openTradeInfo: Record<string, unknown> | null): number | null {
  if (!openTradeInfo) {
    return null;
  }
  const candidates = ["unrealized_pnl", "unrealizedPnl", "upl", "net_pnl", "gross_pnl"];
  for (const key of candidates) {
    const value = readPositionNumber(openTradeInfo, key);
    if (value !== null) {
      return value;
    }
  }
  return null;
}

export default function DashboardPage(): JSX.Element {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  async function loadStatus(): Promise<void> {
    setLoading(true);
    try {
      const payload = await fetchApi<StatusPayload>("/api/status");
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load status");
    } finally {
      setLoading(false);
    }
  }

  useEffect((): (() => void) => {
    void (async () => {
      await loadStatus();
    })();
    const intervalId = window.setInterval(() => {
      void loadStatus();
    }, POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  const openTradeInfo = data?.open_trade_info ?? null;
  const sl = readPositionNumber(openTradeInfo, "sl");
  const tp = readPositionNumber(openTradeInfo, "tp");
  const liquidationPrice = readPositionNumber(openTradeInfo, "liq_price");
  const unrealizedPnl = readUnrealizedPnl(openTradeInfo);

  return (
    <section className="panel">
      <h1>Dashboard</h1>
      <p className="muted">Read-only runtime snapshot (auto-refresh every {POLL_MS / 1000}s).</p>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        <button type="button" onClick={() => void loadStatus()}>
          Refresh now
        </button>
        {loading ? <span className="muted">Loading...</span> : null}
      </div>
      {error ? <p>{error}</p> : null}
      {data ? (
        <div className="kv-grid">
          <div className="kv-item">
            <span className="kv-label">Bot status</span>
            <span>{displayValue(data.bot_running_status)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Trading enabled</span>
            <span>{displayValue(data.trading_enabled)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Balance</span>
            <span>{displayValue(data.balance)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Open position</span>
            <span>{displayValue(data.current_position)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Leverage</span>
            <span>{displayValue(data.leverage)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Current symbol</span>
            <span>{displayValue(data.symbol)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">SL</span>
            <span>{displayValue(sl)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">TP</span>
            <span>{displayValue(tp)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Liquidation price</span>
            <span>{displayValue(liquidationPrice)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Trailing active</span>
            <span>{displayValue(data.trailing_state?.trail_active)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Unrealized PnL</span>
            <span>{displayValue(unrealizedPnl)}</span>
          </div>
          <div className="kv-item">
            <span className="kv-label">Last update</span>
            <span>{displayValue(data.last_update_timestamp)}</span>
          </div>
        </div>
      ) : null}

      {data?.warnings.length ? (
        <>
          <h2>Warnings</h2>
          <ul>
            {data.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </>
      ) : null}

      {openTradeInfo ? (
        <>
          <h2>Open trade info</h2>
          <pre>{JSON.stringify(openTradeInfo, null, 2)}</pre>
        </>
      ) : null}
    </section>
  );
}
