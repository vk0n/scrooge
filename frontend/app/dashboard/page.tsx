"use client";

import { useEffect, useState } from "react";

import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";

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

type ControlAction = "start" | "stop" | "restart" | "close_position" | "update_sl" | "update_tp";
type ControlEndpoint = "start" | "stop" | "restart" | "close-position" | "update-sl" | "update-tp";

type ControlResponse = {
  action: ControlAction;
  command_id: string;
  status: string;
  queued_at: string;
};

type CommandStatusResponse = {
  command_id: string;
  action: ControlAction;
  status: string;
  message: string;
  updated_at: string;
  trading_status?: {
    trading_enabled: boolean;
  } | string;
};

type EditableConfig = {
  symbol: string | null;
  leverage: number | null;
  use_full_balance: boolean | null;
  qty: number | null;
};

type EditableConfigResponse = {
  editable: EditableConfig;
};

type SaveEditableResponse = {
  updated: boolean;
  restart_required: boolean;
  changed_fields: string[];
  backup_path: string | null;
  editable: EditableConfig;
};

type ConfigFormState = {
  symbol: string;
  leverage: string;
  use_full_balance: boolean;
  qty: string;
};

const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;
const COMMAND_POLL_MS = 2000;

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

function botStatusBadgeClass(status: string | undefined): string {
  if (status === "running") {
    return "badge badge-good";
  }
  if (status === "paused" || status === "stopped") {
    return "badge badge-bad";
  }
  return "badge badge-muted";
}

function tradingEnabledBadgeClass(enabled: boolean | undefined): string {
  if (enabled === true) {
    return "badge badge-good";
  }
  if (enabled === false) {
    return "badge badge-bad";
  }
  return "badge badge-muted";
}

function positionSideBadgeClass(side: unknown): string {
  const normalized = typeof side === "string" ? side.trim().toLowerCase() : "";
  if (normalized === "long") {
    return "badge badge-good";
  }
  if (normalized === "short") {
    return "badge badge-bad";
  }
  return "badge badge-muted";
}

function formatPositionSide(side: unknown): string {
  if (side === null || side === undefined) {
    return "N/A";
  }
  const normalized = String(side).trim().toLowerCase();
  if (!normalized) {
    return "N/A";
  }
  if (normalized === "long") {
    return "LONG";
  }
  if (normalized === "short") {
    return "SHORT";
  }
  return normalized.toUpperCase();
}

function unrealizedPnlClass(value: number | null): string {
  if (value === null) {
    return "metric-value value-neutral";
  }
  if (value > 0) {
    return "metric-value value-positive";
  }
  if (value < 0) {
    return "metric-value value-negative";
  }
  return "metric-value value-neutral";
}

function formatUnrealizedPnl(value: number | null): string {
  if (value === null) {
    return "N/A";
  }
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  if (value > 0) {
    return `+${value.toFixed(2)}`;
  }
  return value.toFixed(2);
}

function commandStatusBadgeClass(status: string | undefined): string {
  if (status === "completed") {
    return "badge badge-good";
  }
  if (status === "failed") {
    return "badge badge-bad";
  }
  return "badge badge-muted";
}

function closePositionButtonClass(unrealizedPnl: number | null): string {
  if (unrealizedPnl === null) {
    return "button-neutral";
  }
  if (unrealizedPnl > 0) {
    return "button-success";
  }
  if (unrealizedPnl < 0) {
    return "button-danger";
  }
  return "button-neutral";
}

function numberToInput(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function editableToForm(editable: EditableConfig): ConfigFormState {
  return {
    symbol: editable.symbol ?? "",
    leverage: numberToInput(editable.leverage),
    use_full_balance: Boolean(editable.use_full_balance),
    qty: numberToInput(editable.qty)
  };
}

function parseNumberField(
  raw: string,
  fieldName: string,
  options: { min?: number; max?: number; integer?: boolean; allowEmpty?: boolean; exclusiveMin?: boolean }
): number | null {
  const value = raw.trim();
  if (!value) {
    if (options.allowEmpty) {
      return null;
    }
    throw new Error(`${fieldName} is required`);
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${fieldName} must be a number`);
  }
  if (options.integer && !Number.isInteger(parsed)) {
    throw new Error(`${fieldName} must be an integer`);
  }
  if (options.min !== undefined) {
    if (options.exclusiveMin && parsed <= options.min) {
      throw new Error(`${fieldName} must be > ${options.min}`);
    }
    if (!options.exclusiveMin && parsed < options.min) {
      throw new Error(`${fieldName} must be >= ${options.min}`);
    }
  }
  if (options.max !== undefined && parsed > options.max) {
    throw new Error(`${fieldName} must be <= ${options.max}`);
  }

  return parsed;
}

function buildEditablePayload(form: ConfigFormState): EditableConfig {
  const symbol = form.symbol.trim().toUpperCase();
  if (!/^[A-Z0-9]{3,20}$/.test(symbol)) {
    throw new Error("Symbol must match [A-Z0-9]{3,20}");
  }

  const leverage = parseNumberField(form.leverage, "Leverage", {
    min: 1,
    max: 125,
    integer: true
  });
  const qty = parseNumberField(form.qty, "Qty", {
    min: 0,
    exclusiveMin: true,
    allowEmpty: true
  });

  return {
    symbol,
    leverage,
    use_full_balance: form.use_full_balance,
    qty
  };
}

export default function DashboardPage(): JSX.Element {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [busyAction, setBusyAction] = useState<ControlEndpoint | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [commandResult, setCommandResult] = useState<ControlResponse | null>(null);
  const [commandStatus, setCommandStatus] = useState<CommandStatusResponse | null>(null);
  const [slValue, setSlValue] = useState<string>("");
  const [tpValue, setTpValue] = useState<string>("");
  const [configForm, setConfigForm] = useState<ConfigFormState | null>(null);
  const [configLoading, setConfigLoading] = useState<boolean>(true);
  const [configSaving, setConfigSaving] = useState<boolean>(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configInfo, setConfigInfo] = useState<string | null>(null);

  async function loadStatus(silent = false): Promise<void> {
    if (!silent) {
      setLoading(true);
    }
    try {
      const payload = await fetchApi<StatusPayload>("/api/status");
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load status");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }

  async function runControlAction(
    endpoint: ControlEndpoint,
    options?: { confirmMessage?: string; body?: unknown }
  ): Promise<void> {
    const confirmMessage = options?.confirmMessage;
    if (confirmMessage) {
      const confirmed = window.confirm(confirmMessage);
      if (!confirmed) {
        return;
      }
    }

    setBusyAction(endpoint);
    setControlError(null);
    try {
      const response = await fetchApi<ControlResponse>(`/api/control/${endpoint}`, {
        method: "POST",
        body: options?.body
      });
      setCommandResult(response);
      const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${response.command_id}`);
      setCommandStatus(statusPayload);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : `Failed to execute ${endpoint}`);
    } finally {
      setBusyAction(null);
    }
  }

  async function runUpdate(endpoint: "update-sl" | "update-tp", rawValue: string): Promise<void> {
    const numericValue = Number(rawValue);
    if (!Number.isFinite(numericValue) || numericValue <= 0) {
      setControlError("Provide a positive numeric value.");
      return;
    }

    const label = endpoint === "update-sl" ? "SL" : "TP";
    await runControlAction(endpoint, {
      body: { value: numericValue },
      confirmMessage: `Confirm ${label} update to ${numericValue}?`
    });
  }

  async function loadEditableConfig(): Promise<void> {
    setConfigLoading(true);
    setConfigError(null);
    try {
      const payload = await fetchApi<EditableConfigResponse>("/api/config/editable");
      setConfigForm(editableToForm(payload.editable));
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Failed to load editable config");
    } finally {
      setConfigLoading(false);
    }
  }

  async function saveEditableConfig(saveAndRestart: boolean): Promise<void> {
    if (!configForm) {
      return;
    }

    if (saveAndRestart) {
      const confirmed = window.confirm("Save config and restart trading?");
      if (!confirmed) {
        return;
      }
    }

    setConfigSaving(true);
    setConfigError(null);
    setConfigInfo(null);
    try {
      const payload = buildEditablePayload(configForm);
      const result = await fetchApi<SaveEditableResponse>("/api/config/editable", {
        method: "POST",
        body: payload
      });
      setConfigForm(editableToForm(result.editable));
      if (result.updated) {
        setConfigInfo(`Config saved. Changed: ${result.changed_fields.join(", ")}`);
      } else {
        setConfigInfo("No config changes detected");
      }

      if (saveAndRestart) {
        await runControlAction("restart");
      }
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Failed to save config");
    } finally {
      setConfigSaving(false);
    }
  }

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
        password: creds.password
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
          const payload = JSON.parse(event.data) as { type?: string; data?: StatusPayload };
          if (payload.type === "status" && payload.data) {
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

  useEffect((): void => {
    void loadEditableConfig();
  }, []);

  useEffect((): (() => void) => {
    void loadStatus(false);
    if (wsConnected) {
      return () => undefined;
    }
    const intervalId = window.setInterval(() => {
      void loadStatus(true);
    }, POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [wsConnected]);

  useEffect(() => {
    if (!commandResult) {
      return () => undefined;
    }

    if (commandStatus && commandStatus.status !== "pending" && commandStatus.status !== "processing") {
      return () => undefined;
    }

    const intervalId = window.setInterval(() => {
      void (async () => {
        try {
          const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${commandResult.command_id}`);
          setCommandStatus(statusPayload);
        } catch {
          // keep current status until next poll tick
        }
      })();
    }, COMMAND_POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [commandResult, commandStatus]);

  useEffect(() => {
    if (!commandStatus) {
      return;
    }
    if (commandStatus.status === "completed") {
      void loadStatus(true);
    }
  }, [commandStatus]);

  const openTradeInfo = data?.open_trade_info ?? null;
  const sl = readPositionNumber(openTradeInfo, "sl");
  const tp = readPositionNumber(openTradeInfo, "tp");
  const liquidationPrice = readPositionNumber(openTradeInfo, "liq_price");
  const unrealizedPnl = readUnrealizedPnl(openTradeInfo);
  const hasOpenPosition = openTradeInfo !== null;

  return (
    <section className="panel page-shell">
      <h1>Dashboard</h1>
      <p className="muted">
        {wsConnected
          ? "Live mode: WebSocket updates."
          : `Fallback mode: polling every ${POLL_MS / 1000}s.`}
      </p>
      <div className="toolbar">
        <button
          type="button"
          onClick={() => {
            void loadStatus(false);
            void loadEditableConfig();
          }}
          disabled={configSaving}
        >
          Refresh now
        </button>
        {loading ? <span className="muted">Loading...</span> : null}
      </div>
      {error ? <p>{error}</p> : null}
      {data ? (
        <>
          <h2>Runtime</h2>
          <div className="kv-grid">
            <div className="kv-item metric-card">
              <span className="kv-label">Bot status</span>
              <span className={botStatusBadgeClass(data.bot_running_status)}>{displayValue(data.bot_running_status)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Trading enabled</span>
              <span className={tradingEnabledBadgeClass(data.trading_enabled)}>{displayValue(data.trading_enabled)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Balance</span>
              <span className="metric-value">{displayValue(data.balance)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Current symbol</span>
              <span className="metric-value">{displayValue(data.symbol)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Leverage</span>
              <span className="metric-value">{displayValue(data.leverage)}</span>
            </div>
          </div>
          <div className="toolbar">
            <button type="button" onClick={() => void runControlAction("start")} disabled={busyAction !== null}>
              Start Trading
            </button>
            <button
              type="button"
              onClick={() =>
                void runControlAction("stop", {
                  confirmMessage: "Confirm STOP trading action?"
                })
              }
              disabled={busyAction !== null}
            >
              Stop Trading
            </button>
            <button
              type="button"
              onClick={() =>
                void runControlAction("restart", {
                  confirmMessage: "Confirm RESTART trading action?"
                })
              }
              disabled={busyAction !== null}
            >
              Restart Trading
            </button>
            {busyAction ? <span className="muted">Executing {busyAction}...</span> : null}
          </div>

          <h2>Open Position</h2>
          <div className="kv-grid">
            <div className="kv-item metric-card">
              <span className="kv-label">Position side</span>
              <span className={positionSideBadgeClass(data.current_position)}>
                {formatPositionSide(data.current_position)}
              </span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Unrealized PnL</span>
              <span className={unrealizedPnlClass(unrealizedPnl)}>{formatUnrealizedPnl(unrealizedPnl)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">SL</span>
              <span className="metric-value">{displayValue(sl)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">TP</span>
              <span className="metric-value">{displayValue(tp)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Trailing active</span>
              <span className="metric-value">{displayValue(data.trailing_state?.trail_active)}</span>
            </div>
            <div className="kv-item metric-card">
              <span className="kv-label">Liquidation price</span>
              <span className="metric-value">{displayValue(liquidationPrice)}</span>
            </div>
          </div>
          <div className="position-controls-grid">
            <div className="position-control-item">
              <label className="field-stack">
                New SL
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={slValue}
                  onChange={(event) => setSlValue(event.target.value)}
                  placeholder="e.g. 61234.5"
                  disabled={busyAction !== null || !hasOpenPosition}
                />
              </label>
              <button
                type="button"
                onClick={() => void runUpdate("update-sl", slValue)}
                disabled={busyAction !== null || !hasOpenPosition}
              >
                Update SL
              </button>
            </div>
            <div className="position-control-item">
              <label className="field-stack">
                New TP
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={tpValue}
                  onChange={(event) => setTpValue(event.target.value)}
                  placeholder="e.g. 64500"
                  disabled={busyAction !== null || !hasOpenPosition}
                />
              </label>
              <button
                type="button"
                onClick={() => void runUpdate("update-tp", tpValue)}
                disabled={busyAction !== null || !hasOpenPosition}
              >
                Update TP
              </button>
            </div>
          </div>
          <div className="position-controls-actions">
            <button
              type="button"
              className={`position-close-btn ${closePositionButtonClass(unrealizedPnl)}`}
              onClick={() =>
                void runControlAction("close-position", {
                  confirmMessage: "Confirm manual CLOSE of current position?"
                })
              }
              disabled={busyAction !== null || !hasOpenPosition}
            >
              Close Position
            </button>
          </div>
          {!hasOpenPosition ? <p className="muted">No active position. Position actions are disabled.</p> : null}
          <p className="muted">
            Last state update: {displayValue(data.last_update_timestamp)}
          </p>

          <h2>Config</h2>
          <p className="muted">Quick editor for core runtime fields. Strategy params are hidden for now.</p>
          <div className="toolbar">
            <button
              type="button"
              onClick={() => void loadEditableConfig()}
              disabled={configSaving || busyAction !== null}
            >
              Reload from file
            </button>
            <button
              type="button"
              onClick={() => void saveEditableConfig(false)}
              disabled={configLoading || configSaving || busyAction !== null || !configForm}
            >
              Save
            </button>
            <button
              type="button"
              onClick={() => void saveEditableConfig(true)}
              disabled={configLoading || configSaving || busyAction !== null || !configForm}
            >
              Save & Restart
            </button>
            {configSaving ? <span className="muted">Saving...</span> : null}
          </div>
          {configError ? <p>{configError}</p> : null}
          {configInfo ? <p>{configInfo}</p> : null}
          {configLoading || !configForm ? (
            <p className="muted">Loading editable configuration...</p>
          ) : (
            <div className="form-grid">
              <label className="kv-item">
                <span className="kv-label">Symbol</span>
                <input
                  type="text"
                  value={configForm.symbol}
                  onChange={(event) =>
                    setConfigForm((prev) => (prev ? { ...prev, symbol: event.target.value } : prev))
                  }
                />
              </label>
              <label className="kv-item">
                <span className="kv-label">Leverage</span>
                <input
                  type="number"
                  min={1}
                  max={125}
                  value={configForm.leverage}
                  onChange={(event) =>
                    setConfigForm((prev) => (prev ? { ...prev, leverage: event.target.value } : prev))
                  }
                />
              </label>
              <label className="kv-item">
                <span className="kv-label">Qty (empty = null)</span>
                <input
                  type="number"
                  step="0.000001"
                  min={0}
                  value={configForm.qty}
                  onChange={(event) => setConfigForm((prev) => (prev ? { ...prev, qty: event.target.value } : prev))}
                />
              </label>
              <label className="kv-item">
                <span className="kv-label">Use full balance</span>
                <span className="field-inline">
                  <input
                    type="checkbox"
                    checked={configForm.use_full_balance}
                    onChange={(event) =>
                      setConfigForm((prev) => (prev ? { ...prev, use_full_balance: event.target.checked } : prev))
                    }
                  />
                  <span>{configForm.use_full_balance ? "Enabled" : "Disabled"}</span>
                </span>
              </label>
            </div>
          )}
        </>
      ) : null}

      {controlError ? <p>{controlError}</p> : null}

      {commandStatus ? (
        <div className="kv-item metric-card">
          <span className="kv-label">Last command</span>
          <span className={commandStatusBadgeClass(commandStatus.status)}>
            {commandStatus.action}: {commandStatus.status}
          </span>
          {commandStatus.message ? <span className="muted">{commandStatus.message}</span> : null}
        </div>
      ) : null}

      {data?.warnings.length ? (
        <>
          <h2>Warnings</h2>
          <ul className="warning-list">
            {data.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </>
      ) : null}

      {openTradeInfo ? (
        <>
          <h2>Open trade info</h2>
          <pre className="json-box">{JSON.stringify(openTradeInfo, null, 2)}</pre>
        </>
      ) : null}
    </section>
  );
}
