"use client";

import { useEffect, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type StatusPayload = {
  bot_running_status: string;
  trading_enabled?: boolean;
  balance: number | null;
  last_price: number | null;
  last_price_updated_at?: string | null;
  bot_status: UiStatus | null;
  current_position: string | null;
  leverage: number | null;
  symbol: string | null;
  trailing_state: TrailingState | null;
  trade_status: UiStatus | null;
  open_trade_info: OpenTradeInfo | null;
  last_update_timestamp: string | null;
  warnings: string[];
};

type UiStatus = {
  code: string;
  label: string;
};

type OpenTradeInfo = {
  side: "long" | "short";
  size: number;
  entry: number;
  sl: number | null;
  tp: number | null;
  liq_price: number | null;
  trail_active: boolean;
  trail_price: number | null;
  trail_max: number | null;
  trail_min: number | null;
  entry_time: string;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  position_notional: number | null;
  margin_used: number | null;
  roi_pct: number | null;
  distance_to_sl_pct: number | null;
  distance_to_tp_pct: number | null;
  updated_at: string | null;
};

type TrailingState = {
  trail_active: boolean;
  trail_max: number | null;
  trail_min: number | null;
  trail_price: number | null;
  tp: number | null;
  sl: number | null;
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
  params?: EditableParams | null;
};

type EditableParams = {
  sl_mult?: number | null;
  tp_mult?: number | null;
  trail_atr_mult?: number | null;
  rsi_extreme_long?: number | null;
  rsi_extreme_short?: number | null;
  rsi_long_open_threshold?: number | null;
  rsi_long_qty_threshold?: number | null;
  rsi_long_tp_threshold?: number | null;
  rsi_long_close_threshold?: number | null;
  rsi_short_open_threshold?: number | null;
  rsi_short_qty_threshold?: number | null;
  rsi_short_tp_threshold?: number | null;
  rsi_short_close_threshold?: number | null;
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
const PARAM_DEFS: Array<{ key: keyof EditableParams; label: string }> = [
  { key: "sl_mult", label: "Safety Mult" },
  { key: "tp_mult", label: "Treasure Mult" },
  { key: "trail_atr_mult", label: "Tail Guard" },
  { key: "rsi_extreme_long", label: "Long Extreme" },
  { key: "rsi_extreme_short", label: "Short Extreme" },
  { key: "rsi_long_open_threshold", label: "Long Open" },
  { key: "rsi_long_qty_threshold", label: "Long Size" },
  { key: "rsi_long_tp_threshold", label: "Long Treasure" },
  { key: "rsi_long_close_threshold", label: "Long Exit" },
  { key: "rsi_short_open_threshold", label: "Short Open" },
  { key: "rsi_short_qty_threshold", label: "Short Size" },
  { key: "rsi_short_tp_threshold", label: "Short Treasure" },
  { key: "rsi_short_close_threshold", label: "Short Exit" },
];
const COMMON_QUOTE_ASSETS = ["USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH", "BNB", "EUR", "TRY", "USD"];

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

function formatParamValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(4).replace(/\.?0+$/, "");
}

function statusBadgeClass(status: UiStatus | null): string {
  if (!status) {
    return "badge badge-muted";
  }
  if (status.code === "managing_open_trade") {
    return "badge badge-good";
  }
  if (status.code === "looking_for_buy_opportunity" || status.code === "looking_for_sell_opportunity") {
    return "badge badge-warn";
  }
  if (status.code === "resting") {
    return "badge badge-bad";
  }
  return "badge badge-muted";
}

function tradeSummaryText(status: UiStatus | null): string {
  if (!status) {
    return "No open trade";
  }
  return status.label;
}

function tradeSummaryPhraseClass(status: UiStatus | null): string {
  return "position-summary-phrase";
}

function positionSummaryBadgeClass(
  hasOpenPosition: boolean,
  unrealizedPnlPercent: number | null,
  botStatus: UiStatus | null
): string {
  if (hasOpenPosition) {
    if (unrealizedPnlPercent === null) {
      return "position-upnl-badge position-upnl-badge-neutral";
    }
    if (unrealizedPnlPercent < 0) {
      return "position-upnl-badge position-upnl-badge-negative";
    }
    return "position-upnl-badge position-upnl-badge-positive";
  }

  if (botStatus?.code === "looking_for_buy_opportunity") {
    return "position-upnl-badge position-upnl-badge-positive";
  }
  if (botStatus?.code === "looking_for_sell_opportunity") {
    return "position-upnl-badge position-upnl-badge-negative";
  }
  if (botStatus?.code === "resting") {
    return "position-upnl-badge position-upnl-badge-paused";
  }
  return "position-upnl-badge position-upnl-badge-neutral";
}

function positionSummaryBadgeText(
  hasOpenPosition: boolean,
  unrealizedPnlPercent: number | null,
  botStatus: UiStatus | null
): string | null {
  if (hasOpenPosition) {
    return formatPercent(unrealizedPnlPercent);
  }
  if (botStatus?.code === "looking_for_buy_opportunity" || botStatus?.code === "looking_for_sell_opportunity") {
    return "🔎";
  }
  if (botStatus?.code === "resting") {
    return "😴";
  }
  return null;
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

function formatUnrealizedPnlUsd(value: number | null): string {
  if (value === null) {
    return "N/A";
  }
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  const absolute = Math.abs(value).toFixed(2);
  if (value > 0) {
    return `+$${absolute}`;
  }
  if (value < 0) {
    return `-$${absolute}`;
  }
  return "$0.00";
}

function formatUsd(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "N/A";
  }
  return `$${value.toFixed(2)}`;
}

function resolveBaseAsset(symbol: string | null | undefined): string | null {
  if (!symbol) {
    return null;
  }
  const normalized = symbol.trim().toUpperCase();
  if (!normalized) {
    return null;
  }

  for (const quoteAsset of COMMON_QUOTE_ASSETS) {
    if (normalized.endsWith(quoteAsset) && normalized.length > quoteAsset.length) {
      return normalized.slice(0, -quoteAsset.length);
    }
  }
  return normalized;
}

function formatAssetAmount(value: number | null | undefined, symbol: string | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  const amount = Math.abs(value);
  const formatted = amount.toFixed(6).replace(/\.?0+$/, "");
  const baseAsset = resolveBaseAsset(symbol);
  return baseAsset ? `${formatted} ${baseAsset}` : formatted;
}

function formatPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value > 0) {
    return `+${value.toFixed(2)}%`;
  }
  if (value < 0) {
    return `${value.toFixed(2)}%`;
  }
  return "0.00%";
}

function tailGuardValueClass(trailingState: TrailingState | null): string {
  if (!trailingState) {
    return "metric-value value-neutral";
  }
  const trailActive = trailingState.trail_active;
  if (!trailActive) {
    return "metric-value value-neutral";
  }
  return "metric-value value-positive";
}

function resolveTailGuardValue(trailingState: TrailingState | null): string {
  if (!trailingState) {
    return "N/A";
  }
  const trailActive = trailingState.trail_active;
  if (!trailActive) {
    return "Off";
  }
  const trailPrice = trailingState.trail_price ?? trailingState.tp;
  if (trailPrice === null) {
    return "Active";
  }
  return trailPrice.toFixed(2);
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
  const qty = form.use_full_balance
    ? null
    : parseNumberField(form.qty, "Qty", {
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

function DashboardContent(): JSX.Element {
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
  const [positionExpanded, setPositionExpanded] = useState<boolean>(false);
  const [configParams, setConfigParams] = useState<EditableParams | null>(null);

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
      setConfigParams(payload.editable.params ?? null);
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Failed to load contract");
    } finally {
      setConfigLoading(false);
    }
  }

  async function saveEditableConfig(saveAndRestart: boolean): Promise<void> {
    if (!configForm) {
      return;
    }

    if (saveAndRestart) {
      const confirmed = window.confirm("Update the contract and reopen the office?");
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
      setConfigParams(result.editable.params ?? null);
      if (result.updated) {
        setConfigInfo(`Contract updated. Changed: ${result.changed_fields.join(", ")}`);
      } else {
        setConfigInfo("No contract changes detected.");
      }

      if (saveAndRestart) {
        await runControlAction("restart");
      }
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Failed to update contract");
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
  const sl = openTradeInfo?.sl ?? null;
  const tp = openTradeInfo?.tp ?? null;
  const liquidationPrice = openTradeInfo?.liq_price ?? null;
  const unrealizedPnl = openTradeInfo?.unrealized_pnl ?? null;
  const unrealizedPnlPercent = openTradeInfo?.unrealized_pnl_pct ?? null;
  const entryPrice = openTradeInfo?.entry ?? null;
  const positionSize = openTradeInfo?.size ?? null;
  const marginUsed = openTradeInfo?.margin_used ?? null;
  const lastPrice = data?.last_price ?? null;
  const openTime = formatDateTimeEu(openTradeInfo?.entry_time ?? null);
  const hasOpenPosition = openTradeInfo !== null;
  const botStatus = data?.bot_status ?? null;
  const tradeStatus = data?.trade_status ?? null;
  const traitRows = PARAM_DEFS.map((def) => ({
    key: def.key,
    label: def.label,
    value: formatParamValue(configParams?.[def.key]),
  }));
  const positionSummaryText = tradeSummaryText(tradeStatus);
  const positionSummaryPhraseClass = tradeSummaryPhraseClass(tradeStatus);
  const positionUpnlBadgeText = positionSummaryBadgeText(hasOpenPosition, unrealizedPnlPercent, botStatus);
  const showPositionUpnlBadge = positionUpnlBadgeText !== null;
  const positionUpnlBadgeClass = positionSummaryBadgeClass(hasOpenPosition, unrealizedPnlPercent, botStatus);

  useEffect(() => {
    if (!hasOpenPosition) {
      setPositionExpanded(false);
    }
  }, [hasOpenPosition]);

  return (
    <section className="panel page-shell office-panel">
      {!wsConnected ? (
        <p className="dialog-scrooge dialog-scrooge-warning">
          Wire is down (WebSocket disconnected). Courier polling every {POLL_MS / 1000}s.
        </p>
      ) : null}
      {loading ? <p className="dialog-scrooge dialog-scrooge-compact">Loading...</p> : null}
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
      {data ? (
        <>
          <p className="dialog-scrooge">My current status:</p>
          <div className="section-block">
            <div className="status-strip">
              <div className="status-chip">
                <span className="status-chip-label">Status</span>
                <span className={statusBadgeClass(botStatus)}>
                  {botStatus?.label ?? "N/A"}
                </span>
              </div>
              <div className="status-chip">
                <span className="status-chip-label">Vault</span>
                <span className="metric-value vault-value">
                  {displayValue(data.balance)}
                  {typeof data.balance === "number" && Number.isFinite(data.balance) ? (
                    <span className="vault-dollar" aria-hidden="true">
                      $
                    </span>
                  ) : null}
                </span>
              </div>
            </div>
            <p className="status-helper">
              Ticker: {displayValue(data.symbol)} • Leverage:{" "}
              {data.leverage === null || data.leverage === undefined ? "N/A" : `x${displayValue(data.leverage)}`} • Last Price:{" "}
              {displayValue(lastPrice)}
            </p>
            <details
              className="position-accordion"
              open={hasOpenPosition && positionExpanded}
              onToggle={(event) => {
                if (!hasOpenPosition) {
                  event.currentTarget.open = false;
                  return;
                }
                setPositionExpanded(event.currentTarget.open);
              }}
            >
              <summary
                className={`position-accordion-summary${!hasOpenPosition ? " position-accordion-summary-disabled" : ""}`}
                onClick={(event) => {
                  if (!hasOpenPosition) {
                    event.preventDefault();
                  }
                }}
              >
                <span className="position-accordion-title">Current Trade</span>
                <span className="position-summary-right">
                  <span className={positionSummaryPhraseClass}>
                    <span>{positionSummaryText}</span>
                  </span>
                  {showPositionUpnlBadge ? (
                    <span className={positionUpnlBadgeClass}>{positionUpnlBadgeText}</span>
                  ) : null}
                </span>
              </summary>
              <div className="position-accordion-body">
                <div className="trade-pulse-grid">
                  <div className="kv-item metric-card">
                    <span className="kv-label">Trade Side</span>
                    <span className={positionSideBadgeClass(openTradeInfo?.side ?? data.current_position)}>
                      {formatPositionSide(openTradeInfo?.side ?? data.current_position)}
                    </span>
                  </div>
                  <div className="kv-item metric-card">
                    <span className="kv-label">Entry Price</span>
                    <span className="metric-value">{displayValue(entryPrice)}</span>
                  </div>
                  <div className="kv-item metric-card">
                    <span className="kv-label">Opened At</span>
                    <span className="metric-value">{openTime}</span>
                  </div>
                  <div className="kv-item metric-card">
                    <span className="kv-label">Floating PnL</span>
                    <span className={unrealizedPnlClass(unrealizedPnl)}>{formatUnrealizedPnlUsd(unrealizedPnl)}</span>
                  </div>
                </div>

                <div className="trade-detail-grid">
                  <div className="kv-item metric-card">
                    <span className="kv-label">Margin Used</span>
                    <span className="metric-value">{formatUsd(marginUsed)}</span>
                  </div>
                  <div className="kv-item metric-card">
                    <span className="kv-label">Position Size</span>
                    <span className="metric-value">{formatAssetAmount(positionSize, data?.symbol)}</span>
                  </div>
                  <div className="kv-item metric-card">
                    <span className="kv-label">Tail Guard</span>
                    <span className={tailGuardValueClass(data.trailing_state)}>
                      {resolveTailGuardValue(data.trailing_state)}
                    </span>
                  </div>
                </div>

                <div className="trade-manage-grid">
                  <div className="kv-item trade-manage-card">
                    <span className="kv-label">Safety Net</span>
                    <span className="metric-value">{displayValue(sl)}</span>
                    <div className="trade-edit-row">
                      <label className="field-stack dialog-user-field trade-edit-field">
                        Set Safety Net
                        <input
                          type="number"
                          step="any"
                          min="0"
                          value={slValue}
                          onChange={(event) => setSlValue(event.target.value)}
                          disabled={busyAction !== null || !hasOpenPosition}
                        />
                      </label>
                      <button
                        type="button"
                        className="dialog-user-btn"
                        onClick={() => void runUpdate("update-sl", slValue)}
                        disabled={busyAction !== null || !hasOpenPosition}
                      >
                        Apply Net
                      </button>
                    </div>
                  </div>

                  <div className="kv-item trade-manage-card">
                    <span className="kv-label">Treasure Mark</span>
                    <span className="metric-value">{displayValue(tp)}</span>
                    <div className="trade-edit-row">
                      <label className="field-stack dialog-user-field trade-edit-field">
                        Set Treasure Mark
                        <input
                          type="number"
                          step="any"
                          min="0"
                          value={tpValue}
                          onChange={(event) => setTpValue(event.target.value)}
                          disabled={busyAction !== null || !hasOpenPosition}
                        />
                      </label>
                      <button
                        type="button"
                        className="dialog-user-btn"
                        onClick={() => void runUpdate("update-tp", tpValue)}
                        disabled={busyAction !== null || !hasOpenPosition}
                      >
                        Mark Treasure
                      </button>
                    </div>
                  </div>
                </div>

                {liquidationPrice !== null ? (
                  <p className="trade-danger-line">Danger line: {displayValue(liquidationPrice)}</p>
                ) : null}

                <div className="position-controls-actions">
                  <button
                    type="button"
                    className={`position-close-btn dialog-user-btn ${closePositionButtonClass(unrealizedPnl)}`}
                    onClick={() =>
                      void runControlAction("close-position", {
                        confirmMessage: "Close this trade by hand?"
                      })
                    }
                    disabled={busyAction !== null || !hasOpenPosition}
                  >
                    Close This Trade
                  </button>
                </div>
              </div>
            </details>

            <details className="position-accordion traits-accordion">
              <summary className="position-accordion-summary">
                <span className="position-accordion-title">Trait Sheet</span>
              </summary>
              <div className="position-accordion-body traits-accordion-body">
                <div className="traits-grid">
                  {traitRows.map((trait) => (
                    <div key={trait.key} className="traits-item">
                      <span className="traits-name">{trait.label}</span>
                      <span className="traits-value">{trait.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </details>
            <div className="toolbar">
              <button
                type="button"
                className="dialog-user-btn"
                onClick={() =>
                  void runControlAction("start", {
                    confirmMessage: "Open the trading floor now?"
                  })
                }
                disabled={busyAction !== null}
              >
                Open for Business
              </button>
              <button
                type="button"
                className="dialog-user-btn"
                onClick={() =>
                  void runControlAction("stop", {
                    confirmMessage: "Close the trading floor for now?"
                  })
                }
                disabled={busyAction !== null}
              >
                Close the Floor
              </button>
              <button
                type="button"
                className="dialog-user-btn"
                onClick={() =>
                  void runControlAction("restart", {
                    confirmMessage: "Rewind the office engine?"
                  })
                }
                disabled={busyAction !== null}
              >
                Wind It Again
              </button>
              {busyAction ? <span className="dialog-scrooge dialog-scrooge-compact">Executing {busyAction}...</span> : null}
            </div>
          </div>
          <p className="dialog-scrooge">My contract:</p>
          <div className="section-block">
            {configLoading || !configForm ? (
              <p className="dialog-scrooge">Reviewing the contract...</p>
            ) : (
              <div className="form-grid config-form-grid">
                <label className="kv-item dialog-user-field config-field">
                  <span className="kv-label">Pair</span>
                  <input
                    type="text"
                    value={configForm.symbol}
                    onChange={(event) =>
                      setConfigForm((prev) => (prev ? { ...prev, symbol: event.target.value } : prev))
                    }
                  />
                </label>
                <label className="kv-item dialog-user-field config-field">
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
                <label className="kv-item dialog-user-field config-field">
                  <span className="kv-label">All-in Vault</span>
                  <span className="field-inline dialog-user-toggle config-toggle">
                    <input
                      type="checkbox"
                      className="dialog-user-check"
                      checked={configForm.use_full_balance}
                      onChange={(event) =>
                        setConfigForm((prev) => (prev ? { ...prev, use_full_balance: event.target.checked } : prev))
                      }
                    />
                    <span>{configForm.use_full_balance ? "On" : "Off"}</span>
                  </span>
                </label>
                {!configForm.use_full_balance ? (
                  <label className="kv-item dialog-user-field config-field">
                    <span className="kv-label">Stake (empty = null)</span>
                    <input
                      type="number"
                      step="0.000001"
                      min={0}
                      value={configForm.qty}
                      onChange={(event) =>
                        setConfigForm((prev) => (prev ? { ...prev, qty: event.target.value } : prev))
                      }
                    />
                  </label>
                ) : null}
              </div>
            )}
            <div className="toolbar config-toolbar">
              <button
                type="button"
                className="dialog-user-btn config-action-btn"
                onClick={() => void saveEditableConfig(false)}
                disabled={configLoading || configSaving || busyAction !== null || !configForm}
              >
                Update Contract
              </button>
              <button
                type="button"
                className="dialog-user-btn config-action-btn"
                onClick={() => void saveEditableConfig(true)}
                disabled={configLoading || configSaving || busyAction !== null || !configForm}
              >
                Update and Reopen Office
              </button>
              <button
                type="button"
                className="dialog-user-btn config-action-btn"
                onClick={() => void loadEditableConfig()}
                disabled={configSaving || busyAction !== null}
              >
                Review Contract
              </button>
              {configSaving ? <span className="dialog-scrooge dialog-scrooge-compact">Updating contract...</span> : null}
            </div>
            {configError ? <p className="dialog-scrooge dialog-scrooge-error">{configError}</p> : null}
            {configInfo ? <p className="dialog-scrooge">{configInfo}</p> : null}
          </div>
        </>
      ) : null}

      {controlError ? <p className="dialog-scrooge dialog-scrooge-error">{controlError}</p> : null}

      {commandStatus ? (
        <div className="kv-item metric-card">
          <span className="kv-label">Last Order</span>
          <span className={commandStatusBadgeClass(commandStatus.status)}>
            {commandStatus.action}: {commandStatus.status}
          </span>
          {commandStatus.message ? <span className="dialog-scrooge">{commandStatus.message}</span> : null}
        </div>
      ) : null}

      {data?.warnings.length ? (
        <>
          <h2>Red Flags</h2>
          <ul className="warning-list">
            {data.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </>
      ) : null}

      {data ? (
        <p className="technical-footnote">Last whisper from the market: {formatDateTimeEu(data.last_update_timestamp)}</p>
      ) : null}
    </section>
  );
}

export default function DashboardPage(): JSX.Element {
  return (
    <AuthGate>
      <DashboardContent />
    </AuthGate>
  );
}
