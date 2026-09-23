"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type PortfolioSummary = {
  total_value: number;
  invested_capital: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number | null;
  dry_powder: number;
  dry_powder_pct: number | null;
  largest_position: PortfolioHolding | null;
  holding_count: number;
  prices_updated_at: string | null;
  binance_spot_usdt_free: number | null;
  binance_spot_usdt_locked: number | null;
};

type PortfolioExchange = {
  venue: string;
  account_type: string;
  status: "ok" | "error" | "unavailable";
  captured_at: string | null;
  last_attempt_at: string | null;
  age_seconds: number | null;
  is_stale: boolean;
  is_balance_verified: boolean;
  can_trade: boolean | null;
  error: string | null;
  balances: Array<{
    asset_symbol: string;
    free: number;
    locked: number;
    total: number;
  }>;
  usdt_free: number | null;
  usdt_locked: number | null;
  spot_execution_enabled: boolean;
};

type SpotOrderIntent = {
  intent_id: string;
  client_order_id: string;
  symbol: string;
  asset_symbol: string;
  quote_symbol: string;
  side: "buy" | "sell";
  requested_quantity: number;
  estimated_price: number;
  estimated_quote_value: number;
  available_quote_quantity: number | null;
  available_asset_quantity: number | null;
  protected_floor_quantity: number | null;
  policy_sellable_quantity: number | null;
  projected_holding_quantity: number | null;
  status: string;
  preview_expires_at_ms?: number;
};

type SpotOrderQueueResponse = {
  command_id: string;
  status: string;
  intent: SpotOrderIntent;
  idempotent_replay: boolean;
};

type ControlCommandStatus = {
  command_id: string;
  action: string;
  status: "pending" | "processing" | "completed" | "failed";
  message?: string;
  result?: {
    order_id?: string;
    executed_quantity?: number;
    executed_quote_quantity?: number;
    average_price?: number;
  };
};

type PortfolioHolding = {
  asset_symbol: string;
  quote_symbol: string;
  quantity: number;
  average_cost: number | null;
  invested_capital: number;
  market_price: number | null;
  market_price_updated_at: string | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  allocation_pct: number | null;
  is_dry_powder: boolean;
  custody: Record<CustodyLocation, { quantity: number; cost_basis: number }>;
  binance_quantity: number;
  cold_storage_quantity: number;
  unassigned_quantity: number;
  target_quantity: number | null;
  minimum_holding_pct: number | null;
  protected_floor_quantity: number;
  protected_holding_quantity: number;
  amount_above_protected_floor: number;
  amount_below_protected_floor: number;
  policy_sellable_quantity: number;
  immediately_sellable_quantity: number;
  sellable_inventory_is_exchange_verified: boolean;
  exchange_binance_free_quantity: number;
  exchange_binance_locked_quantity: number;
  exchange_binance_total_quantity: number;
  binance_custody_variance: number;
  target_delta_quantity: number | null;
  target_delta_pct: number | null;
};

type PortfolioTransaction = {
  id: number;
  transaction_id: string;
  executed_at: string;
  tx_type: PortfolioTransactionType;
  asset_symbol: string;
  quote_symbol: string;
  quantity: number;
  price: number | null;
  fee_amount: number | null;
  fee_asset: string | null;
  source: string;
  status: string;
  note: string | null;
  custody_location: CustodyLocation;
  source_custody: CustodyLocation | null;
  destination_custody: CustodyLocation | null;
};

type PortfolioTimelinePoint = {
  snapshot_date: string;
  captured_at_ms: number;
  total_value: number;
  invested_capital: number;
  unrealized_pnl: number;
  dry_powder: number;
};

type PortfolioPayload = {
  path: string;
  summary: PortfolioSummary;
  exchange: PortfolioExchange;
  holdings: PortfolioHolding[];
  timeline: PortfolioTimelinePoint[];
  transactions: PortfolioTransaction[];
  transaction_count: number;
  transaction_limit: number;
  transaction_offset: number;
  warnings: string[];
};

type CreatePortfolioTransactionResponse = {
  transaction: PortfolioTransaction;
  portfolio: Omit<PortfolioPayload, "warnings">;
  warnings: string[];
};

type AssetTransactionPayload = {
  asset_symbol: string;
  quote_symbol: string;
  transactions: PortfolioTransaction[];
  transaction_count: number;
  transaction_limit: number;
  transaction_offset: number;
};

type UpdatePortfolioPolicyResponse = {
  policy: {
    asset_symbol: string;
    quote_symbol: string;
    target_quantity: number;
    minimum_holding_pct: number;
  };
  portfolio: Omit<PortfolioPayload, "warnings">;
  warnings: string[];
};

type PortfolioTransactionType = "buy" | "sell" | "deposit" | "withdraw" | "adjustment" | "custody_transfer";
type CustodyLocation = "unassigned" | "binance" | "cold_storage";

type TransactionFormState = {
  tx_type: PortfolioTransactionType;
  asset_symbol: string;
  quantity: string;
  price: string;
  quote_symbol: string;
  fee_amount: string;
  fee_asset: string;
  executed_at: string;
  note: string;
  custody_location: CustodyLocation;
};

const EMPTY_FORM: TransactionFormState = {
  tx_type: "buy",
  asset_symbol: "",
  quantity: "",
  price: "",
  quote_symbol: "USDT",
  fee_amount: "",
  fee_asset: "USDT",
  executed_at: "",
  note: "",
  custody_location: "unassigned",
};

const CUSTODY_LABELS: Record<CustodyLocation, string> = {
  unassigned: "Unassigned",
  binance: "Binance",
  cold_storage: "Cold Storage",
};

const CUSTODY_LOCATIONS = Object.keys(CUSTODY_LABELS) as CustodyLocation[];

const ALLOCATION_COLORS = [
  "#d9ae45",
  "#39c997",
  "#5b9bd5",
  "#e17c58",
  "#a786d8",
  "#65b8c6",
  "#d36984",
  "#8eaa62",
  "#cc8e45",
  "#7888bd",
];

function asNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatNumber(value: number | null | undefined, maximumFractionDigits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(value);
}

function formatCurrency(value: number | null | undefined, maximumFractionDigits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  return `$${formatNumber(value, maximumFractionDigits)}`;
}

function formatSignedCurrency(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}$${formatNumber(Math.abs(value), 2)}`;
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Pending";
  }
  return `${formatNumber(value, 2)}%`;
}

function signedToneClass(value: number | null | undefined, baseClass: string): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) {
    return `${baseClass} value-neutral`;
  }
  return `${baseClass} ${value > 0 ? "value-positive" : "value-negative"}`;
}

function transactionLabel(type: PortfolioTransactionType): string {
  if (type === "custody_transfer") {
    return "Custody Move";
  }
  if (type === "buy") {
    return "Buy";
  }
  if (type === "sell") {
    return "Sell";
  }
  if (type === "deposit") {
    return "Deposit";
  }
  if (type === "withdraw") {
    return "Withdraw";
  }
  return "Adjustment";
}

function transactionToneClass(type: PortfolioTransactionType): string {
  if (type === "buy" || type === "deposit") {
    return "treasury-ledger-type treasury-ledger-type-positive";
  }
  if (type === "sell" || type === "withdraw") {
    return "treasury-ledger-type treasury-ledger-type-negative";
  }
  return "treasury-ledger-type";
}

function custodyQuantity(holding: PortfolioHolding, location: CustodyLocation): number {
  return holding.custody?.[location]?.quantity ?? 0;
}

function primaryCustody(holding: PortfolioHolding): CustodyLocation {
  return CUSTODY_LOCATIONS.reduce((largest, location) =>
    custodyQuantity(holding, location) > custodyQuantity(holding, largest) ? location : largest
  , "unassigned" as CustodyLocation);
}

function holdingToneClass(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) {
    return "treasury-holding-card treasury-holding-card-neutral";
  }
  return `treasury-holding-card ${value > 0 ? "treasury-holding-card-positive" : "treasury-holding-card-negative"}`;
}

function mergePortfolioPayload(payload: Omit<PortfolioPayload, "warnings">, warnings: string[]): PortfolioPayload {
  return {
    ...payload,
    warnings,
  };
}

function formatTimelineDate(value: string): string {
  const [year, month, day] = value.split("-");
  return year && month && day ? `${day}.${month}.${year}` : value;
}

function buildTimelineCoordinates(
  timeline: PortfolioTimelinePoint[],
  valueKey: "total_value" | "unrealized_pnl"
): Array<{ x: number; y: number; point: PortfolioTimelinePoint }> {
  const width = 600;
  const height = 150;
  const paddingX = 14;
  const paddingY = 16;
  const values = timeline.map((point) => point[valueKey]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = maximum - minimum || Math.max(Math.abs(maximum) * 0.04, 1);
  return timeline.map((point, index) => ({
    x: timeline.length === 1 ? width / 2 : paddingX + (index / (timeline.length - 1)) * (width - paddingX * 2),
    y: timeline.length === 1
      ? height / 2
      : paddingY + ((maximum - point[valueKey]) / spread) * (height - paddingY * 2),
    point,
  }));
}

function TimelineSeries({
  title,
  timeline,
  valueKey,
  tone,
}: {
  title: string;
  timeline: PortfolioTimelinePoint[];
  valueKey: "total_value" | "unrealized_pnl";
  tone: "gold" | "positive" | "negative" | "neutral";
}): JSX.Element {
  const coordinates = timeline.length ? buildTimelineCoordinates(timeline, valueKey) : [];
  const linePoints = coordinates.map(({ x, y }) => `${x},${y}`).join(" ");
  const areaPath = coordinates.length > 1
    ? `M ${coordinates[0].x} 150 L ${coordinates.map(({ x, y }) => `${x} ${y}`).join(" L ")} L ${coordinates[coordinates.length - 1].x} 150 Z`
    : "";
  const latest = timeline.at(-1)?.[valueKey];
  const values = timeline.map((point) => point[valueKey]);
  const formatValue = (value: number | null | undefined): string =>
    valueKey === "unrealized_pnl" ? formatSignedCurrency(value) : formatCurrency(value);

  return (
    <div className={`treasury-timeline-series treasury-timeline-series-${tone}`}>
      <header>
        <span>{title}</span>
        <strong>{formatValue(latest)}</strong>
      </header>
      <div className="treasury-timeline-chart">
        {coordinates.length ? (
          <svg viewBox="0 0 600 150" role="img" aria-label={`${title} daily timeline`} preserveAspectRatio="none">
            <line x1="0" y1="75" x2="600" y2="75" className="treasury-timeline-gridline" />
            {areaPath ? <path d={areaPath} className="treasury-timeline-area" /> : null}
            {linePoints ? <polyline points={linePoints} className="treasury-timeline-line" /> : null}
            {coordinates.map(({ x, y, point }) => (
              <circle key={point.snapshot_date} cx={x} cy={y} r={coordinates.length === 1 ? 5 : 3}>
                <title>
                  {formatTimelineDate(point.snapshot_date)}: {formatValue(point[valueKey])}
                </title>
              </circle>
            ))}
          </svg>
        ) : (
          <span>Awaiting the first daily mark.</span>
        )}
      </div>
      {values.length ? (
        <footer>
          <span>Low {formatValue(Math.min(...values))}</span>
          <span>High {formatValue(Math.max(...values))}</span>
        </footer>
      ) : null}
    </div>
  );
}

function CustodyPanel({
  holding,
  exchange,
  onTransferred,
  onExecuted,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  onTransferred: (response: CreatePortfolioTransactionResponse) => void;
  onExecuted: () => Promise<void>;
}): JSX.Element {
  const initialSource = primaryCustody(holding);
  const [source, setSource] = useState<CustodyLocation>(initialSource);
  const [destination, setDestination] = useState<CustodyLocation>(
    initialSource === "binance" ? "cold_storage" : "binance"
  );
  const [quantity, setQuantity] = useState<string>("");
  const [note, setNote] = useState<string>("");
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [tradeExpanded, setTradeExpanded] = useState<boolean>(false);
  const available = custodyQuantity(holding, source);
  const tradeAvailable = Boolean(
    !holding.is_dry_powder &&
    exchange?.spot_execution_enabled &&
    holding.binance_quantity > 0.00000001
  );

  function changeSource(nextSource: CustodyLocation): void {
    setSource(nextSource);
    if (destination === nextSource) {
      setDestination(CUSTODY_LOCATIONS.find((location) => location !== nextSource) ?? "unassigned");
    }
    setQuantity("");
  }

  async function submitTransfer(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>("/api/portfolio/custody-transfers", {
        method: "POST",
        body: {
          asset_symbol: holding.asset_symbol,
          quote_symbol: holding.quote_symbol,
          quantity: asNumber(quantity),
          source_custody: source,
          destination_custody: destination,
          note,
        },
      });
      setQuantity("");
      setNote("");
      onTransferred(response);
    } catch (transferError) {
      setError(transferError instanceof Error ? transferError.message : "Could not record the custody move.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="treasury-custody-panel">
      <header className="treasury-asset-panel-head">
        <span>Custody</span>
        <span className="treasury-custody-summary">
          {CUSTODY_LOCATIONS.map((location) => (
            <span key={location}>
              {CUSTODY_LABELS[location]} {formatNumber(custodyQuantity(holding, location), 8)}
            </span>
          ))}
        </span>
      </header>
      <div className="treasury-custody-content">
        <div className="treasury-custody-breakdown">
          {CUSTODY_LOCATIONS.map((location) => (
            <div key={location} className={location === "binance" ? "treasury-custody-binance" : undefined}>
              <span className="treasury-custody-location-head">
                <span>{CUSTODY_LABELS[location]}</span>
                {location === "binance" && tradeAvailable ? (
                  <button
                    type="button"
                    className="treasury-custody-trade-btn"
                    aria-expanded={tradeExpanded}
                    onClick={() => setTradeExpanded((current) => !current)}
                  >
                    {tradeExpanded ? "Close" : "Trade"}
                  </button>
                ) : null}
              </span>
              <strong>{formatNumber(custodyQuantity(holding, location), 8)} {holding.asset_symbol}</strong>
            </div>
          ))}
        </div>
        {tradeExpanded && tradeAvailable ? (
          <SpotOrderPanel
            key={[
              holding.target_quantity,
              holding.minimum_holding_pct,
              holding.protected_floor_quantity,
              holding.immediately_sellable_quantity,
              holding.exchange_binance_free_quantity,
              exchange?.captured_at,
            ].join(":")}
            holding={holding}
            exchange={exchange}
            onExecuted={onExecuted}
          />
        ) : null}
        <form className="treasury-custody-form" onSubmit={(event) => void submitTransfer(event)}>
          <label className="dialog-user-field">
            From
            <select value={source} onChange={(event) => changeSource(event.target.value as CustodyLocation)}>
              {CUSTODY_LOCATIONS.map((location) => (
                <option key={location} value={location} disabled={custodyQuantity(holding, location) <= 0}>
                  {CUSTODY_LABELS[location]}
                </option>
              ))}
            </select>
          </label>
          <label className="dialog-user-field">
            To
            <select value={destination} onChange={(event) => setDestination(event.target.value as CustodyLocation)}>
              {CUSTODY_LOCATIONS.filter((location) => location !== source).map((location) => (
                <option key={location} value={location}>{CUSTODY_LABELS[location]}</option>
              ))}
            </select>
          </label>
          <label className="dialog-user-field">
            Stack
            <input
              type="number"
              value={quantity}
              min="0"
              max={available}
              step="any"
              placeholder={formatNumber(available, 8)}
              onChange={(event) => setQuantity(event.target.value)}
              required
            />
          </label>
          <label className="dialog-user-field treasury-custody-note">
            Note
            <input
              type="text"
              value={note}
              maxLength={500}
              placeholder="Optional custody note"
              onChange={(event) => setNote(event.target.value)}
            />
          </label>
          <button type="submit" className="dialog-user-btn" disabled={saving || available <= 0}>
            {saving ? "Recording..." : "Record Move"}
          </button>
        </form>
        <p className="treasury-custody-disclaimer">
          Accounting only. No exchange or blockchain transfer is initiated.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div>
    </section>
  );
}

async function waitForSpotOrder(commandId: string): Promise<ControlCommandStatus> {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    const command = await fetchApi<ControlCommandStatus>(`/api/control/commands/${encodeURIComponent(commandId)}`);
    if (command.status === "completed" || command.status === "failed") {
      return command;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
  throw new Error("Spot order is still awaiting confirmation. Check the Treasury Ledger before retrying.");
}

function SpotOrderPanel({
  holding,
  exchange,
  onExecuted,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  onExecuted: () => Promise<void>;
}): JSX.Element {
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = useState<string>("");
  const [preview, setPreview] = useState<SpotOrderIntent | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const executionReady = Boolean(exchange?.spot_execution_enabled && exchange?.is_balance_verified && exchange?.can_trade);

  function resetPreview(nextSide?: "buy" | "sell"): void {
    if (nextSide) {
      setSide(nextSide);
    }
    setPreview(null);
    setStage(null);
    setError(null);
  }

  async function requestPreview(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setStage("Checking policy and Binance inventory...");
    try {
      const result = await fetchApi<SpotOrderIntent>("/api/portfolio/spot-orders/preview", {
        method: "POST",
        body: {
          asset_symbol: holding.asset_symbol,
          quote_symbol: holding.quote_symbol,
          side,
          quantity: asNumber(quantity),
        },
      });
      setPreview(result);
      setStage("Preview ready. No order has been sent.");
    } catch (previewError) {
      setPreview(null);
      setStage(null);
      setError(previewError instanceof Error ? previewError.message : "Could not prepare the Spot order preview.");
    } finally {
      setBusy(false);
    }
  }

  async function executeOrder(): Promise<void> {
    if (!preview) {
      return;
    }
    const confirmed = window.confirm(
      `Send a REAL Binance Spot ${preview.side.toUpperCase()} for ${formatNumber(preview.requested_quantity, 8)} ${preview.asset_symbol}?\n\nEstimated value: ${formatCurrency(preview.estimated_quote_value)}\nThis action may execute immediately and cannot be undone.`
    );
    if (!confirmed) {
      return;
    }
    setBusy(true);
    setError(null);
    setStage("Order accepted by the Control Plane and awaiting Scrooge...");
    try {
      const queued = await fetchApi<SpotOrderQueueResponse>(
        `/api/portfolio/spot-orders/${encodeURIComponent(preview.intent_id)}/execute`,
        { method: "POST", body: { confirmation: "CONFIRM_SPOT_ORDER" } }
      );
      setStage("Scrooge is validating fresh balances and submitting the order...");
      const command = await waitForSpotOrder(queued.command_id);
      if (command.status !== "completed") {
        throw new Error(command.message || "Binance Spot order failed.");
      }
      const executedQuantity = command.result?.executed_quantity;
      setStage(
        typeof executedQuantity === "number"
          ? `Filled ${formatNumber(executedQuantity, 8)} ${preview.asset_symbol}. Treasury Ledger updated.`
          : command.message || "Order filled. Treasury Ledger updated."
      );
      setPreview(null);
      setQuantity("");
      await onExecuted();
    } catch (executeError) {
      setError(executeError instanceof Error ? executeError.message : "Could not execute the Binance Spot order.");
      setStage("Execution did not complete cleanly. Review the message and Treasury Ledger before taking any further action.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="treasury-spot-order-panel" aria-label={`Trade ${holding.asset_symbol} on Binance`}>
      <div className="treasury-spot-order-content">
        {!exchange?.is_balance_verified ? (
          <p className="treasury-spot-order-lock">A fresh Binance Spot snapshot is required.</p>
        ) : null}
        <form className="treasury-spot-order-form" onSubmit={(event) => void requestPreview(event)}>
          <label className="dialog-user-field">
            Side
            <select
              value={side}
              onChange={(event) => resetPreview(event.target.value as "buy" | "sell")}
              disabled={busy}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </label>
          <label className="dialog-user-field">
            Quantity ({holding.asset_symbol})
            <input
              type="number"
              min="0.00000001"
              step="any"
              value={quantity}
              onChange={(event) => {
                setQuantity(event.target.value);
                setPreview(null);
                setStage(null);
              }}
              required
              disabled={busy}
            />
          </label>
          <button type="submit" className="dialog-user-btn" disabled={busy || !executionReady}>
            {busy ? "Checking..." : "Preview Real Order"}
          </button>
        </form>
        <div className="treasury-spot-order-capacity">
          <span>Available USDT <strong>{formatCurrency(exchange?.usdt_free)}</strong></span>
          <span>Binance Free <strong>{formatNumber(holding.exchange_binance_free_quantity, 8)} {holding.asset_symbol}</strong></span>
          <span>Sellable <strong>{formatNumber(holding.immediately_sellable_quantity, 8)} {holding.asset_symbol}</strong></span>
        </div>
        {preview ? (
          <section className={`treasury-spot-order-preview treasury-spot-order-preview-${preview.side}`}>
            <header>
              <span>REAL ORDER PREVIEW</span>
              <strong>{preview.side.toUpperCase()} {formatNumber(preview.requested_quantity, 8)} {preview.asset_symbol}</strong>
            </header>
            <div>
              <span>Estimated Price <strong>{formatCurrency(preview.estimated_price, 6)}</strong></span>
              <span>Estimated Value <strong>{formatCurrency(preview.estimated_quote_value)}</strong></span>
              <span>Projected Holding <strong>{formatNumber(preview.projected_holding_quantity, 8)} {preview.asset_symbol}</strong></span>
              <span>Protected Floor <strong>{formatNumber(preview.protected_floor_quantity, 8)} {preview.asset_symbol}</strong></span>
            </div>
            <button type="button" className="dialog-user-btn treasury-spot-execute-btn" disabled={busy} onClick={() => void executeOrder()}>
              {busy ? "Executing..." : `Confirm Real ${preview.side === "buy" ? "Buy" : "Sell"}`}
            </button>
          </section>
        ) : null}
        {stage ? <p className="treasury-spot-order-stage">{stage}</p> : null}
        {error ? <p className="form-error">{error}</p> : null}
      </div>
    </section>
  );
}

function AssetPolicyPanel({
  holding,
  exchange,
  onUpdated,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  onUpdated: (response: UpdatePortfolioPolicyResponse) => void;
}): JSX.Element {
  const [targetQuantity, setTargetQuantity] = useState<string>(String(holding.target_quantity ?? holding.quantity));
  const [minimumHoldingPct, setMinimumHoldingPct] = useState<string>(String(holding.minimum_holding_pct ?? 100));
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const inventoryMessage = holding.amount_below_protected_floor > 0
    ? `Current Holding is ${formatNumber(holding.amount_below_protected_floor, 8)} ${holding.asset_symbol} below the Protected Floor.`
    : holding.amount_above_protected_floor <= 0
      ? "The full position is currently protected by policy."
      : !holding.sellable_inventory_is_exchange_verified
        ? "Live Binance Spot inventory is unavailable or stale, so immediate inventory is held at zero."
        : exchange?.can_trade !== true
          ? "Binance reports Spot trading unavailable, so immediate inventory is held at zero."
          : holding.immediately_sellable_quantity < holding.policy_sellable_quantity
            ? `Binance free balance limits immediate inventory to ${formatNumber(holding.immediately_sellable_quantity, 8)} ${holding.asset_symbol}.`
            : "The full policy-approved amount is free on Binance.";

  useEffect(() => {
    setTargetQuantity(String(holding.target_quantity ?? holding.quantity));
    setMinimumHoldingPct(String(holding.minimum_holding_pct ?? 100));
  }, [holding.target_quantity, holding.minimum_holding_pct, holding.quantity]);

  async function submitPolicy(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<UpdatePortfolioPolicyResponse>(
        `/api/portfolio/assets/${encodeURIComponent(holding.asset_symbol)}/policy`,
        {
          method: "POST",
          body: {
            quote_symbol: holding.quote_symbol,
            target_quantity: asNumber(targetQuantity),
            minimum_holding_pct: asNumber(minimumHoldingPct),
          },
        }
      );
      onUpdated(response);
    } catch (policyError) {
      setError(policyError instanceof Error ? policyError.message : "Could not update the asset policy.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="treasury-policy-panel">
      <header className="treasury-asset-panel-head">
        <span>Policy</span>
        <span className="treasury-policy-summary">
          Target {formatNumber(holding.target_quantity, 8)} {holding.asset_symbol}
          <span>Minimum {formatPercent(holding.minimum_holding_pct)}</span>
          <strong className={holding.immediately_sellable_quantity > 0 ? "value-positive" : "value-neutral"}>
            Sellable {formatNumber(holding.immediately_sellable_quantity, 8)}
          </strong>
        </span>
      </header>
      <div className="treasury-policy-content">
        <div className="treasury-policy-metrics">
          <div>
            <span>Current Total</span>
            <strong>{formatNumber(holding.quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Protected Floor</span>
            <strong>{formatNumber(holding.protected_floor_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Above Floor</span>
            <strong>{formatNumber(holding.amount_above_protected_floor, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Recorded Binance</span>
            <strong>{formatNumber(holding.binance_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Binance Free</span>
            <strong>{formatNumber(holding.exchange_binance_free_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Binance Locked</span>
            <strong>{formatNumber(holding.exchange_binance_locked_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div>
            <span>Policy Sellable</span>
            <strong>{formatNumber(holding.policy_sellable_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
          <div className={`treasury-policy-metric-sellable${
            holding.amount_below_protected_floor > 0
              ? " treasury-policy-metric-sellable-warning"
              : holding.immediately_sellable_quantity > 0
                ? " treasury-policy-metric-sellable-positive"
                : ""
          }`}>
            <span>Immediately Sellable</span>
            <strong>{formatNumber(holding.immediately_sellable_quantity, 8)} {holding.asset_symbol}</strong>
          </div>
        </div>
        <p className={`treasury-inventory-note${holding.amount_below_protected_floor > 0 ? " treasury-inventory-note-warning" : ""}`}>
          {inventoryMessage}
        </p>
        {!holding.sellable_inventory_is_exchange_verified ? (
          <p className="treasury-inventory-basis">
            Policy inventory remains visible, but execution inventory requires a fresh Binance Spot snapshot.
          </p>
        ) : null}
        {holding.sellable_inventory_is_exchange_verified && Math.abs(holding.binance_custody_variance) > 0.00000001 ? (
          <p className="treasury-inventory-basis treasury-inventory-basis-warning">
            Reconciliation needed: exchange total differs from recorded Binance custody by {formatNumber(holding.binance_custody_variance, 8)} {holding.asset_symbol}.
          </p>
        ) : null}
        <form className="treasury-policy-form" onSubmit={(event) => void submitPolicy(event)}>
          <label className="dialog-user-field">
            Target Holding
            <input
              type="number"
              value={targetQuantity}
              min="0.00000001"
              step="any"
              onChange={(event) => setTargetQuantity(event.target.value)}
              required
            />
          </label>
          <label className="dialog-user-field">
            Minimum Holding %
            <input
              type="number"
              value={minimumHoldingPct}
              min="0"
              max="100"
              step="any"
              onChange={(event) => setMinimumHoldingPct(event.target.value)}
              required
            />
          </label>
          <button type="submit" className="dialog-user-btn" disabled={saving}>
            {saving ? "Saving..." : "Update Policy"}
          </button>
        </form>
        <p className="treasury-policy-note">
          Target stays fixed when the Stack changes. Minimum Holding is always measured against Target.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div>
    </section>
  );
}

function AssetLedger({
  holding,
  refreshKey,
  onPortfolioUpdated,
}: {
  holding: PortfolioHolding;
  refreshKey: string;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse) => void;
}): JSX.Element {
  const [ledger, setLedger] = useState<AssetTransactionPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [updatingTransactionId, setUpdatingTransactionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadTransactions = useCallback(async (offset: number): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchApi<AssetTransactionPayload>(
        `/api/portfolio/assets/${encodeURIComponent(holding.asset_symbol)}/transactions` +
        `?quote_symbol=${encodeURIComponent(holding.quote_symbol)}&transaction_offset=${offset}`
      );
      setLedger(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the asset ledger.");
    } finally {
      setLoading(false);
    }
  }, [holding.asset_symbol, holding.quote_symbol]);

  useEffect(() => {
    void loadTransactions(0);
  }, [loadTransactions, refreshKey]);

  async function setTransactionStatus(transaction: PortfolioTransaction): Promise<void> {
    const nextStatus = transaction.status === "voided" ? "settled" : "voided";
    if (
      nextStatus === "voided" &&
      !window.confirm(`Void this ${transactionLabel(transaction.tx_type).toLowerCase()} entry for ${transaction.asset_symbol}?`)
    ) {
      return;
    }
    setUpdatingTransactionId(transaction.transaction_id);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>(
        `/api/portfolio/transactions/${encodeURIComponent(transaction.transaction_id)}/status`,
        { method: "POST", body: { status: nextStatus } }
      );
      await loadTransactions(ledger?.transaction_offset ?? 0);
      onPortfolioUpdated(response);
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Could not update the Treasury entry.");
    } finally {
      setUpdatingTransactionId(null);
    }
  }

  const transactions = ledger?.transactions ?? [];
  const count = ledger?.transaction_count ?? 0;
  const limit = ledger?.transaction_limit ?? 5;
  const offset = ledger?.transaction_offset ?? 0;
  const rangeStart = count > 0 ? offset + 1 : 0;
  const rangeEnd = Math.min(offset + transactions.length, count);
  const hasLater = offset > 0;
  const hasEarlier = offset + transactions.length < count;

  return (
    <section className="treasury-asset-ledger">
      <header className="treasury-asset-panel-head">
        <span>Asset Ledger</span>
        <span>{count} {count === 1 ? "entry" : "entries"}</span>
      </header>
      {loading && !ledger ? <p className="status-performance-note">Opening the ledger...</p> : null}
      {error ? <p className="form-error">{error}</p> : null}
      {!loading && transactions.length === 0 ? (
        <p className="trade-history-empty-sheet">No entries for {holding.asset_symbol} yet.</p>
      ) : null}
      {transactions.length ? (
        <>
          <div className="treasury-ledger-stack">
            {transactions.map((transaction) => (
              <article
                key={transaction.transaction_id}
                className={`treasury-ledger-row${transaction.status === "voided" ? " treasury-ledger-row-voided" : ""}`}
              >
                <span className={transactionToneClass(transaction.tx_type)}>{transactionLabel(transaction.tx_type)}</span>
                <span className="treasury-ledger-main">
                  <span className="treasury-ledger-entry">
                    {formatNumber(transaction.quantity, 8)} {transaction.asset_symbol}
                    {transaction.tx_type === "custody_transfer" && transaction.source_custody && transaction.destination_custody
                      ? ` from ${CUSTODY_LABELS[transaction.source_custody]} to ${CUSTODY_LABELS[transaction.destination_custody]}`
                      : transaction.price
                        ? ` at ${formatCurrency(transaction.price, 6)}`
                        : ` in ${CUSTODY_LABELS[transaction.custody_location]}`}
                  </span>
                  {transaction.status === "voided" ? <small>Voided</small> : null}
                </span>
                <span className="treasury-ledger-meta">{formatDateTimeEu(transaction.executed_at)}</span>
                <button
                  type="button"
                  className="treasury-ledger-action"
                  disabled={updatingTransactionId === transaction.transaction_id}
                  onClick={() => void setTransactionStatus(transaction)}
                >
                  {updatingTransactionId === transaction.transaction_id
                    ? "Saving..."
                    : transaction.status === "voided"
                      ? "Restore"
                      : "Void"}
                </button>
              </article>
            ))}
          </div>
          <div className="toolbar trade-history-toolbar treasury-ledger-toolbar">
            <button
              type="button"
              className="dialog-user-btn trade-history-nav-button trade-history-nav-later"
              disabled={loading || !hasLater}
              onClick={() => void loadTransactions(Math.max(0, offset - limit))}
            >
              Later
            </button>
            {hasLater ? (
              <button
                type="button"
                className="dialog-user-btn trade-history-latest-button"
                disabled={loading}
                onClick={() => void loadTransactions(0)}
              >
                Latest
              </button>
            ) : null}
            <span className="trade-history-page-indicator">
              Showing {rangeStart}-{rangeEnd} of {count}
            </span>
            <button
              type="button"
              className="dialog-user-btn trade-history-nav-button trade-history-nav-earlier"
              disabled={loading || !hasEarlier}
              onClick={() => void loadTransactions(offset + limit)}
            >
              Earlier
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}

function HoldingCard({
  holding,
  exchange,
  onPrepareTransaction,
  onPortfolioUpdated,
  onReload,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  onPrepareTransaction: (holding: PortfolioHolding, txType: "buy" | "sell") => void;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse | UpdatePortfolioPolicyResponse) => void;
  onReload: () => Promise<void>;
}): JSX.Element {
  const [expanded, setExpanded] = useState<boolean>(false);
  const refreshKey = [
    holding.quantity,
    holding.invested_capital,
    holding.binance_quantity,
    holding.cold_storage_quantity,
    holding.unassigned_quantity,
  ].join(":");

  return (
    <article className={`${holdingToneClass(holding.unrealized_pnl)}${expanded ? " treasury-holding-card-expanded" : ""}`}>
      <div className="treasury-holding-row">
        <button
          type="button"
          className="treasury-holding-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <span className="treasury-holding-head">
            <span className="treasury-coin">{holding.asset_symbol}</span>
            <span className="treasury-share">{formatPercent(holding.allocation_pct)}</span>
          </span>
          <span className="treasury-holding-lines">
            <span>
              <span>Stack</span>
              <strong>{formatNumber(holding.quantity, 8)} {holding.asset_symbol}</strong>
            </span>
            <span>
              <span>Entry Cost</span>
              <strong>{formatCurrency(holding.average_cost, 6)}</strong>
            </span>
            <span>
              <span>Market Price</span>
              <strong>{formatCurrency(holding.market_price, 6)}</strong>
            </span>
            <span>
              <span>Treasure Value</span>
              <strong>{formatCurrency(holding.market_value)}</strong>
            </span>
            <span>
              <span>Floating Gain</span>
              <strong className={signedToneClass(holding.unrealized_pnl, "treasury-inline-value")}>
                {formatSignedCurrency(holding.unrealized_pnl)} · {formatPercent(holding.unrealized_pnl_pct)}
              </strong>
            </span>
          </span>
          <span className="treasury-holding-chevron" aria-hidden="true" />
        </button>
        <footer className="treasury-holding-actions">
          <button type="button" onClick={() => onPrepareTransaction(holding, "buy")}>Add</button>
          <button type="button" onClick={() => onPrepareTransaction(holding, "sell")}>Reduce</button>
        </footer>
      </div>
      {expanded ? (
        <div className="treasury-asset-controls">
          <CustodyPanel
            key={`${holding.binance_quantity}:${exchange?.spot_execution_enabled ? 1 : 0}`}
            holding={holding}
            exchange={exchange}
            onTransferred={onPortfolioUpdated}
            onExecuted={onReload}
          />
          {!holding.is_dry_powder ? (
            <AssetPolicyPanel holding={holding} exchange={exchange} onUpdated={onPortfolioUpdated} />
          ) : null}
          <AssetLedger holding={holding} refreshKey={refreshKey} onPortfolioUpdated={onPortfolioUpdated} />
        </div>
      ) : null}
    </article>
  );
}

export default function TreasuryPage(): JSX.Element {
  const [portfolio, setPortfolio] = useState<PortfolioPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<TransactionFormState>(EMPTY_FORM);
  const [formExpanded, setFormExpanded] = useState<boolean>(false);
  const formPanelRef = useRef<HTMLElement | null>(null);

  const loadPortfolio = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const payload = await fetchApi<PortfolioPayload>("/api/portfolio");
      setPortfolio(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Treasury is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPortfolio();
  }, [loadPortfolio]);

  async function submitTransaction(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>("/api/portfolio/transactions", {
        method: "POST",
        body: {
          tx_type: form.tx_type,
          asset_symbol: form.asset_symbol,
          quantity: asNumber(form.quantity),
          price: asNumber(form.price),
          quote_symbol: form.quote_symbol,
          fee_amount: asNumber(form.fee_amount),
          fee_asset: form.fee_asset,
          executed_at: form.executed_at,
          note: form.note,
          custody_location: form.custody_location,
        },
      });
      setPortfolio(mergePortfolioPayload(response.portfolio, response.warnings));
      setForm(EMPTY_FORM);
      setFormExpanded(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not add treasure.");
    } finally {
      setSaving(false);
    }
  }

  function prepareHoldingTransaction(holding: PortfolioHolding, txType: "buy" | "sell"): void {
    setForm({
      ...EMPTY_FORM,
      tx_type: txType,
      asset_symbol: holding.asset_symbol,
      quote_symbol: holding.quote_symbol,
      fee_asset: holding.quote_symbol,
      custody_location: primaryCustody(holding),
    });
    setFormExpanded(true);
    window.requestAnimationFrame(() => {
      formPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  const summary = portfolio?.summary;
  const exchange = portfolio?.exchange ?? null;
  const holdings = [...(portfolio?.holdings ?? [])].sort((left, right) => {
    const allocationDifference = (right.allocation_pct ?? -1) - (left.allocation_pct ?? -1);
    return allocationDifference || left.asset_symbol.localeCompare(right.asset_symbol);
  });
  const timeline = portfolio?.timeline ?? [];
  const allocationHoldings = holdings.filter(
    (holding) => typeof holding.allocation_pct === "number" && holding.allocation_pct > 0
  );
  let allocationCursor = 0;
  const allocationStops = allocationHoldings.map((holding, index) => {
    const start = allocationCursor;
    allocationCursor = index === allocationHoldings.length - 1
      ? 100
      : allocationCursor + (holding.allocation_pct ?? 0);
    return `${ALLOCATION_COLORS[index % ALLOCATION_COLORS.length]} ${start}% ${allocationCursor}%`;
  });
  const allocationGradient = allocationStops.length
    ? `conic-gradient(${allocationStops.join(", ")})`
    : "conic-gradient(#273140 0% 100%)";
  const topThreeAllocation = allocationHoldings
    .slice(0, 3)
    .reduce((total, holding) => total + (holding.allocation_pct ?? 0), 0);
  const latestTimelinePnl = timeline.at(-1)?.unrealized_pnl;
  const timelinePnlTone =
    typeof latestTimelinePnl !== "number" || latestTimelinePnl === 0
      ? "neutral"
      : latestTimelinePnl > 0
        ? "positive"
        : "negative";
  const reducingHolding =
    form.tx_type === "sell" || form.tx_type === "withdraw"
      ? holdings.find(
          (holding) => holding.asset_symbol === form.asset_symbol && holding.quote_symbol === form.quote_symbol
        )
      : null;
  const reducingAvailable = reducingHolding
    ? custodyQuantity(reducingHolding, form.custody_location)
    : null;
  return (
    <AuthGate>
      <section className="panel page-shell treasury-page-shell">
        <p className="dialog-scrooge">Scrooge counts the vault before any coin gets a crown.</p>

        <section className="treasury-overview">
          <header className="treasury-section-head">
            <div>
              <h1>Treasury Overview</h1>
              {summary?.prices_updated_at ? (
                <p className="treasury-price-freshness">Prices checked {formatDateTimeEu(summary.prices_updated_at)}</p>
              ) : null}
            </div>
            <button
              type="button"
              className="dialog-user-btn treasury-refresh-btn"
              onClick={() => void loadPortfolio()}
              disabled={loading}
            >
              Refresh Treasury
            </button>
          </header>

          {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
          {loading ? <p className="status-performance-note">Scrooge counts the vault...</p> : null}

          <div className="treasury-summary-grid">
            <div className="treasury-summary-card treasury-summary-card-hero">
              <span className="treasury-summary-label">Total Treasure</span>
              <strong className="vault-value treasury-total-value">
                <span
                  className={signedToneClass(summary?.unrealized_pnl, "treasury-total-dollar")}
                  aria-hidden="true"
                >
                  $
                </span>
                <span>{formatNumber(summary?.total_value ?? 0)}</span>
              </strong>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Invested Capital</span>
              <strong className="vault-value">
                <span className="vault-dollar" aria-hidden="true">$</span>
                <span>{formatNumber(summary?.invested_capital ?? 0)}</span>
              </strong>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Floating Gain</span>
              <strong className={signedToneClass(summary?.unrealized_pnl, "treasury-summary-value")}>
                {formatSignedCurrency(summary?.unrealized_pnl ?? 0)}
              </strong>
              <span className="treasury-summary-note">{formatPercent(summary?.unrealized_pnl_pct)}</span>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Dry Powder</span>
              <strong>{formatCurrency(summary?.dry_powder ?? 0)}</strong>
              <span className="treasury-summary-note">{formatPercent(summary?.dry_powder_pct)} of vault</span>
            </div>
          </div>

          <div className="treasury-visibility-grid">
            <article className="treasury-insight-card treasury-allocation-card">
              <header className="treasury-insight-head">
                <div>
                  <h2>Treasure Map</h2>
                  <p className="muted">How the vault is divided right now.</p>
                </div>
                <span className="treasury-insight-count">{allocationHoldings.length} coins</span>
              </header>

              <div className="treasury-allocation-content">
                <div
                  className="treasury-allocation-donut"
                  style={{ background: allocationGradient }}
                  role="img"
                  aria-label="Current Treasury allocation"
                >
                  <div>
                    <strong>{formatPercent(topThreeAllocation)}</strong>
                    <span>Top 3</span>
                  </div>
                </div>
                {allocationHoldings.length ? <ol className="treasury-allocation-legend">
                  {allocationHoldings.map((holding, index) => (
                    <li key={`${holding.asset_symbol}-${holding.quote_symbol}`}>
                      <span
                        className="treasury-allocation-swatch"
                        style={{ backgroundColor: ALLOCATION_COLORS[index % ALLOCATION_COLORS.length] }}
                        aria-hidden="true"
                      />
                      <strong>{holding.asset_symbol}</strong>
                      <span>{formatPercent(holding.allocation_pct)}</span>
                      <small>{formatCurrency(holding.market_value)}</small>
                    </li>
                  ))}
                </ol> : <p className="treasury-allocation-empty">Add treasure to draw the map.</p>}
              </div>
              <footer className="treasury-allocation-foot">
                <span>Top 3 concentration <strong>{formatPercent(topThreeAllocation)}</strong></span>
                <span>Dry Powder <strong>{formatPercent(summary?.dry_powder_pct)}</strong></span>
              </footer>
            </article>

            <article className="treasury-insight-card treasury-timeline-card">
              <header className="treasury-insight-head">
                <div>
                  <h2>Treasury Timeline</h2>
                  <p className="muted">Daily valuation marks from the Control Plane.</p>
                </div>
                <span className="treasury-insight-count">
                  {timeline.length} {timeline.length === 1 ? "day" : "days"}
                </span>
              </header>
              <div className="treasury-timeline-grid">
                <TimelineSeries title="Treasure Value" timeline={timeline} valueKey="total_value" tone="gold" />
                <TimelineSeries title="Floating PnL" timeline={timeline} valueKey="unrealized_pnl" tone={timelinePnlTone} />
              </div>
              <footer className="treasury-timeline-range">
                {timeline.length ? (
                  <>
                    <span>{formatTimelineDate(timeline[0].snapshot_date)}</span>
                    <span>Daily marks</span>
                    <span>{formatTimelineDate(timeline[timeline.length - 1].snapshot_date)}</span>
                  </>
                ) : (
                  <span>Refresh Treasury to place the first mark.</span>
                )}
              </footer>
            </article>
          </div>

        </section>

        <section ref={formPanelRef} className="section-block">
          <header className="treasury-section-head">
            <div>
              <h2>Vault Holdings</h2>
              <p className="muted">Current stacks derived from settled Treasury entries.</p>
            </div>
            <button
              type="button"
              className="dialog-user-btn treasury-form-toggle"
              aria-expanded={formExpanded}
              onClick={() => setFormExpanded((current) => !current)}
            >
              {formExpanded ? "Close Entry" : "Add Treasure"}
            </button>
          </header>
          {formExpanded ? <form className="treasury-form" onSubmit={(event) => void submitTransaction(event)}>
            <label className="dialog-user-field">
              Action
              <select
                value={form.tx_type}
                onChange={(event) => setForm((current) => ({ ...current, tx_type: event.target.value as PortfolioTransactionType }))}
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
                <option value="deposit">Deposit</option>
                <option value="withdraw">Withdraw</option>
                <option value="adjustment">Adjustment</option>
              </select>
            </label>
            <label className="dialog-user-field">
              Coin
              <input
                type="text"
                value={form.asset_symbol}
                placeholder="BTC"
                autoCapitalize="characters"
                onChange={(event) => setForm((current) => ({ ...current, asset_symbol: event.target.value.toUpperCase() }))}
                required
              />
            </label>
            <label className="dialog-user-field">
              Custody
              <select
                value={form.custody_location}
                onChange={(event) => setForm((current) => ({
                  ...current,
                  custody_location: event.target.value as CustodyLocation,
                }))}
              >
                {CUSTODY_LOCATIONS.map((location) => (
                  <option key={location} value={location}>{CUSTODY_LABELS[location]}</option>
                ))}
              </select>
            </label>
            <label className="dialog-user-field">
              Stack
              <input
                type="number"
                value={form.quantity}
                placeholder="0.25"
                min="0"
                max={reducingAvailable ?? undefined}
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, quantity: event.target.value }))}
                required
              />
              {reducingHolding ? (
                <small className="treasury-field-hint">
                  Available in {CUSTODY_LABELS[form.custody_location]}: {formatNumber(reducingAvailable, 8)} {reducingHolding.asset_symbol}
                </small>
              ) : null}
            </label>
            <label className="dialog-user-field">
              Entry Cost
              <input
                type="number"
                value={form.price}
                placeholder="65000"
                min="0"
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, price: event.target.value }))}
              />
            </label>
            <label className="dialog-user-field">
              Quote
              <input
                type="text"
                value={form.quote_symbol}
                autoCapitalize="characters"
                onChange={(event) => {
                  const quote = event.target.value.toUpperCase();
                  setForm((current) => ({ ...current, quote_symbol: quote, fee_asset: current.fee_asset || quote }));
                }}
                required
              />
            </label>
            <label className="dialog-user-field">
              Fee
              <input
                type="number"
                value={form.fee_amount}
                placeholder="0"
                min="0"
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, fee_amount: event.target.value }))}
              />
            </label>
            <label className="dialog-user-field">
              Fee Coin
              <input
                type="text"
                value={form.fee_asset}
                autoCapitalize="characters"
                onChange={(event) => setForm((current) => ({ ...current, fee_asset: event.target.value.toUpperCase() }))}
              />
            </label>
            <label className="dialog-user-field treasury-form-wide">
              Treasury Notes
              <input
                type="text"
                value={form.note}
                placeholder="Thesis, source, or exit thought"
                onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))}
              />
            </label>
            <button type="submit" className="dialog-user-btn treasury-submit-btn" disabled={saving}>
              {saving ? "Adding Treasure..." : "Add Treasure"}
            </button>
          </form> : null}

          {holdings.length === 0 ? (
            <p className="trade-history-empty-sheet">The treasury is empty. Add your first treasure.</p>
          ) : (
            <div className="treasury-holdings-grid">
              {holdings.map((holding) => (
                <HoldingCard
                  key={`${holding.asset_symbol}-${holding.quote_symbol}`}
                  holding={holding}
                  exchange={exchange}
                  onPrepareTransaction={prepareHoldingTransaction}
                  onPortfolioUpdated={(response) => {
                    setPortfolio(mergePortfolioPayload(response.portfolio, response.warnings));
                  }}
                  onReload={loadPortfolio}
                />
              ))}
            </div>
          )}
        </section>

        {portfolio?.warnings.length ? (
          <section className="section-block">
            <h2>Red Flags</h2>
            <div className="dialog-scrooge dialog-scrooge-warning">
              {portfolio.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          </section>
        ) : null}
      </section>
    </AuthGate>
  );
}
