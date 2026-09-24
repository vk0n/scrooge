"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type PortfolioSummary = {
  total_value: number;
  invested_capital: number;
  total_gain: number;
  total_gain_pct: number | null;
  unrealized_pnl: number;
  unrealized_pnl_pct: number | null;
  vault_reserve: number;
  vault_reserve_pct: number | null;
  dry_powder: number;
  dry_powder_pct: number | null;
  vault_reserve_available: number;
  vault_reserve_committed: number;
  largest_position: PortfolioHolding | null;
  holding_count: number;
  open_swing_count: number;
  closed_swing_count: number;
  total_swing_count: number;
  open_swing_asset_count: number;
  realized_accumulated_cash: number;
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
  trading_objective: "accumulate_cash" | "accumulate_asset" | null;
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
  initial_quantity: number;
  initial_capital: number;
  accumulated_asset_quantity: number;
  accumulated_cash_gain: number;
  open_bargain_pnl: number;
  settled_quantity: number;
  effective_cost_basis: number;
  effective_entry_cost: number | null;
  market_gain: number | null;
  floating_gain: number | null;
  total_gain: number | null;
  total_gain_pct: number | null;
  spot_trading_state: "locked" | "unlocked";
  spot_trading_state_reason:
    | "execution_disabled"
    | "not_policy_managed"
    | "fully_protected"
    | "policy_allows_trading";
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
  total_gain: number;
  unrealized_pnl: number;
  vault_reserve: number;
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

type SpotSwingExecution = {
  execution_id: string;
  side: "buy" | "sell";
  quantity: number;
  price: number;
  quote_quantity: number | null;
  fee_amount: number | null;
  fee_asset: string | null;
  source: "manual" | "strategy";
  reason_text: string | null;
  executed_at: string;
};

type SpotSwingEconomics = {
  status: "open" | "partially_closed" | "accepting_loss" | "closed";
  origin_side: "buy" | "sell";
  closing_side: "buy" | "sell";
  opening_quantity: number;
  opening_quote_quantity: number;
  closing_quantity: number;
  closing_quote_quantity: number;
  remaining_quantity: number;
  weighted_opening_price: number | null;
  weighted_closing_price: number | null;
  realized_pnl_quote: number;
  realized_cash_gain_quote: number | null;
  realized_net_asset_change: number | null;
  realized_asset_gain: number | null;
  target_ratchet_quantity: number;
  unrealized_pnl_quote: number | null;
  fees_by_asset: Record<string, number>;
  unpriced_fees_by_asset: Record<string, number>;
};

type SpotSwing = {
  swing_id: string;
  asset_symbol: string;
  quote_symbol: string;
  origin_side: "buy" | "sell";
  trading_objective: "accumulate_cash" | "accumulate_asset" | null;
  status: SpotSwingEconomics["status"];
  planned_quantity: number | null;
  reference_state: Record<string, unknown>;
  strategy_reason: Record<string, unknown>;
  source: "manual" | "strategy";
  close_reason: string | null;
  opened_at_ms: number;
  closed_at_ms: number | null;
  current_market_price: number | null;
  market_price_updated_at: string | null;
  age_seconds: number;
  economics: SpotSwingEconomics;
  executions: SpotSwingExecution[];
};

type AssetLedgerEntry =
  | {
      entry_type: "transaction";
      entry_id: string;
      occurred_at_ms: number;
      occurred_at: string;
      transaction: PortfolioTransaction;
    }
  | {
      entry_type: "swing";
      entry_id: string;
      occurred_at_ms: number;
      occurred_at: string;
      swing: SpotSwing;
    };

type AssetLedgerFilter = "all" | "open" | "closed";

type AssetLedgerPayload = {
  asset_symbol: string;
  quote_symbol: string;
  filter: AssetLedgerFilter;
  entries: AssetLedgerEntry[];
  entry_count: number;
  entry_limit: number;
  entry_offset: number;
};

type BargainLedgerPayload = {
  filter: AssetLedgerFilter;
  entries: Array<Extract<AssetLedgerEntry, { entry_type: "swing" }>>;
  entry_count: number;
  entry_limit: number;
  entry_offset: number;
  open_count: number;
  closed_count: number;
  total_count: number;
};

type UpdatePortfolioPolicyResponse = {
  policy: {
    asset_symbol: string;
    quote_symbol: string;
    target_quantity: number;
    minimum_holding_pct: number;
    trading_objective: "accumulate_cash" | "accumulate_asset" | null;
  };
  portfolio: Omit<PortfolioPayload, "warnings">;
  warnings: string[];
};

type PortfolioTransactionType = "buy" | "sell" | "deposit" | "withdraw" | "adjustment" | "custody_transfer";
type CustodyLocation = "unassigned" | "binance" | "cold_storage";
type CustodyAction = "move" | "bring_in" | "release";
type TreasureIntakeMode = "bring_in" | "buy_binance";

type TreasureIntakeFormState = {
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

const EMPTY_INTAKE_FORM: TreasureIntakeFormState = {
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
const STABLE_ASSETS = new Set(["USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI"]);

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

function formatAssetQuantity(value: number | null | undefined, assetSymbol: string): string {
  if (assetSymbol !== "USDT") {
    return formatNumber(value, 8);
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
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
    return "Brought In";
  }
  if (type === "withdraw") {
    return "Released";
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

function swingStatusLabel(status: SpotSwingEconomics["status"]): string {
  return status.replaceAll("_", " ").toUpperCase();
}

function swingIdentity(swingId: string): string {
  const compact = swingId.replaceAll("-", "");
  return `Bargain #${compact.slice(-6).toUpperCase()}`;
}

function formatSwingAge(seconds: number): string {
  const normalized = Math.max(0, Math.floor(seconds));
  const days = Math.floor(normalized / 86400);
  if (days > 0) return `${days}d ${Math.floor((normalized % 86400) / 3600)}h`;
  const hours = Math.floor(normalized / 3600);
  if (hours > 0) return `${hours}h ${Math.floor((normalized % 3600) / 60)}m`;
  return `${Math.floor(normalized / 60)}m`;
}

function swingObjectiveLabel(objective: SpotSwing["trading_objective"]): string {
  if (objective === "accumulate_cash") return "Accumulate Cash";
  if (objective === "accumulate_asset") return "Accumulate Asset";
  return "Not Set";
}

function swingReasonText(reason: Record<string, unknown>): string | null {
  for (const key of ["message", "reason", "summary", "signal"]) {
    const value = reason[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return Object.keys(reason).length ? JSON.stringify(reason) : null;
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
  valueKey: "total_value" | "total_gain"
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
  valueKey: "total_value" | "total_gain";
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
    valueKey === "total_gain" ? formatSignedCurrency(value) : formatCurrency(value);

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
  onPortfolioUpdated,
  onExecuted,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse) => void;
  onExecuted: () => Promise<void>;
}): JSX.Element {
  const initialSource = primaryCustody(holding);
  const [action, setAction] = useState<CustodyAction>("move");
  const [source, setSource] = useState<CustodyLocation>(initialSource);
  const [destination, setDestination] = useState<CustodyLocation>(
    initialSource === "binance" ? "cold_storage" : "binance"
  );
  const [quantity, setQuantity] = useState<string>("");
  const [note, setNote] = useState<string>("");
  const [boundaryCustody, setBoundaryCustody] = useState<CustodyLocation>(initialSource);
  const [boundaryQuantity, setBoundaryQuantity] = useState<string>("");
  const [entryCost, setEntryCost] = useState<string>("");
  const [feeAmount, setFeeAmount] = useState<string>("");
  const [feeAsset, setFeeAsset] = useState<string>(holding.quote_symbol);
  const [boundaryNote, setBoundaryNote] = useState<string>("");
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [tradeExpanded, setTradeExpanded] = useState<boolean>(false);
  const available = custodyQuantity(holding, source);
  const releasable = custodyQuantity(holding, boundaryCustody);
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
      onPortfolioUpdated(response);
    } catch (transferError) {
      setError(transferError instanceof Error ? transferError.message : "Could not record the custody move.");
    } finally {
      setSaving(false);
    }
  }

  function changeAction(nextAction: CustodyAction): void {
    setAction(nextAction);
    setError(null);
    setTradeExpanded(false);
    setBoundaryQuantity("");
    setEntryCost("");
    setFeeAmount("");
    setBoundaryNote("");
  }

  async function submitBoundaryChange(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const isBringIn = action === "bring_in";
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>("/api/portfolio/transactions", {
        method: "POST",
        body: {
          tx_type: isBringIn ? "deposit" : "withdraw",
          asset_symbol: holding.asset_symbol,
          quantity: asNumber(boundaryQuantity),
          price: isBringIn ? asNumber(entryCost) : null,
          quote_symbol: holding.quote_symbol,
          fee_amount: isBringIn ? asNumber(feeAmount) : null,
          fee_asset: isBringIn ? feeAsset : holding.quote_symbol,
          note: boundaryNote,
          custody_location: boundaryCustody,
        },
      });
      setBoundaryQuantity("");
      setEntryCost("");
      setFeeAmount("");
      setBoundaryNote("");
      onPortfolioUpdated(response);
    } catch (boundaryError) {
      const fallback = isBringIn
        ? "Could not bring treasure into the vault."
        : "Could not release treasure from the vault.";
      setError(boundaryError instanceof Error ? boundaryError.message : fallback);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="treasury-custody-panel">
      <header className="treasury-asset-panel-head">
        <button
          type="button"
          className="treasury-asset-panel-toggle"
          aria-expanded={expanded}
          onClick={() => {
            setExpanded((current) => !current);
            if (expanded) setTradeExpanded(false);
          }}
        >
          <span>Custody</span>
          <span className="treasury-custody-summary">
            {CUSTODY_LOCATIONS.map((location) => (
              <span key={location}>
                {CUSTODY_LABELS[location]} {formatNumber(custodyQuantity(holding, location), 8)}
              </span>
            ))}
          </span>
          <span className="treasury-asset-panel-chevron" aria-hidden="true" />
        </button>
      </header>
      {expanded ? <div className="treasury-custody-content">
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
        <div className="treasury-custody-action-bar" aria-label="Treasury custody action">
          {([
            ["move", "Move Treasure"],
            ["bring_in", "Bring Treasure In"],
            ["release", "Release Treasure"],
          ] as const).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={`treasury-custody-action-btn${action === value ? " treasury-custody-action-btn-active" : ""}`}
              aria-pressed={action === value}
              onClick={() => changeAction(value)}
            >
              {label}
            </button>
          ))}
        </div>
        {action === "move" ? (
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
        ) : (
          <form className="treasury-custody-boundary-form" onSubmit={(event) => void submitBoundaryChange(event)}>
            <label className="dialog-user-field">
              Custody
              <select
                value={boundaryCustody}
                onChange={(event) => {
                  setBoundaryCustody(event.target.value as CustodyLocation);
                  setBoundaryQuantity("");
                }}
              >
                {CUSTODY_LOCATIONS.map((location) => (
                  <option
                    key={location}
                    value={location}
                    disabled={action === "release" && custodyQuantity(holding, location) <= 0}
                  >
                    {CUSTODY_LABELS[location]}
                  </option>
                ))}
              </select>
            </label>
            <label className="dialog-user-field">
              Stack
              <input
                type="number"
                value={boundaryQuantity}
                min="0"
                max={action === "release" ? releasable : undefined}
                step="any"
                placeholder={action === "release" ? formatNumber(releasable, 8) : "0.25"}
                onChange={(event) => setBoundaryQuantity(event.target.value)}
                required
              />
              {action === "release" ? (
                <small className="treasury-field-hint">
                  Available in {CUSTODY_LABELS[boundaryCustody]}: {formatNumber(releasable, 8)} {holding.asset_symbol}
                </small>
              ) : null}
            </label>
            {action === "bring_in" ? (
              <>
                <label className="dialog-user-field">
                  Entry Cost
                  <input
                    type="number"
                    value={entryCost}
                    min="0"
                    step="any"
                    placeholder={holding.is_dry_powder ? "1" : formatNumber(holding.average_cost, 6)}
                    onChange={(event) => setEntryCost(event.target.value)}
                    required={!holding.is_dry_powder}
                  />
                </label>
                <label className="dialog-user-field">
                  Fee
                  <input
                    type="number"
                    value={feeAmount}
                    min="0"
                    step="any"
                    placeholder="0"
                    onChange={(event) => setFeeAmount(event.target.value)}
                  />
                </label>
                <label className="dialog-user-field">
                  Fee Coin
                  <input
                    type="text"
                    value={feeAsset}
                    autoCapitalize="characters"
                    onChange={(event) => setFeeAsset(event.target.value.toUpperCase())}
                  />
                </label>
              </>
            ) : null}
            <label className="dialog-user-field treasury-custody-boundary-note">
              Note
              <input
                type="text"
                value={boundaryNote}
                maxLength={500}
                placeholder={action === "bring_in" ? "Where this treasure came from" : "Why this treasure leaves the vault"}
                onChange={(event) => setBoundaryNote(event.target.value)}
              />
            </label>
            <button
              type="submit"
              className="dialog-user-btn treasury-custody-boundary-submit"
              disabled={saving || (action === "release" && releasable <= 0)}
            >
              {saving
                ? "Recording..."
                : action === "bring_in"
                  ? "Bring Treasure In"
                  : "Release Treasure"}
            </button>
          </form>
        )}
        <p className="treasury-custody-disclaimer">
          Treasury accounting only. No exchange or blockchain transfer is initiated.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div> : null}
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
  assetSymbol,
  quoteSymbol = "USDT",
  treasuryIntake = false,
  exchange,
  onExecuted,
}: {
  holding?: PortfolioHolding;
  assetSymbol?: string;
  quoteSymbol?: string;
  treasuryIntake?: boolean;
  exchange: PortfolioExchange | null;
  onExecuted: () => Promise<void>;
}): JSX.Element {
  const resolvedAsset = holding?.asset_symbol ?? assetSymbol?.trim().toUpperCase() ?? "";
  const resolvedQuote = holding?.quote_symbol ?? quoteSymbol;
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
          asset_symbol: resolvedAsset,
          quote_symbol: resolvedQuote,
          side,
          quantity: asNumber(quantity),
          treasury_intake: treasuryIntake,
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
    <section
      className={`treasury-spot-order-panel${treasuryIntake ? " treasury-spot-order-panel-intake" : ""}`}
      aria-label={`Trade ${resolvedAsset || "a new treasure"} on Binance`}
    >
      <div className="treasury-spot-order-content">
        {!exchange?.is_balance_verified ? (
          <p className="treasury-spot-order-lock">A fresh Binance Spot snapshot is required.</p>
        ) : null}
        <form className="treasury-spot-order-form" onSubmit={(event) => void requestPreview(event)}>
          {treasuryIntake ? null : (
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
          )}
          <label className="dialog-user-field">
            Quantity ({resolvedAsset || "Coin"})
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
          <button type="submit" className="dialog-user-btn" disabled={busy || !executionReady || !resolvedAsset}>
            {busy ? "Checking..." : treasuryIntake ? "Preview Real Buy" : "Preview Real Order"}
          </button>
        </form>
        <div className="treasury-spot-order-capacity">
          <span>Available USDT <strong>{formatCurrency(exchange?.usdt_free)}</strong></span>
          {holding ? (
            <>
              <span>Binance Free <strong>{formatNumber(holding.exchange_binance_free_quantity, 8)} {holding.asset_symbol}</strong></span>
              <span>Sellable <strong>{formatNumber(holding.immediately_sellable_quantity, 8)} {holding.asset_symbol}</strong></span>
            </>
          ) : (
            <span>New policy <strong>Target quantity · 100% protected</strong></span>
          )}
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
              {treasuryIntake ? (
                <span>Initial Policy <strong>100% protected · Accumulate Cash</strong></span>
              ) : (
                <span>Protected Floor <strong>{formatNumber(preview.protected_floor_quantity, 8)} {preview.asset_symbol}</strong></span>
              )}
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
  const [tradingObjective, setTradingObjective] = useState<string>(holding.trading_objective ?? "accumulate_cash");
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);
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
    setTradingObjective(holding.trading_objective ?? "accumulate_cash");
  }, [holding.target_quantity, holding.minimum_holding_pct, holding.trading_objective, holding.quantity]);

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
            trading_objective: tradingObjective,
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
        <button
          type="button"
          className="treasury-asset-panel-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <span>Policy</span>
          <span className="treasury-policy-summary">
            Target {formatNumber(holding.target_quantity, 8)} {holding.asset_symbol}
            <span>Minimum {formatPercent(holding.minimum_holding_pct)}</span>
            <span>{swingObjectiveLabel(holding.trading_objective ?? "accumulate_cash")}</span>
            <strong className={holding.immediately_sellable_quantity > 0 ? "value-positive" : "value-neutral"}>
              Sellable {formatNumber(holding.immediately_sellable_quantity, 8)}
            </strong>
          </span>
          <span className="treasury-asset-panel-chevron" aria-hidden="true" />
        </button>
      </header>
      {expanded ? <div className="treasury-policy-content">
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
          <label className="dialog-user-field">
            Trading Objective
            <select value={tradingObjective} onChange={(event) => setTradingObjective(event.target.value)}>
              <option value="accumulate_cash">Accumulate Cash</option>
              <option value="accumulate_asset">Accumulate Asset</option>
            </select>
          </label>
          <button type="submit" className="dialog-user-btn" disabled={saving}>
            {saving ? "Saving..." : "Update Policy"}
          </button>
        </form>
        <p className="treasury-policy-note">
          Target stays fixed when the Stack changes. Minimum Holding is always measured against Target.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div> : null}
    </section>
  );
}

function SwingLedgerRow({ swing, occurredAt }: { swing: SpotSwing; occurredAt: string }): JSX.Element {
  const [expanded, setExpanded] = useState<boolean>(false);
  const economics = swing.economics;
  const pnl = economics.status === "closed"
    ? economics.realized_pnl_quote
    : economics.unrealized_pnl_quote;
  const quantity = economics.opening_quantity || swing.planned_quantity;
  const fees = Object.entries(economics.fees_by_asset);
  const reason = swingReasonText(swing.strategy_reason);

  return (
    <article className={`treasury-swing-row treasury-swing-row-${economics.status}`}>
      <button
        type="button"
        className="treasury-swing-toggle"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span className="treasury-ledger-type treasury-swing-type">Bargain</span>
        <span className="treasury-swing-summary">
          <strong>{swingIdentity(swing.swing_id)}</strong>
          <span>
            {swing.origin_side.toUpperCase()} {formatNumber(quantity, 8)} {swing.asset_symbol}
            {economics.weighted_opening_price !== null
              ? ` at ${formatCurrency(economics.weighted_opening_price, 6)}`
              : " · Awaiting first execution"}
          </span>
        </span>
        <span className={`treasury-swing-status treasury-swing-status-${economics.status}`}>
          {swingStatusLabel(economics.status)}
        </span>
        <strong className={signedToneClass(pnl, "treasury-swing-pnl")}>{formatSignedCurrency(pnl)}</strong>
        <span className="treasury-holding-chevron" aria-hidden="true" />
      </button>
      {expanded ? (
        <div className="treasury-swing-details">
          <div className="treasury-swing-metrics">
            <span><small>Origin</small><strong>{swing.origin_side.toUpperCase()}</strong></span>
            <span><small>Objective</small><strong>{swingObjectiveLabel(swing.trading_objective)}</strong></span>
            <span><small>Remaining</small><strong>{formatNumber(economics.remaining_quantity, 8)} {swing.asset_symbol}</strong></span>
            <span><small>Opened</small><strong>{formatDateTimeEu(occurredAt)}</strong></span>
            <span><small>Age</small><strong>{formatSwingAge(swing.age_seconds)}</strong></span>
            <span><small>Market Price</small><strong>{formatCurrency(swing.current_market_price, 6)}</strong></span>
            <span><small>Realized PnL</small><strong className={signedToneClass(economics.realized_pnl_quote, "")}>{formatSignedCurrency(economics.realized_pnl_quote)}</strong></span>
            <span><small>Open PnL</small><strong className={signedToneClass(economics.unrealized_pnl_quote, "")}>{formatSignedCurrency(economics.unrealized_pnl_quote)}</strong></span>
          </div>
          {reason ? <p className="treasury-swing-reason">Scrooge&apos;s note: {reason}</p> : null}
          <div className="treasury-swing-execution-head">
            <span>Executions</span>
            <small>{swing.executions.length} {swing.executions.length === 1 ? "fill" : "fills"}</small>
          </div>
          {swing.executions.length ? (
            <div className="treasury-swing-executions">
              {swing.executions.map((execution) => (
                <div key={execution.execution_id}>
                  <span className={`treasury-swing-side treasury-swing-side-${execution.side}`}>
                    {execution.side.toUpperCase()}
                  </span>
                  <strong>{formatNumber(execution.quantity, 8)} {swing.asset_symbol} at {formatCurrency(execution.price, 6)}</strong>
                  <span>
                    Fee {execution.fee_amount
                      ? `${formatNumber(execution.fee_amount, 8)} ${execution.fee_asset ?? "Unknown"}`
                      : "none"}
                  </span>
                  <time>{formatDateTimeEu(execution.executed_at)}</time>
                </div>
              ))}
            </div>
          ) : (
            <p className="status-performance-note">No executions have reached this Bargain yet.</p>
          )}
          <footer className="treasury-swing-footer">
            <span>Full ID <strong>{swing.swing_id}</strong></span>
            <span>
              Fees <strong>{fees.length
                ? fees.map(([asset, amount]) => `${formatNumber(amount, 8)} ${asset}`).join(" · ")
                : "None"}</strong>
            </span>
          </footer>
        </div>
      ) : null}
    </article>
  );
}

function PortfolioBargainLedger({ refreshKey }: { refreshKey: string }): JSX.Element {
  const [ledger, setLedger] = useState<BargainLedgerPayload | null>(null);
  const [filter, setFilter] = useState<AssetLedgerFilter>("all");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadEntries = useCallback(async (offset: number): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchApi<BargainLedgerPayload>(
        `/api/portfolio/bargains?filter=${encodeURIComponent(filter)}&entry_offset=${offset}`
      );
      setLedger(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the Bargain ledger.");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    void loadEntries(0);
  }, [loadEntries, refreshKey]);

  function changeFilter(nextFilter: AssetLedgerFilter): void {
    if (nextFilter === filter) return;
    setLoading(true);
    setFilter(nextFilter);
    setLedger(null);
  }

  const entries = ledger?.entries ?? [];
  const count = ledger?.entry_count ?? 0;
  const limit = ledger?.entry_limit ?? 5;
  const offset = ledger?.entry_offset ?? 0;
  const rangeStart = count > 0 ? offset + 1 : 0;
  const rangeEnd = Math.min(offset + entries.length, count);
  const hasLater = offset > 0;
  const hasEarlier = offset + entries.length < count;

  return (
    <section id="treasury-bargain-ledger" className="treasury-bargain-ledger" aria-label="Bargain Ledger">
      <header className="treasury-bargain-ledger-head">
        <div>
          <h2>Bargain Ledger</h2>
          <p>Every active and settled Bargain across the vault.</p>
        </div>
        <span>{ledger ? `${count} ${count === 1 ? "entry" : "entries"}` : "Loading"}</span>
      </header>
      <div className="treasury-ledger-filter" aria-label="Bargain Ledger filter">
        {(["all", "open", "closed"] as const).map((value) => (
          <button
            key={value}
            type="button"
            className={filter === value ? "treasury-ledger-filter-active" : undefined}
            aria-pressed={filter === value}
            onClick={() => changeFilter(value)}
          >
            {value[0].toUpperCase() + value.slice(1)}
          </button>
        ))}
      </div>
      {loading && !ledger ? <p className="status-performance-note">Opening the Bargain ledger...</p> : null}
      {error ? <p className="form-error">{error}</p> : null}
      {!loading && entries.length === 0 ? (
        <p className="trade-history-empty-sheet">
          {filter === "all" ? "No Bargains have been recorded yet." : `No ${filter} Bargains.`}
        </p>
      ) : null}
      {entries.length ? (
        <>
          <div className="treasury-ledger-stack">
            {entries.map((entry) => (
              <SwingLedgerRow
                key={entry.entry_id}
                swing={entry.swing}
                occurredAt={entry.occurred_at}
              />
            ))}
          </div>
          <div className="toolbar trade-history-toolbar treasury-ledger-toolbar">
            <button
              type="button"
              className="dialog-user-btn trade-history-nav-button trade-history-nav-later"
              disabled={loading || !hasLater}
              onClick={() => void loadEntries(Math.max(0, offset - limit))}
            >
              Later
            </button>
            {hasLater ? (
              <button
                type="button"
                className="dialog-user-btn trade-history-latest-button"
                disabled={loading}
                onClick={() => void loadEntries(0)}
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
              onClick={() => void loadEntries(offset + limit)}
            >
              Earlier
            </button>
          </div>
        </>
      ) : null}
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
  const [ledger, setLedger] = useState<AssetLedgerPayload | null>(null);
  const [filter, setFilter] = useState<AssetLedgerFilter>("all");
  const [loading, setLoading] = useState<boolean>(true);
  const [updatingTransactionId, setUpdatingTransactionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);

  const loadEntries = useCallback(async (offset: number): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchApi<AssetLedgerPayload>(
        `/api/portfolio/assets/${encodeURIComponent(holding.asset_symbol)}/ledger` +
        `?quote_symbol=${encodeURIComponent(holding.quote_symbol)}` +
        `&filter=${encodeURIComponent(filter)}&entry_offset=${offset}`
      );
      setLedger(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the asset ledger.");
    } finally {
      setLoading(false);
    }
  }, [filter, holding.asset_symbol, holding.quote_symbol]);

  useEffect(() => {
    if (expanded) void loadEntries(0);
  }, [expanded, loadEntries, refreshKey]);

  function changeFilter(nextFilter: AssetLedgerFilter): void {
    if (nextFilter === filter) return;
    setFilter(nextFilter);
    setLedger(null);
  }

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
      await loadEntries(ledger?.entry_offset ?? 0);
      onPortfolioUpdated(response);
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Could not update the Treasury entry.");
    } finally {
      setUpdatingTransactionId(null);
    }
  }

  const entries = ledger?.entries ?? [];
  const count = ledger?.entry_count ?? 0;
  const limit = ledger?.entry_limit ?? 5;
  const offset = ledger?.entry_offset ?? 0;
  const rangeStart = count > 0 ? offset + 1 : 0;
  const rangeEnd = Math.min(offset + entries.length, count);
  const hasLater = offset > 0;
  const hasEarlier = offset + entries.length < count;

  return (
    <section className="treasury-asset-ledger">
      <header className="treasury-asset-panel-head">
        <button
          type="button"
          className="treasury-asset-panel-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <span>Asset Ledger</span>
          <span>{ledger ? `${count} ${count === 1 ? "entry" : "entries"}` : "History"}</span>
          <span className="treasury-asset-panel-chevron" aria-hidden="true" />
        </button>
      </header>
      {expanded ? (
        <div className="treasury-ledger-filter" aria-label="Asset Ledger filter">
          {(["all", "open", "closed"] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={filter === value ? "treasury-ledger-filter-active" : undefined}
              aria-pressed={filter === value}
              onClick={() => changeFilter(value)}
            >
              {value[0].toUpperCase() + value.slice(1)}
            </button>
          ))}
        </div>
      ) : null}
      {expanded && loading && !ledger ? <p className="status-performance-note">Opening the ledger...</p> : null}
      {expanded && error ? <p className="form-error">{error}</p> : null}
      {expanded && !loading && entries.length === 0 ? (
        <p className="trade-history-empty-sheet">
          {filter === "all"
            ? `No entries for ${holding.asset_symbol} yet.`
            : `No ${filter} Bargains for ${holding.asset_symbol}.`}
        </p>
      ) : null}
      {expanded && entries.length ? (
        <>
          <div className="treasury-ledger-stack">
            {entries.map((entry) => {
              if (entry.entry_type === "swing") {
                return <SwingLedgerRow key={entry.entry_id} swing={entry.swing} occurredAt={entry.occurred_at} />;
              }
              const transaction = entry.transaction;
              return (
                <article
                  key={entry.entry_id}
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
              );
            })}
          </div>
          <div className="toolbar trade-history-toolbar treasury-ledger-toolbar">
            <button
              type="button"
              className="dialog-user-btn trade-history-nav-button trade-history-nav-later"
              disabled={loading || !hasLater}
              onClick={() => void loadEntries(Math.max(0, offset - limit))}
            >
              Later
            </button>
            {hasLater ? (
              <button
                type="button"
                className="dialog-user-btn trade-history-latest-button"
                disabled={loading}
                onClick={() => void loadEntries(0)}
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
              onClick={() => void loadEntries(offset + limit)}
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
  realizedAccumulatedCash,
  onPortfolioUpdated,
  onReload,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  realizedAccumulatedCash: number;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse | UpdatePortfolioPolicyResponse) => void;
  onReload: () => Promise<void>;
}): JSX.Element {
  const [expanded, setExpanded] = useState<boolean>(false);
  const executionEnabled = exchange?.spot_execution_enabled === true;
  const tradingState = holding.spot_trading_state ?? (
    executionEnabled && !holding.is_dry_powder && (holding.minimum_holding_pct ?? 100) < 100
      ? "unlocked"
      : "locked"
  );
  const tradingBadge = tradingState === "unlocked" && holding.trading_objective === "accumulate_asset"
    ? {
        icon: "\u{1FA99}",
        tone: "asset",
        title: `Scrooge is using Bargains to accumulate more ${holding.asset_symbol}.`,
      }
    : tradingState === "unlocked" && holding.trading_objective === "accumulate_cash"
      ? {
          icon: "\u{1F4B5}",
          tone: "cash",
          title: "Scrooge is using Bargains to accumulate more USDT.",
        }
      : {
          icon: "\u{1F512}",
          tone: "locked",
          title: !executionEnabled
            ? "Scrooge's automatic Bargains are disabled."
            : holding.is_dry_powder
              ? "Vault Reserve is not managed by an asset trading policy."
              : holding.trading_objective === null
                ? "Choose a Trading Objective before Scrooge can bargain with this asset."
                : "Scrooge keeps the full holding protected by policy.",
        };
  const refreshKey = [
    holding.quantity,
    holding.invested_capital,
    holding.binance_quantity,
    holding.cold_storage_quantity,
    holding.unassigned_quantity,
  ].join(":");

  return (
    <article className={`${holdingToneClass(holding.total_gain)}${expanded ? " treasury-holding-card-expanded" : ""}`}>
      <div className="treasury-holding-row">
        <button
          type="button"
          className="treasury-holding-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <span className="treasury-holding-head">
            <span className="treasury-coin-line">
              <span className="treasury-coin">{holding.asset_symbol}</span>
              <span
                className={`treasury-trading-state treasury-trading-state-${tradingBadge.tone}`}
                title={tradingBadge.title}
                role="img"
                aria-label={tradingBadge.title}
              >
                {tradingBadge.icon}
              </span>
            </span>
            <span className="treasury-share">{formatPercent(holding.allocation_pct)}</span>
          </span>
          <span className="treasury-holding-lines">
            <span>
              <span>Stack</span>
              <strong>{formatAssetQuantity(holding.quantity, holding.asset_symbol)} {holding.asset_symbol}</strong>
            </span>
            <span>
              <span>Target</span>
              <strong>
                {holding.target_quantity === null
                  ? "—"
                  : `${formatNumber(holding.target_quantity, 8)} ${holding.asset_symbol}`}
              </strong>
            </span>
            <span>
              <span>Entry Cost</span>
              <strong>{formatCurrency(
                holding.is_dry_powder ? holding.average_cost : holding.effective_entry_cost,
                6,
              )}</strong>
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
              <span>{holding.is_dry_powder ? "Accumulated Cash" : "Total Gain"}</span>
              {holding.is_dry_powder ? (
                <strong className={signedToneClass(realizedAccumulatedCash, "treasury-inline-value")}>
                  {formatSignedCurrency(realizedAccumulatedCash)}
                </strong>
              ) : (
                <>
                  <strong className={signedToneClass(holding.total_gain, "treasury-inline-value")}>
                    {formatSignedCurrency(holding.total_gain)} · {formatPercent(holding.total_gain_pct)}
                  </strong>
                  <small className="treasury-gain-breakdown">
                    <span>
                      Market {formatSignedCurrency(holding.market_gain)} · Cash {formatSignedCurrency(holding.accumulated_cash_gain)}
                    </span>
                    <span className={signedToneClass(holding.open_bargain_pnl, "treasury-open-pnl")}>
                      Open {formatSignedCurrency(holding.open_bargain_pnl)}
                    </span>
                  </small>
                </>
              )}
            </span>
          </span>
          <span className="treasury-holding-chevron" aria-hidden="true" />
        </button>
      </div>
      {expanded ? (
        <div className="treasury-asset-controls">
          <CustodyPanel
            key={`${holding.binance_quantity}:${exchange?.spot_execution_enabled ? 1 : 0}`}
            holding={holding}
            exchange={exchange}
            onPortfolioUpdated={onPortfolioUpdated}
            onExecuted={onReload}
          />
          {executionEnabled && !holding.is_dry_powder ? (
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
  const [form, setForm] = useState<TreasureIntakeFormState>(EMPTY_INTAKE_FORM);
  const [formExpanded, setFormExpanded] = useState<boolean>(false);
  const [intakeMode, setIntakeMode] = useState<TreasureIntakeMode>("bring_in");
  const [buyAssetSymbol, setBuyAssetSymbol] = useState<string>("");
  const [bargainsExpanded, setBargainsExpanded] = useState<boolean>(false);
  const refreshInFlight = useRef<boolean>(false);

  const loadPortfolio = useCallback(async (background = false): Promise<void> => {
    if (refreshInFlight.current) {
      return;
    }
    refreshInFlight.current = true;
    if (!background) {
      setLoading(true);
    }
    try {
      const payload = await fetchApi<PortfolioPayload>("/api/portfolio");
      setPortfolio(payload);
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Treasury is unavailable.");
    } finally {
      refreshInFlight.current = false;
      if (!background) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void loadPortfolio();
    const refreshTimer = window.setInterval(() => {
      void loadPortfolio(true);
    }, 60_000);
    return () => window.clearInterval(refreshTimer);
  }, [loadPortfolio]);

  async function submitTransaction(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>("/api/portfolio/transactions", {
        method: "POST",
        body: {
          tx_type: "deposit",
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
      setForm(EMPTY_INTAKE_FORM);
      setFormExpanded(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not add treasure.");
    } finally {
      setSaving(false);
    }
  }

  const summary = portfolio?.summary;
  const exchange = portfolio?.exchange ?? null;
  const spotExecutionEnabled = exchange?.spot_execution_enabled === true;
  const activeIntakeMode: TreasureIntakeMode = spotExecutionEnabled ? intakeMode : "bring_in";
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
  const latestTimelinePnl = timeline.at(-1)?.total_gain;
  const timelinePnlTone =
    typeof latestTimelinePnl !== "number" || latestTimelinePnl === 0
      ? "neutral"
      : latestTimelinePnl > 0
        ? "positive"
        : "negative";
  return (
    <AuthGate>
      <section className="panel page-shell treasury-page-shell">
        <p className="dialog-scrooge treasury-mode-banner">
          {portfolio === null
            ? "I am counting the vault before any coin gets a crown."
            : spotExecutionEnabled
              ? "The vault is open for business. I am growing my treasure."
              : "My vault is under lock. I can count every coin, but none leaves my treasury."}
        </p>

        <section className="treasury-overview">
          {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
          {loading ? <p className="status-performance-note">Scrooge counts the vault...</p> : null}

          <div className="treasury-summary-grid">
            <div className="treasury-summary-card treasury-summary-card-hero">
              <span className="treasury-summary-label">Total Treasure</span>
              <strong className="vault-value treasury-total-value">
                <span
                  className={signedToneClass(summary?.total_gain, "treasury-total-dollar")}
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
              <span className="treasury-summary-label">Total Gain</span>
              <strong className={signedToneClass(summary?.total_gain, "treasury-summary-value")}>
                {formatSignedCurrency(summary?.total_gain ?? 0)}
              </strong>
              <span className="treasury-summary-note">{formatPercent(summary?.total_gain_pct)}</span>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Vault Reserve</span>
              <strong>{formatCurrency(summary?.vault_reserve ?? 0)}</strong>
              <span className="treasury-summary-note">
                {formatCurrency(summary?.vault_reserve_available ?? 0)} available
                {(summary?.vault_reserve_committed ?? 0) > 0
                  ? ` · ${formatCurrency(summary?.vault_reserve_committed ?? 0)} committed`
                  : ""}
              </span>
            </div>
            <button
              type="button"
              className={`treasury-summary-card treasury-summary-card-swings${bargainsExpanded ? " treasury-summary-card-swings-expanded" : ""}`}
              aria-expanded={bargainsExpanded}
              aria-controls="treasury-bargain-ledger"
              onClick={() => setBargainsExpanded((current) => !current)}
            >
              <span className="treasury-summary-label">Bargains</span>
              <span className="treasury-bargain-counts">
                <span>
                  <small>Open</small>
                  <strong>{formatNumber(summary?.open_swing_count ?? 0, 0)}</strong>
                </span>
                <span>
                  <small>Closed</small>
                  <strong>{formatNumber(summary?.closed_swing_count ?? 0, 0)}</strong>
                </span>
                <span>
                  <small>Total</small>
                  <strong>{formatNumber(summary?.total_swing_count ?? 0, 0)}</strong>
                </span>
              </span>
              <span className="treasury-summary-card-chevron" aria-hidden="true" />
            </button>
          </div>

          {bargainsExpanded ? (
            <PortfolioBargainLedger
              refreshKey={[
                summary?.open_swing_count ?? 0,
                summary?.closed_swing_count ?? 0,
                summary?.prices_updated_at ?? "",
              ].join(":")}
            />
          ) : null}

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
                <span>Vault Reserve <strong>{formatPercent(summary?.vault_reserve_pct)}</strong></span>
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
                <TimelineSeries title="Total Gain" timeline={timeline} valueKey="total_gain" tone={timelinePnlTone} />
              </div>
              <footer className="treasury-timeline-range">
                {timeline.length ? (
                  <>
                    <span>{formatTimelineDate(timeline[0].snapshot_date)}</span>
                    <span>Daily marks</span>
                    <span>{formatTimelineDate(timeline[timeline.length - 1].snapshot_date)}</span>
                  </>
                ) : (
                  <span>The first daily mark will appear automatically.</span>
                )}
              </footer>
            </article>
          </div>

        </section>

        <section className="section-block">
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
              {formExpanded ? "Close Intake" : "Add Treasure"}
            </button>
          </header>
          {formExpanded ? (
            <div className="treasury-intake-panel">
              {spotExecutionEnabled ? (
                <div className="treasury-intake-modes" aria-label="Add Treasure method">
                  <button
                    type="button"
                    className={activeIntakeMode === "bring_in" ? "treasury-intake-mode-active" : undefined}
                    aria-pressed={activeIntakeMode === "bring_in"}
                    onClick={() => {
                      setIntakeMode("bring_in");
                      setError(null);
                    }}
                  >
                    Bring Into Vault
                  </button>
                  <button
                    type="button"
                    className={activeIntakeMode === "buy_binance" ? "treasury-intake-mode-active" : undefined}
                    aria-pressed={activeIntakeMode === "buy_binance"}
                    onClick={() => {
                      setIntakeMode("buy_binance");
                      setError(null);
                    }}
                  >
                    Buy on Binance
                  </button>
                </div>
              ) : null}

              {activeIntakeMode === "bring_in" ? (
                <form className="treasury-form" onSubmit={(event) => void submitTransaction(event)}>
                  <p className="treasury-intake-copy">
                    Bring a new treasure under Scrooge&apos;s care and place it in its first custody location.
                  </p>
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
                      step="any"
                      onChange={(event) => setForm((current) => ({ ...current, quantity: event.target.value }))}
                      required
                    />
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
                      required={!STABLE_ASSETS.has(form.asset_symbol)}
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
                </form>
              ) : (
                <section className="treasury-binance-intake">
                  <p className="treasury-intake-copy">
                    Buy a new treasure on Binance. It enters the vault only after Binance confirms the fill.
                  </p>
                  <label className="dialog-user-field treasury-binance-intake-coin">
                    Coin
                    <input
                      type="text"
                      value={buyAssetSymbol}
                      placeholder="BTC"
                      autoCapitalize="characters"
                      onChange={(event) => setBuyAssetSymbol(event.target.value.toUpperCase())}
                      required
                    />
                  </label>
                  <SpotOrderPanel
                    key={buyAssetSymbol}
                    assetSymbol={buyAssetSymbol}
                    quoteSymbol="USDT"
                    treasuryIntake
                    exchange={exchange}
                    onExecuted={async () => {
                      setBuyAssetSymbol("");
                      setFormExpanded(false);
                      setIntakeMode("bring_in");
                      await loadPortfolio();
                    }}
                  />
                </section>
              )}
            </div>
          ) : null}

          {holdings.length === 0 ? (
            <p className="trade-history-empty-sheet">The treasury is empty. Add your first treasure.</p>
          ) : (
            <div className="treasury-holdings-grid">
              {holdings.map((holding) => (
                <HoldingCard
                  key={`${holding.asset_symbol}-${holding.quote_symbol}`}
                  holding={holding}
                  exchange={exchange}
                  realizedAccumulatedCash={summary?.realized_accumulated_cash ?? 0}
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
