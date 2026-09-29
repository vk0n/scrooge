"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type PortfolioSummary = {
  total_value: number;
  total_value_24h_change: number | null;
  total_value_24h_change_pct: number | null;
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
  vault_reserve_retained: number;
  vault_reserve_retained_accrued: number;
  vault_reserve_spendable: number;
  free_cash_retention_pct: number;
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
  treasury_transfer_enabled: boolean;
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
  request?: {
    use_protected_cash?: boolean;
    estimated_protected_cash_required?: number;
  };
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

type TreasuryRules = {
  signal_refresh_seconds: number;
  signal: {
    levels_pct: number[];
    base_tranches_pct: number[];
    accumulation_tranches_pct: number[];
  };
  progression: {
    close_profit_pct: number;
    campaign_capacity_pct: number;
    full_deploy_threshold_pct: number;
  };
  waiter_cleanup: {
    enabled: boolean;
    max_open_bargains_per_asset: number;
    deep_loss: {
      min_age_days: number;
      unrealized_pnl_pct: number;
      required_reverse_level: number;
    };
    aging: Array<{ min_age_days: number; required_reverse_level: number }>;
    capacity_cleanup: { enabled: boolean; min_age_days: number };
  };
};

type TreasuryRulesResponse = {
  rules: TreasuryRules;
  raw_text: string;
  path: string;
};

type SaveTreasuryRulesResponse = TreasuryRulesResponse & {
  updated: boolean;
  restart_required: boolean;
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
  rolling_24h_change_pct: number | null;
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
  spot_quote_leg?: boolean;
  swing_id?: string | null;
  office_transfer_direction?: "to_office" | "from_office" | null;
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
      free_cash_delta?: number;
      transaction: PortfolioTransaction;
    }
  | {
      entry_type: "cash";
      entry_id: string;
      occurred_at_ms: number;
      occurred_at: string;
      cash_event: {
        event_type: "bargain_settlement";
        amount_quote: number;
        asset_symbol: string;
        quote_symbol: string;
        swing_id: string;
        label: string;
      };
    }
  | {
      entry_type: "swing";
      entry_id: string;
      occurred_at_ms: number;
      occurred_at: string;
      swing: SpotSwing;
    };

type AssetLedgerFilter = "all" | "open" | "closed";
type BargainLedgerSort = "date" | "pnl";
type BargainLedgerDirection = "asc" | "desc";

type AssetLedgerPayload = {
  asset_symbol: string;
  quote_symbol: string;
  filter: AssetLedgerFilter;
  sort: BargainLedgerSort;
  direction: BargainLedgerDirection;
  entries: AssetLedgerEntry[];
  entry_count: number;
  entry_limit: number;
  entry_offset: number;
};

type BargainLedgerPayload = {
  filter: AssetLedgerFilter;
  sort: BargainLedgerSort;
  direction: BargainLedgerDirection;
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

type UpdateCashPolicyResponse = {
  policy: {
    account_key: string;
    free_cash_retention_pct: number;
    retained_quote_balance: number;
  };
  portfolio: Omit<PortfolioPayload, "warnings">;
  warnings: string[];
};

type TreasuryTransferQueueResponse = {
  command_id: string;
  status: string;
  action: string;
};

type PortfolioTransactionType = "buy" | "sell" | "deposit" | "withdraw" | "adjustment" | "custody_transfer";
type CustodyLocation = "unassigned" | "binance" | "cold_storage";
type CustodyAction = "move" | "bring_in" | "release";
type TreasureIntakeMode = "bring_in" | "buy_binance";
type RetainedTransferMenu = "office" | "spendable";
type RetainedCashDirection = "to_retained" | "to_spendable";

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
const TREASURY_REFRESH_MS = 60_000;
const INITIAL_TREASURY_RETRY_DELAYS_MS = [1_000, 3_000, 7_000] as const;

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

function waitForRetry(delayMs: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, delayMs));
}

function isRetryablePortfolioLoad(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return true;
  }
  const statusMatch = /^API\s+(\d{3}):/.exec(error.message);
  if (!statusMatch) {
    return true;
  }
  const status = Number(statusMatch[1]);
  return status === 408 || status === 429 || status >= 500;
}

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

function formatSignedPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Pending";
  }
  return `${value > 0 ? "+" : ""}${formatNumber(value, 2)}%`;
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
  summary,
  onPortfolioUpdated,
  onExecuted,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  summary: PortfolioSummary;
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
  const [officeDirection, setOfficeDirection] = useState<"to_office" | "from_office">("to_office");
  const [officeQuantity, setOfficeQuantity] = useState<string>("");
  const [officeBusy, setOfficeBusy] = useState<boolean>(false);
  const [officeStage, setOfficeStage] = useState<string | null>(null);
  const [officeUseProtected, setOfficeUseProtected] = useState<boolean>(false);
  const available = custodyQuantity(holding, source);
  const releasable = custodyQuantity(holding, boundaryCustody);
  const tradeAvailable = Boolean(
    !holding.is_dry_powder &&
    exchange?.spot_execution_enabled &&
    holding.binance_quantity > 0.00000001
  );
  const officeTransferAvailable = holding.is_dry_powder && holding.asset_symbol === "USDT";

  async function submitOfficeTransfer(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const amount = asNumber(officeQuantity);
    if (amount === null || amount <= 0) return;
    const directionLabel = officeDirection === "to_office" ? "Treasury to Futures Office" : "Futures Office to Treasury";
    if (!window.confirm(
      `Transfer ${formatCurrency(amount)} USDT from ${directionLabel}?\n\nThis is a REAL Binance account transfer and changes Treasury invested capital.`
    )) return;
    setOfficeBusy(true);
    setError(null);
    setOfficeStage("Transfer queued. I am revalidating both accounts...");
    try {
      const queued = await fetchApi<TreasuryTransferQueueResponse>("/api/portfolio/office-transfers", {
        method: "POST",
        body: {
          direction: officeDirection,
          quantity: amount,
          confirmation: "CONFIRM_TREASURY_TRANSFER",
          use_protected_cash: officeDirection === "to_office" && officeUseProtected,
        },
      });
      const command = await waitForControlCommand(queued.command_id);
      if (command.status !== "completed") {
        throw new Error(command.message || "Treasury Office transfer failed.");
      }
      setOfficeStage(command.message || "Transfer confirmed. Treasury accounting updated.");
      setOfficeQuantity("");
      await onExecuted();
    } catch (transferError) {
      setError(transferError instanceof Error ? transferError.message : "Could not complete the Office transfer.");
      setOfficeStage("Transfer did not complete cleanly. Check the Treasury Ledger before retrying.");
    } finally {
      setOfficeBusy(false);
    }
  }

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
            summary={summary}
            onExecuted={onExecuted}
          />
        ) : null}
        {officeTransferAvailable ? (
          <section className="treasury-office-transfer" aria-label="Treasury and Futures Office transfer">
            <header>
              <span>Futures Office</span>
              <strong>Real Binance Transfer</strong>
            </header>
            <form className="treasury-custody-form" onSubmit={(event) => void submitOfficeTransfer(event)}>
              <label className="dialog-user-field">
                Direction
                <select
                  value={officeDirection}
                  onChange={(event) => {
                    setOfficeDirection(event.target.value as "to_office" | "from_office");
                    setOfficeUseProtected(false);
                    setOfficeStage(null);
                  }}
                  disabled={officeBusy}
                >
                  <option value="to_office">Treasury → Office</option>
                  <option value="from_office">Office → Treasury</option>
                </select>
              </label>
              <label className="dialog-user-field">
                Amount (USDT)
                <input
                  type="number"
                  min="0.01"
                  max={officeDirection === "to_office"
                    ? officeUseProtected
                      ? summary.vault_reserve_available
                      : summary.vault_reserve_spendable
                    : undefined}
                  step="0.01"
                  value={officeQuantity}
                  placeholder={officeDirection === "to_office"
                    ? formatNumber(summary.vault_reserve_available, 2)
                    : "0.00"}
                  onChange={(event) => setOfficeQuantity(event.target.value)}
                  disabled={officeBusy}
                  required
                />
              </label>
              {officeDirection === "to_office" && summary.vault_reserve_retained > 0 ? (
                <label className="treasury-protected-cash-option">
                  <input
                    type="checkbox"
                    checked={officeUseProtected}
                    onChange={(event) => setOfficeUseProtected(event.target.checked)}
                    disabled={officeBusy}
                  />
                  <span>Allow up to {formatCurrency(summary.vault_reserve_retained)} Protected Cash</span>
                </label>
              ) : null}
              <button
                type="submit"
                className="dialog-user-btn"
                disabled={officeBusy || !exchange?.treasury_transfer_enabled}
              >
                {officeBusy ? "Transferring..." : "Transfer USDT"}
              </button>
            </form>
            <p className="treasury-custody-disclaimer">
              {exchange?.treasury_transfer_enabled
                ? `Spendable ${formatCurrency(summary.vault_reserve_spendable)} · Protected ${formatCurrency(summary.vault_reserve_retained)}. Protected cash requires explicit authorization.`
                : "Real Office transfers are locked by the Treasury transfer safety switch."}
            </p>
            {officeStage ? <p className="treasury-spot-order-stage">{officeStage}</p> : null}
          </section>
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

async function waitForControlCommand(commandId: string): Promise<ControlCommandStatus> {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    const command = await fetchApi<ControlCommandStatus>(`/api/control/commands/${encodeURIComponent(commandId)}`);
    if (command.status === "completed" || command.status === "failed") {
      return command;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
  throw new Error("The command is still awaiting confirmation. Check the Treasury Ledger before retrying.");
}

function TreasuryRulesPanel(): JSX.Element {
  const [rules, setRules] = useState<TreasuryRules | null>(null);
  const [rawText, setRawText] = useState<string>("");
  const [draft, setDraft] = useState<string>("");
  const [expanded, setExpanded] = useState<boolean>(false);
  const [editing, setEditing] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function loadRules(): Promise<void> {
      try {
        const payload = await fetchApi<TreasuryRulesResponse>("/api/config/treasury-rules");
        if (!active) return;
        setRules(payload.rules);
        setRawText(payload.raw_text);
        setDraft(payload.raw_text);
      } catch (loadError) {
        if (active) setError(loadError instanceof Error ? loadError.message : "Could not read Treasury Rules.");
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadRules();
    return () => { active = false; };
  }, []);

  async function saveRules(): Promise<void> {
    if (!window.confirm("Seal these Treasury Rules and reopen for business with them?")) return;
    setSaving(true);
    setError(null);
    setInfo(null);
    try {
      const saved = await fetchApi<SaveTreasuryRulesResponse>("/api/config/treasury-rules", {
        method: "POST",
        body: { raw_text: draft },
      });
      setRules(saved.rules);
      setRawText(saved.raw_text);
      setDraft(saved.raw_text);
      setEditing(false);
      if (!saved.updated) {
        setInfo("The rules were already identical. No restart was needed.");
        return;
      }
      setInfo("Rules sealed. I am reopening the Treasury...");
      const queued = await fetchApi<TreasuryTransferQueueResponse>("/api/control/restart", { method: "POST" });
      const command = await waitForControlCommand(queued.command_id);
      if (command.status !== "completed") throw new Error(command.message || "I could not apply the new rules.");
      setInfo("Treasury Rules are active.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not update Treasury Rules.");
    } finally {
      setSaving(false);
    }
  }

  const levelText = rules?.signal.levels_pct.map((value) => `${value}%`).join(" / ") ?? "...";
  const sellText = rules?.signal.base_tranches_pct.map((value) => `${value}%`).join(" / ") ?? "...";
  const buyText = rules?.signal.accumulation_tranches_pct.map((value) => `${value}%`).join(" / ") ?? "...";
  const agingText = rules?.waiter_cleanup.aging
    .map((rule) => `${rule.min_age_days}d needs L${rule.required_reverse_level}`)
    .join(", ") ?? "...";

  return (
    <section className="section-block treasury-rules-section">
      <div className="treasury-rules-badge-row">
        <span className="treasury-insight-count">LIVE CONFIG</span>
      </div>
      {loading ? <p className="dialog-scrooge">Reviewing the rules...</p> : editing ? (
        <div className="contract-editor">
          <p className="contract-editor-note">Only the Treasury strategy section is editable here. The Futures contract remains untouched.</p>
          <label className="dialog-user-field contract-editor-field">
            <span className="kv-label">Technical Rules</span>
            <textarea value={draft} onChange={(event) => setDraft(event.target.value)} disabled={saving} spellCheck={false} />
          </label>
        </div>
      ) : rules ? (
        <div className={`contract-scroll${expanded ? " contract-scroll-open" : ""}`}>
          <button type="button" className="contract-scroll-toggle" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded} aria-controls="treasury-rules-body">
            <span className="contract-scroll-toggle-copy">
              <span className="contract-scroll-toggle-label">{expanded ? "Roll the parchment back up" : "Unroll the parchment"}</span>
              <span className="contract-scroll-toggle-teaser">Signals at {levelText}; Bargain Goal {rules.progression.close_profit_pct}%.</span>
            </span>
            <span className="contract-scroll-toggle-icon" aria-hidden="true">▾</span>
          </button>
          <div className="contract-scroll-body-shell" id="treasury-rules-body">
            <div className="contract-scroll-body">
              <div className="contract-sheet" aria-label="Treasury Rules">
                <p><span className="contract-term">Market Bell</span> Every <span className="contract-value">{rules.signal_refresh_seconds}s</span>, measure the 24-hour move. Levels are <span className="contract-value">{levelText}</span>.</p>
                <p><span className="contract-term">Campaign Stakes</span> Cash-accumulation tranches are <span className="contract-value">{sellText}</span>; asset-accumulation tranches are <span className="contract-value">{buyText}</span>.</p>
                <p><span className="contract-term">Bargain Goal</span> Close profitable Bargains at gross <span className="contract-value">{rules.progression.close_profit_pct}%</span>.</p>
                <p><span className="contract-term">Inventory Discipline</span> A campaign may use <span className="contract-value">{rules.progression.campaign_capacity_pct}%</span> of sellable inventory, then deploy fully below <span className="contract-value">{rules.progression.full_deploy_threshold_pct}%</span> remaining.</p>
                <p><span className="contract-term">Waiter Cleanup</span> Cleanup is <span className="contract-value">{rules.waiter_cleanup.enabled ? "active" : "paused"}</span>. Keep at most <span className="contract-value">{rules.waiter_cleanup.max_open_bargains_per_asset}</span> open Bargains per asset. Aging gates: <span className="contract-value">{agingText}</span>.</p>
                <p><span className="contract-term">Deep Loss</span> After <span className="contract-value">{rules.waiter_cleanup.deep_loss.min_age_days}d</span> at or below <span className="contract-value">{rules.waiter_cleanup.deep_loss.unrealized_pnl_pct}%</span>, an L<span className="contract-value">{rules.waiter_cleanup.deep_loss.required_reverse_level}</span> reverse signal may clean the position.</p>
                <p><span className="contract-term">Capacity Relief</span> It is <span className="contract-value">{rules.waiter_cleanup.capacity_cleanup.enabled ? "active" : "paused"}</span> after <span className="contract-value">{rules.waiter_cleanup.capacity_cleanup.min_age_days}d</span>.</p>
              </div>
            </div>
          </div>
        </div>
      ) : null}
      <div className={`toolbar config-toolbar${!editing ? " config-toolbar-centered" : ""}`}>
        {editing ? <>
          <button type="button" className="dialog-user-btn config-action-btn" onClick={() => void saveRules()} disabled={saving}>Seal Rules and Restart</button>
          <button type="button" className="dialog-user-btn config-action-btn" onClick={() => { setDraft(rawText); setEditing(false); setError(null); }} disabled={saving}>Cancel Revision</button>
        </> : <button type="button" className="dialog-user-btn config-action-btn config-action-btn-centered" onClick={() => { setDraft(rawText); setEditing(true); setExpanded(true); setError(null); setInfo(null); }} disabled={loading || saving || !rules}>Revise Rules</button>}
        {saving ? <span className="dialog-scrooge dialog-scrooge-compact">Sealing rules...</span> : null}
      </div>
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
      {info ? <p className="dialog-scrooge">{info}</p> : null}
    </section>
  );
}

function SpotOrderPanel({
  holding,
  assetSymbol,
  quoteSymbol = "USDT",
  treasuryIntake = false,
  exchange,
  summary,
  onExecuted,
}: {
  holding?: PortfolioHolding;
  assetSymbol?: string;
  quoteSymbol?: string;
  treasuryIntake?: boolean;
  exchange: PortfolioExchange | null;
  summary: PortfolioSummary;
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
  const [useProtectedCash, setUseProtectedCash] = useState<boolean>(false);
  const executionReady = Boolean(exchange?.spot_execution_enabled && exchange?.is_balance_verified && exchange?.can_trade);

  function resetPreview(nextSide?: "buy" | "sell"): void {
    if (nextSide) {
      setSide(nextSide);
    }
    setPreview(null);
    setStage(null);
    setError(null);
    if (nextSide === "sell") setUseProtectedCash(false);
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
          use_protected_cash: side === "buy" && useProtectedCash,
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
    setStage("Order accepted by the Control Plane. I am preparing it...");
    try {
      const queued = await fetchApi<SpotOrderQueueResponse>(
        `/api/portfolio/spot-orders/${encodeURIComponent(preview.intent_id)}/execute`,
        { method: "POST", body: { confirmation: "CONFIRM_SPOT_ORDER" } }
      );
      setStage("I am validating fresh balances and submitting the order...");
      const command = await waitForControlCommand(queued.command_id);
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
          {side === "buy" && summary.vault_reserve_retained > 0 ? (
            <label className="treasury-protected-cash-option treasury-spot-protected-option">
              <input
                type="checkbox"
                checked={useProtectedCash}
                onChange={(event) => {
                  setUseProtectedCash(event.target.checked);
                  setPreview(null);
                  setStage(null);
                }}
                disabled={busy}
              />
              <span>Allow Protected Cash ({formatCurrency(summary.vault_reserve_retained)})</span>
            </label>
          ) : null}
          <button type="submit" className="dialog-user-btn" disabled={busy || !executionReady || !resolvedAsset}>
            {busy ? "Checking..." : treasuryIntake ? "Preview Real Buy" : "Preview Real Order"}
          </button>
        </form>
        <div className="treasury-spot-order-capacity">
          <span>Available USDT <strong>{formatCurrency(exchange?.usdt_free)}</strong></span>
          <span>Spendable Reserve <strong>{formatCurrency(summary.vault_reserve_spendable)}</strong></span>
          <span>Protected <strong>{formatCurrency(summary.vault_reserve_retained)}</strong></span>
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
              {(preview.request?.estimated_protected_cash_required ?? 0) > 0 ? (
                <span>Protected Cash <strong>{formatCurrency(preview.request?.estimated_protected_cash_required)}</strong></span>
              ) : null}
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

function CashPolicyPanel({
  summary,
  exchange,
  onUpdated,
  onExecuted,
}: {
  summary: PortfolioSummary;
  exchange: PortfolioExchange | null;
  onUpdated: (response: UpdateCashPolicyResponse) => void;
  onExecuted: () => Promise<void>;
}): JSX.Element {
  const [retentionPct, setRetentionPct] = useState<string>(String(summary.free_cash_retention_pct));
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [transferMenu, setTransferMenu] = useState<RetainedTransferMenu | null>(null);
  const [cashDirection, setCashDirection] = useState<RetainedCashDirection>("to_retained");
  const [cashQuantity, setCashQuantity] = useState<string>("");
  const [officeDirection, setOfficeDirection] = useState<"to_office" | "from_office">("to_office");
  const [officeQuantity, setOfficeQuantity] = useState<string>("");
  const [officeBusy, setOfficeBusy] = useState<boolean>(false);
  const [officeStage, setOfficeStage] = useState<string | null>(null);

  useEffect(() => {
    setRetentionPct(String(summary.free_cash_retention_pct));
  }, [summary.free_cash_retention_pct]);

  async function submitPolicy(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<UpdateCashPolicyResponse>("/api/portfolio/cash-policy", {
        method: "POST",
        body: { free_cash_retention_pct: asNumber(retentionPct) },
      });
      onUpdated(response);
    } catch (policyError) {
      setError(policyError instanceof Error ? policyError.message : "Could not update the cash policy.");
    } finally {
      setSaving(false);
    }
  }

  function toggleTransferMenu(menu: RetainedTransferMenu): void {
    setTransferMenu((current) => current === menu ? null : menu);
    setCashQuantity("");
    setOfficeQuantity("");
    setOfficeStage(null);
    setError(null);
  }

  async function submitCashTransfer(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const quantity = asNumber(cashQuantity);
    if (quantity === null || quantity <= 0) return;
    const from = cashDirection === "to_retained" ? "Spendable Cash" : "Retained Cash";
    const to = cashDirection === "to_retained" ? "Retained Cash" : "Spendable Cash";
    if (!window.confirm(
      `Move ${formatCurrency(quantity)} from ${from} to ${to}?`
    )) return;
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<UpdateCashPolicyResponse>("/api/portfolio/cash-policy/transfer", {
        method: "POST",
        body: { direction: cashDirection, quantity },
      });
      setCashQuantity("");
      onUpdated(response);
    } catch (transferError) {
      setError(transferError instanceof Error ? transferError.message : "Could not move the cash reserve.");
    } finally {
      setSaving(false);
    }
  }

  async function submitOfficeTransfer(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const quantity = asNumber(officeQuantity);
    if (quantity === null || quantity <= 0) return;
    const directionLabel = officeDirection === "to_office" ? "Retained Cash to Futures Office" : "Futures Office to Retained Cash";
    if (!window.confirm(
      `Transfer ${formatCurrency(quantity)} USDT from ${directionLabel}?\n\nThis is a REAL Binance account transfer and changes Treasury invested capital.`
    )) return;
    setOfficeBusy(true);
    setError(null);
    setOfficeStage("Transfer queued. I am revalidating both accounts...");
    try {
      const queued = await fetchApi<TreasuryTransferQueueResponse>("/api/portfolio/office-transfers", {
        method: "POST",
        body: {
          direction: officeDirection,
          quantity,
          confirmation: "CONFIRM_TREASURY_TRANSFER",
          cash_bucket: "retained",
        },
      });
      const command = await waitForControlCommand(queued.command_id);
      if (command.status !== "completed") {
        throw new Error(command.message || "Protected Cash Office transfer failed.");
      }
      setOfficeStage(command.message || "Transfer confirmed. Protected Cash updated.");
      setOfficeQuantity("");
      await onExecuted();
    } catch (transferError) {
      setError(transferError instanceof Error ? transferError.message : "Could not complete the Office transfer.");
      setOfficeStage("Transfer did not complete cleanly. Check the Treasury Ledger before retrying.");
    } finally {
      setOfficeBusy(false);
    }
  }

  return (
    <section className="treasury-policy-panel treasury-cash-policy-panel">
      <header className="treasury-asset-panel-head">
        <button
          type="button"
          className="treasury-asset-panel-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <span>Policy</span>
          <span className="treasury-policy-summary">
            Retain {formatPercent(summary.free_cash_retention_pct)}
            <span>Protected {formatCurrency(summary.vault_reserve_retained)}</span>
            <strong className="value-positive">Spendable {formatCurrency(summary.vault_reserve_spendable)}</strong>
          </span>
          <span className="treasury-asset-panel-chevron" aria-hidden="true" />
        </button>
      </header>
      {expanded ? <div className="treasury-policy-content">
        <div className="treasury-policy-metrics treasury-cash-policy-metrics">
          <div><span>Free Cash</span><strong>{formatCurrency(summary.vault_reserve_available)}</strong></div>
          <div className="treasury-retained-metric">
            <span>Retained</span>
            <span className="treasury-retained-metric-value">
              <strong>{formatCurrency(summary.vault_reserve_retained)}</strong>
              <span className="treasury-retained-actions">
                <button
                  type="button"
                  aria-label="Transfer between Retained Cash and Futures Office"
                  aria-expanded={transferMenu === "office"}
                  title="Retained Cash / Futures Office"
                  onClick={() => toggleTransferMenu("office")}
                >
                  ⚖️ / 🏦
                </button>
                <button
                  type="button"
                  aria-label="Transfer between Retained Cash and Spendable Cash"
                  aria-expanded={transferMenu === "spendable"}
                  title="Lock / unlock cash"
                  onClick={() => toggleTransferMenu("spendable")}
                >
                  🔒 / 🔓
                </button>
              </span>
            </span>
          </div>
          <div><span>Spendable</span><strong>{formatCurrency(summary.vault_reserve_spendable)}</strong></div>
          <div><span>Committed</span><strong>{formatCurrency(summary.vault_reserve_committed)}</strong></div>
        </div>
        {transferMenu === "office" ? (
          <section className="treasury-retained-transfer-panel treasury-retained-office-panel">
            <header>
              <span>Retained Cash / Futures Office</span>
              <strong>Real Binance Transfer</strong>
            </header>
            <form className="treasury-retained-transfer-form" onSubmit={(event) => void submitOfficeTransfer(event)}>
              <label className="dialog-user-field">
                Direction
                <select
                  value={officeDirection}
                  onChange={(event) => {
                    setOfficeDirection(event.target.value as "to_office" | "from_office");
                    setOfficeQuantity("");
                    setOfficeStage(null);
                  }}
                  disabled={officeBusy}
                >
                  <option value="to_office">Retained → Office</option>
                  <option value="from_office">Office → Retained</option>
                </select>
              </label>
              <label className="dialog-user-field">
                Amount (USDT)
                <input
                  type="number"
                  min="0.01"
                  max={officeDirection === "to_office" ? summary.vault_reserve_retained : undefined}
                  step="0.01"
                  value={officeQuantity}
                  placeholder={officeDirection === "to_office" ? formatNumber(summary.vault_reserve_retained, 2) : "0.00"}
                  onChange={(event) => setOfficeQuantity(event.target.value)}
                  disabled={officeBusy}
                  required
                />
              </label>
              <button
                type="submit"
                className="dialog-user-btn"
                disabled={officeBusy || !exchange?.treasury_transfer_enabled}
              >
                {officeBusy ? "Transferring..." : "Transfer USDT"}
              </button>
            </form>
            <p className="treasury-policy-note">
              {exchange?.treasury_transfer_enabled
                ? "Moves real USDT between Binance Spot Treasury and the Futures Office."
                : "Real Office transfers are locked by the Treasury transfer safety switch."}
            </p>
            {officeStage ? <p className="treasury-spot-order-stage">{officeStage}</p> : null}
          </section>
        ) : null}
        {transferMenu === "spendable" ? (
          <section className="treasury-retained-transfer-panel">
            <header><span>Retained / Spendable Cash</span></header>
            <form className="treasury-retained-transfer-form" onSubmit={(event) => void submitCashTransfer(event)}>
              <label className="dialog-user-field">
                Direction
                <select
                  value={cashDirection}
                  onChange={(event) => {
                    setCashDirection(event.target.value as RetainedCashDirection);
                    setCashQuantity("");
                  }}
                  disabled={saving}
                >
                  <option value="to_retained">Spendable → Retained</option>
                  <option value="to_spendable">Retained → Spendable</option>
                </select>
              </label>
              <label className="dialog-user-field">
                Amount (USDT)
                <input
                  type="number"
                  min="0.01"
                  max={cashDirection === "to_retained" ? summary.vault_reserve_spendable : summary.vault_reserve_retained}
                  step="0.01"
                  value={cashQuantity}
                  placeholder={formatNumber(
                    cashDirection === "to_retained" ? summary.vault_reserve_spendable : summary.vault_reserve_retained,
                    2
                  )}
                  onChange={(event) => setCashQuantity(event.target.value)}
                  disabled={saving}
                  required
                />
              </label>
              <button type="submit" className="dialog-user-btn" disabled={saving}>
                {saving ? "Moving..." : "Move Cash"}
              </button>
            </form>
          </section>
        ) : null}
        <form className="treasury-policy-form treasury-cash-policy-form" onSubmit={(event) => void submitPolicy(event)}>
          <label className="dialog-user-field">
            Free Cash Retention %
            <input
              type="number"
              min="0"
              max="100"
              step="any"
              value={retentionPct}
              onChange={(event) => setRetentionPct(event.target.value)}
              required
            />
          </label>
          <button type="submit" className="dialog-user-btn" disabled={saving}>
            {saving ? "Saving..." : "Update Policy"}
          </button>
        </form>
        <p className="treasury-policy-note">
          This percentage of every realized cash profit is added to protected reserve. Automatic actions cannot spend
          it. The Retained controls move cash to or from Spendable and the Futures Office.
        </p>
        {error ? <p className="form-error">{error}</p> : null}
      </div> : null}
    </section>
  );
}

function SwingLedgerRow({
  swing,
  occurredAt,
  summary,
  onBargainClosed,
}: {
  swing: SpotSwing;
  occurredAt: string;
  summary: PortfolioSummary;
  onBargainClosed: () => Promise<void>;
}): JSX.Element {
  const [expanded, setExpanded] = useState<boolean>(false);
  const [closing, setClosing] = useState<boolean>(false);
  const [closeStatus, setCloseStatus] = useState<string | null>(null);
  const [closeError, setCloseError] = useState<string | null>(null);
  const [useProtectedCash, setUseProtectedCash] = useState<boolean>(false);
  const economics = swing.economics;
  const pnl = economics.status === "closed"
    ? economics.realized_pnl_quote
    : economics.unrealized_pnl_quote;
  const quantity = economics.opening_quantity || swing.planned_quantity;
  const fees = Object.entries(economics.fees_by_asset);
  const reason = swingReasonText(swing.strategy_reason);
  const canClose = economics.status !== "closed" && economics.opening_quantity > 0 && economics.remaining_quantity > 0;
  const canAuthorizeProtectedClose = canClose
    && swing.origin_side === "sell"
    && (economics.unrealized_pnl_quote ?? 0) < 0
    && summary.vault_reserve_retained > 0;

  async function closeBargain(): Promise<void> {
    setClosing(true);
    setCloseError(null);
    setCloseStatus("I am pricing the closing order...");
    try {
      const preview = await fetchApi<SpotOrderIntent>(
        `/api/portfolio/bargains/${encodeURIComponent(swing.swing_id)}/close-preview`,
        { method: "POST", body: { use_protected_cash: useProtectedCash } }
      );
      const confirmed = window.confirm(
        `Close ${swingIdentity(swing.swing_id)} with a REAL Binance Spot ${preview.side.toUpperCase()} for ` +
        `${formatNumber(preview.requested_quantity, 8)} ${preview.asset_symbol}?\n\n` +
        `Estimated value: ${formatCurrency(preview.estimated_quote_value)}\n` +
        `Current open PnL: ${formatSignedCurrency(economics.unrealized_pnl_quote)}\n` +
        ((preview.request?.estimated_protected_cash_required ?? 0) > 0
          ? `Protected Cash required: ${formatCurrency(preview.request?.estimated_protected_cash_required)}\n`
          : "") +
        "This action may execute immediately and cannot be undone."
      );
      if (!confirmed) {
        setCloseStatus(null);
        return;
      }
      setCloseStatus("Closing order accepted. I am validating fresh balances...");
      const queued = await fetchApi<SpotOrderQueueResponse>(
        `/api/portfolio/spot-orders/${encodeURIComponent(preview.intent_id)}/execute`,
        { method: "POST", body: { confirmation: "CONFIRM_SPOT_ORDER" } }
      );
      const command = await waitForControlCommand(queued.command_id);
      if (command.status !== "completed") {
        throw new Error(command.message || "Bargain closing order failed.");
      }
      setCloseStatus(command.message || "Bargain closed. Treasury Ledger updated.");
      await onBargainClosed();
    } catch (error) {
      setCloseStatus(null);
      setCloseError(error instanceof Error ? error.message : "Could not close this Bargain.");
    } finally {
      setClosing(false);
    }
  }

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
          {reason ? <p className="treasury-swing-reason">My note: {reason}</p> : null}
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
          {canClose ? (
            <div className="treasury-swing-close">
              <div aria-live="polite">
                {canAuthorizeProtectedClose ? (
                  <label className="treasury-protected-cash-option treasury-close-protected-option">
                    <input
                      type="checkbox"
                      checked={useProtectedCash}
                      onChange={(event) => setUseProtectedCash(event.target.checked)}
                      disabled={closing}
                    />
                    <span>
                      Use Protected Cash if committed + spendable cash is insufficient
                      ({formatCurrency(summary.vault_reserve_retained)} available)
                    </span>
                  </label>
                ) : null}
                {closeStatus ? <p>{closeStatus}</p> : null}
                {closeError ? <p className="form-error">{closeError}</p> : null}
              </div>
              <button
                type="button"
                className={signedToneClass(pnl, "dialog-user-btn treasury-swing-close-button")}
                disabled={closing}
                onClick={() => void closeBargain()}
              >
                {closing ? "Closing..." : "Close Bargain"}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function LedgerSortControls({
  sortBy,
  sortDirection,
  onSortChange,
  onDirectionToggle,
  ariaLabel,
}: {
  sortBy: BargainLedgerSort;
  sortDirection: BargainLedgerDirection;
  onSortChange: (sort: BargainLedgerSort) => void;
  onDirectionToggle: () => void;
  ariaLabel: string;
}): JSX.Element {
  return (
    <div className="treasury-bargain-sort">
      <span>Sort by</span>
      <div className="treasury-bargain-sort-options" aria-label={`${ariaLabel} sort field`}>
        {(["date", "pnl"] as const).map((value) => (
          <button
            key={value}
            type="button"
            className={sortBy === value ? "treasury-bargain-sort-active" : undefined}
            aria-pressed={sortBy === value}
            onClick={() => onSortChange(value)}
          >
            {value === "pnl" ? "PnL" : "Date"}
          </button>
        ))}
      </div>
      <button
        type="button"
        className="treasury-bargain-sort-direction"
        aria-label={`Sort ${sortDirection === "desc" ? "ascending" : "descending"}`}
        onClick={onDirectionToggle}
      >
        {sortDirection === "desc" ? "Desc" : "Asc"}
        <span
          className={`treasury-bargain-sort-arrow treasury-bargain-sort-arrow-${sortDirection}`}
          aria-hidden="true"
        />
      </button>
    </div>
  );
}

function PortfolioBargainLedger({
  refreshKey,
  summary,
  onBargainClosed,
}: {
  refreshKey: string;
  summary: PortfolioSummary;
  onBargainClosed: () => Promise<void>;
}): JSX.Element {
  const [ledger, setLedger] = useState<BargainLedgerPayload | null>(null);
  const [filter, setFilter] = useState<AssetLedgerFilter>("all");
  const [sortBy, setSortBy] = useState<BargainLedgerSort>("date");
  const [sortDirection, setSortDirection] = useState<BargainLedgerDirection>("desc");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadEntries = useCallback(async (offset: number): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchApi<BargainLedgerPayload>(
        `/api/portfolio/bargains?filter=${encodeURIComponent(filter)}&sort=${encodeURIComponent(sortBy)}&direction=${encodeURIComponent(sortDirection)}&entry_offset=${offset}`
      );
      setLedger(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the Bargains Ledger.");
    } finally {
      setLoading(false);
    }
  }, [filter, sortBy, sortDirection]);

  useEffect(() => {
    void loadEntries(0);
  }, [loadEntries, refreshKey]);

  function changeFilter(nextFilter: AssetLedgerFilter): void {
    if (nextFilter === filter) return;
    setLoading(true);
    setFilter(nextFilter);
    setLedger(null);
  }

  function changeSort(nextSort: BargainLedgerSort): void {
    if (nextSort === sortBy) return;
    setLoading(true);
    setSortBy(nextSort);
    setLedger(null);
  }

  function toggleSortDirection(): void {
    setLoading(true);
    setSortDirection((current) => current === "desc" ? "asc" : "desc");
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
    <section id="treasury-bargain-ledger" className="treasury-bargain-ledger" aria-label="Bargains Ledger">
      <header className="treasury-bargain-ledger-head">
        <div>
          <h2>Bargains Ledger</h2>
          <p>Every active and settled Bargain across the vault.</p>
        </div>
        <span>{ledger ? `${count} ${count === 1 ? "entry" : "entries"}` : "Loading"}</span>
      </header>
      <div className="treasury-bargain-ledger-controls">
        <div className="treasury-ledger-filter" aria-label="Bargains Ledger filter">
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
        <LedgerSortControls
          sortBy={sortBy}
          sortDirection={sortDirection}
          onSortChange={changeSort}
          onDirectionToggle={toggleSortDirection}
          ariaLabel="Bargains Ledger"
        />
      </div>
      {loading && !ledger ? <p className="status-performance-note">Opening the Bargains Ledger...</p> : null}
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
                summary={summary}
                onBargainClosed={async () => {
                  await onBargainClosed();
                  await loadEntries(offset);
                }}
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
              Previous
            </button>
            {hasLater ? (
              <button
                type="button"
                className="dialog-user-btn trade-history-latest-button"
                disabled={loading}
                onClick={() => void loadEntries(0)}
              >
                First page
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
              Next
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}

function AssetLedger({
  holding,
  summary,
  refreshKey,
  onPortfolioUpdated,
  onReload,
}: {
  holding: PortfolioHolding;
  summary: PortfolioSummary;
  refreshKey: string;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse) => void;
  onReload: () => Promise<void>;
}): JSX.Element {
  const [ledger, setLedger] = useState<AssetLedgerPayload | null>(null);
  const [filter, setFilter] = useState<AssetLedgerFilter>("all");
  const [sortBy, setSortBy] = useState<BargainLedgerSort>("date");
  const [sortDirection, setSortDirection] = useState<BargainLedgerDirection>("desc");
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
        `&filter=${encodeURIComponent(filter)}` +
        `&sort=${encodeURIComponent(sortBy)}` +
        `&direction=${encodeURIComponent(sortDirection)}` +
        `&entry_offset=${offset}`
      );
      setLedger(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load the asset ledger.");
    } finally {
      setLoading(false);
    }
  }, [filter, holding.asset_symbol, holding.quote_symbol, sortBy, sortDirection]);

  useEffect(() => {
    if (expanded) void loadEntries(0);
  }, [expanded, loadEntries, refreshKey]);

  function changeFilter(nextFilter: AssetLedgerFilter): void {
    if (nextFilter === filter) return;
    setLoading(true);
    setFilter(nextFilter);
    setLedger(null);
  }

  function changeSort(nextSort: BargainLedgerSort): void {
    if (nextSort === sortBy) return;
    setLoading(true);
    setSortBy(nextSort);
    setLedger(null);
  }

  function toggleSortDirection(): void {
    setLoading(true);
    setSortDirection((current) => current === "desc" ? "asc" : "desc");
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
      {expanded && !holding.is_dry_powder ? (
        <div className="treasury-bargain-ledger-controls treasury-asset-ledger-controls">
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
          <LedgerSortControls
            sortBy={sortBy}
            sortDirection={sortDirection}
            onSortChange={changeSort}
            onDirectionToggle={toggleSortDirection}
            ariaLabel="Asset Ledger"
          />
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
              if (entry.entry_type === "cash") {
                return (
                  <article key={entry.entry_id} className="treasury-ledger-row treasury-cash-ledger-row">
                    <span className={signedToneClass(entry.cash_event.amount_quote, "treasury-ledger-cash-delta")}>
                      {formatSignedCurrency(entry.cash_event.amount_quote)}
                    </span>
                    <span className="treasury-ledger-main">
                      <span className="treasury-ledger-entry">
                        {entry.cash_event.label} · {entry.cash_event.asset_symbol}
                      </span>
                      <small>Bargain #{entry.cash_event.swing_id.slice(-6).toUpperCase()}</small>
                    </span>
                    <span className="treasury-ledger-meta">{formatDateTimeEu(entry.occurred_at)}</span>
                  </article>
                );
              }
              if (entry.entry_type === "swing") {
                return (
                  <SwingLedgerRow
                    key={entry.entry_id}
                    swing={entry.swing}
                    occurredAt={entry.occurred_at}
                    summary={summary}
                    onBargainClosed={async () => {
                      await onReload();
                      await loadEntries(offset);
                    }}
                  />
                );
              }
              const transaction = entry.transaction;
              const isCashEntry = typeof entry.free_cash_delta === "number";
              const cashLabel = transaction.office_transfer_direction === "to_office"
                ? "Transferred to Futures Office"
                : transaction.office_transfer_direction === "from_office"
                  ? "Received from Futures Office"
                  : transaction.spot_quote_leg && transaction.swing_id
                    ? "Bargain cash flow"
                    : transaction.spot_quote_leg
                      ? "Asset accumulation"
                      : transactionLabel(transaction.tx_type);
              const canVoid = transaction.source === "manual";
              return (
                <article
                  key={entry.entry_id}
                  className={`treasury-ledger-row${transaction.status === "voided" ? " treasury-ledger-row-voided" : ""}`}
                >
                  <span className={isCashEntry
                    ? signedToneClass(entry.free_cash_delta, "treasury-ledger-cash-delta")
                    : transactionToneClass(transaction.tx_type)}>
                    {isCashEntry ? formatSignedCurrency(entry.free_cash_delta) : transactionLabel(transaction.tx_type)}
                  </span>
                  <span className="treasury-ledger-main">
                    <span className="treasury-ledger-entry">
                      {isCashEntry ? cashLabel : `${formatNumber(transaction.quantity, 8)} ${transaction.asset_symbol}`}
                      {isCashEntry
                        ? transaction.note ? ` · ${transaction.note}` : ""
                        : ""}
                      {!isCashEntry && transaction.tx_type === "custody_transfer" && transaction.source_custody && transaction.destination_custody
                        ? ` from ${CUSTODY_LABELS[transaction.source_custody]} to ${CUSTODY_LABELS[transaction.destination_custody]}`
                        : !isCashEntry && transaction.price
                          ? ` at ${formatCurrency(transaction.price, 6)}`
                          : !isCashEntry ? ` in ${CUSTODY_LABELS[transaction.custody_location]}` : ""}
                    </span>
                    {transaction.status === "voided" ? <small>Voided</small> : null}
                  </span>
                  <span className="treasury-ledger-meta">{formatDateTimeEu(transaction.executed_at)}</span>
                  {canVoid ? <button
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
                  </button> : <span />}
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
  summary,
  realizedAccumulatedCash,
  onPortfolioUpdated,
  onReload,
}: {
  holding: PortfolioHolding;
  exchange: PortfolioExchange | null;
  summary: PortfolioSummary;
  realizedAccumulatedCash: number;
  onPortfolioUpdated: (response: CreatePortfolioTransactionResponse | UpdatePortfolioPolicyResponse | UpdateCashPolicyResponse) => void;
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
        title: `I am using Bargains to accumulate more ${holding.asset_symbol}.`,
      }
    : tradingState === "unlocked" && holding.trading_objective === "accumulate_cash"
      ? {
          icon: "\u{1F4B5}",
          tone: "cash",
          title: "I am using Bargains to accumulate more USDT.",
        }
      : {
          icon: "\u{1F512}",
          tone: "locked",
          title: !executionEnabled
            ? "My automatic Bargains are disabled."
            : holding.is_dry_powder
              ? "Vault Reserve is not managed by an asset trading policy."
              : holding.trading_objective === null
                ? "Choose a Trading Objective before I can bargain with this asset."
                : "I keep the full holding protected by policy.",
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
            {holding.rolling_24h_change_pct === null ? null : (
              <span
                className={signedToneClass(
                  holding.rolling_24h_change_pct,
                  "treasury-rolling-change",
                )}
                title="Rolling 24-hour price change used by Spot signals"
              >
                {formatSignedPercent(holding.rolling_24h_change_pct)}
              </span>
            )}
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
                    {formatSignedCurrency(holding.total_gain)} · {formatSignedPercent(holding.total_gain_pct)}
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
            summary={summary}
            onPortfolioUpdated={onPortfolioUpdated}
            onExecuted={onReload}
          />
          {holding.is_dry_powder ? (
            <CashPolicyPanel
              summary={summary}
              exchange={exchange}
              onUpdated={onPortfolioUpdated}
              onExecuted={onReload}
            />
          ) : executionEnabled ? (
            <AssetPolicyPanel holding={holding} exchange={exchange} onUpdated={onPortfolioUpdated} />
          ) : null}
          <AssetLedger
            holding={holding}
            summary={summary}
            refreshKey={refreshKey}
            onPortfolioUpdated={onPortfolioUpdated}
            onReload={onReload}
          />
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
  const initialLoadSettled = useRef<boolean>(false);

  const loadPortfolio = useCallback(async (background = false): Promise<void> => {
    if (refreshInFlight.current) {
      return;
    }
    refreshInFlight.current = true;
    if (!background) {
      setLoading(true);
    }
    const retryDelays = !background && !initialLoadSettled.current
      ? INITIAL_TREASURY_RETRY_DELAYS_MS
      : [];
    try {
      for (let attempt = 0; ; attempt += 1) {
        try {
          const payload = await fetchApi<PortfolioPayload>("/api/portfolio");
          setPortfolio(payload);
          setError(null);
          break;
        } catch (loadError) {
          const retryDelay = retryDelays[attempt];
          if (retryDelay === undefined || !isRetryablePortfolioLoad(loadError)) {
            throw loadError;
          }
          await waitForRetry(retryDelay);
        }
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Treasury is unavailable.");
    } finally {
      if (!background) {
        initialLoadSettled.current = true;
      }
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
    }, TREASURY_REFRESH_MS);
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
          {loading ? <p className="status-performance-note">I am counting the vault...</p> : null}

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
              {typeof summary?.total_value_24h_change === "number" ? (
                <span className="treasury-total-change">
                  <span
                    className={signedToneClass(
                      summary.total_value_24h_change,
                      "treasury-total-change-value"
                    )}
                  >
                    {formatSignedCurrency(summary.total_value_24h_change)}
                  </span>
                  <span className="treasury-total-change-separator" aria-hidden="true">·</span>
                  <span
                    className={signedToneClass(
                      summary.total_value_24h_change_pct,
                      "treasury-total-change-pct"
                    )}
                  >
                    {formatSignedPercent(summary.total_value_24h_change_pct)}
                  </span>
                </span>
              ) : null}
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
              <span className={signedToneClass(summary?.total_gain_pct, "treasury-summary-note")}>
                {formatSignedPercent(summary?.total_gain_pct)}
              </span>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Vault Reserve</span>
              <strong>{formatCurrency(summary?.vault_reserve ?? 0)}</strong>
              <span className="treasury-summary-note">
                {formatCurrency(summary?.vault_reserve_available ?? 0)} available
                {(summary?.vault_reserve_committed ?? 0) > 0
                  ? <><br />{formatCurrency(summary?.vault_reserve_committed ?? 0)} committed</>
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
                  <strong className="treasury-bargain-count-open">{formatNumber(summary?.open_swing_count ?? 0, 0)}</strong>
                  <small>open</small>
                </span>
                <span>
                  <strong>{formatNumber(summary?.closed_swing_count ?? 0, 0)}</strong>
                  <small>closed</small>
                </span>
                <span>
                  <strong>{formatNumber(summary?.total_swing_count ?? 0, 0)}</strong>
                  <small>total</small>
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
              summary={summary!}
              onBargainClosed={() => loadPortfolio()}
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

        <p className="dialog-scrooge treasury-role-divider">My treasury holdings:</p>

        <section className="section-block treasury-holdings-section">
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
                    Bring a new treasure under my care and place it in its first custody location.
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
                    summary={summary!}
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
            <p className="trade-history-empty-sheet treasury-holdings-list">The treasury is empty. Add your first treasure.</p>
          ) : (
            <div className="treasury-holdings-grid treasury-holdings-list">
              {holdings.map((holding) => (
                <HoldingCard
                  key={`${holding.asset_symbol}-${holding.quote_symbol}`}
                  holding={holding}
                  exchange={exchange}
                  summary={summary!}
                  realizedAccumulatedCash={summary?.realized_accumulated_cash ?? 0}
                  onPortfolioUpdated={(response) => {
                    setPortfolio(mergePortfolioPayload(response.portfolio, response.warnings));
                  }}
                  onReload={loadPortfolio}
                />
              ))}
            </div>
          )}
          <div className="treasury-add-treasure-row">
            <button
              type="button"
              className="dialog-user-btn treasury-form-toggle"
              aria-expanded={formExpanded}
              onClick={() => setFormExpanded((current) => !current)}
            >
              {formExpanded ? "Close Intake" : "Add Treasure"}
            </button>
          </div>
        </section>

        <p className="dialog-scrooge treasury-role-divider">My treasury rules:</p>

        <TreasuryRulesPanel />

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
