"use client";

import { useEffect, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";
import { buildContractParagraphs, type ContractParagraph, type EditableConfig } from "../../lib/contract";
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

type PositionSide = "long" | "short";

type OpenTradeInfo = {
  side: PositionSide;
  size: number;
  entry: number;
  break_even_price: number | null;
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

type TradeSuggestionSide = "buy" | "sell";
type ControlAction = "start" | "stop" | "restart" | "close_position" | "suggest_trade" | "update_sl" | "update_tp";
type ControlEndpoint = "start" | "stop" | "restart" | "close-position" | "suggest-trade" | "update-sl" | "update-tp";
type ControlActionLike = ControlAction | ControlEndpoint;

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

type RawConfigResponse = {
  raw_text: string;
  editable: EditableConfig;
  path: string;
};

type TradeHistoryItem = {
  side: PositionSide;
  size: number | null;
  entry: number | null;
  sl: number | null;
  tp: number | null;
  liq_price?: number | null;
  trail_active?: boolean | null;
  trail_price?: number | null;
  time?: string | null;
  entry_time: string | null;
  exit: number | null;
  exit_time: string | null;
  exit_reason: string | null;
  net_pnl: number | null;
  gross_pnl?: number | null;
  fee: number | null;
  trigger?: string | null;
  stake_mode?: string | null;
  entry_rsi?: number | null;
  via_tail_guard?: boolean | null;
  exit_rsi?: number | null;
  exit_threshold?: number | null;
};

type TradeHistoryResponse = {
  path: string;
  requested_limit: number | null;
  lookback_days_applied: number | null;
  total_trades: number;
  returned_trades: number;
  trades: TradeHistoryItem[];
  warnings: string[];
};

type SaveRawConfigResponse = {
  updated: boolean;
  restart_required: boolean;
  changed_fields: string[];
  backup_path: string | null;
  raw_text: string;
  editable: EditableConfig;
};

const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;
const COMMAND_POLL_MS = 2000;
const INSTRUCTION_TOAST_AUTO_DISMISS_MS = 3000;
const TRADE_HISTORY_PAGE_SIZE = 3;
const TRADE_HISTORY_LOOKBACK_DAYS = 90;
const TRADE_HISTORY_LOOKBACK_LABEL = "last 3 months";
const COMMON_QUOTE_ASSETS = ["USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH", "BNB", "EUR", "TRY", "USD"];

function formatNumberValue(
  value: number | null | undefined,
  options: {
    minimumFractionDigits?: number;
    maximumFractionDigits?: number;
  } = {}
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: options.minimumFractionDigits ?? 0,
    maximumFractionDigits: options.maximumFractionDigits ?? 2
  }).format(value);
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  if (typeof value === "number") {
    return formatNumberValue(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
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

function statusIntentBadgeClass(status: UiStatus | null): string | null {
  if (status?.code === "looking_for_buy_opportunity") {
    return "badge badge-good";
  }
  if (status?.code === "looking_for_sell_opportunity") {
    return "badge badge-bad";
  }
  if (status?.code === "resting") {
    return "badge badge-muted";
  }
  return null;
}

function statusIntentBadgeText(status: UiStatus | null): string | null {
  if (status?.code === "looking_for_buy_opportunity" || status?.code === "looking_for_sell_opportunity") {
    return "🔎";
  }
  if (status?.code === "resting") {
    return "😴";
  }
  return null;
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

function normalizePositionSide(side: unknown): PositionSide | null {
  const normalized = typeof side === "string" ? side.trim().toLowerCase() : "";
  if (normalized === "long" || normalized === "short") {
    return normalized;
  }
  return null;
}

function positionSummaryBadgeClass(
  hasOpenPosition: boolean,
  roiPercent: number | null
): string {
  if (hasOpenPosition) {
    if (roiPercent === null) {
      return "position-upnl-badge position-upnl-badge-neutral";
    }
    if (roiPercent < 0) {
      return "position-upnl-badge position-upnl-badge-negative";
    }
    return "position-upnl-badge position-upnl-badge-positive";
  }
  return "position-upnl-badge position-upnl-badge-neutral";
}

function positionSummaryBadgeText(
  hasOpenPosition: boolean,
  roiPercent: number | null
): string | null {
  if (hasOpenPosition) {
    return formatPercent(roiPercent);
  }
  return null;
}

function positionSideBadgeClass(side: unknown): string {
  const normalized = normalizePositionSide(side);
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
  const normalized = normalizePositionSide(side);
  if (normalized === "long") {
    return "BUY";
  }
  if (normalized === "short") {
    return "SELL";
  }
  const fallback = String(side).trim();
  return fallback ? fallback.toUpperCase() : "N/A";
}

function unrealizedPnlClass(value: number | null): string {
  if (value === null) {
    return "metric-value floating-pnl-value value-neutral";
  }
  if (value > 0) {
    return "metric-value floating-pnl-value value-positive";
  }
  if (value < 0) {
    return "metric-value floating-pnl-value value-negative";
  }
  return "metric-value floating-pnl-value value-neutral";
}

function formatUnrealizedPnlUsd(value: number | null): string {
  if (value === null) {
    return "N/A";
  }
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  const absolute = formatNumberValue(Math.abs(value), {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
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
  return `$${formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatVaultAmount(value: number | null): string {
  return formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatPrice(value: number | null | undefined): string {
  return formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatLeverage(value: number | null | undefined): string {
  const formatted = formatNumberValue(value, { minimumFractionDigits: 0, maximumFractionDigits: 2 });
  return formatted === "N/A" ? "N/A" : `x${formatted}`;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function buildTradeProgressTrackBackground(entryPercent: number, slPercent: number): string {
  const entry = clampPercent(entryPercent);
  const safetyNet = clampPercent(slPercent);

  if (safetyNet >= entry - 0.2) {
    const warmApproach = clampPercent(Math.max(0, entry - 6));
    const entryBlend = clampPercent(entry + Math.max(6, (safetyNet - entry) * 0.18));
    const greenRise = clampPercent(safetyNet + Math.max(8, (100 - safetyNet) * 0.18));
    const greenMid = clampPercent(safetyNet + (100 - safetyNet) * 0.46);
    return `linear-gradient(90deg,
      rgba(104, 100, 74, 0.22) 0%,
      rgba(128, 122, 84, 0.24) ${warmApproach}%,
      rgba(212, 179, 98, 0.28) ${entry}%,
      rgba(167, 172, 111, 0.28) ${entryBlend}%,
      rgba(94, 152, 108, 0.4) ${safetyNet}%,
      rgba(78, 173, 121, 0.54) ${greenRise}%,
      rgba(103, 214, 148, 0.68) ${greenMid}%,
      rgba(123, 233, 163, 0.74) 100%)`;
  }

  const leftMid = clampPercent(entry * 0.42);
  const leftNear = clampPercent(Math.max(0, entry - 11));
  const rightNear = clampPercent(Math.min(100, entry + 8));
  const rightMid = clampPercent(entry + (100 - entry) * 0.4);
  return `linear-gradient(90deg,
    rgba(177, 61, 76, 0.52) 0%,
    rgba(173, 71, 83, 0.46) ${leftMid}%,
    rgba(150, 93, 79, 0.34) ${leftNear}%,
    rgba(212, 178, 98, 0.26) ${entry}%,
    rgba(160, 151, 100, 0.24) ${rightNear}%,
    rgba(91, 145, 105, 0.36) ${rightMid}%,
    rgba(67, 146, 109, 0.48) 100%)`;
}

type TradeProgressMarkerKey = "sl" | "entry" | "tp";

type TradeProgressMarker = {
  key: TradeProgressMarkerKey;
  label: "SN" | "Entry" | "TM";
  value: number;
  percent: number;
};

type TradeProgressSnapshot = {
  slPercent: number;
  entryPercent: number;
  tpPercent: number;
  currentPercent: number | null;
  currentLabelPercent: number | null;
  currentTone: "positive" | "negative" | "neutral";
  markers: TradeProgressMarker[];
};

function computeTradeProgress(
  sl: number | null,
  entry: number | null,
  tp: number | null,
  current: number | null
): TradeProgressSnapshot | null {
  if (!isFiniteNumber(sl) || !isFiniteNumber(entry) || !isFiniteNumber(tp) || sl === tp) {
    return null;
  }

  const isLong = tp > entry;
  const markers = [
    { key: "sl" as const, label: "SN" as const, value: sl },
    { key: "entry" as const, label: "Entry" as const, value: entry },
    { key: "tp" as const, label: "TM" as const, value: tp }
  ].sort((left, right) => (isLong ? left.value - right.value : right.value - left.value));

  const leftValue = markers[0]?.value;
  const rightValue = markers[markers.length - 1]?.value;
  const span = rightValue - leftValue;
  if (!Number.isFinite(span) || span === 0) {
    return null;
  }

  const toPercent = (value: number): number => ((value - leftValue) / span) * 100;
  const normalizedMarkers = markers.map((marker) => ({
    ...marker,
    percent: clampPercent(toPercent(marker.value))
  }));
  const entryRaw = toPercent(entry);
  const slRaw = toPercent(sl);
  const tpRaw = toPercent(tp);
  const currentRaw = isFiniteNumber(current) ? toPercent(current) : null;
  const currentPercent = currentRaw === null ? null : clampPercent(currentRaw);

  let currentTone: TradeProgressSnapshot["currentTone"] = "neutral";
  if (currentRaw !== null) {
    if (currentRaw > entryRaw + 0.2) {
      currentTone = "positive";
    } else if (currentRaw < entryRaw - 0.2) {
      currentTone = "negative";
    }
  }

  return {
    slPercent: clampPercent(slRaw),
    entryPercent: clampPercent(entryRaw),
    tpPercent: clampPercent(tpRaw),
    currentPercent,
    currentLabelPercent: currentPercent === null ? null : Math.max(8, Math.min(92, currentPercent)),
    currentTone,
    markers: normalizedMarkers
  };
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
  const formatted = formatNumberValue(Math.abs(value), {
    minimumFractionDigits: 0,
    maximumFractionDigits: 6
  });
  const baseAsset = resolveBaseAsset(symbol);
  return baseAsset ? `${formatted} ${baseAsset}` : formatted;
}

function formatPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value > 0) {
    return `+${formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
  }
  if (value < 0) {
    return `${formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
  }
  return "0.00%";
}

function parseTradeTimestamp(value: string | null | undefined): number | null {
  if (!value || typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().replace(" ", "T");
  const timestamp = Date.parse(normalized);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function formatTradeDuration(entryTime: string | null | undefined, exitTime: string | null | undefined): string {
  const entryTimestamp = parseTradeTimestamp(entryTime);
  const exitTimestamp = parseTradeTimestamp(exitTime);
  if (entryTimestamp === null || exitTimestamp === null || exitTimestamp <= entryTimestamp) {
    return "N/A";
  }

  const durationSeconds = Math.round((exitTimestamp - entryTimestamp) / 1000);
  const durationMinutes = Math.floor(durationSeconds / 60);
  const days = Math.floor(durationMinutes / 1440);
  const hours = Math.floor((durationMinutes % 1440) / 60);
  const minutes = durationMinutes % 60;

  if (days > 0) {
    return `${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${Math.max(1, minutes)}m`;
}

function tradeHistoryToneClass(netPnl: number | null): string {
  if (netPnl === null || !Number.isFinite(netPnl)) {
    return "trade-history-item trade-history-item-neutral";
  }
  if (netPnl > 0) {
    return "trade-history-item trade-history-item-positive";
  }
  if (netPnl < 0) {
    return "trade-history-item trade-history-item-negative";
  }
  return "trade-history-item trade-history-item-neutral";
}

function tradeHistoryPnlPillClass(netPnl: number | null): string {
  if (netPnl === null || !Number.isFinite(netPnl)) {
    return "trade-history-pnl trade-history-pnl-neutral";
  }
  if (netPnl > 0) {
    return "trade-history-pnl trade-history-pnl-positive";
  }
  if (netPnl < 0) {
    return "trade-history-pnl trade-history-pnl-negative";
  }
  return "trade-history-pnl trade-history-pnl-neutral";
}

function formatSignedUsd(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  const absolute = formatNumberValue(Math.abs(value), {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
  if (value > 0) {
    return `+$${absolute}`;
  }
  if (value < 0) {
    return `-$${absolute}`;
  }
  return "$0.00";
}

function formatExitReason(value: string | null | undefined): string {
  if (!value || typeof value !== "string") {
    return "N/A";
  }
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatTriggerTrace(value: string | null | undefined): string {
  if (!value || typeof value !== "string") {
    return "Opened on a prior signal";
  }
  if (value === "manual_suggestion") {
    return "Opened from your suggestion";
  }
  if (value === "strategy_rules") {
    return "Opened by strategy rules";
  }
  return `Opened by ${formatExitReason(value)}`;
}

function formatStakeModeTrace(value: string | null | undefined): string | null {
  if (!value || typeof value !== "string") {
    return null;
  }
  if (value === "half") {
    return "with half stake";
  }
  if (value === "full") {
    return "with full stake";
  }
  return `with ${formatExitReason(value).toLowerCase()}`;
}

function formatTraceNumber(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return formatNumberValue(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function buildTradeOpenTrace(trade: TradeHistoryItem): string {
  const parts: string[] = [formatTriggerTrace(trade.trigger)];
  const stakeModeTrace = formatStakeModeTrace(trade.stake_mode);
  const entryRsiTrace = formatTraceNumber(trade.entry_rsi);

  if (stakeModeTrace) {
    parts.push(stakeModeTrace);
  }
  if (entryRsiTrace) {
    parts.push(`at RSI ${entryRsiTrace}`);
  }

  return `${parts.join(" ")}.`;
}

function buildTradeCloseTrace(trade: TradeHistoryItem): string {
  const exitReason = trade.exit_reason ?? "";
  const exitRsiTrace = formatTraceNumber(trade.exit_rsi);
  const exitThresholdTrace = formatTraceNumber(trade.exit_threshold);

  if (!exitReason) {
    return "Closed for a reason not yet recorded.";
  }

  if (exitReason === "manual_close") {
    return "Closed by your request.";
  }
  if (exitReason === "liquidation") {
    return "Closed by liquidation.";
  }
  if (exitReason === "stop_loss") {
    return "Closed by the Safety Net.";
  }
  if (exitReason === "take_profit" && trade.via_tail_guard) {
    return "Closed under Tail Guard after profit lock-in.";
  }
  if (exitReason === "take_profit") {
    return "Closed by the Treasure Mark.";
  }
  if (exitReason === "rsi_extreme") {
    if (exitRsiTrace && exitThresholdTrace) {
      return `Closed on RSI extreme: ${exitRsiTrace} against ${exitThresholdTrace}.`;
    }
    if (exitRsiTrace) {
      return `Closed on RSI extreme at ${exitRsiTrace}.`;
    }
    return "Closed on RSI extreme.";
  }
  return `Closed by ${formatExitReason(trade.exit_reason).toLowerCase()}.`;
}

function buildTradeNarration(trade: TradeHistoryItem): string {
  return `${buildTradeOpenTrace(trade)} ${buildTradeCloseTrace(trade)}`.replace(/\s+/g, " ").trim();
}

function formatDateTimeEuCompact(value: unknown, fallback = "N/A"): string {
  const formatted = formatDateTimeEu(value, fallback);
  if (formatted === fallback) {
    return fallback;
  }

  const fullMatch = formatted.match(/^(\d{2})\.(\d{2})\.\d{4}\s(\d{2}:\d{2}):\d{2}$/);
  if (fullMatch) {
    return `${fullMatch[1]}.${fullMatch[2]} ${fullMatch[3]}`;
  }

  return formatted.replace(/:\d{2}$/, "");
}

function contractParagraphToPlainText(paragraph: ContractParagraph): string {
  return paragraph
    .map((segment) => ("display" in segment ? segment.display : segment.text))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function buildContractTeaser(paragraphs: ContractParagraph[]): string {
  const firstParagraph = paragraphs[0];
  if (!firstParagraph) {
    return "No covenant is loaded yet.";
  }

  const firstText = contractParagraphToPlainText(firstParagraph);
  if (firstText.length <= 152) {
    return firstText;
  }
  return `${firstText.slice(0, 149).trimEnd()}...`;
}

type TargetPnlPreview = {
  pnlUsd: number;
  tone: "positive" | "negative" | "neutral";
};

function computeTargetPnlPreview(
  side: "long" | "short" | null,
  entryPrice: number | null,
  positionSize: number | null,
  rawValue: string,
  fallbackTargetPrice: number | null
): TargetPnlPreview | null {
  const trimmed = rawValue.trim();
  const targetPrice = trimmed ? Number(trimmed) : fallbackTargetPrice;
  const safeTargetPrice = typeof targetPrice === "number" && Number.isFinite(targetPrice) ? targetPrice : null;
  if (
    !side ||
    safeTargetPrice === null ||
    safeTargetPrice <= 0 ||
    !isFiniteNumber(entryPrice) ||
    !isFiniteNumber(positionSize)
  ) {
    return null;
  }

  const pnlUsd =
    side === "long" ? (safeTargetPrice - entryPrice) * positionSize : (entryPrice - safeTargetPrice) * positionSize;
  const tone = pnlUsd > 0 ? "positive" : pnlUsd < 0 ? "negative" : "neutral";
  return { pnlUsd, tone };
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
  return formatPrice(trailPrice);
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

function normalizeControlAction(action: ControlActionLike | null | undefined): ControlAction | null {
  if (!action) {
    return null;
  }

  const normalized = action.replaceAll("-", "_");
  if (
    normalized === "start" ||
    normalized === "stop" ||
    normalized === "restart" ||
    normalized === "close_position" ||
    normalized === "suggest_trade" ||
    normalized === "update_sl" ||
    normalized === "update_tp"
  ) {
    return normalized;
  }

  return null;
}

function instructionActionLabel(action: ControlActionLike | null | undefined): string {
  const normalized = normalizeControlAction(action);
  switch (normalized) {
    case "start":
      return "Open the office";
    case "stop":
      return "Close the office";
    case "restart":
      return "Restart the office";
    case "close_position":
      return "Close the trade";
    case "suggest_trade":
      return "Act on your suggestion";
    case "update_sl":
      return "Reset the Safety Net";
    case "update_tp":
      return "Move the Treasure Mark";
    default:
      return "Carry out your instruction";
  }
}

function isInstructionPending(status: string | undefined): boolean {
  return status === "pending" || status === "processing";
}

function isInstructionTerminal(status: string | undefined): boolean {
  return status === "completed" || status === "failed";
}

function isProminentInstruction(action: ControlActionLike | null | undefined): boolean {
  const normalized = normalizeControlAction(action);
  return (
    normalized === "start" ||
    normalized === "stop" ||
    normalized === "restart" ||
    normalized === "close_position" ||
    normalized === "suggest_trade"
  );
}

function instructionStatusTitle(status: string | undefined): string {
  if (status === "completed") {
    return "Instruction fulfilled";
  }
  if (status === "failed") {
    return "Instruction failed";
  }
  return "Instruction in motion";
}

function instructionStatusLabel(status: string | undefined): string {
  if (status === "completed") {
    return "Completed";
  }
  if (status === "failed") {
    return "Failed";
  }
  if (status === "processing") {
    return "Processing";
  }
  return "Pending";
}

function instructionOverlaySummary(action: ControlActionLike | null | undefined): string {
  const normalized = normalizeControlAction(action);
  switch (normalized) {
    case "start":
      return "Scrooge is opening the office and taking his post.";
    case "stop":
      return "Scrooge is closing the office and securing the floor.";
    case "restart":
      return "Scrooge is straightening the ledgers and reopening the office.";
    case "close_position":
      return "Scrooge is closing the trade and settling the books.";
    case "suggest_trade":
      return "Scrooge is acting on your instruction at the next live tick.";
    default:
      return "Scrooge is carrying out your instruction.";
  }
}

function DashboardContent(): JSX.Element {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [wsFallbackActive, setWsFallbackActive] = useState<boolean>(false);
  const [busyAction, setBusyAction] = useState<ControlEndpoint | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [commandResult, setCommandResult] = useState<ControlResponse | null>(null);
  const [commandStatus, setCommandStatus] = useState<CommandStatusResponse | null>(null);
  const [dismissedInstructionId, setDismissedInstructionId] = useState<string | null>(null);
  const [pinnedInstructionId, setPinnedInstructionId] = useState<string | null>(null);
  const [showTradeSuggestions, setShowTradeSuggestions] = useState<boolean>(false);
  const [slValue, setSlValue] = useState<string>("");
  const [tpValue, setTpValue] = useState<string>("");
  const [editableConfig, setEditableConfig] = useState<EditableConfig | null>(null);
  const [rawConfigText, setRawConfigText] = useState<string>("");
  const [rawConfigDraft, setRawConfigDraft] = useState<string>("");
  const [contractEditing, setContractEditing] = useState<boolean>(false);
  const [configLoading, setConfigLoading] = useState<boolean>(true);
  const [configSaving, setConfigSaving] = useState<boolean>(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configInfo, setConfigInfo] = useState<string | null>(null);
  const [positionExpanded, setPositionExpanded] = useState<boolean>(false);
  const [tradeHistory, setTradeHistory] = useState<TradeHistoryItem[]>([]);
  const [tradeHistoryError, setTradeHistoryError] = useState<string | null>(null);
  const [tradeHistoryLoading, setTradeHistoryLoading] = useState<boolean>(true);
  const [tradeHistoryPage, setTradeHistoryPage] = useState<number>(0);
  const [tradeHistoryTotal, setTradeHistoryTotal] = useState<number>(0);
  const [contractExpanded, setContractExpanded] = useState<boolean>(false);

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

  async function loadTradeHistory(silent = false): Promise<void> {
    if (!silent) {
      setTradeHistoryLoading(true);
    }
    try {
      const payload = await fetchApi<TradeHistoryResponse>(
        `/api/history/trades?lookback_days=${TRADE_HISTORY_LOOKBACK_DAYS}`
      );
      setTradeHistory(payload.trades);
      setTradeHistoryTotal(payload.total_trades);
      setTradeHistoryError(null);
    } catch (err) {
      setTradeHistoryError(err instanceof Error ? err.message : "Failed to load trade history");
    } finally {
      if (!silent) {
        setTradeHistoryLoading(false);
      }
    }
  }

  async function submitControlAction(endpoint: ControlEndpoint, body?: unknown): Promise<boolean> {
    setBusyAction(endpoint);
    setControlError(null);
    setDismissedInstructionId(null);
    try {
      const response = await fetchApi<ControlResponse>(`/api/control/${endpoint}`, {
        method: "POST",
        body
      });
      setCommandResult(response);
      const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${response.command_id}`);
      setCommandStatus(statusPayload);
      return true;
    } catch (err) {
      setControlError(err instanceof Error ? err.message : `Failed to execute ${endpoint}`);
      return false;
    } finally {
      setBusyAction(null);
    }
  }

  async function runControlAction(
    endpoint: ControlEndpoint,
    options?: { confirmMessage?: string; body?: unknown }
  ): Promise<boolean> {
    const confirmMessage = options?.confirmMessage;
    if (confirmMessage) {
      const confirmed = window.confirm(confirmMessage);
      if (!confirmed) {
        return false;
      }
    }

    return submitControlAction(endpoint, options?.body);
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

  async function runBreakEvenSafetyNet(): Promise<void> {
    if (!isFiniteNumber(breakEvenTargetPrice) || breakEvenTargetPrice <= 0) {
      setControlError("Break-even price is not available.");
      return;
    }

    await runControlAction("update-sl", {
      body: { value: breakEvenTargetPrice },
      confirmMessage: `Bring the Net to Even at ${breakEvenTargetPrice}?`
    });
  }

  async function runSuggestedTrade(side: TradeSuggestionSide): Promise<void> {
    const actionLabel = side === "buy" ? "buy" : "sell";
    await runControlAction("suggest-trade", {
      body: { side },
      confirmMessage: `Are you sure I should ${actionLabel} right now?`
    });
    setShowTradeSuggestions(false);
  }

  async function runCloseFloorFlow(): Promise<void> {
    if (hasOpenPosition) {
      const confirmed = window.confirm(
        "There is an open trade on the floor. It will be closed before I shut the office. Proceed?"
      );
      if (!confirmed) {
        return;
      }

      const closeQueued = await submitControlAction("close-position");
      if (!closeQueued) {
        return;
      }

      await submitControlAction("stop");
      return;
    }

    await runControlAction("stop", {
      confirmMessage: "Close the trading floor for now?"
    });
  }

  async function loadEditableConfig(): Promise<void> {
    setConfigLoading(true);
    setConfigError(null);
    try {
      const payload = await fetchApi<RawConfigResponse>("/api/config/raw");
      setEditableConfig(payload.editable);
      setRawConfigText(payload.raw_text);
      setRawConfigDraft(payload.raw_text);
      setContractEditing(false);
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Failed to load contract");
    } finally {
      setConfigLoading(false);
    }
  }

  async function saveEditableConfig(): Promise<void> {
    if (!editableConfig) {
      return;
    }

    const confirmed = window.confirm("Update the contract and reopen the office?");
    if (!confirmed) {
      return;
    }

    setConfigSaving(true);
    setConfigError(null);
    setConfigInfo(null);
    try {
      const result = await fetchApi<SaveRawConfigResponse>("/api/config/raw", {
        method: "POST",
        body: { raw_text: rawConfigDraft }
      });
      setEditableConfig(result.editable);
      setRawConfigText(result.raw_text);
      setRawConfigDraft(result.raw_text);
      setContractEditing(false);
      if (result.updated) {
        setConfigInfo("Contract updated.");
      } else {
        setConfigInfo("No contract changes detected.");
      }

      if (result.updated) {
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
      setWsFallbackActive(false);
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
        setWsFallbackActive(false);
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
          setWsFallbackActive(true);
          reconnectTimer = window.setTimeout(connect, WS_RECONNECT_MS);
        }
      };
      socket.onerror = () => {
        setWsConnected(false);
        setWsFallbackActive(true);
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
    void loadTradeHistory(false);
    const intervalId = window.setInterval(() => {
      void loadTradeHistory(true);
    }, POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    const maxPageIndex = Math.max(0, Math.ceil(tradeHistory.length / TRADE_HISTORY_PAGE_SIZE) - 1);
    setTradeHistoryPage((currentPage) => Math.min(currentPage, maxPageIndex));
  }, [tradeHistory.length]);

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
      void loadTradeHistory(true);
    }
  }, [commandStatus]);

  useEffect(() => {
    if (!commandStatus?.command_id) {
      return;
    }
    setDismissedInstructionId(null);
    setPinnedInstructionId(null);
  }, [commandStatus?.command_id]);

  useEffect(() => {
    if (!commandStatus?.command_id || !isInstructionTerminal(commandStatus.status)) {
      return () => undefined;
    }
    if (dismissedInstructionId === commandStatus.command_id) {
      return () => undefined;
    }
    if (pinnedInstructionId === commandStatus.command_id) {
      return () => undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setDismissedInstructionId(commandStatus.command_id);
    }, INSTRUCTION_TOAST_AUTO_DISMISS_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [commandStatus, dismissedInstructionId, pinnedInstructionId]);

  const openTradeInfo = data?.open_trade_info ?? null;
  const sl = openTradeInfo?.sl ?? null;
  const tp = openTradeInfo?.tp ?? null;
  const liquidationPrice = openTradeInfo?.liq_price ?? null;
  const unrealizedPnl = openTradeInfo?.unrealized_pnl ?? null;
  const roiPercent = openTradeInfo?.roi_pct ?? null;
  const entryPrice = openTradeInfo?.entry ?? null;
  const breakEvenPrice = openTradeInfo?.break_even_price ?? null;
  const breakEvenTargetPrice = isFiniteNumber(breakEvenPrice) ? breakEvenPrice : entryPrice;
  const positionSize = openTradeInfo?.size ?? null;
  const marginUsed = openTradeInfo?.margin_used ?? null;
  const lastPrice = data?.last_price ?? null;
  const openTime = formatDateTimeEu(openTradeInfo?.entry_time ?? null);
  const hasOpenPosition = openTradeInfo !== null;
  const openTradeSide = openTradeInfo?.side ?? null;
  const tradingEnabled = data?.trading_enabled ?? true;
  const botStatus = data?.bot_status ?? null;
  const tradeStatus = data?.trade_status ?? null;
  const contractParagraphs = editableConfig ? buildContractParagraphs(editableConfig) : [];
  const contractTeaser = buildContractTeaser(contractParagraphs);
  const tradeHistoryPageCount = Math.max(1, Math.ceil(tradeHistory.length / TRADE_HISTORY_PAGE_SIZE));
  const tradeHistoryPageIndex = Math.min(tradeHistoryPage, tradeHistoryPageCount - 1);
  const pagedTrades = tradeHistory.slice(
    tradeHistoryPageIndex * TRADE_HISTORY_PAGE_SIZE,
    tradeHistoryPageIndex * TRADE_HISTORY_PAGE_SIZE + TRADE_HISTORY_PAGE_SIZE
  );
  const positionSummaryText = tradeSummaryText(tradeStatus);
  const positionSummaryPhraseClass = tradeSummaryPhraseClass(tradeStatus);
  const positionUpnlBadgeText = positionSummaryBadgeText(hasOpenPosition, roiPercent);
  const showPositionUpnlBadge = positionUpnlBadgeText !== null;
  const positionUpnlBadgeClass = positionSummaryBadgeClass(hasOpenPosition, roiPercent);
  const tradeSummaryDirectionSide = hasOpenPosition ? normalizePositionSide(openTradeSide) : null;
  const statusIntentText = statusIntentBadgeText(botStatus);
  const statusIntentClass = statusIntentBadgeClass(botStatus);
  const tradeProgress = computeTradeProgress(sl, entryPrice, tp, lastPrice);
  const slDraftPreview = computeTargetPnlPreview(openTradeSide, entryPrice, positionSize, slValue, sl);
  const tpDraftPreview = computeTargetPnlPreview(openTradeSide, entryPrice, positionSize, tpValue, tp);
  const alreadyAtBreakEvenOrBetter =
    !isFiniteNumber(breakEvenTargetPrice) ||
    (openTradeSide === "long" && isFiniteNumber(sl) && sl >= breakEvenTargetPrice) ||
    (openTradeSide === "short" && isFiniteNumber(sl) && sl <= breakEvenTargetPrice);
  const showBreakEvenSafetyNetAction =
    hasOpenPosition &&
    isFiniteNumber(breakEvenTargetPrice) &&
    isFiniteNumber(unrealizedPnl) &&
    unrealizedPnl > 0 &&
    !alreadyAtBreakEvenOrBetter;
  const latestCommandStatusMatchesResult =
    commandStatus !== null && commandResult !== null && commandStatus.command_id === commandResult.command_id;
  const pendingInstructionAction: ControlActionLike | null =
    busyAction ?? (latestCommandStatusMatchesResult && isInstructionPending(commandStatus.status) ? commandStatus.action : null);
  const showInstructionOverlay =
    pendingInstructionAction !== null && isProminentInstruction(pendingInstructionAction);
  const showInstructionToast =
    commandStatus !== null &&
    isInstructionTerminal(commandStatus.status) &&
    dismissedInstructionId !== commandStatus.command_id;
  const toastCanAutoDismiss =
    commandStatus !== null &&
    isInstructionTerminal(commandStatus.status) &&
    pinnedInstructionId !== commandStatus.command_id;
  const toastIsPinned =
    commandStatus !== null &&
    pinnedInstructionId === commandStatus.command_id;

  useEffect(() => {
    if (!hasOpenPosition) {
      setPositionExpanded(false);
      return;
    }
    setShowTradeSuggestions(false);
  }, [hasOpenPosition]);

  return (
    <section className="panel page-shell office-panel">
      {showInstructionOverlay ? (
        <div className="instruction-overlay" role="status" aria-live="polite" aria-label="Instruction in motion">
          <div className="instruction-overlay-card">
            <span className="instruction-overlay-seal" aria-hidden="true" />
            <div className="instruction-overlay-copy">
              <span className="instruction-overlay-eyebrow">Instruction in motion</span>
              <strong className="instruction-overlay-title">{instructionActionLabel(pendingInstructionAction)}</strong>
              <p className="instruction-overlay-message">{instructionOverlaySummary(pendingInstructionAction)}</p>
            </div>
          </div>
        </div>
      ) : null}
      {showInstructionToast && commandStatus ? (
        <div
          className={`instruction-toast instruction-toast-${commandStatus.status ?? "pending"}${toastCanAutoDismiss ? " instruction-toast-timer-active" : ""}${toastIsPinned ? " instruction-toast-pinned" : ""}`}
          role="status"
          aria-live="polite"
          onClick={() => {
            if (isInstructionTerminal(commandStatus.status)) {
              setPinnedInstructionId(commandStatus.command_id);
            }
          }}
        >
          {isInstructionTerminal(commandStatus.status) ? (
            <span
              className="instruction-toast-timer-bar"
              style={{ animationDuration: `${INSTRUCTION_TOAST_AUTO_DISMISS_MS}ms` }}
              aria-hidden="true"
            />
          ) : null}
          <div className="instruction-toast-head">
            <div className="instruction-toast-title-group">
              <span className="instruction-toast-title">{instructionStatusTitle(commandStatus.status)}</span>
              <span className="instruction-toast-action">{instructionActionLabel(commandStatus.action)}</span>
            </div>
            <button
              type="button"
              className="instruction-toast-dismiss"
              onClick={(event) => {
                event.stopPropagation();
                setDismissedInstructionId(commandStatus.command_id);
              }}
              aria-label="Dismiss instruction notice"
            >
              ×
            </button>
          </div>
          <div className="instruction-toast-meta">
            <span className={commandStatusBadgeClass(commandStatus.status)}>{instructionStatusLabel(commandStatus.status)}</span>
          </div>
          <p className="dialog-scrooge instruction-toast-message">
            {commandStatus.message || `${instructionActionLabel(commandStatus.action)} is underway.`}
          </p>
        </div>
      ) : null}
      {wsFallbackActive ? (
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
                <span className="status-chip-value">
                  <span className={statusBadgeClass(botStatus)}>
                    {botStatus?.label ?? "N/A"}
                  </span>
                  {statusIntentText && statusIntentClass ? (
                    <span className={statusIntentClass}>{statusIntentText}</span>
                  ) : null}
                </span>
              </div>
              <div className="status-chip">
                <span className="status-chip-label">Vault</span>
                <span className="metric-value vault-value">
                  {formatVaultAmount(data.balance)}
                  {typeof data.balance === "number" && Number.isFinite(data.balance) ? (
                    <span className="vault-dollar" aria-hidden="true">
                      $
                    </span>
                  ) : null}
                </span>
              </div>
            </div>
            <p className="status-helper">
              <span className="status-helper-primary">
                Ticker: <span className="status-helper-value">{displayValue(data.symbol)}</span> • Leverage:{" "}
                <span className="status-helper-value">{formatLeverage(data.leverage)}</span>
              </span>
              <span className="status-helper-last-price">
                <span className="status-helper-separator" aria-hidden="true">
                  {" "}
                  •{" "}
                </span>
                Last Price: <span className="status-helper-value">{formatPrice(lastPrice)}</span>
              </span>
            </p>
            <details
              className="position-accordion position-accordion-featured"
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
                className={`position-accordion-summary${!hasOpenPosition ? " position-accordion-summary-disabled position-accordion-summary-empty" : ""}`}
                onClick={(event) => {
                  if (!hasOpenPosition) {
                    event.preventDefault();
                  }
                }}
              >
                <span className="position-summary-title">Current Trade</span>
                {tradeSummaryDirectionSide ? (
                  <span className={`position-summary-side ${positionSideBadgeClass(tradeSummaryDirectionSide)}`}>
                    {formatPositionSide(tradeSummaryDirectionSide)}
                  </span>
                ) : null}
                <span className={`position-summary-phrase-wrap${!hasOpenPosition ? " position-summary-phrase-wrap-empty" : ""}`}>
                  <span className={`${positionSummaryPhraseClass}${!hasOpenPosition ? " position-summary-phrase-empty" : ""}`}>
                    <span>{positionSummaryText}</span>
                  </span>
                </span>
                {showPositionUpnlBadge ? (
                  <span className={`position-summary-pnl ${positionUpnlBadgeClass}`}>{positionUpnlBadgeText}</span>
                ) : null}
              </summary>
              <div className="position-accordion-body">
                <div className="trade-pulse-grid">
                  <div className="kv-item metric-card">
                    <span className="kv-label">Entry Price</span>
                    <span className="metric-value">{formatPrice(entryPrice)}</span>
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

                {tradeProgress ? (
                  <div className="kv-item trade-progress-card">
                    <div className="trade-progress-header">
                      <span className="kv-label">Trade Progress</span>
                      <span className={`trade-progress-now trade-progress-now-${tradeProgress.currentTone}`}>
                        Now {formatPrice(lastPrice)}
                      </span>
                    </div>
                    <div className="trade-progress-track-shell">
                      <div
                        className="trade-progress-track"
                        aria-label="Trade progress from safety net to treasure mark"
                        style={{
                          background: buildTradeProgressTrackBackground(
                            tradeProgress.entryPercent,
                            tradeProgress.slPercent
                          )
                        }}
                      >
                        {Math.abs(tradeProgress.slPercent - tradeProgress.entryPercent) > 0.6 ? (
                          <span
                            className="trade-progress-level-marker trade-progress-level-marker-sl"
                            style={{ left: `${tradeProgress.slPercent}%` }}
                            aria-hidden="true"
                          >
                            <span className="trade-progress-level-dot" />
                          </span>
                        ) : null}
                        {Math.abs(tradeProgress.tpPercent - tradeProgress.entryPercent) > 0.6 ? (
                          <span
                            className="trade-progress-level-marker trade-progress-level-marker-tp"
                            style={{ left: `${tradeProgress.tpPercent}%` }}
                            aria-hidden="true"
                          >
                            <span className="trade-progress-level-dot" />
                          </span>
                        ) : null}
                        <span
                          className="trade-progress-entry-marker"
                          style={{ left: `${tradeProgress.entryPercent}%` }}
                          aria-hidden="true"
                        >
                          <span className="trade-progress-entry-dot" />
                        </span>
                        {tradeProgress.currentPercent !== null && tradeProgress.currentLabelPercent !== null ? (
                          <>
                            <span
                              className={`trade-progress-current-label trade-progress-current-label-${tradeProgress.currentTone}`}
                              style={{ left: `${tradeProgress.currentLabelPercent}%` }}
                            >
                              Now
                            </span>
                            <span
                              className={`trade-progress-current-marker trade-progress-current-marker-${tradeProgress.currentTone}`}
                              style={{ left: `${tradeProgress.currentPercent}%` }}
                              aria-hidden="true"
                            >
                              <span className="trade-progress-current-dot" />
                            </span>
                          </>
                        ) : null}
                      </div>
                    </div>
                    <div className="trade-progress-legend">
                      {tradeProgress.markers.map((marker, index) => (
                        <div
                          key={marker.key}
                          className={`trade-progress-legend-item trade-progress-legend-item-${marker.key} trade-progress-legend-item-${
                            index === 0 ? "start" : index === tradeProgress.markers.length - 1 ? "end" : "center"
                          }`}
                        >
                          <span className="trade-progress-legend-name">{marker.label}</span>
                          <span className="trade-progress-legend-value">{formatPrice(marker.value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="trade-manage-grid">
                  <div className="kv-item trade-manage-card trade-manage-card-with-quick-btn">
                    <div className="trade-manage-card-head">
                      <span className="kv-label">Safety Net</span>
                      {showBreakEvenSafetyNetAction ? (
                        <button
                          type="button"
                          className="dialog-user-btn trade-manage-quick-btn"
                          onClick={() => void runBreakEvenSafetyNet()}
                          disabled={busyAction !== null}
                          title="Bring the Net to Even"
                          aria-label="Bring the Net to Even"
                        >
                          <span className="trade-manage-quick-icon" aria-hidden="true">
                            ⚖️
                          </span>
                        </button>
                      ) : null}
                    </div>
                    <span className="metric-value">{formatPrice(sl)}</span>
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
                    <p className={`trade-edit-preview floating-pnl-value value-${slDraftPreview?.tone ?? "neutral"}`}>
                      If hit: {slDraftPreview ? formatUnrealizedPnlUsd(slDraftPreview.pnlUsd) : "N/A"}
                    </p>
                  </div>

                  <div className="kv-item trade-manage-card">
                    <span className="kv-label">Treasure Mark</span>
                    <span className="metric-value">{formatPrice(tp)}</span>
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
                    <p className={`trade-edit-preview floating-pnl-value value-${tpDraftPreview?.tone ?? "neutral"}`}>
                      If hit: {tpDraftPreview ? formatUnrealizedPnlUsd(tpDraftPreview.pnlUsd) : "N/A"}
                    </p>
                  </div>
                </div>

                {liquidationPrice !== null ? (
                  <p className="trade-danger-line">Danger line: {formatPrice(liquidationPrice)}</p>
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
            {!hasOpenPosition ? (
              <div className="toolbar trade-suggest-toolbar">
                <button
                  type="button"
                  className="dialog-user-btn"
                  onClick={() => setShowTradeSuggestions((prev) => !prev)}
                  disabled={busyAction !== null}
                >
                  Suggest a Trade
                </button>
                {showTradeSuggestions ? (
                  <div className="trade-suggest-actions">
                    <button
                      type="button"
                      className="dialog-user-btn button-success"
                      onClick={() => void runSuggestedTrade("buy")}
                      disabled={busyAction !== null}
                    >
                      Buy
                    </button>
                    <button
                      type="button"
                      className="dialog-user-btn button-danger"
                      onClick={() => void runSuggestedTrade("sell")}
                      disabled={busyAction !== null}
                    >
                      Sell
                    </button>
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="toolbar office-runtime-toolbar">
              {!tradingEnabled ? (
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
              ) : null}
              {tradingEnabled ? (
                <button
                  type="button"
                  className="dialog-user-btn"
                  onClick={() => void runCloseFloorFlow()}
                  disabled={busyAction !== null}
                >
                  Close the Floor
                </button>
              ) : null}
            </div>
          </div>
          <p className="dialog-scrooge">My previous trades:</p>
          <div className="section-block">
            {tradeHistoryLoading ? (
              <p className="dialog-scrooge dialog-scrooge-compact">Opening the ledger...</p>
            ) : tradeHistoryError ? (
              <p className="dialog-scrooge dialog-scrooge-error">{tradeHistoryError}</p>
            ) : tradeHistory.length === 0 ? (
              <div className="trade-history-empty-sheet">
                No closed trades. I have not sealed any bargains to recount yet.
              </div>
            ) : (
              <div className="trade-history-stack">
                {pagedTrades.map((trade, index) => {
                  const isPositive = typeof trade.net_pnl === "number" && trade.net_pnl > 0;
                  const isNegative = typeof trade.net_pnl === "number" && trade.net_pnl < 0;
                  const sideClass = positionSideBadgeClass(trade.side);
                  const sideLabel = formatPositionSide(trade.side);
                  const summaryTime = formatDateTimeEu(trade.exit_time ?? trade.entry_time ?? trade.time ?? null);
                  const summaryTimeCompact = formatDateTimeEuCompact(
                    trade.exit_time ?? trade.entry_time ?? trade.time ?? null
                  );
                  const durationLabel = formatTradeDuration(trade.entry_time ?? trade.time ?? null, trade.exit_time);

                  return (
                    <details
                      key={`recent-trade-${index}-${trade.entry_time ?? "na"}-${trade.exit_time ?? "na"}`}
                      className={tradeHistoryToneClass(trade.net_pnl)}
                    >
                      <summary className="trade-history-summary">
                        <span className="trade-history-summary-main">
                          <span className={`trade-history-side ${sideClass}`}>{sideLabel}</span>
                          <span className="trade-history-summary-closed-mobile">{summaryTimeCompact}</span>
                          <span className="trade-history-summary-copy">
                            <span className="trade-history-summary-title">
                              {formatExitReason(trade.exit_reason)} at {formatPrice(trade.exit)}
                            </span>
                            <span className="trade-history-summary-meta">
                              {summaryTime} • Entry {formatPrice(trade.entry)}
                            </span>
                          </span>
                        </span>
                        <span className={tradeHistoryPnlPillClass(trade.net_pnl)}>{formatSignedUsd(trade.net_pnl)}</span>
                      </summary>
                      <div className="trade-history-body">
                        <p className="dialog-scrooge trade-history-trace-note">{buildTradeNarration(trade)}</p>
                        <div className="trade-history-detail-grid">
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Entry</span>
                            <strong>{formatPrice(trade.entry)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Exit</span>
                            <strong>{formatPrice(trade.exit)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Opened</span>
                            <strong>{formatDateTimeEu(trade.entry_time ?? trade.time ?? null)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Closed</span>
                            <strong>{summaryTime}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Duration</span>
                            <strong>{durationLabel}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Size</span>
                            <strong>{formatAssetAmount(trade.size, data?.symbol)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Fee</span>
                            <strong>{formatUsd(trade.fee)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Exit Reason</span>
                            <strong>{formatExitReason(trade.exit_reason)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Safety Net</span>
                            <strong>{formatPrice(trade.sl)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Treasure Mark</span>
                            <strong>{formatPrice(trade.tp)}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Tail Guard</span>
                            <strong>{trade.trail_active ? "Armed" : "Off"}</strong>
                          </div>
                          <div className="trade-history-detail">
                            <span className="trade-history-detail-label">Outcome</span>
                            <strong className={isPositive ? "value-positive" : isNegative ? "value-negative" : "value-neutral"}>
                              {formatSignedUsd(trade.net_pnl)}
                            </strong>
                          </div>
                        </div>
                      </div>
                    </details>
                  );
                })}
                {tradeHistory.length > TRADE_HISTORY_PAGE_SIZE ? (
                  <div className="toolbar trade-history-toolbar">
                    <button
                      type="button"
                      className="dialog-user-btn trade-history-nav-button"
                      onClick={() => setTradeHistoryPage((currentPage) => Math.max(0, currentPage - 1))}
                      disabled={tradeHistoryPageIndex === 0}
                    >
                      Later
                    </button>
                    <span className="trade-history-page-indicator">
                      Showing {tradeHistoryPageIndex * TRADE_HISTORY_PAGE_SIZE + 1}-
                      {Math.min((tradeHistoryPageIndex + 1) * TRADE_HISTORY_PAGE_SIZE, tradeHistory.length)} of{" "}
                      {tradeHistoryTotal} for {TRADE_HISTORY_LOOKBACK_LABEL}
                    </span>
                    <button
                      type="button"
                      className="dialog-user-btn trade-history-nav-button"
                      onClick={() =>
                        setTradeHistoryPage((currentPage) => Math.min(tradeHistoryPageCount - 1, currentPage + 1))
                      }
                      disabled={tradeHistoryPageIndex >= tradeHistoryPageCount - 1}
                    >
                      Earlier
                    </button>
                  </div>
                ) : null}
              </div>
            )}
          </div>
          <p className="dialog-scrooge">My contract:</p>
          <div className="section-block">
            {configLoading || !editableConfig ? (
              <p className="dialog-scrooge">Reviewing the contract...</p>
            ) : contractEditing ? (
              <div className="contract-editor">
                <p className="contract-editor-note">
                  Revise the technical YAML below. The contract text above will refresh only after the office accepts the new file.
                </p>
                <label className="dialog-user-field contract-editor-field">
                  <span className="kv-label">Technical Config</span>
                  <textarea
                    value={rawConfigDraft}
                    onChange={(event) => setRawConfigDraft(event.target.value)}
                    disabled={configSaving || busyAction !== null}
                    spellCheck={false}
                  />
                </label>
              </div>
            ) : (
              <div className={`contract-scroll${contractExpanded ? " contract-scroll-open" : ""}`}>
                <button
                  type="button"
                  className="contract-scroll-toggle"
                  onClick={() => setContractExpanded((current) => !current)}
                  aria-expanded={contractExpanded}
                  aria-controls="contract-scroll-body"
                >
                  <span className="contract-scroll-toggle-copy">
                    <span className="contract-scroll-toggle-label">
                      {contractExpanded ? "Roll the parchment back up" : "Unroll the parchment"}
                    </span>
                    <span className="contract-scroll-toggle-teaser">
                      {contractExpanded
                        ? `${contractParagraphs.length} clauses are open for inspection.`
                        : contractTeaser}
                    </span>
                  </span>
                  <span className="contract-scroll-toggle-icon" aria-hidden="true">
                    ▾
                  </span>
                </button>
                <div className="contract-scroll-body-shell" id="contract-scroll-body">
                  <div className="contract-scroll-body">
                    <div className="contract-sheet" aria-label="My contract">
                      {contractParagraphs.map((paragraph, paragraphIndex) => (
                        <p key={`contract-paragraph-${paragraphIndex}`}>
                          {paragraph.map((segment, segmentIndex) =>
                            segment.kind === "value" ? (
                              <span
                                key={`contract-segment-${paragraphIndex}-${segmentIndex}`}
                                className="contract-value"
                                title={segment.path}
                              >
                                {segment.display}
                              </span>
                            ) : segment.kind === "term" ? (
                              <span key={`contract-segment-${paragraphIndex}-${segmentIndex}`} className="contract-term">
                                {segment.text}
                              </span>
                            ) : (
                              <span key={`contract-segment-${paragraphIndex}-${segmentIndex}`}>{segment.text}</span>
                            )
                          )}
                        </p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div className="toolbar config-toolbar">
              {contractEditing ? (
                <>
                  <button
                    type="button"
                    className="dialog-user-btn config-action-btn"
                    onClick={() => void saveEditableConfig()}
                    disabled={configLoading || configSaving || busyAction !== null || !editableConfig}
                  >
                    Update and Reopen Office
                  </button>
                  <button
                    type="button"
                    className="dialog-user-btn config-action-btn"
                    onClick={() => {
                      setRawConfigDraft(rawConfigText);
                      setContractEditing(false);
                      setConfigError(null);
                    }}
                    disabled={configSaving || busyAction !== null}
                  >
                    Cancel Revision
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="dialog-user-btn config-action-btn"
                  onClick={() => {
                    setRawConfigDraft(rawConfigText);
                    setContractEditing(true);
                    setContractExpanded(true);
                    setConfigError(null);
                    setConfigInfo(null);
                  }}
                  disabled={configLoading || configSaving || busyAction !== null || !editableConfig}
                >
                  Update Contract
                </button>
              )}
              {configSaving ? <span className="dialog-scrooge dialog-scrooge-compact">Updating contract...</span> : null}
            </div>
            {configError ? <p className="dialog-scrooge dialog-scrooge-error">{configError}</p> : null}
            {configInfo ? <p className="dialog-scrooge">{configInfo}</p> : null}
          </div>
        </>
      ) : null}

      {controlError ? <p className="dialog-scrooge dialog-scrooge-error">{controlError}</p> : null}

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
