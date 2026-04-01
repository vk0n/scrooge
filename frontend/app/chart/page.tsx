"use client";

import { useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";
import { parseTimestampMs, toUtcIsoString } from "../../lib/datetime";
type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type Marker = {
  time: string;
  price: number;
  side?: string;
  reason?: string;
  net_pnl?: number | null;
};

type IndicatorPoint = {
  time: string;
  value: number | null;
};

type ChartPayload = {
  symbol: string;
  interval: string;
  requested_interval?: string;
  source?: string;
  requested_source?: string;
  period: string;
  candles: Candle[];
  rsi_levels?: {
    long_tp?: number | null;
    short_tp?: number | null;
  };
  entry_rule_spec?: {
    long?: {
      rsi_open_threshold?: number | null;
    };
    short?: {
      rsi_open_threshold?: number | null;
    };
  };
  markers: {
    entries: Marker[];
    exits: Marker[];
    stop_loss: Marker[];
    take_profit: Marker[];
    liquidation: Marker[];
  };
  current_levels: Array<{ type: string; price: number }>;
  indicator_spec?: {
    ema?: {
      period?: number;
      interval?: string;
    };
    rsi?: {
      period?: number;
      interval?: string;
    };
    bollinger?: {
      period?: number;
      std_mult?: number;
      interval?: string;
    };
  };
  open_position: {
    side?: string;
    entry?: number | null;
    sl?: number | null;
    tp?: number | null;
    trail_price?: number | null;
    liq_price?: number | null;
    size?: number | null;
    time?: string | null;
    trail_active?: boolean;
  } | null;
  indicators: {
    ema?: IndicatorPoint[];
    bollinger?: {
      upper: IndicatorPoint[];
      middle: IndicatorPoint[];
      lower: IndicatorPoint[];
    };
    rsi?: IndicatorPoint[];
  };
  equity_curve: Array<{ time: string; balance: number }>;
  range_start: string | null;
  range_end: string | null;
  warnings: string[];
};

type LiveOpenTradeInfo = {
  side: "long" | "short";
  entry: number;
  sl: number | null;
  tp: number | null;
  trail_active: boolean;
  trail_price: number | null;
  entry_time: string;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  roi_pct: number | null;
};

type LiveTrailingState = {
  trail_active: boolean;
  trail_max: number | null;
  trail_min: number | null;
  trail_price: number | null;
  tp: number | null;
  sl: number | null;
};

type LiveStatusPayload = {
  symbol: string | null;
  last_price: number | null;
  last_price_updated_at?: string | null;
  last_update_timestamp: string | null;
  open_trade_info: LiveOpenTradeInfo | null;
  trailing_state: LiveTrailingState | null;
};

function normalizeTimeString(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  return toUtcIsoString(value);
}

function normalizeIndicatorPoints(points: IndicatorPoint[] | undefined): IndicatorPoint[] {
  if (!Array.isArray(points)) {
    return [];
  }
  return points
    .map((point) => ({
      ...point,
      time: normalizeTimeString(point.time) ?? point.time,
    }))
    .filter((point) => typeof point.time === "string" && point.time.length > 0);
}

function normalizeMarkers(markers: Marker[] | undefined): Marker[] {
  if (!Array.isArray(markers)) {
    return [];
  }
  return markers
    .map((marker) => ({
      ...marker,
      time: normalizeTimeString(marker.time) ?? marker.time,
    }))
    .filter((marker) => typeof marker.time === "string" && marker.time.length > 0);
}

function normalizeChartPayload(payload: ChartPayload): ChartPayload {
  return {
    ...payload,
    candles: payload.candles.map((candle) => ({
      ...candle,
      time: normalizeTimeString(candle.time) ?? candle.time,
    })),
    markers: {
      entries: normalizeMarkers(payload.markers?.entries),
      exits: normalizeMarkers(payload.markers?.exits),
      stop_loss: normalizeMarkers(payload.markers?.stop_loss),
      take_profit: normalizeMarkers(payload.markers?.take_profit),
      liquidation: normalizeMarkers(payload.markers?.liquidation),
    },
    open_position: payload.open_position
      ? {
          ...payload.open_position,
          time: normalizeTimeString(payload.open_position.time ?? null),
        }
      : null,
    indicators: {
      ema: normalizeIndicatorPoints(payload.indicators?.ema),
      bollinger: payload.indicators?.bollinger
        ? {
            upper: normalizeIndicatorPoints(payload.indicators.bollinger.upper),
            middle: normalizeIndicatorPoints(payload.indicators.bollinger.middle),
            lower: normalizeIndicatorPoints(payload.indicators.bollinger.lower),
          }
        : undefined,
      rsi: normalizeIndicatorPoints(payload.indicators?.rsi),
    },
    equity_curve: payload.equity_curve.map((point) => ({
      ...point,
      time: normalizeTimeString(point.time) ?? point.time,
    })),
    range_start: normalizeTimeString(payload.range_start),
    range_end: normalizeTimeString(payload.range_end),
  };
}

function normalizeLiveStatusPayload(payload: LiveStatusPayload): LiveStatusPayload {
  return {
    ...payload,
    last_price_updated_at: normalizeTimeString(payload.last_price_updated_at ?? null),
    last_update_timestamp: normalizeTimeString(payload.last_update_timestamp),
    open_trade_info: payload.open_trade_info
      ? {
          ...payload.open_trade_info,
          entry_time: normalizeTimeString(payload.open_trade_info.entry_time) ?? payload.open_trade_info.entry_time,
        }
      : null,
  };
}

type ChartLegendItem = {
  key: string;
  label: string;
  swatchClassName: string;
};

const PERIOD_OPTIONS = ["15m", "30m", "1h", "2h", "3h", "4h", "6h", "12h", "1d", "3d", "1w", "2w", "4w", "12w", "26w", "52w"];
const INTERVAL_OPTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"];
const SOURCE_OPTIONS = ["auto", "dataset", "binance"] as const;
const POLL_MS = 60000;
const WS_RECONNECT_MS = 5000;
const CHART_THEME = {
  upCandle: "#34d399",
  downCandle: "#f87171",
  longEntry: "#22c55e",
  shortEntry: "#ef4444",
  neutralMarker: "#9ca3af",
  exitOutline: "#e5e7eb",
  openPosition: "#ffb020",
  ema: "#38bdf8",
  bbUpperLower: "#a78bfa",
  bbMiddle: "#c4b5fd",
  slLevel: "#fb7185",
  tpLevel: "#4ade80",
  equity: "#34d399",
  rsi: "#f59e0b",
  bg: "#0d141d",
  text: "#e6edf6",
  mutedText: "#9db0c8",
  rsiUpper: "#ef4444",
  rsiLower: "#22c55e",
  livePrice: "#d8dee8",
  livePricePositive: "#34d399",
  livePriceNegative: "#fb7185",
} as const;

function formatSignedUsd(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "PnL N/A";
  }
  const absValue = Math.abs(value);
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(absValue);
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `PnL ${sign}$${formatted}`;
}

function formatSignedPercent(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const absValue = Math.abs(value);
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(absValue);
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${formatted}%`;
}

function resolveLivePriceTone(unrealizedPnl: number | null | undefined): "neutral" | "positive" | "negative" {
  if (typeof unrealizedPnl !== "number" || !Number.isFinite(unrealizedPnl) || unrealizedPnl === 0) {
    return "neutral";
  }
  return unrealizedPnl > 0 ? "positive" : "negative";
}

function resolveLivePriceColor(tone: "neutral" | "positive" | "negative"): string {
  if (tone === "positive") {
    return CHART_THEME.livePricePositive;
  }
  if (tone === "negative") {
    return CHART_THEME.livePriceNegative;
  }
  return CHART_THEME.livePrice;
}

function getTradeTriangleSymbol(side?: string | null): "triangle-up" | "triangle-down" {
  return String(side || "").trim().toLowerCase() === "short" ? "triangle-down" : "triangle-up";
}

function buildTradeEntryMarker(side: string | null | undefined, size: number): {
  color: string;
  symbol: "triangle-up" | "triangle-down";
  size: number;
} {
  return {
    color: CHART_THEME.openPosition,
    symbol: getTradeTriangleSymbol(side),
    size,
  };
}

function buildCurrentLevelShapes(
  levels: Array<{ type: string; price: number }>,
  currentPrice: number | null,
  currentPriceColor: string = CHART_THEME.livePrice
): Array<Record<string, unknown>> {
  const shapes: Array<Record<string, unknown>> = levels.map((level) => ({
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: level.price,
    y1: level.price,
    line: {
      color: level.type === "sl" ? CHART_THEME.slLevel : CHART_THEME.tpLevel,
      width: level.type === "trail" ? 1.25 : 1,
      dash: level.type === "sl" ? "dot" : level.type === "trail" ? "dashdot" : "dash",
    },
  }));

  if (typeof currentPrice === "number" && Number.isFinite(currentPrice)) {
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: currentPrice,
      y1: currentPrice,
      line: {
        color: currentPriceColor,
        width: 1,
        dash: "dot",
      },
    });
  }

  return shapes;
}

function buildLivePriceAnnotations(
  currentPrice: number | null,
  openTrade: LiveOpenTradeInfo | null | undefined,
  chartData?: ChartPayload | null
): Array<Record<string, unknown>> {
  if (typeof currentPrice !== "number" || !Number.isFinite(currentPrice)) {
    return [];
  }

  if (!openTrade && chartData) {
    const latestEma = findLatestIndicatorValue(chartData.indicators?.ema);
    const latestRsi = findLatestIndicatorValue(chartData.indicators?.rsi);
    const latestBbl = findLatestIndicatorValue(chartData.indicators?.bollinger?.lower);
    const latestBbu = findLatestIndicatorValue(chartData.indicators?.bollinger?.upper);
    const longRsiOpenThreshold = chartData.entry_rule_spec?.long?.rsi_open_threshold;
    const shortRsiOpenThreshold = chartData.entry_rule_spec?.short?.rsi_open_threshold;

    if (
      typeof latestEma !== "number" ||
      !Number.isFinite(latestEma) ||
      typeof latestRsi !== "number" ||
      !Number.isFinite(latestRsi)
    ) {
      return [];
    }

    const crossedLowerBand =
      typeof latestBbl === "number" && Number.isFinite(latestBbl) && currentPrice < latestBbl;
    const crossedUpperBand =
      typeof latestBbu === "number" && Number.isFinite(latestBbu) && currentPrice > latestBbu;
    const activeSignalSide: "long" | "short" | null = crossedLowerBand
      ? "long"
      : crossedUpperBand
        ? "short"
        : null;
    const directionalSide: "long" | "short" = currentPrice >= latestEma ? "long" : "short";
    const sideContext = activeSignalSide ?? directionalSide;
    const bandOk = activeSignalSide !== null;
    const emaOk =
      activeSignalSide === null ? true : sideContext === "long" ? currentPrice >= latestEma : currentPrice < latestEma;
    const rsiThreshold = sideContext === "long" ? longRsiOpenThreshold : shortRsiOpenThreshold;
    const rsiOk =
      typeof rsiThreshold === "number" &&
      Number.isFinite(rsiThreshold) &&
      (sideContext === "long" ? latestRsi < rsiThreshold : latestRsi > rsiThreshold);

    return [
      {
        xref: "paper",
        x: 0,
        xanchor: "left",
        xshift: 10,
        yref: "y",
        y: currentPrice,
        yanchor: "middle",
        text: "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        showarrow: false,
        align: "left",
        font: {
          color: "rgba(15, 23, 42, 0)",
          size: 11,
        },
        bordercolor: CHART_THEME.livePrice,
        borderwidth: 1.2,
        borderpad: 4,
        bgcolor: "rgba(244, 248, 252, 0.95)",
      },
      {
        xref: "paper",
        x: 0,
        xanchor: "left",
        xshift: 19,
        yref: "y",
        y: currentPrice,
        yanchor: "middle",
        text: "BB",
        showarrow: false,
        align: "center",
        font: {
          color: bandOk ? "#0b1220" : "#fff7f7",
          size: 11,
        },
        bordercolor: "rgba(255, 255, 255, 0.18)",
        borderwidth: 0.8,
        borderpad: 2,
        bgcolor: bandOk ? "#84cc16" : "#ef4444",
      },
      {
        xref: "paper",
        x: 0,
        xanchor: "left",
        xshift: 55,
        yref: "y",
        y: currentPrice,
        yanchor: "middle",
        text: "EMA",
        showarrow: false,
        align: "center",
        font: {
          color: emaOk ? "#0b1220" : "#fff7f7",
          size: 11,
        },
        bordercolor: "rgba(255, 255, 255, 0.18)",
        borderwidth: 0.8,
        borderpad: 2,
        bgcolor: emaOk ? "#84cc16" : "#ef4444",
      },
      {
        xref: "paper",
        x: 0,
        xanchor: "left",
        xshift: 91,
        yref: "y",
        y: currentPrice,
        yanchor: "middle",
        text: "RSI",
        showarrow: false,
        align: "center",
        font: {
          color: rsiOk ? "#0b1220" : "#fff7f7",
          size: 11,
        },
        bordercolor: "rgba(255, 255, 255, 0.18)",
        borderwidth: 0.8,
        borderpad: 2,
        bgcolor: rsiOk ? "#84cc16" : "#ef4444",
      },
    ];
  }

  if (!openTrade) {
    return [];
  }

  const tone = resolveLivePriceTone(openTrade.unrealized_pnl);
  const color = resolveLivePriceColor(tone);
  const pnlText = formatSignedUsd(openTrade.unrealized_pnl);
  const roiText = formatSignedPercent(openTrade.roi_pct ?? openTrade.unrealized_pnl_pct);
  const text = roiText ? `${pnlText}  ROI ${roiText}` : pnlText;

  return [
    {
      xref: "paper",
      x: 0,
      xanchor: "left",
      xshift: 10,
      yref: "y",
      y: currentPrice,
      yanchor: "middle",
      text,
      showarrow: false,
      align: "left",
      font: {
        color: tone === "neutral" ? CHART_THEME.text : "#f4fbff",
        size: 11,
      },
      bordercolor: color,
      borderwidth: 1,
      borderpad: 5,
      bgcolor:
        tone === "positive"
          ? "rgba(20, 62, 44, 0.92)"
          : tone === "negative"
            ? "rgba(77, 24, 30, 0.92)"
            : "rgba(25, 32, 44, 0.9)",
    },
  ];
}

function findLatestIndicatorValue(points: IndicatorPoint[] | undefined): number | null {
  if (!points?.length) {
    return null;
  }

  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = points[index]?.value;
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return null;
}

function resolveLiveLevels(data: ChartPayload | null, liveStatus: LiveStatusPayload | null): Array<{ type: string; price: number }> | null {
  if (!data || !liveStatus) {
    return null;
  }

  const liveSymbol = liveStatus.symbol?.trim().toUpperCase();
  if (!liveSymbol || liveSymbol !== data.symbol.trim().toUpperCase()) {
    return null;
  }

  const openTrade = liveStatus.open_trade_info;
  if (!openTrade) {
    return [];
  }

  const levels: Array<{ type: string; price: number }> = [];
  if (typeof openTrade.sl === "number" && Number.isFinite(openTrade.sl)) {
    levels.push({ type: "sl", price: openTrade.sl });
  }

  if (openTrade.trail_active) {
    const liveTrail = typeof liveStatus.trailing_state?.trail_price === "number" && Number.isFinite(liveStatus.trailing_state.trail_price)
      ? liveStatus.trailing_state.trail_price
      : openTrade.trail_price;
    if (typeof liveTrail === "number" && Number.isFinite(liveTrail)) {
      levels.push({ type: "trail", price: liveTrail });
    }
  } else if (typeof openTrade.tp === "number" && Number.isFinite(openTrade.tp)) {
    levels.push({ type: "tp", price: openTrade.tp });
  }

  return levels;
}

function parsePeriodMs(period: string): number {
  const match = /^(\d+)([mhdw])$/i.exec(period.trim());
  if (!match) {
    return 24 * 60 * 60 * 1000;
  }
  const count = Number(match[1]);
  const unit = match[2].toLowerCase();
  const unitToMs: Record<string, number> = {
    m: 60 * 1000,
    h: 60 * 60 * 1000,
    d: 24 * 60 * 60 * 1000,
    w: 7 * 24 * 60 * 60 * 1000,
  };
  return count * (unitToMs[unit] ?? unitToMs.d);
}

function clampVisibleXRange(
  range: [number, number] | null,
  minMs: number,
  maxMs: number
): [number, number] | null {
  if (!range) {
    return null;
  }

  const overlapStart = Math.max(range[0], minMs);
  const overlapEnd = Math.min(range[1], maxMs);
  if (!Number.isFinite(overlapStart) || !Number.isFinite(overlapEnd) || overlapEnd <= overlapStart) {
    return null;
  }

  return [overlapStart, overlapEnd];
}

function alignTimestampToStep(startMs: number, stepMs: number): number {
  if (!Number.isFinite(startMs) || !Number.isFinite(stepMs) || stepMs <= 0) {
    return startMs;
  }
  return Math.floor(startMs / stepMs) * stepMs;
}

function resolveSharedDateTickMs(startMs: number, endMs: number): number {
  const spanMs = endMs - startMs;
  const hour = 60 * 60 * 1000;
  const day = 24 * hour;

  if (spanMs <= 12 * hour) {
    return hour;
  }
  if (spanMs <= 2 * day) {
    return 3 * hour;
  }
  if (spanMs <= 7 * day) {
    return 12 * hour;
  }
  if (spanMs <= 31 * day) {
    return day;
  }
  if (spanMs <= 90 * day) {
    return 3 * day;
  }
  if (spanMs <= 365 * day) {
    return 14 * day;
  }
  return 30 * day;
}

function buildSharedDateAxisSettings(
  startMs: number | null,
  endMs: number | null
): { tick0?: string; dtick?: number } {
  if (startMs === null || endMs === null || !Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
    return {};
  }

  const dtick = resolveSharedDateTickMs(startMs, endMs);
  return {
    tick0: new Date(alignTimestampToStep(startMs, dtick)).toISOString(),
    dtick,
  };
}

function pointsToXY(points: IndicatorPoint[]): { x: string[]; y: Array<number | null> } {
  return {
    x: points.map((point) => point.time),
    y: points.map((point) => point.value),
  };
}

function collectIndicatorValues(points: IndicatorPoint[] | undefined): Array<{ tsMs: number; value: number }> {
  if (!points?.length) {
    return [];
  }

  return points.flatMap((point) => {
    const tsMs = toTimestampMs(point.time);
    if (tsMs === null || typeof point.value !== "number" || !Number.isFinite(point.value)) {
      return [];
    }
    return [{ tsMs, value: point.value }];
  });
}

type PlotlyRelayoutEvent = Record<string, unknown>;
type PlotlyEventTarget = HTMLDivElement & {
  on?: (event: string, callback: (event: PlotlyRelayoutEvent) => void) => void;
  removeAllListeners?: (event: string) => void;
};

function toTimestampMs(value: unknown): number | null {
  return parseTimestampMs(value);
}

function parseRelayoutXRange(eventData: PlotlyRelayoutEvent): [number, number] | null {
  const directStart = toTimestampMs(eventData["xaxis.range[0]"]);
  const directEnd = toTimestampMs(eventData["xaxis.range[1]"]);
  if (directStart !== null && directEnd !== null) {
    return [directStart, directEnd];
  }

  const rawRange = eventData["xaxis.range"];
  if (!Array.isArray(rawRange) || rawRange.length < 2) {
    return null;
  }
  const arrayStart = toTimestampMs(rawRange[0]);
  const arrayEnd = toTimestampMs(rawRange[1]);
  if (arrayStart === null || arrayEnd === null) {
    return null;
  }
  return [arrayStart, arrayEnd];
}

function buildPriceLegendItems(
  data: ChartPayload | null,
  includeIndicators: boolean
): ChartLegendItem[] {
  if (!data) {
    return [];
  }

  const items: ChartLegendItem[] = [{ key: "price", label: "Price", swatchClassName: "chart-legend-swatch-price" }];

  const hasOpenPosition =
    data.open_position?.entry !== null && data.open_position?.entry !== undefined && Boolean(data.open_position?.time);
  const hasTradeEntryMarkers = Array.isArray(data.markers?.entries) && data.markers.entries.length > 0;

  if (hasOpenPosition || hasTradeEntryMarkers) {
    items.push({ key: "open-trade", label: "Trade Entries", swatchClassName: "chart-legend-swatch-open-trade" });
  }

  if (includeIndicators && data.indicators?.ema?.length) {
    const emaPeriod = data.indicator_spec?.ema?.period ?? 50;
    const emaInterval = data.indicator_spec?.ema?.interval;
    items.push({
      key: "ema",
      label: emaInterval ? `EMA(${emaPeriod}, ${emaInterval})` : `EMA(${emaPeriod})`,
      swatchClassName: "chart-legend-swatch-ema",
    });
  }

  if (includeIndicators && data.indicators?.bollinger) {
    items.push(
      { key: "bb-upper", label: "BB Upper", swatchClassName: "chart-legend-swatch-bb-upper" },
      { key: "bb-middle", label: "BB Middle", swatchClassName: "chart-legend-swatch-bb-middle" },
      { key: "bb-lower", label: "BB Lower", swatchClassName: "chart-legend-swatch-bb-lower" }
    );
  }

  return items;
}

function ChartContent(): JSX.Element {
  const priceChartRef = useRef<HTMLDivElement | null>(null);
  const equityChartRef = useRef<HTMLDivElement | null>(null);
  const rsiChartRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<any>(null);
  const plotlyImportPromiseRef = useRef<Promise<any> | null>(null);
  const visibleXRangeRef = useRef<[number, number] | null>(null);

  const [symbol, setSymbol] = useState<string>("BTCUSDT");
  const [period, setPeriod] = useState<string>("1d");
  const [interval, setInterval] = useState<string>("1m");
  const [source, setSource] = useState<string>("auto");
  const [includeIndicators, setIncludeIndicators] = useState<boolean>(true);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [endCursorMs, setEndCursorMs] = useState<number | null>(null);
  const [controlsExpanded, setControlsExpanded] = useState<boolean>(true);
  const [chartsExpanded, setChartsExpanded] = useState<boolean>(false);
  const [compactChartUi, setCompactChartUi] = useState<boolean>(false);

  const [data, setData] = useState<ChartPayload | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatusPayload | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const priceLegendItems = buildPriceLegendItems(data, includeIndicators);

  async function ensurePlotly(): Promise<any> {
    if (plotlyRef.current) {
      return plotlyRef.current;
    }
    if (!plotlyImportPromiseRef.current) {
      plotlyImportPromiseRef.current = import("plotly.js-dist-min").then((module) => {
        plotlyRef.current = module.default;
        return module.default;
      });
    }
    return plotlyImportPromiseRef.current;
  }

  async function loadChart(silent = false): Promise<void> {
    const safeSymbol = symbol.trim().toUpperCase() || "BTCUSDT";
    if (!silent) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    try {
      const query = new URLSearchParams({
        symbol: safeSymbol,
        period,
        interval,
        source,
        indicators: String(includeIndicators),
      });
      if (endCursorMs !== null) {
        query.set("end", new Date(endCursorMs).toISOString());
      }
      const payload = await fetchApi<ChartPayload>(`/api/chart?${query.toString()}`);
      setData(normalizeChartPayload(payload));
      setSymbol(payload.symbol);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load chart");
    } finally {
      if (!silent) {
        setLoading(false);
      }
      setRefreshing(false);
    }
  }

  useEffect((): (() => void) => {
    void loadChart(false);
    if (!autoRefresh) {
      return () => undefined;
    }
    const intervalId = window.setInterval(() => {
      void loadChart(true);
    }, POLL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [autoRefresh, period, interval, source, includeIndicators, endCursorMs]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    visibleXRangeRef.current = null;
  }, [symbol, period, interval, source, endCursorMs]);

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
          const payload = JSON.parse(event.data) as { type?: string; data?: LiveStatusPayload };
          if (payload.type === "status" && payload.data) {
            setLiveStatus(normalizeLiveStatusPayload(payload.data));
          }
        } catch {
          // Ignore malformed WS payloads. Polling still refreshes chart data.
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

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const viewportMedia = window.matchMedia("(max-width: 760px)");
    const coarsePointerMedia = window.matchMedia("(hover: none), (pointer: coarse)");

    const applyViewportMode = (): void => {
      const isCompact = viewportMedia.matches || coarsePointerMedia.matches;
      setCompactChartUi(isCompact);
      if (viewportMedia.matches) {
        setControlsExpanded(false);
      }
    };

    applyViewportMode();

    const handleViewportChange = (): void => applyViewportMode();

    viewportMedia.addEventListener("change", handleViewportChange);
    coarsePointerMedia.addEventListener("change", handleViewportChange);

    return () => {
      viewportMedia.removeEventListener("change", handleViewportChange);
      coarsePointerMedia.removeEventListener("change", handleViewportChange);
    };
  }, []);

  useEffect(() => {
    document.body.classList.toggle("chart-fullscreen-active", chartsExpanded);
    return () => {
      document.body.classList.remove("chart-fullscreen-active");
    };
  }, [chartsExpanded]);

  useEffect(() => {
    if (!chartsExpanded) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setChartsExpanded(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [chartsExpanded]);

  useEffect(() => {
    const resizeTimer = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 120);
    return () => {
      window.clearTimeout(resizeTimer);
    };
  }, [chartsExpanded]);

  useEffect(() => {
    let cancelled = false;

    const renderCharts = async (): Promise<void> => {
      if (!data) {
        return;
      }
      const Plotly = await ensurePlotly();
      if (cancelled) {
        return;
      }

      const candleX = data.candles.map((candle) => candle.time);
      const candleOpen = data.candles.map((candle) => candle.open);
      const candleHigh = data.candles.map((candle) => candle.high);
      const candleLow = data.candles.map((candle) => candle.low);
      const candleClose = data.candles.map((candle) => candle.close);
      const candleTimesMs = data.candles.map((candle) => toTimestampMs(candle.time) ?? Number.NaN);
      const chartStartMs = candleTimesMs.find((value) => Number.isFinite(value)) ?? null;
      const chartEndMs = [...candleTimesMs].reverse().find((value) => Number.isFinite(value)) ?? null;
      const boundedVisibleRange =
        chartStartMs !== null && chartEndMs !== null
          ? clampVisibleXRange(visibleXRangeRef.current, chartStartMs, chartEndMs)
          : null;
      visibleXRangeRef.current = boundedVisibleRange;

      const chartRevisionKey = `${data.symbol}:${data.interval}:${period}:${source}:${endCursorMs ?? "live"}`;
      const priceShapes = buildCurrentLevelShapes(data.current_levels, null);
      const initialXRange = boundedVisibleRange
        ? [
            new Date(boundedVisibleRange[0]).toISOString(),
            new Date(boundedVisibleRange[1]).toISOString(),
          ]
        : undefined;
      const activeRangeStartMs = boundedVisibleRange?.[0] ?? chartStartMs;
      const activeRangeEndMs = boundedVisibleRange?.[1] ?? chartEndMs;
      const sharedDateAxisSettings = buildSharedDateAxisSettings(activeRangeStartMs, activeRangeEndMs);
      const sharedChartMargins = {
        t: compactChartUi ? 16 : 18,
        r: compactChartUi ? 12 : 16,
        b: compactChartUi ? 28 : 40,
        l: compactChartUi ? 46 : 55,
        autoexpand: false,
      };
      const plotConfig = {
        responsive: true,
        displaylogo: false,
        displayModeBar: compactChartUi ? false : "hover",
        scrollZoom: !compactChartUi,
        doubleClick: "reset+autosize",
      };
      const openPositionTimeMs = toTimestampMs(data.open_position?.time);
      const openPositionInRange =
        chartStartMs !== null &&
        chartEndMs !== null &&
        typeof openPositionTimeMs === "number" &&
        Number.isFinite(openPositionTimeMs) &&
        openPositionTimeMs >= chartStartMs &&
        openPositionTimeMs <= chartEndMs;

      const traces: Array<Record<string, unknown>> = [
        {
          type: "candlestick",
          name: "Price",
          x: candleX,
          open: candleOpen,
          high: candleHigh,
          low: candleLow,
          close: candleClose,
          increasing: { line: { color: CHART_THEME.upCandle } },
          decreasing: { line: { color: CHART_THEME.downCandle } },
        },
      ];

      const longEntries = data.markers.entries.filter((marker) => marker.side?.toLowerCase() === "long");
      const shortEntries = data.markers.entries.filter((marker) => marker.side?.toLowerCase() === "short");

      if (longEntries.length) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "Long Entries",
          x: longEntries.map((marker) => marker.time),
          y: longEntries.map((marker) => marker.price),
          marker: buildTradeEntryMarker("long", 10),
        });
      }
      if (shortEntries.length) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "Short Entries",
          x: shortEntries.map((marker) => marker.time),
          y: shortEntries.map((marker) => marker.price),
          marker: buildTradeEntryMarker("short", 10),
        });
      }
      if (data.markers.exits.length) {
        const exitColors = data.markers.exits.map((marker) => {
          if (typeof marker.net_pnl !== "number") {
            return CHART_THEME.neutralMarker;
          }
          return marker.net_pnl >= 0 ? CHART_THEME.longEntry : CHART_THEME.shortEntry;
        });
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "Exits",
          x: data.markers.exits.map((marker) => marker.time),
          y: data.markers.exits.map((marker) => marker.price),
          marker: {
            color: exitColors,
            symbol: "x",
            size: 10,
            line: { width: 1.1, color: CHART_THEME.exitOutline },
          },
        });
      }

      if (
        openPositionInRange &&
        data.open_position?.entry !== null &&
        data.open_position?.entry !== undefined &&
        data.open_position?.time
      ) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "Open Trade",
          x: [data.open_position.time],
          y: [data.open_position.entry],
          marker: buildTradeEntryMarker(data.open_position.side, 11),
        });
      }

      if (includeIndicators && data.indicators?.ema?.length) {
        const ema = pointsToXY(data.indicators.ema);
        const emaPeriod = data.indicator_spec?.ema?.period ?? 50;
        const emaInterval = data.indicator_spec?.ema?.interval;
        const emaLabel = emaInterval ? `EMA(${emaPeriod}, ${emaInterval})` : `EMA(${emaPeriod})`;
        traces.push({
          type: "scatter",
          mode: "lines",
          name: emaLabel,
          x: ema.x,
          y: ema.y,
          line: { color: CHART_THEME.ema, width: 1.4 },
        });
      }

      if (includeIndicators && data.indicators?.bollinger) {
        const upper = pointsToXY(data.indicators.bollinger.upper);
        const middle = pointsToXY(data.indicators.bollinger.middle);
        const lower = pointsToXY(data.indicators.bollinger.lower);
        traces.push(
          {
            type: "scatter",
            mode: "lines",
            name: "BB Upper",
            x: upper.x,
            y: upper.y,
            line: { color: CHART_THEME.bbUpperLower, width: 1, dash: "dot" },
          },
          {
            type: "scatter",
            mode: "lines",
            name: "BB Middle",
            x: middle.x,
            y: middle.y,
            line: { color: CHART_THEME.bbMiddle, width: 1, dash: "dash" },
          },
          {
            type: "scatter",
            mode: "lines",
            name: "BB Lower",
            x: lower.x,
            y: lower.y,
            line: { color: CHART_THEME.bbUpperLower, width: 1, dash: "dot" },
          }
        );
      }

      if (priceChartRef.current) {
        await Plotly.react(
          priceChartRef.current,
          traces,
          {
            paper_bgcolor: CHART_THEME.bg,
            plot_bgcolor: CHART_THEME.bg,
            font: { color: CHART_THEME.text },
            uirevision: chartRevisionKey,
            hovermode: "x unified",
            xaxis: {
              type: "date",
              rangeslider: { visible: !compactChartUi },
              range: initialXRange,
              automargin: false,
              ...sharedDateAxisSettings,
            },
            yaxis: { title: "Price", automargin: false },
            showlegend: false,
            margin: sharedChartMargins,
            shapes: priceShapes,
          },
          plotConfig
        );

        const chartEl = priceChartRef.current as PlotlyEventTarget;
        chartEl.removeAllListeners?.("plotly_relayout");

        const currentLevelPrices = data.current_levels
          .map((level) => level.price)
          .filter((price): price is number => Number.isFinite(price));
        const visibleIndicatorValues = includeIndicators
          ? [
              ...collectIndicatorValues(data.indicators?.ema),
              ...collectIndicatorValues(data.indicators?.bollinger?.upper),
              ...collectIndicatorValues(data.indicators?.bollinger?.middle),
              ...collectIndicatorValues(data.indicators?.bollinger?.lower),
            ]
          : [];
        let relayoutInProgress = false;

        const syncAuxiliaryXRange = (range: [number, number] | null): void => {
          const targets = [equityChartRef.current, rsiChartRef.current].filter(
            (node): node is HTMLDivElement => node !== null
          );
          if (!targets.length) {
            return;
          }
          const syncedRange = range ?? (chartStartMs !== null && chartEndMs !== null ? [chartStartMs, chartEndMs] : null);
          const syncedAxisSettings = buildSharedDateAxisSettings(syncedRange?.[0] ?? null, syncedRange?.[1] ?? null);
          const payload = syncedRange
            ? {
                "xaxis.range": [new Date(syncedRange[0]).toISOString(), new Date(syncedRange[1]).toISOString()],
                ...(syncedAxisSettings.tick0 ? { "xaxis.tick0": syncedAxisSettings.tick0 } : {}),
                ...(syncedAxisSettings.dtick ? { "xaxis.dtick": syncedAxisSettings.dtick } : {}),
              }
            : { "xaxis.autorange": true };
          targets.forEach((target) => {
            void Plotly.relayout(target, payload);
          });
        };

        const rescaleVisibleY = (startMs: number, endMs: number): void => {
          let minPrice = Number.POSITIVE_INFINITY;
          let maxPrice = Number.NEGATIVE_INFINITY;
          for (let index = 0; index < candleTimesMs.length; index += 1) {
            const ts = candleTimesMs[index];
            if (!Number.isFinite(ts) || ts < startMs || ts > endMs) {
              continue;
            }
            minPrice = Math.min(minPrice, candleLow[index]);
            maxPrice = Math.max(maxPrice, candleHigh[index]);
          }
          visibleIndicatorValues.forEach(({ tsMs, value }) => {
            if (tsMs < startMs || tsMs > endMs) {
              return;
            }
            minPrice = Math.min(minPrice, value);
            maxPrice = Math.max(maxPrice, value);
          });
          currentLevelPrices.forEach((price) => {
            minPrice = Math.min(minPrice, price);
            maxPrice = Math.max(maxPrice, price);
          });
          if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) {
            return;
          }
          const span = maxPrice - minPrice;
          const padding = span > 0 ? span * 0.03 : Math.max(Math.abs(maxPrice) * 0.003, 1);
          relayoutInProgress = true;
          void Plotly.relayout(chartEl, { "yaxis.range": [minPrice - padding, maxPrice + padding] }).finally(() => {
            relayoutInProgress = false;
          });
        };

        chartEl.on?.("plotly_relayout", (eventData: PlotlyRelayoutEvent) => {
          if (relayoutInProgress) {
            return;
          }
          if (eventData["xaxis.autorange"] === true) {
            visibleXRangeRef.current = null;
            syncAuxiliaryXRange(null);
            if (candleTimesMs.length) {
              rescaleVisibleY(candleTimesMs[0], candleTimesMs[candleTimesMs.length - 1]);
            }
            return;
          }

          const range = parseRelayoutXRange(eventData);
          if (!range) {
            return;
          }
          visibleXRangeRef.current = range;
          rescaleVisibleY(range[0], range[1]);
          syncAuxiliaryXRange(range);
        });

        if (candleTimesMs.length) {
          const initialRange = visibleXRangeRef.current ?? [candleTimesMs[0], candleTimesMs[candleTimesMs.length - 1]];
          rescaleVisibleY(initialRange[0], initialRange[1]);
        }
      }

      if (equityChartRef.current) {
        await Plotly.react(
          equityChartRef.current,
          [
            {
              type: "scatter",
              mode: "lines",
              name: "Equity",
              x: data.equity_curve.map((point) => point.time),
              y: data.equity_curve.map((point) => point.balance),
              line: { color: CHART_THEME.equity, width: 1.8, shape: "hv" },
            },
          ],
          {
            paper_bgcolor: CHART_THEME.bg,
            plot_bgcolor: CHART_THEME.bg,
            font: { color: CHART_THEME.text },
            uirevision: `${chartRevisionKey}:equity`,
            hovermode: "x unified",
            xaxis: {
              type: "date",
              range: initialXRange,
              automargin: false,
              ...sharedDateAxisSettings,
            },
            yaxis: { title: "Vault", automargin: false },
            margin: sharedChartMargins,
          },
          plotConfig
        );
      }

      if (rsiChartRef.current) {
        const rsiPoints = data.indicators?.rsi ?? [];
        const hasRsi = includeIndicators && rsiPoints.length > 0;
        if (hasRsi) {
          const rsi = pointsToXY(rsiPoints);
          const rsiLongTp = data.rsi_levels?.long_tp;
          const rsiShortTp = data.rsi_levels?.short_tp;
          const rsiPeriod = data.indicator_spec?.rsi?.period ?? 11;
          const rsiInterval = data.indicator_spec?.rsi?.interval;
          const rsiLabel = rsiInterval ? `RSI (${rsiPeriod}, ${rsiInterval})` : `RSI (${rsiPeriod})`;
          const rsiShapes = [
            rsiLongTp === null || rsiLongTp === undefined
              ? null
              : {
                  type: "line",
                  xref: "paper",
                  x0: 0,
                  x1: 1,
                  y0: rsiLongTp,
                  y1: rsiLongTp,
                  line: { color: CHART_THEME.rsiUpper, dash: "dot", width: 1 },
                },
            rsiShortTp === null || rsiShortTp === undefined
              ? null
              : {
                  type: "line",
                  xref: "paper",
                  x0: 0,
                  x1: 1,
                  y0: rsiShortTp,
                  y1: rsiShortTp,
                  line: { color: CHART_THEME.rsiLower, dash: "dot", width: 1 },
                },
          ].filter(Boolean);
          await Plotly.react(
            rsiChartRef.current,
            [
              {
                type: "scatter",
                mode: "lines",
                name: rsiLabel,
                x: rsi.x,
                y: rsi.y,
                line: { color: CHART_THEME.rsi, width: 1.4 },
              },
            ],
            {
              paper_bgcolor: CHART_THEME.bg,
              plot_bgcolor: CHART_THEME.bg,
              font: { color: CHART_THEME.text },
              uirevision: `${chartRevisionKey}:rsi`,
              hovermode: "x unified",
              xaxis: {
                type: "date",
                range: initialXRange,
                automargin: false,
                ...sharedDateAxisSettings,
              },
              yaxis: { title: "RSI", range: [0, 100], automargin: false },
              margin: sharedChartMargins,
              shapes: rsiShapes,
            },
            plotConfig
          );
        } else {
          await Plotly.react(
            rsiChartRef.current,
            [],
            {
              paper_bgcolor: CHART_THEME.bg,
              plot_bgcolor: CHART_THEME.bg,
              font: { color: CHART_THEME.mutedText },
              uirevision: `${chartRevisionKey}:rsi-empty`,
              margin: {
                ...sharedChartMargins,
                b: compactChartUi ? 24 : 30,
              },
            },
            plotConfig
          );
        }
      }
    };

    void renderCharts();

    return () => {
      cancelled = true;
    };
  }, [compactChartUi, data, includeIndicators, period, source, endCursorMs]);

  useEffect(() => {
    let cancelled = false;

    const applyLiveOverlay = async (): Promise<void> => {
      if (!data || !priceChartRef.current) {
        return;
      }

      const Plotly = await ensurePlotly();
      if (cancelled) {
        return;
      }

      const liveSymbol = liveStatus?.symbol?.trim().toUpperCase();
      const chartSymbol = data.symbol.trim().toUpperCase();
      const liveSymbolMatchesChart = liveSymbol === chartSymbol;
      const liveCurrentPrice = liveSymbolMatchesChart ? liveStatus?.last_price ?? null : null;
      const liveOpenTrade = liveSymbolMatchesChart ? liveStatus?.open_trade_info ?? null : null;
      const priceTone = resolveLivePriceTone(liveOpenTrade?.unrealized_pnl);
      const priceShapes = buildCurrentLevelShapes(
        resolveLiveLevels(data, liveStatus) ?? data.current_levels,
        liveCurrentPrice,
        resolveLivePriceColor(priceTone)
      );
      const priceAnnotations = buildLivePriceAnnotations(liveCurrentPrice, liveOpenTrade, data);
      await Plotly.relayout(priceChartRef.current, {
        shapes: priceShapes,
        annotations: priceAnnotations,
      });
    };

    void applyLiveOverlay();

    return () => {
      cancelled = true;
    };
  }, [data, liveStatus]);

  return (
    <section className="panel page-shell chart-page-shell">
      <p className="dialog-scrooge">Read-only map of candles, trades, and vault curve.</p>

      <details
        className="chart-controls-accordion"
        open={controlsExpanded}
        onToggle={(event) => setControlsExpanded(event.currentTarget.open)}
      >
        <summary className="chart-controls-summary">
          <span className="chart-controls-title">Scout controls</span>
          <span className="chart-controls-summary-hint">{controlsExpanded ? "Hide" : "Show"}</span>
        </summary>
        <div className="chart-controls-body">
          <div className="form-grid chart-controls-grid">
            <label htmlFor="chart-symbol" className="field-stack dialog-user-field chart-control-field">
              Pair
              <input
                id="chart-symbol"
                type="text"
                value={symbol}
                onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              />
            </label>

            <label htmlFor="chart-period" className="field-stack dialog-user-field chart-control-field">
              Window
              <select
                id="chart-period"
                value={period}
                onChange={(event) => setPeriod(event.target.value)}
              >
                {PERIOD_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>

            <label htmlFor="chart-interval" className="field-stack dialog-user-field chart-control-field">
              Candle Step
              <select
                id="chart-interval"
                value={interval}
                onChange={(event) => setInterval(event.target.value)}
              >
                {INTERVAL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>

            <label htmlFor="chart-source" className="field-stack dialog-user-field chart-control-field">
              Feed
              <select
                id="chart-source"
                value={source}
                onChange={(event) => setSource(event.target.value)}
              >
                {SOURCE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="chart-toolbar">
            <div className="chart-toolbar-group chart-toolbar-group-toggles">
              <label htmlFor="chart-indicators" className="field-inline dialog-user-toggle chart-toolbar-toggle">
                <input
                  id="chart-indicators"
                  type="checkbox"
                  className="dialog-user-check"
                  checked={includeIndicators}
                  onChange={(event) => setIncludeIndicators(event.target.checked)}
                />
                Show Indicators
              </label>

              <label htmlFor="chart-autorefresh" className="field-inline dialog-user-toggle chart-toolbar-toggle">
                <input
                  id="chart-autorefresh"
                  type="checkbox"
                  className="dialog-user-check"
                  checked={autoRefresh}
                  onChange={(event) => setAutoRefresh(event.target.checked)}
                />
                Auto Scout ({POLL_MS / 1000}s)
              </label>
            </div>

            <div className="chart-toolbar-rail">
              <div className="chart-toolbar-group chart-toolbar-group-nav">
                <button
                  type="button"
                  className="dialog-user-btn chart-toolbar-btn"
                  onClick={() => {
                    const stepMs = parsePeriodMs(period);
                    setEndCursorMs((prev) => (prev === null ? Date.now() - stepMs : prev - stepMs));
                  }}
                >
                  Earlier
                </button>

                <button
                  type="button"
                  className="dialog-user-btn chart-toolbar-btn"
                  onClick={() => {
                    const stepMs = parsePeriodMs(period);
                    setEndCursorMs((prev) => {
                      if (prev === null) {
                        return null;
                      }
                      const next = prev + stepMs;
                      return next >= Date.now() ? null : next;
                    });
                  }}
                  disabled={endCursorMs === null}
                >
                  Later
                </button>

                <button
                  type="button"
                  className="dialog-user-btn chart-toolbar-btn"
                  onClick={() => setEndCursorMs(null)}
                  disabled={endCursorMs === null}
                >
                  Now
                </button>
              </div>

              <div className="chart-toolbar-group chart-toolbar-group-primary">
                {loading ? <span className="dialog-scrooge dialog-scrooge-compact chart-toolbar-status">Loading...</span> : null}
                <button type="button" className="dialog-user-btn chart-toolbar-btn" onClick={() => void loadChart(false)}>
                  Scout Now
                </button>
              </div>
            </div>
          </div>
        </div>
      </details>

      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}

      <div className="chart-view-toolbar">
        <span
          className={`chart-view-status ${
            refreshing ? "chart-view-status-refreshing" : wsConnected ? "chart-view-status-live" : "chart-view-status-polling"
          }`}
        >
          {refreshing ? "Updating map..." : wsConnected ? "Wire live" : autoRefresh ? `Polling ${POLL_MS / 1000}s` : "Manual map"}
        </span>
        <button
          type="button"
          className="dialog-user-btn chart-toolbar-btn chart-view-btn"
          onClick={() => setChartsExpanded((prev) => !prev)}
        >
          {chartsExpanded ? "Exit Fullscreen" : "Fullscreen Charts"}
        </button>
      </div>

      <div className={`chart-stack ${chartsExpanded ? "chart-stack-fullscreen" : ""}`}>
        {chartsExpanded ? (
          <button
            type="button"
            className="dialog-user-btn chart-toolbar-btn chart-fullscreen-close"
            onClick={() => setChartsExpanded(false)}
          >
            Exit Fullscreen
          </button>
        ) : null}
        <section className="chart-panel chart-panel-price">
          <header className="chart-panel-head">
            <div className="chart-panel-title-wrap">
              <h3 className="chart-panel-title">{data?.symbol ?? symbol} Price</h3>
            </div>
            {priceLegendItems.length ? (
              <div className="chart-panel-legend" aria-label="Price chart legend">
                {priceLegendItems.map((item) => (
                  <span key={item.key} className="chart-legend-item">
                    <span className={`chart-legend-swatch ${item.swatchClassName}`} aria-hidden="true" />
                    <span className="chart-legend-label">{item.label}</span>
                  </span>
                ))}
              </div>
            ) : null}
          </header>
          <div ref={priceChartRef} className="chart-surface chart-surface-lg" />
        </section>
        <section className="chart-panel">
          <header className="chart-panel-head chart-panel-head-simple">
            <h3 className="chart-panel-title">Equity Curve</h3>
          </header>
          <div ref={equityChartRef} className="chart-surface chart-surface-md" />
        </section>
        <section className="chart-panel">
          <header className="chart-panel-head chart-panel-head-simple">
            <h3 className="chart-panel-title">
              {includeIndicators && data?.indicators?.rsi?.length
                ? (() => {
                    const rsiPeriod = data.indicator_spec?.rsi?.period ?? 11;
                    const rsiInterval = data.indicator_spec?.rsi?.interval;
                    return rsiInterval ? `RSI (${rsiPeriod}, ${rsiInterval})` : `RSI (${rsiPeriod})`;
                  })()
                : "RSI"}
            </h3>
          </header>
          <div ref={rsiChartRef} className="chart-surface chart-surface-sm" />
        </section>
      </div>

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
    </section>
  );
}

export default function ChartPage(): JSX.Element {
  return (
    <AuthGate>
      <ChartContent />
    </AuthGate>
  );
}
