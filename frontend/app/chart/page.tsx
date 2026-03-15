"use client";

import { useEffect, useRef, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { getSavedBasicCredentials } from "../../lib/auth";
import { buildWebSocketUrl, fetchApi } from "../../lib/api";
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
  openPosition: "#f59e0b",
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
  livePrice: "#fbbf24",
} as const;

function buildCurrentLevelShapes(
  levels: Array<{ type: string; price: number }>,
  currentPrice: number | null
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
        color: CHART_THEME.livePrice,
        width: 1.1,
        dash: "solid",
      },
    });
  }

  return shapes;
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
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value !== "string") {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
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

  const [data, setData] = useState<ChartPayload | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatusPayload | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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
      setData(payload);
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
            setLiveStatus(payload.data);
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
    if (window.matchMedia("(max-width: 760px)").matches) {
      setControlsExpanded(false);
    }
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
      const candleTimesMs = data.candles.map((candle) => Date.parse(candle.time));
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
      const openPositionTimeMs =
        typeof data.open_position?.time === "string" ? Date.parse(data.open_position.time) : null;
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
          marker: { color: CHART_THEME.longEntry, symbol: "triangle-up", size: 10 },
        });
      }
      if (shortEntries.length) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "Short Entries",
          x: shortEntries.map((marker) => marker.time),
          y: shortEntries.map((marker) => marker.price),
          marker: { color: CHART_THEME.shortEntry, symbol: "triangle-down", size: 10 },
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
          marker: { color: CHART_THEME.openPosition, symbol: "diamond", size: 11 },
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
            title: {
              text: `${data.symbol} Price`,
              x: 0.5,
              xanchor: "center",
              y: 0.975,
              yanchor: "top",
            },
            paper_bgcolor: CHART_THEME.bg,
            plot_bgcolor: CHART_THEME.bg,
            font: { color: CHART_THEME.text },
            uirevision: chartRevisionKey,
            hovermode: "x unified",
            xaxis: {
              type: "date",
              rangeslider: { visible: true },
              range: initialXRange,
            },
            yaxis: { title: "Price" },
            margin: { t: 92, r: 16, b: 40, l: 55 },
            shapes: priceShapes,
            legend: {
              orientation: "h",
              x: 0,
              xanchor: "left",
              y: 1.14,
              yanchor: "bottom",
              font: { size: 11 },
            },
          },
          {
            responsive: true,
            displaylogo: false,
            scrollZoom: true,
            doubleClick: "reset+autosize",
          }
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
          const payload = range
            ? {
                "xaxis.range": [new Date(range[0]).toISOString(), new Date(range[1]).toISOString()],
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
            title: "Equity Curve",
            paper_bgcolor: CHART_THEME.bg,
            plot_bgcolor: CHART_THEME.bg,
            font: { color: CHART_THEME.text },
            uirevision: `${chartRevisionKey}:equity`,
            hovermode: "x unified",
            xaxis: {
              type: "date",
              range: initialXRange,
            },
            yaxis: { title: "Vault" },
            margin: { t: 40, r: 16, b: 40, l: 55 },
          },
          { responsive: true, displaylogo: false, scrollZoom: true, doubleClick: "reset+autosize" }
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
              title: rsiLabel,
              paper_bgcolor: CHART_THEME.bg,
              plot_bgcolor: CHART_THEME.bg,
              font: { color: CHART_THEME.text },
              uirevision: `${chartRevisionKey}:rsi`,
              hovermode: "x unified",
              xaxis: {
                type: "date",
                range: initialXRange,
              },
              yaxis: { title: "RSI", range: [0, 100] },
              margin: { t: 40, r: 16, b: 40, l: 55 },
              shapes: rsiShapes,
            },
            { responsive: true, displaylogo: false, scrollZoom: true, doubleClick: "reset+autosize" }
          );
        } else {
          await Plotly.react(
            rsiChartRef.current,
            [],
            {
              title: "RSI (not available)",
              paper_bgcolor: CHART_THEME.bg,
              plot_bgcolor: CHART_THEME.bg,
              font: { color: CHART_THEME.mutedText },
              uirevision: `${chartRevisionKey}:rsi-empty`,
              margin: { t: 40, r: 16, b: 30, l: 40 },
            },
            { responsive: true, displaylogo: false, scrollZoom: true, doubleClick: "reset+autosize" }
          );
        }
      }
    };

    void renderCharts();

    return () => {
      cancelled = true;
    };
  }, [data, includeIndicators, period, source, endCursorMs]);

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

      const liveCurrentPrice =
        liveStatus?.symbol?.trim().toUpperCase() === data.symbol.trim().toUpperCase()
          ? liveStatus.last_price ?? null
          : null;
      const activeLevels = resolveLiveLevels(data, liveStatus) ?? data.current_levels;
      const priceShapes = buildCurrentLevelShapes(activeLevels, liveCurrentPrice);
      await Plotly.relayout(priceChartRef.current, { shapes: priceShapes });
    };

    void applyLiveOverlay();

    return () => {
      cancelled = true;
    };
  }, [data, liveStatus]);

  return (
    <section className="panel page-shell">
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
              <button type="button" className="dialog-user-btn chart-toolbar-btn" onClick={() => void loadChart(false)}>
                Scout Now
              </button>
              {loading ? <span className="dialog-scrooge dialog-scrooge-compact chart-toolbar-status">Loading...</span> : null}
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
        <div ref={priceChartRef} className="chart-surface chart-surface-lg" />
        <div ref={equityChartRef} className="chart-surface chart-surface-md" />
        <div ref={rsiChartRef} className="chart-surface chart-surface-sm" />
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
