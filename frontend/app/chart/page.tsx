"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { fetchApi } from "../../lib/api";

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
  markers: {
    entries: Marker[];
    exits: Marker[];
    stop_loss: Marker[];
    take_profit: Marker[];
    liquidation: Marker[];
  };
  current_levels: Array<{ type: string; price: number }>;
  open_position: {
    side?: string;
    entry?: number | null;
    sl?: number | null;
    tp?: number | null;
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

const PERIOD_OPTIONS = ["6h", "12h", "1d", "3d", "1w", "2w", "4w", "12w", "26w", "52w"];
const INTERVAL_OPTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"];
const SOURCE_OPTIONS = ["auto", "dataset", "binance"] as const;
const POLL_MS = 60000;
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
} as const;

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

function pointsToXY(points: IndicatorPoint[]): { x: string[]; y: Array<number | null> } {
  return {
    x: points.map((point) => point.time),
    y: points.map((point) => point.value),
  };
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

export default function ChartPage(): JSX.Element {
  const priceChartRef = useRef<HTMLDivElement | null>(null);
  const equityChartRef = useRef<HTMLDivElement | null>(null);
  const rsiChartRef = useRef<HTMLDivElement | null>(null);

  const [symbol, setSymbol] = useState<string>("BTCUSDT");
  const [period, setPeriod] = useState<string>("1d");
  const [interval, setInterval] = useState<string>("1m");
  const [source, setSource] = useState<string>("auto");
  const [includeIndicators, setIncludeIndicators] = useState<boolean>(true);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [endCursorMs, setEndCursorMs] = useState<number | null>(null);

  const [data, setData] = useState<ChartPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  async function loadChart(silent = false): Promise<void> {
    const safeSymbol = symbol.trim().toUpperCase() || "BTCUSDT";
    if (!silent) {
      setLoading(true);
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

  const chartMeta = useMemo(() => {
    if (!data) {
      return "No chart data";
    }
    const rangeLabel =
      data.range_start && data.range_end
        ? `${new Date(data.range_start).toLocaleString()} → ${new Date(data.range_end).toLocaleString()}`
        : "Range unavailable";
    const requested = data.requested_interval ?? data.interval;
    const intervalLabel = requested === data.interval ? data.interval : `${requested} -> ${data.interval}`;
    const sourceLabel = data.requested_source && data.source ? `${data.requested_source} -> ${data.source}` : data.source ?? source;
    return `${data.symbol} | ${sourceLabel} | ${intervalLabel} | ${data.candles.length} candles | ${rangeLabel}`;
  }, [data, source]);

  useEffect(() => {
    let cancelled = false;

    const renderCharts = async (): Promise<void> => {
      if (!data) {
        return;
      }
      const Plotly = (await import("plotly.js-dist-min")).default;
      if (cancelled) {
        return;
      }

      const candleX = data.candles.map((candle) => candle.time);
      const candleOpen = data.candles.map((candle) => candle.open);
      const candleHigh = data.candles.map((candle) => candle.high);
      const candleLow = data.candles.map((candle) => candle.low);
      const candleClose = data.candles.map((candle) => candle.close);

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

      if (data.open_position?.entry !== null && data.open_position?.entry !== undefined && data.open_position?.time) {
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
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "EMA(20)",
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

      const priceShapes = data.current_levels.map((level) => ({
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: level.price,
        y1: level.price,
        line: {
          color: level.type === "sl" ? CHART_THEME.slLevel : CHART_THEME.tpLevel,
          width: 1,
          dash: level.type === "sl" ? "dot" : "dash",
        },
      }));

      if (priceChartRef.current) {
        await Plotly.react(
          priceChartRef.current,
          traces,
          {
            title: `${data.symbol} Price`,
            paper_bgcolor: CHART_THEME.bg,
            plot_bgcolor: CHART_THEME.bg,
            font: { color: CHART_THEME.text },
            xaxis: { type: "date", rangeslider: { visible: true } },
            yaxis: { title: "Price" },
            margin: { t: 45, r: 16, b: 40, l: 55 },
            shapes: priceShapes,
            legend: { orientation: "h", y: 1.12 },
          },
          { responsive: true, displaylogo: false }
        );

        const chartEl = priceChartRef.current as PlotlyEventTarget;
        chartEl.removeAllListeners?.("plotly_relayout");

        const candleTimesMs = data.candles.map((candle) => Date.parse(candle.time));
        let relayoutInProgress = false;

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
            relayoutInProgress = true;
            void Plotly.relayout(chartEl, { "yaxis.autorange": true }).finally(() => {
              relayoutInProgress = false;
            });
            return;
          }

          const range = parseRelayoutXRange(eventData);
          if (!range) {
            return;
          }
          rescaleVisibleY(range[0], range[1]);
        });
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
            xaxis: { type: "date" },
            yaxis: { title: "Vault" },
            margin: { t: 40, r: 16, b: 40, l: 55 },
          },
          { responsive: true, displaylogo: false }
        );
      }

      if (rsiChartRef.current) {
        const rsiPoints = data.indicators?.rsi ?? [];
        const hasRsi = includeIndicators && rsiPoints.length > 0;
        if (hasRsi) {
          const rsi = pointsToXY(rsiPoints);
          await Plotly.react(
            rsiChartRef.current,
            [
              {
                type: "scatter",
                mode: "lines",
                name: "RSI(14)",
                x: rsi.x,
                y: rsi.y,
                line: { color: CHART_THEME.rsi, width: 1.4 },
              },
            ],
            {
              title: "RSI",
              paper_bgcolor: CHART_THEME.bg,
              plot_bgcolor: CHART_THEME.bg,
              font: { color: CHART_THEME.text },
              xaxis: { type: "date" },
              yaxis: { title: "RSI", range: [0, 100] },
              margin: { t: 40, r: 16, b: 40, l: 55 },
              shapes: [
                {
                  type: "line",
                  xref: "paper",
                  x0: 0,
                  x1: 1,
                  y0: 70,
                  y1: 70,
                  line: { color: CHART_THEME.rsiUpper, dash: "dot", width: 1 },
                },
                {
                  type: "line",
                  xref: "paper",
                  x0: 0,
                  x1: 1,
                  y0: 30,
                  y1: 30,
                  line: { color: CHART_THEME.rsiLower, dash: "dot", width: 1 },
                },
              ],
            },
            { responsive: true, displaylogo: false }
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
              margin: { t: 40, r: 16, b: 30, l: 40 },
            },
            { responsive: true, displaylogo: false }
          );
        }
      }
    };

    void renderCharts();

    return () => {
      cancelled = true;
    };
  }, [data, includeIndicators]);

  return (
    <section className="panel page-shell">
      <p className="dialog-scrooge">Read-only map of candles, trades, and vault curve.</p>

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

      <div className="toolbar chart-toolbar">
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

        <button type="button" className="dialog-user-btn chart-toolbar-btn" onClick={() => void loadChart(false)}>
          Scout Now
        </button>
        {loading ? <span className="dialog-scrooge dialog-scrooge-compact">Loading...</span> : null}
      </div>

      <p className="dialog-scrooge">{chartMeta}</p>
      <p className="dialog-scrooge">Mode: {endCursorMs === null ? "Live scouting" : "Historical scouting"}</p>
      {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}

      <div ref={priceChartRef} className="chart-surface chart-surface-lg" />
      <div ref={equityChartRef} className="chart-surface chart-surface-md" />
      <div ref={rsiChartRef} className="chart-surface chart-surface-sm" />

      {data?.open_position ? (
        <>
          <h2>Current Trade Snapshot</h2>
          <pre className="json-box">{JSON.stringify(data.open_position, null, 2)}</pre>
        </>
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
    </section>
  );
}
