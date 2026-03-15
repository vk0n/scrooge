export type EditableParams = {
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

export type EditableIntervals = {
  small?: string | null;
  medium?: string | null;
  big?: string | null;
};

export type EditableLimits = {
  small?: number | null;
  medium?: number | null;
  big?: number | null;
};

export type EditableConfig = {
  live: boolean | null;
  symbol: string | null;
  leverage: number | null;
  initial_balance: number | null;
  use_full_balance: boolean | null;
  qty: number | null;
  intervals?: EditableIntervals | null;
  limits?: EditableLimits | null;
  params?: EditableParams | null;
};

export type ContractSegment =
  | {
      kind: "text";
      text: string;
    }
  | {
      kind: "term";
      text: string;
    }
  | {
      kind: "value";
      path: string;
      display: string;
    };

export type ContractParagraph = ContractSegment[];

function formatContractValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "null";
    }
    return Number.isInteger(value) ? String(value) : String(value);
  }
  return String(value);
}

function textSegment(text: string): ContractSegment {
  return { kind: "text", text };
}

function termSegment(text: string): ContractSegment {
  return { kind: "term", text };
}

function valueSegment(path: string, value: unknown): ContractSegment {
  return {
    kind: "value",
    path,
    display: formatContractValue(value),
  };
}

export function buildContractParagraphs(config: EditableConfig): ContractParagraph[] {
  const intervals = config.intervals ?? {};
  const limits = config.limits ?? {};
  const params = config.params ?? {};
  const sizingTail =
    config.qty === null || config.qty === undefined
      ? [textSegment(".")]
      : [
          textSegment("; if all-in mode is ever false, Scrooge shall use the fixed stake "),
          valueSegment("qty", config.qty),
          textSegment("."),
        ];

  return [
    [
      textSegment("1. "),
      termSegment("Market mandate."),
      textSegment(" Scrooge shall operate on "),
      valueSegment("symbol", config.symbol),
      textSegment(" futures with leverage "),
      valueSegment("leverage", config.leverage),
      textSegment(". Position sizing shall follow all-in mode "),
      valueSegment("use_full_balance", config.use_full_balance),
      ...sizingTail,
    ],
    [
      textSegment("2. "),
      termSegment("Observation windows."),
      textSegment(" Scrooge shall read "),
      valueSegment("intervals.small", intervals.small),
      textSegment(" candles for execution, "),
      valueSegment("intervals.medium", intervals.medium),
      textSegment(" candles for Bollinger Bands and ATR, and "),
      valueSegment("intervals.big", intervals.big),
      textSegment(" candles for RSI and EMA. He shall keep up to "),
      valueSegment("limits.small", limits.small),
      textSegment(", "),
      valueSegment("limits.medium", limits.medium),
      textSegment(", and "),
      valueSegment("limits.big", limits.big),
      textSegment(" candles in those three books."),
    ],
    [
      textSegment("3. "),
      termSegment("Long entry."),
      textSegment(" Scrooge shall open a long only when price slips below the lower Bollinger Band, RSI is below the long-open bar "),
      valueSegment("params.rsi_long_open_threshold", params.rsi_long_open_threshold),
      textSegment(", and price still stands above the EMA trend filter. If RSI remains above the long-size bar "),
      valueSegment("params.rsi_long_qty_threshold", params.rsi_long_qty_threshold),
      textSegment(", he shall cut the opening size to half of the computed amount; otherwise he shall deploy the full computed size."),
    ],
    [
      textSegment("4. "),
      termSegment("Short entry."),
      textSegment(" Scrooge shall open a short only when price rises above the upper Bollinger Band, RSI is above the short-open bar "),
      valueSegment("params.rsi_short_open_threshold", params.rsi_short_open_threshold),
      textSegment(", and price remains below the EMA trend filter. If RSI stays below the short-size bar "),
      valueSegment("params.rsi_short_qty_threshold", params.rsi_short_qty_threshold),
      textSegment(", he shall cut the opening size to half of the computed amount; otherwise he shall deploy the full computed size."),
    ],
    [
      textSegment("5. "),
      termSegment("Protective terms."),
      textSegment(" On every new trade, Scrooge shall place the Safety Net at ATR multiplied by "),
      valueSegment("params.sl_mult", params.sl_mult),
      textSegment(" from entry and the base Treasure Mark at ATR multiplied by "),
      valueSegment("params.tp_mult", params.tp_mult),
      textSegment(" from entry. He shall reject any opening whose Safety Net would cross the liquidation line. If the office sends a manual trade suggestion, Scrooge may skip the entry timing test, but he shall still obey every risk and management clause in this contract."),
    ],
    [
      textSegment("6. "),
      termSegment("Long supervision."),
      textSegment(" If RSI climbs above the long-extreme ceiling "),
      valueSegment("params.rsi_extreme_long", params.rsi_extreme_long),
      textSegment(", Scrooge shall close the long at once. If price clears the base Treasure Mark while RSI stays below the long-treasure bar "),
      valueSegment("params.rsi_long_tp_threshold", params.rsi_long_tp_threshold),
      textSegment(", he shall arm Tail Guard. While Tail Guard is armed, he shall trail the market by ATR multiplied by "),
      valueSegment("params.trail_atr_mult", params.trail_atr_mult),
      textSegment(" and close the long if price falls through that trail or RSI climbs above the long-exit bar "),
      valueSegment("params.rsi_long_close_threshold", params.rsi_long_close_threshold),
      textSegment(". If Tail Guard has not armed yet, the Safety Net remains the only hard exit on the downside."),
    ],
    [
      textSegment("7. "),
      termSegment("Short supervision."),
      textSegment(" If RSI falls below the short-extreme floor "),
      valueSegment("params.rsi_extreme_short", params.rsi_extreme_short),
      textSegment(", Scrooge shall close the short at once. If price clears the base Treasure Mark while RSI stays above the short-treasure bar "),
      valueSegment("params.rsi_short_tp_threshold", params.rsi_short_tp_threshold),
      textSegment(", he shall arm Tail Guard. While Tail Guard is armed, he shall trail the market by the same ATR multiplier and close the short if price climbs back through that trail or RSI falls below the short-exit bar "),
      valueSegment("params.rsi_short_close_threshold", params.rsi_short_close_threshold),
      textSegment(". If Tail Guard has not armed yet, the Safety Net remains the only hard exit on the upside."),
    ],
  ];
}
