"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchApi } from "../../lib/api";

type EditableParams = {
  sl_mult: number | null;
  tp_mult: number | null;
  trail_atr_mult: number | null;
  rsi_extreme_long: number | null;
  rsi_extreme_short: number | null;
  rsi_long_open_threshold: number | null;
  rsi_long_qty_threshold: number | null;
  rsi_long_tp_threshold: number | null;
  rsi_long_close_threshold: number | null;
  rsi_short_open_threshold: number | null;
  rsi_short_qty_threshold: number | null;
  rsi_short_tp_threshold: number | null;
  rsi_short_close_threshold: number | null;
};

type EditableConfig = {
  symbol: string | null;
  leverage: number | null;
  use_full_balance: boolean | null;
  qty: number | null;
  params: EditableParams;
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

type ControlResponse = {
  command_id: string;
  action: "start" | "stop" | "restart";
  status: string;
  queued_at: string;
};

type CommandStatusResponse = {
  command_id: string;
  action: "start" | "stop" | "restart";
  status: string;
  message: string;
  updated_at: string;
  trading_status?: {
    trading_enabled: boolean;
  } | string;
};

type ParamsFormState = {
  [K in keyof EditableParams]: string;
};

type FormState = {
  symbol: string;
  leverage: string;
  use_full_balance: boolean;
  qty: string;
  params: ParamsFormState;
};

const PARAM_FIELDS: Array<{ key: keyof EditableParams; label: string; min?: number; max?: number; step?: string }> = [
  { key: "sl_mult", label: "SL multiplier", min: 0, step: "0.01" },
  { key: "tp_mult", label: "TP multiplier", min: 0, step: "0.01" },
  { key: "trail_atr_mult", label: "Trail ATR multiplier", min: 0, step: "0.001" },
  { key: "rsi_extreme_long", label: "RSI extreme long", min: 0, max: 100, step: "1" },
  { key: "rsi_extreme_short", label: "RSI extreme short", min: 0, max: 100, step: "1" },
  { key: "rsi_long_open_threshold", label: "RSI long open", min: 0, max: 100, step: "1" },
  { key: "rsi_long_qty_threshold", label: "RSI long qty", min: 0, max: 100, step: "1" },
  { key: "rsi_long_tp_threshold", label: "RSI long TP", min: 0, max: 100, step: "1" },
  { key: "rsi_long_close_threshold", label: "RSI long close", min: 0, max: 100, step: "1" },
  { key: "rsi_short_open_threshold", label: "RSI short open", min: 0, max: 100, step: "1" },
  { key: "rsi_short_qty_threshold", label: "RSI short qty", min: 0, max: 100, step: "1" },
  { key: "rsi_short_tp_threshold", label: "RSI short TP", min: 0, max: 100, step: "1" },
  { key: "rsi_short_close_threshold", label: "RSI short close", min: 0, max: 100, step: "1" }
];

const STATUS_POLL_MS = 2000;

function numberToInput(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function editableToForm(editable: EditableConfig): FormState {
  const params: ParamsFormState = {
    sl_mult: numberToInput(editable.params.sl_mult),
    tp_mult: numberToInput(editable.params.tp_mult),
    trail_atr_mult: numberToInput(editable.params.trail_atr_mult),
    rsi_extreme_long: numberToInput(editable.params.rsi_extreme_long),
    rsi_extreme_short: numberToInput(editable.params.rsi_extreme_short),
    rsi_long_open_threshold: numberToInput(editable.params.rsi_long_open_threshold),
    rsi_long_qty_threshold: numberToInput(editable.params.rsi_long_qty_threshold),
    rsi_long_tp_threshold: numberToInput(editable.params.rsi_long_tp_threshold),
    rsi_long_close_threshold: numberToInput(editable.params.rsi_long_close_threshold),
    rsi_short_open_threshold: numberToInput(editable.params.rsi_short_open_threshold),
    rsi_short_qty_threshold: numberToInput(editable.params.rsi_short_qty_threshold),
    rsi_short_tp_threshold: numberToInput(editable.params.rsi_short_tp_threshold),
    rsi_short_close_threshold: numberToInput(editable.params.rsi_short_close_threshold)
  };

  return {
    symbol: editable.symbol ?? "",
    leverage: numberToInput(editable.leverage),
    use_full_balance: Boolean(editable.use_full_balance),
    qty: numberToInput(editable.qty),
    params
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

function buildSavePayload(form: FormState): EditableConfig {
  const symbol = form.symbol.trim().toUpperCase();
  if (!/^[A-Z0-9]{3,20}$/.test(symbol)) {
    throw new Error("Symbol must match [A-Z0-9]{3,20}");
  }

  const leverage = parseNumberField(form.leverage, "Leverage", { min: 1, max: 125, integer: true });
  const qty = parseNumberField(form.qty, "Qty", { min: 0, exclusiveMin: true, allowEmpty: true });

  const params = PARAM_FIELDS.reduce<EditableParams>((acc, field) => {
    const value = parseNumberField(form.params[field.key], field.label, {
      min: field.min,
      max: field.max,
      exclusiveMin: field.key === "sl_mult" || field.key === "tp_mult" || field.key === "trail_atr_mult"
    });
    acc[field.key] = value;
    return acc;
  }, {
    sl_mult: null,
    tp_mult: null,
    trail_atr_mult: null,
    rsi_extreme_long: null,
    rsi_extreme_short: null,
    rsi_long_open_threshold: null,
    rsi_long_qty_threshold: null,
    rsi_long_tp_threshold: null,
    rsi_long_close_threshold: null,
    rsi_short_open_threshold: null,
    rsi_short_qty_threshold: null,
    rsi_short_tp_threshold: null,
    rsi_short_close_threshold: null
  });

  return {
    symbol,
    leverage,
    use_full_balance: form.use_full_balance,
    qty,
    params
  };
}

export default function ConfigPage(): JSX.Element {
  const [form, setForm] = useState<FormState | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [saveResult, setSaveResult] = useState<SaveEditableResponse | null>(null);
  const [restartStatus, setRestartStatus] = useState<CommandStatusResponse | null>(null);
  const [restartCommandId, setRestartCommandId] = useState<string | null>(null);

  const isRestartInProgress = useMemo(() => {
    if (!restartStatus) {
      return false;
    }
    return restartStatus.status === "pending" || restartStatus.status === "processing";
  }, [restartStatus]);

  async function loadEditable(): Promise<void> {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchApi<EditableConfigResponse>("/api/config/editable");
      setForm(editableToForm(payload.editable));
      setInfo(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load editable config");
    } finally {
      setLoading(false);
    }
  }

  useEffect((): void => {
    void loadEditable();
  }, []);

  useEffect(() => {
    if (!restartCommandId) {
      return () => undefined;
    }

    if (restartStatus && restartStatus.status !== "pending" && restartStatus.status !== "processing") {
      return () => undefined;
    }

    const intervalId = window.setInterval(() => {
      void (async () => {
        try {
          const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${restartCommandId}`);
          setRestartStatus(statusPayload);
        } catch {
          // Keep previous status and retry on next tick.
        }
      })();
    }, STATUS_POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [restartCommandId, restartStatus]);

  async function saveConfig(saveAndRestart: boolean): Promise<void> {
    if (!form) {
      return;
    }
    if (saveAndRestart) {
      const confirmed = window.confirm("Save config and restart trading?");
      if (!confirmed) {
        return;
      }
    }

    setSaving(true);
    setError(null);
    setInfo(null);

    try {
      const payload = buildSavePayload(form);
      const result = await fetchApi<SaveEditableResponse>("/api/config/editable", {
        method: "POST",
        body: payload
      });
      setSaveResult(result);
      setForm(editableToForm(result.editable));

      if (result.updated) {
        setInfo(`Config saved. Changed: ${result.changed_fields.join(", ")}`);
      } else {
        setInfo("No config changes detected");
      }

      if (saveAndRestart) {
        const command = await fetchApi<ControlResponse>("/api/control/restart", { method: "POST" });
        setRestartCommandId(command.command_id);
        const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${command.command_id}`);
        setRestartStatus(statusPayload);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save config");
    } finally {
      setSaving(false);
    }
  }

  if (loading || !form) {
    return (
      <section className="panel">
        <h1>Config</h1>
        <p className="muted">Loading editable configuration...</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h1>Config</h1>
      <p className="muted">Limited safe editor for selected runtime parameters.</p>

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => void loadEditable()} disabled={saving || isRestartInProgress}>
          Reload from file
        </button>
        <button type="button" onClick={() => void saveConfig(false)} disabled={saving || isRestartInProgress}>
          Save
        </button>
        <button type="button" onClick={() => void saveConfig(true)} disabled={saving || isRestartInProgress}>
          Save & Restart
        </button>
        {saving ? <span className="muted">Saving...</span> : null}
      </div>

      {error ? <p>{error}</p> : null}
      {info ? <p>{info}</p> : null}

      <div className="kv-grid" style={{ marginBottom: "1rem" }}>
        <label className="kv-item">
          <span className="kv-label">Symbol</span>
          <input
            value={form.symbol}
            onChange={(event) => setForm((prev) => (prev ? { ...prev, symbol: event.target.value } : prev))}
          />
        </label>

        <label className="kv-item">
          <span className="kv-label">Leverage</span>
          <input
            type="number"
            min={1}
            max={125}
            value={form.leverage}
            onChange={(event) => setForm((prev) => (prev ? { ...prev, leverage: event.target.value } : prev))}
          />
        </label>

        <label className="kv-item">
          <span className="kv-label">Qty (empty = null)</span>
          <input
            type="number"
            step="0.000001"
            min={0}
            value={form.qty}
            onChange={(event) => setForm((prev) => (prev ? { ...prev, qty: event.target.value } : prev))}
          />
        </label>

        <label className="kv-item" style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <span className="kv-label">Use full balance</span>
          <input
            type="checkbox"
            checked={form.use_full_balance}
            onChange={(event) =>
              setForm((prev) => (prev ? { ...prev, use_full_balance: event.target.checked } : prev))
            }
          />
        </label>
      </div>

      <h2>Strategy Params</h2>
      <div className="kv-grid">
        {PARAM_FIELDS.map((field) => (
          <label className="kv-item" key={field.key}>
            <span className="kv-label">{field.label}</span>
            <input
              type="number"
              min={field.min}
              max={field.max}
              step={field.step ?? "0.01"}
              value={form.params[field.key]}
              onChange={(event) =>
                setForm((prev) =>
                  prev
                    ? {
                        ...prev,
                        params: {
                          ...prev.params,
                          [field.key]: event.target.value
                        }
                      }
                    : prev
                )
              }
            />
          </label>
        ))}
      </div>

      {saveResult ? (
        <>
          <h2>Save result</h2>
          <pre>{JSON.stringify(saveResult, null, 2)}</pre>
        </>
      ) : null}

      {restartStatus ? (
        <>
          <h2>Restart status</h2>
          <pre>{JSON.stringify(restartStatus, null, 2)}</pre>
        </>
      ) : null}
    </section>
  );
}
