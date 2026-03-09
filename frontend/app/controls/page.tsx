"use client";

import { useState } from "react";

import { fetchApi } from "../../lib/api";

type ControlAction = "start" | "stop" | "restart";

type ControlResponse = {
  action: ControlAction;
  service_status?: {
    name: string;
    running: boolean;
    active_state: string;
    sub_state: string;
    unit_file_state: string;
  };
};

export default function ControlsPage(): JSX.Element {
  const [busyAction, setBusyAction] = useState<ControlAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ControlResponse | null>(null);

  async function runAction(action: ControlAction): Promise<void> {
    if (action === "stop" || action === "restart") {
      const confirmed = window.confirm(`Confirm ${action.toUpperCase()} service action?`);
      if (!confirmed) {
        return;
      }
    }

    setBusyAction(action);
    setError(null);
    try {
      const response = await fetchApi<ControlResponse>(`/api/control/${action}`, { method: "POST" });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} service`);
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <section className="panel">
      <h1>Controls</h1>
      <p className="muted">Dangerous actions require explicit confirmation.</p>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => void runAction("start")} disabled={busyAction !== null}>
          Start
        </button>
        <button type="button" onClick={() => void runAction("stop")} disabled={busyAction !== null}>
          Stop
        </button>
        <button type="button" onClick={() => void runAction("restart")} disabled={busyAction !== null}>
          Restart
        </button>
        {busyAction ? <span className="muted">Executing {busyAction}...</span> : null}
      </div>

      {error ? <p>{error}</p> : null}

      {result ? (
        <>
          <h2>Result</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </>
      ) : null}
    </section>
  );
}
