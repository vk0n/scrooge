"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type ControlAction = "start" | "stop" | "restart";

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

const POLL_MS = 2000;

export default function ControlsPage(): JSX.Element {
  const [busyAction, setBusyAction] = useState<ControlAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ControlResponse | null>(null);
  const [commandStatus, setCommandStatus] = useState<CommandStatusResponse | null>(null);

  async function runAction(action: ControlAction): Promise<void> {
    if (action === "stop" || action === "restart") {
      const confirmed = window.confirm(`Confirm ${action.toUpperCase()} trading action?`);
      if (!confirmed) {
        return;
      }
    }

    setBusyAction(action);
    setError(null);
    try {
      const response = await fetchApi<ControlResponse>(`/api/control/${action}`, { method: "POST" });
      setResult(response);
      const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${response.command_id}`);
      setCommandStatus(statusPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} trading`);
    } finally {
      setBusyAction(null);
    }
  }

  useEffect(() => {
    if (!result) {
      return () => undefined;
    }

    if (commandStatus && commandStatus.status !== "pending" && commandStatus.status !== "processing") {
      return () => undefined;
    }

    const intervalId = window.setInterval(() => {
      void (async () => {
        try {
          const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${result.command_id}`);
          setCommandStatus(statusPayload);
        } catch {
          // keep existing status until next poll/manual action
        }
      })();
    }, POLL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [result, commandStatus]);

  return (
    <section className="panel">
      <h1>Controls</h1>
      <p className="muted">Dangerous actions require explicit confirmation. Commands are queued for bot runtime.</p>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => void runAction("start")} disabled={busyAction !== null}>
          Start Trading
        </button>
        <button type="button" onClick={() => void runAction("stop")} disabled={busyAction !== null}>
          Stop Trading
        </button>
        <button type="button" onClick={() => void runAction("restart")} disabled={busyAction !== null}>
          Restart Trading
        </button>
        {busyAction ? <span className="muted">Executing {busyAction}...</span> : null}
      </div>

      {error ? <p>{error}</p> : null}

      {result ? (
        <>
          <h2>Command</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </>
      ) : null}

      {commandStatus ? (
        <>
          <h2>Command status</h2>
          <pre>{JSON.stringify(commandStatus, null, 2)}</pre>
        </>
      ) : null}
    </section>
  );
}
