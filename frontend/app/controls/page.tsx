"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../../lib/api";

type ControlAction = "start" | "stop" | "restart" | "close_position" | "update_sl" | "update_tp";
type ControlEndpoint = "start" | "stop" | "restart" | "close-position" | "update-sl" | "update-tp";

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
  const [busyAction, setBusyAction] = useState<ControlEndpoint | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ControlResponse | null>(null);
  const [commandStatus, setCommandStatus] = useState<CommandStatusResponse | null>(null);
  const [slValue, setSlValue] = useState<string>("");
  const [tpValue, setTpValue] = useState<string>("");

  async function runAction(
    endpoint: ControlEndpoint,
    options?: { confirmMessage?: string; body?: unknown }
  ): Promise<void> {
    const confirmMessage = options?.confirmMessage;
    if (confirmMessage) {
      const confirmed = window.confirm(confirmMessage);
      if (!confirmed) {
        return;
      }
    }

    setBusyAction(endpoint);
    setError(null);
    try {
      const response = await fetchApi<ControlResponse>(`/api/control/${endpoint}`, {
        method: "POST",
        body: options?.body
      });
      setResult(response);
      const statusPayload = await fetchApi<CommandStatusResponse>(`/api/control/commands/${response.command_id}`);
      setCommandStatus(statusPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to execute ${endpoint}`);
    } finally {
      setBusyAction(null);
    }
  }

  async function runUpdate(endpoint: "update-sl" | "update-tp", rawValue: string): Promise<void> {
    const numericValue = Number(rawValue);
    if (!Number.isFinite(numericValue) || numericValue <= 0) {
      setError("Provide a positive numeric value.");
      return;
    }

    const label = endpoint === "update-sl" ? "SL" : "TP";
    await runAction(endpoint, {
      body: { value: numericValue },
      confirmMessage: `Confirm ${label} update to ${numericValue}?`
    });
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
        <button
          type="button"
          onClick={() =>
            void runAction("stop", {
              confirmMessage: "Confirm STOP trading action?"
            })
          }
          disabled={busyAction !== null}
        >
          Stop Trading
        </button>
        <button
          type="button"
          onClick={() =>
            void runAction("restart", {
              confirmMessage: "Confirm RESTART trading action?"
            })
          }
          disabled={busyAction !== null}
        >
          Restart Trading
        </button>
        <button
          type="button"
          onClick={() =>
            void runAction("close-position", {
              confirmMessage: "Confirm manual CLOSE of current position?"
            })
          }
          disabled={busyAction !== null}
        >
          Close Position
        </button>
        {busyAction ? <span className="muted">Executing {busyAction}...</span> : null}
      </div>

      <h2>Update Levels</h2>
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "end" }}>
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          New SL
          <input
            type="number"
            step="any"
            min="0"
            value={slValue}
            onChange={(event) => setSlValue(event.target.value)}
            placeholder="e.g. 61234.5"
            disabled={busyAction !== null}
          />
        </label>
        <button type="button" onClick={() => void runUpdate("update-sl", slValue)} disabled={busyAction !== null}>
          Update SL
        </button>
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          New TP
          <input
            type="number"
            step="any"
            min="0"
            value={tpValue}
            onChange={(event) => setTpValue(event.target.value)}
            placeholder="e.g. 64500"
            disabled={busyAction !== null}
          />
        </label>
        <button type="button" onClick={() => void runUpdate("update-tp", tpValue)} disabled={busyAction !== null}>
          Update TP
        </button>
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
