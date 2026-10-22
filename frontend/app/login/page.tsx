"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { startSession } from "../../lib/auth";

export default function LoginPage(): JSX.Element {
  const router = useRouter();
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<boolean>(false);

  async function submitLogin(): Promise<void> {
    if (!username.trim() || !password) {
      setError("Username and password are required");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await startSession(username.trim(), password);
      setPassword("");
      router.replace("/dashboard");
      router.refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Login failed";
      setError(message.includes("401") ? "Invalid username or password" : message);
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void submitLogin();
  }

  return (
    <section className="panel page-shell auth-shell">
      <div className="auth-hero">
        <Image
          src="/brand/png/scrooge-mark-light-512.png"
          alt="Scrooge logo"
          width={148}
          height={148}
          className="auth-hero-logo"
          priority
          unoptimized
        />
        <p className="auth-greeting dialog-scrooge">
          Hello!
          <br />
          I am Scrooge...
        </p>
      </div>
      <h1>Enter the Office</h1>
      <form className="auth-form" onSubmit={onSubmit}>
        <label htmlFor="guiUser" className="field-stack dialog-user-field">
          Your Name
          <input
            id="guiUser"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label htmlFor="guiPassword" className="field-stack dialog-user-field">
          Vault Key
          <input
            id="guiPassword"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>

        <div className="toolbar">
          <button type="submit" className="dialog-user-btn" disabled={busy}>
            {busy ? "Opening..." : "Open the Office Door"}
          </button>
        </div>
      </form>

      {error ? <p className="auth-error dialog-scrooge dialog-scrooge-error">{error}</p> : null}
    </section>
  );
}
