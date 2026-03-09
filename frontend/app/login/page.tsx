"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { saveBasicAuth } from "../../lib/auth";

export default function LoginPage(): JSX.Element {
  const router = useRouter();
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  function submitLogin(): void {
    if (!username.trim() || !password) {
      setError("Username and password are required");
      return;
    }
    saveBasicAuth(username.trim(), password);
    setError(null);
    router.push("/dashboard");
  }

  return (
    <section className="panel">
      <h1>Login</h1>
      <p className="muted">Provide basic auth credentials to access protected API endpoints.</p>
      <div style={{ display: "grid", gap: "0.75rem", maxWidth: "420px" }}>
        <label htmlFor="guiUser">
          Username
          <input
            id="guiUser"
            type="text"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            style={{ width: "100%", marginTop: "0.4rem" }}
          />
        </label>
        <label htmlFor="guiPassword">
          Password
          <input
            id="guiPassword"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            style={{ width: "100%", marginTop: "0.4rem" }}
          />
        </label>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <button type="button" onClick={submitLogin}>
          Save credentials
        </button>
      </div>

      {error ? <p>{error}</p> : null}
    </section>
  );
}
