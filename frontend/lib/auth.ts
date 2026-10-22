"use client";

import { fetchApi } from "./api";


const LEGACY_AUTH_STORAGE_KEY = "scrooge_gui_auth";

type SessionResponse = {
  authenticated: boolean;
  username?: string;
  expires_in_seconds?: number;
};

function clearLegacyBasicAuth(): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(LEGACY_AUTH_STORAGE_KEY);
  }
}

export async function startSession(username: string, password: string): Promise<void> {
  clearLegacyBasicAuth();
  await fetchApi<SessionResponse>("/api/auth/login", {
    method: "POST",
    body: { username, password }
  });
}

export async function hasActiveSession(): Promise<boolean> {
  clearLegacyBasicAuth();
  try {
    const session = await fetchApi<SessionResponse>("/api/auth/session");
    return session.authenticated;
  } catch {
    return false;
  }
}

export async function endSession(): Promise<void> {
  clearLegacyBasicAuth();
  try {
    await fetchApi<SessionResponse>("/api/auth/logout", { method: "POST" });
  } catch {
    // A failed logout request must not leave legacy browser credentials behind.
  }
}
