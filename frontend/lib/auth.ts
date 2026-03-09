"use client";

type StoredAuth = { mode: "basic"; username: string; password: string };

const AUTH_STORAGE_KEY = "scrooge_gui_auth";

function toBase64(raw: string): string {
  return window.btoa(raw);
}

export function getAuthorizationHeader(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as StoredAuth;
    if (parsed.mode === "basic" && parsed.username.trim() && parsed.password) {
      return `Basic ${toBase64(`${parsed.username}:${parsed.password}`)}`;
    }
    return null;
  } catch {
    return null;
  }
}

export function saveBasicAuth(username: string, password: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({ mode: "basic", username, password }));
}

export function clearAuth(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

export function hasSavedAuth(): boolean {
  return getAuthorizationHeader() !== null;
}
