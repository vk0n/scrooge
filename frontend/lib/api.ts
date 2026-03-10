"use client";

import { getAuthorizationHeader } from "./auth";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

type ApiRequestOptions = {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
  body?: unknown;
};

export async function fetchApi<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const authHeader = getAuthorizationHeader();
  const hasBody = options.body !== undefined;
  const headers: Record<string, string> = {
    ...(authHeader ? { Authorization: authHeader } : {}),
    ...(hasBody ? { "Content-Type": "application/json" } : {}),
    ...(options.headers ?? {})
  };

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    cache: "no-store",
    headers,
    ...(hasBody ? { body: JSON.stringify(options.body) } : {})
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API ${response.status}: ${detail}`);
  }

  return (await response.json()) as T;
}

export function buildWebSocketUrl(path: string, query?: Record<string, string>): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const baseOrigin = API_BASE_URL || window.location.origin;
  let url: URL;
  try {
    url = new URL(baseOrigin);
  } catch {
    return null;
  }

  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = normalizedPath;

  if (query) {
    Object.entries(query).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
  }

  return url.toString();
}
