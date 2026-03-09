"use client";

import { getAuthorizationHeader } from "./auth";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type ApiRequestOptions = {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
};

export async function fetchApi<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const authHeader = getAuthorizationHeader();
  const headers: Record<string, string> = {
    ...(authHeader ? { Authorization: authHeader } : {}),
    ...(options.headers ?? {})
  };

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    cache: "no-store",
    headers
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API ${response.status}: ${detail}`);
  }

  return (await response.json()) as T;
}
