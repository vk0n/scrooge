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
