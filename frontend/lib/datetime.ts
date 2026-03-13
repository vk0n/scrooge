"use client";

const EU_DATETIME_FORMATTER = new Intl.DateTimeFormat("uk-UA", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false
});

const LOCAL_DATETIME_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?$/;

function toDate(value: unknown): Date | null {
  if (value instanceof Date) {
    return Number.isFinite(value.getTime()) ? value : null;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    if (value >= 1_000_000_000_000) {
      return new Date(value);
    }
    if (value >= 1_000_000_000) {
      return new Date(value * 1000);
    }
    return null;
  }

  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const numeric = Number(trimmed);
  if (Number.isFinite(numeric)) {
    return toDate(numeric);
  }

  const normalized = LOCAL_DATETIME_RE.test(trimmed) ? trimmed.replace(" ", "T") : trimmed;
  const parsed = new Date(normalized);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

export function formatDateTimeEu(value: unknown, fallback = "N/A"): string {
  const parsed = toDate(value);
  if (!parsed) {
    return fallback;
  }
  return EU_DATETIME_FORMATTER.format(parsed).replace(", ", " ");
}
