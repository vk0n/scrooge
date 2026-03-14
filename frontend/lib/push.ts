"use client";

export type BrowserPushSubscription = {
  endpoint: string;
  expirationTime?: number | null;
  keys: {
    p256dh: string;
    auth: string;
  };
};

function normalizeBase64Url(value: string): string {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  return (value + padding).replace(/-/g, "+").replace(/_/g, "/");
}

export function base64UrlToUint8Array(value: string): Uint8Array {
  const normalized = normalizeBase64Url(value);
  const raw = window.atob(normalized);
  const output = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) {
    output[index] = raw.charCodeAt(index);
  }
  return output;
}

export async function ensureServiceWorkerRegistration(): Promise<ServiceWorkerRegistration> {
  const registration = await navigator.serviceWorker.register("/sw.js");
  return navigator.serviceWorker.ready.then(() => registration);
}

export function serializePushSubscription(subscription: PushSubscription): BrowserPushSubscription {
  const json = subscription.toJSON();
  const endpoint = json.endpoint ?? subscription.endpoint;
  const keys = json.keys ?? {};
  if (!endpoint || !keys.p256dh || !keys.auth) {
    throw new Error("Browser push subscription is missing endpoint or encryption keys.");
  }
  return {
    endpoint,
    expirationTime: json.expirationTime ?? subscription.expirationTime,
    keys: {
      p256dh: keys.p256dh,
      auth: keys.auth
    }
  };
}
