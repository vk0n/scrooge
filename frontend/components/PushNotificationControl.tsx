"use client";

import { useEffect, useState } from "react";

import { fetchApi } from "../lib/api";
import {
  base64UrlToUint8Array,
  ensureServiceWorkerRegistration,
  serializePushSubscription,
  type BrowserPushSubscription
} from "../lib/push";

type NotificationConfigResponse = {
  enabled: boolean;
  configured: boolean;
  public_key: string | null;
  subject: string | null;
  subscription_count: number;
};

type NotificationTestResponse = {
  queued: boolean;
};

type NotificationStatus = "unsupported" | "blocked" | "idle" | "enabled";

function resolveNotificationStatus(
  supported: boolean,
  permission: NotificationPermission | "unsupported",
  subscribed: boolean
): NotificationStatus {
  if (!supported || permission === "unsupported") {
    return "unsupported";
  }
  if (permission === "denied") {
    return "blocked";
  }
  if (subscribed) {
    return "enabled";
  }
  return "idle";
}

async function loadCurrentSubscription(): Promise<PushSubscription | null> {
  const registration = await ensureServiceWorkerRegistration();
  return registration.pushManager.getSubscription();
}

export default function PushNotificationControl(): JSX.Element {
  const [supported, setSupported] = useState<boolean>(false);
  const [permission, setPermission] = useState<NotificationPermission | "unsupported">("unsupported");
  const [subscribed, setSubscribed] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<NotificationConfigResponse | null>(null);

  useEffect(() => {
    const browserSupported =
      typeof window !== "undefined" &&
      "Notification" in window &&
      "serviceWorker" in navigator &&
      "PushManager" in window;

    setSupported(browserSupported);
    if (!browserSupported) {
      setPermission("unsupported");
      return;
    }

    setPermission(Notification.permission);

    void (async () => {
      try {
        const [configPayload, existingSubscription] = await Promise.all([
          fetchApi<NotificationConfigResponse>("/api/notifications"),
          loadCurrentSubscription()
        ]);
        setConfig(configPayload);
        setSubscribed(existingSubscription !== null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load notification settings.");
      }
    })();
  }, []);

  async function enableNotifications(): Promise<void> {
    if (!supported) {
      setError("This browser does not support push notifications.");
      return;
    }

    setBusy(true);
    setError(null);
    setInfo(null);

    try {
      const nextConfig = config ?? (await fetchApi<NotificationConfigResponse>("/api/notifications"));
      setConfig(nextConfig);
      if (!nextConfig.enabled || !nextConfig.configured || !nextConfig.public_key) {
        throw new Error("Push delivery is not configured on the server.");
      }

      let currentPermission = Notification.permission;
      if (currentPermission !== "granted") {
        currentPermission = await Notification.requestPermission();
      }
      setPermission(currentPermission);
      if (currentPermission !== "granted") {
        throw new Error("Browser notification permission was not granted.");
      }

      const registration = await ensureServiceWorkerRegistration();
      const existingSubscription = await registration.pushManager.getSubscription();
      const browserSubscription =
        existingSubscription ??
        (await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: base64UrlToUint8Array(nextConfig.public_key)
        }));

      const payload: BrowserPushSubscription = serializePushSubscription(browserSubscription);
      await fetchApi("/api/notifications/subscribe", {
        method: "POST",
        body: { subscription: payload }
      });
      const testResponse = await fetchApi<NotificationTestResponse>("/api/notifications/test", {
        method: "POST",
        body: { subscription: payload }
      });

      setSubscribed(true);
      setInfo(
        testResponse.queued
          ? "The bell is wired. A test notification should arrive shortly."
          : "The bell is wired, but the server could not send a test notification yet."
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to enable notifications.");
    } finally {
      setBusy(false);
    }
  }

  async function disableNotifications(): Promise<void> {
    if (!supported) {
      return;
    }

    setBusy(true);
    setError(null);
    setInfo(null);

    try {
      const existingSubscription = await loadCurrentSubscription();
      if (existingSubscription) {
        const payload = serializePushSubscription(existingSubscription);
        await fetchApi("/api/notifications/unsubscribe", {
          method: "POST",
          body: { endpoint: payload.endpoint }
        });
        await existingSubscription.unsubscribe();
      }
      setSubscribed(false);
      setInfo("The bell has been muted on this device.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disable notifications.");
    } finally {
      setBusy(false);
    }
  }

  const status = resolveNotificationStatus(supported, permission, subscribed);
  const statusLabel =
    status === "enabled"
      ? "The bell is wired for important updates."
      : status === "blocked"
        ? "This browser has blocked the bell. Allow notifications in site settings."
        : status === "unsupported"
          ? "This browser cannot keep Scrooge's bell."
          : "The bell is quiet until you allow notifications.";

  return (
    <div className="notifications-panel">
      <div className="toolbar notification-toolbar">
        {status !== "enabled" ? (
          <button
            type="button"
            className="dialog-user-btn notification-action-btn"
            onClick={() => void enableNotifications()}
            disabled={busy || !supported}
          >
            Ring My Bell
          </button>
        ) : (
          <button
            type="button"
            className="dialog-user-btn notification-action-btn"
            onClick={() => void disableNotifications()}
            disabled={busy}
          >
            Mute the Bell
          </button>
        )}
        {busy ? <span className="dialog-scrooge dialog-scrooge-compact">Wiring the bell...</span> : null}
      </div>
      <div className="notification-meta">
        <p className="notification-status">{statusLabel}</p>
        {info ? <p className="dialog-scrooge dialog-scrooge-compact">{info}</p> : null}
        {error ? <p className="dialog-scrooge dialog-scrooge-error dialog-scrooge-compact">{error}</p> : null}
      </div>
    </div>
  );
}
