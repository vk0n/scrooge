self.addEventListener("push", (event) => {
  let payload = {
    title: "Scrooge Control",
    body: "Scrooge has an update.",
    tag: "scrooge-control",
    url: "/dashboard"
  };

  if (event.data) {
    try {
      payload = { ...payload, ...event.data.json() };
    } catch (_error) {
      payload.body = event.data.text();
    }
  }

  const title = payload.title || "Scrooge Control";
  const options = {
    body: payload.body || "Scrooge has an update.",
    tag: payload.tag || "scrooge-control",
    data: {
      url: payload.url || "/dashboard"
    },
    icon: "/brand/png/scrooge-app-icon-192.png",
    badge: "/brand/png/scrooge-app-icon-192.png"
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  const targetUrl = event.notification?.data?.url || "/dashboard";
  event.notification.close();

  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const client of clients) {
        if ("focus" in client) {
          const clientUrl = new URL(client.url);
          if (clientUrl.pathname === targetUrl) {
            return client.focus();
          }
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
      return undefined;
    })
  );
});
