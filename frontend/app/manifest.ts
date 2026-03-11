import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Scrooge Control Plane",
    short_name: "Scrooge",
    description: "Control and monitoring interface for Scrooge bot",
    start_url: "/",
    display: "standalone",
    background_color: "#d71926",
    theme_color: "#d71926",
    icons: [
      {
        src: "/brand/png/scrooge-app-icon-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any maskable"
      },
      {
        src: "/brand/png/scrooge-app-icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any maskable"
      }
    ]
  };
}
