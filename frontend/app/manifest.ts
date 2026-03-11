import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Scrooge Control Plane",
    short_name: "Scrooge",
    description: "Control and monitoring interface for Scrooge bot",
    start_url: "/",
    display: "standalone",
    background_color: "#05070a",
    theme_color: "#0e1118",
    icons: [
      {
        src: "/brand/png/scrooge-mark-red-badge-192.png",
        sizes: "192x192",
        type: "image/png"
      },
      {
        src: "/brand/png/scrooge-mark-red-badge-512.png",
        sizes: "512x512",
        type: "image/png"
      }
    ]
  };
}
