/** @type {import('next').NextConfig} */
const internalApiBaseUrl = process.env.INTERNAL_API_BASE_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${internalApiBaseUrl}/api/:path*`
      },
      {
        source: "/health",
        destination: `${internalApiBaseUrl}/health`
      }
    ];
  }
};

export default nextConfig;
