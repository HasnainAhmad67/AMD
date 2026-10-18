import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI server. Set VITE_API_PROXY in apps/web/.env
// (e.g. http://127.0.0.1:8100 for a locally started backend). In Docker dev it is http://api:8000.
// In production, nginx does the same proxying (see nginx.conf), so the app always calls relative /api URLs.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = process.env.VITE_API_PROXY ?? env.VITE_API_PROXY ?? "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      proxy: { "/api": { target: apiTarget, changeOrigin: true } },
      // Inside Docker on Windows/macOS, file-change events don't reach the container: poll instead.
      watch: process.env.VITE_USE_POLLING === "true" ? { usePolling: true, interval: 300 } : undefined,
    },
    build: {
      rollupOptions: {
        output: {
          // Separate long-lived vendor chunks so app changes don't bust the charting/React caches.
          manualChunks: {
            react: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"],
            charts: ["recharts"],
          },
        },
      },
    },
  };
});
