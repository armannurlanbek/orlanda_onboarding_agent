import { defineConfig } from "vite";
import type { IncomingMessage } from "http";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

const apiTarget = "http://127.0.0.1:8000";

// Some paths (`/auth`, `/chat`, `/admin/logs`) are BOTH SPA routes and API routes.
// For browser navigation (Accept: text/html GETs) we must let Vite serve the dev
// index.html so the React app boots; only fetch/XHR API calls should be proxied
// to FastAPI. Without this, hard-refreshing /chat returns FastAPI's production
// index.html which references built assets that the dev server doesn't have,
// resulting in a blank page.
const bypassSpaNavigation = (req: IncomingMessage) => {
  const accept = String(req.headers.accept || "");
  if (req.method === "GET" && accept.includes("text/html")) {
    return "/index.html";
  }
  return undefined;
};

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      // The OAuth callback is a real backend route reached by a browser redirect. It must be
      // proxied to FastAPI (NOT served the SPA), so it is listed before "/auth" and has no
      // SPA-navigation bypass. Only relevant if the monday redirect URI points at the dev
      // server; if it points straight at :8000 this rule is simply unused.
      "/auth/monday/callback": { target: apiTarget, changeOrigin: true },
      "/auth": { target: apiTarget, changeOrigin: true, bypass: bypassSpaNavigation },
      "/chat": { target: apiTarget, changeOrigin: true, bypass: bypassSpaNavigation },
      "/knowledge": { target: apiTarget, changeOrigin: true },
      "/integrations": { target: apiTarget, changeOrigin: true },
      "/memories": { target: apiTarget, changeOrigin: true },
      "/me/memory-settings": { target: apiTarget, changeOrigin: true },
      "/branding": { target: apiTarget, changeOrigin: true },
      "/health": { target: apiTarget, changeOrigin: true },
      "/admin/logs": { target: apiTarget, changeOrigin: true, bypass: bypassSpaNavigation },
      "/admin/documents/metadata": { target: apiTarget, changeOrigin: true },
      "/admin/users": { target: apiTarget, changeOrigin: true },
      "/admin/model": { target: apiTarget, changeOrigin: true },
      // Client portal: /client/portal/* are API calls; /client/tasks etc. are SPA
      // routes (HTML navigations bypass to index.html like /chat above).
      "/client": { target: apiTarget, changeOrigin: true, bypass: bypassSpaNavigation },
      "/invites": { target: apiTarget, changeOrigin: true },
      "/admin/invites": { target: apiTarget, changeOrigin: true, bypass: bypassSpaNavigation },
      "/admin/orlanda": { target: apiTarget, changeOrigin: true },
      "/admin/clients": { target: apiTarget, changeOrigin: true },
    },
    hmr: {
      overlay: false,
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
