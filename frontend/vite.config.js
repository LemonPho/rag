import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api/* is proxied to the FastAPI backend, so the browser makes same-origin
// requests and CORS never comes up. Point the target at wherever the backend
// is reachable from the machine running Vite (e.g. an SSH tunnel).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // No rewrite: the backend serves its routes under /api too, so dev and
      // the production build hit identical URLs.
      '/api': {
        target: process.env.BACKEND || 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
})
