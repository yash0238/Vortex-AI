import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

// Two pages: index.html = research dashboard (for the panel), money.html = VORTEX Money consumer app.
export default defineConfig({
  plugins: [react()],
  // mfapi.in sends no CORS header, so the research dashboard's fund pages go through this proxy (dev and preview).
  server: { proxy: { '/mfapi': { target: 'https://api.mfapi.in', changeOrigin: true, rewrite: (p) => p.replace(/^\/mfapi/, '') } } },
  preview: { proxy: { '/mfapi': { target: 'https://api.mfapi.in', changeOrigin: true, rewrite: (p) => p.replace(/^\/mfapi/, '') } } },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        money: resolve(__dirname, 'money.html'),
      },
    },
  },
})
