import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

// Two pages: index.html = research dashboard (for the panel), money.html = VORTEX Money consumer app.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        money: resolve(__dirname, 'money.html'),
      },
    },
  },
})
