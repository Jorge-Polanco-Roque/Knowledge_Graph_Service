import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3477,
    proxy: {
      '/api': 'http://localhost:8899',
      '/ws': { target: 'http://localhost:8899', ws: true },
    },
  },
})
