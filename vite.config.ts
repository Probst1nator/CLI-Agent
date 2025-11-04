import { defineConfig } from 'vite'
import path from 'path'

export default defineConfig({
  root: 'frontend',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'frontend/index.html')
    },
    sourcemap: true
  },
  server: {
    port: 8765,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:7860',
        ws: true
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend/js'),
      '@types': path.resolve(__dirname, './frontend/types')
    }
  }
})
