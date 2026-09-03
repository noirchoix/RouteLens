import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const backendOrigin = 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': backendOrigin,
      '/health': backendOrigin,
      '/ready': backendOrigin
    }
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true
  }
});
