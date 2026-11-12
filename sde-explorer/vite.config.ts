import { defineConfig } from 'vite';

// Relative base so the built site works from any sub-path (e.g. GitHub Pages).
// Fixed port: Plot Lab runs on 5174 and the two link to each other.
export default defineConfig({
  base: './',
  server: { port: 5173, strictPort: true },
});
