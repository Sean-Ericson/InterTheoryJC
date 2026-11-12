import { defineConfig } from 'vite';

// Relative base so the built site works from any sub-path (e.g. GitHub Pages).
// Fixed port: SDE Explorer runs on 5173 and the two link to each other.
export default defineConfig({
  base: './',
  server: { port: 5174, strictPort: true },
});
