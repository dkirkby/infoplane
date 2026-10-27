import { resolve } from "node:path";
import { defineConfig } from "vite";

// One entry per figure: these are standalone static pages, not a single-page app,
// so each gets its own HTML document and its own bundle.
const pages = ["index", "fig2", "fig3", "fig4", "fig5"];

export default defineConfig({
  base: "./",
  build: {
    rollupOptions: {
      input: Object.fromEntries(pages.map((p) => [p, resolve(import.meta.dirname, `${p}.html`)])),
    },
  },
  server: { open: "/index.html" },
});
