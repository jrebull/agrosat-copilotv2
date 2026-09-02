import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vitest/config";

// Nuxt replaces `import.meta.client` / `import.meta.server` at build time. Vitest
// does NOT, and its `define` option does not transform `import.meta.*` member
// access, so the composables' client-only guards would read `undefined` and
// bail. This tiny transform rewrites those two tokens to literals in source
// modules (skipping node_modules) so the browser path runs under jsdom.
function nuxtImportMetaFlags(): Plugin {
  return {
    name: "vitest-nuxt-import-meta-flags",
    enforce: "pre",
    transform(code, id) {
      if (id.includes("node_modules")) return null;
      if (!/\.(ts|vue)($|\?)/.test(id)) return null;
      if (!code.includes("import.meta.client") && !code.includes("import.meta.server")) {
        return null;
      }
      const next = code
        .replace(/import\.meta\.client/g, "true")
        .replace(/import\.meta\.server/g, "false");
      return { code: next, map: null };
    },
  };
}

// Vitest config for the frontend unit suite.
//
// Environment: jsdom. The chat store's `persist` block and the persist
// rehydration test need `window.localStorage`; the useChat retry test needs
// `fetch`, `ReadableStream` and `TextDecoder` — all provided by jsdom (plus the
// test's own fetch mock); the markdown sanitisation test needs a DOM for
// isomorphic-dompurify's browser path.
//
// `setupFiles` installs the `piniaPluginPersistedstate` global that the Nuxt
// `pinia-plugin-persistedstate/nuxt` module injects at runtime, so the store
// module imports cleanly outside the Nuxt runtime (see tests/setup/nuxt-globals).
//
// The `~`/`@` aliases mirror Nuxt's srcDir so `~/stores`, `~/utils`,
// `~/composables`, `~/types` imports resolve in tests.
export default defineConfig({
  plugins: [nuxtImportMetaFlags()],
  resolve: {
    alias: {
      "~": resolve(__dirname, "."),
      "@": resolve(__dirname, "."),
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.{test,spec}.ts"],
    // `tests/e2e/*` are Playwright specs (`@playwright/test`), incompatible with
    // the vitest runner; they run via `pnpm test:e2e`, not `pnpm test`.
    exclude: ["**/node_modules/**", "tests/e2e/**"],
    setupFiles: ["tests/setup/nuxt-globals.ts"],
    globals: true,
    // Gate de cobertura de CLAUDE.md (>= 50 % frontend): solo aplica con
    // `pnpm test:coverage` (make test-frontend); `pnpm test` sigue siendo rapido.
    coverage: {
      provider: "v8",
      reporter: ["text-summary"],
      thresholds: { lines: 50, statements: 50, functions: 50 },
    },
  },
});
