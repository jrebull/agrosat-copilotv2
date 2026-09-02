# Frontend — AgroSatCopilot

> Sub-guía del orquestador. Las reglas NON-NEGOTIABLE viven en [`../CLAUDE.md`](../CLAUDE.md) — aquí no se repiten, solo lo operativo de `frontend/`.

Web app **Nuxt 4 SSR** trilingüe (it/es/en). Mapa MapLibre, chat SSE y switch A/B LLM son el destino del producto, **implementados** (EPIC 9, US-057/058).

## Estado

**IMPLEMENTADO** (EPIC 9). Gates verdes (validacion 2026-06-25): vitest 53/7, typecheck 0 err, i18n parity (it/es/en), eslint limpio, build SSR completo.

- Componentes reales: `ChatDock.vue` (chat, refactor del legacy `ChatPanel.vue`), `MessageBubble.vue` (markdown via `marked` + `isomorphic-dompurify`), `ToolActivity.vue`, `MapCanvas.vue`, `AppHeader.vue`.
- Composables reales: `useChat` (SSE + `parseSseFrame`), `useMap`, `useAoi`, `useBasemap`, `useSession`. **NO** existe `useSSE.ts` (el parser vive en `useChat`).
- Stores Pinia: `chat.ts` (persistido via `pinia-plugin-persistedstate`, pick `llmVariant`+`cropModel`; el transcript vive en Postgres desde US-080, ya NO se persiste), `map.ts`.
- La validacion E2E en navegador real (Playwright) requiere `backend/.env.local` + sesion sembrada; pendiente de entorno (no de codigo).

## Comandos

```bash
pnpm dev          # Nuxt dev server :3000
pnpm build        # SSR build → .output/
pnpm lint         # eslint (.vue,.ts,.tsx)
pnpm typecheck    # nuxt typecheck (vue-tsc)
pnpm test         # vitest run  (sin tests aún)
pnpm test:e2e     # playwright  (sin tests aún)
pnpm i18n:check   # paridad de claves it/es/en

make i18n-check     # == pnpm i18n:check (entra a frontend/)
make test-frontend  # == pnpm test
make test-e2e       # == pnpm test:e2e
make bootstrap      # poetry install + (cd frontend && pnpm install)
```

`pnpm` exclusivo (`pnpm>=10`, `node>=20`). Nunca npm/yarn.

## Stack local

| Capa | Lib (real en `package.json`) |
|------|------------------------------|
| Framework | Nuxt 4 SSR (Vue 3 Composition) |
| UI | `@nuxt/ui-pro` |
| i18n | `@nuxtjs/i18n` (it default, prefix_except_default) |
| Estado | `@pinia/nuxt` + `pinia` |
| CSS | Tailwind **v4** — tema vía `@theme` en `assets/css/main.css`, NO config v3 |
| Mapa | `maplibre-gl` 5.24 — en uso (MapCanvas + `useMap`). `deck.gl` **removido** (US-058, 0 usos reales; reañadir con `pnpm add` y acuerdo de equipo si una US futura necesita densidad alta) |
| Chat | `@ai-sdk/vue` — instalado, sin usar |
| Test | `vitest` + `@playwright/test` — instalados, sin config |

`tailwind.config.ts` existe solo para tooling legacy; la fuente de verdad del tema es `@theme` en `main.css`.

## Convenciones (✅/❌)

- ✅ Componentes con `<script setup lang="ts">` y `const { t } = useI18n()` para todo texto visible → `t('key')`.
- ❌ Strings de UI hardcodeados en template o script.
- ✅ Al agregar una clave i18n, añadirla a `it.json` **y** `es.json` **y** `en.json` simultáneamente (lo valida `scripts/i18n_check.mjs` comparando claves aplanadas; **no** es un plugin de eslint).
- ✅ SSR-safe: `import.meta.client` antes de tocar `window`/browser APIs.
- ✅ Secretos solo en `runtimeConfig` privado server-side, nunca en `runtimeConfig.public`.
- ❌ Inferencia ML, Vertex AI / vLLM / GEE o llamadas a modelos desde el cliente — todo va por `/chat` SSE al backend.

## No tocar

- `pnpm-lock.yaml` — solo cambia vía `pnpm add`/`pnpm install`.
- `.nuxt/`, `.output/`, `node_modules/` — generados; nunca editar ni commitear.
- Nunca agregar una clave i18n a un solo locale: rompe `i18n:check` y bloquea el merge.
- `pinia-plugin-persistedstate` SI esta en `package.json` (lo usa `stores/chat.ts`). `vue-echarts` + `echarts` **YA estan** (E12, via `pnpm add`): los usa `components/chat/CropProbabilityChart.vue`, que registra solo los modulos necesarios (BarChart + Grid + Tooltip + Canvas) y monta client-only. Importar otro modulo de echarts = registrarlo ahi, no anadir dependencia.

## Tests

**53 tests vitest** en 7 archivos (sse-parser, markdown, chat-store, chat-persist, use-chat-retry, use-map, map-store). `vitest.config.ts` presente. `@playwright/test` instalado para E2E (los flujos en vivo requieren backend + `.env.local` + sesion sembrada). Cobertura objetivo ≥50 % frontend (ver checklist root).

`pnpm test` corre los 53 casos; `pnpm test:e2e` queda para la validacion en navegador real cuando la app este levantada.

## Skills

| Acción | Skill |
|--------|-------|
| Componente / página / layout Vue | `agrosat-frontend-components` |
| Composable, Pinia store, SSE, middleware | `agrosat-frontend-composables` |
| MapLibre / deck.gl / AOI / overlay COG | `agrosat-maplibre-geo` (+ `agrosat-titiler-cog`) |
| Auth Clerk, role guard, CSP | `agrosat-security` |
| Tests Vitest / Playwright | `agrosat-testing` |
