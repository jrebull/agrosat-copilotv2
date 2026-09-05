# Plantilla APP — Fase 3

> Te lanzo el orquestador con el numero y titulo de la US. Subagentes: `backend-engineer`
> para `backend/`, `frontend-engineer` para `frontend/`, `agent-engineer` para `ml/agent/`.
> El sistema esta en **mantenimiento**: el articulo no lo evalua y ninguna US de las EPIC 18-27
> depende de el salvo la decision de despliegue real (US-154), que se lee, no se reprograma.
> No toques otros directorios.

1. Lee la guia del directorio SOLO si tu harness no la cargo ya (`backend/AGENTS.md`,
   `frontend/AGENTS.md` o `ml/agent/AGENTS.md`; sus `CLAUDE.md` son espejos identicos) y el spec
   `docs/us-planning/us-XXX.md` (§2 Arquitectura y §3 Interfaces). El spec esta congelado: no
   lo edites; si tienes que desviarte, reportalo en tu resumen citando la seccion.
2. Si Graphify está disponible, úsalo para orientar el impacto y confirma con `rg`, imports y
   tests. Eres consumidor: no ejecutes `make graph-update`.
3. Consulta con Context7 (`--c7`) FastAPI, Pydantic v2, SQLModel, Nuxt 4, Vue 3, MapLibre GL,
   Google ADK.

## Reglas duras del dominio

Backend:
- Router recibe -> service procesa -> model persiste. Sin SQL ni logica de negocio en routers.
- Multi-tenant por `session_id` con RLS forzada; `_check_session_owner` en todo endpoint que
  toque datos de sesion. Respuestas Pydantic, nunca `SQLModel` crudo.
- Config solo via `get_settings()` (`extra="forbid"`); migraciones solo `dbmate new`.
- Vertex/Gemini, GEE y el LLM local se llaman desde services y se mockean en tests.

Frontend:
- `pnpm` exclusivo. `<script setup lang="ts">`; todo texto visible con `t('key')` y la clave en
  `it.json`, `es.json` y `en.json` a la vez (`pnpm i18n:check`).
- SSR-safe (`import.meta.client`); cleanup en `onBeforeUnmount`; secretos solo en
  `runtimeConfig` privado. Para UI carga `/ui-ux-pro-max`.

Agente:
- Tools ADK en `ml/agent/tools/` como `FunctionTool` + Pydantic, con citas en la respuesta
  final; el reasoner es Gemini 2.5 Pro (nunca "3.5 Flash"); el on-prem es Qwen3-30B-A3B
  servido con llama.cpp (el id `Qwen3.5-35B-A3B` no existe).
- Ningun test llama a un LLM real; una evaluacion con Gemini en lote exige confirmacion del
  humano (cuesta dinero).

## Cierre

- `make lint` y la suite que toque (`make test` para backend, `pnpm test` + `pnpm typecheck`
  para frontend, `poetry run pytest tests/ml/agent -q` para el agente) — si fallan,
  corrigelos antes de reportar.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=30 lineas: endpoints/componentes/tools creados, decisiones, desviaciones del spec,
  pendientes o conflictos de frontera (esquemas Pydantic vs tipos TypeScript, contrato de las
  tools).
- No sincronices Engram ni reindexes el grafo: el orquestador integra tu resumen en fuentes
  revisables y decide si actualiza herramientas locales.
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: identico; sin llamadas reales a Vertex/Gemini ni a GEE.
