# Runbook — MVP conversacional Be My Eyes (EPIC 7/8/9)

Cómo levantar y probar el sistema conversacional implementado la noche del 15-jun.
Decision de inferencia sincrona del MVP en [ADR-012](../decisions/ADR-012-inferencia-sincrona-mvp.md); no existe un ADR de arquitectura completa del sistema conversacional.

## Qué se construyó

| Capa | Ubicación | Estado |
|------|-----------|--------|
| Núcleo agente (Orquestador + Agente Visión + 2 tools) | `ml/agent/` | ✅ 18 tests, mypy limpio |
| Backend (modelos, repos, adaptadores, transporte WS/SSE, 6 endpoints) | `backend/app/` | ✅ 30 tests, 79% cobertura, mypy limpio |
| Frontend (store, composables, ChatPanel, MapView) | `frontend/` | ✅ i18n OK, vitest 4/4, nuxt typecheck limpio |
| Migración `chat_messages` | `db/migrations/` | ✅ creada (pendiente `dbmate up`) |
| Seed demo de parcelas | `scripts/seed_demo_parcels.py` | ✅ ruff+mypy limpio |

Flujo (Be My Eyes asimétrico): el usuario pregunta -> `POST /chat` despacha un job y responde al instante -> un background task corre el **Orquestador** (Gemini/Qwen) que delega en el **Agente Visión** (XGBoost+AlphaEarth + NDVI sobre las parcelas) -> los `AgentEvent` se transmiten por **WebSocket** -> el `ChatPanel` los muestra y el `MapView` pinta las parcelas. El front queda libre tras el POST.

## Prerrequisitos

- Docker (Postgres + PostGIS + pgvector), Poetry, pnpm.
- `.env.local` con `DATABASE_URL`. Para LLM real: `GOOGLE_GENAI_USE_VERTEXAI=true` + credenciales GCP, o `GEMINI_API_KEY`. Sin LLM, el orquestador degrada a una síntesis determinista (marcada) y el resto del flujo funciona igual.

## Levantar el sistema

```bash
# 1. Base de datos
docker compose up -d postgres
dbmate up                                  # aplica migraciones (incl. chat_messages)
make db-seed                               # crea sesión + AOI demo (Toscana)
poetry run python scripts/seed_demo_parcels.py   # 4 parcelas + features en el AOI

# 2. Backend (:8000)
poetry run uvicorn backend.app.main:app --reload --port 8000

# 3. Frontend (:3000)
cd frontend && pnpm dev
```

Abrir http://localhost:3000, seleccionar el AOI y preguntar p.ej. *"¿qué cultivos hay y cómo está su salud?"*.

## Probar el backend sin frontend

```bash
SID=$(curl -s -XPOST localhost:8000/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -XPOST localhost:8000/chat -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"que cultivos hay y su salud\",\"llm_variant\":\"gemini\"}"
# -> {job_id, ws_url}. Conéctate al WS, o usa el fallback SSE:
curl -N localhost:8000/chat/<job_id>/events
```

## Tests

```bash
poetry run pytest tests/ml/agent -q                 # agente (18)
cd backend && poetry run pytest tests/unit -q       # backend (30); integration auto-skip sin Docker
cd frontend && pnpm test && pnpm i18n:check         # frontend (4) + paridad i18n
```

> Correr las suites de `tests/` (raíz) y `backend/tests/` juntas desde la raíz falla por colisión del paquete `tests` — ejecútalas por separado (como arriba).

## Limitaciones conocidas del MVP (por diseño / alcance v8)

- **Ojos = stack ML** (XGBoost+AlphaEarth + NDVI). El ojo VLM (Gemma/Qwen-VL) es post-presentación: se enchufa como sub-tool `describe_scene_vlm` sin tocar el orquestador.
- `classify_parcel` usa el **fallback honesto** a `crop_class` almacenado cuando no hay modelo en el MLflow Registry (cita `source="stored:crop_class"`). Con el modelo registrado usa el embedding real.
- **`JobRegistry` en memoria**: no sobrevive a reinicios ni escala multi-instancia. Pub/Sub + Cloud Function lo reemplaza post-presentación (OUT en v8).
- **Auth**: `user_id` demo (sin Clerk productivo). RLS por sesión se endurece en US-051.
- Embeddings del seed demo son placeholders (64-dim a cero); el `crop_class`/NDVI sí son realistas.

## Pendiente (siguiente sesión)

- `dbmate up` de la migración `chat_messages` (requiere Docker corriendo).
- Validar el flujo end-to-end con Postgres real + un LLM (Vertex o API key).
- `/timeseries`, `/tiles` (TiTiler), `/stac/search`, switch A/B completo, RLS (US-051), variante Qwen en H100.
- ESLint del frontend: falta config flat ESLint 9 + plugins (`eslint-plugin-vue`, etc.) — requiere `pnpm add` con acuerdo de equipo.
