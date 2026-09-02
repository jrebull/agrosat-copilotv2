# Database — AgroSatCopilot

> Scope `db/`. Hereda y NO repite las reglas root: ver [`../AGENTS.md`](../AGENTS.md) (regla 3 multi-tenant por `session_id`, regla 11 solo `dbmate`, regla 5 secrets).

PostgreSQL 15 + PostGIS + pgvector. Migraciones en SQL puro con dbmate (framework-agnóstico). El ORM aún no existe.

## Estado

10 migraciones aplicadas (`dbmate status`; fuente de verdad: [`docs/STATUS.md`](../docs/STATUS.md)) → **6 tablas reales**:

- `chat_sessions` (UUID, `user_id`, `llm_model` con CHECK para `gemini | qwen-api | qwen-onprem | gemma | qwen-vl`, timestamps) — clave multi-tenant.
- `chat_messages` (FK `session_id`, rol, contenido, timestamps) — US-080; función `list_chat_sessions(text)`.
- `aois` (BIGSERIAL, FK `session_id`, `geom GEOMETRY(POLYGON,4326)`, `label`, `area_ha`).
- `parcels` (FK `session_id`/`aoi_id`, `geom`, `crop_class`, `confidence`, `area_ha`, `year`, `canonical_parcel_id`) — US-015 / US-081.
- `features_parcels` (FK `parcel_id`, `alphaearth_embedding VECTOR(64)` nullable, `ndvi_stats`/`phenology` JSONB, columnas fenológicas escalares) — US-015.
- `rag_documents` (embeddings pgvector para Spatial-RAG) — US-047.

Extensiones activas: `postgis`, `postgis_topology`, `vector`, `pg_stat_statements`. **`pgstac` NO está instalado** (CREATE EXTENSION comentado, requiere imagen Docker con la extensión).

**RLS aplicada** (US-051, migración `20260620000418_rls_multi_tenant.sql`): `ENABLE` + `FORCE ROW LEVEL SECURITY` y política `tenant_isolation` por `session_id` en `chat_sessions`, `aois`, `parcels`, `features_parcels` y `chat_messages`; rol de aplicación `agrosat_app` (NOLOGIN, solo privilegios). No existen: STAC (`pgstac`), `alphaearth_tiles`, `sentinel2_scenes` ni `agent_sessions`.

## Comandos

```bash
make db-migrate              # dbmate up
make db-rollback             # dbmate down
make db-new name=create_xxx  # dbmate new -> 20260YYYYMMDDHHMMSS_create_xxx.sql
make db-status               # dbmate status
make db-seed                 # poetry run python scripts/seed.py (idempotente)
make db-shell                # docker compose exec postgres psql -U agrosat -d agrosat
make db-test-us015           # pytest round-trip migraciones US-015
```

## Stack local

Postgres vía `docker compose` (servicio `postgres`). DSN en `.env.local` (`DATABASE_URL`); el seed normaliza `postgresql+asyncpg://` → `postgresql://`. Tests usan testcontainers con imagen `agrosat-postgres:15-3.4-pgvector` (fallback `postgis/postgis:15-3.4`, se saltan si falta pgvector/Docker).

## Convenciones (✅/❌)

- ✅ Migración reversible: `-- migrate:up` + `-- migrate:down`, idempotente con `IF NOT EXISTS`.
- ✅ Geometría `GEOMETRY(POLYGON,4326)` + índice GIST (`USING GIST (geom)`).
- ✅ Embeddings `VECTOR(64)` nullable y **sin índice** (se poblará en US-016; IVFFlat/HNSW vendrá después).
- ✅ Timestamps `TIMESTAMPTZ NOT NULL DEFAULT now()`.
- ✅ Enum-like vía `TEXT CHECK (llm_variant IN ('gemini','qwen35'))`, **no** un ENUM nativo de PG.
- ✅ FK con `ON DELETE` explícito (`CASCADE` en features/aois, `SET NULL` en `parcels.aoi_id`).
- ✅ BTREE en columnas de filtro frecuente (`session_id`, `year`, `parcel_id`).
- ❌ Editar una migración ya aplicada → siempre `dbmate new` rollforward.
- ❌ `SQLModel.metadata.create_all()` en prod (regla root 11).

## No tocar

- Historial `schema_migrations` (lo gestiona dbmate; no editar a mano).
- Dimensión `VECTOR(64)` bloqueada: `test_dbmate_up_creates_features_parcels` asegura `atttypmod == 64`.
- `UNIQUE (parcel_id, year)` (`features_parcels_parcel_year_uniq`): una fila por parcela/año; multi-ciclo añadirá `season` por rollforward.
- Idempotencia del seed: `DEMO_AOI_LABEL` ("Demo parcel - Tuscany") es la clave de skip; `DEMO_USER_ID`, `DEMO_AOI_WKT` y SRID 4326 son contrato del demo.

## Tests

- `tests/db/test_migrations_us015.py` — testcontainers: tipos vía `information_schema`, `VECTOR(64)`, FK CASCADE, UNIQUE, índices GIST/BTREE y round-trip up→down→up. Ejecuta `make db-test-us015`.
- `backend/app/models/` sigue **vacío** (solo `__init__.py`): no hay modelos SQLModel/GeoAlchemy2 aún. Cualquier ORM debe reflejar exactamente este schema.

## Skills

- [agrosat-db-migrations](../.claude/skills/agrosat-db-migrations/SKILL.md) — dbmate, índices GIST/IVFFlat/HNSW, extensiones, seed.
- [agrosat-db-models](../.claude/skills/agrosat-db-models/SKILL.md) — SQLModel + GeoAlchemy2 (futuro ORM).
- [agrosat-security](../.claude/skills/agrosat-security/SKILL.md) — RLS/ACL (aún no aplicado en `db/`).
