# Catálogo de Skills — AgroSatCopilot v2 (MICAI 2027)

> Catálogo de las 32 skills `agrosat-*`. Resumen ejecutivo en [`AGENTS.md`](../../AGENTS.md). Detalle de cada skill en su `SKILL.md`.

Las skills viven en `.claude/skills/<nombre>/SKILL.md` con frontmatter YAML (`name`, `description`, `allowed-tools`). Claude las carga automáticamente por `description` o por invocación manual `/<nombre>`. La skill de Azure H100 sigue gobernada por ADR-009; que el artículo MICAI no dependa de ella no la retira del producto.

## Artículo MICAI 2027 (2)

| Skill | Descripción |
|---|---|
| `agrosat-protocolo-articulo` | **La más importante.** Preregistro, contrato del estimando (`estimando-v1.json`), régimen nombrado, unidad parcela / clúster `patch_id`, los tres defectos reparados, ledger y sellado, candados ADR-014 |
| `agrosat-paper-micai` | Manuscrito LNCS en `paper/micai/`: doble ciego, bib generado desde la matriz verificada, cifras solo desde filas `SELLADO`, figuras reproducibles, gates `micai-*` |

## Modelado y evaluación (5)

| Skill | Descripción |
|---|---|
| `agrosat-ml-evaluation` | Métricas por parcela y por píxel, figuras interpretadas, benchmarks del sistema (AgroMind, GeoAnalystBench) cuando una US los pide |
| `agrosat-ml-ensemble` | Voting, bagging, blending (Optuna) y stacking en `ml/ensemble/`; el arnés OOF y la trampa del meta-modelo in-sample |
| `agrosat-ml-segmentation` | Miembros densos implementados: U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT(-pheno), AnySat; reentrenamiento en RTX 4070 o L4 spot |
| `agrosat-ml-baseline` | XGBoost, LightGBM y RF sobre AlphaEarth 64-dim + índices, con split espacial |
| `agrosat-llm-finetuning` | Gemma 4 LoRA y serving de LLM: **FUTURE** (ADR-011) y fuera del alcance del artículo |

## Datos y features (2)

| Skill | Descripción |
|---|---|
| `agrosat-ml-features` | Cargadores PASTIS-R y BreizhCrops, índices espectrales, features temporales, fusión Polars, `build_spatial_kfold`, `canonical_parcel_id` |
| `agrosat-gee-alphaearth` | AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 vía GEE con ADC, export a GCS/COG |

## MLOps e infraestructura (7)

| Skill | Descripción |
|---|---|
| `agrosat-dvc-mlflow` | DVC sobre GCS, MLflow con `data_version` + `code_version`, sellado de artefactos y gates probados en negativo |
| `agrosat-dagster-mlops` | Assets Dagster con lineage DVC ↔ MLflow |
| `agrosat-terraform` | Terraform GCP y Azure según la arquitectura ratificada |
| `agrosat-gcp-services` | Cloud Run, Cloud SQL, GCS, Pub/Sub, Vertex AI, Secret Manager, IAM |
| `agrosat-azure-h100` | Operación acotada de la VM Azure H100, solo con US y presupuesto autorizados |
| `agrosat-finops` | Costos GCP y Azure: scale-to-zero, GPU por US y topes presupuestarios |
| `agrosat-evidently-drift` | Reportes de drift sobre bandas Sentinel-2 y embeddings (dormido) |

## Sistema — backend, datos, frontend, agente (10)

| Skill | Descripción |
|---|---|
| `agrosat-backend-api` | Endpoints FastAPI `/chat` SSE, `/aois`, `/timeseries`, `/stac/search`, `/tiles` |
| `agrosat-backend-services` | Service layer, DI, workers, integración ADK |
| `agrosat-titiler-cog` | TiTiler dinámico sobre COG, colormaps, mosaic JSON |
| `agrosat-db-migrations` | dbmate SQL, extensiones, índices GIST/HNSW, RLS por `session_id` |
| `agrosat-db-models` | SQLModel + GeoAlchemy2, columnas geometry y vector |
| `agrosat-frontend-components` | Vue 3 / Nuxt UI Pro, chat streaming, i18n it/es/en |
| `agrosat-frontend-composables` | `useChat`, `useMap`, stores Pinia, middleware |
| `agrosat-maplibre-geo` | MapLibre GL, AOI, overlays COG |
| `agrosat-google-adk-agent` | Google ADK, 9 FunctionTools, Gemini 2.5 Pro + Qwen on-prem (llama.cpp) |
| `agrosat-spatial-rag` | Spatial-RAG híbrido PostGIS + pgvector |

## Seguridad y QA (4)

| Skill | Descripción |
|---|---|
| `agrosat-security` | Clerk OAuth, JWT, RBAC, rate limiting, CSP, RLS, audit logging |
| `agrosat-security-audit` | OWASP Top 10, CIS GCP, pre-deploy checklist, gitleaks, aislamiento cross-session |
| `agrosat-testing` | pytest en dos suites separadas, fixtures con filas reales, mocks de Vertex/GEE/LLM, Playwright |
| `agrosat-code-review` | Fase 4 del loop: DRY, SoC, reglas del orquestador y anti-patrones del artículo sobre el diff |

## Transversal (2)

| Skill | Descripción |
|---|---|
| `agrosat-git-workflow` | Ramas `feature/E{epic}-US-XXX-{slug}`, Conventional Commits con scope de épica, PR y cierre verificable de US |
| `agrosat-engram-memory` | Memoria local y opcional de desarrollo; compartirla requiere aceptar ADR-015 y todavía no está habilitado |
