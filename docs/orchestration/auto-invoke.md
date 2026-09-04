# Auto-Invoke Table — AgroSatCopilot

> Tabla operativa de qué skill cargar antes de cada acción. Catálogo en [`skills-catalog.md`](skills-catalog.md). Mapa skill→subagente en [`skill-owners.md`](skill-owners.md).

## Backend & Database

| Acción | Skill |
|--------|-------|
| Crear/modificar endpoint FastAPI | `agrosat-backend-api` |
| Crear/modificar service backend | `agrosat-backend-services` |
| Integrar TiTiler / servir COG | `agrosat-titiler-cog` |
| Endpoint `/chat` SSE con ADK | `agrosat-backend-api` + `agrosat-google-adk-agent` |
| Crear migración dbmate | `agrosat-db-migrations` |
| Crear/modificar modelo SQLModel + PostGIS | `agrosat-db-models` |
| Agregar índice GIST/BTREE | `agrosat-db-migrations` |
| Configurar pgstac collection | `agrosat-db-migrations` + `agrosat-backend-api` |
| Agregar RLS policy por session_id | `agrosat-security` + `agrosat-db-migrations` |
| Crear Pub/Sub worker async | `agrosat-backend-services` + `agrosat-gcp-services` |

## Frontend

| Acción | Skill |
|--------|-------|
| Crear componente Vue/Nuxt | `agrosat-frontend-components` |
| Crear composable / Pinia store | `agrosat-frontend-composables` |
| Integrar mapa MapLibre / deck.gl | `agrosat-maplibre-geo` |
| Dibujar AOI con maplibre-gl-draw | `agrosat-maplibre-geo` |
| Streaming SSE chat | `agrosat-frontend-composables` |
| Switch A/B LLM en UI | `agrosat-frontend-components` |
| i18n it/es/en (3 locales) | `agrosat-frontend-components` |

## ML / Modelado

| Acción | Skill |
|--------|-------|
| Calcular índice espectral | `agrosat-ml-features` |
| Entrenar baseline XGBoost + AlphaEarth | `agrosat-ml-baseline` |
| Implementar U-Net / DeepLab / SegFormer | `agrosat-ml-segmentation` |
| Implementar U-TAE / TSViT / Swin-UNETR | `agrosat-ml-segmentation` |
| Fine-tune Gemma 4 26B-MoE LoRA | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Fine-tune Qwen3-VL LoRA | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Serving vLLM Qwen3.5-35B-A3B | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Construir ensamble | `agrosat-ml-ensemble` |
| Evaluar contra AgroMind / GeoAnalystBench | `agrosat-ml-evaluation` |

## Geoespacial & Agente

| Acción | Skill |
|--------|-------|
| Descargar AlphaEarth Foundations | `agrosat-gee-alphaearth` + `agrosat-dagster-mlops` |
| Descargar Sentinel-2 L2A vía CDSE | `agrosat-ml-features` |
| Crear tool ADK nuevo | `agrosat-google-adk-agent` |
| Implementar Spatial-RAG híbrido | `agrosat-spatial-rag` |
| Configurar planner Plan-and-React | `agrosat-google-adk-agent` |
| Deploy agente a Vertex AI Agent Engine | `agrosat-google-adk-agent` + `agrosat-gcp-services` |

## MLOps & Infra

| Acción | Skill |
|--------|-------|
| Definir asset Dagster | `agrosat-dagster-mlops` |
| Versionar dataset con DVC | `agrosat-dvc-mlflow` |
| Registrar experimento MLflow | `agrosat-dvc-mlflow` |
| Crear módulo Terraform | `agrosat-terraform` |
| Configurar servicio GCP | `agrosat-gcp-services` |
| Encender / apagar VM H100 | `agrosat-azure-h100` |
| Pipeline drift Evidently | `agrosat-evidently-drift` |
| Auditar costo cloud | `agrosat-finops` |
| Crear workflow GitHub Actions | `agrosat-terraform` + `agrosat-gcp-services` |

## Seguridad & QA

| Acción | Skill |
|--------|-------|
| Implementar auth Clerk / RBAC | `agrosat-security` |
| Configurar rate limit, CSP, CORS | `agrosat-security` |
| Audit logging | `agrosat-security` |
| Audit OWASP / CIS GCP | `agrosat-security-audit` |
| Pre-deploy security checklist | `agrosat-security-audit` |
| Escribir tests pytest / Playwright | `agrosat-testing` |
| Mockear Vertex AI / vLLM / GEE | `agrosat-testing` |
| Review de PR | `agrosat-code-review` |
| Crear commit / branch / PR | `agrosat-git-workflow` |
| Cerrar User Story | `agrosat-git-workflow` |
| Persistir decisión entre sesiones Claude Code, solo si el binario y MCP están conectados | `agrosat-engram-memory` (dev opcional) |
