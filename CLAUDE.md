# AgroSatCopilot — Guía Operativa del Orquestador

**Proyecto**: SaaS conversacional open-source para análisis satelital agrícola — orquestador único de agentes IA.

**Stack**: FastAPI + Polars | Nuxt 4 SSR | PostgreSQL 15 + PostGIS + pgvector | Google ADK | Gemini 2.5-pro + Qwen vLLM on-prem (Gemma 4 LoRA = FUTURE, [ADR-011](docs/decisions/ADR-011-gemma4-lora-future.md)) | Dagster + dbmate + DVC + MLflow | Terraform GCP + Azure H100 NVL 96GB.

> Plan vigente, US, calendario, presupuesto y métricas de éxito: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](context/RefinamientoPlaneacionAgroSatCopilot_v8.md) (ratificado por [ADR-009](docs/decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md); calendario en [ADR-008](docs/decisions/ADR-008-rediseno-calendario-presentacion-27jun.md)).

## Cómo usar esta guía

- **Raíz** ([`CLAUDE.md`](CLAUDE.md) y [`AGENTS.md`](AGENTS.md), espejos idénticos): normas transversales — aplican a todo el repo.
- **Guía de carpeta** (`<dir>/AGENTS.md`): sobreescribe la raíz en caso de conflicto dentro de su scope.
- Las guías anidadas se cargan **on-demand** al entrar al directorio relevante.

## Comandos

```bash
make check            # lint + secrets-scan + i18n-check (OBLIGATORIO antes de PR)
make lint             # ruff + mypy + pnpm lint
make test             # pytest backend con cobertura (>=70 %)
make test-ml          # pytest tests/ml (excluye `slow`)
make test-frontend    # vitest con cobertura (>=50 %)
make test-all         # los tres anteriores
make notebooks-check  # papermill end-to-end (notebooks con outputs preservados)
make i18n-check       # it/es/en sincronizadas

poetry add <pkg>      # deps Python (nunca pip ad-hoc)
pnpm add <pkg>        # deps frontend (nunca npm/yarn)

dbmate up             # aplicar migraciones
dbmate new <slug>     # crear migración rollforward

pytest tests/ml/train/test_baseline.py::test_name -q   # un solo test
```

## Stack — Decisiones Irrevocables (NO cambiar sin equipo)

| Capa | Modelo / lib | Nota clave |
|------|--------------|------------|
| FM EO | AlphaEarth Foundations (`SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1) | GEE gratis CC-BY-4.0 · global incl. México · NO entrenar FM propio |
| Feature self-sup | DINOv3-satellite | `facebook/dinov3-vitl16-pretrain-sat493m` frozen |
| VLM principal | Gemma 4 26B-A4B-MoE (`google/gemma-4-26B-A4B-it`) | Apache 2.0 · LoRA `target_parameters` (QLoRA bloqueado, MoE 3D) · **FUTURE post-27jun** ([ADR-011](docs/decisions/ADR-011-gemma4-lora-future.md)) · el id `gemma-4-26b-it` NO existe |
| VLM comparativo | Qwen3-VL-30B-A3B | MoE 30B/3B · 256K ctx |
| LLM cloud | Gemini 2.5 Pro (GA, default) | Vertex AI · **1M ctx** · $1.25/$10 por M · reasoner del copiloto |
| LLM on-prem | Qwen MoE-A3B Int4 (`Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`) | vLLM single-GPU GPTQ-Int4 ([US-048](docs/serving/qwen35.md)) · el id `Qwen3.5-35B-A3B` no existe |
| Framework agente | Google ADK | Tracing built-in + Vertex AI Agent Engine |

**Arquitecturas por rúbrica**: EPIC 5 segmentación (U-Net · DeepLabv3+ · SegFormer-B2 · U-TAE · TSViT · Swin-UNETR) · EPIC 6 ensambles (Voting top-3 · Bagging XGB+AlphaEarth · Stacking +Gemma 4 · Blending Optuna).

**Descartados — no reactivar**: Prithvi-EO-2.0, MiniMax-M2.7, Kimi K2.6, Llama 3.3-70B QLoRA, LangGraph, Prefect, Alembic, DuckDB principal, PWA+Tauri.

## Reglas de código NON-NEGOTIABLE

- **Idioma**: código (identificadores, comentarios, docstrings Google-style) en inglés; prosa visible al lector (notebooks markdown/prints/plots, docs `.md`) en español neutro.
- **Sin emojis** en código, comentarios, prints, commits ni logs.
- **Logging**: `structlog.get_logger()`, nunca `print()` en producción.
- **Type hints** obligatorios en todo Python.
- **DRY**: función usada 2+ veces → `backend/app/utils/`, `ml/utils/` o `frontend/composables/`.
- **SoC**: router recibe → service procesa → model persiste. Tools ADK en `ml/agent/tools/`, nunca en routers; sin lógica de negocio en routers ni componentes Vue.
- **Multi-tenant por `session_id`**: toda query filtra por sesión/usuario.
- **i18n**: todo texto visible en `frontend/i18n/locales/{it,es,en}.json` simultáneamente; sin strings sin `t('key')`.
- **Secrets**: jamás hardcodear. `.env.local` en dev, Secret Manager (GCP) / Key Vault (Azure) en prod.
- **DVC** para rasters/COG/GeoTIFF/pesos — nunca al repo Git. MLflow con tags `data_version` + `code_version`.
- **Migraciones**: solo `dbmate up` / `dbmate new`. Jamás `SQLModel.metadata.create_all()` en prod ni modificar migraciones aplicadas.
- **Notebooks**: se commitean ejecutados end-to-end con todos sus outputs (HTML tables + PNG inline). Sin `.pre-commit-config.yaml` ni `nbstripout` en quality gates.
- **Commits sin trailer `Co-Authored-By`** de asistentes IA — la autoría queda en el `Author:` real.

## QA Gate antes de PR

1. `make check` limpio (lint + secrets + i18n).
2. Tests cobertura ≥70 % backend, ≥50 % frontend.
3. Si tocó schema: `dbmate up`. Si entrenó: MLflow con `data_version` + `code_version`. Si generó data: `dvc add` + commit del `.dvc`.
4. Si la US incluye notebook: papermill end-to-end + commit con outputs poblados.
5. Rúbrica del Avance verificada en [`docs/general/Rubricas Integrador.html`](docs/general/Rubricas%20Integrador.html).

## Git y PR

- Rama: `feature/E{epic}-US-XXX-{slug}`.
- Conventional Commits con scope de épica: `feat(E6): ...`, `fix(E3): ...`, `docs(E5): ...`.
- `make check` limpio antes de abrir PR a `develop`.

## Routing por directorio

| Directorio | Guía | Especialidad |
|------------|------|--------------|
| `backend/` | [backend/AGENTS.md](backend/AGENTS.md) | FastAPI, SQLModel, TiTiler, SSE, Pub/Sub workers |
| `frontend/` | [frontend/AGENTS.md](frontend/AGENTS.md) | Nuxt 4 SSR, MapLibre + deck.gl, @ai-sdk/vue, i18n |
| `ml/` | [ml/AGENTS.md](ml/AGENTS.md) | Segmentación, fine-tune LoRA, AlphaEarth, DINOv3 |
| `ml/agent/` | [ml/agent/AGENTS.md](ml/agent/AGENTS.md) | Google ADK, 9 tools geoespaciales, Spatial-RAG |
| `db/` | [db/AGENTS.md](db/AGENTS.md) | dbmate, PostGIS, pgvector, pgstac, RLS por sesión |
| `infrastructure/` | [infrastructure/AGENTS.md](infrastructure/AGENTS.md) | Terraform GCP + Azure H100, Cloud Build |
| `dagster_project/` | [dagster_project/AGENTS.md](dagster_project/AGENTS.md) | Assets, jobs, schedules, DVC ↔ MLflow lineage |
| `notebooks/` | [notebooks/AGENTS.md](notebooks/AGENTS.md) | Avances curso, EDA Polars, papermill |
| `paper/` | [paper/AGENTS.md](paper/AGENTS.md) | Paper Track opcional, GEO-Bench-2, AgroMind-IT/ES |

## Skills y plan

- Qué skill `agrosat-*` cargar antes de cada acción (30 skills): [`docs/orchestration/auto-invoke.md`](docs/orchestration/auto-invoke.md).
- Plan SCRUM, US, calendario, presupuesto y métricas: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](context/RefinamientoPlaneacionAgroSatCopilot_v8.md).

## Estilo de respuesta

- Antes del primer tool call: una frase con el plan (≤20 palabras).
- Tareas con >3 tool calls o >30 s: `TodoWrite` al inicio.
- Código > prosa: el diff es la respuesta; respuestas triviales ≤4 líneas.
- Tool calls independientes en paralelo · Grep antes que Read · solo lo preguntado.
- Sin preámbulos ("Perfecto, voy a...", "Listo, he..."), sin narrar tool calls, sin emojis.
