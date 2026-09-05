# Mapa Skill ↔ Subagente — AgroSatCopilot v2 (MICAI 2027)

> Qué subagente es dueño de qué skills, y a quién lanza la Fase 3 en cada épica. Auto-invoke en [`auto-invoke.md`](auto-invoke.md). Plantillas por dominio en [`subagent-prompts/`](subagent-prompts/).

## Skills y sus owners (31)

| Skill | Owner(s) |
|---|---|
| `agrosat-protocolo-articulo` | `ml-engineer`, `qa-reviewer`, `paper-writer`, `mlops-engineer` (**la más importante**) |
| `agrosat-paper-micai` | `paper-writer` |
| `agrosat-ml-evaluation` | `ml-engineer`, `paper-writer` |
| `agrosat-ml-ensemble` | `ml-engineer` |
| `agrosat-ml-segmentation` | `ml-engineer` |
| `agrosat-ml-baseline` | `ml-engineer` |
| `agrosat-ml-features` | `geo-data-engineer` |
| `agrosat-gee-alphaearth` | `geo-data-engineer` |
| `agrosat-llm-finetuning` | `ml-engineer` (FUTURE, sin hardware) |
| `agrosat-dvc-mlflow` | `mlops-engineer`, `ml-engineer` |
| `agrosat-dagster-mlops` | `mlops-engineer`, `geo-data-engineer` |
| `agrosat-terraform` | `mlops-engineer`, `finops-auditor` |
| `agrosat-gcp-services` | `mlops-engineer`, `finops-auditor` |
| `agrosat-finops` | `finops-auditor`, `mlops-engineer` |
| `agrosat-evidently-drift` | `mlops-engineer` (dormido) |
| `agrosat-backend-api` | `backend-engineer` |
| `agrosat-backend-services` | `backend-engineer` |
| `agrosat-titiler-cog` | `backend-engineer`, `geo-data-engineer` |
| `agrosat-db-migrations` | `backend-engineer`, `geo-data-engineer` |
| `agrosat-db-models` | `backend-engineer` |
| `agrosat-frontend-components` | `frontend-engineer` |
| `agrosat-frontend-composables` | `frontend-engineer` |
| `agrosat-maplibre-geo` | `frontend-engineer` |
| `agrosat-google-adk-agent` | `agent-engineer`, `backend-engineer` |
| `agrosat-spatial-rag` | `agent-engineer` |
| `agrosat-security` | `backend-engineer`, `frontend-engineer`, `security-reviewer` |
| `agrosat-security-audit` | `security-reviewer` |
| `agrosat-testing` | `qa-reviewer`, todos los ingenieros |
| `agrosat-code-review` | `qa-reviewer`, `security-reviewer` |
| `agrosat-git-workflow` | transversal (cualquier sesión) |
| `agrosat-engram-memory` | transversal (cualquier sesión) |

## Los 10 subagentes (`.claude/agents/`)

| Subagente | Cuándo invocarlo | Plantilla F3 |
|---|---|---|
| **`ml-engineer`** | Contrastes, intervalos, mecanismos con valores de conjunto, arnés OOF, reentrenar un miembro | `modeling.md` |
| **`geo-data-engineer`** | Bancos, cargadores, features, AlphaEarth, split espacial, sellado de ground truth | `geo-data.md` |
| **`paper-writer`** | Manuscrito LNCS, preregistro y prosa de `docs/paper/`, bib, figuras, entrega | `paper.md` |
| **`mlops-engineer`** | Ledger y sellado, gates, DVC, MLflow, Makefile, CI | `mlops.md` (en serie, al final) |
| **`qa-reviewer`** | Fase 4 y Fase 6 de toda US; Fase 4 del modo nocturno | — (audita el diff) |
| `backend-engineer` | FastAPI, services, TiTiler, RLS (mantenimiento) | `app.md` |
| `frontend-engineer` | Nuxt 4, MapLibre, i18n (mantenimiento) | `app.md` |
| `agent-engineer` | Tools ADK, Spatial-RAG, evaluación del agente (mantenimiento) | `app.md` |
| `security-reviewer` | Solo si el diff toca backend, frontend, infra o IAM | — |
| `finops-auditor` | Auditoría mensual y cierre de US que gastó GPU, GEE o LLM | — |

## Aislamiento por directorio

En la Fase 3 cada subagente trabaja **solo en su directorio**. Los conflictos que sí hay que resolver al integrar:

| Frontera | Riesgo |
|---|---|
| `ml/ingest`, `ml/features` ↔ `ml/eval` | Columnas y tipos del parquet (`parcel_id` Utf8, `patch_id`, `fold`) |
| `ml/eval` ↔ `paper/micai` | Esquema del JSON de artefactos y el **nombre del régimen** que la sección imprime |
| `ml/eval` ↔ `paper/ARTIFACTS.md` | Qué archivo se sella, con qué commit, y qué fila pasa a `OBSOLETO` |
| `scripts/run_paper_micai_*` ↔ `Makefile` / CI | Targets y gates que consumen la salida nueva |
| `backend/` ↔ `frontend/` | Esquemas Pydantic contra tipos TypeScript |
| `ml/agent/` ↔ `backend/` | Contrato de las tools y del streaming |

## Subagentes por épica (plan por épicas, EPIC 18-27)

| Épica | US | Subagente principal | Apoyo |
|---|---|---|---|
| EPIC 27 Identificar la pérdida, el estimando y la población (**camino crítico**) | US-172 · 173 · 174 · 175 | `paper-writer` (elicitación, preregistro) | `ml-engineer` (`decision_cost`) · `mlops-engineer` (gates) |
| EPIC 18 Cimientos: recuperar, sanear, gobernar | US-118 · 158 · 159 · 119 · 120 · 121 · 122 · 123 | `ml-engineer` (arnés, sanidad, identidad) | `mlops-engineer` (rutas, gates) · `paper-writer` (ADR, consentimiento) |
| EPIC 19 El protocolo, sobre la pérdida ya identificada | US-160 · 124 · 125 · 126 · 155 · 171 · 127 · 128 · 129 | `ml-engineer` | `mlops-engineer` (US-129 gate) |
| EPIC 20 Los cuatro mecanismos bajo el eje común | US-130 · 131 · 132 | `ml-engineer` | — |
| EPIC 21 Quién paga: la tesis | US-133 · 134 · 135 | `ml-engineer` | `paper-writer` |
| EPIC 25 El criterio: qué mecanismo conviene y a quién | US-151 · 152 · 153 · 154 | `ml-engineer` | `paper-writer` (US-153) · `backend-engineer` (US-154, lectura del despliegue real) |
| EPIC 22 Validez externa, y separar el hallazgo del artefacto | US-157 · 136 · 139 | `geo-data-engineer` (US-136 banco nuevo) | `ml-engineer` |
| EPIC 26 El campo de tiro: cada prueba mata un ataque | US-161 a 170 | `ml-engineer` | `paper-writer` (US-170) |
| EPIC 23 Manuscrito: veinte páginas, MICAI y solo MICAI | US-140 · 156 · 141 · 142 · 143 · 149 · 144 | `paper-writer` | `mlops-engineer` (US-149 gate) · `ml-engineer` (cifras) |
| EPIC 24 Entrega | US-145 · 146 · 147 · 148 | `mlops-engineer` | `paper-writer` |
| EPIC 15 / 17 (restos abiertos, opcionales) | reentrenamiento OOF, empaquetado | `ml-engineer` / `mlops-engineer` | — |

`qa-reviewer` entra en la Fase 4 de **todas** las US. `security-reviewer` solo si el diff toca el sistema o la infraestructura. `finops-auditor` al cerrar una US con gasto real. Ninguna US de las EPIC 20, 21, 22 ni 25 corre antes del preregistro firmado (ADR-014 §7): sus subagentes implementan y testean la mecánica, no la corrida.
