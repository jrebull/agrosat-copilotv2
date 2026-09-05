# Auto-Invoke Table — AgroSatCopilot v2 (MICAI 2027)

> Qué skill cargar antes de cada acción. Catálogo en [`skills-catalog.md`](skills-catalog.md). Mapa skill→subagente en [`skill-owners.md`](skill-owners.md). Fases del loop en [`prompts-optimizers-fable.md`](prompts-optimizers-fable.md).

## Reglas previas a todo

1. **Antes de escribir o auditar cualquier comparación, intervalo, contraste, artefacto o cifra, cargar `agrosat-protocolo-articulo`.** Es la traducción ejecutable del preregistro, del estimando y del ledger.
2. **Antes de tocar `paper/micai/` o `docs/paper/`, cargar `agrosat-paper-micai`.**
3. **Herramientas opcionales no son bloqueos.** Si Graphify está instalado, puede orientar el
   impacto antes de confirmar con `rg`, imports y tests. Engram puede aportar memoria local, nunca
   una fuente normativa ni una sincronización implícita (ADR-015).

## Artículo MICAI 2027

| Acción | Skill |
|---|---|
| Definir o revisar un contraste, un intervalo pareado, la multiplicidad o la potencia | `agrosat-protocolo-articulo` |
| Generar un artefacto en `reports/paper_micai/<fase>/` | `agrosat-protocolo-articulo` + `agrosat-ml-evaluation` |
| Sellar un artefacto o cambiar su estado en `paper/ARTIFACTS.md` | `agrosat-dvc-mlflow` + `agrosat-protocolo-articulo` |
| Regenerar los artefactos `OBSOLETO` con el módulo reparado (US-124, US-125) | `agrosat-protocolo-articulo` |
| Implementar un mecanismo con valores de conjunto (conforme, retroceso jerárquico) | `agrosat-protocolo-articulo` + `agrosat-ml-ensemble` |
| Elicitar o codificar la tabla de pérdidas y el espacio de costes (US-172, US-175, `decision_cost`) | `agrosat-protocolo-articulo` |
| Sanidad, provenance e identidad de los miembros heredados (US-119, US-120) | `agrosat-protocolo-articulo` + `agrosat-ml-evaluation` |
| Escribir una sección, figura o tabla del manuscrito | `agrosat-paper-micai` |
| Añadir una referencia o regenerar el bib | `agrosat-paper-micai` |
| Compilar, verificar anonimato, camera-ready, empaquetar | `agrosat-paper-micai` |
| Editar el preregistro, el estimando o el registro de afirmaciones retiradas | `agrosat-protocolo-articulo` + `agrosat-paper-micai` |
| Cambiar el estado de una US en el cuaderno (`plan.html`) y validarlo | `agrosat-git-workflow` (`make plan-check`) |

## Datos y features

| Acción | Skill |
|---|---|
| Descargar o agregar AlphaEarth (GEE, ADC) | `agrosat-gee-alphaearth` + `agrosat-dagster-mlops` |
| Cargar PASTIS-R, BreizhCrops o un banco nuevo; features Polars | `agrosat-ml-features` |
| Sellar un banco nuevo (ground truth por parcela, soporte por clase, procedencia) | `agrosat-ml-features` + `agrosat-protocolo-articulo` |
| Split espacial (`build_spatial_kfold`) | `agrosat-ml-features` |
| Calcular índices espectrales o series temporales | `agrosat-ml-features` |

## Modelado (miembros y predictores)

| Acción | Skill |
|---|---|
| Reentrenar un miembro denso (TSViT, U-TAE, U-Net, DeepLabv3+, SegFormer, AnySat) en RTX 4070 o L4 spot | `agrosat-ml-segmentation` + `agrosat-dvc-mlflow` |
| Predictor tabular (XGBoost sobre AlphaEarth) en un banco | `agrosat-ml-baseline` |
| Ensamble, OOF, arnés `ml/eval/oof/` | `agrosat-ml-ensemble` + `agrosat-protocolo-articulo` |
| Benchmarks del sistema (AgroMind, GeoAnalystBench) — solo si una US lo pide | `agrosat-ml-evaluation` |
| Gemma 4 LoRA / serving de LLM | `agrosat-llm-finetuning` + `agrosat-azure-h100` (FUTURE según ADR-011; fuera del alcance del artículo) |

## Sistema (mantenimiento)

| Acción | Skill |
|---|---|
| Crear/modificar endpoint FastAPI o service | `agrosat-backend-api` / `agrosat-backend-services` |
| Servir COG con TiTiler | `agrosat-titiler-cog` |
| Migración dbmate, modelo SQLModel, RLS | `agrosat-db-migrations` / `agrosat-db-models` / `agrosat-security` |
| Componente Vue, composable, mapa MapLibre, i18n | `agrosat-frontend-components` / `agrosat-frontend-composables` / `agrosat-maplibre-geo` |
| Tool ADK, planner, Spatial-RAG | `agrosat-google-adk-agent` / `agrosat-spatial-rag` |

## MLOps e infraestructura

| Acción | Skill |
|---|---|
| Definir asset Dagster | `agrosat-dagster-mlops` |
| Versionar dataset, OOF o checkpoint con DVC; registrar corrida MLflow | `agrosat-dvc-mlflow` |
| Crear o probar en negativo un gate (`scripts/*_check.py`, `make *-check`) | `agrosat-dvc-mlflow` + `agrosat-protocolo-articulo` |
| Terraform GCP `dev` (dormido) | `agrosat-terraform` + `agrosat-gcp-services` |
| Workflow de GitHub Actions | `agrosat-terraform` + `agrosat-gcp-services` |
| Alquilar GPU L4 spot (`make train-l4`) o auditar costo | `agrosat-finops` |
| Operar la H100 de Azure cuando una US y el presupuesto lo autoricen | `agrosat-azure-h100` + `agrosat-finops` |
| Drift Evidently (dormido) | `agrosat-evidently-drift` |

## Seguridad y QA

| Acción | Skill |
|---|---|
| Auth, rate limit, CSP, RLS por `session_id` | `agrosat-security` |
| Audit OWASP / CIS GCP, pre-deploy | `agrosat-security-audit` |
| Escribir tests (dos suites separadas, fixtures reales) | `agrosat-testing` |
| Fase 4 — QA del diff de una US | `agrosat-code-review` + `agrosat-testing` (+ `agrosat-protocolo-articulo` si toca cifras) |

## Harness (transversal)

| Acción | Skill o comando |
|---|---|
| Saber qué existe y dónde; impacto río abajo | `rg` / imports / tests; Graphify es apoyo opcional |
| Reindexar el grafo tras integrar | `make graph-update` (un solo escritor) |
| Persistir una decisión compartida | ADR, `docs/us-resolved/` o plan; Engram queda local y opcional mientras ADR-015 sea PROPUESTA |
| Editar cualquier `AGENTS.md`; auditar el harness | `make guides-sync` + `make harness-check` |
| Crear rama, commitear, abrir PR, cerrar US | `agrosat-git-workflow` |
