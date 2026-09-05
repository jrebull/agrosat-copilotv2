---
name: agrosat-code-review
description: Revision del diff de una US en la Fase 4 del loop y antes de cada PR a main — DRY, SoC y reglas de codigo del orquestador, mas los anti-patrones del articulo MICAI 2027 (regimen nombrado, unidad y cluster, denominador, punto de operacion, ledger, OBSOLETO, ADR-014 §7, doble ciego) y los del sistema (RLS, i18n, mocks). Use sobre los archivos del diff, nunca sobre el repo entero.
allowed-tools: Read, Bash, Glob, Grep
---

# Code review — Fase 4 del loop

Trabaja **solo sobre los archivos del diff** de la US, nunca sobre el repo entero.

```bash
git diff --name-only HEAD~N                       # N = commits de la US segun la bitacora
graphify affected "<modulo principal>" --depth 2  # consumidores rio abajo que NO estan en el diff
```

## Checklist general

- **DRY**: logica duplicada con algo que ya existe en `ml/utils/`, `backend/app/utils/` o `frontend/composables/`? Funcion usada 2+ veces vive ahi.
- **SoC**: logica de negocio en un router FastAPI, en un componente Vue o en un script de runner? Es error.
- **Type hints** completos; docstrings Google-style en ingles; prosa visible en espanol neutro.
- **`structlog`, nunca `print()`** en codigo de app o de `ml/`.
- **Sin emojis** en codigo, comentarios, logs ni commits. **Sin secretos** hardcodeados.
- Codigo muerto o comentado se borra. Un `TODO` sin US asociada no entra.
- **Polars, no pandas** en pipelines; `parcel_id` canonico `pl.Utf8`.
- El spec (`docs/us-planning/us-XXX.md`) es el contrato: cada criterio de §1 contra el codigo real; cada desviacion de la bitacora §1.2 evaluada como aceptable o bug.

## Checklist del articulo — lo que de verdad rompe este proyecto

| Revision | Por que |
|---|---|
| Alguna cifra nueva sin fila `SELLADO` en `paper/ARTIFACTS.md` o sin `% src:`? | Es lo que retiro el manuscrito heredado |
| Alguna comparacion sin regimen nombrado (fold-5 held-out por parcela / in-sample meta-modelo / OOF / pixel)? | Un numero sin regimen no es evidencia |
| `_stacking_metrics`, Optuna o cualquier reajuste sobre el fold 5 presentado como held-out? | In-sample para el meta-modelo |
| Universo de clases del macro calculado sobre parcelas entregadas? | Denominador movil (defecto 1) |
| Umbral, tasa o igualacion "a igual cobertura" elegidos mirando la prueba? | Fuga del punto de operacion (defecto 2) |
| Bootstrap sobre parcelas dentro del bloque; `paired_interval` sin `unit`; intervalo con < 3 clusteres? | Replicas que no existen (defecto 3) |
| Contraste sin Holm ni etiqueta confirmatorio / exploratorio declarada en el spec? | Multiplicidad sin definir |
| Artefacto `OBSOLETO` citado fuera de un bloque CUARENTENA? | `make paper-obsoletos-check` en rojo |
| Archivo con fila `SELLADO` sobrescrito, o parquet nuevo sin `.dvc`? | Custodia rota |
| Corrida de las EPIC 20, 21, 22 o 25 sin preregistro firmado? | ADR-014 §7 |
| "Mejor predictor", "el modelo del articulo", un ganador elegido con las etiquetas de evaluacion? | Panel de >= 3 familias, sin ganador (ADR-014 §6) |
| "Se transporta", "generaliza", "en Mexico"? | Afirmacion de transporte, retirada por diseno |
| `KFold` / `train_test_split` sin bloques espaciales? | Fuga espacial |
| Semilla no fijada o no registrada en el artefacto? | Conclusion que depende de la semilla |
| MLflow sin `data_version` + `code_version`, o entrenamiento sin `train_folds` / `val_folds`? | Provenance perdida (US-119) |
| Dependencia de H100, `azure_h100_*`, Gemma 4 LoRA, `Qwen3.5-35B-A3B`, "Gemini 3.5 Flash", "AlphaEarth v2.1"? | Descartados o inexistentes |
| Entrada a mano en `paper/micai/refs.bib`; nombre, correo o "Team 17" en el PDF anonimo? | Bib generado; `make micai-anon-check` |
| Gate nuevo sin prueba en negativo? | Un control que nunca fallo no se sabe si funciona |

## Checklist del sistema (solo si el diff toca `backend/`, `frontend/`, `ml/agent/`, `db/`)

- `_check_session_owner` y RLS por `session_id` en todo endpoint que toque datos de sesion; respuestas Pydantic, nunca `SQLModel` crudo.
- Config solo via `get_settings()`; migracion `dbmate new` si toco schema; nada de `create_all()`.
- i18n: la clave en `it.json`, `es.json` y `en.json`; `<script setup lang="ts">`; SSR-safe.
- Tools ADK como `FunctionTool` + Pydantic con citas; ningun test llama a Vertex, GEE, vLLM o llama.cpp reales.

## Checklist de tests

- Solo archivos de esta US; fixtures con filas reales para metricas titulares; sin tests para inflar cobertura.
- Cobertura leida archivo por archivo (>= 70 %), jamas el TOTAL; dos suites separadas (`backend/tests` y `tests/`).
- Una reparacion de protocolo trae un test que falla sobre la version anterior.

## Arbol de decision

```
PR toca ml/eval, ml/ensemble, scripts/run_paper_micai_* o reports/  -> general + articulo + tests
PR toca paper/micai o docs/paper                                   -> general + articulo (cifras, bib, anonimato)
PR toca ml/ingest, ml/features, dagster_project                     -> general + articulo (split, .dvc, sellado de banco) + tests
PR toca backend, frontend, ml/agent, db                             -> general + sistema + tests (+ security-reviewer)
PR toca Makefile, scripts/*_check.py, ci.yml                        -> general + gate probado en negativo
PR toca AGENTS.md, skills, agents, docs/orchestration               -> make harness-check en verde
```

## Salida

Tabla de hallazgos ordenada por severidad, con archivo, linea y por que importa; tabla de criterios del spec contra estado; cobertura por archivo. Sin recomendaciones genericas de estilo que el linter ya cubre.
