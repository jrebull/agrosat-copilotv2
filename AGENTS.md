# AgroSatCopilot v2 — Guía Operativa del Orquestador

**Proyecto**: artículo para **MICAI 2027** sobre el punto de operación en mapeo de cultivos por parcela: las cuatro maneras de que un mapa prometa menos —recortar el catálogo, abstenerse, devolver un conjunto, retroceder a una clase gruesa— no son comparables por acierto; ordenarlas exige declarar una pérdida, situarlas en el espacio de costes resultante y medir quién paga cada una ([ADR-014](docs/decisions/ADR-014-micai-2027.md)). El sistema AgroSatCopilot (SaaS conversacional de análisis satelital, EPIC 1-12, cerrado) es el **sustrato**: sus datos, sus diez miembros y su arnés OOF alimentan el estudio; su producto no es el objeto del artículo.

**Stack**: Python 3.12 + Poetry | Polars | PyTorch + XGBoost + scikit-learn | PostgreSQL 15 + PostGIS + pgvector | FastAPI + Nuxt 4 (sistema, en mantenimiento) | Google ADK + Gemini 2.5 Pro (sistema) | DVC sobre GCS + MLflow + Dagster | LaTeX `llncs` | Terraform GCP `dev` (dormido).

**Hardware**: CPU (macOS Apple Silicon y Windows) para todo el protocolo del artículo. GPU de consumo (RTX 4070) o L4 spot en GCP (`make train-l4`) solo para reentrenamientos opcionales. **No hay H100**: la VM Azure del sponsor se perdió con tres checkpoints FarSLIP dentro ([`docs/paper/artefactos-perdidos.md`](docs/paper/artefactos-perdidos.md)); nada del artículo puede depender de ella.

> **Plan vigente**: el cuaderno público [`agrosat2027.netlify.app/plan`](https://agrosat2027.netlify.app/plan) — 15 épicas, 89 US, 376 SP; EPIC 13-17 son historia cerrada, EPIC 18-27 el trabajo nuevo. Su fuente única es `plan.html` del repo hermano `agrosat-micai-site` (clonado junto a este repo; `make plan-check` lo parsea y valida dependencias, ciclos, estados y camino crítico). Fases y criterios de cierre: [`docs/plan-micai-2027.md`](docs/plan-micai-2027.md). Alcance, modelo y destino de lo heredado: [ADR-014](docs/decisions/ADR-014-micai-2027.md).
> **El plan v8 del sistema** (`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`, EPIC 1-12, US-001 a US-082) está cumplido y cerrado. La numeración continúa desde EPIC 13 / US-084, pero ninguna decisión nueva sale de él ni de los planes v5/v6.

## Cómo usar esta guía

- **Raíz** ([`AGENTS.md`](AGENTS.md) y [`CLAUDE.md`](CLAUDE.md), espejos idénticos): normas transversales, aplican a todo el repo.
- **Guía de carpeta** (`<dir>/AGENTS.md` y `<dir>/CLAUDE.md`, también espejos): sobreescribe la raíz dentro de su scope. Se cargan **on-demand** al entrar al directorio.
- **Nunca leas el espejo contrario** del que tu harness ya cargó: es pagar el mismo contenido dos veces.

> **Regla de sincronización**: cada par `AGENTS.md` / `CLAUDE.md` es byte a byte idéntico. Claude Code lee los `CLAUDE.md`; Codex, Copilot y Cursor leen los `AGENTS.md`. Se edita **`AGENTS.md`** y se propaga con `make guides-sync`; `make guides-check` (dentro de `make check` y de CI) falla si un par difiere.

---

## Hechos verificados del estudio — NO re-derivar, NO contradecir

Auditados contra el ledger, el contrato del estimando y los ADR. Si un análisis nuevo contradice esta tabla, **el análisis está mal** hasta probar lo contrario. Las cifras concretas (F1 por miembro, deltas, intervalos) **no se citan de memoria ni desde esta guía**: salen de una fila sellada de [`paper/ARTIFACTS.md`](paper/ARTIFACTS.md) o no salen.

| Hecho | Valor |
|---|---|
| Banco primario | PASTIS-R, **fold 5 held-out**, **16 640 parcelas** en la intersección de los diez miembros, **18 clases** (`semantic18`), 496 patches. Ground truth sellado (< 500 kB, viaja en git) en `reports/paper_micai/fase1/` |
| Unidad y dependencia | Unidad de análisis: **parcela**. Clúster de dependencia: **`patch_id`**. El número de bloques K es sensibilidad espacial, **no** réplica |
| Régimen de evaluación | **Uno solo por comparación, nombrado en la frase**: fold-5 held-out por parcela · out-of-fold · píxel. Cruzarlos sin decirlo tumbó el manuscrito heredado |
| La cifra campeona | 0,7486 (Stacking-5) y 0,7470 (Stacking-3) son **in-sample para el meta-modelo**: `StackingEnsemble.fit` reajusta sobre todas las parcelas del fold 5. Nunca se imprimen como held-out |
| Ensamble frente a individual | Libre de fuga, **ninguna combinación mejora al mejor miembro individual**. El árbitro entrenado y el refinamiento por vecindad **salen del artículo** como contribución (ADR-014 §3) |
| Los tres defectos | Denominador móvil · punto de operación elegido dentro del bloque evaluado · remuestreo por parcela en vez de por bloque. **Reparados** en `ml/eval/paper_micai_coverage.py` con tests que fallan sobre la versión anterior; **artefactos sin regenerar** → filas `OBSOLETO` en el ledger y cuarentena obligatoria (`make paper-obsoletos-check`) |
| Custodia | `paper/ARTIFACTS.md`: elemento, ruta, MD5, bytes, commit y estado por fila. `make paper-artifacts-check` recalcula digests y está **probado en negativo**. Lo que no tiene fila no se imprime |
| Estimando | [`docs/paper/estimando-v1.json`](docs/paper/estimando-v1.json) es la **fuente normativa** (`make preregistro-check`): condicional al banco, población = todas las parcelas elegibles de prueba, universo de clases **desde entrenamiento**, punto de operación **desde train/val**, sin re-emparejar en prueba, sin agrupar bancos, **sin afirmación de transporte**, mínimo 3 clústeres pareados para publicar intervalo o p |
| Función de pérdida | **No existe todavía.** La fija US-172 con usuarios reales de mapas de cultivo. Es la causa raíz que dio la auditoría externa y por eso EPIC 27 va primero |
| Camino crítico (96 SP) | US-172 → 160 → 124 → 125 → 171 → 155 → 127 → 128 → 129 → 136 → 139 → 152 → 154 → 144 → 146 → 148. En curso: US-118, US-121, US-172 |
| Candado del preregistro | ADR-014 §7: **ningún cálculo de las EPIC 20, 21, 22 ni 25 antes del preregistro firmado**, y el preregistro no se firma antes de US-172. Fecha de corte del ADR: **18-sep-2026** (sin firma, valen los `[POR DEFECTO]`) |
| Predictor | **Panel predeclarado de ≥ 3 familias por banco, sin ganador** (ADR-014 §6, US-139). Un ganador solo por selección anidada independiente de la evaluación |
| Sede y formato | **MICAI 2027, y solo MICAI**. Hipótesis de 20 páginas (a reverificar con la convocatoria; si baja, el recorte va de resultados a apéndice, nunca de robustez). Doble ciego, `.zip` con el proyecto LaTeX |
| Autoría | Arthur J. Zizumbo Velasco (primero, correspondencia) · Javier A. Rebull-Saucedo (segundo). Isaac Ávila y Aaron Bocanegra: autores del código, acreditados en créditos y README, no del artículo |
| Lo heredado | Manuscrito de 24 páginas (`paper/main.tex`) → informe técnico interno. Borrador retirado de 15 páginas (`paper/micai/`) → archivo con `ESTADO.md` delante. **No se destruyen, no se publican, no se reparan** |

---

## Documentos normativos — la fuente de verdad del artículo

| Documento | Qué gobierna |
|---|---|
| [`docs/paper/preregistro-v2-borrador.md`](docs/paper/preregistro-v2-borrador.md) | El protocolo: hipótesis, contrastes confirmatorios frente a exploratorios, criterio de no envío (§8) |
| [`docs/paper/estimando-v1.json`](docs/paper/estimando-v1.json) | El contrato ejecutable del estimando; `make preregistro-check` falla si diverge de la prosa |
| [`docs/paper/perdidas-protocolo.md`](docs/paper/perdidas-protocolo.md) | El protocolo de elicitación de la pérdida (US-172); `make protocolo-check` impide congelarlo con campos vacíos |
| [`paper/ARTIFACTS.md`](paper/ARTIFACTS.md) | Custodia: cada cifra impresa se rederiva desde una fila sellada |
| [ADR-014](docs/decisions/ADR-014-micai-2027.md) y [ADR-013](docs/decisions/ADR-013-angulo-micai.md) | Alcance vigente y las **afirmaciones prohibidas** (siete en ADR-013 más las de ADR-014); su registro con gate es US-149 |
| [`docs/paper/campo-de-tiro.md`](docs/paper/campo-de-tiro.md) | Los diez ataques de un revisor estricto y la prueba que desactiva cada uno (EPIC 26) |
| [`docs/paper/respuesta-auditoria-externa.md`](docs/paper/respuesta-auditoria-externa.md) | Las ocho rondas de auditoría externa y lo que cada una cerró |

**Toda cifra, contraste o afirmación nueva debe ser trazable a uno de estos documentos.** Si no lo es, se registra en la bitácora de la US como desviación y se consulta antes de seguir.

---

## Decisiones irrevocables

| Decisión | Razón |
|---|---|
| **Cifras reales o nada** | Cada número impreso se rederiva desde un artefacto sellado. Las que solo viven en markdown o en el título de un PNG se retiran hasta que exista el artefacto |
| **Un régimen por comparación, nombrado** | Mezclar in-sample con held-out fue el defecto bloqueante del manuscrito heredado |
| **Preregistro antes de mirar** | Qué contraste es confirmatorio y cuál exploratorio, y la corrección por multiplicidad, se escriben antes de correr. El artículo del resultado nulo se escribe antes de conocerlo (US-156) |
| **Panel, no ganador** | Elegir el predictor con las mismas etiquetas que después evalúan sesga hacia arriba todo lo que sigue |
| **Inferencia condicional al banco, sin transporte** | Es el precio declarado del estimando. Dos bancos franceses que comparten sistema de etiquetado no demuestran transporte |
| **Gates probados en negativo** | Un control que nunca ha fallado no se sabe si funciona. Cada gate nuevo se rompe a propósito una vez antes de confiar en él |
| **Resultados negativos se reportan** con su matiz | A "future work" solo va lo que quedó fuera por diseño |
| **MICAI 2027, y solo MICAI** | Sin artículo de revista en paralelo; 20 páginas de hipótesis |
| **Stack del sistema intacto** | Polars, dbmate, Dagster, ADK, PostGIS: el sustrato no se migra. Migrar es trabajo neto negativo para el artículo |
| **AlphaEarth = `SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1, CC-BY-4.0** | Nunca "v2.1". El embedding anual carece de sensibilidad temporal: **no** codifica fenología |
| **Commits con autoría real** | Sin trailer `Co-Authored-By` de asistentes IA ni pies "generado con" en PRs |

### Descartados — no reactivar

**Azure H100 y todo lo que dependía de ella** (skill `agrosat-azure-h100`, `make azure-h100-*`, `make train-h100`, ventanas V1-V6) · **Gemma 4 LoRA** ([ADR-011](docs/decisions/ADR-011-gemma4-lora-future.md), sin hardware y fuera del alcance del artículo) · **vLLM on-prem como requisito** (Qwen se sirvió con llama.cpp) · los ids **`Qwen3.5-35B-A3B`** y **`gemma-4-26b-it`** (no existen) · **"Gemini 3.5 Flash"** (el reasoner es Gemini 2.5 Pro) · **árbitro entrenado y refinamiento por vecindad como contribución** · **mecanismos aprendidos de renuncia** (fuera de alcance, declarado) · **cualquier afirmación de transporte** · **reparar el manuscrito heredado** · **retirar clases "por poca muestra"** como premisa (el criterio real fue una curva por F1 con umbral 0,90) · Prithvi-EO-2.0 · MiniMax-M2.7 · Kimi K2.6 · Llama 3.3-70B QLoRA · LangGraph · Prefect · Alembic · DuckDB principal · PWA+Tauri.

Si una US parece necesitarlos, **detente y reporta**: casi siempre significa que se está reintroduciendo una tesis que ya cayó.

---

## Comandos

```bash
make check                   # lint + secrets-scan + i18n-check + guides-check (OBLIGATORIO antes de PR)
make lint                    # ruff + mypy (backend, ml, dagster_project, scripts) + pnpm lint
make test                    # pytest backend con cobertura (>= 70 %)
make test-ml                 # pytest tests/ml (excluye `slow`)
make test-frontend           # vitest con cobertura (>= 50 %)
make test-all                # los tres anteriores

make paper-artifacts-check   # el ledger: recalcula el MD5 de cada fila sellada
make paper-obsoletos-check   # ningun documento activo cita OBSOLETO sin cuarentena
make preregistro-check       # estimando-v1.json == preregistro seccion 4.5
make protocolo-check         # el protocolo de US-172 no se congela con campos vacios
make oof-manifest-check      # cada oof_parcel_* del arnes tiene entrada en manifest.json
make plan-check              # plan por epicas: dependencias, ciclos, estados, camino critico
make micai-pdf               # compila paper/micai (anonimo), cero errores, cero overfull
make micai-anon-check        # gate de doble ciego, con autoprueba en negativo
make micai-bib               # regenera refs.bib desde la matriz verificada (nunca a mano)

make graph-update            # reindexa el grafo de codigo (AST, 0 LLM)
make graph-check             # el grafo describe HEAD?  (built_at_commit vs git)
make graph-hooks             # hooks post-commit / post-checkout de graphify (por clon)
make memory-sync             # exporta las memorias nuevas del equipo a .engram/ (se commitea)
make memory-import           # repara el manifest y aplica los chunks de .engram/ tras un git pull
make memory-check            # verifica el manifest sin tocarlo (exit 1 si hay drift)
make memory-status           # chunks locales, remotos y pendientes
make memory-setup            # merge driver para .engram/manifest.json (por clon)
make harness-check           # espejos, settings, plantillas, skills: el harness se audita solo
make harness-status          # que hay en vuelo: US abiertas, grafo, memoria, espejos
make guides-sync             # AGENTS.md -> CLAUDE.md en todos los pares

poetry add <pkg>             # deps Python (nunca pip ad-hoc)
pnpm add <pkg>               # deps frontend (nunca npm/yarn)
dbmate up · dbmate new <slug>              # migraciones, solo rollforward
dvc pull data/breizhcrops · dvc add <ruta> · dvc push
poetry run pytest tests/ml/eval/test_paper_micai_coverage.py::test_name -q   # un solo test

graphify query "<pregunta>" --budget 1500   # que existe y donde — ANTES de grep
graphify affected "<nodo>" --depth 2        # que se rompe si toco esto
```

Lista completa de targets: [`docs/orchestration/commands.md`](docs/orchestration/commands.md).

---

## Grafo de conocimiento — consultar antes que leer

`graphify-out/graph.json` indexa el código del repo (AST, 1 056 archivos; ~18 000 nodos) y, cuando se corre la pasada semántica, los documentos vigentes del artículo. Devuelve nodos con `source_location` en vez de archivos completos: es el camino barato para responder "¿qué existe y dónde?".

**Orden obligatorio**: `graphify query` → verificar el hit con `sed -n 'A,Bp'` sobre el `source_location` que citó → **solo si devuelve 0 nodos**, `grep -r` y lectura completa.

| Comando | Para qué |
|---|---|
| `graphify query "<pregunta>" --budget 1500` | qué existe y dónde (BFS) |
| `graphify affected "<nodo>" --depth 2` | qué depende de esto — impacto río abajo |
| `graphify god-nodes --top 10` | hubs arquitectónicos que no hay que romper |
| `make graph-update` | reindexar el código cambiado (AST, 0 LLM, ~3 min en frío, segundos incremental) |
| `/graphify . --update` | reindexar docs vigentes (extracción semántica, con LLM; **solo en el cierre de una US**) |

**El grafo estampa su propio commit**: `graph.json` guarda `built_at_commit` y `make graph-check` lo compara con `git rev-parse HEAD`. Si no coinciden, el grafo describe otro estado del repo: `make graph-update` antes de confiar en una consulta. No se versiona `graphify-out/` (reconstruible).

**Lo que el grafo NO indexa, a propósito** ([`.graphifyignore`](.graphifyignore)): el manuscrito heredado, los planes v6/v8, los handoffs y manual-tests del ciclo antiguo, la presentación del curso, notebooks, `reports/` y datos. Son historia; indexarlos hace que una consulta devuelva la era H100 o la tesis caída como contexto vigente.

**El grafo responde por el código, nunca por la norma.** Las cifras salen del ledger; el protocolo, del preregistro y el estimando; el alcance de cada US, del cuaderno. Citar el grafo como respaldo científico es un error de la misma familia que re-derivar un hecho verificado.

**Un solo escritor.** Con sub-agentes en paralelo, todos consultan y ninguno reindexa: `make graph-update` lo corre el orquestador después de integrar, porque dos `update` concurrentes se pisan `graph.json`. Los hooks git (`make graph-hooks`, estado con `graphify hook status`) reindexan el código en cada commit y checkout.

Cuándo reindexar en cada fase: [`docs/orchestration/prompts-optimizers-fable.md`](docs/orchestration/prompts-optimizers-fable.md) §Grafo.

---

## Memoria compartida del equipo — engram viaja en cada PR

Engram es la memoria persistente **de desarrollo** (decisiones, bugs con causa raíz, gotchas, cierres de US). No es parte del runtime ni del artículo.

- **La base local** (`~/.engram/engram.db`) contiene todos los proyectos de cada máquina y **nunca se commitea**. Lo que viaja son los **chunks** exportados a [`.engram/`](.engram/) (`chunks/*.jsonl.gz` + `manifest.json`), que sí van en cada PR.
- **Proyecto canónico**: `agrosat-copilotv2`, fijado en [`.engram/config.json`](.engram/config.json) (primera prioridad de detección de engram: sobrevive a forks y renombres del remoto). El historial del proyecto del curso (388 observaciones de mayo a septiembre de 2026) ya está fusionado ahí y exportado en el primer chunk.
- **Protocolo por sesión**: al empezar, `mem_search` con las palabras clave de la US (o `mem_context`); tras cada decisión, bug corregido, convención o hallazgo no obvio, `mem_save` con el porqué; al cerrar, `mem_session_summary`.
- **Protocolo por PR**: antes de abrir el PR, `make memory-sync` y commitear `.engram/`. Tras un `git pull`, el plugin de Claude Code importa los chunks nuevos solo al arrancar la sesión; `make memory-import` es la vía manual (otros hosts o a mitad de sesión). `make harness-status` avisa si hay chunks pendientes.
- **Conflicto en `.engram/manifest.json`**: `make memory-setup` registra un merge driver que reconcilia las dos ramas contra su ancestro (una vez por clon): lo que un lado añadió se conserva y lo que ambos purgaron se queda purgado. Si el conflicto llega igual, `python scripts/engram_manifest_merge.py` lo repara — es lo que corre `make memory-import` antes de importar; nunca resolverlo tomando "un lado".
- **Qué guardar**: el porqué de una decisión (con su ADR si existe), la causa raíz de un bug, un gotcha entre máquinas, la lección de una US; `scope: project` y en español neutro, la lengua franca del equipo (FTS5 no cruza idiomas). **Qué no**: secretos, tokens, `session_id` reales, correos, datos de parcelas de usuario, notas personales (van bajo otro nombre de proyecto: `engram sync` exporta el proyecto entero) y **cifras del artículo** (viven en el ledger, no en la memoria).
- **Nunca `engram sync --all`**: exportaría todos los proyectos de la laptop al repo. `make harness-check` falla si un chunk trae otro proyecto o un token, y mira las cuatro listas del chunk (`observations`, `prompts`, `sessions`, `mutations`), no solo la primera.

Detalle e instalación: skill [`agrosat-engram-memory`](.claude/skills/agrosat-engram-memory/SKILL.md).

---

## Reglas de código NON-NEGOTIABLE

- **Idioma**: código (identificadores, comentarios, docstrings Google-style) en **inglés**; prosa visible al lector (notebooks markdown, prints, plots, docs `.md`) en **español neutro**; el manuscrito en **inglés americano**.
- **Sin emojis** en código, comentarios, prints, commits ni logs.
- **Logging**: `structlog.get_logger()`, nunca `print()` en producción.
- **Type hints** obligatorios en todo Python.
- **Polars, no pandas** en pipelines. `LazyFrame` por defecto; `parcel_id` canónico `pl.Utf8` vía `canonical_parcel_id`.
- **DRY**: función usada 2+ veces → `backend/app/utils/`, `ml/utils/` o `frontend/composables/`.
- **SoC**: router recibe → service procesa → model persiste. Tools ADK en `ml/agent/tools/`, nunca en routers; sin lógica de negocio en routers ni componentes Vue.
- **Multi-tenant por `session_id`**: toda query del sistema filtra por sesión/usuario (RLS forzada).
- **i18n**: todo texto visible en `frontend/i18n/locales/{it,es,en}.json` simultáneamente.
- **Secrets**: jamás hardcodear. `.env.local` en dev, Secret Manager en prod. GEE por ADC sobre el proyecto GCP `agrosat-copilot`.
- **DVC** para parquet, OOF, embeddings, checkpoints y datasets — nunca al repo Git. Los `.dvc` sí van al repo y su `md5` es parte de la custodia.
- **MLflow** con tags `data_version` (hash DVC) + `code_version` (git sha) en toda corrida; el fallo del servidor degrada a warning, nunca aborta el artefacto.
- **Migraciones**: solo `dbmate up` / `dbmate new`. Jamás `SQLModel.metadata.create_all()` ni modificar migraciones aplicadas.
- **Notebooks**: se commitean ejecutados end-to-end con outputs (entregable del curso, historia). Sin `nbstripout` en quality gates. El trabajo nuevo del artículo **no** vive en notebooks: vive en `ml/` + `scripts/run_paper_micai_*.py` + `reports/paper_micai/`.
- **Commits sin trailer `Co-Authored-By`** de asistentes IA — la autoría queda en el `Author:` real.
- **Solo los tests necesarios. Cero placeholders, cero cifras sintéticas.** Cada test cierra un hueco nombrado. Las fixtures de evaluación usan **filas reales** (el ground truth sellado del fold 5, un subconjunto de un parquet OOF), no parcelas inventadas: un dato fabricado prueba que el código corre, no que es correcto sobre el banco que existe. Los tests de mecánica pura (esquemas, parsers, CLI) sí pueden usar datos mínimos, marcados como tales.
- **Una reparación de protocolo lleva un test que falla sobre la versión anterior.** Así se hizo con los tres defectos (`tests/ml/eval/test_paper_micai_coverage.py`); es la regla, no la excepción.
- **Una US testea solo los archivos que ella crea o modifica.** Un defecto en un módulo ajeno se reporta como cambio pendiente, no se parchea desde la US en curso.
- **Dos suites separadas**: `backend/tests` y `tests/` tienen cada uno su `conftest.py` y no se cargan juntos. La cobertura se lee **archivo por archivo** sobre los archivos de la US (`--cov=<directorio acotado> --cov-report=term-missing`), jamás el TOTAL.

### Reglas científicas del artículo

- **Toda comparación nombra su régimen** y su unidad. Un F1 sin régimen no se reporta.
- **El universo de clases sale del entrenamiento**, nunca de las parcelas entregadas por el mecanismo (`macro_over` exige el universo del bloque). Es el denominador móvil.
- **El punto de operación se elige en train/val y se aplica sin tocarlo en prueba.** Igualar pérdida, cobertura o tasa usando la prueba es fuga (dos auditorías la encontraron en `confidence_baseline`).
- **El intervalo remuestrea la unidad declarada** (`paired_interval` exige declararla); por debajo de 3 clústeres pareados no se publica intervalo ni p. Partir el mismo territorio más fino no crea réplicas.
- **Multiplicidad declarada antes** (Holm sobre los K contrastes) y separación confirmatorio / exploratorio por escrito.
- **Semillas fijas y registradas**; si una conclusión depende de la semilla, se retira (US-161).
- **Artefacto = archivo + semilla + versiones + commit + prueba pareada con intervalo**, en `reports/paper_micai/<fase>/`, y fila en el ledger antes de citarse.
- **Nunca sobrescribir un artefacto sellado.** Se genera uno nuevo, se sella con `make paper-artifacts-seal` y el anterior cambia de estado en el ledger con su motivo. `OBSOLETO` es un estado, no un borrado.
- **Nada de las EPIC 20, 21, 22 ni 25 antes del preregistro firmado** (ADR-014 §7). Si el spec de una US lo pide, se detiene y se reporta.
- **Ningún predictor "ganador"**: panel de ≥ 3 familias por banco y el predictor como factor de sensibilidad.
- **Ninguna afirmación de transporte**, ni en el título, ni en el abstract, ni en la discusión.

---

## QA Gate antes de PR

1. `make check` limpio (lint + secrets + i18n + espejos).
2. Cobertura ≥ 70 % sobre los archivos nuevos/modificados de la US, leída por archivo.
3. Si tocó evaluación o artefactos: `make paper-artifacts-check`, `make paper-obsoletos-check` y `make oof-manifest-check` en verde; artefacto nuevo con fila en el ledger.
4. Si tocó el protocolo o el estimando: `make preregistro-check` y `make protocolo-check`.
5. Si tocó el manuscrito: `make micai-pdf`, `make micai-anon-check`, `make paper-cite-check`; cada cifra con su `% src:`.
6. Si entrenó: MLflow con `data_version` + `code_version`; checkpoint y OOF con `.dvc` y `dvc push`.
7. Si tocó el plan: `make plan-check` sobre el cuaderno.
8. `make memory-sync` y `.engram/` en el commit; `docs/us-work/us-XXX.md` al día.

---

## Git y PR

- Rama: `feature/E{epic}-US-XXX-{slug}` (`fix/...` para correcciones acotadas). Base y destino: **`main`** (no existe `develop`).
- Conventional Commits con scope de épica: `feat(E19): ...`, `fix(E18): ...`, `docs(E23): ...`, `chore(harness): ...`.
- `make check` limpio antes de abrir PR; plantilla de PR en [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
- Nunca `--no-verify`, nunca `--force` sobre `main`, nunca un `dvc push` que sobrescriba un artefacto sellado.

---

## Routing por directorio

| Directorio | Guía | Especialidad |
|---|---|---|
| `ml/` | [ml/AGENTS.md](ml/AGENTS.md) | Miembros, arnés OOF, evaluación del artículo (`ml/eval/paper_micai_*`, `set_valued.py`), features, FarSLIP |
| `ml/agent/` | [ml/agent/AGENTS.md](ml/agent/AGENTS.md) | Google ADK, 9 tools geoespaciales, Spatial-RAG (sistema) |
| `paper/` | [paper/AGENTS.md](paper/AGENTS.md) | `paper/micai/` (manuscrito MICAI, LNCS, doble ciego), `paper/ARTIFACTS.md` (ledger), manuscrito heredado (archivo) |
| `docs/paper/` | esta guía + [`docs/plan-micai-2027.md`](docs/plan-micai-2027.md) | Preregistro, estimando, protocolo de pérdidas, hallazgos por fase, auditorías |
| `reports/paper_micai/` | esta guía, §Reglas científicas | Artefactos por fase; solo salidas de `scripts/run_paper_micai_*.py`, nunca editados a mano |
| `scripts/` | esta guía | Runners del artículo, gates (`*_check.py`), sellado, builders de figuras |
| `backend/` | [backend/AGENTS.md](backend/AGENTS.md) | FastAPI, SQLModel, TiTiler, SSE (sistema, mantenimiento) |
| `frontend/` | [frontend/AGENTS.md](frontend/AGENTS.md) | Nuxt 4 SSR, MapLibre, chat, i18n (sistema, mantenimiento) |
| `db/` | [db/AGENTS.md](db/AGENTS.md) | dbmate, PostGIS, pgvector, RLS por sesión |
| `dagster_project/` | [dagster_project/AGENTS.md](dagster_project/AGENTS.md) | Assets, DVC ↔ MLflow lineage |
| `infrastructure/` | [infrastructure/AGENTS.md](infrastructure/AGENTS.md) | Terraform GCP `dev` (dormido); Azure retirada |
| `notebooks/` | [notebooks/AGENTS.md](notebooks/AGENTS.md) | Entregables del curso (historia, papermill) |

## Skills y orquestación

- Qué skill `agrosat-*` cargar antes de cada acción (31 skills): [`docs/orchestration/auto-invoke.md`](docs/orchestration/auto-invoke.md). Las dos que gobiernan el artículo: **`agrosat-protocolo-articulo`** (estimando, régimen, intervalos, ledger) y **`agrosat-paper-micai`** (LNCS, doble ciego, bib, gates).
- Catálogo completo: [`docs/orchestration/skills-catalog.md`](docs/orchestration/skills-catalog.md). Mapa skill ↔ subagente y subagentes por épica: [`docs/orchestration/skill-owners.md`](docs/orchestration/skill-owners.md).
- **Sistema de prompts por fases (F1-F7, ruta corta, modo nocturno)**: [`docs/orchestration/prompts-optimizers-fable.md`](docs/orchestration/prompts-optimizers-fable.md). Cada US tiene exactamente dos archivos en vuelo: el **spec** (`docs/us-planning/us-XXX.md`, se congela al aprobarse) y la **bitácora** (`docs/us-work/us-XXX.md`, se destila a `docs/us-resolved/` al cierre). `docs/us-handoff/` y `docs/manual-test/` son el formato antiguo: se leen, no se extienden.
- Plantillas de sub-agente por dominio (las lee el sub-agente, no el orquestador): [`docs/orchestration/subagent-prompts/`](docs/orchestration/subagent-prompts/).
- Subagentes profundos (10, en `.claude/agents/`): `ml-engineer` · `geo-data-engineer` · `paper-writer` · `mlops-engineer` · `qa-reviewer` · `backend-engineer` · `frontend-engineer` · `agent-engineer` · `security-reviewer` · `finops-auditor`.

---

## Anti-patrones — señales de que algo se está desviando

| Señal | Qué significa |
|---|---|
| Una cifra impresa sin fila en `paper/ARTIFACTS.md` o sin `% src:` | Es exactamente lo que retiró el manuscrito heredado. Se retira hasta que exista el artefacto |
| "held-out" junto a un número que salió de `_stacking_metrics` o de Optuna sobre el fold 5 | Es in-sample para el meta-modelo. Se etiqueta o no se imprime |
| Se cita `reports/paper_micai/fase3/*`, `fase4/replica_*`, `potencia/*`, `bloques/*` o `equidad/*` fuera de un bloque CUARENTENA | Artefactos `OBSOLETO` del módulo con los tres defectos. `make paper-obsoletos-check` lo caza |
| "A igual cobertura" sin decir cómo se define para el conjunto conforme y la clase gruesa | No está definido para dos de los cuatro mecanismos. Sin pérdida declarada no hay comparación |
| El universo de clases se calcula sobre las parcelas entregadas | Denominador móvil: el defecto 1 |
| Un umbral, una tasa o una igualación se eligen mirando la prueba | Fuga del punto de operación: el defecto 2 |
| Un bootstrap que remuestrea parcelas dentro del bloque | Dieciséis mil réplicas que no existen: el defecto 3 |
| Aparece un "mejor predictor" o "el modelo del artículo" | ADR-014 §6: panel de ≥ 3 familias, sin ganador |
| Un experimento de las EPIC 20, 21, 22 o 25 antes del preregistro firmado | ADR-014 §7. Se detiene y se reporta |
| "Se transporta a otras regiones", "generaliza", "en México" | Afirmación de transporte: retirada por diseño del estimando |
| `AlphaEarth v2.1`, "AlphaEarth codifica la fenología", `Gemini 3.5 Flash`, `Qwen3.5-35B-A3B`, `gemma-4-26b-it` | Atribuciones o ids inexistentes que ya se corrigieron una vez |
| Se propone "pedir la H100", "reactivar la VM" o `make azure-h100-*` | No existe. Todo corre en CPU, RTX 4070 o L4 spot |
| Una entrada escrita a mano en `paper/micai/refs.bib` | El bib se **genera** con `make micai-bib` desde la matriz verificada por API; a mano es como entraron las citas inventadas |
| Un nombre, correo, matrícula, "Team 17" o el nombre del sistema en el PDF anónimo | `make micai-anon-check` en rojo. Doble ciego desde el primer borrador |
| Se sobrescribe un archivo con fila `SELLADO` | Rompe la custodia. Archivo nuevo + sello nuevo + estado del viejo |
| `KFold`, `train_test_split` o un split aleatorio sobre parcelas | Fuga espacial. `build_spatial_kfold` (H3 + KMeans + colchón de 1 km) |
| `pandas` en código nuevo de pipeline | El proyecto usa Polars |
| Un test con parcelas inventadas para una métrica titular, o tests añadidos para subir cobertura | Prueba que el código corre, no que acierta sobre el banco real |
| Una US añadiendo tests a archivos que no creó ni modificó | Invade el gate de otra US. Un defecto ajeno se reporta |
| Barrido `grep -r` + lectura completa sin consultar el grafo | Se paga el camino caro por algo que `graphify query` devuelve en un comando |
| Se cita el grafo o la memoria engram para respaldar una cifra o una regla del protocolo | Indexan código y decisiones, no evidencia. La fuente es el ledger o el preregistro |
| Un PR sin `.engram/` actualizado o una sesión que arranca sin `mem_search` | La memoria del equipo se parte en dos máquinas |
| Editar `CLAUDE.md` sin propagar a `AGENTS.md` (o al revés) | `make guides-check` en rojo; los agentes de otros hosts leen otra versión |

---

## Estilo de respuesta

- Antes del primer tool call: una frase con el plan (≤ 20 palabras).
- Tareas con > 3 tool calls o > 30 s: `TodoWrite` al inicio.
- Código > prosa: el diff es la respuesta; respuestas triviales ≤ 4 líneas.
- Tool calls independientes en paralelo · grafo antes que Grep, Grep antes que Read · solo lo preguntado.
- Sin preámbulos ("Perfecto, voy a...", "Listo, he..."), sin narrar tool calls, sin emojis.
- Toda métrica con su régimen, su unidad y su intervalo. Un número sin protocolo no se reporta; un número sin fila sellada no se imprime.
