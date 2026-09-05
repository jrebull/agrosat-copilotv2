# Prompts Optimizados — Loop de ingeniería por fases · AgroSatCopilot v2 (MICAI 2027)

**Cómo usar**: escribe `Fase N — US-XXX: [titulo]` y pega el prompt de esa fase, copiado
verbatim (un prompt estable se cachea; uno reescrito a mano, no). Cada US tiene exactamente dos archivos en vuelo: el **spec** (`docs/us-planning/us-XXX.md`, se congela al aprobarse y se conserva para siempre) y la **bitácora** (`docs/us-work/us-XXX.md`, estado de ejecución que se destila a `docs/us-resolved/us-XXX.md` y se borra al cierre). Ninguna fase depende de la ventana de otra: si esos dos archivos no bastan para retomar, el archivo está incompleto — se corrige el archivo, no se alarga la sesión.

**Convención de nombres**: cuando este documento dice "AGENTS.md" se refiere a la guía que tu harness **ya cargó solo**: `CLAUDE.md` en Claude Code, `AGENTS.md` en Codex/Copilot/Cursor — son espejos byte a byte. Jamás leas el espejo contrario: es pagar el mismo contenido dos veces. Lo mismo aplica a las guías de directorio (`ml/AGENTS.md` ≡ `ml/CLAUDE.md`).

**Plan canónico**: el cuaderno público [`agrosat2027.netlify.app/plan`](https://agrosat2027.netlify.app/plan), cuya fuente es `plan.html` del repo hermano `agrosat-micai-site` (`make plan-check`); fases en [`docs/plan-micai-2027.md`](../plan-micai-2027.md); alcance en [ADR-014](../decisions/ADR-014-micai-2027.md).
**Documentos normativos**: [`docs/paper/preregistro-v2-borrador.md`](../paper/preregistro-v2-borrador.md) · [`docs/paper/estimando-v1.json`](../paper/estimando-v1.json) · [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md).
**Regla de oro**: cifras reales o nada — cada número impreso se rederiva desde un artefacto sellado, con su régimen nombrado (AGENTS.md raíz, §Decisiones irrevocables).

---

## Economía de contexto — las 4 reglas de este flujo

| # | Regla | Qué elimina |
|---|---|---|
| R1 | **Las instrucciones estáticas viven en AGENTS.md, no en los prompts.** El protocolo graphify, los hechos verificados, los anti-patrones, las reglas científicas y las de código ya se cargan solos en cada sesión y en cada sub-agente. Las fases los referencian en una línea | Los instructivos repetidos en cada fase y los anti-patrones re-listados en QA |
| R2 | **Dos archivos por US, con roles distintos.** El spec (`us-planning`) es el contrato: se escribe en F2, se congela al aprobarse, dirige F3-F4 y **se conserva** — es la programación por specs (SDD). La bitácora (`us-work`) absorbe al handoff y al manual-test del ciclo antiguo: estado de ejecución, QA y bugs, desechable tras destilarse a `us-resolved` | `us-handoff`, `manual-test` y notas sueltas, y las relecturas cruzadas entre 5 documentos |
| R3 | **El orquestador no carga los prompts de sub-agentes.** Lanza cada dominio con 3 líneas; el sub-agente lee su plantilla de `docs/orchestration/subagent-prompts/` en su propio contexto limpio — y del par de archivos, solo el spec | Los bloques de dominio pegados íntegros en Fase 3 y duplicados otra vez en el modo nocturno |
| R4 | **Frontera de sesión = `/clear`.** El disco es la memoria entre fases; la sesión es desechable. Los sub-agentes nacen limpios, devuelven un resumen de ≤ 30 líneas y **no escriben** ningún archivo de estado — la bitácora la escribe el orquestador, una vez, integrado | Sub-agentes heredando 100k+ tokens de fases previas, y escrituras concurrentes sobre el archivo de estado |

Las reglas duras de dominio (régimen nombrado, unidad parcela / clúster `patch_id`, universo desde entrenamiento, ledger, doble ciego, DVC) **sí se repiten** en las plantillas de sub-agente. Es redundancia deliberada: son el seguro contra el anti-patrón más caro de cada dominio, y su costo es de líneas, no de miles de tokens.

**Por qué el spec va aparte y se conserva**: los flujos spec-driven de referencia (GitHub
Spec Kit, AWS Kiro) separan el spec de los artefactos de ejecución precisamente porque el
contrato debe ser inmutable mientras se implementa y sobrevivir al cierre. Aquí además la
trazabilidad científica (contraste → sección del preregistro → fila del ledger) vive en el spec: es evidencia de auditoría, no un borrador. Y la lección local: el ciclo antiguo cerró decenas de US con handoffs que nadie destiló — con un único archivo desechable, un cierre flojo pierde todo; con el spec permanente, hasta el peor cierre conserva el contrato.

### Fuentes (verificadas 2026-08-19)

1. Anthropic — [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
   (sep-2025, sigue siendo la referencia vigente): buscar "el conjunto más pequeño de tokens de alta
   señal"; sub-agentes con contexto limpio que devuelven un resumen destilado (1,000-2,000 tokens);
   notas persistentes fuera de la ventana como memoria entre sesiones. → R3, R4, y la bitácora de R2.
2. Sourcegraph — [*Context Engineering: A Practical Guide for AI Agents*](https://sourcegraph.com/blog/context-engineering)
   (may-2026): el scratchpad se escribe a archivo fuera de la ventana; las instrucciones de
   comportamiento se declaran una vez en la capa de instrucciones, no en cada turno; presupuesto de
   tokens con tope y reinicio de sesión cuando el contexto se degrada. → R1, R2, R4.
3. [*Harness Engineering for Agentic AI Coding Tools: An Exploratory Study*](https://arxiv.org/abs/2602.14690)
   (arXiv 2602.14690, feb-2026, v5 jun-2026): sobre 2,853 repos de GitHub, los context files tipo
   `AGENTS.md` son el estándar interoperable de configuración y el punto natural para las
   instrucciones estáticas; skills y sub-agentes son los mecanismos avanzados menos adoptados —
   este repo ya los tiene, lo que faltaba era dejar de duplicar en prompts lo que el harness ya carga. → R1.
4. [*TokenPilot: Cache-Efficient Context Management for LLM Agents*](https://arxiv.org/abs/2606.17016)
   (arXiv 2606.17016, jun-2026): el costo lo gobierna la continuidad del caché KV — un prefijo de
   prompt estable se reutiliza, mutarlo lo invalida (reducciones medidas de 56-87 %). → prompts de
   fase copiados verbatim y sesiones cortas de R4.

---

## Fronteras de sesión

| Frontera | Acción | Por qué |
|---|---|---|
| F1 → F2 | misma sesión | el research alimenta directo al spec |
| **F2 → F3** | **`/clear` obligatorio** (o sesión nueva) | el orquestador solo necesita el spec aprobado; sin `/clear`, cada sub-agente corre sobre el costo de una ventana de 100k+ tokens de deliberación de planeación |
| F3 → F4 | `/clear` | QA audita el diff contra el spec y la bitácora, no la conversación de programación |
| F4 → F5 | sesión aparte siempre | diálogo interactivo largo; acumula contexto rápido |
| F5 → F6-F7 | `/clear`; F6 y F7 pueden compartir sesión | cierre corto sobre estado ya consolidado |

**La memoria cruza la frontera sola.** Cada sesión nueva (o `/clear`) arranca con el hook del plugin
de engram: importa los chunks de `.engram/` que aún no estén en la DB local e inyecta el contexto
reciente del proyecto. Por eso cada fase abre con un `mem_search` acotado a la US y no con una
relectura de la sesión anterior. Lo que una fase decide se persiste con `mem_save` antes de cerrarla
(F3, F6) y sale del repo con `make memory-sync` en F7; entre medias, la sesión es desechable.

---

## Los dos archivos de la US

### El spec: `docs/us-planning/us-XXX.md` — permanente, congelado tras aprobación

Lo abre la Fase 1 (o la 2 si no hubo research) y lo completa la Fase 2. **Al aprobarse se
congela**: ninguna fase posterior lo edita. Una desviación durante la implementación se
registra en la bitácora (§1.2) citando la sección del spec que contradice; si la desviación
invalida un criterio de aceptación, se detiene el trabajo y se consulta al humano. Un cambio
de alcance real exige una revisión explícita marcada `Rev N — fecha — motivo` al final del
spec, nunca edición silenciosa.

> Las US del cuaderno traen criterios de aceptación, SP, dependencias y artefacto de salida.
> El spec los **copia** en §1 (verbatim) y los vuelve verificables en §6: el cuaderno es el
> "qué"; el spec es el "cómo se comprueba".

```markdown
# Spec US-XXX — [Titulo]

**Estado**: draft | aprobado (congelado)
**Epica**: EPIC NN · **Rama**: feature/ENN-US-XXX-{slug} · **Depende de**: [US previas y artefactos] · **SP**: N

## 1. Criterios de aceptacion (verbatim del cuaderno + metrica verificable de cada uno)
## 2. Arquitectura y archivos — crear vs extender, segun el grafo
## 3. Interfaces publicas (signatures de modulos nuevos; columnas de parquet; claves de JSON)
## 4. Dominios tocados — que sub-agentes lanza F3
- [ ] geo-data  [ ] modeling  [ ] paper  [ ] app  [ ] mlops
## 5. Trazabilidad cientifica
| Contraste / cifra / afirmacion nueva | Seccion del preregistro o del estimando que la autoriza | Fila del ledger que la sostendra |
|---|---|---|
## 6. Pruebas de falsabilidad y gates
| Prueba | Umbral | Que pasa si falla | Gate probado en negativo? |
|---|---|---|---|
## 7. Plan de tests (cobertura >= 70 % por archivo del diff) · riesgos · presupuesto (GPU h · GEE exports · LLM USD)
## 8. Candados: regimen nombrado · unidad y cluster · universo desde entrenamiento · punto de operacion desde train/val · ADR-014 §7 (nada de EPIC 20/21/22/25 antes del preregistro firmado)

## A. Research (F1, destilado a <= 40 lineas: que existe ya, libreria/version, us-resolved benchmark)
## Revisiones (solo si el alcance cambio despues de aprobado)
```

### La bitácora: `docs/us-work/us-XXX.md` — desechable, se destila a `us-resolved` en F7

La escribe el orquestador (F3) y las fases de QA/correcciones. Los sub-agentes **nunca** la tocan: devuelven su resumen y el orquestador integra.

```markdown
# US-XXX — bitacora de ejecucion

**Estado**: coding | qa | testing | ready-to-close | ready-for-human | done · **Ultima fase**: [N — fecha]

## 1. Bitacora  <!-- escribe el orquestador en F3 -->
### 1.1 Archivos tocados / existentes reutilizados (no duplicar)
### 1.2 Decisiones tecnicas, desviaciones del spec (citando su seccion) y zonas sensibles
### 1.3 Falsabilidad y gates ejecutados
| Prueba | Umbral | Resultado | Veredicto |
|---|---|---|---|
### 1.4 Artefactos: ruta en reports/paper_micai/<fase>/ · semilla · versiones · commit · regimen y unidad · metrica con intervalo · fila del ledger (SELLADO / pendiente de sellar) · MLflow run id · DVC
### 1.5 Presupuesto consumido: GPU [h/tope] · GEE [exports/tope] · LLM [USD/tope]
### 1.6 Grafo y memoria: reindexado [fase+comando] · consultas que orientaron decisiones · affected rio abajo · mem_save hechos

## 2. QA  <!-- escribe F4 -->
### 2.1 Hallazgos (criterio vs estado, issues)
### 2.2 Pruebas manuales — [Paso a paso] -> [Resultado esperado], solo lo que exige ojo humano

## 3. Bugs  <!-- escriben F5-F6 -->
| Bug | Causa | Solucion | Estado |
|---|---|---|---|
```

---

## Grafo de conocimiento

Protocolo completo en **AGENTS.md raíz** (§Grafo de conocimiento): query → verificar hit con `sed` → grep solo con 0 nodos; un solo escritor; el grafo responde por el código, nunca por la norma. Lo único que vive aquí es el calendario de reindexado:

| Momento | Comando | Quién |
|---|---|---|
| Fin de F3 (integrado) y tras TESTS si agregó archivos | `make graph-update` | orquestador, una sola vez |
| Fin de F4 y F6 | `make graph-update` | la sesión de QA |
| F5, tras cada corrección que cambie lógica o firmas | `make graph-update` | la sesión de correcciones |
| F7 (cierre) | `/graphify . --update` — entran `us-resolved` y el spec congelado, exige semántica | tú; único punto que paga LLM |
| Modo nocturno | solo `make graph-update` (0 LLM) | nunca semántico sin supervisión |

Los hooks git (`make graph-hooks`) reindexan el código solos en cada commit y checkout; el calendario cubre lo que pasa entre commits.

---

## Fase 1 — Research
> Solo si la US involucra tecnología, método estadístico o patrón nuevo no establecido en el codebase (un mecanismo nuevo, una librería de conformal, un banco nuevo). Si no, directo a Fase 2.

```text
Investiga para la US-XXX: [titulo].

Protocolo graphify del AGENTS.md raiz: grafo -> verificar -> grep solo con 0 nodos.
1. graphify query "[titulo + dominio]" --budget 1500 y god-nodes --top 10 — que existe ya.
2. Cuaderno: la US-XXX y su epica (make plan-check para dependencias). Preregistro y
   estimando-v1.json solo si la US toca protocolo, contrastes o cifras.
3. graphify query "us-resolved [dominio]" --budget 1200 — los 2 mas recientes como benchmark.
4. Documentacion con Context7 (--c7); web search solo si falta.
5. Carga las skills del dominio segun docs/orchestration/auto-invoke.md.
6. mem_search "[titulo + tecnologia]" — la memoria del equipo ya tiene 388 observaciones.

Crea docs/us-planning/us-XXX.md con el template de prompts-optimizers-fable.md:
solo encabezado (estado "draft") y §A Research, destilado a <=40 lineas.
Solo investiga; no planees ni programes.
```

---

## Fase 2 — Planeación (escribe el spec)

```text
Planifica la US-XXX: [titulo].

Criterios de aceptacion (del cuaderno):
"""
[Pegar los criterios de la US-XXX tal cual aparecen en agrosat2027.netlify.app/plan]
"""

Lee: docs/us-planning/us-XXX.md §A si existe; la epica de la US en el cuaderno; el
preregistro (docs/paper/preregistro-v2-borrador.md) y docs/paper/estimando-v1.json solo si
la US toca protocolo, contrastes o cifras; ADR-014 §7 si la US pertenece a las EPIC 20, 21,
22 o 25; y los us-resolved que el grafo senale como benchmark. AGENTS.md ya esta cargado — no
lo releas, aplicalo.

Que existe ya — protocolo graphify del AGENTS.md raiz:
  graphify query "[dominio]" --budget 2000 · god-nodes --top 10 · affected "[modulo a tocar]" --depth 2
Context7 (--c7) para la API vigente de lo que vayas a usar (Polars, scikit-learn, XGBoost,
MAPIE/conformal, SciPy/statsmodels, matplotlib; FastAPI, Nuxt 4; DVC, MLflow). No asumas que
la conoces.
mem_search "[titulo]".

Escribe el spec docs/us-planning/us-XXX.md completo (§1 a §8), estado "draft" —
crealo tu si no hubo Fase 1 (el caso normal) y omite §A.
En §2 marca que archivos YA existen segun el grafo: se extienden, no se duplican.
En §5 cada contraste o cifra nueva cita la seccion del preregistro que la autoriza; si
ninguna la autoriza, la US necesita una revision del preregistro ANTES de programar.
Este spec es el contrato de la US — tras aprobarse se congela y nadie lo edita.
No programes nada.

Al terminar dime: "Spec listo. Al aprobarlo cambia su estado a 'aprobado (congelado)'
y haz /clear antes de pegar Fase 3."
```

---

## Fase 3 — Programación (orquestador)
> **Sesión limpia obligatoria** (`/clear` tras aprobar el spec). Hasta 4 sub-agentes en paralelo,
> aislados por directorio. Los prompts de dominio viven en `docs/orchestration/subagent-prompts/`
> — el orquestador no los carga; cada sub-agente lee el suyo y trabaja contra el spec congelado.

```text
Programa la US-XXX: [titulo].

0. Chequeo de sesion: si esta conversacion ya contiene la deliberacion de Fase 2 u otra
   fase, detente y pideme /clear antes de continuar — R4 no es opcional.

Lee el spec docs/us-planning/us-XXX.md (verifica que su estado sea "aprobado"; si no,
detente y reporta). AGENTS.md ya esta cargado. El spec NO se edita: una desviacion se
registra en la bitacora citando la seccion que contradice; si invalida un criterio de
aceptacion, detente y consultame.

Lanza en paralelo (background) SOLO los dominios marcados en el spec §4, cada uno con
este prompt de 3 lineas — nada mas:

  "Implementa la capa [DOMINIO] de la US-XXX: [titulo].
   Lee docs/orchestration/subagent-prompts/[dominio].md y siguelo al pie.
   El spec (congelado): docs/us-planning/us-XXX.md — respeta §2, §3 y §8."

  [dominio] -> plantilla y subagente:
  geo-data -> geo-data.md (geo-data-engineer) · modeling -> modeling.md (ml-engineer)
  paper -> paper.md (paper-writer) · app -> app.md (backend/frontend/agent-engineer)
  mlops -> mlops.md (mlops-engineer, EN SERIE al final, cuando los demas integraron)

Los sub-agentes devuelven un resumen de <=30 lineas — mas un ANEXO TECNICO opcional sin
limite para advertencias que QA necesita (deprecations, workarounds, fallos intermitentes)
— y NO tocan el spec ni la bitacora.
Si falta la plantilla de un dominio marcado: no improvises en silencio — redacta tu el
prompt desde el spec + AGENTS.md y reportalo como incidencia.

Cuando todos terminen:
1. Integra y resuelve conflictos de frontera: columnas y tipos del parquet
   (geo-data<->modeling), esquema del JSON de artefactos y nombre del regimen
   (modeling<->paper), fila del ledger y .dvc (modeling<->mlops), Pydantic vs TypeScript
   (backend<->frontend).
2. make graph-update — TU, una sola vez (un solo escritor).
3. Lanza TESTS (foreground) con el mismo formato de 3 lineas -> subagent-prompts/tests.md.
4. make graph-update otra vez si TESTS agrego archivos.
5. Crea la bitacora docs/us-work/us-XXX.md con §1 completo (1.1 a 1.6, resumenes
   integrados, anexos tecnicos volcados en 1.2, desviaciones del spec citadas), estado "qa".
6. mem_save con las decisiones tecnicas clave (el porque, no el que).
7. Reporta git status --short.
```

---

## Fase 4 — QA y Testing
> Sesión nueva o `/clear`.

```text
QA de la US-XXX: [titulo].

0. Chequeo de sesion: si esta conversacion contiene la Fase 3, detente y pideme /clear.

Lee el spec docs/us-planning/us-XXX.md y la bitacora docs/us-work/us-XXX.md §1.
Diff de la US:
git diff --name-only HEAD~N   (N = commits de esta US segun la bitacora)
Trabaja SOLO sobre esos archivos. El spec es el contrato: audita contra el, no contra
lo que la bitacora diga que se intento.

0. graphify affected "[modulo principal]" --depth 2 — lo que aparezca rio abajo y NO este
   en el diff es candidato a regresion silenciosa: verifica esos consumidores explicitamente.
1. make check && la suite que toque (make test-ml / make test) — cobertura >=70 % sobre los
   archivos del diff, leida archivo por archivo como manda AGENTS.md, jamas el TOTAL.
2. /agrosat-code-review sobre el diff + verificar una a una la tabla "Anti-patrones" del
   AGENTS.md raiz.
3. Si toca protocolo, contrastes o artefactos: /agrosat-protocolo-articulo — regimen
   nombrado, unidad y cluster, universo desde entrenamiento, punto de operacion desde
   train/val, intervalo sobre la unidad declarada, multiplicidad; cada artefacto con
   semilla + versiones + commit; make paper-artifacts-check, paper-obsoletos-check y
   oof-manifest-check en verde; ADR-014 §7 respetado.
4. Si toca el manuscrito: /agrosat-paper-micai — make micai-pdf, micai-anon-check,
   paper-cite-check; cada cifra con % src: y fila SELLADA; bib generado, no editado.
5. Si toca datos o features: /agrosat-ml-features o /agrosat-gee-alphaearth — split
   espacial, parcel_id canonico, .dvc presentes, ningun export GEE en tests.
6. Si toca el sistema (backend/frontend/agente): /agrosat-security — RLS por session_id,
   i18n en los tres locales, mocks de Vertex/GEE/vLLM.
7. Cada criterio del spec §1 contra el codigo real; cada desviacion de bitacora §1.2
   evaluada: aceptable o bug. Cada fila del spec §5 con su artefacto o su "pendiente".
8. mem_search "[keyword]" por bugs similares previos.

Escribe bitacora §2 (hallazgos + pruebas manuales), estado "testing".
make graph-update
Reporta: tabla criterios del spec vs estado + archivos auditados + issues + cobertura por archivo.
```

---

## Fase 5 — Pruebas manuales y correcciones
> Interactiva — siempre en sesión aparte.

```text
Correcciones de la US-XXX: [titulo].

Lee la bitacora docs/us-work/us-XXX.md completa. El spec docs/us-planning/us-XXX.md
esta congelado: consultalo cuando un bug toque un criterio o una interfaz. Luego:
git diff --name-only HEAD~N && git diff HEAD~N -- [archivos de bitacora §1.1]

- Te reporto bugs uno por uno.
- Antes de tocar un archivo: graphify affected "[archivo o funcion]" --depth 2.
- Tras cada correccion que cambie logica o firmas (no textos): make graph-update
  (AST, 0 LLM, segundos) — el affected del siguiente bug y el QA de F6 no deben
  consultar un grafo rancio.
- Si la correccion contradice una decision de bitacora §1.2 o una seccion del spec,
  explicame antes de cambiar.
- mem_search "[descripcion del bug]" antes de corregir.
- Si el bug exige RE-GENERAR un artefacto sellado, alquilar GPU, exportar de GEE o llamar
  a un LLM en lote: NO sin mi confirmacion.
- Si el bug es "la metrica bajo": verifica primero que no fuera fuga (regimen, punto de
  operacion, denominador) lo que la subia.
- Registra cada bug en bitacora §3 y corre linters + tests de los archivos tocados
  tras cada correccion.

Confirma que leiste la bitacora.
```

---

## Fase 6 — QA final post-correcciones
> Solo si en Fase 5 hubo cambios de lógica; si solo hubo textos o estilos, omitir.

```text
QA final de la US-XXX: [titulo].

Lee la bitacora docs/us-work/us-XXX.md §3. Diff post-correcciones:
git diff --name-only HEAD~N   (N = commits post-Fase 4, segun bitacora §3)
Trabaja SOLO sobre esos archivos.

1. make check && la suite que toque — sin regresiones.
2. /agrosat-code-review: las correcciones no reintrodujeron ningun anti-patron de la tabla
   del AGENTS.md raiz; sin codigo muerto ni DRY roto.
3. Si toco protocolo o artefactos: los candados del spec §8 siguen intactos y los gates
   del paper en verde. Si toco el manuscrito: micai-pdf + micai-anon-check en verde.
   Si toco datos: .dvc al dia y dvc status limpio.

make graph-update
Marca los bugs de bitacora §3 como verificados, estado "ready-to-close".
mem_save con la observacion final + patrones que funcionaron.
Reporta: tabla bugs corregidos vs verificados.
```

---

## Fase 7 — Cierre

```text
Cierra la US-XXX: [titulo].

Lee el spec docs/us-planning/us-XXX.md y la bitacora docs/us-work/us-XXX.md completa —
es la ultima vez que la bitacora existe.

1. git status --short
2. Destila a docs/us-resolved/us-XXX.md: resumen ejecutivo; criterios del spec §1
   (que se hizo + evidencia); desviaciones del spec y su resolucion; trazabilidad
   cientifica (spec §5 con la fila del ledger de cada cifra); falsabilidad y gates con
   resultados; cumplimiento (linters, cobertura por archivo); artefactos con regimen,
   semilla, versiones, commit y estado en el ledger; MLflow run id + DVC si entreno;
   presupuesto consumido si gasto GPU, GEE o LLM.
3. Conventional Commit: feat(ENN): [descripcion corta]. Sin trailer de IA.
4. make memory-sync — los chunks nuevos de .engram/ van en el mismo commit.
5. Si la US cambia de estado en el cuaderno: editalo en el repo hermano agrosat-micai-site
   (plan.html) y corre make plan-check; reportame el cambio para que lo publique.
6. Reindexado semantico — el unico del flujo que paga LLM:
   graphify check-update .  y luego  /graphify . --update
   (entran us-resolved y el spec congelado). Confirma:
   graphify query "US-XXX [titulo]" --budget 800
7. graphify reflect — destila las consultas del sprint en LESSONS.md.
8. mem_session_summary.
9. Borra docs/us-work/us-XXX.md (ya esta destilada).
```

---

## Ruta corta — US ligeras
> US de un solo dominio, sin artefacto nuevo del artículo, sin presupuesto (docs, fix acotado,
> refactor chico, gate nuevo). Dos sesiones, spec breve, sin bitácora ni sub-agentes.

```text
Sesion A: pega Fase 2 (el spec puede quedar en ~30 lineas, pero existe y se congela) -> /clear
Sesion B: "Implementa la US-XXX segun el spec docs/us-planning/us-XXX.md" — el orquestador
          programa directo, sin sub-agentes; luego el bloque 0-8 de Fase 4 en la misma sesion,
          reportando en el chat en vez de bitacora.
Cierre:   Fase 7 sin paso de bitacora (el us-resolved puede ser breve, pero existe: el grafo
          lo indexa junto con el spec).
```

---

## Sesiones por escenario

La Sesion 1 arranca en F2 en casi todas las US — F1 solo se antepone (misma sesion) cuando
hay tecnologia o metodo genuinamente nuevo.

| Escenario | Sesion 1 | Sesion 2 | Sesion 3+ |
|---|---|---|---|
| US ligera (docs, gate, fix, 1 dominio chico) | F2 | implementacion + F4 | F7 |
| US de cimientos (EPIC 18: arnes OOF, identidad de modelos, rutas rotas) | F2 | F3 (modeling + mlops) | F4 · F5 aparte si hay bugs · F6-F7 |
| US de protocolo o contraste (EPIC 19, 20, 21, 22, 26, 27) | F2 + candados de §8 | F3 (validar regimen y unidad primero) | F4 · F5-F7 |
| US con coste real (GPU, GEE, LLM en lote) | F2 + presupuesto | F3 con tope confirmado | F4-F7 |
| US de manuscrito (EPIC 23) | F2 | F3 (paper) con cifras ya selladas | F4 · F5 · F6-F7 |
| US de entrega (EPIC 24: zip, DOI, licencias) | F2 | implementacion + F4 | F7 |
| US multi-dominio (US-139, US-144, US-152) | F2 | F3 con 3-4 sub-agentes | F4 · F5 · F6-F7 |
| Deuda tecnica ya programada | F5 | F6-F7 | — |

---

## Modo nocturno / desatendido
> Prerequisito: spec aprobado (congelado) por ti. Ejecuta F3-F4-F6 y se detiene antes de F7.
> Las plantillas de sub-agente son las mismas de Fase 3 — no se duplican aqui.

**Reglas absolutas**: no borra archivos ni commits (reporta); no sobrescribe ni re-sella un
artefacto con fila `SELLADO` u `OBSOLETO`; no ejecuta nada de las EPIC 20, 21, 22 ni 25 sin
preregistro firmado; no alquila GPU, no lanza exports de GEE ni llamadas a Gemini en lote —
cuesta dinero real; no hace `dvc push`; no toca `paper/ARTIFACTS.md`, el preregistro ni el
estimando; solo `make graph-update` (0 LLM), jamás semántico de noche; el spec no se toca.

```text
Arranca la US-XXX: [titulo] en modo nocturno. Autonomo total, sin pedirme confirmacion,
con las excepciones de "Reglas absolutas" de prompts-optimizers-fable.md: si ibas a
hacerlo, NO lo hagas y reportalo al final.

PREREQUISITOS (si falla alguno, detente y reporta):
- docs/us-planning/us-XXX.md existe con estado "aprobado"; presupuesto estimado <= $5 USD.
- docs/orchestration/subagent-prompts/ contiene la plantilla de cada dominio marcado
  en el spec §4 (mejor abortar antes de empezar que programar de noche sin gobierno).
- Crea la rama feature/ENN-US-XXX-[slug].
- Si graphify-out/graph.json no existe: trabaja con grep y reportalo; NO construyas el grafo.

FASE 3: ejecuta el prompt "Fase 3 — Programacion" de prompts-optimizers-fable.md tal cual
(sub-agentes por plantilla contra el spec, integracion, TESTS, graph-update, bitacora §1).
Nota nocturna: cada plantilla trae su modo nocturno — nada que gaste dinero corre de noche.
Si make lint / make test-ml fallan: corrigelos antes de reportar.

FASE 4: ejecuta el prompt "Fase 4 — QA" tal cual, via sub-agente qa-reviewer (foreground).

FASE 6: solo bugs corregibles sin decision humana (linter, tests rojos, tags MLflow,
regimen mal etiquetado, .dvc faltante). Lo que exija criterio humano: documentalo en
bitacora §3 y continua.

AL TERMINAR: make graph-update final; estado "ready-for-human"; escribe en bitacora §2 el
REPORTE NOCTURNO: dominios trabajados; por dominio archivos + lint/tests + cobertura;
falsabilidad y gates (resultado vs umbral); artefactos generados con regimen, semilla,
versiones y commit, y si quedaron PENDIENTES DE SELLAR; desviaciones del spec; bugs
corregidos vs pendientes de humano; anti-patrones detectados; grafo reindexado si/no;
"quise borrar / sellar / alquilar GPU / exportar / llamar LLM pero no lo hice";
siguiente paso para ti. Guarda ese mismo reporte resumido con mem_save (type: project),
para que la sesion del despertar lo encuentre con mem_search aunque no abra la bitacora.
```

**Si se interrumpe**: `Reanuda la US-XXX. Lee la bitacora docs/us-work/us-XXX.md — el Estado
dice donde quedamos (coding→F3, qa→F4, testing→F6, ready-for-human→solo reporta); si no
existe, F3 no arranco: empieza por ahi con el spec. Verifica la rama y git status antes de
repetir nada.` Programable: `/schedule "Reanuda la US-XXX..." at 03:00`.

**Al despertar**: pregunta "¿Qué pasó con la US-XXX anoche?" — Claude lee el reporte de la
bitácora §2. Todo verde → F7 directo · bugs de humano → F5 · quiso gastar → lanzas tú la
corrida real y retoma F4 · artefactos pendientes de sellar → `make paper-artifacts-seal` tras
revisarlos y retoma F4.

---

## GitHub Copilot Pro — diferencias

Copilot lee los `AGENTS.md` por directorio automáticamente, pero no tiene skills, MCP, engram,
sub-agentes ni `graphify`. Compensación: los prompts de fase le sirven casi idénticos —
sustituye cada paso de graphify por revisión manual de `ml/` (o resuélvele tú la consulta en
Claude Code y pégale el resultado en la bitácora §1.6), ignora skills y engram (sin plugin no hay import automático: `make memory-import` tras cada `git pull`,
y la memoria del equipo se lee con `engram search "<tema>" --project agrosat-copilotv2` en la terminal), y en
Fase 3 programa capa por capa secuencial leyendo la plantilla de
`docs/orchestration/subagent-prompts/` del dominio en turno. El spec y la bitácora funcionan igual.
