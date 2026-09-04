# Workflow Fable por fases para US

**Estado**: adoptado como guía complementaria, revisada el 4 de septiembre de 2026  
**Gobierno**: `AGENTS.md` y la guía de la carpeta prevalecen ante cualquier conflicto  
**Origen recibido**: `/Users/haowei/Downloads/prompts_optimizers_fable.md`  
**SHA-256 del original**: `13063f3d81b921a769559c6992c5fc7ed0a76f5b82df289e1c2fcf6b39e180e3`

Este documento conserva la idea útil del material recibido de Arthur: separar contrato,
implementación, auditoría y cierre, de modo que una US se pueda retomar desde el repositorio. No
convierte capacidades opcionales de un cliente en requisitos del proyecto ni declara activos
controles que aquí no existen.

## Cambios hechos durante la incorporación

1. **El spec se conserva y la bitácora se destila.** Para US nuevas, el contrato vive en
   `docs/us-planning/us-XXX.md`; el estado de ejecución puede vivir en `docs/us-work/us-XXX.md` y al
   cierre se destila a `docs/us-resolved/us-XXX.md`. `docs/us-handoff/` es legado: no se borra ni se
   migra en masa.
2. **La frontera de sesión es una recomendación, no un gate.** Se abre una sesión limpia cuando el
   cliente lo permita y aporte claridad. Un agente no se detiene solo porque `/clear` no exista.
3. **La delegación es condicional.** Solo se usan subagentes cuando el usuario lo pide, el arnés los
   ofrece y el trabajo puede dividirse sin escrituras conflictivas. Este repositorio no contiene aún
   `docs/orchestration/subagent-prompts/`, por lo que esa ruta no se presenta como disponible.
4. **Graphify es opcional.** No hay binario ni targets `graph-update`/`graph-hooks` acreditados en el
   repositorio. Se usa `rg`, `rg --files`, imports, tests y `git log` hasta que exista una instalación
   verificable. Nunca se afirma que un grafo fue actualizado si el comando no corrió.
5. **Engram degrada de forma segura.** La configuración actual contiene una lista de permisos, pero
   `enabledPlugins` está vacío, el binario no está en `PATH` y no existen `make memory-sync` ni
   `make memory-import`. Por tanto, la memoria Engram **no está activa en esta máquina**. Su ausencia
   nunca bloquea una US y su contenido nunca sustituye al spec, al ADR, al ledger ni a Git.
6. **No hay sincronización automática de memoria.** `engram sync` escribe `.engram/` y puede incluir
   sesiones, prompts y observaciones del proyecto, incluso de alcance personal. `.engram/` permanece
   ignorado. Lo compartible se redacta y revisa primero como ADR o `us-resolved`; no se hace `git add
   -f .engram` ni se habilita nube sin una decisión explícita del equipo.
7. **Los comandos se comprueban antes de exigirlos.** Los gates MICAI disponibles son
   `plan-check`, `preregistro-check`, `protocolo-check`, `paper-artifacts-check`,
   `paper-obsoletos-check` y `oof-manifest-check`. La lista completa de comandos está en
   [`commands.md`](commands.md).
8. **Las acciones externas conservan su frontera.** `git push`, `dvc push`, despliegues, alquiler de
   GPU, exportaciones GEE, llamadas pagadas y mensajes a terceros solo se ejecutan si el encargo los
   autoriza.

## Fuentes verificadas

- Anthropic, [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents),
  29 de septiembre de 2025: recomienda contexto de alta señal, recuperación justo a tiempo,
  compactación, notas estructuradas y arquitecturas multiagente cuando correspondan.
- Sourcegraph, [*Context Engineering: A Practical Guide for AI Agents*](https://sourcegraph.com/blog/context-engineering),
  28 de mayo de 2026: trata instrucciones, recuperación, memoria y herramientas como un sistema de
  contexto, no como un único prompt.
- Galster et al., [*Harness Engineering for Agentic AI Coding Tools*](https://arxiv.org/abs/2602.14690),
  arXiv v5, 30 de junio de 2026: estudia 2 853 repositorios y encuentra que los archivos de contexto
  dominan; skills y subagentes son menos frecuentes. Es evidencia descriptiva, no una obligación de
  arquitectura.
- Xu et al., [*TokenPilot*](https://arxiv.org/abs/2606.17016), arXiv v2, 28 de agosto de 2026:
  reporta reducciones de coste de 56 % a 87 % en sus propios bancos mediante continuidad de caché y
  compactación. No demuestra que copiar un prompt literalmente active caché en todos los clientes.

## Fuentes de verdad de una US

| Fuente | Función | Se edita cuándo |
|---|---|---|
| Cuaderno público `agrosat-micai-site/plan.html` | Estado, dependencia, criterio y avance visibles | Cuando cambia el plan verificable |
| `docs/us-planning/us-XXX.md` | Contrato, interfaces, riesgos y pruebas | Hasta aprobación; después, revisión explícita |
| `docs/us-work/us-XXX.md` | Estado temporal, hallazgos y correcciones | Durante implementación y QA |
| `docs/us-resolved/us-XXX.md` | Cierre destilado y evidencia | Al cerrar |
| ADR, preregistro, estimando y ledger | Decisiones y custodia científicas | Solo cuando la US los afecta |
| Engram local, si existe | Índice de razones y hallazgos recuperables | Nunca como fuente normativa |

Un número destinado al artículo se rederiva desde un artefacto sellado. Un log, una memoria o una
frase en el plan no son su fuente.

## Plantillas mínimas

### Spec permanente

```markdown
# Spec US-XXX — Título

**Estado**: borrador | aprobado (congelado)
**Épica**: EPIC NN · **Rama**: feature/ENN-US-XXX-slug · **Depende de**: ... · **SP**: N

## 1. Criterios de aceptación y cómo se verifican
## 2. Arquitectura: archivos que se extienden y archivos nuevos
## 3. Interfaces y esquemas públicos
## 4. Dominios tocados
## 5. Trazabilidad científica: afirmación → contrato → artefacto → ledger
## 6. Pruebas de falsabilidad y gates negativos
## 7. Tests, riesgos y presupuesto
## 8. Candados: régimen, unidad, dependencia, multiplicidad y acciones externas
## A. Investigación previa, si fue necesaria
## Revisiones explícitas posteriores a la aprobación
```

### Bitácora temporal

```markdown
# US-XXX — bitácora de ejecución

**Estado**: coding | qa | testing | ready-to-close | ready-for-human

## 1. Archivos tocados y reutilizados
## 2. Decisiones y desviaciones del spec
## 3. Pruebas, gates y resultados
## 4. Artefactos, versiones, semillas, DVC/MLflow y ledger
## 5. Hallazgos de QA
## 6. Bugs: causa, arreglo y verificación
## 7. Siguiente acción exacta
```

## Fase 1 — Investigación, solo cuando hace falta

Usar si la US introduce un método, biblioteca o banco no establecido.

```text
Investiga la US-XXX: [título].

1. Lee su objeto en plan.html y las guías AGENTS.md aplicables.
2. Busca primero en el repositorio con rg/rg --files y git log.
3. Lee preregistro, estimando o ledger solo si la US toca protocolo o cifras.
4. Consulta documentación primaria vigente cuando la API o el método puedan haber cambiado.
5. Si Engram está disponible, úsalo solo para localizar razones y verifica cada hit en Git.
6. Escribe como máximo 40 líneas en §A del spec. No programes.
```

## Fase 2 — Planeación y congelación del contrato

```text
Planifica la US-XXX: [título].

1. Copia los criterios del cuaderno sin reinterpretarlos y vuelve cada uno verificable.
2. Declara qué existe y se extiende; no dupliques productores ni fuentes de verdad.
3. Si habrá una cifra o conclusión científica, enlázala al preregistro/estimando y al artefacto.
4. Diseña al menos una prueba negativa por control nuevo: prueba el mecanismo, no el conteo.
5. Declara unidad de análisis, dependencia, multiplicidad, población y punto de operación.
6. Declara presupuesto y toda acción externa que necesitará autorización.
7. Escribe el spec como borrador. No implementes.

Al aprobarlo, cambia a "aprobado (congelado)". Un cambio posterior lleva revisión fechada.
```

## Fase 3 — Implementación

```text
Implementa la US-XXX desde su spec aprobado.

1. Verifica rama, HEAD y árbol; preserva cambios ajenos.
2. Lee solo las guías de los directorios que tocarás.
3. Implementa por capas y resuelve explícitamente sus fronteras de datos.
4. Si el usuario pidió subagentes y el arnés los ofrece, delega solo tareas independientes.
5. Registra desviaciones; si invalidan un criterio, detén esa vía y consulta.
6. Añade tests que fallen con la conducta anterior y distingan el mecanismo nuevo.
7. Ejecuta checks focalizados y actualiza la bitácora con evidencia y siguiente acción.
8. Si Engram está activo, guarda solo la razón no obvia, sin secretos, PII ni datos de clientes.
```

## Fase 4 — QA independiente del relato

```text
Audita la US-XXX contra el spec y el diff, no contra la bitácora.

1. Recorre consumidores río abajo con imports, rg y tests.
2. Ejecuta make check y la suite proporcional al riesgo.
3. Revisa cada criterio contra comportamiento real.
4. Para paper: ejecuta los gates de custodia, obsoletos, manifiesto y preregistro aplicables.
5. Para datos: comprueba DVC, cobertura, población y ausencia como no entrega.
6. Intenta el camino de al lado de cada control nuevo.
7. Busca la afirmación corregida en plan, preregistro, cuaderno, ledger, docstrings y manuscrito.
8. Registra hallazgos con severidad y evidencia reproducible.
```

## Fase 5 — Correcciones

```text
Corrige los hallazgos de la US-XXX uno a uno.

1. Reproduce el defecto antes de editar.
2. Arregla la causa y busca todas sus apariciones.
3. Añade o fortalece el test para que distinga la conducta correcta.
4. Si una cifra queda invalidada, identifica también qué conclusiones caen con ella.
5. No regeneres o reselles artefactos, gastes recursos ni publiques sin autorización aplicable.
6. Ejecuta checks focalizados después de cada cambio lógico y actualiza la bitácora.
```

## Fase 6 — QA posterior a las correcciones

```text
Verifica el diff de correcciones contra los hallazgos y el spec.

1. Ejecuta make check y las suites afectadas.
2. Repite las pruebas negativas y los caminos alternos.
3. Confirma que los documentos consumidores no conservan conclusiones invalidadas.
4. Confirma DVC, ledger y gates del paper cuando apliquen.
5. Marca cada bug como verificado solo con evidencia de comportamiento.
```

## Fase 7 — Cierre y publicación

```text
Cierra la US-XXX solo si sus criterios están satisfechos.

1. Revisa git status y git diff --check.
2. Destila spec y bitácora en docs/us-resolved/us-XXX.md.
3. Incluye criterios, evidencia, desviaciones, artefactos, versiones, gates y límites.
4. Actualiza plan.html únicamente con avances que ya existen en código o artefactos.
5. Corre make plan-check en el sitio y los gates relevantes en el repositorio principal.
6. Haz commits convencionales sin trailers de asistentes.
7. Push y despliegue solo si el encargo los autorizó; comprueba después el estado remoto.
8. Elimina la bitácora solo después de que el cierre destilado esté versionado.
```

## Ruta corta

Una US documental o un fix acotado de un dominio puede usar spec breve, implementación, QA y cierre
en dos sesiones. No omite la prueba negativa si crea un control, ni la custodia si cambia cifras.

## Modo desatendido

Solo se admite con spec aprobado y límites ya escritos. Nunca borra, resella, hace push, gasta,
contacta terceros ni cambia preregistro/estimando/ledger por inferencia. Si falta una capacidad, usa
el fallback local o deja la US `ready-for-human`; jamás la simula con una frase de éxito.

## Engram: función y límite

Engram es una memoria local de **desarrollo** basada en SQLite/FTS5. Sirve para recuperar por qué se
tomó una decisión, un fallo no obvio o una preferencia estable. No es memoria del agente de producto,
no entra a FastAPI, ADK ni Spatial-RAG y no almacena usuarios, correos, `session_id`, tokens,
credenciales, parcelas de clientes ni resultados que ya tienen artefacto.

Cuando el binario y las herramientas MCP estén realmente disponibles:

```text
inicio:  confirmar proyecto y buscar la US o decisión
durante: guardar solo razones estables y verificadas
cierre:  resumen de sesión sin secretos ni datos personales
```

Hasta entonces, el mecanismo de continuidad es Git: spec, bitácora, `us-resolved`, ADR y ledger.
