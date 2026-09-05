---
name: agrosat-engram-memory
description: Memoria persistente de desarrollo del equipo con Engram (Gentleman-Programming/engram) — base SQLite local por maquina, chunks exportados a .engram/ que viajan en cada PR y que el plugin de Claude Code importa solo al arrancar la sesion, proyecto canonico agrosat-copilotv2 fijado en .engram/config.json. Use para persistir decisiones, causas raiz, gotchas y cierres de US entre sesiones y maquinas, para instalar engram en un clon nuevo, y para sincronizar (make memory-sync / memory-import) o reparar el manifest. Nunca guarda secretos, session_ids reales, datos de usuario ni cifras del articulo.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Engram — memoria compartida del equipo (dev-time)

Referencia oficial: [DOCS.md → Git Sync (Chunked)](https://github.com/Gentleman-Programming/engram/blob/main/DOCS.md#git-sync-chunked) y [docs/TEAM-USAGE.md](https://github.com/Gentleman-Programming/engram/blob/main/docs/TEAM-USAGE.md). Lo de aqui es la aplicacion de ese diseno a este repo, no una invencion propia.

## Frontera dura

Engram es herramienta de **desarrollo**. No es parte del runtime del sistema ni del articulo.

| Capa | Memoria | Owner |
|---|---|---|
| Runtime del agente (`/chat`, ADK) | PostgreSQL `chat_sessions` + pgvector con RLS por `session_id` | `agrosat-google-adk-agent` |
| Evidencia del articulo | `paper/ARTIFACTS.md` (ledger) + `reports/paper_micai/` | `agrosat-protocolo-articulo` |
| **Memoria de desarrollo entre sesiones y maquinas** | **Engram: SQLite local + chunks en `.engram/`** | **esta skill** |

Engram **no se importa** desde FastAPI, ADK, Dagster ni scripts del articulo. Una cifra o un
contraste nunca se "recuerda": se lee del ledger.

## Como viaja la memoria (Git Sync, el mecanismo nativo)

```
.engram/
├── config.json            <- fija project_name = agrosat-copilotv2 (se commitea; primera prioridad de deteccion)
├── manifest.json          <- indice de chunks (se commitea)
├── chunks/<id>.jsonl.gz   <- memorias comprimidas, inmutables (se commitean)
└── engram.db              <- si engram la creara aqui: DB de trabajo, gitignorada
```

- **DB local**: `~/.engram/engram.db` (override `ENGRAM_DATA_DIR`). Contiene todos los proyectos de
  esa maquina; **jamas al repo**.
- **Chunks**: `engram sync --project agrosat-copilotv2` exporta las memorias nuevas como un chunk
  nuevo (los viejos nunca se modifican) y lo registra en `manifest.json`. Se commitean y viajan en
  cada PR. `engram sync --import` aplica los chunks listados que aun no se importaron; es
  idempotente (cada chunk se importa una vez, dedup por `sync_id`).
- **Import automatico**: el plugin de Claude Code lo hace solo en el hook `SessionStart` cuando
  encuentra `.engram/manifest.json` en el repo. Tras un `git pull`, la siguiente sesion ya ve la
  memoria del equipo. `make memory-import` es la via manual: otros hosts (Copilot, Codex, terminal)
  o a mitad de una sesion.
- **Proyecto canonico**: `agrosat-copilotv2`, fijado en `.engram/config.json` (`project_name`).
  Es la primera prioridad de deteccion de engram y sobrevive a forks, renombres del remoto y
  clones sin `origin`; ningun clon puede partir la memoria en dos nombres. El historial del curso
  (`agrosat-copilot` + `agro_sat_copilot`, 388 observaciones de mayo a septiembre de 2026) se fusiono
  ahi con `mem_merge_projects` el 4-sep-2026 y viaja en el primer chunk.
- **Cloud OFF**: no se enrola ningun proyecto en Engram Cloud (`engram cloud status` = no enrolado).
  La sincronizacion es git.
- **Nunca `engram sync --all`**: exporta TODOS los proyectos de la laptop (incluidos ajenos) al
  repo. `make harness-check` falla si un chunk trae memorias de otro proyecto o un token.

```bash
make memory-status    # chunks locales, remotos y pendientes de importar
make memory-import    # manual: repara el manifest y aplica los chunks nuevos (el plugin lo hace solo al arrancar)
make memory-check     # solo verifica el manifest, exit 1 si hay drift
make memory-sync      # antes de abrir PR: exporta lo nuevo a .engram/ (commitear)
make memory-setup     # una vez por clon: merge driver para .engram/manifest.json
engram doctor         # diagnostico de solo lectura si algo no cuadra
```

## Protocolo en cada sesion de Claude Code

1. **Inicio**: el hook del plugin importa los chunks nuevos e inyecta contexto reciente; `mem_search`
   con las palabras clave de la US o del problema (o `mem_context`).
2. **Durante**: `mem_save` inmediatamente tras una decision (con su ADR si existe), un bug
   corregido (con causa raiz), una convencion nueva, un gotcha entre maquinas, o cuando el
   humano confirma o rechaza una recomendacion. Guarda el **porque**; el que ya esta en git.
3. **Conflictos**: si `mem_save` devuelve `judgment_required`, resuelve con `mem_judge` por
   candidato; pregunta al humano solo si la relacion es `supersedes` o `conflicts_with` sobre
   una decision o arquitectura.
4. **Cierre**: `mem_session_summary` (objetivo, hallazgos, hecho, siguientes pasos, archivos).
5. **Antes del PR**: `make memory-sync` y `.engram/` en el commit.

## Convenciones de equipo (TEAM-USAGE)

- **`scope: project`** (por defecto) para todo lo que otro coautor o su agente deba encontrar;
  se escribe en **espanol neutro**, la lengua franca del equipo, con identificadores de codigo en
  ingles. FTS5 no cruza idiomas: una memoria en ingles no aparece en una busqueda en espanol.
- **`scope: personal`** NO evita que la memoria viaje: `engram sync` exporta el proyecto entero.
  Las notas personales van bajo **otro nombre de proyecto** (por ejemplo `arthu`, que ya existe)
  o fuera de engram. Nunca bajo `agrosat-copilotv2`.
- Formato de `mem_save`: QUE / POR QUE / DONDE / APRENDIDO, con titulo que se pueda buscar.

## Que guardar y que no

**SI**: el porque de una decision de protocolo o de arquitectura · la causa raiz de un bug
(`dump_oof.py` no pasaba `n_timesteps` y un T=37 recibia T=10) · un gotcha de entorno (MPS no
sirve para TSViT; `poetry run` crea un venv vacio si nadie hizo `poetry install`) · la leccion de
una US al cerrarla · punteros a documentos canonicos.

**NO**: nada de `.env*`, Secret Manager ni tokens (`sk-`, `Bearer`, `hf_`, `AIza`) · `session_id`
reales, correos, matriculas · geometrias de parcelas de usuario · lo que ya esta en codigo o git
· lo que cambia cada semana (estado del tablero) · **cifras del articulo**: una metrica en engram
es una cifra sin fila sellada.

## Formato sugerido de `mem_save`

```json
{
  "title": "paired_interval exige declarar la unidad y rechaza < 3 clusteres",
  "type": "decision",
  "content": "QUE: ml/eval/paper_micai_coverage.py::paired_interval recibe unit= obligatorio. POR QUE: el bootstrap remuestreaba parcelas dentro del bloque (defecto 3 de la auditoria externa, 4-sep-2026); con 5 bloques no hay replicas. DONDE: tests/ml/eval/test_paper_micai_coverage.py falla sobre la version anterior. APRENDIDO: los artefactos de fase3/fase4 quedan OBSOLETO hasta regenerarse (US-124/125)."
}
```

Desde la terminal (Copilot, Codex u otra maquina sin MCP):

```bash
engram search "denominador movil" --project agrosat-copilotv2 --limit 5
engram save "titulo" "contenido" --type decision --project agrosat-copilotv2
```

## Instalacion en un clon nuevo

```bash
# 1. Binario (Go) — Windows, macOS y Linux
go install github.com/Gentleman-Programming/engram/cmd/engram@latest   # o brew install gentleman-programming/tap/engram
engram --version        # >= 1.20

# 2. Plugin de Claude Code (hooks, scripts, skill) + registro durable del MCP
claude plugin marketplace add Gentleman-Programming/engram
claude plugin install engram
engram setup claude-code          # escribe ~/.claude/mcp/engram.json y ofrece el allowlist de usuario
# reiniciar Claude Code

# 3. Merge driver del manifest (una vez por clon); la memoria se importa sola al arrancar la sesion
make memory-setup
```

`.claude/settings.json` ya permite las herramientas `mem_*` no destructivas bajo los dos ids que
Claude Code puede usar (`mcp__plugin_engram_engram__*` y `mcp__engram__*`); `mem_delete` y
`mem_merge_projects` (perfil `admin`) exigen aprobacion por llamada. No duplicar un
`mcpServers.engram` manual en el proyecto: `engram setup claude-code` es el unico dueno del registro.

## Por que los chunks ya pueden viajar en git

La objecion que bloqueaba el sync de equipo era real: `engram sync` exporta el proyecto entero
—sesiones, prompts y observaciones— y upstream documenta que `scope: personal` **no** queda fuera
automaticamente. Compartir chunks sin poder mirar dentro era pedir una fuga.

Lo que la desbloquea es el gate que faltaba: `make harness-check` abre cada `.jsonl.gz` y recorre
las cuatro listas del chunk (`observations`, `prompts`, `sessions`, `mutations`) buscando dos
cosas, y falla si encuentra cualquiera:

- una fila cuyo `project` no sea `agrosat-copilotv2` (la firma de un `engram sync --all`);
- un token con forma de secreto (`sk-`, `hf_`, `AIza`, `ghp_`, `AKIA`, `Bearer`, clave privada).

Por eso las notas personales van bajo **otro nombre de proyecto**, nunca bajo este: el filtro es
el proyecto, no la etiqueta `scope`. Y por eso `engram sync --all` esta prohibido de plano.

## Reparar `.engram/manifest.json`

El manifest es el indice de chunks; `engram sync --import` **solo importa lo que esta listado**.
Dos PRs que exportan a la vez chocan en el JSON (los dos anaden al final del array). `make memory-setup`
registra un merge driver que une las entradas por `id`; si el conflicto llega igual:

```bash
python scripts/engram_manifest_merge.py          # repara el manifest (union por id, orden por fecha); lo corre make memory-import
python scripts/engram_manifest_merge.py --check  # solo verifica: cada chunk listado existe y viceversa
```

Nunca resolverlo "tomando un lado": se pierde el chunk del otro y su memoria no llega.

## Checklist de verificacion

- [ ] `engram --version` >= 1.20 en cada maquina
- [ ] `claude mcp list` muestra engram conectado
- [ ] `mem_current_project` devuelve `agrosat-copilotv2` con `project_source: config`
- [ ] `engram cloud status` = no enrolado
- [ ] `make memory-status` sin chunks pendientes tras la primera sesion
- [ ] `.engram/config.json`, `manifest.json` y `chunks/` trackeados; ningun `*.db*` en git
- [ ] `make harness-check` en verde: chunks solo de `agrosat-copilotv2`, sin tokens

## Cuando NO usar esta skill

- Memoria del chat del usuario final → `agrosat-google-adk-agent` + `agrosat-spatial-rag`.
- Evidencia del articulo, metricas, artefactos → `agrosat-protocolo-articulo` + `agrosat-dvc-mlflow`.
- Versiones de datasets y pesos → DVC.
