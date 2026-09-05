# Plantilla MLOPS — Fase 3 (en serie, al final)

> Te lanzo el orquestador cuando las capas de dominio ya integraron. Subagente: `mlops-engineer`.
> Alcance: `Makefile`, `scripts/*_check.py`, `scripts/paper_artifacts_seal.py`,
> `.github/workflows/ci.yml`, los `.dvc`, `ml/utils/mlflow_utils.py`,
> `docs/orchestration/commands.md` y — solo cuando el humano lo autoriza — `paper/ARTIFACTS.md`.
> **No toques el codigo de dominio de esta US ni de otra**: un defecto en `ml/` se reporta al
> orquestador, no se parchea desde aqui.

1. Lee el spec `docs/us-planning/us-XXX.md` (§2 archivos exactos, §5 filas del ledger que la US
   promete, §6 gates, §7 checklist) y el resumen de artefactos que te pase el orquestador. El
   spec esta congelado: si tienes que desviarte, reportalo citando la seccion.
2. Carga `/agrosat-dvc-mlflow` y, si tocas el ledger o un gate del articulo,
   `/agrosat-protocolo-articulo`.
3. Protocolo graphify del AGENTS.md raiz: `query` + `affected` antes de tocar un gate. Eres
   consumidor del grafo — NO ejecutes `make graph-update`.

## Reglas duras del dominio

- **Sellar = fila nueva en `paper/ARTIFACTS.md`** con elemento, ruta, MD5, bytes, commit y
  estado, via `make paper-artifacts-seal`; nunca editar un MD5 a mano. Un artefacto que
  sustituye a otro cambia el estado del viejo (`OBSOLETO`, con motivo) en vez de borrar la fila.
- **Todo gate nuevo se prueba en negativo** antes de confiar en el: se rompe a proposito un
  byte, un campo o una cita, se comprueba el rojo, y ese caso queda como test.
- Los gates del articulo son stdlib puro (`scripts/*_check.py`) para que CI los corra sin
  `poetry install`; si un gate necesita dependencias pesadas, se declara y queda en local.
- `.dvc` + `dvc push` para parquet, OOF, checkpoints y datasets; al repo solo el `.dvc`. El
  `md5` del `.dvc` es parte de la custodia. Un `dvc push` que sobrescriba un artefacto sellado
  esta prohibido.
- MLflow con `data_version` (hash DVC) + `code_version` (git sha) en toda corrida; el fallo del
  servidor degrada a warning, nunca aborta el artefacto auditable.
- `make check` limpio incluye `secrets-scan`: ninguna credencial en YAML, log ni manifiesto.
- **No existe H100**: ningun target, script ni doc nuevo puede depender de `azure_h100_*` ni de
  `train-h100`. Si el spec lo pide, reportalo.
- CI barata: la suite completa (ML con datos DVC, testcontainers) corre en local; en CI solo
  lint, unit, gates y espejos.

## Cierre

- `make harness-check`, `make paper-artifacts-check`, `make paper-obsoletos-check` y
  `make oof-manifest-check` en verde; `dvc status` limpio para lo que tocaste. Si dejas algo
  ajeno rancio, **dilo con su nombre** en vez de repararlo en silencio.
- Verifica idempotencia: dos corridas seguidas del runner producen los mismos hashes.
- `make lint` — si falla, corrigelo antes de reportar.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=30 lineas: filas selladas o pendientes, gates creados y su prueba en negativo, `.dvc`
  nuevos, targets Make, cambios en CI, desviaciones del spec, y que queda rancio y por que.
- No guardes memoria engram ni reindexes el grafo: el orquestador integra tu resumen y hace
  el unico `mem_save` y el unico `make graph-update` de la fase (un solo escritor, regla R4).
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: identico salvo que **no sella ni hace `dvc push`**: deja la lista de
artefactos pendientes de sellar en el resumen para que el humano decida al despertar.
