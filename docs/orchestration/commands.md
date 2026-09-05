# Comandos Make — AgroSatCopilot v2 (MICAI 2027)

> Lista operativa de targets `make`. Resumen ejecutivo en [`AGENTS.md`](../../AGENTS.md). `make help` imprime todos los targets con descripción.

El ciclo recomendado para una US esta en [`prompts-optimizers-fable.md`](prompts-optimizers-fable.md).

## Quality gates (reemplazan pre-commit)

```bash
make check                  # lint + secrets-scan + i18n-check + guides-check (obligatorio antes de PR)
make lint                   # ruff check + ruff format --check + mypy (backend, ml, dagster_project, scripts) + pnpm lint
make format                 # ruff format
make secrets-scan           # gitleaks detect --no-banner --redact
make i18n-check             # claves it/es/en sincronizadas
make notebooks-check        # papermill end-to-end (notebooks del curso, historia)
```

## Harness: grafo, memoria compartida y guías espejo

```bash
make graph-update           # reindexa el grafo de codigo (AST, 0 LLM); GRAPHIFY_FORCE=1 tras borrar codigo
make graph-check            # falla si graphify-out/graph.json no describe HEAD
make graph-hooks            # hooks git post-commit / post-checkout de graphify (una vez por clon)
make memory-sync            # exporta las memorias nuevas de engram a .engram/ (commitear)
make memory-import          # manual: repara el manifest y aplica los chunks nuevos (el plugin de Claude Code lo hace solo al arrancar)
make memory-check           # verifica el manifest sin tocarlo (exit 1 si hay drift)
make memory-status          # chunks locales, remotos y pendientes
make memory-setup           # merge driver para .engram/manifest.json (una vez por clon)
make harness-check          # espejos, settings, plantillas, skills, agents, chunks, grafo
make harness-status         # rama, grafo, memoria pendiente, US en vuelo, espejos, plan
make guides-sync            # AGENTS.md -> CLAUDE.md en todos los pares
make guides-check           # falla si algun par difiere
make plan-check             # plan por epicas del cuaderno: dependencias, ciclos, estados, camino critico
```

## Artículo MICAI 2027

```bash
make paper-artifacts-check  # recalcula el MD5 de cada fila sellada del ledger
make paper-artifacts-seal   # sella artefactos nuevos en paper/ARTIFACTS.md
make paper-obsoletos-check  # ningun documento activo cita OBSOLETO sin cuarentena
make preregistro-check      # estimando-v1.json == preregistro seccion 4.5
make protocolo-check        # el protocolo de US-172 no se congela con campos vacios
make oof-manifest-check     # cada oof_parcel_* del arnes tiene entrada en manifest.json
make us172-adjuntos         # los cuatro PDF de la consulta al comite de etica
make micai-pdf              # manuscrito anonimo: cero errores, cero overfull
make micai-pdf-es           # version en espanol
make micai-anon-check       # gate de doble ciego, con autoprueba en negativo
make micai-bib              # regenera paper/micai/refs.bib desde la matriz verificada
make micai-pdf-cr           # camera-ready (comprueba que SI revela identidad)
make paper-cite-check       # \cite{} <-> bib
```

## Tests

```bash
make test                   # pytest backend con cobertura >= 70 %
make test-unit              # backend/tests/unit
make test-integration       # backend/tests/integration (Docker)
make test-ml                # pytest tests/ml (excluye slow)
make test-frontend          # vitest
make test-e2e               # Playwright
make test-all               # backend + ml + frontend
```

## Datos, modelos y experimentos

```bash
make dvc-pull / make dvc-push          # remoto gs://agrosat-dvc-remote
make mlflow-up / make mlflow-down      # server MLflow en Docker (:5010)
make mlflow-ui / make dagster-ui
make train-baseline                    # RF + XGB sobre AlphaEarth + indices (CPU)
make train-l4 epic=E19 us=US-138 script=train_segmentation.py   # L4 spot en GCP, con tope en el spec
make baseline-test / ensembles-test / interpretability-test / learning-curves-test
make farslip-extract-embeddings        # embeddings FarSLIP desde el student registrado
```

Retirados el 4-sep-2026 (sin hardware): `train-h100`, `azure-h100-start`, `azure-h100-stop`, `azure-h100-status`.

## Base de datos (dbmate)

```bash
make db-migrate             # dbmate up
make db-rollback            # dbmate down
make db-new name=xxx
make db-status
make db-seed
make db-shell
```

## Desarrollo local del sistema (mantenimiento)

```bash
make dev                    # docker-compose (api, frontend, postgres, redis, titiler, mlflow, dagster)
make stop
make demo / make demo-down
```

## Infraestructura, FinOps y seguridad

```bash
make tf-init env=dev / tf-plan env=dev / tf-apply env=dev / tf-fmt / tf-validate env=dev
make deploy-staging / make deploy-prod
make cost-audit             # scripts/cost_audit.sh (GCP)
make scale-to-zero-check
make security-audit
```

## Manuscrito heredado (informe técnico interno, ADR-014 §5)

```bash
make paper-pdf / paper-pdf-docker / paper-pdf-clean   # paper/main.tex, solo para regenerar el informe
make paper-tables / paper-figures                     # US-070, data-driven desde reports/
```
