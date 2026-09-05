---
name: agrosat-git-workflow
description: Convenciones de ramas (feature/E{epic}-US-XXX-{slug} sobre main), Conventional Commits con scope de epica, PR con plantilla y memoria engram sincronizada, cierre de US por el loop (spec -> bitacora -> us-resolved) y actualizacion del cuaderno. Use al crear ramas, commitear, abrir PR, cerrar una US o cambiar su estado en el plan por epicas.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Git workflow

## Rules — NON-NEGOTIABLE

- Rama por US: `feature/E{epic}-US-XXX-{slug}`; correcciones acotadas: `fix/E{epic}-US-XXX-{slug}`.
  Base y destino: **`main`** (no existe `develop`; los deploys del sistema estan dormidos).
- **Conventional Commits** con scope de epica: `feat(E19): ...`, `fix(E18): ...`, `docs(E23): ...`,
  `chore(harness): ...`. El cuerpo explica **por que**; el diff ya dice que.
- **Sin trailer `Co-Authored-By`** de asistentes IA ni pies "generado con" en PRs. Autoria real.
- **Sin emojis** en mensajes de commit.
- `make check` limpio antes de abrir PR; artefactos grandes por DVC (al repo solo el `.dvc`).
- `make memory-sync` antes del PR: los chunks nuevos de `.engram/` van en el mismo commit.
- Nunca `--no-verify`, nunca `--force` sobre `main`, nunca un `dvc push` que sobrescriba un
  artefacto con fila `SELLADO`.
- Nunca commitear `graphify-out/`, `*.engram.db*`, `.env.local`, auxiliares LaTeX ni PDFs
  compilados del manuscrito.

## Flujo por US (integrado con el loop)

```bash
git checkout main && git pull
make memory-import                                   # la memoria del equipo, antes de empezar
git checkout -b feature/E19-US-124-denominador-comun

# F2: spec en docs/us-planning/us-124.md (aprobado -> congelado)
git commit -m "docs(E19): spec de US-124, denominador comun fijado antes de la entrega"

# F3-F6: codigo, tests, bitacora docs/us-work/us-124.md
git add ml/eval/paper_micai_coverage.py tests/ml/eval/test_paper_micai_coverage.py
git commit -m "feat(E19): macro_over exige el universo del bloque desde entrenamiento"
dvc add reports/paper_micai/fase3/frontera_v2.parquet && git add reports/paper_micai/fase3/*.dvc
git commit -m "feat(E19): frontera regenerada con el modulo reparado, pendiente de sellar"

# F7: cierre
make check && make test-ml
make memory-sync && git add .engram/
git add docs/us-resolved/us-124.md && git rm docs/us-work/us-124.md
git commit -m "docs(E19): cierra US-124 con su us-resolved y la memoria del equipo"
git push -u origin feature/E19-US-124-denominador-comun
gh pr create --base main --title "feat(E19): US-124 denominador comun fijado antes de la entrega"
```

La plantilla de PR (`.github/PULL_REQUEST_TEMPLATE.md`) trae el gate previo; se rellena, no se borra.

## Cierre de US (Fase 7 del loop)

1. `docs/us-resolved/us-XXX.md` destilado desde la bitacora; la bitacora se borra.
2. Cifras del cierre con su fila del ledger; artefactos con `.dvc` y `dvc push`; MLflow con tags.
3. Estado de la US en el cuaderno: editar `plan.html` del repo hermano `agrosat-micai-site`,
   `make plan-check` en verde, commit alli (`docs: cierra US-XXX`) y aviso al humano para publicar.
4. `/graphify . --update` (semantico, unico punto que paga LLM) y `mem_session_summary`.
5. PR a `main`; squash merge; borrar la rama.

## Tipos de commit

| Tipo | Uso |
|---|---|
| `feat` | Funcionalidad, experimento o artefacto nuevo de una US |
| `fix` | Correccion de bug (con la causa raiz en el cuerpo) |
| `refactor` | Reestructuracion sin cambio de comportamiento |
| `docs` | Spec, us-resolved, ADR, preregistro, guias, manuscrito |
| `test` | Tests |
| `build` | DVC, artefactos versionados, dependencias |
| `chore` | Configuracion, CI, harness |

## Quality gates (sin pre-commit)

```bash
make check             # lint + secrets-scan + i18n-check + guides-check
make test-ml           # o make test para backend; cobertura por archivo del diff
make harness-check     # si toco AGENTS.md, skills, agents o docs/orchestration
make plan-check        # si toco el estado de una US en el cuaderno
```

CI (`.github/workflows/ci.yml`) corre lint, unit tests, migraciones, gitleaks, gates stdlib del
articulo y `harness-check` sobre cada PR a `main`; la suite pesada se corre en local.

## QA checklist

- [ ] Rama y commits siguen la convencion; sin trailer de IA; sin emojis
- [ ] PR referencia la US, el spec y el estado de la bitacora
- [ ] `make check` y la suite en verde; cobertura por archivo >= 70 %
- [ ] `.engram/` actualizado en el PR
- [ ] Si cerro la US: `us-resolved` presente, bitacora borrada, cuaderno actualizado
- [ ] CI verde y review de un coautor antes del merge
