## US-XXX — [titulo]

**Epica**: EPIC NN · **Spec**: `docs/us-planning/us-XXX.md` (estado: aprobado / congelado) ·
**Bitacora**: `docs/us-work/us-XXX.md` (estado al abrir el PR: ...) · **Cuaderno**: estado de la US en `plan.html`

### Que entra

-

### Trazabilidad cientifica (solo si toca cifras, contrastes o artefactos)

| Cifra / contraste | Regimen y unidad | Fila del ledger | Seccion del preregistro |
|---|---|---|---|
| | | | |

### Gate previo al PR

- [ ] `make check` limpio (lint + secrets + i18n + espejos `AGENTS.md`/`CLAUDE.md`)
- [ ] Cobertura >= 70 % por archivo del diff (dos suites: `backend/tests` y `tests/`)
- [ ] Si toco evaluacion o artefactos: `make paper-artifacts-check`, `make paper-obsoletos-check`, `make oof-manifest-check`
- [ ] Si toco protocolo o estimando: `make preregistro-check`, `make protocolo-check`
- [ ] Si toco el manuscrito: `make micai-pdf`, `make micai-anon-check`, `make paper-cite-check`
- [ ] Si entreno o genero datos: MLflow con `data_version` + `code_version`; `.dvc` + `dvc push`
- [ ] Si cambio el estado de la US en el cuaderno: `make plan-check`
- [ ] `make memory-sync` ejecutado y `.engram/` incluido en el PR
- [ ] Sin trailer `Co-Authored-By` de asistentes IA; sin emojis; commits `tipo(ENN): ...`
- [ ] Ninguna dependencia nueva de H100, Azure, Gemma 4 LoRA ni ids inexistentes

### Desviaciones del spec (citando la seccion) y pendientes de humano

-
