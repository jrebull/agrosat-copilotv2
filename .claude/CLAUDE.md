# AgroSatCopilot v2 — pointer

Claude Code carga `../CLAUDE.md` y otros agentes de código (Codex, Copilot, Cursor) cargan `../AGENTS.md`. Ambos son **espejos byte a byte** del orquestador único: se edita `AGENTS.md` y se propaga con `make guides-sync`.

**Estructura canónica de orquestación**:

- [`../AGENTS.md`](../AGENTS.md) y [`../CLAUDE.md`](../CLAUDE.md) — orquestador único: identidad (artículo MICAI 2027 sobre el sustrato AgroSatCopilot), hechos verificados del estudio, documentos normativos, decisiones irrevocables y descartados, comandos, grafo, memoria compartida, reglas de código y científicas, QA gate, routing, anti-patrones.
- **Plan vigente**: cuaderno [`agrosat2027.netlify.app/plan`](https://agrosat2027.netlify.app/plan) (fuente: `plan.html` del repo hermano `agrosat-micai-site`, `make plan-check`) · fases en [`../docs/plan-micai-2027.md`](../docs/plan-micai-2027.md) · alcance en [ADR-014](../docs/decisions/ADR-014-micai-2027.md).
- **Documentos normativos**: [`../docs/paper/preregistro-v2-borrador.md`](../docs/paper/preregistro-v2-borrador.md) · [`../docs/paper/estimando-v1.json`](../docs/paper/estimando-v1.json) · [`../paper/ARTIFACTS.md`](../paper/ARTIFACTS.md).
- [`../docs/orchestration/prompts-optimizers-fable.md`](../docs/orchestration/prompts-optimizers-fable.md) — loop por fases F1-F7, ruta corta, modo nocturno; spec en `docs/us-planning/`, bitácora en `docs/us-work/`, cierre en `docs/us-resolved/`.
- [`../docs/orchestration/subagent-prompts/`](../docs/orchestration/subagent-prompts/) — plantillas por dominio (geo-data, modeling, paper, app, mlops, tests) que lee cada sub-agente.
- [`../docs/orchestration/`](../docs/orchestration/) — auto-invoke, catálogo de skills, mapa skill↔subagente y subagentes por épica, comandos Make.
- `skills/` — 31 skills `agrosat-*`; las que gobiernan el artículo son `agrosat-protocolo-articulo` y `agrosat-paper-micai`.
- `agents/` — 10 subagentes (Task tool), incluido `qa-reviewer` para la Fase 4.
- `settings.json` — permisos (engram, context7, graphify, make, git de solo lectura) y el hook `SessionStart` que corre `scripts/harness_status.py`.

> **Regla de oro**: cifras reales o nada, con su régimen nombrado; cada número impreso se rederiva desde una fila sellada del ledger. **No hay H100**: todo el protocolo corre en CPU; el harness no enruta a Azure.
