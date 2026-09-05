# AgroSatCopilot — pointer

Claude Code carga `../CLAUDE.md` y otros agentes de código (Codex, Cursor, etc.) cargan `../AGENTS.md`. Ambos archivos son **espejos idénticos** del orquestador único — modificar uno requiere sincronizar el otro.

**Estructura canónica de orquestación**:

- [`../AGENTS.md`](../AGENTS.md) y [`../CLAUDE.md`](../CLAUDE.md) — orquestador único (213 líneas, mismo contenido): identidad, equipo, decisiones irrevocables, calendario, presupuesto, reglas globales, quality gates, anti-patrones, estilo respuesta, checklist US.
- [`../context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](../context/RefinamientoPlaneacionAgroSatCopilot_v6.md) — plan SCRUM completo US-001 a US-056 (EPICs, SP, calendario §10.3, riesgos, métricas).
- [`../docs/orchestration/`](../docs/orchestration/) — catálogo skills, auto-invoke table, mapa skill↔subagente, comandos Make.
- `skills/` — 30 skills `agrosat-*`.
- `agents/` — 9 subagentes profundos (Task tool).
- `settings.json` — configuración Claude Code (plugins).

> **Quality gates sin pre-commit**: el proyecto no usa `.pre-commit-config.yaml`. Las garantías (ruff, gitleaks, i18n-check) viven en `make check` y en GitHub Actions. Notebooks NO se strippean — se commitean con outputs poblados (entregable visual del curso). Reproducibilidad via `make notebooks-check` (papermill). Ver §"Quality Gates" en `../AGENTS.md`.
