# Agent Sub-Agent — AgroSatCopilot (`ml/agent/`)

> Sobreescribe a [`ml/CLAUDE.md`](../CLAUDE.md) y al root cuando trabajes el agente conversacional. NON-NEGOTIABLEs globales (idioma, secrets, session_id, sin emojis) viven en [`../../CLAUDE.md`](../../CLAUDE.md) — no se repiten aquí.

**Rol planeado**: agente Plan-and-React con tools geoespaciales (FunctionTool + Pydantic), Spatial-RAG híbrido PostGIS + pgvector, backend LLM dual (Gemini 2.5 Pro / Qwen3-30B-A3B-Instruct-2507 GPTQ-Int4 con vLLM), streaming SSE y memoria por `session_id`.

## Estado

ESQUELETO PURO. Solo existen `__init__.py` y `tools/__init__.py` (vacíos). NO existe todavía: `agent.py`, `backends.py`, `rag.py`, `memory.py`, ningún archivo de tool, ni el directorio `eval/`. Toda la sección de Convenciones es **diseño objetivo**, no código presente. `google-adk` fue removido del lock (chocaba con `genai` 2.x, requiere `<2`).

## Comandos

```bash
# AMBOS FALLAN HOY: referencian ml/agent/eval/ que aun no existe.
# Crear ml/agent/eval/eval_agromind.py y eval_geoanalyst.py antes de usar.
make eval-agromind variant=gemini      # -> ml/agent/eval/eval_agromind.py
make eval-geoanalyst variant=gemini    # -> ml/agent/eval/eval_geoanalyst.py
```

## Stack local (planeado)

- LLM del agente: **Qwen3-30B-A3B-Instruct-2507 GPTQ-Int4** (texto, vLLM on-prem H100) y Gemini 2.5 Pro (Vertex AI). NO Qwen3-VL (ese es VLM de fine-tune, otro scope).
- Tool de clasificación de cultivo: **baseline XGBoost + AlphaEarth** (EPIC 4). NO "Gemma 4 + ensamble".
- Embeddings RAG: e5-mistral-7b sobre pgvector; filtro espacial `ST_DWithin` en PostGIS.

## Convenciones (todas PLANEADAS, aun sin implementar)

- ✅ Cada tool = `FunctionTool` con schema Pydantic de input y output validado.
- ✅ Abstracción `LLMBackend` (`GeminiBackend`, `VLLMOpenAIBackend`); nunca hardcodear el backend.
- ✅ Streaming SSE con eventos `plan_created`, `tool_call`, `tool_result`, `final_answer`.
- ✅ Logging `structlog`: `tool_call_started` + `tool_call_finished` con `duration_ms`.
- ✅ Memoria de sesión persistida en PostgreSQL; toda query filtra por `session_id`.
- ✅ Citaciones obligatorias en `final_answer` (scene_id, fechas, tool calls rastreables).
- ❌ Tool sin schema Pydantic.
- ❌ Cifra (hectáreas, NDVI, fechas) sin origen en un tool call.
- ❌ Inferencia ML pesada inline en un tool — delegar a Pub/Sub worker.

## No tocar

- No re-agregar `google-adk` ad-hoc al lock: choca con `genai` 2.x. Coordinar resolución de deps con el equipo antes.
- No saltarse el filtro por `session_id` en ninguna query (multi-tenant).

## Tests

Ninguno todavía. Al crear lógica, ubicarlos en `tests/ml/agent/` (pytest + pytest-asyncio). Mockear GEE, PostGIS, pgvector, Vertex AI y vLLM — sin llamadas reales. Ver skill `agrosat-testing`.

## Skills

- [`agrosat-google-adk-agent`](../../.claude/skills/agrosat-google-adk-agent/SKILL.md) — planner, FunctionTool, routing LLM.
- [`agrosat-spatial-rag`](../../.claude/skills/agrosat-spatial-rag/SKILL.md) — híbrido PostGIS + pgvector.
- [`agrosat-llm-finetuning`](../../.claude/skills/agrosat-llm-finetuning/SKILL.md) — backend vLLM Qwen3-30B-A3B-Instruct-2507.
- [`agrosat-ml-evaluation`](../../.claude/skills/agrosat-ml-evaluation/SKILL.md) — AgroMind, GeoAnalystBench.
- [`agrosat-testing`](../../.claude/skills/agrosat-testing/SKILL.md) — mocks de tools.
