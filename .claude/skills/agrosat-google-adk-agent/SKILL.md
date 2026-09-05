---
name: agrosat-google-adk-agent
description: Build the Google ADK Plan-and-React conversational agent for AgroSatCopilot with 9 geospatial FunctionTools, dual LLM backend (Gemini 2.5 Pro cloud plus Qwen3-30B-A3B-Instruct-2507 GPTQ-Int4 on-prem), session memory in Postgres, SSE streaming, and built-in ADK tracing. Use when building or modifying the conversational agent layer.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Google ADK Agent Skill

## Rules — NON-NEGOTIABLE

- Cada tool es `FunctionTool` con schema Pydantic input/output validado
- Backend abstraído via `LLMBackend` (Gemini ↔ vLLM OpenAI-compat)
- Session memory en Postgres con `SessionService`
- Streaming SSE con eventos: `plan_created`, `tool_call`, `tool_result`, `final_answer`
- Tracing built-in ADK habilitado (no observabilidad custom)
- Citaciones obligatorias en final_answer (scene_id, fechas, tool outputs)
- Latencias: p50 <2s simple, p95 <5s simple, p95 <15s multi-step

## Estructura del Agente

```python
# ml/agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from ml.agent.tools import (
    alphaearth_query, sentinel_search, rasterio_tool,
    geopandas_intersect, ndvi_calculator, timeseries_extractor,
    phenology_descriptor, dinov3_extract, crop_classifier_tool,
)
from ml.agent.backends import GeminiBackend, VLLMOpenAIBackend
from ml.agent.rag import SpatialRAG

INSTRUCTION_IT = """
Sei un agronomo esperto. Rispondi in italiano usando i tool.
Cita sempre lo scene_id e le date. Non inventare numeri.
"""

INSTRUCTION_ES = """
Eres un agrónomo experto. Responde en español usando los tools.
Cita siempre el scene_id y las fechas. No inventes números.
"""

INSTRUCTION_EN = "..."

def build_agent(llm_variant: str = "gemini", language: str = "it"):
    backend = {"gemini": GeminiBackend, "qwen35": VLLMOpenAIBackend}[llm_variant]()
    instruction = {"it": INSTRUCTION_IT, "es": INSTRUCTION_ES, "en": INSTRUCTION_EN}[language]
    return Agent(
        model=backend.model_name(),
        name="agrosat_agent",
        instruction=instruction,
        tools=[
            FunctionTool(alphaearth_query),
            FunctionTool(sentinel_search),
            FunctionTool(rasterio_tool),
            FunctionTool(geopandas_intersect),
            FunctionTool(ndvi_calculator),
            FunctionTool(timeseries_extractor),
            FunctionTool(phenology_descriptor),
            FunctionTool(dinov3_extract),
            FunctionTool(crop_classifier_tool),
        ],
    )
```

## LLM Backends

```python
# ml/agent/backends.py
from typing import Protocol
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

class LLMBackend(Protocol):
    def model_name(self) -> str: ...

class GeminiBackend(LLMBackend):
    def model_name(self) -> str:
        return "gemini-3.1-pro"

class VLLMOpenAIBackend(LLMBackend):
    def model_name(self) -> str:
        return LiteLlm(
            model="openai/agrosat-qwen35",
            base_url="http://vllm-qwen35.internal:8000/v1",
            api_key="not-needed",
        )
```

## Tool Pattern

```python
# ml/agent/tools/ndvi_calculator.py
from pydantic import BaseModel
from typing import Any
import structlog

logger = structlog.get_logger()

class NDVIInput(BaseModel):
    scene_id: str
    aoi: dict  # GeoJSON Feature

class NDVIResult(BaseModel):
    mean: float
    median: float
    p05: float
    p95: float
    pixel_count: int
    scene_id: str
    date: str

async def ndvi_calculator(scene_id: str, aoi: dict) -> dict:
    """Compute NDVI statistics for AOI from given Sentinel-2 scene.

    Args:
        scene_id: Sentinel-2 scene identifier (e.g., 'S2A_T32TQM_20250715').
        aoi: GeoJSON Feature with Polygon geometry.

    Returns:
        Dict with mean, median, percentiles, pixel_count, scene_id, date.
    """
    inp = NDVIInput(scene_id=scene_id, aoi=aoi)
    logger.info("tool_call_started", tool="ndvi_calculator", scene_id=scene_id)
    # ... actual computation reading B04/B08 from gs://agrosat-data/raw/s2/...
    result = NDVIResult(...)
    logger.info("tool_call_finished", tool="ndvi_calculator", duration_ms=...)
    return result.model_dump()
```

## Session Memory (Postgres)

```python
# ml/agent/memory.py
from google.adk.sessions import DatabaseSessionService

session_service = DatabaseSessionService(
    db_url=settings.DATABASE_URL,
    schema="agent",
)
```

## Streaming SSE (consumido por backend/app/api/chat.py)

```python
# ml/agent/agent.py (continued)
async def run_async(self, query: str, session: str, context: dict):
    async for event in self.runner.run_async(
        user_id=context["user_id"],
        session_id=session,
        new_message=query,
        run_config={"stream": True},
    ):
        yield event  # plan_created, tool_call, tool_result, final_answer
```

## Deploy a Vertex AI Agent Engine

```python
from vertexai.preview.reasoning_engines import ReasoningEngine

remote_agent = ReasoningEngine.create(
    agent=agrosat_agent,
    requirements=["google-adk", "google-cloud-aiplatform[reasoning_engines]"],
    display_name="agrosat-copilot",
)
```

## QA Checklist Agent

- [ ] 9 tools con FunctionTool + Pydantic schemas
- [ ] LLMBackend abstracción (Gemini + vLLM)
- [ ] Session memory en Postgres
- [ ] SSE con 4 eventos
- [ ] Tracing ADK habilitado
- [ ] Citaciones en final_answer
- [ ] Tests por tool con fixtures
- [ ] Latencias dentro de objetivo
- [ ] System prompt en it/es/en
