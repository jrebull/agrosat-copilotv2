"""``retrieve_context`` deferred-tool tests (US-046 AC-5, AC-9, AC-10).

Covers the graceful-degradation contract of the Spatial-RAG *lite* tool:

- with ``rag_enabled`` off (the default), ``run`` returns an empty
  :class:`RetrievedContext` (``rag_enabled=False``, no grounding) WITHOUT touching
  the database -- proven by passing ``pool=None`` and asserting ``spatial_rag`` is
  never invoked;
- with ``rag_enabled`` on, ``run`` calls the RAG pipeline and packs the retrieved
  documents into a citation-tagged ``grounding_text`` block;
- the tool is registered as ``deferred=True`` (background / ``NON_BLOCKING``) and
  appears among the deferred tools, never the synchronous ones.

The RAG pipeline (:func:`ml.agent.rag.spatial_rag`) is mocked so no DB / pgvector
is touched. Reuses :func:`make_ctx` from the US-045 conftest.
"""

from __future__ import annotations

import ml.agent.tools.retrieve_context as retrieve_mod
from ml.agent.rag import RAGDocument
from ml.agent.tools import TOOL_REGISTRY, get_deferred_tools, get_sync_tools
from ml.agent.tools.retrieve_context import (
    RetrieveContextInput,
    RetrievedContext,
    run,
)

from .conftest import SESSION_A

_AOI = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
}


class _SettingsRagOn:
    """Settings stub with the RAG feature flag enabled."""

    rag_enabled = True


class _SettingsRagOff:
    """Settings stub with the RAG feature flag explicitly disabled."""

    rag_enabled = False


def _input() -> RetrieveContextInput:
    return RetrieveContextInput(
        session_id=SESSION_A, query="que cultivos hay cerca", aoi=_AOI, top_k=3
    )


# ---------------------------------------------------------------------------
# AC-10: disabled -> empty result, no DB access
# ---------------------------------------------------------------------------
async def test_disabled_returns_empty_without_db(monkeypatch, make_ctx) -> None:
    """``rag_enabled=False`` -> empty context and ``spatial_rag`` never called.

    The context's ``pool`` is ``None`` (the conftest default), so any DB access
    would raise; the test instead asserts the RAG pipeline is never entered.
    """
    called = {"spatial_rag": False}

    async def _must_not_run(*args, **kwargs):
        called["spatial_rag"] = True
        raise AssertionError("spatial_rag must not run when rag_enabled is False")

    monkeypatch.setattr(retrieve_mod, "spatial_rag", _must_not_run)

    ctx = make_ctx()
    ctx.settings = _SettingsRagOff()  # type: ignore[attr-defined]
    out = await run(_input(), ctx)

    assert isinstance(out, RetrievedContext)
    assert out.rag_enabled is False
    assert out.documents == []
    assert out.grounding_text == ""
    assert called["spatial_rag"] is False


async def test_disabled_by_default_when_flag_absent(monkeypatch, make_ctx) -> None:
    """A Settings object lacking ``rag_enabled`` is treated as disabled (no DB)."""

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("spatial_rag must not run without the flag")

    monkeypatch.setattr(retrieve_mod, "spatial_rag", _must_not_run)

    # The conftest FakeSettings has no ``rag_enabled`` attribute -> getattr default.
    out = await run(_input(), make_ctx())

    assert out.rag_enabled is False
    assert out.documents == []


# ---------------------------------------------------------------------------
# AC-9: enabled -> RAG runs, grounding injected
# ---------------------------------------------------------------------------
async def test_enabled_injects_grounding(monkeypatch, make_ctx) -> None:
    """``rag_enabled=True`` -> documents retrieved and grounding text built."""
    retrieved = [
        RAGDocument(
            id=1,
            content="Trigo en fase de senescencia.",
            source="phenology_caption",
            parcel_id="10000_1",
            distance_m=120.0,
            score=0.92,
        ),
        RAGDocument(
            id=2,
            content="Maiz con pico NDVI tardio.",
            source="phenology_caption",
            parcel_id="10000_2",
            distance_m=300.0,
            score=0.81,
        ),
    ]

    captured = {}

    async def _fake_spatial_rag(ctx, *, query, aoi, top_k):
        captured["query"] = query
        captured["aoi"] = aoi
        captured["top_k"] = top_k
        return retrieved

    monkeypatch.setattr(retrieve_mod, "spatial_rag", _fake_spatial_rag)

    ctx = make_ctx()
    ctx.settings = _SettingsRagOn()  # type: ignore[attr-defined]
    out = await run(_input(), ctx)

    assert out.rag_enabled is True
    assert [d.id for d in out.documents] == [1, 2]
    # The grounding block cites every retrieved document by source + parcel id.
    assert out.grounding_text
    assert "Trigo en fase de senescencia." in out.grounding_text
    assert "[phenology_caption:10000_1]" in out.grounding_text
    assert "[phenology_caption:10000_2]" in out.grounding_text
    # The tool forwarded the AOI and top_k to the pipeline.
    assert captured["top_k"] == 3
    assert captured["aoi"]["type"] == "Polygon"


async def test_enabled_empty_corpus_returns_blank_grounding(monkeypatch, make_ctx) -> None:
    """RAG on but nothing nearby -> enabled flag stays True with empty grounding."""

    async def _empty(ctx, *, query, aoi, top_k):
        return []

    monkeypatch.setattr(retrieve_mod, "spatial_rag", _empty)

    ctx = make_ctx()
    ctx.settings = _SettingsRagOn()  # type: ignore[attr-defined]
    out = await run(_input(), ctx)

    assert out.rag_enabled is True
    assert out.documents == []
    assert out.grounding_text == ""


# ---------------------------------------------------------------------------
# Registered as a SYNCHRONOUS in-loop tool (a fast pgvector query); the agent
# exposes it only when ``rag_enabled`` (see test_agent), so the reasoner can ground
# itself in cited corpus evidence within the turn (anti-hallucination).
# ---------------------------------------------------------------------------
def test_retrieve_context_is_synchronous() -> None:
    """``retrieve_context`` is registered synchronous (in-loop / BLOCKING)."""
    spec = TOOL_REGISTRY["retrieve_context"]
    assert spec.deferred is False
    assert spec.input_model is RetrieveContextInput
    assert spec.output_model is RetrievedContext


def test_retrieve_context_in_sync_not_deferred_set() -> None:
    """The tool appears among synchronous tools and never among the deferred ones."""
    deferred_names = {spec.name for spec in get_deferred_tools()}
    sync_names = {spec.name for spec in get_sync_tools()}
    assert "retrieve_context" in sync_names
    assert "retrieve_context" not in deferred_names
