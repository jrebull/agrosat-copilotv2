"""Spatial-RAG *lite* tests (US-046 AC-8: hybrid fusion + corpus ingest).

These tests exercise :func:`ml.agent.rag.spatial_rag` and
:func:`ml.agent.rag.ingest_rag_documents` against a scripted asyncpg double (the
real PostGIS is mocked -- no live DB, no pgvector server). They assert:

- the two-stage pipeline (``ST_DWithin`` candidates -> pgvector cosine) feeds a
  weighted fusion that ranks documents by descending fused score, using rows with
  *known* distances and cosine distances so the resulting order is deterministic;
- the pipeline degrades to spatial-only ranking when no candidate carries an
  embedding (the nearest stays first);
- ``spatial_rag`` returns ``[]`` when no document lies within the radius;
- ``ingest_rag_documents`` inserts N rows in a single ``executemany`` batch and
  renders the AlphaEarth vector as a pgvector literal.

Reuses :func:`make_ctx` from the US-045 conftest for the :class:`ToolContext`.
"""

from __future__ import annotations

from typing import Any

import ml.agent.rag as rag_mod
from ml.agent.rag import RAGDocument, ingest_rag_documents, spatial_rag

from .conftest import SESSION_A, FakeRecord, fake_session_scoped_conn

_AOI = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
}


def _vec(value: float) -> str:
    """Render a constant 64-dim AlphaEarth vector as a pgvector text literal."""
    return "[" + ",".join(repr(value) for _ in range(64)) + "]"


class _ScriptedConn:
    """asyncpg ``Connection`` double returning a different row list per ``fetch``.

    ``spatial_rag`` issues two ``fetch`` calls in order: (1) the ST_DWithin
    candidate query, (2) the pgvector cosine query. This double pops the scripted
    result for each call and records the SQL so tests can assert which stage ran.
    """

    def __init__(self, fetch_results: list[list[FakeRecord]]) -> None:
        self._fetch_results = list(fetch_results)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return "SELECT 1"

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        self.calls.append((sql, args))
        if self._fetch_results:
            return self._fetch_results.pop(0)
        return []


class _CapturingConn:
    """Double for the ingest path: captures the ``executemany`` batch."""

    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, list[tuple]]] = []

    async def executemany(self, sql: str, rows: list[tuple]) -> None:
        self.executemany_calls.append((sql, list(rows)))


# ---------------------------------------------------------------------------
# spatial_rag: hybrid fusion ranking
# ---------------------------------------------------------------------------
async def test_spatial_rag_ranks_by_weighted_fusion(monkeypatch, make_ctx) -> None:
    """Documents are ordered by the fused (spatial + semantic) score, descending.

    Three candidates with hand-picked distances and cosine distances are scripted.
    With ``spatial_weight=0.5`` and ``radius_m=1000``:

      doc A: dist=0,    cos=0.9 -> spatial=1.0,   sem=0.1  -> 0.55
      doc B: dist=100,  cos=0.0 -> spatial=0.909, sem=1.0  -> 0.954
      doc C: dist=900,  cos=0.5 -> spatial=0.526, sem=0.5  -> 0.513

    Expected order by score: B (0.954) > A (0.55) > C (0.513).
    """
    candidates = [
        FakeRecord(
            id=1,
            content="doc A",
            source="phenology_caption",
            parcel_id="10000_1",
            embedding=_vec(0.1),
            distance_m=0.0,
        ),
        FakeRecord(
            id=2,
            content="doc B",
            source="phenology_caption",
            parcel_id="10000_2",
            embedding=_vec(0.2),
            distance_m=100.0,
        ),
        FakeRecord(
            id=3,
            content="doc C",
            source="phenology_caption",
            parcel_id="10000_3",
            embedding=_vec(0.3),
            distance_m=900.0,
        ),
    ]
    cosine_rows = [
        FakeRecord(id=1, cosine_distance=0.9),
        FakeRecord(id=2, cosine_distance=0.0),
        FakeRecord(id=3, cosine_distance=0.5),
    ]
    conn = _ScriptedConn([candidates, cosine_rows])
    monkeypatch.setattr(rag_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    docs = await spatial_rag(
        make_ctx(), query="trigo", aoi=_AOI, top_k=5, spatial_weight=0.5, radius_m=1000.0
    )

    assert [d.id for d in docs] == [2, 1, 3]
    assert all(isinstance(d, RAGDocument) for d in docs)
    # Scores are strictly descending (the ranking invariant).
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True)
    # The RLS hook was primed before any data query.
    assert "set_config" in conn.calls[0][0]
    # Both pipeline stages ran (ST_DWithin then the cosine <=> scan).
    assert any("ST_DWithin" in c[0] for c in conn.calls)
    assert any("<=>" in c[0] for c in conn.calls)


async def test_spatial_rag_respects_top_k(monkeypatch, make_ctx) -> None:
    """Only ``top_k`` documents are returned even with more candidates."""
    candidates = [
        FakeRecord(
            id=i,
            content=f"doc {i}",
            source="phenology_caption",
            parcel_id=f"10000_{i}",
            embedding=_vec(0.1 * i),
            distance_m=float(i * 10),
        )
        for i in range(1, 6)
    ]
    cosine_rows = [FakeRecord(id=i, cosine_distance=0.1 * i) for i in range(1, 6)]
    conn = _ScriptedConn([candidates, cosine_rows])
    monkeypatch.setattr(rag_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    docs = await spatial_rag(make_ctx(), query="q", aoi=_AOI, top_k=2)

    assert len(docs) == 2


async def test_spatial_rag_degrades_to_spatial_only(monkeypatch, make_ctx) -> None:
    """With no usable embedding, ranking is spatial-only (nearest first).

    The candidates carry no embedding, so the semantic stage is skipped and the
    fused score is the spatial term alone -> the closest document ranks first.
    """
    candidates = [
        FakeRecord(
            id=10,
            content="near",
            source="phenology_caption",
            parcel_id="10000_1",
            embedding=None,
            distance_m=50.0,
        ),
        FakeRecord(
            id=11,
            content="far",
            source="phenology_caption",
            parcel_id="10000_2",
            embedding=None,
            distance_m=800.0,
        ),
    ]
    # No second fetch should be needed; script an empty cosine result as backstop.
    conn = _ScriptedConn([candidates, []])
    monkeypatch.setattr(rag_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    docs = await spatial_rag(make_ctx(), query="q", aoi=_AOI, radius_m=1000.0)

    assert [d.id for d in docs] == [10, 11]  # nearest first
    # The cosine scan must NOT have run (no embedding to query with).
    assert not any("<=>" in c[0] for c in conn.calls)


async def test_spatial_rag_empty_when_no_candidates(monkeypatch, make_ctx) -> None:
    """No document within the radius -> empty list, no cosine query issued."""
    conn = _ScriptedConn([[]])
    monkeypatch.setattr(rag_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    docs = await spatial_rag(make_ctx(), query="q", aoi=_AOI)

    assert docs == []
    assert not any("<=>" in c[0] for c in conn.calls)


# ---------------------------------------------------------------------------
# ingest_rag_documents: batch insert
# ---------------------------------------------------------------------------
async def test_ingest_inserts_n_rows() -> None:
    """``ingest_rag_documents`` returns N and issues one ``executemany`` batch."""
    conn = _CapturingConn()
    documents = [
        {
            "parcel_id": f"10000_{i}",
            "content": f"descripcion fenologica {i}",
            "source": "phenology_caption",
            "embedding": [0.1 * i] * 64,
            "geom_geojson": '{"type":"Point","coordinates":[0,0]}',
            "geom_srid": 2154,
        }
        for i in range(3)
    ]

    inserted = await ingest_rag_documents(conn, documents)

    assert inserted == 3
    assert len(conn.executemany_calls) == 1
    sql, rows = conn.executemany_calls[0]
    assert "INSERT INTO rag_documents" in sql
    assert len(rows) == 3
    # The embedding is rendered as a pgvector text literal (positional arg 3).
    first_embedding_literal = rows[0][3]
    assert isinstance(first_embedding_literal, str)
    assert first_embedding_literal.startswith("[") and first_embedding_literal.endswith("]")
    # content/source/parcel_id bound positionally.
    assert rows[0][1] == "descripcion fenologica 0"
    assert rows[0][2] == "phenology_caption"
    assert rows[0][0] == "10000_0"


async def test_ingest_empty_is_noop() -> None:
    """An empty document list inserts nothing and issues no SQL."""
    conn = _CapturingConn()
    assert await ingest_rag_documents(conn, []) == 0
    assert conn.executemany_calls == []


async def test_ingest_null_embedding_is_kept_none() -> None:
    """A document without an embedding binds ``None`` (not a literal)."""
    conn = _CapturingConn()
    documents = [
        {
            "parcel_id": None,
            "content": "scene metadata",
            "source": "scene_meta",
            "geom_wkt": "POINT(0 0)",
        }
    ]

    inserted = await ingest_rag_documents(conn, documents)

    assert inserted == 1
    _sql, rows = conn.executemany_calls[0]
    assert rows[0][3] is None  # embedding literal absent
    assert rows[0][6] == "POINT(0 0)"  # geom_wkt fallback bound


async def test_spatial_rag_clamps_opposite_hemisphere_cosine(monkeypatch, make_ctx) -> None:
    """A cosine distance > 1 (opposite-hemisphere embeddings) keeps score >= 0.

    pgvector ``<=>`` ranges over [0, 2]; without clamping ``1 - cosine_distance``
    goes negative and the fused score can fall below the documented [0, 1] range.
    With ``spatial_weight=0.1`` and a far candidate, an unclamped score would be
    negative; the fix clamps the semantic term to 0.
    """
    candidates = [
        FakeRecord(
            id=1,
            content="opp",
            source="phenology_caption",
            parcel_id="10000_1",
            embedding=_vec(0.1),
            distance_m=900.0,
        ),
    ]
    cosine_rows = [FakeRecord(id=1, cosine_distance=1.9)]
    conn = _ScriptedConn([candidates, cosine_rows])
    monkeypatch.setattr(rag_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    docs = await spatial_rag(
        make_ctx(), query="q", aoi=_AOI, top_k=5, spatial_weight=0.1, radius_m=1000.0
    )

    assert docs, "candidate should still be returned"
    assert all(d.score >= 0.0 for d in docs), "fused score must stay >= 0 after clamp"


def test_to_pgvector_literal_sanitizes_non_finite() -> None:
    """NaN/Inf/-Inf embedding components become 0.0 (pgvector rejects them)."""
    literal = rag_mod._to_pgvector_literal([1.5, float("nan"), float("inf"), float("-inf"), -2.0])
    assert "nan" not in literal and "inf" not in literal
    assert literal == "[1.5,0.0,0.0,0.0,-2.0]"


def test_session_a_is_the_test_tenant() -> None:
    """Sanity: the shared fixture session id is a stable UUID."""
    assert str(SESSION_A) == "11111111-1111-1111-1111-111111111111"
