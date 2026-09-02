"""Tests de la variante por-parcela de fenologia (US-033, AC-7).

Liga la variante por-parcela a US-033: descripcion individual segun la curva
NDVI real de cada parcela, cacheada por ``parcel_id``, **una fila por
``parcel_id``** con columnas ``pheno_text_000..pheno_text_383``. El motor es
``ml.features.phenology_description.build_phenology_text_block`` (consumido por
``ml.utils.phenology_text.materialize_phenology_text``).

Reglas duras (R7 + R-KEY): Gemini se mockea SIEMPRE via ``set_llm_client``
(cero red); el encoder MiniLM se monkeypatchea para no descargar
``sentence-transformers`` en CI. El test del parquet REAL
``phenology_text_pastis.parquet`` lleva ``@pytest.mark.skipif`` (el binario no
esta en git; requiere ``dvc pull``).

Golden values (verificados en disco, recon US-033):
``phenology_text_pastis.parquet`` ``(1080, 386)``, 1080 ``parcel_id`` Utf8
unicos, 384 columnas ``pheno_text_*``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.features.phenology_description as pd_mod
from ml.features.phenology_description import (
    build_phenology_text_block,
    set_llm_client,
)

#: Path del parquet REAL de la variante por-parcela (no esta en git).
_REAL_PARQUET = Path("data/features/phenology_text_pastis.parquet")

#: Dimension del embedding mockeado (no descarga MiniLM real).
_MOCK_EMB_DIM = 8


@pytest.fixture(autouse=True)
def restore_client():
    """Resetea el cliente LLM tras cada test (aisla mocks)."""
    yield
    set_llm_client(None)


@pytest.fixture
def mock_encoder(monkeypatch: pytest.MonkeyPatch):
    """Mock global de ``encode_descriptions`` (no descarga st-models).

    Devuelve un embedding deterministico distinto por texto, para que filas
    con descripciones distintas tengan embeddings distintos.
    """

    def fake_encode(
        descriptions,
        *,
        encoder: str = "sentence-transformers",
        model_name: str | None = None,
    ) -> np.ndarray:
        rows = []
        for desc in descriptions:
            seed = abs(hash(desc)) % (2**32)
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(_MOCK_EMB_DIM).astype(np.float32))
        return np.stack(rows).astype(np.float32)

    monkeypatch.setattr(pd_mod, "encode_descriptions", fake_encode)
    return fake_encode


def _ndvi_df(parcel_ids: list[str]) -> pl.DataFrame:
    """DataFrame minimo con curva NDVI discreta por parcela (parcel_id Utf8)."""
    n = len(parcel_ids)
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "parcel_id": parcel_ids,
            "year": [2019] * n,
            "NDVI_t_00": rng.uniform(0.1, 0.5, n).tolist(),
            "NDVI_t_01": rng.uniform(0.5, 0.9, n).tolist(),
            "NDVI_t_02": rng.uniform(0.2, 0.6, n).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# 6.4 Variante por-parcela (AC-7) — sintetico, mock.
# ---------------------------------------------------------------------------


def test_per_parcel_one_row_per_parcel_id(
    tmp_path: Path,
    mock_encoder,
) -> None:
    """``build_phenology_text_block`` -> 1 fila por ``parcel_id`` distinto."""
    parcel_ids = ["p1", "p2", "p3"]

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        return f"Descripcion para prompt {hash(prompt) % 997}."

    set_llm_client(mock_client)
    df = _ndvi_df(parcel_ids)
    block = build_phenology_text_block(df, skip_llm=False, cache_dir=tmp_path)

    assert block.height == len(parcel_ids)
    assert block["parcel_id"].to_list() == parcel_ids
    assert block["parcel_id"].n_unique() == len(parcel_ids)
    assert block.schema["parcel_id"] == pl.Utf8
    text_cols = [c for c in block.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == _MOCK_EMB_DIM


def test_per_parcel_cached_by_parcel_id(
    tmp_path: Path,
    mock_encoder,
) -> None:
    """Misma curva + mismo ``parcel_id`` -> cache-hit (no re-llama al LLM)."""
    call_count = {"count": 0}

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        call_count["count"] += 1
        return "Descripcion cacheable por parcela."

    set_llm_client(mock_client)
    df = _ndvi_df(["p1", "p2"])

    build_phenology_text_block(df, skip_llm=False, cache_dir=tmp_path)
    calls_first = call_count["count"]
    assert calls_first == 2  # 2 parcelas distintas -> 2 cache-miss.

    # 2.a corrida: misma curva + mismo parcel_id -> cache-hit, sin LLM nuevo.
    build_phenology_text_block(df, skip_llm=False, cache_dir=tmp_path)
    assert call_count["count"] == calls_first


def test_per_parcel_distinct_descriptions_distinct_embeddings(
    tmp_path: Path,
    mock_encoder,
) -> None:
    """Parcelas con curvas distintas producen embeddings distintos."""

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        # Texto dependiente del prompt (cada curva -> prompt -> texto distinto).
        return f"Descripcion {hash(prompt) % 997}."

    set_llm_client(mock_client)
    df = _ndvi_df(["p1", "p2", "p3"])
    block = build_phenology_text_block(df, skip_llm=False, cache_dir=tmp_path)

    emb = block.select([c for c in block.columns if c.startswith("pheno_text_")]).to_numpy()
    # Al menos dos filas deben diferir (no todas iguales).
    assert not np.allclose(emb[0], emb[1])


# ---------------------------------------------------------------------------
# Parquet REAL de la variante por-parcela (skipif si no existe).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="parquet real no presente (requiere dvc pull)",
)
def test_real_phenology_text_parquet() -> None:
    """El parquet REAL por-parcela: (1080, 386), parcel_id Utf8 unico."""
    df = pl.read_parquet(_REAL_PARQUET)
    assert df.shape == (1080, 386)
    assert df.schema["parcel_id"] == pl.Utf8
    assert df["parcel_id"].n_unique() == 1080
    text_cols = [c for c in df.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == 384
