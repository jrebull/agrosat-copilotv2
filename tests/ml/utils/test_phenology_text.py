"""Tests de ``ml.utils.phenology_text`` (wrapper de materializacion full).

Gemini SIEMPRE se mockea via ``set_llm_client`` (R7 + regla dura del
proyecto: cero llamadas de red en CI). Los tests verifican:

- Bloqueo duro si faltan credenciales y no hay cliente inyectado.
- ``FileNotFoundError`` si el parquet de entrada no existe.
- Cache idempotente: si el output existe y ``overwrite=False`` no se
  llama al LLM.
- Estratificacion ``balanced_by_class`` respeta ``min_per_class``.
- Shape y esquema canonico del parquet generado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.features.phenology_description as pd_mod
from ml.features.phenology_description import _CREDENTIAL_ENV_VARS, set_llm_client
from ml.utils.phenology_text import (
    _stratified_sample,
    materialize_phenology_text,
)


@pytest.fixture(autouse=True)
def restore_llm_client():
    """Aisla mocks: resetea el cliente LLM tras cada test."""
    yield
    set_llm_client(None)


@pytest.fixture
def parcels_parquet(tmp_path: Path) -> Path:
    """Parquet de features minimo con N parcelas y 3 clases."""
    n = 12
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "parcel_id": [f"p{i:03d}" for i in range(n)],
            "year": [2019] * n,
            "class_id": [i % 3 for i in range(n)],
            "NDVI_t_00": rng.uniform(0.1, 0.9, n).tolist(),
            "NDVI_t_01": rng.uniform(0.1, 0.9, n).tolist(),
            "NDVI_t_02": rng.uniform(0.1, 0.9, n).tolist(),
        }
    )
    path = tmp_path / "parcels_features.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def mock_encoder(monkeypatch: pytest.MonkeyPatch):
    """Mock global de ``encode_descriptions`` para no descargar st-models."""
    def fake_encode(
        descriptions, *, encoder="sentence-transformers", model_name=None
    ):
        return np.ones((len(descriptions), 8), dtype=np.float32)

    monkeypatch.setattr(pd_mod, "encode_descriptions", fake_encode)
    return fake_encode


def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_materialize_phenology_text_raises_when_no_input(
    tmp_path: Path,
) -> None:
    """Path de entrada inexistente -> FileNotFoundError."""
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        materialize_phenology_text(
            missing,
            output_path=tmp_path / "out.parquet",
            enforce_api_key=False,
        )


def test_materialize_phenology_text_raises_without_credentials(
    parcels_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin env vars y sin cliente inyectado -> RuntimeError."""
    _clear_credentials(monkeypatch)
    set_llm_client(None)
    with pytest.raises(RuntimeError, match="Gemini is not configured"):
        materialize_phenology_text(
            parcels_parquet,
            output_path=tmp_path / "out.parquet",
            enforce_api_key=True,
        )


def test_materialize_phenology_text_caches_existing_parquet(
    parcels_parquet: Path,
    tmp_path: Path,
) -> None:
    """Si el output ya existe y overwrite=False, no llama al LLM."""
    output_path = tmp_path / "phenology_text.parquet"
    cached = pl.DataFrame(
        {
            "parcel_id": ["p000"],
            "year": [2019],
            "pheno_text_000": [0.1],
        }
    )
    cached.write_parquet(output_path)

    n_calls = {"count": 0}

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        n_calls["count"] += 1
        return "no deberia llamarse"

    set_llm_client(mock_client)

    result = materialize_phenology_text(
        parcels_parquet,
        output_path=output_path,
        overwrite=False,
        enforce_api_key=False,
    )
    assert result == output_path
    assert n_calls["count"] == 0


def test_materialize_phenology_text_balanced_sampling(
    parcels_parquet: Path,
    tmp_path: Path,
    mock_encoder,
) -> None:
    """Con ``balanced_by_class`` toma min_per_class por clase."""
    set_llm_client(lambda prompt, *, model, temperature: "desc sintetica")

    output_path = tmp_path / "phenology_text.parquet"
    result = materialize_phenology_text(
        parcels_parquet,
        output_path=output_path,
        balanced_by_class=True,
        min_per_class=2,
        overwrite=True,
        enforce_api_key=False,
    )
    assert result.exists()
    out = pl.read_parquet(result)
    # 3 clases * 2 por clase = 6 (fuente tiene 4 por clase).
    assert out.height == 6
    assert out.schema["parcel_id"] == pl.Utf8
    text_cols = [c for c in out.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == 8


def test_materialize_phenology_text_runs_with_injected_client(
    parcels_parquet: Path,
    tmp_path: Path,
    mock_encoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con cliente mock no exige env vars y persiste parquet con esquema canonico."""
    _clear_credentials(monkeypatch)
    set_llm_client(lambda prompt, *, model, temperature: "desc")

    output_path = tmp_path / "phenology_text.parquet"
    result = materialize_phenology_text(
        parcels_parquet,
        output_path=output_path,
        balanced_by_class=False,
        overwrite=True,
        enforce_api_key=False,
    )
    out = pl.read_parquet(result)
    assert out.height == 12  # todas las filas (sin balance ni cap).
    assert out.schema["parcel_id"] == pl.Utf8


def test_materialize_phenology_text_respects_max_parcels(
    parcels_parquet: Path,
    tmp_path: Path,
    mock_encoder,
) -> None:
    """``max_parcels`` recorta despues del muestreo balanceado."""
    set_llm_client(lambda prompt, *, model, temperature: "desc")
    output_path = tmp_path / "phenology_text.parquet"
    result = materialize_phenology_text(
        parcels_parquet,
        output_path=output_path,
        balanced_by_class=True,
        min_per_class=10,
        max_parcels=4,
        overwrite=True,
        enforce_api_key=False,
    )
    out = pl.read_parquet(result)
    assert out.height == 4


def test_stratified_sample_keeps_all_if_class_smaller(
    parcels_parquet: Path,
) -> None:
    """Clases con menos filas que min_per_class deben mantenerse intactas."""
    df = pl.read_parquet(parcels_parquet)
    sampled = _stratified_sample(df, class_col="class_id", min_per_class=100, seed=0)
    assert sampled.height == df.height


def test_materialize_phenology_text_balanced_missing_class_col_raises(
    parcels_parquet: Path,
    tmp_path: Path,
) -> None:
    """balanced_by_class con columna inexistente -> KeyError explicito."""
    set_llm_client(lambda prompt, *, model, temperature: "desc")
    with pytest.raises(KeyError, match="missing_class"):
        materialize_phenology_text(
            parcels_parquet,
            output_path=tmp_path / "out.parquet",
            balanced_by_class=True,
            class_col="missing_class",
            enforce_api_key=False,
        )
