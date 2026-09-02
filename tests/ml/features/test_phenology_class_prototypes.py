"""Tests de ``ml.features.phenology_class_prototypes`` (US-033).

Endurece los gates de los 18 prototipos fenologicos por clase (curva NDVI
media -> descripcion ES via Gemini Flash -> embedding MiniLM 384). El nucleo
ya esta implementado y el parquet ya esta materializado + DVC-trackeado; estos
tests **validan** lo existente y cierran el gap de testing (tarea tecnica v8
linea 1218: "test de determinismo del cache").

Reglas duras (R7 + R-KEY de la US): Gemini se mockea SIEMPRE via
``set_llm_client`` (cero red, sin re-llamar a la API real); el encoder MiniLM
se monkeypatchea para no descargar ``sentence-transformers`` en CI; el escaneo
PASTIS-R se monkeypatchea con un dict de curvas sinteticas (no se lee
``data/PASTIS-R/``). Los tests del parquet REAL llevan ``@pytest.mark.skipif``
porque el binario no esta en git (requiere ``dvc pull``).

Golden values (verificados en disco, recon US-033):

- ``shape == (18, 388)``; ``class_id == [1..18]``.
- longitudes de descripcion ``[807, 586, 686, 870, 717, 733, 671, 646, 726,
  766, 636, 584, 651, 810, 799, 666, 723, 594]`` (min 584, todas > 50).
- ``abs(emb).sum()`` por fila ~ 15.5 (> 0); norma L2 ~ 1.0.
- variante por-parcela ``phenology_text_pastis.parquet`` ``(1080, 386)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.features.phenology_class_prototypes as proto_mod
import ml.features.phenology_description as pd_mod
from ml.features.phenology_class_prototypes import (
    _CROP_CLASS_IDS,
    _DEFAULT_OUTPUT,
    _EMB_DIM,
    _encode_descriptions,
    generate_class_prototypes,
    load_class_names,
    load_class_prototype_embeddings,
)
from ml.features.phenology_description import _hash_curve, set_llm_client

#: Numero de bins DOY usados por las curvas sinteticas de los tests.
_N_TIME_BINS = 37

#: Path del parquet REAL (no esta en git; requiere ``dvc pull``).
_REAL_PARQUET = _DEFAULT_OUTPUT

#: Golden lens verificadas en disco (recon US-033), una por clase 1..18.
_GOLDEN_DESC_LENS = [
    807,
    586,
    686,
    870,
    717,
    733,
    671,
    646,
    726,
    766,
    636,
    584,
    651,
    810,
    799,
    666,
    723,
    594,
]


@pytest.fixture(autouse=True)
def restore_client():
    """Resetea el cliente LLM tras cada test (aisla mocks)."""
    yield
    set_llm_client(None)


@pytest.fixture
def synthetic_curves() -> dict[int, np.ndarray]:
    """Curvas NDVI medias sinteticas, una por clase 1..18.

    Cada clase recibe una curva distinta (offset por ``class_id``) para que
    el hash de cache y el embedding sean distinguibles entre clases.

    Returns:
        ``{class_id: curve (37,)}`` con valores en ``[0, 1]``.
    """
    curves: dict[int, np.ndarray] = {}
    base = np.linspace(0.2, 0.8, _N_TIME_BINS, dtype=np.float64)
    for c in _CROP_CLASS_IDS:
        curves[c] = np.clip(base + 0.01 * c, 0.0, 1.0)
    return curves


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    curves: dict[int, np.ndarray],
    call_count: dict[str, int],
) -> None:
    """Instala los mocks de NDVI-scan, LLM y encoder (cero red, cero disco).

    Args:
        monkeypatch: Fixture de pytest para parchear el modulo.
        curves: Dict de curvas sinteticas a devolver en vez de escanear
            PASTIS-R.
        call_count: Contador mutable que cuenta invocaciones reales al LLM
            (cache-miss). Una 2.a corrida con cache poblada lo deja igual.
    """
    monkeypatch.setattr(
        proto_mod,
        "compute_class_mean_ndvi_curves",
        lambda *_a, **_k: curves,
    )

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        call_count["count"] += 1
        # Texto deterministico por curva (depende del contenido del prompt).
        return f"Descripcion fenologica deterministica para prompt {hash(prompt) % 997}."

    set_llm_client(mock_client)

    def mock_encode(descriptions) -> np.ndarray:
        # Embedding deterministico: hash estable por texto, L2-norm.
        rows = []
        for desc in descriptions:
            seed = abs(hash(desc)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(_EMB_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-12
            rows.append(vec)
        return np.stack(rows).astype(np.float32)

    monkeypatch.setattr(proto_mod, "_encode_descriptions", mock_encode)


# ---------------------------------------------------------------------------
# 6.1 Determinismo del cache SHA256 (gap #1, AC-4).
# ---------------------------------------------------------------------------


def test_generate_class_prototypes_deterministic_two_runs(
    synthetic_curves: dict[int, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dos corridas con mocks -> parquet bit-identico; 2.a corrida = cache-hit.

    Es el entregable central de US-033 (tarea tecnica v8 linea 1218). La
    cache SHA256 por ``parcel_id=f"class_{c}"`` garantiza que la 2.a corrida
    no vuelve a invocar al LLM (``call_count`` no incrementa) y que los bytes
    del parquet son identicos.
    """
    call_count = {"count": 0}
    _install_mocks(monkeypatch, synthetic_curves, call_count)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pd_mod, "default_cache_dir", lambda: cache_dir)

    out1 = tmp_path / "run1.parquet"
    out2 = tmp_path / "run2.parquet"

    generate_class_prototypes(output_path=out1)
    calls_after_run1 = call_count["count"]
    assert calls_after_run1 == len(_CROP_CLASS_IDS)  # 18 cache-miss la 1.a vez.

    generate_class_prototypes(output_path=out2)
    # 2.a corrida: misma curva + mismo parcel_id + mismo model -> cache-hit.
    assert call_count["count"] == calls_after_run1

    assert out1.read_bytes() == out2.read_bytes()


def test_hash_curve_stable(synthetic_curves: dict[int, np.ndarray]) -> None:
    """``_hash_curve`` es estable para el mismo input y sensible a cambios."""
    curve = synthetic_curves[1].astype(np.float32)
    h1 = _hash_curve("class_1", curve, "gemini-3.5-flash")
    h2 = _hash_curve("class_1", curve, "gemini-3.5-flash")
    assert h1 == h2
    assert len(h1) == 16
    # Sensible a parcel_id, curva y modelo.
    assert _hash_curve("class_2", curve, "gemini-3.5-flash") != h1
    assert _hash_curve("class_1", curve + 0.1, "gemini-3.5-flash") != h1
    assert _hash_curve("class_1", curve, "other-model") != h1


def test_generate_class_prototypes_schema(
    synthetic_curves: dict[int, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El parquet sintetico tiene 18 filas y el esquema canonico (18, 388)."""
    call_count = {"count": 0}
    _install_mocks(monkeypatch, synthetic_curves, call_count)
    monkeypatch.setattr(pd_mod, "default_cache_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache").mkdir()

    out = tmp_path / "proto.parquet"
    generate_class_prototypes(output_path=out)
    df = pl.read_parquet(out)
    assert df.height == len(_CROP_CLASS_IDS)
    assert df["class_id"].to_list() == list(_CROP_CLASS_IDS)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert len(emb_cols) == _EMB_DIM
    for col in ("class_id", "class_name", "ndvi_curve", "description"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# 6.3 Reuso de phenology_description + encoder MiniLM (AC-5/6).
# ---------------------------------------------------------------------------


def test_delegates_to_generate_phenology_description(
    synthetic_curves: dict[int, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``generate_class_prototypes`` delega 18 veces con parcel_id/crop_hint."""
    monkeypatch.setattr(
        proto_mod,
        "compute_class_mean_ndvi_curves",
        lambda *_a, **_k: synthetic_curves,
    )
    monkeypatch.setattr(
        proto_mod,
        "_encode_descriptions",
        lambda descriptions: np.ones((len(descriptions), _EMB_DIM), dtype=np.float32),
    )

    seen: list[dict[str, object]] = []

    def fake_generate(
        *,
        ndvi_curve: np.ndarray,
        doy: np.ndarray,
        parcel_id: str,
        crop_type_hint: str,
        model: str,
    ) -> str:
        seen.append(
            {
                "parcel_id": parcel_id,
                "crop_type_hint": crop_type_hint,
                "model": model,
            }
        )
        return f"desc {parcel_id}"

    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        fake_generate,
    )

    class_names = load_class_names()
    out = tmp_path / "proto.parquet"
    generate_class_prototypes(output_path=out)

    assert len(seen) == len(_CROP_CLASS_IDS)
    for offset, c in enumerate(_CROP_CLASS_IDS):
        assert seen[offset]["parcel_id"] == f"class_{c}"
        assert seen[offset]["crop_type_hint"] == class_names[c]
        assert seen[offset]["model"] == "gemini-3.5-flash"


def test_encode_descriptions_shape_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_encode_descriptions`` -> (N, 384) float32 L2-norm (encoder mockeado).

    El encoder se mockea a nivel de ``sentence_transformers.SentenceTransformer``
    para no descargar ``all-MiniLM-L6-v2`` en CI, conservando el contrato real
    (``normalize_embeddings=True`` -> filas L2-norm).
    """

    class _FakeEncoder:
        def __init__(self, model_name: str) -> None:
            self._model_name = model_name

        def encode(
            self,
            texts: list[str],
            *,
            normalize_embeddings: bool = True,
            convert_to_numpy: bool = True,
        ) -> np.ndarray:
            rng = np.random.default_rng(0)
            emb = rng.standard_normal((len(texts), _EMB_DIM)).astype(np.float32)
            if normalize_embeddings:
                emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            return emb

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEncoder)

    out = _encode_descriptions(["cultivo a", "cultivo b"])
    assert out.shape == (2, _EMB_DIM)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 6.2 Schema y contenido del parquet REAL (AC-1/2/3) — skipif si no existe.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="parquet real no presente (requiere dvc pull)",
)
def test_real_parquet_schema() -> None:
    """El parquet REAL cumple el esquema: (18, 388), class_id 1..18."""
    df = pl.read_parquet(_REAL_PARQUET)
    assert df.shape == (18, 388)
    assert df["class_id"].to_list() == list(range(1, 19))
    assert df.schema["description"] == pl.Utf8
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert len(emb_cols) == _EMB_DIM


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="parquet real no presente (requiere dvc pull)",
)
def test_real_descriptions_are_real_es() -> None:
    """Descripciones ES reales: todas > 50 chars, no placeholder, no vacias."""
    df = pl.read_parquet(_REAL_PARQUET)
    descriptions = df["description"].to_list()
    assert len(descriptions) == 18
    for desc in descriptions:
        assert isinstance(desc, str)
        assert len(desc) > 50
        assert not desc.startswith("class_")
        assert "placeholder" not in desc.lower()
    # Las longitudes reproducen las golden lens verificadas en disco.
    assert [len(d) for d in descriptions] == _GOLDEN_DESC_LENS


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="parquet real no presente (requiere dvc pull)",
)
def test_real_embeddings_nontrivial() -> None:
    """Embeddings REALES: abs-sum por fila > 0 y norma L2 ~ 1.0."""
    proto, class_ids = load_class_prototype_embeddings(_REAL_PARQUET)
    assert proto.shape == (18, _EMB_DIM)
    assert proto.dtype == np.float32
    assert np.isfinite(proto).all()
    abs_sum = np.abs(proto).sum(axis=1)
    assert (abs_sum > 0).all()
    norms = np.linalg.norm(proto, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    assert class_ids == list(range(1, 19))


@pytest.mark.skipif(
    not _REAL_PARQUET.exists(),
    reason="parquet real no presente (requiere dvc pull)",
)
def test_load_class_prototype_embeddings_ordered() -> None:
    """``load_class_prototype_embeddings`` -> matriz (18,384) y class_ids."""
    proto, class_ids = load_class_prototype_embeddings(_REAL_PARQUET)
    assert proto.shape == (18, _EMB_DIM)
    assert class_ids == sorted(class_ids)
    assert class_ids == list(range(1, 19))
