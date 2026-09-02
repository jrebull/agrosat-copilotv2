"""Tests de ``ml.features.phenology_description`` (US-022b-D).

Gemini se mockea SIEMPRE en CI (R7, regla dura del plan). Los tests cubren:

- generacion de prompt 3-bloques (estructura Wen Fig. 2),
- cache hit / miss por hash de curva,
- determinismo de ``temperature=0`` (enforce),
- shape estable del encoder (mockeado tambien para no requerir descargar
  sentence-transformers en CI),
- integracion con :func:`build_phenology_text_block`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.features.phenology_description as pd_mod
from ml.features.phenology_description import (
    _CREDENTIAL_ENV_VARS,
    DEFAULT_TEXT_EMBED_DIM,
    PROMPT_TEMPLATE,
    _has_credentials,
    build_phenology_text_block,
    encode_descriptions,
    generate_phenology_description,
    set_llm_client,
)


@pytest.fixture(autouse=True)
def restore_client():
    """Resetea el cliente LLM tras cada test (aisla mocks)."""
    yield
    set_llm_client(None)


@pytest.fixture
def synthetic_ndvi_curve() -> np.ndarray:
    """Curva NDVI sintetica con peak en DOY ~180 (anio agronomico tipico)."""
    t = np.arange(72, dtype=np.float64)
    return (0.2 + 0.6 * np.sin(np.pi * t / 72.0) ** 2).astype(np.float64)


@pytest.fixture
def synthetic_doy() -> np.ndarray:
    return np.linspace(1.0, 365.0, 72, dtype=np.float64)


def test_generate_description_calls_llm_client_with_prompt(
    synthetic_ndvi_curve: np.ndarray, synthetic_doy: np.ndarray, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return "Cultivo con pico medio y temporada de verano corta."

    set_llm_client(mock_client)
    desc = generate_phenology_description(
        synthetic_ndvi_curve,
        synthetic_doy,
        parcel_id=42,
        cache_dir=tmp_path,
    )
    assert len(calls) == 1
    assert "[BLOQUE 1" in calls[0]["prompt"]
    assert "[BLOQUE 2" in calls[0]["prompt"]
    assert "[BLOQUE 3" in calls[0]["prompt"]
    assert calls[0]["temperature"] == 0.0
    assert desc.startswith("Cultivo con pico")


def test_generate_description_uses_cache_on_second_call(
    synthetic_ndvi_curve: np.ndarray, tmp_path: Path
) -> None:
    n_calls = {"count": 0}

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        n_calls["count"] += 1
        return "Descripcion cacheable."

    set_llm_client(mock_client)
    desc1 = generate_phenology_description(synthetic_ndvi_curve, parcel_id=99, cache_dir=tmp_path)
    desc2 = generate_phenology_description(synthetic_ndvi_curve, parcel_id=99, cache_dir=tmp_path)
    assert desc1 == desc2
    assert n_calls["count"] == 1  # segundo call: cache hit, no llamada LLM.


def test_generate_description_temperature_nonzero_raises(
    synthetic_ndvi_curve: np.ndarray, tmp_path: Path
) -> None:
    set_llm_client(lambda p, **_: "no debe llegar aqui")
    with pytest.raises(ValueError, match="temperature"):
        generate_phenology_description(synthetic_ndvi_curve, temperature=0.7, cache_dir=tmp_path)


def test_generate_description_empty_curve_raises(tmp_path: Path) -> None:
    set_llm_client(lambda p, **_: "")
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_phenology_description(np.array([], dtype=np.float64), cache_dir=tmp_path)


def test_generate_description_nd_curve_raises(tmp_path: Path) -> None:
    set_llm_client(lambda p, **_: "")
    with pytest.raises(ValueError, match="1D"):
        generate_phenology_description(np.zeros((10, 10), dtype=np.float64), cache_dir=tmp_path)


def test_generate_description_handles_nan_in_curve(tmp_path: Path) -> None:
    curve = np.array([0.3, np.nan, 0.7, np.nan, 0.5, 0.2], dtype=np.float64)
    captured: dict[str, str] = {}

    def mock_client(prompt: str, **_: object) -> str:
        captured["prompt"] = prompt
        return "ok"

    set_llm_client(mock_client)
    generate_phenology_description(curve, parcel_id=1, cache_dir=tmp_path)
    # El prompt no debe contener "nan" literal (NaN se imputa antes).
    assert "nan" not in captured["prompt"].lower()


def test_prompt_template_contains_three_blocks() -> None:
    assert PROMPT_TEMPLATE.count("[BLOQUE") == 3


def test_encode_descriptions_empty_returns_empty_matrix() -> None:
    out = encode_descriptions([], encoder="sentence-transformers")
    assert out.shape == (0, DEFAULT_TEXT_EMBED_DIM)


def test_encode_descriptions_invalid_encoder_raises() -> None:
    with pytest.raises(ValueError, match="encoder"):
        encode_descriptions(["test"], encoder="invalid")


def test_encode_descriptions_farslip_clip_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        encode_descriptions(["test"], encoder="farslip-clip")


def test_build_phenology_text_block_skip_llm(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 6
    df = pl.DataFrame(
        {
            "parcel_id": list(range(n)),
            "year": [2019] * n,
            # FFT (1 DC + 3 armonicos = 4 cada uno) para fallback NDVI.
            **{f"NDVI_fft_amp_{k}": rng.normal(size=n).tolist() for k in range(4)},
            **{f"NDVI_fft_phase_{k}": rng.normal(size=n).tolist() for k in range(4)},
        }
    )
    with warnings.catch_warnings():
        # skip_llm=True ahora emite DeprecationWarning fuera de tests;
        # aqui es legitimo porque estamos testeando.
        warnings.simplefilter("ignore", DeprecationWarning)
        block = build_phenology_text_block(
            df,
            skip_llm=True,
            cache_dir=tmp_path,
        )
    assert block.height == n
    assert "parcel_id" in block.columns
    assert "year" in block.columns
    text_cols = [c for c in block.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == DEFAULT_TEXT_EMBED_DIM


def test_build_phenology_text_block_raises_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si no hay cliente inyectado ni env vars, debe levantar RuntimeError."""
    set_llm_client(None)
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    df = pl.DataFrame(
        {
            "parcel_id": [0, 1],
            "year": [2019, 2019],
            "NDVI_t_00": [0.3, 0.4],
            "NDVI_t_01": [0.5, 0.6],
        }
    )
    with pytest.raises(RuntimeError, match="Gemini is not configured"):
        build_phenology_text_block(df, skip_llm=False, cache_dir=tmp_path)


def test_build_phenology_text_block_runs_with_injected_client(
    tmp_path: Path,
) -> None:
    """Con cliente inyectado, no exige env vars y produce shape correcto."""
    n = 3

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        return "Descripcion sintetica deterministica."

    set_llm_client(mock_client)

    def mock_encode(
        descriptions: list[str],
        *,
        encoder: str = "sentence-transformers",
        model_name: str | None = None,
    ) -> np.ndarray:
        return np.ones((len(descriptions), 12), dtype=np.float32)

    original_encode = pd_mod.encode_descriptions
    pd_mod.encode_descriptions = mock_encode  # type: ignore[assignment]
    try:
        df = pl.DataFrame(
            {
                "parcel_id": [f"p{i}" for i in range(n)],
                "year": [2019] * n,
                "NDVI_t_00": [0.2, 0.3, 0.5],
                "NDVI_t_01": [0.6, 0.7, 0.8],
            }
        )
        block = build_phenology_text_block(
            df,
            skip_llm=False,
            cache_dir=tmp_path,
            progress_every=1,
        )
    finally:
        pd_mod.encode_descriptions = original_encode  # type: ignore[assignment]

    assert block.height == n
    text_cols = [c for c in block.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == 12


def test_has_credentials_detects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_has_credentials`` lee la primera env var presente y no vacia."""
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert _has_credentials() is False
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-value")
    assert _has_credentials() is True


def test_has_credentials_ignores_falsy_vertex_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GOOGLE_GENAI_USE_VERTEXAI=false`` no cuenta como credencial valida."""
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    assert _has_credentials() is False


def test_build_phenology_text_block_skip_llm_emits_deprecation(
    tmp_path: Path,
) -> None:
    """``skip_llm=True`` debe emitir DeprecationWarning explicito."""
    df = pl.DataFrame(
        {
            "parcel_id": [0],
            "year": [2019],
            "NDVI_t_00": [0.5],
        }
    )
    with pytest.warns(DeprecationWarning, match="skip_llm=True"):
        build_phenology_text_block(df, skip_llm=True, cache_dir=tmp_path)


def test_build_phenology_text_block_calls_llm_per_row(tmp_path: Path) -> None:
    n = 4
    calls: list[int | str] = []

    def mock_client(prompt: str, *, model: str, temperature: float) -> str:
        return f"Descripcion {len(calls)}"

    set_llm_client(mock_client)

    # Mock del encoder para no descargar sentence-transformers en CI.
    def mock_encode(descriptions, *, encoder="sentence-transformers", model_name=None):
        return np.ones((len(descriptions), 8), dtype=np.float32)

    monkey_target = pd_mod
    original_encode = monkey_target.encode_descriptions
    monkey_target.encode_descriptions = mock_encode  # type: ignore[assignment]

    try:
        rng = np.random.default_rng(0)
        df = pl.DataFrame(
            {
                "parcel_id": list(range(n)),
                "year": [2019] * n,
                **{f"NDVI_fft_amp_{k}": rng.normal(size=n).tolist() for k in range(4)},
                **{f"NDVI_fft_phase_{k}": rng.normal(size=n).tolist() for k in range(4)},
            }
        )
        block = build_phenology_text_block(
            df,
            skip_llm=False,
            cache_dir=tmp_path,
        )
    finally:
        monkey_target.encode_descriptions = original_encode  # type: ignore[assignment]

    assert block.height == n
    text_cols = [c for c in block.columns if c.startswith("pheno_text_")]
    assert len(text_cols) == 8
