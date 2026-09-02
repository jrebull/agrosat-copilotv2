"""Tests US-023-preview-v2 P5 — ``ml.ingest.remoteclip_extractor``.

Mockea ``transformers.CLIPModel`` y ``CLIPProcessor`` para evitar descarga
de pesos en CI/local. Cubre:

- Schema output: parcel_id Utf8 + year Int16 + 512 cols Float32.
- Preprocesamiento RGB: stretch percentil + uint8 + resize aplicado bien.
- Error claro si ``imagery_path`` no existe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import torch

from ml.ingest import remoteclip_extractor
from ml.ingest.remoteclip_extractor import (
    EMBED_DIM,
    extract_remoteclip_embeddings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subset_parquet(path: Path, n: int = 3) -> None:
    """Subset metadata con parcel_id + year + label_id."""
    df = pl.DataFrame(
        {
            "parcel_id": [f"q{i:03d}" for i in range(n)],
            "year": [2019] * n,
            "label_id": list(range(n)),
        },
        schema={
            "parcel_id": pl.Utf8,
            "year": pl.Int16,
            "label_id": pl.Int64,
        },
    )
    df.write_parquet(path)


def _make_imagery_parquet(path: Path, n: int = 3, t: int = 2, c: int = 4, hw: int = 16) -> None:
    """Imagery parquet con un crop sintetico por parcela.

    Encoding: ``image`` como ``List[List[Float32]]`` (lista anidada) con
    shape ``(T, C, H, W)`` aplanada en la dim mayor. Cada fila guarda el
    ndarray como lista de listas para que Polars lo persista.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=42)
    for i in range(n):
        arr = rng.uniform(0.05, 0.95, size=(t, c, hw, hw)).astype(np.float32)
        rows.append(
            {
                "parcel_id": f"q{i:03d}",
                "year": 2019,
                "image": arr.tolist(),
                "shape": [t, c, hw, hw],
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _patch_clip(
    monkeypatch: pytest.MonkeyPatch, fail_primary: bool = False
) -> dict[str, MagicMock]:
    """Parchea ``CLIPModel.from_pretrained`` y ``CLIPProcessor.from_pretrained``.

    El modelo fake devuelve un tensor determinista en ``get_image_features``,
    el processor fake devuelve ``pixel_values`` ya tensorizados.

    Args:
        fail_primary: si True, ``from_pretrained(DEFAULT_MODEL_ID)`` lanza
            excepcion para verificar el fallback OpenAI.

    Returns:
        dict con los mocks para assertions.
    """
    import transformers

    # Mock model: get_image_features devuelve (B, 512) determinista.
    fake_model = MagicMock(name="CLIPModel")
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model

    def _fake_get_features(pixel_values: torch.Tensor) -> torch.Tensor:
        b = pixel_values.shape[0]
        return torch.linspace(0.0, 1.0, steps=b * EMBED_DIM).reshape(b, EMBED_DIM)

    fake_model.get_image_features = MagicMock(side_effect=_fake_get_features)

    # Processor mock: devuelve dict con pixel_values (B, 3, 224, 224).
    def _fake_processor_call(
        images: list[np.ndarray], return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        b = len(images)
        return {"pixel_values": torch.zeros((b, 3, 224, 224), dtype=torch.float32)}

    fake_processor = MagicMock(name="CLIPProcessor", side_effect=_fake_processor_call)

    primary_calls: list[str] = []

    def _model_from_pretrained(model_id: str, *a: Any, **kw: Any) -> MagicMock:
        primary_calls.append(model_id)
        if fail_primary and model_id == remoteclip_extractor.DEFAULT_MODEL_ID:
            raise RuntimeError("simulated network failure for RemoteCLIP")
        return fake_model

    def _processor_from_pretrained(model_id: str, *a: Any, **kw: Any) -> MagicMock:
        if fail_primary and model_id == remoteclip_extractor.DEFAULT_MODEL_ID:
            raise RuntimeError("simulated network failure for RemoteCLIP processor")
        return fake_processor

    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", _model_from_pretrained)
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", _processor_from_pretrained)
    return {
        "model": fake_model,
        "processor": fake_processor,
        "primary_calls": primary_calls,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_schema_has_512_cols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet final tiene parcel_id Utf8 + year Int16 + 512 cols Float32."""
    subset_path = tmp_path / "subset.parquet"
    imagery_path = tmp_path / "imagery.parquet"
    out_path = tmp_path / "remoteclip.parquet"
    _make_subset_parquet(subset_path, n=3)
    _make_imagery_parquet(imagery_path, n=3, t=2)
    _patch_clip(monkeypatch)

    result = extract_remoteclip_embeddings(
        pastis_eval_subset_path=subset_path,
        imagery_path=imagery_path,
        output_path=out_path,
        device="cpu",
        batch_size=2,
    )
    assert result.exists()
    df = pl.read_parquet(result)
    assert df.height == 3
    assert df.width == 2 + EMBED_DIM  # parcel_id + year + 512 embeddings
    assert df.schema["parcel_id"] == pl.Utf8
    assert df.schema["year"] == pl.Int16
    # 512 cols con prefijo correcto y dtype Float32.
    embed_cols = [c for c in df.columns if c.startswith("remoteclip_")]
    assert len(embed_cols) == EMBED_DIM
    for col in embed_cols:
        assert df.schema[col] == pl.Float32
    # Orden numerico estable.
    assert embed_cols == [f"remoteclip_{i:03d}" for i in range(EMBED_DIM)]


def test_rgb_preprocessing_stretch_uint8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_stretch_percentile_uint8`` produce uint8 ``[0, 255]`` por banda."""
    # Test directo del preprocessor (no requiere model patching).
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.1, 0.9, size=(2, 3, 16, 16)).astype(np.float32)
    out = remoteclip_extractor._stretch_percentile_uint8(rgb)
    assert out.dtype == np.uint8
    # Shape conversion (T, C, H, W) -> (T, H, W, C).
    assert out.shape == (2, 16, 16, 3)
    assert out.min() >= 0
    assert out.max() <= 255
    # Stretch real: al menos un pixel sube cerca de 255 y otro baja a 0.
    assert out.max() > 200
    assert out.min() < 50

    # Tambien valida _select_rgb (band_indices PASTIS-R = (2,1,0)).
    arr_4d = rng.uniform(0, 1, size=(1, 4, 8, 8)).astype(np.float32)
    rgb_sel = remoteclip_extractor._select_rgb(arr_4d, (2, 1, 0))
    assert rgb_sel.shape == (1, 3, 8, 8)
    np.testing.assert_array_equal(rgb_sel[0, 0], arr_4d[0, 2])  # R = B04 idx 2
    np.testing.assert_array_equal(rgb_sel[0, 1], arr_4d[0, 1])  # G = B03 idx 1
    np.testing.assert_array_equal(rgb_sel[0, 2], arr_4d[0, 0])  # B = B02 idx 0


def test_raises_when_imagery_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si ``imagery_path`` no existe lanza ``FileNotFoundError`` claro."""
    subset_path = tmp_path / "subset.parquet"
    _make_subset_parquet(subset_path, n=2)
    missing = tmp_path / "does_not_exist.parquet"
    _patch_clip(monkeypatch)

    with pytest.raises(FileNotFoundError, match="imagery_path does not exist"):
        extract_remoteclip_embeddings(
            pastis_eval_subset_path=subset_path,
            imagery_path=missing,
            output_path=tmp_path / "out.parquet",
            device="cpu",
        )


def test_fallback_to_openai_clip_when_remoteclip_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si RemoteCLIP no se puede descargar, usa OpenAI CLIP como fallback."""
    subset_path = tmp_path / "subset.parquet"
    imagery_path = tmp_path / "imagery.parquet"
    _make_subset_parquet(subset_path, n=2)
    _make_imagery_parquet(imagery_path, n=2, t=1)
    mocks = _patch_clip(monkeypatch, fail_primary=True)

    out_path = tmp_path / "remoteclip.parquet"
    extract_remoteclip_embeddings(
        pastis_eval_subset_path=subset_path,
        imagery_path=imagery_path,
        output_path=out_path,
        device="cpu",
        batch_size=2,
    )
    # El primer intento fue al modelo RemoteCLIP, el segundo al fallback.
    assert remoteclip_extractor.DEFAULT_MODEL_ID in mocks["primary_calls"]
    assert remoteclip_extractor.FALLBACK_MODEL_ID in mocks["primary_calls"]
