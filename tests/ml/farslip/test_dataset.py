"""Tests del builder + Dataset FarSLIP.

Cobertura objetivo ``ml/farslip/dataset.py`` >= 75 %.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.farslip.dataset import (
    MANIFEST_SCHEMA,
    FarSLIPDataset,
    _doy_to_phenology,
    _generate_synthetic_parcels,
    _region_to_display,
    build_farslip_pairs,
)


@pytest.fixture
def vocab_path() -> Path:
    return Path("ml/farslip/cap_vocabulary.yaml")


@pytest.fixture
def small_parcels() -> pl.DataFrame:
    """3 parcelas por ROI, todas con cloud_prob baja."""
    rng = np.random.default_rng(123)
    rois = ["pianura_padana", "toscana", "puglia"]
    out = []
    for r in rois:
        df = _generate_synthetic_parcels(roi=r, n=3, rng=rng, cap_classes=["mais", "vite", "olivo"])
        df = df.with_columns(pl.lit(0.05).alias("cloud_prob"))
        out.append(df)
    return pl.concat(out, how="vertical")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_pairs_writes_manifest_and_tiffs(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    out = build_farslip_pairs(
        rois=("pianura_padana", "toscana", "puglia"),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels,
        seed=1,
        crop_size_px=64,  # mas rapido en CI
    )
    assert out.height == 9
    for roi in ("pianura_padana", "toscana", "puglia"):
        m = tmp_path / roi / "manifest.parquet"
        assert m.exists()
        df = pl.read_parquet(m)
        assert df.height == 3
        # Verifica columnas del schema canonico
        for col in MANIFEST_SCHEMA:
            assert col in df.columns
        # los crops fisicos existen
        for p in df["crop_path"].to_list():
            assert Path(p).exists()


def test_build_pairs_three_languages_present(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    out = build_farslip_pairs(
        rois=("pianura_padana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels.filter(pl.col("region") == "pianura_padana"),
        seed=1,
        crop_size_px=32,
    )
    for r in out.iter_rows(named=True):
        assert isinstance(r["text_it"], str) and len(r["text_it"]) > 0
        assert isinstance(r["text_es"], str) and len(r["text_es"]) > 0
        assert isinstance(r["text_en"], str) and len(r["text_en"]) > 0


def test_build_pairs_qa_filters_cloudy(tmp_path: Path, vocab_path: Path) -> None:
    """Records con cloud_prob > threshold no se incluyen."""
    rng = np.random.default_rng(7)
    df = _generate_synthetic_parcels(roi="toscana", n=10, rng=rng, cap_classes=["mais"])
    df = df.with_columns(pl.Series("cloud_prob", [0.05] * 5 + [0.9] * 5))
    out = build_farslip_pairs(
        rois=("toscana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=df,
        qa_cloud_threshold=0.2,
        seed=1,
        crop_size_px=32,
    )
    assert out.height == 5


def test_build_pairs_idempotent_no_duplicates(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    """Llamar dos veces con los mismos parcels no duplica filas."""
    build_farslip_pairs(
        rois=("pianura_padana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels.filter(pl.col("region") == "pianura_padana"),
        seed=1,
        crop_size_px=32,
    )
    out2 = build_farslip_pairs(
        rois=("pianura_padana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels.filter(pl.col("region") == "pianura_padana"),
        seed=1,
        crop_size_px=32,
    )
    df = pl.read_parquet(tmp_path / "pianura_padana" / "manifest.parquet")
    assert df.height == 3
    assert out2.height == 3


def test_balance_min_max_ratio(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    """3 ROIs con 3 parcelas cada uno => ratio min/max = 1.0 >= 0.20."""
    build_farslip_pairs(
        rois=("pianura_padana", "toscana", "puglia"),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels,
        seed=1,
        crop_size_px=32,
    )
    counts = []
    for r in ("pianura_padana", "toscana", "puglia"):
        df = pl.read_parquet(tmp_path / r / "manifest.parquet")
        counts.append(df.height)
    ratio = min(counts) / max(counts)
    assert ratio >= 0.20


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def test_doy_to_phenology_buckets() -> None:
    assert _doy_to_phenology(50) == "germinazione"
    assert _doy_to_phenology(120) == "fioritura"
    assert _doy_to_phenology(180) == "fruttificazione"
    assert _doy_to_phenology(250) == "maturazione"
    assert _doy_to_phenology(330) == "raccolta"


def test_region_to_display() -> None:
    assert _region_to_display("pianura_padana") == "Pianura Padana"
    assert _region_to_display("toscana") == "Toscana"
    assert _region_to_display("puglia") == "Puglia"
    assert _region_to_display("unknown_roi") == "Unknown Roi"


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------


def test_dataset_getitem_shape(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    """__getitem__ devuelve image (4, 224, 224) tras resize."""
    build_farslip_pairs(
        rois=("toscana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels.filter(pl.col("region") == "toscana"),
        seed=1,
        crop_size_px=64,
    )
    manifest = tmp_path / "toscana" / "manifest.parquet"
    ds = FarSLIPDataset(manifest, tokenizer=None, crop_resize_to=224)
    assert len(ds) == 3
    item = ds[0]
    assert item["image"].shape == (4, 224, 224)
    assert item["input_ids"].shape == (77,)
    assert item["attention_mask"].shape == (77,)
    assert item["region_id"].dim() == 0
    assert item["category_id"].dim() == 0


def test_dataset_round_robin_languages(
    tmp_path: Path, vocab_path: Path, small_parcels: pl.DataFrame
) -> None:
    build_farslip_pairs(
        rois=("toscana",),
        output_root=tmp_path,
        vocabulary_path=vocab_path,
        parcel_records=small_parcels.filter(pl.col("region") == "toscana"),
        seed=1,
        crop_size_px=32,
    )
    manifest = tmp_path / "toscana" / "manifest.parquet"
    ds = FarSLIPDataset(manifest, tokenizer=None, lang_strategy="round_robin")
    texts = [ds[i]["text"] for i in range(len(ds))]
    # Por el round_robin de 3, los 3 items deberian usar 3 idiomas distintos
    assert len(set(texts)) == len(texts)
