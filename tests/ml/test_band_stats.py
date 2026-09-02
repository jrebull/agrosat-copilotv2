"""Tests para `ml.analysis.band_stats`."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.analysis.band_stats import ndvi_temporal, summarize_bands

BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


@pytest.fixture
def normal_band_df() -> pl.DataFrame:
    """DataFrame long-format con 10 bandas y 1000 valores normales por banda."""
    rng = np.random.default_rng(42)
    rows = []
    for i, band in enumerate(BANDS_10):
        vals = rng.normal(loc=1000.0 + i * 100, scale=200.0, size=1000)
        for v in vals:
            rows.append({"band": band, "value": float(v)})
    return pl.DataFrame(rows)


def test_summarize_bands_shape(normal_band_df: pl.DataFrame) -> None:
    """Debe retornar 10 filas (bandas) x 10 columnas (stats + band)."""
    summary = summarize_bands(normal_band_df)
    assert summary.height == 10
    expected_cols = {
        "band",
        "mean",
        "std",
        "min",
        "max",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
    }
    assert expected_cols.issubset(set(summary.columns))


def test_summarize_bands_means_match(normal_band_df: pl.DataFrame) -> None:
    """Las medias por banda deben aproximar 1000 + i*100 con tolerancia 50."""
    summary = summarize_bands(normal_band_df).sort("band")
    summary_dict = {r["band"]: r["mean"] for r in summary.iter_rows(named=True)}
    for i, band in enumerate(BANDS_10):
        assert abs(summary_dict[band] - (1000 + i * 100)) < 50


def test_summarize_bands_raises_on_missing_columns() -> None:
    """Falta de columna requerida -> ValueError."""
    bad = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        summarize_bands(bad)


def test_ndvi_temporal_basic() -> None:
    """`ndvi_temporal` debe agrupar por mes y producir NDVI en [-1, 1]."""
    df = pl.DataFrame(
        {
            "patch_id": ["p1"] * 8,
            "t": [0, 0, 1, 1, 2, 2, 3, 3],
            "date": [
                20180915,
                20180915,
                20181020,
                20181020,
                20190415,
                20190415,
                20190520,
                20190520,
            ],
            "band": ["B04", "B08", "B04", "B08", "B04", "B08", "B04", "B08"],
            "value": [500.0, 3000.0, 600.0, 3500.0, 700.0, 3200.0, 800.0, 4000.0],
            "class_name": ["Corn"] * 8,
            "y": [0] * 8,
            "x": [0] * 8,
        }
    )
    out = ndvi_temporal(df, group_by=["month", "class_name"])
    assert "ndvi_mean" in out.columns
    assert out.height > 0
    assert out["ndvi_mean"].min() >= -1.001
    assert out["ndvi_mean"].max() <= 1.001
