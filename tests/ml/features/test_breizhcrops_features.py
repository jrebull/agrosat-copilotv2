"""Tests del adaptador :mod:`ml.features.breizhcrops_features`.

Verifica que el long-format de bandas crudas BreizhCrops se convierte al
mismo vector de 185 features que produce PASTIS-R, reusando el pipeline
canonico (``compute_index`` + ``extract_temporal_features``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml.features.breizhcrops_features import (
    REFLECTANCE_SCALE,
    build_breizhcrops_features,
    pixel_series_to_index_dataarray,
)
from ml.features.temporal_features import DEFAULT_INDICES
from ml.ingest.pastis_loader import PASTIS_S2_BANDS


def _synthetic_long(
    *,
    parcel_id: str,
    class_id: int,
    class_name: str,
    n_steps: int = 12,
    seed: int = 0,
) -> pl.DataFrame:
    """Construye un long-format sintetico de UNA parcela (bandas crudas DN)."""
    rng = np.random.default_rng(seed)
    # Fechas mensuales 2017 como enteros YYYYMMDD.
    dates = np.array([2017_0000 + (m + 1) * 100 + 15 for m in range(n_steps)], dtype=np.int64)
    doys = np.array([(m + 1) * 30 for m in range(n_steps)], dtype=np.int64)
    rows: list[dict[str, object]] = []
    for band in PASTIS_S2_BANDS:
        # DN realistas (0-6000), NIR mas alto que rojo para NDVI positivo.
        base = 4000.0 if band in ("B08", "B8A", "B06", "B07") else 1500.0
        values = base + rng.uniform(-300, 300, size=n_steps)
        for t in range(n_steps):
            rows.append(
                {
                    "parcel_id": parcel_id,
                    "t": t,
                    "date": int(dates[t]),
                    "doy": int(doys[t]),
                    "band": band,
                    "value": float(values[t]),
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )
    return pl.DataFrame(rows)


def test_pixel_series_to_index_dataarray_shape_and_attrs() -> None:
    """El DataArray resultante tiene dims (time, band=indices) y attrs."""
    long_df = _synthetic_long(parcel_id="42", class_id=1, class_name="wheat")
    da = pixel_series_to_index_dataarray(long_df, parcel_id_int=42, year=2017)

    assert da is not None
    assert da.dims == ("time", "band")
    assert list(da.coords["band"].values) == list(DEFAULT_INDICES)
    assert da.attrs["parcel_id"] == 42
    assert da.attrs["year"] == 2017
    # NDVI calculado debe ser finito y en rango plausible.
    ndvi = da.sel(band="NDVI").values
    assert np.isfinite(ndvi).all()
    assert ((ndvi >= -1.0) & (ndvi <= 1.0)).all()


def test_pixel_series_to_index_dataarray_too_short_returns_none() -> None:
    """Una parcela con <2 pasos temporales se rechaza (insuficiente FFT)."""
    long_df = _synthetic_long(parcel_id="1", class_id=1, class_name="wheat", n_steps=1)
    da = pixel_series_to_index_dataarray(long_df, parcel_id_int=1, year=2017)
    assert da is None


def test_build_breizhcrops_features_matches_pastis_185_schema() -> None:
    """El vector de features tiene las mismas 185 columnas que PASTIS-R."""
    frames = [
        _synthetic_long(parcel_id="a", class_id=1, class_name="wheat", seed=1),
        _synthetic_long(parcel_id="b", class_id=3, class_name="corn", seed=2),
        _synthetic_long(parcel_id="c", class_id=0, class_name="barley", seed=3),
    ]
    pixel_series = pl.concat(frames, how="vertical_relaxed")

    feats = build_breizhcrops_features(pixel_series)

    assert feats.height == 3
    meta = {"parcel_id", "year", "class_id", "class_name"}
    feature_cols = [c for c in feats.columns if c not in meta]
    # 17 indices x 9 stats + 3 fft x 8 + 8 fenologicas = 185.
    assert len(feature_cols) == 185
    # Columnas canonicas representativas presentes.
    for col in ("NDVI_mean", "EVI_p95", "NDVI_fft_amp_1", "ndvi_auc", "peak_doy"):
        assert col in feats.columns
    # Metadata propagada.
    assert set(feats.get_column("class_name").to_list()) == {"wheat", "corn", "barley"}


def test_reflectance_scale_default() -> None:
    """El divisor DN -> reflectancia es 10000 (contrato Sentinel-2)."""
    assert REFLECTANCE_SCALE == 10_000.0


def test_build_breizhcrops_features_empty_input() -> None:
    """Input vacio devuelve DataFrame con esquema minimo (no crashea)."""
    empty = pl.DataFrame(
        schema={
            "parcel_id": pl.Utf8,
            "t": pl.Int64,
            "date": pl.Int64,
            "doy": pl.Int64,
            "band": pl.Utf8,
            "value": pl.Float64,
            "class_id": pl.Int16,
            "class_name": pl.Utf8,
        }
    )
    feats = build_breizhcrops_features(empty)
    assert feats.height == 0
    assert "parcel_id" in feats.columns
