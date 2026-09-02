"""Tests de ml.analysis.paper_methods (metodos derivados de 4 papers).

Cubren las 7 funciones publicas + helpers usando fixtures sinteticas
deterministas (numpy ``default_rng`` con semilla fija). Los tests que
requieren ``data/PASTIS-R/`` o parquets generados se marcan con
``@pytest.mark.empirical`` y degradan a ``skip`` si el dato no esta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from ml.analysis.paper_methods import (
    aggregate_rare_classes,
    boundary_interior_stats,
    boundary_pixel_mask,
    cloud_gap_robustness,
    compute_boundary_ratio,
    confusion_symmetry_analysis,
    phenology_calendar_features,
    temporal_sampling_stats,
)
from ml.features.temporal_features import extract_temporal_features

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"


# ---------------------------------------------------------------------------
# Fixtures sinteticas
# ---------------------------------------------------------------------------


def _synthetic_patch(
    *,
    n_timesteps: int = 5,
    size: int = 16,
    seed: int = 42,
) -> dict[str, object]:
    """Construye un patch PASTIS-R sintetico con 2 parcelas + fondo.

    La mascara semantica tiene un cuadrante con clase 2 y otro con clase 5;
    el resto es fondo (clase 0). El tensor S2 da a la clase 2 un NIR alto y a
    la clase 5 un NIR bajo para que los grupos sean distinguibles.
    """
    rng = np.random.default_rng(seed)
    half = size // 2

    semantic = np.zeros((size, size), dtype=np.uint8)
    semantic[:half, :half] = 2  # parcela A (cuadrante superior izquierdo)
    semantic[half:, half:] = 5  # parcela B (cuadrante inferior derecho)

    instance = np.zeros((size, size), dtype=np.int16)
    instance[:half, :half] = 1
    instance[half:, half:] = 2

    s2 = rng.normal(loc=1000.0, scale=50.0, size=(n_timesteps, 10, size, size))
    # Banda 6 (B08 NIR): firma distinta por clase.
    s2[:, 6, :half, :half] += 3000.0  # parcela A NIR alto
    s2[:, 6, half:, half:] += 500.0  # parcela B NIR bajo
    s2 = s2.astype(np.int16)

    return {
        "s2": s2,
        "semantic": semantic,
        "instance": instance,
        "zone": np.zeros((size, size), dtype=np.uint8),
        "dates_s2": [20190101 + i * 100 for i in range(n_timesteps)],
        "patch_id": "synthetic",
        "fold": 1,
    }


def _synthetic_timeseries(
    *,
    n_timesteps: int = 24,
    seed: int = 7,
) -> xr.DataArray:
    """Serie temporal sintetica (time, band) con curva fenologica NDVI."""
    rng = np.random.default_rng(seed)
    bands = ["NDVI", "NDWI", "EVI"]
    times = np.array(
        [
            np.datetime64("2019-01-01", "ns") + np.timedelta64(15 * i, "D")
            for i in range(n_timesteps)
        ],
        dtype="datetime64[ns]",
    )
    # NDVI con campana fenologica + ruido leve.
    doy = np.array([(t - times[0]) / np.timedelta64(1, "D") for t in times])
    ndvi = 0.2 + 0.6 * np.exp(-((doy - 180) ** 2) / (2 * 60.0**2))
    ndvi = ndvi + rng.normal(0.0, 0.01, size=n_timesteps)
    ndwi = -0.1 + rng.normal(0.0, 0.02, size=n_timesteps)
    evi = ndvi * 0.8

    values = np.stack([ndvi, ndwi, evi], axis=1)
    da = xr.DataArray(
        values,
        dims=("time", "band"),
        coords={"time": times, "band": bands},
    )
    da.attrs["parcel_id"] = 1
    da.attrs["year"] = 2019
    return da


def _extract_ndwi_evi_ndvi(ts: xr.DataArray) -> pl.DataFrame:
    """Wrapper de extract_temporal_features para series con 3 indices."""
    return extract_temporal_features(
        ts,
        indices=("NDVI", "NDWI", "EVI"),
        fft_indices=("NDVI", "NDWI", "EVI"),
    )


# ---------------------------------------------------------------------------
# Grupo: boundary (Paper B — Tarasiou et al. 2021)
# ---------------------------------------------------------------------------


def test_boundary_pixel_mask_detects_edges() -> None:
    """La mascara marca True en la frontera entre dos parcelas adyacentes."""
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[:, :4] = 1
    semantic[:, 4:] = 2
    mask = boundary_pixel_mask(semantic, neighbourhood=3)
    # Columnas 3 y 4 (a ambos lados del borde) deben ser frontera.
    assert mask[:, 3].all()
    assert mask[:, 4].all()
    # El interior homogeneo (columna 0, columna 7) no es frontera.
    assert not mask[:, 0].any()
    assert not mask[:, 7].any()


def test_boundary_interior_stats_groups() -> None:
    """boundary_interior_stats devuelve los 3 grupos esperados con conteos."""
    patch = _synthetic_patch(size=16)
    stats = boundary_interior_stats(patch, band_index=6, neighbourhood=3)
    assert set(stats.get_column("group").to_list()) == {
        "interior",
        "boundary",
        "exterior",
    }
    total = stats.get_column("count").sum()
    assert total == 16 * 16
    interior_count = stats.filter(pl.col("group") == "interior").get_column("count")[0]
    assert interior_count > 0


def test_boundary_ratio_range() -> None:
    """compute_boundary_ratio devuelve ratios en [0, 1] por instancia."""
    patch = _synthetic_patch(size=16)
    ratios = compute_boundary_ratio(patch, neighbourhood=3)
    assert set(ratios.keys()) == {1, 2}
    for ratio in ratios.values():
        assert 0.0 <= ratio <= 1.0


def test_boundary_pixel_mask_rejects_even_neighbourhood() -> None:
    """neighbourhood par debe lanzar ValueError."""
    semantic = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="neighbourhood"):
        boundary_pixel_mask(semantic, neighbourhood=4)


# ---------------------------------------------------------------------------
# Grupo: temporal sampling (Paper A — Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def test_temporal_sampling_stats_regular() -> None:
    """Serie con revisita regular: gap medio == gap min == gap max."""
    # 10 fechas espaciadas exactamente 10 dias.
    dates = [
        int(
            (np.datetime64("2019-01-01") + np.timedelta64(10 * i, "D"))
            .astype("datetime64[D]")
            .astype(str)
            .replace("-", "")
        )
        for i in range(10)
    ]
    stats = temporal_sampling_stats(dates)
    assert stats["n_obs"] == 10
    assert stats["mean_gap_days"] == pytest.approx(10.0)
    assert stats["std_gap_days"] == pytest.approx(0.0, abs=1e-9)
    assert stats["min_gap_days"] == stats["max_gap_days"]


def test_temporal_sampling_stats_irregular_gaps() -> None:
    """Serie irregular: std de gaps > 0 y max > min."""
    dates = [20190101, 20190105, 20190201, 20190203, 20190401]
    stats = temporal_sampling_stats(dates)
    assert stats["n_obs"] == 5
    assert stats["std_gap_days"] > 0.0
    assert stats["max_gap_days"] > stats["min_gap_days"]


def test_doy_coverage() -> None:
    """doy_coverage crece al anadir observaciones bien distribuidas."""
    sparse = temporal_sampling_stats([20190101, 20190701])
    dense = temporal_sampling_stats(
        [20190101, 20190301, 20190501, 20190701, 20190901, 20191101]
    )
    assert 0.0 <= sparse["doy_coverage"] <= 1.0
    assert dense["doy_coverage"] > sparse["doy_coverage"]


def test_temporal_sampling_stats_empty() -> None:
    """Lista vacia degrada graceful con ceros."""
    stats = temporal_sampling_stats([])
    assert stats["n_obs"] == 0
    assert stats["doy_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Grupo: confusion (Paper A — Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def test_confusion_symmetry_detects_symmetric_pair() -> None:
    """Confusion balanceada entre 2 clases -> componente simetrica domina."""
    # 10 verdaderos clase 1 predichos como 2, y 10 verdaderos 2 predichos 1.
    y_true = np.array([1] * 10 + [2] * 10 + [1] * 30 + [2] * 30)
    y_pred = np.array([2] * 10 + [1] * 10 + [1] * 30 + [2] * 30)
    df = confusion_symmetry_analysis(y_true, y_pred)
    row = df.row(0, named=True)
    assert row["symmetric"] == 10
    assert row["asymmetric"] == 0
    assert row["interpretation"] == "spectral_similarity"


def test_confusion_symmetry_detects_asymmetric() -> None:
    """Confusion en una sola direccion -> componente asimetrica domina."""
    y_true = np.array([1] * 20 + [2] * 20)
    y_pred = np.array([2] * 18 + [1] * 2 + [2] * 20)
    df = confusion_symmetry_analysis(y_true, y_pred)
    pair = df.filter(
        (pl.col("class_a") == "1") & (pl.col("class_b") == "2")
    ).row(0, named=True)
    assert pair["asymmetric"] > pair["symmetric"]
    assert pair["interpretation"] == "external_factor"


def test_confusion_interpretation_labels() -> None:
    """class_names mapea ids a nombres legibles en el resultado."""
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([2, 1, 1, 2])
    df = confusion_symmetry_analysis(
        y_true, y_pred, class_names={1: "wheat", 2: "rye"}
    )
    labels = set(df.get_column("class_a").to_list()) | set(
        df.get_column("class_b").to_list()
    )
    assert labels == {"wheat", "rye"}


def test_confusion_symmetry_length_mismatch() -> None:
    """y_true/y_pred de distinta longitud lanzan ValueError."""
    with pytest.raises(ValueError, match="must have equal length"):
        confusion_symmetry_analysis(np.array([1, 2]), np.array([1]))


# ---------------------------------------------------------------------------
# Grupo: rare classes (Paper A — Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def test_aggregate_rare_classes_collapses_below_threshold() -> None:
    """Clases con conteo < min_count se reasignan a other_label."""
    y = pl.Series("class", [1] * 500 + [2] * 500 + [3] * 10)
    remapped, report = aggregate_rare_classes(y, min_count=400, other_label=-1)
    assert 3 in report["aggregated"]
    assert (remapped == -1).sum() == 10
    assert (remapped == 3).sum() == 0


def test_aggregate_rare_classes_keeps_frequent() -> None:
    """Clases frecuentes permanecen intactas."""
    y = pl.Series("class", [1] * 500 + [2] * 450 + [3] * 5)
    remapped, report = aggregate_rare_classes(y, min_count=400)
    assert 1 not in report["aggregated"]
    assert 2 not in report["aggregated"]
    assert (remapped == 1).sum() == 500
    assert (remapped == 2).sum() == 450


def test_aggregate_rare_classes_report() -> None:
    """El report contiene conteos por clase original + metadata."""
    y = pl.Series("class", [1] * 500 + [7] * 3)
    _, report = aggregate_rare_classes(y, min_count=400, other_label=-9)
    assert report[1] == 500
    assert report[7] == 3
    assert report["min_count"] == 400
    assert report["other_label"] == -9
    assert report["aggregated"] == [7]


# ---------------------------------------------------------------------------
# Grupo: phenology calendar (Paper C — PVM 2025)
# ---------------------------------------------------------------------------


def test_phenology_calendar_creates_stages() -> None:
    """phenology_calendar_features agrega growth_stage en [0, n_stages-1]."""
    df = pl.DataFrame({"parcel_id": [1, 2, 3, 4], "peak_doy": [30, 130, 230, 330]})
    out = phenology_calendar_features(df, doy_col="peak_doy", n_stages=4)
    assert "growth_stage" in out.columns
    stages = out.get_column("growth_stage").to_list()
    assert all(0 <= s <= 3 for s in stages)
    # DOYs crecientes -> etapas no decrecientes.
    assert stages == sorted(stages)


def test_phenology_calendar_stage_names() -> None:
    """Con n_stages=4 las etapas usan los nombres fenologicos canonicos."""
    df = pl.DataFrame({"parcel_id": [1], "peak_doy": [200]})
    out = phenology_calendar_features(df, doy_col="peak_doy", n_stages=4)
    name = out.get_column("growth_stage_name")[0]
    assert name in {"dormant", "green_up", "peak", "senescence"}


def test_phenology_calendar_n_stages() -> None:
    """n_stages personalizado genera el numero correcto de etapas distintas."""
    df = pl.DataFrame(
        {"parcel_id": list(range(6)), "peak_doy": [30, 90, 150, 210, 270, 330]}
    )
    out = phenology_calendar_features(df, doy_col="peak_doy", n_stages=6)
    assert out.get_column("growth_stage").n_unique() == 6
    # Nombres genericos cuando n_stages != 4.
    assert out.get_column("growth_stage_name")[0].startswith("stage_")


def test_phenology_calendar_handles_null_doy() -> None:
    """peak_doy nulo -> growth_stage -1 y nombre 'unknown'."""
    df = pl.DataFrame({"parcel_id": [1, 2], "peak_doy": [180, None]})
    out = phenology_calendar_features(df, doy_col="peak_doy", n_stages=4)
    assert out.get_column("growth_stage").to_list()[1] == -1
    assert out.get_column("growth_stage_name").to_list()[1] == "unknown"


# ---------------------------------------------------------------------------
# Grupo: cloud gap robustness (Paper D — STCLN 2025)
# ---------------------------------------------------------------------------


def test_cloud_gap_robustness_baseline_zero_drift() -> None:
    """mask_fraction=0.0 debe tener drift exactamente 0."""
    ts = _synthetic_timeseries(n_timesteps=24)
    df = cloud_gap_robustness(
        _extract_ndwi_evi_ndvi, ts, mask_fractions=(0.0, 0.4), seed=42
    )
    baseline = df.filter(pl.col("mask_fraction") == 0.0)
    assert baseline.height > 0
    assert (baseline.get_column("drift_from_baseline") == 0.0).all()


def test_cloud_gap_robustness_drift_increases() -> None:
    """A mayor mask_fraction, la deriva media tiende a crecer."""
    ts = _synthetic_timeseries(n_timesteps=30)
    df = cloud_gap_robustness(
        _extract_ndwi_evi_ndvi, ts, mask_fractions=(0.0, 0.2, 0.6), seed=42
    )
    drift_02 = (
        df.filter(pl.col("mask_fraction") == 0.2)
        .get_column("drift_from_baseline")
        .drop_nulls()
        .drop_nans()
        .mean()
    )
    drift_06 = (
        df.filter(pl.col("mask_fraction") == 0.6)
        .get_column("drift_from_baseline")
        .drop_nulls()
        .drop_nans()
        .mean()
    )
    assert drift_02 is not None and drift_06 is not None
    assert drift_06 >= drift_02


def test_cloud_gap_robustness_shape() -> None:
    """El DataFrame de salida tiene las 5 columnas del contrato."""
    ts = _synthetic_timeseries(n_timesteps=20)
    df = cloud_gap_robustness(
        _extract_ndwi_evi_ndvi, ts, mask_fractions=(0.0, 0.3), seed=1
    )
    assert df.columns == [
        "mask_fraction",
        "n_timesteps_kept",
        "feature_name",
        "value",
        "drift_from_baseline",
    ]
    assert df.height > 0


def test_cloud_gap_robustness_short_series() -> None:
    """Serie con < 2 timesteps degrada a DataFrame vacio con esquema."""
    ts = _synthetic_timeseries(n_timesteps=24).isel(time=[0])
    df = cloud_gap_robustness(_extract_ndwi_evi_ndvi, ts, mask_fractions=(0.0,))
    assert df.height == 0
    assert "drift_from_baseline" in df.columns


# ---------------------------------------------------------------------------
# Tests empiricos (requieren data/PASTIS-R/)
# ---------------------------------------------------------------------------


@pytest.mark.empirical
def test_boundary_interior_stats_on_real_pastis() -> None:
    """boundary_interior_stats corre sobre un patch PASTIS-R real."""
    if not (_PASTIS_ROOT / "DATA_S2").exists():
        pytest.skip("data/PASTIS-R/ no disponible")
    from ml.ingest.pastis_loader import load_pastis_patch

    s2_files = sorted((_PASTIS_ROOT / "DATA_S2").glob("S2_*.npy"))
    if not s2_files:
        pytest.skip("Sin patches PASTIS-R en DATA_S2")
    patch_id = s2_files[0].stem.replace("S2_", "")
    patch = load_pastis_patch(patch_id, root=_PASTIS_ROOT)
    stats = boundary_interior_stats(patch, band_index=6)
    assert stats.height == 3
    assert stats.get_column("count").sum() > 0
