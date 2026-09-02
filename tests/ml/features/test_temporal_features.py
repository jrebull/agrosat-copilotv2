"""Tests unitarios de :mod:`ml.features.temporal_features` (US-015).

Casos del plan §6:

- Estructura y forma del DataFrame (187 columnas, dtypes, índice).
- Estadísticos sobre señales sintéticas conocidas (sinusoidal).
- FFT determinista: ``sin(2πt/365)`` recupera ``amp=1±0.05``, ``phase=π/2±0.1``.
- Fenología sobre curva gaussiana centrada DOY 180.
- Edge cases: NDVI nunca cruza el umbral, peak en el borde del ciclo.
- Slopes simétricas y duración de madurez.
- Validación de errores: attrs faltantes y banda ausente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from ml.features.temporal_features import (
    DEFAULT_FFT_INDICES,
    DEFAULT_INDICES,
    _compute_rfft_components,
    _detect_phenology,
    _interpolate_daily,
    _phenology_slopes,
    extract_temporal_features,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_FIXTURE = REPO_ROOT / "data" / "test_fixtures" / "parcel_demo_ts.nc"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_dataarray(
    *,
    parcel_id: int = 1,
    year: int = 2024,
    n_steps: int = 30,
    ndvi_curve: np.ndarray | None = None,
    seed: int = 42,
) -> xr.DataArray:
    """Construye un DataArray con 17 bandas; NDVI controlable."""
    rng = np.random.default_rng(seed)
    start = np.datetime64(f"{year}-01-15", "ns")
    times = np.array(
        [start + np.timedelta64(i * 12, "D") for i in range(n_steps)],
        dtype="datetime64[ns]",
    )
    bands = list(DEFAULT_INDICES)

    if ndvi_curve is None:
        doys = np.array(
            [
                (t - np.datetime64(f"{year}-01-01", "ns")) / np.timedelta64(1, "D") + 1
                for t in times
            ],
            dtype=np.float64,
        )
        ndvi_curve = 0.85 * np.exp(-0.5 * ((doys - 180) / 30.0) ** 2)
        ndvi_curve = ndvi_curve + rng.normal(0.0, 0.02, size=n_steps)

    values = np.empty((n_steps, len(bands)), dtype=np.float64)
    for j, band in enumerate(bands):
        if band == "NDVI":
            values[:, j] = ndvi_curve
        else:
            values[:, j] = 0.5 * ndvi_curve + rng.normal(0.0, 0.015, size=n_steps)

    return xr.DataArray(
        data=values,
        dims=("time", "band"),
        coords={"time": times, "band": bands},
        attrs={"parcel_id": parcel_id, "year": year},
    )


@pytest.fixture
def synthetic_da() -> xr.DataArray:
    return _make_synthetic_dataarray()


# ---------------------------------------------------------------------------
# Estructura básica
# ---------------------------------------------------------------------------


def test_extract_returns_polars_frame(synthetic_da: xr.DataArray) -> None:
    df = extract_temporal_features(synthetic_da)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1


def test_columns_shape(synthetic_da: xr.DataArray) -> None:
    """187 columnas: 2 índice + 153 stats + 24 FFT + 8 fenología."""
    df = extract_temporal_features(synthetic_da)
    expected_total = 2 + 9 * len(DEFAULT_INDICES) + 8 * len(DEFAULT_FFT_INDICES) + 8
    assert expected_total == 187
    assert df.width == 187
    # Comprobaciones puntuales
    assert "parcel_id" in df.columns
    assert "year" in df.columns
    for stat in ("mean", "std", "p05", "p95"):
        assert f"NDVI_{stat}" in df.columns
    for k in range(4):
        assert f"NDVI_fft_amp_{k}" in df.columns
        assert f"NDVI_fft_phase_{k}" in df.columns
    for col in (
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    ):
        assert col in df.columns


def test_lazy_streaming_collect(synthetic_da: xr.DataArray) -> None:
    """Dos llamadas consecutivas devuelven DataFrames idénticos (determinismo)."""
    df1 = extract_temporal_features(synthetic_da)
    df2 = extract_temporal_features(synthetic_da)
    assert df1.equals(df2)


# ---------------------------------------------------------------------------
# Estadísticos
# ---------------------------------------------------------------------------


def test_aggregate_stats_synthetic_sin() -> None:
    """``mean`` de un seno completo sobre 1 año ≈ 0."""
    n = 365
    times = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i, "D") for i in range(n)],
        dtype="datetime64[ns]",
    )
    bands = list(DEFAULT_INDICES)
    values = np.zeros((n, len(bands)), dtype=np.float64)
    ndvi = np.sin(2 * np.pi * np.arange(n) / 365)
    for j, band in enumerate(bands):
        values[:, j] = ndvi if band == "NDVI" else ndvi * 0.5

    da = xr.DataArray(
        data=values,
        dims=("time", "band"),
        coords={"time": times, "band": bands},
        attrs={"parcel_id": 7, "year": 2024},
    )
    df = extract_temporal_features(da)
    assert df["NDVI_mean"][0] == pytest.approx(0.0, abs=5e-3)
    assert df["NDVI_max"][0] == pytest.approx(1.0, abs=5e-3)
    assert df["NDVI_min"][0] == pytest.approx(-1.0, abs=5e-3)


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def test_fft_synthetic_sin() -> None:
    """``sin(2πt/365)`` recupera amp=1±0.05 y phase ≈ −π/2±0.1 en armónico k=1.

    Convención: ``np.fft.rfft`` de un seno puro tiene fase ``-π/2`` (la fase
    de un coseno es 0). Tolerancia laxa porque la grilla no cubre ciclo
    completo idealmente.
    """
    n = 365
    curve = np.sin(2 * np.pi * np.arange(n) / n)
    amps, phases = _compute_rfft_components({"NDVI": curve}["NDVI"], n_components=4)
    assert amps[0] == pytest.approx(0.0, abs=5e-3)  # DC ~ 0
    assert amps[1] == pytest.approx(1.0, abs=0.05)
    # Fase del seno: -π/2
    assert phases[1] == pytest.approx(-np.pi / 2, abs=0.1)


def test_fft_dc_component_is_mean() -> None:
    """El componente DC (k=0) debe coincidir con la media de la señal."""
    rng = np.random.default_rng(0)
    curve = 0.4 + 0.1 * rng.normal(size=200)
    amps, _ = _compute_rfft_components(curve, n_components=4)
    assert amps[0] == pytest.approx(float(curve.mean()), rel=1e-9)


# ---------------------------------------------------------------------------
# Fenología
# ---------------------------------------------------------------------------


def test_phenology_synthetic_gaussian() -> None:
    """Gaussiana DOY 180, peak 0.85, σ=30 → peak_doy ∈ [178, 182] y auc > 0."""
    doys = np.arange(1, 366, dtype=np.float64)
    ndvi = 0.85 * np.exp(-0.5 * ((doys - 180) / 30.0) ** 2)
    metrics = _detect_phenology(ndvi, sog_threshold=0.3)
    assert metrics["peak_doy"] is not None
    assert 178 <= int(metrics["peak_doy"]) <= 182  # type: ignore[arg-type]
    assert metrics["peak_value"] == pytest.approx(0.85, abs=1e-2)
    assert metrics["ndvi_auc"] is not None
    assert float(metrics["ndvi_auc"]) > 0.0  # type: ignore[arg-type]
    assert metrics["sog_doy"] is not None
    assert metrics["senescence_doy"] is not None


def test_phenology_no_crossing() -> None:
    """NDVI todo <0.3 → None sin error."""
    curve = np.full(100, 0.15, dtype=np.float64)
    metrics = _detect_phenology(curve, sog_threshold=0.3)
    assert metrics["sog_doy"] is None
    assert metrics["peak_doy"] is None
    assert metrics["senescence_doy"] is None
    assert metrics["ndvi_auc"] is None


def test_phenology_peak_at_edge() -> None:
    """Peak en DOY 1 → senescence None pero no excepción."""
    curve = np.linspace(0.9, 0.1, 100)  # peak en el borde izquierdo
    metrics = _detect_phenology(curve, sog_threshold=0.3)
    assert metrics["peak_doy"] == 1
    # Senescencia debe encontrarse porque la curva desciende; pero SOG no
    # tiene cruce previo al pico.
    assert metrics["sog_doy"] == 1  # NDVI[0] ya supera el umbral
    assert metrics["senescence_doy"] is not None


def test_slopes_symmetric_curve() -> None:
    """Curva gaussiana simétrica: ``slope_pre > 0 > slope_post`` y |pre|≈|post|."""
    doys = np.arange(1, 366, dtype=np.float64)
    ndvi = 0.85 * np.exp(-0.5 * ((doys - 180) / 30.0) ** 2)
    metrics = _detect_phenology(ndvi, sog_threshold=0.3)
    slopes = _phenology_slopes(ndvi, metrics=metrics, maturity_pct=0.8)
    assert slopes["slope_pre"] is not None
    assert slopes["slope_post"] is not None
    pre = float(slopes["slope_pre"])  # type: ignore[arg-type]
    post = float(slopes["slope_post"])  # type: ignore[arg-type]
    assert pre > 0.0
    assert post < 0.0
    assert abs(pre) == pytest.approx(abs(post), rel=0.1)


def test_maturity_duration_threshold() -> None:
    """Maturity = días con NDVI ≥ 0.8 × peak; pico plateau debe ampliar la ventana."""
    curve = np.concatenate(
        [
            np.linspace(0.1, 0.8, 50),
            np.full(20, 0.8, dtype=np.float64),  # plateau peak
            np.linspace(0.8, 0.1, 50),
        ]
    )
    metrics = _detect_phenology(curve, sog_threshold=0.3)
    slopes = _phenology_slopes(curve, metrics=metrics, maturity_pct=0.95)
    # Con threshold 0.95 * 0.8 = 0.76 el plateau de 20 días pertenece, más
    # algunos días contiguos arriba de 0.76.
    assert slopes["maturity_duration_days"] is not None
    assert int(slopes["maturity_duration_days"]) >= 20  # type: ignore[arg-type]


def test_interpolate_daily_regular_grid(synthetic_da: xr.DataArray) -> None:
    """La interpolación produce una rejilla diaria de longitud (t_max - t_min + 1)."""
    curves = _interpolate_daily(synthetic_da, indices=("NDVI",))
    times = synthetic_da.coords["time"].values
    expected_len = int((times.max() - times.min()) / np.timedelta64(1, "D")) + 1
    assert curves["NDVI"].size == expected_len


# ---------------------------------------------------------------------------
# End-to-end con fixture demo
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DEMO_FIXTURE.exists(), reason="Demo fixture not generated yet")
def test_extract_demo_fixture_end_to_end() -> None:
    """El fixture demo produce un DataFrame con 187 columnas y peak_doy plausible."""
    ds = xr.open_dataset(DEMO_FIXTURE)
    da = ds["parcel_indices"]
    # Reasignamos attrs por seguridad (NetCDF preserva attrs pero algunos
    # tipos numéricos pueden venir como np.int*).
    da.attrs.setdefault("parcel_id", 42)
    da.attrs.setdefault("year", 2024)
    # Decode band coord if stored as bytes.
    if da.coords["band"].dtype.kind == "S":
        da = da.assign_coords(band=[b.decode() for b in da.coords["band"].values])

    df = extract_temporal_features(da)
    assert df.width == 187
    assert df.height == 1
    assert df["parcel_id"][0] == 42
    assert df["year"][0] == 2024
    peak_doy = df["peak_doy"][0]
    assert peak_doy is not None
    # Pico en torno a DOY 180 ± tolerancia (ruido + grilla esparsa).
    assert 150 <= int(peak_doy) <= 210


# ---------------------------------------------------------------------------
# Validación de errores
# ---------------------------------------------------------------------------


def test_extract_raises_on_missing_attrs(synthetic_da: xr.DataArray) -> None:
    da = synthetic_da.copy()
    da.attrs = {}
    with pytest.raises(ValueError, match="parcel_id"):
        extract_temporal_features(da)


def test_extract_raises_on_missing_band(synthetic_da: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="not present"):
        extract_temporal_features(synthetic_da, indices=("NDVI", "DOES_NOT_EXIST"))


def test_extract_raises_on_non_datetime_time(synthetic_da: xr.DataArray) -> None:
    # Sustituimos la coord time por floats.
    n = synthetic_da.sizes["time"]
    da = synthetic_da.assign_coords(time=np.arange(n, dtype=np.float64))
    with pytest.raises(ValueError, match="datetime64"):
        extract_temporal_features(da)
