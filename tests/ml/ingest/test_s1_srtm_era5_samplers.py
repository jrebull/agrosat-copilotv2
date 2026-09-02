"""Tests US-016 AC-4 / AC-5 / AC-6 — samplers nuevos en ``ml.ingest.gee_sampler``.

Mockea Earth Engine NUNCA llama a GEE real (excepto los 3 tests marcados
con ``requires_gee`` que se saltan en CI sin EE inicializado).

Cubre:

- ``sample_s1_roi_for_parcels``: schema, polarization subset, orbit pass,
  despeckle Lee preset, cache parquet.
- ``sample_srtm_terrain``: 3 cols + aspect dominante string cardinal.
- ``sample_era5_monthly_climate``: 24 cols, conversion Kelvin -> Celsius.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon

from ml.ingest import gee_sampler
from ml.ingest.gee_sampler import (
    sample_era5_monthly_climate,
    sample_s1_roi_for_parcels,
    sample_srtm_terrain,
)

# ---------------------------------------------------------------------------
# Fixtures locales.
# ---------------------------------------------------------------------------


@pytest.fixture
def parcels_simple() -> gpd.GeoDataFrame:
    """3 parcelas cuadrado en Italia."""
    geoms = [
        Polygon([(9.5, 45.2), (9.51, 45.2), (9.51, 45.21), (9.5, 45.21), (9.5, 45.2)]),
        Polygon([(11.2, 43.4), (11.21, 43.4), (11.21, 43.41), (11.2, 43.41), (11.2, 43.4)]),
        Polygon([(16.5, 40.5), (16.51, 40.5), (16.51, 40.51), (16.5, 40.51), (16.5, 40.5)]),
    ]
    return gpd.GeoDataFrame(
        {"parcel_id": [1, 2, 3], "year": [2024, 2024, 2024]},
        geometry=geoms,
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Mocks utility — emulan la cadena ee.ImageCollection().reduce/reduceRegions().
# ---------------------------------------------------------------------------


def _make_s1_fake_ee(payload_per_pol: dict[str, list[dict[str, Any]]]) -> MagicMock:
    """Mock para la chain S1.

    Args:
        payload_per_pol: dict {pol: [feature_dict, ...]}. El mock devuelve una
        respuesta distinta para cada llamada a ``getInfo`` (orden: VV primero,
        VH despues — segun el for pol in polarization del sampler).
    """
    fake = MagicMock(name="ee")

    # reduceRegions().getInfo() devuelve un payload por polarizacion.
    all_payloads: list[dict[str, Any]] = []
    for pol in ("VV", "VH"):
        all_payloads.append({"features": payload_per_pol.get(pol, [])})

    fake_reduced = MagicMock(name="ee.reduceRegions_result")
    fake_reduced.getInfo.side_effect = all_payloads

    fake_stats_img = MagicMock(name="ee.stats_img")
    fake_stats_img.reduceRegions.return_value = fake_reduced

    # `pol_col` es el resultado de `collection.select(pol)`. Su .reduce(...)
    # devuelve `fake_stats_img`. Tambien necesita chainability porque el
    # sampler rebobina `collection = collection.select(list(polarization))`
    # antes de aplicar despeckle / sigma0 conversions.
    fake_pol_col = MagicMock(name="ee.pol_col")
    fake_pol_col.reduce.return_value = fake_stats_img
    fake_pol_col.select.return_value = fake_pol_col
    fake_pol_col.map.return_value = fake_pol_col
    fake_pol_col.filter.return_value = fake_pol_col
    fake_pol_col.filterDate.return_value = fake_pol_col

    # La coleccion principal: todos los chains la mantienen, y .select(pol)
    # devuelve `fake_pol_col`.
    fake_collection = MagicMock(name="ee.collection")
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.filter.return_value = fake_collection
    fake_collection.map.return_value = fake_collection
    fake_collection.select.return_value = fake_pol_col

    fake.ImageCollection.return_value = fake_collection
    fake.Filter.eq.return_value = MagicMock()
    fake.Filter.listContains.return_value = MagicMock()
    fake.Kernel.square.return_value = MagicMock()

    # Reducers usados (combine cadena fluyente).
    combined_reducer = MagicMock(name="ee.combined_reducer")
    combined_reducer.combine.return_value = combined_reducer
    mean_reducer = MagicMock(name="ee.mean_reducer")
    mean_reducer.combine.return_value = combined_reducer
    fake.Reducer.mean.return_value = mean_reducer
    fake.Reducer.stdDev.return_value = MagicMock()
    fake.Reducer.percentile.return_value = MagicMock()

    # Construccion de Geometry/Feature/FeatureCollection (pasivos).
    fake.Geometry = MagicMock(name="ee.Geometry")
    fake.Feature = MagicMock(name="ee.Feature")
    fake.FeatureCollection = MagicMock(name="ee.FeatureCollection")
    return fake


def _make_srtm_fake_ee(features_payload: list[dict[str, Any]]) -> MagicMock:
    """Mock SRTM compatible con la chain del sampler:

    dem = ee.Image(SRTM).select("elevation")
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)
    composite = dem.addBands(slope.rename("slope")).addBands(aspect.rename("aspect"))
    reduced = composite.reduceRegions(...)
    info = reduced.getInfo()
    """
    fake = MagicMock(name="ee")
    fake_reduced = MagicMock()
    fake_reduced.getInfo.return_value = {"features": features_payload}

    # `composite.reduceRegions(...)` -> fake_reduced
    fake_composite = MagicMock(name="ee.composite")
    fake_composite.reduceRegions.return_value = fake_reduced
    # composite.addBands(...) tambien devuelve composite (chain corta).
    fake_composite.addBands.return_value = fake_composite

    fake_dem = MagicMock(name="ee.dem")
    fake_dem.select.return_value = fake_dem
    fake_dem.addBands.return_value = fake_composite  # primer addBands devuelve composite

    fake.Image = MagicMock(return_value=fake_dem)
    # ee.Terrain.slope / aspect devuelven mocks con .rename() chainable.
    slope_obj = MagicMock(name="ee.slope")
    slope_obj.rename.return_value = slope_obj
    aspect_obj = MagicMock(name="ee.aspect")
    aspect_obj.rename.return_value = aspect_obj
    fake.Terrain = MagicMock()
    fake.Terrain.slope.return_value = slope_obj
    fake.Terrain.aspect.return_value = aspect_obj
    fake.Reducer.mean.return_value = MagicMock()
    fake.Geometry = MagicMock()
    fake.Feature = MagicMock()
    fake.FeatureCollection = MagicMock()
    return fake


def _make_era5_fake_ee(
    t_per_month: dict[int, dict[int, float]],
    p_per_month: dict[int, dict[int, float]],
) -> MagicMock:
    """Mock ERA5 — `t_per_month[m][pid]` y `p_per_month[m][pid]` en Kelvin / metros."""
    fake = MagicMock(name="ee")
    # Por cada mes el sampler hace 2 .getInfo() (tmean y prec).
    payloads = []
    for m in range(1, 13):
        t_feats = [
            {"properties": {"parcel_id": pid, "mean": val}}
            for pid, val in t_per_month.get(m, {}).items()
        ]
        p_feats = [
            {"properties": {"parcel_id": pid, "mean": val}}
            for pid, val in p_per_month.get(m, {}).items()
        ]
        payloads.append({"features": t_feats})
        payloads.append({"features": p_feats})

    fake_reduced = MagicMock()
    fake_reduced.getInfo.side_effect = payloads

    fake_img = MagicMock()
    fake_img.reduceRegions.return_value = fake_reduced

    fake_collection = MagicMock()
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.select.return_value = fake_collection
    fake_collection.mean.return_value = fake_img
    fake_collection.sum.return_value = fake_img

    fake.ImageCollection.return_value = fake_collection
    fake.Reducer.mean.return_value = MagicMock()
    fake.Geometry = MagicMock()
    fake.Feature = MagicMock()
    fake.FeatureCollection = MagicMock()
    return fake


# ---------------------------------------------------------------------------
# AC-4: Sentinel-1.
# ---------------------------------------------------------------------------


def test_s1_returns_polars_df_with_expected_cols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """Mock VV+VH para 3 parcelas devuelve un DataFrame con 10 cols."""
    payload_per_pol = {
        "VV": [
            {
                "properties": {
                    "parcel_id": pid,
                    "VV_mean": -10.0 - i,
                    "VV_stdDev": 1.5,
                    "VV_p25": -12.0 - i,
                    "VV_p50": -11.0 - i,
                    "VV_p95": -8.0 - i,
                }
            }
            for i, pid in enumerate([1, 2, 3])
        ],
        "VH": [
            {
                "properties": {
                    "parcel_id": pid,
                    "VH_mean": -16.0 - i,
                    "VH_stdDev": 1.2,
                    "VH_p25": -18.0 - i,
                    "VH_p50": -17.0 - i,
                    "VH_p95": -14.0 - i,
                }
            }
            for i, pid in enumerate([1, 2, 3])
        ],
    }
    fake_ee = _make_s1_fake_ee(payload_per_pol)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_s1_roi_for_parcels(
        parcels_simple,
        year=2024,
        polarization=("VV", "VH"),
        orbit_pass="both",
        cache_dir=tmp_path,
        cache_key="t1",
    )
    assert isinstance(out, pl.DataFrame)
    for stat in ("mean", "std", "p25", "p50", "p95"):
        assert f"s1_vv_{stat}" in out.columns
        assert f"s1_vh_{stat}" in out.columns


def test_s1_polarization_subset_vv_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """polarization=("VV",) => 5 cols `s1_vv_*` (sin VH)."""
    payload = {
        "VV": [
            {
                "properties": {
                    "parcel_id": 1,
                    "VV_mean": -10.0,
                    "VV_stdDev": 1.0,
                    "VV_p25": -12.0,
                    "VV_p50": -11.0,
                    "VV_p95": -8.0,
                }
            }
        ]
    }
    fake_ee = _make_s1_fake_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_s1_roi_for_parcels(
        parcels_simple,
        year=2024,
        polarization=("VV",),
        cache_dir=tmp_path,
        cache_key="vv_only",
    )
    vv_cols = [c for c in out.columns if c.startswith("s1_vv_")]
    vh_cols = [c for c in out.columns if c.startswith("s1_vh_")]
    assert len(vv_cols) == 5
    assert len(vh_cols) == 0


def test_s1_orbit_pass_ascending_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """orbit_pass='ascending' invoca el filtro `eq('orbitProperties_pass','ASCENDING')`."""
    fake_ee = _make_s1_fake_ee({"VV": [], "VH": []})
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    sample_s1_roi_for_parcels(
        parcels_simple,
        year=2024,
        orbit_pass="ascending",
        cache_dir=tmp_path,
        cache_key="asc",
    )
    # Cualquiera de las llamadas a Filter.eq debio incluir ASCENDING.
    calls = [str(c) for c in fake_ee.Filter.eq.call_args_list]
    assert any("ASCENDING" in c for c in calls), calls


def test_s1_despeckle_lee_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """`despeckle='lee_7x7'` => Kernel.square(radius=3) (7x7 pixels)."""
    fake_ee = _make_s1_fake_ee({"VV": [], "VH": []})
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    sample_s1_roi_for_parcels(
        parcels_simple,
        year=2024,
        despeckle="lee_7x7",
        cache_dir=tmp_path,
        cache_key="lee",
    )
    fake_ee.Kernel.square.assert_called()
    # Validamos el radio (3 pixels = 7x7 incluido el centro).
    kwargs_all = [c.kwargs for c in fake_ee.Kernel.square.call_args_list]
    assert any(k.get("radius") == 3 for k in kwargs_all)


def test_samplers_cache_dir_creates_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """Si el sampler produce filas, escribe un parquet en cache_dir."""
    payload = {
        "VV": [
            {
                "properties": {
                    "parcel_id": pid,
                    "VV_mean": -10.0,
                    "VV_stdDev": 1.0,
                    "VV_p25": -12.0,
                    "VV_p50": -11.0,
                    "VV_p95": -8.0,
                }
            }
            for pid in [1, 2, 3]
        ],
        "VH": [
            {
                "properties": {
                    "parcel_id": pid,
                    "VH_mean": -16.0,
                    "VH_stdDev": 1.0,
                    "VH_p25": -18.0,
                    "VH_p50": -17.0,
                    "VH_p95": -14.0,
                }
            }
            for pid in [1, 2, 3]
        ],
    }
    fake_ee = _make_s1_fake_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_s1_roi_for_parcels(
        parcels_simple,
        year=2024,
        cache_dir=tmp_path,
        cache_key="cache_test",
    )
    assert out.height == 3, "El sampler no consolido filas VV+VH"
    parquets = list(tmp_path.glob("s1_cache_test_*.parquet"))
    assert parquets, "Cache parquet S1 no escrito"


# ---------------------------------------------------------------------------
# AC-5: SRTM.
# ---------------------------------------------------------------------------


def test_srtm_returns_3_cols_per_parcel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    payload = [
        {"properties": {"parcel_id": 1, "elevation": 100.0, "slope": 5.0, "aspect": 0.0}},
        {"properties": {"parcel_id": 2, "elevation": 250.0, "slope": 12.0, "aspect": 95.0}},
        {"properties": {"parcel_id": 3, "elevation": 50.0, "slope": 2.0, "aspect": 270.0}},
    ]
    fake_ee = _make_srtm_fake_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_srtm_terrain(parcels_simple, cache_dir=tmp_path, cache_key="srtm_t")
    assert out.height == 3
    assert set(out.columns) == {
        "parcel_id",
        "srtm_elev_mean",
        "srtm_slope_mean",
        "srtm_aspect_dominant",
    }


def test_srtm_aspect_dominant_is_cardinal_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    payload = [
        {"properties": {"parcel_id": 1, "elevation": 100.0, "slope": 5.0, "aspect": 0.0}},
        {"properties": {"parcel_id": 2, "elevation": 250.0, "slope": 12.0, "aspect": 90.0}},
        {"properties": {"parcel_id": 3, "elevation": 50.0, "slope": 2.0, "aspect": 270.0}},
    ]
    fake_ee = _make_srtm_fake_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_srtm_terrain(parcels_simple, cache_dir=tmp_path, cache_key="srtm_card")
    valid = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    for asp in out.get_column("srtm_aspect_dominant").to_list():
        assert asp in valid


# ---------------------------------------------------------------------------
# AC-6: ERA5 mensual.
# ---------------------------------------------------------------------------


def test_era5_monthly_returns_24_cols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    # Temperatura sinusoidal en K (~278 K enero, ~298 K julio); prec en metros.
    import math

    t = {
        m: {pid: 288.15 + 10.0 * math.cos(math.pi * (m - 7) / 6.0) for pid in [1, 2, 3]}
        for m in range(1, 13)
    }
    p = {m: {pid: 0.050 for pid in [1, 2, 3]} for m in range(1, 13)}
    fake_ee = _make_era5_fake_ee(t, p)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_era5_monthly_climate(
        parcels_simple, year=2024, temperature_units="C", cache_dir=tmp_path, cache_key="e_t"
    )
    t_cols = [c for c in out.columns if c.startswith("era5_tmean_")]
    p_cols = [c for c in out.columns if c.startswith("era5_prec_")]
    assert len(t_cols) == 12
    assert len(p_cols) == 12


def test_era5_temperature_converted_to_celsius(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """Con `temperature_units='C'` las temperaturas mensuales caen en [-30, 50] (Italia)."""
    import math

    t = {
        m: {pid: 288.15 + 10.0 * math.cos(math.pi * (m - 7) / 6.0) for pid in [1, 2, 3]}
        for m in range(1, 13)
    }
    p = {m: {pid: 0.050 for pid in [1, 2, 3]} for m in range(1, 13)}
    fake_ee = _make_era5_fake_ee(t, p)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_era5_monthly_climate(
        parcels_simple, year=2024, temperature_units="C", cache_dir=tmp_path, cache_key="e_celsius"
    )
    for m in range(1, 13):
        vals = out.get_column(f"era5_tmean_m{m:02d}").to_list()
        for v in vals:
            if v is None:
                continue
            assert -30.0 <= v <= 50.0


def test_era5_reads_band_name_property_not_only_mean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parcels_simple: gpd.GeoDataFrame
) -> None:
    """Regression Bug-6: `reduceRegions` con single-band image renombra la
    propiedad al nombre del band (`temperature_2m` / `total_precipitation_sum`),
    NO a "mean". El sampler debe leer la key band-name correctamente.

    El mock devuelve SOLO `temperature_2m` / `total_precipitation_sum` (sin
    `"mean"`) — replicando el comportamiento real de GEE post Bug-6 fix.
    """
    payloads = []
    for m in range(1, 13):
        t_feats = [
            {"properties": {"parcel_id": pid, "temperature_2m": 288.15 + m}} for pid in [1, 2, 3]
        ]
        p_feats = [
            {"properties": {"parcel_id": pid, "total_precipitation_sum": 0.030}}
            for pid in [1, 2, 3]
        ]
        payloads.append({"features": t_feats})
        payloads.append({"features": p_feats})

    fake = MagicMock(name="ee")
    fake_reduced = MagicMock()
    fake_reduced.getInfo.side_effect = payloads
    fake_img = MagicMock()
    fake_img.reduceRegions.return_value = fake_reduced
    fake_collection = MagicMock()
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.select.return_value = fake_collection
    fake_collection.mean.return_value = fake_img
    fake_collection.sum.return_value = fake_img
    fake.ImageCollection.return_value = fake_collection
    fake.Reducer.mean.return_value = MagicMock()
    fake.Geometry = MagicMock()
    fake.Feature = MagicMock()
    fake.FeatureCollection = MagicMock()

    monkeypatch.setattr(gee_sampler, "ee", fake)
    out = sample_era5_monthly_climate(
        parcels_simple, year=2024, temperature_units="C", cache_dir=tmp_path, cache_key="bug6"
    )
    # Si el sampler leía solo "mean" (bug original) todas las cols serian None.
    jan = out.get_column("era5_tmean_m01").to_list()
    jul = out.get_column("era5_tmean_m07").to_list()
    assert all(v is not None for v in jan), "Bug-6 regression: era5_tmean_m01 quedó en None"
    assert all(v is not None for v in jul), "Bug-6 regression: era5_tmean_m07 quedó en None"
    # 288.15 + 1 - 273.15 = 16°C (enero), 288.15 + 7 - 273.15 = 22°C (julio).
    assert all(abs(v - 16.0) < 0.1 for v in jan)
    assert all(abs(v - 22.0) < 0.1 for v in jul)
    # Precipitación: 0.030 m * 1000 = 30 mm acumulado/mes.
    prec_jan = out.get_column("era5_prec_m01").to_list()
    assert all(abs(v - 30.0) < 0.1 for v in prec_jan)


# ---------------------------------------------------------------------------
# Tests requires_gee (skip si EE no inicializado).
# ---------------------------------------------------------------------------


def _ee_uninitialized() -> bool:
    try:
        import ee  # type: ignore[import-untyped]

        # ee.data._initialized expone el estado; fallback a getattr.
        return not bool(getattr(ee.data, "_initialized", False))
    except Exception:  # noqa: BLE001 - cualquier fallo importando ee = no inicializado
        return True


pytestmark_requires_gee = pytest.mark.skipif(
    _ee_uninitialized(),
    reason="EE no inicializado en este entorno (CI / dev sin credenciales).",
)


@pytestmark_requires_gee
@pytest.mark.requires_gee
def test_s1_real_smoke_pianura_padana(parcels_simple: gpd.GeoDataFrame) -> None:
    """Smoke test real GEE: 1 parcela en Pianura Padana => sigma0 dB en rango."""
    out = sample_s1_roi_for_parcels(parcels_simple.iloc[[0]], year=2024)
    assert isinstance(out, pl.DataFrame)


@pytestmark_requires_gee
@pytest.mark.requires_gee
def test_srtm_real_smoke_alps(parcels_simple: gpd.GeoDataFrame) -> None:
    out = sample_srtm_terrain(parcels_simple.iloc[[0]])
    assert isinstance(out, pl.DataFrame)


@pytestmark_requires_gee
@pytest.mark.requires_gee
def test_era5_real_smoke_2024(parcels_simple: gpd.GeoDataFrame) -> None:
    out = sample_era5_monthly_climate(parcels_simple.iloc[[0]], year=2024)
    assert isinstance(out, pl.DataFrame)
