"""Tests del catálogo canónico de los 17 índices espectrales (US-014).

Suite organizada en grupos A-H según `docs/us-planning/us-014.md` §6.2:

- A: Estructura y registro (4 tests)
- B: API principal (6 tests)
- C: Valores agronómicos conocidos por índice (17 tests)
- D: Series temporales y reduce (4 tests)
- E: Earth Engine wrapper (2 tests)
- F: Cache Redis (4 tests, fakeredis)
- G: Smoke test PASTIS-R real (1 test, skip-graceful)
- H: Documentación cruzada (1 test)
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ml.features import (
    INDEX_NAMES,
    compute_index,
    compute_index_cached,
    compute_index_ee,
    compute_index_timeseries,
)
from ml.features.spectral_indices import (
    _BAND_TO_SPYNDEX,
    _INDEX_REGISTRY,
)
from ml.ingest.pastis_loader import PASTIS_S2_BANDS
from tests.ml.features.fixtures.synthetic_pixels import (
    make_bare_soil_pixel,
    make_forest_pixel,
    make_synthetic_s2_dataarray,
    make_synthetic_s2_timeseries,
    make_water_pixel,
)

# ---------------------------------------------------------------------------
# Grupo A — Estructura y registro
# ---------------------------------------------------------------------------


def test_all_17_indices_in_INDEX_NAMES() -> None:
    """INDEX_NAMES contiene exactamente los 17 índices del plan v6 línea 1104."""
    expected = [
        "NDVI",
        "NDWI",
        "NDMI",
        "EVI",
        "SAVI",
        "MSAVI2",
        "NBR",
        "MCARI",
        "CCCI",
        "LAI",
        "FAPAR",
        "PSRI",
        "NDCI",
        "GCVI",
        "RENDVI",
        "NDRE",
        "TSAVI",
    ]
    assert INDEX_NAMES == expected
    assert len(INDEX_NAMES) == 17


def test_all_17_indices_in_registry() -> None:
    """El registro canónico cubre los 17 índices declarados en INDEX_NAMES."""
    assert set(_INDEX_REGISTRY.keys()) == set(INDEX_NAMES)
    assert len(_INDEX_REGISTRY) == 17


def test_band_naming_convention_matches_pastis_loader() -> None:
    """El mapeo _BAND_TO_SPYNDEX cubre todas las bandas de PASTIS_S2_BANDS."""
    assert set(_BAND_TO_SPYNDEX.keys()) == set(PASTIS_S2_BANDS)
    # No alias colisionados al mapear
    assert len(set(_BAND_TO_SPYNDEX.values())) == len(PASTIS_S2_BANDS)


def test_registry_entries_have_required_metadata() -> None:
    """Cada entrada del registro tiene fórmula, uso agronómico y referencia."""
    for name, entry in _INDEX_REGISTRY.items():
        assert entry.formula, f"{name}: falta fórmula"
        assert entry.agronomic_use, f"{name}: falta uso agronómico"
        assert entry.reference, f"{name}: falta referencia"
        assert entry.required_bands, f"{name}: faltan bandas requeridas"
        if entry.backend == "spyndex":
            assert entry.spyndex_name is not None
        else:
            assert entry.custom_fn is not None


# ---------------------------------------------------------------------------
# Grupo B — API principal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", INDEX_NAMES)
def test_compute_index_returns_dataarray_for_all_17(index: str) -> None:
    """Cada uno de los 17 índices devuelve un DataArray sin NaN inesperados."""
    da = make_synthetic_s2_dataarray(shape=(4, 4), archetype="forest")
    result = compute_index(da, index)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == np.float32
    # dims espaciales preservadas
    assert "y" in result.dims and "x" in result.dims
    # no NaN sobre fixture forest (no hay máscara/sombras)
    assert not bool(result.isnull().any())


def test_compute_index_raises_value_error_on_unknown_index() -> None:
    da = make_forest_pixel()
    with pytest.raises(ValueError, match="Índice desconocido"):
        compute_index(da, "FOOBAR")


def test_compute_index_raises_key_error_on_missing_band() -> None:
    """NDMI requiere B11; un DataArray sin B11 lanza KeyError."""
    da_full = make_synthetic_s2_dataarray(shape=(2, 2), archetype="forest")
    da_partial = da_full.sel(band=[b for b in PASTIS_S2_BANDS if b != "B11"])
    with pytest.raises(KeyError, match="B11"):
        compute_index(da_partial, "NDMI")


def test_compute_index_raises_value_error_without_band_dim() -> None:
    arr = xr.DataArray(np.zeros((4, 4), dtype=np.float32), dims=("y", "x"))
    with pytest.raises(ValueError, match="dimensión 'band'"):
        compute_index(arr, "NDVI")


def test_compute_index_preserves_spatial_coords() -> None:
    """Las coords espaciales del input se conservan en el output."""
    da = make_synthetic_s2_dataarray(shape=(3, 5), archetype="dense_canopy")
    result = compute_index(da, "NDVI")
    np.testing.assert_array_equal(result.coords["x"].values, da.coords["x"].values)
    np.testing.assert_array_equal(result.coords["y"].values, da.coords["y"].values)


def test_compute_index_attrs_include_reference() -> None:
    """``result.attrs['reference']`` cita autor + año."""
    da = make_forest_pixel()
    result = compute_index(da, "NDVI")
    assert "reference" in result.attrs
    assert "Rouse" in result.attrs["reference"]
    assert result.attrs["index_name"] == "NDVI"
    assert result.attrs["computed_with"].startswith("spyndex")


# ---------------------------------------------------------------------------
# Grupo C — Valores agronómicos conocidos (17 tests, uno por índice)
# ---------------------------------------------------------------------------


def test_ndvi_forest_pixel_in_0p7_0p9() -> None:
    """CA US-014 plan v6 línea 1108: NDVI bosque caducifolio junio ∈ [0.7, 0.9]."""
    value = float(compute_index(make_forest_pixel(), "NDVI").mean())
    assert 0.7 <= value <= 0.9, f"NDVI forest={value} fuera de [0.7, 0.9]"


def test_ndvi_bare_soil_pixel_below_0p2() -> None:
    value = float(compute_index(make_bare_soil_pixel(), "NDVI").mean())
    assert 0.0 <= value < 0.2, f"NDVI bare_soil={value} fuera de [0.0, 0.2)"


def test_ndwi_water_pixel_above_0p5() -> None:
    value = float(compute_index(make_water_pixel(), "NDWI").mean())
    assert value > 0.5, f"NDWI water={value} debería ser > 0.5"


def test_ndmi_dry_canopy_negative() -> None:
    da = make_synthetic_s2_dataarray(archetype="dry_canopy")
    value = float(compute_index(da, "NDMI").mean())
    assert value < 0.0, f"NDMI dry_canopy={value} debería ser negativo"


def test_evi_dense_canopy_above_0p4() -> None:
    da = make_synthetic_s2_dataarray(archetype="dense_canopy")
    value = float(compute_index(da, "EVI").mean())
    assert value > 0.4, f"EVI dense_canopy={value} debería ser > 0.4"


def test_savi_sparse_canopy_below_ndvi() -> None:
    """SAVI deprime el efecto suelo en pixeles dispersos → SAVI < NDVI."""
    da = make_synthetic_s2_dataarray(archetype="sparse_canopy")
    savi = float(compute_index(da, "SAVI").mean())
    ndvi = float(compute_index(da, "NDVI").mean())
    assert savi < ndvi, f"SAVI={savi} debería ser < NDVI={ndvi} en suelo expuesto"


def test_msavi2_self_adjusting_positive_on_canopy() -> None:
    da = make_synthetic_s2_dataarray(archetype="dense_canopy")
    value = float(compute_index(da, "MSAVI2").mean())
    assert 0.4 < value <= 1.0, f"MSAVI2 dense_canopy={value} fuera de (0.4, 1.0]"


def test_nbr_burned_pixel_negative() -> None:
    da = make_synthetic_s2_dataarray(archetype="burned")
    value = float(compute_index(da, "NBR").mean())
    assert value < 0.0, f"NBR burned={value} debería ser negativo"


def test_mcari_high_chlorophyll_positive() -> None:
    da = make_synthetic_s2_dataarray(archetype="forest")
    value = float(compute_index(da, "MCARI").mean())
    assert value > 0.0, f"MCARI forest={value} debería ser positivo"


def test_ccci_normalized_in_valid_range() -> None:
    """CCCI = NDRE/NDVI; en vegetación sana debe estar en [0, 1.2]."""
    da = make_forest_pixel()
    value = float(compute_index(da, "CCCI").mean())
    assert 0.0 <= value <= 1.2, f"CCCI forest={value} fuera de [0, 1.2]"


def test_lai_forest_above_2() -> None:
    """Bosque caducifolio: LAI esperado típicamente > 2 (Pettorelli 2005)."""
    value = float(compute_index(make_forest_pixel(), "LAI").mean())
    assert value > 2.0, f"LAI forest={value} debería ser > 2"


def test_fapar_lineal_with_ndvi() -> None:
    """FAPAR = 1.24·NDVI - 0.168; verificación numérica directa."""
    da = make_forest_pixel()
    ndvi = float(compute_index(da, "NDVI").mean())
    fapar = float(compute_index(da, "FAPAR").mean())
    expected = 1.24 * ndvi - 0.168
    assert abs(fapar - expected) < 1e-4


def test_psri_senescent_canopy_positive() -> None:
    da = make_synthetic_s2_dataarray(archetype="senescent")
    value = float(compute_index(da, "PSRI").mean())
    assert value > 0.0, f"PSRI senescent={value} debería ser positivo"


def test_ndci_chla_water_positive() -> None:
    da = make_synthetic_s2_dataarray(archetype="chla_water")
    value = float(compute_index(da, "NDCI").mean())
    assert value > 0.0, f"NDCI chla_water={value} debería ser positivo"


def test_gcvi_dense_canopy_above_3() -> None:
    """GCVI = N/G - 1; canopy denso típicamente > 3."""
    da = make_synthetic_s2_dataarray(archetype="dense_canopy")
    value = float(compute_index(da, "GCVI").mean())
    assert value > 3.0, f"GCVI dense_canopy={value} debería ser > 3"


def test_rendvi_red_edge_in_range() -> None:
    """RENDVI (RE2-RE1)/(RE2+RE1) — esperado positivo en vegetación sana."""
    da = make_synthetic_s2_dataarray(archetype="forest")
    value = float(compute_index(da, "RENDVI").mean())
    assert 0.0 < value < 1.0


def test_ndre_dense_canopy_lower_than_ndvi() -> None:
    """En canopy denso NDVI satura: NDRE < NDVI (sensibilidad red-edge)."""
    da = make_synthetic_s2_dataarray(archetype="dense_canopy")
    ndre = float(compute_index(da, "NDRE").mean())
    ndvi = float(compute_index(da, "NDVI").mean())
    assert ndre < ndvi, f"NDRE={ndre} debería ser < NDVI={ndvi} en canopy denso"


def test_tsavi_calibrated_with_default_soil_line() -> None:
    """Con sla=1, slb=0 (línea de suelo neutral), TSAVI ≈ NDVI sobre vegetación."""
    da = make_forest_pixel()
    tsavi = float(compute_index(da, "TSAVI").mean())
    ndvi = float(compute_index(da, "NDVI").mean())
    # No deben divergir más de 0.05 con parámetros default neutros
    assert abs(tsavi - ndvi) < 0.05


# ---------------------------------------------------------------------------
# Grupo D — Time-series y reduce
# ---------------------------------------------------------------------------


def test_compute_index_timeseries_preserves_time_axis() -> None:
    ts = make_synthetic_s2_timeseries(n_timesteps=6, shape=(3, 3), archetype="forest")
    result = compute_index_timeseries(ts, "NDVI")
    assert result.dims == ("time", "y", "x")
    assert result.shape == (6, 3, 3)


def test_compute_index_timeseries_with_reduce_median() -> None:
    ts = make_synthetic_s2_timeseries(n_timesteps=6, shape=(3, 3), archetype="forest")
    result = compute_index_timeseries(ts, "NDVI", reduce="median")
    assert result.dims == ("y", "x")
    assert result.shape == (3, 3)


def test_compute_index_timeseries_p95_above_p50() -> None:
    ts = make_synthetic_s2_timeseries(n_timesteps=6, shape=(2, 2), archetype="forest")
    p95 = compute_index_timeseries(ts, "NDVI", reduce="p95")
    p50 = compute_index_timeseries(ts, "NDVI", reduce="p50")
    assert float(p95.mean()) >= float(p50.mean())


def test_compute_index_timeseries_phenology_curve() -> None:
    """Curva NDVI sintética sube 0.2 → 0.85 → 0.30 sobre 6 timesteps."""
    ts = make_synthetic_s2_timeseries(n_timesteps=6, shape=(2, 2), archetype="forest")
    ndvi_ts = compute_index_timeseries(ts, "NDVI").mean(dim=("y", "x")).values
    assert ndvi_ts[0] < ndvi_ts[2]  # ascenso
    assert ndvi_ts[2] > ndvi_ts[-1]  # descenso post-peak
    assert ndvi_ts[2] == pytest.approx(0.85, abs=0.05)  # peak


def test_compute_index_timeseries_invalid_reduce() -> None:
    ts = make_synthetic_s2_timeseries(n_timesteps=3, shape=(2, 2))
    with pytest.raises(ValueError, match="no soportado"):
        compute_index_timeseries(ts, "NDVI", reduce="foo")  # type: ignore[arg-type]


def test_compute_index_timeseries_reduce_without_time_dim() -> None:
    da = make_forest_pixel()
    with pytest.raises(ValueError, match="requiere dimensión 'time'"):
        compute_index_timeseries(da, "NDVI", reduce="median")


# ---------------------------------------------------------------------------
# Grupo E — Earth Engine wrapper (sin credenciales, mock)
# ---------------------------------------------------------------------------


def test_compute_index_ee_calls_spectralIndices() -> None:
    """compute_index_ee delega en ee_image.spectralIndices([alias])."""
    fake_ee_image = MagicMock()
    fake_ee_image.spectralIndices.return_value = "FAKE_EE_IMAGE_NDVI"

    with patch.dict("sys.modules", {"eemont": MagicMock()}):
        result = compute_index_ee(fake_ee_image, "NDVI")

    assert result == "FAKE_EE_IMAGE_NDVI"
    fake_ee_image.spectralIndices.assert_called_once_with(["NDVI"])


def test_compute_index_ee_translates_alias() -> None:
    """compute_index_ee usa el alias spyndex (NDRE→NDREI, MSAVI2→MSAVI, GCVI→CIG)."""
    fake_ee_image = MagicMock()
    fake_ee_image.spectralIndices.return_value = "X"
    with patch.dict("sys.modules", {"eemont": MagicMock()}):
        compute_index_ee(fake_ee_image, "MSAVI2")
    fake_ee_image.spectralIndices.assert_called_with(["MSAVI"])


def test_compute_index_ee_raises_for_custom_formula() -> None:
    fake_ee_image = MagicMock()
    with patch.dict("sys.modules", {"eemont": MagicMock()}):
        with pytest.raises(ValueError, match="custom-formula"):
            compute_index_ee(fake_ee_image, "LAI")


def test_compute_index_ee_raises_for_unknown_index() -> None:
    fake_ee_image = MagicMock()
    with patch.dict("sys.modules", {"eemont": MagicMock()}):
        with pytest.raises(ValueError, match="Índice desconocido"):
            compute_index_ee(fake_ee_image, "FOOBAR")


def test_compute_index_ee_raises_if_eemont_unavailable() -> None:
    """Si eemont falla al importar, propaga ImportError con mensaje claro."""
    fake_ee_image = MagicMock()
    real_import = importlib.import_module

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "eemont":
            raise RuntimeError("ee not initialized")
        return real_import(name, *args, **kwargs)

    with patch(
        "builtins.__import__",
        side_effect=lambda name, *a, **k: (
            fake_import(name, *a, **k) if name == "eemont" else real_import(name)
        ),
    ):
        # Asegurar que eemont no quede pre-cargado en sys.modules
        import sys

        sys.modules.pop("eemont", None)
        with pytest.raises(ImportError, match="eemont"):
            compute_index_ee(fake_ee_image, "NDVI")


# ---------------------------------------------------------------------------
# Grupo F — Cache Redis (fakeredis)
# ---------------------------------------------------------------------------


def test_compute_index_cached_no_redis_client_computes_directly() -> None:
    da = make_forest_pixel()
    result = compute_index_cached(da, "NDVI", scene_id="abc", redis_client=None)
    assert float(result.mean()) == pytest.approx(0.80, abs=1e-4)


def test_compute_index_cached_miss_computes_and_stores() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    da = make_forest_pixel()
    result = compute_index_cached(da, "NDVI", scene_id="patch_42", redis_client=client)
    assert float(result.mean()) == pytest.approx(0.80, abs=1e-4)
    # El SET debe haberse ejecutado
    assert client.get("patch_42:NDVI") is not None


def test_compute_index_cached_hit_returns_cached() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    da = make_forest_pixel()
    first = compute_index_cached(da, "NDVI", scene_id="patch_42", redis_client=client)
    # Second call debería hit
    second = compute_index_cached(da, "NDVI", scene_id="patch_42", redis_client=client)
    np.testing.assert_array_equal(first.values, second.values)


def test_compute_index_cached_corrupt_cache_falls_back_to_compute() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    client.set("patch_42:NDVI", b"not_a_pickle")
    da = make_forest_pixel()
    result = compute_index_cached(da, "NDVI", scene_id="patch_42", redis_client=client)
    assert float(result.mean()) == pytest.approx(0.80, abs=1e-4)


def test_compute_index_cached_redis_error_falls_back_to_compute() -> None:
    da = make_forest_pixel()

    class BrokenRedis:
        def get(self, _key: str) -> bytes | None:
            raise RuntimeError("connection lost")

        def setex(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
            raise RuntimeError("connection lost")

    result = compute_index_cached(
        da,
        "NDVI",
        scene_id="patch_x",
        redis_client=BrokenRedis(),  # type: ignore[arg-type]
    )
    assert float(result.mean()) == pytest.approx(0.80, abs=1e-4)


# ---------------------------------------------------------------------------
# Grupo G — Smoke test PASTIS-R real (skip-graceful)
# ---------------------------------------------------------------------------


_PASTIS_ROOT = Path(__file__).resolve().parents[3] / "data" / "PASTIS-R"


def _have_pastis_subset() -> bool:
    """Verifica que al menos un patch S2 está en disco."""
    return (_PASTIS_ROOT / "DATA_S2").is_dir() and any((_PASTIS_ROOT / "DATA_S2").glob("S2_*.npy"))


@pytest.mark.skipif(
    not _have_pastis_subset(),
    reason="Ejecutar `dvc pull data/PASTIS-R/` para habilitar el smoke test real",
)
def test_compute_ndvi_on_real_pastis_patch() -> None:
    """Smoke test: NDVI sobre un patch real PASTIS-R sin NaN/inf."""
    from ml.ingest.pastis_loader import load_pastis_patch

    sample = next(iter((_PASTIS_ROOT / "DATA_S2").glob("S2_*.npy")))
    patch_id = sample.stem.replace("S2_", "")
    data = load_pastis_patch(patch_id, load_annotations=False)
    s2 = data["s2"]  # (T, 10, 128, 128) int16

    # Tomar t=0, dividir DN por 10000 → reflectancia
    refl = (s2[0].astype(np.float32) / 10000.0).clip(0.0, 1.5)
    da = xr.DataArray(
        refl,
        dims=("band", "y", "x"),
        coords={
            "band": PASTIS_S2_BANDS,
            "y": np.arange(refl.shape[1]),
            "x": np.arange(refl.shape[2]),
        },
    )
    ndvi = compute_index(da, "NDVI")
    finite = np.isfinite(ndvi.values)
    assert finite.any(), "NDVI todo NaN/inf en patch real"
    mean_ndvi = float(np.nanmean(ndvi.values))
    # Rango laxo: PASTIS cubre cultivos diversos en varias estaciones.
    assert -0.5 < mean_ndvi < 1.0, f"NDVI medio fuera de rango razonable: {mean_ndvi}"


# ---------------------------------------------------------------------------
# Grupo H — Documentación cruzada
# ---------------------------------------------------------------------------


def test_spectral_indices_doc_covers_all_17() -> None:
    """docs/spectral_indices.md menciona los 17 nombres canónicos."""
    doc = Path(__file__).resolve().parents[3] / "docs" / "spectral_indices.md"
    assert doc.exists(), "docs/spectral_indices.md no encontrado"
    text = doc.read_text(encoding="utf-8")
    for name in INDEX_NAMES:
        # nombre debe aparecer como palabra completa al menos una vez
        assert re.search(rf"\b{re.escape(name)}\b", text), (
            f"Índice '{name}' no documentado en docs/spectral_indices.md"
        )
