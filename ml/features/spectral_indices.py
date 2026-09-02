"""Canonical library of the 17 spectral indices of the AgroSatCopilot project.

This is the **single source of truth** for computing NDVI, NDWI, NDMI,
EVI, SAVI, MSAVI2, NBR, MCARI, CCCI, LAI, FAPAR, PSRI, NDCI, GCVI, RENDVI,
NDRE and TSAVI on Sentinel-2 tensors in ``xarray.DataArray`` format.
Any consumer of the repo (notebooks, Dagster assets, ADK agent, paper)
must invoke this module and never recompute formulas by hand.

Band convention
---------------

The input is an ``xarray.DataArray`` with a ``band`` dimension whose
canonical labels are those imported from
:data:`ml.ingest.pastis_loader.PASTIS_S2_BANDS`::

    ("B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12")

The internal mapping to the ``spyndex`` nomenclature (`B`, `G`, `R`, `RE1`,
`RE2`, `RE3`, `N`, `RE4`, `S1`, `S2`) is applied transparently in
:data:`_BAND_TO_SPYNDEX`.

Caller contract (responsibility **outside** this module)
--------------------------------------------------------

The Avance 1 EDA (`notebooks/eda/Avance1.Equipo17.ipynb`, §3 Bivariate and
§5 Global conclusions) reported:

- NDVI saturates to 1.0 in 75 % of parcels if no SCL mask is applied => the
  caller must filter clouds/shadows/snow before calling ``compute_index``.
- The Sentinel-2 bands distributed as DN (0-10000) must be divided by
  10000 to enter as reflectance in [0, 1].
- Strong redundancy detected in the quartet {NDVI, NDRE, NDWI, SAVI}
  (Pearson 0.95-0.97); the final selection (US-018) retains {NDVI, NDMI, EVI}.
  This module delivers the 17 candidates in full; the selection is not its
  responsibility.

Supported backends
-------------------

1. ``spyndex`` 0.10 (main engine, offline over xarray) — covers 14/17:
   11 literal indices + 3 aliases (MSAVI2->MSAVI, NDRE->NDREI, GCVI->CIG).
2. Custom formulas audited with DOI for the remaining 3 (LAI Boegh 2002,
   FAPAR Myneni 1997, CCCI Barnes 2000).
3. ``eemont`` (optional wrapper, server-side GEE) via :func:`compute_index_ee`
   for ingestion pipelines US-006/US-009. Lazy import: does not fail if the
   GEE credentials are not available locally.

Full references with DOI per index in
:doc:`docs/spectral_indices.md </spectral_indices>`.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import spyndex  # type: ignore[import-untyped]  # no py.typed marker upstream
import xarray as xr

from ml.ingest.pastis_loader import PASTIS_S2_BANDS

if TYPE_CHECKING:  # pragma: no cover
    import ee
    import redis


__all__ = [
    "INDEX_NAMES",
    "ReduceMethod",
    "compute_index",
    "compute_index_cached",
    "compute_index_ee",
    "compute_index_timeseries",
]


# Literal canonical order from plan v6 line 1104 (CA US-014).
INDEX_NAMES: list[str] = [
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

ReduceMethod = Literal["mean", "median", "max", "min", "p10", "p50", "p90", "p95"]


# Mapping of canonical Sentinel-2 bands (PASTIS_S2_BANDS) -> spyndex nomenclature.
_BAND_TO_SPYNDEX: dict[str, str] = {
    "B02": "B",
    "B03": "G",
    "B04": "R",
    "B05": "RE1",
    "B06": "RE2",
    "B07": "RE3",
    "B08": "N",
    "B8A": "RE4",
    "B11": "S1",
    "B12": "S2",
}


@dataclass(frozen=True)
class _IndexEntry:
    """Entry of the canonical registry for an index."""

    name: str
    backend: Literal["spyndex", "custom"]
    spyndex_name: str | None  # only for backend='spyndex' (aliases allowed)
    custom_fn: Callable[[xr.DataArray], xr.DataArray] | None = None
    required_bands: tuple[str, ...] = field(default_factory=tuple)
    formula: str = ""
    agronomic_use: str = ""
    expected_range: tuple[float, float] = (-1.0, 1.0)
    reference: str = ""
    extra_params: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audited custom formulas (not present in spyndex 0.10).
# Single canonical version of the project -- do not revisit without ADR.
# Full references in docs/spectral_indices.md.
# ---------------------------------------------------------------------------


def _ndvi_array(da: xr.DataArray) -> xr.DataArray:
    """Internal helper: NDVI without going through spyndex (avoids recursion)."""
    n = da.sel(band="B08").astype(np.float32)
    r = da.sel(band="B04").astype(np.float32)
    denom = n + r
    # safe division: 0/0 -> NaN, not inf
    ndvi = xr.where(denom != 0, (n - r) / denom, np.float32(np.nan))
    return cast(xr.DataArray, ndvi.astype(np.float32))


def _ndre_array(da: xr.DataArray) -> xr.DataArray:
    """Internal helper: NDRE/NDREI without going through spyndex."""
    n = da.sel(band="B08").astype(np.float32)
    re1 = da.sel(band="B05").astype(np.float32)
    denom = n + re1
    ndre = xr.where(denom != 0, (n - re1) / denom, np.float32(np.nan))
    return cast(xr.DataArray, ndre.astype(np.float32))


def _lai_boegh_2002(da: xr.DataArray) -> xr.DataArray:
    """LAI = -ln(1 - (NDVI - 0.05) / 0.95) / 0.5 — Boegh et al. 2002.

    Boegh, E., Soegaard, H., Broge, N., Hasager, C.B., Jensen, N.O., Schelde, K.,
    Thomsen, A. (2002). *Airborne multispectral data for quantifying leaf area
    index, nitrogen concentration, and photosynthetic efficiency in agriculture*.
    Remote Sensing of Environment 81(2-3), 179-193.
    DOI: 10.1016/S0034-4257(01)00342-X.
    """
    ndvi = _ndvi_array(da)
    # arg = 1 - (NDVI - 0.05) / 0.95. If NDVI>=1.0 -> arg<=0 -> log undefined -> NaN.
    arg = 1.0 - (ndvi - 0.05) / 0.95
    safe = xr.where(arg > 0, arg, np.float32(np.nan))
    lai = -xr.apply_ufunc(np.log, safe, dask="allowed") / 0.5
    return cast(xr.DataArray, lai.astype(np.float32))


def _fapar_myneni_1997(da: xr.DataArray) -> xr.DataArray:
    """FAPAR = 1.24 * NDVI - 0.168 -- Myneni & Williams 1994 (linear fit).

    Myneni, R.B., Williams, D.L. (1994). *On the relationship between FAPAR
    and NDVI*. Remote Sensing of Environment 49(3), 200-211.
    DOI: 10.1016/0034-4257(94)90016-7. The constant 1.24 / -0.168 is the linear
    fit over temperate forests frequently cited in later literature
    (Myneni 1997, Ross 1981).
    """
    ndvi = _ndvi_array(da)
    return cast(xr.DataArray, (1.24 * ndvi - 0.168).astype(np.float32))


def _ccci_barnes_2000(da: xr.DataArray) -> xr.DataArray:
    """CCCI = NDRE / NDVI — Barnes et al. 2000 (Canopy Chlorophyll Content Index).

    Barnes, E.M., Clarke, T.R., Richards, S.E., Colaizzi, P.D., Haberland, J.,
    Kostrzewski, M., Waller, P., Choi, C., Riley, E., Thompson, T., Lascano, R.J.,
    Li, H., Moran, M.S. (2000). *Coincident detection of crop water stress,
    nitrogen status and canopy density using ground-based multispectral data*.
    Proceedings of the 5th International Conference on Precision Agriculture.
    Normalized ratio that corrects NDRE by the canopy density estimated with NDVI.
    """
    ndvi = _ndvi_array(da)
    ndre = _ndre_array(da)
    ccci = xr.where(ndvi != 0, ndre / ndvi, np.float32(np.nan))
    return cast(xr.DataArray, ccci.astype(np.float32))


# ---------------------------------------------------------------------------
# Canonical registry of the 17 indices.
# ---------------------------------------------------------------------------

_INDEX_REGISTRY: dict[str, _IndexEntry] = {
    "NDVI": _IndexEntry(
        name="NDVI",
        backend="spyndex",
        spyndex_name="NDVI",
        required_bands=("B04", "B08"),
        formula="(N - R) / (N + R)",
        agronomic_use="Vigor vegetativo general; satura a partir de LAI ~3.",
        expected_range=(-1.0, 1.0),
        reference="Rouse et al. 1974",
    ),
    "NDWI": _IndexEntry(
        name="NDWI",
        backend="spyndex",
        spyndex_name="NDWI",
        required_bands=("B03", "B08"),
        formula="(G - N) / (G + N)",
        agronomic_use="Cuerpos de agua y contenido de agua en hoja.",
        expected_range=(-1.0, 1.0),
        reference="McFeeters 1996",
    ),
    "NDMI": _IndexEntry(
        name="NDMI",
        backend="spyndex",
        spyndex_name="NDMI",
        required_bands=("B08", "B11"),
        formula="(N - S1) / (N + S1)",
        agronomic_use="Humedad del canopy; estrés hídrico.",
        expected_range=(-1.0, 1.0),
        reference="Gao 1996",
    ),
    "EVI": _IndexEntry(
        name="EVI",
        backend="spyndex",
        spyndex_name="EVI",
        required_bands=("B02", "B04", "B08"),
        formula="g · (N - R) / (N + C1·R - C2·B + L)",
        agronomic_use="Vigor en canopy denso; corrige saturación y aerosoles.",
        expected_range=(-1.0, 1.0),
        reference="Huete et al. 2002",
        extra_params={"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0},
    ),
    "SAVI": _IndexEntry(
        name="SAVI",
        backend="spyndex",
        spyndex_name="SAVI",
        required_bands=("B04", "B08"),
        formula="(1+L) · (N - R) / (N + R + L), L=0.5",
        agronomic_use="Vigor con corrección por suelo expuesto.",
        expected_range=(-1.5, 1.5),
        reference="Huete 1988",
        extra_params={"L": 0.5},
    ),
    "MSAVI2": _IndexEntry(
        name="MSAVI2",
        backend="spyndex",
        spyndex_name="MSAVI",  # alias: spyndex MSAVI == Qi 1994 MSAVI2
        required_bands=("B04", "B08"),
        formula="0.5·(2N+1 - √((2N+1)² - 8(N-R)))",
        agronomic_use="SAVI auto-calibrado, no requiere ajustar L.",
        expected_range=(-1.0, 1.0),
        reference="Qi et al. 1994 (alias spyndex 'MSAVI')",
    ),
    "NBR": _IndexEntry(
        name="NBR",
        backend="spyndex",
        spyndex_name="NBR",
        required_bands=("B08", "B12"),
        formula="(N - S2) / (N + S2)",
        agronomic_use="Detección de áreas quemadas y severidad de fuego.",
        expected_range=(-1.0, 1.0),
        reference="Key & Benson 2006",
    ),
    "MCARI": _IndexEntry(
        name="MCARI",
        backend="spyndex",
        spyndex_name="MCARI",
        required_bands=("B03", "B04", "B05"),
        formula="((RE1 - R) - 0.2·(RE1 - G)) · (RE1 / R)",
        agronomic_use="Contenido de clorofila en hoja.",
        expected_range=(-2.0, 2.0),
        reference="Daughtry et al. 2000",
    ),
    "CCCI": _IndexEntry(
        name="CCCI",
        backend="custom",
        spyndex_name=None,
        custom_fn=_ccci_barnes_2000,
        required_bands=("B04", "B05", "B08"),
        formula="NDRE / NDVI",
        agronomic_use="Clorofila corregida por densidad de canopy (N status).",
        expected_range=(-2.0, 2.0),
        reference="Barnes et al. 2000",
    ),
    "LAI": _IndexEntry(
        name="LAI",
        backend="custom",
        spyndex_name=None,
        custom_fn=_lai_boegh_2002,
        required_bands=("B04", "B08"),
        formula="-ln(1 - (NDVI - 0.05) / 0.95) / 0.5",
        agronomic_use="Índice de área foliar empírico (Boegh 2002).",
        expected_range=(0.0, 8.0),
        reference="Boegh et al. 2002",
    ),
    "FAPAR": _IndexEntry(
        name="FAPAR",
        backend="custom",
        spyndex_name=None,
        custom_fn=_fapar_myneni_1997,
        required_bands=("B04", "B08"),
        formula="1.24 · NDVI - 0.168",
        agronomic_use="Fracción de PAR absorbida (fotosíntesis).",
        expected_range=(-0.2, 1.1),
        reference="Myneni & Williams 1994",
    ),
    "PSRI": _IndexEntry(
        name="PSRI",
        backend="spyndex",
        spyndex_name="PSRI",
        required_bands=("B02", "B04", "B06"),
        formula="(R - B) / RE2",
        agronomic_use="Senescencia / pigmentos plantas.",
        expected_range=(-1.0, 2.0),
        reference="Merzlyak et al. 1999",
    ),
    "NDCI": _IndexEntry(
        name="NDCI",
        backend="spyndex",
        spyndex_name="NDCI",
        required_bands=("B04", "B05"),
        formula="(RE1 - R) / (RE1 + R)",
        agronomic_use="Clorofila-a en aguas continentales.",
        expected_range=(-1.0, 1.0),
        reference="Mishra & Mishra 2012",
    ),
    "GCVI": _IndexEntry(
        name="GCVI",
        backend="spyndex",
        spyndex_name="CIG",  # alias: Gitelson Chlorophyll Index Green
        required_bands=("B03", "B08"),
        formula="N / G - 1",
        agronomic_use="Clorofila verde en canopy.",
        expected_range=(0.0, 20.0),
        reference="Gitelson et al. 2003 (alias spyndex 'CIG')",
    ),
    "RENDVI": _IndexEntry(
        name="RENDVI",
        backend="spyndex",
        spyndex_name="RENDVI",
        required_bands=("B05", "B06"),
        formula="(RE2 - RE1) / (RE2 + RE1)",
        agronomic_use="NDVI red-edge; sensible a estrés temprano.",
        expected_range=(-1.0, 1.0),
        reference="Gitelson & Merzlyak 1994",
    ),
    "NDRE": _IndexEntry(
        name="NDRE",
        backend="spyndex",
        spyndex_name="NDREI",  # alias spyndex 0.10
        required_bands=("B05", "B08"),
        formula="(N - RE1) / (N + RE1)",
        agronomic_use="Cultivos densos donde NDVI satura.",
        expected_range=(-1.0, 1.0),
        reference="Barnes et al. 2000 (alias spyndex 'NDREI')",
    ),
    "TSAVI": _IndexEntry(
        name="TSAVI",
        backend="spyndex",
        spyndex_name="TSAVI",
        required_bands=("B04", "B08"),
        formula="sla·(N - sla·R - slb) / (sla·N + R - sla·slb)",
        agronomic_use="SAVI calibrado con línea de suelo regional.",
        expected_range=(-1.5, 1.5),
        reference="Baret & Guyot 1991",
        extra_params={"sla": 1.0, "slb": 0.0},
    ),
}


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _validate_input(da: xr.DataArray, required_bands: Sequence[str]) -> None:
    """Validate that ``da`` has dim 'band' with the required labels.

    Args:
        da: DataArray to validate.
        required_bands: Bands the index needs (e.g. ('B04','B08')).

    Raises:
        ValueError: if the 'band' dimension is missing or not string-indexable.
        KeyError: if any required band is not present.
    """
    if "band" not in da.dims:
        raise ValueError(
            "DataArray debe tener dimensión 'band' con labels en "
            f"{PASTIS_S2_BANDS}; dims recibidos: {da.dims}"
        )
    available = set(da.coords["band"].values.tolist()) if "band" in da.coords else set()
    missing = [b for b in required_bands if b not in available]
    if missing:
        raise KeyError(
            f"Bandas faltantes en DataArray para el índice solicitado: {missing}. "
            f"Bandas disponibles: {sorted(available)}"
        )


def _compute_via_spyndex(
    da: xr.DataArray,
    spyndex_name: str,
    required_bands: Sequence[str],
    extra_params: dict[str, float],
) -> xr.DataArray:
    """Compute an index by delegating to ``spyndex.computeIndex``.

    Builds the ``params`` dict with bands mapped to the spyndex
    nomenclature (B/G/R/RE1/.../S2) and constants (L, g, C1, ...). Rebuilds
    the resulting ``DataArray`` preserving the input coords.
    """
    params: dict[str, Any] = {}
    for band in required_bands:
        params[_BAND_TO_SPYNDEX[band]] = da.sel(band=band).astype(np.float32)
    params.update(extra_params)
    result = spyndex.computeIndex(index=spyndex_name, params=params)

    # Defensive: spyndex may return numpy in some paths; we rebuild the
    # DataArray with spatial coords from the input (excludes 'band').
    if not isinstance(result, xr.DataArray):
        spatial_dims = tuple(d for d in da.dims if d != "band")
        spatial_coords = {d: da.coords[d] for d in spatial_dims if d in da.coords}
        result = xr.DataArray(
            np.asarray(result, dtype=np.float32),
            dims=spatial_dims,
            coords=spatial_coords,
        )
    return result.astype(np.float32)


def _attach_attrs(result: xr.DataArray, entry: _IndexEntry, backend: str) -> xr.DataArray:
    """Attach academic metadata to ``result.attrs``."""
    result.attrs.update(
        {
            "index_name": entry.name,
            "formula": entry.formula,
            "reference": entry.reference,
            "computed_with": backend,
            "agronomic_use": entry.agronomic_use,
            "expected_range": str(entry.expected_range),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def compute_index(da: xr.DataArray, index: str) -> xr.DataArray:
    """Compute one of the 17 canonical spectral indices of the project.

    Args:
        da: ``xarray.DataArray`` with a 'band' dimension whose labels are
            those of :data:`ml.ingest.pastis_loader.PASTIS_S2_BANDS` and
            reflectance scaled to the range [0, 1]. The caller must have
            applied the SCL mask and divided DN values /10000 beforehand
            (see ``notebooks/eda/Avance1.Equipo17.ipynb`` §5 Conclusions).
        index: Canonical name (one of :data:`INDEX_NAMES`). Case-sensitive.

    Returns:
        DataArray with the computed index, preserving the input spatial
        coords (y/x or lat/lon). If ``da`` has a 'time' dimension, the
        natural broadcast returns shape (time, y, x). Dtype ``float32``.
        Attributes: ``{"index_name", "formula", "reference", "computed_with",
        "agronomic_use", "expected_range"}``.

    Raises:
        ValueError: if ``index`` is not in :data:`INDEX_NAMES`.
        KeyError: if ``da`` lacks any required band.

    References:
        Full academic table with DOIs per index in
        ``docs/spectral_indices.md``.
    """
    if index not in _INDEX_REGISTRY:
        raise ValueError(
            f"Índice desconocido '{index}'. Disponibles: {INDEX_NAMES}. "
            f"Ver docs/spectral_indices.md para la lista oficial."
        )
    entry = _INDEX_REGISTRY[index]
    _validate_input(da, entry.required_bands)

    if entry.backend == "spyndex":
        assert entry.spyndex_name is not None
        result = _compute_via_spyndex(
            da,
            spyndex_name=entry.spyndex_name,
            required_bands=entry.required_bands,
            extra_params=entry.extra_params,
        )
        return _attach_attrs(result, entry, backend=f"spyndex:{entry.spyndex_name}")

    # backend == "custom"
    assert entry.custom_fn is not None
    result = entry.custom_fn(da)
    return _attach_attrs(result, entry, backend="custom")


def compute_index_timeseries(
    da: xr.DataArray,
    index: str,
    reduce: ReduceMethod | None = None,
) -> xr.DataArray:
    """Compute an index over a time series with optional reduction.

    Args:
        da: DataArray dims ('time', 'band', y, x) in reflectance [0, 1].
        index: Canonical index name.
        reduce: Temporal aggregation strategy:

            - ``None`` (default): keeps the 'time' axis, output shape (T, y, x).
            - ``'mean'``, ``'median'``, ``'max'``, ``'min'``: simple reduction.
            - ``'p10'``, ``'p50'``, ``'p90'``, ``'p95'``: temporal percentile.

    Returns:
        DataArray with the computed index. Without reduce: dims (time, y, x).
        With reduce: dims (y, x).

    Raises:
        ValueError: if ``reduce`` is not a recognized value.
    """
    full = compute_index(da, index)
    if reduce is None:
        return full
    if "time" not in full.dims:
        raise ValueError(
            f"reduce='{reduce}' requiere dimensión 'time' en el input; dims actuales: {full.dims}"
        )
    if reduce in ("mean", "median", "max", "min"):
        return cast(xr.DataArray, getattr(full, reduce)(dim="time").astype(np.float32))
    if reduce in ("p10", "p50", "p90", "p95"):
        q = int(reduce[1:]) / 100.0
        reduced = full.quantile(q, dim="time").astype(np.float32)
        return cast(xr.DataArray, reduced.drop_vars("quantile", errors="ignore"))
    raise ValueError(
        f"reduce='{reduce}' no soportado. Opciones: mean/median/max/min/p10/p50/p90/p95."
    )


def compute_index_cached(
    da: xr.DataArray,
    index: str,
    scene_id: str,
    redis_client: redis.Redis | None = None,
    ttl_seconds: int = 86_400,
) -> xr.DataArray:
    """Variant of :func:`compute_index` with optional Redis cache.

    Key: ``"{scene_id}:{index}"``. If ``redis_client`` is ``None`` or the
    cache fails, it degrades gracefully and simply computes the index.

    Args:
        da: Sentinel-2 DataArray (see :func:`compute_index`).
        index: Canonical name.
        scene_id: Unique scene identifier (e.g. PASTIS patch_id or GEE
            tile_id) used as the key prefix.
        redis_client: Redis client (or ``fakeredis.FakeRedis`` in tests).
        ttl_seconds: Cache TTL in seconds (default 24h).

    Returns:
        Same result as :func:`compute_index`.
    """
    if redis_client is None:
        return compute_index(da, index)

    key = f"{scene_id}:{index}"
    try:
        cached = redis_client.get(key)
    except Exception:  # noqa: BLE001 — graceful degradation if Redis goes down
        return compute_index(da, index)

    if cached is not None:
        try:
            payload = pickle.loads(cast(bytes, cached))
            return xr.DataArray(
                np.frombuffer(payload["data"], dtype=np.float32).reshape(payload["shape"]),
                dims=payload["dims"],
                attrs=payload.get("attrs", {}),
            )
        except Exception:  # noqa: BLE001, S110 — fallback if cache is corrupt
            pass  # silently falls through to recompute (cache is best-effort)

    result = compute_index(da, index)
    try:
        payload = {
            "data": np.ascontiguousarray(result.values, dtype=np.float32).tobytes(),
            "shape": result.shape,
            "dims": result.dims,
            "attrs": dict(result.attrs),
        }
        redis_client.setex(key, ttl_seconds, pickle.dumps(payload))
    except Exception:  # noqa: BLE001, S110 — cache is best-effort, do not block
        pass  # SET failed: we return the computation anyway
    return result


def compute_index_ee(ee_image: ee.Image, index: str) -> ee.Image:
    """Server-side GEE wrapper: compute the index via ``eemont.spectralIndices``.

    Useful for ingestion pipelines US-006/US-009 where the computation lives
    in Earth Engine and is not downloaded to the client. **Lazy import** of
    ``eemont`` so as not to break if the GEE credentials are not configured
    locally.

    Args:
        ee_image: Sentinel-2 ``ee.Image`` with scaled bands (eemont scales
            automatically if ``.scale()`` was previously called).
        index: Canonical project name (translated to the spyndex alias if applicable).

    Returns:
        ``ee.Image`` with the index band(s) added.

    Raises:
        ImportError: if ``eemont`` is not installed or the Earth Engine
            initialization fails.
        ValueError: if ``index`` is not in :data:`INDEX_NAMES`.
    """
    if index not in _INDEX_REGISTRY:
        raise ValueError(f"Índice desconocido '{index}'. Disponibles: {INDEX_NAMES}.")
    try:
        import eemont  # noqa: F401  # registers .spectralIndices() on ee.Image
    except Exception as exc:
        raise ImportError(
            "eemont no está disponible o falló al inicializar Earth Engine. "
            "Instalar con `poetry install --with geo` y autenticar con "
            "`earthengine authenticate`."
        ) from exc

    entry = _INDEX_REGISTRY[index]
    if entry.backend == "spyndex":
        assert entry.spyndex_name is not None
        ee_name = entry.spyndex_name
    else:
        # Custom: eemont does not know it, we raise an explicit error.
        raise ValueError(
            f"Índice '{index}' es custom-formula del proyecto (sin equivalente "
            f"en eemont/awesome-spectral-indices). Computar offline con "
            f"compute_index() en su lugar."
        )

    # eemont dynamically adds `spectralIndices` to ee.Image on import,
    # so the attribute does not exist in the static stubs of earthengine-api.
    return cast("ee.Image", ee_image.spectralIndices([ee_name]))  # type: ignore[attr-defined]
