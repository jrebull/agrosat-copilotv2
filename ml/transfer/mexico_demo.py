"""Qualitative zero-shot demo over Mexican crops (US-077).

Applies the SAME methodology pipeline used for the European baseline
(AlphaEarth zonal embedding + Sentinel-2 NDVI phenological curve +
text-description via Gemini) to two real Mexican producing areas:

- **AOI-1 (avocado)**: Uruapan / Tancitaro foothills, Michoacan.
- **AOI-2 (guava)**: Calvillo, Aguascalientes (national guava capital).

Both crops are **perennial woody** (orchard-like): the expected
phenological signature is a high, relatively stable NDVI all year (no
sow-harvest cycle of an annual cereal). The point of US-077 is purely
**methodological**: showing the pipeline replicates to another region
that has NO curated ground-truth.

HONESTY RULE (Arthur + AC): there is **no curated ground-truth** for
avocado/guava in Mexico. This module is 100% qualitative and **never**
reports an F1/accuracy/classifier over Mexico. It only produces the REAL
GEE-derived artefacts: the 64-dim AlphaEarth zonal embedding, the real
S2 NDVI time series, and the Gemini phenological description. It does
NOT import any classification metric (a meta-test enforces this).

Graceful degradation (Arthur): if GEE fails (quota/network), the
extraction helpers return an EMPTY frame/array with a valid dtype so the
notebook enters ``degraded`` mode with an explicit placeholder. A curve
is NEVER fabricated.

Reuse:
- :func:`ml.features.phenology_description.generate_phenology_description`
  for the text description (Gemini 3.5 Flash, ``temperature=0.0``).
- :func:`ml.eval.class_remap.get_label_space` (``"hcat-macro"``, US-074)
  for the qualitative ``PERMANENT_WOODY`` framing (analogy, NOT a
  prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from ml.features.phenology_description import generate_phenology_description

try:
    import ee  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_AOIS",
    "MexicoAOI",
    "aoi_geometry",
    "describe_phenology",
    "extract_alphaearth_zonal",
    "extract_s2_ndvi_series",
    "hcat_perennial_framing",
]

#: AlphaEarth annual embedding collection (global, includes Mexico, v1.1).
ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
#: Sentinel-2 surface reflectance harmonized collection for the NDVI series.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
#: AlphaEarth has 64 embedding bands ``A00..A63``.
ALPHAEARTH_N_DIMS = 64
#: Default local parquet cache for raw GEE pulls (NOT committed to Git).
DEFAULT_CACHE_DIR = Path("data/cache/gee")


@dataclass(frozen=True)
class MexicoAOI:
    """A single Mexican area of interest for the qualitative demo.

    Attributes:
        name: Logical AOI identifier (used in cache keys and outputs).
        crop: Crop grown in the AOI (e.g. ``"aguacate"``).
        lon: Centroid longitude in EPSG:4326.
        lat: Centroid latitude in EPSG:4326.
        buffer_m: Buffer radius in meters around the centroid point.
        expected_phenology: Human-readable description of the expected
            phenological signature (perennial woody = high, stable NDVI).
        crop_type_hint: Hint fed to block 2 of the Wen et al. phenology
            prompt (e.g. ``"aguacate (perenne arboreo)"``).
    """

    name: str
    crop: str
    lon: float
    lat: float
    buffer_m: int
    expected_phenology: str
    crop_type_hint: str


#: The two real Mexican AOIs (coords verified in GEE this session).
DEFAULT_AOIS: tuple[MexicoAOI, ...] = (
    MexicoAOI(
        name="aguacate_uruapan",
        crop="aguacate",
        lon=-102.05,
        lat=19.41,
        buffer_m=1500,
        expected_phenology=(
            "Perenne arboreo: NDVI alto y estable (~0.6-0.8) todo el anio, "
            "sin ciclo estacional marcado de siembra-cosecha."
        ),
        crop_type_hint="aguacate (perenne arboreo, huerta de Michoacan)",
    ),
    MexicoAOI(
        name="guayaba_calvillo",
        crop="guayaba",
        lon=-102.72,
        lat=21.85,
        buffer_m=1500,
        expected_phenology=(
            "Perenne arboreo subtropical: NDVI alto, con ligera modulacion "
            "estacional por riego y poda."
        ),
        crop_type_hint="guayaba (perenne arboreo subtropical, Calvillo)",
    ),
)


def aoi_geometry(aoi: MexicoAOI) -> Any:
    """Build the ``ee.Geometry`` of an AOI: point + circular buffer.

    Args:
        aoi: The Mexican AOI to convert.

    Returns:
        ``ee.Geometry`` = ``Point([lon, lat]).buffer(buffer_m)``.

    Raises:
        ImportError: if ``earthengine-api`` is not installed.
    """
    if ee is None:  # pragma: no cover - exercised only without the SDK
        raise ImportError("earthengine-api is not installed. Run `poetry install --with ml,geo`.")
    return ee.Geometry.Point([aoi.lon, aoi.lat]).buffer(aoi.buffer_m)


def _alphaearth_band_names() -> list[str]:
    """AlphaEarth band names ``A00..A63``."""
    return [f"A{i:02d}" for i in range(ALPHAEARTH_N_DIMS)]


def _empty_alphaearth_vector() -> np.ndarray:
    """Empty float64 array used as the degraded AlphaEarth result.

    Returns:
        ``np.ndarray`` of shape ``(0,)`` and dtype ``float64`` (valid dtype,
        no fabricated values).
    """
    return np.empty((0,), dtype=np.float64)


def extract_alphaearth_zonal(
    aoi: MexicoAOI,
    year: int,
    *,
    scale: int = 10,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Extract the REAL 64-dim AlphaEarth zonal embedding of an AOI.

    Mosaics the annual AlphaEarth image over ``year`` and reduces it to a
    single mean vector over the AOI buffer with ``ee.Reducer.mean()``.
    ``mosaic()`` (not ``first()``) is used because AlphaEarth distributes
    the annual embedding across disjoint tiles; a single tile would leave
    pixels outside it null.

    Args:
        aoi: The Mexican AOI.
        year: Year of the annual embedding (2023 verified available).
        scale: Reduction resolution in meters (AlphaEarth native = 10).
        cache_dir: Parquet cache folder (default ``data/cache/gee/``).

    Returns:
        ``np.ndarray`` of shape ``(64,)`` float64 with the REAL zonal mean
        embedding. If GEE is unavailable or fails, returns an EMPTY array
        ``(0,)`` with a valid dtype (degraded mode, never fabricated).
    """
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"mexico_alphaearth_{aoi.name}_{year}_{scale}.parquet"
    band_names = _alphaearth_band_names()
    if cache_file.exists():
        cached = pl.read_parquet(cache_file)
        if cached.height == 1 and all(b in cached.columns for b in band_names):
            return cached.select(band_names).to_numpy().reshape(-1).astype(np.float64)

    if ee is None:
        return _empty_alphaearth_vector()

    try:
        geometry = aoi_geometry(aoi)
        image = (
            ee.ImageCollection(ALPHAEARTH_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic()
            .select(band_names)
        )
        stat = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1_000_000_000,
        )
        info = stat.getInfo() or {}
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        logger.warning("mexico_alphaearth_failed", aoi=aoi.name, year=year, error=str(exc))
        return _empty_alphaearth_vector()

    raw = [info.get(b) for b in band_names]
    if any(v is None for v in raw):
        logger.warning("mexico_alphaearth_incomplete", aoi=aoi.name, year=year)
        return _empty_alphaearth_vector()

    values = [float(v) for v in raw if v is not None]
    vector = np.asarray(values, dtype=np.float64)
    pl.DataFrame({b: [v] for b, v in zip(band_names, values, strict=True)}).write_parquet(
        cache_file
    )
    return vector


def _ndvi_empty_frame() -> pl.DataFrame:
    """Empty NDVI series frame with the canonical schema."""
    return pl.DataFrame(schema={"date": pl.Utf8, "doy": pl.Int64, "ndvi": pl.Float64})


def _mask_s2_clouds_and_add_ndvi(image: Any) -> Any:
    """Apply QA60 cloud mask and append the NDVI band to an S2 image.

    QA60 bit 10 = opaque clouds, bit 11 = cirrus; both must be 0 for a
    clear pixel. NDVI = ``(B8 - B4) / (B8 + B4)`` via
    ``normalizedDifference(['B8', 'B4'])``.

    Args:
        image: Single ``ee.Image`` from ``COPERNICUS/S2_SR_HARMONIZED``.

    Returns:
        The masked image with an extra ``NDVI`` band.
    """
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi).updateMask(mask)


def extract_s2_ndvi_series(
    aoi: MexicoAOI,
    year: int,
    *,
    cloud_pct_max: int = 40,
    scale: int = 10,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Extract the REAL Sentinel-2 NDVI time series of an AOI for ``year``.

    Builds a cloud-masked (QA60) S2 collection over the AOI, computes per
    image the zonal mean NDVI with ``ee.Reducer.mean()`` and returns a
    ``(date, doy, ndvi)`` frame. ``cloud_pct_max=40`` is more permissive
    than the European 30 because of tropical Michoacan cloudiness.

    Args:
        aoi: The Mexican AOI.
        year: Year of the time series (2023 verified: ~104 images).
        cloud_pct_max: Maximum ``CLOUDY_PIXEL_PERCENTAGE`` to keep an image.
        scale: Reduction resolution in meters.
        cache_dir: Parquet cache folder (default ``data/cache/gee/``).

    Returns:
        ``pl.DataFrame`` with columns ``date`` (``YYYY-MM-DD``), ``doy``
        (1..366) and ``ndvi`` (in ``[-1, 1]``), sorted by date. If GEE is
        unavailable or fails, returns an EMPTY frame with valid schema
        (degraded mode, never fabricated).
    """
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"mexico_ndvi_{aoi.name}_{year}_{cloud_pct_max}_{scale}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None:
        return _ndvi_empty_frame()

    try:
        geometry = aoi_geometry(aoi)
        collection = (
            ee.ImageCollection(S2_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .map(_mask_s2_clouds_and_add_ndvi)
        )

        def _reduce_image(image: Any) -> Any:
            mean_ndvi = image.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1_000_000_000,
            )
            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "ndvi": mean_ndvi.get("NDVI"),
                },
            )

        features = collection.map(_reduce_image).getInfo()
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        logger.warning("mexico_ndvi_failed", aoi=aoi.name, year=year, error=str(exc))
        return _ndvi_empty_frame()

    rows: list[dict[str, Any]] = []
    for feat in (features or {}).get("features", []):
        props = feat.get("properties", {}) or {}
        date_str = props.get("date")
        ndvi_val = props.get("ndvi")
        if date_str is None or ndvi_val is None:
            continue
        rows.append({"date": str(date_str), "ndvi": float(ndvi_val)})

    if not rows:
        return _ndvi_empty_frame()

    frame = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d").dt.ordinal_day().cast(pl.Int64).alias("doy")
        )
        .select("date", "doy", "ndvi")
        .sort("date")
    )
    frame.write_parquet(cache_file)
    return frame


def describe_phenology(
    ndvi_series: pl.DataFrame,
    aoi: MexicoAOI,
    *,
    model: str = "gemini-3.5-flash",
    cache_dir: Path | None = None,
) -> str:
    """Generate the REAL phenological text description of an AOI (Gemini).

    Adapts the ``(date, doy, ndvi)`` series to the ``(ndvi_curve, doy)``
    contract of
    :func:`ml.features.phenology_description.generate_phenology_description`
    and delegates to it (reuse, no re-implementation). ``temperature=0.0``
    is mandatory (R7) and enforced by the underlying module.

    Args:
        ndvi_series: Frame from :func:`extract_s2_ndvi_series` with columns
            ``date, doy, ndvi``.
        aoi: The AOI (its ``crop_type_hint`` enters block 2 of the prompt).
        model: Gemini model id (default ``"gemini-3.5-flash"``).
        cache_dir: Optional cache dir for the description (delegated).

    Returns:
        The generated phenological description (one paragraph in Spanish).

    Raises:
        ValueError: if ``ndvi_series`` is empty (no real curve to describe;
            the caller must degrade explicitly instead of fabricating one).
        RuntimeError: propagated from the phenology module if Gemini
            credentials are missing (no silent mock in deliverable notebooks).
    """
    if ndvi_series.is_empty():
        raise ValueError(
            f"NDVI series for AOI {aoi.name!r} is empty; cannot describe "
            "phenology without a real curve (degrade explicitly instead)."
        )
    ordered = ndvi_series.sort("doy")
    ndvi_curve = ordered.get_column("ndvi").to_numpy().astype(np.float64)
    doy = ordered.get_column("doy").to_numpy().astype(np.float64)
    return generate_phenology_description(
        ndvi_curve,
        doy,
        parcel_id=aoi.name,
        crop_type_hint=aoi.crop_type_hint,
        model=model,
        temperature=0.0,
        cache_dir=cache_dir,
    )


def hcat_perennial_framing() -> dict[str, str]:
    """Qualitative HCAT framing of a perennial woody crop (US-074 space).

    Returns the ``PERMANENT_WOODY`` classes (``orchard`` / ``vineyard``) of
    the registered ``hcat-macro`` label-space as the analogous family of an
    orchard-like perennial. This is a **qualitative analogy** of the
    phenological signature, NOT a prediction and NOT an F1: no classifier
    is invoked.

    Returns:
        Mapping ``{crop_name: macro_group_label}`` for every ``hcat-macro``
        class whose macro family is ``PERMANENT_WOODY`` (e.g.
        ``{"Orchard": "PERMANENT_WOODY|orchard", ...}``).
    """
    from ml.eval.class_remap import get_label_space

    space = get_label_space("hcat-macro")
    framing: dict[str, str] = {}
    for label in space.class_names.values():
        # label format: "MACRO_L1_6|macro_hcat_group|crop_name"
        parts = label.split("|")
        if len(parts) == 3 and parts[0] == "PERMANENT_WOODY":
            macro, group, crop_name = parts
            framing[crop_name] = f"{macro}|{group}"
    return framing
