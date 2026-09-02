"""Sentinel-2 sampling helpers from Google Earth Engine for EDA.

Does not perform bulk downloads. Uses server-side `sampleRegions` and caches
results in local parquet under `data/cache/gee/`. Suitable only for EDA
(US-010/011/012). Production ingestion with Dagster + GCS is closed in
US-006/007/009.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

try:
    import ee  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

_log = structlog.get_logger(__name__)

ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
S1_COLLECTION = "COPERNICUS/S1_GRD"
SRTM_COLLECTION = "USGS/SRTMGL1_003"

#: 8 cardinal quadrants used by `sample_srtm_terrain` to discretize
#: the dominant orientation (aspect) in degrees [0, 360) -> cardinal string.
_ASPECT_CARDINALS: tuple[str, ...] = (
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
)
DYNAMIC_WORLD_CLASSES: dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}
ALPHAEARTH_DIM_COLS: list[str] = [f"dim_{i:02d}" for i in range(64)]

DEFAULT_CACHE_DIR = Path("data/cache/gee")
DEFAULT_S2_BANDS: list[str] = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]


def init_ee(
    service_account_json: Path | None = None,
    project: str | None = None,
    interactive_auth: bool = False,
) -> None:
    """Initialize Earth Engine with service account, ADC or interactive auth.

    Preference order:

    1. Service account JSON if `service_account_json` points to a valid file.
    2. ADC (`gcloud auth application-default login`) — credentials reused
       from `~/.config/gcloud/application_default_credentials.json` or
       `~/.config/earthengine/credentials`.
    3. Only if `interactive_auth=True` and `ee.Initialize` fails, triggers
       `ee.Authenticate()` in browser. Default `False` to avoid blocking
       in notebooks run with papermill / CI / non-interactive contexts.

    Args:
        service_account_json: Path to the service account JSON. If None or it
            does not exist, falls back to ADC.
        project: GCP project ID associated with the EE quota (required for
            Cloud-registered projects since 2024).
        interactive_auth: If True and everything else fails, triggers
            `ee.Authenticate()` (opens browser, requires intervention). Default
            False — the caller must run `earthengine authenticate` or
            generate the service account outside the process.

    Raises:
        ImportError: If `earthengine-api` is not installed.
        ee.EEException / RuntimeError: If `ee.Initialize` fails and
            `interactive_auth=False`.
    """
    if ee is None:
        raise ImportError("earthengine-api is not installed. Run `poetry install --with ml,geo`.")
    sa_path = Path(service_account_json) if service_account_json is not None else None
    if sa_path is not None and sa_path.is_file():
        creds = ee.ServiceAccountCredentials(  # type: ignore[attr-defined]
            email=None, key_file=str(sa_path)
        )
        ee.Initialize(creds, project=project)
        return
    try:
        ee.Initialize(project=project)
    except Exception:
        if not interactive_auth:
            raise
        ee.Authenticate()
        ee.Initialize(project=project)


def _cache_key(roi_name: str, start_date: str, end_date: str, n_pixels: int) -> str:
    """Generate a reproducible cache filename."""
    return f"{roi_name}_{start_date}_{end_date}_{n_pixels}.parquet"


def sample_s2_roi(
    roi: Any,
    start_date: str,
    end_date: str,
    bands: list[str] | None = None,
    n_pixels: int = 100_000,
    cloud_pct_max: int = 30,
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 10,
) -> pl.DataFrame:
    """Sample Sentinel-2 L2A over a ROI with local parquet cache.

    Args:
        roi: `ee.Geometry` or `ee.FeatureCollection` defining the region.
        start_date: Start date in `YYYY-MM-DD` format.
        end_date: End date in `YYYY-MM-DD` format.
        bands: Bands to extract (default S2 SR without atmospheric ones).
        n_pixels: Approximate total number of pixels to sample.
        cloud_pct_max: Maximum `CLOUDY_PIXEL_PERCENTAGE` to filter images.
        cache_path: Cache folder (default `data/cache/gee/`).
        roi_name: Logical name of the ROI used in cache + column `roi`.
        scale: Resolution in meters for `sampleRegions`.

    Returns:
        Polars DataFrame with columns `roi, date, band, value, lon, lat`.
        If EE fails (auth/quota), returns an empty DataFrame with correct schema.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _cache_key(roi_name, start_date, end_date, n_pixels)
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None:
        return pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )

    selected = bands or DEFAULT_S2_BANDS

    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .select(selected)
        )
        median = collection.median()
        sample = median.sample(
            region=roi,
            scale=scale,
            numPixels=n_pixels,
            geometries=True,
            seed=42,
        )
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        # Quota / auth / network — degraded mode: return an empty DataFrame
        # so the EDA notebook is not blocked. Logging done in notebook via print.
        return pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )

    rows: list[dict[str, Any]] = []
    composite_date = f"{start_date}__{end_date}"
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        lon, lat = (coords[0], coords[1]) if len(coords) >= 2 else (None, None)
        for band in selected:
            if band in props and props[band] is not None:
                rows.append(
                    {
                        "roi": roi_name,
                        "date": composite_date,
                        "band": band,
                        "value": float(props[band]),
                        "lon": float(lon) if lon is not None else None,
                        "lat": float(lat) if lat is not None else None,
                    }
                )

    df = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )
    )
    if not df.is_empty():
        df.write_parquet(cache_file)
    return df


def _alphaearth_empty_schema() -> dict[str, Any]:
    """Canonical schema of the AlphaEarth DataFrame (64 dims)."""
    base: dict[str, Any] = {
        "px_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "roi": pl.Utf8,
        "year": pl.Int64,
    }
    for col in ALPHAEARTH_DIM_COLS:
        base[col] = pl.Float64
    return base


def _alphaearth_band_names() -> list[str]:
    """Conventional AlphaEarth band names (`A00`..`A63`).

    Validated with `ee.ImageCollection(...).first().bandNames()` at runtime,
    but the documented pattern is `A{ii}`.
    """
    return [f"A{i:02d}" for i in range(64)]


def sample_alphaearth_roi(
    roi: Any,
    year: int,
    n_pixels: int = 100_000,
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 10,
) -> pl.DataFrame:
    """Sample the 64-dim AlphaEarth embedding over a ROI/year with parquet cache.

    Args:
        roi: `ee.Geometry` or `ee.FeatureCollection` delimiting the region.
        year: Year (2017-2025) — selects the corresponding annual image.
        n_pixels: Number of pixels to sample via `sample(numPixels=...)`.
        cache_path: Local cache folder (default `data/cache/gee/`).
        roi_name: Logical name of the ROI used in cache and column `roi`.
        scale: Resolution in meters (AlphaEarth native = 10).

    Returns:
        Polars DataFrame with columns `px_id, lon, lat, roi, year, dim_00..dim_63`.
        If EE is unavailable or fails, returns an empty DataFrame with valid schema.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"alphaearth_{roi_name}_{year}_{n_pixels}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    empty_schema = _alphaearth_empty_schema()
    if ee is None:
        return pl.DataFrame(schema=empty_schema)

    band_names = _alphaearth_band_names()
    try:
        collection = (
            ee.ImageCollection(ALPHAEARTH_COLLECTION)
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        )
        # mosaic() vs first(): AlphaEarth distributes the annual embedding as
        # disjoint tiles (~10k images/year in Europe). first() returns a
        # single image with limited footprint -> pixels outside that tile
        # fall to null. mosaic() merges all tiles of the year touching the ROI
        # into a continuous raster. Since each (px, year) has a single value,
        # mosaic() introduces no ambiguity.
        image = collection.mosaic().select(band_names)
        sample = image.sample(
            region=roi,
            scale=scale,
            numPixels=n_pixels,
            geometries=True,
            seed=42,
        )
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=empty_schema)

    rows: list[dict[str, Any]] = []
    for idx, feat in enumerate(info.get("features", [])):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None])
        lon = float(coords[0]) if len(coords) >= 2 and coords[0] is not None else None
        lat = float(coords[1]) if len(coords) >= 2 and coords[1] is not None else None
        row: dict[str, Any] = {
            "px_id": f"{roi_name}_{year}_{idx}",
            "lon": lon,
            "lat": lat,
            "roi": roi_name,
            "year": int(year),
        }
        for i, band in enumerate(band_names):
            val = props.get(band)
            row[ALPHAEARTH_DIM_COLS[i]] = float(val) if val is not None else None
        rows.append(row)

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    df = pl.DataFrame(rows, schema=empty_schema)
    df.write_parquet(cache_file)
    return df


def sample_alphaearth_aoi_mean(
    geometry: dict[str, Any],
    year: int,
    project: str | None = None,
    service_account_json: Path | None = None,
    scale: int = 10,
) -> np.ndarray | None:
    """Sample the mean 64-dim AlphaEarth embedding over a single AOI polygon.

    Live, on-demand counterpart of `sample_alphaearth_roi` for one drawn polygon:
    it initializes Earth Engine (service account or ADC), mosaics the annual
    `SATELLITE_EMBEDDING/V1/ANNUAL` tiles touching the AOI for `year` and reduces
    them to the per-band spatial mean over the polygon. Returns a `(64,)` vector
    aligned to `dim_00..dim_63` (band order `A00..A63`).

    Used by the conversational `classify_new_parcel` tool as the "download"
    branch: when no persisted parcel embedding intersects the AOI, the embedding
    is sampled here instead of returning the `needs_gee_sampling` sentinel.

    Args:
        geometry: GeoJSON geometry (`{"type": ..., "coordinates": ...}`) in
            EPSG:4326 delimiting the AOI.
        year: Year (2017-2025) selecting the annual embedding image.
        project: GCP project id with the EE quota (e.g. `agrosat-copilot`).
        service_account_json: Optional service-account key; falls back to ADC.
        scale: Reduction resolution in meters (AlphaEarth native = 10).

    Returns:
        A `(64,)` `float64` embedding, or `None` when EE is unavailable,
        initialization fails, the AOI has no coverage, or any band is null.
    """
    if ee is None:
        _log.warning("alphaearth_aoi_ee_missing")
        return None
    band_names = _alphaearth_band_names()
    try:
        init_ee(service_account_json=service_account_json, project=project)
        roi = ee.Geometry(geometry)
        image = (
            ee.ImageCollection(ALPHAEARTH_COLLECTION)
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic()
            .select(band_names)
        )
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=int(1e9),
            bestEffort=True,
        ).getInfo()
    except Exception as exc:  # noqa: BLE001
        _log.warning("alphaearth_aoi_sample_failed", year=int(year), error=str(exc))
        return None

    if not stats:
        _log.warning("alphaearth_aoi_no_coverage", year=int(year))
        return None
    values: list[float] = []
    for band in band_names:
        val = stats.get(band)
        if val is None:
            _log.warning("alphaearth_aoi_null_band", band=band, year=int(year))
            return None
        values.append(float(val))
    return np.asarray(values, dtype=np.float64)


def sample_alphaearth_at_coords(
    coords: pl.DataFrame,
    year: int,
    cache_path: Path | None = None,
    cache_key: str = "coords",
    scale: int = 10,
    batch_size: int = 500,
) -> pl.DataFrame:
    """Sample 64-dim AlphaEarth at given (lon, lat) EPSG:4326 coordinates.

    Useful for joining with external labels (e.g. PASTIS-R). Internally builds
    an `ee.FeatureCollection` from the DataFrame and calls `reduceRegions` with
    `ee.Reducer.first()` in batches of `batch_size` points.

    Args:
        coords: DataFrame with columns `px_id, lon, lat` in EPSG:4326.
        year: Year of the annual embedding to query.
        cache_path: Local parquet cache folder.
        cache_key: Logical identifier for the cache (e.g. `pastis_fold1`).
        scale: Resolution in meters (default 10).
        batch_size: Batch size per server-side request. AlphaEarth has
            64 bands, so the payload is ~64x larger than DW; we keep
            500 points per batch to avoid timeouts.

    Returns:
        DataFrame with columns `px_id, lon, lat, year, dim_00..dim_63`.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"alphaearth_at_{cache_key}_{year}_{coords.height}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    schema: dict[str, Any] = {
        "px_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "year": pl.Int64,
    }
    for col in ALPHAEARTH_DIM_COLS:
        schema[col] = pl.Float64

    if ee is None or coords.is_empty():
        return pl.DataFrame(schema=schema)

    band_names = _alphaearth_band_names()
    collection = ee.ImageCollection(ALPHAEARTH_COLLECTION).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    # mosaic() vs first(): the AlphaEarth collection has ~10k tiles/year.
    # first() returns an arbitrary tile with limited footprint -> points
    # outside it fall to null. mosaic() merges all tiles of the year into
    # a continuous raster. Each (px, year) has a single canonical value,
    # so mosaic() is deterministic.
    image = collection.mosaic().select(band_names)

    by_id: dict[str, dict[str, float | None]] = {
        str(r["px_id"]): {"lon": float(r["lon"]), "lat": float(r["lat"])}
        for r in coords.iter_rows(named=True)
    }

    rows: list[dict[str, Any]] = []
    total = coords.height
    for start in range(0, total, batch_size):
        chunk = coords.slice(start, batch_size)
        try:
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                    {"px_id": str(r["px_id"])},
                )
                for r in chunk.iter_rows(named=True)
            ]
            fc = ee.FeatureCollection(features)
            sampled = image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=scale,
            )
            info = sampled.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "alphaearth_batch_failed",
                start=start,
                size=chunk.height,
                year=int(year),
                error=str(exc),
            )
            continue

        for feat in info.get("features", []):
            props = feat.get("properties", {}) or {}
            pid = str(props.get("px_id", ""))
            geo = by_id.get(pid, {"lon": None, "lat": None})
            row: dict[str, Any] = {
                "px_id": pid,
                "lon": geo["lon"],
                "lat": geo["lat"],
                "year": int(year),
            }
            for i, band in enumerate(band_names):
                val = props.get(band)
                row[ALPHAEARTH_DIM_COLS[i]] = float(val) if val is not None else None
            rows.append(row)

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_dynamic_world_at(
    coords: pl.DataFrame,
    year: int,
    cache_path: Path | None = None,
    cache_key: str = "coords",
    scale: int = 10,
    batch_size: int = 500,
) -> pl.DataFrame:
    """Extract the Dynamic World mode class of the given year for each (lon, lat).

    Processes coords in batches of `batch_size` points to avoid timeouts in
    the GEE server-side compute graph (a `reduceRegions` with >1000 points
    often exceeds the 5 min limit and returns an empty/partial response,
    leaving all rows with `dw_class_id=-1`).

    Args:
        coords: DataFrame with columns `px_id, lon, lat` in EPSG:4326.
        year: Year to filter the Dynamic World collection.
        cache_path: Parquet cache folder.
        cache_key: Logical identifier for the cache.
        scale: Resolution in meters.
        batch_size: Maximum number of points per `reduceRegions` request.
            Default 500 — empirically safe for Italy with DW 2024.

    Returns:
        DataFrame with columns `px_id, dw_class_id, dw_class_name, dw_confidence`.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"dw_at_{cache_key}_{year}_{coords.height}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    schema: dict[str, Any] = {
        "px_id": pl.Utf8,
        "dw_class_id": pl.Int16,
        "dw_class_name": pl.Utf8,
        "dw_confidence": pl.Float64,
    }

    if ee is None or coords.is_empty():
        return pl.DataFrame(schema=schema)

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(["label"])
    )
    mode_img = collection.mode()

    rows: list[dict[str, Any]] = []
    total = coords.height
    for start in range(0, total, batch_size):
        chunk = coords.slice(start, batch_size)
        try:
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                    {"px_id": str(r["px_id"])},
                )
                for r in chunk.iter_rows(named=True)
            ]
            fc = ee.FeatureCollection(features)
            sampled = mode_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=scale,
            )
            info = sampled.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "dynamic_world_batch_failed",
                start=start,
                size=chunk.height,
                year=int(year),
                error=str(exc),
            )
            # Failed batch: annotate px_id with class -1 so the join is not corrupted.
            for r in chunk.iter_rows(named=True):
                rows.append(
                    {
                        "px_id": str(r["px_id"]),
                        "dw_class_id": -1,
                        "dw_class_name": "unknown",
                        "dw_confidence": 0.0,
                    }
                )
            continue

        for feat in info.get("features", []):
            props = feat.get("properties", {}) or {}
            pid = str(props.get("px_id", ""))
            # reduceRegions with ee.Reducer.first() renames the band to "first";
            # with sampleRegions it would be "label". We cover both for compatibility.
            cls_val = props.get("first", props.get("label"))
            cls_id = int(cls_val) if cls_val is not None else -1
            rows.append(
                {
                    "px_id": pid,
                    "dw_class_id": cls_id,
                    "dw_class_name": DYNAMIC_WORLD_CLASSES.get(cls_id, "unknown"),
                    "dw_confidence": float(props.get("confidence", 1.0)),
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def fetch_s2_ndvi_rgb_for_parcel(
    parcel_geom: Any,
    date: str,
    cloud_pct_max: int = 20,
    scale: int = 10,
    max_pixels: int = 1_000_000,
) -> dict[str, Any]:
    """Return Sentinel-2 RGB + NDVI for a parcel at an approximate date.

    Takes a +/- 15 day window around `date` and returns the median of the
    cloud-filtered collection. If EE is unavailable or fails, returns empty
    arrays so the caller can degrade gracefully.

    Args:
        parcel_geom: `ee.Geometry` of the parcel.
        date: Central date `YYYY-MM-DD`.
        cloud_pct_max: Maximum `CLOUDY_PIXEL_PERCENTAGE`.
        scale: Resolution in meters.
        max_pixels: Limit of pixels to retrieve.

    Returns:
        Dictionary with keys:
            - `rgb`: ndarray (H, W, 3) float, values in [0, 1] after stretch.
            - `ndvi`: ndarray (H, W) float, values in [-1, 1].
            - `date_used`: str of the actual date used.
    """
    empty = {
        "rgb": np.zeros((0, 0, 3), dtype=np.float32),
        "ndvi": np.zeros((0, 0), dtype=np.float32),
        "date_used": "",
    }
    if ee is None:
        return empty

    from datetime import datetime, timedelta

    try:
        center = datetime.strptime(date, "%Y-%m-%d")
        start = (center - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=15)).strftime("%Y-%m-%d")
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(parcel_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .select(["B2", "B3", "B4", "B8"])
        )
        # median() collapses the S2 native projection (UTM 10m) to the default
        # (EPSG:4326 with scale=1 degree/pixel). Without reprojecting, sampleRectangle
        # returns a single pixel because a 0.01deg bbox << 1 degree of scale.
        # We reproject to the first image's projection (S2 native UTM at 10m).
        ref_proj = collection.first().select("B4").projection()
        median = collection.median().reproject(crs=ref_proj, scale=scale)
        sample = median.sampleRectangle(region=parcel_geom, defaultValue=0)
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        return empty

    props = info.get("properties", {}) if isinstance(info, dict) else {}
    try:
        b2 = np.asarray(props.get("B2", []), dtype=np.float32)
        b3 = np.asarray(props.get("B3", []), dtype=np.float32)
        b4 = np.asarray(props.get("B4", []), dtype=np.float32)
        b8 = np.asarray(props.get("B8", []), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return empty

    if b4.size == 0 or b8.size == 0:
        return empty

    denom = np.where((b8 + b4) == 0, 1.0, b8 + b4)
    ndvi = (b8 - b4) / denom

    # 2-98 percentile stretch PER BAND before the stack. Applying the global
    # stretch to the RGB after stacking collapses the dynamic range of the
    # lower-magnitude bands, producing an apparently uniform image when
    # B4 (red) >> B3 (green) >> B2 (blue) over vegetated surfaces.
    def _stretch(band: np.ndarray) -> np.ndarray:
        if band.size == 0:
            return band
        lo, hi = np.percentile(band, [2.0, 98.0])
        stretched: np.ndarray = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return stretched

    rgb = np.stack([_stretch(b4), _stretch(b3), _stretch(b2)], axis=-1)
    _ = max_pixels
    return {"rgb": rgb.astype(np.float32), "ndvi": ndvi.astype(np.float32), "date_used": date}


def era5_annual_precip(
    roi: Any,
    years: list[int],
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 11132,
) -> pl.DataFrame:
    """Accumulate ERA5-Land annual total precipitation over a ROI with parquet cache.

    Aggregates `total_precipitation_sum` (meters) over the full year via
    `reduceRegion(ee.Reducer.mean())` on the temporal axis and then multiplies
    by 1000 to report in mm. ERA5-Land native resolution ~11132 m.

    Args:
        roi: `ee.Geometry` delimiting the region.
        years: List of integer years to process (e.g. `[2018, 2019, 2020]`).
        cache_path: Local cache folder (default `data/cache/gee/`).
        roi_name: Logical name of the ROI used in cache and column `roi_name`.
        scale: Resolution in meters for `reduceRegion` (default 11132 = native).

    Returns:
        Polars DataFrame with columns `year, roi_name, precip_mm`. If EE is
        unavailable or fails, returns an empty DataFrame with valid schema
        to degrade the notebook without breaking the Polars chain.
    """
    schema: dict[str, Any] = {
        "year": pl.Int64,
        "roi_name": pl.Utf8,
        "precip_mm": pl.Float64,
    }
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    years_key = "-".join(str(y) for y in sorted(years)) if years else "none"
    cache_file = cache_dir / f"era5_precip_{roi_name}_{years_key}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or not years:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    try:
        for year in years:
            collection = (
                ee.ImageCollection(ERA5_COLLECTION)
                .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                .select(["total_precipitation_sum"])
            )
            # ee.Reducer.sum over the temporal axis accumulates the daily
            # precipitation of the full year (meters). Then we reduce spatially.
            annual_img = collection.sum()
            stat = annual_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=scale,
                maxPixels=1_000_000_000,
            )
            info = stat.getInfo() or {}
            precip_m = info.get("total_precipitation_sum")
            if precip_m is None:
                continue
            rows.append(
                {
                    "year": int(year),
                    "roi_name": roi_name,
                    "precip_mm": float(precip_m) * 1000.0,
                }
            )
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=schema)

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


# ===========================================================================
# US-016 — new samplers (Sentinel-1, SRTM, monthly ERA5) per parcel.
# ===========================================================================
#
# Common convention:
# - `parcels` is a `gpd.GeoDataFrame` with columns `parcel_id`, `geometry`
#   (POLYGON EPSG:4326) and optionally `year`. Each parcel is converted to
#   `ee.Geometry` server-side and reduced with `ee.Reducer.mean()`.
# - Polars outputs with local parquet cache in `data/cache/gee/` (same pattern
#   as `sample_alphaearth_*` and `era5_annual_precip`).
# - Degraded mode: if `ee` is not available or GEE fails, an empty DataFrame
#   with the correct schema is returned so the Polars chain is not broken
#   in the rest of the pipeline (the blocks in `ml/features/fusion.py` fill
#   the missing cols with None).


def _parcels_to_feature_collection(parcels: Any) -> Any:
    """Convert a parcels GeoDataFrame to ``ee.FeatureCollection``.

    Args:
        parcels: GeoDataFrame with `parcel_id` and `geometry` POLYGON EPSG:4326.

    Returns:
        ``ee.FeatureCollection`` with a `parcel_id` property per feature.
    """
    if ee is None:
        raise ImportError("earthengine-api is not available.")
    features = []
    for row in parcels.itertuples(index=False):
        geom = getattr(row, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        gj = geom.__geo_interface__
        ee_geom = ee.Geometry(gj)
        features.append(ee.Feature(ee_geom, {"parcel_id": int(row.parcel_id)}))
    return ee.FeatureCollection(features)


def sample_s1_roi_for_parcels(
    parcels: Any,
    year: int,
    *,
    polarization: tuple[str, ...] = ("VV", "VH"),
    orbit_pass: Literal["both", "ascending", "descending"] = "both",  # noqa: S107
    despeckle: Literal["lee_7x7", "none"] = "lee_7x7",
    sigma0_units: Literal["dB", "linear"] = "dB",
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Sample Sentinel-1 GRD VV+VH per parcel with temporal stats (US-016 AC-4).

    Operational preset:

    - Collection ``COPERNICUS/S1_GRD`` mode IW (Interferometric Wide).
    - ``ascending + descending`` mosaicked (``orbit_pass="both"``),
      10 m resolution.
    - Lee 7x7 despeckle (default) applied per image before the stack.
    - Output in sigma0 dB (default; ``"linear"`` for raw data).

    Stats returned per (parcel_id, polarization): ``mean, std, p25, p50,
    p95`` over the temporal stack -> 5 stats x 2 pol = 10 columns with
    prefixes ``s1_vv_*`` and ``s1_vh_*``.

    Args:
        parcels: GeoDataFrame with `parcel_id` and `geometry` POLYGON EPSG:4326.
        year: Year to sample (window ``[YYYY-01-01, (YYYY+1)-01-01)``).
        polarization: Polarizations to extract (default ``("VV", "VH")``).
        orbit_pass: Orbital pass filter. ``"both"`` mosaicks asc+desc.
        despeckle: Speckle filter. ``"lee_7x7"`` applies a Lee filter with a
            7x7 kernel; ``"none"`` disables the filter.
        sigma0_units: Output units; GEE GRD data already comes in dB.
            If ``"linear"`` is requested, ``10^(x/10)`` is applied.
        cache_dir: Local cache folder (default ``data/cache/gee/``).
        cache_key: Logical name of the subset for the cache.

    Returns:
        ``pl.DataFrame`` with columns ``parcel_id, year, s1_vv_mean, ...,
        s1_vh_p95``. Returns an empty frame with valid schema if GEE fails.
    """
    pol_cols: list[str] = []
    for pol in polarization:
        for stat in ("mean", "std", "p25", "p50", "p95"):
            pol_cols.append(f"s1_{pol.lower()}_{stat}")
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "year": pl.Int16,
        **{c: pl.Float64 for c in pol_cols},
    }

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = (
        cache_root / f"s1_{cache_key}_{year}_{orbit_pass}_{despeckle}_{sigma0_units}.parquet"
    )
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(list(polarization))
        )
        if orbit_pass == "ascending":  # noqa: S105
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        elif orbit_pass == "descending":  # noqa: S105
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))

        if despeckle == "lee_7x7":
            # focalMean with a square kernel of radius 3 (7x7 pixels).
            kernel = ee.Kernel.square(radius=3, units="pixels")
            collection = collection.map(lambda img: img.focalMean(kernel=kernel))

        if sigma0_units == "linear":
            collection = collection.map(
                lambda img: img.expression("pow(10, x / 10)", {"x": img.select(list(polarization))})
            )

        rows: list[dict[str, Any]] = []
        year_int = int(year)
        for pol in polarization:
            pol_col = collection.select(pol)
            stats_img = pol_col.reduce(
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(
                    ee.Reducer.percentile([25, 50, 95]),
                    sharedInputs=True,
                )
            )
            reduced = stats_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=10,
            )
            info = reduced.getInfo()
            for feat in info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props["parcel_id"])
                row: dict[str, Any] = {"parcel_id": pid, "year": year_int}
                row[f"s1_{pol.lower()}_mean"] = _safe_float(props.get(f"{pol}_mean"))
                row[f"s1_{pol.lower()}_std"] = _safe_float(props.get(f"{pol}_stdDev"))
                row[f"s1_{pol.lower()}_p25"] = _safe_float(props.get(f"{pol}_p25"))
                row[f"s1_{pol.lower()}_p50"] = _safe_float(props.get(f"{pol}_p50"))
                row[f"s1_{pol.lower()}_p95"] = _safe_float(props.get(f"{pol}_p95"))
                rows.append(row)
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        _log.warning("s1_sample_failed", error=str(exc), year=int(year))
        return pl.DataFrame(schema=schema)

    if not rows:
        return pl.DataFrame(schema=schema)

    # Merge VV and VH rows of the same parcel.
    merged: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = int(row["parcel_id"])
        if pid not in merged:
            merged[pid] = {"parcel_id": pid, "year": int(year)}
        for k, v in row.items():
            if k in ("parcel_id", "year"):
                continue
            if v is not None:
                merged[pid][k] = v
    df = pl.DataFrame(list(merged.values()), schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_srtm_terrain(
    parcels: Any,
    *,
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Sample SRTM elevation + slope + dominant aspect per parcel (US-016 AC-5).

    Uses ``USGS/SRTMGL1_003`` (global 30m DEM) plus ``ee.Terrain.slope`` and
    ``ee.Terrain.aspect``. ``aspect_dominant`` is discretized into 8 cardinal
    quadrants (N, NE, ..., NW) using the center of each 45 degree bin.

    Args:
        parcels: GeoDataFrame with `parcel_id` and `geometry` POLYGON EPSG:4326.
        cache_dir: Local cache folder.
        cache_key: Logical name for the cache.

    Returns:
        ``pl.DataFrame`` with cols ``parcel_id, srtm_elev_mean,
        srtm_slope_mean, srtm_aspect_dominant`` (cardinal string).
    """
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "srtm_elev_mean": pl.Float64,
        "srtm_slope_mean": pl.Float64,
        "srtm_aspect_dominant": pl.Utf8,
    }
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"srtm_{cache_key}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        dem = ee.Image(SRTM_COLLECTION).select("elevation")
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        composite = dem.addBands(slope.rename("slope")).addBands(aspect.rename("aspect"))
        reduced = composite.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=30,
        )
        info = reduced.getInfo()
    except Exception as exc:  # noqa: BLE001
        _log.warning("srtm_sample_failed", error=str(exc))
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for feat in info.get("features", []) or []:
        props = feat.get("properties", {}) or {}
        pid = int(props["parcel_id"])
        aspect_deg = _safe_float(props.get("aspect"))
        rows.append(
            {
                "parcel_id": pid,
                "srtm_elev_mean": _safe_float(props.get("elevation")),
                "srtm_slope_mean": _safe_float(props.get("slope")),
                "srtm_aspect_dominant": _aspect_to_cardinal(aspect_deg),
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_alphaearth_for_parcels(
    parcels: Any,
    year: int,
    *,
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
    batch_size: int = 100,
    scale: int = 10,
) -> pl.DataFrame:
    """Sample 64-dim AlphaEarth per parcel polygon (US-018.3).

    Variant of ``sample_alphaearth_at_coords`` that operates on polygons
    instead of points. Uses ``ee.Reducer.mean()`` to aggregate the embedding
    over each individual parcel. Useful for PASTIS-R vectorized by parcel
    (not by patch centroid).

    Unlike the other parcel samplers (``sample_srtm_terrain``,
    ``sample_s1_roi_for_parcels``), here ``parcel_id`` may be a string
    (format ``"<patch_id>_<instance_id>"``) because PASTIS-R has duplicate
    IDs across patches. Internally we reassign to sequential integers so
    GEE can return them in the properties.

    Args:
        parcels: GeoDataFrame with `parcel_id` (str or int) and `geometry`
            POLYGON EPSG:4326.
        year: Year of the annual embedding to query.
        cache_dir: Local cache folder.
        cache_key: Logical name for the cache.
        batch_size: Batch size per server-side request. Polygons are more
            expensive than points; default 100 avoids timeouts.
        scale: Sampling resolution (default 10 m, AlphaEarth native).

    Returns:
        ``pl.DataFrame`` with columns ``parcel_id, year, dim_00..dim_63``.
        Empty frame with valid schema if GEE fails.
    """
    schema: dict[str, Any] = {
        "parcel_id": pl.Utf8,
        "year": pl.Int64,
    }
    for col in ALPHAEARTH_DIM_COLS:
        schema[col] = pl.Float64

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"alphaearth_parcels_{cache_key}_{year}_{len(parcels)}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    band_names = _alphaearth_band_names()
    image = (
        ee.ImageCollection(ALPHAEARTH_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(band_names)
    )

    # Reassign parcel_id (potentially a string) to a sequential int so that
    # GEE returns it in properties. Inverse mapping at the end.
    int_to_str: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    total = len(parcels)

    for start in range(0, total, batch_size):
        chunk = parcels.iloc[start : start + batch_size]
        try:
            features = []
            for offset, row in enumerate(chunk.itertuples(index=False)):
                geom = getattr(row, "geometry", None)
                if geom is None or geom.is_empty:
                    continue
                seq_id = start + offset
                int_to_str[seq_id] = str(row.parcel_id)
                features.append(
                    ee.Feature(
                        ee.Geometry(geom.__geo_interface__),
                        {"seq_id": int(seq_id)},
                    )
                )
            if not features:
                continue
            fc = ee.FeatureCollection(features)
            sampled = image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=scale,
            )
            info = sampled.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "alphaearth_parcels_batch_failed",
                start=start,
                size=len(chunk),
                year=int(year),
                error=str(exc),
            )
            continue

        for feat in info.get("features", []):
            props = feat.get("properties", {}) or {}
            seq_id = int(props.get("seq_id", -1))
            pid_str = int_to_str.get(seq_id)
            if pid_str is None:
                continue
            row_out: dict[str, Any] = {
                "parcel_id": pid_str,
                "year": int(year),
            }
            for i, band in enumerate(band_names):
                val = props.get(band)
                row_out[ALPHAEARTH_DIM_COLS[i]] = float(val) if val is not None else None
            rows.append(row_out)

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_era5_monthly_climate(
    parcels: Any,
    year: int,
    *,
    temperature_units: Literal["K", "C"] = "C",
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Sample monthly ERA5-Land: tmean (12) + accumulated prec (12) (US-016 AC-6).

    Uses ``ECMWF/ERA5_LAND/DAILY_AGGR`` grouping server-side by month:

    - ``temperature_2m`` reduced with ``mean()`` per month -> degrees C if
      ``temperature_units="C"``.
    - ``total_precipitation_sum`` reduced with ``sum()`` per month (meters)
      -> accumulated mm (multiplied x 1000).

    Args:
        parcels: GeoDataFrame with `parcel_id` and `geometry` POLYGON EPSG:4326.
        year: Year (generates window ``[YYYY-01-01, (YYYY+1)-01-01)``).
        temperature_units: ``"C"`` (default) or ``"K"``.
        cache_dir: Local cache folder.
        cache_key: Logical name for the cache.

    Returns:
        ``pl.DataFrame`` with cols ``parcel_id, year, era5_tmean_m01..m12,
        era5_prec_m01..m12`` (24 cols).
    """
    t_cols = [f"era5_tmean_m{m:02d}" for m in range(1, 13)]
    p_cols = [f"era5_prec_m{m:02d}" for m in range(1, 13)]
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "year": pl.Int16,
        **{c: pl.Float64 for c in t_cols + p_cols},
    }
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"era5_monthly_{cache_key}_{year}_{temperature_units}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        result_rows: dict[int, dict[str, Any]] = {}
        # Bug-6 fix (real smoke test, 2026-05-17):
        # reduceRegions with scale=11132 (ERA5-Land native ~11 km/pixel) and
        # sub-pixel parcels (~1 km2 from the demo fixture) does NOT intersect any
        # pixel: the payload omits the reduced property entirely and only
        # `{parcel_id}` remains. We lower scale to 1000 m (11x oversampling) and
        # add `tileScale=4` to avoid memory errors on large parcels.
        # Result: scale=1000 correctly interpolates the containing ERA5 pixel
        # and fills the 24 cols with physically plausible values.
        # Also, depending on the scale, GEE renames the property to the original
        # band (`temperature_2m` / `total_precipitation_sum`) or to `mean`;
        # we read the band-name with a fallback to `mean` to cover both paths.
        tband = "temperature_2m"
        pband = "total_precipitation_sum"
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
            month_collection = ee.ImageCollection(ERA5_COLLECTION).filterDate(start, end)
            tmean_img = month_collection.select(tband).mean()
            prec_img = month_collection.select(pband).sum()

            tmean_reduced = tmean_img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4
            )
            prec_reduced = prec_img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4
            )
            t_info = tmean_reduced.getInfo()
            p_info = prec_reduced.getInfo()
            for feat in t_info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props["parcel_id"])
                if pid not in result_rows:
                    result_rows[pid] = {"parcel_id": pid, "year": int(year)}
                # `reduceRegions` with a single-band image renames the property
                # to the band name (not to "mean"); fallback to "mean" for
                # compatibility with old mocks / multi-band reducers.
                tval = _safe_float(props.get(tband, props.get("mean")))
                if tval is not None and temperature_units == "C":
                    tval = tval - 273.15
                result_rows[pid][f"era5_tmean_m{month:02d}"] = tval
            for feat in p_info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props["parcel_id"])
                if pid not in result_rows:
                    result_rows[pid] = {"parcel_id": pid, "year": int(year)}
                pval = _safe_float(props.get(pband, props.get("mean")))
                # `pval` comes from the monthly `sum()` (meters) reduced in
                # space; we multiply by 1000 to report in mm.
                result_rows[pid][f"era5_prec_m{month:02d}"] = (
                    pval * 1000.0 if pval is not None else None
                )
    except Exception as exc:  # noqa: BLE001
        _log.warning("era5_monthly_sample_failed", error=str(exc), year=int(year))
        return pl.DataFrame(schema=schema)

    if not result_rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(list(result_rows.values()), schema=schema)
    df.write_parquet(cache_file)
    return df


def _safe_float(val: Any) -> float | None:
    """Convert a value to float or return None if it is null/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _aspect_to_cardinal(aspect_deg: float | None) -> str | None:
    """Discretize an angle ``[0, 360)`` into one of 8 cardinal quadrants.

    Bins centered on each cardinal (N=0, NE=45, ..., NW=315) with 45 degree
    width. ``None`` or NaN returns ``None``.
    """
    if aspect_deg is None:
        return None
    deg = float(aspect_deg) % 360.0
    idx = int(((deg + 22.5) // 45.0) % 8)
    return _ASPECT_CARDINALS[idx]
