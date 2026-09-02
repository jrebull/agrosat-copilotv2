"""Materialize the FR/ES domain-gap evidence for the multi-region paper (US-073).

Produces the two real artefacts that the dense France -> Catalonia transfer section
(B-073-1 UMAP, B-073-2 NDVI phenology) depends on, with NO fabricated values:

1. **Per-class AlphaEarth embeddings of Catalonia (ES)** sampled at real
   Sen4AgriNet 31TCG pixel centroids. The Sen4AgriNet ``.nc`` patches are HDF5
   (NetCDF4 with groups) georeferenced in EPSG:32631 (UTM 31N); we read them with
   :mod:`h5py`, reproject the per-class pixel centroids to EPSG:4326 with
   :mod:`pyproj`, and pull the 64-dim AlphaEarth Satellite Embedding V1 Annual
   (v1.1) at those coordinates via the existing
   :func:`ml.ingest.gee_sampler.sample_alphaearth_at_coords`. The France (FR) side
   reuses the already-cached PASTIS parcel embeddings (US-011) with their 18
   PASTIS classes collapsed to the shared HCAT macro label-space.

2. **Per-class Sentinel-2 NDVI temporal series** for both regions, computed with
   :func:`ml.transfer.mexico_demo.extract_s2_ndvi_series` over a small AOI per
   macro-class (one representative AOI in PASTIS-FR tile 31TCJ vs Catalonia tile
   31TCG). The seasonal NDVI offset (sowing/harvest shifted by latitude and
   climate) is the qualitative half of the Franco-Iberian domain gap; the dense
   transfer mIoU collapse (zero-shot 0.0 -> few-shot 0.2468) is the quantitative
   half (``reports/segmentation/sen4agrinet_transfer_result.json``).

Attributions (mandatory, also in the figure captions):
- AlphaEarth: Brown/Khanna et al., "AlphaEarth Foundations", GEE
  ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` (v1.1), CC-BY-4.0.
- Sen4AgriNet (ES transfer target): Sykas et al. 2022, CC-BY-SA-4.0.
- PASTIS-R (FR source): Garnot & Landrieu, ICCV 2021.

Project conventions: Polars, ``structlog``, type hints, English docstrings,
Spanish visible prose in figures, no emojis, never fabricate a missing value.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

from ml.data.sen4agrinet_adapter import FAO_ICC_TO_MACRO

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Sen4AgriNet Catalonia (ES) subset patches (tile 31TCG, EPSG:32631 / UTM 31N).
ES_PATCH_GLOB = "data/sen4agrinet/data/*/31TCG/*.nc"
#: Sen4AgriNet France (FR) subset patches (tile 31TCJ, EPSG:32631 / UTM 31N). Same
#: schema and labels as ES, so FR and ES are sampled from one homogeneous source.
FR_PATCH_GLOB = "data/sen4agrinet/data/*/31TCJ/*.nc"
#: Cached PASTIS-FR parcel AlphaEarth embeddings with 18-class labels (US-011),
#: kept as an alternative FR embedding source (parcel-level, no coordinates).
FR_EMBED_PARQUET = "data/cache/gee/alphaearth_pastis_parcels_2019_85951_enriched.parquet"

#: Default output locations.
DATA_OUT_DIR = Path("data/transfer")
FIGURES_DIR = Path("paper/figures/us-073")

#: AlphaEarth annual embedding year aligned with the Sen4AgriNet 2019 patches.
ALPHAEARTH_YEAR = 2019
#: Sen4AgriNet labels CRS (UTM 31N) -> EPSG:4326 for GEE point sampling.
_ES_CRS = "EPSG:32631"
_LONLAT_CRS = "EPSG:4326"

#: PASTIS-18 class name -> shared HCAT macro group (FR side of the gap). Mirrors
#: ``ml.data.hcat_crosswalk`` macro semantics; "void"-like classes are dropped so
#: only the macro groups that the ES side also has are compared apples-to-apples.
PASTIS_NAME_TO_MACRO: dict[str, str] = {
    "Meadow": "grassland",
    "Soft winter wheat": "cereals",
    "Corn": "cereals",
    "Winter barley": "cereals",
    "Spring barley": "cereals",
    "Winter rapeseed": "oilseed_industrial",
    "Winter triticale": "cereals",
    "Spring wheat": "cereals",
    "Grapevine": "vineyard",
    "Sunflower": "oilseed_industrial",
    "Sugar beet": "sugar_beet",
    "Beet": "sugar_beet",
    "Potatoes": "potato",
    "Leguminous fodder": "legumes_fodder",
    "Soybeans": "soybean",
    "Orchard": "orchard",
    "Mixed cereal": "cereals",
    "Sorghum": "cereals",
}

#: Stable per-macro colour (shared by FR and ES panels so the gap is readable).
MACRO_COLORS: dict[str, str] = {
    "grassland": "#4daf4a",
    "cereals": "#e08214",
    "oilseed_industrial": "#984ea3",
    "vineyard": "#7b3294",
    "sugar_beet": "#a6611a",
    "vegetables": "#66c2a5",
    "potato": "#d6604d",
    "legumes_fodder": "#2c7fb8",
    "soybean": "#1b9e77",
    "orchard": "#bf812d",
}

ALPHAEARTH_DIM_COLS: list[str] = [f"dim_{i:02d}" for i in range(64)]


def _read_es_patch_centroids(nc_path: Path, *, max_per_class: int = 60) -> list[dict[str, Any]]:
    """Sample per-macro-class pixel centroids of one ES Sen4AgriNet patch.

    Reads the HDF5 (NetCDF4) ``labels`` group with :mod:`h5py`, decodes the
    GDAL-style ``transform`` / ``x`` / ``y`` georeferencing (EPSG:32631), maps the
    raw FAO-ICC codes to the shared macro groups via :data:`FAO_ICC_TO_MACRO`, and
    returns up to ``max_per_class`` deterministically-spaced pixel centroids per
    macro-class as ``{px_id, easting, northing, macro}`` records (UTM 31N).

    Args:
        nc_path: Path to a Catalonia ``.nc`` patch (tile 31TCG).
        max_per_class: Cap of sampled pixels per macro-class per patch (keeps the
            GEE payload bounded; the centroids are equispaced, not random).

    Returns:
        List of centroid records in UTM 31N (reprojection happens once for all
        patches in :func:`build_es_alphaearth`).
    """
    import h5py  # local import: optional heavy dependency

    records: list[dict[str, Any]] = []
    with h5py.File(str(nc_path), "r") as f:
        labels = np.asarray(f["labels"]["labels"][:], dtype=np.int64)
        x = np.asarray(f["labels"]["x"][:], dtype=np.float64)  # easting per column
        y = np.asarray(f["labels"]["y"][:], dtype=np.float64)  # northing per row
    patch = nc_path.stem
    for code, macro in FAO_ICC_TO_MACRO.items():
        rows, cols = np.where(labels == code)
        if rows.size == 0:
            continue
        # Deterministic equispaced subsample (no RNG): keeps spatial spread.
        if rows.size > max_per_class:
            sel = np.linspace(0, rows.size - 1, max_per_class).astype(np.int64)
            rows, cols = rows[sel], cols[sel]
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            records.append(
                {
                    "px_id": f"{patch}_{r}_{c}",
                    "easting": float(x[c]),
                    "northing": float(y[r]),
                    "macro": macro,
                }
            )
    return records


def collect_centroids(*, patch_glob: str, region: str, max_per_class: int = 60) -> pl.DataFrame:
    """Collect per-class centroids across the Sen4AgriNet patches of one region.

    Reads every patch matched by ``patch_glob``, samples per-class centroids, and
    reprojects the UTM 31N eastings/northings to lon/lat with :mod:`pyproj` (one
    batched transform). Works identically for ES (31TCG) and FR (31TCJ) because
    both subsets share the EPSG:32631 georeferencing and FAO-ICC labels.

    Args:
        patch_glob: Glob (repo-relative) for the region ``.nc`` patches.
        region: Region tag (``"ES"`` / ``"FR"``) attached to every record.
        max_per_class: Per-class centroid cap per patch.

    Returns:
        Frame ``(px_id, lon, lat, macro, region)`` in EPSG:4326. Empty if no match.

    Raises:
        FileNotFoundError: if the glob resolves to zero patches (never fabricated).
    """
    from pyproj import Transformer

    patches = sorted((_REPO_ROOT / p) for p in glob.glob(patch_glob, root_dir=_REPO_ROOT))
    if not patches:
        raise FileNotFoundError(
            f"No Sen4AgriNet patches matched {patch_glob!r}; run `dvc pull "
            "data/sen4agrinet.dvc` first. This routine never fabricates coordinates."
        )
    records: list[dict[str, Any]] = []
    for nc_path in patches:
        records.extend(_read_es_patch_centroids(nc_path, max_per_class=max_per_class))
    frame = pl.DataFrame(records)
    transformer = Transformer.from_crs(_ES_CRS, _LONLAT_CRS, always_xy=True)
    lon, lat = transformer.transform(
        frame.get_column("easting").to_numpy(),
        frame.get_column("northing").to_numpy(),
    )
    out = frame.with_columns(
        pl.Series("lon", lon, dtype=pl.Float64),
        pl.Series("lat", lat, dtype=pl.Float64),
        pl.lit(region).alias("region"),
    ).select("px_id", "lon", "lat", "macro", "region")
    logger.info(
        "centroids_collected",
        region=region,
        n_patches=len(patches),
        n_points=out.height,
        macros=sorted(out.get_column("macro").unique().to_list()),
    )
    return out


def build_region_alphaearth(
    *,
    region: str,
    patch_glob: str,
    out_parquet: Path,
    max_per_class: int = 60,
    year: int = ALPHAEARTH_YEAR,
) -> pl.DataFrame:
    """Sample AlphaEarth at a region's Sen4AgriNet centroids and persist to parquet.

    Reuses :func:`ml.ingest.gee_sampler.sample_alphaearth_at_coords` (DRY) so the
    request batching, mosaicking and caching match the rest of the pipeline. The
    macro label + region tag are re-attached after sampling (the sampler returns
    only geometry + dims). Idempotent: if ``out_parquet`` exists it is returned.

    Args:
        region: Region tag (``"ES"`` / ``"FR"``).
        patch_glob: Glob for that region's ``.nc`` patches.
        out_parquet: Destination for the materialized embedding + label frame.
        max_per_class: Per-class centroid cap per patch.
        year: AlphaEarth annual embedding year (2019, the patch year).

    Returns:
        Frame ``(px_id, lon, lat, year, dim_00..dim_63, macro, region)``.

    Raises:
        RuntimeError: if GEE returns no embedding (auth/quota); never fabricated.
    """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists():
        logger.info("region_alphaearth_cache_hit", region=region, path=str(out_parquet))
        return pl.read_parquet(out_parquet)

    from ml.ingest.gee_sampler import init_ee, sample_alphaearth_at_coords

    init_ee(project="agrosat-copilot")
    centroids = collect_centroids(patch_glob=patch_glob, region=region, max_per_class=max_per_class)
    sampled = sample_alphaearth_at_coords(
        centroids.select("px_id", "lon", "lat"),
        year=year,
        cache_key=f"sen4agrinet_{region.lower()}",
    )
    valid = sampled.drop_nulls(subset=ALPHAEARTH_DIM_COLS)
    if valid.is_empty():
        raise RuntimeError(
            f"AlphaEarth returned no embedding for the {region} centroids (check GEE "
            "auth / project / quota). No synthetic embedding is ever written."
        )
    out = valid.join(centroids.select("px_id", "macro", "region"), on="px_id", how="inner")
    out.write_parquet(out_parquet)
    logger.info(
        "region_alphaearth_written",
        region=region,
        path=str(out_parquet),
        n_points=out.height,
        macros=sorted(out.get_column("macro").unique().to_list()),
    )
    return out


def compute_joint_umap(
    fr: pl.DataFrame,
    es: pl.DataFrame,
    *,
    seed: int = 17,
    n_neighbors: int = 25,
    min_dist: float = 0.15,
) -> pl.DataFrame:
    """Fit ONE UMAP on the stacked FR+ES AlphaEarth vectors for the gap panel.

    Standardizes the 64-dim vectors, fits a single :class:`umap.UMAP` so both
    regions live in the same 2-D space (so a visible FR/ES separation per macro IS
    the domain gap), and returns the 2-D coordinates tagged by region and macro.

    Args:
        fr: France frame ``(px_id, dim_*, macro)``.
        es: Catalonia frame ``(px_id, dim_*, macro)``.
        seed: UMAP ``random_state`` for reproducibility.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum distance.

    Returns:
        Frame ``(region, macro, x, y)`` with ``region in {FR, ES}``.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    fr_x = fr.select(ALPHAEARTH_DIM_COLS).to_numpy()
    es_x = es.select(ALPHAEARTH_DIM_COLS).to_numpy()
    stacked = np.vstack([fr_x, es_x])
    regions = np.array(["FR"] * fr.height + ["ES"] * es.height)
    macros = np.concatenate([fr.get_column("macro").to_numpy(), es.get_column("macro").to_numpy()])
    scaled = StandardScaler().fit_transform(stacked)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="euclidean",
    )
    emb = reducer.fit_transform(scaled)
    return pl.DataFrame(
        {
            "region": regions,
            "macro": macros,
            "x": emb[:, 0].astype(np.float64),
            "y": emb[:, 1].astype(np.float64),
        }
    )


def macro_aoi_bbox(
    region_points: pl.DataFrame, macro: str, *, half_deg: float = 0.01
) -> Sequence[float] | None:
    """Return a small lon/lat bbox centred on a macro-class for the NDVI series.

    Picks the median lon/lat of the macro-class points (a representative location
    inside the region) and grows a +/- ``half_deg`` square AOI around it.

    Args:
        region_points: Frame ``(lon, lat, macro)`` of one region.
        macro: Macro-class to centre on.
        half_deg: Half side of the AOI square in degrees (0.01 ~ 1 km).

    Returns:
        ``[min_lon, min_lat, max_lon, max_lat]`` or ``None`` if absent.
    """
    sub = region_points.filter(pl.col("macro") == macro)
    if sub.is_empty():
        return None
    clon = float(sub.get_column("lon").median())  # type: ignore[arg-type]
    clat = float(sub.get_column("lat").median())  # type: ignore[arg-type]
    return [clon - half_deg, clat - half_deg, clon + half_deg, clat + half_deg]


def _ndvi_empty_frame() -> pl.DataFrame:
    """Empty zonal-NDVI series frame with the canonical schema."""
    return pl.DataFrame(schema={"date": pl.Utf8, "doy": pl.Int64, "ndvi": pl.Float64})


def extract_bbox_ndvi_series(
    bbox: Sequence[float],
    year: int,
    *,
    cloud_pct_max: int = 30,
    scale: int = 20,
    cache_key: str = "bbox",
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Extract the REAL Sentinel-2 zonal-mean NDVI time series over a lon/lat bbox.

    Mirrors :func:`ml.transfer.mexico_demo.extract_s2_ndvi_series` (QA60 cloud mask,
    ``normalizedDifference(['B8','B4'])``, ``ee.Reducer.mean()``) but accepts an
    arbitrary rectangle so it serves both PASTIS-FR (31TCJ) and Catalonia (31TCG)
    AOIs. ``cloud_pct_max=30`` and ``scale=20`` suit the European tiles. Cached and
    degraded-safe: an empty frame is returned (never fabricated) on any failure.

    Args:
        bbox: ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
        year: Year of the time series.
        cloud_pct_max: Maximum ``CLOUDY_PIXEL_PERCENTAGE`` to keep an image.
        scale: Reduction resolution in meters.
        cache_key: Logical cache identifier (region + macro).
        cache_dir: Parquet cache folder (default ``data/cache/gee/``).

    Returns:
        ``pl.DataFrame`` with columns ``date`` (``YYYY-MM-DD``), ``doy`` (1..366)
        and ``ndvi`` (in ``[-1, 1]``), sorted by date.
    """
    from ml.transfer.mexico_demo import DEFAULT_CACHE_DIR, S2_COLLECTION
    from ml.transfer.mexico_demo import _mask_s2_clouds_and_add_ndvi as _mask

    try:
        import ee  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover
        return _ndvi_empty_frame()

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"sen4_ndvi_{cache_key}_{year}_{cloud_pct_max}_{scale}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    try:
        geometry = ee.Geometry.Rectangle(list(bbox))
        collection = (
            ee.ImageCollection(S2_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .map(_mask)
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
        logger.warning("bbox_ndvi_failed", cache_key=cache_key, year=year, error=str(exc))
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
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d").dt.ordinal_day().alias("doy"))
        .select("date", "doy", "ndvi")
        .sort("date")
    )
    frame.write_parquet(cache_file)
    return frame
