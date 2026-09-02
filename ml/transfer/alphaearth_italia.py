"""Per-parcel AlphaEarth embedding extraction for the Italian homologue (US-079).

Re-materializes the AlphaEarth Satellite Embedding V1 Annual v1.1 (64-dim,
columns ``dim_00..dim_63``) for the EuroCrops Italy 2018 parcels that fall inside
the US-078 dense patches, so the ``xgb-alphaearth`` member of the champion
Voting-3 (``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth``, F1 0.9069 in France)
can be replicated on Italy with CLEAN embeddings.

Why re-download instead of reusing the cache
---------------------------------------------
The existing Italian AlphaEarth cache
(``data/cache/gee/alphaearth_at_italia_stab_*``) was sampled on 500 ``stab``
points unrelated to the homologue parcels: it does NOT cover the parcels of the
US-078 dataset, so it cannot feed the parcel-level vote. This module samples
AlphaEarth fresh over the REAL parcel polygons via ``reduceRegions(mean)``.

AlphaEarth is GEE-free (no Sentinel Hub quota), so this runs independently of the
patch download that consumes the Sentinel Hub quota for US-078.

Pipeline
--------
1. :func:`load_italia_label_space` -- read the materialized
   ``class_mapping.json`` of US-078 (the CANONICAL Italian label space, id 0 =
   background); never rebuilt here so the class axis stays aligned with the dense
   masks.
2. :func:`load_patch_bboxes` -- read ``metadata.parquet`` to get each patch's
   EPSG:4326 bbox + its ``fold_espacial`` (spatial fold for leak-free CV).
3. :func:`parcels_in_patches` -- load the EuroCrops Italy 2018 polygons, label
   each with its canonical ``class_id`` (``original_code`` -> HCAT4 ->
   ``class_id``), keep only parcels whose centroid falls in a patch bbox, and
   inherit the patch's spatial fold.
4. :func:`extract_alphaearth_parcels` -- sample the 64-dim AlphaEarth embedding
   per parcel polygon (``reduceRegions(mean)``, batched) for the same year as the
   patches (2018). Real GEE data only -- never fabricated.
5. :func:`build_alphaearth_italia_features` -- orchestrate 1-4 + materialize the
   feature parquet (``canonical_parcel_id`` + ``class_id`` + ``fold`` +
   ``dim_00..dim_63``).

Project conventions: ``polars`` (never pandas in pipelines; geopandas only at the
geometry boundary), ``structlog`` for logging, type hints + Google-style
docstrings, visible prose Spanish / code identifiers English, no emojis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import structlog

from ml.ingest.gee_sampler import (
    ALPHAEARTH_DIM_COLS,
    init_ee,
    sample_alphaearth_for_parcels,
)
from ml.utils.parcel_id import canonical_parcel_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

logger = structlog.get_logger(__name__)

__all__ = [
    "ALPHAEARTH_DIM_COLS",
    "DEFAULT_FEATURES_PATH",
    "ITALIA_BACKGROUND_ID",
    "ITALIA_YEAR",
    "build_alphaearth_italia_features",
    "extract_alphaearth_parcels",
    "load_italia_label_space",
    "load_patch_bboxes",
    "parcels_in_patches",
]

#: Repo root (this file is ``<root>/ml/transfer/alphaearth_italia.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: EuroCrops Italy 2018 reference inputs (DVC-tracked).
ITALY_PARCELS_PARQUET: Path = (
    _REPO_ROOT / "data" / "reference" / "eurocrops_v2" / "iti1_2018.parquet"
)
EUROCROPS_MAPPING_CSV: Path = (
    _REPO_ROOT / "data" / "reference" / "eurocrops_v2" / "eurocrops_mapping.csv"
)

#: US-078 dense dataset root (default; the runner can override for the full set).
DEFAULT_DATASET_DIR: Path = _REPO_ROOT / "data" / "pastis_italia_2018"

#: Materialized per-parcel feature parquet (DVC-tracked at close).
DEFAULT_FEATURES_PATH: Path = _REPO_ROOT / "data" / "features" / "alphaearth_italia_2018.parquet"

#: Year of the homologue patches; the AlphaEarth annual image must match it.
ITALIA_YEAR: int = 2018

#: Background / no-crop id in the Italian label space (mirrors PASTIS TARGET fill).
ITALIA_BACKGROUND_ID: int = 0

#: EuroCrops native CRS (ETRS89-LAEA, metric) and the geographic CRS for GEE.
PROJECTED_CRS: str = "EPSG:3035"
GEOGRAPHIC_CRS: str = "EPSG:4326"


def load_italia_label_space(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, int]:
    """Load the canonical Italian label space from the US-078 mapping.

    Reads ``class_mapping.json`` (materialized by US-078) and returns the
    ``hcat4_name -> class_id`` map. This is the SAME label space the dense masks
    (``TARGET_<id>.npy``) use, so the ``xgb-alphaearth`` member shares the class
    axis with TSViT-pheno / U-TAE in the Voting. The label space is NOT rebuilt
    here (no ``min_support`` recomputation): re-deriving it could shift ids and
    break the alignment with the dense members.

    Args:
        dataset_dir: US-078 dataset directory holding ``class_mapping.json``.

    Returns:
        Mapping ``hcat4_name -> class_id`` (id 0 reserved for background; never
        returned as a crop key).

    Raises:
        FileNotFoundError: if ``class_mapping.json`` is absent.
    """
    mapping_path = dataset_dir / "class_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"class_mapping.json not found at {mapping_path}; it is the canonical "
            "Italian label space of US-078. Run the US-078 builder or `dvc pull` "
            "the dataset first."
        )
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    name_to_id: dict[str, int] = {
        str(entry["hcat4_name"]): int(entry["class_id"]) for entry in mapping.get("classes", [])
    }
    logger.info(
        "italia_label_space_loaded",
        n_classes=len(name_to_id),
        background_id=int(mapping.get("background_id", ITALIA_BACKGROUND_ID)),
    )
    return name_to_id


def load_patch_bboxes(dataset_dir: Path = DEFAULT_DATASET_DIR) -> pl.DataFrame:
    """Read the patch bboxes + spatial folds from the US-078 metadata.

    Args:
        dataset_dir: US-078 dataset directory holding ``metadata.parquet``.

    Returns:
        A Polars frame with columns ``patch_id, bbox_min_lon, bbox_min_lat,
        bbox_max_lon, bbox_max_lat, fold_espacial`` (EPSG:4326 bboxes).

    Raises:
        FileNotFoundError: if ``metadata.parquet`` is absent.
    """
    metadata_path = dataset_dir / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata.parquet not found at {metadata_path}; it carries the patch "
            "bboxes the parcels are intersected with. Point --patches-metadata to "
            "the full-dataset metadata when the download finishes."
        )
    cols = [
        "patch_id",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
        "fold_espacial",
    ]
    bboxes = pl.read_parquet(metadata_path).select(cols)
    logger.info(
        "patch_bboxes_loaded",
        n_patches=bboxes.height,
        n_folds=bboxes["fold_espacial"].n_unique(),
    )
    return bboxes


def _label_parcels(
    parcels_parquet: Path,
    mapping_csv: Path,
    name_to_id: dict[str, int],
    *,
    region_prefix: str = "it",
) -> gpd.GeoDataFrame:
    """Load and label the EuroCrops Italy 2018 polygons with canonical ids.

    Reuses the US-078 Italy crosswalk (``original_code`` -> HCAT4 name) and maps
    the HCAT names onto the canonical ``class_id`` of :func:`load_italia_label_space`.
    Codes absent from the canonical map fall into background (id 0) and are dropped
    downstream (they are not part of the homologue crop space).

    Args:
        parcels_parquet: Path to ``iti1_2018.parquet`` (EPSG:3035 polygons).
        mapping_csv: Path to ``eurocrops_mapping.csv``.
        name_to_id: Canonical ``hcat4_name -> class_id`` map.

    Returns:
        A GeoDataFrame (EPSG:4326) with ``original_code, class_id, geometry`` and
        a projected centroid ``cx, cy`` (EPSG:3035) for fast bbox binning.

    Raises:
        FileNotFoundError: if the parcels parquet or mapping CSV is absent.
    """
    import geopandas as gpd

    from ml.data.eurocrops_pastis_builder import _load_region_code_to_hcat

    if not parcels_parquet.is_file():
        raise FileNotFoundError(
            f"parcels parquet not found at {parcels_parquet}; pull it with "
            "`dvc pull data/reference/eurocrops_v2/<region>.parquet`."
        )
    gdf = gpd.read_parquet(parcels_parquet)
    if gdf.crs is None or "3035" not in str(gdf.crs.to_epsg() or gdf.crs):
        gdf = gdf.to_crs(PROJECTED_CRS)

    # Drop empty/null geometries (EuroCrops Italy ships some): their centroid is
    # NaN and they carry no rasterisable area.
    valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_dropped = int((~valid).sum())
    gdf = gdf[valid].reset_index(drop=True)
    if n_dropped:
        logger.info("italia_polygons_empty_dropped", n=n_dropped)

    crosswalk = _load_region_code_to_hcat(mapping_csv, region_prefix=region_prefix).to_pandas()
    gdf["original_code"] = gdf["original_code"].astype(str)
    gdf = gdf.merge(crosswalk, on="original_code", how="left")

    # Map HCAT name -> canonical class_id; unknown / unmapped -> background (0).
    gdf["class_id"] = gdf["hcat4_name"].map(name_to_id).fillna(ITALIA_BACKGROUND_ID).astype("int64")

    # Projected centroid (metric, no geographic warning) for the bbox binning.
    centroids = gdf.geometry.centroid
    gdf["cx_3035"] = centroids.x.to_numpy()
    gdf["cy_3035"] = centroids.y.to_numpy()

    # Reproject to EPSG:4326 so the centroid lon/lat aligns with the patch bboxes
    # and the geometry is GEE-ready (reduceRegions consumes 4326 polygons).
    gdf = gdf.to_crs(GEOGRAPHIC_CRS)
    geo_centroids = gpd.GeoSeries(centroids, crs=PROJECTED_CRS).to_crs(GEOGRAPHIC_CRS)
    gdf["lon"] = geo_centroids.x.to_numpy()
    gdf["lat"] = geo_centroids.y.to_numpy()
    logger.info(
        "italia_polygons_labeled",
        n_parcels=len(gdf),
        n_crops=int((gdf["class_id"] != ITALIA_BACKGROUND_ID).sum()),
    )
    return gdf


def parcels_in_patches(
    bboxes: pl.DataFrame,
    name_to_id: dict[str, int],
    *,
    parcels_parquet: Path = ITALY_PARCELS_PARQUET,
    mapping_csv: Path = EUROCROPS_MAPPING_CSV,
    region_prefix: str = "it",
) -> gpd.GeoDataFrame:
    """Keep the labelled parcels whose centroid falls in a patch bbox.

    A parcel is assigned to the FIRST patch whose EPSG:4326 bbox contains its
    centroid; it inherits that patch's ``fold_espacial`` so the downstream spatial
    CV uses the SAME fold map as the dense members (no cross-member leakage).
    Background parcels (id 0, no crop label) are dropped: they are not part of the
    homologue crop space the Voting reasons over.

    Args:
        bboxes: Patch bbox frame from :func:`load_patch_bboxes`.
        name_to_id: Canonical ``hcat4_name -> class_id`` map.
        parcels_parquet: Path to ``iti1_2018.parquet``.
        mapping_csv: Path to ``eurocrops_mapping.csv``.

    Returns:
        A GeoDataFrame (EPSG:4326) of the in-patch crop parcels with columns
        ``parcel_id`` (sequential int surrogate), ``canonical_parcel_id`` (Utf8),
        ``class_id``, ``fold``, ``patch_id``, ``geometry`` (POLYGON).

    Raises:
        ValueError: if no parcel falls inside any patch bbox.
    """
    import numpy as np

    gdf = _label_parcels(parcels_parquet, mapping_csv, name_to_id, region_prefix=region_prefix)
    # Drop background parcels (no crop label) before the spatial join.
    gdf = gdf[gdf["class_id"] != ITALIA_BACKGROUND_ID].reset_index(drop=True)

    lon = gdf["lon"].to_numpy()
    lat = gdf["lat"].to_numpy()
    patch_id = np.full(len(gdf), -1, dtype=np.int64)
    fold = np.full(len(gdf), -1, dtype=np.int64)

    # Vectorized point-in-bbox assignment, first-match wins. Patch count is small
    # (20 pilot, ~1226 full), so an O(n_patches) loop over numpy masks is cheap.
    for row in bboxes.iter_rows(named=True):
        unassigned = patch_id == -1
        inside = (
            unassigned
            & (lon >= row["bbox_min_lon"])
            & (lon <= row["bbox_max_lon"])
            & (lat >= row["bbox_min_lat"])
            & (lat <= row["bbox_max_lat"])
        )
        patch_id[inside] = int(row["patch_id"])
        fold[inside] = int(row["fold_espacial"])

    gdf["patch_id"] = patch_id
    gdf["fold"] = fold
    in_patch = gdf[gdf["patch_id"] != -1].reset_index(drop=True)
    if len(in_patch) == 0:
        raise ValueError(
            "no parcel centroid fell inside any patch bbox; check that the metadata "
            "and the iti1_2018 parcels share the EPSG:4326 frame."
        )

    # Sequential integer surrogate + canonical Utf8 id (the Voting key).
    in_patch = in_patch.reset_index(drop=True)
    in_patch["parcel_id"] = np.arange(len(in_patch), dtype=np.int64)
    in_patch["canonical_parcel_id"] = [
        f"iti1_2018_p{int(pid)}_{int(seq)}"
        for pid, seq in zip(
            in_patch["patch_id"].to_numpy(),
            in_patch["parcel_id"].to_numpy(),
            strict=True,
        )
    ]
    logger.info(
        "parcels_in_patches_selected",
        n_parcels=len(in_patch),
        n_patches_hit=int(in_patch["patch_id"].nunique()),
        n_classes=int(in_patch["class_id"].nunique()),
        n_folds=int(in_patch["fold"].nunique()),
    )
    return in_patch


def extract_alphaearth_parcels(
    parcels: gpd.GeoDataFrame,
    *,
    year: int = ITALIA_YEAR,
    batch_size: int = 100,
    cache_dir: Path | None = None,
    cache_key: str = "italia_2018",
) -> pl.DataFrame:
    """Sample the 64-dim AlphaEarth embedding per parcel polygon (real GEE).

    Thin orchestrator over :func:`ml.ingest.gee_sampler.sample_alphaearth_for_parcels`
    (``reduceRegions(mean)`` over each polygon, batched to avoid GEE compute-graph
    timeouts). The sampler returns per-parcel embeddings keyed by ``parcel_id``;
    here ``parcel_id`` is the sequential surrogate built in
    :func:`parcels_in_patches`, which the sampler preserves as a string.

    Args:
        parcels: GeoDataFrame with ``parcel_id`` (int surrogate) + ``geometry``
            POLYGON EPSG:4326 (the output of :func:`parcels_in_patches`).
        year: Annual AlphaEarth image year (default 2018, matches the patches).
        batch_size: Polygons per GEE request (default 100; polygons are heavier
            than points so a smaller batch avoids timeouts).
        cache_dir: Local parquet cache folder for the GEE sampler.
        cache_key: Cache key for the sampler.

    Returns:
        A Polars frame with ``canonical_parcel_id`` (Utf8) + ``dim_00..dim_63``;
        empty (schema only) if GEE is unavailable / fails (the caller reports it).
    """
    sampled = sample_alphaearth_for_parcels(
        parcels[["parcel_id", "geometry"]],
        year,
        cache_dir=cache_dir,
        cache_key=cache_key,
        batch_size=batch_size,
    )
    if sampled.is_empty():
        logger.warning(
            "alphaearth_italia_empty",
            note="the GEE sampler returned no embeddings (auth/quota/network); "
            "no parcel was sampled.",
        )
        return sampled

    # The sampler keys by the surrogate parcel_id (as Utf8); re-attach the
    # canonical id so the join with labels/folds is unambiguous.
    surrogate_to_canonical = {
        str(int(pid)): cid
        for pid, cid in zip(
            parcels["parcel_id"].to_numpy(),
            parcels["canonical_parcel_id"].tolist(),
            strict=True,
        )
    }
    keep_cols = ["canonical_parcel_id", *ALPHAEARTH_DIM_COLS]
    out = (
        sampled.with_columns(
            pl.col("parcel_id")
            .replace_strict(surrogate_to_canonical, default=None)
            .alias("canonical_parcel_id")
        )
        .filter(pl.col("canonical_parcel_id").is_not_null())
        .select(keep_cols)
    )
    logger.info(
        "alphaearth_italia_extracted",
        n_parcels=out.height,
        n_dims=len(ALPHAEARTH_DIM_COLS),
        year=int(year),
    )
    return out


def build_alphaearth_italia_features(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    patches_metadata: Path | None = None,
    parcels_parquet: Path = ITALY_PARCELS_PARQUET,
    mapping_csv: Path = EUROCROPS_MAPPING_CSV,
    region_prefix: str = "it",
    year: int = ITALIA_YEAR,
    batch_size: int = 100,
    project: str | None = None,
    service_account_json: Path | None = None,
    cache_dir: Path | None = None,
    out_path: Path = DEFAULT_FEATURES_PATH,
) -> pl.DataFrame:
    """Build + materialize the per-parcel AlphaEarth feature table for Italy.

    Full pipeline: load the canonical label space, intersect the EuroCrops parcels
    with the patch bboxes, extract the AlphaEarth embedding per parcel (real GEE),
    join the embedding to ``class_id`` + ``fold`` + ``canonical_parcel_id`` and
    write the feature parquet (DVC-tracked at close).

    Args:
        dataset_dir: US-078 dataset directory (``class_mapping.json`` lives here).
        patches_metadata: Override path to ``metadata.parquet`` (e.g. the full
            dataset on the VM). Defaults to ``dataset_dir/metadata.parquet``.
        parcels_parquet: Path to ``iti1_2018.parquet``.
        mapping_csv: Path to ``eurocrops_mapping.csv``.
        year: AlphaEarth annual image year (default 2018).
        batch_size: Polygons per GEE request.
        project: GCP project for the EE quota (ADC). Defaults to the active one.
        service_account_json: Optional service-account key (else ADC).
        cache_dir: Cache folder for the GEE sampler.
        out_path: Destination parquet for the materialized features.

    Returns:
        The materialized feature frame (``parcel_id, canonical_parcel_id,
        class_id, fold, patch_id, year, dim_00..dim_63``). Empty (no rows) if GEE
        returned nothing.

    Raises:
        FileNotFoundError: if a required input is missing.
        ValueError: if no parcel falls in any patch bbox.
    """
    name_to_id = load_italia_label_space(dataset_dir)
    metadata_dir = patches_metadata.parent if patches_metadata is not None else dataset_dir
    bboxes = load_patch_bboxes(metadata_dir)
    parcels = parcels_in_patches(
        bboxes,
        name_to_id,
        parcels_parquet=parcels_parquet,
        mapping_csv=mapping_csv,
        region_prefix=region_prefix,
    )

    # Authenticate GEE via ADC (or service account) BEFORE sampling so an auth
    # failure surfaces clearly instead of an empty frame mid-pipeline.
    init_ee(service_account_json=service_account_json, project=project)

    embeddings = extract_alphaearth_parcels(
        parcels,
        year=year,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    labels = canonical_parcel_id(
        pl.from_pandas(
            parcels[["parcel_id", "canonical_parcel_id", "class_id", "fold", "patch_id"]]
        ),
        col="canonical_parcel_id",
    )
    if embeddings.is_empty():
        logger.warning(
            "alphaearth_italia_features_empty",
            note="no embedding extracted; the feature parquet is not written.",
        )
        return embeddings

    features = (
        labels.join(embeddings, on="canonical_parcel_id", how="inner")
        .with_columns(pl.lit(int(year)).alias("year"))
        .select(
            [
                "parcel_id",
                "canonical_parcel_id",
                "class_id",
                "fold",
                "patch_id",
                "year",
                *ALPHAEARTH_DIM_COLS,
            ]
        )
    )

    # Drop parcels whose polygon intersected NO AlphaEarth pixel (every dim is
    # null): sub-pixel / edge geometries that reduceRegions(mean) could not sample.
    # They have no REAL embedding, so they are dropped (never imputed with a full
    # synthetic vector -- that would fabricate the very data US-079 re-downloads to
    # keep clean). The parcel_id surrogate is re-densified so the spatial-CV
    # positional indexing stays gap-free.
    all_null = pl.all_horizontal(pl.col(c).is_null() for c in ALPHAEARTH_DIM_COLS)
    n_total = features.height
    features = features.filter(~all_null)
    n_dropped = n_total - features.height
    if n_dropped:
        features = (
            features.drop("parcel_id")
            .with_row_index(name="parcel_id")
            .with_columns(pl.col("parcel_id").cast(pl.Int64))
            .select(
                [
                    "parcel_id",
                    "canonical_parcel_id",
                    "class_id",
                    "fold",
                    "patch_id",
                    "year",
                    *ALPHAEARTH_DIM_COLS,
                ]
            )
        )
        logger.info(
            "alphaearth_italia_unsampled_dropped",
            n_dropped=n_dropped,
            n_kept=features.height,
            note="parcels with no AlphaEarth pixel intersection (sub-pixel/edge).",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out_path)
    logger.info(
        "alphaearth_italia_features_written",
        path=str(out_path),
        n_parcels=features.height,
        n_classes=int(features["class_id"].n_unique()),
        n_folds=int(features["fold"].n_unique()),
    )
    return features
