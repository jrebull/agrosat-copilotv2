"""Build a PASTIS-homologous dense dataset from EuroCrops Italy 2018 (US-078).

The dense champion members (TSViT-pheno, U-TAE) were trained on PASTIS-R: per
patch a temporal stack ``S2_<id>.npy (T, 10, 128, 128)`` of Sentinel-2 L2A
reflectance plus a dense semantic mask ``TARGET_<id>.npy (128, 128) int32`` where
every pixel carries its parcel's crop class. To measure a genuine France ->
Italy transfer (US-079) the Italian polygons must be turned into the SAME
structure, so the dense models run in their native format with no input change.

This module orchestrates steps 1-5 of the US-078 plan as pure, testable
functions:

1. :func:`load_labeled_polygons` -- read the EuroCrops Italy 2018 polygons,
   map ``original_code`` -> HCAT4 name -> a contiguous class id ``[0, K)``
   (id 0 reserved for background), grouping rare classes into an explicit
   ``other`` bucket so the label space is not inflated.
2. :func:`select_dense_patches` -- grid the parcel centroids (in the projected
   CRS, EPSG:3035) and keep the densest cells; each dense cell defines one
   ``1.28 km`` patch bbox. A spatial fold is assigned per grid super-cell so the
   downstream CV (US-079) has no spatial leakage.
3. :func:`download_patch_series` -- pull the patch's temporal Sentinel-2 stack
   via :mod:`ml.ingest.sh_path` (one ORBIT tile per patch, per-pixel SCL
   cloud-masked), reusing the Sentinel Hub client + on-disk cache. Network /
   quota cost is real and reported; nothing is fabricated.
4. :func:`rasterize_patch_mask` -- rasterize the parcels falling in the patch
   onto the EXACT tile grid (the child transform derived from the tile's own
   ``rasterio`` transform and the crop window offset), so image and mask are
   pixel-perfect aligned.
5. :func:`save_pastis_format` -- persist ``S2_<id>.npy`` + ``TARGET_<id>.npy`` +
   ``dates_<id>.npy`` (acquisition DOY, like PASTIS ``dates-S2``) mirroring the
   ``data/PASTIS-R`` layout.

Honesty
-------
- Reflectance is scaled to the PASTIS DN convention (``x 10000``, int16) so the
  spectral magnitude matches; NDVI (scale-invariant) is used to compare texture.
- A patch whose tile fails, is all-cloud, or carries < 2 dates is dropped and
  counted, never fabricated.
- The on-disk Sentinel Hub cache guarantees no (patch) tile is re-downloaded, so
  the quota is spent once and the consumed request count is reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.ingest.sh_client import PASTIS_BANDS

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd
    from affine import Affine
    from rasterio.crs import CRS

    from ml.ingest.sh_client import SentinelHubClient

logger = structlog.get_logger(__name__)

__all__ = [
    "PASTIS_BANDS",
    "PatchPlan",
    "PatchResult",
    "build_class_table",
    "download_patch_series",
    "load_labeled_polygons",
    "rasterize_patch_mask",
    "save_pastis_format",
    "select_dense_patches",
]

#: Repo root (this file is ``<root>/ml/data/eurocrops_pastis_builder.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: EuroCrops Italy 2018 reference inputs (already downloaded + DVC-tracked).
ITALY_PARCELS_PARQUET: Path = (
    _REPO_ROOT / "data" / "reference" / "eurocrops_v2" / "iti1_2018.parquet"
)
EUROCROPS_MAPPING_CSV: Path = (
    _REPO_ROOT / "data" / "reference" / "eurocrops_v2" / "eurocrops_mapping.csv"
)

#: Default output root for the homologue dataset (DVC-tracked, never to Git).
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "data" / "pastis_italia_2018"

#: EuroCrops native CRS (ETRS89-LAEA, metric -- centroids must be computed here,
#: never in geographic 4326 which warns and is imprecise).
PROJECTED_CRS: str = "EPSG:3035"
#: Geographic CRS the Sentinel Hub bbox / mask rasterisation work in.
GEOGRAPHIC_CRS: str = "EPSG:4326"

#: Sentinel-2 native resolution (m/px); PASTIS is 10 m, the homologue matches it.
S2_RES_M: float = 10.0
#: Patch side in pixels (= PASTIS); 128 px x 10 m = 1.28 km.
PATCH_PX: int = 128
#: Patch side in metres (the dense-cell bbox span in EPSG:3035).
PATCH_SIDE_M: float = PATCH_PX * S2_RES_M  # 1280.0

#: PASTIS stores reflectance as int16 DN ~ ``reflectance * 10000``. The Sentinel
#: Hub Process API returns FLOAT32 reflectance in [0, 1]-ish, so we scale by this
#: to mirror the PASTIS magnitude (the dense models read that scale).
PASTIS_DN_SCALE: float = 10000.0

#: Number of spatial folds assigned in the metadata (for leak-free CV in US-079).
N_SPATIAL_FOLDS: int = 5
#: Side, in patch-units, of the super-cell that groups neighbouring patches into
#: one fold (so adjacent patches never split across train/test).
FOLD_SUPERCELL_PATCHES: int = 4

#: Background / no-crop class id in the dense mask (mirrors PASTIS TARGET fill).
BACKGROUND_ID: int = 0
#: Readable name reserved for rare classes folded together (explicit, not silent).
OTHER_CLASS_NAME: str = "other"


@dataclass(frozen=True)
class PatchPlan:
    """A selected dense patch to download and rasterise.

    Attributes:
        patch_id: Stable integer id (PASTIS-style, used in file names).
        bbox_3035: Patch bbox ``(min_x, min_y, max_x, max_y)`` in EPSG:3035.
        bbox_4326: Same bbox in EPSG:4326 ``(min_lon, min_lat, max_lon, max_lat)``.
        n_parcels: Number of parcel centroids that fall in the cell.
        fold: Spatial fold id ``[0, N_SPATIAL_FOLDS)`` for leak-free CV.
        classes_present: Sorted class ids whose parcels touch the cell.
    """

    patch_id: int
    bbox_3035: tuple[float, float, float, float]
    bbox_4326: tuple[float, float, float, float]
    n_parcels: int
    fold: int
    classes_present: tuple[int, ...]


@dataclass
class PatchResult:
    """Outcome of building one patch (image + mask + dates).

    Attributes:
        patch_id: The patch id.
        n_dates: Number of temporal frames kept (``T``).
        dates_doy: Day-of-year of each kept frame (length ``T``).
        coverage: Fraction of pixels with a crop class (mask != background).
        n_classes_present: Distinct crop classes in the mask (background excluded).
        class_support: Per-class pixel counts ``{class_id: n_pixels}``.
        requests: Sentinel Hub Process API requests this patch issued (0 = cache).
        ndvi_std: Mean over frames of the per-frame spatial NDVI std (texture).
        residual_cloud: Fraction of frame-pixels that were SCL-masked to 0.
        ok: ``True`` when the patch was successfully written.
    """

    patch_id: int
    n_dates: int = 0
    dates_doy: list[int] = field(default_factory=list)
    coverage: float = 0.0
    n_classes_present: int = 0
    class_support: dict[int, int] = field(default_factory=dict)
    requests: int = 0
    ndvi_std: float = 0.0
    residual_cloud: float = 0.0
    ok: bool = False


# --------------------------------------------------------------------------- #
# Step 1 -- labelled polygons + class table
# --------------------------------------------------------------------------- #
def _load_region_code_to_hcat(
    mapping_csv: Path = EUROCROPS_MAPPING_CSV, *, region_prefix: str = "it"
) -> pl.DataFrame:
    """Load a region's ``original_code`` -> ``hcat4_name`` mapping from EuroCrops.

    Filters the EuroCrops crosswalk to rows whose ``nuts`` begins with
    ``region_prefix`` (``"it"`` for Italy regions like ``iti1``; ``"de4"`` for
    Lower Saxony, etc.), keeping the columns needed to label parcels. Codes stay
    ``Utf8`` to preserve any leading characters.

    Args:
        mapping_csv: Path to the EuroCrops crosswalk CSV (``eurocrops.csv``).
        region_prefix: Lowercase NUTS prefix selecting the region(s) to load
            (``"it"``, ``"de4"``, ``"nl"``, ...).

    Returns:
        A Polars frame with unique ``original_code`` -> ``hcat4_name`` for the
        region.

    Raises:
        FileNotFoundError: if the mapping CSV is absent.
        ValueError: if no crosswalk row matches ``region_prefix``.
    """
    if not mapping_csv.is_file():
        raise FileNotFoundError(
            f"EuroCrops mapping CSV not found at {mapping_csv}; it is required to "
            f"label the {region_prefix!r} parcels (original_code -> HCAT4)."
        )
    # ``infer_schema_length=0`` reads every column as Utf8: ``original_code`` mixes
    # numeric and alphabetic codes (e.g. 'AAR'), so a numeric inference fails.
    mapping = pl.read_csv(mapping_csv, infer_schema_length=0)
    region = (
        mapping.filter(pl.col("nuts").str.to_lowercase().str.starts_with(region_prefix.lower()))
        .select(
            original_code=pl.col("original_code").cast(pl.Utf8),
            hcat4_name=pl.col("hcat4_name").cast(pl.Utf8),
        )
        .unique(subset="original_code", keep="first")
    )
    if region.height == 0:
        raise ValueError(
            f"no EuroCrops crosswalk rows for NUTS prefix {region_prefix!r} in "
            f"{mapping_csv}; check the prefix (e.g. 'it', 'de4', 'nl')."
        )
    logger.info("region_crosswalk_loaded", region=region_prefix, n_codes=region.height)
    return region


def _load_italy_code_to_hcat(mapping_csv: Path = EUROCROPS_MAPPING_CSV) -> pl.DataFrame:
    """Backwards-compatible Italy crosswalk loader (delegates to the generic one)."""
    return _load_region_code_to_hcat(mapping_csv, region_prefix="it")


def build_class_table(
    gdf: gpd.GeoDataFrame, *, min_support: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Map HCAT names to a contiguous class id ``[1, K]`` (0 = background).

    Crop classes with fewer than ``min_support`` parcels are folded into a single
    explicit :data:`OTHER_CLASS_NAME` bucket so the label space is not inflated by
    long-tail classes (documented threshold, never silently dropped).

    Args:
        gdf: The labelled parcels with an ``hcat4_name`` column.
        min_support: Minimum parcel count for a class to keep its own id.

    Returns:
        A tuple ``(class_table, name_to_id)`` where ``class_table`` is a Polars
        frame ``(class_id, hcat4_name, n_parcels)`` sorted by id and ``name_to_id``
        maps every HCAT name (incl. the rare ones) to its assigned id.
    """
    counts = (
        pl.DataFrame({"hcat4_name": gdf["hcat4_name"].to_numpy()})
        .group_by("hcat4_name")
        .len()
        .rename({"len": "n_parcels"})
        .sort(["n_parcels", "hcat4_name"], descending=[True, False])
    )
    kept = counts.filter(pl.col("n_parcels") >= min_support)
    rare = counts.filter(pl.col("n_parcels") < min_support)

    name_to_id: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    next_id = 1  # 0 is background
    for name, n in zip(kept["hcat4_name"], kept["n_parcels"], strict=True):
        name_to_id[str(name)] = next_id
        rows.append({"class_id": next_id, "hcat4_name": str(name), "n_parcels": int(n)})
        next_id += 1

    if rare.height > 0:
        other_id = next_id
        other_total = int(rare["n_parcels"].sum())
        for name in rare["hcat4_name"]:
            name_to_id[str(name)] = other_id
        rows.append(
            {"class_id": other_id, "hcat4_name": OTHER_CLASS_NAME, "n_parcels": other_total}
        )

    class_table = pl.DataFrame(rows).sort("class_id")
    logger.info(
        "class_table_built",
        n_classes=class_table.height,
        min_support=min_support,
        n_rare_folded=rare.height,
    )
    return class_table, name_to_id


def load_labeled_polygons(
    *,
    parcels_parquet: Path = ITALY_PARCELS_PARQUET,
    mapping_csv: Path = EUROCROPS_MAPPING_CSV,
    min_support: int,
    region_prefix: str = "it",
) -> tuple[gpd.GeoDataFrame, pl.DataFrame]:
    """Load EuroCrops Italy 2018 polygons labelled with contiguous class ids.

    Reads the polygons (EPSG:3035, MultiPolygon), joins each parcel's
    ``original_code`` to its HCAT4 name (Italy crosswalk, 100% coverage), folds
    rare classes into :data:`OTHER_CLASS_NAME`, and attaches the contiguous
    ``class_id`` (0 reserved for background). The projected centroid (in 3035,
    not geographic) is precomputed for the dense-cell grid.

    Args:
        parcels_parquet: Path to ``iti1_2018.parquet``.
        mapping_csv: Path to ``eurocrops_mapping.csv``.
        min_support: Minimum parcel count for a class to keep its own id.

    Returns:
        A tuple ``(gdf, class_table)``: the labelled GeoDataFrame (with
        ``class_id``, ``cx``, ``cy`` projected-centroid columns) and the Polars
        class table.

    Raises:
        FileNotFoundError: if the parcels parquet is absent.
    """
    import geopandas as gpd

    if not parcels_parquet.is_file():
        raise FileNotFoundError(
            f"Italy parcels parquet not found at {parcels_parquet}; pull it with "
            "`dvc pull data/reference/eurocrops_v2/iti1_2018.parquet`."
        )
    gdf = gpd.read_parquet(parcels_parquet)
    if gdf.crs is None or "3035" not in str(gdf.crs.to_epsg() or gdf.crs):
        gdf = gdf.to_crs(PROJECTED_CRS)
    # Drop empty/null geometries (EuroCrops Italy ships ~908): their centroid is
    # NaN, which would bin into a spurious dense cell and reproject to an inf
    # bbox. They carry no rasterisable area, so dropping them is lossless.
    n_before = len(gdf)
    valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
    gdf = gdf[valid].reset_index(drop=True)
    n_dropped = n_before - len(gdf)
    if n_dropped:
        logger.info("polygons_empty_dropped", n=n_dropped)

    crosswalk = _load_region_code_to_hcat(mapping_csv, region_prefix=region_prefix).to_pandas()
    gdf["original_code"] = gdf["original_code"].astype(str)
    gdf = gdf.merge(crosswalk, on="original_code", how="left")
    n_unmapped = int(gdf["hcat4_name"].isna().sum())
    if n_unmapped:
        # Honest fallback: any unmapped code becomes ``other`` (the plan verified
        # 0 unmapped, this guards a future input drift instead of crashing).
        gdf["hcat4_name"] = gdf["hcat4_name"].fillna(OTHER_CLASS_NAME)
        logger.warning("polygons_unmapped_to_other", n=n_unmapped)

    class_table, name_to_id = build_class_table(gdf, min_support=min_support)
    gdf["class_id"] = gdf["hcat4_name"].map(name_to_id).astype("int32")

    centroids = gdf.geometry.centroid  # projected CRS -> metric, no warning
    gdf["cx"] = centroids.x.to_numpy()
    gdf["cy"] = centroids.y.to_numpy()
    logger.info(
        "labeled_polygons_loaded",
        n_parcels=len(gdf),
        n_classes=class_table.height,
        n_unmapped=n_unmapped,
    )
    return gdf, class_table


# --------------------------------------------------------------------------- #
# Step 2 -- select dense patches + spatial folds
# --------------------------------------------------------------------------- #
#: Minimum parcel count a cell must hold to be a patch candidate. Floors out the
#: single-giant-field cells (high coverage but one class) so each patch is a
#: multi-class mosaic like a PASTIS patch -- the diversity US-079's transfer needs.
MIN_PARCELS_PER_PATCH: int = 30


def select_dense_patches(
    gdf: gpd.GeoDataFrame, *, n_patches: int, min_parcels: int = MIN_PARCELS_PER_PATCH
) -> list[PatchPlan]:
    """Pick the ``n_patches`` highest-coverage grid cells as patch bboxes.

    Bins parcel centroids onto a :data:`PATCH_SIDE_M`-metre grid in EPSG:3035 and
    keeps the cells whose parcels cover the MOST AREA. Ranking by parcel area (not
    just centroid count) directly targets the AC3 coverage objective: a cell can
    hold hundreds of tiny parcels yet cover little ground (roads/built-up between
    them), so the count-densest cell is not the coverage-densest one (verified:
    one 296-parcel cell covered only 13% while a 349-parcel cell covered 80%).
    Each cell's bbox is the patch's 1.28 km footprint, reprojected to EPSG:4326
    for the Sentinel Hub request. The spatial fold is assigned by a coarser
    super-cell so neighbouring patches share a fold and never leak across the CV
    split (US-079).

    Args:
        gdf: Labelled parcels with projected centroid columns ``cx``/``cy`` and an
            ``area_ha`` column.
        n_patches: Number of dense cells (patches) to return.
        min_parcels: Minimum parcel count for a cell to qualify (drops the
            single-giant-field cells; keeps multi-class mosaics).

    Returns:
        A list of :class:`PatchPlan`, highest-coverage first, length
        ``<= n_patches``.
    """
    import geopandas as gpd
    from shapely.geometry import box

    cx = gdf["cx"].to_numpy()
    cy = gdf["cy"].to_numpy()
    class_ids = gdf["class_id"].to_numpy()
    area_ha = gdf["area_ha"].to_numpy()

    # Integer cell index per centroid (floor division by the patch side in metres).
    ix = np.floor(cx / PATCH_SIDE_M).astype(np.int64)
    iy = np.floor(cy / PATCH_SIDE_M).astype(np.int64)
    cell = pl.DataFrame(
        {
            "ix": ix,
            "iy": iy,
            "class_id": class_ids.astype(np.int64),
            "area_ha": area_ha.astype(np.float64),
        }
    )
    # Rank by total parcel area in the cell (the coverage proxy), count as tiebreak,
    # after dropping cells below the parcel-count floor (single-field cells).
    dense = (
        cell.group_by(["ix", "iy"])
        .agg(
            n_parcels=pl.len(),
            area_ha=pl.col("area_ha").sum(),
            classes_present=pl.col("class_id").unique().sort(),
        )
        .filter(pl.col("n_parcels") >= min_parcels)
        .sort(["area_ha", "n_parcels"], descending=[True, True])
        .head(n_patches)
    )

    # Reproject all selected cell bboxes (3035) -> 4326 in one batch.
    boxes_3035 = [
        box(
            int(r["ix"]) * PATCH_SIDE_M,
            int(r["iy"]) * PATCH_SIDE_M,
            (int(r["ix"]) + 1) * PATCH_SIDE_M,
            (int(r["iy"]) + 1) * PATCH_SIDE_M,
        )
        for r in dense.iter_rows(named=True)
    ]
    gboxes = gpd.GeoSeries(boxes_3035, crs=PROJECTED_CRS).to_crs(GEOGRAPHIC_CRS)

    plans: list[PatchPlan] = []
    for patch_id, (row, geom_3035, geom_4326) in enumerate(
        zip(dense.iter_rows(named=True), boxes_3035, gboxes, strict=True)
    ):
        ixi, iyi = int(row["ix"]), int(row["iy"])
        fold = _fold_for_cell(ixi, iyi)
        plans.append(
            PatchPlan(
                patch_id=patch_id,
                bbox_3035=tuple(map(float, geom_3035.bounds)),  # type: ignore[arg-type]
                bbox_4326=tuple(map(float, geom_4326.bounds)),  # type: ignore[arg-type]
                n_parcels=int(row["n_parcels"]),
                fold=fold,
                classes_present=tuple(int(c) for c in row["classes_present"]),
            )
        )
    logger.info(
        "dense_patches_selected",
        n_requested=n_patches,
        n_selected=len(plans),
        max_parcels=plans[0].n_parcels if plans else 0,
        min_parcels=plans[-1].n_parcels if plans else 0,
    )
    return plans


def _fold_for_cell(ix: int, iy: int) -> int:
    """Assign a spatial fold from the patch's coarse super-cell.

    Patches inside the same :data:`FOLD_SUPERCELL_PATCHES`-wide super-cell share a
    fold, so geographically adjacent patches never split across train/test in the
    US-079 cross-validation (no spatial leakage).

    Args:
        ix: Patch grid column index.
        iy: Patch grid row index.

    Returns:
        A deterministic fold id in ``[0, N_SPATIAL_FOLDS)``.
    """
    sx = ix // FOLD_SUPERCELL_PATCHES
    sy = iy // FOLD_SUPERCELL_PATCHES
    # Deterministic hash of the super-cell so the fold map is stable across runs.
    return int((sx * 73_856_093) ^ (sy * 19_349_663)) % N_SPATIAL_FOLDS


# --------------------------------------------------------------------------- #
# Step 3 -- download the temporal stack (one ORBIT tile per patch)
# --------------------------------------------------------------------------- #
@dataclass
class _PatchStack:
    """A downloaded patch tile aligned to the rasterisation grid.

    Attributes:
        stack: Reflectance ``(T, 10, 128, 128)`` float32 (SH scale, [0, 1]-ish).
        transform: The ``rasterio`` Affine of the 128x128 crop (its OWN grid).
        crs: The tile CRS (for reprojecting parcels into the mask grid).
        residual_cloud: Fraction of (frame, pixel) entries SCL-masked to 0.
    """

    stack: np.ndarray
    transform: Affine
    crs: CRS
    residual_cloud: float


def download_patch_series(
    client: SentinelHubClient,
    plan: PatchPlan,
    *,
    date_from: str,
    date_to: str,
    n_frames: int,
    max_cloud: float,
) -> _PatchStack | None:
    """Download one patch's temporal stack + its exact rasterisation grid.

    Issues ONE ORBIT tile request for the patch bbox (the whole season per-pixel
    SCL cloud-masked) via :func:`ml.ingest.sh_path._download_tile`, then crops the
    centred ``128x128`` window and derives the window's child Affine from the
    tile's own transform so the mask can be rasterised on the identical grid.

    Args:
        client: The Sentinel Hub client (token + http + 429 retry + disk cache).
        plan: The patch plan (its 4326 bbox is the request footprint).
        date_from: Season start ISO date (``YYYY-MM-DD``).
        date_to: Season end ISO date.
        n_frames: Max temporal frames to request.
        max_cloud: Max scene cloud cover (scene gate; SCL masks per pixel on top).

    Returns:
        A :class:`_PatchStack` with the aligned image + transform, or ``None``
        when the tile failed or carried fewer than two usable frames.
    """
    from affine import Affine

    from ml.ingest.sh_path import _download_tile

    min_lon, min_lat, max_lon, max_lat = plan.bbox_4326
    # Request the tile slightly oversized so the centred 128x128 window is fully
    # inside it (avoids edge resampling artefacts at the crop border).
    margin = max(PATCH_PX // 4, 8)
    width = PATCH_PX + 2 * margin
    height = PATCH_PX + 2 * margin
    tile = _download_tile(
        client,
        (min_lon, min_lat, max_lon, max_lat),
        date_from=date_from,
        date_to=date_to,
        n_frames=n_frames,
        width=width,
        height=height,
        max_cloud=max_cloud,
    )
    if tile is None:
        logger.info("patch_tile_failed", patch_id=plan.patch_id)
        return None
    stack_full, ds = tile
    try:
        h, w = stack_full.shape[2], stack_full.shape[3]
        r0 = (h - PATCH_PX) // 2
        c0 = (w - PATCH_PX) // 2
        if r0 < 0 or c0 < 0:
            logger.info("patch_tile_too_small", patch_id=plan.patch_id, shape=(h, w))
            return None
        window = stack_full[:, :, r0 : r0 + PATCH_PX, c0 : c0 + PATCH_PX]
        # Child transform: the crop's grid is the tile grid translated by (c0, r0).
        win_transform = ds.transform * Affine.translation(c0, r0)
        crs = ds.crs
    finally:
        ds.close()

    keep = [f for f in range(window.shape[0]) if np.abs(window[f]).sum() > 0.0]
    if len(keep) < 2:
        logger.info("patch_insufficient_frames", patch_id=plan.patch_id, n=len(keep))
        return None
    kept = window[keep].astype(np.float32)
    # Residual cloud = fraction of (frame, pixel) positions zeroed by the SCL mask
    # (a fully-clear band has no exact-zero pixel after reflectance scaling).
    zeroed = float(np.mean(np.all(kept == 0.0, axis=1)))
    return _PatchStack(stack=kept, transform=win_transform, crs=crs, residual_cloud=zeroed)


# --------------------------------------------------------------------------- #
# Step 4 -- rasterise the dense semantic mask on the tile grid
# --------------------------------------------------------------------------- #
def rasterize_patch_mask(
    gdf: gpd.GeoDataFrame,
    patch_stack: _PatchStack,
) -> np.ndarray:
    """Rasterise parcel classes onto the patch's exact image grid.

    Reprojects the labelled parcels into the tile CRS, keeps those intersecting
    the patch window, and burns each parcel's ``class_id`` with
    :func:`rasterio.features.rasterize` using the WINDOW transform (the image's
    own grid) so the mask is pixel-perfect aligned with the reflectance stack.
    ``fill=0`` leaves un-covered pixels as background.

    Args:
        gdf: The labelled parcels (with ``class_id``); any CRS, reprojected here.
        patch_stack: The downloaded patch carrying its transform + CRS.

    Returns:
        The dense mask ``(128, 128) int32`` (0 = background, else ``class_id``).
    """
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds

    transform = patch_stack.transform
    minx, miny, maxx, maxy = array_bounds(PATCH_PX, PATCH_PX, transform)
    parcels = gdf.to_crs(patch_stack.crs)
    # Spatial filter to the patch bbox (cheap; avoids rasterising the whole region).
    in_patch = parcels.cx[minx:maxx, miny:maxy]
    if in_patch.empty:
        return np.zeros((PATCH_PX, PATCH_PX), dtype=np.int32)
    shapes = [
        (geom, int(cid))
        for geom, cid in zip(in_patch.geometry, in_patch["class_id"], strict=True)
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((PATCH_PX, PATCH_PX), dtype=np.int32)
    mask: np.ndarray = rasterize(
        shapes,
        out_shape=(PATCH_PX, PATCH_PX),
        transform=transform,
        fill=BACKGROUND_ID,
        dtype="int32",
        all_touched=False,
    )
    return mask.astype(np.int32)


# --------------------------------------------------------------------------- #
# Step 5 -- persist in the PASTIS layout
# --------------------------------------------------------------------------- #
def _frame_doys(n_frames: int, date_from: str, date_to: str) -> list[int]:
    """Spread ``n_frames`` acquisition day-of-years across the season.

    The ORBIT evalscript returns the most-recent ``n_frames`` clear acquisitions
    but not their exact timestamps, so the season is sampled at even intervals to
    give each kept frame a plausible, monotonically increasing DOY (the dense
    models use DOY only for the temporal position encoding). This is an
    approximation of the acquisition calendar, documented as such.

    Args:
        n_frames: Number of kept frames ``T``.
        date_from: Season start ISO date.
        date_to: Season end ISO date.

    Returns:
        A list of ``T`` day-of-year integers, ascending, spanning the season.
    """
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    span = max((d1 - d0).days, 1)
    if n_frames == 1:
        mids = [d0 + timedelta(days=span // 2)]
    else:
        mids = [d0 + timedelta(days=round(span * i / (n_frames - 1))) for i in range(n_frames)]
    return [int(d.timetuple().tm_yday) for d in mids]


def save_pastis_format(
    out_dir: Path,
    plan: PatchPlan,
    patch_stack: _PatchStack,
    mask: np.ndarray,
    *,
    date_from: str,
    date_to: str,
) -> PatchResult:
    """Write one patch in the PASTIS layout and return its summary stats.

    Persists, mirroring ``data/PASTIS-R``:
      * ``DATA_S2/S2_<id>.npy`` -- ``(T, 10, 128, 128)`` int16 reflectance DN.
      * ``ANNOTATIONS/TARGET_<id>.npy`` -- ``(128, 128)`` int32 class mask.
      * ``ANNOTATIONS/dates_<id>.npy`` -- ``(T,)`` int32 day-of-year per frame.

    Args:
        out_dir: Dataset root (``DATA_S2`` / ``ANNOTATIONS`` created under it).
        plan: The patch plan.
        patch_stack: The downloaded aligned stack.
        mask: The rasterised dense mask ``(128, 128)``.
        date_from: Season start ISO date (for the DOY calendar).
        date_to: Season end ISO date.

    Returns:
        A :class:`PatchResult` with the patch's coverage / class / texture stats.
    """
    s2_dir = out_dir / "DATA_S2"
    ann_dir = out_dir / "ANNOTATIONS"
    s2_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    stack = patch_stack.stack  # (T, 10, 128, 128) float32 reflectance
    t = stack.shape[0]
    dn = np.clip(stack * PASTIS_DN_SCALE, -32_768, 32_767).astype(np.int16)
    doys = _frame_doys(t, date_from, date_to)

    np.save(s2_dir / f"S2_{plan.patch_id}.npy", dn)
    np.save(ann_dir / f"TARGET_{plan.patch_id}.npy", mask.astype(np.int32))
    np.save(ann_dir / f"dates_{plan.patch_id}.npy", np.asarray(doys, dtype=np.int32))

    crop_px = mask.size
    fg = mask != BACKGROUND_ID
    coverage = float(fg.mean())
    present, counts = np.unique(mask[fg], return_counts=True)
    class_support = {int(c): int(n) for c, n in zip(present, counts, strict=True)}
    ndvi_std = _mean_ndvi_std(stack)
    result = PatchResult(
        patch_id=plan.patch_id,
        n_dates=t,
        dates_doy=doys,
        coverage=coverage,
        n_classes_present=len(class_support),
        class_support=class_support,
        ndvi_std=ndvi_std,
        residual_cloud=patch_stack.residual_cloud,
        ok=True,
    )
    logger.info(
        "patch_saved",
        patch_id=plan.patch_id,
        n_dates=t,
        coverage=round(coverage, 3),
        n_classes=len(class_support),
        ndvi_std=round(ndvi_std, 3),
        crop_px=crop_px,
    )
    return result


def _mean_ndvi_std(stack: np.ndarray) -> float:
    """Mean over frames of the per-frame spatial NDVI std (texture proxy).

    NDVI is scale-invariant, so this compares directly with PASTIS regardless of
    the reflectance DN scaling. A flat (texture-less) patch gives ~0; a real crop
    mosaic gives ~0.2, the PASTIS reference.

    Args:
        stack: Reflectance stack ``(T, 10, 128, 128)`` in PASTIS band order.

    Returns:
        The mean per-frame spatial standard deviation of NDVI.
    """
    b04 = stack[:, PASTIS_BANDS.index("B04")].astype(np.float64)
    b08 = stack[:, PASTIS_BANDS.index("B08")].astype(np.float64)
    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
    per_frame_std = [float(np.nanstd(ndvi[f])) for f in range(ndvi.shape[0])]
    return float(np.mean(per_frame_std)) if per_frame_std else 0.0


def write_metadata(
    out_dir: Path, class_table: pl.DataFrame, results: list[tuple[PatchPlan, PatchResult]]
) -> Path:
    """Write the dataset ``metadata.parquet`` (mirrors PASTIS ``metadata.geojson``).

    One row per successfully written patch with ``patch_id``, the 4326 bbox,
    ``n_parcelas``, ``n_fechas``, ``clases_presentes``, ``pct_cubierto`` and the
    ``fold_espacial`` (for leak-free CV in US-079). The class table is written
    alongside as ``class_table.parquet`` so consumers resolve ids -> names.

    Args:
        out_dir: Dataset root.
        class_table: The contiguous class id table.
        results: ``(plan, result)`` pairs for the written patches.

    Returns:
        The path of the written ``metadata.parquet``.
    """
    rows: list[dict[str, object]] = []
    for plan, res in results:
        rows.append(
            {
                "patch_id": plan.patch_id,
                "bbox_min_lon": plan.bbox_4326[0],
                "bbox_min_lat": plan.bbox_4326[1],
                "bbox_max_lon": plan.bbox_4326[2],
                "bbox_max_lat": plan.bbox_4326[3],
                "n_parcelas": plan.n_parcels,
                "n_fechas": res.n_dates,
                "clases_presentes": sorted(res.class_support.keys()),
                "pct_cubierto": res.coverage,
                "ndvi_std": res.ndvi_std,
                "residual_cloud": res.residual_cloud,
                "requests": res.requests,
                "fold_espacial": plan.fold,
            }
        )
    meta = pl.DataFrame(rows) if rows else pl.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.parquet"
    meta.write_parquet(meta_path)
    class_table.write_parquet(out_dir / "class_table.parquet")
    logger.info("metadata_written", path=str(meta_path), n_patches=meta.height)
    return meta_path


def patch_artifacts_exist(out_dir: Path, patch_id: int) -> bool:
    """Return whether a patch's three artefacts already exist (for resume).

    Args:
        out_dir: Dataset root.
        patch_id: The patch id.

    Returns:
        ``True`` when ``S2_<id>``, ``TARGET_<id>`` and ``dates_<id>`` are all
        present, so the runner can skip an already-built patch on resume.
    """
    s2 = out_dir / "DATA_S2" / f"S2_{patch_id}.npy"
    tgt = out_dir / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
    dts = out_dir / "ANNOTATIONS" / f"dates_{patch_id}.npy"
    return s2.is_file() and tgt.is_file() and dts.is_file()


def load_patch_result_stats(out_dir: Path, plan: PatchPlan) -> PatchResult:
    """Recompute a written patch's stats from disk (for resumed runs).

    When the runner resumes and a patch is already on disk, its in-memory stats
    are gone; this reloads the saved arrays and recomputes coverage / classes /
    NDVI std so the final report (and metadata) is complete and honest.

    Args:
        out_dir: Dataset root.
        plan: The patch plan whose artefacts are on disk.

    Returns:
        A :class:`PatchResult` reconstructed from the persisted arrays
        (``requests=0`` since a resumed patch issues no new request).
    """
    dn = np.load(out_dir / "DATA_S2" / f"S2_{plan.patch_id}.npy").astype(np.float32)
    mask = np.load(out_dir / "ANNOTATIONS" / f"TARGET_{plan.patch_id}.npy")
    doys = np.load(out_dir / "ANNOTATIONS" / f"dates_{plan.patch_id}.npy").tolist()
    refl = dn / PASTIS_DN_SCALE
    fg = mask != BACKGROUND_ID
    present, counts = np.unique(mask[fg], return_counts=True)
    class_support = {int(c): int(n) for c, n in zip(present, counts, strict=True)}
    zeroed = float(np.mean(np.all(dn == 0, axis=1)))
    return PatchResult(
        patch_id=plan.patch_id,
        n_dates=int(dn.shape[0]),
        dates_doy=[int(d) for d in doys],
        coverage=float(fg.mean()),
        n_classes_present=len(class_support),
        class_support=class_support,
        requests=0,
        ndvi_std=_mean_ndvi_std(refl),
        residual_cloud=zeroed,
        ok=True,
    )


def write_class_mapping_doc(out_dir: Path, class_table: pl.DataFrame) -> Path:
    """Write a small JSON documenting the HCAT -> class-id mapping (AC4).

    Args:
        out_dir: Dataset root.
        class_table: The contiguous class id table.

    Returns:
        The path of the written ``class_mapping.json``.
    """
    mapping = {
        "background_id": BACKGROUND_ID,
        "other_class_name": OTHER_CLASS_NAME,
        "classes": class_table.to_dicts(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "class_mapping.json"
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
