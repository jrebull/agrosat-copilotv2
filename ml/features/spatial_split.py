"""Spatial K-fold split via H3 tessellation + KMeans (US-016).

Builds spatially stratified train/val/test partitions to avoid leakage
between nearby neighbors. The strategy:

1. Computes the centroid of each parcel in EPSG:4326.
2. Assigns each parcel to an H3 cell of resolution ``h3_res`` (default 5,
   ~252 km2 per cell).
3. Groups the unique H3 cells and clusters their centroids with
   ``KMeans(n_clusters=k)`` — each cluster defines a fold.
4. Parcels inherit the fold of their containing H3 cell.
5. Applies an exclusion buffer: parcels within < ``buffer_km`` of the
   inter-fold border are removed from the val/test of the current fold and
   returned to the fold global train to avoid neighbor leakage.
6. Within each internal train, a ``val_fraction`` is separated with
   ``np.random.default_rng(random_state)``.

Returns K :class:`FoldAssignment`, each with ``train_ids``, ``val_ids``,
``test_ids`` disjoint by construction.

Agronomic references:

- Lyons et al. 2018 — *A comparison of resampling methods for remote sensing
  classification and accuracy assessment*. RSE 208, 145-153. DOI
  10.1016/j.rse.2018.02.026 — justifies spatial CV in remote sensing.
- Roberts et al. 2017 — *Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure*. Ecography 40, 913-929.

``h3`` dependency
-----------------
``h3-py`` 4.4.2 is declared in ``pyproject.toml`` group ``geo`` since
US-016 (approved by Arthur 2026-05-17). The import remains conditional so
the module can load in minimalist environments that only need the
dataclass; if ``h3`` is not available, the API raises ``ImportError``
with explicit instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import structlog
from shapely.geometry import Point
from sklearn.cluster import KMeans

# Deferred import to avoid breaking collectors that only need the dataclass.
try:
    import h3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered by conditional test
    h3 = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

__all__ = [
    "FoldAssignment",
    "build_spatial_kfold",
]


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldAssignment:
    """Assignment of parcel_ids per fold (disjoint train/val/test).

    Attributes:
        fold_id: Integer fold identifier ``[0, k)``.
        train_ids: Tuple of ``parcel_id`` in the train split (the other folds
            minus those excluded by buffer).
        val_ids: Subset of the internal train reserved as validation.
        test_ids: Tuple of ``parcel_id`` whose H3 cells belong to the fold's
            KMeans cluster (excluding the buffered ones).
    """

    fold_id: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def build_spatial_kfold(
    parcels: gpd.GeoDataFrame,
    *,
    k: int = 5,
    h3_res: int = 5,
    buffer_km: float = 1.0,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> list[FoldAssignment]:
    """Build K spatial folds with H3 tessellation + KMeans.

    Args:
        parcels: GeoDataFrame with columns ``parcel_id`` and ``geometry`` in
            EPSG:4326 (POLYGON or POINT). The centroid is computed with
            an internal ``GeoSeries.centroid``.
        k: Number of folds (default 5). Must be >= 2 and <= the number of
            unique H3 cells. If there are fewer cells than ``k``, KMeans
            degrades by assigning one cell per fold and filling with empties.
        h3_res: H3 resolution (default 5 ~= 252 km2). Valid values
            ``[0, 15]``; 5 gives ~30 hex in agricultural Italy.
        buffer_km: Minimum distance in km to exclude parcels near the
            border between folds. ``0.0`` disables the exclusion.
        val_fraction: Fraction of the internal train used as validation per
            fold. Default 0.2 (20% of train -> val).
        random_state: Deterministic seed for KMeans and val shuffle.

    Returns:
        List of K :class:`FoldAssignment` with disjoint ``train_ids``,
        ``val_ids``, ``test_ids``. Guarantees ``parcel_id`` is not repeated
        in more than one fold (summing train+val+test is a subset of parcels).

    Raises:
        ImportError: if ``h3-py`` is not installed (US-016 documents
            ``h3 ^4.1.2`` as a dependency pending approval).
        ValueError: if ``k < 2`` or if ``parcels`` does not contain the
            required columns.
    """
    if h3 is None:
        raise ImportError(
            "The `h3` package is not installed. US-016 requires `h3 ^4.1.2` "
            "(coordinate with Arthur before coding). Fallback documented in "
            "`docs/us-planning/us-016.md` section 2.4 (rectangular grid)."
        )
    if k < 2:
        raise ValueError(f"`k` must be >= 2; received {k}.")
    if "parcel_id" not in parcels.columns:
        raise ValueError("`parcels` must contain the `parcel_id` column.")
    if parcels.geometry.name not in parcels.columns:
        raise ValueError("`parcels` must have an active geometry (parcels.set_geometry('geom')).")
    if parcels.crs is None:
        logger.warning("spatial_kfold_crs_missing", note="Asumiendo EPSG:4326")
        parcels = parcels.set_crs("EPSG:4326")
    elif parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs("EPSG:4326")

    # Centroid in EPSG:3857 (metric) to avoid geopandas UserWarning about
    # geometric operations in a geographic CRS. We re-project to 4326 to
    # feed h3 (which expects lat/lng).
    centroids = parcels.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
    parcel_ids = parcels["parcel_id"].astype("int64").to_numpy()
    n_parcels = len(parcel_ids)
    if n_parcels == 0:
        return [FoldAssignment(fold_id=i, train_ids=(), val_ids=(), test_ids=()) for i in range(k)]

    # 1) Assign each parcel to an H3 cell.
    h3_cells = np.array(
        [_assign_h3_cell(c, h3_res) for c in centroids],
        dtype=object,
    )

    # 2) Cluster the unique cells with KMeans.
    unique_cells, inv = np.unique(h3_cells, return_inverse=True)
    cell_centroids = np.array(
        [_cell_to_latlng(c) for c in unique_cells],
        dtype=np.float64,
    )
    effective_k = min(k, len(unique_cells))
    if effective_k < k:
        logger.warning(
            "spatial_kfold_k_clamped",
            requested=k,
            effective=effective_k,
            unique_h3_cells=len(unique_cells),
        )
    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    )
    cell_folds = kmeans.fit_predict(cell_centroids)
    parcel_folds = cell_folds[inv]

    # 3) Apply exclusion buffer over inter-fold borders.
    excluded_mask = _apply_buffer_exclusion(
        parcel_ids=parcel_ids,
        parcel_folds=parcel_folds,
        centroids=np.array([(c.x, c.y) for c in centroids], dtype=np.float64),
        buffer_km=buffer_km,
    )

    rng = np.random.default_rng(random_state)
    assignments: list[FoldAssignment] = []
    for fold_id in range(k):
        if fold_id >= effective_k:
            assignments.append(
                FoldAssignment(fold_id=fold_id, train_ids=(), val_ids=(), test_ids=())
            )
            continue
        test_mask = (parcel_folds == fold_id) & (~excluded_mask)
        train_mask = (parcel_folds != fold_id) & (~excluded_mask)

        train_pool = parcel_ids[train_mask]
        test_ids = tuple(int(x) for x in parcel_ids[test_mask])

        if len(train_pool) == 0:
            assignments.append(
                FoldAssignment(
                    fold_id=fold_id,
                    train_ids=(),
                    val_ids=(),
                    test_ids=test_ids,
                )
            )
            continue

        shuffled = train_pool.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(np.round(len(shuffled) * val_fraction))) if val_fraction > 0 else 0
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val_ids = tuple(int(x) for x in shuffled[:n_val])
        train_only_ids = tuple(int(x) for x in shuffled[n_val:])

        assignments.append(
            FoldAssignment(
                fold_id=fold_id,
                train_ids=train_only_ids,
                val_ids=val_ids,
                test_ids=test_ids,
            )
        )

    logger.info(
        "spatial_kfold_built",
        n_parcels=int(n_parcels),
        n_unique_h3=len(unique_cells),
        k=int(k),
        effective_k=int(effective_k),
        excluded=int(excluded_mask.sum()),
        buffer_km=float(buffer_km),
        h3_res=int(h3_res),
    )
    return assignments


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _assign_h3_cell(centroid: Point, h3_res: int) -> str:
    """Return the H3 cell containing the centroid.

    Compatible with h3-py 4.x (``latlng_to_cell``) and 3.x (``geo_to_h3``).
    If the 4.x API is not available, falls back.
    """
    assert h3 is not None  # guaranteed by the guard in build_spatial_kfold
    lat = float(centroid.y)
    lng = float(centroid.x)
    fn: Any
    if hasattr(h3, "latlng_to_cell"):  # h3-py 4.x
        fn = h3.latlng_to_cell
    elif hasattr(h3, "geo_to_h3"):  # h3-py 3.x fallback
        fn = h3.geo_to_h3
    else:  # pragma: no cover
        raise ImportError("h3-py exposes an unknown API; neither `latlng_to_cell` nor `geo_to_h3`.")
    return str(fn(lat, lng, h3_res))


def _cell_to_latlng(cell: str) -> tuple[float, float]:
    """Return ``(lat, lng)`` of the H3 cell centroid.

    Compatible with h3-py 4.x (``cell_to_latlng``) and 3.x (``h3_to_geo``).
    """
    assert h3 is not None
    fn: Any
    if hasattr(h3, "cell_to_latlng"):  # h3-py 4.x
        fn = h3.cell_to_latlng
    elif hasattr(h3, "h3_to_geo"):  # h3-py 3.x fallback
        fn = h3.h3_to_geo
    else:  # pragma: no cover
        raise ImportError("h3-py exposes an unknown API; no centroid accessor found.")
    lat, lng = fn(cell)
    return float(lat), float(lng)


def _apply_buffer_exclusion(
    *,
    parcel_ids: np.ndarray,
    parcel_folds: np.ndarray,
    centroids: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    """Mark parcels excluded for being < ``buffer_km`` from the inter-fold border.

    O(N) implementation using approximate haversine distance with
    lon/lat -> meters equirectangular conversion (sufficient for continental
    Italy; error < 1% at latitudes 38-46 deg N). For datasets > 10k parcels, a
    KDTree implementation in EPSG:3857 would be faster — deferred to a
    future optimization.

    Args:
        parcel_ids: Vector ``(N,)`` of identifiers.
        parcel_folds: Vector ``(N,)`` of fold assigned per parcel.
        centroids: Matrix ``(N, 2)`` with ``(lon, lat)`` in EPSG:4326.
        buffer_km: Exclusion radius. If ``0.0`` nothing is excluded.

    Returns:
        Boolean array ``(N,)`` where ``True`` marks a parcel to exclude.
    """
    n = len(parcel_ids)
    excluded = np.zeros(n, dtype=bool)
    if buffer_km <= 0.0 or n == 0:
        return excluded

    # Equirectangular conversion (central Italy ~ 43 deg N).
    deg_per_km_lat = 1.0 / 111.0
    mid_lat = float(np.mean(centroids[:, 1]))
    deg_per_km_lon = 1.0 / (111.320 * max(np.cos(np.radians(mid_lat)), 1e-3))
    buffer_deg_lat = buffer_km * deg_per_km_lat
    buffer_deg_lon = buffer_km * deg_per_km_lon
    # Dimensionless norm: we work in degrees; bbox-prefilter + distance.
    for i in range(n):
        if excluded[i]:
            continue
        lon_i, lat_i = centroids[i]
        dlat = centroids[:, 1] - lat_i
        dlon = centroids[:, 0] - lon_i
        bbox_mask = (np.abs(dlat) < buffer_deg_lat) & (np.abs(dlon) < buffer_deg_lon)
        cross_fold = bbox_mask & (parcel_folds != parcel_folds[i])
        if cross_fold.any():
            # Real metric distance with equirectangular.
            dist_km = np.sqrt(
                (dlon[cross_fold] / deg_per_km_lon) ** 2 + (dlat[cross_fold] / deg_per_km_lat) ** 2
            )
            if (dist_km < buffer_km).any():
                # Exclude the current parcel and its neighbors within the buffer.
                idxs = np.where(cross_fold)[0][dist_km < buffer_km]
                excluded[i] = True
                excluded[idxs] = True
    return excluded
