"""Tile ("path") Sentinel Hub download: one big crop, sample parcels in memory.

The per-parcel client (:mod:`ml.ingest.sh_client`) issues one Process API request
per parcel. When parcels are geographically clustered, that is wasteful: a single
georeferenced TILE covering a whole neighbourhood can be downloaded once and every
parcel inside it cropped locally (in memory), turning ~N requests into ~N_clusters
requests (20-30x fewer for concentrated datasets).

Strategy
--------
1. Cluster the parcel centroids spatially (KMeans on lon/lat) so each cluster
   spans an area small enough for one tile under the Sentinel Hub size limit
   (S2L2A: >= ~1500 m/px floor, and a per-request pixel cap; we keep tiles well
   under it by capping the cluster span and the output width/height).
2. For each cluster, download ONE multi-temporal ORBIT tile (the whole season in
   one request, per-pixel SCL cloud-masked exactly like the per-parcel path).
3. For each parcel in the cluster, crop a ``patch x patch`` window centred on its
   centroid out of the tile, in memory (no extra request), and assemble its
   ``(T, 10, patch, patch)`` stack.

PROJ note (Windows): rasterio can pick up a stale ``proj.db`` from a PostgreSQL/
PostGIS install on PATH, breaking CRS transforms. This module forces rasterio's
own PROJ data at import (see ``_force_rasterio_proj``).

Honesty
-------
- A parcel whose window falls outside its tile, or whose frames are all cloud, is
  dropped and counted -- never fabricated.
- The tile is real Sentinel-2 L2A reflectance; the in-memory crop is an exact
  pixel subset, identical to what a per-parcel request would have returned for the
  same window (modulo tile-edge resampling, which the cluster padding avoids).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ml.ingest.sh_client import PASTIS_BANDS, SentinelHubClient, _orbit_evalscript

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rasterio.io import DatasetReader

logger = structlog.get_logger(__name__)

__all__ = ["TileParcels", "download_parcels_by_tile"]


def _force_rasterio_proj() -> None:
    """Point PROJ at rasterio's own data dir (avoid a stale PostGIS proj.db)."""
    import rasterio

    proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    if os.path.isdir(proj_dir):
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"] = proj_dir


_force_rasterio_proj()

#: Sentinel-2 native resolution (m/px) for the 10 m bands.
_S2_RES_M: float = 10.0

#: Max tile side in pixels (well under the Process API cap; a 1024 px tile at 10 m
#: covers ~10 km, enough for a tight parcel cluster).
_MAX_TILE_PX: int = 1024

#: Degrees of padding added around a cluster bbox so every parcel's patch window
#: stays fully inside the tile (~110 m at this latitude).
_PAD_DEG: float = 0.001


@dataclass
class TileParcels:
    """Per-parcel stacks assembled from clustered tiles.

    Attributes:
        stacks: List aligned with the input order; each item is a parcel's
            ``(T, 10, patch, patch)`` stack or ``None`` when it could not be built.
        n_tiles: Number of tile requests issued.
        n_ok: Number of parcels with a usable stack.
    """

    stacks: list[np.ndarray | None]
    n_tiles: int
    n_ok: int


def _cluster_centroids(
    lons: np.ndarray, lats: np.ndarray, *, patch: int, max_tile_px: int
) -> np.ndarray:
    """Assign each parcel to a spatial cluster small enough for one tile.

    Picks the cluster count so the average cluster spans fewer than
    ``max_tile_px`` pixels per side at 10 m. Falls back to one cluster when KMeans
    is unavailable.

    Args:
        lons: Parcel longitudes ``(n,)``.
        lats: Parcel latitudes ``(n,)``.
        patch: Per-parcel patch side in pixels (for the span budget).
        max_tile_px: Max tile side in pixels.

    Returns:
        Integer cluster label per parcel ``(n,)``.
    """
    n = lons.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    # Rough span of the whole set in pixels, to choose a cluster count that keeps
    # each cluster tile under the pixel cap (with room for the patch + padding).
    lat0 = float(np.mean(lats))
    m_per_deg_lon = 111_000.0 * np.cos(np.radians(lat0))
    span_px_lon = (lons.max() - lons.min()) * m_per_deg_lon / _S2_RES_M
    span_px_lat = (lats.max() - lats.min()) * 111_000.0 / _S2_RES_M
    budget = max(max_tile_px - patch - 32, max_tile_px // 2)
    k = int(max(1, np.ceil(max(span_px_lon, span_px_lat) / budget)))
    k = min(k, n)
    if k <= 1:
        return np.zeros(n, dtype=int)
    try:
        from sklearn.cluster import KMeans

        coords = np.column_stack([lons, lats])
        labels: np.ndarray = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
        return labels
    except Exception as exc:  # noqa: BLE001 -- documented fallback
        logger.warning("tile_cluster_fallback_single", error=str(exc))
        return np.zeros(n, dtype=int)


def _download_tile(
    client: SentinelHubClient,
    bbox: tuple[float, float, float, float],
    *,
    date_from: str,
    date_to: str,
    n_frames: int,
    width: int,
    height: int,
    max_cloud: float,
) -> tuple[np.ndarray, DatasetReader] | None:
    """Download one multi-temporal ORBIT tile and return ``(stack, transform)``.

    Args:
        client: The Sentinel Hub client (reused for its token + http).
        bbox: Tile bbox ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326.
        date_from: Season start ISO date.
        date_to: Season end ISO date.
        n_frames: Max temporal frames.
        width: Tile width in pixels.
        height: Tile height in pixels.
        max_cloud: Max scene cloud cover.

    Returns:
        ``(stack (n_frames, 10, H, W), rasterio_dataset_reader)`` or ``None``.
    """
    import rasterio

    nb = len(PASTIS_BANDS)
    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_from}T00:00:00Z",
                            "to": f"{date_to}T23:59:59Z",
                        },
                        "maxCloudCoverage": max_cloud,
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _orbit_evalscript(n_frames),
    }
    # Reuse the client's public Process API entry point (token + URL + 429 retry),
    # instead of touching private members or hardcoding the endpoint.
    response = client.post_process(payload)
    if response is None:
        logger.warning("tile_download_failed", bbox=bbox)
        return None
    memfile = rasterio.io.MemoryFile(response.content)
    ds = memfile.open()
    flat = ds.read().astype(np.float32)  # (n_frames*10, H, W)
    if flat.shape[0] != n_frames * nb:
        logger.info("tile_unexpected_bands", got=flat.shape)
        ds.close()
        return None
    stack = flat.reshape(n_frames, nb, flat.shape[1], flat.shape[2])
    return stack, ds


def download_parcels_by_tile(
    client: SentinelHubClient,
    coords: list[tuple[float, float]],
    *,
    date_from: str,
    date_to: str,
    n_frames: int = 12,
    patch: int = 128,
    max_cloud: float = 25.0,
) -> TileParcels:
    """Download parcels by clustered tiles, cropping each in memory.

    Clusters the centroids, downloads one ORBIT tile per cluster, and crops a
    ``patch x patch`` window per parcel from its tile. Drops parcels whose window
    falls outside the tile or whose frames are all cloud.

    Args:
        client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        coords: Parcel centroids ``[(lon, lat), ...]``.
        date_from: Season start ISO date.
        date_to: Season end ISO date.
        n_frames: Max temporal frames per tile.
        patch: Per-parcel patch side in pixels.
        max_cloud: Max scene cloud cover.

    Returns:
        A :class:`TileParcels` with the per-parcel stacks (input order).
    """
    from rasterio.warp import transform as warp_tf

    lons = np.array([c[0] for c in coords], dtype=np.float64)
    lats = np.array([c[1] for c in coords], dtype=np.float64)
    labels = _cluster_centroids(lons, lats, patch=patch, max_tile_px=_MAX_TILE_PX)

    stacks: list[np.ndarray | None] = [None] * len(coords)
    n_tiles = 0
    half = patch // 2
    for cl in sorted(set(labels.tolist())):
        idx = np.where(labels == cl)[0]
        min_lon = float(lons[idx].min()) - _PAD_DEG
        max_lon = float(lons[idx].max()) + _PAD_DEG
        min_lat = float(lats[idx].min()) - _PAD_DEG
        max_lat = float(lats[idx].max()) + _PAD_DEG
        lat0 = (min_lat + max_lat) / 2.0
        m_per_deg_lon = 111_000.0 * np.cos(np.radians(lat0))
        width = int(min(_MAX_TILE_PX, max(patch, (max_lon - min_lon) * m_per_deg_lon / _S2_RES_M)))
        height = int(min(_MAX_TILE_PX, max(patch, (max_lat - min_lat) * 111_000.0 / _S2_RES_M)))
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
        n_tiles += 1
        if tile is None:
            continue
        stack, ds = tile
        try:
            xs, ys = warp_tf("EPSG:4326", ds.crs, lons[idx].tolist(), lats[idx].tolist())
            for j, parcel_i in enumerate(idx.tolist()):
                row, col = ds.index(xs[j], ys[j])
                r0, r1 = row - half, row + half
                c0, c1 = col - half, col + half
                if r0 < 0 or c0 < 0 or r1 > ds.height or c1 > ds.width:
                    continue  # window outside tile; dropped (counted via n_ok)
                window = stack[:, :, r0:r1, c0:c1]  # (n_frames, 10, patch, patch)
                keep = [f for f in range(window.shape[0]) if np.abs(window[f]).sum() > 0.0]
                if len(keep) < 2:
                    continue
                stacks[parcel_i] = window[keep].astype(np.float32)
        finally:
            ds.close()
        logger.info(
            "tile_processed",
            cluster=int(cl),
            n_parcels=int(idx.size),
            tile_px=f"{width}x{height}",
        )

    n_ok = sum(1 for s in stacks if s is not None)
    logger.info("tile_download_done", n_parcels=len(coords), n_tiles=n_tiles, n_ok=n_ok)
    return TileParcels(stacks=stacks, n_tiles=n_tiles, n_ok=n_ok)
