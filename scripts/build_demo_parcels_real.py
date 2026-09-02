"""Vectorize REAL PASTIS-R parcels into a WGS84 GeoJSON for the map demo.

The Tuscany "demo" used 4 hardcoded rectangles with invented crops. This builds
a HONEST replacement: it reads real PASTIS-R annotations — ``ParcelIDs_<id>.npy``
(real parcel boundaries) and ``TARGET_<id>.npy[0]`` (the real semantic crop per
pixel) — for a few French patches, vectorizes each parcel, assigns its majority
crop class, reprojects from the patch UTM CRS to EPSG:4326, and writes a GeoJSON
``FeatureCollection`` (``geometry`` + ``crop_class`` + ``parcel_id``) the frontend
paints with its existing crop palette + legend.

This is GROUND TRUTH (real parcels, real labels). A "prediction" layer can be
added later by joining the model's per-parcel class; the geometry pipeline is the
same.

Usage:
    poetry run python scripts/build_demo_parcels_real.py --n-patches 3
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio.features
import structlog
import typer
from pyproj import Transformer
from rasterio.transform import from_bounds

from ml.data.pastis_filter import PASTIS_CLASS_NAMES

log = structlog.get_logger(__name__)

_PASTIS_ROOT = Path("data/PASTIS-R")
_ANN = _PASTIS_ROOT / "ANNOTATIONS"
_META = _PASTIS_ROOT / "metadata.geojson"
_OUT = Path("frontend/public/demo/parcelas_reales_francia.geojson")

_PATCH_SIDE = 128
#: Drop background (0) and the void label (19): not real crops.
_SKIP_CLASSES = {0, 19}
#: Skip slivers smaller than this (pixels) so the map stays clean.
_MIN_PARCEL_PX = 12
#: PASTIS-R metadata.geojson footprints are in RGF93 / Lambert-93 (France's
#: national CRS), NOT per-tile UTM. Reproject from here to WGS84.
_SOURCE_EPSG = 2154


def _patch_bounds_utm(geometry: dict) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) UTM bounds of a patch footprint."""
    coords = list(
        itertools.chain.from_iterable(itertools.chain.from_iterable(geometry["coordinates"]))
    )
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _majority_class(semantic: np.ndarray, mask: np.ndarray) -> int:
    """Most frequent semantic class id over the masked pixels."""
    vals = semantic[mask]
    return int(Counter(vals.tolist()).most_common(1)[0][0])


def _reproject_ring(ring: list, transformer: Transformer) -> list:
    """Reproject one GeoJSON ring (UTM) to EPSG:4326, rounded to 6 decimals."""
    out = []
    for x, y in ring:
        lon, lat = transformer.transform(x, y)
        out.append([round(lon, 6), round(lat, 6)])
    return out


def _vectorize_patch(patch_id: int, geometry: dict, tile: str) -> list[dict]:
    """Vectorize one patch's real parcels into WGS84 GeoJSON features."""
    target = np.load(_ANN / f"TARGET_{patch_id}.npy")
    semantic = target[0] if target.ndim == 3 else target
    parcel_ids = np.load(_ANN / f"ParcelIDs_{patch_id}.npy").astype(np.int32)

    minx, miny, maxx, maxy = _patch_bounds_utm(geometry)
    transform = from_bounds(minx, miny, maxx, maxy, _PATCH_SIDE, _PATCH_SIDE)
    transformer = Transformer.from_crs(_SOURCE_EPSG, 4326, always_xy=True)

    features: list[dict] = []
    for pid in np.unique(parcel_ids):
        if pid <= 0:
            continue
        mask = parcel_ids == pid
        if int(mask.sum()) < _MIN_PARCEL_PX:
            continue
        cls = _majority_class(semantic, mask)
        if cls in _SKIP_CLASSES:
            continue
        crop = PASTIS_CLASS_NAMES.get(cls, "Unknown")
        for geom, _value in rasterio.features.shapes(parcel_ids, mask=mask, transform=transform):
            rings = [_reproject_ring(r, transformer) for r in geom["coordinates"]]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": {
                        "parcel_id": int(pid),
                        "patch_id": int(patch_id),
                        "crop_class": crop,
                    },
                }
            )
    return features


def _select_patches(n_patches: int) -> list[dict]:
    """Pick the first locally-available patches with >=3 distinct crops."""
    meta = json.loads(_META.read_text(encoding="utf-8"))
    chosen: list[dict] = []
    for feat in meta["features"]:
        props = feat["properties"]
        pid = int(props["ID_PATCH"])
        if not (_ANN / f"TARGET_{pid}.npy").exists():
            continue
        if not (_ANN / f"ParcelIDs_{pid}.npy").exists():
            continue
        semantic = np.load(_ANN / f"TARGET_{pid}.npy")[0]
        crops = {int(c) for c in np.unique(semantic)} - _SKIP_CLASSES
        if len(crops) < 3:
            continue
        chosen.append({"id": pid, "geometry": feat["geometry"], "tile": str(props["TILE"])})
        if len(chosen) >= n_patches:
            break
    return chosen


def main(n_patches: int = typer.Option(3, help="Number of PASTIS patches to vectorize.")) -> None:
    """Build the real-parcels demo GeoJSON from local PASTIS-R annotations."""
    patches = _select_patches(n_patches)
    if not patches:
        raise typer.Exit(code=1)

    features: list[dict] = []
    for p in patches:
        feats = _vectorize_patch(p["id"], p["geometry"], p["tile"])
        features.extend(feats)
        log.info("patch_vectorized", patch=p["id"], tile=p["tile"], parcels=len(feats))

    # Overall WGS84 bbox for the frontend fly-to.
    all_pts = [pt for f in features for ring in f["geometry"]["coordinates"] for pt in ring]
    lons = [pt[0] for pt in all_pts]
    lats = [pt[1] for pt in all_pts]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    crop_counts = Counter(f["properties"]["crop_class"] for f in features)

    fc = {
        "type": "FeatureCollection",
        "bbox": bbox,
        "metadata": {
            "source": "PASTIS-R ground truth (real parcels + real crop labels)",
            "patches": [p["id"] for p in patches],
            "n_parcels": len(features),
            "crop_counts": dict(crop_counts),
        },
        "features": features,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(fc), encoding="utf-8")
    log.info(
        "geojson_written",
        path=str(_OUT),
        n_parcels=len(features),
        bbox=[round(b, 4) for b in bbox],
        crops=dict(crop_counts),
    )


if __name__ == "__main__":
    typer.run(main)
