"""Vectorizes the PASTIS-R instance masks into per-parcel polygons.

Permanent operational tool. For each patch (2433 total) it reads
`TARGET_<id>.npy`, extracts the instance (channel 1) and semantic (channel 0)
masks, vectorizes each parcel (unique instance_id) with
`rasterio.features.shapes`, georeferences the polygon using the patch bbox in
`metadata.geojson` (EPSG:2154) and reprojects to EPSG:4326.

Generates a GeoParquet with one row per parcel:
    parcel_id (str: "<patch_id>_<instance_id>"), patch_id, instance_id,
    class_id, class_name, fold, area_m2, n_pixels, geometry (EPSG:4326).

Filters applied:
- Discards class 0 (Background) and class 19 (Void label).
- Discards parcels with n_pixels < --min-pixels (default 10).
- Discards invalid geometries after a defensive buffer(0).

Usage::

    poetry run python scripts/vectorize_pastis_parcels.py \\
        --min-pixels 10 \\
        --out data/processed/pastis_parcels_full.geoparquet
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import structlog
import typer
from rasterio.transform import from_bounds
from shapely.geometry import shape

from ml.ingest.pastis_loader import PASTIS_CLASS_MAP

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _vectorize_patch(
    patch_id: int,
    fold: int,
    bbox_2154: tuple[float, float, float, float],
    target_arr: np.ndarray,
    min_pixels: int,
) -> list[dict]:
    """Vectorizes a patch into individual parcel-polygons (in EPSG:2154).

    Args:
        patch_id: PASTIS-R patch identifier.
        fold: Official fold (1-5) inherited from the patch.
        bbox_2154: ``(minx, miny, maxx, maxy)`` of the patch in EPSG:2154.
        target_arr: Array ``(3, H, W)`` loaded from ``TARGET_<id>.npy``.
            Channel 0 = semantic, Channel 1 = instance.
        min_pixels: Minimum pixel filter per parcel.

    Returns:
        List of dicts with `parcel_id`, `patch_id`, `instance_id`, `class_id`,
        `class_name`, `fold`, `area_m2`, `n_pixels`, `geometry` (shapely still in
        EPSG:2154; reprojection is done at the end in bulk).
    """
    semantic = target_arr[0]
    instance = target_arr[1]
    h, w = semantic.shape

    # Affine transform: bbox 2154 -> (col, row) pixel grid.
    minx, miny, maxx, maxy = bbox_2154
    transform = from_bounds(minx, miny, maxx, maxy, width=w, height=h)

    records: list[dict] = []
    unique_instances = np.unique(instance)
    for inst_id in unique_instances:
        inst_id_int = int(inst_id)
        if inst_id_int == 0:
            # 0 = background with no parcel assigned in this pixel.
            continue
        mask = (instance == inst_id_int).astype(np.uint8)
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue

        # Dominant semantic class (there may be 1-2 minority pixels at the
        # edges; we take the most frequent one).
        sem_in_parcel = semantic[mask.astype(bool)]
        if sem_in_parcel.size == 0:
            continue
        cls_values, cls_counts = np.unique(sem_in_parcel, return_counts=True)
        class_id = int(cls_values[np.argmax(cls_counts)])
        if class_id in (0, 19):
            # Background or Void.
            continue

        # Vectorize the mask. shapes() returns a generator of
        # (geom_dict, value).
        try:
            shapes_iter = list(
                rasterio.features.shapes(mask, mask=mask.astype(bool), transform=transform)
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("shapes_failed", patch_id=patch_id, inst_id=inst_id_int, error=str(exc))
            continue
        if not shapes_iter:
            continue

        # If there are several disjoint polygons for the same instance_id
        # (rare but possible due to fragmented masking), we take the largest
        # one. Alternative: join them as a MultiPolygon, but downstream
        # GEE prefers simple polygons.
        best_geom = None
        best_area = -1.0
        for geom_dict, _val in shapes_iter:
            try:
                g = shape(geom_dict)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
                if g.area > best_area:
                    best_geom = g
                    best_area = g.area
            except (ValueError, AttributeError) as exc:
                logger.warning("geom_build_failed", inst_id=inst_id_int, error=str(exc))
                continue

        if best_geom is None or best_geom.is_empty:
            continue

        records.append(
            {
                "parcel_id": f"{patch_id}_{inst_id_int}",
                "patch_id": int(patch_id),
                "instance_id": inst_id_int,
                "class_id": class_id,
                "class_name": PASTIS_CLASS_MAP.get(class_id, "unknown"),
                "fold": int(fold) if fold else 0,
                "area_m2": float(best_geom.area),
                "n_pixels": n_pixels,
                "geometry": best_geom,
            }
        )

    return records


@app.command()
def main(
    pastis_root: Path = typer.Option(
        Path("data/PASTIS-R"),
        "--pastis-root",
        help="Raíz del dataset PASTIS-R",
    ),
    metadata: Path = typer.Option(
        Path("data/PASTIS-R/metadata.geojson"),
        "--metadata",
        help="metadata.geojson con bboxes y folds",
    ),
    out: Path = typer.Option(
        Path("data/processed/pastis_parcels_full.geoparquet"),
        "--out",
        help="GeoParquet de salida",
    ),
    min_pixels: int = typer.Option(10, "--min-pixels", help="Filtro mínimo de píxeles por parcela"),
    limit_patches: int = typer.Option(
        0,
        "--limit-patches",
        help="Si > 0, procesa solo los primeros N patches (smoke test)",
    ),
) -> None:
    """Vectorizes the PASTIS-R instance masks into per-parcel polygons."""
    if not metadata.exists():
        logger.error("metadata_missing", path=str(metadata))
        raise typer.Exit(code=2)
    if not pastis_root.exists():
        logger.error("pastis_root_missing", path=str(pastis_root))
        raise typer.Exit(code=2)

    annotations_dir = pastis_root / "ANNOTATIONS"
    if not annotations_dir.exists():
        logger.error("annotations_dir_missing", path=str(annotations_dir))
        raise typer.Exit(code=2)

    logger.info("vectorize_started", min_pixels=min_pixels, out=str(out))
    with metadata.open(encoding="utf-8") as fh:
        gj = json.load(fh)

    features = gj.get("features", [])
    if limit_patches > 0:
        features = features[:limit_patches]
    logger.info("patches_to_process", n=len(features))

    all_records: list[dict] = []
    n_ok = 0
    n_skip = 0
    n_pixels_total = 0
    t0 = time.time()

    for idx, feat in enumerate(features):
        props = feat.get("properties") or {}
        patch_id = props.get("ID_PATCH")
        fold = props.get("Fold", 0)
        geom_data = feat.get("geometry") or {}
        if patch_id is None or geom_data.get("type") not in ("Polygon", "MultiPolygon"):
            n_skip += 1
            continue
        try:
            geom_2154 = shape(geom_data)
            if not geom_2154.is_valid:
                geom_2154 = geom_2154.buffer(0)
            bbox = geom_2154.bounds
        except (ValueError, AttributeError) as exc:
            logger.warning("patch_geom_failed", patch_id=patch_id, error=str(exc))
            n_skip += 1
            continue

        target_path = annotations_dir / f"TARGET_{patch_id}.npy"
        if not target_path.exists():
            n_skip += 1
            continue
        try:
            target_arr = np.load(target_path)
        except (OSError, ValueError) as exc:
            logger.warning("target_load_failed", path=str(target_path), error=str(exc))
            n_skip += 1
            continue

        recs = _vectorize_patch(
            patch_id=int(patch_id),
            fold=int(fold) if fold else 0,
            bbox_2154=bbox,
            target_arr=target_arr,
            min_pixels=min_pixels,
        )
        all_records.extend(recs)
        n_ok += 1
        n_pixels_total += sum(r["n_pixels"] for r in recs)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info(
                "vectorize_progress",
                processed=idx + 1,
                total=len(features),
                n_parcels_so_far=len(all_records),
                elapsed_s=round(elapsed, 1),
                rate_parcels_per_s=round(len(all_records) / max(elapsed, 0.01), 1),
            )

    elapsed = time.time() - t0
    logger.info(
        "vectorize_done",
        patches_ok=n_ok,
        patches_skipped=n_skip,
        n_parcels=len(all_records),
        n_pixels_total=n_pixels_total,
        elapsed_s=round(elapsed, 1),
    )

    if not all_records:
        logger.error("no_parcels_extracted")
        raise typer.Exit(code=3)

    # Build GeoDataFrame in EPSG:2154 and reproject to EPSG:4326 in bulk.
    gdf_2154 = gpd.GeoDataFrame(all_records, geometry="geometry", crs="EPSG:2154")
    logger.info("reprojecting_to_4326", n=len(gdf_2154))
    gdf_4326 = gdf_2154.to_crs("EPSG:4326")

    out.parent.mkdir(parents=True, exist_ok=True)
    gdf_4326.to_parquet(out)
    file_size_mb = out.stat().st_size / 1e6
    logger.info(
        "geoparquet_written",
        path=str(out),
        n_parcels=len(gdf_4326),
        file_size_mb=round(file_size_mb, 1),
    )

    # Useful stats for validation.
    class_counts = gdf_4326["class_id"].value_counts().sort_index().to_dict()
    fold_counts = gdf_4326["fold"].value_counts().sort_index().to_dict()
    px_stats = {
        "min": int(gdf_4326["n_pixels"].min()),
        "median": int(gdf_4326["n_pixels"].median()),
        "max": int(gdf_4326["n_pixels"].max()),
        "mean": round(float(gdf_4326["n_pixels"].mean()), 1),
    }
    area_stats = {
        "min_m2": int(gdf_4326["area_m2"].min()),
        "median_m2": int(gdf_4326["area_m2"].median()),
        "max_ha": round(float(gdf_4326["area_m2"].max() / 10000), 2),
        "total_ha": round(float(gdf_4326["area_m2"].sum() / 10000), 1),
    }
    print("\n=== Stats ===")
    print(f"N parcelas: {len(gdf_4326)}")
    print(f"N clases: {gdf_4326['class_id'].nunique()}")
    print(f"Folds: {fold_counts}")
    print(f"Pixels per parcel: {px_stats}")
    print(f"Area: {area_stats}")
    print(f"Top 5 classes by count: {dict(list(class_counts.items())[:5])}")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    app()
