"""Generates an AlphaEarth cache for ALL the PASTIS-R centroids (2433 patches).

Permanent operational tool. Reads `data/PASTIS-R/metadata.geojson` extracting
centroids from Polygon and MultiPolygon (the US-016 helper only processes
MultiPolygon), reprojects to EPSG:4326, and samples the AlphaEarth annual 2019
embedding with `sample_alphaearth_at_coords` in batches of 500 points.

Outputs:
    data/cache/gee/alphaearth_at_pastis_fr_full_2019_N.parquet

Usage::

    poetry run python scripts/generate_alphaearth_pastis_full.py \\
        --year 2019 \\
        --batch-size 500
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl
import structlog
import typer

from ml.ingest.gee_sampler import init_ee, sample_alphaearth_at_coords

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _polygon_centroid_2154(geom: dict) -> tuple[float, float] | None:
    """Computes the EPSG:2154 centroid of a GeoJSON Polygon or MultiPolygon.

    Returns None if the geometry is invalid.
    """
    try:
        from shapely.geometry import shape

        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        c = g.centroid
        return float(c.x), float(c.y)
    except Exception:  # noqa: BLE001 - invalid geometry of any kind yields no centroid
        return None


def _extract_all_centroids(metadata_geojson: Path) -> pl.DataFrame:
    """Reads all features (Polygon + MultiPolygon) and reprojects to EPSG:4326."""
    from pyproj import Transformer

    with metadata_geojson.open(encoding="utf-8") as fh:
        gj = json.load(fh)

    features = gj.get("features", [])
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    rows: list[dict] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        centroid = _polygon_centroid_2154(geom)
        if centroid is None:
            continue
        cx, cy = centroid
        lon, lat = transformer.transform(cx, cy)
        props = feat.get("properties") or {}
        pid = props.get("ID_PATCH") or feat.get("id")
        if pid is None:
            continue
        rows.append(
            {
                "px_id": str(pid),
                "lon": float(lon),
                "lat": float(lat),
                "tile": str(props.get("TILE", "")),
                "fold": int(props.get("Fold", 0) or 0),
            }
        )

    return pl.DataFrame(rows)


@app.command()
def main(
    metadata: Path = typer.Option(
        Path("data/PASTIS-R/metadata.geojson"),
        "--metadata",
        help="Ruta al metadata.geojson de PASTIS-R",
    ),
    out_dir: Path = typer.Option(
        Path("data/cache/gee"),
        "--out-dir",
        help="Directorio de salida del parquet AE",
    ),
    year: int = typer.Option(2019, "--year", help="Año del embedding AlphaEarth"),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        help="Tamaño de lote para reduceRegions de GEE",
    ),
) -> None:
    """Samples AlphaEarth 64-dim for all the PASTIS-R centroids."""
    if not metadata.exists():
        logger.error("metadata_missing", path=str(metadata))
        raise typer.Exit(code=2)

    logger.info("centroid_extraction_started", source=str(metadata))
    coords = _extract_all_centroids(metadata)
    logger.info(
        "centroid_extraction_done",
        n_centroids=coords.height,
        cols=coords.columns,
    )

    if coords.is_empty():
        logger.error("no_centroids_extracted")
        raise typer.Exit(code=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alphaearth_at_pastis_fr_full_{year}_{coords.height}.parquet"

    logger.info("ae_sampling_started", n=coords.height, year=year, batch_size=batch_size)
    t0 = time.time()
    init_ee()
    df_ae = sample_alphaearth_at_coords(
        coords.select(["px_id", "lon", "lat"]),
        year=year,
        cache_path=out_dir,
        cache_key=f"pastis_fr_full_{year}_{coords.height}",
        batch_size=batch_size,
    )
    dt = time.time() - t0
    logger.info(
        "ae_sampling_done",
        n_rows=df_ae.height,
        n_cols=df_ae.width,
        elapsed_seconds=round(dt, 1),
        out_path=str(out_path),
    )

    # Re-attach fold info (sample_alphaearth_at_coords solo retorna px_id, lon, lat, year, dim_*).
    df_with_meta = df_ae.join(coords.select(["px_id", "tile", "fold"]), on="px_id", how="left")
    df_with_meta.write_parquet(out_path)
    print(f"Wrote: {out_path} ({df_with_meta.shape})")


if __name__ == "__main__":
    app()
    sys.exit(0)
