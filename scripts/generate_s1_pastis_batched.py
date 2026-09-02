"""Sample Sentinel-1 GRD VV+VH over PASTIS-R polygons in batches.

The US-016 sampler sends all polygons in a single request, which exceeds the
GEE 5-minute compute limit when there are ~2433 polygons with ~130
images/parcel and Lee 7x7 despeckle. Batched chunks of 200-500 let each
request finish in <2 min individually.

Typical usage (overnight ~9h over 2433 PASTIS-R patches):

    poetry run python scripts/generate_s1_pastis_batched.py \\
        --metadata data/PASTIS-R/metadata.geojson \\
        --year 2019 \\
        --batch-size 300

Generates `data/cache/gee/s1_pastis_fr_full_<year>_both_lee_7x7_dB_enriched.parquet`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import polars as pl
import structlog
import typer
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from ml.ingest.gee_sampler import init_ee, sample_s1_roi_for_parcels

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _extract_pastis_geometries(metadata_geojson: Path, year: int) -> gpd.GeoDataFrame:
    """Read the metadata.geojson and return an EPSG:4326 GeoDataFrame."""
    with metadata_geojson.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    records = []
    for feat in gj.get("features", []):
        gd = feat.get("geometry") or {}
        if gd.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        try:
            g = shape(gd)
            if not g.is_valid:
                g = g.buffer(0)
            g_4326 = shp_transform(transformer.transform, g)
        except (ValueError, AttributeError) as exc:
            logger.warning("pastis_geom_failed", error=str(exc))
            continue
        pid = (feat.get("properties") or {}).get("ID_PATCH")
        if pid is None:
            continue
        records.append({"parcel_id": int(pid), "year": year, "geometry": g_4326})
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


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
        help="Directorio de salida",
    ),
    year: int = typer.Option(2019, "--year", help="Año de referencia S1"),
    cache_key: str = typer.Option(
        "pastis_fr_full",
        "--cache-key",
        help="Cache key naming",
    ),
    batch_size: int = typer.Option(
        300,
        "--batch-size",
        help="Polígonos por batch (S1 es caro por despeckle Lee 7x7; default 300)",
    ),
) -> None:
    """Sample S1 over PASTIS-R in resilient batches."""
    if not metadata.exists():
        logger.error("metadata_missing", path=str(metadata))
        raise typer.Exit(code=2)

    logger.info("extracting_geometries", path=str(metadata))
    gdf = _extract_pastis_geometries(metadata, year)
    logger.info("geometries_extracted", n_parcels=len(gdf))
    if gdf.empty:
        raise typer.Exit(code=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    init_ee()

    frames: list[pl.DataFrame] = []
    total = len(gdf)
    t_start = time.time()
    n_batches = (total + batch_size - 1) // batch_size

    for idx, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        chunk = gdf.iloc[start:end].copy()
        chunk_key = f"{cache_key}_batch_{start:06d}_{end:06d}"

        t_batch = time.time()
        try:
            df_chunk = sample_s1_roi_for_parcels(
                chunk,
                year=year,
                cache_dir=out_dir,
                cache_key=chunk_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_failed", start=start, end=end, error=str(exc))
            continue
        dt = time.time() - t_batch

        if df_chunk.height > 0:
            frames.append(df_chunk)
            elapsed_total = time.time() - t_start
            rate = sum(f.height for f in frames) / max(elapsed_total, 0.01)
            eta_s = (total - end) / max(rate, 0.01)
            logger.info(
                "batch_done",
                idx=idx + 1,
                total_batches=n_batches,
                start=start,
                end=end,
                n_rows=df_chunk.height,
                elapsed_batch_s=round(dt, 1),
                elapsed_total_min=round(elapsed_total / 60, 1),
                rate_parcels_per_s=round(rate, 2),
                eta_min=round(eta_s / 60, 1),
            )
        else:
            logger.warning(
                "batch_empty",
                start=start,
                end=end,
                elapsed_batch_s=round(dt, 1),
            )

    if not frames:
        logger.error("s1_no_data")
        raise typer.Exit(code=4)

    df_final = pl.concat(frames, how="vertical_relaxed")
    final_path = out_dir / f"s1_{cache_key}_{year}_both_lee_7x7_dB_enriched.parquet"
    df_final.write_parquet(final_path)
    total_elapsed = time.time() - t_start
    logger.info(
        "s1_sampling_done",
        n_rows=df_final.height,
        path=str(final_path),
        size_mb=round(final_path.stat().st_size / 1e6, 2),
        elapsed_min=round(total_elapsed / 60, 1),
    )
    print(f"Wrote: {final_path} ({df_final.shape})")


if __name__ == "__main__":
    app()
