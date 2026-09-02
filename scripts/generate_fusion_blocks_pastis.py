"""Generate the S1, SRTM, ERA5 and geometry blocks of the fusion matrix over PASTIS-R.

Permanent operative. Reads `data/PASTIS-R/metadata.geojson`, reprojects the
Polygon and MultiPolygon geometries to EPSG:4326, calls the existing samplers of
`ml.ingest.gee_sampler` (S1, SRTM, ERA5) and persists the cached parquets so the
03c notebook can build the 189-dim fusion matrix without re-doing the requests to
GEE.

Outputs in `data/cache/gee/`:
    s1_pastis_fr_full_2019_both_lee_7x7_dB.parquet
    srtm_pastis_fr_full.parquet
    era5_pastis_fr_full_2019_C.parquet

Geometry is computed locally with shapely+pyproj (does not require GEE).

Usage::

    poetry run python scripts/generate_fusion_blocks_pastis.py --year 2019
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import structlog
import typer
from pyproj import Transformer
from shapely.geometry import shape

from ml.ingest.gee_sampler import (
    init_ee,
    sample_era5_monthly_climate,
    sample_s1_roi_for_parcels,
    sample_srtm_terrain,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _extract_pastis_geodataframe(metadata_geojson: Path, year: int) -> gpd.GeoDataFrame:
    """Extract all PASTIS geometries (Polygon + MultiPolygon) reprojected to EPSG:4326.

    Returns a GeoDataFrame with columns `parcel_id`, `year`, `geometry`.
    """
    with metadata_geojson.open(encoding="utf-8") as fh:
        gj = json.load(fh)

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    from shapely.ops import transform as shp_transform

    records = []
    for feat in gj.get("features", []):
        geom_data = feat.get("geometry") or {}
        if geom_data.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        try:
            geom_2154 = shape(geom_data)
            if not geom_2154.is_valid:
                geom_2154 = geom_2154.buffer(0)
            geom_4326 = shp_transform(transformer.transform, geom_2154)
        except (ValueError, AttributeError) as exc:
            logger.warning("pastis_geom_failed", error=str(exc))
            continue
        props = feat.get("properties") or {}
        pid = props.get("ID_PATCH") or feat.get("id")
        if pid is None:
            continue
        records.append(
            {
                "parcel_id": int(pid),
                "year": year,
                "geometry": geom_4326,
            }
        )

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
    year: int = typer.Option(2019, "--year", help="Año de referencia"),
    cache_key: str = typer.Option(
        "pastis_fr_full",
        "--cache-key",
        help="Identificador lógico del subset para naming de cache files",
    ),
    skip_s1: bool = typer.Option(False, "--skip-s1", help="Saltar muestreo S1"),
    skip_srtm: bool = typer.Option(False, "--skip-srtm", help="Saltar muestreo SRTM"),
    skip_era5: bool = typer.Option(False, "--skip-era5", help="Saltar muestreo ERA5"),
) -> None:
    """Generate the S1, SRTM, ERA5 fusion blocks for the full PASTIS-R."""
    if not metadata.exists():
        logger.error("metadata_missing", path=str(metadata))
        raise typer.Exit(code=2)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("extracting_pastis_geometries", source=str(metadata))
    parcels = _extract_pastis_geodataframe(metadata, year=year)
    logger.info("pastis_geometries_extracted", n_parcels=len(parcels))

    if parcels.empty:
        logger.error("no_geometries_extracted")
        raise typer.Exit(code=3)

    init_ee()

    if not skip_srtm:
        logger.info("srtm_sampling_started", n=len(parcels))
        t0 = time.time()
        df_srtm = sample_srtm_terrain(parcels, cache_dir=out_dir, cache_key=cache_key)
        dt = time.time() - t0
        logger.info(
            "srtm_sampling_done",
            n_rows=df_srtm.height,
            elapsed_seconds=round(dt, 1),
        )

    if not skip_era5:
        logger.info("era5_sampling_started", n=len(parcels))
        t0 = time.time()
        df_era5 = sample_era5_monthly_climate(
            parcels, year=year, cache_dir=out_dir, cache_key=cache_key
        )
        dt = time.time() - t0
        logger.info(
            "era5_sampling_done",
            n_rows=df_era5.height,
            elapsed_seconds=round(dt, 1),
        )

    if not skip_s1:
        logger.info("s1_sampling_started", n=len(parcels))
        t0 = time.time()
        df_s1 = sample_s1_roi_for_parcels(
            parcels, year=year, cache_dir=out_dir, cache_key=cache_key
        )
        dt = time.time() - t0
        logger.info(
            "s1_sampling_done",
            n_rows=df_s1.height,
            elapsed_seconds=round(dt, 1),
        )

    logger.info("all_blocks_done", out_dir=str(out_dir), cache_key=cache_key)


if __name__ == "__main__":
    app()
    sys.exit(0)
