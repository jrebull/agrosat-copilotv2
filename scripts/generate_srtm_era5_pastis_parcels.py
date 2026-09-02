"""Re-sample SRTM and ERA5 over the real polygons of the 85k PASTIS-R parcels.

Variant of `generate_fusion_blocks_pastis.py` that, instead of operating on
patch centroids, takes the vectorized parcel-level geoparquet (output of
`scripts/vectorize_pastis_parcels.py`).

Processes in batches because the US-016 samplers send all polygons in a
single request, which exceeds the GEE payload limit (10 MB) with
85k polygons.

Outputs:
    data/cache/gee/srtm_pastis_fr_parcels_enriched.parquet
    data/cache/gee/era5_pastis_fr_parcels_<year>_enriched.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import polars as pl
import structlog
import typer

from ml.ingest.gee_sampler import (
    init_ee,
    sample_era5_monthly_climate,
    sample_srtm_terrain,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _batch_sample(
    gdf: gpd.GeoDataFrame,
    sampler_fn,
    cache_dir: Path,
    cache_key_prefix: str,
    batch_size: int,
    **sampler_kwargs,
) -> pl.DataFrame:
    """Call the sampler in batches and concatenate the results.

    Each batch uses a distinct cache_key so as not to collide with previous
    caches. The temporary caches are kept so the process can be resumed if
    it is interrupted.
    """
    frames: list[pl.DataFrame] = []
    total = len(gdf)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = gdf.iloc[start:end].copy()
        chunk_key = f"{cache_key_prefix}_batch_{start:06d}_{end:06d}"

        t_batch = time.time()
        try:
            df_chunk = sampler_fn(chunk, cache_dir=cache_dir, cache_key=chunk_key, **sampler_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_sample_failed", start=start, end=end, error=str(exc))
            continue
        dt = time.time() - t_batch
        if df_chunk.height > 0:
            frames.append(df_chunk)
            logger.info(
                "batch_done",
                start=start,
                end=end,
                n_rows=df_chunk.height,
                elapsed_s=round(dt, 1),
            )
        else:
            logger.warning("batch_empty", start=start, end=end)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


@app.command()
def main(
    parcels: Path = typer.Option(
        Path("data/processed/pastis_parcels_full.geoparquet"),
        "--parcels",
        help="GeoParquet de parcelas vectorizadas",
    ),
    out_dir: Path = typer.Option(
        Path("data/cache/gee"),
        "--out-dir",
        help="Directorio de salida",
    ),
    year: int = typer.Option(2019, "--year", help="Año de referencia ERA5"),
    cache_key: str = typer.Option(
        "pastis_fr_parcels",
        "--cache-key",
        help="Cache key (afecta naming de output)",
    ),
    skip_srtm: bool = typer.Option(False, "--skip-srtm", help="Saltar SRTM"),
    skip_era5: bool = typer.Option(False, "--skip-era5", help="Saltar ERA5"),
    batch_size: int = typer.Option(
        2000,
        "--batch-size",
        help="Parcelas por batch (límite GEE payload 10MB, ~3000 max para polígonos pequeños)",
    ),
    limit: int = typer.Option(0, "--limit", help="Si > 0, procesa solo las primeras N parcelas"),
) -> None:
    """Re-sample SRTM and ERA5 over real PASTIS parcel polygons."""
    if not parcels.exists():
        logger.error("parcels_missing", path=str(parcels))
        raise typer.Exit(code=2)

    logger.info("loading_parcels", path=str(parcels))
    gdf = gpd.read_parquet(parcels)
    logger.info("parcels_loaded", n_parcels=len(gdf))

    if limit > 0:
        gdf = gdf.head(limit).copy()
        logger.info("limited_for_smoke", n=len(gdf))

    # The existing samplers expect an int parcel_id. We reassign to a sequential
    # int for compatibility. The mapping is preserved in parcel_id_str.
    gdf = gdf.reset_index(drop=True)
    gdf["parcel_id_str"] = gdf["parcel_id"].astype(str)
    gdf["parcel_id"] = gdf.index.astype("int64")

    out_dir.mkdir(parents=True, exist_ok=True)
    init_ee()

    mapping = pl.DataFrame(
        {
            "parcel_id": gdf["parcel_id"].astype("int64").tolist(),
            "parcel_id_str": gdf["parcel_id_str"].tolist(),
        }
    )

    if not skip_srtm:
        logger.info("srtm_sampling_started", n=len(gdf), batch_size=batch_size)
        t0 = time.time()
        df_srtm = _batch_sample(
            gdf,
            sample_srtm_terrain,
            cache_dir=out_dir,
            cache_key_prefix=cache_key,
            batch_size=batch_size,
        )
        elapsed = time.time() - t0
        logger.info(
            "srtm_sampling_done",
            n_rows=df_srtm.height,
            elapsed_s=round(elapsed, 1),
            elapsed_min=round(elapsed / 60, 1),
        )
        if df_srtm.height > 0:
            df_srtm_enriched = df_srtm.join(mapping, on="parcel_id", how="left")
            out_srtm = out_dir / f"srtm_{cache_key}_enriched.parquet"
            df_srtm_enriched.write_parquet(out_srtm)
            logger.info("srtm_enriched_written", path=str(out_srtm), n_rows=df_srtm_enriched.height)

    if not skip_era5:
        logger.info("era5_sampling_started", n=len(gdf), year=year, batch_size=batch_size)
        t0 = time.time()
        df_era5 = _batch_sample(
            gdf,
            sample_era5_monthly_climate,
            cache_dir=out_dir,
            cache_key_prefix=cache_key,
            batch_size=batch_size,
            year=year,
        )
        elapsed = time.time() - t0
        logger.info(
            "era5_sampling_done",
            n_rows=df_era5.height,
            elapsed_s=round(elapsed, 1),
            elapsed_min=round(elapsed / 60, 1),
        )
        if df_era5.height > 0:
            df_era5_enriched = df_era5.join(mapping, on="parcel_id", how="left")
            out_era5 = out_dir / f"era5_{cache_key}_{year}_enriched.parquet"
            df_era5_enriched.write_parquet(out_era5)
            logger.info("era5_enriched_written", path=str(out_era5), n_rows=df_era5_enriched.height)


if __name__ == "__main__":
    app()
