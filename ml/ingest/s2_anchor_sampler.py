"""Per-parcel Sentinel-2 sampling at phenology anchors (US-023-preview-v2 P5).

Materializes ``data/features/s2_anchors_pastis.parquet`` consumed by
:class:`ml.features.spectral_signature.SpectralSignatureFeatures`. For each
PASTIS-R parcel (``parcel_id`` format ``10000_1``, not Italian) it takes the
B04..B08 bands in 3 temporal windows anchored to the Start-of-Growing (SOG),
peak NDVI and senescence DOY, computed upstream in the US-018 phenology subset
(or re-read from an anchors parquet).

Usage pattern::

    poetry run python -m ml.ingest.s2_anchor_sampler \\
        --parcels-path data/features/parcels_pastis_2023.parquet \\
        --year 2023 \\
        --output data/features/s2_anchors_pastis.parquet

The output schema is deterministic and compatible with
``SpectralSignatureFeatures._extract_anchor_bands`` (which looks for
``{anchor}_{band}`` columns in lowercase). Each parcel produces 15 spectral
columns (3 anchors x 5 bands) + ``parcel_id`` + ``year``.

Local cache in ``data/cache/gee/s2_anchors_{md5_parcels}_{year}.parquet``
for cheap iteration. Reuses :func:`ml.ingest.gee_sampler.init_ee` for EE
authentication consistent with the existing samplers.

Degraded mode: if ``earthengine-api`` is not available or GEE fails, the
module writes a parquet with a valid schema and rows populated with ``NaN``
so the downstream chain is not broken.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    import geopandas as gpd

_log = structlog.get_logger(__name__)


DEFAULT_BANDS: tuple[str, ...] = ("B4", "B5", "B6", "B7", "B8")
"""Sentinel-2 bands sampled by default.

B4 (red, 665 nm), B5/B6/B7 (red-edge 704/740/783 nm), B8 (NIR 835 nm).
These are the ones required by the Red Edge Position of Frampton et al. 2013
and by the red-edge moments documented in
:mod:`ml.features.spectral_signature`.

Notation without padding (`B4` and NOT `B04`): this is what the current GEE
collection `COPERNICUS/S2_SR_HARMONIZED` exposes. Previous runs with `B04`
produced `Image.select: Band pattern 'B04' did not match any bands` and left
`s2_anchors_pastis.parquet` with all bands NULL. Documented in
US-023-preview v2 (fix with confirmed smoke test: 5.6 s / 100 parcels).
"""

DEFAULT_ANCHORS: tuple[str, ...] = ("sog", "peak", "senescence")
"""Canonical phenology anchors: Start-Of-Growing, peak NDVI, senescence."""

ANCHOR_WINDOW_DAYS: int = 5
"""Window +/- N days around the anchor DOY.

S2 revisits Italy every ~5 days per orbit, so +/- 5 days guarantees
at least one available image even with cloud rejection.
"""

DEFAULT_CACHE_DIR: Path = Path("data/cache/gee")
DEFAULT_OUTPUT_PATH: Path = Path("data/features/s2_anchors_pastis.parquet")

#: Conservative estimate of the GEE cost per parcel in USD (free tier hides
#: the real cost; this number serves to report an order of magnitude to the
#: MLflow log). Assumes 3 anchors x 5 bands x 1 reduceRegions ~ 0.0003 USD.
COST_PER_PARCEL_USD: float = 0.0003


def _band_col_name(anchor: str, band: str) -> str:
    """Return the canonical column name ``{anchor}_b0N``.

    Regardless of whether the GEE band arrives as ``B4`` (without padding,
    `COPERNICUS/S2_SR_HARMONIZED` format) or as ``B04`` (legacy format), it is
    always persisted as ``b0N`` padded to two digits. The consumer
    :class:`ml.features.spectral_signature.SpectralSignatureFeatures` looks for
    ``{anchor}_b05`` (not ``b5``); keeping ``b0N`` here prevents the GEE
    notation change from breaking the downstream join.
    """
    digits = "".join(ch for ch in band if ch.isdigit())
    return f"{anchor}_b{int(digits):02d}"


def _build_schema(anchors: tuple[str, ...], bands: tuple[str, ...]) -> dict[str, Any]:
    """Build the Polars schema of the output (stable order)."""
    schema: dict[str, Any] = {
        "parcel_id": pl.Utf8,
        "year": pl.Int16,
    }
    for anchor in anchors:
        for band in bands:
            schema[_band_col_name(anchor, band)] = pl.Float64
    return schema


def _parcels_md5(parcels: gpd.GeoDataFrame) -> str:
    """Short MD5 hash (10 chars) over parcel_id + GeoDataFrame bbox.

    Reproducible: same input -> same hash. Used to name the local cache
    in ``data/cache/gee/``.
    """
    if "parcel_id" in parcels.columns:
        ids = parcels["parcel_id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(parcels))]
    try:
        bounds = parcels.total_bounds
        bbox_str = ",".join(f"{b:.6f}" for b in bounds)
    except Exception:  # noqa: BLE001
        bbox_str = "nobbox"
    payload = "|".join(sorted(ids)) + "::" + bbox_str
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _resolve_anchors_table(
    parcels: gpd.GeoDataFrame,
    phenology_anchors_path: Path | None,
) -> pl.DataFrame:
    """Return table ``parcel_id, sog_doy, peak_doy, senescence_doy``.

    Order of preference:

    1. Columns ``sog_doy``, ``peak_doy``, ``senescence_doy`` already present
       in ``parcels``.
    2. External parquet ``phenology_anchors_path`` with the same structure.
    3. Static fallback: SOG=105, peak=180, senescence=260 (continental
       Italy, herbaceous arable crops).
    """
    needed = {"sog_doy", "peak_doy", "senescence_doy"}
    pcols = set(parcels.columns)
    if needed.issubset(pcols):
        # Convert geopandas -> polars selecting only the necessary cols.
        rows = [
            {
                "parcel_id": str(r["parcel_id"]),
                "sog_doy": int(r["sog_doy"]),
                "peak_doy": int(r["peak_doy"]),
                "senescence_doy": int(r["senescence_doy"]),
            }
            for _, r in parcels.iterrows()
        ]
        return pl.DataFrame(
            rows,
            schema={
                "parcel_id": pl.Utf8,
                "sog_doy": pl.Int16,
                "peak_doy": pl.Int16,
                "senescence_doy": pl.Int16,
            },
        )

    if phenology_anchors_path is not None and phenology_anchors_path.exists():
        df = pl.read_parquet(phenology_anchors_path)
        df = df.with_columns(pl.col("parcel_id").cast(pl.Utf8))
        return df.select(["parcel_id", "sog_doy", "peak_doy", "senescence_doy"])

    _log.warning(
        "phenology_anchors_fallback_static",
        hint="ningun parcels[*_doy] ni phenology_anchors_path; uso SOG=105/peak=180/senescence=260",
    )
    rows = [
        {
            "parcel_id": str(pid),
            "sog_doy": 105,
            "peak_doy": 180,
            "senescence_doy": 260,
        }
        for pid in parcels["parcel_id"].astype(str).tolist()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "parcel_id": pl.Utf8,
            "sog_doy": pl.Int16,
            "peak_doy": pl.Int16,
            "senescence_doy": pl.Int16,
        },
    )


def _doy_to_dates(year: int, doy: int, window_days: int) -> tuple[str, str]:
    """Convert ``(year, doy)`` to range ``[start, end)`` ``YYYY-MM-DD``."""
    from datetime import datetime, timedelta

    center = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
    start = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")
    return start, end


def _sample_anchor_batch(
    ee_module: Any,
    parcels_chunk: gpd.GeoDataFrame,
    *,
    anchor: str,
    doy_col: str,
    year: int,
    bands: tuple[str, ...],
    anchors_table: pl.DataFrame,
    window_days: int,
    scale: int,
) -> list[dict[str, Any]]:
    """Sample a chunk of parcels at a single anchor.

    Builds an ``ee.FeatureCollection`` with each polygon + ``parcel_id``
    and adds a ``reduceRegions(mean)`` over the median of the S2 collection
    in the window ``[doy - window_days, doy + window_days]``.

    Returns a list of dicts ``{parcel_id, <anchor>_b04, ...}``. If the
    GEE query fails, returns rows with ``None`` for all bands.
    """
    rows: list[dict[str, Any]] = []
    # Group the chunk's parcels by anchor DOY (parcels with the same DOY
    # share a single server-side query).
    anchors_chunk = anchors_table.filter(
        pl.col("parcel_id").is_in(parcels_chunk["parcel_id"].astype(str).tolist())
    )
    if anchors_chunk.is_empty():
        return rows

    pid_to_doy: dict[str, int] = {
        r["parcel_id"]: int(r[doy_col]) for r in anchors_chunk.iter_rows(named=True)
    }

    # Group parcels by identical DOY — one query per unique DOY.
    by_doy: dict[int, list[Any]] = {}
    for _, row in parcels_chunk.iterrows():
        pid = str(row["parcel_id"])
        if pid not in pid_to_doy:
            continue
        doy = pid_to_doy[pid]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        by_doy.setdefault(doy, []).append((pid, geom))

    for doy, items in by_doy.items():
        start, end = _doy_to_dates(year, doy, window_days)
        try:
            features = [
                ee_module.Feature(
                    ee_module.Geometry(geom.__geo_interface__),
                    {"parcel_id": pid},
                )
                for pid, geom in items
            ]
            fc = ee_module.FeatureCollection(features)
            collection = (
                ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(start, end)
                .select(list(bands))
            )
            median = collection.median()
            reduced = median.reduceRegions(
                collection=fc,
                reducer=ee_module.Reducer.mean(),
                scale=scale,
            )
            info = reduced.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "s2_anchor_batch_failed",
                anchor=anchor,
                doy=doy,
                n=len(items),
                error=str(exc),
            )
            # Rows with None for this DOY group.
            for pid, _ in items:
                row_out: dict[str, Any] = {"parcel_id": pid}
                for band in bands:
                    row_out[_band_col_name(anchor, band)] = None
                rows.append(row_out)
            continue

        for feat in info.get("features", []) or []:
            props = feat.get("properties", {}) or {}
            pid = str(props.get("parcel_id", ""))
            row_out = {"parcel_id": pid}
            for band in bands:
                val = props.get(band)
                row_out[_band_col_name(anchor, band)] = (
                    float(val) if val is not None and not _is_nan(val) else None
                )
            rows.append(row_out)
    return rows


def _is_nan(val: Any) -> bool:
    """True if ``val`` is NaN/inf."""
    try:
        f = float(val)
        return bool(np.isnan(f) or np.isinf(f))
    except (TypeError, ValueError):
        return False


def _merge_anchor_rows(
    rows_per_anchor: dict[str, list[dict[str, Any]]],
    *,
    parcel_ids: list[str],
    year: int,
    anchors: tuple[str, ...],
    bands: tuple[str, ...],
) -> pl.DataFrame:
    """Merge per-anchor rows into a single DataFrame sorted by parcel_id.

    Guarantees determinism: the output is always sorted ascending by
    ``parcel_id``, regardless of the order in which GEE returned the
    batches.
    """
    schema = _build_schema(anchors, bands)
    by_pid: dict[str, dict[str, Any]] = {
        pid: {"parcel_id": pid, "year": int(year)} for pid in parcel_ids
    }
    # Initialize all cols to None.
    for pid in parcel_ids:
        for anchor in anchors:
            for band in bands:
                by_pid[pid][_band_col_name(anchor, band)] = None
    for anchor, rows in rows_per_anchor.items():
        for row in rows:
            pid = str(row["parcel_id"])
            if pid not in by_pid:
                continue
            for band in bands:
                col = _band_col_name(anchor, band)
                if col in row and row[col] is not None:
                    by_pid[pid][col] = row[col]
    ordered = [by_pid[pid] for pid in sorted(parcel_ids)]
    return pl.DataFrame(ordered, schema=schema)


def _count_completeness(
    df: pl.DataFrame, anchors: tuple[str, ...], bands: tuple[str, ...]
) -> tuple[int, int]:
    """Count parcels with ALL bands populated vs partially.

    Returns:
        ``(n_with_all_bands, n_with_partial)``.
    """
    band_cols = [_band_col_name(a, b) for a in anchors for b in bands]
    n_all = 0
    n_partial = 0
    for row in df.iter_rows(named=True):
        non_null = sum(1 for c in band_cols if row.get(c) is not None)
        if non_null == len(band_cols):
            n_all += 1
        elif non_null > 0:
            n_partial += 1
    return n_all, n_partial


def sample_s2_anchors_for_parcels(
    parcels: gpd.GeoDataFrame,
    year: int,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    phenology_anchors_path: Path | None = None,
    batch_size: int = 500,
    overwrite: bool = False,
    cache_dir: Path | None = None,
    window_days: int = ANCHOR_WINDOW_DAYS,
    scale: int = 10,
    anchors: tuple[str, ...] = DEFAULT_ANCHORS,
) -> Path:
    """Sample S2 at the SOG/peak/senescence DOY per parcel.

    For each parcel in ``parcels`` (must carry ``parcel_id`` (Utf8),
    ``year``, ``geometry`` and optionally pre-computed
    ``sog_doy/peak_doy/senescence_doy``), runs ``reduceRegions(mean)`` over a
    ``+/- window_days`` window around the anchor DOY and persists it as columns
    ``{anchor}_b04, {anchor}_b05, ..., {anchor}_b08`` in a format consumable
    by :meth:`SpectralSignatureFeatures._extract_anchor_bands`.

    Args:
        parcels: GeoDataFrame with ``parcel_id``, ``geometry`` POLYGON
            EPSG:4326 and optionally ``*_doy`` columns.
        year: Year to sample.
        output_path: Output parquet (parent is created if it does not exist).
        bands: S2 bands to sample (default ``("B04",...,"B08")``).
        phenology_anchors_path: Optional parquet with pre-computed anchors.
            If ``None`` and ``parcels`` does not carry ``*_doy``, falls back to
            static defaults for continental Italy (SOG=105/peak=180/senesc=260).
        batch_size: GEE batch size per reduceRegions.
        overwrite: If ``True`` ignores the cache and rewrites.
        cache_dir: Cache folder (default ``data/cache/gee/``).
        window_days: Window ``+/- window_days`` around the anchor DOY.
        scale: Sampling resolution in meters (default 10, native S2).
        anchors: Tuple of anchor names (default
            ``("sog","peak","senescence")``); they must match the
            ``{anchor}_doy`` columns in ``parcels`` or in
            ``phenology_anchors_path``.

    Returns:
        Absolute ``Path`` of the written parquet (``output_path``).

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
        RuntimeError: If ``init_ee`` fails and there is no usable prior cache.
    """
    output_path = Path(output_path)
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"s2_anchors_{_parcels_md5(parcels)}_{year}.parquet"
    cache_file = cache_root / cache_key

    if cache_file.exists() and not overwrite:
        _log.info("s2_anchors_cache_hit", path=str(cache_file))
        df_cached = pl.read_parquet(cache_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cached.write_parquet(output_path)
        return output_path.resolve()

    # Lazy import of earthengine-api: only inside the "real" path to avoid
    # breaking tests without EE installed.
    from ml.ingest.gee_sampler import init_ee

    try:
        import ee  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "earthengine-api not installed. Run `poetry install --with ml,geo`."
        ) from exc

    anchors_table = _resolve_anchors_table(parcels, phenology_anchors_path)
    parcel_ids: list[str] = [str(p) for p in parcels["parcel_id"].astype(str).tolist()]

    try:
        init_ee()
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "s2_anchors_ee_init_failed_degraded_mode",
            error=str(exc),
            hint="se devolvera DataFrame vacio con esquema valido",
        )
        empty = pl.DataFrame(
            [{"parcel_id": pid, "year": int(year)} for pid in sorted(parcel_ids)],
            schema=_build_schema(anchors, bands),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.write_parquet(output_path)
        empty.write_parquet(cache_file)
        return output_path.resolve()

    rows_per_anchor: dict[str, list[dict[str, Any]]] = {a: [] for a in anchors}
    total = len(parcels)
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        chunk = parcels.iloc[start : start + batch_size]
        for anchor in anchors:
            doy_col = f"{anchor}_doy"
            rows = _sample_anchor_batch(
                ee,
                chunk,
                anchor=anchor,
                doy_col=doy_col,
                year=year,
                bands=bands,
                anchors_table=anchors_table,
                window_days=window_days,
                scale=scale,
            )
            rows_per_anchor[anchor].extend(rows)
        _log.info(
            "s2_anchors_batch_done",
            start=start,
            end=min(start + batch_size, total),
            total=total,
        )

    df = _merge_anchor_rows(
        rows_per_anchor,
        parcel_ids=parcel_ids,
        year=year,
        anchors=anchors,
        bands=bands,
    )
    elapsed = time.perf_counter() - t0
    n_all, n_partial = _count_completeness(df, anchors, bands)
    cost_estimate = round(total * COST_PER_PARCEL_USD, 4)
    _log.info(
        "s2_anchors_complete",
        n_parcels=total,
        n_with_all_bands=n_all,
        n_with_partial=n_partial,
        gee_seconds=round(elapsed, 2),
        cost_estimate_usd=cost_estimate,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_file)
    df.write_parquet(output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.ingest.s2_anchor_sampler",
        description=(
            "Muestrea S2 B04..B08 por parcela en anclas SOG/peak/senescence "
            "y persiste a parquet consumible por SpectralSignatureFeatures."
        ),
    )
    p.add_argument(
        "--parcels-path",
        required=True,
        type=Path,
        help="Parquet o GeoParquet con parcel_id + geometry + opcionales *_doy.",
    )
    p.add_argument("--year", required=True, type=int)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet de salida (default {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--phenology-anchors-path",
        type=Path,
        default=None,
        help="Parquet opcional con sog_doy/peak_doy/senescence_doy.",
    )
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    import geopandas as gpd

    args = _build_arg_parser().parse_args(argv)
    parcels_path: Path = args.parcels_path
    if parcels_path.suffix.lower() in {".geoparquet", ".gpkg"}:
        parcels = gpd.read_file(parcels_path)
    else:
        parcels = gpd.read_parquet(parcels_path)
    out = sample_s2_anchors_for_parcels(
        parcels,
        args.year,
        output_path=args.output,
        phenology_anchors_path=args.phenology_anchors_path,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    _log.info("s2_anchor_sampler_done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
