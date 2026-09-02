"""Generate spectro-temporal features per individual parcel (not per dominant patch).

Parcel-level variant of `generate_feature_selection_subset.py`. For each PASTIS-R
patch it iterates over the parcels (instance_id) inside and aggregates the pixels
of each one via spatial mean, then computes spectral indices + temporal features.

Output:
    data/test_fixtures/feature_selection_parcels_subset.parquet

Columns: parcel_id (str "<patch>_<instance>"), patch_id, instance_id, year,
class_id, fold, n_pixels, area_m2, + 187 features (153 stats + 24 FFT + 8 phenology + 2 aux).

Filters:
- Discard instance_id=0 (patch background with no assigned parcel).
- Discard class_id=0 (Background PASTIS) and class_id=19 (Void).
- Discard parcels with n_pixels < --min-pixels (default 10).
- Discard patches with dates_s2 < 3 (insufficient time series).

Parallelization: joblib.Parallel(n_jobs) per patch (each patch is atomic).
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import typer
import xarray as xr
from joblib import Parallel, delayed

from ml.features.spectral_indices import compute_index
from ml.features.temporal_features import DEFAULT_INDICES, extract_temporal_features
from ml.ingest.pastis_loader import (
    PASTIS_S2_BANDS,
    iter_pastis_patches,
    pastis_patch_index,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _parcel_to_dataarray(s2_patch: np.ndarray, mask: np.ndarray, dates: list[int]) -> xr.DataArray:
    """Convert the masked pixels of a patch to a DataArray (time, band)."""
    # s2_patch shape (T, 10, H, W). mask shape (H, W).
    # Spatial mean over the masked pixels.
    masked_pixels = s2_patch[:, :, mask].mean(axis=2).astype(np.float32) / 10_000.0
    times = np.array(
        [np.datetime64(f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}", "ns") for d in dates],
        dtype="datetime64[ns]",
    )
    return xr.DataArray(
        masked_pixels,
        dims=("time", "band"),
        coords={"time": times, "band": PASTIS_S2_BANDS},
    )


def _enrich_with_indices(s2_da: xr.DataArray, indices: tuple[str, ...]) -> xr.DataArray:
    """Add spectral index bands to the DataArray of Sentinel-2 bands."""
    arr = s2_da.expand_dims(y=1, x=1)
    new_bands: list[np.ndarray] = []
    new_names: list[str] = []
    for idx in indices:
        try:
            result = compute_index(arr, idx)
            vals = np.asarray(result.values).reshape(-1)
            new_bands.append(vals)
            new_names.append(idx)
        except (KeyError, ValueError):
            continue
    if not new_bands:
        return s2_da
    stack = np.stack(new_bands, axis=1)
    return xr.DataArray(
        stack,
        dims=("time", "band"),
        coords={"time": s2_da.coords["time"], "band": new_names},
    )


def _process_patch(patch: dict, min_pixels: int, indices_to_compute: tuple[str, ...]) -> list[dict]:
    """Process an entire patch: iterate per parcel, aggregate + extract features.

    Returns a list of dicts (one per valid parcel).
    """
    semantic = patch.get("semantic")
    instance = patch.get("instance") if isinstance(patch, dict) else None
    if semantic is None or instance is None:
        return []
    dates = patch.get("dates_s2") or []
    if len(dates) < 3:
        return []
    patch_id = int(patch["patch_id"])
    fold = patch.get("fold")
    year = int(str(dates[0])[:4])

    s2 = patch["s2"]

    out: list[dict] = []
    unique_instances = np.unique(instance)
    for inst_id in unique_instances:
        iid = int(inst_id)
        if iid == 0:
            continue
        mask = instance == iid
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue
        # Dominant class in those pixels.
        sem_in_parcel = semantic[mask]
        if sem_in_parcel.size == 0:
            continue
        values, counts = np.unique(sem_in_parcel, return_counts=True)
        cls = int(values[np.argmax(counts)])
        if cls == 0 or cls == 19:
            continue

        try:
            s2_da = _parcel_to_dataarray(s2, mask, dates)
            indices_da = _enrich_with_indices(s2_da, indices_to_compute)
            indices_da.attrs["parcel_id"] = iid
            indices_da.attrs["year"] = year
            features_df = extract_temporal_features(indices_da, indices=indices_to_compute)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "parcel_skipped",
                patch_id=patch_id,
                instance_id=iid,
                error=str(exc),
            )
            continue

        row = features_df.row(0, named=True)
        # Overwrite parcel_id with the canonical string format.
        row["parcel_id"] = f"{patch_id}_{iid}"
        row["patch_id"] = patch_id
        row["instance_id"] = iid
        row["year"] = year
        row["class_id"] = cls
        row["fold"] = int(fold) if fold else 0
        row["n_pixels"] = n_pixels
        out.append(row)
    return out


def _process_patch_id(
    patch_id: str, root: Path, min_pixels: int, indices: tuple[str, ...]
) -> list[dict]:
    """Wrapper for joblib: read the patch and return rows."""
    for p in iter_pastis_patches([patch_id], root=root, load_annotations=True):
        return _process_patch(p, min_pixels=min_pixels, indices_to_compute=indices)
    return []


@app.command()
def main(
    root: Path = typer.Option(
        Path("data/PASTIS-R"),
        "--root",
        help="Raíz del dataset PASTIS-R",
    ),
    out: Path = typer.Option(
        Path("data/test_fixtures/feature_selection_parcels_subset.parquet"),
        "--out",
        help="Parquet de salida (~50 MB para 85k parcelas)",
    ),
    min_pixels: int = typer.Option(10, "--min-pixels", help="Mínimo de píxeles por parcela"),
    n_jobs: int = typer.Option(
        -1,
        "--n-jobs",
        help="Jobs paralelos joblib (-1 = todos los cores)",
    ),
    limit_patches: int = typer.Option(
        0,
        "--limit-patches",
        help="Si > 0, procesa solo los primeros N patches (smoke test)",
    ),
) -> None:
    """Generate spectral-temporal features per individual PASTIS-R parcel."""
    if not root.exists():
        logger.warning("pastis_root_missing", root=str(root))
        raise typer.Exit(code=0)

    metadata_path = root / "metadata.geojson"
    if not metadata_path.exists():
        logger.error("metadata_missing", path=str(metadata_path))
        raise typer.Exit(code=2)

    index_df = pastis_patch_index(metadata_path)
    if index_df.is_empty():
        logger.error("pastis_index_empty")
        raise typer.Exit(code=2)

    patch_ids = index_df.get_column("patch_id").to_list()
    if limit_patches > 0:
        patch_ids = patch_ids[:limit_patches]

    indices_to_compute = tuple(DEFAULT_INDICES)
    logger.info(
        "parcel_extraction_started",
        n_patches=len(patch_ids),
        min_pixels=min_pixels,
        n_jobs=n_jobs,
    )

    t0 = time.time()
    # Parallelize per patch. Each patch produces ~35 parcels on average.
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_process_patch_id)(pid, root, min_pixels, indices_to_compute) for pid in patch_ids
    )
    elapsed = time.time() - t0

    all_rows = [r for batch in results for r in batch]
    logger.info(
        "parcel_extraction_done",
        n_patches=len(patch_ids),
        n_parcels=len(all_rows),
        elapsed_min=round(elapsed / 60, 1),
        rate_parcels_per_s=round(len(all_rows) / max(elapsed, 0.01), 1),
    )

    if not all_rows:
        logger.error("no_parcels_extracted")
        raise typer.Exit(code=3)

    df = pl.DataFrame(all_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    per_class = Counter(r["class_id"] for r in all_rows)
    file_size_mb = out.stat().st_size / 1e6
    logger.info(
        "subset_parcels_generated",
        path=str(out),
        n_rows=df.height,
        n_cols=df.width,
        file_size_mb=round(file_size_mb, 1),
    )
    print("\n=== Stats ===")
    print(f"N parcelas: {df.height}")
    print(f"N features: {df.width}")
    print(f"Folds: {dict(Counter(r['fold'] for r in all_rows))}")
    print(f"Top 5 clases: {dict(sorted(per_class.items(), key=lambda kv: -kv[1])[:5])}")
    print(f"Output: {out} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    app()
