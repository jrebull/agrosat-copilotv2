"""Typer CLI to generate the PASTIS-R subset used by US-018.

Builds a parquet ``data/test_fixtures/feature_selection_subset.parquet`` with
``>= min_per_class * n_classes`` samples x 187 columns (153 stats + 24 FFT +
8 phenology + parcel_id + year + class_id + fold). The samples are stratified by
class using the dominant label of each PASTIS-R patch and are labeled with the
official fold (1..5) read from ``metadata.geojson``.

Usage::

    poetry run python scripts/generate_feature_selection_subset.py \\
        --root data/PASTIS-R \\
        --out data/test_fixtures/feature_selection_subset.parquet \\
        --min-per-class 10 \\
        --max-samples 500

If ``data/PASTIS-R/`` does not exist (laptops without download), the script exits
with code 0 and a structured warning: it does NOT fail CI nor break dev
environments without heavy data.

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import typer
import xarray as xr

from ml.features.spectral_indices import compute_index
from ml.features.temporal_features import DEFAULT_INDICES, extract_temporal_features
from ml.ingest.pastis_loader import (
    PASTIS_CLASS_MAP,
    PASTIS_S2_BANDS,
    iter_pastis_patches,
    pastis_patch_index,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _patch_dominant_class(semantic: np.ndarray) -> int:
    """Return the majority class of the patch (excluding background and void)."""
    flat = semantic.ravel().astype(np.int64)
    mask = (flat != 0) & (flat != 19)
    if mask.sum() == 0:
        return 0
    values, counts = np.unique(flat[mask], return_counts=True)
    return int(values[np.argmax(counts)])


def _patch_to_dataarray(s2_patch: np.ndarray, dates: list[int]) -> xr.DataArray:
    """Convert a patch ``(T, 10, H, W)`` to an aggregated ``(time, band)`` DataArray.

    Spatially averages the patch to obtain a temporal series per band; the caller
    adds the spectral indices as additional bands.
    """
    # Spatial mean over H, W -> (T, 10).
    spatial_mean = s2_patch.mean(axis=(2, 3)).astype(np.float32) / 10_000.0
    times = np.array(
        [np.datetime64(f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}", "ns") for d in dates],
        dtype="datetime64[ns]",
    )
    return xr.DataArray(
        spatial_mean,
        dims=("time", "band"),
        coords={"time": times, "band": PASTIS_S2_BANDS},
    )


def _enrich_with_indices(s2_da: xr.DataArray, indices: tuple[str, ...]) -> xr.DataArray:
    """Add spectral index columns to the Sentinel-2 bands DataArray.

    Args:
        s2_da: DataArray ``(time, band)`` with 10 Sentinel-2 bands.
        indices: Canonical indices to compute.

    Returns:
        DataArray ``(time, band)`` with original bands + stacked indices.
    """
    # Reshape to (time, band, y, x) required by compute_index (it needs
    # spatial dims even if they are 1x1).
    arr = s2_da.expand_dims(y=1, x=1)
    new_bands: list[np.ndarray] = []
    new_names: list[str] = []
    for idx in indices:
        try:
            result = compute_index(arr, idx)
            # result shape (time, 1, 1) -> squeeze to (time,)
            vals = np.asarray(result.values).reshape(-1)
            new_bands.append(vals)
            new_names.append(idx)
        except (KeyError, ValueError) as exc:
            logger.warning("index_compute_failed", index=idx, error=str(exc))
            continue
    if not new_bands:
        return s2_da
    stack = np.stack(new_bands, axis=1)
    return xr.DataArray(
        stack,
        dims=("time", "band"),
        coords={"time": s2_da.coords["time"], "band": new_names},
    )


@app.command()
def main(
    root: Path = typer.Option(
        Path("data/PASTIS-R"),
        "--root",
        help="Raiz del dataset PASTIS-R descargado",
    ),
    out: Path = typer.Option(
        Path("data/test_fixtures/feature_selection_subset.parquet"),
        "--out",
        help="Parquet de salida",
    ),
    min_per_class: int = typer.Option(
        10,
        "--min-per-class",
        help="Muestras minimas por clase para estratificacion",
    ),
    max_samples: int = typer.Option(
        500,
        "--max-samples",
        help="Maximo total de muestras a producir",
    ),
    seed: int = typer.Option(42, "--seed", help="Semilla para muestreo"),
) -> None:
    """Generate the class-stratified PASTIS-R subset for US-018."""
    if not root.exists():
        logger.warning(
            "pastis_root_missing",
            root=str(root),
            note="Saltando generacion del subset; modo degradado.",
        )
        raise typer.Exit(code=0)

    logger.info("subset_generation_started", root=str(root), out=str(out))

    metadata_path = root / "metadata.geojson"
    if not metadata_path.exists():
        logger.error("pastis_metadata_missing", path=str(metadata_path))
        raise typer.Exit(code=2)

    index_df = pastis_patch_index(metadata_path)
    if index_df.is_empty():
        logger.error("pastis_index_empty")
        raise typer.Exit(code=2)

    rng = np.random.default_rng(seed)

    # Broad initial sampling: we take random patches and filter until
    # filling the per-class quota.
    patch_ids = index_df.get_column("patch_id").to_list()
    fold_map = dict(
        zip(
            index_df.get_column("patch_id").to_list(),
            index_df.get_column("Fold").to_list(),
            strict=True,
        )
    )
    rng.shuffle(patch_ids)

    rows: list[dict[str, object]] = []
    per_class_counter: Counter[int] = Counter()
    max_per_class = max(min_per_class, max_samples // max(1, len(PASTIS_CLASS_MAP)))

    indices_to_compute = tuple(DEFAULT_INDICES)

    for patch in iter_pastis_patches(patch_ids, root=root, load_annotations=True):
        if len(rows) >= max_samples:
            break
        semantic = patch.get("semantic")
        if semantic is None:
            continue
        cls = _patch_dominant_class(semantic)
        if cls == 0 or cls == 19:
            continue
        if per_class_counter[cls] >= max_per_class:
            continue
        dates = patch.get("dates_s2") or []
        if len(dates) < 3:
            continue
        try:
            s2_da = _patch_to_dataarray(patch["s2"], dates)
            indices_da = _enrich_with_indices(s2_da, indices_to_compute)
            indices_da.attrs["parcel_id"] = int(patch["patch_id"])
            indices_da.attrs["year"] = int(str(dates[0])[:4])
            features_df = extract_temporal_features(indices_da, indices=indices_to_compute)
        except Exception as exc:  # noqa: BLE001
            logger.warning("patch_skipped", patch_id=patch["patch_id"], error=str(exc))
            continue
        row = features_df.row(0, named=True)
        row["class_id"] = cls
        row["fold"] = int(fold_map.get(str(patch["patch_id"]), 0))
        rows.append(row)
        per_class_counter[cls] += 1
        logger.info(
            "patch_processed",
            patch_id=patch["patch_id"],
            class_id=cls,
            class_name=PASTIS_CLASS_MAP.get(cls, "unknown"),
            n_so_far=len(rows),
        )

    if not rows:
        logger.error("no_patches_processed")
        raise typer.Exit(code=3)

    df = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    logger.info(
        "subset_generated",
        path=str(out),
        n_rows=df.height,
        n_cols=df.width,
        per_class=dict(per_class_counter),
    )


if __name__ == "__main__":
    app()
    sys.exit(0)
