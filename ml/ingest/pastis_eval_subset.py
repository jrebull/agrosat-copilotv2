"""Build a REAL stratified subset of PASTIS-R for FarSLIP/RemoteCLIP evaluation.

This module is the single source of truth for the fixture
``data/test_fixtures/pastis_eval_subset.parquet`` consumed by
``notebooks/baseline/04_farslip_eval_pastis.ipynb`` and by the smoke tests
of the FarSLIP / RemoteCLIP encoders in EPIC 4.

Rules:
    - NEVER generates synthetic data. If PASTIS-R is not present on disk
      (``data/PASTIS-R/metadata.geojson`` and ``DATA_S2/``), a
      ``FileNotFoundError`` is raised with the download instruction (DVC pull or
      link to the official INRAE dataset).
    - Full determinism: ``seed=42`` by default -> the parquet MD5 must be
      stable run-to-run.
    - Polars 1.x for parquet I/O, no pandas.
    - Structured logs via ``structlog``.

CLI:
    poetry run python -m ml.ingest.pastis_eval_subset \\
        --output data/test_fixtures/pastis_eval_subset.parquet \\
        --n-samples 1024 --seed 42 --stratify-by class
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

from ml.ingest.pastis_loader import (
    PASTIS_R_CLASSES,
    PASTIS_S2_BANDS,
    pastis_patch_coords,
    pastis_patch_index,
)

_log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "test_fixtures" / "pastis_eval_subset.parquet"

_VOID_CLASS = 19
_BACKGROUND_CLASS = 0
_VALID_CLASS_RANGE = range(1, 19)  # 1..18 inclusive


_StratifyBy = Literal["class", "tile", "fold"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _raise_missing_pastis(pastis_root: Path) -> None:
    """Raise FileNotFoundError with download instruction.

    Args:
        pastis_root: Expected root of the PASTIS-R dataset.

    Raises:
        FileNotFoundError: Always. Includes the missing path and the download
            commands (DVC pull) or link to the official dataset.
    """
    msg = (
        f"PASTIS-R not found at {pastis_root}. "
        "Expected: metadata.geojson + DATA_S2/ + ANNOTATIONS/. "
        "To obtain it: `dvc pull data/PASTIS-R.dvc` "
        "or manual download from "
        "https://zenodo.org/record/5735646 (PASTIS-R, INRAE, CC-BY-SA-4.0)."
    )
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pastis_present(pastis_root: Path) -> None:
    """Validate that the minimal PASTIS-R structure exists on disk.

    Args:
        pastis_root: Expected root of the dataset.

    Raises:
        FileNotFoundError: If ``metadata.geojson`` or ``DATA_S2/`` is missing.
    """
    metadata = pastis_root / "metadata.geojson"
    data_s2 = pastis_root / "DATA_S2"
    if not metadata.exists() or not data_s2.exists() or not data_s2.is_dir():
        _raise_missing_pastis(pastis_root)


def _patch_majority_class(
    semantic: np.ndarray,
    exclude: tuple[int, ...] = (_BACKGROUND_CLASS, _VOID_CLASS),
) -> int:
    """Return the majority class (1..18) of a patch.

    Args:
        semantic: 2D class map ``(H, W)``.
        exclude: Classes to exclude from the count.

    Returns:
        int in 1..18 with the majority class, or 0 if everything is background/void.
    """
    flat = semantic.ravel()
    mask = ~np.isin(flat, np.asarray(exclude, dtype=flat.dtype))
    filtered = flat[mask]
    if filtered.size == 0:
        return 0
    vals, counts = np.unique(filtered, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _load_target(pastis_root: Path, patch_id: str) -> np.ndarray | None:
    """Load ``TARGET_<patch_id>.npy`` or ``None`` if it does not exist.

    Args:
        pastis_root: Root of the dataset.
        patch_id: Patch identifier.

    Returns:
        ndarray of shape ``(3, H, W)`` or ``None`` if it does not exist.
    """
    tgt = pastis_root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
    if not tgt.exists():
        return None
    return np.load(tgt)


def _load_s2(pastis_root: Path, patch_id: str) -> np.ndarray | None:
    """Load ``S2_<patch_id>.npy`` or ``None`` if it does not exist.

    Args:
        pastis_root: Root of the dataset.
        patch_id: Patch identifier.

    Returns:
        ndarray ``(T, 10, H, W)`` or ``None``.
    """
    s2 = pastis_root / "DATA_S2" / f"S2_{patch_id}.npy"
    if not s2.exists():
        return None
    return np.load(s2)


def _enumerate_parcels(
    pastis_root: Path,
    patch_ids: list[str],
) -> pl.DataFrame:
    """Enumerate all parcels ``(patch_id, instance_id, class_id, n_pixels)``.

    A parcel = unique (patch_id, instance_id) derived from channel 1
    (instance) of ``TARGET_<patch_id>.npy``. The class is taken from the mode
    of channel 0 (semantic) restricted to the pixels of that instance.

    Args:
        pastis_root: PASTIS-R root.
        patch_ids: List of patch_ids to scan.

    Returns:
        DataFrame with columns ``patch_id`` (Utf8), ``instance_id`` (Int64),
        ``class_id`` (Int64), ``n_pixels`` (Int64).
    """
    rows: list[dict[str, Any]] = []
    for pid in patch_ids:
        target = _load_target(pastis_root, pid)
        if target is None or target.ndim != 3 or target.shape[0] < 2:
            continue
        semantic = target[0]
        instance = target[1]
        inst_ids = np.unique(instance)
        for iid in inst_ids:
            iid_int = int(iid)
            if iid_int == 0:
                # 0 = no instance (background)
                continue
            mask = instance == iid
            n_pixels = int(np.count_nonzero(mask))
            if n_pixels == 0:
                continue
            cls_pixels = semantic[mask]
            cls_pixels = cls_pixels[
                ~np.isin(cls_pixels, np.asarray([_BACKGROUND_CLASS, _VOID_CLASS]))
            ]
            if cls_pixels.size == 0:
                continue
            vals, counts = np.unique(cls_pixels, return_counts=True)
            class_id = int(vals[int(np.argmax(counts))])
            if class_id not in _VALID_CLASS_RANGE:
                continue
            rows.append(
                {
                    "patch_id": pid,
                    "instance_id": iid_int,
                    "class_id": class_id,
                    "n_pixels": n_pixels,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "patch_id": pl.Utf8,
                "instance_id": pl.Int64,
                "class_id": pl.Int64,
                "n_pixels": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def _stratified_sample(
    parcels: pl.DataFrame,
    n_samples: int,
    stratify_by: _StratifyBy,
    seed: int,
) -> pl.DataFrame:
    """Sample ``n_samples`` parcels stratified by the indicated dimension.

    Guarantee (only for ``stratify_by='class'``): each class available in
    ``parcels`` receives at least ``max(8, n_samples // 36)`` samples (or all
    available if there are fewer).

    Args:
        parcels: DataFrame with columns ``patch_id, instance_id, class_id,
            n_pixels, tile, fold``.
        n_samples: Target size of the subset.
        stratify_by: Stratification dimension (``class``, ``tile``, ``fold``).
        seed: numpy seed for reproducibility.

    Returns:
        Sampled DataFrame with size <= ``n_samples`` (may be smaller if
        there are not enough parcels in the catalog).
    """
    rng = np.random.default_rng(seed)
    col_map = {"class": "class_id", "tile": "tile", "fold": "fold"}
    strat_col = col_map[stratify_by]

    if parcels.is_empty():
        return parcels

    groups = parcels.group_by(strat_col).agg(pl.len().alias("_count"))
    n_groups = groups.height
    per_group_floor = max(1, n_samples // max(n_groups, 1))
    class_min = max(8, n_samples // 36) if stratify_by == "class" else per_group_floor

    selected_indices: list[int] = []
    parcels_with_idx = parcels.with_row_index(name="_row_idx")

    # Pass 1: guarantee a minimum per group
    for grp_val in groups[strat_col].to_list():
        sub = parcels_with_idx.filter(pl.col(strat_col) == grp_val)
        sub_idx = sub["_row_idx"].to_list()
        target = min(class_min, len(sub_idx))
        choice = rng.choice(len(sub_idx), size=target, replace=False)
        selected_indices.extend(int(sub_idx[i]) for i in choice)

    # Pass 2: fill up to n_samples with the remainder distributed proportionally
    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        already = set(selected_indices)
        pool = parcels_with_idx.filter(~pl.col("_row_idx").is_in(list(already)))
        if not pool.is_empty():
            extra = min(remaining, pool.height)
            pool_idx = pool["_row_idx"].to_list()
            choice = rng.choice(len(pool_idx), size=extra, replace=False)
            selected_indices.extend(int(pool_idx[i]) for i in choice)
    elif remaining < 0:
        # Edge case: pass 1 already exceeded n_samples (n_groups * class_min > n_samples).
        # We truncate keeping at least 1 per present group.
        kept: dict[Any, list[int]] = defaultdict(list)
        for idx in selected_indices:
            grp = parcels.row(idx, named=True)[strat_col]
            kept[grp].append(idx)
        # Round-robin until n_samples is filled
        new_selection: list[int] = []
        cursors = {k: 0 for k in kept}
        while len(new_selection) < n_samples:
            progressed = False
            for k in list(kept.keys()):
                c = cursors[k]
                if c < len(kept[k]):
                    new_selection.append(kept[k][c])
                    cursors[k] = c + 1
                    progressed = True
                    if len(new_selection) >= n_samples:
                        break
            if not progressed:
                break
        selected_indices = new_selection

    selected_indices = sorted(set(selected_indices))
    return parcels_with_idx.filter(pl.col("_row_idx").is_in(selected_indices)).drop("_row_idx")


def _build_imagery_blob(
    pastis_root: Path,
    subset: pl.DataFrame,
) -> pl.DataFrame:
    """Serialize multitemporal S2 crops ONLY of the pixels of each instance.

    For each parcel in ``subset``, loads the corresponding ``S2_<patch_id>.npy``
    and ``TARGET_<patch_id>.npy``, computes the instance mask and emits a
    long-format row ``(parcel_id, t_index, band_NN)`` with the mean of the
    pixels of that instance in that band and timestep.

    Averaging within the instance keeps the parquet bounded
    (N parcels * T * 10 bands), sufficient for a FarSLIP/RemoteCLIP eval
    in a notebook.

    Args:
        pastis_root: PASTIS-R root.
        subset: DataFrame with columns ``parcel_id``, ``patch_id``,
            ``instance_id``.

    Returns:
        DataFrame with columns ``parcel_id, t_index`` + ``band_B02..band_B12``
        (10 bands, Float32). Empty if no patch could be read.
    """
    s2_cache: dict[str, np.ndarray] = {}
    target_cache: dict[str, np.ndarray] = {}

    rows: list[dict[str, Any]] = []
    for record in subset.iter_rows(named=True):
        pid = record["patch_id"]
        iid = int(record["instance_id"])
        parcel_id = record["parcel_id"]

        if pid not in s2_cache:
            s2_arr = _load_s2(pastis_root, pid)
            if s2_arr is None:
                continue
            s2_cache[pid] = s2_arr
        if pid not in target_cache:
            tgt = _load_target(pastis_root, pid)
            if tgt is None or tgt.ndim != 3 or tgt.shape[0] < 2:
                continue
            target_cache[pid] = tgt

        s2 = s2_cache[pid]
        instance = target_cache[pid][1]
        mask = instance == iid
        if not np.any(mask):
            continue

        T = s2.shape[0]
        for t in range(T):
            row: dict[str, Any] = {"parcel_id": parcel_id, "t_index": t}
            for b_idx, band_name in enumerate(PASTIS_S2_BANDS):
                vals = s2[t, b_idx][mask]
                row[f"band_{band_name}"] = float(vals.mean()) if vals.size else float("nan")
            rows.append(row)

    schema: dict[str, Any] = {"parcel_id": pl.Utf8, "t_index": pl.Int64}
    for band in PASTIS_S2_BANDS:
        schema[f"band_{band}"] = pl.Float32
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _md5_file(path: Path) -> str:
    """Return the hex MD5 of a file.

    Args:
        path: Path to the file.

    Returns:
        MD5 hash in hex.
    """
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pastis_eval_subset(
    output_path: Path | str = _DEFAULT_OUTPUT,
    *,
    n_samples: int = 1024,
    seed: int = 42,
    pastis_root: Path | None = None,
    overwrite: bool = False,
    stratify_by: _StratifyBy = "class",
    save_imagery: bool = True,
) -> Path:
    """Build a REAL stratified subset of PASTIS-R for FarSLIP/RemoteCLIP evaluation.

    Does NOT generate synthetic data. If PASTIS-R is not on disk, raises
    ``FileNotFoundError`` with download instruction.

    The subset is materialized as parquet with one row per parcel
    ``(patch_id, instance_id)`` and the columns:

    - ``parcel_id`` (Utf8, schema ``{patch_id}_{instance_id}``)
    - ``patch_id`` (Int64)
    - ``instance_id`` (Int64)
    - ``class_id`` (Int64, 1..18)
    - ``class_name`` (Utf8, via ``PASTIS_R_CLASSES``)
    - ``tile`` (Utf8)
    - ``fold`` (Int64, 1..5)
    - ``lon`` / ``lat`` (Float64, EPSG:4326, patch centroid)
    - ``n_pixels`` (Int64, instance size)

    If ``save_imagery=True``, also materializes
    ``<output_path>.imagery.parquet`` with the S2 crops averaged per
    instance (rows ``parcel_id, t_index, band_B02..band_B12``).

    Args:
        output_path: Destination path of the main parquet.
        n_samples: Target number of parcels. Default 1024.
        seed: numpy seed for reproducibility. Default 42.
        pastis_root: Root of the dataset. Default ``data/PASTIS-R/``.
        overwrite: If False and the file already exists, does not regenerate.
        stratify_by: Stratification dimension (``class``, ``tile``, ``fold``).
        save_imagery: If True, materializes the auxiliary imagery blob.

    Returns:
        Path to the generated main parquet (or existing one if ``overwrite=False``).

    Raises:
        FileNotFoundError: If PASTIS-R is not present on disk.
        ValueError: If after enumerating instances there are no valid parcels.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not overwrite:
        _log.info(
            "pastis_eval_subset.skip_existing",
            output=str(output),
            md5=_md5_file(output),
        )
        return output

    root = Path(pastis_root) if pastis_root is not None else _DEFAULT_PASTIS_ROOT
    _validate_pastis_present(root)

    index_df = pastis_patch_index(root / "metadata.geojson")
    if index_df.is_empty():
        raise ValueError(f"PASTIS-R metadata.geojson has no features: {root}")

    coords_df = pastis_patch_coords(root / "metadata.geojson", target_crs="EPSG:4326")

    patch_ids = index_df["patch_id"].to_list()
    parcels_raw = _enumerate_parcels(root, patch_ids)
    if parcels_raw.is_empty():
        raise ValueError(f"No valid instances found in {root / 'ANNOTATIONS'}.")

    # Enrich with tile, fold, lon, lat
    parcels_enriched = parcels_raw.join(
        index_df.rename({"TILE": "tile", "Fold": "fold"}),
        on="patch_id",
        how="left",
    )
    if not coords_df.is_empty():
        parcels_enriched = parcels_enriched.join(
            coords_df.select(["patch_id", "lon", "lat"]),
            on="patch_id",
            how="left",
        )
    else:
        parcels_enriched = parcels_enriched.with_columns(
            pl.lit(0.0).alias("lon"),
            pl.lit(0.0).alias("lat"),
        )

    sampled = _stratified_sample(
        parcels_enriched, n_samples=n_samples, stratify_by=stratify_by, seed=seed
    )

    # Build the canonical Utf8 parcel_id + class_name
    class_name_map = {int(k): v for k, v in PASTIS_R_CLASSES.items()}
    sampled = sampled.with_columns(
        (
            pl.col("patch_id").cast(pl.Utf8) + pl.lit("_") + pl.col("instance_id").cast(pl.Utf8)
        ).alias("parcel_id"),
        pl.col("class_id")
        .cast(pl.Int64)
        .replace_strict(class_name_map, default="unknown")
        .alias("class_name"),
    )

    final = sampled.select(
        [
            pl.col("parcel_id").cast(pl.Utf8),
            pl.col("patch_id").cast(pl.Int64),
            pl.col("instance_id").cast(pl.Int64),
            pl.col("class_id").cast(pl.Int64),
            pl.col("class_name").cast(pl.Utf8),
            pl.col("tile").cast(pl.Utf8),
            pl.col("fold").cast(pl.Int64),
            pl.col("lon").cast(pl.Float64),
            pl.col("lat").cast(pl.Float64),
            pl.col("n_pixels").cast(pl.Int64),
        ]
    ).sort(["class_id", "patch_id", "instance_id"])

    final.write_parquet(output, compression="zstd")

    if save_imagery:
        imagery_path = output.with_suffix(output.suffix + ".imagery.parquet")
        imagery_df = _build_imagery_blob(root, final)
        imagery_df.write_parquet(imagery_path, compression="zstd")
        imagery_meta = {"path": str(imagery_path), "rows": imagery_df.height}
    else:
        imagery_meta = {"path": None, "rows": 0}

    class_counts: dict[int, int] = dict(Counter(final["class_id"].to_list()))
    _log.info(
        "pastis_eval_subset.built",
        output=str(output),
        n_parcels=final.height,
        n_parcels_per_class=class_counts,
        n_unique_tiles=final["tile"].n_unique(),
        n_unique_patches=final["patch_id"].n_unique(),
        md5=_md5_file(output),
        imagery=imagery_meta,
        stratify_by=stratify_by,
        seed=seed,
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    """Build the CLI ArgumentParser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.ingest.pastis_eval_subset",
        description=(
            "Genera el subset REAL de PASTIS-R consumido por el notebook de "
            "evaluacion FarSLIP/RemoteCLIP (US-023). NO sintetico."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Ruta destino del parquet (default: data/test_fixtures/pastis_eval_subset.parquet).",
    )
    parser.add_argument("--n-samples", type=int, default=1024, help="Numero objetivo de parcelas.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla numpy.")
    parser.add_argument(
        "--stratify-by",
        choices=("class", "tile", "fold"),
        default="class",
        help="Dimension de estratificacion.",
    )
    parser.add_argument(
        "--pastis-root",
        type=Path,
        default=None,
        help="Raiz PASTIS-R (default: data/PASTIS-R/).",
    )
    parser.add_argument(
        "--no-imagery",
        action="store_true",
        help="No materializar el blob auxiliar <output>.imagery.parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir el parquet si ya existe.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional arguments (for tests). If None, reads from sys.argv.

    Returns:
        0 if the subset was generated correctly; error code otherwise.
    """
    args = _build_cli().parse_args(argv)
    out = build_pastis_eval_subset(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        pastis_root=args.pastis_root,
        overwrite=args.overwrite,
        stratify_by=args.stratify_by,
        save_imagery=not args.no_imagery,
    )
    print(json.dumps({"output": str(out), "md5": _md5_file(out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_pastis_eval_subset", "main"]
