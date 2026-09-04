"""Per-pixel softmax / OOF dump from the six segmentation ``best.pt`` (US-031).

For each checkpoint in ``CHECKPOINT_REGISTRY`` (the same registry US-030 uses),
:func:`dump_oof` runs forward-only inference over the PASTIS held-out fold,
applies ``softmax`` (POST-softmax, never logits) and unifies every model into the
contiguous 18-class / 128 space IN PROBABILITY SPACE
(:func:`ml.eval.class_remap.remap_probs_20_to_18` for the 20-class models,
:func:`ml.eval.class_remap.resample_probs_128_bilinear` for SegFormer's 256 grid).
It persists one ``oof_{model}_fold{fold}.parquet`` per model (per-pixel softmax +
argmax pred), an optional per-parcel sidecar via
:func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs`, and a ``manifest.json``.

Anti-leakage invariants (plan Section 2):

- The persisted tensor is POST-softmax (sums to 1 over the class axis), never
  logits (US-040 averages probabilities, not logits).
- Only ``fold == 5`` is genuinely held out for a single ``best.pt`` per model, so
  every row carries ``held_out`` (True iff ``fold == 5``); folds 1-4 are NOT
  dumped by default.
- Normalization statistics use train folds (1, 2, 3) only, via
  :func:`ml.eval.dense_metrics._apply_train_norm` (reused as-is, no recompute).

Reuses the US-030 harness end-to-end (it does NOT reimplement inference):
``load_checkpoint_model`` / ``softmax_patch_for_kind`` /
``softmax_logits_segformer`` from :mod:`ml.eval.segmentation_inference`, the
fold-5 ``PASTISSegmentationDataset``, and ``_apply_train_norm``.

CLI: ``python -m ml.eval.oof.dump_oof [--fold 5] [--out-dir ...] [--max-patches N]``.
The CLI never downloads checkpoints; missing ones are skipped gracefully.

Project conventions: Polars I/O, ``structlog`` (never ``print``), type hints,
``canonical_parcel_id`` Utf8, no emojis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
import torch

from ml.eval.class_remap import (
    HARNESS_NUM_CLASSES,
    HARNESS_SIZE,
    remap_probs_20_to_18,
    resample_probs_128_bilinear,
)
from ml.eval.oof.parquet_io import write_softmax_parquet
from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.parcel_reconcile import (
    PROB_COLUMNS,
    load_pastis_parcel_ids,
    pixel_to_parcel_probs,
)

if TYPE_CHECKING:
    import polars as pl

    from ml.eval.checkpoint_registry import CheckpointSpec

logger = structlog.get_logger(__name__)

__all__ = ["dump_oof"]

#: Default output directory for the OOF parquet artifacts + manifest.
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent

#: Fold whose predictions are a genuine OOF held-out for a single best.pt/model.
_HELD_OUT_FOLD = 5

#: DVC path of the PASTIS-R dataset, used to tag ``data_version`` in the manifest.
_PASTIS_DVC_PATH = "data/PASTIS-R"

DumpDtype = Literal["float16", "float32"]


def _data_version() -> str:
    """Return the PASTIS-R DVC data version tag (``<path>@untracked`` if absent)."""
    return dvc_data_version(_PASTIS_DVC_PATH)


def _unify_to_18_at_128(probs_native: np.ndarray, spec: CheckpointSpec) -> np.ndarray:
    """Bring a native softmax to the contiguous 18-class / 128 space.

    Applies, in probability space (never on discrete maps):

    1. ``resample_probs_128_bilinear`` if the native map is not already 128 px
       (SegFormer is the only 256 px model -> ``spec.needs_resize``).
    2. ``remap_probs_20_to_18`` if the model is native 20-class (U-Net, U-TAE,
       AnySat, SegFormer); the native 18-class models pass through unchanged.

    Args:
        probs_native: Native-space softmax ``(C_native, H, W)`` (sum 1 per pixel).
        spec: Checkpoint descriptor (drives the resize / remap decisions).

    Returns:
        Unified softmax ``(18, 128, 128)`` float32, ``probs.sum(0) ~ 1``.
    """
    probs = np.asarray(probs_native, dtype=np.float32)
    # Resample to 128 in probability space first, so the 20->18 renorm operates
    # on the harness grid (SegFormer emits 20 channels at 256).
    if probs.shape[1:] != (HARNESS_SIZE, HARNESS_SIZE):
        probs = resample_probs_128_bilinear(probs)
    if spec.native_num_classes >= 20:
        probs = remap_probs_20_to_18(probs)
    return probs.astype(np.float32)


#: Model kinds whose dataset must keep the FULL Sentinel-2 time series (no
#: temporal collapse). These forwards consume ``(T, C, H, W)``; collapsing to a
#: single ``(C, H, W)`` frame breaks them (R-TEMPORAL). ``tsvit-pheno-fullm`` is
#: the US-039 Full-M retrain and MUST be here -- omitting it makes the dataset
#: deliver a 4-D tensor and the TSViT rearrange raises an EinopsError.
_TEMPORAL_KINDS: frozenset[str] = frozenset(
    {"tsvit", "tsvit-pheno", "tsvit-pheno-fullm", "utae", "anysat"}
)


def _is_temporal_kind(model_kind: str) -> bool:
    """Return whether a model kind needs the full time series (no collapse).

    Args:
        model_kind: the checkpoint's ``model_kind``.

    Returns:
        True if the dataset must keep ``(T, C, H, W)`` (temporal model), False if
        the time axis is collapsed to a single median frame (e.g. U-Net/DeepLab).
    """
    return model_kind in _TEMPORAL_KINDS


def _softmax_for_patch(
    model: torch.nn.Module,
    *,
    spec: CheckpointSpec,
    x: torch.Tensor,
    pid: str,
    root: Path,
    device: torch.device,
) -> np.ndarray:
    """Compute the unified 18-class / 128 softmax for one patch.

    Dispatches SegFormer to its dedicated 3-RGB / 256 sub-pipeline
    (:func:`ml.eval.segmentation_inference.softmax_logits_segformer`) and every
    other architecture to :func:`softmax_patch_for_kind`, then unifies the result
    via :func:`_unify_to_18_at_128`.

    Args:
        model: Loaded model in ``eval()``.
        spec: Checkpoint descriptor.
        x: Patch tensor delivered by the dataset (unused for SegFormer, which
            reloads the raw S2 by ``pid``).
        pid: PASTIS patch id (SegFormer reloads raw S2 with it).
        root: PASTIS-R root directory.
        device: Inference device.

    Returns:
        Unified softmax ``(18, 128, 128)`` float32.
    """
    from ml.eval.segmentation_inference import (
        softmax_logits_segformer,
        softmax_patch_for_kind,
    )

    if spec.model_kind == "segformer":
        probs_native = softmax_logits_segformer(model, pid, root=root, device=device)
    else:
        probs_native = softmax_patch_for_kind(model, x, model_kind=spec.model_kind)
    return _unify_to_18_at_128(probs_native, spec)


def _missing_entry(spec: CheckpointSpec, *, fold: int, reason: str, detail: str) -> dict[str, Any]:
    """Build the manifest entry for a checkpoint that could not be scored.

    Args:
        spec: Checkpoint descriptor.
        fold: Scoring fold (held-out flag derives from it).
        reason: Short machine reason (``checkpoint_absent`` / ``model_load_failed`` ...).
        detail: Human-readable detail for the log.

    Returns:
        A manifest entry dict with ``status="missing"`` and ``n_patches=0``.
    """
    logger.warning(
        "oof_checkpoint_missing",
        model=spec.name,
        model_kind=spec.model_kind,
        path=str(spec.path),
        reason=reason,
        detail=detail,
    )
    return {
        "model": spec.name,
        "model_kind": spec.model_kind,
        "path": None,
        "parcel_path": None,
        "shape": [HARNESS_NUM_CLASSES, HARNESS_SIZE, HARNESS_SIZE],
        "dtype": None,
        "n_patches": 0,
        "status": "missing",
        "held_out": fold == _HELD_OUT_FOLD,
        "reason": reason,
    }


def _dump_one(
    spec: CheckpointSpec,
    *,
    fold: int,
    out_dir: Path,
    data_root: Path | None,
    device: str,
    dtype: DumpDtype,
    max_patches: int | None,
    skip_missing: bool,
    write_parcel: bool,
    code_version: str,
    data_version: str,
) -> dict[str, Any]:
    """Dump the per-pixel softmax + OOF prediction for a single checkpoint.

    Mirrors ``ml.eval.dense_metrics._rescore_one`` but persists POST-softmax
    probabilities instead of accumulating a confusion matrix. Loads the model,
    iterates the held-out ``fold`` patches, computes the unified 18-class / 128
    softmax, and writes ``oof_{model}_fold{fold}.parquet`` (and the per-parcel
    sidecar when ``write_parcel``). Returns the manifest entry for this model.

    Args:
        spec: Checkpoint descriptor.
        fold: Held-out fold to score (5 by default upstream).
        out_dir: Output directory for the parquet artifacts.
        data_root: PASTIS-R root (``None`` -> dataset default).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        dtype: Stored softmax dtype (``float16`` default).
        max_patches: Optional cap on scored patches (smoke/CI).
        skip_missing: If ``True``, a missing checkpoint/dataset yields a
            ``status="missing"`` entry instead of raising.
        write_parcel: Also write the per-parcel sidecar parquet.
        code_version: Git SHA tag persisted in every row.
        data_version: DVC data version tag persisted in every row.

    Returns:
        The manifest entry dict for this model.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.dense_metrics import _apply_train_norm
    from ml.eval.segmentation_inference import load_checkpoint_model

    held_out = fold == _HELD_OUT_FOLD

    if not spec.path.exists():
        if skip_missing:
            return _missing_entry(
                spec, fold=fold, reason="checkpoint_absent", detail="path does not exist"
            )
        raise FileNotFoundError(f"checkpoint path does not exist: {spec.path}")

    is_temporal = _is_temporal_kind(spec.model_kind)
    collapse_time = None if is_temporal else "median"
    ds_kwargs: dict[str, object] = {
        "folds": (fold,),
        "collapse_time": collapse_time,
        "target": "semantic18",
        "ignore_index": 255,
    }
    if is_temporal:
        # CRITICAL (US-038/039): the temporal dataset MUST subsample the SAME number
        # of dates the model was trained with, otherwise the model receives a series
        # of a different length than it learned and its ordinal temporal PE desaligns,
        # collapsing the scores (e.g. TSViT Full-M trained with T=37 scored 0.17 when
        # the harness fed it T=10). Mirrors the same guard in ml.eval.dense_metrics:
        # the capacity lives in spec.model_kwargs (TSVIT_FULLM_CONFIG -> n_timesteps=37)
        # and models without it (L4 tsvit-pheno-v1) keep the historical default of 10,
        # matching how they were trained.
        ds_kwargs["n_timesteps"] = int(spec.model_kwargs.get("n_timesteps", 10))
    if data_root is not None:
        ds_kwargs["root"] = data_root

    try:
        dataset = PASTISSegmentationDataset(**ds_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        if skip_missing:
            return _missing_entry(spec, fold=fold, reason="dataset_absent", detail=str(exc))
        raise
    _apply_train_norm(dataset)

    try:
        model = load_checkpoint_model(spec, device=device)
    except (FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        if skip_missing:
            return _missing_entry(spec, fold=fold, reason="model_load_failed", detail=str(exc))
        raise

    resolved_device = next((p.device for p in model.parameters()), torch.device("cpu"))

    n_total = len(dataset)
    if max_patches is not None:
        n_total = min(n_total, max_patches)

    pixel_rows: list[dict[str, Any]] = []
    parcel_frames: list[pl.DataFrame] = []
    n_scored = 0
    with torch.no_grad():
        for idx in range(n_total):
            x, _y = dataset[idx]
            pid = dataset.patch_ids[idx]
            probs_18 = _softmax_for_patch(
                model,
                spec=spec,
                x=x,
                pid=pid,
                root=dataset.root,
                device=resolved_device,
            )
            pred = probs_18.argmax(axis=0).astype(np.int8)
            pixel_rows.append(
                {
                    "patch_id": str(pid),
                    "fold": fold,
                    "held_out": held_out,
                    "model": spec.name,
                    "status": "ok",
                    "softmax": probs_18,
                    "pred": pred,
                    "code_version": code_version,
                    "data_version": data_version,
                }
            )
            if write_parcel:
                parcel_frame = _parcel_frame_for_patch(
                    probs_18,
                    pid=pid,
                    spec=spec,
                    fold=fold,
                    held_out=held_out,
                    data_root=dataset.root,
                    code_version=code_version,
                    data_version=data_version,
                )
                if parcel_frame is not None:
                    parcel_frames.append(parcel_frame)
            n_scored += 1

    pixel_path = out_dir / f"oof_{spec.name}_fold{fold}.parquet"
    write_softmax_parquet(
        pixel_rows,
        pixel_path,
        num_classes=HARNESS_NUM_CLASSES,
        size=HARNESS_SIZE,
        dtype=dtype,
    )

    parcel_path: Path | None = None
    if write_parcel and parcel_frames:
        import polars as pl

        parcel_path = out_dir / f"oof_parcel_{spec.name}_fold{fold}.parquet"
        pl.concat(parcel_frames, how="vertical").write_parquet(parcel_path, compression="zstd")

    logger.info(
        "oof_checkpoint_done",
        model=spec.name,
        fold=fold,
        n_patches=n_scored,
        held_out=held_out,
        path=str(pixel_path),
    )
    return {
        "model": spec.name,
        "model_kind": spec.model_kind,
        "path": str(pixel_path),
        "parcel_path": str(parcel_path) if parcel_path is not None else None,
        "shape": [HARNESS_NUM_CLASSES, HARNESS_SIZE, HARNESS_SIZE],
        "dtype": dtype,
        "n_patches": n_scored,
        "status": "ok",
        "held_out": held_out,
        "reason": None,
    }


def _parcel_frame_for_patch(
    probs_18: np.ndarray,
    *,
    pid: str,
    spec: CheckpointSpec,
    fold: int,
    held_out: bool,
    data_root: Path,
    code_version: str,
    data_version: str,
) -> pl.DataFrame | None:
    """Reduce one patch's dense softmax to a per-parcel frame (or ``None``).

    Loads the patch's ParcelIDs raster and calls
    :func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs` (mean of
    probabilities, anti-leakage). Adds the per-patch provenance columns and the
    ``held_out`` flag. When the ParcelIDs file is absent (DVC not pulled) the
    parcel sidecar is skipped for that patch with a debug log, never raising.

    Args:
        probs_18: Unified softmax ``(18, 128, 128)`` for the patch.
        pid: PASTIS patch id.
        spec: Checkpoint descriptor (for the ``model`` column).
        fold: Scoring fold.
        held_out: Held-out flag for the rows.
        data_root: PASTIS-R root directory.
        code_version: Git SHA tag.
        data_version: DVC data version tag.

    Returns:
        A Polars DataFrame with one row per parcel (provenance columns added), or
        ``None`` if the ParcelIDs raster is missing or no parcel pixel exists.
    """
    import polars as pl

    try:
        parcel_ids = load_pastis_parcel_ids(pid, data_root)
    except FileNotFoundError as exc:
        logger.debug("oof_parcel_ids_missing", model=spec.name, patch_id=str(pid), detail=str(exc))
        return None

    frame = pixel_to_parcel_probs(probs_18, parcel_ids, patch_id=pid, method="mean")
    if frame.height == 0:
        return None
    return frame.select(
        "canonical_parcel_id",
        pl.lit(str(pid)).alias("patch_id"),
        pl.lit(fold).cast(pl.Int8).alias("fold"),
        pl.lit(held_out).alias("held_out"),
        pl.lit(spec.name).alias("model"),
        *PROB_COLUMNS,
        "pred_class",
        "n_pixels",
        pl.lit(code_version).alias("code_version"),
        pl.lit(data_version).alias("data_version"),
    )


def dump_oof(
    registry: dict[str, CheckpointSpec] | None = None,
    *,
    fold: int = 5,
    out_dir: Path | str | None = None,
    data_root: Path | str | None = None,
    device: str = "auto",
    dtype: DumpDtype = "float16",
    max_patches: int | None = None,
    skip_missing: bool = True,
    write_parcel: bool = True,
) -> dict[str, Any]:
    """Persist per-pixel POST-softmax + OOF predictions for every checkpoint.

    Reuses the US-030 harness (``load_checkpoint_model``, the fold-5 dataset,
    ``_apply_train_norm``) and :func:`softmax_patch_for_kind`. For each model
    writes ``oof_{model}_fold{fold}.parquet`` (one row per patch: patch_id, fold,
    held_out, model, status, softmax ``(18,128,128)`` ``dtype``, pred int8,
    code/data_version) and, if ``write_parcel``, the per-parcel sidecar
    ``oof_parcel_{model}_fold{fold}.parquet`` via
    :func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs`. Writes a
    ``manifest.json`` in ``out_dir``.

    ``held_out`` is True only for ``fold == 5`` (the single held-out fold with one
    best.pt per model). Folds 1-4 are NOT dumped by default (leakage).

    Args:
        registry: Model -> :class:`CheckpointSpec` map. Defaults to
            ``CHECKPOINT_REGISTRY`` (the US-030 registry).
        fold: Held-out fold to score (5 = official held-out).
        out_dir: Output directory (default ``ml/eval/oof``).
        data_root: PASTIS-R root (``None`` -> dataset default).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        dtype: Stored softmax dtype (``"float16"`` default, ``"float32"`` opt-in).
        max_patches: Optional cap on scored patches per model (CI/smoke).
        skip_missing: If ``True``, missing checkpoints/encoders yield a
            ``status="missing"`` entry instead of raising.
        write_parcel: Also write the per-parcel sidecar parquet per model.

    Returns:
        Manifest dict ``{"fold", "held_out", "code_version", "data_version",
        "dtype", "num_classes", "size", "models": {name: entry}}`` where each
        entry holds ``path, parcel_path, shape, dtype, n_patches, status,
        held_out, reason``. The same dict is also written to ``manifest.json``.
    """
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY

    active_registry = registry if registry is not None else CHECKPOINT_REGISTRY
    resolved_out = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    resolved_out.mkdir(parents=True, exist_ok=True)
    root = Path(data_root) if data_root is not None else None

    code_version = git_sha()
    data_version = _data_version()

    models: dict[str, Any] = {}
    for name, spec in active_registry.items():
        logger.info("oof_checkpoint_start", model=name, fold=fold)
        models[name] = _dump_one(
            spec,
            fold=fold,
            out_dir=resolved_out,
            data_root=root,
            device=device,
            dtype=dtype,
            max_patches=max_patches,
            skip_missing=skip_missing,
            write_parcel=write_parcel,
            code_version=code_version,
            data_version=data_version,
        )

    manifest: dict[str, Any] = {
        "fold": fold,
        "held_out": fold == _HELD_OUT_FOLD,
        "code_version": code_version,
        "data_version": data_version,
        "dtype": dtype,
        "num_classes": HARNESS_NUM_CLASSES,
        "size": HARNESS_SIZE,
        "models": models,
    }
    manifest_path = resolved_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "oof_dump_done",
        fold=fold,
        n_models=len(models),
        n_ok=sum(1 for m in models.values() if m["status"] == "ok"),
        manifest=str(manifest_path),
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m ml.eval.oof.dump_oof``."""
    parser = argparse.ArgumentParser(
        description=(
            "Dump per-pixel POST-softmax + OOF predictions for the six "
            "segmentation checkpoints over the PASTIS held-out fold."
        )
    )
    parser.add_argument("--fold", type=int, default=5, help="Held-out fold (default 5).")
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Output dir (default ml/eval/oof)."
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="PASTIS-R root (default dataset)."
    )
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="Stored softmax dtype (default float16).",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Cap scored patches per model (smoke).",
    )
    parser.add_argument(
        "--no-parcel",
        action="store_true",
        help="Skip the per-parcel sidecar parquet.",
    )
    parser.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Raise instead of skipping missing checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args and run :func:`dump_oof`.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    args = _build_arg_parser().parse_args(argv)
    dump_oof(
        fold=args.fold,
        out_dir=args.out_dir,
        data_root=args.data_root,
        device=args.device,
        dtype=args.dtype,
        max_patches=args.max_patches,
        skip_missing=not args.no_skip_missing,
        write_parcel=not args.no_parcel,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
