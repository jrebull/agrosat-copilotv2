"""Pixel-level metrics for dense semantic segmentation (EPIC 5/6).

Complements :mod:`ml.eval.metrics` (which operates at the parcel level) with the
three segmentation metrics required by the Avance 4 rubric: **mIoU**,
**F1-macro** and **pixel-accuracy**, computed at the pixel level over 2D maps.

The implementation accumulates a ``(C, C)`` confusion matrix in pure torch (no
``torchmetrics`` dependency), which allows aggregating batches in streaming
during validation and deriving the three metrics exactly at the end. The
``ignore_index`` class (void = 19 in PASTIS-R) is excluded from both the
accumulation and the macro average.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
import torch
from matplotlib.figure import Figure

from ml.eval.metrics import confusion_matrix_figure

if TYPE_CHECKING:
    from ml.eval.checkpoint_registry import CheckpointSpec

logger = structlog.get_logger(__name__)

__all__ = [
    "DenseConfusionAccumulator",
    "compute_dense_metrics",
    "dense_confusion_figure",
    "rescore_all_checkpoints",
]

#: Train folds used ONLY for normalization statistics (anti-leakage): the
#: held-out scoring fold (5) is never used to compute norm stats.
_TRAIN_NORM_FOLDS: tuple[int, ...] = (1, 2, 3)

#: SegFormer (Isaac, notebook 04i) 3-RGB normalization constants and train size.
#: Reproduced verbatim from :func:`ml.eval.avance4_figures.regen_isaac_model`.
_SEGFORMER_RGB_MEAN = np.array([1158.0, 1244.7, 1416.3], dtype=np.float32)[:, None, None]
_SEGFORMER_RGB_STD = np.array([671.7, 698.1, 761.3], dtype=np.float32)[:, None, None]
_SEGFORMER_SIZE = 256


def _as_long_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert a numpy/torch input to a ``torch.Tensor`` ``long`` on CPU."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    return x.detach().long()


class DenseConfusionAccumulator:
    """Pixel-level confusion matrix accumulator for dense metrics.

    Allows ``update`` per batch during validation and ``compute`` at the end,
    deriving mIoU, F1-macro and pixel-accuracy from the accumulated matrix. The
    ``ignore_index`` class is filtered out of the ground truth before
    accumulating.

    Attributes:
        num_classes: Number of classes in the problem.
        ignore_index: Class to ignore (contributes neither to the confusion nor
            to the macro).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ignore_index: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the accumulator with a ``(C, C)`` zeroed matrix.

        Args:
            num_classes: Number of classes ``C``.
            ignore_index: Class to ignore (``None`` to ignore none).
            device: Device on which to keep the accumulated matrix.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Reset the accumulated confusion matrix to zeros."""
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64, device=self._device
        )

    def update(self, preds: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        """Accumulate a batch of predictions against the ground truth.

        Args:
            preds: Predicted class map(s), integers of any shape.
            target: Ground truth class map(s), same shape as ``preds``.

        Raises:
            ValueError: if ``preds`` and ``target`` differ in shape.
        """
        preds_t = _as_long_tensor(preds).to(self._device).reshape(-1)
        target_t = _as_long_tensor(target).to(self._device).reshape(-1)
        if preds_t.shape != target_t.shape:
            raise ValueError(
                f"`preds` and `target` must have the same number of pixels; "
                f"received {preds_t.numel()} vs {target_t.numel()}."
            )

        valid = torch.ones_like(target_t, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= target_t != self.ignore_index
        # Defensive: discard out-of-range pixels (e.g. pred==num_classes).
        valid &= (target_t >= 0) & (target_t < self.num_classes)
        valid &= (preds_t >= 0) & (preds_t < self.num_classes)

        t = target_t[valid]
        p = preds_t[valid]
        if t.numel() == 0:
            return
        indices = t * self.num_classes + p
        binned = torch.bincount(indices, minlength=self.num_classes**2)
        self._confusion += binned.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        """Derive mIoU, F1-macro and pixel-accuracy from the accumulated matrix.

        The macro average (mIoU and F1) is taken only over the classes present in
        the ground truth (support > 0), excluding ``ignore_index``. This avoids
        biasing the metric downward due to classes absent from the val split.

        Returns:
            Dictionary with ``miou``, ``f1_macro`` and ``pixel_accuracy`` (floats
            in ``[0, 1]``). If no valid pixel was accumulated, it returns zeros.
        """
        conf = self._confusion.double()
        total = conf.sum()
        if total <= 0:
            return {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}

        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)  # real support per class
        col_sum = conf.sum(dim=0)  # predictions per class

        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))

        precision = torch.where(col_sum > 0, diag / col_sum, torch.zeros_like(diag))
        recall = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(diag))

        present = row_sum > 0
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            present[self.ignore_index] = False

        n_present = int(present.sum().item())
        miou = float(iou[present].mean().item()) if n_present > 0 else 0.0
        f1_macro = float(f1[present].mean().item()) if n_present > 0 else 0.0
        pixel_accuracy = float((diag.sum() / total).item())
        return {"miou": miou, "f1_macro": f1_macro, "pixel_accuracy": pixel_accuracy}

    def confusion_matrix(self) -> np.ndarray:
        """Return a copy of the accumulated ``(C, C)`` confusion matrix as numpy.

        Exposes the accumulated matrix so derived metrics (per-class IoU/F1,
        Cohen kappa, balanced accuracy) can be computed once via
        :func:`ml.eval.metrics.dense_metrics_from_cm` without re-running
        inference (used by the US-030 re-score harness).

        Returns:
            ``int64`` numpy array of shape ``(num_classes, num_classes)``.
        """
        return self._confusion.detach().cpu().numpy().astype(np.int64, copy=True)

    def per_class_iou(self) -> dict[int, float]:
        """Return the per-class IoU (for the per-class IoU barplot).

        Returns:
            Dictionary ``{class_id: iou}`` only for the classes with support in
            the ground truth (excluding ``ignore_index``). Empty if there are no
            pixels.
        """
        conf = self._confusion.double()
        if conf.sum() <= 0:
            return {}
        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)
        col_sum = conf.sum(dim=0)
        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
        out: dict[int, float] = {}
        for c in range(self.num_classes):
            if c == self.ignore_index or row_sum[c] <= 0:
                continue
            out[c] = float(iou[c].item())
        return out


def compute_dense_metrics(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Compute mIoU + F1-macro + pixel-accuracy in a single pass (one-shot).

    Convenience over :class:`DenseConfusionAccumulator` to evaluate a full
    ``(preds, target)`` pair at once (tests, final evaluation).

    Args:
        preds: Predicted class map(s).
        target: Ground truth class map(s).
        num_classes: Number of classes ``C``.
        ignore_index: Class to ignore (default ``None``).

    Returns:
        Dictionary with ``miou``, ``f1_macro`` and ``pixel_accuracy``.
    """
    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    acc.update(preds, target)
    return acc.compute()


def dense_confusion_figure(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    normalize: bool = True,
) -> Figure:
    """Pixel-level confusion matrix reusing :func:`confusion_matrix_figure`.

    Flattens the 2D maps into pixel vectors, discards the pixels whose ground
    truth is ``ignore_index`` and delegates the rendering to the existing
    baseline helper (DRY, same visual style as the parcel-level matrices).

    Args:
        preds: Predicted class map(s).
        target: Ground truth class map(s).
        class_names: Map ``{class_id: name}`` to label the axes.
        ignore_index: Class to exclude from the plot (default ``None``).
        normalize: If ``True`` normalizes by row (per-class recall).

    Returns:
        matplotlib figure ready for ``savefig``/``display``.
    """
    p = _as_long_tensor(preds).reshape(-1).cpu().numpy()
    t = _as_long_tensor(target).reshape(-1).cpu().numpy()
    if ignore_index is not None:
        mask = t != ignore_index
        p, t = p[mask], t[mask]
    return confusion_matrix_figure(t, p, class_names=class_names, normalize=normalize)


# ---------------------------------------------------------------------------
# Apples-to-apples re-score harness (US-030)
# ---------------------------------------------------------------------------


def _train_norm_stats(
    raw_stats: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    folds: tuple[int, ...] = _TRAIN_NORM_FOLDS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Average per-fold S2 normalization stats over the train folds only.

    Anti-leakage (plan R4/AC-6): the held-out scoring fold (5) must never feed the
    normalization statistics. We average the per-band mean/std of the train folds
    (1, 2, 3) and apply them to every scored patch.

    Args:
        raw_stats: ``{fold: (mean[10], std[10])}`` as loaded from
            ``NORM_S2_patch.json``.
        folds: Train folds to average over (default (1, 2, 3)).

    Returns:
        Tuple ``(mean[10], std[10])`` averaged over the available train folds, or
        ``None`` if none of those folds has stats (caller then falls back to the
        plain ``/10000`` scale).
    """
    means = [raw_stats[f][0] for f in folds if f in raw_stats]
    stds = [raw_stats[f][1] for f in folds if f in raw_stats]
    if not means:
        return None
    return (
        np.mean(np.stack(means, axis=0), axis=0),
        np.mean(np.stack(stds, axis=0), axis=0),
    )


def _apply_train_norm(dataset: object) -> None:
    """Force a dataset to normalize every patch with train-fold stats (no leakage).

    Overwrites the dataset's internal per-fold normalization map so that all
    patches (including the held-out fold) are standardized with the averaged
    train-fold statistics. If the dataset has no stats, it is left untouched (it
    then falls back to the ``/10000`` scale, identically for every model).

    Args:
        dataset: A :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`
            instance whose ``_norm_stats`` / ``_fold_of`` attributes are mutated
            in place.
    """
    raw_stats = getattr(dataset, "_norm_stats", None)
    if not raw_stats:
        return
    train_stats = _train_norm_stats(raw_stats)
    if train_stats is None:
        return
    fold_of = getattr(dataset, "_fold_of", {}) or {}
    # Map EVERY patch's fold to the same averaged train stats so `_normalize`
    # always uses train-only statistics regardless of the patch's real fold.
    folds_present = set(fold_of.values())
    dataset._norm_stats = dict.fromkeys(folds_present, train_stats)
    logger.info(
        "rescore_norm_train_only",
        train_folds=_TRAIN_NORM_FOLDS,
        n_folds_overwritten=len(folds_present),
    )


def _segformer_predict_18(
    model: torch.nn.Module,
    pid: str,
    *,
    root: Path,
    device: torch.device,
) -> np.ndarray:
    """Run SegFormer's 3-RGB / 256 sub-pipeline and return an 18-class map at 128.

    SegFormer (Isaac) was trained on a 3-band RGB temporal-median composite at
    256 px with its own normalization (``_SEGFORMER_RGB_*``). This reproduces
    that exact input pipeline (plan R2), then resamples the 256 prediction to 128
    NEAREST and maps the 20-class output into the contiguous 18-class space.

    Args:
        model: Loaded ``SegformerForSemanticSegmentation``.
        pid: PASTIS patch id.
        root: PASTIS-R root directory.
        device: Inference device.

    Returns:
        Predicted class map ``(128, 128)`` int64 in the contiguous 18-class space
        (Background/Void -> ``HARNESS_IGNORE_INDEX``).
    """
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    from ml.eval.class_remap import remap_20_to_18, resample_mask_128_nearest

    s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")  # (T, C, H, W)
    img = np.median(s2, axis=0)[:3].astype(np.float32)  # RGB composite
    img = (img - _SEGFORMER_RGB_MEAN) / (_SEGFORMER_RGB_STD + 1e-6)
    t_img = (
        TF.resize(
            torch.from_numpy(img),
            [_SEGFORMER_SIZE, _SEGFORMER_SIZE],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        .unsqueeze(0)
        .to(device)
    )
    logits = model(pixel_values=t_img).logits
    logits = F.interpolate(
        logits,
        size=(_SEGFORMER_SIZE, _SEGFORMER_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    pred_256 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    pred_128 = resample_mask_128_nearest(pred_256)
    return remap_20_to_18(pred_128)


def _rescore_one(
    spec: CheckpointSpec,
    *,
    fold: int,
    data_root: Path | None,
    device: str,
    max_patches: int | None,
    skip_missing: bool,
) -> dict[str, object]:
    """Re-score a single checkpoint over the held-out fold in the 18-class space.

    Loads the model, runs inference over the held-out ``fold`` patches, maps
    every prediction into the contiguous 18-class space at 128 NEAREST and
    accumulates a unified :class:`DenseConfusionAccumulator` (num_classes=18,
    ignore_index=255). The ground-truth target comes from
    ``PASTISSegmentationDataset(target="semantic18")`` (already contiguous
    18-class), so apples-to-apples holds across the six models. Normalization
    uses train-fold statistics only (anti-leakage).

    Args:
        spec: Checkpoint descriptor.
        fold: Held-out fold to score.
        data_root: PASTIS-R root (``None`` -> dataset default).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        max_patches: Optional cap on the number of scored patches (smoke/CI).
        skip_missing: If ``True``, a missing checkpoint/encoder yields a
            ``status="missing"`` row instead of raising.

    Returns:
        A metrics row dict with keys: ``model``, ``model_kind``, ``miou``,
        ``f1_macro``, ``pixel_accuracy``, ``fold``, ``n_patches``, ``status``,
        ``needs_resize``, ``in_channels``, ``cohen_kappa``, ``balanced_acc`` and
        ``per_class_iou`` (list of 18, ``None`` for absent classes).
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.class_remap import (
        HARNESS_IGNORE_INDEX,
        HARNESS_NUM_CLASSES,
        remap_20_to_18,
        resample_mask_128_nearest,
    )
    from ml.eval.metrics import dense_metrics_from_cm
    from ml.eval.segmentation_inference import (
        load_checkpoint_model,
        predict_patch_for_kind,
    )

    def _missing_row(reason: str, detail: str) -> dict[str, object]:
        logger.warning(
            "rescore_checkpoint_missing",
            model=spec.name,
            model_kind=spec.model_kind,
            path=str(spec.path),
            reason=reason,
            detail=detail,
        )
        return {
            "model": spec.name,
            "model_kind": spec.model_kind,
            "miou": None,
            "f1_macro": None,
            "pixel_accuracy": None,
            "fold": fold,
            "n_patches": 0,
            "status": "missing",
            "needs_resize": spec.needs_resize,
            "in_channels": spec.in_channels,
            "cohen_kappa": None,
            "balanced_acc": None,
            "per_class_iou": None,
        }

    if not spec.path.exists():
        if skip_missing:
            return _missing_row("checkpoint_absent", "path does not exist")
        raise FileNotFoundError(f"checkpoint path does not exist: {spec.path}")

    # Temporal models consume the full series; the 2D CNNs a median composite.
    is_temporal = spec.model_kind in (
        "tsvit",
        "tsvit-pheno",
        "tsvit-pheno-fullm",
        "utae",
        "anysat",
    )
    collapse_time = None if is_temporal else "median"
    ds_kwargs: dict[str, object] = {
        "folds": (fold,),
        "collapse_time": collapse_time,
        "target": "semantic18",
        "ignore_index": HARNESS_IGNORE_INDEX,
    }
    if is_temporal:
        # CRITICAL (US-038/039): the temporal dataset MUST subsample the SAME number
        # of dates the model was trained with, otherwise the model receives a series
        # of a different length than it learned and its ordinal temporal PE desaligns,
        # collapsing the mIoU (e.g. TSViT Full-M trained with T=37 scored 0.17 when
        # the harness fed it T=10). The capacity lives in spec.model_kwargs
        # (TSVIT_FULLM_CONFIG -> n_timesteps=37); models without it (L4 tsvit-pheno-v1)
        # keep the historical default of 10, matching how they were trained.
        ds_kwargs["n_timesteps"] = int(spec.model_kwargs.get("n_timesteps", 10))
    if data_root is not None:
        ds_kwargs["root"] = data_root

    try:
        dataset = PASTISSegmentationDataset(**ds_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        if skip_missing:
            return _missing_row("dataset_absent", str(exc))
        raise
    _apply_train_norm(dataset)

    try:
        model = load_checkpoint_model(spec, device=device)
    except (FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        if skip_missing:
            return _missing_row("model_load_failed", str(exc))
        raise

    resolved_device = next((p.device for p in model.parameters()), torch.device("cpu"))
    acc = DenseConfusionAccumulator(HARNESS_NUM_CLASSES, ignore_index=HARNESS_IGNORE_INDEX)

    n_total = len(dataset)
    if max_patches is not None:
        n_total = min(n_total, max_patches)
    n_scored = 0
    with torch.no_grad():
        for idx in range(n_total):
            x, y = dataset[idx]
            target_18 = y.cpu().numpy().astype(np.int64)  # already contiguous 18
            if spec.model_kind == "segformer":
                pid = dataset.patch_ids[idx]
                pred_18 = _segformer_predict_18(
                    model, pid, root=dataset.root, device=resolved_device
                )
            else:
                pred_native = predict_patch_for_kind(model, x, model_kind=spec.model_kind)
                if spec.needs_resize:
                    pred_native = resample_mask_128_nearest(pred_native)
                pred_18 = (
                    remap_20_to_18(pred_native)
                    if spec.native_num_classes >= 20
                    else pred_native.astype(np.int64)
                )
            acc.update(pred_18, target_18)
            n_scored += 1

    flat = acc.compute()
    cm = acc.confusion_matrix()
    derived = dense_metrics_from_cm(cm)
    logger.info(
        "rescore_checkpoint_done",
        model=spec.name,
        fold=fold,
        n_patches=n_scored,
        miou=round(float(flat["miou"]), 4),
        f1_macro=round(float(flat["f1_macro"]), 4),
        pixel_accuracy=round(float(flat["pixel_accuracy"]), 4),
    )
    return {
        "model": spec.name,
        "model_kind": spec.model_kind,
        "miou": float(flat["miou"]),
        "f1_macro": float(flat["f1_macro"]),
        # `compute()` -> "pixel_accuracy"; `dense_metrics_from_cm` -> "pixel_acc".
        # Normalize the public column to "pixel_accuracy" (plan R10).
        "pixel_accuracy": float(flat["pixel_accuracy"]),
        "fold": fold,
        "n_patches": n_scored,
        "status": "ok",
        "needs_resize": spec.needs_resize,
        "in_channels": spec.in_channels,
        "cohen_kappa": float(derived["cohen_kappa"]),
        "balanced_acc": float(derived["balanced_acc"]),
        "per_class_iou": derived["per_class_iou"],
    }


def rescore_all_checkpoints(
    registry: dict[str, CheckpointSpec] | None = None,
    *,
    fold: int = 5,
    data_root: Path | str | None = None,
    device: str = "auto",
    max_patches: int | None = None,
    skip_missing: bool = True,
) -> pl.DataFrame:
    """Re-evaluate every registered segmentation checkpoint apples-to-apples.

    Loads each ``best.pt`` from the registry, runs inference over the PASTIS
    held-out ``fold`` (default 5, NOT 4), maps every prediction into the
    contiguous 18-class space at 128 NEAREST, accumulates a unified
    :class:`DenseConfusionAccumulator` (num_classes=18, ignore_index=255) and
    returns one row per model. Background/Void are excluded via the ignore index;
    normalization statistics use only train folds (1, 2, 3) -> no leakage.

    The four 20-class models (U-Net, U-TAE, AnySat, SegFormer) get their
    predictions mapped 20->18 AFTER argmax (the checkpoint keys, e.g. U-TAE
    ``out_conv``, stay intact); the two native 18-class models (DeepLabv3+,
    TSViT-pheno) skip the remap. SegFormer additionally runs its own 3-RGB / 256
    sub-pipeline and resamples to 128 NEAREST before the remap.

    Args:
        registry: Model -> :class:`CheckpointSpec` map. Defaults to
            ``CHECKPOINT_REGISTRY``.
        fold: Held-out fold to score (5 = official held-out, NOT the fold-4
            selection set).
        data_root: PASTIS root; ``None`` uses the dataset default path.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        max_patches: Optional cap on scored patches per model (CI/smoke);
            ``None`` = the full fold.
        skip_missing: If ``True``, missing checkpoints/encoders yield a
            ``status="missing"`` row instead of raising.

    Returns:
        Polars DataFrame, one row per model, sorted by ``miou`` descending
        (``status="missing"`` rows last). Columns: ``model``, ``model_kind``,
        ``miou``, ``f1_macro``, ``pixel_accuracy``, ``fold``, ``n_patches``,
        ``status``, ``needs_resize``, ``in_channels``, ``cohen_kappa``,
        ``balanced_acc``, ``per_class_iou``.
    """
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY

    active_registry = registry if registry is not None else CHECKPOINT_REGISTRY
    root = Path(data_root) if data_root is not None else None

    rows: list[dict[str, object]] = []
    for name, spec in active_registry.items():
        logger.info("rescore_checkpoint_start", model=name, fold=fold)
        rows.append(
            _rescore_one(
                spec,
                fold=fold,
                data_root=root,
                device=device,
                max_patches=max_patches,
                skip_missing=skip_missing,
            )
        )

    df = pl.DataFrame(rows)
    # Sort by mIoU descending; rows with null mIoU (status="missing") go last.
    return df.sort("miou", descending=True, nulls_last=True)
