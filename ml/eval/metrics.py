"""Metrics for the crop classification baseline (US-019, EPIC 4) and for
pixel-level dense segmentation (US-025, EPIC 5).

Reusable module consumed by the tabular baseline (RF/XGB) and by the
segmentation architectures of EPIC 5/6. Exposes:

* Parcel level (AC-3 US-019): ``compute_baseline_metrics`` with the five
  exact metrics plus two artifacts (confusion matrix and report).
* Pixel level (US-025): ``dense_miou``, ``dense_f1_macro``,
  ``dense_pixel_accuracy`` and ``segmentation_metrics_report``, which accept
  ``torch`` or ``numpy`` tensors, support logits ``(B, C, H, W)`` or
  labels ``(B, H, W)`` and ignore an ``ignore_index`` (Background/Void).

Decision D6 (plan US-019 2.1): the ``mIoU`` of the tabular baseline is computed
as ``jaccard_score(average="macro")`` at parcel level. It is a *proxy* of the
pixel-level dense segmentation mIoU of EPIC 5; it is documented as such to
keep table comparability across epics. The ``dense_*`` functions of this
module are the real dense mIoU (per-class Jaccard aggregated over all the
valid pixels of the batch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

if TYPE_CHECKING:  # pragma: no cover - only for type annotations
    import torch

    DenseArray = np.ndarray | torch.Tensor
else:
    DenseArray = Any

__all__ = [
    "classification_report_text",
    "compute_baseline_metrics",
    "confusion_matrix_figure",
    "dense_confusion_matrix",
    "dense_f1_macro",
    "dense_metrics_from_cm",
    "dense_miou",
    "dense_pixel_accuracy",
    "segmentation_metrics_report",
]


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
) -> dict[str, float]:
    """Computes the five baseline metrics (criterion AC-3).

    Args:
        y_true: True labels, integer vector ``(n_samples,)``.
        y_pred: Predicted labels, integer vector ``(n_samples,)`` of the
            same length as ``y_true``.
        labels: Explicit set of labels to consider. If
            ``None`` it is inferred from the union of classes present in
            ``y_true`` and ``y_pred`` (ascending order). Passing the full
            class universe guarantees stable metrics across folds.

    Returns:
        Dictionary with the exact keys ``f1_macro``, ``f1_weighted``,
        ``miou``, ``accuracy`` and ``cohen_kappa``, all ``float``. The
        first four live in ``[0, 1]``; ``cohen_kappa`` can be
        negative (agreement worse than chance).

    Raises:
        ValueError: if ``y_true`` and ``y_pred`` differ in length or if
            both vectors are empty.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` and `y_pred` must have the same shape; got {y_true.shape} vs {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("`y_true` and `y_pred` cannot be empty.")

    if labels is None:
        resolved_labels: list[int] = sorted(int(c) for c in np.union1d(y_true, y_pred))
    else:
        resolved_labels = [int(c) for c in labels]

    return {
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "miou": float(
            jaccard_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    normalize: bool = True,
) -> Figure:
    """Builds the confusion matrix as a :class:`matplotlib.figure.Figure`.

    Uses matplotlib's ``Agg`` backend (non-interactive) so that the
    figure is serializable to PNG in CI and in notebooks executed with
    papermill.

    Args:
        y_true: True labels, vector ``(n_samples,)``.
        y_pred: Predicted labels, vector ``(n_samples,)``.
        class_names: Map ``{class_id: name}`` to label axes. If
            ``None`` the class integers are used as labels.
        normalize: If ``True`` (default) normalizes each row so it
            sums to 1.0 (per-class recall); if ``False`` shows counts.

    Returns:
        matplotlib figure ready for ``fig.savefig(...)`` or ``display``.

    Raises:
        ValueError: if ``y_true`` and ``y_pred`` differ in length.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` and `y_pred` must have the same shape; got {y_true.shape} vs {y_pred.shape}."
        )

    labels = sorted(int(c) for c in np.union1d(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_matrix = matrix.astype(np.float64)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero in classes absent from the ground truth.
        row_sums[row_sums == 0.0] = 1.0
        display_matrix = display_matrix / row_sums

    tick_labels = [(class_names.get(c, str(c)) if class_names else str(c)) for c in labels]

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.6),) * 2)
    image = ax.imshow(display_matrix, cmap="Blues", vmin=0.0, vmax=display_matrix.max() or 1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de confusion " + ("normalizada (recall)" if normalize else "(conteos)"))

    text_fmt = "{:.2f}" if normalize else "{:.0f}"
    threshold = display_matrix.max() / 2.0 if display_matrix.size else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = display_matrix[i, j]
            ax.text(
                j,
                i,
                text_fmt.format(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    return fig


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
) -> str:
    """Returns the sklearn classification report as text.

    Args:
        y_true: True labels, vector ``(n_samples,)``.
        y_pred: Predicted labels, vector ``(n_samples,)``.
        class_names: Map ``{class_id: name}`` to use readable class
            names instead of integers.

    Returns:
        Multiline string with precision, recall, F1 and support per class
        plus the macro and weighted averages, ready for ``log_artifact`` or
        for printing in the notebook.

    Raises:
        ValueError: if ``y_true`` and ``y_pred`` differ in length.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` and `y_pred` must have the same shape; got {y_true.shape} vs {y_pred.shape}."
        )

    labels = sorted(int(c) for c in np.union1d(y_true, y_pred))
    target_names = [(class_names.get(c, str(c)) if class_names else str(c)) for c in labels]
    report: str = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    return report


# ---------------------------------------------------------------------------
# Dense pixel-level segmentation (US-025, EPIC 5)
# ---------------------------------------------------------------------------


def _to_numpy(arr: DenseArray) -> np.ndarray:
    """Converts a ``torch`` tensor or ``numpy`` array to ``numpy`` without a free copy.

    Args:
        arr: ``torch`` tensor (on any device) or ``numpy.ndarray``.

    Returns:
        The content as ``numpy.ndarray`` on CPU. If ``arr`` is already
        ``numpy`` it is returned as is (``np.asarray`` does not copy if the dtype
        and contiguity already match).
    """
    if hasattr(arr, "detach"):  # torch.Tensor (avoids hard import of torch)
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _to_label_array(arr: DenseArray, *, n_classes: int) -> np.ndarray:
    """Normalizes the input to flattened integer labels ``(N,)``.

    Accepts both logits/probabilities ``(B, C, H, W)`` (``argmax`` is
    applied over the ``C`` channel axis) and hard labels of
    any shape ``(B, H, W)``, ``(H, W)`` or already flattened.

    The heuristic to detect logits is: floating-point array with an
    axis whose size matches ``n_classes`` at the channel position
    (axis 1 for ``(B, C, H, W)``). Integer labels are always treated
    as labels, never as logits.

    Args:
        arr: Logits ``(B, C, H, W)`` or labels of arbitrary shape.
        n_classes: Number of classes ``C`` expected to recognize logits.

    Returns:
        1-D ``numpy`` vector of integer labels (``int64``).
    """
    data = _to_numpy(arr)
    is_float = np.issubdtype(data.dtype, np.floating)
    if is_float and data.ndim == 4 and data.shape[1] == n_classes:
        data = data.argmax(axis=1)
    elif is_float and data.ndim == 4:
        # Float 4-D without channel == n_classes: assume channel on axis 1 anyway.
        data = data.argmax(axis=1)
    return data.reshape(-1).astype(np.int64, copy=False)


def dense_confusion_matrix(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> np.ndarray:
    """Builds the dense confusion matrix ``(n_classes, n_classes)``.

    Aggregates over all valid pixels of the batch. Pixels whose
    *ground truth* is ``ignore_index`` (Background/Void) are excluded
    entirely, as are pixels whose true label falls outside
    ``[0, n_classes)`` (defense against mis-mapped targets).

    Args:
        y_pred: Logits ``(B, C, H, W)`` or predicted labels ``(B, H, W)``.
        y_true: True labels (``numpy`` or ``torch``) with the same
            number of pixels as ``y_pred`` after flattening.
        n_classes: Number of classes ``C`` (18 for semantic PASTIS-R).
        ignore_index: Label value to ignore (Background/Void).

    Returns:
        ``int64`` ``numpy`` matrix of shape ``(n_classes, n_classes)`` with
        ``cm[i, j]`` = pixels of true class ``i`` predicted as ``j``.

    Raises:
        ValueError: if ``y_pred`` and ``y_true`` do not have the same number
            of pixels after flattening.
    """
    pred = _to_label_array(y_pred, n_classes=n_classes)
    true = _to_label_array(y_true, n_classes=n_classes)
    if pred.shape != true.shape:
        raise ValueError(
            f"`y_pred` and `y_true` must have the same number of pixels; "
            f"got {pred.shape} vs {true.shape}."
        )

    valid = (true != ignore_index) & (true >= 0) & (true < n_classes)
    true = true[valid]
    pred = pred[valid]
    # Out-of-range predictions are clamped to the last class so as not to
    # break the bincount; in practice argmax over n_classes never exceeds it.
    pred = np.clip(pred, 0, n_classes - 1)

    indices = true * n_classes + pred
    counts = np.bincount(indices, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(np.int64, copy=False)


def _per_class_iou_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU (Jaccard) from a confusion matrix.

    Args:
        cm: Dense ``(n_classes, n_classes)`` confusion matrix.

    Returns:
        ``(n_classes,)`` vector with the per-class IoU. Classes absent
        from both the *ground truth* and the predictions (empty union) receive
        ``nan`` to be excluded from the macro average.
    """
    cm = cm.astype(np.float64)
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, intersection / union, np.nan)
    return iou


def dense_miou(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Computes the pixel-level dense mIoU (mean Jaccard).

    Macro average of the per-class IoU over the classes present in the
    union (ground truth or prediction). Fully absent classes are
    excluded from the average (they do not penalize with zero), following the
    PASTIS/U-TAE convention for folds where not all classes appear.

    Args:
        y_pred: Logits ``(B, C, H, W)`` or labels ``(B, H, W)`` (``torch``
            or ``numpy``).
        y_true: True labels (``torch`` or ``numpy``).
        n_classes: Number of classes (18 for semantic PASTIS-R).
        ignore_index: Label to ignore (Background/Void).

    Returns:
        mIoU in ``[0, 1]``. Returns ``0.0`` if there is no valid class
        (all pixels were ``ignore_index``).
    """
    cm = dense_confusion_matrix(y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index)
    iou = _per_class_iou_from_cm(cm)
    if np.all(np.isnan(iou)):
        return 0.0
    return float(np.nanmean(iou))


def dense_f1_macro(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Computes the pixel-level dense F1-macro (averaged per-class Dice).

    Equivalent to the macro average of the per-class F1 over the valid pixels.
    Classes absent from the union (with neither GT nor prediction) are excluded
    from the average, in line with ``dense_miou``.

    Args:
        y_pred: Logits ``(B, C, H, W)`` or labels ``(B, H, W)``.
        y_true: True labels.
        n_classes: Number of classes.
        ignore_index: Label to ignore.

    Returns:
        F1-macro in ``[0, 1]``. Returns ``0.0`` if there are no valid classes.
    """
    cm = dense_confusion_matrix(
        y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index
    ).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


def dense_pixel_accuracy(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Computes the global pixel-level accuracy (correct / valid pixels).

    Args:
        y_pred: Logits ``(B, C, H, W)`` or labels ``(B, H, W)``.
        y_true: True labels.
        n_classes: Number of classes (defines the valid target range).
        ignore_index: Label to ignore (Background/Void).

    Returns:
        Accuracy in ``[0, 1]``. Returns ``0.0`` if there are no valid pixels.
    """
    cm = dense_confusion_matrix(y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index)
    total = int(cm.sum())
    if total == 0:
        return 0.0
    return float(np.trace(cm)) / float(total)


def segmentation_metrics_report(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Complete dense segmentation metrics report in a single pass.

    Builds the confusion matrix once and derives all the
    metrics, avoiding recomputations (DRY/efficiency). Useful for logging to
    MLflow at the close of each epoch/eval of EPIC 5.

    Args:
        y_pred: Logits ``(B, C, H, W)`` or labels ``(B, H, W)`` (``torch``
            or ``numpy``).
        y_true: True labels (``torch`` or ``numpy``).
        n_classes: Number of classes (18 semantic PASTIS-R, 6 HCAT L1).
        ignore_index: Label to ignore (Background/Void).

    Returns:
        Dictionary with the keys:

        * ``miou`` (``float``): macro mean IoU over present classes.
        * ``f1_macro`` (``float``): dense F1-macro.
        * ``pixel_acc`` (``float``): global pixel-level accuracy.
        * ``per_class_iou`` (``list[float | None]``): per-class IoU from ``0``
          to ``n_classes - 1``; ``None`` for classes absent from the union.
    """
    cm = dense_confusion_matrix(y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index)
    cm_f = cm.astype(np.float64)

    iou = _per_class_iou_from_cm(cm)
    miou = 0.0 if np.all(np.isnan(iou)) else float(np.nanmean(iou))

    tp = np.diag(cm_f)
    fp = cm_f.sum(axis=0) - tp
    fn = cm_f.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    present = (cm_f.sum(axis=1) + cm_f.sum(axis=0)) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    f1_macro = 0.0 if not np.any(present) else float(f1[present].mean())

    total = int(cm.sum())
    pixel_acc = 0.0 if total == 0 else float(np.trace(cm)) / float(total)

    per_class_iou: list[float | None] = [
        (None if np.isnan(value) else float(value)) for value in iou
    ]

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
    }


def dense_metrics_from_cm(cm: np.ndarray) -> dict[str, Any]:
    """Extracts all segmentation metrics from a confusion matrix.

    Computes the complete report from an already-accumulated ``cm`` (over the
    whole split), without recomputing predictions. Includes macro, per-class and
    two metrics robust to imbalance (PASTIS has classes with ~50x frequency
    difference): Cohen kappa and balanced accuracy (average recall).

    Args:
        cm: Dense ``(n_classes, n_classes)`` confusion matrix (rows =
            ground truth, columns = prediction).

    Returns:
        Dictionary with:
        - ``miou``: macro mean IoU (absent classes excluded).
        - ``f1_macro``: macro F1 over present classes.
        - ``pixel_acc``: global accuracy (overall accuracy).
        - ``balanced_acc``: mean of the per-present-class recalls.
        - ``cohen_kappa``: chance-corrected agreement.
        - ``per_class_iou`` / ``per_class_f1``: ``(n_classes,)`` lists with
          ``None`` for the classes absent from the split.
    """
    cm_f = cm.astype(np.float64)
    n_classes = cm.shape[0]

    iou = _per_class_iou_from_cm(cm)
    miou = 0.0 if np.all(np.isnan(iou)) else float(np.nanmean(iou))

    tp = np.diag(cm_f)
    fp = cm_f.sum(axis=0) - tp
    fn = cm_f.sum(axis=1) - tp
    support = cm_f.sum(axis=1)  # n real pixels per class
    present = (support + cm_f.sum(axis=0)) > 0.0

    denom_f1 = 2.0 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom_f1 > 0.0, 2.0 * tp / denom_f1, np.nan)
    f1_macro = 0.0 if not np.any(present) else float(np.nanmean(f1[present]))

    total = float(cm_f.sum())
    pixel_acc = 0.0 if total == 0.0 else float(np.trace(cm_f) / total)

    # Balanced accuracy = mean of recalls (TP / support) over classes with
    # at least one real pixel. Robust to imbalance because each class weighs
    # the same regardless of its frequency.
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0.0, tp / support, np.nan)
    has_support = support > 0.0
    balanced_acc = 0.0 if not np.any(has_support) else float(np.nanmean(recall[has_support]))

    # Cohen kappa from the confusion matrix (direct formula).
    p_o = pixel_acc
    expected = float((cm_f.sum(axis=0) * cm_f.sum(axis=1)).sum())
    p_e = 0.0 if total == 0.0 else expected / (total * total)
    cohen_kappa = 0.0 if (1.0 - p_e) == 0.0 else float((p_o - p_e) / (1.0 - p_e))

    per_class_iou: list[float | None] = [(None if np.isnan(v) else float(v)) for v in iou]
    per_class_f1: list[float | None] = [
        (None if (i >= n_classes or np.isnan(f1[i])) else float(f1[i])) for i in range(n_classes)
    ]

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "pixel_acc": pixel_acc,
        "balanced_acc": balanced_acc,
        "cohen_kappa": cohen_kappa,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }
