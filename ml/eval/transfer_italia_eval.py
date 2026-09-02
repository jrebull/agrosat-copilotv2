"""Hierarchical dense evaluation of the Italian transfer (US-079 step 4).

Scores the fine-tuned dense members and the Voting-3 on the held-out Italian
patches at TWO granularities and quantifies the transfer:

- **FINE**: the Italian dense label space (``ItaliaLabelSpace``); pixel mIoU +
  F1-macro over the crop classes (background excluded).
- **COARSE**: every fine class collapsed to a bucket shared with PASTIS
  (``ItaliaLabelSpace.coarse_of``); the SAME metrics, so a model that only knows
  the coarse PASTIS taxonomy is comparable to the granularity-enriched one (the
  "papaya/fruits" hypothesis).

Two more deliverables of the US-079 plan live here:

- **Honest discard curve**: F1-macro as a function of how many of the best fine
  classes are retained (ranked by per-class F1), to locate the ~10-class subset
  with F1 > 0.9 (mirror of the ``france-10`` 0.9069 of the Voting-3 on PASTIS).
- **Transfer delta**: fine-tuned vs ZERO-SHOT (the French champion applied to the
  Italian patches with no fine-tune, mapping its PASTIS predictions onto the
  Italian classes via the conserved crosswalk) -- the cota inferior the fine-tune
  must beat.

Everything reuses :mod:`ml.eval.dense_metrics` (the pixel accumulator) so the
numbers are apples-to-apples with the EPIC 5 segmentation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import structlog

from ml.transfer.italia_label_space import ItaliaLabelSpace

logger = structlog.get_logger(__name__)

__all__ = [
    "DenseEvalResult",
    "best_subset_over_threshold",
    "build_coarse_label_space",
    "discard_curve",
    "evaluate_dense_predictions",
    "per_class_f1",
    "probs_to_class_map",
    "project_parcel_vote_to_dense",
    "transfer_delta",
]


@dataclass
class DenseEvalResult:
    """The fine + coarse dense metrics of a model on the Italian test set.

    Attributes:
        name: The model/combiner name (e.g. ``"voting-3"``, ``"tsvit-pheno"``).
        fine_miou: Pixel mIoU at the fine Italian granularity.
        fine_f1_macro: Pixel F1-macro at the fine granularity.
        fine_pixel_accuracy: Pixel accuracy at the fine granularity.
        coarse_miou: Pixel mIoU at the coarse (PASTIS-shared) granularity.
        coarse_f1_macro: Pixel F1-macro at the coarse granularity.
        coarse_pixel_accuracy: Pixel accuracy at the coarse granularity.
        per_class: Per-fine-class ``{"leaf", "is_new", "f1", "iou", "support"}``.
        n_pixels: Number of supervised (non-background) pixels scored.
    """

    name: str
    fine_miou: float
    fine_f1_macro: float
    fine_pixel_accuracy: float
    coarse_miou: float
    coarse_f1_macro: float
    coarse_pixel_accuracy: float
    per_class: list[dict[str, object]] = field(default_factory=list)
    n_pixels: int = 0

    def summary(self) -> dict[str, object]:
        """Return a flat JSON-friendly summary (no per-class list)."""
        return {
            "name": self.name,
            "fine_miou": round(self.fine_miou, 4),
            "fine_f1_macro": round(self.fine_f1_macro, 4),
            "fine_pixel_accuracy": round(self.fine_pixel_accuracy, 4),
            "coarse_miou": round(self.coarse_miou, 4),
            "coarse_f1_macro": round(self.coarse_f1_macro, 4),
            "coarse_pixel_accuracy": round(self.coarse_pixel_accuracy, 4),
            "n_pixels": self.n_pixels,
        }


def build_coarse_label_space(
    label_space: ItaliaLabelSpace,
) -> tuple[np.ndarray, dict[int, str]]:
    """Build a fine-id -> coarse-id LUT and the coarse id -> name map.

    Collapses every fine Italian class (and background) to its coarse bucket via
    :meth:`ItaliaLabelSpace.coarse_of`, assigning the coarse buckets contiguous
    ids in first-appearance order (background keeps id 0).

    Args:
        label_space: The Italian fine label space.

    Returns:
        ``(lut, coarse_names)`` where ``lut`` maps a fine id ``[0, num_classes)``
        to its coarse id, and ``coarse_names`` maps the coarse id to its name.
    """
    coarse_name_to_id: dict[str, int] = {}
    coarse_names: dict[int, str] = {}
    background_name = label_space.leaves[label_space.background_id]
    coarse_name_to_id[background_name] = 0
    coarse_names[0] = background_name
    lut = np.zeros(label_space.num_classes, dtype=np.int64)
    next_id = 1
    for fid in range(label_space.num_classes):
        if fid == label_space.background_id:
            continue
        leaf = label_space.leaves[fid]
        coarse = label_space.coarse_of(leaf)
        if coarse not in coarse_name_to_id:
            coarse_name_to_id[coarse] = next_id
            coarse_names[next_id] = coarse
            next_id += 1
        lut[fid] = coarse_name_to_id[coarse]
    return lut, coarse_names


def _stack_dense(
    preds_by_patch: dict[int, np.ndarray],
    masks_by_patch: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate dense pred/target maps of the shared patches into pixel arrays.

    Args:
        preds_by_patch: ``{patch_id: (H, W)}`` predicted class maps.
        masks_by_patch: ``{patch_id: (H, W)}`` ground-truth class masks.

    Returns:
        ``(preds, target)`` flat int64 arrays over the shared patches.
    """
    ids = sorted(set(preds_by_patch) & set(masks_by_patch))
    preds = np.concatenate([preds_by_patch[i].reshape(-1) for i in ids], axis=0)
    target = np.concatenate([masks_by_patch[i].reshape(-1) for i in ids], axis=0)
    return preds.astype(np.int64), target.astype(np.int64)


def evaluate_dense_predictions(
    name: str,
    preds_by_patch: dict[int, np.ndarray],
    masks_by_patch: dict[int, np.ndarray],
    *,
    label_space: ItaliaLabelSpace,
) -> DenseEvalResult:
    """Score a model's dense predictions at the fine and coarse granularities.

    Args:
        name: The model/combiner name (for the report row).
        preds_by_patch: ``{patch_id: (H, W)}`` predicted fine class maps.
        masks_by_patch: ``{patch_id: (H, W)}`` ground-truth fine class masks.
        label_space: The Italian fine label space.

    Returns:
        A :class:`DenseEvalResult` with the fine + coarse metrics and the
        per-fine-class F1 / IoU (background excluded from both).
    """
    from ml.eval.dense_metrics import DenseConfusionAccumulator

    preds, target = _stack_dense(preds_by_patch, masks_by_patch)
    bg = label_space.background_id

    fine_acc = DenseConfusionAccumulator(label_space.num_classes, ignore_index=bg)
    fine_acc.update(preds, target)
    fine = fine_acc.compute()

    lut, coarse_names = build_coarse_label_space(label_space)
    n_coarse = len(coarse_names)
    coarse_acc = DenseConfusionAccumulator(n_coarse, ignore_index=0)
    coarse_acc.update(lut[preds], lut[target])
    coarse = coarse_acc.compute()

    per_iou = fine_acc.per_class_iou()
    per_f1 = per_class_f1(fine_acc.confusion_matrix(), ignore_index=bg)
    supports = _per_class_support(target, num_classes=label_space.num_classes, ignore_index=bg)
    id_to_leaf = label_space.id_to_leaf()
    per_class = [
        {
            "leaf": id_to_leaf.get(cid, str(cid)),
            "is_new": id_to_leaf.get(cid) in set(label_space.new),
            "f1": round(float(per_f1.get(cid, 0.0)), 4),
            "iou": round(float(per_iou.get(cid, 0.0)), 4),
            "support": int(supports.get(cid, 0)),
        }
        for cid in sorted(supports)
    ]

    result = DenseEvalResult(
        name=name,
        fine_miou=float(fine["miou"]),
        fine_f1_macro=float(fine["f1_macro"]),
        fine_pixel_accuracy=float(fine["pixel_accuracy"]),
        coarse_miou=float(coarse["miou"]),
        coarse_f1_macro=float(coarse["f1_macro"]),
        coarse_pixel_accuracy=float(coarse["pixel_accuracy"]),
        per_class=per_class,
        n_pixels=int((target != bg).sum()),
    )
    logger.info("italia_dense_eval", **result.summary())
    return result


def per_class_f1(confusion: np.ndarray, *, ignore_index: int | None = None) -> dict[int, float]:
    """Compute per-class F1 from a ``(C, C)`` confusion matrix.

    Args:
        confusion: Pixel confusion matrix (rows = truth, cols = pred).
        ignore_index: Class excluded from the result (e.g. background).

    Returns:
        ``{class_id: f1}`` for the classes with support (>0), excluding
        ``ignore_index``.
    """
    conf = confusion.astype(np.float64)
    diag = np.diag(conf)
    row_sum = conf.sum(axis=1)
    col_sum = conf.sum(axis=0)
    precision = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
    recall = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(diag), where=denom > 0)
    out: dict[int, float] = {}
    for cid in range(conf.shape[0]):
        if cid == ignore_index or row_sum[cid] <= 0:
            continue
        out[cid] = float(f1[cid])
    return out


def _per_class_support(
    target: np.ndarray, *, num_classes: int, ignore_index: int | None
) -> dict[int, int]:
    """Per-class pixel support in the ground truth (background excluded)."""
    counts = np.bincount(target[(target >= 0) & (target < num_classes)], minlength=num_classes)
    return {
        cid: int(counts[cid])
        for cid in range(num_classes)
        if cid != ignore_index and counts[cid] > 0
    }


def discard_curve(result: DenseEvalResult) -> list[dict[str, object]]:
    """F1-macro vs number of best fine classes retained (honest discard curve).

    Ranks the fine crop classes by per-class F1 (descending) and reports, for each
    prefix of length ``n``, the macro F1 over those ``n`` best classes. This
    locates the largest subset with F1 > 0.9 (the deployment label space, mirror
    of the ``france-10`` 0.9069 of the Voting-3 on PASTIS). No class is silently
    dropped: the full ranking is returned so the choice of cut-off is explicit.

    Args:
        result: A :class:`DenseEvalResult` carrying the per-class F1.

    Returns:
        A list of ``{"n_classes", "macro_f1", "classes"}`` rows, ``n`` from 1 to
        the number of classes with support, macro F1 over the top-``n``.
    """

    def _f1(row: dict[str, object]) -> float:
        return float(cast("float", row["f1"]))

    ranked = sorted(result.per_class, key=_f1, reverse=True)
    curve: list[dict[str, object]] = []
    for n in range(1, len(ranked) + 1):
        top = ranked[:n]
        macro = float(np.mean([_f1(r) for r in top]))
        curve.append(
            {
                "n_classes": n,
                "macro_f1": round(macro, 4),
                "classes": [r["leaf"] for r in top],
            }
        )
    return curve


def best_subset_over_threshold(
    result: DenseEvalResult, *, threshold: float = 0.9
) -> dict[str, object]:
    """Return the largest top-``n`` subset whose macro F1 stays above ``threshold``.

    Args:
        result: The dense eval result.
        threshold: The macro-F1 floor (default 0.9, the US-079 quality target).

    Returns:
        ``{"n_classes", "macro_f1", "classes"}`` of the largest qualifying prefix,
        or the singleton best class if none reaches the threshold.
    """
    curve = discard_curve(result)
    qualifying = [row for row in curve if float(cast("float", row["macro_f1"])) >= threshold]
    if qualifying:
        return max(qualifying, key=lambda r: int(cast("int", r["n_classes"])))
    return curve[0] if curve else {"n_classes": 0, "macro_f1": 0.0, "classes": []}


def transfer_delta(finetuned: DenseEvalResult, zero_shot: DenseEvalResult) -> dict[str, float]:
    """Quantify the fine-tune gain over the zero-shot French champion.

    Args:
        finetuned: The fine-tuned model/combiner result.
        zero_shot: The zero-shot (no-fine-tune) French-champion result.

    Returns:
        ``{"delta_fine_f1", "delta_coarse_f1", "delta_fine_miou",
        "delta_coarse_miou"}`` (fine-tuned minus zero-shot).
    """
    delta = {
        "delta_fine_f1": round(finetuned.fine_f1_macro - zero_shot.fine_f1_macro, 4),
        "delta_coarse_f1": round(finetuned.coarse_f1_macro - zero_shot.coarse_f1_macro, 4),
        "delta_fine_miou": round(finetuned.fine_miou - zero_shot.fine_miou, 4),
        "delta_coarse_miou": round(finetuned.coarse_miou - zero_shot.coarse_miou, 4),
    }
    logger.info("italia_transfer_delta", **delta)
    return delta


def probs_to_class_map(probs_by_patch: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Argmax each ``(K, H, W)`` post-softmax map into a ``(H, W)`` class map.

    Args:
        probs_by_patch: ``{patch_id: (K, H, W)}`` post-softmax maps.

    Returns:
        ``{patch_id: (H, W)}`` int64 class maps.
    """
    return {pid: probs.argmax(axis=0).astype(np.int64) for pid, probs in probs_by_patch.items()}


def project_parcel_vote_to_dense(
    parcel_probs: dict[str, np.ndarray],
    crop_class_ids: tuple[int, ...],
    parcel_rasters: dict[int, tuple[np.ndarray, dict[int, str]]],
    *,
    num_classes: int,
    patch_px: int = 128,
    background_id: int = 0,
) -> dict[int, np.ndarray]:
    """Project the per-parcel Voting-3 distribution back onto a dense map.

    The champion votes per PARCEL, but the EPIC 5 rubric (and the fine/coarse
    hierarchical eval) is a DENSE metric. This re-paints each parcel's voted
    distribution onto every pixel that belongs to it, using the EuroCrops ParcelID
    rasters (:func:`ml.transfer.dense_to_parcel_italia.load_eurocrops_parcel_rasters`,
    keyed by ``canonical_parcel_id``). A pixel with no parcel (background, or a
    parcel the vote did not score) stays an all-zero column whose ``argmax`` is the
    background id, so it is dropped by the ``ignore_index=background_id`` dense
    metric -- exactly the supervised-pixel convention
    :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` uses.

    The voted distribution lives over the GLOBAL crop-class column space
    (``crop_class_ids``, background excluded), so column ``i`` is scattered onto
    the dense channel ``crop_class_ids[i]`` -- the SAME id the masks
    ``TARGET_<id>.npy`` carry (the Italian label space does not reindex its crop
    ids).

    Args:
        parcel_probs: ``{canonical_parcel_id: (n_crops,)}`` the blended Voting-3
            distribution per parcel over ``crop_class_ids`` (post-softmax).
        crop_class_ids: The global crop class ids the parcel-prob columns map to,
            in column order (background excluded).
        parcel_rasters: ``{patch_id: (parcel_id_map, id_to_canonical)}`` the
            per-patch EuroCrops ParcelID rasters (int surrogate 1-based, 0 =
            background) and their surrogate -> ``canonical_parcel_id`` inverse.
        num_classes: The dense class axis size ``K`` (background channel included,
            so the projected maps are ``(K, patch_px, patch_px)``).
        patch_px: Patch side in pixels (default 128 = PASTIS).
        background_id: The dense background channel id (default 0); its column is
            left at zero for every pixel.

    Returns:
        ``{patch_id: (K, patch_px, patch_px)}`` dense post-softmax maps where every
        pixel carries its parcel's voted distribution (background pixels all-zero).
    """
    col_of_channel = np.asarray(crop_class_ids, dtype=np.int64)
    dense_by_patch: dict[int, np.ndarray] = {}
    n_painted_parcels = 0
    for pid, (parcel_map, id_to_canonical) in parcel_rasters.items():
        dense = np.zeros((num_classes, patch_px, patch_px), dtype=np.float32)
        flat = dense.reshape(num_classes, -1)
        flat_parcel = parcel_map.reshape(-1)
        for surrogate, canonical in id_to_canonical.items():
            row = parcel_probs.get(canonical)
            if row is None:
                continue  # parcel not scored by the vote (e.g. no xgb embedding)
            sel = flat_parcel == surrogate
            if not sel.any():
                continue
            # Scatter the parcel distribution onto the dense crop channels.
            for col, channel in enumerate(col_of_channel):
                if channel == background_id:
                    continue
                flat[channel, sel] = float(row[col])
            n_painted_parcels += 1
        dense_by_patch[pid] = flat.reshape(num_classes, patch_px, patch_px)
    logger.info(
        "parcel_vote_projected_to_dense",
        n_patches=len(dense_by_patch),
        n_painted_parcels=n_painted_parcels,
        num_classes=num_classes,
    )
    return dense_by_patch
