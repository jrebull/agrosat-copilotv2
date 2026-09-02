"""Tests for :mod:`ml.eval.transfer_italia_eval` (US-079 hierarchical eval).

Scores dense Italian predictions at the FINE and COARSE granularities, builds the
honest discard curve, locates the best F1>threshold subset and the transfer delta
(fine-tuned vs zero-shot). The tests use a tiny toy label space and synthetic
dense pred/target maps with KNOWN agreement so the metrics are exact:

- A perfect prediction yields fine F1 == 1 and the background is excluded.
- The fine->coarse LUT collapses conserved leaves to their PASTIS parent and new
  leaves to their agronomic group (a single id maps to the right coarse id).
- ``discard_curve`` ranks classes by F1 and ``best_subset_over_threshold`` returns
  the largest top-n prefix above the floor.
- ``transfer_delta`` is the signed FT - zero-shot gap.

All numpy; reuses the real :mod:`ml.eval.dense_metrics` accumulator (apples to
apples with the segmentation harness). No torch GPU, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.eval.transfer_italia_eval import (
    DenseEvalResult,
    best_subset_over_threshold,
    build_coarse_label_space,
    discard_curve,
    evaluate_dense_predictions,
    per_class_f1,
    probs_to_class_map,
    transfer_delta,
)
from ml.transfer.italia_label_space import ItaliaLabelSpace

# Toy label space: bg + 2 conserved (PASTIS parents) + 2 new (agronomic groups).
_LEAVES = (
    "__background__",
    "common_soft_wheat",  # conserved -> "Soft winter wheat"
    "maize_corn_popcorn",  # conserved -> "Corn"
    "olive",  # new -> "Permanent woody crop"
    "tree_wood_forest",  # new -> "Forest"
)


def _toy_space() -> ItaliaLabelSpace:
    return ItaliaLabelSpace(
        leaves=_LEAVES,
        class_ids=(1, 2, 3, 4),
        conserved=("common_soft_wheat", "maize_corn_popcorn"),
        new=("olive", "tree_wood_forest"),
        leaf_to_pastis={
            "common_soft_wheat": "Soft winter wheat",
            "maize_corn_popcorn": "Corn",
        },
        background_id=0,
    )


def _quadrant_mask() -> np.ndarray:
    """A 4x4 map: four classes 1..4 in the four quadrants, no background."""
    mask = np.zeros((4, 4), dtype=np.int64)
    mask[:2, :2] = 1
    mask[:2, 2:] = 2
    mask[2:, :2] = 3
    mask[2:, 2:] = 4
    return mask


# --------------------------------------------------------------------------- #
# build_coarse_label_space.
# --------------------------------------------------------------------------- #
def test_coarse_lut_collapses_to_right_buckets() -> None:
    """Each fine id maps to the coarse id of its bucket; background stays 0."""
    space = _toy_space()
    lut, coarse_names = build_coarse_label_space(space)
    assert lut.shape == (space.num_classes,)
    assert lut[0] == 0 and coarse_names[0] == "__background__"
    # Conserved fine ids -> their PASTIS parent buckets (distinct coarse ids).
    assert coarse_names[lut[1]] == "Soft winter wheat"
    assert coarse_names[lut[2]] == "Corn"
    # New fine ids -> their agronomic groups.
    assert coarse_names[lut[3]] == "Permanent woody crop"
    assert coarse_names[lut[4]] == "Forest"
    # All four leaves are in distinct coarse buckets here.
    assert len({lut[1], lut[2], lut[3], lut[4]}) == 4


def test_coarse_lut_merges_leaves_sharing_a_bucket() -> None:
    """Two new leaves in the same coarse group share one coarse id."""
    space = ItaliaLabelSpace(
        leaves=("__background__", "tree_wood_forest", "sweet_chestnuts"),
        class_ids=(1, 2),
        conserved=(),
        new=("tree_wood_forest", "sweet_chestnuts"),
        leaf_to_pastis={},
        background_id=0,
    )
    lut, coarse_names = build_coarse_label_space(space)
    # Both collapse to "Forest" -> same coarse id.
    assert lut[1] == lut[2]
    assert coarse_names[lut[1]] == "Forest"


# --------------------------------------------------------------------------- #
# evaluate_dense_predictions.
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_one_fine_and_coarse() -> None:
    """A perfect dense prediction yields fine + coarse F1/mIoU == 1."""
    space = _toy_space()
    mask = _quadrant_mask()
    preds = {0: mask.copy()}
    masks = {0: mask.copy()}
    res = evaluate_dense_predictions("perfect", preds, masks, label_space=space)
    assert res.fine_f1_macro == pytest.approx(1.0)
    assert res.fine_miou == pytest.approx(1.0)
    assert res.coarse_f1_macro == pytest.approx(1.0)
    # Background excluded: n_pixels counts only the supervised pixels (all 16).
    assert res.n_pixels == 16
    # Per-class rows exist for the 4 crop classes (no background row).
    leaves = {row["leaf"] for row in res.per_class}
    assert leaves == {"common_soft_wheat", "maize_corn_popcorn", "olive", "tree_wood_forest"}
    assert all(row["f1"] == pytest.approx(1.0) for row in res.per_class)


def test_per_class_flags_new_classes() -> None:
    """The per-class rows mark the new Mediterranean leaves with is_new=True."""
    space = _toy_space()
    mask = _quadrant_mask()
    res = evaluate_dense_predictions("m", {0: mask}, {0: mask}, label_space=space)
    new_flags = {row["leaf"]: row["is_new"] for row in res.per_class}
    assert new_flags["olive"] is True
    assert new_flags["tree_wood_forest"] is True
    assert new_flags["common_soft_wheat"] is False


def test_imperfect_prediction_lowers_specific_class_f1() -> None:
    """Confusing one class with another drops exactly that class's F1 below 1."""
    space = _toy_space()
    mask = _quadrant_mask()
    wrong = mask.copy()
    wrong[wrong == 4] = 3  # predict class 3 where truth is class 4
    res = evaluate_dense_predictions("imperfect", {0: wrong}, {0: mask}, label_space=space)
    f1_by_leaf = {row["leaf"]: row["f1"] for row in res.per_class}
    assert f1_by_leaf["tree_wood_forest"] == pytest.approx(0.0)  # class 4 never recalled
    assert f1_by_leaf["common_soft_wheat"] == pytest.approx(1.0)  # untouched
    assert res.fine_f1_macro < 1.0


# --------------------------------------------------------------------------- #
# per_class_f1 from a confusion matrix.
# --------------------------------------------------------------------------- #
def test_per_class_f1_excludes_ignore_and_unsupported() -> None:
    """per_class_f1 skips the ignore index and classes with no ground truth."""
    # 3x3 confusion: class 0 ignored, class 1 perfect, class 2 absent (row sum 0).
    conf = np.array([[5, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.int64)
    f1 = per_class_f1(conf, ignore_index=0)
    assert 0 not in f1  # ignored
    assert 2 not in f1  # no support
    assert f1[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# discard curve + best subset.
# --------------------------------------------------------------------------- #
def _result_with_f1(f1s: dict[str, float]) -> DenseEvalResult:
    per_class = [
        {"leaf": leaf, "is_new": False, "f1": f1, "iou": f1, "support": 100}
        for leaf, f1 in f1s.items()
    ]
    return DenseEvalResult(
        name="x",
        fine_miou=0.0,
        fine_f1_macro=float(np.mean(list(f1s.values()))),
        fine_pixel_accuracy=0.0,
        coarse_miou=0.0,
        coarse_f1_macro=0.0,
        coarse_pixel_accuracy=0.0,
        per_class=per_class,
    )


def test_discard_curve_is_descending_prefix_macro() -> None:
    """The curve ranks classes by F1 and reports the macro over each top-n prefix."""
    res = _result_with_f1({"a": 0.95, "b": 0.92, "c": 0.40, "d": 0.10})
    curve = discard_curve(res)
    assert [row["n_classes"] for row in curve] == [1, 2, 3, 4]
    # Top-1 is the best class; macro is non-increasing as weaker classes join.
    assert curve[0]["macro_f1"] == pytest.approx(0.95)
    macros = [row["macro_f1"] for row in curve]
    assert macros == sorted(macros, reverse=True)
    assert curve[0]["classes"] == ["a"]


def test_best_subset_over_threshold_picks_largest_qualifying_prefix() -> None:
    """The largest top-n prefix with macro F1 >= threshold is returned."""
    res = _result_with_f1({"a": 0.95, "b": 0.92, "c": 0.40, "d": 0.10})
    best = best_subset_over_threshold(res, threshold=0.9)
    assert best["n_classes"] == 2  # a+b macro 0.935 >= 0.9; adding c drops below
    assert set(best["classes"]) == {"a", "b"}


def test_best_subset_falls_back_to_best_single_when_none_qualify() -> None:
    """If no prefix reaches the floor, the single best class is returned."""
    res = _result_with_f1({"a": 0.5, "b": 0.4})
    best = best_subset_over_threshold(res, threshold=0.9)
    assert best["n_classes"] == 1
    assert best["classes"] == ["a"]


# --------------------------------------------------------------------------- #
# transfer delta + probs_to_class_map.
# --------------------------------------------------------------------------- #
def test_transfer_delta_is_signed_gap() -> None:
    """The delta is fine-tuned minus zero-shot for each metric."""
    ft = DenseEvalResult("ft", 0.6, 0.7, 0.8, 0.65, 0.75, 0.85)
    zs = DenseEvalResult("zs", 0.3, 0.4, 0.5, 0.35, 0.45, 0.55)
    delta = transfer_delta(ft, zs)
    assert delta["delta_fine_f1"] == pytest.approx(0.3)
    assert delta["delta_coarse_f1"] == pytest.approx(0.3)
    assert delta["delta_fine_miou"] == pytest.approx(0.3)
    assert delta["delta_coarse_miou"] == pytest.approx(0.3)


def test_probs_to_class_map_argmaxes() -> None:
    """probs_to_class_map argmaxes each (K, H, W) post-softmax map to (H, W)."""
    probs = np.zeros((3, 2, 2), dtype=np.float32)
    probs[0, 0, 0] = 1.0
    probs[2, 1, 1] = 1.0
    probs[1, 0, 1] = 1.0
    probs[1, 1, 0] = 1.0
    out = probs_to_class_map({7: probs})
    assert out[7].shape == (2, 2)
    assert out[7].dtype == np.int64
    np.testing.assert_array_equal(out[7], np.array([[0, 1], [1, 2]]))
