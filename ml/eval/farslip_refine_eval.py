"""Evaluate the US-080 FarSLIP refinement: F1-macro Voting-3 vs Voting-3+refine.

Measures, on the real PASTIS-R fold-5 OOF, how much the conditional FarSLIP
second stage (:mod:`ml.agent.refine`) moves the F1-macro of the deployment champion
-- globally and on the subset of parcels where the refinement actually fired
(AC5/AC6). REAL VALUES ONLY: the FarSLIP scoring is injected, so the pure
computation is unit-tested offline; the live run needs the FarSLIP model + the
per-parcel chips (the documented blocker) and reports the delta as measured,
positive or not.

The core (:func:`f1_macro`, :func:`run_refine_eval`) is pure -- it consumes the
Voting-3 posteriors, the ground truth and a per-parcel FarSLIP scorer -- so a test
drives it with fakes. :func:`main` wires the REAL inputs (the cached Voting-3 OOF,
the reconstructed GT, the FarSLIP zero-shot head over the chips) and logs to MLflow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ml.eval.class_remap import LabelSpace

logger = structlog.get_logger(__name__)

__all__ = [
    "RefineEvalReport",
    "build_farslip_zeroshot_scorer",
    "f1_macro",
    "load_real_inputs",
    "run_real_eval",
    "run_refine_eval",
]

#: A per-parcel FarSLIP scorer: ``canonical_id -> {class_name: score}`` (or ``None``
#: when the chip / FarSLIP signal is unavailable for that parcel).
FarSLIPScorer = Callable[[str], dict[str, float] | None]

#: FarSLIP fold-5 zero-shot OOF: the REAL precomputed FarSLIP signal that already
#: covers the Voting-3 fold-5 parcels (same canonical id), so the delta-F1 runs
#: without the FarSLIP model or the chips.
_FARSLIP_ZEROSHOT_OOF: str = "oof_parcel_farslip-zeroshot_fold5.parquet"


def f1_macro(y_true: list[str], y_pred: list[str], *, labels: list[str] | None = None) -> float:
    """Compute the macro-averaged F1 over ``labels`` (pure, no sklearn needed).

    Args:
        y_true: Ground-truth class names, aligned with ``y_pred``.
        y_pred: Predicted class names.
        labels: Class set to average over; inferred from ``y_true`` when ``None``.

    Returns:
        The unweighted mean per-class F1 in ``[0, 1]`` (``0.0`` for empty input).
    """
    if not y_true:
        return 0.0
    classes = labels if labels is not None else sorted(set(y_true))
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


class RefineEvalReport(dict):
    """Plain dict report (keys documented in :func:`run_refine_eval`)."""


def run_refine_eval(
    voting_posteriors: Mapping[str, dict[str, float]],
    ground_truth: Mapping[str, str],
    scorer: FarSLIPScorer,
    *,
    member_predictions: Mapping[str, dict[str, str]] | None = None,
    alpha: float = 0.4,
    margin_tau: float = 0.15,
) -> RefineEvalReport:
    """Compare Voting-3 vs Voting-3+refine F1-macro over the labelled parcels.

    For every parcel with a ground-truth label, the Voting-3 argmax is the baseline
    prediction; the gated FarSLIP refinement (:func:`ml.agent.refine.apply_refinement`)
    may re-rank it. F1-macro is computed before and after, globally and on the
    "fired" subset (the parcels where the refinement actually engaged).

    Args:
        voting_posteriors: ``canonical_id -> {class_name: probability}`` Voting-3
            posterior (restricted to the active label-space).
        ground_truth: ``canonical_id -> true class name``.
        scorer: Per-parcel FarSLIP scorer (injected; a fake in tests).
        member_predictions: Optional ``canonical_id -> {member: argmax class}`` for
            the disagreement trigger.
        alpha: Convex weight of the FarSLIP signal when the refinement fires.
        margin_tau: Uncertainty margin threshold for the trigger.

    Returns:
        A :class:`RefineEvalReport` with ``f1_before`` / ``f1_after`` / ``delta_f1``
        (global), ``f1_before_fired`` / ``f1_after_fired`` / ``delta_f1_fired``
        (the fired subset), ``n_parcels`` / ``n_fired`` / ``n_changed``.
    """
    from ml.agent.refine import apply_refinement

    labels = sorted({cls for post in voting_posteriors.values() for cls in post})
    y_true: list[str] = []
    y_before: list[str] = []
    y_after: list[str] = []
    fired_idx: list[int] = []
    n_changed = 0

    for canonical_id, truth in ground_truth.items():
        posterior = voting_posteriors.get(canonical_id)
        if not posterior:
            continue
        result = apply_refinement(
            dict(posterior),
            scorer(canonical_id),
            member_predictions=member_predictions.get(canonical_id) if member_predictions else None,
            alpha=alpha,
            margin_tau=margin_tau,
        )
        y_true.append(truth)
        y_before.append(result.top_class_before)
        y_after.append(result.top_class_after)
        if result.refined:
            fired_idx.append(len(y_true) - 1)
            if result.top_class_after != result.top_class_before:
                n_changed += 1

    f1_before = f1_macro(y_true, y_before, labels=labels)
    f1_after = f1_macro(y_true, y_after, labels=labels)
    fired_true = [y_true[i] for i in fired_idx]
    fired_before = [y_before[i] for i in fired_idx]
    fired_after = [y_after[i] for i in fired_idx]
    f1_before_fired = f1_macro(fired_true, fired_before, labels=labels)
    f1_after_fired = f1_macro(fired_true, fired_after, labels=labels)

    report = RefineEvalReport(
        n_parcels=len(y_true),
        n_fired=len(fired_idx),
        n_changed=n_changed,
        f1_before=round(f1_before, 4),
        f1_after=round(f1_after, 4),
        delta_f1=round(f1_after - f1_before, 4),
        f1_before_fired=round(f1_before_fired, 4),
        f1_after_fired=round(f1_after_fired, 4),
        delta_f1_fired=round(f1_after_fired - f1_before_fired, 4),
    )
    logger.info("farslip_refine_eval_done", **report)
    return report


# ---------------------------------------------------------------------------
# REAL inputs: the precomputed FarSLIP zero-shot OOF already covers the Voting-3
# fold-5 parcels, so the delta-F1 runs with no FarSLIP model and no chips.
# ---------------------------------------------------------------------------
def _restricted_names(proba: np.ndarray, label_space: LabelSpace) -> dict[str, float]:
    """Restrict an 18-class posterior to the label-space and key it by class name."""
    from ml.eval.class_remap import restrict_posterior

    restricted = restrict_posterior(proba, label_space)
    return {label_space.class_names.get(cid, str(cid)): float(p) for cid, p in restricted.items()}


def build_farslip_zeroshot_scorer(label_space: LabelSpace) -> FarSLIPScorer:
    """Build the REAL per-parcel FarSLIP scorer from the zero-shot fold-5 OOF.

    The precomputed FarSLIP zero-shot posterior is restricted to the active
    label-space and renormalized, indexed by canonical id -- the real FarSLIP signal
    with no model / no chips. NOTE: this OOF used FarSLIP's own class prompts; the
    US-080 variant that re-scores FarSLIP with the LLM phenology prompts is the next
    step (needs the FarSLIP image embeddings for fold-5, not yet keyed to the OOF).

    Args:
        label_space: The active :class:`~ml.eval.class_remap.LabelSpace`.

    Returns:
        A :data:`FarSLIPScorer` (``canonical_id -> {class_name: score}``).

    Raises:
        FileNotFoundError: if the FarSLIP zero-shot OOF parquet is absent.
    """
    import numpy as np
    import polars as pl

    from ml.agent.tools import classify
    from ml.utils.parcel_id import canonical_parcel_id
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    path = classify._OOF_DIR / _FARSLIP_ZEROSHOT_OOF
    if not path.exists():
        raise FileNotFoundError(
            f"FarSLIP zero-shot OOF missing: {path}. Run `dvc pull ml/eval/oof`."
        )
    frame = canonical_parcel_id(pl.read_parquet(path), col="canonical_parcel_id")
    by_id: dict[str, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        probs = np.asarray([row[c] for c in PROB_COLUMNS], dtype=np.float64)
        by_id[row["canonical_parcel_id"]] = _restricted_names(probs, label_space)

    def scorer(canonical_id: str) -> dict[str, float] | None:
        return by_id.get(canonical_id)

    return scorer


def load_real_inputs(
    label_space: LabelSpace,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]], dict[str, str]]:
    """Build the REAL Voting-3 posteriors, member predictions and GT (label-space names).

    Args:
        label_space: The active :class:`~ml.eval.class_remap.LabelSpace`.

    Returns:
        ``(voting_posteriors, member_predictions, ground_truth)`` keyed by the fold-5
        canonical id; GT keeps only parcels whose true class is in the label-space.
    """
    import numpy as np

    from ml.agent.tools import classify

    voting = classify._load_voting_three()
    voting_posteriors: dict[str, dict[str, float]] = {}
    member_predictions: dict[str, dict[str, str]] = {}
    for cid, member_rows in voting.member_probs_by_id.items():
        proba = voting.posterior_for_parcel(cid)
        if proba is None:
            continue
        voting_posteriors[cid] = _restricted_names(proba, label_space)
        preds: dict[str, str] = {}
        for i, member in enumerate(classify._VOTING_MEMBERS):
            names = _restricted_names(np.asarray(member_rows[i], dtype=np.float64), label_space)
            if names:
                preds[member] = max(names, key=lambda k: names[k])
        member_predictions[cid] = preds

    gt_frame = classify._build_parcel_ground_truth(list(voting_posteriors))
    ground_truth: dict[str, str] = {}
    for row in gt_frame.iter_rows(named=True):
        name = label_space.class_names.get(int(row["label"]))
        if name is not None:
            ground_truth[row["canonical_parcel_id"]] = name
    logger.info(
        "farslip_refine_inputs_loaded",
        n_voting=len(voting_posteriors),
        n_gt=len(ground_truth),
    )
    return voting_posteriors, member_predictions, ground_truth


def run_real_eval(*, alpha: float = 0.4, margin_tau: float = 0.15) -> RefineEvalReport:
    """Run the REAL delta-F1 (Voting-3 vs Voting-3 + FarSLIP-zeroshot) over fold-5.

    Ties the real Voting-3 posteriors + member predictions + GT to the real FarSLIP
    zero-shot scorer and runs :func:`run_refine_eval`. No FarSLIP model / no chips
    (the signal is the precomputed zero-shot OOF). Reports the delta as measured.

    Args:
        alpha: Convex weight of the FarSLIP signal when the refinement fires.
        margin_tau: Uncertainty margin threshold for the trigger.

    Returns:
        The :class:`RefineEvalReport`.
    """
    from ml.eval.class_remap import get_label_space

    label_space = get_label_space("france-9")
    voting_posteriors, member_predictions, ground_truth = load_real_inputs(label_space)
    scorer = build_farslip_zeroshot_scorer(label_space)
    return run_refine_eval(
        voting_posteriors,
        ground_truth,
        scorer,
        member_predictions=member_predictions,
        alpha=alpha,
        margin_tau=margin_tau,
    )
