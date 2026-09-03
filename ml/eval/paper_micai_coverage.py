"""Quality-coverage frontier for the MICAI manuscript, rebuilt after the blind audit.

The first implementation of this experiment (phase 2) was retired because three
independent defects each invalidated its headline. This module exists to make those
three failures impossible to repeat, and each is answered by name here:

1. **The estimand was not aligned.** Retiring classes averaged the macro over the K
   best classes while the abstention baseline averaged over the up-to-eighteen present,
   so the reported delta was mostly the denominator moving. Every comparison in this
   module scores both mechanisms over the SAME class set, and reports the native
   (own-legend) view separately and clearly labelled as not comparable.
2. **One mechanism read the answer.** Delivery was decided by the parcel's true label,
   which no deployment knows. Here a mechanism delivers on what it can observe: the
   unrestricted argmax falling inside the promised legend, or the confidence rank.
3. **The interval was not paired.** Two independent resamples produced a non-degenerate
   interval for comparing an object with itself. Here one index per block is drawn and
   both mechanisms are recomputed on it, so the full-legend row must return a
   degenerate interval; if it does not, the harness is broken again.

A fourth correction the audit raised: parcels inside a PASTIS patch share an image and
are not independent, so a cluster bootstrap over patches is reported next to the
parcel-level one rather than instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, f1_score

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

logger = structlog.get_logger(__name__)

NUM_CLASSES = 18


@dataclass(frozen=True)
class BlockPoint:
    """One mechanism measured on one spatial block at one legend size.

    Attributes:
        mechanism: Human-readable name of the mechanism.
        k: Size of the promised legend.
        block: Index of the spatial block.
        legend: Classes the product promises in this block.
        delivered: Boolean mask over the block's parcels.
        emitted: Predicted labels for the block's parcels.
        aligned_f1: Macro-F1 over the shared legend, the comparable number.
        native_f1: Macro-F1 over the classes the mechanism itself promises.
        accuracy: Accuracy over the delivered parcels.
    """

    mechanism: str
    k: int
    block: int
    legend: tuple[int, ...]
    delivered: np.ndarray
    emitted: np.ndarray
    aligned_f1: float
    native_f1: float
    accuracy: float


def macro_over(labels: np.ndarray, predicted: np.ndarray, classes: Sequence[int]) -> float:
    """Macro-F1 restricted to the classes that are both promised and present.

    A promised class absent from this block would enter the average as a zero and
    report a failure that never happened, so the average runs over the intersection.

    Args:
        labels: Ground-truth labels of the delivered parcels.
        predicted: Predicted labels of the delivered parcels.
        classes: Class set the average runs over.

    Returns:
        The macro-F1, or ``0.0`` when the intersection is empty.
    """
    if labels.size == 0:
        return 0.0
    evaluated = sorted(set(classes) & set(labels.tolist()))
    if not evaluated:
        return 0.0
    return float(f1_score(labels, predicted, labels=evaluated, average="macro", zero_division=0))


def legend_by_f1(
    labels: np.ndarray, predicted: np.ndarray, train_pos: np.ndarray, k: int
) -> tuple[int, ...]:
    """Pick the K classes with the best binary F1 on the blocks not being measured.

    Args:
        labels: Ground-truth labels for the whole universe.
        predicted: Unrestricted predictions for the whole universe.
        train_pos: Positional indices of the blocks used to decide.
        k: Legend size.

    Returns:
        The promised classes, sorted.
    """
    ranked = sorted(
        (
            (
                float(
                    f1_score(
                        labels[train_pos] == c,
                        predicted[train_pos] == c,
                        average="binary",
                        zero_division=0,
                    )
                ),
                c,
            )
            for c in range(NUM_CLASSES)
        ),
        reverse=True,
    )
    return tuple(sorted(c for _, c in ranked[:k]))


def legend_by_support(labels: np.ndarray, train_pos: np.ndarray, k: int) -> tuple[int, ...]:
    """Pick the K most frequent classes on the blocks not being measured.

    This is the criterion the team reported having used to deploy: the six classes it
    dropped "had very little sample" and "pulled the macro-F1 down". Measuring it makes
    the comparison one against practice rather than against a rule invented for the paper.

    Args:
        labels: Ground-truth labels for the whole universe.
        train_pos: Positional indices of the blocks used to decide.
        k: Legend size.

    Returns:
        The promised classes, sorted.
    """
    counts = np.bincount(labels[train_pos], minlength=NUM_CLASSES)
    ranked = sorted(((int(counts[c]), c) for c in range(NUM_CLASSES)), reverse=True)
    return tuple(sorted(c for _, c in ranked[:k]))


def _emit_restricted(proba: np.ndarray, rows: np.ndarray, legend: Sequence[int]) -> np.ndarray:
    """Argmax restricted to the promised legend.

    Args:
        proba: Posterior matrix for the whole universe.
        rows: Positional indices to emit for.
        legend: Classes the product is allowed to emit.

    Returns:
        Predicted labels, one per row.
    """
    columns = np.asarray(legend, dtype=int)
    emitted: np.ndarray = columns[proba[np.ix_(rows, columns)].argmax(axis=1)]
    return emitted


def frontier(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    k_values: Sequence[int],
    *,
    legend_fn: Callable[[np.ndarray, int], tuple[int, ...]],
    mechanism: str,
) -> list[BlockPoint]:
    """Measure a legend-shrinking mechanism block by block.

    The legend of a block is chosen on the OTHER blocks. A parcel is delivered when the
    predictor's unrestricted argmax lands inside that legend, which is observable at
    inference time; the retired implementation used the true label instead.

    Args:
        proba: Posterior matrix for the whole universe.
        labels: Ground-truth labels.
        splits: ``(train_pos, test_pos)`` index pairs, one per block.
        k_values: Legend sizes to evaluate.
        legend_fn: Takes ``(train_pos, k)`` and returns the promised classes.
        mechanism: Name recorded on every point.

    Returns:
        One :class:`BlockPoint` per ``(k, block)``.
    """
    free = proba.argmax(axis=1)
    points: list[BlockPoint] = []
    for k in k_values:
        for block, (train_pos, test_pos) in enumerate(splits):
            if train_pos.size == 0 or test_pos.size == 0:
                continue
            legend = legend_fn(train_pos, k)
            columns = np.asarray(legend, dtype=int)
            delivered = np.isin(free[test_pos], columns)
            emitted = _emit_restricted(proba, test_pos, legend)
            truth = labels[test_pos]
            points.append(
                BlockPoint(
                    mechanism=mechanism,
                    k=k,
                    block=block,
                    legend=legend,
                    delivered=delivered,
                    emitted=emitted,
                    aligned_f1=macro_over(truth[delivered], emitted[delivered], legend),
                    native_f1=macro_over(truth[delivered], emitted[delivered], legend),
                    accuracy=float(accuracy_score(truth[delivered], emitted[delivered]))
                    if delivered.any()
                    else 0.0,
                )
            )
    return points


def confidence_baseline(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: Sequence[BlockPoint],
) -> list[BlockPoint]:
    """Deliver the most confident parcels, matching a reference mechanism's count.

    The legend stays complete, so this mechanism emits over all eighteen classes; what
    shrinks is how many parcels it answers. Its count comes from how many the reference
    delivered in that same block, and its threshold is a quantile of its own confidences,
    so no label is read at any point.

    Args:
        proba: Posterior matrix for the whole universe.
        labels: Ground-truth labels.
        splits: ``(train_pos, test_pos)`` index pairs, one per block.
        reference: Points of the mechanism whose coverage is being matched.

    Returns:
        One :class:`BlockPoint` per reference point, in the same order.
    """
    free = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    points: list[BlockPoint] = []
    for ref in reference:
        _, test_pos = splits[ref.block]
        block_conf = confidence[test_pos]
        order = np.argsort(-block_conf, kind="stable")
        delivered = np.zeros(test_pos.size, dtype=bool)
        delivered[order[: int(ref.delivered.sum())]] = True
        emitted = free[test_pos]
        truth = labels[test_pos]
        points.append(
            BlockPoint(
                mechanism="rechazo por confianza",
                k=ref.k,
                block=ref.block,
                legend=tuple(range(NUM_CLASSES)),
                delivered=delivered,
                emitted=emitted,
                aligned_f1=macro_over(truth[delivered], emitted[delivered], ref.legend),
                native_f1=macro_over(
                    truth[delivered], emitted[delivered], sorted(set(truth[delivered].tolist()))
                ),
                accuracy=float(accuracy_score(truth[delivered], emitted[delivered]))
                if delivered.any()
                else 0.0,
            )
        )
    return points


def no_mechanism_reference(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: Sequence[BlockPoint],
) -> list[BlockPoint]:
    """Score the untouched predictor over the same legend, delivering everything.

    This is the control the audit asked for and the retired experiment lacked: if the
    macro over the K easiest classes is already high WITHOUT retiring or rejecting
    anything, then a mechanism that reports a similar number has not bought the quality,
    the class set has.

    Args:
        proba: Posterior matrix for the whole universe.
        labels: Ground-truth labels.
        splits: ``(train_pos, test_pos)`` index pairs, one per block.
        reference: Points whose legend and block are reused.

    Returns:
        One :class:`BlockPoint` per reference point.
    """
    free = proba.argmax(axis=1)
    points: list[BlockPoint] = []
    for ref in reference:
        _, test_pos = splits[ref.block]
        delivered = np.ones(test_pos.size, dtype=bool)
        emitted = free[test_pos]
        truth = labels[test_pos]
        points.append(
            BlockPoint(
                mechanism="sin mecanismo",
                k=ref.k,
                block=ref.block,
                legend=ref.legend,
                delivered=delivered,
                emitted=emitted,
                aligned_f1=macro_over(truth, emitted, ref.legend),
                native_f1=macro_over(truth, emitted, ref.legend),
                accuracy=float(accuracy_score(truth, emitted)),
            )
        )
    return points


def paired_interval(
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    left: Sequence[BlockPoint],
    right: Sequence[BlockPoint],
    *,
    n_boot: int,
    random_state: int,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap the paired difference of two mechanisms' mean block macro-F1.

    One index is drawn per block and BOTH mechanisms are rescored on it, which is what
    makes the interval paired. When ``clusters`` is given the resampling unit is the
    cluster (a PASTIS patch) rather than the parcel, because parcels inside a patch share
    an image and are not independent.

    Args:
        labels: Ground-truth labels for the whole universe.
        splits: ``(train_pos, test_pos)`` index pairs, one per block.
        left: Points of the mechanism being tested, one per block.
        right: Points of the comparator, aligned with ``left``.
        n_boot: Number of resamples.
        random_state: Seed.
        clusters: Optional cluster id per parcel of the whole universe.

    Returns:
        Observed delta, percentile interval, a two-sided bootstrap p-value and the
        per-block deltas.
    """
    rng = np.random.default_rng(random_state)
    observed = float(np.mean([p.aligned_f1 for p in left]) - np.mean([p.aligned_f1 for p in right]))

    prepared = []
    for a, b in zip(left, right, strict=True):
        _, test_pos = splits[a.block]
        truth = labels[test_pos]
        group = None if clusters is None else clusters[test_pos]
        prepared.append((truth, group, a, b))

    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        left_scores, right_scores = [], []
        for truth, group, a, b in prepared:
            if group is None:
                idx = rng.integers(0, truth.size, size=truth.size)
            else:
                unique = np.unique(group)
                picked = unique[rng.integers(0, unique.size, size=unique.size)]
                idx = np.concatenate([np.flatnonzero(group == g) for g in picked])
            left_scores.append(
                macro_over(truth[idx][a.delivered[idx]], a.emitted[idx][a.delivered[idx]], a.legend)
            )
            right_scores.append(
                macro_over(truth[idx][b.delivered[idx]], b.emitted[idx][b.delivered[idx]], a.legend)
            )
        draws[i] = float(np.mean(left_scores) - np.mean(right_scores))

    low, high = np.percentile(draws, [2.5, 97.5])
    below = float((draws <= 0).mean())
    above = float((draws >= 0).mean())
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "excluye_cero": float(low > 0 or high < 0),
        "p_bootstrap": float(min(1.0, 2 * min(below, above))),
        "deltas_por_bloque": [
            float(a.aligned_f1 - b.aligned_f1) for a, b in zip(left, right, strict=True)
        ],
    }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of a family of p-values.

    Args:
        p_values: Raw p-values of one family.

    Returns:
        Adjusted p-values in the input order, each capped at one and made monotone.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * p_values[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted
