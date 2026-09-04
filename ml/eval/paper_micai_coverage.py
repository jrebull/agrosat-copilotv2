"""Quality-coverage frontier for the MICAI manuscript, after three rounds of audit.

The phase-2 implementation was retired for three independent defects. This module was
written to make them impossible to repeat — and **two of the three came back in a
different form**, which a third-round external audit found in this file after the
project had already declared them closed. The current version answers all three, and
each repair is expressed as a REQUIRED PARAMETER, because every one of these defects
survived as a silent default:

1. **The estimand was not aligned.** Retiring classes averaged the macro over the K best
   classes while the abstention baseline averaged over the up-to-eighteen present, so the
   reported delta was mostly the denominator moving. The first repair scored both
   mechanisms over the same legend — and then intersected it with the ground truth OF THE
   DELIVERED PARCELS, which is mechanism-dependent, so the denominator moved again.
   :func:`macro_over` now requires ``presentes``, a property of the BLOCK.
2. **One mechanism read the answer.** Delivery was decided by the parcel's true label,
   which no deployment knows. A mechanism now delivers on what it can observe. And the
   confidence baseline's THRESHOLD comes from the training blocks: the first repair
   ranked the evaluated block's own confidences to match the reference count exactly,
   which is choosing the operating point inside the block that scores it.
3. **The interval was not paired, and then it was paired at the wrong unit.** Two
   independent resamples once produced a non-degenerate interval for comparing an object
   with itself. The repair paired them — but resampled PARCELS inside each block, turning
   five spatial blocks into sixteen thousand pretend replicates.
   :func:`paired_interval` now requires ``unidad``; ``"bloque"`` is the estimand-level
   interval, and the parcel and cluster bootstraps stay available, labelled as the
   narrower descriptive question they actually answer.

Everything this module produced before these repairs is exploratory and stays labelled as
such. Regenerating those artefacts is the artefact step of US-124 and US-125.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from scipy import stats
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


def presentes_en_bloque(labels: np.ndarray) -> tuple[int, ...]:
    """Classes present in a block, computed once from its FULL ground truth.

    This exists so that no caller is tempted to derive the class universe from the parcels a
    mechanism happened to deliver. It is a property of the block, not of the mechanism.

    Args:
        labels: Ground-truth labels of every parcel in the block.

    Returns:
        The sorted classes present in the block.
    """
    return tuple(sorted(set(labels.tolist())))


def macro_over(
    labels: np.ndarray,
    predicted: np.ndarray,
    classes: Sequence[int],
    *,
    presentes: Sequence[int],
) -> float:
    """Macro-F1 over the promised classes that exist in the block.

    A promised class absent from the block would enter the average as a zero and report a
    failure that never happened, so the average runs over the intersection with what the
    block contains.

    **`presentes` is required and comes from the block, never from the delivered parcels.**
    That was defect 1 of the external audit, and it survived the first repair: the previous
    version intersected with ``set(labels)``, and ``labels`` was the ground truth OF THE
    DELIVERED SUBSET, which differs between mechanisms. Two mechanisms were averaged over
    two different class sets and the difference was reported as quality. It is the same
    moving denominator the article denounces, one level down, and it is why this parameter
    has no default: a default here is the defect coming back silently.

    Args:
        labels: Ground-truth labels of the delivered parcels.
        predicted: Predicted labels of the delivered parcels.
        classes: Class set the average is meant to run over.
        presentes: Classes present in the whole block, from :func:`presentes_en_bloque`.

    Returns:
        The macro-F1, or **NaN** when the estimand is undefined: nothing delivered, or no promised
        class present in the block. It used to return ``0.0``, which is a legitimate-looking value
        for a perfect failure and averages into the mean as if it had been measured. An undefined
        cell has to be visible to whoever aggregates it.
    """
    if labels.size == 0:
        return float("nan")
    evaluated = sorted(set(classes) & set(presentes))
    if not evaluated:
        return float("nan")
    return float(f1_score(labels, predicted, labels=evaluated, average="macro", zero_division=0))


def legend_by_f1(
    labels: np.ndarray,
    predicted: np.ndarray,
    train_pos: np.ndarray,
    k: int,
    *,
    num_classes: int = NUM_CLASSES,
) -> tuple[int, ...]:
    """Pick the K classes with the best binary F1 on the blocks not being measured.

    Args:
        labels: Ground-truth labels for the whole universe.
        predicted: Unrestricted predictions for the whole universe.
        train_pos: Positional indices of the blocks used to decide.
        k: Legend size.
        num_classes: Size of the label space; the second dataset has nine, not eighteen.

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
            for c in range(num_classes)
        ),
        reverse=True,
    )
    return tuple(sorted(c for _, c in ranked[:k]))


def legend_by_support(
    labels: np.ndarray, train_pos: np.ndarray, k: int, *, num_classes: int = NUM_CLASSES
) -> tuple[int, ...]:
    """Pick the K most frequent classes on the blocks not being measured.

    This is the criterion the team reported having used to deploy: the six classes it
    dropped "had very little sample" and "pulled the macro-F1 down". Measuring it makes
    the comparison one against practice rather than against a rule invented for the paper.

    Args:
        labels: Ground-truth labels for the whole universe.
        train_pos: Positional indices of the blocks used to decide.
        k: Legend size.
        num_classes: Size of the label space.

    Returns:
        The promised classes, sorted.
    """
    counts = np.bincount(labels[train_pos], minlength=num_classes)
    ranked = sorted(((int(counts[c]), c) for c in range(num_classes)), reverse=True)
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
            presentes = presentes_en_bloque(truth)
            points.append(
                BlockPoint(
                    mechanism=mechanism,
                    k=k,
                    block=block,
                    legend=legend,
                    delivered=delivered,
                    emitted=emitted,
                    aligned_f1=macro_over(
                        truth[delivered], emitted[delivered], legend, presentes=presentes
                    ),
                    native_f1=macro_over(
                        truth[delivered], emitted[delivered], legend, presentes=presentes
                    ),
                    accuracy=float(accuracy_score(truth[delivered], emitted[delivered]))
                    if delivered.any()
                    else 0.0,
                )
            )
    return points


def umbral_desde_entrenamiento(
    proba: np.ndarray, train_pos: np.ndarray, legend: Sequence[int]
) -> float:
    """Confidence threshold that matches a legend's delivery rate, computed on TRAINING only.

    Both halves of the operating point come from the training blocks, and that is the whole
    point. The first repair moved the threshold out of the evaluated block but kept its target
    rate as ``ref.delivered.mean()`` — the coverage the other mechanism REALISED in the test
    block — so a quantity measured on the evaluated data still crossed into the comparator's
    definition. An external audit showed it: holding training fixed and changing only the test
    mask moved the comparator's delivery.

    The rate is now the fraction of TRAINING parcels whose unrestricted argmax lands inside the
    legend, which is exactly what the legend mechanism would deliver there, and is observable
    before seeing any test parcel.

    Args:
        proba: Posterior matrix for the whole universe.
        train_pos: Positional indices of the training parcels.
        legend: Classes the reference mechanism promises.

    Returns:
        The confidence floor, or infinity when nothing can be delivered.
    """
    if train_pos.size == 0:
        return float("inf")
    columnas = np.asarray(legend, dtype=int)
    tasa = float(np.isin(proba[train_pos].argmax(axis=1), columnas).mean())
    if tasa <= 0.0:
        return float("inf")
    if tasa >= 1.0:
        return float("-inf")
    return float(np.quantile(proba[train_pos].max(axis=1), 1.0 - tasa))


def confidence_baseline(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: Sequence[BlockPoint],
    *,
    num_classes: int = NUM_CLASSES,
) -> list[BlockPoint]:
    """Deliver the most confident parcels, matching a reference mechanism's count.

    The legend stays complete, so this mechanism emits over all eighteen classes; what
    shrinks is how many parcels it answers.

    **The whole operating point comes from the TRAINING blocks**, threshold and target rate
    alike: see :func:`umbral_desde_entrenamiento`. Two repairs were needed. The first moved the
    threshold out of the evaluated block but kept its target rate as the coverage the reference
    REALISED in the test block, which is the same leak through a smaller hole. The realised
    coverage here is whatever it is; matching a count exactly is precisely what cannot be done
    without looking.

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
        train_pos, test_pos = splits[ref.block]
        umbral = umbral_desde_entrenamiento(proba, train_pos, ref.legend)
        delivered = confidence[test_pos] >= umbral
        emitted = free[test_pos]
        truth = labels[test_pos]
        presentes = presentes_en_bloque(truth)
        points.append(
            BlockPoint(
                mechanism="rechazo por confianza",
                k=ref.k,
                block=ref.block,
                legend=tuple(range(num_classes)),
                delivered=delivered,
                emitted=emitted,
                aligned_f1=macro_over(
                    truth[delivered], emitted[delivered], ref.legend, presentes=presentes
                ),
                native_f1=macro_over(
                    truth[delivered],
                    emitted[delivered],
                    presentes,
                    presentes=presentes,
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
        presentes = presentes_en_bloque(truth)
        points.append(
            BlockPoint(
                mechanism="sin mecanismo",
                k=ref.k,
                block=ref.block,
                legend=ref.legend,
                delivered=delivered,
                emitted=emitted,
                aligned_f1=macro_over(truth, emitted, ref.legend, presentes=presentes),
                native_f1=macro_over(truth, emitted, ref.legend, presentes=presentes),
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
    unidad: str,
    n_boot: int = 0,
    random_state: int = 0,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired interval for the difference of two mechanisms' mean block macro-F1.

    **The resampling unit has to be declared, and it is the block.** That was defect 2 of the
    internal diagnosis and it survived the first repair: resampling parcels inside each block
    treats sixteen thousand parcels as sixteen thousand independent draws when the design has
    five blocks, and produces an interval far narrower than the design supports. There is no
    default for ``unidad`` on purpose — the previous silent default is what made the defect
    invisible for two rounds of audit.

    ``"bloque"`` is the estimand-level interval: a paired t over the per-block deltas, with the
    block as the unit, which is what five spatial blocks entitle anyone to claim. It is
    degenerate when the two mechanisms coincide, which is the harness self-check. **Below three
    defined blocks it returns the deltas and nothing else** — no interval, no p — because that is
    the declared contract for a two-region bench and the code was breaking it.

    ``"parcela"`` and ``"cluster"`` keep the old bootstrap, but they answer a **different and
    narrower question** — how much the estimate moves if this block's parcels (or patches)
    had been sampled differently, holding the blocks fixed. They are descriptive and are not
    the interval of the article.

    Args:
        labels: Ground-truth labels for the whole universe.
        splits: ``(train_pos, test_pos)`` index pairs, one per block.
        left: Points of the mechanism being tested, one per block.
        right: Points of the comparator, aligned with ``left``.
        unidad: ``"bloque"``, ``"parcela"`` or ``"cluster"``. Required.
        n_boot: Number of resamples. Only used when the unit is not the block.
        random_state: Seed. Only used when the unit is not the block.
        clusters: Cluster id per parcel of the whole universe. Required for ``"cluster"``.

    Returns:
        Observed delta, interval, two-sided p-value, the per-block deltas and the unit used.

    Raises:
        ValueError: on an unknown unit, or a cluster interval without cluster ids.
    """
    if unidad not in {"bloque", "parcela", "cluster"}:
        raise ValueError(f"unidad de remuestreo desconocida: {unidad!r}")
    if unidad == "cluster" and clusters is None:
        raise ValueError("la unidad 'cluster' necesita los identificadores de cluster")

    izq_f1 = np.asarray([p.aligned_f1 for p in left], dtype=float)
    der_f1 = np.asarray([p.aligned_f1 for p in right], dtype=float)
    observed = float(np.nanmean(izq_f1) - np.nanmean(der_f1))
    por_bloque = [float(a.aligned_f1 - b.aligned_f1) for a, b in zip(left, right, strict=True)]
    indefinidos = int(np.isnan(por_bloque).sum())

    if unidad == "bloque":
        d = np.asarray(por_bloque, dtype=float)
        d = d[~np.isnan(d)]
        n = d.size
        base: dict[str, Any] = {
            "delta": observed,
            "unidad": unidad,
            "n_unidades": n,
            "bloques_indefinidos": indefinidos,
            "deltas_por_bloque": por_bloque,
        }
        # Con menos de tres bloques NO se publica intervalo ni p. Es el contrato de US-125 y el
        # codigo lo estaba incumpliendo: BreizhCrops tiene exactamente dos regiones, y el
        # productor le aplicaba Holm a un p que no deberia existir.
        min_bloques = 3
        if n < min_bloques:
            return {
                **base,
                "ci_low": None,
                "ci_high": None,
                "excluye_cero": None,
                "p_valor": None,
                "motivo": (
                    f"con {n} bloque(s) definidos no se publica intervalo ni p: solo los deltas"
                ),
            }
        sd = float(d.std(ddof=1))
        if sd == 0.0:
            # Varianza entre bloques nula. Hay que separar los dos casos, porque el anterior
            # devolvia p=1 para los dos y el unico test usaba el que no distingue: cero.
            media = float(d.mean())
            if media == 0.0:
                # Los dos mecanismos coinciden bloque a bloque: autocomprobacion del arnes.
                return {
                    **base,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "excluye_cero": float(False),
                    "p_valor": 1.0,
                    "motivo": "los dos mecanismos coinciden en todos los bloques",
                }
            return {
                **base,
                "ci_low": media,
                "ci_high": media,
                "excluye_cero": float(True),
                "p_valor": None,
                "motivo": (
                    "varianza entre bloques exactamente nula con efecto no nulo: la t no esta "
                    "definida y no se inventa un p. El intervalo degenerado dice lo que hay"
                ),
            }
        error = sd / np.sqrt(n)
        low, high = stats.t.interval(0.95, n - 1, float(d.mean()), error)
        p = float(2 * stats.t.sf(abs(float(d.mean())) / error, n - 1))
        return {
            **base,
            "ci_low": float(low),
            "ci_high": float(high),
            "excluye_cero": float(low > 0 or high < 0),
            "p_valor": p,
        }

    rng = np.random.default_rng(random_state)
    prepared = []
    for a, b in zip(left, right, strict=True):
        _, test_pos = splits[a.block]
        truth = labels[test_pos]
        group = None if unidad == "parcela" or clusters is None else clusters[test_pos]
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
            # El universo de clases es el del BLOQUE, no el de la remuestra ni el de lo
            # entregado: si se recalcula aqui, el denominador vuelve a moverse.
            presentes = presentes_en_bloque(truth)
            left_scores.append(
                macro_over(
                    truth[idx][a.delivered[idx]],
                    a.emitted[idx][a.delivered[idx]],
                    a.legend,
                    presentes=presentes,
                )
            )
            right_scores.append(
                macro_over(
                    truth[idx][b.delivered[idx]],
                    b.emitted[idx][b.delivered[idx]],
                    a.legend,
                    presentes=presentes,
                )
            )
        draws[i] = float(np.nanmean(left_scores) - np.nanmean(right_scores))

    low, high = np.percentile(draws, [2.5, 97.5])
    below = float((draws <= 0).mean())
    above = float((draws >= 0).mean())
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "excluye_cero": float(low > 0 or high < 0),
        "p_valor": float(min(1.0, 2 * min(below, above))),
        "p_bootstrap": float(min(1.0, 2 * min(below, above))),
        "unidad": unidad,
        "n_unidades": int(sum(t.size for t, _, _, _ in prepared)),
        "bloques_indefinidos": indefinidos,
        "deltas_por_bloque": por_bloque,
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
