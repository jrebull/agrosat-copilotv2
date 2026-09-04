"""Leak-free arbitration and coverage experiments for the MICAI manuscript.

The manuscript compares three ways of combining the fold-5 parcel posteriors of
several members, and two ways of trading coverage for quality. Every routine here
scores on the same 16 640 parcels shared by the members, against the sealed fold-5
ground truth, and never lets a model see the labels of the parcels it is measured
on.

The distinction that motivates the module: :class:`~ml.ensemble.stacking.StackingEnsemble`
refits its meta-learner on every fold-5 row before :meth:`predict_proba`, so the
figure it reports is in-sample for the meta-learner even though the base members are
held out. :func:`pooled_spatial_oof_posteriors` instead keeps each parcel's
prediction from the sub-fold model that never saw it, and pools the five blocks into
one posterior matrix, which is the estimate that can be compared against a single
member scored on the same parcels.

Averaging the per-sub-fold macro-F1 is also avoided on purpose: a geographic block
rarely contains all 18 classes, so the absent ones enter the mean as zeros and drag
the average below what the model actually delivers. The pooled matrix is scored once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import accuracy_score, f1_score

from ml.eval.oof.inventario import exigir_canonicos
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

KEY = "canonical_parcel_id"
NUM_CLASSES = len(PROB_COLUMNS)


@dataclass(frozen=True)
class Combination:
    """A named per-parcel posterior matrix ready to be scored.

    Attributes:
        name: Human-readable name of the combination rule.
        proba: ``(n_parcels, 18)`` posterior matrix aligned with the shared keys.
        regime: Either ``"held-out"`` (no model saw these labels) or
            ``"in-sample"`` (the arbiter was refit on the parcels it is scored on).
    """

    name: str
    proba: np.ndarray
    regime: str


def load_member_posteriors(
    oof_dir: Path,
    members: Sequence[str],
    keys: Sequence[str],
    *,
    permitir_no_canonicos: bool = False,
) -> dict[str, np.ndarray]:
    """Load each member's fold-5 parcel posteriors aligned to ``keys``.

    **Rechaza por defecto los miembros que el inventario no marca como canonicos.** Este es el
    unico punto por el que el analisis MICAI lee posteriores, asi que es el unico sitio donde la
    regla se puede imponer una vez y valer para todas las fases. Marcar un fichero como
    ``legacy_unverified`` en un JSON no impide que se lea: eso lo aprendio este proyecto con los
    trece artefactos ``OBSOLETO``, que llevaban su aviso y se citaban igual.

    **Pendiente, y se dice aqui para que no se olvide**: bajo `estimando-v1.json` la ausencia de
    una parcela en un miembro es NO ENTREGA, no un error. Esta funcion todavia levanta
    `ValueError` cuando un miembro no cubre todo lo pedido. Hoy no muerde —los miembros canonicos
    cubren la poblacion entera— pero en cuanto entre uno que no la cubra habra que decidir como se
    representa la no entrega en una matriz densa, y esa decision no la toma un `except`.

    Args:
        oof_dir: Directory with ``oof_parcel_{member}_fold5.parquet``.
        members: Member names to load.
        keys: Canonical parcel ids defining the row order.
        permitir_no_canonicos: Escape hatch for uses that are NOT the MICAI analysis —
            diagnostics, migrations, the re-dump verification itself. No lo use el analisis.

    Returns:
        Mapping member name to a ``(len(keys), 18)`` float64 posterior matrix.

    Raises:
        EstadoNoCanonicoError: if any member is not canonical and the escape hatch is off.
        FileNotFoundError: if a member's parquet is absent.
        ValueError: if a member does not cover every requested parcel.
    """
    if not permitir_no_canonicos:
        exigir_canonicos(list(members))
    order = pl.DataFrame({KEY: list(keys)}).with_row_index("_pos")
    out: dict[str, np.ndarray] = {}
    for member in members:
        path = oof_dir / f"oof_parcel_{member}_fold5.parquet"
        if not path.exists():
            raise FileNotFoundError(f"falta el OOF del miembro {member}: {path}")
        frame = pl.read_parquet(path, columns=[KEY, *PROB_COLUMNS])
        joined = order.join(frame, on=KEY, how="left").sort("_pos")
        if joined.select(pl.col(PROB_COLUMNS[0]).is_null().any()).item():
            raise ValueError(f"el miembro {member} no cubre todas las parcelas pedidas")
        out[member] = joined.select(PROB_COLUMNS).to_numpy().astype(np.float64)
    logger.info("members_loaded", n_members=len(out), n_parcels=len(keys))
    return out


def combine_mean(posteriors: Sequence[np.ndarray]) -> np.ndarray:
    """Average the members' posteriors with uniform weights.

    This is the homogeneous rule: one weight per member, shared by every class.

    Args:
        posteriors: Member posterior matrices of identical shape.

    Returns:
        The averaged posterior matrix.
    """
    averaged: np.ndarray = np.mean(np.stack(posteriors, axis=0), axis=0)
    return averaged


def combine_weighted(posteriors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Average the members' posteriors with one global weight each.

    Still homogeneous across classes: a member that is excellent on one crop and
    useless on another gets a single weight for both.

    Args:
        posteriors: Member posterior matrices of identical shape.
        weights: One non-negative weight per member.

    Returns:
        The weighted posterior matrix, renormalised to sum to one per parcel.

    Raises:
        ValueError: if the weights do not match the members or sum to zero.
    """
    if len(posteriors) != len(weights):
        raise ValueError("weights and posteriors must have the same length")
    array = np.asarray(weights, dtype=np.float64)
    if array.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    array = array / array.sum()
    stacked = np.stack(posteriors, axis=0)
    blended: np.ndarray = np.tensordot(array, stacked, axes=(0, 0))
    normalised: np.ndarray = blended / blended.sum(axis=1, keepdims=True)
    return normalised


def per_class_table(
    labels: np.ndarray,
    named_predictions: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Per-class F1 of several predictors side by side on the same parcels.

    The question the arbitration claim rests on is whether a trained combiner buys
    anything on the classes a uniform average cannot serve, so the comparison has
    to be readable class by class and next to each class's support.

    Args:
        labels: Ground-truth labels.
        named_predictions: Mapping predictor name to its hard predictions.

    Returns:
        One row per class with its support and one F1 column per predictor.
    """
    rows: list[dict[str, float | int]] = []
    for class_id in range(NUM_CLASSES):
        present = labels == class_id
        row: dict[str, float | int] = {"clase": class_id, "soporte": int(present.sum())}
        for name, pred in named_predictions.items():
            row[f"f1_{name}"] = float(
                f1_score(
                    labels == class_id,
                    pred == class_id,
                    average="binary",
                    zero_division=0,
                )
            )
        rows.append(row)
    return pl.DataFrame(rows).sort("soporte", descending=True)


def _macro_on_legend(
    labels: np.ndarray,
    emitted: np.ndarray,
    legend: Sequence[int],
) -> float:
    """Macro-F1 over the promised classes that actually occur in the evaluated set.

    A class in the legend that does not occur in this block would enter the average
    as a zero and report a failure that never happened, so the average runs over the
    intersection of the legend and the classes present.

    Args:
        labels: Ground-truth labels of the delivered parcels.
        emitted: Predicted labels of the delivered parcels.
        legend: Classes the predictor was allowed to emit.

    Returns:
        The macro-F1 over the evaluated classes, or ``0.0`` if none occur.
    """
    evaluated = sorted(set(legend) & set(labels.tolist()))
    if not evaluated:
        return 0.0
    return float(f1_score(labels, emitted, labels=evaluated, average="macro", zero_division=0))


def coverage_by_class_retirement(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    k_values: Sequence[int],
) -> list[dict[str, object]]:
    """Quality-coverage points obtained by retiring whole classes from the legend.

    Evaluated block by block: the legend of a block is the ``K`` best classes on the
    OTHER blocks, so it is never chosen with the numbers it is scored against, and
    every block is measured against ONE legend. Pooling the blocks under a shared
    legend was rejected on purpose: the legends of different blocks disagree (at
    ``K = 16`` their union is all 18 classes while their intersection is 11), so a
    pooled number would be scoring a product that no deployment would ship.

    Args:
        proba: Posterior matrix of the predictor under study.
        labels: Ground-truth labels.
        splits: ``(train_pos, test_pos)`` positional index pairs, one per block.
        k_values: Legend sizes to evaluate.

    Returns:
        One dictionary per ``(K, block)`` with its legend, coverage and quality.
    """
    pred_full = proba.argmax(axis=1)
    records: list[dict[str, object]] = []
    for k in k_values:
        for block, (train_pos, test_pos) in enumerate(splits):
            if train_pos.size == 0 or test_pos.size == 0:
                continue
            ranking = sorted(
                (
                    (
                        float(
                            f1_score(
                                labels[train_pos] == class_id,
                                pred_full[train_pos] == class_id,
                                average="binary",
                                zero_division=0,
                            )
                        ),
                        class_id,
                    )
                    for class_id in range(NUM_CLASSES)
                ),
                reverse=True,
            )
            legend = sorted(class_id for _, class_id in ranking[:k])
            columns = np.asarray(legend, dtype=int)
            block_labels = labels[test_pos]
            delivered = np.isin(block_labels, columns)
            emitted = columns[proba[np.ix_(test_pos, columns)].argmax(axis=1)]
            records.append(
                {
                    "mecanismo": "retirada de clases",
                    "k": k,
                    "bloque": block,
                    "n_bloque": int(test_pos.size),
                    "n_entregadas": int(delivered.sum()),
                    "cobertura": float(delivered.mean()),
                    "f1_macro": _macro_on_legend(
                        block_labels[delivered], emitted[delivered], legend
                    ),
                    "accuracy": float(accuracy_score(block_labels[delivered], emitted[delivered]))
                    if delivered.any()
                    else 0.0,
                    "leyenda": legend,
                    "posiciones_entregadas": test_pos[delivered],
                }
            )
    return records


def coverage_by_confidence(
    proba: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    targets_by_block: dict[int, Sequence[float]],
) -> list[dict[str, object]]:
    """Quality-coverage points obtained by rejecting the least confident parcels.

    The threshold is the quantile of the EVALUATED block's own confidences that
    hits the target coverage. That is not leakage: hitting a coverage target reads
    only the posteriors, never a label, and any deployment can rank its parcels by
    confidence and answer the most confident 80 % without knowing a single truth.

    Taking the quantile from the other blocks instead — the first thing this
    function did — looks more cautious and is actually wrong: the confidence
    distribution shifts from block to block, so the same threshold delivered a very
    different fraction in each one. Measured mismatch against the mechanism it is
    supposed to be matched with: 0.064 on average and 0.261 in the worst block,
    which is not a comparison at equal coverage but a comparison at whatever
    coverage came out.

    The legend stays complete: what shrinks is the set of parcels answered, which
    is why the macro runs over every class present among the delivered ones.

    Args:
        proba: Posterior matrix of the predictor under study.
        labels: Ground-truth labels.
        splits: ``(train_pos, test_pos)`` positional index pairs, one per block.
        targets_by_block: Coverage targets to match, per block index.

    Returns:
        One dictionary per ``(target, block)`` with its coverage and quality.
    """
    confidence = proba.max(axis=1)
    pred_full = proba.argmax(axis=1)
    records: list[dict[str, object]] = []
    for block, (train_pos, test_pos) in enumerate(splits):
        if train_pos.size == 0 or test_pos.size == 0:
            continue
        for index, target in enumerate(targets_by_block.get(block, [])):
            block_conf = confidence[test_pos]
            # Deliver exactly the n most confident parcels of this block, where n is
            # the count the other mechanism delivered here. Ties are broken by order
            # so the delivered count is exact rather than approximately right.
            n_deliver = max(0, min(test_pos.size, round(target * test_pos.size)))
            order = np.argsort(-block_conf, kind="stable")
            delivered = np.zeros(test_pos.size, dtype=bool)
            delivered[order[:n_deliver]] = True
            block_labels = labels[test_pos]
            emitted = pred_full[test_pos]
            records.append(
                {
                    "mecanismo": "rechazo por confianza",
                    "punto": index,
                    "objetivo_cobertura": float(target),
                    "bloque": block,
                    "n_bloque": int(test_pos.size),
                    "n_entregadas": int(delivered.sum()),
                    "cobertura": float(delivered.mean()),
                    "f1_macro": _macro_on_legend(
                        block_labels[delivered],
                        emitted[delivered],
                        sorted(set(block_labels[delivered].tolist())),
                    ),
                    "accuracy": float(accuracy_score(block_labels[delivered], emitted[delivered]))
                    if delivered.any()
                    else 0.0,
                    "posiciones_entregadas": test_pos[delivered],
                }
            )
    return records


def pooled_weighted_vote(
    posteriors: Sequence[np.ndarray],
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted vote whose weights are estimated outside the parcels they score.

    Weighting members by the macro-F1 they obtain on the very parcels being scored
    would be an in-sample choice, and a generous one: it hands the rule the answer
    it is being tested on. Here each block's weights come from the members' macro-F1
    on the OTHER blocks, exactly like the meta-learner's training set.

    Args:
        posteriors: Member posterior matrices of identical shape.
        labels: Ground-truth labels aligned with the matrices.
        splits: ``(train_pos, test_pos)`` positional index pairs, one per block.

    Returns:
        Tuple of the pooled posterior matrix and the mask of scored parcels.

    Raises:
        ValueError: if no block produced predictions.
    """
    pooled = np.zeros_like(posteriors[0])
    covered = np.zeros(posteriors[0].shape[0], dtype=bool)
    for block, (train_pos, test_pos) in enumerate(splits):
        if train_pos.size == 0 or test_pos.size == 0:
            continue
        weights = [
            f1_score(
                labels[train_pos],
                proba[train_pos].argmax(axis=1),
                average="macro",
                zero_division=0,
            )
            for proba in posteriors
        ]
        total = float(sum(weights))
        if total <= 0:
            weights = [1.0] * len(posteriors)
            total = float(len(posteriors))
        blended = np.tensordot(
            np.asarray(weights, dtype=np.float64) / total,
            np.stack([proba[test_pos] for proba in posteriors], axis=0),
            axes=(0, 0),
        )
        pooled[test_pos] = blended / blended.sum(axis=1, keepdims=True)
        covered[test_pos] = True
        logger.info("weighted_block_done", block=block, n_test=int(test_pos.size))
    if not covered.any():
        raise ValueError("ningun bloque espacial produjo predicciones")
    return pooled, covered


def pooled_spatial_oof_posteriors(
    meta_features: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one meta-learner per spatial block and keep each parcel's held-out row.

    Every parcel is predicted by the model trained on the OTHER blocks, so the
    pooled matrix contains no in-sample row. Parcels excluded from every test
    block (the inter-fold buffer) are reported so the caller can drop them.

    Args:
        meta_features: ``(n_parcels, n_members * 18)`` meta-feature matrix.
        labels: Aligned ground-truth labels.
        splits: ``(train_pos, test_pos)`` positional index pairs, one per block.
        random_state: Seed of the logistic-regression meta-learner.

    Returns:
        Tuple of the pooled ``(n_parcels, 18)`` posterior matrix and the boolean
        mask of parcels that received a held-out prediction.

    Raises:
        ValueError: if no split yields a prediction.
    """
    from sklearn.linear_model import LogisticRegression

    pooled = np.zeros((meta_features.shape[0], NUM_CLASSES), dtype=np.float64)
    covered = np.zeros(meta_features.shape[0], dtype=bool)
    for block, (train_pos, test_pos) in enumerate(splits):
        if train_pos.size == 0 or test_pos.size == 0:
            continue
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(meta_features[train_pos], labels[train_pos])
        local = np.asarray(model.predict_proba(meta_features[test_pos]))
        expanded = np.zeros((test_pos.size, NUM_CLASSES), dtype=np.float64)
        expanded[:, np.asarray(model.classes_, dtype=int)] = local
        pooled[test_pos] = expanded
        covered[test_pos] = True
        logger.info(
            "pooled_block_done", block=block, n_train=int(train_pos.size), n_test=int(test_pos.size)
        )
    if not covered.any():
        raise ValueError("ningun bloque espacial produjo predicciones")
    return pooled, covered


def score(
    labels: np.ndarray,
    proba: np.ndarray,
    *,
    class_set: Sequence[int] | None = None,
) -> dict[str, float]:
    """Score a posterior matrix with macro-F1 and accuracy.

    When ``class_set`` is given the predictor is restricted to that legend: the
    argmax runs over those columns only and the macro average covers exactly those
    classes. That is the operational meaning of retiring a class --- a legend you do
    not promise is a legend your model does not emit --- and leaving the argmax over
    all 18 columns instead would count every retired class as a zero and report a
    quality the deployed system never had.

    Args:
        labels: Ground-truth labels.
        proba: Posterior matrix.
        class_set: Legend the predictor is allowed to emit. ``None`` keeps all 18.

    Returns:
        ``{"f1_macro": ..., "accuracy": ...}``.
    """
    if class_set is None:
        pred = proba.argmax(axis=1)
        present = sorted(set(labels.tolist()))
    else:
        columns = np.asarray(sorted(class_set), dtype=int)
        pred = columns[proba[:, columns].argmax(axis=1)]
        present = columns.tolist()
    return {
        "f1_macro": float(f1_score(labels, pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
    }


def paired_bootstrap_delta(
    labels: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the paired macro-F1 difference between two predictors.

    Both predictors are resampled on the SAME parcel indices at every draw, so
    the interval is of the difference and not of two independent quantities.

    Args:
        labels: Ground-truth labels.
        pred_a: Hard predictions of the first predictor.
        pred_b: Hard predictions of the second predictor.
        n_boot: Number of bootstrap resamples.
        random_state: Seed.
        alpha: Two-sided significance level.

    Returns:
        Observed delta, its percentile interval and the share of draws below zero.
    """
    rng = np.random.default_rng(random_state)
    n = labels.size
    observed = float(
        f1_score(labels, pred_a, average="macro", zero_division=0)
        - f1_score(labels, pred_b, average="macro", zero_division=0)
    )
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = f1_score(labels[idx], pred_a[idx], average="macro", zero_division=0) - f1_score(
            labels[idx], pred_b[idx], average="macro", zero_division=0
        )
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_f1_macro": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_below_zero": float((deltas < 0).mean()),
        "n_boot": float(n_boot),
        "excludes_zero": float(lo > 0 or hi < 0),
    }


def mcnemar(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on the parcels where the two predictors disagree.

    Args:
        labels: Ground-truth labels.
        pred_a: Hard predictions of the first predictor.
        pred_b: Hard predictions of the second predictor.

    Returns:
        Discordant counts, the statistic and the exact two-sided p-value.
    """
    from scipy.stats import binomtest

    correct_a = pred_a == labels
    correct_b = pred_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    total = only_a + only_b
    p_value = 1.0 if total == 0 else float(binomtest(only_a, total, 0.5).pvalue)
    return {
        "solo_a_acierta": float(only_a),
        "solo_b_acierta": float(only_b),
        "discordantes": float(total),
        "p_value": p_value,
    }
