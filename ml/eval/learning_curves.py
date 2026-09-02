"""Learning/validation curves and under/overfitting diagnosis (US-021, EPIC 4).

Reusable module consumed by the tabular baseline (RF/XGB of US-019) and, later,
by the EPIC 5/6 architectures to diagnose under/overfitting of any sklearn
estimator. Exposes three families of functions:

- **Learning curve** (criterion AC-1): :func:`plot_learning_curve` wraps
  :func:`sklearn.model_selection.learning_curve` to plot train and validation
  accuracy against the number of training samples. The curves RE-TRAIN fresh
  estimators for each point (decision D3) — they do not load the production
  joblib of US-019.
- **Validation curve** (criterion AC-2): :func:`plot_validation_curve` wraps
  :func:`sklearn.model_selection.validation_curve` to plot accuracy against a
  critical hyperparameter (``max_depth``, ``n_estimators``, ``learning_rate``).
- **Diagnosis** (criterion AC-4): :func:`diagnose_fit` derives an
  ``overfit``/``underfit``/``good_fit`` verdict from the result of a learning
  curve with parametric thresholds. Pure function — it does not re-train
  (decision D8).

Decision D2 (plan US-021 2.1): the ``cv`` received by ``learning_curve`` and
``validation_curve`` must be a **materialized list** of ``(train_idx, test_idx)``
tuples — ``learning_curve`` reuses the ``cv`` once per ``train_size``; a generator
is exhausted after the first use and the remaining sizes are left without folds.
:func:`_materialize_cv_splits` guarantees the materialization.

Decision D4: the metric of the curves is ``accuracy`` (the acceptance criterion
requests it literally); F1-macro is the main metric of the baseline (US-019) but
the AC of US-021 specifies accuracy for the curves.

Polars is the I/O format; the conversion to numpy happens exclusively at the
sklearn boundary, in the private helper :func:`_to_numpy_xy`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import structlog
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.model_selection import learning_curve, validation_curve

# `ml.train.baseline` is imported lazily inside `_to_numpy_xy`
# to break the import cycle: `baseline` imports from `ml.eval.metrics`,
# and `ml.eval.__init__` re-exports this module — a module-level import
# would trigger a circular import when loading the `ml.eval` package.

logger = structlog.get_logger(__name__)

__all__ = [
    "FitDiagnosis",
    "FitVerdict",
    "LearningCurveResult",
    "TemporalLossHistory",
    "ValidationCurveResult",
    "diagnose_fit",
    "diagnose_temporal_fit",
    "fetch_loss_history_from_mlflow",
    "plot_learning_curve",
    "plot_loss_history_from_mlflow",
    "plot_validation_curve",
]

FitVerdict = Literal["overfit", "underfit", "good_fit"]

# Figure resolution for the visual deliverables of Avance 3 (criterion AC-8).
_PLOT_DPI: int = 200

# Default sample fractions for the learning curve (decision D6:
# fractions, not absolute counts — they adapt to any dataset size).
_DEFAULT_TRAIN_SIZES: tuple[float, ...] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


# ---------------------------------------------------------------------------
# Output dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearningCurveResult:
    """Result of a learning curve.

    Attributes:
        train_sizes_abs: Absolute number of training samples per curve point,
            vector ``(n,)``.
        train_scores_mean: Mean train accuracy per size, vector ``(n,)``.
        train_scores_std: Standard deviation of the train accuracy per size,
            vector ``(n,)``.
        val_scores_mean: Mean validation accuracy per size, vector ``(n,)``.
        val_scores_std: Standard deviation of the validation accuracy per size,
            vector ``(n,)``.
        scoring: Metric used in the curve (``"accuracy"`` by default).
    """

    train_sizes_abs: np.ndarray
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    val_scores_mean: np.ndarray
    val_scores_std: np.ndarray
    scoring: str


@dataclass(frozen=True)
class ValidationCurveResult:
    """Result of a validation curve over a hyperparameter.

    Attributes:
        param_name: Name of the varied hyperparameter.
        param_range: Evaluated values, in the same order as the curves.
        train_scores_mean: Mean train accuracy per value, vector ``(n,)``.
        train_scores_std: Standard deviation of the train accuracy, vector
            ``(n,)``.
        val_scores_mean: Mean validation accuracy per value, vector ``(n,)``.
        val_scores_std: Standard deviation of the validation accuracy, vector
            ``(n,)``.
    """

    param_name: str
    param_range: list
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    val_scores_mean: np.ndarray
    val_scores_std: np.ndarray


@dataclass(frozen=True)
class FitDiagnosis:
    """Under/overfitting diagnosis derived from a learning curve.

    Attributes:
        verdict: ``"overfit"``, ``"underfit"`` or ``"good_fit"``.
        gap: ``accuracy_train - accuracy_val`` at the maximum size of the
            learning curve.
        train_acc_max: Train accuracy at the maximum size.
        val_acc_max: Validation accuracy at the maximum size.
        explanation: Text in Spanish that justifies the verdict with the
            concrete numbers.
    """

    verdict: FitVerdict
    gap: float
    train_acc_max: float
    val_acc_max: float
    explanation: str


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _materialize_cv_splits(
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize the spatial splits into a reusable list.

    Decision D2 (risk R2, the most likely bug): ``learning_curve`` reuses the
    ``cv`` once per ``train_size``; a generator is exhausted after the first use
    and the remaining sizes are left without folds. This function forces a
    ``list`` of ``(train_idx, test_idx)`` tuples of ``np.int64`` arrays —
    reusable as many times as sklearn consumes it.

    Args:
        cv_splits: Sequence (list or generator) of ``(train_idx, test_idx)``
            tuples of positional indices, typically the output of
            ``ml.train.baseline._build_cv_splits``.

    Returns:
        Materialized list of ``(train_idx, test_idx)`` tuples with the indices
        converted to ``np.int64`` arrays.

    Raises:
        ValueError: if ``cv_splits`` is empty or if any split has no samples in
            train or test.
    """
    materialized: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_arr = np.asarray(train_idx, dtype=np.int64)
        test_arr = np.asarray(test_idx, dtype=np.int64)
        if train_arr.size == 0 or test_arr.size == 0:
            raise ValueError(
                f"Spatial split {fold_idx} has no samples in train "
                f"({train_arr.size}) or in test ({test_arr.size})."
            )
        materialized.append((train_arr, test_arr))
    if not materialized:
        raise ValueError("`cv_splits` is empty; the curves require at least one split.")
    return materialized


def _to_numpy_xy(
    df: pl.DataFrame,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the features DataFrame to the numpy matrix at the sklearn boundary.

    Reuses the baseline helpers (``_feature_columns``, ``_feature_matrix``,
    ``_impute``, ``_encode_labels``) so that the X matrix, the labels and the
    imputation are identical to those used by US-019. The conversion to numpy
    happens only here — the rest of the module operates on Polars.

    When ``max_samples > 0`` and the dataset is larger, it returns a
    class-stratified subsample (decision D7) to speed up dev/CI; it also returns
    the kept positional indices so the caller can realign the ``cv``.

    Args:
        df: Polars features DataFrame already prepared.
        max_samples: Upper bound of samples; ``0`` disables the subsample.
        random_state: Deterministic seed of the stratified subsample.

    Returns:
        Tuple ``(matrix, y_encoded, kept_idx)`` where ``matrix`` is the imputed
        features matrix ``(n, n_features)``, ``y_encoded`` the contiguous labels
        ``(n,)`` and ``kept_idx`` the positional indices of the original ``df``
        that were kept (all if there was no subsample).
    """
    # Lazy import: breaks the `baseline` <-> `ml.eval` cycle (see header).
    from ml.train.baseline import (
        _encode_labels,
        _feature_columns,
        _feature_matrix,
        _impute,
    )

    feature_cols = _feature_columns(df)
    _encoder, y_all = _encode_labels(df)
    matrix_all = _impute(_feature_matrix(df, feature_cols))
    n_rows = df.height
    kept_idx = np.arange(n_rows, dtype=np.int64)

    if max_samples <= 0 or max_samples >= n_rows:
        return matrix_all, y_all, kept_idx

    # Class-stratified subsample: preserves the proportion of each class.
    rng = np.random.default_rng(random_state)
    fraction = max_samples / n_rows
    selected: list[np.ndarray] = []
    for cls in np.unique(y_all):
        cls_idx = np.where(y_all == cls)[0]
        # At least one sample per class so no label is lost.
        n_take = max(1, round(cls_idx.size * fraction))
        n_take = min(n_take, cls_idx.size)
        selected.append(rng.choice(cls_idx, size=n_take, replace=False))
    kept_idx = np.sort(np.concatenate(selected)).astype(np.int64)
    logger.info(
        "learning_curve_subsampled",
        n_original=n_rows,
        n_kept=int(kept_idx.size),
        max_samples=max_samples,
    )
    return matrix_all[kept_idx], y_all[kept_idx], kept_idx


def _remap_cv_splits(
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    kept_idx: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Realign the spatial splits after a sample subsample.

    When :func:`_to_numpy_xy` subsamples the dataset, the positional indices of
    ``cv_splits`` no longer point to the correct rows of the reduced matrix.
    This function translates each original index to its new position in the
    subset and discards the indices that the subsample left out.

    Args:
        cv_splits: Materialized list of splits over the complete dataset.
        kept_idx: Positional indices kept by the subsample, sorted.

    Returns:
        List of ``(train_idx, test_idx)`` splits with positional indices of the
        reduced dataset; folds left without train or test are discarded.
    """
    # `position[i]` = new position of original index `i`, or -1 if discarded.
    max_original = int(kept_idx.max()) + 1 if kept_idx.size else 0
    position = np.full(max_original, -1, dtype=np.int64)
    position[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)

    remapped: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in cv_splits:
        train_in = train_idx[train_idx < max_original]
        test_in = test_idx[test_idx < max_original]
        new_train = position[train_in]
        new_test = position[test_in]
        new_train = new_train[new_train >= 0]
        new_test = new_test[new_test >= 0]
        if new_train.size == 0 or new_test.size == 0:
            continue
        remapped.append((new_train, new_test))
    if not remapped:
        raise ValueError(
            "The subsample left all spatial folds without samples; increase `max_samples`."
        )
    return remapped


def _curve_figure(
    x_values: Sequence,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    *,
    x_label: str,
    title: str,
) -> Figure:
    """Build a curve figure with a shaded +/-std band.

    US-019 pattern (``ml/eval/metrics.py``): non-interactive ``Agg`` backend so
    the figure is serializable to PNG in CI and in notebooks executed with
    papermill. The shaded band (``fill_between``) covers +/-1 standard deviation
    around the mean of each curve (criterion AC-8).

    Args:
        x_values: X-axis values (sample sizes or hyperparameter values). They
            are labeled as categories to support non-numeric values such as
            ``None`` (risk R4).
        train_mean: Mean train accuracy per point, vector ``(n,)``.
        train_std: Standard deviation of train per point, vector ``(n,)``.
        val_mean: Mean validation accuracy per point, vector ``(n,)``.
        val_std: Standard deviation of validation per point, vector ``(n,)``.
        x_label: X-axis label.
        title: Figure title.

    Returns:
        Matplotlib figure ready for ``fig.savefig(...)`` or ``display``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    # Evenly spaced positions: supports non-numeric values (e.g. `None` in
    # `max_depth`) without breaking the axis (risk R4).
    positions = np.arange(len(x_values))
    tick_labels = [str(v) for v in x_values]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=_PLOT_DPI)
    ax.plot(positions, train_mean, marker="o", color="#2c7fb8", label="Train")
    ax.fill_between(
        positions,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#2c7fb8",
    )
    ax.plot(positions, val_mean, marker="s", color="#d95f0e", label="Validacion")
    ax.fill_between(
        positions,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#d95f0e",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def plot_learning_curve(
    estimator: ClassifierMixin,
    df: pl.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    train_sizes: list[float] | None = None,
    scoring: str = "accuracy",
    max_samples: int = 0,
    random_state: int = 42,
) -> tuple[LearningCurveResult, Figure]:
    """Plot the learning curve of an estimator with spatial cross-validation.

    Wraps :func:`sklearn.model_selection.learning_curve` to measure how train
    and validation accuracy evolve as the number of training samples grows.
    ``learning_curve`` re-trains the estimator
    ``len(train_sizes) * len(cv_splits)`` times — the ``estimator`` must be
    unfitted (decision D3).

    ``cv_splits`` MUST be a materialized list of ``(train_idx, test_idx)`` tuples
    (not a generator): the function reuses the ``cv`` per ``train_size`` and a
    generator would be exhausted (decision D2). It is materialized again
    internally for safety.

    Args:
        estimator: Unfitted sklearn/xgboost estimator (US-019 factory via
            ``ml.train.baseline.build_estimator``).
        df: Polars features DataFrame already prepared (with ``parcel_id``,
            ``class_id`` and numeric feature columns).
        cv_splits: Materialized list of spatial splits ``(train_idx, test_idx)``
            of positional indices.
        train_sizes: Train fractions per curve point; if ``None`` the values
            ``(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)`` are used (decision D6).
        scoring: Curve metric; ``"accuracy"`` by default (decision D4).
        max_samples: Upper bound of samples for the stratified subsample in
            dev/CI; ``0`` (default) uses the complete dataset (decision D7).
        random_state: Deterministic seed of the subsample.

    Returns:
        Tuple ``(LearningCurveResult, Figure)`` with the scores aggregated by
        size and the figure with the +/-std band.

    Raises:
        ValueError: if ``cv_splits`` is empty or if ``df`` lacks mandatory
            columns.
    """
    sizes = list(train_sizes) if train_sizes is not None else list(_DEFAULT_TRAIN_SIZES)
    splits = _materialize_cv_splits(cv_splits)
    matrix, y_encoded, kept_idx = _to_numpy_xy(
        df, max_samples=max_samples, random_state=random_state
    )
    if kept_idx.size != df.height:
        splits = _remap_cv_splits(splits, kept_idx)

    logger.info(
        "learning_curve_start",
        n_samples=int(matrix.shape[0]),
        n_features=int(matrix.shape[1]),
        n_train_sizes=len(sizes),
        n_folds=len(splits),
        scoring=scoring,
    )
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        matrix,
        y_encoded,
        train_sizes=np.asarray(sizes, dtype=np.float64),
        cv=splits,
        scoring=scoring,
        n_jobs=None,
        shuffle=False,
        random_state=random_state,
    )
    result = LearningCurveResult(
        train_sizes_abs=np.asarray(train_sizes_abs, dtype=np.int64),
        train_scores_mean=train_scores.mean(axis=1),
        train_scores_std=train_scores.std(axis=1),
        val_scores_mean=val_scores.mean(axis=1),
        val_scores_std=val_scores.std(axis=1),
        scoring=scoring,
    )
    figure = _curve_figure(
        result.train_sizes_abs.tolist(),
        result.train_scores_mean,
        result.train_scores_std,
        result.val_scores_mean,
        result.val_scores_std,
        x_label="Muestras de entrenamiento",
        title=f"Curva de aprendizaje ({scoring})",
    )
    logger.info(
        "learning_curve_done",
        train_acc_max=round(float(result.train_scores_mean[-1]), 4),
        val_acc_max=round(float(result.val_scores_mean[-1]), 4),
    )
    return result, figure


def plot_validation_curve(
    estimator: ClassifierMixin,
    df: pl.DataFrame,
    param_name: str,
    param_range: list,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    scoring: str = "accuracy",
    max_samples: int = 0,
    random_state: int = 42,
) -> tuple[ValidationCurveResult, Figure]:
    """Plot the validation curve of a hyperparameter with spatial cross-validation.

    Wraps :func:`sklearn.model_selection.validation_curve` to measure how train
    and validation accuracy change when varying a critical hyperparameter
    (``max_depth``, ``n_estimators``, ``learning_rate``). ``validation_curve``
    re-instantiates the estimator for each value of the range — the ``estimator``
    must be unfitted.

    The X-axis supports non-numeric values such as ``None`` (risk R4): the
    ``param_range`` is preserved as-is in the result and the figure labels it as
    a category (``"None"``).

    Args:
        estimator: Unfitted sklearn/xgboost estimator.
        df: Polars features DataFrame already prepared.
        param_name: Name of the hyperparameter to vary (e.g. ``"max_depth"``).
        param_range: Hyperparameter values to evaluate; may contain ``None``
            (e.g. ``max_depth`` without a cap).
        cv_splits: Materialized list of spatial splits.
        scoring: Curve metric; ``"accuracy"`` by default (decision D4).
        max_samples: Upper bound of samples for the subsample in dev/CI; ``0``
            uses the complete dataset (decision D7).
        random_state: Deterministic seed of the subsample.

    Returns:
        Tuple ``(ValidationCurveResult, Figure)`` with the scores aggregated by
        hyperparameter value and the figure with the +/-std band.

    Raises:
        ValueError: if ``cv_splits`` is empty, if ``param_range`` is empty or if
            ``df`` lacks mandatory columns.
    """
    if not param_range:
        raise ValueError("`param_range` cannot be empty.")
    splits = _materialize_cv_splits(cv_splits)
    matrix, y_encoded, kept_idx = _to_numpy_xy(
        df, max_samples=max_samples, random_state=random_state
    )
    if kept_idx.size != df.height:
        splits = _remap_cv_splits(splits, kept_idx)

    logger.info(
        "validation_curve_start",
        param_name=param_name,
        n_values=len(param_range),
        n_samples=int(matrix.shape[0]),
        n_folds=len(splits),
        scoring=scoring,
    )
    train_scores, val_scores = validation_curve(
        estimator,
        matrix,
        y_encoded,
        param_name=param_name,
        param_range=param_range,
        cv=splits,
        scoring=scoring,
        n_jobs=None,
    )
    result = ValidationCurveResult(
        param_name=param_name,
        param_range=list(param_range),
        train_scores_mean=train_scores.mean(axis=1),
        train_scores_std=train_scores.std(axis=1),
        val_scores_mean=val_scores.mean(axis=1),
        val_scores_std=val_scores.std(axis=1),
    )
    figure = _curve_figure(
        result.param_range,
        result.train_scores_mean,
        result.train_scores_std,
        result.val_scores_mean,
        result.val_scores_std,
        x_label=param_name,
        title=f"Curva de validacion — {param_name} ({scoring})",
    )
    logger.info(
        "validation_curve_done",
        param_name=param_name,
        best_val_acc=round(float(result.val_scores_mean.max()), 4),
    )
    return result, figure


def diagnose_fit(
    result: LearningCurveResult,
    *,
    gap_threshold: float = 0.10,
    low_acc_threshold: float = 0.65,
) -> FitDiagnosis:
    """Diagnose under/overfitting from a learning curve.

    Pure function (decision D8): derives the verdict from the scores already
    computed in ``LearningCurveResult`` evaluated at the maximum size of the
    curve — it does not re-train any model.

    Rules (criterion AC-4):

    - ``gap > gap_threshold``                                  -> ``"overfit"``
    - ``train_acc < low_acc AND val_acc < low_acc``            -> ``"underfit"``
    - otherwise                                                -> ``"good_fit"``

    Overfitting is evaluated first: a model with a large gap is overfit even if
    both accuracies are modest. Underfitting only applies when the model fails
    to fit even the train set.

    Args:
        result: Result of :func:`plot_learning_curve`.
        gap_threshold: Threshold of the train-val gap above which there is
            overfitting; ``0.10`` by default (criterion AC-4).
        low_acc_threshold: Threshold below which both accuracies are considered
            low (underfitting); ``0.65`` by default.

    Returns:
        A :class:`FitDiagnosis` with the verdict, the gap, the accuracies at the
        maximum size and a textual explanation.

    Raises:
        ValueError: if the curve has no points.
    """
    if result.train_scores_mean.size == 0:
        raise ValueError("The learning curve has no points to diagnose.")

    train_acc = float(result.train_scores_mean[-1])
    val_acc = float(result.val_scores_mean[-1])
    gap = train_acc - val_acc

    if gap > gap_threshold:
        verdict: FitVerdict = "overfit"
        explanation = (
            f"Sobreajuste: el gap train-val es {gap:.3f} (> {gap_threshold:.2f}). "
            f"El modelo memoriza el train (accuracy {train_acc:.3f}) pero "
            f"generaliza peor en validacion (accuracy {val_acc:.3f})."
        )
    elif train_acc < low_acc_threshold and val_acc < low_acc_threshold:
        verdict = "underfit"
        explanation = (
            f"Subajuste: train ({train_acc:.3f}) y validacion ({val_acc:.3f}) "
            f"estan ambos por debajo de {low_acc_threshold:.2f}. El modelo no "
            f"captura la senal ni en el conjunto de entrenamiento; falta "
            f"capacidad o las features no son suficientemente informativas."
        )
    else:
        verdict = "good_fit"
        explanation = (
            f"Buen ajuste: el gap train-val es {gap:.3f} (<= {gap_threshold:.2f}) "
            f"y la accuracy de validacion ({val_acc:.3f}) no es baja. El modelo "
            f"generaliza de forma consistente con su desempeno en train "
            f"({train_acc:.3f})."
        )

    logger.info(
        "fit_diagnosed",
        verdict=verdict,
        gap=round(gap, 4),
        train_acc_max=round(train_acc, 4),
        val_acc_max=round(val_acc, 4),
    )
    return FitDiagnosis(
        verdict=verdict,
        gap=gap,
        train_acc_max=train_acc,
        val_acc_max=val_acc,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Diagnosis for temporal models via loss history in MLflow.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalLossHistory:
    """Per-epoch loss history of a temporal model trained with CV.

    ``train_temporal_model`` logs to MLflow the metric ``fold{i}_train_loss``
    (and ``fold{i}_val_loss`` when there is an internal validation split) per
    epoch. This dataclass aggregates those histories into a consolidated view to
    diagnose under/overfitting in PyTorch models that do not implement the
    sklearn API consumed by :func:`plot_learning_curve`.

    Attributes:
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        run_id: ID of the MLflow run from which the history was read.
        epochs: Array of epoch indices (X-axis of the curve).
        train_loss_mean: Training loss averaged over the spatial CV folds, per
            epoch.
        train_loss_std: Standard deviation across folds.
        val_loss_mean: Internal validation loss averaged over the folds. Empty
            if the training did not open an internal val split.
        val_loss_std: Standard deviation across folds (val).
        n_folds: Number of folds detected in MLflow for this run.
    """

    model_kind: str
    run_id: str
    epochs: np.ndarray
    train_loss_mean: np.ndarray
    train_loss_std: np.ndarray
    val_loss_mean: np.ndarray
    val_loss_std: np.ndarray
    n_folds: int


def fetch_loss_history_from_mlflow(
    run_id: str,
    *,
    model_kind: str,
    tracking_uri: str | None = None,
) -> TemporalLossHistory:
    """Read the per-epoch loss history of a temporal-model MLflow run.

    Queries ``MlflowClient.get_metric_history`` for the metrics
    ``fold{i}_train_loss`` and ``fold{i}_val_loss`` (i in [0, k)) that
    :func:`ml.train.phenology_models.train_temporal_model` logs during the
    spatial CV. Aggregates the folds per epoch computing mean and standard
    deviation.

    Args:
        run_id: ID of the temporal-model MLflow run.
        model_kind: ``"tempcnn"`` or ``"inceptiontime"`` (only for labeling).
        tracking_uri: Override of the tracking URI; if ``None`` it is resolved
            via :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.

    Returns:
        :class:`TemporalLossHistory` with epochs and aggregated curves.

    Raises:
        RuntimeError: if the run does not expose any ``fold{i}_train_loss``
            metric (probably it was not logged to MLflow during training).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri is None:
        from ml.utils.mlflow_utils import resolve_tracking_uri

        tracking_uri = resolve_tracking_uri(None, probe_server=False)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(run_id)
    metric_keys = list(run.data.metrics.keys())
    train_keys = sorted(k for k in metric_keys if k.endswith("_train_loss"))
    val_keys = sorted(k for k in metric_keys if k.endswith("_val_loss"))

    if not train_keys:
        raise RuntimeError(
            f"Run {run_id} does not expose `fold{{i}}_train_loss` metrics. "
            "Verify that the training received `mlflow_uri` and that the "
            "server was available during the run."
        )

    def _collect(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        per_fold: list[np.ndarray] = []
        max_epochs = 0
        for key in keys:
            history = client.get_metric_history(run_id, key)
            if not history:
                continue
            history_sorted = sorted(history, key=lambda m: m.step)
            arr = np.asarray([m.value for m in history_sorted], dtype=np.float64)
            per_fold.append(arr)
            max_epochs = max(max_epochs, arr.size)
        if not per_fold:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty
        # Align folds by padding with NaN at the end if they had early stopping.
        aligned = np.full((len(per_fold), max_epochs), np.nan, dtype=np.float64)
        for i, arr in enumerate(per_fold):
            aligned[i, : arr.size] = arr
        epochs = np.arange(max_epochs, dtype=np.int64)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        return epochs, mean, std

    epochs, train_mean, train_std = _collect(train_keys)
    _, val_mean, val_std = (
        _collect(val_keys)
        if val_keys
        else (
            epochs,
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    )

    return TemporalLossHistory(
        model_kind=model_kind,
        run_id=run_id,
        epochs=epochs,
        train_loss_mean=train_mean,
        train_loss_std=train_std,
        val_loss_mean=val_mean,
        val_loss_std=val_std,
        n_folds=len(train_keys),
    )


def plot_loss_history_from_mlflow(
    history: TemporalLossHistory,
    *,
    title: str | None = None,
) -> Figure:
    """Plot the per-epoch train vs val loss history for a temporal model.

    Replicates the visual format of :func:`plot_learning_curve` (blue train
    curve, orange val, with a +/-std band across folds) but reading the real
    PyTorch history from MLflow instead of re-training the estimator.

    Args:
        history: Result of :func:`fetch_loss_history_from_mlflow`.
        title: Figure title; if ``None`` it is generated from
            ``history.model_kind``.

    Returns:
        Matplotlib figure ``dpi=200`` ready for ``fig.savefig`` or ``display``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if history.epochs.size == 0:
        raise ValueError(f"The history of run {history.run_id} is empty; there is nothing to plot.")

    if title is None:
        title = (
            f"Curva de loss por epoca ({history.model_kind.upper()}) — "
            f"{history.n_folds} folds spatial CV"
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=_PLOT_DPI)
    ax.plot(
        history.epochs,
        history.train_loss_mean,
        color="#4C72B0",
        label="train_loss (media folds)",
    )
    ax.fill_between(
        history.epochs,
        history.train_loss_mean - history.train_loss_std,
        history.train_loss_mean + history.train_loss_std,
        color="#4C72B0",
        alpha=0.15,
    )
    if history.val_loss_mean.size > 0:
        ax.plot(
            history.epochs,
            history.val_loss_mean,
            color="#DD8452",
            label="val_loss (media folds)",
        )
        ax.fill_between(
            history.epochs,
            history.val_loss_mean - history.val_loss_std,
            history.val_loss_mean + history.val_loss_std,
            color="#DD8452",
            alpha=0.15,
        )
    ax.set_xlabel("epoca")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def diagnose_temporal_fit(
    history: TemporalLossHistory,
    *,
    gap_threshold: float = 0.10,
    high_loss_threshold: float = 1.5,
) -> FitDiagnosis:
    """Diagnose under/overfitting of a temporal model from its loss history.

    Applies the same logic as :func:`diagnose_fit` but adapted to loss instead
    of accuracy: a large train-val gap in loss indicates overfit; both losses
    high indicate underfit.

    Args:
        history: Result of :func:`fetch_loss_history_from_mlflow`.
        gap_threshold: Absolute train-val gap in loss above which there is
            overfitting. Default 0.10 (cross-entropy ~ log of probability).
        high_loss_threshold: Loss threshold above which both curves are
            considered high (underfit). Default 1.5 (random over 18 classes is
            ~log(18) = 2.89; a reasonable model drops to < 1.5).

    Returns:
        :class:`FitDiagnosis` with verdict and explanation in accessible
        language.

    Raises:
        ValueError: if the history has no epochs or if there is no val_loss.
    """
    if history.epochs.size == 0:
        raise ValueError("The loss history is empty; it cannot be diagnosed.")
    if history.val_loss_mean.size == 0:
        raise ValueError(
            f"Run {history.run_id} has no val_loss logged; the training did "
            "not open an internal validation split (val_fraction=0)."
        )

    train_loss = float(history.train_loss_mean[-1])
    val_loss = float(history.val_loss_mean[-1])
    # In loss, "large gap" = val_loss >> train_loss (opposite to accuracy).
    gap = val_loss - train_loss

    if gap > gap_threshold:
        verdict: FitVerdict = "overfit"
        explanation = (
            f"Sobreajuste: la val_loss ({val_loss:.3f}) supera a la train_loss "
            f"({train_loss:.3f}) por {gap:.3f} (> {gap_threshold:.2f}). El modelo "
            f"memoriza el conjunto de entrenamiento."
        )
    elif train_loss > high_loss_threshold and val_loss > high_loss_threshold:
        verdict = "underfit"
        explanation = (
            f"Subajuste: train_loss ({train_loss:.3f}) y val_loss "
            f"({val_loss:.3f}) estan ambos por encima de {high_loss_threshold:.2f}. "
            f"El modelo no captura la senal ni en el entrenamiento."
        )
    else:
        verdict = "good_fit"
        explanation = (
            f"Buen ajuste: gap val-train = {gap:.3f} (<= {gap_threshold:.2f}) y "
            f"val_loss = {val_loss:.3f} (<= {high_loss_threshold:.2f}). El modelo "
            f"generaliza de forma consistente con su desempeno en train "
            f"({train_loss:.3f})."
        )

    logger.info(
        "temporal_fit_diagnosed",
        model_kind=history.model_kind,
        run_id=history.run_id,
        verdict=verdict,
        gap=round(gap, 4),
        train_loss=round(train_loss, 4),
        val_loss=round(val_loss, 4),
    )
    return FitDiagnosis(
        verdict=verdict,
        gap=gap,
        train_acc_max=train_loss,  # Reuses the field as train_loss_min.
        val_acc_max=val_loss,  # Reuses the field as val_loss_min.
        explanation=explanation,
    )
