"""Tabular crop classification baseline: Random Forest + XGBoost (US-019).

EPIC 4 library (Avance 3). Trains two tabular models on the combined
feature vector from EPIC 3 (AlphaEarth + spectral indices + temporal
statistics + SRTM + ERA5) with **spatial** cross-validation evaluation
and optional light tuning.

Canonical decisions (plan ``docs/us-planning/us-019.md`` 2.1):

- **D1**: the CV is spatial via :func:`ml.features.spatial_split.build_spatial_kfold`
  (H3 + KMeans + 1 km buffer). ZERO random ``KFold``/``train_test_split``.
- **D2**: only ``RandomForestClassifier`` + ``xgboost.XGBClassifier``.
- **D3**: ``tree_method="hist"``; XGBoost uses ``device="cuda"`` if an
  NVIDIA GPU is available and degrades automatically to CPU otherwise (CI
  without GPU, laptop without CUDA). RandomForest is always CPU (sklearn has
  no GPU backend). The problem (85 k x 187) runs in minutes on either one.
- **D5**: class balancing (``class_weight="balanced"`` for RF,
  frequency-inverse ``sample_weight`` for XGB).
- **D12**: ``LabelEncoder`` on ``class_id`` persisted in the result;
  XGB ``multi:softprob`` requires contiguous labels ``[0, n_classes)``.

The feature dataset does not include parcel geometry; the spatial
centroid is derived from the PASTIS-R ``patch_id`` via
``data/PASTIS-R/metadata.geojson`` (per-patch geometry in EPSG:2154).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import structlog
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.eval.metrics import compute_baseline_metrics
from ml.features.scaler import fit_scaler_on_train
from ml.features.spatial_split import FoldAssignment, build_spatial_kfold

logger = structlog.get_logger(__name__)

__all__ = [
    "BaselineResult",
    "ModelKind",
    "SpatialXGBClassifier",
    "build_estimator",
    "evaluate_with_spatial_cv",
    "train_one_model",
    "tune_baseline",
]

ModelKind = Literal["rf", "xgb", "lgbm"]

# Metadata columns that are NOT features (excluded from the X matrix).
_META_COLS: tuple[str, ...] = (
    "parcel_id",
    "year",
    "patch_id",
    "instance_id",
    "class_id",
    "class_name",
    "fold",
    "n_pixels",
    "area_m2",
    "geometry",
)

# Column suffixes that indicate a join without prior coalesce and are never
# features. Defense in depth over `_META_COLS`: the US-023-preview-v2 bug
# (patch_id_right importance=0.27 in XGB) entered here via a Polars left join.
_META_SUFFIXES: tuple[str, ...] = ("_right", "_left", "_x", "_y")

# Non-agronomic PASTIS-R classes to discard (Background, Void label).
_DROP_CLASS_IDS: tuple[int, ...] = (0, 19)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FEATURES_PATH = (
    _REPO_ROOT / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet"
)
_PASTIS_METADATA_PATH = _REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"

# Documented base hyperparameters (criterion AC-1).
# `max_depth` and `min_samples_leaf` bounded (not None / not 1): an unpruned RF
# over 85k parcels grows down to pure leaves -> a ~700 MB model unmanageable
# for the Model Registry and with severe overfitting. The pruning (depth 20,
# min_samples_leaf 10, 150 trees) keeps the model at ~100-150 MB, loggable,
# without losing material F1 (deviation justified from plan US-019; see handoff).
_RF_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}
_XGB_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "multi:softprob",
    "random_state": 42,
}
# LightGBM (3rd model of the tabular baseline). Hyperparameters aligned with XGB
# for a fair comparison: same effective depth (`num_leaves=63 ~ 2^6`
# with `max_depth=-1`), same `learning_rate=0.05`, same subsample/colsample.
# `class_weight="balanced"` replaces the manual `sample_weight` that XGB requires
# (LGBM does expose the parameter natively, decision D5). LGBM accepts NaN without
# prior imputation but we keep the same `_impute_with` for consistency.
# Note: the PyPI wheel for `lightgbm` does not include a CUDA build; it stays on CPU.
_LGBM_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "objective": "multiclass",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def resolve_xgb_device() -> str:
    """Resolve the XGBoost device based on NVIDIA GPU availability.

    Detects a CUDA GPU via ``nvidia-smi``. If present, returns
    ``"cuda"`` (XGBoost 3.x uses ``tree_method="hist"`` + ``device="cuda"``
    for accelerated training); otherwise, degrades to ``"cpu"`` so the
    baseline runs in CI and on laptops without CUDA (decision D3).

    Returns:
        ``"cuda"`` if a detectable NVIDIA GPU exists, ``"cpu"`` otherwise.
    """
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "cpu"
    try:
        result = subprocess.run(  # noqa: S603 — path resolved with shutil.which
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    if result.returncode == 0 and result.stdout.strip():
        logger.info("xgb_device_resolved", device="cuda", gpu=result.stdout.strip())
        return "cuda"
    return "cpu"


# Light tuning grids (criterion AC-4): 8 combinations per model.
# `max_depth` and `min_samples_leaf` bounded: avoids the ~700 MB RF and the
# overfitting of unpruned trees (see _RF_BASE_PARAMS).
_RF_PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 150],
    "max_depth": [15, 20],
    "min_samples_leaf": [10, 20],
}
_XGB_PARAM_GRID: dict[str, list] = {
    "n_estimators": [300, 400],
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
}
# LightGBM: 8 combinations (2 x 2 x 2). `num_leaves` bounded to [31, 63] to avoid
# growing trees that double the model in memory (same criterion as RF).
_LGBM_PARAM_GRID: dict[str, list] = {
    "n_estimators": [300, 400],
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
}

_METRIC_KEYS: tuple[str, ...] = (
    "f1_macro",
    "f1_weighted",
    "miou",
    "accuracy",
    "cohen_kappa",
)


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Result of training a tabular baseline.

    Attributes:
        model: sklearn/xgboost estimator already fitted on the full
            dataset (labels encoded with ``LabelEncoder``).
        model_kind: ``"rf"`` or ``"xgb"``.
        metrics: The five metrics from :func:`compute_baseline_metrics`
            computed on the out-of-fold predictions of the spatial CV.
        cv_metrics: Map ``{metric: (mean, std)}`` over the
            spatial CV folds.
        feature_cols: Names of the columns used as features, in the
            same order as the columns of the X matrix.
        best_params: Hyperparameters used (from ``GridSearchCV`` if tuning
            occurred, or the base ones otherwise).
        label_classes: Original classes in ``LabelEncoder`` order;
            ``label_classes[i]`` is the real class of label ``i``.
        label_encoder: The fitted ``LabelEncoder``, to decode
            predictions downstream (US-020, inference).
    """

    model: ClassifierMixin
    model_kind: ModelKind
    metrics: dict[str, float]
    cv_metrics: dict[str, tuple[float, float]]
    feature_cols: tuple[str, ...]
    best_params: dict[str, object]
    label_classes: tuple[int, ...]
    label_encoder: LabelEncoder


# ---------------------------------------------------------------------------
# Estimator construction.
# ---------------------------------------------------------------------------


class SpatialXGBClassifier(XGBClassifier):
    """``XGBClassifier`` tolerant to folds missing some classes.

    Under spatial cross-validation the disjoint geographic blocks may leave
    a training fold without the rarest crops, so ``y_train`` is not the
    contiguous ``[0, K)`` range that XGBoost >= 1.6 demands. The base
    estimator then raises ``ValueError: Invalid classes inferred from
    unique values of y`` and that fold is scored ``nan``, silently
    corrupting both the GridSearchCV selection and the spatial-CV metrics.

    This subclass fits a local :class:`LabelEncoder` on each ``fit`` call,
    trains the booster on the remapped contiguous labels, and decodes the
    predictions back to the original (global) label space on ``predict`` /
    ``predict_proba``. To the rest of the pipeline (GridSearchCV, the manual
    fold loop, the interpretability helpers) it behaves like a regular
    ``XGBClassifier`` whose ``classes_`` are the original labels.

    The remapping only takes effect when a fold is missing classes; when all
    classes are present the local encoder is the identity and behaviour is
    unchanged.
    """

    def fit(  # type: ignore[override]
        self, X: np.ndarray, y: np.ndarray, **kwargs: object
    ) -> SpatialXGBClassifier:
        """Fit on locally re-encoded labels so missing classes do not crash.

        The booster keeps the local contiguous ``[0, k)`` labels internally
        (so ``XGBClassifier.classes_`` stays consistent with its own
        validation); the original labels are stored in ``global_classes_``
        and restored on :meth:`predict`.

        Args:
            X: Feature matrix.
            y: Target labels in the original (global) label space.
            **kwargs: Forwarded to :meth:`xgboost.XGBClassifier.fit`
                (e.g. ``sample_weight``).

        Returns:
            The fitted estimator.
        """
        self._local_encoder = LabelEncoder().fit(np.asarray(y))
        self.global_classes_ = self._local_encoder.classes_
        y_local = self._local_encoder.transform(np.asarray(y))
        super().fit(X, y_local, **kwargs)
        return self

    def predict(  # type: ignore[override]
        self, X: np.ndarray, **kwargs: object
    ) -> np.ndarray:
        """Predict and decode back to the original (global) label space."""
        local_pred = super().predict(X, **kwargs)
        decoded = self._local_encoder.inverse_transform(local_pred.astype(int))
        return np.asarray(decoded)


def build_estimator(model: ModelKind, hyperparams: dict[str, object]) -> ClassifierMixin:
    """Instantiate an RF, XGB or LGBM estimator with the given hyperparameters.

    Args:
        model: ``"rf"`` for :class:`RandomForestClassifier`, ``"xgb"``
            for :class:`xgboost.XGBClassifier` or ``"lgbm"`` for
            :class:`lightgbm.LGBMClassifier`.
        hyperparams: Dictionary of constructor hyperparameters.

    Returns:
        The instantiated estimator (unfitted).

    Raises:
        ValueError: if ``model`` is not ``"rf"``, ``"xgb"`` nor ``"lgbm"``.
    """
    if model == "rf":
        return RandomForestClassifier(**hyperparams)
    if model == "xgb":
        # Inject the device (cuda/cpu) if the caller did not set it; allows
        # acceleration on local GPU without breaking CI without CUDA (decision D3).
        xgb_params = dict(hyperparams)
        xgb_params.setdefault("device", resolve_xgb_device())
        # SpatialXGBClassifier re-encodes labels per fit so spatial folds
        # missing rare classes do not raise "Invalid classes inferred".
        return SpatialXGBClassifier(**xgb_params)
    if model == "lgbm":
        # LGBM stays on CPU: the PyPI wheel does not ship a CUDA build. For GPU
        # one would need `pip install lightgbm --config-settings=cmake.define...`
        # with `device_type="gpu"`, out of scope for the baseline.
        return LGBMClassifier(**hyperparams)  # type: ignore[arg-type]
    raise ValueError(f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}.")


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def train_one_model(
    df: pl.DataFrame,
    *,
    model: ModelKind,
    hyperparams: dict[str, object] | None = None,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> BaselineResult:
    """Train a baseline (RF or XGB) with spatial CV evaluation.

    Loads the features, builds the spatial folds with
    :func:`build_spatial_kfold`, evaluates with a per-fold anti-leakage
    scaler, obtains out-of-fold predictions and fits the final model on
    the whole dataset.

    Args:
        df: Polars feature DataFrame (must contain ``parcel_id``,
            ``class_id`` and at least one numeric feature column).
        model: ``"rf"`` or ``"xgb"``.
        hyperparams: Estimator hyperparameters; if ``None`` the
            documented base values are used (``_RF_BASE_PARAMS`` /
            ``_XGB_BASE_PARAMS``).
        k_folds: Number of spatial CV folds (default 5).
        buffer_km: Anti-leakage buffer in km between folds (default 1.0).
        random_state: Deterministic seed.

    Returns:
        A :class:`BaselineResult` with the fitted model, the out-of-fold
        metrics, the per-fold metrics and the feature metadata.

    Raises:
        ValueError: if ``df`` lacks mandatory columns or if no samples
            remain after discarding the non-agronomic classes.
    """
    clean_df = _prepare_dataframe(df)
    feature_cols = _feature_columns(clean_df)
    encoder, y_encoded = _encode_labels(clean_df)

    params = dict(hyperparams) if hyperparams is not None else _base_params(model)
    # No fixed `num_class` for XGB: SpatialXGBClassifier infers the class
    # count from each fold's locally re-encoded labels, so folds missing
    # rare crops do not crash and no global override is needed.

    def factory() -> ClassifierMixin:
        return build_estimator(model, params)

    cv_metrics, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(
        clean_df,
        factory,
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    oof_metrics = compute_baseline_metrics(
        y_true_oof,
        y_pred_oof,
        labels=list(range(len(encoder.classes_))),
    )

    # Final fit over the full dataset (production model). The trees
    # (RF/XGB) are invariant to monotonic scaling, so the final model
    # operates on raw imputed features (without StandardScaler); the
    # CV does scale because `fit_scaler_on_train` is the repo pattern.
    matrix = _impute(_feature_matrix(clean_df, feature_cols))
    final_model = build_estimator(model, params)
    # XGB does not expose `class_weight`: we inject `sample_weight` inverse to
    # frequency (decision D5). LGBM with `class_weight="balanced"` already
    # handles it natively; if the caller removes it, we fall back to sample_weight.
    sample_weight: np.ndarray | None
    if model == "xgb":
        sample_weight = _sample_weights(y_encoded)
    elif model == "lgbm" and "class_weight" not in params:
        sample_weight = _sample_weights(y_encoded)
    else:
        sample_weight = None
    if sample_weight is not None:
        final_model.fit(matrix, y_encoded, sample_weight=sample_weight)
    else:
        final_model.fit(matrix, y_encoded)

    logger.info(
        "baseline_trained",
        model=model,
        n_samples=clean_df.height,
        n_features=len(feature_cols),
        n_classes=len(encoder.classes_),
        f1_macro_oof=oof_metrics["f1_macro"],
    )
    return BaselineResult(
        model=final_model,
        model_kind=model,
        metrics=oof_metrics,
        cv_metrics=cv_metrics,
        feature_cols=feature_cols,
        best_params=params,
        label_classes=tuple(int(c) for c in encoder.classes_),
        label_encoder=encoder,
    )


def tune_baseline(
    df: pl.DataFrame,
    *,
    model: ModelKind,
    param_grid: dict[str, list] | None = None,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    scoring: str = "f1_macro",
    random_state: int = 42,
) -> dict[str, object]:
    """Light hyperparameter tuning via ``GridSearchCV`` over spatial CV.

    The ``cv`` parameter of :class:`GridSearchCV` receives the **list of
    spatial splits** ``(train_idx, test_idx)`` (not an integer), so that
    the tuning respects the geographic partition and introduces no leakage.

    Args:
        df: Polars feature DataFrame.
        model: ``"rf"`` or ``"xgb"``.
        param_grid: Hyperparameter grid; if ``None`` the documented light
            grids are used (8 combinations per model).
        k_folds: Number of spatial CV folds (default 5).
        buffer_km: Anti-leakage buffer in km (default 1.0).
        scoring: Selection metric for ``GridSearchCV`` (default
            ``"f1_macro"``).
        random_state: Deterministic seed.

    Returns:
        The ``best_params_`` dictionary from ``GridSearchCV``.

    Raises:
        ValueError: if ``df`` lacks mandatory columns.
    """
    clean_df = _prepare_dataframe(df)
    feature_cols = _feature_columns(clean_df)
    _, y_encoded = _encode_labels(clean_df)
    matrix = _impute(_feature_matrix(clean_df, feature_cols))

    grid = param_grid if param_grid is not None else _default_grid(model)
    cv_splits = _build_cv_splits(
        clean_df, k_folds=k_folds, buffer_km=buffer_km, random_state=random_state
    )

    base_params = _base_params(model)
    # No fixed `num_class` for XGB: SpatialXGBClassifier re-encodes labels
    # per fold, so the class count is inferred locally and a fold missing
    # rare crops no longer conflicts with a global 18-class setting.
    # Remove from the base estimator the keys that the grid will overwrite.
    for key in grid:
        base_params.pop(key, None)
    estimator = build_estimator(model, base_params)

    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)
    # With XGB on GPU, GridSearchCV uses n_jobs=1: a single GPU cannot
    # serve several fits in parallel and N workers competing for it
    # cause thrashing (each one creates its own CUDA context). XGB boosting
    # already parallelizes internally on the GPU. RF (CPU) does use all cores.
    xgb_on_gpu = model == "xgb" and resolve_xgb_device() == "cuda"
    search_n_jobs = 1 if xgb_on_gpu else -1
    logger.info(
        "baseline_tuning_start",
        model=model,
        n_combos=n_combos,
        n_folds=len(cv_splits),
        n_fits=n_combos * len(cv_splits),
        search_n_jobs=search_n_jobs,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv_splits,
        n_jobs=search_n_jobs,
        refit=True,
        verbose=2,
    )
    search.fit(matrix, y_encoded)
    logger.info(
        "baseline_tuned",
        model=model,
        n_combos=len(search.cv_results_["params"]),
        best_score=float(search.best_score_),
        best_params=search.best_params_,
    )
    return dict(search.best_params_)


def evaluate_with_spatial_cv(
    df: pl.DataFrame,
    model_factory: Callable[[], ClassifierMixin],
    *,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> tuple[dict[str, tuple[float, float]], np.ndarray, np.ndarray]:
    """Evaluate an estimator with anti-leakage spatial cross-validation.

    For each spatial fold it fits a :class:`StandardScaler` on the train
    only (anti-leakage, via :func:`fit_scaler_on_train`), trains a fresh
    estimator and predicts on the fold's test. It aggregates the mean and
    std of the five metrics and also returns the concatenated out-of-fold
    predictions.

    Args:
        df: Already prepared Polars feature DataFrame (see
            :func:`_prepare_dataframe`).
        model_factory: Argument-less callable that returns a fresh
            unfitted estimator (invoked once per fold).
        k_folds: Number of spatial CV folds (default 5).
        buffer_km: Anti-leakage buffer in km (default 1.0).
        random_state: Deterministic seed.

    Returns:
        Tuple ``(cv_metrics, y_true_oof, y_pred_oof)`` where ``cv_metrics``
        is ``{metric: (mean, std)}``, ``y_true_oof`` are the encoded true
        labels concatenated per fold and ``y_pred_oof`` the corresponding
        predictions.
    """
    feature_cols = _feature_columns(df)
    encoder, y_encoded = _encode_labels(df)
    matrix = _feature_matrix(df, feature_cols)
    n_classes = len(encoder.classes_)

    cv_splits = _build_cv_splits(
        df, k_folds=k_folds, buffer_km=buffer_km, random_state=random_state
    )

    per_fold: list[dict[str, float]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    logger.info("spatial_cv_start", n_folds=len(cv_splits), n_classes=n_classes)
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if train_idx.size == 0 or test_idx.size == 0:
            logger.warning("spatial_cv_fold_skipped", fold=fold_idx)
            continue
        logger.info(
            "spatial_cv_fold_start",
            fold=f"{fold_idx + 1}/{len(cv_splits)}",
            n_train=int(train_idx.size),
            n_test=int(test_idx.size),
        )

        scaler, scaler_cols = _fit_fold_scaler(
            df, feature_cols=feature_cols, train_idx=train_idx, fold_idx=fold_idx
        )
        # `fit_scaler_on_train` may drop all-NaN columns: we align the
        # matrix to the columns the scaler knows before `transform`.
        col_idx = np.array([feature_cols.index(c) for c in scaler_cols], dtype=np.int64)
        raw_train = matrix[np.ix_(train_idx, col_idx)]
        raw_test = matrix[np.ix_(test_idx, col_idx)]
        # Anti-leakage imputation: the medians are computed only over train.
        train_medians = _column_medians(raw_train)
        x_train = scaler.transform(_impute_with(raw_train, train_medians))
        x_test = scaler.transform(_impute_with(raw_test, train_medians))
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        estimator = model_factory()
        if _is_xgb(estimator):
            estimator.fit(x_train, y_train, sample_weight=_sample_weights(y_train))
        elif _is_lgbm(estimator) and getattr(estimator, "class_weight", None) is None:
            # LGBM without `class_weight="balanced"` receives the sample_weight
            # inverse to frequency for alignment with XGB (decision D5).
            estimator.fit(x_train, y_train, sample_weight=_sample_weights(y_train))
        else:
            estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)

        fold_metrics = compute_baseline_metrics(y_test, y_pred, labels=list(range(n_classes)))
        per_fold.append(fold_metrics)
        y_true_chunks.append(y_test)
        y_pred_chunks.append(np.asarray(y_pred))
        logger.info(
            "spatial_cv_fold_done",
            fold=f"{fold_idx + 1}/{len(cv_splits)}",
            f1_macro=round(fold_metrics["f1_macro"], 4),
        )

    cv_metrics = _aggregate_fold_metrics(per_fold)
    y_true_oof = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_pred_oof = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
    return cv_metrics, y_true_oof, y_pred_oof


# ---------------------------------------------------------------------------
# Private helpers — loading and cleaning.
# ---------------------------------------------------------------------------


def _load_baseline_dataset(features_path: Path | str | None = None) -> pl.DataFrame:
    """Load the baseline feature parquet from disk.

    Args:
        features_path: Path to the parquet; if ``None`` uses the canonical
            US-018 subset (``feature_selection_parcels_subset.parquet``).

    Returns:
        The raw (uncleaned) Polars DataFrame.

    Raises:
        FileNotFoundError: if the parquet does not exist.
    """
    path = Path(features_path) if features_path is not None else _DEFAULT_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset de features no encontrado en {path}. "
            "Genera el subset con `make feature-selection-subset` o ejecuta "
            "el pipeline de extraccion del EPIC 3."
        )
    return pl.read_parquet(path)


def _prepare_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and clean the feature DataFrame for the baseline.

    Discards the non-agronomic PASTIS-R classes (0 Background, 19 Void) and
    removes rows without ``class_id``.

    Args:
        df: Raw Polars feature DataFrame.

    Returns:
        The filtered DataFrame, ready for training.

    Raises:
        ValueError: if ``parcel_id`` or ``class_id`` are missing, or if no
            rows remain after filtering.
    """
    for col in ("parcel_id", "class_id"):
        if col not in df.columns:
            raise ValueError(f"`df` debe contener la columna obligatoria `{col}`.")

    clean = df.filter(
        pl.col("class_id").is_not_null() & ~pl.col("class_id").is_in(list(_DROP_CLASS_IDS))
    )
    if clean.height == 0:
        raise ValueError("Tras descartar las clases no agronomicas el DataFrame quedo vacio.")

    # The real dataset carries +/-inf in some spectral slopes/ratios;
    # we normalize them to null so the scaler (which only handles NaN) and the
    # downstream imputation manage them uniformly.
    float_cols = [c for c in clean.columns if clean.schema[c] in (pl.Float32, pl.Float64)]
    if float_cols:
        clean = clean.with_columns(
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        )
    return clean


def _feature_columns(df: pl.DataFrame) -> tuple[str, ...]:
    """Return the numeric columns usable as features.

    Excludes the metadata (``_META_COLS``) and any non-numeric column.

    Args:
        df: Already prepared Polars DataFrame.

    Returns:
        Ordered tuple of feature column names.

    Raises:
        ValueError: if no feature column remains.
    """
    cols = [
        c
        for c in df.columns
        if c not in _META_COLS and not c.endswith(_META_SUFFIXES) and df.schema[c].is_numeric()
    ]
    if not cols:
        raise ValueError("No se encontraron columnas numericas de feature en `df`.")
    return tuple(cols)


def _feature_matrix(df: pl.DataFrame, feature_cols: tuple[str, ...]) -> np.ndarray:
    """Extract the feature matrix as a float64 ``np.ndarray``.

    Args:
        df: Already prepared Polars DataFrame.
        feature_cols: Columns to select, in order.

    Returns:
        Matrix ``(n_samples, n_features)`` of dtype float64.
    """
    return df.select(feature_cols).to_numpy().astype(np.float64)


def _encode_labels(df: pl.DataFrame) -> tuple[LabelEncoder, np.ndarray]:
    """Encode ``class_id`` to contiguous labels ``[0, n_classes)``.

    PASTIS-R has no contiguous class_ids after discarding 0 and 19; XGBoost
    ``multi:softprob`` requires contiguous labels (decision D12).

    Args:
        df: Already prepared Polars DataFrame.

    Returns:
        Tuple ``(encoder, y_encoded)`` with the fitted ``LabelEncoder`` and
        the encoded label vector.
    """
    raw = df.get_column("class_id").to_numpy().astype(np.int64)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(raw)
    return encoder, y_encoded.astype(np.int64)


def _base_params(model: ModelKind) -> dict[str, object]:
    """Return a copy of the base hyperparameters for the given model."""
    if model == "rf":
        return dict(_RF_BASE_PARAMS)
    if model == "xgb":
        return dict(_XGB_BASE_PARAMS)
    if model == "lgbm":
        return dict(_LGBM_BASE_PARAMS)
    raise ValueError(f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}.")


def _default_grid(model: ModelKind) -> dict[str, list]:
    """Return a copy of the light tuning grid for the given model."""
    if model == "rf":
        grid = _RF_PARAM_GRID
    elif model == "xgb":
        grid = _XGB_PARAM_GRID
    elif model == "lgbm":
        grid = _LGBM_PARAM_GRID
    else:
        raise ValueError(f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}.")
    return {k: list(v) for k, v in grid.items()}


def _sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Compute per-sample weights inversely proportional to frequency.

    Reproduces the effect of ``class_weight="balanced"`` for XGBoost, which
    does not expose that parameter (decision D5).

    Args:
        y_encoded: Encoded label vector.

    Returns:
        Weight vector ``(n_samples,)`` float64.
    """
    classes, counts = np.unique(y_encoded, return_counts=True)
    n_samples = y_encoded.size
    n_classes = classes.size
    weight_per_class = {
        int(c): n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts, strict=True)
    }
    return np.array([weight_per_class[int(c)] for c in y_encoded], dtype=np.float64)


def _is_xgb(estimator: ClassifierMixin) -> bool:
    """Indicate whether ``estimator`` is an ``XGBClassifier``."""
    return isinstance(estimator, XGBClassifier)


def _is_lgbm(estimator: ClassifierMixin) -> bool:
    """Indicate whether ``estimator`` is an ``LGBMClassifier``."""
    return isinstance(estimator, LGBMClassifier)


def _column_medians(matrix: np.ndarray) -> np.ndarray:
    """Compute the median of each column ignoring NaN and infinities.

    Args:
        matrix: Matrix ``(n_samples, n_features)`` that may contain NaN
            or ``+/-inf`` (the real dataset carries ``inf`` in some
            spectral slopes/ratios).

    Returns:
        Vector ``(n_features,)`` of medians; ``0.0`` for entirely
        non-finite columns.
    """
    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    medians = np.nanmedian(finite, axis=0)
    return np.where(np.isnan(medians), 0.0, medians)


def _impute_with(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Impute non-finite values using a precomputed median vector.

    Treats ``NaN`` and ``+/-inf`` equally: sklearn accepts neither.

    Args:
        matrix: Matrix ``(n_samples, n_features)`` that may contain NaN
            or infinities.
        medians: Vector ``(n_features,)`` of imputation values (the
            train-split medians, to avoid leakage into test).

    Returns:
        A copy of the matrix with all finite values.
    """
    out = np.array(matrix, dtype=np.float64, copy=True)
    non_finite = ~np.isfinite(out)
    if not non_finite.any():
        return out
    bad_idx = np.where(non_finite)
    out[bad_idx] = np.take(medians, bad_idx[1])
    return out


def _impute(matrix: np.ndarray) -> np.ndarray:
    """Impute NaN with the median of each column of ``matrix`` itself.

    Shortcut for the final fit on the full dataset, where the train/test
    separation does not apply.

    Args:
        matrix: Matrix ``(n_samples, n_features)`` that may contain NaN.

    Returns:
        A copy of the matrix without NaN.
    """
    return _impute_with(matrix, _column_medians(matrix))


# ---------------------------------------------------------------------------
# Private helpers — spatial CV.
# ---------------------------------------------------------------------------


_SPATIAL_FOLDS_CACHE_DIR = Path("data/test_fixtures")


def _spatial_folds_cache_path(
    n_rows: int, k_folds: int, buffer_km: float, random_state: int
) -> Path:
    """Path of the cache parquet for the spatial splits.

    The key includes the number of rows, ``k``, the buffer and the seed:
    any change invalidates the cache and forces a recompute.
    """
    buffer_tag = f"{buffer_km:g}".replace(".", "p")
    name = f"baseline_spatial_folds_n{n_rows}_k{k_folds}_b{buffer_tag}_s{random_state}.parquet"
    return _SPATIAL_FOLDS_CACHE_DIR / name


def _load_cached_cv_splits(path: Path) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Read the cached spatial splits, or ``None`` if they do not exist."""
    if not path.exists():
        return None
    try:
        cached = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        logger.warning("spatial_folds_cache_unreadable", path=str(path), error=str(exc))
        return None
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in sorted(cached["fold"].unique().to_list()):
        fold_df = cached.filter(pl.col("fold") == fold_idx)
        train_idx = fold_df.filter(pl.col("split") == "train")["idx"].to_numpy()
        test_idx = fold_df.filter(pl.col("split") == "test")["idx"].to_numpy()
        splits.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    logger.info("spatial_folds_cache_hit", path=str(path), n_folds=len(splits))
    return splits


def _save_cached_cv_splits(path: Path, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Persist the spatial splits to parquet for future runs."""
    rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        rows.extend({"fold": fold_idx, "split": "train", "idx": int(i)} for i in train_idx)
        rows.extend({"fold": fold_idx, "split": "test", "idx": int(i)} for i in test_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    logger.info("spatial_folds_cache_saved", path=str(path), n_folds=len(splits))


def _build_cv_splits(
    df: pl.DataFrame,
    *,
    k_folds: int,
    buffer_km: float,
    random_state: int,
    use_cache: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert the spatial folds into positional ``(train, test)`` splits.

    Builds a :class:`geopandas.GeoDataFrame` with a synthetic integer
    ``parcel_id`` (position in the DataFrame) and the PASTIS-R patch
    centroid geometry, calls :func:`build_spatial_kfold` and translates the
    ``parcel_id`` of each :class:`FoldAssignment` to positional indices.

    ``build_spatial_kfold`` is O(N^2) due to the anti-leakage buffer; over
    85k parcels it takes minutes. That is why the splits are cached in a
    parquet (key: n_rows + k + buffer + seed) and reused in subsequent
    runs (handoff US-019 R3).

    Args:
        df: Already prepared Polars DataFrame.
        k_folds: Number of folds.
        buffer_km: Anti-leakage buffer in km.
        random_state: Deterministic seed.
        use_cache: If ``True`` (default) reads/writes the splits cache.

    Returns:
        List of tuples ``(train_idx, test_idx)`` of positional index
        arrays, one per fold with samples on both sides.
    """
    cache_path = _spatial_folds_cache_path(df.height, k_folds, buffer_km, random_state)
    if use_cache:
        cached = _load_cached_cv_splits(cache_path)
        if cached is not None:
            return cached

    logger.info(
        "spatial_folds_building",
        n_rows=df.height,
        k_folds=k_folds,
        buffer_km=buffer_km,
        note="O(N^2) — puede tardar minutos en datasets grandes",
    )
    parcels_gdf = _build_parcels_geodataframe(df)
    folds: list[FoldAssignment] = build_spatial_kfold(
        parcels_gdf,
        k=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    n_rows = df.height
    all_idx = np.arange(n_rows, dtype=np.int64)
    for fold in folds:
        # train_ids of the FoldAssignment already equal positional indices
        # because the GeoDataFrame uses the position as a synthetic `parcel_id`.
        train_pool = np.array(sorted(fold.train_ids) + sorted(fold.val_ids), dtype=np.int64)
        test_idx = np.array(sorted(fold.test_ids), dtype=np.int64)
        if train_pool.size == 0 or test_idx.size == 0:
            continue
        # We filter for safety against out-of-range ids.
        train_idx = train_pool[np.isin(train_pool, all_idx)]
        test_idx = test_idx[np.isin(test_idx, all_idx)]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError(
            "El CV espacial no produjo ningun fold con train y test no vacios. "
            "Revisa el numero de parcelas o reduce `k_folds`."
        )
    if use_cache:
        _save_cached_cv_splits(cache_path, splits)
    return splits


def _build_parcels_geodataframe(df: pl.DataFrame):  # type: ignore[no-untyped-def]
    """Build the parcels GeoDataFrame for the spatial CV.

    The feature dataset does not include geometry; the centroid is derived
    from the PASTIS-R ``patch_id`` via ``data/PASTIS-R/metadata.geojson``.
    Each parcel receives a synthetic ``parcel_id`` equal to its position in
    the DataFrame so that folds can be translated to indices.

    Args:
        df: Already prepared Polars DataFrame.

    Returns:
        A ``GeoDataFrame`` in EPSG:4326 with ``parcel_id`` (position) and
        ``geometry`` (patch centroid, or deterministic jitter if the
        metadata is not available).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    n_rows = df.height
    positions = np.arange(n_rows, dtype=np.int64)

    patch_centroids = _load_patch_centroids()
    if patch_centroids is not None and "patch_id" in df.columns:
        patch_ids = df.get_column("patch_id").to_numpy()
        coords = np.array(
            [patch_centroids.get(int(p), (np.nan, np.nan)) for p in patch_ids],
            dtype=np.float64,
        )
    else:
        coords = np.full((n_rows, 2), np.nan, dtype=np.float64)

    # Deterministic fallback: if the metadata or some patch is missing, distribute
    # the centroids on a pseudo-random grid stable per patch_id.
    missing = np.isnan(coords).any(axis=1)
    if missing.any():
        logger.warning(
            "spatial_cv_centroid_fallback",
            n_missing=int(missing.sum()),
            note="metadata.geojson ausente o incompleto; rejilla determinista por patch.",
        )
        key = df.get_column("patch_id").to_numpy() if "patch_id" in df.columns else positions
        rng = np.random.default_rng(20240519)
        # Centroids in a box over continental France (PASTIS-R).
        grid = rng.uniform(low=[-1.0, 43.0], high=[7.0, 49.0], size=(n_rows, 2))
        # Ensure that parcels from the same patch share a centroid.
        unique_keys, inverse = np.unique(key, return_inverse=True)
        per_key = rng.uniform(low=[-1.0, 43.0], high=[7.0, 49.0], size=(unique_keys.size, 2))
        grid = per_key[inverse]
        coords[missing] = grid[missing]

    geometry = [Point(float(lon), float(lat)) for lon, lat in coords]
    return gpd.GeoDataFrame(
        {"parcel_id": positions},
        geometry=geometry,
        crs="EPSG:4326",
    )


def _load_patch_centroids() -> dict[int, tuple[float, float]] | None:
    """Load the PASTIS-R patch centroids from ``metadata.geojson``.

    Returns:
        Map ``{patch_id: (lon, lat)}`` in EPSG:4326, or ``None`` if the
        metadata is not available on disk.
    """
    if not _PASTIS_METADATA_PATH.exists():
        return None
    try:
        import geopandas as gpd

        meta = gpd.read_file(_PASTIS_METADATA_PATH)
        # Centroid in a projected CRS (3857) to avoid the geopandas UserWarning
        # about geometric operations in a geographic CRS; then it is
        # reprojected to 4326 (lat/lng), which is what the consumer expects.
        centroids = meta.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
        id_col = "ID_PATCH" if "ID_PATCH" in meta.columns else meta.columns[0]
        return {
            int(pid): (float(geom.x), float(geom.y))
            for pid, geom in zip(meta[id_col], centroids, strict=True)
        }
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover
        logger.warning("pastis_metadata_load_failed", error=str(exc))
        return None


def _fit_fold_scaler(
    df: pl.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    train_idx: np.ndarray,
    fold_idx: int,
) -> tuple[StandardScaler, tuple[str, ...]]:
    """Fit a :class:`StandardScaler` on the fold's train only.

    Reuses :func:`fit_scaler_on_train` (anti-leakage); the scaler is
    persisted in a per-fold temporary file that is discarded.

    Args:
        df: Already prepared Polars DataFrame.
        feature_cols: Feature columns.
        train_idx: Positional indices of the fold's train.
        fold_idx: Fold index (to name the temporary file).

    Returns:
        Tuple ``(scaler, scaler_cols)`` with the fitted ``StandardScaler``
        and the tuple of columns it effectively knows (may be a subset of
        ``feature_cols`` if there were all-NaN columns).
    """
    import tempfile

    # `fit_scaler_on_train` filters by `parcel_id`; we substitute that column
    # with the row position to align with the positional `train_idx`.
    positional = (
        df.drop("parcel_id")
        .with_row_index(name="parcel_id")
        .with_columns(pl.col("parcel_id").cast(pl.Int64))
    )
    train_ids = tuple(int(i) for i in train_idx)
    with tempfile.TemporaryDirectory() as tmp:
        scaler_path = Path(tmp) / f"fold_{fold_idx}_scaler.joblib"
        scaler = fit_scaler_on_train(
            positional,
            train_ids,
            feature_cols,
            scaler_path=scaler_path,
        )
    meta = getattr(scaler, "_agrosat_meta", {})
    scaler_cols = tuple(meta.get("feature_cols", feature_cols))
    return scaler, scaler_cols


def _aggregate_fold_metrics(
    per_fold: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Aggregate the per-fold metrics into ``{metric: (mean, std)}``.

    Args:
        per_fold: List of metric dictionaries, one per fold.

    Returns:
        Map ``{metric: (mean, std)}`` over the five metrics;
        ``(nan, nan)`` for each metric if there were no valid folds.
    """
    if not per_fold:
        return {key: (float("nan"), float("nan")) for key in _METRIC_KEYS}
    return {
        key: (
            float(np.mean([fold[key] for fold in per_fold])),
            float(np.std([fold[key] for fold in per_fold])),
        )
        for key in _METRIC_KEYS
    }
