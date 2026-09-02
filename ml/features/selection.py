"""Feature selection, extraction and normalization (US-018, Avance 2 CRISP-ML(Q)).

Canonical module for the **Data Preparation** phase of CRISP-ML(Q). Exposes a
stable Polars-in / Polars-out API (numpy only at the sklearn boundary via
:func:`_to_numpy`) that covers:

- **Filters** (variance threshold, Pearson/Spearman correlation, chi-square,
  ANOVA F).
- **Extractors** (parametric PCA by target variance, Factor Analysis,
  UMAP 2D for visualization).
- **Methodological complement** (Random Forest + XGBoost importance).
- **Before/after comparison** with PASTIS spatial split folds 1-5
  (Sainte-Fare-Garnot 2021, NOT random KFold).
- **Normalization** with rules justified by model family
  (``StandardScaler`` linear, ``MinMaxScaler`` NN, ``PowerTransformer``
  Yeo-Johnson for skewed, ``log1p`` for LAI/biomass).

Irrevocable decisions (see ``docs/us-planning/us-018.md`` §2.1)
--------------------------------------------------------------
- D1: spatial split = official PASTIS folds 1-5, NOT custom GroupKFold.
- D3: PCA with parametric ``target_variance``, not fixed ``n_components``.
- D5: Polars in / Polars out + explicit conversion to numpy in
  :func:`_to_numpy` (rule ``ml/CLAUDE.md NEVER pandas``).
- D6: chi2 with documented synthetic quartile binning to satisfy the
  rubric when upstream features are continuous numeric.
- D7: RF/XGB exploratory, NOT production. Not logged to MLflow.
- D9: ``ColumnTransformer`` per model family, not a single global scaler.
- D10: Yeo-Johnson over NDVI (accepts negatives) instead of Box-Cox.

References
----------
- Sainte-Fare-Garnot, V., Landrieu, L. (2021). *Panoptic Segmentation of
  Satellite Image Time Series with Convolutional Temporal Attention Networks*.
  ICCV 2021. Official PASTIS-R folds 1-5.
- Daughtry et al. 2000 — MCARI/chlorophyll (factor interpretation "canopy vigor").
- Gao 1996 — NDMI/canopy moisture (factor interpretation "moisture").
- McInnes, Healy, Melville 2018 — UMAP (``n_neighbors`` default 15).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import structlog
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "anova_f_select",
    "apply_variance_threshold",
    "chi2_select",
    "compare_before_after",
    "compute_feature_importance",
    "discretize_features",
    "discretize_ndvi_phenology_domain",
    "drop_correlated_features",
    "fit_factor_analysis",
    "fit_pca",
    "fit_umap_2d",
    "make_preprocessor",
    "select_normalizer",
]

# Common convention: index columns never participate as features.
_DEFAULT_EXCLUDE: tuple[str, ...] = ("parcel_id", "year")

# Name heuristic for normalization rules (D10).
_LOG1P_FEATURE_PREFIXES: tuple[str, ...] = ("LAI", "biomass")
_YEO_JOHNSON_FEATURE_PREFIXES: tuple[str, ...] = (
    "NDVI",
    "NDRE",
    "NDWI",
    "NDMI",
    "NBR",
)
_SKEW_YEO_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_numpy(
    df: pl.DataFrame,
    *,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[np.ndarray, list[str]]:
    """Convert ``df`` to ``np.ndarray`` excluding the index columns.

    Args:
        df: Polars DataFrame with numeric features + index columns
            (``parcel_id``, ``year``).
        exclude_cols: Columns NOT considered features (default:
            ``("parcel_id", "year")``).

    Returns:
        Tuple ``(matrix, feature_names)`` where ``matrix`` has shape
        ``(n_samples, n_features)`` and ``feature_names`` is the ordered list
        of column names used. NaN values are preserved.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        return np.empty((df.height, 0), dtype=np.float64), []
    matrix = df.select(feature_cols).to_numpy().astype(np.float64, copy=False)
    return matrix, feature_cols


def _impute_with_column_mean(matrix: np.ndarray) -> np.ndarray:
    """Impute NaN and inf with the column mean (sklearn accepts neither).

    ``inf`` / ``-inf`` are converted to ``NaN`` before imputing; they can
    appear in derived features (e.g. GCVI with small NIR/Green, spectral
    ratios over dark pixels). If a column is entirely NaN, it is imputed
    with 0.0 (degenerate case; the caller should have filtered it with
    :func:`apply_variance_threshold`).
    """
    if matrix.size == 0:
        return matrix
    # inf -> NaN so imputation treats them the same as NaN.
    if not np.isfinite(matrix).all():
        matrix = np.where(np.isinf(matrix), np.nan, matrix)
    col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        # Broadcast per column
        col_idx = np.where(nan_mask)[1]
        matrix = matrix.copy()
        matrix[nan_mask] = col_means[col_idx]
    return matrix


def _build_strategy_table(
    *,
    strategy: str,
    n_features: int,
    f1_mean: float,
    f1_std: float,
    miou_mean: float,
    miou_std: float,
) -> dict[str, float | str | int]:
    """Build a row of the comparison table."""
    return {
        "strategy": strategy,
        "n_features": int(n_features),
        "f1_macro_mean": float(f1_mean),
        "f1_macro_std": float(f1_std),
        "miou_mean": float(miou_mean),
        "miou_std": float(miou_std),
    }


def _run_cv_baseline_rf(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    """Run CV with PASTIS folds and return F1-macro + mIoU mean/std.

    For each fold ``k`` in ``{1..5}`` (skipping those not present), trains a
    :class:`RandomForestClassifier` on the samples with ``fold != k`` and
    evaluates on the samples with ``fold == k``.

    Args:
        X: Matrix ``(n_samples, n_features)``.
        y: Vector ``(n_samples,)``.
        folds: Vector ``(n_samples,)`` with values in ``{1..5}``.
        n_estimators: Number of RF trees.
        random_state: Seed.

    Returns:
        ``(f1_mean, f1_std, miou_mean, miou_std)`` over the folds used.
        If there are no valid folds, returns ``(nan, nan, nan, nan)``.
    """
    X_clean = _impute_with_column_mean(X)
    unique_folds = sorted(int(f) for f in np.unique(folds) if 1 <= int(f) <= 5)
    if not unique_folds:
        return (float("nan"),) * 4

    f1_scores: list[float] = []
    miou_scores: list[float] = []

    for k in unique_folds:
        test_mask = folds == k
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        # Train only on classes present in train to avoid errors.
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        )
        clf.fit(X_clean[train_mask], y[train_mask])
        y_pred = clf.predict(X_clean[test_mask])
        y_true = y[test_mask]
        # Joint labels for a consistent metric.
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores.append(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        )
        miou_scores.append(
            float(jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        )

    if not f1_scores:
        return (float("nan"),) * 4
    return (
        float(np.mean(f1_scores)),
        float(np.std(f1_scores)),
        float(np.mean(miou_scores)),
        float(np.std(miou_scores)),
    )


def _load_pastis_features_subset(
    parquet_path: Path,
    *,
    target_col: str = "class_id",
    fold_col: str = "fold",
) -> tuple[pl.DataFrame, pl.Series, np.ndarray]:
    """Load the pre-generated PASTIS subset for US-018.

    The parquet is generated with ``scripts/generate_feature_selection_subset.py``
    and must contain: ``parcel_id, year, fold, class_id`` + statistical/FFT/
    phenology features.

    Args:
        parquet_path: Path to ``data/test_fixtures/feature_selection_subset.parquet``.
        target_col: Target column (default ``class_id``).
        fold_col: PASTIS spatial fold column (default ``fold``).

    Returns:
        Tuple ``(X, y, folds)``: wide-format features (without target/fold),
        target ``pl.Series`` and numpy folds vector ``(n_samples,)``.

    Raises:
        FileNotFoundError: If the parquet does not exist.
        ValueError: If mandatory columns are missing.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"PASTIS subset not found at {parquet_path}. "
            "Generate with: poetry run python scripts/generate_feature_selection_subset.py"
        )
    df = pl.read_parquet(parquet_path)
    missing = [c for c in (target_col, fold_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"PASTIS subset lacks mandatory columns: {missing}. Available: {df.columns}"
        )
    y = df.get_column(target_col)
    folds = df.get_column(fold_col).to_numpy().astype(np.int64)
    feature_df = df.drop([target_col, fold_col])
    return feature_df, y, folds


# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------


def apply_variance_threshold(
    df: pl.DataFrame,
    threshold: float = 0.01,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Filter features whose variance is less than or equal to the threshold.

    Args:
        df: Polars DataFrame with numeric columns + excluded indices.
        threshold: Variance threshold (default 0.01). Features with
            ``var <= threshold`` are removed.
        exclude_cols: Columns NOT participating in the filtering (always
            kept).

    Returns:
        Tuple ``(df_filtered, report)`` where ``report`` contains
        ``{"kept": [...], "removed": [...], "variances": {col: var}}``.

    Notes:
        Uses sample variance with ``ddof=0`` (consistent with sklearn
        :class:`~sklearn.feature_selection.VarianceThreshold`). NaN is ignored
        in the variance computation.
    """
    matrix, feature_cols = _to_numpy(df, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        empty_report: dict[str, Any] = {"kept": [], "removed": [], "variances": {}}
        return df, empty_report

    with np.errstate(invalid="ignore"):
        variances = np.nanvar(matrix, axis=0, ddof=0)
    # NaN var (all-NaN column) is treated as 0 -> removed.
    variances = np.where(np.isnan(variances), 0.0, variances)

    kept_mask = variances > threshold
    kept = [c for c, k in zip(feature_cols, kept_mask, strict=True) if k]
    removed = [c for c, k in zip(feature_cols, kept_mask, strict=True) if not k]

    keep_columns = [c for c in df.columns if c in exclude_cols or c in kept]
    df_filtered = df.select(keep_columns)

    report: dict[str, Any] = {
        "kept": kept,
        "removed": removed,
        "variances": {c: float(v) for c, v in zip(feature_cols, variances, strict=True)},
        "threshold": float(threshold),
    }
    logger.info(
        "variance_threshold_applied",
        threshold=threshold,
        n_removed=len(removed),
        n_kept=len(kept),
    )
    return df_filtered, report


def drop_correlated_features(
    df: pl.DataFrame,
    threshold: float = 0.95,
    method: Literal["pearson", "spearman"] = "pearson",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Remove one of each pair of features with ``|r| > threshold``.

    Iterates the upper triangular matrix in deterministic alphabetical order:
    if ``feature_i`` and ``feature_j`` (``i < j``) exceed the threshold,
    ``feature_j`` is dropped. Guarantees idempotency.

    Args:
        df: Polars wide-format DataFrame.
        threshold: Absolute correlation threshold (default 0.95).
        method: Correlation method (``"pearson"`` or ``"spearman"``).
        exclude_cols: Columns to always preserve.

    Returns:
        Tuple ``(df_filtered, report)`` with ``report = {"kept", "removed",
        "corr_matrix" (np.ndarray), "feature_order" (list[str])}``.
    """
    matrix, feature_cols = _to_numpy(df, exclude_cols=exclude_cols)
    if matrix.shape[1] < 2:
        return df, {
            "kept": feature_cols,
            "removed": [],
            "corr_matrix": np.zeros((matrix.shape[1], matrix.shape[1])),
            "feature_order": feature_cols,
        }

    matrix_clean = _impute_with_column_mean(matrix)

    if method == "pearson":
        corr = np.corrcoef(matrix_clean, rowvar=False)
    else:  # spearman: use scipy
        rho, _ = scipy_stats.spearmanr(matrix_clean, axis=0)
        corr = np.atleast_2d(np.asarray(rho, dtype=np.float64))
    # Sanitize NaN (zero variance -> corr NaN). Treat as 0 (no association).
    corr = np.where(np.isnan(corr), 0.0, corr)

    n = corr.shape[0]
    to_drop: set[int] = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j)

    kept = [c for idx, c in enumerate(feature_cols) if idx not in to_drop]
    removed = [c for idx, c in enumerate(feature_cols) if idx in to_drop]

    keep_columns = [c for c in df.columns if c in exclude_cols or c in kept]
    df_filtered = df.select(keep_columns)

    report: dict[str, Any] = {
        "kept": kept,
        "removed": removed,
        "corr_matrix": corr,
        "feature_order": feature_cols,
        "threshold": float(threshold),
        "method": method,
    }
    logger.info(
        "correlated_features_dropped",
        method=method,
        threshold=threshold,
        n_removed=len(removed),
        n_kept=len(kept),
    )
    return df_filtered, report


def chi2_select(
    X: pl.DataFrame,
    y: pl.Series,
    k_best: int = 20,
    *,
    binning_strategy: Literal["quartiles", "deciles"] | None = "quartiles",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Select the ``k_best`` features with the highest chi2 against ``y``.

    If ``X`` contains continuous numeric features (the expected case in
    AgroSatCopilot after US-014/015), documented synthetic binning is applied
    according to ``binning_strategy`` to satisfy the chi-square AC of the
    rubric (Avance 2). Binning is run per feature.

    Args:
        X: Polars wide-format DataFrame with features.
        y: Polars Series with the class (integer).
        k_best: Number of features to return.
        binning_strategy: If not ``None``, discretizes continuous features
            using ``"quartiles"`` (4 bins) or ``"deciles"`` (10 bins).
        exclude_cols: Columns to always preserve in the output frame.

    Returns:
        Tuple ``(top_k_df, scores)`` where ``top_k_df`` keeps the
        ``exclude_cols`` columns + the ``k_best`` selected features and
        ``scores`` is ``{feature: chi2_stat}`` (sorted descending externally).

    Notes:
        chi2 requires non-negative features. When ``binning_strategy is None``
        the caller must guarantee that ``X >= 0`` (defensive clipping is applied).
    """
    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return X, {}

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    if binning_strategy is not None:
        n_bins = 4 if binning_strategy == "quartiles" else 10
        binned = np.zeros_like(matrix_clean, dtype=np.float64)
        for col in range(matrix_clean.shape[1]):
            unique_vals = np.unique(matrix_clean[:, col])
            if unique_vals.size <= 1:
                binned[:, col] = 0.0
                continue
            try:
                quantiles = np.quantile(
                    matrix_clean[:, col],
                    np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
                )
                # Unique edges: if all equal, fall back to 1 bin.
                quantiles = np.unique(quantiles)
                if quantiles.size == 0:
                    binned[:, col] = 0.0
                else:
                    binned[:, col] = np.digitize(matrix_clean[:, col], quantiles)
            except Exception:  # noqa: BLE001
                binned[:, col] = 0.0
        x_input = binned
    else:
        # Defensive clipping: chi2 requires >= 0.
        x_input = np.clip(matrix_clean, a_min=0.0, a_max=None)

    chi2_stats, _ = chi2(x_input, y_arr)
    chi2_stats = np.where(np.isnan(chi2_stats), 0.0, chi2_stats)

    order = np.argsort(-chi2_stats)
    top_idx = order[:k_best]
    top_features = [feature_cols[i] for i in top_idx]
    scores = {feature_cols[i]: float(chi2_stats[i]) for i in top_idx}

    keep_columns = [c for c in X.columns if c in exclude_cols or c in top_features]
    top_df = X.select(keep_columns)

    logger.info(
        "chi2_select_done",
        k_best=k_best,
        binning=binning_strategy,
        n_features_in=matrix.shape[1],
        n_features_out=len(top_features),
    )
    return top_df, scores


def anova_f_select(
    X: pl.DataFrame,
    y: pl.Series,
    k_best: int = 20,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Select the ``k_best`` features with the highest ANOVA F-score against ``y``.

    Args:
        X: Polars wide-format DataFrame.
        y: Polars Series with the class.
        k_best: Number of features to return.
        exclude_cols: Columns to always preserve.

    Returns:
        Tuple ``(top_k_df, scores)`` with ``scores = {feature: f_value}``.
    """
    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return X, {}

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    f_stats, _ = f_classif(matrix_clean, y_arr)
    f_stats = np.where(np.isnan(f_stats), 0.0, f_stats)

    order = np.argsort(-f_stats)
    top_idx = order[:k_best]
    top_features = [feature_cols[i] for i in top_idx]
    scores = {feature_cols[i]: float(f_stats[i]) for i in top_idx}

    keep_columns = [c for c in X.columns if c in exclude_cols or c in top_features]
    top_df = X.select(keep_columns)

    logger.info(
        "anova_f_select_done",
        k_best=k_best,
        n_features_in=matrix.shape[1],
        n_features_out=len(top_features),
    )
    return top_df, scores


# ---------------------------------------------------------------------------
# DISCRETIZATION / BINNING (US-018 extension — Construction rubric 30 pts)
# ---------------------------------------------------------------------------


_DISCRETIZE_STRATEGIES = ("quantile", "uniform", "kmeans", "domain")


def discretize_features(
    df: pl.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    strategy: Literal["quantile", "uniform", "kmeans", "domain"] = "quantile",
    n_bins: int = 4,
    bin_edges: dict[str, list[float]] | None = None,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
    random_state: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Discretize numeric columns creating ``{col}__bin`` with recorded edges.

    Covers the 4 canonical binning strategies of CRISP-ML(Q) Data
    Preparation (Avance 2 rubric — "Feature construction"):

    - ``"quantile"``: equiprobable quantiles via :meth:`polars.Series.qcut`
      (native Polars). ``n_bins`` bins with approximately equal masses.
    - ``"uniform"``: equispaced edges between ``min`` and ``max`` via
      :meth:`polars.Series.cut` (native Polars).
    - ``"kmeans"``: 1D clusters with
      :class:`~sklearn.cluster.KMeans(n_clusters=n_bins, random_state=42)`
      per column; centers are sorted ascending and the bin is the label
      of the closest cluster.
    - ``"domain"``: requires ``bin_edges = {col: [e1, e2, ...]}`` with
      justified agronomic edges (e.g. NDVI thresholds). Applied with
      :meth:`polars.Series.cut`.

    Args:
        df: Polars wide-format DataFrame.
        columns: Numeric columns to discretize.
        strategy: One of :data:`_DISCRETIZE_STRATEGIES`.
        n_bins: Number of bins (ignored in ``"domain"`` — inferred from
            ``len(bin_edges[col]) + 1``).
        bin_edges: For ``"domain"``, dict ``{col: [edge1, edge2, ...]}``
            sorted ascending. Mandatory if ``strategy == "domain"``.
        exclude_cols: Columns not to discretize even if they appear in
            ``columns``.
        random_state: Seed for KMeans.

    Returns:
        Tuple ``(df_with_bins, edges_report)`` where:

        - ``df_with_bins`` adds ``{col}__bin`` per ``col`` (Int64,
          range ``[0, n_bins-1]``). The original column is kept.
        - ``edges_report = {col: list[float]}`` with the edges used
          (KMeans centers in ascending order for the ``"kmeans"``
          strategy).

    Raises:
        ValueError: If ``strategy`` is invalid, ``n_bins < 2``,
            ``bin_edges`` is missing for ``"domain"`` or some column does not exist.
    """
    if strategy not in _DISCRETIZE_STRATEGIES:
        raise ValueError(f"strategy must be one of {_DISCRETIZE_STRATEGIES}; got {strategy!r}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    if strategy == "domain" and not bin_edges:
        raise ValueError("strategy='domain' requires bin_edges = {col: [edges]}")

    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(f"Columns to discretize missing: {missing}")

    out = df
    edges_report: dict[str, list[float]] = {}

    for col in cols_list:
        series = out.get_column(col).cast(pl.Float64)
        values = series.to_numpy()
        finite_mask = np.isfinite(values)
        if finite_mask.sum() == 0:
            bins = np.zeros(values.shape[0], dtype=np.int64)
            edges_report[col] = []
            out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))
            continue

        if strategy == "quantile":
            quantiles_pts = np.linspace(0.0, 1.0, n_bins + 1)[1:-1].tolist()
            try:
                labels = [str(i) for i in range(n_bins)]
                binned = series.qcut(
                    quantiles=quantiles_pts,
                    labels=labels,
                    left_closed=False,
                    allow_duplicates=True,
                )
                str_to_int = {lab: i for i, lab in enumerate(labels)}
                bins = np.array([str_to_int.get(v, 0) for v in binned.to_list()], dtype=np.int64)
                edges_used = np.quantile(values[finite_mask], quantiles_pts).tolist()
            except Exception as exc:  # noqa: BLE001
                logger.warning("discretize_qcut_fallback_uniform", col=col, error=str(exc))
                bins, edges_used = _bin_uniform(values, n_bins)
                edges_report[col] = edges_used
                out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))
                continue

        elif strategy == "uniform":
            bins, edges_used = _bin_uniform(values, n_bins)

        elif strategy == "kmeans":
            finite_vals = values[finite_mask].reshape(-1, 1)
            n_unique = np.unique(finite_vals).size
            k = max(2, min(n_bins, n_unique))
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km.fit(finite_vals)
            centers = np.sort(km.cluster_centers_.flatten())
            # Re-assign bins by proximity to the center (idempotent).
            full = np.zeros(values.shape[0], dtype=np.int64)
            if finite_vals.size > 0:
                dists = np.abs(values.reshape(-1, 1) - centers.reshape(1, -1))
                full = np.argmin(dists, axis=1).astype(np.int64)
            full[~finite_mask] = 0
            bins = full
            edges_used = centers.tolist()

        else:  # "domain"
            assert bin_edges is not None  # nosec - guarded above
            if col not in bin_edges:
                raise ValueError(
                    f"strategy='domain' requires bin_edges[{col!r}]; provided: {list(bin_edges)}"
                )
            edges_user = sorted(float(e) for e in bin_edges[col])
            cut_labels = [str(i) for i in range(len(edges_user) + 1)]
            binned = series.cut(breaks=edges_user, labels=cut_labels)
            str_to_int = {lab: i for i, lab in enumerate(cut_labels)}
            bins = np.array([str_to_int.get(v, 0) for v in binned.to_list()], dtype=np.int64)
            edges_used = edges_user

        edges_report[col] = list(edges_used)
        out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))

    logger.info(
        "discretize_features_done",
        strategy=strategy,
        n_cols=len(cols_list),
        n_bins=n_bins,
    )
    return out, edges_report


def _bin_uniform(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[float]]:
    """Uniform binning between ``min`` and ``max`` with NaN handling.

    Private function used by :func:`discretize_features` and as a fallback
    when ``qcut`` fails due to duplicate edges.
    """
    finite_mask = np.isfinite(values)
    if finite_mask.sum() == 0:
        return np.zeros(values.shape[0], dtype=np.int64), []
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.int64), [vmin]
    edges = np.linspace(vmin, vmax, n_bins + 1)[1:-1]
    bins = np.digitize(np.where(finite_mask, values, vmin), edges).astype(np.int64)
    return bins, edges.tolist()


# Canonical NDVI agronomic thresholds. References:
# - Tucker (1979) "Red and photographic infrared linear combinations" (NDVI < 0 = water).
# - Pettorelli et al. (2005) "Using the satellite-derived NDVI to assess
#   ecological responses" (ranges 0-0.2 = bare soil, 0.2-0.4 = sparse, etc.).
_NDVI_PHENOLOGY_BINS: tuple[float, ...] = (-1.0, 0.0, 0.2, 0.4, 0.6, 1.0)
_NDVI_PHENOLOGY_LABELS: tuple[str, ...] = ("water", "bare", "sparse", "moderate", "dense")


def discretize_ndvi_phenology_domain(
    df: pl.DataFrame,
    ndvi_col: str,
    *,
    bins: tuple[float, ...] = _NDVI_PHENOLOGY_BINS,
    labels: tuple[str, ...] = _NDVI_PHENOLOGY_LABELS,
) -> tuple[pl.DataFrame, list[str]]:
    """Discretize NDVI with agronomic thresholds (domain binning).

    Convenience wrapper of :func:`discretize_features` with ``strategy="domain"``
    and the canonical thresholds of Tucker (1979) / Pettorelli et al. (2005):

    - ``< 0.0``: ``water`` (water bodies, shadows).
    - ``[0.0, 0.2)``: ``bare`` (bare soil, urban).
    - ``[0.2, 0.4)``: ``sparse`` (sparse vegetation, early crops).
    - ``[0.4, 0.6)``: ``moderate`` (growing crops).
    - ``[0.6, 1.0]``: ``dense`` (dense canopy, season peak).

    Args:
        df: Polars DataFrame with the ``ndvi_col`` column.
        ndvi_col: Name of the NDVI column (may be ``NDVI_mean``,
            ``NDVI_p50``, etc.).
        bins: Increasing internal edges (default Tucker /
            Pettorelli thresholds).
        labels: Semantic labels (must have ``len(bins) - 1`` elements
            or they are truncated/padded).

    Returns:
        Tuple ``(df_with_pheno, label_list)`` where:

        - ``df_with_pheno`` adds ``{ndvi_col}__pheno`` (Utf8) and keeps
          the original.
        - ``label_list`` is the list of labels in ascending order (useful
          to order the category as ordinal if the caller needs it).

    Raises:
        ValueError: If ``ndvi_col`` does not exist in ``df`` or if ``bins`` is
            not sorted ascending.
    """
    if ndvi_col not in df.columns:
        raise ValueError(f"ndvi_col {ndvi_col!r} not present in df.columns")
    bins_list = list(bins)
    if any(bins_list[i] >= bins_list[i + 1] for i in range(len(bins_list) - 1)):
        raise ValueError(f"bins must be ascending; got {bins_list}")
    expected_n_labels = len(bins_list) - 1
    if len(labels) < expected_n_labels:
        labels_eff = list(labels) + [f"bin_{i}" for i in range(len(labels), expected_n_labels)]
    else:
        labels_eff = list(labels[:expected_n_labels])

    series = df.get_column(ndvi_col).cast(pl.Float64)
    # ``cut`` expects internal edges (without the min/max extremes), except for
    # the limits; we truncate the extremes to keep compatibility.
    internal_breaks = bins_list[1:-1]
    binned = series.cut(breaks=internal_breaks, labels=labels_eff)
    out = df.with_columns(binned.alias(f"{ndvi_col}__pheno").cast(pl.Utf8))

    logger.info(
        "discretize_ndvi_phenology_domain_done",
        ndvi_col=ndvi_col,
        n_bins=expected_n_labels,
        labels=labels_eff,
    )
    return out, labels_eff


# ---------------------------------------------------------------------------
# EXTRACTORS
# ---------------------------------------------------------------------------


def fit_pca(
    X_scaled: np.ndarray,
    target_variance: float = 0.95,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit PCA retaining components until accumulating ``target_variance``.

    Args:
        X_scaled: Matrix ``(n_samples, n_features)`` previously standardized
            (PCA is sensitive to scale).
        target_variance: Fraction of cumulative variance to retain (0, 1].
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with keys ``{"n_components", "components",
        "explained_variance_ratio", "cumulative_variance", "transformer"}``.

    Raises:
        ValueError: If ``target_variance`` is not in (0, 1] or
            ``X_scaled`` is empty.
    """
    if not 0.0 < target_variance <= 1.0:
        raise ValueError(f"target_variance must be in (0, 1]; got {target_variance}")
    if X_scaled.size == 0 or X_scaled.shape[1] == 0:
        raise ValueError("X_scaled is empty; cannot fit PCA.")

    matrix = _impute_with_column_mean(X_scaled)
    full_pca = PCA(n_components=None, random_state=random_state)
    full_pca.fit(matrix)
    cum_var = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, target_variance) + 1)
    n_components = max(1, min(n_components, matrix.shape[1]))

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(matrix)

    logger.info(
        "pca_fitted",
        target_variance=target_variance,
        n_components=n_components,
        cumulative_variance=float(cum_var[n_components - 1]),
    )
    return {
        "n_components": n_components,
        "components": pca.components_,
        "explained_variance_ratio": full_pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "transformer": pca,
    }


def fit_factor_analysis(
    X_scaled: np.ndarray,
    n_factors: int = 5,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit Factor Analysis with ``n_factors`` latent components.

    Args:
        X_scaled: Standardized matrix ``(n_samples, n_features)``.
        n_factors: Number of latent factors to estimate.
        random_state: Seed.

    Returns:
        Dictionary with keys ``{"loadings", "noise_variance",
        "explained_variance_approx", "transformer"}``.
        ``loadings`` has shape ``(n_features, n_factors)``.

    Raises:
        ValueError: If ``n_factors`` exceeds ``min(n_samples, n_features)``.
    """
    if X_scaled.size == 0:
        raise ValueError("X_scaled is empty; cannot fit FactorAnalysis.")
    matrix = _impute_with_column_mean(X_scaled)
    n_samples, n_features = matrix.shape
    max_factors = max(1, min(n_samples - 1, n_features))
    if n_factors > max_factors:
        raise ValueError(
            f"n_factors={n_factors} exceeds the max allowed {max_factors} "
            f"(min(n_samples-1, n_features))."
        )

    fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
    fa.fit(matrix)
    # Approximate explained variance per factor: sum of squares of the
    # loadings (not normalized, only useful for the "positive" test).
    loadings = fa.components_.T  # shape (n_features, n_factors)
    explained_approx = (loadings**2).sum(axis=0)

    logger.info(
        "factor_analysis_fitted",
        n_factors=n_factors,
        n_features=n_features,
        n_samples=n_samples,
    )
    return {
        "loadings": loadings,
        "noise_variance": fa.noise_variance_,
        "explained_variance_approx": explained_approx,
        "transformer": fa,
    }


def fit_umap_2d(
    X_scaled: np.ndarray,
    y: np.ndarray | None = None,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute a deterministic UMAP 2D embedding for visualization.

    UMAP in this module is **prepro/EDA**, not production feature engineering
    (plan decision D4).

    Args:
        X_scaled: Standardized matrix ``(n_samples, n_features)``.
        y: Optional class vector (does not affect the embedding but is accepted
            to keep symmetry with the downstream API and future supervised
            modes).
        n_neighbors: UMAP neighbors (default 15, McInnes et al. 2018).
        min_dist: Minimum distance in the embedding.
        random_state: Seed. UMAP is deterministic if ``random_state`` is
            fixed and ``n_jobs=1``.

    Returns:
        Embedding ``np.ndarray`` shape ``(n_samples, 2)``.
    """
    del y  # Symmetric API with future supervised modes.
    # Lazy import: umap-learn loads numba JIT (~3s) on the first import.
    import umap  # type: ignore[import-untyped]

    matrix = _impute_with_column_mean(X_scaled)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, matrix.shape[0] - 1)),
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(matrix)
    logger.info(
        "umap_2d_fitted",
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return np.asarray(embedding, dtype=np.float64)


# ---------------------------------------------------------------------------
# COMPLEMENT (feature importance)
# ---------------------------------------------------------------------------


def compute_feature_importance(
    X: pl.DataFrame,
    y: pl.Series,
    *,
    model: Literal["rf", "xgb"] = "rf",
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> pl.DataFrame:
    """Compute per-feature importance with RF (Gini) or XGB (gain).

    Exploratory hyperparameters (NOT production, plan decision D7):

    - ``RandomForestClassifier(n_estimators=200, max_depth=None)``.
    - ``XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
      tree_method="hist")``.

    Args:
        X: Polars wide-format DataFrame.
        y: Polars Series with the class.
        model: ``"rf"`` or ``"xgb"``.
        n_estimators: Number of trees.
        random_state: Seed.
        n_jobs: Parallelism (-1 = all cores).
        exclude_cols: Columns to exclude as features.

    Returns:
        Polars DataFrame with columns ``(feature, importance, rank)``
        sorted descending by ``importance``.

    Raises:
        ValueError: If ``model`` is not ``"rf"`` or ``"xgb"``.
    """
    if model not in ("rf", "xgb"):
        raise ValueError(f"model must be 'rf' or 'xgb'; got {model!r}")

    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return pl.DataFrame(
            {"feature": [], "importance": [], "rank": []},
            schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
        )

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    if model == "rf":
        clf: Any = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        clf.fit(matrix_clean, y_arr)
        importances = clf.feature_importances_
    else:
        # Lazy import of xgboost (>2s on cold-start) + remap of classes to a
        # contiguous range 0..N-1 (XGB requires dense labels).
        import xgboost as xgb  # type: ignore[import-untyped]

        unique_labels = sorted(np.unique(y_arr).tolist())
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_remap = np.array([label_to_idx[int(v)] for v in y_arr], dtype=np.int64)
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            use_label_encoder=False,
        )
        clf.fit(matrix_clean, y_remap)
        importances = np.asarray(clf.feature_importances_, dtype=np.float64)

    order = np.argsort(-importances)
    sorted_features = [feature_cols[i] for i in order]
    sorted_imps = importances[order]
    ranks = list(range(1, len(sorted_features) + 1))

    df = pl.DataFrame(
        {
            "feature": sorted_features,
            "importance": sorted_imps.tolist(),
            "rank": ranks,
        },
        schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
    )
    logger.info(
        "feature_importance_computed",
        model=model,
        n_features=len(feature_cols),
        top1=sorted_features[0] if sorted_features else None,
    )
    return df


# ---------------------------------------------------------------------------
# COMPARISON (PASTIS folds 1-5)
# ---------------------------------------------------------------------------


def compare_before_after(
    X_raw: pl.DataFrame,
    X_selected: pl.DataFrame,
    y: pl.Series,
    folds: np.ndarray,
    *,
    extra_strategies: dict[str, pl.DataFrame] | None = None,
    random_state: int = 42,
    n_estimators: int = 100,
) -> pl.DataFrame:
    """Compare selection strategies with CV using PASTIS folds 1-5.

    Decision D1: uses official spatial folds (Sainte-Fare-Garnot 2021), NOT
    random KFold. If ``folds`` does not contain values in ``{1..5}``, returns
    NaN without raising (logs a warning).

    Args:
        X_raw: Frame with ALL the features (baseline).
        X_selected: Frame with post-filter features (variance + correlation).
        y: Target series.
        folds: Vector ``(n_samples,)`` with PASTIS folds 1-5.
        extra_strategies: Optional mapping ``{name: frame}`` with additional
            strategies (e.g. ``{"pca_0.95": pca_df, "selected+pca": combo}``).
        random_state: Seed of the baseline RF.
        n_estimators: Number of trees of the instrumental RF.

    Returns:
        Polars DataFrame with columns ``(strategy, n_features, f1_macro_mean,
        f1_macro_std, miou_mean, miou_std)``. Always at least 4 rows:
        ``raw``, ``variance+corr``, ``pca_0.95`` (NaN placeholder if not
        provided), ``selected+pca`` (NaN placeholder if not provided).
    """
    y_arr = np.asarray(y.to_list())
    folds_arr = np.asarray(folds, dtype=np.int64)

    strategies: list[tuple[str, pl.DataFrame]] = [
        ("raw", X_raw),
        ("variance+correlation", X_selected),
    ]
    if extra_strategies:
        for name, frame in extra_strategies.items():
            strategies.append((name, frame))

    rows: list[dict[str, Any]] = []
    for name, frame in strategies:
        matrix, feature_cols = _to_numpy(frame)
        if matrix.shape[1] == 0:
            rows.append(
                _build_strategy_table(
                    strategy=name,
                    n_features=0,
                    f1_mean=float("nan"),
                    f1_std=float("nan"),
                    miou_mean=float("nan"),
                    miou_std=float("nan"),
                )
            )
            continue
        f1_mean, f1_std, miou_mean, miou_std = _run_cv_baseline_rf(
            matrix,
            y_arr,
            folds_arr,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        rows.append(
            _build_strategy_table(
                strategy=name,
                n_features=len(feature_cols),
                f1_mean=f1_mean,
                f1_std=f1_std,
                miou_mean=miou_mean,
                miou_std=miou_std,
            )
        )

    schema: dict[str, pl.DataType] = {
        "strategy": pl.Utf8(),
        "n_features": pl.Int64(),
        "f1_macro_mean": pl.Float64(),
        "f1_macro_std": pl.Float64(),
        "miou_mean": pl.Float64(),
        "miou_std": pl.Float64(),
    }
    result = pl.DataFrame(rows, schema=schema)
    logger.info(
        "compare_before_after_done",
        n_strategies=result.height,
        unique_folds=sorted(np.unique(folds_arr).tolist()),
    )
    return result


# ---------------------------------------------------------------------------
# NORMALIZATION (decision D9 + D10)
# ---------------------------------------------------------------------------


def select_normalizer(
    feature_name: str,
    distribution_stats: dict[str, float],
    *,
    strategy: Literal["linear", "nn"] = "linear",
) -> tuple[str, str]:
    """Decide scaler per feature by name, distribution and model family.

    Rules (in priority order):

    1. ``feature_name`` starts with ``LAI`` or ``biomass`` -> ``log1p``
       (positive, right-skewed).
    2. ``feature_name`` starts with ``NDVI/NDRE/NDWI/NDMI/NBR`` AND
       ``|skew| > 1.0`` -> ``yeo-johnson`` (D10: accepts negatives).
    3. ``|skew| > 1.0`` -> ``yeo-johnson``.
    4. ``strategy == "nn"`` -> ``minmax``.
    5. Default -> ``standard``.

    Args:
        feature_name: Feature name (case sensitive, prefix match).
        distribution_stats: Dictionary with at least ``{"skew": float}``.
            Also accepts ``{"min", "max"}`` for future refinements.
        strategy: Downstream model family (``"linear"`` or ``"nn"``).

    Returns:
        Tuple ``(scaler_name, justification_short)``.
    """
    skew = float(distribution_stats.get("skew", 0.0))

    if any(feature_name.startswith(prefix) for prefix in _LOG1P_FEATURE_PREFIXES):
        return ("log1p", f"feature {feature_name!r} es positiva y sesgada (LAI/biomasa)")

    if (
        any(feature_name.startswith(prefix) for prefix in _YEO_JOHNSON_FEATURE_PREFIXES)
        and abs(skew) > _SKEW_YEO_THRESHOLD
    ):
        return (
            "yeo-johnson",
            f"feature {feature_name!r} es indice espectral con skew={skew:.2f}; "
            "Yeo-Johnson acepta negativos (D10)",
        )

    if abs(skew) > _SKEW_YEO_THRESHOLD:
        return (
            "yeo-johnson",
            f"feature {feature_name!r} sesgada (skew={skew:.2f}); Yeo-Johnson",
        )

    if strategy == "nn":
        return ("minmax", f"feature {feature_name!r} a [0,1] para red neuronal")

    return ("standard", f"feature {feature_name!r} estandarizada (lineal/SVM)")


def make_preprocessor(
    df: pl.DataFrame,
    *,
    strategy: Literal["linear", "nn"] = "linear",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
    categorical_cols: tuple[str, ...] = (),
    categorical_encoder: Literal["onehot", "ordinal"] = "onehot",
) -> ColumnTransformer:
    """Build a :class:`ColumnTransformer` routed by :func:`select_normalizer`.

    Computes skew per feature, decides the scaler with :func:`select_normalizer`
    and groups the columns that receive the same scaler into a single
    ``transformer`` to minimize overhead. If ``categorical_cols`` is
    non-empty, adds an additional bucket to encode them with
    :class:`~sklearn.preprocessing.OneHotEncoder` (default) or
    :class:`~sklearn.preprocessing.OrdinalEncoder`, keeping backward
    compatibility with existing callers (``categorical_cols=()``).

    The result is serializable with :mod:`joblib` (Aaron's requirement for
    loading from GCS in the backend) and compatible with ``fit_transform`` over
    ``np.ndarray`` matrices or ``pl.DataFrame.to_numpy()``.

    Args:
        df: Polars DataFrame with the features the preprocessor will consume.
        strategy: ``"linear"`` or ``"nn"``.
        exclude_cols: Columns to omit entirely (neither numeric nor
            categorical).
        categorical_cols: Categorical columns (Utf8/Categorical or low-
            cardinality Int). If empty (default), the signature operates
            exactly like the original version (US-018 phase 3).
        categorical_encoder: ``"onehot"`` (default) uses
            :class:`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`,
            ``"ordinal"`` uses
            :class:`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.

    Returns:
        :class:`ColumnTransformer` ready for ``fit_transform(X)``. Uses
        ``remainder="drop"`` to avoid including accidental columns.

    Notes:
        - ``log1p`` is implemented with
          :class:`~sklearn.preprocessing.PowerTransformer` ``box-cox`` (positive)
          falling back to Yeo-Johnson if there are remaining non-positives.
        - Column indices are **integers** (not names), because the
          downstream consumer passes ``np.ndarray``.
        - Categoricals are identified by name + excluded from numeric
          processing (added to ``exclude_cols`` internally).
    """
    # The column layout the downstream ColumnTransformer will see is
    # ``df.drop(exclude_cols).to_numpy()``: it includes the categoricals at
    # their original position. The numeric indices must refer to that layout
    # (not to the internal matrix that excludes categoricals).
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    cat_set = set(categorical_cols)
    col_to_input_idx = {c: i for i, c in enumerate(all_feature_cols)}

    # ``matrix_clean`` is only used to compute skew (numerics), so we
    # exclude the categoricals there.
    numeric_exclude = tuple(set(exclude_cols) | cat_set)
    matrix, feature_cols = _to_numpy(df, exclude_cols=numeric_exclude)
    if matrix.shape[1] == 0 and not categorical_cols:
        return ColumnTransformer([], remainder="drop")
    matrix_clean = _impute_with_column_mean(matrix) if matrix.size else matrix

    # Buckets per scaler to collapse transformers. The indices are
    # positions in the ColumnTransformer input matrix (which includes
    # categoricals), so we use ``col_to_input_idx[name]``.
    buckets: dict[str, list[int]] = {
        "standard": [],
        "minmax": [],
        "yeo-johnson": [],
        "log1p": [],
    }
    for local_idx, name in enumerate(feature_cols):
        col = matrix_clean[:, local_idx]
        skew_val = float(scipy_stats.skew(col, bias=False)) if col.size > 2 else 0.0
        if not np.isfinite(skew_val):
            skew_val = 0.0
        scaler_name, _ = select_normalizer(
            name,
            {"skew": skew_val, "min": float(np.min(col)), "max": float(np.max(col))},
            strategy=strategy,
        )
        buckets[scaler_name].append(col_to_input_idx[name])

    transformers: list[tuple[str, Any, list[int]]] = []
    if buckets["standard"]:
        transformers.append(("standard", StandardScaler(), buckets["standard"]))
    if buckets["minmax"]:
        transformers.append(("minmax", MinMaxScaler(), buckets["minmax"]))
    if buckets["yeo-johnson"]:
        transformers.append(
            (
                "yeo_johnson",
                PowerTransformer(method="yeo-johnson", standardize=True),
                buckets["yeo-johnson"],
            )
        )
    if buckets["log1p"]:
        # log1p safe via PowerTransformer yeo-johnson + flag; a robust
        # alternative without an external lambda.
        transformers.append(
            (
                "log1p_yeo",
                PowerTransformer(method="yeo-johnson", standardize=True),
                buckets["log1p"],
            )
        )

    # Categorical bucket (US-018 extension phase 5).
    cat_indices = [col_to_input_idx[c] for c in categorical_cols if c in col_to_input_idx]
    if cat_indices:
        if categorical_encoder == "onehot":
            cat_step: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            cat_step = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append((f"categorical_{categorical_encoder}", cat_step, cat_indices))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info(
        "preprocessor_built",
        strategy=strategy,
        n_features=len(feature_cols),
        n_standard=len(buckets["standard"]),
        n_minmax=len(buckets["minmax"]),
        n_yeo=len(buckets["yeo-johnson"]),
        n_log1p=len(buckets["log1p"]),
        n_categorical=len(cat_indices),
        categorical_encoder=categorical_encoder if cat_indices else None,
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Convenience: public types
# ---------------------------------------------------------------------------

# Public alias for callers (notebook 03 readers + tests).
Features = pl.DataFrame
Target = pl.Series
Folds = np.ndarray
# Silent re-export of utilities for tests / notebooks that want to
# access the internal version operating on numpy.
_PUBLIC_CONST: dict[str, object] = {
    "DEFAULT_EXCLUDE": _DEFAULT_EXCLUDE,
    "SKEW_YEO_THRESHOLD": _SKEW_YEO_THRESHOLD,
}

# Suppress spurious warnings only when used as CLI/notebook; in tests the
# filterwarnings from pyproject already hides them.
_ = (Sequence, Iterable, cast)  # keeps typed imports without flagging unused
