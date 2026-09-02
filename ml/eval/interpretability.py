"""Interpretability for the crop classification baseline (US-020, EPIC 4).

Reusable module that explains the *production* models of the tabular baseline
(Random Forest and XGBoost from US-019) using two families of techniques:

- **Native importance** (criterion AC-1): Gini/MDI for Random Forest
  (``feature_importances_``) and *gain* for XGBoost
  (``Booster.get_score(importance_type="gain")``). Extracted from the already
  fitted model — nothing is re-trained.
- **SHAP** (criteria AC-2, AC-3, AC-6): exact Shapley values with
  ``shap.TreeExplainer`` over a stratified subsample of the dataset.
  The analysis is **multiclass** (18-20 PASTIS-R classes): ``compute_shap_values``
  normalizes the three output shapes that ``TreeExplainer`` produces depending
  on version (per-class list, 3D array, ``Explanation`` object) into a single
  tensor ``(n_samples, n_features, n_classes)``.

It also quantifies *AlphaEarth dominance* (criterion AC-4): it classifies each
feature into its source family (``is_alphaearth_dim`` + ``alphaearth_dominance_table``)
to answer how many of the top-N SHAP features are AlphaEarth embedding
dimensions — an input for the Paper Track.

Decision D1 (plan US-020 2.1): this module is independent of
``ml/eval/metrics.py`` (metrics) and ``ml/features/selection.py`` (exploratory
feature engineering). Interpretability of production models is its own domain,
also consumed by the EPIC 5/6 architectures.

Decision D6: SHAP runs over a stratified subsample (``sample_size=3000`` by
default), not over the ~85k rows of the full dataset — TreeSHAP is exact but
O(samples x trees x depth) and the subsample gives a stable summary.

Polars is the I/O format and the format of the output tables; the conversion to
numpy/pandas happens exclusively at the SHAP boundary, in private helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "FeatureFamily",
    "ModelKind",
    "ShapResult",
    "alphaearth_dominance_table",
    "compute_shap_values",
    "feature_importance_table",
    "is_alphaearth_dim",
    "shap_dependence_plots",
    "shap_summary_plot",
    "shap_waterfall_plot",
]

ModelKind = Literal["rf", "xgb"]
FeatureFamily = Literal["alphaearth", "spectral_index", "s1", "srtm", "era5", "geom", "other"]

# Figure resolution for the Avance 3 visual deliverables (criterion AC-7).
_PLOT_DPI: int = 200

# Regex for the AlphaEarth embedding dimensions: `dim_00`..`dim_63`
# (real prefix confirmed 2026-05-21 in the enriched parcel-level parquet).
_ALPHAEARTH_DIM_RE = re.compile(r"^dim_\d{2}$")

# Statistical suffixes of the spectral indices (NDVI_mean, EVI_p95, ...) and
# FFT harmonics (NDVI_fft_amp_0, ...). Used to classify the
# `spectral_index` family by naming convention in `_classify_family`.
_SPECTRAL_PREFIXES: tuple[str, ...] = (
    "NDVI",
    "NDWI",
    "EVI",
    "NDMI",
    "NBR",
    "MSAVI2",
    "NDRE",
    "MCARI",
    "CCCI",
    "GCVI",
    "PSRI",
    "NDCI",
    "FAPAR",
    "LAI",
    "RENDVI",
    "SAVI",
    "TSAVI",
)


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapResult:
    """Result of a multiclass SHAP analysis.

    Attributes:
        values: Normalized SHAP array, shape
            ``(n_samples, n_features, n_classes)``. For binary classification
            with 2D output it is also expanded to 3 axes (``n_classes`` = 2 or 1).
        global_importance: ``pl.DataFrame`` ``(feature, mean_abs_shap, rank)``;
            the global ranking is the mean of ``|SHAP|`` over classes and samples
            (decision D4).
        feature_cols: Names of the features in the order of axis 1 of
            ``values``.
        base_values: Expected values of the explainer, shape ``(n_classes,)``.
        model_kind: ``"rf"`` or ``"xgb"``.
    """

    values: np.ndarray
    global_importance: pl.DataFrame
    feature_cols: tuple[str, ...]
    base_values: np.ndarray
    model_kind: ModelKind


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _to_numpy_sample(
    X: pl.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the feature matrix as ``np.ndarray`` and optionally subsample.

    Converts the ``pl.DataFrame`` to ``float64`` (numpy boundary required for
    SHAP) selecting the ``feature_cols`` columns in order. If ``sample_size`` is
    smaller than the number of rows it takes a reproducible random sample;
    otherwise it returns all rows.

    Args:
        X: Polars DataFrame with at least the ``feature_cols`` columns.
        feature_cols: Columns to select, in order.
        sample_size: Subsample size; if ``None`` or ``>= X.height`` all rows are
            used.
        random_state: Sampling seed.

    Returns:
        Tuple ``(matrix, row_index)`` where ``matrix`` is
        ``(n_sample, n_features)`` float64 and ``row_index`` are the original
        indices of the selected rows.

    Raises:
        ValueError: if any ``feature_cols`` column is missing from ``X``.
    """
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"`X` does not contain the required feature columns: {missing}.")

    n_rows = X.height
    if sample_size is None or sample_size >= n_rows:
        row_index = np.arange(n_rows, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        row_index = np.sort(rng.choice(n_rows, size=sample_size, replace=False)).astype(np.int64)

    matrix = X.select(feature_cols).to_numpy().astype(np.float64)
    # Impute NaN/inf with the column mean: TreeExplainer does not accept NaN for
    # some models and the +/-inf of spectral ratios break the algorithm.
    matrix = _impute_columns(matrix)
    return matrix[row_index], row_index


def _impute_columns(matrix: np.ndarray) -> np.ndarray:
    """Replace NaN and infinities with the finite mean of each column.

    Args:
        matrix: Matrix ``(n_samples, n_features)`` that may contain NaN or
            +/-inf.

    Returns:
        A copy of ``matrix`` without non-finite values. Columns with no finite
        value at all are filled with ``0.0``.
    """
    clean = np.array(matrix, dtype=np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    col_means = np.nanmean(np.where(np.isnan(clean), np.nan, clean), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = np.isnan(clean)
    if nan_mask.any():
        clean[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return clean


def _normalize_shap_multiclass(
    raw: Any,
    *,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Normalize the output of ``TreeExplainer.shap_values`` into a 3D tensor.

    ``shap.TreeExplainer`` returns different shapes depending on the version and
    the model type (decision D3, risk R2):

    - **Per-class list**: ``list`` of ``n_classes`` arrays
      ``(n_samples, n_features)`` — classic multiclass API.
    - **3D array**: ``(n_samples, n_features, n_classes)`` — new API.
    - **2D array**: ``(n_samples, n_features)`` — binary or regression; expanded
      to ``(n_samples, n_features, 1)``.
    - **``Explanation`` object**: its ``.values`` attribute is accessed.

    Args:
        raw: Raw output of ``shap_values`` or of ``explainer(X)``.
        n_samples: Expected number of samples (axis 0).
        n_features: Expected number of features (axis 1).

    Returns:
        float64 array ``(n_samples, n_features, n_classes)``.

    Raises:
        ValueError: if the output does not match any of the known shapes.
    """
    # Explanation object -> extract .values and recurse.
    if hasattr(raw, "values") and not isinstance(raw, list | tuple | np.ndarray):
        return _normalize_shap_multiclass(
            np.asarray(raw.values, dtype=np.float64),
            n_samples=n_samples,
            n_features=n_features,
        )

    # List/tuple of 2D arrays, one per class.
    if isinstance(raw, list | tuple):
        per_class = [np.asarray(arr, dtype=np.float64) for arr in raw]
        if not per_class:
            raise ValueError("`shap_values` returned an empty list.")
        # stack over the last axis -> (n_samples, n_features, n_classes).
        stacked = np.stack(per_class, axis=-1)
        return _validate_shape(stacked, n_samples, n_features)

    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 2:
        return _validate_shape(array[:, :, np.newaxis], n_samples, n_features)
    if array.ndim == 3:
        # Some versions return (n_classes, n_samples, n_features);
        # reorder to the canonical layout if axis 0 is not n_samples.
        if array.shape[0] != n_samples and array.shape[1] == n_samples:
            array = np.transpose(array, (1, 2, 0))
        return _validate_shape(array, n_samples, n_features)

    raise ValueError(f"Unrecognized SHAP shape: ndim={array.ndim}, shape={array.shape}.")


def _validate_shape(array: np.ndarray, n_samples: int, n_features: int) -> np.ndarray:
    """Validate that a 3D SHAP tensor has the axes ``(n_samples, n_features, *)``.

    Args:
        array: Candidate 3-axis tensor.
        n_samples: Expected number of samples.
        n_features: Expected number of features.

    Returns:
        The ``array`` itself if the first two axes match.

    Raises:
        ValueError: if the axes do not match what was expected.
    """
    if array.shape[:2] != (n_samples, n_features):
        raise ValueError(
            f"SHAP tensor with unexpected axes {array.shape}; "
            f"expected (n_samples={n_samples}, n_features={n_features}, *)."
        )
    return array


def _global_importance_table(values: np.ndarray, feature_cols: tuple[str, ...]) -> pl.DataFrame:
    """Build the global SHAP importance table.

    The global importance of each feature is the mean of ``|SHAP|`` over all
    samples and all classes (decision D4) — the standard ranking for the summary
    plots.

    Args:
        values: SHAP tensor ``(n_samples, n_features, n_classes)``.
        feature_cols: Names of the features.

    Returns:
        ``pl.DataFrame`` ``(feature, mean_abs_shap, rank)`` sorted descending by
        ``mean_abs_shap``.
    """
    mean_abs = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(-mean_abs)
    return pl.DataFrame(
        {
            "feature": [feature_cols[i] for i in order],
            "mean_abs_shap": mean_abs[order].astype(np.float64).tolist(),
            "rank": list(range(1, len(order) + 1)),
        },
        schema={
            "feature": pl.Utf8,
            "mean_abs_shap": pl.Float64,
            "rank": pl.Int64,
        },
    )


def _classify_family(feature_name: str) -> FeatureFamily:
    """Classify a feature into its source family by naming convention.

    Args:
        feature_name: Name of the feature column.

    Returns:
        The family: ``alphaearth`` (``dim_NN``), ``spectral_index`` (indices and
        their FFT harmonics), ``s1`` (Sentinel-1 radar, prefix ``VV``/``VH``),
        ``srtm`` (elevation/slope), ``era5`` (climate), ``geom`` (parcel
        geometry) or ``other``.
    """
    if is_alphaearth_dim(feature_name):
        return "alphaearth"

    upper = feature_name.upper()
    if upper.startswith(("VV", "VH", "S1_")):
        return "s1"
    if upper.startswith(("SRTM", "ELEV", "SLOPE", "ASPECT", "DEM")):
        return "srtm"
    if upper.startswith(("ERA5", "TEMP", "PRECIP", "T2M", "TP_")):
        return "era5"
    if upper.startswith(("AREA", "PERIMETER", "GEOM", "N_PIXELS")):
        return "geom"

    base = feature_name.split("_", 1)[0]
    if base in _SPECTRAL_PREFIXES:
        return "spectral_index"
    # Phenology derived from NDVI (sog_doy, peak_doy, ndvi_auc, ...): treated
    # as a spectral index because it derives from the index series.
    if feature_name.lower().startswith(("sog_", "peak_", "senescence_", "ndvi_", "maturity_")):
        return "spectral_index"
    return "other"


# ---------------------------------------------------------------------------
# Native importance (criterion AC-1).
# ---------------------------------------------------------------------------


def feature_importance_table(
    model: ClassifierMixin,
    model_kind: ModelKind,
    feature_cols: tuple[str, ...],
) -> pl.DataFrame:
    """Compute the native importance of an already fitted tree model.

    Random Forest exposes the Gini/MDI importance in ``feature_importances_``;
    XGBoost exposes the gain in
    ``Booster.get_score(importance_type="gain")``. Unlike
    :func:`ml.features.selection.compute_feature_importance` (which re-trains an
    exploratory model), here the attribute is extracted from the US-019
    *production* model — decision D2: nothing is re-trained.

    Args:
        model: Already fitted ``RandomForestClassifier`` or ``XGBClassifier``
            estimator.
        model_kind: ``"rf"`` for Gini or ``"xgb"`` for gain.
        feature_cols: Names of the features in the order in which the model was
            fitted.

    Returns:
        ``pl.DataFrame`` ``(feature, importance, rank)`` sorted descending by
        ``importance``. For XGBoost the features the booster never used receive
        ``importance = 0.0``.

    Raises:
        ValueError: if ``model_kind`` is neither ``"rf"`` nor ``"xgb"``, or if
            the model's number of features does not match ``len(feature_cols)``.
    """
    if model_kind not in ("rf", "xgb"):
        raise ValueError(f"`model_kind` must be 'rf' or 'xgb'; received {model_kind!r}.")

    n_features = len(feature_cols)
    if model_kind == "rf":
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        if importances.shape[0] != n_features:
            raise ValueError(
                f"The RF model has {importances.shape[0]} features but "
                f"`feature_cols` has {n_features}."
            )
    else:
        importances = _xgb_gain_importances(model, feature_cols)

    order = np.argsort(-importances)
    df = pl.DataFrame(
        {
            "feature": [feature_cols[i] for i in order],
            "importance": importances[order].astype(np.float64).tolist(),
            "rank": list(range(1, n_features + 1)),
        },
        schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
    )
    logger.info(
        "feature_importance_table_computed",
        model_kind=model_kind,
        n_features=n_features,
        top_feature=df["feature"][0] if df.height else None,
    )
    return df


def _xgb_gain_importances(model: ClassifierMixin, feature_cols: tuple[str, ...]) -> np.ndarray:
    """Extract the gain importance of an ``XGBClassifier`` aligned to the given order.

    The XGBoost booster indexes the features as ``f0``, ``f1``, ... when trained
    with an ``np.ndarray``. ``get_score`` only returns the features the model
    actually used; the rest are filled with ``0.0``.

    Args:
        model: Already fitted ``XGBClassifier``.
        feature_cols: Names of the features in training order.

    Returns:
        Array ``(n_features,)`` of gain importances, aligned to
        ``feature_cols``.
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    n_features = len(feature_cols)
    importances = np.zeros(n_features, dtype=np.float64)
    booster_names = list(getattr(booster, "feature_names", []) or [])
    for key, gain in score.items():
        if key in booster_names:
            idx = booster_names.index(key)
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
        elif key in feature_cols:
            idx = feature_cols.index(key)
        else:
            continue
        if 0 <= idx < n_features:
            importances[idx] = float(gain)
    return importances


# ---------------------------------------------------------------------------
# SHAP (criteria AC-2, AC-3, AC-6).
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: ClassifierMixin,
    X: pl.DataFrame,
    model_kind: ModelKind,
    *,
    feature_cols: tuple[str, ...],
    sample_size: int = 3000,
    random_state: int = 42,
) -> ShapResult:
    """Compute the SHAP values of a tree model with ``TreeExplainer``.

    Instantiates ``shap.TreeExplainer`` (exact TreeSHAP algorithm, CPU) over a
    stratified subsample of ``X`` (decision D6) and normalizes the multiclass
    output into a tensor ``(n_samples, n_features, n_classes)`` via
    :func:`_normalize_shap_multiclass` (decision D3).

    Args:
        model: Already fitted ``RandomForestClassifier`` or ``XGBClassifier``
            estimator.
        X: Polars DataFrame with the ``feature_cols`` columns.
        model_kind: ``"rf"`` or ``"xgb"``.
        feature_cols: Names of the features in training order.
        sample_size: SHAP subsample size; if ``X`` has fewer rows all are used.
        random_state: Sampling seed (reproducibility).

    Returns:
        :class:`ShapResult` with the SHAP tensor, the global importance table
        and the base values of the explainer.

    Raises:
        ValueError: if ``model_kind`` is invalid or columns are missing in ``X``.
    """
    import shap

    if model_kind not in ("rf", "xgb"):
        raise ValueError(f"`model_kind` must be 'rf' or 'xgb'; received {model_kind!r}.")

    matrix, row_index = _to_numpy_sample(
        X, feature_cols, sample_size=sample_size, random_state=random_state
    )
    n_samples, n_features = matrix.shape

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(matrix, check_additivity=False)
    values = _normalize_shap_multiclass(raw, n_samples=n_samples, n_features=n_features)

    expected = np.atleast_1d(
        np.asarray(getattr(explainer, "expected_value", 0.0), dtype=np.float64)
    )

    result = ShapResult(
        values=values,
        global_importance=_global_importance_table(values, feature_cols),
        feature_cols=tuple(feature_cols),
        base_values=expected,
        model_kind=model_kind,
    )
    logger.info(
        "shap_values_computed",
        model_kind=model_kind,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=values.shape[2],
        sample_rows=int(row_index.size),
    )
    return result


def shap_summary_plot(
    shap_result: ShapResult,
    X: pl.DataFrame,
    *,
    top_n: int = 20,
) -> Figure:
    """Generate the summary plot (beeswarm) of the top-N SHAP features.

    Aggregates the SHAP values over the classes (mean of ``|SHAP|``) to produce
    a global beeswarm of the ``top_n`` most important features. Uses matplotlib's
    ``Agg`` backend so the figure is serializable to PNG in CI and in notebooks
    executed with papermill.

    Args:
        shap_result: Result of :func:`compute_shap_values`.
        X: Polars DataFrame with the ``shap_result.feature_cols`` columns; it
            must have at least as many rows as the SHAP subsample.
        top_n: Number of features to show (the most important globally).

    Returns:
        matplotlib figure ``dpi=200`` ready for ``fig.savefig`` or ``display``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    feature_cols = shap_result.feature_cols
    matrix, _ = _to_numpy_sample(X, feature_cols, sample_size=shap_result.values.shape[0])
    # Importance aggregated over classes -> 2D array (n_samples, n_features).
    aggregated = np.abs(shap_result.values).mean(axis=2)

    fig = plt.figure(dpi=_PLOT_DPI)
    shap.summary_plot(
        aggregated,
        features=matrix,
        feature_names=list(feature_cols),
        max_display=top_n,
        plot_type="bar",
        show=False,
    )
    fig = plt.gcf()
    fig.set_dpi(_PLOT_DPI)
    ax = fig.gca()
    ax.set_title(f"SHAP — importancia global top-{top_n} ({shap_result.model_kind.upper()})")
    fig.tight_layout()
    return fig


def shap_dependence_plots(
    shap_result: ShapResult,
    X: pl.DataFrame,
    *,
    top_features: int = 5,
) -> list[tuple[str, Figure]]:
    """Generate one dependence plot for each of the top-N SHAP features.

    The features are sorted by global SHAP importance (decision D4); for each one
    the SHAP value of the most explained class is plotted against the feature
    value.

    Args:
        shap_result: Result of :func:`compute_shap_values`.
        X: Polars DataFrame with the ``shap_result.feature_cols`` columns.
        top_features: Number of features to plot.

    Returns:
        List of tuples ``(feature_name, figure)``, one per feature, sorted by
        descending global SHAP importance.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    feature_cols = shap_result.feature_cols
    matrix, _ = _to_numpy_sample(X, feature_cols, sample_size=shap_result.values.shape[0])
    top = shap_result.global_importance.sort("rank").head(top_features)["feature"].to_list()
    # Reference class: the one concentrating the most global SHAP signal.
    class_idx = int(np.abs(shap_result.values).mean(axis=(0, 1)).argmax())
    class_values = shap_result.values[:, :, class_idx]

    plots: list[tuple[str, Figure]] = []
    for feature_name in top:
        col_idx = feature_cols.index(feature_name)
        fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=_PLOT_DPI)
        feature_values = matrix[:, col_idx]
        scatter = ax.scatter(
            feature_values,
            class_values[:, col_idx],
            c=feature_values,
            cmap="viridis",
            s=12,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax, label=feature_name)
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feature_name)
        ax.set_ylabel(f"valor SHAP (clase {class_idx})")
        ax.set_title(f"Dependence — {feature_name}")
        fig.tight_layout()
        plots.append((feature_name, fig))

    logger.info(
        "shap_dependence_plots_generated",
        model_kind=shap_result.model_kind,
        n_plots=len(plots),
        class_idx=class_idx,
    )
    return plots


def shap_waterfall_plot(
    shap_result: ShapResult,
    *,
    row: int = 0,
    class_idx: int | None = None,
) -> Figure:
    """Generate the waterfall plot of a sample prediction.

    The waterfall decomposes an individual prediction by showing the
    contribution of each feature to the shift from the base value to the model
    output.

    Args:
        shap_result: Result of :func:`compute_shap_values`.
        row: Index of the row (sample) to explain.
        class_idx: Index of the class to explain; if ``None`` the class with the
            largest sum of ``|SHAP|`` for that row is used (proxy of the
            predicted class).

    Returns:
        matplotlib figure ``dpi=200`` with the waterfall plot.

    Raises:
        IndexError: if ``row`` is out of the sample range.
        ValueError: if ``class_idx`` is out of the class range.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    n_samples, _n_features, n_classes = shap_result.values.shape
    if not 0 <= row < n_samples:
        raise IndexError(f"`row`={row} out of range; the SHAP subsample has {n_samples} samples.")

    if class_idx is None:
        resolved_class = int(np.abs(shap_result.values[row]).sum(axis=0).argmax())
    else:
        if not 0 <= class_idx < n_classes:
            raise ValueError(
                f"`class_idx`={class_idx} out of range; there are {n_classes} classes."
            )
        resolved_class = class_idx

    row_values = shap_result.values[row, :, resolved_class]
    base = shap_result.base_values
    base_value = float(base[resolved_class] if base.size > resolved_class else base.flat[0])

    explanation = shap.Explanation(
        values=row_values,
        base_values=base_value,
        feature_names=list(shap_result.feature_cols),
    )
    fig = plt.figure(dpi=_PLOT_DPI)
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_dpi(_PLOT_DPI)
    fig.suptitle(
        f"SHAP waterfall — fila {row}, clase {resolved_class} ({shap_result.model_kind.upper()})"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# AlphaEarth dominance analysis (criterion AC-4).
# ---------------------------------------------------------------------------


def is_alphaearth_dim(feature_name: str) -> bool:
    """Indicate whether a feature is an AlphaEarth embedding dimension.

    The AlphaEarth dimensions are named ``dim_00``..``dim_63`` (64-dimensional
    embedding — real convention confirmed by inspecting the AlphaEarth
    parcel-level parquet). The function applies the regex ``^dim_\\d{2}$``.

    Args:
        feature_name: Name of the feature column.

    Returns:
        ``True`` if the name matches the pattern of an AlphaEarth dimension.
    """
    return bool(_ALPHAEARTH_DIM_RE.match(feature_name))


def alphaearth_dominance_table(
    importance_df: pl.DataFrame,
    *,
    top_n: int = 20,
) -> pl.DataFrame:
    """Classify the top-N features by family and quantify dominance.

    Takes an importance table (native importance or global SHAP), trims it to the
    ``top_n`` most important and adds the source family of each feature
    (``alphaearth``, ``spectral_index``, ``s1``, ``srtm``, ``era5``, ``geom``,
    ``other``). It is the input for the quantified conclusion of criterion AC-4
    ("how many of the top-20 are AlphaEarth dimensions").

    Args:
        importance_df: ``pl.DataFrame`` with a ``feature`` column and a numeric
            importance column (``importance`` or ``mean_abs_shap``). If it has a
            ``rank`` column its order is respected; otherwise it is sorted by the
            importance column.
        top_n: Number of features to retain.

    Returns:
        ``pl.DataFrame`` ``(rank, feature, family, importance)`` with the first
        ``top_n`` features.

    Raises:
        ValueError: if ``importance_df`` does not contain the ``feature`` column
            or has no recognizable importance column.
    """
    if "feature" not in importance_df.columns:
        raise ValueError("`importance_df` must contain the `feature` column.")

    importance_col: str | None = None
    for candidate in ("importance", "mean_abs_shap"):
        if candidate in importance_df.columns:
            importance_col = candidate
            break
    if importance_col is None:
        raise ValueError(
            "`importance_df` must contain an importance column (`importance` or `mean_abs_shap`)."
        )

    if "rank" in importance_df.columns:
        ordered = importance_df.sort("rank")
    else:
        ordered = importance_df.sort(importance_col, descending=True)

    top = ordered.head(top_n)
    families = [_classify_family(name) for name in top["feature"].to_list()]

    table = pl.DataFrame(
        {
            "rank": list(range(1, top.height + 1)),
            "feature": top["feature"].to_list(),
            "family": families,
            "importance": top[importance_col].cast(pl.Float64).to_list(),
        },
        schema={
            "rank": pl.Int64,
            "feature": pl.Utf8,
            "family": pl.Utf8,
            "importance": pl.Float64,
        },
    )
    n_alphaearth = sum(1 for f in families if f == "alphaearth")
    logger.info(
        "alphaearth_dominance_computed",
        top_n=top.height,
        n_alphaearth=n_alphaearth,
        dominance_ratio=round(n_alphaearth / max(top.height, 1), 3),
    )
    return table
