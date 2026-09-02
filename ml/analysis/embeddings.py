"""Analysis of 64-dim AlphaEarth embeddings for US-011 (EDA).

Provides 8 reusable functions over Polars DataFrames with the dimensions
`dim_00..dim_63`:

- `correlation_matrix`: correlation matrix in long format.
- `qq_test_dims`: per-dimension statistics + Shapiro test against N(0,1).
- `tsne_2d`: 2D reduction via t-SNE (cubic subsampling).
- `umap_2d`: 2D reduction via UMAP.
- `rf_feature_importance`: Random Forest importance against labels.
- `temporal_stability`: inter-annual cosine similarity per parcel.
- `compare_alphaearth_vs_ndvi`: comparative figure AE pseudo-RGB vs NDVI vs RGB S2.
- `cross_region_consistency`: comparative table of importance Italy vs France.

Polars note: the pandas adapter is used only as a technical boundary for
sklearn / scipy / statsmodels / seaborn functions when those libraries do not
accept Polars directly. All persistence and aggregation goes through Polars.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import rankdata, shapiro, skew

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from polars._typing import ColumnNameOrSelector, PolarsDataType

_log = structlog.get_logger(__name__)

DIM_COLS: list[str] = [f"dim_{i:02d}" for i in range(64)]
"""Canonical names of the 64 AlphaEarth dimensions (`dim_00`..`dim_63`)."""


def _select_dim_cols(df: pl.DataFrame, cols: list[str] | None) -> list[str]:
    """Select valid dimension columns within the DataFrame."""
    if cols is None:
        return [c for c in DIM_COLS if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")
    return cols


def correlation_matrix(
    df: pl.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
    cols: list[str] | None = None,
) -> pl.DataFrame:
    """Long-format correlation matrix between embedding dimensions.

    Args:
        df: DataFrame with columns `dim_00..dim_63`.
        method: `pearson` (linear) or `spearman` (rank).
        cols: Subset of columns, default all `DIM_COLS` present.

    Returns:
        DataFrame with columns `dim_i, dim_j, pearson|spearman, abs_corr`,
        sorted by `abs_corr` desc. Includes the diagonal and the upper half.
    """
    selected = _select_dim_cols(df, cols)
    if df.is_empty() or not selected:
        return pl.DataFrame(
            schema={
                "dim_i": pl.Utf8,
                "dim_j": pl.Utf8,
                method: pl.Float64,
                "abs_corr": pl.Float64,
            }
        )
    arr = df.select(selected).to_numpy()
    if method == "pearson":
        # numpy is Polars-friendly and much faster than pandas.corr
        valid = ~np.isnan(arr).any(axis=1)
        arr_v = arr[valid]
        if arr_v.shape[0] < 2:
            corr = np.full((len(selected), len(selected)), np.nan)
        else:
            corr = np.corrcoef(arr_v, rowvar=False)
    else:
        # Spearman: ranks per column + Pearson of the ranks
        valid = ~np.isnan(arr).any(axis=1)
        arr_v = arr[valid]
        if arr_v.shape[0] < 2:
            corr = np.full((len(selected), len(selected)), np.nan)
        else:
            ranks = np.apply_along_axis(rankdata, 0, arr_v)
            corr = np.corrcoef(ranks, rowvar=False)

    rows: list[dict[str, Any]] = []
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            value = float(corr[i, j]) if np.isfinite(corr[i, j]) else float("nan")
            rows.append(
                {
                    "dim_i": selected[i],
                    "dim_j": selected[j],
                    method: value,
                    "abs_corr": abs(value) if np.isfinite(value) else float("nan"),
                }
            )
    out = pl.DataFrame(rows)
    return out.sort("abs_corr", descending=True, nulls_last=True)


def qq_test_dims(
    df: pl.DataFrame,
    against: Literal["normal"] = "normal",
    cols: list[str] | None = None,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Descriptive statistics + Shapiro-Wilk per dimension.

    Args:
        df: DataFrame with `dim_00..dim_63`.
        against: Reference distribution (only `normal` supported).
        cols: Subset of dimensions, default all.
        alpha: Significance level for the `is_unit_normal_at_005` flag.

    Returns:
        DataFrame with columns `dim, mean, std, skewness, kurtosis,
        shapiro_stat, shapiro_pvalue, is_unit_normal_at_005`.
    """
    if against != "normal":  # pragma: no cover - defensivo
        raise ValueError(f"Unsupported distribution: {against}")

    selected = _select_dim_cols(df, cols)
    schema = {
        "dim": pl.Utf8,
        "mean": pl.Float64,
        "std": pl.Float64,
        "skewness": pl.Float64,
        "kurtosis": pl.Float64,
        "shapiro_stat": pl.Float64,
        "shapiro_pvalue": pl.Float64,
        "is_unit_normal_at_005": pl.Boolean,
    }
    if df.is_empty() or not selected:
        return pl.DataFrame(schema=schema)

    arr = df.select(selected).to_numpy()
    rows: list[dict[str, Any]] = []
    for i, col in enumerate(selected):
        vals = arr[:, i]
        vals = vals[~np.isnan(vals)]
        if vals.size < 3:
            rows.append(
                {
                    "dim": col,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "skewness": float("nan"),
                    "kurtosis": float("nan"),
                    "shapiro_stat": float("nan"),
                    "shapiro_pvalue": float("nan"),
                    "is_unit_normal_at_005": False,
                }
            )
            continue
        # Shapiro-Wilk saturates at 5000 samples (scipy doc): we subsample if it exceeds
        if vals.size > 5000:
            rng = np.random.default_rng(42)
            sample = rng.choice(vals, size=5000, replace=False)
        else:
            sample = vals
        stat, pval = shapiro(sample)
        rows.append(
            {
                "dim": col,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "skewness": float(skew(vals, bias=False)),
                "kurtosis": float(sp_kurtosis(vals, bias=False)),
                "shapiro_stat": float(stat),
                "shapiro_pvalue": float(pval),
                "is_unit_normal_at_005": bool(pval > alpha),
            }
        )
    return pl.DataFrame(rows, schema=schema)


def tsne_2d(
    df: pl.DataFrame,
    perplexity: int = 30,
    seed: int = 42,
    subsample: int = 10_000,
    cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """2D projection via t-SNE with subsampling for cubic complexity.

    Args:
        df: DataFrame with `dim_00..dim_63`.
        perplexity: t-SNE hyperparameter (default 30).
        seed: Seed for reproducibility.
        subsample: Maximum number of rows to process.
        cols: Subset of dimensional columns.

    Returns:
        Tuple `(X_2d shape (m, 2), idx_subsample)` where `m <= subsample` and
        `idx_subsample` are the selected original indices.
    """
    selected = _select_dim_cols(df, cols)
    if df.is_empty() or not selected:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)

    from sklearn.manifold import TSNE

    arr = df.select(selected).to_numpy()
    # Filter rows with NaN: sklearn rejects missing values. `valid_idx` are
    # indices against the original df, so the caller can map labels with
    # `df[col].to_numpy()[idx]` without misalignment.
    valid_mask = ~np.isnan(arr).any(axis=1)
    valid_idx = np.flatnonzero(valid_mask)
    n_dropped = int(arr.shape[0] - valid_idx.size)
    if n_dropped > 0:
        _log.warning(
            "tsne_2d_dropped_nan_rows",
            n_dropped=n_dropped,
            n_total=int(arr.shape[0]),
            pct=round(100.0 * n_dropped / arr.shape[0], 2),
        )
    if valid_idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if valid_idx.size > subsample:
        chosen = rng.choice(valid_idx.size, size=subsample, replace=False)
        idx = valid_idx[chosen]
    else:
        idx = valid_idx
    idx.sort()
    X = arr[idx]
    safe_perplex = min(perplexity, max(5, X.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplex,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X)
    return X_2d, idx


def umap_2d(
    df: pl.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """2D projection via UMAP. Scales to ~100k points.

    Filters rows with NaN before the fit (UMAP rejects them). The returned
    indices are against the original df so the caller can map labels with
    `df[col].to_numpy()[idx]` without misalignment.

    Args:
        df: DataFrame with `dim_00..dim_63`.
        n_neighbors: Local neighbors (default 15).
        min_dist: Minimum distance in the 2D embedding.
        seed: Seed for reproducibility.
        cols: Subset of dimensions.

    Returns:
        Tuple `(X_2d shape (m, 2), idx)` where `m` is the number of rows
        without NaN. If UMAP is not installed or df is empty returns empty
        arrays.
    """
    selected = _select_dim_cols(df, cols)
    if df.is_empty() or not selected:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)
    try:
        import umap  # type: ignore[import-untyped]
    except ImportError:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)

    arr = df.select(selected).to_numpy()
    valid_mask = ~np.isnan(arr).any(axis=1)
    idx = np.flatnonzero(valid_mask)
    n_dropped = int(arr.shape[0] - idx.size)
    if n_dropped > 0:
        _log.warning(
            "umap_2d_dropped_nan_rows",
            n_dropped=n_dropped,
            n_total=int(arr.shape[0]),
            pct=round(100.0 * n_dropped / arr.shape[0], 2),
        )
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)
    X = arr[idx]
    safe_n = min(n_neighbors, max(2, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_n,
        min_dist=min_dist,
        random_state=seed,
    )
    X_2d = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    return X_2d, idx


def rf_feature_importance(
    X: pl.DataFrame,
    y: pl.Series,
    n_estimators: int = 200,
    max_depth: int = 12,
    seed: int = 42,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Random Forest feature importance against categorical labels.

    Args:
        X: Polars DataFrame with the dimensions (`dim_00..dim_63`).
        y: Polars Series with the class / label per row.
        n_estimators: Number of trees.
        max_depth: Maximum depth.
        seed: Seed.
        n_jobs: sklearn parallelism (default -1 = all cores).

    Returns:
        DataFrame with columns `dim, importance, rank, cumulative_importance,
        oob_score` (the `oob_score` is replicated across all rows as training
        metadata).
    """
    from sklearn.ensemble import RandomForestClassifier

    selected = _select_dim_cols(X, None)
    schema = {
        "dim": pl.Utf8,
        "importance": pl.Float64,
        "rank": pl.Int64,
        "cumulative_importance": pl.Float64,
        "oob_score": pl.Float64,
    }
    if X.is_empty() or not selected or y.is_empty():
        return pl.DataFrame(schema=schema)

    arr = X.select(selected).to_numpy()
    labels = y.to_numpy()
    # Filter rows with NaN in X or null labels: RandomForestClassifier rejects
    # missing values and the importance signal gets contaminated with class 'unknown'.
    valid_mask = ~np.isnan(arr).any(axis=1)
    # `labels` may be dtype object with None: treat None as invalid.
    if labels.dtype == object:
        label_mask = np.array([lbl is not None for lbl in labels])
    else:
        label_mask = ~(labels != labels)  # NaN-aware without assuming float
    valid_mask &= label_mask
    n_dropped = int(arr.shape[0] - valid_mask.sum())
    if n_dropped > 0:
        _log.warning(
            "rf_feature_importance_dropped_nan_rows",
            n_dropped=n_dropped,
            n_total=int(arr.shape[0]),
            pct=round(100.0 * n_dropped / arr.shape[0], 2),
        )
    if not valid_mask.any():
        return pl.DataFrame(schema=schema)
    arr = arr[valid_mask]
    labels = labels[valid_mask]
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight="balanced",
        n_jobs=n_jobs,
        oob_score=True,
        bootstrap=True,
    )
    rf.fit(arr, labels)
    importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    order = np.argsort(-importances)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    cumulative = np.cumsum(importances[order])
    cumulative_per_col = np.empty_like(importances)
    cumulative_per_col[order] = cumulative
    oob = float(getattr(rf, "oob_score_", float("nan")))
    return pl.DataFrame(
        {
            "dim": selected,
            "importance": importances,
            "rank": ranks.astype(np.int64),
            "cumulative_importance": cumulative_per_col,
            "oob_score": np.full(len(selected), oob, dtype=np.float64),
        },
        schema=schema,
    ).sort("importance", descending=True)


def temporal_stability(
    df_long: pl.DataFrame,
    parcel_col: str = "px_id",
    year_col: str = "year",
    class_col: str | None = "class_name",
) -> pl.DataFrame:
    """Inter-annual cosine similarity of the 64-dim embedding per parcel.

    For each parcel present in >=2 consecutive years, computes the cosine
    similarity between the vector of year `t` and that of year `t+1`.
    Aggregates the mean similarity per parcel as `cosine_mean` and enumerates
    the consecutive pairs in `cosine_YYYY_YYYY` columns when available.

    Args:
        df_long: DataFrame with `parcel_col, year_col, dim_00..dim_63`.
        parcel_col: Parcel identifier column.
        year_col: Year column.
        class_col: Class column to keep in the output (optional).

    Returns:
        DataFrame with `parcel_col, [class_col], cosine_mean` and
        `cosine_YYYY_YYYY` columns per consecutive year-year pair.
    """
    selected = _select_dim_cols(df_long, None)
    base_cols = [parcel_col, year_col] + ([class_col] if class_col else [])
    schema_min = {parcel_col: pl.Utf8, "cosine_mean": pl.Float64}
    if df_long.is_empty() or not selected:
        return pl.DataFrame(schema=schema_min)

    df = df_long.select(base_cols + selected).drop_nulls(subset=[parcel_col, year_col])
    if df.is_empty():
        return pl.DataFrame(schema=schema_min)

    years = sorted({int(y) for y in df[year_col].to_list()})
    pairs = list(pairwise(years))
    rows: dict[str, dict[str, Any]] = {}

    for y1, y2 in pairs:
        a = df.filter(pl.col(year_col) == y1).select([parcel_col, *selected])
        b = df.filter(pl.col(year_col) == y2).select([parcel_col, *selected])
        joined = a.join(b, on=parcel_col, suffix="_b")
        if joined.is_empty():
            continue
        arr_a = joined.select(selected).to_numpy()
        arr_b = joined.select([f"{c}_b" for c in selected]).to_numpy()
        # cosine similarity per row
        num = np.sum(arr_a * arr_b, axis=1)
        denom = np.linalg.norm(arr_a, axis=1) * np.linalg.norm(arr_b, axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        cos = num / denom
        pid_arr = joined[parcel_col].to_list()
        col_name = f"cosine_{y1}_{y2}"
        for pid, val in zip(pid_arr, cos, strict=True):
            entry = rows.setdefault(pid, {parcel_col: pid})
            entry[col_name] = float(val)

    # class_col is reincorporated by majority
    if class_col and class_col in df.columns:
        class_lookup = (
            df.group_by(parcel_col)
            .agg(pl.col(class_col).mode().first().alias(class_col))
            .to_dict(as_series=False)
        )
        for pid, cname in zip(class_lookup[parcel_col], class_lookup[class_col], strict=True):
            if pid in rows:
                rows[pid][class_col] = cname

    out_rows = list(rows.values())
    if not out_rows:
        return pl.DataFrame(schema=schema_min)

    out = pl.DataFrame(out_rows)
    cos_cols = [c for c in out.columns if c.startswith("cosine_")]
    if cos_cols:
        out = out.with_columns(
            pl.mean_horizontal([pl.col(c) for c in cos_cols]).alias("cosine_mean")
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("cosine_mean"))
    return out


def compare_alphaearth_vs_ndvi(
    parcel_id: str,
    df_embeddings: pl.DataFrame,
    top_dims: list[str],
    s2_date: str,
    rgb_array: np.ndarray | None = None,
    ndvi_array: np.ndarray | None = None,
    out_path: Path | None = None,
    dpi: int = 200,
) -> matplotlib.figure.Figure:
    """Grid 1x3: RGB stretch 2-98 | NDVI | embedding pseudo-RGB with top 3 dims.

    Args:
        parcel_id: Parcel identifier for the title.
        df_embeddings: DataFrame with the parcel's AlphaEarth dimensions
            (must contain at least `top_dims[:3]`).
        top_dims: Ranking of dimensions (the first 3 are used as RGB).
        s2_date: S2 date used for the RGB / NDVI.
        rgb_array: Preloaded RGB image (H, W, 3). If None a placeholder is shown.
        ndvi_array: Preloaded NDVI image (H, W). If None a placeholder is shown.
        out_path: If provided, saves PNG.
        dpi: Resolution.

    Returns:
        matplotlib Figure with 3 subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

    ax0 = axes[0]
    if rgb_array is not None and rgb_array.size > 0:
        ax0.imshow(np.clip(rgb_array, 0.0, 1.0))
    else:
        ax0.text(0.5, 0.5, "RGB no disponible", ha="center", va="center")
    ax0.set_title(f"RGB S2 {s2_date}")
    ax0.set_axis_off()

    ax1 = axes[1]
    if ndvi_array is not None and ndvi_array.size > 0:
        im = ax1.imshow(ndvi_array, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
        plt.colorbar(im, ax=ax1, fraction=0.046)
    else:
        ax1.text(0.5, 0.5, "NDVI no disponible", ha="center", va="center")
    ax1.set_title("NDVI")
    ax1.set_axis_off()

    ax2 = axes[2]
    dims3 = [d for d in top_dims[:3] if d in df_embeddings.columns]
    if len(dims3) == 3 and not df_embeddings.is_empty():
        vec = df_embeddings.select(dims3).to_numpy()
        # We normalize each channel by min/max to visualize as pseudo-RGB
        norm = np.empty_like(vec, dtype=np.float32)
        for c in range(3):
            v = vec[:, c]
            lo, hi = float(np.min(v)), float(np.max(v))
            norm[:, c] = (v - lo) / max(hi - lo, 1e-6)
        # Reshape to a horizontal strip if there is no spatial info
        h = int(np.ceil(np.sqrt(norm.shape[0])))
        w = int(np.ceil(norm.shape[0] / h))
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(norm.shape[0]):
            canvas[i // w, i % w] = norm[i]
        ax2.imshow(canvas)
    else:
        ax2.text(0.5, 0.5, "Top-3 dims no disponibles", ha="center", va="center")
    ax2.set_title(f"AlphaEarth pseudo-RGB ({', '.join(dims3) if dims3 else 'n/a'})")
    ax2.set_axis_off()

    fig.suptitle(f"Parcela {parcel_id} — AlphaEarth vs NDVI", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig


def cross_region_consistency(
    rf_italia: pl.DataFrame,
    rf_francia: pl.DataFrame,
    top_k: int = 10,
) -> pl.DataFrame:
    """Compare feature importance between Italy and France (AC-15 US-011).

    Args:
        rf_italia: Output of `rf_feature_importance()` over Italy x DW.
        rf_francia: Output of `rf_feature_importance()` over France x PASTIS.
        top_k: How many dims of the ranking to mark as "top" in each region.

    Returns:
        DataFrame with columns `dim, rank_italia, importance_italia,
        rank_francia, importance_francia, consistente_top10, delta_rank,
        importance_sum` sorted by `importance_sum` desc.
    """
    schema = {
        "dim": pl.Utf8,
        "rank_italia": pl.Int64,
        "importance_italia": pl.Float64,
        "rank_francia": pl.Int64,
        "importance_francia": pl.Float64,
        "consistente_top10": pl.Boolean,
        "delta_rank": pl.Int64,
        "importance_sum": pl.Float64,
    }
    if rf_italia.is_empty() or rf_francia.is_empty():
        return pl.DataFrame(schema=schema)

    left = rf_italia.select(
        [
            pl.col("dim"),
            pl.col("rank").alias("rank_italia"),
            pl.col("importance").alias("importance_italia"),
        ]
    )
    right = rf_francia.select(
        [
            pl.col("dim"),
            pl.col("rank").alias("rank_francia"),
            pl.col("importance").alias("importance_francia"),
        ]
    )
    merged = left.join(right, on="dim", how="full", coalesce=True)
    merged = merged.with_columns(
        [
            pl.col("rank_italia").fill_null(9999),
            pl.col("rank_francia").fill_null(9999),
            pl.col("importance_italia").fill_null(0.0),
            pl.col("importance_francia").fill_null(0.0),
        ]
    )
    merged = merged.with_columns(
        [
            ((pl.col("rank_italia") <= top_k) & (pl.col("rank_francia") <= top_k)).alias(
                "consistente_top10"
            ),
            (pl.col("rank_italia") - pl.col("rank_francia")).abs().alias("delta_rank"),
            (pl.col("importance_italia") + pl.col("importance_francia")).alias("importance_sum"),
        ]
    )
    return merged.sort("importance_sum", descending=True).cast(
        cast("Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType]", schema)
    )
