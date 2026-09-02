"""Univariate per-band outlier detection.

Two methods:
- Classic IQR (Tukey): outlier if value < Q1 - k*IQR or > Q3 + k*IQR.
- Multivariate IsolationForest on the wide band x pixel pivot.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest


def detect_outliers_iqr(
    df: pl.DataFrame,
    band: str | None = None,
    k: float = 1.5,
    band_col: str = "band",
    value_col: str = "value",
) -> pl.DataFrame:
    """Detects per-band outliers using the IQR (Tukey) rule.

    Args:
        df: long-format DataFrame.
        band: Specific band to evaluate. If None, evaluates all.
        k: IQR multiplier (1.5 light, 3.0 aggressive).
        band_col: Band column name.
        value_col: Value column name.

    Returns:
        DataFrame with columns `band, n, n_outliers, pct_outliers, q1, q3, lower, upper`.
    """
    if band is not None:
        df = df.filter(pl.col(band_col) == band)

    stats = df.group_by(band_col).agg(
        pl.col(value_col).quantile(0.25).alias("q1"),
        pl.col(value_col).quantile(0.75).alias("q3"),
        pl.len().alias("n"),
    )
    stats = stats.with_columns(
        (pl.col("q3") - pl.col("q1")).alias("iqr"),
    ).with_columns(
        (pl.col("q1") - k * pl.col("iqr")).alias("lower"),
        (pl.col("q3") + k * pl.col("iqr")).alias("upper"),
    )

    joined = df.join(stats, on=band_col)
    out_counts = (
        joined.with_columns(
            ((pl.col(value_col) < pl.col("lower")) | (pl.col(value_col) > pl.col("upper"))).alias(
                "is_outlier"
            )
        )
        .group_by(band_col)
        .agg(
            pl.col("is_outlier").sum().alias("n_outliers"),
        )
    )

    result = stats.join(out_counts, on=band_col).with_columns(
        (pl.col("n_outliers") / pl.col("n") * 100.0).alias("pct_outliers")
    )
    return result.select(
        [
            pl.col(band_col).alias("band"),
            "n",
            "n_outliers",
            "pct_outliers",
            "q1",
            "q3",
            "lower",
            "upper",
        ]
    ).sort("band")


def detect_outliers_isoforest(
    df: pl.DataFrame,
    contamination: float = 0.05,
    seed: int = 42,
    band_col: str = "band",
    value_col: str = "value",
    pixel_id_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Detects multivariate outliers with Isolation Forest.

    Pivots the long-format DataFrame to wide (1 column per band) and trains
    `IsolationForest(contamination=contamination)`. Reports the global
    percentage of detected outliers and the (non-native) approximate
    importance as pct outliers per band in the detected subset.

    Args:
        df: long-format DataFrame with columns `band`, `value`, and a pixel
            identifier (by default `patch_id, t, y, x`).
        contamination: Expected proportion of outliers in [0, 0.5).
        seed: Seed for reproducibility.
        band_col: Band column name.
        value_col: Value column name.
        pixel_id_cols: Columns that uniquely identify each pixel-date.

    Returns:
        DataFrame with columns `band, n, n_outliers, pct_outliers,
        contamination_target`. If there is not enough data, returns an empty DataFrame.
    """
    pixel_id_cols = pixel_id_cols or [c for c in ("patch_id", "t", "y", "x") if c in df.columns]
    if not pixel_id_cols:
        return pl.DataFrame()

    wide = df.pivot(
        on=band_col,
        index=pixel_id_cols,
        values=value_col,
        aggregate_function="mean",
    )
    band_cols = [c for c in wide.columns if c not in pixel_id_cols]
    if not band_cols:
        return pl.DataFrame()

    arr = wide.select(band_cols).drop_nulls().to_numpy()
    if arr.shape[0] < 10:
        return pl.DataFrame()

    iso = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    preds = iso.fit_predict(arr)
    is_out = preds == -1

    # Attribute outliers per band: pct of pixels outside p1/p99 per band
    rows = []
    for i, band in enumerate(band_cols):
        col_vals = arr[:, i]
        out_vals = col_vals[is_out]
        rows.append(
            {
                "band": band,
                "n": int(col_vals.size),
                "n_outliers": int(out_vals.size),
                "pct_outliers": (
                    float(out_vals.size) / col_vals.size * 100.0 if col_vals.size else 0.0
                ),
                "contamination_target": contamination * 100.0,
            }
        )
    # Adjust n_outliers per band as a contribution (not per actual band)
    total_out = int(is_out.sum())
    for r in rows:
        r["n_outliers"] = total_out
        r["pct_outliers"] = float(total_out) / arr.shape[0] * 100.0 if arr.shape[0] else 0.0
        band_pos = band_cols.index(cast("str", r["band"]))
        r["band_mean_outliers"] = float(np.mean(arr[is_out, band_pos])) if total_out else 0.0
    return pl.DataFrame(rows).sort("band")
