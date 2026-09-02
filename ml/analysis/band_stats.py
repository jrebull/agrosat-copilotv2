"""Summary statistics per Sentinel-2 band over Polars DataFrames.

Provides `summarize_bands` (means, std, percentiles) and `ndvi_temporal`
(monthly NDVI curve per class x ROI) for the EDA notebooks.
"""

from __future__ import annotations

import polars as pl


def summarize_bands(
    df: pl.DataFrame,
    band_col: str = "band",
    value_col: str = "value",
) -> pl.DataFrame:
    """Compute summary statistics per band.

    Args:
        df: Long-format DataFrame with columns `band_col` and `value_col`.
        band_col: Name of the column containing the band code.
        value_col: Name of the numeric column to summarize.

    Returns:
        DataFrame with columns `band, mean, std, min, max, p5, p25, p50,
        p75, p95` (10 statistics per band).

    Raises:
        ValueError: If `df` does not contain the required columns.
    """
    if band_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Required columns not found: {band_col}, {value_col}")

    return (
        df.group_by(band_col)
        .agg(
            pl.col(value_col).mean().alias("mean"),
            pl.col(value_col).std().alias("std"),
            pl.col(value_col).min().alias("min"),
            pl.col(value_col).max().alias("max"),
            pl.col(value_col).quantile(0.05).alias("p5"),
            pl.col(value_col).quantile(0.25).alias("p25"),
            pl.col(value_col).quantile(0.50).alias("p50"),
            pl.col(value_col).quantile(0.75).alias("p75"),
            pl.col(value_col).quantile(0.95).alias("p95"),
        )
        .sort(band_col)
        .rename({band_col: "band"})
    )


def ndvi_temporal(
    df: pl.DataFrame,
    group_by: list[str] | None = None,
    nir_band: str = "B08",
    red_band: str = "B04",
) -> pl.DataFrame:
    """Compute temporally aggregated NDVI for monthly curves.

    Expects a long-format DataFrame with columns `date` (YYYYMMDD int or str),
    `band`, `value` and any additional grouping column.

    Args:
        df: Long-format DataFrame.
        group_by: Columns to group by (default `["month", "class_name"]`).
        nir_band: Code of the NIR band (default B08).
        red_band: Code of the Red band (default B04).

    Returns:
        DataFrame with `group_by + [ndvi_mean, ndvi_std, n]`.
    """
    group_by = group_by or ["month", "class_name"]

    # Pivot to obtain NIR and Red per pixel
    pivoted = df.filter(pl.col("band").is_in([nir_band, red_band])).pivot(
        on="band",
        index=[c for c in df.columns if c not in ("band", "value")],
        values="value",
        aggregate_function="mean",
    )

    if nir_band not in pivoted.columns or red_band not in pivoted.columns:
        return pl.DataFrame()

    pivoted = pivoted.with_columns(
        (
            (pl.col(nir_band) - pl.col(red_band)) / (pl.col(nir_band) + pl.col(red_band) + 1e-6)
        ).alias("ndvi")
    )

    if "month" in group_by and "month" not in pivoted.columns and "date" in pivoted.columns:
        # date assumed int YYYYMMDD; extract month
        pivoted = pivoted.with_columns(
            ((pl.col("date").cast(pl.Int64) // 100) % 100).alias("month")
        )

    available = [c for c in group_by if c in pivoted.columns]
    if not available:
        return pl.DataFrame()

    return (
        pivoted.group_by(available)
        .agg(
            pl.col("ndvi").mean().alias("ndvi_mean"),
            pl.col("ndvi").std().alias("ndvi_std"),
            pl.len().alias("n"),
        )
        .sort(available)
    )
