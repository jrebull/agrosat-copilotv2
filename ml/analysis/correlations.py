"""Bivariate, multivariate and temporal analysis for US-012 (EDA).

Provides 7 reusable functions over Polars DataFrames with Sentinel-2 bands,
derived spectral indices and per-parcel time series:

- `compute_indices_subset`: adds columns with the core subset of 6 spectral
  indices (NDVI, NDWI, NDMI, EVI, SAVI, NDRE).
- `correlation_pair`: long-format correlation matrix between two column
  subsets (Pearson or Spearman).
- `vif_table`: Variance Inflation Factor per column with statsmodels.
- `phenology_peaks`: detects the NDVI peak per parcel and returns month, doy
  and year.
- `acf_pacf_per_parcel`: ACF and PACF of the per-parcel NDVI series after
  monthly resampling with linear interpolation.
- `dtw_cluster_temporal`: DTW clustering (`tslearn.TimeSeriesKMeans`) with a
  Sakoe-Chiba band over z-normalized NDVI series.
- `era5_ndvi_anomaly`: crosses annual ERA5 precipitation with annual maximum
  NDVI, flagging dry vs normal years by percentile.

Polars note: the pandas adapter is used only as a technical boundary for
`statsmodels.tsa.stattools` (ACF/PACF) and `variance_inflation_factor` when
those libraries do not accept Polars directly. All persistence and
aggregation stays in Polars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from polars._typing import ColumnNameOrSelector, PolarsDataType

SPECTRAL_INDICES_CORE: dict[str, str] = {
    "NDVI": "(B08-B04)/(B08+B04)",
    "NDWI": "(B03-B08)/(B03+B08)",
    "NDMI": "(B08-B11)/(B08+B11)",
    "EVI": "2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)",
    "SAVI": "1.5*(B08-B04)/(B08+B04+0.5)",
    "NDRE": "(B08-B05)/(B08+B05)",
}
"""Core subset of 6 spectral indices for US-012.

The full library (17 indices) is delivered in US-014 with `spyndex`. The
formulas use Sentinel-2 bands at the original scale (not divided by 10000);
internally the implementation casts to Float64 and applies an epsilon to
avoid divisions by zero.
"""

_EPS: float = 1e-6
_DEFAULT_REQUIRED_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B05", "B08", "B11")


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Safe division with epsilon in the denominator (Polars-vectorized)."""
    return num / pl.when(den.abs() < _EPS).then(_EPS).otherwise(den)


def compute_indices_subset(
    df_bands: pl.DataFrame,
    indices: list[str] | None = None,
    scale: float = 1.0,
    clip_negative: bool = False,
    mask_invalid_band_range: tuple[float, float] | None = None,
    clip_evi_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Compute the core subset of 6 spectral indices, vectorized in Polars.

    The bands must be in separate columns (wide format) with the canonical
    PASTIS-R names: `B02, B03, B04, B05, B08, B11`. If you receive a
    long-format DataFrame from `pastis_to_polars`, pivot first by
    `(patch_id, t, y, x)` before passing it to this function.

    Args:
        df_bands: Polars DataFrame with the Sentinel-2 bands as float columns
            (or castable to float). At least `B02, B03, B04, B05, B08, B11`.
        indices: Optional subset of indices to compute; default all 6.
        scale: Multiplicative factor applied to each band before the
            computation (default 1.0, no scaling). For Sentinel-2 L2A in raw
            DN (range 0-10000) use 1e-4 to bring it to reflectance [0, 1] and
            avoid EVI overflow from a near-zero denominator.
        clip_negative: If True, clips negative values to 0 before the
            computation (BOA artifacts can produce negative DN, which breaks
            the normalized indices when a + b is near 0). Applied before the
            index computation. Using `mask_invalid_band_range` instead is
            recommended to preserve real variability.
        mask_invalid_band_range: Post-scale `(min, max)` tuple. Timesteps with
            any required band outside this range are filtered (dropped) before
            computing indices. Example: `(0.0, 1.5)` with `scale=1e-4`
            discards negative DN (BOA artifacts) and reflectances > 1.5
            (saturated clouds). Mutually exclusive with `clip_negative`.
        clip_evi_range: Optional `(min, max)` tuple applied to `EVI` after the
            computation. Useful because the EVI formula can produce outliers
            from denominator geometry even with valid bands. Default does not
            clip.

    Returns:
        The original DataFrame with additional columns `NDVI, NDWI, NDMI, EVI,
        SAVI, NDRE` (or the requested subset). If `mask_invalid_band_range` is
        passed, the returned DataFrame may have fewer rows than the input
        (invalid timesteps are filtered out). If `df_bands` is empty or is
        missing required bands for the requested indices, it returns the
        original df unchanged.
    """
    if clip_negative and mask_invalid_band_range is not None:
        raise ValueError("clip_negative and mask_invalid_band_range are mutually exclusive")
    requested = indices or list(SPECTRAL_INDICES_CORE.keys())
    unknown = [i for i in requested if i not in SPECTRAL_INDICES_CORE]
    if unknown:
        raise ValueError(f"Unsupported indices: {unknown}")

    if df_bands.is_empty():
        # Append empty Float64 columns to preserve the downstream contract
        new_cols = [pl.lit(None, dtype=pl.Float64).alias(name) for name in requested]
        return df_bands.with_columns(new_cols) if new_cols else df_bands

    missing = [b for b in _DEFAULT_REQUIRED_BANDS if b not in df_bands.columns]
    if missing:
        # We cannot compute; return the original df unchanged.
        return df_bands

    df = df_bands
    if mask_invalid_band_range is not None:
        lo, hi = mask_invalid_band_range
        # Scale the bands to evaluate the range in the same unit as `scale`
        band_scaled = [
            (pl.col(b).cast(pl.Float64) * scale).alias(f"__scaled_{b}")
            for b in _DEFAULT_REQUIRED_BANDS
        ]
        df = df.with_columns(band_scaled)
        keep = pl.lit(True)
        for b in _DEFAULT_REQUIRED_BANDS:
            keep = keep & pl.col(f"__scaled_{b}").is_between(lo, hi, closed="both")
        df = df.filter(keep).drop([f"__scaled_{b}" for b in _DEFAULT_REQUIRED_BANDS])
        if df.is_empty():
            new_cols = [pl.lit(None, dtype=pl.Float64).alias(name) for name in requested]
            return df.with_columns(new_cols) if new_cols else df

    def _cast_band(name: str) -> pl.Expr:
        expr = pl.col(name).cast(pl.Float64)
        if scale != 1.0:
            expr = expr * scale
        if clip_negative:
            expr = pl.when(expr < 0).then(0.0).otherwise(expr)
        return expr

    casts = {b: _cast_band(b) for b in _DEFAULT_REQUIRED_BANDS}

    exprs: list[pl.Expr] = []
    if "NDVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + casts["B04"]
        exprs.append(_safe_div(num, den).alias("NDVI"))
    if "NDWI" in requested:
        num = casts["B03"] - casts["B08"]
        den = casts["B03"] + casts["B08"]
        exprs.append(_safe_div(num, den).alias("NDWI"))
    if "NDMI" in requested:
        num = casts["B08"] - casts["B11"]
        den = casts["B08"] + casts["B11"]
        exprs.append(_safe_div(num, den).alias("NDMI"))
    if "EVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + 6.0 * casts["B04"] - 7.5 * casts["B02"] + 1.0
        exprs.append((2.5 * _safe_div(num, den)).alias("EVI"))
    if "SAVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + casts["B04"] + 0.5
        exprs.append((1.5 * _safe_div(num, den)).alias("SAVI"))
    if "NDRE" in requested:
        num = casts["B08"] - casts["B05"]
        den = casts["B08"] + casts["B05"]
        exprs.append(_safe_div(num, den).alias("NDRE"))

    df_out = df.with_columns(exprs)
    if clip_evi_range is not None and "EVI" in requested:
        lo, hi = clip_evi_range
        df_out = df_out.with_columns(pl.col("EVI").clip(lo, hi).alias("EVI"))
    return df_out


def correlation_pair(
    df: pl.DataFrame,
    cols_a: list[str],
    cols_b: list[str],
    method: Literal["pearson", "spearman"] = "pearson",
) -> pl.DataFrame:
    """Long-format correlation matrix between two column subsets.

    Args:
        df: Polars DataFrame with the columns in `cols_a` and `cols_b`.
        cols_a: Columns of the first subset (heatmap rows).
        cols_b: Columns of the second subset (heatmap columns).
        method: `pearson` or `spearman`.

    Returns:
        Polars DataFrame with columns `feature_a, feature_b, corr, abs_corr`,
        sorted by `abs_corr` desc. If `df` is empty or any column is missing,
        returns a DataFrame with the correct schema but no rows.
    """
    schema = {
        "feature_a": pl.Utf8,
        "feature_b": pl.Utf8,
        "corr": pl.Float64,
        "abs_corr": pl.Float64,
    }
    if df.is_empty() or not cols_a or not cols_b:
        return pl.DataFrame(schema=schema)
    missing = [c for c in (cols_a + cols_b) if c not in df.columns]
    if missing:
        return pl.DataFrame(schema=schema)

    all_cols = list(dict.fromkeys(cols_a + cols_b))
    arr = df.select(all_cols).cast(pl.Float64, strict=False).to_numpy()
    mask = ~np.isnan(arr).any(axis=1)
    arr_v = arr[mask]
    if arr_v.shape[0] < 2:
        return pl.DataFrame(schema=schema)

    if method == "spearman":
        from scipy.stats import rankdata

        arr_v = np.apply_along_axis(rankdata, 0, arr_v)
    corr_full = np.corrcoef(arr_v, rowvar=False)

    idx = {c: i for i, c in enumerate(all_cols)}
    rows: list[dict[str, Any]] = []
    for ca in cols_a:
        for cb in cols_b:
            val = corr_full[idx[ca], idx[cb]]
            v = float(val) if np.isfinite(val) else float("nan")
            rows.append(
                {
                    "feature_a": ca,
                    "feature_b": cb,
                    "corr": v,
                    "abs_corr": abs(v) if np.isfinite(v) else float("nan"),
                }
            )
    out = pl.DataFrame(rows, schema=schema)
    return out.sort("abs_corr", descending=True, nulls_last=True)


def vif_table(
    df: pl.DataFrame,
    cols: list[str],
    drop_na: bool = True,
    near_perfect_corr_threshold: float = 0.99,
) -> pl.DataFrame:
    """Variance Inflation Factor per column using statsmodels.

    Pre-filters columns with absolute correlation greater than
    `near_perfect_corr_threshold` to avoid singular matrices (the first of
    each pair is kept, the second is dropped and documented as
    `dropped_near_perfect_corr`).

    Args:
        df: Polars DataFrame with the numeric columns.
        cols: List of columns over which to compute the VIF.
        drop_na: Whether to drop rows with NaN in any of `cols` before the VIF.
        near_perfect_corr_threshold: `|corr|` threshold above which a pair is
            considered near-perfect redundancy (default 0.99).

    Returns:
        Polars DataFrame with columns `feature, vif, status`
        (`status in {"ok", "warning", "drop", "dropped_near_perfect_corr"}`),
        sorted by VIF desc. If statsmodels is not installed or `cols` is
        empty, returns an empty DataFrame with a valid schema.
    """
    schema = {"feature": pl.Utf8, "vif": pl.Float64, "status": pl.Utf8}
    if df.is_empty() or not cols:
        return pl.DataFrame(schema=schema)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pl.DataFrame(schema=schema)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:  # pragma: no cover - statsmodels is in the ml group
        return pl.DataFrame(schema=schema)

    sub = df.select(cols).cast(pl.Float64, strict=False)
    if drop_na:
        sub = sub.drop_nulls()
    arr = sub.to_numpy()
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return pl.DataFrame(schema=schema)

    # Pre-filter of nearly perfectly correlated pairs to avoid
    # singular matrices when inverting.
    corr_mat = np.corrcoef(arr, rowvar=False)
    n = arr.shape[1]
    to_drop: set[int] = set()
    drop_reasons: dict[str, str] = {}
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            val = corr_mat[i, j]
            if np.isfinite(val) and abs(val) >= near_perfect_corr_threshold:
                to_drop.add(j)
                drop_reasons[cols[j]] = (
                    f"|corr|={abs(float(val)):.3f} con {cols[i]} >= {near_perfect_corr_threshold}"
                )

    keep_idx = [i for i in range(n) if i not in to_drop]
    keep_cols = [cols[i] for i in keep_idx]
    arr_keep = arr[:, keep_idx]

    rows: list[dict[str, Any]] = []
    if arr_keep.shape[1] >= 2:
        for k, name in enumerate(keep_cols):
            try:
                v = float(variance_inflation_factor(arr_keep, k))
            except Exception:  # noqa: BLE001
                v = float("inf")
            if not np.isfinite(v):
                status = "drop"
            elif v >= 10.0:
                status = "drop"
            elif v >= 5.0:
                status = "warning"
            else:
                status = "ok"
            rows.append({"feature": name, "vif": v, "status": status})

    for name in cols:
        if name in drop_reasons:
            rows.append(
                {
                    "feature": name,
                    "vif": float("inf"),
                    "status": "dropped_near_perfect_corr",
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort("vif", descending=True, nulls_last=True)


def phenology_peaks(
    df_ts: pl.DataFrame,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    ndvi_col: str = "ndvi",
    class_col: str = "class_name",
) -> pl.DataFrame:
    """Detect the NDVI peak per parcel.

    The `date` column can be a Date / Datetime type or an integer `YYYYMMDD`.
    The function normalizes it to `pl.Date` before extracting month / doy /
    year.

    Args:
        df_ts: Polars DataFrame with per-parcel NDVI series. Required columns:
            `parcel_col, date_col, ndvi_col, class_col`.
        parcel_col: Name of the parcel identifier column.
        date_col: Name of the date column (Date / Datetime / Int).
        ndvi_col: Name of the NDVI column.
        class_col: Name of the class column (preserved in the output).

    Returns:
        Polars DataFrame with columns `parcel_id, class_name, peak_ndvi_value,
        peak_ndvi_month, peak_ndvi_doy, peak_ndvi_year`. Empty (with schema)
        if the input is empty or required columns are missing.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "peak_ndvi_value": pl.Float64,
        "peak_ndvi_month": pl.Int64,
        "peak_ndvi_doy": pl.Int64,
        "peak_ndvi_year": pl.Int64,
    }
    if df_ts.is_empty():
        return pl.DataFrame(schema=schema)
    required = {parcel_col, date_col, ndvi_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return pl.DataFrame(schema=schema)

    df = df_ts.select([parcel_col, date_col, ndvi_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, ndvi_col]
    )
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Normalize date to pl.Date regardless of the original dtype.
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        # Try parsing ISO format; if it fails fall back to YYYYMMDD
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        # Assume integer YYYYMMDD (PASTIS dates-S2 format).
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )

    df = df.drop_nulls(subset=["_d"])
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Peak per parcel
    idx_max = (
        df.group_by(parcel_col)
        .agg(pl.col(ndvi_col).arg_max().alias("__idx"))
        .with_columns(pl.col("__idx").cast(pl.Int64))
    )
    df_with_pos = df.with_columns(pl.int_range(0, pl.len()).over(parcel_col).alias("__pos"))
    df_peak = df_with_pos.join(
        idx_max,
        left_on=[parcel_col, "__pos"],
        right_on=[parcel_col, "__idx"],
        how="inner",
    )

    out = df_peak.select(
        [
            pl.col(parcel_col).cast(pl.Utf8).alias("parcel_id"),
            pl.col(class_col).cast(pl.Utf8).alias("class_name"),
            pl.col(ndvi_col).cast(pl.Float64).alias("peak_ndvi_value"),
            pl.col("_d").dt.month().cast(pl.Int64).alias("peak_ndvi_month"),
            pl.col("_d").dt.ordinal_day().cast(pl.Int64).alias("peak_ndvi_doy"),
            pl.col("_d").dt.year().cast(pl.Int64).alias("peak_ndvi_year"),
        ]
    )
    return out.unique(subset=["parcel_id"], keep="first").cast(
        cast("Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType]", schema)
    )


def _resample_monthly_pandas(
    parcel_id: str,
    dates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample monthly with linear interpolation using pandas as an adapter.

    `pandas.resample("MS").mean()` is the shortest path: Sentinel-2 dates are
    irregular (median ~5 days due to cloud filtering) and we need a uniform
    step before ACF/PACF.

    Args:
        parcel_id: Parcel identifier (logging only).
        dates: `datetime64[D]` array sorted ascending.
        values: NDVI array aligned with `dates`.

    Returns:
        Tuple `(months_ts, ndvi_monthly)` with a monthly step (`MS` = month
        start).
    """
    import pandas as pd

    _ = parcel_id
    if dates.size == 0 or values.size == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float64)
    ser = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    ser = ser[~ser.index.duplicated(keep="first")]
    monthly = ser.resample("MS").mean().interpolate(method="linear", limit_direction="both")
    return monthly.index.to_numpy(), monthly.to_numpy(dtype=np.float64)


def acf_pacf_per_parcel(
    df_ts: pl.DataFrame,
    max_lag: int = 6,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    series_col: str = "ndvi",
    class_col: str = "class_name",
) -> pl.DataFrame:
    """ACF and PACF of the per-parcel NDVI series after monthly resampling.

    Pre-filters classes (`class_id`) outside `[1, 18]` when the `class_id`
    column exists, to exclude background (0) and void (19) in PASTIS-R.
    Before the computation, each series is resampled at a monthly step and
    linearly interpolated (PASTIS Sentinel-2 has a median ~5 days between
    acquisitions but is irregular due to cloud filtering).

    Polars note: `pandas` is used as a boundary adapter because
    `statsmodels.tsa.stattools.acf/pacf` does not accept Polars and because
    `pl.DataFrame.upsample` does not support linear interpolation
    out-of-the-box over series with arbitrary real values.

    Args:
        df_ts: Long Polars DataFrame with required columns
            `parcel_col, date_col, series_col, class_col`. Optional `class_id`
            to filter `[1, 18]`.
        max_lag: Number of lags to compute (default 6, justified by PASTIS
            coverage ~14 months).
        parcel_col: Parcel identifier column.
        date_col: Date column (Date / Datetime / Int YYYYMMDD).
        series_col: NDVI column.
        class_col: Class column (preserved in the output).

    Returns:
        Long Polars DataFrame with columns
        `parcel_id, class_name, lag, acf, pacf`. ACF and PACF are bounded in
        `[-1, 1]`. `acf[0] = 1.0` always.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "lag": pl.Int64,
        "acf": pl.Float64,
        "pacf": pl.Float64,
    }
    if df_ts.is_empty():
        return pl.DataFrame(schema=schema)
    required = {parcel_col, date_col, series_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return pl.DataFrame(schema=schema)

    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:  # pragma: no cover
        return pl.DataFrame(schema=schema)

    df = df_ts.clone()
    if "class_id" in df.columns:
        df = df.filter(pl.col("class_id").is_between(1, 18))
    df = df.select([parcel_col, date_col, series_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, series_col]
    )
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Normalize date to pl.Date for clean conversion to numpy datetime64
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )
    df = df.drop_nulls(subset=["_d"]).sort([parcel_col, "_d"])
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for pid, sub in df.group_by(parcel_col, maintain_order=True):
        parcel_id = str(pid[0]) if isinstance(pid, tuple) else str(pid)
        sub_sorted = sub.sort("_d")
        dates = sub_sorted["_d"].to_numpy()
        vals = sub_sorted[series_col].cast(pl.Float64).to_numpy()
        cls_series = sub_sorted[class_col].to_list()
        class_name = str(cls_series[0]) if cls_series else "unknown"

        _, monthly = _resample_monthly_pandas(parcel_id, dates, vals)
        if monthly.size < 3:
            continue
        # Cap max_lag to the effective series size - 1
        eff_lag = max(1, min(max_lag, monthly.size - 1))
        try:
            acf_vals = acf(monthly, nlags=eff_lag, fft=False, missing="drop")
        except Exception:  # noqa: BLE001, S112
            # statsmodels may fail with degenerate series (zero variance);
            # skip the parcel without registering it.
            continue
        try:
            pacf_vals = pacf(monthly, nlags=eff_lag, method="ywm")
        except Exception:  # noqa: BLE001
            pacf_vals = np.full(eff_lag + 1, np.nan, dtype=np.float64)

        for lag_i in range(eff_lag + 1):
            a = float(acf_vals[lag_i]) if lag_i < acf_vals.size else float("nan")
            p = float(pacf_vals[lag_i]) if lag_i < pacf_vals.size else float("nan")
            # Clip to [-1, 1] for numerical safety
            if np.isfinite(a):
                a = float(max(-1.0, min(1.0, a)))
            if np.isfinite(p):
                p = float(max(-1.0, min(1.0, p)))
            rows.append(
                {
                    "parcel_id": parcel_id,
                    "class_name": class_name,
                    "lag": int(lag_i),
                    "acf": a,
                    "pacf": p,
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def dtw_cluster_temporal(
    df_ts: pl.DataFrame,
    n_clusters: int = 4,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    series_col: str = "ndvi",
    class_col: str = "class_name",
    sakoe_chiba_radius: int = 3,
    seed: int = 42,
) -> tuple[pl.DataFrame, Any]:
    """DTW clustering with `tslearn.TimeSeriesKMeans` and a Sakoe-Chiba band.

    Pre-filters `class_id in [1, 18]` when the column exists. Each NDVI series
    is resampled monthly, interpolated and z-normalized per parcel before the
    DTW fit. The Sakoe-Chiba band (`sakoe_chiba_radius`) bounds the DTW cost
    to O(T*radius) instead of O(T^2).

    Args:
        df_ts: Long Polars DataFrame with per-parcel NDVI series. Required
            columns: `parcel_col, date_col, series_col, class_col`. Optional
            `class_id` to filter `[1, 18]`.
        n_clusters: Number of DTW clusters (default 4).
        parcel_col: Parcel identifier column.
        date_col: Date column.
        series_col: NDVI column.
        class_col: Class column (preserved in the output).
        sakoe_chiba_radius: Radius of the Sakoe-Chiba band (default 3).
        seed: Seed for reproducibility.

    Returns:
        Tuple `(df_with_cluster, fitted_model)` where `df_with_cluster` has
        columns `parcel_id, class_name, cluster_id`, and `fitted_model` is the
        fitted `TimeSeriesKMeans` (with `cluster_centers_` accessible). If
        `tslearn` is not installed or there are not enough series, returns an
        empty DataFrame + `None`.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "cluster_id": pl.Int64,
    }
    empty = pl.DataFrame(schema=schema)
    if df_ts.is_empty():
        return empty, None
    required = {parcel_col, date_col, series_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return empty, None

    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.utils import to_time_series_dataset
    except ImportError:  # pragma: no cover
        return empty, None

    # Compat tslearn 0.6.3 + scikit-learn >= 1.6: sklearn renamed the kwarg
    # `force_all_finite` to `ensure_all_finite`. tslearn still calls it with
    # the old name. The shim is applied only once per process.
    try:
        import inspect

        import sklearn.utils.validation as _skv

        _check_array_sig = inspect.signature(_skv.check_array)
        if "force_all_finite" not in _check_array_sig.parameters:
            _orig_check_array = _skv.check_array

            def _check_array_compat(*args: Any, **kwargs: Any) -> Any:
                if "force_all_finite" in kwargs:
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return _orig_check_array(*args, **kwargs)

            _skv.check_array = _check_array_compat
            try:  # tslearn.clustering.kmeans imports check_array by name
                import tslearn.clustering.kmeans as _tskm

                _tskm.check_array = _check_array_compat
            except Exception:  # noqa: BLE001, S110  # pragma: no cover
                pass
    except Exception:  # noqa: BLE001, S110  # pragma: no cover
        pass

    df = df_ts.clone()
    if "class_id" in df.columns:
        df = df.filter(pl.col("class_id").is_between(1, 18))
    df = df.select([parcel_col, date_col, series_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, series_col]
    )
    if df.is_empty():
        return empty, None

    # Normalize date to pl.Date
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )
    df = df.drop_nulls(subset=["_d"]).sort([parcel_col, "_d"])
    if df.is_empty():
        return empty, None

    series_list: list[np.ndarray] = []
    parcel_ids: list[str] = []
    class_names: list[str] = []
    for pid, sub in df.group_by(parcel_col, maintain_order=True):
        parcel_id = str(pid[0]) if isinstance(pid, tuple) else str(pid)
        sub_sorted = sub.sort("_d")
        dates = sub_sorted["_d"].to_numpy()
        vals = sub_sorted[series_col].cast(pl.Float64).to_numpy()
        cls_series = sub_sorted[class_col].to_list()
        class_name = str(cls_series[0]) if cls_series else "unknown"

        _, monthly = _resample_monthly_pandas(parcel_id, dates, vals)
        if monthly.size < 3:
            continue
        # z-score per parcel
        mean = float(np.mean(monthly))
        std = float(np.std(monthly))
        if std < _EPS:
            continue
        z = (monthly - mean) / std
        series_list.append(z.astype(np.float64))
        parcel_ids.append(parcel_id)
        class_names.append(class_name)

    if len(series_list) < n_clusters:
        return empty, None

    X = to_time_series_dataset(series_list)
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        metric_params={"sakoe_chiba_radius": sakoe_chiba_radius},
        random_state=seed,
        n_init=2,
        max_iter=10,
    )
    labels = model.fit_predict(X)

    out = pl.DataFrame(
        {
            "parcel_id": parcel_ids,
            "class_name": class_names,
            "cluster_id": [int(x) for x in labels],
        },
        schema=schema,
    )
    return out, model


def era5_ndvi_anomaly(
    df_era5: pl.DataFrame,
    df_ndvi_annual: pl.DataFrame,
    dry_year_percentile: float = 0.25,
) -> pl.DataFrame:
    """Cross annual ERA5 precipitation with annual maximum NDVI, flag dry years.

    Args:
        df_era5: Polars DataFrame with `year, roi_name, precip_mm`.
        df_ndvi_annual: Polars DataFrame with `year, roi_name, ndvi_max`.
        dry_year_percentile: Percentile to consider a year as "dry"
            (default 0.25 = below the first quartile of historical
            precipitation per ROI).

    Returns:
        Polars DataFrame with columns `year, roi_name, precip_mm, ndvi_max,
        ndvi_anomaly_z, is_dry_year`. The z anomaly is computed per ROI over
        `ndvi_max`. If any of the inputs is empty, returns an empty DataFrame
        with a valid schema.
    """
    schema = {
        "year": pl.Int64,
        "roi_name": pl.Utf8,
        "precip_mm": pl.Float64,
        "ndvi_max": pl.Float64,
        "ndvi_anomaly_z": pl.Float64,
        "is_dry_year": pl.Boolean,
    }
    if df_era5.is_empty() or df_ndvi_annual.is_empty():
        return pl.DataFrame(schema=schema)
    required_era5 = {"year", "roi_name", "precip_mm"}
    required_ndvi = {"year", "roi_name", "ndvi_max"}
    if not required_era5.issubset(set(df_era5.columns)):
        return pl.DataFrame(schema=schema)
    if not required_ndvi.issubset(set(df_ndvi_annual.columns)):
        return pl.DataFrame(schema=schema)

    merged = df_era5.join(df_ndvi_annual, on=["year", "roi_name"], how="inner")
    if merged.is_empty():
        return pl.DataFrame(schema=schema)

    mean_expr = pl.col("ndvi_max").mean().over("roi_name")
    std_expr = pl.col("ndvi_max").std().over("roi_name").fill_null(1.0)
    dry_thr = pl.col("precip_mm").quantile(dry_year_percentile).over("roi_name")
    merged = merged.with_columns(
        [
            ((pl.col("ndvi_max") - mean_expr) / std_expr).alias("ndvi_anomaly_z"),
            (pl.col("precip_mm") <= dry_thr).alias("is_dry_year"),
        ]
    )
    return merged.select(
        [
            pl.col("year").cast(pl.Int64),
            pl.col("roi_name").cast(pl.Utf8),
            pl.col("precip_mm").cast(pl.Float64),
            pl.col("ndvi_max").cast(pl.Float64),
            pl.col("ndvi_anomaly_z").cast(pl.Float64),
            pl.col("is_dry_year").cast(pl.Boolean),
        ]
    ).sort(["roi_name", "year"])
