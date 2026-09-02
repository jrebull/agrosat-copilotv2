"""Persistence of the :class:`StandardScaler` fitted on the train split (US-016).

The scaler is fitted only with the ``parcel_id`` of the train split (Fold-0 by
convention) to avoid leakage toward val/test. Persisted with :mod:`joblib`
(not bare ``pickle``) — a signed format, supported by scikit-learn and
compatible with DVC tracking.

The ``StandardScaler`` frame received here must always be the output of
:func:`ml.features.fusion.build_fused_features`. Categorical columns
(``srtm_aspect_dominant``) are excluded automatically if present in
``feature_cols``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

__all__ = [
    "fit_scaler_on_train",
    "load_scaler",
]


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def fit_scaler_on_train(
    df: pl.DataFrame,
    train_ids: tuple[int, ...],
    feature_cols: tuple[str, ...],
    *,
    scaler_path: Path | str,
    version: str = "v1",
    val_ids: tuple[int, ...] | None = None,
    test_ids: tuple[int, ...] | None = None,
) -> StandardScaler:
    """Fit a :class:`StandardScaler` on the train subset and persist it.

    Args:
        df: Fused DataFrame (output of
            :func:`ml.features.fusion.build_fused_features`).
        train_ids: ``parcel_id`` of the train split. Must be disjoint from
            ``val_ids`` and ``test_ids`` (validated if provided).
        feature_cols: Names of numeric columns to standardize.
            Present categorical columns (e.g.
            ``srtm_aspect_dominant``) are filtered out silently with a
            log.warning.
        scaler_path: joblib destination. Convention
            ``artifacts/scaler_{version}.pkl``. The parent directory is
            created if it does not exist.
        version: Version tag injected into the scaler metadata for
            downstream traceability.
        val_ids: ``parcel_id`` of the val split (optional, to validate
            absence of leakage).
        test_ids: ``parcel_id`` of the test split (optional, same purpose).

    Returns:
        The fitted :class:`StandardScaler`. Additional attributes in
        ``_agrosat_meta``: ``{"version", "feature_cols", "n_train"}``.

    Raises:
        ValueError: if ``train_ids`` intersects with ``val_ids``/``test_ids``,
            if ``feature_cols`` is empty after filtering categoricals, or if
            ``df`` does not contain ``parcel_id``.
    """
    if "parcel_id" not in df.columns:
        raise ValueError("`df` must contain the `parcel_id` column.")

    if val_ids is not None or test_ids is not None:
        _validate_no_leakage(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    numeric_cols = _filter_numeric(df=df, feature_cols=feature_cols)
    if not numeric_cols:
        raise ValueError(
            "No numeric columns remain after filtering categoricals; check `feature_cols`."
        )

    train_set = set(int(x) for x in train_ids)
    # Match the column dtype explicitly: ``parcel_id`` may be stored as String while
    # the ids arrive as ints, and polars >= 1.3x refuses ``is_in`` across dtypes.
    train_id_series = pl.Series("parcel_id", sorted(train_set)).cast(df.schema["parcel_id"])
    train_df = df.filter(pl.col("parcel_id").is_in(train_id_series)).select(numeric_cols)
    if train_df.height == 0:
        raise ValueError(
            "After filtering by `train_ids` the frame is empty. IDs in another fold?"
        )

    matrix = train_df.to_numpy()
    # Convert +/-inf to NaN before any statistic. The spectral indices
    # with divisions (MCARI, GCVI, PSRI, etc.) can produce
    # inf when the denominator is zero or very close to it. StandardScaler does
    # not accept inf; treating them as NaN allows imputing them with the column
    # mean like any missing value.
    n_inf = int(np.isinf(matrix).sum())
    if n_inf > 0:
        inf_cols_mask = np.any(np.isinf(matrix), axis=0)
        inf_cols = [c for c, has_inf in zip(numeric_cols, inf_cols_mask, strict=True) if has_inf]
        logger.warning(
            "scaler_replaced_inf_with_nan",
            n_inf_values=n_inf,
            n_cols_affected=int(inf_cols_mask.sum()),
            examples=inf_cols[:5],
            note=(
                "Indices espectrales con division por ~0 producen inf. "
                "Se reemplazan por NaN para imputacion por media de columna."
            ),
        )
        matrix = np.where(np.isinf(matrix), np.nan, matrix)
    # Filter all-NaN columns before the fit to avoid `RuntimeWarning: Mean
    # of empty slice` in np.nanmean + `invalid value encountered in divide` in
    # sklearn. It happens when the frame comes from the demo mode without GEE (all
    # the columns of non-injected blocks are null).
    all_nan_mask = np.all(np.isnan(matrix), axis=0)
    if all_nan_mask.any():
        dropped_all_nan = [c for c, drop in zip(numeric_cols, all_nan_mask, strict=True) if drop]
        logger.warning(
            "scaler_dropped_all_nan_columns",
            n_dropped=len(dropped_all_nan),
            examples=dropped_all_nan[:5],
            note="Frame sin GEE poblado; el scaler ignora estas columnas.",
        )
        numeric_cols = [c for c, drop in zip(numeric_cols, all_nan_mask, strict=True) if not drop]
        if not numeric_cols:
            raise ValueError(
                "All numeric columns were all-NaN. Frame without GEE populated?"
            )
        matrix = matrix[:, ~all_nan_mask]
    # Replace remaining NaN with the column mean (StandardScaler
    # does not accept NaN). We already guaranteed that no column is all-NaN, so
    # nanmean emits no warnings.
    col_means = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    # polars >= 1.3x may hand back a read-only zero-copy array; imputation
    # below writes in place, so own the buffer first.
    if not matrix.flags.writeable:
        matrix = matrix.copy()
    matrix[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    scaler.fit(matrix)

    # Inject metadata for downstream + audit.
    scaler._agrosat_meta = {  # type: ignore[attr-defined]
        "version": version,
        "feature_cols": tuple(numeric_cols),
        "n_train": int(train_df.height),
    }

    out_path = Path(scaler_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)
    logger.info(
        "scaler_persisted",
        path=str(out_path),
        version=version,
        n_features=len(numeric_cols),
        n_train=int(train_df.height),
    )
    return scaler


def load_scaler(path: Path | str) -> StandardScaler:
    """Load a :class:`StandardScaler` persisted with :func:`fit_scaler_on_train`.

    Args:
        path: Path to the joblib file (``artifacts/scaler_v1.pkl`` by
            convention).

    Returns:
        Scaler with ``_agrosat_meta`` metadata if it was fitted by this
        module, or the raw scaler if it was produced by another pipeline.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a valid :class:`StandardScaler`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scaler not found at {p}.")
    obj = joblib.load(p)
    if not isinstance(obj, StandardScaler):
        raise ValueError(
            f"The file at {p} is not a StandardScaler (type={type(obj).__name__})."
        )
    return cast(StandardScaler, obj)


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _validate_no_leakage(
    *,
    train_ids: tuple[int, ...],
    val_ids: tuple[int, ...] | None,
    test_ids: tuple[int, ...] | None,
) -> None:
    """Raise :class:`ValueError` if ``train_ids`` intersects val/test."""
    train_set = set(int(x) for x in train_ids)
    for label, ids in (("val", val_ids), ("test", test_ids)):
        if not ids:
            continue
        overlap = train_set.intersection(int(x) for x in ids)
        if overlap:
            raise ValueError(
                f"Leakage detected: train_ids intersects {label}_ids in "
                f"{len(overlap)} parcel_ids (e.g. {sorted(overlap)[:3]}...)."
            )


def _filter_numeric(*, df: pl.DataFrame, feature_cols: tuple[str, ...]) -> list[str]:
    """Filter `feature_cols` keeping only the numeric ones present in df."""
    keep: list[str] = []
    dropped: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            dropped.append(col)
            continue
        dtype: Any = df.schema[col]
        # Polars dtype helpers: numeric == Float* | Int* | UInt*.
        if dtype.is_numeric():
            keep.append(col)
        else:
            dropped.append(col)
    if dropped:
        logger.warning(
            "scaler_dropped_non_numeric",
            n_dropped=len(dropped),
            examples=dropped[:5],
        )
    return keep
