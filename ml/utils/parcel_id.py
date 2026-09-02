"""Canonical ``parcel_id`` schema for the AgroSatCopilot project.

Context and motivation
----------------------

The parcel identifier ``parcel_id`` travels through multiple layers of the
project: ingest GeoDataFrames, embedding parquets (FarSLIP, DINOv3,
AlphaEarth), fusion Polars frames (``ml.features.fusion``) and spatial CV
caches. Historically two representations coexisted:

1. ``Int64`` — inherited from PASTIS (numeric ``parcel_id`` from the original
   GeoJSON) and consumed by the GEE samplers.
2. ``Utf8`` — emergent from the baselines that build the id as
   ``"{patch_id}_{i}"`` to identify pixels within patches.

When a ``LEFT JOIN`` mixes both schemas the result is silently empty in
Polars 1.x (the join produces NaN in all the columns of the right-hand side).
The bug manifested in ``05_reencuadre_fenologico.ipynb`` with FarSLIP omitted
despite the canonical parquet existing.

Canonical schema
----------------

As of US-023-preview v2 the official schema is::

    parcel_id: pl.Utf8  (always, throughout the project)

Every block incorporated via a ``LEFT JOIN`` must cast its ``parcel_id``
column to ``pl.Utf8`` before the join. The utility
:func:`canonical_parcel_id` applies the cast idempotently and without losing
precision (it does not use scientific notation for large integers).
"""

from __future__ import annotations

import polars as pl
from polars.datatypes import DataType

__all__ = ["canonical_parcel_id"]


def _is_numeric_dtype(dtype: DataType) -> bool:
    """Returns ``True`` if ``dtype`` is Int/UInt/Float directly castable to Utf8.

    In Polars 1.x ``cast(pl.Utf8)`` over integers produces a decimal
    representation without scientific notation; over floats it may produce
    scientific notation for very large values — in practice the ``parcel_id`` are
    integers.
    """
    return dtype.is_integer() or dtype.is_float()


def canonical_parcel_id(df: pl.DataFrame, col: str = "parcel_id") -> pl.DataFrame:
    """Normalizes the ``parcel_id`` column to the canonical ``pl.Utf8`` schema.

    The function is idempotent: if ``col`` is already ``pl.Utf8``, it returns the
    DataFrame unchanged. For numeric columns (Int8..Int64, UInt*,
    Float32/Float64) it applies a ``cast(pl.Utf8)`` that preserves the decimal
    value without scientific notation for reasonable integers.

    Args:
        df: Polars DataFrame whose ``col`` column is to be normalized.
        col: Name of the column to normalize. Default ``"parcel_id"``.

    Returns:
        A new ``pl.DataFrame`` with ``col`` in dtype ``pl.Utf8``. If ``col`` was
        already ``Utf8``, ``df`` is returned as is (without defensive cloning —
        Polars handles copy-on-write internally).

    Raises:
        KeyError: if ``col`` does not exist in ``df.columns``. An explicit error
            is raised to avoid silent bugs when the caller confuses the column
            name.
        TypeError: if ``col`` is neither numeric nor ``Utf8`` (e.g. ``Datetime``,
            ``List``, ``Struct``). These types require explicit conversion by the
            caller.

    Examples:
        Useful to align the schema before a ``LEFT JOIN``::

            >>> base = pl.DataFrame({"parcel_id": ["p1", "p2"]})
            >>> rhs = pl.DataFrame({"parcel_id": [1, 2], "x": [10.0, 20.0]})
            >>> rhs = canonical_parcel_id(rhs)
            >>> base.join(rhs, on="parcel_id", how="left")  # now both Utf8
    """
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' is not present in the DataFrame. Available columns: {df.columns}"
        )
    dtype = df.schema[col]
    if dtype == pl.Utf8:
        return df
    if _is_numeric_dtype(dtype):
        return df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
    raise TypeError(
        f"Column '{col}' has an unsupported dtype for canonical casting: "
        f"{dtype}. Accepted types: pl.Utf8 or a numeric dtype "
        f"(Int*, UInt*, Float32, Float64)."
    )
