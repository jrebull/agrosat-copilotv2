"""Persistence of temporal features into ``features_parcels`` (US-015).

Loads the ``pl.DataFrame`` produced by
:func:`ml.features.temporal_features.extract_temporal_features` into the
``features_parcels`` table with UPSERT on ``(parcel_id, year)``.

Column conventions
------------------
The ``features_parcels`` table (see ``db/migrations/*_create_features_parcels.sql``
of plan US-015 §4.3) mixes scalar columns for frequent queries and two
JSONB columns to avoid 153 individual columns:

- **Scalars**: ``parcel_id``, ``year``, ``sog_doy``, ``peak_doy``,
  ``peak_value``, ``senescence_doy``, ``ndvi_auc``, ``ndvi_slope_pre_peak``,
  ``ndvi_slope_post_peak``, ``maturity_duration_days``.
- **JSONB ``ndvi_stats``**: the 153 descriptive stats (all the
  ``{idx}_{stat}``).
- **JSONB ``phenology``**: the 24 FFT columns (``{idx}_fft_amp_*`` and
  ``{idx}_fft_phase_*``).

The ``alphaearth_embedding`` field is left ``NULL`` in US-015 (populated in
US-016 with the multisensor fusion).
"""

from __future__ import annotations

import json
from typing import Final, Literal

import polars as pl
import structlog
from sqlalchemy import Engine, text

logger = structlog.get_logger(__name__)

#: Scalar columns mapped 1:1 with the table (except ``parcel_id`` and
#: ``year`` which make up the logical PK).
_SCALAR_PHENOLOGY_COLS: Final[tuple[str, ...]] = (
    "sog_doy",
    "peak_doy",
    "peak_value",
    "senescence_doy",
    "ndvi_auc",
    "ndvi_slope_pre_peak",
    "ndvi_slope_post_peak",
    "maturity_duration_days",
)

_PK_COLS: Final[tuple[str, ...]] = ("parcel_id", "year")


def load_features_parcels(
    df: pl.DataFrame,
    engine: Engine,
    *,
    on_conflict: Literal["update", "skip", "raise"] = "update",
) -> int:
    """Load the features DataFrame into ``features_parcels`` with UPSERT.

    Args:
        df: output of
            :func:`ml.features.temporal_features.extract_temporal_features`.
        engine: SQLAlchemy ``Engine`` pointing to Postgres 15 + PostGIS +
            pgvector.
        on_conflict: behavior on violation of
            ``UNIQUE(parcel_id, year)``:

            - ``"update"``: ``ON CONFLICT DO UPDATE SET ...`` (default).
            - ``"skip"``: ``ON CONFLICT DO NOTHING``.
            - ``"raise"``: ``INSERT`` without ``ON CONFLICT``; propagates
              ``sqlalchemy.exc.IntegrityError`` on collision.

    Returns:
        Number of inserted or updated rows. ``0`` if ``df`` is empty.

    Raises:
        sqlalchemy.exc.IntegrityError: in ``"raise"`` mode if the row
            collides with an existing ``(parcel_id, year)`` pair.
        ValueError: if ``df`` does not contain the PK columns.
    """
    if df.is_empty():
        logger.info("load_features_parcels skipped (empty frame)")
        return 0

    for pk in _PK_COLS:
        if pk not in df.columns:
            raise ValueError(f"DataFrame missing PK column '{pk}'")

    stat_cols = _detect_stat_columns(df.columns)
    fft_cols = _detect_fft_columns(df.columns)

    rows: list[dict[str, object]] = []
    for record in df.to_dicts():
        ndvi_stats = {col: _serializable(record[col]) for col in stat_cols}
        phenology_json = {col: _serializable(record[col]) for col in fft_cols}
        row: dict[str, object] = {
            "parcel_id": int(record["parcel_id"]),
            "year": int(record["year"]),
            "ndvi_stats": json.dumps(ndvi_stats),
            "phenology": json.dumps(phenology_json),
        }
        for col in _SCALAR_PHENOLOGY_COLS:
            row[col] = _serializable(record.get(col))
        rows.append(row)

    sql = _build_upsert_sql(on_conflict=on_conflict)

    with engine.begin() as conn:
        result = conn.execute(text(sql), rows)
        affected = result.rowcount if result.rowcount is not None else len(rows)

    logger.info(
        "load_features_parcels completed",
        rows_input=len(rows),
        rows_affected=int(affected),
        on_conflict=on_conflict,
    )
    return int(affected)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_stat_columns(columns: list[str]) -> list[str]:
    """Return the statistical columns (``{idx}_{stat}``) in stable order."""
    stat_suffixes = ("_mean", "_std", "_min", "_max", "_p05", "_p25", "_p50", "_p75", "_p95")
    return sorted(c for c in columns if c.endswith(stat_suffixes))


def _detect_fft_columns(columns: list[str]) -> list[str]:
    """Return the FFT columns (``*_fft_amp_*`` and ``*_fft_phase_*``)."""
    return sorted(c for c in columns if ("_fft_amp_" in c) or ("_fft_phase_" in c))


def _serializable(value: object) -> object:
    """Convert ``NaN``/``None`` to ``None`` and numpy scalars to native Python."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[no-any-return]
        except (AttributeError, ValueError):  # pragma: no cover - defensivo
            return value
    return value


def _build_upsert_sql(*, on_conflict: Literal["update", "skip", "raise"]) -> str:
    """Build the ``INSERT`` with the requested conflict clause.

    Uses named parameters (``:param``) — SQLAlchemy + psycopg bind them
    safely (without string interpolation).
    """
    cols = (
        "parcel_id",
        "year",
        "ndvi_stats",
        "phenology",
        *_SCALAR_PHENOLOGY_COLS,
    )
    col_list = ", ".join(cols)
    placeholder_list = ", ".join(f":{c}" for c in cols)
    # S608: column names come from internal constants
    # (`_SCALAR_PHENOLOGY_COLS`), not from external input. The values are bound
    # via named `:param` parameters.
    base = (
        f"INSERT INTO features_parcels ({col_list}, updated_at) VALUES ({placeholder_list}, now())"  # noqa: S608
    )

    if on_conflict == "raise":
        return base

    if on_conflict == "skip":
        return base + " ON CONFLICT (parcel_id, year) DO NOTHING"

    # update
    update_targets = (
        "ndvi_stats",
        "phenology",
        *_SCALAR_PHENOLOGY_COLS,
    )
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_targets)
    return (
        base + " ON CONFLICT (parcel_id, year) DO UPDATE SET " + set_clause + ", updated_at = now()"
    )


__all__ = ["load_features_parcels"]
