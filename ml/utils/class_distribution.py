"""Class distribution report for baseline notebooks.

Replaces the ad-hoc report "Classes with < 1000 parcels: [...]" that appears
in `notebooks/baseline/05_reencuadre_fenologico.ipynb` and produces
information useful to decide a support threshold, phenological merge via
`PASTIS_R_GROUPINGS`, and stratification of the spatial CV.

Public functions:

- :func:`class_distribution_report` — Polars DataFrame with `class_id`,
  `class_name`, `n_parcels`, `share`, `support_band` (high/med/low/very_low),
  `agronomic_group`, `phenological_cycle`.
- :func:`recommend_threshold` — suggests a sensible threshold based on
  support percentiles, instead of the hardcoded 1000 that broke the report.
- :func:`merge_to_phenological_groups` — groups class_ids according to
  `PASTIS_R_GROUPINGS["phenological_cycle"]` to reduce cardinality and
  enable baselines with balanced classes.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
import structlog

from ml.ingest.pastis_loader import PASTIS_R_CLASSES, PASTIS_R_GROUPINGS

logger = structlog.get_logger(__name__)

__all__ = [
    "SupportBand",
    "class_distribution_report",
    "merge_to_phenological_groups",
    "recommend_threshold",
]

SupportBand = Literal["high", "med", "low", "very_low"]


def class_distribution_report(
    df: pl.DataFrame,
    *,
    class_col: str = "class_id",
    thresholds: tuple[int, int, int] = (1000, 200, 30),
    drop_class_ids: tuple[int, ...] = (0, 19),
) -> pl.DataFrame:
    """Build a detailed class distribution report.

    Resolves the noise produced by the report "Classes with < 1000 parcels:
    [3, 8, ...]" by replacing it with a table of support bands and readable
    names.

    Args:
        df: Polars DataFrame with the `class_col` column.
        class_col: Name of the class column. Default `"class_id"`.
        thresholds: Tuple `(high, med, low)` to classify into support
            bands. `n >= high` is "high"; `med <= n < high` is "med"; `low
            <= n < med` is "low"; `n < low` is "very_low". Default
            `(1000, 200, 30)`.
        drop_class_ids: Class IDs to discard before counting (PASTIS-R 0
            Background and 19 Void). Default `(0, 19)`.

    Returns:
        DataFrame with columns `class_id`, `class_name`, `n_parcels`,
        `share` (proportion), `support_band` (`high|med|low|very_low`),
        `agronomic_group`, `phenological_cycle`. Sorted by `n_parcels`
        descending.
    """
    if class_col not in df.columns:
        raise ValueError(f"`df` no contiene la columna `{class_col}`.")

    filtered = df.filter(
        pl.col(class_col).is_not_null() & ~pl.col(class_col).is_in(list(drop_class_ids))
    )
    counts = (
        filtered.group_by(class_col)
        .len()
        .rename({"len": "n_parcels", class_col: "class_id"})
        .with_columns(pl.col("class_id").cast(pl.Int64))
        .sort("n_parcels", descending=True)
    )
    total = counts["n_parcels"].sum()
    if total == 0:
        logger.warning("class_distribution_empty", n_total=0)
        return counts.with_columns(
            pl.lit(0.0).alias("share"),
            pl.lit("very_low").alias("support_band"),
            pl.lit(None, dtype=pl.Utf8).alias("class_name"),
            pl.lit(None, dtype=pl.Utf8).alias("agronomic_group"),
            pl.lit(None, dtype=pl.Utf8).alias("phenological_cycle"),
        )

    high_t, med_t, low_t = thresholds

    def _band(n: int) -> str:
        if n >= high_t:
            return "high"
        if n >= med_t:
            return "med"
        if n >= low_t:
            return "low"
        return "very_low"

    class_names = {int(k): v for k, v in PASTIS_R_CLASSES.items()}
    agronomic = PASTIS_R_GROUPINGS.get("agronomic_group", {})
    phenological = PASTIS_R_GROUPINGS.get("phenological_cycle", {})

    enriched = counts.with_columns(
        (pl.col("n_parcels") / total).alias("share"),
        pl.col("n_parcels").map_elements(_band, return_dtype=pl.Utf8).alias("support_band"),
        pl.col("class_id")
        .map_elements(
            lambda cid: class_names.get(int(cid), f"class_{int(cid)}"),
            return_dtype=pl.Utf8,
        )
        .alias("class_name"),
        pl.col("class_id")
        .map_elements(
            lambda cid: agronomic.get(int(cid), "unknown"),
            return_dtype=pl.Utf8,
        )
        .alias("agronomic_group"),
        pl.col("class_id")
        .map_elements(
            lambda cid: phenological.get(int(cid), "unknown"),
            return_dtype=pl.Utf8,
        )
        .alias("phenological_cycle"),
    )

    logger.info(
        "class_distribution_report",
        n_classes=enriched.height,
        n_total=int(total),
        n_high=int(enriched.filter(pl.col("support_band") == "high").height),
        n_med=int(enriched.filter(pl.col("support_band") == "med").height),
        n_low=int(enriched.filter(pl.col("support_band") == "low").height),
        n_very_low=int(enriched.filter(pl.col("support_band") == "very_low").height),
    )
    return enriched


def recommend_threshold(
    report: pl.DataFrame,
    *,
    n_count_col: str = "n_parcels",
    method: Literal["p25", "p50", "minmax_balance"] = "p25",
) -> int:
    """Suggest a sensible support threshold for reports.

    The hardcoded threshold of 1000 that appeared in notebooks produced the
    noise "only 1 class qualifies" because PASTIS-R Italy is highly
    imbalanced (1 majority class with ~30k parcels, the rest with <500).

    Args:
        report: DataFrame from `class_distribution_report`.
        n_count_col: Column with the per-class count.
        method: Computation strategy:

            - `"p25"`: 25th percentile of the count (more tolerant).
            - `"p50"`: median of the count.
            - `"minmax_balance"`: geometric mean between min and max.

    Returns:
        Recommended integer threshold. For Italy's 18 classes it typically
        falls in the range [30, 200].
    """
    counts = report[n_count_col].to_numpy()
    if counts.size == 0:
        return 0
    if method == "p25":
        import numpy as np

        return int(np.percentile(counts, 25))
    if method == "p50":
        import numpy as np

        return int(np.percentile(counts, 50))
    if method == "minmax_balance":
        import numpy as np

        return int(np.sqrt(counts.min() * counts.max()))
    raise ValueError(f"`method` no soportado: {method!r}.")


def merge_to_phenological_groups(
    df: pl.DataFrame,
    *,
    class_col: str = "class_id",
    grouping_name: str = "phenological_cycle",
    output_col: str = "pheno_group_id",
) -> pl.DataFrame:
    """Add an agronomic/phenological group column to reduce cardinality.

    Uses `PASTIS_R_GROUPINGS` (loaded from
    `data/reference/pastis_class_mapping.json`). Allows training baselines
    over balanced groups when the 18-class set is too sparse.

    Args:
        df: DataFrame with `class_col`.
        class_col: Column with the PASTIS `class_id`.
        grouping_name: Key of `PASTIS_R_GROUPINGS`. Default
            `"phenological_cycle"` (winter/spring/perennial/... cereals).
        output_col: Name of the new column.

    Returns:
        A copy of the DataFrame with the additional `output_col` column.
    """
    grouping = PASTIS_R_GROUPINGS.get(grouping_name)
    if not grouping:
        raise ValueError(
            f"Agrupacion `{grouping_name}` no disponible en "
            f"PASTIS_R_GROUPINGS. Opciones: {list(PASTIS_R_GROUPINGS)}."
        )

    return df.with_columns(
        pl.col(class_col)
        .map_elements(
            lambda cid: grouping.get(int(cid), "other"),
            return_dtype=pl.Utf8,
        )
        .alias(output_col)
    )
