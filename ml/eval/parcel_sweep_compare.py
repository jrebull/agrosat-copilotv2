"""Compare the parcel-level FarSLIP N-class sweep with and without the 3:1 filter.

The per-parcel sweep (``scripts/run_us036b_parcel_sweep.py``) writes a CSV with
one row per N: ``n_classes, macro_f1, macro_iou, n_well_resolved,
n_eval_parcels, dominance_ratio, best_ckpt``. To answer "does the legacy 3:1
Meadow dominance filter help or hurt at the parcel grain?", we run the sweep
twice -- once with ``dominance_ratio=None`` (the default; parcel grain handles
imbalance) and once with ``dominance_ratio=3.0`` -- and join the two curves on
``n_classes``.

This module loads both CSVs and produces (a) a tidy comparison table with the
per-N delta and (b) an N vs macro-F1 figure overlaying both curves. It is
presentation/eval only: it reads metrics already computed on the H100, never
trains. Conventions: Polars LazyFrame I/O, matplotlib Agg, structlog, type hints,
English docstrings, Spanish prose in user-facing artifacts, no emojis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
import structlog

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

__all__ = [
    "build_dominance_comparison",
    "dominance_curves_figure",
]

#: Columns the sweep CSV is guaranteed to carry (subset required for the join).
_REQUIRED_COLS: tuple[str, ...] = ("n_classes", "macro_f1")


def _load_sweep_csv(path: Path) -> pl.DataFrame:
    """Load a parcel-sweep CSV, validating the required columns are present.

    Args:
        path: path to a CSV written by ``run_parcel_sweep`` (``--metrics-out``).

    Returns:
        The eager :class:`polars.DataFrame` sorted by ``n_classes``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a required column (``n_classes``, ``macro_f1``) is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sweep CSV not found: {path}")
    df = pl.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"sweep CSV {path} is missing required columns {missing}; got {df.columns}."
        )
    return df.sort("n_classes")


def build_dominance_comparison(
    without_filter_csv: Path,
    with_filter_csv: Path,
) -> pl.DataFrame:
    """Join the no-filter and 3:1-filter sweep curves on ``n_classes``.

    Produces one row per N present in BOTH curves with the two macro-F1 values,
    their difference, and (when available) the evaluated-parcel counts so the
    effect of the filter on dataset size is visible alongside the metric effect.

    Args:
        without_filter_csv: CSV from the sweep with ``dominance_ratio`` off
            (the default parcel-grain run).
        with_filter_csv: CSV from the sweep with ``--dominance-ratio 3.0``.

    Returns:
        A :class:`polars.DataFrame` with columns ``n_classes``,
        ``macro_f1_no_filter``, ``macro_f1_dom3``, ``delta_macro_f1``
        (``dom3 - no_filter``), and, when both CSVs carry it,
        ``n_eval_no_filter`` / ``n_eval_dom3``. Sorted by ``n_classes``.

    Raises:
        ValueError: if the join is empty (no shared ``n_classes`` between curves).
    """
    base = _load_sweep_csv(without_filter_csv)
    dom = _load_sweep_csv(with_filter_csv)

    left = base.select(
        "n_classes",
        pl.col("macro_f1").alias("macro_f1_no_filter"),
        *(
            [pl.col("n_eval_parcels").alias("n_eval_no_filter")]
            if "n_eval_parcels" in base.columns
            else []
        ),
    )
    right = dom.select(
        "n_classes",
        pl.col("macro_f1").alias("macro_f1_dom3"),
        *(
            [pl.col("n_eval_parcels").alias("n_eval_dom3")]
            if "n_eval_parcels" in dom.columns
            else []
        ),
    )
    merged = left.join(right, on="n_classes", how="inner").sort("n_classes")
    if merged.height == 0:
        raise ValueError("no shared n_classes between the two sweep curves; cannot compare.")
    merged = merged.with_columns(
        (pl.col("macro_f1_dom3") - pl.col("macro_f1_no_filter")).round(4).alias("delta_macro_f1")
    )
    logger.info(
        "dominance_comparison_built",
        n_rows=merged.height,
        n_classes=merged["n_classes"].to_list(),
        mean_delta=round(float(cast("float", merged["delta_macro_f1"].mean())), 4),
    )
    return merged


def dominance_curves_figure(
    comparison: pl.DataFrame,
    *,
    title: str = "FarSLIP por-parcela: macro-F1 vs N clases (con y sin filtro 3:1)",
) -> Figure:
    """Plot both N vs macro-F1 curves overlaid for the 3:1-filter A/B.

    Args:
        comparison: the table from :func:`build_dominance_comparison`.
        title: figure title (Spanish, user-facing).

    Returns:
        A matplotlib :class:`~matplotlib.figure.Figure` with two lines
        (no-filter vs 3:1) over the shared ``n_classes`` axis.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    n = comparison["n_classes"].to_list()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        n,
        comparison["macro_f1_no_filter"].to_list(),
        marker="o",
        label="sin filtro 3:1 (grano-parcela)",
    )
    ax.plot(
        n,
        comparison["macro_f1_dom3"].to_list(),
        marker="s",
        linestyle="--",
        label="con filtro 3:1 (Meadow por-patch)",
    )
    ax.set_xlabel("Numero de clases (N)")
    ax.set_ylabel("macro-F1 (por parcela)")
    ax.set_title(title)
    ax.set_xticks(n)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
