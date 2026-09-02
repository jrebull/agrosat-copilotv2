"""Reusable embedding-separability evaluation for per-parcel feature spaces.

Given a per-parcel embedding table (``parcel_id`` + ``class_id`` + numeric
columns), these helpers measure how well the space separates the PASTIS-R crops:

- :func:`build_balanced_eval_set` draws a stratified, per-class capped sample from
  the full parcel universe (fights the ~31x class imbalance so F1-macro is fair).
- :func:`eval_space` reports LogisticRegression F1-macro (stratified 5-fold) plus
  silhouette on a single embedding matrix.
- :func:`plot_umap_by_class` projects a space to UMAP 2D coloured by crop.
- :func:`load_alphaearth_embeddings` and :func:`combine_year_embeddings` load and
  concatenate AlphaEarth yearly tables (e.g. 2018 + 2019) to a multi-year space.
- :func:`align_spaces_on_parcels` inner-joins several embedding tables on the
  shared parcels so every space is compared on the exact same rows and labels.

The notebook ``notebooks/baseline/04_farslip_eval_pastis.ipynb`` calls these; the
logic lives here (not inline) so it is testable in ``tests/ml/`` and reusable by
the ensemble work in later avances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.utils.parcel_id import canonical_parcel_id

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SeparabilityResult:
    """Separability metrics for one embedding space.

    Attributes:
        label: Human-readable name of the space (e.g. ``"FarSLIP-s2"``).
        f1_macro_mean: Mean LogReg F1-macro across the stratified folds.
        f1_macro_std: Standard deviation of the F1-macro across folds.
        silhouette: Silhouette score of the space using the class labels.
        n_samples: Number of parcels evaluated.
        n_dims: Embedding dimensionality.
        n_classes: Number of distinct classes present.
    """

    label: str
    f1_macro_mean: float
    f1_macro_std: float
    silhouette: float
    n_samples: int
    n_dims: int
    n_classes: int


def build_balanced_eval_set(
    universe: pl.DataFrame,
    *,
    per_class_cap: int,
    min_class_samples: int,
    random_state: int = 42,
    class_names: dict[int, str] | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Draw a stratified, per-class capped evaluation set from a parcel universe.

    Classes with fewer than ``min_class_samples`` parcels are dropped so the
    downstream stratified 5-fold CV stays valid. Within each kept class the
    parcels are deterministically shuffled (seeded hash) and capped at
    ``per_class_cap`` to balance the natural imbalance.

    Args:
        universe: DataFrame with at least ``parcel_id`` and ``class_id``.
        per_class_cap: Maximum parcels kept per class.
        min_class_samples: Minimum parcels for a class to be kept.
        random_state: Seed for the deterministic per-class shuffle.
        class_names: Optional ``class_id -> name`` map; adds a ``class_name``
            column when provided.

    Returns:
        Tuple ``(balanced_df, dropped_class_ids)``. ``balanced_df`` carries
        ``parcel_id``, ``class_id`` (and ``class_name`` if ``class_names`` given).
    """
    df = canonical_parcel_id(universe.select(["parcel_id", "class_id"]))
    counts = df.group_by("class_id").len().sort("len", descending=True)
    keep = counts.filter(pl.col("len") >= min_class_samples)["class_id"].to_list()
    dropped = counts.filter(pl.col("len") < min_class_samples)["class_id"].to_list()

    balanced = (
        df.filter(pl.col("class_id").is_in(keep))
        .with_columns(pl.col("parcel_id").hash(seed=random_state).alias("__h"))
        .sort("__h")
        .group_by("class_id", maintain_order=True)
        .head(per_class_cap)
        .drop("__h")
    )
    if class_names is not None:
        balanced = balanced.with_columns(
            pl.col("class_id")
            .map_elements(lambda c: class_names.get(int(c), f"c{int(c)}"), return_dtype=pl.Utf8)
            .alias("class_name")
        )
    return balanced, dropped


def embedding_columns(df: pl.DataFrame, prefix: str) -> list[str]:
    """Return the sorted embedding column names starting with ``prefix``.

    Args:
        df: Embedding table.
        prefix: Column prefix (e.g. ``"emb_"`` for FarSLIP, ``"dim_"`` for
            AlphaEarth).

    Returns:
        Sorted list of matching column names.
    """
    return sorted(c for c in df.columns if c.startswith(prefix))


def eval_space(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    label: str,
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 2000,
    silhouette_sample_size: int = 10000,
) -> SeparabilityResult:
    """Evaluate linear separability + silhouette of one embedding space.

    Trains a LogisticRegression with stratified k-fold cross-validation and
    reports the mean/std F1-macro plus the silhouette score of the space.

    Args:
        matrix: Embedding matrix ``(n_samples, n_dims)``.
        labels: Integer class vector ``(n_samples,)``.
        label: Human-readable name for the space.
        n_splits: Stratified CV folds.
        random_state: Seed for the CV shuffle.
        max_iter: LogisticRegression max iterations.
        silhouette_sample_size: Cap on samples for the silhouette score (it is
            O(n^2); above this the score is estimated on a seeded random subset).

    Returns:
        A :class:`SeparabilityResult`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1 = cross_val_score(
        LogisticRegression(max_iter=max_iter),
        matrix,
        labels,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    # silhouette_score is O(n^2): subsample for large universes to keep it fast.
    _sil_kwargs = (
        {"sample_size": silhouette_sample_size, "random_state": random_state}
        if matrix.shape[0] > silhouette_sample_size
        else {}
    )
    sil = float(silhouette_score(matrix, labels, **_sil_kwargs))
    result = SeparabilityResult(
        label=label,
        f1_macro_mean=float(f1.mean()),
        f1_macro_std=float(f1.std()),
        silhouette=sil,
        n_samples=int(matrix.shape[0]),
        n_dims=int(matrix.shape[1]),
        n_classes=len(set(labels.tolist())),
    )
    _log.info(
        "embedding_space_evaluated",
        label=label,
        f1_macro_mean=result.f1_macro_mean,
        silhouette=result.silhouette,
        n_samples=result.n_samples,
        n_dims=result.n_dims,
    )
    return result


def plot_umap_by_class(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out_path: Path | None = None,
    random_state: int = 42,
) -> Figure:
    """Project an embedding space to UMAP 2D coloured by class and return the fig.

    Args:
        matrix: Embedding matrix ``(n_samples, n_dims)``.
        labels: Integer class vector ``(n_samples,)``.
        title: Plot title (reader-facing).
        out_path: If given, save the figure there (PNG).
        random_state: Seed forwarded to the UMAP reducer.

    Returns:
        The matplotlib ``Figure`` (caller is responsible for ``display`` /
        ``plt.close``).
    """
    import matplotlib.pyplot as plt

    from ml.features.selection import fit_umap_2d

    embedding = fit_umap_2d(matrix, random_state=random_state)
    unique_classes = sorted(set(labels.tolist()))
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_classes), 1))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    for i, cls in enumerate(unique_classes):
        mask = labels == cls
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.55,
            color=cmap(i),
            label=f"c{cls}",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    if len(unique_classes) <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def load_alphaearth_embeddings(path: str | Path, *, year_suffix: str = "") -> pl.DataFrame:
    """Load an AlphaEarth per-parcel parquet and standardize its columns.

    Keeps ``parcel_id`` + ``class_id`` (if present) + the 64 ``dim_NN`` columns,
    optionally renaming the dims with a year suffix so several years can be
    concatenated without collisions.

    Args:
        path: Path to the AlphaEarth parquet.
        year_suffix: Appended to each ``dim_NN`` name (e.g. ``"_2019"``); empty
            keeps the original names.

    Returns:
        DataFrame with ``parcel_id`` (Utf8), optional ``class_id`` and the
        (possibly suffixed) ``dim_*`` columns.
    """
    df = canonical_parcel_id(pl.read_parquet(path))
    dim_cols = embedding_columns(df, "dim_")
    keep = ["parcel_id", *(["class_id"] if "class_id" in df.columns else []), *dim_cols]
    df = df.select(keep)
    if year_suffix:
        df = df.rename({c: f"{c}{year_suffix}" for c in dim_cols})
    return df


def combine_year_embeddings(
    earlier: pl.DataFrame,
    later: pl.DataFrame,
    *,
    earlier_prefix: str = "dim_",
    later_prefix: str = "dim_",
) -> pl.DataFrame:
    """Inner-join two yearly AlphaEarth tables into one multi-year space.

    Concatenates the dim columns of both years per parcel (e.g. 64 + 64 = 128
    dims), keyed on the shared ``parcel_id``. The ``class_id`` is taken from the
    ``earlier`` table when present.

    Args:
        earlier: First-year embeddings (already suffixed to avoid name clashes).
        later: Second-year embeddings (already suffixed).
        earlier_prefix: Dim prefix in ``earlier`` (used only to keep the API
            explicit; columns are taken as-is).
        later_prefix: Dim prefix in ``later``.

    Returns:
        DataFrame with ``parcel_id``, optional ``class_id`` and the concatenated
        dim columns of both years.
    """
    del earlier_prefix, later_prefix  # Columns are pre-suffixed by the caller.
    later_dims = [c for c in later.columns if c not in ("parcel_id", "class_id")]
    return earlier.join(later.select(["parcel_id", *later_dims]), on="parcel_id", how="inner")


def align_spaces_on_parcels(
    labels_df: pl.DataFrame,
    spaces: dict[str, tuple[pl.DataFrame, str]],
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Inner-join several embedding spaces onto a shared parcel/label set.

    Every space is renamed with a per-space prefix so columns never collide, and
    only parcels present in ALL spaces survive, guaranteeing each space is later
    evaluated on the exact same rows and labels.

    Args:
        labels_df: DataFrame with ``parcel_id``, ``class_id`` (and optionally
            ``class_name``) defining the evaluation parcels.
        spaces: Mapping ``space_key -> (embedding_df, column_prefix)`` where
            ``column_prefix`` selects the numeric columns of that space.

    Returns:
        Tuple ``(merged, prefixed_cols)`` where ``merged`` holds the labels plus
        the renamed columns of every space, and ``prefixed_cols`` maps each
        space key to its list of renamed column names.
    """
    keys = ["parcel_id", "class_id"]
    if "class_name" in labels_df.columns:
        keys.append("class_name")
    merged = canonical_parcel_id(labels_df.select(keys))

    prefixed_cols: dict[str, list[str]] = {}
    for key, (emb_df, prefix) in spaces.items():
        emb_df = canonical_parcel_id(emb_df)
        cols = embedding_columns(emb_df, prefix)
        renamed = {c: f"{key}__{c}" for c in cols}
        prefixed_cols[key] = list(renamed.values())
        merged = merged.join(
            emb_df.select(["parcel_id", *cols]).rename(renamed),
            on="parcel_id",
            how="inner",
        )
    return merged, prefixed_cols


def space_matrix(merged: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract a float64 matrix for one space from the aligned table.

    Args:
        merged: Output of :func:`align_spaces_on_parcels`.
        cols: The renamed column names of the target space.

    Returns:
        ``(n_samples, len(cols))`` float64 array.
    """
    return merged.select(cols).to_numpy().astype(np.float64)


__all__ = [
    "SeparabilityResult",
    "align_spaces_on_parcels",
    "build_balanced_eval_set",
    "combine_year_embeddings",
    "embedding_columns",
    "eval_space",
    "load_alphaearth_embeddings",
    "plot_umap_by_class",
    "space_matrix",
]
