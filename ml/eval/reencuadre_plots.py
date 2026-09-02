"""Reusable plots for notebook 05 of the phenology reframing.

Visualization functions for the US-022b-C/D analyses. Each function returns a
``matplotlib.figure.Figure`` so the notebook decides whether to show it with
``display(fig)`` and/or persist it with ``fig.savefig(...)``.

Canonical pattern (consistent with :mod:`ml.eval.learning_curves` and
:mod:`ml.eval.interpretability`):

- Accepts typed inputs (``FeatureAblationResult`` / ``TemporalModelResult`` /
  ndarrays / Polars DataFrames), never paths.
- Returns the figure, never persists nor closes it.
- No ``plt.show()`` nor global side-effects (the matplotlib backend is
  configured by the caller).

Provided plots:

- :func:`plot_ablation_bars` — F1-macro per feature set.
- :func:`plot_model_comparison_bars` — F1-macro of several models vs baseline.
- :func:`plot_class_support_bars` — class distribution with threshold.
- :func:`plot_per_class_f1` — per-class F1 of the best model (highlight weak classes).
- :func:`plot_umap_clusters` — UMAP 2D colored by KMeans cluster.
- :func:`plot_cluster_ndvi_curves` — mean NDVI curve per cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ml.eval.feature_ablation import FeatureAblationResult


__all__ = [
    "plot_ablation_bars",
    "plot_class_support_bars",
    "plot_cluster_ndvi_curves",
    "plot_confusion_matrix_heatmap",
    "plot_geom_leakage_comparison",
    "plot_model_comparison_bars",
    "plot_model_comparison_v2_with_v1_overlay",
    "plot_optional_blocks_ablation",
    "plot_per_class_f1",
    "plot_umap_clusters",
]


# ---------------------------------------------------------------------------
# 1. Feature ablation (F1-macro per set, same model).
# ---------------------------------------------------------------------------


def plot_ablation_bars(
    results: Sequence[FeatureAblationResult],
    *,
    metric: str = "f1_macro",
    title: str | None = None,
    baseline_value: float | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Horizontal bar plot of F1-macro per feature set.

    Each bar is a ``(feature_set, model)`` pair. If there are several models in
    ``results``, the per-model groups are separated with distinct colors.

    Args:
        results: Results of :func:`run_feature_ablation`.
        metric: ``"f1_macro"``, ``"f1_weighted"`` or ``"miou"``.
        title: Optional title; if ``None`` it is generated automatically.
        baseline_value: Vertical reference line (e.g. F1-macro of the
            closed tabular baseline). ``None`` omits it.
        figsize: Tuple ``(width, height)`` in inches.

    Returns:
        Matplotlib figure ready for ``display(fig)`` or ``fig.savefig(...)``.
    """
    if not results:
        raise ValueError("`results` is empty.")
    if metric not in {"f1_macro", "f1_weighted", "miou"}:
        raise ValueError(f"metric={metric!r} not supported.")

    # Group by model keeping the stable order of appearance.
    by_model: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        by_model.setdefault(r.model_kind, []).append((r.feature_set, float(getattr(r, metric))))

    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    if len(by_model) == 1:
        # A single model: simple horizontal bars.
        model_kind, items = next(iter(by_model.items()))
        # Filter NaN (trainings that failed due to null coverage of the set).
        items_clean = [(fs, v) for fs, v in items if v == v]
        items_sorted = sorted(items_clean, key=lambda kv: kv[1], reverse=False)
        labels = [s for s, _ in items_sorted]
        values = [v for _, v in items_sorted]
        if not labels:
            ax.text(
                0.5,
                0.5,
                "Sin metricas validas para graficar.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#888",
            )
            ax.set_axis_off()
            fig.tight_layout()
            return fig
        bars = ax.barh(labels, values, color="#4C72B0", edgecolor="white")
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_xlabel(metric.replace("_", "-"))
        ax.set_title(title or f"{metric.replace('_', '-')} por conjunto de features ({model_kind})")
    else:
        # Several models: bars grouped vertically.
        feature_sets: list[str] = []
        for items in by_model.values():
            for fs, _ in items:
                if fs not in feature_sets:
                    feature_sets.append(fs)
        n_models = len(by_model)
        width = 0.8 / n_models
        x_positions = np.arange(len(feature_sets))
        palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
        for idx, (model_kind, items) in enumerate(by_model.items()):
            lookup = dict(items)
            values = [lookup.get(fs, np.nan) for fs in feature_sets]
            offsets = x_positions + (idx - (n_models - 1) / 2) * width
            ax.bar(
                offsets,
                values,
                width=width,
                label=model_kind,
                color=palette[idx % len(palette)],
                edgecolor="white",
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_sets, rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", "-"))
        ax.set_title(title or f"{metric.replace('_', '-')} por conjunto x modelo")
        ax.legend(loc="best", frameon=False)

    if baseline_value is not None:
        ax.axvline(baseline_value, color="#888", linestyle="--", linewidth=1)
        ax.text(
            baseline_value,
            ax.get_ylim()[1] * 0.95 if len(by_model) == 1 else 0,
            f"  baseline {baseline_value:.3f}",
            color="#666",
            fontsize=8,
            rotation=90 if len(by_model) > 1 else 0,
            va="top",
            ha="left",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Model comparison (XGBoost vs TempCNN vs InceptionTime over the
#    same feature set).
# ---------------------------------------------------------------------------


def plot_model_comparison_bars(
    metric_by_model: Mapping[str, float],
    *,
    baseline_label: str = "baseline 0.32",
    baseline_value: float = 0.32,
    title: str = "Comparativa de modelos sobre el conjunto ganador",
    metric_name: str = "F1-macro",
    figsize: tuple[float, float] = (6.5, 4.0),
) -> matplotlib.figure.Figure:
    """Vertical bar plot comparing several models against a baseline line.

    Args:
        metric_by_model: Mapping ``{model_name: F1_macro}``. Typical:
            ``{"xgboost": 0.34, "tempcnn": 0.41, "inceptiontime": 0.39}``.
        baseline_label: Label of the baseline line in the legend.
        baseline_value: Reference value (horizontal line).
        title: Plot title.
        metric_name: Readable name of the metric.
        figsize: Tuple ``(width, height)``.

    Returns:
        Matplotlib figure.
    """
    if not metric_by_model:
        raise ValueError("`metric_by_model` is empty.")

    items = list(metric_by_model.items())
    items_sorted = sorted(items, key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items_sorted]
    values = [v for _, v in items_sorted]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    colors = ["#55A868" if v >= baseline_value else "#C44E52" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(
        baseline_value,
        color="#444",
        linestyle="--",
        linewidth=1.2,
        label=baseline_label,
    )
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values), baseline_value) * 1.20)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Per-class support (imbalance ~31x).
# ---------------------------------------------------------------------------


def plot_class_support_bars(
    class_counts: pl.DataFrame,
    *,
    class_col: str = "class_id",
    count_col: str = "len",
    weak_threshold: int = 1000,
    title: str = "Numero de parcelas por clase (resaltadas clases con soporte debil)",
    figsize: tuple[float, float] = (8.0, 4.5),
) -> matplotlib.figure.Figure:
    """Horizontal bar plot with per-class support.

    Classes with support ``< weak_threshold`` are colored differently to
    highlight the imbalance.

    Args:
        class_counts: DataFrame with columns ``class_col`` and ``count_col``,
            typical output of ``df.group_by("class_id").len()``.
        class_col: Name of the column with the class id.
        count_col: Name of the column with the count.
        weak_threshold: Threshold below which the bar is marked as weak.
        title: Title.
        figsize: Tuple ``(width, height)``.

    Returns:
        Matplotlib figure.
    """
    if class_col not in class_counts.columns or count_col not in class_counts.columns:
        raise ValueError(f"`class_counts` must contain `{class_col}` and `{count_col}`.")

    ordered = class_counts.sort(count_col, descending=False)
    labels = [str(v) for v in ordered[class_col].to_list()]
    values = ordered[count_col].to_list()
    colors = ["#C44E52" if v < weak_threshold else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.barh(labels, values, color=colors, edgecolor="white")
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:,}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Numero de parcelas")
    ax.set_ylabel(class_col)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.axvline(weak_threshold, color="#888", linestyle="--", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Per-class F1 of the best model (highlight weak classes).
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_labels: Sequence[int] | None = None,
    class_names: Mapping[int, str] | None = None,
    title: str = "F1 por clase",
    weak_threshold: float = 0.10,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Horizontal bar plot of per-class F1 with weak-class highlighting.

    Args:
        y_true: True labels (1D).
        y_pred: Predicted labels (1D).
        class_labels: List of class ids to report (plot order).
            ``None`` infers from the union of ``y_true`` and ``y_pred``.
        class_names: Optional mapping ``{class_id: readable name}``.
        title: Plot title.
        weak_threshold: F1 below which the bar is colored as weak.
        figsize: Tuple ``(width, height)``.

    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import f1_score

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("`y_true` and `y_pred` cannot be empty.")
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"Shape mismatch: y_true.shape={y_true_arr.shape} vs y_pred.shape={y_pred_arr.shape}."
        )

    if class_labels is None:
        class_labels = sorted({*y_true_arr.tolist(), *y_pred_arr.tolist()})
    labels_list = list(class_labels)
    per_class = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels_list,
        average=None,
        zero_division=0,
    )
    ordered = sorted(zip(labels_list, per_class, strict=True), key=lambda kv: kv[1])
    y_labels = [(class_names.get(cid, str(cid)) if class_names else str(cid)) for cid, _ in ordered]
    values = [float(v) for _, v in ordered]
    colors = ["#C44E52" if v < weak_threshold else "#55A868" for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.barh(y_labels, values, color=colors, edgecolor="white")
    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", ha="left", fontsize=8)
    ax.axvline(weak_threshold, color="#888", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1")
    ax.set_ylabel("clase")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. 2D UMAP colored by KMeans cluster (not by class_id, to validate
#    structure without coordinates).
# ---------------------------------------------------------------------------


def plot_umap_clusters(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    title: str = "UMAP de la firma fenologica sin coordenadas, coloreado por cluster",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """UMAP 2D scatter colored by KMeans cluster.

    Args:
        embedding: Array ``(N, 2)`` with the UMAP projection.
        cluster_labels: Array ``(N,)`` with the KMeans assignment.
        title: Title.
        figsize: Tuple ``(width, height)``.

    Returns:
        Matplotlib figure.
    """
    emb = np.asarray(embedding)
    labels = np.asarray(cluster_labels).ravel()
    if emb.ndim != 2 or emb.shape[1] != 2:
        raise ValueError(f"`embedding` must be (N, 2), got {emb.shape}.")
    if emb.shape[0] != labels.shape[0]:
        raise ValueError(f"Shape mismatch: embedding={emb.shape}, labels={labels.shape}.")

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    n_clusters = int(labels.max()) + 1 if labels.size > 0 else 1
    palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.6,
            color=palette[cid % len(palette)],
            label=f"cluster {cid}",
            edgecolors="none",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Mean synthetic NDVI curve per cluster (agronomic interpretability).
# ---------------------------------------------------------------------------


def plot_cluster_ndvi_curves(
    df: pl.DataFrame,
    cluster_labels: np.ndarray,
    *,
    fft_cols: Sequence[str] | None = None,
    sequence_length: int = 72,
    title: str = "Curva NDVI media reconstruida por cluster",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Reconstruct the mean NDVI curve per cluster from the FFT cols.

    For each KMeans cluster, averages the NDVI FFT coefficients of the
    cluster's parcels, reconstructs the daily series by partial IDFT and
    plots the curve over the day of the year.

    Args:
        df: Polars DataFrame with the NDVI FFT columns present.
        cluster_labels: Array ``(N,)`` with the KMeans assignment (same order
            as ``df``).
        fft_cols: List of FFT columns to reconstruct; ``None`` auto-detects
            ``NDVI_fft_amp_k`` and ``NDVI_fft_phase_k``.
        sequence_length: Reconstructed temporal length.
        title: Title.
        figsize: Tuple ``(width, height)``.

    Returns:
        Matplotlib figure.
    """
    if df.height != cluster_labels.shape[0]:
        raise ValueError(
            f"`df.height`={df.height} must equal `cluster_labels`={cluster_labels.shape[0]}."
        )

    if fft_cols is None:
        cols = [
            c for c in df.columns if c.startswith("NDVI_fft_amp") or c.startswith("NDVI_fft_phase")
        ]
        fft_cols = tuple(sorted(cols))

    amp_cols = [c for c in fft_cols if "_fft_amp_" in c]
    phase_cols = [c for c in fft_cols if "_fft_phase_" in c]

    if not amp_cols or not phase_cols:
        # Fallback: without FFT, empty plot with a message (the notebook will keep
        # running without breaking in CI with a reduced dataset).
        fig, ax = plt.subplots(figsize=figsize, dpi=110)
        ax.text(
            0.5,
            0.5,
            "No hay columnas FFT NDVI en el DataFrame.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig

    n_harmonics = min(len(amp_cols), len(phase_cols))
    amp_cols_sorted = sorted(amp_cols, key=lambda c: int(c.rsplit("_", 1)[-1]))[:n_harmonics]
    phase_cols_sorted = sorted(phase_cols, key=lambda c: int(c.rsplit("_", 1)[-1]))[:n_harmonics]

    amps = df.select(amp_cols_sorted).to_numpy().astype(np.float64)
    phases = df.select(phase_cols_sorted).to_numpy().astype(np.float64)
    amps = np.where(np.isfinite(amps), amps, 0.0)
    phases = np.where(np.isfinite(phases), phases, 0.0)

    t = np.linspace(0.0, 1.0, sequence_length, endpoint=False)
    # Reconstruct: y(t) = sum_k amp_k * cos(2*pi*k*t + phase_k); k = 1..K.
    k_indices = np.arange(1, n_harmonics + 1).reshape(1, -1)
    # series shape (N, T)
    series = np.zeros((df.height, sequence_length), dtype=np.float64)
    for ti, tv in enumerate(t):
        arg = 2 * np.pi * k_indices * tv + phases
        series[:, ti] = (amps * np.cos(arg)).sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size > 0 else 1
    doy = t * 365.0
    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if not np.any(mask):
            continue
        mean_curve = series[mask].mean(axis=0)
        ax.plot(
            doy,
            mean_curve,
            color=palette[cid % len(palette)],
            label=f"cluster {cid} (n={int(mask.sum())})",
            linewidth=1.5,
        )
    ax.set_xlabel("Dia del anio (DOY)")
    ax.set_ylabel("NDVI reconstruido (FFT)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Geographic leakage: explicit comparison full vs no_geom + geom_only.
# ---------------------------------------------------------------------------


def plot_geom_leakage_comparison(
    results: Sequence[FeatureAblationResult],
    *,
    title: str = "Aporte de las columnas geom_*: leakage espacial vs. señal real",
    figsize: tuple[float, float] = (7.0, 4.0),
) -> matplotlib.figure.Figure:
    """Bar plot with 3 bars `full`, `no_geom`, `geom_only` side by side.

    Isolates the effect of the `geom_*` columns (area, perimeter, elongation).
    If `geom_only` has high F1-macro, there is spatial leakage. If `full -
    no_geom` is ~0 and `geom_only` is ~0, the cols add nothing and can be
    discarded.

    Args:
        results: Results of :func:`run_feature_ablation`. Must contain
            the keys `full`, `no_geom`, `geom_only` (or a subset; the
            missing ones are omitted with an annotation in the plot).
        title: Plot title.
        figsize: Tuple (width, height).

    Returns:
        Matplotlib figure.
    """
    target_sets = ("full", "no_geom", "geom_only")
    by_set = {r.feature_set: float(r.f1_macro) for r in results if r.feature_set in target_sets}

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    labels = [s for s in target_sets if s in by_set]
    values = [by_set[s] for s in labels]
    palette = {"full": "#4C72B0", "no_geom": "#55A868", "geom_only": "#C44E52"}
    colors = [palette[s] for s in labels]

    if not labels:
        ax.text(
            0.5,
            0.5,
            "Sin resultados validos para `full`, `no_geom` ni `geom_only`.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888",
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if "full" in by_set and "no_geom" in by_set:
        delta = by_set["no_geom"] - by_set["full"]
        ax.text(
            0.02,
            0.98,
            f"delta(no_geom - full) = {delta:+.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#444",
        )

    ax.set_ylabel("F1-macro")
    ax.set_title(title)
    ax.set_ylim(0, max(0.1, max(values) * 1.20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Optional blocks: with_farslip / with_pheno_text / with_spectral_signature.
# ---------------------------------------------------------------------------


def plot_optional_blocks_ablation(
    results: Sequence[FeatureAblationResult],
    *,
    baseline_set: str = "full",
    title: str = "Aporte de los bloques opcionales sobre el baseline `full`",
    figsize: tuple[float, float] = (8.5, 4.5),
) -> matplotlib.figure.Figure:
    """Plot the deltas of optional blocks against a baseline.

    For each `(set, model)` pair computes the delta vs `baseline_set` (default
    `full`) and plots it as horizontal bars colored by sign
    (green = improves, red = worsens, gray = neutral |delta| < 0.005).

    Args:
        results: Results of :func:`run_feature_ablation`.
        baseline_set: Reference set.
        title: Plot title.
        figsize: Tuple (width, height).

    Returns:
        Matplotlib figure.
    """
    by_set: dict[str, float] = {}
    for r in results:
        if r.feature_set == baseline_set:
            continue
        if not r.feature_set.startswith("with_") and not r.feature_set.endswith("_only"):
            continue
        if r.f1_macro != r.f1_macro:  # NaN check
            continue
        by_set.setdefault(r.feature_set, float(r.delta_vs_full))

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    if not by_set:
        ax.text(
            0.5,
            0.5,
            "No hay bloques opcionales con resultados validos.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888",
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    items_sorted = sorted(by_set.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items_sorted]
    deltas = [v for _, v in items_sorted]

    def _color(delta: float) -> str:
        if abs(delta) < 0.005:
            return "#888"
        return "#55A868" if delta > 0 else "#C44E52"

    colors = [_color(d) for d in deltas]
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white")
    for bar, value in zip(bars, deltas, strict=True):
        ax.text(
            value + (0.001 if value >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    ax.axvline(0.0, color="#444", linewidth=1)
    ax.set_xlabel(f"delta F1-macro vs `{baseline_set}`")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Confusion matrix (heatmap).
# ---------------------------------------------------------------------------


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_labels: Sequence[int] | None = None,
    class_names: Mapping[int, str] | None = None,
    title: str = "Matriz de confusion (out-of-fold)",
    normalize: Literal["true", "pred", "all", "none"] = "true",
    figsize: tuple[float, float] = (8.5, 7.0),
    cmap: str = "Blues",
) -> matplotlib.figure.Figure:
    """Confusion matrix heatmap with per-cell annotations.

    Args:
        y_true: True labels (1D).
        y_pred: Predicted labels (1D).
        class_labels: List of class_ids to report.
        class_names: Mapping {class_id: readable name}.
        title: Title.
        normalize: `"true"` normalizes by row (per-class recall), `"pred"`
            by column (per-class precision), `"all"` by total, `"none"`
            without normalizing.
        figsize: Tuple (width, height).
        cmap: Matplotlib colormap.

    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import confusion_matrix

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    if y_true_arr.size == 0:
        raise ValueError("`y_true` cannot be empty.")
    if class_labels is None:
        class_labels = sorted({*y_true_arr.tolist(), *y_pred_arr.tolist()})
    labels_list = list(class_labels)

    cm_norm: str | None = None if normalize == "none" else normalize
    cm = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=labels_list,
        normalize=cm_norm,  # type: ignore[arg-type]
    )

    names = [(class_names.get(cid, str(cid)) if class_names else str(cid)) for cid in labels_list]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)

    thresh = float(cm.max()) / 2.0 if cm.size else 0.0
    fmt = ".2f" if normalize != "none" else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Model comparison v2 (XGB + LGBM + RF + TempCNN + InceptionTime) with
#     optional overlay of the v1 baseline (US-022).
# ---------------------------------------------------------------------------


def plot_model_comparison_v2_with_v1_overlay(
    v2_metrics: Mapping[str, float],
    *,
    v1_metrics: Mapping[str, float] | None = None,
    metric_name: str = "F1-macro",
    title: str = "Comparativa modelos baseline v2 (con overlay v1 US-022)",
    figsize: tuple[float, float] = (8.0, 4.5),
) -> matplotlib.figure.Figure:
    """Grouped v2 bars with overlay of v1 bars when available.

    Args:
        v2_metrics: Mapping {model: metric} of the v2 run.
        v1_metrics: Mapping {model: metric} of the v1 run (US-022).
            Only the models present in both dictionaries are overlaid.
        metric_name: Readable name (default `"F1-macro"`).
        title: Title.
        figsize: Tuple (width, height).

    Returns:
        Matplotlib figure.
    """
    if not v2_metrics:
        raise ValueError("`v2_metrics` is empty.")

    models = list(v2_metrics)
    values_v2 = [float(v2_metrics[m]) for m in models]
    values_v1 = [
        float(v1_metrics[m]) if v1_metrics is not None and m in v1_metrics else None for m in models
    ]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    x_positions = np.arange(len(models))
    width = 0.38

    bars_v2 = ax.bar(
        x_positions - width / 2,
        values_v2,
        width=width,
        label="v2 (US-023-preview)",
        color="#4C72B0",
        edgecolor="white",
    )
    for bar, value in zip(bars_v2, values_v2, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if any(v is not None for v in values_v1):
        v1_for_plot = [v if v is not None else 0.0 for v in values_v1]
        bars_v1 = ax.bar(
            x_positions + width / 2,
            v1_for_plot,
            width=width,
            label="v1 (US-022)",
            color="#C44E52",
            edgecolor="white",
            alpha=0.7,
        )
        for bar, value, original in zip(bars_v1, v1_for_plot, values_v1, strict=True):
            if original is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
