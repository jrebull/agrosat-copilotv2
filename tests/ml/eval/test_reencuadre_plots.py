"""Tests unitarios de ``ml.eval.reencuadre_plots`` (US-022b-C/D · QA-4).

Cobertura objetivo >=75% sobre los 6 plot helpers. Los tests no abren
ventanas (``MPLBACKEND=Agg`` configurado a nivel modulo) y cierran cada
figura para evitar acumular memoria entre pruebas.

Cada plot devuelve ``matplotlib.figure.Figure`` y debe ser robusto ante:
- inputs vacios -> ``ValueError`` claro.
- shape mismatch -> ``ValueError`` claro.
- ramas one-model vs multi-model (ablation_bars).
- highlight de barras debiles vs fuertes (umbral).
- fallback graceful cuando faltan cols FFT (cluster_ndvi_curves).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml.eval.feature_ablation import FeatureAblationResult
from ml.eval.reencuadre_plots import (
    plot_ablation_bars,
    plot_class_support_bars,
    plot_cluster_ndvi_curves,
    plot_confusion_matrix_heatmap,
    plot_geom_leakage_comparison,
    plot_model_comparison_bars,
    plot_model_comparison_v2_with_v1_overlay,
    plot_optional_blocks_ablation,
    plot_per_class_f1,
    plot_umap_clusters,
)


@pytest.fixture(autouse=True)
def _close_figs():
    """Cierra todas las figuras tras cada test (libera memoria)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 1. plot_ablation_bars
# ---------------------------------------------------------------------------


def _make_ablation_result(
    feature_set: str,
    model_kind: str,
    f1_macro: float,
    *,
    delta: float = 0.0,
) -> FeatureAblationResult:
    return FeatureAblationResult(
        feature_set=feature_set,
        model_kind=model_kind,  # type: ignore[arg-type]
        f1_macro=f1_macro,
        f1_weighted=f1_macro * 0.95,
        miou=f1_macro * 0.8,
        n_features=64,
        delta_vs_full=delta,
    )


def test_plot_ablation_bars_single_model_returns_figure():
    results = [
        _make_ablation_result("full", "xgb", 0.40),
        _make_ablation_result("no_geom", "xgb", 0.40, delta=0.0),
        _make_ablation_result("phenology_only", "xgb", 0.28, delta=-0.12),
    ]
    fig = plot_ablation_bars(results, baseline_value=0.32)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "f1-macro"


def test_plot_ablation_bars_multi_model_renders_grouped_bars():
    results = [
        _make_ablation_result("full", "xgb", 0.40),
        _make_ablation_result("full", "rf", 0.35),
        _make_ablation_result("no_geom", "xgb", 0.40),
        _make_ablation_result("no_geom", "rf", 0.34),
    ]
    fig = plot_ablation_bars(results, metric="f1_macro")
    ax = fig.axes[0]
    # Multi-model: legend + xticklabels = feature_sets.
    assert ax.get_legend() is not None
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert "full" in xticklabels and "no_geom" in xticklabels


def test_plot_ablation_bars_filters_nan_values_one_model():
    results = [
        _make_ablation_result("full", "xgb", float("nan")),
        _make_ablation_result("phenology_only", "xgb", float("nan")),
    ]
    fig = plot_ablation_bars(results)
    ax = fig.axes[0]
    # Todos NaN -> mensaje "Sin metricas validas".
    text_objs = [t.get_text() for t in ax.texts]
    assert any("Sin metricas validas" in t for t in text_objs)


def test_plot_ablation_bars_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_ablation_bars([])


def test_plot_ablation_bars_alternate_metric():
    results = [
        _make_ablation_result("full", "xgb", 0.40),
        _make_ablation_result("no_geom", "xgb", 0.40),
    ]
    fig = plot_ablation_bars(results, metric="miou", title="Custom title")
    assert fig.axes[0].get_title() == "Custom title"


# ---------------------------------------------------------------------------
# 2. plot_model_comparison_bars
# ---------------------------------------------------------------------------


def test_plot_model_comparison_bars_renders_three_models():
    metrics = {"xgboost": 0.41, "tempcnn": 0.18, "inceptiontime": 0.35}
    fig = plot_model_comparison_bars(metrics, baseline_value=0.32)
    ax = fig.axes[0]
    # 3 barras + xticks ordenados por valor descendente.
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels[0] == "xgboost"  # mejor F1 primero
    # baseline line presente.
    assert any(line.get_linestyle() == "--" for line in ax.lines)


def test_plot_model_comparison_bars_above_below_baseline_color_split():
    """Verifica que barras >= baseline se colorean diferente que < baseline."""
    metrics = {"good": 0.50, "bad": 0.10}
    fig = plot_model_comparison_bars(metrics, baseline_value=0.32)
    bars = [p for p in fig.axes[0].patches]
    colors = {tuple(b.get_facecolor()) for b in bars}
    # 2 colores distintos: verde good vs rojo bad.
    assert len(colors) == 2


def test_plot_model_comparison_bars_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_model_comparison_bars({})


# ---------------------------------------------------------------------------
# 3. plot_class_support_bars
# ---------------------------------------------------------------------------


def test_plot_class_support_bars_highlights_weak_classes():
    df = pl.DataFrame(
        {
            "class_id": [1, 2, 3, 4, 5],
            "len": [50000, 10000, 800, 1500, 300],
        }
    )
    fig = plot_class_support_bars(df, weak_threshold=1000)
    bars = list(fig.axes[0].patches)
    # 5 clases -> 5 barras; al menos 2 debiles (800, 300) y 3 fuertes.
    assert len(bars) == 5
    colors = {tuple(b.get_facecolor()) for b in bars}
    # Dos colores distintos por threshold split.
    assert len(colors) == 2


def test_plot_class_support_bars_accepts_custom_cols():
    df = pl.DataFrame({"clase": [1, 2], "soporte": [100, 5]})
    fig = plot_class_support_bars(df, class_col="clase", count_col="soporte", weak_threshold=50)
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# 4. plot_per_class_f1
# ---------------------------------------------------------------------------


def test_plot_per_class_f1_renders_with_class_names():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=120)
    y_pred = y_true.copy()
    # Introduce algun error en clase 2 para que F1 < 1.0 ahi.
    y_pred[y_true == 2] = rng.integers(0, 3, size=int((y_true == 2).sum()))
    fig = plot_per_class_f1(
        y_true,
        y_pred,
        class_names={0: "maiz", 1: "trigo", 2: "soja"},
        weak_threshold=0.50,
    )
    ax = fig.axes[0]
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "maiz" in ytick_labels and "trigo" in ytick_labels


def test_plot_per_class_f1_empty_arrays_raise():
    with pytest.raises(ValueError, match="empty"):
        plot_per_class_f1(np.array([], dtype=int), np.array([], dtype=int))


def test_plot_per_class_f1_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        plot_per_class_f1(np.array([0, 1, 2]), np.array([0, 1]))


def test_plot_per_class_f1_with_explicit_class_labels():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    fig = plot_per_class_f1(y_true, y_pred, class_labels=[0, 1, 2])
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# 5. plot_umap_clusters
# ---------------------------------------------------------------------------


def test_plot_umap_clusters_renders_scatter():
    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(80, 2))
    labels = rng.integers(0, 4, size=80)
    fig = plot_umap_clusters(embedding, labels)
    ax = fig.axes[0]
    # 4 clusters -> 4 scatters (cada cluster). Legend con ncol=2.
    assert ax.get_legend() is not None
    assert ax.get_xlabel() == "UMAP 1"


def test_plot_umap_clusters_invalid_embedding_shape_raises():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        plot_umap_clusters(np.zeros((10, 3)), np.zeros(10, dtype=int))


def test_plot_umap_clusters_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        plot_umap_clusters(np.zeros((10, 2)), np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# 6. plot_cluster_ndvi_curves
# ---------------------------------------------------------------------------


def test_plot_cluster_ndvi_curves_renders_curve_per_cluster():
    rng = np.random.default_rng(0)
    n = 50
    cols: dict[str, list[float]] = {}
    for k in range(4):
        cols[f"NDVI_fft_amp_{k}"] = rng.normal(size=n).tolist()
        cols[f"NDVI_fft_phase_{k}"] = rng.uniform(-np.pi, np.pi, size=n).tolist()
    df = pl.DataFrame(cols)
    labels = rng.integers(0, 3, size=n)
    fig = plot_cluster_ndvi_curves(df, labels, sequence_length=24)
    ax = fig.axes[0]
    # 3 clusters -> al menos 3 curvas plot.
    assert len(ax.lines) >= 3


def test_plot_cluster_ndvi_curves_fallback_when_no_fft_cols():
    df = pl.DataFrame({"col_a": [1.0, 2.0, 3.0], "col_b": [4.0, 5.0, 6.0]})
    labels = np.array([0, 1, 0])
    fig = plot_cluster_ndvi_curves(df, labels)
    # Fallback con mensaje en lugar de curvas.
    text_objs = [t.get_text() for t in fig.axes[0].texts]
    assert any("No hay columnas FFT NDVI" in t for t in text_objs)


def test_plot_cluster_ndvi_curves_shape_mismatch_raises():
    df = pl.DataFrame({"NDVI_fft_amp_0": [1.0, 2.0], "NDVI_fft_phase_0": [0.1, 0.2]})
    with pytest.raises(ValueError, match="cluster_labels"):
        plot_cluster_ndvi_curves(df, np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# 7. plot_geom_leakage_comparison
# ---------------------------------------------------------------------------


def test_plot_geom_leakage_comparison_renders_three_bars():
    results = [
        _make_ablation_result("full", "xgb", 0.42),
        _make_ablation_result("no_geom", "xgb", 0.42, delta=0.0),
        _make_ablation_result("geom_only", "xgb", 0.05, delta=-0.37),
    ]
    fig = plot_geom_leakage_comparison(results)
    assert isinstance(fig, matplotlib.figure.Figure)
    bars = [p for p in fig.axes[0].patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert len(bars) == 3


def test_plot_geom_leakage_comparison_with_missing_sets_shows_placeholder():
    results = [_make_ablation_result("phenology_only", "xgb", 0.30)]
    fig = plot_geom_leakage_comparison(results)
    text_objs = [t.get_text() for t in fig.axes[0].texts]
    assert any("Sin resultados validos" in t for t in text_objs)


# ---------------------------------------------------------------------------
# 8. plot_optional_blocks_ablation
# ---------------------------------------------------------------------------


def test_plot_optional_blocks_ablation_renders_deltas():
    results = [
        _make_ablation_result("full", "xgb", 0.40, delta=float("nan")),
        _make_ablation_result("with_farslip", "xgb", 0.43, delta=0.03),
        _make_ablation_result("with_pheno_text", "xgb", 0.39, delta=-0.01),
        _make_ablation_result("farslip_only", "xgb", 0.20, delta=-0.20),
    ]
    fig = plot_optional_blocks_ablation(results)
    assert isinstance(fig, matplotlib.figure.Figure)
    bars = [p for p in fig.axes[0].patches if isinstance(p, matplotlib.patches.Rectangle)]
    # 3 sets opcionales (full no cuenta, NaN no cuenta)
    assert len(bars) == 3


def test_plot_optional_blocks_ablation_empty_shows_placeholder():
    results = [_make_ablation_result("full", "xgb", 0.40)]
    fig = plot_optional_blocks_ablation(results)
    text_objs = [t.get_text() for t in fig.axes[0].texts]
    assert any("No hay bloques opcionales" in t for t in text_objs)


# ---------------------------------------------------------------------------
# 9. plot_confusion_matrix_heatmap
# ---------------------------------------------------------------------------


def test_plot_confusion_matrix_heatmap_renders_with_labels():
    y_true = np.array([1, 2, 1, 3, 2, 1, 2, 3])
    y_pred = np.array([1, 2, 2, 3, 2, 1, 1, 3])
    class_names = {1: "Trigo", 2: "Maiz", 3: "Vinedo"}
    fig = plot_confusion_matrix_heatmap(
        y_true, y_pred, class_labels=[1, 2, 3], class_names=class_names
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert "Trigo" in xticklabels


def test_plot_confusion_matrix_heatmap_empty_raises():
    with pytest.raises(ValueError):
        plot_confusion_matrix_heatmap(np.array([]), np.array([]))


def test_plot_confusion_matrix_heatmap_normalize_none():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 2])
    fig = plot_confusion_matrix_heatmap(y_true, y_pred, normalize="none")
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# 10. plot_model_comparison_v2_with_v1_overlay
# ---------------------------------------------------------------------------


def test_plot_model_comparison_v2_with_v1_overlay_renders_both():
    v2 = {"xgboost": 0.45, "lgbm": 0.44, "tempcnn": 0.20}
    v1 = {"xgboost": 0.41, "tempcnn": 0.14}
    fig = plot_model_comparison_v2_with_v1_overlay(v2, v1_metrics=v1)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Buscamos al menos las 3 barras v2 + 2 barras v1.
    bars = [p for p in fig.axes[0].patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert len(bars) >= 5


def test_plot_model_comparison_v2_without_v1_overlay():
    v2 = {"xgboost": 0.45, "lgbm": 0.44}
    fig = plot_model_comparison_v2_with_v1_overlay(v2)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_model_comparison_v2_empty_raises():
    with pytest.raises(ValueError):
        plot_model_comparison_v2_with_v1_overlay({})
