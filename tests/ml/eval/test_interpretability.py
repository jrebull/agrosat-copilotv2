"""Tests del modulo ``ml.eval.interpretability`` (US-020, EPIC 4).

Cubre los criterios de aceptacion AC-1..AC-8 en cinco grupos:

- Grupo A: importancia nativa RF/XGB (AC-1).
- Grupo B: SHAP multiclase y subsampling (AC-2).
- Grupo C: plots SHAP — summary, dependence, waterfall (AC-3, AC-6, AC-7).
- Grupo D: clasificacion y dominancia AlphaEarth (AC-4).
- Grupo E: estructura del notebook ``04_baseline.ipynb`` §3-§5 (AC-6).

Los modelos son sinteticos y autocontenidos (``make_trained_tree_model``) — no
dependen de los artefactos joblib de US-019 ni del parquet de US-018.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from ml.eval.interpretability import (
    ShapResult,
    _normalize_shap_multiclass,
    _to_numpy_sample,
    alphaearth_dominance_table,
    compute_shap_values,
    feature_importance_table,
    is_alphaearth_dim,
    shap_dependence_plots,
    shap_summary_plot,
    shap_waterfall_plot,
)
from tests.ml.eval.fixtures.interpretability_synthetic import (
    make_trained_tree_model,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NOTEBOOK_PATH = _REPO_ROOT / "notebooks" / "04_baseline.ipynb"


# ---------------------------------------------------------------------------
# Fixtures de modulo.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rf_model():
    """Random Forest sintetico de 5 clases ya ajustado."""
    return make_trained_tree_model("rf", n_classes=5, n_features=12)


@pytest.fixture(scope="module")
def xgb_model():
    """XGBoost sintetico de 5 clases ya ajustado."""
    return make_trained_tree_model("xgb", n_classes=5, n_features=12)


@pytest.fixture(scope="module")
def rf_shap(rf_model):
    """Resultado SHAP del Random Forest sintetico (subsample 80)."""
    return compute_shap_values(
        rf_model.model,
        rf_model.X,
        "rf",
        feature_cols=rf_model.feature_cols,
        sample_size=80,
    )


@pytest.fixture(scope="module")
def xgb_shap(xgb_model):
    """Resultado SHAP del XGBoost sintetico (subsample 80)."""
    return compute_shap_values(
        xgb_model.model,
        xgb_model.X,
        "xgb",
        feature_cols=xgb_model.feature_cols,
        sample_size=80,
    )


# ===========================================================================
# Grupo A — importancia nativa (AC-1).
# ===========================================================================


def test_feature_importance_rf_returns_ranked_df(rf_model):
    """RF: ``feature_importance_table`` devuelve ``(feature, importance, rank)``."""
    table = feature_importance_table(rf_model.model, "rf", rf_model.feature_cols)
    assert isinstance(table, pl.DataFrame)
    assert table.columns == ["feature", "importance", "rank"]
    assert table.height == len(rf_model.feature_cols)
    assert set(table["feature"].to_list()) == set(rf_model.feature_cols)
    assert table["rank"].to_list() == list(range(1, table.height + 1))


def test_feature_importance_xgb_uses_gain(xgb_model):
    """XGB: la importancia *gain* es no negativa y alineada a feature_cols."""
    table = feature_importance_table(xgb_model.model, "xgb", xgb_model.feature_cols)
    assert table.height == len(xgb_model.feature_cols)
    assert (table["importance"] >= 0.0).all()
    # Al menos un feature debe tener gain positivo (el modelo aprendio senal).
    assert table["importance"].max() > 0.0


def test_importance_table_sorted_descending(rf_model):
    """La tabla de importancia queda ordenada de mayor a menor."""
    table = feature_importance_table(rf_model.model, "rf", rf_model.feature_cols)
    importances = table["importance"].to_list()
    assert importances == sorted(importances, reverse=True)


def test_feature_importance_rejects_bad_model_kind(rf_model):
    """Un ``model_kind`` invalido lanza ``ValueError``."""
    with pytest.raises(ValueError, match="model_kind"):
        feature_importance_table(rf_model.model, "lgbm", rf_model.feature_cols)  # type: ignore[arg-type]


def test_feature_importance_rejects_mismatched_feature_cols(rf_model):
    """Un ``feature_cols`` de largo distinto al del modelo RF lanza error."""
    with pytest.raises(ValueError, match="features"):
        feature_importance_table(rf_model.model, "rf", rf_model.feature_cols[:-1])


# ===========================================================================
# Grupo B — SHAP multiclase (AC-2).
# ===========================================================================


def test_compute_shap_returns_shap_result(rf_shap):
    """``compute_shap_values`` devuelve un ``ShapResult`` bien formado."""
    assert isinstance(rf_shap, ShapResult)
    assert rf_shap.model_kind == "rf"
    assert isinstance(rf_shap.values, np.ndarray)
    assert isinstance(rf_shap.global_importance, pl.DataFrame)
    assert rf_shap.global_importance.columns == [
        "feature",
        "mean_abs_shap",
        "rank",
    ]


def test_shap_values_shape_matches_sample(rf_model, rf_shap):
    """El tensor SHAP tiene ejes ``(n_samples, n_features, n_classes)``."""
    n_samples, n_features, n_classes = rf_shap.values.shape
    assert n_samples == 80
    assert n_features == len(rf_model.feature_cols)
    assert n_classes == rf_model.n_classes


def test_shap_handles_multiclass_output():
    """``_normalize_shap_multiclass`` normaliza lista, 2D y 3D al mismo tensor."""
    n_samples, n_features, n_classes = 6, 4, 3
    rng = np.random.default_rng(0)
    per_class = [rng.standard_normal((n_samples, n_features)) for _ in range(n_classes)]

    # Forma 1: lista por clase.
    from_list = _normalize_shap_multiclass(per_class, n_samples=n_samples, n_features=n_features)
    assert from_list.shape == (n_samples, n_features, n_classes)

    # Forma 2: array 3D ya canonico.
    array_3d = np.stack(per_class, axis=-1)
    from_3d = _normalize_shap_multiclass(array_3d, n_samples=n_samples, n_features=n_features)
    np.testing.assert_allclose(from_list, from_3d)

    # Forma 3: array 2D (binario/regresion) -> se expande a 3 ejes.
    from_2d = _normalize_shap_multiclass(per_class[0], n_samples=n_samples, n_features=n_features)
    assert from_2d.shape == (n_samples, n_features, 1)


def test_shap_normalize_handles_class_first_layout():
    """Un tensor ``(n_classes, n_samples, n_features)`` se reordena al canonico."""
    n_samples, n_features, n_classes = 5, 4, 3
    rng = np.random.default_rng(1)
    class_first = rng.standard_normal((n_classes, n_samples, n_features))
    out = _normalize_shap_multiclass(class_first, n_samples=n_samples, n_features=n_features)
    assert out.shape == (n_samples, n_features, n_classes)


def test_shap_normalize_rejects_unknown_shape():
    """Una forma de SHAP no reconocida (4D) lanza ``ValueError``."""
    with pytest.raises(ValueError, match="Unrecognized SHAP shape"):
        _normalize_shap_multiclass(np.zeros((2, 2, 2, 2)), n_samples=2, n_features=2)


def test_shap_global_importance_is_mean_abs(rf_shap):
    """La importancia global SHAP es la media de ``|SHAP|`` sobre ejes 0 y 2."""
    expected = np.abs(rf_shap.values).mean(axis=(0, 2))
    table = rf_shap.global_importance
    by_feature = dict(
        zip(
            table["feature"].to_list(),
            table["mean_abs_shap"].to_list(),
            strict=True,
        )
    )
    for idx, name in enumerate(rf_shap.feature_cols):
        assert by_feature[name] == pytest.approx(float(expected[idx]), rel=1e-9)
    # Ordenada descendentemente.
    values = table["mean_abs_shap"].to_list()
    assert values == sorted(values, reverse=True)


def test_compute_shap_subsamples_large_input(rf_model):
    """``sample_size`` recorta filas; ``_to_numpy_sample`` respeta el limite."""
    result = compute_shap_values(
        rf_model.model,
        rf_model.X,
        "rf",
        feature_cols=rf_model.feature_cols,
        sample_size=30,
    )
    assert result.values.shape[0] == 30
    matrix, row_index = _to_numpy_sample(rf_model.X, rf_model.feature_cols, sample_size=30)
    assert matrix.shape[0] == 30
    assert row_index.size == 30


def test_compute_shap_rejects_bad_model_kind(rf_model):
    """Un ``model_kind`` invalido en ``compute_shap_values`` lanza error."""
    with pytest.raises(ValueError, match="model_kind"):
        compute_shap_values(
            rf_model.model,
            rf_model.X,
            "svm",  # type: ignore[arg-type]
            feature_cols=rf_model.feature_cols,
        )


def test_to_numpy_sample_rejects_missing_columns(rf_model):
    """``_to_numpy_sample`` exige que existan todas las columnas pedidas."""
    with pytest.raises(ValueError, match="feature columns"):
        _to_numpy_sample(rf_model.X, ("col_inexistente",))


# ===========================================================================
# Grupo C — plots SHAP (AC-3, AC-6, AC-7).
# ===========================================================================


def test_shap_summary_plot_top20_is_figure(rf_shap, rf_model):
    """``shap_summary_plot`` devuelve una ``Figure`` a dpi 200."""
    fig = shap_summary_plot(rf_shap, rf_model.X, top_n=10)
    assert isinstance(fig, Figure)
    assert fig.get_dpi() == 200


def test_shap_dependence_generates_five_plots(xgb_shap, xgb_model):
    """``shap_dependence_plots`` genera exactamente ``top_features`` figuras."""
    plots = shap_dependence_plots(xgb_shap, xgb_model.X, top_features=5)
    assert len(plots) == 5
    for name, fig in plots:
        assert isinstance(name, str)
        assert isinstance(fig, Figure)


def test_shap_dependence_uses_top5_by_importance(rf_shap, rf_model):
    """Los dependence plots usan las top-N features por importancia SHAP global."""
    expected = rf_shap.global_importance.sort("rank").head(5)["feature"].to_list()
    plots = shap_dependence_plots(rf_shap, rf_model.X, top_features=5)
    assert [name for name, _ in plots] == expected


def test_shap_waterfall_returns_figure(rf_shap):
    """``shap_waterfall_plot`` devuelve una ``Figure`` a dpi 200."""
    fig = shap_waterfall_plot(rf_shap, row=0)
    assert isinstance(fig, Figure)
    assert fig.get_dpi() == 200


def test_shap_waterfall_accepts_explicit_class(rf_shap):
    """``shap_waterfall_plot`` acepta una clase explicita valida."""
    fig = shap_waterfall_plot(rf_shap, row=1, class_idx=2)
    assert isinstance(fig, Figure)


def test_shap_waterfall_rejects_bad_row(rf_shap):
    """Una fila fuera de rango en el waterfall lanza ``IndexError``."""
    with pytest.raises(IndexError, match="row"):
        shap_waterfall_plot(rf_shap, row=10_000)


def test_shap_waterfall_rejects_bad_class(rf_shap):
    """Una clase fuera de rango en el waterfall lanza ``ValueError``."""
    with pytest.raises(ValueError, match="class_idx"):
        shap_waterfall_plot(rf_shap, row=0, class_idx=999)


def test_plots_dpi_is_200(rf_shap, rf_model):
    """Todas las figuras del modulo se exportan a dpi 200 (criterio AC-7)."""
    summary = shap_summary_plot(rf_shap, rf_model.X, top_n=8)
    dependence = shap_dependence_plots(rf_shap, rf_model.X, top_features=2)
    waterfall = shap_waterfall_plot(rf_shap, row=0)
    assert summary.get_dpi() == 200
    assert all(fig.get_dpi() == 200 for _, fig in dependence)
    assert waterfall.get_dpi() == 200


# ===========================================================================
# Grupo D — dominancia AlphaEarth (AC-4).
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    ["dim_00", "dim_07", "dim_23", "dim_63"],
)
def test_is_alphaearth_dim_classifies_correctly(name):
    """Los nombres ``dim_NN`` se reconocen como dimensiones AlphaEarth."""
    assert is_alphaearth_dim(name) is True


@pytest.mark.parametrize(
    "name",
    ["NDVI_mean", "EVI_p95", "dim_7", "dim_007", "Dim_07", "ndvi_auc", ""],
)
def test_is_alphaearth_dim_rejects_spectral_index(name):
    """Indices espectrales y nombres mal formados no son dims AlphaEarth."""
    assert is_alphaearth_dim(name) is False


def test_alphaearth_dominance_table_has_family_column(rf_shap):
    """La tabla de dominancia expone ``(rank, feature, family, importance)``."""
    table = alphaearth_dominance_table(rf_shap.global_importance, top_n=8)
    assert table.columns == ["rank", "feature", "family", "importance"]
    assert table.height == 8
    assert set(table["family"].to_list()).issubset(
        {"alphaearth", "spectral_index", "s1", "srtm", "era5", "geom", "other"}
    )


def test_alphaearth_dominance_counts_top20():
    """La dominancia cuenta correctamente cuantas top-N son AlphaEarth."""
    importance = pl.DataFrame(
        {
            "feature": [
                "dim_01",
                "NDVI_mean",
                "dim_42",
                "EVI_p95",
                "dim_07",
            ],
            "importance": [0.9, 0.8, 0.7, 0.6, 0.5],
            "rank": [1, 2, 3, 4, 5],
        }
    )
    table = alphaearth_dominance_table(importance, top_n=5)
    families = table["family"].to_list()
    assert families.count("alphaearth") == 3
    assert families.count("spectral_index") == 2
    # Respeta el orden de `rank`.
    assert table["feature"].to_list() == [
        "dim_01",
        "NDVI_mean",
        "dim_42",
        "EVI_p95",
        "dim_07",
    ]


def test_alphaearth_dominance_accepts_native_importance(rf_model):
    """La dominancia opera sobre la importancia nativa (columna ``importance``)."""
    native = feature_importance_table(rf_model.model, "rf", rf_model.feature_cols)
    table = alphaearth_dominance_table(native, top_n=6)
    assert table.height == 6
    assert "family" in table.columns


def test_alphaearth_dominance_rejects_missing_feature_column():
    """Una tabla sin columna ``feature`` lanza ``ValueError``."""
    with pytest.raises(ValueError, match="feature"):
        alphaearth_dominance_table(pl.DataFrame({"importance": [1.0], "rank": [1]}))


def test_alphaearth_dominance_rejects_missing_importance_column():
    """Una tabla sin columna de importancia reconocible lanza ``ValueError``."""
    with pytest.raises(ValueError, match="importance column"):
        alphaearth_dominance_table(pl.DataFrame({"feature": ["dim_00"], "rank": [1]}))


# ===========================================================================
# Grupo E — estructura del notebook (AC-6).
# ===========================================================================


@pytest.mark.empirical
def test_notebook_has_sections_3_4_5():
    """El notebook ``04_baseline.ipynb`` tiene las secciones 3, 4 y 5 reales."""
    if not _NOTEBOOK_PATH.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado")
    nb = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )
    assert "## 3. Importancia de features nativa" in markdown
    assert "## 4. Analisis SHAP" in markdown
    assert "## 5. Conclusiones de feature engineering" in markdown
    # Los placeholders de US-020 deben haberse reemplazado por contenido real.
    assert "_Placeholder — completado por US-020" not in markdown


@pytest.mark.empirical
def test_notebook_has_shap_waterfall_cell():
    """El notebook §4 invoca el waterfall plot de SHAP."""
    if not _NOTEBOOK_PATH.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado")
    nb = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "shap_waterfall_plot" in code


@pytest.mark.empirical
def test_notebook_has_fe_validation_section():
    """El notebook §5 cruza las top SHAP con el feature engineering de US-018."""
    if not _NOTEBOOK_PATH.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado")
    nb = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "## 5. Conclusiones de feature engineering" in text
    assert "feature_selection" in text


@pytest.mark.empirical
def test_notebook_quantifies_alphaearth_dominance():
    """El notebook §4 cuantifica la dominancia AlphaEarth."""
    if not _NOTEBOOK_PATH.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado")
    nb = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "alphaearth_dominance_table" in code
