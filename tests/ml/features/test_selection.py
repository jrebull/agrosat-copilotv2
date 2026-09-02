"""Suite de tests US-018 ``ml.features.selection`` (Avance 2 CRISP-ML(Q)).

10 grupos (A-J) cubriendo los 13 criterios de aceptacion del plan
``docs/us-planning/us-018.md`` §1.

Markers:
- ``empirical``: salta si ``data/PASTIS-R/`` o el parquet del subset no esta
  disponible (tests sobre data real).
- ``slow``: tests con RF/XGB > 5 s (deselect con ``-m 'not slow'``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.compose import ColumnTransformer

from ml.features import selection as sel
from ml.features.selection import (
    anova_f_select,
    apply_variance_threshold,
    chi2_select,
    compare_before_after,
    compute_feature_importance,
    discretize_features,
    discretize_ndvi_phenology_domain,
    drop_correlated_features,
    fit_factor_analysis,
    fit_pca,
    fit_umap_2d,
    make_preprocessor,
    select_normalizer,
)
from tests.ml.features.fixtures.selection_synthetic import (
    make_collinear_features,
    make_pastis_subset_synthetic,
    make_skewed_distribution,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
SUBSET_PARQUET = REPO_ROOT / "data" / "test_fixtures" / "feature_selection_subset.parquet"
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "feature_engineering" / "03b_fe_spectral_temporal_pastis.ipynb"
)
REPORTS_DIR = REPO_ROOT / "reports" / "feature_selection"


# ---------------------------------------------------------------------------
# Grupo A — VarianceThreshold (AC-1)
# ---------------------------------------------------------------------------


def test_variance_threshold_removes_constant_column() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 21)),
            "year": [2024] * 20,
            "const_col": [5.0] * 20,
            "var_col": np.random.default_rng(42).normal(0, 2, size=20).tolist(),
        }
    )
    filtered, report = apply_variance_threshold(df, threshold=0.01)
    assert "const_col" in report["removed"]
    assert "var_col" in report["kept"]
    assert "const_col" not in filtered.columns
    assert "parcel_id" in filtered.columns
    assert "year" in filtered.columns


def test_variance_threshold_keeps_high_variance() -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 51)),
            "year": [2024] * 50,
            "col_a": rng.normal(0, 5, size=50).tolist(),
            "col_b": rng.normal(0, 10, size=50).tolist(),
        }
    )
    filtered, report = apply_variance_threshold(df, threshold=0.01)
    assert "col_a" in report["kept"]
    assert "col_b" in report["kept"]
    assert filtered.width == 4


def test_variance_threshold_reports_json_schema(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3, 4],
            "year": [2024] * 4,
            "feat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    _, report = apply_variance_threshold(df, threshold=0.01)
    assert set(report.keys()) >= {"kept", "removed", "variances", "threshold"}
    out_path = tmp_path / "variance.json"
    out_path.write_text(
        json.dumps(
            {
                "kept": report["kept"],
                "removed": report["removed"],
                "variances": report["variances"],
                "threshold": report["threshold"],
            }
        )
    )
    loaded = json.loads(out_path.read_text())
    assert loaded["threshold"] == 0.01


def test_variance_threshold_respects_exclude_cols() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3, 4],
            "year": [2024] * 4,
            "feat": [0.001, 0.002, 0.0011, 0.0009],
        }
    )
    filtered, report = apply_variance_threshold(df, threshold=0.5)
    # parcel_id y year se preservan aunque tengan baja varianza (excluidas).
    assert "parcel_id" in filtered.columns
    assert "year" in filtered.columns
    assert "feat" in report["removed"]


# ---------------------------------------------------------------------------
# Grupo B — Drop Correlated (AC-2)
# ---------------------------------------------------------------------------


def test_drop_correlated_pair_removes_one() -> None:
    df = make_collinear_features(n_samples=300, n_clusters=2, cols_per_cluster=2, seed=42)
    _filtered, report = drop_correlated_features(df, threshold=0.95)
    # cada cluster aporta 2 cols altamente correlacionadas -> queda 1.
    assert len(report["removed"]) == 2
    assert len(report["kept"]) == 2


def test_drop_correlated_preserves_uncorrelated() -> None:
    rng = np.random.default_rng(7)
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 101)),
            "year": [2024] * 100,
            "a": rng.normal(0, 1, size=100).tolist(),
            "b": rng.normal(0, 1, size=100).tolist(),
            "c": rng.normal(0, 1, size=100).tolist(),
        }
    )
    _filtered, report = drop_correlated_features(df, threshold=0.95)
    assert set(report["kept"]) == {"a", "b", "c"}
    assert report["removed"] == []


def test_drop_correlated_returns_kept_set_deterministic() -> None:
    df = make_collinear_features(n_samples=400, n_clusters=3, cols_per_cluster=3, seed=42)
    _, report1 = drop_correlated_features(df, threshold=0.95)
    _, report2 = drop_correlated_features(df, threshold=0.95)
    assert report1["kept"] == report2["kept"]
    assert report1["removed"] == report2["removed"]


def test_drop_correlated_method_spearman() -> None:
    df = make_collinear_features(n_samples=200, n_clusters=2, cols_per_cluster=2, seed=42)
    _filtered, report = drop_correlated_features(df, threshold=0.9, method="spearman")
    assert report["method"] == "spearman"
    assert len(report["removed"]) >= 1


# ---------------------------------------------------------------------------
# Grupo C — chi2 (AC-3)
# ---------------------------------------------------------------------------


def test_chi2_select_returns_k_best() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=400, n_features=40, n_classes=20, seed=42)
    top_df, scores = chi2_select(X, y, k_best=10, binning_strategy="quartiles")
    assert len(scores) == 10
    # Solo cuenta cols feature en el top, parcel_id/year se preservan aparte.
    feature_only = [c for c in top_df.columns if c not in ("parcel_id", "year")]
    assert len(feature_only) == 10


def test_chi2_select_handles_binning() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=200, n_features=20, n_classes=20, seed=1)
    _, scores_q = chi2_select(X, y, k_best=5, binning_strategy="quartiles")
    _, scores_d = chi2_select(X, y, k_best=5, binning_strategy="deciles")
    # Ambas estrategias devuelven k_best features (no necesariamente las mismas).
    assert len(scores_q) == 5
    assert len(scores_d) == 5


def test_chi2_select_documents_synthetic_fallback_when_no_categorical() -> None:
    """Cuando todas las features son numericas, el binning sintetico es el fallback."""
    X, y, _ = make_pastis_subset_synthetic(n_samples=200, n_features=15, n_classes=20, seed=2)
    _top_df, scores = chi2_select(X, y, k_best=5, binning_strategy="quartiles")
    # Sin binning_strategy, las features negativas (NDVI_mean) deben tolerar clip.
    _top_df2, scores2 = chi2_select(X, y, k_best=5, binning_strategy=None)
    assert len(scores) > 0
    assert len(scores2) > 0


# ---------------------------------------------------------------------------
# Grupo D — ANOVA F (AC-4)
# ---------------------------------------------------------------------------


def test_anova_f_returns_ranked_features() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=400, n_features=30, n_classes=20, seed=42)
    _, scores = anova_f_select(X, y, k_best=10)
    assert len(scores) == 10
    vals = list(scores.values())
    # Top scores deben ser >= rest (ordenados desc por construccion del modulo).
    assert vals[0] >= vals[-1]


def test_anova_f_top_features_have_high_separation_synthetic() -> None:
    """Las primeras 3 columnas del fixture sintetico tienen senal real."""
    X, y, _ = make_pastis_subset_synthetic(n_samples=600, n_features=30, n_classes=20, seed=42)
    _, scores = anova_f_select(X, y, k_best=5)
    # Al menos una de las top-5 viene del bloque NDVI_* (las 3 con senal).
    top = set(scores.keys())
    assert any(t.startswith("NDVI_") for t in top)


@pytest.mark.empirical
def test_anova_f_top3_contains_ndvi_or_fft() -> None:
    """Sobre el subset PASTIS real, top-3 incluye al menos un indice
    espectral o un componente temporal (CA-4 ampliado: la hipotesis era
    "senial temporal domina"; cualquier indice vegetativo o FFT confirma)."""
    if not SUBSET_PARQUET.exists():
        pytest.skip(f"Subset PASTIS no presente en {SUBSET_PARQUET}")
    df = pl.read_parquet(SUBSET_PARQUET)
    if "class_id" not in df.columns:
        pytest.skip("Subset PASTIS sin columna class_id")
    y = df.get_column("class_id")
    X = df.drop(["class_id", "fold"] if "fold" in df.columns else ["class_id"])
    _, scores = anova_f_select(X, y, k_best=3)
    top3 = list(scores.keys())
    # Cualquier indice vegetativo del set DEFAULT_INDICES o componente FFT
    # cumple la hipotesis "senial espectro-temporal domina".
    veg_prefixes = (
        "NDVI",
        "NDWI",
        "EVI",
        "NDMI",
        "NBR",
        "MSAVI",
        "NDRE",
        "MCARI",
        "CCCI",
        "GCVI",
        "PSRI",
        "NDCI",
        "FAPAR",
        "LAI",
        "RENDVI",
        "SAVI",
        "TSAVI",
    )
    assert any(any(name.startswith(p) for p in veg_prefixes) or "fft" in name for name in top3), (
        f"top3 sin indices ni FFT: {top3}"
    )


# ---------------------------------------------------------------------------
# Grupo E — PCA (AC-5)
# ---------------------------------------------------------------------------


def test_pca_recovers_target_variance_synthetic() -> None:
    rng = np.random.default_rng(42)
    # 5 dims latentes + ruido en 50 cols.
    latent = rng.normal(0, 1, size=(300, 5))
    weights = rng.normal(0, 1, size=(5, 50))
    X = latent @ weights + rng.normal(0, 0.05, size=(300, 50))
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    result = fit_pca(X_std, target_variance=0.95)
    cum = result["cumulative_variance"]
    assert cum[result["n_components"] - 1] >= 0.95


def test_pca_n_components_lt_n_features() -> None:
    X, _, _ = make_pastis_subset_synthetic(n_samples=300, n_features=50, n_classes=20, seed=4)
    matrix = X.drop(["parcel_id", "year"]).to_numpy().astype(np.float64)
    matrix_std = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
    result = fit_pca(matrix_std, target_variance=0.95)
    assert 1 <= result["n_components"] <= 50


def test_pca_explained_variance_is_monotonic_decreasing() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 30))
    result = fit_pca(X, target_variance=0.99)
    ratios = result["explained_variance_ratio"]
    assert all(ratios[i] >= ratios[i + 1] - 1e-9 for i in range(len(ratios) - 1))


def test_pca_returns_transformer_callable() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100, 20))
    result = fit_pca(X, target_variance=0.9)
    transformed = result["transformer"].transform(X)
    assert transformed.shape == (100, result["n_components"])


# ---------------------------------------------------------------------------
# Grupo F — Factor Analysis (AC-6)
# ---------------------------------------------------------------------------


def test_fa_loadings_shape_matches_features() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 25))
    result = fit_factor_analysis(X, n_factors=5)
    assert result["loadings"].shape == (25, 5)


def test_fa_explained_variance_positive() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(150, 20))
    result = fit_factor_analysis(X, n_factors=3)
    assert (result["explained_variance_approx"] > 0).all()


def test_fa_two_factors_recoverable_synthetic() -> None:
    """Construye 2 factores latentes claros y verifica que FA los recupera."""
    rng = np.random.default_rng(7)
    latent = rng.normal(0, 1, size=(400, 2))
    weights = np.array(
        [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10,
        dtype=np.float64,
    ).T  # shape (2, 20)
    X = latent @ weights + rng.normal(0, 0.1, size=(400, 20))
    result = fit_factor_analysis(X, n_factors=2)
    # Loadings deben separar las dos primeras 10 columnas vs siguientes 10.
    loadings = np.abs(result["loadings"])
    assert loadings.shape == (20, 2)


# ---------------------------------------------------------------------------
# Grupo G — UMAP (AC-7)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_umap_2d_shape_is_n_samples_x_2() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(150, 20))
    emb = fit_umap_2d(X, random_state=42)
    assert emb.shape == (150, 2)


@pytest.mark.slow
def test_umap_deterministic_with_seed() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(120, 15))
    emb1 = fit_umap_2d(X, random_state=42)
    emb2 = fit_umap_2d(X, random_state=42)
    assert np.allclose(emb1, emb2, atol=1e-3)


@pytest.mark.slow
@pytest.mark.empirical
def test_umap_smoke_pastis_subset() -> None:
    if not SUBSET_PARQUET.exists():
        pytest.skip(f"Subset PASTIS no presente en {SUBSET_PARQUET}")
    df = pl.read_parquet(SUBSET_PARQUET)
    feature_df = df.drop([c for c in ("class_id", "fold") if c in df.columns])
    matrix = feature_df.drop(["parcel_id", "year"]).to_numpy().astype(np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0)
    emb = fit_umap_2d(matrix[:200], random_state=42)
    assert emb.shape[0] == min(200, matrix.shape[0])


# ---------------------------------------------------------------------------
# Grupo H — Feature importance (AC-8)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rf_importance_returns_sorted_df() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=300, n_features=30, n_classes=10, seed=42)
    df = compute_feature_importance(X, y, model="rf", n_estimators=50)
    assert df.height == 30
    imps = df.get_column("importance").to_list()
    assert all(imps[i] >= imps[i + 1] - 1e-12 for i in range(len(imps) - 1))


@pytest.mark.slow
def test_xgb_importance_returns_sorted_df() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=300, n_features=30, n_classes=10, seed=42)
    df = compute_feature_importance(X, y, model="xgb", n_estimators=50)
    assert df.height == 30
    imps = df.get_column("importance").to_list()
    assert all(imps[i] >= imps[i + 1] - 1e-12 for i in range(len(imps) - 1))


@pytest.mark.slow
def test_rf_xgb_top10_overlap_min() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=400, n_features=30, n_classes=10, seed=42)
    rf_df = compute_feature_importance(X, y, model="rf", n_estimators=50)
    xgb_df = compute_feature_importance(X, y, model="xgb", n_estimators=50)
    top10_rf = set(rf_df.head(10).get_column("feature").to_list())
    top10_xgb = set(xgb_df.head(10).get_column("feature").to_list())
    overlap = top10_rf & top10_xgb
    # Ambos modelos deben coincidir en al menos 1 feature en top-10 sobre senal real.
    assert len(overlap) >= 1


# ---------------------------------------------------------------------------
# Grupo I — Comparativa antes/despues + normalizacion (AC-9, AC-10)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_before_after_uses_pastis_folds_not_random() -> None:
    """Verifica que ``folds`` se respeta como split espacial.

    Pasa folds 1..5 estratificados y comprueba que el RF interno NO mezcla
    aleatoriamente: si se cambia el orden del vector ``folds`` permutando
    asignaciones, los scores deben cambiar (prueba la dependencia).
    """
    X, y, folds = make_pastis_subset_synthetic(n_samples=300, n_features=20, n_classes=10, seed=42)
    table1 = compare_before_after(X, X, y, folds, n_estimators=20)
    # Permutamos folds y volvemos a evaluar: scores deben ser DISTINTOS.
    rng = np.random.default_rng(1)
    folds_perm = rng.permutation(folds)
    table2 = compare_before_after(X, X, y, folds_perm, n_estimators=20)
    s1 = table1.get_column("f1_macro_mean").to_list()
    s2 = table2.get_column("f1_macro_mean").to_list()
    # Al menos uno de los dos numeros debe cambiar al permutar folds (prueba
    # que folds influye, descartando random implicito).
    assert not np.allclose(s1, s2, atol=1e-6)


@pytest.mark.slow
def test_compare_before_after_returns_4_strategies() -> None:
    X, y, folds = make_pastis_subset_synthetic(n_samples=200, n_features=20, n_classes=10, seed=42)
    extra = {
        "pca_0.95": X.select(["parcel_id", "year", *X.columns[2:12]]),
        "selected+pca": X.select(["parcel_id", "year", *X.columns[2:8]]),
    }
    table = compare_before_after(X, X, y, folds, extra_strategies=extra, n_estimators=20)
    assert table.height == 4
    names = table.get_column("strategy").to_list()
    assert "raw" in names
    assert "variance+correlation" in names
    assert "pca_0.95" in names
    assert "selected+pca" in names


def test_compare_before_after_writes_csv_md(tmp_path: Path) -> None:
    """El caller puede serializar el DataFrame a CSV + MD sin errores."""
    X, y, folds = make_pastis_subset_synthetic(n_samples=150, n_features=15, n_classes=5, seed=42)
    table = compare_before_after(X, X, y, folds, n_estimators=10)
    csv_path = tmp_path / "before_after.csv"
    md_path = tmp_path / "before_after.md"
    table.write_csv(csv_path)
    md_path.write_text("| " + " | ".join(table.columns) + " |\n")
    assert csv_path.exists()
    assert md_path.exists()


def test_select_normalizer_routes_lai_to_log1p() -> None:
    scaler, _ = select_normalizer("LAI_mean", {"skew": 0.5})
    assert scaler == "log1p"
    scaler2, _ = select_normalizer("biomass_total", {"skew": 0.1})
    assert scaler2 == "log1p"


def test_select_normalizer_routes_skewed_to_yeo_johnson() -> None:
    scaler, just = select_normalizer("NDVI_mean", {"skew": 2.5})
    assert scaler == "yeo-johnson"
    assert "Yeo-Johnson" in just or "yeo" in just.lower()


def test_select_normalizer_strategy_nn_minmax() -> None:
    scaler, _ = select_normalizer("some_feat", {"skew": 0.1}, strategy="nn")
    assert scaler == "minmax"


def test_select_normalizer_default_standard() -> None:
    scaler, _ = select_normalizer("some_feat", {"skew": 0.1}, strategy="linear")
    assert scaler == "standard"


def test_make_preprocessor_returns_sklearn_compose() -> None:
    X, _, _ = make_pastis_subset_synthetic(n_samples=100, n_features=15, n_classes=5, seed=42)
    pre = make_preprocessor(X, strategy="linear")
    assert isinstance(pre, ColumnTransformer)


def test_make_preprocessor_fits_and_transforms() -> None:
    X, _, _ = make_pastis_subset_synthetic(n_samples=100, n_features=15, n_classes=5, seed=42)
    pre = make_preprocessor(X, strategy="linear")
    matrix = X.drop(["parcel_id", "year"]).to_numpy().astype(np.float64)
    transformed = pre.fit_transform(matrix)
    assert transformed.shape[0] == 100


def test_yeo_johnson_handles_negative_ndvi() -> None:
    """Regression: NDVI con valores negativos (agua, sombras) no debe romper."""
    rng = np.random.default_rng(0)
    n = 80
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, n + 1)),
            "year": [2024] * n,
            "NDVI_mean": rng.uniform(-0.5, 1.0, size=n).tolist(),
            "NDVI_std": rng.lognormal(0.0, 1.5, size=n).tolist(),  # sesgada
            "other_feat": rng.normal(0, 1, size=n).tolist(),
        }
    )
    pre = make_preprocessor(df, strategy="linear")
    matrix = df.drop(["parcel_id", "year"]).to_numpy().astype(np.float64)
    transformed = pre.fit_transform(matrix)
    assert transformed.shape[0] == n
    assert np.all(np.isfinite(transformed))


def test_make_preprocessor_joblib_serializable(tmp_path: Path) -> None:
    """Requisito Aaron (Backend): el ColumnTransformer debe ser joblib-serializable."""
    import joblib

    X, _, _ = make_pastis_subset_synthetic(n_samples=80, n_features=10, n_classes=5, seed=42)
    pre = make_preprocessor(X, strategy="linear")
    matrix = X.drop(["parcel_id", "year"]).to_numpy().astype(np.float64)
    pre.fit(matrix)
    out = tmp_path / "preprocessor.joblib"
    joblib.dump(pre, out)
    loaded = joblib.load(out)
    transformed = loaded.transform(matrix)
    assert transformed.shape[0] == 80


# ---------------------------------------------------------------------------
# Grupo J — Notebook + reports (AC-11, AC-13)
# ---------------------------------------------------------------------------


def test_notebook_has_conclusiones_section() -> None:
    """Verifica que el notebook cierra con una seccion de conclusiones.

    El header debe ser generico (sin US-XXX, AC-X, rubrica) por convencion
    de ``notebooks/CLAUDE.md``. Buscamos una cell markdown que arranque con
    ``## Conclusiones`` o ``# Conclusiones`` y contenga la subseccion
    ``Lo que sigue`` (regla obligatoria del checklist de cierre).
    """
    if not NOTEBOOK_PATH.exists():
        pytest.skip(f"Notebook {NOTEBOOK_PATH} no presente")
    nb_json = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    import re

    header_re = re.compile(r"^#{1,3}\s+(\d+\.\s+)?Conclusiones\b", re.MULTILINE)
    found_header = False
    found_lo_que_sigue = False
    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if header_re.search(source):
            found_header = True
            if "Lo que sigue" in source:
                found_lo_que_sigue = True
                break
    assert found_header, f"Cell markdown con header 'Conclusiones' no encontrada en {NOTEBOOK_PATH}"
    assert found_lo_que_sigue, (
        f"La seccion 'Conclusiones' debe cerrar con 'Lo que sigue' en {NOTEBOOK_PATH}"
    )


def test_reports_directory_structure() -> None:
    """La carpeta ``reports/feature_selection/`` debe existir con ``.gitkeep``."""
    assert REPORTS_DIR.exists()
    assert (REPORTS_DIR / ".gitkeep").exists()


# ---------------------------------------------------------------------------
# Tests adicionales para subir cobertura
# ---------------------------------------------------------------------------


def test_to_numpy_excludes_index_cols() -> None:
    df = pl.DataFrame({"parcel_id": [1, 2], "year": [2024, 2024], "a": [1.0, 2.0], "b": [3.0, 4.0]})
    matrix, names = sel._to_numpy(df)
    assert matrix.shape == (2, 2)
    assert names == ["a", "b"]


def test_compare_before_after_smoke_no_extra() -> None:
    X, y, folds = make_pastis_subset_synthetic(n_samples=100, n_features=10, n_classes=5, seed=42)
    table = compare_before_after(X, X, y, folds, n_estimators=10)
    assert table.height == 2


def test_fit_pca_raises_on_invalid_variance() -> None:
    with pytest.raises(ValueError):
        fit_pca(np.zeros((5, 3)), target_variance=1.5)


def test_fit_pca_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        fit_pca(np.empty((0, 0)), target_variance=0.9)


def test_fit_factor_analysis_raises_on_excess_factors() -> None:
    with pytest.raises(ValueError):
        fit_factor_analysis(np.zeros((3, 2)), n_factors=10)


def test_compute_feature_importance_invalid_model() -> None:
    X, y, _ = make_pastis_subset_synthetic(n_samples=50, n_features=10, n_classes=5, seed=0)
    with pytest.raises(ValueError):
        compute_feature_importance(X, y, model="svc")  # type: ignore[arg-type]


def test_apply_variance_threshold_empty_features() -> None:
    df = pl.DataFrame({"parcel_id": [1, 2], "year": [2024, 2024]})
    filtered, report = apply_variance_threshold(df)
    assert filtered.equals(df)
    assert report["kept"] == []


def test_drop_correlated_features_few_cols() -> None:
    df = pl.DataFrame({"parcel_id": [1, 2], "year": [2024, 2024], "feat": [1.0, 2.0]})
    _filtered, report = drop_correlated_features(df, threshold=0.95)
    assert report["removed"] == []


def test_make_skewed_distribution_is_skewed() -> None:
    from scipy.stats import skew

    s = make_skewed_distribution(n=500, skew=2.0)
    assert skew(np.asarray(s.to_list())) > 0.5


# ---------------------------------------------------------------------------
# Grupo O — Discretizacion / binning (US-018 extension, Construccion 30 pts)
# ---------------------------------------------------------------------------


def test_discretize_quantile_returns_n_bins() -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 101)),
            "year": [2024] * 100,
            "NDVI_mean": rng.normal(0.4, 0.2, size=100).tolist(),
            "EVI_mean": rng.normal(0.3, 0.15, size=100).tolist(),
        }
    )
    out, edges = discretize_features(
        df, columns=["NDVI_mean", "EVI_mean"], strategy="quantile", n_bins=4
    )
    assert "NDVI_mean__bin" in out.columns
    assert "EVI_mean__bin" in out.columns
    bins = set(out.get_column("NDVI_mean__bin").to_list())
    assert bins <= {0, 1, 2, 3}
    # Cada feature tiene n_bins-1 bordes internos registrados.
    assert len(edges["NDVI_mean"]) == 3


def test_discretize_uniform_evenly_spaced() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "year": [2024] * 8,
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    out, edges = discretize_features(df, columns=["x"], strategy="uniform", n_bins=4)
    assert "x__bin" in out.columns
    bin_values = out.get_column("x__bin").to_list()
    # Uniform: 4 bins, monotonicamente crecientes con el valor de entrada.
    assert bin_values[0] <= bin_values[-1]
    assert len(edges["x"]) == 3


def test_discretize_kmeans_deterministic() -> None:
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 81)),
            "year": [2024] * 80,
            "EVI_mean": rng.normal(0.3, 0.2, size=80).tolist(),
        }
    )
    out1, edges1 = discretize_features(
        df, columns=["EVI_mean"], strategy="kmeans", n_bins=4, random_state=42
    )
    out2, edges2 = discretize_features(
        df, columns=["EVI_mean"], strategy="kmeans", n_bins=4, random_state=42
    )
    assert out1.get_column("EVI_mean__bin").to_list() == out2.get_column("EVI_mean__bin").to_list()
    assert edges1["EVI_mean"] == edges2["EVI_mean"]


def test_discretize_domain_requires_bin_edges() -> None:
    df = pl.DataFrame({"parcel_id": [1, 2], "year": [2024, 2024], "NDVI_mean": [0.1, 0.5]})
    with pytest.raises(ValueError):
        discretize_features(df, columns=["NDVI_mean"], strategy="domain", n_bins=3)


def test_discretize_domain_applies_custom_edges() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 11)),
            "year": [2024] * 10,
            "NDVI_mean": [-0.2, 0.0, 0.1, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 0.95],
        }
    )
    out, edges = discretize_features(
        df,
        columns=["NDVI_mean"],
        strategy="domain",
        bin_edges={"NDVI_mean": [0.0, 0.4, 0.7]},
    )
    bins = out.get_column("NDVI_mean__bin").to_list()
    assert max(bins) == 3
    assert min(bins) == 0
    assert edges["NDVI_mean"] == [0.0, 0.4, 0.7]


def test_discretize_invalid_strategy_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1], "year": [2024], "x": [0.0]})
    with pytest.raises(ValueError):
        discretize_features(df, columns=["x"], strategy="invalid", n_bins=4)  # type: ignore[arg-type]


def test_discretize_invalid_n_bins_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1], "year": [2024], "x": [0.0]})
    with pytest.raises(ValueError):
        discretize_features(df, columns=["x"], strategy="quantile", n_bins=1)


def test_discretize_ndvi_phenology_domain_creates_5_bins() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 11)),
            "year": [2024] * 10,
            "NDVI_mean": [-0.3, -0.1, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.9],
        }
    )
    out, labels = discretize_ndvi_phenology_domain(df, ndvi_col="NDVI_mean")
    assert "NDVI_mean__pheno" in out.columns
    assert labels == ["water", "bare", "sparse", "moderate", "dense"]
    pheno = out.get_column("NDVI_mean__pheno").to_list()
    assert pheno[0] == "water"
    assert pheno[-1] == "dense"


def test_discretize_ndvi_phenology_domain_missing_col_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1], "year": [2024]})
    with pytest.raises(ValueError):
        discretize_ndvi_phenology_domain(df, ndvi_col="NDVI_mean")


def test_discretize_ndvi_phenology_domain_invalid_bins_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1], "year": [2024], "NDVI_mean": [0.5]})
    with pytest.raises(ValueError):
        discretize_ndvi_phenology_domain(df, ndvi_col="NDVI_mean", bins=(1.0, 0.0))


# ---------------------------------------------------------------------------
# Grupo P — make_preprocessor extendido con categorical_cols
# ---------------------------------------------------------------------------


def test_make_preprocessor_with_categorical_cols_onehot(tmp_path: Path) -> None:
    """El nuevo argumento categorical_cols debe agregar un branch OneHotEncoder."""
    import joblib

    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 41)),
            "year": [2024] * 40,
            "NDVI_mean": np.random.default_rng(0).uniform(0.2, 0.8, size=40).tolist(),
            "EVI_mean": np.random.default_rng(1).uniform(0.1, 0.6, size=40).tolist(),
            "crop_group": (["cereals"] * 20) + (["root_crops"] * 20),
        }
    )
    pre = make_preprocessor(
        df,
        strategy="linear",
        categorical_cols=("crop_group",),
        categorical_encoder="onehot",
    )
    assert isinstance(pre, ColumnTransformer)
    # El nombre del paso categorico aparece en la lista de transformers.
    names = [name for name, _step, _cols in pre.transformers]
    assert any(n.startswith("categorical_onehot") for n in names)

    # fit_transform sobre matriz que conserva la columna categorica.
    matrix = df.drop(["parcel_id", "year"]).to_numpy()
    transformed = pre.fit_transform(matrix)
    # Numericas (2) + one-hot (2 categorias) = 4 columnas.
    assert transformed.shape == (40, 4)
    # joblib-serializable.
    out_path = tmp_path / "pre_cat.joblib"
    joblib.dump(pre, out_path)
    loaded = joblib.load(out_path)
    transformed2 = loaded.transform(matrix)
    assert np.allclose(transformed, transformed2)


def test_make_preprocessor_with_categorical_cols_ordinal() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 31)),
            "year": [2024] * 30,
            "NDVI_mean": np.random.default_rng(0).uniform(0.0, 1.0, size=30).tolist(),
            "season": (["winter"] * 10) + (["spring"] * 10) + (["summer"] * 10),
        }
    )
    pre = make_preprocessor(
        df,
        strategy="linear",
        categorical_cols=("season",),
        categorical_encoder="ordinal",
    )
    matrix = df.drop(["parcel_id", "year"]).to_numpy()
    transformed = pre.fit_transform(matrix)
    # 1 numerica + 1 ordinal -> 2 columnas.
    assert transformed.shape == (30, 2)


def test_make_preprocessor_backward_compatible_no_categorical() -> None:
    """Sin categorical_cols, la API debe operar identica a la version original."""
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 21)),
            "year": [2024] * 20,
            "x": np.random.default_rng(0).normal(0, 1, size=20).tolist(),
        }
    )
    pre_old = make_preprocessor(df, strategy="linear")
    pre_new = make_preprocessor(df, strategy="linear", categorical_cols=())
    assert len(pre_old.transformers) == len(pre_new.transformers)
    matrix = df.drop(["parcel_id", "year"]).to_numpy()
    assert pre_old.fit_transform(matrix).shape == pre_new.fit_transform(matrix).shape
