"""Tests smoke de ml.train.baseline (US-019).

Conjunto minimo de validacion de la libreria del baseline tabular. La
suite exhaustiva (~22 tests, grupos A-E) la completa el sub-agente de
tests. Todos los tests core usan el fixture sintetico determinista, sin
depender del parquet PASTIS-R de 76 MB.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from ml.train.baseline import (
    BaselineResult,
    SpatialXGBClassifier,
    build_estimator,
    evaluate_with_spatial_cv,
    train_one_model,
    tune_baseline,
)
from tests.ml.train.fixtures.baseline_synthetic import make_baseline_dataset

_METRIC_KEYS = {"f1_macro", "f1_weighted", "miou", "accuracy", "cohen_kappa"}


@pytest.fixture(scope="module")
def synthetic_df() -> pl.DataFrame:
    """DataFrame sintetico determinista compartido por el modulo."""
    return make_baseline_dataset(n=300, n_classes=4, n_features=10, n_patches=14, seed=42)


def test_train_rf_returns_baseline_result(synthetic_df: pl.DataFrame) -> None:
    """train_one_model('rf') devuelve un BaselineResult completo."""
    result = train_one_model(synthetic_df, model="rf")
    assert isinstance(result, BaselineResult)
    assert result.model_kind == "rf"
    assert set(result.metrics.keys()) == _METRIC_KEYS
    assert len(result.feature_cols) == 10


def test_train_xgb_returns_baseline_result(synthetic_df: pl.DataFrame) -> None:
    """train_one_model('xgb') maneja class_ids no contiguos via LabelEncoder."""
    result = train_one_model(synthetic_df, model="xgb")
    assert result.model_kind == "xgb"
    # Class ids del fixture no son contiguos (saltan 0); el encoder los mapea.
    assert result.label_classes == tuple(sorted(result.label_classes))
    assert min(result.label_classes) >= 1


def test_baseline_result_feature_cols_excludes_meta(
    synthetic_df: pl.DataFrame,
) -> None:
    """Las columnas de metadata no entran como features."""
    result = train_one_model(synthetic_df, model="rf")
    for meta_col in ("parcel_id", "class_id", "fold", "patch_id", "n_pixels"):
        assert meta_col not in result.feature_cols


def test_cv_metrics_have_mean_and_std(synthetic_df: pl.DataFrame) -> None:
    """cv_metrics expone (media, std) por metrica."""
    result = train_one_model(synthetic_df, model="rf")
    for key in _METRIC_KEYS:
        assert key in result.cv_metrics
        mean, std = result.cv_metrics[key]
        assert isinstance(mean, float)
        assert isinstance(std, float)


def test_evaluate_returns_oof_predictions(synthetic_df: pl.DataFrame) -> None:
    """evaluate_with_spatial_cv devuelve metricas CV y predicciones OOF."""
    from ml.train.baseline import _prepare_dataframe

    clean = _prepare_dataframe(synthetic_df)

    def factory():  # type: ignore[no-untyped-def]
        return build_estimator("rf", {"n_estimators": 50, "random_state": 42})

    cv_metrics, y_true, y_pred = evaluate_with_spatial_cv(clean, factory)
    assert set(cv_metrics.keys()) == _METRIC_KEYS
    assert y_true.shape == y_pred.shape
    assert y_true.size > 0


def test_tune_returns_best_params(synthetic_df: pl.DataFrame) -> None:
    """tune_baseline devuelve un diccionario de best_params."""
    best = tune_baseline(
        synthetic_df,
        model="rf",
        param_grid={"n_estimators": [50, 100], "max_depth": [5, None]},
    )
    assert isinstance(best, dict)
    assert "n_estimators" in best


def test_grid_combos_within_budget() -> None:
    """Las grillas por defecto no exceden 8 combinaciones (criterio AC-4)."""
    from ml.train.baseline import _RF_PARAM_GRID, _XGB_PARAM_GRID

    for grid in (_RF_PARAM_GRID, _XGB_PARAM_GRID):
        combos = int(np.prod([len(v) for v in grid.values()]))
        assert combos <= 8


def test_train_deterministic_with_seed(synthetic_df: pl.DataFrame) -> None:
    """Dos entrenamientos con la misma semilla dan el mismo F1-macro."""
    a = train_one_model(synthetic_df, model="rf", random_state=7)
    b = train_one_model(synthetic_df, model="rf", random_state=7)
    assert a.metrics["f1_macro"] == pytest.approx(b.metrics["f1_macro"])


# ---------------------------------------------------------------------------
# LightGBM (3er modelo del baseline tabular, paralelo a RF/XGB).
# ---------------------------------------------------------------------------


def test_build_estimator_lgbm_returns_lgbm() -> None:
    """build_estimator('lgbm', {}) devuelve una instancia de LGBMClassifier."""
    estimator = build_estimator("lgbm", {})
    assert isinstance(estimator, LGBMClassifier)


def test_train_one_model_lgbm_returns_baseline_result(
    synthetic_df: pl.DataFrame,
) -> None:
    """train_one_model('lgbm') corre fit + spatial CV y devuelve BaselineResult."""
    result = train_one_model(synthetic_df, model="lgbm", k_folds=2)
    assert isinstance(result, BaselineResult)
    assert result.model_kind == "lgbm"
    assert set(result.metrics.keys()) == _METRIC_KEYS
    f1 = result.metrics["f1_macro"]
    assert 0.0 <= f1 <= 1.0
    assert len(result.feature_cols) == 10


def test_tune_baseline_lgbm_returns_best_params(synthetic_df: pl.DataFrame) -> None:
    """tune_baseline('lgbm') con grilla minima devuelve dict de best_params."""
    best = tune_baseline(
        synthetic_df,
        model="lgbm",
        param_grid={"n_estimators": [50, 100], "num_leaves": [15, 31]},
        k_folds=2,
    )
    assert isinstance(best, dict)
    assert "n_estimators" in best
    assert "num_leaves" in best


def test_lgbm_handles_nan_natively(synthetic_df: pl.DataFrame) -> None:
    """LGBM acepta NaN en la matriz X sin que el fit se rompa.

    Inyectamos NaN en ~5% de las celdas de las features y verificamos que
    el entrenamiento completa y devuelve metricas finitas.
    """
    rng = np.random.default_rng(123)
    feature_cols = [c for c in synthetic_df.columns if c.startswith("feat_")]
    matrix = synthetic_df.select(feature_cols).to_numpy().astype(np.float64)
    mask = rng.random(matrix.shape) < 0.05
    matrix[mask] = np.nan
    df_with_nan = synthetic_df.with_columns(
        [pl.Series(name=c, values=matrix[:, j]) for j, c in enumerate(feature_cols)]
    )
    result = train_one_model(df_with_nan, model="lgbm", k_folds=2)
    assert result.model_kind == "lgbm"
    assert np.isfinite(result.metrics["f1_macro"])


def test_build_estimator_xgb_returns_spatial_subclass() -> None:
    """build_estimator('xgb') devuelve SpatialXGBClassifier (subclase de XGB)."""
    estimator = build_estimator("xgb", {"n_estimators": 10, "random_state": 42})
    assert isinstance(estimator, SpatialXGBClassifier)
    # Sigue siendo un XGBClassifier para isinstance/interpretabilidad/_is_xgb.
    assert isinstance(estimator, XGBClassifier)


def test_spatial_xgb_fits_fold_missing_classes() -> None:
    """SpatialXGBClassifier no falla cuando faltan clases (labels no contiguos).

    Reproduce el bug original: bajo CV espacial un fold puede no contener
    todas las clases, dejando ``y`` no contiguo (p.ej. ``[0,1,2,5,7]``).
    XGBoost >= 1.6 lanzaria ``ValueError: Invalid classes inferred``; el
    wrapper re-encoda localmente y entrena sin error.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 6)).astype(np.float64)
    # Etiquetas globales con huecos: faltan las clases 3, 4 y 6.
    global_labels = np.array([0, 1, 2, 5, 7])
    y = rng.choice(global_labels, size=60)

    clf = SpatialXGBClassifier(n_estimators=10, random_state=42, device="cpu")
    clf.fit(x, y)  # no debe lanzar ValueError
    preds = clf.predict(x)

    # Las predicciones viven en el espacio de etiquetas GLOBAL, no en [0, k).
    assert set(np.unique(preds)).issubset(set(global_labels))
    # `global_classes_` preserva las etiquetas originales; `classes_` queda
    # en el espacio local [0, k) que el booster valida internamente.
    assert set(clf.global_classes_) == set(global_labels)


def test_spatial_xgb_predict_roundtrips_global_labels() -> None:
    """Con todas las clases presentes el wrapper es identidad (sin remapeo)."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 5)).astype(np.float64)
    y = rng.integers(0, 4, size=80)  # clases 0..3 contiguas

    clf = SpatialXGBClassifier(n_estimators=10, random_state=42, device="cpu")
    clf.fit(x, y)
    preds = clf.predict(x)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})


def test_train_xgb_no_nan_with_imbalanced_spatial_folds() -> None:
    """train_one_model('xgb') produce metricas finitas con folds desbalanceados.

    Antes del wrapper, los folds sin clases raras se puntuaban ``nan`` y
    contaminaban la seleccion. Generamos un dataset con clases minoritarias
    concentradas en pocos patches para forzar folds incompletos.
    """
    df = make_baseline_dataset(n=400, n_classes=8, n_features=10, n_patches=20, seed=3)
    result = train_one_model(df, model="xgb", k_folds=3)
    assert result.model_kind == "xgb"
    assert np.isfinite(result.metrics["f1_macro"])
    assert np.isfinite(result.cv_metrics["f1_macro"][0])
