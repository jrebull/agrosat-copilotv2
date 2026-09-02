"""Tests de ``ml.eval.learning_curves`` (US-021, EPIC 4).

Cubre las curvas de aprendizaje/validacion y el diagnostico de sub/sobreajuste
en seis grupos:

- A — curva de aprendizaje (:func:`plot_learning_curve`).
- B — curva de validacion (:func:`plot_validation_curve`).
- C — diagnostico (:func:`diagnose_fit`).
- D — CV espacial materializado y sin leakage.
- E — propiedades de los plots (dpi, banda +/-std).
- F — integracion: notebook seccion 5b + documentacion del spatial CV.

Los tests core usan :func:`make_curve_dataset` (fixture sintetica con
separabilidad ajustable) — autocontenidos, sin tocar el parquet de US-018 ni el
CV espacial real H3+KMeans, que es O(N^2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from ml.eval.learning_curves import (
    FitDiagnosis,
    LearningCurveResult,
    ValidationCurveResult,
    _materialize_cv_splits,
    diagnose_fit,
    plot_learning_curve,
    plot_validation_curve,
)
from ml.train.baseline import build_estimator
from tests.ml.eval.fixtures.curves_synthetic import make_curve_dataset, make_cv_splits

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Factories de estimadores ligeros (rapidos para CI).
# ---------------------------------------------------------------------------


def _rf(max_depth: int | None = 8, n_estimators: int = 40):
    """RF ligero y determinista para los tests de las curvas."""
    return build_estimator(
        "rf",
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": 1,
            "random_state": 42,
        },
    )


def _xgb(n_estimators: int = 30, max_depth: int = 4, learning_rate: float = 0.2):
    """XGB ligero y determinista para los tests de las curvas."""
    return build_estimator(
        "xgb",
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
            "verbosity": 0,
        },
    )


# ===========================================================================
# Grupo A — curva de aprendizaje.
# ===========================================================================


def test_learning_curve_returns_result_and_figure() -> None:
    """``plot_learning_curve`` devuelve un ``LearningCurveResult`` y una ``Figure``."""
    df = make_curve_dataset(n=240, n_classes=4, n_features=12, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    result, figure = plot_learning_curve(_rf(), df, splits, train_sizes=[0.3, 0.6, 1.0])
    assert isinstance(result, LearningCurveResult)
    assert isinstance(figure, Figure)
    assert result.scoring == "accuracy"
    assert result.train_sizes_abs.shape == (3,)
    assert result.train_scores_mean.shape == (3,)
    assert result.val_scores_mean.shape == (3,)


def test_learning_curve_train_sizes_monotonic() -> None:
    """Los tamanos absolutos de la curva crecen de forma monotona."""
    df = make_curve_dataset(n=300, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.1, 0.4, 0.7, 1.0])
    diffs = np.diff(result.train_sizes_abs)
    assert np.all(diffs > 0)


def test_learning_curve_val_score_le_train_typically() -> None:
    """En el tamano maximo la accuracy de validacion no supera a la de train."""
    df = make_curve_dataset(n=300, n_classes=4, n_features=12, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.4, 0.7, 1.0])
    # Holgura por ruido de muestreo: validacion <= train + epsilon.
    assert result.val_scores_mean[-1] <= result.train_scores_mean[-1] + 1e-6


def test_learning_curve_respects_max_samples() -> None:
    """``max_samples`` recorta el dataset usado por la curva (decision D7)."""
    df = make_curve_dataset(n=600, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.5, 1.0], max_samples=200)
    # El tamano de train mas grande no puede exceder el subset de 200 muestras.
    assert int(result.train_sizes_abs.max()) <= 200


def test_learning_curve_default_train_sizes_has_seven_points() -> None:
    """Sin ``train_sizes`` explicito la curva usa las 7 fracciones por defecto."""
    df = make_curve_dataset(n=350, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(), df, splits)
    assert result.train_sizes_abs.shape == (7,)


def test_learning_curve_works_with_xgb() -> None:
    """La curva de aprendizaje tambien funciona con un estimador XGBoost."""
    df = make_curve_dataset(n=240, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    result, figure = plot_learning_curve(_xgb(), df, splits, train_sizes=[0.5, 1.0])
    assert isinstance(result, LearningCurveResult)
    assert isinstance(figure, Figure)


# ===========================================================================
# Grupo B — curva de validacion.
# ===========================================================================


def test_validation_curve_rf_max_depth() -> None:
    """Curva de validacion de RF sobre ``max_depth`` con ``None`` en el rango."""
    df = make_curve_dataset(n=240, n_classes=4, n_features=12, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    result, figure = plot_validation_curve(_rf(), df, "max_depth", [3, 5, 10, None], splits)
    assert isinstance(result, ValidationCurveResult)
    assert isinstance(figure, Figure)
    assert result.param_name == "max_depth"
    assert result.param_range == [3, 5, 10, None]
    assert result.val_scores_mean.shape == (4,)


def test_validation_curve_xgb_n_estimators() -> None:
    """Curva de validacion de XGB sobre ``n_estimators``."""
    df = make_curve_dataset(n=240, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    result, _ = plot_validation_curve(_xgb(), df, "n_estimators", [20, 40, 80], splits)
    assert result.param_name == "n_estimators"
    assert result.val_scores_mean.shape == (3,)


def test_validation_curve_xgb_learning_rate() -> None:
    """Curva de validacion de XGB sobre ``learning_rate``."""
    df = make_curve_dataset(n=240, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    result, _ = plot_validation_curve(_xgb(), df, "learning_rate", [0.05, 0.1, 0.3], splits)
    assert result.param_name == "learning_rate"
    assert result.train_scores_mean.shape == (3,)


def test_validation_curve_returns_figure() -> None:
    """``plot_validation_curve`` devuelve siempre una ``Figure``."""
    df = make_curve_dataset(n=200, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    _, figure = plot_validation_curve(_rf(), df, "max_depth", [3, 8], splits)
    assert isinstance(figure, Figure)


def test_validation_curve_param_range_length_matches() -> None:
    """Las curvas tienen un punto por cada valor del ``param_range``."""
    df = make_curve_dataset(n=240, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    param_range = [2, 4, 6, 10, None]
    result, _ = plot_validation_curve(_rf(), df, "max_depth", param_range, splits)
    assert len(result.param_range) == len(param_range)
    assert result.train_scores_mean.shape == (len(param_range),)
    assert result.val_scores_mean.shape == (len(param_range),)


def test_validation_curve_empty_range_raises() -> None:
    """Un ``param_range`` vacio lanza ``ValueError``."""
    df = make_curve_dataset(n=120, n_classes=3, n_features=8, separability="clean")
    splits = make_cv_splits(df.height, k=3)
    with pytest.raises(ValueError, match="param_range"):
        plot_validation_curve(_rf(), df, "max_depth", [], splits)


# ===========================================================================
# Grupo C — diagnostico de sub/sobreajuste.
# ===========================================================================


def _curve_result(train: list[float], val: list[float]) -> LearningCurveResult:
    """Construye un ``LearningCurveResult`` sintetico para tests de ``diagnose_fit``."""
    return LearningCurveResult(
        train_sizes_abs=np.arange(1, len(train) + 1, dtype=np.int64),
        train_scores_mean=np.asarray(train, dtype=np.float64),
        train_scores_std=np.zeros(len(train), dtype=np.float64),
        val_scores_mean=np.asarray(val, dtype=np.float64),
        val_scores_std=np.zeros(len(val), dtype=np.float64),
        scoring="accuracy",
    )


def test_diagnose_fit_overfit_when_gap_above_threshold() -> None:
    """Un gap train-val > 0.10 produce el veredicto ``overfit``."""
    result = _curve_result([0.80, 0.90, 0.98], [0.70, 0.74, 0.78])
    diag = diagnose_fit(result)
    assert isinstance(diag, FitDiagnosis)
    assert diag.verdict == "overfit"
    assert diag.gap > 0.10


def test_diagnose_fit_underfit_when_both_low() -> None:
    """Train y val ambos por debajo de 0.65 producen ``underfit``."""
    result = _curve_result([0.40, 0.45, 0.48], [0.38, 0.42, 0.44])
    diag = diagnose_fit(result)
    assert diag.verdict == "underfit"
    assert diag.train_acc_max < 0.65
    assert diag.val_acc_max < 0.65


def test_diagnose_fit_good_fit_otherwise() -> None:
    """Un gap pequeno y accuracy de validacion alta producen ``good_fit``."""
    result = _curve_result([0.84, 0.87, 0.89], [0.80, 0.82, 0.84])
    diag = diagnose_fit(result)
    assert diag.verdict == "good_fit"
    assert diag.gap <= 0.10


def test_diagnose_fit_thresholds_parametric() -> None:
    """Los umbrales son parametricos: cambiarlos altera el veredicto."""
    result = _curve_result([0.84, 0.87, 0.89], [0.80, 0.82, 0.84])
    # Con el umbral por defecto (0.10) el gap de 0.05 es good_fit.
    assert diagnose_fit(result).verdict == "good_fit"
    # Bajando el umbral de gap a 0.02, el mismo gap pasa a overfit.
    assert diagnose_fit(result, gap_threshold=0.02).verdict == "overfit"
    # Subiendo el umbral de accuracy baja a 0.95, el mismo modelo es underfit.
    assert diagnose_fit(result, low_acc_threshold=0.95).verdict == "underfit"


def test_diagnose_fit_explanation_mentions_verdict() -> None:
    """La explicacion textual menciona el veredicto y el gap numerico."""
    result = _curve_result([0.95, 0.97, 0.99], [0.60, 0.62, 0.64])
    diag = diagnose_fit(result)
    assert diag.verdict == "overfit"
    assert "obreajuste" in diag.explanation
    assert f"{diag.gap:.3f}" in diag.explanation


def test_diagnose_fit_empty_curve_raises() -> None:
    """Una curva sin puntos lanza ``ValueError``."""
    empty = LearningCurveResult(
        train_sizes_abs=np.array([], dtype=np.int64),
        train_scores_mean=np.array([], dtype=np.float64),
        train_scores_std=np.array([], dtype=np.float64),
        val_scores_mean=np.array([], dtype=np.float64),
        val_scores_std=np.array([], dtype=np.float64),
        scoring="accuracy",
    )
    with pytest.raises(ValueError, match="no points to diagnose"):
        diagnose_fit(empty)


def test_diagnose_fit_underfit_end_to_end() -> None:
    """Un RF de capacidad minima sobre datos casi-ruido se diagnostica ``underfit``."""
    df = make_curve_dataset(n=300, n_classes=6, n_features=12, separability="low")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(
        _rf(max_depth=1, n_estimators=5), df, splits, train_sizes=[0.3, 0.6, 1.0]
    )
    assert diagnose_fit(result).verdict == "underfit"


def test_diagnose_fit_overfit_end_to_end() -> None:
    """Un RF sin poda sobre pocas muestras y muchas features se diagnostica ``overfit``."""
    df = make_curve_dataset(n=120, n_classes=4, n_features=40, separability="memorizable")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(max_depth=None), df, splits, train_sizes=[0.3, 0.6, 1.0])
    assert diagnose_fit(result).verdict == "overfit"


def test_diagnose_fit_good_fit_end_to_end() -> None:
    """Un RF sobre datos bien separados se diagnostica ``good_fit``."""
    df = make_curve_dataset(n=300, n_classes=4, n_features=12, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.3, 0.6, 1.0])
    assert diagnose_fit(result).verdict == "good_fit"


# ===========================================================================
# Grupo D — CV espacial materializado y sin leakage.
# ===========================================================================


def test_spatial_splits_are_materialized_list() -> None:
    """``_materialize_cv_splits`` devuelve una ``list``, no un generador (D2, R2)."""
    splits_gen = (s for s in make_cv_splits(200, k=4))
    materialized = _materialize_cv_splits(splits_gen)
    assert isinstance(materialized, list)
    # Reutilizable: dos pasadas completas devuelven el mismo numero de folds.
    assert len(list(materialized)) == len(list(materialized)) == 4


def test_materialize_cv_splits_returns_index_tuples() -> None:
    """Cada split materializado es una tupla de arrays ``np.int64`` de indices."""
    materialized = _materialize_cv_splits(make_cv_splits(150, k=5))
    for train_idx, test_idx in materialized:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert train_idx.dtype == np.int64
        assert test_idx.dtype == np.int64


def test_materialize_cv_splits_empty_raises() -> None:
    """Una secuencia de splits vacia lanza ``ValueError``."""
    with pytest.raises(ValueError, match="empty"):
        _materialize_cv_splits([])


def test_materialize_cv_splits_empty_fold_raises() -> None:
    """Un fold sin muestras de train o test lanza ``ValueError``."""
    bad = [(np.array([], dtype=np.int64), np.array([1, 2], dtype=np.int64))]
    with pytest.raises(ValueError, match="has no samples"):
        _materialize_cv_splits(bad)


def test_curves_use_spatial_splits_not_random() -> None:
    """El ``cv`` de la curva es la lista de splits espaciales, no un entero aleatorio."""
    df = make_curve_dataset(n=240, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    # `splits` es una lista de tuplas de indices (la forma que exige D2/AC-3).
    assert isinstance(splits, list)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in splits)
    # La curva consume esa lista sin error (no recibe un `cv` entero).
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.5, 1.0])
    assert result.train_sizes_abs.size == 2


def test_spatial_cv_no_neighbor_leakage() -> None:
    """Train y test de cada fold espacial son disjuntos (sin leakage de indices)."""
    splits = _materialize_cv_splits(make_cv_splits(300, k=5))
    for train_idx, test_idx in splits:
        overlap = np.intersect1d(train_idx, test_idx)
        assert overlap.size == 0


def test_learning_curve_subsample_remaps_cv_splits() -> None:
    """Con ``max_samples`` la curva realinea los splits sin indices fuera de rango."""
    df = make_curve_dataset(n=500, n_classes=4, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=5)
    # No debe lanzar IndexError pese a que los splits indexan hasta 499.
    result, _ = plot_learning_curve(_rf(), df, splits, train_sizes=[0.5, 1.0], max_samples=150)
    assert int(result.train_sizes_abs.max()) <= 150


# ===========================================================================
# Grupo E — propiedades de los plots.
# ===========================================================================


def test_curve_plots_dpi_is_200(tmp_path: Path) -> None:
    """Las figuras de las curvas se guardan a dpi >= 200 (criterio AC-8)."""
    df = make_curve_dataset(n=200, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    _, figure = plot_learning_curve(_rf(), df, splits, train_sizes=[0.5, 1.0])
    assert figure.dpi >= 200
    out = tmp_path / "lc.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    assert out.exists() and out.stat().st_size > 0


def test_curve_plots_have_std_band() -> None:
    """La figura de la curva dibuja una banda sombreada (``fill_between``)."""
    df = make_curve_dataset(n=200, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    _, figure = plot_learning_curve(_rf(), df, splits, train_sizes=[0.5, 1.0])
    ax = figure.axes[0]
    # `fill_between` agrega una PolyCollection a las colecciones del eje.
    from matplotlib.collections import PolyCollection

    assert any(isinstance(c, PolyCollection) for c in ax.collections)


def test_validation_curve_plot_handles_none_label() -> None:
    """La figura de la curva de validacion rotula ``None`` como categoria (R4)."""
    df = make_curve_dataset(n=200, n_classes=3, n_features=10, separability="clean")
    splits = make_cv_splits(df.height, k=4)
    _, figure = plot_validation_curve(_rf(), df, "max_depth", [3, 8, None], splits)
    ax = figure.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "None" in labels


# ===========================================================================
# Grupo F — integracion: notebook seccion 5b + doc del spatial CV.
# ===========================================================================


def test_notebook_has_section_5b() -> None:
    """El notebook ``04_baseline.ipynb`` contiene la seccion 5b de curvas."""
    nb_path = _REPO_ROOT / "notebooks" / "04_baseline.ipynb"
    if not nb_path.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado.")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    assert "5b" in text
    assert "Curvas de aprendizaje" in text
    # No debe quedar el placeholder de US-021 sin reemplazar.
    assert "completado por US-021" not in text


def test_notebook_5b_has_fit_diagnosis() -> None:
    """La seccion 5b del notebook usa ``diagnose_fit`` para el veredicto."""
    nb_path = _REPO_ROOT / "notebooks" / "04_baseline.ipynb"
    if not nb_path.exists():
        pytest.skip("notebooks/04_baseline.ipynb aun no generado.")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    assert "diagnose_fit" in text
    assert "plot_learning_curve" in text
    assert "plot_validation_curve" in text


def test_spatial_cv_doc_exists_and_cites_references() -> None:
    """``docs/spatial_cv_baseline.md`` existe y cita las referencias academicas."""
    doc_path = _REPO_ROOT / "docs" / "spatial_cv_baseline.md"
    assert doc_path.exists(), "Falta docs/spatial_cv_baseline.md (AC-5)."
    content = doc_path.read_text(encoding="utf-8")
    assert "Lyons" in content
    assert "Roberts" in content
    # Debe explicar el criterio H3 + KMeans + buffer.
    assert "H3" in content
    assert "buffer" in content.lower()
    assert "leakage" in content.lower()


# ===========================================================================
# Grupo G — historial de loss desde MLflow (modelos temporales).
# ===========================================================================


class _FakeMetric:
    """Stub de mlflow.entities.Metric con los atributos que usa el helper."""

    def __init__(self, step: int, value: float) -> None:
        self.step = step
        self.value = value


def _make_fake_mlflow_client(
    *,
    train_per_fold: list[list[float]],
    val_per_fold: list[list[float]] | None,
):
    """Construye un stub de :class:`MlflowClient` con metric history controlada.

    Args:
        train_per_fold: Lista de listas; entrada ``i`` es la curva de
            ``fold{i}_train_loss`` por epoca.
        val_per_fold: Idem para ``fold{i}_val_loss``; ``None`` deshabilita val.
    """
    train_keys = [f"fold{i}_train_loss" for i in range(len(train_per_fold))]
    val_keys = (
        [f"fold{i}_val_loss" for i in range(len(val_per_fold))] if val_per_fold is not None else []
    )
    metric_summary = {k: 0.0 for k in train_keys + val_keys}
    history_map: dict[str, list[_FakeMetric]] = {}
    for key, curve in zip(train_keys, train_per_fold, strict=True):
        history_map[key] = [_FakeMetric(step, v) for step, v in enumerate(curve)]
    if val_per_fold is not None:
        for key, curve in zip(val_keys, val_per_fold, strict=True):
            history_map[key] = [_FakeMetric(step, v) for step, v in enumerate(curve)]

    class _Run:
        class _Data:
            metrics = metric_summary

        data = _Data()

    class _FakeClient:
        def __init__(self, *, tracking_uri: str | None = None) -> None:
            self.tracking_uri = tracking_uri

        def get_run(self, run_id: str) -> _Run:
            return _Run()

        def get_metric_history(self, run_id: str, key: str) -> list[_FakeMetric]:
            return history_map.get(key, [])

    return _FakeClient


def test_fetch_loss_history_aggregates_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fetch_loss_history_from_mlflow`` agrega los folds en media y std."""
    from ml.eval import learning_curves

    fake_client = _make_fake_mlflow_client(
        train_per_fold=[[1.0, 0.5, 0.2], [1.2, 0.6, 0.3]],
        val_per_fold=[[1.1, 0.6, 0.4], [1.3, 0.7, 0.5]],
    )

    # Bypass mlflow.set_tracking_uri y resolve_tracking_uri en el modulo
    # bajo test (no toca el server real).
    monkeypatch.setattr(
        "ml.eval.learning_curves.MlflowClient" if False else "mlflow.tracking.MlflowClient",
        fake_client,
        raising=False,
    )

    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    history = learning_curves.fetch_loss_history_from_mlflow(
        run_id="fake-run-id",
        model_kind="tempcnn",
        tracking_uri="file:///tmp/fake-mlruns",
    )
    assert history.model_kind == "tempcnn"
    assert history.run_id == "fake-run-id"
    assert history.n_folds == 2
    np.testing.assert_array_equal(history.epochs, np.array([0, 1, 2]))
    # Media de los dos folds en train.
    np.testing.assert_allclose(
        history.train_loss_mean,
        np.array([(1.0 + 1.2) / 2, (0.5 + 0.6) / 2, (0.2 + 0.3) / 2]),
        atol=1e-9,
    )
    assert history.train_loss_std.size == 3
    assert history.val_loss_mean.size == 3


def test_fetch_loss_history_raises_when_no_train_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si el run no expone metricas `fold{i}_train_loss`, lanza RuntimeError."""
    from ml.eval import learning_curves

    fake_client = _make_fake_mlflow_client(
        train_per_fold=[],
        val_per_fold=None,
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", fake_client, raising=False)

    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    with pytest.raises(RuntimeError, match="does not expose"):
        learning_curves.fetch_loss_history_from_mlflow(
            run_id="empty-run",
            model_kind="inceptiontime",
            tracking_uri="file:///tmp/fake-mlruns",
        )


def test_plot_loss_history_returns_figure() -> None:
    """``plot_loss_history_from_mlflow`` devuelve una ``Figure`` con dpi 200."""
    from ml.eval.learning_curves import (
        TemporalLossHistory,
        plot_loss_history_from_mlflow,
    )

    history = TemporalLossHistory(
        model_kind="tempcnn",
        run_id="r1",
        epochs=np.arange(5),
        train_loss_mean=np.array([1.0, 0.7, 0.5, 0.3, 0.2]),
        train_loss_std=np.array([0.05, 0.04, 0.03, 0.03, 0.02]),
        val_loss_mean=np.array([1.1, 0.8, 0.6, 0.4, 0.35]),
        val_loss_std=np.array([0.05, 0.04, 0.04, 0.03, 0.03]),
        n_folds=3,
    )
    fig = plot_loss_history_from_mlflow(history)
    assert isinstance(fig, Figure)
    assert fig.get_dpi() == 200


def test_plot_loss_history_raises_when_empty() -> None:
    """``plot_loss_history_from_mlflow`` falla si el historial esta vacio."""
    from ml.eval.learning_curves import (
        TemporalLossHistory,
        plot_loss_history_from_mlflow,
    )

    history = TemporalLossHistory(
        model_kind="tempcnn",
        run_id="r1",
        epochs=np.array([], dtype=np.int64),
        train_loss_mean=np.array([], dtype=np.float64),
        train_loss_std=np.array([], dtype=np.float64),
        val_loss_mean=np.array([], dtype=np.float64),
        val_loss_std=np.array([], dtype=np.float64),
        n_folds=0,
    )
    with pytest.raises(ValueError, match="empty"):
        plot_loss_history_from_mlflow(history)


def test_diagnose_temporal_fit_overfit() -> None:
    """Loss train baja, val_loss sube al final -> overfit."""
    from ml.eval.learning_curves import (
        TemporalLossHistory,
        diagnose_temporal_fit,
    )

    history = TemporalLossHistory(
        model_kind="tempcnn",
        run_id="r1",
        epochs=np.arange(5),
        train_loss_mean=np.array([1.0, 0.7, 0.4, 0.2, 0.1]),
        train_loss_std=np.zeros(5),
        val_loss_mean=np.array([1.1, 0.8, 0.6, 0.7, 0.9]),
        val_loss_std=np.zeros(5),
        n_folds=1,
    )
    diag = diagnose_temporal_fit(history, gap_threshold=0.10)
    assert diag.verdict == "overfit"
    # gap = val_loss - train_loss en el ultimo punto
    assert diag.gap == pytest.approx(0.9 - 0.1, abs=1e-9)


def test_diagnose_temporal_fit_underfit() -> None:
    """Ambas losses altas y similares -> underfit."""
    from ml.eval.learning_curves import (
        TemporalLossHistory,
        diagnose_temporal_fit,
    )

    history = TemporalLossHistory(
        model_kind="inceptiontime",
        run_id="r2",
        epochs=np.arange(3),
        train_loss_mean=np.array([2.5, 2.3, 2.2]),
        train_loss_std=np.zeros(3),
        val_loss_mean=np.array([2.6, 2.4, 2.25]),
        val_loss_std=np.zeros(3),
        n_folds=1,
    )
    diag = diagnose_temporal_fit(history, high_loss_threshold=1.5)
    assert diag.verdict == "underfit"


def test_diagnose_temporal_fit_good_fit() -> None:
    """Loss train baja, val_loss baja y cerca -> good_fit."""
    from ml.eval.learning_curves import (
        TemporalLossHistory,
        diagnose_temporal_fit,
    )

    history = TemporalLossHistory(
        model_kind="tempcnn",
        run_id="r3",
        epochs=np.arange(4),
        train_loss_mean=np.array([1.0, 0.6, 0.4, 0.3]),
        train_loss_std=np.zeros(4),
        val_loss_mean=np.array([1.1, 0.7, 0.45, 0.35]),
        val_loss_std=np.zeros(4),
        n_folds=1,
    )
    diag = diagnose_temporal_fit(history, gap_threshold=0.10, high_loss_threshold=1.5)
    assert diag.verdict == "good_fit"
