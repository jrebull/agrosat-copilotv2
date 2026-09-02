"""Tests smoke de ml.eval.metrics (US-019).

Conjunto minimo de validacion del modulo de metricas del baseline. La
suite exhaustiva (~14 tests) la completa el sub-agente de tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from ml.eval.metrics import (
    classification_report_text,
    compute_baseline_metrics,
    confusion_matrix_figure,
)

_EXPECTED_KEYS = {"f1_macro", "f1_weighted", "miou", "accuracy", "cohen_kappa"}


def test_compute_metrics_returns_five_keys() -> None:
    """Las cinco metricas exactas del criterio AC-3 estan presentes."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    metrics = compute_baseline_metrics(y_true, y_pred)
    assert set(metrics.keys()) == _EXPECTED_KEYS
    assert all(isinstance(v, float) for v in metrics.values())


def test_compute_metrics_perfect_prediction() -> None:
    """Prediccion perfecta da 1.0 en las cuatro metricas acotadas."""
    y = np.array([0, 1, 2, 3, 0, 1])
    metrics = compute_baseline_metrics(y, y)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)


def test_compute_metrics_miou_matches_jaccard_macro() -> None:
    """La mIoU coincide con jaccard_score(average='macro')."""
    from sklearn.metrics import jaccard_score

    y_true = np.array([0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0])
    metrics = compute_baseline_metrics(y_true, y_pred)
    expected = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    assert metrics["miou"] == pytest.approx(expected)


def test_compute_metrics_ranges_valid() -> None:
    """Las metricas acotadas viven en [0, 1]."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    metrics = compute_baseline_metrics(y_true, y_pred)
    for key in ("f1_macro", "f1_weighted", "miou", "accuracy"):
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_metrics_length_mismatch_raises() -> None:
    """Vectores de distinta longitud lanzan ValueError."""
    with pytest.raises(ValueError, match="must have the same shape"):
        compute_baseline_metrics(np.array([0, 1]), np.array([0, 1, 2]))


def test_confusion_matrix_figure_returns_figure() -> None:
    """confusion_matrix_figure devuelve una matplotlib Figure."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    fig = confusion_matrix_figure(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_classification_report_text_includes_class_names() -> None:
    """El reporte usa los nombres de clase suministrados."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = classification_report_text(y_true, y_pred, class_names={0: "trigo", 1: "maiz"})
    assert "trigo" in report
    assert "maiz" in report
