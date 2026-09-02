"""Tests de las metricas densas de segmentacion y los helpers de prototipos.

Cubre las funciones nuevas de ``ml.eval.metrics`` (US-025, EPIC 5):
``dense_miou``, ``dense_f1_macro``, ``dense_pixel_accuracy`` y
``segmentation_metrics_report``; y los helpers de
``ml.features.phenology_class_prototypes`` (``load_class_names`` y
``load_class_prototype_embeddings``) contra el parquet real ya generado.

Casos deterministas con tensores conocidos (prediccion perfecta, todo-una-clase,
ignore_index). Acepta ``numpy`` y ``torch`` indistintamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.eval.metrics import (
    dense_confusion_matrix,
    dense_f1_macro,
    dense_miou,
    dense_pixel_accuracy,
    segmentation_metrics_report,
)
from ml.features.phenology_class_prototypes import (
    load_class_names,
    load_class_prototype_embeddings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_PROTO = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"
_CLASS_MAP = _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"


# ---------------------------------------------------------------------------
# dense_miou
# ---------------------------------------------------------------------------


def test_dense_miou_perfect_is_one() -> None:
    """Prediccion identica al GT -> mIoU == 1.0."""
    true = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]], dtype=np.int64)
    pred = true.copy()
    assert dense_miou(pred[None], true[None], n_classes=6) == pytest.approx(1.0)


def test_dense_miou_all_one_class_known_value() -> None:
    """Predecir siempre la clase 0 sobre un GT 50/50 da un valor bajo conocido.

    GT mitad clase 0, mitad clase 1; prediccion todo 0.
      - clase 0: interseccion = #(true=0) = N/2; union = N (todos predichos 0
        + los true=0) -> IoU = (N/2)/N = 0.5.
      - clase 1: interseccion = 0 -> IoU = 0.
    mIoU = (0.5 + 0.0)/2 = 0.25.
    """
    true = np.array([0, 0, 1, 1], dtype=np.int64).reshape(1, 2, 2)
    pred = np.zeros_like(true)
    assert dense_miou(pred, true, n_classes=2) == pytest.approx(0.25)


def test_dense_miou_accepts_logits() -> None:
    """Acepta logits ``(B,C,H,W)`` aplicando argmax internamente."""
    true = np.array([[0, 1], [1, 0]], dtype=np.int64)[None]
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    # Logit del canal correcto mas alto en cada pixel -> argmax == true.
    for i in range(2):
        for j in range(2):
            logits[0, true[0, i, j], i, j] = 10.0
    assert dense_miou(logits, true, n_classes=2) == pytest.approx(1.0)


def test_dense_miou_accepts_torch_tensors() -> None:
    """Funciona con tensores ``torch`` (mueve a CPU/numpy internamente)."""
    true = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    pred = true.clone()
    assert dense_miou(pred, true, n_classes=4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ignore_index
# ---------------------------------------------------------------------------


def test_ignore_index_excluded() -> None:
    """Los pixeles ``ignore_index`` en el GT no entran en ninguna metrica.

    GT con la mitad en ``ignore_index`` y la otra mitad bien predicha: mIoU,
    F1 y pixel-acc deben ser 1.0 (los ignorados no penalizan ni cuentan).
    """
    true = np.array([[0, 1], [255, 255]], dtype=np.int64)[None]
    pred = np.array([[0, 1], [7, 9]], dtype=np.int64)[None]  # basura en ignorados
    assert dense_miou(pred, true, n_classes=2, ignore_index=255) == pytest.approx(1.0)
    assert dense_f1_macro(pred, true, n_classes=2, ignore_index=255) == pytest.approx(1.0)
    assert dense_pixel_accuracy(pred, true, n_classes=2, ignore_index=255) == pytest.approx(1.0)


def test_confusion_matrix_excludes_ignore() -> None:
    """La matriz de confusion densa no cuenta pixeles ignorados."""
    true = np.array([[0, 0], [255, 255]], dtype=np.int64)[None]
    pred = np.array([[0, 1], [0, 1]], dtype=np.int64)[None]
    cm = dense_confusion_matrix(pred, true, n_classes=2, ignore_index=255)
    # Solo cuentan los 2 pixeles de la fila 0: true=0 pred=0 y true=0 pred=1.
    assert int(cm.sum()) == 2
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1


def test_all_ignored_returns_zero() -> None:
    """Un GT enteramente ignorado devuelve 0.0 (no NaN) en todas las metricas."""
    true = np.full((1, 4, 4), 255, dtype=np.int64)
    pred = np.zeros((1, 4, 4), dtype=np.int64)
    assert dense_miou(pred, true, n_classes=3, ignore_index=255) == 0.0
    assert dense_f1_macro(pred, true, n_classes=3, ignore_index=255) == 0.0
    assert dense_pixel_accuracy(pred, true, n_classes=3, ignore_index=255) == 0.0


# ---------------------------------------------------------------------------
# segmentation_metrics_report
# ---------------------------------------------------------------------------


def test_report_keys_and_types() -> None:
    """El reporte devuelve ``miou/f1_macro/pixel_acc/per_class_iou``."""
    true = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)[None]
    pred = true.copy()
    report = segmentation_metrics_report(pred, true, n_classes=3)
    assert set(report.keys()) == {"miou", "f1_macro", "pixel_acc", "per_class_iou"}
    assert isinstance(report["miou"], float)
    assert isinstance(report["f1_macro"], float)
    assert isinstance(report["pixel_acc"], float)
    assert isinstance(report["per_class_iou"], list)
    assert len(report["per_class_iou"]) == 3


def test_report_perfect_prediction() -> None:
    """Prediccion perfecta -> todas las metricas 1.0 y per_class_iou == 1.0."""
    true = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)[None]
    report = segmentation_metrics_report(true.copy(), true, n_classes=3)
    assert report["miou"] == pytest.approx(1.0)
    assert report["f1_macro"] == pytest.approx(1.0)
    assert report["pixel_acc"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in report["per_class_iou"])


def test_report_absent_class_is_none() -> None:
    """Una clase ausente de GT y prediccion aparece como ``None`` en per_class_iou."""
    # n_classes=4 pero solo aparecen 0 y 1; clases 2 y 3 -> None.
    true = np.array([[0, 1], [1, 0]], dtype=np.int64)[None]
    report = segmentation_metrics_report(true.copy(), true, n_classes=4)
    assert report["per_class_iou"][2] is None
    assert report["per_class_iou"][3] is None
    assert report["per_class_iou"][0] == pytest.approx(1.0)


def test_report_matches_individual_functions() -> None:
    """El reporte coincide con las funciones individuales (una sola pasada)."""
    rng = np.random.default_rng(0)
    true = rng.integers(0, 6, size=(2, 16, 16)).astype(np.int64)
    pred = rng.integers(0, 6, size=(2, 16, 16)).astype(np.int64)
    report = segmentation_metrics_report(pred, true, n_classes=6)
    assert report["miou"] == pytest.approx(dense_miou(pred, true, n_classes=6))
    assert report["f1_macro"] == pytest.approx(dense_f1_macro(pred, true, n_classes=6))
    assert report["pixel_acc"] == pytest.approx(dense_pixel_accuracy(pred, true, n_classes=6))


def test_pixel_accuracy_known_fraction() -> None:
    """Pixel accuracy = fraccion de pixeles validos correctos."""
    true = np.array([0, 1, 2, 3], dtype=np.int64).reshape(1, 2, 2)
    pred = np.array([0, 1, 2, 0], dtype=np.int64).reshape(1, 2, 2)  # 3/4 correctos
    assert dense_pixel_accuracy(pred, true, n_classes=4) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# phenology_class_prototypes helpers (parquet real)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CLASS_MAP.exists(), reason="pastis_class_mapping.json no presente.")
def test_load_class_names() -> None:
    """``load_class_names`` mapea los 18 cultivos a nombres legibles."""
    names = load_class_names()
    assert 1 in names
    assert isinstance(names[1], str)
    # Clase 1 es Meadow en el mapping canonico.
    assert names[1] == "Meadow"
    # Cubre las 18 clases agronomicas.
    assert all(c in names for c in range(1, 19))


@pytest.mark.skipif(
    not _REAL_PROTO.exists(),
    reason="phenology_class_prototypes_pastis.parquet no presente.",
)
def test_load_class_prototype_embeddings_shape() -> None:
    """``load_class_prototype_embeddings`` devuelve ``(18,384)`` + 18 class_ids."""
    protos, class_ids = load_class_prototype_embeddings(_REAL_PROTO)
    assert protos.shape == (18, 384)
    assert protos.dtype == np.float32
    assert len(class_ids) == 18
    assert class_ids == list(range(1, 19))
    assert np.isfinite(protos).all()
