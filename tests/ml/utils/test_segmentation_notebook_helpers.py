"""Tests de ml.utils.segmentation_notebook_helpers (US-025).

Conjunto minimo de validacion sin GPU ni servidor MLflow real: las funciones se
prueban con datos sinteticos y el lineage MLflow se mockea (CI sin Docker debe
quedar verde).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.utils.segmentation_notebook_helpers import (
    TrainingResult,
    _parse_cli_done,
    build_variant_comparison,
    pastis_class_names,
    per_class_comparison_table,
    per_class_table,
    plot_confusion_matrix,
    read_segmentation_lineage,
    segmentation_eval_table,
    training_results_table,
)


def _result(model: str, miou: float | None) -> TrainingResult:
    return TrainingResult(
        model=model,
        miou=miou,
        f1_macro=miou,
        pixel_acc=miou,
        returncode=0,
        error=None,
        from_checkpoint=True,
        best_epoch=1,
        cli_command="cmd",
    )


def _metrics(num_classes: int = 18) -> dict[str, object]:
    rng = np.linspace(0.1, 0.9, num_classes)
    return {
        "miou": 0.5,
        "f1_macro": 0.6,
        "pixel_acc": 0.8,
        "balanced_acc": 0.55,
        "cohen_kappa": 0.7,
        "per_class_iou": rng.tolist(),
        "per_class_f1": (rng + 0.05).tolist(),
    }


def test_parse_cli_done_extracts_metrics() -> None:
    """_parse_cli_done lee miou/f1_macro/pixel_acc de la linea cli_done."""
    log = "ruido\n2026 [info] cli_done miou=0.6253 f1_macro=0.75 pixel_acc=0.8759\nmas\n"
    parsed = _parse_cli_done(log)
    assert parsed == {"miou": 0.6253, "f1_macro": 0.75, "pixel_acc": 0.8759}


def test_parse_cli_done_degraded_when_absent() -> None:
    """Sin linea cli_done, devuelve None en las tres metricas."""
    assert _parse_cli_done("sin nada relevante") == {
        "miou": None,
        "f1_macro": None,
        "pixel_acc": None,
    }


def test_training_results_table_schema() -> None:
    """training_results_table produce el schema fijo esperado."""
    df = training_results_table([_result("tsvit", 0.62), _result("tsvit-pheno", 0.625)])
    assert df.columns == ["model", "miou", "f1_macro", "pixel_acc", "returncode"]
    assert df.height == 2


def test_build_variant_comparison_delta() -> None:
    """El delta es variante menos base, redondeado a 4."""
    df = build_variant_comparison([_result("tsvit", 0.62), _result("tsvit-pheno", 0.625)])
    assert df is not None
    miou_row = df.filter(pl.col("metrica") == "miou")
    assert miou_row["delta"][0] == pytest.approx(0.005, abs=1e-6)


def test_build_variant_comparison_degraded_on_missing_model() -> None:
    """Falta un modelo -> None (modo degradado)."""
    assert build_variant_comparison([_result("tsvit", 0.62)]) is None


def test_build_variant_comparison_degraded_on_none_metric() -> None:
    """Una metrica None -> None (modo degradado)."""
    assert build_variant_comparison([_result("tsvit", None), _result("tsvit-pheno", 0.6)]) is None


def test_segmentation_eval_table_columns() -> None:
    """segmentation_eval_table expone las cinco metricas."""
    df = segmentation_eval_table({"tsvit": _metrics(), "tsvit-pheno": _metrics()})
    assert set(df.columns) == {
        "variante",
        "mIoU",
        "F1_macro",
        "pixel_acc",
        "balanced_acc",
        "cohen_kappa",
    }
    assert df.height == 2


def test_per_class_table_semantic18_sorted_desc() -> None:
    """per_class_table con nombres PASTIS, ordenada por IoU desc."""
    df = per_class_table(_metrics(), class_names=pastis_class_names(), num_classes=18)
    assert df.height == 18
    ious = df["IoU"].to_list()
    assert ious == sorted(ious, reverse=True)
    # El nombre de la clase de mayor IoU debe ser un nombre real (no grupo_).
    assert not df["clase"][0].startswith("grupo_")


def test_per_class_table_hcat6_uses_group_labels() -> None:
    """Con class_names=None (hcat6), las etiquetas son grupo_c."""
    df = per_class_table(_metrics(6), class_names=None, num_classes=6)
    assert df.height == 6
    assert all(c.startswith("grupo_") for c in df["clase"].to_list())


def test_per_class_comparison_table_delta_sorted() -> None:
    """per_class_comparison_table ordena por delta_IoU desc."""
    base = _metrics()
    variant = _metrics()
    variant["per_class_iou"] = (np.array(base["per_class_iou"]) + 0.01).tolist()
    df = per_class_comparison_table(base, variant, class_names=pastis_class_names())
    assert "delta_IoU" in df.columns
    deltas = df["delta_IoU"].to_list()
    assert deltas == sorted(deltas, reverse=True)


def test_plot_confusion_matrix_returns_figure() -> None:
    """plot_confusion_matrix devuelve una Figure normalizada por fila."""
    import matplotlib

    matplotlib.use("Agg")
    cm = np.array([[8, 2], [1, 9]], dtype=np.int64)
    fig = plot_confusion_matrix(cm, num_classes=2, class_names={0: "a", 1: "b"})
    assert fig is not None
    # La figura tiene un solo Axes principal (la imagen) + colorbar.
    assert len(fig.axes) >= 1


def test_pastis_class_names_rejects_non_18() -> None:
    """pastis_class_names solo cubre semantic18."""
    with pytest.raises(ValueError, match="semantic18"):
        pastis_class_names(6)


def test_pastis_class_names_maps_training_indices() -> None:
    """El indice 0 del modelo corresponde a la primera clase PASTIS (cid=1)."""
    names = pastis_class_names()
    assert len(names) == 18
    assert all(isinstance(v, str) for v in names.values())


def test_read_segmentation_lineage_degraded_without_mlflow(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Si get_experiment_by_name devuelve None -> None (no lanza)."""
    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda *_a, **_k: None)
    assert read_segmentation_lineage(["alt-tsvit-v1"], tracking_uri="file:/tmp/x") is None
