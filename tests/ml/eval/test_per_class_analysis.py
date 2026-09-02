"""Tests del analisis de cardinalidad por clase sobre el ensamble final.

Usa arrays/CM sinteticos en el espacio contiguo [0..17] (sin GPU, sin PASTIS).
Cubre: tabla por clase correcta, curva top-K monotona en K, recomendacion que
cruza F1 + banda de soporte, y manejo de clases ausentes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.eval.per_class_analysis import (
    cardinality_cutoff_curve,
    honest_class_dropout_curve,
    per_class_report,
    recommend_classes_to_drop,
)


def test_per_class_report_perfect_prediction() -> None:
    """Prediccion perfecta -> precision = recall = f1 = iou = 1 en clases presentes."""
    y_true = np.array([0, 0, 1, 2, 2, 2])
    report = per_class_report(y_true, y_true, num_classes=3)

    assert report.height == 3
    present = report.filter(pl.col("support") > 0)
    assert present.height == 3
    for col in ("precision", "recall", "f1", "iou"):
        values = present[col].to_list()
        assert all(abs(v - 1.0) < 1e-9 for v in values)
    # Soporte por clase correcto.
    by_id = {row["class_id"]: row["support"] for row in report.iter_rows(named=True)}
    assert by_id == {0: 2, 1: 1, 2: 3}


def test_per_class_report_contiguous_names() -> None:
    """El indice contiguo c mapea al nombre PASTIS_R_CLASSES[c+1]."""
    y_true = np.array([0, 1, 2])
    report = per_class_report(y_true, y_true, num_classes=18)
    by_id = {row["class_id"]: row["name"] for row in report.iter_rows(named=True)}
    # c=0 -> raw 1 = Meadow ; c=2 -> raw 3 = Corn ; c=17 -> raw 18 = Sorghum.
    assert by_id[0] == "Meadow"
    assert by_id[2] == "Corn"
    assert by_id[17] == "Sorghum"


def test_per_class_report_absent_class_is_null() -> None:
    """Una clase sin soporte tiene support=0 y metricas null, va al final."""
    # Clase 2 nunca aparece en y_true.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = per_class_report(y_true, y_pred, num_classes=3)

    absent = report.filter(pl.col("class_id") == 2)
    assert absent.height == 1
    row = absent.row(0, named=True)
    assert row["support"] == 0
    assert row["f1"] is None
    assert row["iou"] is None
    # Ordenado por f1 desc con nulls al final: la ultima fila es la ausente.
    assert report.row(report.height - 1, named=True)["class_id"] == 2


def test_per_class_report_sorted_by_f1_desc() -> None:
    """La tabla queda ordenada por f1 descendente (nulls al final)."""
    # Clase 0 perfecta, clase 1 con errores -> f1(0) > f1(1).
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    report = per_class_report(y_true, y_pred, num_classes=2)
    f1_order = [v for v in report["f1"].to_list() if v is not None]
    assert f1_order == sorted(f1_order, reverse=True)
    assert report.row(0, named=True)["class_id"] == 0


def test_per_class_report_ignore_index() -> None:
    """Los pixeles con target == ignore_index no afectan el reporte."""
    y_true = np.array([0, 1, 255, 255])
    y_pred = np.array([0, 1, 0, 1])
    report = per_class_report(y_true, y_pred, num_classes=2, ignore_index=255)
    by_id = {row["class_id"]: row for row in report.iter_rows(named=True)}
    # Solo cuentan los dos primeros, ambos correctos.
    assert by_id[0]["support"] == 1
    assert by_id[1]["support"] == 1
    assert abs(by_id[0]["f1"] - 1.0) < 1e-9


def test_per_class_report_with_proba_adds_ap() -> None:
    """Con proba se agrega la columna ap (Average Precision one-vs-rest)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # Probabilidades perfectas -> AP = 1 en ambas clases.
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=np.float64)
    report = per_class_report(y_true, y_pred, proba, num_classes=2)
    assert "ap" in report.columns
    aps = [v for v in report["ap"].to_list() if v is not None]
    assert all(abs(v - 1.0) < 1e-9 for v in aps)


def test_per_class_report_proba_shape_mismatch_raises() -> None:
    """Una proba con columnas incorrectas lanza ValueError."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    bad_proba = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="proba"):
        per_class_report(y_true, y_pred, bad_proba, num_classes=2)


def test_per_class_report_length_mismatch_raises() -> None:
    """y_true e y_pred de distinto tamano lanzan ValueError."""
    with pytest.raises(ValueError, match="same number"):
        per_class_report(np.array([0, 1]), np.array([0]), num_classes=2)


def _three_class_report() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Construye un reporte sintetico de 3 clases con F1 decreciente.

    Clase 0 perfecta, clase 1 buena, clase 2 mala -> f1(0) > f1(1) > f1(2).
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2], dtype=np.int64)
    report = per_class_report(y_true, y_pred, num_classes=3)
    return report, y_true, y_pred


def test_cardinality_curve_monotone_in_k() -> None:
    """macro_f1_topk es monotona no creciente y support_share no decreciente."""
    report, y_true, y_pred = _three_class_report()
    curve = cardinality_cutoff_curve(report, y_true, y_pred, num_classes=3)

    assert curve["k"].to_list() == [1, 2, 3]
    macro = curve["macro_f1_topk"].to_list()
    share = curve["cumulative_support_share"].to_list()
    # macro-F1 top-K monotona no creciente.
    assert all(macro[i] >= macro[i + 1] - 1e-12 for i in range(len(macro) - 1))
    # cuota de soporte acumulada monotona no decreciente, termina en 1.0.
    assert all(share[i] <= share[i + 1] + 1e-12 for i in range(len(share) - 1))
    assert abs(share[-1] - 1.0) < 1e-9


def test_cardinality_curve_kept_ids_best_first() -> None:
    """kept_class_ids retiene primero las clases con mejor F1."""
    report, y_true, y_pred = _three_class_report()
    curve = cardinality_cutoff_curve(report, y_true, y_pred, num_classes=3)
    # K=1 retiene la mejor clase (la primera del reporte ordenado por F1).
    best_id = report.row(0, named=True)["class_id"]
    assert curve.row(0, named=True)["kept_class_ids"] == [best_id]
    # K=3 retiene las 3 clases presentes.
    assert sorted(curve.row(2, named=True)["kept_class_ids"]) == [0, 1, 2]


def test_cardinality_curve_full_macro_matches_present_mean() -> None:
    """macro_f1_topk en K=n_present == media de F1 de las clases presentes."""
    report, y_true, y_pred = _three_class_report()
    curve = cardinality_cutoff_curve(report, y_true, y_pred, num_classes=3)
    present_f1 = [v for v in report["f1"].to_list() if v is not None]
    expected = float(np.mean(present_f1))
    assert abs(curve.row(curve.height - 1, named=True)["macro_f1_topk"] - expected) < 1e-9


def test_cardinality_curve_skips_absent_classes() -> None:
    """Las clases ausentes (f1 null) no entran en la curva."""
    # Solo clases 0 y 1 presentes de un espacio de 4.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = per_class_report(y_true, y_pred, num_classes=4)
    curve = cardinality_cutoff_curve(report, y_true, y_pred, num_classes=4)
    # Solo 2 clases presentes -> la curva tiene a lo sumo K=2.
    assert curve.height == 2
    assert curve["k"].to_list() == [1, 2]


def test_cardinality_curve_empty_when_no_f1() -> None:
    """Sin clases con F1 definido la curva sale vacia con el schema correcto."""
    # y_true todo ignore_index -> ninguna clase presente.
    y_true = np.array([255, 255])
    y_pred = np.array([0, 1])
    report = per_class_report(y_true, y_pred, num_classes=3, ignore_index=255)
    curve = cardinality_cutoff_curve(report, y_true, y_pred, num_classes=3, ignore_index=255)
    assert curve.height == 0
    assert "macro_f1_topk" in curve.columns


def _dist_report(bands: dict[int, str], n_parcels: dict[int, int]) -> pl.DataFrame:
    """Construye un dist_report sintetico (raw class_id, banda, n_parcels)."""
    rows = [
        {
            "class_id": cid,
            "support_band": bands[cid],
            "n_parcels": n_parcels[cid],
        }
        for cid in bands
    ]
    return pl.DataFrame(rows)


def test_recommend_drop_crosses_f1_and_support() -> None:
    """Solo se descartan clases con f1<umbral AND banda baja."""
    # Reporte: c0 buena, c1 mala, c2 mala (contiguo). raw = c+1.
    report = pl.DataFrame(
        {
            "class_id": [0, 1, 2],
            "name": ["a", "b", "c"],
            "support": [100, 50, 40],
            "precision": [0.9, 0.2, 0.2],
            "recall": [0.9, 0.2, 0.2],
            "f1": [0.90, 0.10, 0.10],
            "iou": [0.8, 0.05, 0.05],
        }
    )
    # raw 1 (c0): high ; raw 2 (c1): very_low ; raw 3 (c2): high.
    dist = _dist_report(
        bands={1: "high", 2: "very_low", 3: "high"},
        n_parcels={1: 5000, 2: 10, 3: 4000},
    )
    rec = recommend_classes_to_drop(report, dist, f1_threshold=0.30)

    by_raw = {row["class_id"]: row for row in rec.iter_rows(named=True)}
    # c0 (raw1): f1 alto -> no drop.
    assert by_raw[1]["drop"] is False
    # c1 (raw2): f1 bajo Y soporte muy bajo -> drop.
    assert by_raw[2]["drop"] is True
    # c2 (raw3): f1 bajo PERO soporte alto -> NO drop (criterio cruzado).
    assert by_raw[3]["drop"] is False
    assert by_raw[3]["below_f1"] is True
    assert by_raw[3]["low_support"] is False


def test_recommend_drop_translates_contiguous_to_raw() -> None:
    """El join usa el raw class_id = contiguous + 1."""
    report = pl.DataFrame(
        {
            "class_id": [0],
            "name": ["a"],
            "support": [10],
            "precision": [0.1],
            "recall": [0.1],
            "f1": [0.05],
            "iou": [0.02],
        }
    )
    dist = _dist_report(bands={1: "low"}, n_parcels={1: 20})
    rec = recommend_classes_to_drop(report, dist, f1_threshold=0.30)
    row = rec.row(0, named=True)
    assert row["class_id"] == 1  # raw
    assert row["contiguous_id"] == 0
    assert row["drop"] is True


def test_recommend_drop_null_f1_counts_as_below() -> None:
    """Una clase ausente (f1 null) cuenta como debajo del umbral."""
    report = pl.DataFrame(
        {
            "class_id": [0],
            "name": ["a"],
            "support": [0],
            "precision": [None],
            "recall": [None],
            "f1": [None],
            "iou": [None],
        },
        schema={
            "class_id": pl.Int64,
            "name": pl.Utf8,
            "support": pl.Int64,
            "precision": pl.Float64,
            "recall": pl.Float64,
            "f1": pl.Float64,
            "iou": pl.Float64,
        },
    )
    dist = _dist_report(bands={1: "very_low"}, n_parcels={1: 5})
    rec = recommend_classes_to_drop(report, dist, f1_threshold=0.30)
    row = rec.row(0, named=True)
    assert row["below_f1"] is True
    assert row["drop"] is True


def test_recommend_drop_dropped_first() -> None:
    """Las clases a descartar quedan ordenadas primero."""
    report = pl.DataFrame(
        {
            "class_id": [0, 1],
            "name": ["a", "b"],
            "support": [100, 20],
            "precision": [0.9, 0.1],
            "recall": [0.9, 0.1],
            "f1": [0.90, 0.05],
            "iou": [0.8, 0.02],
        }
    )
    dist = _dist_report(bands={1: "high", 2: "very_low"}, n_parcels={1: 5000, 2: 10})
    rec = recommend_classes_to_drop(report, dist, f1_threshold=0.30)
    # La primera fila debe ser la descartada (drop=True).
    assert rec.row(0, named=True)["drop"] is True


# ---------------------------------------------------------------------------
# honest_class_dropout_curve: rankea en OOF, mide en fold-5 (anti-fuga R-LEAK).
# ---------------------------------------------------------------------------


def test_honest_dropout_keeps_best_oof_classes() -> None:
    """El descarte sigue el ranking OOF: las peores-F1 OOF se quitan primero."""
    # OOF: clase 0 perfecta, 1 buena, 2 mala -> orden de retener: [0, 1, 2].
    y_oof = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    p_oof = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2], dtype=np.int64)
    # fold-5 cualquiera (mismo espacio de 3 clases).
    y_f5 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    p_f5 = np.array([0, 0, 1, 1, 2, 0], dtype=np.int64)
    curve = honest_class_dropout_curve(y_oof, p_oof, y_f5, p_f5, k_values=(3, 2, 1), num_classes=3)
    assert curve["k"].to_list() == [3, 2, 1]
    # K=2 retiene las 2 mejores OOF (0 y 1); la peor OOF (2) se descarta primero.
    k2 = curve.filter(pl.col("k") == 2).row(0, named=True)
    assert sorted(k2["retained_class_ids"]) == [0, 1]
    # La clase 2 (peor OOF) aparece en dropped_names (raw id 3 = Corn).
    assert any("Corn" in n for n in k2["dropped_names"])


def test_honest_dropout_excludes_dropped_parcels() -> None:
    """Las parcelas cuya GT es una clase descartada salen de la medicion."""
    # OOF marca la clase 2 como la peor -> a K=2 se descarta.
    y_oof = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    p_oof = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)  # clase 2 siempre mal
    # fold-5: 2 parcelas por clase. La clase 2 (descartada a K=2) no debe contar.
    y_f5 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    p_f5 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    curve = honest_class_dropout_curve(y_oof, p_oof, y_f5, p_f5, k_values=(3, 2), num_classes=3)
    k3 = curve.filter(pl.col("k") == 3).row(0, named=True)
    k2 = curve.filter(pl.col("k") == 2).row(0, named=True)
    assert k3["n_parcels_fold5"] == 6  # todas
    assert k2["n_parcels_fold5"] == 4  # se quitan las 2 de la clase descartada


def test_honest_dropout_ranking_uses_oof_not_fold5() -> None:
    """El ranking lo decide OOF, NO fold-5 (la prueba anti-cherry-picking)."""
    # OOF dice que la clase 0 es la PEOR (siempre mal) y la 1 la mejor.
    y_oof = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    p_oof = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)  # 0 F1=0, 1 perfecta
    # En fold-5, en cambio, la clase 0 se predice PERFECTA. Si el ranking mirara
    # fold-5 retendria la 0; como mira OOF, a K=1 debe retener la 1, no la 0.
    y_f5 = np.array([0, 0, 1, 1], dtype=np.int64)
    p_f5 = np.array([0, 0, 1, 1], dtype=np.int64)
    curve = honest_class_dropout_curve(y_oof, p_oof, y_f5, p_f5, k_values=(2, 1), num_classes=2)
    k1 = curve.filter(pl.col("k") == 1).row(0, named=True)
    # Retiene la clase 1 (mejor OOF), aunque la 0 fuese perfecta en fold-5.
    assert k1["retained_class_ids"] == [1]


def test_honest_dropout_k_clamped_to_available() -> None:
    """Un K mayor que las clases presentes se recorta sin romper."""
    y_oof = np.array([0, 0, 1, 1], dtype=np.int64)
    p_oof = np.array([0, 0, 1, 1], dtype=np.int64)
    y_f5 = np.array([0, 1], dtype=np.int64)
    p_f5 = np.array([0, 1], dtype=np.int64)
    # Pedimos K=18 sobre un espacio donde solo 2 clases tienen soporte.
    curve = honest_class_dropout_curve(y_oof, p_oof, y_f5, p_f5, k_values=(18,), num_classes=18)
    row = curve.row(0, named=True)
    # K se recorta a las 18 del ranking (incluye ausentes); las 2 presentes
    # cuentan y el F1 es 1.0 (prediccion perfecta en fold-5).
    assert abs(row["macro_f1_fold5"] - 1.0) < 1e-9
    assert row["n_parcels_fold5"] == 2
