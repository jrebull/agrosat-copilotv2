"""Tests de ml.analysis.hcat_grouping (agrupamiento HCAT Level-1).

Cubren el mapeo de las 18 clases PASTIS-R a los 6 grupos HCAT L1, la
anexion de columnas de grupo y el evaluador apples-to-apples flat-18 vs
grouped-6 sobre un dataset sintetico pequeno y determinista.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.analysis.hcat_grouping import (
    HCAT_L1_GROUP_CODES,
    HCAT_L1_GROUP_ORDER,
    HCAT_L1_GROUPS,
    PASTIS_CLASS_TO_HCAT_L1,
    add_hcat_l1_group,
    evaluate_flat_vs_grouped,
    hcat_group_id_map,
    per_label_f1_table,
)


def test_mapping_covers_18_active_classes() -> None:
    """El mapeo cubre exactamente las 18 clases agronomicas (sin 0 ni 19)."""
    assert len(PASTIS_CLASS_TO_HCAT_L1) == 18
    assert 0 not in PASTIS_CLASS_TO_HCAT_L1
    assert 19 not in PASTIS_CLASS_TO_HCAT_L1
    assert set(PASTIS_CLASS_TO_HCAT_L1) == set(range(1, 19))


def test_exactly_six_groups_with_expected_membership() -> None:
    """Hay 6 grupos y la composicion coincide con la taxonomia HCAT L1."""
    assert len(HCAT_L1_GROUP_ORDER) == 6
    assert set(HCAT_L1_GROUPS) == set(HCAT_L1_GROUP_ORDER)
    assert HCAT_L1_GROUPS["CEREALS"] == [2, 3, 4, 6, 10, 11, 17, 18]
    assert HCAT_L1_GROUPS["OILSEEDS"] == [5, 7]
    assert HCAT_L1_GROUPS["ROOT_CROPS"] == [9, 13]
    assert HCAT_L1_GROUPS["LEGUMES"] == [14, 15]
    assert HCAT_L1_GROUPS["PERMANENT_WOODY"] == [8, 16]
    assert HCAT_L1_GROUPS["OTHER"] == [1, 12]


def test_group_codes_present_for_all_groups() -> None:
    """Cada grupo tiene un codigo HCAT documentado."""
    assert set(HCAT_L1_GROUP_CODES) == set(HCAT_L1_GROUP_ORDER)
    assert all(code.isdigit() for code in HCAT_L1_GROUP_CODES.values())


def test_group_id_map_starts_at_one_to_avoid_drop_collision() -> None:
    """Los ids de grupo van de 1 a 6 (evitan el drop de class_id 0)."""
    id_map = hcat_group_id_map()
    assert sorted(id_map.values()) == list(range(1, 7))
    assert id_map["CEREALS"] == 1
    assert id_map["ROOT_CROPS"] == 6
    # Ningun grupo cae en los ids de fondo del pipeline baseline (0, 19).
    assert 0 not in id_map.values()
    assert 19 not in id_map.values()


def test_add_hcat_l1_group_assigns_and_leaves_void_null() -> None:
    """add_hcat_l1_group mapea agronomicas y deja null las clases 0/19."""
    df = pl.DataFrame({"class_id": [2, 11, 8, 1, 0, 19], "x": [1.0] * 6})
    out = add_hcat_l1_group(df)
    names = out.get_column("hcat6_group_name").to_list()
    ids = out.get_column("hcat6_group_id").to_list()
    id_map = hcat_group_id_map()
    assert names[:4] == ["CEREALS", "CEREALS", "PERMANENT_WOODY", "OTHER"]
    assert ids[:4] == [
        id_map["CEREALS"],
        id_map["CEREALS"],
        id_map["PERMANENT_WOODY"],
        id_map["OTHER"],
    ]
    assert names[4] is None and names[5] is None
    assert ids[4] is None and ids[5] is None


def test_add_hcat_l1_group_requires_class_col() -> None:
    """Sin class_id, add_hcat_l1_group levanta ValueError."""
    with pytest.raises(ValueError, match="class_id"):
        add_hcat_l1_group(pl.DataFrame({"foo": [1, 2]}))


def test_per_label_f1_table_shape_and_support() -> None:
    """per_label_f1_table devuelve f1 + soporte por etiqueta."""
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 2, 2])
    table = per_label_f1_table(y_true, y_pred, label_names={0: "a", 1: "b", 2: "c"})
    assert table.columns == ["label_id", "label_name", "f1", "support"]
    assert table.height == 3
    support = dict(zip(table["label_id"].to_list(), table["support"].to_list(), strict=True))
    assert support == {0: 2, 1: 2, 2: 1}


def _synthetic_dataset(seed: int = 7) -> pl.DataFrame:
    """Dataset sintetico separable: features correlan con el grupo HCAT.

    Construye 12 parcelas por clase agronomica con dos features gaussianas
    centradas en el id del grupo, de modo que el agrupamiento a 6 grupos sea
    mas separable que las 18 clases planas (clases hermanas se solapan).
    """
    rng = np.random.default_rng(seed)
    rows = []
    id_map = hcat_group_id_map()
    n_per_class = 40
    for class_id, group in PASTIS_CLASS_TO_HCAT_L1.items():
        gid = id_map[group]
        for _ in range(n_per_class):
            rows.append(
                {
                    "parcel_id": f"p{len(rows)}",
                    # patch_id numerico: el CV espacial deriva el centroide del
                    # patch via int(patch_id); un id sin metadata cae a jitter
                    # determinista, suficiente para la prueba.
                    "patch_id": str(len(rows) % 20),
                    "class_id": class_id,
                    # features centradas en el grupo + ruido por clase hermana
                    "f0": float(gid * 3.0 + rng.normal(0, 0.6) + (class_id % 3) * 0.2),
                    "f1": float(gid * 2.0 + rng.normal(0, 0.6)),
                }
            )
    return pl.DataFrame(rows)


def test_evaluate_flat_vs_grouped_runs_and_groups_help() -> None:
    """El evaluador corre y el esquema agrupado no empeora el F1-macro."""
    df = _synthetic_dataset()
    result = evaluate_flat_vs_grouped(df, model="rf", k_folds=3, buffer_km=0.0, random_state=42)
    # 18 clases planas; el esquema agrupado tiene a lo sumo 6 grupos (sobre la
    # rejilla espacial sintetica algun grupo pequeno puede no entrar en algun
    # fold; en datos reales los 6 grupos tienen miles de parcelas).
    assert result.flat_per_class.height == 18
    assert 1 <= result.grouped_per_group.height <= 6
    assert set(result.grouped_per_group["label_name"].to_list()) <= set(HCAT_L1_GROUP_ORDER)
    assert set(result.flat_metrics) >= {"f1_macro", "f1_weighted", "miou", "accuracy"}
    # En datos separables por grupo, agregar las hermanas sube el macro.
    assert result.grouped_metrics["f1_macro"] >= result.flat_metrics["f1_macro"]
    assert result.delta_f1_macro == pytest.approx(
        result.grouped_metrics["f1_macro"] - result.flat_metrics["f1_macro"]
    )
    # OOF de ambos esquemas cubre todas las parcelas.
    assert result.flat_y_true.size == df.height
    assert result.grouped_y_true.size == df.height
