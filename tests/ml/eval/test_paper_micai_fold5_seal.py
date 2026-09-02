"""Guarda el sello del fold 5 que sostiene las cifras del manuscrito MICAI.

El eje de cobertura de la curva de cardinalidad (`us043_honest_dropout_curve.csv`)
se escribio con un guion que no quedo versionado. Estas pruebas lo re-derivan desde
el ground truth sellado en `reports/paper_micai/fase1/`, obtenido de PASTIS-R de
forma independiente, para que cualquier divergencia futura salte en `make test-ml`
y no en la revision del articulo.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GT_PATH = REPO_ROOT / "reports" / "paper_micai" / "fase1" / "parcel_gt_fold5.parquet"
CURVE_PATH = REPO_ROOT / "reports" / "ensemble" / "metrics" / "us043_honest_dropout_curve.csv"

#: Universo compartido por los diez miembros OOF de Francia en el fold 5.
EXPECTED_PARCELS = 16640
#: Espacio semantic18 del arnes de segmentacion.
EXPECTED_CLASSES = 18


@pytest.fixture(scope="module")
def ground_truth() -> pl.DataFrame:
    """Devuelve el ground truth sellado del fold 5.

    Returns:
        Marco con ``canonical_parcel_id`` y ``label``.
    """
    if not GT_PATH.exists():
        pytest.skip(f"falta el sello {GT_PATH}; regenerar con scripts/paper_micai_seal_fold5.py")
    return pl.read_parquet(GT_PATH)


def test_universe_size(ground_truth: pl.DataFrame) -> None:
    """El sello cubre exactamente las 16 640 parcelas y las 18 clases del arnes."""
    assert ground_truth.height == EXPECTED_PARCELS
    assert ground_truth["canonical_parcel_id"].n_unique() == EXPECTED_PARCELS
    assert ground_truth["label"].n_unique() == EXPECTED_CLASSES
    assert ground_truth["label"].min() == 0
    assert ground_truth["label"].max() == EXPECTED_CLASSES - 1


def test_coverage_axis_matches_sealed_curve(ground_truth: pl.DataFrame) -> None:
    """Cada cobertura de la curva de cardinalidad se re-deriva desde el ground truth.

    Es la comprobacion cruzada que permite citar la curva pese a que su guion
    original no este versionado: el soporte de las clases retenidas en cada K debe
    dar el mismo ``n_parcels_fold5`` que el CSV sellado.
    """
    if not CURVE_PATH.exists():
        pytest.skip(f"falta la curva sellada {CURVE_PATH}")
    support = dict(ground_truth.group_by("label").len().rename({"len": "n"}).iter_rows())
    with CURVE_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "la curva sellada esta vacia"
    for row in rows:
        retained = [int(x) for x in row["retained_class_ids"].split(",") if x.strip()]
        recomputed = sum(support[class_id] for class_id in retained)
        assert recomputed == int(row["n_parcels_fold5"]), (
            f"K={row['k']}: el CSV dice {row['n_parcels_fold5']} parcelas y el ground "
            f"truth sellado da {recomputed}"
        )


def test_class_imbalance_is_the_one_the_paper_claims(ground_truth: pl.DataFrame) -> None:
    """El desbalance del que trata el articulo: la clase mayor supera 50 veces a la menor."""
    counts = ground_truth.group_by("label").len().rename({"len": "n"})["n"].to_list()
    assert max(counts) / min(counts) > 50
