"""Tests para `ml.utils.class_distribution`."""

from __future__ import annotations

import polars as pl
import pytest

from ml.utils.class_distribution import (
    class_distribution_report,
    merge_to_phenological_groups,
    recommend_threshold,
)


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """DataFrame sintetico con distribucion desbalanceada PASTIS-like."""
    # Clase 1 mayoritaria (~30k), clase 4 (~1500), clase 7 (~500), clase 14 (~30)
    # mas la clase 0 (Background, debe ser descartada).
    return pl.DataFrame(
        {
            "class_id": (
                [1] * 30000
                + [4] * 1500
                + [7] * 500
                + [14] * 30
                + [0] * 100  # debe ser dropped
                + [19] * 50  # debe ser dropped
            ),
        }
    )


def test_class_distribution_report_drops_background_and_void(
    synthetic_df: pl.DataFrame,
) -> None:
    report = class_distribution_report(synthetic_df)
    class_ids = set(report["class_id"].to_list())
    assert 0 not in class_ids
    assert 19 not in class_ids
    # Las 4 clases agronomicas estan
    assert {1, 4, 7, 14}.issubset(class_ids)


def test_class_distribution_report_assigns_support_bands(
    synthetic_df: pl.DataFrame,
) -> None:
    report = class_distribution_report(synthetic_df)
    bands = {row["class_id"]: row["support_band"] for row in report.iter_rows(named=True)}
    assert bands[1] == "high"  # 30000 >= 1000
    assert bands[4] == "high"  # 1500 >= 1000
    assert bands[7] == "med"  # 500 >= 200
    assert bands[14] == "low"  # 30 >= 30 pero < 200


def test_class_distribution_report_share_sums_to_one(
    synthetic_df: pl.DataFrame,
) -> None:
    report = class_distribution_report(synthetic_df)
    total_share = report["share"].sum()
    assert abs(total_share - 1.0) < 1e-9


def test_class_distribution_report_enriches_with_class_names(
    synthetic_df: pl.DataFrame,
) -> None:
    report = class_distribution_report(synthetic_df)
    assert "class_name" in report.columns
    assert "agronomic_group" in report.columns
    assert "phenological_cycle" in report.columns
    # class_name no debe ser None para clases reales
    names = report["class_name"].to_list()
    assert all(n is not None and len(n) > 0 for n in names)


def test_class_distribution_report_empty_df_returns_empty() -> None:
    df = pl.DataFrame(
        {"class_id": [0, 19]},
        schema={"class_id": pl.Int64},
    )  # solo clases dropeadas
    report = class_distribution_report(df)
    assert report.height == 0


def test_class_distribution_report_raises_on_missing_column() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError, match="class_id"):
        class_distribution_report(df)


def test_recommend_threshold_p25(synthetic_df: pl.DataFrame) -> None:
    report = class_distribution_report(synthetic_df)
    t = recommend_threshold(report, method="p25")
    # p25 de [30000, 1500, 500, 30] es aprox 380; debe ser menor que el max
    # y mayor que el min
    assert 30 <= t <= 30000


def test_recommend_threshold_p50(synthetic_df: pl.DataFrame) -> None:
    report = class_distribution_report(synthetic_df)
    t = recommend_threshold(report, method="p50")
    assert t > 0


def test_recommend_threshold_invalid_method(synthetic_df: pl.DataFrame) -> None:
    report = class_distribution_report(synthetic_df)
    with pytest.raises(ValueError):
        recommend_threshold(report, method="foo")  # type: ignore[arg-type]


def test_merge_to_phenological_groups_adds_column(
    synthetic_df: pl.DataFrame,
) -> None:
    merged = merge_to_phenological_groups(synthetic_df)
    assert "pheno_group_id" in merged.columns
    # Todas las filas tienen un grupo asignado (puede ser "other")
    groups = merged["pheno_group_id"].to_list()
    assert all(g is not None for g in groups)


def test_merge_to_phenological_groups_invalid_grouping(
    synthetic_df: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="no disponible"):
        merge_to_phenological_groups(synthetic_df, grouping_name="bogus")
