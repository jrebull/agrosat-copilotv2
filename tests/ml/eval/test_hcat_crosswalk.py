"""Tests for the US-074 PASTIS-18 -> HCAT v3 crosswalk and the ``hcat-macro``
label-space extension of the US-053 registry.

Every assertion is grounded in the REAL reference data in the repo
(``eurocrops_hcat3.csv``, ``pastis_class_mapping.json``); nothing is synthetic.
The parquet is re-derived in a tmp path for the roundtrip test so the committed
artifact is never mutated by the suite.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.data.hcat_crosswalk import (
    EUROCROPS_HCAT3_CSV,
    MACRO_HCAT_GROUPS,
    build_crosswalk,
    load_crosswalk,
    write_crosswalk,
)
from ml.eval.class_remap import get_label_space, list_label_spaces


@pytest.fixture(scope="module")
def crosswalk() -> pl.DataFrame:
    """Re-derive the crosswalk from the real CSV/JSON sources once per module."""
    return build_crosswalk()


def test_crosswalk_has_18_rows(crosswalk: pl.DataFrame) -> None:
    """One row per PASTIS crop class (ids 1..18)."""
    assert crosswalk.height == 18


def test_all_leaf_codes_exist_in_eurocrops(crosswalk: pl.DataFrame) -> None:
    """Every HCAT leaf code appears verbatim in eurocrops_hcat3.csv (no invented codes)."""
    ref = pl.read_csv(EUROCROPS_HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8})
    valid = set(ref["HCAT3_code"].to_list())
    for code in crosswalk["hcat_leaf_code"].to_list():
        assert code in valid, f"leaf code {code} absent from eurocrops_hcat3.csv"


def test_semantic18_id_is_pastis_minus_one(crosswalk: pl.DataFrame) -> None:
    """semantic18_id is the contiguous pastis_id - 1 for every row."""
    bad = crosswalk.filter(pl.col("semantic18_id") != pl.col("pastis_id") - 1)
    assert bad.height == 0


def test_macro_count_in_range(crosswalk: pl.DataFrame) -> None:
    """Distinct HCAT macro-groups in [10, 15]; legacy l1_6 view has exactly 6."""
    n_macro = crosswalk["macro_hcat_group"].n_unique()
    assert 10 <= n_macro <= 15, f"expected 10..15 macro groups, got {n_macro}"
    assert crosswalk["macro_hcat_l1_6"].n_unique() == 6


def test_meadow_isolated_in_grassland(crosswalk: pl.DataFrame) -> None:
    """Meadow (sem18=0) is the only class in the ``grassland`` macro (long-tail mitigation)."""
    grassland = crosswalk.filter(pl.col("macro_hcat_group") == "grassland")
    assert grassland.height == 1
    assert grassland["semantic18_id"].item() == 0
    assert grassland["pastis_name"].item() == "Meadow"


def test_three_approx_matches(crosswalk: pl.DataFrame) -> None:
    """Exactly three classes lack a 1:1 leaf and are flagged ``approx``."""
    approx = crosswalk.filter(pl.col("match_quality") == "approx")
    assert approx.height == 3
    names = set(approx["pastis_name"].to_list())
    assert names == {
        "Mixed cereal",
        "Fruits, vegetables, flowers",
        "Leguminous fodder",
    }


def test_macro_vocabulary_matches_constant(crosswalk: pl.DataFrame) -> None:
    """The 10 crop macros in the table are exactly MACRO_HCAT_GROUPS minus ``void``."""
    crop_macros = set(crosswalk["macro_hcat_group"].unique().to_list())
    expected = set(MACRO_HCAT_GROUPS) - {"void"}
    assert crop_macros == expected
    assert "void" in MACRO_HCAT_GROUPS
    assert len(MACRO_HCAT_GROUPS) == 11


def test_void_convention_is_crop_for_all_rows(crosswalk: pl.DataFrame) -> None:
    """All 18 crop rows carry void_convention='crop' (background/void live outside)."""
    assert crosswalk["void_convention"].unique().to_list() == ["crop"]


def test_parquet_roundtrip(tmp_path: Path) -> None:
    """write -> read preserves 18 rows and zero-padded string codes."""
    out = tmp_path / "hcat_crosswalk.parquet"
    write_crosswalk(out)
    df = load_crosswalk(out)
    assert df.height == 18
    assert df.schema["hcat_leaf_code"] == pl.Utf8
    assert df.schema["hcat_group_code"] == pl.Utf8
    # Leading-zero / trailing-zero codes survive the roundtrip as strings.
    meadow = df.filter(pl.col("semantic18_id") == 0)
    assert meadow["hcat_leaf_code"].item() == "3302000000"


def test_committed_parquet_matches_rebuild() -> None:
    """The committed parquet equals a fresh re-derivation (no manual drift)."""
    rebuilt = build_crosswalk()
    committed = load_crosswalk()
    assert committed.sort("semantic18_id").equals(rebuilt.sort("semantic18_id"))


def test_hcat_macro_registered() -> None:
    """``hcat-macro`` is registered and keeps all 18 semantic18 ids."""
    assert "hcat-macro" in list_label_spaces()
    space = get_label_space("hcat-macro")
    assert len(space.kept_class_ids) == 18
    assert space.dropped_class_ids == ()
    # The macro label rides along in class_names without changing the kept set.
    assert space.class_names[0].startswith("GRASSLAND_OTHER|grassland")


def test_france9_intact() -> None:
    """US-053 ``france-9`` still has its nine kept ids (registry extension is additive)."""
    space = get_label_space("france-9")
    assert len(space.kept_class_ids) == 9


def test_classify_not_imported_change() -> None:
    """Meta-guard: the classifier still consumes the registry by name only.

    US-074 must not touch ``ml.agent.tools.classify``; it imports
    ``get_label_space``/``restrict_posterior`` from ``class_remap`` and resolves
    label-spaces by name. Asserting that contract here documents the seam.
    """
    from ml.agent.tools import classify

    assert hasattr(classify, "get_label_space")
    assert hasattr(classify, "restrict_posterior")
