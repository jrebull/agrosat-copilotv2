"""Tests for :mod:`ml.transfer.italia_label_space` (US-079, the kept-class flag).

The label space generalises the Baltic one to the Italian dense homologue. The
critical behaviour under test is the WARM-START at the matrix level (the dense
head row copy is tested separately in ``test_finetune_italia``):

- ``build_italia_label_space`` reads the US-078 ``class_table.parquet`` and tags
  each crop leaf CONSERVED (maps to a PASTIS-18 name) or NEW (Mediterranean),
  keeping background id 0 and the dense head size ``K + 1``.
- ``warm_start_head`` copies the PASTIS head rows ONLY into the conserved Italian
  rows; the new + background rows keep their init. We assert the conserved rows
  equal the PASTIS rows (``allclose``) and the new rows stay distinct.
- ``coarse_of`` / FINE_TO_COARSE collapse a fine leaf to the right coarse bucket.
- ``stratified_pixel_patch_sample`` covers each class deterministically.

Everything is numpy with a fixed RNG; no torch, no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
from ml.transfer.italia_label_space import (
    ItaliaLabelSpace,
    build_italia_label_space,
    stratified_pixel_patch_sample,
    warm_start_head,
)

# Three conserved leaves (present in CONSERVED_LEAF_TO_PASTIS) + two new leaves.
_CONSERVED = ["common_soft_wheat", "maize_corn_popcorn", "vineyards_wine_vine_rebland_grapes"]
_NEW = ["olive", "tree_wood_forest"]


def _write_class_table(tmp_path: Path) -> Path:
    """Write a toy US-078 ``class_table.parquet`` (background NOT a row; ids 1..K)."""
    names = _CONSERVED + _NEW
    pl.DataFrame(
        {
            "class_id": list(range(1, len(names) + 1)),
            "hcat4_name": names,
            "n_parcels": [500] * len(names),
        }
    ).write_parquet(tmp_path / "class_table.parquet")
    return tmp_path


def _label_space(tmp_path: Path) -> ItaliaLabelSpace:
    _write_class_table(tmp_path)
    return build_italia_label_space(italia_root=tmp_path)


# --------------------------------------------------------------------------- #
# build_italia_label_space.
# --------------------------------------------------------------------------- #
def test_build_label_space_indexes_background_then_crops(tmp_path: Path) -> None:
    """Index 0 is background; ids 1..K are the crops in class_table order."""
    space = _label_space(tmp_path)
    assert space.background_id == 0
    assert space.leaves[0] == "__background__"
    # K crops + background.
    assert space.num_classes == len(_CONSERVED) + len(_NEW) + 1
    assert space.class_ids == tuple(range(1, len(_CONSERVED) + len(_NEW) + 1))
    # Each crop leaf indexes to its dense class id.
    for cid, name in enumerate(_CONSERVED + _NEW, start=1):
        assert space.index[name] == cid
        assert space.leaves[cid] == name


def test_build_label_space_splits_conserved_and_new(tmp_path: Path) -> None:
    """Conserved leaves map to PASTIS-18; the rest are flagged new."""
    space = _label_space(tmp_path)
    assert set(space.conserved) == set(_CONSERVED)
    assert set(space.new) == set(_NEW)
    # The conserved crosswalk targets are real PASTIS-18 names.
    pastis_names = set(SEMANTIC18_CLASS_NAMES.values())
    for leaf in space.conserved:
        assert space.leaf_to_pastis[leaf] in pastis_names


def test_build_label_space_reads_json_when_parquet_absent(tmp_path: Path) -> None:
    """When only class_mapping.json exists, the label space still builds."""
    names = _CONSERVED + _NEW
    mapping = {
        "background_id": 0,
        "other_class_name": "other",
        "classes": [
            {"class_id": i, "hcat4_name": n, "n_parcels": 500} for i, n in enumerate(names, start=1)
        ],
    }
    (tmp_path / "class_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    space = build_italia_label_space(italia_root=tmp_path)
    assert space.num_classes == len(names) + 1


def test_build_label_space_missing_inputs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"class_table|class_mapping"):
        build_italia_label_space(italia_root=tmp_path)


# --------------------------------------------------------------------------- #
# coarse_of / fine -> coarse collapse.
# --------------------------------------------------------------------------- #
def test_coarse_of_conserved_collapses_to_pastis_parent(tmp_path: Path) -> None:
    """A conserved leaf collapses to its PASTIS parent name."""
    space = _label_space(tmp_path)
    assert space.coarse_of("common_soft_wheat") == "Soft winter wheat"
    assert space.coarse_of("vineyards_wine_vine_rebland_grapes") == "Grapevine"


def test_coarse_of_new_leaf_uses_agronomic_group(tmp_path: Path) -> None:
    """A new Mediterranean leaf collapses to its explicit FINE_TO_COARSE group."""
    space = _label_space(tmp_path)
    assert space.coarse_of("tree_wood_forest") == "Forest"
    # 'olive' has no FINE_TO_COARSE override here only if absent; it does have one.
    assert space.coarse_of("olive") == "Permanent woody crop"


def test_coarse_of_unknown_leaf_keeps_own_name(tmp_path: Path) -> None:
    """A leaf with no crosswalk and no FINE_TO_COARSE keeps its own name."""
    space = _label_space(tmp_path)
    assert space.coarse_of("totally_unknown_leaf") == "totally_unknown_leaf"


# --------------------------------------------------------------------------- #
# warm_start_head: conserved rows == PASTIS rows, new rows stay random.
# --------------------------------------------------------------------------- #
def _pastis_head(dim: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """A toy PASTIS-18 head (18, dim) with the contiguous semantic id->name map."""
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(18, dim)).astype(np.float64)
    bias = rng.normal(size=(18,)).astype(np.float64)
    return weight, bias, dict(SEMANTIC18_CLASS_NAMES)


def test_warm_start_copies_conserved_rows_keeps_new_random(tmp_path: Path) -> None:
    """Conserved rows equal the PASTIS rows; new + background rows stay at init."""
    space = _label_space(tmp_path)
    dim = 16
    rng = np.random.default_rng(123)
    new_w = rng.normal(size=(space.num_classes, dim)).astype(np.float64)
    new_b = rng.normal(size=(space.num_classes,)).astype(np.float64)
    new_w_init = new_w.copy()
    pw, pb, names = _pastis_head(dim)
    name_to_pastis_id = {name: cid for cid, name in names.items()}

    out_w, out_b, warmed = warm_start_head(
        new_w, new_b, pw, pb, label_space=space, pastis_class_names=names
    )

    assert set(warmed) == set(_CONSERVED)
    # Each conserved Italian row equals the matching PASTIS row.
    for leaf in _CONSERVED:
        row = space.index[leaf]
        pastis_id = name_to_pastis_id[space.leaf_to_pastis[leaf]]
        np.testing.assert_allclose(out_w[row], pw[pastis_id])
        np.testing.assert_allclose(out_b[row], pb[pastis_id])
    # New + background rows are UNCHANGED from their random init (not warmed).
    for leaf in _NEW:
        row = space.index[leaf]
        np.testing.assert_allclose(out_w[row], new_w_init[row])
        # And they differ from every PASTIS row (genuinely random, not copied).
        assert not any(np.allclose(out_w[row], pw[i]) for i in range(18))
    np.testing.assert_allclose(out_w[space.background_id], new_w_init[space.background_id])


def test_warm_start_dim_mismatch_returns_unwarmed(tmp_path: Path) -> None:
    """A head/PASTIS feature-dim mismatch warms nothing (honest, no crash)."""
    space = _label_space(tmp_path)
    new_w = np.zeros((space.num_classes, 8), dtype=np.float64)
    pw, pb, names = _pastis_head(16)  # 16 != 8
    out_w, _out_b, warmed = warm_start_head(
        new_w, None, pw, pb, label_space=space, pastis_class_names=names
    )
    assert warmed == []
    np.testing.assert_array_equal(out_w, np.zeros((space.num_classes, 8)))


def test_warm_start_skips_conserved_without_pastis_row(tmp_path: Path) -> None:
    """A conserved leaf whose PASTIS name is absent stays random and is omitted."""
    space = _label_space(tmp_path)
    dim = 12
    new_w = np.zeros((space.num_classes, dim), dtype=np.float64)
    pw = np.ones((18, dim), dtype=np.float64)
    # Drop one conserved target name from the PASTIS id->name map.
    names = dict(SEMANTIC18_CLASS_NAMES)
    drop_name = space.leaf_to_pastis["maize_corn_popcorn"]
    names = {cid: n for cid, n in names.items() if n != drop_name}
    _, _, warmed = warm_start_head(
        new_w, None, pw, None, label_space=space, pastis_class_names=names
    )
    assert "maize_corn_popcorn" not in warmed
    # The other conserved leaves are still warmed.
    assert "common_soft_wheat" in warmed
    # The skipped leaf row stays at init (zeros), not copied from PASTIS ones.
    np.testing.assert_array_equal(new_w[space.index["maize_corn_popcorn"]], np.zeros(dim))


# --------------------------------------------------------------------------- #
# stratified_pixel_patch_sample.
# --------------------------------------------------------------------------- #
def test_stratified_sample_covers_each_class_deterministically() -> None:
    """Each class id with carriers is covered; the result is seed-deterministic."""
    # patch 0 has classes {1,2}, patch 1 {2,3}, patch 2 {1}, patch 3 {3}.
    patch_classes = [{1, 2}, {2, 3}, {1}, {3}]
    sel_a = stratified_pixel_patch_sample(
        patch_classes, class_ids=(1, 2, 3), min_patches_per_class=1, seed=0
    )
    sel_b = stratified_pixel_patch_sample(
        patch_classes, class_ids=(1, 2, 3), min_patches_per_class=1, seed=0
    )
    assert sel_a == sel_b  # deterministic
    covered: set[int] = set()
    for i in sel_a:
        covered |= patch_classes[i]
    assert {1, 2, 3} <= covered  # every requested class is represented


def test_stratified_sample_ignores_classes_without_carriers() -> None:
    """A class with no carrier patch is silently skipped (no crash)."""
    patch_classes = [{1}, {1}]
    sel = stratified_pixel_patch_sample(
        patch_classes, class_ids=(1, 9), min_patches_per_class=2, seed=1
    )
    assert sel == [0, 1]  # class 9 has no carrier; class 1 covered by both patches
