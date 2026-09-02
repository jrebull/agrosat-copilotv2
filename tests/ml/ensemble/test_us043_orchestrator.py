"""Tests for the US-043 FarSLIP Stacking-5 / Blending-5 orchestrator.

Covers the two leak-sensitive pieces of ``scripts.run_us043_farslip_ensembles``
that do not need real PASTIS-R, the FarSLIP checkpoint or a GPU:

1. ``_fill_missing_parcels_uniform``: a parcel absent from a FarSLIP OOF is filled
   with the uniform prior ``1/18`` (the honest abstention that keeps the universe
   comparable with the 0.747 Stacking US-040 universe, R-MISSING) -- never dropped
   by an inner-join, and existing rows are preserved verbatim.
2. The FIVE-member assembly: a :class:`StackingEnsemble` built over the two new
   FarSLIP members plus the three legacy ones trains a meta-learner on
   ``5 x 18 = 90`` OOF meta-features and predicts valid post-softmax probabilities.

All inputs are tiny deterministic synthetic OOF + GT + geometry frames (no real
PASTIS-R, no DVC blobs, no inference).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point

from ml.ensemble.stacking import StackingEnsemble
from ml.utils.parcel_reconcile import PROB_COLUMNS
from scripts.run_us043_farslip_ensembles import (
    _BASE_MEMBERS_5,
    _FARSLIP_MEMBERS,
    _NUM_CLASSES,
    _fill_missing_parcels_uniform,
)

#: Two well-separated geographic clusters so build_spatial_kfold forms folds.
_CLUSTER_CENTERS: tuple[tuple[float, float], ...] = (
    (2.0, 44.0),
    (3.5, 45.5),
    (5.0, 47.0),
)


# ---------------------------------------------------------------------------
# Synthetic OOF builders (mirror tests/ml/ensemble/test_stacking.py).
# ---------------------------------------------------------------------------


def _make_parcel_probs(
    n_parcels: int, *, seed: int, signal_from: np.ndarray | None = None
) -> np.ndarray:
    """Build an ``(n_parcels, 18)`` post-softmax matrix (optional class signal)."""
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-1.0, 1.0, size=(n_parcels, _NUM_CLASSES))
    if signal_from is not None:
        logits[np.arange(n_parcels), signal_from] += 3.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _parcel_frame(ids: list[str], probs: np.ndarray) -> pl.DataFrame:
    """Assemble a parcel-space OOF frame (canonical schema)."""
    data: dict[str, object] = {"canonical_parcel_id": ids}
    for c, col in enumerate(PROB_COLUMNS):
        data[col] = probs[:, c].astype(np.float32)
    data["pred_class"] = probs.argmax(axis=1).astype(np.int64)
    data["n_pixels"] = np.full(len(ids), 100, dtype=np.int64)
    return pl.DataFrame(data)


def _write_five_member_fixture(
    oof_dir: Path, *, n_parcels: int = 60, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Write the five member parcel OOF parquets + return (geoms, gt, ids)."""
    oof_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"10000_{i:04d}" for i in range(n_parcels)]
    labels = rng.integers(0, 4, size=n_parcels).astype(np.int64)

    for m, member in enumerate(_BASE_MEMBERS_5):
        probs = _make_parcel_probs(n_parcels, seed=seed + 100 * (m + 1), signal_from=labels)
        _parcel_frame(ids, probs).write_parquet(oof_dir / f"oof_parcel_{member}_fold5.parquet")

    coords: list[tuple[float, float]] = []
    for i in range(n_parcels):
        cx, cy = _CLUSTER_CENTERS[i % len(_CLUSTER_CENTERS)]
        jitter = rng.uniform(-0.05, 0.05, size=2)
        coords.append((cx + float(jitter[0]), cy + float(jitter[1])))
    parcel_geoms = pl.DataFrame(
        {
            "canonical_parcel_id": ids,
            "geometry": [Point(lon, lat).wkt for lon, lat in coords],
        }
    )
    gt_labels = pl.DataFrame({"canonical_parcel_id": ids, "label": labels})
    return parcel_geoms, gt_labels, ids


# ---------------------------------------------------------------------------
# 1. Uniform-prior fill of the missing FarSLIP parcels (R-MISSING).
# ---------------------------------------------------------------------------


def test_fill_missing_parcels_uniform_adds_uniform_rows() -> None:
    """A parcel absent from the FarSLIP OOF gets a uniform 1/18 distribution."""
    present_ids = ["100_1", "100_2"]
    probs = _make_parcel_probs(2, seed=1)
    oof = _parcel_frame(present_ids, probs)
    universe = ["100_1", "100_2", "100_3", "100_4"]  # two missing

    filled = _fill_missing_parcels_uniform(oof, universe)

    # Every universe parcel is covered exactly once, sorted by the canonical key.
    assert filled.height == len(universe)
    assert filled["canonical_parcel_id"].to_list() == sorted(universe)

    uniform = 1.0 / _NUM_CLASSES
    by_id = {row["canonical_parcel_id"]: row for row in filled.iter_rows(named=True)}
    for missing in ("100_3", "100_4"):
        vec = np.array([by_id[missing][c] for c in PROB_COLUMNS], dtype=np.float64)
        np.testing.assert_allclose(vec, uniform, atol=1e-6)
        assert by_id[missing]["n_pixels"] == -1  # marks the abstention


def test_fill_missing_parcels_uniform_preserves_present_rows() -> None:
    """Existing FarSLIP rows are kept verbatim (only the missing ones are filled)."""
    present_ids = ["100_1", "100_2"]
    probs = _make_parcel_probs(2, seed=2)
    oof = _parcel_frame(present_ids, probs)
    universe = ["100_1", "100_2", "100_9"]

    filled = _fill_missing_parcels_uniform(oof, universe)
    by_id = {row["canonical_parcel_id"]: row for row in filled.iter_rows(named=True)}
    for row, pid in enumerate(present_ids):
        original = probs[row]
        kept = np.array([by_id[pid][c] for c in PROB_COLUMNS], dtype=np.float64)
        np.testing.assert_allclose(kept, original, atol=1e-6)


def test_fill_missing_parcels_uniform_rows_sum_to_one() -> None:
    """Every filled row (present or uniform) is a valid post-softmax distribution."""
    oof = _parcel_frame(["100_1"], _make_parcel_probs(1, seed=3))
    filled = _fill_missing_parcels_uniform(oof, ["100_1", "100_2", "100_3"])
    matrix = filled.select(PROB_COLUMNS).to_numpy().astype(np.float64)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-5)
    assert (matrix >= 0).all()


def test_fill_missing_parcels_uniform_noop_when_complete() -> None:
    """When the OOF already covers the universe nothing is filled."""
    ids = ["100_1", "100_2"]
    oof = _parcel_frame(ids, _make_parcel_probs(2, seed=4))
    filled = _fill_missing_parcels_uniform(oof, ids)
    assert filled.height == 2
    # No abstention rows (n_pixels == -1) were introduced.
    assert (filled["n_pixels"] != -1).all()


def test_fill_missing_parcels_uniform_requires_prob_columns() -> None:
    """An OOF without the prob_* columns raises a clear error."""
    bad = pl.DataFrame({"canonical_parcel_id": ["100_1"]})
    with pytest.raises(ValueError, match="prob"):
        _fill_missing_parcels_uniform(bad, ["100_1"])


# ---------------------------------------------------------------------------
# 2. Five-member assembly (the two FarSLIP members + the three legacy ones).
# ---------------------------------------------------------------------------


def test_five_member_set_is_three_plus_two_farslip() -> None:
    """The 5-member contract is the 3 legacy members plus the 2 FarSLIP members."""
    assert _FARSLIP_MEMBERS == ("farslip-ft18", "farslip-zeroshot")
    # The base TSViT member is ``tsvit-pheno-fullm`` (0.6764 macro-F1, the best
    # TSViT the sponsor selected), NOT the older ``tsvit-pheno`` (0.6253). See
    # ``scripts/run_us043_farslip_ensembles.py`` _BASE_MEMBERS_3.
    assert _BASE_MEMBERS_5 == (
        "tsvit-pheno-fullm",
        "utae",
        "xgb-alphaearth",
        "farslip-ft18",
        "farslip-zeroshot",
    )
    assert len(_BASE_MEMBERS_5) == 5


def test_stacking5_builds_90_meta_features(tmp_path: Path) -> None:
    """Stacking over the 5 members lays out 5 x 18 = 90 OOF meta-features."""
    _, gt, _ = _write_five_member_fixture(tmp_path, n_parcels=40, seed=5)
    ens = StackingEnsemble(_BASE_MEMBERS_5, oof_dir=tmp_path, n_spatial_folds=3)
    keys_df, x_meta, y = ens.build_meta_features(gt_labels=gt)
    assert x_meta.shape == (keys_df.height, len(_BASE_MEMBERS_5) * _NUM_CLASSES)
    assert x_meta.shape[1] == 90
    assert y is not None and y.shape[0] == keys_df.height


def test_stacking5_fit_predict_valid_probs(tmp_path: Path) -> None:
    """Stacking-5 trains and emits (n, 18) post-softmax probabilities."""
    parcel_geoms, gt, _ = _write_five_member_fixture(tmp_path, n_parcels=60, seed=6)
    ens = StackingEnsemble(_BASE_MEMBERS_5, oof_dir=tmp_path, n_spatial_folds=3)
    ens.fit(parcel_geoms, gt_labels=gt)

    assert ens.meta_model_ is not None
    assert "f1_macro" in ens.oof_cv_metrics_
    proba = ens.predict_proba()
    keys_df, _, _ = ens.build_meta_features(gt_labels=None)
    assert proba.shape == (keys_df.height, _NUM_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba >= 0).all()


def test_stacking5_uniform_filled_farslip_member(tmp_path: Path) -> None:
    """A FarSLIP member that abstained on some parcels still joins (uniform-filled).

    Simulates the production path: a FarSLIP OOF covers fewer parcels, then
    ``_fill_missing_parcels_uniform`` extends it to the full universe so the inner
    join keeps every parcel. Stacking then fits over the complete 5-member set.
    """
    parcel_geoms, gt, ids = _write_five_member_fixture(tmp_path, n_parcels=60, seed=7)
    # Drop half the parcels from one FarSLIP member to simulate abstentions.
    ft18_path = tmp_path / "oof_parcel_farslip-ft18_fold5.parquet"
    partial = pl.read_parquet(ft18_path).head(30)
    filled = _fill_missing_parcels_uniform(partial, ids)
    filled.write_parquet(ft18_path)
    assert filled.height == len(ids)  # the universe is restored before joining.

    ens = StackingEnsemble(_BASE_MEMBERS_5, oof_dir=tmp_path, n_spatial_folds=3)
    ens.fit(parcel_geoms, gt_labels=gt)
    keys_df, _, _ = ens.build_meta_features(gt_labels=None)
    # Every parcel survives the inner join thanks to the uniform fill.
    assert keys_df.height == len(ids)
