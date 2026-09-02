"""Tests for pixel->parcel reconciliation (US-031, ml/B).

Exercises :func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs` and
:func:`ml.utils.parcel_reconcile.load_pastis_parcel_ids` over synthetic grids
with known ParcelIDs, so the per-parcel probabilities, predicted classes,
support counts and canonical ids are closed-form golden values. No PASTIS-R data
or checkpoint is loaded (the file reader is tested against a tmp_path npy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.utils.parcel_reconcile import (
    PROB_COLUMNS,
    load_pastis_parcel_ids,
    pixel_to_parcel_probs,
)
from tests.ml.eval.oof.fixtures.oof_synthetic import (
    make_constant_class_probs,
    make_parcel_grid,
    make_softmax,
)

_SUM_TOL = 1e-5


def test_parcel_mean_sums_to_one() -> None:
    """Per-parcel mean probabilities sum to 1 over the 18 classes."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    probs = make_softmax(num_classes=18, size=6, seed=20)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")

    assert df.height == 2  # two parcels (101, 202), background excluded
    prob_matrix = df.select(PROB_COLUMNS).to_numpy()
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, atol=_SUM_TOL)


def test_canonical_parcel_id_prefix() -> None:
    """canonical_parcel_id is f'{patch_id}_{local}' (Utf8) for every parcel."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    probs = make_softmax(num_classes=18, size=6, seed=21)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")

    assert df.schema["canonical_parcel_id"] == pl.Utf8
    ids = set(df["canonical_parcel_id"].to_list())
    assert ids == {"40000_101", "40000_202"}


def test_n_pixels_support() -> None:
    """n_pixels per parcel matches the synthetic grid pixel counts."""
    parcel_ids, counts = make_parcel_grid(size=6)
    probs = make_softmax(num_classes=18, size=6, seed=22)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")

    support = dict(
        zip(
            df["canonical_parcel_id"].to_list(),
            df["n_pixels"].to_list(),
            strict=True,
        )
    )
    assert support["40000_101"] == counts[101]
    assert support["40000_202"] == counts[202]


def test_ignore_parcel_excluded() -> None:
    """ParcelID == 0 (Background) never produces a row."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    assert (parcel_ids == 0).any()  # the grid does contain background
    probs = make_softmax(num_classes=18, size=6, seed=23)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")

    assert not any(cid.endswith("_0") for cid in df["canonical_parcel_id"].to_list())


def test_pred_class_matches_known_one_hot() -> None:
    """With a one-hot-per-parcel softmax, pred_class is exactly the seeded class."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    class_of = {101: 3, 202: 11}
    probs = make_constant_class_probs(parcel_ids, class_of_parcel=class_of, num_classes=18)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000").sort("canonical_parcel_id")

    pred = dict(
        zip(
            df["canonical_parcel_id"].to_list(),
            df["pred_class"].to_list(),
            strict=True,
        )
    )
    assert pred["40000_101"] == 3
    assert pred["40000_202"] == 11

    # The mean of a one-hot is the same one-hot: prob on the seeded class is 1.
    row_101 = df.filter(pl.col("canonical_parcel_id") == "40000_101")
    assert pytest.approx(row_101["prob_003"].item(), abs=_SUM_TOL) == 1.0


def test_mode_method_majority_vote() -> None:
    """method='mode' returns the majority per-pixel argmax per parcel."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    half = 6 // 2
    # Build a softmax where parcel 101 is mostly class 4 but has one class-9 pixel.
    probs = np.full((18, 6, 6), 1e-4, dtype=np.float32)
    probs[4, :, :half] = 10.0  # parcel 101 region -> class 4 dominant
    probs[9, 0, 0] = 100.0  # a single outlier pixel votes class 9
    # renormalize to a valid distribution per pixel
    probs = probs / probs.sum(axis=0, keepdims=True)

    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000", method="mode").filter(
        pl.col("canonical_parcel_id") == "40000_101"
    )
    # Majority of parcel 101 pixels argmax to class 4 despite the outlier.
    assert df["pred_class"].item() == 4


def test_empty_when_all_background() -> None:
    """An all-background grid yields an empty frame with the canonical schema."""
    parcel_ids = np.zeros((5, 5), dtype=np.int64)
    probs = make_softmax(num_classes=18, size=5, seed=24)
    df = pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")
    assert df.height == 0
    assert df.schema["canonical_parcel_id"] == pl.Utf8
    for col in PROB_COLUMNS:
        assert df.schema[col] == pl.Float32


def test_shape_mismatch_raises() -> None:
    """Inconsistent probs/parcel shapes raise ValueError."""
    probs = make_softmax(num_classes=18, size=6, seed=25)
    bad_ids = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        pixel_to_parcel_probs(probs, bad_ids, patch_id="40000")


def test_wrong_num_classes_raises() -> None:
    """A non-18-channel probs array raises ValueError."""
    probs = make_softmax(num_classes=20, size=6, seed=26)
    parcel_ids, _counts = make_parcel_grid(size=6)
    with pytest.raises(ValueError):
        pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000")


def test_invalid_method_raises() -> None:
    """An unknown reduction method raises ValueError."""
    parcel_ids, _counts = make_parcel_grid(size=6)
    probs = make_softmax(num_classes=18, size=6, seed=27)
    with pytest.raises(ValueError):
        pixel_to_parcel_probs(probs, parcel_ids, patch_id="40000", method="median")  # type: ignore[arg-type]


def test_load_pastis_parcel_ids_roundtrip(tmp_path: Path) -> None:
    """load_pastis_parcel_ids reads ANNOTATIONS/ParcelIDs_<id>.npy as (128,128)."""
    annot = tmp_path / "ANNOTATIONS"
    annot.mkdir()
    expected = np.arange(128 * 128, dtype=np.int32).reshape(128, 128)
    np.save(annot / "ParcelIDs_99999.npy", expected)

    out = load_pastis_parcel_ids("99999", tmp_path)
    assert out.shape == (128, 128)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, expected.astype(np.int64))


def test_load_pastis_parcel_ids_missing_raises(tmp_path: Path) -> None:
    """A missing ParcelIDs file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pastis_parcel_ids("00000", tmp_path)
