"""Tests for :class:`ml.ensemble.voting.VotingEnsemble` (US-040, E1).

Anti-leakage first (R-LEAK): the soft-vote averages POST-softmax probabilities,
never logits, and reports on fold-5 ONLY. The golden-value tests build three
members with KNOWN softmax maps so the mean is exact, then assert:

1. ``predict_proba`` == the arithmetic mean of the members' post-softmax maps.
2. A member map that is NOT post-softmax (logits, sum != 1) is rejected.
3. The dense output shape is ``(18, 128, 128)`` (single) / ``(N, 18, 128, 128)``.
4. ``evaluate`` is fold-5 only (``fold=4`` raises) and runs against a real-shaped
   ground truth (here mocked: the production loader reads PASTIS-R, too heavy for
   a unit test).

The OOF parquet readers are exercised end-to-end against tiny REAL parquet
fixtures (``write_pixel_oof``); the PASTIS-R ground-truth loader is mocked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.ensemble.voting import DEFAULT_VOTING_MEMBERS, VotingEnsemble
from ml.eval.oof.parquet_io import write_softmax_parquet
from tests.ml.ensemble.fixtures.synthetic_oof import (
    NUM_CLASSES,
    SMALL_SIZE,
    make_softmax_map,
    write_pixel_oof,
)

#: The three default homogeneous voters (R-VOTE).
_MEMBERS = DEFAULT_VOTING_MEMBERS


# ---------------------------------------------------------------------------
# Helpers: write members with KNOWN maps so the mean is exact (golden values).
# ---------------------------------------------------------------------------


def _write_known_pixel_oof(
    oof_dir: Path,
    member: str,
    *,
    softmax_by_patch: dict[str, np.ndarray],
    size: int = SMALL_SIZE,
) -> Path:
    """Write a pixel OOF parquet whose ``softmax`` maps are supplied verbatim.

    Args:
        oof_dir: Directory to write into.
        member: Member name (file stem ``oof_{member}_fold5.parquet``).
        softmax_by_patch: ``{patch_id: softmax (18, size, size)}`` post-softmax.
        size: Spatial side of the maps.

    Returns:
        Path of the written parquet.
    """
    oof_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for pid, sm in softmax_by_patch.items():
        rows.append(
            {
                "patch_id": pid,
                "fold": 5,
                "held_out": True,
                "model": member,
                "status": "ok",
                "softmax": sm,
                "pred": np.asarray(sm).argmax(axis=0).astype(np.int8),
                "code_version": "test",
                "data_version": "test",
            }
        )
    path = oof_dir / f"oof_{member}_fold5.parquet"
    # float32 storage keeps the known maps near-exact (float16 would lose bits).
    write_softmax_parquet(rows, path, num_classes=NUM_CLASSES, size=size, dtype="float32")
    return path


# ---------------------------------------------------------------------------
# Construction / defaults.
# ---------------------------------------------------------------------------


def test_default_members_are_three_dense(tmp_path: Path) -> None:
    """The default terna is TSViT-pheno + U-TAE + U-Net (R-VOTE)."""
    ens = VotingEnsemble(oof_dir=tmp_path)
    assert ens.members == ("tsvit-pheno", "utae", "unet")
    assert _MEMBERS == ("tsvit-pheno", "utae", "unet")


def test_members_are_substitutable(tmp_path: Path) -> None:
    """The third voter can be swapped (e.g. deeplabv3plus) -- R-VOTE."""
    ens = VotingEnsemble(members=("tsvit-pheno", "utae", "deeplabv3plus"), oof_dir=tmp_path)
    assert ens.members == ("tsvit-pheno", "utae", "deeplabv3plus")


def test_single_member_rejected(tmp_path: Path) -> None:
    """A single member is not an ensemble -> ValueError."""
    with pytest.raises(ValueError, match="at least 2 members"):
        VotingEnsemble(members=("utae",), oof_dir=tmp_path)


def test_fit_is_noop_returns_self(tmp_path: Path) -> None:
    """Voting is parameter-free: fit returns self without doing work."""
    ens = VotingEnsemble(oof_dir=tmp_path)
    assert ens.fit() is ens


# ---------------------------------------------------------------------------
# Golden value: soft-vote == mean of post-softmax maps.
# ---------------------------------------------------------------------------


def test_soft_vote_is_mean_of_probs(tmp_path: Path) -> None:
    """predict_proba == arithmetic mean of the members' post-softmax maps."""
    pid = "10000"
    maps = {
        m: make_softmax_map(size=SMALL_SIZE, seed=seed)
        for m, seed in zip(_MEMBERS, (1, 2, 3), strict=True)
    }
    for member, sm in maps.items():
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch={pid: sm})

    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])

    expected = np.mean(np.stack([maps[m].astype(np.float64) for m in _MEMBERS], axis=0), axis=0)
    assert proba.shape == (NUM_CLASSES, SMALL_SIZE, SMALL_SIZE)
    np.testing.assert_allclose(proba, expected, atol=1e-6)
    # The averaged map is itself post-softmax (sum-to-1 over the class axis).
    np.testing.assert_allclose(proba.sum(axis=0), 1.0, atol=1e-5)


def test_soft_vote_known_uniform_maps(tmp_path: Path) -> None:
    """With three identical uniform maps the vote is that uniform map exactly."""
    pid = "20000"
    uniform = np.full((NUM_CLASSES, SMALL_SIZE, SMALL_SIZE), 1.0 / NUM_CLASSES, dtype=np.float32)
    for member in _MEMBERS:
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch={pid: uniform})

    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])
    np.testing.assert_allclose(proba, uniform.astype(np.float64), atol=1e-6)


def test_predict_is_argmax_of_mean(tmp_path: Path) -> None:
    """predict == argmax over the averaged class axis (deterministic)."""
    pid = "30000"
    maps = {
        m: make_softmax_map(size=SMALL_SIZE, seed=seed)
        for m, seed in zip(_MEMBERS, (4, 5, 6), strict=True)
    }
    for member, sm in maps.items():
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch={pid: sm})

    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])
    pred = ens.predict([pid])
    assert pred.shape == (SMALL_SIZE, SMALL_SIZE)
    np.testing.assert_array_equal(pred, proba.argmax(axis=0))


# ---------------------------------------------------------------------------
# Anti-leakage: never averages logits.
# ---------------------------------------------------------------------------


def test_never_averages_logits(tmp_path: Path) -> None:
    """A member map that is NOT post-softmax (logits) is rejected before averaging."""
    pid = "40000"
    rng = np.random.default_rng(0)
    logit_map = rng.uniform(-5.0, 5.0, size=(NUM_CLASSES, SMALL_SIZE, SMALL_SIZE))
    good_map = make_softmax_map(size=SMALL_SIZE, seed=7)

    # tsvit-pheno carries logits; the other two are valid softmax.
    _write_known_pixel_oof(tmp_path, "tsvit-pheno", softmax_by_patch={pid: logit_map})
    _write_known_pixel_oof(tmp_path, "utae", softmax_by_patch={pid: good_map})
    _write_known_pixel_oof(tmp_path, "unet", softmax_by_patch={pid: good_map})

    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(ValueError, match=r"logits|sum to 1|negative"):
        ens.predict_proba([pid])


def test_inputs_are_post_softmax(tmp_path: Path) -> None:
    """The maps consumed by the vote are post-softmax (sum-to-1, non-negative)."""
    pid = "41000"
    sm = make_softmax_map(size=SMALL_SIZE, seed=11)
    for member in _MEMBERS:
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch={pid: sm})

    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])
    assert (proba >= -1e-9).all()
    np.testing.assert_allclose(proba.sum(axis=0), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


def test_shape_single_patch(tmp_path: Path) -> None:
    """A single patch id yields a dense (18, 128-shaped) map without a batch axis."""
    pid = "50000"
    for member in _MEMBERS:
        write_pixel_oof(tmp_path, member, patch_ids=(pid,), seed=0)
    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])
    assert proba.shape == (NUM_CLASSES, SMALL_SIZE, SMALL_SIZE)


def test_shape_multi_patch(tmp_path: Path) -> None:
    """Several patch ids yield (N, 18, H, W) in the requested order."""
    pids = ("60000", "60001", "60002")
    for member in _MEMBERS:
        write_pixel_oof(tmp_path, member, patch_ids=pids, seed=0)
    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba(list(pids))
    assert proba.shape == (3, NUM_CLASSES, SMALL_SIZE, SMALL_SIZE)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_preserves_order(tmp_path: Path) -> None:
    """The output rows follow the requested patch_ids order, not the file order."""
    pids = ("70000", "70001", "70002")
    # Distinct per-patch maps so order is observable.
    maps = {pid: make_softmax_map(size=SMALL_SIZE, seed=100 + i) for i, pid in enumerate(pids)}
    for member in _MEMBERS:
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch=dict(maps))

    ens = VotingEnsemble(oof_dir=tmp_path)
    reordered = [pids[2], pids[0], pids[1]]
    proba = ens.predict_proba(reordered)
    for row, pid in enumerate(reordered):
        np.testing.assert_allclose(proba[row], maps[pid].astype(np.float64), atol=1e-6)


def test_empty_patch_ids_raises(tmp_path: Path) -> None:
    """An empty patch_ids list raises."""
    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one patch_id"):
        ens.predict_proba([])


def test_missing_patch_in_member_raises(tmp_path: Path) -> None:
    """A patch present in some members but absent in another raises."""
    pid_common = "80000"
    pid_only_two = "80001"
    sm = make_softmax_map(size=SMALL_SIZE, seed=9)
    _write_known_pixel_oof(
        tmp_path,
        "tsvit-pheno",
        softmax_by_patch={pid_common: sm, pid_only_two: sm},
    )
    _write_known_pixel_oof(tmp_path, "utae", softmax_by_patch={pid_common: sm, pid_only_two: sm})
    # unet lacks pid_only_two.
    _write_known_pixel_oof(tmp_path, "unet", softmax_by_patch={pid_common: sm})

    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(ValueError, match="absent from member"):
        ens.predict_proba([pid_only_two])


def test_missing_member_oof_raises(tmp_path: Path) -> None:
    """A missing member parquet surfaces the dvc-pull FileNotFoundError."""
    pid = "90000"
    sm = make_softmax_map(size=SMALL_SIZE, seed=3)
    # Only two of the three members exist on disk.
    _write_known_pixel_oof(tmp_path, "tsvit-pheno", softmax_by_patch={pid: sm})
    _write_known_pixel_oof(tmp_path, "utae", softmax_by_patch={pid: sm})
    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        ens.predict_proba([pid])


# ---------------------------------------------------------------------------
# Anti-leakage: evaluate is fold-5 only (ground truth mocked).
# ---------------------------------------------------------------------------


def _setup_two_patch_voting(tmp_path: Path) -> tuple[VotingEnsemble, list[str]]:
    """Write a 2-patch voting ensemble and return it with the patch ids."""
    pids = ["a1", "a2"]
    for member in _MEMBERS:
        write_pixel_oof(tmp_path, member, patch_ids=tuple(pids), seed=0)
    return VotingEnsemble(oof_dir=tmp_path), pids


def test_evaluate_patches_fold5_against_mocked_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_patches scores the vote against a (mocked) fold-5 ground truth.

    The production ``load_ground_truth`` reads PASTIS-R (too heavy here), so we
    patch it to return the model's own predictions -> perfect score, proving the
    plumbing (predict -> evaluate fold-5) is wired correctly.
    """
    ens, pids = _setup_two_patch_voting(tmp_path)
    preds = ens.predict(pids)
    monkeypatch.setattr(ens, "load_ground_truth", lambda patch_ids: preds)
    metrics = ens.evaluate_patches(pids, fold=5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_evaluate_patches_fold4_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_patches(fold=4) -> ValueError (fold-4 was selection, never reported)."""
    ens, pids = _setup_two_patch_voting(tmp_path)
    preds = ens.predict(pids)
    monkeypatch.setattr(ens, "load_ground_truth", lambda patch_ids: preds)
    with pytest.raises(ValueError, match="fold-5-only"):
        ens.evaluate_patches(pids, fold=4)


def test_evaluate_imperfect_gt_below_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ground truth that disagrees with the vote drives accuracy below 1."""
    ens, pids = _setup_two_patch_voting(tmp_path)
    preds = ens.predict(pids)
    # Flip every label that is not the ignore index -> guaranteed disagreement.
    flipped = np.where(preds == 255, 255, (preds + 1) % NUM_CLASSES)
    monkeypatch.setattr(ens, "load_ground_truth", lambda patch_ids: flipped)
    metrics = ens.evaluate_patches(pids, fold=5)
    assert metrics["accuracy"] < 1.0


def test_evaluate_direct_proba_fold5(tmp_path: Path) -> None:
    """The base evaluate(proba=...) path works on a flattened voted map."""
    pid = "b1"
    sm = make_softmax_map(size=SMALL_SIZE, seed=42)
    for member in _MEMBERS:
        _write_known_pixel_oof(tmp_path, member, softmax_by_patch={pid: sm})
    ens = VotingEnsemble(oof_dir=tmp_path)
    proba = ens.predict_proba([pid])  # (18, H, W)
    y_pred = proba.argmax(axis=0).reshape(-1)
    metrics = ens.evaluate(y_true=y_pred.copy(), y_pred=y_pred, fold=5)
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ground-truth loader alignment (PASTIS dataset mocked).
# ---------------------------------------------------------------------------


def test_load_ground_truth_aligns_to_patch_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_ground_truth stacks labels in the requested patch_ids order.

    The PASTIS dataset is mocked: it exposes patch_ids ["p1","p2","p3"] each with
    a distinct constant label map; we request them out of order and assert the
    output rows follow the request, not the dataset order.
    """
    import ml.ensemble.voting as voting_mod

    class _FakeDataset:
        def __init__(self, **_kw: object) -> None:
            self.patch_ids = ["p1", "p2", "p3"]
            self._labels = {
                "p1": np.zeros((SMALL_SIZE, SMALL_SIZE), dtype=np.int64),
                "p2": np.ones((SMALL_SIZE, SMALL_SIZE), dtype=np.int64),
                "p3": np.full((SMALL_SIZE, SMALL_SIZE), 2, dtype=np.int64),
            }

        def __getitem__(self, pos: int) -> tuple[object, np.ndarray]:
            pid = self.patch_ids[pos]
            return object(), self._labels[pid]

    monkeypatch.setattr(voting_mod, "PASTISSegmentationDataset", _FakeDataset, raising=False)
    # Patch the symbol used inside load_ground_truth (imported lazily).
    import ml.data.pastis_seg_dataset as ds_mod

    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeDataset)

    ens = VotingEnsemble(oof_dir=tmp_path)
    gt = ens.load_ground_truth(["p3", "p1"])
    assert gt.shape == (2, SMALL_SIZE, SMALL_SIZE)
    assert int(gt[0].max()) == 2  # p3 first
    assert int(gt[1].max()) == 0  # p1 second


def test_load_ground_truth_unknown_patch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting a patch not in the fold-5 split raises."""

    class _FakeDataset:
        def __init__(self, **_kw: object) -> None:
            self.patch_ids = ["p1"]

        def __getitem__(self, pos: int) -> tuple[object, np.ndarray]:
            return object(), np.zeros((SMALL_SIZE, SMALL_SIZE), dtype=np.int64)

    import ml.data.pastis_seg_dataset as ds_mod

    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeDataset)

    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(ValueError, match="not in the fold-5 split"):
        ens.load_ground_truth(["nope"])


def test_load_ground_truth_empty_raises(tmp_path: Path) -> None:
    """An empty patch_ids list raises before touching the dataset."""
    ens = VotingEnsemble(oof_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one patch_id"):
        ens.load_ground_truth([])


# ---------------------------------------------------------------------------
# Index helper edge cases.
# ---------------------------------------------------------------------------


def test_patch_softmax_index_skips_none() -> None:
    """A missing-checkpoint row (softmax=None) is skipped in the index."""
    df = pl.DataFrame(
        {"patch_id": ["x", "y"], "softmax": [None, make_softmax_map(seed=0)]},
        schema_overrides={"softmax": pl.Object},
    )
    index = VotingEnsemble._patch_softmax_index("utae", df)
    assert "x" not in index
    assert "y" in index


def test_patch_softmax_index_requires_columns() -> None:
    """A frame without the expected columns raises a clear error."""
    df = pl.DataFrame({"patch_id": ["x"]})
    with pytest.raises(ValueError, match=r"patch_id.*softmax|softmax"):
        VotingEnsemble._patch_softmax_index("utae", df)
