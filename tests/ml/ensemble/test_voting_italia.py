"""Tests for :class:`ml.ensemble.voting_italia.ItaliaPixelVotingEnsemble` (US-079).

The adapter reuses the EPIC 6 WINNER (``WeightedVotingEnsemble``) weight learner
over DENSE Italian member predictions, learning the convex vote OUT-OF-FOLD by
leave-one-spatial-fold-out. The tests build small synthetic dense member maps
(post-softmax, seeded) over patches spanning >= 2 spatial folds and assert:

- ``load_member_softmax`` round-trips a fine-tune ``.npz`` dump.
- ``fit_predict`` learns CONVEX weights (``w_i >= 0``, ``sum == 1``).
- The OOF CV is anti-leakage: ``assert_oof_only`` never fires (train/test patches
  are disjoint per held-out fold), and the per-fold diagnostics are produced.
- The vote favours the BETTER member (a near-perfect member outweighs a noisy one).
- The blended maps are post-softmax with the right ``(K, H, W)`` shape.
- Fewer than 2 members is rejected; a single test fold degrades with a warning.

All numpy, fixed RNG; the underlying ``WeightedVotingEnsemble`` is constructed but
its PASTIS ``fit`` (which needs OOF parquet blobs) is never called.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.ensemble.voting_italia import (
    DenseMemberPreds,
    ItaliaPixelVotingEnsemble,
    load_member_softmax,
)

_K = 4  # background + 3 crop classes
_H = _W = 8


def _onehot_map(label_map: np.ndarray, *, sharp: float = 0.9) -> np.ndarray:
    """A post-softmax (K, H, W) map peaked on ``label_map`` (rest spread evenly)."""
    k = _K
    probs = np.full((k, *label_map.shape), (1.0 - sharp) / (k - 1), dtype=np.float32)
    for c in range(k):
        probs[c][label_map == c] = sharp
    # Renormalise exactly to 1 along the class axis.
    probs /= probs.sum(axis=0, keepdims=True)
    return probs.astype(np.float32)


def _noisy_map(label_map: np.ndarray, *, seed: int) -> np.ndarray:
    """A weak post-softmax map only loosely correlated with the truth."""
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-1.0, 1.0, size=(_K, *label_map.shape)).astype(np.float32)
    for c in range(_K):
        logits[c][label_map == c] += 0.3  # slight signal
    logits -= logits.max(axis=0, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


def _toy_problem(seed: int = 0) -> tuple[dict[int, np.ndarray], dict[int, int], list[np.ndarray]]:
    """Build 4 patches over 2 spatial folds with crop-only ground-truth masks."""
    rng = np.random.default_rng(seed)
    masks: dict[int, np.ndarray] = {}
    folds: dict[int, int] = {}
    label_maps: list[np.ndarray] = []
    for pid in range(4):
        # Crop ids 1..3 (no background) so every pixel is supervised.
        lm = rng.integers(1, _K, size=(_H, _W)).astype(np.int64)
        masks[pid] = lm
        folds[pid] = 0 if pid < 2 else 1  # 2 patches per spatial fold
        label_maps.append(lm)
    return masks, folds, label_maps


def _members(label_maps: list[np.ndarray], *, strong: bool) -> dict[str, DenseMemberPreds]:
    """Build 3 dense members; if ``strong``, the first is near-perfect."""
    m_a, m_b, m_c = {}, {}, {}
    for pid, lm in enumerate(label_maps):
        m_a[pid] = _onehot_map(lm, sharp=0.97) if strong else _noisy_map(lm, seed=10 + pid)
        m_b[pid] = _noisy_map(lm, seed=100 + pid)
        m_c[pid] = _noisy_map(lm, seed=200 + pid)
    return {
        "tsvit-pheno": DenseMemberPreds("tsvit-pheno", m_a, _K),
        "utae": DenseMemberPreds("utae", m_b, _K),
        "tsvit-pheno-fullm": DenseMemberPreds("tsvit-pheno-fullm", m_c, _K),
    }


# --------------------------------------------------------------------------- #
# load_member_softmax round-trip.
# --------------------------------------------------------------------------- #
def test_load_member_softmax_roundtrip(tmp_path: Path) -> None:
    """A saved test_softmax.npz reloads as DenseMemberPreds keyed by patch id."""
    _, _, label_maps = _toy_problem()
    probs = {str(pid): _onehot_map(lm) for pid, lm in enumerate(label_maps)}
    npz = tmp_path / "test_softmax.npz"
    np.savez_compressed(npz, **probs)

    loaded = load_member_softmax("tsvit-pheno", npz)
    assert loaded.member == "tsvit-pheno"
    assert loaded.num_classes == _K
    assert set(loaded.probs_by_patch) == {0, 1, 2, 3}
    np.testing.assert_allclose(loaded.probs_by_patch[0], probs["0"])


def test_load_member_softmax_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="member softmax dump"):
        load_member_softmax("utae", tmp_path / "absent.npz")


# --------------------------------------------------------------------------- #
# Construction guard.
# --------------------------------------------------------------------------- #
def test_single_member_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 members"):
        ItaliaPixelVotingEnsemble(("tsvit-pheno",), num_classes=_K)


# --------------------------------------------------------------------------- #
# fit_predict: convex weights, OOF, blended maps.
# --------------------------------------------------------------------------- #
def test_fit_predict_learns_convex_weights_oof() -> None:
    """The learned weights are convex and the OOF CV runs over >= 2 folds."""
    masks, folds, label_maps = _toy_problem()
    members = _members(label_maps, strong=True)
    ens = ItaliaPixelVotingEnsemble(
        ("tsvit-pheno", "utae", "tsvit-pheno-fullm"),
        num_classes=_K,
        n_restarts=4,
    )
    result = ens.fit_predict(members, masks, folds)

    # 3 convex weights: non-negative and sum to 1.
    assert result.weights.shape == (3,)
    assert (result.weights >= -1e-9).all()
    assert float(result.weights.sum()) == pytest.approx(1.0, abs=1e-6)
    # OOF estimate exists because there are 2 spatial folds (leave-one-out CV).
    assert len(result.per_fold) == 2
    assert not np.isnan(result.oof_f1_macro)
    for fold_row in result.per_fold:
        assert "weights" in fold_row and "n_pixels" in fold_row
    # weight_map exposes the interpretable vote (AC2).
    wmap = result.weight_map()
    assert set(wmap) == {"tsvit-pheno", "utae", "tsvit-pheno-fullm"}
    assert sum(wmap.values()) == pytest.approx(1.0, abs=1e-4)


def test_fit_predict_favours_the_better_member() -> None:
    """A near-perfect member earns more weight than the noisy ones."""
    masks, folds, label_maps = _toy_problem(seed=3)
    members = _members(label_maps, strong=True)
    ens = ItaliaPixelVotingEnsemble(
        ("tsvit-pheno", "utae", "tsvit-pheno-fullm"), num_classes=_K, n_restarts=6
    )
    result = ens.fit_predict(members, masks, folds)
    wmap = result.weight_map()
    # The strong member (tsvit-pheno) carries the largest weight.
    assert wmap["tsvit-pheno"] == max(wmap.values())
    assert wmap["tsvit-pheno"] > wmap["utae"]


def test_fit_predict_blended_maps_are_post_softmax() -> None:
    """The blended dense maps keep the (K, H, W) shape and sum to 1 per pixel."""
    masks, folds, label_maps = _toy_problem()
    members = _members(label_maps, strong=True)
    ens = ItaliaPixelVotingEnsemble(
        ("tsvit-pheno", "utae", "tsvit-pheno-fullm"), num_classes=_K, n_restarts=4
    )
    result = ens.fit_predict(members, masks, folds)
    assert set(result.blended_probs_by_patch) == {0, 1, 2, 3}
    for blended in result.blended_probs_by_patch.values():
        assert blended.shape == (_K, _H, _W)
        np.testing.assert_allclose(blended.sum(axis=0), 1.0, atol=1e-4)
        assert (blended >= -1e-6).all()


def test_fit_predict_oof_is_leak_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """assert_oof_only never fires: each held-out fold's patches are disjoint.

    We spy on EnsembleModel.assert_oof_only to prove it is called per fold with
    disjoint train/test patch ids (a real overlap would raise inside it).
    """
    from ml.ensemble.base import EnsembleModel

    masks, folds, label_maps = _toy_problem()
    members = _members(label_maps, strong=True)
    seen: list[tuple[set[int], set[int]]] = []
    original = EnsembleModel.assert_oof_only

    def _spy(train_ids, test_ids, *, context="meta-learner"):  # type: ignore[no-untyped-def]
        seen.append((set(map(int, train_ids)), set(map(int, test_ids))))
        return original(train_ids, test_ids, context=context)

    monkeypatch.setattr(EnsembleModel, "assert_oof_only", staticmethod(_spy))
    ItaliaPixelVotingEnsemble(("tsvit-pheno", "utae"), num_classes=_K, n_restarts=3).fit_predict(
        members, masks, folds
    )

    assert len(seen) == 2  # one OOF split per spatial fold
    for train_set, test_set in seen:
        assert train_set.isdisjoint(test_set)  # anti-leakage holds


def test_fit_predict_single_fold_degrades_with_warning() -> None:
    """One spatial fold -> no OOF CV; the OOF estimate is NaN, honestly."""
    masks, _, label_maps = _toy_problem()
    folds_one = {pid: 0 for pid in range(4)}  # every patch in the same fold
    members = _members(label_maps, strong=True)
    ens = ItaliaPixelVotingEnsemble(("tsvit-pheno", "utae"), num_classes=_K, n_restarts=3)
    result = ens.fit_predict(members, masks, folds_one)
    assert result.per_fold == []
    assert np.isnan(result.oof_f1_macro)
    # The production weights are still learned (refit on all pixels).
    assert float(result.weights.sum()) == pytest.approx(1.0, abs=1e-6)


def test_fit_predict_unaligned_members_raise() -> None:
    """Members predicting disjoint patch sets share no common patch -> ValueError."""
    masks, folds, label_maps = _toy_problem()
    a = {0: _onehot_map(label_maps[0]), 1: _onehot_map(label_maps[1])}
    b = {2: _onehot_map(label_maps[2]), 3: _onehot_map(label_maps[3])}
    members = {
        "tsvit-pheno": DenseMemberPreds("tsvit-pheno", a, _K),
        "utae": DenseMemberPreds("utae", b, _K),
    }
    ens = ItaliaPixelVotingEnsemble(("tsvit-pheno", "utae"), num_classes=_K, n_restarts=2)
    with pytest.raises(ValueError, match="no patch is predicted by every member"):
        ens.fit_predict(members, masks, folds)
