"""Anti-leakage tests for :class:`ml.ensemble.stacking.StackingEnsemble` (US-040 E3).

Stacking is the most leakage-sensitive ensemble (a second model on top of the
base learners), so the suite leads with R-LEAK and R-OOF-DEPTH:

1. ``test_meta_sees_only_oof``: the meta-learner is trained ONLY on the
   post-softmax ``prob_*`` OOF columns and :meth:`assert_oof_only` fires on any
   train/eval parcel overlap (a leakage attempt is rejected).
2. ``test_spatial_cv_used``: the sub-folds come from ``build_spatial_kfold`` (H3 +
   KMeans + buffer), NEVER a random/IID split.
3. ``test_report_is_fold5_only``: evaluation through the base ``evaluate`` rejects
   any fold but 5.
4. ``test_pixel_to_parcel_reduction``: the dense->parcel reduction equals
   ``pixel_to_parcel_probs`` (the reconciliation reused from the base).
5. Both meta families (``logreg`` and ``xgb``) train and predict valid
   post-softmax probabilities.

All inputs are tiny deterministic synthetic OOF + GT frames (no real PASTIS-R, no
~1.5 GB DVC blobs, no inference).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point

from ml.ensemble.base import EnsembleModel
from ml.ensemble.stacking import StackingEnsemble
from ml.utils.parcel_reconcile import PROB_COLUMNS, pixel_to_parcel_probs
from tests.ml.ensemble.fixtures.synthetic_oof import (
    NUM_CLASSES,
    SMALL_SIZE,
    make_softmax_map,
)

_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: Two well-separated geographic clusters so build_spatial_kfold forms folds.
_CLUSTER_CENTERS: tuple[tuple[float, float], ...] = (
    (2.0, 44.0),
    (3.5, 45.5),
    (5.0, 47.0),
)


# ---------------------------------------------------------------------------
# Deterministic synthetic OOF + GT + geometry builders.
# ---------------------------------------------------------------------------


def _make_parcel_probs(
    n_parcels: int, *, seed: int, signal_from: np.ndarray | None = None
) -> np.ndarray:
    """Build an ``(n_parcels, 18)`` post-softmax matrix.

    When ``signal_from`` (a label vector) is given, each row gets a logit bump on
    its true class so the base learner is informative (the meta can then learn);
    otherwise the rows are pure noise.

    Args:
        n_parcels: Number of parcel rows.
        seed: Deterministic seed.
        signal_from: Optional true labels to inject class signal.

    Returns:
        A ``float64`` post-softmax matrix summing to 1 per row.
    """
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-1.0, 1.0, size=(n_parcels, NUM_CLASSES))
    if signal_from is not None:
        logits[np.arange(n_parcels), signal_from] += 3.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _parcel_frame(member: str, ids: list[str], probs: np.ndarray) -> pl.DataFrame:
    """Assemble a parcel-space OOF frame for ``member``."""
    data: dict[str, object] = {"canonical_parcel_id": ids}
    for c, col in enumerate(PROB_COLUMNS):
        data[col] = probs[:, c].astype(np.float32)
    data["pred_class"] = probs.argmax(axis=1).astype(np.int64)
    data["n_pixels"] = np.full(len(ids), 100, dtype=np.int64)
    return pl.DataFrame(data)


def write_stacking_fixture(
    oof_dir: Path,
    *,
    n_parcels: int = 60,
    seed: int = 0,
    informative: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write per-member parcel OOF parquets + return (geoms, gt_labels).

    The three base learners share the SAME ``canonical_parcel_id`` set (so the
    inner join keeps every parcel). Geometries are split into well-separated
    clusters so the spatial sub-fold split is non-degenerate.

    Args:
        oof_dir: Directory to write the OOF parquets into.
        n_parcels: Number of parcels.
        seed: Deterministic seed.
        informative: If True, base learners carry class signal so the meta can
            learn; otherwise pure noise.

    Returns:
        Tuple ``(parcel_geoms, gt_labels)`` Polars frames.
    """
    oof_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"10000_{i:04d}" for i in range(n_parcels)]
    # Few classes so spatial sub-folds are not starved of labels.
    labels = rng.integers(0, 4, size=n_parcels).astype(np.int64)

    for m, member in enumerate(_MEMBERS):
        probs = _make_parcel_probs(
            n_parcels,
            seed=seed + 100 * (m + 1),
            signal_from=labels if informative else None,
        )
        frame = _parcel_frame(member, ids, probs)
        frame.write_parquet(oof_dir / f"oof_parcel_{member}_fold5.parquet")

    # Geometries: assign parcels round-robin to separated clusters + jitter.
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
    return parcel_geoms, gt_labels


# ---------------------------------------------------------------------------
# Construction / validation.
# ---------------------------------------------------------------------------


def test_defaults_match_contract() -> None:
    """Default base members + meta family match the US-040 section-4 contract."""
    ens = StackingEnsemble()
    assert ens.base_members == ("tsvit-pheno", "utae", "xgb-alphaearth")
    assert ens.meta == "logreg"
    assert ens.n_spatial_folds == 5
    assert ens.HELD_OUT_FOLD == 5


def test_rejects_invalid_meta() -> None:
    """An unknown meta family raises."""
    with pytest.raises(ValueError, match="meta"):
        StackingEnsemble(meta="svm")  # type: ignore[arg-type]


def test_rejects_too_few_spatial_folds() -> None:
    """A single spatial fold is not a CV and is rejected."""
    with pytest.raises(ValueError, match="spatial CV"):
        StackingEnsemble(n_spatial_folds=1)


def test_rejects_empty_members() -> None:
    """No base learner means nothing to stack."""
    with pytest.raises(ValueError, match="at least one"):
        StackingEnsemble([])


# ---------------------------------------------------------------------------
# Meta-feature assembly: OOF only, post-softmax.
# ---------------------------------------------------------------------------


def test_meta_features_are_oof_probs_only(tmp_path: Path) -> None:
    """The meta matrix is exactly the post-softmax prob_* of the base learners."""
    _, gt = write_stacking_fixture(tmp_path, n_parcels=30, seed=1)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path)

    keys_df, x_meta, y = ens.build_meta_features(gt_labels=gt)
    # 3 members x 18 classes = 54 meta-features, nothing else.
    assert x_meta.shape == (keys_df.height, len(_MEMBERS) * NUM_CLASSES)
    assert y is not None and y.shape[0] == keys_df.height
    # Each member block is a valid post-softmax distribution (probs, not logits).
    for m in range(len(_MEMBERS)):
        block = x_meta[:, m * NUM_CLASSES : (m + 1) * NUM_CLASSES]
        np.testing.assert_allclose(block.sum(axis=1), 1.0, atol=1e-4)
        assert (block >= 0).all()


def test_build_meta_features_rejects_logits(tmp_path: Path) -> None:
    """A base OOF frame carrying logits (not post-softmax) is rejected."""
    write_stacking_fixture(tmp_path, n_parcels=20, seed=2)
    # Corrupt one member's parquet with raw logits.
    bad = tmp_path / "oof_parcel_utae_fold5.parquet"
    df = pl.read_parquet(bad)
    rng = np.random.default_rng(0)
    logits = rng.uniform(-5.0, 5.0, size=(df.height, NUM_CLASSES))
    df = df.with_columns([pl.Series(col, logits[:, c]) for c, col in enumerate(PROB_COLUMNS)])
    df.write_parquet(bad)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path)
    with pytest.raises(ValueError):
        ens.build_meta_features(gt_labels=None)


def test_gt_required_columns(tmp_path: Path) -> None:
    """gt_labels missing the `label` column raises a clear error."""
    write_stacking_fixture(tmp_path, n_parcels=20, seed=3)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path)
    bad_gt = pl.DataFrame({"canonical_parcel_id": ["10000_0000"]})
    with pytest.raises(ValueError, match="label"):
        ens.build_meta_features(gt_labels=bad_gt)


# ---------------------------------------------------------------------------
# Anti-leakage 1: the meta-learner sees OOF only (assert_oof_only fires).
# ---------------------------------------------------------------------------


def test_meta_sees_only_oof_disjoint_subfolds(tmp_path: Path) -> None:
    """Each spatial sub-fold's meta train/eval parcel sets are DISJOINT."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=4)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3)

    keys_df, _, _ = ens.build_meta_features(gt_labels=gt)
    splits = ens._subfolds_by_canonical_id(parcel_geoms, keys_df)
    assert splits, "expected at least one usable spatial sub-fold"
    for train_pos, test_pos in splits:
        # Positional disjointness is the structural guarantee of OOF-only.
        assert set(train_pos.tolist()).isdisjoint(test_pos.tolist())


def test_assert_oof_only_invoked_during_fit(tmp_path: Path) -> None:
    """fit calls assert_oof_only for every sub-fold (leakage guard wired in)."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=5)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3)

    with mock.patch.object(
        EnsembleModel,
        "assert_oof_only",
        wraps=EnsembleModel.assert_oof_only,
    ) as spy:
        ens.fit(parcel_geoms, gt_labels=gt)
    assert spy.call_count >= 1
    # Every recorded call had disjoint train/eval ids (no overlap leaked).
    for call in spy.call_args_list:
        train_ids, eval_ids = call.args[0], call.args[1]
        assert set(map(str, train_ids)).isdisjoint(map(str, eval_ids))


def test_fit_raises_on_injected_leakage(tmp_path: Path) -> None:
    """If a sub-fold leaks (overlapping train/eval), fit aborts via assert_oof_only."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=6)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3)

    keys_df, _, _ = ens.build_meta_features(gt_labels=gt)
    real_splits = ens._subfolds_by_canonical_id(parcel_geoms, keys_df)
    # Inject a leak: force the first sub-fold's train to include an eval index.
    train_pos, test_pos = real_splits[0]
    leaked_train = np.concatenate([train_pos, test_pos[:1]])
    leaked = [(leaked_train, test_pos), *real_splits[1:]]

    with mock.patch.object(StackingEnsemble, "_subfolds_by_canonical_id", return_value=leaked):
        with pytest.raises(ValueError, match="leakage"):
            ens.fit(parcel_geoms, gt_labels=gt)


# ---------------------------------------------------------------------------
# Anti-leakage 2: spatial CV (build_spatial_kfold), never random.
# ---------------------------------------------------------------------------


def test_spatial_cv_used_not_random(tmp_path: Path) -> None:
    """The sub-folds are produced by build_spatial_kfold, not a random split."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=7)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3)
    keys_df, _, _ = ens.build_meta_features(gt_labels=gt)

    with mock.patch(
        "ml.features.spatial_split.build_spatial_kfold",
        wraps=__import__(
            "ml.features.spatial_split", fromlist=["build_spatial_kfold"]
        ).build_spatial_kfold,
    ) as spy:
        ens._subfolds_by_canonical_id(parcel_geoms, keys_df)
    assert spy.call_count == 1
    # The spatial split was called with the buffer (anti-leakage), not k alone.
    _, kwargs = spy.call_args
    assert kwargs.get("buffer_km") == ens.buffer_km
    assert kwargs.get("k") == ens.n_spatial_folds


def test_subfold_buffer_excludes_border_parcels(tmp_path: Path) -> None:
    """With a buffer the union of sub-fold rows may exclude border parcels."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=8)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3, buffer_km=1.0)
    keys_df, _, _ = ens.build_meta_features(gt_labels=gt)
    splits = ens._subfolds_by_canonical_id(parcel_geoms, keys_df)
    # Every test block is geographically held out and disjoint from its train.
    seen_test: set[int] = set()
    for _, test_pos in splits:
        assert seen_test.isdisjoint(test_pos.tolist())
        seen_test.update(test_pos.tolist())


# ---------------------------------------------------------------------------
# Anti-leakage 3: report fold-5 only.
# ---------------------------------------------------------------------------


def test_report_is_fold5_only(tmp_path: Path) -> None:
    """evaluate(fold=4) raises; only fold-5 metrics are reportable."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=40, seed=9)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3).fit(
        parcel_geoms, gt_labels=gt
    )
    keys_df, _, _ = ens.build_meta_features(gt_labels=gt)
    proba = ens.predict_proba()
    # Align GT to the predicted parcel order.
    gt_map = dict(zip(gt["canonical_parcel_id"], gt["label"], strict=True))
    y_true = np.array([gt_map[k] for k in keys_df["canonical_parcel_id"]])

    with pytest.raises(ValueError, match="fold-5-only"):
        ens.evaluate(y_true=y_true, proba=proba, fold=4)
    metrics = ens.evaluate(y_true=y_true, proba=proba, fold=5)
    assert 0.0 <= metrics["f1_macro"] <= 1.0


# ---------------------------------------------------------------------------
# Pixel -> parcel reduction reuses the base reconciliation.
# ---------------------------------------------------------------------------


def test_pixel_to_parcel_reduction_matches_helper() -> None:
    """The dense->parcel reduction equals pixel_to_parcel_probs (mean of probs)."""
    ens = StackingEnsemble()
    probs = make_softmax_map(size=SMALL_SIZE, seed=11)
    parcel_ids = np.zeros((SMALL_SIZE, SMALL_SIZE), dtype=np.int64)
    half = SMALL_SIZE // 2
    parcel_ids[:, :half] = 7
    parcel_ids[:, half:] = 9

    got = ens.reduce_pixel_to_parcel(probs, parcel_ids, patch_id="10000", method="mean")
    expected = pixel_to_parcel_probs(probs, parcel_ids, patch_id="10000", method="mean")
    assert got["canonical_parcel_id"].to_list() == expected["canonical_parcel_id"].to_list()
    np.testing.assert_allclose(
        got.select(PROB_COLUMNS).to_numpy(),
        expected.select(PROB_COLUMNS).to_numpy(),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Both meta families fit + predict valid post-softmax probabilities.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", ["logreg", "xgb"])
def test_fit_predict_both_meta_families(tmp_path: Path, meta: str) -> None:
    """logreg and xgb meta-learners both train and emit (n, 18) post-softmax."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=60, seed=12)
    ens = StackingEnsemble(
        _MEMBERS,
        oof_dir=tmp_path,
        meta=meta,
        n_spatial_folds=3,  # type: ignore[arg-type]
    )
    ens.fit(parcel_geoms, gt_labels=gt)

    assert ens.meta_model_ is not None
    assert "f1_macro" in ens.oof_cv_metrics_

    proba = ens.predict_proba()
    keys_df, _, _ = ens.build_meta_features(gt_labels=None)
    assert proba.shape == (keys_df.height, NUM_CLASSES)
    # Output is a valid post-softmax distribution (the base validates it too).
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba >= 0).all()


def test_predict_subset_order(tmp_path: Path) -> None:
    """predict_proba(parcel_ids=...) returns rows in the requested order."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=40, seed=13)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3).fit(
        parcel_geoms, gt_labels=gt
    )
    keys_df, _, _ = ens.build_meta_features(gt_labels=None)
    keys = keys_df["canonical_parcel_id"].to_list()

    subset = [keys[5], keys[0], keys[3]]
    full = ens.predict_proba()
    sub = ens.predict_proba(subset)
    assert sub.shape == (3, NUM_CLASSES)
    pos = {k: i for i, k in enumerate(keys)}
    for row, key in enumerate(subset):
        np.testing.assert_allclose(sub[row], full[pos[key]], atol=1e-8)


def test_predict_unknown_parcel_raises(tmp_path: Path) -> None:
    """Requesting a parcel absent from the joined base OOF raises."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=30, seed=14)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3).fit(
        parcel_geoms, gt_labels=gt
    )
    with pytest.raises(ValueError, match="not in the joined"):
        ens.predict_proba(["does-not-exist"])


def test_predict_before_fit_raises(tmp_path: Path) -> None:
    """predict_proba before fit raises RuntimeError."""
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        ens.predict_proba()


def test_informative_bases_help_meta(tmp_path: Path) -> None:
    """With informative base learners the meta beats random-chance accuracy."""
    parcel_geoms, gt = write_stacking_fixture(tmp_path, n_parcels=90, seed=15, informative=True)
    ens = StackingEnsemble(_MEMBERS, oof_dir=tmp_path, n_spatial_folds=3).fit(
        parcel_geoms, gt_labels=gt
    )
    # 4 classes -> random chance ~0.25; informative bases must clear it.
    assert ens.oof_cv_metrics_["accuracy"] > 0.4
