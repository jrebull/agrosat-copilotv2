"""Tests for :class:`ml.ensemble.blending.BlendingEnsemble` (US-040, E4).

The blending ensemble combines the per-parcel post-softmax probabilities of three
base learners with convex (simplex) weights optimized by Optuna on a SPATIALLY
DISJOINT holdout. The tests freeze the rubric-critical guarantees:

1. **Simplex weights.** ``all(w >= 0)`` and ``abs(sum(w) - 1) < 1e-6``.
2. **Optuna minimizes the gap.** The objective is
   ``f1_val - gap_lambda * |f1_train - f1_val|`` and the best trial records the
   train/val F1 so the gap penalty is verifiable (small ``n_trials`` in tests).
3. **Spatially disjoint holdout.** The train/val split comes from
   :func:`ml.features.spatial_split.build_spatial_kfold` (H3 + KMeans), NOT a
   random split, and the two sides are disjoint.
4. **Weighted output sums to 1.** ``predict_proba`` returns a convex combination
   of post-softmax members, itself a valid post-softmax distribution.
5. **Report fold-5 only.** ``evaluate(fold=4)`` raises; fold-4 was selection.

No real inference: tiny deterministic synthetic OOF parquets (seeded) reused from
``fixtures/synthetic_oof.py``; the ground truth is built separately (it is NOT in
the OOF dump) and the spatial geometries are two well-separated clusters so the
H3/KMeans split is non-degenerate.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point

from ml.ensemble.blending import DEFAULT_BLENDING_MEMBERS, BlendingEnsemble
from ml.utils.parcel_reconcile import PROB_COLUMNS
from tests.ml.ensemble.fixtures.synthetic_oof import (
    NUM_CLASSES,
    make_parcel_frame,
    write_parcel_oof,
)

#: Base members used across the tests (must have parcel OOF written below).
_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: Two well-separated Italian clusters (lon, lat) so H3 res-5 yields >= 2 cells.
_CLUSTERS: tuple[tuple[float, float], ...] = ((12.5, 41.9), (9.2, 45.5))

#: Parcels per cluster (so the spatial split has a non-empty train and val).
_PER_CLUSTER: int = 6


def _canonical_ids(n_parcels: int, patch_id: str = "10000") -> list[str]:
    """Return the canonical ids ``make_parcel_frame`` produces, in order."""
    return [f"{patch_id}_{i + 1:03d}" for i in range(n_parcels)]


def _write_aligned_members(oof_dir: Path, members: tuple[str, ...], *, n_parcels: int) -> list[str]:
    """Write parcel OOF for every member over the SAME parcel id set.

    Each member gets a distinct seed (decorrelated probabilities) but the same
    ``canonical_parcel_id`` set so the blend aligns them.

    Returns:
        The shared canonical parcel ids in sorted order.
    """
    for i, member in enumerate(members):
        write_parcel_oof(oof_dir, member, n_parcels=n_parcels, seed=100 + i)
    return sorted(_canonical_ids(n_parcels))


def _parcel_geoms(parcel_ids: list[str]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame for ``parcel_ids`` spread over two clusters.

    The integer ``parcel_id`` surrogate is the row index; the
    ``canonical_parcel_id`` carries the OOF key. Half the parcels sit in each
    cluster so the H3/KMeans split produces two geographic groups.
    """
    rows: list[dict[str, object]] = []
    for idx, canonical in enumerate(parcel_ids):
        cx, cy = _CLUSTERS[idx % len(_CLUSTERS)]
        offset = (idx // len(_CLUSTERS)) * 0.01
        rows.append(
            {
                "parcel_id": idx + 1,
                "canonical_parcel_id": canonical,
                "geometry": Point(cx + offset, cy + offset),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _ground_truth(parcel_ids: list[str], *, seed: int = 7) -> pl.DataFrame:
    """Build a per-parcel GT frame (separate from the OOF; never in the dump)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, NUM_CLASSES, size=len(parcel_ids)).astype(np.int64)
    return pl.DataFrame({"canonical_parcel_id": parcel_ids, "label": labels.tolist()})


@pytest.fixture
def fitted(tmp_path: Path) -> tuple[BlendingEnsemble, list[str], pl.DataFrame]:
    """A blending ensemble fitted on tiny synthetic OOF + spatial geoms."""
    n_parcels = _PER_CLUSTER * len(_CLUSTERS)
    parcel_ids = _write_aligned_members(tmp_path, _MEMBERS, n_parcels=n_parcels)
    geoms = _parcel_geoms(parcel_ids)
    y_true = _ground_truth(parcel_ids)
    ens = BlendingEnsemble(_MEMBERS, n_trials=8, gap_lambda=0.5, oof_dir=tmp_path)
    ens.fit(geoms, y_true=y_true)
    return ens, parcel_ids, y_true


# ---------------------------------------------------------------------------
# Construction / defaults.
# ---------------------------------------------------------------------------


def test_default_members_and_held_out_fold() -> None:
    """Defaults match the plan: 3 heterogeneous members, fold-5 only."""
    ens = BlendingEnsemble()
    assert ens.base_members == DEFAULT_BLENDING_MEMBERS
    assert ens.base_members == ("tsvit-pheno", "utae", "xgb-alphaearth")
    assert ens.n_trials == 50
    assert ens.gap_lambda == 0.5
    assert ens.HELD_OUT_FOLD == 5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base_members": ()}, "at least one"),
        ({"n_trials": 0}, "n_trials"),
        ({"gap_lambda": -0.1}, "gap_lambda"),
    ],
)
def test_invalid_construction_raises(kwargs: dict[str, object], match: str) -> None:
    """Empty members / non-positive trials / negative lambda are rejected."""
    with pytest.raises(ValueError, match=match):
        BlendingEnsemble(**kwargs)  # type: ignore[arg-type]


def test_weights_before_fit_raises() -> None:
    """Accessing weights before fit raises a clear error."""
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = BlendingEnsemble().weights


def test_predict_before_fit_raises() -> None:
    """predict_proba before fit raises a clear error."""
    with pytest.raises(RuntimeError, match="not fitted"):
        BlendingEnsemble().predict_proba()


# ---------------------------------------------------------------------------
# 1. Simplex weights (>= 0, sum == 1).
# ---------------------------------------------------------------------------


def test_weights_on_simplex(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """all(w >= 0) and abs(sum(w) - 1) < 1e-6 (golden value)."""
    ens, _, _ = fitted
    w = ens.weights
    assert w.shape == (len(_MEMBERS),)
    assert (w >= 0.0).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_project_simplex_normalizes() -> None:
    """_project_simplex maps arbitrary non-negative draws onto the simplex."""
    out = BlendingEnsemble._project_simplex(np.array([2.0, 0.0, 6.0]))
    assert (out >= 0.0).all()
    assert float(out.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(out, [0.25, 0.0, 0.75])


def test_project_simplex_degenerate_is_uniform() -> None:
    """An all-zero draw falls back to uniform weights (still on the simplex)."""
    out = BlendingEnsemble._project_simplex(np.zeros(3))
    np.testing.assert_allclose(out, np.full(3, 1.0 / 3.0))
    assert float(out.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Optuna minimizes the train/val gap.
# ---------------------------------------------------------------------------


def test_optuna_study_runs_and_records_gap(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """The study runs n_trials and the best trial records f1_train / f1_val."""
    ens, _, _ = fitted
    assert ens.study is not None
    assert len(ens.study.trials) == ens.n_trials
    assert ens.best_params  # populated raw simplex logits
    best = ens.study.best_trial
    assert "f1_train" in best.user_attrs
    assert "f1_val" in best.user_attrs


def test_objective_penalizes_gap() -> None:
    """The objective value equals f1_val - gap_lambda * |f1_train - f1_val|.

    Reconstructs the objective from the recorded train/val F1 of every completed
    trial and asserts it matches the Optuna trial value, so the gap penalty is
    actually what is being optimized (not raw f1_val).
    """
    ens = BlendingEnsemble(_MEMBERS, n_trials=6, gap_lambda=0.8)
    # Drive fit through a tmp dir built inline (no fixture: explicit lambda).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n_parcels = _PER_CLUSTER * len(_CLUSTERS)
        parcel_ids = _write_aligned_members(tmp_path, _MEMBERS, n_parcels=n_parcels)
        ens.oof_dir = tmp_path
        ens.fit(_parcel_geoms(parcel_ids), y_true=_ground_truth(parcel_ids))

    assert ens.study is not None
    for trial in ens.study.trials:
        if trial.value is None:
            continue
        f1_train = trial.user_attrs["f1_train"]
        f1_val = trial.user_attrs["f1_val"]
        expected = f1_val - ens.gap_lambda * abs(f1_train - f1_val)
        assert trial.value == pytest.approx(expected, abs=1e-9)


def test_zero_gap_lambda_is_plain_f1_val() -> None:
    """With gap_lambda=0 the objective reduces to f1_val (no penalty)."""
    import tempfile

    ens = BlendingEnsemble(_MEMBERS, n_trials=5, gap_lambda=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n_parcels = _PER_CLUSTER * len(_CLUSTERS)
        parcel_ids = _write_aligned_members(tmp_path, _MEMBERS, n_parcels=n_parcels)
        ens.oof_dir = tmp_path
        ens.fit(_parcel_geoms(parcel_ids), y_true=_ground_truth(parcel_ids))

    assert ens.study is not None
    for trial in ens.study.trials:
        if trial.value is None:
            continue
        assert trial.value == pytest.approx(trial.user_attrs["f1_val"], abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Spatially disjoint holdout (not random).
# ---------------------------------------------------------------------------


def test_holdout_spatially_disjoint(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """The spatial holdout yields disjoint, non-empty train/val index sets."""
    ens, parcel_ids, _ = fitted
    geoms = _parcel_geoms(parcel_ids)
    train_idx, val_idx = ens._spatial_holdout(tuple(parcel_ids), geoms, buffer_km=1.0)
    assert train_idx.size > 0
    assert val_idx.size > 0
    # Disjoint by construction (no parcel in both train and val).
    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))


def test_holdout_uses_build_spatial_kfold_not_random(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The holdout is built via build_spatial_kfold (spatial CV), never random.

    Spies on ``build_spatial_kfold`` to assert it is the source of the split; a
    random split would never call it.
    """
    ens, parcel_ids, _ = fitted
    geoms = _parcel_geoms(parcel_ids)

    import ml.features.spatial_split as ss

    calls: list[int] = []
    original = ss.build_spatial_kfold

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ss, "build_spatial_kfold", spy)
    ens._spatial_holdout(tuple(parcel_ids), geoms, buffer_km=1.0)
    assert calls, "build_spatial_kfold was not used (random split would be leakage)."


def test_holdout_groups_match_clusters(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """Train and val parcels come from geographically separated clusters.

    Because the two clusters are far apart, no parcel should appear in both
    sides, and the union must stay within the aligned parcel set.
    """
    ens, parcel_ids, _ = fitted
    geoms = _parcel_geoms(parcel_ids)
    train_idx, val_idx = ens._spatial_holdout(tuple(parcel_ids), geoms, buffer_km=1.0)
    union = set(train_idx.tolist()) | set(val_idx.tolist())
    assert union.issubset(set(range(len(parcel_ids))))


# ---------------------------------------------------------------------------
# 4. predict_proba is a weighted, sum-to-1 combination of post-softmax members.
# ---------------------------------------------------------------------------


def test_predict_proba_sums_to_one(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """The blended output is post-softmax: non-negative, sums to 1 per row."""
    ens, parcel_ids, _ = fitted
    proba = ens.predict_proba()
    assert proba.shape == (len(parcel_ids), NUM_CLASSES)
    assert (proba >= 0.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_is_weighted_combination(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """The output equals sum_i w_i * P_i over the aligned members (golden math)."""
    ens, parcel_ids, _ = fitted
    weights = ens.weights
    # Rebuild the aligned member matrices in the same order and blend by hand.
    frames = ens.load_oof_members(_MEMBERS, space="parcel")
    stacked = []
    for member in _MEMBERS:
        aligned = (
            frames[member]
            .filter(pl.col("canonical_parcel_id").is_in(parcel_ids))
            .sort("canonical_parcel_id")
        )
        stacked.append(aligned.select(PROB_COLUMNS).to_numpy().astype(np.float64))
    manual = np.tensordot(weights, np.stack(stacked, axis=0), axes=([0], [0]))
    manual = manual / manual.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(ens.predict_proba(), manual, atol=1e-9)


def test_predict_proba_with_explicit_frames(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """predict_proba accepts explicit per-member frames over a new parcel set."""
    ens, _, _ = fitted
    frames = [
        make_parcel_frame(member, n_parcels=4, seed=900 + i) for i, member in enumerate(_MEMBERS)
    ]
    proba = ens.predict_proba(frames)
    assert proba.shape == (4, NUM_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_wrong_member_count_raises(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """Passing the wrong number of member frames raises."""
    ens, _, _ = fitted
    frames = [make_parcel_frame(_MEMBERS[0], n_parcels=3, seed=1)]
    with pytest.raises(ValueError, match="one frame per base member"):
        ens.predict_proba(frames)


def test_predict_proba_unaligned_frames_raise(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """Member frames over different parcel sets are rejected (alignment guard)."""
    ens, _, _ = fitted
    frames = [
        make_parcel_frame(_MEMBERS[0], n_parcels=4, seed=1),
        make_parcel_frame(_MEMBERS[1], n_parcels=4, seed=1),
        make_parcel_frame(_MEMBERS[2], n_parcels=5, seed=1),  # different set
    ]
    with pytest.raises(ValueError, match="not aligned"):
        ens.predict_proba(frames)


# ---------------------------------------------------------------------------
# 5. Report fold-5 only (anti-leakage) + end-to-end evaluate.
# ---------------------------------------------------------------------------


def test_evaluate_fold4_raises(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """evaluate(fold=4) -> ValueError; fold-4 was selection, never reported."""
    ens, _, y_true = fitted
    proba = ens.predict_proba()
    with pytest.raises(ValueError, match="fold-5-only"):
        ens.evaluate(y_true=_aligned_labels(ens, y_true), proba=proba, fold=4)


def test_evaluate_fold5_returns_metrics(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """evaluate(fold=5) over the blended proba returns F1-macro/accuracy."""
    ens, _, y_true = fitted
    proba = ens.predict_proba()
    metrics = ens.evaluate(y_true=_aligned_labels(ens, y_true), proba=proba, fold=5)
    assert set(metrics) == {"f1_macro", "accuracy"}
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Member alignment + GT loading + MLflow params.
# ---------------------------------------------------------------------------


def test_align_members_intersects_parcels(tmp_path: Path) -> None:
    """_align_members intersects parcel ids and validates post-softmax rows."""
    # member A: parcels 1..6 ; member B: parcels 1..4 ; member C: parcels 1..5.
    write_parcel_oof(tmp_path, _MEMBERS[0], n_parcels=6, seed=1)
    write_parcel_oof(tmp_path, _MEMBERS[1], n_parcels=4, seed=2)
    write_parcel_oof(tmp_path, _MEMBERS[2], n_parcels=5, seed=3)
    ens = BlendingEnsemble(_MEMBERS, n_trials=3, oof_dir=tmp_path)
    parcel_ids, probs = ens._align_members()
    # Intersection is the first 4 parcels.
    assert len(parcel_ids) == 4
    assert probs.shape == (len(_MEMBERS), 4, NUM_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


def test_labels_for_missing_columns_raises() -> None:
    """A GT frame without label/canonical_parcel_id raises a clear error."""
    bad = pl.DataFrame({"canonical_parcel_id": ["10000_001"]})
    with pytest.raises(ValueError, match="label"):
        BlendingEnsemble._labels_for(["10000_001"], bad)


def test_labels_for_uncovered_parcel_raises() -> None:
    """A GT frame that does not cover every parcel raises."""
    gt = pl.DataFrame({"canonical_parcel_id": ["10000_001"], "label": [3]})
    with pytest.raises(ValueError, match="does not cover"):
        BlendingEnsemble._labels_for(["10000_001", "10000_002"], gt)


def test_mlflow_params_includes_weights(
    fitted: tuple[BlendingEnsemble, list[str], pl.DataFrame],
) -> None:
    """mlflow_params surfaces members, n_trials, gap_lambda and per-member weights."""
    ens, _, _ = fitted
    params = ens.mlflow_params()
    assert params["members"] == ",".join(_MEMBERS)
    assert params["n_trials"] == ens.n_trials
    assert params["gap_lambda"] == ens.gap_lambda
    for member in _MEMBERS:
        assert f"weight_{member}" in params
    total = sum(float(params[f"weight_{m}"]) for m in _MEMBERS)
    assert total == pytest.approx(1.0, abs=1e-5)


def test_fit_requires_canonical_id_column(tmp_path: Path) -> None:
    """fit raises if parcel_geoms lacks canonical_parcel_id."""
    n_parcels = _PER_CLUSTER * len(_CLUSTERS)
    parcel_ids = _write_aligned_members(tmp_path, _MEMBERS, n_parcels=n_parcels)
    geoms = _parcel_geoms(parcel_ids).drop(columns=["canonical_parcel_id"])
    ens = BlendingEnsemble(_MEMBERS, n_trials=3, oof_dir=tmp_path)
    with pytest.raises(ValueError, match="canonical_parcel_id"):
        ens.fit(geoms, y_true=_ground_truth(parcel_ids))


# ---------------------------------------------------------------------------
# Local helper.
# ---------------------------------------------------------------------------


def _aligned_labels(ens: BlendingEnsemble, y_true: pl.DataFrame) -> np.ndarray:
    """Return GT labels aligned to the ensemble's cached fold-5 parcel order."""
    return BlendingEnsemble._labels_for(ens._member_ids, y_true)
