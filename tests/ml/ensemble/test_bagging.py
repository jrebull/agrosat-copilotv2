"""Tests for :class:`ml.ensemble.bagging.BaggingEnsemble` (US-040, E2).

Covers the rubric criteria for the tabular bagging ensemble (plan Section 6):

- ``n_bags`` bootstraps drawn with DISTINCT seeds -> diverse resamples.
- ``predict_proba`` is the MEAN of the per-bag probabilities and sums to 1.
- Optuna runs (small ``n_trials``) and persists ``best_params`` / the study.
- The CV inside the Optuna objective is SPATIAL (``build_spatial_kfold``), not
  a random/IID split.
- Anti-leakage: fold-5 (held-out) never enters the bootstraps.

The real XGBoost-AlphaEarth fit and the O(N^2) spatial CV are heavy, so the
suite uses a tiny deterministic tabular dataset and monkeypatches the spatial CV
in the bagging namespace for the Optuna objective. One dedicated test exercises
the REAL ``evaluate_with_spatial_cv`` wiring to prove the spatial path is used.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.ensemble.bagging import ALPHAEARTH_PREFIX, BaggingEnsemble

NUM_CLASSES = 18
N_AE_DIMS = 64


# ---------------------------------------------------------------------------
# Deterministic synthetic tabular dataset (AlphaEarth 64-dim per parcel).
# ---------------------------------------------------------------------------


def make_tabular(
    *,
    n_per_fold: int = 24,
    n_classes: int = 6,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a tiny separable tabular dataset shaped like features_fused_pastis.

    Each row is a parcel with ``parcel_id``, ``class_id`` (1-based), ``fold``
    (1-5), ``patch_id`` and the AlphaEarth 64-dim vector ``dim_000..dim_063``.
    The features are class-conditioned (a per-class mean shift) so XGBoost can
    actually learn, keeping the fit deterministic and fast.

    Args:
        n_per_fold: Number of parcels per fold (5 folds total).
        n_classes: Number of agronomic classes (1..n_classes).
        seed: Deterministic seed.

    Returns:
        A Polars DataFrame with the columns above.
    """
    rng = np.random.default_rng(seed)
    n_folds = 5
    n_rows = n_per_fold * n_folds
    class_ids = (np.arange(n_rows) % n_classes) + 1  # 1-based, cycles classes
    folds = np.repeat(np.arange(1, n_folds + 1), n_per_fold)

    # Class-conditioned features: a deterministic mean per class + small noise.
    centers = rng.normal(0.0, 3.0, size=(n_classes, N_AE_DIMS))
    feats = centers[class_ids - 1] + rng.normal(0.0, 0.3, size=(n_rows, N_AE_DIMS))

    data: dict[str, object] = {
        "parcel_id": [f"10000_{i:04d}" for i in range(n_rows)],
        "patch_id": (np.arange(n_rows) % 10 + 10000).astype(np.int64),
        "class_id": class_ids.astype(np.int64),
        "fold": folds.astype(np.int64),
    }
    for d in range(N_AE_DIMS):
        data[f"{ALPHAEARTH_PREFIX}{d:03d}"] = feats[:, d].astype(np.float64)
    return pl.DataFrame(data)


def fake_spatial_cv_factory():
    """Return a stand-in for ``evaluate_with_spatial_cv`` that records calls.

    The stub fits the candidate estimator once on the full pool and reports a
    deterministic F1 proxy, so the Optuna objective runs in milliseconds without
    the O(N^2) spatial CV. It records the ``k_folds`` it was asked for so a test
    can assert the spatial parameters were forwarded.
    """
    calls: list[dict[str, object]] = []

    def _fake(df, factory, *, k_folds, buffer_km, random_state):  # type: ignore[no-untyped-def]
        calls.append({"k_folds": k_folds, "buffer_km": buffer_km, "n_rows": df.height})
        # A cheap, monotone-ish score so Optuna has something to optimize.
        depth = 0
        est = factory()
        depth = int(getattr(est, "max_depth", 6) or 6)
        score = 0.4 + 0.01 * depth
        cv_metrics = {"f1_macro": (float(min(score, 0.95)), 0.01)}
        empty = np.array([], dtype=np.int64)
        return cv_metrics, empty, empty

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


# ---------------------------------------------------------------------------
# Construction / validation.
# ---------------------------------------------------------------------------


def test_init_rejects_too_few_bags() -> None:
    """n_bags < 2 is rejected (bagging must aggregate)."""
    with pytest.raises(ValueError, match="n_bags"):
        BaggingEnsemble(n_bags=1)


def test_init_rejects_zero_trials() -> None:
    """n_trials < 1 is rejected (Optuna must run at least one trial)."""
    with pytest.raises(ValueError, match="n_trials"):
        BaggingEnsemble(n_bags=3, n_trials=0)


def test_inherits_held_out_fold() -> None:
    """The ensemble inherits the fold-5-only contract from the base."""
    ens = BaggingEnsemble(n_bags=3, n_trials=2)
    assert ens.HELD_OUT_FOLD == 5


# ---------------------------------------------------------------------------
# Bootstrap diversity (distinct seeds -> distinct resamples).
# ---------------------------------------------------------------------------


def test_bootstraps_distinct() -> None:
    """n_bags bootstraps use distinct seeds and produce distinct index arrays."""
    ens = BaggingEnsemble(n_bags=5, n_trials=1, random_state=42)
    indices = ens.bootstrap_indices(n_rows=200)

    assert len(indices) == 5
    # Distinct seeds.
    assert len(set(ens._bag_seeds)) == 5
    assert ens._bag_seeds == (42, 43, 44, 45, 46)
    # No two bags share an identical resample.
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            assert not np.array_equal(indices[i], indices[j]), (i, j)


def test_bootstrap_is_with_replacement() -> None:
    """A bootstrap of size N drawn from N rows repeats some indices."""
    ens = BaggingEnsemble(n_bags=2, n_trials=1, random_state=7)
    idx = ens.bootstrap_indices(n_rows=100)[0]
    assert idx.shape == (100,)
    # With replacement -> fewer unique than N (probability ~1 for N=100).
    assert np.unique(idx).size < 100


def test_bootstrap_indices_reproducible() -> None:
    """Same random_state -> identical bootstraps (golden / deterministic)."""
    a = BaggingEnsemble(n_bags=3, n_trials=1, random_state=11).bootstrap_indices(50)
    b = BaggingEnsemble(n_bags=3, n_trials=1, random_state=11).bootstrap_indices(50)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_bootstrap_rejects_empty() -> None:
    """A zero-row pool cannot be bootstrapped."""
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    with pytest.raises(ValueError, match="n_rows"):
        ens.bootstrap_indices(n_rows=0)


# ---------------------------------------------------------------------------
# Anti-leakage: fold-5 never enters the training pool.
# ---------------------------------------------------------------------------


def test_training_pool_excludes_fold5() -> None:
    """The bootstrap source drops every fold-5 (held-out) parcel."""
    df = make_tabular(n_per_fold=10, n_classes=4)
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    pool = ens._training_pool(df)
    assert pool.height == 40  # 4 folds x 10
    assert 5 not in pool.get_column("fold").unique().to_list()
    assert set(pool.get_column("fold").unique().to_list()) == {1, 2, 3, 4}


def test_training_pool_empty_without_train_folds() -> None:
    """A dataset that is only fold-5 yields an empty pool -> error."""
    df = make_tabular(n_per_fold=8, n_classes=3).filter(pl.col("fold") == 5)
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    with pytest.raises(ValueError, match="empty"):
        ens._training_pool(df)


def test_fit_never_sees_fold5(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: fit builds bags only from folds 1-4 (fold-5 excluded)."""
    df = make_tabular(n_per_fold=20, n_classes=5)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=3, n_trials=2, random_state=0)
    ens.fit(df)

    # Each Optuna trial's CV was handed only the 80 train-pool rows (4 folds x 20).
    assert all(c["n_rows"] == 80 for c in fake.calls)  # type: ignore[attr-defined]
    assert len(ens.bags) == 3


# ---------------------------------------------------------------------------
# Optuna runs and persists best params.
# ---------------------------------------------------------------------------


def test_optuna_study_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """fit runs an Optuna study with n_trials and populates best_params."""
    df = make_tabular(n_per_fold=16, n_classes=4)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=2, n_trials=4, random_state=0)
    ens.fit(df)

    assert ens.study is not None
    assert len(ens.study.trials) == 4
    assert ens.best_params  # non-empty, persisted
    # The tuned search-space keys are present in the persisted params.
    for key in ("max_depth", "learning_rate", "n_estimators", "subsample"):
        assert key in ens.best_params
    # Static params are kept.
    assert ens.best_params["objective"] == "multi:softprob"


def test_optuna_best_params_feed_every_bag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tuned hyperparameters are the ones used to fit each bag."""
    df = make_tabular(n_per_fold=16, n_classes=4)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=3, n_trials=3, random_state=0)
    ens.fit(df)

    tuned_depth = int(ens.best_params["max_depth"])  # type: ignore[arg-type]
    for bag in ens.bags:
        assert int(bag.max_depth) == tuned_depth


# ---------------------------------------------------------------------------
# Spatial CV (not random) wiring.
# ---------------------------------------------------------------------------


def test_optuna_uses_spatial_cv_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """The objective forwards the spatial CV parameters (k_folds, buffer_km)."""
    df = make_tabular(n_per_fold=16, n_classes=4)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=2, n_trials=2, n_spatial_folds=4, buffer_km=2.0, random_state=0)
    ens.fit(df)

    assert fake.calls  # type: ignore[attr-defined]
    for call in fake.calls:  # type: ignore[attr-defined]
        assert call["k_folds"] == 4
        assert call["buffer_km"] == 2.0


def test_tune_calls_real_spatial_kfold(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real CV path routes through build_spatial_kfold, never a random split.

    Patches ``build_spatial_kfold`` (the spatial primitive) and asserts the
    Optuna objective reaches it -- proving the CV is SPATIAL, not IID. The patch
    returns one fold so the underlying ``evaluate_with_spatial_cv`` stays cheap.
    """
    from ml.features.spatial_split import FoldAssignment

    df = make_tabular(n_per_fold=12, n_classes=3)
    ens = BaggingEnsemble(n_bags=2, n_trials=1, n_spatial_folds=2, random_state=0)
    pool = ens._training_pool(df)
    ens.feature_cols = ens._alphaearth_columns(pool)

    calls: list[int] = []

    def fake_kfold(parcels, *, k, buffer_km, random_state, **kw):  # type: ignore[no-untyped-def]
        calls.append(k)
        ids = [int(x) for x in parcels["parcel_id"].to_numpy()]
        half = len(ids) // 2
        return [
            FoldAssignment(0, tuple(ids[half:]), (), tuple(ids[:half])),
            FoldAssignment(1, tuple(ids[:half]), (), tuple(ids[half:])),
        ]

    # Patch where baseline._build_cv_splits looks it up.
    monkeypatch.setattr("ml.train.baseline.build_spatial_kfold", fake_kfold)
    # Avoid the disk cache short-circuiting the spatial build.
    monkeypatch.setattr("ml.train.baseline._load_cached_cv_splits", lambda path: None)
    monkeypatch.setattr("ml.train.baseline._save_cached_cv_splits", lambda path, splits: None)

    ens.tune(pool)
    assert calls, "build_spatial_kfold was never reached: CV is not spatial."
    assert all(k == 2 for k in calls)


# ---------------------------------------------------------------------------
# predict_proba: mean over bags, sum-to-1.
# ---------------------------------------------------------------------------


def test_predict_proba_is_mean_and_sums_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_proba == mean of per-bag predict_proba and sums to 1 per row."""
    df = make_tabular(n_per_fold=24, n_classes=5)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=4, n_trials=2, random_state=0)
    ens.fit(df)

    fold5 = df.filter(pl.col("fold") == 5)
    proba = ens.predict_proba(fold5)

    assert proba.shape == (fold5.height, NUM_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all()

    # Independently reconstruct the mean over bags and compare.
    matrix = ens._feature_matrix(fold5, ens.feature_cols)
    acc = np.zeros((matrix.shape[0], NUM_CLASSES), dtype=np.float64)
    for est in ens.bags:
        acc += ens._align_to_full_classes(
            np.asarray(est.predict_proba(matrix), dtype=np.float64), est
        )
    expected = acc / len(ens.bags)
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(proba, expected, atol=1e-9)


def test_predict_proba_learns_separable_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a separable dataset the mean proba recovers most fold-5 labels."""
    df = make_tabular(n_per_fold=40, n_classes=4, seed=3)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=5, n_trials=2, random_state=0)
    ens.fit(df)

    fold5 = df.filter(pl.col("fold") == 5)
    proba = ens.predict_proba(fold5)
    # Encoded predictions: argmax over the 18-class space; class_id is 1-based so
    # encoded label e corresponds to class_id e+1.
    pred_class_id = proba.argmax(axis=1) + 1
    accuracy = float((pred_class_id == fold5.get_column("class_id").to_numpy()).mean())
    assert accuracy >= 0.8


def test_predict_proba_before_fit_raises() -> None:
    """predict_proba before fit is a clear runtime error."""
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    df = make_tabular(n_per_fold=4, n_classes=2).filter(pl.col("fold") == 5)
    with pytest.raises(RuntimeError, match="before fit"):
        ens.predict_proba(df)


# ---------------------------------------------------------------------------
# Feature selection and evaluation integration.
# ---------------------------------------------------------------------------


def test_fit_requires_class_id() -> None:
    """fit without class_id raises immediately."""
    df = make_tabular(n_per_fold=8, n_classes=3).drop("class_id")
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    with pytest.raises(ValueError, match="class_id"):
        ens.fit(df)


def test_alphaearth_columns_required() -> None:
    """A pool without dim_* columns raises a descriptive error."""
    df = pl.DataFrame({"parcel_id": ["a", "b"], "class_id": [1, 2], "fold": [1, 2]})
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    with pytest.raises(ValueError, match="AlphaEarth"):
        ens._alphaearth_columns(df)


def test_alphaearth_columns_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """fit selects exactly the 64 AlphaEarth dimensions as features."""
    df = make_tabular(n_per_fold=12, n_classes=3)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=2, n_trials=2, random_state=0)
    ens.fit(df)
    assert len(ens.feature_cols) == N_AE_DIMS
    assert all(c.startswith(ALPHAEARTH_PREFIX) for c in ens.feature_cols)


def test_feature_matrix_imputes_non_finite() -> None:
    """NaN/inf in the AlphaEarth features are imputed with the column median."""
    df = make_tabular(n_per_fold=6, n_classes=3)
    # Inject a NaN and a +inf into dim_000.
    col = f"{ALPHAEARTH_PREFIX}000"
    vals = df.get_column(col).to_list()
    vals[0] = float("nan")
    vals[1] = float("inf")
    df = df.with_columns(pl.Series(col, vals))
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    matrix = ens._feature_matrix(df, (col,))
    assert np.isfinite(matrix).all()


def test_training_pool_without_fold_column() -> None:
    """A df pre-filtered by the caller (no `fold`) is used as-is with a warning."""
    df = make_tabular(n_per_fold=6, n_classes=3).drop("fold")
    ens = BaggingEnsemble(n_bags=2, n_trials=1)
    pool = ens._training_pool(df)
    assert pool.height == df.height


def test_bag_global_classes_uses_classes_fallback() -> None:
    """An estimator without _local_encoder maps columns via classes_."""

    class _Dummy:
        classes_ = np.array([2, 5, 9], dtype=np.int64)

    ids = BaggingEnsemble._bag_global_classes(_Dummy(), n_cols=3)
    np.testing.assert_array_equal(ids, np.array([2, 5, 9]))


def test_bag_global_classes_identity_fallback() -> None:
    """An estimator without _local_encoder nor classes_ falls back to identity."""

    class _Bare:
        pass

    ids = BaggingEnsemble._bag_global_classes(_Bare(), n_cols=4)
    np.testing.assert_array_equal(ids, np.arange(4))


def test_evaluate_on_fold5_after_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bagging proba feeds the base fold-5-only evaluate."""
    df = make_tabular(n_per_fold=30, n_classes=4, seed=1)
    fake = fake_spatial_cv_factory()
    monkeypatch.setattr("ml.ensemble.bagging.evaluate_with_spatial_cv", fake)

    ens = BaggingEnsemble(n_bags=4, n_trials=2, random_state=0)
    ens.fit(df)

    fold5 = df.filter(pl.col("fold") == 5)
    proba = ens.predict_proba(fold5)
    # Base evaluate must reject fold-4 (anti-leakage) and accept fold-5.
    with pytest.raises(ValueError, match="fold-5-only"):
        ens.evaluate(y_true=fold5.get_column("class_id").to_numpy() - 1, proba=proba, fold=4)
    metrics = ens.evaluate(y_true=fold5.get_column("class_id").to_numpy() - 1, proba=proba, fold=5)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
