"""Contract tests for :class:`ml.ensemble.base.EnsembleModel` (US-040, phase 1).

These freeze the invariants every rubric ensemble inherits, with the anti-leakage
guarantees (R-LEAK) first:

1. ``evaluate(fold=4)`` raises ``ValueError`` (fold-4 was selection); only
   fold-5 is reportable.
2. Probabilities must be POST-softmax: :meth:`validate_probs` rejects logits.
3. The meta-learner sees OOF only: :meth:`assert_oof_only` rejects overlapping
   train/eval parcel sets.

Plus the shared plumbing: OOF loading in both spaces (against tiny real parquet
fixtures), the pixel->parcel reduction reusing
:func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs`, the F1-macro/accuracy
helper and the MLflow logging path (file store, no server probe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.ensemble.base import (
    DEFAULT_OOF_DIR,
    ENSEMBLE_EXPERIMENT,
    EnsembleModel,
)
from ml.utils.parcel_reconcile import PROB_COLUMNS, pixel_to_parcel_probs
from tests.ml.ensemble.fixtures.synthetic_oof import (
    NUM_CLASSES,
    SMALL_SIZE,
    make_softmax_map,
    write_parcel_oof,
    write_pixel_oof,
)


class _DummyEnsemble(EnsembleModel):
    """Minimal concrete ensemble so the ABC can be instantiated in tests.

    ``fit`` is a no-op and ``predict_proba`` returns a stored post-softmax
    matrix, exercising the base contract without any real inference.
    """

    def __init__(self, *, proba: np.ndarray | None = None, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._proba = proba

    def fit(self, *args: object, **kwargs: object) -> _DummyEnsemble:
        return self

    def predict_proba(self, *args: object, **kwargs: object) -> np.ndarray:
        if self._proba is None:
            raise RuntimeError("no proba configured for the dummy ensemble.")
        return self._proba


@pytest.fixture
def dummy() -> _DummyEnsemble:
    """A dummy ensemble with the default oof_dir / random_state."""
    return _DummyEnsemble()


# ---------------------------------------------------------------------------
# Construction / defaults.
# ---------------------------------------------------------------------------


def test_defaults_match_contract(dummy: _DummyEnsemble) -> None:
    """HELD_OUT_FOLD is 5, oof_dir/random_state defaults match the plan."""
    assert dummy.HELD_OUT_FOLD == 5
    assert dummy.oof_dir == Path(DEFAULT_OOF_DIR)
    assert dummy.random_state == 42
    assert ENSEMBLE_EXPERIMENT == "ensemble"


def test_cannot_instantiate_abstract_base() -> None:
    """EnsembleModel is abstract: fit/predict_proba must be implemented."""
    with pytest.raises(TypeError):
        EnsembleModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Anti-leakage 1: evaluate is fold-5 only.
# ---------------------------------------------------------------------------


def test_evaluate_fold4_raises(dummy: _DummyEnsemble) -> None:
    """evaluate(fold=4) -> ValueError (fold-4 was selection, never reported)."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="fold-5-only"):
        dummy.evaluate(y_true=y_true, y_pred=y_pred, fold=4)


@pytest.mark.parametrize("bad_fold", [0, 1, 2, 3, 4, 6, 10])
def test_evaluate_rejects_every_non_held_out_fold(dummy: _DummyEnsemble, bad_fold: int) -> None:
    """Only fold-5 is valid; every other fold raises."""
    with pytest.raises(ValueError):
        dummy.evaluate(y_true=np.array([0, 1]), y_pred=np.array([0, 1]), fold=bad_fold)


def test_evaluate_fold5_ok_perfect(dummy: _DummyEnsemble) -> None:
    """evaluate(fold=5) with perfect predictions -> f1=accuracy=1."""
    y_true = np.array([0, 1, 2, 3, 0, 1])
    metrics = dummy.evaluate(y_true=y_true, y_pred=y_true.copy(), fold=5)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluate_default_fold_is_five(dummy: _DummyEnsemble) -> None:
    """The default fold is the held-out fold-5 (no explicit fold needed)."""
    y_true = np.array([0, 1, 2])
    metrics = dummy.evaluate(y_true=y_true, y_pred=y_true.copy())
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluate_from_proba_argmax(dummy: _DummyEnsemble) -> None:
    """evaluate derives hard labels from a post-softmax proba via argmax."""
    # 3 parcels, one-hot on classes 2, 0, 1 -> deterministic argmax.
    proba = np.zeros((3, NUM_CLASSES), dtype=np.float64)
    proba[0, 2] = 1.0
    proba[1, 0] = 1.0
    proba[2, 1] = 1.0
    y_true = np.array([2, 0, 1])
    metrics = dummy.evaluate(y_true=y_true, proba=proba, fold=5)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_evaluate_requires_pred_or_proba(dummy: _DummyEnsemble) -> None:
    """evaluate needs at least one of y_pred / proba."""
    with pytest.raises(ValueError, match=r"y_pred.*proba"):
        dummy.evaluate(y_true=np.array([0, 1]), fold=5)


def test_evaluate_accepts_label_dataframe(dummy: _DummyEnsemble) -> None:
    """y_true may be a parcel DataFrame carrying a `label` column."""
    y_true = pl.DataFrame({"canonical_parcel_id": ["a", "b", "c"], "label": [1, 2, 3]})
    metrics = dummy.evaluate(y_true=y_true, y_pred=np.array([1, 2, 3]), fold=5)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluate_label_dataframe_needs_label_column(dummy: _DummyEnsemble) -> None:
    """A DataFrame y_true without `label` raises a clear error."""
    bad = pl.DataFrame({"canonical_parcel_id": ["a", "b"]})
    with pytest.raises(ValueError, match="label"):
        dummy.evaluate(y_true=bad, y_pred=np.array([0, 1]), fold=5)


# ---------------------------------------------------------------------------
# Anti-leakage 2: probabilities, not logits.
# ---------------------------------------------------------------------------


def test_validate_probs_accepts_softmax() -> None:
    """A valid post-softmax matrix passes and is returned unchanged."""
    sm = make_softmax_map(size=SMALL_SIZE, seed=1)  # (18, 8, 8) sums to 1 axis 0
    out = EnsembleModel.validate_probs(sm, class_axis=0, name="pixel")
    assert out is sm


def test_validate_probs_rejects_logits() -> None:
    """Raw logits (negative + not summing to 1) are rejected."""
    rng = np.random.default_rng(0)
    logits = rng.uniform(-5.0, 5.0, size=(4, NUM_CLASSES))  # not normalized
    with pytest.raises(ValueError, match=r"logits|sum to 1|negative"):
        EnsembleModel.validate_probs(logits, class_axis=-1, name="logits")


def test_validate_probs_rejects_negative_values() -> None:
    """A row that sums to 1 but has a negative entry is still rejected."""
    bad = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    bad[0, 0] = -0.1
    bad[0, 1] += 0.1  # keeps the row sum at 1 but with a negative entry
    with pytest.raises(ValueError, match="negative"):
        EnsembleModel.validate_probs(bad, class_axis=-1)


def test_validate_probs_rejects_nan() -> None:
    """Non-finite probabilities are rejected."""
    bad = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite|NaN"):
        EnsembleModel.validate_probs(bad, class_axis=-1)


def test_parcel_probs_matrix_extracts_and_validates() -> None:
    """parcel_probs_matrix returns an (n, 18) post-softmax matrix."""
    df = write_parcel_oof_frame(seed=3)
    matrix = EnsembleModel.parcel_probs_matrix(df)
    assert matrix.shape == (df.height, NUM_CLASSES)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-4)


def test_parcel_probs_matrix_missing_columns() -> None:
    """A frame without prob_* columns raises."""
    df = pl.DataFrame({"canonical_parcel_id": ["a", "b"]})
    with pytest.raises(ValueError, match="prob columns"):
        EnsembleModel.parcel_probs_matrix(df)


def test_evaluate_proba_rejects_logits(dummy: _DummyEnsemble) -> None:
    """evaluate(proba=logits) is caught by the post-softmax guard."""
    rng = np.random.default_rng(7)
    logits = rng.uniform(-5.0, 5.0, size=(3, NUM_CLASSES))
    with pytest.raises(ValueError):
        dummy.evaluate(y_true=np.array([0, 1, 2]), proba=logits, fold=5)


# ---------------------------------------------------------------------------
# OOF loading (pixel + parcel) against tiny real parquet fixtures.
# ---------------------------------------------------------------------------


def test_load_oof_members_pixel(tmp_path: Path) -> None:
    """space='pixel' reconstructs the dense (18, H, W) softmax per patch."""
    write_pixel_oof(tmp_path, "tsvit-pheno", seed=0)
    write_pixel_oof(tmp_path, "utae", seed=10)
    ens = _DummyEnsemble(oof_dir=tmp_path)

    loaded = ens.load_oof_members(["tsvit-pheno", "utae"], space="pixel")
    assert list(loaded) == ["tsvit-pheno", "utae"]
    df = loaded["tsvit-pheno"]
    assert "softmax" in df.columns and "pred" in df.columns
    sm = df["softmax"].to_list()[0]
    assert sm.shape == (NUM_CLASSES, SMALL_SIZE, SMALL_SIZE)
    np.testing.assert_allclose(sm.astype(np.float64).sum(axis=0), 1.0, atol=2e-3)


def test_load_oof_members_parcel(tmp_path: Path) -> None:
    """space='parcel' loads the prob_000..017 per-parcel frame."""
    write_parcel_oof(tmp_path, "tsvit-pheno", n_parcels=5, seed=0)
    ens = _DummyEnsemble(oof_dir=tmp_path)

    loaded = ens.load_oof_members(["tsvit-pheno"], space="parcel")
    df = loaded["tsvit-pheno"]
    assert "canonical_parcel_id" in df.columns
    for col in PROB_COLUMNS:
        assert col in df.columns
    matrix = df.select(PROB_COLUMNS).to_numpy()
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-4)


def test_load_oof_members_missing_raises(tmp_path: Path) -> None:
    """A missing member parquet raises a FileNotFoundError mentioning dvc pull."""
    ens = _DummyEnsemble(oof_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        ens.load_oof_members(["does-not-exist"], space="parcel")


def test_oof_path_invalid_space(dummy: _DummyEnsemble) -> None:
    """An invalid space raises."""
    with pytest.raises(ValueError, match="space"):
        dummy.oof_path("utae", space="voxel")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pixel -> parcel reduction reuses pixel_to_parcel_probs.
# ---------------------------------------------------------------------------


def test_reduce_pixel_to_parcel_matches_helper(dummy: _DummyEnsemble) -> None:
    """reduce_pixel_to_parcel == pixel_to_parcel_probs (same reconciliation)."""
    probs = make_softmax_map(size=SMALL_SIZE, seed=2)
    parcel_ids = np.zeros((SMALL_SIZE, SMALL_SIZE), dtype=np.int64)
    half = SMALL_SIZE // 2
    parcel_ids[:, :half] = 101
    parcel_ids[:, half:] = 202

    got = dummy.reduce_pixel_to_parcel(probs, parcel_ids, patch_id="10000", method="mean")
    expected = pixel_to_parcel_probs(probs, parcel_ids, patch_id="10000", method="mean")

    assert got["canonical_parcel_id"].to_list() == expected["canonical_parcel_id"].to_list()
    np.testing.assert_allclose(
        got.select(PROB_COLUMNS).to_numpy(),
        expected.select(PROB_COLUMNS).to_numpy(),
        atol=1e-6,
    )


def test_reduce_pixel_to_parcel_rejects_logits(dummy: _DummyEnsemble) -> None:
    """Feeding logits to the reducer is caught before reconciliation."""
    rng = np.random.default_rng(5)
    logits = rng.uniform(-5.0, 5.0, size=(NUM_CLASSES, SMALL_SIZE, SMALL_SIZE))
    parcel_ids = np.ones((SMALL_SIZE, SMALL_SIZE), dtype=np.int64)
    with pytest.raises(ValueError):
        dummy.reduce_pixel_to_parcel(logits, parcel_ids, patch_id="1")


# ---------------------------------------------------------------------------
# Anti-leakage 3: meta-learner sees OOF only.
# ---------------------------------------------------------------------------


def test_assert_oof_only_rejects_overlap() -> None:
    """Overlapping meta train/eval parcel ids raise (leakage)."""
    with pytest.raises(ValueError, match="leakage"):
        EnsembleModel.assert_oof_only(["a", "b", "c"], ["c", "d"])


def test_assert_oof_only_disjoint_ok() -> None:
    """Disjoint meta train/eval parcel ids pass silently."""
    EnsembleModel.assert_oof_only(["a", "b"], ["c", "d"])


def test_assert_oof_only_handles_mixed_types() -> None:
    """Overlap detection is type-agnostic (int vs str ids)."""
    with pytest.raises(ValueError, match="leakage"):
        EnsembleModel.assert_oof_only([1, 2, 3], ["3", "4"])


# ---------------------------------------------------------------------------
# Metrics helper.
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect() -> None:
    """compute_metrics is 1.0 on perfect predictions."""
    y = np.array([0, 1, 2, 3, 4])
    metrics = EnsembleModel.compute_metrics(y, y.copy())
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_shape_mismatch() -> None:
    """Mismatched prediction/label sizes raise."""
    with pytest.raises(ValueError, match="same"):
        EnsembleModel.compute_metrics(np.array([0, 1, 2]), np.array([0, 1]))


def test_compute_metrics_ignore_index() -> None:
    """The ignore_index label is excluded from the metric."""
    y_true = np.array([0, 1, 255, 2])
    y_pred = np.array([0, 1, 7, 2])  # the 255 position is wrong but ignored
    metrics = EnsembleModel.compute_metrics(y_true, y_pred, ignore_index=255)
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# timed_predict helper.
# ---------------------------------------------------------------------------


def test_timed_predict_returns_output_and_time(dummy: _DummyEnsemble) -> None:
    """timed_predict returns the callable output and a non-negative duration."""
    ens = _DummyEnsemble(proba=np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES))
    out, elapsed = EnsembleModel.timed_predict(ens.predict_proba)
    assert out.shape == (2, NUM_CLASSES)
    assert elapsed >= 0.0


def test_timed_predict_requires_callable() -> None:
    """timed_predict rejects a non-callable."""
    with pytest.raises(TypeError):
        EnsembleModel.timed_predict(123)


# ---------------------------------------------------------------------------
# MLflow logging path (file store, no server probe).
# ---------------------------------------------------------------------------


def test_log_to_mlflow_file_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """log_to_mlflow opens a run with the mandatory tags + chosen_model.

    Uses an explicit file-store tracking URI so no Docker server is contacted; we
    then read the run back from the MLflow client to assert the tags/metrics.
    """
    import mlflow

    store = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    ens = _DummyEnsemble()
    ens.log_to_mlflow(
        {"f1_macro": 0.64, "accuracy": 0.80},
        run_name="e1-voting",
        params={"members": "tsvit-pheno,utae,unet", "n_bags": 0},
        chosen=True,
        inference_time_s=1.5,
        tracking_uri=store,
        probe_server=False,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=store)
    exp = client.get_experiment_by_name(ENSEMBLE_EXPERIMENT)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags.get("chosen_model") == "e1-voting"
    assert run.data.tags.get("ensemble") == "e1-voting"
    assert "code_version" in run.data.tags
    assert "data_version" in run.data.tags
    assert run.data.metrics["f1_macro_fold5"] == pytest.approx(0.64)
    assert run.data.metrics["accuracy_fold5"] == pytest.approx(0.80)
    assert run.data.metrics["inference_time_s"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Local helper.
# ---------------------------------------------------------------------------


def write_parcel_oof_frame(*, seed: int = 0) -> pl.DataFrame:
    """Return an in-memory parcel OOF frame (no disk) for matrix tests."""
    from tests.ml.ensemble.fixtures.synthetic_oof import make_parcel_frame

    return make_parcel_frame("utae", n_parcels=6, seed=seed)
