"""Golden-value tests for the softmax/OOF dump core (US-031, agente ml/A).

Exercises :mod:`ml.eval.oof.dump_oof`, :mod:`ml.eval.oof.parquet_io` and the
new :func:`ml.eval.segmentation_inference.softmax_patch_for_kind` /
``_forward_logits`` refactor with MOCKED models and dataset: no checkpoint is
ever loaded, ``torch.hub`` (AnySat encoder) is never contacted, and NO real
inference runs over PASTIS. The probability-space remap/resample helpers
(``test_softmax_remap.py``) and the parcel reconciler
(``tests/ml/utils/test_parcel_reconcile.py``) are covered by their own suites.

Coverage:

- ``softmax_patch_for_kind`` sums to 1, stays in [0, 1] and is POST-softmax.
- ``argmax(softmax) == predict_patch_for_kind`` (the ``_forward_logits`` refactor
  does not alter the US-030 prediction).
- ``dump_oof`` writes the per-pixel parquet schema; the softmax reconstructs to
  ``(18, 128, 128)`` and ``pred`` to ``(128, 128)``.
- ``held_out`` is True for fold 5 and False for fold 4.
- A missing checkpoint yields a ``status="missing"`` manifest entry, no crash.
- ``parquet_io`` roundtrip preserves shape and values (float16 tolerance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from torch import nn

import ml.eval.oof.dump_oof as dump_mod
import ml.eval.segmentation_inference as seg_inf
from ml.eval.checkpoint_registry import CheckpointSpec
from ml.eval.class_remap import HARNESS_NUM_CLASSES, HARNESS_SIZE
from ml.eval.oof.parquet_io import read_softmax_parquet, write_softmax_parquet
from tests.ml.eval.oof.fixtures.oof_synthetic import make_logits

_SUM_TOL_F32 = 1e-6
_SUM_TOL_F16 = 1e-3


# ===========================================================================
# softmax_patch_for_kind + _forward_logits parity (no checkpoints).
# ===========================================================================


class _FixedLogitsModel(nn.Module):
    """Model whose forward returns a fixed logits tensor regardless of input.

    Stands in for a real segmentation head: the harness forward dispatch in
    ``_forward_logits`` only reads ``next(model.parameters()).device`` and calls
    ``model(xb)``; returning a fixed ``(1, C, H, W)`` lets the softmax/argmax be
    asserted in closed form.
    """

    def __init__(self, logits: np.ndarray) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))
        self._logits = torch.from_numpy(logits.astype(np.float32))

    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        return self._logits


def test_softmax_patch_sums_to_one() -> None:
    """softmax_patch_for_kind yields a valid post-softmax map (sum 1, >= 0)."""
    logits = make_logits(num_classes=18, size=8, scale=6.0, seed=1)
    model = _FixedLogitsModel(logits)
    x = torch.zeros(10, 8, 8)

    probs = seg_inf.softmax_patch_for_kind(model, x, model_kind="deeplabv3plus")

    assert probs.shape == (18, 8, 8)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=_SUM_TOL_F32)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_softmax_patch_is_not_logits() -> None:
    """The output is a probability map, not the raw out-of-[0,1] logits."""
    logits = make_logits(num_classes=18, size=8, scale=9.0, seed=2)
    assert logits.min() < 0.0 and logits.max() > 1.0
    model = _FixedLogitsModel(logits)
    x = torch.zeros(10, 8, 8)

    probs = seg_inf.softmax_patch_for_kind(model, x, model_kind="unet")
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    # The raw logits are not a distribution; the softmax is.
    assert not np.allclose(probs, logits[0])


def test_argmax_softmax_equals_predict_patch_for_kind() -> None:
    """argmax(softmax) == predict_patch_for_kind (the _forward_logits refactor)."""
    logits = make_logits(num_classes=20, size=12, scale=5.0, seed=3)
    model = _FixedLogitsModel(logits)
    x = torch.zeros(10, 12, 12)

    probs = seg_inf.softmax_patch_for_kind(model, x, model_kind="unet")
    pred = seg_inf.predict_patch_for_kind(model, x, model_kind="unet")

    np.testing.assert_array_equal(probs.argmax(axis=0), pred)


def test_predict_patch_for_kind_unchanged_signature() -> None:
    """predict_patch_for_kind still returns an int64 native-space class map."""
    logits = make_logits(num_classes=20, size=8, seed=4)
    model = _FixedLogitsModel(logits)
    x = torch.zeros(10, 8, 8)
    pred = seg_inf.predict_patch_for_kind(model, x, model_kind="deeplabv3plus")
    assert pred.dtype == np.int64
    assert pred.shape == (8, 8)
    assert pred.min() >= 0 and pred.max() < 20


def test_forward_logits_rejects_segformer() -> None:
    """_forward_logits does not handle segformer (its own sub-pipeline)."""
    model = _FixedLogitsModel(make_logits(num_classes=20, size=4, seed=5))
    with pytest.raises(ValueError, match=r"segformer|sub-pipeline"):
        seg_inf._forward_logits(model, torch.zeros(10, 4, 4), model_kind="segformer")


# ===========================================================================
# parquet_io roundtrip.
# ===========================================================================


def test_softmax_parquet_roundtrip(tmp_path: Path) -> None:
    """Writing then reading a (18,128,128) softmax reconstructs shape and values."""
    rng = np.random.default_rng(7)
    raw = rng.uniform(size=(HARNESS_NUM_CLASSES, HARNESS_SIZE, HARNESS_SIZE))
    softmax = (raw / raw.sum(axis=0, keepdims=True)).astype(np.float32)
    pred = softmax.argmax(axis=0).astype(np.int8)
    rows: list[dict[str, Any]] = [
        {
            "patch_id": "10000",
            "fold": 5,
            "held_out": True,
            "model": "unet",
            "status": "ok",
            "softmax": softmax,
            "pred": pred,
            "code_version": "abc123",
            "data_version": "data/PASTIS-R@untracked",
        }
    ]
    path = tmp_path / "oof_unet_fold5.parquet"
    write_softmax_parquet(rows, path, num_classes=18, size=128, dtype="float16")

    df = read_softmax_parquet(path)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["patch_id"] == "10000"
    assert row["held_out"] is True
    rec = row["softmax"]
    assert isinstance(rec, np.ndarray)
    assert rec.shape == (18, 128, 128)
    # float16 storage -> relaxed tolerance.
    np.testing.assert_allclose(rec.astype(np.float32), softmax, atol=_SUM_TOL_F16)
    rec_pred = row["pred"]
    assert rec_pred.shape == (128, 128)
    np.testing.assert_array_equal(rec_pred, pred)


def test_softmax_parquet_missing_row_roundtrips_to_none(tmp_path: Path) -> None:
    """A row with no softmax/pred (missing checkpoint) reconstructs to None."""
    rows: list[dict[str, Any]] = [
        {
            "patch_id": "x",
            "fold": 5,
            "held_out": True,
            "model": "anysat",
            "status": "missing",
            "softmax": None,
            "pred": None,
            "code_version": "abc",
            "data_version": "d",
        }
    ]
    path = tmp_path / "oof_anysat_fold5.parquet"
    write_softmax_parquet(rows, path)
    df = read_softmax_parquet(path)
    assert df.row(0, named=True)["softmax"] is None
    assert df.row(0, named=True)["pred"] is None


def test_softmax_parquet_rejects_wrong_shape(tmp_path: Path) -> None:
    """A softmax whose shape disagrees with the declared (C,H,W) raises."""
    bad = np.zeros((18, 64, 64), dtype=np.float32)
    rows = [{"patch_id": "x", "softmax": bad, "pred": np.zeros((128, 128), np.int8)}]
    with pytest.raises(ValueError, match="shape"):
        write_softmax_parquet(rows, tmp_path / "bad.parquet", size=128)


# ===========================================================================
# dump_oof with mocked dataset/model (no checkpoints, no PASTIS, no torch.hub).
# ===========================================================================


class _FakeSegDataset:
    """Dataset fake registering ``folds`` and yielding deterministic (x, y).

    Reproduces the minimal contract ``dump_oof._dump_one`` consumes from
    :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`: ``folds``,
    ``__len__``, ``__getitem__``, ``patch_ids``, ``root`` and the normalization
    attributes mutated by ``_apply_train_norm``.
    """

    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.folds = tuple(kwargs.get("folds", ()))  # type: ignore[arg-type]
        self.root = Path(str(kwargs.get("root", "data/PASTIS-R")))
        self.patch_ids = ["10000", "10001", "10002"]
        self._norm_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {
            f: (np.full(10, float(f)), np.full(10, float(f))) for f in (1, 2, 3, 4, 5)
        }
        self._fold_of: dict[str, int] = dict.fromkeys(self.patch_ids, 5)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(10, HARNESS_SIZE, HARNESS_SIZE, dtype=torch.float32)
        y = torch.zeros(HARNESS_SIZE, HARNESS_SIZE, dtype=torch.int64)
        return x, y


class _DummyModel(nn.Module):
    """Model with a parameter so ``next(model.parameters())`` works."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))


def _install_dump_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    native_num_classes: int,
) -> type[_FakeSegDataset]:
    """Wire dataset/loader/softmax mocks into the dump import sites.

    ``_dump_one`` imports ``PASTISSegmentationDataset`` from
    ``ml.data.pastis_seg_dataset`` and ``load_checkpoint_model`` from
    ``ml.eval.segmentation_inference``; ``_softmax_for_patch`` imports
    ``softmax_patch_for_kind`` / ``softmax_logits_segformer`` from
    ``ml.eval.segmentation_inference``. They are patched at their source module.

    Args:
        monkeypatch: pytest fixture.
        native_num_classes: 20 -> the mocked softmax is in 20-space (remapped to
            18 by the dump); 18 -> already contiguous.

    Returns:
        The ``_FakeSegDataset`` class (to introspect ``last_kwargs``).
    """
    import ml.data.pastis_seg_dataset as ds_mod

    _FakeSegDataset.last_kwargs = {}
    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeSegDataset)
    monkeypatch.setattr(seg_inf, "load_checkpoint_model", lambda spec, **_kw: _DummyModel())

    def _fake_softmax(model: nn.Module, x: torch.Tensor, *, model_kind: str) -> np.ndarray:
        # Deterministic native-space softmax at 128 px.
        logits = make_logits(num_classes=native_num_classes, size=HARNESS_SIZE, seed=42)[0]
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(seg_inf, "softmax_patch_for_kind", _fake_softmax)
    # Skip the per-parcel sidecar by making ParcelIDs always absent (the dump
    # degrades gracefully); the parcel reconciler has its own test suite.
    monkeypatch.setattr(
        dump_mod,
        "load_pastis_parcel_ids",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("no parcels")),
    )
    return _FakeSegDataset


def _spec(model_kind: str, native_num_classes: int) -> CheckpointSpec:
    """CheckpointSpec pointing at this test file (so ``path.exists()`` is True)."""
    return CheckpointSpec(
        name=model_kind,
        model_kind=model_kind,  # type: ignore[arg-type]
        path=Path(__file__).resolve(),
        native_num_classes=native_num_classes,
        native_ignore_index=19 if native_num_classes >= 20 else 255,
    )


def test_dump_writes_pixel_parquet_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dump_oof writes a per-pixel parquet with the contract columns + shapes."""
    _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)

    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": spec},
        fold=5,
        out_dir=tmp_path,
        device="cpu",
        max_patches=2,
        write_parcel=True,
    )

    entry = manifest["models"]["deeplabv3plus"]
    assert entry["status"] == "ok"
    assert entry["n_patches"] == 2
    assert entry["held_out"] is True
    assert entry["shape"] == [18, 128, 128]

    parquet_path = Path(entry["path"])
    assert parquet_path.exists()
    df = read_softmax_parquet(parquet_path)
    assert df.height == 2
    for col in (
        "patch_id",
        "fold",
        "held_out",
        "model",
        "status",
        "softmax",
        "pred",
        "code_version",
        "data_version",
    ):
        assert col in df.columns, col
    row = df.row(0, named=True)
    assert row["softmax"].shape == (18, 128, 128)
    assert row["pred"].shape == (128, 128)
    # Persisted softmax is POST-softmax (sum 1 over the class axis).
    np.testing.assert_allclose(
        row["softmax"].astype(np.float32).sum(axis=0), 1.0, atol=_SUM_TOL_F16
    )


def test_dump_pred_equals_softmax_argmax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted pred equals the argmax of the persisted softmax.

    ``pred`` is the argmax of the float32 softmax computed before storage; the
    softmax is stored as float16. On the handful of pixels where the top-2
    probabilities collapse to the same float16 value (a tie), the float16 argmax
    can differ. The parity is asserted exactly EXCEPT on those documented ties,
    where the persisted ``pred`` is still a valid co-maximizer of the float16 map.
    """
    _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("tsvit-pheno", 18)
    manifest = dump_mod.dump_oof(
        {"tsvit-pheno": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=1
    )
    df = read_softmax_parquet(Path(manifest["models"]["tsvit-pheno"]["path"]))
    row = df.row(0, named=True)
    softmax16 = row["softmax"].astype(np.float32)  # float16 values widened
    pred = row["pred"]

    mismatch = pred != softmax16.argmax(axis=0).astype(np.int8)
    if mismatch.any():
        # Every mismatch must be a float16 tie: the persisted pred's probability
        # equals the per-pixel max within float16 resolution.
        h_idx, w_idx = np.where(mismatch)
        pred_prob = softmax16[pred[mismatch], h_idx, w_idx]
        max_prob = softmax16.max(axis=0)[mismatch]
        np.testing.assert_allclose(pred_prob, max_prob, atol=_SUM_TOL_F16)


def test_dump_20class_model_remaps_to_18(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 20-class model is unified to (18,128,128) in probability space."""
    _install_dump_mocks(monkeypatch, native_num_classes=20)
    spec = _spec("unet", 20)
    manifest = dump_mod.dump_oof(
        {"unet": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=1
    )
    df = read_softmax_parquet(Path(manifest["models"]["unet"]["path"]))
    softmax = df.row(0, named=True)["softmax"]
    assert softmax.shape == (18, 128, 128)
    np.testing.assert_allclose(softmax.astype(np.float32).sum(axis=0), 1.0, atol=_SUM_TOL_F16)


def test_dump_held_out_flag_fold5_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dump_oof(fold=5) marks every row held_out=True."""
    _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=2
    )
    assert manifest["held_out"] is True
    df = read_softmax_parquet(Path(manifest["models"]["deeplabv3plus"]["path"]))
    assert df["held_out"].to_list() == [True, True]


def test_dump_held_out_flag_fold4_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dump_oof(fold=4) marks every row held_out=False (selection fold, leak)."""
    fake_ds = _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": spec}, fold=4, out_dir=tmp_path, device="cpu", max_patches=1
    )
    assert fake_ds.last_kwargs.get("folds") == (4,)
    assert manifest["held_out"] is False
    df = read_softmax_parquet(Path(manifest["models"]["deeplabv3plus"]["path"]))
    assert df["held_out"].to_list() == [False]


def test_dump_uses_fold_dataset_and_semantic18(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset is instantiated with folds=(fold,) and target=semantic18."""
    fake_ds = _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    dump_mod.dump_oof(
        {"deeplabv3plus": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=1
    )
    assert fake_ds.last_kwargs.get("folds") == (5,)
    assert fake_ds.last_kwargs.get("target") == "semantic18"
    assert fake_ds.last_kwargs.get("ignore_index") == 255


def test_dump_norm_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalization stats are overwritten with the train-only (1,2,3) average."""
    captured: dict[str, _FakeSegDataset] = {}
    _install_dump_mocks(monkeypatch, native_num_classes=20)

    import ml.eval.dense_metrics as dense_metrics

    real_apply = dense_metrics._apply_train_norm

    def _spy(dataset: object) -> None:
        captured["ds"] = dataset  # type: ignore[assignment]
        real_apply(dataset)

    monkeypatch.setattr(dense_metrics, "_apply_train_norm", _spy)

    spec = _spec("unet", 20)
    dump_mod.dump_oof({"unet": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=1)
    ds = captured["ds"]
    for mean, std in ds._norm_stats.values():  # type: ignore[attr-defined]
        # Mean of train folds 1,2,3 == 2.0; the held-out fold-5 (5.0) is gone.
        np.testing.assert_allclose(mean, np.full(10, 2.0))
        np.testing.assert_allclose(std, np.full(10, 2.0))
        assert not np.allclose(mean, np.full(10, 5.0))


def test_dump_skip_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing checkpoint yields status='missing', no crash, no parquet."""
    # Do NOT install dataset/loader mocks: the spec path does not exist.
    spec = CheckpointSpec(
        name="ghost",
        model_kind="unet",
        path=tmp_path / "does_not_exist.pt",
        native_num_classes=20,
        native_ignore_index=19,
    )
    manifest = dump_mod.dump_oof(
        {"ghost": spec}, fold=5, out_dir=tmp_path, device="cpu", skip_missing=True
    )
    entry = manifest["models"]["ghost"]
    assert entry["status"] == "missing"
    assert entry["n_patches"] == 0
    assert entry["path"] is None
    assert entry["held_out"] is True
    assert not (tmp_path / "oof_ghost_fold5.parquet").exists()


def test_dump_missing_raises_when_not_skipping(
    tmp_path: Path,
) -> None:
    """skip_missing=False re-raises on a missing checkpoint."""
    spec = CheckpointSpec(
        name="ghost",
        model_kind="unet",
        path=tmp_path / "nope.pt",
        native_num_classes=20,
        native_ignore_index=19,
    )
    with pytest.raises(FileNotFoundError):
        dump_mod.dump_oof(
            {"ghost": spec}, fold=5, out_dir=tmp_path, device="cpu", skip_missing=False
        )


def test_dump_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """manifest.json is written with one entry per model + provenance tags."""
    import json

    _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    dump_mod.dump_oof(
        {"deeplabv3plus": spec}, fold=5, out_dir=tmp_path, device="cpu", max_patches=1
    )
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["fold"] == 5
    assert loaded["num_classes"] == 18
    assert loaded["size"] == 128
    assert "code_version" in loaded
    assert "data_version" in loaded
    entry = loaded["models"]["deeplabv3plus"]
    assert entry["status"] == "ok"
    assert entry["dtype"] == "float16"


def test_dump_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI parses args and runs dump_oof on the (mocked) default registry."""
    _install_dump_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    monkeypatch.setattr("ml.eval.checkpoint_registry.CHECKPOINT_REGISTRY", {"deeplabv3plus": spec})
    rc = dump_mod.main(
        [
            "--fold",
            "5",
            "--out-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--max-patches",
            "1",
            "--no-parcel",
        ]
    )
    assert rc == 0
    assert (tmp_path / "manifest.json").exists()
