"""Deterministic CPU tests for the E-b FarSLIP fine-tuned 18-class head (US-042).

Covers ``ml/ensemble/farslip_ft18.py``: the leak-free OOF materializer
:func:`materialize_ft18_oof`. The FarSLIP extractor and the parcel dataset are
STUBBED in memory (no PASTIS-R, no network, no GPU); the instance->ParcelIDs
bridge is monkeypatched so no PASTIS rasters are read. The stub extractor returns
deterministic, class-separable 512-dim embeddings so the ``LogisticRegression``
head learns a clean decision boundary, and the stub dataset is multi-fold so the
test asserts the head trains on folds 1-4 and predicts fold-5 ONLY.

Project conventions: numpy/torch only at the boundary, no emojis, Spanish prose,
English identifiers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from ml.ensemble import farslip_ft18
from ml.ensemble.farslip_ft18 import _EMBED_DIM, materialize_ft18_oof
from ml.utils.parcel_reconcile import PROB_COLUMNS

_N_CLASSES = 18


class _StubExtractor:
    """FarSLIP extractor stub: class-separable 512-dim embeddings.

    Each crop encodes its RAW PASTIS ``class_id`` in channel 0 (constant value =
    ``class_id``); the embedding places a 1.0 in the column ``class_id`` (plus a
    tiny deterministic ripple so the matrix is full-rank), then L2-normalizes.
    Distinct classes therefore live on (almost) orthogonal axes -> a logistic
    head separates them perfectly, with NO randomness.
    """

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        b = crops.shape[0]
        # Channel-0 mean over each crop recovers the injected class_id.
        class_ids = crops[:, 0].reshape(b, -1).mean(dim=1).round().long()
        emb = torch.zeros((b, _EMBED_DIM), dtype=torch.float32)
        idx = torch.arange(_EMBED_DIM, dtype=torch.float32)
        for i, cid in enumerate(class_ids.tolist()):
            col = int(cid) % _EMBED_DIM
            emb[i, col] = 1.0
            emb[i] += 1e-4 * torch.sin(idx * (cid + 1))
        return torch.nn.functional.normalize(emb, p=2, dim=-1)


class _StubParcelDataset:
    """In-memory ParcelCropDataset stub for ONE fold.

    Exposes the minimal ``collate_parcel_batch`` contract: ``__len__``,
    ``__getitem__`` -> ``{"image", "parcel_id", "patch_id", "class_id", ...}``.
    The crop's channel 0 encodes the RAW PASTIS ``class_id`` so the stub extractor
    is class-separable. ``parcel_id`` is the dataset key ``"{patch}_{instance}"``.
    """

    def __init__(self, samples: list[tuple[str, int, int]]) -> None:
        # samples: list of (patch_id, instance_id, class_id RAW 1..18).
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        patch, inst, cid = self._samples[idx]
        img = torch.zeros((4, 224, 224), dtype=torch.float32)
        img[0] = float(cid)  # channel 0 carries the class id (extractor reads it).
        return {
            "image": img,
            "parcel_id": f"{patch}_{inst}",
            "patch_id": patch,
            "class_id": cid,
            "caption": "",
            "bbox": (0, 0, 4, 4),
        }


def _make_train_dataset(per_class: int = 6) -> _StubParcelDataset:
    """Folds-1-4 stub: ``per_class`` parcels for each agronomic class 1..18."""
    samples: list[tuple[str, int, int]] = []
    inst = 1
    for cid in range(1, _N_CLASSES + 1):
        for _ in range(per_class):
            samples.append((f"1{cid:04d}", inst, cid))
            inst += 1
    return _StubParcelDataset(samples)


def _make_test_dataset() -> _StubParcelDataset:
    """Fold-5 stub: a handful of parcels spanning several classes."""
    samples = [
        ("90001", 1, 1),
        ("90001", 2, 3),
        ("90002", 1, 8),
        ("90002", 2, 18),
        ("90003", 1, 2),
    ]
    return _StubParcelDataset(samples)


@pytest.fixture
def _patch_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the instance->ParcelIDs bridge to an identity-ish map.

    Maps every instance id to a distinct raster id (``instance * 1000 + 7``) so
    the canonical translation runs without reading PASTIS-R rasters, while still
    proving the dataset key (``{patch}_{instance}``) is rewritten to the canonical
    (``{patch}_{raster_id}``) space.
    """

    def _fake_map(patch_id: str, pastis_root: object) -> dict[int, int]:
        # Cover the fold-5 instance ids used by _make_test_dataset.
        return {inst: inst * 1000 + 7 for inst in range(1, 50)}

    monkeypatch.setattr("ml.utils.parcel_reconcile.instance_to_parcel_id_map", _fake_map)


# ---------------------------------------------------------------------------
# Happy path: shape, schema, sum-to-one, canonical keys.
# ---------------------------------------------------------------------------


def test_materialize_writes_canonical_schema(tmp_path, _patch_bridge) -> None:
    """The OOF parquet has the canonical schema identical to the dense members."""
    out = tmp_path / "oof_parcel_farslip-ft18_fold5.parquet"
    path = materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,  # unused: bridge + datasets are stubbed.
        extractor=_StubExtractor(),
        train_dataset=_make_train_dataset(),
        test_dataset=_make_test_dataset(),
        batch_size=8,
    )
    assert path == out
    df = pl.read_parquet(out)
    expected_cols = ["canonical_parcel_id", *PROB_COLUMNS, "pred_class", "n_pixels"]
    assert df.columns == expected_cols
    assert df.schema["canonical_parcel_id"] == pl.Utf8
    for col in PROB_COLUMNS:
        assert df.schema[col] == pl.Float32
    assert df.schema["pred_class"] == pl.Int64
    assert df.schema["n_pixels"] == pl.Int64


def test_materialize_probs_shape_and_sum_to_one(tmp_path, _patch_bridge) -> None:
    """The fold-5 probabilities are (n, 18) and each row sums to 1 (post-softmax)."""
    out = tmp_path / "oof.parquet"
    materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=_StubExtractor(),
        train_dataset=_make_train_dataset(),
        test_dataset=_make_test_dataset(),
        batch_size=8,
    )
    df = pl.read_parquet(out)
    assert df.height == 5  # exactly the fold-5 parcels.
    probs = df.select(PROB_COLUMNS).to_numpy()
    assert probs.shape == (5, _N_CLASSES)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_keys_are_canonical_translated(tmp_path, _patch_bridge) -> None:
    """The dataset's {patch}_{instance} keys are rewritten to the ParcelIDs space."""
    out = tmp_path / "oof.parquet"
    materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=_StubExtractor(),
        train_dataset=_make_train_dataset(),
        test_dataset=_make_test_dataset(),
        batch_size=8,
    )
    df = pl.read_parquet(out)
    keys = set(df.get_column("canonical_parcel_id").to_list())
    # _make_test_dataset: ("90001", inst 1 & 2), ("90002", inst 1 & 2), ("90003", 1).
    # Fake bridge: raster_id = instance * 1000 + 7.
    assert keys == {"90001_1007", "90001_2007", "90002_1007", "90002_2007", "90003_1007"}
    # The raw dataset keys ("{patch}_{instance}") must NOT survive.
    assert "90001_1" not in keys


def test_head_predicts_separable_classes(tmp_path, _patch_bridge) -> None:
    """With class-separable embeddings the head recovers each fold-5 parcel's class.

    semantic18 = RAW class_id - 1; the argmax pred_class must match for the clean
    synthetic separation (sanity that the head trains and predicts, not noise).
    """
    out = tmp_path / "oof.parquet"
    materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=_StubExtractor(),
        train_dataset=_make_train_dataset(),
        test_dataset=_make_test_dataset(),
        batch_size=8,
    )
    df = pl.read_parquet(out).sort("canonical_parcel_id")
    pred = dict(
        zip(
            df.get_column("canonical_parcel_id").to_list(),
            df.get_column("pred_class").to_list(),
            strict=True,
        )
    )
    # raster_id = instance * 1000 + 7; class_id RAW -> semantic18 = cid - 1.
    assert pred["90001_1007"] == 0  # patch 90001 inst 1 -> class 1 -> idx 0
    assert pred["90001_2007"] == 2  # inst 2 -> class 3 -> idx 2
    assert pred["90002_1007"] == 7  # inst 1 -> class 8 -> idx 7
    assert pred["90002_2007"] == 17  # inst 2 -> class 18 -> idx 17
    assert pred["90003_1007"] == 1  # inst 1 -> class 2 -> idx 1


# ---------------------------------------------------------------------------
# Anti-leakage: the head never sees fold-5.
# ---------------------------------------------------------------------------


def test_head_never_fits_on_fold5(tmp_path, _patch_bridge, monkeypatch) -> None:
    """The LogisticRegression is fit ONLY on the fold-1-4 embeddings (R-LEAK).

    Captures the matrix handed to ``LogisticRegression.fit`` and asserts its row
    count equals the TRAIN dataset size (post Background/Void filter, none here),
    never the TRAIN+TEST union -- proving fold-5 is excluded from training.
    """
    from sklearn.linear_model import LogisticRegression

    captured: dict[str, int] = {}
    real_fit = LogisticRegression.fit

    def _spy_fit(self, x, y, *args, **kwargs):
        captured["n_rows"] = int(np.asarray(x).shape[0])
        return real_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", _spy_fit)

    train_ds = _make_train_dataset(per_class=4)  # 18 * 4 = 72 train parcels.
    test_ds = _make_test_dataset()  # 5 fold-5 parcels.
    out = tmp_path / "oof.parquet"
    materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=_StubExtractor(),
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=8,
    )
    assert captured["n_rows"] == len(train_ds) == 72
    assert captured["n_rows"] != len(train_ds) + len(test_ds)


def test_background_void_rows_dropped_from_training(tmp_path, _patch_bridge, monkeypatch) -> None:
    """RAW class_id 0 (Background) / 19 (Void) parcels never reach the head.

    The semantic18 LUT maps 0 and 19 to the ignore label; those rows must be
    filtered before fitting, so the captured fit row-count counts only the 18
    agronomic-class parcels.
    """
    from sklearn.linear_model import LogisticRegression

    captured: dict[str, int] = {}
    real_fit = LogisticRegression.fit

    def _spy_fit(self, x, y, *args, **kwargs):
        captured["n_rows"] = int(np.asarray(x).shape[0])
        return real_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", _spy_fit)

    # 6 valid agronomic parcels (classes 1..3 x2) + 2 ignore (Background 0, Void 19).
    train_samples = [
        ("10001", 1, 1),
        ("10001", 2, 1),
        ("10002", 3, 2),
        ("10002", 4, 2),
        ("10003", 5, 3),
        ("10003", 6, 3),
        ("10004", 7, 0),  # Background -> dropped
        ("10004", 8, 19),  # Void -> dropped
    ]
    out = tmp_path / "oof.parquet"
    materialize_ft18_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=_StubExtractor(),
        train_dataset=_StubParcelDataset(train_samples),
        test_dataset=_make_test_dataset(),
        batch_size=8,
    )
    assert captured["n_rows"] == 6  # the 2 Background/Void parcels were dropped.


def test_empty_fold5_raises(tmp_path, _patch_bridge) -> None:
    """An empty fold-5 dataset is a hard error (no silent empty OOF)."""
    with pytest.raises(ValueError, match="empty"):
        materialize_ft18_oof(
            out_path=tmp_path / "oof.parquet",
            pastis_root=tmp_path,
            extractor=_StubExtractor(),
            train_dataset=_make_train_dataset(),
            test_dataset=_StubParcelDataset([]),
            batch_size=8,
        )


def test_extractor_wrong_dim_rejected(tmp_path, _patch_bridge) -> None:
    """An extractor that does not return 512-dim embeddings is rejected."""

    class _BadExtractor:
        def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
            return torch.zeros((crops.shape[0], 64), dtype=torch.float32)  # 64 != 512

    with pytest.raises(ValueError, match="512"):
        materialize_ft18_oof(
            out_path=tmp_path / "oof.parquet",
            pastis_root=tmp_path,
            extractor=_BadExtractor(),
            train_dataset=_make_train_dataset(),
            test_dataset=_make_test_dataset(),
            batch_size=8,
        )


def test_default_checkpoint_constant_is_safetensors() -> None:
    """The default fine-tuned checkpoint points at a FarSLIP student weight file."""
    assert farslip_ft18.DEFAULT_FARSLIP_CHECKPOINT.endswith(".safetensors")
    assert "farslip" in farslip_ft18.DEFAULT_FARSLIP_CHECKPOINT
