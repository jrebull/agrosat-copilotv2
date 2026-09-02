"""Tests for the FarSLIP zero-shot parcel member (US-042, EPIC 6).

The whole module is exercised with deterministic STUBS (no real PASTIS-R, no
FarSLIP/CLIP weights, no GPU, no DVC blobs):

- ``_StubExtractor`` returns deterministic ``encode_text`` / ``extract_embeddings``
  tensors keyed by the input text / a hidden per-parcel signature, so the bank,
  the proba matrix and the parquet are fully reproducible and assertable.
- ``_StubParcelDataset`` mimics the :class:`ParcelCropDataset` contract
  (``image`` + ``parcel_id`` ``"{patch}_{instance}"`` + RAW ``class_id``).

Coverage: text bank shape ``(18, 512)`` + L2-norm, per-parcel proba shape
``(18,)`` summing to 1, the molde parquet schema, that NOTHING is trained
(zero-shot), and that the default checkpoint is FarSLIP (not ``None`` / CLIP).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from ml.ensemble.farslip_zeroshot import (
    DEFAULT_FARSLIP_CHECKPOINT,
    DEFAULT_LOGIT_SCALE,
    SEMANTIC18_CROP_NAMES,
    build_text_class_bank,
    materialize_zeroshot_oof,
    zeroshot_parcel_proba,
)
from ml.utils.parcel_reconcile import PROB_COLUMNS

_EMBED_DIM = 512
_NUM_CLASSES = 18


def _seeded_vec(key: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """Deterministic L2-normalized vector seeded from ``key``."""
    seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / float(np.linalg.norm(vec))


class _StubExtractor:
    """Deterministic stub of the text + vision sides of ``FarSLIPExtractor``.

    Records whether any gradient-bearing op ran so the test can assert that
    zero-shot trains nothing.
    """

    def __init__(self) -> None:
        self.fit_called = False
        self.train_called = False
        self.n_text_calls = 0
        self.n_image_calls = 0

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        self.n_text_calls += 1
        rows = [_seeded_vec(f"text::{t}") for t in texts]
        return torch.from_numpy(np.stack(rows, axis=0))

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        self.n_image_calls += 1
        # Signature from the crop content so each parcel is deterministic.
        sig = float(crops.float().sum().item())
        vec = _seeded_vec(f"img::{sig:.4f}")
        return torch.from_numpy(vec[None, :])

    # The presence of these would-be training hooks lets the test prove they are
    # never invoked by the zero-shot path.
    def fit(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        self.fit_called = True

    def train(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        self.train_called = True


class _StubParcelDataset:
    """Minimal stand-in for :class:`ParcelCropDataset`.

    Each item exposes ``image`` ``(4, 8, 8)``, ``parcel_id`` ``"{patch}_{inst}"``
    and a RAW PASTIS ``class_id`` (1..18).
    """

    def __init__(self, n: int = 6) -> None:
        self._items: list[dict[str, Any]] = []
        for i in range(n):
            img = torch.full((4, 8, 8), float(i + 1) * 0.01, dtype=torch.float32)
            self._items.append(
                {
                    "image": img,
                    "parcel_id": f"1000{i}_{i + 1}",
                    "patch_id": f"1000{i}",
                    "class_id": (i % _NUM_CLASSES) + 1,  # RAW 1..18
                    "caption": "",
                    "bbox": (0, 0, 8, 8),
                }
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._items[idx]


def test_text_bank_shape_and_l2_norm() -> None:
    """The text bank is ``(18, 512)`` with every row L2-normalized."""
    extractor = _StubExtractor()
    bank = build_text_class_bank(extractor, SEMANTIC18_CROP_NAMES)
    assert bank.shape == (_NUM_CLASSES, _EMBED_DIM)
    assert bank.dtype == np.float32
    norms = np.linalg.norm(bank, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_text_bank_orders_by_class_names() -> None:
    """Row order follows ``class_names``; distinct names give distinct vectors."""
    extractor = _StubExtractor()
    bank = build_text_class_bank(extractor, SEMANTIC18_CROP_NAMES)
    # No two crop classes collapse to the same averaged vector.
    gram = bank @ bank.T
    off_diag = gram - np.eye(_NUM_CLASSES)
    assert float(off_diag.max()) < 0.999


def test_zeroshot_parcel_proba_is_row_stochastic() -> None:
    """Each parcel row has shape ``(18,)`` and sums to 1 (post-softmax)."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=5)
    bank = build_text_class_bank(extractor, SEMANTIC18_CROP_NAMES)
    proba, parcel_ids, class_ids = zeroshot_parcel_proba(extractor, dataset, bank)
    assert proba.shape == (5, _NUM_CLASSES)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0.0).all()
    assert len(parcel_ids) == 5
    assert class_ids == [1, 2, 3, 4, 5]


def test_single_parcel_row_sums_to_one() -> None:
    """A single parcel yields a valid ``(18,)`` distribution."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=1)
    bank = build_text_class_bank(extractor, SEMANTIC18_CROP_NAMES)
    proba, _ids, _cls = zeroshot_parcel_proba(extractor, dataset, bank)
    assert proba.shape == (1, _NUM_CLASSES)
    assert proba.ndim == 2
    assert pytest.approx(1.0, abs=1e-5) == float(proba[0].sum())


def test_logit_scale_changes_sharpness_not_sum() -> None:
    """A higher fixed temperature sharpens the distribution but keeps sum 1."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=4)
    bank = build_text_class_bank(extractor, SEMANTIC18_CROP_NAMES)
    soft, _i, _c = zeroshot_parcel_proba(extractor, dataset, bank, logit_scale=1.0)
    sharp, _i2, _c2 = zeroshot_parcel_proba(extractor, dataset, bank, logit_scale=50.0)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(sharp.sum(axis=1), 1.0, atol=1e-5)
    # Sharper temperature -> higher peak probability per row.
    assert (sharp.max(axis=1) >= soft.max(axis=1) - 1e-6).all()


def test_materialize_writes_molde_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The parquet has the EXACT molde schema and post-softmax probabilities."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=6)

    # The key-translation bridge needs PASTIS rasters; stub it (identity on inst).
    def _fake_map(patch: str, _root: Path) -> dict[int, int]:
        return {i: i * 1000 for i in range(1, 50)}

    monkeypatch.setattr("ml.utils.parcel_reconcile.instance_to_parcel_id_map", _fake_map)

    out = tmp_path / "oof_parcel_farslip-zeroshot_fold5.parquet"
    materialize_zeroshot_oof(
        out_path=out,
        pastis_root=tmp_path,
        extractor=extractor,
        dataset=dataset,
    )
    assert out.exists()
    df = pl.read_parquet(out)

    expected_cols = ["canonical_parcel_id", *PROB_COLUMNS, "pred_class", "n_pixels"]
    assert df.columns == expected_cols
    assert df.schema["canonical_parcel_id"] == pl.Utf8
    for col in PROB_COLUMNS:
        assert df.schema[col] == pl.Float32
    assert df.schema["pred_class"] == pl.Int64
    assert df.schema["n_pixels"] == pl.Int64

    prob_matrix = df.select(PROB_COLUMNS).to_numpy()
    assert np.allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-5)
    # pred_class is the argmax of the prob columns.
    assert (df["pred_class"].to_numpy() == prob_matrix.argmax(axis=1)).all()
    # Canonical key was translated to the ParcelIDs space ("{patch}_{raster}").
    assert df["canonical_parcel_id"].to_list()[0].endswith("000")


def test_canonical_keys_translated_not_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialized keys use the ParcelIDs space, never the raw instance space."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=3)

    def _fake_map(patch: str, _root: Path) -> dict[int, int]:
        return {i: i * 1000 for i in range(1, 50)}

    monkeypatch.setattr("ml.utils.parcel_reconcile.instance_to_parcel_id_map", _fake_map)
    out = tmp_path / "oof.parquet"
    materialize_zeroshot_oof(
        out_path=out, pastis_root=tmp_path, extractor=extractor, dataset=dataset
    )
    keys = set(pl.read_parquet(out)["canonical_parcel_id"].to_list())
    # Raw instance keys ("10000_1") must NOT appear; translated ("10000_1000") must.
    assert "10000_1" not in keys
    assert "10000_1000" in keys


def test_zeroshot_trains_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No fit/train hook is ever called: zero-shot is a pure forward pass."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=4)

    def _fake_map(patch: str, _root: Path) -> dict[int, int]:
        return {i: i * 1000 for i in range(1, 50)}

    monkeypatch.setattr("ml.utils.parcel_reconcile.instance_to_parcel_id_map", _fake_map)
    materialize_zeroshot_oof(
        out_path=tmp_path / "oof.parquet",
        pastis_root=tmp_path,
        extractor=extractor,
        dataset=dataset,
    )
    assert extractor.fit_called is False
    assert extractor.train_called is False
    # The vision tower was queried once per parcel; the text bank once per class.
    assert extractor.n_image_calls == len(dataset)
    assert extractor.n_text_calls == _NUM_CLASSES


def test_default_checkpoint_is_farslip_not_clip() -> None:
    """The default checkpoint is the FarSLIP weights file, NEVER None (CLIP)."""
    assert DEFAULT_FARSLIP_CHECKPOINT == "checkpoints/farslip/faithful_v2/best.safetensors"
    assert DEFAULT_FARSLIP_CHECKPOINT is not None


def test_default_logit_scale_is_fixed_clip_temperature() -> None:
    """The default temperature is the fixed CLIP value (a priori, not tuned)."""
    assert DEFAULT_LOGIT_SCALE == pytest.approx(1.0 / 0.07)


def test_active_subset_scatters_to_18(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reduced active-class run still writes full 18-column rows summing to 1."""
    extractor = _StubExtractor()
    dataset = _StubParcelDataset(n=3)

    def _fake_map(patch: str, _root: Path) -> dict[int, int]:
        return {i: i * 1000 for i in range(1, 50)}

    monkeypatch.setattr("ml.utils.parcel_reconcile.instance_to_parcel_id_map", _fake_map)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "oof.parquet"
        materialize_zeroshot_oof(
            out_path=out,
            pastis_root=Path(tmp),
            extractor=extractor,
            dataset=dataset,
            n_classes=4,
        )
        df = pl.read_parquet(out)
    prob_matrix = df.select(PROB_COLUMNS).to_numpy()
    assert prob_matrix.shape[1] == _NUM_CLASSES
    assert np.allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-5)
