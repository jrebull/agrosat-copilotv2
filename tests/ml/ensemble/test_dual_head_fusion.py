"""Deterministic CPU tests for the E-a dual-head fusion (US-041).

Covers ``ml/ensemble/dual_head_fusion.py``: the visual class-prototype builder,
the per-pixel cosine broadcast, and the convex fusion of the
:class:`DualHeadFusionHead` (alpha=0/1 limits, post-softmax contract,
anti-leakage). The FarSLIP student and the parcel dataset are STUBBED in memory
(no PASTIS-R, no network, no GPU); the band-indexing guard (4 vs 10) is asserted
against the real :class:`ParcelCropDataset` contract.

Project conventions: numpy/torch only at the boundary, no emojis, Spanish prose,
English identifiers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.ensemble.dual_head_fusion import (
    DualHeadFusionHead,
    _renormalize,
    build_class_prototypes,
    farslip_cosine_map,
)

_EMBED_DIM = 768
_N_CLASSES = 18
_SIDE = 128


class _StubStudent(torch.nn.Module):
    """Minimal stand-in for the FarSLIP CLIPVisionModel student.

    Returns a fixed CLS-768 per input that depends only on the sum of the crop,
    so distinct parcels get distinct (deterministic) embeddings. Output mimics the
    HF contract: an object with ``last_hidden_state`` ``(B, 1+P, 768)``.
    """

    def __init__(self, embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, *, pixel_values: torch.Tensor):
        b = pixel_values.shape[0]
        # Deterministic per-sample CLS whose DIRECTION (not just magnitude) depends
        # on the crop mean, so distinct parcels survive L2-normalization as
        # distinct embeddings (a pure magnitude scale would collapse after norm).
        means = pixel_values.reshape(b, -1).mean(dim=1)  # (B,)
        idx = torch.arange(self.embed_dim, dtype=torch.float32)
        # Phase the ramp by the crop mean -> different angle per parcel.
        cls = torch.stack([torch.sin(idx * (1.0 + m) * 0.01) + 1.0 for m in means])
        last_hidden = cls.unsqueeze(1)  # (B, 1, 768) -> [:,0,:] is the CLS
        return type("Out", (), {"last_hidden_state": last_hidden})()


class _StubParcelDataset:
    """In-memory ParcelCropDataset stub with the same public contract used here.

    Exposes ``_samples`` (``(parcel_id, src, local_id, class_id)``), ``__len__``,
    ``__getitem__`` -> ``{"image", "class_id", ...}``, matching what
    ``build_class_prototypes`` / ``farslip_cosine_map`` consume.
    """

    def __init__(self, patch_id: str, parcels: list[tuple[int, int]]) -> None:
        # parcels: list of (local_id, class_id).
        self._samples = [(f"{patch_id}_{lid}", patch_id, lid, cid) for lid, cid in parcels]
        self._patch_id = patch_id

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        _pid, _src, lid, cid = self._samples[idx]
        # Crop value encodes the local id so embeddings differ per parcel.
        img = torch.full((4, 224, 224), 0.1 * (lid + 1), dtype=torch.float32)
        return {
            "image": img,
            "parcel_id": _pid,
            "patch_id": self._patch_id,
            "class_id": cid,
            "caption": "",
            "bbox": (0, 0, 4, 4),
        }


def _proto_bank(n_classes: int = _N_CLASSES) -> np.ndarray:
    """A valid L2-normalized ``(C, 768)`` prototype bank (one-hot-ish per class)."""
    rng = np.random.default_rng(0)
    bank = rng.standard_normal((n_classes, _EMBED_DIM)).astype(np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    return (bank / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# _renormalize + fusion arithmetic.
# ---------------------------------------------------------------------------


def test_renormalize_makes_class_axis_sum_to_one() -> None:
    """A perturbed map renormalizes to sum-to-1 over the class axis."""
    arr = np.abs(np.random.default_rng(1).standard_normal((_N_CLASSES, 8, 8)))
    out = _renormalize(arr)
    sums = out.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_fuse_alpha_limits_recover_each_head() -> None:
    """alpha=1 returns the TSViT head; alpha=0 returns the FarSLIP head."""
    head = DualHeadFusionHead()
    rng = np.random.default_rng(2)
    p_t = _renormalize(np.abs(rng.standard_normal((_N_CLASSES, 4, 4))))
    p_f = _renormalize(np.abs(rng.standard_normal((_N_CLASSES, 4, 4))))
    fused_1 = head._fuse(p_t, p_f, 1.0)
    fused_0 = head._fuse(p_t, p_f, 0.0)
    assert np.allclose(fused_1, p_t, atol=1e-6)
    assert np.allclose(fused_0, p_f, atol=1e-6)


def test_fuse_is_post_softmax() -> None:
    """Any alpha in [0,1] keeps the fused map a valid post-softmax distribution."""
    head = DualHeadFusionHead()
    rng = np.random.default_rng(3)
    p_t = _renormalize(np.abs(rng.standard_normal((_N_CLASSES, 4, 4))))
    p_f = _renormalize(np.abs(rng.standard_normal((_N_CLASSES, 4, 4))))
    for alpha in (0.0, 0.25, 0.5, 0.85, 1.0):
        fused = head._fuse(p_t, p_f, alpha)
        assert np.allclose(fused.sum(axis=0), 1.0, atol=1e-4)
        assert (fused >= 0).all()


# ---------------------------------------------------------------------------
# build_class_prototypes.
# ---------------------------------------------------------------------------


def test_build_class_prototypes_shape_and_norm() -> None:
    """The bank is ``(C, 768)`` and each populated row is L2-normalized."""
    student = _StubStudent()
    # Two classes present (3 and 8), each with two parcels.
    ds = _StubParcelDataset("10000", [(1, 3), (2, 3), (3, 8), (4, 8)])
    bank = build_class_prototypes(
        student, ds, class_ids=list(range(1, 19)), device="cpu", batch_size=2
    )
    assert bank.shape == (_N_CLASSES, _EMBED_DIM)
    # Rows for class 3 (idx 2) and class 8 (idx 7) are populated and normalized.
    assert np.isclose(np.linalg.norm(bank[2]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(bank[7]), 1.0, atol=1e-5)
    # An absent class keeps a zero row (honest "unknown").
    assert np.allclose(bank[0], 0.0)


def test_build_class_prototypes_empty_dataset_raises() -> None:
    """An empty dataset cannot yield prototypes."""
    student = _StubStudent()
    ds = _StubParcelDataset("10000", [])
    with pytest.raises(ValueError, match="dataset is empty"):
        build_class_prototypes(student, ds, class_ids=[1, 2], device="cpu")


# ---------------------------------------------------------------------------
# farslip_cosine_map.
# ---------------------------------------------------------------------------


def test_farslip_cosine_map_is_post_softmax_and_broadcasts() -> None:
    """Cosine map is post-softmax and assigns each parcel's pixels its distribution."""
    student = _StubStudent()
    ds = _StubParcelDataset("10000", [(1, 3), (2, 8)])
    bank = _proto_bank()
    parcel_ids_map = np.zeros((_SIDE, _SIDE), dtype=np.int64)
    parcel_ids_map[:64, :] = 1  # parcel 1 top half
    parcel_ids_map[64:, :] = 2  # parcel 2 bottom half
    cmap = farslip_cosine_map(
        student,
        bank,
        patch_id="10000",
        dataset=ds,
        parcel_ids_map=parcel_ids_map,
        device="cpu",
    )
    assert cmap.shape == (_N_CLASSES, _SIDE, _SIDE)
    assert np.allclose(cmap.sum(axis=0), 1.0, atol=1e-5)
    # Within a parcel every pixel shares the same distribution (broadcast).
    top = cmap[:, 10, 10]
    top2 = cmap[:, 50, 80]
    assert np.allclose(top, top2, atol=1e-6)
    # Different parcels -> different distributions.
    bottom = cmap[:, 100, 10]
    assert not np.allclose(top, bottom, atol=1e-6)


def test_farslip_cosine_map_background_is_uniform() -> None:
    """Background pixels (id 0) get the uniform distribution (no information)."""
    student = _StubStudent()
    ds = _StubParcelDataset("10000", [(1, 3)])
    bank = _proto_bank()
    parcel_ids_map = np.zeros((_SIDE, _SIDE), dtype=np.int64)
    parcel_ids_map[:32, :32] = 1
    cmap = farslip_cosine_map(
        student,
        bank,
        patch_id="10000",
        dataset=ds,
        parcel_ids_map=parcel_ids_map,
        device="cpu",
    )
    bg = cmap[:, 100, 100]
    assert np.allclose(bg, 1.0 / _N_CLASSES, atol=1e-6)


def test_farslip_cosine_map_rejects_wrong_proto_dim() -> None:
    """A prototype bank that is not (C, 768) is rejected (R-DIM-768)."""
    student = _StubStudent()
    ds = _StubParcelDataset("10000", [(1, 3)])
    bad = np.zeros((_N_CLASSES, 512), dtype=np.float32)  # 512, not 768
    with pytest.raises(ValueError, match="prototypes must be"):
        farslip_cosine_map(
            student,
            bad,
            patch_id="10000",
            dataset=ds,
            parcel_ids_map=np.zeros((_SIDE, _SIDE), dtype=np.int64),
            device="cpu",
        )


def test_farslip_cosine_map_scatters_n4_bank_into_18_space() -> None:
    """An N=4 bank is scattered into the 18-class space (R-CLASSES-MISMATCH).

    The N=4 parcel-level FarSLIP produces a 4-row bank; the map MUST come out
    (18, H, W) -- with the 4 active classes at their PASTIS slots (class_id-1) and
    the other 14 at zero -- so it is fusible with the TSViT (18, H, W) softmax.
    This is the exact bug that crashed E-a:
    ``operands could not be broadcast (18,128,128) (4,128,128)``.
    """
    student = _StubStudent()
    active = [1, 3, 2, 8]  # the N=4 curriculum head (Meadow, Corn, SWWheat, Grape)
    ds = _StubParcelDataset("10000", [(1, 3), (2, 8)])
    bank = _proto_bank(n_classes=len(active))  # (4, 768)
    parcel_ids_map = np.zeros((_SIDE, _SIDE), dtype=np.int64)
    parcel_ids_map[:64, :] = 1
    parcel_ids_map[64:, :] = 2
    cmap = farslip_cosine_map(
        student,
        bank,
        patch_id="10000",
        dataset=ds,
        parcel_ids_map=parcel_ids_map,
        class_ids=active,
        device="cpu",
    )
    # Always 18-wide -> fusible with TSViT.
    assert cmap.shape == (_N_CLASSES, _SIDE, _SIDE)
    assert np.allclose(cmap.sum(axis=0), 1.0, atol=1e-5)
    # A parcel pixel: probability lives ONLY on the active slots (class_id-1).
    px = cmap[:, 10, 10]
    active_slots = {c - 1 for c in active}
    inactive_slots = set(range(_N_CLASSES)) - active_slots
    assert all(px[s] == 0.0 for s in inactive_slots)
    assert px[[c - 1 for c in active]].sum() == pytest.approx(1.0, abs=1e-6)


def test_farslip_cosine_map_no_class_ids_needs_18_bank() -> None:
    """Without class_ids, a non-18 bank cannot be placed in the 18-class space."""
    student = _StubStudent()
    ds = _StubParcelDataset("10000", [(1, 3)])
    bank = _proto_bank(n_classes=4)  # 4 rows, no class_ids -> ambiguous
    with pytest.raises(ValueError, match="no class_ids"):
        farslip_cosine_map(
            student,
            bank,
            patch_id="10000",
            dataset=ds,
            parcel_ids_map=np.zeros((_SIDE, _SIDE), dtype=np.int64),
            device="cpu",
        )


# ---------------------------------------------------------------------------
# DualHeadFusionHead contract.
# ---------------------------------------------------------------------------


def test_alpha_before_fit_raises() -> None:
    """Accessing alpha before fit is an error (no silent default)."""
    head = DualHeadFusionHead()
    with pytest.raises(RuntimeError, match="alpha is not set"):
        _ = head.alpha


def test_set_prototypes_validates_shape() -> None:
    """set_prototypes enforces the (n_classes, 768) contract."""
    head = DualHeadFusionHead()
    head.set_prototypes(np.zeros((_N_CLASSES, _EMBED_DIM), dtype=np.float32))
    assert head._prototypes is not None
    with pytest.raises(ValueError, match="prototypes must be"):
        head.set_prototypes(np.zeros((_N_CLASSES, 512), dtype=np.float32))


def test_mlflow_params_reports_embedding_dim_768() -> None:
    """mlflow_params documents the 768 embedding dim and the member name."""
    head = DualHeadFusionHead()
    params = head.mlflow_params()
    assert params["embedding_dim"] == 768
    assert params["tsvit_member"] == "tsvit-pheno-fullm"
    assert params["n_classes"] == 18


def test_evaluate_rejects_non_fold5() -> None:
    """The inherited evaluate is fold-5-only (anti-leakage)."""
    head = DualHeadFusionHead()
    y = np.zeros((10,), dtype=np.int64)
    with pytest.raises(ValueError, match="fold-5"):
        head.evaluate(y_true=y, y_pred=y, fold=4)


def test_validate_probs_rejects_logits() -> None:
    """The inherited guard rejects logit-like (negative) inputs as a fusion output."""
    head = DualHeadFusionHead()
    logits = np.full((_N_CLASSES, 2, 2), -0.5)  # negative -> not a softmax
    with pytest.raises(ValueError, match=r"negative|sum to 1"):
        head.validate_probs(logits, class_axis=0, name="bad")


# ---------------------------------------------------------------------------
# Band-indexing guard (4 vs 10) — the real ParcelCropDataset contract.
# ---------------------------------------------------------------------------


def test_parcel_crop_is_four_band_not_ten() -> None:
    """The FarSLIP student consumes 4-band peak-NDVI crops, NOT the 10 S2 bands.

    Guards the plan's band-mismatch risk: the parcel crop must be 4-channel
    (peak-NDVI composite aligned with the 4-in-channel student), never the raw
    10-band S2 stack. Asserted on the real dataset's crop helper with a synthetic
    composite (no PASTIS-R needed).
    """
    from ml.farslip.parcel_crop_dataset import _crop_parcel, _resize_crop

    composite = np.zeros((4, 16, 16), dtype=np.float32)
    instance = np.zeros((16, 16), dtype=np.int64)
    instance[2:8, 2:8] = 1
    composite[:, 2:8, 2:8] = 0.5
    crop = _crop_parcel(composite, instance, instance_id=1)
    assert crop.shape[0] == 4, "FarSLIP crop must be 4-band (peak-NDVI), not 10"
    resized = _resize_crop(crop, 224)
    assert resized.shape == (4, 224, 224)
