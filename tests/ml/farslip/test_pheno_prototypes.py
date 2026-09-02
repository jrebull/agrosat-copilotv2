"""Tests for the US-034 phenological-prototype fix.

Covers the critical fix that replaced ``torch.randn`` (contrastive alignment
against noise) with the REAL phenological prototypes of US-033:

- the CAP-32 -> PASTIS-18 cardinality bridge (``cap_pastis_mapping``),
- the frozen orthogonal reprojection MiniLM-384 -> CLS-768 (``_proto_to_clip_proj``),
- the dimension assert in ``set_text_prototypes`` (fail fast on wrong D),
- the region-major row order consumed by the loss targets,
- a smoke check that the loss decreases with real (non-random) prototypes.

All algebra runs on CPU with no network and no real dataset. The trainer-bound
tests use a lightweight fake teacher (only ``config.hidden_size`` is needed by
the reprojection / assert path) so CLIP is never downloaded; the loss smoke uses
the standalone ``RegionCategoryAlignmentLoss``. A single optional test loads the
real US-033 parquet and is skipped when the DVC binary is absent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.farslip.cap_pastis_mapping import (
    CAP_TO_PASTIS,
    expand_to_cap,
    load_cap_to_pastis,
)
from ml.farslip.distill import (
    FarSLIPDistillationTrainer,
    RegionCategoryAlignmentLoss,
)
from ml.farslip.train import build_text_prototypes

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARQUET = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"

# The 32 CAP classes in cap_vocabulary.yaml declaration order (category_id 0..31).
_CAP_CLASSES_32 = [
    "mais",
    "frumento",
    "vite",
    "olivo",
    "riso",
    "foraggio",
    "ortaggi",
    "girasole",
    "soia",
    "colza",
    "orzo",
    "sorgo",
    "pomodoro",
    "patata",
    "barbabietola",
    "tabacco",
    "lino",
    "canapa",
    "agrumi",
    "melo",
    "pero",
    "pesco",
    "mandorlo",
    "noce",
    "prato_permanente",
    "pascolo",
    "set_aside",
    "serra",
    "vivai",
    "floricoltura",
    "leguminose",
    "altro",
]


# ---------------------------------------------------------------------------
# Fake trainer: only exposes what _proto_to_clip_proj / set_text_prototypes use.
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Minimal stand-in binding ``_proto_to_clip_proj``/``set_text_prototypes``.

    Avoids loading CLIP: the reprojection and the dimension assert only read
    ``teacher.config.hidden_size`` and ``config.seed`` and write
    ``_text_prototypes``. We reuse the real unbound methods of the trainer.
    """

    def __init__(self, hidden_size: int = 768, seed: int = 42) -> None:
        self.teacher = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size))
        self.config = SimpleNamespace(seed=seed)
        self.device = torch.device("cpu")
        self._text_prototypes: torch.Tensor | None = None

    # Bind the real implementations under test.
    _proto_to_clip_proj = FarSLIPDistillationTrainer._proto_to_clip_proj
    set_text_prototypes = FarSLIPDistillationTrainer.set_text_prototypes


# ---------------------------------------------------------------------------
# CAP-32 -> PASTIS-18 mapping + expand_to_cap (AC-3 cardinality).
# ---------------------------------------------------------------------------


def test_cap_mapping_covers_all_32_classes() -> None:
    mapping = load_cap_to_pastis()
    assert set(mapping) == set(_CAP_CLASSES_32)
    assert len(mapping) == 32
    # Every value is a valid PASTIS crop class_id 1..18 (never 0 / Background).
    assert all(1 <= v <= 18 for v in mapping.values())
    assert mapping is not CAP_TO_PASTIS  # returns a copy, not the module dict


def test_expand_to_cap_cardinality() -> None:
    rng = np.random.default_rng(0)
    proto_18 = rng.standard_normal((18, 384)).astype(np.float32)
    proto_cap = expand_to_cap(proto_18, _CAP_CLASSES_32)
    assert proto_cap.shape == (32, 384)
    tiled = np.tile(proto_cap, (3, 1))
    assert tiled.shape == (96, 384)
    # 96 rows pass the loss validation (3 regions x 32 categories).
    loss = RegionCategoryAlignmentLoss(n_regions=3, n_categories=32)
    student = torch.randn(4, 384)
    region_ids = torch.tensor([0, 1, 2, 0])
    cat_ids = torch.tensor([0, 5, 31, 17])
    out = loss(student, torch.from_numpy(tiled).float(), region_ids, cat_ids)
    assert torch.isfinite(out)


def test_expand_to_cap_assigns_correct_pastis_prototype() -> None:
    # One-hot per PASTIS class so we can recover the assigned class_id by argmax.
    proto_18 = np.eye(18, dtype=np.float32)  # row r -> class_id r+1
    proto_cap = expand_to_cap(proto_18, _CAP_CLASSES_32)
    mapping = load_cap_to_pastis()
    for i, cap_slug in enumerate(_CAP_CLASSES_32):
        expected_class_id = mapping[cap_slug]
        assigned_row = int(proto_cap[i].argmax())
        assert assigned_row + 1 == expected_class_id, (
            f"{cap_slug}: got class {assigned_row + 1}, want {expected_class_id}"
        )


def test_expand_to_cap_respects_explicit_class_ids() -> None:
    # Reversed class_ids: row 0 is class_id 18, row 17 is class_id 1.
    proto_18 = np.eye(18, dtype=np.float32)
    class_ids = list(range(18, 0, -1))
    proto_cap = expand_to_cap(proto_18, ["mais"], pastis_class_ids=class_ids)
    # mais -> Corn (class_id 3); with reversed ids, class_id 3 is at row 15.
    assert int(proto_cap[0].argmax()) == class_ids.index(3)


def test_expand_to_cap_unknown_class_raises() -> None:
    proto_18 = np.eye(18, dtype=np.float32)
    with pytest.raises(ValueError, match="no entry"):
        expand_to_cap(proto_18, ["not_a_real_cap_class"])


def test_expand_to_cap_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        expand_to_cap(np.zeros((18,), dtype=np.float32), ["mais"])


# ---------------------------------------------------------------------------
# Frozen orthogonal reprojection 384 -> 768 (AC-4).
# ---------------------------------------------------------------------------


def test_proj_is_semi_orthogonal() -> None:
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    emb = torch.randn(5, 384)
    trainer._proto_to_clip_proj(emb)  # triggers W construction
    w = trainer._proto_proj_w_384  # type: ignore[attr-defined]
    assert w.shape == (768, 384)
    gram = w.t() @ w  # (384, 384) should be ~ I_384
    assert torch.allclose(gram, torch.eye(384), atol=1e-4)


def test_proj_preserves_norm() -> None:
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    x = F.normalize(torch.randn(8, 384), dim=-1)  # unit-norm rows
    y = trainer._proto_to_clip_proj(x)
    assert y.shape == (8, 768)
    assert torch.allclose(y.norm(dim=-1), torch.ones(8), atol=1e-4)


def test_proj_preserves_inner_products() -> None:
    # Orthonormal columns preserve relative angles/inner products between rows.
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    x = torch.randn(6, 384)
    y = trainer._proto_to_clip_proj(x)
    assert torch.allclose(x @ x.t(), y @ y.t(), atol=1e-3)


def test_proj_is_frozen_and_deterministic() -> None:
    t1 = _FakeTrainer(hidden_size=768, seed=42)
    t2 = _FakeTrainer(hidden_size=768, seed=42)
    emb = torch.randn(3, 384)
    y1 = t1._proto_to_clip_proj(emb)
    y2 = t2._proto_to_clip_proj(emb)
    assert torch.allclose(y1, y2)  # same seed -> identical
    assert t1._proto_proj_w_384.requires_grad is False  # type: ignore[attr-defined]
    assert y1.requires_grad is False
    # Different seed -> different projection.
    t3 = _FakeTrainer(hidden_size=768, seed=7)
    assert not torch.allclose(t3._proto_to_clip_proj(emb), y1)


def test_proj_rejects_downcast() -> None:
    trainer = _FakeTrainer(hidden_size=256, seed=42)
    with pytest.raises(ValueError, match="hidden_size"):
        trainer._proto_to_clip_proj(torch.randn(2, 384))


# ---------------------------------------------------------------------------
# set_text_prototypes: reprojection to 768 + assert (AC-2).
# ---------------------------------------------------------------------------


def test_set_text_prototypes_reprojects_384_to_768() -> None:
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    emb = torch.randn(96, 384)
    trainer.set_text_prototypes(emb)
    assert trainer._text_prototypes is not None
    assert trainer._text_prototypes.shape == (96, 768)
    assert trainer._text_prototypes.requires_grad is False


def test_set_text_prototypes_passes_768_through() -> None:
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    emb = torch.randn(96, 768)
    trainer.set_text_prototypes(emb)
    assert trainer._text_prototypes.shape == (96, 768)
    # Already at hidden_size -> no reprojection buffer created.
    assert not hasattr(trainer, "_proto_proj_w_768")


def test_set_text_prototypes_wrong_dim_raises_clear_error() -> None:
    # 512 (the inference CLIP-shared space) is NOT the loss space; must fail fast
    # with a clear ValueError, not the opaque matmul RuntimeError.
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    with pytest.raises(ValueError, match="unsupported"):
        trainer.set_text_prototypes(torch.randn(96, 512))


def test_set_text_prototypes_non_2d_raises() -> None:
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    with pytest.raises(ValueError, match="2-D"):
        trainer.set_text_prototypes(torch.randn(96, 384, 1))


# ---------------------------------------------------------------------------
# build_text_prototypes: not random, region-major order (AC-1, AC-3 order).
# ---------------------------------------------------------------------------


def test_build_text_prototypes_random_fallback_is_random() -> None:
    protos, meta = build_text_prototypes(
        n_regions=3,
        n_categories=32,
        hidden_dim=768,
        seed=42,
        proto_source="random",
    )
    assert protos.shape == (96, 768)
    assert meta["proto_source"] == "random"
    expected = torch.randn(96, 768, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(protos, expected)


def test_build_text_prototypes_pastis_not_random(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inject a synthetic 18x384 "parquet" so the test never touches DVC/network.
    # Per-class distinct rows (tiled identity, trimmed to 384 columns).
    fake_proto = np.tile(np.eye(18, dtype=np.float32), (1, 22))[:, :384].copy()
    fake_class_ids = list(range(1, 19))

    def _fake_loader(path: Path | None = None) -> tuple[np.ndarray, list[int]]:
        return fake_proto, fake_class_ids

    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        _fake_loader,
    )
    protos, meta = build_text_prototypes(
        n_regions=3,
        n_categories=32,
        hidden_dim=768,
        seed=42,
        proto_source="pastis",
        cap_classes=_CAP_CLASSES_32,
    )
    # 384-dim tile (reprojected later by set_text_prototypes), region-major.
    assert protos.shape == (96, 384)
    assert meta["proto_source"] == "pastis_prototypes"
    assert meta["proto_proj"] == "ortho_384_768"
    assert meta["caveat"] == "ortho_proj_crude_approx"
    # NOT the legacy torch.randn(96, *) at the same seed.
    legacy = torch.randn(96, 384, generator=torch.Generator().manual_seed(42))
    assert not torch.allclose(protos, legacy)


def test_build_text_prototypes_region_major_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Distinguishable per-class prototypes: row r (class_id r+1) is one-hot at r.
    fake_proto = np.eye(18, dtype=np.float32)
    fake_class_ids = list(range(1, 19))

    def _fake_loader(path: Path | None = None) -> tuple[np.ndarray, list[int]]:
        return fake_proto, fake_class_ids

    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        _fake_loader,
    )
    protos, _ = build_text_prototypes(
        n_regions=3,
        n_categories=32,
        hidden_dim=768,
        seed=42,
        proto_source="pastis",
        cap_classes=_CAP_CLASSES_32,
    )
    arr = protos.numpy()
    mapping = load_cap_to_pastis()
    # Row (region * 32 + category) must hold the prototype of that category,
    # replicated identically across the 3 regions.
    for region in range(3):
        for cat in range(32):
            row = arr[region * 32 + cat]
            expected_class_id = mapping[_CAP_CLASSES_32[cat]]
            assert int(row[:18].argmax()) + 1 == expected_class_id
        # Region block 0 == block 1 == block 2 (tile replication).
    assert np.allclose(arr[0:32], arr[32:64])
    assert np.allclose(arr[0:32], arr[64:96])


def test_build_text_prototypes_falls_back_when_parquet_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raising_loader(path: Path | None = None) -> tuple[np.ndarray, list[int]]:
        raise FileNotFoundError("parquet not available (no DVC pull)")

    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        _raising_loader,
    )
    protos, meta = build_text_prototypes(
        n_regions=3,
        n_categories=32,
        hidden_dim=768,
        seed=42,
        proto_source="pastis",
        cap_classes=_CAP_CLASSES_32,
    )
    assert meta["proto_source"] == "random"
    assert protos.shape == (96, 768)


# ---------------------------------------------------------------------------
# Loss decreases with real (non-random) prototypes (AC-6 smoke).
# ---------------------------------------------------------------------------


def test_loss_cls_decreases_with_real_prototypes() -> None:
    # Synthetic but separable: 4 classes, prototypes = reprojected one-hots.
    torch.manual_seed(0)
    trainer = _FakeTrainer(hidden_size=64, seed=42)
    proto_4 = torch.eye(4, 32)  # (4, 32) one-hot per class in MiniLM-like space
    proto_64 = trainer._proto_to_clip_proj(proto_4)  # (4, 64) frozen ortho lift
    loss_fn = RegionCategoryAlignmentLoss(temperature=0.07, n_regions=1, n_categories=4)
    student = (torch.randn(16, 64) * 0.1).requires_grad_(True)
    region_ids = torch.zeros(16, dtype=torch.long)
    cat_ids = torch.repeat_interleave(torch.arange(4), 4)
    opt = torch.optim.Adam([student], lr=0.2)
    losses: list[float] = []
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(student, proto_64, region_ids, cat_ids)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_optimizer_param_count_unchanged_by_projection() -> None:
    # The orthogonal projection W is a plain attribute, never a Parameter, so it
    # is invisible to any optimizer built over the trainer's "parameters".
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    trainer.set_text_prototypes(torch.randn(96, 384))  # builds W
    w = trainer._proto_proj_w_384  # type: ignore[attr-defined]
    assert not isinstance(w, torch.nn.Parameter)
    assert w.requires_grad is False


# ---------------------------------------------------------------------------
# Optional: real US-033 parquet (skipped without DVC binary).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PARQUET.exists(), reason="US-033 parquet not pulled (DVC)")
def test_real_parquet_18x384_expands_to_96x768() -> None:
    from ml.features.phenology_class_prototypes import (
        load_class_prototype_embeddings,
    )

    proto_18, class_ids = load_class_prototype_embeddings(_PARQUET)
    assert proto_18.shape == (18, 384)
    proto_cap = expand_to_cap(proto_18, _CAP_CLASSES_32, pastis_class_ids=class_ids)
    tiled = np.tile(proto_cap, (3, 1))
    assert tiled.shape == (96, 384)
    trainer = _FakeTrainer(hidden_size=768, seed=42)
    trainer.set_text_prototypes(torch.from_numpy(tiled).float())
    assert trainer._text_prototypes.shape == (96, 768)
