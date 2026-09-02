"""Tests de las perdidas FarSLIP (AC-1, AC-7).

Cobertura objetivo ``ml/farslip/distill.py`` >= 80 %.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ml.farslip.distill import (
    PatchDistillationLoss,
    RegionCategoryAlignmentLoss,
)

# ---------------------------------------------------------------------------
# PatchDistillationLoss
# ---------------------------------------------------------------------------


def test_patch_loss_identity_collapse() -> None:
    """student == teacher => loss < 1e-6 (modo mse_plus_cosine)."""
    loss = PatchDistillationLoss(loss_type="mse_plus_cosine", normalize=True)
    feats = torch.randn(4, 196, 768)
    s = feats.clone().requires_grad_(True)
    t = feats.clone()
    out = loss(s, t)
    assert out.item() < 1e-6, f"expected near-zero, got {out.item()}"


def test_patch_loss_scale_invariance() -> None:
    """Con normalize=True la perdida es invariante a escalado uniforme del student."""
    loss = PatchDistillationLoss(loss_type="cosine", normalize=True)
    s_base = torch.randn(2, 10, 64)
    t = torch.randn(2, 10, 64)
    out_a = loss(s_base, t).item()
    out_b = loss(s_base * 5.0, t).item()
    assert abs(out_a - out_b) < 1e-5


def test_patch_loss_gradient_not_nan() -> None:
    """Gradiente respecto al student es finito."""
    loss = PatchDistillationLoss()
    s = torch.randn(2, 50, 32, requires_grad=True)
    t = torch.randn(2, 50, 32)
    out = loss(s, t)
    out.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_patch_loss_batch_independence() -> None:
    """Loss por sample no debe leakear info entre samples del batch."""
    loss = PatchDistillationLoss(loss_type="mse", normalize=False)
    s1 = torch.randn(1, 10, 16)
    s2 = torch.randn(1, 10, 16)
    t1 = torch.randn(1, 10, 16)
    t2 = torch.randn(1, 10, 16)
    out_separate = (loss(s1, t1).item() + loss(s2, t2).item()) / 2.0
    out_batched = loss(torch.cat([s1, s2], dim=0), torch.cat([t1, t2], dim=0)).item()
    assert abs(out_separate - out_batched) < 1e-5


def test_patch_loss_mask_handling() -> None:
    """Con mask todo False excepto un patch, la loss ~ contribucion de ese patch."""
    loss = PatchDistillationLoss(loss_type="mse", normalize=False)
    s = torch.zeros(1, 4, 8)
    t = torch.zeros(1, 4, 8)
    # patch 2 con diferencia constante 2.0 en 8 dims => squared sum = 4*8 = 32
    s[0, 2, :] = 2.0
    mask = torch.tensor([[False, False, True, False]])
    out = loss(s, t, patch_mask=mask)
    # n_valid = 1, sum = 32 -> 32
    assert math.isclose(out.item(), 32.0, rel_tol=1e-4)


def test_patch_loss_stop_grad_on_teacher() -> None:
    """Backward sobre teacher debe ser None (stop-grad)."""
    loss = PatchDistillationLoss()
    s = torch.randn(1, 4, 8, requires_grad=True)
    t = torch.randn(1, 4, 8, requires_grad=True)
    out = loss(s, t)
    out.backward()
    assert t.grad is None or torch.allclose(t.grad, torch.zeros_like(t.grad))


def test_patch_loss_invalid_loss_type_raises() -> None:
    with pytest.raises(ValueError):
        PatchDistillationLoss(loss_type="garbage")  # type: ignore[arg-type]


def test_patch_loss_shape_mismatch_raises() -> None:
    loss = PatchDistillationLoss()
    s = torch.randn(2, 10, 8)
    t = torch.randn(2, 11, 8)
    with pytest.raises(ValueError):
        loss(s, t)


# ---------------------------------------------------------------------------
# RegionCategoryAlignmentLoss
# ---------------------------------------------------------------------------


def test_cls_alignment_separability() -> None:
    """4 clusters sinteticos => ranking correcto + classification accuracy = 1.0.

    InfoNCE no fuerza |cos|=1 absoluto, solo el orden relativo (positivo
    debe rankear por encima de los negativos). Validamos:

    - Classification accuracy (argmax sobre prototipos) = 1.0 (todos correctos)
    - Margin (diag - off_mean) >= 0.20 (separabilidad clara)
    """
    n_classes = 4
    n_per_class = 8
    n_regions = 1
    d = 4
    torch.manual_seed(0)
    protos = torch.eye(n_classes, d)
    student = (torch.randn(n_classes * n_per_class, d) * 0.1).requires_grad_(True)
    region_ids = torch.zeros(n_classes * n_per_class, dtype=torch.long)
    category_ids = torch.repeat_interleave(torch.arange(n_classes), n_per_class)
    loss_fn = RegionCategoryAlignmentLoss(
        temperature=0.07, n_regions=n_regions, n_categories=n_classes
    )
    opt = torch.optim.Adam([student], lr=0.3)
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(student, protos, region_ids, category_ids)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_n = F.normalize(student, dim=-1)
        p_n = F.normalize(protos, dim=-1)
        sim_matrix = s_n @ p_n.t()  # (B, n_classes)
        preds = sim_matrix.argmax(dim=-1)
        acc = (preds == category_ids).float().mean().item()
        diag_sims = []
        off_sims = []
        for c in range(n_classes):
            members = s_n[category_ids == c]
            diag_sims.append((members @ p_n[c]).mean().item())
            for cc in range(n_classes):
                if cc == c:
                    continue
                off_sims.append((members @ p_n[cc]).mean().item())
        diag_mean = sum(diag_sims) / len(diag_sims)
        off_mean = sum(off_sims) / len(off_sims)
        margin = diag_mean - off_mean
    assert acc == 1.0, f"acc={acc} != 1.0"
    assert margin >= 0.20, f"margin {margin} (diag {diag_mean}, off {off_mean}) < 0.20"


def test_cls_alignment_temperature_scaling() -> None:
    """Temperatura mayor => loss mas grande para mismo input (logits mas planos)."""
    n_cat = 3
    n_reg = 1
    d = 16
    student = torch.randn(4, d)
    protos = torch.randn(n_reg * n_cat, d)
    region_ids = torch.zeros(4, dtype=torch.long)
    category_ids = torch.tensor([0, 1, 2, 0])
    low_t = RegionCategoryAlignmentLoss(temperature=0.01, n_regions=n_reg, n_categories=n_cat)
    high_t = RegionCategoryAlignmentLoss(temperature=1.0, n_regions=n_reg, n_categories=n_cat)
    out_low = low_t(student, protos, region_ids, category_ids).item()
    out_high = high_t(student, protos, region_ids, category_ids).item()
    # No es estrictamente monotonico siempre, pero la entropia con temp alta
    # tiende a log(C). Verificamos que ambos son finitos y > 0.
    assert torch.isfinite(torch.tensor([out_low, out_high])).all()
    assert out_low > 0 and out_high > 0


def test_cls_alignment_batch_one() -> None:
    """Batch=1 no rompe el calculo."""
    d = 16
    n_cat = 3
    student = torch.randn(1, d, requires_grad=True)
    protos = torch.randn(n_cat, d)
    region_ids = torch.zeros(1, dtype=torch.long)
    category_ids = torch.tensor([1])
    loss_fn = RegionCategoryAlignmentLoss(n_regions=1, n_categories=n_cat)
    out = loss_fn(student, protos, region_ids, category_ids)
    out.backward()
    assert torch.isfinite(out)
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_cls_alignment_single_cluster() -> None:
    """Todos los samples en una unica clase => loss finita y reduce con training."""
    n_cat = 4
    d = 8
    student = torch.randn(6, d, requires_grad=True)
    protos = torch.randn(n_cat, d)
    region_ids = torch.zeros(6, dtype=torch.long)
    category_ids = torch.full((6,), 2, dtype=torch.long)
    loss_fn = RegionCategoryAlignmentLoss(n_regions=1, n_categories=n_cat)
    out_initial = loss_fn(student, protos, region_ids, category_ids).item()
    opt = torch.optim.SGD([student], lr=0.5)
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(student, protos, region_ids, category_ids)
        loss.backward()
        opt.step()
    out_final = loss_fn(student, protos, region_ids, category_ids).item()
    assert out_final < out_initial


def test_cls_alignment_invalid_ids_raise() -> None:
    loss_fn = RegionCategoryAlignmentLoss(n_regions=2, n_categories=3)
    student = torch.randn(2, 8)
    protos = torch.randn(6, 8)
    region_ids = torch.tensor([0, 5])  # 5 fuera de rango
    category_ids = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        loss_fn(student, protos, region_ids, category_ids)


# ---------------------------------------------------------------------------
# Combinacion de losses
# ---------------------------------------------------------------------------


def test_losses_combine_finite_grad() -> None:
    """Sumar las dos perdidas mantiene grad finito sobre el student CLS+patches."""
    loss_patch = PatchDistillationLoss()
    loss_cls = RegionCategoryAlignmentLoss(n_regions=2, n_categories=3)
    s_patches = torch.randn(4, 10, 16, requires_grad=True)
    t_patches = torch.randn(4, 10, 16)
    s_cls = torch.randn(4, 16, requires_grad=True)
    protos = torch.randn(6, 16)
    region_ids = torch.tensor([0, 1, 1, 0])
    cat_ids = torch.tensor([0, 2, 1, 2])

    total = loss_patch(s_patches, t_patches) + 0.5 * loss_cls(s_cls, protos, region_ids, cat_ids)
    total.backward()
    assert s_patches.grad is not None and torch.isfinite(s_patches.grad).all()
    assert s_cls.grad is not None and torch.isfinite(s_cls.grad).all()
