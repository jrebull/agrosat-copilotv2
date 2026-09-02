"""Tests golden de las perdidas FarSLIP faithful (US-036-a v2, T3).

Verifica las ecuaciones del paper Li et al. 2025 (arXiv:2511.14901) transcritas
en ``docs/us-planning/us-036-a-v2-faithful.md`` Seccion 0:

1. MPCL equivalencia con ``F.cross_entropy`` cuando ``|P(i)|=1`` (prueba de
   fidelidad: MPCL generaliza la v1, no la contradice).
2. MPCL multi-positivo: el gradiente del ancla empuja hacia el centroide de SUS
   positivos (caso analitico).
3. Estabilidad: positivos vacios (categoria unica) y temperaturas extremas sin
   ``NaN``.
4. ``L_glo``: InfoNCE simetrico imagen<->texto, diagonal positiva, shape.
5. Combinacion ``L_total = L_glo + lambda_loc * L_loc`` (ablacion ``lambda_loc=0``).

Logica pura: cero GPU/red/dataset.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ml.farslip.mpcl_loss import (
    DEFAULT_LAMBDA_LOC,
    DEFAULT_TEMPERATURE,
    GlobalImageTextLoss,
    MultiPositiveRegionCategoryLoss,
    _directional_mpcl,
    combine_losses,
)

# ---------------------------------------------------------------------------
# 1. MPCL equivalencia con cross_entropy (golden de fidelidad)
# ---------------------------------------------------------------------------


def test_mpcl_equals_cross_entropy_when_single_positive() -> None:
    """``|P(i)|=1`` => MPCL == ``F.cross_entropy(logits, cat_ids)`` a atol 1e-5.

    Caso golden: ``R == C`` con biyeccion region<->categoria y el prototipo de la
    categoria ``j`` ES el embedding de su unica region ``j``. La matriz de logits
    es entonces simetrica, asi que ambas direcciones (R->C y C->R) colapsan a la
    MISMA cross-entropy direccional == la perdida v1
    (:class:`RegionCategoryAlignmentLoss`). Esto PRUEBA que MPCL generaliza la v1.
    """
    torch.manual_seed(0)
    n_categories = 6
    dim = 16
    region_visual = torch.randn(n_categories, dim)
    # prototipo de categoria j = la region j (1 region por categoria).
    category_text = region_visual.clone()
    region_cat_ids = torch.arange(n_categories)

    loss_fn = MultiPositiveRegionCategoryLoss(temperature=DEFAULT_TEMPERATURE)
    mpcl = loss_fn(region_visual, category_text, region_cat_ids)

    # Referencia v1: F.cross_entropy sobre los mismos logits coseno/tau.
    region_n = F.normalize(region_visual, p=2, dim=-1)
    category_n = F.normalize(category_text, p=2, dim=-1)
    logits = region_n @ category_n.t() / DEFAULT_TEMPERATURE
    ce = F.cross_entropy(logits, region_cat_ids)

    assert torch.allclose(mpcl, ce, atol=1e-5), f"MPCL={mpcl.item()} CE={ce.item()}"


def test_mpcl_equivalence_holds_for_distinct_categories_random_seed() -> None:
    """La equivalencia con CE se mantiene en varias semillas (no es casualidad)."""
    for seed in (1, 7, 42, 123):
        torch.manual_seed(seed)
        c = 5
        d = 8
        region_visual = torch.randn(c, d)
        category_text = region_visual.clone()
        cat_ids = torch.arange(c)
        mpcl = MultiPositiveRegionCategoryLoss()(region_visual, category_text, cat_ids)
        region_n = F.normalize(region_visual, p=2, dim=-1)
        logits = region_n @ region_n.t() / DEFAULT_TEMPERATURE
        ce = F.cross_entropy(logits, cat_ids)
        assert torch.allclose(mpcl, ce, atol=1e-5), f"seed={seed}"


# ---------------------------------------------------------------------------
# 2. MPCL multi-positivo (golden analitico del gradiente)
# ---------------------------------------------------------------------------


def test_mpcl_multipositive_gradient_points_to_positive_centroid() -> None:
    """2 categorias x 3 regiones: el gradiente del ancla empuja a SUS positivos.

    Caso analitico: si el ancla aprende (un paso de gradiente descendente sobre
    ``region_visual``), su embedding se mueve hacia el prototipo de su categoria
    (que coincide con el centroide de SUS positivos), aumentando la similitud con
    el positivo y reduciendola con el negativo. Verificamos el SIGNO del cambio.
    """
    torch.manual_seed(0)
    dim = 8
    # Dos prototipos de categoria ortogonales.
    proto_a = torch.zeros(dim)
    proto_a[0] = 1.0
    proto_b = torch.zeros(dim)
    proto_b[1] = 1.0
    category_text = torch.stack([proto_a, proto_b])  # (2, D)

    # 3 regiones cat 0 + 3 regiones cat 1, inicializadas con ruido pequeno.
    region_visual = (torch.randn(6, dim) * 0.1).requires_grad_(True)
    region_cat_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    loss_fn = MultiPositiveRegionCategoryLoss(temperature=0.5)

    sim_pos_before = []
    sim_neg_before = []
    with torch.no_grad():
        rn = F.normalize(region_visual, p=2, dim=-1)
        cn = F.normalize(category_text, p=2, dim=-1)
        sims = rn @ cn.t()  # (6, 2)
        for i in range(6):
            sim_pos_before.append(sims[i, region_cat_ids[i]].item())
            sim_neg_before.append(sims[i, 1 - region_cat_ids[i]].item())

    loss = loss_fn(region_visual, category_text, region_cat_ids)
    loss.backward()
    assert region_visual.grad is not None
    assert torch.isfinite(region_visual.grad).all()

    with torch.no_grad():
        updated = region_visual - 0.5 * region_visual.grad
        rn = F.normalize(updated, p=2, dim=-1)
        cn = F.normalize(category_text, p=2, dim=-1)
        sims = rn @ cn.t()
        for i in range(6):
            sim_pos_after = sims[i, region_cat_ids[i]].item()
            sim_neg_after = sims[i, 1 - region_cat_ids[i]].item()
            # Tras el paso: mas cerca del positivo, mas lejos del negativo.
            assert sim_pos_after > sim_pos_before[i], (
                f"region {i}: pos sim no aumento ({sim_pos_before[i]} -> {sim_pos_after})"
            )
            assert sim_neg_after < sim_neg_before[i], (
                f"region {i}: neg sim no disminuyo ({sim_neg_before[i]} -> {sim_neg_after})"
            )


def test_mpcl_multipositive_groups_same_category() -> None:
    """Optimizar MPCL agrupa regiones de la misma categoria (clasificacion 100%)."""
    torch.manual_seed(0)
    dim = 8
    n_cat = 3
    per_cat = 4
    category_text = torch.eye(n_cat, dim)
    region_visual = (torch.randn(n_cat * per_cat, dim) * 0.1).requires_grad_(True)
    region_cat_ids = torch.repeat_interleave(torch.arange(n_cat), per_cat)
    loss_fn = MultiPositiveRegionCategoryLoss(temperature=0.1)
    opt = torch.optim.Adam([region_visual], lr=0.2)
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(region_visual, category_text, region_cat_ids)
        loss.backward()
        opt.step()
    with torch.no_grad():
        rn = F.normalize(region_visual, p=2, dim=-1)
        cn = F.normalize(category_text, p=2, dim=-1)
        preds = (rn @ cn.t()).argmax(dim=-1)
        acc = (preds == region_cat_ids).float().mean().item()
    assert acc == 1.0, f"acc={acc} tras optimizar MPCL multi-positivo"


def test_mpcl_symmetric_directions_present() -> None:
    """La perdida total es la media de L_{R->C} y L_{C->R} (no solo una direccion).

    Construye un caso asimetrico (mas regiones que categorias) donde la media
    simetrica difiere de cualquiera de las dos direcciones por separado: prueba
    que ambos terminos se computan y se promedian.
    """
    torch.manual_seed(3)
    dim = 8
    region_visual = torch.randn(6, dim)
    category_text = torch.randn(2, dim)
    region_cat_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    total = MultiPositiveRegionCategoryLoss()(region_visual, category_text, region_cat_ids)
    # Reconstruimos las dos direcciones manualmente y verificamos la media.
    rn = F.normalize(region_visual, p=2, dim=-1)
    cn = F.normalize(category_text, p=2, dim=-1)
    logits = rn @ cn.t() / DEFAULT_TEMPERATURE  # (6, 2)
    # R->C: positivo = columna de su categoria (|P(i)|=1 sobre el eje texto).
    l_rc = F.cross_entropy(logits, region_cat_ids)
    # C->R: positivo = todas las regiones de la categoria (multi-positivo).
    log_prob_cr = F.log_softmax(logits.t(), dim=1)  # (2, 6)
    pos_cr = torch.zeros(2, 6)
    pos_cr[0, :3] = 1.0
    pos_cr[1, 3:] = 1.0
    l_cr = (-(log_prob_cr * pos_cr).sum(dim=1) / pos_cr.sum(dim=1)).mean()
    expected = 0.5 * (l_rc + l_cr)
    assert torch.allclose(total, expected, atol=1e-5), (
        f"total={total.item()} expected={expected.item()}"
    )


# ---------------------------------------------------------------------------
# 3. Estabilidad: positivos vacios, temperaturas extremas
# ---------------------------------------------------------------------------


def test_mpcl_single_category_no_other_positive_no_nan() -> None:
    """Categoria que aparece 1 sola vez (sin positivos extra) no produce NaN.

    En L_{R->C} el unico positivo es su columna de texto (``|P|=1``, finito). En
    L_{C->R} las categorias ausentes del batch no tienen regiones positivas y se
    excluyen del promedio: la perdida es finita y sin ``NaN``.
    """
    torch.manual_seed(0)
    dim = 8
    n_categories = 5
    # 3 regiones, categorias distintas: ninguna comparte categoria.
    region_visual = torch.randn(3, dim, requires_grad=True)
    category_text = torch.randn(n_categories, dim)
    region_cat_ids = torch.tensor([0, 2, 4])
    loss = MultiPositiveRegionCategoryLoss()(region_visual, category_text, region_cat_ids)
    assert torch.isfinite(loss), f"loss no finita: {loss}"
    loss.backward()
    assert region_visual.grad is not None
    assert torch.isfinite(region_visual.grad).all()


def test_mpcl_extreme_temperatures_finite() -> None:
    """Temperaturas extremas (muy baja y alta) dan perdida finita sin NaN."""
    torch.manual_seed(0)
    dim = 8
    region_visual = torch.randn(6, dim)
    category_text = torch.randn(2, dim)
    region_cat_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    for tau in (1e-3, 0.07, 10.0, 100.0):
        loss = MultiPositiveRegionCategoryLoss(temperature=tau)(
            region_visual, category_text, region_cat_ids
        )
        assert torch.isfinite(loss), f"tau={tau} -> loss={loss}"
        assert loss.item() >= 0.0, f"tau={tau} -> loss negativa {loss}"


def test_mpcl_invalid_temperature_raises() -> None:
    for bad in (0.0, -0.07, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            MultiPositiveRegionCategoryLoss(temperature=bad)


def test_mpcl_out_of_range_category_raises() -> None:
    region_visual = torch.randn(3, 8)
    category_text = torch.randn(2, 8)
    bad_ids = torch.tensor([0, 1, 2])  # 2 fuera de rango [0,2)
    with pytest.raises(ValueError):
        MultiPositiveRegionCategoryLoss()(region_visual, category_text, bad_ids)


def test_mpcl_shape_mismatch_raises() -> None:
    loss_fn = MultiPositiveRegionCategoryLoss()
    with pytest.raises(ValueError):
        loss_fn(torch.randn(3, 8), torch.randn(2, 16), torch.tensor([0, 1, 0]))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(3, 8), torch.randn(2, 8), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(3), torch.randn(2, 8), torch.tensor([0, 1, 0]))
    # category_text must be 2-D as well.
    with pytest.raises(ValueError):
        loss_fn(torch.randn(3, 8), torch.randn(8), torch.tensor([0, 1, 0]))


def test_directional_mpcl_all_anchors_without_positives_returns_zero() -> None:
    """``_directional_mpcl`` con mascara totalmente vacia -> 0.0 sin NaN.

    Cubre el caso degenerado interno (ningun ancla tiene positivos): el helper
    mantiene el grafo de gradiente y devuelve 0.0 (no ``NaN`` por 0/0).
    """
    logits = torch.randn(4, 5, requires_grad=True)
    empty_mask = torch.zeros(4, 5, dtype=torch.bool)
    out = _directional_mpcl(logits, empty_mask)
    assert out.item() == 0.0
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mpcl_empty_batch_returns_zero() -> None:
    loss = MultiPositiveRegionCategoryLoss()(
        torch.randn(0, 8, requires_grad=True), torch.randn(2, 8), torch.zeros(0).long()
    )
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 4. L_glo: InfoNCE simetrico imagen<->texto
# ---------------------------------------------------------------------------


def test_global_image_text_loss_identical_diagonal_minimal() -> None:
    """CLS imagen == CLS caption (diagonal alineada) => perdida minima.

    Con pares identicos y muy ortogonales entre si, la similitud diagonal domina
    y la InfoNCE simetrica tiende a ~0.
    """
    image_cls = torch.eye(6, 32)  # filas ortonormales => diagonal perfecta
    caption_cls = image_cls.clone()
    loss = GlobalImageTextLoss(temperature=0.01)(image_cls, caption_cls)
    assert loss.item() < 1e-3, f"esperado ~0, got {loss.item()}"


def test_global_image_text_loss_symmetric_value() -> None:
    """``L_glo`` == media de las dos cross-entropies direccionales (I->T, T->I)."""
    torch.manual_seed(0)
    image_cls = torch.randn(5, 16)
    caption_cls = torch.randn(5, 16)
    loss = GlobalImageTextLoss(temperature=DEFAULT_TEMPERATURE)(image_cls, caption_cls)
    image_n = F.normalize(image_cls, p=2, dim=-1)
    caption_n = F.normalize(caption_cls, p=2, dim=-1)
    logits = image_n @ caption_n.t() / DEFAULT_TEMPERATURE
    targets = torch.arange(5)
    expected = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_global_image_text_loss_shape_and_grad() -> None:
    """Salida escalar, gradiente finito sobre la imagen, texto detached."""
    image_cls = torch.randn(4, 16, requires_grad=True)
    caption_cls = torch.randn(4, 16, requires_grad=True)
    loss = GlobalImageTextLoss()(image_cls, caption_cls)
    assert loss.dim() == 0
    loss.backward()
    assert image_cls.grad is not None and torch.isfinite(image_cls.grad).all()
    # caption esta detached internamente: sin gradiente.
    assert caption_cls.grad is None


def test_global_image_text_loss_swaps_negatives() -> None:
    """Diagonal alineada da menor perdida que la diagonal permutada (negativos)."""
    torch.manual_seed(0)
    image_cls = torch.randn(6, 16)
    caption_cls = image_cls.clone()
    aligned = GlobalImageTextLoss(temperature=0.05)(image_cls, caption_cls)
    permuted = caption_cls[torch.tensor([1, 2, 3, 4, 5, 0])]
    misaligned = GlobalImageTextLoss(temperature=0.05)(image_cls, permuted)
    assert aligned.item() < misaligned.item()


def test_global_image_text_loss_invalid_shapes_raise() -> None:
    loss_fn = GlobalImageTextLoss()
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 16), torch.randn(5, 16))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4), torch.randn(4))


def test_global_image_text_loss_empty_batch_returns_zero() -> None:
    loss = GlobalImageTextLoss()(torch.randn(0, 16, requires_grad=True), torch.randn(0, 16))
    assert loss.item() == 0.0


def test_global_image_text_loss_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError):
        GlobalImageTextLoss(temperature=-1.0)


# ---------------------------------------------------------------------------
# 5. Combinacion L_total = L_glo + lambda_loc * L_loc
# ---------------------------------------------------------------------------


def test_combine_losses_default_lambda() -> None:
    """``lambda_loc=1.0`` (default) => total == suma simple."""
    loss_glo = torch.tensor(2.0, requires_grad=True)
    loss_loc = torch.tensor(3.0, requires_grad=True)
    total = combine_losses(loss_glo, loss_loc)
    assert math.isclose(total.item(), 5.0)
    assert math.isclose(DEFAULT_LAMBDA_LOC, 1.0)


def test_combine_losses_lambda_zero_ablation() -> None:
    """``lambda_loc=0`` => total == L_glo EXACTO (ablacion)."""
    loss_glo = torch.tensor(2.5, requires_grad=True)
    loss_loc = torch.tensor(99.0, requires_grad=True)
    total = combine_losses(loss_glo, loss_loc, lambda_loc=0.0)
    assert total.item() == loss_glo.item()


def test_combine_losses_weighted() -> None:
    total = combine_losses(torch.tensor(1.0), torch.tensor(4.0), lambda_loc=0.5)
    assert math.isclose(total.item(), 3.0)


def test_combine_losses_keeps_grad() -> None:
    loss_glo = torch.tensor(2.0, requires_grad=True)
    loss_loc = torch.tensor(3.0, requires_grad=True)
    total = combine_losses(loss_glo, loss_loc, lambda_loc=0.5)
    total.backward()
    assert loss_glo.grad is not None and math.isclose(loss_glo.grad.item(), 1.0)
    assert loss_loc.grad is not None and math.isclose(loss_loc.grad.item(), 0.5)


def test_combine_losses_negative_lambda_raises() -> None:
    with pytest.raises(ValueError):
        combine_losses(torch.tensor(1.0), torch.tensor(1.0), lambda_loc=-0.1)


def test_combine_losses_end_to_end_with_real_losses() -> None:
    """Integracion: L_glo + L_loc reales se combinan y son diferenciables."""
    torch.manual_seed(0)
    dim = 16
    image_cls = torch.randn(4, dim, requires_grad=True)
    caption_cls = torch.randn(4, dim)
    region_visual = torch.randn(6, dim, requires_grad=True)
    category_text = torch.randn(3, dim)
    region_cat_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    l_glo = GlobalImageTextLoss()(image_cls, caption_cls)
    l_loc = MultiPositiveRegionCategoryLoss()(region_visual, category_text, region_cat_ids)
    total = combine_losses(l_glo, l_loc, lambda_loc=1.0)
    total.backward()
    assert torch.isfinite(total)
    assert image_cls.grad is not None and torch.isfinite(image_cls.grad).all()
    assert region_visual.grad is not None and torch.isfinite(region_visual.grad).all()


def test_mpcl_uniform_weights_equal_unweighted() -> None:
    """class_weights uniformes == sin pesos (no altera el contrato base)."""
    torch.manual_seed(1)
    dim = 16
    region_visual = torch.randn(6, dim, requires_grad=True)
    category_text = torch.randn(3, dim)
    region_cat_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    base = MultiPositiveRegionCategoryLoss()(region_visual, category_text, region_cat_ids)
    weighted = MultiPositiveRegionCategoryLoss(class_weights=torch.ones(3))(
        region_visual, category_text, region_cat_ids
    )
    assert torch.allclose(base, weighted, atol=1e-6)


def test_mpcl_class_weights_upweight_rare_class() -> None:
    """Subir el peso de una clase rara cambia la loss y mantiene gradiente finito."""
    torch.manual_seed(2)
    dim = 16
    region_visual = torch.randn(6, dim, requires_grad=True)
    category_text = torch.randn(3, dim)
    # clase 2 es rara (1 region) vs clases 0,1 frecuentes.
    region_cat_ids = torch.tensor([0, 0, 1, 1, 1, 2])
    plain = MultiPositiveRegionCategoryLoss()(region_visual, category_text, region_cat_ids)
    rare_up = MultiPositiveRegionCategoryLoss(class_weights=torch.tensor([1.0, 1.0, 10.0]))(
        region_visual, category_text, region_cat_ids
    )
    assert not torch.allclose(plain, rare_up, atol=1e-4)
    rare_up.backward()
    assert region_visual.grad is not None and torch.isfinite(region_visual.grad).all()
