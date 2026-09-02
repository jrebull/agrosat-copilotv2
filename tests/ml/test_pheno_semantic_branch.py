"""Tests de la rama semantica + loss contrastivo fenologico (US-025 Seccion A).

Cubren :func:`ml.models.pheno_semantic_branch.phenology_contrastive_loss` y
:class:`ml.models.pheno_semantic_branch.PhenoSemanticBranch` (Wen et al. 2025,
ec. 15-16). Los tests son deterministas (seed + ``torch.Generator``) y no tocan
red ni sentence-transformers: ``PhenoSemanticBranch`` se construye con un parquet
de prototipos sintetico en ``tmp_path`` (mismo esquema que el real, 384-dim), y
el loss se ejercita con prototipos fabricados a mano.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from ml.models.pheno_semantic_branch import (
    PhenoSemanticBranch,
    phenology_contrastive_loss,
)

_DIM = 384
_IGNORE_INDEX = 255


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Semilla global torch/numpy para reproducibilidad."""
    torch.manual_seed(0)
    np.random.seed(0)


def _orthonormal_prototypes(num_classes: int, dim: int = _DIM) -> torch.Tensor:
    """Prototipos casi-ortogonales y L2-normalizados ``(num_classes, dim)``.

    Usa los primeros ``num_classes`` vectores canonicos (one-hot) embebidos en
    ``dim``: maximamente separados, lo que da un caso de contraste limpio.
    """
    protos = torch.zeros(num_classes, dim)
    for c in range(num_classes):
        protos[c, c] = 1.0
    return protos


def _visual_proj_from_labels(
    labels_hw: torch.Tensor, prototypes: torch.Tensor, *, noise: float, dim: int = _DIM
) -> torch.Tensor:
    """Construye ``visual_proj (1,dim,H,W)`` cuyos pixeles tienden a su prototipo.

    Cada pixel valido toma el prototipo de su clase + ruido gaussiano escalado
    por ``noise``. Con ``noise`` bajo el pixel queda alineado con su clase (loss
    bajo); con ``noise`` alto se aleja (loss alto).
    """
    h, w = labels_hw.shape
    proj = torch.randn(dim, h, w) * noise
    for c in range(prototypes.shape[0]):
        mask = labels_hw == c
        if mask.any():
            # Suma el prototipo de la clase a cada pixel de esa clase.
            proj[:, mask] = proj[:, mask] + prototypes[c].unsqueeze(1)
    return proj.unsqueeze(0)


def _make_labels(num_classes: int, h: int = 8, w: int = 8) -> torch.Tensor:
    """Mascara ``(H,W)`` con varias clases presentes (contraste bien definido)."""
    labels = torch.zeros(h, w, dtype=torch.long)
    per = (h * w) // num_classes
    flat = labels.reshape(-1)
    for c in range(num_classes):
        flat[c * per : (c + 1) * per] = c
    return flat.reshape(h, w)


# ---------------------------------------------------------------------------
# phenology_contrastive_loss
# ---------------------------------------------------------------------------


def test_contrastive_loss_scalar_positive() -> None:
    """El loss contrastivo es un escalar finito y positivo."""
    num_classes = 4
    protos = _orthonormal_prototypes(num_classes)
    labels = _make_labels(num_classes)
    proj = _visual_proj_from_labels(labels, protos, noise=1.0)
    gen = torch.Generator().manual_seed(7)
    loss = phenology_contrastive_loss(
        proj, labels.unsqueeze(0), protos, ignore_index=_IGNORE_INDEX, generator=gen
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_contrastive_loss_backward_produces_grad() -> None:
    """El backward del loss produce gradiente finito en ``visual_proj``."""
    num_classes = 4
    protos = _orthonormal_prototypes(num_classes)
    labels = _make_labels(num_classes)
    proj = _visual_proj_from_labels(labels, protos, noise=1.0).requires_grad_(True)
    gen = torch.Generator().manual_seed(7)
    loss = phenology_contrastive_loss(proj, labels.unsqueeze(0), protos, generator=gen)
    loss.backward()
    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()
    assert proj.grad.abs().sum() > 0.0


def test_contrastive_loss_decreases_when_aligned() -> None:
    """Sanity de direccion: el loss BAJA si ``visual_proj`` se acerca a los prototipos.

    Mismo target y prototipos; solo cambia el ruido (alineacion). Con ruido bajo
    cada pixel queda casi sobre el prototipo de su clase -> contraste facil ->
    loss menor que con ruido alto.
    """
    num_classes = 5
    protos = _orthonormal_prototypes(num_classes)
    labels = _make_labels(num_classes).unsqueeze(0)

    proj_aligned = _visual_proj_from_labels(labels[0], protos, noise=0.05)
    proj_noisy = _visual_proj_from_labels(labels[0], protos, noise=5.0)

    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    loss_aligned = phenology_contrastive_loss(proj_aligned, labels, protos, generator=gen_a)
    loss_noisy = phenology_contrastive_loss(proj_noisy, labels, protos, generator=gen_b)

    assert loss_aligned.item() < loss_noisy.item()


def test_contrastive_loss_ignores_padding() -> None:
    """Pixeles ``ignore_index`` no rompen el loss (sigue finito y > 0)."""
    num_classes = 3
    protos = _orthonormal_prototypes(num_classes)
    labels = _make_labels(num_classes)
    labels[:4, :] = _IGNORE_INDEX  # mitad superior ignorada
    proj = _visual_proj_from_labels(torch.clamp(labels, 0, num_classes - 1), protos, noise=1.0)
    gen = torch.Generator().manual_seed(7)
    loss = phenology_contrastive_loss(
        proj, labels.unsqueeze(0), protos, ignore_index=_IGNORE_INDEX, generator=gen
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_contrastive_loss_all_ignored_returns_zero() -> None:
    """Sin pixeles validos el loss es ``0.0`` (con grafo, no NaN)."""
    num_classes = 3
    protos = _orthonormal_prototypes(num_classes)
    h = w = 8
    labels = torch.full((1, h, w), _IGNORE_INDEX, dtype=torch.long)
    proj = torch.randn(1, _DIM, h, w, requires_grad=True)
    loss = phenology_contrastive_loss(proj, labels, protos, ignore_index=_IGNORE_INDEX)
    assert loss.item() == 0.0
    # Debe seguir teniendo grafo (no escalar pelado).
    loss.backward()


def test_contrastive_loss_single_class_returns_zero() -> None:
    """Con una sola clase presente el contraste es indefinido -> ``0.0``."""
    num_classes = 4
    protos = _orthonormal_prototypes(num_classes)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)  # todo clase 0
    proj = torch.randn(1, _DIM, 8, 8, requires_grad=True)
    loss = phenology_contrastive_loss(proj, labels, protos)
    assert loss.item() == 0.0


def test_contrastive_loss_rejects_non_4d() -> None:
    """``visual_proj`` que no es ``(B,D,H,W)`` lanza ``ValueError``."""
    protos = _orthonormal_prototypes(3)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    bad = torch.randn(_DIM, 8, 8)  # 3-D
    with pytest.raises(ValueError, match="B, D, H, W"):
        phenology_contrastive_loss(bad, labels, protos)


def test_contrastive_loss_subsample_deterministic() -> None:
    """Con el mismo ``generator`` el submuestreo y el loss son reproducibles."""
    num_classes = 4
    protos = _orthonormal_prototypes(num_classes)
    # 64x64 -> 4096 validos == max_pixels default, fuerza la rama de submuestreo
    # con un patch mas grande.
    labels = _make_labels(num_classes, h=80, w=80).unsqueeze(0)
    proj = _visual_proj_from_labels(labels[0], protos, noise=1.0)
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)
    loss_a = phenology_contrastive_loss(proj, labels, protos, max_pixels=512, generator=gen_a)
    loss_b = phenology_contrastive_loss(proj, labels, protos, max_pixels=512, generator=gen_b)
    assert loss_a.item() == pytest.approx(loss_b.item())


# ---------------------------------------------------------------------------
# PhenoSemanticBranch
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prototype_parquet(tmp_path: Path) -> Path:
    """Parquet de prototipos sintetico (6 clases, 384-dim) en el esquema real.

    Mismo esquema que ``data/features/phenology_class_prototypes_pastis.parquet``:
    ``class_id, class_name, emb_000..emb_383`` (sin red, sin Gemini).
    """
    num_classes = 6
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((num_classes, _DIM)).astype(np.float32)
    rows = []
    for c in range(num_classes):
        row: dict[str, object] = {
            "class_id": c + 1,
            "class_name": f"class_{c + 1}",
        }
        for j in range(_DIM):
            row[f"emb_{j:03d}"] = float(emb[c, j])
        rows.append(row)
    out = tmp_path / "protos.parquet"
    pl.DataFrame(rows).write_parquet(out)
    return out


def test_branch_get_prototypes_shape_and_l2norm(synthetic_prototype_parquet: Path) -> None:
    """``get_class_prototypes`` devuelve ``(K,semantic_dim)`` L2-normalizado."""
    branch = PhenoSemanticBranch(semantic_dim=_DIM, prototype_path=synthetic_prototype_parquet)
    assert branch.num_classes == 6
    protos = branch.get_class_prototypes()
    assert protos.shape == (6, _DIM)
    norms = protos.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_branch_prototypes_frozen_as_buffer(synthetic_prototype_parquet: Path) -> None:
    """Con ``freeze_prototypes=True`` los prototipos crudos son buffer (no Parameter).

    Ancla fija del setup oficial: solo la proyeccion es aprendible.
    """
    branch = PhenoSemanticBranch(
        semantic_dim=_DIM, prototype_path=synthetic_prototype_parquet, freeze_prototypes=True
    )
    buffer_names = {name for name, _ in branch.named_buffers()}
    param_names = {name for name, _ in branch.named_parameters()}
    assert "raw_prototypes" in buffer_names
    assert "raw_prototypes" not in param_names
    # La proyeccion si es aprendible.
    assert any(name.startswith("proj") for name in param_names)


def test_branch_prototypes_trainable_when_unfrozen(synthetic_prototype_parquet: Path) -> None:
    """Con ``freeze_prototypes=False`` los prototipos crudos son ``Parameter``."""
    branch = PhenoSemanticBranch(
        semantic_dim=_DIM, prototype_path=synthetic_prototype_parquet, freeze_prototypes=False
    )
    param_names = {name for name, _ in branch.named_parameters()}
    assert "raw_prototypes" in param_names


def test_branch_loss_uses_detached_prototypes_as_anchors(
    synthetic_prototype_parquet: Path,
) -> None:
    """El setup oficial pasa los prototipos fijos (detached) como anclas.

    Se replica el flujo: prototipos de la rama -> ``.detach()`` -> al loss. El
    gradiente fluye a ``visual_proj`` pero NO a los prototipos detached.
    """
    branch = PhenoSemanticBranch(semantic_dim=_DIM, prototype_path=synthetic_prototype_parquet)
    protos = branch.get_class_prototypes().detach()
    assert not protos.requires_grad

    labels = _make_labels(branch.num_classes).unsqueeze(0)
    proj = _visual_proj_from_labels(labels[0], protos, noise=1.0).requires_grad_(True)
    gen = torch.Generator().manual_seed(3)
    loss = phenology_contrastive_loss(proj, labels, protos, generator=gen)
    loss.backward()
    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()


def test_branch_rejects_wrong_prototype_dim(tmp_path: Path) -> None:
    """Un parquet sin las 384 columnas ``emb_*`` no carga (contrato 384-dim).

    ``load_class_prototype_embeddings`` selecciona siempre ``emb_000..emb_383``;
    si faltan, Polars lanza ``ColumnNotFoundError`` (subclase de ``Exception``)
    antes del guard explicito de 384-dim de ``PhenoSemanticBranch``. Cualquiera
    de las dos rutas debe impedir construir una rama con dimension distinta a
    384, que es el invariante que protege este test.
    """
    rows = []
    for c in range(3):
        row: dict[str, object] = {"class_id": c + 1, "class_name": f"c{c}"}
        for j in range(10):  # solo 10 dims -> faltan emb_010..emb_383
            row[f"emb_{j:03d}"] = 0.1
        rows.append(row)
    bad = tmp_path / "bad.parquet"
    pl.DataFrame(rows).write_parquet(bad)
    with pytest.raises(Exception, match=r"emb_|384"):
        PhenoSemanticBranch(prototype_path=bad)


# ---------------------------------------------------------------------------
# Integracion con el parquet real (18 prototipos)
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_PROTO = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"


@pytest.mark.skipif(
    not _REAL_PROTO.exists(),
    reason="phenology_class_prototypes_pastis.parquet no presente.",
)
def test_branch_loads_real_18_prototypes() -> None:
    """La rama carga los 18 prototipos reales (384-dim) sin red."""
    branch = PhenoSemanticBranch(prototype_path=_REAL_PROTO)
    assert branch.num_classes == 18
    protos = branch.get_class_prototypes()
    assert protos.shape == (18, _DIM)
    assert torch.isfinite(protos).all()
