"""Tests de las factories de segmentacion DeepLabv3+ y TSViT (US-025).

Corren en CPU con batch chico y tensores aleatorios deterministas (seed). Para
DeepLabv3+ se usa ``encoder_weights=None`` por defecto: evita la descarga HF de
los pesos ImageNet de ``tu-mobilenetv3_large_100`` (~21 MB) en la primera
ejecucion, que rompe en CI sin red. El forward shape se prueba sin pesos
preentrenados (la arquitectura no cambia). Un test ``@pytest.mark.slow`` opt-in
ejercita la ruta ImageNet si hay red.
"""

from __future__ import annotations

import pytest
import torch

from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet, build_dice_ce_loss
from ml.models.tsvit_wrapper import TSViT, build_tsvit

_IGNORE_INDEX = 255


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Fija la semilla global de torch para reproducibilidad de los forwards."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# DeepLabv3+ MobileNetV3
# ---------------------------------------------------------------------------


def test_deeplabv3plus_forward_shape() -> None:
    """Forward ``(2,10,128,128) -> (2,18,128,128)`` sin NaN (CPU, sin pesos HF)."""
    model = build_deeplabv3plus_mobilenet(in_channels=10, classes=18, encoder_weights=None).eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 18, 128, 128)
    assert torch.isfinite(logits).all()


def test_deeplabv3plus_hcat6_classes() -> None:
    """Con ``classes=6`` (HCAT) la salida tiene 6 canales."""
    model = build_deeplabv3plus_mobilenet(in_channels=10, classes=6, encoder_weights=None).eval()
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 6, 128, 128)


def test_deeplabv3plus_invalid_atrous_rates_raises() -> None:
    """``atrous_rates`` con != 3 valores lanza ``ValueError``."""
    with pytest.raises(ValueError, match="atrous_rates"):
        build_deeplabv3plus_mobilenet(atrous_rates=(6, 12), encoder_weights=None)  # type: ignore[arg-type]


def test_deeplabv3plus_backward_produces_grad() -> None:
    """Un backward sobre la perdida produce gradiente finito en los parametros."""
    model = build_deeplabv3plus_mobilenet(in_channels=10, classes=18, encoder_weights=None).train()
    loss_fn = build_dice_ce_loss(ignore_index=_IGNORE_INDEX, n_classes=18)
    x = torch.randn(2, 10, 64, 64)
    target = torch.randint(0, 18, (2, 64, 64))
    logits = model(x)
    loss = loss_fn(logits, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "el backward deberia poblar gradientes"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.slow
def test_deeplabv3plus_imagenet_weights_optin() -> None:
    """Ruta ImageNet (descarga HF ~21MB). Opt-in: ``-m slow`` con red."""
    model = build_deeplabv3plus_mobilenet(
        in_channels=10, classes=18, encoder_weights="imagenet"
    ).eval()
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 18, 128, 128)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# build_dice_ce_loss
# ---------------------------------------------------------------------------


def test_dice_ce_loss_scalar_positive() -> None:
    """La perdida combinada es un escalar finito y positivo."""
    loss_fn = build_dice_ce_loss(ignore_index=_IGNORE_INDEX, n_classes=18)
    logits = torch.randn(2, 18, 32, 32)
    target = torch.randint(0, 18, (2, 32, 32))
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_dice_ce_loss_respects_ignore_index() -> None:
    """Los pixeles ``ignore_index`` no rompen la perdida (no NaN) y se excluyen.

    Mitad de la mascara en ``ignore_index``: la perdida sigue siendo finita y la
    parte ignorada no contribuye (no produce NaN aunque sea > n_classes).
    """
    loss_fn = build_dice_ce_loss(ignore_index=_IGNORE_INDEX, n_classes=18)
    logits = torch.randn(2, 18, 16, 16)
    target = torch.randint(0, 18, (2, 16, 16))
    target[:, :8, :] = _IGNORE_INDEX
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_dice_ce_loss_all_ignored_is_finite() -> None:
    """Una mascara enteramente ignorada no produce NaN."""
    loss_fn = build_dice_ce_loss(ignore_index=_IGNORE_INDEX, n_classes=18)
    logits = torch.randn(1, 18, 8, 8)
    target = torch.full((1, 8, 8), _IGNORE_INDEX, dtype=torch.long)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)


def test_dice_ce_loss_class_weights_validation() -> None:
    """``class_weights`` de longitud incorrecta lanza ``ValueError``."""
    with pytest.raises(ValueError, match="class_weights"):
        build_dice_ce_loss(n_classes=18, class_weights=[1.0, 2.0])


def test_dice_ce_loss_class_weights_accepted() -> None:
    """``class_weights`` de longitud correcta se acepta y produce escalar finito."""
    weights = [1.0] * 18
    loss_fn = build_dice_ce_loss(n_classes=18, class_weights=weights)
    logits = torch.randn(1, 18, 16, 16)
    target = torch.randint(0, 18, (1, 16, 16))
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TSViT
# ---------------------------------------------------------------------------


def _build_small_tsvit(num_classes: int = 18) -> TSViT:
    """TSViT diminuto para tests rapidos en CPU (dim/depth reducidos)."""
    return TSViT(
        num_classes=num_classes,
        n_timesteps=10,
        img_size=128,
        in_channels=10,
        patch_size=16,  # 8x8 = 64 tokens espaciales (mas chico que el default)
        dim=32,
        depth_temporal=1,
        depth_spatial=1,
        heads=2,
        dim_head=16,
        semantic_dim=384,
    )


def test_tsvit_forward_shape() -> None:
    """Forward ``(2,10,10,128,128) + doy -> (2,18,128,128)`` sin NaN."""
    model = _build_small_tsvit().eval()
    x = torch.randn(2, 10, 10, 128, 128)
    doy = torch.randint(1, 366, (2, 10))
    with torch.no_grad():
        logits = model(x, doy)
    assert logits.shape == (2, 18, 128, 128)
    assert torch.isfinite(logits).all()


def test_tsvit_forward_without_doy() -> None:
    """Sin ``doy`` cae al PE temporal ordinal y produce la misma forma."""
    model = _build_small_tsvit().eval()
    x = torch.randn(1, 10, 10, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 18, 128, 128)
    assert torch.isfinite(logits).all()


def test_tsvit_returns_visual_proj() -> None:
    """``return_visual_proj=True`` devuelve ``(logits, (B,384,128,128))``."""
    model = _build_small_tsvit().eval()
    x = torch.randn(2, 10, 10, 128, 128)
    doy = torch.randint(1, 366, (2, 10))
    with torch.no_grad():
        logits, visual_proj = model(x, doy, return_visual_proj=True)
    assert logits.shape == (2, 18, 128, 128)
    assert visual_proj.shape == (2, 384, 128, 128)
    assert torch.isfinite(visual_proj).all()


def test_tsvit_factory_defaults() -> None:
    """``build_tsvit`` produce un TSViT con la salida densa esperada."""
    model = build_tsvit(
        num_classes=18, n_timesteps=4, img_size=64, dim=32, depth_temporal=1, depth_spatial=1
    ).eval()
    x = torch.randn(1, 4, 10, 64, 64)
    doy = torch.randint(1, 366, (1, 4))
    with torch.no_grad():
        logits = model(x, doy)
    assert logits.shape == (1, 18, 64, 64)


def test_tsvit_invalid_img_size_raises() -> None:
    """``img_size`` no divisible por ``patch_size`` lanza ``ValueError``."""
    with pytest.raises(ValueError, match="divisible"):
        TSViT(img_size=130, patch_size=8)


def test_tsvit_variable_timesteps() -> None:
    """El modelo es invariante al numero de fechas (DOY indexa el PE temporal)."""
    model = _build_small_tsvit().eval()
    x5 = torch.randn(1, 5, 10, 128, 128)
    doy5 = torch.randint(1, 366, (1, 5))
    x12 = torch.randn(1, 12, 10, 128, 128)
    doy12 = torch.randint(1, 366, (1, 12))
    with torch.no_grad():
        out5 = model(x5, doy5)
        out12 = model(x12, doy12)
    assert out5.shape == (1, 18, 128, 128)
    assert out12.shape == (1, 18, 128, 128)


def test_tsvit_backward_produces_grad() -> None:
    """Backward sobre logits TSViT produce gradiente finito."""
    model = _build_small_tsvit().train()
    x = torch.randn(1, 6, 10, 128, 128)
    doy = torch.randint(1, 366, (1, 6))
    target = torch.randint(0, 18, (1, 128, 128))
    loss_fn = build_dice_ce_loss(n_classes=18)
    logits = model(x, doy)
    loss = loss_fn(logits, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
