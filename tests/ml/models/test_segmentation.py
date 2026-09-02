"""Tests de las arquitecturas de segmentacion (#1 U-Net, #6 AnySat).

U-Net se construye con ``encoder_weights=None`` para no descargar pesos ImageNet
(test hermetico). AnySat se prueba con un encoder sinteptico inyectado, evitando
la descarga via ``torch.hub`` (que se valida en el notebook Colab).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.models.anysat_wrapper import AnySatSegmenter
from ml.models.segmentation import build_segmentation_model, build_unet


def test_build_unet_forward_shape() -> None:
    """U-Net mapea (B,10,H,W) -> (B,num_classes,H,W)."""
    model = build_unet(20, encoder_weights=None)
    model.eval()
    x = torch.randn(2, 10, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 20, 64, 64)


def test_build_segmentation_model_unknown_raises() -> None:
    """Un modelo no registrado levanta ValueError."""
    with pytest.raises(ValueError, match="Unknown segmentation model"):
        build_segmentation_model("does-not-exist", 20)


class _DummyAnySatEncoder(nn.Module):
    """Encoder sinteptico que imita la salida densa (B, D, h, w) de AnySat."""

    def __init__(self, feature_dim: int = 16, grid: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.grid = grid
        # Parametro real para verificar el congelamiento del encoder.
        self.proj = nn.Parameter(torch.randn(feature_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # (B, T, C, H, W)
        batch = image.shape[0]
        base = torch.ones(batch, self.feature_dim, self.grid, self.grid)
        return base * self.proj.view(1, -1, 1, 1)


def test_anysat_segmenter_forward_shape_and_freeze() -> None:
    """AnySat frozen: salida (B,num_classes,target,target); encoder congelado."""
    encoder = _DummyAnySatEncoder(feature_dim=16, grid=8)
    model = AnySatSegmenter(20, target_size=32, encoder=encoder, freeze=True)

    image = torch.randn(2, 10, 10, 64, 64)  # (B, T, C, H, W)
    dates = torch.zeros(2, 10, dtype=torch.int64)
    logits = model(image, dates)
    assert logits.shape == (2, 20, 32, 32)

    # Encoder congelado, cabeza entrenable.
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_anysat_handles_token_output() -> None:
    """_to_feature_map reacomoda tokens (B,N,D) cuadrados a mapa (B,D,h,w)."""
    feats = torch.randn(2, 64, 16)  # N=64 -> 8x8
    mapped = AnySatSegmenter._to_feature_map(feats)
    assert mapped.shape == (2, 16, 8, 8)
