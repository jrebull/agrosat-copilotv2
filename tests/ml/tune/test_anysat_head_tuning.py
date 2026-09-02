"""Tests del ajuste fino de la cabeza de AnySat sobre features cacheadas.

``train_head`` se prueba hermetico con features sinteticas (sin dataset ni encoder).
``cache_encoder_features`` se valida con un smoke contra el PASTIS-R local inyectando
un encoder sinteptico (sin descargar pesos via torch.hub); se omite si no hay dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ml.models.anysat_wrapper import AnySatSegmenter
from ml.tune.anysat_head_tuning import (
    CachedFeatures,
    cache_encoder_features,
    train_head,
)

_PASTIS_ROOT = Path("data/PASTIS-R")
_GROUPED_KEYS = {"miou_grouped", "f1_macro_grouped", "pixel_accuracy_grouped"}


def _synthetic_cache(n: int, feature_dim: int, grid: int, target: int) -> CachedFeatures:
    """Crea un cache sintetico (features ruidosas correlacionadas con el label)."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 19, (n, target, target), generator=generator)
    features = torch.randn(n, feature_dim, grid, grid, generator=generator).to(torch.float16)
    return CachedFeatures(features, labels)


def test_train_head_returns_grouped_metrics() -> None:
    """train_head devuelve metricas planas y agrupadas en [0,1] sobre features sinteticas."""
    train_cache = _synthetic_cache(n=6, feature_dim=16, grid=8, target=32)
    val_cache = _synthetic_cache(n=4, feature_dim=16, grid=8, target=32)

    best = train_head(
        train_cache,
        val_cache,
        target_size=32,
        epochs=2,
        batch_size=2,
        device="cpu",
        seed=0,
    )
    assert _GROUPED_KEYS.issubset(best.keys())
    assert {"miou", "f1_macro", "pixel_accuracy"}.issubset(best.keys())
    assert 0.0 <= best["miou_grouped"] <= 1.0


def test_train_head_invokes_on_epoch_callback() -> None:
    """El callback on_epoch se llama una vez por epoca (lo usa el pruning de Optuna)."""
    cache = _synthetic_cache(n=4, feature_dim=8, grid=8, target=16)
    seen: list[int] = []

    train_head(
        cache,
        cache,
        target_size=16,
        epochs=3,
        batch_size=2,
        device="cpu",
        on_epoch=lambda epoch, metrics: seen.append(epoch),
    )
    assert seen == [0, 1, 2]


class _DummyAnySatEncoder(nn.Module):
    """Encoder sinteptico que imita la salida densa (B, D, h, w) de AnySat."""

    def __init__(self, feature_dim: int = 12, grid: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.grid = grid
        self.proj = nn.Parameter(torch.randn(feature_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # (B, T, C, H, W)
        batch = image.shape[0]
        base = torch.ones(batch, self.feature_dim, self.grid, self.grid)
        return base * self.proj.view(1, -1, 1, 1)


@pytest.mark.skipif(
    not (_PASTIS_ROOT / "metadata.geojson").exists(),
    reason="PASTIS-R local no disponible (falta metadata.geojson)",
)
def test_cache_encoder_features_smoke() -> None:
    """Cachea features de unos pocos patches reales con un encoder inyectado y tunea la cabeza."""
    from ml.ingest.pastis_dataset import load_norm_stats, pastis_fold_split

    split = pastis_fold_split(_PASTIS_ROOT, train_folds=(1,), val_folds=(4,), test_folds=())
    train_ids, val_ids = split["train"][:2], split["val"][:2]
    norm = load_norm_stats(_PASTIS_ROOT, folds=(1,))

    feature_dim, grid, target = 12, 8, 32
    model = AnySatSegmenter(
        20, target_size=target, encoder=_DummyAnySatEncoder(feature_dim, grid), freeze=True
    )
    cache_kwargs = {"root": _PASTIS_ROOT, "target_size": target, "norm": norm, "device": "cpu"}
    train_cache = cache_encoder_features(model, train_ids, batch_size=2, **cache_kwargs)
    val_cache = cache_encoder_features(model, val_ids, batch_size=2, **cache_kwargs)
    assert train_cache.feature_dim == feature_dim
    assert train_cache.features.shape[2:] == (grid, grid)
    assert train_cache.labels.shape == (len(train_ids), target, target)

    best = train_head(
        train_cache, val_cache, target_size=target, epochs=1, batch_size=2, device="cpu"
    )
    assert "miou_grouped" in best
