"""Fine-tune the AnySat linear head over cached encoder features.

The AnySat encoder is frozen, so its dense features do not change between
Optuna trials. Re-running it per trial (tens of minutes per epoch at batch 1
over the time series) is the tuning bottleneck. Here the features are
precomputed **only once** and each trial trains only the Conv 1x1 head over
them (seconds), so >=30 trials run in minutes instead of hours.

Reuses the dense metrics (:mod:`ml.eval.dense_metrics`) and the HCAT 18-class
-> 6-group grouping (:mod:`ml.analysis.hcat_grouping`) from the main pipeline,
so that the ``miou_grouped`` Optuna optimizes is the same one reported by the
final model (separation of concerns, CLAUDE.md rule 8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ml.analysis.hcat_grouping import hcat6_dense_lut
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTIS_NUM_CLASSES,
    PASTISDataset,
)

if TYPE_CHECKING:  # pragma: no cover - only for type annotations
    from ml.models.anysat_wrapper import AnySatSegmenter

logger = structlog.get_logger(__name__)

__all__ = ["CachedFeatures", "cache_encoder_features", "train_head"]

# "Non-crop" class for the grouped metrics (background/void predicted over a
# crop pixel): never a target, so the macro averages only the 6 groups.
_NON_CROP_GROUP = 6
_GROUPED_CLASSES = 7


class CachedFeatures:
    """Precomputed ``(features, labels)`` pair from the frozen encoder, on CPU.

    Attributes:
        features: ``(N, D, h, w)`` float16 on CPU (dense encoder map).
        labels: ``(N, target_size, target_size)`` long on CPU (semantic 0-19).
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    @property
    def feature_dim(self) -> int:
        """Number of channels ``D`` of the dense encoder features."""
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return int(self.features.shape[0])


@torch.no_grad()
def cache_encoder_features(
    model: AnySatSegmenter,
    patch_ids: Sequence[str],
    *,
    root: Path,
    target_size: int,
    norm: tuple,
    device: str = "auto",
    batch_size: int = 4,
    num_workers: int = 0,
) -> CachedFeatures:
    """Run the frozen encoder once per patch and cache ``(features, labels)``.

    Args:
        model: :class:`~ml.models.anysat_wrapper.AnySatSegmenter` (or one
            compatible with ``extract_features(image, dates)``), with the
            encoder already loaded.
        patch_ids: Ids of the PASTIS patches to cache.
        root: Root of the PASTIS-R dataset.
        target_size: Spatial side of the labels (must match the model's).
        norm: Per-band normalization stats (from ``load_norm_stats``).
        device: ``cpu``, ``cuda`` or ``auto``.
        batch_size: Batch for the encoder pass (larger = faster if it fits).
        num_workers: DataLoader workers.

    Returns:
        :class:`CachedFeatures` with the features ``(N, D, h, w)`` in float16
        (CPU) and the labels ``(N, target_size, target_size)`` long (CPU).
    """
    dev = _resolve_device(device)
    model.to(dev)
    model.eval()
    dataset = PASTISDataset(
        list(patch_ids),
        root=root,
        target_size=target_size,
        temporal_reduction="none",
        norm=norm,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    feats_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for batch in loader:
        image = batch["image"].to(dev)
        dates = batch.get("dates")
        dates = dates.to(dev) if dates is not None else None
        feats = model.extract_features(image, dates)  # (B, D, h, w)
        feats_chunks.append(feats.detach().to("cpu", dtype=torch.float16))
        label_chunks.append(batch["semantic"].cpu())
    features = torch.cat(feats_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    logger.info(
        "anysat_features_cached",
        n=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feat_hw=tuple(features.shape[2:]),
    )
    return CachedFeatures(features, labels)


def _resolve_device(device: str) -> torch.device:
    """Resolve the device (``auto`` -> cuda if available, otherwise cpu)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_group_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 18-class -> 6-group LUTs for target and for prediction."""
    group_lut = hcat6_dense_lut()
    lut_target = torch.as_tensor(group_lut, device=device)
    pred_lut = group_lut.copy()
    pred_lut[pred_lut == 255] = _NON_CROP_GROUP  # background/void predicted -> "non-crop"
    lut_pred = torch.as_tensor(pred_lut, device=device)
    return lut_target, lut_pred


def _evaluate_head(
    head: nn.Module,
    cached: CachedFeatures,
    *,
    target_size: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate the head over cached features: flat and grouped mIoU/F1/pixacc."""
    head.eval()
    acc = DenseConfusionAccumulator(
        PASTIS_NUM_CLASSES, ignore_index=PASTIS_IGNORE_INDEX, device=str(device)
    )
    acc_grouped = DenseConfusionAccumulator(_GROUPED_CLASSES, ignore_index=255, device=str(device))
    lut_target, lut_pred = _build_group_luts(device)
    n = len(cached)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            feats = cached.features[i : i + batch_size].to(device, dtype=torch.float32)
            target = cached.labels[i : i + batch_size].to(device)
            logits = F.interpolate(
                head(feats), size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)
            acc.update(preds, target)
            acc_grouped.update(lut_pred[preds.clamp(0, 19)], lut_target[target.clamp(0, 19)])
    flat = acc.compute()
    grouped = {f"{k}_grouped": v for k, v in acc_grouped.compute().items()}
    return {**flat, **grouped}


def train_head(
    train_cache: CachedFeatures,
    val_cache: CachedFeatures,
    *,
    num_classes: int = PASTIS_NUM_CLASSES,
    target_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 8,
    batch_size: int = 8,
    device: str = "auto",
    seed: int = 0,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Train a Conv 1x1 head over cached features and return the best metric.

    Each epoch trains the head (AdamW + CrossEntropy with ``ignore_index``
    void) and evaluates flat and grouped mIoU/F1/pixel-accuracy over the
    validation cache. The selection metric is ``miou_grouped`` (comparable
    with the baseline and the final model). It is the unit of work of each
    Optuna trial.

    Args:
        train_cache: Train features+labels (from :func:`cache_encoder_features`).
        val_cache: Validation features+labels.
        num_classes: Number of output classes (20 in PASTIS-R).
        target_size: Spatial side of the logits/labels.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Number of head epochs.
        batch_size: Batch over the cached features (cheap; can be large).
        device: ``cpu``, ``cuda`` or ``auto``.
        seed: Seed for the minibatch order (per-trial reproducibility).
        on_epoch: Callback ``(epoch, metrics)`` after evaluating each epoch
            (Optuna pruning).

    Returns:
        Dictionary with the best metrics (highest ``miou_grouped``) observed.
    """
    dev = _resolve_device(device)
    feature_dim = train_cache.feature_dim
    head = nn.Conv2d(feature_dim, num_classes, kernel_size=1).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PASTIS_IGNORE_INDEX)

    generator = torch.Generator().manual_seed(seed)
    n = len(train_cache)
    best: dict[str, float] = {"miou_grouped": -1.0}
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            feats = train_cache.features[idx].to(dev, dtype=torch.float32)
            target = train_cache.labels[idx].to(dev)
            optimizer.zero_grad()
            logits = F.interpolate(
                head(feats), size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        metrics = _evaluate_head(
            head, val_cache, target_size=target_size, device=dev, batch_size=batch_size
        )
        if metrics["miou_grouped"] >= best["miou_grouped"]:
            best = metrics
        if on_epoch is not None:
            on_epoch(epoch, metrics)
    return best
