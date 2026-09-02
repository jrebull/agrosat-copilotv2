"""2D dense semantic segmentation architectures (EPIC 5, Avance 4).

Factory for segmentation models based on 2D temporal composites
(``image (B, 10, H, W) -> logits (B, num_classes, H, W)``). The team's
assigned model #1, **U-Net ResNet-50**, is built directly from
``segmentation_models_pytorch`` (ImageNet-pretrained encoder, adapted to 10
Sentinel-2 channels). The ``build_segmentation_model`` factory is registrable
so the rest of the team can hook in #2 (DeepLabv3+) and #3 (SegFormer) on the
same dense pipeline (:mod:`ml.ingest.pastis_dataset`).

The temporal models (#4 U-TAE, #5 TSViT, #6 AnySat) consume the full series
and live in dedicated wrappers (see :mod:`ml.models.anysat_wrapper`).
"""

from __future__ import annotations

from collections.abc import Callable

import segmentation_models_pytorch as smp
import torch.nn as nn

__all__ = [
    "SEGMENTATION_BUILDERS",
    "build_deeplabv3plus",
    "build_segmentation_model",
    "build_unet",
]

_S2_CHANNELS = 10


def build_unet(
    num_classes: int,
    *,
    in_channels: int = _S2_CHANNELS,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build U-Net with a ResNet-50 encoder pretrained on ImageNet (#1).

    The encoder's first conv is automatically adapted from 3 to ``in_channels``
    channels (smp replicates/averages the RGB weights). Output without
    activation (logits) to use ``CrossEntropyLoss`` with ``ignore_index``.

    Args:
        num_classes: Number of output classes (20 in PASTIS-R).
        in_channels: Input channels (10 Sentinel-2 bands).
        encoder_name: smp backbone (default ``resnet50``).
        encoder_weights: Encoder weights (``imagenet`` or ``None``).

    Returns:
        ``nn.Module`` mapping ``(B, in_channels, H, W) -> (B, num_classes, H, W)``.
    """
    model: nn.Module = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model


def build_deeplabv3plus(
    num_classes: int,
    *,
    in_channels: int = _S2_CHANNELS,
    encoder_name: str = "mobilenet_v2",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a lightweight DeepLabv3+ (#2) on the same dense pipeline.

    Provided so the member in charge of #2 reuses the factory without
    duplicating the pipeline. ``mobilenet_v2`` is the lightweight encoder (smp
    does not expose ``mobilenet_v3`` for DeepLabv3+; v2 is the available
    efficient equivalent).

    Args:
        num_classes: Number of output classes.
        in_channels: Input channels.
        encoder_name: smp backbone.
        encoder_weights: Encoder weights.

    Returns:
        Dense segmentation ``nn.Module``.
    """
    model: nn.Module = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model


SEGMENTATION_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "unet": build_unet,
    "deeplabv3plus": build_deeplabv3plus,
}
"""``name -> builder`` registry of 2D models. The team adds entries here."""


def build_segmentation_model(kind: str, num_classes: int, **kwargs: object) -> nn.Module:
    """Build a 2D segmentation model by registered name.

    Args:
        kind: Key in :data:`SEGMENTATION_BUILDERS` (``unet``, ``deeplabv3plus``).
        num_classes: Number of output classes.
        **kwargs: Overrides passed to the concrete builder (``encoder_name``, etc.).

    Returns:
        The built ``nn.Module``.

    Raises:
        ValueError: if ``kind`` is not registered.
    """
    builder = SEGMENTATION_BUILDERS.get(kind)
    if builder is None:
        valid = ", ".join(sorted(SEGMENTATION_BUILDERS))
        raise ValueError(f"Unknown segmentation model: {kind!r}. Valid: {valid}.")
    return builder(num_classes, **kwargs)  # type: ignore[arg-type]
