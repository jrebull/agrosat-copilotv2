"""DeepLabv3+ (MobileNetV3-Large) factory for 10-band dense segmentation.

Base CNN segmenter of EPIC 5 (US-025, Task 2). Wraps
``segmentation_models_pytorch`` 0.5 to produce a DeepLabv3+ with a
MobileNetV3-Large encoder (timm) adapted to a 10-band Sentinel-2 input and a
dense output of 18 PASTIS-R semantic classes.

The native encoder registry of smp 0.5 does NOT include ``mobilenet_v3_large``
(only ResNet/EfficientNet/MiT/MobileOne + a ``timm-*`` subset). For
MobileNetV3-Large the timm-universal prefix ``tu-`` is used: the encoder name
is ``tu-mobilenetv3_large_100``, which delegates to ``timm`` and downloads the
ImageNet weights adapting the first convolution to ``in_channels`` channels.

Unlike smp's default usage (RGB input, ``in_channels=3``), here
``in_channels=10``: smp/timm adapt the first convolution by replicating and
averaging the ImageNet weights over the extra channels. If the ImageNet
initialization fails with 10 channels it retries with random weights
(``encoder_weights=None``) emitting a structured warning.

The module also exposes a combined Dice + CrossEntropy loss
(:func:`build_dice_ce_loss`), a common pattern in imbalanced crop
segmentation: Dice stabilizes minority classes and CrossEntropy provides
per-pixel gradient. Both respect ``ignore_index`` for Background/Void.

Technical decisions
-------------------

- ``decoder_atrous_rates=(6, 12, 18)`` per the US-025 acceptance criterion
  (smp's default is ``(12, 24, 36)``); smaller rates favor
  small parcels at 128px.
- ``decoder_aspp_separable=True`` reduces ASPP module parameters
  (depthwise separable convolutions), useful for training on an RTX laptop.
- ``encoder_name="mobilenet_v3_large"`` (timm) for VRAM budget
  (<8 GB, batch 8, 128px per plan US-025).

Acknowledgment
--------------

- segmentation-models-pytorch 0.5 (MIT License), Pavel Iakubovskii.
- Chen et al. ``Encoder-Decoder with Atrous Separable Convolution for
  Semantic Image Segmentation`` (DeepLabv3+). ECCV 2018.
- Howard et al. ``Searching for MobileNetV3``. ICCV 2019.
"""

from __future__ import annotations

from collections.abc import Sequence

import segmentation_models_pytorch as smp
import structlog
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn

__all__ = [
    "DiceCrossEntropyLoss",
    "build_deeplabv3plus_mobilenet",
    "build_dice_ce_loss",
]

logger = structlog.get_logger(__name__)

# Default ignore index: aligned with PASTISSegmentationDataset
# (Background/Void mapped to 255, outside the range [0..n_classes-1]).
_DEFAULT_IGNORE_INDEX = 255

# Encoder name in smp 0.5: timm-universal prefix ``tu-`` because
# ``mobilenet_v3_large`` is not in the native smp registry. timm exposes it
# as ``mobilenetv3_large_100``.
_ENCODER_NAME = "tu-mobilenetv3_large_100"


def build_deeplabv3plus_mobilenet(
    in_channels: int = 10,
    classes: int = 18,
    atrous_rates: tuple[int, int, int] = (6, 12, 18),
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a DeepLabv3+ with a MobileNetV3-Large encoder for 10 bands.

    Args:
        in_channels: Number of input channels. Default 10 (PASTIS-R
            Sentinel-2 bands). smp/timm adapt the first convolution
            when it differs from 3 (RGB).
        classes: Number of output classes (dense logit channels). Default
            18 (PASTIS-R semantic classes without Background/Void).
        atrous_rates: Dilation rates of the ASPP module (3 integers). Default
            ``(6, 12, 18)`` per the US-025 acceptance criterion.
        encoder_weights: Initial encoder weights. ``"imagenet"`` (default)
            or ``None`` (random). If ``"imagenet"`` fails to adapt
            to ``in_channels`` it retries with ``None``.

    Returns:
        A ``torch.nn.Module`` model mapping ``(B, in_channels, H, W)`` to
        dense logits ``(B, classes, H, W)``.

    Raises:
        ValueError: If ``atrous_rates`` does not contain exactly 3 integers.
    """
    rates = tuple(atrous_rates)
    if len(rates) != 3:
        raise ValueError(f"atrous_rates debe tener 3 valores enteros, recibido {rates!r}")

    try:
        model: nn.Module = smp.DeepLabV3Plus(
            encoder_name=_ENCODER_NAME,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_atrous_rates=rates,
            decoder_aspp_separable=True,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        if encoder_weights is None:
            # No fallback possible: random initialization was already requested.
            raise
        logger.warning(
            "deeplabv3plus_imagenet_init_failed",
            in_channels=in_channels,
            classes=classes,
            error=str(exc),
            fallback="encoder_weights=None",
        )
        model = smp.DeepLabV3Plus(
            encoder_name=_ENCODER_NAME,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            decoder_atrous_rates=rates,
            decoder_aspp_separable=True,
        )

    logger.info(
        "deeplabv3plus_built",
        encoder=_ENCODER_NAME,
        in_channels=in_channels,
        classes=classes,
        atrous_rates=rates,
        aspp_separable=True,
        encoder_weights=encoder_weights,
    )
    return model


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice + CrossEntropy loss for multiclass segmentation.

    Weighted sum of :class:`segmentation_models_pytorch.losses.DiceLoss`
    (``multiclass`` mode, over logits) and
    :class:`torch.nn.CrossEntropyLoss`. Both terms ignore ``ignore_index``
    (Background/Void). CrossEntropy accepts optional per-class weights to
    counteract crop imbalance.

    The total term is ``dice_weight * dice + ce_weight * ce``. Dice stabilizes
    minority classes (region overlap) and CrossEntropy provides per-pixel
    gradient; the combination is standard in crop segmentation.

    Attributes:
        dice: Multiclass Dice term over logits.
        ce: Per-pixel CrossEntropy term.
        dice_weight: Weight of the Dice term.
        ce_weight: Weight of the CrossEntropy term.
    """

    def __init__(
        self,
        ignore_index: int = _DEFAULT_IGNORE_INDEX,
        n_classes: int = 18,
        class_weights: torch.Tensor | Sequence[float] | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
    ) -> None:
        """Initialize the combined loss.

        Args:
            ignore_index: Label to ignore in both terms (Background/Void).
            n_classes: Number of dense logit classes. Used only to
                validate the shape of ``class_weights``.
            class_weights: Per-class weights for CrossEntropy. Tensor or sequence
                of length ``n_classes``, or ``None`` (unweighted).
            dice_weight: Weight of the Dice term in the sum.
            ce_weight: Weight of the CrossEntropy term in the sum.

        Raises:
            ValueError: If ``class_weights`` does not have length ``n_classes``.
        """
        super().__init__()

        weight_tensor: torch.Tensor | None
        if class_weights is None:
            weight_tensor = None
        else:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if weight_tensor.numel() != n_classes:
                raise ValueError(
                    "class_weights debe tener longitud n_classes "
                    f"({n_classes}), recibido {weight_tensor.numel()}"
                )

        self.dice = DiceLoss(
            mode="multiclass",
            from_logits=True,
            ignore_index=ignore_index,
        )
        # ``weight`` is registered as a buffer inside CrossEntropyLoss and
        # moves with ``.to(device)`` along with the parent module.
        self.ce = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=ignore_index,
        )
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            logits: Dense logits ``(B, C, H, W)`` (without softmax).
            target: Per-pixel integer labels ``(B, H, W)``, with
                ``ignore_index`` on pixels to omit.

        Returns:
            A scalar ``torch.Tensor`` with the weighted combined loss.
        """
        target_long = target.long()
        dice_term: torch.Tensor = self.dice(logits, target_long)
        ce_term: torch.Tensor = self.ce(logits, target_long)
        # A batch/patch entirely ``ignore_index`` (Background/Void) leaves
        # CrossEntropyLoss averaging over zero valid pixels -> NaN, which would
        # poison training. It is neutralized to zero while preserving the
        # graph (``nan_to_num`` keeps the gradient of the valid paths).
        ce_term = torch.nan_to_num(ce_term, nan=0.0)
        return self.dice_weight * dice_term + self.ce_weight * ce_term


def build_dice_ce_loss(
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    n_classes: int = 18,
    class_weights: torch.Tensor | Sequence[float] | None = None,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> DiceCrossEntropyLoss:
    """Build the weighted combined Dice + CrossEntropy loss.

    Args:
        ignore_index: Label to ignore (Background/Void). Default 255.
        n_classes: Number of dense logit classes. Default 18.
        class_weights: Optional per-class weights for CrossEntropy (length
            ``n_classes``) or ``None``.
        dice_weight: Weight of the Dice term in the sum.
        ce_weight: Weight of the CrossEntropy term in the sum.

    Returns:
        A ready-to-use :class:`DiceCrossEntropyLoss` instance.
    """
    return DiceCrossEntropyLoss(
        ignore_index=ignore_index,
        n_classes=n_classes,
        class_weights=class_weights,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    )
