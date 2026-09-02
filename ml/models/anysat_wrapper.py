"""AnySat frozen + linear head for dense segmentation (#6, Avance 4).

AnySat (Astruc et al., 2024, IGN — "AnySat: An Earth Observation Model for Any
Resolutions, Scales, and Modalities") is a multimodal/multi-temporal
foundation model loaded via ``torch.hub`` from ``gastruc/anysat``. Here the
encoder is used **frozen** as a dense feature extractor and only a **linear
head** (Conv 1x1) is trained, which projects those features to the 20
PASTIS-R classes and upsamples them to the target resolution. It is the
cheapest setup of the split (frozen encoder -> gradients only in the head,
~2-3 h L4).

Defensive design: the integration with AnySat's exact API (``output='dense'``)
is isolated in :meth:`AnySatSegmenter._encode` and the encoder is
**injectable**, so tests run with a synthetic encoder without downloading
weights. The real load via ``torch.hub`` and the exact forward signature are
validated in the dedicated cell of the Colab notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)

__all__ = ["AnySatSegmenter", "load_anysat_encoder"]

_HUB_REPO = "gastruc/anysat"
_HUB_MODEL = "anysat"


def load_anysat_encoder(
    *,
    repo: str = _HUB_REPO,
    model: str = _HUB_MODEL,
    pretrained: bool = True,
    flash_attn: bool = False,
) -> nn.Module:
    """Load the pretrained AnySat encoder via ``torch.hub`` (remote download).

    Args:
        repo: ``torch.hub`` repository (default ``gastruc/anysat``).
        model: Hub entrypoint (default ``anysat``).
        pretrained: Whether to load the pretrained weights.
        flash_attn: Whether to enable FlashAttention (requires support; default
            False for portability on L4/Colab).

    Returns:
        AnySat's ``nn.Module`` encoder.

    Raises:
        RuntimeError: if the hub load fails (no internet, inaccessible repo).
    """
    try:
        encoder: nn.Module = torch.hub.load(
            repo,
            model,
            pretrained=pretrained,
            flash_attn=flash_attn,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load AnySat from torch.hub ({repo}). "
            "Requires internet and an accessible repo. In Colab run the AnySat "
            "setup cell before instantiating AnySatSegmenter."
        ) from exc
    logger.info("anysat_encoder_loaded", repo=repo, pretrained=pretrained)
    return encoder


class AnySatSegmenter(nn.Module):
    """AnySat frozen + Conv 1x1 linear head for dense segmentation.

    Forward: ``image (B, T, C, H, W)`` (+ ``dates (B, T)``) -> ``logits
    (B, num_classes, target_size, target_size)``. The encoder runs without
    gradients (frozen); only the ``head`` is trained.

    The head uses ``nn.LazyConv2d`` when ``feature_dim`` is unknown, so the
    number of channels of AnySat's dense features is inferred at the first
    forward (the head is then materialized and is trainable).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        target_size: int = 256,
        patch_size: int = 10,
        modality: str = "s2",
        feature_dim: int | None = None,
        encoder: nn.Module | Callable[..., Any] | None = None,
        freeze: bool = True,
    ) -> None:
        """Initialize the segmenter.

        Args:
            num_classes: Number of output classes (20 in PASTIS-R).
            target_size: Spatial side of the output logits.
            patch_size: ``patch_size`` AnySat uses for the dense granularity.
            modality: Modality key in AnySat's input dict (``s2``).
            feature_dim: Dimension of AnySat's dense features. If ``None``
                it is inferred at the first forward via ``LazyConv2d``.
            encoder: Injectable encoder (for tests). If ``None`` AnySat is loaded
                via ``torch.hub`` with :func:`load_anysat_encoder`.
            freeze: Whether to freeze the encoder (default ``True``).
        """
        super().__init__()
        self.num_classes = num_classes
        self.target_size = target_size
        self.patch_size = patch_size
        self.modality = modality
        self._frozen = freeze

        self.encoder = encoder if encoder is not None else load_anysat_encoder()
        if freeze and isinstance(self.encoder, nn.Module):
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        if feature_dim is not None:
            self.head: nn.Module = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        else:
            self.head = nn.LazyConv2d(num_classes, kernel_size=1)

    def _encode(self, image: torch.Tensor, dates: torch.Tensor | None) -> torch.Tensor:
        """Run the AnySat encoder and return a dense feature map.

        Isolates AnySat's concrete signature. Builds the modality dict,
        invokes ``output='dense'`` and normalizes the output to ``(B, D, h, w)``.
        Also accepts synthetic encoders (callables) that already return
        ``(B, D, h, w)`` directly (tests).

        Args:
            image: ``(B, T, C, H, W)`` normalized Sentinel-2 temporal series.
            dates: ``(B, T)`` day-of-year per frame (or ``None``).

        Returns:
            Dense feature map ``(B, D, h, w)``.
        """
        data: dict[str, Any] = {self.modality: image}
        if dates is not None:
            data[f"{self.modality}_dates"] = dates

        try:
            feats = self.encoder(data, patch_size=self.patch_size, output="dense")
        except TypeError:
            # Synthetic test encoder: simple signature encoder(image) -> (B, D, h, w).
            return self._to_feature_map(self.encoder(image))

        # AnySat 'dense' returns channels-last (B, H, W, D) at full resolution;
        # it is permuted to channels-first (B, D, H, W), which is what the Conv2d head expects.
        if feats.dim() == 4:
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return self._to_feature_map(feats)

    @staticmethod
    def _to_feature_map(feats: torch.Tensor) -> torch.Tensor:
        """Normalize the encoder output to a spatial map ``(B, D, h, w)``.

        AnySat ``output='dense'`` may return ``(B, N, D)`` (tokens) or already
        ``(B, D, h, w)``. If it arrives as square tokens it is reshaped to a map.

        Args:
            feats: Raw encoder output.

        Returns:
            Tensor ``(B, D, h, w)``.

        Raises:
            ValueError: if the output is not interpretable as a dense map.
        """
        if feats.dim() == 4:
            return feats  # (B, D, h, w)
        if feats.dim() == 3:
            # (B, N, D) -> (B, D, sqrt(N), sqrt(N)) if N is a perfect square.
            b, n, d = feats.shape
            side = round(n**0.5)
            if side * side != n:
                raise ValueError(
                    f"Non-square dense features (N={n}); adjust patch_size or "
                    "the token handling in _to_feature_map."
                )
            return feats.transpose(1, 2).reshape(b, d, side, side)
        raise ValueError(f"Unsupported dense feature shape: {tuple(feats.shape)}")

    @torch.no_grad()
    def extract_features(
        self, image: torch.Tensor, dates: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the dense feature map of the frozen encoder ``(B, D, h, w)``.

        Designed to cache the features once and tune only the linear head
        without re-running the encoder (the bottleneck in AnySat). It does not
        apply the head or the upsample; the caller trains its own head over the cache.

        Args:
            image: ``(B, T, C, H, W)`` normalized Sentinel-2 series.
            dates: ``(B, T)`` day-of-year per frame (optional).

        Returns:
            Dense feature map ``(B, D, h, w)``.
        """
        return self._encode(image, dates)

    def forward(self, image: torch.Tensor, dates: torch.Tensor | None = None) -> torch.Tensor:
        """Produce dense segmentation logits.

        Args:
            image: ``(B, T, C, H, W)`` normalized Sentinel-2 series.
            dates: ``(B, T)`` day-of-year per frame (optional).

        Returns:
            Logits ``(B, num_classes, target_size, target_size)``.
        """
        if self._frozen:
            with torch.no_grad():
                feats = self._encode(image, dates)
        else:
            feats = self._encode(image, dates)

        logits = self.head(feats)
        return F.interpolate(
            logits, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False
        )
