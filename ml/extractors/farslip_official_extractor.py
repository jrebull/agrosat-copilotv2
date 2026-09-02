"""Official FarSLIP image-embedding extractor (Li et al. 2025, arXiv:2511.14901).

Loads the published FarSLIP visual tower (ViT-B/16, RGB, 224x224) from the
authors' checkpoint and exposes a single ``encode_images`` call returning the
512-dim L2-normalized CLIP image embedding.

This is a DIFFERENT model from :class:`ml.extractors.farslip_extractor.FarSLIPExtractor`,
which targets a project-internal 4-band Sentinel-2 distillation (used by the
multisensor fusion and the SegFormer open-vocab head). The official model:

- Is an RGB model (3 channels), input 224x224, embed dim 512.
- Ships as a training checkpoint (``state_dict`` with a ``module.`` prefix and a
  text tower trained with LongCLIP, which is irrelevant for image embeddings).
- Loads with the standard ``open_clip`` ``ViT-B-16`` architecture and
  ``force_quick_gelu=True`` (the authors train with quick-GELU).

Weights: published at HuggingFace ``ZhenShiL/FarSLIP`` (variants s1/s2 x
ViT-B/16,B/32). The text tower uses a 248-token positional embedding (LongCLIP);
only the visual tower is loaded here, so that mismatch is expected and ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL.Image import Image

_log = structlog.get_logger(__name__)

#: Sentinel-2 band order in PASTIS-R DATA_S2 arrays.
_PASTIS_S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
#: RGB channel indices (B04, B03, B02) into a PASTIS-R band-first array.
_RGB_IDX = (
    _PASTIS_S2_BANDS.index("B04"),
    _PASTIS_S2_BANDS.index("B03"),
    _PASTIS_S2_BANDS.index("B02"),
)
#: Output embedding dimension (CLIP projection). FarSLIP resizes inputs to 224
#: internally via the open_clip preprocessing transform.
EMBED_DIM = 512


class FarSLIPOfficialExtractor:
    """Lazy loader of the official FarSLIP visual tower for image embeddings.

    Args:
        checkpoint_path: Local path to the FarSLIP ``.pt`` checkpoint
            (e.g. ``FarSLIP2_ViT-B-16.pt`` from ``ZhenShiL/FarSLIP``).
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        import open_clip

        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"FarSLIP checkpoint not found: {self.checkpoint_path}")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=None, force_quick_gelu=True
        )
        self._preprocess = preprocess

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        visual_sd = {
            k.replace("module.", "")[len("visual.") :]: v
            for k, v in state.items()
            if k.replace("module.", "").startswith("visual.")
        }
        missing, unexpected = model.visual.load_state_dict(visual_sd, strict=True)
        _log.info(
            "FarSLIP official visual tower loaded",
            checkpoint=str(self.checkpoint_path),
            n_params=len(visual_sd),
            missing=len(missing),
            unexpected=len(unexpected),
        )

        model.eval()
        self.model = model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def s2_to_rgb_pil(median_band_first: np.ndarray) -> Image:
        """Convert a band-first Sentinel-2 composite to an RGB PIL image.

        Takes the B04/B03/B02 channels, applies a 2-98 percentile stretch per
        composite (robust to outliers) and returns an 8-bit RGB image ready for
        the FarSLIP preprocessing transform.

        Args:
            median_band_first: ``(10, H, W)`` float array (e.g. temporal median
                of a PASTIS-R patch or a parcel crop).

        Returns:
            An RGB ``PIL.Image`` of size ``(H, W)``.
        """
        from PIL import Image as _Image

        rgb = median_band_first[list(_RGB_IDX)]
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        arr = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
        return _Image.fromarray(arr)

    @torch.inference_mode()
    def encode_images(self, images: list[Image], batch_size: int = 128) -> np.ndarray:
        """Encode RGB images into L2-normalized 512-dim FarSLIP embeddings.

        Args:
            images: List of RGB ``PIL.Image`` (any size; resized by the transform).
            batch_size: Forward-pass batch size.

        Returns:
            ``(N, 512)`` float32 array, each row L2-normalized.
        """
        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            x = torch.stack([self._preprocess(im) for im in chunk]).to(self.device)
            feats = self.model.encode_image(x)
            feats = F.normalize(feats, p=2, dim=-1)
            out.append(feats.cpu().float().numpy())
        return np.concatenate(out) if out else np.empty((0, EMBED_DIM), dtype=np.float32)


__all__ = ["EMBED_DIM", "FarSLIPOfficialExtractor"]
