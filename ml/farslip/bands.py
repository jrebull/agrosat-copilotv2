"""Band selection / reordering for the FarSLIP ablation (US-035).

The FarSLIP crops in ``data/farslip_pairs/{roi}/crops/*.tif`` store FOUR
Sentinel-2 bands in the fixed channel layout (see ``dataset.py:1-6``)::

    B02 = index 0   (blue)
    B03 = index 1   (green)
    B04 = index 2   (red)
    B08 = index 3   (NIR)

US-035 runs an A/B ablation over THREE band selections, each combined with a
prototype source (US-034). ``band_selection`` is the single source of truth for
two coupled facts: (a) which bands / order the model receives and (b) how many
channels the student ``patch_embed`` must accept (``n_in_channels``):

    - ``rgb``     -> ``[2, 1, 0]`` = B04,B03,B02 (RGB true color), 3 channels.
    - ``nir_rgb`` -> ``[3, 2, 1]`` = B08,B04,B03 (false color NIR-R-G), 3 channels.
    - ``4band``   -> ``[0, 1, 2, 3]`` = B02,B03,B04,B08 (identity, US-034 compat),
      4 channels.

The 3-channel variants are a no-op on the patch_embed (``adapt_patch_embed_to_
n_channels`` returns early when ``in == target == 3``); the 4-band variant
reuses the 3->4 adaptation with NIR = mean(RGB) of US-034. The 4-band identity
is mandatory so the variant stays compatible with the US-034 checkpoint's
learned ``patch_embedding`` weights.

This module is pure tensor slicing (CPU, no network, no GPU): it is wired as the
``transform`` of :class:`ml.farslip.dataset.FarSLIPDataset` via
``functools.partial``.
"""

from __future__ import annotations

from typing import Literal

import structlog
import torch

_log = structlog.get_logger(__name__)

#: Allowed band-selection modes (US-035).
BandSelection = Literal["rgb", "nir_rgb", "4band"]

#: Canonical channel index of each Sentinel-2 band in the FarSLIP crop layout.
#: Source of truth for the "which physical band lives where" contract; the
#: unit test asserts the crop has >= 4 channels in THIS order (mitigates the
#: 4-vs-10-band mismatch risk R-BANDS).
_BAND_INDEX: dict[str, int] = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}

#: Channel indices (into the 4-band crop) produced by each selection.
#: ``rgb``/``nir_rgb`` reorder to 3 channels; ``4band`` is the identity.
_SELECTIONS: dict[str, tuple[int, ...]] = {
    "rgb": (
        _BAND_INDEX["B04"],
        _BAND_INDEX["B03"],
        _BAND_INDEX["B02"],
    ),  # (2, 1, 0)
    "nir_rgb": (
        _BAND_INDEX["B08"],
        _BAND_INDEX["B04"],
        _BAND_INDEX["B03"],
    ),  # (3, 2, 1)
    "4band": (
        _BAND_INDEX["B02"],
        _BAND_INDEX["B03"],
        _BAND_INDEX["B04"],
        _BAND_INDEX["B08"],
    ),  # (0, 1, 2, 3) identity
}

#: Number of channels the student ``patch_embed`` must accept per selection.
#: Single source of truth for ``n_in_channels`` (R-NCHAN): 3 for the baselines,
#: 4 for the four-band variant.
_N_CHANNELS: dict[str, int] = {"rgb": 3, "nir_rgb": 3, "4band": 4}

#: Number of channels the FarSLIP crop is expected to carry.
_MIN_CROP_CHANNELS: int = 4


def _validate_selection(sel: str) -> None:
    """Validate ``sel`` against the allowed modes.

    Args:
        sel: candidate band-selection mode.

    Raises:
        ValueError: if ``sel`` is not one of ``rgb``/``nir_rgb``/``4band``.
    """
    if sel not in _SELECTIONS:
        raise ValueError(f"invalid band_selection {sel!r}; expected one of {sorted(_SELECTIONS)}")


def n_in_channels_for(sel: BandSelection) -> int:
    """Return how many channels the patch_embed must accept for ``sel``.

    Single source of truth for ``n_in_channels`` (US-035 R-NCHAN): ``rgb`` and
    ``nir_rgb`` need 3 channels (no patch_embed adaptation); ``4band`` needs 4
    (the 3->4 mean-RGB adaptation of US-034).

    Args:
        sel: band-selection mode (``rgb``/``nir_rgb``/``4band``).

    Returns:
        ``3`` for ``rgb``/``nir_rgb``, ``4`` for ``4band``.

    Raises:
        ValueError: if ``sel`` is not a valid mode.
    """
    _validate_selection(sel)
    return _N_CHANNELS[sel]


def select_and_reorder_bands(img: torch.Tensor, sel: BandSelection) -> torch.Tensor:
    """Select / reorder the channels of a FarSLIP crop per variant.

    Indexes the channel dimension of a ``(C, H, W)`` (or ``(N, C, H, W)``) tensor
    whose layout is B02=0, B03=1, B04=2, B08=3. Returns:

        - ``rgb``     -> ``[2, 1, 0]`` = B04,B03,B02 (RGB true color), ``(3, H, W)``.
        - ``nir_rgb`` -> ``[3, 2, 1]`` = B08,B04,B03 (false color NIR-R-G), ``(3, H, W)``.
        - ``4band``   -> ``[0, 1, 2, 3]`` = B02,B03,B04,B08 (identity, US-034 compat),
          ``(4, H, W)``.

    The channel dimension is located at ``dim=-3`` so the helper works on both a
    single crop ``(C, H, W)`` and a batch ``(N, C, H, W)`` (the dataset applies it
    per-crop, but a batched caller is supported too).

    Args:
        img: input tensor with the 4-band crop layout; channel dim is the third
            from the end (``(..., C, H, W)``) and must hold at least 4 channels.
        sel: band-selection mode (``rgb``/``nir_rgb``/``4band``).

    Returns:
        Tensor with the channels selected/reordered for ``sel`` (3 channels for
        the baselines, 4 for the four-band identity). The selection never copies
        data beyond ``index_select`` (a view-friendly gather on the channel dim).

    Raises:
        ValueError: if ``sel`` is invalid, ``img`` has fewer than 3 dims, or the
            channel dim carries fewer than 4 channels (R-BANDS: an S2 raster of
            10 bands would silently misalign indices 0-3 to the wrong bands).
    """
    _validate_selection(sel)
    if img.dim() < 3:
        raise ValueError(f"img must be at least 3-D (C, H, W); got shape {tuple(img.shape)}")
    channel_dim = img.dim() - 3
    n_channels = img.shape[channel_dim]
    if n_channels < _MIN_CROP_CHANNELS:
        raise ValueError(
            f"FarSLIP crop must carry >= {_MIN_CROP_CHANNELS} channels in order "
            f"B02,B03,B04,B08; got {n_channels} on channel dim {channel_dim}. "
            "If this is a 10-band Sentinel-2 raster the indices 0-3 do NOT map "
            "to B02-B08 (R-BANDS)."
        )
    index = torch.as_tensor(_SELECTIONS[sel], dtype=torch.long, device=img.device)
    out = img.index_select(channel_dim, index)
    _log.debug(
        "bands selected",
        band_selection=sel,
        indices=_SELECTIONS[sel],
        in_channels=n_channels,
        out_channels=out.shape[channel_dim],
    )
    return out


__all__ = [
    "BandSelection",
    "n_in_channels_for",
    "select_and_reorder_bands",
]
