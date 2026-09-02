"""Tests for the US-035 band-selection / ablation helper.

The FarSLIP crops carry FOUR Sentinel-2 bands in the fixed layout B02=0, B03=1,
B04=2, B08=3. US-035 slices/reorders them into three variants:

    - ``rgb``     -> [2, 1, 0] = B04,B03,B02 (3 channels),
    - ``nir_rgb`` -> [3, 2, 1] = B08,B04,B03 (3 channels),
    - ``4band``   -> [0, 1, 2, 3] = B02,B03,B04,B08 (4 channels, identity).

The CRITICAL test is the reference-parcel indexing check (R-BANDS): a 10-band
Sentinel-2 raster would silently misalign indices 0-3, so we assert both the
exact per-channel mapping and the >= 4-channel contract. The indexing tests are
pure CPU slicing (no network, no GPU, no real dataset). The patch_embed
adaptation test reuses the 3->4 mean-RGB machinery of US-034 and is skipped when
CLIP is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from ml.farslip.bands import (
    _BAND_INDEX,
    _N_CHANNELS,
    _SELECTIONS,
    n_in_channels_for,
    select_and_reorder_bands,
)


def _marker_crop(n_channels: int = 4, hw: int = 8) -> torch.Tensor:
    """Build a ``(n_channels, hw, hw)`` crop where every pixel of channel ``c``
    holds the constant value ``c`` (so an output channel's value reveals which
    input channel it came from)."""
    base = torch.arange(n_channels, dtype=torch.float32).view(n_channels, 1, 1)
    return base.expand(n_channels, hw, hw).clone()


# ---------------------------------------------------------------------------
# Indexing per variant (AC-2).
# ---------------------------------------------------------------------------


def test_select_rgb_order() -> None:
    img = _marker_crop()
    out = select_and_reorder_bands(img, "rgb")
    assert out.shape == (3, 8, 8)
    # rgb -> [2, 1, 0] = B04, B03, B02.
    assert [int(out[c, 0, 0]) for c in range(3)] == [2, 1, 0]
    assert torch.equal(out, img[[2, 1, 0]])


def test_select_nir_rgb_order() -> None:
    img = _marker_crop()
    out = select_and_reorder_bands(img, "nir_rgb")
    assert out.shape == (3, 8, 8)
    # nir_rgb -> [3, 2, 1] = B08, B04, B03.
    assert [int(out[c, 0, 0]) for c in range(3)] == [3, 2, 1]
    assert torch.equal(out, img[[3, 2, 1]])


def test_select_4band_identity() -> None:
    img = _marker_crop()
    out = select_and_reorder_bands(img, "4band")
    assert out.shape == (4, 8, 8)
    # 4band -> [0, 1, 2, 3] identity (US-034 checkpoint compat).
    assert [int(out[c, 0, 0]) for c in range(4)] == [0, 1, 2, 3]
    assert torch.equal(out, img[:4])
    assert torch.equal(out, img)  # the marker crop has exactly 4 channels


def test_select_batched_input() -> None:
    # The channel dim is dim=-3, so a batched (N, C, H, W) crop is supported too.
    batch = _marker_crop().unsqueeze(0).expand(5, 4, 8, 8).clone()
    out = select_and_reorder_bands(batch, "rgb")
    assert out.shape == (5, 3, 8, 8)
    assert [int(out[0, c, 0, 0]) for c in range(3)] == [2, 1, 0]


# ---------------------------------------------------------------------------
# n_in_channels derived from band_selection (AC-3, single source of truth).
# ---------------------------------------------------------------------------


def test_n_in_channels_for() -> None:
    assert n_in_channels_for("rgb") == 3
    assert n_in_channels_for("nir_rgb") == 3
    assert n_in_channels_for("4band") == 4


def test_n_channels_table_matches_selection_lengths() -> None:
    # _N_CHANNELS must agree with the number of indices each selection yields.
    for sel, indices in _SELECTIONS.items():
        assert _N_CHANNELS[sel] == len(indices)


def test_n_in_channels_for_invalid_raises() -> None:
    with pytest.raises(ValueError, match="invalid band_selection"):
        n_in_channels_for("rgbn")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reference-parcel layout: mismatch 4 vs 10 bands (AC-7, R-BANDS — critical).
# ---------------------------------------------------------------------------


def test_band_index_layout_is_4band_b02_b08() -> None:
    # The crop layout contract: B02=0, B03=1, B04=2, B08=3 (dataset.py:1-6).
    assert _BAND_INDEX == {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
    # A marker crop lets us assert each named band index points at its band.
    img = _marker_crop()
    for band, idx in _BAND_INDEX.items():
        assert int(img[idx, 0, 0]) == idx, f"{band} expected at channel {idx}"
    # rgb selects exactly B04, B03, B02 (true color) from this layout.
    rgb = select_and_reorder_bands(img, "rgb")
    assert int(rgb[0, 0, 0]) == _BAND_INDEX["B04"]
    assert int(rgb[1, 0, 0]) == _BAND_INDEX["B03"]
    assert int(rgb[2, 0, 0]) == _BAND_INDEX["B02"]


def test_select_rejects_fewer_than_4_channels() -> None:
    # A 3-band raster (or anything < 4) is rejected: indices 0-3 of a non-4-band
    # crop do NOT map to B02-B08, so we fail fast instead of misaligning.
    img = _marker_crop(n_channels=3)
    with pytest.raises(ValueError, match=">= 4 channels"):
        select_and_reorder_bands(img, "rgb")


def test_select_accepts_more_than_4_channels_but_indexes_first_4() -> None:
    # Defensive: if a 10-band S2 raster slipped in, the >= 4 assert passes but
    # the documented contract is that indices 0-3 are B02-B08. We only verify
    # that the helper indexes the declared positions (the assert above is the
    # primary guard; the contract is documented in bands.py).
    img = _marker_crop(n_channels=10)
    out = select_and_reorder_bands(img, "4band")
    assert out.shape == (4, 8, 8)
    assert [int(out[c, 0, 0]) for c in range(4)] == [0, 1, 2, 3]


def test_select_rejects_non_chw() -> None:
    with pytest.raises(ValueError, match="at least 3-D"):
        select_and_reorder_bands(torch.zeros(4, 8), "rgb")


def test_select_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="invalid band_selection"):
        select_and_reorder_bands(_marker_crop(), "false_color")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# patch_embed in_channels per variant (AC-3/AC-4/AC-8) — reuses US-034 adapter.
# ---------------------------------------------------------------------------


def _clip_available() -> bool:
    try:
        from transformers import CLIPVisionModel  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _build_student() -> torch.nn.Module | None:
    """Deep-clone a CLIP vision model to a 3-channel student, or ``None``."""
    if not _clip_available():
        return None
    import copy

    from transformers import CLIPVisionModel

    try:
        teacher = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    except (OSError, RuntimeError, ValueError):
        return None
    return copy.deepcopy(teacher)


@pytest.mark.skipif(not _clip_available(), reason="transformers/CLIP unavailable")
@pytest.mark.parametrize(
    ("band_selection", "expected_in"),
    [("rgb", 3), ("nir_rgb", 3), ("4band", 4)],
)
def test_patch_embed_channels_match_variant(band_selection: str, expected_in: int) -> None:
    from ml.farslip.distill import adapt_patch_embed_to_n_channels

    student = _build_student()
    if student is None:
        pytest.skip("could not load CLIP weights")
    target = n_in_channels_for(band_selection)  # type: ignore[arg-type]
    assert target == expected_in
    adapt_patch_embed_to_n_channels(student, target)
    pe = student.embeddings.patch_embedding  # type: ignore[union-attr]
    assert pe.in_channels == expected_in
    # forward with (B, target, 224, 224) must not break.
    out = student(pixel_values=torch.randn(1, expected_in, 224, 224))
    assert out.last_hidden_state.shape[0] == 1


@pytest.mark.skipif(not _clip_available(), reason="transformers/CLIP unavailable")
def test_4band_nir_channel_is_mean_rgb() -> None:
    from ml.farslip.distill import adapt_patch_embed_to_n_channels

    student = _build_student()
    if student is None:
        pytest.skip("could not load CLIP weights")
    rgb_weight = student.embeddings.patch_embedding.weight.detach().clone()  # type: ignore[union-attr]
    adapt_patch_embed_to_n_channels(student, 4)
    pe = student.embeddings.patch_embedding  # type: ignore[union-attr]
    # First 3 channels copied verbatim; 4th (NIR) = mean of the 3 RGB (no dead).
    assert torch.allclose(pe.weight[:, :3], rgb_weight, atol=1e-9)
    expected_nir = rgb_weight.mean(dim=1)
    assert torch.allclose(pe.weight[:, 3], expected_nir, atol=1e-6)
