"""Deterministic CPU tests for the per-parcel crop dataset (US-036-b).

Covers ``ml/farslip/parcel_crop_dataset.py``: the per-parcel crop (bbox of the
instance mask, background zeroed, resize), the ``iter_parcel_crops`` generator,
and the ``collate_parcel_batch`` contract whose ``region_to_patch = arange(B)``
makes the faithful trainer step give each parcel its own CLS.

No disk, no network: synthetic ``composite`` / ``instance`` arrays built in
memory. The full ``ParcelCropDataset`` (which reads PASTIS-R) is exercised only
when the dataset is present, gated by a skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.farslip.parcel_crop_dataset import (
    _MIN_CROP_SIGNAL,
    _crop_parcel,
    collate_parcel_batch,
    iter_parcel_crops,
)

_PASTIS_ROOT = Path("data/PASTIS-R")


def _meadow_dominated_patch(
    h: int = 16, w: int = 16, *, meadow_px: int, soft_wheat_px: int
) -> dict[str, np.ndarray]:
    """Build a synthetic PASTIS-like patch dict with controlled class areas.

    Lays ``meadow_px`` Meadow pixels (class 1) and ``soft_wheat_px`` Soft winter
    wheat pixels (class 2) into the semantic mask, the rest Background (0). The
    instance mask gives Meadow instance id 1 and wheat instance id 2 over their
    pixels so ``extract_regions`` sees two parcels. Used to A/B the 3:1 filter.
    """
    semantic = np.zeros((h, w), dtype=np.int64)
    instance = np.zeros((h, w), dtype=np.int64)
    flat_sem = semantic.ravel()
    flat_inst = instance.ravel()
    flat_sem[:meadow_px] = 1
    flat_inst[:meadow_px] = 1
    flat_sem[meadow_px : meadow_px + soft_wheat_px] = 2
    flat_inst[meadow_px : meadow_px + soft_wheat_px] = 2
    s2 = np.full((1, 10, h, w), 5000, dtype=np.int16)
    return {
        "s2": s2,
        "semantic": flat_sem.reshape(h, w),
        "instance": flat_inst.reshape(h, w),
    }


def _synthetic_patch(h: int = 16, w: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic ``(4, H, W)`` composite + instance mask with 2 parcels.

    Parcel 1 occupies the top-left 4x4 block (value 0.5); parcel 2 the
    bottom-right 6x6 block (value 0.8); the rest is background (no parcel).
    """
    composite = np.zeros((4, h, w), dtype=np.float32)
    instance = np.zeros((h, w), dtype=np.int64)
    instance[0:4, 0:4] = 1
    composite[:, 0:4, 0:4] = 0.5
    instance[h - 6 :, w - 6 :] = 2
    composite[:, h - 6 :, w - 6 :] = 0.8
    return composite, instance


def test_crop_parcel_zeroes_background() -> None:
    """The crop keeps only the parcel's pixels; everything else is exactly 0."""
    composite, instance = _synthetic_patch()
    crop = _crop_parcel(composite, instance, instance_id=1)
    assert crop.shape == (4, 4, 4)
    assert np.allclose(crop, 0.5)  # parcel 1 fills its own bbox entirely


def test_crop_parcel_irregular_zeroes_outside() -> None:
    """A parcel that does not fill its bbox has the outside pixels zeroed."""
    composite = np.full((4, 8, 8), 0.7, dtype=np.float32)
    instance = np.zeros((8, 8), dtype=np.int64)
    # L-shaped parcel inside an 8x8 patch.
    instance[1:5, 1] = 3
    instance[4, 1:5] = 3
    crop = _crop_parcel(composite, instance, instance_id=3)
    # bbox is rows 1..5, cols 1..5 -> (4, 4, 4); only the L cells keep 0.7.
    assert crop.shape == (4, 4, 4)
    n_signal = int((crop[0] > 0).sum())
    assert n_signal == 7  # 4 vertical + 4 horizontal - 1 shared corner


def test_iter_parcel_crops_yields_per_region() -> None:
    """One crop per region, each resized to (4, resize_to, resize_to)."""
    composite, instance = _synthetic_patch()
    regions = [(1, 2), (2, 8)]  # (instance_id, category_id)
    out = list(iter_parcel_crops("10000", composite, instance, regions, resize_to=224))
    assert len(out) == 2
    pid0, cat0, bbox0, crop0 = out[0]
    assert pid0 == "10000_1"
    assert cat0 == 2
    assert crop0.shape == (4, 224, 224)
    assert bbox0 == (0, 0, 4, 4)
    pid1, cat1, _bbox1, crop1 = out[1]
    assert pid1 == "10000_2"
    assert cat1 == 8
    assert crop1.shape == (4, 224, 224)


def test_collate_region_to_patch_is_arange() -> None:
    """The collate sets region_to_patch = arange(B) (identity in the trainer)."""
    items = [
        {
            "image": torch.full((4, 224, 224), 0.5),
            "parcel_id": f"10000_{i}",
            "patch_id": "10000",
            "class_id": 1 + i,
            "caption": f"caption {i}",
            "bbox": (0, 0, 4, 4),
        }
        for i in range(5)
    ]
    batch = collate_parcel_batch(items)
    assert batch["images"].shape == (5, 4, 224, 224)
    assert batch["region_to_patch"].tolist() == [0, 1, 2, 3, 4]
    assert batch["region_cat_ids"].tolist() == [1, 2, 3, 4, 5]
    assert batch["parcel_ids"] == [f"10000_{i}" for i in range(5)]
    assert batch["captions"] == [f"caption {i}" for i in range(5)]


def test_min_crop_signal_threshold_is_small() -> None:
    """The empty-crop threshold is small enough to keep real low-NDVI parcels."""
    assert 0.0 < _MIN_CROP_SIGNAL < 0.01


@pytest.mark.skipif(
    not (_PASTIS_ROOT / "DATA_S2").exists(),
    reason="PASTIS-R not present on disk",
)
def test_dataset_on_real_pastis_drops_empty() -> None:
    """On real PASTIS-R, the dataset yields per-parcel crops with signal > 0."""
    from ml.farslip.parcel_crop_dataset import ParcelCropDataset

    ds = ParcelCropDataset(
        captions={}, folds=(1,), active_class_ids=tuple(range(1, 19)), max_patches=2
    )
    assert len(ds) > 0
    for i in range(min(8, len(ds))):
        item = ds[i]
        assert item["image"].shape == (4, 224, 224)
        assert float(item["image"].max()) > _MIN_CROP_SIGNAL
        assert "_" in item["parcel_id"]
        assert 1 <= item["class_id"] <= 18


@pytest.mark.skipif(
    not (_PASTIS_ROOT / "DATA_S2").exists(),
    reason="PASTIS-R not present on disk",
)
def test_require_caption_keeps_only_captioned_parcels() -> None:
    """``require_caption=True`` keeps exactly the parcels with a non-empty caption.

    Builds the full (uncaptioned) dataset first to learn real parcel ids, then
    rebuilds with a caption map covering only a subset and ``require_caption=True``.
    The captioned dataset must be a non-empty subset whose every sample has a
    caption. This guards the balanced-sample sweep from training ``L_glo`` on
    empty-caption parcels.
    """
    from ml.farslip.parcel_crop_dataset import ParcelCropDataset

    full = ParcelCropDataset(
        captions={}, folds=(1,), active_class_ids=tuple(range(1, 19)), max_patches=2
    )
    assert len(full) >= 2
    # Caption only the first half of the discovered parcels.
    all_ids = [full[i]["parcel_id"] for i in range(len(full))]
    keep = all_ids[: max(1, len(all_ids) // 2)]
    captions = {pid: f"fenologia sintetica {pid}" for pid in keep}

    captioned = ParcelCropDataset(
        captions=captions,
        folds=(1,),
        active_class_ids=tuple(range(1, 19)),
        max_patches=2,
        require_caption=True,
    )
    assert 0 < len(captioned) <= len(full)
    seen = {captioned[i]["parcel_id"] for i in range(len(captioned))}
    assert seen <= set(keep)
    for i in range(len(captioned)):
        assert captioned[i]["caption"] != ""


def test_dominance_filter_drops_meadow_dominated_patch(monkeypatch) -> None:
    """``dominance_ratio=3.0`` drops a patch where Meadow > 3x the 2nd class.

    No disk: stubs ``_fold_patch_ids`` to two synthetic patches and
    ``load_pastis_patch`` to return them. Patch "A" is 4:1 Meadow (dropped at
    ratio 3.0); patch "B" is 2:1 Meadow (kept). With the filter on, only B's
    parcels survive; without it (None), both patches' parcels are kept. This is
    the A/B knob: the parcel grain keeps rare parcels by default, the filter
    re-imposes the legacy patch-level imbalance guard.
    """
    from ml.farslip import parcel_crop_dataset as mod

    patch_a = _meadow_dominated_patch(meadow_px=40, soft_wheat_px=10)  # 4:1 -> drop
    patch_b = _meadow_dominated_patch(meadow_px=20, soft_wheat_px=10)  # 2:1 -> keep
    patches = {"A": patch_a, "B": patch_b}

    monkeypatch.setattr(mod.ParcelCropDataset, "_fold_patch_ids", lambda self: ["A", "B"])
    monkeypatch.setattr(mod, "load_pastis_patch", lambda pid, **kw: patches[pid])

    active = (1, 2)  # Meadow + Soft winter wheat
    unfiltered = mod.ParcelCropDataset(
        captions={},
        folds=(1,),
        active_class_ids=active,
        min_area_px=1,
    )
    filtered = mod.ParcelCropDataset(
        captions={},
        folds=(1,),
        active_class_ids=active,
        min_area_px=1,
        dominance_ratio=3.0,
    )

    patches_unfiltered = {unfiltered[i]["patch_id"] for i in range(len(unfiltered))}
    patches_filtered = {filtered[i]["patch_id"] for i in range(len(filtered))}
    assert patches_unfiltered == {"A", "B"}
    assert patches_filtered == {"B"}  # A (4:1) dropped, B (2:1) kept
    assert len(filtered) < len(unfiltered)


def test_dominance_filter_none_is_noop(monkeypatch) -> None:
    """``dominance_ratio=None`` keeps every patch (backward-compatible default)."""
    from ml.farslip import parcel_crop_dataset as mod

    patch_a = _meadow_dominated_patch(meadow_px=90, soft_wheat_px=10)  # 9:1
    monkeypatch.setattr(mod.ParcelCropDataset, "_fold_patch_ids", lambda self: ["A"])
    monkeypatch.setattr(mod, "load_pastis_patch", lambda pid, **kw: patch_a)

    ds = mod.ParcelCropDataset(
        captions={},
        folds=(1,),
        active_class_ids=(1, 2),
        min_area_px=1,
        dominance_ratio=None,
    )
    # Even a 9:1 Meadow patch is kept when the filter is off.
    assert {ds[i]["patch_id"] for i in range(len(ds))} == {"A"}
    assert ds.dominance_ratio is None


def test_dominance_filter_matches_pastis_filter(monkeypatch) -> None:
    """The dataset's keep decision is byte-for-byte ``PastisFilter._passes_dominance``.

    For each synthetic patch, the dataset must keep it iff the standalone
    ``PastisFilter`` (same ratio, same active classes as target_classes) keeps it.
    This proves no reimplementation drift between the patch-level filter and the
    parcel-level A/B path.
    """
    from ml.data.pastis_filter import PastisFilter
    from ml.farslip import parcel_crop_dataset as mod

    cases = {
        "p_3to1": _meadow_dominated_patch(meadow_px=30, soft_wheat_px=10),  # 3:1 keep
        "p_4to1": _meadow_dominated_patch(meadow_px=40, soft_wheat_px=10),  # 4:1 drop
        "p_nomeadow": _meadow_dominated_patch(meadow_px=0, soft_wheat_px=10),  # keep
    }
    active = (1, 2)
    # _passes_dominance only needs these three attrs; build the filter without
    # __init__ so the test does not require a real metadata.geojson on disk.
    filt = object.__new__(PastisFilter)
    filt.target_classes = set(active)
    filt.ratio = 3.0
    filt.meadow_class = 1
    expected_keep = {pid for pid, p in cases.items() if filt._passes_dominance(p["semantic"])[0]}

    monkeypatch.setattr(mod.ParcelCropDataset, "_fold_patch_ids", lambda self: list(cases))
    monkeypatch.setattr(mod, "load_pastis_patch", lambda pid, **kw: cases[pid])

    ds = mod.ParcelCropDataset(
        captions={},
        folds=(1,),
        active_class_ids=active,
        min_area_px=1,
        dominance_ratio=3.0,
    )
    kept = {ds[i]["patch_id"] for i in range(len(ds))}
    assert kept == expected_keep
    assert "p_3to1" in kept and "p_nomeadow" in kept and "p_4to1" not in kept
