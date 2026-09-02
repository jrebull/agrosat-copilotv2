"""Deterministic CPU tests for the US-036-a v2 region-category dataset (T2).

Covers ``ml/farslip/region_category_dataset.py``: the multi-object region
extraction (N > 1 per patch via ``ParcelIDs`` + majority class), the area and
bg/void filters, the dataset construction (caption injection, regions
precomputed, ``mean_regions_per_patch > 1``), the ``__getitem__`` contract, the
cross-patch flattening of :func:`collate_region_batch`, and the spatial-CV
anti-leakage guard ``assert_disjoint_folds``.

No disk, no network, no Gemma: ``load_pastis_patch`` is monkeypatched on the
module under test with synthetic ``s2`` / ``semantic`` / ``instance`` arrays,
and the ``PastisFilter`` fold map is monkeypatched so ``metadata.geojson`` is
never required. Captions are an injected ``dict[patch_id, caption]`` (the dataset
never calls a model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import ml.farslip.region_category_dataset as rcd
from ml.farslip.region_category_dataset import (
    RegionCategoryPairDataset,
    assert_disjoint_folds,
    collate_region_batch,
    extract_regions,
)

_RESIZE = 224


# ---------------------------------------------------------------------------
# Synthetic-patch helpers (tests only; never imported by production).
# ---------------------------------------------------------------------------


def _make_s2(ndvi_per_t: list[float], h: int = 16, w: int = 16) -> np.ndarray:
    """Builds a synthetic ``(T, 10, H, W)`` int16 patch with a target NDVI per t.

    Mirrors the v1 helper so ``peak_ndvi_composite`` selects the expected t* and
    the composite stays in ``[0, 1]``.

    Args:
        ndvi_per_t: target spatial-mean NDVI for each timestep.
        h: patch height.
        w: patch width.

    Returns:
        int16 array ``(len(ndvi_per_t), 10, h, w)``.
    """
    t = len(ndvi_per_t)
    s2 = np.zeros((t, 10, h, w), dtype=np.int16)
    red = 1000.0
    for ti, ndvi in enumerate(ndvi_per_t):
        nir = red * (1.0 + ndvi) / (1.0 - ndvi)
        s2[ti, rcd_pastis_b("B02")] = 200
        s2[ti, rcd_pastis_b("B03")] = 400
        s2[ti, rcd_pastis_b("B04")] = int(red)
        s2[ti, rcd_pastis_b("B08")] = round(nir)
    return s2


def rcd_pastis_b(band: str) -> int:
    """Maps a band name to its index in the v1 10-band layout (test helper)."""
    from ml.farslip import pastis_pair_dataset as ppd

    return {
        "B02": ppd._PASTIS_B02,
        "B03": ppd._PASTIS_B03,
        "B04": ppd._PASTIS_B04,
        "B08": ppd._PASTIS_B08,
    }[band]


def _make_panoptic(
    parcels: dict[int, tuple[int, int]],
    h: int = 16,
    w: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds synthetic ``(parcel_ids, semantic)`` masks from a parcel spec.

    Lays out parcels sequentially over the flattened grid. Each parcel is a
    contiguous block of ``area`` pixels all tagged with the same instance id and
    the same semantic class (PASTIS parcels are monoculture). Leftover pixels are
    Background (instance 0, class 0).

    Args:
        parcels: ``{instance_id: (category_id, area_px)}``.
        h: mask height.
        w: mask width.

    Returns:
        Tuple ``(parcel_ids (h, w) int32, semantic (h, w) int32)``.
    """
    inst = np.zeros(h * w, dtype=np.int32)
    sem = np.zeros(h * w, dtype=np.int32)
    pos = 0
    for instance_id, (category_id, area) in parcels.items():
        inst[pos : pos + area] = instance_id
        sem[pos : pos + area] = category_id
        pos += area
    return inst.reshape(h, w), sem.reshape(h, w)


def _patch_construction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    patches: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    fold_map: dict[int, list[int]],
) -> None:
    """Monkeypatches ``load_pastis_patch`` and the ``PastisFilter`` fold map.

    Args:
        monkeypatch: pytest fixture.
        patches: ``{pid: (s2, parcel_ids, semantic)}`` synthetic patches.
        fold_map: ``{fold: [pid, ...]}`` for the non-dominance path.
    """

    def _fake_loader(
        patch_id: object,
        root: object = None,
        load_annotations: bool = True,
    ) -> dict[str, object]:
        s2, parcel_ids, semantic = patches[int(patch_id)]
        return {
            "s2": s2,
            "semantic": semantic if load_annotations else None,
            "instance": parcel_ids if load_annotations else None,
        }

    monkeypatch.setattr(rcd, "load_pastis_patch", _fake_loader)

    class _FakeFilter:
        """Stand-in for ``PastisFilter`` exposing only ``_fold_map``."""

        def __init__(self, **kwargs: object) -> None:
            self._fold_map = fold_map

        def filter_folds(self, folds: object) -> list[int]:
            ids: list[int] = []
            for fold in folds:  # type: ignore[union-attr]
                ids.extend(self._fold_map.get(int(fold), []))
            return ids

    monkeypatch.setattr("ml.data.pastis_filter.PastisFilter", _FakeFilter)


# ---------------------------------------------------------------------------
# 1. extract_regions: multi-object (N > 1) + majority class.
# ---------------------------------------------------------------------------


def test_extract_regions_multi_object() -> None:
    # 3 parcels of 2 distinct classes -> 3 region-category entries (NOT 1).
    parcels, semantic = _make_panoptic({1: (3, 40), 2: (8, 30), 3: (3, 20)})
    regions = extract_regions(parcels, semantic)
    assert len(regions) == 3
    cats = sorted(cat for _, cat in regions)
    assert cats == [3, 3, 8]
    # Deterministic order by instance id.
    assert [inst for inst, _ in regions] == [1, 2, 3]


def test_region_category_uses_majority_class() -> None:
    # Build an instance with mixed pixels: class 5 majority (30) over class 7 (10).
    h = w = 16
    inst = np.zeros(h * w, dtype=np.int32)
    sem = np.zeros(h * w, dtype=np.int32)
    inst[0:40] = 1
    sem[0:30] = 5
    sem[30:40] = 7
    regions = extract_regions(inst.reshape(h, w), sem.reshape(h, w))
    assert regions == [(1, 5)]


def test_extract_regions_drops_background_and_void() -> None:
    # An instance whose majority class is Background (0) or Void (19) is dropped.
    parcels, semantic = _make_panoptic({1: (0, 40), 2: (19, 30), 3: (6, 25)})
    regions = extract_regions(parcels, semantic)
    assert regions == [(3, 6)]


def test_min_area_filter() -> None:
    # Parcel 2 has 8 px < min_area_px=16 -> excluded; parcel 1 (40 px) kept.
    parcels, semantic = _make_panoptic({1: (3, 40), 2: (8, 8)})
    regions = extract_regions(parcels, semantic, min_area_px=16)
    assert regions == [(1, 3)]
    # With a permissive threshold both survive.
    regions_all = extract_regions(parcels, semantic, min_area_px=4)
    assert sorted(c for _, c in regions_all) == [3, 8]


def test_extract_regions_respects_active_class_ids() -> None:
    # category 8 is excluded from active_class_ids -> its region is dropped.
    parcels, semantic = _make_panoptic({1: (3, 40), 2: (8, 30)})
    regions = extract_regions(parcels, semantic, active_class_ids=(1, 2, 3))
    assert regions == [(1, 3)]


def test_extract_regions_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        extract_regions(np.zeros((4, 4), dtype=np.int32), np.zeros((8, 8)))


# ---------------------------------------------------------------------------
# 2. assert_disjoint_folds: spatial-CV anti-leakage.
# ---------------------------------------------------------------------------


def test_assert_disjoint_folds_ok() -> None:
    assert_disjoint_folds((1, 2, 3), (4,))  # no raise


def test_assert_disjoint_folds_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_folds((1, 2, 3), (3, 4))


# ---------------------------------------------------------------------------
# 3. Dataset construction: multi-object, mean_regions_per_patch > 1.
# ---------------------------------------------------------------------------


def test_dataset_mean_regions_per_patch_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2 = _make_s2([0.6])
    # pid 100: 3 parcels (2 classes); pid 200: 2 parcels (2 classes).
    p100, s100 = _make_panoptic({1: (3, 40), 2: (8, 30), 3: (3, 20)})
    p200, s200 = _make_panoptic({1: (5, 40), 2: (6, 30)})
    patches = {100: (s2, p100, s100), 200: (s2, p200, s200)}
    _patch_construction(monkeypatch, patches=patches, fold_map={1: [100, 200]})

    ds = RegionCategoryPairDataset(
        captions={"100": "escena", "200": "escena"},
        root=Path("unused"),
        folds=(1,),
    )
    assert len(ds) == 2
    # 3 + 2 = 5 regions over 2 patches -> mean 2.5 > 1.
    assert ds.mean_regions_per_patch == pytest.approx(2.5)
    assert ds.mean_regions_per_patch > 1.0


def test_dataset_drops_patch_with_no_valid_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2 = _make_s2([0.6])
    p_ok, s_ok = _make_panoptic({1: (3, 40), 2: (8, 30)})
    # pid 300: only background / tiny slivers -> zero valid regions -> dropped.
    p_bad, s_bad = _make_panoptic({1: (0, 40), 2: (3, 4)})
    patches = {100: (s2, p_ok, s_ok), 300: (s2, p_bad, s_bad)}
    _patch_construction(monkeypatch, patches=patches, fold_map={1: [100, 300]})

    ds = RegionCategoryPairDataset(
        captions={"100": "escena", "300": "escena"},
        root=Path("unused"),
        folds=(1,),
        min_area_px=16,
    )
    kept = {pid for pid, _ in ds._samples}
    assert kept == {"100"}


def test_dataset_skips_patch_without_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pid 400 has no semantic / instance annotation -> skipped at construction.
    s2 = _make_s2([0.6])
    p_ok, s_ok = _make_panoptic({1: (3, 40), 2: (8, 30)})
    patches = {
        100: (s2, p_ok, s_ok),
        400: (s2, None, None),
    }

    def _fake_loader(
        patch_id: object, root: object = None, load_annotations: bool = True
    ) -> dict[str, object]:
        s2_arr, parcel_ids, semantic = patches[int(patch_id)]
        return {
            "s2": s2_arr,
            "semantic": semantic if load_annotations else None,
            "instance": parcel_ids if load_annotations else None,
        }

    monkeypatch.setattr(rcd, "load_pastis_patch", _fake_loader)

    class _FakeFilter:
        def __init__(self, **kwargs: object) -> None:
            self._fold_map = {1: [100, 400]}

        def filter_folds(self, folds: object) -> list[int]:
            return [100, 400]

    monkeypatch.setattr("ml.data.pastis_filter.PastisFilter", _FakeFilter)

    ds = RegionCategoryPairDataset(captions={"100": "escena"}, root=Path("unused"), folds=(1,))
    assert {pid for pid, _ in ds._samples} == {"100"}


# ---------------------------------------------------------------------------
# 4. __getitem__ contract.
# ---------------------------------------------------------------------------


def test_getitem_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    s2 = _make_s2([0.3, 0.8])
    p100, s100 = _make_panoptic({1: (3, 40), 2: (8, 30), 3: (3, 20)})
    patches = {100: (s2, p100, s100)}
    _patch_construction(monkeypatch, patches=patches, fold_map={1: [100]})

    ds = RegionCategoryPairDataset(
        captions={"100": "imagen satelital con varias parcelas"},
        root=Path("unused"),
        folds=(1,),
    )
    item = ds[0]
    assert set(item) == {"image", "patch_id", "caption", "region_cat_ids"}
    assert item["image"].shape == (4, _RESIZE, _RESIZE)
    assert item["image"].dtype == torch.float32
    assert float(item["image"].min()) >= 0.0
    assert float(item["image"].max()) <= 1.0
    assert item["patch_id"] == "100"
    assert item["caption"] == "imagen satelital con varias parcelas"
    assert item["region_cat_ids"].dtype == torch.long
    # 3 parcels -> 3 region categories (multi-object).
    assert item["region_cat_ids"].tolist() == [3, 8, 3]


def test_getitem_negative_and_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    s2 = _make_s2([0.6])
    p100, s100 = _make_panoptic({1: (3, 40), 2: (8, 30)})
    patches = {100: (s2, p100, s100)}
    _patch_construction(monkeypatch, patches=patches, fold_map={1: [100]})

    ds = RegionCategoryPairDataset(captions={"100": "escena"}, root=Path("unused"), folds=(1,))
    assert ds[-1]["patch_id"] == "100"
    with pytest.raises(IndexError, match="idx out of range"):
        _ = ds[5]


def test_getitem_requires_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    # The patch survives construction (valid regions) but has NO caption -> the
    # dataset fails explicitly at access time, never silently.
    s2 = _make_s2([0.6])
    p100, s100 = _make_panoptic({1: (3, 40), 2: (8, 30)})
    patches = {100: (s2, p100, s100)}
    _patch_construction(monkeypatch, patches=patches, fold_map={1: [100]})

    ds = RegionCategoryPairDataset(captions={"999": "otra"}, root=Path("unused"), folds=(1,))
    with pytest.raises(KeyError, match="no caption for patch_id 100"):
        _ = ds[0]


# ---------------------------------------------------------------------------
# 5. collate_region_batch: cross-patch flattening.
# ---------------------------------------------------------------------------


def test_collate_flattens_regions() -> None:
    # B=2 patches with (2, 3) regions -> flat batch of 5 regions + region_to_patch.
    item_a = {
        "image": torch.zeros(4, _RESIZE, _RESIZE),
        "patch_id": "100",
        "caption": "a",
        "region_cat_ids": torch.tensor([3, 8], dtype=torch.long),
    }
    item_b = {
        "image": torch.ones(4, _RESIZE, _RESIZE),
        "patch_id": "200",
        "caption": "b",
        "region_cat_ids": torch.tensor([5, 6, 3], dtype=torch.long),
    }
    batch = collate_region_batch([item_a, item_b])

    assert batch["images"].shape == (2, 4, _RESIZE, _RESIZE)
    assert batch["patch_ids"] == ["100", "200"]
    assert batch["captions"] == ["a", "b"]
    # 2 + 3 = 5 flattened regions.
    assert batch["region_cat_ids"].tolist() == [3, 8, 5, 6, 3]
    # First 2 regions belong to patch 0, last 3 to patch 1 (cross-patch index).
    assert batch["region_to_patch"].tolist() == [0, 0, 1, 1, 1]
    assert batch["region_cat_ids"].dtype == torch.long
    assert batch["region_to_patch"].dtype == torch.long


def test_collate_shared_category_is_cross_patch() -> None:
    # Category 3 appears in BOTH patches -> after flattening, the two regions of
    # category 3 are at positions whose region_to_patch differ (0 and 1). This is
    # the cross-patch positive pair the MPCL (T3) will group via P(i).
    item_a = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "100",
        "caption": "a",
        "region_cat_ids": torch.tensor([3], dtype=torch.long),
    }
    item_b = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "200",
        "caption": "b",
        "region_cat_ids": torch.tensor([3], dtype=torch.long),
    }
    batch = collate_region_batch([item_a, item_b])
    cat3_positions = (batch["region_cat_ids"] == 3).nonzero().flatten().tolist()
    patches_of_cat3 = {int(batch["region_to_patch"][p]) for p in cat3_positions}
    assert patches_of_cat3 == {0, 1}


def test_collate_handles_empty_region_patch() -> None:
    # A patch with zero regions contributes nothing to the flat region axis but
    # still occupies an image slot (B unchanged).
    item_a = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "100",
        "caption": "a",
        "region_cat_ids": torch.empty((0,), dtype=torch.long),
    }
    item_b = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "200",
        "caption": "b",
        "region_cat_ids": torch.tensor([7], dtype=torch.long),
    }
    batch = collate_region_batch([item_a, item_b])
    assert batch["images"].shape[0] == 2
    assert batch["region_cat_ids"].tolist() == [7]
    assert batch["region_to_patch"].tolist() == [1]


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_region_batch([])


def test_collate_all_patches_empty_regions() -> None:
    # Both patches contribute zero regions -> empty region axis, B preserved.
    item_a = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "100",
        "caption": "a",
        "region_cat_ids": torch.empty((0,), dtype=torch.long),
    }
    item_b = {
        "image": torch.zeros(4, 4, 4),
        "patch_id": "200",
        "caption": "b",
        "region_cat_ids": torch.empty((0,), dtype=torch.long),
    }
    batch = collate_region_batch([item_a, item_b])
    assert batch["images"].shape[0] == 2
    assert batch["region_cat_ids"].numel() == 0
    assert batch["region_to_patch"].numel() == 0
    assert batch["region_cat_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# 6. Dominance filter path (optional 3:1) + config validation.
# ---------------------------------------------------------------------------


def test_dataset_dominance_ratio_uses_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2 = _make_s2([0.6])
    p100, s100 = _make_panoptic({1: (3, 40), 2: (8, 30)})
    p200, s200 = _make_panoptic({1: (5, 40), 2: (6, 30)})
    patches = {100: (s2, p100, s100), 200: (s2, p200, s200)}

    captured: dict[str, object] = {}

    class _FilterSpy:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self._fold_map = {1: [100, 200]}

        def filter_folds(self, folds: object) -> list[int]:
            # Pretend the 3:1 rule drops pid 200.
            return [100]

    monkeypatch.setattr("ml.data.pastis_filter.PastisFilter", _FilterSpy)

    def _fake_loader(
        patch_id: object, root: object = None, load_annotations: bool = True
    ) -> dict[str, object]:
        s2_arr, parcel_ids, semantic = patches[int(patch_id)]
        return {
            "s2": s2_arr,
            "semantic": semantic if load_annotations else None,
            "instance": parcel_ids if load_annotations else None,
        }

    monkeypatch.setattr(rcd, "load_pastis_patch", _fake_loader)

    ds = RegionCategoryPairDataset(
        captions={"100": "x", "200": "y"},
        root=Path("unused"),
        folds=(1,),
        dominance_ratio=3.0,
    )
    assert {pid for pid, _ in ds._samples} == {"100"}
    assert captured.get("mode") == "dominance_ratio"
    assert captured.get("ratio") == 3.0


def test_dataset_rejects_bad_active_class_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_construction(monkeypatch, patches={}, fold_map={1: []})
    with pytest.raises(ValueError, match=r"\[1, 18\]"):
        RegionCategoryPairDataset(
            captions={},
            root=Path("unused"),
            folds=(1,),
            active_class_ids=(0, 3),
        )
