"""Deterministic CPU tests for the US-036 PASTIS pair dataset builder.

Covers ``ml/farslip/pastis_pair_dataset.py`` (ml/A write-set): the peak-NDVI
composite, the active-restricted dominant class, the cardinality curriculum, the
3:1 Meadow filter applied through the dataset, ``create_incremental_dataset``
shapes / DIRECT prototype rows, and the ``__getitem__`` contract.

No disk, no network: ``load_pastis_patch`` and ``PastisFilter`` are monkeypatched
on the module under test with synthetic ``s2``/``semantic`` arrays, and
``load_class_prototype_embeddings`` is mocked with a deterministic ``(18, 384)``
matrix (seed 42) so the US-033 parquet is never required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import ml.farslip.pastis_pair_dataset as ppd
from ml.farslip.pastis_pair_dataset import (
    INCREMENTAL_CURRICULUM,
    active_classes,
    create_incremental_dataset,
    dominant_class,
    peak_ndvi_composite,
)

_RESIZE = 224
_PROTO_DIM = 384


# ---------------------------------------------------------------------------
# Synthetic-patch helpers (tests only; never imported by production).
# ---------------------------------------------------------------------------


def _make_s2(ndvi_per_t: list[float], h: int = 8, w: int = 8) -> np.ndarray:
    """Builds a synthetic ``(T, 10, H, W)`` int16 patch with a target NDVI per t.

    For a constant red (B04) and a chosen NDVI, the NIR (B08) is
    ``nir = red * (1 + ndvi) / (1 - ndvi)`` so that ``(nir-red)/(nir+red)`` is
    exactly ``ndvi``. Other bands carry a distinguishable constant.

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
        s2[ti, ppd._PASTIS_B02] = 200  # B02
        s2[ti, ppd._PASTIS_B03] = 400  # B03
        s2[ti, ppd._PASTIS_B04] = int(red)  # B04 (red)
        s2[ti, ppd._PASTIS_B08] = round(nir)  # B08 (nir)
    return s2


def _make_semantic(counts: dict[int, int], h: int = 8, w: int = 8) -> np.ndarray:
    """Builds a ``(H, W)`` semantic mask with a known per-class pixel histogram.

    Fills the flattened mask sequentially with ``count`` pixels per class_id;
    any leftover pixels become Background (0).

    Args:
        counts: ``{class_id: n_pixels}`` to place.
        h: mask height.
        w: mask width.

    Returns:
        uint8 mask ``(h, w)``.
    """
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for cid, n in counts.items():
        flat[pos : pos + n] = cid
        pos += n
    return flat.reshape(h, w)


def _fake_proto_18() -> tuple[np.ndarray, list[int]]:
    """Deterministic ``(18, 384)`` prototypes (seed 42) + curriculum class_ids.

    Row r is a one-hot-ish distinct vector so a wrong row selection is
    detectable; the returned class_ids are intentionally NOT sorted (curriculum
    order) to assert the loader maps class_id -> row, not row == class_id-1.
    """
    rng = np.random.default_rng(42)
    proto = rng.standard_normal((18, _PROTO_DIM)).astype(np.float32)
    # class_ids in curriculum order so row r corresponds to curriculum[r].
    class_ids = list(INCREMENTAL_CURRICULUM)
    return proto, class_ids


# ---------------------------------------------------------------------------
# 1. peak_ndvi_composite: argmax timestep + shape/bands/range.
# ---------------------------------------------------------------------------


def test_peak_ndvi_composite_picks_argmax() -> None:
    # Timestep 2 has the highest mean NDVI -> it must be selected. NDVI 0.6 keeps
    # nir (=red*1.6/0.4=4000) within the 0..10000 range so the /10000 scaling is
    # exact and not clipped.
    s2 = _make_s2([0.1, 0.4, 0.6, 0.3])
    composite = peak_ndvi_composite(s2)

    assert composite.shape == (4, 8, 8)
    assert composite.dtype == np.float32
    assert float(composite.min()) >= 0.0
    assert float(composite.max()) <= 1.0

    # Bands are [B02, B03, B04, B08] / 10000 at t* = 2.
    t_star = 2
    expected_b02 = s2[t_star, ppd._PASTIS_B02, 0, 0] / 10000.0
    expected_b03 = s2[t_star, ppd._PASTIS_B03, 0, 0] / 10000.0
    expected_b04 = s2[t_star, ppd._PASTIS_B04, 0, 0] / 10000.0
    expected_b08 = s2[t_star, ppd._PASTIS_B08, 0, 0] / 10000.0
    assert composite[0, 0, 0] == pytest.approx(expected_b02, abs=1e-4)
    assert composite[1, 0, 0] == pytest.approx(expected_b03, abs=1e-4)
    assert composite[2, 0, 0] == pytest.approx(expected_b04, abs=1e-4)
    assert composite[3, 0, 0] == pytest.approx(expected_b08, abs=1e-4)


def test_peak_ndvi_composite_clamps_invalid_ndvi() -> None:
    # An out-of-range / saturated timestep must not win the argmax just because
    # of an artifact: t with valid high NDVI should still be selected.
    s2 = _make_s2([0.2, 0.8, 0.4])
    # Corrupt timestep 0 with a near-zero denominator (NDVI undefined -> dropped).
    s2[0, ppd._PASTIS_B04] = 0
    s2[0, ppd._PASTIS_B08] = 0
    composite = peak_ndvi_composite(s2)
    expected_b08 = s2[1, ppd._PASTIS_B08, 0, 0] / 10000.0
    assert composite[3, 0, 0] == pytest.approx(expected_b08, abs=1e-4)


def test_peak_ndvi_composite_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="PASTIS 10-band"):
        peak_ndvi_composite(np.zeros((10, 8, 8), dtype=np.int16))


# ---------------------------------------------------------------------------
# 2. dominant_class: restricted to active + None.
# ---------------------------------------------------------------------------


def test_dominant_class_restricted_to_active() -> None:
    # class 8 has the most pixels overall, but only {1,3,2} are active here:
    # among active, class 3 has the largest count -> dominant is 3.
    semantic = _make_semantic({8: 30, 3: 20, 2: 5, 1: 3})
    active = active_classes(3)  # (1, 3, 2)
    assert dominant_class(semantic, active) == 3


def test_dominant_class_ignores_background_and_void() -> None:
    # Background (0) and Void (19) dominate by count but must be excluded.
    semantic = _make_semantic({0: 40, 19: 20, 2: 4})
    active = active_classes(4)  # (1, 3, 2, 8)
    assert dominant_class(semantic, active) == 2


def test_dominant_class_none_when_no_active() -> None:
    # Only class 8 present, but it is not in the active set -> None.
    semantic = _make_semantic({8: 30})
    active = active_classes(2)  # (1, 3)
    assert dominant_class(semantic, active) is None


# ---------------------------------------------------------------------------
# 3. active_classes: curriculum + ValueError.
# ---------------------------------------------------------------------------


def test_active_classes_curriculum() -> None:
    assert active_classes(4) == (1, 3, 2, 8)
    assert active_classes(18) == INCREMENTAL_CURRICULUM
    assert len(INCREMENTAL_CURRICULUM) == 18
    assert set(INCREMENTAL_CURRICULUM) == set(range(1, 19))
    with pytest.raises(ValueError, match=r"\[1, 18\]"):
        active_classes(0)
    with pytest.raises(ValueError, match=r"\[1, 18\]"):
        active_classes(19)


# ---------------------------------------------------------------------------
# Shared monkeypatch fixtures for dataset construction.
# ---------------------------------------------------------------------------


class _FakeFilter:
    """Stand-in for ``PastisFilter`` capturing the 3:1 config and kept ids."""

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def filter_folds(self, folds: object) -> list[int]:
        return list(_FakeFilter.kept_ids)


def _patch_construction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    patches: dict[int, tuple[np.ndarray, np.ndarray]],
    kept_ids: list[int],
) -> None:
    """Monkeypatches ``PastisFilter`` and ``load_pastis_patch`` on the module.

    Args:
        monkeypatch: pytest fixture.
        patches: ``{pid: (s2, semantic)}`` synthetic patches.
        kept_ids: ids the fake filter returns from ``filter_folds``.
    """
    _FakeFilter.kept_ids = kept_ids  # type: ignore[attr-defined]
    monkeypatch.setattr("ml.data.pastis_filter.PastisFilter", _FakeFilter)

    def _fake_loader(
        patch_id: object,
        root: object = None,
        load_annotations: bool = True,
    ) -> dict[str, object]:
        s2, semantic = patches[int(patch_id)]
        return {"s2": s2, "semantic": semantic if load_annotations else None}

    monkeypatch.setattr(ppd, "load_pastis_patch", _fake_loader)


# ---------------------------------------------------------------------------
# 4. Filter 3:1 applied (over-Meadow patch excluded).
# ---------------------------------------------------------------------------


def test_filter_3to1_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two patches: pid 100 has a healthy crop dominance, pid 200 is so
    # Meadow-dominated that the real 3:1 filter would drop it. We exercise the
    # REAL PastisFilter.dominance logic by mocking only its mask loader.
    s2_ok = _make_s2([0.6])
    s2_bad = _make_s2([0.6])
    # pid 100: Corn (class 3) dominates; Meadow small -> kept by 3:1.
    sem_ok = _make_semantic({3: 40, 1: 10})
    # pid 200: Meadow (1) hugely over the 2nd class (Corn) -> dropped by 3:1.
    sem_bad = _make_semantic({1: 60, 3: 4})

    masks = {100: sem_ok, 200: sem_bad}

    from ml.data.pastis_filter import PastisFilter

    monkeypatch.setattr(PastisFilter, "_load_mask", lambda self, pid: masks[pid].astype(np.int32))
    monkeypatch.setattr(PastisFilter, "__init__", lambda self, **kw: _init_real_filter(self, **kw))

    patches = {100: (s2_ok, sem_ok), 200: (s2_bad, sem_bad)}

    def _fake_loader(
        patch_id: object,
        root: object = None,
        load_annotations: bool = True,
    ) -> dict[str, object]:
        s2, semantic = patches[int(patch_id)]
        return {"s2": s2, "semantic": semantic if load_annotations else None}

    monkeypatch.setattr(ppd, "load_pastis_patch", _fake_loader)

    ds = ppd.PastisPairDataset(n_classes=4, folds=(1,), root=Path("unused"))
    kept_pids = {pid for pid, _ in ds._samples}
    assert kept_pids == {"100"}, f"expected only the 3:1-passing patch, got {kept_pids}"


def _init_real_filter(self: object, **kw: object) -> None:
    """Lightweight ``PastisFilter.__init__`` replacement (no metadata.geojson).

    Sets only the attributes the ``dominance_ratio`` path and ``filter_folds``
    read, and a single-fold map covering the synthetic patch ids.
    """
    self.root = Path(kw.get("pastis_root", "."))  # type: ignore[attr-defined]
    self.target_classes = set(kw.get("target_classes") or [])  # type: ignore[attr-defined]
    self.min_coverage = 0.5  # type: ignore[attr-defined]
    self.ignore_index = 255  # type: ignore[attr-defined]
    self.annotation_key = 0  # type: ignore[attr-defined]
    self.verbose = False  # type: ignore[attr-defined]
    self.mode = kw.get("mode", "dominance_ratio")  # type: ignore[attr-defined]
    self.ratio = float(kw.get("ratio", 3.0))  # type: ignore[attr-defined]
    self.meadow_class = int(kw.get("meadow_class", 1))  # type: ignore[attr-defined]
    self.total_scanned = 0  # type: ignore[attr-defined]
    self._fold_map = {1: [100, 200]}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. create_incremental_dataset shapes + DIRECT prototype rows.
# ---------------------------------------------------------------------------


def test_create_incremental_dataset_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    proto, class_ids = _fake_proto_18()

    def _fake_proto_loader(path: Path | None = None) -> tuple[np.ndarray, list[int]]:
        return proto, class_ids

    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        _fake_proto_loader,
    )

    # One synthetic patch per active class so the dataset is non-empty.
    s2 = _make_s2([0.7])
    patches = {
        pid: (s2, _make_semantic({cid: 40, 1: 5}))
        for pid, cid in zip(range(1000, 1018), INCREMENTAL_CURRICULUM, strict=True)
    }
    # Avoid Meadow-as-dominant for the Meadow class itself: give it many Meadow.
    patches[1000] = (s2, _make_semantic({1: 50}))
    _patch_construction(monkeypatch, patches=patches, kept_ids=list(patches))

    # n_classes = 4
    _ds4, n_regions4, n_cat4, proto4 = create_incremental_dataset(4)
    assert n_regions4 == 1
    assert n_cat4 == 4
    assert proto4.shape == (4, _PROTO_DIM)
    # DIRECT rows: proto_active[i] == proto_18[row_of[active[i]]].
    active4 = active_classes(4)
    row_of = {cid: r for r, cid in enumerate(class_ids)}
    for i, cid in enumerate(active4):
        assert torch.allclose(proto4[i], torch.from_numpy(proto[row_of[cid]]))

    # n_classes = 18
    _ds18, n_regions18, n_cat18, proto18 = create_incremental_dataset(18)
    assert n_regions18 == 1
    assert n_cat18 == 18
    assert proto18.shape == (18, _PROTO_DIM)
    active18 = active_classes(18)
    for i, cid in enumerate(active18):
        assert torch.allclose(proto18[i], torch.from_numpy(proto[row_of[cid]]))


def test_create_incremental_dataset_validates_n_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proto, class_ids = _fake_proto_18()
    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        lambda path=None: (proto, class_ids),
    )
    with pytest.raises(ValueError, match=r"\[1, 18\]"):
        create_incremental_dataset(0)


# ---------------------------------------------------------------------------
# 6. __getitem__ contract.
# ---------------------------------------------------------------------------


def test_getitem_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    s2 = _make_s2([0.3, 0.8])
    # pid 100 dominant = Corn (3) -> category_id == index of 3 in active(4) == 1.
    patches = {100: (s2, _make_semantic({3: 40, 1: 5}))}
    _patch_construction(monkeypatch, patches=patches, kept_ids=[100])

    ds = ppd.PastisPairDataset(n_classes=4, folds=(1,), root=Path("unused"))
    assert len(ds) == 1

    item = ds[0]
    assert set(item) == {"image", "region_id", "category_id"}
    assert item["image"].shape == (4, _RESIZE, _RESIZE)
    assert item["image"].dtype == torch.float32
    assert float(item["image"].min()) >= 0.0
    assert float(item["image"].max()) <= 1.0

    assert item["region_id"].ndim == 0
    assert item["region_id"].dtype == torch.long
    assert int(item["region_id"]) == 0

    assert item["category_id"].ndim == 0
    assert item["category_id"].dtype == torch.long
    assert 0 <= int(item["category_id"]) <= 3
    assert int(item["category_id"]) == active_classes(4).index(3)  # == 1


# ---------------------------------------------------------------------------
# 7. Construction-time exclusions and __getitem__ edge cases.
# ---------------------------------------------------------------------------


def test_dataset_excludes_patch_without_active_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pid 100 dominant = Corn (3, active); pid 200 only has class 8 (not active in
    # n_classes=2) -> dominant_class None -> excluded; pid 300 has no semantic
    # mask -> also excluded. Only pid 100 survives.
    s2 = _make_s2([0.6])
    patches = {
        100: (s2, _make_semantic({3: 40, 1: 5})),
        200: (s2, _make_semantic({8: 40})),
        300: (s2, None),
    }

    def _fake_loader(
        patch_id: object,
        root: object = None,
        load_annotations: bool = True,
    ) -> dict[str, object]:
        s2_arr, semantic = patches[int(patch_id)]
        return {"s2": s2_arr, "semantic": semantic if load_annotations else None}

    monkeypatch.setattr(ppd, "load_pastis_patch", _fake_loader)
    monkeypatch.setattr("ml.data.pastis_filter.PastisFilter", _FakeFilter)
    _FakeFilter.kept_ids = [100, 200, 300]  # type: ignore[attr-defined]

    ds = ppd.PastisPairDataset(n_classes=2, folds=(1,), root=Path("unused"))
    kept_pids = {pid for pid, _ in ds._samples}
    assert kept_pids == {"100"}


def test_getitem_negative_index_and_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2 = _make_s2([0.6])
    patches = {100: (s2, _make_semantic({3: 40, 1: 5}))}
    _patch_construction(monkeypatch, patches=patches, kept_ids=[100])

    ds = ppd.PastisPairDataset(n_classes=4, folds=(1,), root=Path("unused"))
    # Negative index resolves from the end (Python convention).
    assert ds[-1]["image"].shape == (4, _RESIZE, _RESIZE)
    with pytest.raises(IndexError, match="idx out of range"):
        _ = ds[5]


def test_create_incremental_dataset_missing_prototype_class_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The parquet is missing one active class_id -> ValueError, never a wrong row.
    proto = np.random.default_rng(0).standard_normal((3, _PROTO_DIM)).astype(np.float32)
    class_ids = [1, 3, 8]  # active_classes(4) == (1, 3, 2, 8): class 2 is missing.

    monkeypatch.setattr(
        "ml.features.phenology_class_prototypes.load_class_prototype_embeddings",
        lambda path=None: (proto, class_ids),
    )

    s2 = _make_s2([0.6])
    patches = {100: (s2, _make_semantic({3: 40, 1: 5}))}
    _patch_construction(monkeypatch, patches=patches, kept_ids=[100])

    with pytest.raises(ValueError, match="missing active class_ids"):
        create_incremental_dataset(4)
