"""Regression tests for the dense->parcel bridge of the Italian Voting-3 (US-079).

Guards the canonical-id JOIN invariant the run2 vote-collapse bug violated: the
``canonical_parcel_id`` a patch's parcels receive from
:func:`ml.transfer.dense_to_parcel_italia.load_eurocrops_parcel_rasters` MUST NOT
depend on the ``patch_ids`` subset requested, because the
``xgb-alphaearth-italia`` member always enumerates the parcels over the FULL
metadata. If the bridge re-enumerated the seq over a smaller universe (the bug),
the same physical polygon would get a different id and the per-parcel vote would
collapse to a handful of spurious seq collisions (run2 voted over 32 parcels
instead of ~22k).

The EuroCrops parquet is heavy and not on CI, so ``parcels_in_patches`` and the
label-space loaders are monkeypatched with a tiny deterministic pandas fixture
whose ``canonical_parcel_id`` mirrors the production global-running-index scheme.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml.transfer import dense_to_parcel_italia as bridge


def _full_parcels() -> pd.DataFrame:
    """Two patches (p0, p1) with global-running-index seqs over the FULL universe.

    The seq of patch 1's parcels (2, 3) reflects that p0 contributed the first two
    rows of the global enumeration -- exactly the production scheme. If the bridge
    re-enumerated over a subset, p1 alone would wrongly produce seqs 0, 1.
    """
    return pd.DataFrame(
        {
            "patch_id": [0, 0, 1, 1],
            "geometry": [object(), object(), object(), object()],
            "canonical_parcel_id": [
                "iti1_2018_p0_0",
                "iti1_2018_p0_1",
                "iti1_2018_p1_2",
                "iti1_2018_p1_3",
            ],
        }
    )


@pytest.fixture
def _patched_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the heavy EuroCrops loaders with the tiny full-universe fixture."""
    import ml.transfer.alphaearth_italia as ae

    bboxes = pl.DataFrame(
        {
            "patch_id": [0, 1],
            "bbox_min_lon": [0.0, 0.0],
            "bbox_min_lat": [0.0, 0.0],
            "bbox_max_lon": [1.0, 1.0],
            "bbox_max_lat": [1.0, 1.0],
            "fold_espacial": [3, 3],
        }
    )
    monkeypatch.setattr(ae, "load_italia_label_space", lambda _root: {"x": 1})
    monkeypatch.setattr(ae, "load_patch_bboxes", lambda _root: bboxes)
    # ALWAYS return the FULL-universe parcels regardless of the bboxes passed,
    # asserting the bridge never re-enumerates seqs over a subset.
    # ``**_kw`` absorbs keyword arguments the bridge added later (``region_prefix``).
    monkeypatch.setattr(ae, "parcels_in_patches", lambda _b, _n, **_kw: _full_parcels())

    def _fake_raster(parcels: pd.DataFrame, _bbox, *, patch_px: int = 4):
        """Paint each parcel's surrogate onto a row; deterministic, no rasterio."""
        canonical = [str(c) for c in parcels["canonical_parcel_id"].tolist()]
        parcel_map = np.zeros((patch_px, patch_px), dtype=np.int64)
        id_to_canonical: dict[int, str] = {}
        for i, cid in enumerate(canonical):
            parcel_map[0, i] = i + 1
            id_to_canonical[i + 1] = cid
        return parcel_map, id_to_canonical

    monkeypatch.setattr(bridge, "eurocrops_parcel_id_raster", _fake_raster)


def test_canonical_ids_are_subset_independent(_patched_bridge: None) -> None:
    """The id of a patch's parcels is identical whether or not it is the only one.

    This is the exact invariant the run2 vote-collapse violated: requesting only
    patch 1 must yield the SAME ``iti1_2018_p1_2`` / ``iti1_2018_p1_3`` ids it has
    in the full build, never a re-enumerated ``iti1_2018_p1_0`` / ``_1``.
    """
    full = bridge.load_eurocrops_parcel_rasters(Path("/nonexistent"))
    subset = bridge.load_eurocrops_parcel_rasters(Path("/nonexistent"), patch_ids=[1])

    full_p1 = set(full[1][1].values())
    subset_p1 = set(subset[1][1].values())

    assert full_p1 == subset_p1 == {"iti1_2018_p1_2", "iti1_2018_p1_3"}
    # The subset must NOT contain patch 0 (only patch 1 was requested).
    assert 0 not in subset
    assert 0 in full


def test_subset_ids_are_subset_of_full(_patched_bridge: None) -> None:
    """Every subset id is a full id (no drift), so the xgb join stays exact."""
    full = bridge.load_eurocrops_parcel_rasters(Path("/nonexistent"))
    subset = bridge.load_eurocrops_parcel_rasters(Path("/nonexistent"), patch_ids=[1])
    full_ids = {c for _, can in full.values() for c in can.values()}
    subset_ids = {c for _, can in subset.values() for c in can.values()}
    assert subset_ids.issubset(full_ids)
    assert subset_ids == {"iti1_2018_p1_2", "iti1_2018_p1_3"}
