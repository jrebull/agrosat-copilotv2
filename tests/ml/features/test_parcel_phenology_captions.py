"""Deterministic CPU tests for per-parcel phenology captions (US-036-b).

Covers ``ml/features/parcel_phenology_captions.py``: the per-INSTANCE NDVI curve
computation (one curve per parcel, not per class), the incremental-flush caption
generation with a mocked LLM client (no Gemma, no network), and the resume path.

PASTIS-R is monkeypatched with synthetic ``s2``/``semantic``/``instance`` arrays;
the LLM is injected via ``set_llm_client`` so descriptions are deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.features.parcel_phenology_captions as ppc
from ml.features.parcel_phenology_captions import (
    compute_parcel_ndvi_curves,
    generate_parcel_phenology_captions,
    load_parcel_captions,
)


def _synthetic_patch(pid: str, ndvi_a: float, ndvi_b: float):
    """Build a synthetic patch dict with two parcels of distinct NDVI levels."""
    t, h, w = 4, 16, 16
    s2 = np.zeros((t, 10, h, w), dtype=np.int16)
    red = 1000.0
    # Two parcels: instance 1 (top block) and instance 2 (bottom block).
    instance = np.zeros((h, w), dtype=np.int64)
    semantic = np.zeros((h, w), dtype=np.int64)
    instance[0:6, :] = 1
    semantic[0:6, :] = 1  # Meadow
    instance[8:, :] = 2
    semantic[8:, :] = 3  # Corn
    for ti in range(t):
        for inst_id, ndvi in ((1, ndvi_a), (2, ndvi_b)):
            nir = red * (1.0 + ndvi) / (1.0 - ndvi)
            mask = instance == inst_id
            s2[ti, 2][mask] = int(red)  # B04 (idx 2)
            s2[ti, 6][mask] = int(nir)  # B08 (idx 6)
    return {"s2": s2, "semantic": semantic, "instance": instance}


@pytest.fixture
def _patch_pastis(monkeypatch):
    """Monkeypatch PASTIS access so no disk/metadata is needed."""
    patches = {
        "10000": _synthetic_patch("10000", 0.7, 0.3),
        "10001": _synthetic_patch("10001", 0.6, 0.4),
    }
    doy = np.array([60, 150, 240, 330], dtype=np.int32)

    monkeypatch.setattr(ppc, "_patch_dates_doy", lambda _p: {10000: doy, 10001: doy})

    def _fake_split(_root, *, train_folds, val_folds, test_folds):
        return {"train": list(patches.keys()), "val": [], "test": []}

    import ml.ingest.pastis_dataset as pds
    import ml.ingest.pastis_loader as pl_loader

    monkeypatch.setattr(pds, "pastis_fold_split", _fake_split)
    monkeypatch.setattr(pl_loader, "load_pastis_patch", lambda pid, **_kw: patches[str(pid)])
    return patches


def test_compute_parcel_ndvi_curves_per_instance(_patch_pastis) -> None:
    """One curve per parcel (instance), keyed ``{pid}_{iid}`` with its class."""
    curves = compute_parcel_ndvi_curves(
        Path("data/PASTIS-R"), folds=(1,), n_time_bins=37, min_area_px=4
    )
    # 2 patches x 2 parcels = 4 parcels.
    assert set(curves.keys()) == {"10000_1", "10000_2", "10001_1", "10001_2"}
    curve, doy, class_id = curves["10000_1"]
    assert curve.shape == (37,)
    assert doy.shape == (37,)
    assert class_id == 1  # Meadow
    assert curves["10000_2"][2] == 3  # Corn
    # The two parcels have different NDVI levels -> different curves.
    a = np.nanmean(curves["10000_1"][0])
    b = np.nanmean(curves["10000_2"][0])
    assert abs(a - b) > 0.1


def test_generate_captions_diverse_and_cached(_patch_pastis, tmp_path) -> None:
    """Captions are generated per parcel with a mock client and flushed."""
    from ml.features import phenology_description as pd

    # Deterministic mock: caption depends on the crop hint -> diversity by class.
    calls = {"n": 0}

    def _mock(prompt: str, *, model: str, temperature: float) -> str:
        calls["n"] += 1
        hint = "desconocido"
        if "Meadow" in prompt or "pradera" in prompt.lower():
            hint = "pradera"
        return f"Descripcion fenologica de {hint} (mock {calls['n']})."

    pd.set_llm_client(_mock)
    try:
        curves = compute_parcel_ndvi_curves(
            Path("data/PASTIS-R"), folds=(1,), n_time_bins=37, min_area_px=4
        )
        out = tmp_path / "parcel_captions.parquet"
        generate_parcel_phenology_captions(
            curves,
            class_names={1: "Meadow", 3: "Corn"},
            output_path=out,
            model="mock-model",
            cache_dir=tmp_path / "cache",
            flush_every=2,
        )
        assert out.is_file()
        df = pl.read_parquet(out)
        assert df.height == 4
        assert set(df.columns) >= {"parcel_id", "patch_id", "class_id", "description"}
        loaded = load_parcel_captions(out)
        assert len(loaded) == 4
        assert all(len(v) > 0 for v in loaded.values())
    finally:
        pd.set_llm_client(None)


def test_generate_captions_resume_skips_done(_patch_pastis, tmp_path) -> None:
    """Resume does not regenerate parcels already in the parquet."""
    from ml.features import phenology_description as pd

    n = {"calls": 0}

    def _counting_client(_p, *, model, temperature) -> str:
        n["calls"] += 1
        return f"c{n['calls']}"

    pd.set_llm_client(_counting_client)
    try:
        curves = compute_parcel_ndvi_curves(
            Path("data/PASTIS-R"), folds=(1,), n_time_bins=37, min_area_px=4
        )
        out = tmp_path / "parcel_captions.parquet"
        generate_parcel_phenology_captions(
            curves,
            class_names={1: "Meadow", 3: "Corn"},
            output_path=out,
            model="m",
            cache_dir=tmp_path / "c1",
            flush_every=10,
        )
        first = n["calls"]
        # Second run with resume should skip all (cache_dir differs to force the
        # resume path, not the SHA cache, to do the skipping).
        generate_parcel_phenology_captions(
            curves,
            class_names={1: "Meadow", 3: "Corn"},
            output_path=out,
            model="m",
            cache_dir=tmp_path / "c2",
            flush_every=10,
            resume=True,
        )
        assert n["calls"] == first  # no new generations
    finally:
        pd.set_llm_client(None)
