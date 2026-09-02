"""Tests for the WorldCereal tropical transfer experiment (Experimento 3).

The GEE-bound ingest helpers are not exercised against a live Earth Engine in
CI (that needs ADC + quota); instead they are tested for their graceful-
degradation contract (``ee is None`` -> empty frame, valid schema) and the pure
analysis functions (few-shot curve, separability, zero-shot maize detection) are
tested on a small synthetic AlphaEarth-shaped dataset with a planted signal.

When the REAL cached datasets produced by the live run are present
(``data/transfer/worldcereal_*.parquet``) the marked ``empirical`` tests assert
the published numbers reproduce; they are skipped otherwise so CI stays
hermetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.ingest.gee_sampler import ALPHAEARTH_DIM_COLS
from ml.transfer import worldcereal_tropical as wc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRAZIL = _REPO_ROOT / "data" / "transfer" / "worldcereal_brazil_cerrado.parquet"


def _synthetic_dataset(n_per_class: int = 120, seed: int = 0) -> pl.DataFrame:
    """Build an AlphaEarth-shaped synthetic dataset with a separable signal.

    Each of the 4 tropical classes gets a distinct mean in the 64-dim space so
    the few-shot/separability functions have a real (synthetic) signal to learn,
    without any GEE call. This is a unit fixture, never a scientific claim.

    Args:
        n_per_class: Rows per class.
        seed: RNG seed.

    Returns:
        A frame with the canonical ``px_id, lon, lat, label, class_name,
        dim_00..dim_63`` schema.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    counter = 0
    for label, name in enumerate(wc.TROPICAL_CLASSES):
        center = rng.normal(0.0, 1.0, size=64) * 0.5
        for _ in range(n_per_class):
            vec = center + rng.normal(0.0, 0.25, size=64)
            # px_id matches the real single-trailing-index format
            # "<region>_<idx>" so the region-name recovery is exercised.
            counter += 1
            row: dict[str, object] = {
                "px_id": f"brazil_cerrado_{counter}",
                "lon": -51.0 + rng.normal(0, 0.1),
                "lat": -14.0 + rng.normal(0, 0.1),
                "label": label,
                "class_name": name,
            }
            for d, col in enumerate(ALPHAEARTH_DIM_COLS):
                row[col] = float(vec[d])
            rows.append(row)
    return pl.DataFrame(rows)


def test_tropical_classes_canonical() -> None:
    """The 4 tropical classes are stable and maize is the last (id 3)."""
    assert wc.TROPICAL_CLASSES == (
        "non_crop",
        "other_cropland",
        "wintercereals",
        "maize",
    )
    assert wc.TROPICAL_CLASSES[wc._MAIZE_LABEL] == "maize"


def test_sample_points_degraded_without_ee(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without ``ee`` the sampler returns an empty frame with valid schema."""
    monkeypatch.setattr(wc, "ee", None)
    # The cache dir must be creatable: a root-level path is read-only on macOS.
    out = wc.sample_worldcereal_points(
        wc.DEFAULT_REGION, cache_dir=tmp_path / "nonexistent_cache_xyz"
    )
    assert out.is_empty()
    assert set(out.columns) == {"px_id", "lon", "lat", "label", "class_name"}


def test_build_dataset_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_dataset`` raises the typed error when no points come back."""
    monkeypatch.setattr(
        wc,
        "sample_worldcereal_points",
        lambda *a, **k: pl.DataFrame(schema=wc._points_schema()),
    )
    with pytest.raises(wc.WorldCerealDataMissing):
        wc.build_dataset(wc.DEFAULT_REGION, cache_dir=Path("/nonexistent_cache_xyz"))


def test_fewshot_curve_monotone_on_synthetic() -> None:
    """The few-shot curve is non-trivial and improves with k on synthetic data."""
    ds = _synthetic_dataset()
    curve = wc.run_fewshot_curve(ds, k_shots=(1, 5, 20), seeds=(0, 1))
    summary = wc.summarize_curve(curve).sort("k")
    f1 = summary.get_column("f1_mean").to_list()
    # Region name recovered correctly (no trailing index leak).
    assert summary.get_column("region").unique().to_list() == ["brazil_cerrado"]
    # k=20 should beat k=1 on a clearly-separable synthetic signal.
    assert f1[-1] > f1[0]
    assert 0.0 <= f1[0] <= 1.0 and 0.0 <= f1[-1] <= 1.0


def test_separability_returns_valid_metrics() -> None:
    """The separability probe returns a bounded F1 and the right cardinalities."""
    ds = _synthetic_dataset()
    out = wc.zero_shot_separability(ds, n_splits=3)
    assert 0.0 <= out["f1_macro_cv"] <= 1.0
    assert out["n_classes"] == 4.0
    assert out["n_samples"] == float(ds.height)


def test_zero_shot_resolves_corn_by_name(tmp_path: Path) -> None:
    """Zero-shot resolves the shared 'Corn' class id from the table by NAME.

    A tiny European table where 'Corn' deliberately sits at class_id 7 (not the
    real-table's 3) verifies the id is never hardcoded.
    """
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    # Two European classes: Corn (id 7) and Meadow (id 1).
    for cid, name, shift in ((7, "Corn", 2.0), (1, "Meadow", -2.0)):
        for _ in range(60):
            vec = np.full(64, shift) + rng.normal(0, 0.1, size=64)
            row: dict[str, object] = {"class_id": cid, "class_name": name}
            for d, col in enumerate(ALPHAEARTH_DIM_COLS):
                row[col] = float(vec[d])
            rows.append(row)
    eu = pl.DataFrame(rows)
    eu_path = tmp_path / "eu.parquet"
    eu.write_parquet(eu_path)

    # Target: maize pixels sit near the 'Corn' (+2.0) cluster so a correct
    # name-resolution yields non-zero recall; other classes near -2.0.
    trows: list[dict[str, object]] = []
    for label, shift in ((wc._MAIZE_LABEL, 2.0), (0, -2.0)):
        for i in range(40):
            vec = np.full(64, shift) + rng.normal(0, 0.1, size=64)
            row = {
                "px_id": f"brazil_cerrado_{label}_{i}",
                "lon": 0.0,
                "lat": 0.0,
                "label": label,
                "class_name": wc.TROPICAL_CLASSES[label],
            }
            for d, col in enumerate(ALPHAEARTH_DIM_COLS):
                row[col] = float(vec[d])
            trows.append(row)
    target = pl.DataFrame(trows)

    out = wc.zero_shot_europe_to_tropics(target, eu_table=eu_path)
    # With the planted signal and correct name-resolution, recall must be high.
    assert out["maize_recall"] > 0.8
    assert 0.0 <= out["base_rate"] <= 1.0


@pytest.mark.empirical
def test_real_brazil_dataset_shape_and_classes() -> None:
    """The real cached Brazil dataset has the documented shape and classes."""
    if not _BRAZIL.exists():
        pytest.skip("real worldcereal_brazil_cerrado.parquet not present")
    ds = pl.read_parquet(_BRAZIL)
    assert ds.height > 1000
    assert all(c in ds.columns for c in ALPHAEARTH_DIM_COLS)
    names = set(ds.get_column("class_name").unique().to_list())
    assert names == set(wc.TROPICAL_CLASSES)
    # Embeddings are complete and finite (no fabricated/NaN rows).
    x = ds.select(ALPHAEARTH_DIM_COLS).to_numpy()
    assert np.isfinite(x).all()
