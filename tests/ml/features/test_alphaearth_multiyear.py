"""Tests for the multi-year AlphaEarth averaging (US-042 E-b).

Covers ``ml/features/alphaearth_multiyear.py``: the per-dimension mean over years
(inner join on ``parcel_id``), the single-year degenerate case, the
missing-column guard, and the disk wrapper's graceful skip of a missing year.

In-memory synthetic frames (no GEE, no DVC). Conventions: Polars, no emojis.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.features.alphaearth_multiyear import (
    ALPHAEARTH_DIM,
    alphaearth_dim_columns,
    average_alphaearth_years,
    build_averaged_alphaearth,
    build_avg_features_for_xgb,
)


def _synthetic_year(parcel_ids: list[str], fill: float) -> pl.DataFrame:
    """Build a synthetic AlphaEarth year frame (all dims = ``fill``)."""
    data: dict[str, object] = {"parcel_id": parcel_ids, "year": [2019] * len(parcel_ids)}
    for c in alphaearth_dim_columns():
        data[c] = [fill] * len(parcel_ids)
    return pl.DataFrame(data)


def test_dim_columns_count() -> None:
    """There are exactly 64 embedding columns named dim_00..dim_63."""
    cols = alphaearth_dim_columns()
    assert len(cols) == ALPHAEARTH_DIM == 64
    assert cols[0] == "dim_00"
    assert cols[-1] == "dim_63"


def test_average_two_years_is_per_dim_mean() -> None:
    """Averaging two years yields the per-dimension mean on the shared parcels."""
    y18 = _synthetic_year(["10000_1", "10000_2"], fill=0.2)
    y19 = _synthetic_year(["10000_1", "10000_2"], fill=0.6)
    out = average_alphaearth_years([y18, y19])
    assert out.height == 2
    assert out["n_years"].unique().to_list() == [2]
    for c in alphaearth_dim_columns():
        assert out[c].to_list() == pytest.approx([0.4, 0.4])


def test_average_inner_joins_on_parcel_id() -> None:
    """Only parcels present in EVERY year survive the average (inner join)."""
    y18 = _synthetic_year(["10000_1", "10000_2", "10000_3"], fill=0.2)
    y19 = _synthetic_year(["10000_1", "10000_2"], fill=0.6)
    out = average_alphaearth_years([y18, y19])
    assert sorted(out["parcel_id"].to_list()) == ["10000_1", "10000_2"]


def test_single_year_is_passthrough() -> None:
    """A single frame returns its embeddings unchanged with n_years=1 (fallback)."""
    y19 = _synthetic_year(["10000_1"], fill=0.5)
    out = average_alphaearth_years([y19])
    assert out["n_years"].to_list() == [1]
    assert out["dim_00"].to_list() == [0.5]


def test_missing_columns_raises() -> None:
    """A frame without the embedding columns is rejected."""
    bad = pl.DataFrame({"parcel_id": ["10000_1"], "year": [2019]})
    with pytest.raises(ValueError, match="missing columns"):
        average_alphaearth_years([bad])


def test_empty_frames_raises() -> None:
    """No frames is an error."""
    with pytest.raises(ValueError, match="at least one frame"):
        average_alphaearth_years([])


def test_build_skips_missing_year_file(tmp_path: Path) -> None:
    """The disk wrapper averages only the existing years (graceful fallback)."""
    y19 = _synthetic_year(["10000_1"], fill=0.5)
    p19 = tmp_path / "ae_2019.parquet"
    y19.write_parquet(p19)
    out_path = tmp_path / "ae_mean.parquet"
    result = build_averaged_alphaearth(
        [tmp_path / "ae_2018_missing.parquet", p19], out_path=out_path
    )
    assert result == out_path
    df = pl.read_parquet(out_path)
    assert df["n_years"].to_list() == [1]  # only 2019 existed
    assert df["dim_00"].to_list() == [0.5]


def test_build_all_missing_raises(tmp_path: Path) -> None:
    """If NO year file exists the builder raises (no silent empty output)."""
    with pytest.raises(FileNotFoundError, match="none of the AlphaEarth"):
        build_averaged_alphaearth(
            [tmp_path / "a.parquet", tmp_path / "b.parquet"],
            out_path=tmp_path / "out.parquet",
        )


def _fused_stub(parcel_ids: list[str], fill: float) -> pl.DataFrame:
    """Synthetic fused-features frame: key/label cols + single-year dims + extras."""
    n = len(parcel_ids)
    data: dict[str, object] = {
        "parcel_id": parcel_ids,
        "patch_id": ["10000"] * n,
        "instance_id": list(range(1, n + 1)),
        "class_id": [3] * n,
        "fold": [5] * n,
        "NDVI_mean": [0.7] * n,  # an extra non-dim feature, must survive
    }
    for c in alphaearth_dim_columns():
        data[c] = [fill] * n  # single-year dims to be REPLACED by the average
    return pl.DataFrame(data)


def test_build_avg_features_replaces_dims_keeps_metadata(tmp_path: Path) -> None:
    """The XGB features parquet has averaged dims + the fused key/label/extra cols."""
    ids = ["10000_1", "10000_2"]
    y18 = _synthetic_year(ids, fill=0.2)
    y19 = _synthetic_year(ids, fill=0.6)
    p18, p19 = tmp_path / "y18.parquet", tmp_path / "y19.parquet"
    y18.write_parquet(p18)
    y19.write_parquet(p19)
    fused = _fused_stub(ids, fill=999.0)  # single-year dims = 999 (must be replaced)
    fused_path = tmp_path / "fused.parquet"
    fused.write_parquet(fused_path)

    out = build_avg_features_for_xgb(
        [p18, p19], fused_path, out_path=tmp_path / "xgb_feats.parquet"
    )
    df = pl.read_parquet(out)
    # Key/label/extra columns survive.
    for c in ("patch_id", "instance_id", "class_id", "fold", "NDVI_mean"):
        assert c in df.columns
    # Dims are the 2018+2019 mean (0.4), NOT the fused single-year 999.
    assert df["dim_00"].to_list() == pytest.approx([0.4, 0.4])
    assert df.height == 2


def test_build_avg_features_missing_fused_raises(tmp_path: Path) -> None:
    """A missing fused features parquet is a hard error."""
    y19 = _synthetic_year(["10000_1"], fill=0.5)
    p19 = tmp_path / "y19.parquet"
    y19.write_parquet(p19)
    with pytest.raises(FileNotFoundError, match="fused features parquet"):
        build_avg_features_for_xgb(
            [p19], tmp_path / "nope.parquet", out_path=tmp_path / "out.parquet"
        )


def test_build_avg_features_rejects_fused_without_keys(tmp_path: Path) -> None:
    """A fused parquet lacking key/label columns is rejected."""
    ids = ["10000_1"]
    y19 = _synthetic_year(ids, fill=0.5)
    p19 = tmp_path / "y19.parquet"
    y19.write_parquet(p19)
    bad_fused = _synthetic_year(ids, fill=0.1)  # has parcel_id + dims but no fold/etc
    bad_path = tmp_path / "bad_fused.parquet"
    bad_fused.write_parquet(bad_path)
    with pytest.raises(ValueError, match="missing key/label columns"):
        build_avg_features_for_xgb([p19], bad_path, out_path=tmp_path / "out.parquet")
