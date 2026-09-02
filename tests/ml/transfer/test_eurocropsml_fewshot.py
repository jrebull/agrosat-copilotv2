"""Tests for the EuroCropsML few-shot transfer pipeline (US-076).

All tests use synthetic ``.npz`` fixtures and monkeypatched network so NO real
Zenodo download nor real EuroCropsML data is needed: the shapes/labels are
exercised, never fabricated as scientific results. The real curve lives in the
notebook (``notebooks/eda/02g_eurocropsml_fewshot.ipynb``) over downloaded data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.transfer import eurocropsml_fewshot as fs
from ml.transfer.eurocropsml_fewshot import (
    DEFAULT_REGIONS,
    K_SHOTS,
    EuroCropsMLDataMissing,
    build_fewshot_splits,
    download_eurocropsml,
    load_region_samples,
    parcel_feature_vector,
    run_fewshot_curve,
    train_xgb_kshot,
)
from ml.transfer.label_align import (
    NULL_CLASS,
    align_codes_to_hcat_macro,
    to_group_code,
)

# Real HCAT leaf codes present in the US-074 crosswalk (group -> macro).
_CEREAL_CODE = 3301010101  # group 3301010000 -> cereals
_OILSEED_CODE = 3301060401  # group 3301060000 -> oilseed_industrial
_VINE_CODE = 3303060000  # group 3303060000 -> vineyard
_GRASS_CODE = 3302000000  # group 3302000000 -> grassland
_UNKNOWN_CODE = 9999999999  # not in crosswalk -> null-class

_N_BANDS = 13


def _make_npz(path: Path, n_steps: int, seed: int) -> None:
    """Write a synthetic per-parcel ``.npz`` (shape only, never a real result)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.2, scale=0.05, size=(n_steps, _N_BANDS)).astype(np.float32)
    dates = np.arange(n_steps, dtype=np.int64)
    center = np.array([rng.uniform(20, 28), rng.uniform(57, 59)], dtype=np.float64)
    np.savez(path, data=data, dates=dates, center=center)


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a tiny EuroCropsML-shaped tree of ``.npz`` parcels under tmp_path.

    Estonia (EE, target) and Latvia (LV, source) each get several parcels per
    macro class so a k-shot split with multiple classes is buildable.
    """
    npz_dir = tmp_path / "preprocess" / "S2" / "2021"
    npz_dir.mkdir(parents=True, exist_ok=True)
    codes = [_CEREAL_CODE, _OILSEED_CODE, _VINE_CODE, _GRASS_CODE]
    counter = 0
    for region in ("EE001", "LV001"):
        for code in codes:
            for _ in range(8):  # 8 parcels per class per region
                counter += 1
                n_steps = 6 + (counter % 4)
                path = npz_dir / f"{region}_{counter}_{code}.npz"
                _make_npz(path, n_steps=n_steps, seed=counter)
    return tmp_path


# --------------------------------------------------------------------------- #
# AC-1: the k-shot ladder matches the EuroCropsML protocol.
# --------------------------------------------------------------------------- #


def test_k_shots_match_protocol() -> None:
    assert K_SHOTS == (1, 5, 10, 20, 100, 200, 500)
    assert DEFAULT_REGIONS == ("estonia", "latvia")


# --------------------------------------------------------------------------- #
# Feature vector: fixed dimensionality independent of T.
# --------------------------------------------------------------------------- #


def test_parcel_feature_vector_shape() -> None:
    short = parcel_feature_vector(np.ones((3, _N_BANDS)))
    long = parcel_feature_vector(np.ones((40, _N_BANDS)))
    assert short.shape == long.shape
    assert short.ndim == 1
    # 7 base stats (mean/std/min/max/range/last-first/slope) + 5 percentiles.
    assert short.shape[0] == _N_BANDS * (7 + 5)
    assert np.isfinite(short).all()


def test_parcel_feature_vector_handles_single_step_and_nan() -> None:
    series = np.full((1, _N_BANDS), np.nan)
    series[0, :3] = 0.5
    vec = parcel_feature_vector(series)
    assert np.isfinite(vec).all()


def test_parcel_feature_vector_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parcel_feature_vector(np.empty((0, _N_BANDS)))


# --------------------------------------------------------------------------- #
# AC-3: label alignment to HCAT macro via the US-074 crosswalk.
# --------------------------------------------------------------------------- #


def test_align_labels_to_hcat_macro() -> None:
    codes = [_CEREAL_CODE, _OILSEED_CODE, _VINE_CODE, _GRASS_CODE, _UNKNOWN_CODE]
    macro = align_codes_to_hcat_macro(codes)
    assert macro == [
        "cereals",
        "oilseed_industrial",
        "vineyard",
        "grassland",
        NULL_CLASS,
    ]


def test_to_group_code_truncates_leaf_to_group() -> None:
    assert to_group_code(_CEREAL_CODE) == 3301010000
    assert to_group_code(0) == 0


def test_hcat_codes_resolve_via_real_crosswalk() -> None:
    # Reads the real data/reference/hcat_crosswalk.parquet (US-074 artifact).
    macro = align_codes_to_hcat_macro([_CEREAL_CODE])
    assert macro[0] == "cereals"


def test_align_series_polars() -> None:
    series = pl.Series("EC_hcat_c", [_CEREAL_CODE, _UNKNOWN_CODE])
    out = fs.align_labels_to_hcat_macro(series)
    assert out.to_list() == ["cereals", NULL_CLASS]


# --------------------------------------------------------------------------- #
# Reading parcels + missing-data contract.
# --------------------------------------------------------------------------- #


def test_load_region_samples_reads_npz(fake_dataset: Path) -> None:
    ee = load_region_samples(fake_dataset, "estonia")
    lv = load_region_samples(fake_dataset, "latvia")
    assert len(ee) == 32  # 4 classes x 8 parcels
    assert len(lv) == 32
    assert all(s.region == "estonia" for s in ee)
    assert ee[0].series.shape[1] == _N_BANDS


def test_load_region_samples_unknown_region(fake_dataset: Path) -> None:
    with pytest.raises(ValueError):
        load_region_samples(fake_dataset, "france")


def test_missing_data_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(EuroCropsMLDataMissing):
        load_region_samples(tmp_path, "estonia")


# --------------------------------------------------------------------------- #
# AC-2: k-shot training + curve schema (real recipe, synthetic shapes).
# --------------------------------------------------------------------------- #


def test_train_xgb_kshot_runs(fake_dataset: Path) -> None:
    split = build_fewshot_splits(fake_dataset, source=["latvia"], target="estonia", k=5, seed=0)
    metrics = train_xgb_kshot(
        split.x_target_train,
        split.y_target_train,
        split.x_target_test,
        split.y_target_test,
    )
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["n_train"] >= 1


def test_train_xgb_kshot_empty_support_raises() -> None:
    with pytest.raises(ValueError):
        train_xgb_kshot(np.empty((0, 5)), np.array([]), np.ones((2, 5)), np.array(["a", "b"]))


def test_run_fewshot_curve_schema(fake_dataset: Path) -> None:
    curve = run_fewshot_curve(
        fake_dataset,
        source=["latvia"],
        target="estonia",
        k_shots=(1, 5),
        seeds=(0, 1),
    )
    assert set(curve.columns) == {
        "source",
        "target",
        "k",
        "seed",
        "f1_macro",
        "n_classes",
        "use_pretrain",
    }
    assert curve.height == 4  # 2 k x 2 seeds
    assert curve["target"].unique().to_list() == ["EE"]
    assert curve["f1_macro"].min() >= 0.0
    assert curve["f1_macro"].max() <= 1.0


# --------------------------------------------------------------------------- #
# AC-6 / no-AlphaEarth: the base path never imports GEE (earthengine).
# --------------------------------------------------------------------------- #


def test_no_alphaearth_gee_import_in_base_path() -> None:
    source = Path(fs.__file__).read_text(encoding="utf-8")
    assert "import ee" not in source
    assert "earthengine" not in source


# --------------------------------------------------------------------------- #
# AC / Plan B: download is non-interactive (no get_user_choice / input()).
# --------------------------------------------------------------------------- #


def test_download_is_noninteractive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """download_eurocropsml must resolve the version itself, never prompt."""
    import ml.transfer.eurocropsml_fewshot as mod

    calls: dict[str, int] = {"get": 0, "stream": 0}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "hits": {
                    "hits": [
                        {
                            "metadata": {
                                "publication_date": "2025-03-31",
                                "relations": {"version": [{"index": 10}]},
                                "title": "EuroCropsML",
                            },
                            "links": {"doi": "10.5281/zenodo.15095445"},
                            "files": [
                                {
                                    "key": "split.zip",
                                    "links": {"self": "http://x/split.zip"},
                                }
                            ],
                        }
                    ]
                }
            }

    def _fake_get(url: str, timeout: int = 0, **kwargs: object) -> _FakeResp:
        calls["get"] += 1
        return _FakeResp()

    def _fake_stream(url: str, local_path: Path, **kwargs: object) -> None:
        calls["stream"] += 1
        local_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty-zip stub

    def _fake_unzip(zip_path: Path, extract_to: Path, **kwargs: object) -> None:
        (extract_to / "split").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_stream_download", _fake_stream)
    monkeypatch.setattr(mod, "_unzip", _fake_unzip)

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    # input() would raise OSError under pytest; assert it is never reached.
    monkeypatch.setattr(
        "builtins.input",
        lambda *a, **k: pytest.fail("download_eurocropsml prompted interactively"),
    )

    out = download_eurocropsml(tmp_path, regions=("estonia",), files=("split.zip",))
    assert out == tmp_path
    assert calls["get"] == 1
    assert calls["stream"] == 1


def test_download_skips_when_data_present(
    monkeypatch: pytest.MonkeyPatch, fake_dataset: Path
) -> None:
    """A second call is a no-op when .npz parcels already exist (idempotent)."""
    import requests

    def _boom(*a: object, **k: object) -> None:
        pytest.fail("download_eurocropsml hit the network despite existing data")

    monkeypatch.setattr(requests, "get", _boom)
    out = download_eurocropsml(fake_dataset, regions=("estonia",))
    assert out == fake_dataset
