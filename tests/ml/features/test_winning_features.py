"""Tests para `ml.features.winning_features`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.features.winning_features import (
    persist_winning_features,
    select_winning_features,
)


@pytest.fixture
def ablation_table_promotes_farslip_only() -> pl.DataFrame:
    """Tabla de ablation que solo promueve FarSLIP (+0.02)."""
    return pl.DataFrame(
        {
            "feature_set": [
                "full",
                "with_farslip",
                "with_pheno_text",
                "with_spectral_signature",
            ],
            "model": ["xgb"] * 4,
            "f1_macro": [0.40, 0.42, 0.39, 0.40],
            "f1_weighted": [0.70, 0.72, 0.69, 0.70],
            "miou": [0.30, 0.32, 0.29, 0.30],
            "delta_vs_full": [0.0, 0.02, -0.01, 0.0],
        }
    )


@pytest.fixture
def available_cols() -> list[str]:
    """Set tipico de columnas disponibles en el fused."""
    return [
        "parcel_id",
        "year",
        "class_id",
        "patch_id",
        "sog_doy",
        "peak_doy",
        "ndvi_auc",
        "NDVI_fft_amp_0",
        "NDVI_fft_phase_0",
        "ae_00",
        "ae_01",
        "ae_02",
        "geom_area_ha",
        "geom_perimeter_m",
        "era5_tmean_m01",
        "srtm_elev_mean",
        "farslip_000",
        "farslip_001",
        "pheno_text_000",
        "spectral_signature_000",
    ]


def test_select_winning_promotes_farslip(
    ablation_table_promotes_farslip_only: pl.DataFrame,
    available_cols: list[str],
) -> None:
    winning = select_winning_features(ablation_table_promotes_farslip_only, available_cols)
    assert winning.decisions["farslip"] is True
    assert winning.decisions["pheno_text"] is False
    assert winning.decisions["spectral_signature"] is False
    assert "farslip_000" in winning.feature_cols
    assert "farslip_001" in winning.feature_cols
    assert "pheno_text_000" not in winning.feature_cols
    assert "spectral_signature_000" not in winning.feature_cols


def test_select_winning_discards_geom_by_default(
    ablation_table_promotes_farslip_only: pl.DataFrame,
    available_cols: list[str],
) -> None:
    winning = select_winning_features(ablation_table_promotes_farslip_only, available_cols)
    assert winning.decisions["geom"] is False
    assert "geom_area_ha" not in winning.feature_cols


def test_select_winning_includes_alphaearth_and_phenology(
    ablation_table_promotes_farslip_only: pl.DataFrame,
    available_cols: list[str],
) -> None:
    winning = select_winning_features(ablation_table_promotes_farslip_only, available_cols)
    assert "ae_00" in winning.feature_cols
    assert "sog_doy" in winning.feature_cols
    assert "NDVI_fft_amp_0" in winning.feature_cols


def test_select_winning_includes_era5_and_srtm(
    ablation_table_promotes_farslip_only: pl.DataFrame,
    available_cols: list[str],
) -> None:
    winning = select_winning_features(ablation_table_promotes_farslip_only, available_cols)
    assert "era5_tmean_m01" in winning.feature_cols
    assert "srtm_elev_mean" in winning.feature_cols


def test_persist_winning_features_creates_manifest(
    ablation_table_promotes_farslip_only: pl.DataFrame,
    available_cols: list[str],
    tmp_path: Path,
) -> None:
    winning = select_winning_features(ablation_table_promotes_farslip_only, available_cols)
    fused = pl.DataFrame(
        {
            "parcel_id": ["1_0", "1_1", "2_0"],
            "year": [2023, 2023, 2023],
            "class_id": [1, 2, 3],
            "patch_id": [1, 1, 2],
            "sog_doy": [50, 60, 70],
            "ae_00": [0.1, 0.2, 0.3],
            "farslip_000": [0.5, 0.6, 0.7],
            "pheno_text_000": [0.0, 0.0, 0.0],
        }
    )
    out_path = tmp_path / "winning.parquet"
    persist_winning_features(winning, fused, output_path=out_path, overwrite=True)
    assert out_path.exists()
    manifest = out_path.with_suffix(".manifest.json")
    assert manifest.exists()
    import json

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["decisions"]["farslip"] is True
    assert "ae_00" in payload["feature_cols"]
    assert "farslip_000" in payload["feature_cols"]
    # pheno_text NO debe estar persistido como winning porque su delta es negativo.
    assert "pheno_text_000" not in payload["feature_cols"]


def test_select_winning_handles_empty_ablation_table(
    available_cols: list[str],
) -> None:
    empty = pl.DataFrame(
        schema={
            "feature_set": pl.Utf8,
            "model": pl.Utf8,
            "f1_macro": pl.Float64,
            "delta_vs_full": pl.Float64,
        }
    )
    winning = select_winning_features(empty, available_cols)
    # Sin ablation valida: ninguno se promueve.
    assert all(v is False for v in winning.decisions.values())
