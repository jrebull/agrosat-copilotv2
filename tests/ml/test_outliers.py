"""Tests para `ml.analysis.outliers`."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.analysis.outliers import detect_outliers_iqr, detect_outliers_isoforest


@pytest.fixture
def df_with_known_outliers() -> pl.DataFrame:
    """DataFrame con 1000 valores N(0, 1) + 10 outliers extremos por banda."""
    rng = np.random.default_rng(42)
    rows = []
    for band in ["B02", "B04", "B08"]:
        inliers = rng.normal(loc=0.0, scale=1.0, size=1000)
        outliers = rng.normal(loc=50.0, scale=1.0, size=10)
        vals = np.concatenate([inliers, outliers])
        for i, v in enumerate(vals):
            rows.append(
                {
                    "patch_id": "p0",
                    "t": 0,
                    "y": i // 100,
                    "x": i % 100,
                    "band": band,
                    "value": float(v),
                }
            )
    return pl.DataFrame(rows)


def test_detect_outliers_iqr_finds_known(df_with_known_outliers: pl.DataFrame) -> None:
    """IQR debe detectar los 10 outliers inyectados al p99.5+."""
    out = detect_outliers_iqr(df_with_known_outliers, k=1.5)
    assert out.height == 3
    for r in out.iter_rows(named=True):
        # n_outliers >= 10 (los inyectados), pct entre 0.99% y ~5%
        assert r["n_outliers"] >= 10
        assert 0.5 < r["pct_outliers"] < 10.0


def test_detect_outliers_iqr_per_band_columns(df_with_known_outliers: pl.DataFrame) -> None:
    """Debe retornar columnas esperadas."""
    out = detect_outliers_iqr(df_with_known_outliers)
    expected = {"band", "n", "n_outliers", "pct_outliers", "q1", "q3", "lower", "upper"}
    assert expected.issubset(set(out.columns))


def test_detect_outliers_isoforest_respects_contamination(
    df_with_known_outliers: pl.DataFrame,
) -> None:
    """IsolationForest debe respetar contamination ±1 % sobre dataset suficiente.

    Generamos un dataset multivariado limpio (3 bandas i.i.d. N(0,1), 3000
    muestras) sin outliers inyectados — el comportamiento de IsolationForest
    está dominado por el parámetro `contamination` cuando los datos son
    homogéneos, por lo que el pct reportado debe coincidir con el objetivo
    dentro de ±1 punto porcentual.
    """
    rng = np.random.default_rng(42)
    n = 3000
    rows = []
    for band in ["B02", "B04", "B08"]:
        vals = rng.normal(loc=0.0, scale=1.0, size=n)
        for i, v in enumerate(vals):
            rows.append(
                {
                    "patch_id": "p0",
                    "t": 0,
                    "y": i // 100,
                    "x": i % 100,
                    "band": band,
                    "value": float(v),
                }
            )
    clean_df = pl.DataFrame(rows)
    out = detect_outliers_isoforest(clean_df, contamination=0.05, seed=42)
    assert not out.is_empty()
    pct = out["pct_outliers"][0]
    # AC-4 del plan US-010 §6: contamination respetada ±1 %.
    assert abs(pct - 5.0) <= 1.0, f"pct_outliers={pct:.2f}, esperado 5.0 ± 1.0"
    # `contamination_target` reportado coincide con el input *100.
    assert out["contamination_target"][0] == pytest.approx(5.0)


def test_detect_outliers_isoforest_detects_injected_outliers(
    df_with_known_outliers: pl.DataFrame,
) -> None:
    """Con outliers inyectados extremos (loc=50), pct supera el contamination objetivo."""
    out = detect_outliers_isoforest(df_with_known_outliers, contamination=0.05, seed=42)
    assert not out.is_empty()
    pct = out["pct_outliers"][0]
    # Los 10/1010 = ~0.99% outliers inyectados son detectados por IsoForest
    # junto con la contamination objetivo del 5%; debe quedar acotado en [3, 8]%.
    assert 3.0 <= pct <= 8.0


def test_detect_outliers_isoforest_empty_returns_empty() -> None:
    """Sin datos suficientes retorna DataFrame vacío."""
    empty = pl.DataFrame(
        schema={
            "patch_id": pl.Utf8,
            "t": pl.Int64,
            "y": pl.Int64,
            "x": pl.Int64,
            "band": pl.Utf8,
            "value": pl.Float64,
        }
    )
    out = detect_outliers_isoforest(empty)
    assert out.is_empty()
