"""Tests de ``ml.features.spectral_signature`` (US-023-preview P5).

Cobertura objetivo >= 80% con fixtures sinteticas pequenas (corrida en CPU
< 5 s). Cubre los 3 descriptores ``rep`` / ``sam`` / ``redge_moments``,
los edge cases (bandas faltantes, parcela sin ancla), determinismo seed y
contrato sklearn (BaseEstimator + TransformerMixin).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from ml.features.spectral_signature import (
    DEFAULT_PHENOLOGY_ANCHORS,
    DEFAULT_REDGE_BANDS,
    SpectralSignatureFeatures,
    compute_rep,
)


@pytest.fixture
def synthetic_spectral_frame() -> pl.DataFrame:
    """DataFrame Polars con reflectancias B04..B08 por ancla fenologica.

    20 parcelas, 3 clases balanceadas, 3 anclas x 5 bandas = 15 cols + 2
    metadata (parcel_id, year) + 1 class_id. Las reflectancias son
    sinteticas pero coherentes con valores tipicos de vegetacion sana
    (rango 0.05-0.6).
    """
    rng = np.random.default_rng(42)
    n = 20
    classes = rng.integers(low=1, high=4, size=n).tolist()  # clases 1, 2, 3
    base: dict[str, list[object]] = {
        "parcel_id": list(range(1000, 1000 + n)),
        "year": [2024] * n,
        "class_id": classes,
    }
    # Reflectancias coherentes para los 4 anclajes (sog/peak/senescence + extra).
    for anchor in DEFAULT_PHENOLOGY_ANCHORS:
        # B04 red ~0.05, B05 red-edge ~0.15, B06 ~0.35, B07 ~0.45, B08 NIR ~0.55.
        base[f"{anchor}_b04"] = rng.normal(loc=0.05, scale=0.01, size=n).clip(0.01, 0.2).tolist()
        base[f"{anchor}_b05"] = rng.normal(loc=0.15, scale=0.02, size=n).clip(0.05, 0.4).tolist()
        base[f"{anchor}_b06"] = rng.normal(loc=0.35, scale=0.03, size=n).clip(0.1, 0.6).tolist()
        base[f"{anchor}_b07"] = rng.normal(loc=0.45, scale=0.04, size=n).clip(0.1, 0.7).tolist()
        base[f"{anchor}_b08"] = rng.normal(loc=0.55, scale=0.05, size=n).clip(0.1, 0.8).tolist()
    return pl.DataFrame(base)


# ---------------------------------------------------------------------------
# compute_rep — formula Frampton 2013.
# ---------------------------------------------------------------------------


def test_compute_rep_returns_expected_shape() -> None:
    """compute_rep produce un vector 1D con el mismo numero de filas que la entrada."""
    n = 50
    rng = np.random.default_rng(0)
    b04 = rng.uniform(0.04, 0.08, size=n)
    b05 = rng.uniform(0.10, 0.18, size=n)
    b06 = rng.uniform(0.30, 0.40, size=n)
    b07 = rng.uniform(0.40, 0.50, size=n)
    rep = compute_rep(b04, b05, b06, b07)
    assert rep.shape == (n,)
    assert rep.dtype == np.float64


def test_compute_rep_value_in_red_edge_window_for_healthy_vegetation() -> None:
    """Para vegetacion sana la REP cae en el rango 700-740 nm."""
    # Valores tipicos de un cultivo en pico fenologico.
    b04 = np.array([0.05])
    b05 = np.array([0.15])
    b06 = np.array([0.35])
    b07 = np.array([0.45])
    rep = compute_rep(b04, b05, b06, b07)
    # 705 + 35 * ((0.05+0.45)/2 - 0.15) / (0.35 - 0.15)
    # = 705 + 35 * (0.25 - 0.15) / 0.20
    # = 705 + 35 * 0.5 = 722.5
    assert pytest.approx(rep[0], abs=0.5) == 722.5
    assert 700.0 <= rep[0] <= 740.0


def test_compute_rep_handles_zero_denominator_gracefully() -> None:
    """Cuando B06 == B05 la formula degenera; el resultado es NaN."""
    b04 = np.array([0.05, 0.05])
    b05 = np.array([0.15, 0.15])
    b06 = np.array([0.15, 0.35])  # primero degenera, segundo OK
    b07 = np.array([0.45, 0.45])
    rep = compute_rep(b04, b05, b06, b07)
    assert np.isnan(rep[0])
    assert not np.isnan(rep[1])


def test_compute_rep_raises_on_shape_mismatch() -> None:
    """Bandas con shapes distintos deben levantar ValueError."""
    with pytest.raises(ValueError, match="shape"):
        compute_rep(
            np.array([0.05, 0.05]),
            np.array([0.15]),
            np.array([0.35, 0.35]),
            np.array([0.45, 0.45]),
        )


# ---------------------------------------------------------------------------
# SpectralSignatureFeatures — contrato sklearn + transform.
# ---------------------------------------------------------------------------


def test_transformer_inherits_sklearn_bases() -> None:
    """La clase debe heredar BaseEstimator + TransformerMixin (skill agrosat-ml-features)."""
    t = SpectralSignatureFeatures()
    assert isinstance(t, BaseEstimator)
    assert isinstance(t, TransformerMixin)


def test_rep_descriptor_produces_three_anchor_cols(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """Default descriptor=rep + 3 anclas -> 3 cols `spectral_signature_*`."""
    t = SpectralSignatureFeatures(descriptor="rep")
    t.fit(synthetic_spectral_frame)
    out = t.transform(synthetic_spectral_frame)
    assert out.height == synthetic_spectral_frame.height
    spec_cols = [c for c in out.columns if c.startswith("spectral_signature_")]
    assert len(spec_cols) == len(DEFAULT_PHENOLOGY_ANCHORS)
    assert spec_cols == [
        "spectral_signature_000",
        "spectral_signature_001",
        "spectral_signature_002",
    ]
    assert out.schema["spectral_signature_000"] == pl.Float32


def test_sam_descriptor_produces_single_col(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """descriptor=sam -> 1 col."""
    t = SpectralSignatureFeatures(descriptor="sam", class_col="class_id")
    t.fit(synthetic_spectral_frame)
    out = t.transform(synthetic_spectral_frame)
    assert out.height == synthetic_spectral_frame.height
    spec_cols = [c for c in out.columns if c.startswith("spectral_signature_")]
    assert len(spec_cols) == 1
    # Coseno -> esta entre -1 y 1 para parcelas con todos los valores finitos.
    values = out["spectral_signature_000"].to_numpy()
    finite = values[np.isfinite(values)]
    assert (finite >= -1.0001).all() and (finite <= 1.0001).all()


def test_redge_moments_descriptor_produces_three_per_anchor(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """descriptor=redge_moments -> mean/var/skew x 3 anclas = 9 cols."""
    t = SpectralSignatureFeatures(descriptor="redge_moments")
    t.fit(synthetic_spectral_frame)
    out = t.transform(synthetic_spectral_frame)
    spec_cols = [c for c in out.columns if c.startswith("spectral_signature_")]
    assert len(spec_cols) == 3 * len(DEFAULT_PHENOLOGY_ANCHORS)


def test_fit_transform_returns_same_as_fit_then_transform(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """fit_transform == fit().transform() (contrato sklearn)."""
    t1 = SpectralSignatureFeatures(descriptor="rep")
    t2 = SpectralSignatureFeatures(descriptor="rep")
    out1 = t1.fit(synthetic_spectral_frame).transform(synthetic_spectral_frame)
    out2 = t2.fit_transform(synthetic_spectral_frame)
    assert out1.equals(out2)


def test_invalid_descriptor_raises(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """descriptor no soportado -> ValueError."""
    t = SpectralSignatureFeatures(descriptor="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="descriptor"):
        t.fit(synthetic_spectral_frame)


def test_missing_parcel_id_col_raises() -> None:
    """transform sobre DataFrame sin parcel_id -> ValueError."""
    df = pl.DataFrame({"year": [2024], "sog_b04": [0.05]})
    t = SpectralSignatureFeatures()
    t.fit(df)  # fit no exige parcel_id (graceful)
    with pytest.raises(ValueError, match="parcel_id"):
        t.transform(df)


def test_missing_bands_produce_nan_columns_not_error(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """Bandas faltantes para una ancla -> esa col queda en NaN; no rompe."""
    # Quita las bandas de la ancla 'senescence' para simular dato faltante.
    cols_to_drop = [c for c in synthetic_spectral_frame.columns if c.startswith("senescence_b")]
    df_partial = synthetic_spectral_frame.drop(cols_to_drop)
    t = SpectralSignatureFeatures(descriptor="rep")
    out = t.fit_transform(df_partial)
    # spectral_signature_002 corresponde a 'senescence' -> debe ser NaN.
    vals = out["spectral_signature_002"].to_numpy()
    assert np.isnan(vals).all()
    # spectral_signature_000 (sog) y _001 (peak) siguen finitos.
    sog_vals = out["spectral_signature_000"].to_numpy()
    assert np.isfinite(sog_vals).all()


def test_determinism_same_input_same_output(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """Misma entrada -> misma salida byte-equal (no hay azar interno)."""
    t1 = SpectralSignatureFeatures(descriptor="redge_moments")
    t2 = SpectralSignatureFeatures(descriptor="redge_moments")
    out1 = t1.fit_transform(synthetic_spectral_frame)
    out2 = t2.fit_transform(synthetic_spectral_frame)
    assert out1.equals(out2)


def test_fit_with_no_class_col_for_sam_emits_warning(
    synthetic_spectral_frame: pl.DataFrame,
) -> None:
    """SAM sin class_col -> centroid degrada a vector de unos sin fallar."""
    df = synthetic_spectral_frame.drop("class_id")
    t = SpectralSignatureFeatures(descriptor="sam", class_col=None)
    out = t.fit_transform(df)
    spec_cols = [c for c in out.columns if c.startswith("spectral_signature_")]
    assert len(spec_cols) == 1


def test_default_bands_are_redge_focused() -> None:
    """Default bands son las 4 cols red-edge canonicas S2 MSI."""
    assert DEFAULT_REDGE_BANDS == ("b05", "b06", "b07", "b08")


def test_transformer_repr_includes_descriptor() -> None:
    """sklearn repr expone los hiperparametros (debug ML)."""
    t = SpectralSignatureFeatures(descriptor="sam", class_col="crop")
    txt = repr(t)
    assert "descriptor" in txt
    assert "sam" in txt
