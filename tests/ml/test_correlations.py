"""Tests para `ml.analysis.correlations` (US-012).

Fixtures sinteticos cubren las 7 funciones del modulo:
- Bandas Sentinel-2 conocidas para validar indices (NDVI en [-1, 1]).
- Correlaciones inyectadas controladas para `correlation_pair`.
- 3 features independientes + 1 clon perfecto para `vif_table`.
- 5 series sinusoidales con pico conocido en doy=180 para `phenology_peaks`
  y `acf_pacf_per_parcel`.
- 100 series mono-pico vs 100 series doble-ciclo para `dtw_cluster_temporal`.
- DataFrame ERA5 sintetico 2018-2024 + NDVI anual para `era5_ndvi_anomaly`.

statsmodels / tslearn estan en el grupo `ml` de poetry; los tests se saltan
limpiamente si no estan instalados.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

from ml.analysis.correlations import (
    SPECTRAL_INDICES_CORE,
    acf_pacf_per_parcel,
    compute_indices_subset,
    correlation_pair,
    dtw_cluster_temporal,
    era5_ndvi_anomaly,
    phenology_peaks,
    vif_table,
)

_HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
_HAS_TSLEARN = importlib.util.find_spec("tslearn") is not None


# -------------------------------------------------------------------
# compute_indices_subset
# -------------------------------------------------------------------


def _synthetic_bands(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Construye DataFrame con bandas Sentinel-2 dentro de rangos plausibles."""
    rng = np.random.default_rng(seed)
    b02 = rng.uniform(0.02, 0.15, size=n)
    b03 = rng.uniform(0.04, 0.20, size=n)
    b04 = rng.uniform(0.05, 0.25, size=n)
    b05 = rng.uniform(0.10, 0.35, size=n)
    b08 = rng.uniform(0.20, 0.60, size=n)
    b11 = rng.uniform(0.10, 0.40, size=n)
    return pl.DataFrame(
        {
            "B02": b02,
            "B03": b03,
            "B04": b04,
            "B05": b05,
            "B08": b08,
            "B11": b11,
        }
    )


def test_spectral_indices_core_has_six_entries() -> None:
    """El catalogo expone exactamente los 6 indices core US-012."""
    assert set(SPECTRAL_INDICES_CORE.keys()) == {"NDVI", "NDWI", "NDMI", "EVI", "SAVI", "NDRE"}


def test_compute_indices_subset_appends_six_columns() -> None:
    """`compute_indices_subset` anade las 6 columnas sin tocar las originales."""
    df = _synthetic_bands()
    out = compute_indices_subset(df)
    for idx in ("NDVI", "NDWI", "NDMI", "EVI", "SAVI", "NDRE"):
        assert idx in out.columns
    # Bandas originales intactas
    for b in ("B02", "B03", "B04", "B05", "B08", "B11"):
        assert b in out.columns


def test_compute_indices_subset_ndvi_in_range() -> None:
    """NDVI esta acotado en [-1, 1] para bandas plausibles."""
    df = _synthetic_bands(n=500)
    out = compute_indices_subset(df, indices=["NDVI"])
    ndvi = out["NDVI"].drop_nulls().to_numpy()
    assert ndvi.size == 500
    assert (ndvi >= -1.0001).all()
    assert (ndvi <= 1.0001).all()


def test_compute_indices_subset_unknown_index_raises() -> None:
    """Un indice desconocido levanta ValueError que nombra el indice culpable.

    La asercion es sobre el nombre del indice, no sobre la redaccion del mensaje: cuando
    ``c646263`` tradujo los mensajes al ingles esta prueba se quedo en rojo, y como la CI
    no corre este fichero (ver ``scripts/ci_test_coverage_check.py``) siguio asi meses.
    """
    with pytest.raises(ValueError, match="XYZ"):
        compute_indices_subset(_synthetic_bands(), indices=["XYZ"])


def test_compute_indices_subset_empty_df() -> None:
    """DataFrame vacio devuelve DataFrame vacio (con columnas nuevas null) sin error."""
    empty = pl.DataFrame(
        {
            "B02": pl.Series([], dtype=pl.Float64),
            "B03": pl.Series([], dtype=pl.Float64),
            "B04": pl.Series([], dtype=pl.Float64),
            "B05": pl.Series([], dtype=pl.Float64),
            "B08": pl.Series([], dtype=pl.Float64),
            "B11": pl.Series([], dtype=pl.Float64),
        }
    )
    out = compute_indices_subset(empty)
    assert "NDVI" in out.columns
    assert out.is_empty()


def test_compute_indices_subset_missing_band_returns_original() -> None:
    """Si faltan bandas requeridas, el df vuelve sin cambios."""
    df = pl.DataFrame({"B02": [0.05, 0.06], "B04": [0.1, 0.12]})
    out = compute_indices_subset(df)
    # No tiene B08, no se pueden calcular los indices
    assert "NDVI" not in out.columns
    assert out.equals(df)


def test_compute_indices_subset_subset_selection() -> None:
    """Pidiendo solo NDVI/SAVI no agrega los otros indices."""
    df = _synthetic_bands(n=50)
    out = compute_indices_subset(df, indices=["NDVI", "SAVI"])
    assert "NDVI" in out.columns
    assert "SAVI" in out.columns
    for missing in ("NDWI", "NDMI", "EVI", "NDRE"):
        assert missing not in out.columns


def _synthetic_bands_dn(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """DataFrame con bandas Sentinel-2 en DN crudo PASTIS-R (int16, 0-10000).

    Replica el formato de PASTIS-R con artefactos BOA realistas:
    - rangos tipicos 500-5000 DN (reflectancia 0.05-0.50 tras escala 1e-4),
    - ~5 % de pixeles con DN negativo en B04 (artefactos BOA),
    - ~3 % de pixeles con DN saturados > 15000 en B08 (nubes).
    """
    rng = np.random.default_rng(seed)
    b02 = rng.integers(500, 1500, size=n).astype(np.float64)
    b03 = rng.integers(700, 2000, size=n).astype(np.float64)
    b04 = rng.integers(800, 2500, size=n).astype(np.float64)
    b05 = rng.integers(1500, 3500, size=n).astype(np.float64)
    b08 = rng.integers(3000, 6000, size=n).astype(np.float64)
    b11 = rng.integers(1500, 4000, size=n).astype(np.float64)
    # Inyecta artefactos: 5 % negativos en B04, 3 % saturados en B08
    neg_idx = rng.choice(n, size=max(1, n // 20), replace=False)
    sat_idx = rng.choice(n, size=max(1, n // 33), replace=False)
    b04[neg_idx] = -200.0
    b08[sat_idx] = 18000.0
    return pl.DataFrame({"B02": b02, "B03": b03, "B04": b04, "B05": b05, "B08": b08, "B11": b11})


def test_compute_indices_subset_scale_brings_ndvi_into_range() -> None:
    """`scale=1e-4` sobre DN crudo limpio (sin artefactos) entrega NDVI en [-1, 1].

    Nota: con artefactos BOA (DN negativo) y solo `scale`, NDVI puede exceder 1.0;
    para garantizar rango fisico hay que combinar con `mask_invalid_band_range`
    o `clip_negative` (cubierto en tests posteriores).
    """
    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame(
        {
            "B02": rng.integers(500, 1500, size=n).astype(np.float64),
            "B03": rng.integers(700, 2000, size=n).astype(np.float64),
            "B04": rng.integers(800, 2500, size=n).astype(np.float64),
            "B05": rng.integers(1500, 3500, size=n).astype(np.float64),
            "B08": rng.integers(3000, 6000, size=n).astype(np.float64),
            "B11": rng.integers(1500, 4000, size=n).astype(np.float64),
        }
    )
    out_dn = compute_indices_subset(df, indices=["NDVI"])
    out_scaled = compute_indices_subset(df, indices=["NDVI"], scale=1e-4)
    # NDVI normalizado es invariante al factor de escala global (a-b)/(a+b);
    # comprobamos el contrato del parametro: produce los mismos valores y caen en [-1, 1].
    ndvi_dn = out_dn["NDVI"].to_numpy()
    ndvi_sc = out_scaled["NDVI"].to_numpy()
    np.testing.assert_allclose(ndvi_dn, ndvi_sc, atol=1e-6)
    assert ndvi_sc.size == n
    assert (ndvi_sc >= -1.0001).all()
    assert (ndvi_sc <= 1.0001).all()


def test_compute_indices_subset_clip_negative_replaces_negative_dn() -> None:
    """`clip_negative=True` elimina los DN negativos antes del computo de indices."""
    df = pl.DataFrame(
        {
            "B02": [500.0, 600.0],
            "B03": [700.0, 800.0],
            "B04": [-200.0, 1000.0],
            "B05": [1500.0, 1600.0],
            "B08": [4000.0, 4500.0],
            "B11": [2000.0, 2100.0],
        }
    )
    out = compute_indices_subset(df, indices=["NDVI"], scale=1e-4, clip_negative=True)
    # Fila 0: B04 clipeado a 0 -> NDVI = (B08-0)/(B08+0) = 1.0
    # Fila 1: B04 = 0.10, B08 = 0.45 -> NDVI = 0.35/0.55 ~= 0.636
    ndvi = out["NDVI"].to_numpy()
    assert ndvi[0] == pytest.approx(1.0, abs=1e-4)
    assert ndvi[1] == pytest.approx(0.35 / 0.55, abs=1e-3)


def test_compute_indices_subset_mask_invalid_band_range_drops_rows() -> None:
    """`mask_invalid_band_range` descarta timesteps con bandas fuera del rango post-escala."""
    df = _synthetic_bands_dn(n=400, seed=7)
    n_in = df.height
    out = compute_indices_subset(
        df,
        indices=["NDVI", "EVI"],
        scale=1e-4,
        mask_invalid_band_range=(0.0, 1.5),
    )
    # Filas con DN negativo (5 %) o saturado > 15000 (3 %) deben quedar fuera.
    # Permitimos solape de inyeccion (mismo idx puede haber recibido ambos artefactos),
    # asi que el descarte real esta entre [n*0.05, n*0.08].
    n_out = out.height
    assert n_out < n_in
    assert n_out >= int(n_in * 0.90)
    # Los NDVI sobrevivientes deben estar dentro del rango fisico.
    ndvi = out["NDVI"].drop_nulls().to_numpy()
    assert (ndvi >= -1.0001).all()
    assert (ndvi <= 1.0001).all()


def test_compute_indices_subset_mask_invalid_band_range_drops_all() -> None:
    """Si todas las filas violan el rango, devuelve DataFrame vacio con columnas Float64."""
    df = pl.DataFrame(
        {
            "B02": [-1.0, -2.0],
            "B03": [-1.0, -2.0],
            "B04": [-1.0, -2.0],
            "B05": [-1.0, -2.0],
            "B08": [-1.0, -2.0],
            "B11": [-1.0, -2.0],
        }
    )
    out = compute_indices_subset(
        df,
        indices=["NDVI"],
        scale=1.0,
        mask_invalid_band_range=(0.0, 1.0),
    )
    assert out.is_empty()
    assert "NDVI" in out.columns
    assert out.schema["NDVI"] == pl.Float64


def test_compute_indices_subset_clip_evi_range_bounds_output() -> None:
    """`clip_evi_range` acota EVI post-computo al rango pedido."""
    # Construimos un caso donde el denominador EVI se acerca a cero y produce outlier.
    # B08 + 6*B04 - 7.5*B02 + 1 ~ 0 cuando B02 domina; usamos valores que disparan eso.
    df = pl.DataFrame(
        {
            "B02": [0.50, 0.05],
            "B03": [0.10, 0.10],
            "B04": [0.05, 0.10],
            "B05": [0.20, 0.20],
            "B08": [0.10, 0.50],
            "B11": [0.20, 0.20],
        }
    )
    out_unclipped = compute_indices_subset(df, indices=["EVI"])
    out_clipped = compute_indices_subset(df, indices=["EVI"], clip_evi_range=(-1.0, 2.0))
    evi_un = out_unclipped["EVI"].to_numpy()
    evi_cl = out_clipped["EVI"].to_numpy()
    # La fila 0 debe ser un outlier sin clip y caer dentro del rango con clip.
    assert (evi_cl >= -1.0001).all()
    assert (evi_cl <= 2.0001).all()
    # La fila 1 (denominador sano) no debe cambiar.
    assert evi_cl[1] == pytest.approx(evi_un[1], abs=1e-6)


def test_compute_indices_subset_clip_negative_and_mask_are_mutually_exclusive() -> None:
    """Pasar ambos `clip_negative=True` y `mask_invalid_band_range` levanta ValueError."""
    df = _synthetic_bands(n=20)
    with pytest.raises(ValueError, match="mask_invalid_band_range"):
        compute_indices_subset(
            df,
            indices=["NDVI"],
            clip_negative=True,
            mask_invalid_band_range=(0.0, 1.5),
        )


# -------------------------------------------------------------------
# correlation_pair
# -------------------------------------------------------------------


def _df_with_known_corr(n: int = 500, seed: int = 0) -> pl.DataFrame:
    """DataFrame con correlaciones conocidas: A~B alta (~0.9), A~C baja (~0)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = 0.9 * a + 0.1 * rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = -0.7 * a + 0.3 * rng.standard_normal(n)
    return pl.DataFrame({"A": a, "B": b, "C": c, "D": d})


def test_correlation_pair_high_corr_detected() -> None:
    """Pares (A,B) con corr inyectada 0.9 deben aparecer con `abs_corr > 0.8`."""
    df = _df_with_known_corr()
    out = correlation_pair(df, cols_a=["A"], cols_b=["B", "C", "D"], method="pearson")
    assert out.height == 3
    by_pair = {(r["feature_a"], r["feature_b"]): r["corr"] for r in out.iter_rows(named=True)}
    assert by_pair[("A", "B")] > 0.85
    assert abs(by_pair[("A", "C")]) < 0.2
    assert by_pair[("A", "D")] < -0.6


def test_correlation_pair_shape_and_abs_range() -> None:
    """Output shape = `(len(a) * len(b))` y `abs_corr in [0, 1]`."""
    df = _df_with_known_corr()
    out = correlation_pair(df, cols_a=["A", "C"], cols_b=["B", "D"], method="pearson")
    assert out.height == 4
    abs_vals = out["abs_corr"].drop_nulls().to_numpy()
    assert (abs_vals >= 0.0).all()
    assert (abs_vals <= 1.0).all()


def test_correlation_pair_spearman_runs() -> None:
    """Spearman tambien produce el mismo numero de filas."""
    df = _df_with_known_corr(n=200)
    out = correlation_pair(df, cols_a=["A", "B"], cols_b=["C", "D"], method="spearman")
    assert out.height == 4
    assert "corr" in out.columns


def test_correlation_pair_empty_inputs() -> None:
    """DataFrame vacio o subsets vacios -> DataFrame vacio con esquema."""
    out = correlation_pair(pl.DataFrame(), cols_a=["A"], cols_b=["B"])
    assert out.is_empty()
    assert set(["feature_a", "feature_b", "corr", "abs_corr"]).issubset(set(out.columns))
    out2 = correlation_pair(_df_with_known_corr(), cols_a=[], cols_b=["B"])
    assert out2.is_empty()


def test_correlation_pair_missing_column() -> None:
    """Columna inexistente -> DataFrame vacio (modo degradado)."""
    df = _df_with_known_corr()
    out = correlation_pair(df, cols_a=["A"], cols_b=["Z_NO_EXISTE"])
    assert out.is_empty()


# -------------------------------------------------------------------
# vif_table
# -------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels no instalado")
def test_vif_table_flags_clone_as_drop() -> None:
    """3 features independientes + 1 clon perfecto: el clon debe quedar dropped."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = x1.copy()  # clon perfecto de x1
    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    out = vif_table(df, cols=["x1", "x2", "x3", "x4"])
    statuses = {r["feature"]: r["status"] for r in out.iter_rows(named=True)}
    # x4 (o x1) debe quedar marcado como dropped_near_perfect_corr
    assert "dropped_near_perfect_corr" in set(statuses.values())
    # Las features independientes tienen VIF cercano a 1
    vifs = {r["feature"]: r["vif"] for r in out.iter_rows(named=True)}
    for f in ("x2", "x3"):
        if statuses.get(f) == "ok":
            assert vifs[f] < 5.0


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels no instalado")
def test_vif_table_independent_features_have_vif_near_one() -> None:
    """Features completamente independientes deben tener VIF cercano a 1."""
    rng = np.random.default_rng(1)
    n = 1000
    df = pl.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    out = vif_table(df, cols=["a", "b", "c"])
    assert out.height == 3
    for r in out.iter_rows(named=True):
        assert r["status"] == "ok"
        assert r["vif"] < 2.0


def test_vif_table_empty_returns_empty() -> None:
    """Inputs vacios -> DataFrame vacio con esquema."""
    out = vif_table(pl.DataFrame(), cols=["a"])
    assert out.is_empty()
    assert set(["feature", "vif", "status"]).issubset(set(out.columns))


def test_vif_table_missing_col_returns_empty() -> None:
    """Columna ausente -> DataFrame vacio."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    out = vif_table(df, cols=["a", "b"])
    assert out.is_empty()


# -------------------------------------------------------------------
# phenology_peaks
# -------------------------------------------------------------------


def _synthetic_ndvi_sinusoidal(
    n_parcels: int = 5,
    peak_doy: int = 180,
    year: int = 2019,
    n_samples: int = 24,
    seed: int = 42,
) -> pl.DataFrame:
    """Series NDVI sinusoidales con pico inyectado en `peak_doy`."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    start = np.datetime64(f"{year}-01-01")
    step_days = 365 // n_samples
    for p in range(n_parcels):
        # Pico simulado: gaussiana centrada en peak_doy con sigma=45 dias
        for k in range(n_samples):
            day_offset = k * step_days
            date = start + np.timedelta64(day_offset, "D")
            doy = day_offset + 1
            base = 0.7 * np.exp(-((doy - peak_doy) ** 2) / (2 * 45**2))
            ndvi = base + 0.05 * rng.standard_normal()
            rows.append(
                {
                    "parcel_id": f"p{p}",
                    "date": np.datetime_as_string(date, unit="D"),
                    "ndvi": float(ndvi),
                    "class_name": "wheat" if p % 2 == 0 else "corn",
                    "class_id": 1 if p % 2 == 0 else 2,
                }
            )
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").str.to_date())
    return df


def test_phenology_peaks_detects_known_peak_doy() -> None:
    """El pico simulado en doy=180 (junio-julio) se detecta con tolerancia."""
    df = _synthetic_ndvi_sinusoidal(n_parcels=5, peak_doy=180)
    out = phenology_peaks(df)
    assert out.height == 5
    months = out["peak_ndvi_month"].to_list()
    # doy=180 cae en junio (mes 6) o julio (mes 7) segun discretizacion
    assert all(m in (6, 7) for m in months), f"Meses inesperados: {months}"
    assert (out["peak_ndvi_month"].to_numpy() >= 1).all()
    assert (out["peak_ndvi_month"].to_numpy() <= 12).all()


def test_phenology_peaks_empty_input() -> None:
    """DataFrame vacio -> DataFrame vacio con esquema correcto."""
    out = phenology_peaks(pl.DataFrame())
    assert out.is_empty()
    assert "peak_ndvi_value" in out.columns


def test_phenology_peaks_missing_columns_returns_empty() -> None:
    """Falta de columnas requeridas devuelve DataFrame vacio."""
    df = pl.DataFrame({"foo": [1, 2]})
    out = phenology_peaks(df)
    assert out.is_empty()


def test_phenology_peaks_integer_yyyymmdd_date() -> None:
    """Acepta fechas como int YYYYMMDD (formato PASTIS dates-S2)."""
    df = pl.DataFrame(
        {
            "parcel_id": ["p1"] * 4,
            "date": [20190401, 20190615, 20190820, 20191015],
            "ndvi": [0.2, 0.85, 0.6, 0.3],
            "class_name": ["wheat"] * 4,
        }
    )
    out = phenology_peaks(df)
    assert out.height == 1
    assert int(out["peak_ndvi_month"][0]) == 6
    assert int(out["peak_ndvi_year"][0]) == 2019


# -------------------------------------------------------------------
# acf_pacf_per_parcel
# -------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels no instalado")
def test_acf_pacf_per_parcel_shape_and_lag_zero() -> None:
    """ACF[0] = 1.0 y |acf|, |pacf| <= 1."""
    df = _synthetic_ndvi_sinusoidal(n_parcels=3, peak_doy=180)
    out = acf_pacf_per_parcel(df, max_lag=4)
    assert out.height > 0
    assert {"parcel_id", "class_name", "lag", "acf", "pacf"}.issubset(set(out.columns))
    lag0 = out.filter(pl.col("lag") == 0)
    for r in lag0.iter_rows(named=True):
        assert abs(r["acf"] - 1.0) < 1e-6
    vals = out["acf"].drop_nulls().to_numpy()
    assert (vals >= -1.0001).all() and (vals <= 1.0001).all()


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels no instalado")
def test_acf_pacf_per_parcel_filters_classes() -> None:
    """Si llega `class_id` solo se procesan `[1, 18]`; bg(0) y void(19) se excluyen."""
    df = _synthetic_ndvi_sinusoidal(n_parcels=2, peak_doy=180)
    # Forzar class_id fuera de rango en una parcela
    df = df.with_columns(
        pl.when(pl.col("parcel_id") == "p0").then(0).otherwise(pl.col("class_id")).alias("class_id")
    )
    out = acf_pacf_per_parcel(df, max_lag=3)
    pids = set(out["parcel_id"].to_list())
    assert "p0" not in pids
    assert "p1" in pids


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels no instalado")
def test_acf_pacf_per_parcel_empty_input() -> None:
    """DataFrame vacio -> vacio con esquema valido."""
    out = acf_pacf_per_parcel(pl.DataFrame(), max_lag=3)
    assert out.is_empty()


# -------------------------------------------------------------------
# dtw_cluster_temporal
# -------------------------------------------------------------------


def _two_cluster_series(
    n_per_cluster: int = 50,
    n_samples: int = 18,
    year: int = 2019,
    seed: int = 42,
) -> pl.DataFrame:
    """Genera 2 clusters DTW: mono-pico (clase A) y doble-ciclo (clase B)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    start = np.datetime64(f"{year}-01-01")
    step_days = 365 // n_samples
    doys = np.array([(k * step_days) + 1 for k in range(n_samples)])
    for p in range(n_per_cluster):
        # Mono pico: gaussiana centrada en doy=180
        signal_a = 0.75 * np.exp(-((doys - 180) ** 2) / (2 * 45**2))
        noise = 0.03 * rng.standard_normal(n_samples)
        for k in range(n_samples):
            date = start + np.timedelta64(int(doys[k] - 1), "D")
            rows.append(
                {
                    "parcel_id": f"mono_{p}",
                    "date": np.datetime_as_string(date, unit="D"),
                    "ndvi": float(signal_a[k] + noise[k]),
                    "class_name": "mono",
                    "class_id": 1,
                }
            )
    for p in range(n_per_cluster):
        # Doble ciclo: dos gaussianas en doy=100 y doy=260
        signal_b = 0.55 * np.exp(-((doys - 100) ** 2) / (2 * 30**2)) + 0.55 * np.exp(
            -((doys - 260) ** 2) / (2 * 30**2)
        )
        noise = 0.03 * rng.standard_normal(n_samples)
        for k in range(n_samples):
            date = start + np.timedelta64(int(doys[k] - 1), "D")
            rows.append(
                {
                    "parcel_id": f"dual_{p}",
                    "date": np.datetime_as_string(date, unit="D"),
                    "ndvi": float(signal_b[k] + noise[k]),
                    "class_name": "dual",
                    "class_id": 2,
                }
            )
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").str.to_date())
    return df


@pytest.mark.skipif(not _HAS_TSLEARN, reason="tslearn no instalado")
def test_dtw_cluster_temporal_separates_two_patterns() -> None:
    """Dos patrones (mono vs doble ciclo) se agrupan en clusters distintos."""
    df = _two_cluster_series(n_per_cluster=30)
    out, model = dtw_cluster_temporal(df, n_clusters=2, sakoe_chiba_radius=3)
    assert out.height >= 50
    assert model is not None
    assert hasattr(model, "cluster_centers_")
    # Verifica que las dos clases queden mayoritariamente en clusters distintos
    cross = (
        out.group_by(["class_name", "cluster_id"])
        .agg(pl.len().alias("n"))
        .sort(["class_name", "n"], descending=[False, True])
    )
    top_per_class = cross.group_by("class_name").agg(
        pl.col("cluster_id").first().alias("top_cluster")
    )
    clusters_seen = set(top_per_class["top_cluster"].to_list())
    assert len(clusters_seen) == 2, f"Esperaba 2 clusters distintos: {clusters_seen}"


@pytest.mark.skipif(not _HAS_TSLEARN, reason="tslearn no instalado")
def test_dtw_cluster_temporal_returns_centers_shape() -> None:
    """`cluster_centers_` tiene shape `(n_clusters, T, 1)` de tslearn."""
    df = _two_cluster_series(n_per_cluster=20)
    _, model = dtw_cluster_temporal(df, n_clusters=2, sakoe_chiba_radius=3)
    centers = np.asarray(model.cluster_centers_)
    assert centers.shape[0] == 2


def test_dtw_cluster_temporal_empty_input() -> None:
    """DataFrame vacio -> tupla `(df_vacio, None)`."""
    out, model = dtw_cluster_temporal(pl.DataFrame())
    assert out.is_empty()
    assert model is None


@pytest.mark.skipif(not _HAS_TSLEARN, reason="tslearn no instalado")
def test_dtw_cluster_temporal_filters_classes() -> None:
    """`class_id in [1, 18]` filtra bg/void antes de fit."""
    df = _two_cluster_series(n_per_cluster=15)
    df = df.with_columns(
        pl.when(pl.col("parcel_id").str.starts_with("mono_"))
        .then(0)
        .otherwise(pl.col("class_id"))
        .alias("class_id")
    )
    out, _ = dtw_cluster_temporal(df, n_clusters=2, sakoe_chiba_radius=3)
    # Las parcelas mono_* (class_id=0) deberian quedar fuera
    pids = set(out["parcel_id"].to_list())
    assert not any(pid.startswith("mono_") for pid in pids)


# -------------------------------------------------------------------
# era5_ndvi_anomaly
# -------------------------------------------------------------------


def _synthetic_era5_ndvi() -> tuple[pl.DataFrame, pl.DataFrame]:
    """ERA5 sintetico 2018-2024 + NDVI anual para una ROI."""
    years = list(range(2018, 2025))
    precip = [600.0, 650.0, 720.0, 400.0, 380.0, 700.0, 690.0]  # 2021 y 2022 secos
    ndvi = [0.72, 0.74, 0.78, 0.55, 0.50, 0.76, 0.75]
    df_era5 = pl.DataFrame(
        {
            "year": years,
            "roi_name": ["pianura"] * len(years),
            "precip_mm": precip,
        }
    )
    df_ndvi = pl.DataFrame(
        {
            "year": years,
            "roi_name": ["pianura"] * len(years),
            "ndvi_max": ndvi,
        }
    )
    return df_era5, df_ndvi


def test_era5_ndvi_anomaly_marks_dry_years() -> None:
    """Anos con precipitacion <= p25 (380, 400) deben quedar marcados `is_dry_year`."""
    df_era5, df_ndvi = _synthetic_era5_ndvi()
    out = era5_ndvi_anomaly(df_era5, df_ndvi, dry_year_percentile=0.25)
    assert out.height == 7
    dry = out.filter(pl.col("is_dry_year"))
    dry_years = set(dry["year"].to_list())
    # 2022 (380mm) y 2021 (400mm) son los anos secos
    assert 2022 in dry_years


def test_era5_ndvi_anomaly_returns_anomaly_z() -> None:
    """`ndvi_anomaly_z` mantiene media cercana a 0 por ROI."""
    df_era5, df_ndvi = _synthetic_era5_ndvi()
    out = era5_ndvi_anomaly(df_era5, df_ndvi)
    z = out["ndvi_anomaly_z"].drop_nulls().to_numpy()
    assert abs(float(np.mean(z))) < 1e-6


def test_era5_ndvi_anomaly_empty_inputs() -> None:
    """Inputs vacios -> output vacio con esquema correcto."""
    out = era5_ndvi_anomaly(pl.DataFrame(), pl.DataFrame())
    assert out.is_empty()
    assert "is_dry_year" in out.columns


def test_era5_ndvi_anomaly_missing_columns() -> None:
    """Falta de columnas requeridas devuelve DataFrame vacio."""
    df_era5 = pl.DataFrame({"year": [2018], "roi_name": ["a"]})  # falta precip_mm
    df_ndvi = pl.DataFrame({"year": [2018], "roi_name": ["a"], "ndvi_max": [0.7]})
    out = era5_ndvi_anomaly(df_era5, df_ndvi)
    assert out.is_empty()
