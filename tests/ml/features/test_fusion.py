"""Tests US-016 — ``ml.features.fusion.build_fused_features``.

Cubre AC-1..AC-8, AC-11 (lazy streaming), AC-12 (determinismo) y bordes:
- Shape exacto (N, 2 + 189) con nombres de columna canonicos.
- Bloques inyectados por dependency injection (ae_frame, indices_frame, etc.).
- Bloque FarSLIP opcional (LEFT join, warning si default path missing).
- Errores controlados (parcel_id missing, year mismatch, stats invalidos).
- Determinismo: bytes del parquet identicos en dos builds consecutivos.
- Geometria: cuadrado ~1 ha del fixture demo => elongation cerca de 1.27.
- Ablation: subset de bloques reduce el conteo de columnas.

Convenciones:
- Mock de EE NUNCA se llama (los samplers reales se evitan via injected frames).
- Polars 1.x para asserts (no pandas).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from shapely.geometry import Polygon

from ml.features.fusion import (
    AE_COLS,
    BLOCK_NAMES,
    EXPECTED_COL_COUNT_NO_FARSLIP,
    EXPECTED_COL_COUNT_WITH_FARSLIP,
    FUSION_STATS,
    build_fused_features,
)
from ml.features.spectral_indices import INDEX_NAMES

# ---------------------------------------------------------------------------
# Helpers internos del modulo de tests.
# ---------------------------------------------------------------------------


def _make_indices_frame(
    parcels: gpd.GeoDataFrame, year: int, stats: tuple[str, ...] = FUSION_STATS
) -> pl.DataFrame:
    """Construye un frame indices*stats con datos sinteticos finitos."""
    n = len(parcels)
    pids = parcels["parcel_id"].astype("int64").tolist()
    cols: dict[str, pl.Series] = {
        "parcel_id": pl.Series("parcel_id", pids, dtype=pl.Int64),
        "year": pl.Series("year", [year] * n, dtype=pl.Int16),
    }
    for k, idx in enumerate(INDEX_NAMES):
        for j, stat in enumerate(stats):
            name = f"{idx.lower()}_{stat}"
            values = [round(0.05 * (i + 1) + 0.001 * k + 0.0005 * j, 6) for i in range(n)]
            cols[name] = pl.Series(name, values, dtype=pl.Float64)
    return pl.DataFrame(cols)


def _make_s1_frame(parcels: gpd.GeoDataFrame, year: int) -> pl.DataFrame:
    n = len(parcels)
    pids = parcels["parcel_id"].astype("int64").tolist()
    cols: dict[str, pl.Series] = {
        "parcel_id": pl.Series("parcel_id", pids, dtype=pl.Int64),
        "year": pl.Series("year", [year] * n, dtype=pl.Int16),
    }
    for pol in ("vv", "vh"):
        for stat in FUSION_STATS:
            name = f"s1_{pol}_{stat}"
            values = [round(-12.0 + i * 0.7, 4) for i in range(n)]
            cols[name] = pl.Series(name, values, dtype=pl.Float64)
    return pl.DataFrame(cols)


def _make_srtm_frame(parcels: gpd.GeoDataFrame) -> pl.DataFrame:
    n = len(parcels)
    pids = parcels["parcel_id"].astype("int64").tolist()
    aspect_pool = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return pl.DataFrame(
        {
            "parcel_id": pl.Series("parcel_id", pids, dtype=pl.Int64),
            "srtm_elev_mean": pl.Series(
                "srtm_elev_mean", [120.0 + i * 13.0 for i in range(n)], dtype=pl.Float64
            ),
            "srtm_slope_mean": pl.Series(
                "srtm_slope_mean", [2.0 + i * 0.5 for i in range(n)], dtype=pl.Float64
            ),
            "srtm_aspect_dominant": pl.Series(
                "srtm_aspect_dominant",
                [aspect_pool[i % 8] for i in range(n)],
                dtype=pl.Utf8,
            ),
        }
    )


def _make_era5_frame(parcels: gpd.GeoDataFrame, year: int) -> pl.DataFrame:
    """ERA5 sintetico con tmean(m01) < tmean(m07) (hemisferio norte)."""
    n = len(parcels)
    pids = parcels["parcel_id"].astype("int64").tolist()
    cols: dict[str, pl.Series] = {
        "parcel_id": pl.Series("parcel_id", pids, dtype=pl.Int64),
        "year": pl.Series("year", [year] * n, dtype=pl.Int16),
    }
    # Temperatura sinusoidal con minimo en enero y maximo en julio.
    for m in range(1, 13):
        # Simulado: enero ~5C, julio ~24C, sinusoide centrada en mes 7.
        base_t = 14.5 + 9.5 * math.cos(math.pi * (m - 7) / 6.0)
        cols[f"era5_tmean_m{m:02d}"] = pl.Series(
            f"era5_tmean_m{m:02d}", [base_t + 0.2 * i for i in range(n)], dtype=pl.Float64
        )
    for m in range(1, 13):
        cols[f"era5_prec_m{m:02d}"] = pl.Series(
            f"era5_prec_m{m:02d}", [50.0 + (m - 6) * 1.5 + i for i in range(n)], dtype=pl.Float64
        )
    return pl.DataFrame(cols)


def _build_default_injection(
    parcels: gpd.GeoDataFrame,
    *,
    synthetic_ae: pl.DataFrame,
    year: int = 2024,
) -> dict[str, pl.DataFrame]:
    """Atajo: arma los 5 frames inyectables para una build completa offline."""
    return dict(
        ae_frame=synthetic_ae,
        indices_frame=_make_indices_frame(parcels, year=year),
        s1_frame=_make_s1_frame(parcels, year=year),
        srtm_frame=_make_srtm_frame(parcels),
        era5_frame=_make_era5_frame(parcels, year=year),
    )


def _make_square_polygon_gdf(
    *,
    parcel_id: int = 1,
    year: int = 2024,
    side_m: float = 100.0,
    lon0: float = 9.7095,
    lat0: float = 45.4515,
) -> gpd.GeoDataFrame:
    """Construye una GDF de una sola parcela con un cuadrado de lado `side_m` en EPSG:3857."""
    # Construimos en EPSG:3857 y reproyectamos a 4326 para preservar dimensiones metricas.
    import shapely.geometry as sg

    centre = gpd.GeoSeries([sg.Point(lon0, lat0)], crs="EPSG:4326").to_crs("EPSG:3857")
    cx, cy = float(centre.iloc[0].x), float(centre.iloc[0].y)
    half = side_m / 2.0
    sq = Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
            (cx - half, cy - half),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"parcel_id": [parcel_id], "year": [year]},
        geometry=[sq],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    return gdf


def _make_rectangle_gdf(
    *, parcel_id: int = 1, year: int = 2024, width_m: float = 1.0, height_m: float = 100.0
) -> gpd.GeoDataFrame:
    import shapely.geometry as sg

    centre = gpd.GeoSeries([sg.Point(9.7095, 45.4515)], crs="EPSG:4326").to_crs("EPSG:3857")
    cx, cy = float(centre.iloc[0].x), float(centre.iloc[0].y)
    rect = Polygon(
        [
            (cx - width_m / 2, cy - height_m / 2),
            (cx + width_m / 2, cy - height_m / 2),
            (cx + width_m / 2, cy + height_m / 2),
            (cx - width_m / 2, cy + height_m / 2),
            (cx - width_m / 2, cy - height_m / 2),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"parcel_id": [parcel_id], "year": [year]},
        geometry=[rect],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    return gdf


def _make_octagon_gdf(
    *, parcel_id: int = 1, year: int = 2024, radius_m: float = 50.0
) -> gpd.GeoDataFrame:
    """Octagono regular = aproximacion circular para Polsby-Popper cercano a 1."""
    import shapely.geometry as sg

    centre = gpd.GeoSeries([sg.Point(9.7095, 45.4515)], crs="EPSG:4326").to_crs("EPSG:3857")
    cx, cy = float(centre.iloc[0].x), float(centre.iloc[0].y)
    pts = []
    n = 16  # poligono regular de 16 lados, casi circulo
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        pts.append((cx + radius_m * math.cos(ang), cy + radius_m * math.sin(ang)))
    pts.append(pts[0])
    poly = Polygon(pts)
    gdf = gpd.GeoDataFrame(
        {"parcel_id": [parcel_id], "year": [year]},
        geometry=[poly],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# AC-1: shape + nombres exactos.
# ---------------------------------------------------------------------------


def test_fused_vector_shape_and_columns(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """191 nombres exactos (parcel_id + year + 189 de los 6 bloques)."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, blocks=BLOCK_NAMES, **injected)

    assert df.height == len(parcels)
    assert df.width == 2 + EXPECTED_COL_COUNT_NO_FARSLIP

    cols = df.columns
    assert cols[0] == "parcel_id"
    assert cols[1] == "year"

    # Bloques esperados (prefijos / sufijos).
    ae_cols = [c for c in cols if c.startswith("ae_")]
    assert len(ae_cols) == 64
    assert ae_cols[0] == "ae_00" and ae_cols[-1] == "ae_63"

    idx_stats = [c for c in cols if any(c.startswith(f"{idx.lower()}_") for idx in INDEX_NAMES)]
    assert len(idx_stats) == len(INDEX_NAMES) * len(FUSION_STATS) == 85

    s1_cols = [c for c in cols if c.startswith("s1_")]
    assert set(s1_cols) == {f"s1_{pol}_{stat}" for pol in ("vv", "vh") for stat in FUSION_STATS}
    assert len(s1_cols) == 10

    srtm_cols = [c for c in cols if c.startswith("srtm_")]
    assert set(srtm_cols) == {"srtm_elev_mean", "srtm_slope_mean", "srtm_aspect_dominant"}

    era5_cols = [c for c in cols if c.startswith("era5_")]
    assert len(era5_cols) == 24
    for m in range(1, 13):
        assert f"era5_tmean_m{m:02d}" in era5_cols
        assert f"era5_prec_m{m:02d}" in era5_cols

    geom_cols = [c for c in cols if c.startswith("geom_")]
    assert set(geom_cols) == {"geom_area_ha", "geom_perimeter_m", "geom_elongation"}


# ---------------------------------------------------------------------------
# AC-2: bloque AlphaEarth.
# ---------------------------------------------------------------------------


def test_alphaearth_block_64dim_finite(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """64 cols ae_*, sin NaN/Inf cuando inyectamos ae_frame."""
    parcels = parcels_fixture_3regions
    df = build_fused_features(
        parcels,
        year=2024,
        blocks=("alphaearth",),
        ae_frame=synthetic_alphaearth_64d,
    )
    ae_cols = [c for c in df.columns if c.startswith("ae_")]
    assert len(ae_cols) == 64
    # Sin NaN ni Inf en datos sinteticos finitos.
    arr = df.select(ae_cols).to_numpy()
    assert np.isfinite(arr).all()
    # Dtype: el contrato del helper preserva float64 (no fuerza float32) cuando
    # se inyectan datos. Validamos que sea numerico.
    for c in AE_COLS:
        assert df.schema[c].is_numeric()


def test_alphaearth_block_none_injection_allows_empty_cols(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    """Sin ae_frame inyectado y sin GEE real, las 64 cols ae_* existen pero
    estan rellenas con null (el contrato no rompe)."""
    parcels = parcels_fixture_3regions
    df = build_fused_features(parcels, year=2024, blocks=("alphaearth",))
    for c in AE_COLS:
        assert c in df.columns
    # Todas las filas estan en null (sin GEE real).
    nulls = df.select([pl.col(c).is_null().sum().alias(c) for c in AE_COLS])
    assert int(nulls.row(0)[0]) == df.height


# ---------------------------------------------------------------------------
# AC-3: bloque indices x stats (5 * 17 = 85).
# ---------------------------------------------------------------------------


def test_indices_stats_block_85cols(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    parcels = parcels_fixture_3regions
    indices_frame = _make_indices_frame(parcels, year=2024)
    df = build_fused_features(
        parcels,
        year=2024,
        blocks=("indices_stats",),
        indices_frame=indices_frame,
    )
    idx_cols = [c for c in df.columns if c not in ("parcel_id", "year")]
    assert len(idx_cols) == 85
    # Validamos que existe la combinacion {idx}_{stat} para los 17 x 5.
    for idx in INDEX_NAMES:
        for stat in FUSION_STATS:
            assert f"{idx.lower()}_{stat}" in idx_cols


# ---------------------------------------------------------------------------
# AC-4: bloque Sentinel-1.
# ---------------------------------------------------------------------------


def test_s1_block_10cols_finite(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    parcels = parcels_fixture_3regions
    s1_frame = _make_s1_frame(parcels, year=2024)
    df = build_fused_features(parcels, year=2024, blocks=("sentinel1",), s1_frame=s1_frame)
    s1_cols = [c for c in df.columns if c.startswith("s1_")]
    assert len(s1_cols) == 10
    assert {c for c in s1_cols} == {
        f"s1_{pol}_{stat}" for pol in ("vv", "vh") for stat in FUSION_STATS
    }
    arr = df.select(s1_cols).to_numpy()
    assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# AC-5: bloque SRTM.
# ---------------------------------------------------------------------------


def test_srtm_block_3cols_with_synthetic_terrain(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    parcels = parcels_fixture_3regions
    srtm_frame = _make_srtm_frame(parcels)
    df = build_fused_features(parcels, year=2024, blocks=("srtm",), srtm_frame=srtm_frame)
    srtm_cols = [c for c in df.columns if c.startswith("srtm_")]
    assert set(srtm_cols) == {"srtm_elev_mean", "srtm_slope_mean", "srtm_aspect_dominant"}
    # Aspect dominante: string cardinal de los 8 cuadrantes.
    valid_cardinals = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    aspects = df.get_column("srtm_aspect_dominant").to_list()
    assert all(a in valid_cardinals for a in aspects)


# ---------------------------------------------------------------------------
# AC-6: ERA5 mensual.
# ---------------------------------------------------------------------------


def test_era5_monthly_block_24cols(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    parcels = parcels_fixture_3regions
    era5_frame = _make_era5_frame(parcels, year=2024)
    df = build_fused_features(parcels, year=2024, blocks=("era5_monthly",), era5_frame=era5_frame)
    era5_cols = [c for c in df.columns if c.startswith("era5_")]
    assert len(era5_cols) == 24
    # jan < jul en hemisferio norte (sintetico, pero respeta la fisica).
    jan = float(df.get_column("era5_tmean_m01").to_numpy().mean())
    jul = float(df.get_column("era5_tmean_m07").to_numpy().mean())
    assert jan < jul


# ---------------------------------------------------------------------------
# AC-7: bloque geometria.
# ---------------------------------------------------------------------------


def test_geometry_block_known_shapes() -> None:
    """Cuadrado 1 ha (lado 100m) y rectangulo 1x100m con metricas esperadas."""
    # Cuadrado lado 100 m => area = 10000 m2 = 1 ha, perimetro = 400 m.
    square = _make_square_polygon_gdf(parcel_id=1, year=2024, side_m=100.0)
    df_sq = build_fused_features(square, year=2024, blocks=("geometry",))
    row_sq = df_sq.row(0, named=True)
    assert math.isclose(row_sq["geom_area_ha"], 1.0, rel_tol=0.05)
    assert math.isclose(row_sq["geom_perimeter_m"], 400.0, rel_tol=0.05)
    # Polsby-Popper inverso: 400^2 / (4 pi * 10000) ~= 1.273.
    assert math.isclose(row_sq["geom_elongation"], 400.0**2 / (4 * math.pi * 10000.0), rel_tol=0.05)

    # Rectangulo 1x100 => area=100 m2, perimetro=202 m, elongation = 202^2/(4*pi*100) ~= 32.5.
    rect = _make_rectangle_gdf(parcel_id=2, year=2024, width_m=1.0, height_m=100.0)
    df_rt = build_fused_features(rect, year=2024, blocks=("geometry",))
    row_rt = df_rt.row(0, named=True)
    expected_elongation = (2 * (1.0 + 100.0)) ** 2 / (4 * math.pi * 100.0)
    assert math.isclose(row_rt["geom_elongation"], expected_elongation, rel_tol=0.05)
    # Mucho mas alto que 1.27 (alargado).
    assert row_rt["geom_elongation"] > row_sq["geom_elongation"]


def test_geometry_polsby_popper_value_for_perfect_circle_approx() -> None:
    """Octagono ~circulo => elongation cercano a 1.0 (Polsby-Popper)."""
    octa = _make_octagon_gdf(parcel_id=1, year=2024, radius_m=50.0)
    df = build_fused_features(octa, year=2024, blocks=("geometry",))
    elong = float(df.row(0, named=True)["geom_elongation"])
    # Para un poligono de 16 lados muy cercano a un circulo, el ratio P^2/(4*pi*A)
    # se queda muy proximo a 1.0 (< 1.05 en la practica).
    assert 1.0 <= elong < 1.05


# ---------------------------------------------------------------------------
# AC-8: bloque FarSLIP opcional.
# ---------------------------------------------------------------------------


def test_farslip_block_optional_left_join(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con parquet sintetico => cols farslip_000..farslip_511; sin parquet => no se incluye.

    US-023-preview P2 promovio FarSLIP al path canonico, asi que para validar
    el branch ``default_path_not_found`` cambiamos el cwd a un tmp_path donde
    el path relativo ``data/farslip/embeddings_pastis.parquet`` no existe.
    """
    parcels = parcels_fixture_3regions
    n = len(parcels)
    far_cols = {f"farslip_{i:03d}": [0.01 * (i + 1)] * n for i in range(512)}
    schema = {"parcel_id": pl.Int64}
    schema.update({c: pl.Float32 for c in far_cols})
    far_df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            **far_cols,
        },
        schema=schema,
    )
    far_path = tmp_path / "embeddings_pastis.parquet"
    far_df.write_parquet(far_path)

    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df_with = build_fused_features(
        parcels,
        year=2024,
        include_farslip=True,
        farslip_path=str(far_path),
        **injected,
    )
    assert df_with.width == 2 + EXPECTED_COL_COUNT_WITH_FARSLIP
    assert "farslip_000" in df_with.columns
    assert "farslip_511" in df_with.columns
    assert df_with.schema["farslip_000"] == pl.Float32

    # Sin parquet (default path no existe en tmp_path) => warning + no incluido.
    monkeypatch.chdir(tmp_path)
    df_without = build_fused_features(
        parcels,
        year=2024,
        include_farslip=True,
        farslip_path=None,
        **injected,
    )
    assert "farslip_000" not in df_without.columns
    assert df_without.width == 2 + EXPECTED_COL_COUNT_NO_FARSLIP


def test_fusion_with_farslip_returns_701_cols(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Path explicito a FarSLIP fixture => total 2 + 701 cols."""
    parcels = parcels_fixture_3regions
    n = len(parcels)
    far = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            **{f"farslip_{i:03d}": [0.0] * n for i in range(512)},
        }
    )
    far_path = tmp_path / "embeddings.parquet"
    far.write_parquet(far_path)
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_farslip=True,
        farslip_path=str(far_path),
        **injected,
    )
    assert df.width == 2 + EXPECTED_COL_COUNT_WITH_FARSLIP


# ---------------------------------------------------------------------------
# AC-11: lazy streaming.
# ---------------------------------------------------------------------------


def test_lazy_streaming_collect(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """Con ``lazy=True`` la pipeline interna usa streaming. Validamos que un
    build con ``lazy=True`` y ``lazy=False`` produce el mismo contenido
    (ordenando por ``parcel_id`` para tolerar reordering del streaming engine,
    que es propiedad documentada de Polars streaming JOIN)."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df_lazy = build_fused_features(parcels, year=2024, lazy=True, **injected).sort("parcel_id")
    df_eager = build_fused_features(parcels, year=2024, lazy=False, **injected).sort("parcel_id")
    assert df_lazy.equals(df_eager)


def test_fusion_lazy_false_returns_eager_df(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, lazy=False, **injected)
    assert isinstance(df, pl.DataFrame)
    assert df.height == len(parcels)


# ---------------------------------------------------------------------------
# Validaciones de entrada.
# ---------------------------------------------------------------------------


def test_build_raises_on_missing_parcel_id() -> None:
    """Falta `parcel_id` => ValueError descriptivo."""
    gdf = gpd.GeoDataFrame(
        {"year": [2024]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="parcel_id"):
        build_fused_features(gdf, year=2024, blocks=("geometry",))


def test_build_raises_on_year_mismatch(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    """Si `year` no coincide con `parcels['year']`, ValueError."""
    with pytest.raises(ValueError, match="year"):
        build_fused_features(parcels_fixture_3regions, year=2099, blocks=("geometry",))


def test_fusion_empty_parcels_raises() -> None:
    """GeoDataFrame vacio: la build no debe crashear; produce frame vacio o
    levanta error controlado segun la implementacion. Aceptamos cualquiera
    de las dos pero validamos que NO produce silenciosamente N != 0."""
    empty = gpd.GeoDataFrame(
        {"parcel_id": [], "year": []},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    try:
        df = build_fused_features(empty, year=2024, blocks=("geometry",))
    except (ValueError, IndexError):
        # Implementacion conservadora: levanta. Aceptable.
        return
    # Si no levanta, el frame debe ser vacio.
    assert df.height == 0


def test_fusion_year_int_validation(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """`year=2024` (int positivo razonable) debe funcionar; `year` con otro
    valor distinto al de las parcelas levanta ValueError (cubierto en
    test_build_raises_on_year_mismatch). Aqui validamos el caso happy path
    con un int."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, **injected)
    assert df.height == len(parcels)


def test_fusion_stats_subset_changes_col_count(
    parcels_fixture_3regions: gpd.GeoDataFrame,
) -> None:
    """Con un subset de stats el bloque indices x stats cambia su conteo."""
    parcels = parcels_fixture_3regions
    custom_stats = ("mean", "p50")
    indices_frame = _make_indices_frame(parcels, year=2024, stats=custom_stats)
    df = build_fused_features(
        parcels,
        year=2024,
        blocks=("indices_stats",),
        stats=custom_stats,
        indices_frame=indices_frame,
    )
    idx_cols = [c for c in df.columns if c not in ("parcel_id", "year")]
    assert len(idx_cols) == len(INDEX_NAMES) * len(custom_stats)


def test_fusion_handles_polygon_with_invalid_geometry() -> None:
    """Geometria invalida (ej. lineas degeneradas): no debe propagar excepcion."""
    # Polygon casi degenerado (area ~ 0): genera elongation Inf o muy alto.
    pts = [(0.0, 0.0), (1e-8, 0.0), (1e-8, 1e-8), (0.0, 1e-8), (0.0, 0.0)]
    gdf = gpd.GeoDataFrame(
        {"parcel_id": [42], "year": [2024]},
        geometry=[Polygon(pts)],
        crs="EPSG:4326",
    )
    df = build_fused_features(gdf, year=2024, blocks=("geometry",))
    # Aceptamos NaN/Inf en elongation, pero el frame debe construirse.
    assert df.height == 1


# ---------------------------------------------------------------------------
# AC-12: determinismo (MD5 byte-equal).
# ---------------------------------------------------------------------------


def test_fusion_determinism_same_inputs_same_md5(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Dos builds consecutivas con los mismos inputs producen parquet identico
    tras ordenar por ``parcel_id`` (el streaming JOIN de Polars no preserva
    orden, propiedad documentada — el contrato AC-12 se cumple sobre
    contenido, no sobre orden fisico de filas)."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df1 = build_fused_features(parcels, year=2024, **injected).sort("parcel_id")
    df2 = build_fused_features(parcels, year=2024, **injected).sort("parcel_id")
    p1 = tmp_path / "fused1.parquet"
    p2 = tmp_path / "fused2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)
    h1 = hashlib.md5(p1.read_bytes(), usedforsecurity=False).hexdigest()
    h2 = hashlib.md5(p2.read_bytes(), usedforsecurity=False).hexdigest()
    assert h1 == h2


# ---------------------------------------------------------------------------
# Ablation por bloques.
# ---------------------------------------------------------------------------


def test_block_ablation_excluding_s1(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """blocks sin `sentinel1` => 10 columnas menos vs build completa."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df_full = build_fused_features(parcels, year=2024, **injected)
    df_no_s1 = build_fused_features(
        parcels,
        year=2024,
        blocks=("alphaearth", "indices_stats", "srtm", "era5_monthly", "geometry"),
        **injected,
    )
    assert df_full.width - df_no_s1.width == 10
    s1_cols_no = [c for c in df_no_s1.columns if c.startswith("s1_")]
    assert s1_cols_no == []


# ---------------------------------------------------------------------------
# End-to-end demo 3 regiones.
# ---------------------------------------------------------------------------


def test_extract_demo_3regions_end_to_end(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
) -> None:
    """Build completa con el fixture demo + frames inyectados sinteticos."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, **injected)
    assert df.height == 9
    assert df.width == 2 + EXPECTED_COL_COUNT_NO_FARSLIP
    # Las 9 parcelas estan presentes y unicas. Con el esquema canonico
    # `parcel_id: Utf8` comparamos contra la representacion string.
    pids = df.get_column("parcel_id").to_list()
    expected = sorted(parcels["parcel_id"].astype("string").tolist())
    assert sorted(pids) == expected
    assert len(set(pids)) == 9


# ---------------------------------------------------------------------------
# Dtypes y orden de primeras columnas.
# ---------------------------------------------------------------------------


def test_fusion_first_two_cols_are_parcel_id_year_correct_dtypes(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """Las primeras 2 cols del frame final son `parcel_id` (Utf8) y `year` (Int16).

    Cambio US-023-preview v2: el esquema canonico de ``parcel_id`` paso a
    ``pl.Utf8`` (ver ``ml.utils.parcel_id``) para soportar ids compuestos
    como ``"{patch_id}_{i}"`` que emergen de los baselines.
    """
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, **injected)
    assert df.columns[0] == "parcel_id"
    assert df.columns[1] == "year"
    assert df.schema["parcel_id"] == pl.Utf8
    assert df.schema["year"] == pl.Int16


# ---------------------------------------------------------------------------
# US-023-preview v2: esquema canonico parcel_id=Utf8 y join cross-dtype.
# ---------------------------------------------------------------------------


def test_fusion_farslip_parquet_int64_joins_with_utf8_parcels(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """FarSLIP parquet con `parcel_id: Int64` se joinea sin error con `parcels` Utf8.

    Bug US-023-preview v2: el esquema canonico de los baselines genera
    `parcel_id` como cadena (`"{patch_id}_{i}"`), pero los parquets FarSLIP
    historicos persistieron `parcel_id: Int64`. Antes del fix, el LEFT JOIN
    fallaba por dtype mismatch y la build emitia un warning saltandose el
    bloque FarSLIP. El fix castea ambos lados a `pl.Utf8` antes del join.
    """
    parcels = parcels_fixture_3regions
    n = len(parcels)
    far_cols = {f"farslip_{i:03d}": [0.03 * (i + 1)] * n for i in range(512)}
    schema: dict[str, pl.DataType] = {"parcel_id": pl.Int64}
    schema.update({c: pl.Float32 for c in far_cols})
    # Construimos `parcel_id` como Int64 (caso real FarSLIP v2 historico).
    far_df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            **far_cols,
        },
        schema=schema,
    )
    assert far_df.schema["parcel_id"] == pl.Int64
    far_path = tmp_path / "embeddings_int64.parquet"
    far_df.write_parquet(far_path)

    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_farslip=True,
        farslip_path=str(far_path),
        **injected,
    )
    # El join se ejecuta y materializa las 512 cols FarSLIP.
    assert "farslip_000" in df.columns
    assert "farslip_511" in df.columns
    # Y `parcel_id` quedo en el esquema canonico Utf8.
    assert df.schema["parcel_id"] == pl.Utf8


def test_fusion_result_parcel_id_is_utf8_after_join(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
) -> None:
    """Tras el join con todos los bloques, `parcel_id` queda Utf8 (no Int64)."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(parcels, year=2024, **injected)
    assert df.schema["parcel_id"] == pl.Utf8
    # Los valores son strings (no enteros casteados con .0 ni notacion cientifica).
    sample = df.get_column("parcel_id").to_list()
    assert all(isinstance(v, str) for v in sample)
    assert not any("e" in v.lower() for v in sample)


# ---------------------------------------------------------------------------
# US-022b-D: bloque opcional `pheno_text_*` (rama semantica fenologica).
# ---------------------------------------------------------------------------


def test_pheno_text_block_optional_inyectado(
    parcels_fixture_3regions: gpd.GeoDataFrame, synthetic_alphaearth_64d: pl.DataFrame
) -> None:
    """Con `phenology_text_frame` inyectado, las cols `pheno_text_*` se unen via LEFT JOIN."""
    from ml.features.fusion import (
        EXPECTED_COL_COUNT_WITH_PHENO_TEXT,
        PHENOLOGY_TEXT_EMBED_DIM,
    )

    parcels = parcels_fixture_3regions
    n = len(parcels)
    pheno_cols = {
        f"pheno_text_{i:03d}": [0.01 * (i + 1)] * n for i in range(PHENOLOGY_TEXT_EMBED_DIM)
    }
    schema = {"parcel_id": pl.Int64, "year": pl.Int16}
    schema.update({c: pl.Float32 for c in pheno_cols})
    pheno_df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            "year": [2024] * n,
            **pheno_cols,
        },
        schema=schema,
    )
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_phenology_text=True,
        phenology_text_frame=pheno_df,
        **injected,
    )
    assert df.width == 2 + EXPECTED_COL_COUNT_WITH_PHENO_TEXT
    assert "pheno_text_000" in df.columns
    assert "pheno_text_383" in df.columns


def test_pheno_text_default_path_missing_emits_warning_and_omits(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin parquet en el default path: warning + no se incluye (no falla).

    Cambiamos el cwd a tmp_path porque el repo real puede contener un
    ``data/features/phenology_text_pastis.parquet`` materializado por
    US-022-c P5 (216 parcelas) — esa presencia haria que el branch
    `default_path_not_found` no se ejecutara.
    """
    parcels = parcels_fixture_3regions
    monkeypatch.chdir(tmp_path)
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_phenology_text=True,
        phenology_text_path=None,
        **injected,
    )
    # No hay cols pheno_text_*; el width queda igual al base sin pheno_text.
    assert "pheno_text_000" not in df.columns
    assert df.width == 2 + EXPECTED_COL_COUNT_NO_FARSLIP


def test_pheno_text_explicit_path_missing_raises(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Path explicito inexistente => FileNotFoundError (no enmascarar config)."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    with pytest.raises(FileNotFoundError, match="pheno_text"):
        build_fused_features(
            parcels,
            year=2024,
            include_phenology_text=True,
            phenology_text_path=str(tmp_path / "does_not_exist.parquet"),
            **injected,
        )


# ---------------------------------------------------------------------------
# US-023-preview P2 — patch defensivo prefijo legacy `farslip_emb_`.
# ---------------------------------------------------------------------------


def test_farslip_block_accepts_legacy_emb_prefix(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Parquet con prefijo legacy ``farslip_emb_NNN`` se renombra in-memory.

    US-023-preview P2: el path canonico usa ``farslip_NNN``, pero los
    parquets v1/v2 originales (anteriores a la promocion) usan
    ``farslip_emb_NNN``. El patch defensivo en ``_build_farslip_block``
    detecta el prefijo legacy y renombra silenciosamente para no romper
    a callers que apunten a v1/v2 directamente.
    """
    parcels = parcels_fixture_3regions
    n = len(parcels)
    # Construye parquet con prefijo LEGACY (farslip_emb_NNN).
    legacy_cols = {f"farslip_emb_{i:03d}": [0.02 * (i + 1)] * n for i in range(512)}
    schema = {"parcel_id": pl.Int64}
    schema.update({c: pl.Float32 for c in legacy_cols})
    legacy_df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            **legacy_cols,
        },
        schema=schema,
    )
    legacy_path = tmp_path / "embeddings_italy_v2.parquet"
    legacy_df.write_parquet(legacy_path)

    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    # La build no debe levantar ValueError aunque el parquet use el prefijo legacy.
    df = build_fused_features(
        parcels,
        year=2024,
        include_farslip=True,
        farslip_path=str(legacy_path),
        **injected,
    )
    assert df.width == 2 + EXPECTED_COL_COUNT_WITH_FARSLIP
    # Las cols quedaron renombradas al patron canonico.
    assert "farslip_000" in df.columns
    assert "farslip_511" in df.columns
    assert "farslip_emb_000" not in df.columns


# ---------------------------------------------------------------------------
# US-023-preview P5 — bloque opcional `spectral_signature_*`.
# ---------------------------------------------------------------------------


def test_spectral_signature_block_optional_inyectado(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
) -> None:
    """Con `spectral_signature_frame` inyectado, las cols se unen via LEFT JOIN."""
    from ml.features.fusion import EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE

    parcels = parcels_fixture_3regions
    n = len(parcels)
    spec_cols = {f"spectral_signature_{i:03d}": [720.0 + i * 0.5] * n for i in range(3)}
    schema = {"parcel_id": pl.Int64, "year": pl.Int16}
    schema.update({c: pl.Float32 for c in spec_cols})
    spec_df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").tolist(),
            "year": [2024] * n,
            **spec_cols,
        },
        schema=schema,
    )
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_spectral_signature=True,
        spectral_signature_frame=spec_df,
        **injected,
    )
    assert df.width == 2 + EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE
    assert "spectral_signature_000" in df.columns
    assert "spectral_signature_002" in df.columns


def test_spectral_signature_default_path_missing_emits_warning_and_omits(
    parcels_fixture_3regions: gpd.GeoDataFrame,
    synthetic_alphaearth_64d: pl.DataFrame,
) -> None:
    """Sin parquet default + sin frame inyectado: warning + bloque omitido."""
    parcels = parcels_fixture_3regions
    injected = _build_default_injection(parcels, synthetic_ae=synthetic_alphaearth_64d)
    df = build_fused_features(
        parcels,
        year=2024,
        include_spectral_signature=True,
        spectral_signature_path=None,
        **injected,
    )
    assert "spectral_signature_000" not in df.columns
    assert df.width == 2 + EXPECTED_COL_COUNT_NO_FARSLIP
