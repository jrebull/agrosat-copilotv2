"""Tests US-023-preview-v2 P5 — ``ml.ingest.s2_anchor_sampler``.

Mockea Earth Engine NUNCA llama a GEE real. Cubre:

- Schema y orden estable (parcel_id Utf8, year Int16, 15 cols espectrales).
- Determinismo: misma entrada -> mismo output binario.
- Batching: ``batch_size < n_parcels`` no pierde filas.
- Cache hit: segunda llamada con misma entrada lee el parquet existente
  sin volver a llamar al mock de EE.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon

from ml.ingest import s2_anchor_sampler
from ml.ingest.s2_anchor_sampler import (
    DEFAULT_ANCHORS,
    DEFAULT_BANDS,
    _band_col_name,
    sample_s2_anchors_for_parcels,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_parcels(n: int = 4) -> gpd.GeoDataFrame:
    """Genera ``n`` parcelas cuadrado en Italia con DOY pre-calculados."""
    geoms = [
        Polygon(
            [
                (9.5 + i * 0.01, 45.2),
                (9.51 + i * 0.01, 45.2),
                (9.51 + i * 0.01, 45.21),
                (9.5 + i * 0.01, 45.21),
                (9.5 + i * 0.01, 45.2),
            ]
        )
        for i in range(n)
    ]
    return gpd.GeoDataFrame(
        {
            "parcel_id": [f"p{i:03d}" for i in range(n)],
            "year": [2023] * n,
            "sog_doy": [100 + i for i in range(n)],
            "peak_doy": [180] * n,
            "senescence_doy": [260] * n,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )


def _fake_ee_module(payload_per_doy: dict[int, list[dict[str, Any]]]) -> types.ModuleType:
    """Crea un modulo ``ee`` fake con ``getInfo`` retornando payloads por DOY.

    El sampler hace ``reduceRegions(mean).getInfo()`` por (anchor, doy unico).
    Aqui devolvemos siempre el mismo payload para mantener el mock simple;
    el contenido se parametriza fila a fila desde ``payload_per_doy``.
    """
    fake = types.ModuleType("ee")

    # Reusable mock chain: ImageCollection().filterDate().select().median().reduceRegions().getInfo()
    # Para mantener simple, devolvemos el mismo payload (la primera lista de
    # payload_per_doy) en cada llamada — el sampler invoca getInfo una vez
    # por DOY unico por ancla, asi que side_effect rota.
    all_payloads: list[dict[str, Any]] = []
    for doy in sorted(payload_per_doy.keys()):
        for _anchor in DEFAULT_ANCHORS:
            all_payloads.append({"features": payload_per_doy[doy]})

    reduced = MagicMock(name="ee.reduced")
    reduced.getInfo.side_effect = all_payloads * 10  # buffer abundante

    median = MagicMock(name="ee.median")
    median.reduceRegions.return_value = reduced

    collection = MagicMock(name="ee.collection")
    collection.filterDate.return_value = collection
    collection.select.return_value = collection
    collection.median.return_value = median

    fake.ImageCollection = MagicMock(return_value=collection)  # type: ignore[attr-defined]
    fake.Feature = MagicMock(side_effect=lambda geom, props: {"geom": geom, "props": props})  # type: ignore[attr-defined]
    fake.Geometry = MagicMock(side_effect=lambda x: x)  # type: ignore[attr-defined]
    fake.FeatureCollection = MagicMock(side_effect=lambda feats: {"feats": feats})  # type: ignore[attr-defined]
    fake.Reducer = MagicMock()  # type: ignore[attr-defined]
    fake.Reducer.mean = MagicMock(return_value="mean_reducer")  # type: ignore[attr-defined]
    return fake


def _payload_for_parcels(
    parcels: gpd.GeoDataFrame, bands: tuple[str, ...] = DEFAULT_BANDS
) -> dict[int, list[dict[str, Any]]]:
    """Construye payloads sinteticos: cada DOY tiene 1 feature por parcela."""
    by_doy: dict[int, list[dict[str, Any]]] = {}
    for _, row in parcels.iterrows():
        for doy in (int(row["sog_doy"]), int(row["peak_doy"]), int(row["senescence_doy"])):
            props = {"parcel_id": str(row["parcel_id"])}
            for j, band in enumerate(bands):
                # Valor sintetico estable y reproducible.
                props[band] = float(1000 + j * 100 + doy)
            by_doy.setdefault(doy, []).append({"properties": props})
    return by_doy


# ---------------------------------------------------------------------------
# Helpers de patching
# ---------------------------------------------------------------------------


def _patch_ee(monkeypatch: pytest.MonkeyPatch, fake_ee: types.ModuleType) -> None:
    """Inyecta el modulo ``ee`` fake y noop sobre ``init_ee``."""
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    # ``init_ee`` se importa lazy dentro de la funcion, asi que parchamos el
    # modulo gee_sampler para que init_ee sea un noop.
    from ml.ingest import gee_sampler

    monkeypatch.setattr(gee_sampler, "init_ee", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_and_15_spectral_cols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output tiene parcel_id Utf8 + year Int16 + 15 cols espectrales Float64."""
    parcels = _make_parcels(n=3)
    payload = _payload_for_parcels(parcels)
    fake_ee = _fake_ee_module(payload)
    _patch_ee(monkeypatch, fake_ee)

    out_path = tmp_path / "anchors.parquet"
    cache_dir = tmp_path / "cache"

    result_path = sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out_path,
        cache_dir=cache_dir,
        batch_size=10,
    )
    assert result_path.exists()
    df = pl.read_parquet(result_path)
    assert df.height == 3
    # 2 meta + 3 anchors * 5 bands = 17 cols totales.
    assert df.width == 17
    assert df.schema["parcel_id"] == pl.Utf8
    assert df.schema["year"] == pl.Int16
    expected_cols = {"parcel_id", "year"}
    for anchor in DEFAULT_ANCHORS:
        for band in DEFAULT_BANDS:
            expected_cols.add(_band_col_name(anchor, band))
    assert set(df.columns) == expected_cols
    # Cols espectrales son Float64.
    for anchor in DEFAULT_ANCHORS:
        for band in DEFAULT_BANDS:
            assert df.schema[_band_col_name(anchor, band)] == pl.Float64


def test_determinism_and_stable_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Misma entrada -> mismo output (orden estable por parcel_id)."""
    parcels = _make_parcels(n=4)
    payload = _payload_for_parcels(parcels)

    fake_ee = _fake_ee_module(payload)
    _patch_ee(monkeypatch, fake_ee)
    out1 = tmp_path / "run1.parquet"
    sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out1,
        cache_dir=tmp_path / "cache1",
        batch_size=10,
    )
    df1 = pl.read_parquet(out1)

    # Segunda corrida con cache distinto -> mock fresco.
    fake_ee2 = _fake_ee_module(payload)
    _patch_ee(monkeypatch, fake_ee2)
    out2 = tmp_path / "run2.parquet"
    sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out2,
        cache_dir=tmp_path / "cache2",
        batch_size=10,
    )
    df2 = pl.read_parquet(out2)

    assert df1.equals(df2)
    # Orden estable ascendente por parcel_id.
    assert df1["parcel_id"].to_list() == sorted(df1["parcel_id"].to_list())


def test_batching_respects_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``batch_size < n_parcels`` no pierde filas."""
    parcels = _make_parcels(n=8)
    payload = _payload_for_parcels(parcels)
    fake_ee = _fake_ee_module(payload)
    _patch_ee(monkeypatch, fake_ee)

    out_path = tmp_path / "anchors_batched.parquet"
    sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out_path,
        cache_dir=tmp_path / "cache",
        batch_size=3,  # 8/3 = 3 batches (3+3+2)
    )
    df = pl.read_parquet(out_path)
    assert df.height == 8
    assert set(df["parcel_id"].to_list()) == {f"p{i:03d}" for i in range(8)}


def test_cache_hit_skips_ee_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Segunda llamada con misma entrada usa cache y no llama EE."""
    parcels = _make_parcels(n=2)
    payload = _payload_for_parcels(parcels)
    fake_ee = _fake_ee_module(payload)
    _patch_ee(monkeypatch, fake_ee)

    cache_dir = tmp_path / "cache_shared"
    out_path = tmp_path / "first.parquet"
    sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out_path,
        cache_dir=cache_dir,
        batch_size=10,
    )
    n_calls_first = fake_ee.ImageCollection.call_count  # type: ignore[attr-defined]
    assert n_calls_first > 0

    # Segunda llamada -> cache hit, NO debe llamar ImageCollection mas veces.
    out_path2 = tmp_path / "second.parquet"
    sample_s2_anchors_for_parcels(
        parcels,
        year=2023,
        output_path=out_path2,
        cache_dir=cache_dir,
        batch_size=10,
    )
    n_calls_second = fake_ee.ImageCollection.call_count  # type: ignore[attr-defined]
    assert n_calls_second == n_calls_first  # no nuevas llamadas
    assert out_path2.exists()
    # Y los contenidos coinciden.
    assert pl.read_parquet(out_path).equals(pl.read_parquet(out_path2))


def test_md5_cache_key_stable() -> None:
    """``_parcels_md5`` es deterministico sobre la misma entrada."""
    parcels = _make_parcels(n=5)
    h1 = s2_anchor_sampler._parcels_md5(parcels)
    h2 = s2_anchor_sampler._parcels_md5(parcels)
    assert h1 == h2
    assert len(h1) == 10
