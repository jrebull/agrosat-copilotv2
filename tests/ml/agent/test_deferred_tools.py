"""Deferred-tool tests (US-045 AC-5, AC-7).

Covers ``search_stac``, ``get_tiles`` and ``add_aoi`` with the external backends
(pgstac, TiTiler, PostGIS) mocked -- no network, no live database.

- ``search_stac``: graceful empty result when ``pgstac.search`` is absent
  (the ``UndefinedFunctionError`` class is raised by the fake), and the happy
  path returning the real ``features`` of a FeatureCollection.
- ``get_tiles``: pure URL assembly (no HTTP) with the ``{z}/{x}/{y}`` template
  preserved and the right index expression/colormap.
- ``add_aoi``: INSERT returning the generated id, surfaced as an ``AoiRef``.
"""

from __future__ import annotations

import json

import asyncpg

import ml.agent.tools.add_aoi as add_aoi_mod
import ml.agent.tools.stac as stac_mod
from ml.agent.schemas import (
    AddAoiInput,
    AoiRef,
    BBox,
    GetTilesInput,
    SceneList,
    SearchStacInput,
    TileUrl,
)
from ml.agent.tools.tiles import run as tiles_run
from ml.ingest.cdse_client import CDSEScene

from .conftest import SESSION_A, FakeConn, FakeRecord, fake_session_scoped_conn

_BBOX = BBox(minx=-3.7, miny=40.0, maxx=-3.6, maxy=40.1)
_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


class _RaisingConn(FakeConn):
    """FakeConn whose ``fetchval`` raises a given exception (pgstac absent)."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def fetchval(self, sql: str, *args):
        self.calls.append((sql, args))
        raise self._exc


# ---------------------------------------------------------------------------
# search_stac
# ---------------------------------------------------------------------------
async def test_search_stac_empty_when_pgstac_absent(monkeypatch, make_ctx) -> None:
    """A missing ``pgstac.search`` degrades to an empty ``SceneList`` (no crash)."""
    conn = _RaisingConn(asyncpg.UndefinedFunctionError("function pgstac.search does not exist"))
    monkeypatch.setattr(stac_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await stac_mod.run(
        SearchStacInput(bbox=_BBOX, datetime_range="2019-01-01/2019-12-31"), make_ctx()
    )

    assert isinstance(out, SceneList)
    assert out.scenes == []
    assert out.count == 0


async def test_search_stac_happy_path_returns_real_features(monkeypatch, make_ctx) -> None:
    """A FeatureCollection from pgstac is unpacked into the scene list."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"id": "S2_2019_06_01", "properties": {"eo:cloud_cover": 5}},
            {"id": "S2_2019_06_11", "properties": {"eo:cloud_cover": 12}},
        ],
    }
    # asyncpg may surface the JSON as a string; exercise that codepath.
    conn = FakeConn(fetchval_value=json.dumps(collection))
    monkeypatch.setattr(stac_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await stac_mod.run(
        SearchStacInput(bbox=_BBOX, datetime_range="2019-01-01/2019-12-31", cloud_cover_max=20.0),
        make_ctx(),
    )

    assert out.count == 2
    assert out.scenes[0]["id"] == "S2_2019_06_01"
    # The request bound to pgstac.search must carry the bbox and cloud filter.
    request_arg = conn.calls[-1][1][0]
    request = json.loads(request_arg)
    assert request["bbox"] == [-3.7, 40.0, -3.6, 40.1]
    assert request["filter"]["args"][1] == 20.0


class _CdseSettings:
    """Settings stub with CDSE credentials populated (no live call is made)."""

    cdse_client_id = "fake-client-id"
    cdse_client_secret = "fake-client-secret"
    cdse_token_url = "https://identity.example/token"


def _ctx_with_cdse(make_ctx):
    """Build a ToolContext whose settings carry (fake) CDSE credentials."""
    ctx = make_ctx()
    ctx.settings = _CdseSettings()  # type: ignore[assignment]
    return ctx


async def test_search_stac_uses_cdse_when_credentials_present(monkeypatch, make_ctx) -> None:
    """With CDSE configured, real CDSE scenes are mapped to STAC items."""
    captured: dict = {}

    def _fake_search_s2(self, bbox, datetime_range, *, cloud_cover_max=10.0, **kwargs):
        captured["bbox"] = bbox
        captured["datetime_range"] = datetime_range
        captured["cloud_cover_max"] = cloud_cover_max
        return [
            CDSEScene(
                scene_id="S2B_MSIL2A_20210601T101559_T32TPP",
                datetime="2021-06-01T10:15:59.000Z",
                cloud_cover=3.2,
                bbox=(10.0, 43.0, 10.5, 43.5),
            ),
            CDSEScene(
                scene_id="S2A_MSIL2A_20210527T101031_T32TPP",
                datetime="2021-05-27T10:10:31.000Z",
                cloud_cover=7.8,
                bbox=(10.0, 43.0, 10.5, 43.5),
            ),
        ]

    # Mock the network boundary only: search_s2 never hits CDSE.
    monkeypatch.setattr("ml.ingest.cdse_client.CDSEClient.search_s2", _fake_search_s2)

    out = await stac_mod.run(
        SearchStacInput(bbox=_BBOX, datetime_range="2021-05-01/2021-09-30", cloud_cover_max=10.0),
        _ctx_with_cdse(make_ctx),
    )

    assert isinstance(out, SceneList)
    assert out.count == 2
    # CDSEScene -> STAC item mapping preserves the citable fields.
    first = out.scenes[0]
    assert first["id"] == "S2B_MSIL2A_20210601T101559_T32TPP"
    assert first["properties"]["datetime"] == "2021-06-01T10:15:59.000Z"
    assert first["properties"]["eo:cloud_cover"] == 3.2
    assert first["bbox"] == [10.0, 43.0, 10.5, 43.5]
    assert first["source"] == "cdse"
    # The tool forwarded the input bbox / window / cloud bound to CDSE.
    assert captured["bbox"] == (-3.7, 40.0, -3.6, 40.1)
    assert captured["datetime_range"] == "2021-05-01/2021-09-30"
    assert captured["cloud_cover_max"] == 10.0


async def test_search_stac_cdse_empty_catalogue(monkeypatch, make_ctx) -> None:
    """An empty CDSE catalogue yields an empty SceneList (no fabrication)."""
    monkeypatch.setattr(
        "ml.ingest.cdse_client.CDSEClient.search_s2",
        lambda self, bbox, datetime_range, **kwargs: [],
    )

    out = await stac_mod.run(
        SearchStacInput(bbox=_BBOX, datetime_range="2021-05-01/2021-09-30"),
        _ctx_with_cdse(make_ctx),
    )

    assert out.scenes == []
    assert out.count == 0


async def test_search_stac_degrades_to_pgstac_without_credentials(monkeypatch, make_ctx) -> None:
    """Empty CDSE credentials degrade cleanly to pgstac (no crash, no CDSE call)."""

    # CDSE must never be touched when credentials are absent.
    def _boom(self, *args, **kwargs):
        raise AssertionError("CDSE must not be called without credentials")

    monkeypatch.setattr("ml.ingest.cdse_client.CDSEClient.search_s2", _boom)
    # pgstac returns a FeatureCollection (the preserved fallback path).
    collection = {"type": "FeatureCollection", "features": [{"id": "PGSTAC_1"}]}
    conn = FakeConn(fetchval_value=json.dumps(collection))
    monkeypatch.setattr(stac_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    # The default make_ctx settings stub has empty CDSE credentials.
    out = await stac_mod.run(
        SearchStacInput(bbox=_BBOX, datetime_range="2019-01-01/2019-12-31"), make_ctx()
    )

    assert isinstance(out, SceneList)
    assert out.count == 1
    assert out.scenes[0]["id"] == "PGSTAC_1"


# ---------------------------------------------------------------------------
# get_tiles
# ---------------------------------------------------------------------------
async def test_get_tiles_ndvi_url(make_ctx) -> None:
    """NDVI tiles build a TiTiler URL with the index expression and template."""
    out = await tiles_run(GetTilesInput(scene_id="S2_X", index="ndvi"), make_ctx())

    assert isinstance(out, TileUrl)
    assert out.scene_id == "S2_X"
    assert out.index == "ndvi"
    # XYZ template placeholders preserved (MapLibre fills them per tile).
    assert "/stac/tiles/{z}/{x}/{y}" in out.tile_url
    assert out.tile_url.startswith("http://localhost:8001/")
    assert "expression=" in out.tile_url
    assert "colormap_name=" in out.tile_url
    assert "url=S2_X" in out.tile_url


async def test_get_tiles_rgb_uses_band_assets(make_ctx) -> None:
    """RGB tiles select natural-colour bands instead of an index expression."""
    out = await tiles_run(GetTilesInput(scene_id="S2_Y", index="rgb"), make_ctx())

    assert out.index == "rgb"
    assert "assets=B04" in out.tile_url
    assert "assets=B03" in out.tile_url
    assert "assets=B02" in out.tile_url
    assert "expression=" not in out.tile_url
    assert "{z}/{x}/{y}" in out.tile_url


# ---------------------------------------------------------------------------
# add_aoi
# ---------------------------------------------------------------------------
async def test_add_aoi_inserts_and_returns_ref(monkeypatch, make_ctx) -> None:
    """``add_aoi`` inserts the polygon and returns the generated ``AoiRef``."""
    conn = FakeConn(fetchrow_row=FakeRecord(id=42, area_ha=12.5))
    monkeypatch.setattr(add_aoi_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await add_aoi_mod.run(
        AddAoiInput(session_id=SESSION_A, aoi=_POLYGON, name="Demo Field"), make_ctx()
    )

    assert isinstance(out, AoiRef)
    assert out.aoi_id == 42
    assert out.label == "Demo Field"
    assert out.area_ha == 12.5

    # The INSERT bound the session id, the GeoJSON and the name (defence in depth).
    insert_call = next(c for c in conn.calls if "INSERT INTO aois" in c[0])
    bound = insert_call[1]
    assert bound[0] == SESSION_A
    assert json.loads(bound[1])["type"] == "Polygon"
    assert bound[2] == "Demo Field"
    # The RLS hook was primed for the session first.
    assert conn.set_config_calls()[0][1] == (str(SESSION_A),)


async def test_add_aoi_handles_null_area(monkeypatch, make_ctx) -> None:
    """A NULL server-computed area surfaces as ``area_ha=None`` (no crash)."""
    conn = FakeConn(fetchrow_row=FakeRecord(id=7, area_ha=None))
    monkeypatch.setattr(add_aoi_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await add_aoi_mod.run(
        AddAoiInput(session_id=SESSION_A, aoi=_POLYGON, name="Tiny"), make_ctx()
    )

    assert out.aoi_id == 7
    assert out.area_ha is None
