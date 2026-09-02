"""Synchronous DB-tool tests with a mocked asyncpg connection (US-045 AC-4).

Covers ``list_parcels``, ``get_parcel_timeseries`` and ``get_aoi_stats``. Each
tool's ``session_scoped_conn`` symbol is replaced by a fake that yields a
:class:`FakeConn` returning scripted rows, so no database is needed. The tests
assert the typed output shape, that the session id reaches the RLS hook, and the
honest empty-result behaviour (no fabricated data) when the DB has no rows.
"""

from __future__ import annotations

import json
from datetime import date

import ml.agent.tools.aoi_stats as aoi_stats_mod
import ml.agent.tools.parcels as parcels_mod
import ml.agent.tools.timeseries as timeseries_mod
from ml.agent.schemas import (
    AoiStats,
    AoiStatsInput,
    ListParcelsInput,
    ParcelList,
    ParcelTimeseriesInput,
    TimeSeries,
)

from .conftest import SESSION_A, FakeConn, FakeRecord, fake_session_scoped_conn

_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


# ---------------------------------------------------------------------------
# list_parcels
# ---------------------------------------------------------------------------
async def test_list_parcels_returns_typed_rows(monkeypatch, make_ctx) -> None:
    """``list_parcels`` maps DB rows to a typed ``ParcelList``."""
    conn = FakeConn(
        fetch_rows=[
            FakeRecord(id=1, crop_class="wheat", confidence=0.91),
            FakeRecord(id=2, crop_class="maize", confidence=0.77),
        ]
    )
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(ListParcelsInput(session_id=SESSION_A), make_ctx())

    assert isinstance(out, ParcelList)
    assert out.count == 2
    assert [p.parcel_id for p in out.parcels] == [1, 2]
    assert out.parcels[0].crop_class == "wheat"
    # RLS hook primed with the session id.
    assert conn.set_config_calls()[0][1] == (str(SESSION_A),)


async def test_list_parcels_empty_session(monkeypatch, make_ctx) -> None:
    """An empty session yields an empty list (count 0), nothing fabricated."""
    conn = FakeConn(fetch_rows=[])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(ListParcelsInput(session_id=SESSION_A), make_ctx())

    assert out.count == 0
    assert out.parcels == []


async def test_list_parcels_with_aoi_passes_geojson(monkeypatch, make_ctx) -> None:
    """With an AOI, the GeoJSON is serialised and bound to the spatial query."""
    conn = FakeConn(fetch_rows=[FakeRecord(id=5, crop_class=None, confidence=None)])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(ListParcelsInput(session_id=SESSION_A, aoi=_POLYGON), make_ctx())

    assert out.count == 1
    assert out.parcels[0].crop_class is None
    # The AOI-restricted SQL must have used ST_Intersects and bound the GeoJSON.
    aoi_call = next(c for c in conn.calls if "ST_Intersects" in c[0])
    geojson_arg = aoi_call[1][1]
    assert json.loads(geojson_arg)["type"] == "Polygon"


# ---------------------------------------------------------------------------
# get_parcel_timeseries
#
# Contract (honest-by-construction, see ``ml/agent/tools/timeseries.py``): the
# series is NOT a daily curve and NOT a distributional summary spread over made-up
# dates. It surfaces ONLY real phenology anchors -- points whose date is a genuine
# stored day-of-year and whose value is the measured value at that date. The sole
# such anchor the DB holds is the NDVI peak (``peak_value`` on ``peak_doy``). SOG
# and senescence store a day-of-year but no value, so they are never emitted; NDWI
# and EVI carry no temporal anchor and therefore yield an empty series.
#
# Regression guards below:
#   B-3 (honesty): no fabricated/evenly-spaced percentile dates anymore.
#   B-8 (short windows): no equispacing => no silent date collapse.
# ---------------------------------------------------------------------------
def _ndvi_stats_json() -> str:
    """Build a realistic ``ndvi_stats`` JSONB string (asyncpg surfaces str)."""
    return json.dumps(
        {
            "NDVI_p05": 0.12,
            "NDVI_p25": 0.31,
            "NDVI_p50": 0.55,
            "NDVI_p75": 0.74,
            "NDVI_p95": 0.88,
            "EVI_p50": 0.40,
        }
    )


async def test_timeseries_ndvi_returns_only_measured_peak_anchor(monkeypatch, make_ctx) -> None:
    """NDVI series is exactly the measured peak anchor (real date + real value).

    New honest contract (replaces the old "percentiles spread over evenly spaced
    dates" behaviour): the only emitted point is ``peak_value`` measured on the
    real ``peak_doy`` date. The stored percentiles are NOT laid out on fabricated
    dates -- that was the B-3 honesty bug.
    """
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=_ndvi_stats_json(),
            sog_doy=90,
            peak_doy=180,  # 2019-06-29, inside the window
            peak_value=0.93,
            senescence_doy=270,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert isinstance(out, TimeSeries)
    assert out.parcel_id == 7
    assert out.index == "ndvi"
    # Exactly one aligned point: the measured peak on its real day-of-year.
    assert out.dates == [date(2019, 6, 29)]  # DOY 180 of 2019
    assert out.values == [0.93]
    # None of the percentile values leak in on fabricated dates (B-3 regression).
    for fabricated in (0.12, 0.31, 0.55, 0.74, 0.88):
        assert fabricated not in out.values


async def test_timeseries_ndvi_empty_when_peak_outside_window(monkeypatch, make_ctx) -> None:
    """If the peak day-of-year falls outside the window, no point is emitted.

    The series carries only real, in-window anchors; with the single anchor out of
    range the honest answer is empty (no value placed on a made-up in-window date).
    """
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=_ndvi_stats_json(),
            sog_doy=90,
            peak_doy=180,  # 2019-06-29, OUTSIDE the requested July window
            peak_value=0.93,
            senescence_doy=270,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 7, 1),
            end=date(2019, 7, 31),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


async def test_timeseries_short_window_does_not_collapse(monkeypatch, make_ctx) -> None:
    """B-8 regression: a degenerate ``start == end`` window no longer collapses.

    The old ``_spread_dates`` produced duplicate dates on short windows and the
    dedup silently dropped percentiles. Now the single real anchor is emitted iff
    it falls on that exact day; here the peak is on it, so we get exactly one point
    with no silent data loss.
    """
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=_ndvi_stats_json(),
            sog_doy=90,
            peak_doy=180,  # 2019-06-29
            peak_value=0.93,
            senescence_doy=270,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 6, 29),  # single-day window exactly on the peak
            end=date(2019, 6, 29),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert out.dates == [date(2019, 6, 29)]
    assert out.values == [0.93]


async def test_timeseries_ndwi_empty_no_temporal_anchor(monkeypatch, make_ctx) -> None:
    """NDWI has no phenology anchor in the DB => empty series even with stats.

    Only NDVI carries a measured temporal anchor (the peak). Other indices have
    percentile stats but no real date to attach them to, so the honest answer is
    empty rather than percentiles on fabricated dates.
    """
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=json.dumps({"NDWI_p50": 0.22, "NDWI_p95": 0.41}),
            sog_doy=90,
            peak_doy=180,
            peak_value=0.93,  # NDVI peak; irrelevant to NDWI
            senescence_doy=270,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndwi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


async def test_timeseries_empty_when_no_feature_row(monkeypatch, make_ctx) -> None:
    """No feature row (parcel not visible / no data) => empty series."""
    conn = FakeConn(fetchrow_row=None)
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


async def test_timeseries_empty_when_index_missing(monkeypatch, make_ctx) -> None:
    """A feature row without the requested index's stats => empty series."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=json.dumps({"NDVI_p50": 0.5}),  # no NDWI keys at all
            sog_doy=None,
            peak_doy=None,
            peak_value=None,
            senescence_doy=None,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndwi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


# ---------------------------------------------------------------------------
# get_aoi_stats
# ---------------------------------------------------------------------------
async def test_aoi_stats_aggregates_dominant_and_fractions(monkeypatch, make_ctx) -> None:
    """``get_aoi_stats`` reports area, dominant crop and per-class fractions."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(area_sqm=120_000.0),  # 12 ha
        fetch_rows=[
            FakeRecord(crop_class="wheat", n=3),
            FakeRecord(crop_class="maize", n=1),
            FakeRecord(crop_class=None, n=2),  # unlabelled parcels still counted
        ],
    )
    monkeypatch.setattr(aoi_stats_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await aoi_stats_mod.run(
        AoiStatsInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert isinstance(out, AoiStats)
    assert out.area_ha == 12.0
    assert out.n_parcels == 6  # includes the two unlabelled
    assert out.dominant_crop == "wheat"
    # Fractions are over labelled parcels only (3 + 1 = 4).
    assert out.crop_fractions["wheat"] == 0.75
    assert out.crop_fractions["maize"] == 0.25
    assert abs(sum(out.crop_fractions.values()) - 1.0) < 1e-9


async def test_aoi_stats_empty_aoi(monkeypatch, make_ctx) -> None:
    """An AOI with no intersecting parcels yields empty crop stats."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(area_sqm=50_000.0),  # 5 ha footprint
        fetch_rows=[],
    )
    monkeypatch.setattr(aoi_stats_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await aoi_stats_mod.run(
        AoiStatsInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert out.area_ha == 5.0
    assert out.n_parcels == 0
    assert out.dominant_crop == ""
    assert out.crop_fractions == {}
