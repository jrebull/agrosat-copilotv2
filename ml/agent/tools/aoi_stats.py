"""``get_aoi_stats`` tool: zonal crop statistics over an AOI for one year.

This synchronous demo tool aggregates the session's parcels that fall inside a
caller-supplied AOI polygon for a given campaign year and reports:

- ``area_ha``: the area of the AOI polygon itself, computed on the geography
  type (``ST_Area(geography)`` returns square metres on the spheroid) and
  converted to hectares. This is the AOI footprint, independent of how many
  parcels it contains.
- ``n_parcels``: number of the session's parcels (of that ``year``) whose
  geometry intersects the AOI.
- ``dominant_crop``: the most frequent ``crop_class`` among those parcels
  (statistical mode); empty string when no parcel carries a class label.
- ``crop_fractions``: per-class share of the intersecting parcels in ``[0, 1]``,
  summing to ~1 over the labelled parcels.

Every query filters by ``session_id`` and runs inside
:func:`ml.agent.db.session_scoped_conn`. No data is fabricated: an empty AOI
yields zero parcels, an empty ``crop_fractions`` map and an empty
``dominant_crop``.
"""

from __future__ import annotations

import json
import time

import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import AoiStats, AoiStatsInput

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# Square metres per hectare (1 ha = 10_000 m^2). ``ST_Area(geography)`` returns
# square metres on the WGS84 spheroid, so we divide by this to get hectares.
_SQM_PER_HA: float = 10_000.0

# Area of the AOI polygon on the spheroid (geography cast => metres), converted
# to hectares. Built from the GeoJSON the caller drew (EPSG:4326).
_AOI_AREA_SQL = """
SELECT ST_Area(
    ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)::geography
) AS area_sqm
"""

# Per-class parcel counts for the session's parcels of ``year`` that intersect
# the AOI. ``crop_class`` may be NULL (unlabelled parcels); it is kept so the
# total parcel count reflects every intersecting parcel.
_CROP_COUNTS_SQL = """
SELECT crop_class, COUNT(*) AS n
FROM parcels
WHERE session_id = $1
  AND year = $2
  AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326))
GROUP BY crop_class
"""


async def run(inp: AoiStatsInput, ctx: ToolContext) -> AoiStats:
    """Aggregate crop statistics over an AOI for a campaign year.

    Args:
        inp: Validated arguments (session id, AOI polygon, year).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        An :class:`AoiStats` with the AOI area in hectares, the intersecting
        parcel count, the dominant crop and the per-class area fractions.
    """
    started = time.perf_counter()
    logger.info(
        "tool_call_started",
        tool="get_aoi_stats",
        session_id=str(ctx.session_id),
        year=inp.year,
    )

    aoi_geojson = json.dumps(inp.aoi.model_dump())

    async with session_scoped_conn(inp.session_id) as conn:
        area_record = await conn.fetchrow(_AOI_AREA_SQL, aoi_geojson)
        count_records = await conn.fetch(
            _CROP_COUNTS_SQL,
            inp.session_id,
            inp.year,
            aoi_geojson,
        )

    area_sqm = float(area_record["area_sqm"]) if area_record is not None else 0.0
    area_ha = area_sqm / _SQM_PER_HA

    # Total parcels include unlabelled ones; fractions are over labelled parcels
    # only (a fraction over an unknown class is meaningless).
    n_parcels = sum(int(record["n"]) for record in count_records)
    labelled_counts: dict[str, int] = {
        record["crop_class"]: int(record["n"])
        for record in count_records
        if record["crop_class"] is not None
    }
    n_labelled = sum(labelled_counts.values())

    if n_labelled > 0:
        crop_fractions = {crop: count / n_labelled for crop, count in labelled_counts.items()}
        # Mode: highest count, ties broken by class name for determinism.
        dominant_crop = max(
            labelled_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
    else:
        crop_fractions = {}
        dominant_crop = ""

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="get_aoi_stats",
        session_id=str(ctx.session_id),
        n_parcels=n_parcels,
        dominant_crop=dominant_crop,
        area_ha=round(area_ha, 4),
        duration_ms=round(duration_ms, 2),
    )
    return AoiStats(
        area_ha=area_ha,
        dominant_crop=dominant_crop,
        crop_fractions=crop_fractions,
        n_parcels=n_parcels,
    )
