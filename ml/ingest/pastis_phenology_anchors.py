"""Build calendar phenology anchors per PASTIS-R parcel.

PASTIS-R does not expose calendar phenology DOY in its metadata: it only
provides `dates-S2` (Sentinel-2 acquisition timestamps per patch, first date
~17-sep-2018, last ~oct-2019). The US-016 subset derived
`sog_doy`/`peak_doy`/`senescence_doy` as **days since the patch's first S2
image**, not as calendar DOY (1-365 of the agronomic year).

This module converts the subset's relative DOY to calendar DOY of the
sampling year (default 2019), using the first real `dates-S2` date per
patch as reference. The output parquet is directly consumable by
:func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`
as ``phenology_anchors_path`` and removes the warning
``phenology_anchors_fallback_static``.

Canonical decisions (US-023-preview v2 fix S2 sampler):

- ``sog_doy`` that falls in 2018 (the year before sampling): wrap to the
  start of the sampling year with the agronomic fallback ``SOS_BRITTANY=90``
  (DOY 90 = 31-mar; Brittany literature places winter wheat
  emergence/stem-elongation between DOY 90-110). Keeping the SOS before
  DOY=1 of the sampling year invalidates the sampler window (`+/- window_days`).
- ``peak_doy`` and ``senescence_doy`` that fall in 2019: direct conversion
  ``base_date + relative_days -> calendar DOY``.
- Parcels without patch_id, without a PASTIS base date or with NULL relative
  DOY: static Brittany fallback ``(SOS=90, peak=180, senescence=220)``
  derived from literature (MDPI Brittany 2022).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["build_pastis_phenology_anchors"]


#: Brittany agronomic fallback when the conversion fails per parcel.
#: SOS=90 (winter wheat emergence / stem elongation, DOY 90-110 in literature).
#: peak=180 (wheat flowering-grain filling + LAI peak of summer crops).
#: senescence=220 (winter wheat harvest + early maize senescence).
FALLBACK_DOY_BRITTANY: dict[str, int] = {
    "sog_doy": 90,
    "peak_doy": 180,
    "senescence_doy": 220,
}

#: Lower bound of the acceptable calendar DOY. SOS before DOY=1 wraps
#: to the Brittany fallback to avoid producing a negative DOY.
MIN_VALID_DOY: int = 1

#: Upper bound of the acceptable calendar DOY.
MAX_VALID_DOY: int = 365


def build_pastis_phenology_anchors(
    *,
    metadata_geojson_path: Path | str = Path("data/PASTIS-R/metadata.geojson"),
    features_subset_path: Path | str = Path(
        "data/test_fixtures/feature_selection_parcels_subset.parquet"
    ),
    output_path: Path | str = Path("data/features/pastis_phenology_anchors_2019.parquet"),
    target_year: int = 2019,
    overwrite: bool = False,
) -> Path:
    """Generate ``parcel_id, sog_doy, peak_doy, senescence_doy`` (calendar DOY).

    Reads PASTIS-R's ``metadata.geojson``, extracts the first Sentinel-2
    date per patch as the base date, reads the US-016 subset with relative
    DOY per parcel, and persists a parquet with calendar DOY of the
    ``target_year``. Parcels whose conversion falls outside the range
    ``[1, 365]`` or whose base patch is not identifiable use the Brittany
    fallback documented in :data:`FALLBACK_DOY_BRITTANY`.

    Args:
        metadata_geojson_path: Path to PASTIS-R's ``metadata.geojson``.
        features_subset_path: Path to the US-016 subset with relative DOY.
        output_path: Target parquet with a schema directly consumable
            by :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`.
        target_year: Sampling year (default 2019; PASTIS-R range).
        overwrite: If True regenerates even if the parquet exists.

    Returns:
        Path of the generated parquet with schema
        ``parcel_id (Utf8), sog_doy (Int16), peak_doy (Int16), senescence_doy (Int16)``.

    Raises:
        FileNotFoundError: if the metadata.geojson or the subset do not exist.
        ValueError: if the subset does not expose ``patch_id`` or the relative DOY.
    """
    out = Path(output_path)
    if out.exists() and not overwrite:
        logger.info("pastis_phenology_anchors_cache_hit", path=str(out))
        return out

    meta_path = Path(metadata_geojson_path)
    sub_path = Path(features_subset_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.geojson not found at {meta_path}.")
    if not sub_path.exists():
        raise FileNotFoundError(f"features subset not found at {sub_path}.")

    # 1. Extract base date per patch from metadata.geojson.
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)

    patch_base_date: dict[int, datetime] = {}
    for feat in meta["features"]:
        props = feat["properties"]
        dates_s2 = props.get("dates-S2", {})
        if not dates_s2:
            continue
        # First chronological date (we sort by int key).
        first_key = min(dates_s2.keys(), key=int)
        first_yyyymmdd = str(dates_s2[first_key])
        try:
            patch_base_date[int(props["ID_PATCH"])] = datetime.strptime(first_yyyymmdd, "%Y%m%d")
        except (KeyError, ValueError):
            continue

    if not patch_base_date:
        raise ValueError(
            "metadata.geojson exposes neither `dates-S2` nor a parseable `ID_PATCH`. "
            "Check that the PASTIS-R dataset is complete."
        )

    logger.info(
        "pastis_dates_indexed",
        n_patches=len(patch_base_date),
        first_date_min=min(patch_base_date.values()).isoformat(),
    )

    # 2. Read US-016 subset with relative DOY + patch_id.
    df = pl.read_parquet(sub_path)
    required = {"parcel_id", "patch_id", "sog_doy", "peak_doy", "senescence_doy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subset {sub_path} is missing required columns: {sorted(missing)}.")

    # 3. Convert relative DOY -> calendar DOY per parcel.
    rows: list[dict[str, int | str]] = []
    n_fallback_unknown_patch = 0
    n_fallback_null_doy = 0
    n_fallback_out_of_range = 0
    n_real = 0

    for row in df.iter_rows(named=True):
        parcel_id = str(row["parcel_id"])
        patch_id = row["patch_id"]
        base = patch_base_date.get(int(patch_id)) if patch_id is not None else None

        if base is None:
            n_fallback_unknown_patch += 1
            rows.append(
                {
                    "parcel_id": parcel_id,
                    "sog_doy": FALLBACK_DOY_BRITTANY["sog_doy"],
                    "peak_doy": FALLBACK_DOY_BRITTANY["peak_doy"],
                    "senescence_doy": FALLBACK_DOY_BRITTANY["senescence_doy"],
                }
            )
            continue

        converted: dict[str, int] = {}
        any_fallback = False
        for anchor in ("sog_doy", "peak_doy", "senescence_doy"):
            rel = row.get(anchor)
            if rel is None:
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            real_date = base + timedelta(days=int(rel))
            if real_date.year != target_year:
                # Falls in 2018 (winter sowing) or 2020 (unexpected): fallback.
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            cal_doy = real_date.timetuple().tm_yday
            if cal_doy < MIN_VALID_DOY or cal_doy > MAX_VALID_DOY:
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            converted[anchor] = cal_doy

        if any_fallback:
            # If at least one anchor fell into fallback, we count it
            # but keep using what WAS real for the other 2.
            if all(
                converted[a] == FALLBACK_DOY_BRITTANY[a]
                for a in ("sog_doy", "peak_doy", "senescence_doy")
            ):
                n_fallback_out_of_range += 1
            else:
                n_real += 1
                n_fallback_null_doy += 1
        else:
            n_real += 1

        rows.append({"parcel_id": parcel_id, **converted})

    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "parcel_id": pl.Utf8,
            "sog_doy": pl.Int16,
            "peak_doy": pl.Int16,
            "senescence_doy": pl.Int16,
        },
    ).write_parquet(out)

    logger.info(
        "pastis_phenology_anchors_persisted",
        path=str(out),
        n_total=len(rows),
        n_real_at_least_one_ancla=n_real,
        n_fallback_unknown_patch=n_fallback_unknown_patch,
        n_fallback_out_of_range=n_fallback_out_of_range,
        n_fallback_null_doy_partial=n_fallback_null_doy,
        target_year=target_year,
    )
    return out
