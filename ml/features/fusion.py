"""Multisensor fusion at parcel level (US-016).

This module builds the fused tabular vector that feeds the RF/XGBoost
baselines (US-019/020/021) and the tabular heads of the segmentation
architectures (EPIC 5). The vector concatenates 6 heterogeneous blocks aligned
by ``(parcel_id, year)`` with a deterministic layout, plus an optional seventh
block (FarSLIP, 512-dim) that is incorporated via ``LEFT JOIN`` when the
embeddings are delivered to ``data/farslip/embeddings_pastis.parquet``.

Column layout (stable order, downstream depends on it):

::

    parcel_id (Utf8) | year (i16) |
    ae_00 .. ae_63 (64)                                | AlphaEarth block
    {idx}_{stat} (17 * 5 = 85)                         | indices x stats block
    s1_vv_{stat} | s1_vh_{stat} (2 * 5 = 10)           | Sentinel-1 block
    srtm_elev_mean | srtm_slope_mean | srtm_aspect_dominant (3) |
    era5_tmean_m01..m12 | era5_prec_m01..m12 (24)       |
    geom_area_ha | geom_perimeter_m | geom_elongation (3) |
    [farslip_000 .. farslip_511 (512)]                  | optional

Technical decisions (see ``docs/us-planning/us-016.md`` §2):

- Polars 1.x ``LazyFrame`` with final ``collect(engine="streaming")``.
- Temporal stats subset ``("mean", "std", "p25", "p50", "p95")`` (5 stats,
  not the full 9 stats of US-015) to keep 85 cols per indices block and
  favor downstream economy.
- The FarSLIP block is optional via ``LEFT JOIN``. If ``include_farslip=True``
  and ``farslip_path`` does not exist, it emits a structured warning and omits
  the block without failing the build.
- The geometry columns (``geom_area_ha``, ``geom_perimeter_m``,
  ``geom_elongation``) are computed with ``GeoSeries.to_crs("EPSG:3857")`` to
  report real metric units (the Polsby-Popper elongation computation is
  dimensionless and, by construction, >= 1).
- ``srtm_aspect_dominant`` is returned as a cardinal string of 8 quadrants
  ``{N, NE, E, SE, S, SW, W, NW}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import polars as pl
import structlog

from ml.features.spectral_indices import INDEX_NAMES
from ml.utils.dataset_paths import resolve_dataset_path
from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public constants (downstream imports them to validate shape).
# ---------------------------------------------------------------------------

#: Subset of temporal statistics applied to the indices block and to the
#: Sentinel-1 block. Differs from the 9-stat set of US-015 for economy.
FUSION_STATS: Final[tuple[str, ...]] = ("mean", "std", "p25", "p50", "p95")

#: Canonical names of the 9 blocks of the fused vector.
#: The ``phenology_text`` block (US-022b-D) and ``spectral_signature``
#: (US-023-preview P5) are optional and are incorporated via ``LEFT JOIN``
#: when ``include_phenology_text=True`` / ``include_spectral_signature=True``
#: and the corresponding parquets exist.
BLOCK_NAMES: Final[tuple[str, ...]] = (
    "alphaearth",
    "indices_stats",
    "sentinel1",
    "srtm",
    "era5_monthly",
    "geometry",
    "farslip",
    "phenology_text",
    "spectral_signature",
)

#: Expected column count WITHOUT the FarSLIP block (excludes `parcel_id`, `year`).
#: 64 (AE) + 85 (idx*stats) + 10 (S1) + 3 (SRTM) + 24 (ERA5) + 3 (geom) = 189.
EXPECTED_COL_COUNT_NO_FARSLIP: Final[int] = 189

#: Expected column count WITH the FarSLIP block (excludes `parcel_id`, `year`).
#: 189 + 512 (FarSLIP) = 701.
EXPECTED_COL_COUNT_WITH_FARSLIP: Final[int] = EXPECTED_COL_COUNT_NO_FARSLIP + 512

#: Dimensions of the optional ``pheno_text_*`` block (US-022b-D). Matches
#: the default text-encoder ``sentence-transformers/all-MiniLM-L6-v2``
#: (see :data:`ml.features.phenology_description.DEFAULT_TEXT_EMBED_DIM`).
PHENOLOGY_TEXT_EMBED_DIM: Final[int] = 384

#: Expected column count WITH the pheno_text block (without FarSLIP).
#: 189 + 384 (pheno_text) = 573.
EXPECTED_COL_COUNT_WITH_PHENO_TEXT: Final[int] = (
    EXPECTED_COL_COUNT_NO_FARSLIP + PHENOLOGY_TEXT_EMBED_DIM
)

#: Expected column count with BOTH optional blocks.
EXPECTED_COL_COUNT_WITH_FARSLIP_AND_PHENO_TEXT: Final[int] = (
    EXPECTED_COL_COUNT_WITH_FARSLIP + PHENOLOGY_TEXT_EMBED_DIM
)

#: Number of features of the optional ``spectral_signature_*`` block (US-023-preview
#: P5). For the default ``rep`` descriptor with 3 phenological anchors you get
#: 3 columns; if the descriptor or the anchors change, this count varies.
#: The constant is exposed so that downstream consumers assuming the default
#: descriptor can pre-validate the expected shape.
DEFAULT_SPECTRAL_SIGNATURE_DIM: Final[int] = 3

#: Expected column count with the spectral_signature block (without FarSLIP or pheno_text).
EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE: Final[int] = (
    EXPECTED_COL_COUNT_NO_FARSLIP + DEFAULT_SPECTRAL_SIGNATURE_DIM
)

#: Canonical Sentinel-1 polarizations of the block (fixed order).
_S1_POLARIZATIONS: Final[tuple[str, ...]] = ("vv", "vh")

#: AlphaEarth column names (``ae_00 .. ae_63``).
AE_COLS: Final[tuple[str, ...]] = tuple(f"ae_{i:02d}" for i in range(64))

#: Default path for the FarSLIP embeddings (US-016b). Canonical naming
#: ``_pastis`` (the content is PASTIS-R, not Italian); resolved via
#: :func:`resolve_dataset_path` which falls back to the legacy ``_italy`` if applicable.
_DEFAULT_FARSLIP_PATH: Final[Path] = Path("data/farslip/embeddings_pastis.parquet")

#: Default path for the materialized pheno_text block (US-022b-D).
_DEFAULT_PHENO_TEXT_PATH: Final[Path] = Path("data/features/phenology_text_pastis.parquet")

#: Default path for the materialized spectral_signature block (US-023-preview P5).
_DEFAULT_SPECTRAL_SIGNATURE_PATH: Final[Path] = Path(
    "data/features/spectral_signature_pastis.parquet"
)


__all__ = [
    "AE_COLS",
    "BLOCK_NAMES",
    "DEFAULT_SPECTRAL_SIGNATURE_DIM",
    "EXPECTED_COL_COUNT_NO_FARSLIP",
    "EXPECTED_COL_COUNT_WITH_FARSLIP",
    "EXPECTED_COL_COUNT_WITH_FARSLIP_AND_PHENO_TEXT",
    "EXPECTED_COL_COUNT_WITH_PHENO_TEXT",
    "EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE",
    "FUSION_STATS",
    "PHENOLOGY_TEXT_EMBED_DIM",
    "build_fused_features",
]


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def build_fused_features(
    parcels: gpd.GeoDataFrame,
    year: int,
    *,
    blocks: tuple[str, ...] = BLOCK_NAMES,
    include_farslip: bool = False,
    farslip_path: str | Path | None = None,
    include_phenology_text: bool = False,
    phenology_text_path: str | Path | None = None,
    include_spectral_signature: bool = False,
    spectral_signature_path: str | Path | None = None,
    stats: tuple[str, ...] = FUSION_STATS,
    lazy: bool = True,
    ae_frame: pl.DataFrame | None = None,
    indices_frame: pl.DataFrame | None = None,
    s1_frame: pl.DataFrame | None = None,
    srtm_frame: pl.DataFrame | None = None,
    era5_frame: pl.DataFrame | None = None,
    phenology_text_frame: pl.DataFrame | None = None,
    spectral_signature_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the fused feature vector by ``(parcel_id, year)``.

    Args:
        parcels: GeoDataFrame with columns ``parcel_id`` (int), ``year`` (int)
            and ``geometry`` (POLYGON in EPSG:4326). Optionally ``crop_class``
            and ``region``. The year must match ``year``.
        year: Reference year for AlphaEarth / S1 / ERA5.
        blocks: Subset of blocks to compute. Enables ablation.
            Default ``BLOCK_NAMES`` (the 7 blocks, FarSLIP only if
            ``include_farslip=True``).
        include_farslip: If ``True`` attempts to join the FarSLIP block. If
            ``farslip_path`` does not exist, emits a warning and omits the
            block without failing the build.
        farslip_path: Path to the parquet with FarSLIP embeddings. Default
            ``data/farslip/embeddings_pastis.parquet`` (resolved via
            :func:`resolve_dataset_path`, falls back to the legacy ``_italy``
            if applicable).
        include_phenology_text: If ``True`` attempts to join the
            ``pheno_text_*`` block (US-022b-D, Wen et al. 2025). Same pattern
            as FarSLIP: if the path does not exist it is omitted without
            failing. If ``phenology_text_frame`` is passed explicitly the
            path is ignored.
        phenology_text_path: Path to the parquet with the text embeddings of
            the semantic branch (output of
            :func:`ml.features.phenology_description.build_phenology_text_block`).
            Default ``data/features/phenology_text_pastis.parquet`` (resolved
            via :func:`resolve_dataset_path`, falls back to the legacy
            ``_italy`` if applicable).
        stats: Temporal stats applied to indices and S1. Default
            :data:`FUSION_STATS`. Changing this parameter breaks the 85-column
            contract of the indices block — use only in ablation.
        lazy: If ``True`` (default) the joins are done in ``LazyFrame`` with
            final ``collect(engine="streaming")``. If ``False`` eager is used
            (useful for debugging).
        ae_frame: Optional injection of the already-sampled AlphaEarth block
            (testing and dependency injection from CLI script / Dagster).
        indices_frame: Optional injection of the indices*stats block.
        s1_frame: Optional injection of the Sentinel-1 block.
        srtm_frame: Optional injection of the SRTM block.
        era5_frame: Optional injection of the monthly ERA5 block.
        phenology_text_frame: Optional injection of the ``pheno_text_*`` block
            (testing); when passed, ``phenology_text_path`` is ignored.
        include_spectral_signature: If ``True`` attempts to join the
            ``spectral_signature_*`` block (US-023-preview P5). Same pattern as
            FarSLIP: if the path does not exist it is omitted without failing.
            If ``spectral_signature_frame`` is passed explicitly the path is
            ignored.
        spectral_signature_path: Path to the parquet with the spectral
            signature (output of
            :class:`ml.features.spectral_signature.SpectralSignatureFeatures`).
            Default ``data/features/spectral_signature_pastis.parquet``
            (resolved via :func:`resolve_dataset_path`, falls back to the
            legacy ``_italy`` if applicable).
        spectral_signature_frame: Optional injection of the
            ``spectral_signature_*`` block (testing); when passed,
            ``spectral_signature_path`` is ignored.

    Returns:
        ``pl.DataFrame`` with shape ``(N, 2 + 189)`` or ``(N, 2 + 701)`` if
        FarSLIP was included. First column ``parcel_id`` (canonical Utf8, see
        :func:`ml.utils.parcel_id.canonical_parcel_id`), second ``year``
        (i16). The rest in the order documented in the module.

    Raises:
        ValueError: if ``parcels`` does not contain the required columns, if
            ``year`` does not match ``parcels['year']``, or if ``stats`` is
            not a subset of the supported stats.
        FileNotFoundError: if ``include_farslip=True`` and an explicit
            ``farslip_path`` that does not exist was passed (structured
            warning for the default path, without failing).
    """
    _validate_parcels(parcels, year=year)
    _validate_stats(stats)

    selected_blocks = tuple(b for b in blocks if b in BLOCK_NAMES)
    block_frames: list[pl.LazyFrame] = []

    # Canonical schema: `parcel_id` always Utf8. We convert from the
    # GeoDataFrame with `.astype("string")` to preserve identities such as
    # `"{patch_id}_{i}"` (baselines US-023-preview) and integers inherited from
    # PASTIS without scientific notation.
    base = pl.from_pandas(
        parcels[["parcel_id", "year"]].astype({"parcel_id": "string", "year": "int16"})
    ).lazy()

    if "alphaearth" in selected_blocks:
        block_frames.append(_build_ae_block(parcels, year=year, injected=ae_frame))
    if "indices_stats" in selected_blocks:
        block_frames.append(
            _build_indices_stats_block(parcels, year=year, stats=stats, injected=indices_frame)
        )
    if "sentinel1" in selected_blocks:
        block_frames.append(_build_s1_block(parcels, year=year, stats=stats, injected=s1_frame))
    if "srtm" in selected_blocks:
        block_frames.append(_build_srtm_block(parcels, injected=srtm_frame))
    if "era5_monthly" in selected_blocks:
        block_frames.append(_build_era5_block(parcels, year=year, injected=era5_frame))
    if "geometry" in selected_blocks:
        block_frames.append(_build_geom_block(parcels))
    if "farslip" in selected_blocks and include_farslip:
        farslip_block = _build_farslip_block(parcels, farslip_path=farslip_path)
        if farslip_block is not None:
            block_frames.append(farslip_block)
    if "phenology_text" in selected_blocks and include_phenology_text:
        pheno_block = _build_phenology_text_block_lf(
            parcels,
            phenology_text_path=phenology_text_path,
            injected=phenology_text_frame,
        )
        if pheno_block is not None:
            block_frames.append(pheno_block)
    if "spectral_signature" in selected_blocks and include_spectral_signature:
        spec_block = _build_spectral_signature_block_lf(
            parcels,
            spectral_signature_path=spectral_signature_path,
            injected=spectral_signature_frame,
        )
        if spec_block is not None:
            block_frames.append(spec_block)

    joined = base
    for block in block_frames:
        joined = joined.join(block, on=["parcel_id", "year"], how="left")

    result = joined.collect(engine="streaming") if lazy else joined.collect()
    # AC-12: stable order by parcel_id to guarantee byte-exact MD5 across
    # re-executions. The Polars streaming engine does not preserve order after
    # lazy joins, so the final sort is necessary for determinism.
    return result.sort("parcel_id")


# ---------------------------------------------------------------------------
# Internal validators.
# ---------------------------------------------------------------------------


def _validate_parcels(parcels: gpd.GeoDataFrame, *, year: int) -> None:
    """Validate that the parcels GeoDataFrame has the minimum columns."""
    if not isinstance(parcels, gpd.GeoDataFrame):  # pragma: no cover - guard
        raise ValueError(f"`parcels` must be a geopandas.GeoDataFrame; received {type(parcels)!r}")
    missing = [c for c in ("parcel_id", "year") if c not in parcels.columns]
    if missing:
        raise ValueError(
            f"`parcels` does not contain required columns: {missing}. "
            "Expected at least: ['parcel_id', 'year', 'geometry']."
        )
    if parcels.geometry.name not in parcels.columns:
        raise ValueError(
            "`parcels` does not contain an active geometry column. "
            "Verify that the GeoDataFrame has `geometry` or set_geometry()."
        )
    unique_years = set(int(y) for y in parcels["year"].unique().tolist())
    if unique_years and unique_years != {int(year)}:
        raise ValueError(
            f"`year={year}` does not match the unique values in parcels['year']="
            f"{sorted(unique_years)}. The fusion assumes a single year per build."
        )


def _validate_stats(stats: tuple[str, ...]) -> None:
    """Validate that `stats` is a recognized subset."""
    supported = {"mean", "std", "p25", "p50", "p75", "p95", "min", "max"}
    invalid = [s for s in stats if s not in supported]
    if invalid:
        raise ValueError(f"Unsupported stats: {invalid}. Available: {sorted(supported)}.")


# ---------------------------------------------------------------------------
# Private helpers — one helper per block.
# ---------------------------------------------------------------------------


def _build_ae_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Build the AlphaEarth block (64 cols) by ``(parcel_id, year)``.

    When ``injected`` is ``None`` and the GEE helper returned no data, the 64
    dims are filled with ``None`` (downstream must impute if required).
    """
    if injected is not None:
        df = injected
    else:
        df = _empty_ae_frame(parcels, year=year)
    df = canonical_parcel_id(df)

    expected_cols = {"parcel_id", "year", *AE_COLS}
    actual_cols = set(df.columns)
    missing_ae = expected_cols - actual_cols
    if missing_ae:
        # Partial injection: we fill with None preserving the contract.
        fill: dict[str, list[float | None]] = {
            c: [None] * df.height for c in sorted(missing_ae) if c in AE_COLS
        }
        if fill:
            df = df.with_columns(
                [pl.Series(name=k, values=v, dtype=pl.Float64) for k, v in fill.items()]
            )
    select_cols = ["parcel_id", "year", *AE_COLS]
    return df.select(select_cols).lazy()


def _empty_ae_frame(parcels: gpd.GeoDataFrame, *, year: int) -> pl.DataFrame:
    """Return an AE frame with the 64 dims filled with ``None``."""
    pids = parcels["parcel_id"].astype("string").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in AE_COLS:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Utf8(),
        "year": pl.Int16(),
    }
    for c in AE_COLS:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)


def _build_indices_stats_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    stats: tuple[str, ...],
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Build the ``{idx}_{stat}`` block (5 stats x 17 indices = 85 cols).

    The column order is ``idx`` outer + ``stat`` inner (NDVI_mean,
    NDVI_std, ... NDVI_p95, NDWI_mean, ...). When ``injected`` is ``None``
    the helper fills with ``None`` preserving the exact contract.
    """
    expected_cols = tuple(f"{idx.lower()}_{stat}" for idx in INDEX_NAMES for stat in stats)
    if injected is not None:
        df = injected
    else:
        df = _empty_indices_frame(parcels, year=year, expected_cols=expected_cols)
    df = canonical_parcel_id(df)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    select_cols = ["parcel_id", "year", *expected_cols]
    return df.select(select_cols).lazy()


def _empty_indices_frame(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    expected_cols: tuple[str, ...],
) -> pl.DataFrame:
    """Indices*stats frame filled with ``None``."""
    pids = parcels["parcel_id"].astype("string").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in expected_cols:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Utf8(),
        "year": pl.Int16(),
    }
    for c in expected_cols:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)


def _build_s1_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    stats: tuple[str, ...],
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Sentinel-1 block VV+VH x stats = 10 cols."""
    expected_cols = tuple(f"s1_{pol}_{stat}" for pol in _S1_POLARIZATIONS for stat in stats)
    if injected is not None:
        df = injected
    else:
        df = _empty_generic_frame(parcels, year=year, columns=expected_cols)
    df = canonical_parcel_id(df)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_srtm_block(
    parcels: gpd.GeoDataFrame,
    *,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """SRTM block (elevation, slope, dominant aspect) = 3 cols.

    Parcels have no associated year in SRTM (static DEM); the helper
    synthesizes ``year`` from the GDF to preserve the join.
    """
    expected_cols = ("srtm_elev_mean", "srtm_slope_mean", "srtm_aspect_dominant")
    year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
    if injected is not None:
        df = injected
    else:
        pids = parcels["parcel_id"].astype("string").tolist()
        n = len(pids)
        df = pl.DataFrame(
            {
                "parcel_id": pids,
                "year": [year_val] * n,
                "srtm_elev_mean": [None] * n,
                "srtm_slope_mean": [None] * n,
                "srtm_aspect_dominant": [None] * n,
            },
            schema={
                "parcel_id": pl.Utf8(),
                "year": pl.Int16(),
                "srtm_elev_mean": pl.Float64(),
                "srtm_slope_mean": pl.Float64(),
                "srtm_aspect_dominant": pl.Utf8(),
            },
        )
    df = canonical_parcel_id(df)

    # SRTM may arrive without year (static DEM): we fill with the base year.
    if "year" not in df.columns:
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        for col in missing:
            dtype = pl.Utf8() if col.endswith("aspect_dominant") else pl.Float64()
            df = df.with_columns(pl.Series(name=col, values=[None] * df.height, dtype=dtype))
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_era5_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Monthly ERA5 block = 24 cols (tmean_m01..12 + prec_m01..12)."""
    expected_cols = tuple(
        [f"era5_tmean_m{m:02d}" for m in range(1, 13)]
        + [f"era5_prec_m{m:02d}" for m in range(1, 13)]
    )
    if injected is not None:
        df = injected
    else:
        df = _empty_generic_frame(parcels, year=year, columns=expected_cols)
    df = canonical_parcel_id(df)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_geom_block(parcels: gpd.GeoDataFrame) -> pl.LazyFrame:
    """Geometry block = 3 cols derived from the EPSG:4326 geometry.

    Reprojects to EPSG:3857 (Web Mercator) to obtain areas and perimeters in
    metric units (suitable approximation for temperate zones / Italy).
    ``geom_elongation = perimeter^2 / (4 * pi * area)`` is the inverse of the
    Polsby-Popper compactness (1 = perfect circle, > 1 elongated shapes).
    """
    if len(parcels) == 0:
        return pl.DataFrame(
            schema={
                "parcel_id": pl.Utf8(),
                "year": pl.Int16(),
                "geom_area_ha": pl.Float64(),
                "geom_perimeter_m": pl.Float64(),
                "geom_elongation": pl.Float64(),
            }
        ).lazy()

    metric = parcels.to_crs("EPSG:3857")
    area_m2 = metric.geometry.area.astype("float64").to_numpy()
    perimeter_m = metric.geometry.length.astype("float64").to_numpy()
    # Avoid division by zero in degenerate geometries.
    safe_area = np.where(area_m2 > 0, area_m2, np.nan)
    elongation = (perimeter_m**2) / (4.0 * np.pi * safe_area)
    area_ha = area_m2 / 10_000.0

    df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("string").tolist(),
            "year": parcels["year"].astype("int16").to_numpy(),
            "geom_area_ha": area_ha,
            "geom_perimeter_m": perimeter_m,
            "geom_elongation": elongation,
        },
        schema={
            "parcel_id": pl.Utf8(),
            "year": pl.Int16(),
            "geom_area_ha": pl.Float64(),
            "geom_perimeter_m": pl.Float64(),
            "geom_elongation": pl.Float64(),
        },
    )
    return df.lazy()


def _build_farslip_block(
    parcels: gpd.GeoDataFrame,
    *,
    farslip_path: str | Path | None,
) -> pl.LazyFrame | None:
    """Read and prepare the FarSLIP block (512 cols) for ``LEFT JOIN``.

    If the path does not exist:

    - If the caller passed an explicit path, raises ``FileNotFoundError`` to
      avoid masking configuration errors.
    - If the default was used and does not exist, emits a structured warning
      and returns ``None`` (the block is omitted without failing).
    """
    explicit_path = farslip_path is not None
    resolved = Path(farslip_path) if explicit_path else resolve_dataset_path(_DEFAULT_FARSLIP_PATH)

    if not resolved.exists():
        if explicit_path:
            raise FileNotFoundError(
                f"FarSLIP embeddings parquet not found: {resolved}. "
                "Pass `include_farslip=False` or verify the path."
            )
        logger.warning(
            "farslip_block_skipped",
            reason="default_path_not_found",
            path=str(resolved),
            note="US-016b aún no ha entregado los embeddings",
        )
        return None

    df = pl.read_parquet(resolved)
    df = canonical_parcel_id(df) if "parcel_id" in df.columns else df
    farslip_cols = tuple(f"farslip_{i:03d}" for i in range(512))
    missing = [c for c in farslip_cols if c not in df.columns]
    if missing:
        # Defensive patch US-023-preview P2 (decision D-1): also accepts
        # the legacy prefix ``farslip_emb_NNN`` (original v1/v2 before
        # the promotion to the canonical path). If we detect the 512 cols with
        # the legacy prefix, we rename them in-memory without touching the parquet.
        legacy_cols = tuple(f"farslip_emb_{i:03d}" for i in range(512))
        if all(c in df.columns for c in legacy_cols):
            df = df.rename({lc: fc for lc, fc in zip(legacy_cols, farslip_cols, strict=True)})
            missing = [c for c in farslip_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"FarSLIP parquet at {resolved} does not bring the 512 expected columns. "
                f"Missing {len(missing)} cols (e.g. {missing[:3]}...). "
                "Accepted prefixes: 'farslip_NNN' (canonical) or 'farslip_emb_NNN' (legacy v1/v2)."
            )
    if "parcel_id" not in df.columns:
        raise ValueError(
            f"FarSLIP parquet at {resolved} does not contain `parcel_id` for the join."
        )

    # If the FarSLIP frame does not bring `year`, we infer the year from parcels.
    if "year" not in df.columns:
        year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))

    return df.select(
        [
            pl.col("parcel_id").cast(pl.Utf8),
            pl.col("year").cast(pl.Int16),
            *[pl.col(c).cast(pl.Float32) for c in farslip_cols],
        ]
    ).lazy()


def _build_phenology_text_block_lf(
    parcels: gpd.GeoDataFrame,
    *,
    phenology_text_path: str | Path | None,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame | None:
    """Prepare the ``pheno_text_*`` block (US-022b-D) for LEFT JOIN.

    Symmetric pattern to the FarSLIP block:

    - If ``injected`` is not ``None``, it is used directly (testing).
    - If the path does not exist and the default was used: warning + ``None``
      (the build continues without failing — the block is omitted).
    - If the path does not exist and was passed explicitly:
      ``FileNotFoundError``.
    - If the ``pheno_text_NNN`` columns do not reach the expected
      :data:`PHENOLOGY_TEXT_EMBED_DIM`, the subset is accepted (there is no
      fixed dimensional contract — the text-encoder can vary).
    """
    if injected is not None:
        df = injected
    else:
        explicit_path = phenology_text_path is not None
        resolved = (
            Path(phenology_text_path)
            if explicit_path
            else resolve_dataset_path(_DEFAULT_PHENO_TEXT_PATH)
        )
        if not resolved.exists():
            if explicit_path:
                raise FileNotFoundError(
                    f"pheno_text block not found at {resolved}. "
                    "Pass `include_phenology_text=False` or generate the parquet with "
                    "ml.features.phenology_description.build_phenology_text_block."
                )
            logger.warning(
                "phenology_text_block_skipped",
                reason="default_path_not_found",
                path=str(resolved),
                note="US-022b-D aun no ha materializado los embeddings textuales",
            )
            return None
        df = pl.read_parquet(resolved)

    if "parcel_id" not in df.columns:
        raise ValueError("pheno_text block does not contain `parcel_id` for the join.")
    df = canonical_parcel_id(df)
    pheno_cols = tuple(c for c in df.columns if c.startswith("pheno_text_"))
    if not pheno_cols:
        raise ValueError("pheno_text block does not contain columns with prefix `pheno_text_`.")
    if "year" not in df.columns:
        year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))
    return df.select(
        [
            pl.col("parcel_id").cast(pl.Utf8),
            pl.col("year").cast(pl.Int16),
            *[pl.col(c).cast(pl.Float32) for c in pheno_cols],
        ]
    ).lazy()


def _build_spectral_signature_block_lf(
    parcels: gpd.GeoDataFrame,
    *,
    spectral_signature_path: str | Path | None,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame | None:
    """Prepare the ``spectral_signature_*`` block (US-023-preview P5) for LEFT JOIN.

    Symmetric pattern to the pheno_text block:

    - If ``injected`` is not ``None``, it is used directly (testing).
    - If the path does not exist and the default was used: warning + ``None``
      (the build continues without failing — the block is omitted).
    - If the path does not exist and was passed explicitly:
      ``FileNotFoundError``.
    - The expected columns have prefix ``spectral_signature_NNN``; the K count
      depends on the descriptor (default 3 for REP / 3 anchors).
    """
    if injected is not None:
        df = injected
    else:
        explicit_path = spectral_signature_path is not None
        resolved = (
            Path(spectral_signature_path)
            if explicit_path
            else resolve_dataset_path(_DEFAULT_SPECTRAL_SIGNATURE_PATH)
        )
        if not resolved.exists():
            if explicit_path:
                raise FileNotFoundError(
                    f"spectral_signature block not found at {resolved}. "
                    "Pass `include_spectral_signature=False` or generate the "
                    "parquet with ml.features.spectral_signature."
                )
            logger.warning(
                "spectral_signature_block_skipped",
                reason="default_path_not_found",
                path=str(resolved),
                note="US-023-preview P5 aun no ha materializado el bloque",
            )
            return None
        df = pl.read_parquet(resolved)

    if "parcel_id" not in df.columns:
        raise ValueError("spectral_signature block does not contain `parcel_id` for the join.")
    df = canonical_parcel_id(df)
    spec_cols = tuple(c for c in df.columns if c.startswith("spectral_signature_"))
    if not spec_cols:
        raise ValueError(
            "spectral_signature block does not contain columns with prefix `spectral_signature_`."
        )
    if "year" not in df.columns:
        year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))
    return df.select(
        [
            pl.col("parcel_id").cast(pl.Utf8),
            pl.col("year").cast(pl.Int16),
            *[pl.col(c).cast(pl.Float32) for c in spec_cols],
        ]
    ).lazy()


def _empty_generic_frame(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Generic frame filled with ``None`` for the given column names."""
    pids = parcels["parcel_id"].astype("string").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in columns:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Utf8(),
        "year": pl.Int16(),
    }
    for c in columns:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)
