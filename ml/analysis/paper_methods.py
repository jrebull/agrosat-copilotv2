"""EDA and feature engineering methods derived from academic literature.

This module translates into reproducible Polars/sklearn code seven methods
extracted from the full reading of four agricultural remote sensing papers.
Each public function cites its source (author + arXiv ID / DOI) in the docstring.

Reference papers
----------------
- Paper A: Russwurm, M., Korner, M. (2018). *Multi-Temporal Land Cover
  Classification with Sequential Recurrent Encoders*. ISPRS International
  Journal of Geo-Information 7(4):129. arXiv:1802.02080. (Provided by the
  sponsor.)
- Paper B: Tarasiou, M., Guler, R.A., Zafeiriou, S. (2021). *Context-self
  contrastive pretraining for crop type semantic segmentation*. IEEE TGRS.
  arXiv:2104.04310. (Provided by the sponsor.)
- Paper C: *Phenology-Aware Transformer (PVM)* (2025). Remote Sensing
  17(14):2346. DOI 10.3390/rs17142346.
- Paper D: Qin, R. et al. (2025). *Spatiotemporal masked pre-training for
  advancing crop mapping on satellite image time series with limited labels
  (STCLN)*. International Journal of Applied Earth Observation and
  Geoinformation.

Polars convention
-----------------
All public functions receive/return :class:`polars.DataFrame` or native
Python structures; ``numpy`` appears only internally at the technical boundary
of ``scipy``/``sklearn`` or when operating on the raw PASTIS-R ``.npy``
tensors. ``structlog`` is used for structured logging, never ``print``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "aggregate_rare_classes",
    "boundary_interior_stats",
    "boundary_pixel_mask",
    "cloud_gap_robustness",
    "compute_boundary_ratio",
    "confusion_symmetry_analysis",
    "phenology_calendar_features",
    "temporal_sampling_stats",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical names of the 4 phenological stages (Paper C, PVM crop-growth
#: calendar). The order matches the ``growth_stage`` index 0..3.
_PHENOLOGY_STAGE_NAMES: tuple[str, ...] = (
    "dormant",
    "green_up",
    "peak",
    "senescence",
)

#: Tolerance in days to consider that a day of the year is "covered" by
#: a satellite observation (Paper A, irregular revisit analysis).
_DOY_COVERAGE_TOLERANCE_DAYS: int = 15


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_numpy(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Convert a sequence or array to a 1D ``np.ndarray``.

    Technical boundary to enter ``scipy``/``sklearn``. If ``values`` is
    already an ``np.ndarray`` it is returned without copy.

    Args:
        values: Python sequence or ``np.ndarray``.

    Returns:
        Resulting ``np.ndarray`` (not necessarily 1D if the input is
        multidimensional).
    """
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values))


def _neighbourhood_varies(window: np.ndarray) -> bool:
    """Indicate whether an NxN window contains more than one distinct value.

    Helper of :func:`boundary_pixel_mask`. A pixel is a boundary (Paper B,
    Tarasiou et al. 2021) when not all the ground truths of its neighborhood
    share the same value.

    Args:
        window: 2D sub-array of the semantic mask.

    Returns:
        ``True`` if the window has at least two distinct values.
    """
    return bool(np.unique(window).size > 1)


def _doy_from_yyyymmdd(date_int: int) -> int:
    """Convert a ``YYYYMMDD`` integer to day-of-year (1..366).

    Args:
        date_int: Date as a ``YYYYMMDD`` integer (PASTIS-R ``dates-S2``
            format).

    Returns:
        Day-of-year in ``[1, 366]``. If the date is invalid returns ``0``.
    """
    try:
        date_str = f"{int(date_int):08d}"
        dt = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        year_start = np.datetime64(f"{date_str[:4]}-01-01")
        return int((dt - year_start) / np.timedelta64(1, "D")) + 1
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# (B) boundary_pixel_mask — Paper B (Tarasiou et al. 2021, arXiv:2104.04310)
# ---------------------------------------------------------------------------


def boundary_pixel_mask(
    semantic: np.ndarray,
    *,
    neighbourhood: int = 3,
) -> np.ndarray:
    """Mark parcel boundary pixels over a semantic mask.

    Implements the boundary definition of Tarasiou et al. 2021 (Context-self
    contrastive pretraining, arXiv:2104.04310): a pixel is a boundary when
    **not all** the ground truths of its ``NxN`` neighborhood share the same
    value. The paper shows (Fig. 2) that the spectral variance of a parcel
    comes almost entirely from these edge pixels.

    Args:
        semantic: 2D semantic mask ``(H, W)`` with per-pixel class ids
            (PASTIS-R ``TARGET[0]`` channel).
        neighbourhood: Side of the square window (default 3 -> 3x3
            neighborhood). Must be odd and >= 3.

    Returns:
        Boolean array ``(H, W)`` with ``True`` at the boundary pixels.

    Raises:
        ValueError: If ``semantic`` is not 2D or ``neighbourhood`` is even or < 3.
    """
    arr = np.asarray(semantic)
    if arr.ndim != 2:
        raise ValueError(f"semantic must be 2D (H, W); got ndim={arr.ndim}")
    if neighbourhood < 3 or neighbourhood % 2 == 0:
        raise ValueError(f"neighbourhood must be odd and >= 3; got {neighbourhood}")

    h, w = arr.shape
    radius = neighbourhood // 2
    mask = np.zeros((h, w), dtype=bool)

    # Edge-replication padding to avoid introducing false contours.
    padded = np.pad(arr, radius, mode="edge")
    for i in range(h):
        for j in range(w):
            window = padded[i : i + neighbourhood, j : j + neighbourhood]
            mask[i, j] = _neighbourhood_varies(window)

    logger.info(
        "boundary_pixel_mask_computed",
        shape=(h, w),
        neighbourhood=neighbourhood,
        n_boundary=int(mask.sum()),
        boundary_ratio=float(mask.mean()) if mask.size else 0.0,
    )
    return mask


# ---------------------------------------------------------------------------
# (A) boundary_interior_stats — Paper B (Tarasiou et al. 2021, Fig. 2)
# ---------------------------------------------------------------------------


def boundary_interior_stats(
    patch: dict[str, Any],
    *,
    band_index: int = 6,
    neighbourhood: int = 3,
) -> pl.DataFrame:
    """Spectral statistics per interior / boundary / exterior group.

    Reproduces the analysis of Figure 2 of Tarasiou et al. 2021
    (arXiv:2104.04310): classifies each pixel of a PASTIS-R patch into one of
    three groups from the semantic mask and reports descriptive statistics of
    the chosen band. The paper demonstrates that interior pixels are
    homogeneous and that the spectral dispersion lives at the boundary.

    Group definition:
        - ``exterior``: background pixels (class 0).
        - ``boundary``: pixels whose ``NxN`` neighborhood is not homogeneous
          (see :func:`boundary_pixel_mask`).
        - ``interior``: parcel pixels (class > 0) that are not boundary.

    Args:
        patch: PASTIS-R patch dictionary as returned by
            ``ml.ingest.pastis_loader.load_pastis_patch`` (keys ``s2`` with
            shape ``(T, 10, H, W)`` and ``semantic`` with shape ``(H, W)``).
        band_index: Sentinel-2 band index to analyze (default 6 = B08
            NIR, the band used in Fig. 2 of the paper).
        neighbourhood: Side of the window to detect boundaries (default 3).

    Returns:
        :class:`polars.DataFrame` with one row per group and columns
        ``group, mean, std, p25, p50, p75, count``. The band is averaged
        temporally over the ``T`` axis before computing the descriptive
        statistics.

    Raises:
        ValueError: If ``patch`` lacks ``s2``/``semantic`` or ``band_index``
            is out of range.
    """
    schema: dict[str, Any] = {
        "group": pl.Utf8,
        "mean": pl.Float64,
        "std": pl.Float64,
        "p25": pl.Float64,
        "p50": pl.Float64,
        "p75": pl.Float64,
        "count": pl.Int64,
    }

    s2 = patch.get("s2")
    semantic = patch.get("semantic")
    if s2 is None or semantic is None:
        logger.warning("boundary_interior_stats_missing_data")
        return pl.DataFrame(schema=schema)

    s2_arr = np.asarray(s2, dtype=np.float64)
    if s2_arr.ndim != 4:
        raise ValueError(f"patch['s2'] must be 4D (T, bands, H, W); got ndim={s2_arr.ndim}")
    n_bands = s2_arr.shape[1]
    if not 0 <= band_index < n_bands:
        raise ValueError(f"band_index={band_index} out of range [0, {n_bands - 1}]")

    semantic_arr = np.asarray(semantic)
    # Temporal average of the chosen band -> 2D map (H, W).
    band_map = s2_arr[:, band_index, :, :].mean(axis=0)

    boundary = boundary_pixel_mask(semantic_arr, neighbourhood=neighbourhood)
    is_parcel = semantic_arr > 0
    exterior = ~is_parcel
    boundary_in_parcel = boundary & is_parcel
    interior = is_parcel & ~boundary

    groups: dict[str, np.ndarray] = {
        "interior": interior,
        "boundary": boundary_in_parcel,
        "exterior": exterior,
    }

    rows: list[dict[str, Any]] = []
    for name, group_mask in groups.items():
        vals = band_map[group_mask]
        if vals.size == 0:
            rows.append(
                {
                    "group": name,
                    "mean": None,
                    "std": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "group": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "count": int(vals.size),
            }
        )

    logger.info(
        "boundary_interior_stats_computed",
        band_index=band_index,
        n_interior=int(interior.sum()),
        n_boundary=int(boundary_in_parcel.sum()),
        n_exterior=int(exterior.sum()),
    )
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# (compute_boundary_ratio) — Paper B (Tarasiou et al. 2021)
# ---------------------------------------------------------------------------


def compute_boundary_ratio(
    patch: dict[str, Any],
    *,
    neighbourhood: int = 3,
) -> dict[int, float]:
    """Compute the fraction of boundary pixels per parcel instance.

    New per-parcel feature motivated by Tarasiou et al. 2021
    (arXiv:2104.04310): since the edge pixel concentrates the discriminant
    signal, the ratio ``boundary_pixels / total_pixels`` of a parcel is a
    descriptor of its geometry (small or irregular parcels have a high ratio;
    large and compact parcels a low ratio).

    Documented here (and not in ``ml/features/selection.py``) because it
    depends directly on :func:`boundary_pixel_mask` and operates on the raw
    PASTIS-R patch tensor, not on the wide-format feature DataFrame consumed
    by ``selection.py``.

    Args:
        patch: PASTIS-R patch dictionary (``ml.ingest.pastis_loader.
            load_pastis_patch``) with keys ``semantic`` and ``instance``
            (channels ``TARGET[0]`` and ``TARGET[1]``).
        neighbourhood: Side of the window to detect boundaries (default 3).

    Returns:
        Dictionary ``{instance_id: boundary_ratio}`` with a float in
        ``[0, 1]`` per parcel instance. Instance 0 (background) is
        excluded. Empty if the patch does not carry ``instance``.
    """
    semantic = patch.get("semantic")
    instance = patch.get("instance")
    if semantic is None or instance is None:
        logger.warning("compute_boundary_ratio_missing_instance")
        return {}

    semantic_arr = np.asarray(semantic)
    instance_arr = np.asarray(instance)
    boundary = boundary_pixel_mask(semantic_arr, neighbourhood=neighbourhood)

    ratios: dict[int, float] = {}
    for inst_id in np.unique(instance_arr):
        iid = int(inst_id)
        if iid == 0:
            continue  # background
        inst_mask = instance_arr == inst_id
        total = int(inst_mask.sum())
        if total == 0:
            continue
        n_boundary = int((inst_mask & boundary).sum())
        ratios[iid] = float(n_boundary / total)

    logger.info(
        "compute_boundary_ratio_computed",
        n_instances=len(ratios),
        mean_ratio=float(np.mean(list(ratios.values()))) if ratios else 0.0,
    )
    return ratios


# ---------------------------------------------------------------------------
# (C) temporal_sampling_stats — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def temporal_sampling_stats(dates: list[int]) -> dict[str, float | int]:
    """Characterize the irregularity of the satellite revisit.

    Analysis derived from Russwurm & Korner 2018 (arXiv:1802.02080), which
    documents how Sentinel-2 delivers acquisitions with non-uniform spacing
    (gaps due to cloud cover) and treats these gaps as temporal noise.
    This function quantifies that irregularity for a specific series.

    Args:
        dates: List of acquisition dates as ``YYYYMMDD`` integers
            (PASTIS-R ``dates-S2`` format). The internal order does not matter;
            it is sorted before computing gaps.

    Returns:
        Dictionary with:
            - ``n_obs``: number of observations.
            - ``mean_gap_days``: mean gap between consecutive acquisitions.
            - ``max_gap_days``: maximum gap.
            - ``min_gap_days``: minimum gap.
            - ``std_gap_days``: standard deviation of the gaps.
            - ``doy_coverage``: fraction of the year (0..1) with at least one
              observation within +/- 15 days.
        If ``dates`` has fewer than 2 valid dates, the gap fields are
        ``0.0`` and ``doy_coverage`` is computed with the available dates.
    """
    valid = [int(d) for d in dates if int(d) > 0]
    n_obs = len(valid)
    base: dict[str, float | int] = {
        "n_obs": n_obs,
        "mean_gap_days": 0.0,
        "max_gap_days": 0.0,
        "min_gap_days": 0.0,
        "std_gap_days": 0.0,
        "doy_coverage": 0.0,
    }
    if n_obs == 0:
        return base

    # Day-of-year of each acquisition for the annual coverage.
    doys = sorted(d for d in (_doy_from_yyyymmdd(v) for v in valid) if d > 0)

    if n_obs >= 2:
        # Gaps in absolute calendar days (not DOY, to cross years).
        ordered = sorted(valid)
        days = np.array(
            [
                (
                    np.datetime64(f"{d:08d}"[:4] + "-" + f"{d:08d}"[4:6] + "-" + f"{d:08d}"[6:8])
                    - np.datetime64("2000-01-01")
                )
                / np.timedelta64(1, "D")
                for d in ordered
            ],
            dtype=np.float64,
        )
        gaps = np.diff(days)
        base["mean_gap_days"] = float(np.mean(gaps))
        base["max_gap_days"] = float(np.max(gaps))
        base["min_gap_days"] = float(np.min(gaps))
        base["std_gap_days"] = float(np.std(gaps))

    # Coverage: fraction of the 365 days of the year with an observation at <=15 days.
    if doys:
        doy_arr = np.array(doys, dtype=np.int64)
        all_days = np.arange(1, 366)
        covered = np.zeros(all_days.size, dtype=bool)
        for d in doy_arr:
            covered |= np.abs(all_days - d) <= _DOY_COVERAGE_TOLERANCE_DAYS
        base["doy_coverage"] = float(covered.mean())

    logger.info(
        "temporal_sampling_stats_computed",
        n_obs=n_obs,
        mean_gap_days=base["mean_gap_days"],
        doy_coverage=base["doy_coverage"],
    )
    return base


# ---------------------------------------------------------------------------
# (D) confusion_symmetry_analysis — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def confusion_symmetry_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Decompose the confusion matrix into symmetric/asymmetric components.

    Russwurm & Korner 2018 (arXiv:1802.02080) distinguish two types of
    confusion between classes: the **symmetric** ones (e.g. triticale<->rye)
    reveal spectral/phenological similarity, while the **asymmetric** ones
    point to external factors (class imbalance, annotation errors). For each
    pair of classes ``(i, j)`` this function computes:

        - symmetric component: ``min(C[i, j], C[j, i])``.
        - asymmetric component: ``abs(C[i, j] - C[j, i])``.

    Args:
        y_true: Vector of true labels.
        y_pred: Vector of predicted labels (same length as ``y_true``).
        class_names: Optional mapping ``{class_id: name}`` to label the
            result in a readable way.

    Returns:
        :class:`polars.DataFrame` with columns ``class_a, class_b, symmetric,
        asymmetric, interpretation`` sorted by total confusion
        (``symmetric + asymmetric``) descending. ``interpretation`` is
        ``"spectral_similarity"`` when the symmetric component dominates, or
        ``"external_factor"`` otherwise.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
    """
    schema: dict[str, Any] = {
        "class_a": pl.Utf8,
        "class_b": pl.Utf8,
        "symmetric": pl.Int64,
        "asymmetric": pl.Int64,
        "interpretation": pl.Utf8,
    }
    yt = _to_numpy(y_true).astype(np.int64).ravel()
    yp = _to_numpy(y_pred).astype(np.int64).ravel()
    if yt.size != yp.size:
        raise ValueError(f"y_true and y_pred must have equal length; {yt.size} != {yp.size}")
    if yt.size == 0:
        return pl.DataFrame(schema=schema)

    classes = sorted(set(yt.tolist()) | set(yp.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist(), strict=True):
        cm[idx[t], idx[p]] += 1

    def _label(class_id: int) -> str:
        if class_names is not None and class_id in class_names:
            return str(class_names[class_id])
        return str(class_id)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = int(cm[i, j])
            cji = int(cm[j, i])
            if cij == 0 and cji == 0:
                continue
            symmetric = min(cij, cji)
            asymmetric = abs(cij - cji)
            interpretation = "spectral_similarity" if symmetric >= asymmetric else "external_factor"
            rows.append(
                {
                    "class_a": _label(classes[i]),
                    "class_b": _label(classes[j]),
                    "symmetric": symmetric,
                    "asymmetric": asymmetric,
                    "interpretation": interpretation,
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df = (
        df.with_columns((pl.col("symmetric") + pl.col("asymmetric")).alias("__total"))
        .sort("__total", descending=True)
        .drop("__total")
    )

    logger.info(
        "confusion_symmetry_analysis_computed",
        n_classes=n,
        n_pairs=df.height,
    )
    return df


# ---------------------------------------------------------------------------
# (E) aggregate_rare_classes — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def aggregate_rare_classes(
    y: pl.Series,
    *,
    min_count: int = 400,
    other_label: int = -1,
) -> tuple[pl.Series, dict[Any, Any]]:
    """Collapse infrequent classes into a single aggregated label.

    Russwurm & Korner 2018 (arXiv:1802.02080) report a very imbalanced class
    distribution (maize 919k px vs peas 6k px) and aggregate the infrequent
    classes applying a count threshold ("classes occurring >= 400 times"),
    reducing ~200 labels to 17. This function replicates that strategy: the
    classes with fewer than ``min_count`` occurrences are reassigned to
    ``other_label``.

    Args:
        y: Polars Series with the class labels (integer).
        min_count: Minimum occurrence threshold to keep a class as its own
            category (default 400, the paper's value).
        other_label: Target label for the aggregated classes (default -1).

    Returns:
        Tuple ``(remapped_series, report)`` where:
            - ``remapped_series`` is the series with the rare classes collapsed
              (same ``name`` as ``y``).
            - ``report`` contains ``{original_class: count}`` for each original
              class plus the key ``"aggregated"`` with the list of collapsed
              classes and ``"min_count"`` with the threshold used.
    """
    # value_counts returns columns [<name>, "count"]; the first column
    # contains the distinct values, the second their frequency.
    vc = y.value_counts(sort=True)
    value_col = vc.columns[0]
    count_map: dict[int, int] = {
        int(v): int(c)
        for v, c in zip(
            vc.get_column(value_col).to_list(),
            vc.get_column("count").to_list(),
            strict=True,
        )
        if v is not None
    }

    aggregated = sorted(c for c, n in count_map.items() if n < min_count)
    aggregated_set = set(aggregated)

    original = y.to_list()
    remapped = [
        other_label if (v is not None and int(v) in aggregated_set) else v for v in original
    ]
    remapped_series = pl.Series(y.name or "class", remapped, dtype=pl.Int64)

    # The report mixes int keys (per-class count) with str keys
    # ("aggregated", "min_count", "other_label"), hence the type is dict[Any, Any].
    report: dict[Any, Any] = {int(c): int(n) for c, n in count_map.items()}
    report["aggregated"] = aggregated
    report["min_count"] = int(min_count)
    report["other_label"] = int(other_label)

    logger.info(
        "aggregate_rare_classes_done",
        n_classes_original=len(count_map),
        n_aggregated=len(aggregated),
        min_count=min_count,
    )
    return remapped_series, report


# ---------------------------------------------------------------------------
# (F) phenology_calendar_features — Paper C (PVM 2025, RS 17(14):2346)
# ---------------------------------------------------------------------------


def phenology_calendar_features(
    temporal_df: pl.DataFrame,
    *,
    doy_col: str = "peak_doy",
    n_stages: int = 4,
) -> pl.DataFrame:
    """Derive a categorical growth stage from a day-of-year.

    Inspired by the "crop-growth calendar" of the Phenology-Aware Transformer
    (PVM, Remote Sensing 17(14):2346, 2025): the model encodes the phenology
    stages (sowing/growth/peak/harvest) as a vector indexed by day-of-year and
    weights the temporal attention with those cues. Here an EDA version of the
    concept is built: the day-of-year of a phenology metric is discretized into
    ``n_stages`` calendar stages.

    Args:
        temporal_df: Temporal feature DataFrame that already contains the
            ``doy_col`` column (typically ``peak_doy``, ``sog_doy`` or
            ``senescence_doy`` produced by
            ``ml.features.temporal_features.extract_temporal_features``).
        doy_col: Name of the day-of-year column to discretize
            (default ``"peak_doy"``).
        n_stages: Number of phenology stages (default 4 ->
            ``dormant/green_up/peak/senescence``). The year is split into
            ``n_stages`` equal DOY intervals.

    Returns:
        Original DataFrame with two new columns:
            - ``growth_stage`` (Int64): stage index in ``[0, n_stages-1]``.
            - ``growth_stage_name`` (Utf8): readable stage name.
        Rows with null ``doy_col`` receive ``growth_stage = -1`` and
        ``growth_stage_name = "unknown"``.

    Raises:
        ValueError: If ``doy_col`` does not exist in ``temporal_df`` or
            ``n_stages < 2``.
    """
    if doy_col not in temporal_df.columns:
        raise ValueError(
            f"doy_col {doy_col!r} not present in temporal_df.columns; "
            f"available: {temporal_df.columns}"
        )
    if n_stages < 2:
        raise ValueError(f"n_stages must be >= 2; got {n_stages}")

    # Stage names: use the canonical ones when n_stages == 4, otherwise
    # generate generic labels stage_0..stage_{n-1}.
    if n_stages == len(_PHENOLOGY_STAGE_NAMES):
        stage_names = list(_PHENOLOGY_STAGE_NAMES)
    else:
        stage_names = [f"stage_{i}" for i in range(n_stages)]

    stage_width = 366.0 / n_stages
    doy = temporal_df.get_column(doy_col).cast(pl.Float64).to_list()

    stage_idx: list[int] = []
    stage_lbl: list[str] = []
    for value in doy:
        if value is None or not np.isfinite(value):
            stage_idx.append(-1)
            stage_lbl.append("unknown")
            continue
        idx = int(min(n_stages - 1, max(0, int((value - 1.0) / stage_width))))
        stage_idx.append(idx)
        stage_lbl.append(stage_names[idx])

    out = temporal_df.with_columns(
        [
            pl.Series("growth_stage", stage_idx, dtype=pl.Int64),
            pl.Series("growth_stage_name", stage_lbl, dtype=pl.Utf8),
        ]
    )
    logger.info(
        "phenology_calendar_features_done",
        doy_col=doy_col,
        n_stages=n_stages,
        n_rows=out.height,
        n_unknown=sum(1 for s in stage_idx if s == -1),
    )
    return out


# ---------------------------------------------------------------------------
# (G) cloud_gap_robustness — Paper D (Qin et al. 2025, STCLN)
# ---------------------------------------------------------------------------


def cloud_gap_robustness(
    temporal_extractor_callable: Callable[[Any], pl.DataFrame],
    parcel_timeseries: Any,
    *,
    mask_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
    seed: int = 42,
) -> pl.DataFrame:
    """Measure feature drift when simulating cloud gaps in the series.

    Inspired by the spatiotemporal masking of Qin et al. 2025 (STCLN, Int. J.
    Applied Earth Obs. Geoinf.): the pretraining masks temporal patches of the
    image series to learn robust representations. Here the same idea is used as
    an EDA tool: increasing fractions of a parcel's timesteps are removed, the
    temporal feature extractor is re-run, and how much the values shift with
    respect to the unmasked baseline is quantified.

    Args:
        temporal_extractor_callable: Function that receives a parcel temporal
            series (``xarray.DataArray`` with dims ``(time, band)``) and
            returns a :class:`polars.DataFrame` of features (typically
            ``ml.features.temporal_features.extract_temporal_features``).
        parcel_timeseries: Parcel temporal series accepted by
            ``temporal_extractor_callable``. Must expose a ``time`` dimension
            indexable via ``.isel(time=...)`` (xarray contract).
        mask_fractions: Fractions of timesteps to remove. Must include
            ``0.0`` (baseline) as the first element so the drift is
            relative to the full series.
        seed: Seed of the random generator (reproducibility).

    Returns:
        :class:`polars.DataFrame` with columns ``mask_fraction,
        n_timesteps_kept, feature_name, value, drift_from_baseline``. One row
        per (fraction, numeric feature). ``drift_from_baseline`` is the
        absolute value of the difference with respect to the run with
        ``mask_fraction == 0``.

    Notes:
        If a fraction leaves fewer than 2 timesteps, that fraction is skipped
        (the temporal extractor requires >= 2 points to interpolate). Only
        numeric columns other than ``parcel_id``/``year`` are tracked.
    """
    schema: dict[str, Any] = {
        "mask_fraction": pl.Float64,
        "n_timesteps_kept": pl.Int64,
        "feature_name": pl.Utf8,
        "value": pl.Float64,
        "drift_from_baseline": pl.Float64,
    }
    rng = np.random.default_rng(seed)

    # Total number of timesteps of the series.
    try:
        n_total = int(parcel_timeseries.sizes["time"])
    except (AttributeError, KeyError, TypeError):
        try:
            n_total = len(parcel_timeseries.coords["time"])
        except Exception:  # noqa: BLE001
            logger.warning("cloud_gap_robustness_no_time_dim")
            return pl.DataFrame(schema=schema)

    if n_total < 2:
        logger.warning("cloud_gap_robustness_series_too_short", n_total=n_total)
        return pl.DataFrame(schema=schema)

    baseline_values: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for fraction in mask_fractions:
        n_keep = round(n_total * (1.0 - fraction))
        n_keep = max(0, min(n_total, n_keep))
        if n_keep < 2:
            logger.info(
                "cloud_gap_robustness_skip_fraction",
                mask_fraction=fraction,
                n_keep=n_keep,
            )
            continue

        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        masked = parcel_timeseries.isel(time=keep_idx)
        try:
            features = temporal_extractor_callable(masked)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cloud_gap_robustness_extractor_failed",
                mask_fraction=fraction,
                error=str(exc),
            )
            continue

        numeric_cols = [
            c
            for c in features.columns
            if c not in ("parcel_id", "year") and features.schema[c].is_numeric()
        ]
        if features.height == 0:
            continue
        feature_row = features.row(0, named=True)

        for col in numeric_cols:
            raw = feature_row[col]
            value = float(raw) if raw is not None and np.isfinite(raw) else float("nan")
            if abs(fraction) < 1e-9:
                baseline_values[col] = value
                drift = 0.0
            else:
                base_val = baseline_values.get(col)
                if base_val is None or not np.isfinite(base_val) or not np.isfinite(value):
                    drift = float("nan")
                else:
                    drift = abs(value - base_val)
            rows.append(
                {
                    "mask_fraction": float(fraction),
                    "n_timesteps_kept": int(n_keep),
                    "feature_name": col,
                    "value": value,
                    "drift_from_baseline": drift,
                }
            )

    logger.info(
        "cloud_gap_robustness_done",
        n_total_timesteps=n_total,
        mask_fractions=list(mask_fractions),
        n_rows=len(rows),
    )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)
