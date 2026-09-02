"""Temporal vs annual feature comparison for phenology-similar crops (EPIC 12).

Motivation (load-bearing finding)
---------------------------------
The multi-region model (:mod:`ml.transfer.multiregion_model`) is built on the
AlphaEarth **annual** embedding: one 64-dim vector per parcel per year. That
collapses the whole growing season into a single point, so two crops that look
the same at peak greenness but are sown in different seasons are nearly
indistinguishable to it. The concrete failure cases in EuroCropsML Estonia +
Latvia are the *phenologically similar* cereals/oilseeds:

  - ``spring_barley`` vs ``winter_barley`` (same crop, autumn vs spring sowing),
  - ``oats`` vs ``spring_common_soft_wheat`` vs ``winter_common_soft_wheat``,
  - ``spring_rapeseed_rape`` vs ``winter_rapeseed_rape``,
  - ``rye`` / ``spring_triticale`` (winter-hardy small grains).

The distinguishing evidence is *when* the canopy greens up, which only the
**Sentinel-2 time series** carries. EuroCropsML ships that series as one
``.npz`` per parcel (``data`` shape ``(T, 13)``, ``dates`` shape ``(T,)``), and
the AlphaEarth parquet of the very same parcels carries the ``npz_name`` column,
so we can build BOTH feature views over the IDENTICAL parcels and labels and
compare them head-to-head on the IDENTICAL spatial-aware split.

What this module does
---------------------
1. Loads the EuroCropsML AlphaEarth parquet (Estonia + Latvia) which already
   pairs every parcel with: 64-dim AlphaEarth annual embedding, ``hcat_code``,
   ``macro_hcat_group``, ``lon``/``lat`` and ``npz_name``.
2. For each parcel, reads the real S2 ``.npz`` series and reduces it to a
   fixed-width **temporal** feature vector: per-band temporal stats + per-index
   (NDVI/NDRE/GCVI/NDWI/PSRI/EVI) temporal stats + NDVI phenology (peak DOY,
   start-of-greenness DOY, AUC, amplitude, early/late-season green-up means) +
   NDVI FFT harmonics. The early-season green-up mean is the explicit
   winter-vs-spring discriminator that the annual embedding cannot encode.
3. Resolves the ``hcat_code`` to the fine HCAT v3 leaf name (the same crosswalk
   the multi-region model uses).
4. Trains the champion XGBoost recipe (:data:`ml.train.baseline._XGB_BASE_PARAMS`)
   TWICE on the SAME spatial-aware train/test split -- once on the AlphaEarth
   annual embedding, once on the temporal vector -- and reports per-class F1 for
   BOTH, plus the per-class delta, focused on the phenology-similar leaves.

The split is spatial (H3 + KMeans buffered folds, :mod:`ml.features.spatial_split`)
so the comparison is not inflated by spatial autocorrelation; both feature views
see the EXACT same parcels in train and test.

Honesty
-------
- No number is fabricated. If a parcel's ``.npz`` is missing it is dropped (and
  counted); if a whole region parquet is missing the builder raises.
- The verdict is whatever the numbers say. If temporal features do NOT lift the
  phenology-similar leaves, that is reported as-is (it would be evidence the
  bottleneck is deeper than annual-vs-temporal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score, precision_recall_fscore_support

from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "PHENO_SIMILAR_LEAVES",
    "S2_BAND_NAMES",
    "TemporalComparisonResult",
    "build_aligned_dataset",
    "make_figure",
    "parcel_temporal_vector",
    "run_temporal_vs_annual",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_TRANSFER_DIR: Path = _REPO_ROOT / "data" / "transfer"
_EUROCROPS_DIR: Path = _TRANSFER_DIR / "eurocropsml"
_PREPROCESS_DIR: Path = _EUROCROPS_DIR / "preprocess"
_REFERENCE_DIR: Path = _REPO_ROOT / "data" / "reference"
_HCAT3_CSV: Path = _REFERENCE_DIR / "eurocrops_hcat3.csv"

#: EuroCropsML S2 band order (``eurocropsml.acquisition.config.S2_BANDS``;
#: "order is important"). 13 bands; B10 (cirrus, index 10) is near-zero over
#: land and excluded from index math.
S2_BAND_NAMES: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)
_BIDX: dict[str, int] = {name: i for i, name in enumerate(S2_BAND_NAMES)}

#: 64-dim AlphaEarth annual embedding columns in the parquet.
_ALPHAEARTH_COLS: tuple[str, ...] = tuple(f"dim_{i:02d}" for i in range(64))

#: Bands used for index math (B10 cirrus excluded).
_INDEX_BANDS: tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

#: The phenology-similar fine leaves the annual embedding struggles to separate.
#: These are the headline classes for the temporal-vs-annual verdict. The set is
#: chosen on agronomy (winter vs spring forms of the same small grains / oilseed,
#: plus oats which overlaps wheat at peak), NOT tuned on results.
PHENO_SIMILAR_LEAVES: tuple[str, ...] = (
    "spring_barley",
    "winter_barley",
    "oats",
    "spring_common_soft_wheat",
    "winter_common_soft_wheat",
    "rye",
    "spring_rapeseed_rape",
    "winter_rapeseed_rape",
    "summer_rapeseed_rape",
)

#: Minimum parcel support for a leaf to enter the classifier (rare-tail guard,
#: mirrors the multi-region MIN_LEAF_SUPPORT so the label space is comparable).
_MIN_LEAF_SUPPORT: int = 50

_TEST_FRACTION: float = 0.30
_RANDOM_STATE: int = 42

#: Reflectance scale of the raw DN (0-10000 -> [0, 1]).
_DN_SCALE: float = 1e4


@dataclass
class TemporalComparisonResult:
    """Per-class F1 of temporal vs annual features on the SAME parcels/split."""

    per_class: pl.DataFrame
    pheno_per_class: pl.DataFrame
    summary: dict[str, object] = field(default_factory=dict)
    feature_meta: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Label resolution
# --------------------------------------------------------------------------- #
def _load_hcat_name_map() -> dict[int, str]:
    """Return ``{hcat_code: hcat_leaf_name}`` from the HCAT v3 reference CSV."""
    if not _HCAT3_CSV.is_file():
        raise FileNotFoundError(f"HCAT v3 reference missing at {_HCAT3_CSV}")
    h = pl.read_csv(_HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8})
    return {int(c): str(n) for c, n in zip(h["HCAT3_code"], h["HCAT3_name"], strict=True)}


# --------------------------------------------------------------------------- #
# Temporal feature engineering
# --------------------------------------------------------------------------- #
def _spectral_index_curves(data: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-timestep spectral index curves from a ``(T, 13)`` DN series.

    Bands are scaled DN/10000 to reflectance. Indices use the canonical project
    formulas (NDVI, NDRE, GCVI, NDWI, PSRI, EVI). Division guards avoid blow-ups
    on near-zero denominators.

    Args:
        data: S2 series of shape ``(T, 13)`` in raw DN (0-10000).

    Returns:
        Mapping ``index_name -> (T,)`` array of index values.
    """
    refl = data.astype(np.float64) / _DN_SCALE
    b = {name: refl[:, _BIDX[name]] for name in _INDEX_BANDS}
    eps = 1e-6

    def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        ratio: np.ndarray = num / np.where(np.abs(den) < eps, np.nan, den)
        return ratio

    blue, green, red = b["B02"], b["B03"], b["B04"]
    re1, nir = b["B05"], b["B08"]
    swir1 = b["B11"]

    return {
        "NDVI": _ratio(nir - red, nir + red),
        "NDRE": _ratio(nir - re1, nir + re1),  # red-edge NDVI (canopy N/structure)
        "GCVI": _ratio(nir, green) - 1.0,  # green chlorophyll vegetation index
        "NDWI": _ratio(nir - swir1, nir + swir1),  # NDMI-style water/moisture
        "PSRI": _ratio(red - green, re1),  # plant senescence reflectance index
        "EVI": 2.5 * _ratio(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0),
    }


def _doy_axis(dates: np.ndarray) -> np.ndarray:
    """Convert ``datetime64`` dates to day-of-year (1-365) float array."""
    d = dates.astype("datetime64[D]")
    year_start = (d.astype("datetime64[Y]")).astype("datetime64[D]")
    doy = (d - year_start) / np.timedelta64(1, "D") + 1.0
    doy_float: np.ndarray = doy.astype(np.float64)
    return doy_float


def _ndvi_phenology(doy: np.ndarray, ndvi: np.ndarray) -> dict[str, float]:
    """Extract phenology metrics from an NDVI curve sampled at ``doy``.

    The metrics target the winter-vs-spring discrimination: early-season and
    late-season green-up means, peak timing/value, amplitude, start-of-greenness
    DOY (first up-crossing of 0.3), and trapezoidal AUC. NaNs (clouds) are
    dropped before computing.

    Args:
        doy: day-of-year axis of shape ``(T,)``.
        ndvi: NDVI curve of shape ``(T,)`` (may contain NaN).

    Returns:
        Mapping of 9 phenology feature names to float values (0.0 when undefined).
    """
    mask = np.isfinite(ndvi)
    if mask.sum() < 2:
        return {k: 0.0 for k in _PHENO_KEYS}
    d, v = doy[mask], ndvi[mask]
    order = np.argsort(d)
    d, v = d[order], v[order]

    peak_idx = int(np.argmax(v))
    peak_doy = float(d[peak_idx])
    peak_val = float(v[peak_idx])
    vmin = float(v.min())
    amplitude = peak_val - vmin

    early = v[d < 120.0]  # before May: winter crops already green, spring bare
    mid = v[(d >= 120.0) & (d < 210.0)]
    late = v[d >= 270.0]  # after Oct: winter regrowth / spring fully senesced
    early_mean = float(early.mean()) if early.size else 0.0
    mid_mean = float(mid.mean()) if mid.size else 0.0
    late_mean = float(late.mean()) if late.size else 0.0

    # Start of greenness: first up-crossing of 0.3 (White et al. 1997).
    sog_doy = 0.0
    thr = 0.3
    for i in range(1, v.size):
        if v[i - 1] < thr <= v[i]:
            sog_doy = float(d[i])
            break

    auc = float(np.trapezoid(v, d)) if v.size >= 2 else 0.0

    return {
        "peak_doy": peak_doy,
        "peak_val": peak_val,
        "amplitude": amplitude,
        "early_mean": early_mean,
        "mid_mean": mid_mean,
        "late_mean": late_mean,
        "sog_doy": sog_doy,
        "ndvi_auc": auc,
        "ndvi_min": vmin,
    }


_PHENO_KEYS: tuple[str, ...] = (
    "peak_doy",
    "peak_val",
    "amplitude",
    "early_mean",
    "mid_mean",
    "late_mean",
    "sog_doy",
    "ndvi_auc",
    "ndvi_min",
)

#: Index curves summarised by temporal stats (mean/std/min/max/p10/p50/p90).
_STAT_INDICES: tuple[str, ...] = ("NDVI", "NDRE", "GCVI", "NDWI", "PSRI", "EVI")
_STAT_FNS: tuple[str, ...] = ("mean", "std", "min", "max", "p10", "p50", "p90")
#: FFT harmonics extracted from the daily-interpolated NDVI curve (DC + 3).
_N_FFT: int = 4


def _temporal_stats(curve: np.ndarray) -> list[float]:
    """Seven robust temporal statistics of one index curve (NaN-aware)."""
    v = curve[np.isfinite(curve)]
    if v.size == 0:
        return [0.0] * len(_STAT_FNS)
    return [
        float(v.mean()),
        float(v.std()),
        float(v.min()),
        float(v.max()),
        float(np.percentile(v, 10)),
        float(np.percentile(v, 50)),
        float(np.percentile(v, 90)),
    ]


def _ndvi_fft(doy: np.ndarray, ndvi: np.ndarray) -> list[float]:
    """Amplitudes of the first ``_N_FFT`` harmonics of the daily-interp NDVI.

    The NDVI is linearly interpolated to a daily grid over its observed window
    then ``rfft`` amplitudes (single-sided) of the DC + 3 harmonics are returned.
    The first harmonic amplitude is the dominant annual greenness wave; its phase
    relative to the season encodes sowing timing.
    """
    mask = np.isfinite(ndvi)
    if mask.sum() < 3:
        return [0.0] * (2 * _N_FFT)
    d, v = doy[mask], ndvi[mask]
    order = np.argsort(d)
    d, v = d[order], v[order]
    grid = np.arange(d.min(), d.max() + 1.0)
    if grid.size < 4:
        return [0.0] * (2 * _N_FFT)
    daily = np.interp(grid, d, v)
    spec = np.fft.rfft(daily)
    n = daily.size
    out: list[float] = []
    for k in range(_N_FFT):
        if k >= spec.size:
            out.extend([0.0, 0.0])
            continue
        amp = float(np.abs(spec[k]) / n) if k == 0 else float(np.abs(spec[k]) * 2.0 / n)
        phase = 0.0 if k == 0 else float(np.angle(spec[k]))
        out.extend([amp, phase])
    return out


def parcel_temporal_vector(data: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Reduce a parcel S2 series to a fixed-width temporal feature vector.

    Layout (length = ``len(_INDEX_BANDS)*4 + len(_STAT_INDICES)*7 + 9 + 8``):

      - per raw band (10 index-bands): temporal mean, std, p10, p90 (40),
      - per index (6): 7 temporal stats (42),
      - NDVI phenology (9),
      - NDVI FFT amp+phase of DC + 3 harmonics (8).

    Total = 99 features. Output dimensionality is independent of ``T`` so the
    vector is comparable across parcels with different revisit counts.

    Args:
        data: S2 series of shape ``(T, 13)`` in raw DN.
        dates: ``datetime64`` array of shape ``(T,)``.

    Returns:
        A 1-D ``float64`` feature vector.
    """
    refl = data.astype(np.float64) / _DN_SCALE
    band_feats: list[float] = []
    for name in _INDEX_BANDS:
        col = refl[:, _BIDX[name]]
        col = col[np.isfinite(col)]
        if col.size == 0:
            band_feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            band_feats.extend(
                [
                    float(col.mean()),
                    float(col.std()),
                    float(np.percentile(col, 10)),
                    float(np.percentile(col, 90)),
                ]
            )

    curves = _spectral_index_curves(data)
    idx_feats: list[float] = []
    for idx in _STAT_INDICES:
        idx_feats.extend(_temporal_stats(curves[idx]))

    doy = _doy_axis(dates)
    pheno = _ndvi_phenology(doy, curves["NDVI"])
    pheno_feats = [pheno[k] for k in _PHENO_KEYS]
    fft_feats = _ndvi_fft(doy, curves["NDVI"])

    vec = np.array(band_feats + idx_feats + pheno_feats + fft_feats, dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_names() -> list[str]:
    """Human-readable names for the temporal feature vector (for importances)."""
    names: list[str] = []
    for name in _INDEX_BANDS:
        names += [f"{name}_mean", f"{name}_std", f"{name}_p10", f"{name}_p90"]
    for idx in _STAT_INDICES:
        names += [f"{idx}_{s}" for s in _STAT_FNS]
    names += [f"pheno_{k}" for k in _PHENO_KEYS]
    for k in range(_N_FFT):
        names += [f"ndvi_fft{k}_amp", f"ndvi_fft{k}_phase"]
    return names


# --------------------------------------------------------------------------- #
# Aligned dataset (annual + temporal over the SAME parcels)
# --------------------------------------------------------------------------- #
@dataclass
class _AlignedDataset:
    annual: np.ndarray  # (n, 64)
    temporal: np.ndarray  # (n, F)
    leaf: np.ndarray  # (n,)
    lon: np.ndarray
    lat: np.ndarray
    n_missing_npz: int
    leaf_vocabulary: list[str]


def build_aligned_dataset(
    *,
    regions: tuple[str, ...] = ("estonia", "latvia"),
    min_leaf_support: int = _MIN_LEAF_SUPPORT,
    max_parcels: int | None = None,
) -> _AlignedDataset:
    """Pair AlphaEarth annual + temporal vectors for the SAME EuroCropsML parcels.

    Args:
        regions: EuroCropsML country keys to pool.
        min_leaf_support: parcel-count floor for a leaf to enter the label space.
        max_parcels: optional cap (per pooled set) for a fast smoke run; ``None``
            uses every parcel with an available ``.npz``.

    Returns:
        An :class:`_AlignedDataset`.

    Raises:
        FileNotFoundError: if a region parquet or the preprocess dir is missing.
    """
    if not _PREPROCESS_DIR.is_dir():
        raise FileNotFoundError(f"EuroCropsML preprocess dir missing at {_PREPROCESS_DIR}")

    # Cache the (slow) per-parcel npz pass for the FULL population so repeated
    # experiments do not re-read ~42k .npz files. Only the uncapped build is
    # cached (a cap is a fast random subsample of that population). The cache
    # lives OUTSIDE the DVC-tracked ``eurocropsml/`` dir so it does not perturb
    # the ``eurocropsml.dvc`` directory hash.
    cache_path = (
        _TRANSFER_DIR
        / "_temporal_cache"
        / f"temporal_aligned_{'_'.join(regions)}_supp{min_leaf_support}.parquet"
    )
    if max_parcels is None and cache_path.is_file():
        cached = pl.read_parquet(cache_path)
        return _aligned_from_frame(cached)

    name_map = _load_hcat_name_map()

    frames: list[pl.DataFrame] = []
    for region in regions:
        p = _TRANSFER_DIR / f"eurocropsml_alphaearth_{region}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"EuroCropsML parquet missing at {p}")
        df = pl.read_parquet(p, columns=["npz_name", "hcat_code", "lon", "lat", *_ALPHAEARTH_COLS])
        df = df.with_columns(
            pl.col("hcat_code")
            .cast(pl.Int64)
            .replace_strict(name_map, default="unknown_hcat", return_dtype=pl.Utf8)
            .alias("leaf")
        )
        frames.append(df)
    pooled = pl.concat(frames)

    # Rare-tail guard: keep leaves with enough support.
    counts = pooled.group_by("leaf").len().filter(pl.col("len") >= min_leaf_support)
    keep = set(counts["leaf"].to_list())
    pooled = pooled.filter(pl.col("leaf").is_in(list(keep)))
    if max_parcels is not None and pooled.height > max_parcels:
        pooled = pooled.sample(n=max_parcels, seed=_RANDOM_STATE, shuffle=True)

    annual_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    leaf_rows: list[str] = []
    lon_rows: list[float] = []
    lat_rows: list[float] = []
    n_missing = 0

    npz_names = pooled["npz_name"].to_list()
    leaves = pooled["leaf"].to_list()
    lons = pooled["lon"].to_list()
    lats = pooled["lat"].to_list()
    annual_mat = pooled.select(_ALPHAEARTH_COLS).to_numpy().astype(np.float64)

    for i, npz_name in enumerate(npz_names):
        npz_path = _PREPROCESS_DIR / npz_name
        if not npz_path.is_file():
            n_missing += 1
            continue
        try:
            payload = np.load(npz_path)  # allow_pickle=False: data/dates arrays planos
            data = payload["data"]
            dates = payload["dates"]
        except Exception:  # noqa: BLE001 -- corrupt npz is dropped, counted
            n_missing += 1
            continue
        if data.ndim != 2 or data.shape[0] < 2:
            n_missing += 1
            continue
        temporal_rows.append(parcel_temporal_vector(data, dates))
        annual_rows.append(annual_mat[i])
        leaf_rows.append(leaves[i])
        lon_rows.append(lons[i])
        lat_rows.append(lats[i])

    annual = np.asarray(annual_rows, dtype=np.float64)
    temporal = np.asarray(temporal_rows, dtype=np.float64)
    leaf = np.asarray(leaf_rows)
    vocab = sorted(set(leaf.tolist()))
    logger.info(
        "aligned_dataset_built",
        n_parcels=int(annual.shape[0]),
        n_missing_npz=n_missing,
        n_leaf_classes=len(vocab),
        temporal_dim=int(temporal.shape[1]) if temporal.size else 0,
    )
    ds = _AlignedDataset(
        annual=annual,
        temporal=temporal,
        leaf=leaf,
        lon=np.asarray(lon_rows, dtype=np.float64),
        lat=np.asarray(lat_rows, dtype=np.float64),
        n_missing_npz=n_missing,
        leaf_vocabulary=vocab,
    )
    if max_parcels is None:
        _cache_aligned(ds, cache_path)
    return ds


def _cache_aligned(ds: _AlignedDataset, cache_path: Path) -> None:
    """Persist the aligned dataset (annual + temporal + meta) to a parquet cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, object] = {
        "leaf": ds.leaf.tolist(),
        "lon": ds.lon.tolist(),
        "lat": ds.lat.tolist(),
        "n_missing_npz": [ds.n_missing_npz] * ds.annual.shape[0],
    }
    for j in range(ds.annual.shape[1]):
        cols[f"a{j:02d}"] = ds.annual[:, j].tolist()
    for j in range(ds.temporal.shape[1]):
        cols[f"t{j:03d}"] = ds.temporal[:, j].tolist()
    pl.DataFrame(cols).write_parquet(cache_path)
    logger.info("temporal_cache_written", path=str(cache_path), n=ds.annual.shape[0])


def _aligned_from_frame(frame: pl.DataFrame) -> _AlignedDataset:
    """Reconstruct an :class:`_AlignedDataset` from the parquet cache frame."""
    a_cols = sorted(c for c in frame.columns if c.startswith("a") and c[1:].isdigit())
    t_cols = sorted(c for c in frame.columns if c.startswith("t") and c[1:].isdigit())
    leaf = frame.get_column("leaf").to_numpy()
    logger.info("temporal_cache_loaded", n=frame.height, temporal_dim=len(t_cols))
    return _AlignedDataset(
        annual=frame.select(a_cols).to_numpy().astype(np.float64),
        temporal=frame.select(t_cols).to_numpy().astype(np.float64),
        leaf=leaf,
        lon=frame.get_column("lon").to_numpy().astype(np.float64),
        lat=frame.get_column("lat").to_numpy().astype(np.float64),
        n_missing_npz=int(frame.get_column("n_missing_npz")[0]) if frame.height else 0,
        leaf_vocabulary=sorted(set(leaf.tolist())),
    )


# --------------------------------------------------------------------------- #
# Spatial-aware split (shared by both feature views)
# --------------------------------------------------------------------------- #
#: Number of spatial blocks; the held-out blocks closest to ``_TEST_FRACTION``
#: of the parcels form the test set.
_N_SPATIAL_BLOCKS: int = 12


def _spatial_split(
    lon: np.ndarray, lat: np.ndarray, leaf: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a spatial-block train/test mask via KMeans on lon/lat.

    Parcels are clustered into :data:`_N_SPATIAL_BLOCKS` spatially contiguous
    blocks by KMeans on their centroids; whole blocks are assigned to the test
    set (greedily, largest-first) until ``~_TEST_FRACTION`` of parcels is held
    out. Because the test blocks are spatially separate from the train blocks the
    F1 is not inflated by spatial autocorrelation between neighbouring parcels --
    the same principle as the project's H3+KMeans buffered folds
    (:mod:`ml.features.spatial_split`) without the geopandas/h3 hard dependency.
    Both feature views (annual, temporal) see the IDENTICAL masks.

    Falls back to a stratified random split only if KMeans cannot run (logged).

    Returns:
        ``(is_train, is_test)`` boolean masks over the parcels.
    """
    n = lon.shape[0]
    try:
        from sklearn.cluster import KMeans

        coords = np.column_stack([lon, lat])
        n_blocks = min(_N_SPATIAL_BLOCKS, n)
        labels = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(coords)
        block_ids, block_sizes = np.unique(labels, return_counts=True)
        # Greedily add whole blocks (largest first) to the test set until we
        # reach the target test fraction.
        order = block_ids[np.argsort(-block_sizes)]
        target = round(_TEST_FRACTION * n)
        is_test = np.zeros(n, dtype=bool)
        held = 0
        for blk in order:
            if held >= target:
                break
            is_test[labels == blk] = True
            held = int(is_test.sum())
        if 0 < is_test.sum() < n:
            logger.info(
                "spatial_split_used",
                n_blocks=int(n_blocks),
                n_test=int(is_test.sum()),
                test_fraction=float(is_test.mean()),
            )
            return ~is_test, is_test
        logger.warning("spatial_split_degenerate_fallback_random")
    except Exception as exc:  # noqa: BLE001 -- documented fallback
        logger.warning("spatial_split_unavailable_fallback_random", error=str(exc))

    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    _, counts = np.unique(leaf, return_counts=True)
    stratify = leaf if counts.min() >= 2 else None
    tr, _te = train_test_split(idx, test_size=_TEST_FRACTION, random_state=seed, stratify=stratify)
    is_train = np.zeros(n, dtype=bool)
    is_train[tr] = True
    return is_train, ~is_train


# --------------------------------------------------------------------------- #
# Train + evaluate both views
# --------------------------------------------------------------------------- #
def _fit_xgb(feats: np.ndarray, y: np.ndarray, seed: int) -> ClassifierMixin:
    """Fit the champion XGBoost classifier (CPU, deterministic)."""
    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    params["device"] = "cpu"
    model = build_estimator("xgb", params)
    model.fit(feats, y)
    return model


def _fit_and_score(
    feats_tr: np.ndarray,
    feats_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train once and return ``(f1_per_class, support, macro_f1)`` on the test."""
    model = _fit_xgb(feats_tr, y_tr, seed)
    pred = model.predict(feats_te)
    _p, _r, f1, sup = precision_recall_fscore_support(
        y_te, pred, labels=np.arange(classes.shape[0]), average=None, zero_division=0
    )
    macro = float(f1_score(y_te, pred, average="macro"))
    return f1, sup.astype(int), macro


def run_temporal_vs_annual(
    *,
    regions: tuple[str, ...] = ("estonia", "latvia"),
    seed: int = _RANDOM_STATE,
    max_parcels: int | None = None,
) -> TemporalComparisonResult:
    """Compare temporal vs AlphaEarth-annual features on the SAME parcels/split.

    Trains the champion XGBoost on THREE feature views on the IDENTICAL
    spatial-block train/test split over the IDENTICAL parcels and labels, and
    reports per-class F1 for each plus the deltas, focused on the
    phenology-similar leaves:

      - **annual**: the AlphaEarth 64-dim annual embedding (the status quo),
      - **temporal**: the S2 time-series feature vector (this experiment),
      - **fusion**: annual ++ temporal concatenated (does temporal ADD signal on
        top of annual, even if it does not beat it alone?).

    Args:
        regions: EuroCropsML country keys to pool.
        seed: RNG seed for the split and the boosters.
        max_parcels: optional parcel cap for a fast run.

    Returns:
        A :class:`TemporalComparisonResult`.
    """
    ds = build_aligned_dataset(regions=regions, max_parcels=max_parcels)
    is_train, is_test = _spatial_split(ds.lon, ds.lat, ds.leaf, seed=seed)

    classes = np.array(sorted(set(ds.leaf.tolist())))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in ds.leaf], dtype=np.int64)
    fusion = np.concatenate([ds.annual, ds.temporal], axis=1)

    f1_annual, sup, macro_annual = _fit_and_score(
        ds.annual[is_train], ds.annual[is_test], y[is_train], y[is_test], classes, seed
    )
    f1_temporal, _, macro_temporal = _fit_and_score(
        ds.temporal[is_train], ds.temporal[is_test], y[is_train], y[is_test], classes, seed
    )
    f1_fusion, _, macro_fusion = _fit_and_score(
        fusion[is_train], fusion[is_test], y[is_train], y[is_test], classes, seed
    )

    per_class = pl.DataFrame(
        {
            "leaf": classes.tolist(),
            "f1_annual": f1_annual.tolist(),
            "f1_temporal": f1_temporal.tolist(),
            "f1_fusion": f1_fusion.tolist(),
            "delta_f1": (f1_temporal - f1_annual).tolist(),
            "delta_f1_fusion": (f1_fusion - f1_annual).tolist(),
            "support_test": sup.tolist(),
            "pheno_similar": [c in PHENO_SIMILAR_LEAVES for c in classes.tolist()],
        }
    ).sort("delta_f1", descending=True)

    pheno = per_class.filter(pl.col("pheno_similar") & (pl.col("support_test") > 0))

    pheno_mean_annual = (
        float(cast(float, pheno.get_column("f1_annual").mean())) if pheno.height else 0.0
    )
    pheno_mean_temporal = (
        float(cast(float, pheno.get_column("f1_temporal").mean())) if pheno.height else 0.0
    )
    pheno_mean_fusion = (
        float(cast(float, pheno.get_column("f1_fusion").mean())) if pheno.height else 0.0
    )

    summary = {
        "regions": list(regions),
        "n_parcels": int(ds.annual.shape[0]),
        "n_missing_npz": int(ds.n_missing_npz),
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "n_leaf_classes": int(classes.shape[0]),
        "annual_dim": int(ds.annual.shape[1]),
        "temporal_dim": int(ds.temporal.shape[1]),
        "fusion_dim": int(fusion.shape[1]),
        "macro_f1_annual": macro_annual,
        "macro_f1_temporal": macro_temporal,
        "macro_f1_fusion": macro_fusion,
        "macro_f1_delta_temporal": macro_temporal - macro_annual,
        "macro_f1_delta_fusion": macro_fusion - macro_annual,
        "pheno_n_leaves": int(pheno.height),
        "pheno_mean_f1_annual": pheno_mean_annual,
        "pheno_mean_f1_temporal": pheno_mean_temporal,
        "pheno_mean_f1_fusion": pheno_mean_fusion,
        "pheno_mean_f1_delta_temporal": pheno_mean_temporal - pheno_mean_annual,
        "pheno_mean_f1_delta_fusion": pheno_mean_fusion - pheno_mean_annual,
        "pheno_n_improved_temporal": int(pheno.filter(pl.col("delta_f1") > 1e-9).height),
        "pheno_n_worsened_temporal": int(pheno.filter(pl.col("delta_f1") < -1e-9).height),
        "pheno_n_improved_fusion": int(pheno.filter(pl.col("delta_f1_fusion") > 1e-9).height),
        "pheno_n_worsened_fusion": int(pheno.filter(pl.col("delta_f1_fusion") < -1e-9).height),
        "pheno_per_class": pheno.select(
            [
                "leaf",
                "f1_annual",
                "f1_temporal",
                "f1_fusion",
                "delta_f1",
                "delta_f1_fusion",
                "support_test",
            ]
        ).to_dicts(),
    }
    logger.info(
        "temporal_vs_annual_done",
        **{
            k: summary[k]
            for k in (
                "n_parcels",
                "n_leaf_classes",
                "macro_f1_annual",
                "macro_f1_temporal",
                "macro_f1_fusion",
                "pheno_mean_f1_delta_temporal",
                "pheno_mean_f1_delta_fusion",
            )
        },
    )
    return TemporalComparisonResult(
        per_class=per_class,
        pheno_per_class=pheno,
        summary=summary,
        feature_meta={"temporal_feature_names": _feature_names()},
    )


# --------------------------------------------------------------------------- #
# Figure + persistence
# --------------------------------------------------------------------------- #
def make_figure(result: TemporalComparisonResult, fig_path: Path) -> Path:
    """Render the phenology-class F1 comparison figure (annual/temporal/fusion).

    Grouped horizontal bars per phenology-similar leaf: annual F1 vs temporal F1
    vs fusion F1, with the temporal and fusion deltas annotated. The verdict is
    visible at a glance.

    Args:
        result: the comparison result.
        fig_path: output PNG path (parent created if missing).

    Returns:
        The written PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    pheno = result.pheno_per_class.sort("delta_f1")
    names = pheno["leaf"].to_list()
    f1_a = pheno["f1_annual"].to_list()
    f1_t = pheno["f1_temporal"].to_list()
    f1_f = pheno["f1_fusion"].to_list()
    deltas = pheno["delta_f1"].to_list()
    deltas_f = pheno["delta_f1_fusion"].to_list()
    sup = pheno["support_test"].to_list()

    y = np.arange(len(names))
    height = 0.26
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.85 * len(names) + 1.5)))
    ax.barh(y + height, f1_a, height=height, color="#9aa0a6", label="AlphaEarth anual (64-dim)")
    ax.barh(y, f1_t, height=height, color="#1f77b4", label="Temporal S2 (99-dim)")
    ax.barh(
        y - height,
        f1_f,
        height=height,
        color="#2ca02c",
        label="Fusion anual+temporal (163-dim)",
    )
    for i, (a, t, f, d, df, s) in enumerate(
        zip(f1_a, f1_t, f1_f, deltas, deltas_f, sup, strict=True)
    ):
        ax.text(
            max(a, t, f) + 0.01,
            i,
            f"dT={d:+.3f} dF={df:+.3f} (n={s})",
            va="center",
            fontsize=7.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("F1 por clase (mismo split espacial, mismas parcelas)")
    ax.set_title(
        "Rescate fenologico: features temporales S2 vs embedding AlphaEarth anual\n"
        "(clases con fenologia similar: invierno vs primavera, avena vs trigo; "
        "dT=delta temporal, dF=delta fusion)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    logger.info("temporal_figure_written", path=str(fig_path))
    return fig_path


def save_outputs(result: TemporalComparisonResult, out_dir: Path) -> None:
    """Persist the per-class tables and the JSON summary to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_class.write_parquet(out_dir / "temporal_vs_annual_per_class.parquet")
    result.pheno_per_class.write_parquet(out_dir / "temporal_vs_annual_pheno.parquet")
    (out_dir / "temporal_vs_annual_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("temporal_outputs_saved", out_dir=str(out_dir))
