"""WorldCereal tropical transfer experiment (Experimento 3, EPIC 12).

Extends the cross-region transfer evidence (so far Europe-only:
EuroCropsML LV/PT -> EE, and Sen4AgriNet FR -> Catalonia) to a different
CLIMATE ZONE -- a tropical agricultural region -- using ESA WorldCereal as
the ground-truth source ingested live from Google Earth Engine.

Dataset
-------
``ESA/WorldCereal/2021/MODELS/v100`` (the Harmonized Global Crops product on
GEE, CC-BY-4.0). It is NOT a single multi-class crop-type map: it is a set of
per-product BINARY masks (``classification`` band, value ``100`` = product
present, ``0`` = absent) organised by agro-ecological zone (``aez_id``) and
season. The products available in the chosen tropical zone are
``temporarycrops`` (annual cropland), ``maize`` (main + second season) and
``wintercereals``. We OVERLAY those binary masks into a single multi-class
crop-type label per pixel (see :func:`worldcereal_label_image`):

  - ``maize``         : pixel flagged maize by WorldCereal (any maize season),
  - ``wintercereals`` : pixel flagged winter cereals (and not maize),
  - ``other_cropland``: temporarycrops=100 but neither maize nor cereals,
  - ``non_crop``      : temporarycrops=0.

Region
------
The default region is the **Brazilian Cerrado / Mato Grosso soy-maize belt**
(WorldCereal AEZ ``20087``, centroid ~14.0 S, 51.7 W), the largest tropical
rain-fed cropping system with all three crop products present. A second
tropical AEZ (Karnataka, India, ``28107``) is provided for a sanity replica.

Features
--------
Each sampled pixel is joined to the REAL 64-dim AlphaEarth annual embedding
(``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1) for 2021 via
:func:`ml.ingest.gee_sampler.sample_alphaearth_at_coords`. This is the SAME
feature space as the European AlphaEarth baseline, so a zero-shot transfer is
representationally well-posed even though the label-spaces differ.

Honesty rules (Arthur + project AC)
-----------------------------------
- All numbers come from REAL GEE pulls. If GEE fails (quota/auth/network) the
  ingest helpers return an EMPTY frame with a valid schema; a curve/metric is
  NEVER fabricated.
- The tropical WorldCereal classes do NOT map cleanly onto the European
  PASTIS-18 / HCAT-macro label-space: WorldCereal only resolves
  maize / winter-cereals / cropland, while the European classifier was trained
  on 18 European agronomic classes. We therefore do NOT force a zero-shot
  PASTIS->Brazil class mapping. Instead the quantitative experiment is a
  FEW-SHOT k-shot classifier trained on the LOCAL tropical classes
  (:func:`run_fewshot_curve`), reusing the baseline XGBoost recipe. The
  zero-shot direction is reported QUALITATIVELY via an embedding-space
  separability score (:func:`zero_shot_separability`), an honest "are the
  tropical classes even linearly separable in AlphaEarth space" probe, never an
  invented PASTIS F1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from ml.ingest.gee_sampler import ALPHAEARTH_DIM_COLS, sample_alphaearth_at_coords
from ml.train.baseline import build_estimator

try:
    import ee  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_REGION",
    "K_SHOTS",
    "TROPICAL_CLASSES",
    "TropicalRegion",
    "WorldCerealDataMissing",
    "build_dataset",
    "run_fewshot_curve",
    "sample_worldcereal_points",
    "summarize_curve",
    "worldcereal_label_image",
    "zero_shot_europe_to_tropics",
    "zero_shot_separability",
]

#: WorldCereal MODELS collection on GEE (Harmonized Global Crops, CC-BY-4.0).
WORLDCEREAL_COLLECTION = "ESA/WorldCereal/2021/MODELS/v100"
#: The reference crop calendar year of WorldCereal MODELS v100.
WORLDCEREAL_YEAR = 2021
#: WorldCereal ``classification`` value flagging "product present".
_PRESENT = 100

#: The multi-class label-space we derive by overlaying the binary products.
TROPICAL_CLASSES: tuple[str, ...] = (
    "non_crop",
    "other_cropland",
    "wintercereals",
    "maize",
)

#: k-shot ladder (mirrors the EuroCropsML protocol so the two curves compare).
K_SHOTS: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)


class WorldCerealDataMissing(RuntimeError):
    """Raised when the WorldCereal/AlphaEarth pull yields no usable real data.

    The pipeline never fabricates numbers: when GEE returns nothing (quota,
    auth, empty footprint) the public entry points raise this typed error so
    a notebook can branch into an explicit ``degraded`` mode.
    """


@dataclass(frozen=True)
class TropicalRegion:
    """A tropical WorldCereal agro-ecological zone used as a transfer target.

    Attributes:
        name: Logical identifier (used in cache keys and outputs).
        aez_id: WorldCereal agro-ecological zone id (``aez_id`` property).
        description: Human-readable region label (country / cropping system).
        lat: Approximate centroid latitude (EPSG:4326), tropical band.
        lon: Approximate centroid longitude (EPSG:4326).
    """

    name: str
    aez_id: int
    description: str
    lat: float
    lon: float


#: Brazilian Cerrado / Mato Grosso soy-maize belt (largest tropical rain-fed
#: cropping system with all three WorldCereal crop products present).
DEFAULT_REGION = TropicalRegion(
    name="brazil_cerrado",
    aez_id=20087,
    description="Brazil Cerrado / Mato Grosso soy-maize belt",
    lat=-14.0,
    lon=-51.7,
)

#: Secondary tropical AEZ for a sanity replica (Karnataka, India).
INDIA_KARNATAKA = TropicalRegion(
    name="india_karnataka",
    aez_id=28107,
    description="India Karnataka semi-arid cropping",
    lat=13.9,
    lon=76.0,
)


def worldcereal_label_image(aez_id: int) -> Any:
    """Overlay WorldCereal binary products of one AEZ into a multi-class label.

    WorldCereal MODELS v100 stores one binary mask per (product, season). We
    build a single integer ``label`` band over the AEZ footprint:

      0 = non_crop        (temporarycrops == 0)
      1 = other_cropland  (temporarycrops == 100, not maize, not wintercereals)
      2 = wintercereals   (wintercereals == 100, not maize)
      3 = maize           (maize == 100 in any maize season)

    Maize takes precedence over winter cereals when both fire (rare; the second
    maize season can overlap the winter-cereals window). The ids match the
    index of :data:`TROPICAL_CLASSES`.

    Args:
        aez_id: WorldCereal ``aez_id`` of the target zone.

    Returns:
        An ``ee.Image`` with a single ``label`` band (Int) over the AEZ
        footprint, plus the AEZ geometry attached as a property is NOT used
        (the caller derives the geometry from ``temporarycrops``).

    Raises:
        ImportError: if ``earthengine-api`` is not installed.
        WorldCerealDataMissing: if the AEZ has no ``temporarycrops`` product.
    """
    if ee is None:  # pragma: no cover - exercised only without the SDK
        raise ImportError("earthengine-api is not installed. Run `poetry install --with ml,geo`.")
    ic = ee.ImageCollection(WORLDCEREAL_COLLECTION).filter(ee.Filter.eq("aez_id", int(aez_id)))

    def _product(product: str) -> Any:
        sub = ic.filter(ee.Filter.eq("product", product))
        # mosaic() merges all seasons of the same product (e.g. maize main +
        # second). present = max over seasons (any season flags the product).
        return sub.max().select("classification")

    tc = _product("temporarycrops")
    maize = _product("maize")
    cereals = _product("wintercereals")

    is_crop = tc.gte(_PRESENT)
    is_maize = maize.gte(_PRESENT)
    is_cereal = cereals.gte(_PRESENT)

    # Priority overlay: maize (3) > wintercereals (2) > other_cropland (1) > non_crop (0)
    label = (
        ee.Image(0)
        .where(is_crop, 1)
        .where(is_cereal.And(is_maize.Not()), 2)
        .where(is_maize, 3)
        .rename("label")
        .toInt()
    )
    return label


def _aez_geometry(aez_id: int) -> Any:
    """Return the WorldCereal AEZ footprint geometry (from temporarycrops tile).

    Args:
        aez_id: WorldCereal ``aez_id``.

    Returns:
        The ``ee.Geometry`` footprint of the ``temporarycrops`` tile.

    Raises:
        WorldCerealDataMissing: if the AEZ has no ``temporarycrops`` image.
    """
    ic = ee.ImageCollection(WORLDCEREAL_COLLECTION).filter(
        ee.Filter.And(
            ee.Filter.eq("aez_id", int(aez_id)),
            ee.Filter.eq("product", "temporarycrops"),
        )
    )
    if ic.size().getInfo() == 0:
        raise WorldCerealDataMissing(f"WorldCereal AEZ {aez_id} has no temporarycrops product.")
    return ic.first().geometry()


def _points_schema() -> dict[str, Any]:
    """Schema of the sampled-points frame (px_id, lon, lat, label, class_name)."""
    return {
        "px_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "label": pl.Int64,
        "class_name": pl.Utf8,
    }


def sample_worldcereal_points(
    region: TropicalRegion = DEFAULT_REGION,
    *,
    n_per_class: int = 400,
    scale: int = 100,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Stratified-sample labelled WorldCereal pixels over a tropical AEZ.

    Builds the multi-class label image (:func:`worldcereal_label_image`) and
    draws up to ``n_per_class`` points per class with ``stratifiedSample`` over
    the AEZ footprint. A coarse ``scale`` (default 100 m, vs WorldCereal's
    native 10 m) keeps the server-side request light while still landing on
    real labelled pixels.

    Args:
        region: The tropical AEZ to sample.
        n_per_class: Target points per class (``stratifiedSample`` numPoints).
        scale: Sampling resolution in meters.
        seed: Deterministic sampling seed.
        cache_dir: Parquet cache folder (default ``data/cache/gee``).

    Returns:
        A Polars frame with columns ``px_id, lon, lat, label, class_name``.
        Empty (valid schema) if GEE is unavailable or the pull fails.
    """
    cache_root = cache_dir or Path("data/cache/gee")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"worldcereal_pts_{region.name}_{n_per_class}_{scale}_{seed}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None:
        return pl.DataFrame(schema=_points_schema())

    try:
        geometry = _aez_geometry(region.aez_id)
        label_img = worldcereal_label_image(region.aez_id)
        sample = label_img.stratifiedSample(
            numPoints=int(n_per_class),
            classBand="label",
            region=geometry,
            scale=int(scale),
            seed=int(seed),
            geometries=True,
            tileScale=4,
        )
        info = sample.getInfo()
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        logger.warning("worldcereal_sample_failed", region=region.name, error=str(exc))
        return pl.DataFrame(schema=_points_schema())

    rows: list[dict[str, object]] = []
    for idx, feat in enumerate(info.get("features", [])):
        props = feat.get("properties", {}) or {}
        coords = (feat.get("geometry", {}) or {}).get("coordinates", [None, None])
        lab = props.get("label")
        if lab is None or len(coords) < 2 or coords[0] is None:
            continue
        lab_int = int(lab)
        rows.append(
            {
                "px_id": f"{region.name}_{idx}",
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "label": lab_int,
                "class_name": TROPICAL_CLASSES[lab_int]
                if 0 <= lab_int < len(TROPICAL_CLASSES)
                else "unknown",
            }
        )

    if not rows:
        return pl.DataFrame(schema=_points_schema())
    frame = pl.DataFrame(rows, schema=_points_schema())
    frame.write_parquet(cache_file)
    return frame


def build_dataset(
    region: TropicalRegion = DEFAULT_REGION,
    *,
    n_per_class: int = 400,
    scale: int = 100,
    seed: int = 42,
    year: int = WORLDCEREAL_YEAR,
    cache_dir: Path | None = None,
    out_path: Path | None = None,
) -> pl.DataFrame:
    """Sample labelled WorldCereal points and join their AlphaEarth embeddings.

    The full ingest pipeline: stratified-sample labelled pixels
    (:func:`sample_worldcereal_points`), pull the REAL 64-dim AlphaEarth annual
    embedding at each point (:func:`ml.ingest.gee_sampler.sample_alphaearth_at_coords`),
    inner-join the two by ``px_id`` and (optionally) persist the result under
    ``data/transfer/worldcereal_<region>.parquet``.

    Args:
        region: The tropical AEZ to ingest.
        n_per_class: Target points per class for the stratified sample.
        scale: Sampling resolution in meters.
        seed: Deterministic sampling seed.
        year: AlphaEarth annual embedding year (default 2021, the WorldCereal
            reference year).
        cache_dir: GEE parquet cache folder.
        out_path: When given, the joined dataset is written here (e.g.
            ``data/transfer/worldcereal_brazil_cerrado.parquet``).

    Returns:
        A Polars frame with columns ``px_id, lon, lat, label, class_name,
        dim_00..dim_63`` (one row per labelled pixel with a complete embedding).

    Raises:
        WorldCerealDataMissing: if no labelled points or no embeddings come back
            from GEE (degraded mode is the caller's responsibility).
    """
    points = sample_worldcereal_points(
        region,
        n_per_class=n_per_class,
        scale=scale,
        seed=seed,
        cache_dir=cache_dir,
    )
    if points.is_empty():
        raise WorldCerealDataMissing(
            f"No WorldCereal labelled points for region {region.name!r}; GEE "
            "returned nothing (check quota/auth)."
        )

    embeddings = sample_alphaearth_at_coords(
        points.select("px_id", "lon", "lat"),
        year=year,
        cache_path=cache_dir,
        cache_key=f"worldcereal_{region.name}",
    )
    if embeddings.is_empty():
        raise WorldCerealDataMissing(
            f"No AlphaEarth embeddings for region {region.name!r}; GEE returned "
            "nothing for the sampled points."
        )

    dataset = (
        points.join(
            embeddings.select("px_id", *ALPHAEARTH_DIM_COLS),
            on="px_id",
            how="inner",
        )
        # Drop pixels with any null embedding dim (incomplete mosaic coverage).
        .drop_nulls(subset=ALPHAEARTH_DIM_COLS)
    )
    if dataset.is_empty():
        raise WorldCerealDataMissing(
            f"WorldCereal x AlphaEarth join for region {region.name!r} is empty "
            "after dropping null embeddings."
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_parquet(out_path)
        logger.info(
            "worldcereal_dataset_saved",
            region=region.name,
            path=str(out_path),
            n=dataset.height,
        )
    return dataset


def _xy(dataset: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the joined dataset into the ``(X_64dim, y_label)`` arrays.

    Args:
        dataset: Frame from :func:`build_dataset`.

    Returns:
        Tuple ``(X, y)`` with the 64-dim AlphaEarth matrix and integer labels.
    """
    x = dataset.select(ALPHAEARTH_DIM_COLS).to_numpy().astype(np.float64)
    y = dataset.get_column("label").to_numpy().astype(np.int64)
    return x, y


def zero_shot_separability(
    dataset: pl.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Honest zero-shot probe: are the tropical classes separable in AlphaEarth?

    The European baseline was trained on 18 European agronomic classes whose
    label-space does NOT include the WorldCereal tropical taxonomy (maize is the
    only shared concept). Rather than fabricate a PASTIS->Brazil mapping, this
    measures how separable the LOCAL tropical classes are *in the same 64-dim
    AlphaEarth space* the European classifier consumes, via a stratified
    cross-validated linear probe (the same feature space, an honest upper bound
    on what any AlphaEarth classifier could resolve here with full supervision).

    Args:
        dataset: Frame from :func:`build_dataset`.
        n_splits: Stratified CV folds.
        seed: Deterministic split seed.

    Returns:
        A dict with ``f1_macro_cv`` (mean over folds), ``f1_macro_std``,
        ``n_samples`` and ``n_classes``. This is the FULLY-supervised in-domain
        separability, the reference ceiling the few-shot curve climbs toward.

    Raises:
        WorldCerealDataMissing: if the dataset is empty.
    """
    if dataset.is_empty():
        raise WorldCerealDataMissing("Empty dataset; cannot probe separability.")
    x, y = _xy(dataset)
    classes = np.unique(y)
    # Need at least n_splits members per class for stratified CV.
    min_count = min(int((y == c).sum()) for c in classes)
    splits = min(n_splits, max(2, min_count))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    f1s: list[float] = []
    for train_idx, test_idx in skf.split(x, y):
        n_tr = len(np.unique(y[train_idx]))
        est = build_estimator("xgb", _baseline_params(n_tr))
        est.fit(x[train_idx], y[train_idx])
        pred = est.predict(x[test_idx])
        f1s.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))
    return {
        "f1_macro_cv": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "n_samples": float(len(y)),
        "n_classes": float(len(classes)),
    }


#: The single PASTIS-18 class name shared with the WorldCereal tropical
#: taxonomy. WorldCereal "maize" == PASTIS "Corn"; every other WorldCereal
#: class (wintercereals/other_cropland/non_crop) has no 1:1 PASTIS leaf.
#: The Corn ``class_id`` is resolved from the table by NAME at runtime (the
#: 1-indexed ``class_id`` of the ``features_xgb_alphaearth_*`` table is 3, but
#: we never hardcode it: a rename or a 0-indexed table would drift silently).
_SHARED_CLASS_EU = "Corn"
#: WorldCereal label id of "maize" (index in :data:`TROPICAL_CLASSES`).
_MAIZE_LABEL = 3

#: Default European AlphaEarth training table (PASTIS-R, 18 French classes).
_EU_TABLE = Path("data/features/features_xgb_alphaearth_avg_2018_2019.parquet")


def zero_shot_europe_to_tropics(
    dataset: pl.DataFrame,
    *,
    eu_table: Path | None = None,
) -> dict[str, float]:
    """Genuine zero-shot: the European AlphaEarth classifier applied to Brazil.

    Trains the baseline XGBoost on the REAL European PASTIS-R AlphaEarth table
    (18 French agronomic classes, ``features_xgb_alphaearth_avg_2018_2019``)
    with NO Brazilian data, then predicts on the Brazilian WorldCereal pixels.
    The European and Brazilian features share the exact 64-dim AlphaEarth space,
    so the model runs -- but its label-space is French.

    HONESTY: only ONE class is shared between the two taxonomies -- PASTIS
    "Corn" == WorldCereal "maize". We therefore do NOT report a multi-class
    French F1 over Brazil (the other 17 French classes have no Brazilian
    ground-truth). Instead we collapse the European prediction to the binary
    "is the model calling this pixel Corn?" and score it against the binary
    WorldCereal "is this pixel maize?" -- a true zero-shot maize-detection F1.
    This is the only metric the label-spaces honestly support.

    Args:
        dataset: Brazilian frame from :func:`build_dataset`.
        eu_table: Override path to the European AlphaEarth training table.

    Returns:
        A dict with ``maize_f1_zero_shot`` (binary F1 of the European Corn
        detector on Brazilian maize), ``maize_precision``, ``maize_recall``,
        ``n_target`` and ``base_rate`` (fraction of Brazilian pixels that are
        maize, the trivial-classifier reference).

    Raises:
        WorldCerealDataMissing: if the dataset or the European table is absent.
    """
    if dataset.is_empty():
        raise WorldCerealDataMissing("Empty target dataset; cannot run zero-shot.")
    table_path = eu_table or _EU_TABLE
    if not table_path.exists():
        raise WorldCerealDataMissing(
            f"European AlphaEarth table not found at {table_path} (DVC artifact "
            "features_xgb_alphaearth_avg_2018_2019.parquet)."
        )
    from sklearn.metrics import precision_recall_fscore_support

    eu = pl.read_parquet(table_path)
    if "class_name" not in eu.columns:
        raise WorldCerealDataMissing(
            f"European table {table_path} lacks a 'class_name' column; cannot "
            f"resolve the shared {_SHARED_CLASS_EU!r} class id by name."
        )
    x_eu = eu.select(ALPHAEARTH_DIM_COLS).to_numpy().astype(np.float64)
    y_eu = eu.get_column("class_id").to_numpy().astype(np.int64)
    # Resolve the shared "Corn" class_id from the table by NAME (never hardcode).
    corn_ids = (
        eu.filter(pl.col("class_name") == _SHARED_CLASS_EU)
        .get_column("class_id")
        .unique()
        .to_list()
    )
    if not corn_ids:
        raise WorldCerealDataMissing(
            f"European table {table_path} has no {_SHARED_CLASS_EU!r} class; the "
            "shared-class zero-shot is undefined."
        )
    corn_class_id = int(corn_ids[0])

    # Train the European 18-class baseline; remap class_ids to contiguous 0..n-1.
    eu_classes = sorted(set(y_eu.tolist()))
    eu_to_idx = {c: i for i, c in enumerate(eu_classes)}
    idx_to_eu = {i: c for c, i in eu_to_idx.items()}
    y_eu_enc = np.array([eu_to_idx[c] for c in y_eu], dtype=np.int64)
    est = build_estimator("xgb", _baseline_params(len(eu_classes)))
    est.fit(x_eu, y_eu_enc)

    x_tgt, y_tgt = _xy(dataset)
    pred_enc = est.predict(x_tgt)
    pred_eu = np.array([idx_to_eu[int(p)] for p in pred_enc], dtype=np.int64)

    # Collapse to the shared concept: European "Corn" vs Brazilian "maize".
    pred_is_maize = (pred_eu == corn_class_id).astype(np.int64)
    true_is_maize = (y_tgt == _MAIZE_LABEL).astype(np.int64)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_is_maize, pred_is_maize, average="binary", zero_division=0
    )
    return {
        "maize_f1_zero_shot": float(f1),
        "maize_precision": float(prec),
        "maize_recall": float(rec),
        "n_target": float(len(y_tgt)),
        "base_rate": float(true_is_maize.mean()),
    }


def _baseline_params(n_classes: int | None = None) -> dict[str, object]:
    """The baseline XGBoost recipe params (same as the European baseline).

    XGBoost rejects ``objective="multi:softprob"`` when the training labels have
    fewer than three classes (it expects ``num_class>=1`` but the sklearn wrapper
    forwards 0 for the binary case). When ``n_classes`` is known to be 2 we fall
    back to ``binary:logistic`` so a degenerate split (e.g. a held-out pool that
    collapses to two classes) never crashes the few-shot loop. The multi-class
    recipe is otherwise untouched, identical to the European baseline.

    Args:
        n_classes: Number of distinct training classes, if known. When 2 the
            binary objective is selected; otherwise the multi-class objective.

    Returns:
        The XGBoost hyperparameter dict.
    """
    objective = "binary:logistic" if n_classes is not None and n_classes <= 2 else "multi:softprob"
    return {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": objective,
        "random_state": 42,
    }


def _sample_k_shot(rng: np.random.Generator, labels: np.ndarray, k: int) -> np.ndarray:
    """Pick up to ``k`` support indices per class.

    Args:
        rng: Seeded generator.
        labels: Per-sample labels.
        k: Shots per class.

    Returns:
        Sorted positional indices of the support set.
    """
    selected: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        selected.extend(idx[:k].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def run_fewshot_curve(
    dataset: pl.DataFrame,
    *,
    k_shots: Iterable[int] = K_SHOTS,
    seeds: Sequence[int] = (0, 1, 2),
    test_fraction: float = 0.4,
) -> pl.DataFrame:
    """Real F1-macro-vs-k few-shot curve over the LOCAL tropical classes.

    For each ``(k, seed)``: hold out ``test_fraction`` per class as the query
    set, draw a k-shot support set per class from the remainder, train the
    baseline XGBoost recipe on the support set and score F1-macro on the query
    set. This quantifies how many local tropical samples are needed before an
    AlphaEarth classifier resolves the WorldCereal classes -- the honest
    few-shot answer that replaces a fabricated zero-shot PASTIS mapping.

    Args:
        dataset: Frame from :func:`build_dataset`.
        k_shots: k ladder (defaults to :data:`K_SHOTS`).
        seeds: Seeds for error bars.
        test_fraction: Per-class fraction held out as the query set.

    Returns:
        A long frame ``(region, k, seed, f1_macro, n_classes, n_train)``.

    Raises:
        WorldCerealDataMissing: if the dataset is empty.
    """
    if dataset.is_empty():
        raise WorldCerealDataMissing("Empty dataset; cannot run few-shot curve.")
    # px_id is "<region>_<idx>"; strip the trailing numeric idx to recover the
    # region name (regions like "brazil_cerrado" carry an internal underscore).
    first_pid = str(dataset.get_column("px_id")[0])
    region_name = first_pid.rsplit("_", 1)[0] if "_" in first_pid else first_pid
    x, y = _xy(dataset)
    rows: list[dict[str, object]] = []
    for k in k_shots:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            test_idx, pool_idx = _stratified_holdout(rng, y, test_fraction)
            support_local = _sample_k_shot(rng, y[pool_idx], int(k))
            support_idx = pool_idx[support_local]
            if support_idx.size == 0:
                continue
            n_sup = len(np.unique(y[support_idx]))
            est = build_estimator("xgb", _baseline_params(n_sup))
            est.fit(x[support_idx], y[support_idx])
            pred = est.predict(x[test_idx])
            f1 = float(f1_score(y[test_idx], pred, average="macro", zero_division=0))
            rows.append(
                {
                    "region": region_name,
                    "k": int(k),
                    "seed": int(seed),
                    "f1_macro": f1,
                    "n_classes": len(np.unique(y[test_idx])),
                    "n_train": int(support_idx.size),
                }
            )
            logger.info(
                "worldcereal_fewshot_point",
                region=region_name,
                k=int(k),
                seed=int(seed),
                f1_macro=round(f1, 4),
            )
    return pl.DataFrame(rows)


def _stratified_holdout(
    rng: np.random.Generator, labels: np.ndarray, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class held-out test split + remaining pool.

    Args:
        rng: Seeded generator.
        labels: Per-sample labels.
        test_fraction: Per-class fraction held out (>=1 sample stays in pool).

    Returns:
        Tuple ``(test_idx, pool_idx)``.
    """
    import math

    test: list[int] = []
    pool: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_test = math.floor(len(idx) * test_fraction)
        if len(idx) >= 2:
            n_test = max(1, min(n_test, len(idx) - 1))
        test.extend(idx[:n_test].tolist())
        pool.extend(idx[n_test:].tolist())
    return np.array(sorted(test), dtype=np.int64), np.array(sorted(pool), dtype=np.int64)


def summarize_curve(curve: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the raw few-shot curve into per-``k`` mean/std F1-macro.

    Args:
        curve: Long frame from :func:`run_fewshot_curve`.

    Returns:
        A frame ``(region, k, f1_mean, f1_std, n_seeds)`` sorted by k.
    """
    return (
        curve.group_by("region", "k")
        .agg(
            pl.col("f1_macro").mean().alias("f1_mean"),
            pl.col("f1_macro").std(ddof=0).fill_null(0.0).alias("f1_std"),
            pl.len().alias("n_seeds"),
        )
        .sort("region", "k")
    )
