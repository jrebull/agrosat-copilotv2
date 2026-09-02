"""Cross-region transfer PoC: PASTIS-R (2019) -> BreizhCrops (2017).

Honest de-risk of the project's cross-region re-pivot. Measures whether a
tabular model trained on PASTIS-R generalizes to BreizhCrops (another region of
France, another year) on the subset of common classes, WITHOUT retraining.

Why it is a hard generalization test
------------------------------------
- Different region: PASTIS-R covers southern/central-eastern France; BreizhCrops
  covers Brittany (frh01 + frh04), oceanic climate, a different agricultural calendar.
- Different year: PASTIS-R 2019 vs BreizhCrops 2017 (different phenology, different
  cloud conditions).
- Different processing: BreizhCrops brings raw DN bands without a cloud mask;
  PASTIS-R arrives filtered. The adapter scales to reflectance but does NOT
  re-mask, so NDVI saturates more (worst case for the model).

Shared feature space
--------------------
The project's 185 base features (153 statistics of 17 indices + 24 FFT
+ 8 phenological) are computed with THE SAME pipeline on both datasets:

- PASTIS-R: US-018 subset already materialized
  (``data/test_fixtures/feature_selection_parcels_subset.parquet``).
- BreizhCrops: raw bands -> ``ml.features.breizhcrops_features
  .build_breizhcrops_features`` (reuses ``compute_index`` +
  ``extract_temporal_features``, the same modules as PASTIS).

AlphaEarth does NOT enter the transfer: BreizhCrops has no AlphaEarth embeddings,
so the only honest common space is the 185 tabular features.

Mapping of the 7 common classes
-------------------------------
Semantic intersection PASTIS-R <-> BreizhCrops::

    common class  PASTIS-R class_id           BreizhCrops class_name
    -----------   -----------------           ----------------------
    wheat         2 (Soft winter wheat),      "wheat"
                  11 (Winter durum wheat)
    barley        4 (Winter barley),          "barley"
                  6 (Spring barley)
    rapeseed      5 (Winter rapeseed)         "rapeseed"
    corn          3 (Corn)                    "corn"
    sunflower     7 (Sunflower)               "sunflower"
    meadow        1 (Meadow)                  "permanent meadows",
                                              "temporary meadows"
    orchard       16 (Orchard)                "orchards"

``sunflower`` barely exists in BreizhCrops (1-2 parcels in frh01/frh04):
it is kept in the mapping for completeness but its F1 will be 0 or noise; it is
reported explicitly.

Output
------
- ``reports/transfer/pastis_to_breizhcrops.parquet``: per-class F1 + F1-macro
  for direct transfer (and pheno-text if feasible).

Usage::

    python scripts/poc_transfer_pastis_to_breizhcrops.py
    python scripts/poc_transfer_pastis_to_breizhcrops.py --n-breiz 2500
    python scripts/poc_transfer_pastis_to_breizhcrops.py --try-pheno-text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from ml.features.breizhcrops_features import build_breizhcrops_features
from ml.features.temporal_features import DEFAULT_INDICES, extract_temporal_features  # noqa: F401
from ml.ingest.breizhcrops_loader import breizhcrops_parcel_index, breizhcrops_pixel_series
from ml.train.baseline import build_estimator

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PASTIS_SUBSET = _REPO_ROOT / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet"
_OUTPUT = _REPO_ROOT / "reports" / "transfer" / "pastis_to_breizhcrops.parquet"

# ---------------------------------------------------------------------------
# Explicit mapping to the 7 common classes (see docstring).
# ---------------------------------------------------------------------------

#: The 7 common classes in stable canonical order.
COMMON_CLASSES: tuple[str, ...] = (
    "wheat",
    "barley",
    "rapeseed",
    "corn",
    "sunflower",
    "meadow",
    "orchard",
)

#: PASTIS-R class_id -> common class. The absent class_id values (8 Grapevine,
#: 9 Beet, etc.) are discarded: they have no counterpart in BreizhCrops.
PASTIS_ID_TO_COMMON: dict[int, str] = {
    2: "wheat",  # Soft winter wheat
    11: "wheat",  # Winter durum wheat
    4: "barley",  # Winter barley
    6: "barley",  # Spring barley
    5: "rapeseed",  # Winter rapeseed
    3: "corn",  # Corn
    7: "sunflower",  # Sunflower
    1: "meadow",  # Meadow
    16: "orchard",  # Orchard
}

#: BreizhCrops class_name -> common class. The classes without a counterpart in
#: PASTIS (nuts) are discarded.
BREIZ_NAME_TO_COMMON: dict[str, str] = {
    "wheat": "wheat",
    "barley": "barley",
    "rapeseed": "rapeseed",
    "corn": "corn",
    "sunflower": "sunflower",
    "permanent meadows": "meadow",
    "temporary meadows": "meadow",
    "orchards": "orchard",
}


# ---------------------------------------------------------------------------
# Load + preparation of PASTIS-R (train).
# ---------------------------------------------------------------------------


def _load_pastis_common(subset_path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Loads the PASTIS-R subset and restricts it to the 7 common classes.

    Args:
        subset_path: Path to the US-018 features parquet.

    Returns:
        Tuple ``(df, feature_cols)`` with the filtered DataFrame (with a
        ``common_class`` column) and the ordered list of feature columns.
    """
    df = pl.read_parquet(subset_path)
    df = df.filter(pl.col("class_id").is_in(list(PASTIS_ID_TO_COMMON.keys())))
    df = df.with_columns(
        pl.col("class_id").replace_strict(PASTIS_ID_TO_COMMON, default=None).alias("common_class")
    )
    feature_cols = _feature_columns(df)
    logger.info(
        "pastis_common_loaded",
        n_parcels=df.height,
        n_features=len(feature_cols),
        per_class=df.group_by("common_class").len().sort("len", descending=True).to_dicts(),
    )
    return df, feature_cols


def _feature_columns(df: pl.DataFrame) -> list[str]:
    """Returns the numeric feature columns (excludes metadata)."""
    meta = {
        "parcel_id",
        "year",
        "class_id",
        "class_name",
        "common_class",
        "patch_id",
        "instance_id",
        "fold",
        "n_pixels",
        "area_m2",
        "geometry",
    }
    return [
        c
        for c in df.columns
        if c not in meta
        and not c.endswith(("_right", "_left", "_x", "_y"))
        and df.schema[c].is_numeric()
    ]


# ---------------------------------------------------------------------------
# Load + feature extraction of BreizhCrops (test).
# ---------------------------------------------------------------------------


def _sample_breizhcrops_parcels(
    n_target: int,
    *,
    regions: tuple[str, ...] = ("frh04", "frh01"),
    seed: int = 42,
) -> pl.DataFrame:
    """Samples BreizhCrops parcels stratified by common class.

    Args:
        n_target: Total target number of parcels to sample.
        regions: BreizhCrops regions to use.
        seed: Sampling seed.

    Returns:
        DataFrame with ``parcel_id, region, class_id, class_name, common_class``
        of the selected parcels (those falling in the 7 common classes).
    """
    frames: list[pl.DataFrame] = []
    for region in regions:
        idx = breizhcrops_parcel_index(region=region, year=2017, level="L2A")
        if idx.height == 0:
            logger.warning("breizhcrops_region_empty", region=region)
            continue
        frames.append(idx)
    if not frames:
        raise RuntimeError(
            "BreizhCrops not available on disk. Run scripts/download_breizhcrops.sh."
        )

    index = pl.concat(frames, how="vertical_relaxed")
    index = index.filter(pl.col("class_name").is_in(list(BREIZ_NAME_TO_COMMON.keys())))
    index = index.with_columns(
        pl.col("class_name")
        .replace_strict(BREIZ_NAME_TO_COMMON, default=None)
        .alias("common_class")
    )

    # Stratified sampling: uniform quota per common class, until exhausted.
    n_classes = index.get_column("common_class").n_unique()
    per_class = max(1, n_target // n_classes)
    sampled: list[pl.DataFrame] = []
    for _cls, sub in index.group_by("common_class"):
        sampled.append(sub.sample(n=min(per_class, sub.height), seed=seed, shuffle=True))
    out = pl.concat(sampled, how="vertical_relaxed")
    logger.info(
        "breizhcrops_sampled",
        n_target=n_target,
        n_sampled=out.height,
        per_class=out.group_by("common_class").len().sort("len", descending=True).to_dicts(),
    )
    return out


def _extract_breizhcrops_features(
    sampled_index: pl.DataFrame,
    *,
    seed: int = 42,
) -> pl.DataFrame:
    """Loads the series of the sampled parcels and extracts the 185 features.

    Loads the series per region (subset by sampled ``parcel_id``) and applies
    the adapter. The sampling of ``breizhcrops_pixel_series`` is positional,
    so we load the WHOLE region and then filter by the chosen ids
    to guarantee that we extract exactly the stratified parcels.

    Args:
        sampled_index: Output of :func:`_sample_breizhcrops_parcels`.
        seed: Seed (reserved; the filtering is deterministic by id).

    Returns:
        Features DataFrame (185 cols) + ``parcel_id, year, class_id,
        class_name, common_class``.
    """
    feature_frames: list[pl.DataFrame] = []
    for region, sub in sampled_index.group_by("region"):
        region_name = region[0] if isinstance(region, tuple) else region
        wanted_ids = set(sub.get_column("parcel_id").to_list())
        common_map = {row["parcel_id"]: row["common_class"] for row in sub.iter_rows(named=True)}
        logger.info(
            "breizhcrops_extracting_region",
            region=region_name,
            n_parcels=len(wanted_ids),
        )
        # Extract ONLY the sampled parcels (efficient H5 read, without
        # expanding hundreds of thousands of parcels of the full region).
        series = breizhcrops_pixel_series(
            region=str(region_name),
            year=2017,
            level="L2A",
            only_parcel_ids=wanted_ids,
        )
        feats = build_breizhcrops_features(series)
        if feats.height == 0:
            continue
        feats = feats.with_columns(
            pl.col("parcel_id").replace_strict(common_map, default=None).alias("common_class")
        )
        feature_frames.append(feats)

    if not feature_frames:
        raise RuntimeError("No features were extracted from any BreizhCrops parcel.")
    out = pl.concat(feature_frames, how="vertical_relaxed")
    logger.info("breizhcrops_features_extracted", n_parcels=out.height)
    return out


# ---------------------------------------------------------------------------
# Direct transfer.
# ---------------------------------------------------------------------------


def _align_features(pastis_cols: list[str], breiz_cols: list[str]) -> list[str]:
    """Returns the features present in BOTH datasets, in stable order."""
    breiz_set = set(breiz_cols)
    shared = [c for c in pastis_cols if c in breiz_set]
    return shared


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Extracts a float64 matrix with median imputation of non-finite values."""
    mat = df.select(cols).to_numpy().astype(np.float64)
    finite = np.where(np.isfinite(mat), mat, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    bad = ~np.isfinite(mat)
    if bad.any():
        idx = np.where(bad)
        mat[idx] = np.take(medians, idx[1])
    return mat


def run_direct_transfer(
    pastis_df: pl.DataFrame,
    breiz_df: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, float], dict[str, float], int]:
    """Trains XGBoost on PASTIS and evaluates on BreizhCrops without retraining.

    Args:
        pastis_df: PASTIS DataFrame with ``common_class`` + features.
        breiz_df: BreizhCrops DataFrame with ``common_class`` + features.
        feature_cols: Shared features (same order in both).

    Returns:
        Tuple ``(per_class_f1, summary, n_shared_features)`` where
        ``per_class_f1`` maps each common class to its F1, ``summary`` has
        ``f1_macro``, ``accuracy`` and ``f1_macro_no_sunflower``.
    """
    encoder = LabelEncoder().fit(list(COMMON_CLASSES))

    x_train = _matrix(pastis_df, feature_cols)
    y_train = encoder.transform(pastis_df.get_column("common_class").to_list())

    x_test = _matrix(breiz_df, feature_cols)
    y_test = encoder.transform(breiz_df.get_column("common_class").to_list())

    params = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": len(encoder.classes_),
        "random_state": 42,
    }
    model = build_estimator("xgb", params)
    # sample_weight inverse to frequency (highly imbalanced classes in PASTIS).
    classes, counts = np.unique(y_train, return_counts=True)
    w_per_class = {
        int(c): y_train.size / (classes.size * cnt) for c, cnt in zip(classes, counts, strict=True)
    }
    sample_weight = np.array([w_per_class[int(c)] for c in y_train], dtype=np.float64)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(x_test)

    labels = list(range(len(encoder.classes_)))
    f1_per = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {encoder.classes_[i]: float(f1_per[i]) for i in range(len(encoder.classes_))}
    f1_macro = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    accuracy = float((y_pred == y_test).mean())

    # F1-macro without sunflower (almost nonexistent in BreizhCrops): a metric more
    # representative of the real transfer signal.
    no_sf = [c for c in COMMON_CLASSES if c != "sunflower"]
    no_sf_idx = encoder.transform(no_sf)
    f1_macro_no_sf = float(
        f1_score(y_test, y_pred, labels=list(no_sf_idx), average="macro", zero_division=0)
    )

    summary = {
        "f1_macro": f1_macro,
        "f1_macro_no_sunflower": f1_macro_no_sf,
        "accuracy": accuracy,
    }
    logger.info(
        "direct_transfer_done",
        f1_macro=round(f1_macro, 4),
        f1_macro_no_sunflower=round(f1_macro_no_sf, 4),
        accuracy=round(accuracy, 4),
        n_train=int(y_train.size),
        n_test=int(y_test.size),
    )
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=list(encoder.classes_), zero_division=0
    )
    logger.info("direct_transfer_report", report="\n" + report)
    return per_class_f1, summary, len(feature_cols)


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------


def _persist(
    per_class_f1: dict[str, float],
    summary: dict[str, float],
    *,
    n_pastis: int,
    n_breiz: int,
    n_features: int,
    output: Path,
) -> None:
    """Persists the transfer report to parquet (one row per class + macro)."""
    rows: list[dict[str, object]] = []
    for cls in COMMON_CLASSES:
        rows.append(
            {
                "approach": "direct_transfer",
                "scope": "per_class",
                "label": cls,
                "f1": round(per_class_f1.get(cls, 0.0), 4),
                "n_pastis_train": n_pastis,
                "n_breiz_test": n_breiz,
                "n_shared_features": n_features,
            }
        )
    for metric, value in summary.items():
        rows.append(
            {
                "approach": "direct_transfer",
                "scope": "summary",
                "label": metric,
                "f1": round(value, 4),
                "n_pastis_train": n_pastis,
                "n_breiz_test": n_breiz,
                "n_shared_features": n_features,
            }
        )
    table = pl.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output)
    logger.info("transfer_report_persisted", path=str(output), n_rows=table.height)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-breiz", type=int, default=2500, help="Parcelas BreizhCrops objetivo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--try-pheno-text",
        action="store_true",
        help="Intenta la variante pheno-text (requiere credenciales Gemini).",
    )
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args(argv)

    if not _PASTIS_SUBSET.exists():
        logger.error("pastis_subset_missing", path=str(_PASTIS_SUBSET))
        return 1

    pastis_df, pastis_cols = _load_pastis_common(_PASTIS_SUBSET)

    sampled = _sample_breizhcrops_parcels(args.n_breiz, seed=args.seed)
    breiz_df = _extract_breizhcrops_features(sampled, seed=args.seed)
    breiz_df = breiz_df.filter(pl.col("common_class").is_not_null())

    shared = _align_features(pastis_cols, _feature_columns(breiz_df))
    logger.info("shared_features", n=len(shared))

    per_class_f1, summary, n_feat = run_direct_transfer(pastis_df, breiz_df, shared)

    if args.try_pheno_text:
        logger.warning(
            "pheno_text_skipped",
            reason=(
                "El bloque pheno-text (ml.features.phenology_description) genera "
                "descripciones con Gemini 3.5 Flash via red + text-encoder. Generar "
                "~2500 descripciones BreizhCrops requiere credenciales Gemini y costo. "
                "Sin GEMINI_API_KEY/GOOGLE_API_KEY disponible, no se ejecuta en esta "
                "PoC; transfer directo es el resultado principal."
            ),
        )

    _persist(
        per_class_f1,
        summary,
        n_pastis=pastis_df.height,
        n_breiz=breiz_df.height,
        n_features=n_feat,
        output=args.output,
    )

    print("\n=== TRANSFER DIRECTO PASTIS-R -> BreizhCrops ===")
    print(f"Parcelas PASTIS (train): {pastis_df.height}")
    print(f"Parcelas BreizhCrops (test): {breiz_df.height}")
    print(f"Features compartidas: {n_feat}")
    print(f"F1-macro (7 clases): {summary['f1_macro']:.4f}")
    print(f"F1-macro (6 clases, sin sunflower): {summary['f1_macro_no_sunflower']:.4f}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print("F1 por clase:")
    for cls in COMMON_CLASSES:
        print(f"  {cls:12s} {per_class_f1.get(cls, 0.0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
