"""XGBoost baseline: 18 flat PASTIS-R classes vs 6 HCAT Level-1 groups.

Permanent operational tool. Re-runs the BEST baseline setup (XGBoost, 5-fold
spatial CV with a 1 km anti-leakage buffer) over the 85951 PASTIS-R parcels in
TWO label schemes and compares them:

1. ``flat18``    : the original 18 flat PASTIS-R classes.
2. ``hcat_l1_6`` : the 18 classes merged into 6 HCAT Level-1 super-classes
   (CEREALS, OILSEEDS, ROOT_CROPS, LEGUMES, PERMANENT_WOODY, OTHER).

The feature vector is the best found in the ablation: the 185 base features
(spectral + temporal indices + phenology) plus the AlphaEarth Foundations
embeddings from 2018 (cols ``ae18_NN``) and 2019 (cols ``ae19_NN``), joined by
``parcel_id``.

Apples-to-apples design: both schemes operate over EXACTLY the same rows in the
same order, so they share the same cached spatial splits (the cache key is
``n_rows + k + buffer + seed``). The only difference between the two runs is the
remapping of ``class_id``.

The 6-group mapping and their HCAT codes live in
``data/reference/pastis_class_mapping.json`` (grouping ``hcat_l1_6``) and are
loaded via ``ml.ingest.pastis_loader.PASTIS_R_GROUPINGS``; this script reuses
them, it does not redefine them.

Usage:
    python scripts/baseline_18_vs_grouped6.py                  # FULL 85951
    python scripts/baseline_18_vs_grouped6.py --max-samples 300  # validation

Artifacts in ``reports/baseline/grouped_vs_flat/``:
    - comparison.parquet        : F1-macro and other metrics per scheme.
    - per_class_f1_flat18.parquet     : per-class F1 (18 classes).
    - per_class_f1_hcat_l1_6.parquet  : per-HCAT-group F1 (6 groups).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score

from ml.ingest.pastis_loader import PASTIS_R_CLASSES, PASTIS_R_GROUPINGS
from ml.train.baseline import (
    build_estimator,
    compute_baseline_metrics,
    evaluate_with_spatial_cv,
)

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO_ROOT / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet"
_AE18 = _REPO_ROOT / "data" / "cache" / "gee" / "alphaearth_parcels_parcels_2018_85951.parquet"
_AE19 = (
    _REPO_ROOT / "data" / "cache" / "gee" / "alphaearth_parcels_pastis_parcels_2019_85951.parquet"
)
_OUT_DIR = _REPO_ROOT / "reports" / "baseline" / "grouped_vs_flat"

# Canonical HCAT Level-1 grouping (6 groups). We reuse the one that already
# lives in data/reference/pastis_class_mapping.json; here we only reference the
# name and document the HCAT codes for defensibility. Do NOT redefine the map:
# it is the single source of truth loaded via PASTIS_R_GROUPINGS.
_HCAT_GROUPING = "hcat_l1_6"

# HCAT v3 Level-1 codes for each fusion (Russwurm et al. 2018 method /
# H2Crop arXiv:2506.06155). Printed for the defense of the grouping.
_HCAT_CODES: dict[str, str] = {
    "CEREALS": "3300000000 cereals (wheat 3301, barley 3302, maize 3303, "
    "triticale 3304, sorghum 3305, mixed cereal 3300010000)",
    "OILSEEDS": "3400000000 oilseed crops (rapeseed 3401010101, sunflower 3401050000)",
    "ROOT_CROPS": "3500000000 root/tuber crops (sugar beet 3500010000, potato 3500020000)",
    "LEGUMES": "3600000000 leguminous crops (leguminous fodder 3600060000, soybean 3601000000)",
    "PERMANENT_WOODY": "3900000000 permanent/woody crops "
    "(grapevine 3901000000, orchard 3902000000)",
    "OTHER": "3000000000 raiz / horticultura mixta (grassland-meadow 3370000000, "
    "fruits/vegetables/flowers 3800000000)",
}


def _load_features(max_samples: int | None) -> pl.DataFrame:
    """Load the 185-feature fixture and join the AlphaEarth embeddings.

    Joins AlphaEarth 2018 (``ae18_NN``) and 2019 (``ae19_NN``) by ``parcel_id``.
    If ``max_samples`` is specified, it subsamples deterministically BEFORE the
    join (the quick validation does not need the 85951 rows).

    Args:
        max_samples: If not ``None``, takes the first ``max_samples`` rows after
            a fixed-seed shuffle.

    Returns:
        Polars DataFrame with the 185 base features + 128 AlphaEarth dims
        (64 from 2018 + 64 from 2019) + metadata (``parcel_id``, ``class_id``,
        ``patch_id``, etc.).
    """
    base = pl.read_parquet(_FIXTURE)
    if max_samples is not None and max_samples < base.height:
        base = base.sample(n=max_samples, seed=42, shuffle=True)

    ae18 = pl.read_parquet(_AE18).drop("year")
    ae18 = ae18.rename(
        {c: f"ae18_{c.removeprefix('dim_')}" for c in ae18.columns if c != "parcel_id"}
    )
    ae19 = pl.read_parquet(_AE19).drop("year")
    ae19 = ae19.rename(
        {c: f"ae19_{c.removeprefix('dim_')}" for c in ae19.columns if c != "parcel_id"}
    )

    merged = base.join(ae18, on="parcel_id", how="left").join(ae19, on="parcel_id", how="left")
    n_ae18 = sum(c.startswith("ae18_") for c in merged.columns)
    n_ae19 = sum(c.startswith("ae19_") for c in merged.columns)
    logger.info(
        "features_loaded",
        n_rows=merged.height,
        n_cols=merged.width,
        n_ae18=n_ae18,
        n_ae19=n_ae19,
    )
    if n_ae18 != 64 or n_ae19 != 64:
        raise ValueError(
            f"Expected 64 AlphaEarth dims per year; got {n_ae18} (2018) "
            f"and {n_ae19} (2019). Check the AlphaEarth caches."
        )
    return merged


def _remap_to_hcat(df: pl.DataFrame) -> pl.DataFrame:
    """Remap ``class_id`` (1..18) to integer IDs of the 6 HCAT L1 groups.

    Uses the canonical grouping ``hcat_l1_6`` from :data:`PASTIS_R_GROUPINGS`. It
    assigns a stable and ordered integer to each group name so XGBoost receives
    clean labels.

    Args:
        df: DataFrame with the ``class_id`` column (18 flat classes).

    Returns:
        Copy of the DataFrame with ``class_id`` replaced by the HCAT group ID
        (contiguous integers in alphabetical group order).
    """
    grouping = PASTIS_R_GROUPINGS[_HCAT_GROUPING]
    group_names = sorted(set(grouping.values()))
    name_to_id = {name: i + 1 for i, name in enumerate(group_names)}
    class_to_group_id = {cid: name_to_id[grp] for cid, grp in grouping.items()}

    return df.with_columns(pl.col("class_id").replace_strict(class_to_group_id).alias("class_id"))


def _group_id_to_name() -> dict[int, str]:
    """Map ``{group_id: group_name}`` consistent with :func:`_remap_to_hcat`."""
    group_names = sorted(set(PASTIS_R_GROUPINGS[_HCAT_GROUPING].values()))
    return {i + 1: name for i, name in enumerate(group_names)}


def _run_scheme(
    df: pl.DataFrame,
    *,
    scheme: str,
    k_folds: int,
    buffer_km: float,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Run the spatial CV for a label scheme and build the per-class F1.

    Calls :func:`evaluate_with_spatial_cv` (the same engine used by
    ``train_one_model``) to obtain the out-of-fold predictions, computes the
    aggregated metrics with :func:`compute_baseline_metrics` and the per-class
    F1 from the same OOF vectors.

    Args:
        df: Feature DataFrame with ``class_id`` already in the desired scheme.
        scheme: Scheme label (``"flat18"`` or ``"hcat_l1_6"``).
        k_folds: Number of spatial CV folds.
        buffer_km: Anti-leakage buffer in km.

    Returns:
        Tuple ``(metrics, per_class_df)`` where ``metrics`` are the five OOF
        metrics and ``per_class_df`` is a DataFrame with ``label_id``,
        ``label_name``, ``f1`` and ``support`` per class.
    """
    from ml.train.baseline import _base_params, _encode_labels  # internal reuse

    # We do not set `num_class`: XGBClassifier (sklearn API) infers it per fold.
    # Forcing it breaks the folds where the train does not contain all 18 classes
    # (artifact of small subsamples; in the full run each fold has all of them).
    params = _base_params("xgb")

    def factory():  # type: ignore[no-untyped-def]
        return build_estimator("xgb", params)

    _, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(
        df, factory, k_folds=k_folds, buffer_km=buffer_km, random_state=42
    )

    encoder, _ = _encode_labels(df)
    labels = list(range(len(encoder.classes_)))
    metrics = compute_baseline_metrics(y_true_oof, y_pred_oof, labels=labels)

    f1_per = f1_score(y_true_oof, y_pred_oof, labels=labels, average=None, zero_division=0)
    support = np.bincount(y_true_oof.astype(np.int64), minlength=len(labels))

    if scheme == "flat18":
        names = [PASTIS_R_CLASSES[int(c)] for c in encoder.classes_]
    else:
        gid_to_name = _group_id_to_name()
        names = [gid_to_name[int(c)] for c in encoder.classes_]

    per_class = pl.DataFrame(
        {
            "scheme": [scheme] * len(labels),
            "label_id": [int(c) for c in encoder.classes_],
            "label_name": names,
            "f1": [float(x) for x in f1_per],
            "support": [int(s) for s in support],
        }
    ).sort("f1")

    logger.info(
        "scheme_done",
        scheme=scheme,
        f1_macro=round(metrics["f1_macro"], 4),
        n_classes=len(labels),
    )
    return metrics, per_class


def _print_per_class(title: str, frame: pl.DataFrame) -> None:
    """Print a per-class F1 in pure ASCII (cp1252-safe console)."""
    print(f"\n{title}")
    print(f"  {'id':>3}  {'clase/grupo':32s}  {'F1':>7}  {'support':>8}")
    for row in frame.iter_rows(named=True):
        print(
            f"  {row['label_id']:>3}  {row['label_name']:32s}  "
            f"{row['f1']:>7.4f}  {row['support']:>8d}"
        )


def main() -> None:
    """Entry point: run both schemes, persist and report."""
    # The Windows console (cp1252) does not encode the Polars Unicode borders nor
    # accents; we force UTF-8 on stdout so the report does not crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Submuestra N parcelas para validacion rapida (default: full 85951).",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--buffer-km", type=float, default=1.0)
    args = parser.parse_args()

    df = _load_features(args.max_samples)

    # Scheme 1: 18 flat classes.
    metrics_18, per_class_18 = _run_scheme(
        df, scheme="flat18", k_folds=args.k_folds, buffer_km=args.buffer_km
    )

    # Scheme 2: 6 HCAT L1 groups (same rows -> same cached folds).
    df_grouped = _remap_to_hcat(df)
    metrics_6, per_class_6 = _run_scheme(
        df_grouped, scheme=_HCAT_GROUPING, k_folds=args.k_folds, buffer_km=args.buffer_km
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pl.DataFrame(
        [
            {"scheme": "flat18", "n_classes": 18, **metrics_18},
            {"scheme": _HCAT_GROUPING, "n_classes": 6, **metrics_6},
        ]
    )
    comparison.write_parquet(_OUT_DIR / "comparison.parquet")
    per_class_18.write_parquet(_OUT_DIR / "per_class_f1_flat18.parquet")
    per_class_6.write_parquet(_OUT_DIR / f"per_class_f1_{_HCAT_GROUPING}.parquet")

    delta = metrics_6["f1_macro"] - metrics_18["f1_macro"]

    print("\n=== Baseline XGBoost: 18 clases vs 6 grupos HCAT L1 ===")
    print(f"n_parcelas={df.height} | spatial CV {args.k_folds}-fold | buffer {args.buffer_km} km")
    print(f"\nF1-macro 18 clases planas : {metrics_18['f1_macro']:.4f}")
    print(f"F1-macro 6 grupos HCAT L1 : {metrics_6['f1_macro']:.4f}")
    print(f"Delta (6 grupos - 18)     : {delta:+.4f}")
    print("\nMetricas completas por esquema:")
    for row in comparison.iter_rows(named=True):
        print(
            f"  {row['scheme']:10s} n_classes={row['n_classes']:2d} "
            f"f1_macro={row['f1_macro']:.4f} f1_weighted={row['f1_weighted']:.4f} "
            f"miou={row['miou']:.4f} accuracy={row['accuracy']:.4f} "
            f"cohen_kappa={row['cohen_kappa']:.4f}"
        )
    _print_per_class("F1 por clase (18 clases planas, peor a mejor):", per_class_18)
    _print_per_class("F1 por grupo HCAT L1 (6 grupos):", per_class_6.sort("label_id"))
    print("\nCodigos HCAT de cada grupo:")
    for name, code in _HCAT_CODES.items():
        print(f"  {name:16s} {code}")
    print(f"\nArtefactos en {_OUT_DIR}")


if __name__ == "__main__":
    main()
