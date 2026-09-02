"""Measure the incremental contribution of AlphaEarth (2018/2019) + ERA5/SRTM/S1 to the baseline.

Context: the tabular baseline (F1-macro 0.4094) runs over 185 spectral/phenology
features WITHOUT AlphaEarth nor the patch-level blocks (ERA5, SRTM, S1), which
exist as real data but were never joined to the training subset (see
docs/audit/us-023-preview-v2-audit.md). This script joins each family in memory
and runs the ablation with spatial CV (1 km buffer) reusing
``run_feature_ablation``, to measure how much each block raises the F1. It does
NOT persist any parquet: it only prints the table to decide whether to
materialize.

Join keys:
- AlphaEarth (2018, 2019): by ``parcel_id`` (PASTIS string, 100% overlap).
- ERA5 / SRTM / S1: by ``patch_id`` (patch level ~1 km^2, propagates to parcels).

Usage:
    python scripts/measure_alphaearth_multisensor_ablation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.eval.feature_ablation import run_feature_ablation  # noqa: E402

log = structlog.get_logger(__name__)

SUBSET = REPO_ROOT / "data/test_fixtures/feature_selection_parcels_subset.parquet"
AE19 = REPO_ROOT / "data/cache/gee/alphaearth_parcels_pastis_parcels_2019_85951.parquet"
AE18 = REPO_ROOT / "data/cache/gee/alphaearth_parcels_parcels_2018_85951.parquet"
ERA5 = REPO_ROOT / "data/cache/gee/era5_monthly_pastis_fr_full_2019_C.parquet"
SRTM = REPO_ROOT / "data/cache/gee/srtm_pastis_fr_full.parquet"
S1 = REPO_ROOT / "data/cache/gee/s1_pastis_fr_full_2019_both_lee_7x7_dB_enriched.parquet"

#: Metadata to NOT carry over from the patch-level blocks (avoids leakage / duplicates).
_META_DROP = (
    "year",
    "patch_id",
    "class_id",
    "class_name",
    "fold",
    "n_pixels",
    "instance_id",
    "area_m2",
)


def _ae_block(path: Path, prefix: str) -> pl.DataFrame:
    """Load an AlphaEarth parquet and rename ``dim_NN -> {prefix}NN`` (parcel-level)."""
    df = pl.read_parquet(path)
    dims = [c for c in df.columns if c.startswith("dim_")]
    ren = {c: f"{prefix}{c.split('_')[1]}" for c in dims}
    return df.select(["parcel_id", *dims]).rename(ren)


def _patch_block(path: Path, keep_prefixes: tuple[str, ...]) -> pl.DataFrame:
    """Load a patch-level block and keep only ``patch_id`` + numeric features."""
    df = pl.read_parquet(path)
    # parcel_id here IS the integer patch_id -> rename to patch_id for the join
    df = df.rename({"parcel_id": "patch_id"})
    feat = [c for c in df.columns if c.startswith(keep_prefixes)]
    return df.select(["patch_id", *feat])


def main(max_samples: int | None) -> int:
    sub = pl.read_parquet(SUBSET)
    base_feats = [c for c in sub.columns if c not in (*_META_DROP, "parcel_id")]
    log.info("subset_loaded", rows=sub.height, base_features=len(base_feats))

    # --- AlphaEarth parcel-level (join by parcel_id) ---
    ae19 = _ae_block(AE19, "ae19_")
    ae18 = _ae_block(AE18, "ae18_")
    df = sub.join(ae19, on="parcel_id", how="left").join(ae18, on="parcel_id", how="left")
    ae19_cols = [c for c in df.columns if c.startswith("ae19_")]
    ae18_cols = [c for c in df.columns if c.startswith("ae18_")]

    # --- Patch-level blocks (join by patch_id) ---
    era5 = _patch_block(ERA5, ("era5",))
    srtm = _patch_block(SRTM, ("srtm",))
    s1 = _patch_block(S1, ("s1_",))
    df = (
        df.join(era5, on="patch_id", how="left")
        .join(srtm, on="patch_id", how="left")
        .join(s1, on="patch_id", how="left")
    )
    era5_cols = [c for c in df.columns if c.startswith("era5")]
    srtm_cols = [c for c in df.columns if c.startswith("srtm")]
    s1_cols = [c for c in df.columns if c.startswith("s1_")]

    log.info(
        "blocks_joined",
        ae19=len(ae19_cols),
        ae18=len(ae18_cols),
        era5=len(era5_cols),
        srtm=len(srtm_cols),
        s1=len(s1_cols),
        s1_null_parcels=int(df.select(pl.col(s1_cols[0]).is_null().sum()).item()) if s1_cols else 0,
        total_cols=df.width,
    )

    # --- incremental feature_sets (each MUST include the 185 base except the baseline) ---
    feature_sets = {
        "full": tuple(base_feats),  # 0.4094 expected (replica)
        "base_plus_ae19": tuple(base_feats + ae19_cols),
        "base_plus_ae18": tuple(base_feats + ae18_cols),
        "base_plus_ae18_ae19": tuple(base_feats + ae19_cols + ae18_cols),
        "base_plus_ae19_era5_srtm": tuple(base_feats + ae19_cols + era5_cols + srtm_cols),
        "base_plus_all": tuple(
            base_feats + ae19_cols + ae18_cols + era5_cols + srtm_cols + s1_cols
        ),
        "ae19_only": tuple(ae19_cols),
        "ae18_only": tuple(ae18_cols),
    }

    log.info(
        "running_ablation",
        n_sets=len(feature_sets),
        cv="spatial 5-fold buffer 1km",
        max_samples=max_samples,
    )
    results = run_feature_ablation(
        df=df, feature_sets=feature_sets, models=("xgb",), max_samples=max_samples
    )

    print("\n=== ABLATION AlphaEarth + multisensor (XGB, spatial CV 5-fold buffer 1km) ===")
    print(f"{'feature_set':28s} {'n_feat':>7s} {'f1_macro':>9s} {'delta_vs_full':>14s}")
    by = {r.feature_set: r for r in results}
    for name in feature_sets:
        r = by.get(name)
        if r is None:
            print(f"{name:28s} {'--':>7s} {'SKIP':>9s}")
            continue
        f1 = f"{r.f1_macro:.4f}" if r.f1_macro is not None else "NaN"
        dv = f"{r.delta_vs_full:+.4f}" if r.delta_vs_full is not None else "ref"
        print(f"{name:28s} {r.n_features:>7d} {f1:>9s} {dv:>14s}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=15000,
        help="Subsample uniforme determinista para lectura rapida (default 15000; 0 = dataset completo).",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.max_samples if args.max_samples > 0 else None))
