"""Pilot: does the FULL ensemble transfer better than xgb-alphaearth alone?

Motivation (Arthur's hypothesis, 2026-06-25)
--------------------------------------------
The champion is the Stacking-5 ensemble (``tsvit-pheno``, ``utae``,
``xgb-alphaearth``, ``farslip-ft18``, ``farslip-zeroshot``). Every transfer
experiment run so far on a NEW dataset (EuroCropsML LV->EE, WorldCereal) used
ONLY the ``xgb-alphaearth`` member, because the other four need an input the new
datasets did not carry aligned: the dense temporal members (TSViT-pheno, U-TAE)
need the Sentinel-2 multi-date RASTER, and FarSLIP additionally needs a textual
caption. The hypothesis is that the champion underperformed in transfer not
because transfer is hard but because we fed it one leg of five.

This module is the CHEAP, MEASURABLE first step before paying for the full dense
pipeline (which has a real format mismatch -- TSViT/U-TAE expect a spatial patch
``(B, T, C, H, W)`` of 10 PASTIS bands, while EuroCropsML ships a per-parcel
series ``(T, 13)`` with no spatial axis; bridging that is non-trivial and is
deferred until this pilot justifies it). It answers the strictly simpler
question first:

    Over the SAME source->target transfer split, does adding the Sentinel-2
    temporal signal to AlphaEarth-annual lift the transfer F1, and by how much?

If even the lightweight temporal vector (which carries the SAME phenological
evidence TSViT/U-TAE would, just pixel-reduced) lifts transfer, that is the
green light to invest in the dense adapter + Gemini captions for the full
ensemble. If it does not, the bottleneck is elsewhere and the dense pipeline
would not have helped either -- an honest negative we want to know cheaply.

What it does
------------
1. Reuses :func:`ml.transfer.temporal_features.build_aligned_dataset` to pair,
   for the SAME EuroCropsML parcels, the AlphaEarth annual embedding (64-dim) and
   the Sentinel-2 temporal vector (99-dim) plus the HCAT leaf label and lon/lat.
2. Builds a source->target SPATIAL transfer split (train on the source region,
   test on the target region) so the measurement is a genuine geographic
   transfer, not an in-distribution split.
3. Trains the champion XGBoost recipe on three feature views on the IDENTICAL
   train/test parcels and reports per-class + macro F1 on the target:
     - ``annual``  : AlphaEarth 64-dim only (the status quo transfer leg),
     - ``temporal``: Sentinel-2 99-dim only (the new leg in isolation),
     - ``fusion``  : annual ++ temporal (annual with the Sentinel signal added).

Honesty
-------
- No number is fabricated. Parcels without an ``.npz`` are dropped and counted.
- It is a PILOT: a small parcel cap by default so it runs in minutes on a laptop
  GPU. The verdict is whatever the numbers say; a null result is reported as-is.
- This trains XGBoost on the temporal vector -- it does NOT re-train the champion
  members. The champion members (TSViT/U-TAE/FarSLIP) are NOT touched here; this
  pilot only decides whether the Sentinel signal is worth wiring into them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score, precision_recall_fscore_support

from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator
from ml.transfer.temporal_features import _AlignedDataset, build_aligned_dataset

logger = structlog.get_logger(__name__)

__all__ = [
    "EnsembleTLPilotResult",
    "run_ensemble_tl_pilot",
    "save_outputs",
]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_OUT_DIR: Path = _REPO_ROOT / "data" / "transfer" / "ensemble_tl_pilot"

#: Default region mapping for the pilot: train on the source, test on the target.
_DEFAULT_SOURCE: str = "latvia"
_DEFAULT_TARGET: str = "estonia"

#: Default per-region parcel cap so the pilot is a few-minutes laptop run. The
#: cap is applied per region BEFORE the npz pass; raise it (or set None) to scale.
_DEFAULT_MAX_PARCELS_PER_REGION: int = 4000

#: Minimum target-test support for a leaf to enter the per-class verdict (the
#: macro F1 is still computed over the full shared label space).
_MIN_TEST_SUPPORT: int = 5

_RANDOM_STATE: int = 42


@dataclass
class EnsembleTLPilotResult:
    """Transfer F1 of annual / temporal / fusion on the SAME source->target split."""

    per_class: pl.DataFrame
    summary: dict[str, object]


def _fit_score(
    feats_tr: np.ndarray,
    feats_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train the champion XGBoost once and score per-class + macro F1 on the test.

    Args:
        feats_tr: Source-region train features ``(n_tr, d)``.
        feats_te: Target-region test features ``(n_te, d)``.
        y_tr: Source train labels ``(n_tr,)`` as contiguous class ids.
        y_te: Target test labels ``(n_te,)``.
        n_classes: Size of the shared label space.
        seed: RNG seed for the booster.

    Returns:
        Tuple ``(f1_per_class, support, macro_f1)`` on the target test set.
    """
    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    model = build_estimator("xgb", params)
    model.fit(feats_tr, y_tr)
    pred = model.predict(feats_te)
    _p, _r, f1, sup = precision_recall_fscore_support(
        y_te, pred, labels=np.arange(n_classes), average=None, zero_division=0
    )
    macro = float(f1_score(y_te, pred, average="macro"))
    return f1, sup.astype(int), macro


def run_ensemble_tl_pilot(
    *,
    source: str = _DEFAULT_SOURCE,
    target: str = _DEFAULT_TARGET,
    max_parcels_per_region: int | None = _DEFAULT_MAX_PARCELS_PER_REGION,
    seed: int = _RANDOM_STATE,
) -> EnsembleTLPilotResult:
    """Measure whether the Sentinel temporal leg lifts source->target transfer.

    Trains the champion XGBoost recipe on three feature views (annual / temporal /
    fusion) over the IDENTICAL source-train / target-test parcels and labels, and
    reports the per-class and macro F1 of each on the target region.

    Args:
        source: EuroCropsML region key trained on (e.g. ``"latvia"``).
        target: EuroCropsML region key tested on (e.g. ``"estonia"``).
        max_parcels_per_region: Per-region cap before the npz pass; ``None`` uses
            every parcel with an available series.
        seed: RNG seed for the boosters.

    Returns:
        An :class:`EnsembleTLPilotResult`.

    Raises:
        FileNotFoundError: if a region parquet or the preprocess dir is missing.
        ValueError: if the two regions share no leaf class.
    """
    # Build the aligned (annual + temporal + leaf + lon/lat) view per region. The
    # builder caps and reads the npz; pooling both regions here would lose the
    # region tag, so build each separately and tag.
    ds_src = build_aligned_dataset(regions=(source,), max_parcels=max_parcels_per_region)
    ds_tgt = build_aligned_dataset(regions=(target,), max_parcels=max_parcels_per_region)

    shared = sorted(set(ds_src.leaf.tolist()) & set(ds_tgt.leaf.tolist()))
    if not shared:
        raise ValueError(f"source={source!r} and target={target!r} share no leaf class.")
    class_to_id = {c: i for i, c in enumerate(shared)}
    keep = set(shared)

    def _filter(ds: _AlignedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.array([leaf in keep for leaf in ds.leaf], dtype=bool)
        y = np.array([class_to_id[c] for c in ds.leaf[mask]], dtype=np.int64)
        return ds.annual[mask], ds.temporal[mask], y

    a_src, t_src, y_src = _filter(ds_src)
    a_tgt, t_tgt, y_tgt = _filter(ds_tgt)
    n_classes = len(shared)

    fus_src = np.concatenate([a_src, t_src], axis=1)
    fus_tgt = np.concatenate([a_tgt, t_tgt], axis=1)

    f1_annual, sup, macro_annual = _fit_score(a_src, a_tgt, y_src, y_tgt, n_classes, seed)
    f1_temporal, _, macro_temporal = _fit_score(t_src, t_tgt, y_src, y_tgt, n_classes, seed)
    f1_fusion, _, macro_fusion = _fit_score(fus_src, fus_tgt, y_src, y_tgt, n_classes, seed)

    per_class = pl.DataFrame(
        {
            "leaf": shared,
            "f1_annual": f1_annual.tolist(),
            "f1_temporal": f1_temporal.tolist(),
            "f1_fusion": f1_fusion.tolist(),
            "delta_fusion_vs_annual": (f1_fusion - f1_annual).tolist(),
            "target_support": sup.tolist(),
        }
    ).sort("delta_fusion_vs_annual", descending=True)

    scored = per_class.filter(pl.col("target_support") >= _MIN_TEST_SUPPORT)
    n_improved = int(scored.filter(pl.col("delta_fusion_vs_annual") > 1e-9).height)
    n_worsened = int(scored.filter(pl.col("delta_fusion_vs_annual") < -1e-9).height)

    summary: dict[str, object] = {
        "source": source,
        "target": target,
        "n_source_parcels": int(a_src.shape[0]),
        "n_target_parcels": int(a_tgt.shape[0]),
        "n_shared_leaves": n_classes,
        "annual_dim": int(a_src.shape[1]),
        "temporal_dim": int(t_src.shape[1]),
        "fusion_dim": int(fus_src.shape[1]),
        "macro_f1_annual": round(macro_annual, 4),
        "macro_f1_temporal": round(macro_temporal, 4),
        "macro_f1_fusion": round(macro_fusion, 4),
        "macro_f1_delta_fusion_vs_annual": round(macro_fusion - macro_annual, 4),
        "macro_f1_delta_temporal_vs_annual": round(macro_temporal - macro_annual, 4),
        "n_scored_leaves": int(scored.height),
        "n_leaves_improved_by_fusion": n_improved,
        "n_leaves_worsened_by_fusion": n_worsened,
        "top_fusion_gains": scored.sort("delta_fusion_vs_annual", descending=True)
        .head(6)
        .select(["leaf", "f1_annual", "f1_fusion", "delta_fusion_vs_annual", "target_support"])
        .to_dicts(),
    }

    logger.info(
        "ensemble_tl_pilot_done",
        source=source,
        target=target,
        n_shared_leaves=n_classes,
        macro_f1_annual=summary["macro_f1_annual"],
        macro_f1_fusion=summary["macro_f1_fusion"],
        macro_f1_delta_fusion_vs_annual=summary["macro_f1_delta_fusion_vs_annual"],
        n_leaves_improved_by_fusion=n_improved,
        n_leaves_worsened_by_fusion=n_worsened,
    )
    return EnsembleTLPilotResult(per_class=per_class, summary=summary)


def save_outputs(result: EnsembleTLPilotResult, out_dir: Path = _OUT_DIR) -> None:
    """Persist the per-class table and the JSON summary to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_class.write_parquet(out_dir / "ensemble_tl_pilot_per_class.parquet")
    (out_dir / "ensemble_tl_pilot_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("ensemble_tl_pilot_saved", out_dir=str(out_dir))
