"""Vectorize REAL PASTIS-R parcels with the VOTING-3 prediction (held-out fold).

Companion to ``build_demo_parcels_real.py`` (which paints ground truth). Here we
show how the EPIC 12 deployment champion -- the Voting-3 weighted soft-vote of
``tsvit-pheno-v2`` + ``U-TAE`` + ``XGB-AlphaEarth`` -- *recognises* crops on a
held-out **fold-5** patch (the members train on folds 1-4, so the prediction is
honest / out-of-sample).

Honesty contract
----------------

The painted posterior is the SAME one the agent serves: each parcel's Voting-3
``(18,)`` posterior is looked up by its canonical id
(:func:`ml.agent.tools.classify._load_voting_three` ->
:meth:`_VotingThree.posterior_for_parcel`) from the cached fold-5 OOF, then masked
+ renormalized over the active label-space's resolved classes
(:data:`ml.eval.class_remap.DEFAULT_LABEL_SPACE`, ``france-12`` for the v2 champion)
with :func:`ml.eval.class_remap.restrict_posterior` -- exactly the pipeline of
``classify._build_result``. A parcel ABSENT from the three-member OOF intersection
is OMITTED (never back-filled with a fabricated class), and if the Voting-3 OOF is
unavailable the script FAILS (no silent degradation to raw XGBoost).

Canonical id scheme
-------------------

The Voting-3 OOF is keyed by ``canonical_parcel_id = f"{patch_id}_{local_id}"``
where ``local_id`` is the value stored in the PASTIS-R ``ParcelIDs_<patch>.npy``
raster (a large native PASTIS parcel id), NOT the small ``INSTANCES_<patch>.npy``
index. Geometry is therefore vectorized from the ``ParcelIDs`` raster so the
polygon's id matches the OOF key.

Output GeoJSON properties per parcel: ``crop_class`` (= predicted, so the map
paints the Voting-3 prediction), ``pred_class``, ``true_class``, ``correct``,
``confidence``, ``class_probabilities`` (top-k of the resolved-class posterior),
``parcel_id`` and ``patch_id``. ``metadata.accuracy`` carries the parcel accuracy
of the Voting-3 over the painted set.

Usage:
    poetry run python scripts/build_demo_parcels_prediction.py
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio.features
import structlog
import typer
from pyproj import Transformer
from rasterio.transform import from_bounds

log = structlog.get_logger(__name__)

_PASTIS_ROOT = Path("data/PASTIS-R")
_ANN = _PASTIS_ROOT / "ANNOTATIONS"
_META = _PASTIS_ROOT / "metadata.geojson"
_OUT = Path("frontend/public/demo/parcelas_prediccion_francia.geojson")

_PATCH_SIDE = 128
_SOURCE_EPSG = "EPSG:2154"  # PASTIS metadata is RGF93 / Lambert-93 (fallback CRS).
_MIN_PARCEL_PX = 12
#: Minimum distinct ground-truth classes a painted patch must show (visual diversity).
_MIN_PATCH_CLASSES = 3
#: How many resolved classes to surface per parcel in ``class_probabilities``.
_TOP_K_PROBS = 6


def _patch_bounds_utm(geometry: dict) -> tuple[float, float, float, float]:
    """Return the ``(minx, miny, maxx, maxy)`` bounds of a patch polygon (metres)."""
    coords = list(
        itertools.chain.from_iterable(itertools.chain.from_iterable(geometry["coordinates"]))
    )
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _reproject_ring(ring: list, transformer: Transformer) -> list:
    """Reproject one polygon ring (list of ``[x, y]``) to rounded lon/lat pairs."""
    out = []
    for x, y in ring:
        lon, lat = transformer.transform(x, y)
        out.append([round(lon, 6), round(lat, 6)])
    return out


def _pick_patch(
    universe_by_patch: dict[str, set[int]],
    gt_by_id: dict[str, int],
) -> str | None:
    """Pick the fold-5 patch maximizing Voting-3 OOF coverage (+ class diversity).

    Ranks the patches by how many of their parcels are present in the Voting-3
    three-member OOF intersection (``universe_by_patch``), preferring the patch
    with the most OOF parcels that also has both PASTIS-R rasters locally and at
    least :data:`_MIN_PATCH_CLASSES` distinct ground-truth classes (so the demo
    paints a visually diverse area, not a monoculture).

    Args:
        universe_by_patch: Mapping ``patch_id -> {local ParcelID}`` of the parcels
            present in the Voting-3 OOF universe, per patch.
        gt_by_id: Mapping ``canonical_parcel_id -> semantic18 label`` for the
            painted set (used only to count distinct classes per candidate patch).

    Returns:
        The chosen patch id (string), or ``None`` if no eligible patch exists.
    """
    ranked = sorted(universe_by_patch, key=lambda p: len(universe_by_patch[p]), reverse=True)
    for patch_id in ranked:
        if not (_ANN / f"ParcelIDs_{patch_id}.npy").exists():
            continue
        if not (_ANN / f"TARGET_{patch_id}.npy").exists():
            continue
        classes = {
            gt_by_id[f"{patch_id}_{local}"]
            for local in universe_by_patch[patch_id]
            if f"{patch_id}_{local}" in gt_by_id
        }
        if len(classes) < _MIN_PATCH_CLASSES:
            continue
        return patch_id
    return None


def _resolve_transformer(meta: dict) -> Transformer:
    """Build the EPSG:2154 -> EPSG:4326 transformer from the metadata CRS.

    Reads the source CRS declared in the PASTIS-R ``metadata.geojson`` (falling
    back to :data:`_SOURCE_EPSG` when absent), so the reprojection matches the
    geometry rebuild in ``ml.agent.tools.classify._build_parcel_geometries``.

    Args:
        meta: Parsed ``metadata.geojson`` mapping.

    Returns:
        A ``pyproj.Transformer`` mapping the source CRS to EPSG:4326 (lon/lat).
    """
    crs_name = meta.get("crs", {}).get("properties", {}).get("name", _SOURCE_EPSG)
    return Transformer.from_crs(crs_name, "EPSG:4326", always_xy=True)


def main() -> None:
    """Build the Voting-3 prediction demo GeoJSON for one held-out fold-5 patch."""
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
    from ml.eval.class_remap import (
        DEFAULT_LABEL_SPACE,
        get_label_space,
        restrict_posterior,
    )
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    # 1) Load the REAL Voting-3 OOF universe (fails loudly if the artifacts are
    #    missing -- never degrades silently to raw XGBoost). The keys are the
    #    canonical ``"{patch}_{local}"`` ids of the three-member OOF intersection.
    try:
        from ml.agent.tools.classify import _build_parcel_ground_truth, _load_voting_three

        voting = _load_voting_three()
        canonical_ids = sorted(voting.member_probs_by_id)
        gt_frame = _build_parcel_ground_truth(canonical_ids)
    except FileNotFoundError as exc:
        log.error("voting3_oof_unavailable", error=str(exc))
        raise typer.Exit(code=1) from exc

    if not canonical_ids:
        log.error("voting3_oof_empty")
        raise typer.Exit(code=1)

    gt_by_id: dict[str, int] = dict(
        zip(
            gt_frame.get_column("canonical_parcel_id").to_list(),
            gt_frame.get_column("label").to_list(),
            strict=True,
        )
    )

    universe_by_patch: dict[str, set[int]] = {}
    for cid in canonical_ids:
        patch_str, local_str = cid.split("_", 1)
        universe_by_patch.setdefault(patch_str, set()).add(int(local_str))

    patch_id = _pick_patch(universe_by_patch, gt_by_id)
    if patch_id is None:
        log.error("no_eligible_fold5_patch")
        raise typer.Exit(code=1)

    # 2) Patch geometry + raster (vectorize from ParcelIDs, the OOF key source).
    meta = json.loads(_META.read_text(encoding="utf-8"))
    feat = next(x for x in meta["features"] if str(x["properties"]["ID_PATCH"]) == patch_id)
    minx, miny, maxx, maxy = _patch_bounds_utm(feat["geometry"])
    transform = from_bounds(minx, miny, maxx, maxy, _PATCH_SIDE, _PATCH_SIDE)
    transformer = _resolve_transformer(meta)
    parcel_ids = load_pastis_parcel_ids(patch_id, _PASTIS_ROOT).astype(np.int32)

    # 3) Active label-space (france-12 for the v2 champion) -- the SAME resolved
    #    vocabulary classify._build_result restricts to.
    label_space = get_label_space(DEFAULT_LABEL_SPACE)

    features: list[dict] = []
    n_correct = 0
    n_painted = 0
    for local in sorted(universe_by_patch[patch_id]):
        canonical_id = f"{patch_id}_{local}"
        posterior = voting.posterior_for_parcel(canonical_id)
        if posterior is None:
            # Not in the three-member OOF intersection -- omit (never fabricate).
            continue
        mask = parcel_ids == local
        if int(mask.sum()) < _MIN_PARCEL_PX:
            continue

        # Restrict + renormalize over the resolved classes, exactly as the agent.
        restricted = restrict_posterior(np.asarray(posterior, dtype=np.float64), label_space)
        named = {
            label_space.class_names.get(cid, SEMANTIC18_CLASS_NAMES.get(cid, str(cid))): float(p)
            for cid, p in restricted.items()
        }
        if not named or max(named.values(), default=0.0) <= 0.0:
            # No mass on the resolved classes -- honest "unresolved", omit.
            continue
        pred_name = max(named, key=lambda k: named[k])
        confidence = float(named[pred_name])
        top_probs = dict(sorted(named.items(), key=lambda kv: kv[1], reverse=True)[:_TOP_K_PROBS])

        gt_label = gt_by_id.get(canonical_id)
        true_name = (
            SEMANTIC18_CLASS_NAMES.get(int(gt_label), "Unknown")
            if gt_label is not None
            else "Unknown"
        )
        correct = true_name != "Unknown" and pred_name == true_name
        n_painted += 1
        n_correct += int(correct)

        for geom, _ in rasterio.features.shapes(parcel_ids, mask=mask, transform=transform):
            rings = [_reproject_ring(r, transformer) for r in geom["coordinates"]]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": {
                        "parcel_id": canonical_id,
                        "patch_id": patch_id,
                        "crop_class": pred_name,  # map paints the Voting-3 prediction
                        "pred_class": pred_name,
                        "true_class": true_name,
                        "correct": correct,
                        "confidence": round(confidence, 3),
                        "class_probabilities": {
                            name: round(prob, 3) for name, prob in top_probs.items()
                        },
                    },
                }
            )
    if not features:
        log.error("no_features_painted", patch=patch_id)
        raise typer.Exit(code=1)

    all_pts = [pt for f in features for ring in f["geometry"]["coordinates"] for pt in ring]
    lons = [pt[0] for pt in all_pts]
    lats = [pt[1] for pt in all_pts]
    accuracy = round(n_correct / max(n_painted, 1), 3)
    fc = {
        "type": "FeatureCollection",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "metadata": {
            "source": "PASTIS-R fold-5 (held-out) -- Voting-3 prediction vs ground truth",
            "model": "Voting-3 (tsvit-pheno-v2 + U-TAE + XGB-AlphaEarth), fold-5 held-out",
            "label_space": label_space.name,
            "patch_id": patch_id,
            "n_parcels": n_painted,
            "accuracy": accuracy,
            "pred_counts": dict(Counter(f["properties"]["pred_class"] for f in features)),
        },
        "features": features,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(fc), encoding="utf-8")
    log.info(
        "prediction_geojson_written",
        path=str(_OUT),
        patch=patch_id,
        model="voting-3",
        label_space=label_space.name,
        n_parcels=n_painted,
        accuracy=accuracy,
    )


if __name__ == "__main__":
    typer.run(main)
