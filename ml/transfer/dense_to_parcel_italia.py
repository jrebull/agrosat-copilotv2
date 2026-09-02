"""Dense -> parcel bridge for the Italian Voting-3 (US-079).

The champion Voting-3 (``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth``, F1 0.9069
in France) votes at the PARCEL level: every member is reduced to one post-softmax
distribution per parcel and the three are combined with
:class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble`. The Italian dense
members (TSViT-pheno, U-TAE) instead emit a DENSE post-softmax map ``(K, 128,
128)`` per patch (:func:`ml.transfer.finetune_italia.run_italia_finetune`). This
module is the missing bridge: it aggregates each member's dense probabilities to a
per-parcel distribution so all three members vote in the SAME per-parcel namespace
as the champion, dumping ``oof_parcel_<member>_italia_fold5.parquet`` (the exact
contract :class:`ml.ensemble.xgb_alphaearth_italia` already writes for the third
member).

Why a bridge is needed (the parcel-id gap)
------------------------------------------
The US-078 builder (:func:`ml.data.eurocrops_pastis_builder.rasterize_patch_mask`)
burns ONLY the crop ``class_id`` into ``TARGET_<id>.npy`` (``fill=0`` background);
it never rasterises a parallel ParcelID channel. So a dense map carries no parcel
identity and cannot be reduced per parcel out of the box. US-078 must NOT be
touched, so this module derives the per-parcel partition at vote time, with two
honest strategies the caller picks between:

- ``"blobs"`` (default, no extra inputs): treat each connected component of a
  single class in the ground-truth mask as one approximate parcel
  (:func:`scipy.ndimage.label`). Each blob's pixels are averaged into a parcel
  distribution. The blob ids live in a SELF-CONTAINED namespace
  ``iti1_2018_p{patch}_blob_{n}`` that does NOT join to the EuroCrops
  ``xgb-alphaearth-italia`` parcels (those are real polygons), so this mode is for
  a dense-only vote (TSViT + U-TAE) or a smoke test of the wiring.
- ``"eurocrops"`` (recommended for the champion terna): rasterise the SAME
  EuroCrops polygons the xgb member used into a per-pixel ParcelID channel,
  reusing :func:`ml.transfer.alphaearth_italia.parcels_in_patches` so the parcel
  ``canonical_parcel_id`` (``iti1_2018_p{patch}_{seq}``) matches the xgb OOF
  1:1. The per-patch window transform is reconstructed from the bbox the US-078
  ``metadata.parquet`` already persists
  (``rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, 128,
  128)``) -- NO Sentinel Hub tile cache is needed. The dense probs are then
  averaged within each real parcel and the join with the xgb OOF is exact (its
  coverage is reported honestly).

Aggregation
-----------
For a parcel, the member distribution is the MEAN of the post-softmax rows of its
pixels, renormalised to sum to 1 (a convex combination of distributions, so still
post-softmax). The class axis is the dense label space ``[0, K)`` (background id 0
kept as a column for shape parity with the dense maps; it is dropped by the
parcel-vote class reconciliation, mirroring the dense metric convention).

Anti-leakage (R-LEAK)
---------------------
Each parcel inherits the ``fold_espacial`` of its patch (US-078), so the OOF
parquet carries the SAME spatial fold map the dense members and the xgb member
use; the downstream parcel Voting learns its weights leave-one-spatial-fold-out
over that fold (no parcel of a scored fold leaks into the weight fit). Every
per-parcel row is renormalised and validated post-softmax.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints + Google-style docstrings; visible
prose Spanish, code identifiers English; no emojis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_ITALIA_ROOT",
    "DEFAULT_OOF_DIR",
    "ParcelAggregation",
    "ParcelStrategy",
    "aggregate_dense_member_to_parcel",
    "blob_parcels_for_patch",
    "eurocrops_parcel_id_raster",
    "load_eurocrops_parcel_rasters",
    "write_parcel_oof",
]

#: Repo root (this file is ``<root>/ml/transfer/dense_to_parcel_italia.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory the parcel-level Voting consumes (same as the xgb member's OOF).
DEFAULT_OOF_DIR: Path = _REPO_ROOT / "ml" / "eval" / "oof"

#: US-078 homologue dataset root (carries ``metadata.parquet`` with the per-patch
#: bbox + ``class_mapping.json`` -- the only inputs the EuroCrops ParcelID raster
#: needs, NO tile cache).
DEFAULT_ITALIA_ROOT: Path = _REPO_ROOT / "data" / "pastis_italia_2018"

#: Numerical floor so an empty/degenerate parcel never divides by zero.
_PROB_EPS: float = 1e-12

#: Patch side in pixels (= PASTIS; the masks ``TARGET_<id>.npy`` are 128x128).
_PATCH_PX: int = 128

#: How to derive the per-parcel partition of a dense patch.
ParcelStrategy = Literal["blobs", "eurocrops"]


@dataclass
class ParcelAggregation:
    """A member's per-parcel distributions aggregated from its dense maps.

    Attributes:
        member: The member name (e.g. ``"tsvit-pheno"``).
        canonical_parcel_ids: Per-parcel canonical id (Utf8), one per row.
        probs: Per-parcel post-softmax matrix ``(n_parcels, K)`` over the dense
            label space (background column included for shape parity).
        class_ids: Per-parcel majority ground-truth class id ``(n_parcels,)``
            (the dense mask's modal crop class inside the parcel).
        folds: Per-parcel spatial fold ``(n_parcels,)`` (US-078 ``fold_espacial``).
        num_classes: The dense class axis size ``K`` (background included).
    """

    member: str
    canonical_parcel_ids: list[str]
    probs: np.ndarray
    class_ids: np.ndarray
    folds: np.ndarray
    num_classes: int


def blob_parcels_for_patch(mask: np.ndarray, *, background_id: int = 0) -> np.ndarray:
    """Label per-class connected components of a dense mask as pseudo-parcels.

    Each contiguous blob of a single crop class is one approximate parcel. Two
    parcels of the SAME class touching diagonally stay merged (4-connectivity is
    used so only edge-adjacent pixels join, the conservative choice); two parcels
    of DIFFERENT classes never merge because the labelling is run per class. The
    background (``background_id``) is left at id 0 and never forms a parcel.

    Args:
        mask: Dense ground-truth class mask ``(H, W)`` (0 = background).
        background_id: The id excluded from blobs (default 0).

    Returns:
        An int64 label map ``(H, W)`` where each crop blob has a unique id
        ``>= 1`` and background is 0.
    """
    from scipy import ndimage

    out = np.zeros(mask.shape, dtype=np.int64)
    offset = 0
    for cid in np.unique(mask):
        if int(cid) == background_id:
            continue
        labelled, n_blobs = ndimage.label(mask == cid)
        if n_blobs == 0:
            continue
        shifted = np.where(labelled > 0, labelled + offset, 0)
        out = np.where(mask == cid, shifted, out)
        offset += n_blobs
    return out


def eurocrops_parcel_id_raster(
    parcels: object,
    bbox_4326: tuple[float, float, float, float],
    *,
    patch_px: int = _PATCH_PX,
) -> tuple[np.ndarray, dict[int, str]]:
    """Rasterise the EuroCrops parcels of one patch into a per-pixel ParcelID map.

    Burns each parcel's ``canonical_parcel_id`` (via a contiguous int surrogate,
    because :func:`rasterio.features.rasterize` needs an integer per geometry)
    onto the patch's 128x128 grid, mirroring the US-078 builder's rasterisation
    pattern (:func:`ml.data.eurocrops_pastis_builder.rasterize_patch_mask`) but
    keying by PARCEL identity instead of crop class. The window transform is
    reconstructed from the patch bbox the US-078 ``metadata.parquet`` already
    persists (``rasterio.transform.from_bounds(min_lon, min_lat, max_lon,
    max_lat, 128, 128)``), so NO Sentinel Hub tile cache is needed: the same
    EuroCrops polygons the ``xgb-alphaearth-italia`` member used
    (:func:`ml.transfer.alphaearth_italia.parcels_in_patches`) map to the SAME
    ``canonical_parcel_id`` (``iti1_2018_p{patch}_{seq}``), so the dense
    aggregation joins to the xgb OOF 1:1.

    Note:
        ``from_bounds`` builds a regular lon/lat grid; the US-078 masks were
        rasterised on the downloaded tile's projected (UTM) transform, so a small
        sub-pixel / projective offset remains at parcel borders (the pilot
        measured ~0.73 region IoU vs ``TARGET_<id>.npy``). That offset only
        perturbs which border pixels fall in which parcel when AVERAGING the dense
        probabilities; the per-parcel JOIN to the xgb OOF is exact regardless
        (it is keyed by ``canonical_parcel_id``, not geometry). The interior of
        every parcel is correctly captured, which is what the per-parcel mean
        needs. The centroid fallback below lifts the join coverage with the xgb
        member from ~80% to ~92% on the pilot (213 sub-pixel parcels would
        otherwise paint zero pixels and be dropped).

    Args:
        parcels: The in-patch parcels of this patch (a ``geopandas`` GeoDataFrame
            in EPSG:4326 with ``canonical_parcel_id`` + ``geometry``), the output
            of :func:`ml.transfer.alphaearth_italia.parcels_in_patches` filtered
            to one patch.
        bbox_4326: The patch bbox ``(min_lon, min_lat, max_lon, max_lat)`` from
            the US-078 ``metadata.parquet``.
        patch_px: Patch side in pixels (default 128 = PASTIS).

    Returns:
        ``(parcel_id_map, id_to_canonical)`` where ``parcel_id_map`` is an int64
        ``(patch_px, patch_px)`` map (0 = background, else a 1-based local parcel
        surrogate) and ``id_to_canonical`` maps that surrogate to the parcel's
        ``canonical_parcel_id`` (Utf8) -- the inverse needed to recover the join
        key after rasterising integers.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds, rowcol

    geoms = list(parcels.geometry)
    canonical = [str(c) for c in parcels["canonical_parcel_id"].tolist()]
    min_lon, min_lat, max_lon, max_lat = bbox_4326
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, patch_px, patch_px)
    # Contiguous int surrogate per parcel (rasterize needs one int per geometry);
    # the inverse map recovers the Utf8 canonical_parcel_id at join time.
    shapes = [
        (geom, i + 1) for i, geom in enumerate(geoms) if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((patch_px, patch_px), dtype=np.int64), {}
    parcel_map = rasterize(
        shapes,
        out_shape=(patch_px, patch_px),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    ).astype(np.int64)
    # Centroid fallback: a sub-pixel parcel (smaller than a 10 m cell, or whose
    # interior falls between cell centres) paints ZERO pixels with
    # ``all_touched=False`` and would be dropped from the per-parcel vote, breaking
    # the 1:1 join with the xgb member. Assign each unpainted parcel its centroid
    # pixel ONLY when that pixel is still background, so a large neighbour is never
    # overwritten (``all_touched=True`` is rejected for the opposite reason: it
    # lets a small parcel steal border pixels and mis-aligns with the
    # ``all_touched=False`` US-078 masks). This lifts the join coverage without
    # corrupting the interior of any well-resolved parcel.
    painted = set(int(v) for v in np.unique(parcel_map) if int(v) != 0)
    for i, geom in enumerate(geoms):
        surrogate = i + 1
        if surrogate in painted or geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        row, col = rowcol(transform, centroid.x, centroid.y)
        if 0 <= row < patch_px and 0 <= col < patch_px and parcel_map[row, col] == 0:
            parcel_map[row, col] = surrogate
    id_to_canonical = {i + 1: canonical[i] for i in range(len(canonical))}
    return parcel_map, id_to_canonical


def load_eurocrops_parcel_rasters(
    italia_root: Path = DEFAULT_ITALIA_ROOT,
    *,
    patch_ids: list[int] | None = None,
    parcels_parquet: Path | None = None,
    mapping_csv: Path | None = None,
    region_prefix: str = "it",
) -> dict[int, tuple[np.ndarray, dict[int, str]]]:
    """Build per-patch EuroCrops ParcelID rasters from the US-078 metadata.

    The ORCHESTRATOR's finding: no Sentinel Hub tile cache is needed. The US-078
    ``metadata.parquet`` already stores each patch's EPSG:4326 bbox (columns
    ``bbox_min_lon``, ``bbox_min_lat``, ``bbox_max_lon``, ``bbox_max_lat``), and
    the patch's window transform is reconstructed with
    ``rasterio.transform.from_bounds(...)``. This loads the canonical Italian
    label space + the patch bboxes, intersects the EuroCrops Italy 2018 parcels
    (reusing :func:`ml.transfer.alphaearth_italia.parcels_in_patches`, the SAME
    polygons + ``canonical_parcel_id`` the xgb member used), and rasterises each
    patch's parcels with :func:`eurocrops_parcel_id_raster`. The result aligns
    pixel-for-pixel with ``TARGET_<id>.npy`` up to the small lon/lat-vs-UTM offset
    noted there, and joins to the xgb OOF 1:1 by ``canonical_parcel_id``.

    Args:
        italia_root: The US-078 homologue dataset root (carries
            ``metadata.parquet`` + ``class_mapping.json``).
        patch_ids: Restrict to these patch ids; ``None`` rasterises every patch in
            the metadata.

    Returns:
        ``{patch_id: (parcel_id_map, id_to_canonical)}`` for every requested patch
        with at least one in-patch parcel.
    """
    from ml.transfer.alphaearth_italia import (
        load_italia_label_space,
        load_patch_bboxes,
        parcels_in_patches,
    )

    name_to_id = load_italia_label_space(italia_root)
    bboxes = load_patch_bboxes(italia_root)
    # CRITICAL (US-079 vote-join bug): ``canonical_parcel_id`` is assigned by
    # ``parcels_in_patches`` as a GLOBAL running row index over EVERY in-patch
    # parcel of the frame it receives. Filtering ``bboxes`` to ``patch_ids`` BEFORE
    # that call would re-enumerate the seqs over a smaller universe, so the SAME
    # physical polygon gets a DIFFERENT id than the ``xgb-alphaearth-italia`` member
    # (which always runs ``parcels_in_patches`` over the FULL metadata) -> the
    # per-parcel join collapses to a handful of spurious seq collisions (run2 voted
    # over 32 parcels instead of ~22k). FIX: ALWAYS build the parcels over the FULL
    # metadata so the canonical ids match the xgb 1:1, then keep only the rasters of
    # the requested patches AFTER rasterisation.
    pip_kwargs: dict[str, object] = {"region_prefix": region_prefix}
    if parcels_parquet is not None:
        pip_kwargs["parcels_parquet"] = parcels_parquet
    if mapping_csv is not None:
        pip_kwargs["mapping_csv"] = mapping_csv
    parcels = parcels_in_patches(bboxes, name_to_id, **pip_kwargs)
    if patch_ids is not None:
        keep = set(int(p) for p in patch_ids)
        bboxes = bboxes.filter(pl.col("patch_id").is_in(list(keep)))

    rasters: dict[int, tuple[np.ndarray, dict[int, str]]] = {}
    for row in bboxes.iter_rows(named=True):
        pid = int(row["patch_id"])
        in_patch = parcels[parcels["patch_id"] == pid]
        if len(in_patch) == 0:
            continue
        bbox = (
            float(row["bbox_min_lon"]),
            float(row["bbox_min_lat"]),
            float(row["bbox_max_lon"]),
            float(row["bbox_max_lat"]),
        )
        parcel_map, id_to_canonical = eurocrops_parcel_id_raster(in_patch, bbox)
        rasters[pid] = (parcel_map, id_to_canonical)
    logger.info(
        "eurocrops_parcel_rasters_built",
        n_patches=len(rasters),
        n_parcels=int(sum(len(c) for _, c in rasters.values())),
        note="from_bounds(metadata bbox); no tile cache needed.",
    )
    return rasters


def aggregate_dense_member_to_parcel(
    member: str,
    probs_by_patch: Mapping[int, np.ndarray],
    masks_by_patch: Mapping[int, np.ndarray],
    folds_by_patch: Mapping[int, int],
    *,
    num_classes: int,
    strategy: ParcelStrategy = "blobs",
    background_id: int = 0,
    parcel_rasters: Mapping[int, tuple[np.ndarray, dict[int, str]]] | None = None,
) -> ParcelAggregation:
    """Aggregate a member's dense post-softmax maps to per-parcel distributions.

    For each patch a per-pixel parcel partition is derived (connected-component
    blobs by default, or the EuroCrops ParcelID raster when ``strategy``
    ``"eurocrops"`` and ``parcel_rasters`` is given). Within each parcel the
    post-softmax rows of its pixels are averaged and renormalised to a single
    distribution. Each parcel keeps the modal ground-truth class id and its
    patch's spatial fold.

    Args:
        member: The member name (for the OOF filename / logs).
        probs_by_patch: ``{patch_id: (K, H, W)}`` post-softmax maps.
        masks_by_patch: ``{patch_id: (H, W)}`` ground-truth class masks.
        folds_by_patch: ``{patch_id: fold_espacial}`` spatial fold per patch.
        num_classes: The dense class axis size ``K`` (background included).
        strategy: ``"blobs"`` (default, self-contained) or ``"eurocrops"``
            (joins to the xgb OOF, needs ``parcel_rasters``).
        background_id: The mask id excluded from parcels (default 0).
        parcel_rasters: For ``strategy="eurocrops"``, ``{patch_id:
            (parcel_id_map, id_to_canonical)}`` from
            :func:`load_eurocrops_parcel_rasters`.

    Returns:
        A :class:`ParcelAggregation` with the per-parcel distributions, canonical
        ids, modal class ids and folds.

    Raises:
        ValueError: if ``strategy="eurocrops"`` without ``parcel_rasters``, or a
            patch is missing from the probability maps.
    """
    if strategy == "eurocrops" and parcel_rasters is None:
        raise ValueError(
            "strategy='eurocrops' needs parcel_rasters (the per-patch EuroCrops "
            "ParcelID maps); build them with load_eurocrops_parcel_rasters."
        )

    canonical_ids: list[str] = []
    rows: list[np.ndarray] = []
    class_ids: list[int] = []
    folds: list[int] = []

    common = sorted(set(probs_by_patch) & set(masks_by_patch))
    for pid in common:
        probs = np.asarray(probs_by_patch[pid], dtype=np.float64)  # (K, H, W)
        mask = np.asarray(masks_by_patch[pid], dtype=np.int64)  # (H, W)
        fold = int(folds_by_patch.get(pid, 0))

        if strategy == "eurocrops":
            assert parcel_rasters is not None  # narrowed above
            if pid not in parcel_rasters:
                continue
            parcel_map, id_to_canonical = parcel_rasters[pid]
        else:
            parcel_map = blob_parcels_for_patch(mask, background_id=background_id)
            id_to_canonical = {
                int(b): f"iti1_2018_p{pid}_blob_{int(b)}"
                for b in np.unique(parcel_map)
                if int(b) != 0
            }

        flat_probs = probs.reshape(probs.shape[0], -1).T  # (n_pix, K)
        flat_parcel = parcel_map.reshape(-1)
        flat_mask = mask.reshape(-1)
        for local_id, canonical in id_to_canonical.items():
            sel = flat_parcel == local_id
            if not sel.any():
                continue
            parcel_probs = flat_probs[sel].mean(axis=0)
            denom = parcel_probs.sum()
            parcel_probs = parcel_probs / (denom if denom > _PROB_EPS else 1.0)
            # Modal crop class of the parcel pixels (background excluded if any).
            parcel_mask = flat_mask[sel]
            crop_pix = parcel_mask[parcel_mask != background_id]
            modal = int(np.bincount(crop_pix).argmax()) if crop_pix.size else background_id
            canonical_ids.append(canonical)
            rows.append(parcel_probs)
            class_ids.append(modal)
            folds.append(fold)

    if not rows:
        raise ValueError(
            f"member {member!r}: no parcel aggregated from the dense maps "
            f"(strategy={strategy!r}); check the masks/probs alignment."
        )

    probs_matrix = np.stack(rows, axis=0)  # (n_parcels, K)
    logger.info(
        "dense_member_aggregated_to_parcel",
        member=member,
        strategy=strategy,
        n_parcels=len(canonical_ids),
        num_classes=num_classes,
        n_patches=len(common),
    )
    return ParcelAggregation(
        member=member,
        canonical_parcel_ids=canonical_ids,
        probs=probs_matrix,
        class_ids=np.asarray(class_ids, dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        num_classes=num_classes,
    )


def write_parcel_oof(
    aggregation: ParcelAggregation,
    *,
    oof_dir: Path = DEFAULT_OOF_DIR,
    crop_class_ids: tuple[int, ...] | None = None,
) -> Path:
    """Dump a dense member's per-parcel distributions as the Voting OOF parquet.

    Writes ``oof_parcel_<member>_italia_fold5.parquet`` with
    ``canonical_parcel_id`` + ``prob_000..`` + ``pred_class`` + ``class_id`` +
    ``fold`` -- the SAME schema :class:`ml.ensemble.xgb_alphaearth_italia` writes,
    so the three members share one contract. The prob columns are laid out over
    ``crop_class_ids`` (the global crop ids, background excluded) when given, so
    every member's column ``i`` is the SAME global class; otherwise the dense
    ``[1, K)`` crop ids are used.

    Args:
        aggregation: The per-parcel aggregation to dump.
        oof_dir: Directory the Voting reads from.
        crop_class_ids: The global crop class ids the prob columns map to, in
            order (background excluded). ``None`` -> the dense crop ids ``[1, K)``.

    Returns:
        The path of the written parquet.
    """
    bg = 0
    if crop_class_ids is None:
        crop_class_ids = tuple(c for c in range(aggregation.num_classes) if c != bg)
    # Scatter the dense (K,) distributions onto the crop-only column space.
    col_of = {cid: i for i, cid in enumerate(crop_class_ids)}
    n_cols = len(crop_class_ids)
    scattered = np.zeros((aggregation.probs.shape[0], n_cols), dtype=np.float64)
    for dense_id in range(aggregation.num_classes):
        if dense_id in col_of:
            scattered[:, col_of[dense_id]] = aggregation.probs[:, dense_id]
    # Renormalise over the crop columns (drop the background mass), keep post-softmax.
    row_sums = scattered.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < _PROB_EPS, 1.0, row_sums)
    scattered = scattered / row_sums

    pred_global = np.asarray(crop_class_ids, dtype=np.int64)[scattered.argmax(axis=1)]
    prob_cols = {f"prob_{i:03d}": scattered[:, i].astype(np.float32) for i in range(n_cols)}
    oof_df = pl.DataFrame(
        {
            "canonical_parcel_id": [str(x) for x in aggregation.canonical_parcel_ids],
            **prob_cols,
            "pred_class": [int(c) for c in pred_global],
            "class_id": [int(c) for c in aggregation.class_ids],
            "fold": [int(f) for f in aggregation.folds],
        }
    )
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"oof_parcel_{aggregation.member}_italia_fold5.parquet"
    oof_df.write_parquet(oof_path)
    logger.info(
        "dense_member_parcel_oof_written",
        member=aggregation.member,
        path=str(oof_path),
        n_parcels=oof_df.height,
        n_cols=n_cols,
    )
    return oof_path
