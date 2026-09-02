"""Region-category multi-object dataset over real PASTIS-R (US-036-a v2, T2).

This module is the FarSLIP v2 redesign (Li et al. 2025, secs. 3.3, 4.1, 4.3) of
the impoverished 1-label-per-patch v1 (``ml/farslip/pastis_pair_dataset.py``).
Whereas v1 collapsed each patch to a single dominant ``category_id``, v2 exposes
**every PASTIS category present in the patch** as a separate region-category
entry, so the Multi-Positive Contrastive Loss (MPCL, T3) can group regions of the
same category across patches.

Pipeline per patch:

    load_pastis_patch(pid) -> s2 (T, 10, 128, 128) int16, instance (H, W)
    peak_ndvi_composite    -> img4 (4, 128, 128) float32 [0,1] (REUSED from v1)
    extract_regions        -> [(parcel_instance_id, category_id PASTIS)] (N >= 1)
    captions[pid]          -> L_glo caption injected externally (NOT generated here)

Design decisions fixed by the plan (US-036-a v2, secs. 2.2-2.3):

    * Multi-object: ``mean_regions_per_patch > 1`` on PASTIS-R (panoptic dataset
      has several parcels/classes per patch). One entry PER category present
      (Background 0 and Void 19 excluded), NOT a single dominant label.
    * ``V_i^r`` = patch CLS token (paper sec. 4.3 Takeaway-1: CLS, not RoI).
      Caveat R-REGION-CROP: PASTIS is one patch per image; all regions of a patch
      share that patch's CLS. The multiplicity enters the contrast as multiple
      (patch, category) pairs that MPCL treats cross-patch. The dataset therefore
      exposes the LIST of category_ids per patch and the collate flattens them.
    * The 3:1 Meadow dominance filter (``PastisFilter`` ``dominance_ratio``) is
      OPTIONAL (``ratio=None`` by default): v2's point is multi-object, so we do
      NOT collapse to a dominant class.
    * Official PASTIS folds for spatial CV; the dataset itself does not split,
      but ``assert_disjoint_folds`` is exposed for the orchestrator (anti-leak).
    * The dataset does NOT call Gemma. ``captions`` is an injected
      ``dict[patch_id, caption]`` materialized upstream (T1).

Public surface consumed by the trainer (T4):

    extract_regions(parcel_ids, semantic, active_class_ids, min_area_px)
    assert_disjoint_folds(train_folds, val_folds)
    RegionCategoryPairDataset(captions, root, folds, ...)
    collate_region_batch(items) -> the flattened cross-patch batch

Project convention: ``torch``/``numpy`` only at the data boundary; logging via
``structlog``; no pandas; type hints everywhere; docstrings in English; visible
prose (logs are structured keys, not prose) without emojis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.farslip.pastis_pair_dataset import (
    _DEFAULT_PASTIS_ROOT,
    _N_PASTIS_CROPS,
    active_classes,
    peak_ndvi_composite,
)
from ml.ingest.pastis_loader import load_pastis_patch

logger = structlog.get_logger(__name__)

#: Non-agronomic classes never turned into a region (paper: only crop categories).
_BACKGROUND_CLASS: int = 0
_VOID_CLASS: int = 19

#: Sentinel parcel instance id meaning "no parcel" in ``ParcelIDs``.
_NO_PARCEL: int = 0

#: Default minimum region area (pixels). Slivers below this add no signal and
#: destabilize the MPCL; the plan (sec. 2.2 step 4) sets 16 px.
_DEFAULT_MIN_AREA_PX: int = 16

#: All 18 agronomic PASTIS categories (1..18) in canonical id order. The default
#: ``active_class_ids`` so v2 keeps every present crop, not just a curriculum head.
_ALL_ACTIVE_CLASS_IDS: tuple[int, ...] = tuple(range(1, _N_PASTIS_CROPS + 1))


def assert_disjoint_folds(
    train_folds: Sequence[int],
    val_folds: Sequence[int],
) -> None:
    """Raises if train and validation folds overlap (spatial-CV anti-leakage).

    Args:
        train_folds: official PASTIS folds used for training.
        val_folds: official PASTIS folds held out for evaluation.

    Raises:
        ValueError: if ``set(train_folds) & set(val_folds)`` is non-empty.
    """
    overlap = sorted(set(int(f) for f in train_folds) & set(int(f) for f in val_folds))
    if overlap:
        raise ValueError(
            "train_folds and val_folds must be disjoint (spatial CV); "
            f"overlapping folds: {overlap}."
        )


def extract_regions(
    parcel_ids: np.ndarray,
    semantic: np.ndarray,
    active_class_ids: tuple[int, ...] = _ALL_ACTIVE_CLASS_IDS,
    min_area_px: int = _DEFAULT_MIN_AREA_PX,
) -> list[tuple[int, int]]:
    """Extracts every valid region-category of a patch from its instance mask.

    Each distinct parcel instance (``parcel_ids != 0``) becomes a region whose
    category is the MAJORITY semantic class inside that instance (PASTIS parcels
    are monoculture, so the majority is the true crop). A region is dropped when
    its majority class is Background (0), Void (19), not in ``active_class_ids``,
    or its area is below ``min_area_px``. This is the v2 multi-object core: a
    patch yields ``N >= 0`` regions (typically ``> 1``), NOT one dominant label.

    Args:
        parcel_ids: ``(H, W)`` integer mask, parcel instance id per pixel
            (``0`` = no parcel). PASTIS ``ANNOTATIONS/ParcelIDs_<pid>.npy``.
        semantic: ``(H, W)`` integer mask, PASTIS class id per pixel (0..19).
            PASTIS ``ANNOTATIONS/TARGET_<pid>.npy[0]``.
        active_class_ids: categories kept (default all 18 crops 1..18). A region
            whose majority class is outside this set is dropped.
        min_area_px: minimum number of pixels of an instance to keep it.

    Returns:
        List of ``(parcel_instance_id, category_id)`` tuples, one per valid
        region, ordered by ascending ``parcel_instance_id`` for determinism.

    Raises:
        ValueError: if ``parcel_ids`` and ``semantic`` shapes differ.
    """
    parcels = np.asarray(parcel_ids)
    sem = np.asarray(semantic)
    if parcels.shape != sem.shape:
        raise ValueError(f"parcel_ids {parcels.shape} and semantic {sem.shape} must match.")

    active = set(int(c) for c in active_class_ids)
    parcels_flat = parcels.ravel().astype(np.int64)
    sem_flat = sem.ravel().astype(np.int64)

    regions: list[tuple[int, int]] = []
    instance_ids = np.unique(parcels_flat)
    for inst in instance_ids:
        inst_id = int(inst)
        if inst_id == _NO_PARCEL:
            continue
        member = parcels_flat == inst_id
        area = int(member.sum())
        if area < min_area_px:
            continue
        # Majority semantic class inside the instance (monoculture assumption).
        classes_here = sem_flat[member]
        counts = Counter(int(c) for c in classes_here)
        majority_class, _ = counts.most_common(1)[0]
        if majority_class in (_BACKGROUND_CLASS, _VOID_CLASS):
            continue
        if majority_class not in active:
            continue
        regions.append((inst_id, majority_class))

    return regions


class RegionCategoryPairDataset(Dataset):
    """Region-category multi-object pair dataset over real PASTIS-R (v2).

    Each item exposes a single patch with: the peak-NDVI 4-band composite
    (training tensor, same as v1), the injected global caption ``L_glo``, and the
    LIST of PASTIS ``category_id`` present in the patch (one per valid region from
    :func:`extract_regions`). The cross-patch region-category batch for the MPCL
    is assembled by :func:`collate_region_batch`, which flattens the per-patch
    category lists and records, for each region, the index of its patch.

    The ``category_id`` values are RAW PASTIS class ids (1..18). The trainer maps
    them to prototype-bank rows via its own canonical category ordering (the same
    ``active_class_ids`` used here), so this dataset stays decoupled from US-033.

    Args:
        captions: injected ``{patch_id: caption_glo}`` (str keys), materialized
            upstream (T1). A patch without a caption is rejected explicitly.
        root: PASTIS-R root (default ``data/PASTIS-R``).
        folds: official PASTIS folds (spatial CV; default ``(1, 2, 3)``).
        active_class_ids: categories kept (default all 18 crops 1..18).
        min_area_px: minimum region area in pixels (default 16).
        dominance_ratio: if not ``None``, apply the 3:1 Meadow ``PastisFilter``
            (``dominance_ratio`` mode) before region extraction. Default ``None``
            (do NOT collapse to a dominant class; v2 is multi-object).
        resize_to: target side of the composite (default 224, CLIP).
        seed: determinism seed (default 42).

    Raises:
        KeyError: at ``__getitem__`` time if a kept patch has no caption.
    """

    def __init__(
        self,
        captions: dict[str, str],
        root: Path = _DEFAULT_PASTIS_ROOT,
        folds: Sequence[int] = (1, 2, 3),
        active_class_ids: tuple[int, ...] = _ALL_ACTIVE_CLASS_IDS,
        min_area_px: int = _DEFAULT_MIN_AREA_PX,
        dominance_ratio: float | None = None,
        resize_to: int = 224,
        seed: int = 42,
    ) -> None:
        self.captions: dict[str, str] = {str(k): str(v) for k, v in captions.items()}
        self.root = Path(root)
        self.folds = tuple(int(f) for f in folds)
        self.active_class_ids = tuple(int(c) for c in active_class_ids)
        self.min_area_px = int(min_area_px)
        self.dominance_ratio = float(dominance_ratio) if dominance_ratio is not None else None
        self.resize_to = int(resize_to)
        self.seed = int(seed)

        # Validate every active id is a real crop (1..18). active_classes also
        # validates the [1, 18] range so a misconfigured set fails fast.
        for cid in self.active_class_ids:
            if not 1 <= cid <= _N_PASTIS_CROPS:
                raise ValueError(f"active_class_ids must be in [1, {_N_PASTIS_CROPS}], got {cid}.")

        candidate_ids = self._fold_patch_ids()

        # Each kept sample stores the patch id and the precomputed region list so
        # ``__getitem__`` does not re-scan annotations. Patches with no valid
        # region are dropped (never given a spurious label).
        self._samples: list[tuple[str, list[tuple[int, int]]]] = []
        n_total_regions = 0
        for pid in candidate_ids:
            patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
            semantic = patch.get("semantic")
            parcel_ids = patch.get("instance")
            if semantic is None or parcel_ids is None:
                continue
            regions = extract_regions(
                np.asarray(parcel_ids),
                np.asarray(semantic),
                active_class_ids=self.active_class_ids,
                min_area_px=self.min_area_px,
            )
            if not regions:
                continue
            self._samples.append((str(pid), regions))
            n_total_regions += len(regions)

        n_patches = len(self._samples)
        self._mean_regions_per_patch = n_total_regions / n_patches if n_patches else 0.0

        logger.info(
            "region_category_dataset_init",
            folds=list(self.folds),
            active_class_ids=list(self.active_class_ids),
            min_area_px=self.min_area_px,
            dominance_ratio=self.dominance_ratio,
            n_candidates=len(candidate_ids),
            n_patches=n_patches,
            n_total_regions=n_total_regions,
            mean_regions_per_patch=round(self._mean_regions_per_patch, 4),
            resize_to=self.resize_to,
        )

    def _fold_patch_ids(self) -> list[int]:
        """Returns the candidate patch ids of ``self.folds``.

        When ``dominance_ratio`` is set, delegates to
        :class:`ml.data.pastis_filter.PastisFilter` (``dominance_ratio`` mode) to
        drop Meadow-over-dominated patches before region extraction. Otherwise it
        reads the full fold-to-patch map directly (no collapse, v2 default).

        Returns:
            List of PASTIS patch ids (ints) belonging to ``self.folds``.
        """
        # Imported here so the filter is reused without a hard module-load
        # coupling (and so tests can monkeypatch the symbol on this module).
        from ml.data.pastis_filter import PastisFilter

        if self.dominance_ratio is not None:
            target_classes = [c for c in self.active_class_ids if c != 1]
            filt = PastisFilter(
                pastis_root=self.root,
                target_classes=target_classes,
                mode="dominance_ratio",
                ratio=self.dominance_ratio,
                meadow_class=1,
            )
            return list(filt.filter_folds(self.folds))

        # No dominance collapse: read the official fold map from the same filter
        # infrastructure (it parses ``metadata.geojson`` into ``_fold_map``).
        filt = PastisFilter(
            pastis_root=self.root,
            target_classes=list(self.active_class_ids),
            mode="coverage",
        )
        ids: list[int] = []
        for fold in self.folds:
            ids.extend(filt._fold_map.get(fold, []))
        return ids

    @property
    def mean_regions_per_patch(self) -> float:
        """Average number of valid regions per kept patch (``> 1`` for v2)."""
        return self._mean_regions_per_patch

    def __len__(self) -> int:
        """Number of kept patches (each with at least one valid region)."""
        return len(self._samples)

    def _resize_composite(self, composite: np.ndarray) -> torch.Tensor:
        """Bilinearly resizes a ``(4, H, W)`` composite to ``(4, resize_to, *)``.

        Args:
            composite: float32 ``(4, H, W)`` composite in ``[0, 1]``.

        Returns:
            float32 tensor ``(4, resize_to, resize_to)``.
        """
        tensor = torch.from_numpy(composite).unsqueeze(0)  # (1, 4, H, W)
        resized = F.interpolate(
            tensor,
            size=(self.resize_to, self.resize_to),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).contiguous()  # (4, resize_to, resize_to)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Loads and transforms patch ``idx`` into the v2 multi-object contract.

        Args:
            idx: Index into the kept-sample list (supports Python negatives).

        Returns:
            Dict with:
                * ``image``: ``(4, resize_to, resize_to)`` float32 in ``[0, 1]``.
                * ``patch_id``: str.
                * ``caption``: str, the injected ``L_glo``.
                * ``region_cat_ids``: LongTensor ``(N,)`` of RAW PASTIS class ids
                  (1..18) present in the patch (one per valid region).

        Raises:
            IndexError: if ``idx`` is out of range.
            KeyError: if the patch has no caption in the injected dict.
        """
        if idx < 0:
            idx += len(self._samples)
        if not 0 <= idx < len(self._samples):
            raise IndexError(f"idx out of range: {idx}")

        pid, regions = self._samples[idx]
        if pid not in self.captions:
            raise KeyError(
                f"no caption for patch_id {pid}; captions must be materialized "
                "for every patch in the split (T1) before building the dataset."
            )

        patch = load_pastis_patch(pid, root=self.root, load_annotations=False)
        composite = peak_ndvi_composite(np.asarray(patch["s2"]))  # (4, H, W)
        image = self._resize_composite(composite)
        region_cat_ids = torch.tensor([cat for _, cat in regions], dtype=torch.long)  # (N,)
        return {
            "image": image,
            "patch_id": pid,
            "caption": self.captions[pid],
            "region_cat_ids": region_cat_ids,
        }


def collate_region_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates B patches into the cross-patch region-category batch for MPCL.

    Flattens the per-patch ``region_cat_ids`` lists into a single region axis so
    the MPCL (T3) can build ``P(i)`` = all regions of the batch that share a
    category, ACROSS patches. ``region_to_patch`` lets the trainer gather each
    region's visual CLS from its source patch (paper sec. 4.3: all regions of a
    patch share the patch CLS; the multi-positive signal is cross-patch).

    Args:
        items: list of ``__getitem__`` dicts (length B). Each has ``image``,
            ``patch_id``, ``caption`` and ``region_cat_ids`` ``(N_b,)``.

    Returns:
        Dict with:
            * ``images``: float32 ``(B, 4, H, W)``.
            * ``patch_ids``: list[str] of length B.
            * ``captions``: list[str] of length B (the injected ``L_glo``).
            * ``region_cat_ids``: LongTensor ``(sum_b N_b,)`` RAW PASTIS class id
              per flattened region.
            * ``region_to_patch``: LongTensor ``(sum_b N_b,)`` index in ``[0, B)``
              of the source patch of each region.

    Raises:
        ValueError: if ``items`` is empty.
    """
    if not items:
        raise ValueError("collate_region_batch received an empty batch.")

    images = torch.stack([it["image"] for it in items], dim=0)  # (B, 4, H, W)
    patch_ids = [str(it["patch_id"]) for it in items]
    captions = [str(it["caption"]) for it in items]

    region_cat_chunks: list[torch.Tensor] = []
    region_to_patch_chunks: list[torch.Tensor] = []
    for patch_index, it in enumerate(items):
        cat_ids = it["region_cat_ids"]
        n_regions = int(cat_ids.numel())
        if n_regions == 0:
            continue
        region_cat_chunks.append(cat_ids.to(torch.long))
        region_to_patch_chunks.append(torch.full((n_regions,), patch_index, dtype=torch.long))

    if region_cat_chunks:
        region_cat_ids = torch.cat(region_cat_chunks, dim=0)
        region_to_patch = torch.cat(region_to_patch_chunks, dim=0)
    else:
        region_cat_ids = torch.empty((0,), dtype=torch.long)
        region_to_patch = torch.empty((0,), dtype=torch.long)

    return {
        "images": images,
        "patch_ids": patch_ids,
        "captions": captions,
        "region_cat_ids": region_cat_ids,
        "region_to_patch": region_to_patch,
    }


__all__ = [
    "RegionCategoryPairDataset",
    "active_classes",
    "assert_disjoint_folds",
    "collate_region_batch",
    "extract_regions",
]
