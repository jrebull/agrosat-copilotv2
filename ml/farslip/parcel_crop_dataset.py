"""Per-parcel crop dataset over real PASTIS-R (US-036-b, parcel-level FarSLIP).

This module is the parcel-level counterpart of
:mod:`ml.farslip.region_category_dataset` (which exposes the WHOLE patch and makes
every parcel share the patch CLS -- the 1-CLS-per-patch bottleneck). Here each
PARCEL becomes its own training item: its instance mask is cropped from the
peak-NDVI 4-band composite (background zeroed), resized to the CLIP input size,
and paired with its own phenology caption. With one crop per item and
``region_to_patch = arange(B)``, the existing faithful trainer step
(:meth:`ml.farslip.distill.FarSLIPDistillationTrainer.step_faithful_v2`) consumes
this unchanged: the ``student_cls[region_to_patch]`` indexing becomes the identity,
so every parcel gets its OWN CLS in the MPCL. That is the degree of freedom that
breaks the ~4-class ceiling of the patch-level model.

Pipeline per parcel:

    load_pastis_patch(pid)        -> s2 (T,10,128,128), instance, semantic
    peak_ndvi_composite(s2)       -> img4 (4,128,128) [0,1] (REUSED)
    extract_regions(...)          -> [(instance_id, category_id)] (REUSED)
    crop instance bbox + zero bg  -> (4, h, w) -> resize 224
    captions[parcel_id]           -> per-parcel phenology caption (injected)

Only real PASTIS-R French data; official spatial folds; conventions: torch/numpy
at the data boundary, structlog, type hints, English docstrings, Spanish prose,
no emojis.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
    peak_ndvi_composite,
)
from ml.farslip.region_category_dataset import (
    _ALL_ACTIVE_CLASS_IDS,
    _DEFAULT_MIN_AREA_PX,
    extract_regions,
)
from ml.ingest.pastis_loader import load_pastis_patch

logger = structlog.get_logger(__name__)

_NO_PARCEL = 0
#: Minimum peak-NDVI reflectance for a parcel crop to carry usable signal. PASTIS
#: composites are raw (/10000, not cloud-masked), so deeply shadowed/tiny parcels
#: can be all-zero; those are dropped (honest filter, not zero-fill). The student
#: was trained on the same raw composite scale, so NO percentile stretch is
#: applied here (it would shift the distribution away from the trained student).
_MIN_CROP_SIGNAL: float = 1e-4


def _crop_parcel(
    composite: np.ndarray,
    instance: np.ndarray,
    instance_id: int,
) -> np.ndarray:
    """Crop one parcel from the composite, zeroing the background.

    Takes the parcel's instance mask, computes its bounding box, crops the 4-band
    composite to that box and sets every pixel outside the parcel to 0. This keeps
    only the parcel's signal (the panoptic polygon), so the student sees one crop
    per parcel instead of the whole patch.

    Args:
        composite: ``(4, H, W)`` float32 peak-NDVI composite in ``[0, 1]``.
        instance: ``(H, W)`` integer parcel instance mask (0 = no parcel).
        instance_id: the parcel instance id to crop.

    Returns:
        ``(4, h, w)`` float32 crop with background zeroed.
    """
    mask = instance == instance_id
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = composite[:, y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    crop[:, ~sub_mask] = 0.0
    return crop


def iter_parcel_crops(
    pid: str,
    composite: np.ndarray,
    instance: np.ndarray,
    regions: Sequence[tuple[int, int]],
    *,
    resize_to: int = 224,
) -> Iterator[tuple[str, int, tuple[int, int, int, int], torch.Tensor]]:
    """Yield ``(parcel_id, class_id, bbox, crop_tensor)`` for each region.

    Args:
        pid: patch id.
        composite: ``(4, H, W)`` peak-NDVI composite in ``[0, 1]``.
        instance: ``(H, W)`` parcel instance mask.
        regions: the patch's ``[(instance_id, category_id)]`` (from
            :func:`extract_regions`).
        resize_to: target side of the resized crop (CLIP input).

    Yields:
        ``(parcel_id="{pid}_{iid}", class_id, (y0,x0,y1,x1), crop (4,R,R))``.
    """
    for inst_id, cat_id in regions:
        mask = instance == inst_id
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        crop = _crop_parcel(composite, instance, inst_id)
        crop_t = _resize_crop(crop, resize_to)
        yield f"{pid}_{int(inst_id)}", int(cat_id), bbox, crop_t


def _resize_crop(crop: np.ndarray, resize_to: int) -> torch.Tensor:
    """Bilinearly resize a ``(4, h, w)`` crop to ``(4, resize_to, resize_to)``."""
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0)
    resized = F.interpolate(
        tensor, size=(resize_to, resize_to), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).contiguous()


class ParcelCropDataset(Dataset):
    """PASTIS-R dataset where each item is ONE parcel crop (not a whole patch).

    Each item exposes the parcel's peak-NDVI 4-band crop (background zeroed,
    resized), its RAW PASTIS ``class_id``, its ``parcel_id`` (``{patch}_{inst}``),
    its source ``patch_id`` and its injected phenology caption. The flat list of
    samples is one entry per valid parcel across the requested folds.

    Args:
        captions: injected ``{parcel_id: caption}`` (per-parcel phenology text).
            A parcel without a caption falls back to an empty string (logged).
        root: PASTIS-R root.
        folds: official PASTIS folds (spatial CV).
        active_class_ids: categories kept (default all 18 crops 1..18). Restrict
            this for the N-class sweep (e.g. ``INCREMENTAL_CURRICULUM[:N]``).
        min_area_px: minimum parcel area in pixels.
        resize_to: target side of the crop (CLIP input, default 224).
        max_patches: optional cap on patches scanned (smoke tests).
        seed: determinism seed.
        require_caption: if ``True``, keep ONLY parcels that have a non-empty
            caption. Use this when validating with a balanced caption SAMPLE
            (e.g. ~1000/class): otherwise the dominant classes contribute tens
            of thousands of empty-caption parcels that dilute ``L_glo`` with
            empty-text embeddings, biasing the result downward. With the full
            caption set this is a no-op. Default ``False`` (backward compatible).
        dominance_ratio: if not ``None``, apply the per-patch 3:1 Meadow
            ``PastisFilter`` (``mode="dominance_ratio"``) BEFORE extracting
            parcels, dropping whole patches where ``Meadow_px > ratio *
            second_px``. This is the legacy patch-level imbalance guard; at the
            parcel grain it is OPTIONAL because each parcel already carries its
            own true label (a rare parcel in a Meadow patch is no longer
            mislabelled). Exposed to A/B the filter's effect on the sweep curve.
            Default ``None`` (keep every patch -- the parcel grain handles the
            imbalance). ``meadow_class`` is fixed at 1 (PASTIS 20-class scheme).
    """

    def __init__(
        self,
        captions: dict[str, str],
        root: Path = _DEFAULT_PASTIS_ROOT,
        folds: Sequence[int] = (1, 2, 3),
        active_class_ids: tuple[int, ...] = _ALL_ACTIVE_CLASS_IDS,
        min_area_px: int = _DEFAULT_MIN_AREA_PX,
        resize_to: int = 224,
        max_patches: int | None = None,
        seed: int = 42,
        require_caption: bool = False,
        dominance_ratio: float | None = None,
    ) -> None:
        self.captions = {str(k): str(v) for k, v in captions.items()}
        self.root = Path(root)
        self.folds = tuple(int(f) for f in folds)
        self.active_class_ids = tuple(int(c) for c in active_class_ids)
        self.min_area_px = int(min_area_px)
        self.resize_to = int(resize_to)
        self.seed = int(seed)
        self.require_caption = bool(require_caption)
        self.dominance_ratio = float(dominance_ratio) if dominance_ratio is not None else None

        for cid in self.active_class_ids:
            if not 1 <= cid <= _N_PASTIS_CROPS:
                raise ValueError(f"active_class_ids must be in [1, {_N_PASTIS_CROPS}], got {cid}.")

        candidate_ids = self._fold_patch_ids()
        if max_patches is not None:
            candidate_ids = candidate_ids[: int(max_patches)]
        n_before_dominance = len(candidate_ids)
        if self.dominance_ratio is not None:
            candidate_ids = self._apply_dominance_filter(candidate_ids)

        # One sample per valid parcel: (parcel_id, patch_id, instance_id, class_id).
        # A parcel whose peak-NDVI crop is all-zero (no observable reflectance:
        # very small, deeply shadowed parcels) carries no signal for the student
        # and is dropped -- honest filtering, not zero-filling.
        self._samples: list[tuple[str, str, int, int]] = []
        n_empty = 0
        n_no_caption = 0
        for pid in candidate_ids:
            patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
            semantic = patch.get("semantic")
            instance = patch.get("instance")
            if semantic is None or instance is None:
                continue
            instance_arr = np.asarray(instance)
            regions = extract_regions(
                instance_arr,
                np.asarray(semantic),
                active_class_ids=self.active_class_ids,
                min_area_px=self.min_area_px,
            )
            if not regions:
                continue
            composite = peak_ndvi_composite(np.asarray(patch["s2"]))
            for inst_id, cat_id in regions:
                parcel_id = f"{pid}_{int(inst_id)}"
                if self.require_caption and not self.captions.get(parcel_id, ""):
                    n_no_caption += 1
                    continue
                crop = _crop_parcel(composite, instance_arr, inst_id)
                if float(crop.max()) <= _MIN_CROP_SIGNAL:
                    n_empty += 1
                    continue
                self._samples.append((parcel_id, str(pid), int(inst_id), int(cat_id)))

        logger.info(
            "parcel_crop_dataset_init",
            n_patches=len(candidate_ids),
            n_patches_before_dominance=n_before_dominance,
            n_patches_dropped_dominance=n_before_dominance - len(candidate_ids),
            dominance_ratio=self.dominance_ratio,
            n_parcels=len(self._samples),
            n_empty_dropped=n_empty,
            n_no_caption_dropped=n_no_caption,
            require_caption=self.require_caption,
            n_active_classes=len(self.active_class_ids),
            folds=list(self.folds),
            with_captions=sum(1 for s in self._samples if s[0] in self.captions),
        )

    def _apply_dominance_filter(self, candidate_ids: list[str]) -> list[str]:
        """Drop patches failing the 3:1 Meadow dominance rule (patch-level A/B).

        Reuses :meth:`ml.data.pastis_filter.PastisFilter._passes_dominance` on
        each candidate patch's semantic mask so the keep decision is byte-for-byte
        the same as the legacy patch-level filter (no reimplementation of the
        histogram). A patch is kept when ``Meadow_px <= dominance_ratio *
        second_px`` over its semantic channel. ``target_classes`` is restricted to
        the sweep's ``active_class_ids`` so the "second class" is one of the
        classes actually under evaluation.

        The filter is built with ``object.__new__`` (only the three attributes
        ``_passes_dominance`` reads are set) so this stays decoupled from
        ``PastisFilter.__init__``, which would require a ``metadata.geojson`` on
        disk that the per-parcel path does not otherwise need.

        Args:
            candidate_ids: patch ids (post fold-split, post ``max_patches``).

        Returns:
            The subset of ``candidate_ids`` that pass the 3:1 filter, order kept.
        """
        from ml.data.pastis_filter import PastisFilter

        filt = object.__new__(PastisFilter)
        filt.target_classes = set(self.active_class_ids)
        filt.ratio = float(self.dominance_ratio)  # type: ignore[arg-type]
        filt.meadow_class = 1
        kept: list[str] = []
        for pid in candidate_ids:
            patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
            semantic = patch.get("semantic")
            if semantic is None:
                continue
            passes, _ = filt._passes_dominance(np.asarray(semantic))
            if passes:
                kept.append(pid)
        return kept

    def _fold_patch_ids(self) -> list[str]:
        """Return sorted patch ids of the requested official folds."""
        from ml.ingest.pastis_dataset import pastis_fold_split

        split = pastis_fold_split(
            self.root,
            train_folds=self.folds,
            val_folds=(),
            test_folds=(),
        )
        return sorted(split["train"], key=lambda p: int(p))

    def __len__(self) -> int:
        """Number of valid parcels across the requested folds."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and crop parcel ``idx`` into the per-parcel training contract.

        Returns:
            Dict with ``image`` ``(4, R, R)`` float32, ``parcel_id`` str,
            ``patch_id`` str, ``class_id`` int (RAW PASTIS 1..18), ``caption`` str,
            ``bbox`` ``(y0, x0, y1, x1)``.
        """
        if idx < 0:
            idx += len(self._samples)
        if not 0 <= idx < len(self._samples):
            raise IndexError(f"idx out of range: {idx}")

        parcel_id, pid, inst_id, cat_id = self._samples[idx]
        patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
        composite = peak_ndvi_composite(np.asarray(patch["s2"]))  # (4, H, W)
        instance = np.asarray(patch["instance"])
        crop = _crop_parcel(composite, instance, inst_id)
        image = _resize_crop(crop, self.resize_to)

        mask = instance == inst_id
        ys, xs = np.where(mask)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        return {
            "image": image,
            "parcel_id": parcel_id,
            "patch_id": pid,
            "class_id": cat_id,
            "caption": self.captions.get(parcel_id, ""),
            "bbox": bbox,
        }


def collate_parcel_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate per-parcel items into the trainer's faithful-step contract.

    Produces a batch where each parcel is its own region: ``images`` stacks the
    per-parcel crops, ``region_cat_ids`` are the RAW PASTIS classes, and
    ``region_to_patch = arange(B)`` so the trainer's
    ``student_cls[region_to_patch]`` indexing is the identity (each parcel uses its
    own CLS in the MPCL -- the whole point of the parcel-level model).

    Args:
        items: list of :meth:`ParcelCropDataset.__getitem__` outputs.

    Returns:
        Dict with ``images`` ``(B,4,R,R)``, ``region_cat_ids`` ``(B,)``,
        ``region_to_patch`` ``(B,)`` ``= arange(B)``, ``parcel_ids`` list,
        ``patch_ids`` list, ``captions`` list, ``class_ids`` ``(B,)``.
    """
    images = torch.stack([it["image"] for it in items], dim=0)
    region_cat_ids = torch.tensor([it["class_id"] for it in items], dtype=torch.long)
    region_to_patch = torch.arange(len(items), dtype=torch.long)
    return {
        "images": images,
        "region_cat_ids": region_cat_ids,
        "region_to_patch": region_to_patch,
        "parcel_ids": [it["parcel_id"] for it in items],
        "patch_ids": [it["patch_id"] for it in items],
        "captions": [it["caption"] for it in items],
        "class_ids": region_cat_ids,
    }
