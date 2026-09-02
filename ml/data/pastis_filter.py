"""ml/data/pastis_filter.py
==========================
Patch-level filter for PASTIS-R to reduce class imbalance before
FarSLIP feature extraction.

**Problem**: PASTIS-R patches are dominated by Meadow (class 1) and
Background (class 0), which together can cover >80 % of a patch.
Training or extracting features on these patches pollutes the embedding
space with uninformative signal, hurting downstream FarSLIP performance
on the minority crop classes.

**Solution**: discard any patch where the *combined pixel coverage of a
configurable set of "target" classes* is below a threshold (default 50 %).
Equivalently: keep only patches where at least ``min_coverage`` of valid
pixels belong to the classes we care about.

Usage
-----
Standalone filter (returns a filtered list of patch IDs)::

    from ml.data.pastis_filter import PastisFilter, PASTIS_CLASS_NAMES

    f = PastisFilter(
        pastis_root    = Path('data/PASTIS-R'),
        target_classes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        min_coverage   = 0.50,
        ignore_index   = 255,
    )
    kept_ids = f.filter_folds([1, 2, 3])
    print(f'Kept {len(kept_ids)} / {f.total_scanned} patches')

Drop-in replacement for PASTISSegmentationDataset::

    from ml.data.pastis_filter import FilteredPASTISDataset

    train_ds = FilteredPASTISDataset(
        root           = PASTIS_ROOT,
        folds          = [1, 2, 3],
        target_classes = [2, 3, 4, 5, 6, 7],   # e.g. only cereals + oilseeds
        min_coverage   = 0.50,
        collapse_time  = None,
        n_timesteps    = 10,
        augment        = True,
    )

Integration with existing repo dataset::

    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.data.pastis_filter import PastisFilter

    f = PastisFilter(pastis_root, target_classes=TARGET_CLASSES, min_coverage=0.50)
    kept_ids = f.filter_folds([1, 2, 3])
    # pass kept_ids to your custom sampler or Subset
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog
import torch
from torch.utils.data import Dataset, Subset

logger = structlog.get_logger(__name__)

# Class ids excluded by literal value from the dominance-ratio histogram so the
# 2nd-class count is never contaminated by non-crop pixels. This is independent
# of ``ignore_index`` (which the legacy coverage mode uses for the denominator).
_BACKGROUND_CLASS: int = 0
_VOID_CLASS: int = 19

# ─────────────────────────────────────────────────────────────────────────────
# PASTIS-R class registry
# ─────────────────────────────────────────────────────────────────────────────

# Full 20-class PASTIS-R label set (indices 0-19)
PASTIS_CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits, vegetables, flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
    19: "Void label",
}

# Semantic-18 remapping (repo's PASTISSegmentationDataset target='semantic18')
# Background (0) and Void (19) are excluded; remaining 18 classes → 0-17
SEMANTIC18_CLASS_NAMES: dict[int, str] = {
    i: name
    for i, (_, name) in enumerate(
        {k: v for k, v in PASTIS_CLASS_NAMES.items() if k not in (0, 19)}.items()
    )
}

# Convenience group: all non-background, non-meadow, non-void crop classes
# in the full 20-class scheme
ALL_CROP_CLASSES: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# filtering classes based on the number of patches that would be kept at 50% coverage threshold
CLASSES_OF_INTEREST: list[int] = [2, 3, 8]

# HCAT Level-1 groups → class indices (full 20-class scheme)
HCAT_GROUPS: dict[str, list[int]] = {
    "CEREALS": [2, 3, 4, 6, 10, 11, 17, 18],
    "OILSEEDS": [5, 7],
    "LEGUMES": [14, 15],
    "ROOT_CROPS": [9, 13],
    "PERMANENT_WOODY": [8, 16],
    "OTHER": [1, 12],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core filter
# ─────────────────────────────────────────────────────────────────────────────


class PastisFilter:
    """Scans PASTIS-R annotation files and returns patch IDs that pass
    a minimum pixel-coverage threshold for a configurable set of classes.

    Args:
        pastis_root:    root directory containing ``ANNOTATIONS/`` and
                        ``metadata.geojson``.
        target_classes: list of class indices whose *combined* coverage
                        must exceed ``min_coverage``.  Defaults to all
                        non-background, non-meadow, non-void crop classes
                        (``ALL_CROP_CLASSES``).
        min_coverage:   minimum fraction of *valid* pixels that must
                        belong to ``target_classes`` (default 0.50 = 50 %).
        ignore_index:   pixel value to exclude from the denominator
                        (void / padded pixels).  Use 255 for the repo's
                        ``PASTISSegmentationDataset`` or 19 for the raw
                        PASTIS-R annotations.
        annotation_key: which annotation channel to read.  PASTIS-R stores
                        ``TARGET_<id>.npy`` as ``(num_tasks, H, W)``; index
                        0 = semantic labels.
        verbose:        log per-patch keep/discard decisions.
        mode:           filtering strategy. ``"coverage"`` (default, legacy) or
                        ``"dominance_ratio"`` (the 3:1 Meadow per-patch rule).
        ratio:          maximum Meadow-to-second-class ratio kept in
                        ``dominance_ratio`` mode (default 3.0 -> the 3:1 rule).
                        Unused in ``coverage`` mode.
        meadow_class:   class id treated as the dominant class to bound in
                        ``dominance_ratio`` mode (default 1 = Meadow in the
                        PASTIS 20-class scheme). Unused in ``coverage`` mode.

    The legacy ``coverage`` mode keeps a patch when the COMBINED coverage of
    ``target_classes`` over valid (``!= ignore_index``) pixels is
    ``>= min_coverage``. The new ``dominance_ratio`` mode keeps a patch when
    ``Meadow_px <= ratio * second_px``, where ``second_px`` is the largest
    count, IN THAT PATCH, among ``target_classes`` excluding Meadow,
    Background (0) and Void (19) -- dropping patches over-dominated by Meadow.
    Both modes return the same ``tuple[bool, float]`` from ``_passes``.
    """

    def __init__(
        self,
        pastis_root: Path | str,
        target_classes: list[int] | None = None,
        min_coverage: float = 0.50,
        ignore_index: int = 255,
        annotation_key: int = 0,
        verbose: bool = False,
        *,
        mode: Literal["coverage", "dominance_ratio"] = "coverage",
        ratio: float = 3.0,
        meadow_class: int = 1,
    ) -> None:
        self.root = Path(pastis_root)
        self.target_classes = set(
            target_classes if target_classes is not None else ALL_CROP_CLASSES
        )
        self.min_coverage = min_coverage
        self.ignore_index = ignore_index
        self.annotation_key = annotation_key
        self.verbose = verbose
        self.mode = mode
        self.ratio = ratio
        self.meadow_class = meadow_class
        self.total_scanned = 0

        meta_path = self.root / "metadata.geojson"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.geojson not found in {self.root}")
        with open(meta_path) as f:
            self._meta = json.load(f)

        # Build fold → patch_id mapping
        self._fold_map: dict[int, list[int]] = {}
        for feat in self._meta["features"]:
            fold = feat["properties"]["Fold"]
            pid = feat["properties"]["ID_PATCH"]
            self._fold_map.setdefault(fold, []).append(pid)

    # ------------------------------------------------------------------
    def _load_mask(self, pid: int) -> np.ndarray:
        """Load semantic label mask for patch ``pid`` → (H, W) int32."""
        path = self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        mask: np.ndarray = np.load(str(path))
        if mask.ndim == 3:
            mask = mask[self.annotation_key]
        return mask.astype(np.int32)

    # ------------------------------------------------------------------
    def _passes(self, mask: np.ndarray) -> tuple[bool, float]:
        """Return ``(passes, metric)`` for a single ``(H, W)`` mask.

        Dispatches on ``self.mode``. In ``coverage`` mode (legacy, default)
        ``metric`` is the coverage fraction. In ``dominance_ratio`` mode
        ``metric`` is the observed ``Meadow_px / second_px`` ratio. The 2-tuple
        shape is PRESERVED so ``filter_patch_ids`` / ``filter_folds`` /
        ``coverage_stats`` are unaffected.

        Args:
            mask: semantic label mask of shape ``(H, W)``.

        Returns:
            Tuple ``(passes, metric)`` where ``passes`` is the keep decision and
            ``metric`` is mode-dependent (coverage fraction or observed ratio).
        """
        if self.mode == "dominance_ratio":
            return self._passes_dominance(mask)
        # --- legacy coverage branch (unchanged) ---
        valid_mask = mask != self.ignore_index
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return False, 0.0
        n_target = np.isin(mask, list(self.target_classes)).sum()
        coverage = float(n_target) / float(n_valid)
        return coverage >= self.min_coverage, coverage

    # ------------------------------------------------------------------
    def _passes_dominance(self, mask: np.ndarray) -> tuple[bool, float]:
        """3:1 Meadow per-patch dominance filter.

        Builds a 20-bin histogram via ``np.bincount(mask.ravel(), minlength=20)``
        over the semantic channel. Keeps the patch when
        ``Meadow_px <= self.ratio * second_px``. ``second_px`` is the largest
        count among ``self.target_classes`` EXCLUDING ``meadow_class``,
        Background (0) and Void (19); Background and Void are excluded by literal
        id (NOT via ``ignore_index``) so the ratio is not contaminated. The
        ``<=`` comparison is inclusive, so an exact 3:1 patch is kept.

        Edge cases (explicit policy, tested):
            - ``Meadow_px == 0``: keep (patch not Meadow-dominated), metric 0.0.
            - ``Meadow_px > 0`` and ``second_px == 0`` (no competing target
              class): drop, metric ``inf``.

        Args:
            mask: semantic label mask of shape ``(H, W)`` with raw class ids
                (Background 0, crops 1-18, Void 19).

        Returns:
            Tuple ``(keep, ratio_observed)`` where ``ratio_observed`` equals
            ``Meadow_px / second_px`` (``inf`` when ``second_px == 0``).
        """
        hist = np.bincount(mask.ravel(), minlength=20)
        meadow_px = int(hist[self.meadow_class])
        if meadow_px == 0:
            return True, 0.0
        # 2nd-largest class among target_classes, excluding Meadow / bg / void.
        excluded = {self.meadow_class, _BACKGROUND_CLASS, _VOID_CLASS}
        candidates = [c for c in self.target_classes if c not in excluded]
        second_px = max((int(hist[c]) for c in candidates), default=0)
        if second_px == 0:
            return False, float("inf")
        ratio_obs = meadow_px / second_px
        return meadow_px <= self.ratio * second_px, ratio_obs

    # ------------------------------------------------------------------
    def filter_patch_ids(self, patch_ids: list[int]) -> list[int]:
        """Filter an arbitrary list of patch IDs.

        Args:
            patch_ids: list of PASTIS-R patch IDs to evaluate.

        Returns:
            Subset of ``patch_ids`` that pass the coverage filter.
        """
        kept = []
        self.total_scanned += len(patch_ids)
        for pid in patch_ids:
            mask = self._load_mask(pid)
            passes, metric = self._passes(mask)
            if passes:
                kept.append(pid)
                if self.verbose:
                    logger.debug("keep_patch", pid=pid, mode=self.mode, metric=metric)
            else:
                if self.verbose:
                    logger.debug("drop_patch", pid=pid, mode=self.mode, metric=metric)
        return kept

    # ------------------------------------------------------------------
    def filter_folds(self, folds: Sequence[int]) -> list[int]:
        """Filter all patches belonging to ``folds``.

        Args:
            folds: list of fold indices (1-5 for PASTIS-R).

        Returns:
            Filtered list of patch IDs across all requested folds.
        """
        all_ids = []
        for fold in folds:
            all_ids.extend(self._fold_map.get(fold, []))

        kept = self.filter_patch_ids(all_ids)
        pct = 100 * len(kept) / max(len(all_ids), 1)
        log_kwargs: dict[str, Any] = {
            "folds": list(folds),
            "target_classes": sorted(self.target_classes),
            "mode": self.mode,
            "kept": len(kept),
            "total": len(all_ids),
            "pct": round(pct, 1),
        }
        if self.mode == "dominance_ratio":
            log_kwargs["ratio"] = self.ratio
            log_kwargs["meadow_class"] = self.meadow_class
        else:
            log_kwargs["min_coverage"] = self.min_coverage
        logger.info("pastis_filter_folds", **log_kwargs)
        return kept

    # ------------------------------------------------------------------
    def coverage_stats(self, folds: Sequence[int]) -> dict[str, Any]:
        """Compute the per-patch metric distribution without filtering (EDA).

        In ``coverage`` mode the metric is the coverage fraction (useful for
        choosing ``min_coverage`` before committing). In ``dominance_ratio``
        mode the metric is the observed ``Meadow_px / second_px`` ratio (which
        can be ``inf``); the percentile / ``pct_above_*`` keys are then computed
        over that ratio and the coverage-specific thresholds are not meaningful.

        Args:
            folds: list of fold indices (1-5 for PASTIS-R).

        Returns:
            Dict with the raw per-patch metric array plus summary statistics.
        """
        all_ids = []
        for fold in folds:
            all_ids.extend(self._fold_map.get(fold, []))

        metrics: list[float] = []
        for pid in all_ids:
            mask = self._load_mask(pid)
            _, cov = self._passes(mask)
            metrics.append(cov)

        coverages = np.array(metrics)
        return {
            "n_patches": len(coverages),
            "mean": float(coverages.mean()),
            "median": float(np.median(coverages)),
            "p25": float(np.percentile(coverages, 25)),
            "p75": float(np.percentile(coverages, 75)),
            "pct_above_50": float((coverages >= 0.50).mean()),
            "pct_above_30": float((coverages >= 0.30).mean()),
            "pct_above_70": float((coverages >= 0.70).mean()),
            "coverages": coverages,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in filtered dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FilteredPASTISDataset(Dataset):
    """Filtered wrapper around the repo's ``PASTISSegmentationDataset``.

    Applies :class:`PastisFilter` at construction time and exposes only
    the patches that pass the coverage threshold.  All other parameters
    are forwarded to the underlying dataset unchanged.

    Args:
        root:           PASTIS-R root directory.
        folds:          fold indices to include.
        target_classes: class indices whose combined coverage must exceed
                        ``min_coverage``.  Defaults to ``ALL_CROP_CLASSES``
                        (everything except Background, Meadow, Void).
        min_coverage:   minimum valid-pixel fraction for target classes.
        ignore_index:   void pixel value (255 for repo dataset, 19 for raw).
        verbose:        log filter decisions.
        mode:           filtering strategy forwarded to :class:`PastisFilter`.
                        ``"coverage"`` (default, legacy) or
                        ``"dominance_ratio"`` (the 3:1 Meadow per-patch rule).
        ratio:          max Meadow-to-second-class ratio in ``dominance_ratio``
                        mode (default 3.0). Forwarded to :class:`PastisFilter`.
        meadow_class:   class id bounded in ``dominance_ratio`` mode (default
                        1 = Meadow). Forwarded to :class:`PastisFilter`.
        **dataset_kwargs: forwarded to ``PASTISSegmentationDataset``
                          (e.g. ``collapse_time``, ``n_timesteps``, ``target``).

    Example::

        ds = FilteredPASTISDataset(
            root           = PASTIS_ROOT,
            folds          = [1, 2, 3],
            target_classes = [2, 3, 4, 5, 6, 7],   # cereals + oilseeds only
            min_coverage   = 0.50,
            collapse_time  = None,
            n_timesteps    = 10,
            target         = 'semantic18',
        )
    """

    def __init__(
        self,
        root: Path | str,
        folds: Sequence[int],
        target_classes: list[int] | None = None,
        min_coverage: float = 0.50,
        ignore_index: int = 255,
        verbose: bool = False,
        *,
        mode: Literal["coverage", "dominance_ratio"] = "coverage",
        ratio: float = 3.0,
        meadow_class: int = 1,
        **dataset_kwargs: Any,
    ) -> None:
        from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

        root = Path(root)

        # ── Run the filter ────────────────────────────────────────────────
        filt = PastisFilter(
            pastis_root=root,
            target_classes=target_classes,
            min_coverage=min_coverage,
            ignore_index=ignore_index,
            verbose=verbose,
            mode=mode,
            ratio=ratio,
            meadow_class=meadow_class,
        )
        kept_ids = filt.filter_folds(folds)
        self._kept_ids = kept_ids
        self._n_total = filt.total_scanned
        self._n_kept = len(kept_ids)
        self._coverage = self._n_kept / max(self._n_total, 1)

        # ── Build full dataset then subset to kept patches ────────────────
        full_ds = PASTISSegmentationDataset(root=root, folds=list(folds), **dataset_kwargs)

        # Map kept patch IDs → dataset indices
        # PASTISSegmentationDataset stores patch_ids in the same order as
        # metadata.geojson filtered by fold — we replicate that order.
        meta_path = root / "metadata.geojson"
        with open(meta_path) as f:
            meta = json.load(f)
        ordered_ids = [
            feat["properties"]["ID_PATCH"]
            for feat in meta["features"]
            if feat["properties"]["Fold"] in folds
        ]
        kept_set = set(kept_ids)
        kept_indices = [i for i, pid in enumerate(ordered_ids) if pid in kept_set]

        self._dataset = Subset(full_ds, kept_indices)

        logger.info(
            "filtered_pastis_dataset",
            folds=list(folds),
            target_classes=sorted(target_classes or ALL_CROP_CLASSES),
            mode=mode,
            kept=self._n_kept,
            total=self._n_total,
            pct=round(100 * self._coverage, 1),
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item: tuple[torch.Tensor, torch.Tensor] = self._dataset[idx]
        return item

    @property
    def n_total(self) -> int:
        """Total patches scanned before filtering."""
        return self._n_total

    @property
    def n_kept(self) -> int:
        """Patches that passed the filter."""
        return self._n_kept
