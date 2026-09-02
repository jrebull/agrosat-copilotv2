"""Image-text pair dataset over real PASTIS-R for the incremental protocol (US-036).

This module builds the *new* infrastructure of the n-class incremental protocol
(Stage-1 4-class -> Stage-2 18-class) on top of the **real French PASTIS-R**
dataset, with a cardinality curriculum and the 3:1 Meadow filter. It is the
frozen contract consumed by ``scripts/train_incremental.py`` (US-036 ml/B).

Pipeline per patch (1 pair per patch):

    load_pastis_patch(pid) -> s2 (T, 10, 128, 128) int16
    peak_ndvi_composite    -> img4 (4, 128, 128) float32 [0,1], bands [B02,B03,B04,B08]
    dominant_class         -> PASTIS class_id (1..18) restricted to the active set
    category_id            -> index of the dominant class inside active_classes(n)

Scope (critical, ordered by the user 2026-06-07): ONLY real French PASTIS-R.
No Italian / synthetic / placeholder data, no ``FarSLIPDataset``, no
``data/farslip_pairs``, no ``expand_to_cap`` / CAP bridge. PASTIS classes are
used directly (1..18) and ``n_regions`` is always 1.

Reuses, without modifying:
    - :func:`ml.ingest.pastis_loader.load_pastis_patch` (real S2 loading),
    - :class:`ml.data.pastis_filter.PastisFilter` (3:1 ``dominance_ratio`` rule),
    - :func:`ml.features.phenology_class_prototypes.load_class_prototype_embeddings`
      (the 18x384 MiniLM prototypes of US-033 and their class_ids).

Project convention: ``torch``/``numpy`` only at the data boundary; logging via
``structlog``; no pandas; type hints everywhere; docstrings in English.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.ingest.pastis_loader import load_pastis_patch

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = structlog.get_logger(__name__)

#: Cardinality curriculum (real n_pixels, larger -> smaller) confirmed by the
#: Avance-1 EDA. Stage-1 uses the first 4 (Meadow, Corn, Soft winter wheat,
#: Grapevine, in dominance order); the order of the 18 fixes which classes each
#: intermediate ``n_classes`` activates. IDs are raw PASTIS class_ids (1..18).
INCREMENTAL_CURRICULUM: tuple[int, ...] = (
    1,
    3,
    2,
    8,
    4,
    14,
    5,
    16,
    10,
    6,
    15,
    7,
    11,
    9,
    12,
    17,
    13,
    18,
)

#: Band indices in the PASTIS-R 10-band layout (B02..B12) for the 4-band
#: composite. Matches ``ml.data.pastis_seg_dataset`` (DATA_S2 = (T,10,128,128))
#: and ``ml.features.phenology_class_prototypes`` (B04=idx2, B08=idx6 for NDVI).
_PASTIS_B02: int = 0
_PASTIS_B03: int = 1
_PASTIS_B04: int = 2
_PASTIS_B08: int = 6

#: 4-band composite order returned by ``peak_ndvi_composite`` (B02,B03,B04,B08).
_COMPOSITE_BANDS: tuple[int, ...] = (_PASTIS_B02, _PASTIS_B03, _PASTIS_B04, _PASTIS_B08)

#: PASTIS reflectance scale: the int16 values live in 0..10000 (S2 L2A).
_S2_SCALE: float = 10000.0

#: Non-agronomic classes excluded from the dominant-class histogram.
_BACKGROUND_CLASS: int = 0
_VOID_CLASS: int = 19

#: Number of agronomic PASTIS classes (1..18). Curriculum length and the cap of
#: ``n_classes``.
_N_PASTIS_CROPS: int = 18

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_DEFAULT_PROTO_PATH = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"


def active_classes(n_classes: int) -> tuple[int, ...]:
    """Returns the first ``n_classes`` PASTIS classes of the curriculum.

    Args:
        n_classes: Number of active classes (4 in Stage-1, 18 in Stage-2).

    Returns:
        Tuple of PASTIS class_ids (1..18) in curriculum order.

    Raises:
        ValueError: if ``n_classes`` is not in ``[1, 18]``.
    """
    if not 1 <= n_classes <= _N_PASTIS_CROPS:
        raise ValueError(f"n_classes must be in [1, {_N_PASTIS_CROPS}], received {n_classes}.")
    return INCREMENTAL_CURRICULUM[:n_classes]


def peak_ndvi_composite(s2: np.ndarray) -> np.ndarray:
    """Selects the peak mean-NDVI timestep and returns a 4-band composite.

    For each timestep it computes the spatial mean NDVI
    ``NDVI_t = (B08 - B04) / (B08 + B04)`` (B04=idx2, B08=idx6 in the 10-band
    layout), clamps NDVI to ``[-1, 1]`` (PASTIS does not mask clouds), takes the
    robust ``nanmean`` over space, and picks ``t* = argmax`` of that mean. It
    returns ``s2[t*, [B02,B03,B04,B08], :, :]`` scaled to ``[0, 1]`` (/10000 and
    clipped).

    Args:
        s2: int16 tensor ``(T, 10, H, W)`` (PASTIS 10-band layout).

    Returns:
        float32 ``(4, H, W)`` in ``[0, 1]``, bands ``[B02, B03, B04, B08]``.

    Raises:
        ValueError: if ``s2`` is not a 4-D ``(T, 10, H, W)`` tensor.
    """
    if s2.ndim != 4 or s2.shape[1] <= max(_COMPOSITE_BANDS):
        raise ValueError(f"s2 must be (T, >=7, H, W) PASTIS 10-band, received shape {s2.shape}.")
    s2f = s2.astype(np.float32)
    red = s2f[:, _PASTIS_B04]  # (T, H, W)
    nir = s2f[:, _PASTIS_B08]
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(np.abs(denom) > 1e-6, (nir - red) / denom, np.nan)
    # NDVI is physically valid in [-1, 1]; out-of-range values are cloud/shadow
    # artifacts (PASTIS is not cloud-masked) -> drop them from the spatial mean.
    ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # A fully-invalid timestep yields an all-NaN slice; nanmean warns and
        # returns NaN, which is mapped to -inf below so it never wins the argmax.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_ndvi = np.nanmean(ndvi.reshape(ndvi.shape[0], -1), axis=1)  # (T,)
    # A timestep that is entirely invalid yields NaN; treat it as -inf so the
    # argmax never selects it (unless every timestep is invalid, then t*=0).
    mean_ndvi = np.where(np.isfinite(mean_ndvi), mean_ndvi, -np.inf)
    t_star = int(np.argmax(mean_ndvi))
    composite = s2f[t_star][list(_COMPOSITE_BANDS)] / _S2_SCALE  # (4, H, W)
    composite = np.clip(composite, 0.0, 1.0)
    return np.ascontiguousarray(composite, dtype=np.float32)


def dominant_class(semantic: np.ndarray, active: tuple[int, ...]) -> int | None:
    """PASTIS dominant class of the patch restricted to the active classes.

    Excludes Background (0) and Void (19); counts pixels per class via a 20-bin
    histogram and returns the ``argmax`` ONLY over ``active``. Returns ``None``
    if the patch contains no active class (the patch is then dropped from the
    dataset, never given a spurious label).

    Args:
        semantic: mask ``(H, W)`` with PASTIS class_id (0..19).
        active: active PASTIS classes (from :func:`active_classes`).

    Returns:
        Dominant PASTIS class_id, or ``None`` if no active class is present.
    """
    hist = np.bincount(semantic.ravel().astype(np.int64), minlength=20)
    best_class: int | None = None
    best_count = 0
    for cid in active:
        if cid in (_BACKGROUND_CLASS, _VOID_CLASS):
            continue
        count = int(hist[cid])
        if count > best_count:
            best_count = count
            best_class = cid
    return best_class


class PastisPairDataset(Dataset):
    """Image-text pair dataset over real PASTIS-R (US-036).

    Each item: ``{"image": (4, 224, 224) float32, "region_id": 0-d long (==0),
    "category_id": 0-d long in [0, n_classes-1]}``. The ``category_id`` indexes
    the prototype bank ``proto_active`` (dominant class -> index inside the
    active set).

    Reuses :class:`ml.data.pastis_filter.PastisFilter` (3:1 ``dominance_ratio``)
    and :func:`ml.ingest.pastis_loader.load_pastis_patch`; the peak-NDVI
    composite and the dominant class are the only new computations. Patches with
    no active class are excluded at construction time.

    Args:
        n_classes: active classes (cardinality curriculum).
        root: PASTIS-R root (default ``data/PASTIS-R``).
        folds: official PASTIS folds (spatial CV; default ``(1, 2, 3)``).
        ratio: Meadow:2nd-class ratio of the 3:1 filter (default 3.0).
        resize_to: target side of the composite (default 224, CLIP).
        seed: determinism seed (default 42).
    """

    def __init__(
        self,
        n_classes: int,
        root: Path = _DEFAULT_PASTIS_ROOT,
        folds: Sequence[int] = (1, 2, 3),
        ratio: float = 3.0,
        resize_to: int = 224,
        seed: int = 42,
    ) -> None:
        # Import here so the 3:1 filter is reused without a hard module-load
        # coupling (and so tests can monkeypatch the symbol on this module).
        from ml.data.pastis_filter import PastisFilter

        self.n_classes = int(n_classes)
        self.root = Path(root)
        self.folds = tuple(int(f) for f in folds)
        self.ratio = float(ratio)
        self.resize_to = int(resize_to)
        self.seed = int(seed)

        self._active: tuple[int, ...] = active_classes(self.n_classes)

        # 3:1 Meadow dominance filter restricted to the active crops, excluding
        # Meadow from the "target" set so it never counts as the 2nd class.
        target_classes = [c for c in self._active if c != 1]
        filt = PastisFilter(
            pastis_root=self.root,
            target_classes=target_classes,
            mode="dominance_ratio",
            ratio=self.ratio,
            meadow_class=1,
        )
        kept_ids = filt.filter_folds(self.folds)

        # Keep only patches whose dominant class (restricted to active) is not
        # None; record both the patch_id and its precomputed category_id.
        self._samples: list[tuple[str, int]] = []
        for pid in kept_ids:
            patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
            semantic = patch.get("semantic")
            if semantic is None:
                continue
            dom = dominant_class(np.asarray(semantic), self._active)
            if dom is None:
                continue
            category_id = self._active.index(dom)
            self._samples.append((str(pid), category_id))

        logger.info(
            "pastis_pair_dataset_init",
            n_classes=self.n_classes,
            active=list(self._active),
            folds=list(self.folds),
            kept_filter=len(kept_ids),
            kept_active=len(self._samples),
            resize_to=self.resize_to,
        )

    def __len__(self) -> int:
        """Number of pairs (patches with an active dominant class)."""
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Loads and transforms pair ``idx`` into the trainer contract.

        Args:
            idx: Index into the kept-sample list.

        Returns:
            Dict with ``image`` ``(4, resize_to, resize_to)`` float32,
            ``region_id`` 0-d long (==0) and ``category_id`` 0-d long in
            ``[0, n_classes-1]``.

        Raises:
            IndexError: if ``idx`` is out of range.
        """
        if idx < 0:
            idx += len(self._samples)
        if not 0 <= idx < len(self._samples):
            raise IndexError(f"idx out of range: {idx}")

        pid, category_id = self._samples[idx]
        patch = load_pastis_patch(pid, root=self.root, load_annotations=False)
        composite = peak_ndvi_composite(np.asarray(patch["s2"]))  # (4, H, W)
        image = self._resize_composite(composite)
        return {
            "image": image,
            "region_id": torch.tensor(0, dtype=torch.long),
            "category_id": torch.tensor(int(category_id), dtype=torch.long),
        }


def create_incremental_dataset(
    n_classes: int,
    *,
    root: Path = _DEFAULT_PASTIS_ROOT,
    folds: Sequence[int] = (1, 2, 3),
    ratio: float = 3.0,
    seed: int = 42,
    prototype_path: Path | None = None,
) -> tuple[PastisPairDataset, int, int, torch.Tensor]:
    """Builds the N-class dataset and its prototypes for the FarSLIP trainer.

    Loads the 18 US-033 prototypes (MiniLM-384) via
    :func:`ml.features.phenology_class_prototypes.load_class_prototype_embeddings`
    and selects DIRECTLY the rows of the active classes (no ``expand_to_cap``).
    ``n_regions`` is always 1. The dominant-class -> active-index map and the
    prototype row order share the same ``active_classes(n)`` ordering, so a
    sample's ``category_id`` indexes its own prototype.

    Args:
        n_classes: active classes (4 Stage-1, 18 Stage-2).
        root: PASTIS-R root.
        folds: official folds (spatial CV).
        ratio: 3:1 filter ratio.
        seed: determinism.
        prototype_path: override of the US-033 parquet (default the DVC-tracked one).

    Returns:
        ``(dataset, n_regions==1, n_categories==n_classes,
        proto_active (n_classes, 384))``.

    Raises:
        FileNotFoundError: if PASTIS-R or the prototype parquet do not exist.
        ValueError: if ``n_classes`` is not in ``[1, 18]`` or the parquet does
            not contain every active class.
    """
    # Imported here so the function (and its callers' tests) can monkeypatch the
    # loader on this module without importing the whole feature pipeline.
    from ml.features.phenology_class_prototypes import (
        load_class_prototype_embeddings,
    )

    active = active_classes(n_classes)  # validates [1, 18]

    dataset = PastisPairDataset(
        n_classes=n_classes,
        root=root,
        folds=folds,
        ratio=ratio,
        seed=seed,
    )

    proto_path = prototype_path or _DEFAULT_PROTO_PATH
    proto_18, class_ids = load_class_prototype_embeddings(proto_path)
    proto_18 = np.asarray(proto_18, dtype=np.float32)

    # Map class_id -> row using the class_ids returned by the loader (do NOT
    # assume row r == class_id r+1). Select the active rows in curriculum order.
    row_of: dict[int, int] = {int(cid): row for row, cid in enumerate(class_ids)}
    missing = [c for c in active if c not in row_of]
    if missing:
        raise ValueError(
            f"prototype parquet is missing active class_ids {missing}; "
            f"available class_ids={sorted(row_of)}."
        )
    rows = [row_of[c] for c in active]
    proto_active = torch.from_numpy(proto_18[rows]).float()  # (n_classes, 384)

    n_regions = 1
    n_categories = n_classes
    logger.info(
        "incremental_dataset_built",
        n_classes=n_classes,
        n_regions=n_regions,
        n_categories=n_categories,
        n_samples=len(dataset),
        proto_shape=tuple(proto_active.shape),
        active=list(active),
    )
    return dataset, n_regions, n_categories, proto_active
