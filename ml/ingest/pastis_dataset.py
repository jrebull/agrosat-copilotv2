"""Dense PyTorch dataset for semantic segmentation over PASTIS-R (EPIC 5).

This module is the **shared dense pipeline** reused by the 6 segmentation
architectures of Avance 4 (U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, AnySat).
It builds training-ready tensors from the raw PASTIS-R patches loaded by
:func:`ml.ingest.pastis_loader.load_pastis_patch`.

Team conventions (comparability of the 6 models):

- ``num_classes = 20`` (0 = background ... 19 = void).
- ``ignore_index = 19`` (void) in loss and metrics.
- Resolution ``256x256`` (bilinear resize image / nearest label).
- Temporal reduction ``median`` for 2D models (U-Net/DeepLabv3+/SegFormer);
  ``none`` mode (series trimmed to ``fixed_t`` frames) for temporal models
  (U-TAE/TSViT/AnySat).

Per-band normalization with the official ``NORM_S2_patch.json`` statistics
averaged over the training folds (no leakage from the test fold).
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.ingest.pastis_loader import (
    PASTIS_S2_BANDS,
    pastis_patch_index,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "PASTIS_IGNORE_INDEX",
    "PASTIS_NUM_CLASSES",
    "PASTIS_TARGET_SIZE",
    "PASTISDataset",
    "load_norm_stats",
    "pastis_fold_split",
]

PASTIS_NUM_CLASSES: int = 20
"""Number of PASTIS-R semantic classes (0 background, 1-18 crops, 19 void)."""

PASTIS_IGNORE_INDEX: int = 19
"""``void`` class ignored in loss and metrics (shared team convention)."""

PASTIS_TARGET_SIZE: int = 256
"""Target spatial resolution after resize (native patches are 128x128)."""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_N_BANDS = len(PASTIS_S2_BANDS)

TemporalReduction = Literal["median", "mean", "none"]


def load_norm_stats(
    root: Path | None = None,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Load and average the per-band normalization statistics of PASTIS-R.

    Reads ``NORM_S2_patch.json`` (a dict ``{Fold_k: {mean: [10], std: [10]}}``)
    and averages the means and deviations of the indicated folds. To avoid
    spatial leakage **only the training folds** must be passed.

    Args:
        root: Root of the PASTIS-R dataset (default ``data/PASTIS-R/``).
        folds: Training folds over which to average (1..5).

    Returns:
        Tuple ``(mean, std)`` of ``float32`` arrays of shape ``(10,)``. If the
        file does not exist it returns ``mean=0``, ``std=1`` (degraded mode,
        no-op).
    """
    root = root or _DEFAULT_ROOT
    norm_path = root / "NORM_S2_patch.json"
    if not norm_path.exists():
        logger.warning("pastis_norm_missing", path=str(norm_path))
        return np.zeros(_N_BANDS, dtype=np.float32), np.ones(_N_BANDS, dtype=np.float32)

    with norm_path.open(encoding="utf-8") as fh:
        stats = json.load(fh)

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for fold in folds:
        entry = stats.get(f"Fold_{fold}")
        if entry is None:
            continue
        means.append(np.asarray(entry["mean"], dtype=np.float32))
        stds.append(np.asarray(entry["std"], dtype=np.float32))

    if not means:
        return np.zeros(_N_BANDS, dtype=np.float32), np.ones(_N_BANDS, dtype=np.float32)

    mean = np.mean(np.stack(means, axis=0), axis=0)
    std = np.mean(np.stack(stds, axis=0), axis=0)
    # Avoid division by zero in degenerate bands.
    std = np.where(std <= 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def pastis_fold_split(
    root: Path | None = None,
    train_folds: tuple[int, ...] = (1, 2, 3),
    val_folds: tuple[int, ...] = (4,),
    test_folds: tuple[int, ...] = (5,),
) -> dict[str, list[str]]:
    """Build the train/val/test split using the 5 official PASTIS-R folds.

    The folds come predefined in ``metadata.geojson`` (``Fold`` field) and are
    spatially disjoint by dataset design, so they avoid the spatial leakage that
    the project ML rule forbids (no random split).

    Args:
        root: Root of the PASTIS-R dataset.
        train_folds: Folds assigned to training.
        val_folds: Folds assigned to validation.
        test_folds: Folds assigned to test.

    Returns:
        Dictionary ``{"train": [...], "val": [...], "test": [...]}`` with lists of
        ``patch_id`` (str). Empty lists if ``metadata.geojson`` does not exist.
    """
    root = root or _DEFAULT_ROOT
    index = pastis_patch_index(root / "metadata.geojson")
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    if index.is_empty():
        logger.warning("pastis_metadata_missing", root=str(root))
        return split

    fold_to_key = {f: "train" for f in train_folds}
    fold_to_key.update({f: "val" for f in val_folds})
    fold_to_key.update({f: "test" for f in test_folds})

    for row in index.iter_rows(named=True):
        key = fold_to_key.get(int(row["Fold"]))
        if key is not None:
            split[key].append(str(row["patch_id"]))
    return split


def _resize_spatial(x: torch.Tensor, size: int, *, label: bool) -> torch.Tensor:
    """Resize the spatial plane ``(..., H, W)`` to ``(..., size, size)``.

    Args:
        x: Tensor ``(C, H, W)`` (image) or ``(H, W)`` (label).
        size: Target side.
        label: If ``True`` uses ``nearest`` interpolation (preserves class ids);
            if ``False`` uses ``bilinear`` (continuous image).

    Returns:
        Resized tensor with the same number of dimensions as the input.
    """
    if label:
        grid = x.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        out = F.interpolate(grid, size=(size, size), mode="nearest")
        return out.squeeze(0).squeeze(0).long()
    grid = x.unsqueeze(0)  # (1, C, H, W)
    out = F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze(0)


def _yyyymmdd_to_doy(date_int: int) -> int:
    """Convert a ``YYYYMMDD`` date to day-of-year (1-366).

    The temporal models (U-TAE, TSViT, AnySat) expect temporal positions as
    day-of-year, not the raw ``YYYYMMDD`` integer that PASTIS-R distributes.

    Args:
        date_int: Date as a ``YYYYMMDD`` integer (e.g. ``20190101``).

    Returns:
        Day of year in ``[1, 366]``; ``0`` if the date is invalid or <= 0.
    """
    if date_int <= 0:
        return 0
    year, month, day = date_int // 10000, (date_int // 100) % 100, date_int % 100
    try:
        return datetime.date(year, month, day).timetuple().tm_yday
    except ValueError:
        return 0


def _select_frames(n_t: int, fixed_t: int) -> list[int]:
    """Return temporal indices to trim/pad the series to ``fixed_t``.

    Args:
        n_t: Number of frames available in the patch.
        fixed_t: Target number of frames.

    Returns:
        List of length ``fixed_t`` with indices in ``[0, n_t)`` (uniform spacing
        if ``n_t >= fixed_t``; repeating the last one if ``n_t < fixed_t``).
    """
    if n_t >= fixed_t:
        indices: list[int] = np.linspace(0, n_t - 1, fixed_t).round().astype(int).tolist()
        return indices
    return list(range(n_t)) + [n_t - 1] * (fixed_t - n_t)


def _load_pastis_metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    """Parse ``metadata.geojson`` only once and return dates and fold per patch.

    Avoids re-parsing the geojson (~19 MB) in each ``__getitem__``, which is
    prohibitive when reading from a mounted Drive. It is invoked only when the
    dataset needs the dates (temporal mode) or the fold (``return_meta``).

    Args:
        root: Root of the PASTIS-R dataset.

    Returns:
        Dictionary ``{patch_id: {"dates": [int], "fold": int | None}}``. Empty if
        ``metadata.geojson`` does not exist.
    """
    meta_path = root / "metadata.geojson"
    if not meta_path.exists():
        return {}
    with meta_path.open(encoding="utf-8") as fh:
        md = json.load(fh)
    index: dict[str, dict[str, Any]] = {}
    for feat in md.get("features", []):
        props = feat.get("properties", {}) or {}
        pid_raw = feat.get("id") or props.get("ID_PATCH")
        if pid_raw is None:
            continue
        dates_raw = props.get("dates-S2", {})
        dates = (
            [int(v) for _, v in sorted(dates_raw.items(), key=lambda kv: int(kv[0]))]
            if isinstance(dates_raw, dict)
            else []
        )
        fold = int(props["Fold"]) if props.get("Fold") is not None else None
        index[str(pid_raw)] = {"dates": dates, "fold": fold}
    return index


class PASTISDataset(Dataset):
    """Dense PASTIS-R dataset for multitemporal semantic segmentation.

    Each item is a dictionary of tensors ready to feed a segmentation model. The
    ``temporal_reduction`` mode determines the image shape:

    - ``"median"`` / ``"mean"``: 2D temporal composite ``image (10, S, S)``,
      suitable for 2D CNNs (U-Net, DeepLabv3+, SegFormer).
    - ``"none"``: series trimmed to ``fixed_t`` frames ``image (fixed_t, 10, S, S)``
      plus ``dates (fixed_t,)``, suitable for temporal models (U-TAE, TSViT,
      AnySat).

    In all modes ``semantic`` is ``(S, S)`` ``long`` with ids 0..19.
    """

    def __init__(
        self,
        patch_ids: Sequence[str | int],
        root: Path | None = None,
        *,
        target_size: int = PASTIS_TARGET_SIZE,
        temporal_reduction: TemporalReduction = "median",
        fixed_t: int = 10,
        norm: tuple[np.ndarray, np.ndarray] | None = None,
        num_classes: int = PASTIS_NUM_CLASSES,
        ignore_index: int = PASTIS_IGNORE_INDEX,
        return_meta: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            patch_ids: List of patch identifiers (from
                :func:`pastis_fold_split`).
            root: Root of the PASTIS-R dataset.
            target_size: Target spatial side after resize.
            temporal_reduction: ``median``, ``mean`` or ``none``.
            fixed_t: Number of frames when ``temporal_reduction="none"``.
            norm: Precomputed ``(mean, std)`` tuple; if ``None`` the stats of all
                folds are loaded (pass the train ones to avoid leakage).
            num_classes: Number of classes (default 20).
            ignore_index: Class to ignore (default 19, void).
            return_meta: If ``True`` includes ``patch_id`` and ``fold`` in the item.
        """
        self.patch_ids = [str(p) for p in patch_ids]
        self.root = root or _DEFAULT_ROOT
        self.target_size = target_size
        self.temporal_reduction = temporal_reduction
        self.fixed_t = fixed_t
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.return_meta = return_meta
        mean, std = norm if norm is not None else load_norm_stats(self.root)
        # (10, 1, 1) for broadcasting over (T, 10, H, W) or (10, H, W).
        self._mean = mean.reshape(_N_BANDS, 1, 1)
        self._std = std.reshape(_N_BANDS, 1, 1)
        # Parse metadata.geojson only once (not in each __getitem__) when
        # needed: dates for the temporal mode, fold for return_meta.
        needs_meta = temporal_reduction == "none" or return_meta
        self._meta_index = _load_pastis_metadata_index(self.root) if needs_meta else {}

    def __len__(self) -> int:
        """Number of patches in the dataset."""
        return len(self.patch_ids)

    def _normalize(self, s2: np.ndarray) -> np.ndarray:
        """Normalize per band ``(T, 10, H, W)`` with the ``(mean, std)`` from init."""
        normalized: np.ndarray = (s2 - self._mean[None]) / self._std[None]
        return normalized

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load, normalize and resize a patch into training tensors.

        Args:
            idx: Index of the patch in ``patch_ids``.

        Returns:
            Dictionary with ``image``, ``semantic`` and, optionally, ``dates``
            (temporal mode) and ``patch_id``/``fold`` (if ``return_meta``).
        """
        pid = self.patch_ids[idx]
        # Direct load of the .npy files (1 file per patch), without re-parsing
        # the ~19 MB metadata.geojson in each item.
        s2 = np.load(self.root / "DATA_S2" / f"S2_{pid}.npy").astype(np.float32)
        s2 = self._normalize(s2)  # (T, 10, 128, 128)

        tgt_path = self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if tgt_path.exists():
            semantic = np.load(tgt_path)[0]  # channel 0 = semantic label
        else:
            semantic = np.zeros(s2.shape[-2:], dtype=np.uint8)
        label = torch.from_numpy(semantic.astype(np.int64))
        label = _resize_spatial(label, self.target_size, label=True)

        item: dict[str, Any] = {"semantic": label}

        if self.temporal_reduction in ("median", "mean"):
            reducer = np.median if self.temporal_reduction == "median" else np.mean
            composite = reducer(s2, axis=0).astype(np.float32)  # (10, 128, 128)
            image = _resize_spatial(torch.from_numpy(composite), self.target_size, label=False)
            item["image"] = image  # (10, S, S)
        else:
            n_t = s2.shape[0]
            frames = _select_frames(n_t, self.fixed_t)
            series = torch.from_numpy(s2[frames])  # (fixed_t, 10, 128, 128) = (N, C, H, W)
            # The series is already in (N, C, H, W) format; the spatial plane
            # is rescaled directly (each frame as an item of the "batch").
            series = F.interpolate(
                series,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            item["image"] = series  # (fixed_t, 10, S, S)
            dates = self._meta_index.get(pid, {}).get("dates") or [0] * n_t
            sel_dates = [int(dates[i]) if i < len(dates) else 0 for i in frames]
            # Temporal models expect day-of-year, not the raw YYYYMMDD.
            sel_doy = [_yyyymmdd_to_doy(d) for d in sel_dates]
            item["dates"] = torch.tensor(sel_doy, dtype=torch.int64)

        if self.return_meta:
            item["patch_id"] = pid
            item["fold"] = self._meta_index.get(pid, {}).get("fold")
        return item
