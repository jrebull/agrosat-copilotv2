"""Pixel-to-parcel reconciliation for the segmentation softmax/OOF dump (US-031).

The dense segmentation models emit a per-pixel softmax ``(18, 128, 128)``, but
the tabular ensemble members (XGBoost over AlphaEarth, Stacking / Blending /
E-b) operate at the **parcel** granularity. This module bridges the two spaces:

- :func:`load_pastis_parcel_ids` reads the per-pixel ParcelIDs map of a PASTIS-R
  patch. PASTIS-R ships the instance/parcel id raster separately from the
  semantic TARGET, as ``ANNOTATIONS/ParcelIDs_<patch_id>.npy`` (``(128, 128)``
  int32, ``0`` = no parcel / background). ``PASTISSegmentationDataset`` only
  exposes the semantic channel, so the ParcelIDs are loaded directly here.
- :func:`pixel_to_parcel_probs` reduces a dense softmax to one probability row
  per parcel (mean of the post-softmax probabilities, or mode of the per-pixel
  argmax), under the ~98% parcel-purity assumption of PASTIS-R (a small number
  of margin pixels may straddle two parcels).

PASTIS-R ParcelIDs are LOCAL per patch (the same integer id can name different
parcels in different patches), so every parcel is keyed by the canonical id
``f"{patch_id}_{local_id}"`` (Utf8) to avoid cross-patch collisions, matching
the project's :func:`ml.utils.parcel_id.canonical_parcel_id` Utf8 schema.

Project conventions: ``polars`` (never pandas) for tabular output, ``numpy``
only at the array boundary, ``structlog`` for logging, type hints everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

#: Number of contiguous agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: Per-class probability column names ``prob_000 .. prob_017`` (Float32).
PROB_COLUMNS: tuple[str, ...] = tuple(f"prob_{c:03d}" for c in range(_NUM_CLASSES))

#: Default ParcelID value that marks pixels not belonging to any parcel
#: (Background) in PASTIS-R; such pixels never produce a parcel row.
_DEFAULT_IGNORE_PARCEL_ID: int = 0

#: Numerical floor to avoid divide-by-zero when renormalizing a parcel's mean
#: probability vector.
_PROB_RENORM_EPS: float = 1e-12

#: Reduction methods supported by :func:`pixel_to_parcel_probs`.
ReduceMethod = Literal["mean", "mode"]


def load_pastis_parcel_ids(
    patch_id: str | int,
    data_root: Path | str,
) -> np.ndarray:
    """Load the per-pixel ParcelIDs raster of a PASTIS-R patch.

    PASTIS-R distributes the instance/parcel id map as
    ``<data_root>/ANNOTATIONS/ParcelIDs_<patch_id>.npy`` (``(128, 128)`` int32),
    SEPARATE from the semantic ``TARGET_<patch_id>.npy``. The value ``0`` marks
    pixels that belong to no parcel (Background). The ids are LOCAL to the patch,
    so they must be prefixed with ``patch_id`` before being used as a global key
    (see :func:`pixel_to_parcel_probs`).

    Args:
        patch_id: PASTIS-R patch identifier (numeric or string, no extension).
        data_root: Root of the PASTIS-R dataset (the directory that contains
            ``ANNOTATIONS/``).

    Returns:
        A ``numpy.ndarray`` of shape ``(128, 128)`` and dtype ``int64`` with the
        per-pixel local ParcelIDs (``0`` = no parcel).

    Raises:
        FileNotFoundError: if the ``ParcelIDs_<patch_id>.npy`` file is absent.
    """
    pid = str(patch_id)
    root = Path(data_root)
    parcel_path = root / "ANNOTATIONS" / f"ParcelIDs_{pid}.npy"
    if not parcel_path.exists():
        raise FileNotFoundError(
            f"PASTIS-R ParcelIDs raster not found: {parcel_path}. The parcel id "
            "channel is shipped separately from TARGET_<id>.npy."
        )
    arr = np.load(parcel_path)
    return np.ascontiguousarray(arr).astype(np.int64)


def _parcel_schema() -> dict[str, Any]:
    """Return the canonical parcel-level Polars schema (column -> dtype)."""
    schema: dict[str, Any] = {"canonical_parcel_id": pl.Utf8}
    for col in PROB_COLUMNS:
        schema[col] = pl.Float32
    schema["pred_class"] = pl.Int64
    schema["n_pixels"] = pl.Int64
    return schema


def _empty_parcel_frame() -> pl.DataFrame:
    """Return an empty parcel-level DataFrame with the canonical schema."""
    return pl.DataFrame(schema=_parcel_schema())


def pixel_to_parcel_probs(
    probs_18: np.ndarray,
    parcel_ids: np.ndarray,
    *,
    patch_id: str | int,
    method: ReduceMethod = "mean",
    ignore_index: int = 255,
    ignore_parcel_id: int = _DEFAULT_IGNORE_PARCEL_ID,
) -> pl.DataFrame:
    """Reduce a dense softmax ``(18, H, W)`` to parcel-level probabilities.

    Aggregates ``probs_18`` over the pixels of each PASTIS-R parcel and returns
    one row per parcel. Two reductions are supported:

    - ``"mean"`` (default): the mean of the post-softmax probabilities over the
      parcel's pixels, renormalized to sum to 1. ``pred_class`` is the argmax of
      that mean distribution. This averages PROBABILITIES (never logits),
      matching the ensemble anti-leakage convention.
    - ``"mode"``: ``pred_class`` is the majority per-pixel argmax; the reported
      probability vector is still the renormalized mean (so the row stays a valid
      distribution) but the predicted class is the discrete majority vote.

    Parcel ids are LOCAL per patch, so each parcel is keyed by
    ``canonical_parcel_id = f"{patch_id}_{local_id}"`` (Utf8) to avoid
    cross-patch collisions. Pixels whose parcel id equals ``ignore_parcel_id``
    (Background, default ``0``) never produce a row.

    Args:
        probs_18: Post-softmax probability map ``(18, H, W)`` float (sum 1 over
            the class axis per pixel).
        parcel_ids: Per-pixel local ParcelIDs map ``(H, W)`` int. Background
            pixels carry ``ignore_parcel_id``.
        patch_id: PASTIS-R patch identifier used to build the canonical key.
        method: ``"mean"`` or ``"mode"`` reduction (see above).
        ignore_index: Reserved harness ignore label (255). ParcelIDs are loaded
            independently of the semantic ignore, so this is accepted for API
            symmetry with the dump; any parcel id equal to ``ignore_index`` is
            also dropped defensively.
        ignore_parcel_id: ParcelID value that marks Background / no-parcel
            pixels and is excluded from the output (default ``0``).

    Returns:
        A Polars DataFrame sorted by ``canonical_parcel_id`` with columns:
        ``canonical_parcel_id`` (Utf8), ``prob_000 .. prob_017`` (Float32,
        summing to 1 per row), ``pred_class`` (Int64) and ``n_pixels`` (Int64).
        Empty (with the canonical schema) when no parcel pixel is present.

    Raises:
        ValueError: if shapes are inconsistent, ``probs_18`` does not have 18
            channels, or ``method`` is invalid.
    """
    if method not in ("mean", "mode"):
        raise ValueError(f"invalid method: {method!r}; use 'mean' or 'mode'.")

    probs = np.ascontiguousarray(probs_18).astype(np.float32)
    if probs.ndim != 3 or probs.shape[0] != _NUM_CLASSES:
        raise ValueError(f"probs_18 must be (18, H, W); received shape {probs.shape}.")
    pids = np.ascontiguousarray(parcel_ids).astype(np.int64)
    if pids.shape != probs.shape[1:]:
        raise ValueError(
            f"parcel_ids spatial shape {pids.shape} does not match probs_18 "
            f"spatial shape {probs.shape[1:]}."
        )

    pid_str = str(patch_id)
    # Flatten the spatial grid: (H*W,) parcel ids and (H*W, 18) probabilities.
    flat_pids = pids.reshape(-1)
    flat_probs = probs.reshape(_NUM_CLASSES, -1).T  # (H*W, 18)

    valid = (flat_pids != ignore_parcel_id) & (flat_pids != ignore_index)
    flat_pids = flat_pids[valid]
    flat_probs = flat_probs[valid]
    if flat_pids.size == 0:
        return _empty_parcel_frame()

    # Per-pixel argmax is needed for both the support count and the mode vote.
    flat_argmax = flat_probs.argmax(axis=1)

    unique_ids, inverse = np.unique(flat_pids, return_inverse=True)
    n_parcels = unique_ids.size

    # Sum probabilities per parcel via np.add.at (handles repeated indices).
    sum_probs = np.zeros((n_parcels, _NUM_CLASSES), dtype=np.float64)
    np.add.at(sum_probs, inverse, flat_probs.astype(np.float64))
    counts = np.bincount(inverse, minlength=n_parcels).astype(np.int64)

    mean_probs = sum_probs / counts[:, None]
    denom = mean_probs.sum(axis=1, keepdims=True)
    denom = np.where(denom < _PROB_RENORM_EPS, 1.0, denom)
    mean_probs = (mean_probs / denom).astype(np.float32)

    if method == "mean":
        pred_class = mean_probs.argmax(axis=1).astype(np.int64)
    else:  # mode: majority per-pixel argmax per parcel
        # vote_counts[p, c] = number of pixels in parcel p whose argmax is c.
        vote_counts = np.zeros((n_parcels, _NUM_CLASSES), dtype=np.int64)
        np.add.at(vote_counts, (inverse, flat_argmax), 1)
        pred_class = vote_counts.argmax(axis=1).astype(np.int64)

    canonical_ids = [f"{pid_str}_{int(local)}" for local in unique_ids]
    data: dict[str, object] = {"canonical_parcel_id": canonical_ids}
    for c, col in enumerate(PROB_COLUMNS):
        data[col] = mean_probs[:, c]
    data["pred_class"] = pred_class
    data["n_pixels"] = counts

    frame = pl.DataFrame(data, schema=_parcel_schema())
    # Idempotent Utf8 cast of the key (canonical project schema before joins).
    frame = canonical_parcel_id(frame, col="canonical_parcel_id")
    frame = frame.sort("canonical_parcel_id")

    logger.debug(
        "pixel_to_parcel_probs",
        patch_id=pid_str,
        method=method,
        n_parcels=n_parcels,
        n_valid_pixels=int(flat_pids.size),
    )
    return frame


def instance_to_parcel_id_map(
    patch_id: str | int,
    data_root: Path | str,
) -> dict[int, int]:
    """Map a patch's instance ids (``TARGET[1]``) to its ParcelIDs raster ids.

    PASTIS-R ships two per-pixel id rasters for the same parcels: the instance
    channel ``TARGET[1]`` (sequential ``1..N``, used by the tabular features and
    by :class:`ml.farslip.parcel_crop_dataset.ParcelCropDataset`) and the
    ``ParcelIDs_<patch>.npy`` raster (the canonical ids used by the dense OOF and
    the ground truth). They are spatially co-registered, so each instance id maps
    to the ParcelIDs value that dominates its pixels (a 1:1 correspondence in
    PASTIS-R). This bridges the instance key space to the canonical one so the
    FarSLIP members (keyed ``"{patch}_{instance}"``) align with the stacking and
    the ground truth (keyed ``"{patch}_{ParcelIDs}"``); without it the inner-join
    silently drops parcels.

    Args:
        patch_id: PASTIS-R patch identifier.
        data_root: Root of the PASTIS-R dataset (contains ``ANNOTATIONS/``).

    Returns:
        A dict ``{instance_id: parcel_raster_id}`` for every parcel of the patch.

    Raises:
        FileNotFoundError: if the patch's TARGET or ParcelIDs raster is missing.
    """
    pid = str(patch_id)
    root = Path(data_root)
    target_path = root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
    if not target_path.exists():
        raise FileNotFoundError(f"PASTIS-R semantic TARGET not found: {target_path}.")
    target = np.load(target_path)
    if target.ndim != 3 or target.shape[0] < 2:
        raise FileNotFoundError(
            f"PASTIS-R TARGET_{pid}.npy lacks the instance channel (shape {target.shape})."
        )
    instance = target[1]
    parcel_raster = load_pastis_parcel_ids(pid, root)

    mapping: dict[int, int] = {}
    for inst_id in np.unique(instance):
        inst_int = int(inst_id)
        if inst_int == 0:
            continue
        mask = instance == inst_int
        values, counts = np.unique(parcel_raster[mask], return_counts=True)
        mapping[inst_int] = int(values[counts.argmax()])
    return mapping
