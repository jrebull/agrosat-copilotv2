"""Pure class-space helpers for the segmentation re-score harness (US-030).

The harness re-evaluates six trained segmentation checkpoints apples-to-apples
in a single, contiguous **18-class** space at a fixed **128** resolution with a
unified ``ignore_index``. Two transformations make that possible and live here,
isolated from :mod:`ml.eval.dense_metrics` (the harness module) so they can be
unit-tested without loading any checkpoint:

- :func:`remap_20_to_18` collapses the PASTIS-R 20-class label space
  ``[0..19]`` (Background + 18 agronomic classes + Void) into the contiguous
  ``[0..17]`` space, sending Background and Void to ``ignore_index``.
- :func:`resample_mask_128_nearest` resamples a discrete class map to
  ``128 x 128`` using nearest-neighbour, so models trained at 256 (U-Net,
  AnySat, SegFormer) are accumulated at the same resolution as the 128-native
  models without inventing interpolated class ids.

Both operate on already-discrete maps (labels or post-``argmax`` predictions),
never on logits, model heads or ``state_dict`` keys: the U-TAE checkpoint keys
(``out_conv`` etc.) must stay intact, so the 20->18 mapping happens purely in
prediction space.

US-031 adds the **probability-space** counterparts
(:func:`remap_probs_20_to_18`, :func:`resample_probs_128_bilinear`) used by the
softmax/OOF dump. The discrete helpers above MUST NOT be used on probability
tensors: ``remap_20_to_18`` shifts class ids and would silently corrupt the
class axis of a softmax, and ``resample_mask_128_nearest`` uses nearest
interpolation, which degrades a continuous distribution (it picks a single
neighbour's value instead of blending). The probability helpers instead DROP the
Background (0) and Void (19) channels, renormalize the remaining 18 to sum to 1,
and resample with bilinear interpolation followed by a renormalization. They
operate on POST-softmax tensors only, never on logits or ``state_dict`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "DEFAULT_LABEL_SPACE",
    "FRANCE_9",
    "FRANCE_12",
    "HARNESS_IGNORE_INDEX",
    "HARNESS_NUM_CLASSES",
    "HARNESS_SIZE",
    "HCAT_MACRO",
    "LabelSpace",
    "get_label_space",
    "list_label_spaces",
    "register_label_space",
    "remap_20_to_18",
    "remap_probs_20_to_18",
    "resample_mask_128_nearest",
    "resample_probs_128_bilinear",
    "restrict_posterior",
]

#: Numerical floor used to avoid divide-by-zero when renormalizing a probability
#: map whose 18 kept channels sum to (near) zero for some pixel.
_PROB_RENORM_EPS: float = 1e-12

#: Number of contiguous classes the harness accumulates over (1..18 -> 0..17).
HARNESS_NUM_CLASSES: int = 18
#: Unified ignore index for Background, Void and out-of-range pixels.
HARNESS_IGNORE_INDEX: int = 255
#: Target side length every mask is resampled to before accumulation.
HARNESS_SIZE: int = 128


def _to_numpy_int(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a contiguous numpy integer array from a numpy/torch input.

    Args:
        x: Discrete class map as a numpy array or torch tensor.

    Returns:
        A ``numpy.ndarray`` with an integer dtype, detached from any autograd
        graph and moved to CPU when the input is a tensor.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr


def remap_20_to_18(
    labels: np.ndarray | torch.Tensor,
    *,
    background_id: int = 0,
    void_id: int = 19,
    ignore_index: int = HARNESS_IGNORE_INDEX,
) -> np.ndarray:
    """Map a 20-class label/prediction map ``[0..19]`` to contiguous ``[0..17]``.

    The PASTIS-R 20-class convention reserves id ``0`` for Background and id
    ``19`` for Void; the 18 agronomic classes occupy ``1..18``. This helper
    reindexes those agronomic classes to ``0..17`` (a simple shift of ``-1``)
    and sends both Background and Void to ``ignore_index`` so they are excluded
    from the unified confusion matrix.

    It operates AFTER ``argmax`` on discrete class maps, never on logits, the
    model head or the ``state_dict`` (the U-TAE checkpoint keys must stay
    intact); the remap lives purely in prediction/label space.

    Args:
        labels: Integer class map of any shape with values in ``[0..19]``.
        background_id: Class id treated as background (mapped to ignore).
        void_id: Class id treated as void (mapped to ignore).
        ignore_index: Value assigned to background, void and any id outside
            the agronomic ``1..18`` range.

    Returns:
        A ``numpy.ndarray`` of dtype ``int64`` and the same shape as ``labels``,
        with agronomic classes in ``[0..17]`` and background/void set to
        ``ignore_index``.
    """
    arr = _to_numpy_int(labels)
    out = np.full(arr.shape, ignore_index, dtype=np.int64)
    # Agronomic classes 1..18 -> 0..17. Anything else (Background, Void,
    # out-of-range) stays at ignore_index by construction.
    agronomic = (arr >= 1) & (arr <= HARNESS_NUM_CLASSES)
    agronomic &= arr != background_id
    agronomic &= arr != void_id
    out[agronomic] = arr[agronomic] - 1
    return out


def resample_mask_128_nearest(
    mask: np.ndarray | torch.Tensor,
    *,
    size: int = HARNESS_SIZE,
) -> np.ndarray:
    """Resample a discrete class map to ``size`` x ``size`` using nearest-neighbour.

    Used for models trained at 256 (U-Net, AnySat, SegFormer) so every model is
    accumulated at the same ``size`` resolution as the 128-native models.
    Nearest-neighbour guarantees no new (interpolated) class ids are introduced:
    every output value already appears in the input.

    Args:
        mask: Discrete class map of shape ``(H, W)``.
        size: Target side length (default :data:`HARNESS_SIZE` = 128).

    Returns:
        A ``numpy.ndarray`` of dtype ``int64`` and shape ``(size, size)``,
        nearest-neighbour resampled. When the input is already ``(size, size)``
        the values are returned unchanged (only dtype is normalized).

    Raises:
        ValueError: if ``mask`` is not a 2D map.
    """
    arr = _to_numpy_int(mask)
    if arr.ndim != 2:
        raise ValueError(f"`mask` must be a 2D (H, W) class map; received shape {arr.shape}.")
    if arr.shape == (size, size):
        return arr.astype(np.int64, copy=True)

    # Nearest interpolation in float would be lossless for integers, but we keep
    # the values exact by interpolating on a float view and casting back. torch
    # interpolate needs a (N, C, H, W) tensor.
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None, :, :]
    resampled = torch.nn.functional.interpolate(tensor, size=(size, size), mode="nearest")
    return resampled[0, 0].round().to(torch.int64).numpy()


def _to_numpy_float(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a contiguous float32 numpy array from a numpy/torch input.

    Args:
        x: Probability map as a numpy array or torch tensor.

    Returns:
        A ``numpy.ndarray`` of dtype ``float32``, detached from any autograd
        graph and moved to CPU when the input is a tensor.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def remap_probs_20_to_18(
    probs: np.ndarray | torch.Tensor,
    *,
    background_id: int = 0,
    void_id: int = 19,
) -> np.ndarray:
    """Map a 20-class probability map to the contiguous 18-class space.

    Probability analogue of :func:`remap_20_to_18` (which only handles discrete
    class maps). The 20-class checkpoints (U-Net, U-TAE, AnySat, SegFormer)
    output a softmax over the PASTIS-R convention ``[0..19]`` where id ``0`` is
    Background and id ``19`` is Void. This helper DROPS those two channels and
    RENORMALIZES the remaining 18 agronomic channels so they sum to 1 per pixel,
    yielding a distribution over the contiguous ``[0..17]`` space identical to
    the one the 18-native models (DeepLabv3+, TSViT-pheno) emit directly.

    The class axis is assumed to be the FIRST axis for a ``(20, H, W)`` input;
    for higher-rank inputs (e.g. a batched ``(N, 20, H, W)``) pass an array whose
    axis ``-3`` is the 20-class axis -- the function locates the class axis as
    the one of length 20. It operates on POST-softmax tensors only, never on
    logits or ``state_dict`` keys.

    Args:
        probs: Probability map with a class axis of length 20. Common shape is
            ``(20, H, W)``; ``(N, 20, H, W)`` is also accepted.
        background_id: Channel index treated as Background and dropped.
        void_id: Channel index treated as Void and dropped.

    Returns:
        A ``float32`` ``numpy.ndarray`` with the 20-class axis replaced by an
        18-class axis (Background/Void removed), renormalized so the kept
        channels sum to 1 along that axis. Shape mirrors the input with the class
        axis shrunk from 20 to 18.

    Raises:
        ValueError: if no axis of length 20 is found, if the input is ambiguous
            (more than one axis of length 20), or if ``background_id``/``void_id``
            are out of range or equal.
    """
    arr = _to_numpy_float(probs)
    class_axis = _find_class_axis(arr.shape, expected=20)
    if not 0 <= background_id < 20 or not 0 <= void_id < 20:
        raise ValueError(f"background_id={background_id} and void_id={void_id} must be in [0, 20).")
    if background_id == void_id:
        raise ValueError(f"background_id and void_id must differ; both were {background_id}.")

    keep = [c for c in range(20) if c not in (background_id, void_id)]
    kept = np.take(arr, keep, axis=class_axis)
    denom = kept.sum(axis=class_axis, keepdims=True)
    denom = np.where(denom < _PROB_RENORM_EPS, 1.0, denom)
    out: np.ndarray = (kept / denom).astype(np.float32)
    return out


def resample_probs_128_bilinear(
    probs: np.ndarray | torch.Tensor,
    *,
    size: int = HARNESS_SIZE,
) -> np.ndarray:
    """Resample a probability map ``(C, H, W)`` to ``(C, size, size)`` bilinearly.

    Probability analogue of :func:`resample_mask_128_nearest`. Bilinear (not
    nearest) interpolation is the correct choice for a continuous distribution:
    nearest would copy a single neighbour's value and destroy the smooth class
    posterior. After interpolation the per-pixel distribution is RENORMALIZED so
    every output pixel still sums to 1 along the class axis (bilinear blending of
    rows/columns that each sum to 1 already preserves the sum, but the explicit
    renormalization guards against float drift).

    Args:
        probs: Probability map ``(C, H, W)`` (class-first). Values are assumed
            POST-softmax (non-negative, sum 1 over ``C``).
        size: Target side length (default :data:`HARNESS_SIZE` = 128).

    Returns:
        A ``float32`` ``numpy.ndarray`` of shape ``(C, size, size)`` whose
        per-pixel distribution sums to 1 along the class axis. When the input is
        already ``(C, size, size)`` the values are returned renormalized only
        (no interpolation).

    Raises:
        ValueError: if ``probs`` is not a 3D ``(C, H, W)`` map.
    """
    arr = _to_numpy_float(probs)
    if arr.ndim != 3:
        raise ValueError(
            f"`probs` must be a 3D (C, H, W) probability map; received shape {arr.shape}."
        )

    if arr.shape[1:] != (size, size):
        tensor = torch.from_numpy(arr)[None, ...]  # (1, C, H, W)
        resampled = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        arr = resampled[0].numpy()

    denom = arr.sum(axis=0, keepdims=True)
    denom = np.where(denom < _PROB_RENORM_EPS, 1.0, denom)
    out: np.ndarray = (arr / denom).astype(np.float32)
    return out


def _find_class_axis(shape: tuple[int, ...], *, expected: int) -> int:
    """Locate the single axis of length ``expected`` in ``shape``.

    Args:
        shape: Array shape to inspect.
        expected: The class-axis length to find (e.g. 20).

    Returns:
        The index of the unique axis whose length equals ``expected``.

    Raises:
        ValueError: if zero or more than one axis matches ``expected``.
    """
    matches = [ax for ax, dim in enumerate(shape) if dim == expected]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no axis of length {expected} found in shape {shape}; the class "
            "axis is required to remap probabilities."
        )
    raise ValueError(
        f"ambiguous class axis: multiple axes of length {expected} in shape "
        f"{shape}. Pass a tensor with a single 20-length axis."
    )


# ---------------------------------------------------------------------------
# Label-space registry (US-053; hook for EPIC 12 US-074 HCAT crosswalk).
# ---------------------------------------------------------------------------
# The 18-class semantic18 posterior the EPIC 6 members emit is NOT uniformly
# trustworthy: an honest discard curve (F1 OOF fold-5, notebook 06c) shows only
# nine classes are reliably resolved. Rather than hardcode that set of nine in
# the classifier, this registry parameterizes *which* contiguous semantic18 ids a
# given "label-space" keeps, so the classifier (``ml.agent.tools.classify``) can
# mask + renormalize its posterior over the active space without knowing the set
# by name. The first space is ``france-9`` (PASTIS-R / France). EPIC 12 US-074
# will REGISTER further national spaces (``iberia-14``, ``hcat-global-20``) via an
# HCAT (Harmonized Crop and Agricultural Types) crosswalk WITHOUT touching the
# classifier: it only calls :func:`register_label_space` with a new
# :class:`LabelSpace`. That extensibility is the whole point of the registry.


@dataclass(frozen=True)
class LabelSpace:
    """A named subset of the contiguous semantic18 space the models resolve well.

    A label-space declares which semantic18 class ids (``[0..17]``, the contiguous
    space :data:`ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES` indexes) a consumer
    should keep when reporting a crop posterior, dropping the rest. Storing the
    KEPT semantic18 ids (not free-form names) lets a ``(18,)`` or ``(N, 18)``
    posterior be masked purely by index, with no name-matching at inference time.

    Attributes:
        name: Stable identifier of the space (e.g. ``"france-9"``).
        kept_class_ids: The semantic18 ids the space keeps, in canonical order.
        dropped_class_ids: The semantic18 ids the space drops (the complement of
            ``kept_class_ids`` within ``[0, 18)``), kept for traceability.
        class_names: Mapping ``{semantic18_id: human-readable crop name}`` for the
            kept ids only.
        source: Provenance string (how the kept set was chosen), e.g. the F1 OOF
            fold-5 discard curve that justified ``france-9``.
    """

    name: str
    kept_class_ids: tuple[int, ...]
    dropped_class_ids: tuple[int, ...]
    class_names: dict[int, str]
    source: str

    @property
    def dropped_class_names(self) -> dict[int, str]:
        """``{semantic18_id: crop name}`` for the DROPPED ids (out-of-vocabulary).

        The complement of :attr:`class_names`: the crops this label-space does NOT
        resolve reliably. Surfaced so the copilot can declare them outside its
        calibrated vocabulary (and hand them to the RAG + reasoner layer for a
        grounded hedge) instead of forcing a wrong in-vocabulary label. Resolved
        against :data:`ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`.

        Returns:
            The ``{id: name}`` mapping for the dropped ids (empty when the space
            keeps all of semantic18).
        """
        from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

        return {
            cid: SEMANTIC18_CLASS_NAMES[cid]
            for cid in self.dropped_class_ids
            if cid in SEMANTIC18_CLASS_NAMES
        }


# ``france-9`` semantic18 ids resolved against
# ``ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`` (verified, not by free-form
# name): the nine classes the Stacking-5 / tsvit-pheno champion resolves well by
# F1 OOF fold-5 (honest discard curve, notebook 06c celda b3c99833):
#   Winter rapeseed (id 4, F1 0.937), Corn (2, 0.935), Grapevine (7, 0.924),
#   Beet (8, 0.920), Meadow (0, 0.902), Soft winter wheat (1, 0.896),
#   Soybeans (14, 0.865), Winter barley (3, 0.844), Sunflower (6, 0.798).
# The nine DROPPED (worst F1) are the complement: Winter durum wheat (10),
# Orchard (15), Fruits/veg/flowers (11), Spring barley (5), Winter triticale (9),
# Leguminous fodder (13), Sorghum (17), Mixed cereal (16), Potatoes (12).
_FRANCE_9_KEPT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 14)

#: Number of contiguous semantic18 classes any label-space is a subset of.
_SEMANTIC18_SIZE: int = HARNESS_NUM_CLASSES


def _build_france9() -> LabelSpace:
    """Construct the ``france-9`` label-space, resolving names from semantic18.

    Returns:
        The frozen :class:`LabelSpace` for ``france-9`` with the nine kept ids,
        their nine-name mapping (resolved against
        :data:`ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`) and the dropped
        complement.

    Raises:
        ValueError: if any kept id is outside ``[0, 18)`` or missing from the
            semantic18 name table (a guard against a future rename drifting the
            ids silently).
    """
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    kept = tuple(sorted(_FRANCE_9_KEPT_IDS))
    for cid in kept:
        if not 0 <= cid < _SEMANTIC18_SIZE or cid not in SEMANTIC18_CLASS_NAMES:
            raise ValueError(
                f"france-9 kept id {cid} is not a valid semantic18 class id; "
                "the SEMANTIC18_CLASS_NAMES table changed -- re-derive the ids."
            )
    dropped = tuple(c for c in range(_SEMANTIC18_SIZE) if c not in set(kept))
    class_names = {cid: SEMANTIC18_CLASS_NAMES[cid] for cid in kept}
    return LabelSpace(
        name="france-9",
        kept_class_ids=kept,
        dropped_class_ids=dropped,
        class_names=class_names,
        source="F1 OOF fold-5 honest discard curve (notebook 06c, Stacking-5 / tsvit-pheno)",
    )


#: Module-level singleton for the first registered space (convenience export).
FRANCE_9: LabelSpace = _build_france9()


# ``france-12`` semantic18 ids: the twelve classes the NEW Voting-3 v2 champion
# (tsvit-pheno-fullm-v2 @ n_timesteps=32 + utae + xgb-alphaearth, deployment
# weights 0.902 / 0.0 / 0.098) resolves at restricted macro-F1 >= 0.90 on PASTIS
# fold-5. Source: the v2 cardinality discard curve in
# ``reports/voting_new/cardinalidad.json`` ("new"), which adds classes in
# resolved-quality order and holds macro-F1 0.9264 at 10 classes, 0.9130 at 11 and
# 0.9001 at 12 (Orchard) before dropping below 0.90 at 13 (Potatoes). The twelve
# are france-9's nine PLUS Spring barley (5), Winter durum wheat (10) and
# Orchard (15). The six DROPPED (below the 0.90 line) are the open-set the copilot
# must declare it cannot resolve: Winter triticale (9), Fruits/veg/flowers (11),
# Potatoes (12), Leguminous fodder (13), Mixed cereal (16), Sorghum (17).
_FRANCE_12_KEPT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15)


def _build_france12() -> LabelSpace:
    """Construct the ``france-12`` label-space (the v2 champion's 12 resolved crops).

    Returns:
        The frozen :class:`LabelSpace` for ``france-12`` with the twelve kept ids,
        their name mapping (resolved against
        :data:`ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`) and the six-class
        dropped complement (the open-set the copilot reports as out-of-vocabulary).

    Raises:
        ValueError: if any kept id is outside ``[0, 18)`` or missing from the
            semantic18 name table.
    """
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    kept = tuple(sorted(_FRANCE_12_KEPT_IDS))
    for cid in kept:
        if not 0 <= cid < _SEMANTIC18_SIZE or cid not in SEMANTIC18_CLASS_NAMES:
            raise ValueError(
                f"france-12 kept id {cid} is not a valid semantic18 class id; "
                "the SEMANTIC18_CLASS_NAMES table changed -- re-derive the ids."
            )
    dropped = tuple(c for c in range(_SEMANTIC18_SIZE) if c not in set(kept))
    class_names = {cid: SEMANTIC18_CLASS_NAMES[cid] for cid in kept}
    return LabelSpace(
        name="france-12",
        kept_class_ids=kept,
        dropped_class_ids=dropped,
        class_names=class_names,
        source=(
            "v2 Voting-3 cardinality discard curve (reports/voting_new/cardinalidad.json, "
            "macro-F1 0.9001 at 12 classes); tsvit-pheno-fullm-v2 + utae + xgb"
        ),
    )


#: Module-level singleton for the v2 champion's twelve-class space.
FRANCE_12: LabelSpace = _build_france12()

#: Mutable registry of label-spaces by name. EPIC 12 US-074 adds entries via
#: :func:`register_label_space` (HCAT crosswalk) without editing the classifier.
_REGISTRY: dict[str, LabelSpace] = {FRANCE_9.name: FRANCE_9, FRANCE_12.name: FRANCE_12}

#: Canonical default label-space for the copilot (SINGLE SOURCE OF TRUTH -- never
#: hardcode a space name at a call site). Consumers that take no explicit name
#: (:func:`get_label_space`, the ``classify_new_parcel`` input default, the
#: perceiver) resolve to this; ops can override per deployment via
#: ``Settings.label_space`` (env ``LABEL_SPACE``) without touching code. Set to the
#: v2 Voting-3 champion's twelve-class space (``france-12``); revert to ``france-9``
#: with one edit here (or the env var) if a narrower vocabulary is wanted.
DEFAULT_LABEL_SPACE: str = FRANCE_12.name


def register_label_space(space: LabelSpace, *, overwrite: bool = False) -> None:
    """Register a label-space so consumers can select it by name.

    This is the EPIC 12 US-074 seam: a future HCAT crosswalk registers
    ``iberia-14`` / ``hcat-global-20`` here and the classifier picks them up by
    name, with no change to :mod:`ml.agent.tools.classify`.

    Args:
        space: The :class:`LabelSpace` to register.
        overwrite: When ``False`` (default) registering an existing name raises;
            pass ``True`` to replace it deliberately.

    Raises:
        ValueError: if ``space.kept_class_ids`` is empty, has an out-of-range id,
            or the name already exists and ``overwrite`` is ``False``.
    """
    if not space.kept_class_ids:
        raise ValueError(f"label-space {space.name!r} must keep at least one class id.")
    for cid in space.kept_class_ids:
        if not 0 <= cid < _SEMANTIC18_SIZE:
            raise ValueError(
                f"label-space {space.name!r} kept id {cid} is outside the "
                f"semantic18 range [0, {_SEMANTIC18_SIZE})."
            )
    if space.name in _REGISTRY and not overwrite:
        raise ValueError(f"label-space {space.name!r} is already registered; pass overwrite=True.")
    _REGISTRY[space.name] = space


def get_label_space(name: str | None = None) -> LabelSpace:
    """Return a registered label-space by name.

    Args:
        name: Registered label-space name. ``None`` (the default) resolves to
            :data:`DEFAULT_LABEL_SPACE`, so no call site has to hardcode a name.

    Returns:
        The :class:`LabelSpace` registered under ``name``.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name is None:
        name = DEFAULT_LABEL_SPACE
    if name not in _REGISTRY:
        raise KeyError(f"unknown label-space {name!r}; registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name]


def list_label_spaces() -> tuple[str, ...]:
    """Return the names of every registered label-space (sorted).

    Returns:
        The registered label-space names in sorted order.
    """
    return tuple(sorted(_REGISTRY))


def restrict_posterior(proba: np.ndarray, label_space: LabelSpace) -> dict[int, float]:
    """Mask a semantic18 posterior to a label-space and renormalize over it.

    Drops the ``dropped_class_ids`` mass of a ``(18,)`` post-softmax row and
    renormalizes the kept ``kept_class_ids`` so they sum to 1, yielding a
    distribution OVER the well-resolved classes of ``label_space`` only. The
    masked mass is discarded (not redistributed proportionally outside the kept
    set) -- the renormalization redistributes it proportionally AMONG the kept
    classes, which is exactly "ignore the classes the model cannot resolve".

    Args:
        proba: A ``(18,)`` post-softmax distribution over the contiguous
            semantic18 space (e.g. ``xgb-alphaearth`` or the Stacking-5 meta
            posterior).
        label_space: The active :class:`LabelSpace` whose ``kept_class_ids`` are
            retained.

    Returns:
        Mapping ``{semantic18_id: renormalized_probability}`` over the kept ids
        only, summing to ~1 (or to 0.0 for every kept id when the kept mass was
        ~0, an honest "no signal in the resolved classes" rather than a fabricated
        certainty).

    Raises:
        ValueError: if ``proba`` is not a 1-D vector of length 18.
    """
    arr = np.asarray(proba, dtype=np.float64).ravel()
    if arr.size != _SEMANTIC18_SIZE:
        raise ValueError(
            f"restrict_posterior expects a ({_SEMANTIC18_SIZE},) semantic18 "
            f"posterior; received size {arr.size}."
        )
    kept = label_space.kept_class_ids
    kept_mass = float(arr[list(kept)].sum())
    if kept_mass > 1e-12:
        return {cid: float(arr[cid]) / kept_mass for cid in kept}
    # No probability mass landed on the resolved classes: report an explicit
    # zero distribution rather than inventing a uniform prior.
    return {cid: 0.0 for cid in kept}


# ---------------------------------------------------------------------------
# EPIC 12 US-074: ``hcat-macro`` label-space (HCAT v3 crosswalk).
# ---------------------------------------------------------------------------
# US-074 AMPLIES this registry with a label-space carrying the PASTIS-18 -> HCAT
# v3 macro-group mapping (data/reference/hcat_crosswalk.parquet), WITHOUT touching
# ``ml.agent.tools.classify`` or the ``restrict_posterior`` signature. Unlike
# ``france-9`` (a *subset* of well-resolved ids), ``hcat-macro`` keeps ALL 18
# semantic18 ids and EXPOSES the macro-group label per id in ``class_names``; the
# actual semantic18->macro aggregation is done by the adapter/consumer reading the
# parquet, not by the registry. So ``restrict_posterior`` (a mask + renormalize
# over a subset) is untouched and US-053 keeps working: the kept set is the
# trivial subset "all 18", and the macro grouping rides along in the names.

#: Mapping ``semantic18_id -> "MACRO_L1_6|macro_hcat_group"`` derived from the
#: US-074 crosswalk (legacy 6-family HCAT L1 used by E4/E6 + the finer 10-group
#: HCAT L2 macro). Hardcoded here (not read from the parquet) so importing the
#: registry never depends on a data file being present; the parquet is the
#: single source of truth and a test asserts the two agree.
_HCAT_MACRO_BY_ID: dict[int, str] = {
    0: "GRASSLAND_OTHER|grassland",
    1: "CEREALS|cereals",
    2: "CEREALS|cereals",
    3: "CEREALS|cereals",
    4: "OILSEEDS|oilseed_industrial",
    5: "CEREALS|cereals",
    6: "OILSEEDS|oilseed_industrial",
    7: "PERMANENT_WOODY|vineyard",
    8: "ROOT_CROPS|sugar_beet",
    9: "CEREALS|cereals",
    10: "CEREALS|cereals",
    11: "GRASSLAND_OTHER|vegetables",
    12: "ROOT_CROPS|potato",
    13: "LEGUMES|legumes_fodder",
    14: "LEGUMES|soybean",
    15: "PERMANENT_WOODY|orchard",
    16: "CEREALS|cereals",
    17: "CEREALS|cereals",
}


def _build_hcat_macro() -> LabelSpace:
    """Construct the ``hcat-macro`` label-space over all 18 semantic18 ids.

    Keeps every semantic18 id (the trivial subset = all 18) and carries the
    ``"MACRO_L1_6|macro_hcat_group|crop_name"`` triple in ``class_names`` so a
    consumer can aggregate the semantic18 posterior into HCAT macro families
    without touching the classifier. Derived from the US-074 crosswalk
    (``data/reference/hcat_crosswalk.parquet``).

    Returns:
        The frozen :class:`LabelSpace` for ``hcat-macro`` with 18 kept ids, an
        empty dropped set, and the macro-annotated name mapping.

    Raises:
        ValueError: if an id is missing from the semantic18 name table or the
            macro mapping (a guard against the crosswalk drifting silently).
    """
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    kept = tuple(range(_SEMANTIC18_SIZE))  # all 18, no class is dropped
    for cid in kept:
        if cid not in SEMANTIC18_CLASS_NAMES or cid not in _HCAT_MACRO_BY_ID:
            raise ValueError(
                f"hcat-macro id {cid} is missing from the semantic18 name table "
                "or the macro mapping; re-derive from the US-074 crosswalk."
            )
    class_names = {cid: f"{_HCAT_MACRO_BY_ID[cid]}|{SEMANTIC18_CLASS_NAMES[cid]}" for cid in kept}
    return LabelSpace(
        name="hcat-macro",
        kept_class_ids=kept,
        dropped_class_ids=(),
        class_names=class_names,
        source=("US-074 crosswalk PASTIS-18 -> HCAT v3 (data/reference/hcat_crosswalk.parquet)"),
    )


#: Module-level singleton for the HCAT macro label-space (convenience export).
HCAT_MACRO: LabelSpace = _build_hcat_macro()
register_label_space(HCAT_MACRO)
