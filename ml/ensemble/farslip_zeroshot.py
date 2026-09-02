"""FarSLIP zero-shot parcel member for the E6 ensembles (US-042 / EPIC 6).

This module turns the FarSLIP vision tower into a zero-shot crop classifier and
materializes its fold-5 parcel out-of-fold (OOF) predictions in the SAME schema
as the dense / tabular OOF members, so Stacking / Blending / Voting can consume
it as one more heterogeneous base learner.

What "FarSLIP zero-shot" means here (the variant the sponsor asked for, NOT
plain CLIP). :class:`ml.extractors.farslip_extractor.FarSLIPExtractor`'s
``_load_student_weights`` loads ONLY the ``vision_model`` of the FarSLIP
checkpoint; the text tower keeps the base CLIP weights. So instantiating
``FarSLIPExtractor(weights_uri="<FarSLIP checkpoint>")`` yields a FarSLIP
fine-tuned vision tower paired with a CLIP-base text tower -- exactly the
zero-shot FarSLIP variant. This is why
:func:`materialize_zeroshot_oof` defaults ``farslip_checkpoint`` to the real
checkpoint (never ``weights_uri=None``, which would be CLIP teacher mode).

The classifier is template-based: for every PASTIS crop a small bank of English
phenological prompts is encoded by the (CLIP-base) text tower and averaged into a
single L2-normalized class vector. A parcel's peak-NDVI crop is embedded by the
FarSLIP vision tower; the dot product with the class bank, scaled by a FIXED
``logit_scale`` and softmaxed, gives the per-parcel ``(18,)`` probability vector.

Anti-leakage (R-LEAK -- the single most important rubric criterion):

1. **Report fold-5 ONLY.** ``materialize_zeroshot_oof`` iterates the held-out
   fold-5 parcels of :class:`ml.farslip.parcel_crop_dataset.ParcelCropDataset`.
2. **Probabilities, not logits.** Every parcel row is post-softmax and sums to 1
   (validated downstream by ``EnsembleModel.validate_probs``).
3. **No fold-5 fitting.** Zero-shot trains NOTHING: the ``logit_scale`` is fixed
   a priori (it is never tuned against fold-5), and the text bank depends only on
   the class names + prompt templates, never on any fold-5 statistic.

Class space. The bank and the OOF columns live in the contiguous ``semantic18``
space ``[0..17]`` (PASTIS class ``c`` shifted to ``c - 1``); the dataset yields
the RAW PASTIS ``class_id`` (1..18) but the labels are only used to scope the
active classes, never to fit anything.

Project conventions: ``polars`` (never pandas), ``numpy``/``torch`` only at the
array boundary, ``structlog`` for logging, type hints and Google-style
docstrings; visible prose Spanish, identifiers English; no emojis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl
import structlog
import torch

from ml.farslip.pastis_pair_dataset import active_classes
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

#: Number of contiguous agronomic classes in the harness semantic18 space.
_NUM_CLASSES: int = 18

#: The only fold whose predictions may be dumped/reported (anti-leakage R-LEAK).
HELD_OUT_FOLD: int = 5

#: Canonical key column shared by every parcel frame.
_KEY: str = "canonical_parcel_id"

#: Embedding width of the FarSLIP / CLIP projection (visual + text are 512-dim).
_EMBED_DIM: int = 512

#: Default CLIP/FarSLIP zero-shot temperature. FIXED a priori (never tuned
#: against fold-5): this is the canonical CLIP ``logit_scale`` value
#: ``exp(log(1 / 0.07)) ~= 14.29``, the temperature CLIP was trained with. It is
#: a constant of the classifier, NOT a hyper-parameter fitted on the held-out
#: fold (that would be a leak).
DEFAULT_LOGIT_SCALE: float = 1.0 / 0.07

#: Emit a progress log roughly every this many parcels during the zero-shot pass
#: (per-parcel forward has no built-in progress; avoids running blind).
_PROGRESS_EVERY: int = 2048

#: Default FarSLIP checkpoint (vision tower). The text tower stays CLIP-base
#: inside :class:`FarSLIPExtractor`, which IS the requested zero-shot variant.
DEFAULT_FARSLIP_CHECKPOINT: str = "checkpoints/farslip/faithful_v2/best.safetensors"

#: PASTIS-R semantic18 crop names indexed by the contiguous label ``[0..17]``
#: (i.e. PASTIS class ``c + 1``). Kept here as an editable constant so the prompt
#: bank is self-contained and reproducible.
SEMANTIC18_CROP_NAMES: tuple[str, ...] = (
    "meadow",  # 0  (PASTIS 1)
    "soft winter wheat",  # 1  (PASTIS 2)
    "corn",  # 2  (PASTIS 3)
    "winter barley",  # 3  (PASTIS 4)
    "winter rapeseed",  # 4  (PASTIS 5)
    "spring barley",  # 5  (PASTIS 6)
    "sunflower",  # 6  (PASTIS 7)
    "grapevine",  # 7  (PASTIS 8)
    "beet",  # 8  (PASTIS 9)
    "winter triticale",  # 9  (PASTIS 10)
    "winter durum wheat",  # 10 (PASTIS 11)
    "fruits, vegetables and flowers",  # 11 (PASTIS 12)
    "potatoes",  # 12 (PASTIS 13)
    "leguminous fodder",  # 13 (PASTIS 14)
    "soybeans",  # 14 (PASTIS 15)
    "orchard",  # 15 (PASTIS 16)
    "mixed cereal",  # 16 (PASTIS 17)
    "sorghum",  # 17 (PASTIS 18)
)

#: English phenological prompt templates. ``{name}`` is filled with each crop
#: name; the set spans the agronomic context (field/crop/parcel) and the four
#: seasons so the averaged text vector is robust to the peak-NDVI composite the
#: vision tower sees. Editable constant -- a richer set (e.g. LLM-authored
#: phenology descriptions) can be swapped in without touching the math.
DEFAULT_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a satellite image of a {name} field",
    "an aerial photo of a {name} crop",
    "a Sentinel-2 image of a {name} agricultural parcel",
    "a {name} field seen from above in spring",
    "a {name} field seen from above in summer",
    "a {name} field seen from above at harvest in autumn",
    "a {name} field seen from above in winter",
    "remote sensing imagery of a {name} cultivation",
)


class _TextEncoder(Protocol):
    """Minimal contract of the text side of a CLIP/FarSLIP extractor."""

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode ``texts`` into ``(N, 512)`` L2-normalized embeddings."""
        ...


class _ImageEncoder(Protocol):
    """Minimal contract of the vision side of a CLIP/FarSLIP extractor."""

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        """Embed ``crops`` ``(B, 4, H, W)`` into ``(B, 512)`` L2-norm."""
        ...


class _ZeroShotExtractor(_TextEncoder, _ImageEncoder, Protocol):
    """An extractor exposing both ``encode_text`` and ``extract_embeddings``."""


def _build_prompts(name: str, templates: Sequence[str]) -> list[str]:
    """Fill ``templates`` with one crop ``name``.

    Args:
        name: Crop name (lower-case English).
        templates: Prompt templates each containing a ``{name}`` placeholder.

    Returns:
        The list of rendered prompts.
    """
    return [tpl.format(name=name) for tpl in templates]


def build_text_class_bank(
    extractor: _TextEncoder,
    class_names: Sequence[str],
    *,
    prompts_per_class: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
) -> np.ndarray:
    """Encode a per-class bank of phenological prompts into ``(18, 512)``.

    For each crop name the templates in ``prompts_per_class`` are rendered, encoded
    by the (CLIP-base) text tower, mean-pooled and L2-normalized into a single
    class vector. The order of ``class_names`` IS the output row order, so callers
    must pass them in semantic18 order ``[0..17]`` (use
    :data:`SEMANTIC18_CROP_NAMES`).

    Args:
        extractor: Object exposing ``encode_text(list[str]) -> (N, 512)`` L2-norm
            embeddings (a :class:`FarSLIPExtractor` or a test stub).
        class_names: Crop names in semantic18 order (length 18 for the full
            space). Each becomes one row of the bank.
        prompts_per_class: Prompt templates (each with a ``{name}`` placeholder),
            shared by every class. Editable constant (default
            :data:`DEFAULT_PROMPT_TEMPLATES`).

    Returns:
        A ``(len(class_names), 512)`` float32 ``numpy.ndarray`` of L2-normalized
        class text vectors, one row per class in ``class_names`` order.

    Raises:
        ValueError: if ``class_names`` or ``prompts_per_class`` is empty, or the
            encoder returns an embedding without 512 columns.
    """
    names = [str(n) for n in class_names]
    templates = [str(t) for t in prompts_per_class]
    if not names:
        raise ValueError("class_names must be non-empty.")
    if not templates:
        raise ValueError("prompts_per_class must be non-empty.")

    rows: list[np.ndarray] = []
    for name in names:
        prompts = _build_prompts(name, templates)
        with torch.inference_mode():
            embeds = extractor.encode_text(prompts)
        arr = np.asarray(embeds.detach().cpu().float().numpy(), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != _EMBED_DIM:
            raise ValueError(
                f"encode_text must return (N, {_EMBED_DIM}); got {arr.shape} for class {name!r}."
            )
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec = mean_vec / norm
        rows.append(mean_vec.astype(np.float32))

    bank = np.stack(rows, axis=0).astype(np.float32)
    logger.info(
        "farslip_text_class_bank_built",
        n_classes=bank.shape[0],
        embed_dim=bank.shape[1],
        n_prompts_per_class=len(templates),
    )
    return bank


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def zeroshot_parcel_proba(
    extractor: _ImageEncoder,
    dataset: Any,
    text_bank: np.ndarray,
    *,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Compute the zero-shot ``(n_parcels, 18)`` probability matrix of a dataset.

    Each parcel's peak-NDVI crop is embedded by the FarSLIP vision tower; its
    image-text similarity against ``text_bank`` is scaled by the FIXED
    ``logit_scale`` and softmaxed. The ``logit_scale`` is a constant of the
    classifier (CLIP temperature), never fitted against the held-out fold.

    Args:
        extractor: Object exposing ``extract_embeddings(crops) -> (B, 512)``
            L2-norm (a :class:`FarSLIPExtractor` or a stub).
        dataset: Indexable dataset of per-parcel items, each a dict with
            ``image`` ``(4, H, W)`` and ``parcel_id`` (the
            :class:`ParcelCropDataset` contract).
        text_bank: ``(18, 512)`` L2-normalized class text vectors (row ``c`` ==
            semantic18 class ``c``).
        logit_scale: Fixed temperature multiplying the cosine similarities
            (default :data:`DEFAULT_LOGIT_SCALE`).

    Returns:
        A tuple ``(proba, parcel_ids, class_ids)`` where ``proba`` is a
        ``(n_parcels, 18)`` float32 row-stochastic matrix, ``parcel_ids`` are the
        dataset's raw ``"{patch}_{instance}"`` keys (row aligned) and ``class_ids``
        are the raw PASTIS labels (row aligned, for downstream diagnostics only).

    Raises:
        ValueError: if ``text_bank`` is not ``(C, 512)`` or an image embedding
            does not have 512 columns.
    """
    bank = np.asarray(text_bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != _EMBED_DIM:
        raise ValueError(f"text_bank must be (C, {_EMBED_DIM}); got {bank.shape}.")
    scale = float(logit_scale)

    probs: list[np.ndarray] = []
    parcel_ids: list[str] = []
    class_ids: list[int] = []
    n = len(dataset)
    for idx in range(n):
        if idx and idx % _PROGRESS_EVERY == 0:
            logger.info("zeroshot_progress", done=idx, total=n)
        item = dataset[idx]
        image = item["image"]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        crops = image.unsqueeze(0)  # (1, 4, H, W)
        with torch.inference_mode():
            embeds = extractor.extract_embeddings(crops)
        img_vec = np.asarray(embeds.detach().cpu().float().numpy(), dtype=np.float32)
        if img_vec.ndim != 2 or img_vec.shape[1] != _EMBED_DIM:
            raise ValueError(
                f"extract_embeddings must return (B, {_EMBED_DIM}); got {img_vec.shape}."
            )
        logits = (img_vec @ bank.T) * scale  # (1, C)
        row = _softmax_rows(logits)[0]  # (C,)
        probs.append(row.astype(np.float32))
        parcel_ids.append(str(item["parcel_id"]))
        class_ids.append(int(item["class_id"]))

    if probs:
        matrix = np.stack(probs, axis=0).astype(np.float32)
    else:
        matrix = np.zeros((0, bank.shape[0]), dtype=np.float32)
    logger.info(
        "farslip_zeroshot_parcel_proba",
        n_parcels=matrix.shape[0],
        n_classes=matrix.shape[1] if matrix.ndim == 2 else 0,
        logit_scale=scale,
    )
    return matrix, parcel_ids, class_ids


def _scatter_to_semantic18(
    proba: np.ndarray,
    bank_classes: Sequence[int],
) -> np.ndarray:
    """Scatter a ``(n, len(bank_classes))`` matrix into the full ``(n, 18)`` space.

    When the text bank covers fewer than 18 classes (an ``active_class_ids``
    subset), each bank column is placed into its semantic18 column and the rows
    are renormalized. With the full 18-class bank this is the identity.

    Args:
        proba: ``(n, len(bank_classes))`` row-stochastic matrix over bank classes.
        bank_classes: semantic18 indices ``[0..17]`` of the bank columns, in
            column order.

    Returns:
        A ``(n, 18)`` float32 row-stochastic matrix.
    """
    if proba.shape[1] == _NUM_CLASSES and list(bank_classes) == list(range(_NUM_CLASSES)):
        return proba.astype(np.float32)
    full = np.zeros((proba.shape[0], _NUM_CLASSES), dtype=np.float64)
    for col, gid in enumerate(bank_classes):
        if 0 <= int(gid) < _NUM_CLASSES:
            full[:, int(gid)] = proba[:, col]
    row_sums = full.sum(axis=1, keepdims=True)
    full = full / np.where(row_sums < 1e-12, 1.0, row_sums)
    return full.astype(np.float32)


def _zeroshot_frame(
    proba18: np.ndarray,
    canonical_ids: Sequence[str],
) -> pl.DataFrame:
    """Assemble the molde parquet frame (key + prob_* + pred_class + n_pixels).

    Args:
        proba18: ``(n, 18)`` float32 row-stochastic matrix in semantic18 order.
        canonical_ids: ``canonical_parcel_id`` strings, row aligned with
            ``proba18``.

    Returns:
        A Polars DataFrame with the exact molde schema, sorted by the key.
    """
    data: dict[str, object] = {_KEY: list(canonical_ids)}
    for c, name in enumerate(PROB_COLUMNS):
        data[name] = proba18[:, c].astype(np.float32)
    data["pred_class"] = proba18.argmax(axis=1).astype(np.int64)
    # Zero-shot embeds one crop per parcel; there is no per-pixel softmax to count.
    data["n_pixels"] = np.full(proba18.shape[0], -1, dtype=np.int64)
    frame = canonical_parcel_id(pl.DataFrame(data), col=_KEY).sort(_KEY)
    return frame


def _translate_to_canonical_keys(
    raw_parcel_ids: Sequence[str],
    pastis_root: Path,
) -> list[str]:
    """Translate raw ``"{patch}_{instance}"`` keys to canonical ParcelIDs keys.

    The :class:`ParcelCropDataset` keys parcels by the instance-id channel
    (``"{patch}_{instance}"``), but the dense OOF members and the ground truth key
    by the SEPARATE ParcelIDs raster (``"{patch}_{ParcelIDs}"``). This reuses the
    shared bridge :func:`scripts.run_us040_ensembles._instance_to_parcel_id_map`
    (cached per patch) so the materialized OOF aligns on the SAME canonical key as
    every other member; otherwise the stacking inner-join would silently drop
    parcels.

    Args:
        raw_parcel_ids: Dataset keys ``"{patch}_{instance}"``.
        pastis_root: PASTIS-R root (contains ``ANNOTATIONS/``).

    Returns:
        Canonical ``"{patch}_{ParcelIDs}"`` keys, row aligned with the input.

    Raises:
        ValueError: if a raw key is malformed or an instance id has no ParcelIDs
            raster match.
    """
    from ml.utils.parcel_reconcile import (
        instance_to_parcel_id_map as _instance_to_parcel_id_map,
    )

    cache: dict[str, dict[int, int]] = {}
    canonical: list[str] = []
    for raw in raw_parcel_ids:
        text = str(raw)
        patch, sep, inst = text.rpartition("_")
        if not sep or not patch:
            raise ValueError(f"malformed parcel_id {raw!r}; expected '{{patch}}_{{instance}}'.")
        if patch not in cache:
            cache[patch] = _instance_to_parcel_id_map(patch, pastis_root)
        raster_id = cache[patch].get(int(inst))
        if raster_id is None:
            raise ValueError(
                f"instance id {inst} of patch {patch} has no ParcelIDs raster "
                "match; the parcel crops and PASTIS-R rasters are out of sync."
            )
        canonical.append(f"{patch}_{raster_id}")
    logger.info(
        "farslip_zeroshot_keys_translated",
        n_rows=len(canonical),
        n_patches=len(cache),
    )
    return canonical


def materialize_zeroshot_oof(
    *,
    out_path: Path,
    pastis_root: Path,
    farslip_checkpoint: str = DEFAULT_FARSLIP_CHECKPOINT,
    n_classes: int = _NUM_CLASSES,
    class_names: Sequence[str] = SEMANTIC18_CROP_NAMES,
    prompts_per_class: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    extractor: _ZeroShotExtractor | None = None,
    dataset: Any | None = None,
    device: str = "auto",
) -> Path:
    """Materialize ``oof_parcel_farslip-zeroshot_fold5.parquet`` (leak-free).

    Instantiates a FarSLIP zero-shot classifier (FarSLIP vision tower + CLIP-base
    text tower), embeds the held-out fold-5 parcel crops, computes the per-parcel
    semantic18 probabilities and writes them in the SAME schema as the other OOF
    members. Nothing is trained: the text bank depends only on the class
    names/prompts and the ``logit_scale`` is fixed a priori, so the result is a
    true held-out prediction (anti-leakage R-LEAK).

    Args:
        out_path: Destination parquet
            (``oof_parcel_farslip-zeroshot_fold5.parquet``).
        pastis_root: PASTIS-R root, used to build the dataset and to translate
            instance ids to canonical ParcelIDs keys.
        farslip_checkpoint: FarSLIP vision checkpoint (default
            :data:`DEFAULT_FARSLIP_CHECKPOINT`). Passed as ``weights_uri`` so the
            vision tower is FarSLIP and the text tower stays CLIP-base -- the
            requested zero-shot variant. NEVER ``None``.
        n_classes: Active class cardinality (default 18; the full space).
        class_names: Crop names in semantic18 order (default
            :data:`SEMANTIC18_CROP_NAMES`); only the active subset is encoded.
        prompts_per_class: Prompt templates for the text bank.
        logit_scale: Fixed zero-shot temperature.
        extractor: Optional pre-built extractor (dependency injection for tests).
            When ``None`` a :class:`FarSLIPExtractor` is constructed from
            ``farslip_checkpoint``.
        dataset: Optional pre-built dataset (dependency injection for tests). When
            ``None`` a :class:`ParcelCropDataset` over fold-5 is constructed.
        device: Device hint forwarded to :class:`FarSLIPExtractor`.

    Returns:
        The :class:`pathlib.Path` of the written parquet.

    Raises:
        ValueError: if ``n_classes`` is out of range or the produced frame has no
            rows.
    """
    if not 1 <= n_classes <= _NUM_CLASSES:
        raise ValueError(f"n_classes must be in [1, {_NUM_CLASSES}]; got {n_classes}.")

    active = active_classes(n_classes)  # raw PASTIS ids (1..18) in curriculum order
    # semantic18 indices of the active classes (PASTIS c -> c - 1), curriculum order.
    bank_classes = [cid - 1 for cid in active]
    bank_names = [class_names[cid - 1] for cid in active]

    if extractor is None:
        from ml.extractors.farslip_extractor import FarSLIPExtractor

        extractor = FarSLIPExtractor(weights_uri=farslip_checkpoint, device=device)

    if dataset is None:
        from ml.farslip.parcel_crop_dataset import ParcelCropDataset

        dataset = ParcelCropDataset(
            captions={},
            root=Path(pastis_root),
            folds=(HELD_OUT_FOLD,),
            active_class_ids=active,
        )

    text_bank = build_text_class_bank(extractor, bank_names, prompts_per_class=prompts_per_class)
    proba_local, raw_parcel_ids, _class_ids = zeroshot_parcel_proba(
        extractor, dataset, text_bank, logit_scale=logit_scale
    )
    proba18 = _scatter_to_semantic18(proba_local, bank_classes)
    canonical_ids = _translate_to_canonical_keys(raw_parcel_ids, Path(pastis_root))
    frame = _zeroshot_frame(proba18, canonical_ids)
    if frame.height == 0:
        raise ValueError("fold-5 produced no FarSLIP zero-shot parcels.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    logger.info(
        "farslip_zeroshot_oof_materialized",
        n_parcels=frame.height,
        n_active_classes=n_classes,
        farslip_checkpoint=farslip_checkpoint,
        logit_scale=float(logit_scale),
        path=str(out_path),
    )
    return out_path


__all__ = [
    "DEFAULT_FARSLIP_CHECKPOINT",
    "DEFAULT_LOGIT_SCALE",
    "DEFAULT_PROMPT_TEMPLATES",
    "SEMANTIC18_CROP_NAMES",
    "build_text_class_bank",
    "materialize_zeroshot_oof",
    "zeroshot_parcel_proba",
]
