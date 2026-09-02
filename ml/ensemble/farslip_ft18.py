"""E-b -- FarSLIP fine-tuned head to 18 classes (US-042, EPIC 6).

Second incremental member ordered by the sponsor: a NATIVE 18-class crop
classifier built on the FarSLIP fine-tuned vision tower. Where E-a
(:mod:`ml.ensemble.dual_head_fusion`) reuses the N=4 curriculum head and SCATTERS
its 4 logits into the 18-class space, this member trains a fresh supervised head
DIRECTLY on the 18 agronomic classes: a multinomial ``LogisticRegression`` over
the FarSLIP 512-dim image embedding (``extract_embeddings`` -- the CLIP visual
projection, L2-normalized). There is no 4->18 scatter -- the head is 18-class
native, calibrated by ``predict_proba``.

It is the exact leak-free mold of
:func:`scripts.run_us040_ensembles.materialize_xgb_parcel_oof`, with the AlphaEarth
feature matrix replaced by FarSLIP embeddings:

1. Fit the head on folds 1-4 ONLY (the model never sees fold-5).
2. Predict the fold-5 parcels and dump their POST-softmax ``prob_000..prob_017``.
3. The result is therefore a true held-out OOF (anti-leakage R-LEAK rule #1).

The "fine-tuned" tower is the FarSLIP student: ``FarSLIPExtractor(weights_uri=
<checkpoint>)`` loads the fine-tuned ``vision_model`` (the ``text_model`` stays at
CLIP base, irrelevant here -- this member is image-only). The RAW PASTIS
``class_id`` (1..18) is mapped to the contiguous ``semantic18`` space ``[0..17]``
with the SAME LUT the dense members use (:func:`_build_semantic18_lut`), dropping
Background/Void. Missing fold-5 parcels (never produced by the dataset, e.g.
all-zero crops) are NOT dropped downstream: the consuming ensemble fills them with
the uniform prior ``1/18`` via the canonical reconcile join.

The dataset (:class:`ml.farslip.parcel_crop_dataset.ParcelCropDataset`) keys
parcels by ``"{patch}_{instance}"`` (the PASTIS instance channel ``TARGET[1]``);
the dense OOF / ground truth key by the SEPARATE ``ParcelIDs`` raster. Every key
is translated to the canonical ``"{patch}_{parcel_raster_id}"`` space via the same
instance->ParcelIDs bridge the tabular member uses, so this OOF inner-joins the
rest without silently dropping parcels.

Project conventions: ``polars`` (never pandas) for tabular output, ``numpy`` only
at the array boundary, ``torch`` for the student, ``structlog`` for logging, type
hints and Google-style docstrings; visible prose Spanish, code identifiers
English; real PASTIS-R French data only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl
import structlog

from ml.farslip.pastis_pair_dataset import _DEFAULT_PASTIS_ROOT
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import torch

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_FARSLIP_CHECKPOINT",
    "EmbeddingExtractor",
    "materialize_ft18_oof",
]

#: Canonical key column shared by every parcel frame (dense OOF + ground truth).
_KEY: str = "canonical_parcel_id"

#: Number of contiguous agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: The only fold whose parcels are predicted/reported (anti-leakage R-LEAK).
HELD_OUT_FOLD: int = 5

#: Training folds: everything BUT the held-out fold (the head never sees fold-5).
_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3, 4)

#: FarSLIP image embedding dimension (``visual_projection`` output, L2-norm).
_EMBED_DIM: int = 512

#: Numerical floor to avoid divide-by-zero when renormalizing a prob vector.
_RENORM_EPS: float = 1e-12

#: Default fine-tuned parcel-level FarSLIP student (US-036-b). Relative path lands
#: on F: on the VM; override per run via ``farslip_checkpoint``.
DEFAULT_FARSLIP_CHECKPOINT: str = "checkpoints/farslip/parcel/04cls/best.safetensors"

#: Inference batch size for the embedding extraction.
_DEFAULT_BATCH_SIZE: int = 64

#: DataLoader workers for the per-parcel crop preparation (CPU-bound: disk read +
#: peak-NDVI + crop + resize). 0 starves the GPU; a high value on the H100 keeps
#: the forward fed. Auto-resolved from CPU count when not overridden.
_DEFAULT_NUM_WORKERS: int = 8

#: Emit an embedding-extraction progress log roughly every this many parcels.
_EMBED_LOG_EVERY: int = 4096


class EmbeddingExtractor(Protocol):
    """Structural type of the FarSLIP embedding extractor used by this member.

    The real implementation is
    :class:`ml.extractors.farslip_extractor.FarSLIPExtractor`; tests inject a
    deterministic stub with the same single method. Only the 512-dim,
    L2-normalized image embedding is consumed (the text tower is irrelevant).
    """

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 512)`` L2-normalized image embeddings for ``crops``."""
        ...


def _embed_dataset(
    dataset: Any,
    extractor: EmbeddingExtractor,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the extractor over a parcel dataset, returning embeddings + labels + ids.

    Iterates ``dataset`` with the parcel collate, extracts the FarSLIP 512-dim
    image embedding per parcel crop and accumulates the RAW PASTIS ``class_id``
    (1..18) and the dataset key ``"{patch}_{instance}"`` (untranslated -- the
    canonical translation happens once, on the fold-5 keys only).

    Args:
        dataset: A :class:`ParcelCropDataset` (or stub) exposing the
            ``collate_parcel_batch`` contract (``images``, ``class_ids``,
            ``parcel_ids``).
        extractor: The FarSLIP embedding extractor.
        batch_size: Inference batch size.

    Returns:
        ``(X, y_raw, parcel_ids)`` where ``X`` is ``(N, 512)`` float64, ``y_raw``
        is ``(N,)`` int64 RAW PASTIS class ids, and ``parcel_ids`` is the list of
        ``"{patch}_{instance}"`` keys (row order preserved).

    Raises:
        ValueError: if the dataset is empty.
    """
    from torch.utils.data import DataLoader

    from ml.farslip.parcel_crop_dataset import collate_parcel_batch

    if len(dataset) == 0:
        raise ValueError("parcel dataset is empty; cannot extract embeddings.")

    # The bottleneck is the per-parcel crop preparation (disk read + peak-NDVI +
    # crop + resize), all on CPU. With num_workers=0 the GPU starves (it idles
    # waiting for each batch). Parallel workers prefetch crops so the GPU forward
    # is the actual rate, turning hours into minutes (R-DATALOADER).
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_parcel_batch,
    )
    n_total = len(dataset)
    n_done = 0
    feats: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for batch in loader:
        emb = extractor.extract_embeddings(batch["images"])
        emb_np = np.asarray(emb.detach().cpu().numpy(), dtype=np.float64)
        if emb_np.ndim != 2 or emb_np.shape[1] != _EMBED_DIM:
            raise ValueError(f"extractor must return (B, {_EMBED_DIM}); got {emb_np.shape}.")
        feats.append(emb_np)
        labels.extend(int(c) for c in batch["class_ids"].tolist())
        n_done += emb_np.shape[0]
        if n_done % _EMBED_LOG_EVERY < batch_size:
            logger.info("ft18_embed_progress", done=n_done, total=n_total)
        ids.extend(str(p) for p in batch["parcel_ids"])

    x = np.concatenate(feats, axis=0) if feats else np.empty((0, _EMBED_DIM))
    x = np.where(np.isfinite(x), x, 0.0)
    y_raw = np.asarray(labels, dtype=np.int64)
    return x, y_raw, ids


def _translate_keys_to_canonical(
    parcel_ids: Sequence[str],
    pastis_root: Path,
) -> list[str]:
    """Translate dataset ``"{patch}_{instance}"`` keys to the canonical space.

    The :class:`ParcelCropDataset` keys parcels by the PASTIS instance channel
    (``TARGET[1]``), while the dense OOF members and the ground truth key by the
    SEPARATE ``ParcelIDs`` raster. This reuses the SAME instance->ParcelIDs bridge
    the tabular member uses (:func:`scripts.run_us040_ensembles._instance_to_parcel_id_map`)
    so the resulting OOF aligns with the rest and the stacking inner-join does not
    drop parcels silently.

    Args:
        parcel_ids: Dataset keys ``"{patch}_{instance}"`` (fold-5 parcels).
        pastis_root: PASTIS-R root, used to read the per-patch id rasters.

    Returns:
        The canonical ``"{patch}_{parcel_raster_id}"`` key per input (order kept).

    Raises:
        ValueError: if a key is malformed or an instance id has no ParcelIDs match.
    """
    from ml.utils.parcel_reconcile import (
        instance_to_parcel_id_map as _instance_to_parcel_id_map,
    )

    cache: dict[str, dict[int, int]] = {}
    keys: list[str] = []
    for raw_key in parcel_ids:
        text = str(raw_key)
        patch, _, instance = text.rpartition("_")
        if not patch or not instance:
            raise ValueError(f"malformed parcel key {text!r}; expected '<patch>_<instance>'.")
        if patch not in cache:
            cache[patch] = _instance_to_parcel_id_map(patch, pastis_root)
        raster_id = cache[patch].get(int(instance))
        if raster_id is None:
            raise ValueError(
                f"instance id {instance} of patch {patch} has no ParcelIDs raster "
                "match; the parcel dataset and PASTIS-R rasters are out of sync."
            )
        keys.append(f"{patch}_{raster_id}")
    logger.info("ft18_keys_translated", n_rows=len(keys), n_patches=len(cache))
    return keys


def materialize_ft18_oof(
    *,
    farslip_checkpoint: str = DEFAULT_FARSLIP_CHECKPOINT,
    out_path: Path,
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    extractor: EmbeddingExtractor | None = None,
    train_dataset: Any | None = None,
    test_dataset: Any | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Path:
    """Materialize ``oof_parcel_farslip-ft18_fold5.parquet`` (E-b member, leak-free).

    Trains an 18-class native head on the FarSLIP image embeddings of folds 1-4
    and predicts the fold-5 parcels, writing their POST-softmax
    ``prob_000..prob_017``. The clone of
    :func:`scripts.run_us040_ensembles.materialize_xgb_parcel_oof` with two
    differences: (a) the feature matrix is the FarSLIP 512-dim image embedding
    (``extractor.extract_embeddings``), not AlphaEarth ``dim_*``; (b) the head is a
    multinomial ``LogisticRegression(class_weight="balanced")``, calibrated by
    ``predict_proba``. Because the head never sees fold-5, the result is a true
    held-out OOF (anti-leakage R-LEAK).

    The RAW PASTIS ``class_id`` (1..18) is mapped to the contiguous ``semantic18``
    space ``[0..17]`` with the dataset's LUT before fitting (Background/Void
    dropped), so the probabilities land in the SAME class columns as every other
    member. The fold-5 keys are translated from the dataset's instance space to the
    canonical ParcelIDs space so this OOF aligns with the dense members.

    Args:
        farslip_checkpoint: Path/URI of the fine-tuned FarSLIP student weights,
            forwarded to :class:`FarSLIPExtractor` (loads the fine-tuned
            ``vision_model``). Ignored when ``extractor`` is injected.
        out_path: Destination ``oof_parcel_farslip-ft18_fold5.parquet``.
        pastis_root: PASTIS-R root (dataset + canonical key translation).
        extractor: Optional pre-built embedding extractor (tests inject a stub).
            Defaults to ``FarSLIPExtractor(weights_uri=farslip_checkpoint)``.
        train_dataset: Optional folds-1-4 parcel dataset (tests inject a stub).
            Defaults to ``ParcelCropDataset(folds=(1,2,3,4))``.
        test_dataset: Optional fold-5 parcel dataset (tests inject a stub).
            Defaults to ``ParcelCropDataset(folds=(5,))``.
        batch_size: Embedding inference batch size.
        max_iter: ``LogisticRegression`` max iterations (default 2000).
        random_state: Deterministic seed for the head.

    Returns:
        The :class:`pathlib.Path` of the written parquet.

    Raises:
        ValueError: if fold-5 has no parcels, or no fold-1-4 parcel survives the
            Background/Void filter.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    from ml.data.pastis_seg_dataset import _build_semantic18_lut

    if extractor is None:
        extractor = _default_extractor(farslip_checkpoint)
    if train_dataset is None:
        train_dataset = _default_dataset(pastis_root, _TRAIN_FOLDS)
    if test_dataset is None:
        test_dataset = _default_dataset(pastis_root, (HELD_OUT_FOLD,))
    train_ds = train_dataset
    test_ds = test_dataset

    pin = num_workers > 0
    x_train, y_raw_pastis, _ = _embed_dataset(
        train_ds,
        extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )
    x_test, _, test_keys = _embed_dataset(
        test_ds,
        extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if x_test.shape[0] == 0:
        raise ValueError("fold-5 has no parcels in the parcel dataset.")

    # Map RAW PASTIS class_id (1..18) -> contiguous semantic18 [0..17]; drop the
    # Background/Void rows so the head learns the SAME 18 columns as every member.
    label_lut = _build_semantic18_lut(255)
    pastis_train = np.clip(y_raw_pastis, 0, 19)
    y_semantic18 = label_lut[pastis_train]
    keep = y_semantic18 != 255
    x_train = x_train[keep]
    y_semantic18 = y_semantic18[keep]
    if x_train.shape[0] == 0:
        raise ValueError("no fold-1-4 parcel survived the Background/Void filter.")

    encoder = LabelEncoder().fit(y_semantic18)
    y_train = encoder.transform(y_semantic18).astype(np.int64)

    # sklearn >= 1.7 dropped the ``multi_class`` arg: multiclass LogisticRegression
    # is always multinomial (softmax) now, which is exactly what this 18-class head
    # needs (calibrated probabilities via ``predict_proba``).
    head = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )
    head.fit(x_train, y_train)

    proba_local = np.asarray(head.predict_proba(x_test), dtype=np.float64)

    # Scatter the per-class probabilities into the global 18-class space. The head
    # is 18-class native (no 4->18 curriculum scatter); this only places each
    # trained class at its semantic18 column and renormalizes (a class absent from
    # the training folds keeps a zero column -- honest "unknown").
    global_classes = encoder.classes_.astype(np.int64)
    full = np.zeros((proba_local.shape[0], _NUM_CLASSES), dtype=np.float64)
    for col, gid in enumerate(global_classes):
        if 0 <= int(gid) < _NUM_CLASSES:
            full[:, int(gid)] = proba_local[:, col]
    row_sums = full.sum(axis=1, keepdims=True)
    full = full / np.where(row_sums < _RENORM_EPS, 1.0, row_sums)

    keys = _translate_keys_to_canonical(test_keys, pastis_root)
    data: dict[str, object] = {_KEY: keys}
    for c, name in enumerate(PROB_COLUMNS):
        data[name] = full[:, c].astype(np.float32)
    data["pred_class"] = full.argmax(axis=1).astype(np.int64)
    data["n_pixels"] = np.full(full.shape[0], -1, dtype=np.int64)  # crop-level: no pixels.
    frame = canonical_parcel_id(pl.DataFrame(data), col=_KEY).sort(_KEY)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    logger.info(
        "ft18_parcel_oof_materialized",
        n_parcels=frame.height,
        n_train=int(x_train.shape[0]),
        n_classes_seen=int(global_classes.size),
        path=str(out_path),
    )
    return out_path


def _default_extractor(farslip_checkpoint: str) -> EmbeddingExtractor:
    """Build the real :class:`FarSLIPExtractor` for the fine-tuned vision tower.

    ``weights_uri=<checkpoint>`` loads ONLY the fine-tuned ``vision_model`` of the
    FarSLIP student; the text tower stays at CLIP base (irrelevant -- this member
    is image-only). Imported lazily so the module loads without ``transformers``.

    Args:
        farslip_checkpoint: Path/URI of the fine-tuned FarSLIP student weights.

    Returns:
        A ready :class:`FarSLIPExtractor`.
    """
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    return FarSLIPExtractor(weights_uri=farslip_checkpoint)


def _default_dataset(pastis_root: Path, folds: Sequence[int]) -> Any:
    """Build the real :class:`ParcelCropDataset` over ``folds`` (all 18 classes).

    Imported lazily so the module loads without the heavy dataset dependencies.
    ``captions={}`` is enough: this member never uses captions (image-only head).

    Args:
        pastis_root: PASTIS-R root.
        folds: Official PASTIS folds to load.

    Returns:
        A :class:`ParcelCropDataset` over the requested folds.
    """
    from ml.farslip.parcel_crop_dataset import ParcelCropDataset

    return ParcelCropDataset(captions={}, root=Path(pastis_root), folds=tuple(folds))
