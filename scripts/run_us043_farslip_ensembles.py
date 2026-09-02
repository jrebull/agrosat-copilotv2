"""US-043 closing run -- Stacking-5 / Blending-5 with the two FarSLIP members.

Extends the US-040 heterogeneous ensembles (3 base members: ``tsvit-pheno``,
``utae``, ``xgb-alphaearth``) with the TWO FarSLIP parcel-level members the
sponsor asked for, to test HONESTLY whether the phenology-contrastive branch adds
complementary signal over the dense + tabular path:

- ``farslip-ft18``: the FarSLIP fine-tuned VISION tower scored by cosine against
  18 VISUAL class prototypes (mean parcel CLS-768 per class over TRAIN folds 1-4),
  softmaxed per parcel into ``prob_000..prob_017``. The student never sees fold-5
  (prototypes from folds 1-4, prediction on fold-5), so the OOF is leak-free.
- ``farslip-zeroshot``: the FarSLIP fine-tuned VISION tower + the CLIP BASE text
  tower (the ``FarSLIPExtractor`` loads only the student ``vision_model``; the
  ``text_model`` / ``text_projection`` keep CLIP base weights). Per-parcel 512-dim
  image embedding scored by cosine against the 18 encoded class-name prompts with a
  FIXED logit scale (never tuned against fold-5, R-LEAK), softmaxed per parcel.

Pipeline on fold-5 (anti-leakage R-LEAK):

    1. Assume/materialize ``oof_parcel_farslip-zeroshot_fold5.parquet`` and
       ``oof_parcel_farslip-ft18_fold5.parquet`` (leak-free; folds 1-4 -> fold-5).
    2. Fill the parcels MISSING from the FarSLIP OOF (parcels with no usable
       peak-NDVI crop) with the uniform prior ``1/18`` so the FarSLIP universe is
       comparable with the 0.747 Stacking US-040 universe (NEVER an inner-join
       that would silently drop them, R-LEAK / R-MISSING).
    3. Fit ``StackingEnsemble`` (logreg meta, OOF spatial sub-folds) and
       ``BlendingEnsemble`` (Optuna simplex, spatial holdout) over the FIVE
       members, plus the THREE-member references on the SAME fold-5 parcels.
    4. Compare HONESTLY 5 vs 3 (18-class; delta in ``oof_cv_metrics_``).
    5. Write ``reports/ensemble/metrics/us043_farslip_stacking_blending.csv`` and
       log one MLflow run per ensemble (``data_version`` + ``code_version``).

Reuses the US-040/042 helpers (``build_parcel_ground_truth``,
``build_parcel_geometries``, ``_fold5_patch_ids``, ``_geoms_for_blending``,
``tabular_parcel_keys``), ``StackingEnsemble``, ``BlendingEnsemble``,
``FarSLIPExtractor``, ``ParcelCropDataset`` and the shared canonical-id bridge. It
does NOT modify ``ml/ensemble/stacking.py``, ``base.py`` or ``blending.py`` (they
are already member-generic). Real PASTIS-R + real FarSLIP checkpoint only.

Conventions: Polars (never pandas), numpy/torch only at the boundary, structlog,
typer, type hints, Google-style English docstrings, Spanish prose, no emojis.

Usage (on the H100 VM, env ``agrosat``)::

    python -m scripts.run_us043_farslip_ensembles run \\
        --farslip-checkpoint checkpoints/farslip/parcel/18cls/best.safetensors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import polars as pl
import structlog
import typer

from ml.ensemble.base import EnsembleModel
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    from ml.farslip.parcel_crop_dataset import ParcelCropDataset

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help="US-043 FarSLIP Stacking-5 / Blending-5 run.")

_HELD_OUT_FOLD: int = EnsembleModel.HELD_OUT_FOLD

#: Canonical key column shared by every parcel OOF frame and the GT frame.
_KEY: str = "canonical_parcel_id"

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = len(PROB_COLUMNS)

#: The two new FarSLIP parcel members and the three heterogeneous base ones.
#: The base TSViT is ``tsvit-pheno-fullm`` (0.6764 macro-F1, the BEST TSViT the
#: sponsor selected for this layer), NOT the older ``tsvit-pheno`` (0.6253) that
#: backed the US-040 champion. So Stacking-3 here is a NEW fullm-based reference,
#: and the 5-vs-3 delta is measured on the SAME fullm base (apples to apples).
_FARSLIP_MEMBERS: tuple[str, ...] = ("farslip-ft18", "farslip-zeroshot")
_BASE_MEMBERS_3: tuple[str, ...] = ("tsvit-pheno-fullm", "utae", "xgb-alphaearth")
_BASE_MEMBERS_5: tuple[str, ...] = (*_BASE_MEMBERS_3, *_FARSLIP_MEMBERS)

#: MLflow run names.
_STACK5_RUN = "ensemble-stacking5-farslip"
_BLEND5_RUN = "ensemble-blending5-farslip"

#: TRAIN folds for the leak-free FarSLIP prototypes / zero-shot calibration.
_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3, 4)

#: Fixed zero-shot logit scale (a priori; NEVER tuned against fold-5, R-LEAK).
#: 100.0 is the CLIP default temperature (``logit_scale = ln(100)``).
_ZEROSHOT_LOGIT_SCALE: float = 100.0

#: Italian/Spanish-neutral English class prompts for the zero-shot text tower.
#: One prompt per PASTIS agronomic class id (1..18), in id order.
_CLASS_PROMPT_TEMPLATE: str = "a satellite image of a {name} crop field"


# ---------------------------------------------------------------------------
# Uniform-prior fill for the parcels missing from a FarSLIP OOF (R-MISSING).
# ---------------------------------------------------------------------------


def _fill_missing_parcels_uniform(
    oof_df: pl.DataFrame,
    all_canonical_ids: Sequence[str],
) -> pl.DataFrame:
    """Fill parcels absent from a FarSLIP OOF with the uniform prior ``1/18``.

    A FarSLIP member only predicts parcels whose peak-NDVI crop carries usable
    signal (:class:`ml.farslip.parcel_crop_dataset.ParcelCropDataset` honestly
    drops all-zero crops), so its OOF covers FEWER parcels than the dense / tabular
    members. Inner-joining the members on those would SILENTLY drop the missing
    parcels and change the evaluation universe (away from the 0.747 Stacking US-040
    universe), inflating or deflating the comparison. Instead, this adds one row
    per missing canonical id with a uniform ``1/18`` distribution (no information,
    the honest "abstention" for a parcel FarSLIP could not embed), preserving the
    comparable universe (R-LEAK / R-MISSING).

    The returned frame carries the SAME schema as a materialized parcel OOF
    (``canonical_parcel_id`` + ``prob_000..prob_017`` Float32 summing to 1 +
    ``pred_class`` + ``n_pixels``), sorted by the canonical key, covering EXACTLY
    ``all_canonical_ids`` (existing rows kept verbatim, missing ones uniform).

    Args:
        oof_df: A FarSLIP parcel OOF with ``canonical_parcel_id`` + ``prob_*``
            (and optionally ``pred_class`` / ``n_pixels``).
        all_canonical_ids: The full canonical-id universe to cover (e.g. every
            fold-5 ground-truth parcel).

    Returns:
        A Polars frame covering every id in ``all_canonical_ids`` with the OOF
        schema, missing parcels filled with the uniform prior.

    Raises:
        ValueError: if ``oof_df`` lacks the ``canonical_parcel_id`` or ``prob_*``
            columns.
    """
    if _KEY not in oof_df.columns:
        raise ValueError(f"oof_df must carry the `{_KEY}` column.")
    missing_prob = [c for c in PROB_COLUMNS if c not in oof_df.columns]
    if missing_prob:
        raise ValueError(f"oof_df is missing prob columns: {missing_prob}.")

    base = canonical_parcel_id(oof_df, col=_KEY)
    universe = [str(p) for p in all_canonical_ids]
    present = set(base[_KEY].cast(pl.Utf8).to_list())
    missing_ids = [pid for pid in universe if pid not in present]

    uniform = 1.0 / float(_NUM_CLASSES)
    fill_data: dict[str, object] = {_KEY: missing_ids}
    for col in PROB_COLUMNS:
        fill_data[col] = np.full(len(missing_ids), uniform, dtype=np.float32)
    # argmax of a uniform vector is class 0; n_pixels = -1 marks the abstention.
    fill_data["pred_class"] = np.zeros(len(missing_ids), dtype=np.int64)
    fill_data["n_pixels"] = np.full(len(missing_ids), -1, dtype=np.int64)
    fill_frame = pl.DataFrame(fill_data)

    # Align the existing rows to the canonical schema (add pred_class / n_pixels
    # if absent, cast prob_* to Float32) so the concat is schema-aligned even with
    # extra input columns or a Float64 source OOF.
    base = _ensure_pred_and_pixels(base).with_columns(
        [pl.col(c).cast(pl.Float32) for c in PROB_COLUMNS]
    )
    base = base.select([_KEY, *PROB_COLUMNS, "pred_class", "n_pixels"])
    fill_frame = fill_frame.select([_KEY, *PROB_COLUMNS, "pred_class", "n_pixels"])

    # Restrict the existing rows to the requested universe, then add the fillers.
    base = base.filter(pl.col(_KEY).is_in(universe))
    out = pl.concat([base, fill_frame], how="vertical").sort(_KEY)
    logger.info(
        "farslip_oof_uniform_filled",
        n_universe=len(universe),
        n_present=base.height,
        n_filled=len(missing_ids),
    )
    return out


def _ensure_pred_and_pixels(frame: pl.DataFrame) -> pl.DataFrame:
    """Ensure a parcel frame carries ``pred_class`` + ``n_pixels`` columns.

    Args:
        frame: Parcel frame with the ``prob_*`` columns.

    Returns:
        The frame with ``pred_class`` (argmax of ``prob_*``) and ``n_pixels``
        (defaulting to -1) added if absent.
    """
    if "pred_class" not in frame.columns:
        preds = frame.select(PROB_COLUMNS).to_numpy().argmax(axis=1).astype(np.int64)
        frame = frame.with_columns(pl.Series("pred_class", preds))
    if "n_pixels" not in frame.columns:
        frame = frame.with_columns(pl.lit(-1, dtype=pl.Int64).alias("n_pixels"))
    # Canonical dtypes so a downstream concat with the uniform fillers aligns.
    return frame.with_columns(
        pl.col("pred_class").cast(pl.Int64),
        pl.col("n_pixels").cast(pl.Int64),
    )


# ---------------------------------------------------------------------------
# FarSLIP parcel OOF materialization (leak-free: folds 1-4 -> fold-5).
# ---------------------------------------------------------------------------


def _parcel_crop_dataset(
    *, pastis_root: Path, folds: Sequence[int], n_classes: int
) -> ParcelCropDataset:
    """Build a :class:`ParcelCropDataset` over the requested folds (DRY)."""
    from ml.farslip.parcel_crop_dataset import ParcelCropDataset
    from ml.farslip.pastis_pair_dataset import active_classes

    return ParcelCropDataset(
        captions={},
        root=pastis_root,
        folds=tuple(int(f) for f in folds),
        active_class_ids=active_classes(n_classes),
    )


def _embed_dataset(
    extractor, dataset: ParcelCropDataset, *, batch_size: int = 32
) -> tuple[list[str], list[int], np.ndarray]:
    """Embed every parcel crop of a dataset to the 512-dim CLIP-projection space.

    Args:
        extractor: A :class:`ml.extractors.farslip_extractor.FarSLIPExtractor`.
        dataset: The :class:`ParcelCropDataset` to embed.
        batch_size: Inference batch size.

    Returns:
        Tuple ``(parcel_ids, class_ids, embeds)`` where ``parcel_ids`` are the
        instance-space ``"{patch}_{instance}"`` keys (row order), ``class_ids`` the
        RAW PASTIS class ids (1..18) and ``embeds`` an ``(N, 512)`` L2-normalized
        ``float32`` matrix.
    """
    import torch
    from torch.utils.data import DataLoader

    from ml.farslip.parcel_crop_dataset import collate_parcel_batch

    parcel_ids: list[str] = []
    class_ids: list[int] = []
    chunks: list[np.ndarray] = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_parcel_batch,
    )
    with torch.inference_mode():
        for batch in loader:
            embeds = extractor.extract_embeddings(batch["images"])  # (B, 512)
            chunks.append(embeds.cpu().numpy().astype(np.float32))
            parcel_ids.extend(batch["parcel_ids"])
            class_ids.extend(int(c) for c in batch["class_ids"].tolist())
    matrix = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 512), dtype=np.float32)
    return parcel_ids, class_ids, matrix


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax of a ``(N, C)`` logit matrix."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _scatter_to_18(proba_curriculum: np.ndarray, curriculum_ids: Sequence[int]) -> np.ndarray:
    """Scatter a per-parcel softmax over curriculum classes to the 18-class space.

    The FarSLIP heads emit a distribution over the active curriculum classes (the
    RAW PASTIS ids 1..18 in curriculum order); the harness lives in the contiguous
    ``semantic18`` space ``[0..17]`` (agronomic ``c`` -> ``c - 1``). This places
    each curriculum column at slot ``class_id - 1`` and renormalizes, mirroring
    ``farslip_cosine_map`` and ``materialize_xgb_parcel_oof``.

    Args:
        proba_curriculum: ``(N, n_curriculum)`` post-softmax matrix.
        curriculum_ids: RAW PASTIS class ids (1..18) of the columns, in order.

    Returns:
        ``(N, 18)`` post-softmax matrix in the contiguous space, summing to 1.
    """
    n = proba_curriculum.shape[0]
    full = np.zeros((n, _NUM_CLASSES), dtype=np.float64)
    for col, cid in enumerate(curriculum_ids):
        slot = int(cid) - 1
        if 0 <= slot < _NUM_CLASSES:
            full[:, slot] = proba_curriculum[:, col]
    row_sums = full.sum(axis=1, keepdims=True)
    return full / np.where(row_sums < 1e-12, 1.0, row_sums)


def _write_parcel_oof(
    *,
    instance_keys: Sequence[str],
    proba_18: np.ndarray,
    pastis_root: Path,
    out_path: Path,
) -> Path:
    """Translate instance keys to canonical ids and write a parcel OOF parquet.

    Mirrors the tail of :func:`materialize_xgb_parcel_oof`: the FarSLIP samples key
    parcels by the instance-id space (``"{patch}_{instance}"``), so they must be
    bridged to the canonical ParcelIDs space before they align with the dense OOF
    and the ground truth.

    Args:
        instance_keys: ``"{patch}_{instance}"`` keys aligned with ``proba_18``.
        proba_18: ``(N, 18)`` post-softmax matrix.
        pastis_root: PASTIS-R root for the instance->ParcelIDs bridge.
        out_path: Destination parquet.

    Returns:
        The written :class:`pathlib.Path`.
    """
    from ml.utils.parcel_reconcile import (
        instance_to_parcel_id_map as _instance_to_parcel_id_map,
    )

    cache: dict[str, dict[int, int]] = {}
    canonical: list[str] = []
    keep_rows: list[int] = []
    for row, key in enumerate(instance_keys):
        patch, _, inst = str(key).rpartition("_")
        if patch not in cache:
            cache[patch] = _instance_to_parcel_id_map(patch, pastis_root)
        raster_id = cache[patch].get(int(inst))
        if raster_id is None:
            # Instance with no ParcelIDs match (rare); drop it (uniform-filled later).
            continue
        canonical.append(f"{patch}_{raster_id}")
        keep_rows.append(row)

    kept = proba_18[keep_rows] if keep_rows else np.zeros((0, _NUM_CLASSES))
    data: dict[str, object] = {_KEY: canonical}
    for c, name in enumerate(PROB_COLUMNS):
        data[name] = kept[:, c].astype(np.float32)
    data["pred_class"] = (
        kept.argmax(axis=1).astype(np.int64) if kept.shape[0] else np.zeros(0, dtype=np.int64)
    )
    data["n_pixels"] = np.full(kept.shape[0], -1, dtype=np.int64)
    frame = canonical_parcel_id(pl.DataFrame(data), col=_KEY).sort(_KEY)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    logger.info("farslip_parcel_oof_written", path=str(out_path), n_parcels=frame.height)
    return out_path


def materialize_farslip_ft18_oof(
    *,
    farslip_checkpoint: Path,
    pastis_root: Path,
    out_path: Path,
    device: str = "auto",
    n_classes: int = _NUM_CLASSES,
) -> Path:
    """Materialize ``oof_parcel_farslip-ft18_fold5.parquet`` (leak-free).

    Builds 18 VISUAL class prototypes from the TRAIN folds (1-4) parcel CLS
    embeddings of the FarSLIP fine-tuned vision tower, then scores every fold-5
    parcel by cosine against them and softmaxes per parcel. The student never sees
    fold-5 (prototypes from folds 1-4), so the result is a true held-out OOF
    (anti-leakage R-LEAK).

    Args:
        farslip_checkpoint: FarSLIP student checkpoint (the fine-tuned vision tower).
        pastis_root: PASTIS-R root.
        out_path: Destination ``oof_parcel_farslip-ft18_fold5.parquet``.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        n_classes: Number of active curriculum classes (default 18).

    Returns:
        The written parquet path.
    """
    from ml.extractors.farslip_extractor import FarSLIPExtractor
    from ml.farslip.pastis_pair_dataset import active_classes

    extractor = FarSLIPExtractor(weights_uri=str(farslip_checkpoint), device=device)
    curriculum_ids = list(active_classes(n_classes))

    # TRAIN prototypes (folds 1-4): mean L2-normalized 512-dim embedding per class.
    train_ds = _parcel_crop_dataset(
        pastis_root=pastis_root, folds=_TRAIN_FOLDS, n_classes=n_classes
    )
    _tr_ids, tr_classes, tr_embeds = _embed_dataset(extractor, train_ds)
    prototypes = _build_prototypes(tr_classes, tr_embeds, curriculum_ids)

    # Fold-5 parcels: cosine vs prototypes (both L2-normalized -> dot product).
    test_ds = _parcel_crop_dataset(
        pastis_root=pastis_root, folds=(_HELD_OUT_FOLD,), n_classes=n_classes
    )
    te_keys, _te_classes, te_embeds = _embed_dataset(extractor, test_ds)
    logits = te_embeds.astype(np.float64) @ prototypes.T  # (N, C) cosine
    proba_curr = _softmax_rows(logits * _ZEROSHOT_LOGIT_SCALE)
    proba_18 = _scatter_to_18(proba_curr, curriculum_ids)

    return _write_parcel_oof(
        instance_keys=te_keys,
        proba_18=proba_18,
        pastis_root=pastis_root,
        out_path=out_path,
    )


def materialize_farslip_zeroshot_oof(
    *,
    farslip_checkpoint: Path,
    pastis_root: Path,
    out_path: Path,
    device: str = "auto",
    n_classes: int = _NUM_CLASSES,
) -> Path:
    """Materialize ``oof_parcel_farslip-zeroshot_fold5.parquet`` (leak-free).

    The zero-shot variant: the FarSLIP fine-tuned VISION tower (loaded by
    :class:`FarSLIPExtractor`, which only loads the student ``vision_model``) +
    the CLIP BASE text tower (the student left ``text_model`` /
    ``text_projection`` untouched). Each fold-5 parcel's 512-dim image embedding is
    scored by cosine against the 18 encoded class-name prompts with a FIXED logit
    scale (a priori, NEVER tuned against fold-5, R-LEAK) and softmaxed per parcel.
    No fit on fold-5 at all, so the OOF is trivially leak-free.

    Args:
        farslip_checkpoint: FarSLIP student checkpoint (the fine-tuned vision tower).
        pastis_root: PASTIS-R root.
        out_path: Destination ``oof_parcel_farslip-zeroshot_fold5.parquet``.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        n_classes: Number of active curriculum classes (default 18).

    Returns:
        The written parquet path.
    """
    from ml.extractors.farslip_extractor import FarSLIPExtractor
    from ml.farslip.pastis_pair_dataset import active_classes
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    extractor = FarSLIPExtractor(weights_uri=str(farslip_checkpoint), device=device)
    curriculum_ids = list(active_classes(n_classes))

    # Encode the 18 class-name prompts once with the CLIP base text tower.
    prompts = [
        _CLASS_PROMPT_TEMPLATE.format(
            name=str(PASTIS_R_CLASSES.get(int(cid), f"class {cid}")).lower()
        )
        for cid in curriculum_ids
    ]
    text_embeds = extractor.encode_text(prompts).cpu().numpy().astype(np.float64)

    test_ds = _parcel_crop_dataset(
        pastis_root=pastis_root, folds=(_HELD_OUT_FOLD,), n_classes=n_classes
    )
    te_keys, _te_classes, te_embeds = _embed_dataset(extractor, test_ds)
    logits = te_embeds.astype(np.float64) @ text_embeds.T  # (N, C) cosine
    proba_curr = _softmax_rows(logits * _ZEROSHOT_LOGIT_SCALE)
    proba_18 = _scatter_to_18(proba_curr, curriculum_ids)

    return _write_parcel_oof(
        instance_keys=te_keys,
        proba_18=proba_18,
        pastis_root=pastis_root,
        out_path=out_path,
    )


def _build_prototypes(
    class_ids: Sequence[int],
    embeds: np.ndarray,
    curriculum_ids: Sequence[int],
) -> np.ndarray:
    """Mean L2-normalized embedding per curriculum class (VISUAL prototypes).

    Args:
        class_ids: RAW PASTIS class id (1..18) of each embedding row.
        embeds: ``(N, D)`` L2-normalized embeddings.
        curriculum_ids: RAW PASTIS class ids of the prototype rows, in order.

    Returns:
        ``(C, D)`` ``float64`` bank; row order == ``curriculum_ids``; each
        non-empty row L2-normalized (an empty class keeps a zero row).
    """
    dim = embeds.shape[1] if embeds.shape[0] else 512
    row_of = {int(cid): i for i, cid in enumerate(curriculum_ids)}
    sums = np.zeros((len(curriculum_ids), dim), dtype=np.float64)
    counts = np.zeros(len(curriculum_ids), dtype=np.int64)
    for emb, cid in zip(embeds, class_ids, strict=True):
        row = row_of.get(int(cid))
        if row is None:
            continue
        sums[row] += emb.astype(np.float64)
        counts[row] += 1
    bank = np.zeros((len(curriculum_ids), dim), dtype=np.float64)
    for row, n in enumerate(counts):
        if n == 0:
            continue
        mean = sums[row] / float(n)
        norm = float(np.linalg.norm(mean))
        bank[row] = mean / norm if norm > 1e-12 else mean
    logger.info(
        "farslip_prototypes_built",
        n_classes=len(curriculum_ids),
        per_class_counts=counts.tolist(),
    )
    return bank


# ---------------------------------------------------------------------------
# Ensemble assembly (member-generic; no class change to stacking/blending).
# ---------------------------------------------------------------------------


def _stacking_metrics(
    members: Sequence[str],
    *,
    parcel_geoms: pl.DataFrame,
    parcel_gt: pl.DataFrame,
    oof_dir: Path,
    n_spatial_folds: int,
    random_state: int,
    meta: str,
) -> dict[str, float]:
    """Fit a stacking ensemble over ``members`` and score it HELD-OUT on fold-5.

    Mirrors the US-040 champion path (``predict_proba`` + ``evaluate(fold=5)``)
    EXACTLY so the number is comparable to the 0.747 champion and to the
    BlendingEnsemble (also fold-5 held-out). Returning ``oof_cv_metrics_`` instead
    would report the pessimistic spatial sub-fold CV (a DIFFERENT, lower metric, not
    comparable to 0.747), so it is NOT used here.

    Args:
        members: Ordered base-member names.
        parcel_geoms: fold-5 parcel geometry frame.
        parcel_gt: fold-5 parcel ground-truth labels.
        oof_dir: OOF directory.
        n_spatial_folds: geographic sub-folds for the meta-learner OOF features.
        random_state: seed.
        meta: meta-learner family (``logreg`` | ``xgb``).

    Returns:
        ``{"f1_macro": ..., "accuracy": ...}`` of the fold-5 held-out prediction.
    """
    from ml.ensemble.stacking import StackingEnsemble
    from scripts.run_us040_ensembles import _aligned_labels

    stack = StackingEnsemble(
        base_members=tuple(members),
        meta=meta,  # type: ignore[arg-type]
        n_spatial_folds=n_spatial_folds,
        oof_dir=oof_dir,
        random_state=random_state,
    )
    stack.fit(parcel_geoms, gt_labels=parcel_gt)
    proba = stack.predict_proba()
    keys, _, _ = stack.build_meta_features(gt_labels=None)
    labels = _aligned_labels(keys[_KEY].to_list(), parcel_gt)
    return stack.evaluate(y_true=labels, proba=proba, fold=_HELD_OUT_FOLD)


def _blending_metrics(
    members: Sequence[str],
    *,
    parcel_geoms: pl.DataFrame,
    parcel_gt: pl.DataFrame,
    oof_dir: Path,
    n_trials: int,
    random_state: int,
) -> dict[str, float]:
    """Fit a blending ensemble over ``members`` and score it on fold-5.

    Args:
        members: Ordered base-member names.
        parcel_geoms: fold-5 parcel geometry frame.
        parcel_gt: fold-5 parcel ground-truth labels.
        oof_dir: OOF directory.
        n_trials: Optuna trials.
        random_state: seed.

    Returns:
        ``{"f1_macro": ..., "accuracy": ...}`` of the blended fold-5 prediction.
    """
    from ml.ensemble.blending import BlendingEnsemble
    from scripts.run_us040_ensembles import _aligned_labels, _geoms_for_blending

    geoms_gdf = _geoms_for_blending(parcel_geoms)
    blending = BlendingEnsemble(
        base_members=tuple(members),
        n_trials=n_trials,
        oof_dir=oof_dir,
        random_state=random_state,
    ).fit(geoms_gdf, y_true=parcel_gt)
    proba = blending.predict_proba()
    labels = _aligned_labels(blending._member_ids, parcel_gt)
    return blending.evaluate(y_true=labels, proba=proba, fold=_HELD_OUT_FOLD)


# ---------------------------------------------------------------------------
# Typer command.
# ---------------------------------------------------------------------------


@app.command()
def run(
    farslip_checkpoint: Annotated[Path, typer.Option("--farslip-checkpoint")] = Path(
        "checkpoints/farslip/parcel/18cls/best.safetensors"
    ),
    oof_dir: Annotated[Path, typer.Option("--oof-dir")] = Path("ml/eval/oof"),
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("reports/ensemble"),
    meta: Annotated[str, typer.Option("--meta", help="logreg | xgb")] = "logreg",
    n_spatial_folds: Annotated[int, typer.Option("--n-spatial-folds")] = 5,
    n_trials_blending: Annotated[int, typer.Option("--n-trials-blending")] = 50,
    device: Annotated[str, typer.Option("--device")] = "auto",
    materialize: Annotated[bool, typer.Option("--materialize/--no-materialize")] = True,
    use_mlflow: Annotated[bool, typer.Option("--use-mlflow/--no-mlflow")] = True,
    random_state: Annotated[int, typer.Option("--random-state")] = 42,
) -> None:
    """Run the Stacking-5 / Blending-5 FarSLIP closing pipeline on fold-5."""
    from scripts.run_us040_ensembles import (
        _fold5_patch_ids,
        build_parcel_geometries,
        build_parcel_ground_truth,
    )

    patch_ids = _fold5_patch_ids(oof_dir)
    logger.info("us043_run_start", n_patches=len(patch_ids), meta=meta)

    # Fase 1: assume/materialize the two FarSLIP parcel OOF parquets (leak-free).
    ft18_path = oof_dir / f"oof_parcel_farslip-ft18_fold{_HELD_OUT_FOLD}.parquet"
    zeroshot_path = oof_dir / f"oof_parcel_farslip-zeroshot_fold{_HELD_OUT_FOLD}.parquet"
    if materialize:
        if not ft18_path.is_file():
            materialize_farslip_ft18_oof(
                farslip_checkpoint=farslip_checkpoint,
                pastis_root=pastis_root,
                out_path=ft18_path,
                device=device,
            )
        if not zeroshot_path.is_file():
            materialize_farslip_zeroshot_oof(
                farslip_checkpoint=farslip_checkpoint,
                pastis_root=pastis_root,
                out_path=zeroshot_path,
                device=device,
            )

    # Fold-5 GT + geometry (GT is NOT in the OOF dump) -> the comparable universe.
    parcel_gt = build_parcel_ground_truth(patch_ids, pastis_root)
    parcel_geoms = build_parcel_geometries(patch_ids, pastis_root)
    universe = parcel_gt[_KEY].cast(pl.Utf8).to_list()

    # Fase 2: uniform-fill the FarSLIP OOF to the comparable universe (R-MISSING).
    for path in (ft18_path, zeroshot_path):
        filled = _fill_missing_parcels_uniform(pl.read_parquet(path), universe)
        filled.write_parquet(path)

    # Fase 3: Stacking-5 / Blending-5 vs the 3-member references (same parcels).
    stack5 = _stacking_metrics(
        _BASE_MEMBERS_5,
        parcel_geoms=parcel_geoms,
        parcel_gt=parcel_gt,
        oof_dir=oof_dir,
        n_spatial_folds=n_spatial_folds,
        random_state=random_state,
        meta=meta,
    )
    stack3 = _stacking_metrics(
        _BASE_MEMBERS_3,
        parcel_geoms=parcel_geoms,
        parcel_gt=parcel_gt,
        oof_dir=oof_dir,
        n_spatial_folds=n_spatial_folds,
        random_state=random_state,
        meta=meta,
    )
    blend5 = _blending_metrics(
        _BASE_MEMBERS_5,
        parcel_geoms=parcel_geoms,
        parcel_gt=parcel_gt,
        oof_dir=oof_dir,
        n_trials=n_trials_blending,
        random_state=random_state,
    )
    blend3 = _blending_metrics(
        _BASE_MEMBERS_3,
        parcel_geoms=parcel_geoms,
        parcel_gt=parcel_gt,
        oof_dir=oof_dir,
        n_trials=n_trials_blending,
        random_state=random_state,
    )

    stack_delta = stack5.get("f1_macro", 0.0) - stack3.get("f1_macro", 0.0)
    blend_delta = blend5.get("f1_macro", 0.0) - blend3.get("f1_macro", 0.0)

    # Fase 4: honest comparison table (CSV) + summary.
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(
        {
            "modelo": [
                "Stacking-3 (US-040 universe)",
                "Stacking-5 (+farslip-ft18,+farslip-zeroshot)",
                "Blending-3 (US-040 universe)",
                "Blending-5 (+farslip-ft18,+farslip-zeroshot)",
            ],
            "f1_macro": [
                stack3.get("f1_macro", float("nan")),
                stack5.get("f1_macro", float("nan")),
                blend3.get("f1_macro", float("nan")),
                blend5.get("f1_macro", float("nan")),
            ],
            "accuracy": [
                stack3.get("accuracy", float("nan")),
                stack5.get("accuracy", float("nan")),
                blend3.get("accuracy", float("nan")),
                blend5.get("accuracy", float("nan")),
            ],
            "delta_f1_vs_3": [
                0.0,
                stack_delta,
                0.0,
                blend_delta,
            ],
            "nota": [
                "referencia 3 miembros (espacio semantic18, fold-5)",
                f"5 miembros; delta {stack_delta:+.4f} vs Stacking-3",
                "referencia 3 miembros (espacio semantic18, fold-5)",
                f"5 miembros; delta {blend_delta:+.4f} vs Blending-3",
            ],
        }
    )
    csv_path = metrics_dir / "us043_farslip_stacking_blending.csv"
    table.write_csv(csv_path)

    summary = {
        "members_5": list(_BASE_MEMBERS_5),
        "members_3": list(_BASE_MEMBERS_3),
        "meta": meta,
        "n_universe": len(universe),
        "stacking_5_oof_cv": stack5,
        "stacking_3_oof_cv": stack3,
        "blending_5_fold5": blend5,
        "blending_3_fold5": blend3,
        "stacking_f1_delta_5_vs_3": stack_delta,
        "blending_f1_delta_5_vs_3": blend_delta,
        "label_space": "semantic18 (18 clases); HCAT-6 0.6535 es otro eje, no comparable",
        "zeroshot_logit_scale": _ZEROSHOT_LOGIT_SCALE,
    }
    (out_dir / "us043_farslip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "us043_run_done",
        stack5_f1=round(stack5.get("f1_macro", 0.0), 4),
        stack3_f1=round(stack3.get("f1_macro", 0.0), 4),
        stack_delta=round(stack_delta, 4),
        blend5_f1=round(blend5.get("f1_macro", 0.0), 4),
        blend3_f1=round(blend3.get("f1_macro", 0.0), 4),
        blend_delta=round(blend_delta, 4),
    )

    if use_mlflow:
        ens = _MlflowLogger(oof_dir=oof_dir, random_state=random_state)
        ens.log_to_mlflow(
            {
                "f1_macro": stack5.get("f1_macro", 0.0),
                "accuracy": stack5.get("accuracy", 0.0),
                "f1_macro_delta_vs_3": stack_delta,
            },
            run_name=_STACK5_RUN,
            params={
                "base_members": ",".join(_BASE_MEMBERS_5),
                "meta": meta,
                "n_spatial_folds": n_spatial_folds,
                "zeroshot_logit_scale": _ZEROSHOT_LOGIT_SCALE,
                "label_space": "semantic18",
            },
        )
        ens.log_to_mlflow(
            {
                "f1_macro": blend5.get("f1_macro", 0.0),
                "accuracy": blend5.get("accuracy", 0.0),
                "f1_macro_delta_vs_3": blend_delta,
            },
            run_name=_BLEND5_RUN,
            params={
                "base_members": ",".join(_BASE_MEMBERS_5),
                "n_trials": n_trials_blending,
                "zeroshot_logit_scale": _ZEROSHOT_LOGIT_SCALE,
                "label_space": "semantic18",
            },
        )

    typer.echo(json.dumps(summary, ensure_ascii=False))


class _MlflowLogger(EnsembleModel):
    """Concrete :class:`EnsembleModel` used only for ``log_to_mlflow`` (DRY).

    The base ``log_to_mlflow`` carries the mandatory ``data_version`` +
    ``code_version`` tags; this thin subclass satisfies the abstract contract so
    the orchestrator can log the two 5-member runs without re-implementing it.
    """

    def fit(self, *args: object, **kwargs: object) -> _MlflowLogger:
        """No-op fit (this concrete subclass exists only to log to MLflow)."""
        return self

    def predict_proba(self, *args: object, **kwargs: object) -> np.ndarray:
        """Unsupported: this logger-only ensemble produces no probabilities."""
        raise NotImplementedError("logger-only ensemble has no predict_proba.")


if __name__ == "__main__":
    app()
