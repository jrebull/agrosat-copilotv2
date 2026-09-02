"""Productive orchestrator of the FAITHFUL FarSLIP re-training (US-036-a v2).

Re-implements the Li et al. 2025 region-category objective (arXiv:2511.14901,
secs. 3.3, 4.1, 4.3) over the **real French PASTIS-R** dataset. It SUPERSEDES the
impoverished single-positive v1 (``run_us036a_farslip_full_incremental.py``): each
batch carries MULTIPLE region-category pairs (one per PASTIS parcel via
``ParcelIDs``) and a rich global caption, and the loss is the paper-faithful
``L_total = L_glo + lambda_loc * L_loc`` with ``L_loc`` the Multi-Positive
Contrastive Loss (MPCL, eq. 3-4) and ``L_glo`` the symmetric image-caption
InfoNCE (eq. 1-2). v1 is NOT deleted; ``--supervision dominant_v1`` runs it as an
ablation through the same trainer flag.

Reuse, NOT modification (write-set disjoint from T1/T2/T3):

    - :class:`ml.farslip.distill.FarSLIPDistillationTrainer` (INSTANTIATED in
      ``supervision="region_category"``): the AdamW BF16 loop,
      ``set_category_prototypes`` (PASTIS id -> [0, C) map + 384->768 frozen
      lift), ``step_faithful_v2`` (MPCL + L_glo), ``save_student``, and its own
      MLflow ``start_run``/``end_run`` with ``data_version`` + ``code_version``.
    - :class:`ml.farslip.region_category_dataset.RegionCategoryPairDataset` +
      ``collate_region_batch`` (T2): the multi-object PASTIS-R batch.
    - :func:`ml.farslip.caption_cache.load_captions` (T1): the cached global
      captions parquet (training READS it, never re-calls Gemma).
    - :func:`ml.features.phenology_class_prototypes.load_class_prototype_embeddings`
      (US-033): the 18 MiniLM-384 prototypes (read/filter only, NEVER regenerate).

Scope (critical): ONLY real French PASTIS-R. No Italian / synthetic / placeholder
data, no ``data/farslip_pairs``, no AlphaEarth (it feeds the ensemble E-b
US-042). Pointing ``pastis_root`` at ``data/farslip_pairs`` is rejected.

Anti-leakage (spatial CV): train on the official PASTIS ``train_folds``, eval on
the disjoint held-out ``val_folds``; overlapping folds raise via
:func:`ml.farslip.region_category_dataset.assert_disjoint_folds`. The captions
parquet must cover every patch of both splits or the orchestrator fails fast.

Per-class eval (comparable to v1): the student CLS of each held-out patch is
classified by cosine similarity against the 768-dim category prototype bank
(``argmax``), the prediction compared to the patch's majority region category.
From the confusion it derives per-class F1/IoU and a ``v1 vs v2`` delta table.

Project convention: ``structlog`` (no ``print``); type hints; English docstrings,
Spanish prose; no emojis; checkpoints under the relative path
``checkpoints/farslip/faithful_v2/`` (lands on ``F:`` on the VM); MLflow on the
Docker server ``:5010`` (``data_version`` + ``code_version``); spatial CV folds.

Typical productive usage on the H100 (run nvidia-smi first, captions ready)::

    poetry run python -m scripts.run_us036a_v2_farslip_faithful train \\
        --run-name farslip-faithful-v2 \\
        --supervision faithful_v2 \\
        --lambda-loc 1.0 --temperature 0.07 \\
        --n-epochs 30 --batch-size 64 --lr 1e-5 --seed 42 \\
        --folds 1,2,3 --val-folds 4 \\
        --pastis-root data/PASTIS-R \\
        --captions-path data/farslip/pastis_captions.parquet \\
        --output-dir checkpoints/farslip/faithful_v2 \\
        --time-cap-hours 8.0 --mlflow-uri http://localhost:5010
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the run CLI. poetry add typer") from exc

from ml.farslip.caption_cache import load_captions
from ml.farslip.caption_encoder import encode_captions_minilm, make_caption_collate
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.farslip.region_category_dataset import (
    RegionCategoryPairDataset,
    assert_disjoint_folds,
    collate_region_batch,
)
from ml.features.phenology_class_prototypes import load_class_prototype_embeddings
from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.seed import propagate_seed

_log = structlog.get_logger(__name__)

#: Default MLflow tracking server (Docker on :5010); the lineage lives here, NOT
#: in ``./mlruns``. Overridable with ``--mlflow-uri`` (e.g. a SQLite file for CI).
_DEFAULT_MLFLOW_URI = "http://localhost:5010"

#: Sentinel-2 input channels of the composite (B02, B03, B04, B08).
_N_IN_CHANNELS = 4

#: Forbidden Italian/synthetic data root (US-034/035 path, discarded here).
_FORBIDDEN_ROOT_NAME = "farslip_pairs"

#: All 18 agronomic PASTIS categories (1..18) in canonical id order.
_ALL_ACTIVE_CLASS_IDS: tuple[int, ...] = tuple(range(1, 19))

#: F1 threshold for a "well-resolved" class (rubric / stop reporting).
_F1_WELL_RESOLVED = 0.50

#: CLI supervision selector -> trainer ``supervision`` value.
_SUPERVISION_MAP: dict[str, str] = {
    "faithful_v2": "region_category",
    "dominant_v1": "dominant",
}

SupervisionChoice = Literal["faithful_v2", "dominant_v1"]


@dataclass
class FaithfulRunResult:
    """Outcome of one faithful-v2 re-training run.

    Attributes:
        supervision: the trainer supervision mode actually used.
        n_categories: number of active categories (prototype-bank rows).
        class_ids: active PASTIS class_ids (canonical order).
        per_class_f1: per-class F1 over the held-out fold.
        per_class_iou: per-class IoU over the held-out fold.
        macro_f1: macro-averaged F1.
        macro_iou: macro-averaged IoU.
        n_eval: number of evaluated held-out patches.
        n_classes_well_resolved: classes with F1 >= 0.50.
        best_ckpt: path of the best student checkpoint (feeds US-037).
        mean_regions_per_patch: dataset multi-object signal (> 1 for v2).
        train_metrics: final loss dict of the run (last epoch).
    """

    supervision: str
    n_categories: int
    class_ids: list[int]
    per_class_f1: dict[int, float]
    per_class_iou: dict[int, float]
    macro_f1: float
    macro_iou: float
    n_eval: int
    n_classes_well_resolved: int
    best_ckpt: Path
    mean_regions_per_patch: float
    train_metrics: dict[str, float] = field(default_factory=dict)


def _validate_pastis_root(pastis_root: Path) -> None:
    """Reject Italian/synthetic data; require a real PASTIS-R root.

    Args:
        pastis_root: the candidate PASTIS-R root.

    Raises:
        ValueError: if the path is the forbidden Italian/synthetic root.
    """
    parts = {p.lower() for p in pastis_root.parts}
    if _FORBIDDEN_ROOT_NAME in pastis_root.name.lower() or _FORBIDDEN_ROOT_NAME in parts:
        raise ValueError(
            f"pastis_root {pastis_root!s} points at the Italian/synthetic "
            f"'{_FORBIDDEN_ROOT_NAME}' data (US-034/035, discarded). US-036-a v2 "
            "is PASTIS-R-only: pass a real PASTIS-R root (e.g. data/PASTIS-R)."
        )


def _require_captions_for_dataset(
    dataset: RegionCategoryPairDataset, captions: dict[str, str], split: str
) -> None:
    """Fail fast if any kept patch of the split lacks a caption.

    The dataset raises lazily at ``__getitem__`` time; this surfaces the missing
    captions eagerly so a productive run does not crash mid-epoch (AC-5).

    Args:
        dataset: the built split dataset.
        captions: the loaded ``{patch_id: caption}`` map.
        split: human-readable split name for the error (``"train"``/``"val"``).

    Raises:
        ValueError: listing the patch_ids without a caption.
    """
    missing = [pid for pid, _regions in dataset._samples if pid not in captions]
    if missing:
        raise ValueError(
            f"{len(missing)} {split} patches have no caption in the parquet "
            f"(e.g. {missing[:5]}). Generate the captions for every patch of the "
            "split (Phase A) before training; the run does not generate silently."
        )


def _category_prototypes(
    prototype_path: Path | None,
    active_class_ids: tuple[int, ...],
) -> tuple[torch.Tensor, list[int]]:
    """Selects the active category prototypes from the US-033 parquet (read-only).

    Loads the 18 MiniLM-384 prototypes and keeps the rows of ``active_class_ids``
    in canonical id order. The bank stays MiniLM-384; the trainer lifts it to the
    student CLS dim (768) via the same frozen orthogonal map the v1 path uses.

    Args:
        prototype_path: override of the US-033 parquet (default the DVC one).
        active_class_ids: PASTIS class ids to keep (canonical order).

    Returns:
        ``(prototypes (C, 384) float tensor, kept class_ids list)``.

    Raises:
        ValueError: if an active class id is absent from the parquet.
    """
    proto_18, ids_all = (
        load_class_prototype_embeddings(prototype_path)
        if prototype_path is not None
        else load_class_prototype_embeddings()
    )
    id_to_row = {int(cid): row for row, cid in enumerate(ids_all)}
    missing = [c for c in active_class_ids if c not in id_to_row]
    if missing:
        raise ValueError(
            f"active class_ids {missing} are absent from the US-033 prototype "
            f"parquet (has {sorted(id_to_row)}); cannot build the category bank."
        )
    rows = [id_to_row[c] for c in active_class_ids]
    bank = torch.from_numpy(proto_18[rows]).float()
    return bank, list(active_class_ids)


def _patch_majority_category(regions: list[tuple[int, int]]) -> int:
    """Returns the majority region category (PASTIS id) of a patch.

    The held-out patch is classified against ONE label for the v1-comparable
    metric: its most frequent region category (ties -> smallest id, deterministic).

    Args:
        regions: the patch's ``[(parcel_instance_id, category_id PASTIS)]`` list.

    Returns:
        The majority PASTIS category id of the patch.
    """
    counts: dict[int, int] = {}
    for _inst, cat in regions:
        counts[cat] = counts.get(cat, 0) + 1
    # Most frequent; tie broken by the smaller PASTIS id for determinism.
    best_cat = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return int(best_cat)


def _inverse_frequency_weights(
    dataset: RegionCategoryPairDataset, class_ids: list[int]
) -> torch.Tensor:
    """Computes inverse-frequency class weights (canonical order) to fight imbalance.

    Counts how many regions of each category the train dataset has, then assigns
    ``w_c = (N_total / (C * n_c))`` (the sklearn "balanced" form), normalized to
    mean 1 so the overall loss scale is preserved. Rare categories (e.g. Sorghum)
    get a weight > 1, dominant ones (Meadow) < 1, so the MPCL stops collapsing to
    the majority. Categories absent from train get weight 1 (neutral).

    Args:
        dataset: the train region-category dataset.
        class_ids: PASTIS ids in canonical category order (bank row order).

    Returns:
        ``(C,)`` float tensor of weights in canonical order, mean ~1.
    """
    import torch

    counts = {cid: 0 for cid in class_ids}
    for _pid, regions in dataset._samples:
        for _inst, cat in regions:
            if cat in counts:
                counts[cat] += 1
    n_total = sum(counts.values())
    n_cat = len(class_ids)
    raw = []
    for cid in class_ids:
        n_c = counts[cid]
        raw.append(n_total / (n_cat * n_c) if n_c > 0 else 1.0)
    weights = torch.tensor(raw, dtype=torch.float32)
    weights = weights / weights.mean().clamp(min=1e-8)  # normalize to mean 1
    return weights


@torch.no_grad()
def eval_per_class_v2(
    student: torch.nn.Module,
    val_dataset: RegionCategoryPairDataset,
    category_prototypes: torch.Tensor,
    class_ids: list[int],
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    f1_well_resolved: float = _F1_WELL_RESOLVED,
) -> FaithfulRunResult:
    """Evaluate the v2 student per class by patch-CLS<->category cosine argmax.

    Comparable to the v1 metric: each held-out patch produces a CLS token (768);
    it is compared by cosine similarity against the 768-dim category prototype
    bank and ``argmax`` is the predicted category index. The true label is the
    patch's MAJORITY region category. From the confusion it derives, per category,
    the one-vs-rest F1/IoU::

        F1  = 2 * TP / (2 * TP + FP + FN)
        IoU = TP / (TP + FP + FN)

    Deterministic: ``eval`` mode under ``no_grad`` over the dataset in order.

    Args:
        student: the trained CLIP vision student.
        val_dataset: held-out PASTIS region-category dataset (``val_folds``).
        category_prototypes: ``(C, 768)`` reprojected category bank (the trainer's
            ``_category_prototypes`` after ``set_category_prototypes``).
        class_ids: PASTIS class ids in the SAME row order as the bank.
        device: torch device (defaults to the student's device).
        batch_size: evaluation batch size (in patches).
        f1_well_resolved: F1 threshold for "well-resolved".

    Returns:
        A :class:`FaithfulRunResult` with per-class and macro metrics; the
        ``best_ckpt``/``train_metrics``/``mean_regions_per_patch`` fields are
        filled by the caller.

    Raises:
        ValueError: if the bank row count does not match ``len(class_ids)``.
    """
    n_categories = len(class_ids)
    if category_prototypes.shape[0] != n_categories:
        raise ValueError(
            f"category_prototypes rows ({category_prototypes.shape[0]}) must equal "
            f"len(class_ids) ({n_categories})."
        )
    if device is None:
        device = next(student.parameters()).device
    student_was_training = student.training
    student.eval()

    protos_n = torch.nn.functional.normalize(
        category_prototypes.to(device=device, dtype=torch.float32), p=2, dim=-1
    )
    pastis_to_idx = {cid: idx for idx, cid in enumerate(class_ids)}

    tp = np.zeros(n_categories, dtype=np.int64)
    fp = np.zeros(n_categories, dtype=np.int64)
    fn = np.zeros(n_categories, dtype=np.int64)
    n_eval = 0

    # Iterate the dataset in order; classify each patch by its majority category.
    images_buf: list[torch.Tensor] = []
    targets_buf: list[int] = []

    def _flush() -> None:
        nonlocal n_eval
        if not images_buf:
            return
        batch = torch.stack(images_buf, dim=0).to(device=device, dtype=torch.float32)
        out = student(pixel_values=batch, output_hidden_states=False)
        cls = out.last_hidden_state[:, 0, :]
        cls_n = torch.nn.functional.normalize(cls.float(), p=2, dim=-1)
        preds = (cls_n @ protos_n.t()).argmax(dim=-1).cpu().numpy()
        for pred, true in zip(preds, targets_buf, strict=True):
            n_eval += 1
            if pred == true:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
        images_buf.clear()
        targets_buf.clear()

    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        _pid, regions = val_dataset._samples[idx]
        majority = _patch_majority_category(regions)
        if majority not in pastis_to_idx:
            continue  # patch dominated by a class outside the active set
        images_buf.append(item["image"])
        targets_buf.append(pastis_to_idx[majority])
        if len(images_buf) >= batch_size:
            _flush()
    _flush()

    if student_was_training:
        student.train()

    per_class_f1: dict[int, float] = {}
    per_class_iou: dict[int, float] = {}
    for idx, cid in enumerate(class_ids):
        denom_f1 = 2 * tp[idx] + fp[idx] + fn[idx]
        denom_iou = tp[idx] + fp[idx] + fn[idx]
        per_class_f1[cid] = float(2 * tp[idx] / denom_f1) if denom_f1 > 0 else 0.0
        per_class_iou[cid] = float(tp[idx] / denom_iou) if denom_iou > 0 else 0.0

    macro_f1 = float(np.mean([per_class_f1[c] for c in class_ids])) if class_ids else 0.0
    macro_iou = float(np.mean([per_class_iou[c] for c in class_ids])) if class_ids else 0.0
    n_well = sum(1 for c in class_ids if per_class_f1[c] >= f1_well_resolved)

    _log.info(
        "eval_per_class_v2_done",
        n_categories=n_categories,
        n_eval=n_eval,
        macro_f1=round(macro_f1, 4),
        macro_iou=round(macro_iou, 4),
        n_classes_well_resolved=n_well,
    )
    return FaithfulRunResult(
        supervision="region_category",
        n_categories=n_categories,
        class_ids=list(class_ids),
        per_class_f1=per_class_f1,
        per_class_iou=per_class_iou,
        macro_f1=macro_f1,
        macro_iou=macro_iou,
        n_eval=n_eval,
        n_classes_well_resolved=n_well,
        best_ckpt=Path(),  # filled by the caller
        mean_regions_per_patch=0.0,  # filled by the caller
    )


@torch.no_grad()
def eval_per_parcel(
    student: torch.nn.Module,
    val_dataset: object,
    category_prototypes: torch.Tensor,
    class_ids: list[int],
    *,
    device: torch.device | None = None,
    batch_size: int = 256,
    f1_well_resolved: float = _F1_WELL_RESOLVED,
) -> FaithfulRunResult:
    """Evaluate the student PER PARCEL (polygon-to-polygon), not per patch.

    Twin of :func:`eval_per_class_v2` but at the parcel grain (US-036-b): each
    item of ``val_dataset`` (a :class:`ml.farslip.parcel_crop_dataset.ParcelCropDataset`)
    is a SINGLE parcel crop with its OWN true ``class_id``. The student CLS of the
    crop is matched by cosine argmax against the 768-dim category prototype bank,
    and compared to the parcel's real class (NOT the patch majority). This makes
    rare parcels recoverable: a Soybean parcel sharing a patch with Meadow is no
    longer forced to the patch label.

    Args:
        student: the trained CLIP vision student.
        val_dataset: held-out ``ParcelCropDataset`` (one item per parcel).
        category_prototypes: ``(C, 768)`` reprojected category bank.
        class_ids: PASTIS class ids in the SAME row order as the bank.
        device: torch device (defaults to the student's device).
        batch_size: evaluation batch size in parcels.
        f1_well_resolved: F1 threshold for "well-resolved".

    Returns:
        A :class:`FaithfulRunResult` with per-class and macro metrics at the
        parcel grain.
    """
    n_categories = len(class_ids)
    if category_prototypes.shape[0] != n_categories:
        raise ValueError(
            f"category_prototypes rows ({category_prototypes.shape[0]}) must equal "
            f"len(class_ids) ({n_categories})."
        )
    if device is None:
        device = next(student.parameters()).device
    student_was_training = student.training
    student.eval()

    protos_n = torch.nn.functional.normalize(
        category_prototypes.to(device=device, dtype=torch.float32), p=2, dim=-1
    )
    pastis_to_idx = {cid: idx for idx, cid in enumerate(class_ids)}

    tp = np.zeros(n_categories, dtype=np.int64)
    fp = np.zeros(n_categories, dtype=np.int64)
    fn = np.zeros(n_categories, dtype=np.int64)
    n_eval = 0

    images_buf: list[torch.Tensor] = []
    targets_buf: list[int] = []

    def _flush() -> None:
        nonlocal n_eval
        if not images_buf:
            return
        batch = torch.stack(images_buf, dim=0).to(device=device, dtype=torch.float32)
        out = student(pixel_values=batch, output_hidden_states=False)
        cls = out.last_hidden_state[:, 0, :]
        cls_n = torch.nn.functional.normalize(cls.float(), p=2, dim=-1)
        preds = (cls_n @ protos_n.t()).argmax(dim=-1).cpu().numpy()
        for pred, true in zip(preds, targets_buf, strict=True):
            n_eval += 1
            if pred == true:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
        images_buf.clear()
        targets_buf.clear()

    for idx in range(len(val_dataset)):  # type: ignore[arg-type]
        item = val_dataset[idx]  # type: ignore[index]
        true_cat = int(item["class_id"])
        if true_cat not in pastis_to_idx:
            continue
        images_buf.append(item["image"])
        targets_buf.append(pastis_to_idx[true_cat])
        if len(images_buf) >= batch_size:
            _flush()
    _flush()

    if student_was_training:
        student.train()

    per_class_f1: dict[int, float] = {}
    per_class_iou: dict[int, float] = {}
    for idx, cid in enumerate(class_ids):
        denom_f1 = 2 * tp[idx] + fp[idx] + fn[idx]
        denom_iou = tp[idx] + fp[idx] + fn[idx]
        per_class_f1[cid] = float(2 * tp[idx] / denom_f1) if denom_f1 > 0 else 0.0
        per_class_iou[cid] = float(tp[idx] / denom_iou) if denom_iou > 0 else 0.0

    macro_f1 = float(np.mean([per_class_f1[c] for c in class_ids])) if class_ids else 0.0
    macro_iou = float(np.mean([per_class_iou[c] for c in class_ids])) if class_ids else 0.0
    n_well = sum(1 for c in class_ids if per_class_f1[c] >= f1_well_resolved)

    _log.info(
        "eval_per_parcel_done",
        n_categories=n_categories,
        n_eval=n_eval,
        macro_f1=round(macro_f1, 4),
        macro_iou=round(macro_iou, 4),
        n_classes_well_resolved=n_well,
    )
    return FaithfulRunResult(
        supervision="region_category",
        n_categories=n_categories,
        class_ids=list(class_ids),
        per_class_f1=per_class_f1,
        per_class_iou=per_class_iou,
        macro_f1=macro_f1,
        macro_iou=macro_iou,
        n_eval=n_eval,
        n_classes_well_resolved=n_well,
        best_ckpt=Path(),
        mean_regions_per_patch=0.0,
    )


def _v1_vs_v2_table_rows(
    result: FaithfulRunResult,
    v1_per_class_f1: dict[int, float] | None,
) -> list[dict[str, Any]]:
    """Build the per-class ``v1 vs v2`` comparison rows (Polars-ready).

    Reports v2 F1/IoU and, when a v1 baseline is supplied, the per-class delta.
    Honest reporting (R-NO-IMPROVE): the table is emitted even when v2 < v1.

    Args:
        result: the v2 evaluation result.
        v1_per_class_f1: optional ``{class_id: f1}`` from the v1 run.

    Returns:
        One dict per active class with ``class_id``/``f1_v2``/``iou_v2``/
        ``f1_v1``/``delta_f1``/``well_resolved_v2``.
    """
    rows: list[dict[str, Any]] = []
    for cid in result.class_ids:
        f1_v2 = result.per_class_f1.get(cid, 0.0)
        f1_v1 = float(v1_per_class_f1.get(cid, 0.0)) if v1_per_class_f1 is not None else None
        rows.append(
            {
                "class_id": int(cid),
                "f1_v2": float(f1_v2),
                "iou_v2": float(result.per_class_iou.get(cid, 0.0)),
                "f1_v1": f1_v1,
                "delta_f1": (float(f1_v2 - f1_v1) if f1_v1 is not None else None),
                "well_resolved_v2": bool(f1_v2 >= _F1_WELL_RESOLVED),
            }
        )
    return rows


def _log_faithful_run(
    *,
    mlflow_uri: str,
    run_name: str,
    result: FaithfulRunResult,
    supervision: str,
    lambda_loc: float,
    temperature: float,
    use_global_caption_loss: bool,
    folds: tuple[int, ...],
    val_folds: tuple[int, ...],
    pastis_root: Path,
    captions_path: Path,
    caption_model: str,
    prompt_version: str,
    train_metrics: dict[str, float],
    v1_per_class_f1: dict[int, float] | None,
) -> None:
    """Log ONE fully-closed MLflow run with the per-class eval and lineage tags.

    The trainer owns its own ``start_run``/``end_run`` for the loss curve; this
    function logs a SEPARATE, FINISHED run carrying the per-class evaluation
    (``f1_class_<id>`` / ``iou_class_<id>``), the ``v1 vs v2`` table artifact and
    the faithful-v2 params. ``data_version`` (PASTIS-R + captions parquet) +
    ``code_version`` (git SHA) are tagged. If MLflow is down it degrades to a
    warning (the run still lives in the logs), never raising (R-MLFLOW-RUNNING:
    the run is opened in a ``with`` so it always closes).

    Args:
        mlflow_uri: tracking URI (Docker :5010 or a SQLite file for CI).
        run_name: MLflow run name.
        result: the v2 evaluation result.
        supervision: trainer supervision mode actually used.
        lambda_loc: ``L_loc`` weight.
        temperature: contrastive temperature.
        use_global_caption_loss: whether ``L_glo`` was active.
        folds: train folds.
        val_folds: held-out eval folds.
        pastis_root: PASTIS-R root (drives part of ``data_version``).
        captions_path: captions parquet (drives part of ``data_version``).
        caption_model: Ollama caption model tag.
        prompt_version: caption prompt template version.
        train_metrics: final loss dict of the run.
        v1_per_class_f1: optional v1 baseline for the delta table.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow optional
        _log.warning("mlflow not installed; faithful-v2 run not logged", run=run_name)
        return

    import tempfile

    import polars as pl

    data_version = f"{dvc_data_version(str(pastis_root))}|{dvc_data_version(str(captions_path))}"
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("farslip")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "code_version": git_sha(),
                    "data_version": data_version,
                    "us": "US-036-a-v2",
                    "supervision": supervision,
                    "proto_source": "pastis_direct",
                }
            )
            mlflow.log_params(
                {
                    "supervision": supervision,
                    "lambda_loc": lambda_loc,
                    "temperature": temperature,
                    "use_global_caption_loss": use_global_caption_loss,
                    "n_categories": result.n_categories,
                    "n_in_channels": _N_IN_CHANNELS,
                    "caption_model": caption_model,
                    "prompt_version": prompt_version,
                    "class_ids": ",".join(str(c) for c in result.class_ids),
                    "folds": ",".join(str(f) for f in folds),
                    "val_folds": ",".join(str(f) for f in val_folds),
                    "dataset": "pastis_r_real",
                    "mean_regions_per_patch": round(result.mean_regions_per_patch, 4),
                }
            )
            metric_payload: dict[str, float] = {
                "macro_f1": result.macro_f1,
                "macro_iou": result.macro_iou,
                "n_classes_well_resolved": float(result.n_classes_well_resolved),
                "n_eval": float(result.n_eval),
                "mean_regions_per_patch": float(result.mean_regions_per_patch),
            }
            for cid in result.class_ids:
                metric_payload[f"f1_class_{cid}"] = result.per_class_f1.get(cid, 0.0)
                metric_payload[f"iou_class_{cid}"] = result.per_class_iou.get(cid, 0.0)
            for key in ("loss_total", "loss_glo", "loss_loc"):
                if key in train_metrics:
                    metric_payload[key] = float(train_metrics[key])
            mlflow.log_metrics(metric_payload)

            table = pl.DataFrame(_v1_vs_v2_table_rows(result, v1_per_class_f1))
            with tempfile.TemporaryDirectory() as tmp:
                art = Path(tmp) / "per_class_v1_vs_v2.parquet"
                table.write_parquet(art)
                mlflow.log_artifact(str(art))
        _log.info("faithful-v2 mlflow run logged and closed", run=run_name)
    except Exception as exc:  # noqa: BLE001 - never let logging kill the run
        _log.warning("mlflow faithful-v2 run failed", run=run_name, error=str(exc))


def run_faithful_v2(
    *,
    pastis_root: Path,
    captions_path: Path,
    output_dir: Path = Path("checkpoints/farslip/faithful_v2"),
    run_name: str = "farslip-faithful-v2",
    supervision: SupervisionChoice = "faithful_v2",
    lambda_loc: float = 1.0,
    temperature: float = 0.07,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-5,
    seed: int = 42,
    folds: tuple[int, ...] = (1, 2, 3),
    val_folds: tuple[int, ...] = (4,),
    active_class_ids: tuple[int, ...] = _ALL_ACTIVE_CLASS_IDS,
    min_area_px: int = 16,
    dominance_ratio: float | None = None,
    time_cap_hours: float = 8.0,
    use_global_caption_loss: bool = True,
    use_class_weights: bool = False,
    prototype_path: Path | None = None,
    caption_model: str = "gemma4:31b-it-q8_0",
    prompt_version: str = "v2",
    mlflow_uri: str = _DEFAULT_MLFLOW_URI,
    v1_per_class_f1: dict[int, float] | None = None,
) -> FaithfulRunResult:
    """Run the faithful FarSLIP v2 re-training over real French PASTIS-R.

    Builds the region-category train dataset (T2) from the cached captions (T1),
    instantiates the trainer in ``supervision="region_category"`` with the active
    category prototype bank (US-033, read-only), trains
    ``L_total = L_glo + lambda_loc * L_loc``, evaluates per class on the disjoint
    held-out fold, logs one closed MLflow run with the ``v1 vs v2`` table, and
    persists the best checkpoint (input to US-037). ``--supervision dominant_v1``
    runs the v1 path through the same flag as an ablation.

    Args:
        pastis_root: PASTIS-R root (rejected if it is the Italian/synthetic root).
        captions_path: cached captions parquet (must cover both splits).
        output_dir: checkpoints dir (relative -> lands on F: on the VM).
        run_name: MLflow run name.
        supervision: ``"faithful_v2"`` (MPCL + L_glo) or ``"dominant_v1"``.
        lambda_loc: ``L_loc`` weight in the combination (paper Table 3, 1.0).
        temperature: contrastive temperature ``tau`` (paper Section 3.3, 0.07).
        n_epochs: training epochs.
        batch_size: DataLoader batch size (in patches).
        lr: AdamW learning rate.
        seed: determinism seed.
        folds: train folds (spatial CV).
        val_folds: held-out eval folds (disjoint from ``folds``).
        active_class_ids: active PASTIS categories (default all 18).
        min_area_px: minimum region area (slivers below are dropped).
        dominance_ratio: optional 3:1 Meadow filter (default None: multi-object).
        time_cap_hours: hard cap forwarded to the trainer.
        use_global_caption_loss: toggle the ``L_glo`` InfoNCE.
        prototype_path: override of the US-033 parquet (read/filter only).
        caption_model: Ollama caption model tag (logged).
        prompt_version: caption prompt template version (logged).
        mlflow_uri: MLflow tracking URI.
        v1_per_class_f1: optional v1 baseline for the honest delta table.

    Returns:
        The :class:`FaithfulRunResult` (its ``best_ckpt`` feeds US-037).

    Raises:
        ValueError: if ``pastis_root`` is the Italian/synthetic root, if
            ``val_folds`` overlaps ``folds`` (leakage), if a split patch lacks a
            caption, or if the active categories are absent from the US-033 bank.
    """
    _validate_pastis_root(pastis_root)
    assert_disjoint_folds(folds, val_folds)
    trainer_supervision = _SUPERVISION_MAP[supervision]
    propagate_seed(seed)

    _log.info(
        "faithful-v2 run start",
        run_name=run_name,
        supervision=supervision,
        trainer_supervision=trainer_supervision,
        lambda_loc=lambda_loc,
        temperature=temperature,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        folds=list(folds),
        val_folds=list(val_folds),
        active_class_ids=list(active_class_ids),
        dominance_ratio=dominance_ratio,
        use_global_caption_loss=use_global_caption_loss,
        pastis_root=str(pastis_root),
        captions_path=str(captions_path),
        output_dir=str(output_dir),
        device="cuda" if torch.cuda.is_available() else "cpu",
        mlflow_uri=mlflow_uri,
    )

    captions = load_captions(captions_path)
    if not captions:
        raise ValueError(
            f"no captions loaded from {captions_path}; run Phase A "
            "(generate-captions) and dvc pull the parquet before training."
        )

    # Train + held-out region-category datasets (multi-object, spatial CV).
    train_ds = RegionCategoryPairDataset(
        captions,
        root=pastis_root,
        folds=folds,
        active_class_ids=active_class_ids,
        min_area_px=min_area_px,
        dominance_ratio=dominance_ratio,
        seed=seed,
    )
    _require_captions_for_dataset(train_ds, captions, "train")
    val_ds = RegionCategoryPairDataset(
        captions,
        root=pastis_root,
        folds=val_folds,
        active_class_ids=active_class_ids,
        min_area_px=min_area_px,
        dominance_ratio=dominance_ratio,
        seed=seed,
    )
    _require_captions_for_dataset(val_ds, captions, "val")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FarSLIPTrainerConfig(
        dataset_root=pastis_root,  # drives MLflow data_version -> PASTIS-R
        output_dir=output_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        time_cap_hours=time_cap_hours,
        n_in_channels=_N_IN_CHANNELS,
        n_categories=len(active_class_ids),
        supervision=trainer_supervision,  # type: ignore[arg-type]
        lambda_loc=lambda_loc,
        temperature=temperature,
        use_global_caption_loss=use_global_caption_loss,
        mlflow_run_name=run_name,
        extra_params={
            "us": "US-036-a-v2",
            "caption_model": caption_model,
            "prompt_version": prompt_version,
            "dataset": "pastis_r_real",
            "mean_regions_per_patch": round(train_ds.mean_regions_per_patch, 4),
        },
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=train_ds)

    # Active category prototype bank (US-033, read-only) + PASTIS id -> [0, C) map.
    bank, class_ids = _category_prototypes(prototype_path, active_class_ids)
    trainer.set_category_prototypes(bank, class_ids)

    # Optional class re-weighting (inverse frequency) to fight PASTIS imbalance:
    # rare-class region anchors weigh more in L_loc so the model stops collapsing
    # to the dominant categories (Meadow). Off by default (faithful baseline).
    if use_class_weights:
        weights = _inverse_frequency_weights(train_ds, class_ids)
        trainer.set_class_weights(weights)

    # Pre-encode captions ONCE (MiniLM-384, US-033 encoder) so the batch carries
    # ``caption_cls`` and ``L_glo`` (image-text InfoNCE, eq. 1-2) is active. Without
    # this the batch has no ``caption_cls`` and the trainer only runs MPCL (L_loc).
    if use_global_caption_loss:
        caption_embeddings = encode_captions_minilm(captions, device=trainer.device.type)
        collate_fn = make_caption_collate(collate_region_batch, caption_embeddings)
    else:
        collate_fn = collate_region_batch

    # The faithful-v2 batch is the cross-patch collate; train with it.
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=trainer.device.type == "cuda",
        collate_fn=collate_fn,
    )
    start = time.monotonic()
    train_metrics = trainer.train(loader)

    # Persist the best (last epoch) as the canonical checkpoint (feeds US-037).
    best_ckpt = output_dir / "best.safetensors"
    last_epoch_ckpt = output_dir / f"student_epoch_{n_epochs - 1}.safetensors"
    if last_epoch_ckpt.exists():
        best_ckpt.write_bytes(last_epoch_ckpt.read_bytes())
    else:  # time cap before the last epoch -> explicit save
        best_ckpt = Path(trainer.save_student(format="safetensors", suffix="best"))

    # Per-class eval on the held-out fold (reuse the trainer's 768-dim bank).
    proto_eval = trainer._category_prototypes
    if proto_eval is None:  # pragma: no cover - set above, defensive
        raise RuntimeError("trainer category prototypes missing after set")
    result = eval_per_class_v2(
        trainer.student,
        val_ds,
        proto_eval,
        class_ids,
        device=trainer.device,
        batch_size=batch_size,
    )
    result.supervision = trainer_supervision
    result.best_ckpt = best_ckpt
    result.mean_regions_per_patch = train_ds.mean_regions_per_patch
    result.train_metrics = train_metrics

    _log_faithful_run(
        mlflow_uri=mlflow_uri,
        run_name=run_name,
        result=result,
        supervision=trainer_supervision,
        lambda_loc=lambda_loc,
        temperature=temperature,
        use_global_caption_loss=use_global_caption_loss,
        folds=folds,
        val_folds=val_folds,
        pastis_root=pastis_root,
        captions_path=captions_path,
        caption_model=caption_model,
        prompt_version=prompt_version,
        train_metrics=train_metrics,
        v1_per_class_f1=v1_per_class_f1,
    )

    _log.info(
        "faithful-v2 run done",
        run_name=run_name,
        supervision=trainer_supervision,
        best_ckpt=str(best_ckpt),
        macro_f1=round(result.macro_f1, 4),
        macro_iou=round(result.macro_iou, 4),
        n_classes_well_resolved=result.n_classes_well_resolved,
        mean_regions_per_patch=round(result.mean_regions_per_patch, 4),
        elapsed_hours=round((time.monotonic() - start) / 3600.0, 4),
        dvc_add_hint=f"dvc add {best_ckpt} && dvc push",
    )
    return result


def _parse_folds(folds: str) -> tuple[int, ...]:
    """Parses a comma-separated fold string into a tuple of ints.

    Args:
        folds: comma-separated PASTIS fold ids (e.g. ``"1,2,3"``).

    Returns:
        Tuple of fold ids in declaration order.

    Raises:
        typer.BadParameter: if any token is not a valid integer or the result
            is empty.
    """
    try:
        parsed = tuple(int(tok) for tok in folds.split(",") if tok.strip() != "")
    except ValueError as exc:
        raise typer.BadParameter(f"folds must be comma-separated ints: {folds!r}") from exc
    if not parsed:
        raise typer.BadParameter(f"folds is empty: {folds!r}")
    return parsed


def _parse_class_ids(class_ids: str) -> tuple[int, ...]:
    """Parses a comma-separated PASTIS class-id string into a tuple of ints.

    Args:
        class_ids: comma-separated PASTIS class ids (e.g. ``"1,2,3"``).

    Returns:
        Tuple of class ids in declaration order.

    Raises:
        typer.BadParameter: if any token is not a valid integer or the result
            is empty / out of the 1..18 range.
    """
    try:
        parsed = tuple(int(tok) for tok in class_ids.split(",") if tok.strip() != "")
    except ValueError as exc:
        raise typer.BadParameter(f"class_ids must be comma-separated ints: {class_ids!r}") from exc
    if not parsed:
        raise typer.BadParameter(f"class_ids is empty: {class_ids!r}")
    bad = [c for c in parsed if not 1 <= c <= 18]
    if bad:
        raise typer.BadParameter(f"class_ids out of [1, 18]: {bad}")
    return parsed


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _main() -> None:
    """FarSLIP fiel al paper (US-036-a v2). Subcomando: ``train``.

    El callback fuerza a Typer a exponer ``train`` como subcomando explicito
    (sin el, un Typer de comando unico ignora el nombre e interpreta ``train``
    como argumento extra).
    """


@app.command()
def train(
    run_name: Annotated[str, typer.Option(help="Nombre del run MLflow")] = "farslip-faithful-v2",
    supervision: Annotated[
        str,
        typer.Option(help="faithful_v2 (MPCL + L_glo, default) o dominant_v1 (ablacion)"),
    ] = "faithful_v2",
    lambda_loc: Annotated[
        float, typer.Option(help="Peso de L_loc en L_total = L_glo + lambda_loc*L_loc")
    ] = 1.0,
    temperature: Annotated[
        float, typer.Option(help="Temperatura tau contrastiva (paper 0.07)")
    ] = 0.07,
    n_epochs: Annotated[int, typer.Option(help="Epochs de entrenamiento")] = 30,
    batch_size: Annotated[int, typer.Option(help="Batch size (en patches)")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    folds: Annotated[str, typer.Option(help="Folds de train PASTIS, coma-separados")] = "1,2,3",
    val_folds: Annotated[str, typer.Option(help="Folds de validacion (disjuntos de train)")] = "4",
    active_class_ids: Annotated[
        str, typer.Option(help="Clases PASTIS activas, coma-separadas (1..18)")
    ] = ",".join(str(c) for c in _ALL_ACTIVE_CLASS_IDS),
    min_area_px: Annotated[int, typer.Option(help="Area minima de region (px)")] = 16,
    dominance_ratio: Annotated[
        float | None,
        typer.Option(help="Filtro 3:1 Meadow (None = multi-objeto, default)"),
    ] = None,
    no_global_loss: Annotated[
        bool, typer.Option("--no-global-loss", help="Desactiva L_glo (ablacion)")
    ] = False,
    use_class_weights: Annotated[
        bool,
        typer.Option(
            "--use-class-weights",
            help="Pondera L_loc (MPCL) por frecuencia inversa de clase (anti-desbalance)",
        ),
    ] = False,
    pastis_root: Annotated[Path, typer.Option(help="Raiz PASTIS-R (frances real)")] = Path(
        "data/PASTIS-R"
    ),
    captions_path: Annotated[
        Path, typer.Option(help="Parquet de captions cacheadas (Fase A)")
    ] = Path("data/farslip/pastis_captions.parquet"),
    output_dir: Annotated[Path, typer.Option(help="Dir checkpoints (cae en F: en la VM)")] = Path(
        "checkpoints/farslip/faithful_v2"
    ),
    time_cap_hours: Annotated[float, typer.Option(help="Hard cap horas")] = 8.0,
    prototype_path: Annotated[
        Path | None,
        typer.Option(help="Override del parquet US-033 (solo LEER/FILTRAR)"),
    ] = None,
    caption_model: Annotated[
        str, typer.Option(help="Tag del modelo Ollama de las captions")
    ] = "gemma4:31b-it-q8_0",
    prompt_version: Annotated[str, typer.Option(help="Version del prompt de captions")] = "v2",
    mlflow_uri: Annotated[
        str,
        typer.Option(help="MLflow tracking URI (Docker :5010; SQLite file:// CI)"),
    ] = _DEFAULT_MLFLOW_URI,
) -> None:
    """Entrena FarSLIP FIEL al paper (MPCL + L_glo) sobre PASTIS-R real.

    Construye el dataset region-category multi-objeto (T2) desde las captions
    cacheadas (T1), instancia ``FarSLIPDistillationTrainer`` en
    ``supervision="region_category"`` con el banco de prototipos de categoria
    (US-033, solo lectura), entrena ``L_total = L_glo + lambda_loc * L_loc``,
    evalua F1/IoU por clase sobre ``val_folds`` (held-out, sin leakage), registra
    un run MLflow `:5010` CERRADO con la tabla honesta ``v1 vs v2`` y persiste el
    best (insumo de US-037). ``--supervision dominant_v1`` corre el path v1.

    Args:
        run_name: nombre del run MLflow.
        supervision: ``faithful_v2`` (MPCL + L_glo) o ``dominant_v1`` (ablacion).
        lambda_loc: peso de L_loc en la combinacion.
        temperature: temperatura contrastiva tau.
        n_epochs: epochs de entrenamiento.
        batch_size: batch size en patches.
        lr: learning rate AdamW.
        seed: semilla de determinismo.
        folds: folds de train (spatial CV), coma-separados.
        val_folds: folds held-out (disjuntos de train), coma-separados.
        active_class_ids: clases PASTIS activas (1..18), coma-separadas.
        min_area_px: area minima de region.
        dominance_ratio: filtro 3:1 opcional (None = multi-objeto).
        no_global_loss: desactiva L_glo (ablacion).
        pastis_root: raiz PASTIS-R.
        captions_path: parquet de captions (Fase A).
        output_dir: dir de checkpoints.
        time_cap_hours: hard cap de horas.
        prototype_path: override del parquet US-033 (solo lectura).
        caption_model: tag del modelo Ollama (logueado).
        prompt_version: version del prompt (logueada).
        mlflow_uri: MLflow tracking URI.
    """
    if supervision not in _SUPERVISION_MAP:
        raise typer.BadParameter(
            f"supervision must be one of {sorted(_SUPERVISION_MAP)}; got {supervision!r}"
        )
    result = run_faithful_v2(
        pastis_root=pastis_root,
        captions_path=captions_path,
        output_dir=output_dir,
        run_name=run_name,
        supervision=supervision,  # type: ignore[arg-type]
        lambda_loc=lambda_loc,
        temperature=temperature,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        folds=_parse_folds(folds),
        val_folds=_parse_folds(val_folds),
        active_class_ids=_parse_class_ids(active_class_ids),
        min_area_px=min_area_px,
        dominance_ratio=dominance_ratio,
        time_cap_hours=time_cap_hours,
        use_global_caption_loss=not no_global_loss,
        use_class_weights=use_class_weights,
        prototype_path=prototype_path,
        caption_model=caption_model,
        prompt_version=prompt_version,
        mlflow_uri=mlflow_uri,
    )
    _log.info(
        "faithful-v2 winner (input to US-037)",
        supervision=result.supervision,
        best_ckpt=str(result.best_ckpt),
        macro_f1=round(result.macro_f1, 4),
        n_classes_well_resolved=result.n_classes_well_resolved,
        mean_regions_per_patch=round(result.mean_regions_per_patch, 4),
        dvc_add_hint=f"dvc add {result.best_ckpt} && dvc push",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
