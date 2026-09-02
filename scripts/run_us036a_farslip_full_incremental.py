"""Productive orchestrator of the full incremental FarSLIP fine-tune (US-036-a).

Runs the cardinality curriculum over the **real French PASTIS-R** dataset: start
from the 4 dominant classes (Meadow, Corn, Soft winter wheat, Grapevine) and add
``step_size`` classes per step up to 18, each step initialized from the previous
step's ``best`` checkpoint (``load_state_dict(strict=False)`` chained init; step 0
from the teacher CLIP default of the trainer), ``epochs_per_step`` epochs to
convergence, per-class F1/IoU eval on a held-out validation fold, one MLflow run
per step, and an explicit stop criterion. The best of the winning step is the
input to US-037 (downstream eval vs AlphaEarth 0.233 / FarSLIP-previo 0.163).

Reuse, NOT modification (write-set disjoint from US-036):

    - :func:`ml.farslip.pastis_pair_dataset.create_incremental_dataset`
      (US-036, IMPORTED): the only source of the real PASTIS-R peak-NDVI pairs,
      3:1 filter, curriculum, direct US-033 prototypes — NO ``expand_to_cap``,
      NO ``data/farslip_pairs``.
    - :class:`ml.farslip.distill.FarSLIPDistillationTrainer` (INSTANTIATED): the
      AdamW loop, ``set_text_prototypes`` (384->768 frozen lift), ``save_student``
      (safetensors), and its own MLflow ``start_run``/``end_run`` lifecycle with
      ``data_version`` + ``code_version``.
    - :mod:`ml.farslip.incremental_curriculum` (logic): ranking, step class_ids,
      prototype selection, stop criterion.
    - ``scripts.train_incremental._load_student_state_dict`` / ``_parse_folds``
      (US-036 helpers, IMPORTED, not reimplemented).

Scope (critical, ordered by the user 2026-06-07): ONLY real French PASTIS-R. No
Italian / synthetic / placeholder data, no ``expand_to_cap`` / CAP bridge, no
``data/farslip_pairs``. ``n_regions`` is always 1; the prototypes are PASTIS
direct (one row per active class) filtered from the US-033 parquet (NEVER
regenerated, Gemini is NEVER called). Pointing ``pastis_root`` at
``data/farslip_pairs`` is rejected with a clear ``ValueError``.

Per-class eval (AC-7): the resulting student is evaluated on a held-out PASTIS
fold by **pair classification** — the student's CLS token (768-dim) is compared
by cosine similarity against the N reprojected text prototypes (the same 768-dim
bank the contrastive loss consumes, read from ``trainer._text_prototypes`` after
``set_text_prototypes``); ``argmax`` is the predicted active index, compared to
the true ``category_id``. This is patch-level classification (no dense masks).
From the confusion it derives per-class F1 and IoU (the macro/per-class metric
the rubric asks for). It is deterministic (no shuffle, eval mode, ``no_grad``).

Project convention: ``structlog`` (no ``print``); type hints everywhere;
docstrings in English, prose in Spanish; no emojis; checkpoints under the
relative path ``checkpoints/farslip/incremental/<NN>cls/`` (lands on ``F:`` on
the VM); MLflow on the Docker server ``:5010`` with ``data_version`` +
``code_version``; official PASTIS folds (spatial CV; ``val_folds`` disjoint from
the train folds, no leakage).

Typical productive usage on the H100 (run nvidia-smi first)::

    poetry run python -m scripts.run_us036a_farslip_full_incremental run \\
        --run-name farslip-full-incr \\
        --step-size 2 --base-classes 4 --max-classes 18 \\
        --epochs-per-step 20 --batch-size 64 --lr 1e-5 --seed 42 \\
        --folds 1,2,3 --val-folds 4 --ratio 3.0 \\
        --pastis-root data/PASTIS-R \\
        --output-dir checkpoints/farslip/incremental \\
        --time-cap-hours 8.0 --mlflow-uri http://localhost:5010
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the run CLI. poetry add typer") from exc

from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.farslip.incremental_curriculum import (
    StepMetrics,
    cardinality_ranking,
    class_ids_for_step,
    n_steps,
    select_step_prototypes,
    stop_criterion,
)
from ml.farslip.pastis_pair_dataset import create_incremental_dataset
from ml.features.phenology_class_prototypes import load_class_prototype_embeddings
from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.seed import propagate_seed
from scripts.train_incremental import _load_student_state_dict, _parse_folds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import Dataset

_log = structlog.get_logger(__name__)

#: Default MLflow tracking server (Docker on :5010); the lineage lives here, NOT
#: in ``./mlruns``. Overridable with ``--mlflow-uri`` (e.g. a local SQLite file
#: for CI/dry-runs).
_DEFAULT_MLFLOW_URI = "http://localhost:5010"

#: Number of Sentinel-2 input channels of the composite (B02, B03, B04, B08).
_N_IN_CHANNELS = 4

#: Forbidden Italian/synthetic data root (US-034/035 path, discarded here).
_FORBIDDEN_ROOT_NAME = "farslip_pairs"


@dataclass
class StepResult:
    """Outcome of one curriculum step.

    Attributes:
        n_classes: active classes at this step.
        class_ids: active PASTIS class_ids (curriculum order).
        metrics: the per-class evaluation summary.
        best_ckpt: path of the step's best student checkpoint.
        init_from: provenance of the init (``"teacher_clip"`` or the previous
            best checkpoint path).
        loss_cls_final: final ``loss_cls`` of the step (last epoch).
        loss_total_final: final ``loss_total`` of the step (last epoch).
        missing_keys: ``missing_keys`` of the chained ``load_state_dict``.
        unexpected_keys: ``unexpected_keys`` of the chained ``load_state_dict``.
        stop_reason: the reason the curriculum stopped after this step
            (``continue`` if it did not stop here).
    """

    n_classes: int
    class_ids: list[int]
    metrics: StepMetrics
    best_ckpt: Path
    init_from: str
    loss_cls_final: float = 0.0
    loss_total_final: float = 0.0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    stop_reason: str = "continue"


def _validate_pastis_root(pastis_root: Path) -> None:
    """Reject Italian/synthetic data; require a real PASTIS-R root.

    The US-034/035 Italian crops lived under ``data/farslip_pairs``. This US is
    PASTIS-R-only: pointing the run at that directory (or any path whose name
    contains it) is a scope violation and fails fast.

    Args:
        pastis_root: the candidate PASTIS-R root.

    Raises:
        ValueError: if the path is the forbidden Italian/synthetic root.
    """
    parts = {p.lower() for p in pastis_root.parts}
    if _FORBIDDEN_ROOT_NAME in pastis_root.name.lower() or _FORBIDDEN_ROOT_NAME in parts:
        raise ValueError(
            f"pastis_root {pastis_root!s} points at the Italian/synthetic "
            f"'{_FORBIDDEN_ROOT_NAME}' data (US-034/035, discarded). US-036-a is "
            "PASTIS-R-only: pass a real PASTIS-R root (e.g. data/PASTIS-R)."
        )


@torch.no_grad()
def eval_per_class(
    student: torch.nn.Module,
    val_dataset: Dataset,
    class_ids: list[int],
    prototypes: torch.Tensor,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    f1_well_resolved: float = 0.50,
) -> StepMetrics:
    """Evaluate the student per class by CLS<->prototype cosine classification.

    For each validation pair, the student produces a CLS token (768-dim); it is
    compared by cosine similarity against the N reprojected text prototypes
    (768-dim, the bank the contrastive loss consumes), and ``argmax`` is the
    predicted active index. The prediction is compared to the true
    ``category_id`` (index of the patch's dominant class inside the step's active
    set). From the resulting confusion the metric derives, per active class, the
    standard one-vs-rest F1 and IoU::

        F1  = 2 * TP / (2 * TP + FP + FN)
        IoU = TP / (TP + FP + FN)

    where TP/FP/FN are pixel-free, per-class counts over the validation pairs.
    This is patch-level classification (no dense masks). It is deterministic: the
    student runs in ``eval`` mode under ``no_grad`` over the dataset in order (no
    shuffle), so the same checkpoint and dataset always yield the same metrics.

    Args:
        student: the trained CLIP vision student (returns ``last_hidden_state``
            with CLS at position 0).
        val_dataset: held-out PASTIS pair dataset (``val_folds``); each item is
            ``{"image": (4, H, W), "region_id": 0, "category_id": idx}``.
        class_ids: active PASTIS class_ids in curriculum order; row ``r`` of
            ``prototypes`` is the prototype of ``class_ids[r]``.
        prototypes: ``(N, 768)`` reprojected text prototypes (the trainer's
            ``_text_prototypes`` after ``set_text_prototypes``).
        device: torch device (defaults to the student's device).
        batch_size: evaluation batch size.
        f1_well_resolved: F1 threshold for "well-resolved" (stored on the result).

    Returns:
        A :class:`StepMetrics` with per-class F1/IoU, macro-F1/IoU and ``n_eval``.

    Raises:
        ValueError: if ``prototypes`` rows do not match ``len(class_ids)``.
    """
    from torch.utils.data import DataLoader

    n_classes = len(class_ids)
    if prototypes.shape[0] != n_classes:
        raise ValueError(
            f"prototypes rows ({prototypes.shape[0]}) must equal len(class_ids) ({n_classes})."
        )
    if device is None:
        device = next(student.parameters()).device
    student_was_training = student.training
    student.eval()

    protos_n = torch.nn.functional.normalize(
        prototypes.to(device=device, dtype=torch.float32), p=2, dim=-1
    )

    # Per-class one-vs-rest confusion counts (index space = active index 0..N-1).
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    n_eval = 0

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # deterministic eval
        num_workers=0,
    )
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        targets = batch["category_id"].long().view(-1).cpu().numpy()
        out = student(pixel_values=images, output_hidden_states=False)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        cls_n = torch.nn.functional.normalize(cls.float(), p=2, dim=-1)
        logits = cls_n @ protos_n.t()  # (B, N)
        preds = logits.argmax(dim=-1).cpu().numpy()
        for pred, true in zip(preds, targets, strict=True):
            n_eval += 1
            if pred == true:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1

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

    metrics = StepMetrics(
        n_classes=n_classes,
        class_ids=list(class_ids),
        per_class_f1=per_class_f1,
        per_class_iou=per_class_iou,
        macro_f1=macro_f1,
        macro_iou=macro_iou,
        n_eval=n_eval,
        f1_well_resolved=f1_well_resolved,
    )
    _log.info(
        "eval_per_class_done",
        n_classes=n_classes,
        n_eval=n_eval,
        macro_f1=round(macro_f1, 4),
        macro_iou=round(macro_iou, 4),
        n_classes_well_resolved=metrics.n_classes_well_resolved,
    )
    return metrics


def _metrics_table_rows(metrics: StepMetrics) -> list[dict[str, Any]]:
    """Build per-class metric rows (Polars-ready) for the MLflow artifact.

    Args:
        metrics: the step's per-class metrics.

    Returns:
        One dict per active class with ``class_id``/``f1``/``iou``/
        ``well_resolved``.
    """
    rows: list[dict[str, Any]] = []
    for cid in metrics.class_ids:
        f1 = metrics.per_class_f1.get(cid, 0.0)
        rows.append(
            {
                "class_id": int(cid),
                "f1": float(f1),
                "iou": float(metrics.per_class_iou.get(cid, 0.0)),
                "well_resolved": bool(f1 >= metrics.f1_well_resolved),
            }
        )
    return rows


def _log_step_run(
    *,
    mlflow_uri: str,
    run_name: str,
    n_classes: int,
    class_ids: list[int],
    metrics: StepMetrics,
    init_from: str,
    epochs: int,
    step_size: int,
    dominance_ratio: float,
    folds: tuple[int, ...],
    val_folds: tuple[int, ...],
    pastis_root: Path,
    train_metrics: dict[str, float],
    stop_reason: str,
) -> None:
    """Log one MLflow run per step with per-class metrics and lineage tags.

    The trainer owns its own ``start_run``/``end_run`` for the loss curve; this
    function logs a SEPARATE, fully-closed run that carries the per-class
    evaluation (``f1_class_<id>`` / ``iou_class_<id>``), ``macro_f1``,
    ``n_classes_well_resolved`` and the curriculum params, plus the per-class
    table as a Polars-written artifact. ``data_version`` (PASTIS-R) +
    ``code_version`` (git SHA) are tagged so the run is auditable. If MLflow is
    not installed or the server is down, the function degrades to a warning (the
    run still lives in the logs).

    Args:
        mlflow_uri: tracking URI (Docker :5010 or a SQLite file for CI).
        run_name: per-step run name (``farslip-full-incr-<NN>cls``).
        n_classes: active classes at this step.
        class_ids: active PASTIS class_ids.
        metrics: per-class evaluation.
        init_from: init provenance (teacher or previous best).
        epochs: epochs trained this step.
        step_size: curriculum step size.
        dominance_ratio: 3:1 filter ratio.
        folds: train folds.
        val_folds: held-out eval folds.
        pastis_root: PASTIS-R root (drives ``data_version``).
        train_metrics: final loss dict of the step.
        stop_reason: the stop reason after this step.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow optional
        _log.warning("mlflow not installed; per-class run not logged", run=run_name)
        return

    import tempfile

    import polars as pl

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("farslip")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "code_version": git_sha(),
                    "data_version": dvc_data_version(str(pastis_root)),
                    "us": "US-036-a",
                    "proto_source": "pastis_direct",
                    "stop_reason": stop_reason,
                }
            )
            mlflow.log_params(
                {
                    "n_classes": n_classes,
                    "class_ids": ",".join(str(c) for c in class_ids),
                    "n_regions": 1,
                    "proto_source": "pastis_direct",
                    "init_from": init_from,
                    "n_epochs": epochs,
                    "dominance_ratio": dominance_ratio,
                    "step_size": step_size,
                    "folds": ",".join(str(f) for f in folds),
                    "val_folds": ",".join(str(f) for f in val_folds),
                    "dataset": "pastis_r_real",
                }
            )
            metric_payload: dict[str, float] = {
                "macro_f1": metrics.macro_f1,
                "macro_iou": metrics.macro_iou,
                "n_classes_well_resolved": float(metrics.n_classes_well_resolved),
                "n_eval": float(metrics.n_eval),
            }
            for cid in class_ids:
                metric_payload[f"f1_class_{cid}"] = metrics.per_class_f1.get(cid, 0.0)
                metric_payload[f"iou_class_{cid}"] = metrics.per_class_iou.get(cid, 0.0)
            for key in ("loss_cls", "loss_total", "loss_patch", "loss_aux"):
                if key in train_metrics:
                    metric_payload[key] = float(train_metrics[key])
            mlflow.log_metrics(metric_payload)

            table = pl.DataFrame(_metrics_table_rows(metrics))
            with tempfile.TemporaryDirectory() as tmp:
                art = Path(tmp) / f"per_class_metrics_{n_classes:02d}cls.parquet"
                table.write_parquet(art)
                mlflow.log_artifact(str(art))
        _log.info("step mlflow run logged and closed", run=run_name)
    except Exception as exc:  # noqa: BLE001 - never let logging kill the run
        _log.warning("mlflow per-class run failed", run=run_name, error=str(exc))


def run_incremental_curriculum(
    *,
    pastis_root: Path,
    output_root: Path = Path("checkpoints/farslip/incremental"),
    run_name: str = "farslip-full-incr",
    step_size: int = 2,
    base_classes: int = 4,
    max_classes: int = 18,
    epochs_per_step: int = 20,
    batch_size: int = 64,
    lr: float = 1e-5,
    seed: int = 42,
    folds: tuple[int, ...] = (1, 2, 3),
    val_folds: tuple[int, ...] = (4,),
    dominance_ratio: float = 3.0,
    time_cap_hours: float = 8.0,
    prototype_path: Path | None = None,
    mlflow_uri: str = _DEFAULT_MLFLOW_URI,
) -> list[StepResult]:
    """Run the full incremental FarSLIP fine-tune over real French PASTIS-R.

    For each curriculum step k: build the PASTIS dataset (US-036 builder,
    n_classes=N_k), select the N_k direct PASTIS prototypes (n_regions=1), init
    the student from best_{k-1} (``load_state_dict(strict=False)``; step 0 = the
    teacher CLIP default of the trainer), train ``epochs_per_step`` epochs to
    convergence, eval per class on the held-out ``val_folds``, log one MLflow run
    per step, and evaluate the stop criterion. Returns the per-step results
    (metrics, best checkpoint path, stop_reason).

    Args:
        pastis_root: PASTIS-R root (rejected if it is the Italian/synthetic root).
        output_root: checkpoints root (relative -> lands on F: on the VM).
        run_name: MLflow run base name (``<run_name>-<NN>cls`` per step).
        step_size: classes added per step (default 2; plan B uses 4).
        base_classes: classes at step 0 (default 4 dominant).
        max_classes: curriculum cap (default 18).
        epochs_per_step: epochs per step (M, default 20; NOT the smoke 2).
        batch_size: DataLoader batch size.
        lr: AdamW learning rate.
        seed: determinism seed.
        folds: train folds (spatial CV).
        val_folds: held-out eval folds (disjoint from ``folds``; no leakage).
        dominance_ratio: 3:1 Meadow filter ratio.
        time_cap_hours: hard cap per step forwarded to the trainer; also the
            wall-clock budget (``budget_exhausted`` stops the curriculum).
        prototype_path: override of the US-033 parquet (read/filter only).
        mlflow_uri: MLflow tracking URI.

    Returns:
        The per-step :class:`StepResult` list, in execution order.

    Raises:
        ValueError: if ``pastis_root`` is the Italian/synthetic root, if
            ``val_folds`` overlaps ``folds`` (leakage), or if ``epochs_per_step``
            is below the productive floor (20).
    """
    _validate_pastis_root(pastis_root)
    overlap = set(folds) & set(val_folds)
    if overlap:
        raise ValueError(
            f"val_folds {sorted(val_folds)} overlaps train folds "
            f"{sorted(folds)} ({sorted(overlap)}): spatial CV leakage."
        )
    if epochs_per_step < 20:
        raise ValueError(
            f"epochs_per_step={epochs_per_step} is below the productive floor "
            "(20). US-036-a is a productive run, not a smoke; pass >= 20."
        )

    propagate_seed(seed)
    ranking = cardinality_ranking()
    total_steps = n_steps(step_size=step_size, base=base_classes, max_classes=max_classes)
    _log.info(
        "incremental curriculum start",
        run_name=run_name,
        ranking=ranking,
        step_size=step_size,
        base_classes=base_classes,
        max_classes=max_classes,
        total_steps=total_steps,
        epochs_per_step=epochs_per_step,
        folds=list(folds),
        val_folds=list(val_folds),
        dominance_ratio=dominance_ratio,
        pastis_root=str(pastis_root),
        output_root=str(output_root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        mlflow_uri=mlflow_uri,
    )

    # Load the US-033 prototype matrix ONCE (read-only; filtered per step).
    proto_18, ids_all = load_class_prototype_embeddings(
        prototype_path if prototype_path is not None else _default_prototype_path()
    )

    results: list[StepResult] = []
    best_prev: Path | None = None
    metrics_prev: StepMetrics | None = None
    curriculum_start = time.monotonic()

    for k in range(total_steps):
        class_ids = class_ids_for_step(k, step_size=step_size, base=base_classes, ranking=ranking)
        n = len(class_ids)
        step_dir = output_root / f"{n:02d}cls"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Wall-clock budget gate (budget_exhausted is the orchestrator's call).
        elapsed_h = (time.monotonic() - curriculum_start) / 3600.0
        if elapsed_h >= time_cap_hours and results:
            _log.warning(
                "curriculum budget exhausted before step",
                step=k,
                elapsed_hours=round(elapsed_h, 4),
                time_cap_hours=time_cap_hours,
            )
            results[-1].stop_reason = "budget_exhausted"
            break

        # Build the real PASTIS-R train dataset for this step (US-036 builder).
        train_ds, n_regions, n_categories, _proto_train = create_incremental_dataset(
            n,
            root=pastis_root,
            folds=folds,
            ratio=dominance_ratio,
            seed=seed,
            prototype_path=prototype_path,
        )
        # Direct PASTIS prototypes for this step, in category_id order (n_regions=1).
        proto_step = select_step_prototypes(proto_18, ids_all, class_ids)
        proto_step_t = torch.from_numpy(proto_step).float()

        cfg = FarSLIPTrainerConfig(
            dataset_root=pastis_root,  # drives MLflow data_version -> PASTIS-R
            output_dir=step_dir,
            n_epochs=epochs_per_step,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            time_cap_hours=time_cap_hours,
            n_in_channels=_N_IN_CHANNELS,
            n_regions=n_regions,  # always 1 for PASTIS
            n_categories=n_categories,
            mlflow_run_name=f"{run_name}-{n:02d}cls",
            extra_params={
                "n_classes": n,
                "n_categories": n_categories,
                "n_regions": n_regions,
                "class_ids": ",".join(str(c) for c in class_ids),
                "init_from": "teacher_clip" if best_prev is None else str(best_prev),
                "step_index": k,
                "step_size": step_size,
                "dominance_ratio": dominance_ratio,
                "folds": ",".join(str(f) for f in folds),
                "val_folds": ",".join(str(f) for f in val_folds),
                "proto_source": "pastis_direct",
                "dataset": "pastis_r_real",
                "incremental_seed": seed,
            },
        )
        trainer = FarSLIPDistillationTrainer(cfg, dataset=train_ds)

        # Chained init: step k>0 warm-starts from the previous best (strict=False).
        init_from = "teacher_clip"
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        if best_prev is not None:
            state_dict = _load_student_state_dict(best_prev)
            incompatible = trainer.student.load_state_dict(state_dict, strict=False)
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
            init_from = str(best_prev)
            _log.info(
                "chained init applied (strict=False)",
                step=k,
                n_classes=n,
                init_from=init_from,
                n_missing_keys=len(missing_keys),
                n_unexpected_keys=len(unexpected_keys),
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            )
        else:
            _log.info("step 0 inits from teacher CLIP default", step=k, n_classes=n)

        # Prototypes are rebuilt per step: the encoder transfers, the bank does not.
        trainer.set_text_prototypes(proto_step_t)
        train_metrics = trainer.train()

        # Persist the best (last epoch) as the canonical step checkpoint.
        best_ckpt = step_dir / "best.safetensors"
        last_epoch_ckpt = step_dir / f"student_epoch_{epochs_per_step - 1}.safetensors"
        if last_epoch_ckpt.exists():
            best_ckpt.write_bytes(last_epoch_ckpt.read_bytes())
        else:
            # Defensive: time cap hit before the last epoch -> explicit save.
            saved = trainer.save_student(format="safetensors", suffix="best")
            best_ckpt = Path(saved)

        # Per-class eval on the held-out validation fold (no leakage).
        val_ds, _vr, _vc, _vp = create_incremental_dataset(
            n,
            root=pastis_root,
            folds=val_folds,
            ratio=dominance_ratio,
            seed=seed,
            prototype_path=prototype_path,
        )
        # Reuse the trainer's reprojected (768-dim) bank so eval matches the loss.
        proto_eval = trainer._text_prototypes
        if proto_eval is None:  # pragma: no cover - set above, defensive
            raise RuntimeError("trainer text prototypes missing after set_text_prototypes")
        metrics = eval_per_class(
            trainer.student,
            val_ds,
            class_ids,
            proto_eval,
            device=trainer.device,
            batch_size=batch_size,
        )

        stop, reason = stop_criterion(metrics, metrics_prev, max_classes=max_classes)
        _log_step_run(
            mlflow_uri=mlflow_uri,
            run_name=f"{run_name}-{n:02d}cls",
            n_classes=n,
            class_ids=class_ids,
            metrics=metrics,
            init_from=init_from,
            epochs=epochs_per_step,
            step_size=step_size,
            dominance_ratio=dominance_ratio,
            folds=folds,
            val_folds=val_folds,
            pastis_root=pastis_root,
            train_metrics=train_metrics,
            stop_reason=reason if stop else "continue",
        )

        result = StepResult(
            n_classes=n,
            class_ids=class_ids,
            metrics=metrics,
            best_ckpt=best_ckpt,
            init_from=init_from,
            loss_cls_final=float(train_metrics.get("loss_cls", 0.0)),
            loss_total_final=float(train_metrics.get("loss_total", 0.0)),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            stop_reason=reason if stop else "continue",
        )
        results.append(result)
        _log.info(
            "step done",
            step=k,
            n_classes=n,
            best_ckpt=str(best_ckpt),
            macro_f1=round(metrics.macro_f1, 4),
            n_classes_well_resolved=metrics.n_classes_well_resolved,
            stop=stop,
            stop_reason=reason,
        )

        if stop:
            break
        best_prev = best_ckpt
        metrics_prev = metrics

    winner = _select_winner(results)
    _log.info(
        "incremental curriculum done",
        n_steps_run=len(results),
        winner_n_classes=winner.n_classes if winner else None,
        winner_best_ckpt=str(winner.best_ckpt) if winner else None,
        winner_macro_f1=round(winner.metrics.macro_f1, 4) if winner else None,
        dvc_add_hint=(f"dvc add {winner.best_ckpt} && dvc push" if winner else None),
        elapsed_hours=round((time.monotonic() - curriculum_start) / 3600.0, 4),
    )
    return results


def _select_winner(results: list[StepResult]) -> StepResult | None:
    """Return the step with the most well-resolved classes (tie -> higher macro-F1).

    Args:
        results: the per-step results.

    Returns:
        The winning :class:`StepResult` (its best checkpoint feeds US-037), or
        ``None`` if no step ran.
    """
    if not results:
        return None
    return max(
        results,
        key=lambda r: (r.metrics.n_classes_well_resolved, r.metrics.macro_f1),
    )


def _default_prototype_path() -> Path:
    """Return the default US-033 prototype parquet path (read-only)."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "features" / "phenology_class_prototypes_pastis.parquet"


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    run_name: Annotated[str, typer.Option(help="Nombre base del run MLflow")] = "farslip-full-incr",
    step_size: Annotated[int, typer.Option(help="Clases anadidas por escalon")] = 2,
    base_classes: Annotated[int, typer.Option(help="Clases en el escalon 0")] = 4,
    max_classes: Annotated[int, typer.Option(help="Tope del curriculum")] = 18,
    epochs_per_step: Annotated[
        int, typer.Option(help="Epochs por escalon (M, >= 20; run completo)")
    ] = 20,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    folds: Annotated[str, typer.Option(help="Folds de train PASTIS, coma-separados")] = "1,2,3",
    val_folds: Annotated[str, typer.Option(help="Folds de validacion (disjuntos de train)")] = "4",
    ratio: Annotated[float, typer.Option(help="Ratio filtro 3:1 Meadow")] = 3.0,
    pastis_root: Annotated[Path, typer.Option(help="Raiz PASTIS-R (frances real)")] = Path(
        "data/PASTIS-R"
    ),
    output_dir: Annotated[Path, typer.Option(help="Dir checkpoints (cae en F: en la VM)")] = Path(
        "checkpoints/farslip/incremental"
    ),
    time_cap_hours: Annotated[
        float, typer.Option(help="Hard cap horas por escalon y presupuesto total")
    ] = 8.0,
    prototype_path: Annotated[
        Path | None,
        typer.Option(help="Override del parquet US-033 (solo LEER/FILTRAR)"),
    ] = None,
    mlflow_uri: Annotated[
        str,
        typer.Option(help="MLflow tracking URI (Docker :5010; SQLite file:// CI)"),
    ] = _DEFAULT_MLFLOW_URI,
) -> None:
    """Ejecuta el fine-tune COMPLETO incremental de FarSLIP sobre PASTIS-R real.

    Curriculum por cardinalidad: escalon 0 = las ``base_classes`` dominantes
    (Meadow, Corn, Soft winter wheat, Grapevine); ``+step_size`` por escalon hasta
    ``max_classes`` (18). Cada escalon: ``create_incremental_dataset`` (US-036,
    PASTIS real, filtro 3:1) -> prototipos PASTIS directos (``n_regions=1``) ->
    ``FarSLIPDistillationTrainer`` (``n_in_channels=4``, ``n_epochs=epochs_per_step``)
    -> init encadenado ``load_state_dict(strict=False)`` desde el best previo
    (escalon 0 = teacher) -> ``set_text_prototypes`` -> ``train`` -> eval F1/IoU por
    clase sobre ``val_folds`` (held-out, sin leakage) -> un run MLflow por escalon
    -> ``stop_criterion``. El best del escalon ganador alimenta US-037.

    Args:
        run_name: MLflow run base name (per-step suffix appended).
        step_size: classes added per step.
        base_classes: classes at step 0.
        max_classes: curriculum cap.
        epochs_per_step: epochs per step (>= 20; productive run).
        batch_size: DataLoader batch size.
        lr: AdamW learning rate.
        seed: determinism seed.
        folds: comma-separated train folds (spatial CV).
        val_folds: comma-separated held-out eval folds (disjoint from train).
        ratio: 3:1 Meadow dominance filter ratio.
        pastis_root: PASTIS-R root.
        output_dir: checkpoints root (relative -> lands on F: on the VM).
        time_cap_hours: hard cap per step and total wall-clock budget.
        prototype_path: override of the US-033 parquet (read/filter only).
        mlflow_uri: MLflow tracking URI.
    """
    parsed_folds = _parse_folds(folds)
    parsed_val_folds = _parse_folds(val_folds)
    results = run_incremental_curriculum(
        pastis_root=pastis_root,
        output_root=output_dir,
        run_name=run_name,
        step_size=step_size,
        base_classes=base_classes,
        max_classes=max_classes,
        epochs_per_step=epochs_per_step,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        folds=parsed_folds,
        val_folds=parsed_val_folds,
        dominance_ratio=ratio,
        time_cap_hours=time_cap_hours,
        prototype_path=prototype_path,
        mlflow_uri=mlflow_uri,
    )
    winner = _select_winner(results)
    if winner is not None:
        _log.info(
            "winner step (input to US-037)",
            n_classes=winner.n_classes,
            best_ckpt=str(winner.best_ckpt),
            macro_f1=round(winner.metrics.macro_f1, 4),
            n_classes_well_resolved=winner.metrics.n_classes_well_resolved,
            stop_reason=results[-1].stop_reason,
            dvc_add_hint=f"dvc add {winner.best_ckpt} && dvc push",
        )


if __name__ == "__main__":  # pragma: no cover
    app()
