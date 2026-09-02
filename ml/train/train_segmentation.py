"""Typer CLI to train PASTIS-R dense segmentation models (EPIC 5, Avance 4).

Orchestrates the training of the architectures assigned to Aaron in the team's
split: **#1 U-Net ResNet-50** (2D temporal composite) and **#6 AnySat frozen +
linear head** (time series). Shares the dense pipeline
(:mod:`ml.ingest.pastis_dataset`), the pixel-level metrics
(:mod:`ml.eval.dense_metrics`) and the MLflow tracking (:mod:`ml.utils.mlflow_utils`).

The modeling logic lives in the factories (:mod:`ml.models.segmentation`,
:mod:`ml.models.anysat_wrapper`); this module only orchestrates the training
loop, the per-epoch evaluation and the artifact persistence (separation of
concerns, CLAUDE.md rule 8).

Usage (local CPU smoke)::

    poetry run python -m ml.train.train_segmentation \\
        --model unet --subset 4 --epochs 1 --device cpu

Usage (real run on Colab L4)::

    poetry run python -m ml.train.train_segmentation \\
        --model unet --epochs 30 --batch-size 8 --device cuda

Permanent operational tool (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import mlflow
import numpy as np
import polars as pl
import structlog
import torch
import typer
from torch import nn
from torch.utils.data import DataLoader

from ml.analysis.hcat_grouping import hcat6_dense_lut
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm
from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTIS_NUM_CLASSES,
    PASTISDataset,
    load_norm_stats,
    pastis_fold_split,
)
from ml.losses.dirpa import DirPALogitAdjuster
from ml.models.deeplabv3plus import build_dice_ce_loss
from ml.utils.mlflow_utils import track_experiment

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from collections.abc import Sequence

    from torch.utils.data import Dataset

    from ml.data.pastis_seg_dataset import CollapseMode, TargetMode

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# MLflow 3.x emits emojis when closing runs; the Windows console uses cp1252 and
# that causes UnicodeEncodeError. Force UTF-8 (no-op on Linux/macOS/Colab).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

_EXPERIMENT_NAME = "agrosat-segmentation"
_DEFAULT_OUTPUT = Path("artifacts/segmentation")
_DEFAULT_COMPARISON = Path("reports/segmentation/model_comparison_avance4_aaron.parquet")
_DEFAULT_ROOT = Path("data/PASTIS-R")
#: Path (relative to the repo) of the PASTIS-R dataset to resolve the ``data_version``
#: DVC in the DeepLab/TSViT trainers (us-025).
_PASTIS_DVC_PATH = "data/PASTIS-R"


def _parse_folds(spec: str) -> tuple[int, ...]:
    """Parses ``"1,2,3"`` into ``(1, 2, 3)``."""
    return tuple(int(x) for x in spec.split(",") if x.strip())


def _build_model(model_name: str, num_classes: int, target_size: int) -> tuple[nn.Module, str]:
    """Builds the model and returns ``(model, temporal_reduction)``.

    Args:
        model_name: ``unet`` or ``anysat``.
        num_classes: Number of output classes.
        target_size: Spatial side of the logits.

    Returns:
        Tuple ``(nn.Module, temporal_reduction)`` where the temporal reduction is
        ``"median"`` for 2D models and ``"none"`` for AnySat.

    Raises:
        typer.BadParameter: if ``model_name`` is not supported by this CLI.
    """
    if model_name == "unet":
        from ml.models.segmentation import build_unet

        return build_unet(num_classes), "median"
    if model_name == "anysat":
        from ml.models.anysat_wrapper import AnySatSegmenter

        return AnySatSegmenter(num_classes, target_size=target_size), "none"
    raise typer.BadParameter("`--model` debe ser 'unet' o 'anysat'.")


def _forward(
    model: nn.Module, model_name: str, batch: dict[str, Any], device: torch.device
) -> torch.Tensor:
    """Runs the forward adapted to each model's signature.

    Args:
        model: Model to evaluate.
        model_name: ``unet`` (2D) or ``anysat`` (temporal with dates).
        batch: DataLoader batch with ``image`` and optionally ``dates``.
        device: Target device.

    Returns:
        Logits ``(B, num_classes, H, W)``.
    """
    image = batch["image"].to(device)
    if model_name == "anysat":
        dates = batch.get("dates")
        dates = dates.to(device) if dates is not None else None
        return cast("torch.Tensor", model(image, dates))
    return cast("torch.Tensor", model(image))


def _make_loader(
    patch_ids: list[str],
    *,
    root: Path,
    reduction: str,
    target_size: int,
    norm: tuple,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    """Builds a ``DataLoader`` over a :class:`PASTISDataset`.

    With a GPU, ``pin_memory`` is advisable (speeds up the transfer to the GPU)
    and, if there are several workers, ``persistent_workers`` to avoid recreating
    them on each epoch.
    """
    dataset = PASTISDataset(
        patch_ids,
        root=root,
        target_size=target_size,
        temporal_reduction=reduction,  # type: ignore[arg-type]
        norm=norm,
    )
    # ``prefetch_factor`` only applies with worker processes; passing it with
    # ``num_workers=0`` raises. With workers it overlaps the per-item temporal
    # median (the U-Net loading cost) with the GPU step so the H100 stays fed.
    extra: dict[str, Any] = {}
    if num_workers > 0:
        extra["prefetch_factor"] = 4
        extra["persistent_workers"] = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        **extra,
    )


def _evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    group_lut: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluates the model and returns mIoU/F1-macro/pixel-accuracy.

    If ``group_lut`` is passed (LUT 18 classes -> 6 HCAT groups), it also computes
    the same three metrics over the 6 agronomic groups (suffix ``_grouped``), for
    comparability with the baseline. Background and void do not enter those
    metrics; predicting background over a crop pixel is penalized with an extra
    "non-crop" class (id 6) that is never a target, so the macro averages only
    the 6 crop groups.
    """
    model.eval()
    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index, device=str(device))
    acc_grouped = None
    lut_target = lut_pred = None
    if group_lut is not None:
        acc_grouped = DenseConfusionAccumulator(7, ignore_index=255, device=str(device))
        lut_target = torch.as_tensor(group_lut, device=device)
        _pred_lut = group_lut.copy()
        _pred_lut[_pred_lut == 255] = 6  # predicted background/void -> "non-crop" class
        lut_pred = torch.as_tensor(_pred_lut, device=device)
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="validacion", leave=False, unit="batch")
    with torch.no_grad():
        for batch in iterator:
            logits = _forward(model, model_name, batch, device)
            preds = logits.argmax(dim=1)
            target = batch["semantic"].to(device)
            acc.update(preds, target)
            if acc_grouped is not None:
                # The three grouped objects are always built together above.
                assert lut_pred is not None and lut_target is not None
                acc_grouped.update(lut_pred[preds.clamp(0, 19)], lut_target[target.clamp(0, 19)])
    flat = acc.compute()
    if acc_grouped is None:
        return flat
    grouped = {f"{k}_grouped": v for k, v in acc_grouped.compute().items()}
    return {**flat, **grouped}


def _upsert_comparison_row(row: dict[str, Any], comparison_path: Path) -> None:
    """Inserts/updates the model's metrics row in the comparison parquet.

    Reads the existing parquet (if any), removes any previous row for the same
    ``model`` and writes the new version. This parquet is consumed by the
    integrator notebook ``Avance4.Equipo17.ipynb`` for the comparison table of
    the 6 models.

    Args:
        row: Metrics row of the just-trained model.
        comparison_path: Path of the comparison parquet.
    """
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([row])
    if comparison_path.exists():
        existing = pl.read_parquet(comparison_path).filter(pl.col("model") != row["model"])
        new = pl.concat([existing, new], how="vertical_relaxed")
    new.write_parquet(comparison_path)


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Saves the full state to be able to resume training.

    Persists model, optimizer, scaler, the last completed epoch, the best metrics
    and the config (to validate that the checkpoint corresponds to the same run).
    It is overwritten each epoch; on Colab it is advisable to point it to Drive so
    that it survives a session restart.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Atomic write: first to a .tmp and then rename, to avoid corrupting the
    # checkpoint if the session is cut off right during the save.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best": best,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)


def run_training(
    *,
    model: str = "unet",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    target_size: int = 256,
    train_folds: str = "1,2,3",
    val_folds: str = "4",
    subset: int = 0,
    device: str = "auto",
    num_workers: int = 0,
    root: Path = _DEFAULT_ROOT,
    output_dir: Path = _DEFAULT_OUTPUT,
    comparison_path: Path = _DEFAULT_COMPARISON,
    mlflow_uri: str = "",
    resume: bool = True,
    checkpoint_every: int = 1,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    dirpa_alpha: float = 1.0,
    dirpa_tau: float = 0.0,
) -> dict[str, Any]:
    """Trains a dense segmentation model and logs metrics to MLflow.

    Function reusable by the CLI (:func:`main`) and by the Colab notebook, so that
    both run exactly the same training logic.

    Args:
        model: ``unet`` or ``anysat``.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        target_size: Target spatial resolution (256 by team convention).
        train_folds: Official PASTIS folds assigned to train (e.g. ``"1,2,3"``).
        val_folds: Folds assigned to validation (e.g. ``"4"``).
        subset: Limits the number of patches per split (dev/CI; 0 = all).
        device: ``cpu``, ``cuda`` or ``auto``.
        num_workers: DataLoader workers (0 recommended on Windows).
        root: Root of the PASTIS-R dataset.
        output_dir: Target directory for the ``.pt`` checkpoints.
        comparison_path: Comparison parquet consumed by the integrator notebook.
        mlflow_uri: Override of the MLflow tracking URI (empty = auto-resolution).
        on_epoch: Optional callback ``(epoch, metrics)`` invoked after evaluating
            each epoch. Used by the Optuna fine-tuning to report the intermediate
            metric and prune bad trials (``optuna.TrialPruned``); if it raises,
            the exception propagates and aborts the training of that trial.

    Returns:
        Dictionary with ``model``, ``miou``, ``f1_macro``, ``pixel_accuracy``,
        ``train_time_s`` and ``checkpoint_path``.

    Raises:
        FileNotFoundError: if the PASTIS-R root does not exist.
        RuntimeError: if the train/val split is empty.
    """
    if not root.exists():
        raise FileNotFoundError(f"PASTIS-R root not found: {root}")

    dev = _resolve_device(device)
    tr_folds = _parse_folds(train_folds)
    va_folds = _parse_folds(val_folds)
    split = pastis_fold_split(root, train_folds=tr_folds, val_folds=va_folds, test_folds=())
    train_ids, val_ids = split["train"], split["val"]
    if subset > 0:
        train_ids, val_ids = train_ids[:subset], val_ids[: max(1, subset // 2)]
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Empty PASTIS split (n_train={len(train_ids)}, n_val={len(val_ids)}). "
            "Check the folds and that metadata.geojson has the Fold field."
        )

    seg_model, reduction = _build_model(model, PASTIS_NUM_CLASSES, target_size)
    seg_model = seg_model.to(dev)
    # Normalization with stats from the train folds (no leakage from the val fold).
    norm = load_norm_stats(root, folds=tr_folds)

    pin = dev.type == "cuda"
    train_loader = _make_loader(
        train_ids,
        root=root,
        reduction=reduction,
        target_size=target_size,
        norm=norm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = _make_loader(
        val_ids,
        root=root,
        reduction=reduction,
        target_size=target_size,
        norm=norm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # Materialize Lazy parameters (the AnySat Conv1x1 head uses nn.LazyConv2d,
    # which infers its channels on the first forward) with a real batch BEFORE
    # building the optimizer; otherwise the param count and AdamW fail
    # over UninitializedParameter.
    seg_model.train()
    with torch.no_grad():
        _forward(seg_model, model, next(iter(train_loader)), dev)

    trainable = [p for p in seg_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PASTIS_IGNORE_INDEX)
    # DirPA (Reuss et al. 2026): when dirpa_tau > 0, perturb the logits by a
    # per-step symmetric-Dirichlet pseudo-prior before the CE, making the model
    # robust to the train/deploy prior shift that sinks minority crop classes
    # (e.g. sunflower/soybeans/durum). No-op at tau=0 and at inference.
    dirpa_unet = DirPALogitAdjuster(alpha=dirpa_alpha, tau=dirpa_tau)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume: if there is a checkpoint from the same run, continue from the
    # next epoch instead of starting from scratch (key in Colab, ephemeral session).
    resume_ckpt_path = output_dir / f"{model}_ckpt.pt"
    final_model_path = output_dir / f"{model}_pastis.pt"
    ckpt_config = {"model": model, "target_size": target_size, "epochs": epochs}
    start_epoch = 0
    best: dict[str, float] = {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}
    if resume and resume_ckpt_path.exists():
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=dev)
            if ckpt.get("config") == ckpt_config:
                seg_model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                best = ckpt["best"]
                start_epoch = ckpt["epoch"] + 1
                logger.info("segmentation_resume", model=model, start_epoch=start_epoch, **best)
            else:
                logger.warning("segmentation_ckpt_config_mismatch", path=str(resume_ckpt_path))
        except Exception as exc:  # noqa: BLE001 - corrupt checkpoint: start from scratch
            logger.warning(
                "segmentation_ckpt_load_failed", path=str(resume_ckpt_path), error=str(exc)
            )

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in seg_model.parameters())
    logger.info(
        "segmentation_train_start",
        model=model,
        device=str(dev),
        n_train=len(train_ids),
        n_val=len(val_ids),
        epochs=epochs,
        n_trainable=n_trainable,
        n_total=n_total,
    )

    run_name = f"seg-{model}-pastis-v1"
    tracking_override = mlflow_uri or None
    start = time.perf_counter()

    with track_experiment(
        _EXPERIMENT_NAME, run_name=run_name, tracking_uri=tracking_override, dvc_path=str(root)
    ):
        mlflow.set_tag("architecture", model)
        mlflow.log_params(
            {
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "target_size": target_size,
                "train_folds": train_folds,
                "val_folds": val_folds,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_trainable_params": n_trainable,
                "n_total_params": n_total,
                "device": str(dev),
            }
        )

        # LUT 18 classes -> 6 HCAT groups to also report the grouped metric
        # (comparable with the baseline). See ml.analysis.hcat_grouping.
        group_lut = hcat6_dense_lut()
        # Per-epoch history for the loss/mIoU curves. It is persisted to Drive
        # alongside the comparison parquet and resumed if it already exists (survives cutoffs).
        history_path = comparison_path.with_name(
            comparison_path.name.replace("model_comparison_avance4", "history")
        )
        history: list[dict[str, float]] = []
        if resume and start_epoch > 0 and history_path.exists():
            history = pl.read_parquet(history_path).to_dicts()
        for epoch in range(start_epoch, epochs):
            seg_model.train()
            if model == "anysat":
                # The frozen encoder stays in eval; only the head trains.
                # ``nn.Module.__getattr__`` types submodules as ``Tensor | Module``.
                cast("nn.Module", seg_model.encoder).eval()
            epoch_loss = 0.0
            # Per-batch progress bar within the epoch (progress, it/s, loss).
            # ``tqdm`` ships no type information, so the bar is typed as ``Any``.
            bar: Any = train_loader
            if tqdm is not None:
                bar = tqdm(
                    train_loader,
                    desc=f"epoca {epoch + 1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            for step, batch in enumerate(bar, 1):
                target = batch["semantic"].to(dev)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = _forward(seg_model, model, batch, dev)
                    logits = dirpa_unet(logits, training=True)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.detach())
                if tqdm is not None:
                    bar.set_postfix(loss=f"{epoch_loss / step:.3f}")

            metrics = _evaluate(
                seg_model,
                model,
                val_loader,
                dev,
                PASTIS_NUM_CLASSES,
                PASTIS_IGNORE_INDEX,
                group_lut=group_lut,
            )
            train_loss = epoch_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)
            if metrics["miou"] >= best["miou"]:
                best = metrics
            logger.info("segmentation_epoch", epoch=epoch, loss=epoch_loss, **metrics)
            # Per-epoch hook (Optuna fine-tuning: reports intermediate metric and prunes).
            # If it raises (TrialPruned), the exception propagates and cuts off this training.
            if on_epoch is not None:
                on_epoch(epoch, metrics)
            # Per-epoch history logging (for the curves) + persistence to Drive.
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(history).write_parquet(history_path)
            # Resumable checkpoint every `checkpoint_every` epochs (and on the last one).
            if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
                _save_checkpoint(
                    resume_ckpt_path,
                    epoch=epoch,
                    model=seg_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    best=best,
                    config=ckpt_config,
                )

        train_time_s = time.perf_counter() - start
        mlflow.log_metric("train_time_s", train_time_s)

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(seg_model.state_dict(), final_model_path)
        # The checkpoint is already on disk (output_dir). Logging it as an MLflow
        # artifact is best-effort: an HTTP tracking server without an artifact
        # proxy (--serve-artifacts) rejects the upload, and that must not abort a
        # finished 30-epoch run. Params/metrics/tags are already registered.
        try:
            mlflow.log_artifact(str(final_model_path))
        except Exception as exc:  # noqa: BLE001 - artifact store optional
            logger.warning(
                "mlflow_log_artifact_skipped",
                path=str(final_model_path),
                error=str(exc),
            )

        comparison_row = {
            "model": model,
            "miou": best["miou"],
            "f1_macro": best["f1_macro"],
            "pixel_accuracy": best["pixel_accuracy"],
            "miou_grouped": best.get("miou_grouped"),
            "f1_macro_grouped": best.get("f1_macro_grouped"),
            "pixel_accuracy_grouped": best.get("pixel_accuracy_grouped"),
            "train_time_s": train_time_s,
            "epochs": epochs,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_trainable_params": n_trainable,
            "target_size": target_size,
            "device": str(dev),
        }
        _upsert_comparison_row(comparison_row, comparison_path)

    logger.info("segmentation_train_done", model=model, **best)
    return {
        "model": model,
        "miou": best["miou"],
        "f1_macro": best["f1_macro"],
        "pixel_accuracy": best["pixel_accuracy"],
        "miou_grouped": best.get("miou_grouped"),
        "f1_macro_grouped": best.get("f1_macro_grouped"),
        "pixel_accuracy_grouped": best.get("pixel_accuracy_grouped"),
        "train_time_s": train_time_s,
        "checkpoint_path": str(final_model_path),
    }


# ===========================================================================
# US-025 trainers: DeepLabv3+ (2D) and TSViT (temporal, + phenology branch).
# Own APIs (train_segmentation / build_and_train) that the 5a/5b notebooks
# invoke by subprocess with --model deeplabv3plus|tsvit|tsvit-pheno.
# ===========================================================================


def phenology_contrastive_loss(
    visual_proj: torch.Tensor,
    target: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    ignore_index: int = 255,
    temperature: float = 0.07,
    max_pixels: int = 4096,
) -> torch.Tensor:
    """Symmetric InfoNCE pixel-visual <-> class-prototype (Wen et al. 2025).

    Aligns each per-pixel visual feature ``visual_proj[:, :, i, j]`` with the
    semantic prototype of **that pixel's class** (``target[:, i, j]``), following
    the paper's contrastive alignment (eq. 15-16, ``L_cl = (L_v + L_s) / 2``).
    Unlike the tabular concatenation (which degraded the baseline), the contrast
    pushes the visual features toward the semantic cluster of their class without
    inflating the head's dimensionality.

    Implementation:

    1. The valid pixels are flattened (``target != ignore_index`` and within
       ``[0, num_prototypes)``).
    2. ``max_pixels`` pixels are subsampled (bounded GPU memory; the contrast
       does not need all the pixels in the batch for a stable gradient signal).
    3. ``visual_proj`` and ``prototypes`` are L2-normalized; the similarity
       matrix ``logits = (v @ p^T) / temperature`` is compared against each
       pixel's class label with CrossEntropy in **both directions**
       (pixel->prototype and prototype->aggregated pixel), then averaged.

    Args:
        visual_proj: Per-pixel visual projection ``(B, S, H, W)`` with ``S`` =
            dimension of the prototypes' semantic space (384).
        target: Per-pixel labels ``(B, H, W)`` int; ``ignore_index`` and
            out-of-range classes are excluded.
        prototypes: Per-class prototype matrix ``(K, S)`` (one per class,
            indexable by the class label).
        ignore_index: Label of pixels to ignore (Background/Void).
        temperature: Temperature of the contrastive softmax (0.07, CLIP/InfoNCE
            standard).
        max_pixels: Maximum number of valid pixels to use per call (deterministic
            subsampling via ``torch.randperm`` to bound VRAM).

    Returns:
        Scalar ``torch.Tensor`` with the symmetric contrastive loss. If there are
        no valid pixels in the batch, returns ``0.0`` (tensor with grad).
    """
    device = visual_proj.device
    semantic_dim = visual_proj.shape[1]
    n_proto = prototypes.shape[0]

    protos = prototypes.to(device=device, dtype=visual_proj.dtype)
    protos = nn.functional.normalize(protos, dim=1)  # (K, S)

    # (B, S, H, W) -> (B*H*W, S) y (B, H, W) -> (B*H*W,)
    v_flat = visual_proj.permute(0, 2, 3, 1).reshape(-1, semantic_dim)
    y_flat = target.reshape(-1).long()

    valid = (y_flat != ignore_index) & (y_flat >= 0) & (y_flat < n_proto)
    if not bool(valid.any()):
        # No valid pixels: neutral term that preserves the grad graph.
        return visual_proj.sum() * 0.0

    v_valid = v_flat[valid]
    y_valid = y_flat[valid]

    # Deterministic subsampling to bound the similarity matrix in VRAM.
    n_valid = v_valid.shape[0]
    if n_valid > max_pixels:
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(n_valid, generator=gen)[:max_pixels].to(device)
        v_valid = v_valid[perm]
        y_valid = y_valid[perm]

    v_valid = nn.functional.normalize(v_valid, dim=1)  # (P, S)

    # Pixel x prototype similarity -> logits (P, K).
    logits = (v_valid @ protos.t()) / temperature

    # Direction 1 (visual): each pixel must classify to its class prototype.
    loss_v = nn.functional.cross_entropy(logits, y_valid)

    # Direction 2 (semantic): for each present class, the prototype must
    # recover its pixels. The prototype->pixels similarity of its class is
    # averaged against all pixels in the batch (symmetric InfoNCE from the paper).
    present = torch.unique(y_valid)
    proto_logits = (protos[present] @ v_valid.t()) / temperature  # (Kp, P)
    # Multi-positive target: for each present prototype, the pixels of its
    # class are the positives; the mean of log-softmax over positives is used.
    log_prob = nn.functional.log_softmax(proto_logits, dim=1)  # (Kp, P)
    pos_mask = (y_valid.unsqueeze(0) == present.unsqueeze(1)).to(log_prob.dtype)
    pos_counts = pos_mask.sum(dim=1).clamp_min(1.0)
    loss_s = -(log_prob * pos_mask).sum(dim=1) / pos_counts
    loss_s = loss_s.mean()

    loss: torch.Tensor = 0.5 * (loss_v + loss_s)
    return loss


# ---------------------------------------------------------------------------
# Device / forward helpers
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> torch.device:
    """Resolves the device, prioritizing CUDA when available.

    Args:
        requested: ``"cuda"``, ``"cpu"`` or ``"auto"``. ``"cuda"`` without a GPU
            degrades to ``"cpu"`` with a structured warning.

    Returns:
        Resolved ``torch.device``.
    """
    if requested in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            logger.warning("cuda_requested_but_unavailable_fallback_cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _forward_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    return_visual_proj: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Runs the model's ``forward`` returning logits and visual projection.

    TSViT accepts the kwarg ``return_visual_proj`` and, when it is ``True``,
    returns the tuple ``(logits, visual_proj)``. DeepLabv3+ (smp) does not accept
    the kwarg: it is called in the standard way and ``visual_proj`` is ``None``.

    Args:
        model: Segmenter (DeepLabv3+ or TSViT).
        x: Input ``(B, C, H, W)`` (2D) or ``(B, T, C, H, W)`` (temporal).
        return_visual_proj: If ``True`` and the model supports it, requests the
            contrastive visual branch.

    Returns:
        Tuple ``(logits (B, K, H, W), visual_proj | None)``.
    """
    if return_visual_proj:
        out = model(x, return_visual_proj=True)
        if isinstance(out, tuple):
            return out[0], out[1]
        # The model did not honor the flag (defensive case): only logits.
        return out, None
    out = model(x)
    return out, None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    use_phenology: bool,
    prototypes: torch.Tensor | None,
    lambda_contrast: float,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    dirpa: DirPALogitAdjuster | None = None,
) -> float:
    """Runs one train epoch (with ``optimizer``) or eval epoch (without it).

    Args:
        model: Segmenter.
        loader: DataLoader of the split.
        criterion: Segmentation loss (Dice + CE).
        device: Resolved device.
        use_phenology: Enables the contrastive branch (only if the model exposes
            it).
        prototypes: Prototype matrix ``(K, S)`` or ``None``.
        lambda_contrast: Weight of the contrastive term.
        ignore_index: Ignored label (Background/Void).
        optimizer: Optimizer for train; ``None`` for eval (no backward).
        scaler: AMP ``GradScaler`` or ``None``.
        use_amp: If ``True`` uses autocast (only effective on CUDA).
        dirpa: Optional DirPA logit adjuster. When set (``tau > 0``) the dense
            logits are perturbed by a per-step Dirichlet pseudo-prior before the
            CE during TRAIN only (eval is a pass-through), so the contrastive
            term and metrics see the unperturbed projection.

    Returns:
        Mean loss of the epoch (Python scalar).
    """
    is_train = optimizer is not None
    model.train(is_train)

    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    n_batches = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, visual_proj = _forward_model(model, x, return_visual_proj=use_phenology)
                if dirpa is not None:
                    logits = dirpa(logits, training=is_train)
                loss = criterion(logits, y)
                if (
                    use_phenology
                    and visual_proj is not None
                    and prototypes is not None
                    and lambda_contrast > 0.0
                ):
                    loss = loss + lambda_contrast * phenology_contrastive_loss(
                        visual_proj, y, prototypes, ignore_index=ignore_index
                    )

            if optimizer is not None:
                # Gradient clipping (max_norm=1.0): essential for TSViT
                # (transformer) — without it, the gradients explode and the loss
                # diverges to NaN after ~8 epochs. With AMP you must `unscale_`
                # before clipping. DeepLabv3+ (CNN) tolerates not clipping, but
                # applying it to both is safe and stabilizes.
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

    return total_loss / max(1, n_batches)


def _evaluate_dense(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    use_phenology: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluates the model accumulating the split's dense confusion matrix.

    Accumulates the confusion over the whole split (not per-batch) so that
    mIoU/F1 are exact at the set level. Reuses the helpers from
    :mod:`ml.eval.metrics`.

    Args:
        model: Segmenter.
        loader: DataLoader of the validation split.
        device: Resolved device.
        num_classes: Number of classes of the dense logit (18 or 6).
        ignore_index: Ignored label.
        use_phenology: If ``True`` the forward is run requesting the visual branch
            (discarded for the metric; only the logits matter).

    Returns:
        Tuple ``(metrics, cm)``: the full metrics dictionary (``miou``,
        ``f1_macro``, ``pixel_acc``, ``balanced_acc``, ``cohen_kappa``,
        ``per_class_iou``, ``per_class_f1``) and the split's accumulated dense
        confusion matrix (for artifacts at the end).
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            logits, _ = _forward_model(model, x, return_visual_proj=use_phenology)
            preds = logits.argmax(dim=1)
            cm += dense_confusion_matrix(preds, y, n_classes=num_classes, ignore_index=ignore_index)

    return dense_metrics_from_cm(cm), cm


# ---------------------------------------------------------------------------
# Per-epoch checkpointing (resume after interruption).
# ---------------------------------------------------------------------------


def _save_checkpoint_seg(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    best_metrics: dict[str, float],
) -> None:
    """Persists the full training state for resuming.

    Saves ``model``/``optimizer``/``scaler`` state_dicts + the already-completed
    ``epoch`` + the best metrics, atomically (writes to ``.tmp`` and renames) so
    as not to corrupt the checkpoint if the process dies mid-write.

    Args:
        path: Target path of the checkpoint (``.pt``).
        epoch: Index of the last COMPLETED epoch (0-based).
        model: Model whose state_dict is saved.
        optimizer: AdamW optimizer.
        scaler: AMP GradScaler (or ``None`` if AMP is not used).
        best_metrics: Best validation metrics so far.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metrics": best_metrics,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint_seg(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    """Loads a checkpoint and restores the training state.

    Args:
        path: Path of the ``.pt`` checkpoint.
        model: Model to restore (in-place).
        optimizer: Optimizer to restore (in-place).
        scaler: GradScaler to restore (in-place) or ``None``.
        device: Target device to map the tensors.

    Returns:
        ``(start_epoch, best_metrics)``: the epoch from which to continue
        (= last completed + 1) and the previous best metrics.
    """
    # Restricted unpickling: the checkpoint holds only state_dicts and scalars
    # (model/optimizer/scaler/scheduler state, epoch, best_metrics), so
    # weights_only=True is sufficient and avoids the pickle RCE vector when a
    # checkpoint is pulled from a shared store (per the us-017 resume convention).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt["epoch"]) + 1
    best = ckpt.get("best_metrics") or {
        "miou": -1.0,
        "f1_macro": 0.0,
        "pixel_acc": 0.0,
    }
    logger.info(
        "checkpoint_resumed",
        path=str(path),
        start_epoch=start_epoch,
        best_miou=round(best.get("miou", -1.0), 4),
    )
    return start_epoch, best


def _log_final_artifacts(
    ckpt_dir: Path,
    *,
    best_cm: np.ndarray,
    best_metrics: dict[str, float],
    num_classes: int,
) -> None:
    """Generates and logs to MLflow the final artifacts of the best epoch.

    Produces two artifacts of the best validation model:
    1. ``confusion_matrix.png``: normalized confusion matrix (recall), useful to
       see which classes/groups the model confuses.
    2. ``per_class_metrics.json``: per-class IoU and F1 + the macro metrics.

    Args:
        ckpt_dir: Directory where the artifacts are written before uploading them.
        best_cm: Dense confusion matrix of the best epoch.
        best_metrics: Metrics dictionary of the best epoch (includes
            ``per_class_iou`` and ``per_class_f1``).
        num_classes: Number of classes (18 or 6).
    """
    import json

    import matplotlib.pyplot as plt
    import mlflow

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix (figure). Reconstructing y_true/y_pred from the cm by
    # expanding counts would be costly; instead we draw the cm directly.
    cm_f = best_cm.astype(np.float64)
    row_sums = cm_f.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0.0, cm_f / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusion normalizada ({num_classes} clases)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = ckpt_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics_path = ckpt_dir / "per_class_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "num_classes": num_classes,
                "best_epoch": best_metrics.get("best_epoch"),
                "miou": best_metrics.get("miou"),
                "f1_macro": best_metrics.get("f1_macro"),
                "pixel_acc": best_metrics.get("pixel_acc"),
                "balanced_acc": best_metrics.get("balanced_acc"),
                "cohen_kappa": best_metrics.get("cohen_kappa"),
                "per_class_iou": best_metrics.get("per_class_iou"),
                "per_class_f1": best_metrics.get("per_class_f1"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.log_artifact(str(cm_path), artifact_path="eval")
    mlflow.log_artifact(str(metrics_path), artifact_path="eval")
    logger.info("final_artifacts_logged", cm=str(cm_path), metrics=str(metrics_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_segmentation(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    mlflow_run_name: str,
    epochs: int,
    batch_size: int,
    device: str = "cuda",
    lr: float = 1e-3,
    use_phenology: bool = False,
    prototypes: np.ndarray | Sequence[Sequence[float]] | torch.Tensor | None = None,
    lambda_contrast: float = 0.3,
    num_workers: int = 0,
    use_amp: bool = True,
    ignore_index: int = 255,
    num_classes: int | None = None,
    mlflow_uri: str | None = None,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    ckpt_dir: str | Path | None = None,
    resume: bool = True,
    warmup_epochs: int = 10,
    lr_min: float = 5e-6,
    patience: int = 0,
    dirpa_alpha: float = 1.0,
    dirpa_tau: float = 0.0,
) -> dict[str, float]:
    """Trains a PASTIS-R dense segmenter with MLflow logging.

    Loop shared by DeepLabv3+ (2D) and TSViT (temporal + optional
    phenology-contrastive branch). On each epoch it trains over ``train_ds``,
    evaluates on ``val_ds`` and logs ``loss``/``miou``/``f1_macro``/``pixel_acc``
    to MLflow (tags ``data_version`` + ``code_version`` via
    :func:`ml.utils.mlflow_utils.track_experiment`). Keeps the best epoch by
    validation mIoU and returns its metrics.

    Args:
        model: Built segmenter (``build_deeplabv3plus_mobilenet`` or
            ``build_tsvit``). For the contrastive branch it must accept
            ``return_visual_proj=True`` (TSViT).
        train_ds: Training dataset (``PASTISSegmentationDataset`` in 2D or
            temporal mode depending on the model).
        val_ds: Validation dataset (folds disjoint from train).
        mlflow_run_name: MLflow run name
            (``"alt-deeplabv3plus-mobilenet-v1"`` or ``"alt-tsvit-v1"`` /
            ``"alt-tsvit-pheno-v1"``).
        epochs: Number of epochs.
        batch_size: Batch size of the ``DataLoader``.
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``. CUDA with a missing GPU
            degrades to CPU.
        lr: Learning rate of the AdamW optimizer.
        use_phenology: If ``True`` adds the contrastive term
            ``lambda_contrast * L_contrast`` (requires ``prototypes`` and a model
            that exposes the visual branch).
        prototypes: Per-class prototype matrix ``(K, S)`` (numpy, list or
            tensor). Mandatory if ``use_phenology=True``.
        lambda_contrast: Weight of the contrastive term in the sum.
        num_workers: ``DataLoader`` workers (0 on Windows/CI to avoid the spawn
            cost).
        use_amp: If ``True`` uses mixed-precision autocast (only effective on
            CUDA; no-op on CPU).
        ignore_index: Ignored label (Background/Void).
        num_classes: Number of classes of the dense logit. If ``None`` it is
            inferred from ``train_ds.num_classes`` (or 18 by default).
        mlflow_uri: Override of the MLflow tracking URI; ``None`` delegates to
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        dice_weight: Weight of the Dice term in the segmentation loss.
        ce_weight: Weight of the CrossEntropy term in the segmentation loss.

    Returns:
        Dictionary ``{"miou", "f1_macro", "pixel_acc"}`` of the **best
        validation epoch** (by mIoU).

    Raises:
        ValueError: if ``use_phenology=True`` but ``prototypes`` is not passed,
            or if ``epochs`` is not positive.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, received {epochs}.")
    if use_phenology and prototypes is None:
        raise ValueError("use_phenology=True requires `prototypes` (per-class (K, S) matrix).")

    resolved_device = _resolve_device(device)
    resolved_classes = int(
        num_classes if num_classes is not None else getattr(train_ds, "num_classes", 18)
    )

    model = model.to(resolved_device)

    proto_tensor: torch.Tensor | None = None
    if use_phenology and prototypes is not None:
        proto_tensor = (
            prototypes
            if isinstance(prototypes, torch.Tensor)
            else torch.as_tensor(np.asarray(prototypes), dtype=torch.float32)
        ).to(resolved_device)

    # `persistent_workers` avoids re-spawning the workers on each epoch (high
    # cost on Windows with spawn); `prefetch_factor` preloads several batches per
    # worker to overlap the temporal collapse (np.median ~79ms/patch) with the
    # GPU step. They only apply with num_workers > 0.
    loader_kwargs: dict[str, Any] = {
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **loader_kwargs,
    )

    criterion = build_dice_ce_loss(
        ignore_index=ignore_index,
        n_classes=resolved_classes,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    amp_enabled = use_amp and resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    # DirPA: built once, applied only in the train epoch (eval passes through).
    dirpa = DirPALogitAdjuster(alpha=dirpa_alpha, tau=dirpa_tau)

    # LR schedule from Tarasiou et al. 2023 (TSViT, §4.1 "Implementation
    # details"): linear warmup 0 -> lr up to `warmup_epochs`, then cosine
    # decay to `lr_min`. The warmup is what stabilizes the transformer (without it,
    # the high LR from step 0 makes the loss diverge to NaN ~epoch 8). Applied
    # per epoch; for DeepLabv3+ (CNN) it also helps but is not critical.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=max(1, warmup_epochs),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=lr_min,
            ),
        ],
        milestones=[max(1, warmup_epochs)],
    )

    # Per-epoch checkpoints: `last.pt` (always) + `best.pt` (best mIoU).
    # They allow resuming after interruption (the L4 VM shut down once).
    resolved_ckpt_dir = (
        Path(ckpt_dir)
        if ckpt_dir is not None
        else Path("checkpoints/segmentation") / mlflow_run_name
    )
    last_ckpt = resolved_ckpt_dir / "last.pt"
    best_ckpt = resolved_ckpt_dir / "best.pt"
    start_epoch = 0
    best_metrics: dict[str, float] = {"miou": -1.0, "f1_macro": 0.0, "pixel_acc": 0.0}
    if resume and last_ckpt.exists():
        start_epoch, best_metrics = _load_checkpoint_seg(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=resolved_device,
        )

    logger.info(
        "train_segmentation_start",
        run_name=mlflow_run_name,
        epochs=epochs,
        batch_size=batch_size,
        device=str(resolved_device),
        num_classes=resolved_classes,
        use_phenology=use_phenology,
        lambda_contrast=lambda_contrast if use_phenology else 0.0,
        amp=amp_enabled,
        n_train=len(train_ds),  # type: ignore[arg-type]
        n_val=len(val_ds),  # type: ignore[arg-type]
        start_epoch=start_epoch,
        ckpt_dir=str(resolved_ckpt_dir),
        patience=patience,
    )

    # Early stopping state and cm of the best epoch (for final artifacts).
    epochs_no_improve = 0
    best_cm = np.zeros((resolved_classes, resolved_classes), dtype=np.int64)

    with track_experiment(
        _EXPERIMENT_NAME,
        run_name=mlflow_run_name,
        tracking_uri=mlflow_uri,
        dvc_path=_PASTIS_DVC_PATH,
    ):
        import mlflow

        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(resolved_device),
                "num_classes": resolved_classes,
                "use_phenology": use_phenology,
                "lambda_contrast": lambda_contrast if use_phenology else 0.0,
                "ignore_index": ignore_index,
                "dice_weight": dice_weight,
                "ce_weight": ce_weight,
                "amp": amp_enabled,
                "optimizer": "AdamW",
            }
        )

        for epoch in range(start_epoch, epochs):
            train_loss = _run_epoch(
                model,
                train_loader,
                criterion=criterion,
                device=resolved_device,
                use_phenology=use_phenology,
                prototypes=proto_tensor,
                lambda_contrast=lambda_contrast,
                ignore_index=ignore_index,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                dirpa=dirpa,
            )
            val_metrics, val_cm = _evaluate_dense(
                model,
                val_loader,
                device=resolved_device,
                num_classes=resolved_classes,
                ignore_index=ignore_index,
                use_phenology=use_phenology,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_miou", val_metrics["miou"], step=epoch)
            mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)
            mlflow.log_metric("val_pixel_acc", val_metrics["pixel_acc"], step=epoch)
            mlflow.log_metric("val_balanced_acc", val_metrics["balanced_acc"], step=epoch)
            mlflow.log_metric("val_cohen_kappa", val_metrics["cohen_kappa"], step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)

            logger.info(
                "train_segmentation_epoch",
                run_name=mlflow_run_name,
                epoch=epoch + 1,
                lr=round(current_lr, 6),
                train_loss=round(train_loss, 4),
                val_miou=round(val_metrics["miou"], 4),
                val_f1_macro=round(val_metrics["f1_macro"], 4),
                val_pixel_acc=round(val_metrics["pixel_acc"], 4),
                val_balanced_acc=round(val_metrics["balanced_acc"], 4),
                val_cohen_kappa=round(val_metrics["cohen_kappa"], 4),
            )

            is_best = val_metrics["miou"] > best_metrics["miou"]
            if is_best:
                best_metrics = dict(val_metrics)
                best_metrics["best_epoch"] = float(epoch + 1)
                best_cm = val_cm.copy()  # cm of the best epoch (for artifacts)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Per-epoch checkpoint: `last.pt` always (for resume), `best.pt`
            # when the validation mIoU improves (for later inference).
            _save_checkpoint_seg(
                last_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_metrics=best_metrics,
            )
            if is_best:
                _save_checkpoint_seg(
                    best_ckpt,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    best_metrics=best_metrics,
                )

            # Advance the LR schedule (warmup -> cosine) at the end of each epoch.
            scheduler.step()

            # Early stopping: cuts off if val_miou does not improve in `patience` epochs
            # (DeepLabv3+ tends to overfit after ~7 epochs). 0 = disabled.
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "early_stopping",
                    run_name=mlflow_run_name,
                    epoch=epoch + 1,
                    best_epoch=int(best_metrics.get("best_epoch", 0)),
                    patience=patience,
                )
                break

        # Initial mIoU -1.0 indicates no epoch ran (should not happen).
        if best_metrics["miou"] < 0.0:
            best_metrics = {"miou": 0.0, "f1_macro": 0.0, "pixel_acc": 0.0}

        mlflow.log_metric("best_val_miou", best_metrics["miou"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["f1_macro"])
        mlflow.log_metric("best_val_pixel_acc", best_metrics["pixel_acc"])
        mlflow.log_metric("best_val_balanced_acc", best_metrics.get("balanced_acc", 0.0))
        mlflow.log_metric("best_val_cohen_kappa", best_metrics.get("cohen_kappa", 0.0))

        # Best epoch artifacts: confusion matrix (PNG figure) +
        # per-class metrics (JSON). For the notebook and the analysis.
        _log_final_artifacts(
            resolved_ckpt_dir,
            best_cm=best_cm,
            best_metrics=best_metrics,
            num_classes=resolved_classes,
        )

        # Upload the best checkpoint to MLflow as an artifact (for reproducible
        # inference from the run, not only from the local disk).
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoint")

    logger.info(
        "train_segmentation_done",
        run_name=mlflow_run_name,
        best_miou=round(best_metrics["miou"], 4),
        best_f1_macro=round(best_metrics["f1_macro"], 4),
        best_pixel_acc=round(best_metrics["pixel_acc"], 4),
    )
    return best_metrics


# ---------------------------------------------------------------------------
# CLI orchestration: builds dataset + model + prototypes and trains.
# The notebook `notebooks/models/5_*` invokes this interface by subprocess so
# that the runs are documented in MLflow without reimplementing logic.
# ---------------------------------------------------------------------------

#: Official PASTIS-R folds for train/val/test (canonical benchmark split).
_DEFAULT_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3)
_DEFAULT_VAL_FOLDS: tuple[int, ...] = (4,)

#: Default MLflow run names according to the model.
_DEFAULT_RUN_NAMES: dict[str, str] = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-v1",
    "tsvit-pheno": "alt-tsvit-pheno-v1",
}


def build_and_train(
    model_kind: str,
    *,
    train_folds: tuple[int, ...] = _DEFAULT_TRAIN_FOLDS,
    val_folds: tuple[int, ...] = _DEFAULT_VAL_FOLDS,
    epochs: int = 30,
    batch_size: int = 4,
    n_timesteps: int = 10,
    target: TargetMode = "semantic18",
    device: str = "auto",
    lr: float = 1e-3,
    lambda_contrast: float = 0.3,
    num_workers: int = 0,
    dim: int = 128,
    depth_temporal: int = 4,
    depth_spatial: int = 4,
    heads: int = 4,
    dim_head: int = 32,
    ckpt_dir: str | Path | None = None,
    resume: bool = True,
    patience: int = 0,
    mlflow_run_name: str | None = None,
    mlflow_uri: str | None = None,
    dirpa_alpha: float = 1.0,
    dirpa_tau: float = 0.0,
) -> dict[str, float]:
    """Builds dataset + model + prototypes and launches the training.

    High-level orchestrator for the CLI: depending on ``model_kind`` it assembles
    the ``PASTISSegmentationDataset`` in the correct mode (2D for DeepLabv3+,
    temporal for TSViT), instantiates the model, loads the phenology prototypes if
    the contrastive branch is requested, and delegates to
    :func:`train_segmentation`.

    Args:
        model_kind: ``"deeplabv3plus"`` (2D CNN), ``"tsvit"`` (temporal without
            phenology) or ``"tsvit-pheno"`` (temporal with contrastive branch).
        train_folds: PASTIS-R training folds.
        val_folds: Validation folds (disjoint from train).
        epochs: Number of epochs.
        batch_size: Batch size.
        n_timesteps: Subsampled T for the temporal models.
        target: ``"semantic18"`` (18 classes) or ``"hcat6"`` (6 HCAT groups).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        lr: AdamW learning rate.
        lambda_contrast: Weight of the contrastive term (tsvit-pheno only).
        dim: TSViT token dimension (L4 default 128; Full-M 192). Ignored for
            DeepLabv3+.
        depth_temporal: TSViT temporal-encoder depth (L4 4; Full-M 6).
        depth_spatial: TSViT spatial-encoder depth (L4 4; Full-M 6).
        heads: TSViT attention heads (L4 4; Full-M 6).
        dim_head: TSViT dimension per head (L4 32; Full-M 64).
        mlflow_run_name: Override of the run name; ``None`` uses the per-model
            default. For the US-038 Full-M retrain pass
            ``"alt-tsvit-fullm-v1"``.
        mlflow_uri: Override of the MLflow tracking URI.

    Returns:
        Metrics of the best validation epoch ``{miou, f1_macro, pixel_acc}``.

    Raises:
        ValueError: if ``model_kind`` is not recognized.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet
    from ml.models.pheno_semantic_branch import PhenoSemanticBranch
    from ml.models.tsvit_wrapper import build_tsvit

    if model_kind not in _DEFAULT_RUN_NAMES:
        raise ValueError(
            f"model_kind not recognized: {model_kind!r}. Options: {sorted(_DEFAULT_RUN_NAMES)}."
        )

    n_classes = 6 if target == "hcat6" else 18
    use_phenology = model_kind == "tsvit-pheno"
    collapse_time: CollapseMode = "median" if model_kind == "deeplabv3plus" else None
    run_name = mlflow_run_name or _DEFAULT_RUN_NAMES[model_kind]

    train_ds = PASTISSegmentationDataset(
        folds=train_folds,
        collapse_time=collapse_time,
        n_timesteps=n_timesteps,
        target=target,
    )
    val_ds = PASTISSegmentationDataset(
        folds=val_folds,
        collapse_time=collapse_time,
        n_timesteps=n_timesteps,
        target=target,
    )

    if model_kind == "deeplabv3plus":
        model: nn.Module = build_deeplabv3plus_mobilenet(in_channels=10, classes=n_classes)
    else:
        # The TSViT capacity (dim/depth/heads/dim_head) flows from the caller so
        # the L4 defaults stay back-compatible (alt-tsvit-v1) and the US-038
        # Full-M run (dim=192, depth 6+6, heads=6, dim_head=64, n_timesteps=64)
        # is selected purely by argument. The SAME capacity must be mirrored in
        # the harness registry entry `tsvit` so the re-score rebuilds an identical
        # topology (R-HARNESS).
        model = build_tsvit(
            num_classes=n_classes,
            n_timesteps=n_timesteps,
            img_size=128,
            in_channels=10,
            dim=dim,
            depth_temporal=depth_temporal,
            depth_spatial=depth_spatial,
            heads=heads,
            dim_head=dim_head,
            semantic_dim=384,
        )

    prototypes = None
    if use_phenology:
        branch = PhenoSemanticBranch(semantic_dim=384)
        prototypes = branch.get_class_prototypes().detach()

    logger.info(
        "build_and_train_start",
        model_kind=model_kind,
        run_name=run_name,
        train_folds=train_folds,
        val_folds=val_folds,
        epochs=epochs,
        use_phenology=use_phenology,
        n_classes=n_classes,
    )

    return train_segmentation(
        model,
        train_ds,
        val_ds,
        mlflow_run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        lr=lr,
        use_phenology=use_phenology,
        prototypes=prototypes,
        lambda_contrast=lambda_contrast,
        num_workers=num_workers,
        ckpt_dir=ckpt_dir,
        resume=resume,
        patience=patience,
        num_classes=n_classes,
        mlflow_uri=mlflow_uri,
        dirpa_alpha=dirpa_alpha,
        dirpa_tau=dirpa_tau,
    )


def _build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover
    p = argparse.ArgumentParser(
        description=(
            "Entrena un segmentador denso PASTIS-R (DeepLabv3+ o TSViT con/sin "
            "rama fenologica) y registra el run en MLflow."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(_DEFAULT_RUN_NAMES),
        help="Arquitectura a entrenar.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--target", choices=("semantic18", "hcat6"), default="semantic18")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-contrast", type=float, default=0.3)
    # TSViT capacity (ignored for DeepLabv3+). Defaults = L4-trimmed (back-compat,
    # run alt-tsvit-v1). US-038 Full-M: --dim 192 --depth-temporal 6
    # --depth-spatial 6 --heads 6 --dim-head 64 --n-timesteps 64.
    p.add_argument(
        "--dim",
        type=int,
        default=128,
        help="TSViT token dim (L4 128; Full-M 192). Ignorado por DeepLabv3+.",
    )
    p.add_argument(
        "--depth-temporal",
        type=int,
        default=4,
        help="Bloques del encoder temporal TSViT (L4 4; Full-M 6).",
    )
    p.add_argument(
        "--depth-spatial",
        type=int,
        default=4,
        help="Bloques del encoder espacial TSViT (L4 4; Full-M 6).",
    )
    p.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Cabezas de atencion TSViT (L4 4; Full-M 6).",
    )
    p.add_argument(
        "--dim-head",
        type=int,
        default=32,
        help="Dimension por cabeza TSViT (L4 32; Full-M 64).",
    )
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help=(
            "Directorio de checkpoints. Default checkpoints/segmentation/"
            "<run-name>. Guarda last.pt (resume) + best.pt (inferencia) por epoch."
        ),
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Early stopping: corta si val_miou no mejora en N epochs. "
            "0 = desactivado. DeepLabv3+ sobreajusta tras ~7 epochs."
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora last.pt y entrena desde cero (por defecto reanuda).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Workers del DataLoader. El colapso temporal (np.median ~79ms/patch) "
            "es CPU-bound; subir esto satura la GPU. Optimo ~3/4 de los cores "
            "fisicos (ej. 12 en un CPU de 16 cores). 0 = serial (CI/debug)."
        ),
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--mlflow-uri", default=None)
    p.add_argument(
        "--train-folds",
        default="1,2,3",
        help="Folds de entrenamiento separados por coma.",
    )
    p.add_argument("--val-folds", default="4", help="Folds de validacion separados por coma.")
    p.add_argument(
        "--dirpa-alpha",
        type=float,
        default=1.0,
        help=(
            "DirPA: concentracion de la Dirichlet simetrica. <1 muestrea priors "
            "sesgados (long-tail), >1 cerca de uniforme. Solo activo si dirpa-tau>0."
        ),
    )
    p.add_argument(
        "--dirpa-tau",
        type=float,
        default=0.0,
        help=(
            "DirPA: escala del ajuste de logits z'=z+tau*log(pi~). 0 desactiva "
            "(entrenamiento normal). >0 robustece a las minoritarias (US-079b)."
        ),
    )
    return p


def main_legacy(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point. Invoked by the ``5_*`` notebook via subprocess."""
    args = _build_arg_parser().parse_args(argv)
    train_folds = tuple(int(x) for x in args.train_folds.split(","))
    val_folds = tuple(int(x) for x in args.val_folds.split(","))
    metrics = build_and_train(
        args.model,
        train_folds=train_folds,
        val_folds=val_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_timesteps=args.n_timesteps,
        target=args.target,
        device=args.device,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        num_workers=args.num_workers,
        dim=args.dim,
        depth_temporal=args.depth_temporal,
        depth_spatial=args.depth_spatial,
        heads=args.heads,
        dim_head=args.dim_head,
        ckpt_dir=args.ckpt_dir,
        resume=not args.no_resume,
        patience=args.patience,
        mlflow_run_name=args.run_name,
        mlflow_uri=args.mlflow_uri,
        dirpa_alpha=args.dirpa_alpha,
        dirpa_tau=args.dirpa_tau,
    )
    logger.info("cli_done", **{k: round(v, 4) for k, v in metrics.items()})
    return 0


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Modelo: 'unet' (#1) o 'anysat' (#6).")] = "unet",
    epochs: Annotated[int, typer.Option(help="Numero de epocas.")] = 30,
    batch_size: Annotated[int, typer.Option(help="Tamano de batch.")] = 8,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW.")] = 1e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay AdamW.")] = 1e-4,
    target_size: Annotated[int, typer.Option(help="Resolucion espacial objetivo.")] = 256,
    train_folds: Annotated[str, typer.Option(help="Folds de train (coma).")] = "1,2,3",
    val_folds: Annotated[str, typer.Option(help="Folds de validacion (coma).")] = "4",
    subset: Annotated[int, typer.Option(help="Limita patches por split (0 = todos).")] = 0,
    device: Annotated[str, typer.Option(help="cpu, cuda o auto.")] = "auto",
    num_workers: Annotated[int, typer.Option(help="Workers del DataLoader.")] = 0,
    root: Annotated[Path, typer.Option(help="Raiz PASTIS-R.")] = _DEFAULT_ROOT,
    output_dir: Annotated[Path, typer.Option(help="Destino de checkpoints.")] = _DEFAULT_OUTPUT,
    comparison_path: Annotated[
        Path, typer.Option(help="Parquet comparativo (lo consume el integrador).")
    ] = _DEFAULT_COMPARISON,
    mlflow_uri: Annotated[str, typer.Option(help="Tracking URI MLflow (vacio = auto).")] = "",
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Reanudar desde checkpoint si existe.")
    ] = True,
    checkpoint_every: Annotated[int, typer.Option(help="Guardar checkpoint cada N epocas.")] = 1,
) -> None:
    """CLI wrapper of :func:`run_training` (see its docstring for the arguments)."""
    try:
        result = run_training(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            target_size=target_size,
            train_folds=train_folds,
            val_folds=val_folds,
            subset=subset,
            device=device,
            num_workers=num_workers,
            root=root,
            output_dir=output_dir,
            comparison_path=comparison_path,
            mlflow_uri=mlflow_uri,
            resume=resume,
            checkpoint_every=checkpoint_every,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("segmentation_train_skipped", reason=str(exc))
        raise typer.Exit(code=0) from exc

    typer.echo(
        f"[{result['model']}] mIoU={result['miou']:.4f} "
        f"F1-macro={result['f1_macro']:.4f} pixacc={result['pixel_accuracy']:.4f} "
        f"({result['train_time_s']:.1f}s) -> {result['checkpoint_path']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI dispatcher
    # Two CLIs coexist in this module: the UNet/AnySat Typer (Aaron) and the
    # DeepLab/TSViT argparse (us-025). Routing is by the --model value so
    # that `python -m ml.train.train_segmentation --model X` works for both.
    _US025_MODELS = {"deeplabv3plus", "tsvit", "tsvit-pheno"}
    _argv = sys.argv[1:]
    _model = None
    for _i, _a in enumerate(_argv):
        if _a == "--model" and _i + 1 < len(_argv):
            _model = _argv[_i + 1]
            break
        if _a.startswith("--model="):
            _model = _a.split("=", 1)[1]
            break
    if _model in _US025_MODELS:
        sys.exit(main_legacy(_argv))
    sys.exit(app())
