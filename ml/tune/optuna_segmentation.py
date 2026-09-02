"""Fine-tuning (Optuna) by warm-start of an already-trained temporal segmenter.

Unlike :mod:`ml.tune.anysat_head_tuning` (frozen encoder, only a Conv 1x1 head
is tuned over cached features), the end-to-end temporal models of team us-025
(``tsvit`` / ``tsvit-pheno``) have no frozen encoder: every step touches all the
weights. Training 30 trials from scratch (30 epochs each) would cost ~16 h of
L4, unfeasible for the budget.

**Warm-start** strategy: start from the best already-trained checkpoint
(``best.pt``, e.g. tsvit-pheno mIoU 0.6253 @ epoch 28) and each trial does a
*short fine-tuning* of a few epochs varying ``lr`` / ``weight_decay`` /
``batch_size``. Since it starts from a good model, 3 epochs are meaningful (it
refines, it does not learn from scratch), the trial ranking is informative and
the full study of >=30 trials runs in ~2-3 h. The ``MedianPruner`` aborts the
worst trials in the 2nd epoch.

Reuses the training loop (:func:`ml.train.train_segmentation._run_epoch`) and
the dense evaluation (:func:`ml.train.train_segmentation._evaluate_dense`) of
the main pipeline, so that the ``miou`` Optuna optimizes is exactly the one the
final model reports (separation of concerns, CLAUDE.md rule 8). It does not
modify ``train_segmentation`` (shared team code).

The study is persisted to a parquet
``reports/segmentation/metrics/tuning_<model>.parquet`` with one row per trial
(value + params + state), consumed by the fine-tuning cell of
``Avance4.Equipo17.ipynb``.

Usage (on the L4 VM)::

    poetry run python -m ml.tune.optuna_segmentation \\
        --model tsvit-pheno \\
        --init-ckpt checkpoints/segmentation/tsvit-pheno-v1/best.pt \\
        --n-trials 30 --epochs 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl
import structlog
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm
from ml.models.deeplabv3plus import build_dice_ce_loss
from ml.train.train_segmentation import (
    _DEFAULT_TRAIN_FOLDS,
    _DEFAULT_VAL_FOLDS,
    _evaluate_dense,
    _resolve_device,
    _run_epoch,
)

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from ml.data.pastis_seg_dataset import TargetMode

logger = structlog.get_logger(__name__)

__all__ = ["build_objective", "build_objective_utae", "run_study"]

#: Supported models. tsvit/tsvit-pheno (us-025) reuse the main loop;
#: utae (Isaac) uses its own dataset+loop (forward with batch_positions).
_TSVIT_MODELS = ("tsvit", "tsvit-pheno")
_SUPPORTED = (*_TSVIT_MODELS, "utae")

#: Search space (same set as Aaron's AnySat tuning: lr +
#: weight_decay + batch_size), bounded for fine-tuning from a good model:
#: low lr (do not destroy the weights), discrete weight_decay and batch.
_LR_RANGE = (1e-5, 5e-4)
_WD_RANGE = (1e-6, 1e-2)
_BATCH_CHOICES = (4, 8, 16)


def _build_model_and_data(
    model_kind: str,
    *,
    n_timesteps: int,
    target: TargetMode,
    train_folds: tuple[int, ...],
    val_folds: tuple[int, ...],
) -> tuple[nn.Module, Any, Any, torch.Tensor | None, int]:
    """Build TSViT model + train/val datasets + prototypes (if pheno).

    Returns:
        Tuple ``(model, train_ds, val_ds, prototypes|None, num_classes)``.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.models.pheno_semantic_branch import PhenoSemanticBranch
    from ml.models.tsvit_wrapper import build_tsvit

    num_classes = 6 if target == "hcat6" else 18
    use_phenology = model_kind == "tsvit-pheno"

    train_ds = PASTISSegmentationDataset(
        folds=train_folds, collapse_time=None, n_timesteps=n_timesteps, target=target
    )
    val_ds = PASTISSegmentationDataset(
        folds=val_folds, collapse_time=None, n_timesteps=n_timesteps, target=target
    )
    model = build_tsvit(
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        img_size=128,
        in_channels=10,
        semantic_dim=384,
    )
    prototypes = None
    if use_phenology:
        prototypes = PhenoSemanticBranch(semantic_dim=384).get_class_prototypes().detach()
    return model, train_ds, val_ds, prototypes, num_classes


def _load_init_weights(model: nn.Module, init_ckpt: Path, device: torch.device) -> dict[str, float]:
    """Load the base checkpoint weights (warm-start) and return its metrics.

    Args:
        model: Freshly built model (random weights).
        init_ckpt: Path to the ``best.pt`` of the already-trained model.
        device: Target device.

    Returns:
        ``best_metrics`` of the checkpoint (mIoU/F1/pixel_acc of the base model).
    """
    ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    base: dict[str, float] = ckpt.get("best_metrics", {})
    logger.info("warmstart_loaded", ckpt=str(init_ckpt), base_miou=base.get("miou"))
    return base


def build_objective(
    model_kind: str,
    *,
    init_ckpt: Path,
    epochs: int,
    n_timesteps: int,
    target: TargetMode,
    device: str,
    train_folds: tuple[int, ...],
    val_folds: tuple[int, ...],
    num_workers: int,
    lambda_contrast: float,
) -> tuple[Callable[[optuna.Trial], float], dict[str, float]]:
    """Build the Optuna ``objective`` for warm-start fine-tuning.

    The dataset and the evaluation are fixed across trials; what varies per
    trial is ``lr`` / ``weight_decay`` / ``batch_size``. Each trial reloads the
    base weights (clean warm-start, no contamination between trials), trains
    ``epochs`` epochs and reports the best validation ``miou``. Supports
    per-epoch pruning.

    Returns:
        Callable ``objective(trial) -> float`` (mIoU to maximize).
    """
    dev = _resolve_device(device)
    use_phenology = model_kind == "tsvit-pheno"
    model, train_ds, val_ds, prototypes, num_classes = _build_model_and_data(
        model_kind,
        n_timesteps=n_timesteps,
        target=target,
        train_folds=train_folds,
        val_folds=val_folds,
    )
    model = model.to(dev)
    if prototypes is not None:
        prototypes = prototypes.to(dev)
    # Snapshot of the base weights on CPU to reload per trial without re-reading disk.
    base_metrics = _load_init_weights(model, init_ckpt, dev)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    criterion = build_dice_ce_loss(ignore_index=255, n_classes=num_classes).to(dev)
    amp_enabled = dev.type == "cuda"

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *_LR_RANGE, log=True)
        weight_decay = trial.suggest_float("weight_decay", *_WD_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", list(_BATCH_CHOICES))

        # Warm-start: reload the base weights (each trial starts from the same point).
        model.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=amp_enabled,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=amp_enabled,
        )

        best_miou = -1.0
        for epoch in range(epochs):
            _run_epoch(
                model,
                train_loader,
                criterion=criterion,
                device=dev,
                use_phenology=use_phenology,
                prototypes=prototypes,
                lambda_contrast=lambda_contrast,
                ignore_index=255,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=amp_enabled,
            )
            metrics, _ = _evaluate_dense(
                model,
                val_loader,
                device=dev,
                num_classes=num_classes,
                ignore_index=255,
                use_phenology=use_phenology,
            )
            miou = float(metrics["miou"])
            best_miou = max(best_miou, miou)
            trial.report(miou, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_miou

    return objective, base_metrics


# ---------------------------------------------------------------------------
# U-TAE (Isaac): multi-temporal dataset with day-of-year + its own objective.
# U-TAE's forward is `model(imgs, batch_positions)` with imgs (B,T,C,H,W) and
# positions (B,T); it differs from the main loop (model(x)), which is why it has
# its own dataset and mini-loop, a faithful replica of notebooks/segmentation/04j.
# ---------------------------------------------------------------------------

#: Per-band S2 normalization used by Isaac (PASTIS scales ~0-10000).
_UTAE_S2_MEAN = (1158.0, 1244.7, 1416.3, 1374.8, 1619.0, 2075.1, 2263.1, 2311.0, 2108.6, 817.4)
_UTAE_S2_STD = (671.7, 698.1, 761.3, 830.8, 795.3, 907.5, 981.1, 993.7, 882.0, 504.2)


class PASTISMultiTempDataset(Dataset):
    """Multi-temporal PASTIS-R dataset for U-TAE (faithful port of notebook 04j).

    Returns T equispaced timesteps per patch, with day-of-year proxy.
    ``__getitem__`` -> ``(imgs (T,C,H,W), labels (H,W), positions (T,))``.
    """

    def __init__(
        self,
        pastis_root: Path,
        fold_ids: list[int],
        *,
        img_size: int = 128,
        num_bands: int = 10,
        t_steps: int = 10,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(pastis_root)
        self.img_size = img_size
        self.num_bands = num_bands
        self.t_steps = t_steps
        self.augment = augment
        meta_path = self.root / "metadata.geojson"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.geojson not found in {self.root}")
        with open(meta_path) as f:
            meta = json.load(f)
        self.patch_ids = [
            feat["properties"]["ID_PATCH"]
            for feat in meta["features"]
            if feat["properties"]["Fold"] in fold_ids
        ]

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _load_image(self, pid: int) -> tuple[np.ndarray, np.ndarray]:
        s2 = np.load(self.root / "DATA_S2" / f"S2_{pid}.npy")  # (T_full, C, H, W)
        t_full = s2.shape[0]
        indices = np.linspace(0, t_full - 1, self.t_steps, dtype=int)
        s2 = s2[indices][:, : self.num_bands]
        mean = np.array(_UTAE_S2_MEAN[: self.num_bands], dtype=np.float32)[None, :, None, None]
        std = np.array(_UTAE_S2_STD[: self.num_bands], dtype=np.float32)[None, :, None, None]
        s2 = (s2.astype(np.float32) - mean) / (std + 1e-6)
        positions = (indices / max(t_full - 1, 1) * 364).astype(np.int64)
        return s2, positions

    def _load_mask(self, pid: int) -> np.ndarray:
        mask: np.ndarray = np.load(self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy")
        if mask.ndim == 3:
            mask = mask[0]
        return mask.astype(np.int64)

    def _resize(self, imgs: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t_imgs = torch.from_numpy(imgs)
        t_mask = torch.from_numpy(mask).unsqueeze(0)
        resized = [
            TF.resize(
                t_imgs[ti],
                [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            for ti in range(t_imgs.shape[0])
        ]
        t_imgs = torch.stack(resized, dim=0)
        t_mask = TF.resize(
            t_mask,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        return t_imgs, t_mask.squeeze(0)

    def _augment(self, imgs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:  # noqa: S311 - data augmentation, not cryptography
            imgs = torch.flip(imgs, dims=[-1])
            mask = torch.flip(mask.unsqueeze(0), dims=[-1]).squeeze(0)
        if random.random() > 0.5:  # noqa: S311 - data augmentation, not cryptography
            imgs = torch.flip(imgs, dims=[-2])
            mask = torch.flip(mask.unsqueeze(0), dims=[-2]).squeeze(0)
        angle = random.choice([0, 90, 180, 270])  # noqa: S311 - augmentation
        if angle:
            imgs = torch.stack([TF.rotate(imgs[t], angle) for t in range(imgs.shape[0])])
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
        return imgs, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = self.patch_ids[idx]
        imgs_np, positions = self._load_image(pid)
        mask_np = self._load_mask(pid)
        imgs, mask = self._resize(imgs_np, mask_np)
        if self.augment:
            imgs, mask = self._augment(imgs, mask)
        return {
            "pixel_values": imgs,
            "labels": mask,
            "positions": torch.from_numpy(positions),
        }


def build_objective_utae(
    *,
    init_ckpt: Path,
    pastis_root: Path,
    epochs: int,
    n_timesteps: int,
    num_classes: int,
    device: str,
    train_folds: tuple[int, ...],
    val_folds: tuple[int, ...],
    num_workers: int,
    ignore_index: int = 19,
) -> tuple[Callable[[optuna.Trial], float], dict[str, float]]:
    """Build the Optuna ``objective`` for warm-start of U-TAE.

    Replicates Isaac's setup (CrossEntropy, clip_grad 5.0, multi-temporal
    dataset with day-of-year) and starts from the base checkpoint; each trial
    varies lr/weight_decay/batch_size and trains ``epochs`` short epochs.

    Returns:
        Tuple ``(objective, base_metrics)``.
    """
    from ml.models.utae import build_utae

    dev = _resolve_device(device)
    model = build_utae(num_classes=num_classes, input_dim=10).to(dev)
    ckpt = torch.load(init_ckpt, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    base_metrics = {"miou": float(ckpt.get("val_miou", float("nan")))}
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    logger.info("warmstart_loaded_utae", ckpt=str(init_ckpt), base_miou=base_metrics["miou"])

    train_ds = PASTISMultiTempDataset(
        pastis_root, list(train_folds), t_steps=n_timesteps, augment=True
    )
    val_ds = PASTISMultiTempDataset(
        pastis_root, list(val_folds), t_steps=n_timesteps, augment=False
    )
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    amp_enabled = dev.type == "cuda"

    def _eval(loader: DataLoader) -> float:
        model.eval()
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for batch in loader:
                imgs = batch["pixel_values"].to(dev, non_blocking=True).float()
                pos = batch["positions"].to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True)
                preds = model(imgs, pos).argmax(dim=1)
                cm += dense_confusion_matrix(
                    preds, labels, n_classes=num_classes, ignore_index=ignore_index
                )
        return float(dense_metrics_from_cm(cm)["miou"])

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *_LR_RANGE, log=True)
        weight_decay = trial.suggest_float("weight_decay", *_WD_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", list(_BATCH_CHOICES))

        model.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=amp_enabled,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=amp_enabled,
        )

        best_miou = -1.0
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                imgs = batch["pixel_values"].to(dev, non_blocking=True).float()
                pos = batch["positions"].to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(imgs, pos), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            miou = _eval(val_loader)
            best_miou = max(best_miou, miou)
            trial.report(miou, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_miou

    return objective, base_metrics


def run_study(
    model_kind: str,
    *,
    init_ckpt: Path,
    n_trials: int = 30,
    epochs: int = 3,
    n_timesteps: int = 10,
    target: TargetMode = "semantic18",
    device: str = "auto",
    train_folds: tuple[int, ...] = _DEFAULT_TRAIN_FOLDS,
    val_folds: tuple[int, ...] = _DEFAULT_VAL_FOLDS,
    num_workers: int = 0,
    lambda_contrast: float = 0.3,
    pastis_root: Path = Path("data/PASTIS-R"),
    utae_num_classes: int = 20,
    utae_ignore_index: int = 19,
    out_dir: Path = Path("reports/segmentation/metrics"),
) -> Path:
    """Run the warm-start Optuna study and persist the trials to parquet.

    Args:
        model_kind: ``"tsvit"``, ``"tsvit-pheno"`` or ``"utae"``.
        init_ckpt: ``best.pt`` (tsvit) / ``best_model.pt`` (utae) of the base model.
        n_trials: Number of trials (fine-tuning rubric: >=30).
        epochs: Fine-tuning epochs per trial (short: warm-start).
        device: ``auto`` / ``cuda`` / ``cpu``.
        pastis_root: PASTIS-R root (utae only; uses metadata.geojson/DATA_S2).
        utae_num_classes: utae output classes (20 in Isaac's ckpt).
        utae_ignore_index: Ignored label in utae (void).
        out_dir: Output folder for the trials parquet.

    Returns:
        Path of the written ``tuning_<model>.parquet`` parquet.

    Raises:
        ValueError: if ``model_kind`` is not supported.
    """
    if model_kind not in _SUPPORTED:
        raise ValueError(f"model_kind {model_kind!r} not supported; use {_SUPPORTED}.")

    if model_kind == "utae":
        objective, base_metrics = build_objective_utae(
            init_ckpt=init_ckpt,
            pastis_root=pastis_root,
            epochs=epochs,
            n_timesteps=n_timesteps,
            num_classes=utae_num_classes,
            device=device,
            train_folds=train_folds,
            val_folds=val_folds,
            num_workers=num_workers,
            ignore_index=utae_ignore_index,
        )
    else:
        objective, base_metrics = build_objective(
            model_kind,
            init_ckpt=init_ckpt,
            epochs=epochs,
            n_timesteps=n_timesteps,
            target=target,
            device=device,
            train_folds=train_folds,
            val_folds=val_folds,
            num_workers=num_workers,
            lambda_contrast=lambda_contrast,
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    base_miou = float(base_metrics.get("miou", float("nan")))
    logger.info(
        "study_done",
        model=model_kind,
        n_trials=len(study.trials),
        best_value=study.best_value,
        base_miou=base_miou,
        elapsed_s=round(elapsed, 1),
    )

    rows = [
        {
            "model": model_kind,
            "trial": t.number,
            "value": t.value,
            "state": t.state.name,
            "lr": t.params.get("lr"),
            "weight_decay": t.params.get("weight_decay"),
            "batch_size": t.params.get("batch_size"),
            "base_miou": base_miou,
            "epochs_per_trial": epochs,
        }
        for t in study.trials
    ]
    df = pl.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tuning_{model_kind}.parquet"
    df.write_parquet(out_path)
    logger.info("tuning_parquet_written", path=str(out_path), rows=df.height)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=_SUPPORTED, default="tsvit-pheno")
    p.add_argument("--init-ckpt", type=Path, required=True, help="best.pt del modelo base.")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--target", choices=("semantic18", "hcat6"), default="semantic18")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lambda-contrast", type=float, default=0.3)
    p.add_argument(
        "--pastis-root",
        type=Path,
        default=Path("data/PASTIS-R"),
        help="Raiz PASTIS-R (solo utae).",
    )
    p.add_argument("--utae-num-classes", type=int, default=20)
    p.add_argument("--utae-ignore-index", type=int, default=19)
    args = p.parse_args(argv)

    out = run_study(
        args.model,
        init_ckpt=args.init_ckpt,
        n_trials=args.n_trials,
        epochs=args.epochs,
        n_timesteps=args.n_timesteps,
        target=args.target,
        device=args.device,
        num_workers=args.num_workers,
        lambda_contrast=args.lambda_contrast,
        pastis_root=args.pastis_root,
        utae_num_classes=args.utae_num_classes,
        utae_ignore_index=args.utae_ignore_index,
    )
    logger.info("optuna_study_persisted", path=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
