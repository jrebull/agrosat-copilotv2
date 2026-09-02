"""Homologate the 4 evaluation figures of the Avance 4 models.

The integrator ``notebooks/segmentation/Avance4.Equipo17.ipynb`` shows four
figures per model: training curves, IoU per class, confusion matrix and
RGB/ground-truth/prediction examples. Not all team members exported the
four, so here we **regenerate the missing ones** from the raw data that
does exist:

- **Training curves** (``curves_<model>.png``): read from the local MLflow
  server (Docker, experiment ``agrosat-segmentation``) which stores
  ``train_loss`` and ``val_miou`` per epoch for the us-025 models (DeepLabv3+,
  TSViT). :func:`curves_from_mlflow`.
- **Confusion matrix** (``confusion_<model>.png``) and **IoU per class**
  (``per_class_iou_<model>.png``): require re-evaluating the checkpoint over
  the validation fold (inference). :func:`confusion_and_per_class_from_ckpt`.

Each function writes the PNG with the name consumed by ``_find_fig`` of the
integrator, so ``show_model_figs`` picks them up without touching the notebook.

Permanent operational tool (reproducible), not a smoke/debug script.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

logger = structlog.get_logger(__name__)

__all__ = [
    "confusion_from_cm",
    "curves_from_mlflow",
    "optuna_convergence_figure",
    "per_class_iou_figure",
    "regen_deeplab_tsvit",
    "regen_isaac_model",
    "samples_grid",
]

#: Integrator model mapping -> MLflow run (experiment 7) with its history.
#: Each run logs ``train_loss`` and ``val_miou`` per epoch (step = epoch).
_MLFLOW_RUNS = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-pheno-v1",  # the phenological variant is the top-2 candidate
}


def _fetch_epoch_history(
    run_name: str, *, experiment: str, tracking_uri: str
) -> dict[str, np.ndarray]:
    """Read the per-epoch metrics of an MLflow run by name.

    Args:
        run_name: Run name (``mlflow.runName``).
        experiment: MLflow experiment name.
        tracking_uri: MLflow server URI.

    Returns:
        Dict with the available per-epoch series: ``train_loss``, ``val_miou``,
        ``val_f1_macro`` (whichever the run logged; absent -> empty array).

    Raises:
        RuntimeError: if the run is not found or has no ``train_loss``.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"MLflow experiment {experiment!r} does not exist in {tracking_uri}.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=5,
    )
    finished = [r for r in runs if r.info.status == "FINISHED"]
    if not finished:
        raise RuntimeError(f"Run {run_name!r} (FINISHED) not found in {experiment!r}.")
    run_id = finished[0].info.run_id

    def _series(metric: str) -> np.ndarray:
        hist = sorted(client.get_metric_history(run_id, metric), key=lambda m: m.step)
        return np.asarray([m.value for m in hist], dtype=np.float64)

    series = {
        "train_loss": _series("train_loss"),
        "val_miou": _series("val_miou"),
        "val_f1_macro": _series("val_f1_macro"),
    }
    if series["train_loss"].size == 0:
        raise RuntimeError(f"Run {run_name!r} without per-epoch `train_loss` history.")
    return series


def curves_from_mlflow(
    model: str,
    *,
    out_dir: Path = Path("reports/segmentation/figures"),
    experiment: str = "agrosat-segmentation",
    tracking_uri: str = "http://localhost:5010",
    run_name: str | None = None,
) -> Path:
    """Generate the training curves figure of a model from MLflow.

    Layout 1x3 (Loss | mIoU | F1-Macro) reading from the local MLflow server
    the per-epoch series the run logged (train_loss, val_miou, val_f1_macro).
    The mIoU and F1 panels mark the best epoch (the checkpoint's). Writes
    ``curves_<model>.png`` in ``out_dir`` with the name consumed by the integrator.

    Args:
        model: Model key in the integrator (``deeplabv3plus`` / ``tsvit``).
        out_dir: Output folder for the figures.
        experiment: MLflow experiment.
        tracking_uri: MLflow server URI (local Docker on :5010).
        run_name: Run name override; if ``None`` uses ``_MLFLOW_RUNS``.

    Returns:
        Path of the written PNG.

    Raises:
        KeyError: if ``model`` is not in the mapping and ``run_name`` is not passed.
    """
    name = run_name or _MLFLOW_RUNS[model]
    h = _fetch_epoch_history(name, experiment=experiment, tracking_uri=tracking_uri)
    train_loss, val_miou, val_f1 = h["train_loss"], h["val_miou"], h["val_f1_macro"]

    # Layout 1x3 (Loss | mIoU | F1-macro), aligned with the team style.
    # Our us-025 runs logged train_loss + val_miou + val_f1_macro per
    # epoch (not train_miou/val_loss), so each panel plots the series
    # actually recorded, without inventing curves.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        np.arange(train_loss.size), train_loss, color="#2b6cb0", marker="o", ms=3, label="Train"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    def _val_panel(ax: Axes, series: np.ndarray, title: str, ylabel: str) -> None:
        if series.size == 0:
            ax.text(0.5, 0.5, "no registrado", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        x = np.arange(series.size)
        ax.plot(x, series, color="#dd6b20", marker="s", ms=3, label="Val")
        best = int(np.argmax(series))
        ax.axvline(best, color="#dd6b20", ls="--", lw=1, alpha=0.7)
        ax.scatter([best], [series[best]], color="#dd6b20", s=60, zorder=5, edgecolor="white")
        ax.annotate(
            f"best ep {best}\n{series[best]:.4f}",
            xy=(best, series[best]),
            xytext=(-6, -26),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#9c4221",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    _val_panel(axes[1], val_miou, "mIoU", "mIoU (val)")
    _val_panel(axes[2], val_f1, "F1-Macro", "F1-macro (val)")

    fig.suptitle(f"Curvas de entrenamiento - {model}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"curves_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "curves_written", model=model, run=name, epochs=int(train_loss.size), path=str(out_path)
    )
    return out_path


def per_class_iou_figure(
    per_class_iou: dict[int, float] | list[float] | np.ndarray,
    model: str,
    *,
    class_names: dict[int, str] | None = None,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Generate the per-class IoU barplot and write it as ``per_class_iou_<model>.png``.

    Args:
        per_class_iou: IoU per class (dict ``{id: iou}``, list or array).
        model: Model key in the integrator.
        class_names: Map ``{id: name}`` to label the axis; ``None`` uses ``C{id}``.
        out_dir: Output folder.

    Returns:
        Path of the written PNG.
    """
    if isinstance(per_class_iou, dict):
        ids = sorted(per_class_iou)
        ious = [per_class_iou[i] for i in ids]
    else:
        ious = list(per_class_iou)
        ids = list(range(len(ious)))
    labels = [(class_names.get(i, f"C{i}") if class_names else f"C{i}") for i in ids]

    fig, ax = plt.subplots(figsize=(8, max(4, len(ids) * 0.32)))
    ax.barh(labels[::-1], list(ious)[::-1], color="#2b6cb0")
    ax.set_xlabel("IoU")
    ax.set_xlim(0, 1)
    ax.set_title(f"IoU por clase - {model}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_class_iou_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("per_class_iou_written", model=model, n_classes=len(ids), path=str(out_path))
    return out_path


def confusion_from_cm(
    cm: np.ndarray,
    model: str,
    *,
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Generate the confusion matrix (row-normalized) from an accumulated cm.

    Reuses :func:`ml.eval.metrics.confusion_matrix_figure` (same visual style
    as the rest of the project). Writes ``confusion_<model>.png``.

    Args:
        cm: Accumulated confusion matrix ``(K, K)`` (rows=truth, cols=pred).
        model: Model key in the integrator.
        class_names: Map ``{id: name}`` to label the axes.
        ignore_index: If given, discards that row/column from the plot.
        out_dir: Output folder.

    Returns:
        Path of the written PNG.
    """
    keep = np.ones(cm.shape[0], dtype=bool)
    if ignore_index is not None and 0 <= ignore_index < cm.shape[0]:
        keep[ignore_index] = False
    cm_k = cm[np.ix_(keep, keep)].astype(np.float64)
    ids = [i for i in range(cm.shape[0]) if keep[i]]
    labels = [(class_names.get(i, f"C{i}") if class_names else f"C{i}") for i in ids]

    # Row-wise normalization (per-class recall); rows without support stay at 0.
    row_sum = cm_k.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_k, row_sum, out=np.zeros_like(cm_k), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.5), max(5, len(ids) * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdad")
    ax.set_title(f"Matriz de confusión - {model}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall por clase")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("confusion_written", model=model, k=cm_k.shape[0], path=str(out_path))
    return out_path


def regen_deeplab_tsvit(
    model: Literal["deeplabv3plus", "tsvit"],
    *,
    checkpoint: Path,
    num_classes: int = 18,
    ignore_index: int = 255,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> tuple[Path, Path]:
    """Regenerate per_class_iou + confusion of a us-025 model (deeplab/tsvit).

    Reuses the flow of the 5* notebooks: ``load_segmentation_model`` +
    ``evaluate_checkpoint`` over the validation fold.

    Args:
        model: ``"deeplabv3plus"`` or ``"tsvit"`` (integrator key).
        checkpoint: Path to ``best.pt``.
        num_classes: 18 (semantic) or 6 (HCAT).
        ignore_index: Ignored label (255 in us-025).
        val_folds: Validation folds.
        n_timesteps: T for the temporal model.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Output folder.

    Returns:
        Tuple ``(path_per_class, path_confusion)``.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.segmentation_inference import evaluate_checkpoint, load_segmentation_model
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    model_kind: Literal["deeplabv3plus", "tsvit-pheno"] = (
        "tsvit-pheno" if model == "tsvit" else model
    )
    collapse: Literal["median"] | None = "median" if model == "deeplabv3plus" else None
    ds = PASTISSegmentationDataset(
        folds=val_folds, collapse_time=collapse, n_timesteps=n_timesteps, target="semantic18"
    )
    net = load_segmentation_model(
        checkpoint,
        model_kind=model_kind,
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        device=device,
    )
    metrics, cm = evaluate_checkpoint(
        net, ds, model_kind=model_kind, num_classes=num_classes, ignore_index=ignore_index
    )
    p_iou = per_class_iou_figure(
        cast("list[float]", metrics["per_class_iou"]),
        model,
        class_names=PASTIS_R_CLASSES,
        out_dir=out_dir,
    )
    p_cm = confusion_from_cm(cm, model, class_names=PASTIS_R_CLASSES, out_dir=out_dir)
    return p_iou, p_cm


def regen_isaac_model(
    model: str,
    *,
    checkpoint: Path,
    pastis_root: Path = Path("data/PASTIS-R"),
    num_classes: int = 20,
    ignore_index: int = 19,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Regenerate the confusion matrix of an Isaac model (utae / segformer).

    Loads the checkpoint with the correct architecture (ported U-TAE or
    SegFormer HF), evaluates the validation fold with the multi-temporal
    dataset (utae) or 2D (segformer) and writes ``confusion_<model>.png``.

    Args:
        model: ``"utae"`` or ``"segformer"``.
        checkpoint: ``best_model.pt`` (utae) or the model folder/file.
        pastis_root: PASTIS-R root.
        num_classes: 20 (Isaac's convention).
        ignore_index: 19 (void).
        val_folds: Validation folds.
        n_timesteps: T for utae.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Output folder.

    Returns:
        Path of the written confusion PNG.

    Raises:
        ValueError: if ``model`` is neither ``utae`` nor ``segformer``.
    """
    import torch

    from ml.eval.metrics import dense_confusion_matrix
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    dev = torch.device(
        "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    if model == "utae":
        from ml.models.utae import build_utae
        from ml.tune.optuna_segmentation import PASTISMultiTempDataset

        net: torch.nn.Module = build_utae(num_classes=num_classes, input_dim=10).to(dev).eval()
        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        net.load_state_dict(ck.get("model_state_dict", ck))
        ds = PASTISMultiTempDataset(
            pastis_root, list(val_folds), t_steps=n_timesteps, augment=False
        )
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                x = s["pixel_values"].unsqueeze(0).to(dev).float()
                pos = s["positions"].unsqueeze(0).to(dev)
                pred = net(x, pos).argmax(dim=1).squeeze(0).cpu().numpy()
                cm += dense_confusion_matrix(
                    pred, s["labels"].numpy(), n_classes=num_classes, ignore_index=ignore_index
                )
    elif model == "segformer":
        import json

        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from transformers import SegformerForSemanticSegmentation

        # Isaac's SegFormer (notebook 04i): 3 RGB bands (temporal median,
        # first 3 S2 bands normalized with S2_MEAN/STD), img 256px.
        seg_mean = np.array([1158.0, 1244.7, 1416.3], dtype=np.float32)[:, None, None]
        seg_std = np.array([671.7, 698.1, 761.3], dtype=np.float32)[:, None, None]
        seg_size = 256

        hf_model = cast(
            "torch.nn.Module",
            SegformerForSemanticSegmentation.from_pretrained(
                str(Path(checkpoint).parent / "hf_model")
            ),
        )
        net = hf_model.to(dev).eval()
        root = Path(pastis_root)
        meta = json.loads((root / "metadata.geojson").read_text())
        pids = [
            f["properties"]["ID_PATCH"]
            for f in meta["features"]
            if f["properties"]["Fold"] in val_folds
        ]
        import torch

        with torch.no_grad():
            for pid in pids:
                s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")  # (T, C, H, W)
                img = np.median(s2, axis=0)[:3].astype(np.float32)  # RGB composite
                img = (img - seg_mean) / (seg_std + 1e-6)
                mask = np.load(root / "ANNOTATIONS" / f"TARGET_{pid}.npy")
                if mask.ndim == 3:
                    mask = mask[0]
                t_img = (
                    TF.resize(
                        torch.from_numpy(img),
                        [seg_size, seg_size],
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                    .unsqueeze(0)
                    .to(dev)
                )
                t_mask = (
                    TF.resize(
                        torch.from_numpy(mask.astype(np.int64)).unsqueeze(0),
                        [seg_size, seg_size],
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    .squeeze(0)
                    .numpy()
                )
                logits = net(pixel_values=t_img).logits
                logits = F.interpolate(
                    logits, size=(seg_size, seg_size), mode="bilinear", align_corners=False
                )
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                cm += dense_confusion_matrix(
                    pred, t_mask, n_classes=num_classes, ignore_index=ignore_index
                )
    else:
        raise ValueError(f"model {model!r} not supported; use 'utae' or 'segformer'.")

    return confusion_from_cm(
        cm, model, class_names=PASTIS_R_CLASSES, ignore_index=ignore_index, out_dir=out_dir
    )


def optuna_convergence_figure(
    metrics_dir: Path = Path("reports/segmentation/metrics"),
    *,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Plot the convergence of the Optuna studies (one panel per model).

    For each ``tuning_<model>.parquet`` it plots the mIoU of every COMPLETE
    trial (points) and the *best-so-far* curve (stepped), which shows how
    Optuna found better hyperparameters across the trials. Pruned trials
    (PRUNED) are marked differently. Uses the ``value`` column (mIoU val) or
    ``miou_grouped`` depending on the parquet schema.

    Args:
        metrics_dir: Folder with the ``tuning_<model>.parquet`` files.
        out_dir: Output folder for the figure.

    Returns:
        Path of the written ``optuna_convergence.png`` PNG.

    Raises:
        FileNotFoundError: if there is no ``tuning_*.parquet``.
    """
    import polars as pl

    parts = sorted(metrics_dir.glob("tuning_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No tuning_*.parquet in {metrics_dir}.")

    n = len(parts)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for idx, p in enumerate(parts):
        ax = axes[idx // ncols][idx % ncols]
        df = pl.read_parquet(p)
        model = df["model"][0] if "model" in df.columns else p.stem.replace("tuning_", "")
        metric_col = "value" if "value" in df.columns else "miou_grouped"
        df = df.sort("trial")
        trials = df["trial"].to_list()
        vals = df[metric_col].to_list()
        states = df["state"].to_list() if "state" in df.columns else ["COMPLETE"] * len(trials)

        comp_x = [t for t, s in zip(trials, states, strict=False) if s == "COMPLETE"]
        comp_y = [
            v for v, s in zip(vals, states, strict=False) if s == "COMPLETE" and v is not None
        ]
        pruned_x = [t for t, s in zip(trials, states, strict=False) if s == "PRUNED"]

        ax.scatter(comp_x, comp_y, color="#2b6cb0", s=28, label="trial (COMPLETE)", zorder=3)
        for px in pruned_x:
            ax.axvline(px, color="#cbd5e0", lw=0.6, alpha=0.6, zorder=1)

        # best-so-far over the COMPLETE ones in trial order.
        if comp_y:
            order = np.argsort(comp_x)
            cx = np.asarray(comp_x)[order]
            cy = np.asarray(comp_y)[order]
            best_so_far = np.maximum.accumulate(cy)
            ax.step(cx, best_so_far, where="post", color="#dd6b20", lw=1.8, label="best-so-far")
            bi = int(np.argmax(cy))
            ax.scatter([cx[bi]], [cy[bi]], color="#dd6b20", s=80, zorder=5, edgecolor="white")
            ax.annotate(
                f"mejor: {cy[bi]:.4f}",
                xy=(cx[bi], cy[bi]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#9c4221",
            )
        ax.set_title(f"{model}  ({len(comp_x)} COMPLETE, {len(pruned_x)} PRUNED)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("mIoU")
        ax.legend(fontsize=7, loc="lower right")

    # Turn off the leftover axes of the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Convergencia del ajuste fino (Optuna) por modelo")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optuna_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("optuna_convergence_written", n_models=n, path=str(out_path))
    return out_path


def samples_grid(
    model: Literal["deeplabv3plus", "tsvit"],
    *,
    checkpoint: Path,
    num_classes: int = 18,
    ignore_index: int = 255,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    n_examples: int = 4,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Grid of ``n_examples`` patches (RGB | truth | prediction) + class legend.

    For the us-025 models (deeplab/tsvit): loads the checkpoint, predicts
    ``n_examples`` patches from the validation fold and builds an
    ``n_examples x 3`` grid with a class legend (PASTIS names) as the figure
    caption. Writes ``samples_<model>.png``.

    Args:
        model: ``"deeplabv3plus"`` or ``"tsvit"`` (integrator key).
        checkpoint: Path to ``best.pt``.
        num_classes: 18 (semantic) or 6 (HCAT).
        ignore_index: Ignored label.
        val_folds: Validation folds.
        n_timesteps: T for the temporal model.
        n_examples: Number of patches to show.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Output folder.

    Returns:
        Path of the written PNG.
    """
    from matplotlib import colors
    from matplotlib.patches import Patch

    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.segmentation_inference import (
        load_segmentation_model,
        predict_patch,
        rgb_from_patch,
    )
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    model_kind: Literal["deeplabv3plus", "tsvit-pheno"] = (
        "tsvit-pheno" if model == "tsvit" else model
    )
    collapse: Literal["median"] | None = "median" if model == "deeplabv3plus" else None
    ds = PASTISSegmentationDataset(
        folds=val_folds, collapse_time=collapse, n_timesteps=n_timesteps, target="semantic18"
    )
    net = load_segmentation_model(
        checkpoint,
        model_kind=model_kind,
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        device=device,
    )

    # Equispaced patches along the split (not the first 4 in a row).
    n = len(ds)  # type: ignore[arg-type]
    idxs = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int).tolist()

    cmap = plt.get_cmap("tab20", num_classes)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)
    rows = len(idxs)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    present: set[int] = set()
    titles = ("Entrada (RGB)", "Verdad", "Prediccion")
    for r, idx in enumerate(idxs):
        x, y = ds[idx]
        x_np = x.numpy()
        rgb = rgb_from_patch(np.median(x_np, axis=0) if x_np.ndim == 4 else x_np)
        pred = predict_patch(net, x, model_kind=model_kind)
        yt = np.where(y.numpy() == ignore_index, np.nan, y.numpy().astype(float))
        axes[r][0].imshow(np.clip(rgb, 0, 1))
        axes[r][1].imshow(yt, cmap=cmap, norm=norm, interpolation="nearest")
        axes[r][2].imshow(pred.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
        for col in range(3):
            axes[r][col].axis("off")
            if r == 0:
                axes[r][col].set_title(titles[col])
        present.update(int(v) for v in np.unique(y.numpy()) if v != ignore_index)
        present.update(int(v) for v in np.unique(pred))

    # Legend of present classes (figure caption).
    handles = [
        Patch(color=cmap(norm(c)), label=f"{c}: {PASTIS_R_CLASSES.get(c, f'C{c}')}")
        for c in sorted(present)
        if c < num_classes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"Ejemplos de prediccion - {model}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"samples_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("samples_grid_written", model=model, n=len(idxs), path=str(out_path))
    return out_path
