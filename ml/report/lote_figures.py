"""Real-data figure generators for the US-030..US-040 batch notebooks.

This module turns the artefacts the batch produced (segmentation checkpoints,
the faithful FarSLIP student, the Gemma captions, the ensemble OOF tables) into
the PNG figures the descriptive notebooks display. It is the single source of
truth so the notebooks stay thin: they call ``build_*`` here and ``display()``
the result, never re-implementing inference logic.

Everything runs on **real PASTIS-R French data** (no synthetic, no placeholder):

- Segmenters (US-030/031/038/039): triptych ``input | ground truth |
  prediction`` per patch of the fold-5 held-out split, plus the per-class IoU
  bar and the confusion matrix, reusing :mod:`ml.eval.segmentation_inference`.
- FarSLIP faithful (US-032..037): the Gemma global caption next to the patch,
  the patch-level predicted vs true category, and the ``why it fails`` panel
  that visualizes the 1-CLS-per-patch limitation (all regions of a patch share
  one prediction).
- Ensembles (US-040): the comparison bar of the 5 models reading the real
  ``comparison_us040.csv``.

Project conventions: ``torch``/``numpy`` only at the data boundary; ``polars``
for tables; logging via ``structlog``; type hints everywhere; English
docstrings; figure titles in neutral Spanish; no emojis.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:  # heavy imports kept local to the functions that need them
    import torch
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

# Repository-relative default roots (resolved by the notebook via find_repo_root).
DEFAULT_PASTIS_ROOT = Path("data/PASTIS-R")
DEFAULT_FIGURES_DIR = Path("reports/lote_us030_040/figures")

# The 6 dense segmenters re-scored by the US-030 harness on fold-5 held-out, in
# the canonical reporting order. ``tsvit`` and ``tsvit-pheno-fullm`` are the
# US-038/US-039 Full-M retrains; the rest are the historical L4 checkpoints.
SEGMENTER_KINDS: tuple[str, ...] = (
    "unet",
    "deeplabv3plus",
    "segformer",
    "utae",
    "tsvit",
    "tsvit-pheno-fullm",
)

# Human-facing model names (Spanish prose for the notebook legends).
SEGMENTER_LABELS: dict[str, str] = {
    "unet": "U-Net (ResNet-50)",
    "deeplabv3plus": "DeepLabv3+",
    "segformer": "SegFormer-B0",
    "utae": "U-TAE (temporal)",
    "tsvit": "TSViT Full-M (US-038)",
    "tsvit-pheno": "TSViT-pheno L4",
    "tsvit-pheno-fullm": "TSViT-pheno Full-M (US-039)",
    "anysat": "AnySat (frozen + head)",
}

# Temporal model kinds need the (T, 10, H, W) series; the rest get a 2D
# composite. The dataset n_timesteps for the Full-M models is 37 (PASTIS T_MIN),
# the single source of truth fixed in TSVIT_FULLM_CONFIG (US-038).
_TEMPORAL_KINDS: frozenset[str] = frozenset(
    {"utae", "tsvit", "tsvit-pheno", "tsvit-pheno-fullm", "anysat"}
)


def _resolve_device(device: str) -> str:
    """Return ``cuda`` when available and requested ``auto``, else ``cpu``."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _n_timesteps_for(kind: str) -> int:
    """Return the dataset n_timesteps a temporal kind was trained with.

    Full-M TSViT variants use 37 (PASTIS T_MIN, the ordinal positional encoding
    shape ``(1, 37, dim)``); the historical L4 temporal models use 10.
    """
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY

    spec = CHECKPOINT_REGISTRY.get(kind)
    if spec is not None and spec.model_kwargs:
        return int(spec.model_kwargs.get("n_timesteps", 10))
    return 10


def build_segmenter_dataset(
    kind: str,
    *,
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    folds: Sequence[int] = (5,),
) -> Any:
    """Build the held-out PASTISSegmentationDataset for a segmenter kind.

    Picks 2D vs temporal mode and the correct ``n_timesteps`` from the registry
    so the dataset matches what the checkpoint was trained on (R-HARNESS).

    Args:
        kind: Segmenter model_kind (key of ``CHECKPOINT_REGISTRY``).
        pastis_root: Real PASTIS-R root.
        folds: Held-out folds (default fold-5, the published comparison split).

    Returns:
        A ``PASTISSegmentationDataset`` returning ``(x, y)`` tuples.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

    is_temporal = kind in _TEMPORAL_KINDS
    return PASTISSegmentationDataset(
        root=Path(pastis_root),
        folds=tuple(int(f) for f in folds),
        n_timesteps=_n_timesteps_for(kind) if is_temporal else 10,
        collapse_time=None if is_temporal else "median",
        target="semantic18",
        ignore_index=255,
    )


def _predict_18class_map(
    model: Any,
    x: torch.Tensor,
    spec: Any,
    *,
    dataset: Any,
    idx: int,
    device: Any,
) -> np.ndarray:
    """Predict a patch's class map in the contiguous 18-class space (harness-faithful).

    Replicates the per-model dispatch of
    :func:`ml.eval.dense_metrics._rescore_one`: SegFormer runs its 3-RGB/256
    sub-pipeline; the other kinds go through ``predict_patch_for_kind`` and are
    remapped 20->18 (and resampled to 128 if ``needs_resize``). This is the ONLY
    correct path for the 6 heterogeneous checkpoints (their forward signatures
    differ); the generic ``predict_patch`` does not handle unet/utae/segformer.

    Args:
        model: Loaded model in eval().
        x: Patch tensor ``(x, _y)`` from the dataset.
        spec: The model's CheckpointSpec.
        dataset: The PASTISSegmentationDataset (needed for SegFormer's patch_id).
        idx: Patch index (needed for SegFormer's raw S2 lookup).
        device: Torch device.

    Returns:
        Predicted class map ``(128, 128)`` int64 in the 18-class contiguous space.
    """
    from ml.eval.class_remap import remap_20_to_18, resample_mask_128_nearest
    from ml.eval.dense_metrics import _segformer_predict_18
    from ml.eval.segmentation_inference import predict_patch_for_kind

    if spec.model_kind == "segformer":
        pid = dataset.patch_ids[idx]
        return _segformer_predict_18(model, pid, root=dataset.root, device=device)
    pred_native = predict_patch_for_kind(model, x, model_kind=spec.model_kind)
    if spec.needs_resize:
        pred_native = resample_mask_128_nearest(pred_native)
    return (
        remap_20_to_18(pred_native)
        if spec.native_num_classes >= 20
        else pred_native.astype(np.int64)
    )


def build_segmenter_triptychs(
    kind: str,
    *,
    indices: Sequence[int],
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    folds: Sequence[int] = (5,),
    device: str = "auto",
) -> list[Figure]:
    """Generate the ``input | GT | prediction`` triptychs for a segmenter.

    Loads the checkpoint via the harness loader, builds the held-out dataset and
    predicts each patch in the 18-class space via :func:`_predict_18class_map`
    (harness-faithful, handles every architecture). All real data: no synthetic
    patches, no placeholder masks.

    Args:
        kind: Segmenter model_kind.
        indices: Dataset indices to visualize (real fold-5 patches).
        pastis_root: Real PASTIS-R root.
        folds: Held-out folds.
        device: ``auto``/``cuda``/``cpu``.

    Returns:
        One matplotlib figure per index (1x3 panels each).
    """
    import torch as _torch

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.segmentation_inference import (
        load_checkpoint_model,
        prediction_figure,
        rgb_from_patch,
    )

    dev = _resolve_device(device)
    spec = CHECKPOINT_REGISTRY[kind]
    model = load_checkpoint_model(spec, n_timesteps=_n_timesteps_for(kind), device=dev)
    dataset = build_segmenter_dataset(kind, pastis_root=pastis_root, folds=folds)
    valid = [i for i in indices if 0 <= i < len(dataset)]
    logger.info(
        "segmenter_triptychs",
        kind=kind,
        n_requested=len(indices),
        n_valid=len(valid),
        n_patches=len(dataset),
        device=dev,
    )
    figs: list[Figure] = []
    torch_device = _torch.device(dev)
    for idx in valid:
        x, y = dataset[idx]
        x_np = x.numpy()
        rgb = rgb_from_patch(np.median(x_np, axis=0)) if x_np.ndim == 4 else rgb_from_patch(x_np)
        pred = _predict_18class_map(model, x, spec, dataset=dataset, idx=idx, device=torch_device)
        figs.append(
            prediction_figure(
                rgb,
                y.numpy(),
                pred,
                num_classes=18,
                titles=("Entrada (RGB)", "Verdad de campo", "Prediccion"),
            )
        )
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return figs


def build_segmenter_confusion(
    kind: str,
    *,
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    folds: Sequence[int] = (5,),
    max_patches: int | None = None,
    device: str = "auto",
) -> tuple[dict[str, object], np.ndarray, Figure]:
    """Evaluate a segmenter on the held-out split and return metrics + CM figure.

    Walks the fold-5 patches predicting each in the 18-class space via
    :func:`_predict_18class_map` (harness-faithful), accumulates the dense
    confusion matrix and derives the metrics (mIoU, F1-macro, pixel_acc,
    per-class IoU) with the same definitions as the US-030 harness. Builds a
    row-normalized confusion matrix heatmap.

    Args:
        kind: Segmenter model_kind.
        pastis_root: Real PASTIS-R root.
        folds: Held-out folds.
        max_patches: Cap for a quick smoke (None = full split).
        device: ``auto``/``cuda``/``cpu``.

    Returns:
        ``(metrics, cm, figure)``.
    """
    import matplotlib.pyplot as plt
    import torch as _torch

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm
    from ml.eval.segmentation_inference import load_checkpoint_model
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    dev = _resolve_device(device)
    spec = CHECKPOINT_REGISTRY[kind]
    model = load_checkpoint_model(spec, n_timesteps=_n_timesteps_for(kind), device=dev)
    dataset = build_segmenter_dataset(kind, pastis_root=pastis_root, folds=folds)
    torch_device = _torch.device(dev)

    n = len(dataset)
    if max_patches is not None:
        n = min(n, max_patches)
    cm = np.zeros((18, 18), dtype=np.int64)
    with _torch.no_grad():
        for idx in range(n):
            x, y = dataset[idx]
            pred_18 = _predict_18class_map(
                model, x, spec, dataset=dataset, idx=idx, device=torch_device
            )
            cm += dense_confusion_matrix(pred_18, y.numpy(), n_classes=18, ignore_index=255)
    metrics = dense_metrics_from_cm(cm)
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    logger.info(
        "segmenter_confusion",
        kind=kind,
        n_patches=n,
        miou=round(float(metrics["miou"]), 4),
        f1_macro=round(float(metrics["f1_macro"]), 4),
    )

    # Row-normalized CM (recall per class) for readability.
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)

    names = [PASTIS_R_CLASSES.get(i + 1, str(i + 1)) for i in range(18)]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(18))
    ax.set_yticks(range(18))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase verdadera")
    ax.set_title(
        f"Matriz de confusion (recall por clase) - {SEGMENTER_LABELS.get(kind, kind)}\n"
        f"mIoU {float(metrics['miou']):.3f} | F1-macro {float(metrics['f1_macro']):.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraccion de pixeles")
    fig.tight_layout()
    return metrics, cm, fig


def per_class_iou_table(metrics: dict[str, object]) -> pl.DataFrame:
    """Build a per-class IoU/F1 table from an evaluate_checkpoint metrics dict.

    Args:
        metrics: The dict returned by ``evaluate_checkpoint`` (has ``per_class_iou``
            and ``per_class_f1`` lists of length 18, ``None`` for absent classes).

    Returns:
        A Polars DataFrame ``class_id | name | iou | f1`` sorted by IoU desc.
    """
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    iou = list(cast("list[object]", metrics.get("per_class_iou", []) or []))
    f1 = list(cast("list[object]", metrics.get("per_class_f1", []) or []))
    n = int(max(len(iou), len(f1)))

    def _val(seq: list[object], i: int) -> float:
        if i >= len(seq) or seq[i] is None:
            return float("nan")
        return float(seq[i])  # type: ignore[arg-type]

    rows = {
        "class_id": list(range(1, n + 1)),
        "name": [PASTIS_R_CLASSES.get(i + 1, str(i + 1)) for i in range(n)],
        "iou": [_val(iou, i) for i in range(n)],
        "f1": [_val(f1, i) for i in range(n)],
    }
    return pl.DataFrame(rows).sort("iou", descending=True, nulls_last=True)


# ---------------------------------------------------------------------------
# FarSLIP faithful (US-032..037): captions Gemma + patch prediction + why-fail
# ---------------------------------------------------------------------------

#: Default faithful student checkpoint (US-036-a v2 winner, in DVC).
DEFAULT_FAITHFUL_CKPT = Path("checkpoints/farslip/faithful_v2/best.safetensors")
DEFAULT_CAPTIONS_PARQUET = Path("data/farslip/pastis_captions.parquet")


def load_faithful_student_and_bank(
    *,
    checkpoint: Path = DEFAULT_FAITHFUL_CKPT,
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    captions_parquet: Path = DEFAULT_CAPTIONS_PARQUET,
    eval_folds: Sequence[int] = (4,),
    device: str = "auto",
) -> tuple[Any, Any, torch.Tensor, list[int]]:
    """Load the faithful FarSLIP student, its eval dataset and category bank.

    Rebuilds exactly the eval pipeline that produced ``faithful_v2_per_class.csv``
    (US-036-a v2 / US-037): the region-category dataset over held-out folds, the
    US-033 MiniLM category prototypes reprojected to the student CLS-768 space,
    and the student with the winning weights loaded. All real PASTIS-R data.

    Args:
        checkpoint: faithful student ``best.safetensors`` (DVC-tracked).
        pastis_root: real PASTIS-R root.
        captions_parquet: the Gemma global captions parquet (DVC-tracked).
        eval_folds: held-out folds (default fold-4, the eval split of the run).
        device: ``auto``/``cuda``/``cpu``.

    Returns:
        ``(trainer, val_dataset, bank_768, class_ids)`` where ``bank_768`` is the
        reprojected category prototype bank on the student's device.
    """
    from safetensors.torch import load_file

    import scripts.run_us036a_v2_farslip_faithful as orch
    from ml.farslip.caption_cache import load_captions
    from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
    from ml.farslip.region_category_dataset import RegionCategoryPairDataset

    active = tuple(range(1, 19))
    captions = load_captions(Path(captions_parquet))
    val_ds = RegionCategoryPairDataset(
        captions,
        root=Path(pastis_root),
        folds=tuple(int(f) for f in eval_folds),
        active_class_ids=active,
    )
    bank, class_ids = orch._category_prototypes(None, active)

    cfg = FarSLIPTrainerConfig(
        dataset_root=Path(pastis_root),
        output_dir=Path(checkpoint).parent,
        n_in_channels=4,
        n_categories=len(active),
        supervision="region_category",
        device=device,
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=val_ds)
    trainer.set_category_prototypes(bank, class_ids)
    state = load_file(str(checkpoint))
    missing, unexpected = trainer.student.load_state_dict(state, strict=False)
    logger.info(
        "faithful_student_loaded",
        checkpoint=str(checkpoint),
        n_missing=len(missing),
        n_unexpected=len(unexpected),
        n_eval_patches=len(val_ds),
        n_categories=len(class_ids),
    )
    prototypes = trainer._category_prototypes
    assert prototypes is not None  # populated by set_category_prototypes above
    return trainer, val_ds, prototypes, class_ids


def _predict_patch_category(
    student: Any,
    image: torch.Tensor,
    bank_768: torch.Tensor,
    device: torch.device,
) -> int:
    """Predict a patch category index by CLS-768 <-> prototype cosine argmax."""
    import torch as _torch

    protos_n = _torch.nn.functional.normalize(
        bank_768.to(device=device, dtype=_torch.float32), p=2, dim=-1
    )
    with _torch.no_grad():
        out = student(
            pixel_values=image.unsqueeze(0).to(device=device, dtype=_torch.float32),
            output_hidden_states=False,
        )
        cls = out.last_hidden_state[:, 0, :]
        cls_n = _torch.nn.functional.normalize(cls.float(), p=2, dim=-1)
        pred = int((cls_n @ protos_n.t()).argmax(dim=-1).item())
    return pred


def build_farslip_prediction_figures(
    *,
    n_examples: int = 6,
    checkpoint: Path = DEFAULT_FAITHFUL_CKPT,
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    captions_parquet: Path = DEFAULT_CAPTIONS_PARQUET,
    eval_folds: Sequence[int] = (4,),
    device: str = "auto",
    seed: int = 42,
) -> list[tuple[Figure, dict[str, Any]]]:
    """Build the FarSLIP faithful per-patch figures (real PASTIS-R, real Gemma).

    For each selected held-out patch produces a 2-panel figure: (1) the peak-NDVI
    RGB composite the student saw, with the Gemma global caption as subtitle and
    the true vs predicted category in the title; (2) the region map colored by
    PASTIS category, which makes the 1-CLS-per-patch limitation visible -- the
    student emits ONE prediction for the whole patch even though it contains
    several categories, so rare parcels sharing the patch with Meadow are
    irrecoverable. This is the visual root-cause of the ~4-class ceiling.

    Args:
        n_examples: number of patches to visualize (mix of correct/incorrect).
        checkpoint: faithful student checkpoint.
        pastis_root: real PASTIS-R root.
        captions_parquet: Gemma captions parquet.
        eval_folds: held-out folds.
        device: ``auto``/``cuda``/``cpu``.
        seed: deterministic example selection.

    Returns:
        List of ``(figure, info_dict)`` where ``info_dict`` carries patch_id,
        true/pred names, caption and the per-category region counts.
    """
    import matplotlib.pyplot as plt
    import torch as _torch
    from matplotlib import colors

    import scripts.run_us036a_v2_farslip_faithful as orch
    from ml.farslip.pastis_pair_dataset import peak_ndvi_composite
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES, load_pastis_patch

    _patch_majority_category = orch._patch_majority_category

    trainer, val_ds, bank_768, class_ids = load_faithful_student_and_bank(
        checkpoint=checkpoint,
        pastis_root=pastis_root,
        captions_parquet=captions_parquet,
        eval_folds=eval_folds,
        device=device,
    )
    dev = trainer.device
    pastis_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    idx_to_pastis = {i: cid for i, cid in enumerate(class_ids)}

    captions_lf = pl.read_parquet(captions_parquet)
    caption_of: dict[str, str] = {
        str(r["patch_id"]): str(r["caption_glo"]) for r in captions_lf.iter_rows(named=True)
    }

    # Deterministic example selection: spread across the dataset, prefer a mix of
    # correct and incorrect by walking with a fixed stride from the seed.
    n = len(val_ds)
    stride = max(1, n // max(1, n_examples * 3))
    candidates = list(range(seed % max(1, stride), n, stride))[: n_examples * 3]

    figs: list[tuple[Figure, dict[str, Any]]] = []
    cmap = plt.get_cmap("tab20", 19)
    norm = colors.Normalize(vmin=0, vmax=18)

    from matplotlib.patches import Patch

    for idx in candidates:
        if len(figs) >= n_examples:
            break
        pid, regions = val_ds._samples[idx]
        majority = _patch_majority_category(regions)
        if majority not in pastis_to_idx:
            continue
        item = val_ds[idx]
        pred_idx = _predict_patch_category(trainer.student, item["image"], bank_768, dev)
        pred_cat = idx_to_pastis.get(pred_idx, -1)
        true_name = PASTIS_R_CLASSES.get(majority, str(majority))
        pred_name = PASTIS_R_CLASSES.get(pred_cat, str(pred_cat))

        # Real PASTIS annotations: parcel instance mask + per-parcel true crop.
        # The per-parcel GT (instance + region category) respects the panoptic
        # polygons, more faithful than the raw dense ``semantic`` raster.
        patch = load_pastis_patch(pid, root=Path(pastis_root))
        s2 = patch["s2"]
        instance = np.asarray(patch["instance"])
        rgb = _rgb_from_peak_ndvi(peak_ndvi_composite(np.asarray(s2)))

        # Per-parcel category map (true crop per parcel) for the GT panel.
        region_cat_map = _region_category_map(instance, regions)
        # The single FarSLIP prediction painted over EVERY parcel of the patch
        # (the model gives one label for the whole patch -> all parcels share it).
        pred_map = _region_category_map(instance, [(inst, pred_cat) for inst, _c in regions])

        # Per-parcel honesty: against how many parcels would the single patch
        # prediction be correct? (a parcel "matches" if its true crop == pred_cat)
        parcels_match = sum(1 for _i, c in regions if c == pred_cat)

        ok = pred_cat == majority
        cap = caption_of.get(str(pid), "(sin caption)")
        cats_present = sorted({c for _i, c in regions})
        n_cats = len(cats_present)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        # Panel 1: real RGB input + Gemma caption.
        axes[0].imshow(np.clip(rgb, 0, 1))
        axes[0].set_title(f"Entrada (RGB) - Patch {pid}", fontsize=10)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel("Caption Gemma: " + _wrap(cap, 64), fontsize=6.5)

        # Panel 2: dense PASTIS ground truth per parcel (the REAL polygons).
        axes[1].imshow(region_cat_map, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1].set_title(
            f"Verdad PASTIS por parcela\n{len(regions)} parcelas, {n_cats} cultivos",
            fontsize=10,
        )
        axes[1].axis("off")
        # Legend with the real crop names present in this patch.
        handles = [
            Patch(color=cmap(norm(c)), label=PASTIS_R_CLASSES.get(c, str(c))) for c in cats_present
        ]
        axes[1].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.02),
            fontsize=6.5,
            ncol=2,
            frameon=False,
        )

        # Panel 3: the single FarSLIP prediction applied to all parcels.
        axes[2].imshow(pred_map, cmap=cmap, norm=norm, interpolation="nearest")
        axes[2].set_title(
            f"Prediccion FarSLIP (1 por patch): {pred_name}\n"
            f"acierta {parcels_match}/{len(regions)} parcelas "
            f"[{'mayoria correcta' if ok else 'mayoria incorrecta'}]",
            fontsize=10,
            color="green" if ok else "firebrick",
        )
        axes[2].axis("off")
        handles_p = [Patch(color=cmap(norm(pred_cat)), label=pred_name)]
        axes[2].legend(
            handles=handles_p,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.02),
            fontsize=6.5,
            frameon=False,
        )
        fig.tight_layout()

        cat_counts: dict[str, int] = {}
        for _inst, cat in regions:
            name = PASTIS_R_CLASSES.get(cat, str(cat))
            cat_counts[name] = cat_counts.get(name, 0) + 1
        figs.append(
            (
                fig,
                {
                    "patch_id": str(pid),
                    "true": true_name,
                    "pred": pred_name,
                    "correct": ok,
                    "caption": cap,
                    "n_regions": len(regions),
                    "n_categories": n_cats,
                    "parcels_matched": parcels_match,
                    "category_counts": cat_counts,
                },
            )
        )

    # Free GPU memory after generating (the student is heavy).
    del trainer
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    logger.info("farslip_prediction_figures", n_generated=len(figs))
    return figs


def _rgb_from_peak_ndvi(img4: np.ndarray) -> np.ndarray:
    """Build an RGB ``(H, W, 3)`` in [0,1] from the 4-band peak-NDVI composite.

    The peak-NDVI composite is ``(4, H, W)`` with bands [B4, B3, B2, NIR] in
    [0,1]; the RGB panel uses B4/B3/B2 with a 2-98 percentile stretch.
    """
    rgb = np.stack([img4[0], img4[1], img4[2]], axis=-1).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1.0
    stretched: np.ndarray = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return stretched


def _region_category_map(instance: np.ndarray, regions: list[tuple[int, int]]) -> np.ndarray:
    """Map each parcel instance to its category id for a colored region map.

    Pixels with no region (or a region outside ``regions``) become NaN (drawn
    neutral). Used to visualize the multi-category content of one patch.
    """
    inst = np.asarray(instance)
    out = np.full(inst.shape, np.nan, dtype=float)
    cat_of = {int(i): int(c) for i, c in regions}
    for inst_id, cat in cat_of.items():
        out[inst == inst_id] = float(cat)
    return out


def _wrap(text: str, width: int) -> str:
    """Word-wrap a caption to ``width`` chars for a multi-line xlabel."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)[:4])


# ---------------------------------------------------------------------------
# Ensembles (US-040): real comparison bar from comparison_us040.csv
# ---------------------------------------------------------------------------

DEFAULT_ENSEMBLE_CSV = Path("reports/ensemble/metrics/comparison_us040.csv")


def build_ensemble_comparison_figure(
    *,
    csv_path: Path = DEFAULT_ENSEMBLE_CSV,
) -> tuple[Figure, pl.DataFrame]:
    """Build the US-040 ensemble comparison bar from the real OOF results CSV.

    Reads ``comparison_us040.csv`` (5 models, the ``chosen`` Stacking flagged) and
    draws an F1-macro bar with the winner highlighted. No re-inference: the table
    is the real fold-5 held-out result of US-040.

    Args:
        csv_path: the ensemble comparison CSV.

    Returns:
        ``(figure, dataframe)``.
    """
    import matplotlib.pyplot as plt

    df = pl.read_csv(csv_path).sort("f1_macro", descending=True)
    models = df["model"].to_list()
    f1 = df["f1_macro"].to_list()
    chosen = df["chosen"].to_list() if "chosen" in df.columns else [False] * len(models)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_colors = ["#2a9d8f" if str(c).lower() in ("true", "1") else "#9bbbd4" for c in chosen]
    ax.barh(range(len(models)), f1, color=bar_colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("F1-macro (fold-5 held-out)")
    ax.set_title("Comparativa de ensambles US-040 (verde = elegido: Stacking heterogeneo)")
    for i, v in enumerate(f1):
        ax.text(float(v) + 0.005, i, f"{float(v):.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(f1) * 1.15)
    fig.tight_layout()
    return fig, df


# ---------------------------------------------------------------------------
# US-031..035 data-preparation steps: one real figure/table each
# ---------------------------------------------------------------------------
DEFAULT_OOF_MANIFEST = Path("ml/eval/oof/manifest.json")
DEFAULT_PROTOTYPES = Path("data/features/phenology_class_prototypes_pastis.parquet")
DEFAULT_BAND_LOGS = Path("reports/farslip/logs")


def build_us031_oof_figure(
    *, manifest_path: Path = DEFAULT_OOF_MANIFEST
) -> tuple[Figure, pl.DataFrame]:
    """US-031: bar of OOF softmax dumps per model from the real manifest.

    Reads ``ml/eval/oof/manifest.json`` (the real US-031 dump over fold-5) and
    shows, per model, how many held-out patches got a valid probability tensor
    (status=ok). It does NOT re-run inference: the manifest is the real artefact
    that fed the ensembles (US-040).

    Args:
        manifest_path: the OOF manifest written by US-031.

    Returns:
        ``(figure, dataframe)`` with one row per model (n_patches, status, shape).
    """
    import json

    import matplotlib.pyplot as plt

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    models = manifest.get("models", {})
    rows = [
        {
            "modelo": name,
            "n_patches": int(v.get("n_patches", 0)),
            "status": v.get("status", "?"),
            "shape": "x".join(str(s) for s in v.get("shape", [])),
        }
        for name, v in models.items()
    ]
    df = pl.DataFrame(rows).sort("modelo")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = df["modelo"].to_list()
    counts = df["n_patches"].to_list()
    ok = [s == "ok" for s in df["status"].to_list()]
    ax.barh(
        range(len(names)),
        counts,
        color=["#2a9d8f" if o else "#c1666b" for o in ok],
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Patches del fold-5 con probabilidades volcadas (status=ok)")
    ax.set_title(
        "US-031: volcado OOF de probabilidades por modelo (fold-5 held-out)\n"
        f"Esquema 18 clases, {manifest.get('num_classes', 18)} clases x "
        f"{manifest.get('size', 128)}x{manifest.get('size', 128)} por patch, "
        "post-softmax (anti-fuga)"
    )
    for i, v in enumerate(counts):
        ax.text(v + 2, i, str(v), va="center", fontsize=8)
    fig.tight_layout()
    return fig, df


def build_us032_filter_figure(
    *,
    pastis_root: Path = DEFAULT_PASTIS_ROOT,
    folds: Sequence[int] = (1, 2, 3),
    ratio: float = 3.0,
) -> tuple[Figure, pl.DataFrame]:
    """US-032: real retention of the 3:1 Meadow-dominance per-patch filter.

    Recomputes the filter on real PASTIS-R with :class:`ml.data.pastis_filter.PastisFilter`
    in ``dominance_ratio`` mode (the 3:1 rule) for two target sets: the Stage-1
    crops {1,2,3,8} and all crops {1..18}. Shows kept vs dropped per set. Real
    recomputation, not copied from prose.

    Args:
        pastis_root: real PASTIS-R root (needs metadata + annotations).
        folds: folds to scan (default train 1,2,3).
        ratio: dominance ratio (3.0 = the 3:1 rule).

    Returns:
        ``(figure, dataframe)`` with kept/total/pct per target set.
    """
    import matplotlib.pyplot as plt

    from ml.data.pastis_filter import PastisFilter

    sets = {
        "Stage-1 {Meadow,Wheat,Corn,Grapevine}": [1, 2, 3, 8],
        "Todos los cultivos {1..18}": list(range(1, 19)),
    }
    rows = []
    for label, targets in sets.items():
        f = PastisFilter(
            Path(pastis_root),
            target_classes=targets,
            mode="dominance_ratio",
            ratio=ratio,
        )
        kept = f.filter_folds(list(folds))
        total = int(getattr(f, "total_scanned", 0)) or len(kept)
        rows.append(
            {
                "conjunto": label,
                "retenidos": len(kept),
                "total": total,
                "pct": round(100.0 * len(kept) / total, 1) if total else 0.0,
            }
        )
    df = pl.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = df["conjunto"].to_list()
    kept_v = df["retenidos"].to_list()
    total_v = df["total"].to_list()
    y = range(len(labels))
    ax.barh(y, total_v, color="#dde3ea", label="total escaneado")
    ax.barh(y, kept_v, color="#2a9d8f", label="retenidos (pasan 3:1)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Numero de patches")
    ax.set_title(
        f"US-032: filtro de dominancia {ratio:.0f}:1 de pradera por patch "
        f"(folds {','.join(map(str, folds))})"
    )
    for i, (k, t) in enumerate(zip(kept_v, total_v, strict=True)):
        pct = 100.0 * k / t if t else 0.0
        ax.text(t + 8, i, f"{k}/{t} ({pct:.1f}%)", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, df


def build_us033_phenology_figure(
    *,
    prototypes_path: Path = DEFAULT_PROTOTYPES,
    class_ids: Sequence[int] = (1, 2, 3, 8),
) -> tuple[Figure, pl.DataFrame]:
    """US-033: real per-class NDVI curve + the Gemini-generated description.

    Reads the real prototype parquet (NDVI curve per class + Spanish text from
    Gemini Flash) and plots the curves of the selected classes, returning a table
    of the descriptions. This is the phenology signal that fed FarSLIP's
    contrastive prototypes.

    Args:
        prototypes_path: the US-033 prototype parquet.
        class_ids: PASTIS classes to plot (default the 4 dominant).

    Returns:
        ``(figure, dataframe)`` with class_id, name and description.
    """
    import matplotlib.pyplot as plt

    df = pl.read_parquet(prototypes_path)
    keep = df.filter(pl.col("class_id").is_in(list(class_ids)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in keep.iter_rows(named=True):
        curve = r["ndvi_curve"]
        ax.plot(range(len(curve)), curve, marker="o", markersize=3, label=r["class_name"])
    ax.set_xlabel("Fecha (indice temporal de la serie Sentinel-2)")
    ax.set_ylabel("NDVI medio")
    ax.set_title(
        "US-033: curva fenologica (NDVI) media por clase\n"
        "De cada curva, Gemini Flash genero una descripcion textual (tabla abajo)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    table = keep.select(["class_id", "class_name", "description"])
    return fig, table


def build_us034_fix_figure(
    *,
    prototypes_path: Path = DEFAULT_PROTOTYPES,
    seed: int = 42,
) -> tuple[Figure, pl.DataFrame]:
    """US-034: separability of random vs real phenology prototypes (the fix).

    Quantifies WHY the fix mattered: the broken path initialized the contrastive
    prototypes with ``torch.randn`` (alignment against noise). We compare the
    mean pairwise cosine SIMILARITY of the 18 real prototypes vs 18 random ones
    of the same shape: real prototypes are structured (distinct classes separate),
    random ones are near-orthogonal noise carrying no class signal. Lower mean
    off-diagonal similarity with MORE spread = usable structure.

    Args:
        prototypes_path: the US-033 real prototype parquet.
        seed: seed for the random baseline (reproducible).

    Returns:
        ``(figure, dataframe)`` with the summary stats of both banks.
    """
    import matplotlib.pyplot as plt

    df = pl.read_parquet(prototypes_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    real = df.select(emb_cols).to_numpy().astype(np.float64)  # (18, 384)
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal(real.shape)

    def _offdiag_cos(mat: np.ndarray) -> np.ndarray:
        norm: np.ndarray = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sim: np.ndarray = norm @ norm.T
        n = sim.shape[0]
        offdiag: np.ndarray = sim[~np.eye(n, dtype=bool)]
        return offdiag

    cos_real = _offdiag_cos(real)
    cos_rand = _offdiag_cos(rand)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(
        cos_rand,
        bins=30,
        alpha=0.6,
        label="Prototipos aleatorios (bug torch.randn)",
        color="#c1666b",
        density=True,
    )
    ax.hist(
        cos_real,
        bins=30,
        alpha=0.6,
        label="Prototipos fenologicos reales (fix)",
        color="#2a9d8f",
        density=True,
    )
    ax.set_xlabel("Similitud coseno entre pares de clases (fuera de la diagonal)")
    ax.set_ylabel("Densidad")
    ax.set_title(
        "US-034: por que importo el fix\n"
        "Los prototipos reales tienen estructura por clase; los aleatorios son ruido"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    summary = pl.DataFrame(
        {
            "banco": ["aleatorio (bug)", "real (fix)"],
            "cos_medio": [round(float(cos_rand.mean()), 4), round(float(cos_real.mean()), 4)],
            "cos_std": [round(float(cos_rand.std()), 4), round(float(cos_real.std()), 4)],
            "cos_max": [round(float(cos_rand.max()), 4), round(float(cos_real.max()), 4)],
        }
    )
    return fig, summary


def build_us035_bands_figure(*, logs_dir: Path = DEFAULT_BAND_LOGS) -> tuple[Figure, pl.DataFrame]:
    """US-035: band-ablation table from the three real H100 training logs.

    Parses the final ``training done`` line of each band variant log
    (baseline-rgb, baseline-nir, 4band-pheno) and builds the apples-to-apples
    ablation table (loss_cls / loss_patch / loss_total). The honest caveat (per
    US-035) is shown in the notebook: the loss is internal; the real embedding
    quality is measured downstream in US-037.

    Args:
        logs_dir: directory with the three band logs.

    Returns:
        ``(figure, dataframe)`` with one row per band variant.
    """
    import re

    import matplotlib.pyplot as plt

    variants = {
        "baseline-rgb": "RGB (3 bandas)",
        "baseline-nir": "NIR+RGB (3 bandas)",
        "4band-pheno": "4 bandas + fenologia",
    }
    rows = []
    for stem, label in variants.items():
        log = Path(logs_dir) / f"{stem}.log"
        if not log.is_file():
            continue
        text = log.read_text(encoding="utf-8", errors="ignore")
        done = [ln for ln in text.splitlines() if "training done" in ln]
        if not done:
            continue
        line = done[-1]

        def _grab(key: str, ln: str = line) -> float:
            m = re.search(rf"{key}=([0-9.]+)", ln)
            return float(m.group(1)) if m else float("nan")

        rows.append(
            {
                "variante": label,
                "loss_cls": round(_grab("loss_cls"), 4),
                "loss_patch": round(_grab("loss_patch"), 4),
                "loss_total": round(_grab("loss_total"), 4),
            }
        )
    df = pl.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = df["variante"].to_list()
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, df["loss_cls"].to_list(), width, label="loss_cls", color="#2a6f97")
    ax.bar(x + width / 2, df["loss_total"].to_list(), width, label="loss_total", color="#89c2d9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Loss (final del entrenamiento)")
    ax.set_title(
        "US-035: ablacion de bandas FarSLIP (3 corridas reales H100)\n"
        "La calidad real del embedding se mide downstream (US-037), no por el loss"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, df
