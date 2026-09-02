"""Dense transfer finetune France -> Catalonia on Sen4AgriNet (US-075, EPIC 12).

Takes the PASTIS-R (France) dense champion ``tsvit-pheno-v1/best.pt`` (TSViT,
18-class native, ``n_timesteps=10``) and adapts it to the Catalonia macro-HCAT
label-space (US-074, 10 trained groups + ``ignore_index=255``) to measure the
**Franco-Iberian domain gap** as a ``Delta mIoU = miou_few_shot - miou_zero_shot``
over the SAME real held-out Catalonia validation patches.

Pipeline (everything on REAL downloaded patches, never synthetic):

1. **Index the subset** via :class:`ml.data.sen4agrinet_adapter.Sen4AgriNetDataset`
   (``countries=("ES",)`` for Catalonia, ``("FR",)`` for the France reference).
   Each 366x366 patch tiles into 128x128 sub-patches; the macro target lives in
   ``[0, N_MACRO)`` union ``{255}``.
2. **Few-shot split**: the Catalonia patches are split at the PATCH level (not the
   tile level) into ``k`` few-shot train patches + the rest held-out for val, so a
   tile of a train patch never leaks into val (spatial honesty).
3. **Zero-shot eval**: the France checkpoint is rebuilt at its NATIVE 18 classes,
   weights loaded, and each Catalonia val prediction is projected 18 -> macro via
   the US-074 crosswalk (:data:`SEMANTIC18_TO_MACRO`). Accumulated in
   :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` (``N_MACRO``,
   ``ignore_index=255``). Always reportable (no convergence required) = plan B.
4. **Few-shot finetune**: a macro head (``num_classes=N_MACRO``) TSViT is built,
   the France encoder loaded with ``strict=False`` (the 18-class
   ``temporal_cls_tokens`` and the seg head are reinitialized; the temporal /
   spatial transformers, patch embedding and positional encodings transfer). Two
   LR groups: encoder LR low (1e-5) + new head LR normal (1e-4); or
   ``--linear-probe`` freezes the encoder entirely (cheap variant). Reuses
   :func:`ml.train.train_segmentation.train_segmentation` (Dice+CE, warmup+cosine,
   best.pt/last.pt, MLflow ``track_experiment``).
5. **Few-shot eval + Delta**: the finetuned macro model is evaluated on the same
   Catalonia val (head already macro, no projection) and the Delta reported.

Run on the VM H100::

    F:\\tools\\micromamba.exe run -n agrosat python -m ml.train.finetune_sen4agrinet \\
        --root F:/projects/agrosat-copilot/data/sen4agrinet \\
        --fr-ckpt F:/projects/agrosat-copilot/checkpoints/segmentation/tsvit-pheno-v1/best.pt \\
        --k 10 --epochs 40 --device cuda

Permanent operational tool (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
import torch

from ml.data.hcat_crosswalk import MACRO_HCAT_GROUPS
from ml.data.sen4agrinet_adapter import (
    IGNORE_INDEX,
    MACRO_GROUP_TO_ID,
    N_MACRO_CLASSES,
    Sen4AgriNetDataset,
)
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.eval.segmentation_inference import predict_patch_for_kind
from ml.models.tsvit_wrapper import build_tsvit

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    import argparse

    from torch.utils.data import Dataset

logger = structlog.get_logger(__name__)

# MLflow 3.x emits emojis when closing runs; the Windows console uses cp1252 and
# that causes UnicodeEncodeError. Force UTF-8 (no-op on Linux/macOS).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

#: France PASTIS-R native class count (semantic18 contiguous head).
_FR_NUM_CLASSES = 18
#: France TSViT-pheno checkpoint capacity (L4 default; matches best.pt: dim=128,
#: depth 4+4, n_timesteps=10, patch_size=8, in_channels=10).
_FR_N_TIMESTEPS = 10
_TILE_SIZE = 128

#: ``semantic18_id -> macro_group_name`` derived from the US-074 crosswalk
#: (``ml.eval.class_remap._HCAT_MACRO_BY_ID``: the ``"L1_6|macro"`` annotation, the
#: macro is the second field). Hardcoded so the projection never depends on the
#: parquet at inference time; a guard below asserts every id maps to a known macro.
_SEMANTIC18_TO_MACRO_NAME: dict[int, str] = {
    0: "grassland",
    1: "cereals",
    2: "cereals",
    3: "cereals",
    4: "oilseed_industrial",
    5: "cereals",
    6: "oilseed_industrial",
    7: "vineyard",
    8: "sugar_beet",
    9: "cereals",
    10: "cereals",
    11: "vegetables",
    12: "potato",
    13: "legumes_fodder",
    14: "soybean",
    15: "orchard",
    16: "cereals",
    17: "cereals",
}


def build_semantic18_to_macro_lut(ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Build the ``semantic18_id -> contiguous macro id`` projection LUT.

    Composes :data:`_SEMANTIC18_TO_MACRO_NAME` (id -> macro name, US-074
    crosswalk) with :data:`ml.data.sen4agrinet_adapter.MACRO_GROUP_TO_ID` (macro
    name -> contiguous id) so the France 18-class predictions land in the EXACT
    same macro label-space the Catalonia target uses (apples-to-apples Delta mIoU).

    Args:
        ignore_index: Fallback id for any unmapped class (never expected for the
            18 crop classes; the guard below would raise first).

    Returns:
        ``int64`` numpy array of length 18 where ``lut[sid]`` is the macro id of
        semantic18 class ``sid``.

    Raises:
        KeyError: if a semantic18 id maps to a macro name absent from
            ``MACRO_GROUP_TO_ID`` (crosswalk drift guard).
    """
    lut = np.full(_FR_NUM_CLASSES, ignore_index, dtype=np.int64)
    for sid, macro in _SEMANTIC18_TO_MACRO_NAME.items():
        if macro not in MACRO_GROUP_TO_ID:
            raise KeyError(
                f"semantic18 id {sid} maps to macro {macro!r} which is absent "
                "from MACRO_GROUP_TO_ID; re-derive from the US-074 crosswalk."
            )
        lut[sid] = MACRO_GROUP_TO_ID[macro]
    return lut


#: France(18) -> macro projection LUT (module singleton).
SEMANTIC18_TO_MACRO: np.ndarray = build_semantic18_to_macro_lut()


@dataclass(frozen=True)
class TransferResult:
    """Outcome of the France -> Catalonia transfer evaluation.

    Attributes:
        miou_zero_shot: mIoU of the France checkpoint projected to macro, on the
            Catalonia held-out val. Always real, never requires convergence.
        miou_few_shot: mIoU of the finetuned macro model on the same val (``None``
            if the finetune was skipped / did not run).
        delta_miou: ``miou_few_shot - miou_zero_shot`` (``None`` if no few-shot).
        n_val_patches: Number of distinct Catalonia patches in the val split.
        n_val_tiles: Number of 128x128 val sub-patches actually scored.
        n_train_patches: Number of few-shot train patches (k).
        n_train_tiles: Number of 128x128 train sub-patches.
        epochs_run: Epochs the finetune ran (0 if skipped).
        best_ckpt: Path of the finetuned ``best.pt`` (``None`` if skipped).
        zero_shot_metrics: Full zero-shot metric dict (miou/f1/pixel_acc).
        few_shot_metrics: Full few-shot metric dict (``None`` if skipped).
    """

    miou_zero_shot: float
    miou_few_shot: float | None
    delta_miou: float | None
    n_val_patches: int
    n_val_tiles: int
    n_train_patches: int
    n_train_tiles: int
    epochs_run: int
    best_ckpt: str | None
    zero_shot_metrics: dict[str, float]
    few_shot_metrics: dict[str, float] | None


def _resolve_device(requested: str) -> torch.device:
    """Resolve the device, preferring CUDA when available.

    Args:
        requested: ``"cuda"``, ``"cpu"`` or ``"auto"``.

    Returns:
        Resolved :class:`torch.device` (CUDA degrades to CPU with a warning).
    """
    if requested in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            logger.warning("cuda_requested_but_unavailable_fallback_cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _patch_level_split(
    dataset: Sen4AgriNetDataset, *, k: int, seed: int
) -> tuple[list[int], list[int], list[Path], list[Path]]:
    """Split the dataset's tile items into few-shot train / held-out val.

    The split is done at the PATCH level (not the tile level): ``k`` patches are
    drawn for the few-shot train set and ALL their tiles go to train; every tile
    of the remaining patches goes to val. This prevents a train patch's tile from
    leaking into val (spatial honesty for the Delta mIoU).

    Args:
        dataset: Indexed :class:`Sen4AgriNetDataset` (Catalonia only).
        k: Number of few-shot train patches.
        seed: RNG seed for the deterministic patch shuffle.

    Returns:
        Tuple ``(train_idx, val_idx, train_patches, val_patches)`` of tile-index
        lists into ``dataset.items`` and the distinct patch paths per split.

    Raises:
        ValueError: if there are fewer than ``k + 1`` distinct patches (need at
            least one held-out val patch).
    """
    # Distinct patches in their indexing order (items are grouped per patch).
    patches: list[Path] = []
    for path, _r, _c in dataset.items:
        if not patches or patches[-1] != path:
            if path not in patches:
                patches.append(path)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patches))
    if len(patches) < k + 1:
        raise ValueError(
            f"need at least k+1={k + 1} distinct Catalonia patches for a held-out "
            f"val, found {len(patches)}. Lower --k."
        )
    train_set = {patches[i] for i in order[:k]}
    train_patches = [patches[i] for i in order[:k]]
    val_patches = [patches[i] for i in order[k:]]
    train_idx = [i for i, (p, _r, _c) in enumerate(dataset.items) if p in train_set]
    val_idx = [i for i, (p, _r, _c) in enumerate(dataset.items) if p not in train_set]
    return train_idx, val_idx, train_patches, val_patches


@torch.no_grad()
def evaluate_zero_shot(
    fr_ckpt: Path,
    val_ds: Dataset,
    val_idx: list[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the France(18) checkpoint on Catalonia, projected to macro.

    Rebuilds the France TSViT at its NATIVE 18 classes, loads ``best.pt``
    (``strict=False`` tolerates buffer mismatches) and, for every val tile,
    predicts the 18-class map and projects it 18 -> macro with
    :data:`SEMANTIC18_TO_MACRO` BEFORE accumulating against the macro target. The
    accumulator ignores ``255`` (background / out-of-nomenclature), so only real
    crop pixels score.

    Args:
        fr_ckpt: Path to the France ``tsvit-pheno-v1/best.pt``.
        val_ds: Indexed Catalonia :class:`Sen4AgriNetDataset`.
        val_idx: Tile indices of the held-out val split.
        device: Inference device.

    Returns:
        Metric dict ``{"miou", "f1_macro", "pixel_accuracy"}`` over the macro
        label-space.
    """
    model = build_tsvit(
        num_classes=_FR_NUM_CLASSES,
        n_timesteps=_FR_N_TIMESTEPS,
        img_size=_TILE_SIZE,
        in_channels=10,
        semantic_dim=384,
    )
    ckpt = torch.load(fr_ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    logger.info(
        "zero_shot_model_loaded",
        ckpt=str(fr_ckpt),
        n_missing=len(missing),
        n_unexpected=len(unexpected),
    )

    lut = torch.as_tensor(SEMANTIC18_TO_MACRO, device=device)
    acc = DenseConfusionAccumulator(N_MACRO_CLASSES, ignore_index=IGNORE_INDEX, device=str(device))
    for i in val_idx:
        x, y = val_ds[i]  # type: ignore[index]
        pred_18 = predict_patch_for_kind(model, x, model_kind="tsvit-pheno")
        pred_macro = lut[torch.as_tensor(pred_18, device=device).clamp(0, 17)]
        acc.update(pred_macro, y.to(device))
    return acc.compute()


@torch.no_grad()
def evaluate_few_shot(
    model: torch.nn.Module,
    val_ds: Dataset,
    val_idx: list[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the finetuned macro model on Catalonia val (no projection).

    The finetuned head already outputs the macro label-space, so predictions are
    accumulated directly against the macro target.

    Args:
        model: Finetuned macro TSViT in ``eval()``.
        val_ds: Indexed Catalonia dataset.
        val_idx: Tile indices of the held-out val split.
        device: Inference device.

    Returns:
        Metric dict ``{"miou", "f1_macro", "pixel_accuracy"}``.
    """
    model.eval()
    acc = DenseConfusionAccumulator(N_MACRO_CLASSES, ignore_index=IGNORE_INDEX, device=str(device))
    for i in val_idx:
        x, y = val_ds[i]  # type: ignore[index]
        pred = predict_patch_for_kind(model, x, model_kind="tsvit-pheno")
        acc.update(torch.as_tensor(pred, device=device), y.to(device))
    return acc.compute()


def build_macro_model_from_fr(
    fr_ckpt: Path, *, linear_probe: bool, device: torch.device
) -> torch.nn.Module:
    """Build a macro TSViT seeded with the France encoder (head reinitialized).

    Constructs ``build_tsvit(num_classes=N_MACRO_CLASSES, ...)`` and loads the
    France ``state_dict`` with ``strict=False``. The 18-class
    ``temporal_cls_tokens`` (shape ``(1, 18, dim)``) and any class-count-dependent
    parameter do NOT match the 10-class model and are skipped (kept at their fresh
    init = the new macro head); every shape-compatible parameter (patch embedding,
    temporal / spatial transformers, positional encodings, the class-agnostic
    ``to_seg`` projection) transfers from France. If ``linear_probe`` is set, the
    whole transferred encoder is frozen and only the reinitialized head trains.

    Args:
        fr_ckpt: Path to the France checkpoint.
        linear_probe: If ``True`` freeze every transferred (loaded) parameter; the
            reinitialized macro params (the ones in ``missing``) stay trainable.
        device: Target device.

    Returns:
        The macro TSViT ready for the finetune.
    """
    model = build_tsvit(
        num_classes=N_MACRO_CLASSES,
        n_timesteps=_FR_N_TIMESTEPS,
        img_size=_TILE_SIZE,
        in_channels=10,
        semantic_dim=384,
    )
    ckpt = torch.load(fr_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    # Drop class-count-dependent params so load_state_dict(strict=False) keeps the
    # fresh macro init for the head instead of warning on every shape mismatch.
    macro_sd = model.state_dict()
    transfer = {k: v for k, v in state.items() if k in macro_sd and v.shape == macro_sd[k].shape}
    missing = [k for k in macro_sd if k not in transfer]
    model.load_state_dict(transfer, strict=False)
    logger.info(
        "macro_model_seeded_from_fr",
        ckpt=str(fr_ckpt),
        n_transferred=len(transfer),
        n_reinitialized=len(missing),
        reinitialized=missing,
        linear_probe=linear_probe,
    )
    if linear_probe:
        for name, param in model.named_parameters():
            param.requires_grad = name not in missing
    return model.to(device)


def run_transfer(
    *,
    root: Path,
    fr_ckpt: Path,
    k: int = 10,
    epochs: int = 40,
    batch_size: int = 8,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
    linear_probe: bool = False,
    warmup_epochs: int = 5,
    seed: int = 17,
    device: str = "cuda",
    num_workers: int = 0,
    ckpt_dir: Path | None = None,
    mlflow_uri: str | None = None,
    skip_finetune: bool = False,
) -> TransferResult:
    """Run the full France -> Catalonia transfer protocol and report Delta mIoU.

    Args:
        root: Sen4AgriNet subset root (holds the ``.nc`` patches).
        fr_ckpt: France ``tsvit-pheno-v1/best.pt`` checkpoint.
        k: Few-shot train patches (e.g. 10 or 50). The rest are held-out val.
        epochs: Finetune epochs.
        batch_size: DataLoader batch size.
        encoder_lr: LR for the transferred encoder params (low, few-shot).
        head_lr: LR for the reinitialized macro head params (normal).
        linear_probe: If ``True`` freeze the encoder (cheap variant).
        warmup_epochs: Linear-warmup epochs before cosine decay.
        seed: RNG seed for the patch-level split.
        device: ``"cuda"`` / ``"cpu"`` / ``"auto"``.
        num_workers: DataLoader workers.
        ckpt_dir: Output dir for the finetuned ``best.pt`` / ``last.pt``.
        mlflow_uri: Override of the MLflow tracking URI.
        skip_finetune: If ``True`` only the zero-shot eval runs (plan B / smoke).

    Returns:
        A :class:`TransferResult` with the zero-shot mIoU (always), the few-shot
        mIoU and the Delta (when the finetune ran).

    Raises:
        FileNotFoundError: if ``root`` or ``fr_ckpt`` is missing.
    """
    if not Path(root).exists():
        raise FileNotFoundError(f"Sen4AgriNet root not found: {root}")
    if not Path(fr_ckpt).exists():
        raise FileNotFoundError(f"France checkpoint not found: {fr_ckpt}")

    dev = _resolve_device(device)
    cat_ds = Sen4AgriNetDataset(
        root=Path(root),
        n_timesteps=_FR_N_TIMESTEPS,
        tile_size=_TILE_SIZE,
        countries=("ES",),
        precache_all=True,  # decode the ~30 ES patches once (shuffle-safe, ~9x).
    )
    train_idx, val_idx, train_patches, val_patches = _patch_level_split(cat_ds, k=k, seed=seed)
    logger.info(
        "transfer_split",
        n_train_patches=len(train_patches),
        n_val_patches=len(val_patches),
        n_train_tiles=len(train_idx),
        n_val_tiles=len(val_idx),
    )

    # --- Step 1: zero-shot (always reportable, no convergence required) -------
    zs = evaluate_zero_shot(Path(fr_ckpt), cat_ds, val_idx, device=dev)
    logger.info("zero_shot_done", **{f"zs_{k_}": round(v, 4) for k_, v in zs.items()})

    if skip_finetune:
        return TransferResult(
            miou_zero_shot=zs["miou"],
            miou_few_shot=None,
            delta_miou=None,
            n_val_patches=len(val_patches),
            n_val_tiles=len(val_idx),
            n_train_patches=len(train_patches),
            n_train_tiles=len(train_idx),
            epochs_run=0,
            best_ckpt=None,
            zero_shot_metrics=zs,
            few_shot_metrics=None,
        )

    # --- Step 2: few-shot finetune (macro head + France encoder) --------------
    from torch.utils.data import Subset

    model = build_macro_model_from_fr(Path(fr_ckpt), linear_probe=linear_probe, device=dev)
    # Two LR groups: low for the transferred encoder, normal for the new head.
    # The head params are exactly the ones reinitialized (not loaded from FR).
    fr_state = torch.load(Path(fr_ckpt), map_location="cpu", weights_only=False)
    fr_sd = fr_state["model_state"] if "model_state" in fr_state else fr_state
    macro_sd = model.state_dict()
    head_param_names = {
        n for n in macro_sd if n not in fr_sd or fr_sd[n].shape != macro_sd[n].shape
    }
    encoder_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n not in head_param_names
    ]
    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n in head_param_names
    ]
    logger.info(
        "transfer_param_groups",
        n_encoder=len(encoder_params),
        n_head=len(head_params),
        encoder_lr=encoder_lr,
        head_lr=head_lr,
    )

    train_subset = Subset(cat_ds, train_idx)
    val_subset = Subset(cat_ds, val_idx)
    # The shared train_segmentation builds a single-LR AdamW; the few-shot protocol
    # needs two LR groups (encoder low + head normal), so _train_two_group mirrors
    # its loss/schedule/checkpoint/MLflow contract with a two-group optimizer.
    resolved_ckpt_dir = (
        Path(ckpt_dir)
        if ckpt_dir is not None
        else Path("checkpoints/segmentation") / "tsvit-pheno-sen4agri-cat-ft-v1"
    )
    start = time.perf_counter()
    fs = _train_two_group(
        model,
        train_subset,
        val_subset,
        encoder_params=encoder_params,
        head_params=head_params,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        device=dev,
        num_workers=num_workers,
        ckpt_dir=resolved_ckpt_dir,
        mlflow_uri=mlflow_uri,
        k=k,
        linear_probe=linear_probe,
    )
    train_time_s = time.perf_counter() - start
    logger.info(
        "few_shot_done",
        train_time_s=round(train_time_s, 1),
        **{f"fs_{k_}": round(v, 4) for k_, v in fs.items()},
    )

    delta = fs["miou"] - zs["miou"]
    return TransferResult(
        miou_zero_shot=zs["miou"],
        miou_few_shot=fs["miou"],
        delta_miou=delta,
        n_val_patches=len(val_patches),
        n_val_tiles=len(val_idx),
        n_train_patches=len(train_patches),
        n_train_tiles=len(train_idx),
        epochs_run=epochs,
        best_ckpt=str(resolved_ckpt_dir / "best.pt"),
        zero_shot_metrics=zs,
        few_shot_metrics=fs,
    )


def _train_two_group(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    encoder_params: list[torch.nn.Parameter],
    head_params: list[torch.nn.Parameter],
    encoder_lr: float,
    head_lr: float,
    epochs: int,
    batch_size: int,
    warmup_epochs: int,
    device: torch.device,
    num_workers: int,
    ckpt_dir: Path,
    mlflow_uri: str | None,
    k: int,
    linear_probe: bool,
) -> dict[str, float]:
    """Two-LR-group finetune loop (Dice+CE, warmup+cosine, best.pt, MLflow).

    Mirrors :func:`ml.train.train_segmentation.train_segmentation` (same loss,
    schedule shape, checkpointing and MLflow ``track_experiment`` tags) but uses a
    two-param-group AdamW (encoder LR low + head LR normal) which the shared helper
    does not expose. Keeps the macro-HCAT ``ignore_index=255`` and the
    ``num_classes=N_MACRO_CLASSES`` evaluation via the dense accumulator.

    Args:
        model: Macro TSViT (encoder seeded from France, head reinitialized).
        train_ds: Few-shot train subset (Catalonia).
        val_ds: Held-out val subset (Catalonia).
        encoder_params: Transferred encoder parameters (LR ``encoder_lr``).
        head_params: Reinitialized head parameters (LR ``head_lr``).
        encoder_lr: Encoder learning rate.
        head_lr: Head learning rate.
        epochs: Finetune epochs.
        batch_size: DataLoader batch size.
        warmup_epochs: Linear-warmup epochs.
        device: Training device.
        num_workers: DataLoader workers.
        ckpt_dir: Output dir for ``best.pt`` / ``last.pt``.
        mlflow_uri: MLflow tracking URI override.
        k: Few-shot k (logged as a param).
        linear_probe: Whether the encoder is frozen (logged as a param).

    Returns:
        Best-epoch metric dict ``{"miou", "f1_macro", "pixel_accuracy"}``.
    """
    import mlflow
    from torch.utils.data import DataLoader

    from ml.models.deeplabv3plus import build_dice_ce_loss
    from ml.utils.mlflow_utils import track_experiment

    param_groups: list[dict[str, object]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups)

    criterion = build_dice_ce_loss(ignore_index=IGNORE_INDEX, n_classes=N_MACRO_CLASSES).to(device)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

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
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=5e-6
            ),
        ],
        milestones=[max(1, warmup_epochs)],
    )

    loader_kwargs: dict[str, Any] = {"pin_memory": device.type == "cuda"}
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

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, float] = {"miou": -1.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}

    with track_experiment(
        "agrosat-segmentation",
        run_name="tsvit-pheno-sen4agri-cat-ft-v1",
        tracking_uri=mlflow_uri,
        dvc_path="data/sen4agrinet",
    ):
        mlflow.set_tag("architecture", "tsvit-pheno")
        mlflow.set_tag("transfer", "france_to_catalonia")
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "encoder_lr": encoder_lr,
                "head_lr": head_lr,
                "linear_probe": linear_probe,
                "k_few_shot": k,
                "num_classes": N_MACRO_CLASSES,
                "ignore_index": IGNORE_INDEX,
                "warmup_epochs": warmup_epochs,
                "device": str(device),
            }
        )
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion(logits, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                epoch_loss += float(loss.detach().item())
                n_batches += 1
            train_loss = epoch_loss / max(1, n_batches)

            val_metrics = evaluate_few_shot(
                model,
                val_ds,
                list(range(len(cast("Sized", val_ds)))),
                device=device,  # type: ignore[arg-type]
            )
            current_lr = optimizer.param_groups[-1]["lr"]
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_miou", val_metrics["miou"], step=epoch)
            mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)
            mlflow.log_metric("val_pixel_accuracy", val_metrics["pixel_accuracy"], step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
            logger.info(
                "transfer_epoch",
                epoch=epoch + 1,
                lr=round(current_lr, 7),
                train_loss=round(train_loss, 4),
                val_miou=round(val_metrics["miou"], 4),
                val_f1_macro=round(val_metrics["f1_macro"], 4),
            )

            if val_metrics["miou"] > best_metrics["miou"]:
                best_metrics = dict(val_metrics)
                best_metrics["best_epoch"] = float(epoch + 1)
                _payload = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_metrics": best_metrics,
                }
                torch.save(_payload, ckpt_dir / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_metrics": best_metrics,
                },
                ckpt_dir / "last.pt",
            )
            scheduler.step()

    return {
        "miou": best_metrics["miou"],
        "f1_macro": best_metrics["f1_macro"],
        "pixel_accuracy": best_metrics["pixel_accuracy"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments for the transfer finetune.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).

    Returns:
        The parsed ``argparse.Namespace``.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/sen4agrinet"))
    p.add_argument(
        "--fr-ckpt",
        type=Path,
        default=Path("checkpoints/segmentation/tsvit-pheno-v1/best.pt"),
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--mlflow-uri", type=str, default=None)
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Only run the zero-shot eval (plan B / smoke).",
    )
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the transfer and print the result as JSON.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).
    """
    args = _parse_args(argv)
    result = run_transfer(
        root=args.root,
        fr_ckpt=args.fr_ckpt,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        linear_probe=args.linear_probe,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        ckpt_dir=args.ckpt_dir,
        mlflow_uri=args.mlflow_uri,
        skip_finetune=args.skip_finetune,
    )
    payload = asdict(result)
    payload["macro_groups"] = [g for g in MACRO_HCAT_GROUPS if g != "void"]
    print("TRANSFER_RESULT_JSON " + json.dumps(payload))
    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "transfer_summary",
        miou_zero_shot=round(result.miou_zero_shot, 4),
        miou_few_shot=(
            round(result.miou_few_shot, 4) if result.miou_few_shot is not None else None
        ),
        delta_miou=(round(result.delta_miou, 4) if result.delta_miou is not None else None),
        n_val_patches=result.n_val_patches,
        n_train_patches=result.n_train_patches,
    )


if __name__ == "__main__":
    main()
