"""Real transfer learning: fine-tune the dense backbone to the Baltic vocabulary.

This is the corrected experiment Arthur asked for. The earlier texture run used the
dense models FROZEN as feature extractors and trained only a head on top -- that is
*feature extraction*, not transfer learning, and it failed because the frozen
backbone kept emitting its PASTIS-France vocabulary regardless of the target. The
verdict (negative) showed the bottleneck is the VOCABULARY, not texture.

Real TL with a "kept-class flag"
--------------------------------
1. Build the dense model (TSViT / U-TAE) with a NEW head sized for the Baltic
   label space -- the six classes that map to PASTIS ("kept" / conserved) PLUS the
   new fine leaves the EDA surfaced (apples, quinces, fresh_vegetables, clover...).
2. Initialise the BACKBONE from the PASTIS checkpoint (what it learned about
   phenology / temporal structure in France), and -- the kept-class flag -- warm
   the new head's rows for the conserved classes from the corresponding PASTIS head
   rows, so the model does not forget the classes it already knew. New-class rows
   start random.
3. UNFREEZE the backbone (optionally a few warmup epochs head-only first) and
   fine-tune backbone + head on the Baltic parcels, so the representation ADAPTS to
   the new vocabulary instead of being forced through the French one.

Per-parcel (not dense)
----------------------
EuroCropsML labels are per-PARCEL (one crop per field), not per-pixel masks. The
dense models output ``(B, K, H, W)``; here we global-average-pool the logits over
``H, W`` to a single ``(B, K)`` per-parcel prediction and train with a plain
cross-entropy on the parcel label. The backbone (temporal+spatial encoders) is
unchanged; only the read-out is per-parcel.

Hierarchical eval (the papaya/fruits hypothesis, measurable)
------------------------------------------------------------
The fine model predicts the FINE leaf (e.g. ``apples``). To compare fairly against
a model that only knows the coarse PASTIS bucket, predictions are also COLLAPSED to
the level each dataset labels (``apples`` -> ``Orchard``/``Fruits``) via the
crosswalk, and F1 is reported at BOTH levels. The qualitative demo: a parcel PASTIS
would call "Fruits, vegetables, flowers" the enriched model calls "apples".

Honesty
-------
- No fabricated numbers; if a checkpoint or band mismatch prevents a warm start it
  is logged and that head row stays random (reported), never silently faked.
- Runs on the H100 (the backbone unfreeze is a real GPU train). Subset + epochs are
  parameters so a pilot is cheap and the full run is one flag away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

logger = structlog.get_logger(__name__)

__all__ = [
    "BalticLabelSpace",
    "FineTuneConfig",
    "build_baltic_label_space",
    "build_finetune_model",
    "run_finetune",
    "stratified_parcel_sample",
    "warm_start_head",
]

#: Conserved classes: Baltic leaf -> PASTIS-18 class name (the "kept-class flag").
#: These rows of the new head are warm-started from the PASTIS head so the model
#: retains what it already knew. Mirrors ``ml.transfer.transfer_eda._LEAF_TO_PASTIS``
#: restricted to the well-supported Baltic leaves.
CONSERVED_LEAF_TO_PASTIS: dict[str, str] = {
    "pasture_meadow_grassland_grass": "Meadow",
    "potatoes": "Potatoes",
    "winter_rapeseed_rape": "Winter rapeseed",
    "winter_common_soft_wheat": "Soft winter wheat",
    "spring_barley": "Spring barley",
    "winter_barley": "Winter barley",
}

#: New fine leaves the EDA surfaced as well-supported and NOT in PASTIS-18. These
#: are the granularity-enriching classes (the papaya/fruits hypothesis). Rows start
#: random. Ordered by EDA support.
NEW_FINE_LEAVES: tuple[str, ...] = (
    "clover",
    "apples",
    "fresh_vegetables",
    "legumes_harvested_green",
    "oats",
    "spring_common_soft_wheat",
    "alfalfa_lucerne",
    "orchards_fruits",
    "spring_rapeseed_rape",
    "quinces",
    "summer_rapeseed_rape",
    "rye",
)

#: Collapse map for hierarchical eval: fine new leaf -> coarse PASTIS bucket it
#: refines (so the fine prediction can be scored at the coarse level a model
#: without the granularity would use). A leaf absent here has no coarse parent.
FINE_TO_COARSE: dict[str, str] = {
    "apples": "Fruits, vegetables, flowers",
    "quinces": "Fruits, vegetables, flowers",
    "orchards_fruits": "Orchard",
    "fresh_vegetables": "Fruits, vegetables, flowers",
    "clover": "Leguminous fodder",
    "alfalfa_lucerne": "Leguminous fodder",
    "legumes_harvested_green": "Leguminous fodder",
    "spring_rapeseed_rape": "Winter rapeseed",
    "summer_rapeseed_rape": "Winter rapeseed",
    "spring_common_soft_wheat": "Soft winter wheat",
    "rye": "Mixed cereal",
    "oats": "Mixed cereal",
}


@dataclass
class BalticLabelSpace:
    """The fine-tune target label space (conserved + new), with PASTIS mapping.

    Attributes:
        leaves: All target leaves in fixed order (conserved first, then new).
        conserved: Subset that maps to a PASTIS class (warm-started).
        new: Subset that is genuinely new (random init).
        leaf_to_pastis: Conserved leaf -> PASTIS class name.
        index: leaf -> contiguous class id.
    """

    leaves: tuple[str, ...]
    conserved: tuple[str, ...]
    new: tuple[str, ...]
    leaf_to_pastis: dict[str, str]
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {leaf: i for i, leaf in enumerate(self.leaves)}


@dataclass
class FineTuneConfig:
    """Hyperparameters for the Baltic fine-tune."""

    model_kind: str = "utae"  # "utae" | "tsvit-pheno-fullm"
    head_warmup_epochs: int = 2  # head-only epochs before unfreezing the backbone
    finetune_epochs: int = 8
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4  # smaller LR for the pretrained backbone
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_parcels_per_region: int = 1500
    #: When set, use a per-class (stratified) sample of this many parcels per leaf
    #: instead of a global random draw -- the representative sample the EDA calls
    #: for (rare leaves like ``winter_barley`` / ``rye`` are not starved).
    per_class: int | None = None
    seed: int = 42


def stratified_parcel_sample(
    leaves: list[str],
    *,
    keep: set[str],
    per_class: int,
    seed: int,
) -> list[int]:
    """Return row indices for a per-class (stratified) parcel sample.

    Samples up to ``per_class`` parcels for EACH leaf in ``keep`` (capped at the
    available support), so rare leaves (e.g. ``winter_barley``, ``rye``) are not
    starved by a global random draw that favours the abundant ones. This is the
    representative sample the EDA calls for.

    Args:
        leaves: Per-parcel leaf labels (row-aligned with the source frame).
        keep: The label-space to restrict to.
        per_class: Target parcels per leaf (capped at its support).
        seed: RNG seed.

    Returns:
        Sorted list of selected row indices.
    """
    rng = np.random.default_rng(seed)
    by_leaf: dict[str, list[int]] = {}
    for i, leaf in enumerate(leaves):
        if leaf in keep:
            by_leaf.setdefault(leaf, []).append(i)
    picked: list[int] = []
    for idxs in by_leaf.values():
        take = min(per_class, len(idxs))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())
    return sorted(picked)


def build_baltic_label_space(
    *,
    conserved: dict[str, str] = CONSERVED_LEAF_TO_PASTIS,
    new: tuple[str, ...] = NEW_FINE_LEAVES,
) -> BalticLabelSpace:
    """Assemble the fine-tune label space (conserved classes first, then new).

    Args:
        conserved: Baltic leaf -> PASTIS class for the kept-class flag.
        new: New fine leaves (granularity enrichment).

    Returns:
        A :class:`BalticLabelSpace`.
    """
    conserved_leaves = tuple(conserved.keys())
    leaves = conserved_leaves + tuple(new)
    return BalticLabelSpace(
        leaves=leaves,
        conserved=conserved_leaves,
        new=tuple(new),
        leaf_to_pastis=dict(conserved),
    )


def warm_start_head(
    new_head_weight: np.ndarray,
    new_head_bias: np.ndarray | None,
    pastis_head_weight: np.ndarray,
    pastis_head_bias: np.ndarray | None,
    *,
    label_space: BalticLabelSpace,
    pastis_class_names: dict[int, str],
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Copy PASTIS head rows into the conserved rows of the new head (kept flag).

    For each conserved Baltic class, find its PASTIS class id and copy that row of
    the PASTIS classification head into the new head, so the model starts knowing
    the classes it already learned. New-class rows are left as initialised. This is
    the concrete "kept-class flag": conserved classes are warm-started, new ones
    learn from scratch.

    Args:
        new_head_weight: New head weight ``(K_new, D)`` (modified in place + returned).
        new_head_bias: New head bias ``(K_new,)`` or ``None``.
        pastis_head_weight: PASTIS head weight ``(18, D)``.
        pastis_head_bias: PASTIS head bias ``(18,)`` or ``None``.
        label_space: The Baltic label space.
        pastis_class_names: PASTIS id -> class name.

    Returns:
        ``(weight, bias, warmed_leaves)`` -- the head with conserved rows copied and
        the list of leaves actually warm-started (a conserved leaf whose PASTIS row
        is missing stays random and is omitted, logged).
    """
    name_to_pastis_id = {name: cid for cid, name in pastis_class_names.items()}
    if new_head_weight.shape[1] != pastis_head_weight.shape[1]:
        logger.warning(
            "warm_start_dim_mismatch",
            new_dim=new_head_weight.shape[1],
            pastis_dim=pastis_head_weight.shape[1],
        )
        return new_head_weight, new_head_bias, []
    warmed: list[str] = []
    for leaf in label_space.conserved:
        pastis_name = label_space.leaf_to_pastis.get(leaf)
        pastis_id = name_to_pastis_id.get(pastis_name) if pastis_name else None
        if pastis_id is None:
            logger.info("warm_start_skip_no_pastis_row", leaf=leaf, pastis_name=pastis_name)
            continue
        row = label_space.index[leaf]
        new_head_weight[row] = pastis_head_weight[pastis_id]
        if new_head_bias is not None and pastis_head_bias is not None:
            new_head_bias[row] = pastis_head_bias[pastis_id]
        warmed.append(leaf)
    logger.info("warm_start_done", n_warmed=len(warmed), warmed=warmed)
    return new_head_weight, new_head_bias, warmed


def build_finetune_model(
    label_space: BalticLabelSpace,
    *,
    model_kind: str,
    pastis_checkpoint: str,
    device: str = "cuda",
) -> nn.Module:
    """Build the dense model with a new Baltic head, backbone init from PASTIS.

    Loads the PASTIS checkpoint into the backbone (shared layers), attaches a fresh
    classification head sized for the Baltic label space, and warm-starts the
    conserved rows of that head from the PASTIS head (the kept-class flag). New-class
    head rows stay at their random init.

    Args:
        label_space: The Baltic target label space.
        model_kind: ``"utae"`` or ``"tsvit-pheno-fullm"``.
        pastis_checkpoint: Path to the PASTIS checkpoint (.pt).
        device: Torch device.

    Returns:
        The model ready to fine-tune (on ``device``).

    Raises:
        ValueError: for an unsupported ``model_kind``.
    """
    import torch

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY, resolve_state_dict

    k_new = len(label_space.leaves)
    if model_kind == "utae":
        from ml.models.utae import build_utae

        model: nn.Module = build_utae(num_classes=k_new, input_dim=10)
        head_names: tuple[str, str] | None = ("out_conv.2.weight", "out_conv.2.bias")
    elif model_kind == "tsvit-pheno-fullm":
        from ml.models.tsvit_wrapper import TSVIT_FULLM_CONFIG, build_tsvit

        cfg = {k: v for k, v in TSVIT_FULLM_CONFIG.items() if k != "img_size"}
        model = build_tsvit(num_classes=k_new, in_channels=10, img_size=128, **cfg)
        head_names = None  # TSViT head warm-start is per-cls-token
    else:
        raise ValueError(f"unsupported model_kind {model_kind!r}")

    # Load PASTIS weights non-strictly: the backbone tensors match by name; the head
    # tensors differ in their class dimension and are skipped (loaded into the new
    # head only via the explicit warm start below).
    spec = CHECKPOINT_REGISTRY[model_kind]
    loaded = torch.load(pastis_checkpoint, map_location="cpu", weights_only=False)
    pastis_state = resolve_state_dict(loaded, spec)
    own = model.state_dict()
    compatible = {k: v for k, v in pastis_state.items() if k in own and own[k].shape == v.shape}
    own.update(compatible)
    model.load_state_dict(own, strict=False)
    logger.info(
        "finetune_backbone_init",
        model_kind=model_kind,
        n_loaded=len(compatible),
        n_total=len(own),
        n_classes_new=k_new,
    )

    # Kept-class flag: warm-start the conserved head rows from the PASTIS head.
    if head_names is not None and head_names[0] in pastis_state:
        from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

        head_w_name, head_b_name = head_names
        new_w = model.state_dict()[head_w_name].clone().cpu().numpy()
        new_b = model.state_dict()[head_b_name].clone().cpu().numpy()
        pw = pastis_state[head_w_name].cpu().numpy()
        pb = pastis_state[head_b_name].cpu().numpy()
        # 1x1 conv head: weight is (K, 32, 1, 1) -> flatten to (K, 32) for the copy.
        new_w2 = new_w.reshape(new_w.shape[0], -1)
        pw2 = pw.reshape(pw.shape[0], -1)
        new_w2, new_b, warmed = warm_start_head(
            new_w2,
            new_b,
            pw2,
            pb,
            label_space=label_space,
            pastis_class_names=dict(SEMANTIC18_CLASS_NAMES),
        )
        with torch.no_grad():
            model.state_dict()[head_w_name].copy_(torch.from_numpy(new_w2.reshape(new_w.shape)))
            model.state_dict()[head_b_name].copy_(torch.from_numpy(new_b))
        logger.info("finetune_head_warmstarted", n_warmed=len(warmed))

    model.to(device)
    return model


def run_finetune(
    config: FineTuneConfig,
    *,
    sh_client: object,
    pastis_checkpoint: str,
    source: str = "latvia",
    target: str = "estonia",
    device: str = "cuda",
) -> dict[str, object]:
    """Fine-tune the dense backbone on Baltic parcels and eval fine + coarse.

    Pipeline: build the Baltic label space, download real-texture SH patches for
    source (train) and target (test), build the model (PASTIS backbone init + warm
    head), train head-only for ``head_warmup_epochs`` then unfreeze the backbone for
    ``finetune_epochs``, and report per-class F1 at the FINE level and collapsed to
    the COARSE PASTIS level (the hierarchical / papaya-fruits eval).

    Args:
        config: Fine-tune hyperparameters.
        sh_client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        pastis_checkpoint: PASTIS checkpoint path for the backbone init.
        source: Region trained on.
        target: Region tested on.
        device: Torch device.

    Returns:
        A summary dict with fine + coarse macro F1 and the warm-started classes.
    """
    import torch
    from sklearn.metrics import f1_score
    from torch import nn

    from ml.transfer.ensemble_texture_tl import (
        _load_region_texture,
        build_season_windows,
    )

    label_space = build_baltic_label_space()
    keep = set(label_space.leaves)
    windows = build_season_windows(2021)

    def _load(region: str) -> tuple[list[np.ndarray], np.ndarray]:
        reg = _load_region_texture(
            region,
            sh_client=sh_client,
            windows=windows,
            max_parcels=config.max_parcels_per_region,
            size=128,
            max_cloud=25.0,
            seed=config.seed,
            stratify_keep=keep if config.per_class else None,
            per_class=config.per_class,
        )
        mask = np.array([leaf in keep for leaf in reg.leaf], dtype=bool)
        patches = [p for p, m in zip(reg.patches, mask, strict=True) if m]
        y = np.array([label_space.index[leaf] for leaf in reg.leaf[mask]], dtype=np.int64)
        return patches, y

    train_patches, y_train = _load(source)
    test_patches, y_test = _load(target)
    logger.info(
        "finetune_data_ready",
        n_train=len(train_patches),
        n_test=len(test_patches),
        n_classes=len(label_space.leaves),
    )

    model = build_finetune_model(
        label_space,
        model_kind=config.model_kind,
        pastis_checkpoint=pastis_checkpoint,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()

    def _forward(xb: torch.Tensor) -> torch.Tensor:
        """Dispatch the per-architecture forward (U-TAE needs DOY positions)."""
        t = xb.shape[1]
        if config.model_kind == "utae":
            doy = (torch.arange(t, device=device).float() / max(t - 1, 1) * 364.0).round().long()
            positions = doy.unsqueeze(0).repeat(xb.shape[0], 1)
            utae_logits: torch.Tensor = model(xb, positions)
            return utae_logits
        logits: torch.Tensor = model(xb)
        return logits

    def _is_head(name: str) -> bool:
        return "out_conv" in name or "head" in name or "cls_token" in name

    def _epoch(
        patches: list[np.ndarray], ys: np.ndarray, *, train: bool, opt: torch.optim.Optimizer
    ) -> float:
        model.train(train)
        if train:
            order = np.random.default_rng(config.seed).permutation(len(patches))
        else:
            order = np.arange(len(patches))
        total, correct = 0, 0
        for start in range(0, len(order), config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            xb = torch.from_numpy(np.stack([patches[i] for i in batch_idx])).float().to(device)
            yb = torch.from_numpy(ys[batch_idx]).to(device)
            logits = _forward(xb)  # (B, K, H, W)
            pooled = logits.mean(dim=(2, 3))  # (B, K) per-parcel
            loss = criterion(pooled, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            correct += int((pooled.argmax(1) == yb).sum().item())
            total += yb.numel()
        return correct / max(total, 1)

    # Phase 1: head-only warmup (backbone frozen).
    for name, p in model.named_parameters():
        p.requires_grad = _is_head(name)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr_head)
    for ep in range(config.head_warmup_epochs):
        acc = _epoch(train_patches, y_train, train=True, opt=opt)
        logger.info("finetune_warmup_epoch", epoch=ep, train_acc=round(acc, 4))

    # Phase 2: unfreeze the backbone, smaller LR for it.
    for p in model.parameters():
        p.requires_grad = True
    head_params = [p for n, p in model.named_parameters() if _is_head(n)]
    backbone_params = [p for n, p in model.named_parameters() if not _is_head(n)]
    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.lr_head},
            {"params": backbone_params, "lr": config.lr_backbone},
        ],
        weight_decay=config.weight_decay,
    )
    for ep in range(config.finetune_epochs):
        acc = _epoch(train_patches, y_train, train=True, opt=opt)
        logger.info("finetune_epoch", epoch=ep, train_acc=round(acc, 4))

    # Eval on the target, fine + collapsed-to-coarse.
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(test_patches), config.batch_size):
            xb = (
                torch.from_numpy(np.stack(test_patches[start : start + config.batch_size]))
                .float()
                .to(device)
            )
            pooled = _forward(xb).mean(dim=(2, 3))
            preds.extend(pooled.argmax(1).cpu().numpy().tolist())
    id_to_leaf = {i: leaf for leaf, i in label_space.index.items()}
    pred_leaves = [id_to_leaf[p] for p in preds]
    true_leaves = [id_to_leaf[t] for t in y_test.tolist()]
    fine_f1 = float(f1_score(true_leaves, pred_leaves, average="macro"))

    def _coarse(leaf: str) -> str:
        return FINE_TO_COARSE.get(leaf, label_space.leaf_to_pastis.get(leaf, leaf))

    coarse_true = [_coarse(t) for t in true_leaves]
    coarse_pred = [_coarse(p) for p in pred_leaves]
    coarse_f1 = float(f1_score(coarse_true, coarse_pred, average="macro"))

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    fine_acc = float(accuracy_score(true_leaves, pred_leaves))
    coarse_acc = float(accuracy_score(coarse_true, coarse_pred))
    labels_sorted = sorted(set(true_leaves))
    prec, rec, f1pc, sup = precision_recall_fscore_support(
        true_leaves, pred_leaves, labels=labels_sorted, average=None, zero_division=0
    )
    per_class = [
        {
            "leaf": leaf,
            "is_new": leaf in label_space.new,
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1pc[i]), 4),
            "support": int(sup[i]),
            "coarse": _coarse(leaf),
        }
        for i, leaf in enumerate(labels_sorted)
    ]
    summary = {
        "model_kind": config.model_kind,
        "source": source,
        "target": target,
        "n_train": len(train_patches),
        "n_test": len(test_patches),
        "n_classes_fine": len(label_space.leaves),
        "n_conserved": len(label_space.conserved),
        "n_new": len(label_space.new),
        "fine_macro_f1": round(fine_f1, 4),
        "coarse_macro_f1": round(coarse_f1, 4),
        "fine_accuracy": round(fine_acc, 4),
        "coarse_accuracy": round(coarse_acc, 4),
        "per_class": per_class,
        # Raw predictions so a notebook can rebuild the confusion matrix / examples
        # without re-running the GPU forward.
        "y_true_leaf": true_leaves,
        "y_pred_leaf": pred_leaves,
        "conserved_leaves": list(label_space.conserved),
        "new_leaves": list(label_space.new),
    }
    logger.info(
        "finetune_done",
        **{k: v for k, v in summary.items() if not isinstance(v, list)},
    )
    return summary
