"""Semantic branch + phenology contrastive loss for TSViT (Wen et al. 2025).

Implements the **semantic branch** and the **contrastive visual-pixel <->
class-prototype alignment** of the method by Wen et al. (2025), "Phenology
Description is All You Need!" (ISPRS J. Photogrammetry RS 228), equations 15-16.
Unlike the tabular baseline (which concatenated ``pheno_text`` as extra columns
and degraded), the paper **does not concatenate**: it aligns by contrast the
visual feature of each pixel with the semantic prototype of ITS class (positive)
against the other 17 prototypes (negatives). Paper ablation (Table 2, zero-shot
area 1): patches only F1 26.8 -> patches + phenology F1 53.4.

Components:

1. :class:`PhenoSemanticBranch`: loads the matrix of 18 per-class prototypes
   (384-dim) from :func:`ml.features.phenology_class_prototypes.\
   load_class_prototype_embeddings`, projects them to a common space
   (``Linear 384 -> semantic_dim``) and L2-normalizes them. Exposes
   :meth:`get_class_prototypes` which returns the matrix ``(num_classes, D)``
   ready for the contrast. The projection to the common space replaces the GCN
   text branch of the paper (§3.2); the GCN over phenological keywords remains a
   post-Avance TODO (see note at the end of the module).

2. :func:`phenology_contrastive_loss`: symmetric CLIP-style InfoNCE (eq. 15-16,
   ``L_cl = (L_v + L_s)/2``) between the per-pixel visual features (from
   :meth:`ml.models.tsvit_wrapper.TSViT.forward` with ``return_visual_proj=True``)
   and the per-class prototypes. Subsamples valid pixels to bound memory.

Attribution: method by Wen et al. (2025), ISPRS J. Photogrammetry RS 228.
Documented in ``docs/licenses/DATA_LICENSE.md``.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from torch import nn

from ml.features.phenology_class_prototypes import (
    load_class_prototype_embeddings,
)

__all__ = [
    "PhenoSemanticBranch",
    "phenology_contrastive_loss",
]

logger = structlog.get_logger(__name__)

#: Dimension of the phenological text prototypes (``all-MiniLM-L6-v2``).
_PROTOTYPE_DIM = 384

#: Maximum number of valid pixels sampled per batch for the contrast. Bounds the
#: similarity matrix (n_sampled x num_classes) and the autograd graph to something
#: that fits in L4 24GB; the sampling is stochastic per step (see Wen §3.3).
_DEFAULT_MAX_PIXELS = 4096


class PhenoSemanticBranch(nn.Module):
    """Semantic branch: per-class phenology prototypes projected and L2-norm.

    Loads the matrix of per-class text prototypes (``num_classes``, 384) and
    projects it to a common space of dimension ``semantic_dim`` via a learnable
    linear layer, L2-normalizing the output. The per-pixel visual feature of
    TSViT (``return_visual_proj=True``) lives in this same space, which enables
    the contrastive alignment of equation 15-16 of the paper.

    The raw prototypes are registered as a buffer (non-trainable, travels with
    the ``state_dict`` and to the module device); only the projection is
    learnable. This preserves the semantics of the frozen text encoder and lets
    the model learn only how to map that semantics to the visual space.

    Args:
        semantic_dim: Dimension of the common alignment space. It must match
            ``semantic_dim`` of :class:`ml.models.tsvit_wrapper.TSViT`
            (384 by default, same as the prototypes, which makes the projection a
            refinement and not a dimensionality change).
        prototype_path: Path to the per-class prototypes parquet. If ``None``
            uses the default of
            :func:`ml.features.phenology_class_prototypes.\
            load_class_prototype_embeddings`.
        freeze_prototypes: If ``True`` (default) the raw prototypes are a
            non-trainable buffer; if ``False`` they are registered as a parameter
            and fine-tuned together with the projection.

    Raises:
        ValueError: If the loaded prototype matrix does not have dimension 384.
    """

    def __init__(
        self,
        semantic_dim: int = _PROTOTYPE_DIM,
        prototype_path: Path | None = None,
        *,
        freeze_prototypes: bool = True,
    ) -> None:
        super().__init__()
        if prototype_path is None:
            prototypes_np, class_ids = load_class_prototype_embeddings()
        else:
            prototypes_np, class_ids = load_class_prototype_embeddings(prototype_path)
        if prototypes_np.shape[1] != _PROTOTYPE_DIM:
            raise ValueError(
                f"The prototypes must be {_PROTOTYPE_DIM}-dim; loaded shape {prototypes_np.shape}."
            )

        self.num_classes = int(prototypes_np.shape[0])
        self.semantic_dim = semantic_dim
        self.class_ids = list(class_ids)

        prototypes = torch.from_numpy(prototypes_np).float()  # (K, 384)
        if freeze_prototypes:
            self.register_buffer("raw_prototypes", prototypes)
        else:
            self.raw_prototypes = nn.Parameter(prototypes)

        # Projection to the common alignment space (semantic branch of the paper,
        # §3.2; replaces the keyword GCN with a linear layer for simplicity).
        self.proj = nn.Linear(_PROTOTYPE_DIM, semantic_dim)

        logger.info(
            "pheno_semantic_branch_init",
            num_classes=self.num_classes,
            semantic_dim=semantic_dim,
            freeze_prototypes=freeze_prototypes,
        )

    def get_class_prototypes(self) -> torch.Tensor:
        """Return the projected and L2-normalized prototypes.

        Returns:
            Tensor ``(num_classes, semantic_dim)`` float, normalized per row
            (unit L2 norm), on the module device. Ready to be used as
            ``prototypes`` in :func:`phenology_contrastive_loss`.
        """
        projected = self.proj(self.raw_prototypes)  # (K, semantic_dim)
        return F.normalize(projected, p=2, dim=-1)

    def forward(self) -> torch.Tensor:
        """Alias of :meth:`get_class_prototypes` (``nn.Module`` interface).

        Returns:
            Projected and L2-normalized prototypes ``(num_classes,
            semantic_dim)``.
        """
        return self.get_class_prototypes()


def phenology_contrastive_loss(
    visual_proj: torch.Tensor,
    target: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    ignore_index: int = 255,
    temperature: float = 0.07,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE phenology contrastive loss (Wen et al. 2025, eq 15-16).

    For each valid pixel (``target != ignore_index``) it aligns its visual
    feature with the prototype of ITS class (positive) against the other
    prototypes (negatives). Two symmetric CLIP-style InfoNCE terms are computed
    and averaged (``L_cl = (L_v + L_s)/2``):

    - ``L_v`` (visual->semantic): for each pixel, softmax over the similarities
      with the ``num_classes`` prototypes; the positive is the prototype of its
      class.
    - ``L_s`` (semantic->visual): for each class **present** in the sampled
      batch, softmax over the similarities with all the sampled pixels; the
      positives are the pixels of that class (multi-positive, log-mean-exp of the
      positives).

    The visual features and the prototypes are L2-normalized, the similarity is
    the dot product divided by ``temperature``. If the valid pixels exceed
    ``max_pixels`` they are stochastically subsampled to bound the similarity
    matrix and the autograd graph (L4 memory).

    Args:
        visual_proj: Per-pixel visual features ``(B, D, H, W)`` (from TSViT with
            ``return_visual_proj=True``).
        target: Per-pixel class ``(B, H, W)`` int. Values in ``[0,
            num_classes)`` index ``prototypes``; ``ignore_index`` is discarded.
        prototypes: Per-class prototypes ``(num_classes, D)``; ideally already
            L2-normalized (renormalized for robustness).
        ignore_index: Value of ``target`` to ignore (Background/Void).
        temperature: Temperature ``tau`` of the InfoNCE softmax (0.07, standard
            CLIP/SimCLR).
        max_pixels: Maximum number of valid pixels to sample per call.
        generator: Optional ``torch.Generator`` for deterministic sampling
            (tests/smoke); if ``None`` uses the global RNG.

    Returns:
        Scalar loss ``(L_v + L_s)/2``. Returns ``0.0`` (with graph) if there are
        no valid pixels or if only one class is present (undefined contrast).
    """
    if visual_proj.dim() != 4:
        raise ValueError(f"visual_proj must be (B, D, H, W); received {tuple(visual_proj.shape)}.")
    num_classes, dim = prototypes.shape
    device = visual_proj.device

    # (B, D, H, W) -> (B*H*W, D) and target -> (B*H*W,)
    feats = visual_proj.permute(0, 2, 3, 1).reshape(-1, dim)  # (P, D)
    labels = target.reshape(-1).to(device)  # (P,)

    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    if not bool(valid.any()):
        return visual_proj.sum() * 0.0

    feats = feats[valid]
    labels = labels[valid].long()

    # Stochastic subsampling to bound memory.
    n_valid = feats.shape[0]
    if n_valid > max_pixels:
        perm = torch.randperm(n_valid, device=device, generator=generator)
        idx = perm[:max_pixels]
        feats = feats[idx]
        labels = labels[idx]

    # Contrast undefined with a single class present.
    if labels.unique().numel() < 2:
        return visual_proj.sum() * 0.0

    feats = F.normalize(feats, p=2, dim=-1)  # (S, D)
    protos = F.normalize(prototypes.to(device), p=2, dim=-1)  # (K, D)

    # Pixel-prototype logits: (S, K).
    logits = (feats @ protos.t()) / temperature

    # --- L_v: visual -> semantic (classify the pixel to its prototype) ---
    loss_v = F.cross_entropy(logits, labels)

    # --- L_s: semantic -> visual (each present prototype attracts its pixels) -
    # Transpose: (K, S). For each present class, the positives are the
    # pixels of that class; log-mean-exp of the positives is used (multi-positive
    # supervised-InfoNCE style, robust to the variable number of positives).
    logits_s = logits.t()  # (K, S)
    log_prob_s = F.log_softmax(logits_s, dim=1)  # (K, S) over the pixels
    present = torch.unique(labels)
    pos_mask = present.unsqueeze(1) == labels.unsqueeze(0)  # (K_present, S)
    log_prob_present = log_prob_s[present]  # (K_present, S)
    # Mean log-prob over the positive pixels of each present class.
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)  # (K_present,)
    pos_log_prob = (log_prob_present * pos_mask).sum(dim=1) / pos_counts
    loss_s = -pos_log_prob.mean()

    loss: torch.Tensor = 0.5 * (loss_v + loss_s)
    return loss


# TODO(post-Avance): text branch with a GCN over phenological keywords (Wen et al.
# 2025, §3.2). The paper builds a graph over the phenological terms extracted from
# the descriptions (sowing/emergence/peak/senescence/harvest) and propagates with
# a GCN before the projection, instead of the direct linear layer of
# :class:`PhenoSemanticBranch`. Here the linear projection is used for simplicity
# and to keep training viable within the L4 window; the GCN is a paper-fidelity
# improvement that does not block the Avance (deferred risk/benefit trade-off).
