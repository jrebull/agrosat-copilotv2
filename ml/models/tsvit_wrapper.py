"""Factorized TSViT for dense segmentation of Sentinel-2 series.

Clean reimplementation (not a clone of the external repo) of the
Temporal-Spatial Vision Transformer (TSViT) by Tarasiou, Chavez & Zafeiriou
(2023), "ViTs for SITS: Vision Transformers for Satellite Image Time Series"
(arXiv:2301.04944, CVPR 2023). The core idea of TSViT is to **invert the
typical order of video-ViTs**: instead of attending first to space and then to
time, TSViT factorizes the self-attention by applying **the temporal encoder
first** (along the acquisitions axis) and **the spatial encoder afterwards**
(along the patch tokens). This exploits the agronomic structure of the series:
the temporal phenology pattern is the most discriminative signal for the crop
type.

Components (Sections 3.1-3.3 of the paper):

1. **3D patch tokenization** ``(t=1, h, w)``: a ``Conv2d`` with
   ``kernel=stride=patch_size`` is applied independently to each temporal
   image, producing ``N = (H/p) * (W/p)`` spatial tokens per timestep with
   dimension ``dim``.
2. **Temporal positional encoding by real date**: learned table indexed by
   day-of-year (DOY, 1..365) instead of by ordinal position. It accepts the
   ``doy`` of the batch, which makes the model invariant to the number of
   acquisitions and aware of the real date (Section 3.2). If ``doy`` is not
   provided it falls back to a learned ordinal temporal PE.
3. **TEMPORAL encoder**: ``K`` separable cls-tokens (one per class) are
   prepended to the temporal sequence of each spatial position; the
   self-attention runs over the temporal axis. Each cls-token learns to
   summarize the temporal evidence for ITS class (Section 3.3, "multiple cls
   tokens").
4. **SPATIAL encoder**: after the temporal one, a token per
   ``(class, spatial-position)`` is kept; a learned spatial PE is added and the
   self-attention runs over the spatial axis for each class.
5. **Dense segmentation head**: reconstructs logits ``(B, K, H, W)`` by
   projecting each patch token to ``p*p`` and reordering to full resolution.
6. **Contrastive visual branch (US-025 Section A)**: besides the segmentation
   head, it exposes an optional projection of the **per-pixel visual features**
   to the semantic space of dimension ``semantic_dim`` (384, that of the
   phenology prototypes from
   :mod:`ml.features.phenology_class_prototypes`). It enables the contrastive
   visual-pixel <-> class-prototype alignment of the method by Wen et al.
   (2025) without concatenating text to the vector (see Section A.0 of the
   US-025 plan).

Trim for L4 (de-risk, no H100 available today): with ``T=10``, 128px,
``patch_size=8`` (16x16 = 256 spatial tokens) and ``dim=128``,
``depth_temporal=depth_spatial=4`` the model fits comfortably in an L4 24GB
with ``batch=4``. ``dim``/``depth`` are not exaggerated to keep the training
viable within the assigned compute window.

Attribution: architecture by Tarasiou et al. (2023), arXiv:2301.04944
(reference repo ``michaeltrs/DeepSatModels``, Apache-2.0 license). This is an
in-house reimplementation; documented in ``docs/licenses/DATA_LICENSE.md``.
"""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

__all__ = ["TSVIT_FULLM_CONFIG", "TSViT", "build_tsvit"]


#: Full-M TSViT capacity (US-038, H100 96GB). Single source of truth shared by
#: the training orchestrator (``ml.train.train_segmentation.build_and_train``),
#: the re-score harness registry (``ml.eval.checkpoint_registry`` entry
#: ``"tsvit"``) and the tests, so the capacity that is TRAINED, the capacity
#: SAVED in the checkpoint and the capacity the harness REBUILDS to load the
#: ``state_dict`` are byte-identical (US-038 R-HARNESS). ``n_timesteps=37`` equals
#: the REAL PASTIS-R minimum series length (T_MIN=37, T_MAX=61, T_MEDIAN=46): the
#: dataset's ``_equispaced_indices`` only subsamples when ``n_select < T``, so a
#: value above T_MIN leaves short patches at their native T and the per-batch
#: ``torch.stack`` fails ("stack expects each tensor to be equal size"). 37 forces
#: a uniform 37-date equispaced subsample on every patch (stackable batch) while
#: covering the whole phenological cycle (US-038 R-TLEN, fixed against real data).
#: The trained/saved checkpoints therefore carry an ordinal temporal PE of shape
#: ``(1, 37, dim)`` (verified on ``alt-tsvit-fullm-v1`` and
#: ``tsvit-pheno-fullm-v1`` best.pt); the harness MUST rebuild with the same 37 or
#: the ordinal PE misaligns and the re-score mIoU collapses (64/10 mismatch ->
#: 0.17 instead of 0.68, US-039 cierre). These are the keyword arguments fed to
#: :func:`build_tsvit` besides ``num_classes`` and ``in_channels``.
TSVIT_FULLM_CONFIG: dict[str, int] = {
    "n_timesteps": 37,
    "img_size": 128,
    "patch_size": 8,
    "dim": 192,
    "depth_temporal": 6,
    "depth_spatial": 6,
    "heads": 6,
    "dim_head": 64,
    "mlp_ratio": 4,
    "semantic_dim": 384,
}


# ---------------------------------------------------------------------------
# Base Transformer blocks (pre-norm, ViT style)
# ---------------------------------------------------------------------------


class _FeedForward(nn.Module):
    """Two-layer MLP with GELU used inside each Transformer block.

    Args:
        dim: Input and output dimension.
        hidden_dim: Hidden dimension (intermediate expansion).
        dropout: Dropout probability applied after each linear layer.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class _Attention(nn.Module):
    """Scaled multi-head self-attention (scaled dot-product).

    Args:
        dim: Token dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        dropout: Dropout over the projection output.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (flattened_batch, n_tokens, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        projected: torch.Tensor = self.to_out(out)
        return projected


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> Attn -> res; LN -> MLP -> res.

    Args:
        dim: Token dimension.
        heads: Attention heads.
        dim_head: Dimension per head.
        mlp_dim: Hidden dimension of the feed-forward.
        dropout: Dropout in attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class _Transformer(nn.Module):
    """Stack of ``depth`` Transformer blocks with a final LayerNorm.

    Args:
        dim: Token dimension.
        depth: Number of blocks.
        heads: Attention heads.
        dim_head: Dimension per head.
        mlp_dim: Hidden dimension of the feed-forward.
        dropout: Internal dropout.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        normed: torch.Tensor = self.norm(x)
        return normed


# ---------------------------------------------------------------------------
# TSViT
# ---------------------------------------------------------------------------


class TSViT(nn.Module):
    """Factorized Temporal-Spatial ViT for dense segmentation of SITS.

    Implements the architecture of Tarasiou et al. (2023): 3D patch
    tokenization, **temporal-first** encoder with ``K`` separable cls-tokens
    (one per class), **spatial-afterwards** encoder, temporal positional
    encoding by real date (DOY) and a dense segmentation head that reconstructs
    ``(B, K, H, W)``.

    Besides the segmentation head it exposes a **per-pixel visual projection**
    to the semantic space (``semantic_dim``) for the contrastive alignment with
    the per-class phenology prototypes (US-025 Section A, Wen et al. 2025).

    Args:
        num_classes: Number ``K`` of classes; also the number of separable
            temporal cls-tokens and the output channels of the segmentation
            head.
        n_timesteps: Expected temporal length ``T`` (defines the fallback
            ordinal temporal PE when ``doy`` is not provided).
        img_size: Side ``H = W`` of the input patch in pixels.
        in_channels: Number of input bands per timestep (10 for Sentinel-2
            PASTIS-R).
        patch_size: Side of the spatial patch ``p``. Produces
            ``(img_size/p)^2`` spatial tokens per timestep.
        dim: Transformer token dimension.
        depth_temporal: Number of blocks of the temporal encoder.
        depth_spatial: Number of blocks of the spatial encoder.
        heads: Attention heads in both encoders.
        dim_head: Dimension per attention head.
        mlp_ratio: Expansion factor of the feed-forward (``mlp_dim = dim *
            mlp_ratio``).
        semantic_dim: Dimension of the semantic space of the contrastive branch
            (384 to match the ``all-MiniLM-L6-v2`` embeddings of the per-class
            prototypes).
        dropout: Dropout applied in the Transformers.
        max_doy: Maximum day-of-year admitted by the temporal PE table (366 to
            cover leap years; index 0 is unused).

    Raises:
        ValueError: If ``img_size`` is not divisible by ``patch_size``.
    """

    def __init__(
        self,
        num_classes: int = 18,
        n_timesteps: int = 10,
        img_size: int = 128,
        in_channels: int = 10,
        patch_size: int = 8,
        dim: int = 128,
        depth_temporal: int = 4,
        depth_spatial: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        mlp_ratio: int = 4,
        semantic_dim: int = 384,
        dropout: float = 0.0,
        max_doy: int = 366,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )

        self.num_classes = num_classes
        self.n_timesteps = n_timesteps
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.semantic_dim = semantic_dim
        self.max_doy = max_doy

        self.grid = img_size // patch_size  # tokens per side
        self.num_patches = self.grid * self.grid  # N spatial tokens

        mlp_dim = dim * mlp_ratio

        # --- 3D tokenization (t=1, p, p): Conv2d per temporal image ---------
        self.to_patch_embedding = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # --- Temporal positional encoding by DOY (learned table) ------------
        # Indexed by real day-of-year (1..max_doy). Row 0 reserved.
        self.temporal_pos_embedding = nn.Parameter(torch.randn(max_doy + 1, dim) * 0.02)
        # Fallback ordinal temporal PE when doy is not provided.
        self.temporal_pos_ordinal = nn.Parameter(torch.randn(1, n_timesteps, dim) * 0.02)

        # --- K separable temporal cls-tokens (one per class) ----------------
        self.temporal_cls_tokens = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)

        # --- Temporal encoder ----------------------------------------------
        self.temporal_transformer = _Transformer(
            dim, depth_temporal, heads, dim_head, mlp_dim, dropout
        )

        # --- Learned spatial positional encoding ----------------------------
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # --- Spatial encoder -----------------------------------------------
        self.spatial_transformer = _Transformer(
            dim, depth_spatial, heads, dim_head, mlp_dim, dropout
        )

        # --- Dense segmentation head ---------------------------------------
        # Each patch token is projected to p*p values; the reorder reconstructs
        # full resolution. One projection per class keeps the separation
        # of the K cls-tokens.
        self.to_seg = nn.Linear(dim, patch_size * patch_size)

        # --- Contrastive visual branch (projection to semantic space) -------
        # Projects the feature per (class, patch) to semantic_dim; the reorder
        # to pixel produces (B, semantic_dim, H, W).
        self.to_visual_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, semantic_dim * patch_size * patch_size),
        )

    def _tokenize(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Tokenize the input ``(B, T, C, H, W)`` into patch tokens.

        Args:
            x: Input tensor ``(B, T, C, H, W)``.

        Returns:
            Tuple ``(tokens, batch)`` where ``tokens`` has shape
            ``(B, T, N, dim)`` with ``N`` spatial tokens per timestep.
        """
        b, t = x.shape[0], x.shape[1]
        # Conv2d is applied to each temporal image independently.
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.to_patch_embedding(x)  # (b*t, dim, grid, grid)
        x = rearrange(x, "(b t) d gh gw -> b t (gh gw) d", b=b, t=t)
        return x, b

    def _temporal_pos(
        self, doy: torch.Tensor | None, batch: int, t: int, device: torch.device
    ) -> torch.Tensor:
        """Return the temporal PE ``(B, T, dim)`` by DOY or ordinal.

        Args:
            doy: Day-of-year per timestep ``(B, T)`` int, or ``None``.
            batch: Batch size ``B``.
            t: Number of timesteps ``T``.
            device: Target device.

        Returns:
            Tensor ``(B, T, dim)`` with the temporal positional encoding.
        """
        if doy is None:
            return self.temporal_pos_ordinal[:, :t, :].expand(batch, -1, -1)
        doy_idx = doy.long().clamp(0, self.max_doy).to(device)
        return self.temporal_pos_embedding[doy_idx]  # (B, T, dim)

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor | None = None,
        *,
        return_visual_proj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pass the series through the factorized encoders and reconstruct logits.

        Args:
            x: Input series ``(B, T, C, H, W)`` float (B4 idx2, B8 idx6,
                10 Sentinel-2 PASTIS-R bands, scale /10000).
            doy: Acquisition day-of-year per timestep ``(B, T)`` int. If
                ``None`` the fallback ordinal temporal PE is used.
            return_visual_proj: If ``True``, also returns the per-pixel visual
                projection to the semantic space ``semantic_dim``.

        Returns:
            - ``return_visual_proj=False``: segmentation logits
              ``(B, num_classes, H, W)``.
            - ``return_visual_proj=True``: tuple ``(logits, visual_proj)`` with
              ``visual_proj`` of shape ``(B, semantic_dim, H, W)``.
        """
        tokens, b = self._tokenize(x)  # (B, T, N, dim)
        t = tokens.shape[1]
        n = tokens.shape[2]
        device = tokens.device

        # --- Temporal PE by DOY (added to each token of each position) ------
        temp_pos = self._temporal_pos(doy, b, t, device)  # (B, T, dim)
        tokens = tokens + temp_pos.unsqueeze(2)  # broadcast over N positions

        # --- Temporal encoder per spatial position --------------------------
        # Flatten (B, N) into the batch axis to attend only over the temporal axis.
        seq = rearrange(tokens, "b t n d -> (b n) t d")
        cls = repeat(self.temporal_cls_tokens, "1 k d -> bn k d", bn=b * n)  # (B*N, K, dim)
        seq = torch.cat([cls, seq], dim=1)  # (B*N, K + T, dim)
        seq = self.temporal_transformer(seq)
        # Keep only the K cls-tokens (temporal summary per class).
        cls_out = seq[:, : self.num_classes, :]  # (B*N, K, dim)

        # --- Spatial encoder per class --------------------------------------
        # Reorder to (B*K, N, dim): for each class, attend over the spatial axis.
        spatial = rearrange(cls_out, "(b n) k d -> (b k) n d", b=b, n=n)
        spatial = spatial + self.spatial_pos_embedding  # (B*K, N, dim)
        spatial = self.spatial_transformer(spatial)  # (B*K, N, dim)

        # --- Segmentation head: patch token -> p*p pixels -------------------
        seg = self.to_seg(spatial)  # (B*K, N, p*p)
        logits: torch.Tensor = rearrange(
            seg,
            "(b k) (gh gw) (ph pw) -> b k (gh ph) (gw pw)",
            b=b,
            k=self.num_classes,
            gh=self.grid,
            gw=self.grid,
            ph=self.patch_size,
            pw=self.patch_size,
        )  # (B, K, H, W)

        if not return_visual_proj:
            return logits

        # --- Visual branch: feature per (class, patch) -> semantic pixel ----
        # The K class branches are averaged to obtain a visual feature per
        # spatial position, and projected to semantic_dim per pixel.
        per_class = rearrange(spatial, "(b k) n d -> b k n d", b=b)
        pooled = per_class.mean(dim=1)  # (B, N, dim) visual feature per patch
        proj = self.to_visual_proj(pooled)  # (B, N, semantic_dim*p*p)
        visual_proj = rearrange(
            proj,
            "b (gh gw) (s ph pw) -> b s (gh ph) (gw pw)",
            gh=self.grid,
            gw=self.grid,
            s=self.semantic_dim,
            ph=self.patch_size,
            pw=self.patch_size,
        )  # (B, semantic_dim, H, W)
        return logits, visual_proj


def build_tsvit(
    num_classes: int = 18,
    n_timesteps: int = 10,
    img_size: int = 128,
    in_channels: int = 10,
    patch_size: int = 8,
    dim: int = 128,
    depth_temporal: int = 4,
    depth_spatial: int = 4,
    semantic_dim: int = 384,
    *,
    heads: int = 4,
    dim_head: int = 32,
    mlp_ratio: int = 4,
    dropout: float = 0.0,
    max_doy: int = 366,
) -> nn.Module:
    """Build a :class:`TSViT` with the defaults trimmed for L4.

    Public factory of the TSViT wrapper (US-025 Task 3). The positional defaults
    (``patch_size=8`` -> 16x16 tokens, ``dim=128``, depth 4+4) keep the model
    trainable on an L4 24GB with ``T=10``, 128px and ``batch=4``. They are kept
    intact for retro-compatibility: ``build_tsvit()`` with no extra argument and
    the existing positional calls (notebooks ``5a``/``5b``, the re-score harness)
    reproduce the L4 model unchanged.

    Full-M (H100, US-038): the capacity is raised by passing the keyword-only
    arguments explicitly, e.g. ``dim=192, depth_temporal=6, depth_spatial=6,
    heads=6, dim_head=64`` with ``n_timesteps=64`` (so the ordinal temporal PE
    ``[1, n_timesteps, dim]`` covers the full PASTIS-R series ``T <= 64`` and is
    never indexed out of range; see US-038 R-TLEN). The :class:`TSViT` class
    already supports ``heads``/``dim_head``/``mlp_ratio``/``dropout``/``max_doy``;
    this factory only exposes them so the trained capacity, the saved checkpoint
    and the harness reconstruction stay identical (US-038 R-HARNESS).

    Args:
        num_classes: Number ``K`` of classes / separable cls-tokens.
        n_timesteps: Expected temporal length ``T`` (also sizes the fallback
            ordinal temporal PE; for Full-M use ``64`` >= the PASTIS-R ``T_max``).
        img_size: Side of the input patch in pixels.
        in_channels: Input bands per timestep (10 for Sentinel-2).
        patch_size: Side of the spatial patch.
        dim: Transformer token dimension (L4 128; Full-M 192).
        depth_temporal: Blocks of the temporal encoder (L4 4; Full-M 6).
        depth_spatial: Blocks of the spatial encoder (L4 4; Full-M 6).
        semantic_dim: Dimension of the semantic space of the contrastive branch
            (384 for the per-class phenology prototypes).
        heads: Attention heads in both encoders (L4 4; Full-M 6).
        dim_head: Dimension per attention head (L4 32; Full-M 64;
            ``inner_dim = heads * dim_head``).
        mlp_ratio: Expansion factor of the feed-forward (``mlp_dim = dim *
            mlp_ratio``); kept at 4 (paper standard) for both L4 and Full-M.
        dropout: Dropout applied in the Transformers (0.0 by default; 0.1 may
            regularize the higher-capacity Full-M if it overfits).
        max_doy: Maximum day-of-year admitted by the DOY temporal PE table.

    Returns:
        :class:`TSViT` module ready to train/infer.
    """
    return TSViT(
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        img_size=img_size,
        in_channels=in_channels,
        patch_size=patch_size,
        dim=dim,
        depth_temporal=depth_temporal,
        depth_spatial=depth_spatial,
        heads=heads,
        dim_head=dim_head,
        mlp_ratio=mlp_ratio,
        semantic_dim=semantic_dim,
        dropout=dropout,
        max_doy=max_doy,
    )
