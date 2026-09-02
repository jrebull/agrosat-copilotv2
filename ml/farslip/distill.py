"""FarSLIP distillation — losses + trainer (US-017 / US-016b).

Implements the Li et al. 2025 procedure (arXiv:2511.14901):

- :class:`PatchDistillationLoss` (paper §3.2): MSE + cosine between the 196
  patches of the student and those of the teacher, with explicit ``stop-grad``
  on the teacher features to avoid back-prop toward the frozen model.
- :class:`RegionCategoryAlignmentLoss` (paper §3.3): contrastive InfoNCE on the
  student's CLS token vs region x category text prototypes. The prototypes are
  computed ONCE per epoch (text encoder frozen).
- :class:`FarSLIPDistillationTrainer`: orchestrates the AdamW bf16 loop with
  MLflow autolog. Initializes the student from the teacher (``copy.deepcopy``),
  adapts ``patch_embed.proj`` from 3 to 4 channels with init = mean(RGB) for the
  NIR channel (avoids dead-neuron). Hard cap 8 h, warning at 6 h.

US-036-a v2 (faithful redesign, additive): a ``supervision="region_category"``
mode optimizes the paper-faithful ``L_total = L_glo + lambda_loc * L_loc`` with
``L_loc`` = :class:`~ml.farslip.mpcl_loss.MultiPositiveRegionCategoryLoss`
(eq. 3-4) over the multi-object region-category batch and ``L_glo`` =
:class:`~ml.farslip.mpcl_loss.GlobalImageTextLoss` (eq. 1-2) between the image
CLS and the global caption CLS. The default ``supervision="dominant"`` preserves
the v1 single-positive path byte-for-byte (back-compat with US-036-a v1, its
orchestrator and its tests). See ``docs/us-planning/us-036-a-v2-faithful.md``.

Expected VRAM on GCP L4 24 GB: ~22 GB (ViT-B/16 bf16, batch=64, grad_accum=2).
On the H100 the v2 student (ViT-B/16 BF16, batch=64) stays < 16 GB.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.farslip.bands import BandSelection
from ml.farslip.mpcl_loss import (
    DEFAULT_LAMBDA_LOC,
    DEFAULT_TEMPERATURE,
    GlobalImageTextLoss,
    MultiPositiveRegionCategoryLoss,
    combine_losses,
)
from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.seed import propagate_seed

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FarSLIP requires transformers>=4.46. Install with `poetry add transformers`."
    ) from exc

_log = structlog.get_logger(__name__)

LossType = Literal["mse", "cosine", "mse_plus_cosine"]
SaveFormat = Literal["safetensors", "pytorch"]

#: Supervision mode of :class:`FarSLIPDistillationTrainer`.
#:
#: - ``"dominant"`` (v1, US-036-a): one dominant ``category_id`` per patch,
#:   single-positive :class:`RegionCategoryAlignmentLoss` (``F.cross_entropy``) +
#:   patch-distillation + cosine aux. Untouched, default for back-compat.
#: - ``"region_category"`` (v2, US-036-a v2): the paper-faithful path. Each batch
#:   carries several region-category pairs (``ParcelIDs``) and a global caption;
#:   the loss is ``L_total = L_glo + lambda_loc * L_loc`` with ``L_loc`` the
#:   Multi-Positive Contrastive Loss (:class:`MultiPositiveRegionCategoryLoss`,
#:   eq. 3-4) and ``L_glo`` the symmetric InfoNCE image-caption
#:   (:class:`GlobalImageTextLoss`, eq. 1-2).
SupervisionMode = Literal["dominant", "region_category"]

#: Dimension of the MiniLM (all-MiniLM-L6-v2) phenological prototypes of US-033.
#: ``set_text_prototypes`` reprojects this to the student CLS dim (768) via a
#: frozen orthogonal map; any other non-768 input dimension is rejected.
MINILM_DIM = 384


# ---------------------------------------------------------------------------
# Patch-to-patch distillation loss (AC-1, AC-7).
# ---------------------------------------------------------------------------


class PatchDistillationLoss(nn.Module):
    """Patch-to-patch distillation loss (FarSLIP §3.2).

    Combines MSE and/or cosine between the features of the 196 patches of the
    student and the teacher. The teacher is assumed frozen; we apply ``.detach()``
    to guarantee explicit stop-gradient (defensive against caller failures).

    Args:
        loss_type: ``"mse"``, ``"cosine"`` or ``"mse_plus_cosine"`` (default).
        cosine_weight: weight of the cosine term when ``loss_type=="mse_plus_cosine"``.
        normalize: if ``True``, L2-normalizes the features before the computation.
    """

    def __init__(
        self,
        loss_type: LossType = "mse_plus_cosine",
        cosine_weight: float = 0.3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine", "mse_plus_cosine"):
            raise ValueError(f"invalid loss_type: {loss_type!r}")
        if not 0.0 <= cosine_weight <= 1.0:
            raise ValueError(f"cosine_weight out of [0,1]: {cosine_weight}")
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.normalize = normalize

    def forward(
        self,
        student_patch_feats: torch.Tensor,
        teacher_patch_feats: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the scalar loss differentiable wrt ``student_patch_feats``.

        Args:
            student_patch_feats: tensor ``(B, P, D)`` (P=196 patches default).
            teacher_patch_feats: tensor ``(B, P, D)`` — will be detached.
            patch_mask: optional bool ``(B, P)`` (True = valid patch).

        Returns:
            Scalar loss tensor with grad active with respect to the student.
        """

        if student_patch_feats.shape != teacher_patch_feats.shape:
            raise ValueError(
                f"shape mismatch student={student_patch_feats.shape} "
                f"teacher={teacher_patch_feats.shape}"
            )
        teacher = teacher_patch_feats.detach()
        student = student_patch_feats

        if self.normalize:
            student = F.normalize(student, p=2, dim=-1)
            teacher = F.normalize(teacher, p=2, dim=-1)

        if patch_mask is not None:
            if patch_mask.shape != student.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch mask={patch_mask.shape} feats={student.shape[:2]}"
                )
            mask = patch_mask.unsqueeze(-1).to(student.dtype)
            n_valid = mask.sum().clamp(min=1.0)
        else:
            mask = None
            n_valid = torch.tensor(
                float(student.shape[0] * student.shape[1]),
                device=student.device,
                dtype=student.dtype,
            )

        mse_term = torch.tensor(0.0, device=student.device, dtype=student.dtype)
        cos_term = torch.tensor(0.0, device=student.device, dtype=student.dtype)

        if self.loss_type in ("mse", "mse_plus_cosine"):
            squared = (student - teacher).pow(2).sum(dim=-1)  # (B, P)
            if mask is not None:
                squared = squared * mask.squeeze(-1)
            mse_term = squared.sum() / n_valid

        if self.loss_type in ("cosine", "mse_plus_cosine"):
            # 1 - cos similarity; both are already L2-norm if self.normalize True.
            if not self.normalize:
                s = F.normalize(student, p=2, dim=-1)
                t = F.normalize(teacher, p=2, dim=-1)
            else:
                s, t = student, teacher
            cos_sim = (s * t).sum(dim=-1)  # (B, P)
            cos_loss = 1.0 - cos_sim
            if mask is not None:
                cos_loss = cos_loss * mask.squeeze(-1)
            cos_term = cos_loss.sum() / n_valid

        if self.loss_type == "mse":
            return mse_term
        if self.loss_type == "cosine":
            return cos_term
        return mse_term + self.cosine_weight * cos_term


# ---------------------------------------------------------------------------
# Region x Category InfoNCE alignment (AC-1, AC-7).
# ---------------------------------------------------------------------------


class RegionCategoryAlignmentLoss(nn.Module):
    """Region-category alignment on the CLS token (FarSLIP §3.3).

    Computes contrastive InfoNCE between the student's CLS and the text
    prototypes ``(n_regions * n_categories, D)`` precomputed by the teacher's text
    encoder (frozen). The positive of each sample is the prototype corresponding
    to its (region_id, category_id) pair.

    Args:
        temperature: softmax temperature (default 0.07, paper §3.3).
        n_regions: number of regions (3 default for Italy).
        n_categories: number of CAP classes (32 default).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_regions: int = 3,
        n_categories: int = 32,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive: {temperature}")
        if n_regions < 1 or n_categories < 1:
            raise ValueError("n_regions and n_categories must be >= 1")
        self.temperature = temperature
        self.n_regions = n_regions
        self.n_categories = n_categories

    def forward(
        self,
        student_cls: torch.Tensor,
        text_prototypes: torch.Tensor,
        region_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Computes contrastive InfoNCE.

        Args:
            student_cls: tensor ``(B, D)`` of the student.
            text_prototypes: tensor ``(n_regions * n_categories, D)``; it is
                detached internally (frozen).
            region_ids: long tensor ``(B,)`` with region index.
            category_ids: long tensor ``(B,)`` with category index.

        Returns:
            Scalar loss tensor with grad active with respect to the student.
        """

        if student_cls.dim() != 2:
            raise ValueError(f"student_cls must be (B,D); got {student_cls.shape}")
        if text_prototypes.dim() != 2:
            raise ValueError(f"text_prototypes must be (R*C,D); got {text_prototypes.shape}")
        expected_protos = self.n_regions * self.n_categories
        if text_prototypes.shape[0] != expected_protos:
            raise ValueError(
                f"text_prototypes rows={text_prototypes.shape[0]} expected={expected_protos}"
            )
        if region_ids.shape != category_ids.shape:
            raise ValueError("region_ids and category_ids must have the same shape")
        if region_ids.shape[0] != student_cls.shape[0]:
            raise ValueError("inconsistent batch size between student_cls and ids")
        if (region_ids < 0).any() or (region_ids >= self.n_regions).any():
            raise ValueError("region_ids out of range")
        if (category_ids < 0).any() or (category_ids >= self.n_categories).any():
            raise ValueError("category_ids out of range")

        protos = text_prototypes.detach()
        student_n = F.normalize(student_cls, p=2, dim=-1)
        protos_n = F.normalize(protos, p=2, dim=-1)

        # logits: (B, n_regions * n_categories)
        logits = student_n @ protos_n.t() / self.temperature
        # target index: region * n_categories + category
        targets = region_ids.long() * self.n_categories + category_ids.long()
        return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Trainer (AC-2, AC-4, AC-5, AC-9).
# ---------------------------------------------------------------------------


@dataclass
class FarSLIPTrainerConfig:
    """Hparams of :class:`FarSLIPDistillationTrainer`.

    Attributes:
        teacher_model_id: HF id of the CLIP teacher.
        dataset_root: path to ``data/farslip_pairs/`` (manifest + crops).
        output_dir: local path of the weights before uploading to GCS.
        gcs_output_uri: optional ``gs://agrosat-models/farslip/{run_name}/``.
        loss_weights: ``{"alpha":1.0, "beta":0.5, "gamma":0.2}`` default.
        n_epochs: AC-4 default 4.
        batch_size: AC-4 default 64 (effective 128 with grad_accum=2).
        grad_accum_steps: AC-4 default 2.
        lr: AC-4 default 1e-5 AdamW.
        weight_decay: 0.01 default.
        warmup_ratio: 0.05 cosine warmup.
        seed: 42 (propagated to torch/np/random + deterministic algos).
        mlflow_run_name: ``"farslip-clip-italy-v1"``.
        device: ``"cuda"`` | ``"cpu"`` | ``"auto"``.
        time_cap_hours: hard cap 8 h (warning at 6 h).
        num_workers: DataLoader workers default 4.
        n_in_channels: 4 (B02 B03 B04 B08). US-035: DERIVED from
            ``band_selection`` by the caller (3 for rgb/nir_rgb, 4 for 4band) —
            single source of truth, never set independently.
        band_selection: US-035 band-ablation variant (``rgb``/``nir_rgb``/
            ``4band``). Logged as an MLflow param; drives ``n_in_channels`` and
            the dataset ``transform`` upstream (``train.py``).
        n_regions: 3.
        n_categories: 32.
        supervision: ``"dominant"`` (v1, default) or ``"region_category"`` (v2).
            Selects which loss the trainer optimizes; v1 is preserved verbatim.
        lambda_loc: weight of ``L_loc`` in the v2 combination
            ``L_total = L_glo + lambda_loc * L_loc`` (paper Table 3, default 1.0).
            With ``lambda_loc=0`` the v2 total equals ``L_glo`` (ablation).
        temperature: contrastive softmax temperature ``tau`` of the v2 losses
            (default 0.07, paper Section 3.3). Independent of the v1
            :class:`RegionCategoryAlignmentLoss` temperature.
        use_global_caption_loss: if ``True`` (v2 default) the trainer adds the
            real ``L_glo`` InfoNCE between the image CLS and the caption CLS.
            If ``False`` only ``L_loc`` is optimized in v2 (``L_glo`` ablation).
    """

    teacher_model_id: str = "openai/clip-vit-base-patch16"
    dataset_root: Path = Path("data/farslip_pairs")
    output_dir: Path = Path("artifacts/farslip")
    gcs_output_uri: str | None = None
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}
    )
    n_epochs: int = 4
    batch_size: int = 64
    grad_accum_steps: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42
    mlflow_run_name: str = "farslip-clip-italy-v1"
    device: str = "auto"
    time_cap_hours: float = 8.0
    warning_hours: float = 6.0
    num_workers: int = 4
    n_in_channels: int = 4
    #: US-035 band-ablation variant; ``n_in_channels`` is derived from it by the
    #: caller (``train.py``). Default ``"4band"`` matches the default
    #: ``n_in_channels=4`` so the two stay coherent out of the box.
    band_selection: BandSelection = "4band"
    n_regions: int = 3
    n_categories: int = 32
    #: Supervision mode (US-036-a v2). ``"dominant"`` keeps the v1 path intact;
    #: ``"region_category"`` activates the paper-faithful MPCL + L_glo route.
    supervision: SupervisionMode = "dominant"
    #: v2 weight of ``L_loc`` in ``L_total = L_glo + lambda_loc * L_loc``.
    lambda_loc: float = DEFAULT_LAMBDA_LOC
    #: v2 contrastive temperature ``tau`` of MPCL and L_glo (paper Section 3.3).
    temperature: float = DEFAULT_TEMPERATURE
    #: v2 toggle of the real global image-caption InfoNCE ``L_glo``.
    use_global_caption_loss: bool = True
    #: Extra MLflow params/tags (US-034: proto_source, proto_proj, caveat,
    #: n_protos, proto_dim_in, proto_dim_out). Logged verbatim in :meth:`train`.
    extra_params: dict[str, Any] = field(default_factory=dict)


def _resolve_device(device: str) -> torch.device:
    """Resolves ``"auto"`` -> ``"cuda"`` if available, otherwise ``"cpu"``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def adapt_patch_embed_to_n_channels(vision_model: nn.Module, target_channels: int) -> None:
    """Adapts the ``patch_embedding`` of a CLIP vision model to ``target_channels``.

    Supports both :class:`CLIPVisionModel` (transformers 5.x, flat: it has
    ``embeddings`` directly) and ``CLIPModel.vision_model`` (with hierarchy). The
    extra channel (NIR) is initialized as the ``mean`` of the 3 RGB to avoid
    dead-neuron (zero init would flatten the NIR signal). Modifies the module
    in-place. Reuses the same bias (there is no bias in the CLIP patch_embed).
    """

    # Resolve ``embeddings`` with fallback (transformers 4.x vs 5.x).
    if hasattr(vision_model, "embeddings"):
        embeddings = vision_model.embeddings  # type: ignore[union-attr]
    elif hasattr(vision_model, "vision_model"):
        embeddings = vision_model.vision_model.embeddings  # type: ignore[union-attr]
    else:
        raise AttributeError("vision_model exposes neither .embeddings nor .vision_model")
    old_proj = embeddings.patch_embedding  # type: ignore[union-attr]
    assert isinstance(old_proj, nn.Conv2d), (
        f"patch_embedding must be Conv2d; got {type(old_proj).__name__}"
    )
    if old_proj.in_channels == target_channels:
        return
    if old_proj.in_channels != 3:
        raise ValueError(f"expected patch_embed with 3 input channels, got {old_proj.in_channels}")
    out_ch = old_proj.out_channels
    # Conv2d.kernel_size/stride/padding are tuple[int, int] at runtime although
    # the type-stub publishes tuple[int, ...]. Explicit cast for mypy.
    k: tuple[int, int] = (old_proj.kernel_size[0], old_proj.kernel_size[1])
    stride: tuple[int, int] = (old_proj.stride[0], old_proj.stride[1])
    if isinstance(old_proj.padding, str):
        padding: str | tuple[int, int] = old_proj.padding
    else:
        padding = (old_proj.padding[0], old_proj.padding[1])
    bias_flag = old_proj.bias is not None

    new_proj = nn.Conv2d(
        in_channels=target_channels,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        bias=bias_flag,
    )
    with torch.no_grad():
        # copy first 3 channels as-is
        new_proj.weight[:, :3, :, :] = old_proj.weight.detach().clone()
        if target_channels > 3:
            rgb_mean = old_proj.weight.detach().mean(dim=1, keepdim=True)  # (O,1,k,k)
            for ch in range(3, target_channels):
                new_proj.weight[:, ch : ch + 1, :, :] = rgb_mean.clone()
        if bias_flag and old_proj.bias is not None and new_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias.detach().clone())
    embeddings.patch_embedding = new_proj  # type: ignore[union-attr]
    _log.info(
        "patch_embed adapted",
        from_channels=3,
        to_channels=target_channels,
        init="mean_rgb_on_extra",
    )


class FarSLIPDistillationTrainer:
    """End-to-end FarSLIP distillation trainer.

    Initializes teacher (frozen) and student (deep clone + trainable) from the
    same HF id, adapts patch_embed to ``n_in_channels``, configures AdamW + cosine
    warmup + AMP bf16 + grad accumulation, records MLflow autolog and saves
    weights in safetensors format.

    Args see :class:`FarSLIPTrainerConfig`.
    """

    def __init__(
        self,
        config: FarSLIPTrainerConfig,
        dataset: torch.utils.data.Dataset | None = None,
        text_prototypes: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        propagate_seed(config.seed)
        self._load_models()
        self._patch_student_proj()
        self._optim = self._build_optimizer()
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._dataset = dataset
        # text_prototypes optional: if None, the trainer expects the caller
        # to provide them via :meth:`set_text_prototypes` before :meth:`train`.
        self._text_prototypes = text_prototypes
        self._patch_loss = PatchDistillationLoss()
        self._cls_loss = RegionCategoryAlignmentLoss(
            n_regions=config.n_regions, n_categories=config.n_categories
        )
        # v2 (region_category) faithful losses. Always constructed (cheap, pure
        # nn.Modules) so the trainer can switch modes without re-instantiation;
        # they are only consumed when ``config.supervision == "region_category"``.
        self._mpcl_loss = MultiPositiveRegionCategoryLoss(temperature=config.temperature)
        self._global_loss = GlobalImageTextLoss(temperature=config.temperature)
        # v2 category prototype bank ``(C, D)`` and the PASTIS-id -> [0, C) map,
        # both injected via :meth:`set_category_prototypes`. None until set.
        self._category_prototypes: torch.Tensor | None = None
        self._pastis_to_category: dict[int, int] | None = None
        # v2 optional frozen text encoder for the caption CLS of ``L_glo``. When
        # absent, captions are encoded by the same MiniLM lift used for the
        # category prototypes (see :meth:`_encode_captions`).
        self._caption_encoder: Any | None = None
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils

    def _load_models(self) -> None:
        teacher = CLIPVisionModel.from_pretrained(self.config.teacher_model_id)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Student starts as an exact copy of the teacher (AC-2)
        student = copy.deepcopy(teacher)
        for p in student.parameters():
            p.requires_grad_(True)
        student.train()
        self.teacher = teacher.to(self.device)  # type: ignore[arg-type]
        self.student = student.to(self.device)  # type: ignore[arg-type]

    def _patch_student_proj(self) -> None:
        """Adapts ONLY the student to ``n_in_channels`` Sentinel-2 bands.

        The teacher is kept with 3 channels (pure RGB, FarSLIP paper §3.2 +
        AC-2: teacher = original ``openai/clip-vit-base-patch16``). The teacher's
        forward receives the first 3 bands of the student via slicing in
        :meth:`_teacher_forward`. This preserves the authentic distillation signal
        of the pretrained CLIP — adapting the teacher as well would contaminate
        the pseudo-label with an untrained NIR projection.
        """
        adapt_patch_embed_to_n_channels(self.student, self.config.n_in_channels)
        # Move back to the device (new layers created on CPU).
        self.student.to(self.device)
        # Teacher stays with 3 channels: its patch_embed is NOT touched.

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self._optim, _lr_lambda)

    # ------------------------------------------------------------------ API

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        """Injects precomputed text prototypes ``(R*C, D_in)``.

        Computed externally to avoid coupling the text encoder to the trainer
        (the text encoder is frozen, running it once per epoch is enough).

        US-034 fix: the contrastive loss (:class:`RegionCategoryAlignmentLoss`)
        contrasts ``student_cls = last_hidden_state[:, 0, :]`` (dim
        ``teacher.config.hidden_size`` == 768 for ViT-B/16) against these
        prototypes. The phenological prototypes of US-033 live in the MiniLM-384
        space (:data:`MINILM_DIM`), so if ``D_in == 384`` they are reprojected
        to 768 via a frozen orthogonal map (:meth:`_proto_to_clip_proj`). If they
        already arrive in 768 they pass through unchanged (back-compat with the
        random path). Any OTHER dimension (e.g. 512, the CLIP-shared inference
        space, which is NOT the loss space) fails fast here with a clear
        ``ValueError`` instead of the opaque ``RuntimeError`` the loss would
        raise in its ``student_n @ protos.t()`` matmul (the loss does NOT
        validate D). A final assert guards the stored dimension.

        Args:
            prototypes: ``(R*C, D_in)`` tensor; ``D_in`` is 384 (MiniLM, US-033)
                or already ``hidden_size`` (768).

        Raises:
            ValueError: if ``prototypes`` is not 2-D, if ``D_in`` is neither 384
                nor ``hidden_size``, or if after reprojection the final dimension
                does not equal ``teacher.config.hidden_size``.
        """
        if prototypes.dim() != 2:
            raise ValueError(f"text_prototypes must be 2-D (R*C, D); got {tuple(prototypes.shape)}")
        hidden_size = int(self.teacher.config.hidden_size)
        d_in = prototypes.shape[1]
        if d_in == hidden_size:
            pass  # already in CLS space, no reprojection
        elif d_in == MINILM_DIM:
            prototypes = self._proto_to_clip_proj(prototypes)
        else:
            raise ValueError(
                f"text_prototypes dim={d_in} unsupported. Expected MiniLM "
                f"{MINILM_DIM} (US-033, reprojected to {hidden_size}) or "
                f"{hidden_size} (student CLS dim) directly. The loss does NOT "
                f"validate D, so injecting {d_in} would raise an opaque "
                f"RuntimeError in the student_cls @ protos.t() matmul."
            )
        d_out = prototypes.shape[1]
        if d_out != hidden_size:  # defensive: reprojection contract guard
            raise ValueError(
                f"text_prototypes final dim={d_out} != student CLS dim "
                f"{hidden_size} (teacher.config.hidden_size)."
            )
        self._text_prototypes = prototypes.detach().to(self.device)

    def _proto_to_clip_proj(self, emb: torch.Tensor) -> torch.Tensor:
        """Frozen orthogonal projection MiniLM-384 -> CLS-``hidden_size`` (768).

        Builds (once, cached as a non-persistent buffer) a semi-orthogonal matrix
        ``W (hidden_size, D_in)`` seeded by ``config.seed``
        (``requires_grad=False``), and returns ``emb @ W.t()`` -> ``hidden_size``.
        ``W`` is generated with :func:`torch.nn.init.orthogonal_` so its columns
        are orthonormal (``W.t() @ W == I_{D_in}`` for ``hidden_size >= D_in``);
        the projection therefore preserves the L2 norm and the relative
        inner-products/angles of the MiniLM subspace. The 18 phenological classes
        stay distinguishable after the lift (unlike the ``torch.randn`` it
        replaces, which encodes no semantics at all).

        CAVEAT (US-034 AC-8 / R-APPROX): this is a crude linear embedding of the
        MiniLM-384 space into 768, NOT the native CLIP text encoder (which would
        produce 768 semantically aligned with the visual CLS). The paper-faithful
        path is :meth:`_load_text_encoder` (post-delivery).

        Args:
            emb: ``(N, D_in)`` prototype matrix (D_in == 384 for MiniLM).

        Returns:
            ``(N, hidden_size)`` reprojected matrix (norm/angle preserving),
            detached and on the same dtype as ``emb``.

        Raises:
            ValueError: if ``emb`` is not 2-D or ``D_in > hidden_size`` (the
                orthonormal-columns guarantee requires ``hidden_size >= D_in``).
        """
        if emb.dim() != 2:
            raise ValueError(f"emb must be 2-D (N, D_in); got {tuple(emb.shape)}")
        hidden_size = int(self.teacher.config.hidden_size)
        d_in = emb.shape[1]
        if d_in > hidden_size:
            raise ValueError(
                f"orthogonal lift requires hidden_size ({hidden_size}) >= "
                f"D_in ({d_in}); cannot preserve norm when projecting down."
            )
        # Cache the frozen projection per input dim (one buffer per D_in).
        buf_name = f"_proto_proj_w_{d_in}"
        w = getattr(self, buf_name, None)
        if w is None:
            gen = torch.Generator().manual_seed(int(self.config.seed))
            w = torch.empty(hidden_size, d_in)
            torch.nn.init.orthogonal_(w, generator=gen)  # columns orthonormal
            w.requires_grad_(False)
            # Plain attribute (NOT nn.Parameter / registered buffer) so it never
            # enters the optimizer nor the student state_dict (AC-5).
            setattr(self, buf_name, w)
        w = w.to(device=emb.device, dtype=emb.dtype)
        return (emb.detach() @ w.t()).detach()

    def _load_text_encoder(self) -> None:  # pragma: no cover - post-delivery stub
        """Paper-faithful prototype path (post-delivery, NOT trained in US-034).

        Documented stub for the clean solution (US-034 R-APPROX / AC-8): load a
        frozen ``CLIPTextModel`` (``self.config.teacher_model_id``), encode the
        CAP templates of ``cap_vocabulary.yaml`` per region x category, and use
        its native ``hidden_size`` (768) output as text prototypes aligned with
        the visual CLS — replacing the crude orthogonal lift of
        :meth:`_proto_to_clip_proj`. NOT invoked in the US-034 run; the
        contrastive loss currently consumes the reprojected MiniLM prototypes.
        """
        raise NotImplementedError(
            "Native CLIPTextModel prototype path is post-delivery (US-034 AC-8); "
            "US-034 uses MiniLM-384 prototypes reprojected via _proto_to_clip_proj."
        )

    # --------------------------------------------------------------- v2 (faithful)

    def set_category_prototypes(
        self,
        prototypes: torch.Tensor,
        pastis_class_ids: Sequence[int],
    ) -> None:
        """Injects the v2 category prototype bank ``(C, D)`` and its id mapping.

        Faithful FarSLIP v2 (US-036-a v2) path. ``prototypes[c]`` is the text
        prototype of the category whose RAW PASTIS class id is
        ``pastis_class_ids[c]``; the rows define the canonical category order
        ``[0, C)``. The bank is reprojected to the student CLS dim with the SAME
        frozen orthogonal lift the v1 path uses (:meth:`set_text_prototypes` ->
        :meth:`_proto_to_clip_proj`: MiniLM-384 -> 768; an already-768 bank passes
        through), so v2 categories live in the exact space the contrastive loss
        consumes.

        The PASTIS-id -> category-index map is stored because
        :class:`ml.farslip.region_category_dataset.RegionCategoryPairDataset`
        emits RAW PASTIS ids (1..18), whereas
        :class:`MultiPositiveRegionCategoryLoss` expects indices in ``[0, C)``.
        :meth:`step_faithful_v2` translates the batch ids through this map.

        Args:
            prototypes: ``(C, D_in)`` category text prototypes; ``D_in`` is 384
                (MiniLM, US-033) or the student CLS dim (768).
            pastis_class_ids: the ``C`` RAW PASTIS class ids, in the SAME row
                order as ``prototypes`` (canonical category order).

        Raises:
            ValueError: if the lengths disagree, the ids are not unique, or the
                reprojection fails the v1 dimensional contract.
        """
        if prototypes.dim() != 2:
            raise ValueError(
                f"category prototypes must be 2-D (C, D); got {tuple(prototypes.shape)}"
            )
        ids = [int(c) for c in pastis_class_ids]
        if len(ids) != prototypes.shape[0]:
            raise ValueError(
                f"pastis_class_ids has {len(ids)} entries but prototypes has "
                f"{prototypes.shape[0]} rows; they must match (one id per row)."
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"pastis_class_ids must be unique; got {ids}.")

        hidden_size = int(self.teacher.config.hidden_size)
        d_in = prototypes.shape[1]
        if d_in == hidden_size:
            bank = prototypes.detach()
        elif d_in == MINILM_DIM:
            bank = self._proto_to_clip_proj(prototypes)
        else:
            raise ValueError(
                f"category prototypes dim={d_in} unsupported. Expected MiniLM "
                f"{MINILM_DIM} (US-033) or {hidden_size} (student CLS dim)."
            )
        if bank.shape[1] != hidden_size:  # defensive
            raise ValueError(
                f"category prototypes final dim={bank.shape[1]} != student CLS dim {hidden_size}."
            )
        self._category_prototypes = bank.detach().to(self.device)
        self._pastis_to_category = {cid: idx for idx, cid in enumerate(ids)}
        _log.info(
            "category prototypes set (v2)",
            n_categories=bank.shape[0],
            d=bank.shape[1],
            pastis_ids=ids,
        )

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """Re-weights ``L_loc`` (MPCL) by category to fight class imbalance (v2).

        Rebuilds :attr:`_mpcl_loss` with per-category weights so rare-class region
        anchors contribute more in the ``R->C`` direction, mitigating the collapse
        to dominant classes (e.g. Meadow) on the small, heavily imbalanced PASTIS.
        ``class_weights[c]`` must follow the SAME canonical category order ``[0, C)``
        as the prototype bank set in :meth:`set_category_prototypes` (typically the
        inverse-frequency weight of category ``c``, normalized to mean 1). Calling
        with uniform weights is numerically equivalent to no weighting.

        Args:
            class_weights: ``(C,)`` non-negative per-category weight in canonical
                order. ``C`` must equal the prototype bank row count.

        Raises:
            ValueError: if the prototype bank is unset or the length disagrees.
        """
        if self._category_prototypes is None:
            raise ValueError("set_category_prototypes() must be called before set_class_weights().")
        n_categories = int(self._category_prototypes.shape[0])
        if class_weights.dim() != 1 or class_weights.shape[0] != n_categories:
            raise ValueError(
                f"class_weights must be (C,) with C={n_categories}; "
                f"got {tuple(class_weights.shape)}"
            )
        self._mpcl_loss = MultiPositiveRegionCategoryLoss(
            temperature=self.config.temperature,
            class_weights=class_weights.detach().float().to(self.device),
        )
        _log.info(
            "mpcl class weights set (v2)",
            n_categories=n_categories,
            min_w=round(float(class_weights.min()), 3),
            max_w=round(float(class_weights.max()), 3),
        )

    def set_caption_encoder(self, encoder: Any) -> None:
        """Injects a frozen text encoder for the v2 caption CLS of ``L_glo``.

        Optional. ``encoder`` must expose ``encode(list[str]) -> ndarray | Tensor``
        of shape ``(B, D_in)`` (the sentence-transformers / MiniLM contract). When
        not set, :meth:`_encode_captions` falls back to the same MiniLM lift used
        for the prototypes via the externally-provided cache (the orchestrator
        pre-encodes captions once; see ``run_us036a_v2_farslip_faithful``). Either
        way the CLS lands in the student CLS dim (768).

        Args:
            encoder: object with ``encode(list[str])`` returning ``(B, D_in)``.
        """
        self._caption_encoder = encoder
        _log.info("caption encoder set (v2)", encoder=type(encoder).__name__)

    def _encode_captions(self, caption_cls: torch.Tensor) -> torch.Tensor:
        """Maps pre-encoded caption embeddings to the student CLS dim (768).

        The orchestrator encodes the batch captions ONCE with a frozen encoder
        (MiniLM-384 or already-768) and feeds the resulting ``(B, D_in)`` tensor
        here; this method lifts MiniLM-384 to 768 with the same frozen orthogonal
        map as the prototypes (norm/angle preserving) or passes an already-768
        tensor through. This keeps the text encoder out of the training loop
        (VRAM/time, plan Section 7) while landing the caption CLS in the visual
        CLS space ``L_glo`` contrasts against.

        Args:
            caption_cls: ``(B, D_in)`` pre-encoded caption embeddings.

        Returns:
            ``(B, 768)`` caption CLS in the student CLS space (detached).

        Raises:
            ValueError: if ``caption_cls`` is not 2-D or ``D_in`` is unsupported.
        """
        if caption_cls.dim() != 2:
            raise ValueError(f"caption_cls must be 2-D (B, D); got {tuple(caption_cls.shape)}")
        hidden_size = int(self.teacher.config.hidden_size)
        d_in = caption_cls.shape[1]
        if d_in == hidden_size:
            return caption_cls.detach().to(self.device)
        if d_in == MINILM_DIM:
            return self._proto_to_clip_proj(caption_cls).detach().to(self.device)
        raise ValueError(
            f"caption_cls dim={d_in} unsupported. Expected MiniLM {MINILM_DIM} "
            f"or {hidden_size} (student CLS dim)."
        )

    def _map_region_cat_ids(self, region_cat_ids: torch.Tensor) -> torch.Tensor:
        """Translates RAW PASTIS region ids (1..18) to category indices ``[0, C)``.

        The dataset emits RAW PASTIS class ids; the MPCL bank is ordered by the
        canonical category order set in :meth:`set_category_prototypes`. This maps
        each region id through ``self._pastis_to_category`` and fails fast on any
        id absent from the bank (a silent drop would corrupt ``P(i)``).

        Args:
            region_cat_ids: long tensor ``(R,)`` of RAW PASTIS class ids.

        Returns:
            long tensor ``(R,)`` of category indices in ``[0, C)``.

        Raises:
            RuntimeError: if the category bank/map is not set.
            ValueError: if a region carries a PASTIS id absent from the bank.
        """
        if self._pastis_to_category is None:
            raise RuntimeError(
                "category prototypes not initialized. Call set_category_prototypes()."
            )
        raw = region_cat_ids.long().view(-1).cpu().tolist()
        mapping = self._pastis_to_category
        unknown = sorted({c for c in raw if c not in mapping})
        if unknown:
            raise ValueError(
                f"region_cat_ids contains PASTIS ids absent from the category "
                f"bank: {unknown}. Known ids: {sorted(mapping)}."
            )
        mapped = [mapping[c] for c in raw]
        return torch.tensor(mapped, dtype=torch.long, device=self.device)

    def step_faithful_v2(
        self,
        images: torch.Tensor,
        region_cat_ids: torch.Tensor,
        region_to_patch: torch.Tensor,
        caption_cls: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward of ONE v2 batch: ``L_total = L_glo + lambda_loc * L_loc``.

        Paper-faithful region-category step (US-036-a v2). For a batch of ``B``
        patches with ``R`` flattened regions (collated by
        :func:`ml.farslip.region_category_dataset.collate_region_batch`):

        1. ``student_cls`` ``(B, D)`` = the student CLS over the batch images.
        2. ``region_visual = student_cls[region_to_patch]`` ``(R, D)`` -- every
           region of a patch shares that patch's CLS (paper Section 4.3
           Takeaway-1: CLS, not RoI; caveat R-REGION-CROP). The multiplicity
           enters the contrast cross-patch via the multi-positive grouping.
        3. ``L_loc`` = MPCL(``region_visual``, ``category_prototypes``,
           mapped ids) -- the RAW PASTIS ids are mapped to ``[0, C)`` first.
        4. ``L_glo`` = symmetric InfoNCE(``student_cls``, ``caption_cls``) when
           ``use_global_caption_loss`` and ``caption_cls`` is given; else 0.
        5. ``L_total`` = :func:`combine_losses`(``L_glo``, ``L_loc``,
           ``lambda_loc``).

        Args:
            images: ``(B, 4, H, W)`` peak-NDVI composites.
            region_cat_ids: ``(R,)`` RAW PASTIS class ids of the flattened regions.
            region_to_patch: ``(R,)`` index in ``[0, B)`` of each region's patch.
            caption_cls: optional ``(B, D_in)`` pre-encoded caption embeddings for
                ``L_glo``; ``None`` disables ``L_glo`` for this step.

        Returns:
            Dict with ``loss_total``, ``loss_glo``, ``loss_loc`` (grad active).

        Raises:
            RuntimeError: if the category prototypes were not set.
            ValueError: on a region->patch index out of range.
        """
        if self._category_prototypes is None:
            raise RuntimeError(
                "category prototypes not initialized. Call set_category_prototypes()."
            )
        images = images.to(self.device)
        region_to_patch = region_to_patch.to(self.device).long().view(-1)
        batch_size = images.shape[0]
        if region_to_patch.numel() > 0 and (
            int(region_to_patch.min()) < 0 or int(region_to_patch.max()) >= batch_size
        ):
            raise ValueError(
                f"region_to_patch out of range [0, {batch_size}); "
                f"got min={int(region_to_patch.min())} max={int(region_to_patch.max())}."
            )

        student_out = self.student(pixel_values=images, output_hidden_states=False)
        student_cls = student_out.last_hidden_state[:, 0, :]  # (B, D)

        # L_loc (MPCL): regions share their patch CLS; ids mapped to [0, C).
        region_visual = student_cls[region_to_patch]  # (R, D)
        mapped_ids = self._map_region_cat_ids(region_cat_ids)
        loss_loc = self._mpcl_loss(region_visual, self._category_prototypes, mapped_ids)

        # L_glo (InfoNCE image-caption): caption CLS lifted to the visual space.
        if self.config.use_global_caption_loss and caption_cls is not None:
            caption_cls_d = self._encode_captions(caption_cls)
            loss_glo = self._global_loss(student_cls, caption_cls_d)
        else:
            loss_glo = student_cls.sum() * 0.0  # keep grad graph, value 0.0

        total = combine_losses(loss_glo, loss_loc, self.config.lambda_loc)
        return {
            "loss_total": total,
            "loss_glo": loss_glo,
            "loss_loc": loss_loc,
        }

    def step(
        self,
        images: torch.Tensor,
        region_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward + backward of ONE batch (without optimizer.step).

        Returns a dict with loss tensors (not detached) so that the caller decides
        when to do ``backward`` + ``optimizer.step`` (smoke tests use this method
        under the hood).
        """
        if self._text_prototypes is None:
            raise RuntimeError("text_prototypes not initialized. Call set_text_prototypes().")
        images = images.to(self.device)
        region_ids = region_ids.to(self.device)
        category_ids = category_ids.to(self.device)

        student_out = self.student(pixel_values=images, output_hidden_states=False)
        # Teacher stays with 3 channels (pure RGB = B04/B03/B02 = BGR slice).
        # AC-2: we preserve the authentic pretrained CLIP; the student learns
        # to map 4 bands to the same semantics the teacher sees in 3.
        teacher_input = images[:, :3, :, :]
        with torch.no_grad():
            teacher_out = self.teacher(pixel_values=teacher_input, output_hidden_states=False)

        # CLIPVisionModel last_hidden_state shape: (B, 1+P, D) with CLS at pos 0.
        student_hidden = student_out.last_hidden_state
        teacher_hidden = teacher_out.last_hidden_state
        student_cls = student_hidden[:, 0, :]
        teacher_cls = teacher_hidden[:, 0, :]
        student_patches = student_hidden[:, 1:, :]
        teacher_patches = teacher_hidden[:, 1:, :]

        loss_patch = self._patch_loss(student_patches, teacher_patches)
        loss_cls = self._cls_loss(student_cls, self._text_prototypes, region_ids, category_ids)
        # auxiliary contrastive image-text-batch: aligns student CLS with teacher CLS
        # (lightweight placeholder for gamma; stabilizes the training)
        cos_aux = (
            1.0
            - F.cosine_similarity(
                F.normalize(student_cls, dim=-1),
                F.normalize(teacher_cls.detach(), dim=-1),
                dim=-1,
            ).mean()
        )

        w = self.config.loss_weights
        total = w["alpha"] * loss_patch + w["beta"] * loss_cls + w["gamma"] * cos_aux
        return {
            "loss_total": total,
            "loss_patch": loss_patch,
            "loss_cls": loss_cls,
            "loss_aux": cos_aux,
        }

    def _forward_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Dispatches one collated batch to the v1 or v2 forward by supervision.

        ``"dominant"`` (v1) consumes ``image``/``region_id``/``category_id`` via
        :meth:`step`. ``"region_category"`` (v2) consumes the cross-patch batch of
        :func:`ml.farslip.region_category_dataset.collate_region_batch`
        (``images``/``region_cat_ids``/``region_to_patch`` + optional
        ``caption_cls``) via :meth:`step_faithful_v2`. Both return a dict with a
        ``loss_total`` key the loop backpropagates.

        Args:
            batch: the collated batch dict (shape depends on the supervision mode).

        Returns:
            The loss dict from the selected forward (grad active).

        Raises:
            KeyError: if the batch lacks the keys the active mode requires.
        """
        if self.config.supervision == "region_category":
            return self.step_faithful_v2(
                batch["images"],
                batch["region_cat_ids"],
                batch["region_to_patch"],
                caption_cls=batch.get("caption_cls"),
            )
        return self.step(batch["image"], batch["region_id"], batch["category_id"])

    def train(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Runs ``n_epochs`` complete with MLflow autolog.

        Args:
            dataloader: optional. If not passed, requires ``self._dataset`` set.

        Returns:
            Dict with ``loss_total`` and the other final metrics (last epoch).
        """
        if dataloader is None:
            if self._dataset is None:
                raise RuntimeError("dataset and dataloader null: nothing to train")
            dataloader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
            )

        total_steps = max(1, len(dataloader) * self.config.n_epochs)
        self._scheduler = self._build_scheduler(total_steps)
        start = time.monotonic()
        warned = False

        # Import mlflow only once (Q14). ``mlflow`` stays ``None`` if the
        # library is not installed — the loop continues without remote logging.
        try:
            import mlflow as _mlflow
        except ImportError as exc:  # pragma: no cover
            _log.warning("mlflow no disponible", error=str(exc))
            _mlflow = None  # type: ignore[assignment]

        run_ctx = None
        if _mlflow is not None:
            try:
                _mlflow.set_experiment("farslip")
                run_ctx = _mlflow.start_run(run_name=self.config.mlflow_run_name)
                _mlflow.set_tags(
                    {
                        "code_version": git_sha(),
                        # data_version = hash of the .dvc file (not local path).
                        # If the dataset is not yet DVC-tracked, returns
                        # ``"<path>@untracked"`` and it is documented in the run.
                        "data_version": dvc_data_version(str(self.config.dataset_root)),
                        "us": "US-017",
                        "us_alias": "US-016b",
                    }
                )
                _mlflow.log_params(
                    {
                        "teacher_model_id": self.config.teacher_model_id,
                        "n_epochs": self.config.n_epochs,
                        "batch_size": self.config.batch_size,
                        "grad_accum_steps": self.config.grad_accum_steps,
                        "lr": self.config.lr,
                        "weight_decay": self.config.weight_decay,
                        "warmup_ratio": self.config.warmup_ratio,
                        "seed": self.config.seed,
                        "n_in_channels": self.config.n_in_channels,
                        "band_selection": self.config.band_selection,
                        "loss_alpha": self.config.loss_weights["alpha"],
                        "loss_beta": self.config.loss_weights["beta"],
                        "loss_gamma": self.config.loss_weights["gamma"],
                    }
                )
                if self.config.supervision == "region_category":
                    # US-036-a v2 faithful params (separate from the v1 path).
                    _mlflow.log_params(
                        {
                            "supervision": self.config.supervision,
                            "lambda_loc": self.config.lambda_loc,
                            "temperature": self.config.temperature,
                            "use_global_caption_loss": (self.config.use_global_caption_loss),
                            "n_categories": self.config.n_categories,
                        }
                    )
                if self.config.extra_params:
                    # US-034: proto_source/proto_proj/caveat + prototype dims.
                    _mlflow.log_params(dict(self.config.extra_params))
            except RuntimeError as exc:  # pragma: no cover
                _log.warning("mlflow init fallo", error=str(exc))
                run_ctx = None

        last_metrics: dict[str, float] = {}
        global_step = 0
        try:
            for epoch in range(self.config.n_epochs):
                for batch in dataloader:
                    elapsed_h = (time.monotonic() - start) / 3600.0
                    if elapsed_h >= self.config.time_cap_hours:
                        _log.error("hard time cap reached, stopping", elapsed_h=elapsed_h)
                        return last_metrics
                    if not warned and elapsed_h >= self.config.warning_hours:
                        _log.warning("training over warning threshold", elapsed_h=elapsed_h)
                        warned = True

                    losses = self._forward_batch(batch)
                    total = losses["loss_total"] / self.config.grad_accum_steps
                    total.backward()

                    if (global_step + 1) % self.config.grad_accum_steps == 0:
                        self._optim.step()
                        if self._scheduler is not None:
                            self._scheduler.step()
                        self._optim.zero_grad(set_to_none=True)

                    if run_ctx is not None and _mlflow is not None:
                        try:
                            _mlflow.log_metrics(
                                {k: float(v.detach().cpu().item()) for k, v in losses.items()},
                                step=global_step,
                            )
                        except RuntimeError as exc:  # pragma: no cover
                            _log.debug("mlflow log_metrics fallo", error=str(exc))

                    last_metrics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
                    global_step += 1

                _log.info("epoch done", epoch=epoch, **last_metrics)
                # Checkpoint per epoch (resilience AC-9 R3)
                self.save_student(format="safetensors", suffix=f"epoch_{epoch}")
        finally:
            if run_ctx is not None and _mlflow is not None:
                try:
                    _mlflow.end_run()
                except RuntimeError as exc:  # pragma: no cover
                    _log.debug("mlflow end_run fallo", error=str(exc))

        return last_metrics

    def save_student(
        self,
        format: SaveFormat = "safetensors",
        suffix: str | None = None,
    ) -> str:
        """Persists the student weights.

        Args:
            format: ``"safetensors"`` (default) or ``"pytorch"``.
            suffix: optional, e.g. ``"epoch_3"``; file suffix.

        Returns:
            Absolute local path of the written file.
        """
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = "student"
        if suffix:
            name = f"{name}_{suffix}"
        # ``self.student`` is CLIPVisionModel: the ENTIRE state_dict belongs to
        # the vision encoder (there is no text encoder in this wrapper). The
        # extractor loads directly into ``vision_model`` with ``strict=False`` to
        # tolerate prefix differences between CLIPVisionModel and CLIPModel.
        # We defensively filter `text_*` or `logit_scale` prefixes in case
        # a future iteration introduces a composite wrapper.
        raw_state = self.student.state_dict()
        state_dict = {
            k: v for k, v in raw_state.items() if not k.startswith(("text_", "logit_scale"))
        }
        if format == "safetensors":
            from safetensors.torch import save_file

            path = out_dir / f"{name}.safetensors"
            # safetensors requires contiguous tensors on CPU
            cpu_state = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
            save_file(cpu_state, str(path))
        else:
            path = out_dir / f"{name}.pt"
            torch.save(state_dict, path)
        _log.info("student weights saved", path=str(path), format=format)
        return str(path.resolve())


def build_default_trainer(
    dataset_root: Path = Path("data/farslip_pairs"),
    output_dir: Path = Path("artifacts/farslip"),
    **overrides: Any,
) -> FarSLIPDistillationTrainer:
    """Ergonomic factory with defaults validated during planning."""

    cfg = FarSLIPTrainerConfig(dataset_root=dataset_root, output_dir=output_dir)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"FarSLIPTrainerConfig has no attribute {k!r}")
        setattr(cfg, k, v)
    return FarSLIPDistillationTrainer(cfg)


__all__ = [
    "FarSLIPDistillationTrainer",
    "FarSLIPTrainerConfig",
    "PatchDistillationLoss",
    "RegionCategoryAlignmentLoss",
    "SupervisionMode",
    "build_default_trainer",
]
