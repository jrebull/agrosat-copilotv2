"""Inference and visualization of dense segmentation predictions.

Loads a trained checkpoint (``best.pt`` from
:mod:`ml.train.train_segmentation`), predicts over PASTIS-R patches and
generates the comparison figure ``Input (RGB) | Ground truth | Prediction``
per patch, plus the metrics of the loaded model. It is the module the
``notebooks/models/5*`` notebook invokes for visual analysis; the notebook
calls, it does not implement the logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ml.eval.checkpoint_registry import CheckpointSpec

logger = structlog.get_logger(__name__)

#: RGB bands in the PASTIS-R .npy files (S2 order of 10: B2,B3,B4,...): B4(red)=2,
#: B3(green)=1, B2(blue)=0.
_RGB_BANDS = (2, 1, 0)

#: Dummy temporal length used to materialize lazy parameters (AnySat head is a
#: ``LazyConv2d`` whose channel count is only known after a first forward).
_DUMMY_TIMESTEPS = 4
#: Spatial side of the dummy forward for the lazy-head materialization.
_DUMMY_SIZE = 256

#: AnySat spatial encoder side used for both the lazy-head dummy forward and the
#: real inference, replicating the configuration that produced the reported
#: mIoU 0.4459 (notebook ``04h_segmentation_anysat_fast``: ``TARGET_SIZE = 64``).
#:
#: Rationale: AnySat's ``forward(x, patch_size, output="dense")`` delegates to
#: ``forward_release(x, patch_size // 10, ...)``; for Sentinel-2 (``res=10`` m)
#: the spatial-token grid is ``(H / (patch_size // 10))`` per side, so with the
#: default ``patch_size=10`` (``scale=1``) there is ONE token per input pixel.
#: The self-attention then scales as ``(H * W) ** 2``: 64 px -> 4096 tokens
#: (manageable), 128 px -> 16384 tokens (~12 GB), 256 px -> 65536 tokens
#: (~206 GB, the observed OOM). Feeding AnySat at 64 px keeps memory bounded and
#: matches the resolution the checkpoint was trained/evaluated at.
_ANYSAT_ENCODER_SIZE = 64

#: AnySat output side. Set to 128 so the upsampled logits already live on the
#: harness 128-grid; the 64 px prediction never has to be resized externally.
_ANYSAT_TARGET_SIZE = 128


def _resolve_inference_device(device: str) -> torch.device:
    """Resolve the inference device, preferring CUDA when available.

    Args:
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved :class:`torch.device` (CUDA only if requested/auto and a GPU
        is present, otherwise CPU).
    """
    return torch.device(
        "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )


def build_model_for_kind(
    spec: CheckpointSpec,
    *,
    n_timesteps: int = 10,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Rebuild the bare model topology for a checkpoint kind (no weights loaded).

    Centralizes the per-architecture construction the harness needs for the six
    real segmentation checkpoints. Crucially it builds the head with the model's
    NATIVE class count (``spec.native_num_classes``): the 20-class models
    (U-Net, U-TAE, AnySat, SegFormer) are reconstructed with 20 outputs so the
    checkpoint keys (e.g. U-TAE ``out_conv``) stay intact; the 20->18 mapping
    happens later, purely in prediction space (``ml/AGENTS.md`` R1).

    SegFormer is NOT built here: it is loaded directly from its HuggingFace
    directory via ``from_pretrained`` inside :func:`load_checkpoint_model`.

    Args:
        spec: Static descriptor of the checkpoint (kind, native class count,
            input bands).
        n_timesteps: Subsampled temporal length ``T`` for the temporal models
            (TSViT / U-TAE / AnySat).
        device: Device on which the dummy forward (AnySat lazy head) runs.

    Returns:
        An un-loaded :class:`torch.nn.Module` with the native topology.

    Raises:
        ValueError: if ``spec.model_kind`` is not one of the six supported kinds
            (or is ``"segformer"``, which is handled by the HF loader).
    """
    kind = spec.model_kind
    classes = spec.native_num_classes
    resolved_device = torch.device(device)

    if kind == "unet":
        from ml.models.segmentation import build_unet

        return build_unet(classes, in_channels=spec.in_channels)
    if kind == "deeplabv3plus":
        from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet

        return build_deeplabv3plus_mobilenet(in_channels=spec.in_channels, classes=classes)
    if kind in ("tsvit", "tsvit-pheno", "tsvit-pheno-fullm"):
        from ml.models.tsvit_wrapper import build_tsvit

        # Default-L4 topology; ``spec.model_kwargs`` overrides it with the trained
        # capacity (US-038 Full-M: dim=192, depth 6+6, heads=6, dim_head=64,
        # n_timesteps=64). Without this override the harness would rebuild an
        # L4 TSViT (dim=128, depth 4+4) and ``load_state_dict`` would raise a
        # shape mismatch against the Full-M ``best.pt`` (R-HARNESS). ``n_timesteps``
        # in ``model_kwargs`` also sizes the ordinal temporal PE to the full
        # series length (R-TLEN).
        tsvit_kwargs: dict[str, int] = {
            "num_classes": classes,
            "n_timesteps": n_timesteps,
            "img_size": 128,
            "in_channels": spec.in_channels,
            "semantic_dim": 384,
        }
        tsvit_kwargs.update(spec.model_kwargs)
        return build_tsvit(**tsvit_kwargs)
    if kind == "utae":
        from ml.models.utae import build_utae

        # ALWAYS 20 classes: renaming `out_conv` would break the checkpoint keys.
        return build_utae(num_classes=classes, input_dim=spec.in_channels)
    if kind == "anysat":
        from ml.models.anysat_wrapper import AnySatSegmenter

        # Build with the default ``patch_size=10`` (``scale=1``) and a 128 px
        # output so the logits land on the harness grid directly. The dummy
        # forward runs at ``_ANYSAT_ENCODER_SIZE`` (64 px): a 256 px dummy would
        # build a 65536-token attention map (~206 GB) and OOM. See the
        # ``_ANYSAT_ENCODER_SIZE`` rationale and the 0.4459 reference config.
        model = AnySatSegmenter(num_classes=classes, target_size=_ANYSAT_TARGET_SIZE)
        # The head is a `LazyConv2d`; a dummy forward materializes its channel
        # count BEFORE `load_state_dict` (otherwise the weights cannot bind).
        # The encoder is loaded by AnySatSegmenter via torch.hub; if that fails
        # (no network), the harness reports status="missing" upstream.
        model.to(resolved_device).eval()
        dummy = torch.zeros(
            1,
            _DUMMY_TIMESTEPS,
            spec.in_channels,
            _ANYSAT_ENCODER_SIZE,
            _ANYSAT_ENCODER_SIZE,
            device=resolved_device,
        )
        # The real AnySat encoder requires per-frame dates (``s2_dates``); calling
        # forward without them raises ``KeyError('s2_dates')``. Pass a monotonic
        # day-of-year dummy ``(1, T)`` so the LazyConv2d head materializes against
        # the genuine encoder feature dimension.
        dummy_dates = torch.arange(
            _DUMMY_TIMESTEPS, device=resolved_device, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            model(dummy, dummy_dates)
        return model
    if kind == "segformer":
        raise ValueError(
            "SegFormer is loaded from its HuggingFace directory, not built here; "
            "use load_checkpoint_model(spec)."
        )
    raise ValueError(f"Unsupported model_kind: {kind!r}.")


def load_checkpoint_model(
    spec: CheckpointSpec,
    *,
    n_timesteps: int = 10,
    device: str = "auto",
) -> torch.nn.Module:
    """Build the model for ``spec`` and load its weights, tolerant to conventions.

    The single entry point the re-score harness uses to materialize any of the
    six checkpoints. It resolves the ``state_dict`` via
    :func:`ml.eval.checkpoint_registry.resolve_state_dict` (three conventions:
    ``model_state`` / ``model_state_dict`` / pure state_dict) and loads it
    non-strictly so harmless buffer/key mismatches do not abort the load.

    SegFormer is special-cased: it is a HuggingFace
    ``SegformerForSemanticSegmentation`` restored with ``from_pretrained`` from
    ``spec.path`` (a directory), not a bare ``state_dict``.

    Args:
        spec: Checkpoint descriptor (path, kind, native class count, state keys).
        n_timesteps: Subsampled ``T`` for the temporal models.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.

    Returns:
        Model in ``eval()`` mode on the resolved device with weights loaded.

    Raises:
        FileNotFoundError: if ``spec.path`` does not exist on disk.
    """
    from ml.eval.checkpoint_registry import resolve_state_dict

    resolved_device = _resolve_inference_device(device)
    if not spec.path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {spec.path}")

    if spec.model_kind == "segformer":
        from transformers import SegformerForSemanticSegmentation

        model: torch.nn.Module = SegformerForSemanticSegmentation.from_pretrained(str(spec.path))
        model.to(resolved_device).eval()
        logger.info(
            "segmentation_model_loaded",
            checkpoint=str(spec.path),
            model_kind=spec.model_kind,
            num_classes=spec.native_num_classes,
            convention="hf_from_pretrained",
            device=str(resolved_device),
        )
        return model

    model = build_model_for_kind(spec, n_timesteps=n_timesteps, device=resolved_device)
    loaded = torch.load(spec.path, map_location=resolved_device, weights_only=False)
    state = resolve_state_dict(loaded, spec)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(resolved_device).eval()
    logger.info(
        "segmentation_model_loaded",
        checkpoint=str(spec.path),
        model_kind=spec.model_kind,
        num_classes=spec.native_num_classes,
        n_missing_keys=len(missing),
        n_unexpected_keys=len(unexpected),
        device=str(resolved_device),
    )
    return model


def load_segmentation_model(
    checkpoint_path: Path | str,
    *,
    model_kind: Literal["deeplabv3plus", "tsvit", "tsvit-pheno"],
    num_classes: int = 18,
    n_timesteps: int = 10,
    device: str = "auto",
) -> torch.nn.Module:
    """Rebuild the model and load the checkpoint weights.

    Args:
        checkpoint_path: Path to ``best.pt`` (full training state).
        model_kind: Architecture to rebuild the exact topology.
        num_classes: Number of head classes (18 semantic or 6 HCAT).
        n_timesteps: Subsampled T (temporal models only).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        Model in ``eval()`` mode with the best-epoch weights loaded.
    """
    from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet
    from ml.models.tsvit_wrapper import build_tsvit

    resolved_device = _resolve_inference_device(device)
    if model_kind == "deeplabv3plus":
        model: torch.nn.Module = build_deeplabv3plus_mobilenet(in_channels=10, classes=num_classes)
    else:
        model = build_tsvit(
            num_classes=num_classes,
            n_timesteps=n_timesteps,
            img_size=128,
            in_channels=10,
            semantic_dim=384,
        )
    ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(resolved_device).eval()
    logger.info(
        "segmentation_model_loaded",
        checkpoint=str(checkpoint_path),
        model_kind=model_kind,
        num_classes=num_classes,
        best_epoch=ckpt.get("best_metrics", {}).get("best_epoch"),
        device=str(resolved_device),
    )
    return model


@torch.no_grad()
def predict_patch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
    doy: torch.Tensor | None = None,
) -> np.ndarray:
    """Predict the dense mask of a patch.

    Args:
        model: Model loaded in ``eval()``.
        x: Patch tensor: ``(10, H, W)`` (2D) or ``(T, 10, H, W)``
            (temporal). The batch dimension is added internally.
        model_kind: Architecture (decides whether ``doy`` is passed).
        doy: DOY vector ``(T,)`` for the temporal models.

    Returns:
        Mask ``(H, W)`` int with the predicted class per pixel (in the
        ``[0..num_classes-1]`` space).
    """
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device).float()
    if model_kind == "deeplabv3plus":
        logits = model(xb)
    else:
        doy_b = doy.unsqueeze(0).to(device) if doy is not None else None
        out = model(xb, doy=doy_b)
        logits = out[0] if isinstance(out, tuple) else out
    mask: np.ndarray = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return mask


def _ordinal_positions(n_timesteps: int, *, device: torch.device) -> torch.Tensor:
    """Build day-of-year positions ``(1, T)`` for the temporal models.

    The harness dataset (:class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`)
    delivers ``(x, y)`` only, without the per-frame acquisition dates. To feed
    U-TAE (whose forward requires ``batch_positions``) we reproduce the training
    convention used in :mod:`ml.tune.optuna_segmentation`: equispaced indices
    mapped onto the ``[0, 364]`` day-of-year range. TSViT instead falls back to
    its learned ordinal temporal PE (``doy=None``), matching how it was trained.

    Args:
        n_timesteps: Number of subsampled frames ``T``.
        device: Target device of the positions tensor.

    Returns:
        Long tensor ``(1, T)`` of day-of-year positions.
    """
    if n_timesteps <= 1:
        return torch.zeros(1, max(n_timesteps, 1), dtype=torch.long, device=device)
    idx = torch.arange(n_timesteps, dtype=torch.float32, device=device)
    doy = (idx / float(n_timesteps - 1) * 364.0).round().long()
    return doy.unsqueeze(0)


@torch.no_grad()
def _forward_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
) -> torch.Tensor:
    """Run the per-architecture forward and return the raw class logits.

    Single source of truth for the forward dispatch shared by
    :func:`predict_patch_for_kind` (which adds an ``argmax``) and
    :func:`softmax_patch_for_kind` (which adds a ``softmax``). Keeping the
    dispatch here avoids duplicating the unet/utae/anysat branches and guarantees
    both consumers see byte-identical logits (so ``argmax(softmax) == argmax``).

    The forward matches the upstream harness contract (the dataset builds the
    correct ``x`` per architecture):

    - ``unet`` / ``deeplabv3plus``: 2D input ``(10, H, W)`` -> ``model(xb)``.
    - ``tsvit`` / ``tsvit-pheno``: temporal ``(T, 10, H, W)`` -> ``model(xb)``
      (ordinal temporal PE, ``doy`` not supplied, as trained).
    - ``utae``: temporal ``(T, 10, H, W)`` -> ``model(xb, batch_positions)`` with
      reconstructed day-of-year positions.
    - ``anysat``: temporal ``(T, 10, H, W)`` resized to the 64 px encoder grid ->
      ``model(image, dates)`` (the model upsamples its logits back to 128 px).

    SegFormer is NOT handled here: it runs its own 3-RGB / 256 sub-pipeline over
    raw S2 (see :func:`softmax_logits_segformer`).

    Args:
        model: Model loaded in ``eval()``.
        x: Patch tensor with the architecture-appropriate shape (the batch
            dimension is added internally).
        model_kind: Architecture tag.

    Returns:
        Raw logits tensor ``(1, C, H, W)`` on the model device (``C`` is the
        model's NATIVE class count; no 20->18 remap applied here).

    Raises:
        ValueError: if ``model_kind`` is not a supported (non-SegFormer) kind.
    """
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device).float()

    logits: torch.Tensor
    if model_kind in ("unet", "deeplabv3plus"):
        logits = model(xb)
    elif model_kind in ("tsvit", "tsvit-pheno", "tsvit-pheno-fullm"):
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
    elif model_kind == "utae":
        positions = _ordinal_positions(xb.shape[1], device=device)
        out = model(xb, positions)
        logits = out[0] if isinstance(out, tuple) else out
    elif model_kind == "anysat":
        # Downsample the temporal series to AnySat's 64 px encoder resolution
        # (the 0.4459 reference config) BEFORE the forward: at the native 128 px
        # the s2 spatial-token grid is 16384 tokens and the self-attention OOMs
        # (256 px would be ~206 GB). Each frame ``(B, T, C, H, W)`` is resized
        # bilinearly on its spatial plane; the model upsamples its logits back to
        # 128 px (``_ANYSAT_TARGET_SIZE``), so the logits land on the harness
        # grid and need no external resize.
        import torch.nn.functional as F

        b, t, c, _, _ = xb.shape
        xb_small = F.interpolate(
            xb.reshape(b * t, c, *xb.shape[-2:]),
            size=(_ANYSAT_ENCODER_SIZE, _ANYSAT_ENCODER_SIZE),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, c, _ANYSAT_ENCODER_SIZE, _ANYSAT_ENCODER_SIZE)
        dates = _ordinal_positions(t, device=device)
        out = model(xb_small, dates)
        logits = out[0] if isinstance(out, tuple) else out
    else:
        raise ValueError(
            f"_forward_logits does not handle model_kind={model_kind!r} "
            "(SegFormer runs its own sub-pipeline)."
        )
    return logits


@torch.no_grad()
def predict_patch_for_kind(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
) -> np.ndarray:
    """Predict the dense class map of a patch for any of the six architectures.

    Thin wrapper over :func:`_forward_logits` plus an ``argmax`` over the class
    axis. The forward dispatch lives in :func:`_forward_logits` (DRY); this
    function only collapses the logits to a class map, preserving the exact
    output the US-030 harness consumes.

    SegFormer is NOT handled here (it runs its own 3-RGB/256 sub-pipeline in the
    harness directly over raw S2).

    Args:
        model: Model loaded in ``eval()``.
        x: Patch tensor with the architecture-appropriate shape (the batch
            dimension is added internally).
        model_kind: Architecture tag.

    Returns:
        Predicted class map ``(H, W)`` as ``int64`` numpy in the model's NATIVE
        class space (no 20->18 remap applied here).

    Raises:
        ValueError: if ``model_kind`` is not a supported (non-SegFormer) kind.
    """
    logits = _forward_logits(model, x, model_kind=model_kind)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


@torch.no_grad()
def softmax_patch_for_kind(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
) -> np.ndarray:
    """Return the POST-softmax probability map of a patch for any architecture.

    Twin of :func:`predict_patch_for_kind` but, instead of the argmax, applies
    ``torch.softmax(logits, dim=1)`` on the class axis and returns the full
    probability tensor. Shares the per-architecture forward via the private
    :func:`_forward_logits` helper (no duplicated unet/utae/anysat dispatch), so
    ``softmax_patch_for_kind(...).argmax(0) == predict_patch_for_kind(...)``.

    The probabilities are POST-softmax by ensemble anti-leakage convention
    (US-040: never average logits). The class count is the model's NATIVE space
    (20 for unet/utae/anysat, 18 for deeplabv3plus/tsvit-pheno); the 20->18 remap
    and 128 resample happen DOWNSTREAM in probability space
    (:func:`ml.eval.class_remap.remap_probs_20_to_18` /
    :func:`ml.eval.class_remap.resample_probs_128_bilinear`).

    SegFormer is NOT handled here (it runs its own 3-RGB / 256 sub-pipeline; see
    :func:`softmax_logits_segformer`).

    Args:
        model: Model loaded in ``eval()``.
        x: Patch tensor with the architecture-appropriate shape (the batch
            dimension is added internally).
        model_kind: Architecture tag.

    Returns:
        Probability map ``(C_native, H, W)`` float32 with ``probs.sum(0) ~ 1``.

    Raises:
        ValueError: if ``model_kind`` is not a supported (non-SegFormer) kind.
    """
    logits = _forward_logits(model, x, model_kind=model_kind)
    probs = torch.softmax(logits, dim=1)
    out: np.ndarray = probs.squeeze(0).cpu().numpy().astype(np.float32)
    return out


#: SegFormer (Isaac, notebook 04i) 3-RGB normalization constants and train size,
#: reproduced verbatim from :mod:`ml.eval.dense_metrics` so the softmax dump can
#: run SegFormer's sub-pipeline without importing (and serializing against) the
#: US-030 harness module.
_SEGFORMER_RGB_MEAN = np.array([1158.0, 1244.7, 1416.3], dtype=np.float32)[:, None, None]
_SEGFORMER_RGB_STD = np.array([671.7, 698.1, 761.3], dtype=np.float32)[:, None, None]
_SEGFORMER_SIZE = 256


@torch.no_grad()
def softmax_logits_segformer(
    model: torch.nn.Module,
    pid: str,
    *,
    root: Path,
    device: torch.device,
) -> np.ndarray:
    """Run SegFormer's 3-RGB / 256 sub-pipeline and return its native softmax.

    SegFormer (Isaac) was trained on a 3-band RGB temporal-median composite at
    256 px with its own normalization (``_SEGFORMER_RGB_*``). This reproduces
    that exact input pipeline (mirror of ``dense_metrics._segformer_predict_18``
    but POST-softmax instead of argmax+remap, so the dump can remap/resample in
    PROBABILITY space downstream). The logits are softmaxed over the 20-class
    axis at 256 px; the caller resamples to 128 bilinear and remaps 20->18.

    Args:
        model: Loaded ``SegformerForSemanticSegmentation``.
        pid: PASTIS patch id.
        root: PASTIS-R root directory.
        device: Inference device.

    Returns:
        Probability map ``(20, 256, 256)`` float32, ``probs.sum(0) ~ 1``.
    """
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")  # (T, C, H, W)
    img = np.median(s2, axis=0)[:3].astype(np.float32)  # RGB composite
    img = (img - _SEGFORMER_RGB_MEAN) / (_SEGFORMER_RGB_STD + 1e-6)
    t_img = (
        TF.resize(
            torch.from_numpy(img),
            [_SEGFORMER_SIZE, _SEGFORMER_SIZE],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        .unsqueeze(0)
        .to(device)
    )
    logits = model(pixel_values=t_img).logits
    logits = F.interpolate(
        logits,
        size=(_SEGFORMER_SIZE, _SEGFORMER_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.softmax(logits, dim=1)
    out: np.ndarray = probs.squeeze(0).cpu().numpy().astype(np.float32)
    return out


def prediction_figure(
    rgb: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 18,
    ignore_index: int = 255,
    titles: tuple[str, str, str] = ("Input (RGB)", "Ground truth", "Prediction"),
) -> Figure:
    """Build the comparison figure RGB | ground truth | prediction.

    Args:
        rgb: RGB image ``(H, W, 3)`` in ``[0, 1]``.
        y_true: Ground-truth mask ``(H, W)`` (classes ``[0..C-1]`` + ``ignore_index``).
        y_pred: Predicted mask ``(H, W)``.
        num_classes: Number of classes for the discrete colormap.
        ignore_index: Value ignored in ``y_true`` (drawn neutral).
        titles: Titles of the three panels.

    Returns:
        1x3 matplotlib figure ready for ``display(fig)`` in the notebook.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cmap = plt.get_cmap("tab20", num_classes)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)

    yt = np.where(y_true == ignore_index, np.nan, y_true.astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[1].imshow(yt, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].imshow(y_pred.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def rgb_from_patch(x_2d: np.ndarray) -> np.ndarray:
    """Extract a normalized RGB image ``(H, W, 3)`` from a 2D patch.

    Takes the B4/B3/B2 bands and rescales by percentiles (2-98) for a
    reasonable visual contrast (raw S2 reflectances are dark).

    Args:
        x_2d: Patch ``(10, H, W)`` (already time-collapsed, any scale).

    Returns:
        ``(H, W, 3)`` float array in ``[0, 1]``.
    """
    rgb = np.stack([x_2d[b] for b in _RGB_BANDS], axis=-1).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1.0
    stretched: np.ndarray = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return stretched


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    num_classes: int = 18,
    ignore_index: int = 255,
    max_patches: int | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate a checkpoint over a split accumulating the confusion matrix.

    Walks the validation ``dataset`` patch by patch, predicts and accumulates
    the dense confusion matrix; at the end it derives all metrics with
    :func:`ml.eval.metrics.dense_metrics_from_cm` (mIoU, F1-macro, pixel_acc,
    balanced accuracy, Cohen kappa, IoU and F1 per class). It is the helper the
    ``5*`` notebooks invoke to reproduce the training figures without
    retraining: they load ``best.pt`` and call here.

    Args:
        model: Model loaded in ``eval()`` (see :func:`load_segmentation_model`).
        dataset: ``PASTISSegmentationDataset`` of the validation split.
        model_kind: Architecture (decides the forward signature).
        num_classes: Number of classes (18 semantic or 6 HCAT).
        ignore_index: Value ignored in the labels.
        max_patches: If given, limits the number of evaluated patches (smoke).

    Returns:
        Tuple ``(metrics, cm)``: ``metrics`` is the full dict of
        ``dense_metrics_from_cm`` and ``cm`` the accumulated confusion matrix
        ``(num_classes, num_classes)``.
    """
    from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm

    n = len(dataset)  # type: ignore[arg-type]
    if max_patches is not None:
        n = min(n, max_patches)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx in range(n):
        x, y = dataset[idx]  # type: ignore[index]
        pred = predict_patch(model, x, model_kind=model_kind)
        cm += dense_confusion_matrix(
            pred, y.numpy(), n_classes=num_classes, ignore_index=ignore_index
        )
    metrics = dense_metrics_from_cm(cm)
    logger.info(
        "checkpoint_evaluated",
        model_kind=model_kind,
        n_patches=n,
        num_classes=num_classes,
        miou=round(float(metrics["miou"]), 4),
        f1_macro=round(float(metrics["f1_macro"]), 4),
        pixel_acc=round(float(metrics["pixel_acc"]), 4),
    )
    return metrics, cm


def predict_examples(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    indices: list[int],
    num_classes: int = 18,
    ignore_index: int = 255,
) -> list[Figure]:
    """Generate the RGB|GT|pred figures for a list of dataset patches.

    High-level helper for the notebook: for each index, it gets the patch,
    predicts, builds the RGB and constructs the comparison figure. For
    temporal models it collapses the series by median only for the RGB panel.

    Args:
        model: Model loaded in ``eval()``.
        dataset: ``PASTISSegmentationDataset`` (2D or temporal depending on the model).
        model_kind: Architecture.
        indices: Indices of the patches to visualize.
        num_classes: Number of classes.
        ignore_index: Ignored value.

    Returns:
        List of matplotlib figures (one per index).
    """
    figs: list[Figure] = []
    for idx in indices:
        x, y = dataset[idx]  # type: ignore[index]
        x_np = x.numpy()
        if x_np.ndim == 4:  # temporal (T,10,H,W) -> RGB from the temporal median
            rgb = rgb_from_patch(np.median(x_np, axis=0))
        else:  # 2D (10,H,W)
            rgb = rgb_from_patch(x_np)
        pred = predict_patch(model, x, model_kind=model_kind)
        figs.append(
            prediction_figure(
                rgb,
                y.numpy(),
                pred,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        )
    return figs
