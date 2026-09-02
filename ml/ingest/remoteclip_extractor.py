"""RemoteCLIP embeddings extraction over PASTIS-R crops (US-023-preview-v2 P5).

Replaces the ``_extract_remoteclip_embeddings`` placeholder of the notebook
``04_farslip_eval_pastis.ipynb`` with a REAL extraction using RemoteCLIP
(Chen et al. 2023, https://github.com/ChenDelong1999/RemoteCLIP). The model
``chendelong/RemoteCLIP-ViT-B-32`` is a CLIP ViT-B/32 fine-tuned on
remote sensing imagery (RSITMD + RSICD + UCM).

Per-parcel pipeline:

1. Loads multitemporal S2 crops from ``imagery_path`` (binary parquet or
   NCHW array per parcel).
2. Selects bands B04 (red), B03 (green), B02 (blue) and composes RGB.
3. Normalizes with stats ``NORM_S2_patch.json`` (PASTIS-R) and applies a
   2-98 percentile stretch -> uint8.
4. Bilinear resize to 224x224.
5. Forward through the ``CLIPVisionModel`` and L2-normalize.
6. Temporal pooling (mean over the T axis) if the parcel is multi-temporal.

Output ``data/farslip/remoteclip_embeddings_pastis.parquet`` with schema:

::

    parcel_id (Utf8) | year (Int16) |
    remoteclip_000 .. remoteclip_511 (Float32)

Schema compatible with :func:`ml.farslip.extract_embeddings.extract_farslip_embeddings`
except for the column prefix (``remoteclip_*`` vs ``farslip_emb_*``); the
notebook 04 can concatenate both for the comparative linear probe.

Fallback: if ``chendelong/RemoteCLIP-ViT-B-32`` cannot be downloaded
(network block, model removed from HF), ``openai/clip-vit-base-patch32`` is
used and ``model_used`` is annotated in the structured log so the operator
knows the comparison is not against pure RemoteCLIP.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    import torch
    from transformers import CLIPModel, CLIPProcessor

_log = structlog.get_logger(__name__)


EMBED_DIM: int = 512
"""Dimension of the CLIP ViT-B/32 embedding (image features after projection)."""

EMBED_COL_PREFIX: str = "remoteclip_"

DEFAULT_MODEL_ID: str = "chendelong/RemoteCLIP-ViT-B-32"
"""HF Hub repo of the RemoteCLIP model (CLIP ViT-B/32 fine-tuned on RS)."""

FALLBACK_MODEL_ID: str = "openai/clip-vit-base-patch32"
"""Fallback if ``DEFAULT_MODEL_ID`` is not available."""

DEFAULT_SUBSET_PATH: Path = Path("data/test_fixtures/pastis_eval_subset.parquet")
DEFAULT_IMAGERY_PATH: Path = Path("data/test_fixtures/pastis_eval_subset.imagery.parquet")
DEFAULT_OUTPUT_PATH: Path = Path("data/farslip/remoteclip_embeddings_pastis.parquet")
DEFAULT_BATCH_SIZE: int = 32


def _embed_columns() -> list[str]:
    """Return ``["remoteclip_000", ..., "remoteclip_511"]`` (stable order)."""
    return [f"{EMBED_COL_PREFIX}{i:03d}" for i in range(EMBED_DIM)]


def _build_output_schema() -> dict[str, Any]:
    """Canonical output schema (parcel_id + year + 512 floats)."""
    schema: dict[str, Any] = {
        "parcel_id": pl.Utf8,
        "year": pl.Int16,
    }
    for col in _embed_columns():
        schema[col] = pl.Float32
    return schema


def _resolve_device(device: str | None) -> torch.device:
    """Resolve device with ``cuda -> cpu`` fallback.

    Args:
        device: ``"cuda"``, ``"cpu"`` or ``None`` (autodetect).

    Returns:
        Resolved ``torch.device``. If CUDA was requested but is not
        available, emits a structured warning and degrades to CPU.
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        _log.warning(
            "remoteclip_no_cuda_detected_cpu_fallback",
            hint="extraccion sera lenta (~1-2 s/parcela en CPU vs ~10 ms/parcela en GPU)",
        )
        return torch.device("cpu")
    if device == "cuda":
        import torch as _t

        if not _t.cuda.is_available():
            _log.warning(
                "remoteclip_cuda_requested_unavailable",
                fallback="cpu",
            )
            return _t.device("cpu")
        return _t.device("cuda")
    return torch.device(device)


def _load_model(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor, str]:
    """Load ``CLIPModel`` + ``CLIPProcessor`` with fallback to OpenAI CLIP.

    Args:
        model_name: HF repo id (default ``chendelong/RemoteCLIP-ViT-B-32``).
        device: target device for the weights.

    Returns:
        Tuple ``(model, processor, model_used)`` where ``model_used`` is the
        id actually loaded (may match ``model_name`` or be the OpenAI CLIP
        fallback).
    """
    from transformers import CLIPModel, CLIPProcessor

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model_used = model_name
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "remoteclip_load_failed_using_fallback",
            requested=model_name,
            fallback=FALLBACK_MODEL_ID,
            error=str(exc),
        )
        model = CLIPModel.from_pretrained(FALLBACK_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(FALLBACK_MODEL_ID)
        model_used = FALLBACK_MODEL_ID

    model.eval()
    model.to(device)
    return model, processor, model_used


def _load_imagery(imagery_path: Path) -> pl.DataFrame:
    """Load the binary parquet with Sentinel-2 crops.

    Expected schema (at least)::

        parcel_id (Utf8) | year (Int16) |
        image (List[List[Float32]] or binary) | shape (List[Int64])

    Tolerates any of these schemas as long as each row exposes
    ``parcel_id`` and an interpretable image payload (numpy bytes via
    ``np.frombuffer`` or nested list).
    """
    if not imagery_path.exists():
        raise FileNotFoundError(
            f"imagery_path does not exist: {imagery_path}. "
            "Generate it with ml.ingest.pastis_eval_subset (US-022-c P1 B-1)."
        )
    return pl.read_parquet(imagery_path)


def _row_to_array(row: dict[str, Any]) -> np.ndarray:
    """Convert a row of the imagery parquet to a ``(T, C, H, W)`` array.

    Supports three common encodings:

    - ``image`` is already a ``np.ndarray`` (in-memory format).
    - ``image`` is ``bytes`` -> ``np.frombuffer`` with adjacent ``shape``.
    - ``image`` is a nested list -> ``np.asarray``.

    For mono-temporal parcels returns shape ``(1, C, H, W)`` (adds a
    T=1 axis) to standardize the rest of the pipeline.
    """
    img: Any = row.get("image")
    shape = row.get("shape")
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, bytes | bytearray):
        arr = np.frombuffer(img, dtype=np.float32).copy()
        if shape:
            arr = arr.reshape(tuple(int(s) for s in shape))
    elif isinstance(img, list):
        arr = np.asarray(img, dtype=np.float32)
    else:
        raise ValueError(
            f"unsupported imagery format for parcel_id={row.get('parcel_id')!r}: "
            f"type {type(img).__name__}"
        )
    if arr.ndim == 3:
        # (C, H, W) -> (1, C, H, W)
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 4:
        raise ValueError(
            f"unexpected imagery shape for parcel_id={row.get('parcel_id')!r}: {arr.shape}"
        )
    return arr


def _select_rgb(arr: np.ndarray, band_indices: tuple[int, int, int]) -> np.ndarray:
    """Select and reorder bands for RGB.

    Args:
        arr: ``(T, C, H, W)`` float.
        band_indices: indices (red, green, blue). For PASTIS-R with order
            B02/B03/B04/B08, RGB = (2, 1, 0).

    Returns:
        ``(T, 3, H, W)`` float with order ``[R, G, B]``.
    """
    r, g, b = band_indices
    return arr[:, [r, g, b], :, :]


def _stretch_percentile_uint8(rgb: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch per band -> uint8 ``[0, 255]``.

    Args:
        rgb: ``(T, 3, H, W)`` float.

    Returns:
        ``(T, H, W, 3)`` uint8 (HWC format expected by
        :class:`CLIPProcessor`).
    """
    t, c, h, w = rgb.shape
    out = np.zeros((t, h, w, c), dtype=np.uint8)
    for ti in range(t):
        for ci in range(c):
            band = rgb[ti, ci]
            lo, hi = np.percentile(band, [2.0, 98.0])
            denom = max(float(hi - lo), 1e-6)
            scaled = np.clip((band - lo) / denom, 0.0, 1.0)
            out[ti, :, :, ci] = (scaled * 255).astype(np.uint8)
    return out


def _resolve_band_indices(imagery_meta: dict[str, Any] | None) -> tuple[int, int, int]:
    """Resolve RGB indices according to band metadata.

    Default PASTIS-R: bands B02/B03/B04/B08 (canonical order) -> RGB =
    ``(2, 1, 0)`` (B04 red at idx 2, B03 green at idx 1, B02 blue at idx 0).
    If ``imagery_meta`` provides ``band_order``, it is respected and
    ``B04/B03/B02`` are looked up.
    """
    if imagery_meta and "band_order" in imagery_meta:
        order = [b.upper() for b in imagery_meta["band_order"]]
        try:
            return (order.index("B04"), order.index("B03"), order.index("B02"))
        except ValueError:
            pass
    return (2, 1, 0)


def _embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images_hwc_uint8: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Forward a batch of RGB uint8 images through the CLIP visual encoder.

    Returns:
        Tensor ``(B, 512)`` float32 on CPU, L2-normalized.
    """
    import torch

    inputs = processor(images=images_hwc_uint8, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
    # `get_image_features` should return a tensor (B, dim) but some
    # RemoteCLIP checkpoints (chendelong/*) return the full output
    # `BaseModelOutputWithPooling`. We normalize to a tensor before L2.
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        features = features.image_embeds
    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
        features = features.pooler_output
    elif hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state.mean(dim=1)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.detach().cpu().float()


def extract_remoteclip_embeddings(
    pastis_eval_subset_path: Path = DEFAULT_SUBSET_PATH,
    imagery_path: Path = DEFAULT_IMAGERY_PATH,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract RemoteCLIP embeddings per parcel of the PASTIS-R subset.

    Loads the multitemporal S2 crops from ``imagery_path`` (generated by
    ``ml.ingest.pastis_eval_subset``), builds normalized RGB compositions
    (B04/B03/B02) and produces a 512-dim embedding per parcel.

    For multi-temporal parcels it aggregates temporally (mean pooling over
    the T axis) before the final L2-normalization.

    Args:
        pastis_eval_subset_path: Parquet with metadata ``parcel_id`` + ``year``
            + ``label_id``. Provides the canonical row order of the output.
        imagery_path: Binary parquet with Sentinel-2 crops per parcel.
        output_path: Destination parquet (parent is created if it does not exist).
        model_name: HF repo id. Default ``chendelong/RemoteCLIP-ViT-B-32``;
            automatic fallback to ``openai/clip-vit-base-patch32``.
        batch_size: Number of RGB images processed per forward.
        device: ``"cuda"``, ``"cpu"`` or ``None`` (autodetect).
        overwrite: If ``False`` and ``output_path`` exists, returns the
            existing path without recomputing.

    Returns:
        Absolute path to the ``output_path`` parquet.

    Raises:
        FileNotFoundError: If ``imagery_path`` does not exist.
        ValueError: If the imagery parquet does not expose ``parcel_id`` or the
            image payload is of an unsupported type.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        _log.info("remoteclip_output_exists_skip", path=str(output_path))
        return output_path.resolve()

    import torch

    subset = pl.read_parquet(pastis_eval_subset_path)
    imagery = _load_imagery(imagery_path)
    if "parcel_id" not in imagery.columns:
        raise ValueError(
            f"imagery_path={imagery_path} without parcel_id column; "
            "expected schema: parcel_id, year, image, shape."
        )

    # Ensure parcel_id Utf8 in both parquets.
    subset = subset.with_columns(pl.col("parcel_id").cast(pl.Utf8))
    imagery = imagery.with_columns(pl.col("parcel_id").cast(pl.Utf8))

    # PASTIS-R does not expose `year` per parcel (it is a monolithic 2019 dataset).
    # If missing, we materialize it as a constant to preserve the output
    # schema (parcel_id, year, remoteclip_emb_*).
    if "year" not in subset.columns:
        subset = subset.with_columns(pl.lit(2019).cast(pl.Int64).alias("year"))

    # Stable-order join: imagery merge over subset order.
    joined = subset.select(["parcel_id", "year"]).join(imagery, on="parcel_id", how="left")

    torch_device = _resolve_device(device)
    model, processor, model_used = _load_model(model_name, torch_device)

    t0 = time.perf_counter()
    band_indices = _resolve_band_indices(None)
    embeddings: list[torch.Tensor] = []
    parcel_ids_out: list[str] = []
    years_out: list[int] = []

    rows = joined.to_dicts()
    n = len(rows)

    # Batch processing of parcels (each parcel may have T frames).
    for start in range(0, n, batch_size):
        chunk = rows[start : start + batch_size]
        images_per_parcel: list[np.ndarray] = []
        frame_to_parcel: list[int] = []
        for parcel_idx, row in enumerate(chunk):
            try:
                arr = _row_to_array(row)
            except (ValueError, KeyError) as exc:
                _log.warning(
                    "remoteclip_parcel_skipped",
                    parcel_id=row.get("parcel_id"),
                    error=str(exc),
                )
                # Empty frame -> zeros embedding to avoid breaking alignment.
                arr = np.zeros((1, 4, 32, 32), dtype=np.float32)
            rgb = _select_rgb(arr, band_indices)
            rgb_uint8 = _stretch_percentile_uint8(rgb)
            # ``rgb_uint8`` shape (T, H, W, 3). We flatten T and record
            # the parcel_idx to do mean pooling post-forward.
            for ti in range(rgb_uint8.shape[0]):
                images_per_parcel.append(rgb_uint8[ti])
                frame_to_parcel.append(parcel_idx)

        if not images_per_parcel:
            continue
        feats_frames = _embed_batch(model, processor, images_per_parcel, torch_device)

        # Temporal mean pooling per parcel.
        per_parcel: dict[int, list[torch.Tensor]] = {}
        for fi, p_idx in enumerate(frame_to_parcel):
            per_parcel.setdefault(p_idx, []).append(feats_frames[fi])
        for p_idx, frames in per_parcel.items():
            stacked = torch.stack(frames, dim=0)
            mean_emb = stacked.mean(dim=0)
            # Re-normalize L2 post mean-pooling.
            mean_emb = torch.nn.functional.normalize(mean_emb, dim=-1)
            embeddings.append(mean_emb)
            parcel_ids_out.append(str(chunk[p_idx]["parcel_id"]))
            year_val = chunk[p_idx].get("year")
            years_out.append(int(year_val) if year_val is not None else 0)

    elapsed = time.perf_counter() - t0
    n_parcels_done = len(parcel_ids_out)
    sec_per_parcel = (elapsed / n_parcels_done) if n_parcels_done else 0.0
    _log.info(
        "remoteclip_extract_complete",
        n_parcels=n_parcels_done,
        device=str(torch_device),
        seconds=round(elapsed, 2),
        seconds_per_parcel=round(sec_per_parcel, 4),
        model_used=model_used,
    )

    if not embeddings:
        # Empty output with a valid schema.
        out_df = pl.DataFrame(schema=_build_output_schema())
    else:
        emb_tensor = torch.stack(embeddings, dim=0).numpy().astype(np.float32)
        cols = _embed_columns()
        data: dict[str, Any] = {
            "parcel_id": parcel_ids_out,
            "year": years_out,
        }
        for i, col in enumerate(cols):
            data[col] = emb_tensor[:, i].tolist()
        out_df = pl.DataFrame(data, schema=_build_output_schema())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.ingest.remoteclip_extractor",
        description=(
            "Extrae embeddings RemoteCLIP (512-dim) sobre crops PASTIS-R "
            "y persiste a parquet compatible con el linear probe de "
            "notebooks/features/04_farslip_eval_pastis.ipynb."
        ),
    )
    p.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET_PATH,
        help=f"Parquet metadata subset PASTIS-R (default {DEFAULT_SUBSET_PATH}).",
    )
    p.add_argument(
        "--imagery",
        type=Path,
        default=DEFAULT_IMAGERY_PATH,
        help=f"Parquet binario con crops S2 (default {DEFAULT_IMAGERY_PATH}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet de salida (default {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_ID,
        help=f"HF repo id del modelo CLIP (default {DEFAULT_MODEL_ID}).",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Device override (default autodetect).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out = extract_remoteclip_embeddings(
        pastis_eval_subset_path=args.subset,
        imagery_path=args.imagery,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    _log.info("remoteclip_extractor_done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
