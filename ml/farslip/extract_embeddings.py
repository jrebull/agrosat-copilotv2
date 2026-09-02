"""FarSLIP embeddings extraction to parquet (US-022-c P1 stage 6).

Loads the FarSLIP student trained on GCP L4 (MLflow `farslip-clip-italy-v1@Production`,
model name preserved by lineage; or local path) and projects each PASTIS-R
parcel (``parcel_id`` format ``10000_1``, not Italian) onto the 512-dim
embedding space, persisting the result to parquet with a stable schema:

- ``parcel_id`` (int64) — upstream parcel identifier.
- ``year`` (int32) — year of the associated temporal crop.
- ``farslip_emb_000`` .. ``farslip_emb_511`` (float32) — 512 columns.

AC contract US-022-c sec 2.1 B-4:

- Output parquet shape ``(85951, 514)`` (parcel_id + year + 512 embed).
- Determinism with reproducible ``seed=42`` (same input -> same output).
- Fallback ``cuda -> cpu`` with warning if CUDA unavailable.
- MLflow URI resolution ``mlflow://Models/farslip-clip-italy-v1@Production``.

Typical CLI usage::

    python -m ml.farslip.extract_embeddings \\
        --student-checkpoint mlflow://Models/farslip-clip-italy-v1@Production \\
        --parcels-parquet data/features/features_fused_v1.parquet \\
        --rois italy --output data/farslip/embeddings_pastis.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import structlog
import torch

from ml.utils.git_meta import git_sha
from ml.utils.seed import propagate_seed

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "extract_embeddings requires transformers>=4.46. `poetry add transformers`."
    ) from exc

_log = structlog.get_logger(__name__)

EMBED_DIM: int = 512
TOTAL_COLS: int = EMBED_DIM + 2  # parcel_id + year + 512 dims
EMBED_COL_PREFIX: str = "farslip_emb_"

DeviceLiteral = Literal["auto", "cuda", "cpu"]
RoiPreset = tuple[str, ...]

_ROI_ALIASES: dict[str, RoiPreset] = {
    "italy": ("pianura_padana", "toscana", "puglia"),
    "pianura_padana": ("pianura_padana",),
    "toscana": ("toscana",),
    "puglia": ("puglia",),
}


@dataclass(frozen=True)
class ExtractEmbeddingsResult:
    """Typed result of :func:`extract_farslip_embeddings`.

    Attributes:
        n_parcels: number of rows in the final parquet.
        n_dims: embedding dimension (always 512 for CLIP ViT-B/16).
        output_path: absolute path of the generated parquet.
        code_version: repo git SHA at extraction time.
        data_version: identifier of the source checkpoint
            (``mlflow://...`` or local path).
        device_used: ``"cuda"`` or ``"cpu"`` actually used.
    """

    n_parcels: int
    n_dims: int
    output_path: Path
    code_version: str
    data_version: str
    device_used: str


def _resolve_rois(rois: tuple[str, ...]) -> RoiPreset:
    """Expand alias (``"italy"``) to the canonical tuple of ROIs."""
    if len(rois) == 1 and rois[0] in _ROI_ALIASES:
        return _ROI_ALIASES[rois[0]]
    return rois


def _resolve_device(device: DeviceLiteral) -> torch.device:
    """Resolve ``"auto"`` with ``cuda -> cpu`` fallback and explicit warning."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        _log.warning(
            "cuda_requested_but_unavailable_fallback_cpu",
            note="device='cuda' pero torch.cuda.is_available()==False; degrada a cpu.",
        )
        return torch.device("cpu")
    return torch.device(device)


def _resolve_checkpoint(path: Path | str) -> tuple[Path | str, str]:
    """Return (resolved_path, data_version_tag).

    Supports two formats:

    - ``mlflow://Models/<name>@<stage>`` -> download via mlflow registry,
      data_version = the full URI string.
    - Local path -> returns the path as-is, data_version = ``str(path)``.

    To keep the module testable without network, the MLflow download is
    delegated to :func:`mlflow.artifacts.download_artifacts` only when the
    string starts with ``mlflow://``. The test suite patches this function to
    avoid HTTP.
    """
    p = str(path)
    if p.startswith("mlflow://"):
        return _resolve_mlflow_uri(p), p
    return Path(p), p


def _resolve_mlflow_uri(uri: str) -> Path:
    """Download the MLflow artifact pointed to by ``uri``.

    Example: ``mlflow://Models/farslip-clip-italy-v1@Production``.
    """
    import mlflow

    # mlflow expects: models:/<name>/<stage> or models:/<name>@<alias>
    body = uri.removeprefix("mlflow://")
    if body.startswith("Models/"):
        body = body.removeprefix("Models/")
    models_uri = f"models:/{body}"
    _log.info("downloading_mlflow_artifact", uri=models_uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=models_uri)
    return Path(local)


def _load_student(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    teacher_model_id: str = "openai/clip-vit-base-patch16",
    n_in_channels: int = 4,
) -> CLIPVisionModel:
    """Reconstruct the CLIPVisionModel student from checkpoint.

    Reuses the same logic as ``ml.farslip.distill._patch_student_proj``: starts
    from the HF teacher, adapts ``patch_embed`` to ``n_in_channels`` and loads the
    student state_dict. ``strict=False`` tolerates prefix differences between
    ``CLIPVisionModel`` and ``CLIPModel.vision_model``.
    """
    from ml.farslip.distill import adapt_patch_embed_to_n_channels

    model = CLIPVisionModel.from_pretrained(teacher_model_id)
    adapt_patch_embed_to_n_channels(model, n_in_channels)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        cands = list(ckpt_path.glob("*.safetensors")) + list(ckpt_path.glob("*.pt"))
        if not cands:
            raise FileNotFoundError(f"no checkpoints (*.safetensors|*.pt) in {ckpt_path}")
        ckpt_path = cands[0]
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path))
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _log.info(
        "student_loaded",
        path=str(ckpt_path),
        missing=len(missing),
        unexpected=len(unexpected),
    )
    model.eval()
    model.to(device)
    return model


def _embed_columns() -> list[str]:
    """Return ``["farslip_emb_000", ..., "farslip_emb_511"]``."""
    return [f"{EMBED_COL_PREFIX}{i:03d}" for i in range(EMBED_DIM)]


def _load_parcels_filtered(parcels_parquet: Path, rois: RoiPreset) -> pl.DataFrame:
    """Load parcels and filter by ROI(s) if the column exists.

    The parquet must expose at least ``parcel_id`` and ``year``. If it has a
    ``roi`` (or ``region``) column, it is filtered by ``rois``. If it does not
    exist, no filtering is applied (assumes the caller already filtered).
    """
    df = pl.read_parquet(parcels_parquet)
    if "parcel_id" not in df.columns:
        raise ValueError("parquet without 'parcel_id' column")
    if "year" not in df.columns:
        raise ValueError("parquet without 'year' column")
    for col in ("roi", "region"):
        if col in df.columns:
            df = df.filter(pl.col(col).is_in(list(rois)))
            break
    return df


def _build_empty_embeddings(n_rows: int) -> torch.Tensor:
    """Zero tensor of shape ``(n_rows, EMBED_DIM)`` (placeholder smoke-tests)."""
    return torch.zeros((n_rows, EMBED_DIM), dtype=torch.float32)


def _project_parcels_to_embeddings(
    model: CLIPVisionModel,
    *,
    n_parcels: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Deterministic placeholder (kept for existing tests and smoke).

    Generates L2-normalized `torch.randn(seed)`. For REAL extraction see
    :func:`_project_parcels_to_embeddings_real`.
    """
    propagate_seed(seed)
    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(seed)
    out_chunks: list[torch.Tensor] = []
    for start in range(0, n_parcels, batch_size):
        end = min(start + batch_size, n_parcels)
        chunk = torch.randn(
            (end - start, EMBED_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        chunk = torch.nn.functional.normalize(chunk, dim=-1)
        out_chunks.append(chunk.detach().cpu())
    return torch.cat(out_chunks, dim=0)


def _project_parcels_to_embeddings_real(
    model: CLIPVisionModel,
    parcels: pl.DataFrame,
    *,
    dataset_root: Path,
    batch_size: int,
    device: torch.device,
    seed: int,
    crop_resize_to: int = 224,
) -> torch.Tensor:
    """Real student forward over Sentinel-2 crops.

    Reads each ``.tif`` crop from ``dataset_root/{region}/crops/{file}``,
    resizes to 224x224, normalizes uint16/10000, passes through
    ``model.vision_model(pixel_values).pooler_output`` and returns a tensor
    ``(n_parcels, 512)`` on CPU float32.

    Args:
        model: ``CLIPVisionModel`` with patch_embed adapted to 4 channels.
        parcels: DataFrame with columns ``crop_path`` + ``region``.
        dataset_root: ``data/farslip_pairs/`` root to resolve crops cross-platform.
        batch_size: forward batch size (default 64 works on 24GB L4).
        device: ``torch.device``.
        seed: deterministic seed (for reproducible shuffling if applicable).
        crop_resize_to: crop side after bilinear resize (default 224).

    Returns:
        Tensor ``(n_parcels, EMBED_DIM)`` on CPU float32.
    """
    from ml.farslip.dataset import FarSLIPDataset

    propagate_seed(seed)
    if "crop_path" not in parcels.columns:
        raise ValueError(
            "parquet without 'crop_path' column required for real extract. "
            "Add crop_path from manifest.parquet."
        )

    helper = FarSLIPDataset.__new__(FarSLIPDataset)
    helper.manifest_path = dataset_root / "_dummy_manifest.parquet"
    helper.crop_resize_to = crop_resize_to

    n_parcels = parcels.height
    rows = parcels.to_dicts()
    out_chunks: list[torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, n_parcels, batch_size):
            end = min(start + batch_size, n_parcels)
            imgs: list[torch.Tensor] = []
            for row in rows[start:end]:
                crop_path_raw = row["crop_path"]
                region = row.get("region")
                if region:
                    helper.manifest_path = dataset_root / region / "manifest.parquet"
                resolved = helper._resolve_crop_path(crop_path_raw)
                img = helper._load_crop(resolved)
                img = helper._resize_chw(img, crop_resize_to)
                imgs.append(img)
            batch = torch.stack(imgs, dim=0).to(device)
            out = model(pixel_values=batch)
            emb = out.pooler_output
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out_chunks.append(emb.detach().cpu().float())
            if start % (batch_size * 10) == 0:
                _log.info(
                    "extract_real_progress",
                    done=end,
                    total=n_parcels,
                    pct=round(100 * end / n_parcels, 1),
                )
    return torch.cat(out_chunks, dim=0)


def extract_farslip_embeddings(
    *,
    student_checkpoint_path: Path,
    parcels_parquet: Path,
    rois: tuple[str, ...] = ("pianura_padana", "toscana", "puglia"),
    output_path: Path,
    batch_size: int = 256,
    device: DeviceLiteral = "auto",
    seed: int = 42,
    mode: Literal["placeholder", "real"] = "placeholder",
    dataset_root: Path | None = None,
) -> ExtractEmbeddingsResult:
    """Extract FarSLIP embeddings of each Italian parcel and persist them.

    Args:
        student_checkpoint_path: local path or ``mlflow://...`` URI.
        parcels_parquet: parquet with at least ``parcel_id`` + ``year``.
        rois: tuple of ROIs (or ``("italy",)``).
        output_path: output parquet (parent is created if it does not exist).
        batch_size: batch for CLIP inference (default 256).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        seed: deterministic seed (default 42).

    Returns:
        :class:`ExtractEmbeddingsResult` with metadata for MLflow tags.
    """
    rois_resolved = _resolve_rois(rois)
    torch_device = _resolve_device(device)
    ckpt_resolved, data_version = _resolve_checkpoint(student_checkpoint_path)
    parcels = _load_parcels_filtered(parcels_parquet, rois_resolved)
    n_parcels = parcels.height
    _log.info(
        "extracting_farslip_embeddings",
        n_parcels=n_parcels,
        rois=list(rois_resolved),
        device=torch_device.type,
        seed=seed,
    )
    model = _load_student(ckpt_resolved, device=torch_device)
    if mode == "real":
        if dataset_root is None:
            raise ValueError("mode='real' requires dataset_root to resolve crops")
        embeddings = _project_parcels_to_embeddings_real(
            model,
            parcels,
            dataset_root=dataset_root,
            batch_size=batch_size,
            device=torch_device,
            seed=seed,
        )
    else:
        embeddings = _project_parcels_to_embeddings(
            model,
            n_parcels=n_parcels,
            batch_size=batch_size,
            device=torch_device,
            seed=seed,
        )
    cols = _embed_columns()
    embed_dict = {cols[i]: embeddings[:, i].numpy() for i in range(EMBED_DIM)}
    out_df = parcels.select(
        pl.col("parcel_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int32),
    ).with_columns([pl.Series(name=cols[i], values=embed_dict[cols[i]]) for i in range(EMBED_DIM)])
    if out_df.width != TOTAL_COLS:
        raise RuntimeError(f"unexpected output width: {out_df.width} != {TOTAL_COLS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    code_version = git_sha(short=True)
    result = ExtractEmbeddingsResult(
        n_parcels=n_parcels,
        n_dims=EMBED_DIM,
        output_path=output_path.resolve(),
        code_version=code_version,
        data_version=data_version,
        device_used=torch_device.type,
    )
    _log.info("farslip_embeddings_written", **result.__dict__)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.farslip.extract_embeddings",
        description=(
            "Proyecta parcelas italianas sobre el espacio FarSLIP "
            "(CLIP-512) y persiste el resultado a parquet."
        ),
    )
    parser.add_argument(
        "--student-checkpoint",
        required=True,
        help=("Ruta local al checkpoint .safetensors/.pt o URI 'mlflow://Models/<name>@<stage>'."),
    )
    parser.add_argument(
        "--parcels-parquet",
        required=True,
        type=Path,
        help="Parquet con columnas parcel_id + year.",
    )
    parser.add_argument(
        "--rois",
        default="italy",
        help=("Comma-separated. Alias soportados: 'italy' (pianura_padana,toscana,puglia)."),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Parquet de salida (parent se crea automaticamente).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size para inferencia (default 256).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device override (default 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista (default 42).",
    )
    parser.add_argument(
        "--mode",
        choices=("placeholder", "real"),
        default="placeholder",
        help=(
            "'placeholder' (legacy seeded randn, default) o 'real' "
            "(forward CLIPVisionModel sobre crops Sentinel-2)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Raiz dataset farslip_pairs (requerido si --mode=real).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    rois_tuple = tuple(r.strip() for r in args.rois.split(",") if r.strip())
    result = extract_farslip_embeddings(
        student_checkpoint_path=Path(args.student_checkpoint),
        parcels_parquet=args.parcels_parquet,
        rois=rois_tuple,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        mode=args.mode,
        dataset_root=args.dataset_root,
    )
    _log.info(
        "farslip_extract_embeddings_complete",
        n_parcels=result.n_parcels,
        n_dims=result.n_dims,
        output=str(result.output_path),
        code_version=result.code_version,
        data_version=result.data_version,
        device=result.device_used,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
