"""Extract official FarSLIP embeddings per PASTIS-R parcel.

For each PASTIS-R patch, computes the temporal median of the Sentinel-2 series,
crops every valid parcel (instance mask), converts it to an RGB 224 image and
runs the official FarSLIP visual tower to obtain a 512-dim embedding. The result
feeds the FarSLIP-vs-AlphaEarth embedding comparison in
``notebooks/baseline/04_farslip_eval_pastis.ipynb`` (replacing the former
placeholder embeddings).

Output parquet columns: ``parcel_id`` (``{patch_id}_{instance_id}``),
``patch_id``, ``class_id`` (majority PASTIS class of the parcel), ``emb_000`` ..
``emb_511``.

Usage::

    poetry run python scripts/extract_farslip_official_pastis.py \\
        --max-patches 0 --batch-size 256 --min-parcel-px 32
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import structlog
import typer

from ml.extractors.farslip_official_extractor import EMBED_DIM, FarSLIPOfficialExtractor

_log = structlog.get_logger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_PASTIS = _REPO / "data" / "PASTIS-R"
_DEFAULT_CKPT = _REPO / "data" / "farslip" / "checkpoints" / "FarSLIP2_ViT-B-16.pt"
_DEFAULT_OUT = _REPO / "data" / "farslip" / "embeddings_pastis.parquet"

_VOID_ID = 19
_BG_ID = 0


def _patch_ids() -> list[str]:
    """Return sorted PASTIS-R patch ids that have S2, target and instance masks."""
    s2_dir = _PASTIS / "DATA_S2"
    ids = []
    for f in sorted(s2_dir.glob("S2_*.npy")):
        pid = f.stem.replace("S2_", "")
        if (_PASTIS / "ANNOTATIONS" / f"TARGET_{pid}.npy").is_file() and (
            _PASTIS / "INSTANCE_ANNOTATIONS" / f"INSTANCES_{pid}.npy"
        ).is_file():
            ids.append(pid)
    return ids


def _iter_parcel_crops(pid: str, min_px: int):
    """Yield ``(parcel_id, class_id, rgb_pil)`` for each valid parcel in a patch.

    The Sentinel-2 series is collapsed by temporal median; each parcel is cropped
    to its instance-mask bounding box with off-parcel pixels zeroed out.
    """
    s2 = np.load(_PASTIS / "DATA_S2" / f"S2_{pid}.npy").astype(np.float32)
    median = np.median(s2, axis=0)  # (10, H, W)
    target = np.load(_PASTIS / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
    inst = np.load(_PASTIS / "INSTANCE_ANNOTATIONS" / f"INSTANCES_{pid}.npy")

    for iid in np.unique(inst):
        if iid == 0:
            continue
        mask = inst == iid
        if int(mask.sum()) < min_px:
            continue
        class_pixels = target[mask]
        class_pixels = class_pixels[(class_pixels != _VOID_ID) & (class_pixels != _BG_ID)]
        if class_pixels.size == 0:
            continue
        class_id = int(np.bincount(class_pixels).argmax())
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = median[:, y0:y1, x0:x1].copy()
        sub_mask = mask[y0:y1, x0:x1]
        crop[:, ~sub_mask] = 0.0
        rgb = FarSLIPOfficialExtractor.s2_to_rgb_pil(crop)
        yield f"{pid}_{int(iid)}", class_id, rgb


def main(
    checkpoint: Annotated[
        Path, typer.Option(help="Ruta al checkpoint FarSLIP .pt")
    ] = _DEFAULT_CKPT,
    out_path: Annotated[Path, typer.Option(help="Parquet de salida")] = _DEFAULT_OUT,
    max_patches: Annotated[int, typer.Option(help="0 = todos los patches")] = 0,
    batch_size: Annotated[int, typer.Option(help="Batch de inferencia")] = 256,
    min_parcel_px: Annotated[int, typer.Option(help="Pixeles minimos por parcela")] = 32,
    flush_every: Annotated[int, typer.Option(help="Patches por checkpoint a disco")] = 200,
) -> None:
    """Extract FarSLIP per-parcel embeddings over PASTIS-R and write the parquet."""
    extractor = FarSLIPOfficialExtractor(checkpoint, device="auto")
    typer.echo(f"Dispositivo: {extractor.device}")

    pids = _patch_ids()
    if max_patches > 0:
        pids = pids[:max_patches]
    typer.echo(f"Patches a procesar: {len(pids)}")

    emb_cols = [f"emb_{i:03d}" for i in range(EMBED_DIM)]
    written = 0
    parts: list[Path] = []
    tmp_dir = out_path.parent / "_farslip_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    buf_ids: list[str] = []
    buf_cls: list[int] = []
    buf_img: list = []

    def _flush(tag: str) -> None:
        nonlocal written, buf_ids, buf_cls, buf_img
        if not buf_img:
            return
        emb = extractor.encode_images(buf_img, batch_size=batch_size)
        meta = pl.DataFrame(
            {
                "parcel_id": buf_ids,
                "patch_id": [p.rsplit("_", 1)[0] for p in buf_ids],
                "class_id": buf_cls,
            }
        )
        df = meta.hstack(pl.DataFrame(emb, schema=emb_cols))
        part = tmp_dir / f"part_{tag}.parquet"
        df.write_parquet(part)
        parts.append(part)
        written += df.height
        _log.info("flush", part=str(part.name), rows=df.height, total=written)
        buf_ids, buf_cls, buf_img = [], [], []

    for i, pid in enumerate(pids):
        for parcel_id, class_id, rgb in _iter_parcel_crops(pid, min_parcel_px):
            buf_ids.append(parcel_id)
            buf_cls.append(class_id)
            buf_img.append(rgb)
        if (i + 1) % flush_every == 0:
            _flush(f"{i + 1:05d}")

    _flush("final")

    full = (
        pl.concat([pl.read_parquet(p) for p in parts], how="vertical") if parts else pl.DataFrame()
    )
    full.write_parquet(out_path)
    for p in parts:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()
    typer.echo(f"Listo: {full.height} parcelas -> {out_path}")
    n_classes = full["class_id"].n_unique() if full.height else 0
    typer.echo(f"Clases: {n_classes} | dim embedding: {EMBED_DIM}")


if __name__ == "__main__":
    typer.run(main)
