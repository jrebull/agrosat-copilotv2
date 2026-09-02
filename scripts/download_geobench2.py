"""US-069 (GEO-Bench-2): download the agricultural subset and build the manifest.

GEO-Bench-2 is the AI Alliance / TUM / IBM / ServiceNow Earth-observation
benchmark (successor of GEO-Bench 2023), distributed on HuggingFace in the
legacy TACO v1 (``.tortilla``) format. This script materialises the
agricultural subset that ``ml.eval.paper_bench.load_geobench2`` consumes: it
reads the REAL tiles + segmentation masks by streaming (GDAL ``/vsisubfile`` +
``/vsicurl`` HTTP range requests, no full multi-GB download), derives a
per-tile **dominant-crop classification** label from each sample's semantic
mask (the modal non-background class), writes a small RGB thumbnail per tile and
emits ``data/geobench2/manifest.json`` with the shape the loader expects:

``{"tasks": [{id, name, modality, label_space, split, items: [{item_id,
image_path, gold_label}]}]}``.

Three agricultural tasks (>= 3 per the US-069 AC), all REAL GEO-Bench-2 data:

- ``m-pastis``     -- PASTIS crop-type (19 crops + background), mask ``semantic``.
- ``m-flair2``     -- FLAIR #2 land cover (12 classes + other), mask ``mask``.
- ``m-fotw``       -- Fields of the World boundary (3 classes), mask
  ``semantic_3class_mask``.

Class names are taken VERBATIM from the canonical sources (PASTIS from
torchgeo; FLAIR #2 from the IGN FLAIR paper; FoTW from the dataset's
3-class semantic head) -- no class name is invented. Tiles whose mask is pure
background are skipped (no defensible dominant crop label).

Zero-synthetic rule (Arthur): every label comes from a real mask. If the remote
read fails the script raises -- it never fabricates an item.

Project conventions: English identifiers/docstrings (Google style); Spanish CLI
prose; ``structlog`` (no ``print`` in logic); full type hints; no emojis.

Usage:
    poetry run python scripts/download_geobench2.py --root data/geobench2 \\
        --max-per-task 60 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

#: Base HF resolve URL of each agricultural dataset's tortilla file(s).
_HF_BASE: str = "https://huggingface.co/datasets/aialliance/{name}/resolve/main/{file}"

#: PASTIS crop-type label space (index 0..19), VERBATIM from torchgeo
#: (microsoft/torchgeo ``datasets/pastis.py``). 18 crops + background + void.
_PASTIS_CLASSES: tuple[str, ...] = (
    "background",
    "meadow",
    "soft_winter_wheat",
    "corn",
    "winter_barley",
    "winter_rapeseed",
    "spring_barley",
    "sunflower",
    "grapevine",
    "beet",
    "winter_triticale",
    "winter_durum_wheat",
    "fruits_vegetables_flowers",
    "potatoes",
    "leguminous_fodder",
    "soybeans",
    "orchard",
    "mixed_cereal",
    "sorghum",
    "void_label",
)

#: FLAIR #2 land-cover label space (index 0..12), from the IGN FLAIR paper
#: (arXiv:2310.13336). Index 0 is the no-data/other sink; 1..12 are the classes.
_FLAIR2_CLASSES: tuple[str, ...] = (
    "other",
    "building",
    "pervious_surface",
    "impervious_surface",
    "bare_soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous_vegetation",
    "agricultural_land",
    "plowed_land",
)

#: Fields of the World 3-class semantic head (index 0..2).
_FOTW_CLASSES: tuple[str, ...] = ("background", "field", "field_boundary")


@dataclass(frozen=True)
class GeoBenchTaskSpec:
    """Download spec of one agricultural GEO-Bench-2 task.

    Attributes:
        task_id: Stable manifest task id (e.g. ``m-pastis``).
        name: Human-readable task name.
        hf_name: HuggingFace dataset name under ``aialliance/``.
        tortilla_files: One or more tortilla part filenames to load + concat.
        mask_subitem: The ``tortilla:id`` of the segmentation mask subitem.
        image_subitem: The ``tortilla:id`` of the RGB-capable image subitem used
            for the thumbnail.
        rgb_bands: 1-based band indices to read for the thumbnail RGB composite.
        classes: The ordered class names indexed by mask value.
        ignore_values: Mask values excluded when picking the dominant crop class
            (background / void / no-data).
    """

    task_id: str
    name: str
    hf_name: str
    tortilla_files: tuple[str, ...]
    mask_subitem: str
    image_subitem: str
    rgb_bands: tuple[int, int, int]
    classes: tuple[str, ...]
    ignore_values: frozenset[int]


#: The three agricultural tasks materialised by this script.
TASK_SPECS: tuple[GeoBenchTaskSpec, ...] = (
    GeoBenchTaskSpec(
        task_id="m-pastis",
        name="PASTIS crop type (GEO-Bench-2)",
        hf_name="pastis",
        tortilla_files=(
            "geobench_pastis.0000.part.tortilla",
            "geobench_pastis.0001.part.tortilla",
            "geobench_pastis.0002.part.tortilla",
        ),
        mask_subitem="semantic",
        image_subitem="s2",
        rgb_bands=(3, 2, 1),
        classes=_PASTIS_CLASSES,
        ignore_values=frozenset({0, 19}),
    ),
    GeoBenchTaskSpec(
        task_id="m-flair2",
        name="FLAIR #2 land cover (GEO-Bench-2)",
        hf_name="flair2",
        tortilla_files=("geobench_flair2.tortilla",),
        mask_subitem="mask",
        image_subitem="aerial",
        rgb_bands=(1, 2, 3),
        classes=_FLAIR2_CLASSES,
        ignore_values=frozenset({0}),
    ),
    GeoBenchTaskSpec(
        task_id="m-fotw",
        name="Fields of the World boundary (GEO-Bench-2)",
        hf_name="fotw",
        tortilla_files=("geobench_fotw.tortilla",),
        mask_subitem="semantic_3class_mask",
        image_subitem="win_a",
        rgb_bands=(3, 2, 1),
        classes=_FOTW_CLASSES,
        ignore_values=frozenset({0}),
    ),
)


def _load_concat(spec: GeoBenchTaskSpec) -> object:
    """Load (and concat for multi-part) a task's tortilla index by streaming.

    Args:
        spec: The task spec.

    Returns:
        A ``tacoreader`` ``TortillaDataFrame`` over the task's samples.
    """
    import tacoreader

    frames = []
    for fname in spec.tortilla_files:
        url = _HF_BASE.format(name=spec.hf_name, file=fname)
        logger.info("geobench2_load_part", task=spec.task_id, url=url)
        frames.append(tacoreader.load(url))
    if len(frames) == 1:
        return frames[0]
    import pandas as pd

    concatenated = pd.concat(frames, ignore_index=True)
    # Re-wrap into a TortillaDataFrame so .read(i) keeps working.
    return type(frames[0])(concatenated)


def _dominant_class(mask: np.ndarray, ignore: frozenset[int], n_classes: int) -> int | None:
    """Return the modal in-range, non-ignored class value of a mask.

    Args:
        mask: The 2D integer segmentation mask.
        ignore: Mask values to exclude (background / void / no-data).
        n_classes: Size of the label space (values >= this are out-of-range).

    Returns:
        The dominant class integer, or ``None`` when the tile carries no
        defensible foreground crop/land-cover class.
    """
    flat = mask.reshape(-1)
    counts = Counter(
        int(v) for v in flat.tolist() if int(v) not in ignore and 0 <= int(v) < n_classes
    )
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _save_thumbnail(image_path: str, rgb_bands: Sequence[int], out_png: Path) -> bool:
    """Read an RGB composite from a streamed raster and save a PNG thumbnail.

    Args:
        image_path: GDAL-readable path (``/vsisubfile/.../vsicurl/...``).
        rgb_bands: 1-based band indices for the R, G, B composite.
        out_png: Destination PNG path.

    Returns:
        ``True`` when the thumbnail was written, ``False`` when the raster could
        not yield an RGB composite (e.g. a temporal cube with too few bands).
    """
    from PIL import Image

    with rasterio.open(image_path) as src:
        if src.count < max(rgb_bands):
            return False
        bands = [src.read(b).astype("float32") for b in rgb_bands]
    stack = np.stack(bands, axis=-1)
    lo, hi = np.nanpercentile(stack, 2), np.nanpercentile(stack, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
    if hi <= lo:
        return False
    scaled = np.clip((stack - lo) / (hi - lo), 0.0, 1.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype("uint8")).resize((256, 256)).save(out_png)
    return True


def build_task(
    spec: GeoBenchTaskSpec, root: Path, *, split: str, max_per_task: int
) -> dict[str, object]:
    """Build one task's manifest entry from REAL streamed GEO-Bench-2 samples.

    Args:
        spec: The task spec.
        root: The output ``data/geobench2`` root.
        split: The data split to keep (``test`` by default).
        max_per_task: Maximum scored tiles to materialise for this task.

    Returns:
        The task dict for the manifest (``id, name, modality, label_space,
        split, items``).
    """
    ds = _load_concat(spec)
    split_col = ds["tortilla:data_split"]
    sample_idx = [i for i in range(len(ds)) if str(split_col.iloc[i]) == split]
    logger.info(
        "geobench2_task_samples",
        task=spec.task_id,
        split=split,
        n_in_split=len(sample_idx),
        target=max_per_task,
    )
    items: list[dict[str, str]] = []
    n_skipped_bg = 0
    n_skipped_thumb = 0
    for i in sample_idx:
        if len(items) >= max_per_task:
            break
        sample = ds.read(i)
        sub_ids = list(sample["tortilla:id"])
        if spec.mask_subitem not in sub_ids or spec.image_subitem not in sub_ids:
            continue
        mask_path = sample.read(sub_ids.index(spec.mask_subitem))
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)
        dom = _dominant_class(mask, spec.ignore_values, len(spec.classes))
        if dom is None:
            n_skipped_bg += 1
            continue
        item_id = f"{spec.task_id}-{len(items):04d}"
        rel_png = f"tiles/{spec.task_id}/{item_id}.png"
        image_path = sample.read(sub_ids.index(spec.image_subitem))
        if not _save_thumbnail(image_path, spec.rgb_bands, root / rel_png):
            n_skipped_thumb += 1
            continue
        items.append({"item_id": item_id, "image_path": rel_png, "gold_label": spec.classes[dom]})
    logger.info(
        "geobench2_task_built",
        task=spec.task_id,
        n_items=len(items),
        n_skipped_background=n_skipped_bg,
        n_skipped_no_rgb=n_skipped_thumb,
    )
    return {
        "id": spec.task_id,
        "name": spec.name,
        "modality": "classification",
        "label_space": list(spec.classes),
        "split": split,
        "items": items,
    }


def build_manifest(
    root: Path, *, split: str, max_per_task: int, tasks: Sequence[str] | None
) -> dict[str, object]:
    """Build the full agricultural GEO-Bench-2 manifest.

    Args:
        root: Output ``data/geobench2`` root.
        split: Data split to keep.
        max_per_task: Max tiles per task.
        tasks: Optional allow-list of task ids (``None`` = all three).

    Returns:
        The manifest dict written to ``manifest.json``.
    """
    allow = frozenset(tasks) if tasks else None
    selected = [s for s in TASK_SPECS if allow is None or s.task_id in allow]
    built = [build_task(s, root, split=split, max_per_task=max_per_task) for s in selected]
    return {
        "dataset": "geobench2-agricultural-subset",
        "source": "https://huggingface.co/collections/aialliance/geo-bench-2",
        "license_note": (
            "GEO-Bench-2 redistributes original datasets under their own licenses "
            "(PASTIS etalab-2.0; FLAIR #2 etalab-2.0; FoTW CC-BY-4.0). AlphaEarth "
            "SATELLITE_EMBEDDING/V1/ANNUAL v1.1 CC-BY-4.0."
        ),
        "modality": "classification_from_dominant_segmentation_class",
        "tasks": built,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code (``0`` on success, ``1`` when no item was produced).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Descarga el subset agricola REAL de GEO-Bench-2 (PASTIS / FLAIR2 / FoTW) "
            "por streaming y construye data/geobench2/manifest.json para paper_bench."
        )
    )
    parser.add_argument("--root", default="data/geobench2", help="Directorio de salida.")
    parser.add_argument("--split", default="test", help="Split a materializar (test).")
    parser.add_argument("--max-per-task", type=int, default=60, help="Maximo de tiles por task.")
    parser.add_argument(
        "--tasks", nargs="*", default=None, help="Allow-list de task ids (default: las 3)."
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        root, split=args.split, max_per_task=args.max_per_task, tasks=args.tasks
    )
    n_total = sum(len(t["items"]) for t in manifest["tasks"])  # type: ignore[arg-type]
    if n_total == 0:
        logger.error("geobench2_no_items", reason="no real masks produced a label")
        return 1
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "geobench2_manifest_written",
        path=str(out),
        n_tasks=len(manifest["tasks"]),  # type: ignore[arg-type]
        n_items=n_total,
    )
    print(
        f"GEO-Bench-2 subset escrito en {out}: {len(manifest['tasks'])} tasks, {n_total} tiles."  # type: ignore[arg-type]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
