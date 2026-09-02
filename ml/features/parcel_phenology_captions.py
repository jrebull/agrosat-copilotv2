"""Per-parcel phenology captions for parcel-level FarSLIP (US-036-b).

The patch-level Gemma captions are ~60% identical because every PASTIS patch is
the same kind of fragmented agricultural mosaic. This module replaces them with
**per-parcel phenology descriptions** (Wen et al. 2025: "Phenology Description is
All You Need"): each parcel's own temporal NDVI curve is turned into a textual
description of its growth dynamics, which is far more discriminative than the
visual appearance of a single composite.

Pipeline:

    compute_parcel_ndvi_curves(root, folds)
        -> {parcel_id: (curve, doy, class_id)}  # one mean NDVI curve PER PARCEL
    generate_parcel_phenology_captions(curves, class_names, client, out)
        -> parquet {parcel_id, patch_id, class_id, description, gen_seconds}

It reuses ``generate_phenology_description`` (Wen 3-block prompt, SHA256 cache,
``temperature=0``) with a **local Gemma client** (Ollama) injected via
``set_llm_client`` -- cost $0, consistent with the project SCOPE. Flush is
incremental (resilient to SSH/tunnel cuts on the H100, like ``caption_cache``).

Conventions: torch/numpy at the data boundary, Polars for the parquet, structlog,
type hints, English docstrings, Spanish prose, no emojis. Real PASTIS-R only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.farslip.region_category_dataset import _DEFAULT_MIN_AREA_PX, extract_regions
from ml.features.phenology_class_prototypes import (
    _BAND_B4,
    _BAND_B8,
    _DEFAULT_PASTIS_ROOT,
    _N_TIME_BINS,
    _patch_dates_doy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ml.features.phenology_description import LlmClient

logger = structlog.get_logger(__name__)

#: Output parquet schema (one row per parcel).
PARCEL_CAPTIONS_SCHEMA = {
    "parcel_id": pl.Utf8,
    "patch_id": pl.Utf8,
    "class_id": pl.Int64,
    "description": pl.Utf8,
    "gen_seconds": pl.Float64,
}


def compute_parcel_ndvi_curves(
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    folds: tuple[int, ...] = (1, 2, 3, 4),
    *,
    n_time_bins: int = _N_TIME_BINS,
    active_class_ids: tuple[int, ...] = tuple(range(1, 19)),
    min_area_px: int = _DEFAULT_MIN_AREA_PX,
    max_patches: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Compute the mean NDVI curve PER PARCEL over a regular DOY grid.

    Per-instance counterpart of
    :func:`ml.features.phenology_class_prototypes.compute_class_mean_ndvi_curves`:
    instead of grouping NDVI by semantic class, it groups by parcel instance, so
    every parcel gets its OWN temporal curve (the source of the per-parcel caption
    diversity). Only parcels whose majority class is in ``active_class_ids`` and
    whose area >= ``min_area_px`` are kept (same filter as ``extract_regions``).

    Args:
        pastis_root: real PASTIS-R root.
        folds: official folds to scan (default train+val 1..4; fold 5 reserved).
        n_time_bins: regular DOY bins (1..365).
        active_class_ids: crop classes kept.
        min_area_px: minimum parcel area.
        max_patches: cap on patches scanned (smoke/tests).

    Returns:
        ``{parcel_id="{pid}_{iid}": (curve (n_time_bins,), doy_grid, class_id)}``
        with NaN in bins without observation.
    """
    from ml.ingest.pastis_dataset import pastis_fold_split
    from ml.ingest.pastis_loader import load_pastis_patch

    split = pastis_fold_split(pastis_root, train_folds=tuple(folds), val_folds=(), test_folds=())
    pids = sorted(split["train"], key=lambda p: int(p))
    if max_patches is not None:
        pids = pids[: int(max_patches)]

    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    doy_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # bin centers (n_time_bins,)
    curves: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    dates_by_patch = _patch_dates_doy(pastis_root / "metadata.geojson")

    for pid in pids:
        doy = dates_by_patch.get(int(pid))
        if doy is None:
            continue
        patch = load_pastis_patch(pid, root=pastis_root, load_annotations=True)
        semantic = patch.get("semantic")
        instance = patch.get("instance")
        if semantic is None or instance is None:
            continue
        instance_arr = np.asarray(instance)
        regions = extract_regions(
            instance_arr,
            np.asarray(semantic),
            active_class_ids=active_class_ids,
            min_area_px=min_area_px,
        )
        if not regions:
            continue
        s2 = np.asarray(patch["s2"]).astype(np.float32) / 10000.0  # (T,10,H,W)
        b4 = s2[:, _BAND_B4]
        b8 = s2[:, _BAND_B8]
        denom = b8 + b4
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom > 1e-6, (b8 - b4) / denom, np.nan)  # (T,H,W)
        ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
        bin_idx = np.clip(np.digitize(doy, bin_edges) - 1, 0, n_time_bins - 1)

        for inst_id, cat_id in regions:
            mask = instance_arr == inst_id  # (H,W)
            ndvi_parcel = ndvi[:, mask]  # (T, n_pix)
            with np.errstate(invalid="ignore"):
                # A timestep fully clouded over the parcel yields an all-NaN
                # slice; nanmean returns NaN (handled by the bin accumulation).
                per_t = np.nanmean(ndvi_parcel, axis=1)  # (T,)
            valid = np.isfinite(per_t)
            sums = np.zeros(n_time_bins, dtype=np.float64)
            counts = np.zeros(n_time_bins, dtype=np.int64)
            np.add.at(sums, bin_idx[valid], per_t[valid])
            np.add.at(counts, bin_idx[valid], 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                curve = np.where(counts > 0, sums / counts, np.nan)
            curves[f"{pid}_{int(inst_id)}"] = (curve, doy_grid, int(cat_id))

    logger.info(
        "parcel_ndvi_curves_computed",
        n_patches=len(pids),
        n_parcels=len(curves),
        n_time_bins=n_time_bins,
    )
    return curves


def make_ollama_text_client(
    base_url: str = "http://127.0.0.1:11434",
    *,
    timeout_s: float = 120.0,
) -> LlmClient:
    """Build a text LLM client (Ollama/Gemma) matching the phenology-desc signature.

    Returns a callable ``(prompt, *, model, temperature) -> str`` so it can be
    injected via :func:`ml.features.phenology_description.set_llm_client`. Uses
    Ollama's ``/api/chat`` with ``think=false`` (Gemma hangs otherwise, validated
    in caption_generator). Cost $0 (local on the H100).

    Args:
        base_url: Ollama base url.
        timeout_s: per-request timeout.

    Returns:
        The text client callable.
    """
    import requests

    def _client(prompt: str, *, model: str, temperature: float) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": float(temperature), "num_predict": 400},
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("message", {}).get("content", "")).strip()

    return _client


def generate_parcel_phenology_captions(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, int]],
    class_names: Mapping[int, str],
    *,
    output_path: Path,
    model: str = "gemma4:31b-it-q8_0",
    cache_dir: Path | None = None,
    flush_every: int = 50,
    resume: bool = True,
    max_workers: int = 1,
) -> Path:
    """Generate per-parcel phenology captions and write them to a parquet.

    For each parcel calls
    :func:`ml.features.phenology_description.generate_phenology_description` with
    the parcel's NDVI curve, DOY grid and ``crop_type_hint = class_names[class_id]``
    (so the text reflects the parcel's own dynamics -> diverse captions). The text
    LLM client must already be injected via ``set_llm_client``. Flush is
    incremental + resume; with ``max_workers > 1`` the (I/O-bound) LLM calls run
    concurrently via threads -- essential for a cloud LLM over ~70k parcels (local
    Gemma stays at ``max_workers=1``).

    Args:
        curves: ``{parcel_id: (curve, doy, class_id)}`` from
            :func:`compute_parcel_ndvi_curves`.
        class_names: ``{class_id: crop name}`` for the crop hint.
        output_path: output parquet.
        model: LLM model id (Gemma local or Gemini).
        cache_dir: SHA256 cache dir for the descriptions (re-runs are free).
        flush_every: parcels per atomic flush.
        resume: skip parcels already in the output parquet.
        max_workers: concurrent LLM calls (1 = sequential for local Gemma;
            16-32 for a cloud LLM like Gemini Flash).

    Returns:
        The output parquet path.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ml.features.phenology_description import generate_phenology_description

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    existing_rows: list[dict[str, object]] = []
    if resume and output_path.is_file():
        prev = pl.read_parquet(output_path)
        existing_rows = prev.to_dicts()
        done = {str(r["parcel_id"]) for r in existing_rows}
        logger.info("parcel_captions_resume", n_done=len(done))

    pending = [(pid, c, d, cid) for pid, (c, d, cid) in curves.items() if pid not in done]

    def _one(item: tuple[str, np.ndarray, np.ndarray, int]) -> dict[str, object]:
        parcel_id, curve, doy, class_id = item
        t0 = time.monotonic()
        description = generate_phenology_description(
            curve,
            doy=doy,
            parcel_id=parcel_id,
            crop_type_hint=class_names.get(int(class_id)),
            model=model,
            temperature=0.0,
            cache_dir=cache_dir,
        )
        return {
            "parcel_id": parcel_id,
            "patch_id": parcel_id.rsplit("_", 1)[0],
            "class_id": int(class_id),
            "description": description,
            "gen_seconds": round(time.monotonic() - t0, 3),
        }

    new_rows: list[dict[str, object]] = []
    if max_workers <= 1:
        results_iter: Iterator[dict[str, object]] = (_one(it) for it in pending)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        results_iter = executor.map(_one, pending)
    try:
        for row in results_iter:
            new_rows.append(row)
            if len(new_rows) >= flush_every:
                existing_rows = _flush_parcel_captions(existing_rows, new_rows, output_path)
                new_rows = []
    finally:
        if max_workers > 1:
            executor.shutdown(wait=True)

    if new_rows:
        _flush_parcel_captions(existing_rows, new_rows, output_path)

    logger.info("parcel_captions_done", path=str(output_path), n_total=len(curves))
    return output_path


def _flush_parcel_captions(
    existing: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    out_path: Path,
) -> list[dict[str, object]]:
    """Atomically append new rows to the parquet (.tmp -> replace)."""
    merged = existing + new_rows
    df = pl.DataFrame(merged, schema=PARCEL_CAPTIONS_SCHEMA)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(out_path)
    logger.info("parcel_captions_flush", n_total=len(merged), path=str(out_path))
    return merged


def load_parcel_captions(path: Path) -> dict[str, str]:
    """Load ``{parcel_id: description}`` from the per-parcel captions parquet."""
    df = pl.read_parquet(path)
    return {str(r["parcel_id"]): str(r["description"]) for r in df.iter_rows(named=True)}


__all__ = [
    "PARCEL_CAPTIONS_SCHEMA",
    "compute_parcel_ndvi_curves",
    "generate_parcel_phenology_captions",
    "load_parcel_captions",
    "make_ollama_text_client",
]
