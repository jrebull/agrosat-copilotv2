"""Generate per-parcel phenology captions over real PASTIS-R (US-036-b, Phase A).

Computes the per-parcel NDVI temporal curve and turns it into a diverse phenology
description with the LOCAL Gemma (Ollama, cost $0), writing them to a parquet with
incremental flush + resume (resilient to SSH/tunnel cuts on the H100). This is the
input of the parcel-level FarSLIP sweep (``run_us036b_parcel_sweep``), and the fix
for the ~60%-identical patch-level captions.

Usage (on the H100 VM, env ``agrosat``)::

    python -m scripts.run_us036b_parcel_captions run \
        --pastis-root data/PASTIS-R \
        --out data/farslip/parcel_phenology_captions.parquet \
        --folds 1,2,3,4 --model gemma4:31b-it-q8_0

Only real PASTIS-R French data; the prompt (Wen et al. 2025) describes phenology
from the NDVI curve and never reveals the label. Conventions: Polars, structlog,
type hints, English docstrings, Spanish prose, no emojis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import typer

from ml.features import phenology_description as pheno_desc
from ml.features.parcel_phenology_captions import (
    compute_parcel_ndvi_curves,
    generate_parcel_phenology_captions,
    make_ollama_text_client,
)
from ml.ingest.pastis_loader import PASTIS_R_CLASSES

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _parse_ints(s: str) -> tuple[int, ...]:
    """Parse a comma-separated int list."""
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _balanced_subsample(curves: dict[str, tuple], max_per_class: int) -> dict[str, tuple]:
    """Keep at most ``max_per_class`` parcels per class (deterministic).

    Groups parcels by ``class_id`` (the 3rd tuple element) and keeps the first
    ``max_per_class`` of each, sorted by parcel_id for reproducibility. Rare
    classes with fewer parcels keep all of theirs. This validates the experiment
    on a balanced sample before spending on the full set.

    Args:
        curves: ``{parcel_id: (curve, doy, class_id)}``.
        max_per_class: cap per class.

    Returns:
        The subsampled mapping.
    """
    by_class: dict[int, list[str]] = {}
    for pid, (_c, _d, cid) in curves.items():
        by_class.setdefault(int(cid), []).append(pid)
    kept: dict[str, tuple] = {}
    for pids in by_class.values():
        for pid in sorted(pids, key=lambda p: (int(p.rsplit("_", 1)[0]), p))[:max_per_class]:
            kept[pid] = curves[pid]
    logger.info(
        "balanced_subsample",
        max_per_class=max_per_class,
        n_classes=len(by_class),
        n_kept=len(kept),
        n_total=len(curves),
    )
    return kept


@app.callback()
def _root() -> None:
    """AgroSatCopilot US-036-b Phase A: per-parcel phenology captions (Gemma)."""


@app.command()
def run(
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    out: Annotated[Path, typer.Option("--out")] = Path(
        "data/farslip/parcel_phenology_captions.parquet"
    ),
    folds: Annotated[str, typer.Option("--folds")] = "1,2,3,4",
    active_class_ids: Annotated[
        str, typer.Option("--active-class-ids", help="empty = all 18 crops")
    ] = "",
    min_area_px: Annotated[int, typer.Option("--min-area-px")] = 16,
    model: Annotated[str, typer.Option("--model")] = "gemma4:31b-it-q8_0",
    base_url: Annotated[str, typer.Option("--base-url")] = "http://127.0.0.1:11434",
    flush_every: Annotated[int, typer.Option("--flush-every")] = 50,
    max_workers: Annotated[
        int, typer.Option("--max-workers", help="1=local Gemma; 16-32=Gemini cloud")
    ] = 1,
    max_per_class: Annotated[
        int,
        typer.Option(
            "--max-per-class",
            help="0=all; else cap parcels per class (balanced sample, deterministic)",
        ),
    ] = 0,
    max_patches: Annotated[int, typer.Option("--max-patches")] = 0,
) -> None:
    """Compute per-parcel NDVI curves and generate diverse phenology captions."""
    # Load .env.local so GEMINI_API_KEY / AGROSAT_LLM_PROVIDER are present when
    # the default (cloud) client is used. No-op when the file is absent (the VM
    # uses local Gemma via the Ollama client, which needs no key).
    from ml.utils.notebook_setup import find_repo_root, load_env_local

    load_env_local(find_repo_root())

    active = _parse_ints(active_class_ids) if active_class_ids else tuple(range(1, 19))
    fold_tuple = _parse_ints(folds)
    logger.info(
        "parcel_captions_phase_a_start",
        pastis_root=str(pastis_root),
        out=str(out),
        folds=list(fold_tuple),
        active_class_ids=list(active),
        model=model,
        max_per_class=max_per_class,
    )

    curves = compute_parcel_ndvi_curves(
        pastis_root,
        folds=fold_tuple,
        active_class_ids=active,
        min_area_px=min_area_px,
        max_patches=max_patches if max_patches > 0 else None,
    )
    if max_per_class > 0:
        curves = _balanced_subsample(curves, max_per_class)
    logger.info("parcel_curves_ready", n_parcels=len(curves))

    # Client selection: a local Gemma (Ollama) model name -> inject the Ollama
    # text client (cost $0). A Gemini model -> leave the default client
    # (AGROSAT_LLM_PROVIDER=google-genai). The phenology prompt is TEXT-ONLY
    # (NDVI curve), so Gemini cost is tiny (~$5 for all parcels, not image
    # pricing). ``max_workers`` drives concurrency for BOTH: Ollama batches the
    # concurrent requests on the H100 GPU when OLLAMA_NUM_PARALLEL is set, and a
    # cloud LLM runs them as concurrent HTTP calls.
    uses_ollama = model.lower().startswith("gemma")
    if uses_ollama:
        pheno_desc.set_llm_client(make_ollama_text_client(base_url=base_url))
    workers = max_workers
    try:
        path = generate_parcel_phenology_captions(
            curves,
            PASTIS_R_CLASSES,
            output_path=out,
            model=model,
            flush_every=flush_every,
            max_workers=workers,
        )
    finally:
        if uses_ollama:
            pheno_desc.set_llm_client(None)

    logger.info(
        "parcel_captions_phase_a_done",
        path=str(path),
        n_parcels=len(curves),
        model=model,
        max_workers=workers,
    )


if __name__ == "__main__":
    app()
