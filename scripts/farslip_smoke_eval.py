"""Permanent operational script: smoke eval of the FarSLIP extractor (US-017 / US-016b).

Downloads weights (or uses a local cache), runs ``extract_embeddings`` over N
synthetic patches and reports time + shape + L2 norm. Does NOT violate the
``scripts/_*.py`` anti-pattern: it is a deployment operational script (verifies
that the model is served correctly from GCS before exposing it to US-016/US-025).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for CLI scripts. poetry add typer") from exc

from ml.extractors.farslip_extractor import FarSLIPExtractor

_log = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def smoke(
    weights_uri: Annotated[
        str | None, typer.Option(help="URI GCS o ruta local a student.safetensors")
    ] = None,
    n_patches: Annotated[int, typer.Option(help="Numero de patches sinteticos")] = 10,
    device: Annotated[str, typer.Option(help="Device cuda|cpu|auto")] = "auto",
    cache_dir: Annotated[Path | None, typer.Option(help="Cache dir local")] = None,
) -> None:
    """Download weights + run extract_embeddings + report diagnostics."""
    _log.info("starting farslip smoke eval", weights_uri=weights_uri, n_patches=n_patches)
    t0 = time.monotonic()
    extractor = FarSLIPExtractor(weights_uri=weights_uri, device=device, cache_dir=cache_dir)
    t_load = time.monotonic() - t0

    torch.manual_seed(42)
    crops = torch.rand(n_patches, 4, 256, 256, dtype=torch.float32)
    t0 = time.monotonic()
    emb = extractor.extract_embeddings(crops)
    t_inf = time.monotonic() - t0

    l2 = emb.norm(p=2, dim=-1)
    _log.info(
        "smoke eval done",
        load_seconds=round(t_load, 2),
        inference_seconds=round(t_inf, 2),
        embedding_shape=list(emb.shape),
        l2_min=float(l2.min().item()),
        l2_max=float(l2.max().item()),
        l2_mean=float(l2.mean().item()),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
