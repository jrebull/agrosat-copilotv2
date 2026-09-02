"""Orquestador de captions globales ``L_glo`` para FarSLIP fiel (US-036-a v2).

Genera la caption por patch de PASTIS-R con Gemma 4 multimodal (Ollama en la VM
H100) y la materializa en ``data/farslip/pastis_captions.parquet`` con flush
incremental (resiliente a cortes de SSH/tunel: un corte pierde a lo sumo
``flush_every`` captions, y ``resume=True`` continua desde el ultimo flush).

Uso (en la VM, env ``agrosat``)::

    python -m scripts.run_us036a_v2_captions \
        --pastis-root data/PASTIS-R \
        --out data/farslip/pastis_captions.parquet \
        --folds 1,2,3,4,5 \
        --flush-every 25

Anti data-leakage: el prompt NO inyecta valores numericos de NDVI/indices del
patch ni AlphaEarth ni la etiqueta como respuesta; al terminar corre
``audit_captions`` (cero coincidencias = limpio, AC-4). Convenciones del
proyecto: Polars, ``structlog``, type hints, docstrings en ingles, prosa en
espanol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import typer

from ml.farslip.caption_cache import audit_captions, generate_captions_parquet
from ml.farslip.caption_generator import GemmaCaptionClient

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _parse_folds(folds: str) -> tuple[int, ...]:
    """Parses a comma-separated folds string into a tuple of ints."""
    return tuple(int(f.strip()) for f in folds.split(",") if f.strip())


@app.command()
def run(
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    out: Annotated[Path, typer.Option("--out")] = Path("data/farslip/pastis_captions.parquet"),
    folds: Annotated[str, typer.Option("--folds")] = "1,2,3,4,5",
    flush_every: Annotated[int, typer.Option("--flush-every")] = 25,
    base_url: Annotated[str, typer.Option("--base-url")] = "http://127.0.0.1:11434",
    model: Annotated[str, typer.Option("--model")] = "gemma4:31b-it-q8_0",
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Genera (idempotente) las captions globales y audita la fuga al terminar."""
    fold_tuple = _parse_folds(folds)
    client = GemmaCaptionClient(base_url=base_url, model=model)
    logger.info(
        "us036a_v2_captions_start",
        pastis_root=str(pastis_root),
        out=str(out),
        folds=fold_tuple,
        flush_every=flush_every,
        model=model,
        resume=resume,
    )
    path = generate_captions_parquet(
        pastis_root=pastis_root,
        out_path=out,
        folds=fold_tuple,
        client=client,
        resume=resume,
        flush_every=flush_every,
    )
    counts = audit_captions(path)
    total_leaks = sum(counts.values())
    logger.info(
        "us036a_v2_captions_done",
        path=str(path),
        total_leaks=total_leaks,
        **counts,
    )
    if total_leaks > 0:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
