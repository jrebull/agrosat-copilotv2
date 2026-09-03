"""Genera `paper/micai/refs.bib` desde la matriz verificada de la fase 0.

La bibliografia no se escribe a mano: se deriva del CSV que resolvio cada entrada contra
arXiv, Crossref u OpenAlex, de modo que ninguna cita del articulo pueda existir sin haber
sido comprobada por una API. Es el mismo principio que el ledger de artefactos aplicado a
las referencias.

Reglas de estilo del articulo que este generador aplica:

- Mas de seis autores se listan seis y `others`, que `splncs04` imprime como «et al.».
- Las siglas y los nombres propios del titulo se protegen con llaves para que el estilo no
  los pase a minusculas.
- Sin campos `note`.

Uso:
    poetry run python scripts/build_paper_micai_bib.py
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED = REPO_ROOT / "reports" / "paper_micai" / "fase0" / "related_work_verified.csv"
OUT = REPO_ROOT / "paper" / "micai" / "refs.bib"

#: Numero de autores a partir del cual el estilo debe imprimir «et al.».
MAX_AUTHORS: int = 6

#: Palabras que se protegen con llaves aunque no parezcan siglas.
PROTECTED = ("Sentinel", "AlphaEarth", "Earth", "EuroCrops", "PASTIS", "BreizhCrops")


def _protect(title: str) -> str:
    """Brace acronyms and proper nouns so the style does not lowercase them.

    Args:
        title: Verified title as the API returned it.

    Returns:
        The title with protected tokens wrapped in braces.
    """
    if title.isupper():
        title = title.title()

    def wrap(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.startswith("{") else "{" + token + "}"

    title = re.sub(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", wrap, title)
    for word in PROTECTED:
        title = re.sub(rf"\b{word}\b", "{" + word + "}", title)
    return title


def _authors(raw: str) -> str:
    """Format the verified author list for BibTeX.

    Args:
        raw: Semicolon-separated author list.

    Returns:
        A BibTeX ``and``-joined list, truncated with ``others`` past the limit.
    """
    people = [a.strip() for a in raw.split(";") if a.strip()]
    if len(people) > MAX_AUTHORS:
        return " and ".join(people[:MAX_AUTHORS]) + " and others"
    return " and ".join(people)


def _entry(row: dict[str, str]) -> str:
    """Render one verified row as a BibTeX entry.

    Args:
        row: Row of the verified matrix.

    Returns:
        The BibTeX entry, without a trailing blank line.
    """
    doi = (row.get("doi_verified") or "").strip()
    venue = (row.get("venue_verified") or "").strip()
    kind = "article" if doi and "arxiv" not in doi.lower() else "misc"
    fields = [
        f"  author    = {{{_authors(row['authors_verified'])}}},",
        f"  title     = {{{_protect(row['title_verified'])}}},",
        f"  year      = {{{row['year_verified']}}},",
    ]
    if venue:
        fields.append(f"  journal   = {{{venue}}},")
    if doi:
        fields.append(f"  doi       = {{{doi}}},")
    elif row.get("id_type") == "arxiv":
        fields.append(f"  eprint    = {{{row['id']}}},")
        fields.append("  archivePrefix = {arXiv},")
    return f"@{kind}{{{row['key']},\n" + "\n".join(fields) + "\n}"


def main() -> None:
    """Write the bibliography from every verified row."""
    rows = pl.read_csv(VERIFIED).filter(pl.col("status") == "OK").sort("key").to_dicts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "% Generado por scripts/build_paper_micai_bib.py desde\n"
        "% reports/paper_micai/fase0/related_work_verified.csv.\n"
        "% NO editar a mano: toda entrada nace de una resolucion por API.\n\n"
    )
    OUT.write_text(header + "\n\n".join(_entry(r) for r in rows) + "\n", encoding="utf-8")
    logger.info("bib_escrito", entradas=len(rows), out=str(OUT.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
