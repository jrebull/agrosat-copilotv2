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
OVERRIDES = REPO_ROOT / "reports" / "paper_micai" / "fase0" / "related_work_overrides.csv"
OUT = REPO_ROOT / "paper" / "micai" / "refs.bib"

#: Numero de autores a partir del cual el estilo debe imprimir «et al.».
MAX_AUTHORS: int = 6

#: Palabras que se protegen con llaves aunque no parezcan siglas.
PROTECTED = (
    "Sentinel",
    "AlphaEarth",
    "Earth",
    "EuroCrops",
    "PASTIS",
    "BreizhCrops",
    "Google",
    "Landsat",
    "European",
    "Europe",
    "African",
    "Africa",
)


def _protect(title: str) -> str:
    """Brace acronyms and proper nouns so the style does not lowercase them.

    Args:
        title: Verified title as the API returned it.

    Returns:
        The title with protected tokens wrapped in braces.
    """
    if title.isupper():
        title = title.title()

    def ya_protegido(text: str, start: int) -> bool:
        """Say whether an offset already sits inside a brace group."""
        return text.count("{", 0, start) > text.count("}", 0, start)

    def wrap(match: re.Match[str]) -> str:
        token = match.group(0)
        if ya_protegido(match.string, match.start()):
            return token
        return token if token.startswith("{") else "{" + token + "}"

    title = re.sub(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", wrap, title)
    for word in PROTECTED:
        title = re.sub(rf"(?<!{{)\b{word}\b(?!}})", "{" + word + "}", title)
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


#: Campos que una correccion puede aportar, en el orden en que se imprimen.
EXTRA_FIELDS: tuple[tuple[str, str], ...] = (
    ("booktitle", "booktitle"),
    ("series", "series"),
    ("volume", "volume"),
    ("number", "number"),
    ("pages", "pages"),
    ("publisher", "publisher"),
    ("address", "address"),
)


def _entry(row: dict[str, str]) -> str:
    """Render one verified row as a BibTeX entry.

    Args:
        row: Row of the verified matrix.

    Returns:
        The BibTeX entry, without a trailing blank line.
    """
    doi = str(row.get("doi_verified") or "").strip()
    venue = str(row.get("venue_verified") or "").strip()
    kind = str(row.get("entry_type") or "").strip()
    if not kind:
        kind = "article" if doi and "arxiv" not in doi.lower() else "misc"
    fields = [
        f"  author    = {{{_authors(row['authors_verified'])}}},",
        f"  title     = {{{_protect(row['title_verified'])}}},",
        f"  year      = {{{row['year_verified']}}},",
    ]
    if venue and kind == "article":
        fields.append(f"  journal   = {{{venue}}},")
    for csv_name, bib_name in EXTRA_FIELDS:
        value = str(row.get(csv_name) or "").strip()
        if value:
            fields.append(f"  {bib_name:<9} = {{{value}}},")
    if doi:
        fields.append(f"  doi       = {{{doi}}},")
    elif row.get("id_type") == "arxiv" and kind == "misc":
        fields.append(f"  eprint    = {{{row['id']}}},")
        fields.append("  archivePrefix = {arXiv},")
    return f"@{kind}{{{row['key']},\n" + "\n".join(fields) + "\n}"


def _apply_overrides(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Overlay the hand-verified corrections on the API-resolved rows.

    The resolved CSV keeps exactly what the API returned, so its provenance stays intact; the
    corrections live apart, each with the reason it was needed, and this is where the two meet.
    An API answer can be wrong —one of these DOIs resolved to a different dataset entirely— and
    the fix has to be auditable rather than pasted over the evidence.

    Args:
        rows: Rows as the resolver produced them.

    Returns:
        The same rows with every correction applied.
    """
    if not OVERRIDES.exists():
        return rows
    fixes = {r["key"]: r for r in pl.read_csv(OVERRIDES).to_dicts()}
    out: list[dict[str, str]] = []
    vistas: set[str] = set()
    for row in rows:
        fix = fixes.get(row["key"])
        if fix:
            row = row | {k: v for k, v in fix.items() if k != "motivo" and v not in (None, "")}
            vistas.add(row["key"])
        out.append(row)
    # Una entrada puede existir solo en las correcciones: la busqueda por API no encontro el
    # trabajo que introduce PASTIS-R, y hace falta para poder decir que NO usamos su radar.
    nuevas = [k for k in fixes if k not in vistas]
    for key in sorted(nuevas):
        out.append({k: v for k, v in fixes[key].items() if k != "motivo"})
    logger.info("correcciones_aplicadas", parcheadas=len(vistas), anadidas=len(nuevas))
    return sorted(out, key=lambda r: r["key"])


def main() -> None:
    """Write the bibliography from every verified row."""
    rows = _apply_overrides(
        pl.read_csv(VERIFIED).filter(pl.col("status") == "OK").sort("key").to_dicts()
    )
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
