"""Impide que un documento activo cite cifras de artefactos marcados OBSOLETO sin decirlo.

El ledger sabe que trece artefactos salieron de un modulo con tres defectos, y lo marca. Pero
marcarlo no impedia nada: el gate de custodia los verificaba, imprimia un aviso y devolvia exito,
mientras varios documentos seguian presentando sus cifras como vigentes. Un aviso en prosa contra
un estado ejecutable lo gana siempre el estado, y por eso una cifra obsoleta llego al cuaderno
publico presentada como el experimento corregido.

Este gate separa **custodia** de **disponibilidad editorial**:

- Un documento que cita cifras de un artefacto OBSOLETO tiene que llevar la marca de cuarentena.
- Un documento que menciona la RUTA de un artefacto OBSOLETO y no esta declarado como consumidor
  falla, para que la lista no envejezca en silencio.

La marca es una linea que empieza por ``> **CUARENTENA**`` en Markdown, o el atributo
``data-cuarentena`` en HTML. Se pone arriba, donde se lee antes que las cifras.

Uso:
    poetry run python scripts/paper_obsoletos_check.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
STALE_STATE = "OBSOLETO"
CODE_RE = re.compile(r"`([^`]+)`")
MIN_CELDAS = 6

#: Marca que un documento tiene que llevar para poder citar cifras obsoletas.
MARCA_MD = "> **CUARENTENA**"
MARCA_HTML = "data-cuarentena"

#: Documentos que consumen cifras de artefactos OBSOLETO y estan declarados como tales.
#: Anadir uno aqui NO le da permiso: solo declara que existe. El permiso lo da la marca.
CONSUMIDORES: tuple[str, ...] = (
    "docs/paper/fase3-hallazgos.md",
    "docs/paper/fase4-hallazgos.md",
    "docs/paper/que-paper-sale.md",
    "docs/paper/campo-de-tiro.md",
    "docs/paper/recomendacion-final.md",
    "docs/paper/reencuadre-2026-09-03.md",
    "docs/paper/auditoria-revisores-2026-09-03.md",
    "docs/paper/revision-arthur-2026-09-03.md",
)

#: Documentos que hablan DE la obsolescencia y no citan cifras como vigentes.
EXENTOS: tuple[str, ...] = (
    "docs/paper/respuesta-auditoria-externa.md",
    "docs/paper/auditoria-externa/prompt-revalidacion.md",
    "docs/paper/auditoria-externa/prompt-auditoria-externa.md",
    "paper/ARTIFACTS.md",
    "paper/micai/ESTADO.md",
)


def rutas_obsoletas(ledger: Path) -> list[str]:
    """Artefact paths whose ledger row is marked ``OBSOLETO``.

    Args:
        ledger: Path to the custody ledger.

    Returns:
        The obsolete artefact paths, in ledger order.
    """
    salida: list[str] = []
    for linea in ledger.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|") or re.match(r"^\|\s*[-:]+\s*\|", linea):
            continue
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) < MIN_CELDAS or celdas[5] != STALE_STATE:
            continue
        ruta = CODE_RE.search(celdas[1])
        if ruta is not None:
            salida.append(ruta.group(1))
    return salida


def tiene_marca(texto: str, sufijo: str) -> bool:
    """Whether a document carries the quarantine banner."""
    return MARCA_HTML in texto if sufijo == ".html" else MARCA_MD in texto


def main() -> int:
    """Check every declared consumer and hunt for undeclared ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    obsoletas = rutas_obsoletas(args.ledger)
    if not obsoletas:
        print("no hay artefactos OBSOLETO: nada que vigilar")
        return 0
    print(f"artefactos OBSOLETO: {len(obsoletas)}")

    fallos: list[str] = []
    for relativo in CONSUMIDORES:
        ruta = REPO_ROOT / relativo
        if not ruta.exists():
            fallos.append(f"{relativo}: declarado consumidor y no existe")
            continue
        if not tiene_marca(ruta.read_text(encoding="utf-8"), ruta.suffix):
            fallos.append(
                f"{relativo}: cita cifras de artefactos OBSOLETO y no lleva la marca de cuarentena"
            )

    # Cualquier documento que nombre una ruta obsoleta y no este declarado ni exento.
    declarados = set(CONSUMIDORES) | set(EXENTOS)
    for ruta in sorted((REPO_ROOT / "docs" / "paper").rglob("*.md")):
        relativo = str(ruta.relative_to(REPO_ROOT))
        if relativo in declarados:
            continue
        texto = ruta.read_text(encoding="utf-8")
        citadas = [x for x in obsoletas if x in texto]
        if citadas:
            fallos.append(
                f"{relativo}: nombra {len(citadas)} artefacto(s) OBSOLETO y no esta declarado "
                f"como consumidor ni exento (p. ej. {citadas[0]})"
            )

    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"paper-obsoletos-check: {len(fallos)} fallo(s)")
        return 1
    print(f"consumidores declarados con cuarentena: {len(CONSUMIDORES)}")
    print("paper-obsoletos-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
