"""Comprueba el catalogo de referencias y su relacion con los dos manuscritos.

Hay DOS bibliografias y no se mezclan:

- `paper/micai/refs.bib` es **historica e inmutable**, ligada al PDF retirado. Este gate comprueba
  que su hash siga siendo el sellado: ninguna correccion puede repararse tocandola, porque eso
  rompe la cadena del manuscrito archivado.
- `paper/micai2027/refs-candidates.bib` es el **catalogo verificado** del manuscrito nuevo. Se
  llama «candidates» a proposito: mientras no exista el manuscrito no existen citas nuevas, y un
  `refs.bib` final se generara despues **solo con las claves efectivamente citadas**. Atribuir al
  contexto de MICAI 2027 las citas del manuscrito retirado seria contar como propio el trabajo de
  otro documento.

Reglas:

1. Toda entrada del catalogo tiene identificador localizable: DOI, eprint de arXiv o URL.
2. Autores completos. Un `and others` deja una entrada cuya atribucion nadie puede comprobar.
3. Los identificadores se conservan LITERALES. Convertir `2511.10370` en un numero produce
   `2511.1037`, que es un eprint que no existe, y el cero final desaparece sin que nada avise.
4. Las entradas sin citar se REPORTAN y no hacen fallar: hoy no hay manuscrito nuevo que las cite.

Uso:
    poetry run python scripts/paper_bib_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOGO = REPO_ROOT / "paper" / "micai2027" / "refs-candidates.bib"
DEFAULT_TEX = REPO_ROOT / "paper" / "micai2027"
HISTORICO = REPO_ROOT / "paper" / "micai" / "refs.bib"
LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
OVERRIDES = REPO_ROOT / "reports" / "paper_micai" / "fase0" / "related_work_overrides.csv"

ENTRADA_RE = re.compile(r"^@(\w+)\{([^,]+),", re.M)
CITA_RE = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")
CAMPO_RE = re.compile(r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", re.M)

#: Campos que hacen localizable una entrada. Basta uno.
IDENTIFICADORES = ("doi", "eprint", "url")

#: Columnas del fichero de correcciones que NUNCA pueden perder su forma textual. Un identificador
#: convertido a numero pierde ceros finales y deja de resolver.
COLUMNAS_LITERALES = ("id", "volume", "number", "pages", "url")


def _entradas(bib: str) -> dict[str, dict[str, str]]:
    """Map every bib key to its fields."""
    salida: dict[str, dict[str, str]] = {}
    posiciones = [(m.start(), m.group(2).strip()) for m in ENTRADA_RE.finditer(bib)]
    for i, (inicio, clave) in enumerate(posiciones):
        fin = posiciones[i + 1][0] if i + 1 < len(posiciones) else len(bib)
        cuerpo = bib[inicio:fin]
        salida[clave] = {k.lower(): v for k, v in CAMPO_RE.findall(cuerpo)}
    return salida


def _hash_sellado(ruta_relativa: str) -> str | None:
    """MD5 the ledger seals for a path, or ``None`` when it has no row."""
    if not LEDGER.exists():
        return None
    for linea in LEDGER.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|"):
            continue
        celdas = linea.strip().strip("|").split("|")
        # La ruta se busca en SU CELDA, no en cualquier parte de la fila. Buscarla como
        # subcadena hacia que la nota de otra fila —que menciona este fichero para decir que es
        # inmutable— se tomara por la fila del fichero, y el gate comparaba su hash con el de
        # otro artefacto. Lo dijo como si el bib historico hubiera cambiado, y no habia cambiado.
        min_celdas = 6
        if len(celdas) < min_celdas:
            continue
        ruta = re.search(r"`([^`]+)`", celdas[1])
        if ruta is None or ruta.group(1) != ruta_relativa:
            continue
        m = re.search(r"`([0-9a-f]{32})`", celdas[2])
        if m:
            return m.group(1)
    return None


def main() -> int:
    """Check the catalogue, the historical bibliography and the corrections file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogo", type=Path, default=DEFAULT_CATALOGO)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    args = parser.parse_args()

    if not args.catalogo.exists():
        print(f"ERROR: no existe el catalogo {args.catalogo}")
        return 2
    entradas = _entradas(args.catalogo.read_text(encoding="utf-8"))
    fallos: list[str] = []

    sin_id = sorted(k for k, campos in entradas.items() if not set(campos) & set(IDENTIFICADORES))
    for clave in sin_id:
        fallos.append(f"{clave}: sin identificador localizable ({'/'.join(IDENTIFICADORES)})")

    con_others = sorted(
        k for k, campos in entradas.items() if "others" in campos.get("author", "").lower()
    )
    for clave in con_others:
        fallos.append(f"{clave}: la lista de autores termina en «others» y no esta completa")

    # 3. Los identificadores, literales. Un cero final perdido no se ve leyendo el bib.
    if args.overrides.exists():
        import polars as pl

        crudo = pl.read_csv(
            args.overrides, schema_overrides={c: pl.Utf8 for c in COLUMNAS_LITERALES}
        )
        for fila in crudo.to_dicts():
            eprint_esperado = str(fila.get("id") or "").strip()
            if not eprint_esperado or fila.get("id_type") != "arxiv":
                continue
            emitido = entradas.get(str(fila["key"]), {}).get("eprint", "")
            if emitido and emitido != eprint_esperado:
                fallos.append(
                    f"{fila['key']}: el bib publica eprint {emitido!r} y la correccion dice "
                    f"{eprint_esperado!r}; un identificador convertido a numero pierde el "
                    "cero final"
                )

    # 4. La bibliografia historica no se toca.
    sellado = _hash_sellado("paper/micai/refs.bib")
    if HISTORICO.exists() and sellado:
        actual = hashlib.md5(HISTORICO.read_bytes()).hexdigest()  # noqa: S324
        if actual != sellado:
            fallos.append(
                f"paper/micai/refs.bib cambio ({actual} frente al sellado {sellado}): es "
                "historica e inmutable, y ninguna correccion puede repararse tocandola"
            )

    citadas: set[str] = set()
    if args.tex.exists():
        tex = "".join(p.read_text(encoding="utf-8") for p in sorted(args.tex.rglob("*.tex")))
        for grupo in CITA_RE.findall(tex):
            citadas |= {c.strip() for c in grupo.split(",") if c.strip()}
    for clave in sorted(citadas - set(entradas)):
        fallos.append(f"{clave}: citada en el manuscrito nuevo y ausente del catalogo")

    sin_citar = sorted(set(entradas) - citadas)
    print(f"entradas en el catalogo: {len(entradas)}")
    print(f"citadas en el manuscrito nuevo: {len(citadas)}")
    print(f"sin citar (se reporta, no falla): {len(sin_citar)}")
    if not citadas:
        print("  todavia no hay manuscrito nuevo: el refs.bib final se genera cuando lo haya")
    print(f"sin identificador localizable: {len(sin_id)}")
    for clave in sin_id:
        print(f"  - {clave}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"paper-bib-check: {len(fallos)} fallo(s)")
        return 1
    print("paper-bib-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
