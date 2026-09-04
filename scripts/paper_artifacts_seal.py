"""Re-derive the provenance column of ``paper/ARTIFACTS.md`` from git, instead of by hand.

The custody ledger states, for every artefact, the commit whose bytes it registers. That column
was filled in by hand and by a one-off script, and an external audit found ten rows attributing
their MD5 to a commit that never produced it: five paths that did not exist there and five blobs
with a different digest.

This script computes the column: for each row it walks the file's history and picks the most
recent commit whose blob matches the registered MD5 (or, for a DVC-tracked artefact, whose
``.dvc`` pointer records it). It also sets the sealing commit of the header to the newest commit
it had to name, because **a seal cannot predate what it seals**.

A file whose current bytes are not committed yet has no commit to name. That is not a bug and it
is not papered over: the script says so and leaves the row alone, so the ledger is sealed in a
commit that comes after the one carrying the bytes.

Uso:
    poetry run python scripts/paper_artifacts_seal.py            # informa, no escribe
    poetry run python scripts/paper_artifacts_seal.py --escribir
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
SIN_GIT = "sin seguimiento en git"
CODE_RE = re.compile(r"`([^`]+)`")
SELLO_RE = re.compile(r"(\*\*Commit de sellado\*\*: `)([0-9a-f]{7,40})(`)")
MIN_CELDAS = 6


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    """Run a git command in the repository root."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, check=False)


def _commits_de(path: str) -> list[str]:
    """Commits that touched a path, newest first."""
    salida = _git("log", "--format=%H", "--", path).stdout.decode()
    return salida.split()


def _commit_del_md5(path: str, md5: str, *, en_git: bool) -> str | None:
    """Most recent commit whose stored bytes for ``path`` have digest ``md5``.

    Args:
        path: Artefact path as written in the ledger.
        md5: Digest the ledger registers.
        en_git: Whether the artefact itself is tracked (otherwise its ``.dvc`` pointer is).

    Returns:
        The short SHA, or ``None`` when no commit carries those bytes.
    """
    objetivo = path if en_git else f"{path}.dvc"
    for sha in _commits_de(objetivo):
        blob = _git("cat-file", "blob", f"{sha}:{objetivo}")
        if blob.returncode != 0:
            continue
        if en_git:
            if hashlib.md5(blob.stdout).hexdigest() == md5:  # noqa: S324 - sello de custodia
                return _git("rev-parse", "--short", sha).stdout.decode().strip()
        elif md5 in blob.stdout.decode("utf-8", errors="replace"):
            return _git("rev-parse", "--short", sha).stdout.decode().strip()
    return None


def main() -> int:
    """Recompute the provenance column and report what could not be sealed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--escribir", action="store_true")
    args = parser.parse_args()

    seguidos = set(_git("ls-files").stdout.decode().splitlines())
    lineas = args.ledger.read_text(encoding="utf-8").splitlines()
    cambios, sin_commit, shas = 0, [], []

    for i, linea in enumerate(lineas):
        if not linea.startswith("|") or re.match(r"^\|\s*[-:]+\s*\|", linea):
            continue
        celdas = linea.strip().strip("|").split("|")
        if len(celdas) < MIN_CELDAS:
            continue
        ruta_cell = CODE_RE.search(celdas[1])
        md5_cell = celdas[2].strip().strip("`").lower()
        if ruta_cell is None or not re.fullmatch(r"[0-9a-f]{32}", md5_cell):
            continue
        ruta = ruta_cell.group(1)
        en_git = ruta in seguidos
        en_dvc = f"{ruta}.dvc" in seguidos
        if not en_git and not en_dvc:
            continue
        sha = _commit_del_md5(ruta, md5_cell, en_git=en_git)
        if sha is None:
            sin_commit.append(ruta)
            continue
        shas.append(sha)
        nueva = f" `{sha}`{'' if en_git else ' (.dvc)'} "
        if celdas[4] != nueva:
            celdas[4] = nueva
            lineas[i] = "|" + "|".join(celdas) + "|"
            cambios += 1

    # El sello no puede ser anterior a ninguna fila: se toma el mas nuevo de los commits usados.
    if shas:
        mas_nuevo = max(shas, key=lambda x: len(_git("rev-list", f"{x}..HEAD").stdout.split()) * -1)
        texto = "\n".join(lineas)
        texto, n = SELLO_RE.subn(rf"\g<1>{mas_nuevo}\g<3>", texto)
        if n:
            lineas = texto.splitlines()
            print(f"commit de sellado: {mas_nuevo}")

    print(f"filas con procedencia recalculada: {cambios}")
    if sin_commit:
        print(f"filas cuyos bytes AUN NO estan en ningun commit ({len(sin_commit)}):")
        for ruta in sin_commit:
            print(f"  - {ruta}")
        print("  commitea esos artefactos y vuelve a sellar: un sello no puede preceder al commit")
    if args.escribir:
        args.ledger.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        print(f"escrito {args.ledger}")
    else:
        print("modo informe: no se ha escrito nada (usa --escribir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
