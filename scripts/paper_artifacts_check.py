"""Verify that every artefact listed in ``paper/ARTIFACTS.md`` still matches its seal.

The ledger is the custody record of the MICAI manuscript: each printed figure must
be re-derivable from a file whose MD5 is registered here. This gate recomputes the
digest of every registered path and fails when a file is missing, changed, or when
a row that claims a seal has no digest.

It also checks **provenance**, and the first version of that check was itself bypassable. It
now verifies three things a ledger row asserts and cannot be trusted on:

1. The row's commit exists, contains the path (or its ``.dvc`` pointer), and the **blob at that
   commit has the registered MD5**. Checking only today's bytes lets a row attribute them to a
   commit that never produced them, which is what three preregistration rows were doing.
2. The sealing commit in the header is an ancestor of HEAD **and no older than any row it
   seals**. Ancestry alone accepted the root commit, 467 commits back.
3. A row that says an artefact is not tracked by git is telling the truth.

Rows whose state is ``OBSOLETO`` are sealed and verified like any other — the bytes are what
they say — but were produced by code since found defective. **They cannot be cited.** The state is
executable so that "84 artefactos sellados, OK" stops reading as "84 cifras utilizables", which is
how an obsolete number reached the public notebook labelled as the corrected experiment.

Rows whose state is ``SIN_ARTEFACTO`` are expected to have no file: they document a
number that cannot be printed until the artefact exists. The gate reports them and
does not fail. A sealed row whose file is absent but has a sibling ``.dvc`` pointer is
reported apart, as a missing ``dvc pull`` rather than as a broken seal, so a fresh
clone gets an actionable message instead of a wall of false alarms.

Usage:
    poetry run python scripts/paper_artifacts_check.py
    poetry run python scripts/paper_artifacts_check.py --ledger paper/ARTIFACTS.md
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
MISSING_STATE = "SIN_ARTEFACTO"
STALE_STATE = "OBSOLETO"
ROW_RE = re.compile(r"^\|(?!\s*[-:]+\s*\|)(?P<cells>.+)\|\s*$")
CODE_RE = re.compile(r"`([^`]+)`")
CHUNK = 1024 * 1024
HEAD_RE = re.compile(r"[Cc]ommit de sellado\*\*:\s*`([0-9a-f]{7,40})`")
SIN_GIT = "sin seguimiento en git"
SHA_RE = re.compile(r"`([0-9a-f]{7,40})`")


def md5_of(path: Path) -> str:
    """Compute the MD5 digest of a file, streamed in chunks.

    Args:
        path: File to digest.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.md5()  # noqa: S324 - custody seal, not a security primitive
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ledger(ledger: Path) -> list[dict[str, str]]:
    """Parse the artefact rows of the ledger.

    A data row has at least six pipe-separated cells and carries the artefact path
    inside backticks in its second cell.

    Args:
        ledger: Path to ``paper/ARTIFACTS.md``.

    Returns:
        One dictionary per artefact row with ``element``, ``path``, ``md5``, ``git`` and
        ``state``.
    """
    rows: list[dict[str, str]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line.strip())
        if match is None:
            continue
        cells = [c.strip() for c in match.group("cells").split("|")]
        min_cells = 6
        if len(cells) < min_cells:
            continue
        path_cell = CODE_RE.search(cells[1])
        if path_cell is None:
            continue
        rows.append(
            {
                "element": cells[0],
                "path": path_cell.group(1),
                "md5": cells[2].strip("`").lower(),
                "git": cells[4],
                "state": cells[5],
            }
        )
    return rows


def _es_ancestro(sha: str, otro: str) -> bool:
    """Whether ``sha`` exists and is reachable from ``otro``."""
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, otro],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _blob(sha: str, path: str) -> bytes | None:
    """Contents of ``path`` at commit ``sha``, or ``None`` when it is not there."""
    proc = subprocess.run(
        ["git", "cat-file", "blob", f"{sha}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def _verificar_blob(sha: str, row: dict[str, str], esta_en_git: bool) -> list[str]:
    """Check that the commit a row names actually produced the bytes it registers.

    Comparing only today's bytes against today's file is not provenance: it accepts a row that
    attributes its content to a commit that never contained it. That bypass was found by an
    external audit, which replaced a row's commit with a nonexistent one and got a green gate.

    Args:
        sha: Commit the row claims.
        row: The parsed ledger row.
        esta_en_git: Whether the artefact itself is tracked by git.

    Returns:
        Zero or more failure messages.
    """
    ruta = row["path"]
    if not _es_ancestro(sha, "HEAD"):
        return [f"{ruta}: el commit {sha} no existe o no esta en la historia de HEAD"]
    if esta_en_git:
        contenido = _blob(sha, ruta)
        if contenido is None:
            return [f"{ruta}: el commit {sha} no contiene esa ruta"]
        digest = hashlib.md5(contenido).hexdigest()  # noqa: S324 - sello de custodia
        if digest != row["md5"]:
            return [f"{ruta}: en {sha} el MD5 es {digest} y el ledger registra {row['md5']}"]
        return []
    puntero = _blob(sha, f"{ruta}.dvc")
    if puntero is None:
        return [f"{ruta}: el commit {sha} no contiene ni la ruta ni su puntero .dvc"]
    if row["md5"] not in puntero.decode("utf-8", errors="replace"):
        return [f"{ruta}: el puntero .dvc de {sha} no registra el MD5 {row['md5']}"]
    return []


def main() -> int:
    """Check every ledger row and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"ERROR: no existe el ledger {args.ledger}")
        return 2

    rows = parse_ledger(args.ledger)
    if not rows:
        print(f"ERROR: el ledger {args.ledger} no declara ningun artefacto")
        return 2

    failures: list[str] = []
    dvc_missing: list[str] = []
    shas_de_fila: list[str] = []

    # Procedencia 1: el commit de sellado tiene que existir y estar en la historia de HEAD.
    # No se exige que sea HEAD exactamente, porque el commit que actualiza el ledger no puede
    # conocer su propio sha; se exige que no sea inventado ni de otra rama, y se imprime cuanto
    # ha quedado atras para que la obsolescencia sea visible en vez de silenciosa.
    declarado = HEAD_RE.search(args.ledger.read_text(encoding="utf-8"))
    sello = declarado.group(1) if declarado else None
    if sello is None:
        failures.append("la cabecera no declara ningun commit de sellado")
    elif not _es_ancestro(sello, "HEAD"):
        failures.append(f"el commit de sellado {sello} no existe o no esta en la historia de HEAD")

    # Procedencia 2: quien dice no estar versionado, no puede estarlo.
    seguidos = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
    )

    pending = 0
    checked = 0
    obsoletos: list[str] = []
    for row in rows:
        path = REPO_ROOT / row["path"]
        if row["state"] == MISSING_STATE:
            pending += 1
            if path.exists():
                failures.append(f"{row['path']}: marcado {MISSING_STATE} pero el archivo existe")
            continue
        if not path.exists():
            if path.with_suffix(path.suffix + ".dvc").exists():
                dvc_missing.append(row["path"])
            else:
                failures.append(f"{row['path']}: registrado con sello pero no esta en disco")
            continue
        if not re.fullmatch(r"[0-9a-f]{32}", row["md5"]):
            failures.append(f"{row['path']}: el ledger no trae un MD5 valido")
            continue
        actual = md5_of(path)
        checked += 1
        if row["state"] == STALE_STATE:
            obsoletos.append(row["path"])
        if actual != row["md5"]:
            failures.append(f"{row['path']}: MD5 {actual} no coincide con el sellado {row['md5']}")
        declara_sin_git = SIN_GIT in row["git"]
        esta_en_git = row["path"] in seguidos
        # Una celda como «sin seguimiento en git (`*.pdf` global)» lleva backticks y no declara
        # ningun commit: solo cuenta lo que tiene forma de sha.
        commit_fila = None if declara_sin_git else SHA_RE.search(row["git"])
        if commit_fila is not None:
            sha_fila = commit_fila.group(1)
            shas_de_fila.append(sha_fila)
            failures.extend(_verificar_blob(sha_fila, row, esta_en_git))
        if declara_sin_git and esta_en_git:
            failures.append(
                f"{row['path']}: el ledger dice «{SIN_GIT}» y el archivo si esta versionado"
            )
        # Un artefacto grande vive en DVC: el commit del ledger es el del puntero `.dvc`, y la
        # celda lo dice. Lo que no puede pasar es atribuir un commit a un archivo que no esta
        # ni en git ni en DVC.
        en_dvc = f"{row['path']}.dvc" in seguidos
        if not declara_sin_git and not esta_en_git and not en_dvc and SHA_RE.search(row["git"]):
            failures.append(
                f"{row['path']}: el ledger le atribuye un commit y no esta ni en git ni en DVC"
            )
        if en_dvc and ".dvc" not in row["git"]:
            failures.append(
                f"{row['path']}: esta versionado por DVC y el ledger no lo declara como `.dvc`"
            )

    # Procedencia 3: un sello no puede ser anterior a los artefactos que sella. La ancestria a
    # secas aceptaba el commit raiz, 467 commits atras, y respondia OK.
    if sello is not None and _es_ancestro(sello, "HEAD"):
        posteriores = [x for x in dict.fromkeys(shas_de_fila) if not _es_ancestro(x, sello)]
        if posteriores:
            failures.append(
                f"el commit de sellado {sello} es anterior a {len(posteriores)} fila(s) que sella: "
                + ", ".join(sorted(posteriores)[:5])
            )
        detras = subprocess.run(
            ["git", "rev-list", "--count", f"{sello}..HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"commit de sellado: {sello} ({detras} commits por detras de HEAD)")

    print(f"artefactos verificados: {checked}")
    print(f"  de los cuales OBSOLETOS, verificados pero NO citables: {len(obsoletos)}")
    print(f"filas sin artefacto (pendientes): {pending}")
    for failure in failures:
        print(f"FALLO: {failure}")
    if dvc_missing:
        print(f"pendientes de `dvc pull` ({len(dvc_missing)}), no es un fallo del sello:")
        for path_str in dvc_missing:
            print(f"  - {path_str}")
    if failures:
        print(f"paper-artifacts-check: {len(failures)} fallo(s)")
        return 1
    if dvc_missing:
        print("paper-artifacts-check: incompleto, ejecuta `dvc pull` y vuelve a correrlo")
        return 1
    if obsoletos:
        print(
            f"AVISO: {len(obsoletos)} artefacto(s) marcados OBSOLETO. Su sello es valido y sus "
            "cifras NO entran en el articulo hasta regenerarlas (US-124, US-125):"
        )
        for ruta in obsoletos:
            print(f"  - {ruta}")
    print("paper-artifacts-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
