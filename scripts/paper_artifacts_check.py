"""Verify that every artefact listed in ``paper/ARTIFACTS.md`` still matches its seal.

The ledger is the custody record of the MICAI manuscript: each printed figure must
be re-derivable from a file whose MD5 is registered here. This gate recomputes the
digest of every registered path and fails when a file is missing, changed, or when
a row that claims a seal has no digest.

It also checks **provenance**, which for two rounds it did not: the header's sealing commit
has to exist in the history of HEAD, and a row that says an artefact is not tracked by git has
to be telling the truth. An external audit found three rows claiming "sin seguimiento en
git" for files versioned in the very commit being audited, and a header pinned four commits
back; the gate passed because it only ever compared bytes.

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
ROW_RE = re.compile(r"^\|(?!\s*[-:]+\s*\|)(?P<cells>.+)\|\s*$")
CODE_RE = re.compile(r"`([^`]+)`")
CHUNK = 1024 * 1024
HEAD_RE = re.compile(r"[Cc]ommit de sellado\*\*:\s*`([0-9a-f]{7,40})`")
SIN_GIT = "sin seguimiento en git"


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

    # Procedencia 1: el commit de sellado tiene que existir y estar en la historia de HEAD.
    # No se exige que sea HEAD exactamente, porque el commit que actualiza el ledger no puede
    # conocer su propio sha; se exige que no sea inventado ni de otra rama, y se imprime cuanto
    # ha quedado atras para que la obsolescencia sea visible en vez de silenciosa.
    declarado = HEAD_RE.search(args.ledger.read_text(encoding="utf-8"))
    if declarado is None:
        failures.append("la cabecera no declara ningun commit de sellado")
    else:
        sha = declarado.group(1)
        alcanzable = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if alcanzable.returncode != 0:
            failures.append(
                f"el commit de sellado {sha} no existe o no esta en la historia de HEAD"
            )
        else:
            detras = subprocess.run(
                ["git", "rev-list", "--count", f"{sha}..HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            print(f"commit de sellado: {sha} ({detras} commits por detras de HEAD)")

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
        if actual != row["md5"]:
            failures.append(f"{row['path']}: MD5 {actual} no coincide con el sellado {row['md5']}")
        declara_sin_git = SIN_GIT in row["git"]
        esta_en_git = row["path"] in seguidos
        if declara_sin_git and esta_en_git:
            failures.append(
                f"{row['path']}: el ledger dice «{SIN_GIT}» y el archivo si esta versionado"
            )
        # Un artefacto grande vive en DVC: el commit del ledger es el del puntero `.dvc`, y la
        # celda lo dice. Lo que no puede pasar es atribuir un commit a un archivo que no esta
        # ni en git ni en DVC.
        en_dvc = f"{row['path']}.dvc" in seguidos
        if not declara_sin_git and not esta_en_git and not en_dvc and CODE_RE.search(row["git"]):
            failures.append(
                f"{row['path']}: el ledger le atribuye un commit y no esta ni en git ni en DVC"
            )
        if en_dvc and ".dvc" not in row["git"]:
            failures.append(
                f"{row['path']}: esta versionado por DVC y el ledger no lo declara como `.dvc`"
            )

    print(f"artefactos sellados verificados: {checked}")
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
    print("paper-artifacts-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
