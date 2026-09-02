"""Verify that every artefact listed in ``paper/ARTIFACTS.md`` still matches its seal.

The ledger is the custody record of the MICAI manuscript: each printed figure must
be re-derivable from a file whose MD5 is registered here. This gate recomputes the
digest of every registered path and fails when a file is missing, changed, or when
a row that claims a seal has no digest.

Rows whose state is ``SIN_ARTEFACTO`` are expected to have no file: they document a
number that cannot be printed until the artefact exists. The gate reports them and
does not fail.

Usage:
    poetry run python scripts/paper_artifacts_check.py
    poetry run python scripts/paper_artifacts_check.py --ledger paper/ARTIFACTS.md
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
MISSING_STATE = "SIN_ARTEFACTO"
ROW_RE = re.compile(r"^\|(?!\s*[-:]+\s*\|)(?P<cells>.+)\|\s*$")
CODE_RE = re.compile(r"`([^`]+)`")
CHUNK = 1024 * 1024


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
        One dictionary per artefact row with ``element``, ``path``, ``md5`` and ``state``.
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
            failures.append(f"{row['path']}: registrado con sello pero no esta en disco")
            continue
        if not re.fullmatch(r"[0-9a-f]{32}", row["md5"]):
            failures.append(f"{row['path']}: el ledger no trae un MD5 valido")
            continue
        actual = md5_of(path)
        checked += 1
        if actual != row["md5"]:
            failures.append(f"{row['path']}: MD5 {actual} no coincide con el sellado {row['md5']}")

    print(f"artefactos sellados verificados: {checked}")
    print(f"filas sin artefacto (pendientes): {pending}")
    for failure in failures:
        print(f"FALLO: {failure}")
    if failures:
        print(f"paper-artifacts-check: {len(failures)} fallo(s)")
        return 1
    print("paper-artifacts-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
