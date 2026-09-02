"""Validate that every LaTeX citation key in the paper resolves in refs.bib.

This is a lightweight, LaTeX-free guard for the EPIC 11 manuscript
(``paper/``). It scans every ``\\cite{...}`` (and variants) across
``paper/main.tex`` and ``paper/sections/*.tex``, collects the BibTeX keys
defined in ``paper/bib/refs.bib``, and reports any citation that lacks a
matching entry (and, informationally, any unused entry).

Run via ``make paper-cite-check`` or directly:

    python scripts/paper_cite_check.py

Exit code is non-zero when an unresolved citation is found, so it can be
wired into CI alongside ``make check``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Citation commands whose mandatory argument is a comma-separated key list.
_CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citeauthor|citeyear|autocite|textcite)"
    r"\s*(?:\[[^\]]*\])*\{([^}]*)\}"
)
# BibTeX entry header: @type{key,
_BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
# A LaTeX comment starts at an unescaped '%' and runs to end of line.
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")

_PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"
_BIB_PATH = _PAPER_DIR / "bib" / "refs.bib"


def _strip_comments(text: str) -> str:
    """Remove LaTeX line comments so commented-out cites are ignored.

    Args:
        text: Raw LaTeX source.

    Returns:
        The source with end-of-line comments removed.
    """
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def collect_cited_keys(tex_files: list[Path]) -> dict[str, list[Path]]:
    """Collect every citation key used across the given LaTeX files.

    Args:
        tex_files: LaTeX source files to scan.

    Returns:
        Mapping from citation key to the files in which it appears.
    """
    used: dict[str, list[Path]] = {}
    for path in tex_files:
        source = _strip_comments(path.read_text(encoding="utf-8"))
        for match in _CITE_RE.finditer(source):
            for raw_key in match.group(1).split(","):
                key = raw_key.strip()
                if key:
                    used.setdefault(key, []).append(path)
    return used


def collect_defined_keys(bib_path: Path) -> set[str]:
    """Collect every BibTeX key defined in the bibliography file.

    Args:
        bib_path: Path to the ``.bib`` file.

    Returns:
        Set of defined BibTeX keys.
    """
    source = bib_path.read_text(encoding="utf-8")
    return {match.group(1) for match in _BIB_KEY_RE.finditer(source)}


def main() -> int:
    """Run the citation check and return a process exit code.

    Returns:
        ``0`` if all citations resolve, ``1`` otherwise.
    """
    if not _BIB_PATH.exists():
        print(f"ERROR: bibliography not found: {_BIB_PATH}")
        return 1

    tex_files = sorted(_PAPER_DIR.glob("*.tex")) + sorted((_PAPER_DIR / "sections").glob("*.tex"))
    if not tex_files:
        print(f"ERROR: no .tex files found under {_PAPER_DIR}")
        return 1

    used = collect_cited_keys(tex_files)
    defined = collect_defined_keys(_BIB_PATH)

    missing = sorted(key for key in used if key not in defined)
    unused = sorted(defined - set(used))

    print(
        f"Scanned {len(tex_files)} LaTeX files; {len(used)} cited keys; {len(defined)} bib entries."
    )

    if unused:
        print("INFO: bib entries not yet cited (allowed): " + ", ".join(unused))

    if missing:
        print("ERROR: citations without a refs.bib entry:")
        for key in missing:
            where = ", ".join(str(p.name) for p in used[key])
            print(f"  - {key}  (in {where})")
        return 1

    print("OK: every \\cite{} resolves in paper/bib/refs.bib.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
