"""Audit the agent harness: mirrored guides, settings, templates, skills, agents, memory chunks.

The harness is what an agent loads before touching code: the root and directory guides
(``AGENTS.md`` / ``CLAUDE.md`` pairs), the skills, the subagent definitions, the subagent prompt
templates, the shared engram chunks and the graph configuration. A drift there is invisible to
lint and tests but changes what every agent believes, so it gets its own gate.

Every check routes its problems through :class:`Audit` and prints its PASS line only when it
found none: a gate that prints PASS next to its own FAIL teaches the reader to skim past both.

Usage:
    python scripts/harness_check.py                # full audit, exit 1 on any failure
    python scripts/harness_check.py --guides-only  # only the AGENTS.md / CLAUDE.md pairs
    python scripts/harness_check.py --sync         # copy AGENTS.md over CLAUDE.md (guides only)

Only the standard library is used so CI can run it without ``poetry install``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories that never hold a guide of ours. Data and caches dominate the tree, so they are
#: pruned before the walk descends into them (the walk is the fallback; git is the fast path).
SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "graphify-out",
        ".git",
        ".nuxt",
        ".output",
        ".dvc",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "data",
        "reports",
        "mlruns",
        "dist",
        "build",
    }
)

#: Every domain the Fase 3 prompt can dispatch needs its template on disk.
REQUIRED_TEMPLATES = ("geo-data", "modeling", "paper", "app", "mlops", "tests")

#: Skills that were retired and must not come back under the same name.
RETIRED_SKILLS = frozenset({"agrosat-azure-h100"})

#: Model ids and hardware the guide declares nonexistent or retired, mapped to what is true
#: instead. Checked only on the always-loaded surface (skill and agent descriptions, ``make``
#: help lines), never on prose that names them precisely in order to forbid them.
RETIRED_TERMS = {
    "Qwen3.5-35B": "el reasoner on-prem real es Qwen3-30B-A3B-Instruct-2507",
    "gemma-4-26b-it": "id inexistente (ADR-011 habla de Gemma 4 26B-A4B, y es FUTURE)",
    "Gemini 3.5 Flash": "el reasoner del sistema es Gemini 2.5 Pro",
    "Gemini 3.1 Pro": "el reasoner del sistema es Gemini 2.5 Pro",
    "AlphaEarth v2.1": "AlphaEarth es SATELLITE_EMBEDDING/V1/ANNUAL, data v1.1",
    "H100": "no hay H100: el protocolo corre en CPU, RTX 4070 o L4 spot",
}

#: Guides that legitimately have no ``AGENTS.md`` sibling because they are pointers, not mirrors.
ORPHAN_CLAUDE_ALLOWED = frozenset({".claude/CLAUDE.md"})

#: Canonical engram project pinned by ``.engram/config.json``; every chunk must belong to it.
ENGRAM_PROJECT = "agrosat-copilotv2"

#: Token shapes that must never travel inside a memory chunk.
SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,})"
)

#: Arrays a chunk can carry. Scanning only ``observations`` left prompts, sessions and mutations
#: — where a pasted token or another project's memory lands just as easily — unaudited.
CHUNK_ARRAYS = ("observations", "prompts", "sessions", "mutations")

#: Fields inside those rows that can carry free text, and therefore a secret.
CHUNK_TEXT_FIELDS = ("content", "payload", "title", "summary", "directory")

#: Harness documents that must not route to a retired skill.
ROUTING_DOCS = (
    "docs/orchestration/auto-invoke.md",
    "docs/orchestration/skill-owners.md",
    "docs/orchestration/skills-catalog.md",
    "docs/orchestration/commands.md",
)


def _normalize(data: bytes) -> bytes:
    """Return ``data`` with CRLF folded to LF so parity survives autocrlf checkouts."""
    return data.replace(b"\r\n", b"\n")


def _tracked_guides() -> list[Path] | None:
    """Return every guide git would carry, or ``None`` when git cannot answer.

    Asking git is both exact and cheap: it skips ``data/`` and the DVC cache without having to
    enumerate them, and it sees untracked-but-not-ignored guides, which is the set that travels
    in a PR.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "*AGENTS.md",
                "*CLAUDE.md",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = [n for n in result.stdout.split("\0") if n]
    return [ROOT / name for name in names]


def _walked_guides() -> list[Path]:
    """Fallback discovery for a tree without git: walk, pruning the heavy directories."""
    found: list[Path] = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in ("AGENTS.md", "CLAUDE.md"):
            if name in files:
                found.append(Path(current) / name)
    return found


def guide_files() -> list[Path]:
    """Return every ``AGENTS.md`` and ``CLAUDE.md`` in the repo, git-fast when possible."""
    return _tracked_guides() or _walked_guides()


def guide_pairs() -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return the ``(AGENTS.md, CLAUDE.md)`` pairs and the orphan CLAUDE.md files.

    Discovering pairs from ``AGENTS.md`` alone made a deleted ``AGENTS.md`` invisible: the
    mirror survived, every non-Claude agent lost its guide, and the gate stayed green.
    """
    files = guide_files()
    agents = sorted({p for p in files if p.name == "AGENTS.md"})
    claudes = {p for p in files if p.name == "CLAUDE.md"}
    pairs = [(a, a.with_name("CLAUDE.md")) for a in agents]
    orphans = sorted(
        c
        for c in claudes
        if not c.with_name("AGENTS.md").exists()
        and c.relative_to(ROOT).as_posix() not in ORPHAN_CLAUDE_ALLOWED
    )
    return pairs, orphans


def _frontmatter(text: str) -> dict[str, str]:
    """Parse the flat YAML frontmatter of a skill or agent file (key: value lines only)."""
    match = re.match(r"^---\r?\n(.*?)\r?\n---", text, flags=re.S)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _read_text(audit: Audit, path: Path) -> str | None:
    """Read a file, turning an unreadable one into a FAIL line instead of a traceback."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        audit.fail(f"{path.relative_to(ROOT).as_posix()} ilegible: {exc}")
        return None


class Audit:
    """Collect PASS / FAIL / WARN lines and the final verdict."""

    def __init__(self) -> None:
        self.failures = 0
        self.lines: list[str] = []

    def ok(self, message: str) -> None:
        self.lines.append(f"PASS  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        self.lines.append(f"FAIL  {message}")

    def warn(self, message: str) -> None:
        self.lines.append(f"WARN  {message}")

    def mark(self) -> int:
        """Return the current failure count, so a check can tell whether it added any."""
        return self.failures

    def ok_if_clean(self, since: int, message: str) -> None:
        """Print the PASS line only when no failure was recorded since ``since``."""
        if self.failures == since:
            self.ok(message)


def check_guides(audit: Audit, sync: bool) -> None:
    """Every AGENTS.md needs a byte-identical CLAUDE.md next to it (CRLF-insensitive)."""
    start = audit.mark()
    pairs, orphans = guide_pairs()
    if not pairs:
        audit.fail("no se encontro ningun AGENTS.md")
        return
    for agents, claude in pairs:
        rel = agents.relative_to(ROOT).as_posix()
        if sync:
            claude.write_bytes(agents.read_bytes())
        if not claude.exists():
            audit.fail(f"{rel}: falta su espejo CLAUDE.md")
            continue
        if _normalize(agents.read_bytes()) != _normalize(claude.read_bytes()):
            audit.fail(f"{rel}: difiere de su espejo CLAUDE.md (make guides-sync)")
    for orphan in orphans:
        rel = orphan.relative_to(ROOT).as_posix()
        audit.fail(f"{rel}: CLAUDE.md sin AGENTS.md hermano (los otros harness pierden la guia)")
    audit.ok_if_clean(start, f"espejos AGENTS.md == CLAUDE.md en {len(pairs)} par(es)")


def check_settings(audit: Audit) -> None:
    """The project settings must parse and carry an allowlist without a blanket ``make``."""
    start = audit.mark()
    path = ROOT / ".claude" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.fail(f".claude/settings.json ilegible: {exc}")
        return
    allow = settings.get("permissions", {}).get("allow")
    if not isinstance(allow, list) or not allow:
        audit.fail(".claude/settings.json: permissions.allow vacio o ausente")
        return
    if "Bash(make:*)" in allow:
        audit.fail(
            ".claude/settings.json: Bash(make:*) auto-aprueba sellado, deploy y dvc push; "
            "enumera los targets seguros"
        )
    audit.ok_if_clean(start, f".claude/settings.json valido ({len(allow)} permisos, sin make:*)")
    if "SessionStart" not in settings.get("hooks", {}):
        audit.warn(".claude/settings.json: sin hook SessionStart (make harness-status manual)")


def check_templates(audit: Audit) -> None:
    """Each dispatchable domain needs its subagent prompt template."""
    start = audit.mark()
    base = ROOT / "docs" / "orchestration" / "subagent-prompts"
    missing = [name for name in REQUIRED_TEMPLATES if not (base / f"{name}.md").exists()]
    if missing:
        audit.fail(f"plantillas de sub-agente ausentes: {', '.join(missing)}")
    for doc in ("prompts-optimizers-fable.md", "auto-invoke.md", "skill-owners.md"):
        if not (base.parent / doc).exists():
            audit.fail(f"docs/orchestration/{doc} ausente")
    audit.ok_if_clean(
        start, f"plantillas y documentos del loop presentes ({len(REQUIRED_TEMPLATES)})"
    )


def check_skills(audit: Audit) -> None:
    """Skill directories must match their frontmatter name and none may be a retired one."""
    start = audit.mark()
    base = ROOT / ".claude" / "skills"
    if not base.is_dir():
        audit.fail(".claude/skills ausente")
        return
    count = 0
    for skill_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if skill_dir.name in RETIRED_SKILLS:
            audit.fail(f"skill retirada presente: {skill_dir.name}")
            continue
        skill = skill_dir / "SKILL.md"
        if not skill.exists():
            audit.fail(f"{skill_dir.name}: sin SKILL.md")
            continue
        text = _read_text(audit, skill)
        if text is None:
            continue
        meta = _frontmatter(text)
        if meta.get("name") != skill_dir.name:
            audit.fail(f"{skill_dir.name}: frontmatter name={meta.get('name')!r}")
        elif not meta.get("description"):
            audit.fail(f"{skill_dir.name}: sin description")
        else:
            count += 1
    audit.ok_if_clean(start, f"skills consistentes: {count}")


def check_agents(audit: Audit) -> None:
    """Subagent files need name, description and tools, with name equal to the filename."""
    start = audit.mark()
    base = ROOT / ".claude" / "agents"
    if not base.is_dir():
        audit.fail(".claude/agents ausente")
        return
    count = 0
    for agent in sorted(base.glob("*.md")):
        text = _read_text(audit, agent)
        if text is None:
            continue
        meta = _frontmatter(text)
        problems = [key for key in ("name", "description", "tools") if not meta.get(key)]
        if problems:
            audit.fail(f"{agent.name}: frontmatter sin {', '.join(problems)}")
        elif meta["name"] != agent.stem:
            audit.fail(f"{agent.name}: name={meta['name']!r} no coincide con el archivo")
        else:
            count += 1
    audit.ok_if_clean(start, f"agents consistentes: {count}")


def check_routing(audit: Audit) -> None:
    """Routing tables must not point to retired skills."""
    start = audit.mark()
    for rel in ROUTING_DOCS:
        path = ROOT / rel
        if not path.exists():
            audit.fail(f"{rel} ausente")
            continue
        text = _read_text(audit, path)
        if text is None:
            continue
        hits = [name for name in RETIRED_SKILLS if name in text]
        if hits:
            audit.fail(f"{rel} enruta a skill retirada: {', '.join(hits)}")
    audit.ok_if_clean(start, "tablas de enrutamiento sin skills retiradas")


def _is_negated(text: str, term: str) -> bool:
    """Say whether ``text`` names ``term`` only to deny it.

    "Sin H100 ni Azure" and "the Azure H100 is gone" are the guide doing its job; flagging them
    would push an author to delete the very sentence that stops the reintroduction.
    """
    escaped = re.escape(term)
    denial = (
        rf"(?:sin|no hay|ya no|no existe|se perdi[oó]|without|retirad\w*)"
        rf"\s+(?:\w+\s+){{0,3}}{escaped}"
        rf"|{escaped}\s+(?:is gone|se perdi[oó]|fue retirad\w*|est[aá] retirad\w*"
        rf"|ya no existe|no existe)"
    )
    return re.search(denial, text, flags=re.I) is not None


def _retired_hits(text: str) -> list[str]:
    """Return the retired ids or hardware a line presents as live."""
    return [term for term in RETIRED_TERMS if term in text and not _is_negated(text, term)]


def check_retired_terms(audit: Audit) -> None:
    """No retired model id or hardware may survive on the surface every session loads.

    Skill and agent descriptions are injected into the skill listing of every session, and
    ``make help`` is the first thing an agent reads about the targets; a dead id there is read
    as live and gets proposed back. The bodies of the guides are exempt on purpose: they name
    these terms precisely in order to forbid them.
    """
    start = audit.mark()
    surfaces: list[tuple[str, str]] = []
    for path in sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md")) + sorted(
        (ROOT / ".claude" / "agents").glob("*.md")
    ):
        text = _read_text(audit, path)
        if text is None:
            continue
        description = _frontmatter(text).get("description", "")
        surfaces.append((path.relative_to(ROOT).as_posix(), description))
    makefile = ROOT / "Makefile"
    if makefile.exists():
        text = _read_text(audit, makefile) or ""
        for number, line in enumerate(text.splitlines(), start=1):
            if "##" in line and not line.startswith("#"):
                surfaces.append((f"Makefile:{number}", line))
    for where, text in surfaces:
        for term in _retired_hits(text):
            audit.fail(f"{where}: nombra {term!r} como vigente; {RETIRED_TERMS[term]}")
    audit.ok_if_clean(
        start, f"superficie siempre cargada sin ids retirados ({len(surfaces)} lineas)"
    )


def _link_docs() -> list[Path]:
    """Documents whose relative links an agent will actually follow."""
    docs: list[Path] = [ROOT / "README.md"]
    docs.extend(guide_files())
    docs.extend(sorted((ROOT / "docs" / "orchestration").glob("*.md")))
    docs.extend(sorted((ROOT / "docs" / "orchestration" / "subagent-prompts").glob("*.md")))
    docs.extend(sorted((ROOT / "docs" / "us-work").glob("*.md")))
    seen: dict[Path, None] = {}
    for doc in docs:
        if doc.exists():
            seen.setdefault(doc.resolve(), None)
    return list(seen)


def check_links(audit: Audit) -> None:
    """Relative links in every harness document must resolve.

    Checking only the root guide left the mirrors, the orchestration docs and the logbook
    READMEs unaudited, which is where an agent that did not enter through AGENTS.md lands.
    """
    start = audit.mark()
    checked = 0
    for doc in _link_docs():
        text = _read_text(audit, doc)
        if text is None:
            continue
        checked += 1
        broken: set[str] = set()
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            if clean and not (doc.parent / clean).exists():
                broken.add(clean)
        if broken:
            rel = doc.relative_to(ROOT).as_posix()
            audit.fail(f"{rel} enlaza rutas inexistentes: {', '.join(sorted(broken))}")
    audit.ok_if_clean(start, f"enlaces relativos resuelven en {checked} documento(s) del harness")


def _chunk_rows(chunk: Path) -> list[dict[str, object]]:
    """Return every row of a gzipped chunk, raising on a malformed one."""
    rows: list[dict[str, object]] = []
    with gzip.open(chunk, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_memory(audit: Audit) -> None:
    """The shared memory store is complete, project-pinned and free of secrets.

    Every chunk listed in the manifest exists and every chunk on disk is listed; the repo pins
    its canonical project in ``.engram/config.json``; no chunk carries another project (an
    ``engram sync --all`` would leak every project of a laptop); and no chunk carries a token.
    All four arrays of a chunk are scanned: a memory of another project, or a pasted token,
    lands in ``prompts`` or ``mutations`` as easily as in ``observations``.
    """
    start = audit.mark()
    engram = ROOT / ".engram"
    manifest = engram / "manifest.json"
    if not manifest.exists():
        audit.warn(".engram/manifest.json ausente (make memory-sync aun no se ha corrido)")
        return
    config = engram / "config.json"
    try:
        pinned = json.loads(config.read_text(encoding="utf-8")).get("project_name")
    except (OSError, ValueError, AttributeError):
        pinned = None
    if pinned != ENGRAM_PROJECT:
        audit.fail(
            f".engram/config.json debe fijar project_name={ENGRAM_PROJECT!r} (tiene {pinned!r})"
        )
    try:
        listed = {
            entry["id"] for entry in json.loads(manifest.read_text(encoding="utf-8"))["chunks"]
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audit.fail(f".engram/manifest.json ilegible: {exc} (scripts/engram_manifest_merge.py)")
        return
    chunks = sorted((engram / "chunks").glob("*.jsonl.gz"))
    on_disk = {p.name.split(".", 1)[0] for p in chunks}
    if listed - on_disk:
        audit.fail(f"chunks listados sin archivo: {', '.join(sorted(listed - on_disk))}")
    if on_disk - listed:
        audit.fail(f"chunks en disco sin listar: {', '.join(sorted(on_disk - listed))}")
    foreign: set[str] = set()
    leaks: list[str] = []
    for chunk in chunks:
        try:
            rows = _chunk_rows(chunk)
        except (OSError, ValueError) as exc:
            audit.fail(f"{chunk.name}: ilegible ({exc})")
            continue
        for row in rows:
            for array in CHUNK_ARRAYS:
                for item in row.get(array, []) or []:
                    if not isinstance(item, dict):
                        continue
                    project = item.get("project")
                    if project is not None and project != ENGRAM_PROJECT:
                        foreign.add(f"{array}:{project}")
                    for field in CHUNK_TEXT_FIELDS:
                        value = item.get(field)
                        if isinstance(value, str) and SECRET_PATTERN.search(value):
                            leaks.append(f"{chunk.name}#{array}:{item.get('id')}")
    if foreign:
        audit.fail(f"chunks con memorias de otros proyectos: {', '.join(sorted(foreign))}")
    if leaks:
        audit.fail(f"posibles secretos en chunks: {', '.join(sorted(set(leaks))[:5])}")
    audit.ok_if_clean(
        start,
        f".engram: {len(listed)} chunk(s) consistentes, solo {ENGRAM_PROJECT}, "
        f"sin tokens en {'/'.join(CHUNK_ARRAYS)}",
    )


def check_graph_config(audit: Audit) -> None:
    """The graph must have its ignore file and stay out of git."""
    start = audit.mark()
    if not (ROOT / ".graphifyignore").exists():
        audit.fail(".graphifyignore ausente")
    gitignore = _read_text(audit, ROOT / ".gitignore")
    if gitignore is None:
        return
    if "graphify-out/" not in gitignore:
        audit.fail(".gitignore no excluye graphify-out/")
    ignored_lines = [line.strip() for line in gitignore.splitlines()]
    if ".engram/" in ignored_lines:
        audit.fail(".gitignore excluye .engram/ entero: los chunks deben viajar en git")
    if ".engram/engram.db" not in ignored_lines:
        audit.fail(".gitignore no excluye .engram/engram.db (la DB de trabajo nunca viaja)")
    audit.ok_if_clean(
        start, "grafo configurado (.graphifyignore, graphify-out/ ignorado, .engram/ trackeado)"
    )


def check_us_work(audit: Audit) -> None:
    """Open logbooks must declare their state, otherwise nobody knows where a US stopped."""
    for log in sorted((ROOT / "docs" / "us-work").glob("us-*.md")):
        text = _read_text(audit, log)
        if text is not None and "**Estado**" not in text:
            audit.warn(f"{log.relative_to(ROOT).as_posix()}: sin linea **Estado**")


def main() -> int:
    """Run the audit and print one line per check.

    ``--sync`` writes the mirrors and reports only on them: making it fall through to the full
    audit meant a successful sync still exited 1 for an unrelated failure elsewhere.

    Returns:
        Process exit code: 0 when every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="copia AGENTS.md sobre CLAUDE.md")
    parser.add_argument("--guides-only", action="store_true", help="solo los espejos")
    args = parser.parse_args()

    audit = Audit()
    check_guides(audit, sync=args.sync)
    if not (args.guides_only or args.sync):
        check_settings(audit)
        check_templates(audit)
        check_skills(audit)
        check_agents(audit)
        check_routing(audit)
        check_retired_terms(audit)
        check_links(audit)
        check_memory(audit)
        check_graph_config(audit)
        check_us_work(audit)

    print("\n".join(audit.lines))
    if audit.failures:
        print(f"harness-check: {audit.failures} fallo(s)")
        return 1
    print("harness-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
