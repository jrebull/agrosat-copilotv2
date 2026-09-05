"""Print where the loop stands: branch, graph freshness, local memory and open user stories.

Runs as the Claude Code ``SessionStart`` hook (see ``.claude/settings.json``) and as
``make harness-status``. It never fails the session: every probe degrades to a short note.

Usage:
    python scripts/harness_status.py                # human-readable status, always exit 0
    python scripts/harness_status.py --graph-check  # exit 1 when the graph does not describe HEAD

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_check import guide_pairs  # mismo directorio, solo stdlib

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "graphify-out" / "graph.json"

#: Misma variable que honra `scripts/plan_check.py`; sin ella, este estado y ese gate podian
#: discrepar sobre que cuaderno existe.
PLAN_ENV = "AGROSAT_PLAN_HTML"
PLAN_CANDIDATES = (
    ROOT.parent / "agrosat-micai-site" / "plan.html",
    Path.home() / "Documents" / "agrosat-micai-site" / "plan.html",
)


#: Estados de `_probe`: el binario no esta, o corrio y fallo. Colapsarlos en ``None`` hacia que
#: un engram roto se reportara como "no instalado".
MISSING = "missing"
BROKEN = "broken"


def _probe(args: list[str], timeout: float = 8.0) -> tuple[str, str]:
    """Run a command and say how it went.

    Returns:
        ``("ok", stdout)``, ``(MISSING, "")`` cuando el binario no esta en PATH, o
        ``(BROKEN, primera linea del error)`` cuando corrio y fallo.
    """
    if shutil.which(args[0]) is None:
        return MISSING, ""
    try:
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return BROKEN, str(exc)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return BROKEN, detail[0] if detail else f"exit {result.returncode}"
    return "ok", result.stdout


def _run(args: list[str], timeout: float = 8.0) -> str | None:
    """Return stdout when the command succeeded, ``None`` otherwise (sondas de git)."""
    state, output = _probe(args, timeout)
    return output if state == "ok" else None


def git_line() -> str:
    """Current branch and short HEAD."""
    branch = (_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "?").strip()
    head = (_run(["git", "rev-parse", "--short", "HEAD"]) or "?").strip()
    dirty = (_run(["git", "status", "--porcelain"]) or "").strip()
    suffix = f", {len(dirty.splitlines())} archivo(s) sin commit" if dirty else ""
    return f"rama {branch} @ {head}{suffix}"


def graph_state() -> tuple[str, bool]:
    """Describe the graph and say whether it matches HEAD.

    Returns:
        A status line and ``True`` when the graph was built from the current HEAD.
    """
    if not GRAPH.exists():
        return "grafo: ausente (make graph-update)", False
    try:
        built = json.loads(GRAPH.read_text(encoding="utf-8")).get("built_at_commit") or ""
    except (OSError, ValueError):
        return "grafo: graph.json ilegible (make graph-update)", False
    head = (_run(["git", "rev-parse", "HEAD"]) or "").strip()
    if not built or not head:
        return "grafo: sin built_at_commit (make graph-update)", False
    if built == head:
        return f"grafo: al dia ({head[:7]})", True
    behind = (_run(["git", "rev-list", "--count", f"{built}..HEAD"]) or "").strip()
    if behind.isdigit():
        return f"grafo: {behind} commit(s) atras de HEAD (make graph-update)", False
    return f"grafo: construido en {built[:7]}, HEAD es {head[:7]} (make graph-update)", False


def memory_line() -> str:
    """Report local Engram availability without invoking disabled shared synchronization."""
    state, status = _probe(["engram", "--version"])
    if state == MISSING:
        return "memoria: Engram local no instalado (opcional; ADR-015)"
    if state == BROKEN:
        return f"memoria: Engram local no responde ({status[:80]})"
    version = status.strip().splitlines()[0] if status.strip() else "version desconocida"
    return f"memoria: Engram local disponible ({version}); sync compartido deshabilitado"


def open_stories() -> list[str]:
    """Logbooks in docs/us-work with their declared state."""
    lines: list[str] = []
    for log in sorted((ROOT / "docs" / "us-work").glob("us-*.md")):
        text = log.read_text(encoding="utf-8")
        match = re.search(r"\*\*Estado\*\*:\s*([^\n·]+)", text)
        state = match.group(1).strip() if match else "sin estado"
        lines.append(f"  {log.stem}: {state}")
    return lines


def guides_line() -> str:
    """Quick parity check of every AGENTS.md / CLAUDE.md pair.

    El descubrimiento de pares vive en `harness_check`: tenerlo dos veces significaba dos listas
    de exclusiones que se desincronizan, y un `os.walk` del repo entero (cientos de miles de
    archivos entre `data/` y la cache de DVC) en cada arranque de sesion.
    """
    pairs, orphans = guide_pairs()
    drift = [c.relative_to(ROOT).as_posix() for c in orphans]
    for agents, claude in pairs:
        if not claude.exists():
            drift.append(agents.relative_to(ROOT).as_posix())
            continue
        if agents.read_bytes().replace(b"\r\n", b"\n") != claude.read_bytes().replace(
            b"\r\n", b"\n"
        ):
            drift.append(agents.relative_to(ROOT).as_posix())
    if drift:
        return f"espejos: DIFIEREN {', '.join(sorted(drift))} (make guides-sync)"
    return f"espejos: AGENTS.md == CLAUDE.md en los {len(pairs)} pares"


def plan_line() -> str:
    """Where the published plan can be parsed from, if anywhere."""
    override = os.environ.get(PLAN_ENV)
    if override:
        marca = "" if Path(override).exists() else " (NO EXISTE)"
        return f"plan: {override}{marca} via {PLAN_ENV} (make plan-check)"
    for candidate in PLAN_CANDIDATES:
        if candidate.exists():
            return f"plan: {candidate} (make plan-check)"
    return "plan: cuaderno no clonado junto al repo (git clone jrebull/agrosat-micai-site)"


def main() -> int:
    """Print the status block.

    Returns:
        0 always, except ``--graph-check`` which returns 1 when the graph is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-check", action="store_true", help="solo el grafo, exit 1 si rancio"
    )
    args = parser.parse_args()

    graph_msg, fresh = graph_state()
    if args.graph_check:
        print(graph_msg)
        return 0 if fresh else 1

    print("harness-status")
    print(f"  {git_line()}")
    print(f"  {graph_msg}")
    print(f"  {memory_line()}")
    print(f"  {guides_line()}")
    print(f"  {plan_line()}")
    stories = open_stories()
    if stories:
        print("  US en vuelo (docs/us-work):")
        print("\n".join(stories))
    else:
        print("  US en vuelo: ninguna (docs/us-work vacio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
