"""Comprobacion mecanica del plan por epicas: dependencias, estados y camino critico.

Un plan con una dependencia que apunta a una historia inexistente, o con una historia dada por
hecha cuyo prerrequisito sigue sin empezar, es un plan que miente sin que nadie lo note. Esto lo
detecta antes de arrancar.

El plan vive como un array de JavaScript dentro de `plan.html` del cuaderno publico. Se parsea de
ahi para que no haya una segunda copia que se desincronice.

Uso:
    poetry run python scripts/plan_check.py
    poetry run python scripts/plan_check.py --plan /ruta/a/plan.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_PLAN = Path.home() / "Documents" / "agrosat-micai-site" / "plan.html"

#: Estados que significan que la historia ya no esta pendiente.
AVANZADOS: frozenset[str] = frozenset({"ok", "wip"})

#: Estado que significa pendiente de empezar. `dead` no esta aqui: una historia retirada no es
#: un prerrequisito sin hacer, es trabajo que se decidio no hacer, y confundirlos hace que el
#: grafo mienta sobre lo que bloquea a que.
PENDIENTES: frozenset[str] = frozenset({"idle", "stop"})

#: Estado de una historia retirada por una auditoria. No bloquea a nadie.
RETIRADAS: frozenset[str] = frozenset({"dead"})


def _quote_keys(js: str) -> str:
    """Quote the unquoted object keys of a JavaScript literal, leaving strings alone.

    A regex over the whole text is not enough: an acceptance criterion that contains
    ``, fechada:`` gets rewritten into a key and the parse dies. This walks the text and only
    rewrites outside string literals, which is the difference between a checker that works and
    one that fails on its own input.

    Args:
        js: The JavaScript array literal.

    Returns:
        The same text with object keys quoted, valid as JSON.
    """
    out: list[str] = []
    i, n, in_str = 0, len(js), False
    while i < n:
        ch = js[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(js[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        match = re.match(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", js[i:])
        if match:
            out.append(f'{match.group(1)}"{match.group(2)}"{match.group(3)}')
            i += match.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract(plan: Path) -> list[dict[str, Any]]:
    """Parse the EPICS array out of the published plan.

    Args:
        plan: Path to ``plan.html``.

    Returns:
        The epics as dictionaries.

    Raises:
        SystemExit: if the array cannot be located or parsed.
    """
    text = plan.read_text(encoding="utf-8")
    start = text.index("var EPICS = [")
    depth, i = 0, text.index("[", start)
    for j in range(i, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                raw = text[i : j + 1]
                break
    else:  # pragma: no cover - solo si el fichero esta truncado
        raise SystemExit("no encuentro el cierre del array EPICS")

    raw = _quote_keys(raw)
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"el array EPICS no es JSON valido tras normalizar: {exc}") from exc


def _stories(epics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index every story by id, recording its epic.

    Args:
        epics: Parsed epics.

    Returns:
        Story id to story, with an added ``epic`` key.
    """
    out: dict[str, dict[str, Any]] = {}
    for e in epics:
        for u in e["us"]:
            if u["id"] in out:
                logger.error("historia_duplicada", id=u["id"])
            out[u["id"]] = {**u, "epic": e["id"]}
    return out


def _deps(story: dict[str, Any]) -> list[str]:
    """Split a dependency field into story ids.

    Args:
        story: One story.

    Returns:
        The story ids it depends on; empty when it depends on nothing.
    """
    raw = str(story.get("dep") or "").strip()
    if raw in {"", "—", "-"}:
        return []
    return [d.strip() for d in re.split(r"[,;y]| y ", raw) if d.strip().startswith("US-")]


def _critical_path(stories: dict[str, dict[str, Any]]) -> tuple[int, list[str]]:
    """Longest chain of story points through the dependency graph.

    Args:
        stories: Indexed stories.

    Returns:
        Total points along the longest chain and the chain itself.
    """
    memo: dict[str, tuple[int, list[str]]] = {}

    def walk(sid: str, seen: frozenset[str]) -> tuple[int, list[str]]:
        if sid in memo:
            return memo[sid]
        if sid in seen:
            return 0, []
        best_sp, best_path = 0, []
        for dep in _deps(stories[sid]):
            if dep not in stories:
                continue
            sp, path = walk(dep, seen | {sid})
            if sp > best_sp:
                best_sp, best_path = sp, path
        result = (best_sp + int(stories[sid]["sp"]), [*best_path, sid])
        memo[sid] = result
        return result

    return max((walk(s, frozenset()) for s in stories), key=lambda r: r[0])


def main() -> int:
    """Run every structural check over the published plan.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()

    epics = _extract(args.plan)
    stories = _stories(epics)
    fallos = 0

    # 1. Dependencias que apuntan a nada.
    for sid, st in stories.items():
        for dep in _deps(st):
            if dep not in stories:
                logger.error("dependencia_inexistente", historia=sid, depende_de=dep)
                fallos += 1

    # 2. Historias avanzadas cuyo prerrequisito sigue pendiente.
    for sid, st in stories.items():
        if st["st"] not in AVANZADOS:
            continue
        for dep in _deps(st):
            if dep in stories and stories[dep]["st"] in PENDIENTES:
                logger.error(
                    "prerrequisito_sin_empezar",
                    historia=sid,
                    estado=st["st"],
                    depende_de=dep,
                    estado_dep=stories[dep]["st"],
                )
                fallos += 1

    # 3. Ciclos.
    def cycle_from(sid: str, stack: tuple[str, ...]) -> list[str] | None:
        if sid in stack:
            return [*stack[stack.index(sid) :], sid]
        for dep in _deps(stories[sid]):
            if dep in stories and (found := cycle_from(dep, (*stack, sid))):
                return found
        return None

    for sid in stories:
        if ciclo := cycle_from(sid, ()):
            logger.error("ciclo", camino=" -> ".join(ciclo))
            fallos += 1
            break

    # 4. Criterios de aceptacion vacios o historias sin salida declarada.
    for sid, st in stories.items():
        if not st.get("ac"):
            logger.error("sin_criterios_de_aceptacion", historia=sid)
            fallos += 1
        if not str(st.get("out") or "").strip():
            logger.error("sin_artefacto_de_salida", historia=sid)
            fallos += 1

    # 5. Resumen.
    sp_total = sum(int(s["sp"]) for s in stories.values())
    por_estado: dict[str, int] = {}
    for s in stories.values():
        por_estado[s["st"]] = por_estado.get(s["st"], 0) + 1
    sp_camino, camino = _critical_path(stories)
    pendientes = {k: v for k, v in stories.items() if v["st"] in PENDIENTES | {"wip"}}
    sp_pend, camino_pend = _critical_path(pendientes) if pendientes else (0, [])
    sin_dep = [s for s in stories if not _deps(stories[s])]

    logger.info(
        "resumen",
        epicas=len(epics),
        historias=len(stories),
        sp_total=sp_total,
        por_estado=por_estado,
        sp_camino_critico=sp_camino,
        sp_pendiente=sum(int(v["sp"]) for v in pendientes.values()),
        sp_camino_critico_pendiente=sp_pend,
        historias_sin_dependencia=len(sin_dep),
    )
    logger.info("camino_critico_historico", camino=" -> ".join(camino))
    logger.info("camino_critico_pendiente", camino=" -> ".join(camino_pend))

    if fallos:
        print(f"plan-check: {fallos} fallo(s)")
        return 1
    print("plan-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
