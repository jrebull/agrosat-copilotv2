#!/usr/bin/env python3
"""Fail when a test file exists that neither CI runs nor the local inventory claims.

``tests/ml/test_correlations.py`` estuvo en rojo desde ``c646263``, que tradujo los mensajes
de error al ingles y dejo dos aserciones buscando el texto en castellano. Nadie lo vio porque
el paso de pruebas de la CI enumera **directorios** —``tests/ml/features``, ``tests/ml/models``,
...— y los ficheros sueltos de ``tests/ml/`` no cuelgan de ninguno. Una suite verde que no
incluye un fichero no dice nada de ese fichero.

Que la CI corra un subconjunto es una decision del equipo (ver el comentario de junio de 2026
en ``ci.yml``): la suite completa, con testcontainers y datos DVC, tarda demasiado. Este gate
no la discute. Lo que impide es que la frontera entre "lo corre la CI" y "se corre en local"
se mueva sin que nadie lo escriba: cada arbol que la CI no toca aparece abajo con su motivo, y
un arbol nuevo —o uno que dejo de existir— rompe el gate.

Solo stdlib: si necesitara instalacion la tentacion seria sacarlo del gate barato, que es
justamente donde los controles se quedan "solo en local".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TEST_ROOTS = ("tests", "backend/tests")

# Arboles que la CI no corre, con el motivo. Los patrones son de ``PurePosixPath.match``, donde
# ``*`` no cruza separadores: ``dir/*.py`` cubre los hijos directos y NO los subdirectorios, para
# que un subarbol nuevo tenga que declararse en vez de heredar el motivo de su padre.
SUITE_LOCAL: dict[str, str] = {
    "backend/tests/integration/*.py": "testcontainers: levantan Postgres y TiTiler reales",
    "tests/ml/features/test_persist_features.py": "testcontainers, excluido con --ignore",
    "tests/app/*.py": "dashboard: requiere los parquet materializados",
    "tests/dagster/*.py": "assets con DVC y MLflow contra el server local",
    "tests/db/*.py": "migraciones contra Postgres real",
    "tests/ml/*.py": "EDA y cargadores: requieren PASTIS y BreizhCrops en disco",
    "tests/ml/data/*.py": "constructores de dataset sobre rasters en disco",
    "tests/ml/ensemble/*.py": "ensambles sobre OOF materializados",
    "tests/ml/extractors/*.py": "extractores con pesos frozen descargados",
    "tests/ml/farslip/*.py": "destilacion FarSLIP con checkpoints en disco",
    "tests/ml/ingest/*.py": "ingesta con credenciales y datos externos",
    "tests/ml/monitoring/*.py": "drift sobre features materializadas",
    "tests/ml/train/*.py": "entrenamientos, aun los cortos, con datos en disco",
    "tests/ml/transfer/*.py": "transferencia a Italia sobre checkpoints en disco",
    "tests/ml/tune/*.py": "tuning con datos en disco",
}


def invocaciones_pytest(workflow: Path) -> list[tuple[str, list[str]]]:
    """Directorio de trabajo y rutas de cada invocacion de pytest del workflow.

    Args:
        workflow: Fichero del workflow de CI.

    Returns:
        Pares ``(working_directory, rutas)``, una entrada por invocacion.
    """
    lineas = workflow.read_text(encoding="utf-8").splitlines()
    invocaciones: list[tuple[str, list[str]]] = []
    trabajo = "."
    indice = 0
    while indice < len(lineas):
        linea = lineas[indice]
        if re.match(r"^\s*-\s+name:", linea):
            trabajo = "."
        directorio = re.match(r"\s*working-directory:\s*(\S+)\s*$", linea)
        if directorio is not None:
            trabajo = directorio.group(1)
        if "pytest" in linea and not linea.lstrip().startswith("#"):
            completa = linea
            while completa.rstrip().endswith("\\") and indice + 1 < len(lineas):
                indice += 1
                completa = completa.rstrip().rstrip("\\") + " " + lineas[indice]
            rutas = [
                token
                for token in completa.split()
                if not token.startswith("-") and "/" in token and not token.endswith(":")
            ]
            if rutas:
                invocaciones.append((trabajo, rutas))
        indice += 1
    return invocaciones


def prefijos_de_ci(workflow: Path) -> set[str]:
    """Rutas que la CI corre, relativas a la raiz del repositorio.

    Args:
        workflow: Fichero del workflow de CI.

    Returns:
        Prefijos de ruta.
    """
    prefijos: set[str] = set()
    for trabajo, rutas in invocaciones_pytest(workflow):
        base = "" if trabajo == "." else f"{trabajo.rstrip('/')}/"
        prefijos.update(f"{base}{ruta.rstrip('/')}" for ruta in rutas)
    return prefijos


def ignorados_de_ci(workflow: Path) -> set[str]:
    """Ficheros que la CI excluye con ``--ignore``."""
    contenido = workflow.read_text(encoding="utf-8")
    return {ignorado.rstrip("/") for ignorado in re.findall(r"--ignore=(\S+)", contenido)}


def ficheros_de_prueba() -> list[str]:
    """Todos los ficheros de prueba del repositorio, relativos a la raiz."""
    encontrados: list[str] = []
    for raiz in TEST_ROOTS:
        base = REPO_ROOT / raiz
        if not base.exists():
            continue
        encontrados.extend(
            str(p.relative_to(REPO_ROOT))
            for p in sorted(base.rglob("test_*.py"))
            if "__pycache__" not in p.parts
        )
    return encontrados


def _cubierto(fichero: str, prefijos: set[str]) -> bool:
    return any(fichero == prefijo or fichero.startswith(f"{prefijo}/") for prefijo in prefijos)


def revisar() -> tuple[list[str], dict[str, int]]:
    """Fallos del gate y el reparto de ficheros entre CI y suite local.

    Returns:
        ``(fallos, {"ci": n, "local": m})``.
    """
    if not WORKFLOW.exists():
        return [f"FALLO: no existe {WORKFLOW.relative_to(REPO_ROOT)}"], {}

    prefijos = prefijos_de_ci(WORKFLOW)
    if not prefijos:
        return ["FALLO: no se reconocio ninguna invocacion de pytest en el workflow"], {}

    ignorados = ignorados_de_ci(WORKFLOW)
    fallos: list[str] = []
    usados: set[str] = set()
    reparto = {"ci": 0, "local": 0}

    for fichero in ficheros_de_prueba():
        candidatos = [p for p in SUITE_LOCAL if fichero == p or PurePosixPath(fichero).match(p)]
        patron = max(candidatos, key=len) if candidatos else None
        if patron is not None:
            usados.add(patron)
            reparto["local"] += 1
            continue
        if fichero in ignorados:
            fallos.append(
                f"FALLO: {fichero} lo ignora la CI con --ignore pero no esta en SUITE_LOCAL "
                "con su motivo"
            )
            continue
        if _cubierto(fichero, prefijos):
            reparto["ci"] += 1
            continue
        fallos.append(
            f"FALLO: {fichero} no lo corre ninguna invocacion de la CI; anadelo al paso de "
            "pruebas o declara su arbol en SUITE_LOCAL con el motivo"
        )

    for patron in sorted(set(SUITE_LOCAL) - usados):
        fallos.append(
            f"FALLO: SUITE_LOCAL declara {patron}, que ya no corresponde a ningun fichero"
        )
    return fallos, reparto


def main() -> int:
    """Punto de entrada del gate."""
    fallos, reparto = revisar()
    for fallo in fallos:
        print(fallo)
    if fallos:
        return 1
    print(
        f"ci-test-coverage-check: OK ({reparto['ci']} ficheros en la CI, "
        f"{reparto['local']} declarados como suite local)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
