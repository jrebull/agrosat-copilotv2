"""Pruebas del gate que vigila la frontera entre la CI y la suite local.

El gate nacio de un fallo silencioso: ``tests/ml/test_correlations.py`` quedo en rojo al
traducir los mensajes de error al ingles y siguio asi porque el paso de pruebas de la CI
enumera directorios y ese fichero no cuelga de ninguno. Un control asi solo vale si se le ha
visto fallar, y aqui se le ve fallar en los tres casos que importan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "ci_test_coverage_check.py"


def _correr() -> tuple[int, str]:
    proceso = subprocess.run(
        [sys.executable, str(GATE)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    return proceso.returncode, proceso.stdout + proceso.stderr


def test_el_repositorio_pasa_el_gate() -> None:
    """Hoy cada fichero de prueba o lo corre la CI o esta declarado con su motivo."""
    codigo, salida = _correr()
    assert codigo == 0, salida
    assert "ci-test-coverage-check: OK" in salida


def test_un_arbol_de_pruebas_nuevo_rompe_el_gate(tmp_path: Path) -> None:
    """Un directorio de pruebas que nadie declaro no puede pasar inadvertido."""
    nuevo = REPO_ROOT / "tests" / "_arbol_de_prueba_del_gate"
    nuevo.mkdir()
    (nuevo / "test_x.py").write_text("def test_x() -> None:\n    assert True\n", encoding="utf-8")
    try:
        codigo, salida = _correr()
        assert codigo == 1, salida
        assert "_arbol_de_prueba_del_gate/test_x.py" in salida
    finally:
        (nuevo / "test_x.py").unlink()
        nuevo.rmdir()
    assert _correr()[0] == 0


def test_un_subdirectorio_no_hereda_el_motivo_de_su_padre() -> None:
    """``tests/ml/train/*.py`` esta declarado; ``tests/ml/train/sub/`` no lo esta por herencia.

    Los patrones son de ``PurePosixPath.match``, donde ``*`` no cruza separadores. Si heredaran,
    un subarbol nuevo se colaria con el motivo escrito para otra cosa.
    """
    sub = REPO_ROOT / "tests" / "ml" / "train" / "_sub_de_prueba_del_gate"
    sub.mkdir()
    (sub / "test_y.py").write_text("def test_y() -> None:\n    assert True\n", encoding="utf-8")
    try:
        codigo, salida = _correr()
        assert codigo == 1, salida
        assert "_sub_de_prueba_del_gate/test_y.py" in salida
    finally:
        (sub / "test_y.py").unlink()
        sub.rmdir()
    assert _correr()[0] == 0


def test_un_fichero_en_un_arbol_que_la_ci_ya_corre_pasa() -> None:
    """Anadir una prueba donde la CI ya mira no obliga a tocar el inventario."""
    nuevo = REPO_ROOT / "tests" / "scripts" / "test_zz_de_prueba_del_gate.py"
    nuevo.write_text("def test_z() -> None:\n    assert True\n", encoding="utf-8")
    try:
        codigo, salida = _correr()
        assert codigo == 0, salida
    finally:
        nuevo.unlink()


def test_una_declaracion_huerfana_rompe_el_gate() -> None:
    """Si se borra un arbol y nadie quita su motivo, el inventario miente y el gate lo dice."""
    from scripts.ci_test_coverage_check import SUITE_LOCAL, revisar

    SUITE_LOCAL["tests/arbol_que_no_existe/*.py"] = "motivo inventado"
    try:
        fallos, _ = revisar()
        assert any("tests/arbol_que_no_existe/*.py" in f for f in fallos), fallos
    finally:
        del SUITE_LOCAL["tests/arbol_que_no_existe/*.py"]
    assert revisar()[0] == []


def test_reconoce_las_invocaciones_de_pytest_del_workflow() -> None:
    """Las rutas se leen del workflow real, con su ``working-directory`` aplicado."""
    from scripts.ci_test_coverage_check import WORKFLOW, prefijos_de_ci

    prefijos = prefijos_de_ci(WORKFLOW)
    assert "backend/tests/unit" in prefijos, prefijos
    assert "tests/scripts" in prefijos, prefijos
    assert "tests/ml/report" in prefijos, prefijos


def test_un_workflow_sin_pytest_no_se_declara_verde(tmp_path: Path) -> None:
    """Si el paso de pruebas desaparece, el gate falla en vez de dar por cubierto todo.

    Es el modo de fallo peligroso: un workflow que ya no corre nada haria que ningun fichero
    estuviera "sin cubrir" solo porque no hay con que compararlo.
    """
    import scripts.ci_test_coverage_check as gate

    vacio = tmp_path / "ci.yml"
    vacio.write_text("name: CI\njobs: {}\n", encoding="utf-8")
    original = gate.WORKFLOW
    gate.WORKFLOW = vacio
    try:
        fallos, _ = gate.revisar()
        assert fallos == ["FALLO: no se reconocio ninguna invocacion de pytest en el workflow"]
    finally:
        gate.WORKFLOW = original


@pytest.mark.parametrize(
    "arbol",
    ["tests/ml/eval", "tests/ml/analysis", "tests/ml/eval/oof"],
)
def test_la_superficie_del_articulo_la_corre_la_ci(arbol: str) -> None:
    """Los arboles que sostienen el articulo -frontera, custodia, inventario OOF- corren en la CI.

    Estuvieron declarados como suite local hasta comprobar que pasaban en un checkout SIN los
    blobs de DVC. Pasan: un git worktree limpio dio 597 pruebas verdes, y el unico fallo era una
    prueba que dependia de un checkpoint versionado y ya no depende.
    """
    from scripts.ci_test_coverage_check import SUITE_LOCAL, WORKFLOW, prefijos_de_ci

    prefijos = prefijos_de_ci(WORKFLOW)
    assert any(arbol == p or arbol.startswith(f"{p}/") for p in prefijos), prefijos
    assert f"{arbol}/*.py" not in SUITE_LOCAL
