"""Los dos gates de procedencia, probados en negativo y versionados.

Estos tests existen por un motivo concreto: los dos controles pasaron una auditoria externa
que despues los burlo a mano en una copia. Una prueba en negativo hecha una vez en una sesion
no protege nada; solo cuenta si esta en la suite. Cada test de aqui muta el documento y exige
que el gate se rompa.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
PLAN_CHECK = REPO_ROOT / "scripts" / "plan_check.py"
ARTIFACTS_CHECK = REPO_ROOT / "scripts" / "paper_artifacts_check.py"


def _plan_check_module():
    """Import ``scripts/plan_check.py`` as a module."""
    spec = importlib.util.spec_from_file_location("plan_check_bajo_prueba", PLAN_CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _correr_plan_check(plan: Path) -> int:
    """Run the plan gate over a given plan file and return its exit code."""
    return subprocess.run(
        [sys.executable, str(PLAN_CHECK), "--plan", str(plan)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def _correr_artifacts_check(ledger: Path) -> tuple[int, str]:
    """Run the custody gate over a given ledger and return its exit code and output."""
    proc = subprocess.run(
        [sys.executable, str(ARTIFACTS_CHECK), "--ledger", str(ledger)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


@pytest.fixture
def plan() -> Path:
    """The published plan the gate runs over, skipping when the sibling repo is absent."""
    module = _plan_check_module()
    ruta = Path(module.DEFAULT_PLAN)
    if not ruta.exists():
        pytest.skip(f"el plan publicado no esta en disco: {ruta}")
    return ruta


def test_el_plan_publicado_pasa_su_gate(plan: Path) -> None:
    """The published plan is the baseline: if it does not pass, nothing below means anything."""
    assert _correr_plan_check(plan) == 0


@pytest.mark.parametrize("campo", ["t", "role", "ac"])
def test_decir_que_no_hay_dependencias_teniendolas_rompe_el_gate(
    plan: Path, tmp_path: Path, campo: str
) -> None:
    """The claim is caught in ANY field, not in the two the first version happened to read.

    La auditoria burlo la primera version moviendo la frase al titulo, que el gate no miraba.
    Se prueban tres campos distintos porque el defecto no era la frase: era el alcance.
    """
    texto = plan.read_text(encoding="utf-8")
    ancla = '{id:"US-140"'
    assert ancla in texto, "el plan cambio de forma y este test ya no muta lo que cree"
    if campo == "t":
        mutado = texto.replace(
            f'{ancla}, t:"Preregistro', f'{ancla}, t:"Preregistro sin dependencias,', 1
        )
    elif campo == "role":
        mutado = texto.replace(f"{ancla}", f"{ancla}", 1).replace(
            'role:"Como equipo al que le refutaron la hipótesis',
            'role:"Sin dependencias. Como equipo al que le refutaron la hipótesis',
            1,
        )
    else:
        mutado = texto.replace(
            '"El criterio principal se fija AQUÍ',
            '"Sin dependencias. El criterio principal se fija AQUÍ',
            1,
        )
    assert mutado != texto, f"la mutacion del campo {campo} no cambio el fichero"
    copia = tmp_path / "plan.html"
    copia.write_text(mutado, encoding="utf-8")
    assert _correr_plan_check(copia) == 1, (
        f"el gate no detecta la contradiccion cuando vive en el campo {campo}"
    )


def test_el_ledger_vigente_pasa_su_gate() -> None:
    """Baseline for the custody gate."""
    codigo, salida = _correr_artifacts_check(LEDGER)
    assert codigo == 0, salida


def test_un_commit_de_sellado_inventado_rompe_el_gate(tmp_path: Path) -> None:
    """A sealing commit that is not in the history of HEAD is provenance, not decoration."""
    texto = LEDGER.read_text(encoding="utf-8")
    import re

    mutado = re.sub(r"(\*\*Commit de sellado\*\*: `)[0-9a-f]{7,40}(`)", r"\g<1>deadbee\g<2>", texto)
    assert mutado != texto, "no se encontro el commit de sellado en la cabecera"
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert "commit de sellado" in salida


def test_decir_que_un_artefacto_no_esta_en_git_teniendolo_rompe_el_gate(tmp_path: Path) -> None:
    """The custody gate compared bytes for two rounds and never looked at provenance."""
    import re

    texto = LEDGER.read_text(encoding="utf-8")
    seguidos = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.splitlines()
    )
    lineas = texto.splitlines()
    mutada = None
    for i, linea in enumerate(lineas):
        if not linea.startswith("|"):
            continue
        celdas = linea.strip().strip("|").split("|")
        min_celdas = 6
        if len(celdas) < min_celdas:
            continue
        ruta = re.search(r"`([^`]+)`", celdas[1])
        if ruta is None or ruta.group(1) not in seguidos or "SELLADO" not in celdas[5]:
            continue
        celdas[4] = " sin seguimiento en git "
        lineas[i] = "|" + "|".join(celdas) + "|"
        mutada = ruta.group(1)
        break
    assert mutada is not None, "no hay ninguna fila sellada y versionada que mutar"
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert mutada in salida
