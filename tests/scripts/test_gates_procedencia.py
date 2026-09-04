"""Los dos gates de procedencia, probados en negativo y versionados.

Estos tests existen por un motivo concreto: los dos controles pasaron una auditoria externa
que despues los burlo a mano en una copia. Una prueba en negativo hecha una vez en una sesion
no protege nada; solo cuenta si esta en la suite. Cada test de aqui muta el documento y exige
que el gate se rompa.
"""

from __future__ import annotations

import importlib.util
import math
import re
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
    # Las mutaciones se anclan en la ESTRUCTURA del objeto, no en frases del plan. Anclarlas en
    # una frase fosiliza esa redaccion: una auditoria encontro este test conservando como ancla
    # justo la frase que el plan acababa de retirar.
    inicio = texto.index(ancla)
    if campo == "t":
        marca = ', t:"'
        pos = texto.index(marca, inicio) + len(marca)
    elif campo == "role":
        marca = ' role:"'
        pos = texto.index(marca, inicio) + len(marca)
    else:
        marca = ' ac:["'
        pos = texto.index(marca, inicio) + len(marca)
    mutado = texto[:pos] + "Sin dependencias. " + texto[pos:]
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


# --------------------------------------------------------------------------------------
# Ronda 4: los caminos de al lado que la auditoria encontro en los dos gates.
# --------------------------------------------------------------------------------------


def test_el_gate_de_dependencias_tambien_mira_dentro_de_un_dict_anidado(
    plan: Path, tmp_path: Path
) -> None:
    """Nested fields are fields: the audit hid the claim in ``meta:{nota:...}`` and got a green."""
    texto = plan.read_text(encoding="utf-8")
    ancla = '{id:"US-140"'
    assert ancla in texto
    mutado = texto.replace(ancla, f'{ancla}, meta:{{nota:"Sin dependencias"}}', 1)
    assert mutado != texto
    copia = tmp_path / "plan.html"
    copia.write_text(mutado, encoding="utf-8")
    assert _correr_plan_check(copia) == 1, "la afirmacion escondida en un dict anidado se cuela"


def test_un_commit_de_fila_inventado_rompe_el_gate(tmp_path: Path) -> None:
    """A row's commit has to exist. The audit replaced one with `deadbee` and the gate said OK."""
    import re

    texto = LEDGER.read_text(encoding="utf-8")
    mutado, n = re.subn(
        r"\| `[0-9a-f]{7,40}` \| SELLADO \|", "| `deadbee` | SELLADO |", texto, count=1
    )
    assert n == 1, "no se encontro ninguna fila sellada con commit que mutar"
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert "deadbee" in salida


def test_un_sello_anterior_a_sus_filas_rompe_el_gate(tmp_path: Path) -> None:
    """Ancestry alone accepted the root commit, 467 back. A seal cannot predate what it seals."""
    import re

    raiz = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()[0][:7]
    texto = LEDGER.read_text(encoding="utf-8")
    mutado, n = re.subn(
        r"(\*\*Commit de sellado\*\*: `)[0-9a-f]{7,40}(`)", rf"\g<1>{raiz}\g<2>", texto, count=1
    )
    assert n == 1
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert "anterior a" in salida


def test_un_md5_que_ese_commit_nunca_produjo_rompe_el_gate(tmp_path: Path) -> None:
    """Today's bytes matching is not provenance: the commit has to have produced them.

    Se muta el MD5 de una fila **y el archivo en disco**, para que la comprobacion de bytes de
    hoy pase y solo pueda fallar la de procedencia. Sin esto el test no distinguiria los dos
    controles.
    """
    import re
    import shutil

    texto = LEDGER.read_text(encoding="utf-8")
    fila = re.search(
        r"^\| [^|]+ \| `(reports/paper_micai/prereg/[^`]+)` \| `([0-9a-f]{32})` \| (\d+) \|",
        texto,
        re.M,
    )
    assert fila is not None, "no hay ninguna fila de preregistro que mutar"
    ruta, md5_real, _ = fila.group(1), fila.group(2), fila.group(3)
    origen = REPO_ROOT / ruta
    respaldo = tmp_path / "respaldo.bin"
    shutil.copy2(origen, respaldo)
    try:
        contenido = origen.read_bytes() + b"\n"
        origen.write_bytes(contenido)
        import hashlib

        nuevo_md5 = hashlib.md5(contenido).hexdigest()
        mutado = texto.replace(f"`{md5_real}`", f"`{nuevo_md5}`", 1).replace(
            f"| {fila.group(3)} |", f"| {len(contenido)} |", 1
        )
        copia = tmp_path / "ARTIFACTS.md"
        copia.write_text(mutado, encoding="utf-8")
        codigo, salida = _correr_artifacts_check(copia)
        assert codigo == 1, salida
        assert "el ledger registra" in salida
    finally:
        shutil.copy2(respaldo, origen)


# --------------------------------------------------------------------------------------
# Ronda 5: los tres caminos de al lado que quedaban en los gates.
# --------------------------------------------------------------------------------------


def test_una_fila_sellada_sin_commit_ni_excusa_rompe_el_gate(tmp_path: Path) -> None:
    """The provenance check only ran when it FOUND a SHA, so a dash skipped it entirely."""
    import re

    texto = LEDGER.read_text(encoding="utf-8")
    mutado, n = re.subn(
        r"\| `[0-9a-f]{7,40}` \| (SELLADO|OBSOLETO) \|", r"| — | \1 |", texto, count=1
    )
    assert n == 1, "no se encontro ninguna fila con commit que mutar"
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert "no declara ni un commit" in salida


def test_la_contradiccion_dentro_del_propio_campo_dep_rompe_el_gate(
    plan: Path, tmp_path: Path
) -> None:
    """``dep`` was excluded from the scan, so the claim could live inside the dependency list."""
    texto = plan.read_text(encoding="utf-8")
    ancla = '{id:"US-140"'
    inicio = texto.index(ancla)
    marca = ' dep:"'
    pos = texto.index(marca, inicio) + len(marca)
    mutado = texto[:pos] + "Sin dependencias; " + texto[pos:]
    assert mutado != texto
    copia = tmp_path / "plan.html"
    copia.write_text(mutado, encoding="utf-8")
    assert _correr_plan_check(copia) == 1, "la afirmacion dentro del propio campo dep se cuela"


def _correr_obsoletos_check(ledger: Path) -> tuple[int, str]:
    """Run the publication gate over a given ledger."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper_obsoletos_check.py"),
            "--ledger",
            str(ledger),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_el_gate_de_publicacion_pasa_con_los_consumidores_marcados() -> None:
    """Baseline: every declared consumer carries its quarantine banner."""
    codigo, salida = _correr_obsoletos_check(LEDGER)
    assert codigo == 0, salida


def test_quitar_la_cuarentena_de_un_consumidor_rompe_el_gate(tmp_path: Path) -> None:
    """The OBSOLETO state was announced and enforced nothing; now it blocks publication.

    Se muta el DOCUMENTO, no el ledger, y se restaura despues: es la unica forma de comprobar que
    el gate mira lo que dice mirar.
    """
    import shutil

    doc = REPO_ROOT / "docs" / "paper" / "fase3-hallazgos.md"
    respaldo = tmp_path / "respaldo.md"
    shutil.copy2(doc, respaldo)
    try:
        texto = doc.read_text(encoding="utf-8")
        assert "> **CUARENTENA**" in texto
        doc.write_text(texto.replace("> **CUARENTENA**", "> Nota", 1), encoding="utf-8")
        codigo, salida = _correr_obsoletos_check(LEDGER)
        assert codigo == 1, salida
        assert "fase3-hallazgos.md" in salida
    finally:
        shutil.copy2(respaldo, doc)


def test_copiar_una_cifra_obsoleta_a_otro_documento_rompe_el_gate(tmp_path: Path) -> None:
    """Prose copies numbers, not paths. Watching only the path was the hole.

    Una auditoria copio 0,0326 al preregistro sin nombrar el artefacto y el gate paso. Ahora las
    cifras distintivas —cuatro decimales o mas— se extraen de los propios artefactos obsoletos.
    """
    codigo, salida = _correr_obsoletos_check(LEDGER)
    assert codigo == 0, salida
    cifra = re.search(r"p\. ej\. (\d+,\d{4})", salida)
    if cifra is None:
        # El gate esta verde, asi que se toma una cifra de un artefacto obsoleto directamente.
        import json

        datos = json.loads(
            (REPO_ROOT / "reports/paper_micai/bloques/bloques.json").read_text(encoding="utf-8")
        )
        crudos: list[float] = []

        def recoger(v: object) -> None:
            if isinstance(v, bool):
                return
            if isinstance(v, int | float):
                crudos.append(float(v))
            elif isinstance(v, dict):
                for x in v.values():
                    recoger(x)
            elif isinstance(v, list):
                for x in v:
                    recoger(x)

        recoger(datos)
        candidatas = [
            f"{abs(x):.10f}".rstrip("0") for x in crudos if abs(x) < 1000 and not math.isnan(x)
        ]
        largas = [c for c in candidatas if len(c.partition(".")[2]) >= 4]
        assert largas, "el artefacto obsoleto no trae ninguna cifra distintiva"
        entero, _, dec = largas[0].partition(".")
        texto_cifra = f"{entero},{dec[:4]}"
    else:
        texto_cifra = cifra.group(1)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "documento-nuevo.md").write_text(
        f"# Un documento activo\n\nEl efecto medido fue {texto_cifra} y lo damos por bueno.\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper_obsoletos_check.py"),
            "--ledger",
            str(LEDGER),
            "--docs",
            str(docs),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout
    assert "reproduce" in proc.stdout
