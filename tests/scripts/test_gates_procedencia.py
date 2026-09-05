"""Los dos gates de procedencia, probados en negativo y versionados.

Estos tests existen por un motivo concreto: los dos controles pasaron una auditoria externa
que despues los burlo a mano en una copia. Una prueba en negativo hecha una vez en una sesion
no protege nada; solo cuenta si esta en la suite. Cada test de aqui muta el documento y exige
que el gate se rompa.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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


@pytest.mark.integration
def test_el_ledger_vigente_pasa_su_gate() -> None:
    """Baseline for the custody gate."""
    codigo, salida = _correr_artifacts_check(LEDGER)
    assert codigo == 0, salida


def test_tracked_minimal_ledger_passes_in_fresh_checkout(tmp_path: Path) -> None:
    """Keep a positive control in CI without requiring deliberately untracked PDFs."""
    text = LEDGER.read_text(encoding="utf-8")
    seal = re.search(r"^\*\*Sellado el\*\*:.+$", text, re.M)
    assert seal is not None, "the ledger has no sealing header"
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    row = None
    for line in text.splitlines():
        cells = line.strip().strip("|").split("|")
        if not line.startswith("|") or len(cells) < 6:
            continue
        path_match = re.search(r"`([^`]+)`", cells[1])
        has_commit = re.search(r"`[0-9a-f]{7,40}`", cells[4])
        if path_match is not None and path_match.group(1) in tracked and has_commit is not None:
            row = line
            break
    assert row is not None, "the ledger has no tracked row for the positive control"
    minimal_ledger = tmp_path / "ARTIFACTS.md"
    minimal_ledger.write_text(f"# Positive control\n\n{seal.group(0)}\n\n{row}\n", encoding="utf-8")

    code, output = _correr_artifacts_check(minimal_ledger)
    assert code == 0, output


def test_un_commit_de_sellado_inventado_rompe_el_gate(tmp_path: Path) -> None:
    """A sealing commit that is not in the history of HEAD is provenance, not decoration."""
    texto = LEDGER.read_text(encoding="utf-8")

    mutado = re.sub(r"(\*\*Commit de sellado\*\*: `)[0-9a-f]{7,40}(`)", r"\g<1>deadbee\g<2>", texto)
    assert mutado != texto, "no se encontro el commit de sellado en la cabecera"
    copia = tmp_path / "ARTIFACTS.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_artifacts_check(copia)
    assert codigo == 1, salida
    assert "commit de sellado" in salida


def test_decir_que_un_artefacto_no_esta_en_git_teniendolo_rompe_el_gate(tmp_path: Path) -> None:
    """The custody gate compared bytes for two rounds and never looked at provenance."""

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

    Una auditoria copio una cifra al preregistro sin nombrar el artefacto y el gate paso. Ahora
    las cifras distintivas —cuatro decimales o mas— se extraen de los propios artefactos.
    """
    cifra = _una_cifra_vigilada()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "documento-nuevo.md").write_text(
        f"# Un documento activo\n\nEl efecto medido fue {cifra} y lo damos por bueno.\n",
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


def _una_cifra_vigilada() -> str:
    """One figure the publication gate is actually watching, taken FROM THE GATE.

    Calcularla aparte fue un error concreto: cuando el gate empezo a descontar las cifras que
    tambien aparecen en artefactos vigentes, el test siguio eligiendo del conjunto sin descontar
    y acabo probando con una cifra que el gate ya no vigila. Un test que no comparte la
    definicion con lo que prueba deja de probarlo sin avisar.
    """
    import importlib.util

    ruta = REPO_ROOT / "scripts" / "paper_obsoletos_check.py"
    spec = importlib.util.spec_from_file_location("obsoletos_bajo_prueba", ruta)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    vigiladas, _ = modulo.cifras_vigiladas(LEDGER)
    assert vigiladas, "el gate no vigila ninguna cifra"
    return sorted(vigiladas)[0]


def test_un_documento_nuevo_en_revisiones_externas_no_queda_invisible() -> None:
    """The exemption is a sealed list, not a folder: a folder hides anything dropped into it."""
    intruso = REPO_ROOT / "docs" / "paper" / "revisiones-externas" / "_intruso_de_prueba.md"
    assert not intruso.exists()
    try:
        # La cifra se toma del propio conjunto vigilado, no de memoria: escribir una a mano es
        # como el test se vuelve una comprobacion de que la cifra elegida sigue existiendo.
        cifra = _una_cifra_vigilada()
        intruso.write_text(
            f"# Documento colado\n\nEl efecto fue {cifra} y lo damos por bueno.\n",
            encoding="utf-8",
        )
        codigo, salida = _correr_obsoletos_check(LEDGER)
        assert codigo == 1, salida
        assert "_intruso_de_prueba.md" in salida
    finally:
        intruso.unlink(missing_ok=True)
    assert _correr_obsoletos_check(LEDGER)[0] == 0


def test_editar_un_documento_recibido_rompe_el_gate(tmp_path: Path) -> None:
    """The exemption promises the received document is untouched; the seal is what proves it."""
    import shutil

    doc = REPO_ROOT / "docs" / "paper" / "revisiones-externas" / "README.md"
    respaldo = tmp_path / "respaldo.md"
    shutil.copy2(doc, respaldo)
    try:
        doc.write_text(doc.read_text(encoding="utf-8") + "\nUna linea nuestra.\n", encoding="utf-8")
        codigo, salida = _correr_obsoletos_check(LEDGER)
        assert codigo == 1, salida
        assert "su MD5 cambio" in salida
    finally:
        shutil.copy2(respaldo, doc)
    assert _correr_obsoletos_check(LEDGER)[0] == 0


# --------------------------------------------------------------------------------------
# El protocolo de US-172: no se congela con campos operativos sin rellenar.
# --------------------------------------------------------------------------------------

PROTOCOLO = REPO_ROOT / "docs" / "paper" / "perdidas-protocolo.md"


def _correr_protocolo_check(protocolo: Path) -> tuple[int, str]:
    """Run the protocol gate over a given file."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "protocolo_check.py"),
            "--protocolo",
            str(protocolo),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_el_protocolo_en_borrador_pasa_con_campos_pendientes() -> None:
    """A draft may have empty fields; that is what a draft is."""
    codigo, salida = _correr_protocolo_check(PROTOCOLO)
    assert codigo == 0, salida
    assert "estado: BORRADOR" in salida


def test_congelar_con_campos_sin_rellenar_rompe_el_gate(tmp_path: Path) -> None:
    """Freezing is the moment those fields stop being optional.

    Son decisiones de personas y el codigo no puede rellenarlas; lo que si puede es impedir que el
    documento se declare congelado sin ellas. Un aviso en prosa no impide nada.
    """
    texto = PROTOCOLO.read_text(encoding="utf-8")
    mutado = texto.replace("**Estado**: BORRADOR", "**Estado**: CONGELADO", 1)
    assert mutado != texto
    copia = tmp_path / "protocolo.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_protocolo_check(copia)
    assert codigo == 1, salida
    assert "sin rellenar" in salida


def test_congelar_sin_determinacion_institucional_rompe_el_gate(tmp_path: Path) -> None:
    """A frozen protocol has to cite the institutional determination it was frozen against."""
    texto = PROTOCOLO.read_text(encoding="utf-8")
    mutado = texto.replace("**Estado**: BORRADOR", "**Estado**: CONGELADO", 1)
    # Se rellenan los campos y se BORRA la fila de la determinacion: el unico fallo posible.
    mutado = mutado.replace("`[POR DEFINIR]`", "`Nombre Apellido`")
    mutado = "\n".join(
        linea
        for linea in mutado.splitlines()
        if "Referencia de la determinación o aprobación institucional" not in linea
    )
    copia = tmp_path / "protocolo.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_protocolo_check(copia)
    assert codigo == 1, salida
    assert "determinacion institucional" in salida


def test_borrar_las_permutaciones_rompe_el_gate(tmp_path: Path) -> None:
    """Generating the reading orders after starting is choosing them knowing who is interviewed."""
    texto = PROTOCOLO.read_text(encoding="utf-8")
    mutado = "\n".join(
        linea for linea in texto.splitlines() if not linea.strip().startswith("| `P")
    )
    copia = tmp_path / "protocolo.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_protocolo_check(copia)
    assert codigo == 1, salida
    assert "permutaciones" in salida


def test_quitar_la_cuarentena_del_estado_del_manuscrito_rompe_el_gate(tmp_path: Path) -> None:
    """``paper/micai/ESTADO.md`` cites invalidated figures, so it is a consumer, not an exemption.

    Estuvo exento y ademas fuera del barrido —que solo recorria `docs/paper/`—, asi que el gate
    daba verde sin verlo mientras el fichero afirmaba un veredicto con las cuatro cifras
    invalidadas. Dos agujeros de la misma clase: eximir por lista y eximir por ubicacion.
    """
    import shutil

    doc = REPO_ROOT / "paper" / "micai" / "ESTADO.md"
    respaldo = tmp_path / "respaldo.md"
    shutil.copy2(doc, respaldo)
    try:
        texto = doc.read_text(encoding="utf-8")
        assert "> **CUARENTENA**" in texto
        doc.write_text(texto.replace("> **CUARENTENA**", "> Nota", 1), encoding="utf-8")
        codigo, salida = _correr_obsoletos_check(LEDGER)
        assert codigo == 1, salida
        assert "ESTADO.md" in salida
    finally:
        shutil.copy2(respaldo, doc)
    assert _correr_obsoletos_check(LEDGER)[0] == 0


# --------------------------------------------------------------------------------------
# US-173: el contrato del estimando, y las tres claves que lo convertirian en otro estudio.
# --------------------------------------------------------------------------------------

CONTRATO = REPO_ROOT / "docs" / "paper" / "estimando-v1.json"
PREREGISTRO = REPO_ROOT / "docs" / "paper" / "preregistro-v2-borrador.md"


def _correr_preregistro_check(contrato: Path, preregistro: Path) -> tuple[int, str]:
    """Run the estimand-contract gate over a given pair of files."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "preregistro_check.py"),
            "--contrato",
            str(contrato),
            "--preregistro",
            str(preregistro),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_el_contrato_vigente_cuadra_con_la_prosa() -> None:
    """Baseline: the JSON and section 4.5 say the same thing."""
    codigo, salida = _correr_preregistro_check(CONTRATO, PREREGISTRO)
    assert codigo == 0, salida


def test_cerrar_el_estimando_y_seguir_exigiendo_cuatro_parametros_rompe_el_gate(
    tmp_path: Path,
) -> None:
    """The same header cannot declare three open parameters and then require closing four."""
    texto = PREREGISTRO.read_text(encoding="utf-8")
    mutado = texto.replace(
        "No se firma hasta cerrar esos tres",
        "No se firma hasta cerrarlos los cuatro",
    )
    copia = tmp_path / "preregistro.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_preregistro_check(CONTRATO, copia)
    assert codigo == 1, salida
    assert "tres parámetros" in salida


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("operating_point_source", "test"),
        ("rematch_on_test", True),
        ("transport_claim", True),
    ],
)
def test_las_tres_claves_que_convertirian_esto_en_otro_estudio(
    tmp_path: Path, clave: str, valor: object
) -> None:
    """Three keys turn the study into a different study; the gate refuses each.

    Elegir el punto de operacion en la prueba, volver a igualar en la prueba, o afirmar transporte
    no son ajustes de configuracion: son otro diseno. Que un JSON pueda cambiarlos en silencio es
    exactamente por lo que el contrato tiene gate.
    """
    import json

    datos = json.loads(CONTRATO.read_text(encoding="utf-8"))
    datos[clave] = valor
    copia = tmp_path / "estimando.json"
    copia.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    codigo, salida = _correr_preregistro_check(copia, PREREGISTRO)
    assert codigo == 1, salida
    assert clave in salida


def test_la_divergencia_entre_el_json_y_la_prosa_rompe_el_gate(tmp_path: Path) -> None:
    """Two sources saying the same thing drift apart; that is what this gate is for."""
    texto = PREREGISTRO.read_text(encoding="utf-8")
    mutado = texto.replace("Queda prohibido volver a igualar", "Se permite volver a igualar", 1)
    assert mutado != texto
    copia = tmp_path / "preregistro.md"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_preregistro_check(CONTRATO, copia)
    assert codigo == 1, salida
    assert "rematch_on_test" in salida


# --------------------------------------------------------------------------------------
# US-118: el inventario de OOF y los consumidores que lo respetan.
# --------------------------------------------------------------------------------------


def test_el_analisis_micai_rechaza_un_oof_no_canonico() -> None:
    """Declaring a file unverified does not stop anyone from reading it; refusing to load it does.

    Es la leccion de los trece artefactos OBSOLETO, que llevaban su aviso en la cabecera del
    ledger y se citaban igual. `load_member_posteriors` es el unico punto por el que el analisis
    MICAI lee posteriores, asi que la regla se impone ahi una vez y vale para todas las fases.
    """
    from ml.eval.oof.inventario import EstadoNoCanonicoError, estado_de_miembro
    from ml.eval.paper_micai_arbitration import load_member_posteriors

    no_canonico = next(
        m
        for m in ("farslip-ft18", "farslip-zeroshot", "xgb-alphaearth", "xgb-alphaearth-italia")
        if estado_de_miembro(m) != "canonical"
    )
    with pytest.raises(EstadoNoCanonicoError, match=no_canonico):
        load_member_posteriors(REPO_ROOT / "ml" / "eval" / "oof", (no_canonico,), ["x"])


def test_la_escapatoria_existe_y_hay_que_pedirla_a_proposito(tmp_path: Path) -> None:
    """Diagnostics and migrations need to read them; the analysis must not do it by accident."""
    import polars as pl

    from ml.eval.oof.inventario import estado_de_miembro
    from ml.eval.paper_micai_arbitration import load_member_posteriors
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    no_canonico = next(
        m for m in ("farslip-ft18", "xgb-alphaearth") if estado_de_miembro(m) != "canonical"
    )
    pl.DataFrame(
        {
            "canonical_parcel_id": ["otra-parcela"],
            **{column: [1.0 / len(PROB_COLUMNS)] for column in PROB_COLUMNS},
        }
    ).write_parquet(tmp_path / f"oof_parcel_{no_canonico}_fold5.parquet")
    # With the escape hatch, state validation passes and coverage validation
    # fails later. The synthetic parquet keeps this distinction testable in CI.
    with pytest.raises(ValueError, match="no cubre"):
        load_member_posteriors(
            tmp_path,
            (no_canonico,),
            ["parcela-que-no-existe"],
            permitir_no_canonicos=True,
        )


def test_el_inventario_no_admite_estados_inventados(tmp_path: Path) -> None:
    """Three states, and a fourth one is a silent way of declaring nothing."""
    import json

    copia = _copia_oof(tmp_path, con_parquet=False)
    datos = json.loads((copia / "inventario.json").read_text(encoding="utf-8"))
    primero = next(iter(datos["ficheros"]))
    datos["ficheros"][primero]["estado"] = "casi_bueno"
    (copia / "inventario.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "oof_manifest_check.py"), "--oof", str(copia)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout
    assert "desconocido" in proc.stdout


def test_un_legacy_sin_siguiente_paso_rompe_el_gate(tmp_path: Path) -> None:
    """Without a written way out, a temporary state becomes a permanent one."""
    import json

    copia = _copia_oof(tmp_path, con_parquet=False)
    datos = json.loads((copia / "inventario.json").read_text(encoding="utf-8"))
    objetivo = next(k for k, v in datos["ficheros"].items() if v["estado"] == "legacy_unverified")
    datos["ficheros"][objetivo].pop("siguiente_paso", None)
    (copia / "inventario.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "oof_manifest_check.py"), "--oof", str(copia)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout
    assert "siguiente_paso" in proc.stdout


def _copia_oof(tmp_path: Path, *, con_parquet: bool) -> Path:
    """A copy of the OOF directory with or without the parquet files.

    Sin ellos es un clon limpio: solo punteros `.dvc`. Es el caso que rompia el gate en CI y por
    el que no se podia incorporar.
    """
    import shutil

    origen = REPO_ROOT / "ml" / "eval" / "oof"
    copia = tmp_path / "oof"
    copia.mkdir()
    for nombre in ("manifest.json", "inventario.json"):
        shutil.copy2(origen / nombre, copia / nombre)
    for puntero in origen.glob("*.dvc"):
        shutil.copy2(puntero, copia / puntero.name)
    if con_parquet:
        import hashlib
        import json

        # Exercise the bytes branch without requiring DVC blobs in CI. The gate
        # only validates identity here, so a tiny opaque payload is sufficient.
        inventory_path = copia / "inventario.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        name = next(
            candidate for candidate in inventory["ficheros"] if candidate.startswith("oof_parcel_")
        )
        payload = b"oof-manifest-check synthetic bytes fixture"
        (copia / name).write_bytes(payload)
        inventory["ficheros"][name]["md5"] = hashlib.md5(payload).hexdigest()
        inventory["ficheros"][name]["bytes"] = len(payload)
        inventory_path.write_text(json.dumps(inventory, ensure_ascii=False), encoding="utf-8")
    return copia


def _correr_oof_check(oof: Path) -> tuple[int, str]:
    """Run the OOF inventory gate over a given directory."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "oof_manifest_check.py"), "--oof", str(oof)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_el_gate_pasa_en_un_clon_limpio_sin_los_parquet(tmp_path: Path) -> None:
    """A clean checkout has the pointers and not the blobs; the gate has to work there.

    Un gate que exige los bytes solo se puede correr en la maquina donde se escribio, que es lo
    mismo que le pasaba al manifiesto con sus rutas de Windows.
    """
    codigo, salida = _correr_oof_check(_copia_oof(tmp_path, con_parquet=False))
    assert codigo == 0, salida
    assert "por puntero .dvc" in salida
    assert "por bytes: 0" in salida


def test_un_puntero_dvc_que_no_cuadra_con_el_inventario_rompe_el_gate(tmp_path: Path) -> None:
    """The DVC path must be able to fail, or it is decoration."""
    import re as _re

    copia = _copia_oof(tmp_path, con_parquet=False)
    puntero = next(copia.glob("oof_parcel_*.dvc"))
    texto = puntero.read_text(encoding="utf-8")
    mutado = _re.sub(r"(md5:\s*)[0-9a-f]{32}", r"\g<1>" + "0" * 32, texto, count=1)
    assert mutado != texto
    puntero.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_oof_check(copia)
    assert codigo == 1, salida
    assert "el puntero .dvc registra" in salida


def test_un_parquet_que_no_cuadra_con_el_inventario_rompe_el_gate(tmp_path: Path) -> None:
    """And the bytes path too, so both roads are proven and not just the one we run locally."""
    copia = _copia_oof(tmp_path, con_parquet=True)
    objetivo = next(copia.glob("oof_parcel_*.parquet"))
    objetivo.write_bytes(objetivo.read_bytes() + b"\x00")
    codigo, salida = _correr_oof_check(copia)
    assert codigo == 1, salida
    assert "el inventario registra" in salida


def test_ni_el_parquet_ni_su_puntero_rompe_el_gate(tmp_path: Path) -> None:
    """Declared and absent in both forms is the one case that must never pass silently."""
    copia = _copia_oof(tmp_path, con_parquet=False)
    next(copia.glob("oof_parcel_*.dvc")).unlink()
    codigo, salida = _correr_oof_check(copia)
    assert codigo == 1, salida
    assert "ni el parquet ni su puntero" in salida


def _convert_manifest_to_v2(oof: Path) -> dict[str, Any]:
    """Upgrade the copied historical manifest to the current executable contract."""
    import json

    path = oof / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    for entry in data["models"].values():
        if entry["status"] != "ok":
            continue
        entry["code_version"] = data["code_version"]
        entry["data_version"] = data["data_version"]
        if entry["model_kind"] in {
            "tsvit",
            "tsvit-pheno",
            "tsvit-pheno-fullm",
            "utae",
            "anysat",
        }:
            entry["n_timesteps_dataset"] = 10
            entry["n_timesteps_model_spec"] = 10
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def test_complete_v2_manifest_passes(tmp_path: Path) -> None:
    """The positive v2 control proves the new checks are satisfiable."""
    oof = _copia_oof(tmp_path, con_parquet=False)
    _convert_manifest_to_v2(oof)
    codigo, salida = _correr_oof_check(oof)
    assert codigo == 0, salida


def test_temporal_entry_without_effective_steps_fails(tmp_path: Path) -> None:
    """A temporal entry cannot hide the parameter that couples model and dataset."""
    import json

    oof = _copia_oof(tmp_path, con_parquet=False)
    data = _convert_manifest_to_v2(oof)
    temporal = next(
        entry
        for entry in data["models"].values()
        if entry["model_kind"] in {"tsvit-pheno", "utae", "anysat"}
    )
    temporal.pop("n_timesteps_dataset")
    (oof / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    codigo, salida = _correr_oof_check(oof)
    assert codigo == 1, salida
    assert "no declara ambas configuraciones" in salida


def test_mismatched_temporal_steps_fail(tmp_path: Path) -> None:
    """Counting both fields is insufficient: the control enforces their coupling."""
    import json

    oof = _copia_oof(tmp_path, con_parquet=False)
    data = _convert_manifest_to_v2(oof)
    temporal = next(
        entry
        for entry in data["models"].values()
        if entry["model_kind"] in {"tsvit-pheno", "utae", "anysat"}
    )
    temporal["n_timesteps_dataset"] = 37
    (oof / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    codigo, salida = _correr_oof_check(oof)
    assert codigo == 1, salida
    assert "no coinciden" in salida


def test_model_appended_from_another_run_fails(tmp_path: Path) -> None:
    """Per-entry provenance catches the append that top-level provenance could not."""
    import json

    oof = _copia_oof(tmp_path, con_parquet=False)
    data = _convert_manifest_to_v2(oof)
    first = next(iter(data["models"].values()))
    first["code_version"] = "otra-corrida"
    (oof / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    codigo, salida = _correr_oof_check(oof)
    assert codigo == 1, salida
    assert "code_version de la entrada" in salida


# --------------------------------------------------------------------------------------
# US-139: el panel congelado.
# --------------------------------------------------------------------------------------

PANEL = REPO_ROOT / "docs" / "paper" / "panel-v1.json"


def _correr_panel_check(panel: Path) -> tuple[int, str]:
    """Run the frozen-panel gate over a given panel file."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "panel_check.py"), "--panel", str(panel)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_el_panel_congelado_es_coherente() -> None:
    """Baseline: five members, four families, none of them excluded."""
    codigo, salida = _correr_panel_check(PANEL)
    assert codigo == 0, salida
    assert "margen sobre el minimo" in salida


def test_meter_en_el_panel_un_miembro_excluido_rompe_el_gate(tmp_path: Path) -> None:
    """A member the analysis cannot read has no business in the panel.

    La contradiccion solo se descubriria al correr el analisis, y para entonces el panel ya esta
    congelado: es el orden equivocado para enterarse.
    """
    import json

    datos = json.loads(PANEL.read_text(encoding="utf-8"))
    excluido = datos["fuera_del_panel"][0]["nombre"]
    datos["miembros"].append({"nombre": excluido, "familia": "colada", "temporal": False})
    copia = tmp_path / "panel.json"
    copia.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    codigo, salida = _correr_panel_check(copia)
    assert codigo == 1, salida
    assert excluido in salida


def test_declarar_un_campeon_rompe_el_gate(tmp_path: Path) -> None:
    """The predictor is a sensitivity factor; a champion is the thing the design cannot support."""
    import json

    datos = json.loads(PANEL.read_text(encoding="utf-8"))
    datos["campeon_declarado"] = datos["miembros"][0]["nombre"]
    copia = tmp_path / "panel.json"
    copia.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    codigo, salida = _correr_panel_check(copia)
    assert codigo == 1, salida
    assert "campeon" in salida


def test_quedarse_por_debajo_del_minimo_de_familias_rompe_el_gate(tmp_path: Path) -> None:
    """The margin is one family; the gate has to notice the day it runs out."""
    import json

    datos = json.loads(PANEL.read_text(encoding="utf-8"))
    for miembro in datos["miembros"]:
        miembro["familia"] = "una_sola"
    datos["familias_distintas"] = 1
    copia = tmp_path / "panel.json"
    copia.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    codigo, salida = _correr_panel_check(copia)
    assert codigo == 1, salida
    assert "familias distintas" in salida


# --------------------------------------------------------------------------------------
# US-142: el catalogo de referencias, y los identificadores que no toleran ser numeros.
# --------------------------------------------------------------------------------------

CATALOGO = REPO_ROOT / "paper" / "micai2027" / "refs-candidates.bib"
OVERRIDES_CSV = REPO_ROOT / "reports" / "paper_micai" / "fase0" / "related_work_overrides.csv"


def _correr_bib_check(catalogo: Path, overrides: Path | None = None) -> tuple[int, str]:
    """Run the bibliography gate over a given catalogue."""
    orden = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "paper_bib_check.py"),
        "--catalogo",
        str(catalogo),
    ]
    if overrides is not None:
        orden += ["--overrides", str(overrides)]
    proc = subprocess.run(orden, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout


def test_el_catalogo_vigente_pasa_su_gate() -> None:
    """Baseline: every entry locatable, no truncated author list."""
    codigo, salida = _correr_bib_check(CATALOGO)
    assert codigo == 0, salida
    assert "sin identificador localizable: 0" in salida


def test_las_entradas_sin_citar_se_reportan_pero_no_hacen_fallar() -> None:
    """There is no new manuscript yet, so every entry is uncited and that is not a defect."""
    codigo, salida = _correr_bib_check(CATALOGO)
    assert codigo == 0, salida
    assert "sin citar (se reporta, no falla)" in salida


def test_un_eprint_con_el_cero_final_perdido_rompe_el_gate(tmp_path: Path) -> None:
    """``2511.10370`` must survive literally; ``2511.1037`` is an eprint that does not exist.

    El defecto no lo tenia el CSV: lo tenia el generador, que leia la columna infiriendo el tipo
    y convertia el identificador en un float. El cero final desaparecia sin que nada avisara, y
    leyendo el bib no se ve.
    """
    texto = CATALOGO.read_text(encoding="utf-8")
    assert "2511.10370" in texto, "el catalogo ya no trae el identificador que este test vigila"
    mutado = texto.replace("2511.10370", "2511.1037", 1)
    copia = tmp_path / "refs.bib"
    copia.write_text(mutado, encoding="utf-8")
    codigo, salida = _correr_bib_check(copia)
    assert codigo == 1, salida
    assert "cero final" in salida


def test_leer_las_correcciones_como_numeros_produce_una_diferencia_detectable(
    tmp_path: Path,
) -> None:
    """Coercing the literal columns is not cosmetic: it silently changes what gets published.

    Se comprueba en el mecanismo, no en el resultado: se lee el CSV de las dos maneras y se exige
    que al menos un valor cambie. Si algun dia dejaran de diferir, este test avisaria de que la
    proteccion sobra o de que se dejo de necesitar.
    """
    import polars as pl

    columnas = ("id", "volume", "number", "pages", "url")
    literal = pl.read_csv(OVERRIDES_CSV, schema_overrides={c: pl.Utf8 for c in columnas})
    inferido = pl.read_csv(OVERRIDES_CSV)
    diferencias = [
        (clave, a, b)
        for clave, a, b in zip(
            literal["key"].to_list(),
            literal["id"].to_list(),
            [None if v is None else str(v) for v in inferido["id"].to_list()],
            strict=True,
        )
        if a != b
    ]
    assert diferencias, (
        "leer los identificadores como numeros ya no cambia nada: o el esquema del CSV cambio, "
        "o esta proteccion dejo de hacer falta y hay que decirlo"
    )
    assert any(a and b and a.rstrip("0") == b.rstrip("0") and a != b for _, a, b in diferencias), (
        f"la diferencia esperada es un cero final perdido y no aparece: {diferencias}"
    )


def test_el_bib_historico_modificado_rompe_el_gate(tmp_path: Path) -> None:
    """The archived bibliography is immutable; the gate has to notice if it moves.

    Y tiene que reconocer la fila POR SU CELDA de ruta: la primera version buscaba la ruta como
    subcadena en la linea entera, asi que la nota de otra fila —que menciona este fichero para
    decir que es inmutable— se tomaba por su fila y comparaba su hash con el de otro artefacto.
    El gate acusaba un cambio que no existia.
    """
    import shutil

    historico = REPO_ROOT / "paper" / "micai" / "refs.bib"
    respaldo = tmp_path / "refs.bib"
    shutil.copy2(historico, respaldo)
    try:
        historico.write_text(
            historico.read_text(encoding="utf-8") + "\n% una linea de mas\n", encoding="utf-8"
        )
        codigo, salida = _correr_bib_check(CATALOGO)
        assert codigo == 1, salida
        assert "inmutable" in salida
    finally:
        shutil.copy2(respaldo, historico)
    assert _correr_bib_check(CATALOGO)[0] == 0


@pytest.mark.parametrize(
    ("etiqueta", "sustituto"),
    [
        ("tupla literal anotada", 'ALL_MEMBERS: tuple[str, ...] = ("unet", "deeplabv3plus")'),
        ("sin anotacion de tipo", 'ALL_MEMBERS = ("unet", "deeplabv3plus")'),
        ("como lista", 'ALL_MEMBERS: list[str] = ["unet", "deeplabv3plus"]'),
        (
            "a traves de un alias",
            'MIEMBROS = ("unet", "deeplabv3plus")\nALL_MEMBERS: tuple[str, ...] = MIEMBROS',
        ),
        ("alias de dos saltos", 'A = ("unet",)\nMIEMBROS = A\nALL_MEMBERS = MIEMBROS'),
        ("otra funcion parecida", "ALL_MEMBERS = miembros_de_otro_sitio()"),
    ],
)
def test_ninguna_variante_sintactica_burla_la_lectura_del_panel(
    etiqueta: str, sustituto: str
) -> None:
    """A member list written twice drifts apart, and this one did.

    Al congelar el panel en cinco miembros, fase 2 y fase 3 seguian pidiendo los diez originales
    -cinco ya excluidos o sin verificar-. Al regenerar sus artefactos habrian usado el conjunto
    equivocado, o habrian reventado, sin que nada relacionara una cosa con la otra.

    La primera version de esta comprobacion miraba un ``AnnAssign`` cuyo valor fuera una tupla
    literal, y se le colaban tres variantes: quitar la anotacion, usar una lista, pasar por un
    alias. Reparar el caso en vez de la clase es el modo de fallo que estas auditorias repiten,
    asi que aqui se prueban las variantes y no solo la que se vio primero.
    """
    import shutil
    import tempfile

    guion = REPO_ROOT / "scripts" / "run_paper_micai_fase3.py"
    original = "ALL_MEMBERS: tuple[str, ...] = miembros_del_panel()"
    with tempfile.TemporaryDirectory() as tmp:
        respaldo = Path(tmp) / "fase3.py"
        shutil.copy2(guion, respaldo)
        try:
            texto = guion.read_text(encoding="utf-8")
            assert original in texto
            guion.write_text(texto.replace(original, sustituto, 1), encoding="utf-8")
            codigo, salida = _correr_panel_check(PANEL)
            assert codigo == 1, f"{etiqueta}: {salida}"
            assert "ALL_MEMBERS no sale de miembros_del_panel()" in salida
        finally:
            shutil.copy2(respaldo, guion)
    assert _correr_panel_check(PANEL)[0] == 0


@pytest.mark.parametrize(
    "modulo", ["scripts.run_paper_micai_fase2", "scripts.run_paper_micai_fase3"]
)
def test_los_guiones_piden_en_ejecucion_exactamente_el_panel_congelado(modulo: str) -> None:
    """La comprobacion estatica vive en un gate sin dependencias; esta mira el valor de verdad.

    El gate barato de la CI solo tiene stdlib, asi que razona sobre el AST. Aqui, donde si hay
    dependencias, se importa el modulo y se compara el valor efectivo con el panel: es la unica
    comprobacion que ninguna forma sintactica puede burlar.
    """
    import importlib

    from ml.eval.paper_micai_arbitration import miembros_del_panel

    assert importlib.import_module(modulo).ALL_MEMBERS == miembros_del_panel()


# ------------------------------------------------------------------------------------------
# El alcance del gate de cuarentena: era `docs/paper/**/*.md` y el manuscrito es `.tex`.
# ------------------------------------------------------------------------------------------


def _correr_obsoletos() -> tuple[int, str]:
    """Correr el gate de cuarentena sobre el repositorio entero."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "paper_obsoletos_check.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _con_fichero(ruta: Path, contenido: str) -> tuple[int, str]:
    """Crear un fichero, correr el gate y borrarlo siempre."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")
    try:
        return _correr_obsoletos()
    finally:
        ruta.unlink()


@pytest.mark.parametrize(
    ("etiqueta", "relativa", "ingles"),
    [
        ("el manuscrito, que es .tex y en ingles", "paper/micai2027/_prueba.tex", True),
        ("el manuscrito, que es .tex y en espanol", "paper/micai2027/_prueba.tex", False),
        ("un ADR, que vive en docs/decisions", "docs/decisions/_prueba.md", False),
        ("el plan, que vive en context/", "context/_prueba.md", False),
        ("un documento en la raiz", "_prueba.md", False),
        ("docs/ fuera de docs/paper", "docs/_prueba.md", False),
    ],
)
def test_el_gate_de_cuarentena_alcanza_donde_estaba_ciego(
    etiqueta: str, relativa: str, ingles: bool
) -> None:
    """El alcance era ``docs/paper/**/*.md``, y el manuscrito es ``.tex``.

    Seis superficies quedaban fuera, incluida la unica que de verdad importa: el fichero donde se
    escribe el articulo. Un control que protege el manuscrito y no sabe leer su formato no protege
    nada.
    """
    cifra = _una_cifra_vigilada()
    escrita = cifra.replace(",", ".") if ingles else cifra
    codigo, salida = _con_fichero(REPO_ROOT / relativa, f"El resultado fue {escrita}.\n")
    assert codigo == 1, f"{etiqueta}: {salida}"
    assert "reproduce" in salida
    assert _correr_obsoletos()[0] == 0


def test_una_cifra_dentro_de_otra_mas_larga_no_acusa_a_nadie() -> None:
    """``0,0342`` estaba dentro de ``arXiv:2310.03425``, y el gate lo contaba como cita.

    Buscar la subcadena acusa a quien no cita nada. Con la frontera puesta, dos ficheros de
    entregables del curso dejaron de aparecer.
    """
    cifra = _una_cifra_vigilada()
    decimales = cifra.split(",")[1]
    codigo, salida = _con_fichero(
        REPO_ROOT / "docs" / "_prueba.md", f"Vease arXiv:2310.{decimales}25 y nada mas.\n"
    )
    assert codigo == 0, salida


def test_la_forma_inglesa_en_prosa_nuestra_no_acusa() -> None:
    """En ``.md`` solo cuenta la coma decimal, y es una decision con motivo.

    Admitir el punto en prosa llenaba el control de coincidencias con metricas de otras historias
    -``0,4094`` sale en catorce documentos de US-022 y US-023- sin anadir ni una cita real. En
    ``.tex`` si valen las dos, porque el manuscrito de envio se escribe en ingles.
    """
    cifra = _una_cifra_vigilada()
    codigo, salida = _con_fichero(
        REPO_ROOT / "docs" / "_prueba.md", f"El resultado fue {cifra.replace(',', '.')}.\n"
    )
    assert codigo == 0, salida


def test_editar_una_fuente_del_manuscrito_retirado_rompe_el_gate() -> None:
    """La exencion del manuscrito retirado promete que no se toca, y esto lo verifica.

    Si se edita, deja de estar retirado y sus cifras vuelven a ser afirmaciones vigentes.
    """
    fuente = REPO_ROOT / "paper" / "micai" / "sections_es" / "04-resultados.tex"
    respaldo = fuente.read_text(encoding="utf-8")
    try:
        fuente.write_text(respaldo + "\n% retoque\n", encoding="utf-8")
        codigo, salida = _correr_obsoletos()
        assert codigo == 1, salida
        assert "es fuente del manuscrito retirado y su MD5 cambio" in salida
    finally:
        fuente.write_text(respaldo, encoding="utf-8")
    assert _correr_obsoletos()[0] == 0


def test_una_colision_declarada_no_tapa_una_cita_nueva() -> None:
    """Las colisiones se declaran POR CIFRA, no por fichero.

    Un documento cuyo numero coincide por casualidad con uno obsoleto no puede quedar libre de
    vigilancia para siempre: la exencion cubre esa cifra y ninguna otra.
    """
    from scripts.paper_obsoletos_check import COLISIONES

    fichero = REPO_ROOT / "docs" / "us-resolved" / "us-018.md"
    declaradas = set(COLISIONES["docs/us-resolved/us-018.md"])
    otra = next(c for c in sorted(_cifras_vigiladas_del_repo()) if c not in declaradas)
    respaldo = fichero.read_text(encoding="utf-8")
    try:
        fichero.write_text(f"{respaldo}\n\nY ademas {otra}.\n", encoding="utf-8")
        codigo, salida = _correr_obsoletos()
        assert codigo == 1, salida
        assert "us-018.md" in salida
    finally:
        fichero.write_text(respaldo, encoding="utf-8")
    assert _correr_obsoletos()[0] == 0


def _cifras_vigiladas_del_repo() -> dict[str, str]:
    """Las cifras vigiladas del ledger vigente."""
    from scripts.paper_obsoletos_check import cifras_vigiladas

    return cifras_vigiladas(LEDGER)[0]


def test_el_gate_de_cuarentena_no_tarda_mas_de_diez_segundos() -> None:
    """Un control que se hace pesado acaba fuera de la CI, y entonces no es un control.

    Con 1 160 cifras y 433 documentos, un patron por cifra son medio millon de barridos: dos
    minutos. Una sola alternacion lo deja por debajo del segundo.
    """
    import time

    inicio = time.monotonic()
    assert _correr_obsoletos()[0] == 0
    assert time.monotonic() - inicio < 10.0


def test_cambiar_la_prosa_normativa_del_preregistro_rompe_el_gate() -> None:
    """Comprobar que una FRASE esta presente no dice nada de lo que el parrafo afirma.

    Es la leccion de la seccion 4.6, que decia que un miembro del panel "se excluye" mientras el
    contrato lo tenia dentro, y el gate pasaba porque el nombre estaba. En la 4.5 no hay forma de
    entender la prosa con un patron, asi que se hace lo unico honesto: si el texto normativo
    cambia, el gate se pone en rojo hasta que alguien lo relea y lo vuelva a sellar. El sello no
    afirma que la prosa sea correcta; afirma que nadie la ha movido desde que se leyo.
    """
    gate = REPO_ROOT / "scripts" / "preregistro_check.py"

    def correr() -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(gate)], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return proc.returncode, proc.stdout + proc.stderr

    respaldo = PREREGISTRO.read_text(encoding="utf-8")
    try:
        PREREGISTRO.write_text(
            respaldo.replace("La parcela", "La parcela, en efecto,", 1), encoding="utf-8"
        )
        codigo, salida = correr()
        assert codigo == 1, salida
        assert "la prosa normativa `seccion_4_5` cambio" in salida
    finally:
        PREREGISTRO.write_text(respaldo, encoding="utf-8")
    assert correr()[0] == 0


def test_resellar_cierra_el_fallo_y_solo_toca_el_sello() -> None:
    """``--resellar`` existe para que la friccion sea un gesto explicito, no un hash a mano."""
    gate = REPO_ROOT / "scripts" / "preregistro_check.py"
    respaldo_doc = PREREGISTRO.read_text(encoding="utf-8")
    respaldo_gate = gate.read_text(encoding="utf-8")
    try:
        PREREGISTRO.write_text(
            respaldo_doc.replace("La parcela", "La parcela, en efecto,", 1), encoding="utf-8"
        )
        sellado = subprocess.run(
            [sys.executable, str(gate), "--resellar"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert sellado.returncode == 0, sellado.stdout
        assert "resellado `seccion_4_5`" in sellado.stdout
        despues = subprocess.run(
            [sys.executable, str(gate)], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        assert despues.returncode == 0, despues.stdout
        # El resellado toca el sello y nada mas: las decisiones del contrato siguen intactas.
        assert "EXIGIDO: dict[str, Any] = {" in gate.read_text(encoding="utf-8")
    finally:
        PREREGISTRO.write_text(respaldo_doc, encoding="utf-8")
        gate.write_text(respaldo_gate, encoding="utf-8")
