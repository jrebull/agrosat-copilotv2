"""El banco del articulo se llama PASTIS, y eso tiene que valer en todas sus superficies.

``e431e09`` corrigio el nombre porque **ningun predictor del trabajo usa radar**: la tarjeta del
modelo declara ``in_channels=10`` sobre Sentinel-2 y las 185 caracteristicas salen de diecisiete
indices opticos. Nombrar un sensor que no se toca es exactamente la imprecision que el articulo
denuncia.

La correccion llego al manuscrito y al ledger. No llego a las figuras, ni a los guiones que las
producen, ni al cuaderno publico. Reparar donde a uno le senalan y en ningun otro sitio es el modo
de fallo que estas auditorias repiten, asi que esto vigila la CLASE: cualquier superficie del
articulo que vuelva a llamar PASTIS-R al banco rompe la prueba.

Lo que si es legitimo decir PASTIS-R: la ruta del dataset en disco -que se llama asi-, la
referencia bibliografica al trabajo que lo introduce, y la frase que explica que el banco se
obtiene por esa distribucion pero solo se usa su modalidad optica.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Superficies VIGENTES del articulo: lo normativo y lo que dibuja. Deliberadamente NO entran los
#: documentos historicos -hallazgos fechados, preregistro de fases 3-4, auditorias-, que son el
#: registro de lo que se creyo en su momento y reescribirlos seria falsificarlo.
SUPERFICIES: tuple[str, ...] = (
    "docs/paper/preregistro-v2-borrador.md",
    "docs/paper/perdidas-protocolo.md",
    "docs/paper/panel-v1.json",
    "docs/paper/estimando-v1.json",
    "docs/decisions/ADR-014-micai-2027.md",
    "paper/micai2027/*.tex",
    "paper/micai2027/*.md",
    "scripts/build_micai2027_figures.py",
    "scripts/build_paper_micai_extra_figures.py",
    "scripts/build_paper_micai_fase3_figure.py",
    "scripts/build_paper_micai_fase4_figure.py",
    "scripts/build_paper_micai_patch_figure.py",
    "ml/report/figuras_micai.py",
)

#: Contextos en los que la linea puede nombrar PASTIS-R sin estar nombrando el banco.
PERMITIDO: tuple[re.Pattern[str], ...] = (
    re.compile(r"""["']?data["']?\s*/\s*["']?PASTIS-R"""),  # la ruta del dataset en disco
    re.compile(r"PASTIS-R/"),  # cualquier otra ruta
    re.compile(r"distribuci[oó]n PASTIS-R"),  # la frase sancionada
    re.compile(r"introduce PASTIS-R"),  # la referencia bibliografica
    re.compile(r"PASTIS-R nace"),  # idem
    re.compile(r"no PASTIS-R"),  # la correccion, escrita
    re.compile(r"espectral de PASTIS-R"),  # el eje de bandas de la distribucion
)


def _es_permitido(linea: str) -> bool:
    """Si la linea nombra PASTIS-R en un contexto legitimo.

    Args:
        linea: Linea de texto.

    Returns:
        ``True`` cuando no esta nombrando el banco del articulo.
    """
    return any(patron.search(linea) for patron in PERMITIDO)


def _lineas_sospechosas() -> list[str]:
    """Lineas de las superficies del articulo que nombran PASTIS-R sin excusa."""
    hallazgos: list[str] = []
    for patron in SUPERFICIES:
        for fichero in sorted(REPO_ROOT.glob(patron)):
            for numero, linea in enumerate(
                fichero.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "PASTIS-R" not in linea:
                    continue
                if _es_permitido(linea):
                    continue
                relativa = fichero.relative_to(REPO_ROOT)
                hallazgos.append(f"{relativa}:{numero}: {linea.strip()[:110]}")
    return hallazgos


def test_ninguna_superficie_del_articulo_llama_pastis_r_al_banco() -> None:
    """Hoy ninguna lo hace, y el dia que vuelva a pasar esta prueba lo dice."""
    hallazgos = _lineas_sospechosas()
    assert not hallazgos, "El banco del articulo es PASTIS:\n" + "\n".join(hallazgos)


def test_la_prueba_detecta_el_defecto_que_existio() -> None:
    """Prueba en negativo sobre el rotulo real que llevaban las figuras hasta esta noche."""
    fichero = REPO_ROOT / "scripts" / "build_micai2027_figures.py"
    respaldo = fichero.read_text(encoding="utf-8")
    assert "PASTIS \u00b7 18 clases" in respaldo
    try:
        fichero.write_text(
            respaldo.replace('"PASTIS \u00b7 18 clases', '"PASTIS-R \u00b7 18 clases', 1),
            encoding="utf-8",
        )
        assert _lineas_sospechosas(), "la prueba no vio el rotulo equivocado"
    finally:
        fichero.write_text(respaldo, encoding="utf-8")
    assert not _lineas_sospechosas()


def test_los_contextos_legitimos_no_son_hallazgos() -> None:
    """Distinguir el banco de la distribucion es justo lo que hace util a esta prueba."""
    assert _es_permitido('PASTIS = REPO_ROOT / "data" / "PASTIS-R"')
    assert _es_permitido("el banco se obtiene por la distribución PASTIS-R, solo su parte óptica")
    assert _es_permitido("el trabajo que introduce PASTIS-R es el ISPRS 2022")
    assert not _es_permitido('titulo = "PASTIS-R · 18 clases"')
