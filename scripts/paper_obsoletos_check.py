"""Impide que un documento activo cite cifras de artefactos marcados OBSOLETO sin decirlo.

El ledger sabe que trece artefactos salieron de un modulo con tres defectos, y lo marca. Pero
marcarlo no impedia nada: el gate de custodia los verificaba, imprimia un aviso y devolvia exito,
mientras varios documentos seguian presentando sus cifras como vigentes. Un aviso en prosa contra
un estado ejecutable lo gana siempre el estado, y por eso una cifra obsoleta llego al cuaderno
publico presentada como el experimento corregido.

Este gate separa **custodia** de **disponibilidad editorial**:

- Un documento que cita cifras de un artefacto OBSOLETO tiene que llevar la marca de cuarentena.
- Un documento que menciona la RUTA de un artefacto OBSOLETO y no esta declarado como consumidor
  falla, para que la lista no envejezca en silencio.
- Un documento que reproduce una CIFRA distintiva de un artefacto OBSOLETO falla igual. Buscar solo
  la ruta era el agujero: una auditoria copio 0,0326 al preregistro, sin nombrar el artefacto ni la
  marca, y el gate paso. Las cifras se extraen de los propios artefactos y se buscan en su forma
  espanola; solo cuentan las de cuatro decimales o mas, y **se descuentan las que aparecen tambien
  en un artefacto vigente**: si un numero esta en los dos sitios no es distintivo del obsoleto, y
  vigilarlo acusa a documentos que miden lo suyo.

La marca es una linea que empieza por ``> **CUARENTENA**`` en Markdown, o el atributo
``data-cuarentena`` en HTML. Se pone arriba, donde se lee antes que las cifras.

**Lo que este gate NO detecta, dicho antes de que lo encuentre nadie**: una cifra REDONDEADA al
escribirla —0,0326 escrito «0,033» o «3,3 %»— o parafraseada. Vigila copias literales de cuatro
decimales, que es donde el riesgo de falso positivo es bajo; bajar a tres decimales llenaria el
control de coincidencias. La cobertura es de copia, no de paraphrase, y esa es su frontera.

Uso:
    poetry run python scripts/paper_obsoletos_check.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"
STALE_STATE = "OBSOLETO"
CODE_RE = re.compile(r"`([^`]+)`")
MIN_CELDAS = 6

#: Marca que un documento tiene que llevar para poder citar cifras obsoletas.
MARCA_MD = "> **CUARENTENA**"
MARCA_HTML = "data-cuarentena"

#: Documentos que consumen cifras de artefactos OBSOLETO y estan declarados como tales.
#: Anadir uno aqui NO le da permiso: solo declara que existe. El permiso lo da la marca.
CONSUMIDORES: tuple[str, ...] = (
    "docs/paper/auditoria-2026-09-02.md",
    "docs/paper/fase2-hallazgos.md",
    "docs/paper/novedad.md",
    "docs/paper/preregistro-v2-borrador.md",
    "docs/paper/hallazgos-harness-oof-fold5-2026-09-03.md",
    "docs/paper/fase3-hallazgos.md",
    "docs/paper/fase4-hallazgos.md",
    "docs/paper/que-paper-sale.md",
    "docs/paper/campo-de-tiro.md",
    "docs/paper/recomendacion-final.md",
    "docs/paper/reencuadre-2026-09-03.md",
    "docs/paper/auditoria-revisores-2026-09-03.md",
    "docs/paper/revision-arthur-2026-09-03.md",
    "paper/micai/ESTADO.md",
    "docs/decisions/ADR-013-angulo-micai.md",
)

#: Documentos que hablan DE la obsolescencia y no citan cifras como vigentes.
EXENTOS: tuple[str, ...] = (
    "docs/paper/respuesta-auditoria-externa.md",
    "docs/paper/auditoria-externa/prompt-revalidacion.md",
    "docs/paper/auditoria-externa/prompt-auditoria-externa.md",
    "paper/ARTIFACTS.md",
)

#: Documentos RECIBIDOS de terceros, que se archivan textualmente. No son afirmaciones nuestras y
#: no se retocan: marcarlos con cuarentena seria editar lo que alguien nos escribio.
#:
#: Es una LISTA CERRADA con el sello de cada archivo, no un prefijo de carpeta. Eximir la carpeta
#: hacia invisible cualquier fichero nuevo que alguien dejara alli, y el sello ademas comprueba lo
#: que la exencion promete: que el documento recibido no se ha tocado. Anadir uno exige registrar
#: su MD5 aqui, que es el mismo gesto que sellar un artefacto.
ARCHIVO_AJENO: dict[str, str] = {
    "docs/paper/revisiones-externas/README.md": "22cf1d34c49a66b392ee183f974e19ff",
    "docs/paper/revisiones-externas/evaluacion-cuaderno-micai-2027.md": (
        "3a6f122e32a1c16fd1fcf85097d95766"
    ),
    "docs/paper/revisiones-externas/rutas-micai-2027-post-hallazgos.md": (
        "c9d435e8e7ec239230ff91b6ba19cd00"
    ),
}


#: Fuentes del MANUSCRITO RETIRADO. Se publica retirado, su estado lo declara
#: `paper/micai/ESTADO.md` -que si lleva la marca- y el cuaderno publico, y sus fuentes estan
#: selladas en el ledger: editarlas para anadir una marca de cuarentena seria falsificar el
#: documento cuyo estado ya esta declarado. Como la exencion del archivo ajeno, es una LISTA
#: CERRADA con sello, no un prefijo: eximir la carpeta haria invisible cualquier fichero nuevo.
MANUSCRITO_RETIRADO: dict[str, str] = {
    "paper/micai/sections/03-method.tex": "00b5c188e3e95db69b66c9cbafc25ad9",
    "paper/micai/sections/04-results.tex": "518eeff83b1763b9f2e1a62f1b00f21f",
    "paper/micai/sections_es/03-metodo.tex": "fd574d37a4dee846a482309b3713ab35",
    "paper/micai/sections_es/04-resultados.tex": "4339cc5bce6e5d96b0a3df6f299805d1",
}

#: COLISIONES: cifras que un documento calculo por su cuenta y que coinciden con una de un
#: artefacto obsoleto. No son citas: son numeros distintos que se escriben igual. Cada entrada
#: nombra el fichero Y las cifras concretas, para que una colision declarada no tape una cita
#: nueva en el mismo documento. Verificadas una a una leyendo su contexto.
COLISIONES: dict[str, tuple[str, ...]] = {
    # Metricas de los baselines de US-070 y US-023, de otra linea de trabajo y varios meses antes.
    "paper/tables/us-070/fm_comparison.tex": ("0,4106", "0,5634", "0,6529", "0,6571"),
    "paper/tables/us-070/farslip_band_ablation.tex": ("0,5547",),
    "paper/tables/us-023-preview/baseline_v2_comparison.tex": ("0,4094", "0,7257"),
    # Perdidas de la ablacion por bandas de FarSLIP, que no son estimadores de cobertura.
    "paper/tables/farslip-method/band_ablation.tex": ("0,0181",),
    "paper/tables/farslip-method/cardinality_sweep.tex": ("0,5547",),
    # F1-macro del baseline de 185 caracteristicas, que coincide con un valor de la frontera.
    "docs/us-resolved/us-018.md": ("0,5394",),
}

#: Donde se busca. `docs/paper` era todo el alcance, y por ahi se colaban el manuscrito -que es
#: `.tex`-, los ADR de `docs/decisions`, el plan de `context/` y el resto de `docs/`.
#:
#: Los CUADERNOS quedan fuera a proposito y conviene decirlo antes de que lo encuentre nadie: son
#: entregables del curso con sus propios numeros calculados, y con 1 160 cifras vigiladas la tasa
#: de coincidencia los llena de falsos positivos -treinta y cinco ficheros, ni una cita real-. La
#: frontera del control es la copia literal en prosa nuestra y en el manuscrito, no el cuaderno.
_AMBITO: tuple[tuple[str, str], ...] = (
    ("docs", "**/*.md"),
    ("docs", "**/*.tex"),
    ("paper", "**/*.md"),
    ("paper", "**/*.tex"),
    ("context", "**/*.md"),
    (".", "*.md"),
)

#: Decimales minimos para que una cifra se considere distintiva de su artefacto. Con menos, un
#: 0,90 o un 0,25 aparecen por todas partes y el control se llena de falsos positivos.
DECIMALES_MINIMOS = 4


def _numeros(valor: object, salida: set[float]) -> None:
    """Collect every float reachable from a decoded JSON value."""
    if isinstance(valor, bool):
        return
    if isinstance(valor, (int, float)):
        salida.add(float(valor))
    elif isinstance(valor, dict):
        for v in valor.values():
            _numeros(v, salida)
    elif isinstance(valor, list):
        for v in valor:
            _numeros(v, salida)


def cifras_distintivas(
    ruta: Path, *, minimo: int = DECIMALES_MINIMOS, redondear: bool = False
) -> set[str]:
    """Spanish-formatted figures a document could only have copied from this artefact.

    Searching for the artefact PATH was the hole: prose copies numbers, not paths. Only figures
    with at least :data:`DECIMALES_MINIMOS` decimals count, because those are the ones that do not
    turn up by coincidence.

    Args:
        ruta: Path to a JSON or CSV artefact.
        minimo: Decimals a figure needs to count. The default is what makes a figure distinctive;
            the discount set uses 1, because there the question is whether the value EXISTS.
        redondear: Emit the four-decimal rounded form instead of the exact one, so that ``0.025``
            in an artefact matches ``0,0250`` in a table. Used only for the discount set.

    Returns:
        The figures as they would be written in the prose of this project.
    """
    numeros: set[float] = set()
    if ruta.suffix == ".json":
        try:
            _numeros(json.loads(ruta.read_text(encoding="utf-8")), numeros)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return set()
    elif ruta.suffix == ".csv":
        try:
            with ruta.open(encoding="utf-8", newline="") as handle:
                for fila in csv.reader(handle):
                    for celda in fila:
                        try:
                            numeros.add(float(celda))
                        except ValueError:
                            continue
        except UnicodeDecodeError:
            return set()
    else:
        return set()

    salida: set[str] = set()
    for x in numeros:
        if x != x or abs(x) >= 1000:  # NaN o magnitudes que no son estimadores
            continue
        if redondear:
            salida.add(f"{abs(x):.4f}".replace(".", ","))
            continue
        texto = f"{abs(x):.10f}".rstrip("0")
        entero, _, decimales = texto.partition(".")
        if len(decimales) < minimo:
            continue
        salida.add(f"{entero},{decimales[:DECIMALES_MINIMOS]}")
    return salida


def buscador(cifras: list[str], *, ambas_formas: bool) -> re.Pattern[str]:
    """Un solo patron que encuentra cualquiera de las cifras, escrita entera.

    Dos cosas a la vez. La primera, **la frontera**: buscar la subcadena acusaba a quien no citaba
    nada -``0,0342`` estaba dentro de ``arXiv:2310.03425``-, asi que se exige que no haya digito ni
    separador decimal pegado por delante ni digito por detras. La segunda, **una sola pasada**: con
    1 160 cifras y 433 documentos, un patron por cifra son medio millon de barridos y el gate
    tardaba dos minutos; asi tarda un segundo, y un control que se hace pesado acaba fuera de la CI.

    Args:
        cifras: Cifras en la forma espanola del proyecto.
        ambas_formas: Admitir tambien la coma como punto. El manuscrito de envio se escribe en
            ingles y su tabla de resultados usa punto decimal; en prosa nuestra, admitirlo llenaba
            el control de coincidencias con metricas de otras historias sin anadir una cita real.

    Returns:
        El patron compilado, con la cifra en el grupo ``cifra``.
    """
    cuerpos = [
        re.escape(c).replace(",", "[.,]") if ambas_formas else re.escape(c)
        for c in sorted(cifras, key=len, reverse=True)
    ]
    alternativa = "|".join(cuerpos) if cuerpos else r"(?!)"
    return re.compile(rf"(?<![\d.,])(?P<cifra>{alternativa})(?!\d)")


def cifras_copiadas(texto: str, patron: re.Pattern[str]) -> set[str]:
    """Cifras del patron que aparecen en el texto, normalizadas a la forma espanola.

    Args:
        texto: Contenido del documento.
        patron: Buscador de :func:`buscador`.

    Returns:
        Las cifras encontradas, siempre con coma decimal.
    """
    return {m.group("cifra").replace(".", ",") for m in patron.finditer(texto)}


def _ambito(docs: Path | None, extra: list[Path] | None) -> list[tuple[Path, str]]:
    """Raices y patrones a vigilar, con las de la linea de ordenes si se dan.

    ``--docs`` y ``--extra`` existen para poder probar el gate en negativo sobre un directorio
    temporal, y quitarlos al ensanchar el alcance habria dejado al control sin su propia prueba.

    Args:
        docs: Raiz unica a vigilar, o ``None`` para el ambito por defecto.
        extra: Raices adicionales.

    Returns:
        Pares ``(raiz, patron glob)``.
    """
    if docs is None and not extra:
        return [(REPO_ROOT / raiz, patron) for raiz, patron in _AMBITO]
    raices = [*([docs] if docs is not None else []), *(extra or [])]
    return [(raiz, patron) for raiz in raices for patron in ("**/*.md", "**/*.tex")]


def rutas_por_estado(ledger: Path, estado: str) -> list[str]:
    """Artefact paths whose ledger row carries a given state.

    Args:
        ledger: Path to the custody ledger.
        estado: State to select.

    Returns:
        The matching artefact paths, in ledger order.
    """
    salida: list[str] = []
    for linea in ledger.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|") or re.match(r"^\|\s*[-:]+\s*\|", linea):
            continue
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) < MIN_CELDAS or celdas[5] != estado:
            continue
        ruta = CODE_RE.search(celdas[1])
        if ruta is not None:
            salida.append(ruta.group(1))
    return salida


def rutas_obsoletas(ledger: Path) -> list[str]:
    """Artefact paths whose ledger row is marked ``OBSOLETO``."""
    return rutas_por_estado(ledger, STALE_STATE)


def cifras_vigiladas(ledger: Path) -> tuple[dict[str, str], int]:
    """Figures that are distinctive of an obsolete artefact, with the artefact that owns each.

    Es UNA sola definicion, y por eso vive aqui: el gate y sus tests la calculaban por separado y
    se separaron en cuanto se anadio el descuento, con el resultado de que los tests elegian una
    cifra que el gate ya no vigilaba y no detectaban nada.

    Args:
        ledger: Path to the custody ledger.

    Returns:
        The watched figures mapped to their obsolete artefact, and how many were discounted.
    """
    origen: dict[str, str] = {}
    for relativo in rutas_obsoletas(ledger):
        for cifra in cifras_distintivas(REPO_ROOT / relativo):
            origen.setdefault(cifra, relativo)
    vigentes: set[str] = set()
    for relativo in rutas_por_estado(ledger, "SELLADO"):
        # Para DESCONTAR se redondea a cuatro decimales sin exigir minimo: un artefacto vigente
        # que guarda 0.025 y una tabla que escribe 0,0250 son el mismo numero, y comparar la
        # forma en vez del valor los daba por distintos.
        vigentes |= cifras_distintivas(REPO_ROOT / relativo, minimo=1, redondear=True)
    colisiones = set(origen) & vigentes
    for cifra in colisiones:
        del origen[cifra]
    return origen, len(colisiones)


def tiene_marca(texto: str, sufijo: str) -> bool:
    """Whether a document carries the quarantine banner."""
    return MARCA_HTML in texto if sufijo == ".html" else MARCA_MD in texto


def main() -> int:
    """Check every declared consumer and hunt for undeclared ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help=(
            "Raiz UNICA a vigilar, en vez del ambito por defecto. Existe para poder probar el "
            "gate en negativo sobre un directorio temporal."
        ),
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=None,
        help="Raices adicionales a vigilar. Por defecto, `paper/`.",
    )
    parser.add_argument(
        "--sitio",
        type=Path,
        default=REPO_ROOT.parent / "agrosat-micai-site",
        help="Raiz del cuaderno publico. Se omite si no esta en disco.",
    )
    args = parser.parse_args()

    obsoletas = rutas_obsoletas(args.ledger)
    if not obsoletas:
        print("no hay artefactos OBSOLETO: nada que vigilar")
        return 0
    print(f"artefactos OBSOLETO: {len(obsoletas)}")

    fallos: list[str] = []
    for relativo in CONSUMIDORES:
        ruta = REPO_ROOT / relativo
        if not ruta.exists():
            fallos.append(f"{relativo}: declarado consumidor y no existe")
            continue
        if not tiene_marca(ruta.read_text(encoding="utf-8"), ruta.suffix):
            fallos.append(
                f"{relativo}: cita cifras de artefactos OBSOLETO y no lleva la marca de cuarentena"
            )

    # El archivo ajeno se comprueba, no se cree: la exencion promete que esos documentos no se han
    # tocado, y un sello es la unica forma de que esa promesa signifique algo.
    for relativo, sello in ARCHIVO_AJENO.items():
        ruta = REPO_ROOT / relativo
        if not ruta.exists():
            fallos.append(f"{relativo}: exento como archivo ajeno y no esta en disco")
            continue
        digest = hashlib.md5(ruta.read_bytes()).hexdigest()  # noqa: S324 - sello de custodia
        if digest != sello:
            fallos.append(
                f"{relativo}: es un documento recibido y su MD5 cambio ({digest} frente a "
                f"{sello}). O se edito, que es lo que la exencion prometia no hacer, o hay que "
                "resellarlo con un motivo"
            )

    origen, descontadas = cifras_vigiladas(args.ledger)
    print(f"cifras distintivas vigiladas: {len(origen)}")
    print(f"  descontadas por aparecer tambien en artefactos vigentes: {descontadas}")

    # Las fuentes del manuscrito retirado se comprueban por sello, igual que el archivo ajeno: la
    # exencion promete que no se tocan, y esto lo verifica en vez de confiarlo.
    for relativo, sello in MANUSCRITO_RETIRADO.items():
        fichero = REPO_ROOT / relativo
        if not fichero.exists():
            fallos.append(f"{relativo}: declarado como manuscrito retirado y no esta en disco")
            continue
        digest = hashlib.md5(fichero.read_bytes()).hexdigest()  # noqa: S324 - sello de custodia
        if digest != sello:
            fallos.append(
                f"{relativo}: es fuente del manuscrito retirado y su MD5 cambio ({digest} frente "
                f"a {sello}). El manuscrito retirado no se edita; si se edito, deja de estar "
                "retirado y sus cifras vuelven a ser afirmaciones vigentes"
            )

    # Cualquier documento que nombre una ruta obsoleta, o reproduzca una de sus cifras, sin estar
    # declarado ni exento. El alcance NO es solo `docs/paper`: el manuscrito es `.tex`, los ADR
    # viven en `docs/decisions` y el plan en `context/`, y todos ellos quedaban fuera.
    declarados = set(CONSUMIDORES) | set(EXENTOS)
    buscador_tex = buscador(list(origen), ambas_formas=True)
    buscador_prosa = buscador(list(origen), ambas_formas=False)
    vigilados = sorted(
        {
            x
            for raiz, patron in _ambito(args.docs, args.extra)
            for x in raiz.glob(patron)
            if raiz.exists()
        }
    )
    print(f"documentos vigilados: {len(vigilados)}")
    for ruta in vigilados:
        absoluta = ruta.resolve()
        relativo = (
            str(absoluta.relative_to(REPO_ROOT))
            if absoluta.is_relative_to(REPO_ROOT)
            else str(absoluta)
        )
        if relativo in declarados or relativo in ARCHIVO_AJENO or relativo in MANUSCRITO_RETIRADO:
            continue
        texto = ruta.read_text(encoding="utf-8", errors="replace")
        citadas = [x for x in obsoletas if x in texto]
        if citadas:
            fallos.append(
                f"{relativo}: nombra {len(citadas)} artefacto(s) OBSOLETO y no esta declarado "
                f"como consumidor ni exento (p. ej. {citadas[0]})"
            )
        # En `.tex` valen las dos formas decimales: el manuscrito de envio se escribe en ingles.
        patron = buscador_tex if ruta.suffix == ".tex" else buscador_prosa
        colisiones = set(COLISIONES.get(relativo, ()))
        copiadas = sorted((cifras_copiadas(texto, patron) & set(origen)) - colisiones)
        if copiadas:
            fallos.append(
                f"{relativo}: reproduce {len(copiadas)} cifra(s) de artefactos OBSOLETO sin "
                f"cuarentena (p. ej. {copiadas[0]}, de {origen[copiadas[0]]})"
            )

    # El cuaderno publico es donde una cifra obsoleta hace mas dano, y no se escaneaba.
    if args.sitio.exists():
        paginas = sorted(args.sitio.glob("*.html"))
        print(f"paginas del cuaderno publico vigiladas: {len(paginas)}")
        for pagina in paginas:
            texto = pagina.read_text(encoding="utf-8")
            if MARCA_HTML in texto:
                continue
            copiadas = sorted(cifras_copiadas(texto, buscador_prosa) & set(origen))
            if copiadas:
                fallos.append(
                    f"{pagina.name}: reproduce {len(copiadas)} cifra(s) de artefactos OBSOLETO "
                    f"sin marca de cuarentena (p. ej. {copiadas[0]}, de {origen[copiadas[0]]})"
                )
    else:
        print(f"cuaderno publico no encontrado en {args.sitio}: no se vigila")

    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"paper-obsoletos-check: {len(fallos)} fallo(s)")
        return 1
    print(f"consumidores declarados con cuarentena: {len(CONSUMIDORES)}")
    print(f"documentos recibidos, exentos y sellados: {len(ARCHIVO_AJENO)}")
    print("paper-obsoletos-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
