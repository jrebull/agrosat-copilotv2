"""Comprueba que el panel congelado sea coherente con el inventario y con el preregistro.

La lista de predictores se fija antes de calcular nada. Un panel elegido despues de ver resultados
no es un panel: es un ganador con otro nombre. Este gate no impide elegir mal —eso lo impide
haberlo escrito antes— pero si impide que el panel se contradiga con lo que el proyecto ya sabe:

1. Ningun miembro del panel esta `excluded` ni `legacy_unverified` en `ml/eval/oof/inventario.json`.
   Meter en el panel algo que el analisis no puede leer es una contradiccion que solo se descubre
   al correrlo.
2. Las familias declaradas llegan al minimo. Con cinco miembros y dos de la misma familia el
   margen es estrecho, y conviene que sea el gate quien avise cuando deje de haberlo.
3. Ningun miembro esta a la vez dentro y fuera del panel.
4. No hay campeon declarado: el predictor es un factor de sensibilidad.
5. La seccion 4.6 del preregistro nombra los mismos miembros. Dos fuentes que dicen lo mismo se
   separan, y ya paso con el estimando.
6. Ningun guion del articulo mantiene su propia lista de miembros en vez de leer este fichero.
7. La prosa coloca a cada miembro del lado que le toca. Comprobar que el nombre APARECE no
   comprueba nada: la seccion 4.6 llego a decir que `segformer` "se excluye" mientras el panel
   congelado lo tenia dentro, y el gate pasaba porque el nombre estaba.

Uso:
    poetry run python scripts/panel_check.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "docs" / "paper" / "panel-v1.json"
DEFAULT_PREREGISTRO = REPO_ROOT / "docs" / "paper" / "preregistro-v2-borrador.md"
DEFAULT_INVENTARIO = REPO_ROOT / "ml" / "eval" / "oof" / "inventario.json"


#: Formas con las que la prosa dice que un miembro esta fuera. Es la lista completa: si aparece
#: una nueva, el gate deja de reconocerla y hay que anadirla aqui, no rodearla.
MARCAS_DE_EXCLUSION: tuple[str, ...] = (
    "se excluye",
    "excluido",
    "excluida",
    "exclusion",
    "queda fuera",
    "quedan fuera",
    "sale por",
    "salen por",
    "fuera del panel",
)

#: Formas de decir, en el mismo parrafo, que el miembro nombrado sigue dentro.
MARCAS_DE_PERMANENCIA: tuple[str, ...] = (
    "dentro del panel",
    "permanece en el panel",
    "sigue en el panel",
    "se mantiene en el panel",
)


def _frases(seccion: str) -> list[str]:
    """Frases de una seccion en markdown con envoltura dura de linea.

    Args:
        seccion: Texto de la seccion.

    Returns:
        Frases, con los saltos de linea de la envoltura ya deshechos.
    """
    frases: list[str] = []
    for parrafo in seccion.split("\n\n"):
        unido = parrafo.replace("\n", " ")
        frases.extend(f.strip() for f in re.split(r"(?<=[.:;])\s+", unido) if f.strip())
    return frases


def _sin_acentos(texto: str) -> str:
    """Texto en minusculas y sin diacriticos, para comparar marcas."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _tabla_del_panel(seccion: str) -> list[str]:
    """Miembros listados en la tabla de la seccion 4.6, en orden.

    Args:
        seccion: Texto de la seccion.

    Returns:
        Nombres tal y como aparecen en la primera columna.
    """
    nombres: list[str] = []
    for linea in seccion.splitlines():
        if not linea.startswith("|"):
            continue
        celdas = linea.strip().strip("|").split("|")
        celda = re.search(r"`([^`]+)`", celdas[0])
        if celda is not None:
            nombres.append(celda.group(1))
    return nombres


def _revisar_prosa(seccion: str, dentro: list[str], fuera: list[str]) -> list[str]:
    """Comprobar que la prosa coloca a cada miembro del lado que le toca.

    Comprobar que el nombre APARECE no comprueba nada: la seccion 4.6 llego a decir que
    ``segformer`` "se excluye" mientras el panel congelado lo tenia dentro, y el gate pasaba
    porque el nombre estaba. La exclusion iba dos frases mas abajo, con el sujeto implicito, asi
    que mirar la frase tampoco basta: se mira el PARRAFO, y la regla editorial es que si en el
    mismo parrafo conviven un miembro del panel y una marca de exclusion, hay que decir con
    todas las letras que ese miembro sigue dentro.

    Args:
        seccion: Texto de la seccion 4.6.
        dentro: Miembros del panel.
        fuera: Miembros excluidos.

    Returns:
        Fallos encontrados.
    """
    fallos: list[str] = []

    tabla = _tabla_del_panel(seccion)
    if sorted(tabla) != sorted(dentro):
        fallos.append(
            f"la tabla de la seccion 4.6 lista {tabla} y el panel congelado {sorted(dentro)}"
        )
    repetidos = sorted({n for n in tabla if tabla.count(n) > 1})
    if repetidos:
        fallos.append(f"la tabla de la seccion 4.6 repite filas: {', '.join(repetidos)}")

    parrafos = [p.replace("\n", " ") for p in seccion.split("\n\n")]
    for parrafo in parrafos:
        plano = _sin_acentos(parrafo)
        marcas = [m for m in MARCAS_DE_EXCLUSION if m in plano]
        if not marcas:
            continue
        if any(m in plano for m in MARCAS_DE_PERMANENCIA):
            continue
        for nombre in dentro:
            if f"`{nombre}`" in parrafo:
                fallos.append(
                    f"{nombre}: esta DENTRO del panel y la seccion 4.6 lo nombra en un parrafo "
                    f"que habla de exclusion ({marcas[0]!r}) sin decir que sigue dentro"
                )

    # Los excluidos se miran por FRASE, no por parrafo: el parrafo de "quien queda fuera" nombra
    # a varios, y con alcance de parrafo bastaba con que UNO llevara la marca para dar por
    # declarados a todos. La direccion contraria necesita el parrafo porque el sujeto va implicito.
    frases = _frases(seccion)
    for nombre in fuera:
        nombra = [f for f in frases if f"`{nombre}`" in f]
        if not nombra:
            fallos.append(f"{nombre}: esta FUERA del panel y la seccion 4.6 no lo nombra")
            continue
        if not any(m in _sin_acentos(p) for p in nombra for m in MARCAS_DE_EXCLUSION):
            fallos.append(
                f"{nombre}: esta FUERA del panel y la seccion 4.6 lo nombra sin decir que esta "
                "fuera"
            )
    return fallos


def main() -> int:
    """Check the frozen panel against the inventory and the pre-registration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--preregistro", type=Path, default=DEFAULT_PREREGISTRO)
    parser.add_argument("--inventario", type=Path, default=DEFAULT_INVENTARIO)
    args = parser.parse_args()

    if not args.panel.exists():
        print(f"ERROR: no existe el panel {args.panel}")
        return 2
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    inventario = json.loads(args.inventario.read_text(encoding="utf-8"))["ficheros"]
    fallos: list[str] = []

    nombres = [m["nombre"] for m in panel["miembros"]]
    for nombre in nombres:
        entrada = inventario.get(f"oof_parcel_{nombre}_fold5.parquet")
        estado = entrada.get("estado") if entrada else "no declarado"
        if estado != "canonical":
            fallos.append(
                f"{nombre}: esta en el panel y su OOF es {estado}; el analisis no puede leerlo"
            )

    familias = {m["familia"] for m in panel["miembros"]}
    minimo = int(panel["minimo_familias_exigido"])
    if len(familias) < minimo:
        fallos.append(
            f"el panel declara {len(familias)} familias distintas y el minimo es {minimo}: "
            f"{sorted(familias)}"
        )
    if int(panel.get("familias_distintas", -1)) != len(familias):
        fallos.append(
            f"el panel dice tener {panel.get('familias_distintas')} familias y sus miembros dan "
            f"{len(familias)}"
        )

    fuera = {m["nombre"] for m in panel["fuera_del_panel"]}
    solapan = sorted(set(nombres) & fuera)
    if solapan:
        fallos.append(f"estan dentro y fuera del panel a la vez: {', '.join(solapan)}")

    if panel.get("campeon_declarado") is not None:
        fallos.append(
            f"el panel declara un campeon ({panel['campeon_declarado']!r}); el predictor es un "
            "factor de sensibilidad y no se declara ganador"
        )

    # 6. Ningun guion mantiene su propia lista de miembros. Una lista escrita dos veces se separa:
    # al congelar el panel en cinco, fase 2 y fase 3 seguian pidiendo los diez originales.
    import ast

    for guion in sorted(Path(REPO_ROOT / "scripts").glob("run_paper_micai_*.py")):
        arbol = ast.parse(guion.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.AnnAssign) or not isinstance(nodo.target, ast.Name):
                continue
            if nodo.target.id != "ALL_MEMBERS" or not isinstance(nodo.value, ast.Tuple):
                continue
            literal = tuple(e.value for e in nodo.value.elts if isinstance(e, ast.Constant))
            if literal:
                fallos.append(
                    f"{guion.name}: mantiene su propia lista de {len(literal)} miembros en vez de "
                    "leer el panel congelado"
                )

    texto = args.preregistro.read_text(encoding="utf-8")
    seccion = texto[texto.find("### 4.6") :] if "### 4.6" in texto else ""
    if not seccion:
        fallos.append("el preregistro no tiene seccion 4.6: el panel no esta declarado en prosa")
    else:
        seccion = seccion[: seccion.find("\n## ")] if "\n## " in seccion else seccion
        for nombre in nombres:
            if nombre not in seccion:
                fallos.append(f"{nombre}: esta en el panel y la seccion 4.6 no lo nombra")
        fallos.extend(_revisar_prosa(seccion, nombres, sorted(fuera)))

    print(f"miembros del panel: {len(nombres)}")
    print(f"familias distintas: {len(familias)} (minimo {minimo})")
    print(f"margen sobre el minimo: {len(familias) - minimo}")
    print(f"fuera del panel, con motivo: {len(fuera)}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"panel-check: {len(fallos)} fallo(s)")
        return 1
    print("panel-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
