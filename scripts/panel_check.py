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

Uso:
    poetry run python scripts/panel_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "docs" / "paper" / "panel-v1.json"
DEFAULT_PREREGISTRO = REPO_ROOT / "docs" / "paper" / "preregistro-v2-borrador.md"
DEFAULT_INVENTARIO = REPO_ROOT / "ml" / "eval" / "oof" / "inventario.json"


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

    texto = args.preregistro.read_text(encoding="utf-8")
    seccion = texto[texto.find("### 4.6") :] if "### 4.6" in texto else ""
    if not seccion:
        fallos.append("el preregistro no tiene seccion 4.6: el panel no esta declarado en prosa")
    else:
        seccion = seccion[: seccion.find("\n## ")] if "\n## " in seccion else seccion
        for nombre in nombres:
            if nombre not in seccion:
                fallos.append(f"{nombre}: esta en el panel y la seccion 4.6 no lo nombra")

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
