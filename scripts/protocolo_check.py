"""Impide congelar el protocolo de US-172 con campos operativos sin rellenar.

El protocolo de elicitacion solo vale si, antes del primer reclutamiento, estan escritos quien es
el investigador responsable, quien custodia las respuestas, donde viven cifradas, quien puede
leerlas, cuando se destruyen las grabaciones, como se atiende una peticion de retirada, quien
transcribe y cual es la determinacion institucional.

Son decisiones de personas y no de repositorio, asi que el codigo no puede rellenarlas. Lo que si
puede es **impedir que el documento se declare CONGELADO mientras falte alguna**, que es donde este
proyecto se ha equivocado siete rondas seguidas: un aviso en prosa no impide nada, un estado
ejecutable si.

Reglas:

1. Si el estado es CONGELADO, ningun campo operativo puede decir ``[POR DEFINIR]``.
2. Si el estado es CONGELADO, la referencia de la determinacion institucional tiene que estar.
3. Las tres permutaciones de lectura y su semilla tienen que estar escritas, en cualquier estado:
   generarlas despues de empezar a entrevistar seria elegirlas viendo a quien se entrevista.

Uso:
    poetry run python scripts/protocolo_check.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOLO = REPO_ROOT / "docs" / "paper" / "perdidas-protocolo.md"
SIN_RELLENAR = "[POR DEFINIR]"
CONGELADO = "CONGELADO"
CAMPOS_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*`?([^|`]*)`?\s*\|\s*$", re.M)
PERMUTACION_RE = re.compile(r"^\|\s*`P([123])`\s*\|", re.M)
SEMILLA_RE = re.compile(r"semilla\s+`(\d+)`")
DETERMINACION = "Referencia de la determinación o aprobación institucional"


def main() -> int:
    """Check the protocol's freeze preconditions and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocolo", type=Path, default=DEFAULT_PROTOCOLO)
    args = parser.parse_args()

    if not args.protocolo.exists():
        print(f"ERROR: no existe el protocolo {args.protocolo}")
        return 2
    texto = args.protocolo.read_text(encoding="utf-8")
    fallos: list[str] = []

    # 3. Las permutaciones y su semilla, en cualquier estado.
    permutaciones = sorted(set(PERMUTACION_RE.findall(texto)))
    if permutaciones != ["1", "2", "3"]:
        fallos.append(
            f"faltan permutaciones de lectura: se encontraron {permutaciones or 'ninguna'} y hacen "
            "falta P1, P2 y P3, escritas antes de la primera entrevista"
        )
    if SEMILLA_RE.search(texto) is None:
        fallos.append("las permutaciones no declaran la semilla con la que se generaron")

    congelado = bool(re.search(rf"\*\*Estado\*\*:\s*{CONGELADO}", texto))
    pendientes = [
        clave.strip("* ")
        for clave, valor in CAMPOS_RE.findall(texto)
        if SIN_RELLENAR in valor or valor.strip() == SIN_RELLENAR.strip("[]")
    ]
    if SIN_RELLENAR in texto and not pendientes:
        # La tabla cambio de forma y el parser dejo de verla: eso es un fallo del control.
        fallos.append(
            "el documento contiene campos sin rellenar y el parser no los reconoce: la tabla de "
            "campos operativos cambio de forma y este gate dejo de mirarla"
        )

    if congelado:
        if pendientes:
            fallos.append(
                f"el protocolo se declara {CONGELADO} con {len(pendientes)} campo(s) operativo(s) "
                f"sin rellenar: {', '.join(pendientes)}"
            )
        if DETERMINACION not in texto:
            fallos.append(
                f"el protocolo se declara {CONGELADO} y no cita la determinacion institucional"
            )
        print(f"estado: {CONGELADO}")
    else:
        print("estado: BORRADOR")
        print(f"campos operativos sin rellenar: {len(pendientes)}")
        for campo in pendientes:
            print(f"  - {campo}")

    print(f"permutaciones de lectura registradas: {len(permutaciones)}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"protocolo-check: {len(fallos)} fallo(s)")
        return 1
    print("protocolo-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
