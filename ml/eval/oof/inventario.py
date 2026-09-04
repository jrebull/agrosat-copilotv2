"""Inventario global de los OOF y su estado, separado del manifiesto de la corrida densa.

`manifest.json` es el manifiesto de **una** corrida de volcado denso, y quien la ejecuta lo
reescribe entero: usarlo como registro global hace que re-volcar un solo modelo borre logicamente
la declaracion de los demas. Este inventario no lo escribe ningun volcado.

Tres estados, y el del medio es el que importa:

- ``canonical``: procedencia declarada y verificable. Los consumidores MICAI lo leen.
- ``legacy_unverified``: existe y se ha usado, sin procedencia verificada. Los consumidores MICAI
  lo **rechazan**. Declarar que algo no esta verificado no impide que se lea, y este proyecto ya
  aprendio en el ledger que un aviso en prosa no impide nada: hace falta el estado ejecutable.
- ``excluded``: decidido formalmente que no entra, con su motivo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "ESTADOS",
    "EstadoNoCanonicoError",
    "cargar_inventario",
    "estado_de_miembro",
    "exigir_canonicos",
]

#: Estados admitidos por el inventario.
ESTADOS: tuple[str, ...] = ("canonical", "legacy_unverified", "excluded")

_DEFECTO = Path(__file__).resolve().parent / "inventario.json"


class EstadoNoCanonicoError(RuntimeError):
    """Raised when a MICAI consumer asks for a member that is not canonical."""


def cargar_inventario(ruta: Path | None = None) -> dict[str, Any]:
    """Load the global OOF inventory.

    Args:
        ruta: Inventory path. Defaults to the one next to this module.

    Returns:
        The parsed inventory.
    """
    datos: dict[str, Any] = json.loads((ruta or _DEFECTO).read_text(encoding="utf-8"))
    return datos


def estado_de_miembro(miembro: str, inventario: dict[str, Any] | None = None) -> str:
    """State of a member's parcel-level OOF.

    Args:
        miembro: Member name, as used by the MICAI scripts.
        inventario: Preloaded inventory, or ``None`` to load the default one.

    Returns:
        One of :data:`ESTADOS`, or ``"unknown"`` when the member is not in the inventory.
    """
    datos = inventario if inventario is not None else cargar_inventario()
    entrada = datos["ficheros"].get(f"oof_parcel_{miembro}_fold5.parquet")
    return str(entrada["estado"]) if entrada else "unknown"


def exigir_canonicos(miembros: list[str] | tuple[str, ...]) -> None:
    """Refuse to proceed when any requested member is not canonical.

    Args:
        miembros: Members a MICAI consumer is about to read.

    Raises:
        EstadoNoCanonicoError: naming every offending member, its state and its reason.
    """
    inventario = cargar_inventario()
    problemas: list[str] = []
    for miembro in miembros:
        estado = estado_de_miembro(miembro, inventario)
        if estado == "canonical":
            continue
        entrada = inventario["ficheros"].get(f"oof_parcel_{miembro}_fold5.parquet", {})
        motivo = entrada.get("motivo", "no esta en el inventario")
        problemas.append(f"{miembro} ({estado}): {motivo}")
    if problemas:
        raise EstadoNoCanonicoError(
            "el analisis MICAI solo lee OOF canonicos y se han pedido "
            f"{len(problemas)} que no lo son:\n  - " + "\n  - ".join(problemas)
        )
