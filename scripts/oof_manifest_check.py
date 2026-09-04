"""Falla si hay un parquet OOF en disco que el manifiesto no declara, o al reves.

El arnes OOF produjo una vez una cifra que era un bug, y el bug entro por ahi: un parquet
huerfano —de otra configuracion, de otra pasada, de otro `n_timesteps`— se lee igual de bien que
uno legitimo, y ninguna tabla distingue un fallo de un resultado.

Comprueba tres cosas, y la tercera es la que importa:

1. Todo parquet de `ml/eval/oof/` esta declarado en `manifest.json`.
2. Todo modelo declarado tiene sus parquets en disco, o dice por que no.
3. **Las rutas del manifiesto se comparan por NOMBRE DE FICHERO, no por ruta absoluta**: el
   manifiesto vigente trae rutas `C:\\Users\\...` de la maquina donde se genero, asi que comparar
   rutas completas daria un verde vacio en cualquier otra maquina. Un control que solo pasa donde
   se escribio no es un control.

Uso:
    poetry run python scripts/oof_manifest_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = REPO_ROOT / "ml" / "eval" / "oof"


def _basename(ruta: str) -> str:
    """Basename of a path written on any platform, Windows included.

    El manifiesto vigente guarda rutas absolutas de Windows. `Path(...).name` no las parte en
    POSIX, asi que se corta a mano por los dos separadores.
    """
    return ruta.replace("\\", "/").rsplit("/", 1)[-1]


def _declarados(manifiesto: dict[str, Any]) -> dict[str, str]:
    """Map every parquet basename the manifest declares to the model that declares it."""
    salida: dict[str, str] = {}
    for modelo, datos in manifiesto.get("models", {}).items():
        for clave in ("path", "parcel_path"):
            ruta = datos.get(clave)
            if ruta:
                salida[_basename(ruta)] = modelo
    return salida


def main() -> int:
    """Check the OOF manifest against what is on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()

    manifiesto_path = args.oof / "manifest.json"
    if not manifiesto_path.exists():
        print(f"ERROR: no existe el manifiesto {manifiesto_path}")
        return 2
    manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))

    declarados = _declarados(manifiesto)
    en_disco = {p.name for p in args.oof.glob("*.parquet")}

    fallos: list[str] = []
    for nombre in sorted(en_disco - set(declarados)):
        fallos.append(f"{nombre}: esta en disco y el manifiesto no lo declara")
    for nombre in sorted(set(declarados) - en_disco):
        modelo = declarados[nombre]
        estado = manifiesto["models"][modelo].get("status")
        motivo = manifiesto["models"][modelo].get("reason")
        if estado == "ok":
            fallos.append(f"{nombre}: el manifiesto lo declara con status ok y no esta en disco")
        else:
            print(f"  ausente y declarado: {nombre} (status={estado}, motivo={motivo})")

    print(f"parquets en disco: {len(en_disco)}")
    print(f"parquets declarados en el manifiesto: {len(declarados)}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"oof-manifest-check: {len(fallos)} fallo(s)")
        return 1
    print("oof-manifest-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
