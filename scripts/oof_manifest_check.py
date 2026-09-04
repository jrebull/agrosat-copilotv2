"""Comprueba el inventario global de los OOF: nada sin declarar, nada declarado sin estar.

El arnes OOF produjo una vez una cifra que era un bug, y el bug entro por ahi: un parquet
huerfano —de otra configuracion, de otra pasada, de otro `n_timesteps`— se lee igual de bien que
uno legitimo, y ninguna tabla distingue un fallo de un resultado.

**Este gate NO mira `manifest.json` como registro global, y ese es el punto.** `manifest.json` es
el manifiesto de UNA corrida de volcado denso, y `dump_oof` lo reescribe entero con los modelos de
esa corrida: re-volcar un solo modelo borraria logicamente la declaracion de los otros seis. El
registro global es `inventario.json`, que ningun volcado escribe.

Comprueba, en este orden:

1. Todo parquet en disco esta declarado en el inventario.
2. Toda entrada del inventario esta en disco y su MD5 coincide.
3. Todo estado es uno de los tres admitidos.
4. Todo `legacy_unverified` declara su `siguiente_paso`: sin salida escrita, un estado temporal
   se vuelve permanente.
5. Todo modelo que `manifest.json` da por `ok` es `canonical` en el inventario, para que las dos
   fuentes no puedan separarse en silencio.

Uso:
    poetry run python scripts/oof_manifest_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = REPO_ROOT / "ml" / "eval" / "oof"
ESTADOS = ("canonical", "legacy_unverified", "excluded")


def main() -> int:
    """Check the global OOF inventory against disk and against the dense manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()

    inventario_path = args.oof / "inventario.json"
    if not inventario_path.exists():
        print(f"ERROR: no existe el inventario {inventario_path}")
        return 2
    inventario = json.loads(inventario_path.read_text(encoding="utf-8"))
    declarados: dict[str, dict[str, object]] = inventario["ficheros"]
    en_disco = {p.name for p in args.oof.glob("*.parquet")}

    fallos: list[str] = []
    for nombre in sorted(en_disco - set(declarados)):
        fallos.append(f"{nombre}: esta en disco y el inventario no lo declara")
    for nombre in sorted(set(declarados) - en_disco):
        fallos.append(f"{nombre}: el inventario lo declara y no esta en disco")

    for nombre in sorted(set(declarados) & en_disco):
        entrada = declarados[nombre]
        estado = entrada.get("estado")
        if estado not in ESTADOS:
            fallos.append(f"{nombre}: estado {estado!r} desconocido")
        digest = hashlib.md5((args.oof / nombre).read_bytes()).hexdigest()  # noqa: S324
        if digest != entrada.get("md5"):
            fallos.append(f"{nombre}: MD5 {digest} y el inventario registra {entrada.get('md5')}")
        if estado == "legacy_unverified" and not entrada.get("siguiente_paso"):
            fallos.append(
                f"{nombre}: es legacy_unverified y no declara siguiente_paso; sin salida "
                "escrita, un estado temporal se vuelve permanente"
            )

    # Las dos fuentes no pueden separarse en silencio.
    manifiesto_path = args.oof / "manifest.json"
    if manifiesto_path.exists():
        manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))
        for modelo, datos in manifiesto.get("models", {}).items():
            if datos.get("status") != "ok":
                continue
            entrada = declarados.get(f"oof_parcel_{modelo}_fold5.parquet", {})
            if entrada.get("estado") != "canonical":
                fallos.append(
                    f"{modelo}: manifest.json lo da por ok y el inventario lo tiene como "
                    f"{entrada.get('estado', 'no declarado')}"
                )

    por_estado: dict[str, int] = {}
    for entrada in declarados.values():
        clave = str(entrada.get("estado"))
        por_estado[clave] = por_estado.get(clave, 0) + 1
    print(f"parquets en disco: {len(en_disco)}")
    for estado in ESTADOS:
        print(f"  {estado}: {por_estado.get(estado, 0)}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"oof-manifest-check: {len(fallos)} fallo(s)")
        return 1
    print("oof-manifest-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
