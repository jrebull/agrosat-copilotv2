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
2. Toda entrada del inventario **existe**: si el parquet esta en disco se comprueban sus bytes; si
   no esta, se comprueba su puntero `.dvc`. **Un clon limpio no trae los parquet, solo los
   punteros**, y un gate que exigiera los bytes seria rojo en CI para siempre — o sea, un gate que
   solo se puede correr en la maquina donde se escribio, que es lo que ya paso con el manifiesto
   de rutas de Windows.
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
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = REPO_ROOT / "ml" / "eval" / "oof"
ESTADOS = ("canonical", "legacy_unverified", "excluded")
TEMPORAL_KINDS = frozenset({"tsvit", "tsvit-pheno", "tsvit-pheno-fullm", "utae", "anysat"})


def _verificar_puntero(puntero: Path, nombre: str, entrada: dict[str, object]) -> list[str]:
    """Check the inventory against a DVC pointer when the parquet itself is not on disk.

    En un clon limpio los parquet no estan: solo estan los `.dvc`. El puntero registra el MD5 y el
    tamano del blob, que es exactamente lo que el inventario declara, asi que la comprobacion es la
    misma con otra fuente.

    Args:
        puntero: Path to the ``.dvc`` pointer.
        nombre: Parquet file name, for the messages.
        entrada: The inventory entry.

    Returns:
        Zero or more failure messages.
    """
    texto = puntero.read_text(encoding="utf-8")
    md5 = re.search(r"^\s*-?\s*md5:\s*([0-9a-f]{32})\s*$", texto, re.M)
    size = re.search(r"^\s*size:\s*(\d+)\s*$", texto, re.M)
    if md5 is None:
        return [f"{nombre}: el puntero .dvc no trae MD5"]
    fallos: list[str] = []
    if md5.group(1) != entrada.get("md5"):
        fallos.append(
            f"{nombre}: el puntero .dvc registra {md5.group(1)} y el inventario "
            f"{entrada.get('md5')}"
        )
    if size is not None and int(size.group(1)) != entrada.get("bytes"):
        fallos.append(
            f"{nombre}: el puntero .dvc registra {size.group(1)} bytes y el inventario "
            f"{entrada.get('bytes')}"
        )
    return fallos


def _check_run_manifest(manifest: dict[str, object]) -> list[str]:
    """Validate effective configuration and provenance in schema-v2 run manifests.

    Version 1 is retained as historical evidence and is protected by the custody
    ledger. Version 2 makes every successful entry self-contained so appending a
    model from another run or omitting the effective temporal configuration fails.
    """
    if manifest.get("schema_version") != 2:
        return []
    failures: list[str] = []
    code_version = manifest.get("code_version")
    data_version = manifest.get("data_version")
    models = manifest.get("models")
    if not isinstance(models, dict):
        return ["manifest.json v2: models no es un objeto"]
    for model, raw_entry in models.items():
        if not isinstance(raw_entry, dict) or raw_entry.get("status") != "ok":
            continue
        if raw_entry.get("code_version") != code_version:
            failures.append(f"{model}: code_version de la entrada no coincide con el de la corrida")
        if raw_entry.get("data_version") != data_version:
            failures.append(f"{model}: data_version de la entrada no coincide con el de la corrida")
        if raw_entry.get("model_kind") in TEMPORAL_KINDS:
            dataset_steps = raw_entry.get("n_timesteps_dataset")
            model_steps = raw_entry.get("n_timesteps_model_spec")
            if dataset_steps is None or model_steps is None:
                failures.append(
                    f"{model}: la entrada temporal no declara ambas configuraciones n_timesteps"
                )
            elif dataset_steps != model_steps:
                failures.append(
                    f"{model}: n_timesteps_dataset={dataset_steps} y "
                    f"n_timesteps_model_spec={model_steps} no coinciden"
                )
    return failures


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
    # Se busca en TODO el arbol, no solo en la raiz. El flujo de re-volcado trabaja en directorios
    # temporales antes de promover, y uno olvidado bajo `oof/` era invisible para este gate
    # mientras cualquier consumidor que recorriera el arbol podia leerlo.
    todos = [p for p in args.oof.rglob("*.parquet") if "__pycache__" not in p.parts]
    en_subdirectorio = [p for p in todos if p.parent != args.oof]
    en_disco = {p.name for p in todos if p.parent == args.oof}

    fallos: list[str] = []
    for extraviado in sorted(en_subdirectorio):
        fallos.append(
            f"{extraviado.relative_to(args.oof)}: hay un parquet en un subdirectorio de `oof/`. "
            "El inventario se indexa por NOMBRE, asi que un fichero ahi no se puede declarar sin "
            "ambiguedad; o se promueve a la raiz y se declara, o se saca de aqui"
        )
    for nombre in sorted(en_disco - set(declarados)):
        fallos.append(f"{nombre}: esta en disco y el inventario no lo declara")

    por_bytes = 0
    por_puntero = 0
    for nombre in sorted(declarados):
        entrada = declarados[nombre]
        estado = entrada.get("estado")
        if estado not in ESTADOS:
            fallos.append(f"{nombre}: estado {estado!r} desconocido")
        if estado == "legacy_unverified" and not entrada.get("siguiente_paso"):
            fallos.append(
                f"{nombre}: es legacy_unverified y no declara siguiente_paso; sin salida "
                "escrita, un estado temporal se vuelve permanente"
            )

        parquet = args.oof / nombre
        puntero = args.oof / f"{nombre}.dvc"
        if parquet.exists():
            por_bytes += 1
            digest = hashlib.md5(parquet.read_bytes()).hexdigest()  # noqa: S324
            if digest != entrada.get("md5"):
                fallos.append(
                    f"{nombre}: MD5 {digest} y el inventario registra {entrada.get('md5')}"
                )
        elif puntero.exists():
            por_puntero += 1
            fallos.extend(_verificar_puntero(puntero, nombre, entrada))
        else:
            fallos.append(
                f"{nombre}: el inventario lo declara y no esta ni el parquet ni su puntero .dvc"
            )

    # Las dos fuentes no pueden separarse en silencio.
    manifiesto_path = args.oof / "manifest.json"
    if manifiesto_path.exists():
        manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))
        fallos.extend(_check_run_manifest(manifiesto))
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
    print(
        f"declarados: {len(declarados)}  (por bytes: {por_bytes}, por puntero .dvc: {por_puntero})"
    )
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
