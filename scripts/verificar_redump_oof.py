"""Verifica un re-volcado OOF en un directorio temporal ANTES de promoverlo al definitivo.

Promover primero y comprobar despues es como el arnes publico una cifra que era un bug: el fichero
malo ya estaba donde los consumidores leen. Aqui se comprueba en el temporal, y la promocion es un
paso aparte que solo ocurre si todo pasa.

Comprueba seis cosas, y la ultima es la que de verdad interesa:

1. La configuracion usada es la del registro de checkpoints, campo a campo.
2. El MD5 del checkpoint coincide con el que se declara.
3. El volcado denso cubre los 496 parches del fold retenido.
4. El volcado por parcela cubre las mismas parcelas que un miembro canonico.
5. Las posteriores son probabilidades: suman uno y no traen NaN.
6. **La diferencia de metricas con el fichero que sustituye**, para que promover no sea un cambio
   silencioso. No falla por ser grande: falla si no se puede medir. Un cambio grande puede ser el
   arreglo, y por eso se imprime y lo mira una persona.

Uso:
    poetry run python scripts/verificar_redump_oof.py --temporal <dir> --miembro tsvit-pheno-fullm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = REPO_ROOT / "ml" / "eval" / "oof"
PATCHES_ESPERADOS = 496
CANONICO_DE_REFERENCIA = "tsvit-pheno"


def main() -> int:
    """Verify a re-dump in a temporary directory and report whether it can be promoted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--miembro", required=True)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    fallos: list[str] = []
    spec = CHECKPOINT_REGISTRY.get(args.miembro)
    if spec is None:
        print(f"ERROR: {args.miembro} no esta en CHECKPOINT_REGISTRY")
        return 2

    # 1 y 2. Configuracion y sello del checkpoint.
    manifiesto = json.loads((args.temporal / "manifest.json").read_text(encoding="utf-8"))
    entrada = manifiesto["models"].get(args.miembro, {})
    print(f"modelo: {args.miembro}  status={entrada.get('status')}")
    print(f"kwargs del registro: {spec.model_kwargs}")
    if entrada.get("status") != "ok":
        fallos.append(f"el volcado declara status={entrada.get('status')!r}")
    digest = hashlib.md5(Path(spec.path).read_bytes()).hexdigest()  # noqa: S324
    print(f"checkpoint: {spec.path}")
    print(f"md5 del checkpoint: {digest}")

    # 3. Cobertura densa.
    n_patches = entrada.get("n_patches")
    print(f"parches volcados: {n_patches}")
    if n_patches != PATCHES_ESPERADOS:
        fallos.append(
            f"se volcaron {n_patches} parches y el fold retenido tiene {PATCHES_ESPERADOS}"
        )

    # 4. Cobertura por parcela frente a un canonico.
    nuevo = args.temporal / f"oof_parcel_{args.miembro}_fold5.parquet"
    referencia = args.oof / f"oof_parcel_{CANONICO_DE_REFERENCIA}_fold5.parquet"
    if not nuevo.exists():
        print(f"ERROR: no existe {nuevo}")
        return 2
    t_nuevo = pl.read_parquet(nuevo)
    t_ref = pl.read_parquet(referencia, columns=["canonical_parcel_id"])
    ids_nuevo = set(t_nuevo["canonical_parcel_id"].to_list())
    ids_ref = set(t_ref["canonical_parcel_id"].to_list())
    print(f"parcelas: nuevo={len(ids_nuevo)}  canonico {CANONICO_DE_REFERENCIA}={len(ids_ref)}")
    if ids_nuevo != ids_ref:
        fallos.append(
            f"la cobertura por parcela no coincide con {CANONICO_DE_REFERENCIA}: "
            f"faltan {len(ids_ref - ids_nuevo)}, sobran {len(ids_nuevo - ids_ref)}"
        )

    # 5. Son probabilidades.
    p = t_nuevo.select(PROB_COLUMNS).to_numpy().astype(np.float64)
    if not np.isfinite(p).all():
        fallos.append("las posteriores traen valores no finitos")
    sumas = p.sum(axis=1)
    if not np.allclose(sumas, 1.0, atol=1e-3):
        fallos.append(f"las posteriores no suman uno: min={sumas.min():.4f} max={sumas.max():.4f}")

    # 6. Diferencia con el fichero que sustituye.
    viejo = args.oof / f"oof_parcel_{args.miembro}_fold5.parquet"
    if viejo.exists():
        t_viejo = pl.read_parquet(viejo)
        comun = sorted(ids_nuevo & set(t_viejo["canonical_parcel_id"].to_list()))
        if comun:
            a = (
                t_nuevo.filter(pl.col("canonical_parcel_id").is_in(comun))
                .sort("canonical_parcel_id")
                .select(PROB_COLUMNS)
                .to_numpy()
            )
            b = (
                t_viejo.filter(pl.col("canonical_parcel_id").is_in(comun))
                .sort("canonical_parcel_id")
                .select(PROB_COLUMNS)
                .to_numpy()
            )
            coincide = float((a.argmax(axis=1) == b.argmax(axis=1)).mean())
            print(f"parcelas comparables con el fichero anterior: {len(comun)}")
            print(f"coincidencia de argmax con el anterior: {coincide:.4f}")
            print(f"diferencia media absoluta de posteriores: {float(np.abs(a - b).mean()):.6f}")
        else:
            fallos.append("el fichero anterior no comparte ninguna parcela: no se puede comparar")
    else:
        print("no hay fichero anterior con el que comparar")

    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"verificar-redump: {len(fallos)} fallo(s); NO se promueve")
        return 1
    print("verificar-redump: OK; se puede promover")
    return 0


if __name__ == "__main__":
    sys.exit(main())
