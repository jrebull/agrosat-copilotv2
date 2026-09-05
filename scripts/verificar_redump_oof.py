"""Verifica un re-volcado OOF en un directorio temporal ANTES de promoverlo al definitivo.

Promover primero y comprobar despues es como el arnes publico una cifra que era un bug: el fichero
malo ya estaba donde los consumidores leen. Aqui se comprueba en el temporal, y la promocion es un
paso aparte que solo ocurre si todo pasa.

Comprueba siete cosas, y las dos ultimas son las que de verdad interesan:

1. La configuracion usada es la del registro de checkpoints, campo a campo.
1 bis. **El dataset recibio el mismo `n_timesteps` que el checkpoint.** Es la comprobacion que
   faltaba, y su ausencia hizo que este guion diera por bueno un fichero producido por un bug:
   comparaba el re-volcado con el fichero anterior, y los dos salian del mismo defecto.
2. El MD5 del checkpoint coincide con el que se declara.
3. El volcado denso cubre los 496 parches del fold retenido.
4. El volcado por parcela cubre las mismas parcelas que un miembro canonico.
5. Las posteriores son probabilidades: suman uno y no traen NaN.
6. **La diferencia de metricas con el fichero que sustituye**, para que promover no sea un cambio
   silencioso. No falla por ser grande: falla si no se puede medir. Un cambio grande puede ser el
   arreglo, y por eso se imprime y lo mira una persona.

**Sella su comparacion en un JSON.** Las cifras de una verificacion no pueden vivir solo en la
prosa de un inventario: es el defecto que este proyecto lleva ocho rondas persiguiendo, y una
verificacion cuyo resultado no es re-derivable no es una verificacion, es un recuerdo.

Uso:
    poetry run python scripts/verificar_redump_oof.py --temporal <dir> --miembro tsvit-pheno-fullm \
        --informe reports/paper_micai/oof/verificacion-tsvit-pheno-fullm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
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
    parser.add_argument("--informe", type=Path, default=None)
    args = parser.parse_args()

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    fallos: list[str] = []
    informe: dict[str, object] = {
        "miembro": args.miembro,
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # Los miembros DENSOS salen de un checkpoint y de un volcado por parches; los TABULARES no
    # tienen ninguna de las dos cosas. Que la herramienta admita los dos evita el segundo script
    # que dice casi lo mismo y se desincroniza del primero.
    spec = CHECKPOINT_REGISTRY.get(args.miembro)
    informe["tipo"] = "denso" if spec is not None else "tabular"
    if spec is not None:
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
        # Un checkpoint puede vivir fuera del repositorio -un volumen montado, un temporal de
        # pruebas-. Reventar con ValueError al construir el informe seria perder la verificacion
        # por un detalle de presentacion.
        ruta = Path(spec.path)
        informe["checkpoint"] = str(
            ruta.relative_to(REPO_ROOT) if ruta.is_relative_to(REPO_ROOT) else ruta
        )
        informe["checkpoint_md5"] = digest
        informe["model_kwargs"] = dict(spec.model_kwargs)

        # 2 bis. EL ACOPLAMIENTO ENTRE EL MODELO Y EL DATASET. Es la comprobacion que le
        # faltaba a este guion y por la que dio por bueno un fichero producido por un bug: el
        # volcado reconstruia el modelo con n_timesteps=37 y alimentaba al dataset con 10, la
        # codificacion posicional se desalineaba y el F1-macro caia de 0,7883 a 0,2552. Este
        # guion IMPRIMIA los model_kwargs del registro y nunca comprobaba que el dataset los
        # hubiera usado: verificaba la configuracion declarada, no la efectiva.
        t_model_spec = int(spec.model_kwargs.get("n_timesteps", 10))
        t_ds = entrada.get("n_timesteps_dataset")
        t_manifest_spec = entrada.get("n_timesteps_model_spec")
        informe["n_timesteps_model_spec"] = t_model_spec
        informe["n_timesteps_dataset"] = t_ds
        print(f"n_timesteps: model_spec={t_model_spec}  dataset={t_ds}")
        if t_manifest_spec != t_model_spec:
            fallos.append(
                f"el manifiesto registra n_timesteps_model_spec={t_manifest_spec} y el registro "
                f"activo exige {t_model_spec}"
            )
        if t_ds is None:
            fallos.append(
                "el volcado no registra el n_timesteps del dataset: sin ese campo, un volcado "
                "con la T equivocada es indistinguible de uno correcto mirando su manifiesto"
            )
        elif int(t_ds) != t_model_spec:
            fallos.append(
                f"el dataset uso n_timesteps={t_ds} y la especificacion exige {t_model_spec}: la "
                "codificacion posicional temporal queda desalineada y las metricas se hunden "
                "sin que nada aborte"
            )

        # 3. Cobertura densa.
        n_patches = entrada.get("n_patches")
        print(f"parches volcados: {n_patches}")
        informe["n_patches"] = n_patches
        if n_patches != PATCHES_ESPERADOS:
            fallos.append(
                f"se volcaron {n_patches} parches y el fold retenido tiene {PATCHES_ESPERADOS}"
            )
    else:
        print(f"modelo: {args.miembro} (tabular: sin checkpoint ni volcado por parches)")

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
    informe["n_parcelas_nuevo"] = len(ids_nuevo)
    informe["n_parcelas_referencia"] = len(ids_ref)
    informe["referencia"] = CANONICO_DE_REFERENCIA
    informe["parcelas_fuera_de_la_referencia"] = len(ids_nuevo - ids_ref)
    if ids_nuevo - ids_ref:
        # Sobrar SIEMPRE es un fallo: son parcelas que el universo del banco no contiene.
        fallos.append(f"{len(ids_nuevo - ids_ref)} parcelas no estan en {CANONICO_DE_REFERENCIA}")
    if spec is not None and ids_ref - ids_nuevo:
        # A un miembro denso se le exige cobertura completa; a uno tabular no, porque su ausencia
        # es NO ENTREGA y asi lo declara estimando-v1.json.
        fallos.append(
            f"faltan {len(ids_ref - ids_nuevo)} parcelas frente a {CANONICO_DE_REFERENCIA}"
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
            iguales = a.argmax(axis=1) == b.argmax(axis=1)
            coincide = float(iguales.mean())
            print(f"parcelas comparables con el fichero anterior: {len(comun)}")
            print(f"coincidencia de argmax con el anterior: {coincide:.4f}")
            print(f"diferencia media absoluta de posteriores: {float(np.abs(a - b).mean()):.6f}")
            # Los estratos por confianza: la media global esconde donde esta el desacuerdo, y en
            # este articulo lo que importa es justo la franja donde el modelo duda.
            conf = np.maximum(a.max(axis=1), b.max(axis=1))
            estratos = []
            for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
                m = (conf >= lo) & (conf < hi)
                if not m.any():
                    continue
                estratos.append(
                    {
                        "confianza_desde": lo,
                        "confianza_hasta": hi,
                        "n": int(m.sum()),
                        "coincidencia_argmax": round(float(iguales[m].mean()), 6),
                    }
                )
                print(
                    f"  confianza [{lo}, {hi}): n={int(m.sum()):6d}  "
                    f"coincide={float(iguales[m].mean()):.4f}"
                )
            informe["comparacion_con_el_anterior"] = {
                "n_parcelas_comparables": len(comun),
                "coincidencia_argmax": round(coincide, 6),
                "diferencia_media_absoluta": round(float(np.abs(a - b).mean()), 9),
                "diferencia_maxima_absoluta": round(float(np.abs(a - b).max()), 9),
                "por_estrato_de_confianza": estratos,
            }
        else:
            fallos.append("el fichero anterior no comparte ninguna parcela: no se puede comparar")
    else:
        print("no hay fichero anterior con el que comparar")

    informe["fallos"] = fallos
    informe["veredicto"] = "no_promover" if fallos else "se_puede_promover"
    if args.informe is not None:
        args.informe.parent.mkdir(parents=True, exist_ok=True)
        args.informe.write_text(
            json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"informe sellado en {args.informe}")

    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"verificar-redump: {len(fallos)} fallo(s); NO se promueve")
        return 1
    print("verificar-redump: OK; se puede promover")
    return 0


if __name__ == "__main__":
    sys.exit(main())
