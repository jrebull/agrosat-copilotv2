"""Compara la metrica que declara cada checkpoint con la que da su volcado sobre el fold 5.

Un modelo puede caer entre su validacion y el fold retenido por dos motivos muy distintos: porque
el fold 5 es mas dificil, o porque el arnes lo esta cargando o alimentando mal. Lo segundo ya paso
—`tsvit-pheno-fullm` marcaba 0,2552 en vez de 0,7883 porque el dataset le daba T=10 y el modelo
esperaba T=37— y se descubrio por casualidad. Este guion lo convierte en una comprobacion.

**Las dos metricas que compara son las dos por PIXEL**, y eso es deliberado: el criterio de
aceptacion de US-119 prohibe mezclar F1 por parcela con mIoU por pixel, y mezclarlas es facil
porque los dos numeros existen para cada miembro y se parecen. El F1 por parcela se calcula
tambien, se publica en su propia columna y **no se resta de nada**.

Lo que este guion NO puede hacer, y lo dice en vez de inventarlo: la mayoria de los checkpoints no
registran sus folds de entrenamiento. Donde no consta, se declara que no consta.

Uso:
    poetry run python scripts/run_us119_sanidad_miembros.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import torch
from sklearn.metrics import f1_score

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "us119"

#: Diferencia a partir de la cual la caida deja de ser atribuible al fold y hay que explicarla.
UMBRAL = 0.15

#: Miembro -> (checkpoint, clave de la metrica declarada). El nombre de la clave se conserva tal
#: cual aparece en el fichero: `val_f1` y `f1_macro` no son necesariamente lo mismo y fingir que
#: si lo son es el primer paso para comparar cosas distintas.
#:
#: **`unet` y `anysat` guardan su metrica en un fichero DISTINTO del que carga el arnes**: el
#: registro apunta a `*_pastis.pt`, que es un state_dict pelado sin metricas, y las metricas viven
#: en `*_ckpt.pt`. Comparar la metrica de un fichero con el volcado de otro solo vale si los pesos
#: son los mismos, y se comprobo: 380 de 380 tensores identicos en unet y 501 de 501 en anysat.
#: Sin esa comprobacion, esta fila entera seria una suposicion.
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "unet": ("unet-aaron/unet_ckpt.pt", "best.f1_macro"),
    "anysat": ("anysat-aaron/anysat_ckpt.pt", "best.f1_macro"),
    "utae": ("utae-isaac/best_model.pt", "val_f1"),
    "segformer": ("segformer-isaac/best_model.pt", "val_f1"),
    "deeplabv3plus": ("deeplab-18/best.pt", "best_metrics.f1_macro"),
    "tsvit-pheno": ("tsvit-pheno-v1/best.pt", "best_metrics.f1_macro"),
    "tsvit-pheno-fullm": ("tsvit-pheno-fullm-v1/best.pt", "best_metrics.f1_macro"),
}


def _metrica_declarada(ruta: Path, clave: str) -> tuple[float | None, dict[str, Any]]:
    """Validation metric a checkpoint records, plus whatever it says about its training.

    Args:
        ruta: Checkpoint path.
        clave: Dotted key of the metric inside the checkpoint.

    Returns:
        The metric, and a dict with the training record found (possibly empty).
    """
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    valor: Any = ck
    for parte in clave.split("."):
        valor = valor.get(parte) if isinstance(valor, dict) else None
        if valor is None:
            break
    registro: dict[str, Any] = {"epoch": ck.get("epoch")}
    cfg = ck.get("config")
    if isinstance(cfg, dict):
        registro["config_keys"] = sorted(cfg)[:12]
        for k in cfg:
            if "fold" in k.lower():
                registro[k] = cfg[k]
    registro["declara_folds_de_entrenamiento"] = any("fold" in k.lower() for k in registro)
    return (float(valor) if isinstance(valor, (int, float)) else None), registro


def _verdad_por_pixel(patch_ids: list[str]) -> np.ndarray:
    """Per-pixel semantic18 ground truth of the given patches, flattened."""
    from ml.data.pastis_seg_dataset import _build_semantic18_lut

    lut = _build_semantic18_lut(255)
    trozos = []
    for pid in patch_ids:
        target = np.load(PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0]
        trozos.append(lut[np.clip(target.astype(np.int64), 0, 19)].ravel())
    return np.concatenate(trozos)


def main() -> None:
    """Compare every canonical member's declared metric with its fold-5 dump."""
    from ml.eval.oof.inventario import cargar_inventario

    inventario = cargar_inventario()
    gt_parcela = pl.read_parquet(
        REPO_ROOT / "reports/paper_micai/fase1/parcel_gt_fold5.parquet"
    ).sort("canonical_parcel_id")
    filas: list[dict[str, Any]] = []

    for miembro, (rel, clave) in CHECKPOINTS.items():
        denso = OOF_DIR / f"oof_{miembro}_fold5.parquet"
        parcela = OOF_DIR / f"oof_parcel_{miembro}_fold5.parquet"
        estado = inventario["ficheros"].get(parcela.name, {}).get("estado", "no declarado")
        if not denso.exists():
            logger.warning("sin_volcado_denso", miembro=miembro)
            continue

        tabla = pl.read_parquet(denso, columns=["patch_id", "pred"]).sort("patch_id")
        pids = tabla["patch_id"].to_list()
        pred = np.concatenate([np.asarray(x, dtype=np.int64) for x in tabla["pred"].to_list()])
        verdad = _verdad_por_pixel(pids)
        valido = verdad != 255
        f1_pixel = float(f1_score(verdad[valido], pred[valido], average="macro", zero_division=0))

        # El F1 por PARCELA se calcula y se publica aparte. No se resta de nada.
        f1_parcela = None
        if parcela.exists():
            from ml.utils.parcel_reconcile import PROB_COLUMNS

            p = pl.read_parquet(parcela).sort("canonical_parcel_id")
            unido = gt_parcela.join(p, on="canonical_parcel_id", how="inner")
            if unido.height:
                probs = unido.select(PROB_COLUMNS).to_numpy()
                f1_parcela = float(
                    f1_score(
                        unido["label"].to_numpy(),
                        probs.argmax(axis=1),
                        average="macro",
                        zero_division=0,
                    )
                )

        ruta_ckpt = REPO_ROOT / "checkpoints/segmentation" / rel
        declarada, registro = _metrica_declarada(ruta_ckpt, clave)
        delta = None if declarada is None else round(declarada - f1_pixel, 4)
        filas.append(
            {
                "miembro": miembro,
                "estado_inventario": estado,
                "checkpoint": rel,
                "clave_declarada": clave,
                "f1_macro_pixel_declarado": None if declarada is None else round(declarada, 4),
                "f1_macro_pixel_fold5": round(f1_pixel, 4),
                "delta_pixel": delta,
                "supera_umbral": None if delta is None else bool(abs(delta) > UMBRAL),
                "f1_macro_parcela_fold5": None if f1_parcela is None else round(f1_parcela, 4),
                "n_patches": len(pids),
                "epoch_del_checkpoint": registro.get("epoch"),
                "declara_folds_de_entrenamiento": registro["declara_folds_de_entrenamiento"],
                "metricas_en_otro_fichero": miembro in {"unet", "anysat"},
            }
        )
        logger.info("miembro", **filas[-1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(OUT_DIR / "sanidad_miembros.csv")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "sanidad_miembros.json").write_text(
        json.dumps(
            {
                "para_que": (
                    "Comparar la metrica que declara cada checkpoint con la que da su volcado "
                    "sobre el fold 5, para no atribuir a un modelo lo que es un checkpoint mal "
                    "cargado o mal alimentado."
                ),
                "umbral": UMBRAL,
                "las_dos_metricas_comparadas_son_por_pixel": True,
                "nota_parcela": (
                    "f1_macro_parcela_fold5 se publica aparte y NO se compara con las de pixel. "
                    "Mezclarlas es lo que US-119 prohibe expresamente."
                ),
                "nota_folds": (
                    "Ningun checkpoint registra sus folds de entrenamiento. Se declara que el "
                    "registro no existe en vez de deducirlo: el volcado usa (1,2,3) para las "
                    "estadisticas de normalizacion, pero eso es una eleccion del arnes y no una "
                    "constancia de como se entreno cada modelo."
                ),
                "miembros": filas,
                "code_version": head or "desconocido",
                "generado": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("sellado", out=str(OUT_DIR), n=len(filas))


if __name__ == "__main__":
    main()
