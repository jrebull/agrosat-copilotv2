"""Cuanto cuesta cada uno de los tres defectos de protocolo, medido y sellado.

La primera contribucion del articulo reencuadrado es el protocolo corregido y **la medida de lo
que cuesta equivocarse en cada pieza**. Esas tres cifras se publicaron primero en prosa, sin
artefacto que las sostuviera, que es justo lo que la regla de la casa prohibe. Esto las produce.

Los tres defectos, cada uno medido contra su version correcta:

1. **Denominador no comun.** El estimando promediaba sobre las clases presentes en la entrega de
   CADA brazo, y esos conjuntos difieren. Se mide el contraste con el denominador comun.
2. **Unidad de remuestreo.** El estimando declarado es la media por bloque, pero el remuestreo
   sorteaba parcelas dentro del bloque. Se compara con el intervalo sobre bloques.
3. **Punto de operacion asimetrico.** El umbral de confianza se elegia dentro del bloque de
   evaluacion mientras la leyenda se elegia fuera. Se mide con el umbral fijado fuera.

Uso:
    poetry run python scripts/run_paper_micai_diagnostico_protocolo.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
from scipy import stats

from ml.eval.paper_micai_arbitration import KEY, load_member_posteriors
from ml.eval.paper_micai_coverage import (
    BlockPoint,
    confidence_baseline,
    frontier,
    legend_by_f1,
    macro_over,
)

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "diagnostico"

#: Criterio principal preregistrado del banco primario.
K_PRINCIPAL: int = 9

#: Predictor sobre el que se diagnostica; el mismo de la fase 3.
PREDICTOR: str = "tsvit-pheno"


def _universo(seed: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Load labels, spatial sub-folds and the predictor's posterior.

    Args:
        seed: Seed forwarded to the spatial split.

    Returns:
        Labels, block index pairs and the posterior matrix.
    """
    from ml.ensemble.stacking import StackingEnsemble

    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    keys = gt[KEY].to_list()
    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    helper = StackingEnsemble(base_members=(PREDICTOR,), oof_dir=OOF_DIR, random_state=seed)
    splits = helper._subfolds_by_canonical_id(geoms, pl.DataFrame({KEY: keys}))
    proba = load_member_posteriors(OOF_DIR, (PREDICTOR,), keys)[PREDICTOR]
    return gt["label"].to_numpy(), splits, proba


def _denominador(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    izq: list[BlockPoint],
    der: list[BlockPoint],
) -> dict[str, Any]:
    """Measure what the non-common denominator costs, block by block.

    Args:
        labels: Ground-truth labels.
        splits: Block index pairs.
        izq: Points of the legend-shrinking arm.
        der: Points of the confidence arm.

    Returns:
        Per-block deltas under both estimands and their means.
    """
    filas = []
    for a, b in zip(izq, der, strict=True):
        _, test = splits[a.block]
        truth = labels[test]
        ya = sorted(set(a.legend) & set(truth[a.delivered].tolist()))
        yb = sorted(set(a.legend) & set(truth[b.delivered].tolist()))
        comun = sorted(set(ya) & set(yb))
        filas.append(
            {
                "bloque": a.block,
                "n_clases_retirada": len(ya),
                "n_clases_confianza": len(yb),
                "n_clases_comun": len(comun),
                "delta_publicado": round(a.aligned_f1 - b.aligned_f1, 6),
                "delta_denominador_comun": round(
                    macro_over(truth[a.delivered], a.emitted[a.delivered], comun)
                    - macro_over(truth[b.delivered], b.emitted[b.delivered], comun),
                    6,
                ),
            }
        )
    pub = float(np.mean([f["delta_publicado"] for f in filas]))
    com = float(np.mean([f["delta_denominador_comun"] for f in filas]))
    return {
        "por_bloque": filas,
        "delta_publicado": round(pub, 6),
        "delta_denominador_comun": round(com, 6),
        "encogimiento_relativo": round(1 - abs(com) / abs(pub), 6) if pub else None,
        "bloques_que_cambian_de_signo": [
            f["bloque"]
            for f in filas
            if np.sign(f["delta_publicado"]) != np.sign(f["delta_denominador_comun"])
        ],
    }


def _unidad_de_remuestreo(deltas: list[float]) -> dict[str, Any]:
    """Compare the published interval with one over the declared unit, the block.

    Args:
        deltas: Per-block deltas.

    Returns:
        The block-level t interval and whether it excludes zero.
    """
    m = float(np.mean(deltas))
    low, high = stats.t.interval(0.95, len(deltas) - 1, m, stats.sem(deltas))
    return {
        "deltas_por_bloque": [round(d, 6) for d in deltas],
        "media": round(m, 6),
        "ic_t_sobre_bloques": [round(float(low), 6), round(float(high), 6)],
        "excluye_cero": bool(low > 0 or high < 0),
        "nota": (
            "El intervalo publicado remuestreaba parcelas dentro de cada bloque y nunca bloques, "
            "aunque el estimando declarado es la media por bloque."
        ),
    }


def _simetria(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    proba: np.ndarray,
    izq: list[BlockPoint],
) -> dict[str, Any]:
    """Measure what choosing the confidence threshold inside the evaluated block was worth.

    The published baseline took the quantile of the evaluation block's own confidences. The
    symmetric version fixes it on the training blocks, exactly as the legend is fixed.

    Args:
        labels: Ground-truth labels.
        splits: Block index pairs.
        proba: Posterior matrix.
        izq: Points of the legend-shrinking arm, whose coverage is matched.

    Returns:
        Both variants of the confidence arm and the gap between them.
    """
    confianza = proba.max(axis=1)
    free = proba.argmax(axis=1)
    dentro, fuera = [], []
    for ref in izq:
        train_pos, test_pos = splits[ref.block]
        objetivo = float(ref.delivered.mean())
        truth = labels[test_pos]

        # Variante publicada: el umbral sale del propio bloque evaluado.
        orden = np.argsort(-confianza[test_pos], kind="stable")
        d_dentro = np.zeros(test_pos.size, dtype=bool)
        d_dentro[orden[: int(ref.delivered.sum())]] = True
        dentro.append(macro_over(truth[d_dentro], free[test_pos][d_dentro], ref.legend))

        # Variante simetrica: el umbral se fija en los bloques de entrenamiento.
        umbral = float(np.quantile(confianza[train_pos], 1.0 - objetivo))
        d_fuera = confianza[test_pos] >= umbral
        fuera.append(macro_over(truth[d_fuera], free[test_pos][d_fuera], ref.legend))

    return {
        "f1_umbral_dentro_del_bloque": round(float(np.mean(dentro)), 6),
        "f1_umbral_fijado_fuera": round(float(np.mean(fuera)), 6),
        "ventaja_de_la_asimetria": round(float(np.mean(dentro) - np.mean(fuera)), 6),
        "nota": (
            "La retirada de leyenda siempre eligio su catalogo fuera del bloque evaluado. El "
            "rechazo por confianza no, y es el brazo que ganaba."
        ),
    }


def main(seed: int = 42) -> None:
    """Measure and seal the cost of the three protocol defects."""
    labels, splits, proba = _universo(seed)
    free = proba.argmax(axis=1)
    izq = frontier(
        proba,
        labels,
        splits,
        (K_PRINCIPAL,),
        legend_fn=partial(legend_by_f1, labels, free),
        mechanism="retirada por F1",
    )
    der = confidence_baseline(proba, labels, splits, izq)

    denom = _denominador(labels, splits, izq, der)
    unidad = _unidad_de_remuestreo([f["delta_publicado"] for f in denom["por_bloque"]])
    simetria = _simetria(labels, splits, proba, izq)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(denom["por_bloque"]).write_csv(OUT_DIR / "denominador_por_bloque.csv")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "diagnostico_protocolo.json").write_text(
        json.dumps(
            {
                "pregunta": (
                    "Cuanto cuesta cada uno de los tres defectos de protocolo que cuatro revisores "
                    "encontraron. Estas cifras se publicaron primero en prosa y sin artefacto."
                ),
                "predictor": PREDICTOR,
                "k_principal": K_PRINCIPAL,
                "defecto_1_denominador_no_comun": denom,
                "defecto_2_unidad_de_remuestreo": unidad,
                "defecto_3_punto_de_operacion_asimetrico": simetria,
                "procedencia": {
                    "semilla": seed,
                    "code_version": head or "desconocido",
                    "polars": pl.__version__,
                    "numpy": np.__version__,
                    "generado": datetime.now(UTC).isoformat(timespec="seconds"),
                    "ground_truth": "reports/paper_micai/fase1/parcel_gt_fold5.parquet",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "diagnostico",
        encogimiento=denom["encogimiento_relativo"],
        cambian_signo=denom["bloques_que_cambian_de_signo"],
        ic_bloques=unidad["ic_t_sobre_bloques"],
        excluye_cero=unidad["excluye_cero"],
        ventaja_asimetria=simetria["ventaja_de_la_asimetria"],
    )


if __name__ == "__main__":
    main()
