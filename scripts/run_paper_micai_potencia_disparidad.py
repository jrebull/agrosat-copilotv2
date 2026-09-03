"""Potencia del criterio principal NUEVO, que se movio sin calcularla.

La tercera vuelta midio que el delta de calidad necesitaba entre trece y veinte bloques y que hay
cinco, y lo dijo. Despues movio el criterio principal a la disparidad de cobertura sobre los MISMOS
cinco bloques, sin recalcular nada, apoyandose en una razon de ocho que resulta ser el maximo de
dieciocho razones cuya mediana es uno. Esto lo mide.

Se evaluan cuatro medidas de disparidad declaradas aqui, ANTES de elegir, y para cada una se
calcula la varianza entre bloques y el efecto minimo detectable con cinco bloques. Ninguna se elige
por su resultado: se publican las cuatro.

Uso:
    poetry run python scripts/run_paper_micai_potencia_disparidad.py
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
from ml.eval.paper_micai_coverage import confidence_baseline, frontier, legend_by_f1

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "potencia"

K_PRINCIPAL: int = 9
PREDICTOR: str = "tsvit-pheno"

#: Soporte minimo para que una clase entre al estadistico. Sin el, una clase de dos parcelas
#: domina cualquier minimo o cualquier razon.
SOPORTE_MINIMO: int = 50


def _coberturas(
    labels: np.ndarray, test_pos: np.ndarray, entregadas: np.ndarray
) -> dict[int, float]:
    """Per-class delivered fraction inside one block.

    Args:
        labels: Ground-truth labels of the whole universe.
        test_pos: Positional indices of the block.
        entregadas: Boolean delivery mask for the block.

    Returns:
        Class to delivered fraction, only for classes above the support floor.
    """
    truth = labels[test_pos]
    out: dict[int, float] = {}
    for c in np.unique(truth):
        en_clase = truth == c
        if en_clase.sum() >= SOPORTE_MINIMO:
            out[int(c)] = float(entregadas[en_clase].mean())
    return out


#: Las cuatro medidas, declaradas antes de mirar ninguna.
MEDIDAS: dict[str, Any] = {
    "cobertura_minima": lambda cob: min(cob.values()),
    "rango": lambda cob: max(cob.values()) - min(cob.values()),
    "desviacion_entre_clases": lambda cob: float(np.std(list(cob.values()), ddof=1)),
    "brecha_p10_p90": lambda cob: float(
        np.percentile(list(cob.values()), 90) - np.percentile(list(cob.values()), 10)
    ),
}


def main(seed: int = 42) -> None:
    """Measure the between-block variance and the MDE of every declared disparity measure."""
    from ml.ensemble.stacking import StackingEnsemble

    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    keys = gt[KEY].to_list()
    labels = gt["label"].to_numpy()
    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    helper = StackingEnsemble(base_members=(PREDICTOR,), oof_dir=OOF_DIR, random_state=seed)
    splits = helper._subfolds_by_canonical_id(geoms, pl.DataFrame({KEY: keys}))
    proba = load_member_posteriors(OOF_DIR, (PREDICTOR,), keys)[PREDICTOR]
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

    filas: list[dict[str, Any]] = []
    for a, b in zip(izq, der, strict=True):
        _, test_pos = splits[a.block]
        cob_ret = _coberturas(labels, test_pos, a.delivered)
        cob_con = _coberturas(labels, test_pos, b.delivered)
        comunes = sorted(set(cob_ret) & set(cob_con))
        fila: dict[str, Any] = {"bloque": a.block, "n_clases": len(comunes)}
        for nombre, f in MEDIDAS.items():
            r = f({c: cob_ret[c] for c in comunes})
            c_ = f({c: cob_con[c] for c in comunes})
            fila[f"{nombre}_retirada"] = round(r, 6)
            fila[f"{nombre}_confianza"] = round(c_, 6)
            fila[f"{nombre}_delta"] = round(c_ - r, 6)
        filas.append(fila)

    resumen: dict[str, Any] = {}
    for nombre in MEDIDAS:
        d = np.array([f[f"{nombre}_delta"] for f in filas])
        m, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd / np.sqrt(len(d))
        lo, hi = stats.t.interval(0.95, len(d) - 1, m, se)
        tcrit = stats.t.ppf(0.975, len(d) - 1)
        mde = (tcrit + stats.t.ppf(0.80, len(d) - 1)) * sd / np.sqrt(len(d))
        resumen[nombre] = {
            "delta_medio": round(m, 6),
            "sd_entre_bloques": round(sd, 6),
            "ic_t": [round(float(lo), 6), round(float(hi), 6)],
            "excluye_cero": bool(lo > 0 or hi < 0),
            "mde_con_5_bloques": round(float(mde), 6),
            "tiene_potencia": bool(abs(m) >= mde),
            "bloques_necesarios": (
                int(np.ceil((sd * (tcrit + stats.t.ppf(0.80, len(d) - 1)) / abs(m)) ** 2))
                if m
                else None
            ),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(OUT_DIR / "disparidad_por_bloque.csv")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "potencia_disparidad.json").write_text(
        json.dumps(
            {
                "pregunta": (
                    "El criterio principal se movio a la disparidad sin calcular su potencia. "
                    "calcula sobre las mismas cinco unidades, con cuatro medidas declaradas "
                    "antes de mirar ninguna."
                ),
                "soporte_minimo_por_clase": SOPORTE_MINIMO,
                "nota": (
                    "El factor de ocho que motivo el giro es el maximo de 18 razones cuya "
                    "mediana es 1,00, leido tras mirar la tabla, y no es monotono en soporte: la "
                    "clase 8, con 167 parcelas, es mas rara que la 10 con 355 y recibe 0,958 "
                    "cobertura bajo recorte."
                ),
                "medidas": resumen,
                "procedencia": {
                    "semilla": seed,
                    "code_version": head or "desconocido",
                    "generado": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for nombre, r in resumen.items():
        logger.info(
            "potencia",
            medida=nombre,
            delta=r["delta_medio"],
            sd=r["sd_entre_bloques"],
            mde=r["mde_con_5_bloques"],
            potencia=r["tiene_potencia"],
            excluye_cero=r["excluye_cero"],
            n_necesario=r["bloques_necesarios"],
        )


if __name__ == "__main__":
    main()
