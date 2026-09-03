"""Cobertura por clase bajo cada mecanismo: quien paga la abstencion.

Un revisor senalo, con razon, que el articulo afirmaba que la abstencion no excluye a nadie de
forma sistematica y citaba para respaldarlo un trabajo que demuestra lo contrario: la
clasificacion selectiva **amplifica** las diferencias entre grupos. El articulo no lo habia
medido ni una vez. Esto lo mide.

Para cada mecanismo, en el criterio principal preregistrado de cada banco, calcula que fraccion
de las parcelas de cada clase verdadera llega a entregarse. Si la abstencion concentra su rechazo
en las clases raras, entonces hace de forma implicita lo mismo que el recorte de leyenda, y el
argumento de equidad del articulo no se sostiene.

Uso:
    poetry run python scripts/run_paper_micai_equidad.py
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from ml.eval.paper_micai_arbitration import KEY, load_member_posteriors
from ml.eval.paper_micai_coverage import confidence_baseline, frontier, legend_by_f1

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
FASE4 = REPO_ROOT / "reports" / "paper_micai" / "fase4"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "equidad"

#: Criterio principal preregistrado de cada banco.
K_PRIMARIO: int = 9
K_REPLICA: int = 5


def _per_class_coverage(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    points: list[Any],
    num_classes: int,
) -> dict[int, float]:
    """Fraction of each true class's parcels that the mechanism delivers.

    Args:
        labels: Ground-truth labels of the whole universe.
        splits: ``(train_pos, test_pos)`` pairs, one per block.
        points: Block points of one mechanism at one legend size.
        num_classes: Size of the label space.

    Returns:
        Mapping class to delivered fraction, pooled over blocks.
    """
    entregadas = np.zeros(num_classes)
    totales = np.zeros(num_classes)
    for p in points:
        _, test_pos = splits[p.block]
        truth = labels[test_pos]
        for c in range(num_classes):
            en_clase = truth == c
            totales[c] += en_clase.sum()
            entregadas[c] += (en_clase & p.delivered).sum()
    return {
        c: float(entregadas[c] / totales[c]) if totales[c] else float("nan")
        for c in range(num_classes)
    }


def _analiza(
    banco: str,
    proba: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    nombres: dict[int, str],
) -> dict[str, Any]:
    """Measure per-class coverage of both mechanisms on one benchmark.

    Args:
        banco: Benchmark name.
        proba: Posterior matrix.
        labels: Ground-truth labels.
        splits: Block index pairs.
        k: Principal legend size.
        nombres: Class id to name.

    Returns:
        A record with the per-class coverage of both mechanisms and the rank correlation
        between coverage and class support.
    """
    n_classes = proba.shape[1]
    free = proba.argmax(axis=1)
    retirada = frontier(
        proba,
        labels,
        splits,
        (k,),
        legend_fn=partial(legend_by_f1, labels, free, num_classes=n_classes),
        mechanism="retirada por F1",
    )
    confianza = confidence_baseline(proba, labels, splits, retirada, num_classes=n_classes)

    cob_ret = _per_class_coverage(labels, splits, retirada, n_classes)
    cob_con = _per_class_coverage(labels, splits, confianza, n_classes)
    soporte = {c: int((labels == c).sum()) for c in range(n_classes)}
    presentes = [c for c in range(n_classes) if soporte[c] > 0]

    from scipy.stats import spearmanr

    rho_con = spearmanr([soporte[c] for c in presentes], [cob_con[c] for c in presentes])
    rho_ret = spearmanr([soporte[c] for c in presentes], [cob_ret[c] for c in presentes])

    filas = [
        {
            "banco": banco,
            "clase": c,
            "nombre": nombres.get(c, str(c)),
            "soporte": soporte[c],
            "cobertura_retirada": round(cob_ret[c], 6),
            "cobertura_confianza": round(cob_con[c], 6),
        }
        for c in sorted(presentes, key=lambda x: -soporte[x])
    ]
    raras = [c for c in presentes if soporte[c] <= np.percentile(list(soporte.values()), 33)]
    return {
        "banco": banco,
        "k_principal": k,
        "filas": filas,
        "spearman_soporte_vs_cobertura_confianza": {
            "rho": float(rho_con.statistic),
            "p": float(rho_con.pvalue),
        },
        "spearman_soporte_vs_cobertura_retirada": {
            "rho": float(rho_ret.statistic),
            "p": float(rho_ret.pvalue),
        },
        "cobertura_media_clases_raras_confianza": float(np.mean([cob_con[c] for c in raras])),
        "cobertura_media_clases_raras_retirada": float(np.mean([cob_ret[c] for c in raras])),
        "clases_con_cobertura_cero_confianza": [c for c in presentes if cob_con[c] == 0.0],
        "clases_con_cobertura_cero_retirada": [c for c in presentes if cob_ret[c] == 0.0],
    }


def main() -> None:
    """Measure who pays for the abstention on both benchmarks."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registros = []

    # --- banco primario ---
    from ml.ensemble.stacking import StackingEnsemble

    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    keys = gt[KEY].to_list()
    labels = gt["label"].to_numpy()
    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    helper = StackingEnsemble(base_members=("tsvit-pheno",), oof_dir=OOF_DIR, random_state=42)
    splits = helper._subfolds_by_canonical_id(geoms, pl.DataFrame({KEY: keys}))
    proba = load_member_posteriors(OOF_DIR, ("tsvit-pheno",), keys)["tsvit-pheno"]
    registros.append(_analiza("PASTIS-R", proba, labels, splits, K_PRIMARIO, {}))

    # --- banco de replica ---
    post = pl.read_parquet(FASE4 / "breizhcrops_posteriores.parquet")
    cols = sorted(c for c in post.columns if c.startswith("prob_"))
    proba4 = post.select(cols).to_numpy()
    labels4 = post["class_id"].to_numpy()
    regiones = sorted(post["region"].unique().to_list())
    bloque = np.array([regiones.index(r) for r in post["region"].to_list()])
    splits4 = [
        (np.flatnonzero(bloque != b), np.flatnonzero(bloque == b)) for b in range(len(regiones))
    ]
    soporte4 = pl.read_csv(FASE4 / "breizhcrops_soporte.csv").unique(subset=["class_id"])
    nombres4 = {r["class_id"]: r["class_name"] for r in soporte4.to_dicts()}
    registros.append(_analiza("BreizhCrops", proba4, labels4, splits4, K_REPLICA, nombres4))

    pl.DataFrame([f for r in registros for f in r["filas"]]).write_csv(
        OUT_DIR / "cobertura_por_clase.csv"
    )
    (OUT_DIR / "equidad.json").write_text(
        json.dumps(
            {
                "pregunta": (
                    "Quien paga la abstencion. Si el rechazo por confianza concentra su rechazo en "
                    "las clases raras, hace de forma implicita lo mismo que el recorte de leyenda."
                ),
                "resultados": [{k: v for k, v in r.items() if k != "filas"} for r in registros],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for r in registros:
        logger.info(
            "equidad",
            banco=r["banco"],
            rho_confianza=round(r["spearman_soporte_vs_cobertura_confianza"]["rho"], 4),
            cob_raras_confianza=round(r["cobertura_media_clases_raras_confianza"], 4),
            cob_raras_retirada=round(r["cobertura_media_clases_raras_retirada"], 4),
            ceros_confianza=len(r["clases_con_cobertura_cero_confianza"]),
        )


if __name__ == "__main__":
    main()
