"""La potencia frente al numero de bloques: cinco era una eleccion, no un dato.

Todas las auditorias trataron los cinco bloques espaciales como una restriccion del conjunto de
datos, y concluyeron que hacian falta entre trece y veinte. Pero el cinco es el valor por defecto de
`build_spatial_kfold`, no una propiedad de PASTIS: hay 176 celdas H3 distintas en el fold retenido,
asi que la teselacion admite bastantes mas bloques.

Esto lo mide. Y deja escrita la trampa que abre: **si `k` se elige despues de ver cual da
significancia, esto es p-hacking con otro nombre**. El valor de `k` tiene que preregistrarse con un
criterio ajeno al resultado, y esa es la conclusion operativa de este guion, no la significancia.

Uso:
    poetry run python scripts/run_paper_micai_bloques.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import polars as pl
import structlog
from scipy import stats
from shapely import wkb, wkt

from ml.eval.paper_micai_arbitration import KEY, load_member_posteriors
from ml.eval.paper_micai_coverage import (
    confidence_baseline,
    frontier,
    legend_by_f1,
    macro_over,
    presentes_en_bloque,
)
from ml.features.spatial_split import build_spatial_kfold

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "bloques"

#: Valores de k del barrido. El 5 es el que se uso hasta ahora, por ser el valor por defecto.
K_BLOQUES: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25)

#: Parcelas minimas para que un bloque entre al analisis.
MIN_PARCELAS: int = 100

K_LEYENDA: int = 9
PREDICTOR: str = "tsvit-pheno"


def _geometrias() -> tuple[gpd.GeoDataFrame, list[str], np.ndarray, np.ndarray]:
    """Load the sealed universe with its geometries, indexed by position.

    Returns:
        Geometries with a positional id, the parcel keys, the labels and the posterior.
    """
    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    keys = gt[KEY].to_list()
    orden = {p: i for i, p in enumerate(keys)}
    g = (
        pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
        .with_columns(pl.col(KEY).replace_strict(orden, default=None).alias("pos"))
        .sort("pos")
    )
    shapes = [
        wkb.loads(x) if isinstance(x, bytes | bytearray) else wkt.loads(x)
        for x in g["geometry"].to_list()
    ]
    gdf = gpd.GeoDataFrame(
        {"parcel_id": list(range(len(shapes)))}, geometry=shapes, crs="EPSG:4326"
    )
    proba = load_member_posteriors(OOF_DIR, (PREDICTOR,), keys)[PREDICTOR]
    return gdf, keys, gt["label"].to_numpy(), proba


def main(seed: int = 42) -> None:
    """Sweep the number of spatial blocks and report what it does to the inference."""
    gdf, _, labels, proba = _geometrias()
    free = proba.argmax(axis=1)
    filas: list[dict[str, Any]] = []

    for k in K_BLOQUES:
        folds = build_spatial_kfold(gdf, k=k, random_state=seed)
        splits = []
        for f in folds:
            te = np.asarray(sorted(int(x) for x in f.test_ids))
            tr = np.asarray(sorted(int(x) for x in [*f.train_ids, *f.val_ids]))
            if te.size >= MIN_PARCELAS and tr.size:
                splits.append((tr, te))
        if len(splits) < 3:
            continue

        izq = frontier(
            proba,
            labels,
            splits,
            (K_LEYENDA,),
            legend_fn=partial(legend_by_f1, labels, free),
            mechanism="retirada por F1",
        )
        der = confidence_baseline(proba, labels, splits, izq)
        deltas = []
        for a, b in zip(izq, der, strict=True):
            _, test_pos = splits[a.block]
            truth = labels[test_pos]
            comun = sorted(
                set(a.legend) & set(truth[a.delivered].tolist()) & set(truth[b.delivered].tolist())
            )
            presentes = presentes_en_bloque(truth)
            deltas.append(
                macro_over(truth[a.delivered], a.emitted[a.delivered], comun, presentes=presentes)
                - macro_over(truth[b.delivered], b.emitted[b.delivered], comun, presentes=presentes)
            )
        d = np.array(deltas)
        n = len(d)
        m, sd = float(d.mean()), float(d.std(ddof=1))
        lo, hi = stats.t.interval(0.95, n - 1, m, sd / np.sqrt(n))
        mde = float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n))
        filas.append(
            {
                "k_solicitado": k,
                "bloques_utiles": n,
                "parcelas_min_por_bloque": int(min(len(t) for _, t in splits)),
                "delta": round(m, 6),
                "sd_entre_bloques": round(sd, 6),
                "ic_low": round(float(lo), 6),
                "ic_high": round(float(hi), 6),
                "excluye_cero": bool(lo > 0 or hi < 0),
                "mde": round(mde, 6),
                "tiene_potencia": bool(abs(m) >= mde),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(OUT_DIR / "barrido_bloques.csv")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "bloques.json").write_text(
        json.dumps(
            {
                "pregunta": (
                    "Las auditorias trataron los cinco bloques como restriccion del dato. "
                    "Son el valor por defecto de build_spatial_kfold, y el fold retenido tiene 176 "
                    "celdas H3 distintas."
                ),
                "advertencia": (
                    "Si k se elige tras ver cual da significancia, esto es p-hacking con otro "
                    "nombre. La conclusion NO es que exista un k que da significancia: es "
                    "que k es un grado de libertad que hay que preregistrar con criterio ajeno al "
                    "resultado, por ejemplo un tamano minimo de bloque o el numero de celdas de la "
                    "teselacion. Y al subir k los bloques se acercan, asi que el colchon tiene que "
                    "crecer con k o la independencia se degrada."
                ),
                "barrido": filas,
                "procedencia": {
                    "semilla": seed,
                    "code_version": head or "desconocido",
                    "k_leyenda": K_LEYENDA,
                    "predictor": PREDICTOR,
                    "min_parcelas_por_bloque": MIN_PARCELAS,
                    "generado": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for f in filas:
        logger.info(
            "bloques", **{k: v for k, v in f.items() if k != "k_solicitado"}, k=f["k_solicitado"]
        )


if __name__ == "__main__":
    main()
