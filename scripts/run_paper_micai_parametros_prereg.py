"""Mide los parametros que el preregistro tiene que fijar, antes de fijarlos.

Un preregistro con huecos no es un preregistro, y uno con numeros inventados es peor. Estos son los
tres grados de libertad que el plan identifico y que hay que declarar con un motivo:

1. **El suelo de soporte por bloque** `S`: cuantas parcelas de una clase hacen falta en un bloque
   para que su F1 sea estimable. Se mide cuantas clases y cuanta parcela sobreviven a cada `S`.
2. **El criterio espacial de `k`**: se mide la separacion real entre prueba y entrenamiento y el
   area de bloque para cada `k`, para poder fijar el umbral sin mirar ningun contraste.
3. **La banda de equivalencia**: se ancla en el efecto minimo detectable del diseno, no en un
   numero redondo.

Este guion NO calcula ningun contraste entre mecanismos. Solo mide propiedades del diseno.

Uso:
    poetry run python scripts/run_paper_micai_parametros_prereg.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import polars as pl
import structlog
from scipy import stats
from shapely import wkb, wkt

from ml.eval.paper_micai_arbitration import KEY
from ml.features.spatial_split import build_spatial_kfold

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "prereg"

#: Suelos candidatos de soporte por clase y bloque.
SUELOS: tuple[int, ...] = (5, 10, 20, 30, 50)

#: Valores de k sobre los que se mide la geometria del diseno.
K_VALORES: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25)


def _universo() -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Load the sealed universe with geometries indexed by position.

    Returns:
        Geometries with a positional id, and the labels in the same order.
    """
    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    orden = {p: i for i, p in enumerate(gt[KEY].to_list())}
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
    return gdf, gt["label"].to_numpy()


def main(seed: int = 42) -> None:
    """Measure the design properties the pre-registration has to declare."""
    gdf, labels = _universo()
    metrico = gdf.to_crs(3035)
    centros = np.column_stack([metrico.geometry.centroid.x, metrico.geometry.centroid.y])

    filas: list[dict[str, Any]] = []
    for k in K_VALORES:
        folds = build_spatial_kfold(gdf, k=k, random_state=seed)
        bloques = []
        for f in folds:
            te = np.asarray(sorted(int(x) for x in f.test_ids))
            tr = np.asarray(sorted(int(x) for x in [*f.train_ids, *f.val_ids]))
            if te.size >= 100 and tr.size:
                bloques.append((tr, te))
        if len(bloques) < 3:
            continue

        # Separacion real entre prueba y entrenamiento, y area del casco de cada bloque.
        seps, areas = [], []
        for tr, te in bloques:
            sub = centros[te]
            paso = max(1, sub.shape[0] // 300)
            paso_tr = max(1, tr.size // 300)
            muestra_tr = centros[tr][None, ::paso_tr, :]
            d = np.sqrt(((sub[::paso, None, :] - muestra_tr) ** 2).sum(-1))
            seps.append(float(d.min()))
            areas.append(float(metrico.iloc[te].union_all().convex_hull.area / 1e6))

        fila: dict[str, Any] = {
            "k": k,
            "bloques": len(bloques),
            "parcelas_min": int(min(t.size for _, t in bloques)),
            "separacion_min_km": round(min(seps) / 1000, 3),
            "separacion_mediana_km": round(float(np.median(seps)) / 1000, 3),
            "area_min_km2": round(min(areas), 1),
        }
        for s_min in SUELOS:
            clases, cobertura = [], []
            for _, te in bloques:
                cnt = np.bincount(labels[te], minlength=18)
                sobreviven = cnt >= s_min
                clases.append(int(sobreviven.sum()))
                cobertura.append(float(cnt[sobreviven].sum() / cnt.sum()))
            fila[f"clases_min_S{s_min}"] = min(clases)
            fila[f"parcelas_cubiertas_S{s_min}"] = round(min(cobertura), 4)
        filas.append(fila)

    # Banda de equivalencia anclada en el MDE, no en un numero redondo.
    diag = json.loads(
        (REPO_ROOT / "reports/paper_micai/diagnostico/diagnostico_protocolo.json").read_text()
    )
    por_bloque = diag["defecto_1_denominador_no_comun"]["por_bloque"]
    d = np.array([f["delta_denominador_comun"] for f in por_bloque])
    sd = float(d.std(ddof=1))
    bandas = {
        str(f["k"]): round(
            float(
                (stats.t.ppf(0.975, f["bloques"] - 1) + stats.t.ppf(0.80, f["bloques"] - 1))
                * sd
                / np.sqrt(f["bloques"])
            ),
            4,
        )
        for f in filas
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(OUT_DIR / "parametros_diseno.csv")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "parametros_prereg.json").write_text(
        json.dumps(
            {
                "para_que": (
                    "Fijar con un motivo los tres grados de libertad del preregistro: el suelo de "
                    "soporte por bloque, el criterio espacial de k, y la banda de equivalencia. "
                    "Este guion NO calcula ningun contraste entre mecanismos."
                ),
                "diseno_por_k": filas,
                "banda_equivalencia_por_k": bandas,
                "nota_banda": (
                    "La banda se ancla en el efecto minimo detectable del diseno: declarar una "
                    "equivalencia mas estrecha de lo que el diseno distingue seria afirmar sin "
                    "poder medir."
                ),
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
    for f in filas:
        logger.info("diseno", **f)


if __name__ == "__main__":
    main()
