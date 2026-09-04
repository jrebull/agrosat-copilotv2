"""Mide los parametros que el preregistro tiene que fijar, antes de fijarlos.

Un preregistro con huecos no es un preregistro, y uno con numeros inventados es peor. Este guion
mide propiedades del diseno y **no calcula ningun contraste entre mecanismos**.

Que se mide, y por que cada cosa esta aqui:

1. **El suelo de soporte por bloque** `S`: cuantas parcelas de una clase hacen falta en un bloque
   para que su F1 sea estimable, y cuantas clases y cuantas parcelas sobreviven a cada `S`.
2. **El criterio espacial de `k`**: la separacion real entre prueba y entrenamiento, exacta, y el
   area de bloque, para poder fijar el umbral sin mirar ningun contraste.
3. **El barrido de colchon**: si subir el colchon permite mas bloques sin romper el criterio. Estos
   numeros estaban solo en la prosa del preregistro; ahora tienen productor y artefacto.
4. **El solapamiento entre entrenamientos**: partir el mismo territorio mas fino no crea replicas
   independientes. Se mide con Jaccard, y se dice exactamente que conjuntos se comparan.
5. **El universo de clases por bloque bajo `S`**: la regla del suelo hace que cada bloque promedie
   sobre un conjunto de clases distinto. Es un denominador movil y hay que verlo antes de firmarlo.

**Lo que este guion ya NO hace, y es una correccion de la auditoria externa**: anclar la banda de
equivalencia en el efecto minimo detectable. Un margen de equivalencia sale de la menor diferencia
practicamente relevante para quien usa el mapa; anclarlo en la resolucion del instrumento hace que
sea el experimento el que define que cuenta como equivalente. La banda la fija US-174 con los casos
de uso, y hasta entonces no se produce ningun numero que pueda pasar por ella.

Uso:
    poetry run python scripts/run_paper_micai_parametros_prereg.py
"""

from __future__ import annotations

import itertools
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import polars as pl
import structlog
from scipy.spatial import cKDTree
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

#: Rejilla del barrido de colchon: la unica via legitima para ganar bloques por diseno.
K_COLCHON: tuple[int, ...] = (5, 8, 10, 12, 15)
COLCHONES_KM: tuple[float, ...] = (1.0, 5.0, 10.0, 15.0)

#: Suelo con el que se resume el barrido de colchon y el universo de clases por bloque.
S_REFERENCIA: int = 20

NUM_CLASSES: int = 18


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


def _bloques(
    gdf: gpd.GeoDataFrame, k: int, seed: int, buffer_km: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Usable blocks for a given ``k`` and buffer.

    Args:
        gdf: The parcel universe.
        k: Requested number of folds.
        seed: KMeans seed.
        buffer_km: Exclusion buffer between folds, in km.

    Returns:
        One ``(train_only, train_plus_val, test)`` triple per block with at least 100 test parcels.
    """
    salida = []
    for f in build_spatial_kfold(gdf, k=k, random_state=seed, buffer_km=buffer_km):
        te = np.asarray(sorted(int(x) for x in f.test_ids))
        tr = np.asarray(sorted(int(x) for x in f.train_ids))
        trv = np.asarray(sorted(int(x) for x in [*f.train_ids, *f.val_ids]))
        if te.size >= 100 and trv.size:
            salida.append((tr, trv, te))
    return salida


def _separacion_exacta_km(centros: np.ndarray, entrena: np.ndarray, prueba: np.ndarray) -> float:
    """Exact distance from the test block to its nearest training parcel, in km.

    The retired implementation subsampled both sides to about 300 points before taking the minimum.
    That is biased **upward** by construction: dropping candidates can only keep or raise a minimum,
    so the design looked more separated than it is. A KD-tree gives the true nearest neighbour at
    negligible cost, and there was never a reason to approximate it.

    Args:
        centros: Projected centroids of every parcel, in metres.
        entrena: Positional ids of the training parcels.
        prueba: Positional ids of the test parcels.

    Returns:
        The minimum test-to-train distance in kilometres.
    """
    arbol = cKDTree(centros[entrena])
    distancias, _ = arbol.query(centros[prueba], k=1)
    return float(distancias.min() / 1000.0)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two id sets."""
    sa, sb = set(a.tolist()), set(b.tolist())
    union = len(sa | sb)
    return float(len(sa & sb) / union) if union else 0.0


def main(seed: int = 42) -> None:
    """Measure the design properties the pre-registration has to declare."""
    gdf, labels = _universo()
    metrico = gdf.to_crs(3035)
    centros = np.column_stack([metrico.geometry.centroid.x, metrico.geometry.centroid.y])

    filas: list[dict[str, Any]] = []
    solapamiento: list[dict[str, Any]] = []
    universos: list[dict[str, Any]] = []
    for k in K_VALORES:
        bloques = _bloques(gdf, k, seed, buffer_km=1.0)
        if len(bloques) < 3:
            continue

        seps = [_separacion_exacta_km(centros, trv, te) for _, trv, te in bloques]
        areas = [float(metrico.iloc[te].union_all().convex_hull.area / 1e6) for _, _, te in bloques]

        fila: dict[str, Any] = {
            "k": k,
            "bloques": len(bloques),
            "parcelas_min": int(min(t.size for _, _, t in bloques)),
            "separacion_min_km": round(min(seps), 3),
            "separacion_mediana_km": round(float(np.median(seps)), 3),
            "area_min_km2": round(min(areas), 1),
        }
        for s_min in SUELOS:
            clases, cobertura = [], []
            for _, _, te in bloques:
                cnt = np.bincount(labels[te], minlength=NUM_CLASSES)
                sobreviven = cnt >= s_min
                clases.append(int(sobreviven.sum()))
                cobertura.append(float(cnt[sobreviven].sum() / cnt.sum()))
            fila[f"clases_min_S{s_min}"] = min(clases)
            fila[f"parcelas_cubiertas_S{s_min}"] = round(min(cobertura), 4)
        filas.append(fila)

        # Cuanto comparten los entrenamientos de dos bloques. Se reportan los dos conjuntos por
        # separado porque no son el mismo numero y confundirlos ya paso una vez.
        solo_tr = [_jaccard(a[0], b[0]) for a, b in itertools.combinations(bloques, 2)]
        tr_val = [_jaccard(a[1], b[1]) for a, b in itertools.combinations(bloques, 2)]
        solapamiento.append(
            {
                "k": k,
                "pares": len(solo_tr),
                "jaccard_medio_train": round(float(np.mean(solo_tr)), 4),
                "jaccard_max_train": round(float(np.max(solo_tr)), 4),
                "jaccard_medio_train_val": round(float(np.mean(tr_val)), 4),
                "jaccard_max_train_val": round(float(np.max(tr_val)), 4),
            }
        )

        # El denominador movil: que clases sobreviven al suelo en CADA bloque.
        por_bloque = []
        for _, _, te in bloques:
            cnt = np.bincount(labels[te], minlength=NUM_CLASSES)
            por_bloque.append(set(np.flatnonzero(cnt >= S_REFERENCIA).tolist()))
        pares = [len(a & b) / len(a | b) for a, b in itertools.combinations(por_bloque, 2) if a | b]
        universos.append(
            {
                "k": k,
                "clases_por_bloque": [len(s) for s in por_bloque],
                "en_todos_los_bloques": len(set.intersection(*por_bloque)) if por_bloque else 0,
                "en_la_union": len(set.union(*por_bloque)) if por_bloque else 0,
                "jaccard_min_entre_universos": round(float(min(pares)), 4) if pares else None,
                "jaccard_medio_entre_universos": round(float(np.mean(pares)), 4) if pares else None,
            }
        )

    # Barrido de colchon: la unica via legitima para ganar bloques por diseno, no por eleccion.
    colchon: list[dict[str, Any]] = []
    for k, km in itertools.product(K_COLCHON, COLCHONES_KM):
        bloques = _bloques(gdf, k, seed, buffer_km=km)
        if len(bloques) < 3:
            continue
        seps = [_separacion_exacta_km(centros, trv, te) for _, trv, te in bloques]
        clases = [
            int((np.bincount(labels[te], minlength=NUM_CLASSES) >= S_REFERENCIA).sum())
            for _, _, te in bloques
        ]
        colchon.append(
            {
                "k": k,
                "colchon_km": km,
                "bloques": len(bloques),
                "separacion_min_km": round(min(seps), 3),
                f"clases_min_S{S_REFERENCIA}": min(clases),
                # El colchon arregla la separacion comiendose las parcelas de los bordes: hay que
                # ver el precio en el mismo renglon que el beneficio.
                "parcelas_de_prueba": sum(int(te.size) for _, _, te in bloques),
            }
        )
        logger.info("colchon", k=k, colchon_km=km, separacion=round(min(seps), 3))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(filas).write_csv(OUT_DIR / "parametros_diseno.csv")
    pl.DataFrame(colchon).write_csv(OUT_DIR / "barrido_colchon.csv")
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
                    "Fijar con un motivo los grados de libertad del preregistro: el suelo de "
                    "soporte por bloque y el criterio espacial de k. Este guion NO calcula ningun "
                    "contraste entre mecanismos y NO produce banda de equivalencia."
                ),
                "diseno_por_k": filas,
                "barrido_colchon": colchon,
                "solapamiento_entre_bloques": solapamiento,
                "universo_de_clases_por_bloque": universos,
                "banda_equivalencia": None,
                "nota_banda": (
                    "RETIRADA tras la auditoria externa. La version anterior la anclaba en el "
                    "efecto minimo detectable del diseno, y eso hace que la resolucion del "
                    "instrumento defina que cuenta como equivalente. El margen sale de la menor "
                    "diferencia practicamente relevante para quien usa el mapa, y lo fija US-174 "
                    "con los casos de uso. Hasta entonces este artefacto no produce ningun numero "
                    "que pueda pasar por una banda."
                ),
                "nota_separacion": (
                    "Distancia exacta al vecino mas cercano con KD-tree. La version anterior "
                    "submuestreaba unos 300 puntos de cada lado antes de tomar el minimo, lo que "
                    "sesga el numero HACIA ARRIBA por construccion: quitar candidatos solo puede "
                    "mantener o subir un minimo. Con k=5 la separacion minima cae de 23,505 a "
                    "22,972 km y con k=8 de 2,877 a 1,975 km; el criterio no cambia de veredicto, "
                    "pero el numero publicado si."
                ),
                "nota_solapamiento": (
                    "Jaccard entre los conjuntos de ENTRENAMIENTO de dos bloques, y aparte entre "
                    "entrenamiento mas validacion. Son dos numeros distintos y confundirlos ya "
                    "paso: el 0,60 medio que se publico era el de train+val, no el de train."
                ),
                "nota_universos": (
                    "El suelo S por bloque hace que cada bloque promedie sobre un conjunto de "
                    "clases distinto, es decir un denominador movil, que es exactamente el defecto "
                    "que el articulo denuncia. Antes de firmar el suelo hay que elegir: universo "
                    "comun fijado desde entrenamiento, o estimando clase-por-bloque declarado con "
                    "su ponderacion. Este artefacto mide el tamano del problema, no lo resuelve."
                ),
                "procedencia": {
                    "semilla": seed,
                    "num_clases": NUM_CLASSES,
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
    for u in universos:
        logger.info("universo", **u)


if __name__ == "__main__":
    main()
