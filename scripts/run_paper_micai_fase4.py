"""Fase 4 del articulo MICAI: replica del protocolo sobre BreizhCrops.

Corre el MISMO protocolo de la fase 3, con el mismo modulo y sin tocar una linea de su
metodo, sobre otra region de Francia, otro anio y otro reparto de clases. Lo unico que
cambia son las adaptaciones que la enmienda 1 del preregistro declaro antes de entrenar:
los bloques son las dos regiones, porque BreizhCrops no trae coordenadas por parcela, y el
contraste se reporta en dos universos, las nueve clases tal como vienen y las siete con
soporte de al menos cien parcelas, cada uno con su propio criterio principal.

El clasificador se entrena dejando una region fuera, asi que la posterior de cada parcela
viene de un modelo que no vio su region.

Uso:
    poetry run python scripts/run_paper_micai_fase4.py replica
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import structlog
import typer
from sklearn.metrics import accuracy_score, f1_score

from ml.eval.paper_micai_coverage import (
    BlockPoint,
    confidence_baseline,
    frontier,
    holm,
    legend_by_f1,
    legend_by_support,
    no_mechanism_reference,
    paired_interval,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help="Fase 4 del articulo MICAI: replica externa.")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase4"

#: Soporte minimo para que una clase sea mensurable, fijado en la enmienda 1.
SOPORTE_MINIMO: int = 100

#: Nombres de los dos universos que la enmienda 1 obliga a publicar.
UNIVERSO_COMPLETO: str = "todas las clases"
UNIVERSO_MENSURABLE: str = f"clases con al menos {SOPORTE_MINIMO} parcelas"


def _k_principal(c_total: int) -> int:
    """Apply the pre-registered principal criterion ``K = round(C / 2)``.

    Rounds half up, which is what amendment 1 spelled out when it wrote K = 5 for nine
    classes and K = 4 for seven; Python's own ``round`` breaks ties to even and would
    return four in both cases.

    Args:
        c_total: Number of classes in this universe.

    Returns:
        The legend size of the principal criterion.
    """
    return int(c_total // 2 + c_total % 2)


def _provenance(seed: int, n_boot: int) -> dict[str, Any]:
    """Build the provenance block of this phase.

    Args:
        seed: Random seed of the classifier and of the resampling.
        n_boot: Bootstrap resamples.

    Returns:
        Dictionary with seed, commit, library versions and timestamp.
    """
    import sklearn
    import xgboost

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "semilla": seed,
        "n_boot": n_boot,
        "code_version": head or "desconocido",
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
        "criterio_principal": "K = round(C / 2) por universo, redondeando hacia arriba",
        "soporte_minimo": SOPORTE_MINIMO,
        "no_finitos": (
            "Las celdas no finitas se imputan con la mediana de la region de ENTRENAMIENTO, "
            "que es el tratamiento que ya usa el baseline tabular del proyecto y deja la "
            "region evaluada fuera de la imputacion."
        ),
        "features": "reports/paper_micai/fase4/breizhcrops_features.parquet",
    }


def _leave_one_region_out(
    table: pl.DataFrame, feature_cols: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Train one classifier per region on the other one and predict the held-out region.

    Args:
        table: Feature table carrying ``region`` and ``class_id``.
        feature_cols: Columns fed to the model.
        seed: Seed of the classifier.

    Non-finite cells are imputed with the median of the TRAINING region only, which is the
    same treatment the project's tabular baseline gives them and keeps the held-out region
    out of the imputation.

    Returns:
        Posterior matrix, labels, region index per parcel, region names and a per-region
        training log.
    """
    from xgboost import XGBClassifier

    from ml.train.baseline import _column_medians, _impute_with

    regions = sorted(table["region"].unique().to_list())
    labels = table["class_id"].cast(pl.Int64).to_numpy()
    n_classes = int(labels.max()) + 1
    features = table.select(feature_cols).to_numpy().astype(np.float64)
    block = np.array([regions.index(r) for r in table["region"].to_list()])
    logger.info(
        "celdas_no_finitas",
        celdas=int((~np.isfinite(features)).sum()),
        parcelas=int((~np.isfinite(features)).any(axis=1).sum()),
        columnas=int((~np.isfinite(features)).any(axis=0).sum()),
    )

    proba = np.zeros((table.height, n_classes), dtype=np.float64)
    log: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        test = block == index
        train = ~test
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        medians = _column_medians(features[train])
        x_train = _impute_with(features[train], medians)
        x_test = _impute_with(features[test], medians)
        present = sorted(set(labels[train].tolist()))
        remap = {c: i for i, c in enumerate(present)}
        model.fit(x_train, np.array([remap[c] for c in labels[train]]))
        local = np.asarray(model.predict_proba(x_test))
        proba[np.ix_(np.flatnonzero(test), np.asarray(present, dtype=int))] = local
        entry = {
            "region_evaluada": region,
            "n_entrenamiento": int(train.sum()),
            "n_evaluacion": int(test.sum()),
            "clases_en_entrenamiento": present,
            "f1_macro": float(
                f1_score(labels[test], proba[test].argmax(axis=1), average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(labels[test], proba[test].argmax(axis=1))),
        }
        log.append(entry)
        logger.info("region_predicha", **entry)
    return proba, labels, block, regions, log


def _restrict(
    proba: np.ndarray, labels: np.ndarray, block: np.ndarray, classes: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the universe down to a class subset, renormalising the posterior.

    Rows whose label falls outside the subset leave the universe, and the posterior keeps
    only the retained columns, renormalised so that confidence still means confidence
    inside this universe. Labels come back remapped to ``0..len(classes) - 1`` so that the
    protocol sees a dense label space, exactly as it saw on the primary dataset.

    Args:
        proba: Posterior over the full class space.
        labels: Ground-truth labels in the full class space.
        block: Block index per parcel.
        classes: Classes that stay.

    Returns:
        Restricted posterior, remapped labels and block index, all aligned.
    """
    columns = np.asarray(classes, dtype=int)
    rows = np.flatnonzero(np.isin(labels, columns))
    sub = proba[np.ix_(rows, columns)]
    total = sub.sum(axis=1, keepdims=True)
    flat = total[:, 0] <= 0.0
    sub = np.where(flat[:, None], 1.0 / len(classes), sub / np.where(total > 0.0, total, 1.0))
    remap = {c: i for i, c in enumerate(classes)}
    return sub, np.array([remap[int(c)] for c in labels[rows]]), block[rows]


def _rows(points: list[BlockPoint], universo: str, regions: list[str]) -> list[dict[str, Any]]:
    """Flatten block points into CSV rows.

    Args:
        points: Points to flatten.
        universo: Name of the label-space variant.
        regions: Region name per block index.

    Returns:
        One dictionary per point.
    """
    return [
        {
            "universo": universo,
            "mecanismo": p.mechanism,
            "k_leyenda": p.k,
            "bloque": regions[p.block],
            "n_entregadas": int(p.delivered.sum()),
            "cobertura": round(float(p.delivered.mean()), 6),
            "f1_alineado": round(p.aligned_f1, 6),
            "f1_nativo": round(p.native_f1, 6),
            "accuracy": round(p.accuracy, 6),
            "leyenda": ",".join(str(c) for c in p.legend),
        }
        for p in points
    ]


def _legend_f1(
    labels: np.ndarray, free: np.ndarray, num_classes: int, train_pos: np.ndarray, k: int
) -> tuple[int, ...]:
    """Adapt :func:`legend_by_f1` to the ``(train_pos, k)`` signature ``frontier`` expects.

    Args:
        labels: Ground-truth labels.
        free: Unrestricted predictions.
        num_classes: Size of this universe's label space.
        train_pos: Blocks the decision is taken on.
        k: Legend size.

    Returns:
        The promised classes.
    """
    return legend_by_f1(labels, free, train_pos, k, num_classes=num_classes)


def _legend_soporte(
    labels: np.ndarray, num_classes: int, train_pos: np.ndarray, k: int
) -> tuple[int, ...]:
    """Adapt :func:`legend_by_support` to the signature ``frontier`` expects.

    Args:
        labels: Ground-truth labels.
        num_classes: Size of this universe's label space.
        train_pos: Blocks the decision is taken on.
        k: Legend size.

    Returns:
        The promised classes.
    """
    return legend_by_support(labels, train_pos, k, num_classes=num_classes)


@app.callback()
def main() -> None:
    """Agrupa los subcomandos para que typer no colapse el primero."""


@app.command()
def replica(
    seed: Annotated[int, typer.Option(help="Semilla del clasificador y del remuestreo.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos pareados.")] = 1000,
) -> None:
    """Replica el protocolo de la fase 3 sobre BreizhCrops, en los dos universos declarados."""
    table = pl.read_parquet(OUT_DIR / "breizhcrops_features.parquet")
    meta_cols = {"parcel_id", "year", "class_id", "class_name", "region"}
    feature_cols = [c for c in table.columns if c not in meta_cols]
    logger.info("tabla", parcelas=table.height, features=len(feature_cols))

    proba, labels, block, regions, entrenamiento = _leave_one_region_out(table, feature_cols, seed)
    names = (
        table.select("class_id", "class_name")
        .unique(subset=["class_id"])
        .sort("class_id")
        .to_dicts()
    )
    pl.DataFrame(
        {
            "parcel_id": table["parcel_id"],
            "region": table["region"],
            "class_id": labels,
            **{f"prob_{i:03d}": proba[:, i] for i in range(proba.shape[1])},
        }
    ).write_parquet(OUT_DIR / "breizhcrops_posteriores.parquet", compression="zstd")

    conteo = np.bincount(labels, minlength=int(labels.max()) + 1)
    universos = {
        UNIVERSO_COMPLETO: [c for c in range(len(conteo)) if conteo[c] > 0],
        UNIVERSO_MENSURABLE: [c for c in range(len(conteo)) if conteo[c] >= SOPORTE_MINIMO],
    }
    logger.info(
        "universos",
        soporte=conteo.tolist(),
        completo=len(universos[UNIVERSO_COMPLETO]),
        mensurable=len(universos[UNIVERSO_MENSURABLE]),
    )

    rows: list[dict[str, Any]] = []
    contrastes: dict[str, Any] = {}
    for nombre, clases in universos.items():
        sub_proba, sub_labels, sub_block = _restrict(proba, labels, block, clases)
        sub_splits = [
            (np.flatnonzero(sub_block != b), np.flatnonzero(sub_block == b))
            for b in range(len(regions))
        ]
        c_total = len(clases)
        k_values = tuple(range(c_total, 2, -1))
        k_principal = _k_principal(c_total)
        free = sub_proba.argmax(axis=1)

        por_f1 = frontier(
            sub_proba,
            sub_labels,
            sub_splits,
            k_values,
            legend_fn=partial(_legend_f1, sub_labels, free, c_total),
            mechanism="retirada por F1",
        )
        por_soporte = frontier(
            sub_proba,
            sub_labels,
            sub_splits,
            k_values,
            legend_fn=partial(_legend_soporte, sub_labels, c_total),
            mechanism="retirada por soporte",
        )
        confianza = confidence_baseline(
            sub_proba, sub_labels, sub_splits, por_f1, num_classes=c_total
        )
        sin_mec = no_mechanism_reference(sub_proba, sub_labels, sub_splits, por_f1)

        for grupo in (por_f1, por_soporte, confianza, sin_mec):
            rows.extend(_rows(grupo, nombre, regions))

        crudos: list[float] = []
        detalle: dict[str, Any] = {}
        for k in k_values:
            izq = [p for p in por_f1 if p.k == k]
            der = [p for p in confianza if p.k == k]
            sop = [p for p in por_soporte if p.k == k]
            ref = [p for p in sin_mec if p.k == k]
            # Unidad declarada: el bloque. Aqui los bloques son las dos regiones, que son dos,
            # y el intervalo lo dice en vez de disimularlo remuestreando parcelas.
            blo = paired_interval(sub_labels, sub_splits, izq, der, unidad="bloque")
            par = paired_interval(
                sub_labels,
                sub_splits,
                izq,
                der,
                unidad="parcela",
                n_boot=n_boot,
                random_state=seed,
            )
            crudos.append(blo["p_valor"])
            detalle[f"k={k}"] = {
                "f1_retirada_f1": float(np.mean([p.aligned_f1 for p in izq])),
                "f1_retirada_soporte": float(np.mean([p.aligned_f1 for p in sop])),
                "f1_confianza": float(np.mean([p.aligned_f1 for p in der])),
                "f1_sin_mecanismo": float(np.mean([p.aligned_f1 for p in ref])),
                "cobertura_media": float(np.mean([p.delivered.mean() for p in izq])),
                "delta_vs_confianza": blo["delta"],
                # El intervalo publicable: unidad bloque. Aqui hay dos bloques, y se dice.
                "ci_low": blo["ci_low"],
                "ci_high": blo["ci_high"],
                "excluye_cero": blo["excluye_cero"],
                "p_valor": blo["p_valor"],
                "unidad_del_intervalo": blo["unidad"],
                "n_bloques": blo["n_unidades"],
                "deltas_por_bloque": blo["deltas_por_bloque"],
                "ci_low_remuestreo_parcela": par["ci_low"],
                "ci_high_remuestreo_parcela": par["ci_high"],
                "p_bootstrap_parcela": par["p_bootstrap"],
            }
        for k, adj in zip(k_values, holm(crudos), strict=True):
            detalle[f"k={k}"]["p_holm"] = adj
            detalle[f"k={k}"]["significativo_holm"] = float(adj < 0.05)

        contrastes[nombre] = {
            "clases_originales": clases,
            "n_parcelas": int(sub_labels.size),
            "k_principal": k_principal,
            "criterio_principal": detalle[f"k={k_principal}"],
            "familia_exploratoria": detalle,
            "f1_macro_sin_recortar": float(
                f1_score(sub_labels, free, average="macro", zero_division=0)
            ),
            "accuracy_sin_recortar": float(accuracy_score(sub_labels, free)),
        }

    pl.DataFrame(rows).write_csv(OUT_DIR / "replica_por_bloque.csv")
    (OUT_DIR / "replica_contrastes.json").write_text(
        json.dumps(
            {
                "conjunto": "BreizhCrops 2017 L2A, regiones frh01 y frh04",
                "clases": names,
                "entrenamiento": entrenamiento,
                "contrastes": contrastes,
                "notas": {
                    "protocolo": (
                        "Mismo modulo y mismo metodo que la fase 3, sin tocar una linea. Cambian "
                        "solo las adaptaciones que la enmienda 1 declaro antes de entrenar."
                    ),
                    "bloques": (
                        "Los bloques son las dos regiones. Con dos bloques el remuestreo captura "
                        "la variacion dentro de cada region, no entre regiones; por eso se "
                        "publican tambien los dos deltas por bloque, que es donde se ve si las "
                        "dos regiones dicen lo mismo."
                    ),
                    "universos": (
                        "El universo de siete clases retiene las de al menos cien parcelas y "
                        "renormaliza la posterior sobre ellas, para que la confianza siga "
                        "significando confianza dentro del universo que se puntua."
                    ),
                },
                "procedencia": _provenance(seed, n_boot),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("replica_done", out=str(OUT_DIR))


if __name__ == "__main__":
    app()
