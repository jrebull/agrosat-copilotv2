"""Fase 2 del articulo MICAI: arbitraje, cobertura y nulo de vecindad.

Cada subcomando escribe un CSV o JSON sellable bajo ``reports/paper_micai/fase2/``
con su semilla, sus versiones de computo y el commit que lo produjo. Todos miden
sobre las mismas 16 640 parcelas del fold 5 y contra el ground truth sellado en
``reports/paper_micai/fase1/parcel_gt_fold5.parquet``.

Uso:
    poetry run python scripts/run_paper_micai_fase2.py arbitraje
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import structlog
import typer

from ml.eval.paper_micai_arbitration import (
    KEY,
    Combination,
    _macro_on_legend,
    combine_mean,
    combine_weighted,
    coverage_by_class_retirement,
    coverage_by_confidence,
    load_member_posteriors,
    mcnemar,
    miembros_del_panel,
    paired_bootstrap_delta,
    per_class_table,
    pooled_spatial_oof_posteriors,
    pooled_weighted_vote,
    score,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help="Experimentos de la fase 2 del articulo MICAI.")

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase2"

#: Los cinco miembros del campeon sellado (universo tsvit-pheno).
CHAMPION_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth",
    "farslip-ft18",
    "farslip-zeroshot",
)

#: Todos los miembros del universo de Francia, para la tabla de individuales.
#: Los miembros del analisis salen del PANEL CONGELADO, no de una lista escrita aqui.
#: Estaban los diez originales, y al congelar el panel en cinco este guion seguia pidiendo
#: los diez: cinco ya excluidos o sin verificar. Al regenerar habria usado el conjunto
#: equivocado sin que nada relacionara una cosa con la otra.
ALL_MEMBERS: tuple[str, ...] = miembros_del_panel()


def provenance(seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the provenance block every artefact of this phase carries.

    Args:
        seed: Random seed used by the experiment.
        extra: Additional fields to merge in.

    Returns:
        Dictionary with seed, commit, library versions and timestamp.
    """
    import sklearn

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    block: dict[str, Any] = {
        "semilla": seed,
        "code_version": head or "desconocido",
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
        "universo": "fold 5 held-out de PASTIS, parcelas compartidas por los miembros",
        "ground_truth": "reports/paper_micai/fase1/parcel_gt_fold5.parquet",
    }
    if extra:
        block.update(extra)
    return block


def _load_ground_truth() -> tuple[list[str], np.ndarray]:
    """Load the sealed fold-5 ground truth.

    Returns:
        Tuple of the ordered parcel ids and their labels.
    """
    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    return gt[KEY].to_list(), gt["label"].to_numpy()


def _spatial_splits(keys: list[str], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the geographic sub-folds of fold-5 over the sealed centroids.

    Args:
        keys: Ordered parcel ids defining the row positions.
        seed: Seed forwarded to the spatial split.

    Returns:
        ``(train_pos, test_pos)`` positional index pairs, one per block.
    """
    from ml.ensemble.stacking import StackingEnsemble

    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    helper = StackingEnsemble(base_members=CHAMPION_MEMBERS, oof_dir=OOF_DIR, random_state=seed)
    return helper._subfolds_by_canonical_id(geoms, pl.DataFrame({KEY: keys}))


def _projected_centroids(keys: list[str]) -> np.ndarray:
    """Project the sealed WGS84 centroids to Lambert-93 metres for the k-NN.

    Args:
        keys: Ordered parcel ids defining the row order.

    Returns:
        ``(n_parcels, 2)`` array of EPSG:2154 coordinates in metres.
    """
    from pyproj import Transformer

    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    order = pl.DataFrame({KEY: keys}).with_row_index("_pos")
    joined = order.join(geoms, on=KEY, how="left").sort("_pos")
    wkt = joined["geometry"].to_list()
    lon = np.array([float(p.split("(")[1].split()[0]) for p in wkt])
    lat = np.array([float(p.split("(")[1].split()[1].rstrip(")")) for p in wkt])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.column_stack([x, y])


@app.callback()
def main() -> None:
    """Agrupa los subcomandos de la fase 2 para que typer no colapse el primero."""


@app.command()
def arbitraje(
    seed: Annotated[int, typer.Option(help="Semilla del meta-modelo y del bootstrap.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos del bootstrap pareado.")] = 1000,
) -> None:
    """Compara promedio, voto ponderado y arbitro entrenado sobre los mismos miembros.

    Escribe la tabla de individuales, la de combinaciones y las pruebas pareadas.
    Reporta el arbitro en sus DOS regimenes: el refit sobre todas las parcelas, que
    es el que produjo las cifras selladas, y el agrupado espacial libre de fuga.
    """
    keys, labels = _load_ground_truth()
    members = load_member_posteriors(OOF_DIR, ALL_MEMBERS, keys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    individuals = [
        {"miembro": name, **score(labels, proba), "regimen": "held-out"}
        for name, proba in sorted(members.items())
    ]
    pl.DataFrame(individuals).sort("f1_macro", descending=True).write_csv(
        OUT_DIR / "individuales_protocolo_unico.csv"
    )

    champion = [members[name] for name in CHAMPION_MEMBERS]
    splits = _spatial_splits(keys, seed)
    weighted, weighted_covered = pooled_weighted_vote(champion, labels, splits)
    combos: list[Combination] = [
        Combination("promedio simple", combine_mean(champion), "held-out"),
        Combination("voto ponderado, pesos de otros bloques", weighted, "held-out"),
        Combination(
            "voto ponderado, pesos del propio conjunto",
            combine_weighted(
                champion,
                [score(labels, members[name])["f1_macro"] for name in CHAMPION_MEMBERS],
            ),
            "in-sample",
        ),
    ]

    from ml.ensemble.stacking import StackingEnsemble

    stack = StackingEnsemble(
        base_members=CHAMPION_MEMBERS,
        meta="logreg",
        n_spatial_folds=5,
        oof_dir=OOF_DIR,
        random_state=seed,
    )
    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    gt_frame = pl.DataFrame({KEY: keys, "label": labels})
    stack.fit(geoms, gt_labels=gt_frame)
    meta_keys, meta_x, meta_y = stack.build_meta_features(gt_labels=gt_frame)
    assert meta_keys[KEY].to_list() == keys, "el orden del meta-modelo no casa con el GT"
    combos.append(
        Combination("arbitro entrenado, refit en todo", stack.predict_proba(), "in-sample")
    )

    splits = _spatial_splits(keys, seed)
    pooled, covered = pooled_spatial_oof_posteriors(meta_x, meta_y, splits, random_state=seed)
    combos.append(Combination("arbitro entrenado, agrupado espacial", pooled, "held-out"))

    rows = []
    for combo in combos:
        if combo.name.endswith("agrupado espacial"):
            mask = covered
        elif combo.name.startswith("voto ponderado, pesos de otros"):
            mask = weighted_covered
        else:
            mask = np.ones_like(covered)
        rows.append(
            {
                "combinacion": combo.name,
                "regimen": combo.regime,
                "n_parcelas": int(mask.sum()),
                **score(labels[mask], combo.proba[mask]),
            }
        )
    pl.DataFrame(rows).write_csv(OUT_DIR / "combinaciones.csv")

    best_individual = max(individuals, key=lambda r: r["f1_macro"])
    pooled_pred = pooled.argmax(axis=1)
    mean_pred = combine_mean(champion).argmax(axis=1)
    weighted_pred = weighted.argmax(axis=1)
    best_pred = members[best_individual["miembro"]].argmax(axis=1)

    tests = {
        "arbitro_agrupado_vs_promedio": {
            **paired_bootstrap_delta(
                labels[covered],
                pooled_pred[covered],
                mean_pred[covered],
                n_boot=n_boot,
                random_state=seed,
            ),
            **mcnemar(labels[covered], pooled_pred[covered], mean_pred[covered]),
        },
        "arbitro_agrupado_vs_mejor_individual": {
            **paired_bootstrap_delta(
                labels[covered],
                pooled_pred[covered],
                best_pred[covered],
                n_boot=n_boot,
                random_state=seed,
            ),
            **mcnemar(labels[covered], pooled_pred[covered], best_pred[covered]),
            "mejor_individual": best_individual["miembro"],
        },
        "arbitro_agrupado_vs_voto_ponderado": {
            **paired_bootstrap_delta(
                labels[covered],
                pooled_pred[covered],
                weighted_pred[covered],
                n_boot=n_boot,
                random_state=seed,
            ),
            **mcnemar(labels[covered], pooled_pred[covered], weighted_pred[covered]),
        },
        "promedio_vs_mejor_individual": {
            **paired_bootstrap_delta(
                labels, mean_pred, best_pred, n_boot=n_boot, random_state=seed
            ),
            **mcnemar(labels, mean_pred, best_pred),
            "mejor_individual": best_individual["miembro"],
        },
    }
    per_class_table(
        labels,
        {
            "mejor_individual": best_pred,
            "promedio": mean_pred,
            "voto_ponderado": weighted_pred,
            "arbitro_agrupado": pooled_pred,
        },
    ).write_csv(OUT_DIR / "arbitraje_por_clase.csv")

    payload = {
        "pruebas": tests,
        "parcelas_con_prediccion_libre_de_fuga": int(covered.sum()),
        "parcelas_totales": int(covered.size),
        "miembros_campeon": list(CHAMPION_MEMBERS),
        "procedencia": provenance(seed, {"n_boot": n_boot}),
    }
    (OUT_DIR / "arbitraje_pruebas.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pl.DataFrame(
        {
            KEY: keys,
            "cubierta": covered,
            **{f"prob_{i:03d}": pooled[:, i] for i in range(pooled.shape[1])},
        }
    ).write_parquet(OUT_DIR / "arbitro_agrupado_posteriores.parquet", compression="zstd")
    logger.info("arbitraje_done", out=str(OUT_DIR))


@app.command()
def cobertura(
    seed: Annotated[int, typer.Option(help="Semilla del bootstrap y de los bloques.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos del bootstrap pareado.")] = 1000,
) -> None:
    """Contrasta los dos mecanismos de recorte de cobertura a igual cobertura.

    Retirar clases enteras de la leyenda frente a rechazar parcelas por confianza,
    ambos con la decision tomada fuera del bloque que se mide, sobre el mejor
    predictor libre de fuga. Los dos entregan productos distintos --- una leyenda
    mas corta en todas sus parcelas frente a la leyenda completa en menos parcelas
    --- asi que la comparacion es de la frontera calidad-cobertura, no de igualdad.
    """
    keys, labels = _load_ground_truth()
    members = load_member_posteriors(OOF_DIR, ALL_MEMBERS, keys)
    splits = _spatial_splits(keys, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    individual_scores = {name: score(labels, proba)["f1_macro"] for name, proba in members.items()}
    best_name = max(individual_scores, key=lambda name: individual_scores[name])
    proba = members[best_name]
    logger.info("cobertura_base", predictor=best_name, f1=round(individual_scores[best_name], 4))

    k_values = (18, 16, 14, 12, 10, 9, 8)
    retirement = coverage_by_class_retirement(proba, labels, splits, k_values)

    targets: dict[int, list[float]] = {}
    for record in retirement:
        targets.setdefault(int(record["bloque"]), []).append(float(record["cobertura"]))
    confidence = coverage_by_confidence(proba, labels, splits, targets)

    rows: list[dict[str, object]] = []
    for record in retirement:
        rows.append(
            {
                "mecanismo": record["mecanismo"],
                "k_leyenda": record["k"],
                "bloque": record["bloque"],
                "n_bloque": record["n_bloque"],
                "n_entregadas": record["n_entregadas"],
                "cobertura": round(float(record["cobertura"]), 6),
                "f1_macro": round(float(record["f1_macro"]), 6),
                "accuracy": round(float(record["accuracy"]), 6),
                "leyenda": ",".join(str(c) for c in record["leyenda"]),
            }
        )
    for record in confidence:
        rows.append(
            {
                "mecanismo": record["mecanismo"],
                "k_leyenda": k_values[int(record["punto"])],
                "bloque": record["bloque"],
                "n_bloque": record["n_bloque"],
                "n_entregadas": record["n_entregadas"],
                "cobertura": round(float(record["cobertura"]), 6),
                "f1_macro": round(float(record["f1_macro"]), 6),
                "accuracy": round(float(record["accuracy"]), 6),
                "leyenda": "completa",
            }
        )
    frame = pl.DataFrame(rows)
    frame.sort(["k_leyenda", "mecanismo", "bloque"], descending=[True, False, False]).write_csv(
        OUT_DIR / "cobertura_por_bloque.csv"
    )

    summary = (
        frame.group_by(["k_leyenda", "mecanismo"])
        .agg(
            pl.col("cobertura").mean().alias("cobertura_media"),
            pl.col("f1_macro").mean().alias("f1_macro_medio"),
            pl.col("f1_macro").min().alias("f1_macro_min"),
            pl.col("f1_macro").max().alias("f1_macro_max"),
            pl.col("accuracy").mean().alias("accuracy_media"),
            pl.col("n_entregadas").sum().alias("n_entregadas_total"),
        )
        .sort(["k_leyenda", "mecanismo"], descending=[True, False])
    )
    summary.write_csv(OUT_DIR / "cobertura_resumen.csv")

    rng = np.random.default_rng(seed)
    comparisons: dict[str, dict[str, float]] = {}
    for k in k_values:
        ret = [r for r in retirement if r["k"] == k]
        conf = [c for c in confidence if k_values[int(c["punto"])] == k]
        observed = float(
            np.mean([r["f1_macro"] for r in ret]) - np.mean([c["f1_macro"] for c in conf])
        )
        draws = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            ret_means, conf_means = [], []
            for r, c in zip(ret, conf, strict=True):
                pos_r = np.asarray(r["posiciones_entregadas"])
                pos_c = np.asarray(c["posiciones_entregadas"])
                idx_r = rng.integers(0, pos_r.size, size=pos_r.size) if pos_r.size else None
                idx_c = rng.integers(0, pos_c.size, size=pos_c.size) if pos_c.size else None
                legend = list(r["leyenda"])
                columns = np.asarray(legend, dtype=int)
                if idx_r is not None:
                    sel = pos_r[idx_r]
                    emitted = columns[proba[np.ix_(sel, columns)].argmax(axis=1)]
                    ret_means.append(_macro_on_legend(labels[sel], emitted, legend))
                if idx_c is not None:
                    sel = pos_c[idx_c]
                    emitted = proba[sel].argmax(axis=1)
                    conf_means.append(
                        _macro_on_legend(labels[sel], emitted, sorted(set(labels[sel].tolist())))
                    )
            draws[i] = float(np.mean(ret_means) - np.mean(conf_means))
        low, high = np.percentile(draws, [2.5, 97.5])
        comparisons[f"k={k}"] = {
            "f1_retirada_medio": float(np.mean([r["f1_macro"] for r in ret])),
            "f1_confianza_medio": float(np.mean([c["f1_macro"] for c in conf])),
            "delta_retirada_menos_confianza": observed,
            "ci_low": float(low),
            "ci_high": float(high),
            "excluye_cero": float(low > 0 or high < 0),
            "cobertura_retirada_media": float(np.mean([r["cobertura"] for r in ret])),
            "cobertura_confianza_media": float(np.mean([c["cobertura"] for c in conf])),
            "clases_en_todas_las_leyendas": float(
                len(set(ret[0]["leyenda"]).intersection(*[set(r["leyenda"]) for r in ret[1:]]))
            ),
            "clases_en_alguna_leyenda": float(len({c for r in ret for c in r["leyenda"]})),
        }

    payload = {
        "predictor": best_name,
        "f1_macro_completo": individual_scores[best_name],
        "comparaciones": comparisons,
        "nota": (
            "Los dos mecanismos entregan productos distintos: la retirada de clases promete "
            "una leyenda mas corta en todas sus parcelas y el rechazo por confianza promete la "
            "leyenda completa en menos parcelas. La comparacion es de la frontera "
            "calidad-cobertura, no una prueba de equivalencia. Cada bloque se mide contra UNA "
            "leyenda, elegida en los otros bloques; las leyendas de bloques distintos no "
            "coinciden, y esa inestabilidad se reporta como parte del resultado."
        ),
        "procedencia": provenance(seed, {"n_boot": n_boot}),
    }
    (OUT_DIR / "cobertura_comparacion.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("cobertura_done", out=str(OUT_DIR))


@app.command()
def vecindad(
    seed: Annotated[int, typer.Option(help="Semilla del bootstrap y de los bloques.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos del bootstrap pareado.")] = 1000,
) -> None:
    """Nulo de vecindad con intervalo, sobre predictores libres de fuga (regla R1).

    Mezcla la posterior de cada parcela con la media de sus k vecinos por centroide
    y mide si eso aporta algo. El punto de operacion (k, alfa) se elige en los otros
    bloques y se aplica al bloque medido, porque elegirlo mirando el resultado seria
    escoger el maximo del ruido. Solo se usan posteriores de vecinos, nunca sus
    etiquetas.
    """
    from ml.ensemble.ec_neighborhood import _knn_indices, _refine

    keys, labels = _load_ground_truth()
    splits = _spatial_splits(keys, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coords = _projected_centroids(keys)
    k_values = (5, 10, 20)
    alphas = (0.0, 0.1, 0.2, 0.3, 0.5)
    neighbor_idx = _knn_indices(coords, max(k_values))

    members = load_member_posteriors(OOF_DIR, ALL_MEMBERS, keys)
    best_name = max(members, key=lambda name: score(labels, members[name])["f1_macro"])
    arbiter = pl.read_parquet(OUT_DIR / "arbitro_agrupado_posteriores.parquet").sort(KEY)
    bases = {
        f"mejor individual ({best_name})": members[best_name],
        "arbitro agrupado": arbiter.select([f"prob_{i:03d}" for i in range(18)]).to_numpy(),
    }

    sweep_rows: list[dict[str, object]] = []
    verdicts: dict[str, dict[str, float]] = {}
    for base_name, base in bases.items():
        for k in k_values:
            for alpha in alphas:
                refined = _refine(base, neighbor_idx, k, alpha)
                metrics = score(labels, refined)
                sweep_rows.append(
                    {
                        "base": base_name,
                        "k": k,
                        "alfa": alpha,
                        "f1_macro": round(metrics["f1_macro"], 6),
                        "accuracy": round(metrics["accuracy"], 6),
                    }
                )

        selected = np.zeros_like(base)
        chosen: list[dict[str, float]] = []
        for train_pos, test_pos in splits:
            if train_pos.size == 0 or test_pos.size == 0:
                continue
            best_point, best_value = (k_values[0], 0.0), -1.0
            for k in k_values:
                for alpha in alphas:
                    value = score(
                        labels[train_pos], _refine(base, neighbor_idx, k, alpha)[train_pos]
                    )["f1_macro"]
                    if value > best_value:
                        best_point, best_value = (k, alpha), value
            selected[test_pos] = _refine(base, neighbor_idx, *best_point)[test_pos]
            chosen.append({"k": float(best_point[0]), "alfa": float(best_point[1])})

        tests = paired_bootstrap_delta(
            labels,
            selected.argmax(axis=1),
            base.argmax(axis=1),
            n_boot=n_boot,
            random_state=seed,
        )
        verdicts[base_name] = {
            **tests,
            **mcnemar(labels, selected.argmax(axis=1), base.argmax(axis=1)),
            "f1_base": score(labels, base)["f1_macro"],
            "f1_refinado": score(labels, selected)["f1_macro"],
            "alfa_cero_en_algun_bloque": float(any(c["alfa"] == 0.0 for c in chosen)),
            "puntos_elegidos": float(len({(c["k"], c["alfa"]) for c in chosen})),
        }

    pl.DataFrame(sweep_rows).write_csv(OUT_DIR / "vecindad_barrido.csv")
    payload = {
        "veredictos": verdicts,
        "ks": list(k_values),
        "alfas": list(alphas),
        "nota": (
            "El punto de operacion se elige en los bloques que no se miden. Un delta cuyo "
            "intervalo cruza el cero es un nulo acotado; si lo excluye, la regla R1 de ADR-013 "
            "obliga a reportarlo como mejora pequena y no accionable, con su cifra."
        ),
        "procedencia": provenance(seed, {"n_boot": n_boot}),
    }
    (OUT_DIR / "vecindad_veredicto.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("vecindad_done", out=str(OUT_DIR))


@app.command()
def farslip(
    seed: Annotated[int, typer.Option(help="Semilla del meta-modelo y del bootstrap.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos del bootstrap pareado.")] = 1000,
) -> None:
    """Aporte de las dos ramas FarSLIP: cinco miembros frente a tres, con intervalo.

    Se mide en un solo universo, el del campeon sellado (tsvit-pheno como TSViT),
    porque mezclar universos fue lo que hizo ilegible la rejilla heredada. Ambos
    ensambles se estiman con el mismo agrupado espacial libre de fuga, de modo que
    la unica diferencia entre las dos columnas son los dos miembros contrastivos.
    """
    keys, labels = _load_ground_truth()
    splits = _spatial_splits(keys, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_frame = pl.DataFrame({KEY: keys, "label": labels})
    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")

    from ml.ensemble.stacking import StackingEnsemble

    three = CHAMPION_MEMBERS[:3]
    variants: dict[str, tuple[str, ...]] = {
        "tres miembros": three,
        "cinco miembros": CHAMPION_MEMBERS,
    }
    pooled: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for name, members in variants.items():
        stack = StackingEnsemble(
            base_members=members,
            meta="logreg",
            n_spatial_folds=5,
            oof_dir=OOF_DIR,
            random_state=seed,
        )
        stack.fit(geoms, gt_labels=gt_frame)
        meta_keys, meta_x, meta_y = stack.build_meta_features(gt_labels=gt_frame)
        assert meta_keys[KEY].to_list() == keys, "el orden del meta-modelo no casa con el GT"
        proba, _ = pooled_spatial_oof_posteriors(meta_x, meta_y, splits, random_state=seed)
        pooled[name] = proba
        rows.append(
            {
                "variante": name,
                "miembros": ",".join(members),
                "regimen": "held-out, agrupado espacial",
                **score(labels, proba),
            }
        )
        rows.append(
            {
                "variante": f"{name} (refit sobre todo)",
                "miembros": ",".join(members),
                "regimen": "in-sample",
                **score(labels, stack.predict_proba()),
            }
        )
    pl.DataFrame(rows).write_csv(OUT_DIR / "farslip_cinco_vs_tres.csv")

    delta = {
        **paired_bootstrap_delta(
            labels,
            pooled["cinco miembros"].argmax(axis=1),
            pooled["tres miembros"].argmax(axis=1),
            n_boot=n_boot,
            random_state=seed,
        ),
        **mcnemar(
            labels,
            pooled["cinco miembros"].argmax(axis=1),
            pooled["tres miembros"].argmax(axis=1),
        ),
    }
    payload = {
        "universo": "tsvit-pheno (el del campeon sellado)",
        "miembros_tres": list(three),
        "miembros_cinco": list(CHAMPION_MEMBERS),
        "delta_cinco_menos_tres": delta,
        "nota": (
            "Se elige este universo porque es el del campeon sellado y porque el otro "
            "(tsvit-pheno-fullm) parte de un TSViT que rinde 0,2552 sobre estas mismas "
            "parcelas, de modo que su delta mediria sobre todo el hueco que dejan los "
            "miembros debiles."
        ),
        "procedencia": provenance(seed, {"n_boot": n_boot}),
    }
    (OUT_DIR / "farslip_delta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("farslip_done", out=str(OUT_DIR))


if __name__ == "__main__":
    app()
