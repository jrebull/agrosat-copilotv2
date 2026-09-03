"""Fase 3 del articulo MICAI: la frontera calidad-cobertura, rehecha tras la auditoria.

Rehace el experimento que la auditoria ciega retiro, con los tres defectos corregidos
(estimando alineado, entrega sin oraculo de etiqueta, remuestreo pareado) y con lo que el
preregistro anade: un segundo predictor, un tercer mecanismo anclado en la practica del
equipo, la correccion por multiplicidad y el control sin mecanismo.

Salidas selladas en ``reports/paper_micai/fase3/``.

Uso:
    poetry run python scripts/run_paper_micai_fase3.py
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

from ml.eval.paper_micai_arbitration import KEY, load_member_posteriors, score
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
app = typer.Typer(add_completion=False, help="Fase 3 del articulo MICAI.")

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase3"

#: Los diez miembros del universo de Francia.
ALL_MEMBERS: tuple[str, ...] = (
    "anysat",
    "deeplabv3plus",
    "farslip-ft18",
    "farslip-zeroshot",
    "segformer",
    "tsvit-pheno",
    "tsvit-pheno-fullm",
    "unet",
    "utae",
    "xgb-alphaearth",
)

#: Tamanos de leyenda del barrido. El criterio principal preregistrado es K = round(C/2).
K_VALUES: tuple[int, ...] = (18, 16, 14, 12, 10, 9, 8)
K_PRINCIPAL: int = 9


def _provenance(seed: int, n_boot: int) -> dict[str, Any]:
    """Build the provenance block every artefact of this phase carries.

    Args:
        seed: Random seed.
        n_boot: Number of bootstrap resamples.

    Returns:
        Dictionary with seed, commit, versions and timestamp.
    """
    import sklearn

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
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
        "criterio_principal": f"K = {K_PRINCIPAL}, preregistrado como round(C/2) con C = 18",
        "ground_truth": "reports/paper_micai/fase1/parcel_gt_fold5.parquet",
    }


def _universe() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the sealed ground truth and the patch id of every parcel.

    Returns:
        Parcel ids, labels and the patch id used as bootstrap cluster.
    """
    gt = pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet").sort(KEY)
    keys = gt[KEY].to_list()
    patches = np.array([k.split("_")[0] for k in keys])
    return keys, gt["label"].to_numpy(), patches


def _splits(keys: list[str], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the geographic sub-folds over the sealed centroids.

    Args:
        keys: Ordered parcel ids.
        seed: Seed forwarded to the spatial split.

    Returns:
        ``(train_pos, test_pos)`` index pairs, one per block.
    """
    from ml.ensemble.stacking import StackingEnsemble

    geoms = pl.read_parquet(FASE1 / "parcel_centroids_fold5.parquet")
    helper = StackingEnsemble(base_members=("tsvit-pheno",), oof_dir=OOF_DIR, random_state=seed)
    return helper._subfolds_by_canonical_id(geoms, pl.DataFrame({KEY: keys}))


def _legend_f1(
    labels: np.ndarray, free: np.ndarray, train_pos: np.ndarray, k: int
) -> tuple[int, ...]:
    """Adapt :func:`legend_by_f1` to the ``(train_pos, k)`` signature ``frontier`` expects.

    Args:
        labels: Ground-truth labels.
        free: Unrestricted predictions.
        train_pos: Blocks used to decide.
        k: Legend size.

    Returns:
        The promised classes.
    """
    return legend_by_f1(labels, free, train_pos, k)


def _legend_soporte(labels: np.ndarray, train_pos: np.ndarray, k: int) -> tuple[int, ...]:
    """Adapt :func:`legend_by_support` to the signature ``frontier`` expects.

    Args:
        labels: Ground-truth labels.
        train_pos: Blocks used to decide.
        k: Legend size.

    Returns:
        The promised classes.
    """
    return legend_by_support(labels, train_pos, k)


def _rows(points: list[BlockPoint], predictor: str) -> list[dict[str, Any]]:
    """Flatten block points into CSV rows.

    Args:
        points: Points to flatten.
        predictor: Name of the base predictor.

    Returns:
        One dictionary per point.
    """
    return [
        {
            "predictor": predictor,
            "mecanismo": p.mechanism,
            "k_leyenda": p.k,
            "bloque": p.block,
            "n_entregadas": int(p.delivered.sum()),
            "cobertura": round(float(p.delivered.mean()), 6),
            "f1_alineado": round(p.aligned_f1, 6),
            "f1_nativo": round(p.native_f1, 6),
            "accuracy": round(p.accuracy, 6),
            "leyenda": ",".join(str(c) for c in p.legend),
        }
        for p in points
    ]


@app.callback()
def main() -> None:
    """Agrupa los subcomandos para que typer no colapse el primero."""


@app.command()
def frontera(
    seed: Annotated[int, typer.Option(help="Semilla de los bloques y del remuestreo.")] = 42,
    n_boot: Annotated[int, typer.Option(help="Remuestreos, mil como fija el preregistro.")] = 1000,
) -> None:
    """Rehace la frontera con el estimando alineado y el remuestreo pareado."""
    keys, labels, patches = _universe()
    splits = _splits(keys, seed)
    members = load_member_posteriors(OOF_DIR, ALL_MEMBERS, keys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranking = sorted(((score(labels, p)["f1_macro"], n) for n, p in members.items()), reverse=True)
    predictors = [ranking[0][1], ranking[1][1]]
    logger.info("predictores", principal=predictors[0], segundo=predictors[1])

    rows: list[dict[str, Any]] = []
    contrasts: dict[str, Any] = {}

    for predictor in predictors:
        proba = members[predictor]
        free = proba.argmax(axis=1)

        por_f1 = frontier(
            proba,
            labels,
            splits,
            K_VALUES,
            legend_fn=partial(_legend_f1, labels, free),
            mechanism="retirada por F1",
        )
        por_soporte = frontier(
            proba,
            labels,
            splits,
            K_VALUES,
            legend_fn=partial(_legend_soporte, labels),
            mechanism="retirada por soporte",
        )
        confianza = confidence_baseline(proba, labels, splits, por_f1)
        sin_mec = no_mechanism_reference(proba, labels, splits, por_f1)

        for grupo in (por_f1, por_soporte, confianza, sin_mec):
            rows.extend(_rows(grupo, predictor))

        crudos: dict[int, float] = {}
        detalle: dict[str, Any] = {}
        for k in K_VALUES:
            izq = [p for p in por_f1 if p.k == k]
            der = [p for p in confianza if p.k == k]
            sop = [p for p in por_soporte if p.k == k]
            ref = [p for p in sin_mec if p.k == k]

            par = paired_interval(labels, splits, izq, der, n_boot=n_boot, random_state=seed)
            clu = paired_interval(
                labels,
                splits,
                izq,
                der,
                n_boot=max(200, n_boot // 4),
                random_state=seed,
                clusters=patches,
            )
            crudos[k] = par["p_bootstrap"]
            detalle[f"k={k}"] = {
                "f1_retirada_f1": float(np.mean([p.aligned_f1 for p in izq])),
                "f1_retirada_soporte": float(np.mean([p.aligned_f1 for p in sop])),
                "f1_confianza": float(np.mean([p.aligned_f1 for p in der])),
                "f1_sin_mecanismo": float(np.mean([p.aligned_f1 for p in ref])),
                "cobertura_media": float(np.mean([p.delivered.mean() for p in izq])),
                "delta_vs_confianza": par["delta"],
                "ci_low": par["ci_low"],
                "ci_high": par["ci_high"],
                "excluye_cero": par["excluye_cero"],
                "p_bootstrap": par["p_bootstrap"],
                "deltas_por_bloque": par["deltas_por_bloque"],
                "ci_low_cluster_parche": clu["ci_low"],
                "ci_high_cluster_parche": clu["ci_high"],
                "excluye_cero_cluster": clu["excluye_cero"],
            }

        ajustados = holm([crudos[k] for k in K_VALUES])
        for k, adj in zip(K_VALUES, ajustados, strict=True):
            detalle[f"k={k}"]["p_holm"] = adj
            detalle[f"k={k}"]["significativo_holm"] = float(adj < 0.05)

        contrasts[predictor] = {
            "criterio_principal": detalle[f"k={K_PRINCIPAL}"],
            "familia_exploratoria": detalle,
        }

    frame = pl.DataFrame(rows)
    frame.sort(
        ["predictor", "k_leyenda", "mecanismo", "bloque"], descending=[False, True, False, False]
    ).write_csv(OUT_DIR / "frontera_por_bloque.csv")
    (
        frame.group_by(["predictor", "k_leyenda", "mecanismo"])
        .agg(
            pl.col("cobertura").mean().alias("cobertura_media"),
            pl.col("f1_alineado").mean().alias("f1_alineado_medio"),
            pl.col("f1_alineado").min().alias("f1_alineado_min"),
            pl.col("f1_alineado").max().alias("f1_alineado_max"),
            pl.col("f1_nativo").mean().alias("f1_nativo_medio"),
            pl.col("accuracy").mean().alias("accuracy_media"),
        )
        .sort(["predictor", "k_leyenda", "mecanismo"], descending=[False, True, False])
        .write_csv(OUT_DIR / "frontera_resumen.csv")
    )

    payload = {
        "predictor_principal": predictors[0],
        "predictor_segundo": predictors[1],
        "contrastes": contrasts,
        "notas": {
            "estimando": (
                "El F1 alineado promedia ambos mecanismos sobre la MISMA leyenda. El F1 nativo "
                "promedia cada uno sobre lo que promete y NO es comparable entre mecanismos: se "
                "publica solo como vista de producto."
            ),
            "entrega": (
                "Ningun mecanismo lee etiquetas para decidir a quien responde. La retirada "
                "entrega donde el argmax libre cae en la leyenda; la abstencion, por rango de "
                "confianza, igualando el numero de parcelas del otro."
            ),
            "control": (
                "La fila 'sin mecanismo' puntua al predictor intacto sobre la misma leyenda "
                "entregando todo. Si iguala o supera a un mecanismo, la calidad la trae el "
                "conjunto de clases y no el mecanismo."
            ),
            "multiplicidad": (
                "Holm-Bonferroni sobre la familia de siete valores de K, dentro de cada "
                "predictor. El criterio principal preregistrado es K = 9 y es uno solo."
            ),
        },
        "procedencia": _provenance(seed, n_boot),
    }
    (OUT_DIR / "frontera_contrastes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("frontera_done", out=str(OUT_DIR))


if __name__ == "__main__":
    app()
