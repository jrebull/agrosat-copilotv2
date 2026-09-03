"""Tres figuras que el articulo necesitaba y no tenia: soporte, leyendas y frontera en cobertura.

- `soporte`: el reparto de clases de los dos bancos en escala logaritmica, que es la premisa del
  articulo y hasta ahora solo estaba en prosa.
- `leyendas`: que clases sobreviven a cada valor de K bajo cada criterio de retirada. Es la figura
  que explica de un vistazo por que la retirada por soporte es peor: saca colza y conserva las dos
  praderas.
- `cobertura`: la misma frontera dibujada contra la cobertura entregada y no contra K, que es como
  la literatura de clasificacion selectiva espera verla.

Uso:
    poetry run python scripts/build_paper_micai_extra_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import structlog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
FASE3 = REPO_ROOT / "reports" / "paper_micai" / "fase3"
FASE4 = REPO_ROOT / "reports" / "paper_micai" / "fase4"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "figuras"

STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.dpi": 220,
    "svg.hashsalt": "agrosat-micai",
}

#: Metadatos anulados para que las figuras sean reproducibles byte a byte.
DETERMINISTIC_METADATA: dict[str, dict[str, None]] = {
    "svg": {"Date": None},
    "png": {"Software": None},
    "pdf": {"CreationDate": None},
}

SERIES: dict[str, dict[str, Any]] = {
    "retirada por F1": {"marker": "o", "ls": "-", "color": "#B4522F"},
    "retirada por soporte": {"marker": "s", "ls": "--", "color": "#A87A1E"},
    "rechazo por confianza": {"marker": "^", "ls": "-.", "color": "#174C4A"},
    "sin mecanismo": {"marker": "x", "ls": ":", "color": "#6E7A72"},
}

#: Umbral por debajo del cual una clase se dibuja en siena: es la cola que el recorte ataca.
RARA: int = 300


def _save(fig: plt.Figure, name: str) -> None:
    """Write one figure in the three formats the project keeps.

    Args:
        fig: Figure to write.
        name: Base name without extension.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png", "pdf"):
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", metadata=DETERMINISTIC_METADATA[ext])
        logger.info("figura_guardada", path=str(out.relative_to(REPO_ROOT)))
    plt.close(fig)


def figura_soporte() -> None:
    """Draw the class-support distribution of both benchmarks on a log scale."""
    primario = (
        pl.read_parquet(FASE1 / "parcel_gt_fold5.parquet")
        .group_by("label")
        .len()
        .sort("len", descending=True)
    )
    replica = (
        pl.read_csv(FASE4 / "breizhcrops_soporte.csv")
        .group_by("class_id", "class_name")
        .agg(pl.col("n_parcelas").sum())
        .sort("n_parcelas", descending=True)
    )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
        for ax, valores, etiquetas, titulo in (
            (
                axes[0],
                primario["len"].to_list(),
                [f"c{c}" for c in primario["label"].to_list()],
                "PASTIS-R · 18 clases · 16 640 parcelas",
            ),
            (
                axes[1],
                replica["n_parcelas"].to_list(),
                [n.replace(" ", "\n") for n in replica["class_name"].to_list()],
                "BreizhCrops · 9 clases · 60 000 parcelas",
            ),
        ):
            colores = ["#B4522F" if v < RARA else "#174C4A" for v in valores]
            ax.bar(range(len(valores)), valores, color=colores, width=0.74)
            ax.set_yscale("log")
            ax.set_xticks(range(len(valores)))
            ax.set_xticklabels(etiquetas, rotation=90, fontsize=6.2)
            ax.set_title(titulo, fontsize=8.5, pad=6)
            ax.axhline(RARA, color="#131A17", linewidth=0.5, linestyle=(0, (1, 3)))
            ax.annotate(
                f"{RARA} parcelas",
                xy=(len(valores) - 0.4, RARA),
                fontsize=6.2,
                va="bottom",
                ha="right",
                color="#131A17",
            )
            for i, v in enumerate(valores):
                if v < 20:
                    ax.annotate(
                        str(v), xy=(i, v), fontsize=6, ha="center", va="bottom", color="#B4522F"
                    )
        axes[0].set_ylabel("parcelas (escala logaritmica)")
        fig.tight_layout()
        _save(fig, "soporte")


def figura_leyendas() -> None:
    """Draw which classes each retirement criterion keeps as the legend shrinks."""
    tabla = pl.read_csv(FASE4 / "replica_por_bloque.csv").filter(
        (pl.col("universo") == "todas las clases") & (pl.col("bloque") == "frh01")
    )
    nombres = {
        n["class_id"]: n["class_name"]
        for n in json.loads((FASE4 / "replica_contrastes.json").read_text())["clases"]
    }
    soporte = {
        r["class_id"]: r["n_parcelas"]
        for r in pl.read_csv(FASE4 / "breizhcrops_soporte.csv")
        .group_by("class_id")
        .agg(pl.col("n_parcelas").sum())
        .to_dicts()
    }
    orden = sorted(soporte, key=lambda c: -soporte[c])
    ks = sorted(tabla["k_leyenda"].unique().to_list(), reverse=True)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        for ax, mecanismo, color in (
            (axes[0], "retirada por F1", "#B4522F"),
            (axes[1], "retirada por soporte", "#A87A1E"),
        ):
            sub = tabla.filter(pl.col("mecanismo") == mecanismo)
            rejilla = np.zeros((len(orden), len(ks)))
            for col, k in enumerate(ks):
                fila = sub.filter(pl.col("k_leyenda") == k).row(0, named=True)
                legend = {int(c) for c in fila["leyenda"].split(",")}
                for row, clase in enumerate(orden):
                    rejilla[row, col] = 1.0 if clase in legend else 0.0
            ax.imshow(
                rejilla,
                cmap=matplotlib.colors.ListedColormap(["#EDEDE7", color]),
                aspect="auto",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels(ks)
            ax.set_xlabel("clases prometidas")
            ax.set_title(mecanismo, fontsize=8.5, pad=6)
            ax.set_xticks(np.arange(-0.5, len(ks), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(orden), 1), minor=True)
            ax.grid(which="minor", color="#FFFFFF", linewidth=1.1)
            ax.tick_params(which="minor", length=0)
        axes[0].set_yticks(range(len(orden)))
        axes[0].set_yticklabels([f"{nombres[c]} ({soporte[c]})" for c in orden], fontsize=6.8)
        fig.tight_layout()
        _save(fig, "leyendas")


def figura_cobertura() -> None:
    """Draw the frontier against delivered coverage instead of against legend size."""
    replica = pl.read_csv(FASE4 / "replica_por_bloque.csv").filter(
        pl.col("universo") == "todas las clases"
    )
    primario = pl.read_csv(FASE3 / "frontera_por_bloque.csv")
    principal = json.loads((FASE3 / "frontera_contrastes.json").read_text())["predictor_principal"]
    primario = primario.filter(pl.col("predictor") == principal)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
        for ax, tabla, titulo in (
            (axes[0], primario, f"PASTIS-R · {principal}"),
            (axes[1], replica, "BreizhCrops · dejando una region fuera"),
        ):
            resumen = (
                tabla.group_by(["mecanismo", "k_leyenda"])
                .agg(
                    pl.col("cobertura").mean().alias("cob"),
                    pl.col("f1_alineado").mean().alias("f1"),
                )
                .sort("cob")
            )
            for nombre, estilo in SERIES.items():
                serie = resumen.filter(pl.col("mecanismo") == nombre)
                if serie.height == 0:
                    continue
                # El control sin mecanismo entrega siempre todo, asi que vive en la vertical de
                # cobertura uno; unir sus puntos con una linea sugeriria una trayectoria que no
                # existe, de modo que se dibuja solo con marcadores.
                estilo = dict(estilo)
                if nombre == "sin mecanismo":
                    estilo["ls"] = "none"
                ax.plot(
                    serie["cob"].to_list(),
                    serie["f1"].to_list(),
                    label=nombre,
                    linewidth=1.2,
                    markersize=4.2,
                    **estilo,
                )
            ax.set_xlabel("cobertura entregada")
            ax.set_title(titulo, fontsize=8.5, pad=6)
        axes[0].set_ylabel("F1-macro sobre la leyenda compartida")
        axes[1].legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        _save(fig, "cobertura")


def main() -> None:
    """Build the three figures."""
    figura_soporte()
    figura_leyendas()
    figura_cobertura()


if __name__ == "__main__":
    main()
