"""Figura de la replica de la fase 4: la misma frontera sobre BreizhCrops.

Dibuja los dos universos declarados en la enmienda 1 lado a lado, con los mismos
mecanismos y el mismo control que la figura de la fase 3, para que la comparacion entre
los dos conjuntos se haga a ojo y sin trucos de escala.

Uso:
    poetry run python scripts/build_paper_micai_fase4_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import polars as pl
import structlog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FASE4 = REPO_ROOT / "reports" / "paper_micai" / "fase4"

#: Mismo estilo sobrio que la fase 3, para que las dos figuras se lean como una serie.
STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.dpi": 200,
}

#: Cada mecanismo con su marcador y su trazo, no solo con su color.
SERIES = {
    "retirada por F1": {"marker": "o", "ls": "-", "color": "#B4522F"},
    "retirada por soporte": {"marker": "s", "ls": "--", "color": "#A87A1E"},
    "rechazo por confianza": {"marker": "^", "ls": "-.", "color": "#174C4A"},
    "sin mecanismo": {"marker": "x", "ls": ":", "color": "#6E7A72"},
}


def main() -> None:
    """Draw the replicated frontier for both declared universes."""
    tabla = pl.read_csv(FASE4 / "replica_por_bloque.csv")
    contrastes = json.loads((FASE4 / "replica_contrastes.json").read_text(encoding="utf-8"))
    universos = list(contrastes["contrastes"].keys())

    resumen = (
        tabla.group_by(["universo", "k_leyenda", "mecanismo"])
        .agg(pl.col("f1_alineado").mean().alias("f1_alineado_medio"))
        .sort(["universo", "k_leyenda", "mecanismo"])
    )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(universos), figsize=(7.2, 3.1), sharey=True)
        for ax, universo in zip(axes, universos, strict=True):
            sub = resumen.filter(pl.col("universo") == universo)
            for nombre, estilo in SERIES.items():
                serie = sub.filter(pl.col("mecanismo") == nombre).sort("k_leyenda")
                if serie.height == 0:
                    continue
                ax.plot(
                    serie["k_leyenda"].to_list(),
                    serie["f1_alineado_medio"].to_list(),
                    label=nombre,
                    linewidth=1.2,
                    markersize=3.6,
                    **estilo,
                )
            principal = contrastes["contrastes"][universo]["k_principal"]
            ax.axvline(principal, color="#131A17", linewidth=0.5, linestyle=(0, (1, 3)))
            ax.set_xticks(sorted(sub["k_leyenda"].unique().to_list()))
            ax.invert_xaxis()
            ax.set_xlabel("clases prometidas")
            ax.set_title(universo, fontsize=9, pad=6)
        axes[0].set_ylabel("F1-macro sobre la leyenda compartida")
        axes[0].legend(loc="lower left", fontsize=7.5)
        fig.tight_layout()

        for ext in ("svg", "png"):
            out = FASE4 / f"replica.{ext}"
            fig.savefig(out, bbox_inches="tight")
            logger.info("figura_guardada", path=str(out.relative_to(REPO_ROOT)))
        plt.close(fig)


if __name__ == "__main__":
    main()
