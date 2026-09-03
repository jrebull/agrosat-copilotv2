"""Figura de la frontera calidad-cobertura de la fase 3, vectorial y legible en gris.

Dibuja los tres mecanismos y el control sin mecanismo sobre el mismo eje, con la banda
que da el remuestreo pareado. La figura existe para que se vea de un golpe lo que la
auditoria costo entender: si el control sin mecanismo va por encima de un mecanismo, la
calidad la trae el conjunto de clases y no el mecanismo.

Uso:
    poetry run python scripts/build_paper_micai_fase3_figure.py
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
FASE3 = REPO_ROOT / "reports" / "paper_micai" / "fase3"

#: Estilo sobrio de articulo: serif, sin rejilla de color, distinguible en blanco y negro.
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
    # Sal fija para los identificadores del SVG: sin ella matplotlib los aleatoriza en cada
    # corrida y la figura deja de ser reproducible byte a byte.
    "svg.hashsalt": "agrosat-micai",
}

#: Matplotlib incrusta la fecha de generacion en SVG y PDF, lo que rompe el sello de custodia
#: en cada regeneracion por un motivo que no tiene que ver con los datos. Anularla deja la figura
#: reproducible byte a byte.
DETERMINISTIC_METADATA: dict[str, dict[str, None]] = {
    "svg": {"Date": None},
    "png": {"Software": None},
    "pdf": {"CreationDate": None},
}

#: Cada mecanismo con su marcador y su trazo, no solo con su color.
SERIES = {
    "retirada por F1": {"marker": "o", "ls": "-", "color": "#B4522F"},
    "retirada por soporte": {"marker": "s", "ls": "--", "color": "#A87A1E"},
    "rechazo por confianza": {"marker": "^", "ls": "-.", "color": "#174C4A"},
    "sin mecanismo": {"marker": "x", "ls": ":", "color": "#6E7A72"},
}


def main() -> None:
    """Draw the frontier for both predictors into one two-panel figure."""
    resumen = pl.read_csv(FASE3 / "frontera_resumen.csv")
    contrastes = json.loads((FASE3 / "frontera_contrastes.json").read_text(encoding="utf-8"))
    predictores = [contrastes["predictor_principal"], contrastes["predictor_segundo"]]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
        for ax, predictor in zip(axes, predictores, strict=True):
            sub = resumen.filter(pl.col("predictor") == predictor)
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
            familia = contrastes["contrastes"][predictor]["familia_exploratoria"]
            ks = sorted(int(k.split("=")[1]) for k in familia)
            principal = int(
                contrastes["procedencia"]["criterio_principal"].split("=")[1].split(",")[0]
            )
            ax.axvline(principal, color="#131A17", linewidth=0.5, linestyle=(0, (1, 3)))
            ax.annotate(
                "criterio principal",
                xy=(principal, ax.get_ylim()[0]),
                xytext=(principal - 0.3, ax.get_ylim()[0] + 0.02),
                fontsize=7,
                rotation=90,
                color="#131A17",
                va="bottom",
                ha="right",
            )
            ax.set_xticks(ks)
            ax.invert_xaxis()
            ax.set_xlabel("clases prometidas")
            ax.set_title(predictor, fontsize=9, pad=6)
        axes[0].set_ylabel("F1-macro sobre la leyenda compartida")
        axes[0].legend(loc="lower left", fontsize=7.5)
        fig.tight_layout()

        for ext in ("svg", "png", "pdf"):
            out = FASE3 / f"frontera.{ext}"
            fig.savefig(out, bbox_inches="tight", metadata=DETERMINISTIC_METADATA[ext])
            logger.info("figura_guardada", path=str(out.relative_to(REPO_ROOT)))
        plt.close(fig)


if __name__ == "__main__":
    main()
