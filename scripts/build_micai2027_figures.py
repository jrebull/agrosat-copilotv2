"""Genera las figuras del manuscrito nuevo bajo el contrato de US-143.

Cuatro reglas, y las cuatro vienen de defectos reales del manuscrito retirado: ancho final, ocho
puntos impresos, ningun significado codificado solo con color, y bytes estables. Viven en
`ml.report.figuras_micai`, no aqui, para que anadir una figura no sea volver a acordarse de ellas.

**Ademas se niega a dibujar sobre insumos marcados OBSOLETO.** Una figura bien tipografiada de
datos invalidos es peor que una fea: parece revisada. Por eso las figuras de la frontera —que
dependen de fase 3 y fase 4— todavia no se generan aqui: sus insumos se regeneran primero.

Uso:
    poetry run python scripts/build_micai2027_figures.py
    poetry run python scripts/build_micai2027_figures.py --idioma es --salida <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import polars as pl
import structlog

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.report.figuras_micai import (
    LNCS_TEXT_WIDTH_INCHES,
    apply_manuscript_style,
    require_current_inputs,
    save_figure,
)

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FASE1 = REPO_ROOT / "reports" / "paper_micai" / "fase1"
FASE4 = REPO_ROOT / "reports" / "paper_micai" / "fase4"
OUT_DIR = REPO_ROOT / "reports" / "micai2027" / "figuras"

#: Umbral por debajo del cual una clase se considera de la cola.
RARA: int = 300

#: Insumos de la figura de soporte, para comprobar su estado antes de dibujar.
INSUMOS_SOPORTE: tuple[str, ...] = (
    "reports/paper_micai/fase1/parcel_gt_fold5.parquet",
    "reports/paper_micai/fase4/breizhcrops_soporte.csv",
)

#: Rotulos de esta figura, por idioma. El banco es **PASTIS**, no PASTIS-R: el articulo usa el
#: optico. La correccion se hizo en el manuscrito y en el ledger y NO llego a las figuras, que es
#: lo primero que mira un revisor.
TEXTOS: dict[str, dict[str, str]] = {
    "es": {
        "eje_y": "parcelas (escala logarítmica)",
        "primario": "PASTIS · 18 clases · 16 640 parcelas",
        "replica": "BreizhCrops · 9 clases · 60 000 parcelas",
        "umbral": f"{RARA} parcelas",
        "cola": "cola (< 300)",
        "resto": "resto",
    },
    "en": {
        "eje_y": "parcels (log scale)",
        "primario": "PASTIS · 18 classes · 16,640 parcels",
        "replica": "BreizhCrops · 9 classes · 60,000 parcels",
        "umbral": f"{RARA} parcels",
        "cola": "tail (< 300)",
        "resto": "rest",
    },
}

#: La cola se distingue por color Y por trama: en fotocopia el color desaparece y la trama no.
COLOR_COLA, COLOR_RESTO = "#B4522F", "#174C4A"
TRAMA_COLA = "///"


def figura_soporte(idioma: str, salida: Path) -> list[Path]:
    """Draw the class-support distribution of both benchmarks.

    Args:
        idioma: ``"es"`` or ``"en"``.
        salida: Output directory.

    Returns:
        The written files.
    """
    require_current_inputs(INSUMOS_SOPORTE)
    t = TEXTOS[idioma]
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

    apply_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(LNCS_TEXT_WIDTH_INCHES, 2.6))
    paneles = (
        (
            axes[0],
            primario["len"].to_list(),
            [f"c{c}" for c in primario["label"].to_list()],
            t["primario"],
        ),
        (
            axes[1],
            replica["n_parcelas"].to_list(),
            [n.split()[0] for n in replica["class_name"].to_list()],
            t["replica"],
        ),
    )
    for ax, valores, etiquetas, titulo in paneles:
        for i, v in enumerate(valores):
            cola = v < RARA
            ax.bar(
                i,
                v,
                width=0.74,
                color=COLOR_COLA if cola else COLOR_RESTO,
                hatch=TRAMA_COLA if cola else None,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(valores)))
        ax.set_xticklabels(etiquetas, rotation=90)
        ax.set_title(titulo, pad=5)
        ax.axhline(RARA, color="#131A17", linewidth=0.6, linestyle=(0, (1, 3)))
    axes[0].set_ylabel(t["eje_y"])
    axes[0].legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=COLOR_COLA,
                hatch=TRAMA_COLA,
                edgecolor="white",
                label=t["cola"],
            ),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_RESTO, edgecolor="white", label=t["resto"]),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout()
    escritos: list[Path] = save_figure(fig, salida / f"soporte-{idioma}")
    plt.close(fig)
    return escritos


def main() -> int:
    """Build every figure whose inputs are current, in both languages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idioma", choices=("es", "en", "ambos"), default="ambos")
    parser.add_argument("--salida", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    idiomas = ("es", "en") if args.idioma == "ambos" else (args.idioma,)
    for idioma in idiomas:
        for ruta in figura_soporte(idioma, args.salida):
            logger.info("figura", idioma=idioma, path=str(ruta.resolve().relative_to(REPO_ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
