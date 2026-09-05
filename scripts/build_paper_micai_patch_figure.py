"""Figura cualitativa: parcelas reales del fold retenido, con su etiqueta agronomica.

Es la figura que ancla el articulo en el dato. Muestra tres parches del fold 5 de PASTIS en
color natural, en una fecha de verano, junto a su anotacion por parcela coloreada por clase. La
leyenda usa la misma paleta que el resto de figuras y separa en siena las clases raras, que son
las que el recorte de leyenda retira primero.

Uso:
    poetry run python scripts/build_paper_micai_patch_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import structlog

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PASTIS = REPO_ROOT / "data" / "PASTIS-R"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "figuras"

#: Indices de las bandas rojo, verde y azul dentro del eje espectral de PASTIS-R.
RGB_BANDS: tuple[int, int, int] = (2, 1, 0)

#: Fold retenido sobre el que se evalua todo el articulo.
FOLD: int = 5

#: Etiquetas que no nombran un cultivo: el fondo y el pixel sin anotacion.
_NOT_A_CROP: frozenset[int] = frozenset({0, 19})

#: Parches elegidos por densidad de parcelas y variedad de clases, no por su aspecto.
PATCHES: tuple[int, ...] = (20367, 40058, 40002)

#: Nombres de las diecinueve etiquetas densas, con el fondo en la posicion cero.
CLASS_NAMES: tuple[str, ...] = (
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits, vegetables, flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
)

#: Diecinueve tonos distinguibles entre si: verdes y ocres para las clases abundantes, calidos
#: para las raras, que son las que el recorte de leyenda retira primero. Cada clase lleva su color
#: propio porque la leyenda de esta figura hay que poder leerla entrada por entrada.
PALETTE: tuple[str, ...] = (
    "#EDEDE7",
    "#3F5F2E",
    "#7C8F5E",
    "#A9A05A",
    "#B9C48E",
    "#D8C55E",
    "#5E8C6A",
    "#E0B84C",
    "#4A7C8C",
    "#C4703A",
    "#94A86F",
    "#B4522F",
    "#8C3F5C",
    "#D98E6A",
    "#6E8878",
    "#A8B08A",
    "#8A4B2A",
    "#E3856B",
    "#96674A",
    "#FFFFFF",
)

STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "figure.dpi": 220,
    "svg.hashsalt": "agrosat-micai",
}

#: Metadatos anulados para que la figura sea reproducible byte a byte.
DETERMINISTIC_METADATA: dict[str, dict[str, None]] = {
    "svg": {"Date": None},
    "png": {"Software": None},
    "pdf": {"CreationDate": None},
}


def _summer_index(dates: dict[str, int]) -> int:
    """Pick the acquisition closest to the first of July.

    Args:
        dates: Mapping from time index to ``YYYYMMDD``.

    Returns:
        The temporal index of the chosen acquisition.
    """
    target = 701
    best, best_gap = 0, 10_000
    for key, value in dates.items():
        gap = abs(int(str(value)[4:]) - target)
        if gap < best_gap:
            best, best_gap = int(key), gap
    return best


def _rgb(patch: int, index: int) -> np.ndarray:
    """Build a contrast-stretched natural-color composite of one acquisition.

    Args:
        patch: Patch identifier.
        index: Temporal index to read.

    Returns:
        An ``(H, W, 3)`` array in ``[0, 1]``.
    """
    cube = np.load(PASTIS / "DATA_S2" / f"S2_{patch}.npy", mmap_mode="r")
    frame = np.asarray(cube[index, list(RGB_BANDS)], dtype=np.float32).transpose(1, 2, 0)
    low, high = np.percentile(frame, [2, 98])
    return np.clip((frame - low) / max(high - low, 1e-6), 0.0, 1.0)


def main() -> None:
    """Draw the qualitative patch figure."""
    meta = json.loads((PASTIS / "metadata.geojson").read_text())
    by_id = {f["properties"]["ID_PATCH"]: f["properties"] for f in meta["features"]}
    cmap = ListedColormap(PALETTE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    present: set[int] = set()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(PATCHES), figsize=(7.2, 5.1))
        for column, patch in enumerate(PATCHES):
            props = by_id[patch]
            index = _summer_index(props["dates-S2"])
            target = np.load(PASTIS / "ANNOTATIONS" / f"TARGET_{patch}.npy")[0]
            present.update(int(c) for c in np.unique(target) if c not in _NOT_A_CROP)

            axes[0, column].imshow(_rgb(patch, index))
            axes[0, column].set_title(
                f"parche {patch} · fold {props['Fold']}\n{props['N_Parcel']} parcelas", pad=4
            )
            axes[1, column].imshow(
                target, cmap=cmap, vmin=0, vmax=len(PALETTE) - 1, interpolation="nearest"
            )
            for row in (0, 1):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_ylabel("color natural", fontsize=8)
        axes[1, 0].set_ylabel("clase agronomica", fontsize=8)

        handles = [
            Patch(facecolor=PALETTE[c], edgecolor="#131A17", linewidth=0.3, label=CLASS_NAMES[c])
            for c in sorted(present)
        ]
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            fontsize=6.6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        for ext in ("svg", "png", "pdf"):
            out = OUT_DIR / f"parcelas.{ext}"
            fig.savefig(out, bbox_inches="tight", metadata=DETERMINISTIC_METADATA[ext])
            logger.info("figura_guardada", path=str(out.relative_to(REPO_ROOT)))
        plt.close(fig)


if __name__ == "__main__":
    main()
