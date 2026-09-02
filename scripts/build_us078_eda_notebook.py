"""Builder of the US-078 EDA notebook for the Italy 2018 PASTIS homologue.

Generates ``notebooks/transfer/us078_italia_pastis_eda.ipynb`` programmatically
and reproducibly (same pattern as the other ``scripts/build_*_notebook.py``
builders). The notebook is step 6 of the US-078 plan: it validates the dataset
that :mod:`ml.data.eurocrops_pastis_builder` produced, with NO placeholders and
NO fabricated numbers -- every figure comes from reading the real ``.npy`` /
``metadata.parquet`` under ``data/pastis_italia_2018``.

What the notebook shows:

1. Cover + framing (why a PASTIS-homologous Italian dataset for the transfer).
2. The HCAT -> contiguous-class mapping (AC4) read from ``class_mapping.json``.
3. Per-pixel and per-patch class distribution (the long-tail and the ``other``
   bucket made explicit).
4. RGB + dense-mask examples for a few patches (the PASTIS ``S2`` / ``TARGET``
   look), proving the image<->mask alignment.
5. The texture check: per-patch NDVI spatial std vs the PASTIS reference (~0.2),
   distinguishing real patch texture from the pixel-point signal (~0.05).
6. Coverage per patch (objective > 70%) and the temporal-series summary
   (n dates, residual cloud).

Visible prose (markdown, captions, prints) is Spanish with accents; code,
identifiers, comments and docstrings stay in English ASCII (project convention).
No emojis.

Usage::

    poetry run python scripts/build_us078_eda_notebook.py \\
        --out notebooks/transfer/us078_italia_pastis_eda.ipynb \\
        --data-dir data/pastis_italia_2018

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us078_italia_pastis_eda.ipynb")
_DEFAULT_DATA = Path("data/pastis_italia_2018")


def _build_cells(data_dir: str) -> list:
    """Build the markdown + code cells of the EDA notebook.

    Args:
        data_dir: Repo-relative path to the homologue dataset, injected into the
            parameters cell so the notebook reads the real artefacts.

    Returns:
        The ordered list of ``nbformat`` cells.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    cells.append(
        md(
            "# US-078 - EDA del homologo PASTIS de Italia 2018\n\n"
            "### Equipo 17 - AgroSatCopilot - Transfer learning mediterraneo (EPIC 12)\n\n"
            "---\n\n"
            "Este cuaderno **valida** el dataset que el constructor "
            "`ml.data.eurocrops_pastis_builder` genero a partir de los poligonos de "
            "EuroCrops Italia 2018. El dataset replica el formato exacto de PASTIS-R "
            "(`S2_<id>.npy` de forma `(T, 10, 128, 128)` mas una mascara densa "
            "`TARGET_<id>.npy` de `(128, 128)`), para que los modelos densos del "
            "campeon (TSViT-pheno, U-TAE) operen sobre Italia en su formato nativo y el "
            "transfer Francia->Italia sea limpio.\n\n"
            "Todas las cifras de abajo se leen de los artefactos reales bajo "
            f"`{data_dir}`: no hay numeros inventados ni celdas con salida fabricada. "
            "Cuando una parcela o fecha no se pudo descargar, el constructor la "
            "descarto y la conto -- aqui solo se analiza lo que realmente existe.\n\n"
            "> Alcance: piloto de 20 patches reales (descarga genuina de Sentinel Hub "
            "sobre las zonas mas densas de la Toscana). El escalado al dataset completo "
            "queda condicionado al GATE revisado por el equipo."
        )
    )

    cells.append(
        code(
            "# Parametros (papermill). data_dir apunta al dataset homologo generado.\n"
            f'data_dir = "{data_dir}"\n'
            "n_examples = 3  # patches a visualizar (RGB + mascara)\n"
        )
    )
    cells[-1].metadata = {"tags": ["parameters"]}

    cells.append(
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import ListedColormap\n"
            "\n"
            "ROOT = Path(data_dir)\n"
            "S2_DIR = ROOT / 'DATA_S2'\n"
            "ANN_DIR = ROOT / 'ANNOTATIONS'\n"
            "meta = pl.read_parquet(ROOT / 'metadata.parquet')\n"
            "class_table = pl.read_parquet(ROOT / 'class_table.parquet')\n"
            "mapping = json.loads((ROOT / 'class_mapping.json').read_text(encoding='utf-8'))\n"
            "id_to_name = {0: 'background', **{r['class_id']: r['hcat4_name'] for r in mapping['classes']}}\n"
            "print(f'Patches en el dataset: {meta.height}')\n"
            "print(f'Clases en la taxonomia (sin background): {class_table.height}')\n"
        )
    )

    # ------------------------------------------------------- mapping (AC4) ---
    cells.append(
        md(
            "## 1. Mapeo `original_code` -> HCAT -> id contiguo (AC4)\n\n"
            "El constructor mapea el codigo original de cada parcela italiana a su "
            "nombre HCAT4 (cobertura 100% para Italia, `nuts` que empieza con `it`) y "
            "luego a un id de clase **contiguo**: `0` reservado para fondo/no-cultivo y "
            "`[1, K]` para los cultivos. Las clases con menos parcelas que el umbral "
            "`min_support` se agrupan en una clase **`other` explicita**, para no "
            "inflar el espacio de etiquetas con la cola larga. La tabla siguiente es "
            "ese mapeo, ordenado por soporte de parcelas."
        )
    )
    cells.append(
        code(
            "with pl.Config(tbl_rows=50):\n"
            "    display(class_table.with_columns(\n"
            "        nombre=pl.col('hcat4_name')\n"
            "    ).select(['class_id', 'nombre', 'n_parcels']))\n"
            "print(f\"id de fondo: {mapping['background_id']}; \"\n"
            "      f\"clase agregada de cola larga: '{mapping['other_class_name']}'\")\n"
        )
    )

    # --------------------------------------------- class distribution ---
    cells.append(
        md(
            "## 2. Distribucion de clases por pixel\n\n"
            "La distribucion **por pixel** (densa) es la que ven los modelos de "
            "segmentacion. La calculamos recorriendo las mascaras `TARGET_<id>.npy` de "
            "todos los patches y contando pixeles por clase. El fondo (`background`) "
            "domina en pixeles porque incluye caminos, agua y parcelas no cultivadas; "
            "lo reportamos aparte para no sesgar la lectura de los cultivos."
        )
    )
    cells.append(
        code(
            "patch_ids = sorted(meta['patch_id'].to_list())\n"
            "pixel_counts = {}\n"
            "for pid in patch_ids:\n"
            "    m = np.load(ANN_DIR / f'TARGET_{pid}.npy')\n"
            "    ids, cnts = np.unique(m, return_counts=True)\n"
            "    for i, c in zip(ids.tolist(), cnts.tolist()):\n"
            "        pixel_counts[i] = pixel_counts.get(i, 0) + c\n"
            "dist = (pl.DataFrame({\n"
            "    'class_id': list(pixel_counts.keys()),\n"
            "    'pixels': list(pixel_counts.values()),\n"
            "}).with_columns(\n"
            "    nombre=pl.col('class_id').replace_strict(id_to_name, default='?'),\n"
            ").sort('pixels', descending=True))\n"
            "total_px = int(dist['pixels'].sum())\n"
            "fg = dist.filter(pl.col('class_id') != 0)\n"
            "print(f'Pixeles totales: {total_px:,}')\n"
            "print(f\"Fondo: {dist.filter(pl.col('class_id')==0)['pixels'][0]/total_px:.1%} de los pixeles\")\n"
            "with pl.Config(tbl_rows=30):\n"
            "    display(fg.with_columns(pct=(pl.col('pixels')/total_px*100).round(2)).head(20))\n"
        )
    )
    cells.append(
        code(
            "top = fg.head(15)\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ax.barh(top['nombre'].to_list()[::-1], top['pixels'].to_list()[::-1], color='#2e7d32')\n"
            "ax.set_xlabel('Pixeles (cultivo)')\n"
            "ax.set_title('Distribucion de clases por pixel (top 15, sin fondo)')\n"
            "ax.grid(axis='x', alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    # ------------------------------------------------- RGB + mask examples ---
    cells.append(
        md(
            "## 3. Ejemplos: RGB compuesto + mascara densa\n\n"
            "Cada patch es una serie temporal `(T, 10, 128, 128)`. Para visualizar "
            "componemos un RGB con las bandas B04/B03/B02 (en orden PASTIS) de una fecha "
            "central de la serie y lo mostramos junto a su mascara densa `TARGET`. La "
            "mascara se rasterizo con **el mismo grid** que la imagen (transform del "
            "tile de Sentinel Hub), de modo que la alineacion imagen<->mascara es "
            "pixel-perfecta: los bordes de las parcelas coinciden con los del RGB."
        )
    )
    cells.append(
        code(
            "def rgb_from_stack(stack):\n"
            "    # Bandas PASTIS: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 -> RGB = B04,B03,B02.\n"
            "    t_mid = stack.shape[0] // 2\n"
            "    rgb = stack[t_mid, [2, 1, 0]].astype('float32')  # (3, H, W) DN\n"
            "    rgb = np.transpose(rgb, (1, 2, 0))\n"
            "    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)\n"
            "    return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)\n"
            "\n"
            "rng = np.random.default_rng(17)\n"
            "sel = sorted(rng.choice(patch_ids, size=min(n_examples, len(patch_ids)), replace=False).tolist())\n"
            "n_cls = class_table['class_id'].max() + 1\n"
            "cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(n_cls, 2))))\n"
            "fig, axes = plt.subplots(len(sel), 2, figsize=(8, 4 * len(sel)))\n"
            "axes = np.atleast_2d(axes)\n"
            "for row, pid in enumerate(sel):\n"
            "    stack = np.load(S2_DIR / f'S2_{pid}.npy')\n"
            "    mask = np.load(ANN_DIR / f'TARGET_{pid}.npy')\n"
            "    axes[row, 0].imshow(rgb_from_stack(stack))\n"
            "    axes[row, 0].set_title(f'Patch {pid} - RGB (B04/B03/B02)')\n"
            "    axes[row, 0].axis('off')\n"
            "    axes[row, 1].imshow(mask, cmap=cmap, vmin=0, vmax=n_cls - 1, interpolation='nearest')\n"
            "    cov = float((mask != 0).mean())\n"
            "    axes[row, 1].set_title(f'Patch {pid} - mascara densa (cobertura {cov:.0%})')\n"
            "    axes[row, 1].axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    # ----------------------------------------------------- NDVI texture ---
    cells.append(
        md(
            "## 4. Textura: std espacial de NDVI vs PASTIS (~0.2)\n\n"
            "El argumento central del homologo es que cada patch tiene **textura "
            "espacial real** (un mosaico de parcelas), no la senal de un punto. Lo "
            "medimos con la desviacion estandar espacial del NDVI por fecha, promediada "
            "sobre la serie. La referencia PASTIS-R esta en ~0.18-0.20; un pixel-punto "
            "daria ~0.05. El NDVI es invariante a la escala de reflectancia, asi que la "
            "comparacion con PASTIS es directa pese al escalado a DN."
        )
    )
    cells.append(
        code(
            "PASTIS_NDVI_STD = 0.181  # referencia verificada sobre PASTIS-R\n"
            "ndvi_std = meta.select(['patch_id', 'ndvi_std']).sort('patch_id')\n"
            "mean_std = float(ndvi_std['ndvi_std'].mean())\n"
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "ax.bar(ndvi_std['patch_id'].cast(pl.Utf8).to_list(), ndvi_std['ndvi_std'].to_list(),\n"
            "       color='#1565c0', label='homologo Italia')\n"
            "ax.axhline(PASTIS_NDVI_STD, color='#c62828', linestyle='--', label=f'PASTIS ref {PASTIS_NDVI_STD}')\n"
            "ax.axhline(0.05, color='grey', linestyle=':', label='pixel-punto ~0.05')\n"
            "ax.set_xlabel('patch_id'); ax.set_ylabel('std espacial NDVI (media de la serie)')\n"
            "ax.set_title(f'Textura por patch - media homologo {mean_std:.3f} vs PASTIS {PASTIS_NDVI_STD}')\n"
            "ax.legend(); ax.grid(axis='y', alpha=0.3)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'std NDVI medio del homologo: {mean_std:.3f} (PASTIS ~{PASTIS_NDVI_STD}; pixel-punto ~0.05)')\n"
        )
    )

    # ------------------------------------------------------ coverage + dates ---
    cells.append(
        md(
            "## 5. Cobertura por patch y resumen temporal\n\n"
            "Verificamos los dos criterios operativos: la **cobertura** (fraccion de "
            "pixeles con clase de cultivo, objetivo > 70%) y la **densidad temporal** "
            "(numero de fechas con poca nube por patch). El nube residual es la fraccion "
            "de pixeles que la mascara SCL puso a cero por nube/sombra; un valor bajo "
            "confirma que la serie es fisicamente coherente."
        )
    )
    cells.append(
        code(
            "cov = meta.select(['patch_id', 'pct_cubierto', 'n_fechas', 'residual_cloud',\n"
            "                   'n_parcelas', 'fold_espacial']).sort('patch_id')\n"
            "with pl.Config(tbl_rows=30):\n"
            "    display(cov.with_columns(\n"
            "        pct_cubierto=(pl.col('pct_cubierto') * 100).round(1),\n"
            "        residual_cloud=(pl.col('residual_cloud') * 100).round(2),\n"
            "    ))\n"
            "mean_cov = float(meta['pct_cubierto'].mean())\n"
            "mean_dates = float(meta['n_fechas'].mean())\n"
            "print(f'Cobertura media: {mean_cov:.1%} (objetivo > 70%)')\n"
            'print(f\'Fechas por patch: media {mean_dates:.1f}, min {meta["n_fechas"].min()}, max {meta["n_fechas"].max()}\')\n'
            "print(f\"Folds espaciales presentes: {sorted(meta['fold_espacial'].unique().to_list())}\")\n"
        )
    )
    cells.append(
        code(
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))\n"
            "a1.bar(cov['patch_id'].cast(pl.Utf8).to_list(), (cov['pct_cubierto']*100).to_list(), color='#2e7d32')\n"
            "a1.axhline(70, color='#c62828', linestyle='--', label='objetivo 70%')\n"
            "a1.set_title('Cobertura de pixeles-con-clase por patch'); a1.set_ylabel('% cubierto')\n"
            "a1.set_xlabel('patch_id'); a1.legend(); a1.grid(axis='y', alpha=0.3)\n"
            "a2.bar(cov['patch_id'].cast(pl.Utf8).to_list(), cov['n_fechas'].to_list(), color='#1565c0')\n"
            "a2.set_title('Fechas (T) por patch'); a2.set_ylabel('n fechas'); a2.set_xlabel('patch_id')\n"
            "a2.grid(axis='y', alpha=0.3)\n"
            "plt.tight_layout(); plt.show()\n"
        )
    )

    # ------------------------------------------------------------ closing ---
    cells.append(
        md(
            "## 6. Conclusiones del piloto\n\n"
            "El piloto de 20 patches reproduce el formato PASTIS exacto sobre un dominio "
            "nuevo (Toscana, clima mediterraneo) con textura espacial comparable a "
            "PASTIS y cobertura por encima del objetivo. La taxonomia italiana aporta "
            "clases mediterraneas que PASTIS no tiene (olive, durum) junto a las "
            "compartidas (grapevine, meadow, wheat), justo el material que US-079 "
            "necesita para medir el transfer Francia->Italia con el combinador "
            "Voting-3. Las cifras agregadas que cierran el GATE (cobertura media, clases "
            "presentes y su soporte, nube residual, fechas medias, cuota consumida y std "
            "NDVI vs PASTIS) viven en `pilot_summary.json` y en el run de MLflow."
        )
    )
    cells.append(
        code(
            "summary = json.loads((ROOT / 'pilot_summary.json').read_text(encoding='utf-8'))\n"
            "print('REPORTE GATE - piloto', summary['n_patches'], 'patches')\n"
            "print(f\"  cobertura media: {summary['mean_coverage']:.1%}\")\n"
            "print(f\"  fechas medias: {summary['mean_dates']:.1f}\")\n"
            "print(f\"  nube residual media: {summary['mean_residual_cloud']:.2%}\")\n"
            "print(f\"  clases presentes: {summary['n_classes_present']}\")\n"
            "print(f\"  std NDVI homologo: {summary['mean_ndvi_std']:.3f} (PASTIS {summary['pastis_ndvi_std']})\")\n"
            "print(f\"  peticiones SH consumidas: {summary['n_requests']}\")\n"
        )
    )
    return cells


@app.command()
def build(
    out: Annotated[Path, typer.Option(help="Ruta de salida del notebook.")] = _DEFAULT_OUT,
    data_dir: Annotated[
        Path, typer.Option(help="Ruta del dataset homologo generado.")
    ] = _DEFAULT_DATA,
) -> None:
    """Write the US-078 EDA notebook (unexecuted; papermill populates outputs).

    Args:
        out: Output ``.ipynb`` path.
        data_dir: Repo-relative path to the homologue dataset the notebook reads.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = _build_cells(str(data_dir).replace("\\", "/"))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    typer.echo(f"Notebook escrito en {out} ({len(nb.cells)} celdas).")


if __name__ == "__main__":
    app()
