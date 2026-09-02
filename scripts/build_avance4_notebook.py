"""Builder of the Avance 4 integrator notebook (Equipo 17): the 6 segmentation models.

Generates ``notebooks/segmentation/Avance4.Equipo17.ipynb`` programmatically and
reproducibly (same pattern as ``scripts/build_avance3_notebook.py``). It is the
consolidated deliverable of the Avance 4 rubric: it consumes the comparison
parquets that each team member exports
(``reports/segmentation/model_comparison_avance4_*.parquet``), builds the table of
the 6 models sorted by the main metric (mIoU), selects the top-2, documents the
fine-tuning (Optuna) and the choice of the final individual model.

The notebook is a **functional skeleton**: it degrades gracefully showing only the
available parquets (today: Aaron's models #1/#6). As Arthur and Isaac export theirs
(#2-#5), the table is completed without touching the notebook.

Usage::

    poetry run python scripts/build_avance4_notebook.py \\
        --out notebooks/segmentation/Avance4.Equipo17.ipynb

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/segmentation/Avance4.Equipo17.ipynb")


def _build_cells() -> list:
    """Build the list of cells (markdown + code) of the integrator notebook."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = []

    cells.append(
        md(
            "# Avance 4 — Modelos alternativos (Equipo 17, AgroSatCopilot)\n\n"
            "## Proyecto Integrador MNA · Tec de Monterrey\n\n"
            "**Equipo 17**\n\n"
            "- Carlos Isaac Ávila Gutiérrez — A01796035\n"
            "- Carlos Aaron Bocanegra Buitrón — A01796345\n"
            "- Arthur Jafed Zizumbo Velasco — A01796363\n\n"
            "**Curso**: MNA — Tec de Monterrey · 20-abr → 3-jul-2026\n\n"
            "**Sponsor académico**: Dr. Gerardo José Camacho — gjcamacho@tec.mx\n\n"
            "**Fecha de entrega**: 2026-05-31.\n\n"
            "---\n\n"
            "## Resumen ejecutivo\n\n"
            "Este cuaderno **consolida** las **6 arquitecturas individuales** (no ensambles) de "
            "segmentación semántica densa de cultivos sobre PASTIS-R, repartidas entre el equipo. "
            "Consume los parquets comparativos que cada integrante exporta, construye la tabla de "
            "los 6 modelos ordenada por la métrica principal (mIoU), selecciona el top-2, documenta "
            "el ajuste fino (Optuna) y la elección del modelo individual final.\n\n"
            "| # | Modelo | Tipo |\n"
            "|---|--------|------|\n"
            "| 1 | U-Net ResNet-50 | CNN clásica 2D |\n"
            "| 2 | DeepLabv3+ MobileNet | CNN eficiente ASPP |\n"
            "| 3 | SegFormer-B0 | Transformer spatial |\n"
            "| 4 | U-TAE | Temporal Attention |\n"
            "| 5 | TSViT | Transformer temporal (Paper 1) |\n"
            "| 6 | AnySat frozen + linear head | Foundation model congelado |\n\n"
            "> Cada integrante entrena sus modelos con `ml.train.train_segmentation.run_training` y "
            "exporta su parquet a `reports/segmentation/model_comparison_avance4_<nombre>.parquet`. "
            "Este notebook los une."
        )
    )

    cells.append(
        md(
            "## Objetivos y rubrica\n\n"
            "- **3.3** Explorar una gama diversa de tecnicas (6 arquitecturas).\n"
            "- **3.4** Encontrar la configuracion optima (ajuste fino del top-2).\n\n"
            "**Distribucion de puntos**:\n\n"
            "- **Comparativa (60 pts)**: >=6 modelos, tabla ordenada por mIoU + F1-macro + "
            "pixel-accuracy + tiempos de entrenamiento.\n"
            "- **Ajuste fino (30 pts)**: Optuna (>=30 trials) sobre los 2 mejores.\n"
            "- **Modelo individual (10 pts)**: justificacion del final (trade-offs, no solo metrica)."
        )
    )

    cells.append(
        code(
            "# --- Setup Colab + Drive compartido del equipo ---\n"
            "import os, subprocess, sys\n"
            "from pathlib import Path\n\n"
            "# Monta Drive y prefija las rutas con shared_folder_path (vacio en local).\n"
            "_IN_COLAB = False\n"
            "shared_folder_path = ''\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    shared_folder_path = '/content/drive/MyDrive/Integrador/'\n"
            "    _IN_COLAB = True\n"
            "except ImportError:\n"
            "    pass\n\n"
            "# Este cuaderno es integrador: solo consolida parquets/figuras desde Drive y NO\n"
            "# importa codigo del repo. Si el repo esta presente se usa como cwd, pero no es\n"
            "# obligatorio (a diferencia de los notebooks de modelo, que si clonan el repo).\n"
            "_search = [Path.cwd().resolve(), *Path.cwd().resolve().parents]\n"
            "if _IN_COLAB:\n"
            "    _search = [Path('/content/agrosat-copilot'), *_search]\n"
            "for _cand in _search:\n"
            "    if (_cand / 'pyproject.toml').is_file():\n"
            "        if str(_cand) not in sys.path:\n"
            "            sys.path.insert(0, str(_cand))\n"
            "        os.chdir(_cand)\n"
            "        break\n\n"
            "# Dependencias de visualizacion (polars no viene por defecto en Colab).\n"
            "if _IN_COLAB:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'polars'], check=False)\n\n"
            "import matplotlib.pyplot as plt\n"
            "import polars as pl\n"
            "print('repo:', Path.cwd(), '| colab:', _IN_COLAB, '| drive:', shared_folder_path or '(local)')"
        )
    )

    cells.append(
        md(
            "## Metodologia\n\n"
            "- **Dataset**: PASTIS-R (2433 patches Sentinel-2 multitemporales, 20 clases, "
            "folds oficiales espacialmente disjuntos -> sin leakage).\n"
            "- **Convencion compartida**: `num_classes=20`, `ignore_index=19` (void), "
            "resolucion **256px**, split train=folds[1,2,3] / val=fold[4] / test=fold[5].\n"
            "- **Metricas**: mIoU (principal) + F1-macro + pixel-accuracy, todas pixel-level "
            "(`ml/eval/dense_metrics.py`).\n"
            "- **Tracking**: 1 run MLflow por modelo con tag `architecture`."
        )
    )

    cells.append(md("## Comparativa de los 6 modelos (60 pts)"))

    cells.append(
        code(
            "# --- Consolidar los parquets de los integrantes (desde el Drive compartido) ---\n"
            "REPORTS = Path((shared_folder_path if shared_folder_path else '')\n"
            "               + 'reports/segmentation/metrics')\n"
            "FIGURES = Path((shared_folder_path if shared_folder_path else '')\n"
            "               + 'reports/segmentation/figures')\n"
            "# Los 6 modelos del Avance 4. El integrador muestra los que ya exportaron su\n"
            "# parquet y deja placeholder para los pendientes (no hay que tocar el notebook).\n"
            "EXPECTED = ['unet', 'anysat', 'deeplabv3plus', 'segformer', 'utae', 'tsvit']\n"
            "parts = sorted(REPORTS.glob('model_comparison_avance4_*.parquet'))\n"
            "_avail = {p.stem.replace('model_comparison_avance4_', '').replace('_fast', '')\n"
            "          for p in parts}\n"
            "print('disponibles:', sorted(_avail))\n"
            "print('pendientes :', [m for m in EXPECTED if m not in _avail])\n\n"
            "if parts:\n"
            "    table = pl.concat([pl.read_parquet(p) for p in parts], how='vertical_relaxed')\n"
            "    table = table.unique(subset=['model'], keep='last').sort('miou', descending=True)\n"
            "else:\n"
            "    print('Aun no hay parquets en', REPORTS)\n"
            "    table = pl.DataFrame()\n\n"
            "_cols = ['model', 'miou_grouped', 'f1_macro_grouped', 'pixel_accuracy_grouped',\n"
            "         'miou', 'f1_macro', 'pixel_accuracy', 'train_time_s', 'epochs']\n"
            "table.select([c for c in _cols if c in table.columns]) if table.height else table"
        )
    )

    cells.append(
        code(
            "# --- Barplot comparativo (mIoU por modelo) ---\n"
            "if table.height:\n"
            "    fig, ax = plt.subplots(figsize=(8, 4))\n"
            "    ax.barh(table['model'].to_list()[::-1], table['miou'].to_list()[::-1], color='#2b6cb0')\n"
            "    ax.set_xlabel('mIoU (val)')\n"
            "    ax.set_title('Avance 4 - Comparativa de arquitecturas de segmentacion')\n"
            "    fig.tight_layout()\n"
            "    display(fig)\n"
            "else:\n"
            "    print('Tabla vacia: nada que graficar todavia.')"
        )
    )

    cells.append(
        md(
            "## Reportes visuales por modelo\n\n"
            "Para cada modelo: curvas de entrenamiento, IoU por clase, matriz de confusión y la "
            "comparación RGB / verdad / predicción, tal como las dejó cada notebook en "
            "`reports/segmentation/figures/`. Los modelos que todavía no corrieron muestran un "
            "placeholder, así el cuaderno se completa solo a medida que cada uno entrena."
        )
    )

    cells.append(
        code(
            "# --- Galeria de figuras por modelo (desde figures/ en Drive) ---\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "def _find_fig(key, model):\n"
            "    # Acepta el nombre exacto y variantes con sufijo (p.ej. anysat -> anysat_fast).\n"
            "    _exact = FIGURES / f'{key}_{model}.png'\n"
            "    if _exact.exists():\n"
            "        return _exact\n"
            "    _alt = sorted(FIGURES.glob(f'{key}_{model}_*.png'))\n"
            "    return _alt[0] if _alt else None\n\n"
            "_fig_types = [('curves', 'Curvas de entrenamiento'),\n"
            "              ('per_class_iou', 'IoU por clase'),\n"
            "              ('confusion', 'Matriz de confusion'),\n"
            "              ('samples', 'RGB / verdad / prediccion')]\n"
            "for _m in EXPECTED:\n"
            "    display(Markdown(f'### {_m}'))\n"
            "    _shown = False\n"
            "    for _key, _label in _fig_types:\n"
            "        _f = _find_fig(_key, _m)\n"
            "        if _f is not None:\n"
            "            display(Markdown(f'**{_label}**'))\n"
            "            display(Image(filename=str(_f)))\n"
            "            _shown = True\n"
            "    if not _shown:\n"
            "        display(Markdown(f'_Pendiente: aun no hay figuras para `{_m}` "
            "(correr su notebook de entrenamiento)._'))"
        )
    )

    cells.append(
        md(
            "## Seleccion de los 2 mejores modelos\n\n"
            "Por la metrica principal (mIoU); empate se rompe por F1-macro -> pixel-accuracy."
        )
    )

    cells.append(
        code(
            "# --- Top-2 ---\n"
            "if table.height >= 2:\n"
            "    top2 = table.head(2)\n"
            "    print('Top-2 por mIoU:', top2['model'].to_list())\n"
            "    top2.select([c for c in _cols if c in top2.columns])\n"
            "else:\n"
            "    print('Se requieren >=2 modelos para seleccionar el top-2.')\n"
            "    top2 = table"
        )
    )

    cells.append(
        md(
            "## Ajuste fino del top-2 (30 pts)\n\n"
            "Cada modelo del top-2 se afina con **Optuna (>=30 trials)** sobre `lr`, `weight_decay` "
            "y `batch_size`, reusando `ml.train.train_segmentation.run_training` (ver el hook al "
            "final de cada notebook de modelo, p. ej. `04d_segmentation_unet`). Los resultados "
            "afinados se exportan a `reports/segmentation/metrics/tuning_<modelo>.parquet` y se "
            "cargan aqui."
        )
    )

    cells.append(
        code(
            "# --- Resultados del ajuste fino (si existen) ---\n"
            "tuning_parts = sorted(REPORTS.glob('tuning_*.parquet'))\n"
            "if tuning_parts:\n"
            "    tuning = pl.concat([pl.read_parquet(p) for p in tuning_parts], how='vertical_relaxed')\n"
            "    display(tuning)\n"
            "else:\n"
            "    print('Pendiente: ejecutar Optuna sobre el top-2 una vez confirmados los 6 modelos.')\n"
            "    tuning = pl.DataFrame()"
        )
    )

    cells.append(
        md(
            "## Modelo individual final (10 pts)\n\n"
            "Eleccion argumentada con trade-offs (no solo la metrica):\n\n"
            "- **Rendimiento**: mIoU / F1-macro tras el ajuste fino.\n"
            "- **Costo de computo**: tiempo de entrenamiento e inferencia (relevante para el "
            "presupuesto L4/H100 del proyecto).\n"
            "- **Interpretabilidad y robustez**: las CNN (U-Net/DeepLabv3+) son mas simples de "
            "diagnosticar; los modelos temporales (U-TAE/TSViT) capturan fenologia; AnySat ofrece "
            "un FM congelado con minima capacidad entrenable.\n\n"
            "> Completar tras la corrida real con el modelo ganador y su justificacion."
        )
    )

    cells.append(
        code(
            "# --- Modelo final + sus reportes visuales ---\n"
            "from IPython.display import Image, Markdown, display\n"
            "if table.height:\n"
            "    final_model = table.row(0, named=True)['model']\n"
            "    _r = table.row(0, named=True)\n"
            "    display(Markdown(\n"
            "        f'**Modelo individual final (por mIoU): `{final_model}`** - '\n"
            "        f'mIoU 6 grupos = {_r.get(\"miou_grouped\")}, '\n"
            "        f'F1-macro 6 grupos = {_r.get(\"f1_macro_grouped\")}'\n"
            "    ))\n"
            "    for _key in ('confusion', 'per_class_iou', 'samples'):\n"
            "        _f = _find_fig(_key, final_model)\n"
            "        if _f is not None:\n"
            "            display(Image(filename=str(_f)))\n"
            "else:\n"
            "    print('Definir el modelo final tras consolidar los modelos.')"
        )
    )

    cells.append(
        md(
            "## Conclusiones y checklist de la rubrica\n\n"
            "- [ ] **Comparativa (60)**: >=6 modelos en la tabla, ordenados por mIoU + F1-macro + "
            "pixel-accuracy + tiempos.\n"
            "- [ ] **Ajuste fino (30)**: Optuna >=30 trials sobre el top-2, mejora documentada.\n"
            "- [ ] **Modelo individual (10)**: final elegido con argumentos de trade-offs.\n\n"
            "**Entrega**: liga de GitHub, ejecucion secuencial, nombre `Avance4.Equipo17`."
        )
    )

    return cells


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Ruta del notebook de salida.")] = _DEFAULT_OUT,
) -> None:
    """Generate the Avance 4 integrator notebook.

    Args:
        out: Destination path of the ``.ipynb``.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
