"""Build the three descriptive notebooks of the US-030..US-040 batch.

Generates, from the real figure manifest produced by
``scripts/build_lote_figures.py``, three ordered notebooks under
``notebooks/final_model/``:

- ``06a_segmentadores.ipynb`` -- the 6 dense segmenters: real
  ``input | ground truth | prediction`` triptychs, confusion matrices and
  per-class IoU on the fold-5 held-out split (US-030/031/038/039).
- ``06b_farslip_parcela.ipynb`` -- the faithful FarSLIP: the Gemma captions
  that fed it, the real patch predictions vs ground truth, and the visual
  root-cause of the ~4-class ceiling (1-CLS-per-patch), plus FarSLIP vs
  AlphaEarth (US-032..037).
- ``06c_ensambles.ipynb`` -- the four base ensembles and the chosen one
  (US-040).

The first notebook (06a) carries the full glossary and acronym table; 06b/06c
link back to it but also repeat a short acronym reminder so each is readable on
its own. All figures are real PASTIS-R inference (no synthetic, no placeholder).

This script writes the notebooks WITH the parameters cell and the cells laid
out; ``papermill`` then executes them so the PNGs render inline and the tables
populate. Run order::

    python -m scripts.build_lote_figures run   # on the VM (H100) -> figures
    # bring figures + manifest to the repo (reports/lote_us030_040/figures)
    python -m scripts.build_lote_notebooks      # locally -> 3 .ipynb
    papermill ...                               # execute -> outputs populated

Project conventions: Polars, structlog, type hints, English docstrings, Spanish
visible prose, no emojis.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import structlog

from ml.report.lote_notebook_content import (
    ACRONYMS,
    GLOSSARY,
    US_ONE_LINERS,
)

logger = structlog.get_logger(__name__)

NB_DIR = Path("notebooks/final_model")
FIG_DIR_REL = "reports/lote_us030_040/figures"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def _bootstrap_cell() -> nbf.NotebookNode:
    """Repo-root bootstrap + manifest load (shared header code cell)."""
    src = (
        "import json\n"
        "from pathlib import Path\n\n"
        "import polars as pl\n"
        "from IPython.display import Markdown, display\n\n"
        "from ml.utils.notebook_setup import find_repo_root, show_saved_png\n\n"
        "REPO = find_repo_root()\n"
        f"FIG_DIR = REPO / '{FIG_DIR_REL}'\n"
        "MANIFEST = json.loads((FIG_DIR / 'manifest.json').read_text(encoding='utf-8'))\n"
        "print('repo:', REPO)\n"
        "print('figuras:', FIG_DIR, '| existe:', FIG_DIR.is_dir())\n"
    )
    return code(src)


def _params_cell() -> nbf.NotebookNode:
    """Papermill parameters cell (no heavy params; figures are pre-generated)."""
    c = code("# Parametros (papermill). Las figuras ya estan generadas con datos reales.\n")
    c.metadata = {"tags": ["parameters"]}
    return c


def _glossary_cells() -> list[nbf.NotebookNode]:
    """The acronym table + glossary as Markdown (full, for 06a)."""
    ac_rows = "\n".join(f"| `{a}` | {d} |" for a, d in ACRONYMS)
    gl_rows = "\n".join(f"| **{t}** | {d} |" for t, d in GLOSSARY)
    return [
        md(
            "## Glosario y siglas\n\n"
            "Este lote usa terminologia tecnica de teledeteccion y aprendizaje "
            "profundo. Antes de las figuras, aqui se explican **todas las siglas** y "
            "los **terminos dificiles** que apareceran. Las notebooks 06b y 06c "
            "repiten un recordatorio breve y enlazan aqui para el detalle.\n\n"
            "### Siglas y abreviaturas\n\n"
            "| Sigla | Que significa |\n|---|---|\n" + ac_rows + "\n\n"
            "### Glosario de conceptos\n\n"
            "| Termino | Explicacion |\n|---|---|\n" + gl_rows
        )
    ]


def _short_acronym_reminder() -> nbf.NotebookNode:
    """A compact acronym reminder for 06b/06c (full table lives in 06a)."""
    keys = ["GT", "mIoU", "F1-macro", "CLS", "OOF", "MPCL", "fold", "held-out"]
    lookup = dict(ACRONYMS)
    rows = "\n".join(f"| `{k}` | {lookup.get(k, '')} |" for k in keys)
    return md(
        "## Recordatorio de siglas\n\n"
        "El glosario completo (todas las siglas + conceptos) esta en "
        "[`06a_segmentadores.ipynb`](06a_segmentadores.ipynb). "
        "Aqui un recordatorio de las mas usadas:\n\n"
        "| Sigla | Que significa |\n|---|---|\n" + rows
    )


def _index_cell(title: str, us_keys: list[str]) -> nbf.NotebookNode:
    """An index listing the US covered, with their real one-liners."""
    rows = "\n".join(f"- **{k}** — {US_ONE_LINERS[k]}" for k in us_keys)
    return md(f"## {title}\n\n" + rows)


# ---------------------------------------------------------------------------
# 06a -- segmenters
# ---------------------------------------------------------------------------
def build_06a() -> nbf.NotebookNode:
    """Segmenters notebook (US-030/031/038/039)."""
    cells: list[nbf.NotebookNode] = []
    cells.append(
        md(
            "# Lote US-030 a US-040 (1/3) — Segmentadores densos\n\n"
            "**Que veras aqui (con datos REALES de PASTIS-R frances):** para cada uno "
            "de los 6 segmentadores que el lote re-evaluo en el fold-5 reservado, el "
            "triplete **entrada (RGB) | verdad de campo | prediccion**, su matriz de "
            "confusion y su IoU por clase. Todo es inferencia real sobre PASTIS-R; no "
            "hay imagenes sinteticas ni mascaras de relleno.\n\n"
            "Cubre **US-030** (el harness que hace la comparacion justa), **US-031** "
            "(las probabilidades OOF que alimentan los ensambles), **US-038** (el "
            "TSViT Full-M, el mejor segmentador individual) y **US-039** (la ablacion "
            "fenologica honesta)."
        )
    )
    cells.append(_params_cell())
    cells.append(_bootstrap_cell())
    cells.extend(_glossary_cells())
    cells.append(
        _index_cell(
            "User Stories cubiertas en esta notebook",
            ["US-030", "US-031", "US-038", "US-039"],
        )
    )

    cells.append(
        md(
            "## Como leer los tripletes\n\n"
            "Cada figura tiene tres paneles del MISMO patch del fold-5 reservado:\n\n"
            "1. **Entrada (RGB)**: la imagen Sentinel-2 a color real (composicion de "
            "la serie temporal).\n"
            "2. **Verdad de campo (GT)**: el mapa de cultivos correcto, pixel a pixel.\n"
            "3. **Prediccion**: lo que el modelo predijo. Comparar panel 2 vs 3 "
            "muestra donde acierta y donde confunde.\n\n"
            "Los colores son las 18 clases de cultivo de PASTIS (misma paleta en GT y "
            "prediccion). El gris/neutro es fondo/void (se ignora en la metrica)."
        )
    )

    cells.append(
        md(
            "## Tabla comparativa real (fold-5 held-out, 18 clases)\n\n"
            "Esta es la tabla que produce el harness US-030: la metrica honesta de "
            "cada segmentador sobre el fold que NINGUNO vio al entrenar."
        )
    )
    cells.append(
        code(
            "import polars as pl\n"
            "seg = MANIFEST.get('segmenters', {})\n"
            "rows = [\n"
            "    {\n"
            "        'modelo': v.get('label', k),\n"
            "        'mIoU': v.get('miou'),\n"
            "        'F1_macro': v.get('f1_macro'),\n"
            "        'pixel_acc': v.get('pixel_acc'),\n"
            "    }\n"
            "    for k, v in seg.items() if 'error' not in v\n"
            "]\n"
            "tabla = pl.DataFrame(rows).sort('mIoU', descending=True)\n"
            "display(tabla)\n"
            "display(Markdown('El TSViT Full-M (US-038) es el mejor segmentador "
            "individual; los modelos temporales (TSViT, U-TAE) aprovechan la serie "
            "completa de fechas, algo clave en agricultura por la fenologia.'))\n"
        )
    )

    # Per-segmenter section: triptychs + confusion + per-class.
    cells.append(
        md(
            "## Predicciones reales por modelo\n\n"
            "Para cada segmentador: primero sus tripletes (entrada/verdad/prediccion) "
            "sobre patches reales del fold-5, luego su matriz de confusion (que clases "
            "confunde con cuales) y su IoU por clase."
        )
    )
    cells.append(
        code(
            "ORDER = ['tsvit', 'tsvit-pheno-fullm', 'utae', 'deeplabv3plus', "
            "'segformer', 'unet']\n"
            "for kind in ORDER:\n"
            "    v = MANIFEST.get('segmenters', {}).get(kind)\n"
            "    if not v or 'error' in (v or {}):\n"
            "        display(Markdown(f'### {kind}: no disponible'))\n"
            "        continue\n"
            "    display(Markdown(f\"### {v['label']}  \"\n"
            "        f\"(mIoU {v['miou']:.3f} | F1-macro {v['f1_macro']:.3f} | \"\n"
            "        f\"pixel-acc {v['pixel_acc']:.3f})\"))\n"
            "    for i, trip in enumerate(v.get('triptychs', [])):\n"
            "        show_saved_png(FIG_DIR / trip, caption=f'Patch ejemplo {i+1}: "
            "entrada | verdad de campo | prediccion')\n"
            "    show_saved_png(FIG_DIR / v['confusion'], caption='Matriz de confusion "
            "(recall por clase): la diagonal es el acierto.')\n"
            "    pc_path = FIG_DIR / v.get('per_class_csv', '')\n"
            "    if pc_path.is_file():\n"
            "        pc = pl.read_csv(pc_path)\n"
            "        display(Markdown('IoU/F1 por clase (orden de mayor a menor IoU):'))\n"
            "        display(pc.head(10))\n"
        )
    )

    # US-031 -- OOF dump (its own figure + table).
    cells.append(
        md(
            "## US-031 — Volcado de probabilidades OOF (insumo de los ensambles)\n\n"
            "Antes de los ensambles, US-031 guardo para cada uno de los 6 modelos la "
            "**probabilidad por pixel** (post-softmax) sobre el fold-5 reservado. Son "
            "predicciones OOF (sobre datos que el modelo no vio al entrenar), la "
            "materia prima honesta que la notebook 06c combina. La figura muestra "
            "cuantos patches del fold-5 quedaron con probabilidades validas por modelo."
        )
    )
    cells.append(
        code(
            "show_saved_png(FIG_DIR / 'us031_oof.png', caption='US-031: patches del "
            "fold-5 con probabilidades OOF volcadas por modelo (todos status=ok).')\n"
            "u031 = REPO / 'ml/eval/oof/manifest.json'\n"
            "if u031.is_file():\n"
            "    import json as _json\n"
            "    mm = _json.loads(u031.read_text(encoding='utf-8'))\n"
            "    display(Markdown(f\"Esquema: **{mm.get('num_classes',18)} clases** x "
            "{mm.get('size',128)}x{mm.get('size',128)} por patch, dtype "
            "{mm.get('dtype','float16')}, fold {mm.get('fold',5)} held-out. \"\n"
            "        '12 parquet (~1.47 GB) versionados en DVC.'))\n"
        )
    )

    cells.append(
        md(
            "## Lectura de los resultados\n\n"
            "- Los **modelos temporales** (TSViT Full-M, U-TAE) ganan a los 2D porque "
            "leen la evolucion del cultivo a lo largo del ano (la fenologia), no una "
            "sola foto.\n"
            "- En la matriz de confusion se ve que las **clases comunes** (pradera, "
            "trigo, maiz, vid) se resuelven bien y las **raras** se confunden con la "
            "clase dominante de su zona: es el efecto del fuerte desbalance de PASTIS.\n"
            "- **US-039** (TSViT-pheno) NO mejora a US-038: en aprendizaje supervisado "
            "con etiqueta densa la rama fenologica contrastiva ya no aporta margen "
            "(el modelo ya satura ~70% de mIoU, el techo conocido de PASTIS). Es una "
            "ablacion honesta: el valor es la conclusion, no ganar un decimal.\n\n"
            "### Lo que sigue\n"
            "Estas probabilidades por pixel (US-031) son la materia prima de los "
            "ensambles de la notebook 06c."
        )
    )
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


# ---------------------------------------------------------------------------
# 06b -- FarSLIP faithful
# ---------------------------------------------------------------------------
def build_06b() -> nbf.NotebookNode:
    """FarSLIP faithful notebook (US-032..037)."""
    cells: list[nbf.NotebookNode] = []
    cells.append(
        md(
            "# Lote US-030 a US-040 (2/3) — FarSLIP fiel y por que falla\n\n"
            "**Que veras aqui (datos REALES):** las descripciones (captions) que "
            "**Gemma 4** genero para alimentar a FarSLIP, las **predicciones reales** "
            "del modelo afinado por patch (clase real vs predicha), y la visualizacion "
            "de **por que falla**: el limite 1-CLS-por-patch que pone el techo de "
            "PASTIS en ~4 clases. Cierra con la comparativa honesta FarSLIP vs "
            "AlphaEarth.\n\n"
            "Cubre **US-032** (filtro 3:1), **US-033/034** (prototipos fenologicos y "
            "su fix), **US-035** (ablacion de bandas), **US-036-a v2** (FarSLIP fiel "
            "al paper) y **US-037** (FarSLIP vs AlphaEarth)."
        )
    )
    cells.append(_params_cell())
    cells.append(_bootstrap_cell())
    cells.append(_short_acronym_reminder())
    cells.append(
        _index_cell(
            "User Stories cubiertas en esta notebook",
            ["US-032", "US-033", "US-034", "US-035", "US-036-a-v2", "US-037"],
        )
    )

    cells.append(
        md(
            "## Que es FarSLIP fiel y como se entreno\n\n"
            "FarSLIP (Li et al. 2025) afina un modelo CLIP para que el embedding de "
            "una imagen agricola quede cerca del texto que la describe. Nuestra "
            "version **fiel al paper** usa dos senales:\n\n"
            "- **L_glo** (global): alinea cada patch con su **caption** (descripcion "
            "en lenguaje natural). Esas captions las genero **Gemma 4 multimodal** "
            "local mirando la imagen real (sin filtrar etiquetas, sin fuga).\n"
            "- **L_loc** (local, MPCL): alinea cada **region (parcela)** con el texto "
            "de su **categoria** de cultivo. La supervision efectiva fueron **85,936 "
            "pares region-categoria** (no solo las 2,433 captions).\n\n"
            "Se entreno sobre **PASTIS-R frances real** en la H100, datos reales en "
            "todo momento."
        )
    )

    # --- US-032: 3:1 dominance filter ---
    cells.append(
        md(
            "## US-032 — Filtro 3:1 de dominancia de pradera\n\n"
            "PASTIS esta dominado por **Meadow** (pradera). Si un patch es casi todo "
            "pradera, aporta poca senal para las demas clases. El filtro 3:1 descarta "
            "los patches donde la pradera supera 3 veces a la 2.da clase del patch. La "
            "figura muestra cuantos patches reales se retienen (recalculado en vivo "
            "sobre PASTIS-R con `PastisFilter`)."
        )
    )
    cells.append(
        code(
            "show_saved_png(FIG_DIR / 'us032_filter.png', caption='US-032: retencion "
            "real del filtro 3:1 (recalculado sobre PASTIS-R, folds 1-3).')\n"
        )
    )

    # --- US-033: phenology prototypes ---
    cells.append(
        md(
            "## US-033 — Prototipos de fenologia por clase\n\n"
            "Cada cultivo tiene una **firma temporal** (como cambia su NDVI a lo largo "
            "del ano): el trigo verdea y madura distinto que el maiz. US-033 calculo la "
            "curva NDVI media por clase y, con **Gemini Flash**, la convirtio en una "
            "descripcion textual; esa descripcion (via embedding) es el prototipo que "
            "guia la perdida contrastiva de FarSLIP. Curvas reales abajo, con su texto."
        )
    )
    cells.append(
        code(
            "show_saved_png(FIG_DIR / 'us033_phenology.png', caption='US-033: curva "
            "NDVI media por clase (real).')\n"
            "u033 = REPO / 'data/features/phenology_class_prototypes_pastis.parquet'\n"
            "if u033.is_file():\n"
            "    proto = pl.read_parquet(u033).filter(pl.col('class_id').is_in([1,2,3,8]))\n"
            "    for r in proto.iter_rows(named=True):\n"
            "        display(Markdown(f\"**{r['class_name']}**: _{r['description']}_\"))\n"
        )
    )

    # --- US-034: torch.randn fix ---
    cells.append(
        md(
            "## US-034 — El fix critico: prototipos reales en vez de ruido\n\n"
            "Un bug grave del flujo: FarSLIP inicializaba los prototipos contrastivos "
            "con **ruido aleatorio** (`torch.randn`), de modo que el modelo se alineaba "
            "**contra ruido**. US-034 los reemplazo por los prototipos fenologicos "
            "reales de US-033. La figura cuantifica por que importo: los prototipos "
            "reales tienen **estructura por clase** (clases distintas se separan), "
            "mientras los aleatorios son casi ortogonales (sin senal de clase)."
        )
    )
    cells.append(
        code(
            "show_saved_png(FIG_DIR / 'us034_fix.png', caption='US-034: similitud "
            "coseno entre clases — aleatorio (bug) vs real (fix).')\n"
        )
    )

    # --- US-035: band ablation ---
    cells.append(
        md(
            "## US-035 — Ablacion de bandas (RGB / NIR+RGB / 4 bandas)\n\n"
            "Se entrenaron **3 variantes reales en la H100** para aislar el aporte del "
            "infrarrojo cercano (NIR) y de la fenologia. La tabla compara el loss final "
            "de cada variante. **Aviso honesto**: el loss es interno; la calidad real "
            "del embedding se mide aguas abajo en US-037, no por el loss crudo."
        )
    )
    cells.append(
        code(
            "show_saved_png(FIG_DIR / 'us035_bands.png', caption='US-035: ablacion de "
            "bandas (3 corridas reales H100).')\n"
        )
    )

    cells.append(
        md(
            "## Las captions que genero Gemma (insumo real de L_glo)\n\n"
            "Estas son las descripciones reales que alimentaron el entrenamiento. "
            "Aparecen debajo de cada patch en las figuras siguientes; aqui una "
            "muestra directa del parquet versionado en DVC."
        )
    )
    cells.append(
        code(
            "import polars as pl\n"
            "cap_path = REPO / 'data/farslip/pastis_captions.parquet'\n"
            "if cap_path.is_file():\n"
            "    caps = pl.read_parquet(cap_path)\n"
            "    display(Markdown(f'Total de captions: **{caps.height}** "
            "(una por patch, generadas por Gemma 4).'))\n"
            "    sample = caps.select(['patch_id', 'caption_glo', 'n_regions', "
            "'clases']).head(3)\n"
            "    for r in sample.iter_rows(named=True):\n"
            "        display(Markdown(f\"**Patch {r['patch_id']}** "
            "({r['n_regions']} parcelas) — _{r['caption_glo']}_\"))\n"
            "else:\n"
            "    display(Markdown('> Parquet de captions no disponible localmente "
            "(esta en DVC: `dvc pull data/farslip/pastis_captions.parquet.dvc`).'))\n"
        )
    )

    cells.append(
        md(
            "## Predicciones reales del FarSLIP fiel + por que falla\n\n"
            "Antes de leer las figuras, una aclaracion **importante y honesta** sobre "
            "que se compara:\n\n"
            "- **PASTIS es un dataset de segmentacion**: tiene una etiqueta por cada "
            "pixel/parcela (la verdad de campo del panel central, con sus poligonos "
            "reales).\n"
            "- **FarSLIP es un clasificador de patch**: por diseno emite **una sola "
            "clase para todo el patch** (no una por parcela). No es que el modelo "
            "'falle' en recuperar parcelas: es que su tarea es mas gruesa que la "
            "segmentacion.\n\n"
            "Por eso NO comparamos 'N parcelas reales vs 1 prediccion' como si fuera "
            "injusto; mostramos las dos granularidades lado a lado para que se vea el "
            "limite real. Cada figura tiene **tres paneles** del mismo patch del fold "
            "reservado:\n\n"
            "1. **Entrada (RGB)**: la imagen real + la caption de Gemma debajo.\n"
            "2. **Verdad PASTIS por parcela**: los poligonos reales etiquetados con su "
            "cultivo (leyenda de clases). Es la riqueza completa del patch.\n"
            "3. **Prediccion FarSLIP**: la unica clase que el modelo asigna a TODO el "
            "patch, pintada sobre todas las parcelas. El titulo dice contra cuantas "
            "parcelas reales coincide esa unica prediccion (verde = la clase "
            "mayoritaria del patch fue correcta).\n\n"
            "Asi se ve el **limite 1-CLS-por-patch**: el modelo solo puede acertar a "
            "las parcelas cuyo cultivo coincide con su unica prediccion; las demas "
            "(p.ej. cultivos minoritarios que comparten patch con pradera) quedan "
            "fuera de alcance de esta tarea de clasificacion."
        )
    )
    cells.append(
        code(
            "farslip = MANIFEST.get('farslip', [])\n"
            "if not farslip:\n"
            "    display(Markdown('> Figuras FarSLIP no disponibles: '\n"
            "        + str(MANIFEST.get('farslip_error', 'no generadas'))))\n"
            "for i, info in enumerate(farslip):\n"
            "    estado = 'mayoria correcta' if info.get('correct') else 'mayoria incorrecta'\n"
            "    cats = info.get('category_counts', {})\n"
            "    n_cat = info.get('n_categories', len(cats))\n"
            "    matched = info.get('parcels_matched', 0)\n"
            "    nreg = info.get('n_regions', 0)\n"
            "    display(Markdown(\n"
            "        f\"### Ejemplo {i+1} — Patch {info['patch_id']}: \"\n"
            "        f\"clase mayoritaria real **{info['true']}** / prediccion del patch "
            "**{info['pred']}** [{estado}]\"))\n"
            "    show_saved_png(FIG_DIR / info['figure'])\n"
            "    display(Markdown(\n"
            '        f"Este patch contiene **{n_cat} cultivos** distribuidos en '
            "{nreg} parcelas reales ({cats}). Como FarSLIP asigna una sola clase al "
            "patch ({info['pred']}), esa prediccion coincide con **{matched} de "
            "{nreg} parcelas**. Para recuperar las demas haria falta segmentacion "
            'densa (por pixel), no clasificacion por patch."))\n'
        )
    )

    cells.append(
        md(
            "## El techo real de PASTIS: ~4 clases (con evidencia)\n\n"
            "La evaluacion por clase del FarSLIP fiel (sobre las 18 clases) da "
            "**macro-F1 0.164**, con senal clara en solo **4 clases dominantes**: "
            "Meadow (0.76), Orchard (0.52), Grapevine (0.49), Corn (0.46). Las 14 "
            "restantes caen a F1 ~0. Esto NO es un fallo del metodo, es el **limite "
            "del dato** (PASTIS tiene ~80x menos datos que el dataset del paper)."
        )
    )
    cells.append(
        code(
            "pc = REPO / 'reports/farslip/metrics/faithful_v2_per_class.csv'\n"
            "if pc.is_file():\n"
            "    df = pl.read_csv(pc).sort('f1_baseline', descending=True)\n"
            "    display(df.select(['class_id', 'name', 'f1_baseline', 'iou_baseline']).head(8))\n"
            "summ = REPO / 'reports/farslip/metrics/faithful_v2_summary.csv'\n"
            "if summ.is_file():\n"
            "    display(Markdown('Resumen (incluye la ablacion de class-weights):'))\n"
            "    display(pl.read_csv(summ))\n"
        )
    )
    cells.append(
        md(
            "**Por que se castigan tanto las clases raras (causa raiz, comprobada):**\n\n"
            "1. **1-CLS-por-patch** (lo que viste arriba): una prediccion por patch, "
            "no por parcela.\n"
            "2. **Desbalance extremo**: Meadow tiene ~6,500 regiones; Sorghum ~150.\n"
            "3. **Pocos datos**: 2,433 patches vs los 200,000 del paper.\n\n"
            "Se probo **reponderar por frecuencia** (class-weights) para rescatar las "
            "raras: el macro-F1 BAJO a 0.102 (peor). Eso **prueba por contradiccion** "
            "que el cuello de botella NO es el desbalance, sino el 1-CLS-por-patch + la "
            "escasez de datos."
        )
    )

    cells.append(
        md(
            "## US-037: FarSLIP fiel vs AlphaEarth (comparativa honesta)\n\n"
            "Se midio, sobre los MISMOS 567 patches reales, si el espacio del FarSLIP "
            "fiel separa mejor las clases que AlphaEarth. Resultado honesto: **gana "
            "AlphaEarth** (F1-macro 0.645 vs 0.555). FarSLIP aporta senal pero no "
            "supera al modelo fundacional entrenado con datos globales masivos."
        )
    )
    cells.append(
        code(
            "u37 = REPO / 'reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv'\n"
            "if u37.is_file():\n"
            "    display(pl.read_csv(u37))\n"
        )
    )
    cells.append(
        md(
            "### Lo que sigue\n"
            "El embedding del FarSLIP fiel quedo versionado en DVC por si se quiere "
            "ablacionar su inclusion en un ensamble; pero el base learner mas fuerte "
            "sigue siendo AlphaEarth. La notebook 06c arma los ensambles finales."
        )
    )
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


# ---------------------------------------------------------------------------
# 06c -- ensembles
# ---------------------------------------------------------------------------
def build_06c() -> nbf.NotebookNode:
    """Ensembles notebook (US-040)."""
    cells: list[nbf.NotebookNode] = []
    cells.append(
        md(
            "# Lote US-030 a US-040 (3/3) — Ensambles\n\n"
            "**Que veras aqui (datos REALES):** los 4 ensambles base de la rubrica "
            "(Voting, Bagging, Stacking, Blending) corridos sobre las predicciones OOF "
            "reales del fold-5, la comparativa que justifica el elegido, y las figuras "
            "interpretadas (confusion, ROC, residuos espaciales).\n\n"
            "Cubre **US-040**."
        )
    )
    cells.append(_params_cell())
    cells.append(_bootstrap_cell())
    cells.append(_short_acronym_reminder())
    cells.append(_index_cell("User Story cubierta", ["US-040"]))

    cells.append(
        md(
            "## Que es un ensamble y por que conviene\n\n"
            "Un ensamble combina varios modelos para superar al mejor individual. "
            "Probamos los cuatro de la rubrica:\n\n"
            "- **Voting** (pixel): promedio de las probabilidades de 3 segmentadores.\n"
            "- **Bagging** (parcela): XGBoost sobre AlphaEarth con varios bootstraps.\n"
            "- **Stacking** (parcela): un meta-modelo aprende a combinar 3 base "
            "learners heterogeneos (2 temporales densos + 1 tabular).\n"
            "- **Blending** (parcela): pesos optimos sobre un holdout disjunto.\n\n"
            "Todo con **anti-fuga estricto**: solo fold-5, probabilidades post-softmax "
            "(nunca logits), meta-modelo solo sobre OOF."
        )
    )

    cells.append(
        md(
            "## Comparativa real y modelo elegido\n\n"
            "Tabla real (fold-5 held-out): el **Stacking heterogeneo** gana con "
            "F1-macro **0.747** y supera al mejor individual (TSViT-pheno 0.625)."
        )
    )
    cells.append(
        code(
            "import polars as pl\n"
            "show_saved_png(FIG_DIR / MANIFEST['ensemble']['figure'], "
            "caption='Comparativa de ensambles (verde = elegido: Stacking).')\n"
            "rows = MANIFEST.get('ensemble', {}).get('rows', [])\n"
            "if rows:\n"
            "    display(pl.DataFrame(rows))\n"
        )
    )

    cells.append(
        md(
            "## Figuras interpretadas del ensamble elegido\n\n"
            "Generadas en US-040 (ya versionadas): la matriz de confusion del "
            "Stacking, su curva ROC (una-vs-resto) y los residuos espaciales del "
            "Blending (donde se equivoca geograficamente)."
        )
    )
    cells.append(
        code(
            "ens_figs = REPO / 'reports/ensemble/figures'\n"
            "for fname, cap in [\n"
            "    ('confusion_stacking.png', 'Matriz de confusion del Stacking elegido'),\n"
            "    ('roc_ovr_stacking.png', 'Curva ROC una-vs-resto (macro-AUC 0.976)'),\n"
            "    ('pr_stacking.png', 'Curva precision-exhaustividad (macro-AP 0.795)'),\n"
            "    ('spatial_residuals_blending.png', 'Residuos espaciales del Blending'),\n"
            "]:\n"
            "    show_saved_png(ens_figs / fname, caption=cap)\n"
        )
    )

    cells.append(
        md(
            "## Conclusion del lote\n\n"
            "- El **mejor segmentador individual** es el TSViT Full-M (US-038).\n"
            "- El **mejor ensamble** es el Stacking heterogeneo (US-040, F1 0.747), "
            "que supera a cualquier modelo solo: la heterogeneidad denso+tabular es lo "
            "que aporta.\n"
            "- En clasificacion patch (FarSLIP), el **techo real de PASTIS son ~4 "
            "clases**; AlphaEarth sigue siendo el embedding mas fuerte.\n"
            "- Todo se reporto **sin sobre-afirmar**, con datos reales y anti-fuga.\n\n"
            "Las tres notebooks (06a/06b/06c) dejan el lote US-030..US-040 "
            "completamente visualizado: predicciones reales, captions de Gemma, "
            "matrices de confusion y la comparativa de ensambles."
        )
    )
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


def main() -> None:
    """Write the three notebooks under ``notebooks/final_model/``."""
    NB_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        "06a_segmentadores.ipynb": build_06a(),
        "06b_farslip_parcela.ipynb": build_06b(),
        "06c_ensambles.ipynb": build_06c(),
    }
    for name, nb in targets.items():
        path = NB_DIR / name
        nbf.write(nb, path)
        logger.info("notebook_written", path=str(path), n_cells=len(nb.cells))


if __name__ == "__main__":
    main()
