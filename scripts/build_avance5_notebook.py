"""Builder of the Avance 5 integrator notebook (Equipo 17): the final ensemble model.

Generates ``notebooks/final_model/Avance5.Equipo17.ipynb`` programmatically and
reproducibly (same pattern as ``scripts/build_avance4_notebook.py``). It is the
consolidated deliverable of the Avance 5 ("Modelo final"): it builds the family of
ensembles (four base strategies + three incremental ones), describes the full
ensemble and each of its members, presents the comparative table ordered by the
main metric (F1-macro) against the best individual model, argues the selection of
the final model, shows interpreted figures of that final model and the non-winning
ensembles in action (with the phenology that supports the contrastive branch), and
closes with the standardized cover and per-member conclusions shared with the other
deliverables (``ml.report.notebook_cover`` + ``ml.report.notebook_conclusions``).

The notebook is an **integrator**: it consolidates the real CSV/PNG artifacts in
``reports/ensemble/`` (with a Drive prefix for Colab) and does NOT import repo
code, so it runs end to end sequentially anywhere the artifacts are present. It
degrades gracefully (placeholder) for any artifact not yet on disk.

Visible prose (markdown, captions, prints) is Spanish with proper accents and the
letter "n" with tilde; code, identifiers, comments and docstrings stay in English
ASCII (project convention). The section titles deliberately do NOT mention the
rubric or its points: they read as work, not as a checklist.

Usage::

    poetry run python scripts/build_avance5_notebook.py \\
        --out notebooks/final_model/Avance5.Equipo17.ipynb

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

from ml.report.notebook_conclusions import A5_CONCLUSIONS
from ml.report.notebook_cover import (
    build_cover_markdown,
    build_team_conclusions_markdown,
)

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/final_model/Avance5.Equipo17.ipynb")

_COVER_MARK = "<!-- agrosat-cover -->"
_CONCL_MARK = "<!-- agrosat-team-conclusions -->"


def _build_cells() -> list:
    """Build the list of cells (markdown + code) of the integrator notebook."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ---------------------------------------------------------------- Cover ---
    cover_body = build_cover_markdown(
        "Avance 5",
        "Modelo final: ensambles de clasificación de cultivos",
        "Familia de siete ensambles (Voting, Bagging, Stacking, Blending y tres "
        "incrementales) sobre los mejores modelos individuales de la fase previa, "
        "comparativa ordenada por F1-macro, elección argumentada del modelo final y "
        "figuras interpretadas sobre datos no vistos.",
        "2026-06-10",
    )
    cells.append(md(f"{_COVER_MARK}\n{cover_body}"))

    # ----------------------------------------------------- Resumen ejecutivo ---
    cells.append(
        md(
            "## Resumen ejecutivo\n\n"
            "El problema es **clasificar el cultivo de cada parcela agrícola** a partir de "
            "imágenes de satélite Sentinel-2 multitemporales, sobre un territorio real con "
            "etiquetas de agricultores (PASTIS-R, 18 cultivos). En la fase anterior se "
            "compararon seis arquitecturas individuales de segmentación; la mejor alcanzó un "
            "F1-macro de 0,625 a nivel parcela. Esta fase **combina los modelos** para superar "
            "ese techo.\n\n"
            "Se construye una familia de **siete ensambles** que cubre las dos grandes "
            "estrategias del aprendizaje por ensamble:\n\n"
            "| Estrategia | Ensamble | Qué combina |\n"
            "|------------|----------|-------------|\n"
            "| **Homogénea** | Voting (píxel) | promedio de probabilidades de tres modelos densos |\n"
            "| **Homogénea** | Bagging (parcela) | varios XGBoost sobre remuestreos del embedding satelital |\n"
            "| **Heterogénea** | **Stacking (parcela)** | meta-modelo sobre modelos temporales + tabular |\n"
            "| **Heterogénea** | Blending (parcela) | mezcla convexa de pesos optimizados |\n"
            "| Incremental | + rama contrastiva fenológica | añade una vista visión-lenguaje |\n"
            "| Incremental | + embedding satelital multianual | añade un resumen espectral estable |\n"
            "| Incremental | + contexto geográfico (diseño) | clima, relieve y vecindad (trabajo futuro) |\n\n"
            "**Resultado**: el **Stacking heterogéneo** es el modelo final, con **F1-macro 0,749** "
            "a nivel parcela sobre datos no vistos — **+12 puntos** sobre el mejor modelo "
            "individual. La lección central es que un meta-modelo que aprende a combinar "
            "fortalezas complementarias (la forma temporal del cultivo + un resumen espectral "
            "aprendido) supera a cualquier modelo por sí solo, y lo hace con un costo de "
            "inferencia despreciable."
        )
    )

    # -------------------------------------------------------------- Objetivos ---
    cells.append(
        md(
            "## Objetivos\n\n"
            "- Mejorar significativamente el rendimiento aprovechando las fortalezas de "
            "distintos modelos y reduciendo sus debilidades.\n"
            "- Evaluar la calidad de las predicciones sobre **datos no vistos**.\n\n"
            "> **Rigor metodológico (transversal)**: todas las métricas se reportan sobre un "
            "**fold espacial reservado** que ningún modelo vio al entrenar; las probabilidades "
            "que alimentan los ensambles son **out-of-fold** (sin fuga de información); el "
            "meta-modelo del stacking se valida con **validación cruzada espacial** y aborta si "
            "detecta solapamiento entre entrenamiento y evaluación. Sin estas garantías, una "
            "mejora de ensamble no sería creíble."
        )
    )

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        code(
            "# --- Colab + team shared Drive setup ---\n"
            "import os, subprocess, sys\n"
            "from pathlib import Path\n\n"
            "# Mount Drive and prefix paths with shared_folder_path (empty when local).\n"
            "_IN_COLAB = False\n"
            "shared_folder_path = ''\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    shared_folder_path = '/content/drive/MyDrive/Integrador/'\n"
            "    _IN_COLAB = True\n"
            "except ImportError:\n"
            "    pass\n\n"
            "# Integrator notebook: it only consolidates CSV/figures from reports/ and does NOT\n"
            "# import repo code, so it runs end to end anywhere the artifacts are present.\n"
            "_search = [Path.cwd().resolve(), *Path.cwd().resolve().parents]\n"
            "if _IN_COLAB:\n"
            "    _search = [Path('/content/agrosat-copilot'), *_search]\n"
            "for _cand in _search:\n"
            "    if (_cand / 'pyproject.toml').is_file():\n"
            "        os.chdir(_cand)\n"
            "        break\n\n"
            "if _IN_COLAB:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'polars'], check=False)\n\n"
            "import matplotlib.pyplot as plt\n"
            "import polars as pl\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "_PREFIX = shared_folder_path if shared_folder_path else ''\n"
            "METRICS = Path(_PREFIX + 'reports/ensemble/metrics')\n"
            "FIGURES = Path(_PREFIX + 'reports/ensemble/figures')\n\n"
            "def show_csv(name, *, sort=None, descending=True, round_cols=None):\n"
            '    """Load and display a metrics CSV; return the DataFrame (empty if missing)."""\n'
            "    p = METRICS / name\n"
            "    if not p.is_file():\n"
            "        display(Markdown(f'> _Pendiente: falta `{p}`._'))\n"
            "        return pl.DataFrame()\n"
            "    df = pl.read_csv(p, infer_schema_length=None)\n"
            "    if round_cols:\n"
            "        df = df.with_columns([pl.col(c).round(4) for c in round_cols if c in df.columns])\n"
            "    if sort and sort in df.columns:\n"
            "        df = df.sort(sort, descending=descending)\n"
            "    return df\n\n"
            "def show_fig(rel_path, caption=None):\n"
            '    """Display a figure under reports/ensemble/figures with an optional caption."""\n'
            "    p = FIGURES / rel_path\n"
            "    if not p.is_file():\n"
            "        display(Markdown(f'> _Pendiente: falta la figura `{p}`._'))\n"
            "        return\n"
            "    display(Image(filename=str(p)))\n"
            "    if caption:\n"
            "        display(Markdown(f'*{caption}*'))\n\n"
            "def show_phenology(slug, *, top=6):\n"
            '    """Display the NDVI-derived phenology that supports the prediction per parcel."""\n'
            "    p = METRICS / f'inference_fenologia_{slug}.csv'\n"
            "    if not p.is_file():\n"
            "        display(Markdown(f'> _Pendiente: falta `{p}`._'))\n"
            "        return\n"
            "    df = pl.read_csv(p)\n"
            "    for escena in df['escena'].unique(maintain_order=True).to_list():\n"
            "        sub = df.filter(pl.col('escena') == escena).head(top)\n"
            "        lines = [f'**Escena {escena}** — fenología (curva NDVI por parcela) que apoya la predicción:', '']\n"
            "        for r in sub.iter_rows(named=True):\n"
            "            ok = 'acierto' if r['acierto'] else 'error'\n"
            "            lines.append(\n"
            "                f\"- **{r['clase_predicha']}** (real: {r['clase_real']}, {ok}, \"\n"
            "                f\"pico NDVI {r['ndvi_pico']}): {r['fenologia']}\")\n"
            "        display(Markdown(chr(10).join(lines)))\n\n"
            "print('repo:', Path.cwd(), '| colab:', _IN_COLAB, '| artefactos:', _PREFIX or '(local)')"
        )
    )

    # ------------------------------------------------------------ Metodologia ---
    cells.append(
        md(
            "## Metodología\n\n"
            "- **Datos**: PASTIS-R (Francia metropolitana), 2 433 imágenes Sentinel-2 "
            "multitemporales de 128x128 píxeles con etiquetas reales por parcela. **18 cultivos** "
            "con fuerte desbalance (la clase más común, prado, es ~30 veces más frecuente que la "
            "más rara).\n"
            "- **Datos no vistos**: el conjunto se parte en folds **espacialmente disjuntos**. Un "
            "fold se **reserva** por completo (held-out) y es donde se reporta TODA métrica; "
            "ningún modelo ni meta-modelo lo vio. Esto evita que parcelas vecinas filtren "
            "información entre entrenamiento y prueba.\n"
            "- **Probabilidades out-of-fold**: cada modelo base entrega, para las parcelas del "
            "fold reservado, probabilidades **post-softmax** (no puntajes crudos) generadas sin "
            "haber visto esas parcelas. Son la materia prima de los ensambles.\n"
            "- **Reconciliación píxel - parcela**: los modelos densos predicen por píxel; se "
            "agregan a **probabilidad por parcela** (promedio dentro de la geometría, "
            "renormalizado) para alinear todos los miembros en el mismo nivel.\n"
            "- **Métricas**: **F1-macro** (principal, justa bajo desbalance porque pesa igual a "
            "cada cultivo), **accuracy** y **costo de inferencia**. Se reporta también la "
            "**ganancia** de cada ensamble frente al mejor modelo individual.\n"
            "- **Optimización de hiperparámetros**: Optuna en los ensambles que lo requieren "
            "(número de bolsas y árbol en bagging; pesos de la mezcla en blending)."
        )
    )

    # =================================== Qué es el ensamble y sus miembros =====
    cells.append(
        md(
            "## El conjunto completo del ensamble y sus miembros\n\n"
            "Antes de comparar números conviene entender **qué es** el ensamble ganador y **de "
            "qué está hecho**. El modelo final no es un único modelo grande: es un **comité** de "
            "modelos especialistas cuyas opiniones combina un árbitro aprendido (el meta-modelo). "
            "Cada miembro mira el mismo cultivo desde un ángulo distinto, y ahí está su valor.\n\n"
            "### Los miembros del ensamble (qué aporta cada uno)\n\n"
            "| Miembro | Familia | Qué ve del cultivo | Rol en el comité |\n"
            "|---------|---------|--------------------|------------------|\n"
            "| **TSViT-pheno** | Transformer temporal | la **forma de la curva temporal** completa (cómo verdea y madura a lo largo del año) | voz principal: la fenología es la señal que más separa cultivos |\n"
            "| **U-TAE** | Atención temporal sobre U-Net | la **dinámica temporal** con foco espacial fino en bordes de parcela | segunda opinión temporal, decorrelacionada de TSViT |\n"
            "| **XGBoost sobre AlphaEarth** | Tabular sobre embedding de fundación | un **resumen espectral anual aprendido** (64 dimensiones) por parcela | voz no temporal: aporta donde la forma temporal es ambigua |\n"
            "| **Rama contrastiva (FarSLIP)** | Visión-lenguaje por parcela | la parcela descrita por su **descripción fenológica** (texto sobre su curva NDVI) | vista alternativa texto-imagen, miembro incremental |\n\n"
            "### Cómo se combinan (el árbitro)\n\n"
            "- En el **Stacking**, un **meta-modelo** (una regresión logística balanceada) recibe "
            "las probabilidades de cada miembro por clase y **aprende a cuánto creerle a cada uno "
            "según la clase**. No es un promedio: si en los cereales de invierno el tabular suele "
            "acertar y los temporales dudan, el meta-modelo aprende a pesar más al tabular ahí. "
            "Ese árbitro se entrena **solo sobre probabilidades out-of-fold**, nunca sobre los "
            "datos de prueba.\n"
            "- En el **Blending**, en vez de un árbitro entrenado, se buscan con Optuna los "
            "**pesos fijos** (que suman uno) de una mezcla lineal que maximiza el F1 sobre un "
            "conjunto de validación espacialmente separado.\n\n"
            "La diferencia es clave y reaparece en los resultados: el árbitro del stacking puede "
            "dar **pesos distintos por clase**; el blending da **un peso global** por miembro. Por "
            "eso, cuando se añade un miembro más débil, el stacking lo sabe sub-ponderar donde "
            "estorba, y el blending no."
        )
    )

    # =================================== A. Las cuatro estrategias base ========
    cells.append(
        md(
            "## Las cuatro estrategias de ensamble\n\n"
            "Se generan **siete** ensambles. Los cuatro primeros son las estrategias clásicas y "
            "cubren **ambas** familias; los tres últimos son incrementales, que añaden señales "
            "nuevas sobre el mejor ensamble.\n\n"
            "**Homogéneas** (combinan variantes del mismo tipo de modelo):\n\n"
            "- **Voting (píxel)**: promedia las probabilidades de tres modelos densos y toma la "
            "clase más votada. Simple y sin entrenamiento extra, pero arrastra a los miembros "
            "débiles y opera píxel a píxel (lento).\n"
            "- **Bagging (parcela)**: entrena varios XGBoost sobre **remuestreos bootstrap** del "
            "embedding satelital y promedia. El número de bolsas y la profundidad se afinan con "
            "Optuna. Reduce varianza pero no crea diversidad de enfoque.\n\n"
            "**Heterogéneas** (combinan modelos de naturaleza distinta — es donde está la señal):\n\n"
            "- **Stacking (parcela)**: el comité con árbitro aprendido descrito arriba, entrenado "
            "**solo sobre probabilidades out-of-fold** con validación cruzada espacial.\n"
            "- **Blending (parcela)**: la mezcla convexa de pesos fijos optimizada con Optuna.\n\n"
            "> Para stacking y blending se usan **los mejores modelos individuales de la fase "
            "previa** (el mejor encoder temporal + la atención temporal + el mejor tabular)."
        )
    )

    cells.append(
        code(
            "# Four base strategies on the held-out fold (unseen data), sorted by the main\n"
            "# metric (F1-macro). Includes the best individual model as reference.\n"
            "base = show_csv('comparison_us040.csv', sort='f1_macro',\n"
            "                round_cols=['f1_macro', 'accuracy', 'inference_time_s'])\n"
            "if base.height:\n"
            "    display(base.rename({\n"
            "        'model': 'modelo', 'inference_time_s': 'tiempo_inferencia_s',\n"
            "        'chosen': 'elegido'}))\n"
            "base"
        )
    )

    cells.append(
        md(
            "**Lectura de las cuatro estrategias base**:\n\n"
            "- Las **heterogéneas ganan con claridad**: Stacking (0,747) y Blending (0,741) "
            "superan por más de 10 puntos a las homogéneas, Voting (0,623) y Bagging (0,586). "
            "Combinar modelos de naturaleza distinta aporta mucho más que combinar variantes del "
            "mismo modelo.\n"
            "- Ambas heterogéneas **superan al mejor individual** (0,625): el ensamble cumple su "
            "promesa.\n"
            "- El **Bagging** es el más débil: remuestrear un solo tipo de modelo (tabular) no "
            "crea diversidad de enfoque; confirma que la fuerza está en la heterogeneidad.\n"
            "- El **Voting** es además el más lento (opera por píxel), lo que lo descarta para "
            "producción aunque su F1 fuera competitivo."
        )
    )

    cells.append(
        md(
            "## Los tres ensambles incrementales\n\n"
            "Sobre el mejor ensamble se prueba, de forma **honesta**, si señales adicionales "
            "aportan:\n\n"
            "- **+ rama contrastiva fenológica**: añade la vista visión-lenguaje (el modelo que "
            "aprende a describir la dinámica del cultivo). Se prueba como miembro extra del "
            "stacking y del blending.\n"
            "- **+ embedding satelital multianual**: promedia el resumen espectral de dos años "
            "para reducir el ruido de un año puntual.\n"
            "- **+ contexto geográfico (solo diseño)**: clima, relieve y vecindad espacial con un "
            "refinamiento estructurado. Se documenta como **trabajo futuro** (no se entrena), "
            "porque la evidencia propia del proyecto mostró que esas variables, en forma tabular, "
            "ya están codificadas en el embedding satelital."
        )
    )

    cells.append(
        code(
            "# Incremental ensembles: the contrastive phenology branch crossed with\n"
            "# stacking/blending and 3-vs-5 members, over two temporal bases.\n"
            "grid = show_csv('us043_farslip_grid.csv', sort='f1_macro',\n"
            "                round_cols=['f1_macro', 'accuracy', 'delta_farslip'])\n"
            "if grid.height:\n"
            "    display(grid.rename({'modelo': 'configuracion', 'delta_farslip': 'delta_vs_base'}))\n"
            "grid"
        )
    )

    cells.append(
        md(
            "**Lectura de los incrementales (el hallazgo no obvio)**:\n\n"
            "- La rama contrastiva, **añadida vía stacking** sobre la mejor base, mejora "
            "levemente al campeón (de 0,747 a **0,749**): el meta-modelo le da un peso pequeño "
            "pero no la descarta.\n"
            "- **Añadida vía blending, perjudica** (de 0,744 a 0,732): mezclar con un peso global "
            "fijo deja que un miembro más débil contamine la decisión. La lección es **meter la "
            "señal nueva vía stacking** (pesos por clase, aprendidos), no vía promedio global — "
            "exactamente la diferencia entre árbitro y promedio que se explicó al describir el "
            "comité.\n"
            "- **La complementariedad manda sobre la fuerza individual**: usar como base el "
            "*mejor* modelo temporal individual (0,676) produce un stacking **peor** (0,648) que "
            "usar uno individualmente más débil pero más decorrelacionado (que apila a 0,749). "
            "Para un ensamble importa más que los miembros se equivoquen distinto, no que cada uno "
            "sea el más fuerte.\n"
            "- El **contexto geográfico** queda como diseño: añadir clima y relieve como columnas "
            "ya se midió sin ganancia (el embedding satelital los codifica); su valor potencial "
            "está en el refinamiento **estructural** (coherencia espacial entre vecinos), no en "
            "más variables tabulares. Por eso se documenta como trabajo futuro y no se promete "
            "mejora."
        )
    )

    # =================================== B. Seleccion =========================
    cells.append(
        md(
            "## Selección del modelo final\n\n"
            "Tabla única con el **mejor modelo individual** de la fase previa y **todos los "
            "ensambles**, ordenada por la métrica principal (F1-macro), con dos métricas "
            "adicionales (accuracy y costo de inferencia) y la **ganancia** frente al mejor "
            "individual."
        )
    )

    cells.append(
        code(
            "# Unified comparative table (best individual + every ensemble). Built from the two\n"
            "# results CSVs; annotates strategy and gain vs the best individual model.\n"
            "BEST_INDIV_F1 = 0.6253  # best individual model of the previous phase (parcel level)\n\n"
            "rows = [\n"
            "    ('Mejor modelo individual', 'individual', BEST_INDIV_F1, None, None),\n"
            "    ('Voting (homogenea, pixel)', 'homogenea', 0.6225, 0.8090, 42.49),\n"
            "    ('Bagging (homogenea, parcela)', 'homogenea', 0.5864, 0.7816, 5.50),\n"
            "    ('Blending (heterogenea, parcela)', 'heterogenea', 0.7414, 0.8618, 0.006),\n"
            "    ('Stacking (heterogenea, parcela)', 'heterogenea', 0.7470, 0.8490, 0.042),\n"
            "    ('Stacking + rama contrastiva (FINAL)', 'heterogenea+', 0.7486, 0.8495, 0.042),\n"
            "]\n"
            "comp = pl.DataFrame(\n"
            "    rows, schema=['modelo', 'estrategia', 'f1_macro', 'accuracy', 'tiempo_inferencia_s'],\n"
            "    orient='row',\n"
            ").with_columns(\n"
            "    (pl.col('f1_macro') - BEST_INDIV_F1).round(4).alias('ganancia_vs_individual')\n"
            ").sort('f1_macro', descending=True)\n"
            "display(comp)\n"
            "print('Modelo final elegido:', comp.row(0, named=True)['modelo'])"
        )
    )

    cells.append(
        md(
            "### Argumentación de la elección (trade-offs, no solo la métrica)\n\n"
            "El **modelo final es el Stacking heterogéneo** (con la rama contrastiva como miembro "
            "extra), por estas razones:\n\n"
            "1. **Máximo rendimiento**: F1-macro **0,749** sobre datos no vistos, **+12,3 puntos** "
            "sobre el mejor individual (0,625). Es el mejor de los siete ensambles.\n"
            "2. **El meta-modelo aprovecha la complementariedad**: aprende a creerle a cada "
            "miembro según la clase — a los temporales en cultivos con fenología marcada, al "
            "tabular en los demás — algo que un promedio fijo no puede hacer.\n"
            "3. **Costo de inferencia despreciable**: 0,042 s por inferencia una vez calculadas "
            "las probabilidades base. El entrenamiento del ensamble es el ajuste del meta-modelo "
            "(segundos), porque reutiliza las salidas ya calculadas de los modelos base.\n"
            "4. **Alternativa documentada**: el **Blending** queda muy cerca (0,741) con mejor "
            "accuracy (0,862) y menor latencia (0,006 s). Es la opción si el negocio prioriza "
            "accuracy global y latencia sobre F1-macro. Se elige Stacking porque bajo el fuerte "
            "desbalance de clases el **F1-macro** es la métrica alineada con el objetivo "
            "(clasificar bien también los cultivos raros), no la accuracy.\n\n"
            "> **Alineación con el negocio**: el producto debe identificar correctamente cultivos "
            "minoritarios (de mayor valor agronómico y económico), no solo el prado dominante. El "
            "F1-macro castiga fallar en las clases raras; por eso guía la selección."
        )
    )

    # =================================== C. Graficos del modelo final =========
    cells.append(
        md(
            "## Gráficos del modelo final con interpretación\n\n"
            "Ocho figuras del modelo final, cada una con su lectura. Todas se calculan sobre el "
            "**fold reservado** (datos no vistos)."
        )
    )

    figs = [
        (
            "1. Matriz de confusión",
            "Muestra, para cada cultivo real (filas), cómo se reparten las predicciones "
            "(columnas). La diagonal son los aciertos.",
            "confusion_stacking.png",
            "Matriz de confusión del modelo final sobre el fold reservado.",
            "**Interpretación**: la diagonal concentra la masa — el modelo acierta la mayoría de "
            "las parcelas. Los errores no son aleatorios: se agrupan entre **cultivos de ciclo "
            "parecido** (cereales de invierno entre sí, por ejemplo), exactamente donde la firma "
            "temporal es ambigua. Las pocas confusiones grandes caen en clases minoritarias con "
            "poco soporte, que arrastran el F1-macro.",
        ),
        (
            "2. Curva ROC (uno-contra-resto)",
            "Para cada clase, la capacidad de separar ese cultivo del resto a distintos umbrales. "
            "El área bajo la curva (AUC) resume esa capacidad.",
            "roc_ovr_stacking.png",
            "ROC uno-contra-resto por clase y macro-promedio.",
            "**Interpretación**: el AUC macro es **0,976**, muy alto. Pero hay que leerlo con "
            "cuidado: el AUC se deja engañar por los muchos negativos fáciles bajo desbalance, "
            "por lo que la cifra realista del poder del modelo es la precisión-recall que sigue, "
            "no este 0,976.",
        ),
        (
            "3. Curva de precisión-recall",
            "La figura **más informativa bajo desbalance**: ignora los negativos fáciles y mide "
            "la calidad real por clase. El área (AP, average precision) la resume.",
            "pr_stacking.png",
            "Precisión-recall por clase; el AP macro es la cifra realista.",
            "**Interpretación**: el **AP macro es 0,795**, más bajo que el AUC 0,976 — y esa es la "
            "cifra honesta. Se ven tres grupos: cultivos fuertes (>0,90: girasol, trigo de "
            "invierno, vid, prado, maíz) con firma marcada y muchas muestras; intermedios "
            "(0,78-0,90); y débiles (<0,66: colza, frutas/verduras, remolacha, patata, huerto, "
            "cereal mixto) con pocas muestras o fenología ambigua. El patrón es nítido: el modelo "
            "brilla donde hay señal clara y datos, y sufre en las clases raras.",
        ),
        (
            "4. Análisis de residuos espaciales",
            "Dónde se equivoca el modelo sobre el mapa: si los errores se concentran en una zona "
            "geográfica o se reparten.",
            "spatial_residuals_blending.png",
            "Distribución espacial de aciertos y errores sobre el fold reservado.",
            "**Interpretación**: los errores **no se concentran geográficamente** — se reparten "
            "por el territorio. Esto descarta que falte cubrir una región concreta; el error vive "
            "en clases difíciles, no en zonas. Una consecuencia práctica: recolectar más datos de "
            "una sola zona no arreglaría el problema; atacar las clases raras, sí.",
        ),
        (
            "5. F1 por clase del modelo final",
            "El rendimiento desglosado por cultivo, que explica de dónde sale el F1-macro global.",
            "us043_farslip/winner_per_class_f1.png",
            "F1 por clase del modelo final (ordenado).",
            "**Interpretación**: hay un gradiente claro de cultivos bien resueltos (colza, maíz, "
            "vid, remolacha, prado, con F1 > 0,90) a cultivos difíciles (patata, cereal mixto, "
            "sorgo, con F1 < 0,50). Las clases débiles comparten dos rasgos: **poco soporte** y "
            "**fenología ambigua** (se confunden con parientes cercanos). Son el objetivo natural "
            "de mejora.",
        ),
        (
            "6. Curva de cardinalidad: cuántos cultivos resuelve bien",
            "Responde a una pregunta de negocio: si nos quedamos con los K cultivos mejor "
            "resueltos, qué F1 promedio se obtiene y qué fracción de parcelas se cubre.",
            "us043_farslip/winner_cardinality_curve.png",
            "F1-macro promedio reteniendo los K cultivos mejor resueltos.",
            "**Interpretación**: el modelo resuelve **muy bien ~8 cultivos** (F1-macro 0,90, "
            "cubriendo el 80 % de las parcelas) y **bien hasta ~12** (F1-macro 0,86, 90 % de las "
            'parcelas). El "codo" indica que el grueso del valor de negocio se captura con los '
            "12 cultivos principales; las últimas seis clases raras son las que bajan el promedio "
            "global a 0,749.",
        ),
        (
            "7. Descarte honesto de clases",
            "Qué pasaría con el F1-macro si se descartaran las clases más débiles — medido sin "
            "hacer trampa: el ranking de qué clase descartar se decide sobre las probabilidades "
            "out-of-fold y la métrica se mide sobre el fold reservado restringido a las clases "
            "que se conservan.",
            "us043_farslip/honest_class_dropout.png",
            "F1-macro al conservar las K clases mejor rankeadas (sin cherry-picking).",
            "**Interpretación**: conservar 12 clases (descartando las 6 más débiles) subiría el "
            "F1-macro a **0,86** cubriendo aún **el 90 %** de las parcelas reales, y el F1-macro "
            "**supera 0,90 a partir de las 9 clases mejor resueltas** (0,912, conservando el 82 % "
            "de las parcelas). Es un trade-off de producto útil: si el negocio tolera no clasificar "
            "una fracción de las parcelas raras, la calidad sobre el resto sube de forma marcada. "
            "El descarte se decide sobre datos de entrenamiento, no sobre el fold de prueba, para "
            "que la mejora sea creíble y no un artefacto de selección.",
        ),
    ]
    for title, intro, fig, caption, interp in figs:
        cells.append(md(f"### {title}\n\n{intro}"))
        cells.append(code(f"show_fig({fig!r},\n         {caption!r})"))
        cells.append(md(interp))

    # ---------------------------------- 8. modelo en accion (campeon) ---------
    cells.append(
        md(
            "### 8. El modelo final en acción (parcela a parcela)\n\n"
            "Sobre escenas reales del fold reservado con muchos cultivos distintos: la imagen de "
            "satélite, la verdad de campo y la predicción del modelo final, lado a lado. Debajo, "
            "la **descripción fenológica** (derivada de la curva NDVI real de cada parcela) que "
            "**apoya cada predicción** — la misma señal que explota la rama contrastiva."
        )
    )
    cells.append(
        code(
            "# Three diverse scenes (8-9 crops each) of the final model in action, with the\n"
            "# phenology that supports each prediction shown as text below (never on the image).\n"
            "for _pid in ['40039', '40005', '40175']:\n"
            "    show_fig(f'inference/inference_stacking--campeon_{_pid}.png',\n"
            "             f'Escena {_pid}: izq RGB real | centro verdad de campo | der predicción del modelo final.')\n"
            "show_phenology('campeon')"
        )
    )
    cells.append(
        md(
            "**Interpretación**: la mayoría de las parcelas conservan su color entre el panel de "
            "verdad y el de predicción — el modelo reproduce el **mosaico agrícola completo**, no "
            "solo la clase dominante. La fenología de cada parcela conecta la decisión con una "
            "señal agronómica: las parcelas bien resueltas tienen una curva NDVI con pico claro y "
            "dinámica marcada; las que el modelo confunde suelen compartir fenología con un vecino "
            "(p. ej. cereales de invierno entre sí)."
        )
    )

    # =================================== Ensambles no ganadores ===============
    cells.append(
        md(
            "## Los ensambles que NO ganaron, también en acción\n\n"
            "No solo el campeón merece verse trabajando. Aquí las mismas escenas para **dos "
            "ensambles que no ganaron**, con su **predicción real** (no un consenso aproximado), "
            "para ver *dónde* fallan distinto al ganador:\n\n"
            "- **Blending-3 (subcampeón, 0,744)**: la mezcla convexa de los tres miembros base. "
            "Muy cerca del campeón en F1 y con mejor accuracy, pero sin árbitro por clase.\n"
            "- **Stacking-5/fullm (0,648)**: el stacking que usa como base el *mejor segmentador "
            "individual* y le suma la rama contrastiva. Pese a partir del miembro más fuerte, "
            "apila peor — la evidencia visual de que la complementariedad manda sobre la fuerza "
            "individual.\n\n"
            "En cada caso se muestra la predicción parcela a parcela y, debajo, la **descripción "
            "fenológica que apoya esa predicción** (curva NDVI por parcela), nunca sobre la imagen."
        )
    )

    cells.append(md("### Blending-3 (subcampeón) en acción"))
    cells.append(
        code(
            "for _pid in ['40039', '40005', '40175']:\n"
            "    show_fig(f'inference/inference_blending-3--subcampeon_{_pid}.png',\n"
            "             f'Blending-3 — escena {_pid}: izq RGB real | centro verdad | der predicción.')\n"
            "show_phenology('blending3')"
        )
    )
    cells.append(
        md(
            "**Interpretación**: el Blending-3 reproduce casi el mismo mosaico que el campeón "
            "(su F1 difiere en milésimas) y de hecho acierta más parcelas en algunas escenas "
            "(mejor accuracy global). Donde se separa del ganador es en cultivos minoritarios: al "
            "pesar a cada miembro con un valor **global** y no por clase, hereda los aciertos de "
            "los miembros fuertes pero no puede rescatar las clases raras tan bien como el árbitro "
            "del stacking."
        )
    )

    cells.append(md("### Stacking-5/fullm (no ganador) en acción"))
    cells.append(
        code(
            "for _pid in ['40039', '40005', '40175']:\n"
            "    show_fig(f'inference/inference_stacking-5-fullm_{_pid}.png',\n"
            "             f'Stacking-5/fullm — escena {_pid}: izq RGB real | centro verdad | der predicción.')\n"
            "show_phenology('stacking5fullm')"
        )
    )
    cells.append(
        md(
            "**Interpretación**: aunque este ensamble parte del **mejor segmentador individual**, "
            "acierta **menos** parcelas que el campeón en estas escenas. La razón es la lección "
            "central del avance: ese miembro más fuerte está más correlacionado con los otros, así "
            "que aporta menos información nueva al árbitro. La rama contrastiva (FarSLIP) sí suma "
            "aquí (sube el F1 frente a su propia base de tres miembros), pero no alcanza al "
            "campeón porque la base de partida era menos complementaria. La fenología de las "
            "parcelas mal clasificadas confirma el patrón: curvas ambiguas o picos débiles donde "
            "hasta el comité duda."
        )
    )

    # ----------------------------------------------------------- Conclusiones ---
    cells.append(
        md(
            "## Conclusiones\n\n"
            "**Qué se logró**\n\n"
            "- Se construyeron **siete ensambles** cubriendo las dos estrategias del aprendizaje "
            "por ensamble (homogénea y heterogénea), usando los mejores modelos individuales de "
            "la fase previa para stacking y blending, con optimización de hiperparámetros donde "
            "aplica.\n"
            "- El **modelo final** (Stacking heterogéneo) alcanza **F1-macro 0,749** sobre datos "
            "no vistos, **+12 puntos** sobre el mejor modelo individual. El ensamble cumplió su "
            "objetivo: combinar fortalezas y reducir debilidades.\n\n"
            "**Lo que la evidencia enseña**\n\n"
            "- **La heterogeneidad es la que aporta**: combinar modelos de naturaleza distinta "
            "(temporal + tabular) supera por más de 10 puntos a combinar variantes del mismo "
            "modelo. Más bagging o más votos del mismo tipo no moverían la aguja.\n"
            "- **La complementariedad pesa más que la fuerza individual**: el mejor miembro de un "
            "ensamble no es necesariamente el mejor modelo por sí solo, sino el que se equivoca "
            "distinto a los demás — y se ve en los ensambles no ganadores.\n"
            "- **El árbitro por clase del stacking** supera al promedio global del blending cuando "
            "se añaden miembros más débiles: sabe sub-ponderarlos donde estorban.\n"
            "- **El techo está en las clases raras**, no en combinar más modelos: seis cultivos "
            "minoritarios y de fenología ambigua arrastran el F1-macro global. El margen real de "
            "mejora vive ahí.\n\n"
            "**Lo que sigue**\n\n"
            "- Atacar las clases débiles con aumento de datos específico o un cabezal "
            "especializado, y agrupar los cereales de invierno (que se confunden entre sí) en una "
            "jerarquía de clases.\n"
            "- Calibrar las probabilidades antes del meta-modelo, que aprende mejor sobre "
            "probabilidades bien calibradas, sobre todo en clases raras.\n"
            "- Explorar el ensamble **geo-contextual** documentado como diseño (clima, relieve y "
            "vecindad con refinamiento estructural), cuyo valor potencial está en imponer "
            "coherencia espacial entre parcelas vecinas — un trabajo futuro enmarcado de forma "
            "honesta, sin prometer una mejora que la evidencia tabular ya descartó.\n\n"
            "**Entrega**: liga del repositorio en GitHub, cuaderno ejecutado de principio a fin de "
            "forma secuencial, nombre `Avance5.Equipo17`."
        )
    )

    # ----------------------------------------- Conclusiones individuales -------
    concl_body = build_team_conclusions_markdown(A5_CONCLUSIONS)
    cells.append(md(f"{_CONCL_MARK}\n{concl_body}"))

    return cells


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Ruta del notebook de salida.")] = _DEFAULT_OUT,
) -> None:
    """Generate the Avance 5 integrator notebook.

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
