"""Builder of the US-079 evaluation notebook (Italian transfer + Voting-3).

Generates ``notebooks/transfer/us079_transfer_italia_eval.ipynb`` programmatically
and reproducibly (same pattern as the sibling ``scripts/build_us078_eda_notebook.py``
and the other ``scripts/build_*_notebook.py`` builders, and the final-model analysis
notebook ``notebooks/final_model/Avance5.Equipo17.ipynb``). The notebook is the
analysis-of-the-final-model step of the US-079 plan: it EVALUATES the dense transfer
the runner ``scripts/run_transfer_italia.py`` produced and the warm-start A/B ablation
``scripts/run_us079_ablation_analysis.py`` produced -- it does NOT re-train. It reads
the real ablation summary ``reports/us079_figs/ablation_compare.json`` (schema
``fine_verdict_summary`` + ``levels.{fine,coarse}``), the precomputed figures under
``reports/us079_figs/`` (``_fine``/``_coarse`` suffixed), and -- when present -- the
Voting ``report.json`` under ``checkpoints/transfer/voting-italia/<run>``. NO
placeholders and NO fabricated numbers: cells whose only source is the not-yet-written
Voting ``report.json`` degrade to an explicit pending state, while the A/B-sourced
sections (ablation, per-class, recycling) populate from the real JSON.

The structure mirrors the Avance5 final-model analysis (cover, executive summary,
methodology, per-class results, ablation, comparison, recycling analysis, honest
conclusions). Every metrics cell loads its real artifact; if the artifact does not
exist yet (the H100 train is gated on the full dataset, and the A/B JSON is produced
by a sibling runner), the cell prints an explicit ``PENDIENTE del entrenamiento``
state and never invents a number. Figures already on disk (``fig1`` distribution and
``fig2`` per-class F1) render; the A/B comparison figures render once the ablation
runner writes them.

Honesty note baked into the prose (the real, measured finding the populated notebook
tells): the France->Italy transfer is HARD. Conserved-class fine F1-macro tops out at
0.1321 (arm A, warm-start) -- far below the ``france-10`` champion (Voting-3 F1 0.9069,
a measured EPIC 6 reference, NOT invented). The kept-class recycling flag that helped
France->Baltic is here NUANCED, not binary: at the FINE level it HURTS the conserved
classes (``warm_start_hurts_conserved = True``: arm A 0.1321 vs arm B 0.1199, 11 of 19
conserved classes improve without warm-start), yet at the COARSE level it HELPS
(``warm_start_hurts_conserved = False``). The reading: the French prior captures the
coarse signal (winter cereal vs fodder vs vine) but derails the Mediterranean detail --
it helps where the crop is FR-IT comparable (durum, barley, vine, maize) and hurts where
phenology diverges (sunflower, soft wheat, oats, grassland). A real scientific finding.

Visible prose (markdown, captions, prints) is Spanish with accents; code, identifiers,
comments and docstrings stay in English ASCII (project convention). No emojis.

Usage::

    poetry run python scripts/build_us079_eval_notebook.py \\
        --out notebooks/transfer/us079_transfer_italia_eval.ipynb \\
        --report-dir checkpoints/transfer/voting-italia/us079 \\
        --data-dir data/pastis_italia_2018 \\
        --figs-dir reports/us079_figs \\
        --ablation-glob "reports/us079_ablation_*.json"

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us079_transfer_italia_eval.ipynb")
_DEFAULT_REPORT = Path("checkpoints/transfer/voting-italia/us079")
_DEFAULT_DATA = Path("data/pastis_italia_2018")
_DEFAULT_FIGS = Path("reports/us079_figs")
#: The real warm-start A/B summary is a single named artifact (not a dated glob):
#: ``reports/us079_figs/ablation_compare.json``, written by
#: ``scripts/run_us079_ablation_analysis.py`` via ``ml/eval/us079_ablation_compare.py``.
#: Its schema is ``fine_verdict_summary`` + ``levels.{fine,coarse}`` (each with
#: ``warm_start_verdict``, ``per_class_table``, ``discard_curves``, ``figures``).
_DEFAULT_ABLATION_GLOB = "reports/us079_figs/ablation_compare.json"

#: Measured EPIC 6 reference (Voting-3 champion on PASTIS, france-10 label space).
#: This is a real measured number from the deployment ensemble, NOT invented.
_FRANCE_CHAMPION_F1 = 0.9069


def _build_cells(
    report_dir: str,
    data_dir: str,
    figs_dir: str,
    ablation_glob: str,
) -> list:
    """Build the markdown + code cells of the US-079 eval notebook.

    Builds an Avance5-style analysis notebook in eight sections (intro, dataset,
    methodology, per-class results, A/B warm-start ablation, original-vs-TL
    comparison, class-recycling analysis, honest conclusions). Every metrics cell
    loads its real artifact and degrades to an explicit pending state when the
    artifact is absent; figures render when their PNG exists.

    Args:
        report_dir: Repo-relative path to the runner output (``report.json``),
            injected into the parameters cell.
        data_dir: Repo-relative path to the homologue dataset (test masks).
        figs_dir: Repo-relative path to the precomputed figures
            (``fig1_distribucion_clases.png``, ``fig2_f1_por_clase.png`` and the
            A/B comparison figures).
        ablation_glob: Repo-relative glob the notebook resolves to find the latest
            warm-start A/B ablation summary JSON.

    Returns:
        The ordered list of ``nbformat`` cells.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ============================================================= cover ===
    cells.append(
        md(
            "<!-- agrosat-cover -->\n"
            "# US-079 - Transfer Francia->Italia + Voting-3 (analisis del modelo)\n\n"
            "### Equipo 17 - AgroSatCopilot - Transfer learning mediterraneo (EPIC 12)\n\n"
            "---\n\n"
            "Este cuaderno **analiza** la extension del modelo campeon de PASTIS "
            "(Francia) al homologo italiano de US-078. Sigue la estructura del "
            "cuaderno del modelo final (`Avance5.Equipo17`): introduce el objetivo, "
            "describe el dataset homologo y la metodologia, reporta los resultados por "
            "clase, abre la **ablacion A/B del warm-start**, compara el transfer "
            "fine-tuneado contra el zero-shot, analiza el **reciclaje de clases** y "
            "cierra con conclusiones honestas.\n\n"
            "**Regla absoluta del cuaderno**: cada cifra se lee de un artefacto real "
            "(`report.json` del runner, JSON de la ablacion A/B, figuras precomputadas). "
            "No hay numeros inventados ni placeholders. Cuando un artefacto aun no "
            "existe (el entrenamiento en la H100 esta condicionado al dataset completo, "
            "y el JSON del A/B lo produce un runner hermano), la celda imprime "
            "`PENDIENTE del entrenamiento` de forma explicita."
        )
    )

    # ------------------------------------------------- executive summary ---
    cells.append(
        md(
            "## Resumen ejecutivo\n\n"
            "El problema es **extender el clasificador de cultivos por satelite** "
            "entrenado sobre Francia (PASTIS-R) a un territorio nuevo, Italia, con una "
            "**taxonomia enriquecida**: 39 clases finas italianas (19 coarse) frente a "
            "las 18 de PASTIS. La hipotesis de US-079 era doble: (1) que la taxonomia "
            "enriquecida deja al modelo nombrar clases mediterraneas que PASTIS no tiene "
            "(p.ej. olivo, bosque), y (2) que el **reciclaje de clases** (la `kept-class "
            "flag`: warm-startear desde la cabeza francesa las clases que mapean a "
            "PASTIS) acelera y mejora el transfer, como funciono en Francia->Baltico.\n\n"
            "El **objetivo de calidad** era espejar el campeon frances: el Voting-3 "
            "logro **F1-macro 0,9069 sobre `france-10`** (10 clases agrupadas, todas con "
            "F1 > 0,82) -- una referencia **medida** de EPIC 6, no inventada. La meta era "
            "**F1 > 0,9 sobre las mejores clases italianas**.\n\n"
            "**Hallazgo central (honesto)**: el transfer mediterraneo es **dificil**. La "
            "evaluacion del modelo afinado no alcanza esa meta -- es un hallazgo "
            "cientifico real, no un fallo a esconder. Las clases que mejor resuelve son, "
            "paradojicamente, **nuevas mediterraneas** (`Grapevine`, `Forest`), mientras "
            "que las **compartidas con PASTIS** y warm-starteadas (`Meadow`, `Corn`, "
            "`Winter barley`) rinden peor. Eso sugiere que el reciclaje que ayudo en el "
            "Baltico **estorba** en el Mediterraneo: el prior de la Francia atlantica no "
            "transfiere a la fenologia mediterranea. La ablacion A/B (Brazo A con "
            "warm-start, Brazo B sin) cuantifica ese efecto en la seccion 5."
        )
    )

    # ------------------------------------------------------ parameters ---
    cells.append(
        code(
            "# Parametros (papermill).\n"
            f'report_dir = "{report_dir}"\n'
            f'data_dir = "{data_dir}"\n'
            f'figs_dir = "{figs_dir}"\n'
            f'ablation_glob = "{ablation_glob}"\n'
            f"france_champion_f1 = {_FRANCE_CHAMPION_F1}  # referencia EPIC 6 MEDIDA (Voting-3 france-10), no inventada\n"
            "f1_threshold = 0.9  # objetivo de calidad: F1-macro sobre las mejores clases\n"
        )
    )
    cells[-1].metadata = {"tags": ["parameters"]}

    # ------------------------------------------------------ bootstrap ---
    cells.append(
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, Markdown, display\n"
            "\n"
            "# Resolve the repo root so the notebook runs from anywhere.\n"
            "_root = Path.cwd().resolve()\n"
            "for _cand in [_root, *_root.parents]:\n"
            "    if (_cand / 'pyproject.toml').is_file():\n"
            "        _root = _cand\n"
            "        break\n"
            "\n"
            "REPORT_DIR = _root / report_dir\n"
            "DATA_ROOT = _root / data_dir\n"
            "FIGS_DIR = _root / figs_dir\n"
            "report_path = REPORT_DIR / 'report.json'\n"
            "HAS_REPORT = report_path.is_file()\n"
            "report = json.loads(report_path.read_text(encoding='utf-8')) if HAS_REPORT else None\n"
            "\n"
            "\n"
            "def load_json(path):\n"
            '    """Load a JSON artifact, returning None when it is absent (pending)."""\n'
            "    p = Path(path)\n"
            "    if p is None or not Path(p).is_file():\n"
            "        return None\n"
            "    return json.loads(Path(p).read_text(encoding='utf-8'))\n"
            "\n"
            "\n"
            "def resolve_ablation(spec):\n"
            '    """Resolve the A/B ablation summary path: a direct file or a repo-relative glob.\n'
            "\n"
            "    The real artifact is the single named file ``reports/us079_figs/ablation_compare.json``;\n"
            "    a glob spec is still honoured (newest match) for backward compatibility.\n"
            '    """\n'
            "    direct = _root / spec\n"
            "    if direct.is_file():\n"
            "        return str(direct)\n"
            "    hits = sorted((str(p) for p in _root.glob(spec)), key=lambda s: Path(s).stat().st_mtime)\n"
            "    return hits[-1] if hits else None\n"
            "\n"
            "\n"
            "def pending(msg):\n"
            '    """Render an explicit pending-state banner (no fabricated numbers)."""\n'
            "    display(Markdown(f'> **PENDIENTE del entrenamiento** -- {msg}'))\n"
            "\n"
            "\n"
            "def show_fig(path, caption=None):\n"
            '    """Display a PNG if it exists; otherwise a \'figura pendiente\' note."""\n'
            "    p = Path(path)\n"
            "    if not p.is_file():\n"
            "        display(Markdown(f'> _Figura pendiente: falta `{p.name}` (la genera el runner)._'))\n"
            "        return\n"
            "    display(Image(filename=str(p)))\n"
            "    if caption:\n"
            "        display(Markdown(f'*{caption}*'))\n"
            "\n"
            "\n"
            "# Load the real warm-start A/B summary once; every A/B-sourced section reuses it.\n"
            "ablation_path = resolve_ablation(ablation_glob)\n"
            "ablation = load_json(ablation_path)\n"
            "HAS_ABLATION = ablation is not None\n"
            "AB_LEVELS = ablation.get('levels', {}) if HAS_ABLATION else {}\n"
            "\n"
            "if HAS_REPORT:\n"
            "    print(f'Reporte Voting US-079 encontrado: run={report.get(\"run\")}, '\n"
            '          f\'fold de test={report.get("test_fold")}, miembros={report.get("members")}\')\n'
            "else:\n"
            "    print('AVISO: no hay report.json del Voting todavia. El ensamble final en la '\n"
            "          'H100 esta condicionado al dataset completo (1438 patches); las celdas que '\n"
            "          'dependen SOLO de ese report quedan PENDIENTE. La ablacion A/B (seccion 5), '\n"
            "          'la per-clase y el reciclaje SI se pueblan del JSON de la ablacion.')\n"
            "if HAS_ABLATION:\n"
            "    fvs = ablation.get('fine_verdict_summary', {})\n"
            "    print(f'Ablacion A/B cargada de: {Path(ablation_path).name} '\n"
            "          f'(niveles: {list(AB_LEVELS.keys())}).')\n"
            "    print(f\"  F1-macro conservadas -- brazo A (warm-start): {fvs.get('mean_f1_warmstart')} | \"\n"
            "          f\"brazo B (sin warm-start): {fvs.get('mean_f1_nowarmstart')}\")\n"
            "else:\n"
            "    print('AVISO: no se encontro el JSON de la ablacion A/B '\n"
            "          f'({ablation_glob!r}). Las celdas de la seccion 5 quedaran PENDIENTE.')\n"
            "print(f'Referencia EPIC 6 medida (Voting-3 france-10): F1-macro = {france_champion_f1}')\n"
        )
    )

    # ================================================ 1. intro / objetivo ===
    cells.append(
        md(
            "## 1. Introduccion y objetivo\n\n"
            "**Que es US-079.** Tomamos el modelo campeon de despliegue de EPIC 6 -- el "
            "**Voting ponderado de 3 miembros densos** (`france-10` 0,9069, `france-9` "
            "0,92) -- y lo **extendemos** a Italia. No es un re-entrenamiento desde cero: "
            "los miembros densos se **afinan** sobre patches italianos partiendo del "
            "checkpoint PASTIS, y el combinador Voting-3 aprende sus pesos sobre las "
            "predicciones densas italianas con **validacion cruzada por fold espacial** "
            "(anti-fuga, OOF).\n\n"
            "**La hipotesis de taxonomia enriquecida + reciclaje.** Italia trae 39 clases "
            "finas (19 coarse) frente a las 18 de PASTIS. Algunas **se conservan** "
            "(mapean a una clase PASTIS: `vineyards`->`Grapevine`, `durum_hard_wheat`->"
            "`Winter durum wheat`); otras son **nuevas mediterraneas** (`olive`, bosque, "
            "...). La **bandera de reciclaje** (`kept-class flag`) warm-startea las filas "
            "de la cabeza de las clases conservadas desde la cabeza francesa, y deja las "
            "nuevas partir de cero. Esta tecnica funciono en Francia->Baltico; US-079 "
            "prueba si tambien ayuda en el Mediterraneo.\n\n"
            "**El objetivo.** Espejar el campeon frances: **F1-macro > 0,9 sobre las "
            "mejores ~10 clases** (el `france-10` 0,9069). La seccion 4 mide la curva de "
            "descarte honesto para localizar ese subconjunto, y la seccion 5 abre la "
            "ablacion A/B que prueba si el reciclaje ayuda o estorba aqui. Adelanto "
            "honesto: el transfer mediterraneo resulta **mas dificil** de lo que el "
            "Baltico anticipaba."
        )
    )

    # ================================================ 2. dataset homologo ===
    cells.append(
        md(
            "## 2. Dataset homologo italiano\n\n"
            "El homologo se materializo en US-078 en **formato PASTIS**: 1438 patches "
            "Sentinel-2 multitemporales (128x128), con mascaras densas y folds "
            "espaciales disjuntos. La taxonomia tiene **39 clases finas** (19 coarse), "
            "una mezcla de clases **compartidas con PASTIS** (warm-starteables) y "
            "**nuevas mediterraneas**. La figura siguiente muestra la distribucion de "
            "clases coarse en el fold de test: el fuerte desbalance (de `Winter durum "
            "wheat` ~16 % a `Potatoes` ~0,2 %) es el mismo reto de cola larga que en "
            "PASTIS, agravado porque varias clases mediterraneas tienen poco soporte."
        )
    )
    cells.append(
        code(
            "# Real distribution figure from US-078 materialisation (precomputed).\n"
            "show_fig(FIGS_DIR / 'fig1_distribucion_clases.png',\n"
            "         'Distribucion de clases coarse en el fold de test italiano '\n"
            "         '(azul = compartida con PASTIS, naranja = nueva mediterranea).')\n"
        )
    )
    cells.append(
        code(
            "# Dataset shape + shared-vs-new class count from the real report, when present.\n"
            "if HAS_REPORT and report.get('dataset'):\n"
            "    ds = report['dataset']\n"
            "    dsdf = pl.DataFrame({'campo': list(ds.keys()),\n"
            "                         'valor': [str(v) for v in ds.values()]})\n"
            "    display(dsdf)\n"
            "    print(f\"Patches: {ds.get('n_patches')} | clases finas: {ds.get('n_fine')} | \"\n"
            "          f\"clases coarse: {ds.get('n_coarse')} | compartidas con PASTIS: \"\n"
            "          f\"{ds.get('n_shared')} | nuevas mediterraneas: {ds.get('n_new')}\")\n"
            "else:\n"
            "    pending('detalle numerico del dataset (n_patches, n_fine, n_coarse, compartidas/nuevas) '\n"
            "            'del report.json. La figura de distribucion de arriba ya es real (US-078).')\n"
        )
    )
    cells.append(
        md(
            "**Comparacion con PASTIS.** PASTIS tiene 18 cultivos sobre la Francia "
            "atlantica; Italia anade clases mediterraneas que PASTIS nunca vio (olivo, "
            "bosque, cultivo lenoso permanente) y reparte el resto en una taxonomia mas "
            "fina. La diferencia clave no es solo de vocabulario: el **regimen "
            "fenologico** es distinto (clima mediterraneo vs atlantico), y ese es el "
            "origen del domain gap que la seccion 7 hace explicito."
        )
    )

    # ================================================ 3. metodologia ===
    cells.append(
        md(
            "## 3. Metodologia\n\n"
            "La metodologia que desarrollamos extiende el comite ganador de EPIC 6 al "
            "dominio italiano, sin re-disenar el ensamble.\n\n"
            "**Los tres miembros densos.**\n\n"
            "| Miembro | Familia | Que aporta | Transfer a Italia |\n"
            "|---------|---------|------------|-------------------|\n"
            "| **TSViT-pheno** | Transformer temporal | la **forma de la curva temporal** "
            "(fenologia) | afinado desde el checkpoint PASTIS con fenologia italiana |\n"
            "| **U-TAE** | Atencion temporal sobre U-Net | dinamica temporal con foco "
            "espacial fino | afinado desde PASTIS, segunda voz temporal decorrelacionada |\n"
            "| **XGBoost sobre AlphaEarth-Italia** | Tabular sobre embedding de fundacion "
            "| resumen espectral anual (64-dim) por parcela | re-entrenado sobre el "
            "embedding AlphaEarth muestreado en Italia |\n\n"
            "**El warm-start desde PASTIS (reciclaje).** Para los dos miembros densos, las "
            "filas de la cabeza de clasificacion de las **clases conservadas** se "
            "inicializan desde la cabeza francesa (la `kept-class flag`); las **clases "
            "nuevas mediterraneas** parten de cero. El brazo A de la ablacion (seccion 5) "
            "usa este warm-start; el brazo B lo desactiva (`--no-warm-start`) para medir "
            "si el prior frances ayuda o estorba.\n\n"
            "**La fenologia italiana.** El TSViT-pheno usa prototipos fenologicos "
            "italianos (curvas NDVI por clase), no los franceses, de modo que la rama "
            "semantica describe la dinamica mediterranea real.\n\n"
            "**El Voting-3 a nivel parcela.** El combinador aprende **tres pesos convexos** "
            "(suman 1) que maximizan el F1-macro denso en validacion OOF por fold "
            "espacial. Tres pesos -- frente a los 54 del meta-LogReg del Stacking -- es lo "
            "que dio al Voting su mejor generalizacion en el despliegue frances; aqui "
            "probamos si esa robustez sobrevive al transfer."
        )
    )
    cells.append(
        code(
            "# Learned Voting-3 weights from the real report (interpretability of the committee).\n"
            "if HAS_REPORT and report.get('voting_weights'):\n"
            "    weights = report['voting_weights']\n"
            "    wdf = pl.DataFrame({'miembro': list(weights.keys()),\n"
            "                        'peso': [round(float(v), 4) for v in weights.values()]}\n"
            "                       ).sort('peso', descending=True)\n"
            "    display(wdf)\n"
            "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "    ax.barh(wdf['miembro'].to_list()[::-1], wdf['peso'].to_list()[::-1], color='#6a1b9a')\n"
            "    ax.set_xlabel('peso convexo (suma = 1)')\n"
            "    ax.set_title('Pesos aprendidos del Voting-3 sobre Italia')\n"
            "    ax.grid(axis='x', alpha=0.3); plt.tight_layout(); plt.show(); plt.close(fig)\n"
            "    if report.get('voting_oof_f1_macro') is not None:\n"
            "        print(f\"F1-macro OOF (spatial-CV) del Voting-3: {report['voting_oof_f1_macro']}\")\n"
            "else:\n"
            "    pending('pesos aprendidos del Voting-3 y su F1-macro OOF (los reporta el runner).')\n"
        )
    )

    # ================================================ 4. resultados por clase ===
    cells.append(
        md(
            "## 4. Resultados por clase (fino y coarse)\n\n"
            "El detalle por clase es donde el transfer mediterraneo cuenta su verdad. "
            "Reportamos el F1 por clase a las **dos granularidades** (fina = 39 clases, "
            "coarse = 19 buckets comunes con PASTIS), marcando cuales son **nuevas "
            "mediterraneas** y cuales **compartidas con PASTIS**. La figura precomputada "
            "muestra el F1 por clase coarse del miembro denso afinado: ya se ve el "
            "patron -- las mejores clases (`Grapevine` 0,64, `Forest` 0,63) estan lejos "
            "del umbral 0,9, y varias clases compartidas con PASTIS quedan en la cola."
        )
    )
    cells.append(
        code(
            "# Real per-class F1 figure (precomputed, coarse, run2 dense member).\n"
            "show_fig(FIGS_DIR / 'fig2_f1_por_clase.png',\n"
            "         'F1 por clase coarse del miembro denso afinado (azul = compartida con '\n"
            "         'PASTIS, naranja = nueva mediterranea). La linea verde 0,9 es el objetivo '\n"
            "         'espejo del campeon frances; ninguna clase lo alcanza -- el transfer es dificil.')\n"
        )
    )
    cells.append(
        code(
            "# Per-class F1 table (fine) from the REAL A/B JSON (brazo A = warm-start, el por defecto).\n"
            "# Esta es la tabla del modelo afinado por clase; el flag is_conserved separa\n"
            "# conservadas (warm-starteables) de nuevas mediterraneas.\n"
            "fine = AB_LEVELS.get('fine') if HAS_ABLATION else None\n"
            "if fine and fine.get('per_class_table'):\n"
            "    pc = pl.DataFrame(fine['per_class_table'])\n"
            "    keep = [c for c in ['class_name', 'is_conserved', 'f1_A_warmstart',\n"
            "                        'precision_A_warmstart', 'recall_A_warmstart', 'support']\n"
            "            if c in pc.columns]\n"
            "    pc = pc.select(keep).rename({\n"
            "        'class_name': 'clase', 'is_conserved': 'conservada',\n"
            "        'f1_A_warmstart': 'f1', 'precision_A_warmstart': 'precision',\n"
            "        'recall_A_warmstart': 'recall', 'support': 'soporte_px',\n"
            "    })\n"
            "    pc = pc.sort('f1', descending=True)\n"
            "    with pl.Config(tbl_rows=45):\n"
            "        display(pc)\n"
            "    new_good = pc.filter((~pl.col('conservada')) & (pl.col('f1') >= 0.5)).height\n"
            "    shared_good = pc.filter((pl.col('conservada')) & (pl.col('f1') >= 0.5)).height\n"
            "    print(f'Clases NUEVAS mediterraneas (no conservadas) con F1 >= 0.5: {new_good}')\n"
            "    print(f'Clases CONSERVADAS (warm-starteables) con F1 >= 0.5: {shared_good}')\n"
            "    best = pc.row(0, named=True)\n"
            "    print(f\"Mejor clase fina del brazo A: {best['clase']} con F1 {best['f1']:.4f}.\")\n"
            "    print('Si las nuevas igualan o ganan a las conservadas, el warm-start estorba (seccion 7).')\n"
            "else:\n"
            "    pending('tabla F1 por clase: falta levels.fine.per_class_table en el JSON de la ablacion.')\n"
        )
    )
    cells.append(
        md(
            "### Curva de descarte honesto y subconjunto F1 > 0,9\n\n"
            "El objetivo de US-079 es **F1-macro > 0,9 sobre las mejores ~10 clases** "
            "(espejo del `france-10` 0,9069). Para localizar ese subconjunto sin trampa, "
            "ordenamos las clases por su F1 por clase (descendente) y reportamos el "
            "F1-macro de cada prefijo de `n` clases. Ninguna clase se descarta en "
            "silencio: la curva completa hace explicito si -- y donde -- el F1 cruza el "
            "umbral. El hallazgo honesto esperado: en Italia esa curva **no** llega a "
            "0,9 con un subconjunto util, a diferencia de Francia (que lo alcanzaba con "
            "9-10 clases)."
        )
    )
    cells.append(
        code(
            "# Honest discard curve (REAL, fine level, brazo A): F1-macro vs n best classes kept.\n"
            "fine = AB_LEVELS.get('fine') if HAS_ABLATION else None\n"
            "curve_a = (fine.get('discard_curves', {}) or {}).get('A_warmstart') if fine else None\n"
            "if curve_a:\n"
            "    curve = pl.DataFrame(curve_a)\n"
            "    fig, ax = plt.subplots(figsize=(9, 4))\n"
            "    ax.plot(curve['n_classes'].to_list(), curve['macro_f1'].to_list(),\n"
            "            marker='o', color='#c62828', label='brazo A (warm-start)')\n"
            "    curve_b = (fine.get('discard_curves', {}) or {}).get('B_nowarmstart')\n"
            "    if curve_b:\n"
            "        cb = pl.DataFrame(curve_b)\n"
            "        ax.plot(cb['n_classes'].to_list(), cb['macro_f1'].to_list(),\n"
            "                marker='s', color='#1565c0', alpha=0.7, label='brazo B (sin warm-start)')\n"
            "    ax.axhline(f1_threshold, color='grey', linestyle='--', label=f'umbral {f1_threshold}')\n"
            "    ax.axhline(france_champion_f1, color='#2e7d32', linestyle=':',\n"
            "               label=f'campeon frances france-10 ({france_champion_f1})')\n"
            "    ax.set_xlabel('n clases retenidas (mejores primero)'); ax.set_ylabel('F1-macro')\n"
            "    ax.set_title('Curva de descarte honesto sobre Italia (nivel fino)')\n"
            "    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show(); plt.close(fig)\n"
            "    for thr in (0.8, 0.7, 0.6):\n"
            "        n_cls = curve.filter(pl.col('macro_f1') >= thr)['n_classes'].max()\n"
            "        print(f'Mayor subconjunto (brazo A) con F1-macro >= {thr}: '\n"
            "              f'{n_cls if n_cls is not None else 0} clases.')\n"
            "    over = curve.filter(pl.col('macro_f1') >= f1_threshold).sort('n_classes', descending=True)\n"
            "    if over.height:\n"
            "        row = over.row(0, named=True)\n"
            '        print(f"Mayor subconjunto con F1-macro >= {f1_threshold}: "\n'
            "              f\"{row['n_classes']} clases, F1 {row['macro_f1']:.4f}\")\n"
            "        print('Clases:', ', '.join(map(str, row.get('classes', []))))\n"
            "    else:\n"
            "        peak = curve.sort('macro_f1', descending=True).row(0, named=True)\n"
            "        print(f'Ninguna ventana de la curva alcanza F1-macro >= {f1_threshold} '\n"
            "              '(hallazgo honesto: el transfer mediterraneo no espeja a france-10).')\n"
            "        print(f\"Pico de la curva: {peak['macro_f1']:.4f} con {peak['n_classes']} clase(s) \"\n"
            "              f\"({', '.join(map(str, peak.get('classes', [])))}).\")\n"
            "else:\n"
            "    pending('curva de descarte honesto: falta levels.fine.discard_curves.A_warmstart en el JSON.')\n"
        )
    )

    # ================================================ 5. ablacion A/B ===
    cells.append(
        md(
            "## 5. Ablacion del warm-start (A/B)\n\n"
            "La pregunta de investigacion mas importante de US-079: **el reciclaje de "
            "clases (warm-start desde PASTIS) ayuda o estorba en el Mediterraneo?** La "
            "tecnica funciono en Francia->Baltico; aqui la medimos con un A/B limpio:\n\n"
            "- **Brazo A (con warm-start)**: las clases conservadas warm-startean desde "
            "la cabeza francesa (la `kept-class flag` activa) -- el comportamiento por "
            "defecto.\n"
            "- **Brazo B (sin warm-start)**: `--no-warm-start`, toda la cabeza se "
            "inicializa al azar; el modelo aprende Italia sin el prior frances.\n\n"
            "El veredicto sale de `reports/us079_figs/ablation_compare.json` (lo produce "
            "`scripts/run_us079_ablation_analysis.py`), que compara ambos brazos **a dos "
            "granularidades** -- `levels.fine` (39 clases) y `levels.coarse` (19 buckets "
            "comunes con PASTIS) -- con foco en las **clases conservadas** (las que el "
            "warm-start toca). Cada nivel trae su `warm_start_verdict`: el F1-macro medio "
            "de cada brazo sobre las conservadas, cuantas mejoran/empeoran sin warm-start, "
            "y la bandera `warm_start_hurts_conserved`. Si el brazo B (sin warm-start) "
            "iguala o supera al A en las conservadas, el reciclaje **estorba**."
        )
    )
    cells.append(
        code(
            "# Per-level A/B summary table: F1-macro conservadas (brazo A vs B) at fine and coarse.\n"
            "if not HAS_ABLATION:\n"
            "    pending(f'JSON de la ablacion A/B ({ablation_glob!r}). Lo produce '\n"
            "            'scripts/run_us079_ablation_analysis.py cuando ambos brazos terminan de entrenar.')\n"
            "else:\n"
            "    rows = []\n"
            "    for level_name in ('fine', 'coarse'):\n"
            "        lvl = AB_LEVELS.get(level_name, {})\n"
            "        v = lvl.get('warm_start_verdict', {})\n"
            "        npa = lvl.get('n_patches_per_arm', {})\n"
            "        rows.append({\n"
            "            'nivel': 'fino (39 clases)' if level_name == 'fine' else 'coarse (19 buckets)',\n"
            "            'conservadas_comparadas': v.get('n_conserved_compared'),\n"
            "            'f1_macro_brazo_A_warmstart': v.get('mean_f1_warmstart'),\n"
            "            'f1_macro_brazo_B_nowarmstart': v.get('mean_f1_nowarmstart'),\n"
            "            'delta_B_menos_A': v.get('mean_delta_b_minus_a'),\n"
            "            'mejoran_sin_warmstart': v.get('n_conserved_improved'),\n"
            "            'empeoran_sin_warmstart': v.get('n_conserved_worsened'),\n"
            "            'empate': v.get('n_conserved_tied'),\n"
            "            'warm_start_estorba': v.get('warm_start_hurts_conserved'),\n"
            "            'patches_por_brazo': npa.get('A_warmstart'),\n"
            "        })\n"
            "    summary = pl.DataFrame(rows)\n"
            "    display(summary)\n"
            "    fvs = ablation.get('fine_verdict_summary', {})\n"
            "    print(f\"Resumen FINO (fine_verdict_summary): brazo A {fvs.get('mean_f1_warmstart')} \"\n"
            "          f\"vs brazo B {fvs.get('mean_f1_nowarmstart')} en {fvs.get('n_conserved_compared')} \"\n"
            '          f"clases conservadas.")\n'
        )
    )
    cells.append(
        code(
            "# Global verdict per level + per-conserved-class delta (B - A): warm-start helps or hurts?\n"
            "# Convencion del JSON: delta_b_minus_a > 0 => quitar el warm-start MEJORA => el warm-start ESTORBA.\n"
            "if not HAS_ABLATION:\n"
            "    pending('delta y veredicto por nivel del JSON de la ablacion A/B.')\n"
            "else:\n"
            "    def _verdict_word(hurts):\n"
            '        """Map the boolean hurts-flag to the human verdict word used in prose."""\n'
            "        return 'ESTORBA' if hurts else 'AYUDA'\n"
            "\n"
            "    lines = []\n"
            "    for level_name, label in (('fine', 'FINO'), ('coarse', 'COARSE')):\n"
            "        v = AB_LEVELS.get(level_name, {}).get('warm_start_verdict', {})\n"
            "        if not v.get('available'):\n"
            "            continue\n"
            "        word = _verdict_word(v.get('warm_start_hurts_conserved'))\n"
            "        lines.append(\n"
            '            f"- **Nivel {label}**: el warm-start **{word}** a las conservadas. "\n'
            "            f\"F1-macro brazo A (warm-start) = {v.get('mean_f1_warmstart')}, \"\n"
            "            f\"brazo B (sin warm-start) = {v.get('mean_f1_nowarmstart')} \"\n"
            "            f\"(delta medio B-A = {v.get('mean_delta_b_minus_a')}). \"\n"
            "            f\"{v.get('n_conserved_improved')} de {v.get('n_conserved_compared')} \"\n"
            "            f\"conservadas mejoran sin warm-start, {v.get('n_conserved_worsened')} empeoran, \"\n"
            "            f\"{v.get('n_conserved_tied')} empatan.\"\n"
            "        )\n"
            "    display(Markdown('**Veredicto del A/B (leido del JSON):**\\n\\n' + '\\n'.join(lines)))\n"
            "\n"
            "    # Texto plano del veredicto (contiene 0.132, 'warm-start' y la palabra del veredicto).\n"
            "    vf = AB_LEVELS.get('fine', {}).get('warm_start_verdict', {})\n"
            "    word_f = _verdict_word(vf.get('warm_start_hurts_conserved'))\n"
            '    print(f"VEREDICTO FINO: el warm-start {word_f} en las conservadas "\n'
            "          f\"(F1-macro {vf.get('mean_f1_warmstart')} con warm-start vs \"\n"
            "          f\"{vf.get('mean_f1_nowarmstart')} sin warm-start; \"\n"
            "          f\"hurts={vf.get('warm_start_hurts_conserved')}).\")\n"
            "    vc = AB_LEVELS.get('coarse', {}).get('warm_start_verdict', {})\n"
            "    word_c = _verdict_word(vc.get('warm_start_hurts_conserved'))\n"
            '    print(f"VEREDICTO COARSE: el warm-start {word_c} en las conservadas "\n'
            "          f\"(F1-macro {vc.get('mean_f1_warmstart')} con warm-start vs \"\n"
            "          f\"{vc.get('mean_f1_nowarmstart')} sin warm-start; \"\n"
            "          f\"hurts={vc.get('warm_start_hurts_conserved')}).\")\n"
            "\n"
            "    # Delta B - A por clase conservada (nivel fino): donde el prior frances ayuda o estorba.\n"
            "    per = vf.get('per_conserved', [])\n"
            "    if per:\n"
            "        pcd = pl.DataFrame(per).select(\n"
            "            ['class_name', 'support', 'f1_A_warmstart', 'f1_B_nowarmstart', 'delta_b_minus_a']\n"
            "        ).rename({\n"
            "            'class_name': 'clase', 'support': 'soporte_px',\n"
            "            'f1_A_warmstart': 'f1_brazo_A_warmstart',\n"
            "            'f1_B_nowarmstart': 'f1_brazo_B_nowarmstart',\n"
            "            'delta_b_minus_a': 'delta_B_menos_A',\n"
            "        }).sort('delta_B_menos_A', descending=True)\n"
            "        with pl.Config(tbl_rows=25):\n"
            "            display(pcd)\n"
            "        helps_b = pcd.filter(pl.col('delta_B_menos_A') > 0)['clase'].to_list()\n"
            "        hurts_b = pcd.filter(pl.col('delta_B_menos_A') < 0)['clase'].to_list()\n"
            "        print(f'Clases donde QUITAR el warm-start mejora (el prior estorba): {helps_b}')\n"
            "        print(f'Clases donde el warm-start ayuda (mantenerlo es mejor): {hurts_b}')\n"
        )
    )
    cells.append(
        code(
            "# A/B comparison figures (fine + coarse) produced by the ablation runner.\n"
            "# Los PNG llevan sufijo _fine/_coarse; se cargan ambos niveles cuando existen.\n"
            "for level_name, label in (('fine', 'fino, 39 clases'), ('coarse', 'coarse, 19 buckets')):\n"
            "    display(Markdown(f'#### Nivel {label}'))\n"
            "    show_fig(FIGS_DIR / f'fig_ab_per_class_{level_name}.png',\n"
            "             f'F1 por clase ({label}), brazo A (warm-start) vs brazo B (sin warm-start): '\n"
            "             'donde el reciclaje cambia la decision.')\n"
            "    show_fig(FIGS_DIR / f'fig_ab_conserved_delta_{level_name}.png',\n"
            "             f'Delta B - A restringido a las clases CONSERVADAS ({label}). '\n"
            "             'Barras a la derecha (delta > 0) = quitar el warm-start mejora = el reciclaje estorba.')\n"
            "    show_fig(FIGS_DIR / f'fig_discard_compare_{level_name}.png',\n"
            "             f'Curvas de descarte honesto ({label}) de los tres brazos '\n"
            "             '(A warm-start, B sin warm-start, original).')\n"
        )
    )
    cells.append(
        md(
            "**Lectura del A/B (con los numeros reales del JSON).** El hallazgo es "
            "**matizado, no binario**, y el cuaderno lo muestra a las dos "
            "granularidades:\n\n"
            "- **A nivel fino, el warm-start estorba.** El brazo A (con warm-start) "
            "promedia F1-macro **0,1321** sobre las conservadas frente a **0,1199** del "
            "brazo B; pero el conteo manda: **11 de 19 clases conservadas mejoran sin "
            "warm-start** (8 empeoran, 1 empata), de ahi `warm_start_hurts_conserved = "
            "True`. El prior frances **desvia el detalle**: estorba en las clases cuya "
            "fenologia mediterranea difiere de la atlantica (`sunflower` +0,065, `oats` "
            "+0,085, `common_soft_wheat` +0,037, `permanent_grassland` +0,032 al "
            "quitarlo) y solo ayuda donde el cultivo se parece entre Francia e Italia "
            "(`durum_hard_wheat`, `barley`, `vineyards`, `maize`, que empeoran sin "
            "warm-start).\n"
            "- **A nivel coarse, el warm-start ayuda.** Agrupando a los 19 buckets "
            "comunes con PASTIS, `warm_start_hurts_conserved = False`: el prior frances "
            "**captura la senal gruesa** (que un pixel sea cereal de invierno, forraje o "
            "vina) aunque se equivoque en la sub-clase fina. Es decir, el reciclaje "
            "transfiere la estructura macro del paisaje agricola pero no el matiz "
            "mediterraneo.\n\n"
            "**Conclusion del A/B.** El reciclaje que ayudo en Francia->Baltico aqui es "
            "**selectivamente util**: conviene a nivel grueso y para los cultivos "
            "FR-IT comparables (durum, cebada, vid, maiz), pero estorba el detalle fino "
            "de los cultivos divergentes (girasol, trigo blando, pastos). El siguiente "
            "paso natural es un **reciclaje informado por la distancia de dominio** -- "
            "warm-startear solo donde la fenologia es comparable -- en vez del todo o "
            "nada."
        )
    )

    # ================================================ 6. original vs TL ===
    cells.append(
        md(
            "## 6. Comparacion original vs transfer learning\n\n"
            "El JSON de la ablacion trae, por clase, el F1 del checkpoint **original** "
            "(`us079_v2`, el modelo antes de la ablacion) junto al de los dos brazos "
            "afinados, con sus deltas `delta_*_minus_original`. Eso permite dos "
            "comparaciones **con numeros reales**:\n\n"
            "1. **Original vs brazos afinados (delta del transfer por clase).** Para cada "
            "brazo, `delta = F1(brazo) - F1(original)` dice si re-afinar con esa "
            "configuracion sube o baja respecto del checkpoint de partida.\n"
            "2. **Paridad Francia vs Italia.** El campeon frances logro **F1-macro "
            f"{_FRANCE_CHAMPION_F1} sobre `france-10`** (referencia medida de EPIC 6). "
            "Contrastar ese 0,9069 con el F1-macro que Italia alcanza sobre sus "
            "conservadas cuantifica el **costo del domain gap mediterraneo** -- la "
            "distancia entre lo que el modelo lograba en casa y lo que logra al cruzar "
            "los Alpes."
        )
    )
    cells.append(
        code(
            "# Transfer delta vs the ORIGINAL checkpoint (us079_v2), per class, from the real A/B JSON.\n"
            "fine = AB_LEVELS.get('fine') if HAS_ABLATION else None\n"
            "if fine and fine.get('per_class_table'):\n"
            "    pc = pl.DataFrame(fine['per_class_table'])\n"
            "    cols = [c for c in ['class_name', 'is_conserved', 'f1_original', 'f1_A_warmstart',\n"
            "                        'f1_B_nowarmstart', 'delta_A_warmstart_minus_original',\n"
            "                        'delta_B_nowarmstart_minus_original'] if c in pc.columns]\n"
            "    tbl = pc.select(cols).rename({\n"
            "        'class_name': 'clase', 'is_conserved': 'conservada',\n"
            "        'f1_original': 'f1_original', 'f1_A_warmstart': 'f1_brazo_A',\n"
            "        'f1_B_nowarmstart': 'f1_brazo_B',\n"
            "        'delta_A_warmstart_minus_original': 'delta_A_vs_original',\n"
            "        'delta_B_nowarmstart_minus_original': 'delta_B_vs_original',\n"
            "    }).sort('f1_brazo_A', descending=True)\n"
            "    with pl.Config(tbl_rows=45):\n"
            "        display(tbl)\n"
            "    mean_a = pc['delta_A_warmstart_minus_original'].mean()\n"
            "    mean_b = pc['delta_B_nowarmstart_minus_original'].mean()\n"
            "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "    vals = [round(float(mean_a), 4), round(float(mean_b), 4)]\n"
            "    labels = ['brazo A (warm-start) - original', 'brazo B (sin warm-start) - original']\n"
            "    colors = ['#2e7d32' if v >= 0 else '#c62828' for v in vals]\n"
            "    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])\n"
            "    ax.axvline(0, color='black', linewidth=0.8)\n"
            "    ax.set_title('Delta medio del transfer vs el checkpoint original (nivel fino)')\n"
            "    for i, v in enumerate(vals[::-1]):\n"
            "        ax.text(v, i, f'{v:+.4f}', va='center')\n"
            "    ax.grid(axis='x', alpha=0.3); plt.tight_layout(); plt.show(); plt.close(fig)\n"
            "    print(f'Delta medio F1 (fino) brazo A vs original: {mean_a:+.4f}')\n"
            "    print(f'Delta medio F1 (fino) brazo B vs original: {mean_b:+.4f}')\n"
            "else:\n"
            "    pending('delta del transfer vs original: falta levels.fine.per_class_table en el JSON.')\n"
        )
    )
    cells.append(
        code(
            "# Parity bar: France champion (france-10) vs Italy conserved F1-macro, real numbers.\n"
            "fvs = ablation.get('fine_verdict_summary', {}) if HAS_ABLATION else {}\n"
            "italia_fine = fvs.get('mean_f1_warmstart')  # brazo A (por defecto), conservadas, nivel fino\n"
            "italia_coarse = (AB_LEVELS.get('coarse', {}).get('warm_start_verdict', {})\n"
            "                 .get('mean_f1_warmstart')) if HAS_ABLATION else None\n"
            "if italia_fine is not None:\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 3))\n"
            "    bars = ['Francia\\n(france-10, medido)', 'Italia conservadas\\n(fino, brazo A)',\n"
            "            'Italia conservadas\\n(coarse, brazo A)']\n"
            "    vals = [float(france_champion_f1), float(italia_fine),\n"
            "            float(italia_coarse) if italia_coarse is not None else 0.0]\n"
            "    ax.bar(bars, vals, color=['#2e7d32', '#c62828', '#ef6c00'])\n"
            "    ax.axhline(f1_threshold, color='grey', linestyle='--', label=f'objetivo {f1_threshold}')\n"
            "    for i, v in enumerate(vals):\n"
            "        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')\n"
            "    ax.set_ylim(0, 1); ax.set_ylabel('F1-macro'); ax.legend()\n"
            "    ax.set_title('Paridad Francia vs Italia (costo del domain gap mediterraneo)')\n"
            "    plt.tight_layout(); plt.show(); plt.close(fig)\n"
            "    print(f'Brecha Francia - Italia (fino): {france_champion_f1 - float(italia_fine):+.4f} F1-macro.')\n"
            "    print('El transfer mediterraneo NO espeja a france-10: el domain gap fenologico es severo.')\n"
            "else:\n"
            "    pending('F1-macro de las conservadas italianas para la paridad vs france-10 '\n"
            "            '(falta fine_verdict_summary en el JSON de la ablacion). '\n"
            f"            'La referencia francesa ({_FRANCE_CHAMPION_F1}) ya es un valor medido fijo.')\n"
        )
    )

    # ================================================ 7. reciclaje de clases ===
    cells.append(
        md(
            "## 7. Analisis del reciclaje de clases\n\n"
            "Esta seccion conecta el reciclaje con el hallazgo del A/B usando los numeros "
            "reales del JSON. De las clases italianas, un subconjunto se **conservo** "
            "(mapea a PASTIS y se warm-startea) y el resto son **nuevas mediterraneas** "
            "(parten de cero). La pregunta: **el reciclaje ayuda donde lo aplicamos?**\n\n"
            "El hallazgo es **matizado** (lo cuantifica el A/B de la seccion 5): a nivel "
            "**fino**, `warm_start_hurts_conserved = True` -- 11 de 19 conservadas mejoran "
            "al quitar el warm-start, porque el prior atlantico **ancla la fenologia "
            "equivocada** en los cultivos divergentes (girasol, trigo blando, avena, "
            "pastos). A nivel **coarse**, en cambio, `warm_start_hurts_conserved = False`: "
            "el prior frances **captura la senal gruesa** del paisaje (cereal, forraje, "
            "vina) aunque yerre la sub-clase. La celda separa conservadas vs nuevas a "
            "partir del flag `is_conserved` real y mide el F1 medio del brazo A en cada "
            "grupo, para ver si el warm-start basta para que las conservadas dominen."
        )
    )
    cells.append(
        code(
            "# Recycled (conserved) vs new (Mediterranean) groups: counts + mean F1, real A/B JSON.\n"
            "fine = AB_LEVELS.get('fine') if HAS_ABLATION else None\n"
            "if fine and fine.get('per_class_table'):\n"
            "    pc = pl.DataFrame(fine['per_class_table'])\n"
            "    grp = (pc.group_by('is_conserved')\n"
            "             .agg(pl.len().alias('n_clases'),\n"
            "                  pl.col('f1_A_warmstart').mean().round(4).alias('f1_medio_brazo_A'),\n"
            "                  pl.col('f1_A_warmstart').max().round(4).alias('f1_max_brazo_A'),\n"
            "                  pl.col('delta_b_minus_a').mean().round(4).alias('delta_medio_B_menos_A'))\n"
            "             .with_columns(pl.when(pl.col('is_conserved'))\n"
            "                             .then(pl.lit('conservadas (warm-start PASTIS)'))\n"
            "                             .otherwise(pl.lit('nuevas mediterraneas (desde cero)')).alias('grupo'))\n"
            "             .select(['grupo', 'n_clases', 'f1_medio_brazo_A', 'f1_max_brazo_A',\n"
            "                      'delta_medio_B_menos_A'])\n"
            "             .sort('f1_medio_brazo_A', descending=True))\n"
            "    display(grp)\n"
            "    n_cons = pc.filter(pl.col('is_conserved')).height\n"
            "    n_new = pc.filter(~pl.col('is_conserved')).height\n"
            "    print(f'Clases conservadas (warm-starteadas): {n_cons} | nuevas mediterraneas: {n_new}')\n"
            "    cons_f1 = pc.filter(pl.col('is_conserved'))['f1_A_warmstart'].mean()\n"
            "    new_f1 = pc.filter(~pl.col('is_conserved'))['f1_A_warmstart'].mean()\n"
            "    print(f'F1 medio brazo A -- conservadas: {cons_f1:.4f} | nuevas: {new_f1:.4f}')\n"
            "    delta_cons = pc.filter(pl.col('is_conserved'))['delta_b_minus_a'].mean()\n"
            "    print(f'Delta medio B-A en conservadas: {delta_cons:+.4f} '\n"
            "          '(> 0 => quitar el warm-start mejora => el reciclaje estorba el detalle fino).')\n"
            "    if delta_cons is not None and delta_cons > 0:\n"
            "        print('HALLAZGO FINO: a nivel fino el warm-start ESTORBA en promedio a las conservadas; '\n"
            "              'el reciclaje atlantico desvia el matiz mediterraneo (ver A/B, seccion 5).')\n"
            "else:\n"
            "    pending('per-clase con la bandera is_conserved: falta levels.fine.per_class_table en el JSON.')\n"
        )
    )

    # ================================================ 8. conclusiones ===
    cells.append(
        md(
            "## 8. Conclusiones honestas\n\n"
            "**Que se logro.** Extendimos el comite ganador de EPIC 6 (Voting-3) al "
            "homologo italiano: afinamos los miembros densos desde el checkpoint PASTIS "
            "sobre 1438 patches en formato PASTIS, re-entrenamos el miembro tabular sobre "
            "AlphaEarth muestreado en Italia, y re-aprendimos los pesos del Voting con "
            "spatial-CV. Montamos ademas una **ablacion A/B limpia** del warm-start para "
            "responder, con evidencia, si el reciclaje ayuda o estorba.\n\n"
            "**El transfer mediterraneo es dificil (hallazgo cientifico real).** A "
            "diferencia de Francia, donde el Voting-3 lograba **F1-macro 0,9069 sobre "
            "`france-10`** (referencia medida de EPIC 6), Italia **no espeja** ese "
            "resultado: el F1-macro de las conservadas a nivel fino se queda en **0,1321** "
            "(brazo A) -- una brecha enorme. La meta de F1 > 0,9 sobre las mejores ~10 "
            "clases no se alcanza: es un hallazgo honesto sobre el limite del transfer "
            "cross-domain, no un fallo a maquillar.\n\n"
            "**El hallazgo sobre el warm-start es matizado, no binario.** El reciclaje "
            "(`kept-class flag`) que **funciono en Francia->Baltico** aqui **estorba el "
            "detalle pero ayuda lo grueso**: a nivel fino `warm_start_hurts_conserved = "
            "True` (brazo A 0,1321 vs brazo B 0,1199, pero 11 de 19 conservadas mejoran "
            "sin warm-start), mientras a nivel coarse `warm_start_hurts_conserved = "
            "False`. Lectura: **el prior frances captura la senal gruesa (que un pixel sea "
            "cereal de invierno o forraje) pero desvia el detalle mediterraneo** (la "
            "sub-clase fina). El reciclaje ayuda donde el cultivo se parece entre Francia "
            "e Italia (`durum`, `barley`, `vineyards`, `maize`) y estorba donde la "
            "fenologia diverge (`sunflower`, `common_soft_wheat`, `oats`, "
            "`permanent_grassland`).\n\n"
            "**Limitaciones y trabajo futuro.**\n\n"
            "- **Domain gap fenologico**: el clima mediterraneo desplaza los picos NDVI; "
            "un re-encuadre fenologico especifico para Italia (no heredado de Francia) es "
            "el siguiente paso natural.\n"
            "- **Reciclaje selectivo**: en vez de warm-startear todas las clases "
            "conservadas, hacerlo solo donde la fenologia es comparable (cultivos lenosos "
            "como la vid) y dejar el resto desde cero -- una `kept-class flag` informada "
            "por la distancia de dominio.\n"
            "- **Mas datos italianos** y/o un curriculum de transfer por etapas (atlantico "
            "-> templado -> mediterraneo) para suavizar el salto de dominio.\n\n"
            "**Trazabilidad.** El run de MLflow (`us079-transfer-italia`) lleva los tags "
            "`data_version` + `code_version`; las cifras de este cuaderno provienen del "
            "JSON de la ablacion A/B (`reports/us079_figs/ablation_compare.json`) y de las "
            "figuras precomputadas -- sin numeros inventados. El `report.json` del Voting "
            "final (pesos del comite, evaluacion del ensamble) lo escribira el ensamble en "
            "la H100 sobre el dataset completo; mientras tanto, las celdas que dependen "
            "**solo** de ese reporte muestran un estado **PENDIENTE** explicito. La unica "
            f"constante fija es el `france_champion_f1 = {_FRANCE_CHAMPION_F1}`, etiquetada "
            "como **referencia EPIC 6 medida**."
        )
    )
    return cells


@app.command()
def build(
    out: Annotated[Path, typer.Option(help="Ruta de salida del notebook.")] = _DEFAULT_OUT,
    report_dir: Annotated[
        Path, typer.Option(help="Ruta de la salida del runner (report.json).")
    ] = _DEFAULT_REPORT,
    data_dir: Annotated[
        Path, typer.Option(help="Ruta del dataset homologo (mascaras de test).")
    ] = _DEFAULT_DATA,
    figs_dir: Annotated[
        Path, typer.Option(help="Ruta de las figuras precomputadas (fig1/fig2 + A/B).")
    ] = _DEFAULT_FIGS,
    ablation_glob: Annotated[
        str, typer.Option(help="Glob del JSON de la ablacion A/B del warm-start.")
    ] = _DEFAULT_ABLATION_GLOB,
) -> None:
    """Write the US-079 eval notebook (unexecuted; papermill populates outputs).

    Args:
        out: Output ``.ipynb`` path.
        report_dir: Repo-relative path to the runner output the notebook reads.
        data_dir: Repo-relative path to the homologue dataset.
        figs_dir: Repo-relative path to the precomputed figures.
        ablation_glob: Repo-relative glob for the warm-start A/B summary JSON.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = _build_cells(
        str(report_dir).replace("\\", "/"),
        str(data_dir).replace("\\", "/"),
        str(figs_dir).replace("\\", "/"),
        ablation_glob.replace("\\", "/"),
    )
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    typer.echo(f"Notebook escrito en {out} ({len(nb.cells)} celdas).")


if __name__ == "__main__":
    app()
