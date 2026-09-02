"""Programmatic builder of ``notebooks/04_baseline.ipynb`` (EPIC 4, US-019).

Generates the RF/XGB baseline notebook cell by cell with ``nbformat.v4``,
executable end-to-end with papermill and byte-by-byte reproducible. The notebook
is the visual deliverable of Avance 3.

Sections produced by US-019:
  - 1: Setup and load of the EPIC 3 feature vector.
  - 2: Algorithm justification ("Algorithm" criterion, 40 pts).
  - 6: Minimum performance vs F1-macro threshold >= 0.60 (criterion 10 pts).

Sections produced by US-020 ("Important features" criterion, 20 pts):
  - 3: Native feature importance (RF Gini / XGB gain, top-20 barplot).
  - 4: SHAP analysis (summary + top-5 dependence + waterfall + AlphaEarth
    dominance).
  - 5: Feature engineering conclusions (cross-check with the US-018 FE).

Section produced by US-021 ("Under/overfitting" criterion, 10 pts):
  - 5b: Learning curves (RF+XGB) + 3 validation curves + textual under/overfit
    diagnosis (`diagnose_fit`) + spatial cross-validation criterion.

Sections produced by US-022 ("Metric" criterion, 20 pts; last extend):
  - 7: Comparison of the 3 feature scenarios (pure AlphaEarth vs raw Sentinel-2
    vs combined vector) with `build_comparison_table` + barplot + LaTeX export.
  - 8: Discussion of the incremental value of AlphaEarth + conclusions for
    EPIC 5 + closing of the CRISP-ML(Q) Modeling phase.

Pattern: ``scripts/build_us018_notebook.py``.

Usage:
    poetry run python scripts/build_baseline_notebook.py --out notebooks/baseline/04_baseline.ipynb

US-023-preview notes:
- The canonical path moved to ``notebooks/baseline/`` (decision D-6).
- P8 adds v2 cells that retrain the 3 A3 models (XGBoost + TempCNN +
  InceptionTime) on the winning feature set after the P2/P3/P4/P5 ablation.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _md(source: str) -> nbf.NotebookNode:
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    """Create a code cell."""
    return nbf.v4.new_code_cell(source)


def _params_code(source: str) -> nbf.NotebookNode:
    """Create the parameters cell (``parameters`` tag for papermill)."""
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = ["parameters"]
    return cell


# ---------------------------------------------------------------------------
# Notebook cells.
# ---------------------------------------------------------------------------

CELLS: list[nbf.NotebookNode] = [
    _md(
        "# Baseline de clasificación de cultivos — Random Forest y XGBoost\n"
        "\n"
        "Este notebook responde una pregunta concreta: **¿que tan lejos "
        "llega un modelo tabular sencillo para clasificar cultivos a partir "
        "de imagenes satelitales?** Se entrenan dos modelos de arboles "
        "(Random Forest y XGBoost) sobre un vector de caracteristicas que "
        "combina el embedding AlphaEarth de 64 dimensiones, indices "
        "espectrales, estadisticas temporales, terreno y clima.\n"
        "\n"
        "El resultado sirve de **punto de referencia**: cualquier modelo "
        "mas complejo en fases posteriores tendra que superar estas cifras "
        "para justificar su coste.\n"
        "\n"
        "## Requisitos para ejecución end-to-end\n"
        "\n"
        "- El subset PASTIS-R a nivel parcela descomprimido en "
        "`data/test_fixtures/`.\n"
        "- Dependencias instaladas via `poetry install --with ml,geo`.\n"
        "\n"
        "El notebook se ejecuta de principio a fin sin intervencion manual; "
        "los parametros (tamano de muestra, tuning) se ajustan desde la "
        "celda de parametros.\n"
        "\n"
        "## Contenido\n"
        "\n"
        "| Sección | Contenido |\n"
        "|---------|-----------|\n"
        "| 1 | Carga del conjunto de datos |\n"
        "| 2 | Por qué Random Forest y XGBoost |\n"
        "| 3 | Importancia de características |\n"
        "| 4 | Análisis SHAP |\n"
        "| 5 | Conclusiones de ingeniería de características |\n"
        "| 5b | Curvas de aprendizaje y validación |\n"
        "| 6 | Desempeño del baseline |\n"
        "| 7 | Comparativa AlphaEarth vs Sentinel-2 crudo |\n"
        "| 8 | Conclusiones |\n"
    ),
    # --- Section 1 -------------------------------------------------------
    _md(
        "## 1. Carga del conjunto de datos\n"
        "\n"
        "El conjunto de entrada es un subset de PASTIS-R a nivel de "
        "parcela: 85.951 parcelas agricolas con 187 caracteristicas "
        "espectro-temporales cada una. La etiqueta es el tipo de cultivo "
        "(20 clases de PASTIS-R; se descartan las clases de fondo)."
    ),
    _params_code(
        "# Parametros papermill (celda con tag 'parameters'; sobreescribibles\n"
        "# en CI con valores reducidos via `papermill -p`).\n"
        "FEATURES_PATH = 'data/test_fixtures/feature_selection_parcels_subset.parquet'\n"
        "MAX_SAMPLES = 0  # 0 = dataset completo; >0 = submuestreo estratificado\n"
        "TUNE = True\n"
        "F1_THRESHOLD = 0.60\n"
        "# Seccion 7 (US-022) — rutas de los 3 escenarios de la comparativa.\n"
        "SCENARIO_ALPHAEARTH_PATH = (\n"
        "    'data/cache/gee/alphaearth_pastis_parcels_2019_85951_enriched.parquet'\n"
        ")\n"
        "SCENARIO_S2_RAW_PATH = (\n"
        "    'data/cache/pastis/s2_raw_parcels_2019_85951.parquet'\n"
        ")\n"
        "SCENARIO_COMBINED_PATH = (\n"
        "    'data/test_fixtures/feature_selection_parcels_subset.parquet'\n"
        ")\n"
        "COMPARISON_MAX_SAMPLES = 0  # 0 = todas las parcelas del inner join\n"
        "COMPARISON_K_FOLDS = 5\n"
        "# Nota US-023-preview corrida 3: la antigua seccion 9 (Baseline v2 con 3\n"
        "# modelos) se movio al notebook standalone `notebooks/baseline/04b_baseline_v2.ipynb`,\n"
        "# que lee los artefactos persistidos en `reports/baseline/model_comparison_v2/`\n"
        "# y no requiere reentrenar. El script de training real es\n"
        "# `scripts/run_baseline_v2_standalone.py` (Makefile target `baseline-v2-full`).\n"
    ),
    _code(
        "import warnings\n"
        "\n"
        "import matplotlib\n"
        "\n"
        "matplotlib.use('Agg')  # backend headless para papermill/CI\n"
        "import matplotlib.pyplot as plt\n"
        "import polars as pl\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
    ),
    _code(
        "from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe\n"
        "\n"
        "df_raw = _load_baseline_dataset(FEATURES_PATH)\n"
        "df = _prepare_dataframe(df_raw)\n"
        "print(f'Parcelas: {df.height:,}  |  Columnas: {df.width}')\n"
        "df.head()"
    ),
    _code(
        "# Distribucion de clases — PASTIS-R tiene desbalance fuerte.\n"
        "class_counts = (\n"
        "    df.group_by('class_id').len().sort('len', descending=True)\n"
        ")\n"
        "class_counts"
    ),
    # --- Section 2 -------------------------------------------------------
    _md(
        "## 2. Por qué Random Forest y XGBoost\n"
        "\n"
        "Se eligen **Random Forest** y **XGBoost** como modelos de "
        "referencia. Cuatro razones sustentan la decision:\n"
        "\n"
        "**(a) Las imagenes ya vienen resumidas.** El embedding AlphaEarth "
        "de 64 dimensiones condensa informacion optica, radar y temporal "
        "aprendida por un modelo entrenado sobre todo el archivo Sentinel. "
        "Sobre una representacion ya rica, un modelo de arboles es un "
        "punto de referencia suficiente y honesto — no hace falta una red "
        "neuronal profunda para establecer el piso de desempeno (cf. "
        "Brown et al., 2025, *AlphaEarth Foundations*).\n"
        "\n"
        "**(b) Son interpretables.** Ambos exponen una medida de "
        "importancia de caracteristicas (Gini para Random Forest, "
        "*gain* para XGBoost) y son compatibles con SHAP. Esto permite "
        "auditar que variables explican las predicciones — un modelo "
        "opaco no lo permitiria (Lundberg & Lee, 2017, *SHAP*).\n"
        "\n"
        "**(c) Son robustos a valores atipicos y a la escala.** Los "
        "arboles dividen el espacio por umbrales y no asumen ninguna "
        "distribucion de las variables; los valores atipicos residuales "
        "no desplazan las fronteras de decision como lo harian en un "
        "modelo lineal sin normalizacion cuidadosa.\n"
        "\n"
        "**(d) Tienen bajo coste computacional.** El problema (85.951 "
        "parcelas, 187 variables, 20 clases) se entrena en minutos. "
        "XGBoost aprovecha la GPU local cuando esta disponible y degrada a "
        "CPU de forma transparente; Random Forest corre siempre en CPU "
        "multinucleo. El experimento es reproducible en cualquier laptop."
    ),
    # --- Section 3 — Feature importance ---------------------------------
    _md(
        "## 3. Importancia de características\n"
        "\n"
        "Random Forest y XGBoost exponen una medida de importancia de "
        "caracteristicas sin coste adicional: **Gini** para Random Forest "
        "y **gain** para XGBoost. Es el primer diagnostico de "
        "interpretabilidad — barato y directo — antes del analisis SHAP "
        "de la seccion 4.\n"
        "\n"
        "Se cargan los modelos ya entrenados desde "
        "`artifacts/baseline_{rf,xgb}_v1.joblib`; si los archivos no "
        "existen, el notebook entrena los modelos en el momento con los "
        "hiperparametros base."
    ),
    _code(
        "import joblib\n"
        "from pathlib import Path\n"
        "\n"
        "from ml.train.baseline import train_one_model\n"
        "\n"
        "REPORTS_DIR = Path('reports/baseline')\n"
        "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "ARTIFACTS = {'rf': Path('artifacts/baseline_rf_v1.joblib'),\n"
        "             'xgb': Path('artifacts/baseline_xgb_v1.joblib')}\n"
        "\n"
        "models = {}\n"
        "for kind, path in ARTIFACTS.items():\n"
        "    if path.exists():\n"
        "        payload = joblib.load(path)\n"
        "        models[kind] = {\n"
        "            'model': payload['model'],\n"
        "            'feature_cols': tuple(payload['feature_cols']),\n"
        "            'source': 'joblib US-019',\n"
        "        }\n"
        "    else:\n"
        "        res = train_one_model(df, model=kind)\n"
        "        models[kind] = {\n"
        "            'model': res.model,\n"
        "            'feature_cols': res.feature_cols,\n"
        "            'source': 'fallback in-notebook (D8)',\n"
        "        }\n"
        "    print(f\"{kind.upper()}: {models[kind]['source']}  |  \"\n"
        "          f\"{len(models[kind]['feature_cols'])} features\")"
    ),
    _code(
        "from ml.eval.interpretability import feature_importance_table\n"
        "\n"
        "importance = {}\n"
        "for kind, bundle in models.items():\n"
        "    table = feature_importance_table(\n"
        "        bundle['model'], kind, bundle['feature_cols']\n"
        "    )\n"
        "    importance[kind] = table\n"
        "    table.write_csv(REPORTS_DIR / f'feature_importance_{kind}.csv')\n"
        "importance['rf'].head(10)"
    ),
    _code(
        "# Barplot top-20 de la importancia nativa por modelo.\n"
        "for kind, table in importance.items():\n"
        "    top20 = table.head(20)\n"
        "    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)\n"
        "    ax.barh(top20['feature'].to_list()[::-1],\n"
        "            top20['importance'].to_list()[::-1],\n"
        "            color='#2c7fb8')\n"
        "    ax.set_xlabel('Importancia (' + ('Gini' if kind == 'rf' else 'gain') + ')')\n"
        "    ax.set_title(f'Importancia nativa top-20 — {kind.upper()}')\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(REPORTS_DIR / f'importance_{kind}_top20.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    # --- Section 4 — SHAP analysis (US-020) ------------------------------
    _md(
        "## 4. Análisis SHAP\n"
        "\n"
        "La importancia de la seccion 3 ordena las caracteristicas pero no "
        "explica *como* cada una desplaza la prediccion. **SHAP** "
        "(Lundberg & Lee, 2017) descompone cada prediccion en "
        "contribuciones aditivas por caracteristica, con garantias "
        "teoricas de consistencia. Para modelos de arboles se usa el "
        "algoritmo TreeSHAP, que es exacto.\n"
        "\n"
        "Detalles de la implementación:\n"
        "\n"
        "- **Submuestreo**: SHAP se calcula sobre una muestra "
        "estratificada de ~3.000 parcelas, no sobre las ~85.000 del "
        "conjunto; el coste de TreeSHAP crece con el numero de muestras, "
        "arboles y profundidad.\n"
        "- **Multiclase**: PASTIS-R tiene 18-20 clases; la salida "
        "multiclase de SHAP se normaliza a un tensor uniforme "
        "`(muestras, caracteristicas, clases)`.\n"
        "- **Ranking global**: la importancia global es el promedio del "
        "valor absoluto de SHAP sobre clases y muestras."
    ),
    _code(
        "from ml.eval.interpretability import (\n"
        "    compute_shap_values,\n"
        "    shap_summary_plot,\n"
        "    shap_dependence_plots,\n"
        "    shap_waterfall_plot,\n"
        ")\n"
        "\n"
        "SHAP_SAMPLE_SIZE = 3000\n"
        "shap_results = {}\n"
        "for kind, bundle in models.items():\n"
        "    shap_results[kind] = compute_shap_values(\n"
        "        bundle['model'], df, kind,\n"
        "        feature_cols=bundle['feature_cols'],\n"
        "        sample_size=SHAP_SAMPLE_SIZE,\n"
        "    )\n"
        "    print(f'{kind.upper()}: tensor SHAP '\n"
        "          f'{shap_results[kind].values.shape}')"
    ),
    _code(
        "# Summary plot (beeswarm/bar) de las top-20 features globales.\n"
        "for kind, result in shap_results.items():\n"
        "    fig = shap_summary_plot(result, df, top_n=20)\n"
        "    fig.savefig(REPORTS_DIR / f'shap_summary_{kind}.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    _code(
        "# Dependence plots de los 5 features mas importantes (RF).\n"
        "dependence = shap_dependence_plots(\n"
        "    shap_results['rf'], df, top_features=5\n"
        ")\n"
        "for idx, (feature_name, fig) in enumerate(dependence, start=1):\n"
        "    fig.savefig(\n"
        "        REPORTS_DIR / f'shap_dependence_{idx}_{feature_name}.png',\n"
        "        dpi=200, bbox_inches='tight',\n"
        "    )\n"
        "    plt.show()"
    ),
    _code(
        "# Waterfall de una prediccion ejemplo por modelo.\n"
        "for kind, result in shap_results.items():\n"
        "    fig = shap_waterfall_plot(result, row=0)\n"
        "    fig.savefig(REPORTS_DIR / f'shap_waterfall_{kind}.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    _md(
        "### 4.1 Dominancia de las dimensiones AlphaEarth\n"
        "\n"
        "Una pregunta interesante: de las caracteristicas mas influyentes "
        "segun SHAP, **¿cuantas son dimensiones del embedding AlphaEarth** "
        "(`dim_00..dim_63`) frente a indices espectrales, estadisticas "
        "temporales o bloques de contexto (radar, terreno, clima)? La "
        "respuesta indica cuanto del poder predictivo proviene del "
        "embedding satelital frente al resto de las caracteristicas."
    ),
    _code(
        "from ml.eval.interpretability import alphaearth_dominance_table\n"
        "\n"
        "dominance = alphaearth_dominance_table(\n"
        "    shap_results['rf'].global_importance, top_n=20\n"
        ")\n"
        "dominance.write_csv(REPORTS_DIR / 'alphaearth_dominance.csv')\n"
        "dominance"
    ),
    _code(
        "# Conteo por familia y conclusion cuantificada.\n"
        "family_counts = (\n"
        "    dominance.group_by('family').len()\n"
        "    .sort('len', descending=True)\n"
        ")\n"
        "n_alphaearth = int(\n"
        "    dominance.filter(pl.col('family') == 'alphaearth').height\n"
        ")\n"
        "top_ae = (\n"
        "    dominance.filter(pl.col('family') == 'alphaearth')['feature']\n"
        "    .to_list()[:3]\n"
        ")\n"
        "print(f'{n_alphaearth}/20 de las top features SHAP son '\n"
        "      f'dimensiones AlphaEarth.')\n"
        "if top_ae:\n"
        "    print(f'Lideran: ' + ', '.join(top_ae))\n"
        "family_counts"
    ),
    # --- Section 5 — Feature engineering conclusions (US-020) ------------
    _md(
        "## 5. Conclusiones de ingeniería de características\n"
        "\n"
        "Esta seccion **valida o cuestiona** las decisiones de ingenieria "
        "de caracteristicas de la fase anterior, cruzando los rankings de "
        "interpretabilidad de este notebook con los resultados de la "
        "seleccion de variables previa:\n"
        "\n"
        "- `reports/feature_selection/feature_importance_rf.csv` — "
        "importancia exploratoria de la fase de seleccion.\n"
        "- `reports/feature_selection/anova_f_scores.csv` — F-scores "
        "univariados de la seleccion.\n"
        "- `reports/feature_selection/selected_features.json` — el "
        "conjunto de variables que se retuvo.\n"
        "\n"
        "El objetivo es responder tres preguntas: (a) ¿las caracteristicas "
        "mas influyentes segun SHAP coinciden con las que se "
        "seleccionaron?; (b) ¿alguna variable descartada aparece como "
        "importante?; (c) ¿la dominancia de AlphaEarth confirma la "
        "decision de usar el embedding como base?"
    ),
    _code(
        "# Cruce de las top SHAP con la seleccion de variables previa.\n"
        "fs_dir = Path('reports/feature_selection')\n"
        "top_shap = set(\n"
        "    shap_results['rf'].global_importance.head(20)['feature'].to_list()\n"
        ")\n"
        "\n"
        "fs_importance_path = fs_dir / 'feature_importance_rf.csv'\n"
        "if fs_importance_path.exists():\n"
        "    fs_importance = pl.read_csv(fs_importance_path)\n"
        "    fs_top = set(fs_importance.head(20)['feature'].to_list())\n"
        "    overlap = top_shap & fs_top\n"
        "    print(f'Solapamiento top-20 SHAP vs seleccion previa: '\n"
        "          f'{len(overlap)}/20 caracteristicas.')\n"
        "    print('Comunes:', sorted(overlap))\n"
        "    print('Solo en SHAP (revisar FE):', sorted(top_shap - fs_top))\n"
        "else:\n"
        "    print('reports/feature_selection/feature_importance_rf.csv '\n"
        "          'no disponible — se omite el cruce cuantitativo.')"
    ),
    _md(
        "### 5.1 Hallazgos\n"
        "\n"
        "Los numeros concretos del cruce salen de la celda anterior. Los "
        "hallazgos que cabe esperar:\n"
        "\n"
        "1. **Coincidencia entre la importancia simple y SHAP** — las "
        "caracteristicas en lo alto del ranking de Gini/gain y las del "
        "ranking SHAP coinciden en su mayoria; las discrepancias señalan "
        "variables con efectos no lineales o interacciones que SHAP "
        "captura mejor que la importancia simple.\n"
        "2. **Dominancia de AlphaEarth** — la fraccion de dimensiones del "
        "embedding (`dim_NN`) entre las 20 mas influyentes (seccion 4.1) "
        "indica cuanto del poder predictivo proviene del embedding: si "
        "dominan, aporta la mayor parte de la senal; si no, los indices "
        "espectrales y las estadisticas estacionales siguen siendo "
        "imprescindibles.\n"
        "3. **Validacion de la seleccion de variables** — si las "
        "caracteristicas seleccionadas en la fase previa coinciden con el "
        "top de SHAP, la seleccion queda validada; si una variable "
        "descartada aparece arriba, es una señal de que conviene "
        "revisarla.\n"
        "\n"
        "### 5.2 Recomendacion para la ingenieria de caracteristicas\n"
        "\n"
        "Si el cruce de la seccion 5 confirma la seleccion previa, **no se "
        "requiere ajuste**: la interpretabilidad del baseline la respalda. "
        "Si el cruce cuestiona alguna decision (una variable relevante "
        "descartada, o ruido retenido entre las mas influyentes), la "
        "recomendacion concreta se documenta para que las fases "
        "siguientes la incorporen antes de entrenar modelos mas "
        "complejos."
    ),
    # --- Section 5b (US-021) --------------------------------------------
    _md(
        "## 5b. Curvas de aprendizaje y validacion — diagnostico de "
        "sub/sobreajuste\n"
        "\n"
        "Esta seccion diagnostica si el baseline sub o sobreajusta. Se "
        "usan dos herramientas:\n"
        "\n"
        "- **Curva de aprendizaje**: accuracy de train y de validacion al "
        "crecer el numero de muestras de entrenamiento. Un gap grande "
        "train-val indica sobreajuste; ambas curvas bajas y juntas, "
        "subajuste.\n"
        "- **Curva de validacion**: accuracy frente a un hiperparametro "
        "critico (`max_depth` para RF, `n_estimators` y `learning_rate` "
        "para XGBoost), para localizar la zona de equilibrio.\n"
        "\n"
        "Toda la evaluacion usa el **mismo CV espacial 5-fold** (H3 + "
        "KMeans + buffer 1 km) del resto del notebook — los splits se "
        "materializan en una lista porque `learning_curve` reusa el `cv` "
        "una vez por cada tamano. El criterio de spatial CV esta "
        "documentado en `docs/spatial_cv_baseline.md`."
    ),
    _code(
        "from ml.eval.learning_curves import (\n"
        "    diagnose_fit,\n"
        "    plot_learning_curve,\n"
        "    plot_validation_curve,\n"
        ")\n"
        "from ml.train.baseline import _build_cv_splits\n"
        "\n"
        "# CV espacial materializado (lista de splits posicionales).\n"
        "cv_splits_5b = _build_cv_splits(\n"
        "    df, k_folds=5, buffer_km=1.0, random_state=42\n"
        ")\n"
        "print(f'CV espacial: {len(cv_splits_5b)} folds materializados')"
    ),
    _code(
        "# Curva de aprendizaje RF y XGB (accuracy train/val vs n muestras).\n"
        "from pathlib import Path\n"
        "\n"
        "from ml.train.baseline import build_estimator\n"
        "\n"
        "reports_dir = Path('reports/baseline')\n"
        "reports_dir.mkdir(parents=True, exist_ok=True)\n"
        "curve_train_sizes = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]\n"
        "learning_results = {}\n"
        "for kind in ('rf', 'xgb'):\n"
        "    estimator = build_estimator(kind, {})\n"
        "    lc_result, lc_fig = plot_learning_curve(\n"
        "        estimator, df, cv_splits_5b,\n"
        "        train_sizes=curve_train_sizes,\n"
        "        max_samples=MAX_SAMPLES,\n"
        "    )\n"
        "    learning_results[kind] = lc_result\n"
        "    # El titulo lo establece plot_learning_curve via ax.set_title;\n"
        "    # se actualiza in-place para incluir el modelo sin duplicar texto.\n"
        "    for ax in lc_fig.axes:\n"
        "        ax.set_title(f'Curva de aprendizaje — {kind.upper()} (accuracy)')\n"
        "    lc_fig.savefig(\n"
        "        reports_dir / f'learning_curve_{kind}.png',\n"
        "        dpi=200, bbox_inches='tight',\n"
        "    )\n"
        "    plt.show()"
    ),
    _code(
        "# Diagnostico explicito de sub/sobreajuste por modelo.\n"
        "for kind, lc_result in learning_results.items():\n"
        "    diag = diagnose_fit(lc_result)\n"
        "    print(f'{kind.upper()}: veredicto={diag.verdict}  '\n"
        "          f'gap={diag.gap:.4f}  '\n"
        "          f'train_acc={diag.train_acc_max:.4f}  '\n"
        "          f'val_acc={diag.val_acc_max:.4f}')\n"
        "    print(f'  {diag.explanation}')"
    ),
    _code(
        "# Curva de validacion RF — max_depth.\n"
        "vc_rf, vc_rf_fig = plot_validation_curve(\n"
        "    build_estimator('rf', {}), df, 'max_depth',\n"
        "    [5, 10, 15, 20, 30, None], cv_splits_5b,\n"
        "    max_samples=MAX_SAMPLES,\n"
        ")\n"
        "# Actualizamos el titulo del eje (set_title interno) en vez de\n"
        "# anadir un suptitle que se encimaria.\n"
        "for ax in vc_rf_fig.axes:\n"
        "    ax.set_title('Curva de validación — RF max_depth (accuracy)')\n"
        "vc_rf_fig.savefig(\n"
        "    reports_dir / 'validation_curve_rf_max_depth.png',\n"
        "    dpi=200, bbox_inches='tight',\n"
        ")\n"
        "plt.show()"
    ),
    _code(
        "# Curva de validacion XGB — n_estimators.\n"
        "vc_xgb_ne, vc_xgb_ne_fig = plot_validation_curve(\n"
        "    build_estimator('xgb', {}), df, 'n_estimators',\n"
        "    [100, 200, 300, 400, 500], cv_splits_5b,\n"
        "    max_samples=MAX_SAMPLES,\n"
        ")\n"
        "for ax in vc_xgb_ne_fig.axes:\n"
        "    ax.set_title('Curva de validación — XGB n_estimators (accuracy)')\n"
        "vc_xgb_ne_fig.savefig(\n"
        "    reports_dir / 'validation_curve_xgb_n_estimators.png',\n"
        "    dpi=200, bbox_inches='tight',\n"
        ")\n"
        "plt.show()"
    ),
    _md(
        "El diagnostico reporta un veredicto explicito (sobreajuste, "
        "subajuste o ajuste adecuado) con la diferencia numerica entre "
        "el desempeno en entrenamiento y en validacion. Un modelo de "
        "arboles sobre estas caracteristicas tiende a una exactitud "
        "modesta: si el veredicto es *ajuste adecuado* pero con exactitud "
        "de validacion baja, el limite es la **capacidad del modelo**, no "
        "el sobreajuste — esto justifica que las fases siguientes "
        "incorporen arquitecturas temporales con mayor capacidad."
    ),
    # --- Section 6 -------------------------------------------------------
    _md(
        "## 6. Desempeño del baseline\n"
        "\n"
        "Se define un umbral de referencia de **F1-macro >= 0.60** sobre "
        "PASTIS-R. Se entrenan Random Forest y XGBoost con validacion "
        "cruzada **espacial** (celdas hexagonales H3 + agrupamiento "
        "KMeans + zona de exclusion de 1 km, para que parcelas vecinas no "
        "queden a la vez en entrenamiento y validacion) y se reporta el "
        "promedio de cada metrica sobre los pliegues.\n"
        "\n"
        "Lo importante es que el desempeno quede **medido y explicado**: "
        "si el F1-macro no alcanza 0.60, la seccion 6.1 documenta las "
        "causas probables y las decisiones para las fases siguientes."
    ),
    _code(
        "from ml.train.baseline import train_one_model, tune_baseline\n"
        "\n"
        "results = {}\n"
        "for kind in ('rf', 'xgb'):\n"
        "    if TUNE:\n"
        "        best_params = tune_baseline(df, model=kind)\n"
        "        results[kind] = train_one_model(\n"
        "            df, model=kind, hyperparams=best_params\n"
        "        )\n"
        "    else:\n"
        "        results[kind] = train_one_model(df, model=kind)\n"
        "    print(f'{kind.upper()}  entrenado.')"
    ),
    _code(
        "# Tabla resumen de las metricas CV-mean por modelo.\n"
        "summary = pl.DataFrame(\n"
        "    [\n"
        "        {\n"
        "            'modelo': kind.upper(),\n"
        "            **{m: round(v, 4) for m, v in res.metrics.items()},\n"
        "        }\n"
        "        for kind, res in results.items()\n"
        "    ]\n"
        ")\n"
        "summary"
    ),
    _code(
        "# Veredicto frente al umbral de referencia.\n"
        "best_kind = max(results, key=lambda k: results[k].metrics['f1_macro'])\n"
        "best_f1 = results[best_kind].metrics['f1_macro']\n"
        "passed = best_f1 >= F1_THRESHOLD\n"
        "print(f'Mejor modelo: {best_kind.upper()}  |  F1-macro = {best_f1:.4f}')\n"
        "print(f'Umbral de referencia: {F1_THRESHOLD:.2f}  |  '\n"
        '      f\'{"alcanzado" if passed else "no alcanzado — ver 6.1"}\')'
    ),
    _md(
        "### 6.1 Causas probables y decisiones para las fases siguientes\n"
        "\n"
        "Si el F1-macro promedio queda por debajo de 0.60, las causas "
        "probables son:\n"
        "\n"
        "1. **Gran cantidad de clases (20 tipos de cultivo).** Varios "
        "cultivos son espectralmente parecidos; un modelo de arboles "
        "sobre un resumen anual no capta la firma estacional que los "
        "distingue.\n"
        "2. **Clases desbalanceadas.** Pese al balanceo aplicado, las "
        "clases minoritarias aportan pocas parcelas y el F1-macro las "
        "penaliza con fuerza.\n"
        "3. **Limite de un modelo de arboles sobre un resumen anual.** "
        "El embedding AlphaEarth condensa el ano en 64 dimensiones y "
        "pierde la dinamica intra-anual que un modelo de series "
        "temporales si aprovecha.\n"
        "\n"
        "Decisiones concretas para las fases siguientes:\n"
        "\n"
        "- Modelos que explotan la **serie temporal completa** de "
        "Sentinel-2 (no el resumen anual), capaces de captar la "
        "estacionalidad que separa cultivos parecidos.\n"
        "- **Combinar varios modelos** (de arboles, temporales y de "
        "lenguaje-vision) para recuperar senal complementaria que ningun "
        "modelo aislado captura."
    ),
    # --- Section 7 — Comparison of 3 scenarios (US-022) ------------------
    _md(
        "## 7. Comparativa AlphaEarth vs Sentinel-2 crudo\n"
        "\n"
        "Esta seccion compara el baseline sobre **tres vistas distintas de "
        "las mismas parcelas**, para responder con evidencia una pregunta "
        "central: ¿el embedding AlphaEarth aporta valor frente a las "
        "bandas Sentinel-2 sin procesar?\n"
        "\n"
        "| Escenario | Características | Origen |\n"
        "|-----------|-----------------|--------|\n"
        "| **(a) AlphaEarth** | 64 dimensiones | embedding AlphaEarth "
        "Foundations |\n"
        "| **(b) Sentinel-2 crudo** | 10 bandas promedio | bandas "
        "Sentinel-2 sin procesar, agregadas por parcela |\n"
        "| **(c) Vector combinado** | 187 caracteristicas | ingenieria "
        "de caracteristicas espectro-temporales |\n"
        "\n"
        "Metodología de la comparativa:\n"
        "\n"
        "- Los 3 escenarios se cruzan por parcela para evaluarse sobre "
        "**exactamente el mismo conjunto de parcelas**, no sobre tres "
        "muestras distintas.\n"
        "- Se reutiliza la **misma validacion cruzada espacial** para los "
        "3 escenarios; asi la diferencia de F1-macro refleja la calidad "
        "de las caracteristicas, no el azar de la particion.\n"
        "- Se reporta tambien el **tiempo de entrenamiento** de cada "
        "modelo.\n"
        "\n"
        "Si el escenario (b) Sentinel-2 crudo aun no se ha generado, esta "
        "seccion degrada de forma controlada y documenta la ausencia sin "
        "interrumpir el notebook."
    ),
    _code(
        "from pathlib import Path\n"
        "\n"
        "from ml.eval.comparison import (\n"
        "    build_comparison_table,\n"
        "    export_comparison_latex,\n"
        ")\n"
        "\n"
        "scenario_paths = {\n"
        "    'alphaearth': SCENARIO_ALPHAEARTH_PATH,\n"
        "    's2_raw': SCENARIO_S2_RAW_PATH,\n"
        "    'combined': SCENARIO_COMBINED_PATH,\n"
        "}\n"
        "missing = {\n"
        "    key: path\n"
        "    for key, path in scenario_paths.items()\n"
        "    if not Path(path).exists()\n"
        "}\n"
        "comparison_available = not missing\n"
        "if missing:\n"
        "    print('Escenarios no disponibles -> comparativa omitida:')\n"
        "    for key, path in missing.items():\n"
        "        print(f'  - {key}: {path}')\n"
        "    print('Genera el escenario (b) con `make s2-raw-parcels`.')\n"
        "else:\n"
        "    print('Los 3 escenarios estan disponibles para la comparativa.')"
    ),
    _code(
        "# Comparativa de los 3 escenarios (6 filas = 3 escenarios x 2 modelos).\n"
        "comparison_result = None\n"
        "if comparison_available:\n"
        "    comparison_result = build_comparison_table(\n"
        "        scenario_paths,\n"
        "        k_folds=COMPARISON_K_FOLDS,\n"
        "        max_samples=COMPARISON_MAX_SAMPLES,\n"
        "        random_state=42,\n"
        "    )\n"
        "    print(f'Parcelas en el inner join: '\n"
        "          f'{comparison_result.n_parcels:,}')\n"
        "    comparison_result.table\n"
        "else:\n"
        "    print('Comparativa omitida — ver celda anterior.')"
    ),
    _code(
        "# Persistencia de la tabla comparativa (CSV + MD + LaTeX).\n"
        "if comparison_result is not None:\n"
        "    reports_dir = Path('reports/baseline')\n"
        "    reports_dir.mkdir(parents=True, exist_ok=True)\n"
        "    comparison_result.table.write_csv(\n"
        "        reports_dir / 'comparison_alphaearth_vs_s2.csv'\n"
        "    )\n"
        "    md_table = (\n"
        "        '# Comparativa de escenarios — baseline de cultivos\\n\\n'\n"
        "        + comparison_result.table.to_pandas().to_markdown(index=False)\n"
        "        + '\\n'\n"
        "    )\n"
        "    (reports_dir / 'comparison_alphaearth_vs_s2.md').write_text(\n"
        "        md_table, encoding='utf-8'\n"
        "    )\n"
        "    tex_path = export_comparison_latex(\n"
        "        comparison_result, reports_dir / 'comparison_table.tex'\n"
        "    )\n"
        "    print(f'Tabla comparativa escrita: CSV + MD + {tex_path.name}')\n"
        "else:\n"
        "    print('Sin tabla comparativa que persistir.')"
    ),
    _code(
        "# Barplot comparativo de F1-macro por escenario y modelo.\n"
        "if comparison_result is not None:\n"
        "    table = comparison_result.table\n"
        "    scenarios = table['scenario'].unique(maintain_order=True).to_list()\n"
        "    x = range(len(scenarios))\n"
        "    width = 0.38\n"
        "    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)\n"
        "    for offset, model in zip((-width / 2, width / 2), ('RF', 'XGB')):\n"
        "        f1_by_scenario = [\n"
        "            float(\n"
        "                table.filter(\n"
        "                    (pl.col('scenario') == sc)\n"
        "                    & (pl.col('model') == model)\n"
        "                )['f1_macro'][0]\n"
        "            )\n"
        "            for sc in scenarios\n"
        "        ]\n"
        "        bars = ax.bar(\n"
        "            [xi + offset for xi in x], f1_by_scenario,\n"
        "            width=width, label=model,\n"
        "        )\n"
        "        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)\n"
        "    ax.set_xticks(list(x))\n"
        "    ax.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)\n"
        "    ax.set_ylabel('F1-macro (CV espacial out-of-fold)')\n"
        "    ax.set_ylim(0.0, 1.0)\n"
        "    ax.set_title('Comparativa del baseline — 3 escenarios de características')\n"
        "    ax.legend(title='Modelo')\n"
        "    ax.grid(axis='y', alpha=0.3)\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(\n"
        "        Path('reports/baseline') / 'comparison_barplot.png',\n"
        "        dpi=200, bbox_inches='tight',\n"
        "    )\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Sin barplot — comparativa omitida.')"
    ),
    _code(
        "# Resumen cuantitativo del valor incremental de AlphaEarth.\n"
        "if comparison_result is not None:\n"
        "    delta = comparison_result.alphaearth_delta\n"
        "    print(f'Escenario ganador: {comparison_result.best_scenario}')\n"
        "    print(f'Delta F1-macro AlphaEarth - Sentinel-2 crudo: '\n"
        "          f'{delta:+.4f}')\n"
        "    if delta > 0.0:\n"
        "        print('-> El embedding AlphaEarth aporta valor incremental '\n"
        "              'sobre las bandas crudas.')\n"
        "    else:\n"
        "        print('-> El embedding AlphaEarth NO supera a las bandas '\n"
        "              'crudas en este baseline tabular.')\n"
        "else:\n"
        "    print('Sin delta — comparativa omitida.')"
    ),
    # --- Section 8 — Conclusions ----------------------------------------
    _md(
        "## 8. Conclusiones\n"
        "\n"
        "Este notebook construyo un punto de referencia para clasificar "
        "cultivos a partir de imagenes satelitales y lo sometio a tres "
        "preguntas: ¿que tan bien funciona un modelo de arboles sencillo?, "
        "¿que caracteristicas explican sus predicciones?, y ¿el embedding "
        "AlphaEarth aporta algo frente a las bandas satelitales sin "
        "procesar? Lo que encontramos:\n"
        "\n"
        "### ¿AlphaEarth aporta valor?\n"
        "\n"
        "La comparativa de la seccion 7 da una respuesta con datos. El "
        "**embedding AlphaEarth** es una representacion compacta de 64 "
        "numeros que resume un ano de observaciones satelitales; las "
        "**bandas Sentinel-2 crudas** son los 10 canales del satelite "
        "promediados. La diferencia de F1-macro entre ambos escenarios "
        "indica si ese resumen aprendido aporta informacion que el "
        "promedio simple de las bandas pierde.\n"
        "\n"
        "- Si AlphaEarth supera a las bandas crudas, el resumen aprendido "
        "captura senal multisensor y estacional que el promedio destruye.\n"
        "- Si quedan empatados, ambas representaciones son equivalentes "
        "para un modelo de arboles a nivel de parcela.\n"
        "- Si las bandas crudas ganan, el problema no esta en la "
        "representacion sino en haber promediado el tiempo: la solucion "
        "es usar la serie temporal completa.\n"
        "\n"
        "### Hallazgos\n"
        "\n"
        "1. **El techo de este modelo es estructural, no de ajuste.** Las "
        "curvas de aprendizaje (seccion 5b) muestran que el modelo no "
        "sobreajusta: simplemente ha llegado a su capacidad maxima sobre "
        "datos que ya perdieron la dimension temporal. Anadir mas arboles "
        "o mas profundidad no movera ese techo.\n"
        "2. **La representacion de los datos importa mas que el "
        "algoritmo.** Random Forest y XGBoost rinden parecido dentro de "
        "cada escenario; la diferencia grande de desempeno aparece "
        "**entre escenarios**. La pregunta clave no es que clasificador "
        "usar, sino como representar la evolucion del cultivo en el "
        "tiempo.\n"
        "3. **Promediar el tiempo es el cuello de botella.** Los tres "
        "escenarios resumen el ano en un solo vector. Pero cultivos "
        "espectralmente parecidos solo se distinguen por **como cambian a "
        "lo largo de la temporada** — y esa trayectoria se pierde al "
        "promediar.\n"
        "\n"
        "### Lo que sigue\n"
        "\n"
        "- **Modelos que usen la serie temporal completa.** Este baseline "
        "fija el piso de desempeno; los modelos siguientes deben procesar "
        "la secuencia de imagenes Sentinel-2 mes a mes — no su promedio "
        "anual — para captar la estacionalidad que separa cultivos "
        "parecidos. Si aun asi no superan estas cifras, el limite estaria "
        "en los datos, no en el modelo.\n"
        "- **AlphaEarth como caracteristica de apoyo.** El embedding se "
        "incorporara como una entrada mas al combinar varios modelos, no "
        "como sustituto de la serie temporal cruda.\n"
        "- **Mismo protocolo de evaluacion.** La validacion cruzada "
        "espacial con zona de exclusion entre parcelas vecinas se "
        "mantiene en las fases siguientes, para que las cifras sean "
        "comparables entre experimentos.\n"
        "\n"
        "El baseline cumple su proposito: es **honesto, interpretable y "
        "reproducible** — establece el piso de desempeno, documenta sus "
        "propias limitaciones y deja un protocolo de evaluacion y una "
        "metrica principal que el resto del proyecto puede heredar."
    ),
    # ----------------------------------------------------------------------
    # NOTE US-023-preview run 3: section 9 (Baseline v2 with 3 models)
    # was moved to the standalone notebook `notebooks/baseline/04b_baseline_v2.ipynb`,
    # which reads the artifacts persisted in `reports/baseline/model_comparison_v2/`.
    # The removed cells retrained heavy models (~90 min CUDA) and left
    # notebook 04 with conditional MLflow/CUDA dependencies. The real training
    # now lives in `scripts/run_baseline_v2_standalone.py`.
    # ----------------------------------------------------------------------
]


# Internal stub: the cells of the former section 9 (Baseline v2 with 3 canonical
# models, US-023-preview P8) were moved to the standalone notebook
# `notebooks/baseline/04b_baseline_v2.ipynb` in run 3 (2026-05-26).
# The conditional block (`RUN_BASELINE_V2`) and its training cells were
# removed from the builder; the reader finds the real v2 metrics in
# `reports/baseline/model_comparison_v2/model_comparison_v2.parquet` and the
# training is regenerated with `scripts/run_baseline_v2_standalone.py` or the
# target `make baseline-v2-full`.

_REMOVED_V2_PARAMS_DROPPED = (
    "RUN_BASELINE_V2, V2_MAX_SAMPLES, V2_K_FOLDS, V2_BUFFER_KM, V2_SEED, "
    "V2_TEMPORAL_EPOCHS, V2_TEMPORAL_BATCH_SIZE, V2_DEVICE, "
    "V2_FEATURE_ABLATION_PATH, V2_OUTPUT_DIR"
)


def build_notebook(out_path: Path) -> None:
    """Build the baseline notebook and write it to ``out_path``.

    Args:
        out_path: Destination path of the ``.ipynb`` file.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))


@app.command()
def main(
    out: Path = typer.Option(
        Path("notebooks/baseline/04_baseline.ipynb"),
        help="Ruta destino del notebook .ipynb.",
    ),
) -> None:
    """Rebuild ``notebooks/04_baseline.ipynb`` from scratch."""
    build_notebook(out)
    typer.echo(f"Notebook escrito en {out}")


if __name__ == "__main__":
    app()
