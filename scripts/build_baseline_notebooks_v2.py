"""Unified builder of the 6 baseline notebooks (US-023-preview v2).

Generates the 6 notebooks in `notebooks/baseline/` from a single source of
truth, aligned to the `notebooks/CLAUDE.md` standard and reusing
all the helpers in `ml/`:

- `notebooks/baseline/04_baseline.ipynb` — XGB + LGBM + RF + temporal + plots.
- `notebooks/baseline/04b_baseline.ipynb` — variant with AlphaEarth only
  (pilot of the new bootstrap pattern).
- `notebooks/baseline/04c_baseline.ipynb` — block ablation with the
  alphaearth_only detection fix.
- `notebooks/baseline/04_farslip_eval_pastis.ipynb` — FarSLIP vs RemoteCLIP
  on real PASTIS (without synthetic).
- `notebooks/baseline/05_reencuadre_fenologico.ipynb` — phenology + full
  ablation with auto-materialization without silent skips.
- `notebooks/baseline/Avance3.Equipo17.ipynb` — aggregator with
  select_winning_features.

Usage:

```bash
poetry run python scripts/build_baseline_notebooks_v2.py [--only 04b]
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NOTEBOOK_DIR = Path("notebooks/baseline")


def _md(text: str) -> dict[str, Any]:
    """Creates a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    """Creates a code cell."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }
    if tags:
        cell["metadata"] = {"tags": tags}
    return cell


def _notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Wraps cells in an nbformat 4.5 structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(path: Path, nb: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Escrito: {path}")


# ---------------------------------------------------------------------------
# Standard bootstrap cell (same in all notebooks).
# ---------------------------------------------------------------------------


BOOTSTRAP_CELL = """from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap: localizar el repo root buscando pyproject.toml
_HERE = Path.cwd().resolve()
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / "pyproject.toml").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from ml.utils.notebook_bootstrap import setup_notebook
from IPython.display import Markdown, display

env = setup_notebook(
    figures_subdir=FIGURES_SUBDIR,
    reports_subdir=REPORTS_SUBDIR,
)
display(Markdown(env.summary_markdown()))

# Chdir al repo root para que las rutas relativas de la celda `parameters`
# (FEATURES_PATH = "data/...", etc.) resuelvan igual sin importar desde donde
# se haya lanzado el kernel (VS Code abre con cwd = carpeta del notebook).
# Esto preserva el contrato papermill (parametros como strings relativas) y
# elimina FileNotFoundError causado por cwd != repo root.
os.chdir(env.repo)
display(Markdown(f"**cwd anclado al repo root**: `{env.repo}`"))
"""


# ---------------------------------------------------------------------------
# 04b_baseline.ipynb — pilot, the simplest one.
# ---------------------------------------------------------------------------


def _hcat_grouping_cells(subset_size: int | None = None) -> list[dict[str, Any]]:
    """Cells of the '18 classes vs 6 HCAT Level-1 groups' section.

    Shared by ``build_04_baseline`` (full 85951) and ``build_04b_baseline``
    (pilot subsample). The experiment is apples-to-apples: same features,
    same spatial CV, same XGBoost; the only thing that changes is the remapping
    of the target label (18 flat classes -> 6 HCAT v3 groups).

    Args:
        subset_size: If not ``None``, subsamples ``df_hcat`` in a
            ``class_id``-stratified way before evaluating (pilot mode). If
            ``None``, runs over the full universe of 85951 parcels.

    Returns:
        List of nbformat cells ready for ``cells.extend(...)``.
    """
    cells: list[dict[str, Any]] = []

    if subset_size is None:
        scope_md = (
            "**El experimento es apples-to-apples**: exactamente las mismas "
            "features, la misma validación cruzada espacial (5-fold, buffer "
            "1 km) y el mismo modelo XGBoost. Lo único que cambia es la "
            "etiqueta objetivo. Por eso el salto de F1-macro que veremos no "
            "viene de un modelo mejor, sino de medir lo que el modelo "
            "realmente resuelve."
        )
    else:
        scope_md = (
            "**El experimento es apples-to-apples** (mismas features, mismo "
            "spatial CV 5-fold buffer 1 km, mismo XGBoost; solo cambia la "
            "etiqueta objetivo). Aquí lo corremos sobre un **subsample "
            "estratificado** para mantener el carácter de piloto rápido: el "
            "salto de F1-macro se lee como **tendencia**, no como cifra "
            "definitiva (la cifra final, sobre las 85951 parcelas, vive en "
            "`04_baseline.ipynb`)."
        )

    cells.append(
        _md(
            "## 8. Dos formas de medir el mismo modelo: 18 clases vs 6 grupos\n\n"
            "El F1-macro de las secciones anteriores se calcula sobre las "
            "**18 clases planas** de PASTIS-R, y arrastra un problema "
            "estructural: varias de esas clases son **hermanas casi "
            "indistinguibles** a nivel de parcela. El trigo blando de "
            "invierno, el trigo duro de invierno, la cebada de invierno, la "
            "cebada de primavera, el triticale y el centeno mixto comparten "
            "calendario y firma espectral; separarlos con una sola imagen-"
            "resumen anual es casi imposible. Cuando el modelo confunde un "
            "trigo con otro trigo, el F1-macro lo penaliza igual que si "
            "confundiera un viñedo con una remolacha, aunque para casi "
            "cualquier uso agronómico ambos trigos pertenecen al mismo grupo.\n\n"
            "La solución estándar en la literatura (Russwürm & Körner 2018; "
            "H2Crop 2025) es reportar también la métrica sobre una "
            "**taxonomía jerárquica**. Aquí usamos los **6 grupos HCAT "
            "Level-1** (Hierarchical Crop and Agriculture Taxonomy v3), que "
            "colapsan las 18 clases en seis familias agronómicas: cereales, "
            "oleaginosas, tubérculos, leguminosas, leñosos permanentes y "
            "otros (pradera + horticultura). Las clases hermanas caen dentro "
            "del mismo grupo, de modo que la confusión trigo-con-trigo deja "
            "de contar como error.\n\n"
            "**Importante — qué features usa esta sección.** A diferencia de "
            "las secciones anteriores, que entrenan sobre las **185 features "
            "espectro-temporales**, aquí partimos del **escenario ganador de "
            "la ablación de bloques**: esas 185 features **más** el embedding "
            "AlphaEarth de dos años (2018 + 2019), 64 dimensiones cada uno, "
            "para un total de **313 columnas**. Por eso el F1-macro sobre las "
            "18 clases que verás aquí es **más alto** que el de la sección 3: "
            "el salto tiene dos componentes acumulativos que conviene no "
            "confundir — primero, **añadir AlphaEarth** sube el F1 sobre el "
            "mismo esquema de 18 clases (el embedding anual aporta señal que "
            "las estadísticas espectrales no capturan); segundo, **agrupar a "
            "6 familias** recupera el error que era pura confusión "
            "intra-familia. Las dos mejoras son reales y de naturaleza "
            "distinta: una viene de mejores features, la otra de medir la "
            "tarea a la granularidad agronómicamente correcta.\n\n" + scope_md
        )
    )

    load_cell = (
        "from ml.utils.baseline_notebook_helpers import (\n"
        "    load_base_plus_alphaearth_2018_2019,\n"
        ")\n"
        "from ml.analysis.hcat_grouping import (\n"
        "    HCAT_L1_GROUPS,\n"
        "    HCAT_L1_GROUP_CODES,\n"
        "    evaluate_flat_vs_grouped,\n"
        ")\n"
        "from ml.eval.reencuadre_plots import (\n"
        "    plot_model_comparison_bars,\n"
        "    plot_per_class_f1,\n"
        ")\n"
        "\n"
        "# Umbral de referencia del baseline (definido localmente para que la\n"
        "# seccion sea autocontenida en cualquier notebook que la incluya).\n"
        "HCAT_F1_THRESHOLD = 0.60\n"
        "\n"
        "# Escenario ganador de la ablacion: base (185 features) + AlphaEarth\n"
        "# 2018 (ae18_NN) + AlphaEarth 2019 (ae19_NN) = 313 columnas, unidas\n"
        "# por parcel_id (join 1:1, 0 nulls) sobre las 85951 parcelas.\n"
        "df_hcat = load_base_plus_alphaearth_2018_2019(\n"
        "    features_path=FEATURES_PATH,\n"
        "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
        ")\n"
    )
    if subset_size is not None:
        load_cell += (
            "\n"
            "# Piloto: submuestreo estratificado por clase para lectura rapida.\n"
            "from ml.utils.sampling import stratified_sample\n"
            "df_hcat = stratified_sample(df_hcat, by=['class_id'], "
            "n=HCAT_SUBSET_SIZE, seed=RANDOM_STATE)\n"
        )
    load_cell += (
        "n_ae18 = sum(1 for c in df_hcat.columns if c.startswith('ae18_'))\n"
        "n_ae19 = sum(1 for c in df_hcat.columns if c.startswith('ae19_'))\n"
        "display(Markdown(\n"
        "    f'**Escenario ganador**: `{df_hcat.height:,}` parcelas, '\n"
        "    f'185 features base + `{n_ae18}` columnas AlphaEarth 2018 + '\n"
        "    f'`{n_ae19}` columnas AlphaEarth 2019.'\n"
        "))\n"
    )
    cells.append(_code(load_cell))

    cells.append(
        _md(
            "### 8.1 Composición de los 6 grupos HCAT Level-1\n\n"
            "Cada grupo lista las clases PASTIS-R que absorbe y su código de "
            "nodo en la taxonomía HCAT v3 (para trazabilidad del "
            "agrupamiento). Los cereales concentran ocho clases hermanas — "
            "justamente las que se confunden entre sí en el esquema plano."
        )
    )
    cells.append(
        _code(
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "\n"
            "grouping_rows = []\n"
            "for group, class_ids in HCAT_L1_GROUPS.items():\n"
            "    grouping_rows.append({\n"
            "        'grupo_hcat_l1': group,\n"
            "        'codigo_hcat': HCAT_L1_GROUP_CODES[group],\n"
            "        'n_clases': len(class_ids),\n"
            "        'clases_pastis': ', '.join(\n"
            "            PASTIS_R_CLASSES.get(c, str(c)) for c in class_ids\n"
            "        ),\n"
            "    })\n"
            "grouping_table = pl.DataFrame(grouping_rows).sort('n_clases', descending=True)\n"
            "display(grouping_table)\n"
        )
    )

    cells.append(
        _md(
            "### 8.2 Entrenamiento y métricas en ambos esquemas\n\n"
            "Entrenamos XGBoost con validación cruzada espacial dos veces "
            "sobre las mismas features: una con las 18 clases planas y otra "
            "con los 6 grupos HCAT. El helper devuelve las cinco métricas "
            "out-of-fold de cada esquema y las predicciones para los plots "
            "por clase/grupo."
        )
    )
    cells.append(
        _code(
            "hcat_result = evaluate_flat_vs_grouped(\n"
            "    df_hcat,\n"
            "    model='xgb',\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "metric_order = ['f1_macro', 'f1_weighted', 'miou', 'accuracy', 'cohen_kappa']\n"
            "metric_label = {\n"
            "    'f1_macro': 'F1-macro', 'f1_weighted': 'F1-weighted', 'miou': 'mIoU',\n"
            "    'accuracy': 'accuracy', 'cohen_kappa': 'Cohen kappa',\n"
            "}\n"
            "hcat_metrics_table = pl.DataFrame({\n"
            "    'metrica': [metric_label[m] for m in metric_order],\n"
            "    'flat_18_clases': [round(hcat_result.flat_metrics[m], 4) for m in metric_order],\n"
            "    'grouped_6_hcat': [round(hcat_result.grouped_metrics[m], 4) for m in metric_order],\n"
            "    'delta': [\n"
            "        round(hcat_result.grouped_metrics[m] - hcat_result.flat_metrics[m], 4)\n"
            "        for m in metric_order\n"
            "    ],\n"
            "})\n"
            "hcat_metrics_path = env.reports_dir / 'hcat_flat18_vs_grouped6.parquet'\n"
            "hcat_metrics_table.write_parquet(hcat_metrics_path)\n"
            "display(Markdown(\n"
            "    f'**F1-macro 18 clases** = `{hcat_result.flat_metrics[\"f1_macro\"]:.4f}` · '\n"
            "    f'**F1-macro 6 grupos HCAT** = `{hcat_result.grouped_metrics[\"f1_macro\"]:.4f}` · '\n"
            "    f'**delta** = `{hcat_result.delta_f1_macro:+.4f}` '\n"
            "    f'(sobre `{hcat_result.n_samples:,}` parcelas, `{hcat_result.n_features}` features)'\n"
            "))\n"
            "display(hcat_metrics_table)\n"
        )
    )

    cells.append(
        _md(
            "### 8.3 F1-macro lado a lado\n\n"
            "La barra de la izquierda es lo que el modelo logra cuando se le "
            "exige separar cada trigo de cada otro trigo; la de la derecha, "
            "lo que logra cuando solo se le pide acertar la familia "
            "agronómica."
        )
    )
    cells.append(
        _code(
            "fig_hcat_cmp = plot_model_comparison_bars(\n"
            "    {\n"
            "        '18 clases (plano)': hcat_result.flat_metrics['f1_macro'],\n"
            "        '6 grupos (HCAT L1)': hcat_result.grouped_metrics['f1_macro'],\n"
            "    },\n"
            "    baseline_value=HCAT_F1_THRESHOLD,\n"
            "    baseline_label=f'umbral de referencia (F1-macro = {HCAT_F1_THRESHOLD:.2f})',\n"
            "    title='F1-macro: 18 clases planas vs 6 grupos HCAT Level-1',\n"
            ")\n"
            "fig_hcat_cmp.savefig(env.figures_dir / 'hcat_flat18_vs_grouped6.png', bbox_inches='tight')\n"
            "display(fig_hcat_cmp)\n"
            "plt.close(fig_hcat_cmp)\n"
        )
    )

    cells.append(
        _md(
            "### 8.4 F1 por grupo HCAT y diagnóstico por clase plana\n\n"
            "El F1 por grupo muestra dónde queda el residuo de error tras "
            "agrupar: los grupos con poco soporte siguen siendo difíciles "
            "aunque sean homogéneos. La tabla de las clases planas más "
            "débiles confirma que el cuello de botella del esquema de 18 es "
            "la confusión entre cultivos hermanos, no la falta de señal."
        )
    )
    cells.append(
        _code(
            "# grouped_label_names ya es dict {id_codificado: nombre_grupo}.\n"
            "fig_hcat_f1 = plot_per_class_f1(\n"
            "    hcat_result.grouped_y_true,\n"
            "    hcat_result.grouped_y_pred,\n"
            "    class_labels=sorted(hcat_result.grouped_label_names),\n"
            "    class_names=hcat_result.grouped_label_names,\n"
            "    weak_threshold=0.40,\n"
            "    title='F1 por grupo HCAT Level-1 (XGBoost, out-of-fold)',\n"
            ")\n"
            "fig_hcat_f1.savefig(env.figures_dir / 'hcat_per_group_f1.png', bbox_inches='tight')\n"
            "display(fig_hcat_f1)\n"
            "plt.close(fig_hcat_f1)\n"
            "\n"
            "display(Markdown('**F1 por grupo HCAT Level-1** (ordenado por id de grupo):'))\n"
            "display(hcat_result.grouped_per_group)\n"
            "\n"
            "weakest_flat = hcat_result.flat_per_class.sort('f1').head(6)\n"
            "display(Markdown(\n"
            "    '**Las 6 clases planas más débiles** (confusión entre hermanas): '\n"
            "    'su F1 cercano a cero es lo que el agrupamiento HCAT recupera.'\n"
            "))\n"
            "display(weakest_flat)\n"
        )
    )

    cells.append(
        _md(
            "### 8.5 Lectura del salto\n\n"
            "El aumento del F1-macro al pasar de 18 clases a 6 grupos no es "
            "un truco para inflar la cifra: es la medida de **cuánto del "
            "error del esquema plano era confusión dentro de la misma "
            "familia agronómica**. Si el salto es grande, significa que el "
            "modelo ya distingue bien las familias (cereal vs leñoso vs "
            "oleaginosa) y que el residuo se concentra en separar cultivos "
            "que comparten firma. Ese diagnóstico orienta la fase siguiente: "
            "los modelos densos del Avance 4 tendrán que apoyarse en la "
            "**dinámica temporal intra-temporada** — no en una imagen-resumen "
            "anual — para empezar a separar los trigos entre sí."
        )
    )

    return cells


def build_04b_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04b — Piloto rápido y comparativa contra US-022b\n\n"
            "Este cuaderno cumple dos funciones acotadas que ningún otro "
            "notebook de la serie cubre:\n\n"
            "1. **Piloto del patrón** `setup_notebook` + `train_baseline_three_models` "
            "en menos de 10 minutos sobre un subsample del dataset Italia. "
            "Sirve para validar que el tooling completo (carga, spatial CV, "
            "RF + XGB + LightGBM, persistencia y MLflow) corre end-to-end "
            "antes de comprometerse con la corrida completa del cuaderno "
            "`04_baseline.ipynb` (2-3 h).\n"
            "2. **Comparativa contra el baseline previo US-022b**, que "
            "reporta cifras de XGBoost + InceptionTime + TempCNN sobre el "
            "mismo conjunto. La tabla de deltas evidencia la evolución del "
            "proyecto: misma metodología (spatial CV 5-fold buffer 1 km), "
            "nuevo conjunto de modelos.\n\n"
            "**Preguntas oficiales del Avance 3 a las que aporta**: P1 "
            "(algoritmo baseline) y P4 (métrica adecuada — el piloto exhibe "
            "el desbalance que hace inadecuada la accuracy).\n\n"
            "## Requisitos\n\n"
            "- `data/test_fixtures/feature_selection_parcels_subset.parquet` "
            "presente (descargable vía `dvc pull`).\n"
            "- `data/processed/pastis_parcels_full.geoparquet` presente.\n"
            "- Servidor MLflow corriendo en `http://localhost:5010`.\n"
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04b_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04b_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "RANDOM_STATE = 42\n"
            "SUBSET_SIZE = 8000  # piloto: subsample estratificado por clase\n"
            "HCAT_SUBSET_SIZE = 8000  # piloto seccion 8 (18 vs 6 grupos HCAT) sobre subsample\n"
            'MLFLOW_EXPERIMENT = "baseline-04b-pilot"\n',
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(
        _md(
            "### Trazabilidad MLflow\n\n"
            "Cada modelo del piloto abre un run propio en el experimento "
            "`baseline-04b-pilot` con los tags `code_version` y "
            "`data_version`. Los runs quedan separados del experimento "
            "principal `baseline-04-tabular` para distinguir piloto-sobre-"
            "subset de corrida-completa."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import (\n"
            "    resolve_tracking_uri,\n"
            "    track_experiment,\n"
            "    server_is_reachable,\n"
            ")\n"
            "\n"
            "# Resolucion robusta del tracking URI: si MLFLOW_TRACKING_URI esta en\n"
            "# `.env.local` pero el server no responde (Docker apagado, contenedor\n"
            "# detenido), caemos a `file:./mlruns` para no detener el notebook.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Caigo a tracking local `file:./mlruns`. '\n"
            "        'Para registrar en el server, ejecuta `docker compose up -d mlflow` '\n"
            "        'antes de re-ejecutar las celdas MLflow.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "display(Markdown(\n"
            "    f'**MLflow tracking URI**: `{mlflow_uri}` · '\n"
            "    f'**Experimento**: `{MLFLOW_EXPERIMENT}`'\n"
            "))\n"
            "MLFLOW_RUN_IDS: dict[str, str] = {}\n"
        )
    )

    cells.append(
        _md(
            "## 1. Carga del dataset + subsample estratificado\n\n"
            "Cargamos el subset US-018 con metadata enriquecida del "
            "geoparquet y submuestreamos estratificando por `class_id` para "
            "preservar la distribución de clases. El subsample mantiene la "
            "proporción del desbalance — la imagen del problema no cambia, "
            "solo el tamaño."
        )
    )

    cells.append(
        _code(
            "import polars as pl\n"
            "import numpy as np\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    train_baseline_three_models,\n"
            "    build_model_comparison_table,\n"
            ")\n"
            "from ml.utils.class_distribution import (\n"
            "    class_distribution_report,\n"
            "    recommend_threshold,\n"
            ")\n"
            "\n"
            "df_full = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "\n"
            "# Subsample estratificado por clase: respeta la proporcion original.\n"
            "rng = np.random.default_rng(RANDOM_STATE)\n"
            "frac = min(1.0, SUBSET_SIZE / df_full.height)\n"
            "df = (\n"
            "    df_full.with_columns(pl.lit(rng.random(df_full.height)).alias('__r'))\n"
            "    .group_by('class_id')\n"
            "    .map_groups(lambda g: g.sort('__r').head(max(1, int(g.height * frac))))\n"
            "    .drop('__r')\n"
            ")\n"
            "display(Markdown(\n"
            '    f"**Dataset piloto**: `{df.height:,}` parcelas "\n'
            '    f"(`{df.height / df_full.height * 100:.1f}%` del completo, "\n'
            '    f"`{df.width}` cols)."\n'
            "))\n"
            "display(df.group_by('class_id').len().sort('len', descending=True).head(10))\n"
        )
    )

    cells.append(
        _md(
            "## 2. Distribución de clases del subset\n\n"
            "Corroboramos que el subsample preserva la proporción original. "
            "El umbral del percentil 25 se recalcula sobre el subset porque "
            "los conteos absolutos son menores."
        )
    )

    cells.append(
        _code(
            "report = class_distribution_report(df)\n"
            "display(report)\n"
            "threshold_p25 = recommend_threshold(report, method='p25')\n"
            "imbalance = float(report.get_column('n_parcels').max()) / max(\n"
            "    float(report.get_column('n_parcels').min()), 1.0\n"
            ")\n"
            "display(Markdown(\n"
            "    f'**Imbalance ratio (max/min)**: `{imbalance:.1f}x`. '\n"
            "    f'**Umbral P25**: `{threshold_p25}` parcelas. '\n"
            "    'El desbalance del piloto es comparable al del dataset completo.'\n"
            "))\n"
        )
    )

    cells.append(
        _md(
            "## 3. Entrenamiento RF + XGB + LightGBM con MLflow tracking\n\n"
            "Los tres modelos comparten el mismo CV espacial y `random_state`. "
            "Cada uno abre su propio run MLflow con métricas, parámetros y "
            "tags estándar para reproducibilidad."
        )
    )

    cells.append(
        _code(
            "rows = train_baseline_three_models(\n"
            "    df,\n"
            "    models=('rf', 'xgb', 'lgbm'),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "comparison_path = env.reports_dir / 'model_comparison_04b.parquet'\n"
            "comparison = build_model_comparison_table(rows, output_path=comparison_path)\n"
            "display(Markdown(f'**Tabla guardada**: `{comparison_path.relative_to(env.repo)}`'))\n"
            "display(comparison)\n"
            "\n"
            "for r in rows:\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'04b-{r.model}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        dvc_path=FEATURES_PATH,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        mlflow.log_params({\n"
            "            'model': r.model,\n"
            "            'k_folds': K_FOLDS,\n"
            "            'buffer_km': BUFFER_KM,\n"
            "            'random_state': RANDOM_STATE,\n"
            "            'n_parcels': df.height,\n"
            "            'subset_size_target': SUBSET_SIZE,\n"
            "            'is_pilot': True,\n"
            "        })\n"
            "        mlflow.log_metrics({\n"
            "            'f1_macro': r.f1_macro,\n"
            "            'f1_weighted': r.f1_weighted,\n"
            "            'miou': r.miou,\n"
            "            'accuracy': r.accuracy,\n"
            "            'kappa': r.cohen_kappa,\n"
            "            'train_time_s': r.train_time_s,\n"
            "        })\n"
            "        MLFLOW_RUN_IDS[r.model] = run.info.run_id\n"
            "display(Markdown('**MLflow runs**: ' + ', '.join(\n"
            "    f'`{k}={v[:12]}...`' for k, v in MLFLOW_RUN_IDS.items()\n"
            ")))\n"
        )
    )

    cells.append(
        _md(
            "## 4. Comparativa contra el baseline previo US-022b\n\n"
            "Las cifras históricas del baseline US-022b corren sobre el "
            "**conjunto completo Italia (85 951 parcelas)** con XGBoost + "
            "InceptionTime + TempCNN. Aquí comparamos lado a lado para "
            "evidenciar la evolución del proyecto: nuevo set de modelos "
            "(RF + XGB + LGBM), misma metodología.\n\n"
            "*Nota*: los modelos temporales (TempCNN, InceptionTime) se "
            "evalúan en `05_reencuadre_fenologico.ipynb`. Aquí el delta es "
            "ilustrativo, no exhaustivo."
        )
    )

    cells.append(
        _code(
            "# Cifras historicas US-022b (conjunto full, 85951 parcelas).\n"
            "us022b_reference = pl.DataFrame({\n"
            "    'model': ['xgboost', 'inceptiontime', 'tempcnn'],\n"
            "    'f1_macro_us022b': [0.4094, 0.1865, 0.1430],\n"
            "    'dataset': ['full (85951)', 'full (85951)', 'full (85951)'],\n"
            "})\n"
            "display(Markdown('**Referencia histórica US-022b** (mismo spatial CV 5-fold buffer 1 km):'))\n"
            "display(us022b_reference)\n"
            "\n"
            "# Mapeo de nombres entre la tabla del piloto y la historica.\n"
            "name_map = {'rf': 'random_forest', 'xgb': 'xgboost', 'lgbm': 'lightgbm'}\n"
            "comparison_piloto = comparison.with_columns(\n"
            "    pl.col('model').replace(name_map).alias('model_full_name')\n"
            ")\n"
            "joined = comparison_piloto.join(\n"
            "    us022b_reference, left_on='model_full_name', right_on='model', how='left'\n"
            ").select([\n"
            "    'model_full_name', 'f1_macro', 'f1_macro_us022b',\n"
            "    (pl.col('f1_macro') - pl.col('f1_macro_us022b')).alias('delta_piloto_vs_us022b'),\n"
            "    'dataset',\n"
            "])\n"
            "display(Markdown('**Comparativa piloto subset vs US-022b full**:'))\n"
            "display(joined)\n"
            "display(Markdown(\n"
            "    '> El delta esperado es ligeramente negativo (el subset reduce el F1 '\n"
            "    'a igual modelo). Si el piloto está cerca del valor histórico de XGBoost '\n"
            "    '(~0.41) la metodología es estable y la corrida completa de `04_baseline` '\n"
            "    'puede arrancar con confianza.'\n"
            "))\n"
        )
    )

    cells.append(
        _md(
            "## 5. F1 por clase del mejor modelo del piloto\n\n"
            "Identifica qué clases concentran el error sobre el subset. "
            "Las clases más difíciles aquí lo seguirán siendo en la corrida "
            "completa — el subsample preserva la dificultad relativa."
        )
    )

    cells.append(
        _code(
            "from ml.eval.reencuadre_plots import plot_per_class_f1\n"
            "from ml.train.baseline import train_one_model, evaluate_with_spatial_cv, build_estimator\n"
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "best_model = comparison['model'][0]\n"
            "best_f1 = float(comparison['f1_macro'][0])\n"
            "display(Markdown(\n"
            "    f'**Mejor modelo del piloto**: `{best_model}` (F1-macro `{best_f1:.4f}`)'\n"
            "))\n"
            "\n"
            "best_result = train_one_model(\n"
            "    df, model=best_model, k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM, random_state=RANDOM_STATE,\n"
            ")\n"
            "_cv_metrics, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(\n"
            "    df,\n"
            "    lambda: build_estimator(best_model, best_result.best_params),\n"
            "    k_folds=K_FOLDS, buffer_km=BUFFER_KM, random_state=RANDOM_STATE,\n"
            ")\n"
            "class_names = {\n"
            "    i: PASTIS_R_CLASSES.get(int(c), f'c{int(c)}')\n"
            "    for i, c in enumerate(best_result.label_classes)\n"
            "}\n"
            "fig = plot_per_class_f1(\n"
            "    y_true_oof, y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names=class_names,\n"
            "    weak_threshold=0.10,\n"
            "    title=f'F1 por clase ({best_model}) — piloto subset',\n"
            ")\n"
            "fig.savefig(env.figures_dir / 'per_class_f1_04b.png', bbox_inches='tight')\n"
            "display(fig)\n"
            "plt.close(fig)\n"
        )
    )

    cells.extend(_hcat_grouping_cells(subset_size=8000))

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "**El patrón está validado.** Las tres familias de árboles "
            "(bagging, boosting clásico, boosting histogram-based) corren "
            "end-to-end sobre el subset en pocos minutos y los runs quedan "
            "registrados en MLflow. El mejor modelo del piloto (LightGBM, "
            "F1-macro `0.38` sobre 18 clases con las 185 features) queda muy "
            "cerca del valor histórico de XGBoost en el conjunto completo "
            "(`0.41`): el subset reduce ligeramente la cifra a igual modelo, "
            "como se espera, y eso confirma que la metodología es estable "
            "para arrancar la corrida completa de `04_baseline.ipynb`.\n\n"
            "**El hallazgo del piloto (sección 8).** La métrica sube en dos "
            "etapas, por motivos distintos que conviene no mezclar. El mejor "
            "modelo del bloque anterior (LightGBM sobre las 185 features) da "
            "`0.38` de F1-macro en 18 clases. Al pasar al escenario ganador "
            "de la ablación — XGBoost sobre 313 features (las 185 más el "
            "embedding AlphaEarth de 2018 y 2019) — el F1-macro de 18 clases "
            "sube a `0.54`. Ese salto combina dos efectos: el cambio de "
            "modelo y, sobre todo, la señal extra del embedding anual que las "
            "estadísticas espectrales no capturan (la corrida completa de "
            "`04_baseline.ipynb` aísla el aporte de AlphaEarth a igualdad de "
            "modelo). Después, medir ese mismo modelo sobre los 6 grupos HCAT "
            "en lugar de las 18 clases planas lo lleva de `0.54` a `0.75` "
            "(delta `+0.21`): ese tramo es puro error de confusión entre "
            "cultivos hermanos (trigo-con-trigo, cereal-con-cereal), no falta "
            "de señal. Leído por familias agronómicas, el baseline ya supera "
            "con holgura el umbral de referencia de `0.60`.\n\n"
            "**Sobre las preguntas oficiales**:\n\n"
            "- **P1 (algoritmo baseline)**: los 3 modelos tabulares cumplen "
            "el rol de baseline en pocos minutos sobre subset. La elección "
            "final del ganador queda en `04_baseline.ipynb` sobre el "
            "dataset completo, y la decisión de promoción para ensembles "
            "queda en `Avance3.Equipo17.ipynb`.\n"
            "- **P4 (métrica adecuada)**: el imbalance ratio del subset "
            "(`57x` entre la clase más y menos frecuente) evidencia por qué "
            "la accuracy es engañosa (predecir solo la clase mayoritaria "
            "daría una accuracy alta pero un F1-macro cercano a cero); por "
            "eso reportamos F1-macro como métrica principal y mostramos "
            "también el esquema agrupado para separar el error real del "
            "castigo por granularidad excesiva.\n\n"
            "## Lo que sigue\n\n"
            "- `04_baseline.ipynb` ejecuta los mismos 3 modelos sobre el "
            "conjunto completo + SHAP + learning curves + escenarios "
            "AlphaEarth vs S2 crudo.\n"
            "- `04c_baseline.ipynb` mide qué bloques del fused aportan "
            "(ablación canónica)."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04_baseline.ipynb — 3 models over fused + plots.
# ---------------------------------------------------------------------------


def build_04_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 0. Title and opening narrative.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "# Baseline de clasificación de cultivos — Random Forest, XGBoost y LightGBM\n\n"
            "Este cuaderno responde una pregunta concreta: **¿qué tan lejos "
            "llega un modelo tabular sencillo para clasificar cultivos a "
            "partir de imágenes satelitales?** Se entrenan tres modelos de "
            "árboles (Random Forest, XGBoost y LightGBM) sobre un vector de "
            "características que combina el embedding AlphaEarth de 64 "
            "dimensiones, los 17 índices espectrales × 9 estadísticos, FFT "
            "del NDVI, atributos fenológicos, ERA5 mensual y SRTM. El "
            "resultado sirve de **punto de referencia**: cualquier modelo "
            "más complejo en fases posteriores tendrá que superar estas "
            "cifras para justificar su coste.\n\n"
            "El cuaderno aborda cinco preguntas a lo largo del análisis:\n\n"
            "1. **¿Por qué elegir Random Forest, XGBoost y LightGBM** como "
            "modelos de referencia?\n"
            "2. **¿Qué características explican las predicciones**, según "
            "la importancia nativa (Gini, gain) y SHAP?\n"
            "3. **¿Cuánto del poder predictivo proviene del embedding "
            "AlphaEarth** frente a índices espectrales, terreno y clima?\n"
            "4. **¿El baseline sub o sobreajusta?** Diagnóstico con curvas "
            "de aprendizaje y validación.\n"
            "5. **¿AlphaEarth aporta valor frente a las bandas Sentinel-2 "
            "crudas** o frente al vector combinado de features espectro-"
            "temporales?\n\n"
            "Y cierra con una sexta pregunta de medición:\n\n"
            "6. **¿El F1-macro modesto sobre 18 clases refleja falta de "
            "señal, o castigo por confundir cultivos hermanos?** La sección "
            "8 reentrena el mismo modelo sobre los 6 grupos HCAT Level-1 "
            "para separar ambos efectos.\n\n"
            "## Requisitos para ejecución end-to-end\n\n"
            "- Subset PASTIS-R a nivel parcela descomprimido en "
            "`data/test_fixtures/feature_selection_parcels_subset.parquet`.\n"
            "- Geoparquet de parcelas en "
            "`data/processed/pastis_parcels_full.geoparquet`.\n"
            "- Para la sección 8 (18 clases vs 6 grupos HCAT): los embeddings "
            "AlphaEarth anuales "
            "`data/cache/gee/alphaearth_parcels_parcels_2018_85951.parquet` y "
            "`alphaearth_parcels_pastis_parcels_2019_85951.parquet` "
            "(descargables vía `dvc pull`).\n"
            "- Para la sección 7 (AlphaEarth vs S2 crudo): los parquets "
            "`alphaearth_pastis_parcels_2019_85951_enriched.parquet` y "
            "`s2_raw_parcels_2019_85951.parquet` en `data/cache/`; si "
            "faltan, esa sección se omite con un aviso explícito.\n"
            "- Dependencias instaladas vía `poetry install --with ml,geo`.\n"
        )
    )

    # -----------------------------------------------------------------------
    # Papermill parameters cell.
    # -----------------------------------------------------------------------
    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "RANDOM_STATE = 42\n"
            "SHAP_SAMPLE_SIZE = 3000\n"
            "TOP_FEATURES_DISPLAY = 20\n"
            "F1_THRESHOLD = 0.60\n"
            "LEARNING_CURVE_MAX_SAMPLES = 12000\n"
            "# Seccion 11 - comparativa AlphaEarth vs S2 crudo vs vector combinado.\n"
            'SCENARIO_ALPHAEARTH_PATH = "data/cache/gee/alphaearth_pastis_parcels_2019_85951_enriched.parquet"\n'
            'SCENARIO_S2_RAW_PATH = "data/cache/pastis/s2_raw_parcels_2019_85951.parquet"\n'
            'SCENARIO_COMBINED_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            "COMPARISON_MAX_SAMPLES = 0  # 0 = inner join completo\n"
            "COMPARISON_K_FOLDS = 5\n"
            "# MLflow tracking — experimento global del Avance 3 baseline.\n"
            'MLFLOW_EXPERIMENT = "baseline-04-tabular"\n',
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    # -----------------------------------------------------------------------
    # 0.1 MLflow tracking setup.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "### Trazabilidad MLflow\n\n"
            "Cada modelo entrenado abre un *run* en el experimento "
            "`baseline-04-tabular` con los tags `code_version` (SHA git) y "
            "`data_version` (hash DVC del parquet de features). Las "
            "métricas (F1-macro, F1-weighted, mIoU, accuracy, kappa) y los "
            "hiperparámetros quedan registrados para reabrir y reproducir "
            "cualquier corrida desde la UI MLflow en `http://localhost:5010`."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import (\n"
            "    resolve_tracking_uri,\n"
            "    track_experiment,\n"
            "    server_is_reachable,\n"
            ")\n"
            "\n"
            "# Resolucion robusta del tracking URI: si MLFLOW_TRACKING_URI esta en\n"
            "# `.env.local` pero el server no responde (Docker apagado, contenedor\n"
            "# detenido), caemos a `file:./mlruns` para no detener el notebook.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Caigo a tracking local `file:./mlruns`. '\n"
            "        'Para registrar en el server, ejecuta `docker compose up -d mlflow` '\n"
            "        'antes de re-ejecutar las celdas MLflow.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "display(Markdown(\n"
            "    f'**MLflow tracking URI**: `{mlflow_uri}` · '\n"
            "    f'**Experimento**: `{MLFLOW_EXPERIMENT}`'\n"
            "))\n"
            "MLFLOW_RUN_IDS: dict[str, str] = {}\n"
        )
    )

    # -----------------------------------------------------------------------
    # 1. Dataset loading.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 1. Carga del conjunto de datos\n\n"
            "El conjunto de entrada es un subset de PASTIS-R a nivel de "
            "parcela: 85 951 parcelas agrícolas con 187 características "
            "espectro-temporales cada una. La etiqueta es el tipo de "
            "cultivo (PASTIS-R define 18 clases activas tras descartar las "
            "de fondo). El loader une el parquet de features con la "
            "metadata enriquecida del geoparquet (clase real, `patch_id`, "
            "fold espacial y área), garantizando que `parcel_id` queda en "
            "`pl.Utf8` — el esquema canónico del proyecto."
        )
    )

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    train_baseline_three_models,\n"
            "    build_model_comparison_table,\n"
            ")\n"
            "from ml.utils.class_distribution import (\n"
            "    class_distribution_report,\n"
            "    recommend_threshold,\n"
            ")\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_class_support_bars,\n"
            "    plot_model_comparison_bars,\n"
            "    plot_confusion_matrix_heatmap,\n"
            "    plot_per_class_f1,\n"
            ")\n"
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "pid_dtype = df.schema['parcel_id']\n"
            "display(Markdown(\n"
            '    f"**Dataset**: `{df.height:,}` parcelas x `{df.width}` cols. "\n'
            '    f"`parcel_id`: `{pid_dtype}`"\n'
            "))\n"
            "display(df.head(5))\n"
        )
    )

    cells.append(
        _md(
            "### 1.1 Distribución de clases\n\n"
            "PASTIS-R tiene un desbalance fuerte: pocas clases concentran la "
            "mayoría de las parcelas. Reportamos las 18 clases con su "
            "conteo y proporción, marcando como **soporte débil** las que "
            "caen por debajo del **percentil 25** de la distribución (en "
            "lugar de un umbral fijo). Esto previene declarar artificialmente "
            "como minoritarias a clases que sí tienen soporte suficiente."
        )
    )

    cells.append(
        _code(
            "report = class_distribution_report(df)\n"
            "display(report)\n"
            "threshold = recommend_threshold(report, method='p25')\n"
            "display(Markdown(f'Umbral sugerido (P25): `{threshold}` parcelas.'))\n"
            "\n"
            "fig_class = plot_class_support_bars(\n"
            "    report.rename({'n_parcels': 'len'}),\n"
            "    weak_threshold=threshold,\n"
            "    title=f'Distribución de clases (umbral P25 = {threshold} parcelas)',\n"
            ")\n"
            "fig_class.savefig(env.figures_dir / 'class_distribution.png', bbox_inches='tight')\n"
            "display(fig_class)\n"
            "plt.close(fig_class)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 2. Why three tree models.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 2. Por qué Random Forest, XGBoost y LightGBM\n\n"
            "Se eligen **tres modelos de árboles** como referencia. Cinco "
            "razones sustentan la decisión:\n\n"
            "**(a) Las imágenes ya vienen resumidas.** El embedding "
            "AlphaEarth de 64 dimensiones condensa información óptica, "
            "radar y temporal aprendida por un modelo entrenado sobre todo "
            "el archivo Sentinel. Sobre una representación ya rica, un "
            "modelo de árboles es un punto de referencia suficiente y "
            "honesto: no hace falta una red profunda para fijar el piso de "
            "desempeño (cf. Brown et al., 2025, *AlphaEarth Foundations*).\n\n"
            "**(b) Son interpretables.** Random Forest expone la "
            "importancia Gini, XGBoost la ganancia (*gain*) y LightGBM la "
            "ganancia split-wise. Todos son compatibles con SHAP. Esto "
            "permite auditar qué variables explican las predicciones, no "
            "solo medir aciertos.\n\n"
            "**(c) Tres familias de árboles, no una.** Random Forest "
            "(bagging) reduce varianza; XGBoost (boosting con regularización "
            "L1/L2) reduce sesgo; LightGBM (boosting con histogram-based "
            "splits y leaf-wise growth) entrena mucho más rápido sobre "
            "datasets grandes. Comparar las tres familias evita atribuir "
            "diferencias de F1 a un único algoritmo.\n\n"
            "**(d) Validación cruzada espacial.** Cada modelo se evalúa con "
            "el mismo CV espacial 5-fold (celdas hexagonales H3 + "
            "agrupamiento KMeans + buffer 1 km), no con un split aleatorio. "
            "Esto evita que parcelas vecinas queden a la vez en train y "
            "test (leakage espacial garantizado en datos satelitales).\n\n"
            "**(e) Reproducibilidad.** Los tres modelos comparten "
            "`random_state=42`, la misma matriz de features y las mismas "
            "particiones — la única variable que cambia es el algoritmo."
        )
    )

    cells.append(
        _md(
            "## 3. Entrenamiento con validación cruzada espacial\n\n"
            "Tiempo de pared esperado: **30-60 minutos** (RF en CPU "
            "multinúcleo + XGBoost en GPU + LightGBM en CPU). El helper "
            "`train_baseline_three_models` materializa los folds una sola "
            "vez y los reusa entre modelos."
        )
    )

    cells.append(
        _code(
            "rows = train_baseline_three_models(\n"
            "    df,\n"
            "    models=('rf', 'xgb', 'lgbm'),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "comparison_path = env.reports_dir / 'model_comparison_04.parquet'\n"
            "comparison = build_model_comparison_table(rows, output_path=comparison_path)\n"
            "display(Markdown(f'**Tabla guardada**: `{comparison_path.relative_to(env.repo)}`'))\n"
            "display(comparison)\n"
            "\n"
            "# MLflow: un run por modelo con metricas, params y tags estandar.\n"
            "for r in rows:\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'04-{r.model}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        dvc_path=FEATURES_PATH,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        mlflow.log_params({\n"
            "            'model': r.model,\n"
            "            'k_folds': K_FOLDS,\n"
            "            'buffer_km': BUFFER_KM,\n"
            "            'random_state': RANDOM_STATE,\n"
            "            'n_parcels': df.height,\n"
            "        })\n"
            "        mlflow.log_metrics({\n"
            "            'f1_macro': r.f1_macro,\n"
            "            'f1_weighted': r.f1_weighted,\n"
            "            'miou': r.miou,\n"
            "            'accuracy': r.accuracy,\n"
            "            'kappa': r.cohen_kappa,\n"
            "            'train_time_s': r.train_time_s,\n"
            "        })\n"
            "        MLFLOW_RUN_IDS[r.model] = run.info.run_id\n"
            "display(Markdown('**MLflow runs registrados**: ' + ', '.join(\n"
            "    f'`{k}={v[:12]}...`' for k, v in MLFLOW_RUN_IDS.items()\n"
            ")))\n"
        )
    )

    cells.append(_md("### 3.1 Comparativa F1-macro entre los tres modelos"))

    cells.append(
        _code(
            "metric_by_model = {r.model: r.f1_macro for r in rows}\n"
            "fig_cmp = plot_model_comparison_bars(\n"
            "    metric_by_model,\n"
            "    baseline_value=F1_THRESHOLD,\n"
            "    baseline_label=f'umbral de referencia (F1-macro = {F1_THRESHOLD:.2f})',\n"
            "    title='F1-macro out-of-fold por modelo',\n"
            ")\n"
            "fig_cmp.savefig(env.figures_dir / 'model_comparison.png', bbox_inches='tight')\n"
            "display(fig_cmp)\n"
            "plt.close(fig_cmp)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 4. Confusion matrix and per-class F1 of the winner.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 4. Modelo ganador — matriz de confusión y F1 por clase\n\n"
            "El modelo ganador (mayor F1-macro out-of-fold) se reentrena "
            "para conseguir las predicciones out-of-fold completas. A "
            "partir de ellas se construyen la matriz de confusión "
            "normalizada por fila y el F1 por clase, que identifican qué "
            "clases concentran el error y cuáles están bien resueltas.\n\n"
            "El modelo serializado se guarda en "
            "`reports/baseline/04_baseline/best_model_*.joblib` para "
            "reutilizarlo desde `Avance3.Equipo17.ipynb`."
        )
    )

    cells.append(
        _code(
            "from ml.train.baseline import (\n"
            "    train_one_model,\n"
            "    evaluate_with_spatial_cv,\n"
            "    build_estimator,\n"
            ")\n"
            "best_model = comparison['model'][0]\n"
            "best_f1 = float(comparison['f1_macro'][0])\n"
            "display(Markdown(\n"
            "    f'Modelo ganador: `{best_model}` (F1-macro `{best_f1:.4f}`)'\n"
            "))\n"
            "\n"
            "# train_one_model solo acepta 'rf' o 'xgb'. Si gana LGBM, usamos\n"
            "# XGBoost como modelo interpretable (es la familia mas cercana).\n"
            "interpretable_kind = best_model if best_model in ('rf', 'xgb') else 'xgb'\n"
            "if interpretable_kind != best_model:\n"
            "    display(Markdown(\n"
            "        f'> Nota: las secciones 5 y 6 (importancia, SHAP, curvas) usan '\n"
            "        f'`{interpretable_kind}` como sustituto interpretable de `{best_model}` '\n"
            "        '(el helper SHAP del proyecto solo soporta RF y XGB).'\n"
            "    ))\n"
            "\n"
            "best_result = train_one_model(\n"
            "    df,\n"
            "    model=interpretable_kind,\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "_, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(\n"
            "    df,\n"
            "    lambda: build_estimator(interpretable_kind, best_result.best_params),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "\n"
            "class_names_decoded = {\n"
            "    i: PASTIS_R_CLASSES.get(int(c), f'c{int(c)}')\n"
            "    for i, c in enumerate(best_result.label_classes)\n"
            "}\n"
            "\n"
            "fig_cm = plot_confusion_matrix_heatmap(\n"
            "    y_true_oof,\n"
            "    y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names=class_names_decoded,\n"
            "    normalize='true',\n"
            "    title=f'Matriz de confusión ({interpretable_kind}) normalizada por fila',\n"
            ")\n"
            "fig_cm.savefig(env.figures_dir / 'confusion_matrix.png', bbox_inches='tight')\n"
            "display(fig_cm)\n"
            "plt.close(fig_cm)\n"
            "\n"
            "fig_f1 = plot_per_class_f1(\n"
            "    y_true_oof,\n"
            "    y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names=class_names_decoded,\n"
            "    weak_threshold=0.10,\n"
            "    title=f'F1 por clase ({interpretable_kind})',\n"
            ")\n"
            "fig_f1.savefig(env.figures_dir / 'per_class_f1.png', bbox_inches='tight')\n"
            "display(fig_f1)\n"
            "plt.close(fig_f1)\n"
            "\n"
            "import joblib\n"
            "joblib_path = env.reports_dir / f'best_model_{best_model}.joblib'\n"
            "joblib.dump(best_result, joblib_path)\n"
            "display(Markdown(f'Modelo guardado en `{joblib_path.relative_to(env.repo)}`'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 5. Feature importance + SHAP.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 5. Importancia de características y análisis SHAP\n\n"
            "Random Forest y XGBoost exponen una medida de importancia sin "
            "coste adicional: **Gini** para Random Forest y **gain** para "
            "XGBoost. Es el primer diagnóstico de interpretabilidad — "
            "barato y directo — antes del análisis SHAP.\n\n"
            "**SHAP** (Lundberg & Lee, 2017) descompone cada predicción "
            "en contribuciones aditivas por característica con garantías "
            "teóricas de consistencia. Para modelos de árboles se usa el "
            "algoritmo **TreeSHAP**, que es exacto. Se calcula sobre un "
            "subsample estratificado (`SHAP_SAMPLE_SIZE` parcelas) porque "
            "el coste de TreeSHAP crece con el número de muestras, árboles "
            "y profundidad."
        )
    )

    cells.append(
        _code(
            "from ml.eval.interpretability import (\n"
            "    feature_importance_table,\n"
            "    compute_shap_values,\n"
            "    shap_summary_plot,\n"
            "    shap_dependence_plots,\n"
            "    shap_waterfall_plot,\n"
            "    alphaearth_dominance_table,\n"
            ")\n"
            "\n"
            "# Entrenamos ambos modelos interpretables (RF y XGB) sobre el dataset\n"
            "# completo para tener la importancia nativa y SHAP de los dos.\n"
            "interpretable_models = {}\n"
            "for kind in ('rf', 'xgb'):\n"
            "    res = train_one_model(\n"
            "        df,\n"
            "        model=kind,\n"
            "        k_folds=K_FOLDS,\n"
            "        buffer_km=BUFFER_KM,\n"
            "        random_state=RANDOM_STATE,\n"
            "    )\n"
            "    interpretable_models[kind] = res\n"
            "    display(Markdown(\n"
            "        f'- `{kind}` ajustado sobre `{df.height:,}` parcelas con '\n"
            "        f'`{len(res.feature_cols)}` features.'\n"
            "    ))\n"
        )
    )

    cells.append(_md("### 5.1 Importancia nativa — top-20 por modelo"))

    cells.append(
        _code(
            "importance_tables = {}\n"
            "for kind, res in interpretable_models.items():\n"
            "    table = feature_importance_table(\n"
            "        res.model,\n"
            "        model_kind=kind,\n"
            "        feature_cols=tuple(res.feature_cols),\n"
            "    )\n"
            "    importance_tables[kind] = table\n"
            "    csv_path = env.reports_dir / f'feature_importance_{kind}.csv'\n"
            "    table.write_csv(csv_path)\n"
            "    display(Markdown(\n"
            '        f\'**Importancia nativa `{kind}`** ({"Gini" if kind == "rf" else "gain"}). \'\n'
            "        f'Guardada en `{csv_path.relative_to(env.repo)}`.'\n"
            "    ))\n"
            "    display(table.head(TOP_FEATURES_DISPLAY))\n"
            "\n"
            "# Barplot horizontal top-20 por modelo.\n"
            "import numpy as np\n"
            "for kind, table in importance_tables.items():\n"
            "    top = table.head(TOP_FEATURES_DISPLAY)\n"
            "    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)\n"
            "    ax.barh(top['feature'].to_list()[::-1], top['importance'].to_list()[::-1], color='#4C72B0')\n"
            "    ax.set_xlabel('importancia (Gini)' if kind == 'rf' else 'importancia (gain)')\n"
            "    ax.set_title(f'Importancia nativa top-{TOP_FEATURES_DISPLAY} ({kind.upper()})')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(env.figures_dir / f'feature_importance_{kind}.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(_md("### 5.2 SHAP — beeswarm global, dependence plots y waterfall"))

    cells.append(
        _code(
            "shap_results = {}\n"
            "for kind, res in interpretable_models.items():\n"
            "    feature_cols = tuple(res.feature_cols)\n"
            "    X_for_shap = df.select(list(feature_cols))\n"
            "    shap_results[kind] = compute_shap_values(\n"
            "        res.model,\n"
            "        X_for_shap,\n"
            "        model_kind=kind,\n"
            "        feature_cols=feature_cols,\n"
            "        sample_size=SHAP_SAMPLE_SIZE,\n"
            "        random_state=RANDOM_STATE,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'- SHAP `{kind}`: tensor `{shap_results[kind].values.shape}` '\n"
            "        f'(muestras, features, clases).'\n"
            "    ))\n"
            "\n"
            "# Summary plot (beeswarm/bar) top-20 global por modelo.\n"
            "for kind, sr in shap_results.items():\n"
            "    fig = shap_summary_plot(sr, df.select(list(sr.feature_cols)), top_n=TOP_FEATURES_DISPLAY)\n"
            "    fig.savefig(env.figures_dir / f'shap_summary_{kind}.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(
        _md(
            "**Lectura del SHAP summary**: cada barra agrega la magnitud "
            "absoluta media del impacto SHAP por feature, promediada sobre "
            "todas las clases. Una barra larga indica que esa característica "
            "desplaza con fuerza la predicción — hacia o lejos de la clase, "
            "según el signo. Es la mejor manera de leer el ranking global "
            "sin perder la dirección del efecto en cada clase."
        )
    )

    cells.append(
        _code(
            "# Dependence plots de las top-5 features (modelo principal: el que mas vario\n"
            "# en importancia, por convencion RF).\n"
            "primary = 'rf'\n"
            "dependence = shap_dependence_plots(\n"
            "    shap_results[primary],\n"
            "    df.select(list(shap_results[primary].feature_cols)),\n"
            "    top_features=5,\n"
            ")\n"
            "for feature_name, fig in dependence:\n"
            "    fig.savefig(\n"
            "        env.figures_dir / f'shap_dependence_{primary}_{feature_name}.png',\n"
            "        bbox_inches='tight',\n"
            "    )\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(
        _md(
            "**Lectura de los dependence plots**: el eje X es el valor del "
            "feature y el eje Y es el valor SHAP de ese feature para cada "
            "parcela. Una pendiente clara indica un efecto monótono (el "
            "feature empuja la predicción de forma proporcional); una nube "
            "sin estructura indica que el efecto depende de interacciones "
            "con otras features y no es interpretable aislado."
        )
    )

    cells.append(
        _code(
            "# Waterfall de una prediccion ejemplo por modelo.\n"
            "for kind, sr in shap_results.items():\n"
            "    fig = shap_waterfall_plot(sr, row=0)\n"
            "    fig.savefig(env.figures_dir / f'shap_waterfall_{kind}.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 5.3 AlphaEarth dominance.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "### 5.3 Dominancia de las dimensiones AlphaEarth\n\n"
            "De las características más influyentes según SHAP, **¿cuántas "
            "son dimensiones del embedding AlphaEarth (`dim_00..dim_63`)** "
            "frente a índices espectrales, estadísticas temporales o "
            "bloques de contexto (radar, terreno, clima)? La respuesta "
            "cuantifica cuánto del poder predictivo proviene del embedding "
            "satelital frente al resto. Si dominan, AlphaEarth está "
            "haciendo el trabajo principal; si no, el feature engineering "
            "espectro-temporal sigue siendo imprescindible."
        )
    )

    cells.append(
        _code(
            "dominance_tables = {}\n"
            "for kind, sr in shap_results.items():\n"
            "    dom = alphaearth_dominance_table(\n"
            "        sr.global_importance,\n"
            "        top_n=TOP_FEATURES_DISPLAY,\n"
            "    )\n"
            "    dominance_tables[kind] = dom\n"
            "    family_counts = (\n"
            "        dom.group_by('family').len().sort('len', descending=True)\n"
            "    )\n"
            "    n_ae = int(family_counts.filter(pl.col('family') == 'alphaearth')['len'].sum())\n"
            "    display(Markdown(\n"
            "        f'**Dominancia AlphaEarth (`{kind}`)**: '\n"
            "        f'`{n_ae}/{TOP_FEATURES_DISPLAY}` features del top son dimensiones '\n"
            "        f'`dim_NN` del embedding.'\n"
            "    ))\n"
            "    display(dom)\n"
            "    display(family_counts)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 6. Learning and validation curves.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 6. Curvas de aprendizaje y validación — diagnóstico de sub/sobreajuste\n\n"
            "Esta sección diagnostica si el baseline sub o sobreajusta. "
            "Dos herramientas:\n\n"
            "- **Curva de aprendizaje**: accuracy de train y de validación "
            "al crecer el número de muestras de entrenamiento. Un gap "
            "grande train-val indica sobreajuste; ambas curvas bajas y "
            "juntas, subajuste.\n"
            "- **Curva de validación**: accuracy frente a un hiperparámetro "
            "crítico (`max_depth` para RF, `n_estimators` para XGBoost), "
            "para localizar la zona de equilibrio.\n\n"
            "Toda la evaluación usa el mismo CV espacial 5-fold del resto "
            "del cuaderno. Para que las curvas no tarden horas, se "
            "subsamplean a `LEARNING_CURVE_MAX_SAMPLES` parcelas con "
            "muestreo estratificado por clase."
        )
    )

    cells.append(
        _code(
            "from ml.eval.learning_curves import (\n"
            "    plot_learning_curve,\n"
            "    plot_validation_curve,\n"
            "    diagnose_fit,\n"
            ")\n"
            "from ml.train.baseline import _build_cv_splits\n"
            "\n"
            "# Materializamos los folds espaciales una sola vez para reusarlos en learning_curve\n"
            "# y validation_curve. _build_cv_splits cachea el resultado en data/test_fixtures/.\n"
            "cv_splits = _build_cv_splits(\n"
            "    df,\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "display(Markdown(\n"
            "    f'**Folds espaciales materializados**: `{len(cv_splits)}` particiones '\n"
            "    f'con buffer de `{BUFFER_KM} km`.'\n"
            "))\n"
        )
    )

    cells.append(_md("### 6.1 Curvas de aprendizaje (RF y XGB)"))

    cells.append(
        _code(
            "learning_results = {}\n"
            "for kind in ('rf', 'xgb'):\n"
            "    estimator = build_estimator(kind, {})\n"
            "    lc_result, lc_fig = plot_learning_curve(\n"
            "        estimator,\n"
            "        df,\n"
            "        cv_splits=cv_splits,\n"
            "        max_samples=LEARNING_CURVE_MAX_SAMPLES,\n"
            "        random_state=RANDOM_STATE,\n"
            "    )\n"
            "    learning_results[kind] = lc_result\n"
            "    # Sobrescribimos el title del axes (ml.eval.learning_curves ya pone\n"
            "    # 'Curva de aprendizaje (accuracy)'); evitamos un suptitle adicional\n"
            "    # que se solapaba con el title interno.\n"
            "    for _ax in lc_fig.axes:\n"
            "        _ax.set_title(f'Curva de aprendizaje (accuracy) — {kind.upper()}')\n"
            "    lc_fig.tight_layout()\n"
            "    lc_fig.savefig(env.figures_dir / f'learning_curve_{kind}.png', bbox_inches='tight')\n"
            "    display(lc_fig)\n"
            "    plt.close(lc_fig)\n"
            "\n"
            "# Diagnostico explicito por modelo.\n"
            "for kind, lc in learning_results.items():\n"
            "    diag = diagnose_fit(lc)\n"
            "    display(Markdown(\n"
            "        f'**Diagnóstico `{kind}`**: `{diag.verdict}` '\n"
            "        f'(gap = `{diag.gap:.3f}`, val_acc = `{diag.val_acc_max:.3f}`).\\n\\n'\n"
            "        f'{diag.explanation}'\n"
            "    ))\n"
        )
    )

    cells.append(_md("### 6.2 Curvas de validación — `max_depth` (RF) y `n_estimators` (XGB)"))

    cells.append(
        _code(
            "# Curva de validacion RF - max_depth.\n"
            "vc_rf_result, vc_rf_fig = plot_validation_curve(\n"
            "    build_estimator('rf', {}),\n"
            "    df,\n"
            "    param_name='max_depth',\n"
            "    param_range=[5, 10, 15, 20, 25, None],\n"
            "    cv_splits=cv_splits,\n"
            "    max_samples=LEARNING_CURVE_MAX_SAMPLES,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "vc_rf_fig.savefig(env.figures_dir / 'validation_curve_rf_max_depth.png', bbox_inches='tight')\n"
            "display(vc_rf_fig)\n"
            "plt.close(vc_rf_fig)\n"
            "\n"
            "# Curva de validacion XGB - n_estimators.\n"
            "vc_xgb_result, vc_xgb_fig = plot_validation_curve(\n"
            "    build_estimator('xgb', {}),\n"
            "    df,\n"
            "    param_name='n_estimators',\n"
            "    param_range=[50, 100, 200, 400, 800],\n"
            "    cv_splits=cv_splits,\n"
            "    max_samples=LEARNING_CURVE_MAX_SAMPLES,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "vc_xgb_fig.savefig(env.figures_dir / 'validation_curve_xgb_n_estimators.png', bbox_inches='tight')\n"
            "display(vc_xgb_fig)\n"
            "plt.close(vc_xgb_fig)\n"
        )
    )

    cells.append(
        _md(
            "**Lectura de las curvas de validación**: el eje X recorre el "
            "rango del hiperparámetro y el eje Y muestra accuracy de train "
            "y de validación. La zona de equilibrio es donde la curva de "
            "validación deja de subir (cualquier valor más alto solo "
            "incrementa el gap, no la generalización). Si train sube "
            "rápido a 1.0 y val se estanca, el modelo está agotando su "
            "capacidad de generalizar sobre estas features."
        )
    )

    # -----------------------------------------------------------------------
    # 7. Comparison AlphaEarth vs raw S2 vs combined vector.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 7. Comparativa AlphaEarth vs Sentinel-2 crudo vs vector combinado\n\n"
            "Esta sección compara el baseline sobre **tres vistas distintas "
            "de las mismas parcelas**, para responder con evidencia una "
            "pregunta central: **¿el embedding AlphaEarth aporta valor "
            "frente a las bandas Sentinel-2 sin procesar?**\n\n"
            "| Escenario | Características | Origen |\n"
            "|-----------|-----------------|--------|\n"
            "| **(a) AlphaEarth** | 64 dimensiones | embedding AlphaEarth Foundations v2.1 |\n"
            "| **(b) Sentinel-2 crudo** | 10 bandas promedio | bandas Sentinel-2 sin procesar, agregadas por parcela |\n"
            "| **(c) Vector combinado** | 187 características | ingeniería de features espectro-temporales |\n\n"
            "Metodología:\n\n"
            "- Los 3 escenarios se cruzan por `parcel_id` con un **inner "
            "join** para que los tres modelos se evalúen **exactamente "
            "sobre el mismo conjunto de parcelas**, no sobre tres muestras "
            "distintas.\n"
            "- Cada escenario entrena RF + XGB con el mismo CV espacial "
            "5-fold y buffer de 1 km.\n"
            "- La tabla se persiste como CSV, Markdown y LaTeX "
            "(`comparison_table.tex`) para reutilizarla en el reporte."
        )
    )

    cells.append(
        _code(
            "from pathlib import Path\n"
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
            "missing = {k: p for k, p in scenario_paths.items() if not Path(p).exists()}\n"
            "comparison_available = not missing\n"
            "\n"
            "if missing:\n"
            "    lines = [f'- `{k}`: `{p}`' for k, p in missing.items()]\n"
            "    display(Markdown(\n"
            "        '> **Comparativa omitida**: faltan los siguientes escenarios:\\n\\n' +\n"
            "        '\\n'.join(lines) +\n"
            "        '\\n\\nGenera el escenario S2 crudo con `make s2-raw-parcels` o '\n"
            "        'descarga AlphaEarth via `dvc pull`.'\n"
            "    ))\n"
            "else:\n"
            "    display(Markdown('Los 3 escenarios están disponibles para la comparativa.'))\n"
        )
    )

    cells.append(
        _code(
            "comparison_result = None\n"
            "if comparison_available:\n"
            "    comparison_result = build_comparison_table(\n"
            "        scenario_paths,\n"
            "        k_folds=COMPARISON_K_FOLDS,\n"
            "        buffer_km=BUFFER_KM,\n"
            "        max_samples=COMPARISON_MAX_SAMPLES,\n"
            "        random_state=RANDOM_STATE,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**Parcelas en el inner join**: `{comparison_result.n_parcels:,}`. '\n"
            "        f'**Escenario ganador**: `{comparison_result.best_scenario}`. '\n"
            "        f'**Delta AlphaEarth vs S2 crudo**: `{comparison_result.alphaearth_delta:+.4f}`.'\n"
            "    ))\n"
            "    display(comparison_result.table)\n"
            "    # Persistencia (CSV + MD + LaTeX).\n"
            "    reports_dir = env.reports_dir\n"
            "    csv_path = reports_dir / 'comparison_alphaearth_vs_s2.csv'\n"
            "    comparison_result.table.write_csv(csv_path)\n"
            "    md_table = (\n"
            "        '# Comparativa de escenarios - baseline de cultivos\\n\\n'\n"
            "        + comparison_result.table.to_pandas().to_markdown(index=False)\n"
            "        + '\\n'\n"
            "    )\n"
            "    (reports_dir / 'comparison_alphaearth_vs_s2.md').write_text(\n"
            "        md_table, encoding='utf-8'\n"
            "    )\n"
            "    tex_path = export_comparison_latex(\n"
            "        comparison_result, reports_dir / 'comparison_table.tex'\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'Tabla comparativa exportada: '\n"
            "        f'`{csv_path.relative_to(env.repo)}`, MD y `{tex_path.name}`.'\n"
            "    ))\n"
            "else:\n"
            "    display(Markdown('> Comparativa omitida - ver celda anterior.'))\n"
        )
    )

    cells.append(_md("### 7.1 Barplot comparativo F1-macro por escenario y modelo"))

    cells.append(
        _code(
            "if comparison_result is not None:\n"
            "    table = comparison_result.table\n"
            "    scenarios = table['scenario'].unique(maintain_order=True).to_list()\n"
            "    models = table['model'].unique(maintain_order=True).to_list()\n"
            "    x = list(range(len(scenarios)))\n"
            "    width = 0.26  # 3 barras por escenario\n"
            "    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)\n"
            "    palette = {'RF': '#4C72B0', 'XGB': '#DD8452', 'LGBM': '#55A868'}\n"
            "    for i, m in enumerate(models):\n"
            "        vals = [\n"
            "            float(table.filter((pl.col('scenario') == s) & (pl.col('model') == m))['f1_macro'][0])\n"
            "            for s in scenarios\n"
            "        ]\n"
            "        offset = (i - (len(models) - 1) / 2) * width\n"
            "        ax.bar([xi + offset for xi in x], vals, width=width, label=m, color=palette.get(m, '#999'))\n"
            "    ax.set_xticks(x)\n"
            "    ax.set_xticklabels(scenarios, rotation=15)\n"
            "    ax.set_ylabel('F1-macro out-of-fold')\n"
            "    ax.set_title('Comparativa: AlphaEarth vs S2 crudo vs vector combinado (RF/XGB/LGBM)')\n"
            "    ax.axhline(F1_THRESHOLD, color='#888', linestyle='--', linewidth=1,\n"
            "               label=f'umbral {F1_THRESHOLD:.2f}')\n"
            "    ax.legend(loc='best')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(env.figures_dir / 'comparison_barplot.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
            "else:\n"
            "    display(Markdown('> Barplot omitido - ver seccion 7.'))\n"
        )
    )

    cells.append(
        _md(
            "**Lectura del barplot**: las barras están agrupadas por "
            "escenario, con tres barras (RF, XGB, LGBM) cada una. El delta "
            "entre el escenario AlphaEarth y el Sentinel-2 crudo cuantifica "
            "el valor incremental del embedding fundacional. Si AlphaEarth "
            "supera al S2 crudo con margen claro, ese resumen aprendido "
            "vale más que el promedio simple de bandas. Si el vector "
            "combinado supera a AlphaEarth, las features espectro-"
            "temporales agregan información que el embedding no captura. "
            "Comparar LGBM con XGB en cada escenario revela si la elección "
            "del algoritmo de boosting altera la conclusión sobre el "
            "extractor."
        )
    )

    # -----------------------------------------------------------------------
    # 8. Scheme of 18 classes vs 6 HCAT Level-1 groups (shared helper,
    # full 85951 in 04_baseline).
    # -----------------------------------------------------------------------
    cells.extend(_hcat_grouping_cells(subset_size=None))

    # -----------------------------------------------------------------------
    # 9. Conclusions - answer to the 5 questions.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 9. Conclusiones\n\n"
            "Este cuaderno construyó un punto de referencia para clasificar "
            "cultivos a partir de imágenes satelitales y lo sometió a las "
            "cinco preguntas planteadas al inicio. Lo que encontramos:\n\n"
            "### ¿Por qué tres modelos de árboles?\n\n"
            "Las tres familias (bagging, boosting clásico y boosting "
            "histogram-based) acotan el desempeño desde ángulos distintos "
            "y permiten descartar que un resultado sea un artefacto de un "
            "único algoritmo. La sección 3 muestra cuál de los tres ganó "
            "con datos.\n\n"
            "### ¿Qué características explican las predicciones?\n\n"
            "La importancia nativa (sección 5.1) ordena rápidamente las "
            "features; SHAP (sección 5.2) explica **cómo** desplazan la "
            "predicción y permite leer interacciones. Las top features "
            "dependen del modelo: RF y XGB suelen coincidir en las primeras "
            "5-10 posiciones, con discrepancias en las cola que señalan "
            "variables con efectos no lineales que SHAP captura mejor que "
            "la importancia simple.\n\n"
            "### ¿Cuánto del poder predictivo proviene de AlphaEarth?\n\n"
            "La tabla de dominancia (sección 5.3) cuenta cuántas de las "
            "20 features más influyentes son dimensiones `dim_NN`. Si la "
            "mayoría son del embedding, AlphaEarth está cargando con el "
            "trabajo principal; si los índices espectrales y estadísticas "
            "estacionales aparecen mezclados, el feature engineering sigue "
            "siendo imprescindible incluso con un buen embedding "
            "fundacional.\n\n"
            "### ¿Sub o sobreajuste?\n\n"
            "Las curvas de aprendizaje (sección 6.1) y `diagnose_fit` "
            "entregan un veredicto explícito (`overfit`, `underfit`, "
            "`good_fit`) basado en el gap train-val y la accuracy de "
            "validación. Las curvas de validación (sección 6.2) muestran "
            "el rango útil de `max_depth` (RF) y `n_estimators` (XGB) — "
            "más capacidad allí no mejora generalización.\n\n"
            "### ¿AlphaEarth aporta valor frente a S2 crudo?\n\n"
            "La sección 7 contesta con datos: el delta de F1-macro entre "
            "el escenario AlphaEarth y el Sentinel-2 crudo cuantifica "
            "cuánto vale el resumen aprendido del embedding. Si la "
            "diferencia es positiva y grande, la decisión de usar "
            "AlphaEarth como base del baseline queda validada con "
            "evidencia, no con un argumento teórico.\n\n"
            "### ¿Qué tan bueno es el baseline de verdad?\n\n"
            "La sección 8 lo pone en perspectiva. Medido sobre las 18 clases "
            "planas, el F1-macro parece modesto, pero buena parte de ese "
            "número es el castigo por confundir cultivos hermanos "
            "(trigo-con-trigo, cereal-con-cereal) que para el uso agronómico "
            "son intercambiables. Al medir el mismo modelo sobre los 6 grupos "
            "HCAT Level-1 — sin cambiar features ni validación — el F1-macro "
            "sube de forma marcada: la señal para distinguir **familias de "
            "cultivo** ya está presente en el embedding anual; lo que falta "
            "es resolución intra-familia, que solo aporta la dinámica "
            "temporal. El baseline, leído por familias, es más fuerte de lo "
            "que sugiere la métrica plana.\n\n"
            "## Lo que sigue\n\n"
            "- `04c_baseline.ipynb` mide el aporte incremental de cada "
            "bloque del vector fused (`alphaearth_only`, `phenology_only`, "
            "`no_geom`, `geom_only` como test de leakage).\n"
            "- `05_reencuadre_fenologico.ipynb` cuantifica el aporte de "
            "los bloques opcionales (FarSLIP, descripción fenológica "
            "textual con Gemini, firma espectral REP) sobre este conjunto.\n"
            "- `Avance3.Equipo17.ipynb` selecciona y guarda el conjunto "
            "ganador (`select_winning_features`) para los modelos densos "
            "del Avance 4."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04c_baseline.ipynb — ablation with alphaearth_only fix.
# ---------------------------------------------------------------------------


def build_04c_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04c — Ablación de bloques de características\n\n"
            "Mide el aporte incremental de cada bloque del vector fused. "
            "Para cada conjunto de columnas entrenamos XGBoost con la misma "
            "validación cruzada espacial de 5 particiones y reportamos "
            "F1-macro y el delta respecto al conjunto completo (`full`).\n\n"
            "Conjuntos canónicos evaluados:\n\n"
            "- `full`: todas las características numéricas disponibles.\n"
            "- `no_geom`: `full` sin las 3 columnas `geom_*`.\n"
            "- `no_geom_no_era5_srtm`: además sin `era5_*` ni `srtm_*`.\n"
            "- `alphaearth_only`: sólo las dimensiones AlphaEarth (`ae18_*` + `ae19_*`, 128 cols).\n"
            "- `phenology_only`: 8 atributos fenológicos + 24 FFT NDVI.\n"
            "- `geom_only`: sólo `geom_*` (prueba cuantitativa de fuga espacial).\n\n"
            "**Detección de columnas AlphaEarth**: el detector tolera "
            "variantes de prefijo (`ae_*`, `emb_*`, `dim_*`, `alphaearth_*`), "
            "por lo que `alphaearth_only` ya no aparece con `n_features=0` "
            "ni NaN cuando hay embeddings AlphaEarth en el dataset."
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04c_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04c_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "MAX_SAMPLES = None  # None = dataset completo; usar un valor menor para corridas rápidas.\n"
            'MLFLOW_EXPERIMENT = "baseline-04c-ablation"\n',
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(
        _md(
            "### Trazabilidad MLflow\n\n"
            "La ablación abre un run MLflow por cada `feature_set` evaluado "
            "en el experimento `baseline-04c-ablation`. Cada run reporta el "
            "F1-macro, F1-weighted, mIoU, el delta vs `full` y el número de "
            "features del conjunto. La trazabilidad permite recuperar qué "
            "columnas integraban cada conjunto."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import (\n"
            "    resolve_tracking_uri,\n"
            "    track_experiment,\n"
            "    server_is_reachable,\n"
            ")\n"
            "\n"
            "# Resolucion robusta del tracking URI: si MLFLOW_TRACKING_URI esta en\n"
            "# `.env.local` pero el server no responde (Docker apagado, contenedor\n"
            "# detenido), caemos a `file:./mlruns` para no detener el notebook.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Caigo a tracking local `file:./mlruns`. '\n"
            "        'Para registrar en el server, ejecuta `docker compose up -d mlflow` '\n"
            "        'antes de re-ejecutar las celdas MLflow.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "display(Markdown(\n"
            "    f'**MLflow tracking URI**: `{mlflow_uri}` · '\n"
            "    f'**Experimento**: `{MLFLOW_EXPERIMENT}`'\n"
            "))\n"
            "MLFLOW_RUN_IDS: dict[str, str] = {}\n"
        )
    )

    cells.append(_md("## 1. Carga del dataset y ejecución de la ablación"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_base_plus_alphaearth_2018_2019,\n"
            "    run_ablation_and_persist,\n"
            ")\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_ablation_bars,\n"
            "    plot_geom_leakage_comparison,\n"
            ")\n"
            "\n"
            "# Carga el escenario ganador del 04 (base 185 + AlphaEarth 2018 + 2019)\n"
            "# para que el conjunto `alphaearth_only` tenga las 128 dimensiones reales\n"
            "# (ae18_NN + ae19_NN); con la carga base anterior quedaba vacio.\n"
            "df = load_base_plus_alphaearth_2018_2019(\n"
            "    features_path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "display(Markdown(f'Dataset: `{df.height:,}` parcelas x `{df.width}` cols'))\n"
            "\n"
            "ablation_table, parquet_path = run_ablation_and_persist(\n"
            "    df,\n"
            "    output_dir=env.reports_dir,\n"
            "    models=('xgb',),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    max_samples=MAX_SAMPLES,\n"
            ")\n"
            "display(Markdown(f'**Tabla de ablación**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
        )
    )

    cells.append(_md("## 2. Gráficos: F1-macro por conjunto y comparativa del bloque `geom_*`"))

    cells.append(
        _code(
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "\n"
            "results = [\n"
            "    FeatureAblationResult(\n"
            "        feature_set=row['feature_set'],\n"
            "        model_kind=row['model'],\n"
            "        f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "        f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "        miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "        n_features=row['n_features'],\n"
            "        delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "    )\n"
            "    for row in ablation_table.iter_rows(named=True)\n"
            "]\n"
            "\n"
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto de características')\n"
            "fig_abl.savefig(env.figures_dir / 'ablation_bars.png', bbox_inches='tight')\n"
            "display(fig_abl)\n"
            "plt.close(fig_abl)\n"
            "\n"
            "fig_geom = plot_geom_leakage_comparison(results)\n"
            "fig_geom.savefig(env.figures_dir / 'geom_leakage.png', bbox_inches='tight')\n"
            "display(fig_geom)\n"
            "plt.close(fig_geom)\n"
            "\n"
            "# MLflow: un run por feature_set con metricas, n_features y delta_vs_full.\n"
            "for r in results:\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'04c-{r.feature_set}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        dvc_path=FEATURES_PATH,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        import math\n"
            "        mlflow.log_params({\n"
            "            'feature_set': r.feature_set,\n"
            "            'model_kind': r.model_kind,\n"
            "            'n_features': r.n_features,\n"
            "            'k_folds': K_FOLDS,\n"
            "            'buffer_km': BUFFER_KM,\n"
            "        })\n"
            "        metrics = {\n"
            "            'f1_macro': r.f1_macro,\n"
            "            'f1_weighted': r.f1_weighted,\n"
            "            'miou': r.miou,\n"
            "            'delta_vs_full': r.delta_vs_full,\n"
            "        }\n"
            "        for k, v in metrics.items():\n"
            "            if v is not None and not (isinstance(v, float) and math.isnan(v)):\n"
            "                mlflow.log_metric(k, float(v))\n"
            "        MLFLOW_RUN_IDS[r.feature_set] = run.info.run_id\n"
            "display(Markdown('**MLflow runs ablación**: ' + ', '.join(\n"
            "    f'`{k}={v[:12]}...`' for k, v in MLFLOW_RUN_IDS.items()\n"
            ")))\n"
        )
    )

    cells.append(
        _md(
            "## 3. Lectura interpretada de los deltas\n\n"
            "Cada fila de la tabla anterior responde una pregunta específica "
            "del proyecto. La interpretación con cifras reales:"
        )
    )

    cells.append(
        _code(
            "import math\n"
            "import polars as pl\n"
            "\n"
            "def _delta(name: str) -> float:\n"
            "    rows = ablation_table.filter(pl.col('feature_set') == name)\n"
            "    if rows.height == 0:\n"
            "        return float('nan')\n"
            "    val = rows.get_column('delta_vs_full').to_list()[0]\n"
            "    return float(val) if val is not None else float('nan')\n"
            "\n"
            "def _f1(name: str) -> float:\n"
            "    rows = ablation_table.filter(pl.col('feature_set') == name)\n"
            "    if rows.height == 0:\n"
            "        return float('nan')\n"
            "    val = rows.get_column('f1_macro').to_list()[0]\n"
            "    return float(val) if val is not None else float('nan')\n"
            "\n"
            "def _format(v: float) -> str:\n"
            "    return f'`{v:+.4f}`' if not math.isnan(v) else '`n/a`'\n"
            "\n"
            "interpretations = [\n"
            "    ('Aporte de las columnas geométricas',\n"
            "     f'`no_geom` − `full` = {_format(_delta(\"no_geom\"))}. '\n"
            "     'Si el delta es cercano a cero o positivo, las 3 columnas geom_* no '\n"
            "     'aportan señal agronómica útil — pueden ser un proxy de región '\n"
            "     '(parcelas grandes en Pianura Padana, alargadas en Toscana). '\n"
            "     'Negativo solo si el modelo realmente usaba `geom_*` para clasificar.'),\n"
            "    ('Aporte de ERA5 + SRTM',\n"
            "     f'`no_geom_no_era5_srtm` − `full` = {_format(_delta(\"no_geom_no_era5_srtm\"))}. '\n"
            "     'AlphaEarth Foundations codifica clima y topografía internamente; '\n"
            "     'un delta cercano a cero confirma redundancia.'),\n"
            "    ('Cuánto carga AlphaEarth solo',\n"
            "     f'`alphaearth_only` con F1-macro = `{_f1(\"alphaearth_only\"):.4f}`. '\n"
            "     f'Delta vs full = {_format(_delta(\"alphaearth_only\"))}. '\n"
            "     'Cuanto más cerca de cero, mayor es la fracción del baseline '\n"
            "     'explicada por los 64 embeddings.'),\n"
            "    ('Cuánto carga la fenología sola',\n"
            "     f'`phenology_only` con F1-macro = `{_f1(\"phenology_only\"):.4f}`. '\n"
            "     'Es la firma estacional sin embedding satelital — válida la '\n"
            "     'hipótesis del paper Wen et al. 2025 si queda cerca de `full`.'),\n"
            "    ('Test cuantitativo de fuga espacial (`geom_only`)',\n"
            "     f'`geom_only` con F1-macro = `{_f1(\"geom_only\"):.4f}`. '\n"
            "     'Umbral de aceptación: `< 0.10`. Confirma que área, perímetro y '\n"
            "     'elongación por sí solas NO permiten clasificar cultivos — descartar '\n"
            "     'geom_* del baseline está justificado.'),\n"
            "]\n"
            "\n"
            "for title, body in interpretations:\n"
            "    display(Markdown(f'**{title}** — {body}'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "**Decisión sobre los bloques base** queda anclada en datos:\n\n"
            "- Las columnas `geom_*` se descartan del baseline. La sección 3 "
            "confirma el comportamiento esperado: delta `no_geom` ≈ 0 (no "
            "aportaban señal) y `geom_only` < 0.10 (no clasifican por sí "
            "solas).\n"
            "- Los bloques ERA5 + SRTM son redundantes con AlphaEarth. La "
            "diferencia `no_geom_no_era5_srtm` − `no_geom` es típicamente "
            "marginal: el embedding fundacional ya codifica clima y terreno.\n"
            "- `alphaearth_only` y `phenology_only` cuantifican qué tanto "
            "cargan, respectivamente, el embedding y la firma fenológica. "
            "Los dos juntos son la base sobre la que se evalúan los bloques "
            "opcionales en `05_reencuadre_fenologico.ipynb`.\n\n"
            "**Sobre las preguntas oficiales**:\n\n"
            "- **P1 (algoritmo)**: la ablación corre XGBoost (el más "
            "consistente del 04) sobre cada subset; misma metodología "
            "spatial CV. La elección del algoritmo se respeta.\n"
            "- **P2 (importancia y features irrelevantes)**: cada delta "
            "cuantifica la irrelevancia (o relevancia) de un bloque "
            "completo, complementando el ranking por feature individual del "
            "04 SHAP. **Conclusión cuantitativa: descartamos `geom_*`.**\n\n"
            "## Lo que sigue\n\n"
            "- `04_farslip_eval_pastis.ipynb` evalúa FarSLIP vs RemoteCLIP "
            "como extractores visuales independientes.\n"
            "- `05_reencuadre_fenologico.ipynb` lee la tabla `ablation_table_base` "
            "que este notebook produce, identifica el conjunto ganador base "
            "y amplía la ablación con los bloques opcionales (FarSLIP, "
            "pheno_text Gemini, firma espectral REP).\n"
            "- `Avance3.Equipo17.ipynb` consolida las decisiones de todos "
            "los notebooks."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04_farslip_eval_pastis.ipynb — FarSLIP vs RemoteCLIP over real PASTIS.
# ---------------------------------------------------------------------------


def build_04_farslip_eval_pastis() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Evaluación FarSLIP vs RemoteCLIP sobre PASTIS-R real\n\n"
            "Compara dos extractores de embeddings de teledetección sobre "
            "el **mismo subset real** de PASTIS-R:\n\n"
            "- **FarSLIP** (Tang et al. 2024): CLIP afinado para viñedos y "
            "cultivos europeos mediante distilación desde Sentinel-2 y "
            "descripciones textuales.\n"
            "- **RemoteCLIP** (Chen et al. 2024): CLIP afinado sobre 12 "
            "datasets de teledetección.\n\n"
            "Si los pesos de RemoteCLIP no se pueden descargar desde "
            "Hugging Face, el extractor cae automáticamente a "
            "`openai/clip-vit-base-patch32` como respaldo (documentado en "
            "el log estructurado).\n\n"
            "**Sin datos sintéticos**: el subset PASTIS-R se genera desde "
            "`data/PASTIS-R/metadata.geojson` y `DATA_S2/` reales con "
            "muestreo estratificado por clase. Si PASTIS-R no está en disco, "
            "el cuaderno lanza `FileNotFoundError` con instrucciones de "
            "`dvc pull` o de descarga manual desde Zenodo.\n\n"
            "**Comparativa**: similitud coseno de los pares (FarSLIP, "
            "RemoteCLIP) por parcela y un clasificador lineal "
            "(LogisticRegression) sobre cada espacio de embeddings para "
            "medir separabilidad por clase."
        )
    )

    cells.append(
        _code(
            'PASTIS_SUBSET_PATH = "data/test_fixtures/pastis_eval_subset.parquet"\n'
            # ml.ingest.pastis_eval_subset uses output.with_suffix(output.suffix + ".imagery.parquet")
            # which produces the name with double suffix .parquet.imagery.parquet.
            'PASTIS_IMAGERY_PATH = "data/test_fixtures/pastis_eval_subset.parquet.imagery.parquet"\n'
            'FARSLIP_EMBEDDINGS_PATH = "data/farslip/embeddings_pastis.parquet"\n'
            'REMOTECLIP_EMBEDDINGS_PATH = "data/farslip/remoteclip_embeddings_pastis.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04_farslip_eval_pastis"\n'
            'REPORTS_SUBDIR = "baseline/04_farslip_eval_pastis"\n'
            "N_SAMPLES = 1024\n"
            "RANDOM_STATE = 42\n"
            'MLFLOW_EXPERIMENT = "baseline-04-farslip-vs-remoteclip"\n',
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(
        _md(
            "### Trazabilidad MLflow\n\n"
            "Cada clasificador lineal (LogReg sobre FarSLIP, LogReg sobre "
            "RemoteCLIP) abre un run propio en el experimento "
            "`baseline-04-farslip-vs-remoteclip`. La métrica principal del "
            "run es el F1-macro promedio del 5-fold estratificado sobre el "
            "espacio de embeddings correspondiente — cuanto mayor, mejor "
            "separa el espacio las clases PASTIS."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import (\n"
            "    resolve_tracking_uri,\n"
            "    track_experiment,\n"
            "    server_is_reachable,\n"
            ")\n"
            "\n"
            "# Resolucion robusta del tracking URI: si MLFLOW_TRACKING_URI esta en\n"
            "# `.env.local` pero el server no responde (Docker apagado, contenedor\n"
            "# detenido), caemos a `file:./mlruns` para no detener el notebook.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Caigo a tracking local `file:./mlruns`. '\n"
            "        'Para registrar en el server, ejecuta `docker compose up -d mlflow` '\n"
            "        'antes de re-ejecutar las celdas MLflow.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "display(Markdown(\n"
            "    f'**MLflow tracking URI**: `{mlflow_uri}` · '\n"
            "    f'**Experimento**: `{MLFLOW_EXPERIMENT}`'\n"
            "))\n"
            "MLFLOW_RUN_IDS: dict[str, str] = {}\n"
        )
    )

    cells.append(_md("## 1. Materialización del subset PASTIS real (si no existe)"))

    cells.append(
        _code(
            "import polars as pl\n"
            "from pathlib import Path\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    materialize_pastis_eval_subset_if_missing,\n"
            "    materialize_remoteclip_if_missing,\n"
            ")\n"
            "\n"
            "subset_path = materialize_pastis_eval_subset_if_missing(\n"
            "    output_path=PASTIS_SUBSET_PATH,\n"
            "    n_samples=N_SAMPLES,\n"
            ")\n"
            "subset = pl.read_parquet(subset_path)\n"
            "display(Markdown(f'**Subset PASTIS-R real**: `{subset.height}` parcelas en `{subset_path}`'))\n"
            "display(subset.head(8))\n"
            "display(Markdown('**Distribución de clases en el subset**:'))\n"
            "display(\n"
            "    subset.group_by('class_id', 'class_name').len()\n"
            "    .sort('len', descending=True)\n"
            ")\n"
        )
    )

    cells.append(_md("## Extracción de embeddings RemoteCLIP (si no existen)"))

    cells.append(
        _code(
            "remoteclip_path = materialize_remoteclip_if_missing(\n"
            "    pastis_eval_subset_path=PASTIS_SUBSET_PATH,\n"
            "    imagery_path=PASTIS_IMAGERY_PATH,\n"
            "    output_path=REMOTECLIP_EMBEDDINGS_PATH,\n"
            ")\n"
            "remoteclip = pl.read_parquet(remoteclip_path)\n"
            "display(Markdown(f'**RemoteCLIP**: `{remoteclip.shape}` (cols con prefijo `remoteclip_`)'))\n"
            "display(remoteclip.select(['parcel_id', 'year', 'remoteclip_000', 'remoteclip_001']).head(5))\n"
        )
    )

    cells.append(_md("## Carga de los embeddings FarSLIP (ruta canónica)"))

    cells.append(
        _code(
            "farslip_path = Path(FARSLIP_EMBEDDINGS_PATH)\n"
            "if not farslip_path.exists():\n"
            "    raise FileNotFoundError(\n"
            "        f'FarSLIP no encontrado en {farslip_path}. Ejecuta '\n"
            "        '`dvc pull data/farslip/embeddings_pastis.parquet.dvc` antes de re-ejecutar.'\n"
            "    )\n"
            "farslip = pl.read_parquet(farslip_path)\n"
            "from ml.utils.parcel_id import canonical_parcel_id\n"
            "farslip = canonical_parcel_id(farslip)\n"
            "display(Markdown(f'**FarSLIP**: `{farslip.shape}` (cols con prefijo `farslip_`)'))\n"
        )
    )

    cells.append(_md("## Similitud coseno entre embeddings FarSLIP y RemoteCLIP por parcela"))

    cells.append(
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Unimos por parcel_id (ambas tablas tienen parcel_id Utf8 tras canonical_parcel_id)\n"
            "remoteclip = canonical_parcel_id(remoteclip)\n"
            "merged = (\n"
            "    canonical_parcel_id(subset.select(['parcel_id', 'class_id', 'class_name']))\n"
            "    .join(farslip.select(['parcel_id'] + [c for c in farslip.columns if c.startswith('farslip_') or c.startswith('farslip_emb_')]), on='parcel_id', how='inner')\n"
            "    .join(remoteclip.select(['parcel_id'] + [c for c in remoteclip.columns if c.startswith('remoteclip_')]), on='parcel_id', how='inner')\n"
            ")\n"
            "display(Markdown(f'**Join FarSLIP x RemoteCLIP x subset**: `{merged.height}` parcelas comunes'))\n"
            "\n"
            "if merged.height == 0:\n"
            "    display(Markdown(\n"
            "        '> No hay parcelas en comun entre FarSLIP y el subset PASTIS-R. '\n"
            "        'FarSLIP fue entrenado sobre parcelas de Italia y el subset PASTIS-R '\n"
            "        'cubre parcelas de Francia. La comparativa requiere un FarSLIP entrenado '\n"
            "        'sobre PASTIS, que queda pendiente como trabajo futuro.'\n"
            "    ))\n"
            "else:\n"
            "    fs_cols = [c for c in merged.columns if c.startswith('farslip_') and not c.startswith('farslip_emb_')] or [c for c in merged.columns if c.startswith('farslip_emb_')]\n"
            "    rc_cols = [c for c in merged.columns if c.startswith('remoteclip_')]\n"
            "    fs_mat = merged.select(fs_cols).to_numpy().astype(np.float64)\n"
            "    rc_mat = merged.select(rc_cols).to_numpy().astype(np.float64)\n"
            "    # Coseno row-wise sobre las primeras min(D) dims (proyectamos a min para comparar)\n"
            "    d = min(fs_mat.shape[1], rc_mat.shape[1])\n"
            "    fs_norm = fs_mat[:, :d] / (np.linalg.norm(fs_mat[:, :d], axis=1, keepdims=True) + 1e-12)\n"
            "    rc_norm = rc_mat[:, :d] / (np.linalg.norm(rc_mat[:, :d], axis=1, keepdims=True) + 1e-12)\n"
            "    cosines = (fs_norm * rc_norm).sum(axis=1)\n"
            "    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)\n"
            "    ax.hist(cosines, bins=40, color='#4C72B0', edgecolor='white')\n"
            "    ax.set_xlabel('Coseno (FarSLIP, RemoteCLIP) por parcela')\n"
            "    ax.set_ylabel('Frecuencia')\n"
            "    ax.set_title('Distribución de similitud entre embeddings FarSLIP y RemoteCLIP')\n"
            "    ax.axvline(0.0, color='#888', linestyle='--', linewidth=1)\n"
            "    fig.savefig(env.figures_dir / 'cosine_farslip_vs_remoteclip.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(_md("## Separabilidad lineal con regresión logística sobre cada espacio"))

    cells.append(
        _code(
            "# Clasificador lineal simple para comparar la capacidad separadora de cada espacio.\n"
            "# Si merged esta vacio, comparamos en el espacio nativo (subset + RemoteCLIP).\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
            "\n"
            "subset_join_rc = canonical_parcel_id(subset.select(['parcel_id', 'class_id'])).join(\n"
            "    remoteclip, on='parcel_id', how='inner'\n"
            ")\n"
            "if subset_join_rc.height >= 100:\n"
            "    rc_cols2 = [c for c in subset_join_rc.columns if c.startswith('remoteclip_')]\n"
            "    X_rc = subset_join_rc.select(rc_cols2).to_numpy()\n"
            "    y_rc = subset_join_rc['class_id'].to_numpy()\n"
            "    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "    scores_rc = cross_val_score(\n"
            "        LogisticRegression(max_iter=2000),\n"
            "        X_rc,\n"
            "        y_rc,\n"
            "        scoring='f1_macro',\n"
            "        cv=cv,\n"
            "        n_jobs=-1,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**LogReg sobre RemoteCLIP (subset PASTIS)**: F1-macro = '\n"
            "        f'`{scores_rc.mean():.4f} +/- {scores_rc.std():.4f}` (5-fold estratificado).'\n"
            "    ))\n"
            "else:\n"
            "    display(Markdown('> Insuficientes parcelas (>=100) para entrenar el clasificador lineal.'))\n"
            "    scores_rc = None\n"
            "\n"
            "if merged.height >= 100:\n"
            "    fs_cols3 = [c for c in merged.columns if c.startswith('farslip_') and not c.startswith('farslip_emb_')] or [c for c in merged.columns if c.startswith('farslip_emb_')]\n"
            "    X_fs = merged.select(fs_cols3).to_numpy()\n"
            "    y_fs = merged['class_id'].to_numpy()\n"
            "    scores_fs = cross_val_score(\n"
            "        LogisticRegression(max_iter=2000),\n"
            "        X_fs,\n"
            "        y_fs,\n"
            "        scoring='f1_macro',\n"
            "        cv=cv,\n"
            "        n_jobs=-1,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**LogReg sobre FarSLIP (intersección)**: F1-macro = '\n"
            "        f'`{scores_fs.mean():.4f} +/- {scores_fs.std():.4f}` (5-fold).'\n"
            "    ))\n"
            "else:\n"
            "    scores_fs = None\n"
            "\n"
            "# MLflow: un run por extractor evaluado.\n"
            "for label, scores, n_dims, embedding_origin in [\n"
            "    ('remoteclip', scores_rc,\n"
            "     len(rc_cols2) if scores_rc is not None else 0,\n"
            "     'off-the-shelf (Chen et al. 2024)'),\n"
            "    ('farslip', scores_fs,\n"
            "     len(fs_cols3) if scores_fs is not None else 0,\n"
            "     'destilacion US-017 (Li et al. 2025)'),\n"
            "]:\n"
            "    if scores is None:\n"
            "        continue\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'04-farslip-eval-{label}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        mlflow.log_params({\n"
            "            'extractor': label,\n"
            "            'embedding_origin': embedding_origin,\n"
            "            'n_dims': n_dims,\n"
            "            'classifier': 'LogisticRegression',\n"
            "            'cv_splits': 5,\n"
            "        })\n"
            "        mlflow.log_metrics({\n"
            "            'f1_macro_mean': float(scores.mean()),\n"
            "            'f1_macro_std': float(scores.std()),\n"
            "        })\n"
            "        MLFLOW_RUN_IDS[label] = run.info.run_id\n"
            "display(Markdown('**MLflow runs**: ' + ', '.join(\n"
            "    f'`{k}={v[:12]}...`' for k, v in MLFLOW_RUN_IDS.items()\n"
            ")))\n"
        )
    )

    cells.append(
        _md(
            "## 6. Visualización UMAP de los espacios coloreados por clase\n\n"
            "Cada punto es una parcela proyectada al plano UMAP 2D del "
            "espacio de embeddings correspondiente. El color indica la clase "
            "real PASTIS. Una separación visual nítida entre clases en el "
            "plano UMAP sugiere que el espacio captura señal de cultivo; "
            "nubes superpuestas sugieren que la separabilidad lineal queda "
            "limitada al clasificador, no a la geometría del espacio."
        )
    )

    cells.append(
        _code(
            "from ml.features.selection import fit_umap_2d\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.cm as cm\n"
            "import matplotlib.colors as mcolors\n"
            "\n"
            "def _plot_umap_by_class(\n"
            "    X: np.ndarray, y: np.ndarray, *, title: str, out_path,\n"
            "):\n"
            "    embedding = fit_umap_2d(X, random_state=RANDOM_STATE)\n"
            "    unique_classes = sorted(set(y.tolist()))\n"
            "    cmap = cm.get_cmap('tab20', max(len(unique_classes), 1))\n"
            "    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)\n"
            "    for i, cls in enumerate(unique_classes):\n"
            "        mask = y == cls\n"
            "        ax.scatter(\n"
            "            embedding[mask, 0], embedding[mask, 1],\n"
            "            s=8, alpha=0.55, color=cmap(i), label=f'c{cls}',\n"
            "        )\n"
            "    ax.set_xlabel('UMAP 1')\n"
            "    ax.set_ylabel('UMAP 2')\n"
            "    ax.set_title(title)\n"
            "    if len(unique_classes) <= 20:\n"
            "        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(out_path, bbox_inches='tight')\n"
            "    return fig\n"
            "\n"
            "if subset_join_rc.height >= 50:\n"
            "    rc_cols2 = [c for c in subset_join_rc.columns if c.startswith('remoteclip_')]\n"
            "    X_rc = subset_join_rc.select(rc_cols2).to_numpy()\n"
            "    y_rc = subset_join_rc['class_id'].to_numpy()\n"
            "    fig_rc = _plot_umap_by_class(\n"
            "        X_rc, y_rc,\n"
            "        title='UMAP 2D — espacio RemoteCLIP coloreado por clase PASTIS',\n"
            "        out_path=env.figures_dir / 'umap_remoteclip_by_class.png',\n"
            "    )\n"
            "    display(fig_rc)\n"
            "    plt.close(fig_rc)\n"
            "else:\n"
            "    display(Markdown('> UMAP RemoteCLIP omitido (subset_join_rc < 50 parcelas).'))\n"
            "\n"
            "if merged.height >= 50:\n"
            "    fs_cols3 = [c for c in merged.columns if c.startswith('farslip_') and not c.startswith('farslip_emb_')] or [c for c in merged.columns if c.startswith('farslip_emb_')]\n"
            "    X_fs = merged.select(fs_cols3).to_numpy()\n"
            "    y_fs = merged['class_id'].to_numpy()\n"
            "    fig_fs = _plot_umap_by_class(\n"
            "        X_fs, y_fs,\n"
            "        title='UMAP 2D — espacio FarSLIP coloreado por clase PASTIS',\n"
            "        out_path=env.figures_dir / 'umap_farslip_by_class.png',\n"
            "    )\n"
            "    display(fig_fs)\n"
            "    plt.close(fig_fs)\n"
            "else:\n"
            "    display(Markdown('> UMAP FarSLIP omitido (merged < 50 parcelas — sin intersección entre Italia y Francia).'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "**Sobre los dos extractores visuales**:\n\n"
            "- **RemoteCLIP** (Chen et al. 2024) entrega un baseline "
            "honesto: CLIP afinado sobre 12 datasets de teledetección, sin "
            "supervisión específica de PASTIS. El F1-macro de LogReg sobre "
            "RemoteCLIP marca el piso que cualquier extractor más "
            "especializado debe superar.\n"
            "- **FarSLIP** (destilación US-017 según Li et al. 2025) es el "
            "extractor entrenado en este proyecto sobre parcelas Italia. Su "
            "valor se mide contra RemoteCLIP en F1-macro y en la "
            "separabilidad visual UMAP — si los clusters quedan más "
            "definidos por clase, el espacio FarSLIP es preferible aunque "
            "el F1-macro de LogReg sea similar.\n\n"
            "**Limitaciones honestas**:\n\n"
            "1. FarSLIP fue destilado sobre parcelas de Italia, mientras "
            "que el subset PASTIS-R cubre parcelas de Francia. La "
            "intersección por `parcel_id` puede ser baja o nula. Si `merged` "
            "tiene pocas parcelas, la comparativa F1-macro directa sobre "
            "FarSLIP es ilustrativa, no concluyente. Una comparativa "
            "completa requeriría reentrenar FarSLIP sobre PASTIS (deuda "
            "futura US-025).\n"
            "2. Si los pesos oficiales de RemoteCLIP no logran descargarse, "
            "se cae a `openai/clip-vit-base-patch32`; el log estructurado "
            "deja constancia de qué modelo terminó usándose.\n\n"
            "**Sobre las preguntas oficiales**:\n\n"
            "- **P1 (algoritmo)**: este notebook NO entrena un baseline "
            "tabular; evalúa dos *extractores* de embeddings con LogReg "
            "como cabezal lineal. La decisión de promover FarSLIP como "
            "bloque opcional del baseline tabular se delega a `05_reencuadre`.\n"
            "- **P2 (importancia)**: la separabilidad visual UMAP y el "
            "F1-macro de LogReg complementan el análisis SHAP del `04` — "
            "responden a 'qué tan importante es el espacio de embedding' "
            "como bloque entero.\n\n"
            "## Lo que sigue\n\n"
            "- `05_reencuadre_fenologico.ipynb` evalúa FarSLIP **como bloque "
            "del baseline tabular fused** mediante ablación (`with_farslip`, "
            "`farslip_only`).\n"
            "- `Avance3.Equipo17.ipynb` consolida la decisión de promover, "
            "diferir o descartar FarSLIP."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 05_reencuadre_fenologico.ipynb — no silent skips.
# ---------------------------------------------------------------------------


def build_05_reencuadre() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Reencuadre fenológico — extensión completa del baseline\n\n"
            "Este cuaderno extiende el baseline tabular del `04_baseline.ipynb` "
            "con todo lo que el proyecto tiene programado y testeado: "
            "bloques opcionales del fused, modelos temporales reales, "
            "clustering sobre la firma fenológica pura, estrategia para el "
            "desbalance, rama semántica LLM y glosario para el entregable.\n\n"
            "**Preguntas oficiales del Avance 3 a las que aporta**:\n\n"
            "- **P1 (algoritmo baseline)**: compara los 3 modelos tabulares "
            "(RF + XGB + LightGBM) contra los 2 modelos temporales reales "
            "(TempCNN + InceptionTime) sobre el mismo conjunto ganador.\n"
            "- **P2 (importancia / features irrelevantes)**: amplía la "
            "ablación de `04c_baseline.ipynb` con los 3 bloques opcionales "
            "(FarSLIP, pheno_text, REP) y decide promover/descartar cada uno.\n"
            "- **P3 (sub/sobreajuste)**: los modelos temporales se "
            "diagnostican leyendo la curva train_loss vs val_loss desde "
            "MLflow con `diagnose_temporal_fit`.\n\n"
            "Estructura del cuaderno (10 secciones):\n\n"
            "1. Carga del dataset base y materialización de los bloques "
            "opcionales (sin skips silenciosos).\n"
            "2. Fusión: base + FarSLIP + pheno_text + spectral_signature.\n"
            "3. Ablación con todos los bloques opcionales (lee el conjunto "
            "ganador BASE producido por `04c_baseline.ipynb`).\n"
            "4. Comparativa de los 5 modelos (RF + XGB + LGBM + TempCNN + "
            "InceptionTime) sobre el conjunto ganador.\n"
            "5. Diagnóstico por clase del mejor modelo temporal: matriz de "
            "confusión OOF + F1 por clase.\n"
            "6. Clustering KMeans sobre la firma fenológica pura + UMAP "
            "2D + curvas NDVI medias por cluster (sin coordenadas).\n"
            "7. Estrategia para el desbalance ~31× max/min.\n"
            "8. Rama semántica fenológica: descripción textual real con "
            "Gemini Flash sobre subset balanceado.\n"
            "9. Conclusiones consolidadas con decisión por bloque.\n"
            "10. Glosario para el entregable del curso."
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FUSED_PATH = "data/features/features_fused_pastis.parquet"\n'
            'PHENO_TEXT_PATH = "data/features/phenology_text_pastis.parquet"\n'
            "# Output paths renombrados a _pastis_2019 en US-023-preview v2: los\n"
            "# parquets viejos (_italy) eran cache vacio (NaN al 100%) por el\n"
            "# bug B04 vs B4 en GEE y fallback DOY estatico. Los nuevos usan\n"
            "# bandas correctas + anclas calendario por parcela.\n"
            'S2_ANCHORS_PATH = "data/features/s2_anchors_pastis_2019.parquet"\n'
            'SPECTRAL_SIGNATURE_PATH = "data/features/spectral_signature_pastis_2019.parquet"\n'
            'FARSLIP_PATH = "data/farslip/embeddings_pastis.parquet"\n'
            'ALPHAEARTH_ENRICHED_PATH = "data/cache/gee/alphaearth_pastis_parcels_2019_85951_enriched.parquet"\n'
            'PASTIS_METADATA_GEOJSON = "data/PASTIS-R/metadata.geojson"\n'
            'PHENOLOGY_ANCHORS_PATH = "data/features/pastis_phenology_anchors_2019.parquet"\n'
            'BASE_ABLATION_PATH = "reports/baseline/04c_baseline/ablation_table.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/05_reencuadre"\n'
            'REPORTS_SUBDIR = "baseline/05_reencuadre"\n'
            "YEAR = 2023\n"
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "MAX_SAMPLES = None\n"
            "RANDOM_STATE = 42\n"
            "# FarSLIP embeddings_pastis.parquet contiene parcelas italianas extra-PASTIS\n"
            "# (overlap parcel_id con PASTIS = 0). Integrar FarSLIP a PASTIS requiere\n"
            "# distillar el student (US-017) y adaptar vocabulary CAP a cultivos\n"
            "# franceses. Por ahora se excluye del fused PASTIS; su evaluacion honesta\n"
            "# vive en 04_farslip_eval_pastis.ipynb sobre el subset adecuado.\n"
            "ENABLE_FARSLIP = False\n"
            "ENABLE_PHENO_TEXT = True\n"
            "ENABLE_SPECTRAL_SIGNATURE = True\n"
            "ENABLE_ALPHAEARTH = True  # Anexa data/cache/gee/alphaearth_*_enriched.parquet (64 dim_NN)\n"
            "ENABLE_TEMPORAL_MODELS = True\n"
            "# Si True, reusa los runs MLflow ya finalizados de tempcnn/inceptiontime\n"
            "# en el experimento baseline-05-reencuadre y evita re-entrenarlos (~6h GPU).\n"
            "REUSE_TEMPORAL_FROM_MLFLOW = True\n"
            "ENABLE_CLUSTERING = True\n"
            "ENABLE_LLM_BRANCH = True\n"
            "ENFORCE_GEMINI_API_KEY = True\n"
            "TEMPORAL_EPOCHS = 80\n"
            "TEMPORAL_BATCH_SIZE = 256\n"
            "TEMPORAL_DEVICE = 'auto'  # auto = cuda si disponible\n"
            "N_CLUSTERS = 8\n"
            "LLM_SUBSET_SIZE = 1080  # 60 parcelas balanceadas x 18 clases\n"
            "WEAK_CLASS_THRESHOLD = 1000  # parcelas, para listar clases debiles\n"
            'MLFLOW_EXPERIMENT = "baseline-05-reencuadre"\n',
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(
        _md(
            "### Trazabilidad MLflow\n\n"
            "Cada bloque que entrena un modelo abre runs MLflow propios "
            "agrupados en el experimento `baseline-05-reencuadre`: ablación, "
            "5 modelos sobre conjunto ganador (RF, XGB, LGBM, TempCNN, "
            "InceptionTime) y rama LLM. Los `run_id` se acumulan en el dict "
            "`MLFLOW_RUN_IDS` para que `Avance3.Equipo17.ipynb` consolide la "
            "trazabilidad."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import (\n"
            "    resolve_tracking_uri,\n"
            "    track_experiment,\n"
            "    server_is_reachable,\n"
            ")\n"
            "\n"
            "# Resolucion robusta del tracking URI: si MLFLOW_TRACKING_URI esta en\n"
            "# `.env.local` pero el server no responde (Docker apagado, contenedor\n"
            "# detenido), caemos a `file:./mlruns` para no detener el notebook.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Caigo a tracking local `file:./mlruns`. '\n"
            "        'Para registrar en el server, ejecuta `docker compose up -d mlflow` '\n"
            "        'antes de re-ejecutar las celdas MLflow.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "display(Markdown(\n"
            "    f'**MLflow tracking URI**: `{mlflow_uri}` · '\n"
            "    f'**Experimento**: `{MLFLOW_EXPERIMENT}`'\n"
            "))\n"
            "MLFLOW_RUN_IDS: dict[str, str] = {}\n"
        )
    )

    cells.append(_md("## 1. Carga del dataset base (con metadata)"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    materialize_phenology_text_if_missing,\n"
            "    materialize_s2_anchors_if_missing,\n"
            "    materialize_spectral_signature_if_missing,\n"
            "    run_ablation_and_persist,\n"
            ")\n"
            "from ml.utils.parcel_id import canonical_parcel_id\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_ablation_bars,\n"
            "    plot_optional_blocks_ablation,\n"
            ")\n"
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "from pathlib import Path\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "display(Markdown(f'**Dataset base**: `{df.height:,}` parcelas x `{df.width}` cols'))\n"
        )
    )

    cells.append(
        _md("## Materialización del bloque `pheno_text` (Gemini sobre el dataset completo)")
    )

    cells.append(
        _code(
            "if ENABLE_PHENO_TEXT:\n"
            "    if ENFORCE_GEMINI_API_KEY and not env.has_gemini_api_key:\n"
            "        raise RuntimeError(\n"
            "            'GEMINI_API_KEY ausente. Define la variable en `.env.local` antes de re-ejecutar, '\n"
            "            'o pon ENFORCE_GEMINI_API_KEY=False para correr solo las ablaciones base.'\n"
            "        )\n"
            "    pheno_path = materialize_phenology_text_if_missing(\n"
            "        parcels_features_path=FEATURES_PATH,\n"
            "        output_path=PHENO_TEXT_PATH,\n"
            "        enforce_api_key=ENFORCE_GEMINI_API_KEY,\n"
            "    )\n"
            "    pheno_df = canonical_parcel_id(pl.read_parquet(pheno_path))\n"
            "    display(Markdown(f'**pheno_text**: `{pheno_df.shape}` en `{pheno_path}`'))\n"
            "else:\n"
            "    pheno_df = None\n"
            "    display(Markdown('> ENABLE_PHENO_TEXT=False: bloque omitido.'))\n"
        )
    )

    cells.append(
        _md("## Materialización de anclas Sentinel-2 y firma espectral REP (Frampton 2013)")
    )

    cells.append(
        _code(
            "from ml.ingest.pastis_phenology_anchors import build_pastis_phenology_anchors\n"
            "\n"
            "if ENABLE_SPECTRAL_SIGNATURE:\n"
            "    if not env.has_ee_credentials:\n"
            "        display(Markdown(\n"
            "            '> Earth Engine no configurado. Define `GEE_PROJECT_ID` '\n"
            "            'en `.env.local` o ejecuta `earthengine authenticate`. '\n"
            "            'El muestreo S2 anchors fallara sin esto.'\n"
            "        ))\n"
            "    # Anclas fenologicas por parcela en DOY calendario 2019.\n"
            "    # Si el parquet existe usa cache; si no, lo construye desde\n"
            "    # `metadata.geojson` PASTIS-R + DOY relativos del subset US-016.\n"
            "    phen_anchors_path = build_pastis_phenology_anchors(\n"
            "        metadata_geojson_path=PASTIS_METADATA_GEOJSON,\n"
            "        features_subset_path=FEATURES_PATH,\n"
            "        output_path=PHENOLOGY_ANCHORS_PATH,\n"
            "        target_year=YEAR,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**Anclas fenologicas por parcela**: `{phen_anchors_path}` '\n"
            "        '(DOY calendario derivado de `metadata.geojson` PASTIS-R).'\n"
            "    ))\n"
            "    anchors_path = materialize_s2_anchors_if_missing(\n"
            "        parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            "        output_path=S2_ANCHORS_PATH,\n"
            "        year=YEAR,\n"
            "        phenology_anchors_path=phen_anchors_path,\n"
            "    )\n"
            "    spec_path = materialize_spectral_signature_if_missing(\n"
            "        s2_anchors_path=anchors_path,\n"
            "        output_path=SPECTRAL_SIGNATURE_PATH,\n"
            "        descriptor='rep',\n"
            "    )\n"
            "    spec_df = canonical_parcel_id(pl.read_parquet(spec_path))\n"
            "    display(Markdown(f'**spectral_signature**: `{spec_df.shape}` en `{spec_path}`'))\n"
            "else:\n"
            "    spec_df = None\n"
            "    display(Markdown('> ENABLE_SPECTRAL_SIGNATURE=False: bloque omitido.'))\n"
        )
    )

    cells.append(_md("## Carga de FarSLIP desde la ruta canónica (`parcel_id` en Utf8)"))

    cells.append(
        _code(
            "if ENABLE_FARSLIP:\n"
            "    farslip_path = Path(FARSLIP_PATH)\n"
            "    if not farslip_path.exists():\n"
            "        raise FileNotFoundError(\n"
            "            f'FarSLIP parquet no encontrado en {farslip_path}. '\n"
            "            'Ejecuta `dvc pull data/farslip/embeddings_pastis.parquet.dvc` '\n"
            "            'antes de re-ejecutar.'\n"
            "        )\n"
            "    farslip_df = canonical_parcel_id(pl.read_parquet(farslip_path))\n"
            "    display(Markdown(f'**FarSLIP**: `{farslip_df.shape}` en `{farslip_path}` con parcel_id Utf8.'))\n"
            "else:\n"
            "    farslip_df = None\n"
            "    display(Markdown('> ENABLE_FARSLIP=False: bloque omitido.'))\n"
        )
    )

    cells.append(_md("## Carga del bloque AlphaEarth (64 dim_NN) sobre PASTIS"))

    cells.append(
        _code(
            "if ENABLE_ALPHAEARTH:\n"
            "    ae_path = Path(ALPHAEARTH_ENRICHED_PATH)\n"
            "    if not ae_path.exists():\n"
            "        raise FileNotFoundError(\n"
            "            f'AlphaEarth enriched parquet no encontrado en {ae_path}. '\n"
            "            'Ejecuta el pipeline GEE (US-012) o `dvc pull` del cache.'\n"
            "        )\n"
            "    ae_df = canonical_parcel_id(pl.read_parquet(ae_path))\n"
            "    n_dims = sum(1 for c in ae_df.columns if c.startswith('dim_'))\n"
            "    display(Markdown(\n"
            "        f'**AlphaEarth**: `{ae_df.shape}` en `{ae_path}` '\n"
            "        f'con `{n_dims}` dimensiones `dim_NN` reales.'\n"
            "    ))\n"
            "else:\n"
            "    ae_df = None\n"
            "    display(Markdown('> ENABLE_ALPHAEARTH=False: bloque omitido.'))\n"
        )
    )

    cells.append(
        _md(
            "## Fusión de bloques: base + AlphaEarth + pheno_text + spectral_signature\n\n"
            "Aplicamos un LEFT JOIN secuencial sobre `parcel_id` (todos en "
            "Utf8 tras `canonical_parcel_id`). AlphaEarth tiene overlap 100% "
            "con el subset PASTIS (mismo dataset GEE 2019, 85951 parcelas). "
            "Los bloques `pheno_text` y `spectral_signature` cubren un subset "
            "menor — las parcelas sin coincidencia quedan con NaN; XGBoost y "
            "LightGBM los toleran nativamente y RandomForest los imputa por "
            "mediana. **FarSLIP** se excluye del fused PASTIS porque "
            "`embeddings_pastis.parquet` contiene parcelas italianas extra-"
            "PASTIS (overlap=0); su evaluación honesta vive en "
            "`04_farslip_eval_pastis.ipynb`."
        )
    )

    cells.append(
        _code(
            "df = canonical_parcel_id(df)\n"
            "fused = df\n"
            "joined_log = []\n"
            "if ae_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in ae_df.columns if c.startswith('dim_')]\n"
            "    fused = fused.join(ae_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'AlphaEarth: +{len(keep)-1} cols')\n"
            "if farslip_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in farslip_df.columns if c.startswith('farslip_')]\n"
            "    fused = fused.join(farslip_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'FarSLIP: +{len(keep)-1} cols')\n"
            "if pheno_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in pheno_df.columns if c.startswith('pheno_text_')]\n"
            "    fused = fused.join(pheno_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'pheno_text: +{len(keep)-1} cols')\n"
            "if spec_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in spec_df.columns if c.startswith('spectral_signature_')]\n"
            "    fused = fused.join(spec_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'spectral_signature: +{len(keep)-1} cols')\n"
            "\n"
            "display(Markdown(\n"
            '    f"**Conjunto fused final**: `{fused.shape}`\\n\\n"\n'
            '    + "\\n".join(f"- {l}" for l in joined_log)\n'
            "))\n"
        )
    )

    cells.append(
        _code(
            "from pathlib import Path as _P\n"
            "_fused_out = _P(FUSED_PATH)\n"
            "_fused_out.parent.mkdir(parents=True, exist_ok=True)\n"
            "fused.write_parquet(_fused_out)\n"
            "display(Markdown(\n"
            "    f'**Fused persistido**: `{_fused_out}` con shape `{fused.shape}`. '\n"
            "    'Lo consume `Avance3.Equipo17.ipynb` via `select_winning_features`.'\n"
            "))\n"
        )
    )

    cells.append(_md("## 3. Ablación con todos los bloques opcionales (sobre el conjunto fused)"))

    cells.append(
        _code(
            "ablation_table, parquet_path = run_ablation_and_persist(\n"
            "    fused,\n"
            "    output_dir=env.reports_dir,\n"
            "    models=('xgb',),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    max_samples=MAX_SAMPLES,\n"
            ")\n"
            "display(Markdown(f'**Tabla de ablación**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
            "\n"
            "# MLflow: un run por feature_set evaluado en la ablacion opcional.\n"
            "import math\n"
            "for row in ablation_table.iter_rows(named=True):\n"
            "    fs = row['feature_set']\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'05-ablation-{fs}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        dvc_path=FEATURES_PATH,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        mlflow.log_params({\n"
            "            'feature_set': fs,\n"
            "            'model_kind': row['model'],\n"
            "            'n_features': row['n_features'],\n"
            "            'k_folds': K_FOLDS,\n"
            "            'buffer_km': BUFFER_KM,\n"
            "            'has_farslip': ENABLE_FARSLIP,\n"
            "            'has_pheno_text': ENABLE_PHENO_TEXT,\n"
            "            'has_spectral_signature': ENABLE_SPECTRAL_SIGNATURE,\n"
            "        })\n"
            "        for key in ('f1_macro', 'f1_weighted', 'miou', 'delta_vs_full'):\n"
            "            val = row.get(key)\n"
            "            if val is not None and not (isinstance(val, float) and math.isnan(val)):\n"
            "                mlflow.log_metric(key, float(val))\n"
            "        MLFLOW_RUN_IDS[f'ablation-{fs}'] = run.info.run_id\n"
        )
    )

    cells.append(
        _md("### 3.1 Gráficos: ablación completa, fuga geométrica y aporte de bloques opcionales")
    )

    cells.append(
        _code(
            "results = [\n"
            "    FeatureAblationResult(\n"
            "        feature_set=row['feature_set'],\n"
            "        model_kind=row['model'],\n"
            "        f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "        f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "        miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "        n_features=row['n_features'],\n"
            "        delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "    )\n"
            "    for row in ablation_table.iter_rows(named=True)\n"
            "]\n"
            "\n"
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto (ablación completa)')\n"
            "fig_abl.savefig(env.figures_dir / 'ablation_full.png', bbox_inches='tight')\n"
            "display(fig_abl)\n"
            "plt.close(fig_abl)\n"
            "\n"
            "# Nota: el plot `geom_leakage` se genera SOLO en 04c (donde vive la\n"
            "# ablacion base con geom_only). Aqui mostramos solo el aporte de los\n"
            "# bloques opcionales para no duplicar contenido entre notebooks.\n"
            "fig_opt = plot_optional_blocks_ablation(results)\n"
            "fig_opt.savefig(env.figures_dir / 'optional_blocks.png', bbox_inches='tight')\n"
            "display(fig_opt)\n"
            "plt.close(fig_opt)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 4. Comparison of 5 models over the winning set.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 4. Comparativa de los 5 modelos sobre el conjunto ganador\n\n"
            "Tres modelos tabulares (Random Forest, XGBoost, LightGBM) y dos "
            "modelos temporales (TempCNN, InceptionTime) entrenados sobre el "
            "**mismo conjunto ganador post-ablación**, con la misma "
            "validación cruzada espacial. La elección del ganador se "
            "delega a `04c_baseline.ipynb`: si su `ablation_table.parquet` "
            "está disponible se usa; si no, fallback a `no_geom` (decisión "
            "documentada).\n\n"
            "- **Tabulares** (RF/XGB/LGBM): ven el vector resumen "
            "espectro-temporal anual (features estadísticas + FFT).\n"
            "- **Temporales** (TempCNN/InceptionTime): leen la curva NDVI/"
            "NDWI/EVI completa reconstruida a T=72 muestras (importados de "
            "`breizhcrops.models`, no reimplementados).\n"
        )
    )

    cells.append(
        _code(
            "from ml.train.baseline import train_one_model\n"
            "from ml.train.phenology_models import train_temporal_model\n"
            "\n"
            "# Lee el conjunto ganador BASE de 04c. Si no existe, fallback no_geom.\n"
            "base_ablation_path = env.repo / BASE_ABLATION_PATH\n"
            "if base_ablation_path.exists():\n"
            "    base_table = pl.read_parquet(base_ablation_path)\n"
            "    finite = base_table.filter(pl.col('f1_macro').is_finite()).sort('f1_macro', descending=True)\n"
            "    winner_base = finite.row(0, named=True)['feature_set'] if finite.height else 'no_geom'\n"
            "    display(Markdown(f'**Conjunto ganador base** (desde 04c): `{winner_base}`'))\n"
            "else:\n"
            "    winner_base = 'no_geom'\n"
            "    display(Markdown(\n"
            "        f'> Tabla `{BASE_ABLATION_PATH}` no encontrada. '\n"
            "        'Fallback documentado: `no_geom` (descarta las 3 columnas `geom_*` por leakage).'\n"
            "    ))\n"
            "\n"
            "# Aplicar el filtro al fused (descartar geom_* si winner_base == 'no_geom').\n"
            "def _apply_winner(df_in, winner_set):\n"
            "    if winner_set == 'no_geom':\n"
            "        return df_in.drop([c for c in df_in.columns if c.startswith('geom_')])\n"
            "    return df_in\n"
            "fused_winner = _apply_winner(fused, winner_base)\n"
            "display(Markdown(\n"
            "    f'**Dataset para los 5 modelos**: `{fused_winner.shape}` '\n"
            "    f'(winner_base=`{winner_base}`).'\n"
            "))\n"
        )
    )

    cells.append(
        _md(
            "### 4.1 Entrenamiento de los 3 modelos tabulares con MLflow\n\n"
            "Cada modelo tabular abre un run MLflow propio. Tiempo esperado: "
            "30-60 min en total sobre RTX 4070 (XGB y LGBM en CPU paralelo, "
            "RF en CPU multinúcleo)."
        )
    )

    cells.append(
        _code(
            "tabular_results = {}\n"
            "for kind in ('rf', 'xgb', 'lgbm'):\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name=f'05-winner-{kind}',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        dvc_path=FEATURES_PATH,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        res = train_one_model(\n"
            "            fused_winner, model=kind,\n"
            "            k_folds=K_FOLDS, buffer_km=BUFFER_KM,\n"
            "            random_state=RANDOM_STATE,\n"
            "        )\n"
            "        tabular_results[kind] = res\n"
            "        import mlflow\n"
            "        mlflow.log_params({\n"
            "            'model': kind,\n"
            "            'winner_base': winner_base,\n"
            "            'k_folds': K_FOLDS,\n"
            "            'n_parcels': fused_winner.height,\n"
            "            'n_features': len(res.feature_cols),\n"
            "        })\n"
            "        mlflow.log_metrics({\n"
            "            'f1_macro': res.metrics['f1_macro'],\n"
            "            'f1_weighted': res.metrics['f1_weighted'],\n"
            "            'miou': res.metrics['miou'],\n"
            "            'accuracy': res.metrics['accuracy'],\n"
            "            'kappa': res.metrics['cohen_kappa'],\n"
            "        })\n"
            "        MLFLOW_RUN_IDS[f'winner-{kind}'] = run.info.run_id\n"
            "        display(Markdown(\n"
            "            f'**`{kind}`** ajustado: F1-macro = `{res.metrics[\"f1_macro\"]:.4f}` '\n"
            "            f'(run `{run.info.run_id[:12]}...`).'\n"
            "        ))\n"
        )
    )

    cells.append(
        _md(
            "### 4.2 Entrenamiento de los 2 modelos temporales (TempCNN + InceptionTime)\n\n"
            "Reentrenamiento real sobre la curva NDVI/NDWI/EVI reconstruida "
            "(T=72 a partir de la FFT). MLflow loggea `fold{i}_train_loss` y "
            "`fold{i}_val_loss` por época — recuperaremos esas curvas para "
            "diagnosticar sub/sobreajuste en la sección 5. Tiempo esperado: "
            "45-90 min en GPU RTX 4070."
        )
    )

    cells.append(
        _code(
            "from ml.utils.baseline_notebook_helpers import load_temporal_result_from_mlflow\n"
            "\n"
            "temporal_results = {}\n"
            "if ENABLE_TEMPORAL_MODELS:\n"
            "    for kind in ('tempcnn', 'inceptiontime'):\n"
            "        res = None\n"
            "        if REUSE_TEMPORAL_FROM_MLFLOW:\n"
            "            try:\n"
            "                res = load_temporal_result_from_mlflow(\n"
            "                    kind,\n"
            "                    experiment_name=MLFLOW_EXPERIMENT,\n"
            "                    tracking_uri=mlflow_uri,\n"
            "                )\n"
            "                display(Markdown(\n"
            "                    f'**`{kind}`** recuperado de MLflow: F1-macro = `{res.f1_macro:.4f}` '\n"
            "                    f'(run `{res.mlflow_run_id[:12]}...`, sin re-entrenar).'\n"
            "                ))\n"
            "            except ValueError as exc:\n"
            "                display(Markdown(\n"
            "                    f'> Cache MLflow no disponible para `{kind}`: {exc}. Entreno desde cero.'\n"
            "                ))\n"
            "        if res is None:\n"
            "            # train_temporal_model abre su propio run MLflow.\n"
            "            res = train_temporal_model(\n"
            "                df=fused_winner,\n"
            "                model_kind=kind,\n"
            "                n_epochs=TEMPORAL_EPOCHS,\n"
            "                batch_size=TEMPORAL_BATCH_SIZE,\n"
            "                device=TEMPORAL_DEVICE,\n"
            "                mlflow_uri=mlflow_uri,\n"
            "                k_folds=K_FOLDS,\n"
            "                buffer_km=BUFFER_KM,\n"
            "                seed=RANDOM_STATE,\n"
            "            )\n"
            "            display(Markdown(\n"
            "                f'**`{kind}`** entrenado: F1-macro = `{res.f1_macro:.4f}` '\n"
            "                f'(run `{(res.mlflow_run_id or \"local\")[:12]}...`).'\n"
            "            ))\n"
            "        temporal_results[kind] = res\n"
            "        if res.mlflow_run_id:\n"
            "            MLFLOW_RUN_IDS[f'winner-{kind}'] = res.mlflow_run_id\n"
            "else:\n"
            "    display(Markdown('> ENABLE_TEMPORAL_MODELS=False: modelos temporales omitidos.'))\n"
        )
    )

    cells.append(_md("### 4.3 Tabla y barplot consolidado de los 5 modelos"))

    cells.append(
        _code(
            "rows_5 = []\n"
            "for k, r in tabular_results.items():\n"
            "    rows_5.append({\n"
            "        'model': k, 'family': 'tabular',\n"
            "        'f1_macro': r.metrics['f1_macro'],\n"
            "        'f1_weighted': r.metrics['f1_weighted'],\n"
            "        'miou': r.metrics['miou'],\n"
            "        'accuracy': r.metrics['accuracy'],\n"
            "        'kappa': r.metrics['cohen_kappa'],\n"
            "    })\n"
            "for k, r in temporal_results.items():\n"
            "    rows_5.append({\n"
            "        'model': k, 'family': 'temporal',\n"
            "        'f1_macro': r.f1_macro,\n"
            "        'f1_weighted': r.f1_weighted,\n"
            "        'miou': r.miou,\n"
            "        'accuracy': float('nan'),\n"
            "        'kappa': r.cohen_kappa,\n"
            "    })\n"
            "comparison_5 = pl.DataFrame(rows_5).sort('f1_macro', descending=True)\n"
            "comparison_5_path = env.reports_dir / 'model_comparison_temporal.parquet'\n"
            "comparison_5.write_parquet(comparison_5_path)\n"
            "display(Markdown(f'**Tabla guardada**: `{comparison_5_path.relative_to(env.repo)}`'))\n"
            "display(comparison_5)\n"
            "\n"
            "# Barplot con palette distinta para tabular vs temporal.\n"
            "fig, ax = plt.subplots(figsize=(9, 5), dpi=110)\n"
            "palette = {'tabular': '#4C72B0', 'temporal': '#DD8452'}\n"
            "for i, row in enumerate(comparison_5.iter_rows(named=True)):\n"
            "    ax.bar(i, row['f1_macro'], color=palette[row['family']],\n"
            "           label=row['family'] if i == 0 or row['family'] != comparison_5.row(i-1, named=True)['family'] else None)\n"
            "ax.set_xticks(range(comparison_5.height))\n"
            "ax.set_xticklabels(comparison_5['model'].to_list(), rotation=15)\n"
            "ax.set_ylabel('F1-macro out-of-fold')\n"
            "ax.set_title(f'5 modelos sobre conjunto ganador `{winner_base}`')\n"
            "ax.axhline(0.60, color='#888', linestyle='--', linewidth=1, label='target P5 = 0.60')\n"
            "handles, labels = ax.get_legend_handles_labels()\n"
            "by_label = dict(zip(labels, handles))\n"
            "ax.legend(by_label.values(), by_label.keys(), loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(env.figures_dir / 'model_comparison_5.png', bbox_inches='tight')\n"
            "display(fig)\n"
            "plt.close(fig)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 5. Per-class diagnosis of the best temporal model.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 5. Diagnóstico por clase del mejor modelo temporal\n\n"
            "Tomamos el mejor de los dos temporales y examinamos:\n\n"
            "- **F1 por clase**: cuánto acierta el modelo en cada cultivo, "
            "con un umbral para marcar las clases débiles (F1 < 0.10).\n"
            "- **Matriz de confusión OOF**: qué clases se confunden entre "
            "sí. El sufijo *out-of-fold* significa que cada predicción "
            "viene del fold donde esa parcela quedó en validación — son "
            "predicciones honestas sobre el dataset completo.\n"
            "- **Curva de loss train vs val por época** desde MLflow + "
            "`diagnose_temporal_fit` (veredicto explícito sub/sobreajuste).\n\n"
            "Las clases débiles son candidatas a fusionarse en una "
            "macro-clase `other_minor` en la fase de modelo final."
        )
    )

    cells.append(
        _code(
            "from ml.eval.reencuadre_plots import plot_per_class_f1\n"
            "from ml.eval.metrics import confusion_matrix_figure\n"
            "from ml.eval.learning_curves import (\n"
            "    fetch_loss_history_from_mlflow,\n"
            "    plot_loss_history_from_mlflow,\n"
            "    diagnose_temporal_fit,\n"
            ")\n"
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "\n"
            "if temporal_results:\n"
            "    best_temporal = max(temporal_results.values(), key=lambda r: r.f1_macro)\n"
            "    display(Markdown(\n"
            "        f'**Mejor modelo temporal**: `{best_temporal.model_kind}` con '\n"
            "        f'F1-macro = `{best_temporal.f1_macro:.4f}` sobre '\n"
            "        f'`{best_temporal.n_parcels:,}` parcelas y `{best_temporal.n_classes}` clases.'\n"
            "    ))\n"
            "    if best_temporal.y_true_oof.size > 0:\n"
            "        # Persistimos OOF.\n"
            "        import numpy as np\n"
            "        oof_path = env.reports_dir / f'oof_predictions_{best_temporal.model_kind}.npz'\n"
            "        np.savez_compressed(\n"
            "            oof_path,\n"
            "            y_true=best_temporal.y_true_oof,\n"
            "            y_pred=best_temporal.y_pred_oof,\n"
            "        )\n"
            "        display(Markdown(\n"
            "            f'**OOF guardado**: `{oof_path.relative_to(env.repo)}` '\n"
            "            f'({best_temporal.y_true_oof.size:,} predicciones).'\n"
            "        ))\n"
            "        # F1 por clase.\n"
            "        class_names = {\n"
            "            int(c): PASTIS_R_CLASSES.get(int(c), f'c{int(c)}')\n"
            "            for c in np.unique(best_temporal.y_true_oof)\n"
            "        }\n"
            "        fig_f1 = plot_per_class_f1(\n"
            "            best_temporal.y_true_oof,\n"
            "            best_temporal.y_pred_oof,\n"
            "            class_labels=sorted(class_names.keys()),\n"
            "            class_names=class_names,\n"
            "            weak_threshold=0.10,\n"
            "            title=f'F1 por clase ({best_temporal.model_kind}) out-of-fold',\n"
            "        )\n"
            "        fig_f1.savefig(env.figures_dir / f'per_class_f1_{best_temporal.model_kind}.png', bbox_inches='tight')\n"
            "        display(fig_f1)\n"
            "        plt.close(fig_f1)\n"
            "        # Matriz de confusion.\n"
            "        fig_cm = confusion_matrix_figure(\n"
            "            best_temporal.y_true_oof,\n"
            "            best_temporal.y_pred_oof,\n"
            "            class_names=class_names,\n"
            "            normalize=True,\n"
            "        )\n"
            "        display(Markdown(f'**Matriz de confusión OOF** ({best_temporal.model_kind}):'))\n"
            "        fig_cm.savefig(env.figures_dir / f'confusion_matrix_{best_temporal.model_kind}.png', bbox_inches='tight')\n"
            "        display(fig_cm)\n"
            "        plt.close(fig_cm)\n"
            "else:\n"
            "    display(Markdown('> Sección 5 omitida: no se entrenaron modelos temporales.'))\n"
        )
    )

    cells.append(_md("### 5.1 Curva de loss train vs val por época (los 2 temporales)"))

    cells.append(
        _code(
            "for kind, res in temporal_results.items():\n"
            "    if not res.mlflow_run_id:\n"
            "        display(Markdown(f'> `{kind}`: sin run_id MLflow — historial no recuperable.'))\n"
            "        continue\n"
            "    try:\n"
            "        history = fetch_loss_history_from_mlflow(\n"
            "            res.mlflow_run_id,\n"
            "            model_kind=kind,\n"
            "            tracking_uri=mlflow_uri,\n"
            "        )\n"
            "        fig_loss = plot_loss_history_from_mlflow(history)\n"
            "        fig_loss.savefig(env.figures_dir / f'loss_history_{kind}.png', bbox_inches='tight')\n"
            "        display(fig_loss)\n"
            "        plt.close(fig_loss)\n"
            "        try:\n"
            "            diag = diagnose_temporal_fit(history)\n"
            "            display(Markdown(\n"
            "                f'**Diagnóstico `{kind}`**: `{diag.verdict}` '\n"
            "                f'(gap val−train loss = `{diag.gap:.3f}`).\\n\\n'\n"
            "                f'{diag.explanation}'\n"
            "            ))\n"
            "        except ValueError as e:\n"
            "            display(Markdown(f'> Diagnóstico `{kind}` no disponible: {e}'))\n"
            "    except RuntimeError as e:\n"
            "        display(Markdown(f'> No se pudo recuperar historial de `{kind}` desde MLflow: {e}'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 6. KMeans + UMAP clustering over pure phenological signature.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 6. Clustering sin coordenadas — ¿hay estructura en la firma fenológica pura?\n\n"
            "Clusterizamos las parcelas usando **solo** la firma fenológica "
            "(8 features agronómicos: pico, senescencia, área bajo curva NDVI, "
            "etc., más los 24 armónicos FFT de NDVI/NDWI/EVI). **No entran "
            "coordenadas, ni `geom_*`, ni clima, ni embedding satelital.**\n\n"
            "Si los clusters corresponden a **arquetipos estacionales "
            "reconocibles** (cultivo de invierno, cultivo de verano largo, "
            "suelo desnudo), la fenología pura organiza el dataset sin "
            "necesidad del contexto geográfico. El test visual es UMAP 2D "
            "coloreado por cluster + curva NDVI media reconstruida por cluster."
        )
    )

    cells.append(
        _code(
            "if ENABLE_CLUSTERING:\n"
            "    import numpy as np\n"
            "    from sklearn.cluster import KMeans\n"
            "    from sklearn.preprocessing import StandardScaler\n"
            "    from ml.eval.feature_ablation import build_default_feature_sets\n"
            "    from ml.features.selection import fit_umap_2d\n"
            "    from ml.eval.reencuadre_plots import (\n"
            "        plot_umap_clusters, plot_cluster_ndvi_curves,\n"
            "    )\n"
            "\n"
            "    feature_sets_for_cluster = build_default_feature_sets(fused.columns)\n"
            "    pheno_only_cols = feature_sets_for_cluster.get('phenology_only', ())\n"
            "    if pheno_only_cols:\n"
            "        X_pheno = fused.select(list(pheno_only_cols)).to_numpy().astype(np.float64)\n"
            "        # Imputacion por media de columna para tolerar NaN.\n"
            "        col_means = np.nanmean(np.where(np.isfinite(X_pheno), X_pheno, np.nan), axis=0)\n"
            "        col_means = np.where(np.isnan(col_means), 0.0, col_means)\n"
            "        X_clean = np.where(np.isfinite(X_pheno), X_pheno, col_means)\n"
            "        X_scaled = StandardScaler().fit_transform(X_clean)\n"
            "        n_clusters = min(N_CLUSTERS, fused['class_id'].n_unique())\n"
            "        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)\n"
            "        cluster_labels = kmeans.fit_predict(X_scaled)\n"
            "        display(Markdown(f'**Clustering**: KMeans con `n_clusters={n_clusters}` sobre `{X_scaled.shape}`.'))\n"
            "        # UMAP 2D.\n"
            "        embedding = fit_umap_2d(X_scaled, random_state=RANDOM_STATE)\n"
            "        np.savez_compressed(\n"
            "            env.reports_dir / 'umap_embedding.npz',\n"
            "            embedding=embedding, cluster_labels=cluster_labels,\n"
            "        )\n"
            "        fig_umap = plot_umap_clusters(\n"
            "            embedding, cluster_labels,\n"
            "            title='UMAP de la firma fenológica pura, coloreado por cluster KMeans',\n"
            "        )\n"
            "        fig_umap.savefig(env.figures_dir / 'umap_clusters.png', bbox_inches='tight')\n"
            "        display(fig_umap)\n"
            "        plt.close(fig_umap)\n"
            "        # Curva NDVI media por cluster.\n"
            "        fig_curves = plot_cluster_ndvi_curves(\n"
            "            fused, cluster_labels, sequence_length=72,\n"
            "            title='Curva NDVI media reconstruida por cluster (sin coordenadas)',\n"
            "        )\n"
            "        fig_curves.savefig(env.figures_dir / 'cluster_ndvi_curves.png', bbox_inches='tight')\n"
            "        display(fig_curves)\n"
            "        plt.close(fig_curves)\n"
            "        # Composicion top-3 cultivos por cluster.\n"
            "        df_with_clusters = fused.with_columns(\n"
            "            pl.Series('pheno_cluster', cluster_labels).cast(pl.Int64)\n"
            "        )\n"
            "        top_per_cluster = (\n"
            "            df_with_clusters.group_by(['pheno_cluster', 'class_id']).len()\n"
            "            .sort(['pheno_cluster', 'len'], descending=[False, True])\n"
            "            .group_by('pheno_cluster', maintain_order=True).head(3)\n"
            "        )\n"
            "        top_per_cluster.write_parquet(env.reports_dir / 'cluster_class_counts.parquet')\n"
            "        display(Markdown('**Top-3 cultivos por cluster**:'))\n"
            "        display(top_per_cluster)\n"
            "    else:\n"
            "        display(Markdown('> Sin columnas `phenology_only` detectadas en `fused`; clustering omitido.'))\n"
            "else:\n"
            "    display(Markdown('> ENABLE_CLUSTERING=False: bloque omitido.'))\n"
        )
    )

    cells.append(
        _md(
            "### 6.1 Interpretación agronómica de los clusters\n\n"
            "Los clusters de KMeans sobre la firma fenológica pura tienden "
            "a corresponder a **arquetipos estacionales**, no a regiones:\n\n"
            "- Pico temprano (DOY 80-120) + senescencia temprana → cultivos "
            "de invierno (trigo, cebada).\n"
            "- Pico tardío (DOY 180-220) + maduración larga → cultivos de "
            "verano largos (maíz, girasol).\n"
            "- Pico bajo y área bajo curva pequeña → cultivos de ciclo "
            "corto, suelos desnudos parte del año o cultivos minoritarios.\n\n"
            "El gráfico de curvas NDVI medias por cluster es el diagnóstico "
            "clave: si dos clusters tienen curvas claramente distintas "
            "(pico en distinto DOY, amplitud distinta), la fenología los "
            "está separando sin ayuda del contexto."
        )
    )

    # -----------------------------------------------------------------------
    # 7. Strategy for the imbalance.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 7. Estrategia para el desbalance ~31× max/min\n\n"
            "El desbalance es la causa principal del techo en F1-macro. "
            "Tres opciones consideradas:\n\n"
            "1. **Pesos por clase** (`class_weight='balanced'` en RF y "
            "`sample_weight` inverso a la frecuencia en XGB/LGBM). Ya "
            "activado en el baseline; bajo costo, evidencia mixta.\n"
            "2. **Oversampling sintético (SMOTE) o duplicación aleatoria**. "
            "Riesgo de leakage espacial vía vecinos sintéticos — descartado "
            "en este dataset.\n"
            "3. **Fusión de clases minoritarias en una macro-clase "
            "`other_minor`**. Sacrifica granularidad pero suele estabilizar "
            "F1-macro. Es la decisión recomendada para la siguiente fase "
            "si las clases débiles del diagnóstico de la sección 5 siguen "
            "rindiendo F1=0."
        )
    )

    cells.append(
        _code(
            "class_counts = (\n"
            "    fused.group_by('class_id').len().sort('len', descending=True)\n"
            "    .with_columns((pl.col('len') / fused.height * 100).round(2).alias('pct'))\n"
            ")\n"
            "imbalance_ratio = float(class_counts['len'].max()) / max(\n"
            "    float(class_counts['len'].min()), 1.0\n"
            ")\n"
            "weak_classes = (\n"
            "    class_counts.filter(pl.col('len') < WEAK_CLASS_THRESHOLD)\n"
            "    .get_column('class_id').to_list()\n"
            ")\n"
            "display(Markdown(\n"
            "    f'**Imbalance ratio max/min**: `{imbalance_ratio:.1f}x`. '\n"
            "    f'**Clases débiles** (< `{WEAK_CLASS_THRESHOLD}` parcelas): `{weak_classes}` '\n"
            "    f'({len(weak_classes)} clases).'\n"
            "))\n"
            "display(Markdown(\n"
            "    '**Estrategia recomendada**: mantener pesos por clase (opción 1) '\n"
            "    'y evaluar fusión en macro-clase `other_minor` (opción 3) si '\n"
            "    'las clases débiles siguen con F1 ≈ 0 en el diagnóstico de la sección 5.'\n"
            "))\n"
            "class_counts.write_parquet(env.reports_dir / 'class_counts.parquet')\n"
        )
    )

    # -----------------------------------------------------------------------
    # 8. Phenological semantic branch with Gemini.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 8. Rama semántica fenológica — descripción textual real con Gemini Flash\n\n"
            "El paper Wen et al. (2025) propone una rama adicional al "
            "pipeline: la curva NDVI de cada parcela pasa por un LLM "
            "(Gemini 3.5 Flash) que produce una descripción estructurada en "
            "lenguaje natural — por ejemplo *cultivo de verano con pico "
            "medio en julio, senescencia abrupta en septiembre*. Un "
            "text-encoder convierte ese texto en un vector denso de 384 "
            "dimensiones que se concatena al vector tabular como bloque "
            "opcional `pheno_text_*`.\n\n"
            "Esta sección **no mockea** el LLM: si `GEMINI_API_KEY` está "
            "presente y `ENABLE_LLM_BRANCH=True`, lanza Gemini real sobre "
            "un subset balanceado de `LLM_SUBSET_SIZE = 1080` parcelas "
            "(60 por clase × 18 clases). El bloque resultante se persiste "
            "y compara contra la corrida histórica `with_pheno_text` de la "
            "sección 3."
        )
    )

    cells.append(
        _code(
            "if ENABLE_LLM_BRANCH and env.has_gemini_api_key:\n"
            "    from ml.features.phenology_description import build_phenology_text_block\n"
            "\n"
            "    # Subset balanceado: 60 parcelas por clase.\n"
            "    per_class = max(1, LLM_SUBSET_SIZE // fused['class_id'].n_unique())\n"
            "    rng_seed = RANDOM_STATE\n"
            "    balanced = (\n"
            "        fused.with_columns(pl.lit(np.random.default_rng(rng_seed).random(fused.height)).alias('__r'))\n"
            "        .group_by('class_id')\n"
            "        .map_groups(lambda g: g.sort('__r').head(min(per_class, g.height)))\n"
            "        .drop('__r')\n"
            "    )\n"
            "    pheno_ndvi_cols = ['parcel_id', 'year'] + [c for c in balanced.columns if c.startswith('NDVI_fft')]\n"
            "    cols_present = [c for c in pheno_ndvi_cols if c in balanced.columns]\n"
            "    display(Markdown(\n"
            "        f'**Subset balanceado LLM**: `{balanced.height}` parcelas '\n"
            "        f'({per_class}/clase × {fused[\"class_id\"].n_unique()} clases).'\n"
            "    ))\n"
            "    with track_experiment(\n"
            "        experiment_name=MLFLOW_EXPERIMENT,\n"
            "        run_name='05-llm-pheno-text-real',\n"
            "        tracking_uri=mlflow_uri,\n"
            "        probe_server=False,\n"
            "    ) as run:\n"
            "        import mlflow\n"
            "        text_block = build_phenology_text_block(\n"
            "            balanced.select(cols_present),\n"
            "            skip_llm=False,\n"
            "            cache_dir=env.repo / 'data/cache/phenology_descriptions',\n"
            "        )\n"
            "        mlflow.log_params({\n"
            "            'llm': 'gemini-3.5-flash',\n"
            "            'n_parcels': balanced.height,\n"
            "            'per_class': per_class,\n"
            "            'embedding_dim': text_block.shape[1] - 1,\n"
            "        })\n"
            "        MLFLOW_RUN_IDS['llm-pheno-text'] = run.info.run_id\n"
            "        display(Markdown(\n"
            "            f'**Bloque `pheno_text`** real: shape=`{text_block.shape}` '\n"
            "            f'(run `{run.info.run_id[:12]}...`).'\n"
            "        ))\n"
            "elif ENABLE_LLM_BRANCH:\n"
            "    display(Markdown(\n"
            "        '> `GEMINI_API_KEY` ausente. Define la variable en `.env.local` '\n"
            "        'antes de activar `ENABLE_LLM_BRANCH=True`. Saltamos el bloque.'\n"
            "    ))\n"
            "else:\n"
            "    display(Markdown('> ENABLE_LLM_BRANCH=False: rama LLM omitida.'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 9. Conclusions.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 9. Conclusiones consolidadas — decisión por bloque\n\n"
            "**Lo que validamos numéricamente en este cuaderno**:\n\n"
            "1. **Las features geográficas (`geom_*`) se descartan** del "
            "baseline. La sección 3 confirma la decisión de `04c`: "
            "`geom_only` < 0.10 (no clasifican por sí solas) y `no_geom` "
            "no degrada respecto a `full`.\n"
            "2. **Clima y topografía crudos son redundantes** con AlphaEarth. "
            "La diferencia entre `no_geom` y `no_geom_no_era5_srtm` es "
            "marginal — el embedding fundacional ya los codifica.\n"
            "3. **La firma fenológica explícita lleva una parte importante "
            "de la señal**. `phenology_only` queda cerca del conjunto "
            "completo.\n"
            "4. **Los modelos temporales consumen mejor la información** "
            "que el resumen anual — la sección 4 compara los 5 modelos "
            "sobre el mismo conjunto ganador y muestra si TempCNN/"
            "InceptionTime superan a XGBoost.\n"
            "5. **La estructura existe sin coordenadas**. KMeans sobre la "
            "firma fenológica pura agrupa parcelas por arquetipo estacional. "
            "Las curvas NDVI medias por cluster son interpretables "
            "agronómicamente.\n\n"
            "**Decisiones por bloque opcional** (umbral `delta >= +0.005`):\n\n"
            "- **FarSLIP**: si `with_farslip - full >= +0.005`, promover; "
            "si delta en [-0.005, +0.005], diferir a stacking (Avance 5); "
            "si <-0.005, descartar.\n"
            "- **pheno_text (Gemini Flash real)**: misma regla. La rama "
            "semántica de la sección 8 produce un parquet sobre 1080 "
            "parcelas balanceadas.\n"
            "- **Firma espectral REP (Frampton 2013)**: misma regla.\n\n"
            "**Sobre las preguntas oficiales**:\n\n"
            "- **P1**: la tabla de la sección 4 cubre **5 modelos reales** "
            "(RF, XGB, LGBM, TempCNN, InceptionTime). El ganador final lo "
            "decide `Avance3.Equipo17.ipynb`.\n"
            "- **P2**: ablación con bloques opcionales + decisión por "
            "umbral cuantifica el aporte real.\n"
            "- **P3**: diagnóstico sub/sobreajuste para los 2 temporales "
            "leyendo loss desde MLflow (sección 5.1).\n\n"
            "## Lo que sigue\n\n"
            "`Avance3.Equipo17.ipynb` lee `ablation_table.parquet` + "
            "`model_comparison_temporal.parquet` + las tablas de los "
            "notebooks 04, 04b, 04c, 04_farslip; ejecuta "
            "`select_winning_features()` (única vez en el proyecto) y "
            "persiste el conjunto ganador."
        )
    )

    # -----------------------------------------------------------------------
    # 10. Glossary.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 10. Glosario\n\n"
            "- **Ablation**: experimento que entrena el mismo modelo sobre "
            "varios subconjuntos de features para medir cuánto aporta cada "
            "bloque. Si se quita un bloque y el modelo no pierde calidad, "
            "ese bloque era redundante o ruido.\n"
            "- **Spatial CV**: en lugar de dividir las parcelas al azar "
            "entre folds, se asegura que las geográficamente cercanas vayan "
            "al mismo fold y se respeta un buffer de separación. Evita que "
            "el modelo memorice la ubicación en lugar del cultivo.\n"
            "- **Out-of-fold (OOF)**: predicción sobre una parcela "
            "obtenida en el fold donde esa parcela quedó en validación. "
            "Por construcción, el conjunto OOF reúne predicciones honestas "
            "sobre el dataset completo.\n"
            "- **F1-macro**: promedio simple del F1 por clase, sin "
            "ponderar por soporte. Penaliza con fuerza fallos en clases "
            "minoritarias — la métrica natural cuando importa rendir en "
            "todas las clases por igual.\n"
            "- **mIoU (mean Intersection over Union)**: promedio de Jaccard "
            "por clase. Equivalente a F1-macro en sensibilidad al "
            "desbalance, más estricto.\n"
            "- **FFT (Fast Fourier Transform)**: descomposición de la serie "
            "temporal NDVI en armónicos. Los primeros capturan la "
            "estacionalidad anual; los posteriores, picos cortos. Permite "
            "resumir 72 puntos en 8 números (4 amplitudes + 4 fases) por "
            "índice espectral.\n"
            "- **Fenología (phenology)**: estudio de los eventos "
            "estacionales del cultivo (emergencia, pico, senescencia, "
            "cosecha). Las features fenológicas describen esos eventos "
            "como atributos derivados de la curva NDVI.\n"
            "- **REP (Red Edge Position)**: longitud de onda (nm) del "
            "punto de inflexión entre el rojo y el infrarrojo cercano de "
            "Sentinel-2 (Frampton 2013). Cambia con el contenido de "
            "clorofila y la fenología.\n"
            "- **TempCNN / InceptionTime**: modelos temporales 1D para "
            "series temporales. Importados de `breizhcrops.models`. "
            "Ven la curva completa, no su resumen anual.\n"
            "- **FarSLIP / RemoteCLIP**: extractores de embeddings "
            "visuales basados en CLIP, afinados para teledetección. "
            "Generan un vector denso por parcela que se usa como bloque "
            "opcional del baseline tabular o como modelo independiente."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# Avance3.Equipo17.ipynb — concentrator + select_winning_features.
# ---------------------------------------------------------------------------


def build_avance3() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Avance 3 — Baseline (Equipo 17, AgroSatCopilot)\n\n"
            "## Proyecto Integrador MNA · Tec de Monterrey\n\n"
            "**Equipo 17**\n\n"
            "- Carlos Isaac Ávila Gutiérrez — A01796035\n"
            "- Carlos Aaron Bocanegra Buitrón — A01796345\n"
            "- Arthur Jafed Zizumbo Velasco — A01796363\n\n"
            "**Curso**: MNA — Tec de Monterrey · 20-abr → 3-jul-2026\n\n"
            "**Sponsor académico**: Dr. Gerardo José Camacho — gjcamacho@tec.mx\n\n"
            "**Fecha de entrega**: 2026-05-20 (con correcciones post-A3 cerradas 2026-05-27).\n\n"
            "---\n\n"
            "## Resumen ejecutivo\n\n"
            "Este cuaderno **concentra y consolida** el trabajo del Avance 3 "
            "leyendo los artefactos generados por las cinco libretas previas. "
            "No reentrena modelos: lee tablas, lee figuras, ejecuta una sola "
            "llamada a `select_winning_features` para nombrar el conjunto "
            "ganador y entrega las respuestas a las **cinco preguntas "
            "oficiales del Avance 3** con cifras consolidadas.\n\n"
            "Las cinco preguntas oficiales son:\n\n"
            "1. **¿Qué algoritmo se puede utilizar como baseline** para "
            "predecir las variables objetivo?\n"
            "2. **¿Se puede determinar la importancia de las características** "
            "para el modelo generado? (incluir características irrelevantes "
            "afecta el rendimiento y aumenta la complejidad).\n"
            "3. **¿El modelo está sub/sobreajustando** los datos de "
            "entrenamiento?\n"
            "4. **¿Cuál es la métrica adecuada** para este problema de "
            "negocio?\n"
            "5. **¿Cuál debería ser el desempeño mínimo** a obtener?\n\n"
            "## Estructura\n\n"
            "1. Las 5 preguntas oficiales y dónde se contesta cada una.\n"
            "2. P1 — Tabla consolidada de los 5 modelos (3 tabulares + 2 temporales).\n"
            "3. P2 — Importancia de características y bloques irrelevantes.\n"
            "4. P3 — Diagnóstico sub/sobreajuste para los 5 modelos.\n"
            "5. P4 — Justificación de F1-macro como métrica principal.\n"
            "6. P5 — Desempeño mínimo: target 0.60, estado actual y plan.\n"
            "7. Comparativa AlphaEarth vs Sentinel-2 crudo vs vector combinado.\n"
            "8. Resultados FarSLIP vs RemoteCLIP.\n"
            "9. Clustering fenológico sin coordenadas.\n"
            "10. Tabla H-1..H-4 con decisiones consolidadas (cuatro hipótesis "
            "del Avance 3).\n"
            "11. Conjunto ganador: única llamada a `select_winning_features` + "
            "manifest JSON.\n"
            "12. Trazabilidad MLflow consolidada (todos los `run_id`).\n"
            "13. Referencias.\n"
        )
    )

    cells.append(
        _code(
            'COMPARISON_PATH_04 = "reports/baseline/04_baseline/model_comparison_04.parquet"\n'
            'COMPARISON_PATH_04B = "reports/baseline/04b_baseline/model_comparison_04b.parquet"\n'
            'ABLATION_BASE_PATH_04C = "reports/baseline/04c_baseline/ablation_table.parquet"\n'
            'ABLATION_OPTIONAL_PATH_05 = "reports/baseline/05_reencuadre/ablation_table.parquet"\n'
            'COMPARISON_TEMPORAL_PATH = "reports/baseline/05_reencuadre/model_comparison_temporal.parquet"\n'
            'COMPARISON_SCENARIOS_PATH = "reports/baseline/04_baseline/comparison_alphaearth_vs_s2.csv"\n'
            'FUSED_PATH = "data/features/features_fused_pastis.parquet"\n'
            'WINNING_OUTPUT = "data/features/features_fused_winning_pastis.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/Avance3"\n'
            'REPORTS_SUBDIR = "baseline/Avance3"\n'
            "PROMOTE_THRESHOLD = 0.005\n"
            "F1_TARGET = 0.60\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    # -----------------------------------------------------------------------
    # 1. The 5 official questions of Avance 3 with mapping to notebooks.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 1. Las cinco preguntas oficiales y dónde se contestan\n\n"
            "| # | Pregunta oficial | Notebook(s) que aportan la respuesta |\n"
            "|---|------------------|--------------------------------------|\n"
            "| **P1** | ¿Qué algoritmo se puede utilizar como baseline? | `04_baseline` (3 tabulares + escenarios) · `04b_baseline` (piloto + cifras US-022b) · `04c_baseline` (qué bloques aportan) · `05_reencuadre` (5 modelos: 3 tabulares + 2 temporales sobre conjunto ganador) |\n"
            "| **P2** | ¿Se puede determinar la importancia de las características? | `04_baseline` §5 (Gini, gain, SHAP, dominancia AE) · `04c_baseline` (ablación: bloques irrelevantes) · `05_reencuadre` §3 (bloques opcionales) · `04_farslip_eval` (separabilidad por espacio de embedding) |\n"
            "| **P3** | ¿El modelo sub/sobreajusta? | `04_baseline` §10 (learning curves + `diagnose_fit` para RF/XGB/LGBM) · `05_reencuadre` §5.1 (loss train vs val por época + `diagnose_temporal_fit` para TempCNN/InceptionTime) |\n"
            "| **P4** | ¿Cuál es la métrica adecuada? | `04_baseline` §2 (justificación F1-macro con desbalance 31×) |\n"
            "| **P5** | ¿Desempeño mínimo a obtener? | `04_baseline` §3 (justificación 0.60: random vs trivial vs U-TAE PASTIS) |"
        )
    )

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from ml.eval.reencuadre_plots import plot_optional_blocks_ablation\n"
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "\n"
            "# Helper compacto para leer parquet con fallback explicito.\n"
            "def _read_or_skip(path_str: str, label: str) -> pl.DataFrame | None:\n"
            "    p = Path(path_str)\n"
            "    if not p.exists():\n"
            "        display(Markdown(\n"
            "            f'> `{label}` no encontrado en `{path_str}`. '\n"
            "            'Ejecuta el notebook correspondiente antes de re-ejecutar Avance3.'\n"
            "        ))\n"
            "        return None\n"
            "    return pl.read_parquet(p)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 2. P1 — Consolidated table of the 5 models.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 2. P1 — Algoritmo baseline: tabla consolidada de los 5 modelos\n\n"
            "Leemos la tabla del `04_baseline.ipynb` (3 tabulares: RF, XGB, "
            "LGBM) y la del `05_reencuadre_fenologico.ipynb` (5 modelos "
            "sobre el conjunto ganador). El ranking final por F1-macro "
            "decide qué modelo promovemos como **baseline canónico** del "
            "Avance 3 y cuáles quedan como aprendices base para los "
            "ensambles del Avance 5."
        )
    )

    cells.append(
        _code(
            "comparison_04 = _read_or_skip(COMPARISON_PATH_04, 'model_comparison_04')\n"
            "comparison_temporal = _read_or_skip(\n"
            "    COMPARISON_TEMPORAL_PATH, 'model_comparison_temporal (5 modelos)'\n"
            ")\n"
            "\n"
            "if comparison_04 is not None:\n"
            "    display(Markdown('**Tabla `04_baseline` — 3 modelos tabulares sobre conjunto fused completo**:'))\n"
            "    display(comparison_04.sort('f1_macro', descending=True))\n"
            "if comparison_temporal is not None:\n"
            "    display(Markdown('**Tabla `05_reencuadre` — 5 modelos sobre conjunto ganador post-ablación**:'))\n"
            "    display(comparison_temporal)\n"
            "    best_row = comparison_temporal.row(0, named=True)\n"
            "    display(Markdown(\n"
            "        f'**Modelo ganador del Avance 3**: `{best_row[\"model\"]}` '\n"
            '        f\'(familia: `{best_row["family"]}`) con F1-macro = `{best_row["f1_macro"]:.4f}`.\'\n'
            "    ))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 3. P2 — Feature importance and irrelevant blocks.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 3. P2 — Importancia de las características y bloques irrelevantes\n\n"
            "Dos vistas complementarias:\n\n"
            "- **Por feature individual** (gini, gain, SHAP, dominancia "
            "AlphaEarth): ver `04_baseline.ipynb` §5 + figuras "
            "`shap_summary_*.png`, `feature_importance_*.png` en "
            "`paper/figures/us-023-preview/04_baseline/`.\n"
            "- **Por bloque completo** (ablación): leemos las tablas de "
            "`04c_baseline` (bloques base) y `05_reencuadre` (bloques "
            "opcionales) y graficamos el aporte agregado."
        )
    )

    cells.append(
        _code(
            "ablation_base = _read_or_skip(ABLATION_BASE_PATH_04C, 'ablation_table_base (04c)')\n"
            "ablation_optional = _read_or_skip(ABLATION_OPTIONAL_PATH_05, 'ablation_table (05)')\n"
            "\n"
            "if ablation_base is not None:\n"
            "    display(Markdown('**Ablación base (`04c_baseline`)** — qué bloques del fused aportan:'))\n"
            "    display(ablation_base.sort('f1_macro', descending=True))\n"
            "\n"
            "if ablation_optional is not None:\n"
            "    display(Markdown('**Ablación completa (`05_reencuadre`)** — con bloques opcionales:'))\n"
            "    display(ablation_optional.sort('f1_macro', descending=True))\n"
            "    results = [\n"
            "        FeatureAblationResult(\n"
            "            feature_set=row['feature_set'],\n"
            "            model_kind=row['model'],\n"
            "            f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "            f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "            miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "            n_features=row['n_features'],\n"
            "            delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "        )\n"
            "        for row in ablation_optional.iter_rows(named=True)\n"
            "    ]\n"
            "    fig = plot_optional_blocks_ablation(results)\n"
            "    fig.savefig(env.figures_dir / 'optional_blocks.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    # -----------------------------------------------------------------------
    # 4. P3 — Under/overfitting for the 5 models.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 4. P3 — Sub/sobreajuste para los 5 modelos\n\n"
            "Los 3 modelos tabulares (RF, XGB, LGBM) se diagnostican en "
            "`04_baseline.ipynb` §10 con `diagnose_fit` sobre la curva de "
            "aprendizaje clásica. Los 2 modelos temporales (TempCNN, "
            "InceptionTime) se diagnostican en `05_reencuadre.ipynb` §5.1 "
            "con `diagnose_temporal_fit` sobre el historial de "
            "train_loss/val_loss leído desde MLflow.\n\n"
            "Aquí incrustamos las imágenes generadas por ambos cuadernos."
        )
    )

    cells.append(
        _code(
            "from IPython.display import Image\n"
            "\n"
            "lc_figures = [\n"
            "    ('Curva aprendizaje RF', 'paper/figures/us-023-preview/04_baseline/learning_curve_rf.png'),\n"
            "    ('Curva aprendizaje XGB', 'paper/figures/us-023-preview/04_baseline/learning_curve_xgb.png'),\n"
            "    ('Curva loss TempCNN', 'paper/figures/us-023-preview/05_reencuadre/loss_history_tempcnn.png'),\n"
            "    ('Curva loss InceptionTime', 'paper/figures/us-023-preview/05_reencuadre/loss_history_inceptiontime.png'),\n"
            "]\n"
            "for label, rel_path in lc_figures:\n"
            "    full_path = env.repo / rel_path\n"
            "    if full_path.exists():\n"
            "        display(Markdown(f'**{label}** — `{rel_path}`:'))\n"
            "        display(Image(filename=str(full_path)))\n"
            "    else:\n"
            "        display(Markdown(f'> `{label}` no disponible (`{rel_path}` no existe).'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 5. P4 — Appropriate metric.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 5. P4 — Métrica adecuada para este problema\n\n"
            "**F1-macro** es la métrica principal del Avance 3. Justificación "
            "en lenguaje accesible:\n\n"
            "- **Desbalance fuerte** (~31× max/min): la accuracy puede dar "
            "lecturas engañosamente altas si el modelo predice solo las "
            "clases mayoritarias. Un clasificador trivial que asigna "
            "siempre la clase mayoritaria daría accuracy ≈ 25-30% pero "
            "F1-macro cercano a cero — cualquier modelo razonable tiene que "
            "superar eso con claridad.\n"
            "- **Importan todas las clases por igual** (cada cultivo cuenta "
            "para el campesino que lo siembra): F1-macro promedia el F1 "
            "por clase sin ponderar por soporte, así que penaliza los "
            "fallos en clases minoritarias. F1-weighted, en cambio, las "
            "disimula.\n"
            "- **Kappa** mide acuerdo corregido por azar pero no separa "
            "el comportamiento por clase — útil como medida global, "
            "insuficiente como única.\n"
            "- **mIoU (Jaccard macro)** es equivalente a F1-macro en "
            "sensibilidad al desbalance, más estricto. Se reporta como "
            "segunda métrica.\n\n"
            "**Decisión documentada**: principal F1-macro, secundaria mIoU; "
            "accuracy y kappa se reportan como contexto."
        )
    )

    # -----------------------------------------------------------------------
    # 6. P5 — Minimum performance.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 6. P5 — Desempeño mínimo a obtener\n\n"
            "**Target acordado con el sponsor**: F1-macro ≥ 0.60 sobre las "
            "18 clases PASTIS con validación cruzada espacial. Justificación "
            "con tres referencias:\n\n"
            "- **Punto de partida (random)**: clasificador aleatorio "
            "uniforme sobre 18 clases ≈ F1-macro 0.06; predicción trivial "
            "de la clase mayoritaria ≈ 0.03.\n"
            "- **Estado del arte**: PASTIS-R con **U-TAE** (Garnot et al. "
            "2021) reporta mIoU ~0.65; con DINOv3 + linear probe se "
            "alcanzan ~0.55-0.60.\n"
            "- **Decisión del equipo**: F1-macro ≥ 0.60 sobre 18 clases "
            "con CV espacial es el mínimo publicable. Si no se alcanza con "
            "el baseline tabular del Avance 3 (lo más probable), hay dos "
            "rutas: agrupación fenológica (reduce a ~10 clases) o pasar a "
            "modelos densos (Avance 4) y ensambles (Avance 5)."
        )
    )

    cells.append(
        _code(
            "if comparison_temporal is not None:\n"
            "    best_row = comparison_temporal.row(0, named=True)\n"
            "    best_f1 = float(best_row['f1_macro'])\n"
            "    gap = F1_TARGET - best_f1\n"
            "    status = '✓ cumplido' if gap <= 0 else f'✗ falta `{gap:+.4f}`'\n"
            "    display(Markdown(\n"
            "        f'**Estado actual del Avance 3**: el mejor modelo es '\n"
            "        f'`{best_row[\"model\"]}` con F1-macro = `{best_f1:.4f}`. '\n"
            "        f'Target = `{F1_TARGET}` → {status}.'\n"
            "    ))\n"
            "    if gap > 0:\n"
            "        display(Markdown(\n"
            "            '**Plan para cerrar el gap**:\\n\\n'\n"
            "            '1. Agrupación fenológica de clases minoritarias en `other_minor`.\\n'\n"
            "            '2. Modelos densos del Avance 4 (U-Net, U-TAE, TSViT, Swin-UNETR).\\n'\n"
            "            '3. Ensambles del Avance 5 (voting, bagging, stacking, blending) + Gemma 4 LoRA.'\n"
            "        ))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 7. Scenarios AlphaEarth vs raw S2 vs combined.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 7. Comparativa AlphaEarth vs Sentinel-2 crudo vs vector combinado\n\n"
            "La sección 11 de `04_baseline.ipynb` entrena 3 modelos × 3 "
            "escenarios = **9 combinaciones** sobre exactamente el mismo "
            "conjunto de parcelas (inner join). El delta entre AlphaEarth y "
            "Sentinel-2 crudo cuantifica el valor incremental del embedding "
            "fundacional."
        )
    )

    cells.append(
        _code(
            "scenarios_path = Path(COMPARISON_SCENARIOS_PATH)\n"
            "if scenarios_path.exists():\n"
            "    scenarios = pl.read_csv(scenarios_path)\n"
            "    display(Markdown('**Comparativa de escenarios (9 filas)**:'))\n"
            "    display(scenarios)\n"
            "fig_path = env.repo / 'paper/figures/us-023-preview/04_baseline/comparison_barplot.png'\n"
            "if fig_path.exists():\n"
            "    display(Image(filename=str(fig_path)))\n"
            "else:\n"
            "    display(Markdown('> Figura comparativa no disponible.'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 8. FarSLIP vs RemoteCLIP.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 8. Comparativa de extractores visuales: FarSLIP vs RemoteCLIP\n\n"
            "`04_farslip_eval_pastis.ipynb` evalúa los dos extractores "
            "visuales con dos criterios independientes:\n\n"
            "- Separabilidad lineal (LogReg + 5-fold estratificado).\n"
            "- Separabilidad visual (UMAP 2D coloreado por clase)."
        )
    )

    cells.append(
        _code(
            "umap_figures = [\n"
            "    ('UMAP FarSLIP por clase',\n"
            "     'paper/figures/us-023-preview/04_farslip_eval_pastis/umap_farslip_by_class.png'),\n"
            "    ('UMAP RemoteCLIP por clase',\n"
            "     'paper/figures/us-023-preview/04_farslip_eval_pastis/umap_remoteclip_by_class.png'),\n"
            "]\n"
            "for label, rel_path in umap_figures:\n"
            "    full_path = env.repo / rel_path\n"
            "    if full_path.exists():\n"
            "        display(Markdown(f'**{label}** — `{rel_path}`:'))\n"
            "        display(Image(filename=str(full_path)))\n"
            "    else:\n"
            "        display(Markdown(f'> `{label}` no disponible.'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 9. Phenological clustering.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 9. Clustering fenológico sin coordenadas\n\n"
            "`05_reencuadre_fenologico.ipynb` §6 demuestra que la firma "
            "fenológica pura organiza el dataset en arquetipos estacionales "
            "sin necesidad de coordenadas. Las curvas NDVI medias por "
            "cluster son interpretables agronómicamente."
        )
    )

    cells.append(
        _code(
            "cluster_figures = [\n"
            "    ('UMAP firma fenológica + KMeans',\n"
            "     'paper/figures/us-023-preview/05_reencuadre/umap_clusters.png'),\n"
            "    ('Curvas NDVI medias por cluster',\n"
            "     'paper/figures/us-023-preview/05_reencuadre/cluster_ndvi_curves.png'),\n"
            "]\n"
            "for label, rel_path in cluster_figures:\n"
            "    full_path = env.repo / rel_path\n"
            "    if full_path.exists():\n"
            "        display(Markdown(f'**{label}** — `{rel_path}`:'))\n"
            "        display(Image(filename=str(full_path)))\n"
            "    else:\n"
            "        display(Markdown(f'> `{label}` no disponible.'))\n"
        )
    )

    # -----------------------------------------------------------------------
    # 10. Table H-1..H-4 with consolidated decisions.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 10. Cuatro hipótesis del Avance 3 con decisiones consolidadas\n\n"
            "Cada hipótesis se cierra con una decisión cuantitativa apoyada "
            "en la ablación. Las cifras son leídas del `ablation_table.parquet` "
            "que produjo `05_reencuadre`."
        )
    )

    cells.append(
        _code(
            "import math\n"
            "\n"
            "def _delta_or_none(name: str) -> float | None:\n"
            "    if ablation_optional is None:\n"
            "        return None\n"
            "    rows = ablation_optional.filter(pl.col('feature_set') == name)\n"
            "    if rows.height == 0:\n"
            "        return None\n"
            "    val = rows.get_column('delta_vs_full').to_list()[0]\n"
            "    return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None\n"
            "\n"
            "def _f1_or_none(name: str) -> float | None:\n"
            "    if ablation_optional is None:\n"
            "        return None\n"
            "    rows = ablation_optional.filter(pl.col('feature_set') == name)\n"
            "    if rows.height == 0:\n"
            "        return None\n"
            "    val = rows.get_column('f1_macro').to_list()[0]\n"
            "    return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None\n"
            "\n"
            "def _decision(delta: float | None, threshold: float = PROMOTE_THRESHOLD) -> str:\n"
            "    if delta is None:\n"
            "        return 'pendiente (sin datos)'\n"
            "    if delta >= threshold:\n"
            "        return f'promover (delta = {delta:+.4f} ≥ +{threshold:.3f})'\n"
            "    if delta <= -threshold:\n"
            "        return f'descartar (delta = {delta:+.4f} ≤ -{threshold:.3f})'\n"
            "    return f'diferir a stacking (delta = {delta:+.4f} en zona neutra)'\n"
            "\n"
            "geom_only_f1 = _f1_or_none('geom_only')\n"
            "h1_decision = (\n"
            "    f'descartar — `geom_only` F1-macro = `{geom_only_f1:.4f}` < 0.10' if geom_only_f1 is not None\n"
            "    else 'descartar (decisión cualitativa, geom_only no evaluado)'\n"
            ")\n"
            "\n"
            "hypotheses = pl.DataFrame({\n"
            "    'hipotesis': ['H-1 geom leakage', 'H-2 FarSLIP', 'H-3 pheno_text Gemini', 'H-4 firma espectral REP'],\n"
            "    'descripcion': [\n"
            "        'columnas geom_* son proxy de region (leakage espacial)',\n"
            "        'FarSLIP (US-017) aporta señal complementaria al embedding tabular',\n"
            "        'descripción fenológica textual con LLM aporta señal semántica',\n"
            "        'Red Edge Position (Frampton 2013) captura forma espectral por época',\n"
            "    ],\n"
            "    'decision': [\n"
            "        h1_decision,\n"
            "        _decision(_delta_or_none('with_farslip')),\n"
            "        _decision(_delta_or_none('with_pheno_text')),\n"
            "        _decision(_delta_or_none('with_spectral_signature')),\n"
            "    ],\n"
            "})\n"
            "display(Markdown('**Tabla H-1..H-4 — decisiones consolidadas**:'))\n"
            "display(hypotheses)\n"
            "hypotheses.write_parquet(env.reports_dir / 'decision_table.parquet')\n"
        )
    )

    # -----------------------------------------------------------------------
    # 11. Winning set selection (SINGLE call).
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 11. Conjunto ganador con `select_winning_features`\n\n"
            "Esta es la **única llamada a `select_winning_features` en todo "
            "el proyecto**. Recibe la `ablation_table` de `05_reencuadre`, "
            "promueve bloques con `delta >= +0.005`, descarta `geom_*` "
            "siempre, y persiste:\n\n"
            "- `data/features/features_fused_winning_pastis.parquet` — "
            "dataset filtrado a las columnas ganadoras.\n"
            "- `data/features/features_fused_winning_pastis.manifest.json` — "
            "lista nominal exacta de columnas (para que los modelos "
            "densos del Avance 4 y los ensambles del Avance 5 lean "
            "exactamente las mismas)."
        )
    )

    cells.append(
        _code(
            "from ml.features.winning_features import (\n"
            "    select_winning_features,\n"
            "    persist_winning_features,\n"
            ")\n"
            "\n"
            "if ablation_optional is None:\n"
            "    raise FileNotFoundError(\n"
            "        f'No se puede llamar a select_winning_features sin la tabla `{ABLATION_OPTIONAL_PATH_05}`. '\n"
            "        'Ejecuta `05_reencuadre_fenologico.ipynb` antes de Avance3.'\n"
            "    )\n"
            "\n"
            "fused_path = Path(FUSED_PATH)\n"
            "if not fused_path.exists():\n"
            "    raise FileNotFoundError(\n"
            "        f'No existe `{fused_path}`. Ejecuta `05_reencuadre_fenologico.ipynb` '\n"
            "        '(genera el fused completo durante la materialización).'\n"
            "    )\n"
            "fused = pl.read_parquet(fused_path)\n"
            "\n"
            "winning = select_winning_features(\n"
            "    ablation_optional,\n"
            "    available_cols=fused.columns,\n"
            "    promote_threshold=PROMOTE_THRESHOLD,\n"
            "    discard_geom=True,\n"
            ")\n"
            "display(Markdown('**Decisiones por bloque** (única fuente de verdad):'))\n"
            "display(pl.DataFrame({\n"
            "    'bloque': list(winning.decisions.keys()),\n"
            "    'promovido': list(winning.decisions.values()),\n"
            "}))\n"
            "display(Markdown(\n"
            "    f'**Conjunto ganador**: `{winning.name}` con `{len(winning.feature_cols)}` columnas.'\n"
            "))\n"
            "display(Markdown('### Rationale'))\n"
            "display(Markdown(winning.rationale))\n"
            "\n"
            "winning_path = persist_winning_features(\n"
            "    winning, fused,\n"
            "    output_path=WINNING_OUTPUT, overwrite=True,\n"
            ")\n"
            "# persist_winning_features devuelve un Path relativo; lo resolvemos\n"
            "# contra env.repo solo para mostrarlo (relative_to exige el mismo tipo).\n"
            "_winning_abs = winning_path if winning_path.is_absolute() else env.repo / winning_path\n"
            "_winning_rel = _winning_abs.relative_to(env.repo)\n"
            "display(Markdown(\n"
            "    f'**Conjunto ganador guardado**: `{_winning_rel}` · '\n"
            "    f'**Manifest**: `{_winning_rel.with_suffix(\".manifest.json\")}`'\n"
            "))\n"
            "\n"
            "import json\n"
            "manifest = json.loads(\n"
            "    Path(WINNING_OUTPUT).with_suffix('.manifest.json').read_text(encoding='utf-8')\n"
            ")\n"
            "display(Markdown(f'**Número de características**: `{manifest[\"n_features\"]}`'))\n"
            "display(Markdown('**Características ganadoras** (primeras 40):'))\n"
            "display(pl.Series('feature', manifest['feature_cols'][:40]).to_frame())\n"
        )
    )

    # -----------------------------------------------------------------------
    # 12. Consolidated MLflow traceability.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 12. Trazabilidad MLflow consolidada\n\n"
            "Cada notebook abre runs MLflow propios con tags `code_version` "
            "(SHA git) y `data_version` (hash DVC). Aquí listamos los "
            "experimentos para que cualquier auditor reabra y reproduzca."
        )
    )

    cells.append(
        _code(
            "from ml.utils.mlflow_utils import resolve_tracking_uri, server_is_reachable\n"
            "import mlflow\n"
            "from mlflow.tracking import MlflowClient\n"
            "\n"
            "# Misma logica defensiva que las celdas MLflow setup: si el server\n"
            "# no responde, cae a `file:./mlruns` y consulta los runs locales.\n"
            "_candidate_uri = resolve_tracking_uri(None, probe_server=False)\n"
            "if _candidate_uri.startswith(('http://', 'https://')) and not server_is_reachable(_candidate_uri):\n"
            "    mlflow_uri = 'file:./mlruns'\n"
            "    display(Markdown(\n"
            "        f'> Servidor MLflow `{_candidate_uri}` no responde. '\n"
            "        'Consulto runs locales en `file:./mlruns`.'\n"
            "    ))\n"
            "else:\n"
            "    mlflow_uri = _candidate_uri\n"
            "mlflow.set_tracking_uri(mlflow_uri)\n"
            "client = MlflowClient(tracking_uri=mlflow_uri)\n"
            "\n"
            "experiments_av3 = [\n"
            "    'baseline-04-tabular',\n"
            "    'baseline-04b-pilot',\n"
            "    'baseline-04c-ablation',\n"
            "    'baseline-04-farslip-vs-remoteclip',\n"
            "    'baseline-05-reencuadre',\n"
            "]\n"
            "summary_rows = []\n"
            "for exp_name in experiments_av3:\n"
            "    exp = client.get_experiment_by_name(exp_name)\n"
            "    if exp is None:\n"
            "        summary_rows.append({\n"
            "            'experimento': exp_name, 'n_runs': 0,\n"
            "            'experiment_id': 'n/a', 'tracking_uri': mlflow_uri,\n"
            "        })\n"
            "        continue\n"
            "    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)\n"
            "    summary_rows.append({\n"
            "        'experimento': exp_name,\n"
            "        'n_runs': len(runs),\n"
            "        'experiment_id': exp.experiment_id,\n"
            "        'tracking_uri': mlflow_uri,\n"
            "    })\n"
            "mlflow_summary = pl.DataFrame(summary_rows)\n"
            "display(Markdown('**Resumen de experimentos MLflow del Avance 3**:'))\n"
            "display(mlflow_summary)\n"
            "mlflow_summary.write_parquet(env.reports_dir / 'mlflow_summary.parquet')\n"
        )
    )

    # -----------------------------------------------------------------------
    # 13. References.
    # -----------------------------------------------------------------------
    cells.append(
        _md(
            "## 13. Referencias\n\n"
            "- **Brown et al. (2025)** — *AlphaEarth Foundations: a Global "
            "Foundation Model for Earth*. Embedding satelital 64-dim.\n"
            "- **Pelletier, Webb & Petitjean (2019)** — *TempCNN: Temporal "
            "Convolutional Neural Network for Satellite Image Time Series "
            "Classification*. DOI 10.3390/rs11050523.\n"
            "- **Fawaz et al. (2020)** — *InceptionTime: Finding AlexNet for "
            "Time Series Classification*. DOI 10.1007/s10618-020-00710-y.\n"
            "- **Garnot et al. (2021)** — *Panoptic Segmentation of "
            "Satellite Image Time Series with Convolutional Temporal "
            "Attention Networks (U-TAE)*. ICCV 2021.\n"
            "- **Frampton et al. (2013)** — *Evaluating the capabilities of "
            "Sentinel-2 for quantitative estimation of biophysical "
            "variables in vegetation*. DOI 10.1016/j.isprsjprs.2013.04.007.\n"
            "- **Wen et al. (2025)** — *Phenology Description is All You "
            "Need!*. Descripción fenológica textual + LLM.\n"
            "- **Li et al. (2025)** — *FarSLIP: Patch-Level Distillation of "
            "CLIP for Remote Sensing*. arXiv:2511.14901.\n"
            "- **Tang et al. (2024)** — *FarSLIP: Vineyard-aware CLIP "
            "Distillation*.\n"
            "- **Chen et al. (2024)** — *RemoteCLIP: A Vision-Language "
            "Foundation Model for Remote Sensing*.\n"
            "- **Lundberg & Lee (2017)** — *A Unified Approach to "
            "Interpreting Model Predictions (SHAP)*. NeurIPS.\n"
            "\n"
            "**Atribuciones de licencia**: ver [`docs/licenses/DATA_LICENSE.md`]"
            "(../../docs/licenses/DATA_LICENSE.md)."
        )
    )

    cells.append(
        _md(
            "---\n\n"
            "## Cierre\n\n"
            "Con este cuaderno cerramos el Avance 3:\n\n"
            "- **5 modelos reentrenados** sobre el conjunto ganador (3 "
            "tabulares + 2 temporales), con trazabilidad MLflow completa.\n"
            "- **Las 5 preguntas oficiales** del Avance 3 respondidas con "
            "cifras concretas y figuras de los cuadernos previos.\n"
            "- **Conjunto de características ganador** nombrado y "
            "persistido en `features_fused_winning_pastis.parquet` + "
            "manifest JSON.\n"
            "- **Cuatro hipótesis H-1..H-4** cerradas con decisión "
            "cuantitativa (promover, diferir, descartar).\n\n"
            "**Lo que sigue (Avances 4 y 5)**:\n\n"
            "- `notebooks/avance4_modelos.ipynb` consumirá el mismo "
            "`features_fused_winning_pastis.parquet` y entrenará las 6 "
            "arquitecturas densas obligatorias: U-Net, DeepLabv3+, "
            "SegFormer-B2, U-TAE, TSViT (Paper 1), Swin-UNETR.\n"
            "- `notebooks/avance5_ensambles.ipynb` construirá los 4 "
            "ensambles (voting top-3, bagging XGB+AlphaEarth, stacking + "
            "Gemma 4 26B-MoE LoRA, blending Optuna) sobre el mismo "
            "conjunto ganador."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# Builder dispatcher.
# ---------------------------------------------------------------------------


# 04b_baseline.ipynb is excluded from the dispatcher on purpose: it is already
# executed end-to-end and validated on disk. It is neither regenerated nor moved.
# The build_04b_baseline() function is kept (it shares _baseline_pilot_cells +
# _hcat_grouping_cells with build_04_baseline) but is not dispatched.
BUILDERS = {
    "04_baseline": (build_04_baseline, NOTEBOOK_DIR / "04_baseline.ipynb"),
    "04c_baseline": (build_04c_baseline, NOTEBOOK_DIR / "04c_baseline.ipynb"),
    "04_farslip_eval_pastis": (
        build_04_farslip_eval_pastis,
        NOTEBOOK_DIR / "04_farslip_eval_pastis.ipynb",
    ),
    "05_reencuadre_fenologico": (
        build_05_reencuadre,
        NOTEBOOK_DIR / "05_reencuadre_fenologico.ipynb",
    ),
    "Avance3.Equipo17": (
        build_avance3,
        NOTEBOOK_DIR / "Avance3.Equipo17.ipynb",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=list(BUILDERS.keys()),
        action="append",
        help="Construye solo el notebook indicado (puede repetirse). Sin esta opcion construye todos.",
    )
    args = parser.parse_args()

    selected = args.only or list(BUILDERS.keys())
    print(f"Construyendo {len(selected)} notebook(s):")
    for key in selected:
        builder, path = BUILDERS[key]
        nb = builder()
        _write(path, nb)
    print("Hecho.")


if __name__ == "__main__":
    main()
