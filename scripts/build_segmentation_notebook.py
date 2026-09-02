"""Builder of the dense segmentation notebooks (Avance 4), one per model.

Generates an independent notebook per architecture so they can be run in
parallel in separate Colab sessions:

- ``04d_segmentation_unet.ipynb``   -> U-Net ResNet-50
- ``04e_segmentation_anysat.ipynb`` -> frozen AnySat + linear head

Each notebook is Colab-first (mounts Drive, clones the repo, reads the dataset,
trains with checkpoint resume) and leaves its artifacts in clear folders of the
shared Drive, to cite them later in the report:

    reports/segmentation/metrics/      per-model metrics parquet
    reports/segmentation/figures/      confusion matrix PNG
    reports/segmentation/checkpoints/  final model + resumable checkpoint

Usage::

    poetry run python scripts/build_segmentation_notebook.py --model unet
    poetry run python scripts/build_segmentation_notebook.py --model anysat

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_OUT_BY_MODEL = {
    "unet": Path("notebooks/segmentation/04d_segmentation_unet.ipynb"),
    "anysat": Path("notebooks/segmentation/04e_segmentation_anysat.ipynb"),
}

# Setup cell (identical in both notebooks): mounts Drive, clones the repo and
# locates the pyproject. Adjust _branch if the code lives in another branch.
_SETUP_CELL = (
    "# Setup del entorno. En Colab se monta Drive (donde vive el dataset) y se\n"
    "# instalan las dependencias que no vienen por defecto; en local no hace falta.\n"
    "import os, sys, subprocess\n"
    "from pathlib import Path\n\n"
    "_IN_COLAB = False\n"
    "shared_folder_path = ''\n"
    "try:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    shared_folder_path = '/content/drive/MyDrive/Integrador/'\n"
    "    _IN_COLAB = True\n"
    "except ImportError:\n"
    "    pass\n\n"
    "# En Colab el repo no esta presente: se clona una vez en /content/agrosat-copilot.\n"
    "if _IN_COLAB:\n"
    "    from getpass import getpass\n"
    "    _repo_dir = '/content/agrosat-copilot'\n"
    "    _branch = 'main'\n"
    "    _repo = 'github.com/ArthurZizumbo/agrosat-copilot.git'\n"
    "    if not Path(_repo_dir, 'pyproject.toml').is_file():\n"
    "        _rc = os.system(f'git clone --branch {_branch} --depth 1 "
    "https://{_repo} {_repo_dir}')\n"
    "        if _rc != 0:  # repo privado: pide token (no se guarda en el notebook)\n"
    "            _tok = getpass('GitHub token (repo privado): ')\n"
    "            os.system(f'git clone --branch {_branch} --depth 1 "
    "https://{_tok}@{_repo} {_repo_dir}')\n\n"
    "# El codigo no vive en Drive: se localiza el repo por su pyproject.toml.\n"
    "_search = [Path.cwd().resolve(), *Path.cwd().resolve().parents]\n"
    "if _IN_COLAB:\n"
    "    _search = [Path('/content/agrosat-copilot'), *_search]\n"
    "for _cand in _search:\n"
    "    if (_cand / 'pyproject.toml').is_file():\n"
    "        if str(_cand) not in sys.path:\n"
    "            sys.path.insert(0, str(_cand))\n"
    "        os.chdir(_cand)\n"
    "        break\n"
    "else:\n"
    "    raise RuntimeError('No se encontro el repo agrosat-copilot (pyproject.toml). '\n"
    "                       'Clonalo en /content/agrosat-copilot o sincronizalo desde VS Code.')\n\n"
    "if _IN_COLAB:\n"
    "    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',\n"
    "                    'segmentation-models-pytorch', 'structlog', 'typer', 'polars', "
    "'mlflow', 'optuna'], check=False)\n\n"
    "print('repo:', Path.cwd(), '| colab:', _IN_COLAB, '| drive:', shared_folder_path or '(local)')"
)

_COPY_CELL = (
    "# Copia del dataset de Drive al disco local, con barra de progreso.\n"
    "import shutil, time\n\n"
    "def copy_pastis_to_local(src_root, dst_root,\n"
    "                         subdirs=('DATA_S2', 'ANNOTATIONS'),\n"
    "                         files=('metadata.geojson', 'NORM_S2_patch.json')):\n"
    "    src_root, dst_root = Path(src_root), Path(dst_root)\n"
    "    dst_root.mkdir(parents=True, exist_ok=True)\n"
    "    todo = []\n"
    "    for sub in subdirs:\n"
    "        for f in sorted((src_root / sub).glob('*')):\n"
    "            if f.is_file():\n"
    "                todo.append((f, dst_root / sub / f.name))\n"
    "    for fname in files:\n"
    "        sp = src_root / fname\n"
    "        if sp.is_file():\n"
    "            todo.append((sp, dst_root / fname))\n"
    "    if not todo:\n"
    "        raise FileNotFoundError(f'No se hallaron DATA_S2/ANNOTATIONS en {src_root}')\n"
    "    total_bytes = sum(s.stat().st_size for s, _ in todo)\n"
    "    try:\n"
    "        from tqdm.auto import tqdm\n"
    "        bar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Copiando PASTIS')\n"
    "    except Exception:\n"
    "        bar = None\n"
    "    t0 = time.time()\n"
    "    for i, (src, dst) in enumerate(todo, 1):\n"
    "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
    "        # Salta el archivo si ya esta copiado con el mismo tamano.\n"
    "        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):\n"
    "            shutil.copy2(src, dst)\n"
    "        if bar is not None:\n"
    "            bar.update(src.stat().st_size)\n"
    "        elif i % 200 == 0:\n"
    "            print(f'  {i}/{len(todo)} archivos...')\n"
    "    if bar is not None:\n"
    "        bar.close()\n"
    "    print(f'Listo: {len(todo)} archivos ({total_bytes / 1e9:.1f} GB) en "
    "{time.time() - t0:.0f}s -> {dst_root}')\n"
    "    return dst_root\n\n"
    "if _IN_COLAB and COPY_TO_LOCAL:\n"
    "    PASTIS_ROOT = copy_pastis_to_local(PASTIS_ROOT, '/content/PASTIS-R')\n"
    "    print('PASTIS_ROOT (local):', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
    "else:\n"
    "    print('Lectura directa desde:', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())"
)

_SPLIT_CELL = (
    "# Split en los folds oficiales de PASTIS (espacialmente disjuntos).\n"
    "from ml.ingest.pastis_dataset import pastis_fold_split\n\n"
    "split = pastis_fold_split(PASTIS_ROOT, train_folds=(1, 2, 3), val_folds=(4,), "
    "test_folds=(5,))\n"
    "print({k: len(v) for k, v in split.items()})"
)

_META = {
    "unet": {
        "title": "# Segmentación de cultivos con U-Net (ResNet-50) sobre PASTIS-R",
        "intro": (
            "Se entrena una U-Net de segmentación densa sobre las series Sentinel-2 de PASTIS-R "
            "y se evalúa su desempeño píxel a píxel. El encoder ResNet-50 viene preentrenado en "
            "ImageNet y se adapta a las diez bandas; como entrada se usa la mediana temporal de la "
            "serie y la salida es un mapa de clases a la resolución de la imagen.\n\n"
            "Este cuaderno corre de forma independiente (en paralelo con el de AnySat) y deja sus "
            "artefactos en carpetas del Drive compartido para el reporte: la tabla de métricas, la "
            "figura de la matriz de confusión y el modelo entrenado."
        ),
        "model_md": (
            "## Entrenamiento\n\n"
            "El entrenamiento guarda un checkpoint por época en Drive; si la sesión se reinicia, al "
            "volver a ejecutar esta celda se reanuda desde la última época completada en vez de "
            "empezar de cero."
        ),
        "batch": "16",
        "reduction": "median",
        "confusion_import": "from ml.models.segmentation import build_unet",
        "confusion_build": "lambda: build_unet(20, encoder_weights=None)",
    },
    "anysat": {
        "title": "# Segmentación de cultivos con AnySat (congelado) sobre PASTIS-R",
        "intro": (
            "Se entrena un segmentador basado en AnySat (Astruc et al., 2024), un modelo "
            "fundacional para datos de observación de la Tierra. AnySat se usa congelado, como "
            "extractor de características, y solo se entrena una cabeza lineal que las proyecta a "
            "las clases de cultivo; el entrenamiento es barato porque el grueso de los pesos no se "
            "actualiza.\n\n"
            "Este cuaderno corre de forma independiente (en paralelo con el de U-Net) y deja sus "
            "artefactos en carpetas del Drive compartido para el reporte: la tabla de métricas, la "
            "figura de la matriz de confusión y el modelo entrenado."
        ),
        "model_md": (
            "## Entrenamiento\n\n"
            "AnySat se descarga la primera vez desde su repositorio (torch.hub). El entrenamiento "
            "guarda un checkpoint por época en Drive; si la sesión se reinicia, al volver a "
            "ejecutar esta celda se reanuda desde la última época completada."
        ),
        "batch": "8",
        "reduction": "none",
        "confusion_import": "from ml.models.anysat_wrapper import AnySatSegmenter",
        "confusion_build": "lambda: AnySatSegmenter(20, target_size=TARGET_SIZE)",
    },
}


def _build_cells(
    model: str,
    *,
    num_workers: int = -1,
    batch: int = -1,
    epochs: int = 30,
    target_size: int = 256,
    subset: int = 0,
    suffix: str = "",
) -> list:
    """Build the notebook cells for a specific architecture.

    Args:
        model: ``unet`` or ``anysat``.
        num_workers: Worker override (``-1`` keeps the default ``4 if colab``).
        batch: Batch override (``-1`` keeps the model's default).
        epochs: Number of epochs.
        target_size: Spatial resolution (256 by default; AnySat uses 64 for VRAM).
        subset: Limit of patches per split (0 = all).
        suffix: Suffix for the artifacts (parquet/checkpoints), useful to run
            a variant in parallel without overwriting the main run.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    meta = _META[model]
    batch_val = batch if batch > 0 else int(meta["batch"])
    workers_expr = str(num_workers) if num_workers >= 0 else "4 if _IN_COLAB else 0"
    cells = []

    cells.append(md(meta["title"] + "\n\n" + meta["intro"]))

    cells.append(
        md(
            "## Datos y métricas\n\n"
            "PASTIS-R entrega parches Sentinel-2 multitemporales de 128x128, que aquí se "
            "reescalan a 256. Las etiquetas tienen 20 clases: fondo, 18 tipos de cultivo y una "
            "clase void que se descarta en la pérdida y en las métricas. El split de "
            "entrenamiento y validación usa los folds oficiales del dataset, espacialmente "
            "disjuntos. Se reportan mIoU, F1-macro y exactitud a nivel de píxel en dos esquemas: "
            "las 18 clases planas y los 6 grupos agronómicos HCAT (cereales, oleaginosas, "
            "tubérculos, leguminosas, leñosos y otros), siendo este último el comparable con el "
            "baseline del avance anterior."
        )
    )

    cells.append(code(_SETUP_CELL))

    cells.append(
        code(
            "# Configuracion de la corrida.\n"
            "import torch\n\n"
            f"MODEL = '{model}'\n"
            f"SUFFIX = '{suffix}'          # sufijo de artefactos (para correr variantes en paralelo)\n"
            f"REDUCTION = '{meta['reduction']}'    # 'median' (U-Net) o 'none' (AnySat, serie temporal)\n"
            "# El dataset vive en Drive; en local se usa la copia del repo.\n"
            "PASTIS_ROOT = Path((shared_folder_path + 'data/PASTIS-R') if shared_folder_path\n"
            "                   else 'data/PASTIS-R')\n"
            "# Carpetas de artefactos en Drive (claras para citarlas en el reporte):\n"
            "#   reports/segmentation/metrics      -> parquet de metricas por modelo\n"
            "#   reports/segmentation/figures      -> PNG de la matriz de confusion\n"
            "#   reports/segmentation/checkpoints  -> modelo final + checkpoint reanudable\n"
            "SEG_DIR = Path((shared_folder_path if shared_folder_path else '') + 'reports/segmentation')\n"
            "METRICS_DIR = SEG_DIR / 'metrics'\n"
            "FIGURES_DIR = SEG_DIR / 'figures'\n"
            f"CHECKPOINT_DIR = SEG_DIR / 'checkpoints{suffix}'\n"
            "for _d in (METRICS_DIR, FIGURES_DIR, CHECKPOINT_DIR):\n"
            "    _d.mkdir(parents=True, exist_ok=True)\n"
            f"COMPARISON_PATH = METRICS_DIR / f'model_comparison_avance4_{{MODEL}}{suffix}.parquet'\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            f"TARGET_SIZE = {target_size}\n"
            f"SUBSET = {subset}            # 0 = todos; reducir si la sesion es corta\n"
            f"EPOCHS = {epochs}\n"
            f"BATCH = {batch_val}\n"
            "MLFLOW_URI = 'file:./mlruns'\n"
            "# Por defecto se lee directo de Drive (sin copiar). Si vas a entrenar muchas epocas y\n"
            "# preferis acelerar, pone COPY_TO_LOCAL=True (copia una vez al disco efimero).\n"
            "COPY_TO_LOCAL = False\n"
            "# Leyendo de Drive, num_workers=0 va mas rapido (el FUSE de Drive penaliza la\n"
            "# concurrencia). Con el dataset copiado a local conviene subirlo a 2-4.\n"
            f"NUM_WORKERS = {workers_expr}\n\n"
            "print('modelo:', MODEL, '| device:', DEVICE, '| batch:', BATCH)\n"
            "print('PASTIS_ROOT:', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
            "print('artefactos en:', SEG_DIR)"
        )
    )

    cells.append(
        md(
            "## Lectura del dataset\n\n"
            "Por defecto el dataset se lee directo desde Drive, sin copiar nada: así se evita la "
            "espera inicial y no se pierde trabajo si la sesión se reinicia. El loader abre cada "
            "parche con un solo acceso a disco (no relee el archivo de metadatos en cada paso) y el "
            "DataLoader usa varios procesos en paralelo. Si preferís acelerar, poné `COPY_TO_LOCAL = "
            "True` en la celda anterior para copiar una vez al disco local de la sesión."
        )
    )

    cells.append(code(_COPY_CELL))
    cells.append(code(_SPLIT_CELL))

    cells.append(md(meta["model_md"]))

    if model == "unet":
        train_code = (
            "# Entrenamiento de la U-Net.\n"
            "from ml.train.train_segmentation import run_training\n\n"
            "result = run_training(\n"
            "    model=MODEL, epochs=EPOCHS, batch_size=BATCH, target_size=TARGET_SIZE,\n"
            "    subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "    comparison_path=COMPARISON_PATH, num_workers=NUM_WORKERS, output_dir=CHECKPOINT_DIR,\n"
            ")\n"
            "result"
        )
    else:
        train_code = (
            "# Carga de AnySat (torch.hub) y entrenamiento de la cabeza lineal.\n"
            "from ml.train.train_segmentation import run_training\n"
            "from ml.models.anysat_wrapper import load_anysat_encoder\n\n"
            "_ = load_anysat_encoder()  # descarga y valida los pesos antes de entrenar\n"
            "result = run_training(\n"
            "    model=MODEL, epochs=EPOCHS, batch_size=BATCH, target_size=TARGET_SIZE,\n"
            "    subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "    comparison_path=COMPARISON_PATH, num_workers=NUM_WORKERS, output_dir=CHECKPOINT_DIR,\n"
            ")\n"
            "result"
        )
    cells.append(code(train_code))

    cells.append(
        md(
            "## Métricas\n\n"
            "Tabla de métricas de este modelo sobre el fold de validación, en los dos esquemas (18 "
            "clases y 6 grupos HCAT). Se guarda en `reports/segmentation/metrics/`; el notebook "
            "integrador la une con la del otro modelo para la comparativa final. Las columnas con "
            "sufijo `grouped` corresponden a los 6 grupos (el fondo no entra en esas métricas)."
        )
    )

    cells.append(
        code(
            "import polars as pl\n\n"
            "table = pl.read_parquet(COMPARISON_PATH)\n"
            "cols = ['model', 'miou_grouped', 'f1_macro_grouped', 'pixel_accuracy_grouped',\n"
            "        'miou', 'f1_macro', 'pixel_accuracy', 'train_time_s', 'epochs']\n"
            "table.select([c for c in cols if c in table.columns])"
        )
    )

    cells.append(
        md(
            "## Matriz de confusión\n\n"
            "Recall por clase a nivel de píxel sobre el fold de validación, sin contar la clase "
            "void. La figura se guarda en `reports/segmentation/figures/` para el reporte."
        )
    )

    cells.append(
        code(
            "# Matriz de confusion a nivel de pixel; se guarda como PNG en Drive.\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "from ml.ingest.pastis_dataset import PASTISDataset, load_norm_stats, PASTIS_IGNORE_INDEX\n"
            "from ml.ingest.pastis_loader import PASTIS_CLASS_MAP\n"
            "from ml.eval.dense_metrics import dense_confusion_figure\n"
            f"{meta['confusion_import']}\n\n"
            "def confusion_figure(model_name, reduction, build_fn, ckpt, max_patches=40):\n"
            "    norm = load_norm_stats(PASTIS_ROOT, folds=(1, 2, 3))\n"
            "    val_ids = split['val'][:max_patches]\n"
            "    ds = PASTISDataset(val_ids, root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                       temporal_reduction=reduction, norm=norm)\n"
            "    loader = DataLoader(ds, batch_size=2)\n"
            "    model = build_fn().to(DEVICE)\n"
            "    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))\n"
            "    model.eval()\n"
            "    preds, tgts = [], []\n"
            "    with torch.no_grad():\n"
            "        for b in loader:\n"
            "            img = b['image'].to(DEVICE)\n"
            "            out = model(img) if model_name == 'unet' else model(img, b['dates'].to(DEVICE))\n"
            "            preds.append(out.argmax(1).cpu().reshape(-1))\n"
            "            tgts.append(b['semantic'].reshape(-1))\n"
            "    return dense_confusion_figure(torch.cat(preds), torch.cat(tgts),\n"
            "                                  class_names=PASTIS_CLASS_MAP, ignore_index=PASTIS_IGNORE_INDEX)\n\n"
            f"fig = confusion_figure(MODEL, '{meta['reduction']}', {meta['confusion_build']},\n"
            "                       result['checkpoint_path'])\n"
            f"_fig_path = FIGURES_DIR / f'confusion_{{MODEL}}{suffix}.png'\n"
            "fig.savefig(_fig_path, bbox_inches='tight', dpi=120)\n"
            "print('Figura guardada en:', _fig_path)\n"
            "fig"
        )
    )

    cells.append(
        md(
            "## IoU por clase\n\n"
            "IoU de cada clase sobre el fold de validación, ordenado de menor a mayor, con la "
            "línea del mIoU. Identifica qué cultivos resuelve bien el modelo y cuáles le cuestan. "
            "Carga el modelo guardado (no re-entrena) y guarda el PNG en `figures/`."
        )
    )
    cells.append(
        code(
            "# IoU por clase (carga el modelo guardado, no re-entrena).\n"
            "import torch, numpy as np, matplotlib.pyplot as plt\n"
            "from torch.utils.data import DataLoader\n"
            "from ml.ingest.pastis_dataset import PASTISDataset, load_norm_stats\n"
            "from ml.ingest.pastis_loader import PASTIS_CLASS_MAP\n"
            "from ml.eval.dense_metrics import DenseConfusionAccumulator\n"
            f"{meta['confusion_import']}\n\n"
            "_norm = load_norm_stats(PASTIS_ROOT, folds=(1, 2, 3))\n"
            "_ds = PASTISDataset(split['val'][:40], root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                    temporal_reduction=REDUCTION, norm=_norm)\n"
            "_loader = DataLoader(_ds, batch_size=2)\n"
            f"_model = ({meta['confusion_build']})().to(DEVICE)\n"
            "_model.load_state_dict(torch.load(result['checkpoint_path'], map_location=DEVICE))\n"
            "_model.eval()\n"
            "_acc = DenseConfusionAccumulator(20, ignore_index=19, device=str(DEVICE))\n"
            "with torch.no_grad():\n"
            "    for _b in _loader:\n"
            "        _img = _b['image'].to(DEVICE)\n"
            "        _out = _model(_img) if MODEL == 'unet' else _model(_img, _b['dates'].to(DEVICE))\n"
            "        _acc.update(_out.argmax(1), _b['semantic'].to(DEVICE))\n"
            "_iou = _acc.per_class_iou()\n"
            "_items = sorted(_iou.items(), key=lambda kv: kv[1])\n"
            "_names = [PASTIS_CLASS_MAP.get(c, str(c)) for c, _ in _items]\n"
            "_vals = [v for _, v in _items]\n"
            "_miou = float(np.mean(_vals)) if _vals else 0.0\n"
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "ax.barh(_names, _vals, color='#4C72B0')\n"
            "ax.axvline(_miou, color='red', linestyle='--', label=f'mIoU={_miou:.3f}')\n"
            "ax.set_xlim(0, 1); ax.set_xlabel('IoU'); ax.legend()\n"
            "ax.set_title(f'IoU por clase - {MODEL} (mIoU={_miou:.3f})')\n"
            "fig.tight_layout()\n"
            "_p = FIGURES_DIR / f'per_class_iou_{MODEL}{SUFFIX}.png'\n"
            "fig.savefig(_p, bbox_inches='tight', dpi=120); print('guardado:', _p)\n"
            "fig"
        )
    )

    cells.append(
        md(
            "## Comparación visual (RGB / verdad / predicción)\n\n"
            "Para unas parcelas de validación: la imagen Sentinel-2 en color (bandas B04/B03/B02), "
            "la máscara verdadera y la predicción del modelo, con el mismo colormap de clases. "
            "Es la vista cualitativa que muestra de un vistazo qué tan bien delinea las parcelas."
        )
    )
    cells.append(
        code(
            "# RGB | ground truth | prediccion para N parcelas (carga el modelo guardado).\n"
            "import torch, numpy as np, matplotlib.pyplot as plt\n"
            "from matplotlib.colors import ListedColormap\n"
            "from ml.ingest.pastis_dataset import PASTISDataset, load_norm_stats\n"
            "from ml.ingest.pastis_loader import PASTIS_S2_BANDS\n"
            f"{meta['confusion_import']}\n\n"
            "_norm = load_norm_stats(PASTIS_ROOT, folds=(1, 2, 3))\n"
            "_ds = PASTISDataset(split['val'][:4], root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                    temporal_reduction=REDUCTION, norm=_norm)\n"
            f"_model = ({meta['confusion_build']})().to(DEVICE)\n"
            "_model.load_state_dict(torch.load(result['checkpoint_path'], map_location=DEVICE))\n"
            "_model.eval()\n"
            "_cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))\n"
            "_ri, _gi, _bi = (PASTIS_S2_BANDS.index(x) for x in ('B04', 'B03', 'B02'))\n"
            "_n = len(_ds)\n"
            "fig, axes = plt.subplots(_n, 3, figsize=(10, 3 * _n))\n"
            "axes = np.atleast_2d(axes)\n"
            "with torch.no_grad():\n"
            "    for _k in range(_n):\n"
            "        _it = _ds[_k]\n"
            "        _img = _it['image']\n"
            "        _arr = _img if _img.dim() == 3 else _img.median(0).values\n"
            "        _rgb = _arr[[_ri, _gi, _bi]].permute(1, 2, 0).numpy()\n"
            "        _lo, _hi = np.percentile(_rgb, 2), np.percentile(_rgb, 98)\n"
            "        _rgb = np.clip((_rgb - _lo) / (_hi - _lo + 1e-6), 0, 1)\n"
            "        _x = _img.unsqueeze(0).to(DEVICE)\n"
            "        _out = (_model(_x) if MODEL == 'unet'\n"
            "                else _model(_x, _it['dates'].unsqueeze(0).to(DEVICE)))\n"
            "        _pred = _out.argmax(1)[0].cpu().numpy()\n"
            "        _gt = _it['semantic'].numpy()\n"
            "        axes[_k, 0].imshow(_rgb)\n"
            "        axes[_k, 1].imshow(_gt, cmap=_cmap, vmin=0, vmax=19)\n"
            "        axes[_k, 2].imshow(_pred, cmap=_cmap, vmin=0, vmax=19)\n"
            "        for _j, _t in enumerate(('RGB', 'Ground truth', 'Prediction')):\n"
            "            axes[_k, _j].axis('off')\n"
            "            if _k == 0:\n"
            "                axes[_k, _j].set_title(_t)\n"
            "fig.tight_layout()\n"
            "_p = FIGURES_DIR / f'samples_{MODEL}{SUFFIX}.png'\n"
            "fig.savefig(_p, bbox_inches='tight', dpi=120); print('guardado:', _p)\n"
            "fig"
        )
    )

    cells.append(
        md(
            "## Curvas de entrenamiento\n\n"
            "Evolución del loss de entrenamiento y del mIoU de validación por época, a partir del "
            "historial que el entrenamiento guarda en `metrics/history_<modelo>.parquet`. Sirve para "
            "ver convergencia y si el modelo se estanca. Si el modelo se entrenó con una versión "
            "anterior sin historial, esta celda lo avisa."
        )
    )
    cells.append(
        code(
            "# Curvas de loss y mIoU por epoca (desde el historial guardado).\n"
            "import polars as pl, matplotlib.pyplot as plt\n\n"
            "_hp = METRICS_DIR / f'history_{MODEL}{SUFFIX}.parquet'\n"
            "if _hp.exists():\n"
            "    _h = pl.read_parquet(_hp).sort('epoch')\n"
            "    _ep = _h['epoch'].to_list()\n"
            "    fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    _a1.plot(_ep, _h['train_loss'].to_list(), label='Train')\n"
            "    _a1.set_title('Loss'); _a1.set_xlabel('Epoch'); _a1.legend()\n"
            "    _a2.plot(_ep, _h['miou'].to_list(), label='Val mIoU (18 clases)', color='orange')\n"
            "    if 'miou_grouped' in _h.columns:\n"
            "        _a2.plot(_ep, _h['miou_grouped'].to_list(), label='Val mIoU (6 grupos)', color='green')\n"
            "    _a2.set_title('mIoU'); _a2.set_xlabel('Epoch'); _a2.legend()\n"
            "    fig.tight_layout()\n"
            "    _p = FIGURES_DIR / f'curves_{MODEL}{SUFFIX}.png'\n"
            "    fig.savefig(_p, bbox_inches='tight', dpi=120); print('guardado:', _p)\n"
            "    display(fig)\n"
            "else:\n"
            "    print('No hay history parquet en', _hp)\n"
            "    print('El modelo se entreno con una version sin historial; las curvas no estan')\n"
            "    print('disponibles para esta corrida (re-entrenar con el codigo nuevo las genera).')"
        )
    )

    cells.append(
        md(
            "## Conclusiones\n\n"
            "Las métricas y la matriz de confusión quedan guardadas en `reports/segmentation/` "
            "(carpetas `metrics/` y `figures/`) y el modelo entrenado en `checkpoints/`. El "
            "notebook integrador `Avance4.Equipo17` reúne este modelo con el otro para la "
            "comparativa final, elige el de mejor desempeño y, si vale la pena, afina sus "
            "hiperparámetros con una búsqueda más fina como la del bloque siguiente."
        )
    )

    cells.extend(_tuning_cells(model))
    return cells


def _tuning_cells(model: str) -> list[dict]:
    """Model-specific Optuna fine-tuning cells (>=30 trials).

    In AnySat the encoder is frozen: its features are cached once and each
    trial trains only the linear head (seconds/trial). In the U-Net the full
    network is retrained per trial on a reduced subset. Both export
    ``tuning_<model>.parquet`` consumed by the integrator.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    if model == "anysat":
        tuning_md = (
            "## Ajuste fino del top-2 (Optuna)\n\n"
            "Como AnySat entra al top-2, se afinan sus hiperparametros con Optuna (>=30 trials). "
            "El encoder esta congelado, asi que sus features densas no cambian entre trials: se "
            "cachean una sola vez y cada trial entrena solo la cabeza lineal (Conv 1x1) sobre ese "
            "cache, en segundos en vez de los ~30 min por epoca que cuesta re-correr el encoder. "
            "Se busca `lr` y `weight_decay` maximizando el mIoU de los 6 grupos HCAT; el resumen se "
            "exporta a `reports/segmentation/metrics/tuning_<modelo>.parquet`, que el integrador "
            "`Avance4.Equipo17` levanta y muestra."
        )
        tuning_code = (
            "# Ajuste fino de la cabeza lineal con Optuna (>=30 trials). El encoder de AnySat\n"
            "# esta congelado: se cachean sus features UNA vez y cada trial entrena solo la\n"
            "# cabeza Conv1x1 sobre ese cache (segundos por trial en vez de ~30 min/epoca).\n"
            "import optuna\n"
            "import polars as pl\n"
            "from ml.models.anysat_wrapper import AnySatSegmenter, load_anysat_encoder\n"
            "from ml.ingest.pastis_dataset import load_norm_stats, pastis_fold_split, PASTIS_NUM_CLASSES\n"
            "from ml.tune.anysat_head_tuning import cache_encoder_features, train_head\n\n"
            "TUNE_TRIALS = 30\n"
            "TUNE_EPOCHS = 8\n"
            "TUNE_SUBSET = SUBSET if SUBSET else 300\n\n"
            "# Split y normalizacion identicos al entrenamiento (folds oficiales, sin leakage).\n"
            "_split = pastis_fold_split(PASTIS_ROOT, train_folds=(1, 2, 3), val_folds=(4,), test_folds=())\n"
            "_tr, _va = _split['train'], _split['val']\n"
            "if TUNE_SUBSET:\n"
            "    _tr, _va = _tr[:TUNE_SUBSET], _va[:max(1, TUNE_SUBSET // 2)]\n"
            "_norm = load_norm_stats(PASTIS_ROOT, folds=(1, 2, 3))\n\n"
            "# Cacheo de features del encoder congelado (una sola pasada por patch).\n"
            "_model = AnySatSegmenter(PASTIS_NUM_CLASSES, target_size=TARGET_SIZE, encoder=load_anysat_encoder())\n"
            "print('cacheando features del encoder (train/val)...')\n"
            "_train_cache = cache_encoder_features(_model, _tr, root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                                      norm=_norm, device=DEVICE, batch_size=4, num_workers=NUM_WORKERS)\n"
            "_val_cache = cache_encoder_features(_model, _va, root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                                    norm=_norm, device=DEVICE, batch_size=4, num_workers=NUM_WORKERS)\n"
            "print('features cacheadas:', len(_train_cache), 'train /', len(_val_cache),\n"
            "      'val | D =', _train_cache.feature_dim)\n\n"
            "def _objective(trial):\n"
            "    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)   # cabeza lineal: tolera lr mas alto\n"
            "    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)\n"
            "    def _report(epoch, metrics):\n"
            "        trial.report(metrics['miou_grouped'], step=epoch)\n"
            "        if trial.should_prune():\n"
            "            raise optuna.TrialPruned()\n"
            "    best = train_head(_train_cache, _val_cache, num_classes=PASTIS_NUM_CLASSES,\n"
            "                      target_size=TARGET_SIZE, lr=lr, weight_decay=wd, epochs=TUNE_EPOCHS,\n"
            "                      batch_size=8, device=DEVICE, seed=trial.number, on_epoch=_report)\n"
            "    return best['miou_grouped']\n\n"
            "study = optuna.create_study(\n"
            "    direction='maximize', study_name=f'tune-{MODEL}',\n"
            "    sampler=optuna.samplers.TPESampler(seed=42),\n"
            "    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),\n"
            ")\n"
            "study.optimize(_objective, n_trials=TUNE_TRIALS)\n\n"
            "# Resumen de todos los trials -> metrics/tuning_<modelo>.parquet (lo consume el Avance4).\n"
            "_rows = [{\n"
            "    'model': MODEL,\n"
            "    'trial': t.number,\n"
            "    'state': t.state.name,\n"
            "    'lr': t.params.get('lr'),\n"
            "    'weight_decay': t.params.get('weight_decay'),\n"
            "    'batch_size': BATCH,\n"
            "    'miou_grouped': t.value,\n"
            "} for t in study.trials]\n"
            "tuning = pl.DataFrame(_rows)\n"
            "TUNING_PARQUET = METRICS_DIR / f'tuning_{MODEL}.parquet'\n"
            "tuning.write_parquet(TUNING_PARQUET)\n"
            "print('mejores hiperparametros:', study.best_params)\n"
            "print('mejor mIoU (6 grupos):', round(study.best_value, 4), '| guardado en', TUNING_PARQUET)\n"
            "tuning.sort('miou_grouped', descending=True, nulls_last=True)"
        )
    else:
        tuning_md = (
            "## Ajuste fino del top-2 (Optuna)\n\n"
            "Como este modelo entra al top-2, se afinan sus hiperparametros con Optuna (>=30 "
            "trials) maximizando el mIoU de los 6 grupos HCAT. Cada trial reentrena la red sobre "
            "un subset reducido y usa pruning para cortar los trials que arrancan mal. El resumen "
            "se exporta a `reports/segmentation/metrics/tuning_<modelo>.parquet`, que el integrador "
            "`Avance4.Equipo17` levanta y muestra."
        )
        tuning_code = (
            "# Ajuste fino con Optuna sobre el top-2 (>=30 trials). Busca lr, weight_decay y\n"
            "# batch_size maximizando el mIoU de los 6 grupos HCAT. Cada trial reentrena la red\n"
            "# sobre un subset reducido, con pruning para abortar los trials malos.\n"
            "import optuna\n"
            "import polars as pl\n"
            "from ml.train.train_segmentation import run_training\n\n"
            "TUNE_TRIALS = 30\n"
            "TUNE_EPOCHS = 5\n"
            "TUNE_SUBSET = SUBSET if SUBSET else 150   # subset reducido: 30 trials viables en ~45-60 min\n"
            "# Artefactos temporales del tuning, aislados de los del modelo final (no los pisan ni\n"
            "# los recoge el integrador, que solo lee metrics/model_comparison_* y metrics/tuning_*).\n"
            "TUNE_DIR = SEG_DIR / 'tuning_tmp'\n"
            "TUNE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TUNE_TMP_COMPARISON = TUNE_DIR / f'model_comparison_avance4_{MODEL}.parquet'\n\n"
            "def _objective(trial):\n"
            "    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)\n"
            "    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)\n"
            "    bs = trial.suggest_categorical('batch_size', [4, 8, 16])\n\n"
            "    def _report(epoch, metrics):\n"
            "        trial.report(metrics['miou_grouped'], step=epoch)\n"
            "        if trial.should_prune():\n"
            "            raise optuna.TrialPruned()\n\n"
            "    res = run_training(\n"
            "        model=MODEL, epochs=TUNE_EPOCHS, batch_size=bs, lr=lr, weight_decay=wd,\n"
            "        target_size=TARGET_SIZE, subset=TUNE_SUBSET, device=DEVICE, root=PASTIS_ROOT,\n"
            "        mlflow_uri=MLFLOW_URI, num_workers=NUM_WORKERS,\n"
            "        output_dir=TUNE_DIR, comparison_path=TUNE_TMP_COMPARISON,\n"
            "        resume=False, on_epoch=_report,\n"
            "    )\n"
            "    return res['miou_grouped']\n\n"
            "study = optuna.create_study(\n"
            "    direction='maximize', study_name=f'tune-{MODEL}',\n"
            "    sampler=optuna.samplers.TPESampler(seed=42),\n"
            "    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),\n"
            ")\n"
            "study.optimize(_objective, n_trials=TUNE_TRIALS)\n\n"
            "# Resumen de todos los trials -> metrics/tuning_<modelo>.parquet (lo consume el Avance4).\n"
            "_rows = [{\n"
            "    'model': MODEL,\n"
            "    'trial': t.number,\n"
            "    'state': t.state.name,\n"
            "    'lr': t.params.get('lr'),\n"
            "    'weight_decay': t.params.get('weight_decay'),\n"
            "    'batch_size': t.params.get('batch_size', BATCH),\n"
            "    'miou_grouped': t.value,\n"
            "} for t in study.trials]\n"
            "tuning = pl.DataFrame(_rows)\n"
            "TUNING_PARQUET = METRICS_DIR / f'tuning_{MODEL}.parquet'\n"
            "tuning.write_parquet(TUNING_PARQUET)\n"
            "print('mejores hiperparametros:', study.best_params)\n"
            "print('mejor mIoU (6 grupos):', round(study.best_value, 4), '| guardado en', TUNING_PARQUET)\n"
            "tuning.sort('miou_grouped', descending=True, nulls_last=True)"
        )
    return [md(tuning_md), code(tuning_code)]


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Arquitectura: 'unet' o 'anysat'.")] = "unet",
    out: Annotated[str, typer.Option(help="Ruta de salida (default segun modelo).")] = "",
    num_workers: Annotated[int, typer.Option(help="Override de workers (-1 = default).")] = -1,
    batch: Annotated[int, typer.Option(help="Override de batch (-1 = default del modelo).")] = -1,
    epochs: Annotated[int, typer.Option(help="Numero de epocas.")] = 30,
    target_size: Annotated[int, typer.Option(help="Resolucion espacial (AnySat: 64).")] = 256,
    subset: Annotated[int, typer.Option(help="Patches por split (0 = todos).")] = 0,
    suffix: Annotated[str, typer.Option(help="Sufijo de artefactos (correr en paralelo).")] = "",
) -> None:
    """Generate the dense segmentation notebook for one architecture.

    Args:
        model: ``unet`` or ``anysat``.
        out: Destination path of the ``.ipynb`` (if empty, the model default is used).
        num_workers: DataLoader worker override.
        batch: Batch override.
        epochs: Number of epochs.
        suffix: Artifact suffix to avoid overwriting another parallel run.
    """
    if model not in _OUT_BY_MODEL:
        raise typer.BadParameter("`--model` debe ser 'unet' o 'anysat'.")
    out_path = Path(out) if out else _OUT_BY_MODEL[model]
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells(
        model,
        num_workers=num_workers,
        batch=batch,
        epochs=epochs,
        target_size=target_size,
        subset=subset,
        suffix=suffix,
    )
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out_path} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
