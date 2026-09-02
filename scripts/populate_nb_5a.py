"""Insert the evaluation Section (real local checkpoints) into
``notebooks/segmentation/5a_deeplabv3plus.ipynb`` and update
parameters/conclusions.

Does not re-train: it loads the DeepLabv3+ ``best.pt`` checkpoints trained
locally (18-class variant and 6-group HCAT variant) and generates the visual
analysis (confusion matrix, per-class IoU/F1, RGB|GT|pred predictions).
DeepLabv3+ is a 2D model: the dataset collapses the time series by median
before the forward pass. Permanent notebook-population operative tool
(reproducible via papermill).
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/segmentation/5a_deeplabv3plus.ipynb")


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# 1. Add checkpoint_dir + eval_max_patches to the parameters cell (1).
param_src = "".join(cells[1]["source"])
if "checkpoint_dir" not in param_src:
    param_src = param_src.replace(
        'figures_dir = "paper/figures/us-025"',
        'figures_dir = "paper/figures/us-025"\n'
        'checkpoint_dir = "checkpoints/segmentation"\n'
        "eval_max_patches = None  # None = fold val completo; int para smoke",
    )
    cells[1]["source"] = param_src.splitlines(keepends=True)

# 2. Skip-if-trained shortcut in run_training (cell 4): if the local best.pt
#    already exists, read best_metrics and do not re-train.
cell4 = "".join(cells[4]["source"])
if "skip-if-trained" not in cell4 and "Checkpoint entrenado ya presente" not in cell4:
    shortcut = """    # Atajo de reproducibilidad (skip-if-trained): si ya existe el checkpoint
    # local entrenado (best.pt), no se re-entrena. Se leen sus metricas del
    # mejor epoch y se reporta como corrida ya completada. La Seccion de
    # evaluacion analiza ese mismo checkpoint en detalle.
    if not run_full:
        sub = "deeplab-18" if target == "semantic18" else "deeplab-6"
        ckpt_real = Path(checkpoint_dir) / sub / "best.pt"
        ckpt_abs = ckpt_real if ckpt_real.is_absolute() else REPO / ckpt_real
        if ckpt_abs.is_file():
            import torch

            ck = torch.load(ckpt_abs, map_location="cpu", weights_only=False)
            bm = ck.get("best_metrics", {})
            display(Markdown(
                f"Checkpoint entrenado ya presente "
                f"(`{ckpt_abs.relative_to(REPO)}`, mejor epoch "
                f"{bm.get('best_epoch')}): se omite el re-entrenamiento y se "
                f"reportan sus metricas. La invocacion CLI documentada es:"
            ))
            cmd_doc = (
                f"python -m ml.train.train_segmentation --model {MODEL_KIND} "
                f"--epochs 15 --batch-size 8 --target {target} "
                f"--run-name {RUN_NAME}"
            )
            display(Markdown(f"`{cmd_doc}`"))
            return {
                "model": MODEL_KIND,
                "miou": float(bm["miou"]) if "miou" in bm else None,
                "f1_macro": float(bm["f1_macro"]) if "f1_macro" in bm else None,
                "pixel_acc": float(bm["pixel_acc"]) if "pixel_acc" in bm else None,
                "returncode": 0,
                "error": None,
            }

"""
    marker = "    cmd = ["
    assert marker in cell4, "could not find the start of cmd in run_training (5a)"
    cell4 = cell4.replace(marker, shortcut + marker, 1)
    cells[4]["source"] = cell4.splitlines(keepends=True)

# 3. Evaluation section (real local checkpoints). DeepLab is 2D:
#    collapse_time="median".
sec_md = _md(
    """## Seccion 3 - Evaluacion del modelo entrenado

Esta seccion analiza el **resultado** del entrenamiento de DeepLabv3+ ya
completado en local: carga su checkpoint `best.pt`, lo evalua sobre el **fold de
validacion completo** (fold 4, 482 parches no vistos) y genera el material de
analisis: matriz de confusion, IoU y F1 por clase, y predicciones visuales
pixel a pixel.

DeepLabv3+ es un segmentador **2D**: a diferencia de TSViT no consume la serie
temporal completa, sino una composicion de la temporada (el dataset colapsa las
fechas por mediana antes del forward). El notebook **no re-entrena** aqui:
reutiliza los pesos del mejor epoch. La logica vive en
`ml/eval/segmentation_inference.py` (`evaluate_checkpoint`)."""
)

sec_setup = _code(
    """# Carga del checkpoint y evaluacion sobre el fold de validacion completo.
import numpy as np

from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
from ml.eval.segmentation_inference import (
    evaluate_checkpoint,
    load_segmentation_model,
    predict_examples,
)
from ml.features.phenology_class_prototypes import load_class_names

# El dataset remapea la clase original PASTIS `cid` (1..18) al indice de
# entrenamiento `cid-1` (0..17); para 6 grupos HCAT el indice ya es directo.
_ORIG_NAMES = load_class_names()
CLASS_NAMES = {c: _ORIG_NAMES.get(c + 1, f"clase_{c}") for c in range(18)}

NUM_CLASSES = 18 if target == "semantic18" else 6
SUBDIR = "deeplab-18" if target == "semantic18" else "deeplab-6"
ckpt_path = (REPO / checkpoint_dir / SUBDIR / "best.pt")

# DeepLabv3+ es 2D: collapse_time="median" produce (10, H, W).
eval_ds = None
if ckpt_path.is_file():
    try:
        eval_ds = PASTISSegmentationDataset(
            folds=(4,), collapse_time="median", target=target
        )
        display(Markdown(
            f"Fold de validacion: **{len(eval_ds)} parches** - checkpoint "
            f"`{ckpt_path.relative_to(REPO)}`."
        ))
    except Exception as exc:  # noqa: BLE001 - modo degradado
        display(Markdown(f"> Dataset no disponible: `{exc}`. Modo degradado."))
else:
    display(Markdown(
        f"> Checkpoint ausente (`{ckpt_path.relative_to(REPO)}`). Se omite la "
        "evaluacion (modo degradado)."
    ))"""
)

sec_eval = _code(
    """# Evalua el checkpoint: acumula la matriz de confusion sobre todo el fold y
# deriva mIoU, F1-macro, pixel_acc, balanced accuracy y Cohen kappa.
eval_metrics: dict | None = None
eval_cm: np.ndarray | None = None

if eval_ds is not None:
    model = load_segmentation_model(
        ckpt_path, model_kind="deeplabv3plus", num_classes=NUM_CLASSES,
        device=device,
    )
    eval_metrics, eval_cm = evaluate_checkpoint(
        model, eval_ds, model_kind="deeplabv3plus", num_classes=NUM_CLASSES,
        max_patches=eval_max_patches,
    )
    del model

    metrics_df = pl.DataFrame([{
        "modelo": f"deeplabv3plus-{target}",
        "mIoU": round(float(eval_metrics["miou"]), 4),
        "F1_macro": round(float(eval_metrics["f1_macro"]), 4),
        "pixel_acc": round(float(eval_metrics["pixel_acc"]), 4),
        "balanced_acc": round(float(eval_metrics["balanced_acc"]), 4),
        "cohen_kappa": round(float(eval_metrics["cohen_kappa"]), 4),
    }])
    display(metrics_df)
else:
    display(Markdown("> Sin dataset/checkpoint; no hay metricas que mostrar."))"""
)

sec_perclass_md = _md(
    """### IoU y F1 por clase

El mIoU global promedia las clases, pero PASTIS esta muy desbalanceado. El
desglose por clase revela en que cultivos DeepLabv3+ acierta y en cuales se
confunde (tipicamente clases minoritarias o de aspecto espectral parecido en la
composicion de la temporada)."""
)

sec_perclass = _code(
    """# IoU y F1 por clase del checkpoint evaluado.
if eval_metrics is not None:
    names = CLASS_NAMES if target == "semantic18" else None
    rows = []
    for c in range(NUM_CLASSES):
        cname = names.get(c, f"clase_{c}") if names else f"grupo_{c}"
        rows.append({
            "clase": cname,
            "IoU": round(float(eval_metrics["per_class_iou"][c]), 4),
            "F1": round(float(eval_metrics["per_class_f1"][c]), 4),
        })
    per_class_df = pl.DataFrame(rows).sort("IoU", descending=True)
    display(per_class_df)

    out_csv = FIGURES / f"deeplab_{target}_per_class_metrics.csv"
    per_class_df.write_csv(out_csv)
    display(Markdown(f"Tabla por clase guardada en `{out_csv.relative_to(REPO)}`."))
else:
    display(Markdown("> Sin metricas por clase (modo degradado)."))"""
)

sec_cm_md = _md(
    """### Matriz de confusion

Muestra, fila a fila, a que clase asigna realmente DeepLabv3+ los pixeles de
cada clase verdadera (normalizada por fila). La diagonal es el acierto; las
celdas fuera de diagonal son las confusiones sistematicas entre cultivos."""
)

sec_cm = _code(
    """# Matriz de confusion normalizada por fila.
if eval_cm is not None:
    names = CLASS_NAMES if target == "semantic18" else None
    row_sums = eval_cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        eval_cm, row_sums, out=np.zeros_like(eval_cm, dtype=float), where=row_sums != 0
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Matriz de confusion (normalizada) - deeplabv3plus-{target}")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdad")
    if names:
        labels = [names.get(c, str(c)) for c in range(NUM_CLASSES)]
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = FIGURES / f"deeplab_confusion_{target}.png"
    fig.savefig(out_path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    display(Markdown(f"Guardada en `{out_path.relative_to(REPO)}`."))
else:
    display(Markdown("> Sin matriz de confusion (modo degradado)."))"""
)

sec_pred_md = _md(
    """### Predicciones visuales: Input RGB | Verdad | Prediccion

La prueba mas directa de que el modelo aprendio: comparar, para parches
concretos del fold de validacion, la imagen RGB, la mascara real de cultivos y
la prediccion de DeepLabv3+."""
)

sec_pred = _code(
    """# Predicciones del modelo sobre algunos parches de validacion.
if eval_ds is not None and ckpt_path.is_file():
    model = load_segmentation_model(
        ckpt_path, model_kind="deeplabv3plus", num_classes=NUM_CLASSES,
        device=device,
    )
    n_show = min(4, len(eval_ds))
    step = max(1, len(eval_ds) // n_show)
    indices = list(range(0, step * n_show, step))[:n_show]
    figs = predict_examples(
        model, eval_ds, model_kind="deeplabv3plus", indices=indices,
        num_classes=NUM_CLASSES,
    )
    for j, fig in enumerate(figs):
        out_path = FIGURES / f"deeplab_{target}_pred_example_{indices[j]}.png"
        fig.savefig(out_path, bbox_inches="tight")
        display(fig)
        plt.close(fig)
    display(Markdown(
        f"Se guardaron {len(figs)} figuras de prediccion (parches {indices})."
    ))
    del model
else:
    display(Markdown("> Sin checkpoint; no se generan predicciones visuales."))"""
)

# 4. Insert before the Conclusions cell.
concl_idx = next(
    i
    for i, c in enumerate(cells)
    if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Conclusiones")
)
new_cells = [
    sec_md,
    sec_setup,
    sec_eval,
    sec_perclass_md,
    sec_perclass,
    sec_cm_md,
    sec_cm,
    sec_pred_md,
    sec_pred,
]
# Spelling-robust idempotency: detected by an English identifier
# (stable to accents) present in the evaluation setup cell.
already = any("ml.eval.segmentation_inference" in "".join(c["source"]) for c in cells)
if not already:
    cells[concl_idx:concl_idx] = new_cells
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"5a poblada: {len(cells)} celdas, Seccion de evaluacion insertada.")
else:
    print("5a: la Seccion de evaluacion ya estaba presente; no se modifica.")
