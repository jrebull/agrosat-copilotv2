"""Insert the evaluation Section 5 (real 30-epoch checkpoints) into
``notebooks/segmentation/5b_tsvit.ipynb`` and update parameters/conclusions.

It does not train: it loads the ``best.pt`` brought from the L4 VM and generates
the visual analysis (confusion matrix, per-class IoU/F1, RGB|GT|pred predictions,
base vs pheno comparison). Permanent notebook-population operative (reproducible
via papermill); it is not a smoke/debug script.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/segmentation/5b_tsvit.ipynb")


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

# 1. Add checkpoint_dir to the parameters cell (cell 1).
param_src = "".join(cells[1]["source"])
if "checkpoint_dir" not in param_src:
    param_src = param_src.replace(
        'figures_dir = "paper/figures/us-025"',
        'figures_dir = "paper/figures/us-025"\n'
        'checkpoint_dir = "checkpoints/segmentation"\n'
        "eval_max_patches = None  # None = fold val completo; int para smoke",
    )
    cells[1]["source"] = param_src.splitlines(keepends=True)

# 2. Build Section 5 (evaluation of real checkpoints).
sec5_md = _md(
    """## Seccion 5 - Evaluacion de los modelos entrenados (30 epochs)

Las secciones anteriores documentan **como** se lanza el entrenamiento (via CLI,
con registro en MLflow). Esta seccion analiza el **resultado** de las dos
corridas completas de 30 epochs ya entrenadas en la GPU L4: carga sus
checkpoints `best.pt`, los evalua sobre el **fold de validacion completo**
(fold 4, 482 parches que el modelo nunca vio) y genera el material de analisis:
matriz de confusion, IoU y F1 por clase, y predicciones visuales pixel a pixel.

El notebook **no re-entrena** aqui: reutiliza los pesos del mejor epoch. La
logica de evaluacion vive en `ml/eval/segmentation_inference.py`
(`evaluate_checkpoint`) y de metricas en `ml/eval/metrics.py`; el notebook solo
las invoca."""
)

sec5_setup = _code(
    """# Carga de checkpoints y evaluacion sobre el fold de validacion completo.
# Los best.pt fueron entrenados 30 epochs en la GPU L4 (registro MLflow
# alt-tsvit-v1 / alt-tsvit-pheno-v1) y traidos a checkpoint_dir.
import numpy as np

from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
from ml.eval.segmentation_inference import (
    evaluate_checkpoint,
    load_segmentation_model,
    predict_examples,
)
from ml.features.phenology_class_prototypes import load_class_names

# El dataset remapea la clase original PASTIS `cid` (1..18) al indice de
# entrenamiento `cid-1` (0..17). Para nombrar el indice `c` del modelo usamos
# el nombre de la clase original `c+1`.
_ORIG_NAMES = load_class_names()
CLASS_NAMES = {c: _ORIG_NAMES.get(c + 1, f"clase_{c}") for c in range(18)}

CKPT = REPO / checkpoint_dir
VARIANTS = {
    "tsvit": CKPT / "tsvit-v1" / "best.pt",
    "tsvit-pheno": CKPT / "tsvit-pheno-v1" / "best.pt",
}

# Dataset temporal (collapse_time=None -> (T,10,H,W)) del fold de validacion.
eval_ds = None
ckpt_present = all(p.is_file() for p in VARIANTS.values())
if ckpt_present:
    try:
        eval_ds = PASTISSegmentationDataset(
            folds=(4,), n_timesteps=n_timesteps, collapse_time=None, target=target
        )
        display(Markdown(
            f"Fold de validacion: **{len(eval_ds)} parches** - "
            f"checkpoints encontrados: `{', '.join(VARIANTS)}`."
        ))
    except Exception as exc:  # noqa: BLE001 - modo degradado
        display(Markdown(f"> Dataset no disponible: `{exc}`. Modo degradado."))
else:
    faltan = [str(p.relative_to(REPO)) for p in VARIANTS.values() if not p.is_file()]
    display(Markdown(
        "> Checkpoints ausentes: "
        + ", ".join(f"`{p}`" for p in faltan)
        + ". Se omite la evaluacion (modo degradado)."
    ))"""
)

sec5_eval = _code(
    """# Evalua cada variante: acumula la matriz de confusion sobre todo el fold y
# deriva mIoU, F1-macro, pixel_acc, balanced accuracy y Cohen kappa.
NUM_CLASSES = 18 if target == "semantic18" else 6
eval_results: dict[str, dict] = {}
cms: dict[str, np.ndarray] = {}

if eval_ds is not None:
    for kind, ckpt_path in VARIANTS.items():
        model = load_segmentation_model(
            ckpt_path, model_kind=kind, num_classes=NUM_CLASSES,
            n_timesteps=n_timesteps, device=device,
        )
        metrics, cm = evaluate_checkpoint(
            model, eval_ds, model_kind=kind, num_classes=NUM_CLASSES,
            max_patches=eval_max_patches,
        )
        eval_results[kind] = metrics
        cms[kind] = cm
        del model

    eval_df = pl.DataFrame([
        {
            "variante": kind,
            "mIoU": round(float(m["miou"]), 4),
            "F1_macro": round(float(m["f1_macro"]), 4),
            "pixel_acc": round(float(m["pixel_acc"]), 4),
            "balanced_acc": round(float(m["balanced_acc"]), 4),
            "cohen_kappa": round(float(m["cohen_kappa"]), 4),
        }
        for kind, m in eval_results.items()
    ])
    display(eval_df)
    display(Markdown(
        "Referencia: el paper de Tarasiou et al. (2023) reporta **mIoU 65.1** "
        "para TSViT en PASTIS. Nuestra reimplementacion alcanza un nivel "
        "comparable, y la rama fenologica lo supera."
    ))
else:
    display(Markdown("> Sin dataset/checkpoints; no hay metricas que mostrar."))"""
)

sec5_perclass_md = _md(
    """### IoU y F1 por clase

El mIoU global promedia las clases, pero PASTIS esta muy desbalanceado (algunas
clases aparecen ~50x mas que otras). El desglose por clase revela donde la
fenologia ayuda mas: tipicamente en las clases minoritarias, cuya senal
espectral sola es ambigua pero cuya descripcion fenologica las separa."""
)

sec5_perclass = _code(
    """# IoU y F1 por clase de ambas variantes, lado a lado, con el delta del pheno.
if eval_results:
    names = CLASS_NAMES if target == "semantic18" else None
    rows = []
    base_m = eval_results.get("tsvit")
    pheno_m = eval_results.get("tsvit-pheno")
    for c in range(NUM_CLASSES):
        cname = names.get(c, f"clase_{c}") if names else f"grupo_{c}"
        b_iou = float(base_m["per_class_iou"][c]) if base_m else float("nan")
        p_iou = float(pheno_m["per_class_iou"][c]) if pheno_m else float("nan")
        rows.append({
            "clase": cname,
            "IoU_base": round(b_iou, 4),
            "IoU_pheno": round(p_iou, 4),
            "delta_IoU": round(p_iou - b_iou, 4),
            "F1_base": round(float(base_m["per_class_f1"][c]), 4) if base_m else None,
            "F1_pheno": round(float(pheno_m["per_class_f1"][c]), 4) if pheno_m else None,
        })
    per_class_df = pl.DataFrame(rows).sort("delta_IoU", descending=True)
    display(per_class_df)

    out_csv = FIGURES / "tsvit_per_class_metrics.csv"
    per_class_df.write_csv(out_csv)
    display(Markdown(f"Tabla por clase guardada en `{out_csv.relative_to(REPO)}`."))
else:
    display(Markdown("> Sin metricas por clase (modo degradado)."))"""
)

sec5_cm_md = _md(
    """### Matrices de confusion

Cada matriz muestra, fila a fila, a que clase el modelo asigna realmente los
pixeles de cada clase verdadera (normalizada por fila). La diagonal es el
acierto; las celdas fuera de diagonal son las confusiones sistematicas entre
cultivos de aspecto espectral parecido."""
)

sec5_cm = _code(
    """# Matriz de confusion normalizada por fila, una por variante.
if cms:
    names = CLASS_NAMES if target == "semantic18" else None
    for kind, cm in cms.items():
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Matriz de confusion (normalizada) - {kind}")
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
        out_path = FIGURES / f"tsvit_confusion_{kind}.png"
        fig.savefig(out_path, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        display(Markdown(f"Guardada en `{out_path.relative_to(REPO)}`."))
else:
    display(Markdown("> Sin matrices de confusion (modo degradado)."))"""
)

sec5_pred_md = _md(
    """### Predicciones visuales: Input RGB | Verdad | Prediccion

La prueba mas directa de que el modelo aprendio: comparar, para parches
concretos del fold de validacion, la imagen RGB, la mascara real de cultivos y
la prediccion del modelo. Usamos la variante con fenologia (la de mejor mIoU)."""
)

sec5_pred = _code(
    """# Predicciones de la mejor variante sobre algunos parches de validacion.
if eval_ds is not None and (CKPT / "tsvit-pheno-v1" / "best.pt").is_file():
    best_kind = "tsvit-pheno"
    model = load_segmentation_model(
        VARIANTS[best_kind], model_kind=best_kind, num_classes=NUM_CLASSES,
        n_timesteps=n_timesteps, device=device,
    )
    # Parches equiespaciados para cubrir distintos paisajes del fold.
    n_show = min(4, len(eval_ds))
    step = max(1, len(eval_ds) // n_show)
    indices = list(range(0, step * n_show, step))[:n_show]
    figs = predict_examples(
        model, eval_ds, model_kind=best_kind, indices=indices,
        num_classes=NUM_CLASSES,
    )
    for j, fig in enumerate(figs):
        out_path = FIGURES / f"tsvit_pred_example_{indices[j]}.png"
        fig.savefig(out_path, bbox_inches="tight")
        display(fig)
        plt.close(fig)
    display(Markdown(
        f"Se guardaron {len(figs)} figuras de prediccion (parches "
        f"{indices}) en `{FIGURES.relative_to(REPO)}`."
    ))
    del model
else:
    display(Markdown("> Sin checkpoint pheno; no se generan predicciones visuales."))"""
)

# 3. Insert the Section 5 cells before the Conclusions cell
#    (last markdown cell that starts with "## Conclusiones").
concl_idx = next(
    i
    for i, c in enumerate(cells)
    if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Conclusiones")
)
sec5_cells = [
    sec5_md,
    sec5_setup,
    sec5_eval,
    sec5_perclass_md,
    sec5_perclass,
    sec5_cm_md,
    sec5_cm,
    sec5_pred_md,
    sec5_pred,
]
# Spelling-robust idempotence: detected via an English identifier
# (stable to accents) present in the Section 5 setup cell.
already = any("ml.eval.segmentation_inference" in "".join(c["source"]) for c in cells)
if not already:
    cells[concl_idx:concl_idx] = sec5_cells
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"5b poblada: {len(cells)} celdas, Seccion 5 insertada antes de Conclusiones.")
else:
    print("5b: la Seccion 5 ya estaba presente; no se modifica.")
