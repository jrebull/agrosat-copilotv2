"""Resolution of Avance 4 segmentation figures.

Centralizes the logic of locating a model's figure by type
(``curves``, ``per_class_iou``, ``confusion``, ``samples``), accepting the
exact name, suffixed variants (``anysat`` -> ``anysat_fast``) and a fallback
map for the DeepLab/TSViT figures published in ``paper/figures/us-025/`` with
their own names. Keeping this outside the notebook avoids repeating paths and
the hardcoded map in each gallery cell.
"""

from __future__ import annotations

from pathlib import Path

# Figure types per model and their readable label (presentation order).
FIGURE_TYPES: tuple[tuple[str, str], ...] = (
    ("curves", "Curvas de entrenamiento"),
    ("per_class_iou", "IoU por clase"),
    ("confusion", "Matriz de confusion"),
    ("samples", "RGB / verdad / prediccion"),
)

# Fallback us-025: DeepLab/TSViT figures with their real names, which
# complement (not replace) the figures from the team's shared Drive.
_US025_DIR = Path("paper/figures/us-025")
_US025_MAP: dict[tuple[str, str], str] = {
    ("confusion", "deeplabv3plus"): "deeplab_confusion_semantic18.png",
    ("samples", "deeplabv3plus"): "deeplab_semantic18_pred_example_0.png",
    ("confusion", "tsvit"): "tsvit_confusion_tsvit-pheno.png",
    ("samples", "tsvit"): "tsvit_pred_example_0.png",
}


def find_figure(figures_dir: Path, key: str, model: str) -> Path | None:
    """Locate the ``key`` figure of the ``model`` or ``None`` if it does not exist.

    Args:
        figures_dir: Main segmentation figures directory.
        key: Figure type (``"confusion"``, ``"samples"``, ...).
        model: Model slug (``"unet"``, ``"tsvit"``, ...).

    Returns:
        Path to the found figure (exact name, suffixed variant or
        us-025 fallback), or ``None`` if none exists.
    """
    exact = figures_dir / f"{key}_{model}.png"
    if exact.exists():
        return exact
    variants = sorted(figures_dir.glob(f"{key}_{model}_*.png"))
    if variants:
        return variants[0]
    fallback_name = _US025_MAP.get((key, model))
    if fallback_name:
        fallback = _US025_DIR / fallback_name
        if fallback.exists():
            return fallback
    return None


def show_model_figs(figures_dir: Path, model: str) -> bool:
    """Show in the notebook the available figures of a model.

    Iterates the four figure types (``curves``, ``per_class_iou``,
    ``confusion``, ``samples``), resolves each one with ``find_figure`` and
    renders it with a readable header. Intended to be called from a cell of
    the integrator notebook (the logic lives here, not inline in the ``.ipynb``).

    Args:
        figures_dir: Main segmentation figures directory.
        model: Model slug (``"unet"``, ``"anysat"``, ``"tsvit"``, ...).

    Returns:
        ``True`` if at least one figure was shown; ``False`` if the model
        still has no exported figures.
    """
    from IPython.display import Image, Markdown, display

    shown = False
    for key, label in FIGURE_TYPES:
        fpath = find_figure(figures_dir, key, model)
        if fpath is not None:
            display(Markdown(f"**{label}**"))
            display(Image(filename=str(fpath)))
            shown = True
    if not shown:
        display(
            Markdown(f"_Aun no hay figuras para `{model}` (correr su notebook de entrenamiento)._")
        )
    return shown
