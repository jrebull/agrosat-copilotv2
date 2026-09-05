"""Reproducible paper figures for AgroSatCopilot (US-070, EPIC 11).

Single matplotlib template (scientific CVPR/ISPRS style: serif font, 300 DPI,
column-width sizing) plus the SVG+PNG exporter used by every paper figure. The
module follows two DRY principles:

1. **Style lives once** (:func:`set_paper_style`): rcParams + a fixed seed so any
   figure is byte-reproducible.
2. **Plots come from real artifacts**: figures are either *recomposed* from the
   numeric CSV/JSON under ``reports/**`` (barplots, sweep curves, transfer
   deltas) or *promoted* (copied + re-exported as SVG/PNG) from an already
   generated PNG (UMAP, confusion matrices, training curves, spatial residuals).
   There is **no fabricated data**: a figure whose source is missing is skipped
   and recorded in ``docs/blockers/epic11-notas.md``.

Captions/labels carry the project's factual corrections: AlphaEarth =
``SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1 (not "v2.1"), SegFormer = B0 RGB 3-band,
AnySat substitutes the never-trained Swin-UNETR, Gemini 2.5 Pro is a frozen
reasoner (Be My Eyes pattern).

**Bilingual output**: every figure is emitted twice, once per language in
:data:`LANGS`. The **English** render is the canonical base file
(``<stem>.png`` / ``<stem>.svg``) for the English paper; the **Spanish** render
carries the ``_es`` suffix (``<stem>_es.png`` / ``<stem>_es.svg``) for the
Spanish paper. Every visible string (titles, axis labels, legends, annotations,
subtitles) comes from a per-figure ``dict[Lang, dict[str, str]]`` so no text is
duplicated across the plotting logic; only the strings differ between languages,
never the numbers or the plotting code. Identifiers and docstrings are in
English per the language policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "FIGURES_DIR",
    "LANGS",
    "PAPER_SEED",
    "REPORTS_DIR",
    "Lang",
    "build_all_figures",
    "fig_benchmark_barplot",
    "fig_farslip_band_ablation",
    "fig_farslip_sweep_curve",
    "fig_llm_benchmark_barplot",
    "fig_transfer_catalonia",
    "fig_tsvit_full_config_delta",
    "promote_png",
    "save_fig_svg_png",
    "set_paper_style",
    "stem_for_lang",
]

#: Fixed seed for any stochastic step (kept for reproducibility even though the
#: current figures are deterministic reads).
PAPER_SEED = 17

#: Repository ``reports/`` root (figure data sources).
REPORTS_DIR = Path("reports")

#: Default output directory for paper figures.
FIGURES_DIR = Path("paper/figures/us-070")

#: Supported figure languages. ``"en"`` is canonical (base file), ``"es"`` gets
#: the ``_es`` suffix. English first so the base file is written before the
#: suffixed variant.
Lang = Literal["en", "es"]

#: Languages every figure is rendered in, English (canonical base) first.
LANGS: tuple[Lang, ...] = ("en", "es")


def stem_for_lang(stem: str, lang: Lang) -> str:
    """Return the language-suffixed file stem for a figure.

    The English render keeps the bare ``stem`` (canonical base file for the
    English paper); the Spanish render gets the ``_es`` suffix.

    Args:
        stem: Base (English) file stem, without extension.
        lang: Target language.

    Returns:
        ``stem`` for English, ``f"{stem}_es"`` for Spanish.
    """
    return stem if lang == "en" else f"{stem}_es"


def set_paper_style() -> None:
    """Apply the scientific (CVPR/ISPRS) matplotlib style and fix the seed.

    Sets serif fonts, 300 DPI, tight column-width defaults and a deterministic
    seed for NumPy. Idempotent; safe to call at the top of every figure.

    It deliberately does NOT set ``figure.constrained_layout.use``. That rcParam is layout
    policy, not typography: once set it applied to every figure drawn later in the same
    process, and ``tight_layout`` -called at 63 sites across ``ml/``- raises on a figure that
    already has a constrained layout engine. The figures of this module ask for the engine one
    by one, where it belongs.
    """
    np.random.seed(PAPER_SEED)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig_svg_png(
    fig: plt.Figure, stem: str, *, out_dir: Path = FIGURES_DIR, lang: Lang = "en"
) -> dict[str, Path]:
    """Export a figure as both SVG (vector) and PNG (300 DPI raster).

    The English render writes the canonical base file ``<stem>.{svg,png}``; the
    Spanish render writes ``<stem>_es.{svg,png}`` (see :func:`stem_for_lang`).

    Args:
        fig: Figure to export.
        stem: Base (English) file stem, without extension.
        out_dir: Destination directory (created if missing).
        lang: Language of the render; controls the ``_es`` suffix.

    Returns:
        Mapping ``{"svg": path, "png": path}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = stem_for_lang(stem, lang)
    svg = out_dir / f"{out_stem}.svg"
    png = out_dir / f"{out_stem}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("paper_figure_saved", stem=out_stem, lang=lang, svg=str(svg), png=str(png))
    return {"svg": svg, "png": png}


def promote_png(
    source_png: Path, stem: str, *, out_dir: Path = FIGURES_DIR, lang: Lang = "en"
) -> dict[str, Path] | None:
    """Promote an already generated PNG into the paper figure set.

    Re-renders the existing raster through the paper style frame and exports it
    as SVG+PNG so the paper figure carries a consistent border/DPI. Returns
    ``None`` if the source artifact does not exist (a blocked figure) -- the plot
    is never fabricated.

    The promoted PNG carries no matplotlib-drawn text of its own (whatever text
    it has is baked into the raster), so the render is language-agnostic; ``lang``
    only selects the output file suffix so both the base and ``_es`` variants
    exist for the paper's language-specific figure directories.

    Args:
        source_png: Existing PNG under ``reports/**``.
        stem: Output (English base) file stem.
        out_dir: Destination directory.
        lang: Language of the render; controls the ``_es`` suffix.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not source_png.exists():
        logger.warning("paper_figure_source_missing", stem=stem, source=str(source_png))
        return None
    set_paper_style()
    img = mpimg.imread(source_png)
    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    ax.imshow(img)
    ax.axis("off")
    ax.grid(False)
    return save_fig_svg_png(fig, stem, out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# F7-seg -- benchmark barplot recomposed from fold-5 metrics
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_benchmark_barplot`.
_STR_BENCHMARK: dict[Lang, dict[str, str]] = {
    "en": {
        "ylabel": "Score",
        "title": "EPIC 5 models (fold-5 re-score, US-030 harness, 18 classes)",
    },
    "es": {
        "ylabel": "Puntaje",
        "title": "Modelos EPIC 5 (re-score fold-5, harness US-030, 18 clases)",
    },
}


def fig_benchmark_barplot(
    metrics_csv: Path = REPORTS_DIR / "segmentation" / "metrics" / "model_comparison_fold5.csv",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Recompose the EPIC 5 benchmark barplot from fold-5 mIoU/F1.

    Reads the re-scored fold-5 metrics and draws a grouped bar chart (mIoU and
    F1-macro) per model, sorted by mIoU. TSViT-pheno is the top individual.

    Args:
        metrics_csv: ``model_comparison_fold5.csv`` source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not metrics_csv.exists():
        logger.warning("paper_figure_source_missing", stem="benchmark_barplot")
        return None
    txt = _STR_BENCHMARK[lang]
    df = pl.read_csv(metrics_csv).sort("miou", descending=True)
    set_paper_style()
    models = df["model"].to_list()
    miou = df["miou"].to_list()
    f1 = df["f1_macro"].to_list()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
    ax.bar(x - width / 2, miou, width, label="mIoU", color="#2c6fbb")
    ax.bar(x + width / 2, f1, width, label="F1-macro", color="#e08214")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.legend()
    return save_fig_svg_png(fig, "benchmark_barplot_fold5", out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# Fx -- FarSLIP parcel cardinality sweep curve
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_farslip_sweep_curve`.
_STR_FARSLIP_SWEEP: dict[Lang, dict[str, str]] = {
    "en": {
        "xlabel": "Number of classes",
        "ylabel": "Score",
        "title": "FarSLIP ablation: class cardinality (parcel level)",
    },
    "es": {
        "xlabel": "Numero de clases",
        "ylabel": "Puntaje",
        "title": "Ablacion FarSLIP: cardinalidad de clases (nivel parcela)",
    },
}


def fig_farslip_sweep_curve(
    sweep_csv: Path = REPORTS_DIR / "farslip" / "metrics" / "parcel_sweep.csv",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Recompose the FarSLIP parcel cardinality sweep curve.

    Reads ``parcel_sweep.csv`` and plots macro-F1 / macro-IoU vs the number of
    classes. Shows the difficulty growth as the label space widens.

    Args:
        sweep_csv: ``parcel_sweep.csv`` source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not sweep_csv.exists():
        logger.warning("paper_figure_source_missing", stem="farslip_sweep")
        return None
    txt = _STR_FARSLIP_SWEEP[lang]
    df = pl.read_csv(sweep_csv).sort("n_classes")
    set_paper_style()
    n = df["n_classes"].to_list()
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    ax.plot(n, df["macro_f1"].to_list(), "o-", label="macro-F1", color="#2c6fbb")
    ax.plot(n, df["macro_iou"].to_list(), "s--", label="macro-IoU", color="#e08214")
    ax.set_xlabel(txt["xlabel"])
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.legend()
    return save_fig_svg_png(fig, "farslip_sweep_curve", out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# F (transfer) -- FR -> Catalonia transfer delta
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_transfer_catalonia`.
_STR_TRANSFER: dict[Lang, dict[str, str]] = {
    "en": {
        "zero_shot": "zero-shot",
        "few_shot": "few-shot (k-shot FT)",
        "ylabel": "Score",
        "title": "FR->Catalonia transfer (Sen4AgriNet, US-075)",
    },
    "es": {
        "zero_shot": "zero-shot",
        "few_shot": "few-shot (ajuste k-shot)",
        "ylabel": "Puntaje",
        "title": "Transferencia FR->Cataluna (Sen4AgriNet, US-075)",
    },
}


def fig_transfer_catalonia(
    transfer_json: Path = REPORTS_DIR / "segmentation" / "sen4agrinet_transfer_result.json",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Plot the FR->Catalonia transfer delta (zero-shot vs few-shot).

    Reads ``sen4agrinet_transfer_result.json`` (US-075) and draws the zero-shot
    vs few-shot mIoU/F1/pixel-accuracy bars, evidencing the limited spatial
    transferability that few-shot fine-tuning recovers.

    Args:
        transfer_json: Sen4AgriNet transfer result source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not transfer_json.exists():
        logger.warning("paper_figure_source_missing", stem="transfer_catalonia")
        return None
    txt = _STR_TRANSFER[lang]
    data = json.loads(transfer_json.read_text(encoding="utf-8"))
    zs = data["zero_shot_metrics"]
    fs = data["few_shot_metrics"]
    set_paper_style()
    labels = ["mIoU", "F1-macro", "Pix-acc"]
    zero = [zs["miou"], zs["f1_macro"], zs["pixel_accuracy"]]
    few = [fs["miou"], fs["f1_macro"], fs["pixel_accuracy"]]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 3.2), layout="constrained")
    ax.bar(x - width / 2, zero, width, label=txt["zero_shot"], color="#9e9e9e")
    ax.bar(x + width / 2, few, width, label=txt["few_shot"], color="#2c6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.legend()
    return save_fig_svg_png(fig, "transfer_fr_catalonia", out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# Fx -- FarSLIP band ablation: faithful FarSLIP vs AlphaEarth separability
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_farslip_band_ablation`. The source
#: attribution line shares its data-provenance clause across languages (only the
#: "Source:" lead word is translated) so the numbers/paths stay identical.
_STR_FARSLIP_ABLATION: dict[Lang, dict[str, str]] = {
    "en": {
        "ylabel_f1": "F1-macro (KMeans, +/- std)",
        "title_f1": "Supervised separability",
        "ylabel_sil": "Silhouette (unsupervised)",
        "title_sil": "Cluster cohesion",
        "suptitle": ("FarSLIP-faithful vs AlphaEarth ablation (3 band variants pending, B-070-5)"),
        "source": (
            "AlphaEarth SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0) | FarSLIP arXiv:2511.14901"
        ),
    },
    "es": {
        "ylabel_f1": "F1-macro (KMeans, +/- std)",
        "title_f1": "Separabilidad supervisada",
        "ylabel_sil": "Silhouette (no supervisada)",
        "title_sil": "Cohesion de clusters",
        "suptitle": (
            "Ablacion FarSLIP fiel vs AlphaEarth (3 variantes de banda pendientes, B-070-5)"
        ),
        "source": (
            "AlphaEarth SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0) | FarSLIP arXiv:2511.14901"
        ),
    },
}


def fig_farslip_band_ablation(
    faithful_csv: Path = REPORTS_DIR
    / "farslip"
    / "metrics"
    / "us037_farslip_fiel_vs_alphaearth.csv",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Plot the real FarSLIP-faithful vs AlphaEarth band-ablation evidence (B-070-5).

    Reads ``us037_farslip_fiel_vs_alphaearth.csv`` and draws the grouped bars
    (KMeans F1-macro mean with std error bars + silhouette) for the two embedding
    spaces that ARE materialized: FarSLIP-faithful v2 (768-dim, 4-band phenology
    distillation) vs AlphaEarth 2019 (64-dim). The 3 explicit band variants
    (rgb / nir-rgb false-colour / 4band-pheno) require per-variant H100
    re-extraction and are annotated as pending -- never fabricated.

    AlphaEarth = GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0). FarSLIP =
    Li et al., "FarSLIP" (arXiv:2511.14901). Source CSV under reports/farslip/.

    Args:
        faithful_csv: FarSLIP-faithful vs AlphaEarth separability source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not faithful_csv.exists():
        logger.warning("paper_figure_source_missing", stem="farslip_band_ablation")
        return None
    txt = _STR_FARSLIP_ABLATION[lang]
    df = pl.read_csv(faithful_csv)
    set_paper_style()
    spaces = df["space"].to_list()
    f1 = df["f1_macro_mean"].to_list()
    f1_std = df["f1_macro_std"].to_list()
    sil = df["silhouette"].to_list()
    x = np.arange(len(spaces))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.2), layout="constrained")
    ax1.bar(x, f1, yerr=f1_std, capsize=4, color=["#7b3294", "#2c6fbb"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(spaces, rotation=20, ha="right")
    ax1.set_ylabel(txt["ylabel_f1"])
    ax1.set_title(txt["title_f1"])
    ax2.bar(x, sil, color=["#7b3294", "#2c6fbb"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(spaces, rotation=20, ha="right")
    ax2.set_ylabel(txt["ylabel_sil"])
    ax2.set_title(txt["title_sil"])
    fig.suptitle(txt["suptitle"], fontsize=10)
    fig.text(0.5, 0.0, txt["source"], ha="center", fontsize=6, color="0.4")
    return save_fig_svg_png(fig, "farslip_band_ablation", out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# F4 -- TSViT base vs pheno full-config (full-M) delta from fold-5 metrics
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_tsvit_full_config_delta`.
_STR_TSVIT_DELTA: dict[Lang, dict[str, str]] = {
    "en": {
        "ylabel": "Score (fold-5)",
        "title": "TSViT full-config (full-M): base vs phenology branch",
        "note": ("Phenology delta ~0 in the supervised regime (saturation, plan v8), valid."),
    },
    "es": {
        "ylabel": "Puntaje (fold-5)",
        "title": "TSViT full-config (full-M): base vs rama fenologica",
        "note": ("Delta fenologico ~0 en supervisado (saturacion, plan v8), valido."),
    },
}


def fig_tsvit_full_config_delta(
    delta_csv: Path = REPORTS_DIR / "segmentation" / "metrics" / "tsvit_pheno_vs_base_fold5.csv",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Plot the real TSViT full-config (full-M) base vs pheno delta (B-070-2).

    Reads ``tsvit_pheno_vs_base_fold5.csv`` (tsvit-base-fullm 0.6789,
    tsvit-pheno-fullm 0.6756) and draws the mIoU / F1-macro / pixel-acc bars for
    both full-config variants. The phenology branch delta is ~0 in the supervised
    regime (saturation: dense labels already teach the temporal signatures), which
    is the documented, valid result -- the contrastive phenology branch pays off in
    the self-supervised FarSLIP zero-shot path, not here. This is the REAL curve
    available; the H100 full-config training curve re-run stays separate (B-070-2).

    Args:
        delta_csv: TSViT base-vs-pheno fold-5 full-M metrics source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not delta_csv.exists():
        logger.warning("paper_figure_source_missing", stem="tsvit_full_config_delta")
        return None
    txt = _STR_TSVIT_DELTA[lang]
    df = pl.read_csv(delta_csv)
    set_paper_style()
    labels = ["mIoU", "F1-macro", "Pix-acc"]
    cols = ["miou_fold5", "f1_macro_fold5", "pixel_acc_fold5"]
    models = df["modelo"].to_list()
    x = np.arange(len(labels))
    width = 0.38
    colors = ["#2c6fbb", "#e08214"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3), layout="constrained")
    for i, model in enumerate(models):
        vals = [float(df.filter(pl.col("modelo") == model)[c][0]) for c in cols]
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            vals,
            width,
            label=model.split(" ")[0],
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(txt["ylabel"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(txt["title"])
    ax.legend()
    fig.text(0.5, -0.02, txt["note"], ha="center", fontsize=6, color="0.4")
    return save_fig_svg_png(fig, "tsvit_full_config_delta", out_dir=out_dir, lang=lang)


# --------------------------------------------------------------------------- #
# F7-LLM -- LLM benchmark barplot recomposed from us049 eval
# --------------------------------------------------------------------------- #
#: Per-language visible strings for :func:`fig_llm_benchmark_barplot`. Metric tick
#: abbreviations (Tool-sel, Arg-match, ...) stay in English in both renders as
#: they are compact technical labels; only prose (ylabel, title) is translated.
_STR_LLM_BENCH: dict[Lang, dict[str, str]] = {
    "en": {
        "ylabel": "Score",
        "title": "Copilot LLM benchmark (US-049, AgroMind-IT/ES pending)",
    },
    "es": {
        "ylabel": "Puntaje",
        "title": "Benchmark LLMs del copiloto (US-049, AgroMind-IT/ES pendiente)",
    },
}


def fig_llm_benchmark_barplot(
    eval_json: Path = REPORTS_DIR / "agent_bench" / "us049_system_eval.json",
    *,
    out_dir: Path = FIGURES_DIR,
    lang: Lang = "en",
) -> dict[str, Path] | None:
    """Recompose the LLM benchmark barplot (Gemini vs Qwen) from real eval.

    Reads ``us049_system_eval.json`` and draws grouped bars for tool selection,
    argument match, crop grounding and routing for the models present.

    Args:
        eval_json: ``us049_system_eval.json`` source.
        out_dir: Destination directory.
        lang: Render language (``"en"`` base file, ``"es"`` gets ``_es`` suffix).

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not eval_json.exists():
        logger.warning("paper_figure_source_missing", stem="llm_benchmark")
        return None
    txt = _STR_LLM_BENCH[lang]
    data = json.loads(eval_json.read_text(encoding="utf-8"))

    def _m(block: dict, sub: str, metric: str) -> float:
        entry = block.get(sub, {}).get(metric, {})
        val = entry.get("mean") if isinstance(entry, dict) else None
        return float(val) if val is not None and not _is_nan(val) else 0.0

    label_map = {"gemini": "Gemini 2.5 Pro", "qwen": "Qwen3-30B-A3B"}
    metrics = [
        ("Tool-sel", "tool_calling", "tool_selection_accuracy"),
        ("Arg-match", "tool_calling", "arg_match_accuracy"),
        ("Crop-match", "grounded_crop", "crop_match_accuracy"),
        ("Routing", "grounded_crop", "routing_accuracy"),
    ]
    present = [k for k in ("gemini", "qwen") if k in data]
    set_paper_style()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(present), 1)
    colors = {"gemini": "#2c6fbb", "qwen": "#e08214"}
    fig, ax = plt.subplots(figsize=(6.0, 3.2), layout="constrained")
    for i, key in enumerate(present):
        vals = [_m(data[key], sub, metric) for _, sub, metric in metrics]
        ax.bar(
            x + (i - (len(present) - 1) / 2) * width,
            vals,
            width,
            label=label_map.get(key, key),
            color=colors.get(key, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.legend()
    return save_fig_svg_png(fig, "llm_benchmark_barplot", out_dir=out_dir, lang=lang)


def _is_nan(value: object) -> bool:
    """Return whether a value is a float NaN.

    Args:
        value: Any value.

    Returns:
        ``True`` if ``value`` is a float NaN.
    """
    return isinstance(value, float) and value != value


# --------------------------------------------------------------------------- #
# Promoted figures (existing PNGs re-exported as SVG+PNG)
# --------------------------------------------------------------------------- #
#: Mapping ``stem -> existing PNG`` for figures promoted as-is. Each is a real
#: artifact already generated by the pipeline. Public so notebooks can iterate it.
PROMOTED_FIGURES: dict[str, Path] = {
    "umap_alphaearth": REPORTS_DIR / "baseline" / "pheno_umap_no_coords.png",
    "curves_tsvit": REPORTS_DIR / "segmentation" / "figures" / "curves_tsvit.png",
    "confusion_tsvit": REPORTS_DIR / "segmentation" / "figures" / "confusion_tsvit.png",
    "confusion_stacking": REPORTS_DIR / "ensemble" / "figures" / "confusion_stacking.png",
    "spatial_residuals": REPORTS_DIR / "ensemble" / "figures" / "spatial_residuals_blending.png",
    "per_class_iou_tsvit": REPORTS_DIR / "segmentation" / "figures" / "per_class_iou_tsvit.png",
}


def export_conversational_examples(
    *,
    out_dir: Path = FIGURES_DIR,
    traces_dir: Path = REPORTS_DIR / "agent_bench" / "traces",
    n_examples: int = 2,
) -> Path | None:
    """Export real ES/EN conversational traces as a JSON snippet for F5.

    Reads the real agent traces (``trace_gemini_*.jsonl``) and dumps the first
    ``n_examples`` task/prompt/prediction triples to a JSON file the LaTeX
    listing consumes. The Italian variant depends on AgroMind-IT (US-068, native
    review) and is left pending -- no synthetic IT trace is fabricated (B-070-4).

    Args:
        out_dir: Destination directory.
        traces_dir: Directory with the real ``*.jsonl`` traces.
        n_examples: Number of examples to extract.

    Returns:
        Path to the written JSON, or ``None`` if no trace exists.
    """
    candidates = sorted(traces_dir.glob("trace_gemini_*.jsonl"))
    if not candidates:
        logger.warning("paper_figure_source_missing", stem="conversational_examples")
        return None
    examples: list[dict[str, str]] = []
    for trace in candidates:
        with trace.open(encoding="utf-8") as fh:
            for line in fh:
                if len(examples) >= n_examples:
                    break
                rec = json.loads(line)
                examples.append(
                    {
                        "benchmark": str(rec.get("benchmark", "")),
                        "lang": "es/en",
                        "task": str(rec.get("task", ""))[:200],
                        "prediction": str(rec.get("prediction", ""))[:400],
                    }
                )
        if len(examples) >= n_examples:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conversational_examples.json"
    path.write_text(
        json.dumps(
            {"note_it": "pendiente AgroMind-IT US-068", "examples": examples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("conversational_examples_written", path=str(path), n=len(examples))
    return path


def build_all_figures(
    out_dir: Path = FIGURES_DIR, *, langs: tuple[Lang, ...] = LANGS
) -> dict[str, dict[str, Path] | None]:
    """Generate every paper figure whose real source exists, in every language.

    Each recomposed figure (barplot, sweep, transfer, LLM bench) and each
    promoted PNG (UMAP, curves, confusion, residuals) is rendered once per
    language in ``langs``: the English render writes the canonical base file
    ``<stem>.{svg,png}``, the Spanish render writes ``<stem>_es.{svg,png}``.
    Result keys carry the ``_es`` suffix for the Spanish renders so both variants
    are addressable. Missing-source figures return ``None`` and are logged +
    documented in ``docs/blockers/epic11-notas.md``; none is fabricated.

    Args:
        out_dir: Output directory for all figures.
        langs: Languages to render (English first so the base file precedes the
            suffixed variant).

    Returns:
        Mapping ``stem -> save mapping`` (``None`` for blocked figures). The key
        is the language-suffixed stem (``<stem>`` for English, ``<stem>_es`` for
        Spanish).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path] | None] = {}
    for lang in langs:
        results[stem_for_lang("benchmark_barplot_fold5", lang)] = fig_benchmark_barplot(
            out_dir=out_dir, lang=lang
        )
        results[stem_for_lang("farslip_sweep_curve", lang)] = fig_farslip_sweep_curve(
            out_dir=out_dir, lang=lang
        )
        results[stem_for_lang("farslip_band_ablation", lang)] = fig_farslip_band_ablation(
            out_dir=out_dir, lang=lang
        )
        results[stem_for_lang("tsvit_full_config_delta", lang)] = fig_tsvit_full_config_delta(
            out_dir=out_dir, lang=lang
        )
        results[stem_for_lang("transfer_fr_catalonia", lang)] = fig_transfer_catalonia(
            out_dir=out_dir, lang=lang
        )
        results[stem_for_lang("llm_benchmark_barplot", lang)] = fig_llm_benchmark_barplot(
            out_dir=out_dir, lang=lang
        )
        for stem, source in PROMOTED_FIGURES.items():
            results[stem_for_lang(stem, lang)] = promote_png(
                source, stem, out_dir=out_dir, lang=lang
            )
    conv = export_conversational_examples(out_dir=out_dir)
    results["conversational_examples"] = {"json": conv} if conv else None
    return results


if __name__ == "__main__":  # pragma: no cover - manual entry point
    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    produced = build_all_figures()
    for name, mapping in produced.items():
        logger.info("figure", name=name, produced=mapping is not None)
