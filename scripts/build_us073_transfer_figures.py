"""Build the reproducible tables and figures of the multi-region paper section (US-073).

Reads ONLY the real EPIC 12 artefacts already versioned in the repo and emits, with
no hand-typed numbers:

- ``paper/tables/us-073-transfer/sen4agrinet_domain_gap.tex`` -- the dense
  France -> Catalonia domain-gap table (zero-shot vs few-shot mIoU / F1-macro /
  pixel-accuracy / Delta), generated from
  ``reports/segmentation/sen4agrinet_transfer_result.json`` (US-075).
- ``paper/tables/us-073-transfer/eurocropsml_kshot.tex`` -- the EuroCropsML
  transnational few-shot curve (scenario x k -> F1-macro mean +/- std over 3 seeds),
  generated from ``data/transfer/eurocropsml_fewshot_results.parquet`` (US-076).
- ``paper/figures/us-073-transfer/kshot_curve.{png,svg}`` -- the k-shot curve plot
  with per-seed error bars for the three real scenarios (LV+PT->EE, LV->EE,
  sin-pretrain->EE).
- (No longer emitted) ``mexico_phenology.{png,svg}`` -- the qualitative Mexico
  avocado/guava NDVI curves (``data/transfer/mexico_demo_ndvi.parquet``, US-077)
  left the manuscript; :func:`build_mexico_figure` is kept but ``build_all`` does
  not call it. NO classifier, NO F1.

Every figure is emitted in two languages: the English version is the canonical base
name (``<name>.{png,svg}``, for the English paper) and the Spanish version carries an
``_es`` suffix (``<name>_es.{png,svg}``, for the Spanish paper). Only the visible
strings differ between the two -- the plotted numbers and logic are identical.

The script is deterministic (fixed seed, ``matplotlib`` Agg backend) and idempotent:
re-running overwrites the artefacts byte-for-byte. Every number traces back to a real
file; if an input is missing the script raises an explicit error and never fabricates
values (Arthur's zero-synthetic rule).

Project conventions: Polars, ``structlog``, type hints, English docstrings, Spanish
visible prose in the figures, no emojis.

Usage::

    python -m scripts.build_us073_transfer_figures
    python -m scripts.build_us073_transfer_figures --repo-root . --dpi 150
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic raster output

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
import typer

from ml.transfer.eurocropsml_fewshot import summarize_curve

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# Real EPIC 12 input artefacts (relative to the repo root).
DENSE_RESULT_JSON = Path("reports/segmentation/sen4agrinet_transfer_result.json")
KSHOT_PARQUET = Path("data/transfer/eurocropsml_fewshot_results.parquet")
MEXICO_NDVI_PARQUET = Path("data/transfer/mexico_demo_ndvi.parquet")

# Output locations.
TABLES_DIR = Path("paper/tables/us-073-transfer")
FIGURES_DIR = Path("paper/figures/us-073-transfer")
DENSE_TABLE = TABLES_DIR / "sen4agrinet_domain_gap.tex"
KSHOT_TABLE = TABLES_DIR / "eurocropsml_kshot.tex"
KSHOT_FIGURE = FIGURES_DIR / "kshot_curve"
MEXICO_FIGURE = FIGURES_DIR / "mexico_phenology"

# Stable scenario order and human labels for the k-shot artefacts.
SCENARIO_ORDER: tuple[str, ...] = ("LV+PT->EE", "LV->EE", "sin-pretrain->EE")
SCENARIO_LABELS: dict[str, str] = {
    "LV+PT->EE": "LV+PT $\\rightarrow$ EE (pre-train)",
    "LV->EE": "LV $\\rightarrow$ EE (pre-train)",
    "sin-pretrain->EE": "LV $\\rightarrow$ EE (no pre-train)",
}

# Languages emitted for every figure. The canonical (base) name is English; the
# Spanish variant gets an ``_es`` suffix on the output stem.
LANGS: tuple[str, ...] = ("en", "es")

# Per-language plot-only scenario labels (matplotlib legends, no LaTeX escaping).
SCENARIO_PLOT_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "LV+PT->EE": "LV+PT -> EE (pre-train)",
        "LV->EE": "LV -> EE (pre-train)",
        "sin-pretrain->EE": "LV -> EE (no pre-train)",
    },
    "es": {
        "LV+PT->EE": "LV+PT -> EE (pre-entrenado)",
        "LV->EE": "LV -> EE (pre-entrenado)",
        "sin-pretrain->EE": "LV -> EE (sin pre-entrenar)",
    },
}

# Per-language visible strings for the k-shot curve figure.
KSHOT_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "xlabel": "k (labelled samples per class in the target country)",
        "ylabel": "F1-macro (Estonia, query set)",
        "title": "EuroCropsML few-shot curve: LV[+PT] -> EE",
        "footnote": "EuroCropsML (Reuter et al. 2025, CC-BY-SA-4.0). 3 seeds, bars = std.",
    },
    "es": {
        "xlabel": "k (muestras etiquetadas por clase del país objetivo)",
        "ylabel": "F1-macro (Estonia, conjunto de consulta)",
        "title": "Curva few-shot EuroCropsML: LV[+PT] -> EE",
        "footnote": (
            "EuroCropsML (Reuter et al. 2025, CC-BY-SA-4.0). 3 semillas, barras = std."
        ),
    },
}

# Per-language visible strings for the Mexico phenology figure.
MEXICO_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "xlabel": "Day of year (DOY)",
        "ylabel": "Zonal mean NDVI (Sentinel-2)",
        "title": "Mexico qualitative demo: perennial woody phenological signature",
        "footnote": (
            "Zero-shot methodological demo, no ground truth: F1/accuracy NOT reported. "
            "AlphaEarth (CC-BY-4.0) + Sentinel-2 (Copernicus)."
        ),
    },
    "es": {
        "xlabel": "Día del año (DOY)",
        "ylabel": "NDVI medio zonal (Sentinel-2)",
        "title": "Demo cualitativa México: firma fenológica perenne arbórea",
        "footnote": (
            "Demo metodológica zero-shot, sin ground-truth: NO se reporta F1/accuracy. "
            "AlphaEarth (CC-BY-4.0) + Sentinel-2 (Copernicus)."
        ),
    },
}

# Per-language AOI legend labels for the Mexico phenology figure.
MEXICO_AOI_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "aguacate_uruapan": "Avocado (Uruapan, Michoacán)",
        "guayaba_calvillo": "Guava (Calvillo, Aguascalientes)",
    },
    "es": {
        "aguacate_uruapan": "Aguacate (Uruapan, Michoacán)",
        "guayaba_calvillo": "Guayaba (Calvillo, Aguascalientes)",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON artefact, failing loudly if it is absent.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON object.

    Raises:
        FileNotFoundError: if the artefact is missing (never fabricated).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required real artefact missing: {path}. This script never fabricates "
            "numbers; run the EPIC 12 pipeline or `dvc pull` first."
        )
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _read_parquet(path: Path) -> pl.DataFrame:
    """Read a parquet artefact, failing loudly if it is absent.

    Args:
        path: Path to the parquet file.

    Returns:
        The loaded Polars frame.

    Raises:
        FileNotFoundError: if the artefact is missing (never fabricated).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required real artefact missing: {path}. This script never fabricates "
            "numbers; run the EPIC 12 pipeline or `dvc pull` first."
        )
    return pl.read_parquet(path)


def build_dense_table(result: dict[str, Any]) -> str:
    """Render the dense France -> Catalonia domain-gap LaTeX table from the JSON.

    The numbers are taken verbatim from the US-075 result JSON: zero-shot and
    few-shot mIoU / F1-macro / pixel-accuracy plus the Delta mIoU that is the
    scientific deliverable of the dense path.

    Args:
        result: Parsed ``sen4agrinet_transfer_result.json`` object.

    Returns:
        The LaTeX ``table`` environment as a string (``\\input``-able).
    """
    zero = result["zero_shot_metrics"]
    few = result["few_shot_metrics"]
    delta = float(result["delta_miou"])
    n_train = int(result["n_train_patches"])
    n_val = int(result["n_val_patches"])

    rows = [
        (
            "Zero-shot (FR 18 $\\rightarrow$ macro)",
            float(zero["miou"]),
            float(zero["f1_macro"]),
            float(zero["pixel_accuracy"]),
        ),
        (
            "Few-shot (10 ES patches)",
            float(few["miou"]),
            float(few["f1_macro"]),
            float(few["pixel_accuracy"]),
        ),
    ]
    body = "\n".join(
        f"{name} & {miou:.4f} & {f1:.4f} & {pa:.4f} \\\\"
        for name, miou, f1, pa in rows
    )
    caption = (
        "Dense France $\\rightarrow$ Catalonia transfer (TSViT-pheno finetuned on the "
        "Sen4AgriNet 31TCG subset, HCAT macro label-space). "
        f"Trained on {n_train} ES patches, evaluated on {n_val}. The catastrophic "
        "zero-shot mIoU and the recovered few-shot mIoU yield "
        f"$\\Delta$mIoU $= +{delta:.4f}$ -- the measured Franco-Iberian domain gap "
        "is the deliverable, not a high accuracy. "
        "Sen4AgriNet (Sykas et al. 2022, CC-BY-SA-4.0)."
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:sen4agrinet_domain_gap}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Protocol & mIoU & F1-macro & Pixel-acc. \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\midrule\n"
        f"$\\Delta$ (few-shot $-$ zero-shot) & $+{delta:.4f}$ & "
        f"$+{float(few['f1_macro']):.4f}$ & "
        f"$+{float(few['pixel_accuracy']):.4f}$ \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _kshot_summary(curve: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the raw k-shot frame into per-(scenario, k) mean/std F1-macro.

    Reuses :func:`ml.transfer.eurocropsml_fewshot.summarize_curve` (DRY) and adds the
    canonical ``scenario`` string back so the table/figure key on a single column.

    Args:
        curve: Raw long frame from the US-076 parquet
            (``source, target, k, seed, f1_macro, n_classes, use_pretrain, scenario``).

    Returns:
        A frame ``(scenario, k, f1_mean, f1_std, n_seeds)`` sorted by canonical
        scenario order then k.
    """
    summary = summarize_curve(curve.drop("scenario"))
    # Re-attach the canonical scenario label from (source, use_pretrain).
    summary = summary.with_columns(
        pl.when(pl.col("use_pretrain"))
        .then(pl.col("source") + "->" + pl.col("target"))
        .otherwise(pl.lit("sin-pretrain") + "->" + pl.col("target"))
        .alias("scenario")
    )
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    return (
        summary.with_columns(
            pl.col("scenario").replace_strict(order, default=len(order)).alias("_ord")
        )
        .sort("_ord", "k")
        .select("scenario", "k", "f1_mean", "f1_std", "n_seeds")
    )


def build_kshot_table(curve: pl.DataFrame) -> str:
    """Render the EuroCropsML k-shot curve as a LaTeX table from the parquet.

    One column per real scenario, one row per k, each cell the F1-macro
    ``mean $\\pm$ std`` over the three seeds. No number is hand-typed.

    Args:
        curve: Raw long frame from the US-076 parquet.

    Returns:
        The LaTeX ``table`` environment as a string (``\\input``-able).
    """
    summary = _kshot_summary(curve)
    ks = sorted(summary.get_column("k").unique().to_list())
    n_classes = int(curve.get_column("n_classes").max())  # type: ignore[arg-type]

    header_cols = " & ".join(SCENARIO_LABELS[s] for s in SCENARIO_ORDER)
    lines: list[str] = []
    for k in ks:
        cells: list[str] = []
        for scenario in SCENARIO_ORDER:
            row = summary.filter(
                (pl.col("scenario") == scenario) & (pl.col("k") == k)
            )
            if row.height == 0:
                cells.append("--")
                continue
            mean = float(row.get_column("f1_mean")[0])
            std = float(row.get_column("f1_std")[0])
            cells.append(f"${mean:.3f} \\pm {std:.3f}$")
        lines.append(f"{k} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)

    caption = (
        "EuroCropsML transnational few-shot transfer curve (XGBoost recipe on the "
        "parcel Sentinel-2 series, HCAT macro label-space, "
        f"{n_classes} classes). F1-macro mean $\\pm$ std over 3 seeds per k-shot. "
        "France is NOT in EuroCropsML; the protocol is LV[+PT] $\\rightarrow$ EE "
        "(Reuss et al. 2025, Table II), not France $\\rightarrow$ Estonia. The "
        "no-pre-train column quantifies how source pre-training closes the gap at "
        "low k. "
        "EuroCropsML (Reuss et al. 2025, CC-BY-SA-4.0)."
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:eurocropsml_kshot}\n"
        "\\begin{tabular}{rccc}\n"
        "\\toprule\n"
        f"$k$ & {header_cols} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_kshot_figure(
    curve: pl.DataFrame, out_stem: Path, *, dpi: int, lang: str = "en"
) -> None:
    """Plot the k-shot F1-macro curve with per-seed error bars (PNG + SVG).

    All visible text is resolved from :data:`KSHOT_STRINGS` and
    :data:`SCENARIO_PLOT_LABELS` for the requested language; only strings change,
    never the plotted numbers or plotting logic.

    Args:
        curve: Raw long frame from the US-076 parquet.
        out_stem: Output path stem (``.png`` and ``.svg`` are appended).
        dpi: Raster resolution for the PNG.
        lang: Language code for the visible text (``"en"`` or ``"es"``).
    """
    txt = KSHOT_STRINGS[lang]
    scenario_labels = SCENARIO_PLOT_LABELS[lang]
    summary = _kshot_summary(curve)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = {"LV+PT->EE": "o", "LV->EE": "s", "sin-pretrain->EE": "^"}
    for scenario in SCENARIO_ORDER:
        sub = summary.filter(pl.col("scenario") == scenario).sort("k")
        if sub.height == 0:
            continue
        ks = sub.get_column("k").to_numpy()
        mean = sub.get_column("f1_mean").to_numpy()
        std = sub.get_column("f1_std").to_numpy()
        ax.errorbar(
            ks,
            mean,
            yerr=std,
            marker=markers.get(scenario, "o"),
            capsize=3,
            linewidth=1.6,
            markersize=5,
            label=scenario_labels[scenario],
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary.get_column("k").unique().to_list()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(txt["xlabel"])
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.01, txt["footnote"], fontsize=7, color="0.4")
    fig.tight_layout()
    _save_fig(fig, out_stem, dpi=dpi)


def build_mexico_figure(
    ndvi: pl.DataFrame, out_stem: Path, *, dpi: int, lang: str = "en"
) -> None:
    """Plot the qualitative Mexico avocado/guava NDVI phenology (PNG + SVG).

    Purely qualitative: shows the real GEE-derived per-AOI NDVI time series of the
    two perennial-woody Mexican crops. NO classifier, NO F1, NO accuracy is computed
    or implied (US-077 honesty rule). All visible text is resolved from
    :data:`MEXICO_STRINGS` and :data:`MEXICO_AOI_LABELS` for the requested language;
    only strings change, never the plotted numbers or plotting logic.

    Args:
        ndvi: Frame from ``mexico_demo_ndvi.parquet`` with ``aoi, date, doy, ndvi``.
        out_stem: Output path stem (``.png`` and ``.svg`` are appended).
        dpi: Raster resolution for the PNG.
        lang: Language code for the visible text (``"en"`` or ``"es"``).
    """
    txt = MEXICO_STRINGS[lang]
    labels = MEXICO_AOI_LABELS[lang]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for aoi in sorted(ndvi.get_column("aoi").unique().to_list()):
        sub = ndvi.filter(pl.col("aoi") == aoi).sort("doy")
        ax.plot(
            sub.get_column("doy").to_numpy(),
            sub.get_column("ndvi").to_numpy(),
            marker=".",
            linewidth=1.2,
            markersize=4,
            alpha=0.85,
            label=labels.get(aoi, aoi),
        )
    ax.set_xlabel(txt["xlabel"])
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.01, txt["footnote"], fontsize=7, color="0.4")
    fig.tight_layout()
    _save_fig(fig, out_stem, dpi=dpi)


def _save_fig(fig: plt.Figure, out_stem: Path, *, dpi: int) -> None:
    """Save a figure as both PNG and SVG and close it.

    Args:
        fig: The matplotlib figure.
        out_stem: Output path stem (extension-less).
        dpi: Raster resolution for the PNG.
    """
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _lang_stem(out_stem: Path, lang: str) -> Path:
    """Return the language-specific output stem for a figure.

    English is canonical and keeps the bare base name; every other language gets a
    ``_<lang>`` suffix (e.g. ``kshot_curve`` -> ``kshot_curve_es`` for Spanish).

    Args:
        out_stem: Base (English) output path stem, extension-less.
        lang: Language code (``"en"`` for the canonical base name).

    Returns:
        The stem to hand to :func:`_save_fig` for this language.
    """
    if lang == "en":
        return out_stem
    return out_stem.with_name(f"{out_stem.name}_{lang}")


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text deterministically (LF newlines), creating parents.

    Args:
        path: Destination path.
        content: Text content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    logger.info("artefact_written", path=str(path), bytes=len(content.encode("utf-8")))


def build_all(repo_root: Path = Path("."), *, dpi: int = 150) -> None:
    """Build every US-073 table and figure from the real EPIC 12 artefacts.

    This is the importable entry point (tests call it directly); :func:`run` is the
    thin Typer CLI wrapper around it.

    Args:
        repo_root: Repository root (inputs and outputs are resolved against it).
        dpi: Raster resolution for the generated PNG figures.
    """
    np.random.seed(0)  # determinism (no randomness in the build, but pin it anyway)
    root = repo_root.resolve()

    dense = _read_json(root / DENSE_RESULT_JSON)
    curve = _read_parquet(root / KSHOT_PARQUET)

    _write_text(root / DENSE_TABLE, build_dense_table(dense))
    _write_text(root / KSHOT_TABLE, build_kshot_table(curve))

    # Emit every figure in each supported language: English is the canonical base
    # name (<name>.{png,svg}); other languages get an ``_<lang>`` suffix.
    # NOTE: the Mexico avocado/guava figure is intentionally NOT built anymore --
    # the paper standardizes on real-metric experiments only, so the metric-less
    # qualitative Mexico demo was removed from the manuscript.
    figures: list[str] = []
    for lang in LANGS:
        kshot_stem = _lang_stem(root / KSHOT_FIGURE, lang)
        build_kshot_figure(curve, kshot_stem, dpi=dpi, lang=lang)
        figures.extend(
            [
                f"{kshot_stem}.png",
                f"{kshot_stem}.svg",
            ]
        )

    logger.info(
        "us073_build_done",
        tables=[str(DENSE_TABLE), str(KSHOT_TABLE)],
        figures=figures,
    )


@app.command()
def run(
    repo_root: Annotated[Path, typer.Option("--repo-root")] = Path("."),
    dpi: Annotated[int, typer.Option("--dpi")] = 150,
) -> None:
    """CLI entry point: build all US-073 tables and figures (see :func:`build_all`).

    Args:
        repo_root: Repository root (inputs and outputs are resolved against it).
        dpi: Raster resolution for the generated PNG figures.
    """
    build_all(repo_root, dpi=dpi)


if __name__ == "__main__":
    app()
